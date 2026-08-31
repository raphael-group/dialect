"""Close the result-blind evidence boundary for a revision artifact registry.

This module is deliberately not a scientific-data transform.  It first validates an
independently SHA-256-anchored artifact registry with the registry's own validator.
Only after that succeeds does it open two explicit evidence roots:

* a gate-receipt root whose release-relative members are the registry ``receipt_id``
  values; and
* a source-data root whose release-relative members are the sources declared by
  every ``ready`` artifact.

Gate receipts and source-data members are handled as opaque byte streams.  Their
contents are never parsed or interpreted.  The tool verifies exact inventories,
single-link regular-file identities, byte counts, SHA-256 digests, and exact
artifact-to-gate/source-role bindings.  It then publishes a canonical JSON closure
receipt atomically and without replacing an existing destination.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

from analysis import build_tcga_revision_artifact_registry as artifact_registry

# Precise validation errors are part of this boundary's operational contract.
# ruff: noqa: EM101, EM102, TRY003, TRY301

CLOSURE_SCHEMA: Final = "dialect-revision-release-evidence-closure-v1"
CLOSURE_CONTRACT: Final = (
    "independent-registry-opaque-gate-source-byte-closure-v1"
)
BUILDER_MEMBER: Final = "analysis/build_tcga_revision_release_evidence.py"
TRUST_MODEL: Final = {
    "artifact_registry": (
        "independently SHA-256 anchored and validated with its native validator"
    ),
    "gate_receipts": (
        "opaque bytes; exact inventory, stable size, and declared SHA-256 only"
    ),
    "source_data": (
        "opaque bytes; exact ready-artifact inventory, declared size, and SHA-256 only"
    ),
    "scientific_scope": (
        "no gate payload, source row, scientific result, or claim is interpreted"
    ),
}

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_READ_CHUNK_BYTES: Final = 1024 * 1024
_MAX_METADATA_BYTES: Final = 16 * 1024 * 1024


class ReleaseEvidenceError(ValueError):
    """Raised when the release-evidence closure cannot be proven."""


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceReceipt:
    """Digest-only receipt for a validated evidence closure."""

    manifest_path: str
    manifest_sha256: str
    gate_receipt_count: int
    source_member_count: int
    ready_count: int
    omitted_count: int


@dataclass(slots=True)
class _PinnedRoot:
    path: Path
    descriptor: int = field(repr=False)
    device: int
    inode: int

    def close(self) -> None:
        """Close the pinned root descriptor."""
        os.close(self.descriptor)


@dataclass(slots=True)
class _PinnedFile:
    path: Path | None
    member: str | None
    descriptor: int = field(repr=False)
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    sha256: str

    def close(self) -> None:
        """Close the pinned file descriptor."""
        os.close(self.descriptor)


@dataclass(frozen=True, slots=True)
class _PublishedDestinationExpectation:
    staged_identity: tuple[int, int]
    sha256: str
    size_bytes: int
    link_count: int


@dataclass(slots=True)
class _PreparedClosure:
    registry_path: Path
    renderer_root: Path
    rendered_output_root: Path
    expected_registry_sha256: str
    registry_file: _PinnedFile
    builder_file: _PinnedFile
    gate_root: _PinnedRoot
    source_root: _PinnedRoot
    gate_files: tuple[_PinnedFile, ...]
    source_files: tuple[_PinnedFile, ...]
    gate_inventory: tuple[str, ...]
    source_inventory: tuple[str, ...]
    payload: dict[str, object]
    ready_count: int
    omitted_count: int

    def close(self) -> None:
        """Close all descriptors retained across the validation boundary."""
        for pinned in (*self.gate_files, *self.source_files):
            pinned.close()
        self.source_root.close()
        self.gate_root.close()
        self.builder_file.close()
        self.registry_file.close()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ReleaseEvidenceError(
            f"{context} must be a lowercase SHA-256 digest",
        )
    return value


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseEvidenceError(f"{context} must be a JSON object")
    return value


def _require_sequence(value: object, *, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ReleaseEvidenceError(f"{context} must be a JSON array")
    return value


def _require_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseEvidenceError(f"{context} must be a nonempty string")
    return value


def _require_size(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReleaseEvidenceError(f"{context} must be a nonnegative integer")
    return value


def _canonical_member(value: object, *, context: str) -> str:
    member = _require_string(value, context=context)
    if (
        not member.isascii()
        or "\\" in member
        or any(ord(character) < 32 or ord(character) == 127 for character in member)
    ):
        raise ReleaseEvidenceError(f"{context} is not a canonical POSIX member")
    path = PurePosixPath(member)
    if path.is_absolute() or path.as_posix() != member:
        raise ReleaseEvidenceError(f"{context} is not a canonical relative member")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseEvidenceError(f"{context} escapes its declared evidence root")
    return member


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, item in pairs:
        if key in parsed:
            raise ReleaseEvidenceError(f"metadata contains duplicate key {key!r}")
        parsed[key] = item
    return parsed


def _reject_nonfinite(constant: str) -> object:
    raise ReleaseEvidenceError(f"metadata contains non-finite constant {constant!r}")


def _parse_canonical(raw: bytes, *, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReleaseEvidenceError(f"{context} is not valid UTF-8 JSON") from exc
    parsed = _require_mapping(value, context=context)
    if raw != _canonical_json(parsed) + b"\n":
        raise ReleaseEvidenceError(f"{context} is not canonical JSON with one newline")
    return parsed


def _digest_descriptor(descriptor: int) -> tuple[str, int, os.stat_result]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or size != after.st_size
    ):
        raise ReleaseEvidenceError("a pinned evidence file changed while hashed")
    return digest.hexdigest(), size, after


def _pin_absolute_file(path: Path, *, context: str) -> _PinnedFile:
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot inspect {context}: {absolute}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ReleaseEvidenceError(f"{context} must be a non-symlink regular file")
    if entry.st_nlink != 1:
        raise ReleaseEvidenceError(f"{context} must be a single-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot pin {context}: {absolute}") from exc
    try:
        digest, size, identity = _digest_descriptor(descriptor)
        if (
            (identity.st_dev, identity.st_ino) != (entry.st_dev, entry.st_ino)
            or identity.st_nlink != 1
            or identity.st_mtime_ns != entry.st_mtime_ns
        ):
            raise ReleaseEvidenceError(f"{context} changed while it was pinned")
        if size > _MAX_METADATA_BYTES and context in {
            "artifact registry",
            "release-evidence closure",
        }:
            raise ReleaseEvidenceError(f"{context} exceeds the metadata size limit")
        return _PinnedFile(
            path=absolute,
            member=None,
            descriptor=descriptor,
            device=identity.st_dev,
            inode=identity.st_ino,
            size_bytes=size,
            modified_ns=identity.st_mtime_ns,
            sha256=digest,
        )
    except Exception:
        os.close(descriptor)
        raise


def _pin_root(path: Path, *, context: str) -> _PinnedRoot:
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot inspect {context}: {absolute}") from exc
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise ReleaseEvidenceError(f"{context} must be a non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot pin {context}: {absolute}") from exc
    identity = os.fstat(descriptor)
    if (identity.st_dev, identity.st_ino) != (entry.st_dev, entry.st_ino):
        os.close(descriptor)
        raise ReleaseEvidenceError(f"{context} changed while it was pinned")
    return _PinnedRoot(
        path=absolute,
        descriptor=descriptor,
        device=identity.st_dev,
        inode=identity.st_ino,
    )


def _open_member(root: _PinnedRoot, member: str) -> int:
    parts = PurePosixPath(member).parts
    directory_fd = os.dup(root.descriptor)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(parts[-1], file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _pin_member(
    root: _PinnedRoot,
    member: str,
    *,
    context: str,
    expected_sha256: str,
    expected_size: int | None,
) -> _PinnedFile:
    try:
        descriptor = _open_member(root, member)
    except OSError as exc:
        raise ReleaseEvidenceError(f"cannot open {context}: {member}") from exc
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise ReleaseEvidenceError(
                f"{context} must be a single-link regular file: {member}",
            )
        digest, size, after = _digest_descriptor(descriptor)
        if (
            (after.st_dev, after.st_ino) != (identity.st_dev, identity.st_ino)
            or after.st_nlink != 1
            or after.st_mtime_ns != identity.st_mtime_ns
        ):
            raise ReleaseEvidenceError(f"{context} changed while pinned: {member}")
        if digest != expected_sha256:
            raise ReleaseEvidenceError(f"{context} SHA-256 mismatch: {member}")
        if expected_size is not None and size != expected_size:
            raise ReleaseEvidenceError(f"{context} byte-count mismatch: {member}")
        return _PinnedFile(
            path=None,
            member=member,
            descriptor=descriptor,
            device=after.st_dev,
            inode=after.st_ino,
            size_bytes=size,
            modified_ns=after.st_mtime_ns,
            sha256=digest,
        )
    except Exception:
        os.close(descriptor)
        raise


def _enumerate_tree(
    root: _PinnedRoot,
    *,
    context: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    files: list[str] = []
    directories: list[str] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def visit(directory_fd: int, prefix: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise ReleaseEvidenceError(f"cannot enumerate {context}") from exc
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ReleaseEvidenceError(f"{context} contains an unsafe entry")
            member = f"{prefix}/{name}" if prefix else name
            try:
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ReleaseEvidenceError(
                    f"cannot inspect {context} member: {member}",
                ) from exc
            if stat.S_ISLNK(entry.st_mode):
                raise ReleaseEvidenceError(f"{context} contains a symlink: {member}")
            if stat.S_ISDIR(entry.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ReleaseEvidenceError(
                        f"cannot pin {context} directory: {member}",
                    ) from exc
                try:
                    child = os.fstat(child_fd)
                    if (child.st_dev, child.st_ino) != (entry.st_dev, entry.st_ino):
                        raise ReleaseEvidenceError(
                            f"{context} directory changed: {member}",
                        )
                    directories.append(member)
                    visit(child_fd, member)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry.st_mode):
                files.append(member)
            else:
                raise ReleaseEvidenceError(
                    f"{context} contains a non-regular entry: {member}",
                )

    visit(root.descriptor, "")
    return tuple(files), tuple(directories)


def _expected_directories(members: Sequence[str]) -> tuple[str, ...]:
    directories: set[str] = set()
    for member in members:
        parts = PurePosixPath(member).parts
        for end in range(1, len(parts)):
            directories.add(PurePosixPath(*parts[:end]).as_posix())
    return tuple(sorted(directories))


def _require_exact_inventory(
    root: _PinnedRoot,
    expected_members: Sequence[str],
    *,
    context: str,
) -> None:
    files, directories = _enumerate_tree(root, context=context)
    expected_files = tuple(sorted(expected_members))
    expected_directories = _expected_directories(expected_files)
    if files != expected_files or directories != expected_directories:
        missing = sorted(set(expected_files) - set(files))
        extra = sorted(set(files) - set(expected_files))
        missing_directories = sorted(set(expected_directories) - set(directories))
        extra_directories = sorted(set(directories) - set(expected_directories))
        raise ReleaseEvidenceError(
            f"{context} inventory mismatch; missing={missing}, extra={extra}, "
            f"missing_directories={missing_directories}, "
            f"extra_directories={extra_directories}",
        )


def _revalidate_root(root: _PinnedRoot, *, context: str) -> None:
    current = os.fstat(root.descriptor)
    try:
        entry = os.lstat(root.path)
    except OSError as exc:
        raise ReleaseEvidenceError(f"{context} disappeared") from exc
    if (
        not stat.S_ISDIR(entry.st_mode)
        or (current.st_dev, current.st_ino) != (root.device, root.inode)
        or (entry.st_dev, entry.st_ino) != (root.device, root.inode)
    ):
        raise ReleaseEvidenceError(f"{context} changed during validation")


def _revalidate_file(
    pinned: _PinnedFile,
    *,
    context: str,
    root: _PinnedRoot | None = None,
) -> None:
    digest, size, identity = _digest_descriptor(pinned.descriptor)
    if (
        (identity.st_dev, identity.st_ino) != (pinned.device, pinned.inode)
        or identity.st_nlink != 1
        or identity.st_mtime_ns != pinned.modified_ns
        or size != pinned.size_bytes
        or digest != pinned.sha256
    ):
        raise ReleaseEvidenceError(f"{context} changed during validation")
    if root is None:
        if pinned.path is None:
            raise ReleaseEvidenceError(f"{context} lacks its absolute path binding")
        replacement = _pin_absolute_file(pinned.path, context=context)
    else:
        if pinned.member is None:
            raise ReleaseEvidenceError(f"{context} lacks its member binding")
        replacement = _pin_member(
            root,
            pinned.member,
            context=context,
            expected_sha256=pinned.sha256,
            expected_size=pinned.size_bytes,
        )
    try:
        if (
            (replacement.device, replacement.inode) != (pinned.device, pinned.inode)
            or replacement.modified_ns != pinned.modified_ns
        ):
            raise ReleaseEvidenceError(f"{context} entry changed during validation")
    finally:
        replacement.close()


def _builder_record(pinned: _PinnedFile) -> dict[str, object]:
    return {
        "script": BUILDER_MEMBER,
        "sha256": pinned.sha256,
        "bytes": pinned.size_bytes,
    }


def _registry_bindings(
    registry: Mapping[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[str, ...],
    tuple[str, ...],
]:
    ledger = _require_sequence(registry.get("gate_ledger"), context="gate_ledger")
    gate_records: list[dict[str, object]] = []
    gate_by_name: dict[str, dict[str, object]] = {}
    gate_members: set[str] = set()
    for index, raw in enumerate(ledger):
        record = _require_mapping(raw, context=f"gate_ledger[{index}]")
        gate = _require_string(record.get("gate"), context=f"gate_ledger[{index}].gate")
        receipt_id = _canonical_member(
            record.get("receipt_id"),
            context=f"gate_ledger[{index}].receipt_id",
        )
        digest = _require_sha256(
            record.get("sha256"),
            context=f"gate_ledger[{index}].sha256",
        )
        if gate in gate_by_name or receipt_id in gate_members:
            raise ReleaseEvidenceError(
                "gate ledger must bind each gate to one unique receipt member",
            )
        normalized = {
            "gate": gate,
            "receipt_id": receipt_id,
            "member": receipt_id,
            "sha256": digest,
            "artifacts": [],
        }
        gate_records.append(normalized)
        gate_by_name[gate] = normalized
        gate_members.add(receipt_id)

    artifacts = _require_sequence(registry.get("artifacts"), context="artifacts")
    artifact_records: list[dict[str, object]] = []
    sources_by_id: dict[str, dict[str, object]] = {}
    source_ids_by_member: dict[str, str] = {}
    for index, raw in enumerate(artifacts):
        artifact = _require_mapping(raw, context=f"artifacts[{index}]")
        semantic_id = _require_string(
            artifact.get("semantic_id"),
            context=f"artifacts[{index}].semantic_id",
        )
        status = _require_string(
            artifact.get("status"),
            context=f"artifacts[{index}].status",
        )
        if status not in {"ready", "omitted"}:
            raise ReleaseEvidenceError(
                f"artifacts[{semantic_id}].status is not ready or omitted",
            )

        required_gates = [
            _require_string(value, context=f"artifacts[{semantic_id}].required_gates")
            for value in _require_sequence(
                artifact.get("required_gates"),
                context=f"artifacts[{semantic_id}].required_gates",
            )
        ]
        gate_receipts = _require_sequence(
            artifact.get("gate_receipts"),
            context=f"artifacts[{semantic_id}].gate_receipts",
        )
        bound_gates: list[str] = []
        receipt_ids: list[str] = []
        for receipt_raw in gate_receipts:
            receipt = _require_mapping(
                receipt_raw,
                context=f"artifacts[{semantic_id}].gate_receipts",
            )
            gate = _require_string(receipt.get("gate"), context="gate receipt gate")
            receipt_id = _require_string(
                receipt.get("receipt_id"),
                context="gate receipt id",
            )
            global_record = gate_by_name.get(gate)
            if (
                global_record is None
                or global_record["receipt_id"] != receipt_id
                or global_record["sha256"] != receipt.get("sha256")
            ):
                raise ReleaseEvidenceError(
                    f"artifact {semantic_id!r} contradicts the gate ledger",
                )
            bound_gates.append(gate)
            receipt_ids.append(receipt_id)
            cast_artifacts = global_record["artifacts"]
            if not isinstance(cast_artifacts, list):
                raise ReleaseEvidenceError("internal gate binding is invalid")
            cast_artifacts.append(semantic_id)
        expected_bound_gates = [
            gate for gate in required_gates if gate in gate_by_name
        ]
        if bound_gates != expected_bound_gates:
            raise ReleaseEvidenceError(
                f"artifact {semantic_id!r} does not exactly bind its available gates",
            )
        if status == "ready" and bound_gates != required_gates:
            raise ReleaseEvidenceError(
                f"artifact {semantic_id!r} does not exactly bind its required gates",
            )

        required_roles = [
            _require_string(value, context=f"artifacts[{semantic_id}].required_roles")
            for value in _require_sequence(
                artifact.get("required_source_roles"),
                context=f"artifacts[{semantic_id}].required_source_roles",
            )
        ]
        if status == "omitted":
            omission = _require_mapping(
                artifact.get("omission"),
                context=f"artifacts[{semantic_id}].omission",
            )
            artifact_records.append(
                {
                    "semantic_id": semantic_id,
                    "status": "omitted",
                    "required_gates": required_gates,
                    "satisfied_gates": bound_gates,
                    "receipt_ids": receipt_ids,
                    "required_source_roles": required_roles,
                    "omission": {
                        "reason_code": _require_string(
                            omission.get("reason_code"),
                            context=f"artifacts[{semantic_id}].reason_code",
                        ),
                        "unsatisfied_gates": [
                            _require_string(
                                gate,
                                context=(
                                    f"artifacts[{semantic_id}].unsatisfied_gates"
                                ),
                            )
                            for gate in _require_sequence(
                                omission.get("unsatisfied_gates"),
                                context=f"artifacts[{semantic_id}].unsatisfied_gates",
                            )
                        ],
                    },
                },
            )
            continue
        sources = _require_sequence(
            artifact.get("source_data"),
            context=f"artifacts[{semantic_id}].source_data",
        )
        source_ids: list[str] = []
        actual_roles: set[str] = set()
        for source_raw in sources:
            source = _require_mapping(
                source_raw,
                context=f"artifacts[{semantic_id}].source_data",
            )
            source_id = _require_string(source.get("source_id"), context="source_id")
            member = _canonical_member(
                source.get("release_member"),
                context=f"source {source_id!r} member",
            )
            role = _require_string(
                source.get("role"),
                context=f"source {source_id!r} role",
            )
            digest = _require_sha256(
                source.get("sha256"),
                context=f"source {source_id!r} sha256",
            )
            size = _require_size(
                source.get("bytes"),
                context=f"source {source_id!r} bytes",
            )
            identity = (member, digest, size)
            existing = sources_by_id.get(source_id)
            if existing is None:
                member_owner = source_ids_by_member.setdefault(member, source_id)
                if member_owner != source_id:
                    raise ReleaseEvidenceError(
                        f"source member {member!r} has multiple source identities",
                    )
                existing = {
                    "source_id": source_id,
                    "member": member,
                    "sha256": digest,
                    "bytes": size,
                    "bindings": [],
                }
                sources_by_id[source_id] = existing
            elif (
                existing["member"],
                existing["sha256"],
                existing["bytes"],
            ) != identity:
                raise ReleaseEvidenceError(
                    f"source {source_id!r} has inconsistent byte identity",
                )
            bindings = existing["bindings"]
            if not isinstance(bindings, list):
                raise ReleaseEvidenceError("internal source binding is invalid")
            binding = {"semantic_id": semantic_id, "role": role}
            if binding in bindings:
                raise ReleaseEvidenceError(
                    f"artifact {semantic_id!r} duplicates source {source_id!r}",
                )
            bindings.append(binding)
            source_ids.append(source_id)
            actual_roles.add(role)
        if actual_roles != set(required_roles):
            raise ReleaseEvidenceError(
                f"artifact {semantic_id!r} source roles are not exact; "
                f"required={sorted(required_roles)}, actual={sorted(actual_roles)}",
            )
        artifact_records.append(
            {
                "semantic_id": semantic_id,
                "status": "ready",
                "required_gates": required_gates,
                "satisfied_gates": bound_gates,
                "receipt_ids": receipt_ids,
                "required_source_roles": required_roles,
                "source_ids": source_ids,
            },
        )

    for record in gate_records:
        artifacts_value = record["artifacts"]
        if isinstance(artifacts_value, list):
            artifacts_value.sort()
    source_records = sorted(
        sources_by_id.values(),
        key=lambda item: str(item["source_id"]),
    )
    for record in source_records:
        bindings = record["bindings"]
        if isinstance(bindings, list):
            bindings.sort(key=lambda item: (item["semantic_id"], item["role"]))
    return (
        gate_records,
        source_records,
        artifact_records,
        tuple(sorted(gate_members)),
        tuple(sorted(source_ids_by_member)),
    )


def _pin_evidence(
    root: _PinnedRoot,
    records: Sequence[Mapping[str, object]],
    *,
    context: str,
    has_declared_size: bool,
) -> tuple[_PinnedFile, ...]:
    pins: list[_PinnedFile] = []
    try:
        for record in records:
            member = _require_string(record["member"], context=f"{context} member")
            pins.append(
                _pin_member(
                    root,
                    member,
                    context=context,
                    expected_sha256=_require_sha256(
                        record["sha256"],
                        context=f"{context} sha256",
                    ),
                    expected_size=(
                        _require_size(record["bytes"], context=f"{context} bytes")
                        if has_declared_size
                        else None
                    ),
                ),
            )
    except Exception:
        for pinned in pins:
            pinned.close()
        raise
    return tuple(pins)


def _validated_registry(
    registry_path: Path,
    renderer_root: Path,
    rendered_output_root: Path,
    *,
    expected_registry_sha256: str,
) -> artifact_registry.ArtifactRegistryReceipt:
    try:
        return artifact_registry.validate_artifact_registry(
            registry_path,
            renderer_root,
            rendered_output_root,
            expected_manifest_sha256=expected_registry_sha256,
        )
    except (artifact_registry.ArtifactRegistryError, OSError) as exc:
        raise ReleaseEvidenceError("artifact registry validation failed") from exc


def _prepare_closure(  # noqa: PLR0913
    artifact_registry_path: Path,
    renderer_root: Path,
    rendered_output_root: Path,
    gate_receipt_root: Path,
    source_data_root: Path,
    *,
    expected_artifact_registry_sha256: str,
) -> _PreparedClosure:
    expected = _require_sha256(
        expected_artifact_registry_sha256,
        context="expected_artifact_registry_sha256",
    )
    validation = _validated_registry(
        artifact_registry_path,
        renderer_root,
        rendered_output_root,
        expected_registry_sha256=expected,
    )
    registry_file = _pin_absolute_file(
        artifact_registry_path,
        context="artifact registry",
    )
    builder_file: _PinnedFile | None = None
    gate_root: _PinnedRoot | None = None
    source_root: _PinnedRoot | None = None
    gate_files: tuple[_PinnedFile, ...] = ()
    source_files: tuple[_PinnedFile, ...] = ()
    try:
        if registry_file.sha256 != expected:
            raise ReleaseEvidenceError(
                "artifact registry changed after native validation",
            )
        registry_raw = _descriptor_bytes(registry_file)
        registry = _parse_canonical(registry_raw, context="artifact registry")
        builder_file = _pin_absolute_file(Path(__file__), context="closure builder")
        (
            gate_records,
            source_records,
            artifact_records,
            gate_inventory,
            source_inventory,
        ) = _registry_bindings(registry)
        ready_count = sum(record["status"] == "ready" for record in artifact_records)
        omitted_count = len(artifact_records) - ready_count
        if (
            ready_count != validation.ready_count
            or omitted_count != validation.omitted_count
        ):
            raise ReleaseEvidenceError(
                "artifact counts changed after native registry validation",
            )

        gate_root = _pin_root(gate_receipt_root, context="gate-receipt root")
        source_root = _pin_root(source_data_root, context="source-data root")
        _require_exact_inventory(
            gate_root,
            gate_inventory,
            context="gate-receipt root",
        )
        _require_exact_inventory(
            source_root,
            source_inventory,
            context="source-data root",
        )
        gate_files = _pin_evidence(
            gate_root,
            gate_records,
            context="gate receipt",
            has_declared_size=False,
        )
        source_files = _pin_evidence(
            source_root,
            source_records,
            context="source-data member",
            has_declared_size=True,
        )
        gate_sizes = {pin.member: pin.size_bytes for pin in gate_files}
        closed_gate_records = [
            {**record, "bytes": gate_sizes[str(record["member"])]}
            for record in gate_records
        ]
        release = _require_mapping(registry.get("release"), context="registry.release")
        registry_payload_sha256 = _require_sha256(
            registry.get("registry_payload_sha256"),
            context="registry.registry_payload_sha256",
        )
        payload = {
            "schema": CLOSURE_SCHEMA,
            "contract": CLOSURE_CONTRACT,
            "trust_model": dict(TRUST_MODEL),
            "release": dict(release),
            "builder": _builder_record(builder_file),
            "artifact_registry": {
                "sha256": registry_file.sha256,
                "bytes": registry_file.size_bytes,
                "registry_payload_sha256": registry_payload_sha256,
                "ready_count": ready_count,
                "omitted_count": omitted_count,
            },
            "gate_receipts": closed_gate_records,
            "source_data": source_records,
            "artifacts": artifact_records,
            "inventory": {
                "gate_receipt_count": len(closed_gate_records),
                "gate_receipt_members_sha256": _sha256(
                    _canonical_json(closed_gate_records),
                ),
                "source_member_count": len(source_records),
                "source_members_sha256": _sha256(_canonical_json(source_records)),
            },
        }
        closure = {
            **payload,
            "closure_payload_sha256": _sha256(_canonical_json(payload)),
        }
        return _PreparedClosure(
            registry_path=artifact_registry_path.absolute(),
            renderer_root=renderer_root.absolute(),
            rendered_output_root=rendered_output_root.absolute(),
            expected_registry_sha256=expected,
            registry_file=registry_file,
            builder_file=builder_file,
            gate_root=gate_root,
            source_root=source_root,
            gate_files=gate_files,
            source_files=source_files,
            gate_inventory=gate_inventory,
            source_inventory=source_inventory,
            payload=closure,
            ready_count=ready_count,
            omitted_count=omitted_count,
        )
    except Exception:
        for pinned in (*gate_files, *source_files):
            pinned.close()
        if source_root is not None:
            source_root.close()
        if gate_root is not None:
            gate_root.close()
        if builder_file is not None:
            builder_file.close()
        registry_file.close()
        raise


def _descriptor_bytes(pinned: _PinnedFile) -> bytes:
    os.lseek(pinned.descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(pinned.descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    if size != pinned.size_bytes:
        raise ReleaseEvidenceError("pinned metadata size changed while read")
    raw = b"".join(chunks)
    if _sha256(raw) != pinned.sha256:
        raise ReleaseEvidenceError("pinned metadata digest changed while read")
    return raw


def _revalidate_prepared(prepared: _PreparedClosure) -> None:
    _validated_registry(
        prepared.registry_path,
        prepared.renderer_root,
        prepared.rendered_output_root,
        expected_registry_sha256=prepared.expected_registry_sha256,
    )
    _revalidate_file(prepared.registry_file, context="artifact registry")
    _revalidate_file(prepared.builder_file, context="closure builder")
    _revalidate_root(prepared.gate_root, context="gate-receipt root")
    _revalidate_root(prepared.source_root, context="source-data root")
    _require_exact_inventory(
        prepared.gate_root,
        prepared.gate_inventory,
        context="gate-receipt root",
    )
    _require_exact_inventory(
        prepared.source_root,
        prepared.source_inventory,
        context="source-data root",
    )
    for pinned in prepared.gate_files:
        _revalidate_file(
            pinned,
            context=f"gate receipt {pinned.member}",
            root=prepared.gate_root,
        )
    for pinned in prepared.source_files:
        _revalidate_file(
            pinned,
            context=f"source-data member {pinned.member}",
            root=prepared.source_root,
        )


def _ensure_destination(
    destination: Path,
    *,
    registry_path: Path,
) -> tuple[Path, int]:
    absolute = destination.absolute()
    if absolute.suffix != ".json":
        raise ReleaseEvidenceError("closure destination must end in .json")
    if absolute == registry_path.absolute():
        raise ReleaseEvidenceError("closure destination cannot replace the registry")
    try:
        parent_entry = os.lstat(absolute.parent)
    except OSError as exc:
        raise ReleaseEvidenceError("closure destination parent does not exist") from exc
    if stat.S_ISLNK(parent_entry.st_mode) or not stat.S_ISDIR(parent_entry.st_mode):
        raise ReleaseEvidenceError(
            "closure destination parent must be a non-symlink directory",
        )
    resolved_parent = absolute.parent.resolve(strict=True)
    if resolved_parent != absolute.parent:
        raise ReleaseEvidenceError("closure destination parent is not canonical")
    resolved_destination = resolved_parent / absolute.name
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = os.open(resolved_parent, flags)
    current = os.fstat(parent_fd)
    if (current.st_dev, current.st_ino) != (parent_entry.st_dev, parent_entry.st_ino):
        os.close(parent_fd)
        raise ReleaseEvidenceError("closure destination parent changed")
    try:
        os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return resolved_destination, parent_fd
    except OSError as exc:
        os.close(parent_fd)
        raise ReleaseEvidenceError("cannot inspect closure destination") from exc
    os.close(parent_fd)
    raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), absolute)


def _require_destination_outside_inputs(
    destination: Path,
    roots: Sequence[Path],
) -> None:
    for root in roots:
        resolved_root = root.resolve(strict=True)
        try:
            destination.relative_to(resolved_root)
        except ValueError:
            continue
        raise ReleaseEvidenceError("closure destination must be outside input roots")


def _revalidate_destination_parent(destination: Path, parent_fd: int) -> None:
    try:
        path_entry = os.lstat(destination.parent)
        resolved_parent = destination.parent.resolve(strict=True)
    except OSError as exc:
        raise ReleaseEvidenceError(
            "closure destination parent disappeared during publication",
        ) from exc
    pinned_parent = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(path_entry.st_mode)
        or stat.S_ISLNK(path_entry.st_mode)
        or resolved_parent != destination.parent
        or (path_entry.st_dev, path_entry.st_ino)
        != (pinned_parent.st_dev, pinned_parent.st_ino)
    ):
        raise ReleaseEvidenceError(
            "closure destination parent changed during publication",
        )


def _validate_published_destination(
    destination: Path,
    parent_fd: int,
    *,
    expected: _PublishedDestinationExpectation,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ReleaseEvidenceError(
            "published closure destination cannot be pinned",
        ) from exc
    try:
        digest, size, identity = _digest_descriptor(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or stat.S_IMODE(identity.st_mode) != 0o400
            or (identity.st_dev, identity.st_ino) != expected.staged_identity
            or identity.st_nlink != expected.link_count
            or size != expected.size_bytes
            or digest != expected.sha256
        ):
            raise ReleaseEvidenceError(
                "published closure destination does not match the staged file",
            )
        try:
            destination_entry = os.stat(
                destination.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ReleaseEvidenceError(
                "published closure destination changed during readback",
            ) from exc
        if (
            not stat.S_ISREG(destination_entry.st_mode)
            or stat.S_IMODE(destination_entry.st_mode) != 0o400
            or (destination_entry.st_dev, destination_entry.st_ino)
            != expected.staged_identity
            or destination_entry.st_nlink != expected.link_count
            or destination_entry.st_size != expected.size_bytes
        ):
            raise ReleaseEvidenceError(
                "published closure destination changed during readback",
            )
        _revalidate_destination_parent(destination, parent_fd)
    finally:
        os.close(descriptor)


def _publish_no_replace(
    destination: Path,
    parent_fd: int,
    raw: bytes,
    *,
    boundary_check: Callable[[], None],
) -> None:
    staging_name = f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    staging_present = False
    destination_linked = False
    published = False
    staged_identity: tuple[int, int] | None = None
    try:
        staging_fd = os.open(
            staging_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        staging_present = True
        written = 0
        while written < len(raw):
            count = os.write(staging_fd, raw[written:])
            if count <= 0:
                raise ReleaseEvidenceError("closure staging write made no progress")
            written += count
        os.fsync(staging_fd)
        os.fchmod(staging_fd, 0o400)
        os.fsync(staging_fd)
        staged = os.fstat(staging_fd)
        if (
            not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o400
            or staged.st_nlink != 1
            or staged.st_size != len(raw)
        ):
            raise ReleaseEvidenceError("closure staging file has an invalid identity")
        staged_digest, staged_size, staged_after = _digest_descriptor(staging_fd)
        if staged_digest != _sha256(raw) or staged_size != len(raw):
            raise ReleaseEvidenceError("closure staging readback failed")
        staged_identity = (staged_after.st_dev, staged_after.st_ino)
        boundary_check()
        _revalidate_destination_parent(destination, parent_fd)
        os.link(
            staging_name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        destination_linked = True
        os.fsync(parent_fd)
        _validate_published_destination(
            destination,
            parent_fd,
            expected=_PublishedDestinationExpectation(
                staged_identity=staged_identity,
                sha256=staged_digest,
                size_bytes=staged_size,
                link_count=2,
            ),
        )
        boundary_check()
        _revalidate_destination_parent(destination, parent_fd)
        os.unlink(staging_name, dir_fd=parent_fd)
        staging_present = False
        os.fsync(parent_fd)
        _validate_published_destination(
            destination,
            parent_fd,
            expected=_PublishedDestinationExpectation(
                staged_identity=staged_identity,
                sha256=staged_digest,
                size_bytes=staged_size,
                link_count=1,
            ),
        )
        boundary_check()
        _revalidate_destination_parent(destination, parent_fd)
        _validate_published_destination(
            destination,
            parent_fd,
            expected=_PublishedDestinationExpectation(
                staged_identity=staged_identity,
                sha256=staged_digest,
                size_bytes=staged_size,
                link_count=1,
            ),
        )
        published = True
    finally:
        try:
            if staging_fd is not None:
                os.close(staging_fd)
        finally:
            try:
                if (
                    not published
                    and destination_linked
                    and staged_identity is not None
                ):
                    with contextlib.suppress(FileNotFoundError):
                        destination_entry = os.stat(
                            destination.name,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        if (destination_entry.st_dev, destination_entry.st_ino) == (
                            staged_identity
                        ):
                            os.unlink(destination.name, dir_fd=parent_fd)
                            os.fsync(parent_fd)
            finally:
                if staging_present:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(staging_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)


def _receipt(
    path: Path,
    raw: bytes,
    prepared: _PreparedClosure,
) -> ReleaseEvidenceReceipt:
    inventory = _require_mapping(
        prepared.payload["inventory"],
        context="closure inventory",
    )
    return ReleaseEvidenceReceipt(
        manifest_path=str(path.absolute()),
        manifest_sha256=_sha256(raw),
        gate_receipt_count=_require_size(
            inventory["gate_receipt_count"],
            context="gate_receipt_count",
        ),
        source_member_count=_require_size(
            inventory["source_member_count"],
            context="source_member_count",
        ),
        ready_count=prepared.ready_count,
        omitted_count=prepared.omitted_count,
    )


def build_release_evidence_closure(  # noqa: PLR0913
    artifact_registry_path: Path,
    renderer_root: Path,
    rendered_output_root: Path,
    gate_receipt_root: Path,
    source_data_root: Path,
    destination: Path,
    *,
    expected_artifact_registry_sha256: str,
) -> ReleaseEvidenceReceipt:
    """Validate all opaque evidence and publish one no-replace closure receipt."""
    absolute_destination, parent_fd = _ensure_destination(
        destination,
        registry_path=artifact_registry_path,
    )
    prepared: _PreparedClosure | None = None
    try:
        prepared = _prepare_closure(
            artifact_registry_path,
            renderer_root,
            rendered_output_root,
            gate_receipt_root,
            source_data_root,
            expected_artifact_registry_sha256=expected_artifact_registry_sha256,
        )
        _require_destination_outside_inputs(
            absolute_destination,
            (
                prepared.renderer_root,
                prepared.rendered_output_root,
                prepared.gate_root.path,
                prepared.source_root.path,
            ),
        )
        raw = _canonical_json(prepared.payload) + b"\n"
        _revalidate_prepared(prepared)
        _publish_no_replace(
            absolute_destination,
            parent_fd,
            raw,
            boundary_check=lambda: _revalidate_prepared(prepared),
        )
        return _receipt(absolute_destination, raw, prepared)
    finally:
        if prepared is not None:
            prepared.close()
        os.close(parent_fd)


def validate_release_evidence_closure(  # noqa: PLR0913
    closure_path: Path,
    artifact_registry_path: Path,
    renderer_root: Path,
    rendered_output_root: Path,
    gate_receipt_root: Path,
    source_data_root: Path,
    *,
    expected_closure_sha256: str,
    expected_artifact_registry_sha256: str,
) -> ReleaseEvidenceReceipt:
    """Rebuild and compare an independently anchored evidence closure."""
    expected_closure = _require_sha256(
        expected_closure_sha256,
        context="expected_closure_sha256",
    )
    closure_file = _pin_absolute_file(
        closure_path,
        context="release-evidence closure",
    )
    prepared: _PreparedClosure | None = None
    try:
        if closure_file.sha256 != expected_closure:
            raise ReleaseEvidenceError(
                "release-evidence closure does not match its independent anchor",
            )
        raw = _descriptor_bytes(closure_file)
        closure = _parse_canonical(raw, context="release-evidence closure")
        declared = dict(closure)
        declared_payload_sha256 = _require_sha256(
            declared.pop("closure_payload_sha256", None),
            context="closure_payload_sha256",
        )
        if _sha256(_canonical_json(declared)) != declared_payload_sha256:
            raise ReleaseEvidenceError("closure payload digest does not match")
        prepared = _prepare_closure(
            artifact_registry_path,
            renderer_root,
            rendered_output_root,
            gate_receipt_root,
            source_data_root,
            expected_artifact_registry_sha256=expected_artifact_registry_sha256,
        )
        if closure != prepared.payload:
            raise ReleaseEvidenceError(
                "release-evidence closure does not match live opaque evidence",
            )
        _revalidate_prepared(prepared)
        _revalidate_file(closure_file, context="release-evidence closure")
        return _receipt(closure_path, raw, prepared)
    finally:
        if prepared is not None:
            prepared.close()
        closure_file.close()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))  # noqa: PTH100


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--artifact-registry", type=Path, required=True)
    shared.add_argument("--renderer-root", type=Path, required=True)
    shared.add_argument("--rendered-output-root", type=Path, required=True)
    shared.add_argument("--gate-receipt-root", type=Path, required=True)
    shared.add_argument("--source-data-root", type=Path, required=True)
    shared.add_argument("--expected-artifact-registry-sha256", required=True)

    build = subparsers.add_parser(
        "build",
        parents=[shared],
        help="publish a canonical no-replace evidence closure",
    )
    build.add_argument("--out", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate",
        parents=[shared],
        help="independently validate an existing evidence closure",
    )
    validate.add_argument("--closure", type=Path, required=True)
    validate.add_argument("--expected-closure-sha256", required=True)
    return parser


def main() -> None:
    """Run the result-blind release-evidence closure CLI."""
    args = _parser().parse_args()
    common = {
        "artifact_registry_path": _absolute(args.artifact_registry),
        "renderer_root": _absolute(args.renderer_root),
        "rendered_output_root": _absolute(args.rendered_output_root),
        "gate_receipt_root": _absolute(args.gate_receipt_root),
        "source_data_root": _absolute(args.source_data_root),
        "expected_artifact_registry_sha256": (
            args.expected_artifact_registry_sha256
        ),
    }
    if args.command == "build":
        receipt = build_release_evidence_closure(
            destination=_absolute(args.out),
            **common,
        )
    else:
        receipt = validate_release_evidence_closure(
            closure_path=_absolute(args.closure),
            expected_closure_sha256=args.expected_closure_sha256,
            **common,
        )
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == "__main__":
    main()

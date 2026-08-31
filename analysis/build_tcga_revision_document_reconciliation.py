"""Seal result-blind manuscript, S1, and rebuttal reconciliation metadata.

This module joins the semantic artifact registry to exact document placements and
reviewer-response items without opening any row-bearing source-data member.  The
artifact registry and document-anchor metadata must be independently hash-anchored,
and the native registry validator must bind opaque renderer/output bytes, before the
document root is opened.  Document files are read only to validate stable placement
markers, exact block hashes, unresolved placeholders, and forbidden stale tokens.

The output is canonical JSON published atomically with no replacement.  It records
document structure and digests, never inserted prose or scientific table rows.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

if __package__:
    from analysis import build_tcga_revision_artifact_registry as _artifact_registry
else:
    import build_tcga_revision_artifact_registry as _artifact_registry

ARTIFACT_REGISTRY_SCHEMA: Final = _artifact_registry.ARTIFACT_REGISTRY_SCHEMA
ARTIFACT_SPECS: Final = _artifact_registry.ARTIFACT_SPECS
ARTIFACT_GATE_ORDER: Final = _artifact_registry.GATE_ORDER
artifact_catalog_sha256 = _artifact_registry.artifact_catalog_sha256

# Validation errors deliberately include their exact field paths.
# Native private normalizers are intentionally reused without opening their members.
# ruff: noqa: EM101, PLR0913, SLF001, TRY003

RECONCILIATION_INPUT_SCHEMA: Final = "dialect-revision-document-reconciliation-input-v1"
DOCUMENT_ANCHOR_SCHEMA: Final = "dialect-revision-document-anchor-v1"
DOCUMENT_RECONCILIATION_SCHEMA: Final = "dialect-revision-document-reconciliation-v1"
DOCUMENT_RECONCILIATION_CONTRACT: Final = "reviewer-document-artifact-claim-closure-v1"
TRUST_MODEL: Final = {
    "artifact_registry": (
        "independently SHA-256 anchored and natively validated against opaque "
        "renderer/output bytes before document bytes are read"
    ),
    "documents": (
        "independently anchored text members opened only for structural reconciliation"
    ),
    "row_bearing_data": (
        "never opened; only artifact semantic identities and registry metadata are used"
    ),
    "scientific_scope": (
        "no result interpretation, scientific approval, or submission approval "
        "is inferred"
    ),
}

DOCUMENT_ORDER: Final = (
    ("main", "manuscript"),
    ("s1", "supporting-information"),
    ("rebuttal", "rebuttal"),
)
DOCUMENT_IDS: Final = tuple(record[0] for record in DOCUMENT_ORDER)
DOCUMENT_ROLES: Final = {record[0]: record[1] for record in DOCUMENT_ORDER}

REVIEWER_ITEM_ORDER: Final = (
    "r1-overall",
    "r1-1",
    "r1-2",
    "r1-3",
    "r1-4",
    "r1-5a",
    "r1-5b",
    "r1-5c",
    "r1-minor-1",
    "r1-minor-2",
    "r1-minor-3",
    "r1-minor-4",
    "r1-minor-5",
    "r2-overall",
    "r2-1",
    "r2-2",
    "r2-3",
    "r2-4",
    "r2-5",
    "r3-overall",
    "r3-1",
    "r3-2",
    "r3-3",
    "r3-4",
    "r3-5",
    "r3-6",
    "data-code-availability",
)
DOCUMENT_GATE_ORDER: Final = (
    "K500",
    "CAL",
    "COAUTH",
    "COMP",
    "MSK",
    "SIM",
    "FIG",
    "TABLE",
    "REL",
)
PLACEMENT_KIND_ORDER: Final = (
    "response",
    "prose",
    "figure",
    "table",
    "supplementary-data",
    "provenance",
)
PLACEMENT_STATUS_ORDER: Final = ("pending", "ready", "omitted")
OMISSION_REASON_ORDER: Final = (
    "required_gate_not_satisfied",
    "coauthor_decision_to_omit",
    "release_scope_exclusion",
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_RELEASE_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_MARKER_RE: Final = re.compile(
    r"RECONCILIATION-TARGET:([a-z0-9][a-z0-9._-]{2,127}):(BEGIN|END)",
)
_MARKDOWN_PLACEHOLDER_RE: Final = re.compile(
    r"\*\*\[[A-Z][A-Z0-9+]*:[\s\S]*?\]\*\*",
)
_PENDING_MARKER_RE: Final = re.compile(r"RECONCILIATION-PENDING:")
_TEX_GATE_RE: Final = re.compile(
    r"(?im)^\s*%\s*[A-Z0-9]+(?:\+[A-Z0-9]+)*\s+gate:",
)
_GENERIC_PLACEHOLDER_RE: Final = re.compile(
    r"(?im)(?:^|\s)(?:TODO|TBD|FIXME)(?=[:\s]|$)",
)
_READ_CHUNK_BYTES: Final = 1024 * 1024
_MAX_METADATA_BYTES: Final = 2 * 1024 * 1024
_MAX_DOCUMENT_BYTES: Final = 16 * 1024 * 1024
_BUILDER_MEMBER: Final = "analysis/build_tcga_revision_document_reconciliation.py"
_EXPECTED_ARTIFACT_COUNT: Final = 13
_EXPECTED_REVIEWER_ITEM_COUNT: Final = 27
_EXPECTED_DOCUMENT_COUNT: Final = 3
_ARTIFACT_KIND_TO_PLACEMENT_KIND: Final = {
    "figure": "figure",
    "table": "table",
    "supplementary-data": "supplementary-data",
    "provenance-record": "provenance",
}
_ARTIFACT_KIND_TO_REPRESENTATION_GATE: Final = {
    "figure": "FIG",
    "table": "TABLE",
    "supplementary-data": "REL",
    "provenance-record": "REL",
}


class DocumentReconciliationError(ValueError):
    """Raised when document reconciliation is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class DocumentReconciliationReceipt:
    """Describe one published or independently validated reconciliation."""

    manifest_path: str
    manifest_sha256: str
    mode: str
    placement_count: int
    ready_count: int
    omitted_count: int
    pending_count: int


@dataclass(slots=True)
class _PinnedFile:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    raw: bytes

    def close(self) -> None:
        """Close the held descriptor."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _PinnedRoot:
    path: Path
    descriptor: int
    device: int
    inode: int
    mtime_ns: int

    def close(self) -> None:
        """Close the held root descriptor."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(frozen=True, slots=True)
class _DocumentBlock:
    placement_id: str
    content: bytes
    start_line: int
    end_line: int


def _fail(message: str) -> NoReturn:
    raise DocumentReconciliationError(message)


def _validate_contract_dimensions() -> None:
    if len(ARTIFACT_SPECS) != _EXPECTED_ARTIFACT_COUNT:
        _fail("native artifact catalog must contain exactly 13 artifacts")
    if len(REVIEWER_ITEM_ORDER) != _EXPECTED_REVIEWER_ITEM_COUNT:
        _fail("reviewer-item catalog must contain exactly 27 items")
    if len(DOCUMENT_ORDER) != _EXPECTED_DOCUMENT_COUNT:
        _fail("document catalog must contain exactly main, S1, and rebuttal")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        _fail(f"value is not canonical-JSON encodable: {error}")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object")
    return value


def _expect_sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        _fail(f"{context} must be an array")
    return value


def _expect_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{context} has invalid keys; expected {sorted(expected)}, "
            f"found {sorted(actual)}",
        )


def _expect_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{context} must be a nonempty string")
    return value


def _expect_token(value: object, *, context: str) -> str:
    token = _expect_string(value, context=context)
    if _TOKEN_RE.fullmatch(token) is None:
        _fail(f"{context} is not a canonical token")
    return token


def _expect_sha256(value: object, *, context: str) -> str:
    digest = _expect_string(value, context=context)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _expect_nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{context} must be a nonnegative integer")
    return value


def _expect_positive_int(value: object, *, context: str) -> int:
    result = _expect_nonnegative_int(value, context=context)
    if result == 0:
        _fail(f"{context} must be positive")
    return result


def _expect_relative_member(value: object, *, context: str) -> str:
    member = _expect_string(value, context=context)
    if (
        not member.isascii()
        or "\\" in member
        or any(ord(character) < 32 or ord(character) == 127 for character in member)
    ):
        _fail(f"{context} is not a canonical POSIX member")
    pure = PurePosixPath(member)
    if (
        pure.is_absolute()
        or pure.as_posix() != member
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{context} must be a safe relative POSIX member")
    return member


def _read_descriptor(descriptor: int, *, maximum: int, context: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _pin_file(path: Path, *, maximum: int, context: str) -> _PinnedFile:
    absolute = path.absolute()
    try:
        path_entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if (
        stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISREG(path_entry.st_mode)
        or resolved != absolute
    ):
        _fail(f"{context} must be a canonical non-symlink regular file")
    if path_entry.st_nlink != 1:
        _fail(f"{context} must have exactly one hard link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode):
            _fail(f"{context} must be a regular file")
        if entry.st_nlink != 1:
            _fail(f"{context} must have exactly one hard link")
        if (
            entry.st_dev,
            entry.st_ino,
            entry.st_size,
            entry.st_mtime_ns,
        ) != (
            path_entry.st_dev,
            path_entry.st_ino,
            path_entry.st_size,
            path_entry.st_mtime_ns,
        ):
            _fail(f"{context} changed while it was pinned")
        raw = _read_descriptor(descriptor, maximum=maximum, context=context)
        if len(raw) != entry.st_size:
            _fail(f"{context} changed while it was read")
        return _PinnedFile(
            path=absolute,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            sha256=_sha256(raw),
            raw=raw,
        )
    except Exception:
        os.close(descriptor)
        raise


def _pin_root(path: Path, *, context: str) -> _PinnedRoot:
    absolute = path.absolute()
    try:
        path_entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if (
        stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISDIR(path_entry.st_mode)
        or resolved != absolute
    ):
        _fail(f"{context} must be a canonical non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    try:
        entry = os.fstat(descriptor)
    except OSError:
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(entry.st_mode):
        os.close(descriptor)
        _fail(f"{context} must be a directory")
    if (entry.st_dev, entry.st_ino) != (path_entry.st_dev, path_entry.st_ino):
        os.close(descriptor)
        _fail(f"{context} changed while it was pinned")
    return _PinnedRoot(
        path=absolute,
        descriptor=descriptor,
        device=entry.st_dev,
        inode=entry.st_ino,
        mtime_ns=entry.st_mtime_ns,
    )


def _open_root_member(
    root: _PinnedRoot,
    member: str,
    *,
    maximum: int,
    context: str,
) -> _PinnedFile:
    parts = PurePosixPath(member).parts
    directory_descriptor = os.dup(root.descriptor)
    try:
        for part in parts[:-1]:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            next_descriptor = os.open(part, flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(parts[-1], flags, dir_fd=directory_descriptor)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    finally:
        os.close(directory_descriptor)
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode):
            _fail(f"{context} must be a regular file")
        if entry.st_nlink != 1:
            _fail(f"{context} must have exactly one hard link")
        raw = _read_descriptor(descriptor, maximum=maximum, context=context)
        if len(raw) != entry.st_size:
            _fail(f"{context} changed while it was read")
        return _PinnedFile(
            path=root.path / member,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            sha256=_sha256(raw),
            raw=raw,
        )
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_file(pinned: _PinnedFile, *, context: str) -> None:
    entry = os.fstat(pinned.descriptor)
    try:
        path_entry = os.lstat(pinned.path)
        resolved = pinned.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} path disappeared after validation: {error}")
    actual = (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns)
    expected = (pinned.device, pinned.inode, pinned.size, pinned.mtime_ns)
    if (
        actual != expected
        or not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISREG(path_entry.st_mode)
        or resolved != pinned.path
        or (path_entry.st_dev, path_entry.st_ino) != (pinned.device, pinned.inode)
    ):
        _fail(f"{context} identity changed after validation")
    raw = _read_descriptor(
        pinned.descriptor,
        maximum=max(pinned.size, 1),
        context=context,
    )
    if _sha256(raw) != pinned.sha256:
        _fail(f"{context} bytes changed after validation")


def _revalidate_root(root: _PinnedRoot, *, context: str) -> None:
    entry = os.fstat(root.descriptor)
    try:
        path_entry = os.lstat(root.path)
        resolved = root.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} path disappeared after validation: {error}")
    actual = (entry.st_dev, entry.st_ino, entry.st_mtime_ns)
    expected = (root.device, root.inode, root.mtime_ns)
    if (
        actual != expected
        or not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISDIR(path_entry.st_mode)
        or resolved != root.path
        or (path_entry.st_dev, path_entry.st_ino) != (root.device, root.inode)
    ):
        _fail(f"{context} identity changed after validation")


def _parse_canonical_json(pinned: _PinnedFile, *, context: str) -> Mapping[str, object]:
    try:
        text = pinned.raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"{context} must be ASCII canonical JSON: {error}")

    def reject_constant(value: str) -> NoReturn:
        _fail(f"{context} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        _fail(f"{context} is not valid JSON: {error}")
    if pinned.raw != _canonical_json(value) + b"\n":
        _fail(f"{context} is not canonical JSON with one trailing newline")
    return _expect_mapping(value, context=context)


def _normalize_gate_ledger(value: object, *, context: str) -> list[dict[str, str]]:
    records = _expect_sequence(value, context=context)
    by_gate: dict[str, dict[str, str]] = {}
    for index, raw_record in enumerate(records):
        record_context = f"{context}[{index}]"
        record = _expect_mapping(raw_record, context=record_context)
        _expect_keys(
            record,
            {"gate", "receipt_id", "sha256"},
            context=record_context,
        )
        gate = _expect_string(record["gate"], context=f"{record_context}.gate")
        if gate not in DOCUMENT_GATE_ORDER:
            _fail(f"{record_context}.gate is not recognized")
        if gate in by_gate:
            _fail(f"{context} duplicates gate {gate!r}")
        by_gate[gate] = {
            "gate": gate,
            "receipt_id": _expect_token(
                record["receipt_id"],
                context=f"{record_context}.receipt_id",
            ),
            "sha256": _expect_sha256(
                record["sha256"],
                context=f"{record_context}.sha256",
            ),
        }
    normalized = [by_gate[gate] for gate in DOCUMENT_GATE_ORDER if gate in by_gate]
    if list(records) != normalized:
        _fail(f"{context} is not canonically ordered")
    return normalized


def _normalize_document_anchor(value: Mapping[str, object]) -> list[dict[str, object]]:
    _validate_contract_dimensions()
    _expect_keys(value, {"schema", "documents"}, context="document anchor")
    if value["schema"] != DOCUMENT_ANCHOR_SCHEMA:
        _fail("document anchor has the wrong schema")
    records = _expect_sequence(value["documents"], context="document anchor.documents")
    if len(records) != len(DOCUMENT_ORDER):
        _fail("document anchor must contain exactly main, S1, and rebuttal")
    normalized: list[dict[str, object]] = []
    members: set[str] = set()
    for index, (expected_id, expected_role) in enumerate(DOCUMENT_ORDER):
        context = f"document anchor.documents[{index}]"
        record = _expect_mapping(records[index], context=context)
        _expect_keys(
            record,
            {"document_id", "role", "member", "bytes", "sha256"},
            context=context,
        )
        if record["document_id"] != expected_id or record["role"] != expected_role:
            _fail(f"{context} is not in canonical document order")
        member = _expect_relative_member(record["member"], context=f"{context}.member")
        if member in members:
            _fail("document anchor reuses a document member")
        members.add(member)
        normalized.append(
            {
                "document_id": expected_id,
                "role": expected_role,
                "member": member,
                "bytes": _expect_positive_int(
                    record["bytes"],
                    context=f"{context}.bytes",
                ),
                "sha256": _expect_sha256(
                    record["sha256"],
                    context=f"{context}.sha256",
                ),
            },
        )
    if list(records) != normalized:
        _fail("document anchor is not canonical")
    return normalized


def _normalize_registry_renderer_metadata(
    value: object,
    *,
    context: str,
) -> dict[str, object]:
    renderer = _expect_mapping(value, context=context)
    _expect_keys(renderer, {"script", "sha256", "bytes"}, context=context)
    try:
        script = _artifact_registry._expect_relative_member(
            renderer["script"],
            context=f"{context}.script",
        )
        size = _artifact_registry._expect_size(
            renderer["bytes"],
            context=f"{context}.bytes",
        )
    except _artifact_registry.ArtifactRegistryError as error:
        _fail(f"artifact registry is not native-canonical: {error}")
    if not script.startswith("analysis/") or not script.endswith(".py"):
        _fail(f"{context}.script must be an analysis Python module")
    return {
        "script": script,
        "sha256": _expect_sha256(renderer["sha256"], context=f"{context}.sha256"),
        "bytes": size,
    }


def _normalize_registry_outputs_metadata(
    value: object,
    *,
    artifact_id: str,
    artifact_kind: str,
    context: str,
) -> list[dict[str, object]]:
    records = _expect_sequence(value, context=context)
    if not records:
        _fail(f"{context} must contain at least one output")
    normalized: list[dict[str, object]] = []
    output_ids: set[str] = set()
    members: set[str] = set()
    for index, raw_record in enumerate(records):
        record_context = f"{context}[{index}]"
        record = _expect_mapping(raw_record, context=record_context)
        _expect_keys(
            record,
            {"output_id", "release_member", "media_type", "sha256", "bytes"},
            context=record_context,
        )
        try:
            output_id = _artifact_registry._expect_token(
                record["output_id"],
                context=f"{record_context}.output_id",
            )
            member = _artifact_registry._expect_relative_member(
                record["release_member"],
                context=f"{record_context}.release_member",
            )
        except _artifact_registry.ArtifactRegistryError as error:
            _fail(f"artifact registry is not native-canonical: {error}")
        required_prefix = f"rendered/{artifact_id}/"
        if not member.startswith(required_prefix):
            _fail(f"{record_context}.release_member has the wrong artifact prefix")
        media_type = _expect_string(
            record["media_type"],
            context=f"{record_context}.media_type",
        )
        suffixes = _artifact_registry._OUTPUT_MEDIA_SUFFIXES.get(media_type)
        if suffixes is None or not member.endswith(suffixes):
            _fail(f"{record_context} has an unsupported media type or suffix")
        if output_id in output_ids or member in members:
            _fail(f"{context} contains duplicate output identity")
        output_ids.add(output_id)
        members.add(member)
        normalized.append(
            {
                "output_id": output_id,
                "release_member": member,
                "media_type": media_type,
                "sha256": _expect_sha256(
                    record["sha256"],
                    context=f"{record_context}.sha256",
                ),
                "bytes": _expect_nonnegative_int(
                    record["bytes"],
                    context=f"{record_context}.bytes",
                ),
            },
        )
    normalized.sort(key=lambda record: str(record["output_id"]))
    required_media = _artifact_registry._REQUIRED_MEDIA_BY_KIND.get(artifact_kind)
    if required_media is None or not any(
        record["media_type"] in required_media for record in normalized
    ):
        _fail(f"{context} lacks a presentation output for {artifact_kind!r}")
    if list(records) != normalized:
        _fail(f"{context} is not canonically ordered")
    return normalized


def _pinned_script_record(
    pinned: _PinnedFile,
    *,
    member: str,
) -> dict[str, object]:
    return {
        "script": member,
        "sha256": pinned.sha256,
        "bytes": pinned.size,
    }


def _validate_native_artifact_registry(
    artifact_registry_path: Path,
    renderer_root_path: Path,
    rendered_output_root_path: Path,
    *,
    expected_sha256: str,
) -> _artifact_registry.ArtifactRegistryReceipt:
    """Run the native opaque-byte validator without opening source-data rows."""
    try:
        receipt = _artifact_registry.validate_artifact_registry(
            artifact_registry_path,
            renderer_root_path,
            rendered_output_root_path,
            expected_manifest_sha256=expected_sha256,
        )
    except (_artifact_registry.ArtifactRegistryError, OSError) as error:
        message = f"native artifact registry validation failed: {error}"
        raise DocumentReconciliationError(message) from error
    if receipt.manifest_sha256 != expected_sha256:
        _fail("native artifact registry validation returned the wrong digest")
    if receipt.ready_count + receipt.omitted_count != len(ARTIFACT_SPECS):
        _fail("native artifact registry validation returned the wrong artifact count")
    return receipt


def _normalize_registry(
    value: Mapping[str, object],
    *,
    expected_builder: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]], list[dict[str, str]]]:
    _validate_contract_dimensions()
    required = {
        "schema",
        "contract",
        "trust_model",
        "release",
        "builder",
        "gate_catalog",
        "gate_ledger",
        "artifact_catalog_sha256",
        "artifacts",
        "registry_payload_sha256",
    }
    _expect_keys(value, required, context="artifact registry")
    if value["schema"] != ARTIFACT_REGISTRY_SCHEMA:
        _fail("artifact registry has the wrong schema")
    if value["contract"] != _artifact_registry.ARTIFACT_REGISTRY_CONTRACT:
        _fail("artifact registry has the wrong native contract")
    if value["trust_model"] != _artifact_registry.TRUST_MODEL:
        _fail("artifact registry changes the native trust model")
    if value["gate_catalog"] != [
        dict(record) for record in _artifact_registry.GATE_CATALOG
    ]:
        _fail("artifact registry changes the native gate catalog")
    if value["artifact_catalog_sha256"] != artifact_catalog_sha256():
        _fail("artifact registry changes the canonical artifact catalog")
    payload = dict(value)
    declared_payload_sha = _expect_sha256(
        payload.pop("registry_payload_sha256"),
        context="artifact registry.registry_payload_sha256",
    )
    if _sha256(_canonical_json(payload)) != declared_payload_sha:
        _fail("artifact registry payload digest does not match")
    try:
        release = _artifact_registry._normalize_release(
            value["release"],
            context="artifact registry.release",
        )
        registry_gate_ledger = _artifact_registry._normalize_gate_receipts(
            value["gate_ledger"],
            permitted_gates=ARTIFACT_GATE_ORDER,
            require_all=False,
            context="artifact registry.gate_ledger",
        )
    except _artifact_registry.ArtifactRegistryError as error:
        _fail(f"artifact registry is not native-canonical: {error}")
    if value["gate_ledger"] != registry_gate_ledger:
        _fail("artifact registry gate ledger is not canonically ordered")
    builder = _normalize_registry_renderer_metadata(
        value["builder"],
        context="artifact registry.builder",
    )
    if builder != dict(expected_builder):
        _fail("artifact registry does not bind the live native builder")
    artifacts_raw = _expect_sequence(
        value["artifacts"],
        context="artifact registry.artifacts",
    )
    if len(artifacts_raw) != len(ARTIFACT_SPECS):
        _fail("artifact registry has the wrong artifact count")
    artifacts: dict[str, dict[str, object]] = {}
    validated_artifacts: list[Mapping[str, object]] = []
    ready_count = 0
    global_gates = frozenset(record["gate"] for record in registry_gate_ledger)
    for index, spec in enumerate(ARTIFACT_SPECS):
        context = f"artifact registry.artifacts[{index}]"
        artifact = _expect_mapping(artifacts_raw[index], context=context)
        status = _expect_string(artifact.get("status"), context=f"{context}.status")
        base_keys = {
            "semantic_id",
            "title",
            "kind",
            "required_gates",
            "required_source_roles",
            "source_requirement",
            "status",
            "gate_receipts",
        }
        if status == "ready":
            _expect_keys(
                artifact,
                base_keys | {"source_data", "renderer", "outputs", "claims"},
                context=context,
            )
        elif status == "omitted":
            _expect_keys(
                artifact,
                base_keys | {"omission", "planned_claims"},
                context=context,
            )
        else:
            _fail(f"{context}.status must be ready or omitted")
        try:
            _artifact_registry._validate_catalog_fields(
                artifact,
                spec=spec,
                context=context,
            )
            receipts = _artifact_registry._normalize_gate_receipts(
                artifact["gate_receipts"],
                permitted_gates=spec.required_gates,
                require_all=status == "ready",
                context=f"{context}.gate_receipts",
            )
            expected_receipts = _artifact_registry._artifact_gate_receipts(
                registry_gate_ledger,
                spec,
            )
        except _artifact_registry.ArtifactRegistryError as error:
            _fail(f"artifact registry is not native-canonical: {error}")
        if artifact["gate_receipts"] != receipts or receipts != expected_receipts:
            _fail(f"{context}.gate_receipts contradicts the native ledger")
        expected_claims = [claim.as_record() for claim in spec.claims]
        reason_code: str | None = None
        if status == "ready":
            try:
                sources = _artifact_registry._normalize_sources(
                    artifact["source_data"],
                    source_requirement=spec.source_requirement,
                    required_source_roles=spec.required_source_roles,
                    upstream_manifest_sha256=(
                        release["postprocess_release_sha256"],
                        release["source_data_manifest_sha256"],
                    ),
                    context=f"{context}.source_data",
                )
            except _artifact_registry.ArtifactRegistryError as error:
                _fail(f"artifact registry is not native-canonical: {error}")
            if artifact["source_data"] != sources:
                _fail(f"{context}.source_data is not canonically ordered")
            renderer = _normalize_registry_renderer_metadata(
                artifact["renderer"],
                context=f"{context}.renderer",
            )
            if artifact["renderer"] != renderer:
                _fail(f"{context}.renderer is not canonical")
            _normalize_registry_outputs_metadata(
                artifact["outputs"],
                artifact_id=spec.semantic_id,
                artifact_kind=spec.kind,
                context=f"{context}.outputs",
            )
            if artifact["claims"] != expected_claims:
                _fail(f"{context}.claims is not canonical")
            ready_count += 1
        else:
            try:
                omission = _artifact_registry._normalize_omission(
                    artifact["omission"],
                    spec=spec,
                    receipts=receipts,
                    global_receipt_gates=global_gates,
                    context=f"{context}.omission",
                    include_reason=True,
                )
            except _artifact_registry.ArtifactRegistryError as error:
                _fail(f"artifact registry is not native-canonical: {error}")
            if artifact["omission"] != omission:
                _fail(f"{context}.omission is not native-canonical")
            if artifact["planned_claims"] != expected_claims:
                _fail(f"{context}.planned_claims is not canonical")
            reason_code = str(omission["reason_code"])
        artifacts[spec.semantic_id] = {
            "semantic_id": spec.semantic_id,
            "status": status,
            "kind": spec.kind,
            "required_gates": list(spec.required_gates),
            "claims": expected_claims,
            "omission_reason_code": reason_code,
        }
        validated_artifacts.append(artifact)
    try:
        _artifact_registry._validate_global_reconciliation(
            validated_artifacts,
            gate_ledger=registry_gate_ledger,
        )
    except _artifact_registry.ArtifactRegistryError as error:
        _fail(f"artifact registry is not native-canonical: {error}")
    summary = {
        "release_id": release["release_id"],
        "registry_payload_sha256": declared_payload_sha,
        "artifact_catalog_sha256": artifact_catalog_sha256(),
        "artifact_count": len(artifacts),
        "ready_count": ready_count,
        "omitted_count": len(artifacts) - ready_count,
    }
    return summary, artifacts, registry_gate_ledger


def _decode_document(pinned: _PinnedFile, *, context: str) -> str:
    try:
        text = pinned.raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{context} must be UTF-8 text: {error}")
    if "\x00" in text or "\r" in text:
        _fail(f"{context} must use NUL-free LF text")
    return text


def _marker_line(document_id: str, placement_id: str, boundary: str) -> str:
    marker = f"RECONCILIATION-TARGET:{placement_id}:{boundary}"
    if document_id == "rebuttal":
        return f"<!-- {marker} -->"
    return f"% {marker}"


def _extract_blocks(document_id: str, text: str) -> dict[str, _DocumentBlock]:
    lines = text.splitlines(keepends=True)
    blocks: dict[str, _DocumentBlock] = {}
    open_id: str | None = None
    content: list[str] = []
    start_line = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.rstrip("\n")
        marker_match = _MARKER_RE.search(stripped)
        if "RECONCILIATION-TARGET:" in stripped and marker_match is None:
            _fail(
                f"document {document_id} has malformed target marker at line "
                f"{line_number}",
            )
        if marker_match is None:
            if open_id is not None:
                content.append(line)
            continue
        placement_id, boundary = marker_match.groups()
        if stripped != _marker_line(document_id, placement_id, boundary):
            _fail(
                f"document {document_id} has noncanonical target marker at line "
                f"{line_number}",
            )
        if boundary == "BEGIN":
            if open_id is not None:
                _fail(f"document {document_id} nests target markers")
            if placement_id in blocks:
                _fail(f"document {document_id} duplicates target {placement_id!r}")
            open_id = placement_id
            content = []
            start_line = line_number
        else:
            if open_id != placement_id:
                _fail(
                    f"document {document_id} has unmatched target end {placement_id!r}",
                )
            blocks[placement_id] = _DocumentBlock(
                placement_id=placement_id,
                content="".join(content).encode("utf-8"),
                start_line=start_line,
                end_line=line_number,
            )
            open_id = None
            content = []
    if open_id is not None:
        _fail(f"document {document_id} has unterminated target {open_id!r}")
    return blocks


def _contains_unresolved(text: str) -> bool:
    return any(
        pattern.search(text) is not None
        for pattern in (
            _MARKDOWN_PLACEHOLDER_RE,
            _PENDING_MARKER_RE,
            _TEX_GATE_RE,
            _GENERIC_PLACEHOLDER_RE,
        )
    )


def _normalize_forbidden_tokens(value: object) -> list[dict[str, object]]:
    records = _expect_sequence(value, context="reconciliation.forbidden_tokens")
    if not records:
        _fail("reconciliation.forbidden_tokens must not be empty")
    normalized: list[dict[str, object]] = []
    token_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        context = f"reconciliation.forbidden_tokens[{index}]"
        record = _expect_mapping(raw_record, context=context)
        _expect_keys(
            record,
            {"token_id", "literal", "document_ids"},
            context=context,
        )
        token_id = _expect_token(record["token_id"], context=f"{context}.token_id")
        if token_id in token_ids:
            _fail("reconciliation.forbidden_tokens duplicates token_id")
        token_ids.add(token_id)
        literal = _expect_string(record["literal"], context=f"{context}.literal")
        if len(literal) > 512 or "\n" in literal or "\r" in literal:
            _fail(f"{context}.literal must be a single line of at most 512 characters")
        raw_documents = _expect_sequence(
            record["document_ids"],
            context=f"{context}.document_ids",
        )
        documents: list[str] = []
        for document_index, raw_document in enumerate(raw_documents):
            document = _expect_string(
                raw_document,
                context=f"{context}.document_ids[{document_index}]",
            )
            if document not in DOCUMENT_IDS or document in documents:
                _fail(f"{context}.document_ids is invalid or duplicated")
            documents.append(document)
        documents = [document for document in DOCUMENT_IDS if document in documents]
        if not documents:
            _fail(f"{context}.document_ids must not be empty")
        normalized.append(
            {
                "token_id": token_id,
                "literal": literal,
                "document_ids": documents,
            },
        )
    normalized.sort(key=lambda record: str(record["token_id"]))
    if list(records) != normalized:
        _fail("reconciliation.forbidden_tokens is not canonical")
    return normalized


def _normalize_page_location(
    value: object,
    *,
    context: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    record = _expect_mapping(value, context=context)
    _expect_keys(
        record,
        {"pdf_sha256", "page", "line_start", "line_end"},
        context=context,
    )
    page = _expect_positive_int(record["page"], context=f"{context}.page")
    line_start = _expect_positive_int(
        record["line_start"],
        context=f"{context}.line_start",
    )
    line_end = _expect_positive_int(
        record["line_end"],
        context=f"{context}.line_end",
    )
    if line_end < line_start:
        _fail(f"{context}.line_end precedes line_start")
    return {
        "pdf_sha256": _expect_sha256(
            record["pdf_sha256"],
            context=f"{context}.pdf_sha256",
        ),
        "page": page,
        "line_start": line_start,
        "line_end": line_end,
    }


def _normalize_artifact_claims(
    value: object,
    *,
    context: str,
    artifact_claim_order: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    records = _expect_sequence(value, context=context)
    allowed = set(artifact_claim_order)
    seen: set[tuple[str, str]] = set()
    claims: list[tuple[str, str]] = []
    for index, raw_record in enumerate(records):
        record_context = f"{context}[{index}]"
        record = _expect_mapping(raw_record, context=record_context)
        _expect_keys(record, {"artifact_id", "claim_id"}, context=record_context)
        pair = (
            _expect_token(
                record["artifact_id"],
                context=f"{record_context}.artifact_id",
            ),
            _expect_token(record["claim_id"], context=f"{record_context}.claim_id"),
        )
        if pair not in allowed:
            _fail(f"{record_context} is not a canonical artifact claim")
        if pair in seen:
            _fail(f"{context} duplicates artifact claim {pair!r}")
        seen.add(pair)
        claims.append(pair)
    order_index = {pair: index for index, pair in enumerate(artifact_claim_order)}
    claims.sort(key=order_index.__getitem__)
    normalized = [
        {"artifact_id": artifact_id, "claim_id": claim_id}
        for artifact_id, claim_id in claims
    ]
    if list(records) != normalized:
        _fail(f"{context} is not canonically ordered")
    return normalized


def _normalize_omission(
    value: object,
    *,
    context: str,
    forbidden_ids: set[str],
) -> dict[str, object]:
    record = _expect_mapping(value, context=context)
    _expect_keys(
        record,
        {"reason_code", "receipt_id", "sha256", "forbidden_token_ids"},
        context=context,
    )
    reason = _expect_string(record["reason_code"], context=f"{context}.reason_code")
    if reason not in OMISSION_REASON_ORDER:
        _fail(f"{context}.reason_code is not recognized")
    raw_token_ids = _expect_sequence(
        record["forbidden_token_ids"],
        context=f"{context}.forbidden_token_ids",
    )
    token_ids: list[str] = []
    for index, raw_token in enumerate(raw_token_ids):
        token = _expect_token(
            raw_token,
            context=f"{context}.forbidden_token_ids[{index}]",
        )
        if token not in forbidden_ids or token in token_ids:
            _fail(f"{context}.forbidden_token_ids is invalid or duplicated")
        token_ids.append(token)
    token_ids.sort()
    if not token_ids or list(raw_token_ids) != token_ids:
        _fail(f"{context}.forbidden_token_ids must be nonempty and canonical")
    return {
        "reason_code": reason,
        "receipt_id": _expect_token(
            record["receipt_id"],
            context=f"{context}.receipt_id",
        ),
        "sha256": _expect_sha256(record["sha256"], context=f"{context}.sha256"),
        "forbidden_token_ids": token_ids,
    }


def _normalize_placements(
    value: object,
    *,
    mode: str,
    artifact_records: Mapping[str, Mapping[str, object]],
    artifact_claim_order: tuple[tuple[str, str], ...],
    gate_ledger: Sequence[Mapping[str, str]],
    forbidden_tokens: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records = _expect_sequence(value, context="reconciliation.placements")
    if not records:
        _fail("reconciliation.placements must not be empty")
    satisfied_gates = {str(record["gate"]) for record in gate_ledger}
    forbidden_ids = {str(record["token_id"]) for record in forbidden_tokens}
    forbidden_documents = {
        str(record["token_id"]): {
            str(document_id)
            for document_id in _expect_sequence(
                record["document_ids"],
                context="forbidden token document_ids",
            )
        }
        for record in forbidden_tokens
    }
    placements: list[dict[str, object]] = []
    placement_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        context = f"reconciliation.placements[{index}]"
        record = _expect_mapping(raw_record, context=context)
        _expect_keys(
            record,
            {
                "placement_id",
                "document_id",
                "kind",
                "status",
                "required_gates",
                "artifact_claims",
                "content_sha256",
                "page_location",
                "omission",
            },
            context=context,
        )
        placement_id = _expect_token(
            record["placement_id"],
            context=f"{context}.placement_id",
        )
        if placement_id in placement_ids:
            _fail("reconciliation.placements duplicates placement_id")
        placement_ids.add(placement_id)
        document_id = _expect_string(
            record["document_id"],
            context=f"{context}.document_id",
        )
        if document_id not in DOCUMENT_IDS:
            _fail(f"{context}.document_id is not recognized")
        kind = _expect_string(record["kind"], context=f"{context}.kind")
        if kind not in PLACEMENT_KIND_ORDER:
            _fail(f"{context}.kind is not recognized")
        if (kind == "response") != (document_id == "rebuttal"):
            _fail(f"{context} response placements must be in the rebuttal only")
        status = _expect_string(record["status"], context=f"{context}.status")
        if status not in PLACEMENT_STATUS_ORDER:
            _fail(f"{context}.status is not recognized")
        if status == "pending" and mode != "draft":
            _fail("final reconciliation cannot contain pending placements")
        raw_gates = _expect_sequence(
            record["required_gates"],
            context=f"{context}.required_gates",
        )
        required_gates: list[str] = []
        for gate_index, raw_gate in enumerate(raw_gates):
            gate = _expect_string(
                raw_gate,
                context=f"{context}.required_gates[{gate_index}]",
            )
            if gate not in DOCUMENT_GATE_ORDER or gate in required_gates:
                _fail(f"{context}.required_gates is invalid or duplicated")
            required_gates.append(gate)
        required_gates = [
            gate for gate in DOCUMENT_GATE_ORDER if gate in required_gates
        ]
        if list(raw_gates) != required_gates:
            _fail(f"{context}.required_gates is not canonical")
        claims = _normalize_artifact_claims(
            record["artifact_claims"],
            context=f"{context}.artifact_claims",
            artifact_claim_order=artifact_claim_order,
        )
        claim_artifacts = {str(claim["artifact_id"]) for claim in claims}
        if kind == "response":
            if claims:
                _fail(f"{context} response placement cannot own artifact claims")
            if required_gates != ["COAUTH"]:
                _fail(f"{context} response placement must require exactly COAUTH")
        elif claim_artifacts:
            artifact_kinds = {
                str(artifact_records[artifact_id]["kind"])
                for artifact_id in claim_artifacts
            }
            expected_kinds = {
                _ARTIFACT_KIND_TO_PLACEMENT_KIND[artifact_kind]
                for artifact_kind in artifact_kinds
            }
            if len(expected_kinds) != 1 or kind not in expected_kinds:
                _fail(
                    f"{context}.kind contradicts its artifact catalog kinds "
                    f"{sorted(artifact_kinds)}",
                )
            expected_gate_set = {
                str(gate)
                for artifact_id in claim_artifacts
                for gate in _expect_sequence(
                    artifact_records[artifact_id]["required_gates"],
                    context=f"artifact {artifact_id} required_gates",
                )
            }
            expected_gate_set.update(
                _ARTIFACT_KIND_TO_REPRESENTATION_GATE[artifact_kind]
                for artifact_kind in artifact_kinds
            )
            expected_gates = [
                gate for gate in DOCUMENT_GATE_ORDER if gate in expected_gate_set
            ]
            if required_gates != expected_gates:
                _fail(
                    f"{context}.required_gates contradicts its artifact catalog; "
                    f"expected {expected_gates}",
                )
        content_sha: str | None
        page_location = _normalize_page_location(
            record["page_location"],
            context=f"{context}.page_location",
        )
        omission: dict[str, object] | None
        if status == "pending":
            if record["content_sha256"] is not None:
                _fail(f"{context}.content_sha256 must be null while pending")
            if page_location is not None or record["omission"] is not None:
                _fail(f"{context} pending placement cannot have final evidence")
            content_sha = None
            omission = None
        elif status == "ready":
            content_sha = _expect_sha256(
                record["content_sha256"],
                context=f"{context}.content_sha256",
            )
            if record["omission"] is not None:
                _fail(f"{context}.omission must be null when ready")
            omission = None
            missing_gates = [
                gate for gate in required_gates if gate not in satisfied_gates
            ]
            if missing_gates:
                _fail(f"{context} is ready but lacks gates {missing_gates}")
            for artifact_id in claim_artifacts:
                if artifact_records[artifact_id]["status"] != "ready":
                    _fail(f"{context} maps an omitted artifact as ready")
        else:
            content_sha = _expect_sha256(
                record["content_sha256"],
                context=f"{context}.content_sha256",
            )
            if "COAUTH" not in satisfied_gates:
                _fail(f"{context} omission lacks a global COAUTH receipt")
            if not claims:
                _fail(f"{context} omission must own at least one artifact claim")
            for artifact_id in claim_artifacts:
                if artifact_records[artifact_id]["status"] != "omitted":
                    _fail(f"{context} cannot omit a ready artifact")
            omission = _normalize_omission(
                record["omission"],
                context=f"{context}.omission",
                forbidden_ids=forbidden_ids,
            )
            if any(
                document_id not in forbidden_documents[str(token_id)]
                for token_id in _expect_sequence(
                    omission["forbidden_token_ids"],
                    context=f"{context}.omission.forbidden_token_ids",
                )
            ):
                _fail(f"{context} omission token does not cover its document")
            artifact_reasons = {
                artifact_records[artifact_id]["omission_reason_code"]
                for artifact_id in claim_artifacts
            }
            if artifact_reasons != {omission["reason_code"]}:
                _fail(f"{context} omission reason contradicts the artifact registry")
        placements.append(
            {
                "placement_id": placement_id,
                "document_id": document_id,
                "kind": kind,
                "status": status,
                "required_gates": required_gates,
                "artifact_claims": claims,
                "content_sha256": content_sha,
                "page_location": page_location,
                "omission": omission,
            },
        )
    placements.sort(key=lambda record: str(record["placement_id"]))
    if list(records) != placements:
        _fail("reconciliation.placements is not canonically ordered")
    represented_documents = {str(placement["document_id"]) for placement in placements}
    if represented_documents != set(DOCUMENT_IDS):
        _fail("reconciliation must contain placements in main, S1, and rebuttal")
    return placements


def _normalize_reviewer_items(
    value: object,
    *,
    placements: Sequence[Mapping[str, object]],
    artifact_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    records = _expect_sequence(value, context="reconciliation.reviewer_items")
    if len(records) != _EXPECTED_REVIEWER_ITEM_COUNT:
        _fail("reconciliation must account for all 27 reviewer items")
    placement_by_id = {
        str(placement["placement_id"]): placement for placement in placements
    }
    artifact_order = {
        artifact_id: index for index, artifact_id in enumerate(artifact_ids)
    }
    normalized: list[dict[str, object]] = []
    used_response_placements: set[str] = set()
    linked_artifacts: set[str] = set()
    for index, expected_id in enumerate(REVIEWER_ITEM_ORDER):
        context = f"reconciliation.reviewer_items[{index}]"
        record = _expect_mapping(records[index], context=context)
        _expect_keys(
            record,
            {
                "reviewer_item_id",
                "response_placement_id",
                "target_placement_ids",
                "artifact_ids",
            },
            context=context,
        )
        if record["reviewer_item_id"] != expected_id:
            _fail(f"{context} is not in canonical reviewer-item order")
        response_id = _expect_token(
            record["response_placement_id"],
            context=f"{context}.response_placement_id",
        )
        response = placement_by_id.get(response_id)
        if response is None or response["kind"] != "response":
            _fail(f"{context}.response_placement_id is not a rebuttal response")
        if response_id in used_response_placements:
            _fail("a response placement is assigned to multiple reviewer items")
        used_response_placements.add(response_id)
        raw_targets = _expect_sequence(
            record["target_placement_ids"],
            context=f"{context}.target_placement_ids",
        )
        target_ids: list[str] = []
        for target_index, raw_target in enumerate(raw_targets):
            target = _expect_token(
                raw_target,
                context=f"{context}.target_placement_ids[{target_index}]",
            )
            placement = placement_by_id.get(target)
            if (
                placement is None
                or placement["document_id"] == "rebuttal"
                or target in target_ids
            ):
                _fail(f"{context}.target_placement_ids is invalid or duplicated")
            target_ids.append(target)
        target_ids.sort()
        if list(raw_targets) != target_ids:
            _fail(f"{context}.target_placement_ids is not canonical")
        raw_artifacts = _expect_sequence(
            record["artifact_ids"],
            context=f"{context}.artifact_ids",
        )
        reviewer_artifacts: list[str] = []
        for artifact_index, raw_artifact in enumerate(raw_artifacts):
            artifact = _expect_token(
                raw_artifact,
                context=f"{context}.artifact_ids[{artifact_index}]",
            )
            if artifact not in artifact_order or artifact in reviewer_artifacts:
                _fail(f"{context}.artifact_ids is invalid or duplicated")
            reviewer_artifacts.append(artifact)
        reviewer_artifacts.sort(key=artifact_order.__getitem__)
        if list(raw_artifacts) != reviewer_artifacts:
            _fail(f"{context}.artifact_ids is not canonical")
        target_artifacts = {
            str(claim["artifact_id"])
            for target in target_ids
            for claim in _expect_sequence(
                placement_by_id[target]["artifact_claims"],
                context=f"{context}.target artifact_claims",
            )
        }
        if not set(reviewer_artifacts).issubset(target_artifacts):
            _fail(f"{context}.artifact_ids are not carried by its target placements")
        linked_artifacts.update(reviewer_artifacts)
        normalized.append(
            {
                "reviewer_item_id": expected_id,
                "response_placement_id": response_id,
                "target_placement_ids": target_ids,
                "artifact_ids": reviewer_artifacts,
            },
        )
    all_response_placements = {
        str(placement["placement_id"])
        for placement in placements
        if placement["kind"] == "response"
    }
    if all_response_placements != used_response_placements:
        _fail("one or more rebuttal response placements are orphaned")
    if linked_artifacts != set(artifact_ids):
        _fail(
            "reviewer-item artifact closure failed; "
            f"missing={sorted(set(artifact_ids) - linked_artifacts)}",
        )
    return normalized


def _verify_claim_closure(
    placements: Sequence[Mapping[str, object]],
    artifact_claim_order: tuple[tuple[str, str], ...],
) -> None:
    owners: dict[tuple[str, str], str] = {}
    for placement in placements:
        placement_id = str(placement["placement_id"])
        for claim_raw in _expect_sequence(
            placement["artifact_claims"],
            context=f"placement {placement_id} artifact_claims",
        ):
            claim = _expect_mapping(claim_raw, context="artifact claim")
            pair = (str(claim["artifact_id"]), str(claim["claim_id"]))
            if pair in owners:
                _fail(
                    f"artifact claim {pair!r} is mapped by both "
                    f"{owners[pair]!r} and {placement_id!r}",
                )
            owners[pair] = placement_id
    expected = set(artifact_claim_order)
    actual = set(owners)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(f"artifact claim closure failed; missing={missing}, extra={extra}")


def _verify_placement_usage(
    placements: Sequence[Mapping[str, object]],
    reviewer_items: Sequence[Mapping[str, object]],
) -> None:
    referenced_targets = {
        str(target)
        for reviewer in reviewer_items
        for target in _expect_sequence(
            reviewer["target_placement_ids"],
            context="reviewer target placements",
        )
    }
    for placement in placements:
        if placement["kind"] == "response":
            continue
        placement_id = str(placement["placement_id"])
        claims = _expect_sequence(
            placement["artifact_claims"],
            context=f"placement {placement_id} artifact_claims",
        )
        if placement_id not in referenced_targets and not claims:
            _fail(f"placement {placement_id!r} is orphaned")


def _verify_document_blocks(
    *,
    mode: str,
    documents: Mapping[str, str],
    placements: Sequence[Mapping[str, object]],
    forbidden_tokens: Sequence[Mapping[str, object]],
) -> None:
    placements_by_document: dict[str, dict[str, Mapping[str, object]]] = {
        document_id: {} for document_id in DOCUMENT_IDS
    }
    for placement in placements:
        document_id = str(placement["document_id"])
        placement_id = str(placement["placement_id"])
        placements_by_document[document_id][placement_id] = placement
    for document_id in DOCUMENT_IDS:
        text = documents[document_id]
        blocks = _extract_blocks(document_id, text)
        expected = placements_by_document[document_id]
        if set(blocks) != set(expected):
            _fail(
                f"document {document_id} target closure failed; "
                f"missing={sorted(set(expected) - set(blocks))}, "
                f"extra={sorted(set(blocks) - set(expected))}",
            )
        outside = text
        for placement_id, block in blocks.items():
            placement = expected[placement_id]
            block_text = block.content.decode("utf-8")
            if placement["status"] != "pending" and not block_text.strip():
                _fail(f"placement {placement_id!r} has empty final content")
            unresolved = _contains_unresolved(block_text)
            if placement["status"] != "pending" and unresolved:
                _fail(f"placement {placement_id!r} contains an unresolved placeholder")
            if placement["status"] == "pending" and mode != "draft":
                _fail(f"final placement {placement_id!r} remains pending")
            declared_content_sha = placement["content_sha256"]
            if (
                declared_content_sha is not None
                and _sha256(block.content) != declared_content_sha
            ):
                _fail(f"placement {placement_id!r} content digest does not match")
            begin = _marker_line(document_id, placement_id, "BEGIN")
            end = _marker_line(document_id, placement_id, "END")
            pattern = re.compile(
                re.escape(begin) + r"\n[\s\S]*?" + re.escape(end) + r"(?:\n|$)",
            )
            outside, count = pattern.subn("", outside, count=1)
            if count != 1:
                _fail(f"document {document_id} cannot isolate target {placement_id!r}")
            if (
                mode == "final"
                and document_id != "rebuttal"
                and placement["page_location"] is None
            ):
                _fail(f"final placement {placement_id!r} lacks page/line evidence")
        if _contains_unresolved(outside):
            _fail(
                f"document {document_id} has an unresolved placeholder outside "
                "a target",
            )
    for token in forbidden_tokens:
        literal = str(token["literal"])
        for document_id in _expect_sequence(
            token["document_ids"],
            context="forbidden token document_ids",
        ):
            if literal in documents[str(document_id)]:
                _fail(
                    f"forbidden stale token {token['token_id']!r} remains in "
                    f"document {document_id}",
                )


def _preflight_reconciliation_binding(
    value: Mapping[str, object],
    *,
    artifact_registry_sha256: str,
    document_anchor_sha256: str,
    registry_release_id: object,
) -> str:
    """Verify reconciliation identity bindings before any document is opened."""
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "binding",
            "gate_ledger",
            "reviewer_items",
            "placements",
            "forbidden_tokens",
        },
        context="reconciliation",
    )
    if value["schema"] != RECONCILIATION_INPUT_SCHEMA:
        _fail("reconciliation has the wrong schema")
    mode = _expect_string(value["mode"], context="reconciliation.mode")
    if mode not in {"draft", "final"}:
        _fail("reconciliation.mode must be draft or final")
    binding = _expect_mapping(value["binding"], context="reconciliation.binding")
    _expect_keys(
        binding,
        {"release_id", "artifact_registry_sha256", "document_anchor_sha256"},
        context="reconciliation.binding",
    )
    release_id = _expect_string(
        binding["release_id"],
        context="reconciliation.binding.release_id",
    )
    if _RELEASE_ID_RE.fullmatch(release_id) is None:
        _fail("reconciliation.binding.release_id is not canonical")
    if release_id != registry_release_id:
        _fail("reconciliation release_id differs from the artifact registry")
    if binding["artifact_registry_sha256"] != artifact_registry_sha256:
        _fail(
            "reconciliation does not bind the independently anchored artifact registry",
        )
    if binding["document_anchor_sha256"] != document_anchor_sha256:
        _fail(
            "reconciliation does not bind the independently anchored document anchor",
        )
    return mode


def _normalize_reconciliation(
    value: Mapping[str, object],
    *,
    reconciliation_sha256: str,
    artifact_registry_sha256: str,
    document_anchor_sha256: str,
    registry_summary: Mapping[str, object],
    artifact_records: Mapping[str, Mapping[str, object]],
    registry_gate_ledger: Sequence[Mapping[str, str]],
    document_records: Sequence[Mapping[str, object]],
    documents: Mapping[str, str],
) -> dict[str, object]:
    mode = _preflight_reconciliation_binding(
        value,
        artifact_registry_sha256=artifact_registry_sha256,
        document_anchor_sha256=document_anchor_sha256,
        registry_release_id=registry_summary["release_id"],
    )
    binding = _expect_mapping(value["binding"], context="reconciliation.binding")
    release_id = str(binding["release_id"])
    gate_ledger = _normalize_gate_ledger(
        value["gate_ledger"],
        context="reconciliation.gate_ledger",
    )
    artifact_gate_receipts = [
        record for record in gate_ledger if record["gate"] in ARTIFACT_GATE_ORDER
    ]
    if artifact_gate_receipts != list(registry_gate_ledger):
        _fail("document gate ledger contradicts the artifact registry gate ledger")
    forbidden_tokens = _normalize_forbidden_tokens(value["forbidden_tokens"])
    artifact_claim_order = tuple(
        (spec.semantic_id, claim.claim_id)
        for spec in ARTIFACT_SPECS
        for claim in spec.claims
    )
    placements = _normalize_placements(
        value["placements"],
        mode=mode,
        artifact_records=artifact_records,
        artifact_claim_order=artifact_claim_order,
        gate_ledger=gate_ledger,
        forbidden_tokens=forbidden_tokens,
    )
    reviewer_items = _normalize_reviewer_items(
        value["reviewer_items"],
        placements=placements,
        artifact_ids=tuple(spec.semantic_id for spec in ARTIFACT_SPECS),
    )
    _verify_claim_closure(placements, artifact_claim_order)
    _verify_placement_usage(placements, reviewer_items)
    _verify_document_blocks(
        mode=mode,
        documents=documents,
        placements=placements,
        forbidden_tokens=forbidden_tokens,
    )
    counts = {
        status: sum(placement["status"] == status for placement in placements)
        for status in PLACEMENT_STATUS_ORDER
    }
    if mode == "final" and counts["pending"]:
        _fail("final reconciliation contains unresolved placements")
    return {
        "schema": DOCUMENT_RECONCILIATION_SCHEMA,
        "contract": DOCUMENT_RECONCILIATION_CONTRACT,
        "trust_model": dict(TRUST_MODEL),
        "mode": mode,
        "release_id": release_id,
        "inputs": {
            "reconciliation_sha256": reconciliation_sha256,
            "artifact_registry_sha256": artifact_registry_sha256,
            "document_anchor_sha256": document_anchor_sha256,
        },
        "artifact_registry": dict(registry_summary),
        "documents": [dict(record) for record in document_records],
        "gate_ledger": gate_ledger,
        "reviewer_items": reviewer_items,
        "placements": placements,
        "forbidden_tokens": forbidden_tokens,
        "summary": {
            "document_count": len(document_records),
            "reviewer_item_count": len(reviewer_items),
            "artifact_count": len(ARTIFACT_SPECS),
            "artifact_claim_count": len(artifact_claim_order),
            "placement_count": len(placements),
            "ready_count": counts["ready"],
            "omitted_count": counts["omitted"],
            "pending_count": counts["pending"],
        },
    }


def _document_builder_record(pinned: _PinnedFile) -> dict[str, object]:
    return {
        "member": _BUILDER_MEMBER,
        "bytes": pinned.size,
        "sha256": pinned.sha256,
    }


def _validate_input_root_paths(
    input_roots: Sequence[tuple[Path, str]],
) -> tuple[tuple[Path, str], ...]:
    validated: list[tuple[Path, str]] = []
    for root_path, context in input_roots:
        absolute_root = root_path.absolute()
        try:
            entry = os.lstat(absolute_root)
            resolved = absolute_root.resolve(strict=True)
        except OSError as error:
            _fail(f"cannot inspect {context}: {error}")
        if (
            stat.S_ISLNK(entry.st_mode)
            or not stat.S_ISDIR(entry.st_mode)
            or resolved != absolute_root
        ):
            _fail(f"{context} must be a canonical non-symlink directory")
        validated.append((absolute_root, context))
    return tuple(validated)


def _validate_destination_scope(
    destination: Path,
    *,
    input_roots: Sequence[tuple[Path, str]],
) -> None:
    absolute_destination = destination.absolute()
    for absolute_root, context in _validate_input_root_paths(input_roots):
        try:
            absolute_destination.relative_to(absolute_root)
        except ValueError:
            continue
        _fail(f"destination must be outside {context}")


def _ensure_new_destination(destination: Path) -> tuple[Path, int]:
    absolute = destination.absolute()
    if absolute.name in {"", ".", ".."} or absolute.suffix != ".json":
        _fail("destination must name a JSON file")
    parent = absolute.parent
    try:
        parent_entry = os.lstat(parent)
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect destination parent: {error}")
    if (
        not stat.S_ISDIR(parent_entry.st_mode)
        or stat.S_ISLNK(parent_entry.st_mode)
        or resolved_parent != parent
    ):
        _fail("destination parent must be a canonical non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = os.open(parent, flags)
    except OSError as error:
        _fail(f"cannot open destination parent: {error}")
    try:
        pinned_parent = os.fstat(parent_descriptor)
        if (pinned_parent.st_dev, pinned_parent.st_ino) != (
            parent_entry.st_dev,
            parent_entry.st_ino,
        ):
            _fail("destination parent changed while it was pinned")
        try:
            os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return absolute, parent_descriptor
        _fail("document reconciliation destination already exists")
    except Exception:
        os.close(parent_descriptor)
        raise


def _revalidate_destination_parent(absolute: Path, parent_descriptor: int) -> None:
    try:
        path_entry = os.lstat(absolute.parent)
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"destination parent disappeared during publication: {error}")
    pinned_parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(path_entry.st_mode)
        or stat.S_ISLNK(path_entry.st_mode)
        or resolved_parent != absolute.parent
        or (path_entry.st_dev, path_entry.st_ino)
        != (pinned_parent.st_dev, pinned_parent.st_ino)
    ):
        _fail("destination parent changed during publication")


def _verify_published_destination(
    absolute: Path,
    parent_descriptor: int,
    *,
    expected_identity: tuple[int, int],
    expected_links: int,
    raw: bytes,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        _fail(f"cannot open published reconciliation: {error}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
            or (before.st_dev, before.st_ino) != expected_identity
            or before.st_nlink != expected_links
            or before.st_size != len(raw)
        ):
            _fail("published reconciliation identity does not match staging")
        if (
            _read_descriptor(
                descriptor,
                maximum=len(raw),
                context="published reconciliation",
            )
            != raw
        ):
            _fail("published reconciliation bytes do not match")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        ):
            _fail("published reconciliation changed during readback")
        named = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != expected_identity
            or named.st_nlink != expected_links
            or stat.S_IMODE(named.st_mode) != 0o400
        ):
            _fail("published reconciliation name changed during readback")
        _revalidate_destination_parent(absolute, parent_descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(
    destination: Path,
    raw: bytes,
    *,
    revalidate: Callable[[], None],
) -> None:
    absolute, parent_descriptor = _ensure_new_destination(destination)
    staging_name = f".{absolute.name}.private-{uuid.uuid4().hex}"
    descriptor = -1
    staging_present = False
    destination_linked = False
    published = False
    staged_identity: tuple[int, int] | None = None
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(staging_name, flags, 0o600, dir_fd=parent_descriptor)
        staging_present = True
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                _fail("staged reconciliation write made no progress")
            offset += count
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        if (
            _read_descriptor(
                descriptor,
                maximum=len(raw),
                context="staged reconciliation",
            )
            != raw
        ):
            _fail("staged reconciliation bytes do not match")
        staged = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o400
            or staged.st_nlink != 1
            or staged.st_size != len(raw)
        ):
            _fail("staged reconciliation identity is invalid")
        staged_identity = (staged.st_dev, staged.st_ino)
        revalidate()
        _revalidate_destination_parent(absolute, parent_descriptor)
        try:
            os.link(
                staging_name,
                absolute.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise DocumentReconciliationError(
                "document reconciliation destination already exists",
            ) from error
        destination_linked = True
        os.fsync(parent_descriptor)
        _verify_published_destination(
            absolute,
            parent_descriptor,
            expected_identity=staged_identity,
            expected_links=2,
            raw=raw,
        )
        revalidate()
        os.unlink(staging_name, dir_fd=parent_descriptor)
        staging_present = False
        os.fsync(parent_descriptor)
        _verify_published_destination(
            absolute,
            parent_descriptor,
            expected_identity=staged_identity,
            expected_links=1,
            raw=raw,
        )
        revalidate()
        _revalidate_destination_parent(absolute, parent_descriptor)
        _verify_published_destination(
            absolute,
            parent_descriptor,
            expected_identity=staged_identity,
            expected_links=1,
            raw=raw,
        )
        revalidate()
        _revalidate_destination_parent(absolute, parent_descriptor)
        _verify_published_destination(
            absolute,
            parent_descriptor,
            expected_identity=staged_identity,
            expected_links=1,
            raw=raw,
        )
        published = True
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            try:
                if not published and destination_linked and staged_identity is not None:
                    with contextlib.suppress(FileNotFoundError):
                        destination = os.stat(
                            absolute.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        if (destination.st_dev, destination.st_ino) == staged_identity:
                            os.unlink(absolute.name, dir_fd=parent_descriptor)
                            os.fsync(parent_descriptor)
            finally:
                try:
                    if staging_present:
                        with contextlib.suppress(FileNotFoundError):
                            os.unlink(staging_name, dir_fd=parent_descriptor)
                            os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)


def _open_documents(
    root: _PinnedRoot,
    document_records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], list[_PinnedFile]]:
    texts: dict[str, str] = {}
    pins: list[_PinnedFile] = []
    try:
        for record in document_records:
            document_id = str(record["document_id"])
            pinned = _open_root_member(
                root,
                str(record["member"]),
                maximum=_MAX_DOCUMENT_BYTES,
                context=f"document {document_id}",
            )
            pins.append(pinned)
            if pinned.size != record["bytes"] or pinned.sha256 != record["sha256"]:
                _fail(f"document {document_id} differs from its independent anchor")
            texts[document_id] = _decode_document(
                pinned,
                context=f"document {document_id}",
            )
    except Exception:
        while pins:
            pins.pop().close()
        raise
    else:
        return texts, pins


def _revalidate_inputs(
    *,
    metadata: Sequence[tuple[_PinnedFile, str]],
    document_root: _PinnedRoot,
    documents: Sequence[_PinnedFile],
    document_records: Sequence[Mapping[str, object]],
) -> None:
    for pinned, context in metadata:
        _revalidate_file(pinned, context=context)
    _revalidate_root(document_root, context="document root")
    for index, (pinned, record) in enumerate(
        zip(documents, document_records, strict=True),
    ):
        _revalidate_file(pinned, context=f"document member {index}")
        current = _open_root_member(
            document_root,
            str(record["member"]),
            maximum=_MAX_DOCUMENT_BYTES,
            context=f"current document member {index}",
        )
        try:
            if (
                current.device,
                current.inode,
                current.size,
                current.mtime_ns,
                current.sha256,
            ) != (
                pinned.device,
                pinned.inode,
                pinned.size,
                pinned.mtime_ns,
                pinned.sha256,
            ):
                _fail(f"document member {index} path changed after validation")
        finally:
            current.close()


def _receipt(
    path: Path,
    manifest: Mapping[str, object],
    raw: bytes,
) -> DocumentReconciliationReceipt:
    summary = _expect_mapping(manifest["summary"], context="manifest.summary")
    return DocumentReconciliationReceipt(
        manifest_path=str(path.absolute()),
        manifest_sha256=_sha256(raw),
        mode=str(manifest["mode"]),
        placement_count=int(summary["placement_count"]),
        ready_count=int(summary["ready_count"]),
        omitted_count=int(summary["omitted_count"]),
        pending_count=int(summary["pending_count"]),
    )


def build_document_reconciliation(
    reconciliation_path: Path,
    artifact_registry_path: Path,
    renderer_root_path: Path,
    rendered_output_root_path: Path,
    document_anchor_path: Path,
    document_root_path: Path,
    destination: Path,
    *,
    expected_reconciliation_sha256: str,
    expected_artifact_registry_sha256: str,
    expected_document_anchor_sha256: str,
) -> DocumentReconciliationReceipt:
    """Validate anchored metadata/documents and publish a no-replace manifest."""
    # Fail on a pre-existing destination before any final document is opened.
    _, parent_descriptor = _ensure_new_destination(destination)
    os.close(parent_descriptor)
    reconciliation: _PinnedFile | None = None
    artifact_registry: _PinnedFile | None = None
    document_anchor: _PinnedFile | None = None
    native_builder: _PinnedFile | None = None
    document_builder: _PinnedFile | None = None
    renderer_root: _PinnedRoot | None = None
    rendered_output_root: _PinnedRoot | None = None
    document_root: _PinnedRoot | None = None
    document_pins: list[_PinnedFile] = []
    try:
        reconciliation = _pin_file(
            reconciliation_path,
            maximum=_MAX_METADATA_BYTES,
            context="reconciliation input",
        )
        artifact_registry = _pin_file(
            artifact_registry_path,
            maximum=_MAX_METADATA_BYTES,
            context="artifact registry",
        )
        document_anchor = _pin_file(
            document_anchor_path,
            maximum=_MAX_METADATA_BYTES,
            context="document anchor",
        )
        native_builder = _pin_file(
            Path(_artifact_registry.__file__),
            maximum=_MAX_METADATA_BYTES,
            context="live artifact registry builder",
        )
        document_builder = _pin_file(
            Path(__file__),
            maximum=_MAX_METADATA_BYTES,
            context="live document reconciliation builder",
        )
        expected_reconciliation = _expect_sha256(
            expected_reconciliation_sha256,
            context="expected_reconciliation_sha256",
        )
        expected_registry = _expect_sha256(
            expected_artifact_registry_sha256,
            context="expected_artifact_registry_sha256",
        )
        expected_anchor = _expect_sha256(
            expected_document_anchor_sha256,
            context="expected_document_anchor_sha256",
        )
        if artifact_registry.sha256 != expected_registry:
            _fail("artifact registry SHA-256 does not match its independent anchor")
        if document_anchor.sha256 != expected_anchor:
            _fail("document anchor SHA-256 does not match its independent anchor")
        if reconciliation.sha256 != expected_reconciliation:
            _fail("reconciliation input SHA-256 does not match its independent anchor")
        input_roots = (
            (renderer_root_path, "renderer root"),
            (rendered_output_root_path, "rendered-output root"),
            (document_root_path, "document root"),
        )
        _validate_destination_scope(destination, input_roots=input_roots)
        renderer_root = _pin_root(renderer_root_path, context="renderer root")
        rendered_output_root = _pin_root(
            rendered_output_root_path,
            context="rendered-output root",
        )
        native_receipt = _validate_native_artifact_registry(
            artifact_registry.path,
            renderer_root_path,
            rendered_output_root_path,
            expected_sha256=artifact_registry.sha256,
        )
        _revalidate_file(artifact_registry, context="artifact registry")
        _revalidate_file(native_builder, context="live artifact registry builder")
        _revalidate_root(renderer_root, context="renderer root")
        _revalidate_root(rendered_output_root, context="rendered-output root")
        registry_value = _parse_canonical_json(
            artifact_registry,
            context="artifact registry",
        )
        registry_summary, artifact_records, registry_gate_ledger = _normalize_registry(
            registry_value,
            expected_builder=_pinned_script_record(
                native_builder,
                member=_artifact_registry._BUILDER_SCRIPT_MEMBER,
            ),
        )
        if (
            native_receipt.ready_count != registry_summary["ready_count"]
            or native_receipt.omitted_count != registry_summary["omitted_count"]
        ):
            _fail("native artifact registry summary contradicts parsed metadata")
        anchor_value = _parse_canonical_json(document_anchor, context="document anchor")
        document_records = _normalize_document_anchor(anchor_value)
        reconciliation_value = _parse_canonical_json(
            reconciliation,
            context="reconciliation input",
        )
        _preflight_reconciliation_binding(
            reconciliation_value,
            artifact_registry_sha256=artifact_registry.sha256,
            document_anchor_sha256=document_anchor.sha256,
            registry_release_id=registry_summary["release_id"],
        )
        for pinned, context in (
            (reconciliation, "reconciliation input"),
            (artifact_registry, "artifact registry"),
            (document_anchor, "document anchor"),
            (native_builder, "live artifact registry builder"),
            (document_builder, "live document reconciliation builder"),
        ):
            _revalidate_file(pinned, context=context)
        _revalidate_root(renderer_root, context="renderer root")
        _revalidate_root(rendered_output_root, context="rendered-output root")
        # Only after all three independent metadata anchors and their structures pass
        # is any final document path opened.
        document_root = _pin_root(document_root_path, context="document root")
        documents, document_pins = _open_documents(document_root, document_records)
        payload = _normalize_reconciliation(
            reconciliation_value,
            reconciliation_sha256=reconciliation.sha256,
            artifact_registry_sha256=artifact_registry.sha256,
            document_anchor_sha256=document_anchor.sha256,
            registry_summary=registry_summary,
            artifact_records=artifact_records,
            registry_gate_ledger=registry_gate_ledger,
            document_records=document_records,
            documents=documents,
        )
        payload["builder"] = _document_builder_record(document_builder)
        manifest = {
            **payload,
            "manifest_payload_sha256": _sha256(_canonical_json(payload)),
        }
        raw = _canonical_json(manifest) + b"\n"

        def revalidate() -> None:
            if (
                document_root is None
                or renderer_root is None
                or rendered_output_root is None
            ):
                _fail("one or more input roots were not pinned")
            _revalidate_root(renderer_root, context="renderer root")
            _revalidate_root(rendered_output_root, context="rendered-output root")
            _revalidate_inputs(
                metadata=(
                    (reconciliation, "reconciliation input"),
                    (artifact_registry, "artifact registry"),
                    (document_anchor, "document anchor"),
                    (native_builder, "live artifact registry builder"),
                    (document_builder, "live document reconciliation builder"),
                ),
                document_root=document_root,
                documents=document_pins,
                document_records=document_records,
            )
            _validate_destination_scope(destination, input_roots=input_roots)
            boundary_receipt = _validate_native_artifact_registry(
                artifact_registry.path,
                renderer_root_path,
                rendered_output_root_path,
                expected_sha256=artifact_registry.sha256,
            )
            if (
                boundary_receipt.ready_count != registry_summary["ready_count"]
                or boundary_receipt.omitted_count != registry_summary["omitted_count"]
            ):
                _fail("native artifact registry summary changed")
            _validate_destination_scope(destination, input_roots=input_roots)
            _revalidate_root(renderer_root, context="renderer root")
            _revalidate_root(rendered_output_root, context="rendered-output root")
            _revalidate_inputs(
                metadata=(
                    (reconciliation, "reconciliation input"),
                    (artifact_registry, "artifact registry"),
                    (document_anchor, "document anchor"),
                    (native_builder, "live artifact registry builder"),
                    (document_builder, "live document reconciliation builder"),
                ),
                document_root=document_root,
                documents=document_pins,
                document_records=document_records,
            )

        revalidate()
        _publish_no_replace(destination, raw, revalidate=revalidate)
        return _receipt(destination, manifest, raw)
    finally:
        for pinned in document_pins:
            pinned.close()
        if document_root is not None:
            document_root.close()
        if rendered_output_root is not None:
            rendered_output_root.close()
        if renderer_root is not None:
            renderer_root.close()
        if document_builder is not None:
            document_builder.close()
        if native_builder is not None:
            native_builder.close()
        if document_anchor is not None:
            document_anchor.close()
        if artifact_registry is not None:
            artifact_registry.close()
        if reconciliation is not None:
            reconciliation.close()


def validate_document_reconciliation(
    manifest_path: Path,
    artifact_registry_path: Path,
    renderer_root_path: Path,
    rendered_output_root_path: Path,
    document_anchor_path: Path,
    document_root_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_artifact_registry_sha256: str,
    expected_document_anchor_sha256: str,
) -> DocumentReconciliationReceipt:
    """Validate an independently anchored reconciliation and all text bindings."""
    manifest_file: _PinnedFile | None = None
    artifact_registry: _PinnedFile | None = None
    document_anchor: _PinnedFile | None = None
    native_builder: _PinnedFile | None = None
    document_builder: _PinnedFile | None = None
    renderer_root: _PinnedRoot | None = None
    rendered_output_root: _PinnedRoot | None = None
    document_root: _PinnedRoot | None = None
    document_pins: list[_PinnedFile] = []
    try:
        manifest_file = _pin_file(
            manifest_path,
            maximum=_MAX_METADATA_BYTES,
            context="document reconciliation manifest",
        )
        artifact_registry = _pin_file(
            artifact_registry_path,
            maximum=_MAX_METADATA_BYTES,
            context="artifact registry",
        )
        document_anchor = _pin_file(
            document_anchor_path,
            maximum=_MAX_METADATA_BYTES,
            context="document anchor",
        )
        native_builder = _pin_file(
            Path(_artifact_registry.__file__),
            maximum=_MAX_METADATA_BYTES,
            context="live artifact registry builder",
        )
        document_builder = _pin_file(
            Path(__file__),
            maximum=_MAX_METADATA_BYTES,
            context="live document reconciliation builder",
        )
        if manifest_file.sha256 != _expect_sha256(
            expected_manifest_sha256,
            context="expected_manifest_sha256",
        ):
            _fail(
                "document reconciliation SHA-256 does not match its independent anchor",
            )
        if artifact_registry.sha256 != _expect_sha256(
            expected_artifact_registry_sha256,
            context="expected_artifact_registry_sha256",
        ):
            _fail("artifact registry SHA-256 does not match its independent anchor")
        if document_anchor.sha256 != _expect_sha256(
            expected_document_anchor_sha256,
            context="expected_document_anchor_sha256",
        ):
            _fail("document anchor SHA-256 does not match its independent anchor")
        input_roots = (
            (renderer_root_path, "renderer root"),
            (rendered_output_root_path, "rendered-output root"),
            (document_root_path, "document root"),
        )
        _validate_input_root_paths(input_roots)
        renderer_root = _pin_root(renderer_root_path, context="renderer root")
        rendered_output_root = _pin_root(
            rendered_output_root_path,
            context="rendered-output root",
        )
        native_receipt = _validate_native_artifact_registry(
            artifact_registry.path,
            renderer_root_path,
            rendered_output_root_path,
            expected_sha256=artifact_registry.sha256,
        )
        _revalidate_file(artifact_registry, context="artifact registry")
        _revalidate_file(native_builder, context="live artifact registry builder")
        _revalidate_root(renderer_root, context="renderer root")
        _revalidate_root(rendered_output_root, context="rendered-output root")
        registry_value = _parse_canonical_json(
            artifact_registry,
            context="artifact registry",
        )
        registry_summary, artifact_records, registry_gate_ledger = _normalize_registry(
            registry_value,
            expected_builder=_pinned_script_record(
                native_builder,
                member=_artifact_registry._BUILDER_SCRIPT_MEMBER,
            ),
        )
        if (
            native_receipt.ready_count != registry_summary["ready_count"]
            or native_receipt.omitted_count != registry_summary["omitted_count"]
        ):
            _fail("native artifact registry summary contradicts parsed metadata")
        anchor_value = _parse_canonical_json(document_anchor, context="document anchor")
        document_records = _normalize_document_anchor(anchor_value)
        manifest = _parse_canonical_json(
            manifest_file,
            context="document reconciliation manifest",
        )
        expected_keys = {
            "schema",
            "contract",
            "trust_model",
            "mode",
            "release_id",
            "inputs",
            "artifact_registry",
            "documents",
            "gate_ledger",
            "reviewer_items",
            "placements",
            "forbidden_tokens",
            "summary",
            "builder",
            "manifest_payload_sha256",
        }
        _expect_keys(
            manifest,
            expected_keys,
            context="document reconciliation manifest",
        )
        if manifest["schema"] != DOCUMENT_RECONCILIATION_SCHEMA:
            _fail("document reconciliation manifest has the wrong schema")
        if manifest["contract"] != DOCUMENT_RECONCILIATION_CONTRACT:
            _fail("document reconciliation manifest has the wrong contract")
        if manifest["trust_model"] != TRUST_MODEL:
            _fail("document reconciliation manifest changes the trust model")
        payload = dict(manifest)
        declared_payload_sha = _expect_sha256(
            payload.pop("manifest_payload_sha256"),
            context="document reconciliation manifest.manifest_payload_sha256",
        )
        if _sha256(_canonical_json(payload)) != declared_payload_sha:
            _fail("document reconciliation manifest payload digest does not match")
        if manifest["builder"] != _document_builder_record(document_builder):
            _fail("document reconciliation manifest does not bind the live builder")
        inputs = _expect_mapping(manifest["inputs"], context="manifest.inputs")
        _expect_keys(
            inputs,
            {
                "reconciliation_sha256",
                "artifact_registry_sha256",
                "document_anchor_sha256",
            },
            context="manifest.inputs",
        )
        if inputs["artifact_registry_sha256"] != artifact_registry.sha256:
            _fail("manifest does not bind the independently anchored artifact registry")
        if inputs["document_anchor_sha256"] != document_anchor.sha256:
            _fail("manifest does not bind the independently anchored document anchor")
        if manifest["artifact_registry"] != registry_summary:
            _fail("manifest artifact-registry summary drifted")
        if manifest["documents"] != list(document_records):
            _fail("manifest document inventory drifted")
        synthetic_input = {
            "schema": RECONCILIATION_INPUT_SCHEMA,
            "mode": manifest["mode"],
            "binding": {
                "release_id": manifest["release_id"],
                "artifact_registry_sha256": artifact_registry.sha256,
                "document_anchor_sha256": document_anchor.sha256,
            },
            "gate_ledger": manifest["gate_ledger"],
            "reviewer_items": manifest["reviewer_items"],
            "placements": manifest["placements"],
            "forbidden_tokens": manifest["forbidden_tokens"],
        }
        _preflight_reconciliation_binding(
            synthetic_input,
            artifact_registry_sha256=artifact_registry.sha256,
            document_anchor_sha256=document_anchor.sha256,
            registry_release_id=registry_summary["release_id"],
        )
        for pinned, context in (
            (manifest_file, "document reconciliation manifest"),
            (artifact_registry, "artifact registry"),
            (document_anchor, "document anchor"),
            (native_builder, "live artifact registry builder"),
            (document_builder, "live document reconciliation builder"),
        ):
            _revalidate_file(pinned, context=context)
        _revalidate_root(renderer_root, context="renderer root")
        _revalidate_root(rendered_output_root, context="rendered-output root")
        document_root = _pin_root(document_root_path, context="document root")
        documents, document_pins = _open_documents(document_root, document_records)
        normalized = _normalize_reconciliation(
            synthetic_input,
            reconciliation_sha256=_expect_sha256(
                inputs["reconciliation_sha256"],
                context="manifest.inputs.reconciliation_sha256",
            ),
            artifact_registry_sha256=artifact_registry.sha256,
            document_anchor_sha256=document_anchor.sha256,
            registry_summary=registry_summary,
            artifact_records=artifact_records,
            registry_gate_ledger=registry_gate_ledger,
            document_records=document_records,
            documents=documents,
        )
        for key, expected_value in normalized.items():
            if manifest[key] != expected_value:
                _fail(f"manifest field {key!r} is not canonical")
        _revalidate_inputs(
            metadata=(
                (manifest_file, "document reconciliation manifest"),
                (artifact_registry, "artifact registry"),
                (document_anchor, "document anchor"),
                (native_builder, "live artifact registry builder"),
                (document_builder, "live document reconciliation builder"),
            ),
            document_root=document_root,
            documents=document_pins,
            document_records=document_records,
        )
        _revalidate_root(renderer_root, context="renderer root")
        _revalidate_root(rendered_output_root, context="rendered-output root")
        _validate_input_root_paths(input_roots)
        final_receipt = _validate_native_artifact_registry(
            artifact_registry.path,
            renderer_root_path,
            rendered_output_root_path,
            expected_sha256=artifact_registry.sha256,
        )
        if (
            final_receipt.ready_count != registry_summary["ready_count"]
            or final_receipt.omitted_count != registry_summary["omitted_count"]
        ):
            _fail("native artifact registry summary changed")
        _validate_input_root_paths(input_roots)
        _revalidate_root(renderer_root, context="renderer root")
        _revalidate_root(rendered_output_root, context="rendered-output root")
        _revalidate_inputs(
            metadata=(
                (manifest_file, "document reconciliation manifest"),
                (artifact_registry, "artifact registry"),
                (document_anchor, "document anchor"),
                (native_builder, "live artifact registry builder"),
                (document_builder, "live document reconciliation builder"),
            ),
            document_root=document_root,
            documents=document_pins,
            document_records=document_records,
        )
        return _receipt(manifest_path, manifest, manifest_file.raw)
    finally:
        for pinned in document_pins:
            pinned.close()
        if document_root is not None:
            document_root.close()
        if rendered_output_root is not None:
            rendered_output_root.close()
        if renderer_root is not None:
            renderer_root.close()
        if document_builder is not None:
            document_builder.close()
        if native_builder is not None:
            native_builder.close()
        if document_anchor is not None:
            document_anchor.close()
        if artifact_registry is not None:
            artifact_registry.close()
        if manifest_file is not None:
            manifest_file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="publish a no-replace reconciliation")
    build.add_argument("--reconciliation", type=Path, required=True)
    build.add_argument("--artifact-registry", type=Path, required=True)
    build.add_argument("--renderer-root", type=Path, required=True)
    build.add_argument("--rendered-output-root", type=Path, required=True)
    build.add_argument("--document-anchor", type=Path, required=True)
    build.add_argument("--document-root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--expected-reconciliation-sha256", required=True)
    build.add_argument("--expected-artifact-registry-sha256", required=True)
    build.add_argument("--expected-document-anchor-sha256", required=True)
    validate = subparsers.add_parser("validate", help="validate a reconciliation")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--artifact-registry", type=Path, required=True)
    validate.add_argument("--renderer-root", type=Path, required=True)
    validate.add_argument("--rendered-output-root", type=Path, required=True)
    validate.add_argument("--document-anchor", type=Path, required=True)
    validate.add_argument("--document-root", type=Path, required=True)
    validate.add_argument("--expected-sha256", required=True)
    validate.add_argument("--expected-artifact-registry-sha256", required=True)
    validate.add_argument("--expected-document-anchor-sha256", required=True)
    return parser


def main() -> None:
    """Run the document-reconciliation command-line interface."""
    args = _parser().parse_args()
    if args.command == "build":
        receipt = build_document_reconciliation(
            args.reconciliation,
            args.artifact_registry,
            args.renderer_root,
            args.rendered_output_root,
            args.document_anchor,
            args.document_root,
            args.out,
            expected_reconciliation_sha256=args.expected_reconciliation_sha256,
            expected_artifact_registry_sha256=(args.expected_artifact_registry_sha256),
            expected_document_anchor_sha256=args.expected_document_anchor_sha256,
        )
    else:
        receipt = validate_document_reconciliation(
            args.manifest,
            args.artifact_registry,
            args.renderer_root,
            args.rendered_output_root,
            args.document_anchor,
            args.document_root,
            expected_manifest_sha256=args.expected_sha256,
            expected_artifact_registry_sha256=(args.expected_artifact_registry_sha256),
            expected_document_anchor_sha256=args.expected_document_anchor_sha256,
        )
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == "__main__":
    main()

"""Build and verify the immutable DIALECT revision public release.

This module is a packaging boundary, not a scientific analysis.  It consumes only
independently SHA-256-anchored manifests plus descriptor-pinned opaque files.  The
native source-data, artifact-registry, release-evidence, and document-reconciliation
validators remain authoritative for their own formats.  Association rows are never
decoded or interpreted here.

The public archive is deliberately distinct from a journal submission package.  It
never contains rebuttal bytes, raw gate receipts, private evidence, or dependency
payloads whose provenance record does not explicitly permit redistribution.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, NoReturn
from urllib.parse import urlsplit

from analysis import build_tcga_revision_artifact_registry as artifact_registry
from analysis import (
    build_tcga_revision_document_reconciliation as document_reconciliation,
)
from analysis import (
    build_tcga_revision_k500_authority_projection as k500_authority_projection,
)
from analysis import build_tcga_revision_release_evidence as release_evidence
from analysis import build_tcga_revision_source_data as source_data

# Detailed validation errors are part of this mechanical boundary's contract.
# ruff: noqa: EM101, EM102, TRY003, TRY301

PLAN_SCHEMA: Final = "dialect-revision-public-release-plan-v1"
PLAN_CONTRACT: Final = "anchored-explicit-allowlist-public-release-v1"
MANIFEST_SCHEMA: Final = "dialect-revision-public-release-manifest-v1"
MANIFEST_CONTRACT: Final = "deterministic-ustar-member-closure-v1"
RECEIPT_SCHEMA: Final = "dialect-revision-public-release-receipt-v1"
READBACK_SCHEMA: Final = "dialect-revision-public-portal-readback-v1"

BUILDER_MEMBER: Final = "analysis/build_tcga_revision_public_release.py"
SOURCE_COMMIT_A: Final = "3ad05c5ad5ddd03d7922343bafa5a2748cddb20d"
GENERATED_VERSION_SHA256: Final = (
    "7da36118d9f8d662fb369b6bea4eba4d3e69491d49a57cef129643f47f8ceee7"
)
GENERATED_VERSION_MEMBER: Final = "src/dialect/_version.py"
EXECUTION_SNAPSHOT_SHA256: Final = (
    "cf365e25e4aac18f937ce669a59923bb1d0650be6e1c4ad7fe5ccae7567373bb"
)
COMPLETION_ATTESTATION_SCHEMA_VERSION: Final = "1.1.0"
COMPLETION_ATTESTATION_TYPE: Final = "tcga-revision-k500-root-completion"
COMPLETION_ATTESTOR_MEMBER: Final = "research/notes/attest_k500_completion.py"
COMPLETION_ATTESTOR_BYTES: Final = 57_923
COMPLETION_ATTESTOR_SHA256: Final = (
    "238a7131a0e3aeb41928939840476d333d5ccfb19a081c19decdb6ea2a4d9de2"
)
EXECUTION_PATH_COUNT: Final = 38
SOURCE_DATA_FILE_COUNT: Final = 35
SOURCE_DATA_COHORT_COUNT: Final = 32
RESTRICTED_EXECUTION_PATH: Final = "external/mutsig2cv_octave_dialect.patch"
RESTRICTED_EXECUTION_DEPENDENCY_ID: Final = "mutsig2cv-patch-1e0aa209"
RESTRICTED_EXECUTION_PATHS: Final = frozenset({RESTRICTED_EXECUTION_PATH})

REQUIRED_CODE_PATHS: Final = frozenset(
    {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "analysis/build_tcga_revision_source_data.py",
        "analysis/build_tcga_revision_artifact_registry.py",
        "analysis/build_tcga_revision_release_evidence.py",
        "analysis/build_tcga_revision_document_reconciliation.py",
        "analysis/build_tcga_revision_k500_authority_projection.py",
        BUILDER_MEMBER,
    },
)
FORBIDDEN_CODE_PREFIXES: Final = (
    ".git/",
    "atlas/",
    "data/",
    "output/",
    "provenance/dependencies/",
    "research/",
)
FORBIDDEN_PUBLIC_PREFIXES: Final = (
    ".git/",
    "gate-receipts/",
    "private/",
    "raw/",
    "research/",
    "rebuttal/",
    "submission/",
    "documents/",
    "manuscript/",
    "supporting-information/",
)

DEPENDENCY_IDS: Final = (
    "atlas-code-v2.3.1-563ae0f",
    "atlas-k100-v1.0.0-0ef212a",
    "cbase-v1.2-dialect-fork",
    "dig-pancan-artifact-4402b76e",
    "digdriver-source-5bb565a",
    "discover-0.9.6-a46d99f",
    "megsa-source-9e75152f",
    "msk-chord-2024-eb53cc4",
    "msk-impact-50k-2026-eb53cc4",
    "mutsig2cv-patch-1e0aa209",
    "mutsig2cv-source-0109e27",
    "oncokb-cancer-gene-list-2024-12-19-56cea460",
    "tcga-datahub-64392ef-32-study",
)
INCLUDED_DEPENDENCY_ID: Final = "cbase-v1.2-dialect-fork"
CBASE_RECORD_MEMBER: Final = "provenance/dependencies/cbase-v1.2-dialect-fork.json"
DEPENDENCY_METADATA_MEMBERS: Final = tuple(
    sorted(
        (
            "README.md",
            "record.schema.json",
            *(f"{dependency_id}.json" for dependency_id in DEPENDENCY_IDS),
        ),
    ),
)
CBASE_RELEASE_MEMBERS: Final = (
    "external/CBaSE/CBaSE_params_v1.2.py",
    "external/CBaSE/CBaSE_qvals_v1.2.py",
    "external/CBaSE/NOTICE",
    "external/CBaSE/cbase_cohort_size.py",
)
CBASE_RELEASE_FILE_ROLES: Final[dict[str, str]] = {
    "external/CBaSE/CBaSE_params_v1.2.py": (
        "historical-upstream-derived-fork-with-dialect-modifications"
    ),
    "external/CBaSE/CBaSE_qvals_v1.2.py": (
        "historical-upstream-derived-fork-with-dialect-modifications"
    ),
    "external/CBaSE/NOTICE": "dialect-authored-provenance-notice",
    "external/CBaSE/cbase_cohort_size.py": "dialect-authored-helper",
}
CBASE_RELEASE_FILE_LICENSES: Final[dict[str, tuple[str, ...]]] = {
    "external/CBaSE/CBaSE_params_v1.2.py": (
        "LicenseRef-CBaSE-Public-Domain",
        "BSD-3-Clause",
    ),
    "external/CBaSE/CBaSE_qvals_v1.2.py": (
        "LicenseRef-CBaSE-Public-Domain",
        "BSD-3-Clause",
    ),
    "external/CBaSE/NOTICE": ("BSD-3-Clause",),
    "external/CBaSE/cbase_cohort_size.py": ("BSD-3-Clause",),
}
CBASE_UPSTREAM_LICENSE_ID: Final = "LicenseRef-CBaSE-Public-Domain"
CBASE_DIALECT_LICENSE_ID: Final = "BSD-3-Clause"
CBASE_COMPOSITE_LICENSE_ID: Final = "LicenseRef-CBaSE-Public-Domain AND BSD-3-Clause"
CBASE_DIALECT_LICENSE_SHA256: Final = (
    "3c900e08a49b06523f496c107d6d1548d3da2e45fba6972e628cde67f2251d16"
)
CBASE_OFFICIAL_ARCHIVE_ROLE: Final = (
    "current-v1.2-comparison-reference-not-exact-parent"
)
CBASE_SCOPE: Final = (
    "Two historical CBaSE v1.2-derived scripts with preserved Public Domain "
    "headers and BSD-3-Clause DIALECT modifications, plus a BSD-3-Clause "
    "DIALECT helper and provenance notice; the current official archive and "
    "auxiliary data are excluded."
)
CBASE_RELEASE_NAMESPACE: Final = "external/CBaSE/"
CBASE_OFFICIAL_ARCHIVE_PUBLIC_MEMBER: Final = "external/CBaSE/CBaSE_v1.2.zip"

ARCHIVE_MANIFEST_MEMBER: Final = "release_manifest.json"
ARCHIVE_CHECKSUM_MEMBER: Final = "SHA256SUMS"
PUBLIC_PLAN_MEMBER: Final = "evidence/public_release_plan.json"
PUBLIC_REGISTRY_MEMBER: Final = "evidence/artifact_registry.json"
PUBLIC_CLOSURE_MEMBER: Final = "evidence/release_evidence.json"
PUBLIC_PROJECTION_MEMBER: Final = "evidence/k500_authority_projection.json"
ARCHIVE_FORMAT: Final = "posix-ustar-uncompressed-v1"
ARCHIVE_FILE_MODE: Final = 0o444
PRIVATE_STAGE_MARKER: Final = ".private-"
MAX_METADATA_BYTES: Final = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS: Final = 4096
MAX_ARCHIVE_METADATA_MEMBER_BYTES: Final = 32 * 1024 * 1024
READ_CHUNK_BYTES: Final = 1024 * 1024
MAX_GIT_EXECUTABLE_BYTES: Final = 256 * 1024 * 1024
MAX_GIT_BLOB_BYTES: Final = 16 * 1024 * 1024
MAX_PUBLIC_CODE_BYTES: Final = 128 * 1024 * 1024
EXACT_ROOT_SCAN_CAP: Final = 64
MAX_PUBLICATION_DIRECTORY_ENTRIES: Final = 1024
PUBLICATION_DIRECTORY_SCAN_CAP: Final = MAX_PUBLICATION_DIRECTORY_ENTRIES + 1
MAX_GIT_CONTROL_STDOUT_BYTES: Final = 4096
MAX_GIT_STDOUT_BYTES: Final = MAX_GIT_BLOB_BYTES
MAX_GIT_STDERR_BYTES: Final = 1024 * 1024
MAX_GIT_SECONDS: Final = 30.0
TAR_BLOCK_BYTES: Final = tarfile.BLOCKSIZE
TAR_RECORD_BYTES: Final = tarfile.RECORDSIZE
GIT_EXECUTABLE: Final = Path("/usr/bin/git")
_GIT_CONFIG_OVERRIDES: Final = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.alternateRefsCommand=",
    "-c",
    "submodule.recurse=false",
    "-c",
    "fetch.recurseSubmodules=false",
    "-c",
    "core.quotePath=false",
)
_FILE_READ_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)

TRUST_MODEL: Final = {
    "plan": "canonical JSON whose SHA-256 is supplied independently",
    "source_data": "native validator; scientific CSV bytes remain opaque",
    "artifact_registry": "native validator and exact ready-output closure",
    "release_evidence": "native validator; gate and source bytes remain opaque",
    "k500_authority": (
        "native anchored projection validator; run and result bytes remain opaque"
    ),
    "documents": (
        "native source-text reconciliation gates claims; v1 packages zero document "
        "bytes until a separate rendered-document and visual-QA contract exists"
    ),
    "dependencies": "exact 15-file ledger and deny-by-default redistribution",
    "git": (
        "source-A ancestry, tag-bound release B, and 38-path byte identity; "
        "restricted execution paths are verified but not packaged"
    ),
    "submission": "not authorized or produced by this public-release builder",
}

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE: Final = re.compile(r"[0-9a-f]{40}")
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_VERSION_RE: Final = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,63}")
_TAG_RE: Final = re.compile(r"[0-9A-Za-z][0-9A-Za-z._/-]{0,127}")
_DOI_RE: Final = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")


class PublicReleaseError(ValueError):
    """Raised when a public release boundary cannot be proven."""


class _RenameNotPerformedError(PublicReleaseError):
    """Report a no-replace rename that definitely did not rename its source."""


@dataclass(frozen=True, slots=True)
class PlanAuditReceipt:
    """Describe one independently anchored plan without opening release inputs."""

    plan_path: str
    plan_sha256: str
    mode: str
    ready_to_publish: bool
    pending_count: int


@dataclass(frozen=True, slots=True)
class PublicReleaseReceipt:
    """Describe a mechanically published archive and external receipt."""

    archive_path: str
    archive_sha256: str
    archive_bytes: int
    manifest_sha256: str
    receipt_path: str
    receipt_sha256: str
    member_count: int


@dataclass(frozen=True, slots=True)
class PortalReadbackReceipt:
    """Describe an independently hash-checked downloaded archive."""

    downloaded_archive_path: str
    archive_sha256: str
    archive_bytes: int
    manifest_sha256: str
    release_id: str
    destination_path: str
    destination_sha256: str


@dataclass(frozen=True, slots=True)
class PublicReleaseBuildConfig:
    """Paths and the single independent trust anchor for one public build."""

    plan_path: Path
    source_data_root: Path
    artifact_registry_path: Path
    release_evidence_path: Path
    renderer_root: Path
    rendered_output_root: Path
    gate_receipt_root: Path
    evidence_source_root: Path
    document_reconciliation_path: Path
    document_anchor_path: Path
    document_root: Path
    k500_authority_projection_path: Path
    expected_k500_authority_projection_sha256: str
    dependency_root: Path
    repository_root: Path
    destination_archive: Path
    destination_receipt: Path
    expected_plan_sha256: str


@dataclass(slots=True)
class _PinnedRoot:
    path: Path
    descriptor: int = field(repr=False)
    device: int
    inode: int

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(slots=True)
class _PinnedFile:
    path: Path | None
    root: _PinnedRoot | None
    member: str | None
    descriptor: int = field(repr=False)
    device: int
    inode: int
    link_count: int
    mode: int
    size_bytes: int
    modified_ns: int
    sha256: str

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(frozen=True, slots=True)
class _MemoryEntry:
    member: str
    raw: bytes = field(repr=False)
    role: str
    origin: Mapping[str, object]

    @property
    def size_bytes(self) -> int:
        return len(self.raw)

    @property
    def sha256(self) -> str:
        return _sha256(self.raw)


@dataclass(frozen=True, slots=True)
class _PinnedEntry:
    member: str
    pinned: _PinnedFile = field(repr=False)
    role: str
    origin: Mapping[str, object]

    @property
    def size_bytes(self) -> int:
        return self.pinned.size_bytes

    @property
    def sha256(self) -> str:
        return self.pinned.sha256


_ArchiveEntry = _MemoryEntry | _PinnedEntry


@dataclass(slots=True)
class _PreparedRelease:
    config: PublicReleaseBuildConfig
    plan_file: _PinnedFile
    metadata_files: tuple[_PinnedFile, ...]
    roots: tuple[_PinnedRoot, ...]
    member_files: tuple[_PinnedFile, ...]
    plan: dict[str, object]
    entries: tuple[_ArchiveEntry, ...]
    manifest_raw: bytes
    checksums_raw: bytes
    native_receipts: tuple[object, object, object, object, object]
    dependency_inventory_sha256: str
    dependency_root: _PinnedRoot
    repository_root: _PinnedRoot
    git_executable: _PinnedFile

    def close(self) -> None:
        for pinned in self.member_files:
            pinned.close()
        for root in self.roots:
            root.close()
        for pinned in self.metadata_files:
            pinned.close()
        self.plan_file.close()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PublicReleaseError(f"metadata contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise PublicReleaseError(f"metadata contains non-finite value {value!r}")


def _parse_canonical(raw: bytes, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise PublicReleaseError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PublicReleaseError(f"{context} must be a JSON object")
    if raw != _canonical_json(value) + b"\n":
        raise PublicReleaseError(f"{context} is not canonical JSON with one newline")
    return value


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PublicReleaseError(f"{context} must be an object")
    return value


def _expect_sequence(value: object, *, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PublicReleaseError(f"{context} must be an array")
    return value


def _expect_keys(
    value: Mapping[str, object],
    keys: set[str],
    *,
    context: str,
) -> None:
    if set(value) != keys:
        raise PublicReleaseError(
            f"{context} has unexpected keys: expected={sorted(keys)!r}, "
            f"actual={sorted(value)!r}",
        )


def _expect_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicReleaseError(f"{context} must be a nonempty string")
    return value


def _expect_optional_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, context=context)


def _expect_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublicReleaseError(f"{context} must be an integer >= {minimum}")
    return value


def _expect_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PublicReleaseError(f"{context} must be a lowercase SHA-256")
    return value


def _expect_git_sha(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise PublicReleaseError(f"{context} must be a lowercase Git object ID")
    return value


def _expect_token(value: object, *, context: str) -> str:
    token = _expect_string(value, context=context)
    if _TOKEN_RE.fullmatch(token) is None:
        raise PublicReleaseError(f"{context} is not a canonical token")
    return token


def _canonical_member(value: object, *, context: str) -> str:
    member = _expect_string(value, context=context)
    if (
        not member.isascii()
        or "\\" in member
        or any(ord(character) < 32 or ord(character) == 127 for character in member)
    ):
        raise PublicReleaseError(f"{context} is not a canonical ASCII POSIX member")
    path = PurePosixPath(member)
    if path.is_absolute() or path.as_posix() != member:
        raise PublicReleaseError(f"{context} is not a canonical relative member")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PublicReleaseError(f"{context} escapes its declared root")
    encoded = member.encode("ascii")
    representable = len(encoded) <= tarfile.LENGTH_NAME
    if not representable:
        for split_at, character in enumerate(member):
            if character != "/":
                continue
            prefix = member[:split_at].encode("ascii")
            name = member[split_at + 1 :].encode("ascii")
            if (
                prefix
                and name
                and len(prefix) <= tarfile.LENGTH_PREFIX
                and len(name) <= tarfile.LENGTH_NAME
            ):
                representable = True
                break
    if not representable:
        raise PublicReleaseError(f"{context} is not representable in POSIX USTAR")
    return member


def _canonical_git_member(value: object, *, context: str) -> str:
    member = _canonical_member(value, context=context)
    if member.endswith("/"):
        raise PublicReleaseError(f"{context} cannot name a directory")
    return member


def _enters_namespace(member: str, prefixes: Sequence[str]) -> bool:
    return any(
        member == prefix.removesuffix("/") or member.startswith(prefix)
        for prefix in prefixes
    )


def _read_descriptor(pinned: _PinnedFile, *, maximum: int | None = None) -> bytes:
    if maximum is not None and pinned.size_bytes > maximum:
        raise PublicReleaseError(f"metadata exceeds {maximum} bytes")
    os.lseek(pinned.descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = pinned.size_bytes
    while remaining:
        chunk = os.read(pinned.descriptor, min(remaining, READ_CHUNK_BYTES))
        if not chunk:
            raise PublicReleaseError("pinned file was truncated while read")
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    observed = os.fstat(pinned.descriptor)
    if (
        observed.st_size != pinned.size_bytes
        or observed.st_mtime_ns != pinned.modified_ns
        or _sha256(raw) != pinned.sha256
    ):
        raise PublicReleaseError("pinned file changed while read")
    return raw


def _digest_descriptor(descriptor: int) -> tuple[str, int, os.stat_result]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    remaining = before.st_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, READ_CHUNK_BYTES))
        if not chunk:
            raise PublicReleaseError("file was truncated while hashing")
        digest.update(chunk)
        size += len(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != before.st_size:
        raise PublicReleaseError("file changed while hashing")
    return digest.hexdigest(), size, after


def _require_absolute_canonical(path: Path, *, context: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise PublicReleaseError(f"{context} must be an absolute pathlib.Path")
    absolute = Path(os.path.abspath(path))  # noqa: PTH100
    if absolute != path:
        raise PublicReleaseError(f"{context} must be lexically normalized")
    return absolute


def _pin_absolute_file(
    path: Path,
    *,
    context: str,
    maximum: int | None = None,
    require_single_link: bool = True,
) -> _PinnedFile:
    absolute = _require_absolute_canonical(path, context=context)
    try:
        entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"{context} is unavailable") from error
    if (
        resolved != absolute
        or not stat.S_ISREG(entry.st_mode)
        or (require_single_link and entry.st_nlink != 1)
    ):
        raise PublicReleaseError(f"{context} is not a canonical regular file")
    try:
        descriptor = os.open(absolute, _FILE_READ_FLAGS)
    except OSError as error:
        raise PublicReleaseError(f"{context} cannot be safely opened") from error
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (require_single_link and observed.st_nlink != 1)
            or (observed.st_dev, observed.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            raise PublicReleaseError(f"{context} changed while opened")
        if maximum is not None and observed.st_size > maximum:
            raise PublicReleaseError(f"{context} exceeds {maximum} bytes")
        digest, size, stable = _digest_descriptor(descriptor)
        return _PinnedFile(
            path=absolute,
            root=None,
            member=None,
            descriptor=descriptor,
            device=stable.st_dev,
            inode=stable.st_ino,
            link_count=stable.st_nlink,
            mode=stat.S_IMODE(stable.st_mode),
            size_bytes=size,
            modified_ns=stable.st_mtime_ns,
            sha256=digest,
        )
    except Exception:
        os.close(descriptor)
        raise


def _pin_root(path: Path, *, context: str) -> _PinnedRoot:
    absolute = _require_absolute_canonical(path, context=context)
    try:
        entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"{context} is unavailable") from error
    if (
        resolved != absolute
        or stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISDIR(entry.st_mode)
    ):
        raise PublicReleaseError(f"{context} must be a canonical non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(absolute, flags)
    observed = os.fstat(descriptor)
    if (observed.st_dev, observed.st_ino) != (entry.st_dev, entry.st_ino):
        os.close(descriptor)
        raise PublicReleaseError(f"{context} changed while opened")
    return _PinnedRoot(
        path=absolute,
        descriptor=descriptor,
        device=observed.st_dev,
        inode=observed.st_ino,
    )


def _open_member_descriptor(root: _PinnedRoot, member: str) -> int:
    parts = PurePosixPath(member).parts
    current = os.dup(root.descriptor)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for part in parts[:-1]:
            try:
                following = os.open(part, directory_flags, dir_fd=current)
            except OSError as error:
                raise PublicReleaseError(
                    f"cannot open directory component for member {member!r}",
                ) from error
            os.close(current)
            current = following
        try:
            return os.open(parts[-1], _FILE_READ_FLAGS, dir_fd=current)
        except OSError as error:
            raise PublicReleaseError(f"cannot safely open member {member!r}") from error
    finally:
        os.close(current)


def _pin_member(
    root: _PinnedRoot,
    member: str,
    *,
    context: str,
    maximum: int | None = None,
    expected_size: int | None = None,
) -> _PinnedFile:
    canonical = _canonical_member(member, context=context)
    descriptor = _open_member_descriptor(root, canonical)
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise PublicReleaseError(f"{context} must be a single-link regular file")
        if maximum is not None and entry.st_size > maximum:
            raise PublicReleaseError(f"{context} exceeds {maximum} bytes")
        if expected_size is not None and entry.st_size != expected_size:
            raise PublicReleaseError(
                f"{context} size differs from its authenticated record",
            )
        digest, size, stable = _digest_descriptor(descriptor)
        return _PinnedFile(
            path=None,
            root=root,
            member=canonical,
            descriptor=descriptor,
            device=stable.st_dev,
            inode=stable.st_ino,
            link_count=stable.st_nlink,
            mode=stat.S_IMODE(stable.st_mode),
            size_bytes=size,
            modified_ns=stable.st_mtime_ns,
            sha256=digest,
        )
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_root(root: _PinnedRoot, *, context: str) -> None:
    try:
        entry = os.lstat(root.path)
        resolved = root.path.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"{context} disappeared") from error
    observed = os.fstat(root.descriptor)
    expected = (root.device, root.inode)
    if (
        resolved != root.path
        or stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISDIR(entry.st_mode)
        or (entry.st_dev, entry.st_ino) != expected
        or (observed.st_dev, observed.st_ino) != expected
    ):
        raise PublicReleaseError(f"{context} changed")


def _revalidate_file(pinned: _PinnedFile, *, context: str) -> None:
    observed = os.fstat(pinned.descriptor)
    expected = (
        pinned.device,
        pinned.inode,
        pinned.size_bytes,
        pinned.modified_ns,
    )
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != pinned.link_count
        or (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns)
        != expected
    ):
        raise PublicReleaseError(f"{context} descriptor changed")
    if pinned.root is None:
        if pinned.path is None:
            raise PublicReleaseError(f"{context} lacks a path")
        entry = pinned.path.stat(follow_symlinks=False)
    else:
        if pinned.member is None:
            raise PublicReleaseError(f"{context} lacks a root member")
        temporary = _open_member_descriptor(pinned.root, pinned.member)
        try:
            entry = os.fstat(temporary)
        finally:
            os.close(temporary)
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != pinned.link_count
        or (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns) != expected
    ):
        raise PublicReleaseError(f"{context} path entry changed")
    digest, size, _ = _digest_descriptor(pinned.descriptor)
    if digest != pinned.sha256 or size != pinned.size_bytes:
        raise PublicReleaseError(f"{context} bytes changed")


def _validate_public_member(member: str, *, context: str) -> str:
    canonical = _canonical_member(member, context=context)
    if canonical in {ARCHIVE_MANIFEST_MEMBER, ARCHIVE_CHECKSUM_MEMBER}:
        raise PublicReleaseError(f"{context} collides with a reserved archive member")
    if _enters_namespace(canonical, FORBIDDEN_PUBLIC_PREFIXES):
        raise PublicReleaseError(f"{context} enters a forbidden public namespace")
    if canonical in RESTRICTED_EXECUTION_PATHS:
        raise PublicReleaseError(
            f"{context} is a forbidden restricted execution path",
        )
    return canonical


def _normalize_release(value: object, *, mode: str) -> dict[str, str]:
    release = _expect_mapping(value, context="plan.release")
    _expect_keys(
        release,
        {
            "release_id",
            "version",
            "archive_name",
            "receipt_name",
            "source_commit_a",
            "release_commit_b",
            "source_tag",
        },
        context="plan.release",
    )
    release_id = _expect_token(release["release_id"], context="release_id")
    version = _expect_string(release["version"], context="version")
    if _VERSION_RE.fullmatch(version) is None:
        raise PublicReleaseError("version is not canonical")
    archive_name = _expect_string(release["archive_name"], context="archive_name")
    receipt_name = _expect_string(release["receipt_name"], context="receipt_name")
    if (
        PurePosixPath(archive_name).name != archive_name
        or not archive_name.isascii()
        or not archive_name.endswith(".tar")
        or archive_name in {".", ".."}
    ):
        raise PublicReleaseError("archive_name must be an ASCII .tar basename")
    if receipt_name != f"{archive_name}.receipt.json":
        raise PublicReleaseError("receipt_name must be archive_name + '.receipt.json'")
    source_a = _expect_git_sha(release["source_commit_a"], context="source_commit_a")
    release_b = _expect_git_sha(release["release_commit_b"], context="release_commit_b")
    if mode == "final" and source_a != SOURCE_COMMIT_A:
        raise PublicReleaseError("final plan does not use the frozen source commit A")
    tag = _expect_string(release["source_tag"], context="source_tag")
    if _TAG_RE.fullmatch(tag) is None or ".." in tag or tag.endswith("/"):
        raise PublicReleaseError("source_tag is not canonical")
    return {
        "release_id": release_id,
        "version": version,
        "archive_name": archive_name,
        "receipt_name": receipt_name,
        "source_commit_a": source_a,
        "release_commit_b": release_b,
        "source_tag": tag,
    }


_ANCHOR_KEYS: Final = {
    "source_data_manifest_sha256",
    "artifact_registry_sha256",
    "release_evidence_sha256",
    "document_reconciliation_sha256",
    "document_anchor_sha256",
    "dependency_inventory_sha256",
    "release_approval_sha256",
    "sealed_completion_sha256",
    "completion_attestation_sha256",
    "k500_authority_projection_sha256",
}


def _normalize_anchors(value: object) -> dict[str, str]:
    anchors = _expect_mapping(value, context="plan.anchors")
    _expect_keys(anchors, _ANCHOR_KEYS, context="plan.anchors")
    return {
        key: _expect_sha256(anchors[key], context=f"plan.anchors.{key}")
        for key in sorted(_ANCHOR_KEYS)
    }


def _normalize_approvals(
    value: object,
    *,
    mode: str,
) -> tuple[list[dict[str, object]], int]:
    records = _expect_sequence(value, context="plan.approvals")
    kinds = ("license-review", "public-boundary")
    if len(records) != len(kinds):
        raise PublicReleaseError("plan.approvals must contain exactly two records")
    normalized: list[dict[str, object]] = []
    pending = 0
    for index, kind in enumerate(kinds):
        context = f"plan.approvals[{index}]"
        record = _expect_mapping(records[index], context=context)
        _expect_keys(
            record,
            {"kind", "status", "receipt_id", "sha256"},
            context=context,
        )
        if record["kind"] != kind:
            raise PublicReleaseError(f"{context} is not in canonical approval order")
        status = _expect_string(record["status"], context=f"{context}.status")
        if status not in {"pending", "ready"}:
            raise PublicReleaseError(f"{context}.status is invalid")
        receipt_id = _expect_optional_string(
            record["receipt_id"],
            context=f"{context}.receipt_id",
        )
        digest = record["sha256"]
        if status == "ready":
            if receipt_id is None:
                raise PublicReleaseError(f"{context} ready approval lacks receipt_id")
            receipt_id = _expect_token(receipt_id, context=f"{context}.receipt_id")
            normalized_digest: str | None = _expect_sha256(
                digest,
                context=f"{context}.sha256",
            )
        else:
            if receipt_id is not None or digest is not None:
                raise PublicReleaseError(f"{context} pending approval must be empty")
            normalized_digest = None
            pending += 1
        normalized.append(
            {
                "kind": kind,
                "status": status,
                "receipt_id": receipt_id,
                "sha256": normalized_digest,
            },
        )
    if mode == "final" and pending:
        raise PublicReleaseError("final plan contains pending approvals")
    if list(records) != normalized:
        raise PublicReleaseError("plan.approvals is not canonical")
    return normalized, pending


def _normalize_execution(value: object, *, mode: str) -> dict[str, object]:
    execution = _expect_mapping(value, context="plan.execution")
    _expect_keys(
        execution,
        {"generated_version_sha256", "paths"},
        context="plan.execution",
    )
    generated = _expect_sha256(
        execution["generated_version_sha256"],
        context="plan.execution.generated_version_sha256",
    )
    if mode == "final" and generated != GENERATED_VERSION_SHA256:
        raise PublicReleaseError("generated _version.py execution hash is not frozen")
    raw_paths = _expect_sequence(execution["paths"], context="plan.execution.paths")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_paths):
        context = f"plan.execution.paths[{index}]"
        record = _expect_mapping(raw, context=context)
        _expect_keys(record, {"path", "sha256"}, context=context)
        member = _canonical_git_member(record["path"], context=f"{context}.path")
        if member == GENERATED_VERSION_MEMBER or member in seen:
            raise PublicReleaseError(f"{context}.path is duplicated or generated")
        seen.add(member)
        normalized.append(
            {
                "path": member,
                "sha256": _expect_sha256(
                    record["sha256"],
                    context=f"{context}.sha256",
                ),
            },
        )
    normalized.sort(key=lambda record: record["path"])
    if list(raw_paths) != normalized:
        raise PublicReleaseError("plan.execution.paths is not canonical")
    if mode == "final":
        frozen_paths = set(k500_authority_projection.GIT_EXECUTION_PATHS)
        if len(normalized) != EXECUTION_PATH_COUNT or seen != frozen_paths:
            raise PublicReleaseError(
                "final plan execution paths differ from the frozen K500 inventory",
            )
    return {"generated_version_sha256": generated, "paths": normalized}


def _normalize_code_paths(
    value: object,
    *,
    execution_paths: Sequence[Mapping[str, str]],
    mode: str,
) -> list[str]:
    raw_paths = _expect_sequence(value, context="plan.code_paths")
    paths: list[str] = []
    for index, raw in enumerate(raw_paths):
        member = _canonical_git_member(raw, context=f"plan.code_paths[{index}]")
        if member in paths:
            raise PublicReleaseError("plan.code_paths contains a duplicate")
        if _enters_namespace(member, FORBIDDEN_CODE_PREFIXES) or _enters_namespace(
            member,
            ("external/",),
        ):
            raise PublicReleaseError(
                f"code path is forbidden from public release: {member}",
            )
        paths.append(member)
    normalized = sorted(paths)
    if list(raw_paths) != normalized:
        raise PublicReleaseError("plan.code_paths is not canonical")
    if mode == "final":
        required = (
            REQUIRED_CODE_PATHS
            | {str(record["path"]) for record in execution_paths}
            - RESTRICTED_EXECUTION_PATHS
        )
        missing = sorted(required - set(normalized))
        if missing:
            raise PublicReleaseError(f"plan.code_paths omits required paths: {missing}")
    return normalized


def _normalize_source_dispositions(
    value: object,
    *,
    mode: str,
) -> tuple[list[dict[str, object]], int]:
    records = _expect_sequence(value, context="plan.source_dispositions")
    normalized: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    release_members: set[str] = set()
    pending = 0
    for index, raw in enumerate(records):
        context = f"plan.source_dispositions[{index}]"
        record = _expect_mapping(raw, context=context)
        _expect_keys(
            record,
            {
                "root",
                "source_member",
                "release_member",
                "disposition",
                "dependency_ids",
                "reason",
            },
            context=context,
        )
        root = _expect_string(record["root"], context=f"{context}.root")
        if root not in {"source-data", "evidence-source"}:
            raise PublicReleaseError(f"{context}.root is invalid")
        source_member = _canonical_member(
            record["source_member"],
            context=f"{context}.source_member",
        )
        identity = (root, source_member)
        if identity in identities:
            raise PublicReleaseError(f"{context} duplicates a source identity")
        identities.add(identity)
        disposition = _expect_string(
            record["disposition"],
            context=f"{context}.disposition",
        )
        if disposition not in {"include", "exclude", "pending"}:
            raise PublicReleaseError(f"{context}.disposition is invalid")
        raw_dependencies = _expect_sequence(
            record["dependency_ids"],
            context=f"{context}.dependency_ids",
        )
        dependencies: list[str] = []
        for dependency_index, raw_dependency in enumerate(raw_dependencies):
            dependency = _expect_string(
                raw_dependency,
                context=f"{context}.dependency_ids[{dependency_index}]",
            )
            if dependency not in DEPENDENCY_IDS or dependency in dependencies:
                raise PublicReleaseError(f"{context}.dependency_ids is invalid")
            dependencies.append(dependency)
        dependencies.sort()
        if list(raw_dependencies) != dependencies:
            raise PublicReleaseError(f"{context}.dependency_ids is not canonical")
        reason = _expect_optional_string(record["reason"], context=f"{context}.reason")
        if disposition == "include":
            release_member: str | None = _validate_public_member(
                record["release_member"],
                context=f"{context}.release_member",
            )
            if reason is not None:
                raise PublicReleaseError(
                    f"{context} included source cannot have reason",
                )
            if release_member in release_members:
                raise PublicReleaseError(f"{context} duplicates an included member")
            release_members.add(release_member)
        else:
            if record["release_member"] is not None or reason is None:
                raise PublicReleaseError(
                    f"{context} non-included source needs null member and a reason",
                )
            release_member = None
            if disposition == "pending":
                pending += 1
        normalized.append(
            {
                "root": root,
                "source_member": source_member,
                "release_member": release_member,
                "disposition": disposition,
                "dependency_ids": dependencies,
                "reason": reason,
            },
        )
    normalized.sort(
        key=lambda record: (str(record["root"]), str(record["source_member"])),
    )
    if list(records) != normalized:
        raise PublicReleaseError("plan.source_dispositions is not canonical")
    if mode == "final" and pending:
        raise PublicReleaseError("final plan contains pending source dispositions")
    return normalized, pending


def _normalize_documents(
    value: object,
    *,
    mode: str,
) -> tuple[list[dict[str, object]], int]:
    records = _expect_sequence(value, context="plan.documents")
    document_ids = ("main", "s1", "rebuttal")
    if len(records) != len(document_ids):
        raise PublicReleaseError(
            "plan.documents must account for main, S1, and rebuttal",
        )
    normalized: list[dict[str, object]] = []
    pending = 0
    for index, document_id in enumerate(document_ids):
        context = f"plan.documents[{index}]"
        record = _expect_mapping(records[index], context=context)
        _expect_keys(
            record,
            {"document_id", "disposition", "release_member", "reason"},
            context=context,
        )
        if record["document_id"] != document_id:
            raise PublicReleaseError(f"{context} is not in canonical document order")
        disposition = _expect_string(
            record["disposition"],
            context=f"{context}.disposition",
        )
        if disposition not in {"exclude", "pending"}:
            raise PublicReleaseError(f"{context}.disposition is invalid")
        release_member_raw = record["release_member"]
        reason = _expect_optional_string(record["reason"], context=f"{context}.reason")
        if release_member_raw is not None or reason is None:
            raise PublicReleaseError(
                f"{context} non-included document needs null member and reason",
            )
        release_member = None
        if disposition == "pending":
            pending += 1
        normalized.append(
            {
                "document_id": document_id,
                "disposition": disposition,
                "release_member": release_member,
                "reason": reason,
            },
        )
    if mode == "final" and pending:
        raise PublicReleaseError("final plan contains pending document dispositions")
    if mode == "final" and any(
        record["disposition"] != "exclude" for record in normalized
    ):
        raise PublicReleaseError(
            "final public release must explicitly exclude all source-text documents; "
            "no rendered-document/visual-QA anchor exists",
        )
    if list(records) != normalized:
        raise PublicReleaseError("plan.documents is not canonical")
    return normalized, pending


def _normalize_plan(value: Mapping[str, object]) -> tuple[dict[str, object], int]:
    _expect_keys(
        value,
        {
            "schema",
            "contract",
            "mode",
            "release",
            "anchors",
            "approvals",
            "execution",
            "code_paths",
            "source_dispositions",
            "documents",
        },
        context="public release plan",
    )
    if value["schema"] != PLAN_SCHEMA or value["contract"] != PLAN_CONTRACT:
        raise PublicReleaseError("public release plan has the wrong schema or contract")
    mode = _expect_string(value["mode"], context="plan.mode")
    if mode not in {"draft", "final"}:
        raise PublicReleaseError("plan.mode must be draft or final")
    release = _normalize_release(value["release"], mode=mode)
    anchors = _normalize_anchors(value["anchors"])
    approvals, approval_pending = _normalize_approvals(value["approvals"], mode=mode)
    execution = _normalize_execution(value["execution"], mode=mode)
    code_paths = _normalize_code_paths(
        value["code_paths"],
        execution_paths=_expect_sequence(execution["paths"], context="execution paths"),
        mode=mode,
    )
    sources, source_pending = _normalize_source_dispositions(
        value["source_dispositions"],
        mode=mode,
    )
    documents, document_pending = _normalize_documents(value["documents"], mode=mode)
    normalized = {
        "schema": PLAN_SCHEMA,
        "contract": PLAN_CONTRACT,
        "mode": mode,
        "release": release,
        "anchors": anchors,
        "approvals": approvals,
        "execution": execution,
        "code_paths": code_paths,
        "source_dispositions": sources,
        "documents": documents,
    }
    if dict(value) != normalized:
        raise PublicReleaseError("public release plan is not canonical")
    return normalized, approval_pending + source_pending + document_pending


def _pin_and_parse_plan(
    path: Path,
    expected_sha256: str,
) -> tuple[_PinnedFile, dict[str, object], int]:
    expected = _expect_sha256(expected_sha256, context="expected_plan_sha256")
    pinned = _pin_absolute_file(
        path,
        context="public release plan",
        maximum=MAX_METADATA_BYTES,
    )
    try:
        if pinned.sha256 != expected:
            raise PublicReleaseError(
                "public release plan does not match its independent anchor",
            )
        raw = _read_descriptor(pinned, maximum=MAX_METADATA_BYTES)
        value = _parse_canonical(raw, context="public release plan")
        normalized, pending = _normalize_plan(value)
    except Exception:
        pinned.close()
        raise
    return pinned, normalized, pending


def audit_public_release_plan(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
) -> PlanAuditReceipt:
    """Audit one independently anchored draft or final plan without input access."""
    pinned, plan, pending = _pin_and_parse_plan(plan_path, expected_plan_sha256)
    try:
        _revalidate_file(pinned, context="public release plan")
        mode = str(plan["mode"])
        return PlanAuditReceipt(
            plan_path=str(plan_path),
            plan_sha256=pinned.sha256,
            mode=mode,
            ready_to_publish=mode == "final" and pending == 0,
            pending_count=pending,
        )
    finally:
        pinned.close()


def _parse_json(raw: bytes, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise PublicReleaseError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PublicReleaseError(f"{context} must be a JSON object")
    return value


def _require_git_byte_limit(value: int, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicReleaseError(f"{context} must be a nonnegative integer")
    return value


def _require_git_timeout(value: float, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PublicReleaseError(f"{context} must be a positive number")
    return float(value)


def _kill_git_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


def _raise_git_output_limit(stream_name: str, limit: int) -> NoReturn:
    raise PublicReleaseError(f"Git {stream_name} exceeded its {limit}-byte limit")


def _raise_git_timeout(timeout_seconds: float) -> NoReturn:
    raise PublicReleaseError(
        f"Git execution exceeded its {timeout_seconds:g}-second timeout",
    )


def _collect_bounded_git_output(
    process: subprocess.Popen[bytes],
    *,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    timeout_seconds: float,
) -> tuple[bytes, bytes, int]:
    if process.stdout is None or process.stderr is None:
        raise PublicReleaseError("bounded Git execution requires two output pipes")
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    limits = {"stdout": max_stdout_bytes, "stderr": max_stderr_bytes}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout_seconds
    try:
        for name, stream in streams.items():
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=name)
        while selector.get_map():
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                _raise_git_timeout(timeout_seconds)
            events = selector.select(remaining_seconds)
            if not events:
                _raise_git_timeout(timeout_seconds)
            for key, _ in events:
                name = str(key.data)
                stream = streams[name]
                remaining = limits[name] + 1 - len(buffers[name])
                try:
                    chunk = os.read(
                        stream.fileno(),
                        min(READ_CHUNK_BYTES, remaining),
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limits[name]:
                    _raise_git_output_limit(name, limits[name])
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _raise_git_timeout(timeout_seconds)
        try:
            return (
                bytes(buffers["stdout"]),
                bytes(buffers["stderr"]),
                process.wait(timeout=remaining_seconds),
            )
        except subprocess.TimeoutExpired:
            _raise_git_timeout(timeout_seconds)
    except BaseException:
        _kill_git_process_group(process)
        process.wait()
        raise
    finally:
        selector.close()
        for stream in streams.values():
            if not stream.closed:
                stream.close()


def _git_command(  # noqa: PLR0913
    repository_root: _PinnedRoot,
    git_executable: _PinnedFile,
    arguments: Sequence[str],
    *,
    max_stdout_bytes: int = MAX_GIT_STDOUT_BYTES,
    max_stderr_bytes: int = MAX_GIT_STDERR_BYTES,
    timeout_seconds: float = MAX_GIT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    stdout_limit = _require_git_byte_limit(
        max_stdout_bytes,
        context="Git stdout byte limit",
    )
    stderr_limit = _require_git_byte_limit(
        max_stderr_bytes,
        context="Git stderr byte limit",
    )
    timeout = _require_git_timeout(timeout_seconds, context="Git timeout")
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    _revalidate_root(repository_root, context="Git repository root before command")
    _revalidate_file(git_executable, context="Git executable before command")
    descriptor_root = Path("/proc/self/fd")
    executable: str | None = None
    preexec_fn: Callable[[], None] | None = None
    if descriptor_root.is_dir():
        executable = f"{descriptor_root.as_posix()}/{git_executable.descriptor}"
        cwd: str | None = f"{descriptor_root.as_posix()}/{repository_root.descriptor}"
    elif hasattr(os, "fchdir"):
        cwd = None

        def enter_pinned_repository() -> None:
            os.fchdir(repository_root.descriptor)

        preexec_fn = enter_pinned_repository
    else:
        raise PublicReleaseError("descriptor-anchored Git execution is unavailable")
    try:
        command = [
            GIT_EXECUTABLE.as_posix(),
            "--no-pager",
            "--no-replace-objects",
            "--no-optional-locks",
            "--work-tree=.",
            *_GIT_CONFIG_OVERRIDES,
            *arguments,
        ]
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable=executable,
            pass_fds=(git_executable.descriptor, repository_root.descriptor),
            preexec_fn=preexec_fn,  # noqa: PLW1509
            start_new_session=True,
        )
        stdout, stderr, returncode = _collect_bounded_git_output(
            process,
            max_stdout_bytes=stdout_limit,
            max_stderr_bytes=stderr_limit,
            timeout_seconds=timeout,
        )
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    except OSError as error:
        raise PublicReleaseError("cannot execute bounded Git verification") from error
    finally:
        _revalidate_root(repository_root, context="Git repository root after command")
        _revalidate_file(git_executable, context="Git executable after command")


def _git_require(
    repository_root: _PinnedRoot,
    git_executable: _PinnedFile,
    arguments: Sequence[str],
    *,
    context: str,
    max_stdout_bytes: int = MAX_GIT_STDOUT_BYTES,
) -> bytes:
    completed = _git_command(
        repository_root,
        git_executable,
        arguments,
        max_stdout_bytes=max_stdout_bytes,
    )
    if completed.returncode != 0:
        raise PublicReleaseError(f"Git verification failed for {context}")
    return completed.stdout


def _git_commit_member(
    repository_root: _PinnedRoot,
    git_executable: _PinnedFile,
    commit: str,
    member: str,
) -> bytes:
    return _git_require(
        repository_root,
        git_executable,
        ["show", f"{commit}:{member}"],
        context=f"{commit}:{member}",
        max_stdout_bytes=MAX_GIT_BLOB_BYTES,
    )


def _git_member_absent(
    repository_root: _PinnedRoot,
    git_executable: _PinnedFile,
    commit: str,
    member: str,
) -> None:
    completed = _git_command(
        repository_root,
        git_executable,
        ["cat-file", "-e", f"{commit}:{member}"],
        max_stdout_bytes=0,
    )
    if completed.returncode == 0:
        raise PublicReleaseError(
            f"generated member unexpectedly exists in {commit}: {member}",
        )
    if completed.returncode not in {1, 128}:
        raise PublicReleaseError(f"cannot prove generated member absence in {commit}")


def _validate_git_lineage(
    repository_root: _PinnedRoot,
    git_executable: _PinnedFile,
    plan: Mapping[str, object],
) -> tuple[dict[str, bytes], dict[str, object]]:
    release = _expect_mapping(plan["release"], context="plan.release")
    execution = _expect_mapping(plan["execution"], context="plan.execution")
    source_a = _expect_git_sha(release["source_commit_a"], context="source commit A")
    release_b = _expect_git_sha(release["release_commit_b"], context="release commit B")
    tag = _expect_string(release["source_tag"], context="source tag")
    for label, commit in (("source A", source_a), ("release B", release_b)):
        observed = _git_require(
            repository_root,
            git_executable,
            ["rev-parse", "--verify", f"{commit}^{{commit}}"],
            context=label,
            max_stdout_bytes=MAX_GIT_CONTROL_STDOUT_BYTES,
        ).strip()
        if observed != commit.encode("ascii"):
            raise PublicReleaseError(f"Git {label} does not resolve exactly")
    ancestor = _git_command(
        repository_root,
        git_executable,
        ["merge-base", "--is-ancestor", source_a, release_b],
        max_stdout_bytes=0,
    )
    if ancestor.returncode != 0:
        raise PublicReleaseError("source commit A is not an ancestor of release B")
    tag_commit = _git_require(
        repository_root,
        git_executable,
        ["rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        context="release tag",
        max_stdout_bytes=MAX_GIT_CONTROL_STDOUT_BYTES,
    ).strip()
    if tag_commit != release_b.encode("ascii"):
        raise PublicReleaseError("release tag does not resolve to release commit B")
    _git_member_absent(
        repository_root,
        git_executable,
        source_a,
        GENERATED_VERSION_MEMBER,
    )
    _git_member_absent(
        repository_root,
        git_executable,
        release_b,
        GENERATED_VERSION_MEMBER,
    )

    code_paths = [
        _canonical_git_member(raw, context="plan code path")
        for raw in _expect_sequence(plan["code_paths"], context="plan.code_paths")
    ]
    execution_records = _expect_sequence(execution["paths"], context="execution paths")
    execution_by_path = {
        _canonical_git_member(
            _expect_mapping(record, context="execution record")["path"],
            context="execution path",
        ): _expect_sha256(
            _expect_mapping(record, context="execution record")["sha256"],
            context="execution path digest",
        )
        for record in execution_records
    }
    code_path_set = set(code_paths)
    restricted_execution_paths = set(execution_by_path) - code_path_set
    if (
        restricted_execution_paths != RESTRICTED_EXECUTION_PATHS
        or code_path_set & RESTRICTED_EXECUTION_PATHS
    ):
        raise PublicReleaseError(
            "public code paths do not exactly exclude the restricted execution "
            "inventory",
        )
    code_bytes: dict[str, bytes] = {}
    code_size_bytes = 0
    for member in code_paths:
        current = _git_commit_member(repository_root, git_executable, release_b, member)
        code_bytes[member] = current
        code_size_bytes += len(current)
        if code_size_bytes > MAX_PUBLIC_CODE_BYTES:
            raise PublicReleaseError("public release code exceeds its total byte limit")
    restricted_execution: list[dict[str, object]] = []
    for member, expected_execution in sorted(execution_by_path.items()):
        current = code_bytes.get(member)
        if current is None:
            current = _git_commit_member(
                repository_root,
                git_executable,
                release_b,
                member,
            )
        original = _git_commit_member(repository_root, git_executable, source_a, member)
        if original != current or _sha256(current) != expected_execution:
            raise PublicReleaseError(
                f"execution path is not implementation-identical at A and B: {member}",
            )
        if member in RESTRICTED_EXECUTION_PATHS:
            restricted_execution.append(
                {
                    "path": member,
                    "dependency_id": RESTRICTED_EXECUTION_DEPENDENCY_ID,
                    "bytes": len(current),
                    "sha256": expected_execution,
                    "verified_at_source_a_and_release_b": True,
                    "included_in_public_release": False,
                },
            )
    live_builder = _pin_absolute_file(
        Path(__file__).absolute(),
        context="live public release builder",
        maximum=MAX_METADATA_BYTES,
    )
    try:
        if code_bytes.get(BUILDER_MEMBER) != _read_descriptor(
            live_builder,
            maximum=MAX_METADATA_BYTES,
        ):
            raise PublicReleaseError(
                "live public release builder is not the release-B blob",
            )
    finally:
        live_builder.close()
    return code_bytes, {
        "source_commit_a": source_a,
        "release_commit_b": release_b,
        "source_tag": tag,
        "execution_path_count": len(execution_by_path),
        "execution_inventory_sha256": _sha256(_canonical_json(list(execution_records))),
        "restricted_execution_paths": restricted_execution,
        "generated_version_member": GENERATED_VERSION_MEMBER,
        "generated_version_present_in_git": False,
        "generated_version_sha256": execution["generated_version_sha256"],
    }


def _validate_restricted_execution_boundary(
    records: Mapping[str, Mapping[str, object]],
    verified_restricted_execution: Sequence[Mapping[str, object]],
) -> None:
    if len(verified_restricted_execution) != len(RESTRICTED_EXECUTION_PATHS):
        raise PublicReleaseError(
            "restricted execution verification inventory is incomplete",
        )
    verified = {
        _canonical_git_member(
            record.get("path"),
            context="restricted execution path",
        ): record
        for record in verified_restricted_execution
    }
    if set(verified) != RESTRICTED_EXECUTION_PATHS:
        raise PublicReleaseError(
            "restricted execution verification inventory is not exact",
        )
    record = records[RESTRICTED_EXECUTION_DEPENDENCY_ID]
    identity = _expect_mapping(
        record.get("identity"),
        context="restricted MutSig patch identity",
    )
    _expect_keys(
        identity,
        {"source", "patch_bytes", "patch_sha256"},
        context="restricted MutSig patch identity",
    )
    unresolved = _expect_sequence(
        record.get("unresolved"),
        context="restricted MutSig patch unresolved",
    )
    if not unresolved or any(
        not isinstance(value, str) or not value for value in unresolved
    ):
        raise PublicReleaseError(
            "restricted MutSig patch must retain an unresolved permission record",
        )
    verification = verified[RESTRICTED_EXECUTION_PATH]
    ledger_patch_bytes = _expect_int(
        identity["patch_bytes"],
        context="restricted MutSig patch bytes",
        minimum=1,
    )
    ledger_patch_sha256 = _expect_sha256(
        identity["patch_sha256"],
        context="restricted MutSig patch sha256",
    )
    if (
        record.get("dependency_class") != "mutsig_patch"
        or record.get("license_id") != "LicenseRef-Broad-MutSig2CV"
        or record.get("license_status") != "restricted"
        or record.get("redistribution") != "exclude"
        or record.get("included_in_public_release") is not False
        or ledger_patch_bytes
        != _expect_int(
            verification.get("bytes"),
            context="verified restricted execution bytes",
            minimum=1,
        )
        or ledger_patch_sha256
        != _expect_sha256(
            verification.get("sha256"),
            context="verified restricted execution sha256",
        )
        or verification.get("dependency_id") != RESTRICTED_EXECUTION_DEPENDENCY_ID
        or verification.get("verified_at_source_a_and_release_b") is not True
        or verification.get("included_in_public_release") is not False
    ):
        raise PublicReleaseError(
            "restricted MutSig execution path contradicts its dependency boundary",
        )


def _collect_sha256_values(value: object) -> set[str]:
    values: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            values.update(_collect_sha256_values(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            values.update(_collect_sha256_values(item))
    elif isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None:
        values.add(value)
    return values


def _require_exact_root_inventory(
    root: _PinnedRoot,
    expected: set[str],
    *,
    context: str,
) -> None:
    if len(expected) >= EXACT_ROOT_SCAN_CAP:
        raise PublicReleaseError(
            f"{context} expected inventory exceeds its scan policy",
        )
    _revalidate_root(root, context=f"{context} root before inventory scan")
    try:
        entries = os.scandir(root.descriptor)
    except OSError as error:
        raise PublicReleaseError(f"cannot list {context}") from error
    seen: set[str] = set()
    scanned = 0
    try:
        with entries:
            for entry in entries:
                scanned += 1
                if scanned >= EXACT_ROOT_SCAN_CAP:
                    raise PublicReleaseError(
                        f"{context} reached its bounded scan cap",
                    )
                name = entry.name
                if name not in expected:
                    raise PublicReleaseError(
                        f"{context} has unexpected inventory member {name!r}",
                    )
                if name in seen:
                    raise PublicReleaseError(
                        f"{context} has duplicate inventory member {name!r}",
                    )
                seen.add(name)
    except OSError as error:
        raise PublicReleaseError(f"cannot scan {context}") from error
    missing = expected - seen
    if missing:
        raise PublicReleaseError(
            f"{context} is missing inventory members {sorted(missing)!r}",
        )
    _revalidate_root(root, context=f"{context} root after inventory scan")


def _validate_cbase_release_boundary(
    record: Mapping[str, object],
    *,
    dialect_license_sha256: str,
    release_member_sha256: Mapping[str, object],
    archive_members: Sequence[str],
) -> None:
    """Close the exact mixed-license CBaSE public-release boundary."""
    if (
        record.get("dependency_id") != INCLUDED_DEPENDENCY_ID
        or record.get("dependency_class") != "cbase_source"
        or record.get("license_id") != CBASE_COMPOSITE_LICENSE_ID
        or record.get("license_status") != "permitted"
        or record.get("redistribution") != "include"
        or record.get("included_in_public_release") is not True
    ):
        raise PublicReleaseError("CBaSE top-level release boundary is invalid")
    unresolved = _expect_sequence(
        record.get("unresolved"),
        context="CBaSE unresolved gates",
    )
    if unresolved:
        raise PublicReleaseError("CBaSE release boundary has unresolved gates")
    source_artifacts = _expect_sequence(
        record.get("source_artifacts"),
        context="CBaSE source artifacts",
    )
    if source_artifacts:
        raise PublicReleaseError(
            "CBaSE official archive or auxiliary source artifacts cannot be released",
        )
    if record.get("scope") != CBASE_SCOPE:
        raise PublicReleaseError("CBaSE archive and auxiliary exclusion scope changed")

    identity = _expect_mapping(record.get("identity"), context="CBaSE identity")
    if identity.get("upstream_license_id") != CBASE_UPSTREAM_LICENSE_ID:
        raise PublicReleaseError("CBaSE upstream license identity changed")
    if identity.get("dialect_license_id") != CBASE_DIALECT_LICENSE_ID:
        raise PublicReleaseError("CBaSE DIALECT license identity changed")
    if identity.get("official_archive_role") != CBASE_OFFICIAL_ARCHIVE_ROLE:
        raise PublicReleaseError(
            "CBaSE official archive is not comparison-only provenance",
        )

    declared_license_sha256 = _expect_sha256(
        identity.get("dialect_license_file_sha256"),
        context="CBaSE DIALECT license",
    )
    observed_license_sha256 = _expect_sha256(
        dialect_license_sha256,
        context="release-B DIALECT license",
    )
    if (
        declared_license_sha256 != CBASE_DIALECT_LICENSE_SHA256
        or observed_license_sha256 != CBASE_DIALECT_LICENSE_SHA256
    ):
        raise PublicReleaseError("root LICENSE does not match the CBaSE boundary")

    release_files = _expect_mapping(
        identity.get("release_files"),
        context="CBaSE release_files",
    )
    if set(release_files) != set(CBASE_RELEASE_MEMBERS):
        raise PublicReleaseError("CBaSE release file allowlist is not exact")
    if set(release_member_sha256) != set(CBASE_RELEASE_MEMBERS):
        raise PublicReleaseError("CBaSE archive payload allowlist is not exact")
    for member in CBASE_RELEASE_MEMBERS:
        declared = _expect_sha256(
            release_files[member],
            context=f"CBaSE release file {member}",
        )
        observed = _expect_sha256(
            release_member_sha256[member],
            context=f"CBaSE archive payload {member}",
        )
        if declared != observed:
            raise PublicReleaseError(f"CBaSE release file digest differs: {member}")

    roles = _expect_mapping(
        identity.get("release_file_roles"),
        context="CBaSE release file roles",
    )
    if dict(roles) != CBASE_RELEASE_FILE_ROLES:
        raise PublicReleaseError("CBaSE release file roles changed")
    raw_licenses = _expect_mapping(
        identity.get("release_file_licenses"),
        context="CBaSE release file licenses",
    )
    if set(raw_licenses) != set(CBASE_RELEASE_MEMBERS):
        raise PublicReleaseError("CBaSE release file license allowlist is not exact")
    licenses: dict[str, tuple[str, ...]] = {}
    for member in CBASE_RELEASE_MEMBERS:
        values = _expect_sequence(
            raw_licenses[member],
            context=f"CBaSE release file licenses {member}",
        )
        licenses[member] = tuple(
            _expect_string(value, context=f"CBaSE license {member}") for value in values
        )
    if licenses != CBASE_RELEASE_FILE_LICENSES:
        raise PublicReleaseError("CBaSE release file licenses changed")

    for raw_member in archive_members:
        member = _canonical_member(raw_member, context="CBaSE archive member")
        if (
            _enters_namespace(
                member,
                (CBASE_RELEASE_NAMESPACE,),
            )
            and member not in CBASE_RELEASE_MEMBERS
        ):
            raise PublicReleaseError(
                f"unexpected CBaSE namespace member must remain excluded: {member}",
            )


def _validate_dependency_ledger(  # noqa: PLR0913
    dependency_root: _PinnedRoot,
    repository_root: _PinnedRoot,
    git_executable: _PinnedFile,
    release_b: str,
    expected_inventory_sha256: str,
    verified_restricted_execution: Sequence[Mapping[str, object]],
) -> tuple[
    tuple[_PinnedFile, ...],
    tuple[_PinnedEntry, ...],
    dict[str, bytes],
    set[str],
    list[dict[str, object]],
]:
    _require_exact_root_inventory(
        dependency_root,
        set(DEPENDENCY_METADATA_MEMBERS),
        context="dependency ledger",
    )
    pins: list[_PinnedFile] = []
    records: dict[str, dict[str, object]] = {}
    inventory: list[dict[str, object]] = []
    try:
        for member in DEPENDENCY_METADATA_MEMBERS:
            pin = _pin_member(
                dependency_root,
                member,
                context=f"dependency ledger {member}",
                maximum=MAX_METADATA_BYTES,
            )
            pins.append(pin)
            inventory.append(
                {"member": member, "bytes": pin.size_bytes, "sha256": pin.sha256},
            )
            repository_member = f"provenance/dependencies/{member}"
            if _git_commit_member(
                repository_root,
                git_executable,
                release_b,
                repository_member,
            ) != _read_descriptor(
                pin,
                maximum=MAX_METADATA_BYTES,
            ):
                raise PublicReleaseError(
                    f"dependency ledger member is not the release-B blob: {member}",
                )
            if member.endswith(".json") and member != "record.schema.json":
                record = _parse_json(
                    _read_descriptor(pin, maximum=MAX_METADATA_BYTES),
                    context=f"dependency record {member}",
                )
                dependency_id = _expect_string(
                    record.get("dependency_id"),
                    context=f"{member}.dependency_id",
                )
                if member != f"{dependency_id}.json" or dependency_id in records:
                    raise PublicReleaseError("dependency record identity is invalid")
                records[dependency_id] = record
        inventory_sha256 = _sha256(_canonical_json(inventory))
        if inventory_sha256 != expected_inventory_sha256:
            raise PublicReleaseError("dependency ledger does not match the plan anchor")
        if set(records) != set(DEPENDENCY_IDS):
            raise PublicReleaseError(
                "dependency ledger does not contain all 13 records",
            )

        excluded_hashes: set[str] = set()
        boundaries: list[dict[str, object]] = []
        for dependency_id in DEPENDENCY_IDS:
            record = records[dependency_id]
            redistribution = _expect_string(
                record.get("redistribution"),
                context=f"{dependency_id}.redistribution",
            )
            included = record.get("included_in_public_release")
            license_status = _expect_string(
                record.get("license_status"),
                context=f"{dependency_id}.license_status",
            )
            unresolved = _expect_sequence(
                record.get("unresolved"),
                context=f"{dependency_id}.unresolved",
            )
            license_id = _expect_string(
                record.get("license_id"),
                context=f"{dependency_id}.license_id",
            )
            should_include = dependency_id == INCLUDED_DEPENDENCY_ID
            if should_include:
                if (
                    redistribution != "include"
                    or included is not True
                    or license_status != "permitted"
                    or unresolved
                    or license_id == "NOASSERTION"
                ):
                    raise PublicReleaseError("CBaSE inclusion boundary is not closed")
            elif redistribution != "exclude" or included is not False:
                raise PublicReleaseError(
                    f"excluded dependency was escalated: {dependency_id}",
                )
            if not should_include:
                excluded_hashes.update(_collect_sha256_values(record))
            boundaries.append(
                {
                    "dependency_id": dependency_id,
                    "license_id": license_id,
                    "license_status": license_status,
                    "redistribution": redistribution,
                    "included_in_public_release": included,
                    "unresolved": list(unresolved),
                },
            )

        _validate_restricted_execution_boundary(
            records,
            verified_restricted_execution,
        )

        cbase = records[INCLUDED_DEPENDENCY_ID]
        cbase_bytes: dict[str, bytes] = {}
        for member in CBASE_RELEASE_MEMBERS:
            cbase_bytes[member] = _git_commit_member(
                repository_root,
                git_executable,
                release_b,
                member,
            )
        license_raw = _git_commit_member(
            repository_root,
            git_executable,
            release_b,
            "LICENSE",
        )
        _validate_cbase_release_boundary(
            cbase,
            dialect_license_sha256=_sha256(license_raw),
            release_member_sha256={
                member: _sha256(raw) for member, raw in cbase_bytes.items()
            },
            archive_members=tuple(cbase_bytes),
        )

        metadata_entries = tuple(
            _PinnedEntry(
                member=f"provenance/dependencies/{pin.member}",
                pinned=pin,
                role="dependency-provenance",
                origin={
                    "kind": "dependency-ledger",
                    "member": str(pin.member),
                },
            )
            for pin in pins
        )
        return (
            tuple(pins),
            metadata_entries,
            cbase_bytes,
            excluded_hashes,
            boundaries,
        )
    except Exception:
        for pin in pins:
            pin.close()
        raise


def _receipt_values(receipt: object, fields: Sequence[str]) -> tuple[object, ...]:
    try:
        return tuple(getattr(receipt, field_name) for field_name in fields)
    except AttributeError as error:
        raise PublicReleaseError(
            "native validator returned an incompatible receipt",
        ) from error


_SOURCE_RECEIPT_FIELDS: Final = (
    "source_data_root",
    "manifest_sha256",
    "file_count",
    "cohort_count",
    "total_bytes",
    "total_rows",
)
_REGISTRY_RECEIPT_FIELDS: Final = (
    "manifest_path",
    "manifest_sha256",
    "ready_count",
    "omitted_count",
)
_CLOSURE_RECEIPT_FIELDS: Final = (
    "manifest_path",
    "manifest_sha256",
    "gate_receipt_count",
    "source_member_count",
    "ready_count",
    "omitted_count",
)
_DOCUMENT_RECEIPT_FIELDS: Final = (
    "manifest_path",
    "manifest_sha256",
    "mode",
    "placement_count",
    "ready_count",
    "omitted_count",
    "pending_count",
)
_PROJECTION_RECEIPT_FIELDS: Final = (
    "projection_path",
    "projection_sha256",
    "completion_attestation_sha256",
    "completion_attestation_payload_sha256",
    "sealed_completion_sha256",
    "run_manifest_sha256",
    "source_a_commit",
    "release_b_commit",
    "release_tag",
    "git_blob_count",
    "generated_file_count",
    "snapshot_file_count",
    "execution_snapshot_sha256",
    "authority_digests",
    "authority_digest_count",
)


def _require_projection_receipt(
    config: PublicReleaseBuildConfig,
    plan: Mapping[str, object],
    projection_receipt: object,
) -> None:
    anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
    release = _expect_mapping(plan["release"], context="plan.release")
    values = _receipt_values(projection_receipt, _PROJECTION_RECEIPT_FIELDS)
    authority_digests = _expect_mapping(
        values[13],
        context="K500 authority projection digests",
    )
    if (
        values[0] != config.k500_authority_projection_path
        or values[1] != anchors["k500_authority_projection_sha256"]
        or values[1] != config.expected_k500_authority_projection_sha256
        or values[2] != anchors["completion_attestation_sha256"]
        or values[4] != anchors["sealed_completion_sha256"]
        or values[6] != release["source_commit_a"]
        or values[7] != release["release_commit_b"]
        or values[8] != release["source_tag"]
        or values[9:13] != (38, 1, 39, EXECUTION_SNAPSHOT_SHA256)
        or set(authority_digests)
        != set(k500_authority_projection.AUTHORITY_DIGEST_FIELDS)
        or any(
            _SHA256_RE.fullmatch(str(digest)) is None
            for digest in authority_digests.values()
        )
        or values[14] != 6
    ):
        raise PublicReleaseError(
            "native K500 authority projection contradicts the public-release plan",
        )


def _require_projection_source_authority(
    source_manifest: Mapping[str, object],
    projection_receipt: object,
) -> None:
    source_authority = _expect_mapping(
        source_manifest.get("authority"),
        context="source-data authority",
    )
    projection_values = _receipt_values(
        projection_receipt,
        _PROJECTION_RECEIPT_FIELDS,
    )
    projection_authority = _expect_mapping(
        projection_values[13],
        context="K500 projection authority digests",
    )
    for key in (
        "canonical_input_manifest_sha256",
        "provider_input_manifest_sha256",
    ):
        if source_authority.get(key) != projection_authority.get(key):
            raise PublicReleaseError(
                f"source-data and K500 projection authority differ for {key}",
            )


def _require_projection_git_executable(
    projection: Mapping[str, object],
    git_executable: _PinnedFile,
) -> None:
    source = _expect_mapping(projection.get("source"), context="K500 projection source")
    record = _expect_mapping(
        source.get("git_executable"),
        context="K500 projection Git executable",
    )
    _expect_keys(
        record,
        {"path", "bytes", "sha256"},
        context="K500 projection Git executable",
    )
    if (
        record["path"] != GIT_EXECUTABLE.as_posix()
        or record["bytes"] != git_executable.size_bytes
        or record["sha256"] != git_executable.sha256
    ):
        raise PublicReleaseError(
            "pinned Git executable differs from the K500 authority projection",
        )


def _run_native_validators(
    config: PublicReleaseBuildConfig,
    plan: Mapping[str, object],
) -> tuple[object, object, object, object, object]:
    anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
    try:
        source_receipt = source_data.validate_source_data_release(
            config.source_data_root,
            str(anchors["source_data_manifest_sha256"]),
        )
        registry_receipt = artifact_registry.validate_artifact_registry(
            config.artifact_registry_path,
            config.renderer_root,
            config.rendered_output_root,
            expected_manifest_sha256=str(anchors["artifact_registry_sha256"]),
        )
        closure_receipt = release_evidence.validate_release_evidence_closure(
            config.release_evidence_path,
            config.artifact_registry_path,
            config.renderer_root,
            config.rendered_output_root,
            config.gate_receipt_root,
            config.evidence_source_root,
            expected_closure_sha256=str(anchors["release_evidence_sha256"]),
            expected_artifact_registry_sha256=str(anchors["artifact_registry_sha256"]),
        )
        document_receipt = document_reconciliation.validate_document_reconciliation(
            config.document_reconciliation_path,
            config.artifact_registry_path,
            config.renderer_root,
            config.rendered_output_root,
            config.document_anchor_path,
            config.document_root,
            expected_manifest_sha256=str(anchors["document_reconciliation_sha256"]),
            expected_artifact_registry_sha256=str(anchors["artifact_registry_sha256"]),
            expected_document_anchor_sha256=str(anchors["document_anchor_sha256"]),
        )
        projection_receipt = (
            k500_authority_projection.validate_k500_authority_projection(
                config.k500_authority_projection_path,
                expected_projection_sha256=(
                    config.expected_k500_authority_projection_sha256
                ),
                repo_root=config.repository_root,
                git_executable=GIT_EXECUTABLE,
            )
        )
    except (
        source_data.SourceDataBuildError,
        artifact_registry.ArtifactRegistryError,
        release_evidence.ReleaseEvidenceError,
        document_reconciliation.DocumentReconciliationError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise PublicReleaseError(
            "one or more native release validators failed",
        ) from error
    source_values = _receipt_values(source_receipt, _SOURCE_RECEIPT_FIELDS)
    if (
        source_values[0] != str(config.source_data_root)
        or source_values[1] != anchors["source_data_manifest_sha256"]
    ):
        raise PublicReleaseError("native source-data receipt contradicts the plan")
    registry_values = _receipt_values(registry_receipt, _REGISTRY_RECEIPT_FIELDS)
    if (
        registry_values[0] != str(config.artifact_registry_path)
        or registry_values[1] != anchors["artifact_registry_sha256"]
    ):
        raise PublicReleaseError(
            "native artifact-registry receipt contradicts the plan",
        )
    closure_values = _receipt_values(closure_receipt, _CLOSURE_RECEIPT_FIELDS)
    if (
        closure_values[0] != str(config.release_evidence_path)
        or closure_values[1] != anchors["release_evidence_sha256"]
    ):
        raise PublicReleaseError("native release-evidence receipt contradicts the plan")
    document_values = _receipt_values(document_receipt, _DOCUMENT_RECEIPT_FIELDS)
    if (
        document_values[0] != str(config.document_reconciliation_path)
        or document_values[1] != anchors["document_reconciliation_sha256"]
        or document_values[2] != "final"
        or document_values[-1] != 0
    ):
        raise PublicReleaseError(
            "document reconciliation is not a closed final receipt",
        )
    _require_projection_receipt(config, plan, projection_receipt)
    return (
        source_receipt,
        registry_receipt,
        closure_receipt,
        document_receipt,
        projection_receipt,
    )


def _source_data_inventory(
    manifest: Mapping[str, object],
    manifest_pin: _PinnedFile,
    anchors: Mapping[str, object],
) -> tuple[list[dict[str, object]], int, int]:
    if manifest.get("schema") != source_data.SOURCE_DATA_SCHEMA:
        raise PublicReleaseError("source-data manifest has the wrong schema")
    authority = _expect_mapping(
        manifest.get("authority"),
        context="source-data authority",
    )
    for key, anchor_key in (
        ("release_approval_manifest_sha256", "release_approval_sha256"),
        ("sealed_completion_sha256", "sealed_completion_sha256"),
    ):
        if authority.get(key) != anchors[anchor_key]:
            raise PublicReleaseError(
                f"source-data authority does not bind {anchor_key}",
            )
    dataset = _expect_mapping(manifest.get("dataset"), context="source-data dataset")
    supporting = _expect_mapping(
        manifest.get("supporting_files"),
        context="source-data supporting_files",
    )
    records: list[dict[str, object]] = [
        {
            "member": source_data.SOURCE_DATA_MANIFEST_NAME,
            "bytes": manifest_pin.size_bytes,
            "sha256": manifest_pin.sha256,
            "role": "source-data-manifest",
        },
    ]
    for role in ("data_dictionary", "readme"):
        record = _expect_mapping(
            supporting.get(role),
            context=f"supporting_files.{role}",
        )
        records.append(
            {
                "member": _canonical_member(record.get("path"), context=f"{role}.path"),
                "bytes": _expect_int(record.get("bytes"), context=f"{role}.bytes"),
                "sha256": _expect_sha256(
                    record.get("sha256"),
                    context=f"{role}.sha256",
                ),
                "role": f"source-data-{role.replace('_', '-')}",
            },
        )
    cohort_files = _expect_sequence(
        dataset.get("cohort_files"),
        context="dataset.cohort_files",
    )
    for index, raw in enumerate(cohort_files):
        record = _expect_mapping(raw, context=f"dataset.cohort_files[{index}]")
        records.append(
            {
                "member": _canonical_member(
                    record.get("path"),
                    context=f"dataset.cohort_files[{index}].path",
                ),
                "bytes": _expect_int(
                    record.get("bytes"),
                    context=f"dataset.cohort_files[{index}].bytes",
                ),
                "sha256": _expect_sha256(
                    record.get("sha256"),
                    context=f"dataset.cohort_files[{index}].sha256",
                ),
                "role": "source-data-cohort",
            },
        )
    members = [str(record["member"]) for record in records]
    if len(records) != SOURCE_DATA_FILE_COUNT or len(set(members)) != len(records):
        raise PublicReleaseError("source-data manifest does not close exactly 35 files")
    expected_order = [
        source_data.SOURCE_DATA_MANIFEST_NAME,
        source_data.DATA_DICTIONARY_NAME,
        source_data.README_NAME,
        *(
            f"{source_data.COHORT_DIRECTORY_NAME}/{cohort}.csv"
            for cohort in source_data.TCGA_COHORTS
        ),
    ]
    if members != expected_order:
        raise PublicReleaseError(
            "source-data member order or inventory is not canonical",
        )
    total_rows = _expect_int(dataset.get("total_rows"), context="dataset.total_rows")
    total_bytes = sum(int(record["bytes"]) for record in records)
    return records, total_rows, total_bytes


def _registry_records(
    registry: Mapping[str, object],
    plan: Mapping[str, object],
) -> tuple[list[dict[str, object]], set[str], dict[str, object]]:
    if registry.get("schema") != artifact_registry.ARTIFACT_REGISTRY_SCHEMA:
        raise PublicReleaseError("artifact registry has the wrong schema")
    release = _expect_mapping(
        registry.get("release"),
        context="artifact registry release",
    )
    plan_release = _expect_mapping(plan["release"], context="plan.release")
    anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
    if (
        release.get("release_id") != plan_release["release_id"]
        or release.get("source_data_manifest_sha256")
        != anchors["source_data_manifest_sha256"]
    ):
        raise PublicReleaseError(
            "artifact registry release binding differs from the plan",
        )
    artifacts = _expect_sequence(
        registry.get("artifacts"),
        context="artifact registry artifacts",
    )
    if len(artifacts) != len(artifact_registry.ARTIFACT_SPECS):
        raise PublicReleaseError(
            "artifact registry does not account for all 13 artifacts",
        )
    outputs: list[dict[str, object]] = []
    renderers: set[str] = set()
    ready = 0
    omitted = 0
    statuses: list[dict[str, object]] = []
    for index, raw in enumerate(artifacts):
        record = _expect_mapping(raw, context=f"artifact[{index}]")
        semantic_id = _expect_token(
            record.get("semantic_id"),
            context="artifact semantic_id",
        )
        status_value = _expect_string(record.get("status"), context="artifact status")
        statuses.append({"semantic_id": semantic_id, "status": status_value})
        if status_value == "ready":
            ready += 1
            renderer = _expect_mapping(
                record.get("renderer"),
                context=f"{semantic_id} renderer",
            )
            renderer_member = _canonical_git_member(
                renderer.get("script"),
                context=f"{semantic_id} renderer script",
            )
            renderers.add(renderer_member)
            renderer_sha = _expect_sha256(
                renderer.get("sha256"),
                context=f"{semantic_id} renderer sha256",
            )
            for output_index, output_raw in enumerate(
                _expect_sequence(
                    record.get("outputs"),
                    context=f"{semantic_id} outputs",
                ),
            ):
                output = _expect_mapping(
                    output_raw,
                    context=f"{semantic_id} output[{output_index}]",
                )
                outputs.append(
                    {
                        "artifact_id": semantic_id,
                        "member": _validate_public_member(
                            output.get("release_member"),
                            context=f"{semantic_id} release_member",
                        ),
                        "bytes": _expect_int(
                            output.get("bytes"),
                            context=f"{semantic_id} output bytes",
                        ),
                        "sha256": _expect_sha256(
                            output.get("sha256"),
                            context=f"{semantic_id} output sha256",
                        ),
                        "media_type": _expect_string(
                            output.get("media_type"),
                            context=f"{semantic_id} output media_type",
                        ),
                        "renderer_member": renderer_member,
                        "renderer_sha256": renderer_sha,
                    },
                )
        elif status_value == "omitted":
            omitted += 1
        else:
            raise PublicReleaseError("artifact status must be ready or omitted")
    output_members = [str(record["member"]) for record in outputs]
    if len(output_members) != len(set(output_members)):
        raise PublicReleaseError("ready artifacts reuse a rendered output member")
    return (
        outputs,
        renderers,
        {
            "artifact_count": len(artifacts),
            "ready_count": ready,
            "omitted_count": omitted,
            "statuses": statuses,
        },
    )


def _closure_records(
    closure: Mapping[str, object],
    plan: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if closure.get("schema") != release_evidence.CLOSURE_SCHEMA:
        raise PublicReleaseError("release-evidence closure has the wrong schema")
    anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
    registry = _expect_mapping(
        closure.get("artifact_registry"),
        context="closure artifact_registry",
    )
    if registry.get("sha256") != anchors["artifact_registry_sha256"]:
        raise PublicReleaseError("release-evidence closure binds another registry")
    sources: list[dict[str, object]] = []
    for index, raw in enumerate(
        _expect_sequence(closure.get("source_data"), context="closure source_data"),
    ):
        record = _expect_mapping(raw, context=f"closure source_data[{index}]")
        sources.append(
            {
                "member": _canonical_member(
                    record.get("member"),
                    context=f"closure source_data[{index}].member",
                ),
                "bytes": _expect_int(
                    record.get("bytes"),
                    context=f"closure source_data[{index}].bytes",
                ),
                "sha256": _expect_sha256(
                    record.get("sha256"),
                    context=f"closure source_data[{index}].sha256",
                ),
            },
        )
    gates: list[dict[str, object]] = []
    for index, raw in enumerate(
        _expect_sequence(closure.get("gate_receipts"), context="closure gate_receipts"),
    ):
        record = _expect_mapping(raw, context=f"closure gate_receipts[{index}]")
        receipt_id = _canonical_member(
            record.get("receipt_id"),
            context="gate receipt id",
        )
        member = _canonical_member(
            record.get("member"),
            context="gate receipt member",
        )
        if member != receipt_id:
            raise PublicReleaseError("gate receipt member differs from its receipt ID")
        gates.append(
            {
                "gate": _expect_string(record.get("gate"), context="gate receipt gate"),
                "receipt_id": receipt_id,
                "member": member,
                "bytes": _expect_int(
                    record.get("bytes"),
                    context="gate receipt bytes",
                    minimum=1,
                ),
                "sha256": _expect_sha256(
                    record.get("sha256"),
                    context="gate receipt sha256",
                ),
            },
        )
    k500 = [record for record in gates if record["gate"] == "K500"]
    if len(k500) != 1 or k500[0]["sha256"] != anchors["completion_attestation_sha256"]:
        raise PublicReleaseError("K500 gate does not bind the completion attestation")
    return sources, gates


def _validate_completion_attestation_metadata(
    attestation: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "attestation_type",
        "status",
        "created_at_utc",
        "scope",
        "generator",
        "frozen_run",
        "completion",
        "contracts",
        "tasks",
        "attempts",
        "resources",
        "sealed_completion",
        "pre_attestation_inventory",
        "attestation_payload_sha256",
    }
    _expect_keys(attestation, expected_keys, context="K500 completion attestation")
    payload = dict(attestation)
    payload_sha256 = _expect_sha256(
        payload.pop("attestation_payload_sha256"),
        context="K500 attestation payload digest",
    )
    if payload_sha256 != _sha256(_canonical_json(payload)):
        raise PublicReleaseError(
            "K500 completion-attestation payload digest is invalid",
        )
    if (
        attestation["schema_version"] != COMPLETION_ATTESTATION_SCHEMA_VERSION
        or attestation["attestation_type"] != COMPLETION_ATTESTATION_TYPE
        or attestation["status"] != "complete"
    ):
        raise PublicReleaseError("K500 completion-attestation identity is invalid")
    generator = _expect_mapping(
        attestation["generator"],
        context="K500 attestation generator",
    )
    if generator != {
        "path": COMPLETION_ATTESTOR_MEMBER,
        "bytes": COMPLETION_ATTESTOR_BYTES,
        "sha256": COMPLETION_ATTESTOR_SHA256,
    }:
        raise PublicReleaseError("K500 completion-attestor identity changed")
    completion = _expect_mapping(
        attestation["completion"],
        context="K500 completion summary",
    )
    if (
        completion.get("cohorts_expected") != 32
        or completion.get("cohorts_validated") != 32
        or completion.get("backgrounds_expected") != 3
        or completion.get("tasks_expected") != 96
        or completion.get("tasks_validated") != 96
        or completion.get("features_per_task") != 500
        or len(_expect_sequence(attestation["contracts"], context="K500 contracts"))
        != 32
        or len(_expect_sequence(attestation["tasks"], context="K500 tasks")) != 96
    ):
        raise PublicReleaseError("K500 completion-attestation grid is incomplete")
    frozen_run = _expect_mapping(
        attestation["frozen_run"],
        context="K500 frozen run",
    )
    release = _expect_mapping(plan["release"], context="plan.release")
    execution = _expect_mapping(plan["execution"], context="plan.execution")
    implementation = _expect_mapping(
        frozen_run.get("implementation_sha256"),
        context="K500 implementation snapshot",
    )
    expected_implementation = {
        str(_expect_mapping(record, context="execution record")["path"]): str(
            _expect_mapping(record, context="execution record")["sha256"],
        )
        for record in _expect_sequence(execution["paths"], context="execution paths")
    }
    expected_implementation[GENERATED_VERSION_MEMBER] = GENERATED_VERSION_SHA256
    if (
        frozen_run.get("outer_git_head") != release["source_commit_a"]
        or frozen_run.get("outer_git_clean") is not True
        or implementation != expected_implementation
        or _sha256(_canonical_json(dict(implementation))) != EXECUTION_SNAPSHOT_SHA256
    ):
        raise PublicReleaseError("K500 attestation execution snapshot differs")
    return {
        "completion_attestation_payload_sha256": payload_sha256,
        "contracts_validated": 32,
        "tasks_validated": 96,
        "result_rows_opened": False,
    }


def _document_records(
    reconciliation: Mapping[str, object],
    anchor: Mapping[str, object],
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
    release = _expect_mapping(plan["release"], context="plan.release")
    if (
        reconciliation.get("schema")
        != document_reconciliation.DOCUMENT_RECONCILIATION_SCHEMA
        or reconciliation.get("mode") != "final"
        or reconciliation.get("release_id") != release["release_id"]
    ):
        raise PublicReleaseError(
            "document reconciliation is not final for this release",
        )
    inputs = _expect_mapping(reconciliation.get("inputs"), context="document inputs")
    if (
        inputs.get("artifact_registry_sha256") != anchors["artifact_registry_sha256"]
        or inputs.get("document_anchor_sha256") != anchors["document_anchor_sha256"]
    ):
        raise PublicReleaseError("document reconciliation input anchors differ")
    summary = _expect_mapping(reconciliation.get("summary"), context="document summary")
    if summary.get("reviewer_item_count") != len(
        document_reconciliation.REVIEWER_ITEM_ORDER,
    ):
        raise PublicReleaseError(
            "document reconciliation does not account for reviewer items",
        )
    if summary.get("artifact_count") != len(artifact_registry.ARTIFACT_SPECS):
        raise PublicReleaseError(
            "document reconciliation does not account for artifacts",
        )
    if summary.get("pending_count") != 0:
        raise PublicReleaseError("document reconciliation retains pending placements")
    if anchor.get("schema") != document_reconciliation.DOCUMENT_ANCHOR_SCHEMA:
        raise PublicReleaseError("document anchor has the wrong schema")
    records = _expect_sequence(
        anchor.get("documents"),
        context="document anchor documents",
    )
    if len(records) != 3:
        raise PublicReleaseError("document anchor must contain three documents")
    normalized: list[dict[str, object]] = []
    for index, document_id in enumerate(("main", "s1", "rebuttal")):
        record = _expect_mapping(records[index], context=f"document anchor[{index}]")
        if record.get("document_id") != document_id:
            raise PublicReleaseError("document anchor order is invalid")
        normalized.append(
            {
                "document_id": document_id,
                "member": _canonical_member(
                    record.get("member"),
                    context="document member",
                ),
                "bytes": _expect_int(
                    record.get("bytes"),
                    context="document bytes",
                    minimum=1,
                ),
                "sha256": _expect_sha256(
                    record.get("sha256"),
                    context="document sha256",
                ),
            },
        )
    return normalized


def _add_entry(entries: dict[str, _ArchiveEntry], entry: _ArchiveEntry) -> None:
    if entry.member in entries:
        raise PublicReleaseError(f"archive member is duplicated: {entry.member}")
    entries[entry.member] = entry


def _require_no_nonpublic_entry_bytes(
    entries: Mapping[str, _ArchiveEntry],
    nonpublic_byte_hashes: set[str],
) -> None:
    aliased = sorted(
        member
        for member, entry in entries.items()
        if entry.sha256 in nonpublic_byte_hashes
    )
    if aliased:
        raise PublicReleaseError(
            f"public archive entries alias nonpublic or restricted bytes: {aliased!r}",
        )


def _pin_expected_member(
    root: _PinnedRoot,
    record: Mapping[str, object],
    *,
    context: str,
    maximum: int | None = None,
) -> _PinnedFile:
    member = _canonical_member(record["member"], context=f"{context}.member")
    expected_size = _expect_int(
        record["bytes"],
        context=f"{context}.bytes",
    )
    expected_digest = _expect_sha256(
        record["sha256"],
        context=f"{context}.sha256",
    )
    pin = _pin_member(
        root,
        member,
        context=context,
        maximum=maximum,
        expected_size=expected_size,
    )
    if pin.sha256 != expected_digest:
        pin.close()
        raise PublicReleaseError(f"{context} does not match its authenticated record")
    return pin


def _validate_build_paths(
    config: PublicReleaseBuildConfig,
    plan: Mapping[str, object],
) -> None:
    for field_name in (
        "plan_path",
        "source_data_root",
        "artifact_registry_path",
        "release_evidence_path",
        "renderer_root",
        "rendered_output_root",
        "gate_receipt_root",
        "evidence_source_root",
        "document_reconciliation_path",
        "document_anchor_path",
        "document_root",
        "k500_authority_projection_path",
        "dependency_root",
        "repository_root",
        "destination_archive",
        "destination_receipt",
    ):
        _require_absolute_canonical(getattr(config, field_name), context=field_name)
    release = _expect_mapping(plan["release"], context="plan.release")
    if config.destination_archive.name != release["archive_name"]:
        raise PublicReleaseError("archive destination basename differs from the plan")
    if config.destination_receipt.name != release["receipt_name"]:
        raise PublicReleaseError("receipt destination basename differs from the plan")
    if config.destination_archive.parent != config.destination_receipt.parent:
        raise PublicReleaseError(
            "archive and receipt must share one destination parent",
        )
    if config.dependency_root != config.repository_root / "provenance/dependencies":
        raise PublicReleaseError("dependency root must be the repository ledger path")
    if config.plan_path in {config.destination_archive, config.destination_receipt}:
        raise PublicReleaseError("a destination cannot replace the public release plan")
    expected_projection = _expect_sha256(
        config.expected_k500_authority_projection_sha256,
        context="expected_k500_authority_projection_sha256",
    )
    anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
    if expected_projection != anchors["k500_authority_projection_sha256"]:
        raise PublicReleaseError(
            "independent K500 authority-projection anchor contradicts the plan",
        )


def _source_disposition_index(
    plan: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    index: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw in _expect_sequence(
        plan["source_dispositions"],
        context="plan.source_dispositions",
    ):
        record = _expect_mapping(raw, context="source disposition")
        identity = (str(record["root"]), str(record["source_member"]))
        if identity in index:
            raise PublicReleaseError("source disposition identity is duplicated")
        index[identity] = record
    return index


def _require_exact_code_closure(
    code_bytes: Mapping[str, bytes],
    plan: Mapping[str, object],
    renderer_members: set[str],
) -> None:
    execution_paths = {
        str(_expect_mapping(record, context="execution record")["path"])
        for record in _expect_sequence(
            _expect_mapping(plan["execution"], context="execution")["paths"],
            context="execution paths",
        )
    }
    expected = (
        set(REQUIRED_CODE_PATHS)
        | (execution_paths - RESTRICTED_EXECUTION_PATHS)
        | renderer_members
    )
    if set(code_bytes) != expected:
        raise PublicReleaseError(
            "code_paths is not the exact execution, builder, and renderer closure",
        )


def _excluded_evidence_source_hashes(
    records: Sequence[Mapping[str, object]],
    dispositions: Mapping[tuple[str, str], Mapping[str, object]],
) -> set[str]:
    return {
        str(record["sha256"])
        for record in records
        if dispositions[("evidence-source", str(record["member"]))]["disposition"]
        == "exclude"
    }


def _require_includable_source(
    disposition: Mapping[str, object],
    *,
    digest: str,
    excluded_hashes: set[str],
) -> None:
    dependencies = [
        str(value)
        for value in _expect_sequence(
            disposition["dependency_ids"],
            context="source dependency_ids",
        )
    ]
    if any(dependency != INCLUDED_DEPENDENCY_ID for dependency in dependencies):
        raise PublicReleaseError(
            "included source cites a non-redistributable dependency",
        )
    if digest in excluded_hashes:
        raise PublicReleaseError(
            "included source matches a forbidden dependency digest",
        )


def _manifest_and_checksums(  # noqa: PLR0913
    *,
    plan_file: _PinnedFile,
    plan: Mapping[str, object],
    entries: Mapping[str, _ArchiveEntry],
    source_lineage: Mapping[str, object],
    artifact_summary: Mapping[str, object],
    source_receipt: object,
    dependency_inventory_sha256: str,
    dependency_boundaries: Sequence[Mapping[str, object]],
    gate_receipts: Sequence[Mapping[str, object]],
    k500_projection_receipt: object,
) -> tuple[bytes, bytes]:
    member_records = [
        {
            "member": member,
            "role": entry.role,
            "bytes": entry.size_bytes,
            "sha256": entry.sha256,
            "origin": dict(entry.origin),
        }
        for member, entry in sorted(entries.items())
    ]
    documents = [
        dict(_expect_mapping(value, context="document disposition"))
        for value in _expect_sequence(plan["documents"], context="plan.documents")
    ]
    source_exclusions = [
        {
            "root": record["root"],
            "source_member": record["source_member"],
            "dependency_ids": list(
                _expect_sequence(record["dependency_ids"], context="dependency_ids"),
            ),
            "reason": record["reason"],
        }
        for record in (
            _expect_mapping(value, context="source disposition")
            for value in _expect_sequence(
                plan["source_dispositions"],
                context="source dispositions",
            )
        )
        if record["disposition"] == "exclude"
    ]
    source_values = _receipt_values(source_receipt, _SOURCE_RECEIPT_FIELDS)
    projection_values = _receipt_values(
        k500_projection_receipt,
        _PROJECTION_RECEIPT_FIELDS,
    )
    payload = {
        "schema": MANIFEST_SCHEMA,
        "contract": MANIFEST_CONTRACT,
        "package_kind": "public-release",
        "submission_package": {
            "created": False,
            "member_bytes_included": False,
        },
        "trust_model": dict(TRUST_MODEL),
        "release": dict(_expect_mapping(plan["release"], context="plan.release")),
        "plan": {
            "schema": PLAN_SCHEMA,
            "contract": PLAN_CONTRACT,
            "sha256": plan_file.sha256,
            "bytes": plan_file.size_bytes,
        },
        "anchors": dict(_expect_mapping(plan["anchors"], context="plan.anchors")),
        "approvals": [
            dict(_expect_mapping(value, context="approval"))
            for value in _expect_sequence(plan["approvals"], context="approvals")
        ],
        "source_lineage": dict(source_lineage),
        "source_data": {
            "manifest_sha256": source_values[1],
            "file_count": source_values[2],
            "cohort_count": source_values[3],
            "total_bytes": source_values[4],
            "total_rows": source_values[5],
        },
        "artifacts": dict(artifact_summary),
        "documents": documents,
        "source_exclusions": source_exclusions,
        "dependency_provenance": {
            "inventory_sha256": dependency_inventory_sha256,
            "metadata_file_count": len(DEPENDENCY_METADATA_MEMBERS),
            "boundaries": [dict(record) for record in dependency_boundaries],
        },
        "gate_receipts": {
            "policy": "hash-references-only; receipt bytes are not public members",
            "records": [dict(record) for record in gate_receipts],
        },
        "k500_authority": {
            "projection_sha256": projection_values[1],
            "completion_attestation_sha256": projection_values[2],
            "completion_attestation_payload_sha256": projection_values[3],
            "sealed_completion_sha256": projection_values[4],
            "run_manifest_sha256": projection_values[5],
            "source_a_commit": projection_values[6],
            "release_b_commit": projection_values[7],
            "release_tag": projection_values[8],
            "git_blob_count": projection_values[9],
            "generated_file_count": projection_values[10],
            "snapshot_file_count": projection_values[11],
            "execution_snapshot_sha256": projection_values[12],
            "revision_authority": dict(
                _expect_mapping(
                    projection_values[13],
                    context="K500 projection authority digests",
                ),
            ),
            "authority_digest_count": projection_values[14],
            "result_rows_opened": False,
        },
        "checksum_policy": {
            "algorithm": "sha256",
            "covered": "release_manifest.json and every payload member",
            "excluded": [ARCHIVE_CHECKSUM_MEMBER],
            "self_coverage": False,
        },
        "builder": {},
        "members": member_records,
        "inventory": {
            "payload_member_count": len(member_records),
            "archive_member_count": len(member_records) + 2,
            "payload_members_sha256": _sha256(_canonical_json(member_records)),
        },
    }
    builder_entry = entries.get(BUILDER_MEMBER)
    if builder_entry is None:
        raise PublicReleaseError("archive omits its public release builder")
    payload["builder"] = {
        "member": BUILDER_MEMBER,
        "bytes": builder_entry.size_bytes,
        "sha256": builder_entry.sha256,
    }
    manifest = {
        **payload,
        "manifest_payload_sha256": _sha256(_canonical_json(payload)),
    }
    manifest_raw = _canonical_json(manifest) + b"\n"
    checksum_records = [(entry.sha256, member) for member, entry in entries.items()] + [
        (_sha256(manifest_raw), ARCHIVE_MANIFEST_MEMBER),
    ]
    checksum_records.sort(key=lambda item: item[1])
    checksums_raw = "".join(
        f"{digest}  {member}\n" for digest, member in checksum_records
    ).encode("ascii")
    return manifest_raw, checksums_raw


def _prepare_release(
    config: PublicReleaseBuildConfig,
    plan_file: _PinnedFile,
    plan: dict[str, object],
) -> _PreparedRelease:
    native_receipts = _run_native_validators(config, plan)
    roots: list[_PinnedRoot] = []
    metadata_files: list[_PinnedFile] = []
    member_files: list[_PinnedFile] = []
    try:
        root_by_name = {
            "source-data": _pin_root(
                config.source_data_root,
                context="source-data root",
            ),
            "renderer": _pin_root(config.renderer_root, context="renderer root"),
            "rendered-output": _pin_root(
                config.rendered_output_root,
                context="rendered-output root",
            ),
            "gate-receipt": _pin_root(
                config.gate_receipt_root,
                context="gate-receipt root",
            ),
            "evidence-source": _pin_root(
                config.evidence_source_root,
                context="evidence-source root",
            ),
            "document": _pin_root(config.document_root, context="document root"),
            "dependency": _pin_root(config.dependency_root, context="dependency root"),
            "repository": _pin_root(config.repository_root, context="repository root"),
        }
        roots.extend(root_by_name.values())
        anchors = _expect_mapping(plan["anchors"], context="plan.anchors")
        for path, context, expected_key in (
            (
                config.artifact_registry_path,
                "artifact registry",
                "artifact_registry_sha256",
            ),
            (
                config.release_evidence_path,
                "release-evidence closure",
                "release_evidence_sha256",
            ),
            (
                config.document_reconciliation_path,
                "document reconciliation",
                "document_reconciliation_sha256",
            ),
            (config.document_anchor_path, "document anchor", "document_anchor_sha256"),
            (
                config.k500_authority_projection_path,
                "K500 authority projection",
                "k500_authority_projection_sha256",
            ),
        ):
            pin = _pin_absolute_file(path, context=context, maximum=MAX_METADATA_BYTES)
            metadata_files.append(pin)
            if pin.sha256 != anchors[expected_key]:
                raise PublicReleaseError(f"{context} changed after native validation")
        (
            registry_pin,
            closure_pin,
            reconciliation_pin,
            document_anchor_pin,
            projection_pin,
        ) = metadata_files
        registry_value = _parse_canonical(
            _read_descriptor(registry_pin, maximum=MAX_METADATA_BYTES),
            context="artifact registry",
        )
        closure_value = _parse_canonical(
            _read_descriptor(closure_pin, maximum=MAX_METADATA_BYTES),
            context="release-evidence closure",
        )
        reconciliation_value = _parse_canonical(
            _read_descriptor(reconciliation_pin, maximum=MAX_METADATA_BYTES),
            context="document reconciliation",
        )
        document_anchor_value = _parse_canonical(
            _read_descriptor(document_anchor_pin, maximum=MAX_METADATA_BYTES),
            context="document anchor",
        )
        projection_value = _parse_canonical(
            _read_descriptor(projection_pin, maximum=MAX_METADATA_BYTES),
            context="K500 authority projection",
        )
        git_executable = _pin_absolute_file(
            GIT_EXECUTABLE,
            context="Git executable",
            maximum=MAX_GIT_EXECUTABLE_BYTES,
            require_single_link=False,
        )
        metadata_files.append(git_executable)
        _require_projection_git_executable(projection_value, git_executable)

        source_manifest_pin = _pin_member(
            root_by_name["source-data"],
            source_data.SOURCE_DATA_MANIFEST_NAME,
            context="source-data manifest",
            maximum=MAX_METADATA_BYTES,
        )
        member_files.append(source_manifest_pin)
        if source_manifest_pin.sha256 != anchors["source_data_manifest_sha256"]:
            raise PublicReleaseError(
                "source-data manifest changed after native validation",
            )
        source_manifest = _parse_canonical(
            _read_descriptor(source_manifest_pin, maximum=MAX_METADATA_BYTES),
            context="source-data manifest",
        )
        source_records, total_rows, source_total_bytes = _source_data_inventory(
            source_manifest,
            source_manifest_pin,
            anchors,
        )
        source_receipt_values = _receipt_values(
            native_receipts[0],
            _SOURCE_RECEIPT_FIELDS,
        )
        if (
            source_receipt_values[2] != SOURCE_DATA_FILE_COUNT
            or source_receipt_values[3] != SOURCE_DATA_COHORT_COUNT
            or source_receipt_values[4] != source_total_bytes
            or source_receipt_values[5] != total_rows
        ):
            raise PublicReleaseError(
                "source-data manifest contradicts its native receipt",
            )

        outputs, renderer_members, artifact_summary = _registry_records(
            registry_value,
            plan,
        )
        registry_receipt_values = _receipt_values(
            native_receipts[1],
            _REGISTRY_RECEIPT_FIELDS,
        )
        if (
            registry_receipt_values[2] != artifact_summary["ready_count"]
            or registry_receipt_values[3] != artifact_summary["omitted_count"]
        ):
            raise PublicReleaseError("artifact registry contradicts its native receipt")
        closure_sources, gate_records = _closure_records(closure_value, plan)
        closure_receipt_values = _receipt_values(
            native_receipts[2],
            _CLOSURE_RECEIPT_FIELDS,
        )
        if (
            closure_receipt_values[2] != len(gate_records)
            or closure_receipt_values[3] != len(closure_sources)
            or closure_receipt_values[4] != artifact_summary["ready_count"]
            or closure_receipt_values[5] != artifact_summary["omitted_count"]
        ):
            raise PublicReleaseError("release-evidence closure contradicts its receipt")
        k500_gate = next(record for record in gate_records if record["gate"] == "K500")
        completion_attestation_pin = _pin_expected_member(
            root_by_name["gate-receipt"],
            k500_gate,
            context="K500 completion attestation",
            maximum=MAX_METADATA_BYTES,
        )
        metadata_files.append(completion_attestation_pin)
        attestation_summary = _validate_completion_attestation_metadata(
            _parse_canonical(
                _read_descriptor(
                    completion_attestation_pin,
                    maximum=MAX_METADATA_BYTES,
                ),
                context="K500 completion attestation",
            ),
            plan,
        )
        projection_receipt_values = _receipt_values(
            native_receipts[4],
            _PROJECTION_RECEIPT_FIELDS,
        )
        _require_projection_source_authority(source_manifest, native_receipts[4])
        if (
            projection_receipt_values[2] != completion_attestation_pin.sha256
            or projection_receipt_values[3]
            != attestation_summary["completion_attestation_payload_sha256"]
        ):
            raise PublicReleaseError(
                "K500 completion attestation contradicts its native projection",
            )
        document_records = _document_records(
            reconciliation_value,
            document_anchor_value,
            plan,
        )

        code_bytes, source_lineage = _validate_git_lineage(
            root_by_name["repository"],
            git_executable,
            plan,
        )
        _require_exact_code_closure(code_bytes, plan, renderer_members)
        release_b = str(
            _expect_mapping(plan["release"], context="release")["release_commit_b"],
        )
        (
            dependency_pins,
            dependency_entries,
            cbase_bytes,
            excluded_hashes,
            dependency_boundaries,
        ) = _validate_dependency_ledger(
            root_by_name["dependency"],
            root_by_name["repository"],
            git_executable,
            release_b,
            str(anchors["dependency_inventory_sha256"]),
            _expect_sequence(
                source_lineage["restricted_execution_paths"],
                context="verified restricted execution paths",
            ),
        )
        member_files.extend(dependency_pins)
        nonpublic_byte_hashes = (
            excluded_hashes
            | {str(record["sha256"]) for record in gate_records}
            | {str(record["sha256"]) for record in document_records}
        )

        entries: dict[str, _ArchiveEntry] = {}
        disposition_by_identity = _source_disposition_index(plan)
        expected_source_identities = {
            ("source-data", str(record["member"])) for record in source_records
        } | {("evidence-source", str(record["member"])) for record in closure_sources}
        if set(disposition_by_identity) != expected_source_identities:
            raise PublicReleaseError(
                "source dispositions do not exactly cover both source roots",
            )
        nonpublic_byte_hashes.update(
            _excluded_evidence_source_hashes(
                closure_sources,
                disposition_by_identity,
            ),
        )

        for record in source_records:
            member = str(record["member"])
            disposition = disposition_by_identity[("source-data", member)]
            expected_release_member = f"source-data/{member}"
            if (
                disposition["disposition"] != "include"
                or disposition["release_member"] != expected_release_member
                or list(
                    _expect_sequence(
                        disposition["dependency_ids"],
                        context="source-data dependency ids",
                    ),
                )
            ):
                raise PublicReleaseError(
                    "complete source-data release must be explicitly included",
                )
            if member == source_data.SOURCE_DATA_MANIFEST_NAME:
                pin = source_manifest_pin
            else:
                pin = _pin_expected_member(
                    root_by_name["source-data"],
                    record,
                    context=f"source-data {member}",
                )
                member_files.append(pin)
            _add_entry(
                entries,
                _PinnedEntry(
                    member=expected_release_member,
                    pinned=pin,
                    role=str(record["role"]),
                    origin={"kind": "source-data-root", "member": member},
                ),
            )

        for record in closure_sources:
            member = str(record["member"])
            disposition = disposition_by_identity[("evidence-source", member)]
            if disposition["disposition"] == "exclude":
                continue
            if disposition["disposition"] != "include":
                raise PublicReleaseError(
                    "final evidence source disposition is not closed",
                )
            _require_includable_source(
                disposition,
                digest=str(record["sha256"]),
                excluded_hashes=nonpublic_byte_hashes,
            )
            if disposition["release_member"] != member:
                raise PublicReleaseError(
                    "evidence source cannot be renamed in the public release",
                )
            pin = _pin_expected_member(
                root_by_name["evidence-source"],
                record,
                context=f"evidence source {member}",
            )
            member_files.append(pin)
            _add_entry(
                entries,
                _PinnedEntry(
                    member=member,
                    pinned=pin,
                    role="release-evidence-source",
                    origin={"kind": "evidence-source-root", "member": member},
                ),
            )

        renderer_sha_by_member: dict[str, str] = {}
        for record in outputs:
            member = str(record["member"])
            pin = _pin_expected_member(
                root_by_name["rendered-output"],
                record,
                context=f"rendered output {member}",
            )
            member_files.append(pin)
            if pin.sha256 in nonpublic_byte_hashes:
                raise PublicReleaseError(
                    "rendered output matches a nonpublic or forbidden byte digest",
                )
            renderer_member = str(record["renderer_member"])
            renderer_sha = str(record["renderer_sha256"])
            previous = renderer_sha_by_member.setdefault(renderer_member, renderer_sha)
            if (
                previous != renderer_sha
                or _sha256(code_bytes[renderer_member]) != renderer_sha
            ):
                raise PublicReleaseError(
                    "ready renderer differs from the release-B code blob",
                )
            _add_entry(
                entries,
                _PinnedEntry(
                    member=member,
                    pinned=pin,
                    role="rendered-output",
                    origin={
                        "kind": "artifact-output",
                        "artifact_id": record["artifact_id"],
                        "media_type": record["media_type"],
                    },
                ),
            )

        if {str(record["document_id"]) for record in document_records} != {
            "main",
            "s1",
            "rebuttal",
        }:
            raise PublicReleaseError("source-document anchor coverage changed")

        for member, raw in sorted(code_bytes.items()):
            _add_entry(
                entries,
                _MemoryEntry(
                    member=member,
                    raw=raw,
                    role="release-code",
                    origin={"kind": "git-blob", "commit": release_b, "path": member},
                ),
            )
        for member, raw in sorted(cbase_bytes.items()):
            _add_entry(
                entries,
                _MemoryEntry(
                    member=member,
                    raw=raw,
                    role="redistributable-dependency-code",
                    origin={
                        "kind": "dependency-record",
                        "dependency_id": INCLUDED_DEPENDENCY_ID,
                        "commit": release_b,
                        "path": member,
                    },
                ),
            )
        for entry in dependency_entries:
            _add_entry(entries, entry)
        for member, pin, role in (
            (PUBLIC_PLAN_MEMBER, plan_file, "release-plan"),
            (PUBLIC_REGISTRY_MEMBER, registry_pin, "artifact-registry"),
            (PUBLIC_CLOSURE_MEMBER, closure_pin, "release-evidence"),
            (
                PUBLIC_PROJECTION_MEMBER,
                projection_pin,
                "k500-authority-projection",
            ),
        ):
            _add_entry(
                entries,
                _PinnedEntry(
                    member=member,
                    pinned=pin,
                    role=role,
                    origin={"kind": "anchored-metadata", "sha256": pin.sha256},
                ),
            )

        _require_no_nonpublic_entry_bytes(entries, nonpublic_byte_hashes)
        if len(entries) + 2 > MAX_ARCHIVE_MEMBERS:
            raise PublicReleaseError("public archive exceeds its member-count limit")
        manifest_raw, checksums_raw = _manifest_and_checksums(
            plan_file=plan_file,
            plan=plan,
            entries=entries,
            source_lineage=source_lineage,
            artifact_summary=artifact_summary,
            source_receipt=native_receipts[0],
            dependency_inventory_sha256=str(anchors["dependency_inventory_sha256"]),
            dependency_boundaries=dependency_boundaries,
            gate_receipts=gate_records,
            k500_projection_receipt=native_receipts[4],
        )
        return _PreparedRelease(
            config=config,
            plan_file=plan_file,
            metadata_files=tuple(metadata_files),
            roots=tuple(roots),
            member_files=tuple(member_files),
            plan=plan,
            entries=tuple(entries[member] for member in sorted(entries)),
            manifest_raw=manifest_raw,
            checksums_raw=checksums_raw,
            native_receipts=native_receipts,
            dependency_inventory_sha256=str(anchors["dependency_inventory_sha256"]),
            dependency_root=root_by_name["dependency"],
            repository_root=root_by_name["repository"],
            git_executable=git_executable,
        )
    except Exception:
        for pin in member_files:
            with contextlib.suppress(OSError):
                pin.close()
        for pin in metadata_files:
            with contextlib.suppress(OSError):
                pin.close()
        for root in roots:
            with contextlib.suppress(OSError):
                root.close()
        raise


def _revalidate_prepared(prepared: _PreparedRelease) -> None:
    _revalidate_file(prepared.plan_file, context="public release plan")
    for index, pin in enumerate(prepared.metadata_files):
        _revalidate_file(pin, context=f"release metadata[{index}]")
    for index, root in enumerate(prepared.roots):
        _revalidate_root(root, context=f"release input root[{index}]")
    for index, pin in enumerate(prepared.member_files):
        _revalidate_file(pin, context=f"public member input[{index}]")
    observed_receipts = _run_native_validators(prepared.config, prepared.plan)
    for observed, expected, fields in zip(
        observed_receipts,
        prepared.native_receipts,
        (
            _SOURCE_RECEIPT_FIELDS,
            _REGISTRY_RECEIPT_FIELDS,
            _CLOSURE_RECEIPT_FIELDS,
            _DOCUMENT_RECEIPT_FIELDS,
            _PROJECTION_RECEIPT_FIELDS,
        ),
        strict=True,
    ):
        if _receipt_values(observed, fields) != _receipt_values(expected, fields):
            raise PublicReleaseError(
                "native validation receipt changed at publication boundary",
            )
    _validate_git_lineage(
        prepared.repository_root,
        prepared.git_executable,
        prepared.plan,
    )
    _require_exact_root_inventory(
        prepared.dependency_root,
        set(DEPENDENCY_METADATA_MEMBERS),
        context="dependency ledger at publication boundary",
    )


@dataclass(frozen=True, slots=True)
class _VerifiedArchive:
    archive_sha256: str
    archive_bytes: int
    manifest_sha256: str
    release_id: str
    member_count: int


@dataclass(frozen=True, slots=True)
class _DirectorySnapshot:
    members: tuple[str, ...]
    stat_signature: tuple[int, int, int, int, int, int]


class _DescriptorReader(io.RawIOBase):
    """Expose a bounded, seekable descriptor view without taking FD ownership."""

    def __init__(self, descriptor: int, size_bytes: int) -> None:
        super().__init__()
        self._descriptor = descriptor
        self._size_bytes = size_bytes
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size_bytes + offset
        else:
            raise ValueError("invalid seek mode")
        if position < 0:
            raise ValueError("negative descriptor-reader position")
        self._position = position
        return position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        if self._position >= self._size_bytes:
            return 0
        view = memoryview(buffer)
        count = min(len(view), self._size_bytes - self._position)
        raw = os.pread(self._descriptor, count, self._position)
        view[: len(raw)] = raw
        self._position += len(raw)
        return len(raw)


def _tar_info(member: str, size_bytes: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(member)
    info.size = size_bytes
    info.mode = ARCHIVE_FILE_MODE
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.linkname = ""
    return info


def _archive_entries(prepared: _PreparedRelease) -> tuple[_ArchiveEntry, ...]:
    entries: list[_ArchiveEntry] = [
        *prepared.entries,
        _MemoryEntry(
            member=ARCHIVE_MANIFEST_MEMBER,
            raw=prepared.manifest_raw,
            role="release-manifest",
            origin={"kind": "archive-metadata"},
        ),
        _MemoryEntry(
            member=ARCHIVE_CHECKSUM_MEMBER,
            raw=prepared.checksums_raw,
            role="archive-checksums",
            origin={"kind": "archive-metadata"},
        ),
    ]
    entries.sort(key=lambda entry: entry.member)
    members = [entry.member for entry in entries]
    if len(members) != len(set(members)):
        raise PublicReleaseError("archive inventory contains duplicate members")
    return tuple(entries)


def _write_archive(descriptor: int, prepared: _PreparedRelease) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    duplicate = os.dup(descriptor)
    with os.fdopen(duplicate, "wb", closefd=True) as output:
        with tarfile.open(
            fileobj=output,
            mode="w",
            format=tarfile.USTAR_FORMAT,
            encoding="ascii",
            errors="strict",
        ) as archive:
            for entry in _archive_entries(prepared):
                info = _tar_info(entry.member, entry.size_bytes)
                if isinstance(entry, _MemoryEntry):
                    source: BinaryIO = io.BytesIO(entry.raw)
                else:
                    source = io.BufferedReader(
                        _DescriptorReader(entry.pinned.descriptor, entry.size_bytes),
                        buffer_size=READ_CHUNK_BYTES,
                    )
                try:
                    archive.addfile(info, source)
                finally:
                    source.close()
        output.flush()
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)


def _pread_exact(
    descriptor: int,
    size_bytes: int,
    offset: int,
    *,
    context: str,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size_bytes
    position = offset
    while remaining:
        raw = os.pread(descriptor, min(remaining, READ_CHUNK_BYTES), position)
        if not raw:
            raise PublicReleaseError(f"{context} is truncated")
        chunks.append(raw)
        remaining -= len(raw)
        position += len(raw)
    return b"".join(chunks)


def _hash_archive_payload(
    descriptor: int,
    *,
    offset: int,
    size_bytes: int,
    capture: bool,
) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    remaining = size_bytes
    position = offset
    captured = bytearray() if capture else None
    while remaining:
        raw = os.pread(descriptor, min(remaining, READ_CHUNK_BYTES), position)
        if not raw:
            raise PublicReleaseError("archive payload is truncated")
        digest.update(raw)
        if captured is not None:
            captured.extend(raw)
        remaining -= len(raw)
        position += len(raw)
    return digest.hexdigest(), bytes(captured) if captured is not None else None


def _parse_checksum_file(raw: bytes) -> list[tuple[str, str]]:
    try:
        text_value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise PublicReleaseError("SHA256SUMS is not ASCII") from error
    if not text_value or not text_value.endswith("\n"):
        raise PublicReleaseError("SHA256SUMS must be nonempty and newline-terminated")
    records: list[tuple[str, str]] = []
    for index, line in enumerate(text_value.splitlines()):
        if len(line) < 67 or line[64:66] != "  ":
            raise PublicReleaseError(f"SHA256SUMS line {index} is malformed")
        digest = _expect_sha256(line[:64], context=f"SHA256SUMS line {index}")
        member = _canonical_member(
            line[66:],
            context=f"SHA256SUMS line {index} member",
        )
        records.append((digest, member))
    if records != sorted(records, key=lambda item: item[1]):
        raise PublicReleaseError("SHA256SUMS is not canonically sorted")
    members = [member for _, member in records]
    if len(members) != len(set(members)):
        raise PublicReleaseError("SHA256SUMS contains duplicate members")
    canonical = "".join(f"{digest}  {member}\n" for digest, member in records).encode(
        "ascii",
    )
    if raw != canonical:
        raise PublicReleaseError("SHA256SUMS is not canonical ASCII text")
    return records


def _validate_cbase_archive_boundary(
    *,
    release: Mapping[str, object],
    manifest_records: Sequence[Mapping[str, object]],
    actual_records: Mapping[str, Mapping[str, object]],
    captured_payloads: Mapping[str, bytes],
) -> None:
    """Reproduce the CBaSE boundary from a downloaded public archive."""
    try:
        cbase_raw = captured_payloads[CBASE_RECORD_MEMBER]
        license_record = actual_records["LICENSE"]
        cbase_actual = {
            member: actual_records[member]["sha256"] for member in CBASE_RELEASE_MEMBERS
        }
    except KeyError as error:
        raise PublicReleaseError(
            "downloaded archive omits the CBaSE license or provenance boundary",
        ) from error
    cbase = _parse_json(cbase_raw, context=CBASE_RECORD_MEMBER)
    _validate_cbase_release_boundary(
        cbase,
        dialect_license_sha256=_expect_sha256(
            license_record.get("sha256"),
            context="downloaded root LICENSE",
        ),
        release_member_sha256=cbase_actual,
        archive_members=tuple(actual_records),
    )

    records_by_member = {str(record["member"]): record for record in manifest_records}
    release_b = _expect_git_sha(
        release.get("release_commit_b"),
        context="downloaded release commit B",
    )
    expected_dependency_origin = {
        "kind": "dependency-record",
        "dependency_id": INCLUDED_DEPENDENCY_ID,
        "commit": release_b,
    }
    for member in CBASE_RELEASE_MEMBERS:
        record = records_by_member.get(member)
        if record is None:
            raise PublicReleaseError("downloaded archive omits a CBaSE release member")
        expected_origin = {**expected_dependency_origin, "path": member}
        if (
            record.get("role") != "redistributable-dependency-code"
            or record.get("origin") != expected_origin
        ):
            raise PublicReleaseError(
                f"downloaded CBaSE member metadata is invalid: {member}",
            )

    provenance_record = records_by_member.get(CBASE_RECORD_MEMBER)
    if provenance_record is None or (
        provenance_record.get("role") != "dependency-provenance"
        or provenance_record.get("origin")
        != {
            "kind": "dependency-ledger",
            "member": f"{INCLUDED_DEPENDENCY_ID}.json",
        }
    ):
        raise PublicReleaseError("downloaded CBaSE provenance metadata is invalid")
    license_manifest_record = records_by_member.get("LICENSE")
    if license_manifest_record is None or (
        license_manifest_record.get("role") != "release-code"
        or license_manifest_record.get("origin")
        != {"kind": "git-blob", "commit": release_b, "path": "LICENSE"}
    ):
        raise PublicReleaseError("downloaded root LICENSE metadata is invalid")


def _validate_release_manifest(
    raw: bytes,
    *,
    actual_records: Mapping[str, Mapping[str, object]],
    captured_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    manifest = _parse_canonical(raw, context=ARCHIVE_MANIFEST_MEMBER)
    expected_keys = {
        "schema",
        "contract",
        "package_kind",
        "submission_package",
        "trust_model",
        "release",
        "plan",
        "anchors",
        "approvals",
        "source_lineage",
        "source_data",
        "artifacts",
        "documents",
        "source_exclusions",
        "dependency_provenance",
        "gate_receipts",
        "k500_authority",
        "checksum_policy",
        "builder",
        "members",
        "inventory",
        "manifest_payload_sha256",
    }
    _expect_keys(manifest, expected_keys, context="release manifest")
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["contract"] != MANIFEST_CONTRACT
        or manifest["package_kind"] != "public-release"
        or manifest["submission_package"]
        != {"created": False, "member_bytes_included": False}
        or manifest["trust_model"] != TRUST_MODEL
    ):
        raise PublicReleaseError("release manifest identity or trust model is invalid")
    payload = dict(manifest)
    payload_sha256 = _expect_sha256(
        payload.pop("manifest_payload_sha256"),
        context="manifest_payload_sha256",
    )
    if payload_sha256 != _sha256(_canonical_json(payload)):
        raise PublicReleaseError("release manifest payload digest is invalid")
    checksum_policy = _expect_mapping(
        manifest["checksum_policy"],
        context="manifest checksum_policy",
    )
    if checksum_policy != {
        "algorithm": "sha256",
        "covered": "release_manifest.json and every payload member",
        "excluded": [ARCHIVE_CHECKSUM_MEMBER],
        "self_coverage": False,
    }:
        raise PublicReleaseError("release manifest checksum policy changed")
    release = _expect_mapping(manifest["release"], context="manifest release")
    if dict(release) != _normalize_release(release, mode="final"):
        raise PublicReleaseError("manifest release identity is not canonical")
    anchors = _expect_mapping(manifest["anchors"], context="manifest anchors")
    if dict(anchors) != _normalize_anchors(anchors):
        raise PublicReleaseError("manifest anchors are not canonical")
    approvals, approval_pending = _normalize_approvals(
        manifest["approvals"],
        mode="final",
    )
    if approval_pending or list(manifest["approvals"]) != approvals:
        raise PublicReleaseError("manifest approvals are not closed")
    documents, document_pending = _normalize_documents(
        manifest["documents"],
        mode="final",
    )
    if document_pending or list(manifest["documents"]) != documents:
        raise PublicReleaseError(
            "v1 public archive does not explicitly exclude all document bytes",
        )

    members = _expect_sequence(manifest["members"], context="manifest members")
    normalized_records: list[dict[str, object]] = []
    for index, raw_record in enumerate(members):
        record = _expect_mapping(raw_record, context=f"manifest member[{index}]")
        _expect_keys(
            record,
            {"member", "role", "bytes", "sha256", "origin"},
            context=f"manifest member[{index}]",
        )
        member = _validate_public_member(
            record["member"],
            context=f"manifest member[{index}].member",
        )
        normalized_records.append(
            {
                "member": member,
                "role": _expect_string(
                    record["role"],
                    context=f"manifest member[{index}].role",
                ),
                "bytes": _expect_int(
                    record["bytes"],
                    context=f"manifest member[{index}].bytes",
                ),
                "sha256": _expect_sha256(
                    record["sha256"],
                    context=f"manifest member[{index}].sha256",
                ),
                "origin": dict(
                    _expect_mapping(
                        record["origin"],
                        context=f"manifest member[{index}].origin",
                    ),
                ),
            },
        )
    if list(members) != sorted(normalized_records, key=lambda record: record["member"]):
        raise PublicReleaseError("manifest member records are not canonical")
    names = [str(record["member"]) for record in normalized_records]
    if len(names) != len(set(names)):
        raise PublicReleaseError("manifest member records are duplicated")
    actual_payload_names = set(actual_records) - {
        ARCHIVE_MANIFEST_MEMBER,
        ARCHIVE_CHECKSUM_MEMBER,
    }
    if set(names) != actual_payload_names:
        raise PublicReleaseError(
            "manifest does not close the archive payload inventory",
        )
    for record in normalized_records:
        actual = actual_records[str(record["member"])]
        if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
            raise PublicReleaseError(
                "manifest payload record contradicts archive bytes",
            )
    _validate_cbase_archive_boundary(
        release=release,
        manifest_records=normalized_records,
        actual_records=actual_records,
        captured_payloads=captured_payloads,
    )

    def require_bound_member(
        member: str,
        digest: object,
        *,
        size_bytes: object | None = None,
        context: str,
    ) -> None:
        actual = actual_records.get(member)
        if actual is None or actual["sha256"] != _expect_sha256(
            digest,
            context=f"{context} sha256",
        ):
            raise PublicReleaseError(f"{context} is not bound to its archive member")
        if size_bytes is not None and actual["bytes"] != _expect_int(
            size_bytes,
            context=f"{context} bytes",
        ):
            raise PublicReleaseError(f"{context} size differs from its archive member")

    plan_binding = _expect_mapping(manifest["plan"], context="manifest plan")
    _expect_keys(
        plan_binding,
        {"schema", "contract", "sha256", "bytes"},
        context="manifest plan",
    )
    if (
        plan_binding["schema"] != PLAN_SCHEMA
        or plan_binding["contract"] != PLAN_CONTRACT
    ):
        raise PublicReleaseError("manifest plan contract is invalid")
    require_bound_member(
        PUBLIC_PLAN_MEMBER,
        plan_binding["sha256"],
        size_bytes=plan_binding["bytes"],
        context="public release plan",
    )
    builder = _expect_mapping(manifest["builder"], context="manifest builder")
    _expect_keys(
        builder,
        {"member", "bytes", "sha256"},
        context="manifest builder",
    )
    if builder["member"] != BUILDER_MEMBER:
        raise PublicReleaseError("manifest builder member is invalid")
    require_bound_member(
        BUILDER_MEMBER,
        builder["sha256"],
        size_bytes=builder["bytes"],
        context="public release builder",
    )
    for member, anchor_key, context in (
        (
            f"source-data/{source_data.SOURCE_DATA_MANIFEST_NAME}",
            "source_data_manifest_sha256",
            "source-data manifest",
        ),
        (PUBLIC_REGISTRY_MEMBER, "artifact_registry_sha256", "artifact registry"),
        (PUBLIC_CLOSURE_MEMBER, "release_evidence_sha256", "release evidence"),
        (
            PUBLIC_PROJECTION_MEMBER,
            "k500_authority_projection_sha256",
            "K500 authority projection",
        ),
    ):
        require_bound_member(member, anchors[anchor_key], context=context)
    source_summary = _expect_mapping(
        manifest["source_data"],
        context="manifest source_data",
    )
    _expect_keys(
        source_summary,
        {
            "manifest_sha256",
            "file_count",
            "cohort_count",
            "total_bytes",
            "total_rows",
        },
        context="manifest source_data",
    )
    if (
        source_summary["manifest_sha256"] != anchors["source_data_manifest_sha256"]
        or source_summary["file_count"] != SOURCE_DATA_FILE_COUNT
        or source_summary["cohort_count"] != SOURCE_DATA_COHORT_COUNT
    ):
        raise PublicReleaseError("manifest source-data summary is invalid")
    _expect_int(source_summary["total_bytes"], context="source-data total_bytes")
    _expect_int(source_summary["total_rows"], context="source-data total_rows")
    k500_authority = _expect_mapping(
        manifest["k500_authority"],
        context="manifest K500 authority",
    )
    _expect_keys(
        k500_authority,
        {
            "projection_sha256",
            "completion_attestation_sha256",
            "completion_attestation_payload_sha256",
            "sealed_completion_sha256",
            "run_manifest_sha256",
            "source_a_commit",
            "release_b_commit",
            "release_tag",
            "git_blob_count",
            "generated_file_count",
            "snapshot_file_count",
            "execution_snapshot_sha256",
            "revision_authority",
            "authority_digest_count",
            "result_rows_opened",
        },
        context="manifest K500 authority",
    )
    for field_name in (
        "projection_sha256",
        "completion_attestation_sha256",
        "completion_attestation_payload_sha256",
        "sealed_completion_sha256",
        "run_manifest_sha256",
        "execution_snapshot_sha256",
    ):
        _expect_sha256(
            k500_authority[field_name],
            context=f"manifest K500 {field_name}",
        )
    if (
        k500_authority.get("projection_sha256")
        != anchors["k500_authority_projection_sha256"]
        or k500_authority.get("completion_attestation_sha256")
        != anchors["completion_attestation_sha256"]
        or k500_authority.get("sealed_completion_sha256")
        != anchors["sealed_completion_sha256"]
        or k500_authority.get("source_a_commit") != release["source_commit_a"]
        or k500_authority.get("release_b_commit") != release["release_commit_b"]
        or k500_authority.get("release_tag") != release["source_tag"]
        or k500_authority.get("git_blob_count") != 38
        or k500_authority.get("generated_file_count") != 1
        or k500_authority.get("snapshot_file_count") != 39
        or k500_authority.get("execution_snapshot_sha256") != EXECUTION_SNAPSHOT_SHA256
        or k500_authority.get("authority_digest_count") != 6
        or k500_authority.get("result_rows_opened") is not False
    ):
        raise PublicReleaseError("manifest K500 authority binding is invalid")
    revision_authority = _expect_mapping(
        k500_authority.get("revision_authority"),
        context="manifest K500 revision authority",
    )
    if set(revision_authority) != set(
        k500_authority_projection.AUTHORITY_DIGEST_FIELDS,
    ) or any(
        _SHA256_RE.fullmatch(str(value)) is None
        for value in revision_authority.values()
    ):
        raise PublicReleaseError("manifest K500 authority digest closure is invalid")
    inventory = _expect_mapping(manifest["inventory"], context="manifest inventory")
    expected_inventory = {
        "payload_member_count": len(normalized_records),
        "archive_member_count": len(normalized_records) + 2,
        "payload_members_sha256": _sha256(_canonical_json(normalized_records)),
    }
    if inventory != expected_inventory:
        raise PublicReleaseError("manifest inventory summary is invalid")
    return manifest


def _verify_archive_descriptor(
    pinned: _PinnedFile,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> _VerifiedArchive:
    outer_digest = _expect_sha256(
        expected_archive_sha256,
        context="expected_archive_sha256",
    )
    manifest_digest = _expect_sha256(
        expected_manifest_sha256,
        context="expected_manifest_sha256",
    )
    if pinned.sha256 != outer_digest:
        raise PublicReleaseError("downloaded archive fails its outer SHA-256 anchor")
    _revalidate_file(pinned, context="downloaded public archive before TAR parsing")
    records: dict[str, dict[str, object]] = {}
    metadata: dict[str, bytes] = {}
    captured_members = frozenset(
        {ARCHIVE_MANIFEST_MEMBER, ARCHIVE_CHECKSUM_MEMBER, CBASE_RECORD_MEMBER},
    )
    names: list[str] = []
    offset = 0
    zero = b"\0" * TAR_BLOCK_BYTES
    while True:
        header = _pread_exact(
            pinned.descriptor,
            TAR_BLOCK_BYTES,
            offset,
            context="TAR header",
        )
        offset += TAR_BLOCK_BYTES
        if header == zero:
            second = _pread_exact(
                pinned.descriptor,
                TAR_BLOCK_BYTES,
                offset,
                context="second TAR end block",
            )
            if second != zero:
                raise PublicReleaseError("archive has only one TAR end block")
            offset += TAR_BLOCK_BYTES
            break
        if len(records) >= MAX_ARCHIVE_MEMBERS:
            raise PublicReleaseError("archive exceeds the member-count limit")
        try:
            info = tarfile.TarInfo.frombuf(header, "ascii", "strict")
        except (tarfile.TarError, UnicodeError, ValueError) as error:
            raise PublicReleaseError(
                "archive contains an invalid TAR header",
            ) from error
        member = _canonical_member(info.name, context="TAR member")
        if (
            info.type != tarfile.REGTYPE
            or info.pax_headers
            or info.mode != ARCHIVE_FILE_MODE
            or info.uid != 0
            or info.gid != 0
            or info.uname != ""
            or info.gname != ""
            or info.mtime != 0
            or info.linkname != ""
            or info.devmajor != 0
            or info.devminor != 0
        ):
            raise PublicReleaseError(
                "archive member metadata differs from the immutable USTAR contract",
            )
        if (
            info.tobuf(
                format=tarfile.USTAR_FORMAT,
                encoding="ascii",
                errors="strict",
            )
            != header
        ):
            raise PublicReleaseError("archive contains a non-canonical USTAR header")
        if member in records:
            raise PublicReleaseError("archive contains a duplicate member")
        if names and member <= names[-1]:
            raise PublicReleaseError("archive members are not ASCII sorted")
        names.append(member)
        capture = member in captured_members
        if capture and info.size > MAX_ARCHIVE_METADATA_MEMBER_BYTES:
            raise PublicReleaseError("archive metadata member is too large")
        digest, raw = _hash_archive_payload(
            pinned.descriptor,
            offset=offset,
            size_bytes=info.size,
            capture=capture,
        )
        records[member] = {"bytes": info.size, "sha256": digest}
        if raw is not None:
            metadata[member] = raw
        offset += info.size
        padding = (-info.size) % TAR_BLOCK_BYTES
        if padding:
            observed_padding = _pread_exact(
                pinned.descriptor,
                padding,
                offset,
                context="TAR member padding",
            )
            if observed_padding != b"\0" * padding:
                raise PublicReleaseError("archive contains nonzero member padding")
            offset += padding
    canonical_size = (
        (offset + TAR_RECORD_BYTES - 1) // TAR_RECORD_BYTES
    ) * TAR_RECORD_BYTES
    if pinned.size_bytes != canonical_size:
        raise PublicReleaseError(
            "archive has trailing, concatenated, or noncanonical bytes",
        )
    if offset < pinned.size_bytes:
        tail = _pread_exact(
            pinned.descriptor,
            pinned.size_bytes - offset,
            offset,
            context="TAR record padding",
        )
        if tail != b"\0" * len(tail):
            raise PublicReleaseError("archive has nonzero bytes after its end markers")
    if set(metadata) != captured_members:
        raise PublicReleaseError(
            "archive omits canonical manifest, checksums, or CBaSE provenance",
        )
    if records[ARCHIVE_MANIFEST_MEMBER]["sha256"] != manifest_digest:
        raise PublicReleaseError(
            "release manifest fails its independent SHA-256 anchor",
        )
    manifest = _validate_release_manifest(
        metadata[ARCHIVE_MANIFEST_MEMBER],
        actual_records=records,
        captured_payloads=metadata,
    )
    checksum_records = _parse_checksum_file(metadata[ARCHIVE_CHECKSUM_MEMBER])
    checksum_names = {member for _, member in checksum_records}
    expected_checksum_names = set(records) - {ARCHIVE_CHECKSUM_MEMBER}
    if checksum_names != expected_checksum_names:
        raise PublicReleaseError("SHA256SUMS coverage differs from its closed policy")
    for digest, member in checksum_records:
        if records[member]["sha256"] != digest:
            raise PublicReleaseError("SHA256SUMS contradicts an archived member")
    _revalidate_file(pinned, context="downloaded public archive after TAR parsing")
    release = _expect_mapping(manifest["release"], context="manifest release")
    return _VerifiedArchive(
        archive_sha256=pinned.sha256,
        archive_bytes=pinned.size_bytes,
        manifest_sha256=manifest_digest,
        release_id=_expect_token(
            release.get("release_id"),
            context="manifest release_id",
        ),
        member_count=len(records),
    )


def _destination_parent(
    destinations: Sequence[Path],
    *,
    forbidden_roots: Sequence[Path],
    context: str,
) -> _PinnedRoot:
    if not destinations:
        raise PublicReleaseError(f"{context} has no destinations")
    canonical = [
        _require_absolute_canonical(path, context=f"{context} destination")
        for path in destinations
    ]
    parent = canonical[0].parent
    if any(path.parent != parent for path in canonical):
        raise PublicReleaseError(f"{context} destinations must share one parent")
    for root_path in forbidden_roots:
        root = _require_absolute_canonical(root_path, context="input root")
        if parent == root or parent.is_relative_to(root):
            raise PublicReleaseError(f"{context} destination is inside an input root")
    pinned = _pin_root(parent, context=f"{context} destination parent")
    try:
        for path in canonical:
            try:
                os.stat(
                    path.name,
                    dir_fd=pinned.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                raise PublicReleaseError(
                    f"cannot preflight {context} destination",
                ) from error
            raise PublicReleaseError(
                f"refusing to replace {context} destination: {path}",
            )
        _revalidate_root(pinned, context=f"{context} destination parent")
    except Exception:
        pinned.close()
        raise
    return pinned


def _create_staging(
    parent: _PinnedRoot,
    basename: str,
) -> tuple[str, int, tuple[int, int]]:
    staging = f"{_private_stage_prefix(basename)}candidate"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(staging, flags, 0o400, dir_fd=parent.descriptor)
    except FileExistsError as error:
        raise PublicReleaseError(
            "retained private stage requires explicit review before retry: "
            f"{parent.path / staging}; candidate_label=unclassified,"
            f"candidate_paths={parent.path / staging},"
            "candidate_state=private-stage-name-may-be-owned-or-replaced-"
            "do-not-auto-delete,expected_sha256=unknown,expected_bytes=unknown",
        ) from error
    except OSError as error:
        raise PublicReleaseError(
            "cannot create exclusive publication staging file",
        ) from error
    try:
        observed = os.fstat(descriptor)
    except OSError as error:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise PublicReleaseError(
            f"cannot inspect exclusive publication staging file; "
            f"candidate_label=unclassified,candidate_paths={parent.path / staging},"
            "candidate_state=private-stage-name-may-be-owned-or-replaced-"
            "do-not-auto-delete,expected_sha256=unknown,expected_bytes=unknown; "
            "inspect its identity and bytes before any explicit review/removal",
        ) from error
    return staging, descriptor, (observed.st_dev, observed.st_ino)


def _write_all(descriptor: int, raw: bytes) -> None:
    position = 0
    while position < len(raw):
        written = os.write(descriptor, raw[position:])
        if written <= 0:
            raise PublicReleaseError("publication staging write made no progress")
        position += written


def _private_stage_prefix(basename: str) -> str:
    return f".{basename}{PRIVATE_STAGE_MARKER}"


def _require_destination_names_absent(
    parent: _PinnedRoot,
    destinations: Sequence[Path],
    *,
    context: str,
) -> None:
    _revalidate_root(parent, context=f"{context} parent before name preflight")
    for destination in destinations:
        try:
            os.stat(
                destination.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PublicReleaseError(
                f"cannot preflight {context} destination: {destination}",
            ) from error
        raise PublicReleaseError(
            f"refusing to replace {context} destination: {destination}",
        )
    _revalidate_root(parent, context=f"{context} parent after name preflight")


def _require_no_retained_stages(
    parent: _PinnedRoot,
    basenames: Sequence[str],
    *,
    context: str,
    reserved_entries: int,
) -> None:
    """Block publication until every prior private candidate is reconciled."""
    if (
        isinstance(reserved_entries, bool)
        or not isinstance(reserved_entries, int)
        or not 0 <= reserved_entries <= MAX_PUBLICATION_DIRECTORY_ENTRIES
    ):
        raise PublicReleaseError(
            "publication directory reserved_entries is outside policy",
        )
    maximum_existing = MAX_PUBLICATION_DIRECTORY_ENTRIES - reserved_entries
    prefixes = tuple(_private_stage_prefix(basename) for basename in basenames)
    _revalidate_root(parent, context=f"{context} parent before stage preflight")
    try:
        entries = os.scandir(parent.descriptor)
    except OSError as error:
        raise PublicReleaseError(
            f"cannot inspect {context} parent for retained private stages",
        ) from error
    retained: str | None = None
    scanned = 0
    try:
        with entries:
            for entry in entries:
                scanned += 1
                if scanned > maximum_existing:
                    raise PublicReleaseError(
                        f"{context} parent exceeds the bounded directory policy of "
                        f"{MAX_PUBLICATION_DIRECTORY_ENTRIES} final entries after "
                        f"reserving {reserved_entries} publication slots",
                    )
                if any(entry.name.startswith(prefix) for prefix in prefixes):
                    retained = entry.name
                    break
    except OSError as error:
        raise PublicReleaseError(
            f"cannot scan {context} parent for retained private stages",
        ) from error
    _revalidate_root(parent, context=f"{context} parent after stage preflight")
    if retained is not None:
        raise PublicReleaseError(
            "retained private stage requires explicit review before retry: "
            f"{parent.path / retained}",
        )


def _pin_staged(parent: _PinnedRoot, member: str, *, context: str) -> _PinnedFile:
    return _pin_member(parent, member, context=context)


def _rename_staged_no_replace(
    parent: _PinnedRoot,
    staging_name: str,
    destination_name: str,
    *,
    context: str,
) -> None:
    """Atomically rename one sibling stage without replacing any destination."""
    try:
        _revalidate_root(
            parent,
            context=f"{context} destination parent before rename",
        )
        library = ctypes.CDLL(None, use_errno=True)
    except (OSError, PublicReleaseError) as error:
        raise _RenameNotPerformedError(str(error)) from error
    if sys.platform == "darwin":
        try:
            rename = library.renameatx_np
        except AttributeError as error:
            raise _RenameNotPerformedError(
                "platform atomic no-replace rename symbol renameatx_np is unavailable",
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent.descriptor,
            os.fsencode(staging_name),
            parent.descriptor,
            os.fsencode(destination_name),
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise _RenameNotPerformedError(
                "platform atomic no-replace rename symbol renameat2 is unavailable",
            ) from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent.descriptor,
            os.fsencode(staging_name),
            parent.descriptor,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
    else:
        raise _RenameNotPerformedError(
            "platform lacks a supported atomic no-replace rename",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PublicReleaseError(
            f"refusing to replace {context} destination; the issued atomic rename "
            "syscall outcome is treated as ambiguous",
        )
    unsupported_errors = {
        number
        for number in (
            getattr(errno, "EINVAL", None),
            getattr(errno, "ENOSYS", None),
            getattr(errno, "ENOTSUP", None),
            getattr(errno, "EOPNOTSUPP", None),
        )
        if number is not None
    }
    if error_number in unsupported_errors:
        raise PublicReleaseError(
            "filesystem/platform does not support atomic no-replace rename: "
            f"{os.strerror(error_number)} (errno {error_number}); the issued syscall "
            "outcome is treated as ambiguous",
        )
    raise PublicReleaseError(
        f"cannot publish {context} destination with atomic no-replace rename: "
        f"{os.strerror(error_number)} (errno {error_number}); the issued syscall "
        "outcome is treated as ambiguous",
    )


def _verify_staged_file(  # noqa: PLR0913
    parent: _PinnedRoot,
    staging_name: str,
    *,
    identity: tuple[int, int],
    expected_sha256: str,
    expected_bytes: int,
    context: str,
) -> None:
    staged = _pin_staged(parent, staging_name, context=context)
    try:
        if (
            (staged.device, staged.inode) != identity
            or staged.sha256 != expected_sha256
            or staged.size_bytes != expected_bytes
            or staged.mode != 0o400
        ):
            raise PublicReleaseError(f"{context} differs from its intended bytes")
        _revalidate_file(staged, context=context)
    finally:
        staged.close()


def _require_published_file(  # noqa: PLR0913
    parent: _PinnedRoot,
    destination: Path,
    *,
    identity: tuple[int, int],
    expected_sha256: str,
    expected_bytes: int,
    context: str,
) -> None:
    _revalidate_root(parent, context=f"{context} destination parent")
    pin = _pin_member(
        parent,
        destination.name,
        context=context,
        expected_size=expected_bytes,
    )
    try:
        if (
            (pin.device, pin.inode) != identity
            or pin.sha256 != expected_sha256
            or pin.size_bytes != expected_bytes
            or pin.mode != 0o400
        ):
            raise PublicReleaseError(f"{context} final readback differs")
        _revalidate_file(pin, context=f"{context} final readback")
        _revalidate_root(parent, context=f"{context} destination parent after readback")
    finally:
        pin.close()


def _directory_snapshot(parent: _PinnedRoot) -> _DirectorySnapshot:
    before = os.fstat(parent.descriptor)
    try:
        entries = os.scandir(parent.descriptor)
    except OSError as error:
        raise PublicReleaseError("cannot inspect publication directory") from error
    observed: set[str] = set()
    scanned = 0
    try:
        with entries:
            for entry in entries:
                scanned += 1
                if scanned >= PUBLICATION_DIRECTORY_SCAN_CAP:
                    raise PublicReleaseError(
                        "publication directory exceeds the bounded policy of "
                        f"{MAX_PUBLICATION_DIRECTORY_ENTRIES} entries",
                    )
                if entry.name in observed:
                    raise PublicReleaseError(
                        f"publication directory returned duplicate name {entry.name!r}",
                    )
                observed.add(entry.name)
    except OSError as error:
        raise PublicReleaseError("cannot scan publication directory") from error
    after = os.fstat(parent.descriptor)

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if signature(before) != signature(after):
        raise PublicReleaseError("destination parent changed during snapshot")
    return _DirectorySnapshot(
        members=tuple(sorted(observed)),
        stat_signature=signature(after),
    )


def _require_directory_snapshot(
    parent: _PinnedRoot,
    expected: _DirectorySnapshot,
    *,
    context: str,
) -> None:
    _revalidate_root(parent, context=context)
    if _directory_snapshot(parent) != expected:
        raise PublicReleaseError(f"{context} membership or metadata changed")


def _publication_candidate_record(  # noqa: PLR0913
    *,
    label: str,
    parent: _PinnedRoot,
    destination: Path,
    staging_name: str,
    stage_created: bool,
    stage_verified: bool,
    rename_attempted: bool,
    rename_confirmed: bool,
    rename_definitely_not_performed: bool,
    expected_sha256: str,
    expected_bytes: int,
) -> str | None:
    if rename_confirmed:
        paths = str(destination)
        state = "destination-name-may-be-owned-or-replaced-do-not-auto-delete"
    elif rename_attempted and not rename_definitely_not_performed:
        paths = f"{destination}|{parent.path / staging_name}"
        state = (
            "destination-or-private-stage-name-may-be-owned-or-replaced-"
            "do-not-auto-delete"
        )
    elif stage_created:
        paths = str(parent.path / staging_name)
        state = "private-stage-name-may-be-owned-or-replaced-do-not-auto-delete"
    else:
        return None
    digest = expected_sha256 if stage_verified else "unknown"
    size: int | str = expected_bytes if stage_verified else "unknown"
    return (
        f"candidate_label={label},candidate_paths={paths},candidate_state={state},"
        f"intended_destination={destination},"
        f"candidate_parent_identity={parent.device}:{parent.inode},"
        f"expected_sha256={digest},expected_bytes={size}"
    )


def _publication_failure(
    error: Exception,
    records: Sequence[str | None],
) -> PublicReleaseError | None:
    candidates = [record for record in records if record is not None]
    if not candidates:
        return None
    return PublicReleaseError(
        f"{error}; {'; '.join(candidates)}; inspect every reported name, identity, "
        "and byte sequence before any explicit review/removal; candidate paths are "
        "lexical and their parent may have moved or been replaced",
    )


def _publication_receipt(
    prepared: _PreparedRelease,
    archive: _VerifiedArchive,
) -> bytes:
    release = _expect_mapping(prepared.plan["release"], context="plan.release")
    payload = {
        "schema": RECEIPT_SCHEMA,
        "contract": "sequential-per-member-atomic-no-replace-public-release-v1",
        "release_id": release["release_id"],
        "version": release["version"],
        "archive": {
            "name": release["archive_name"],
            "format": ARCHIVE_FORMAT,
            "bytes": archive.archive_bytes,
            "sha256": archive.archive_sha256,
            "member_count": archive.member_count,
        },
        "release_manifest_sha256": archive.manifest_sha256,
        "plan_sha256": prepared.plan_file.sha256,
        "source_commit_a": release["source_commit_a"],
        "release_commit_b": release["release_commit_b"],
        "source_tag": release["source_tag"],
        "publication": {
            "pair_atomic": False,
            "publication_order": ["archive", "receipt"],
            "archive_atomic_no_replace": True,
            "receipt_atomic_no_replace": True,
            "partial_publication_retained_for_explicit_reconciliation": True,
            "input_revalidated_after_archive_write": True,
            "submission_package_created": False,
        },
    }
    return _canonical_json(payload) + b"\n"


def _publish_release(
    prepared: _PreparedRelease,
    parent: _PinnedRoot,
) -> PublicReleaseReceipt:
    _require_no_retained_stages(
        parent,
        (
            prepared.config.destination_archive.name,
            prepared.config.destination_receipt.name,
        ),
        context="public release destination",
        reserved_entries=2,
    )
    _require_destination_names_absent(
        parent,
        (
            prepared.config.destination_archive,
            prepared.config.destination_receipt,
        ),
        context="public release",
    )
    archive_staging = ""
    receipt_staging = ""
    archive_fd: int | None = None
    receipt_fd: int | None = None
    archive_staging_identity: tuple[int, int] | None = None
    receipt_staging_identity: tuple[int, int] | None = None
    archive_stage_created = False
    receipt_stage_created = False
    archive_stage_verified = False
    receipt_stage_verified = False
    archive_rename_attempted = False
    receipt_rename_attempted = False
    archive_rename_confirmed = False
    receipt_rename_confirmed = False
    archive_rename_definitely_not_performed = False
    receipt_rename_definitely_not_performed = False
    archive: _VerifiedArchive | None = None
    archive_sha256 = ""
    archive_bytes = 0
    receipt_raw = b""
    receipt_sha256 = ""
    try:
        archive_staging, archive_fd, archive_staging_identity = _create_staging(
            parent,
            prepared.config.destination_archive.name,
        )
        archive_stage_created = True
        _write_archive(archive_fd, prepared)
        staged_archive = _pin_staged(
            parent,
            archive_staging,
            context="staged public archive",
        )
        try:
            archive = _verify_archive_descriptor(
                staged_archive,
                expected_archive_sha256=staged_archive.sha256,
                expected_manifest_sha256=_sha256(prepared.manifest_raw),
            )
            if (
                staged_archive.device,
                staged_archive.inode,
            ) != archive_staging_identity or staged_archive.mode != 0o400:
                raise PublicReleaseError("staged public archive identity is invalid")
        finally:
            staged_archive.close()
        archive_sha256 = archive.archive_sha256
        archive_bytes = archive.archive_bytes
        _verify_staged_file(
            parent,
            archive_staging,
            identity=archive_staging_identity,
            expected_sha256=archive_sha256,
            expected_bytes=archive_bytes,
            context="fully staged public archive",
        )
        archive_stage_verified = True
        archive_descriptor = archive_fd
        archive_fd = None
        os.close(archive_descriptor)
        _revalidate_prepared(prepared)
        receipt_raw = _publication_receipt(prepared, archive)
        receipt_sha256 = _sha256(receipt_raw)
        receipt_staging, receipt_fd, receipt_staging_identity = _create_staging(
            parent,
            prepared.config.destination_receipt.name,
        )
        receipt_stage_created = True
        _write_all(receipt_fd, receipt_raw)
        os.fchmod(receipt_fd, 0o400)
        os.fsync(receipt_fd)
        _verify_staged_file(
            parent,
            receipt_staging,
            identity=receipt_staging_identity,
            expected_sha256=receipt_sha256,
            expected_bytes=len(receipt_raw),
            context="fully staged public release receipt",
        )
        receipt_stage_verified = True
        receipt_descriptor = receipt_fd
        receipt_fd = None
        os.close(receipt_descriptor)
        _revalidate_prepared(prepared)
        _revalidate_root(
            parent,
            context="release destination parent before publication",
        )
        archive_rename_attempted = True
        try:
            _rename_staged_no_replace(
                parent,
                archive_staging,
                prepared.config.destination_archive.name,
                context="public archive",
            )
        except _RenameNotPerformedError:
            archive_rename_definitely_not_performed = True
            raise
        archive_rename_confirmed = True
        os.fsync(parent.descriptor)
        _revalidate_root(
            parent,
            context="release destination parent after archive publication",
        )
        _revalidate_prepared(prepared)
        receipt_rename_attempted = True
        try:
            _rename_staged_no_replace(
                parent,
                receipt_staging,
                prepared.config.destination_receipt.name,
                context="public release receipt",
            )
        except _RenameNotPerformedError:
            receipt_rename_definitely_not_performed = True
            raise
        receipt_rename_confirmed = True
        os.fsync(parent.descriptor)
        _revalidate_root(
            parent,
            context="release destination parent after receipt publication",
        )
        _revalidate_prepared(prepared)
        published_parent_snapshot = _directory_snapshot(parent)
        _require_published_file(
            parent,
            prepared.config.destination_archive,
            identity=archive_staging_identity,
            expected_sha256=archive_sha256,
            expected_bytes=archive_bytes,
            context="published public archive",
        )
        _require_published_file(
            parent,
            prepared.config.destination_receipt,
            identity=receipt_staging_identity,
            expected_sha256=receipt_sha256,
            expected_bytes=len(receipt_raw),
            context="published public release receipt",
        )
        _revalidate_prepared(prepared)
        _require_published_file(
            parent,
            prepared.config.destination_archive,
            identity=archive_staging_identity,
            expected_sha256=archive_sha256,
            expected_bytes=archive_bytes,
            context="final published public archive",
        )
        _require_published_file(
            parent,
            prepared.config.destination_receipt,
            identity=receipt_staging_identity,
            expected_sha256=receipt_sha256,
            expected_bytes=len(receipt_raw),
            context="final published public release receipt",
        )
        _require_no_retained_stages(
            parent,
            (
                prepared.config.destination_archive.name,
                prepared.config.destination_receipt.name,
            ),
            context="final public release destination",
            reserved_entries=0,
        )
        _require_directory_snapshot(
            parent,
            published_parent_snapshot,
            context="release destination parent at final return",
        )
        return PublicReleaseReceipt(
            archive_path=str(prepared.config.destination_archive),
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            manifest_sha256=archive.manifest_sha256,
            receipt_path=str(prepared.config.destination_receipt),
            receipt_sha256=receipt_sha256,
            member_count=archive.member_count,
        )
    except Exception as error:
        failure = _publication_failure(
            error,
            (
                _publication_candidate_record(
                    label="archive",
                    parent=parent,
                    destination=prepared.config.destination_archive,
                    staging_name=archive_staging,
                    stage_created=archive_stage_created,
                    stage_verified=archive_stage_verified,
                    rename_attempted=archive_rename_attempted,
                    rename_confirmed=archive_rename_confirmed,
                    rename_definitely_not_performed=(
                        archive_rename_definitely_not_performed
                    ),
                    expected_sha256=archive_sha256,
                    expected_bytes=archive_bytes,
                ),
                _publication_candidate_record(
                    label="receipt",
                    parent=parent,
                    destination=prepared.config.destination_receipt,
                    staging_name=receipt_staging,
                    stage_created=receipt_stage_created,
                    stage_verified=receipt_stage_verified,
                    rename_attempted=receipt_rename_attempted,
                    rename_confirmed=receipt_rename_confirmed,
                    rename_definitely_not_performed=(
                        receipt_rename_definitely_not_performed
                    ),
                    expected_sha256=receipt_sha256,
                    expected_bytes=len(receipt_raw),
                ),
            ),
        )
        if failure is None:
            raise
        raise failure from error
    finally:
        if archive_fd is not None:
            os.close(archive_fd)
        if receipt_fd is not None:
            os.close(receipt_fd)


def _input_root_paths(config: PublicReleaseBuildConfig) -> tuple[Path, ...]:
    return (
        config.source_data_root,
        config.renderer_root,
        config.rendered_output_root,
        config.gate_receipt_root,
        config.evidence_source_root,
        config.document_root,
        config.dependency_root,
        config.repository_root,
    )


def build_public_release(config: PublicReleaseBuildConfig) -> PublicReleaseReceipt:
    """Build a release, then publish archive and receipt with atomic members.

    Each destination uses an atomic no-replace rename.  The two-name release is
    intentionally sequential rather than pair-atomic: an archive retained after a
    receipt publication failure requires explicit reconciliation before retry.
    """
    plan_file, plan, pending = _pin_and_parse_plan(
        config.plan_path,
        config.expected_plan_sha256,
    )
    prepared: _PreparedRelease | None = None
    destination_parent: _PinnedRoot | None = None
    try:
        if plan["mode"] != "final" or pending:
            raise PublicReleaseError("only a closed final plan may publish an archive")
        _validate_build_paths(config, plan)
        destination_parent = _destination_parent(
            (config.destination_archive, config.destination_receipt),
            forbidden_roots=_input_root_paths(config),
            context="public release",
        )
        prepared = _prepare_release(config, plan_file, plan)
        plan_file = prepared.plan_file
        _revalidate_prepared(prepared)
        return _publish_release(prepared, destination_parent)
    finally:
        if prepared is not None:
            prepared.close()
        else:
            plan_file.close()
        if destination_parent is not None:
            destination_parent.close()


def _normalize_portal_identity(doi: str, locator: str) -> tuple[str, str]:
    if not isinstance(doi, str) or _DOI_RE.fullmatch(doi) is None:
        raise PublicReleaseError("DOI is not canonical")
    if (
        not isinstance(locator, str)
        or not locator
        or not locator.isascii()
        or "\\" in locator
        or any(character.isspace() or ord(character) < 32 for character in locator)
    ):
        raise PublicReleaseError("portal locator is not canonical ASCII")
    try:
        parsed = urlsplit(locator)
        parsed_port = parsed.port
    except ValueError as error:
        raise PublicReleaseError("portal locator is malformed") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed_port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise PublicReleaseError(
            "portal locator must be an HTTPS URL without auth or fragment",
        )
    return doi, locator


def _publish_portal_readback(
    parent: _PinnedRoot,
    destination: Path,
    raw: bytes,
    *,
    revalidate_input: Callable[[], None],
) -> str:
    _require_no_retained_stages(
        parent,
        (destination.name,),
        context="portal readback destination",
        reserved_entries=1,
    )
    _require_destination_names_absent(
        parent,
        (destination,),
        context="portal readback",
    )
    staging = ""
    descriptor: int | None = None
    staging_identity: tuple[int, int] | None = None
    stage_created = False
    stage_verified = False
    rename_attempted = False
    rename_confirmed = False
    rename_definitely_not_performed = False
    digest = _sha256(raw)
    try:
        staging, descriptor, staging_identity = _create_staging(
            parent,
            destination.name,
        )
        stage_created = True
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        _verify_staged_file(
            parent,
            staging,
            identity=staging_identity,
            expected_sha256=digest,
            expected_bytes=len(raw),
            context="fully staged portal readback receipt",
        )
        stage_verified = True
        staged_descriptor = descriptor
        descriptor = None
        os.close(staged_descriptor)
        revalidate_input()
        rename_attempted = True
        try:
            _rename_staged_no_replace(
                parent,
                staging,
                destination.name,
                context="portal readback receipt",
            )
        except _RenameNotPerformedError:
            rename_definitely_not_performed = True
            raise
        rename_confirmed = True
        os.fsync(parent.descriptor)
        _revalidate_root(parent, context="portal readback parent after rename")
        revalidate_input()
        published_parent_snapshot = _directory_snapshot(parent)
        _require_published_file(
            parent,
            destination,
            identity=staging_identity,
            expected_sha256=digest,
            expected_bytes=len(raw),
            context="published portal readback receipt",
        )
        revalidate_input()
        _require_published_file(
            parent,
            destination,
            identity=staging_identity,
            expected_sha256=digest,
            expected_bytes=len(raw),
            context="final published portal readback receipt",
        )
        _require_no_retained_stages(
            parent,
            (destination.name,),
            context="final portal readback destination",
            reserved_entries=0,
        )
        _require_directory_snapshot(
            parent,
            published_parent_snapshot,
            context="portal readback parent at final return",
        )
        return digest  # noqa: TRY300
    except Exception as error:
        failure = _publication_failure(
            error,
            (
                _publication_candidate_record(
                    label="portal-readback-receipt",
                    parent=parent,
                    destination=destination,
                    staging_name=staging,
                    stage_created=stage_created,
                    stage_verified=stage_verified,
                    rename_attempted=rename_attempted,
                    rename_confirmed=rename_confirmed,
                    rename_definitely_not_performed=(rename_definitely_not_performed),
                    expected_sha256=digest,
                    expected_bytes=len(raw),
                ),
            ),
        )
        if failure is None:
            raise
        raise failure from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def verify_download(  # noqa: PLR0913
    downloaded_archive_path: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    doi: str,
    locator: str,
    destination_receipt: Path,
) -> PortalReadbackReceipt:
    """Stream-verify a downloaded public archive and publish a portal receipt."""
    outer_anchor = _expect_sha256(
        expected_archive_sha256,
        context="expected_archive_sha256",
    )
    manifest_anchor = _expect_sha256(
        expected_manifest_sha256,
        context="expected_manifest_sha256",
    )
    normalized_doi, normalized_locator = _normalize_portal_identity(doi, locator)
    archive_path = _require_absolute_canonical(
        downloaded_archive_path,
        context="downloaded archive path",
    )
    destination = _require_absolute_canonical(
        destination_receipt,
        context="portal readback destination",
    )
    if destination == archive_path:
        raise PublicReleaseError("portal receipt cannot replace the downloaded archive")
    pinned: _PinnedFile | None = None
    parent: _PinnedRoot | None = None
    try:
        pinned = _pin_absolute_file(archive_path, context="downloaded public archive")
        verified = _verify_archive_descriptor(
            pinned,
            expected_archive_sha256=outer_anchor,
            expected_manifest_sha256=manifest_anchor,
        )
        parent = _destination_parent(
            (destination,),
            forbidden_roots=(),
            context="portal readback",
        )
        payload = {
            "schema": READBACK_SCHEMA,
            "contract": "outer-hash-first-streaming-no-extraction-v1",
            "release_id": verified.release_id,
            "archive": {
                "name": archive_path.name,
                "bytes": verified.archive_bytes,
                "sha256": verified.archive_sha256,
                "format": ARCHIVE_FORMAT,
                "member_count": verified.member_count,
            },
            "release_manifest_sha256": verified.manifest_sha256,
            "portal": {"doi": normalized_doi, "locator": normalized_locator},
            "verification": {
                "outer_sha256_checked_before_tar": True,
                "streamed": True,
                "extracted": False,
                "unsafe_members_rejected": True,
                "manifest_checksum_inventory_closed": True,
            },
        }
        raw = _canonical_json(payload) + b"\n"

        def revalidate_archive() -> None:
            if pinned is None:
                raise PublicReleaseError("downloaded archive descriptor disappeared")
            _revalidate_file(
                pinned,
                context="downloaded public archive publication input",
            )

        receipt_sha256 = _publish_portal_readback(
            parent,
            destination,
            raw,
            revalidate_input=revalidate_archive,
        )
        return PortalReadbackReceipt(
            downloaded_archive_path=str(archive_path),
            archive_sha256=verified.archive_sha256,
            archive_bytes=verified.archive_bytes,
            manifest_sha256=verified.manifest_sha256,
            release_id=verified.release_id,
            destination_path=str(destination),
            destination_sha256=receipt_sha256,
        )
    finally:
        if pinned is not None:
            pinned.close()
        if parent is not None:
            parent.close()


def _absolute_cli_path(path: Path) -> Path:
    return Path(os.path.abspath(path))  # noqa: PTH100


def _build_config(args: argparse.Namespace) -> PublicReleaseBuildConfig:
    return PublicReleaseBuildConfig(
        plan_path=_absolute_cli_path(args.plan),
        source_data_root=_absolute_cli_path(args.source_data_root),
        artifact_registry_path=_absolute_cli_path(args.artifact_registry),
        release_evidence_path=_absolute_cli_path(args.release_evidence),
        renderer_root=_absolute_cli_path(args.renderer_root),
        rendered_output_root=_absolute_cli_path(args.rendered_output_root),
        gate_receipt_root=_absolute_cli_path(args.gate_receipt_root),
        evidence_source_root=_absolute_cli_path(args.evidence_source_root),
        document_reconciliation_path=_absolute_cli_path(
            args.document_reconciliation,
        ),
        document_anchor_path=_absolute_cli_path(args.document_anchor),
        document_root=_absolute_cli_path(args.document_root),
        k500_authority_projection_path=_absolute_cli_path(
            args.k500_authority_projection,
        ),
        expected_k500_authority_projection_sha256=(
            args.expected_k500_authority_projection_sha256
        ),
        dependency_root=_absolute_cli_path(args.dependency_root),
        repository_root=_absolute_cli_path(args.repository_root),
        destination_archive=_absolute_cli_path(args.destination_archive),
        destination_receipt=_absolute_cli_path(args.destination_receipt),
        expected_plan_sha256=args.expected_plan_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit-plan",
        help="validate only an independently anchored draft or final plan",
    )
    audit_parser.add_argument("--plan", type=Path, required=True)
    audit_parser.add_argument("--expected-plan-sha256", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="build and publish one final immutable public release",
    )
    for flag in (
        "source-data-root",
        "renderer-root",
        "rendered-output-root",
        "gate-receipt-root",
        "evidence-source-root",
        "document-root",
        "dependency-root",
        "repository-root",
    ):
        build_parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in (
        "plan",
        "artifact-registry",
        "release-evidence",
        "document-reconciliation",
        "document-anchor",
        "k500-authority-projection",
        "destination-archive",
        "destination-receipt",
    ):
        build_parser.add_argument(f"--{flag}", type=Path, required=True)
    build_parser.add_argument("--expected-plan-sha256", required=True)
    build_parser.add_argument(
        "--expected-k500-authority-projection-sha256",
        required=True,
    )

    verify_parser = subparsers.add_parser(
        "verify-download",
        help="stream-verify a downloaded archive and publish a portal receipt",
    )
    verify_parser.add_argument("--downloaded-archive", type=Path, required=True)
    verify_parser.add_argument("--expected-archive-sha256", required=True)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    verify_parser.add_argument("--doi", required=True)
    verify_parser.add_argument("--locator", required=True)
    verify_parser.add_argument("--destination-receipt", type=Path, required=True)
    return parser


def main() -> None:
    """Run result-blind plan audit, build, or streaming verification."""
    args = _parser().parse_args()
    if args.command == "audit-plan":
        receipt: object = audit_public_release_plan(
            _absolute_cli_path(args.plan),
            expected_plan_sha256=args.expected_plan_sha256,
        )
    elif args.command == "build":
        receipt = build_public_release(_build_config(args))
    else:
        receipt = verify_download(
            _absolute_cli_path(args.downloaded_archive),
            expected_archive_sha256=args.expected_archive_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            doi=args.doi,
            locator=args.locator,
            destination_receipt=_absolute_cli_path(args.destination_receipt),
        )
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

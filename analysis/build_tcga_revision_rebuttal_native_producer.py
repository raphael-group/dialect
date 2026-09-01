"""Build a host-bound native launcher package for rebuttal PDF derivation.

The package is a candidate, never a self-authenticating authority.  It contains
exactly a thin arm64 launcher and one canonical v2 capsule.  Two distinct private
stage roots are independently compiled, linked, ad-hoc signed, verified, and
required to be byte-identical before one root can be atomically published.  The
other root is retained as evidence; no staging tree is automatically deleted.

``revision`` mode additionally requires Git-command-observed relevant path bytes
at a caller-pinned release commit/ref. ``synthetic-canary`` mode exists so this seam can
be tested before its own source is committed and is never promotable.  The source
bundle is treated as canonical metadata: the capsule projects member hashes and
dependency/output anchors, but never copies rebuttal prose or base64 payloads.

The launcher is deliberately host-bound.  Absolute runtime and renderer paths
are embedded in its Mach-O so it can hand off without PATH lookup, but canonical
authority JSON records only fixed locators and path-byte digests.  Pre-exec path
hashing does not attest the post-exec interpreter, close same-UID TOCTOU windows,
or prove loaded Python/dylib identity, scientific correctness, visual approval,
coauthor approval, submission, upload, or journal acceptance.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import render_tcga_revision_rebuttal as renderer

# This narrow packaging boundary intentionally owns its exact subprocess and
# publication contracts. Complexity is primarily defensive validation.
# ruff: noqa: COM812, PERF203, PLR0913, SLF001

AUTHORITY_SCHEMA: Final = "dialect-revision-rebuttal-native-producer-authority-v2"
AUTHORITY_CONTRACT: Final = (
    "host-bound-thin-arm64-double-build-adhoc-codesign-two-member-package-v2"
)
CONFIG_SCHEMA: Final = "dialect-revision-rebuttal-native-launcher-config-v2"
TREE_HASH_CONTRACT: Final = (
    "u64be-path-type-mode-nlink-size-content-or-symlink-target-v1"
)
MODE_REVISION: Final = "revision"
MODE_SYNTHETIC: Final = "synthetic-canary"
MODES: Final = (MODE_REVISION, MODE_SYNTHETIC)
SYNTHETIC_SOURCE_STATUS: Final = (
    "synthetic-caller-pinned-listed-file-bytes-with-logical-member-labels-not-git-"
    "or-repository-path-bound"
)
PDF_ID: Final = "rebuttal"
PDF_MEMBER: Final = "response-to-reviewers.pdf"
PRODUCER_MEMBER: Final = "derive-rebuttal"
AUTHORITY_MEMBER: Final = "rebuttal-producer-toolchain-authority.json"
PACKAGE_MEMBERS: Final = (PRODUCER_MEMBER, AUTHORITY_MEMBER)
PROTOCOL: Final = renderer.DERIVATION_PROTOCOL
PRODUCER_ARGUMENTS: Final = [
    "--dialect-derivation-protocol",
    PROTOCOL,
    "--pdf-id",
    PDF_ID,
    "--source-fd",
    "{source_fd}",
    "--pdf-output",
    "stdout",
]
EXACT_LAUNCH_ENVIRONMENT: Final = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
EXACT_TOOL_ENVIRONMENT: Final = dict(EXACT_LAUNCH_ENVIRONMENT)
LAUNCHER_SOURCE_MEMBER: Final = "analysis/native/rebuttal_derivation_launcher.c"
BUILDER_MEMBER: Final = "analysis/build_tcga_revision_rebuttal_native_producer.py"
BUNDLE_BUILDER_MEMBER: Final = (
    "analysis/build_tcga_revision_rebuttal_derivation_bundle.py"
)
MACHINE_RUNNER_MEMBER: Final = renderer.MACHINE_RUNNER_MEMBER
RELEVANT_RELEASE_MEMBERS: Final = (
    LAUNCHER_SOURCE_MEMBER,
    BUILDER_MEMBER,
    BUNDLE_BUILDER_MEMBER,
    "analysis/render_tcga_revision_rebuttal.py",
    MACHINE_RUNNER_MEMBER,
)
REVIEW_SCOPE: Final = (
    "native-launcher-source-config-build-toolchain-bundle-projection-v2"
)
SIGNATURE_IDENTIFIER: Final = "org.raphaelgroup.dialect.rebuttal-derivation-launcher"
MACOS_MINIMUM: Final = "13.0"

XCODE_TOOLCHAIN_ROOT: Final = Path(
    "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain"
)
EXPECTED_CLANG: Final = XCODE_TOOLCHAIN_ROOT / "usr/bin/clang"
EXPECTED_LD: Final = XCODE_TOOLCHAIN_ROOT / "usr/bin/ld"
EXPECTED_CODESIGN: Final = Path("/usr/bin/codesign")
EXPECTED_GIT: Final = Path("/Applications/Xcode.app/Contents/Developer/usr/bin/git")
EXPECTED_COMPILER_RESOURCE_ROOT: Final = XCODE_TOOLCHAIN_ROOT / "usr/lib/clang/21"
EXPECTED_SDK_ROOT: Final = Path(
    "/Applications/Xcode.app/Contents/Developer/Platforms/"
    "MacOSX.platform/Developer/SDKs/MacOSX.sdk"
)

CALLER_ANCHOR_KEYS: Final = {
    "source_bundle_sha256",
    "launcher_source_sha256",
    "builder_sha256",
    "bundle_builder_sha256",
    "runtime_sha256",
    "renderer_sha256",
    "machine_runner_sha256",
    "clang_sha256",
    "linker_sha256",
    "codesign_sha256",
    "git_sha256",
    "compiler_resource_tree_sha256",
    "sdk_tree_sha256",
    "renderer_manifest_sha256",
    "pdf_sha256",
}

MAX_SOURCE_BYTES: Final = 512 * 1024
MAX_BUILDER_BYTES: Final = 2 * 1024 * 1024
MAX_BUNDLE_BYTES: Final = renderer.MAX_DERIVATION_BUNDLE_BYTES
MAX_EXECUTABLE_BYTES: Final = 8 * 1024 * 1024
MAX_AUTHORITY_BYTES: Final = 2 * 1024 * 1024
MAX_TOOL_BYTES: Final = 256 * 1024 * 1024
MAX_TOOL_OUTPUT_BYTES: Final = 512 * 1024
MAX_TREE_FILES: Final = 65_536
MAX_TREE_DIRECTORIES: Final = 16_384
MAX_TREE_SYMLINKS: Final = 65_536
MAX_TREE_ENTRIES: Final = 100_000
MAX_TREE_BYTES: Final = 1024 * 1024 * 1024
MAX_TREE_DEPTH: Final = 32
MAX_GIT_BLOB_BYTES: Final = 4 * 1024 * 1024
TOOL_TIMEOUT_SECONDS: Final = 120.0
READ_CHUNK_BYTES: Final = 64 * 1024
MAX_PROCESS_GROUP_MEMBERS: Final = 4096
MAX_MACH_LOAD_COMMANDS: Final = 512
MAX_SIGNATURE_BLOBS: Final = 32
MAX_FAT_SLICES: Final = 8

SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")
SAFE_PATH_RE: Final = re.compile(r"/[A-Za-z0-9_./+:-]+\Z")
SAFE_REF_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")
UUID_RE: Final = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)

MH_MAGIC_64: Final = 0xFEEDFACF
MH_EXECUTE: Final = 2
CPU_TYPE_X86_64: Final = 0x01000007
CPU_TYPE_ARM64: Final = 0x0100000C
CPU_SUBTYPE_ARM64_ALL: Final = 0
CPU_SUBTYPE_ARM64E: Final = 2
LC_CODE_SIGNATURE: Final = 0x1D
LC_UUID: Final = 0x1B
FAT_MAGIC: Final = 0xCAFEBABE
FAT_MAGIC_64: Final = 0xCAFEBABF
CSMAGIC_EMBEDDED_SIGNATURE: Final = 0xFADE0CC0
CSMAGIC_CODEDIRECTORY: Final = 0xFADE0C02
CSSLOT_CODEDIRECTORY: Final = 0
CS_CDHASH_LEN: Final = 20

PACKAGE_CONTRACT: Final = {
    "root_mode": "0500",
    "root_owner": "effective-user-id",
    "member_order": list(PACKAGE_MEMBERS),
    "member_modes": {PRODUCER_MEMBER: "0500", AUTHORITY_MEMBER: "0400"},
    "member_owner": "effective-user-id",
    "member_link_count": 1,
    "authority_self_reference": (
        "authority SHA-256 is excluded and must be supplied by an external caller"
    ),
    "publication": "atomic-no-replace-directory-rename",
    "authority_encoding": "canonical-json-plus-one-lf",
    "seal_scope": "posix-owner-mode-link-count-and-content-only",
}

NON_INFERENCE_LIMITS: Final = {
    "acl_xattr_and_file_flag_topology": "not-recorded-or-attested",
    "ad_hoc_codesign_signer_identity": "not-authenticated",
    "builder_loaded_code_and_invoking_runtime_identity": "not-attested",
    "codesign_executed_slice_attestation": (
        "universal-file-and-arm64e-slice-bytes-parsed; live-mapping-not-attested"
    ),
    "code_signature_validation_scope": (
        "primary-codedirectory-code-slots-and-identity-independently-parsed;"
        "special-slot-semantics-rely-on-bounded-codesign-verify"
    ),
    "compiler_linker_codesign_child_or_dylib_closure": "not-attested",
    "compiler_resource_and_sdk_causal_reads": (
        "declared-tree-byte-projections-anchored-but-causal-member-use-not-observed"
    ),
    "compiler_sdk_codesign_correctness": "not-established-by-byte-anchors",
    "determinism_scope": (
        "two-byte-identical-builds-observed-on-this-host-with-the-declared-"
        "main-file-and-tree-anchors"
    ),
    "detached_setsid_descendant_containment": "not-provided",
    "process_group_or_session_migrated_descendant_containment": "not-provided",
    "synthetic_source_path_scope": (
        "caller-pinned-file-bytes-with-logical-member-labels;"
        "repository-path-binding-not-provided"
    ),
    "git_commit_signer_or_authorship": "not-authenticated",
    "git_release_scope": (
        "git-command-observed-listed-path-bytes-only;object-format-blob-oid-"
        "object-type-repo-tree-index-worktree-cleanliness-tag-immutability-and-"
        "results-authority-not-attested"
    ),
    "git_child_or_dylib_closure": "not-attested",
    "hardlink_equivalence_groups_and_links_outside_root": "not-recorded",
    "host_private_paths": (
        "runtime-and-renderer-paths-embedded-in-private-host-bound-launcher"
    ),
    "host_portability": "private-current-host-package;not-portable",
    "loaded_python_code_identity": (
        "pre-exec-path-bytes-checked;post-exec-interpreter-and-loaded-code-not-attested"
    ),
    "inherited_process_state": (
        "umask-rlimits-signal-mask-and-ignored-signal-dispositions-not-normalized"
    ),
    "main_launcher_process_attestation": (
        "deferred-to-derivation-closure-suspended-process-check"
    ),
    "producer_authority": "capsule-is-a-candidate-requiring-external-sha-anchor",
    "producer_execution_during_build": (
        "launcher-handoff-and-expected-pdf-derivation-not-executed-by-package-build"
    ),
    "producer_filesystem_side_effects": "ambient-same-uid-access-not-contained",
    "receipt_path_fields": "operational-host-private-metadata;not-for-public-manifests",
    "destination_path_persistence_after_pin_release": "not-guaranteed",
    "os_kernel_and_dynamic_dependency_closure": (
        "xcode-usr-lib-dyld-shared-cache-system-frameworks-and-kernel-not-anchored"
    ),
    "renderer_causal_load": "not-observable-after-path-based-execve",
    "scientific_correctness": "not-inferred",
    "source_bundle_prose": "not-copied-into-authority-projection",
    "source_bundle_causal_consumption": "not-inferred-by-package-build",
    "static_code_directory_flags_to_runtime_csops_policy": (
        "actual-suspended-process-status-attestation-required"
    ),
    "text_legibility_or_visual_quality": "not-inferred",
    "toolchain_reproducibility": "cross-host-and-general-reproducibility-not-proven",
    "coauthor_or_submission_approval": "not-inferred",
    "journal_upload_acceptance_or_readback": "not-inferred",
}

FORBIDDEN_CAPSULE_FRAGMENTS: Final = (
    b"/Users/",
    b"/private/",
    b"/tmp/",
    b".codex",
    b".cache",
    b"research/",
    b"output/",
)


class RebuttalNativeProducerError(ValueError):
    """Raised when a native producer package cannot be proven safely."""


class _DarwinSigValue(ctypes.Union):
    _fields_: ClassVar = [("integer", ctypes.c_int), ("pointer", ctypes.c_void_p)]


class _DarwinSigInfo(ctypes.Structure):
    _fields_: ClassVar = [
        ("signal_number", ctypes.c_int),
        ("error_number", ctypes.c_int),
        ("code", ctypes.c_int),
        ("process_id", ctypes.c_int),
        ("user_id", ctypes.c_uint),
        ("status", ctypes.c_int),
        ("address", ctypes.c_void_p),
        ("value", _DarwinSigValue),
        ("band", ctypes.c_long),
        ("reserved", ctypes.c_ulong * 7),
    ]


class _ParentMappingError(RebuttalNativeProducerError):
    """Raised when a held publication parent loses its canonical path mapping."""


class _PublicationError(RebuttalNativeProducerError):
    """Retain the exact visible location after an ambiguous publication failure."""

    def __init__(
        self,
        message: str,
        *,
        location: Path | None,
        renamed: bool,
    ) -> None:
        super().__init__(message)
        self.location = location
        self.renamed = renamed


@dataclass(frozen=True, slots=True)
class RebuttalNativeProducerReceipt:
    """Summarize one sealed candidate package and retained independent build."""

    package_root: str
    independent_build_root: str
    authority_sha256: str
    authority_bytes: int
    producer_sha256: str
    producer_bytes: int
    producer_cdhash: str
    release_id: str
    mode: str
    replay_of: str | None
    promotable: bool


@dataclass(slots=True)
class _ParentPin:
    path: Path
    descriptor: int
    device: int
    inode: int
    mode: int
    uid: int

    def revalidate(self, *, context: str) -> None:
        try:
            opened = os.fstat(self.descriptor)
            named = os.lstat(self.path)
            resolved = self.path.resolve(strict=True)
        except OSError as error:
            message = f"cannot revalidate {context}: {error}"
            raise _ParentMappingError(message) from error
        identity = (self.device, self.inode, self.mode, self.uid)
        if (
            resolved != self.path
            or stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (
                opened.st_dev,
                opened.st_ino,
                stat.S_IMODE(opened.st_mode),
                opened.st_uid,
            )
            != identity
            or (
                named.st_dev,
                named.st_ino,
                stat.S_IMODE(named.st_mode),
                named.st_uid,
            )
            != identity
            or self.uid != os.geteuid()
            or self.mode & 0o022
        ):
            message = f"{context} identity, mapping, ownership, or mode changed"
            raise _ParentMappingError(message)

    def require_absent(self, name: str, *, context: str) -> None:
        try:
            os.stat(
                name,
                dir_fd=self.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as error:
            _fail(f"cannot inspect {context}: {error}")
        _fail(f"{context} already exists")

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(slots=True)
class _PinnedFile:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    mode: int
    uid: int
    sha256: str

    def revalidate(self, *, context: str) -> None:
        try:
            opened = os.fstat(self.descriptor)
            named = self.path.stat(follow_symlinks=False)
        except OSError as error:
            _fail(f"cannot revalidate {context}: {error}")
        identity = (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.mode,
            self.uid,
            1,
        )
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            stat.S_IMODE(opened.st_mode),
            opened.st_uid,
            opened.st_nlink,
        ) != identity or (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            stat.S_IMODE(named.st_mode),
            named.st_uid,
            named.st_nlink,
        ) != identity:
            _fail(f"{context} identity changed while pinned")
        raw = _read_fd(self.descriptor, maximum=self.size, context=context)
        if _sha256(raw) != self.sha256:
            _fail(f"{context} bytes changed while pinned")

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(frozen=True, slots=True)
class _TreeProjection:
    tree_sha256: str
    file_count: int
    directory_count: int
    symlink_count: int
    entry_count: int
    total_file_bytes: int

    def record(self, *, locator: str, root_path: Path) -> dict[str, object]:
        path_raw = os.fsencode(root_path)
        return {
            "locator": locator,
            "root_path_recorded": False,
            "root_path_utf8_bytes": len(path_raw),
            "root_path_utf8_sha256": _sha256(path_raw),
            "tree_hash_contract": TREE_HASH_CONTRACT,
            **asdict(self),
        }


@dataclass(slots=True)
class _PackagePin:
    path: Path
    root_descriptor: int
    root_identity: tuple[int, int, int, int, int]
    member_descriptors: dict[str, int]
    member_identities: dict[str, tuple[int, int, int, int, int, int, int]]
    member_bytes: dict[str, bytes]

    def revalidate(self, *, context: str) -> None:
        _revalidate_package_pin(self, context=context)

    def close(self) -> None:
        errors: list[str] = []
        for member, descriptor in self.member_descriptors.items():
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(f"{member}: {error}")
        try:
            os.close(self.root_descriptor)
        except OSError as error:
            errors.append(f"root: {error}")
        if errors:
            _fail("package descriptor cleanup failed: " + "; ".join(errors))


@dataclass(slots=True)
class _MaterializedCandidates:
    primary: Path
    independent: Path
    producer_raw: bytes
    authority_raw: bytes
    authority: dict[str, object]


def _fail(message: str) -> NoReturn:
    raise RebuttalNativeProducerError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_json_lf(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _json_exact(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""
    return _canonical_json(actual) == _canonical_json(expected)


def _json_clone(value: object) -> object:
    """Return a detached JSON value with no mutable aliases to its source."""
    return json.loads(_canonical_json(value))


def _expect_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{context} must be one lowercase SHA-256 digest")
    return value


def _expect_token(value: object, *, context: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        _fail(f"{context} must be one bounded lowercase token")
    return value


def _expect_positive_int(value: object, *, context: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        _fail(f"{context} must be one bounded positive integer")
    return value


def _expect_nonnegative_int(value: object, *, context: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        _fail(f"{context} must be one bounded nonnegative integer")
    return value


def _expect_exact_keys(
    value: Mapping[str, object], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        _fail(f"{context} has the wrong exact keys")


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _fail(f"{context} must be one JSON object")
    return value


def _sequence(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{context} must be one JSON array")
    return value


def _json_without_duplicates(raw: bytes, *, context: str) -> object:
    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"cannot decode {context}: {error}")


def _read_fd(descriptor: int, *, maximum: int, context: str) -> bytes:
    try:
        before = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _fail(f"{context} must be one single-link regular file")
    if not 0 <= before.st_size <= maximum:
        _fail(f"{context} exceeds its byte bound")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(READ_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
        except OSError as error:
            _fail(f"cannot read {context}: {error}")
        if not chunk:
            _fail(f"{context} ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_nlink,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
        before.st_nlink,
    ):
        _fail(f"{context} changed while read")
    return b"".join(chunks)


def _pin_file(
    path: Path,
    *,
    maximum: int,
    context: str,
    expected_sha256: str,
    require_executable: bool = False,
    require_root_owner: bool = False,
    require_effective_user_owner: bool = False,
    expected_mode: int | None = None,
) -> _PinnedFile:
    expected = _expect_sha256(expected_sha256, context=f"expected {context} SHA-256")
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve {context}: {error}")
    if resolved != absolute or stat.S_ISLNK(entry.st_mode):
        _fail(f"{context} path must be canonical and non-symlinked")
    descriptor = os.open(
        absolute,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not 1 <= opened.st_size <= maximum
            or (require_executable and not opened.st_mode & stat.S_IXUSR)
            or (require_root_owner and opened.st_uid != 0)
            or (require_effective_user_owner and opened.st_uid != os.geteuid())
            or (
                expected_mode is not None
                and stat.S_IMODE(opened.st_mode) != expected_mode
            )
            or opened.st_mode & 0o022
            or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            _fail(f"{context} is not one bounded canonical authority file")
        raw = _read_fd(descriptor, maximum=maximum, context=context)
        digest = _sha256(raw)
        if digest != expected:
            _fail(f"{context} differs from its caller SHA-256 anchor")
        pin = _PinnedFile(
            path=absolute,
            descriptor=descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
            size=opened.st_size,
            mtime_ns=opened.st_mtime_ns,
            mode=stat.S_IMODE(opened.st_mode),
            uid=opened.st_uid,
            sha256=digest,
        )
        pin.revalidate(context=context)
        result = pin
    except BaseException:
        os.close(descriptor)
        raise
    return result


def _pinned_bytes(pin: _PinnedFile, *, maximum: int, context: str) -> bytes:
    raw = _read_fd(pin.descriptor, maximum=maximum, context=context)
    if len(raw) != pin.size or _sha256(raw) != pin.sha256:
        _fail(f"{context} changed while pinned")
    return raw


def _close_pins(
    pins: Mapping[str, _PinnedFile], *, primary_error: BaseException | None = None
) -> None:
    errors: list[str] = []
    seen: set[int] = set()
    for name, pin in pins.items():
        if pin.descriptor in seen:
            continue
        seen.add(pin.descriptor)
        try:
            pin.close()
        except OSError as error:
            errors.append(f"{name}: {error}")
    if errors:
        message = "input descriptor cleanup failed: " + "; ".join(errors)
        if primary_error is not None:
            message = f"{primary_error}; {message}"
        raise RebuttalNativeProducerError(message) from primary_error


def _u64(value: int) -> bytes:
    if not 0 <= value < 2**64:
        _fail("tree-hash integer is outside uint64")
    return value.to_bytes(8, "big")


def _tree_projection(root: Path, *, context: str) -> _TreeProjection:
    """Inventory one bounded, non-followed Xcode dependency tree safely."""
    absolute = root.absolute()
    try:
        entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve {context}: {error}")
    if (
        resolved != absolute
        or not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != 0
        or entry.st_mode & 0o022
    ):
        _fail(f"{context} must be one canonical root-owned Xcode directory")
    digest = hashlib.sha256()
    files = 0
    directories = 1
    symlinks = 0
    entries = 1
    total_bytes = 0
    pending: list[tuple[Path, PurePosixPath, int]] = [(absolute, PurePosixPath("."), 0)]
    digest.update(
        _u64(1) + b"." + b"D" + _u64(stat.S_IMODE(entry.st_mode)) + _u64(entry.st_nlink)
    )
    while pending:
        directory, relative_directory, depth = pending.pop()
        if depth >= MAX_TREE_DEPTH:
            _fail(f"{context} exceeds its maximum depth")
        try:
            children = sorted(
                os.scandir(directory), key=lambda item: os.fsencode(item.name)
            )
        except OSError as error:
            _fail(f"cannot enumerate {context}: {error}")
        for child in children:
            name_raw = os.fsencode(child.name)
            if (
                not name_raw
                or b"/" in name_raw
                or name_raw in {b".", b".."}
                or b"\x00" in name_raw
            ):
                _fail(f"{context} contains an unsafe entry name")
            relative = (
                PurePosixPath(child.name)
                if relative_directory == PurePosixPath(".")
                else relative_directory / child.name
            )
            relative_raw = str(relative).encode("utf-8")
            try:
                observed = os.stat(  # noqa: PTH116 - preserve raw tree contract.
                    child.path,
                    follow_symlinks=False,
                )
            except OSError as error:
                _fail(f"cannot inspect {context} member: {error}")
            entries += 1
            if entries > MAX_TREE_ENTRIES:
                _fail(f"{context} exceeds its entry-count bound")
            mode = stat.S_IMODE(observed.st_mode)
            if observed.st_uid != 0 or observed.st_mode & 0o022:
                _fail(f"{context} contains a non-root-owned or writable member")
            if stat.S_ISDIR(observed.st_mode):
                directories += 1
                if directories > MAX_TREE_DIRECTORIES:
                    _fail(f"{context} exceeds its directory-count bound")
                digest.update(
                    _u64(len(relative_raw))
                    + relative_raw
                    + b"D"
                    + _u64(mode)
                    + _u64(observed.st_nlink)
                )
                pending.append((Path(child.path), relative, depth + 1))
                continue
            if stat.S_ISLNK(observed.st_mode):
                symlinks += 1
                if symlinks > MAX_TREE_SYMLINKS:
                    _fail(f"{context} exceeds its symlink-count bound")
                target = os.fsencode(
                    os.readlink(child.path),  # noqa: PTH115 - preserve raw target.
                )
                if not target or target.startswith(b"/") or b"\x00" in target:
                    _fail(f"{context} contains an absolute or unsafe symlink")
                try:
                    Path(child.path).resolve(strict=True).relative_to(absolute)
                except (OSError, ValueError) as error:
                    _fail(f"{context} symlink escapes or is broken: {error}")
                digest.update(
                    _u64(len(relative_raw))
                    + relative_raw
                    + b"L"
                    + _u64(mode)
                    + _u64(observed.st_nlink)
                    + _u64(len(target))
                    + target
                )
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink < 1:
                _fail(f"{context} contains a special member")
            files += 1
            if files > MAX_TREE_FILES:
                _fail(f"{context} exceeds its file-count bound")
            descriptor = os.open(
                child.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_mode,
                    opened.st_nlink,
                ) != (
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_mode,
                    observed.st_nlink,
                ):
                    _fail(f"{context} member changed while opened")
                if not 0 <= opened.st_size <= MAX_TREE_BYTES:
                    _fail(f"{context} member exceeds its byte bound")
                chunks: list[bytes] = []
                read_offset = 0
                while read_offset < opened.st_size:
                    chunk = os.pread(
                        descriptor,
                        min(READ_CHUNK_BYTES, opened.st_size - read_offset),
                        read_offset,
                    )
                    if not chunk:
                        _fail(f"{context} member ended early")
                    chunks.append(chunk)
                    read_offset += len(chunk)
                raw = b"".join(chunks)
                terminal = os.fstat(descriptor)
                if (
                    terminal.st_dev,
                    terminal.st_ino,
                    terminal.st_size,
                    terminal.st_mtime_ns,
                    terminal.st_mode,
                    terminal.st_nlink,
                ) != (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_mode,
                    opened.st_nlink,
                ):
                    _fail(f"{context} member changed while read")
            finally:
                os.close(descriptor)
            total_bytes += len(raw)
            if total_bytes > MAX_TREE_BYTES:
                _fail(f"{context} exceeds its aggregate byte bound")
            digest.update(
                _u64(len(relative_raw))
                + relative_raw
                + b"F"
                + _u64(mode)
                + _u64(observed.st_nlink)
                + _u64(len(raw))
                + raw
            )
    terminal = os.lstat(absolute)
    if (
        terminal.st_dev,
        terminal.st_ino,
        terminal.st_mtime_ns,
        terminal.st_mode,
    ) != (entry.st_dev, entry.st_ino, entry.st_mtime_ns, entry.st_mode):
        _fail(f"{context} root changed while inventoried")
    return _TreeProjection(
        tree_sha256=digest.hexdigest(),
        file_count=files,
        directory_count=directories,
        symlink_count=symlinks,
        entry_count=entries,
        total_file_bytes=total_bytes,
    )


def _require_tree_anchor(
    root: Path, *, expected_sha256: str, context: str
) -> _TreeProjection:
    expected = _expect_sha256(expected_sha256, context=f"expected {context} SHA-256")
    projection = _tree_projection(root, context=context)
    if projection.tree_sha256 != expected:
        _fail(f"{context} differs from its caller tree SHA-256 anchor")
    return projection


def _tool_record(
    pin: _PinnedFile,
    *,
    locator: str,
    binary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "locator": locator,
        "path_recorded": False,
        "bytes": pin.size,
        "sha256": pin.sha256,
        "mode": f"{pin.mode:04o}",
        "uid": pin.uid,
        "link_count": 1,
        "binary": dict(binary),
    }


def _parse_thin_macho_header(raw: bytes, *, context: str) -> dict[str, object]:
    if len(raw) < 32:
        _fail(f"{context} lacks a complete Mach-O header")
    try:
        magic, cpu_type, cpu_subtype, file_type, commands, command_bytes, _, _ = (
            struct.unpack_from("<IiiIIIII", raw)
        )
    except struct.error as error:  # pragma: no cover - size guarded.
        _fail(f"cannot parse {context} Mach-O header: {error}")
    if magic != MH_MAGIC_64 or file_type != MH_EXECUTE:
        _fail(f"{context} must be one thin Mach-O executable")
    if not 1 <= commands <= MAX_MACH_LOAD_COMMANDS:
        _fail(f"{context} load-command count is out of bounds")
    command_end = 32 + command_bytes
    if command_end > len(raw) or command_bytes < commands * 8:
        _fail(f"{context} load-command extent is invalid")
    offset = 32
    for _ in range(commands):
        if offset + 8 > command_end:
            _fail(f"{context} load-command table is truncated")
        _, command_size = struct.unpack_from("<II", raw, offset)
        if command_size < 8 or command_size % 8 or offset + command_size > command_end:
            _fail(f"{context} contains an invalid load command")
        offset += command_size
    if offset != command_end:
        _fail(f"{context} load-command table has trailing bytes")
    architecture = {
        CPU_TYPE_ARM64: "arm64",
        CPU_TYPE_X86_64: "x86_64",
    }.get(cpu_type)
    if architecture is None:
        _fail(f"{context} has an unsupported CPU type")
    unsigned_subtype = cpu_subtype & 0xFFFFFFFF
    return {
        "binary_container": "thin-macho64",
        "architecture": architecture,
        "cpu_type": cpu_type,
        "cpu_subtype": unsigned_subtype & 0x00FFFFFF,
        "cpu_subtype_capabilities": unsigned_subtype & 0xFF000000,
        "file_type": "execute",
        "load_command_count": commands,
        "load_command_bytes": command_bytes,
    }


def _code_signature_command_count(raw: bytes, *, context: str) -> int:
    header = _parse_thin_macho_header(raw, context=context)
    offset = 32
    count = 0
    for _ in range(int(header["load_command_count"])):
        command, command_size = struct.unpack_from("<II", raw, offset)
        if command == LC_CODE_SIGNATURE:
            count += 1
        offset += command_size
    return count


def _required_macho_uuid(raw: bytes, *, context: str) -> str:
    header = _parse_thin_macho_header(raw, context=context)
    offset = 32
    observed: bytes | None = None
    for _ in range(int(header["load_command_count"])):
        command, command_size = struct.unpack_from("<II", raw, offset)
        if command == LC_UUID:
            if observed is not None or command_size != 24:
                _fail(f"{context} must contain exactly one canonical LC_UUID")
            observed = raw[offset + 8 : offset + 24]
        offset += command_size
    if observed is None or observed == b"\0" * 16:
        _fail(f"{context} must contain one nonzero LC_UUID")
    hexadecimal = observed.hex()
    return "-".join(
        (
            hexadecimal[:8],
            hexadecimal[8:12],
            hexadecimal[12:16],
            hexadecimal[16:20],
            hexadecimal[20:],
        )
    )


def _parse_fat_codesign(raw: bytes) -> dict[str, object]:
    """Parse strict FAT32/FAT64 provenance and select one arm64e slice."""
    if len(raw) < 8:
        _fail("codesign lacks one complete FAT header")
    magic, count = struct.unpack_from(">II", raw)
    if magic not in {FAT_MAGIC, FAT_MAGIC_64} or not 1 <= count <= MAX_FAT_SLICES:
        _fail("codesign is not one bounded big-endian FAT executable")
    is_64 = magic == FAT_MAGIC_64
    entry_size = 32 if is_64 else 20
    header_end = 8 + count * entry_size
    if header_end > len(raw):
        _fail("codesign FAT architecture table is truncated")
    slices: list[dict[str, object]] = []
    extents: list[tuple[int, int]] = []
    selected: dict[str, object] | None = None
    for index in range(count):
        start = 8 + index * entry_size
        if is_64:
            cpu_type, cpu_subtype, offset, size, align, reserved = struct.unpack_from(
                ">iiQQII", raw, start
            )
            if reserved != 0:
                _fail("codesign FAT64 entry has nonzero reserved bits")
        else:
            cpu_type, cpu_subtype, offset, size, align = struct.unpack_from(
                ">iiIII", raw, start
            )
        if (
            not size
            or align > 31
            or offset < header_end
            or offset % (1 << align)
            or offset + size > len(raw)
        ):
            _fail("codesign FAT slice extent or alignment is invalid")
        extent = (offset, offset + size)
        if any(
            not (extent[1] <= other[0] or extent[0] >= other[1]) for other in extents
        ):
            _fail("codesign FAT slices overlap")
        extents.append(extent)
        slice_raw = raw[offset : offset + size]
        header = _parse_thin_macho_header(slice_raw, context="codesign slice")
        subtype = cpu_subtype & 0x00FFFFFF
        subtype_capabilities = cpu_subtype & 0xFF000000
        if (
            int(header["cpu_type"]) != cpu_type
            or int(header["cpu_subtype"]) != subtype
            or int(header["cpu_subtype_capabilities"]) != subtype_capabilities
        ):
            _fail("codesign FAT entry CPU identity differs from its Mach-O slice")
        record = {
            "index": index,
            "architecture": header["architecture"],
            "cpu_type": cpu_type,
            "cpu_subtype": subtype,
            "cpu_subtype_capabilities": subtype_capabilities,
            "alignment_exponent": align,
            "offset": offset,
            "bytes": size,
            "sha256": _sha256(slice_raw),
        }
        slices.append(record)
        if cpu_type == CPU_TYPE_ARM64 and subtype == CPU_SUBTYPE_ARM64E:
            if selected is not None:
                _fail("codesign has duplicate arm64e slices")
            selected = record
    if selected is None:
        _fail("codesign has no selected arm64e executable slice")
    return {
        "binary_container": "fat-macho64" if is_64 else "fat-macho32",
        "fat_endianness": "big",
        "slice_count": count,
        "slices": slices,
        "selected_execution_slice": selected,
        "selected_slice_live_mapping": "not-attested",
    }


def _parse_signed_arm64_launcher(raw: bytes) -> dict[str, object]:
    """Strictly parse the final thin arm64 executable and primary CodeDirectory."""
    header = _parse_thin_macho_header(raw, context="signed launcher")
    if (
        header["architecture"] != "arm64"
        or header["cpu_subtype"] != CPU_SUBTYPE_ARM64_ALL
        or header["cpu_subtype_capabilities"] != 0
    ):
        _fail("signed launcher must target generic thin arm64")
    _, _, _, _, commands, command_bytes, _, _ = struct.unpack_from("<IiiIIIII", raw)
    command_end = 32 + command_bytes
    offset = 32
    signature_extent: tuple[int, int] | None = None
    for _ in range(commands):
        command, command_size = struct.unpack_from("<II", raw, offset)
        if command == LC_CODE_SIGNATURE:
            if signature_extent is not None or command_size != 16:
                _fail("signed launcher must contain one canonical code signature")
            data_offset, data_size = struct.unpack_from("<II", raw, offset + 8)
            signature_extent = (data_offset, data_size)
        offset += command_size
    if signature_extent is None:
        _fail("signed launcher has no embedded code signature")
    data_offset, data_size = signature_extent
    if (
        data_size < 20
        or data_offset < command_end
        or data_offset % 16
        or data_offset + data_size != len(raw)
    ):
        _fail("signed launcher code signature must be aligned at canonical EOF")
    signature = raw[data_offset:]
    magic, signature_length, count = struct.unpack_from(">III", signature)
    if (
        magic != CSMAGIC_EMBEDDED_SIGNATURE
        or not 20 <= signature_length <= len(signature)
        or any(signature[signature_length:])
        or not 1 <= count <= MAX_SIGNATURE_BLOBS
        or 12 + count * 8 > signature_length
    ):
        _fail("signed launcher SuperBlob is malformed")
    superblob = signature[:signature_length]
    index_end = 12 + count * 8
    seen_slots: set[int] = set()
    extents: list[tuple[int, int]] = []
    code_directory: bytes | None = None
    for index in range(count):
        slot, blob_offset = struct.unpack_from(">II", superblob, 12 + index * 8)
        if (
            slot in seen_slots
            or blob_offset < index_end
            or blob_offset + 8 > len(superblob)
        ):
            _fail("signed launcher SuperBlob index is invalid")
        seen_slots.add(slot)
        blob_magic, blob_size = struct.unpack_from(">II", superblob, blob_offset)
        extent = (blob_offset, blob_offset + blob_size)
        if (
            blob_size < 8
            or extent[1] > len(superblob)
            or any(
                not (extent[1] <= other[0] or extent[0] >= other[1])
                for other in extents
            )
        ):
            _fail("signed launcher SuperBlob members overlap or escape bounds")
        extents.append(extent)
        if blob_magic == CSMAGIC_CODEDIRECTORY:
            if slot != CSSLOT_CODEDIRECTORY or code_directory is not None:
                _fail("signed launcher has alternate or duplicate CodeDirectories")
            code_directory = superblob[extent[0] : extent[1]]
    if code_directory is None or len(code_directory) < 88:
        _fail("signed launcher has no complete primary CodeDirectory")
    (
        cd_magic,
        cd_length,
        version,
        flags,
        hash_offset,
        identifier_offset,
        special_slots,
        code_slots,
        code_limit,
    ) = struct.unpack_from(">9I", code_directory)
    hash_size = code_directory[36]
    hash_type = code_directory[37]
    platform = code_directory[38]
    page_exponent = code_directory[39]
    spare2, scatter_offset, team_offset, spare3 = struct.unpack_from(
        ">4I", code_directory, 40
    )
    code_limit_64, exec_segment_base, exec_segment_limit, exec_segment_flags = (
        struct.unpack_from(">4Q", code_directory, 56)
    )
    if (
        cd_magic != CSMAGIC_CODEDIRECTORY
        or cd_length != len(code_directory)
        or version != 0x20400
        or flags != 0x202
        or hash_type != 2
        or hash_size != 32
        or platform != 0
        or page_exponent != 14
        or spare2 != 0
        or scatter_offset != 0
        or team_offset != 0
        or spare3 != 0
        or code_limit_64 != 0
        or exec_segment_base != 0
        or not 0 < exec_segment_limit <= code_limit
        or exec_segment_flags != 1
        or code_limit != data_offset
        or identifier_offset < 88
        or identifier_offset >= len(code_directory)
        or code_directory.find(b"\0", identifier_offset) < 0
        or code_directory.find(b"\0", identifier_offset)
        >= hash_offset - special_slots * hash_size
        or hash_offset < special_slots * hash_size
        or hash_offset + code_slots * hash_size != len(code_directory)
        or code_slots != (code_limit + (1 << page_exponent) - 1) >> page_exponent
    ):
        _fail("signed launcher CodeDirectory fields are outside the strong contract")
    identifier_end = code_directory.find(b"\0", identifier_offset)
    if code_directory[identifier_offset:identifier_end] != SIGNATURE_IDENTIFIER.encode(
        "ascii"
    ):
        _fail("signed launcher CodeDirectory identifier drifted")
    page_size = 1 << page_exponent
    for slot in range(code_slots):
        start = slot * page_size
        observed_hash = code_directory[
            hash_offset + slot * hash_size : hash_offset + (slot + 1) * hash_size
        ]
        expected_hash = hashlib.sha256(
            raw[start : min(start + page_size, code_limit)]
        ).digest()
        if observed_hash != expected_hash:
            _fail("signed launcher CodeDirectory code-slot hash is invalid")
    cdhash = hashlib.sha256(code_directory).digest()[:CS_CDHASH_LEN].hex()
    return {
        "binary_container": "thin-macho64",
        "architecture": "arm64",
        "cpu_subtype": "all",
        "hash_type": "sha256",
        "code_directory_flags": flags,
        "code_directory_bytes": len(code_directory),
        "cdhash": cdhash,
        "code_limit": code_limit,
        "code_slots": code_slots,
        "page_size": page_size,
        "signature_offset": data_offset,
        "signature_bytes": data_size,
    }


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, ChildProcessError):
        process.wait(timeout=5)


def _wait_for_exit_without_reaping(
    process_id: int,
    *,
    deadline: float,
    context: str,
) -> None:
    _require_darwin_siginfo_layout()
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    library = ctypes.CDLL(None, use_errno=True)
    library.waitid.argtypes = [
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_DarwinSigInfo),
        ctypes.c_int,
    ]
    library.waitid.restype = ctypes.c_int
    while True:
        status = _DarwinSigInfo()
        ctypes.set_errno(0)
        result = library.waitid(1, process_id, ctypes.byref(status), options)
        if result != 0:
            error_number = ctypes.get_errno()
            _fail(
                f"{context} waitid failed before process-group containment: "
                f"{os.strerror(error_number)} (errno {error_number})"
            )
        if status.process_id == process_id:
            return
        if status.process_id != 0:
            _fail(f"{context} waitid observed an unexpected child process")
        if time.monotonic() >= deadline:
            _fail(f"{context} exceeded its timeout")
        time.sleep(0.01)


def _require_darwin_siginfo_layout() -> None:
    layout = {
        "size": ctypes.sizeof(_DarwinSigInfo),
        "pid_offset": _DarwinSigInfo.process_id.offset,
        "status_offset": _DarwinSigInfo.status.offset,
    }
    if layout != {"size": 104, "pid_offset": 12, "status_offset": 20}:
        _fail(f"Darwin siginfo ABI layout is unsupported: {layout}")


def _darwin_process_group_members(process_group: int) -> tuple[int, ...]:
    """List one bounded Darwin process group after a signal permission result."""
    if sys.platform != "darwin":
        _fail("Darwin process-group enumeration is unavailable")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_listpgrppids.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.proc_listpgrppids.restype = ctypes.c_int
    buffer = (ctypes.c_int * MAX_PROCESS_GROUP_MEMBERS)()
    ctypes.set_errno(0)
    count = library.proc_listpgrppids(
        process_group,
        buffer,
        ctypes.sizeof(buffer),
    )
    if count < 0:
        error_number = ctypes.get_errno()
        _fail(
            "cannot enumerate process group after signal denial: "
            f"{os.strerror(error_number)} (errno {error_number})"
        )
    if count >= MAX_PROCESS_GROUP_MEMBERS:
        _fail("process group exceeds its member bound")
    members = tuple(buffer[:count])
    if any(process_id <= 0 for process_id in members) or len(set(members)) != len(
        members
    ):
        _fail("process group enumeration contains invalid or duplicate PIDs")
    return tuple(sorted(members))


def _require_process_group_leader_only(
    process_id: int,
    *,
    deadline: float,
    wait_for_signaled_members: bool,
) -> None:
    """Require the WNOWAIT-held leader to be the only exact PGID member twice."""
    while True:
        members = _darwin_process_group_members(process_id)
        if members == (process_id,):
            break
        if not wait_for_signaled_members or time.monotonic() >= deadline:
            _fail("process group membership is not the retained leader alone")
        time.sleep(0.01)
    if _darwin_process_group_members(process_id) != (process_id,):
        _fail("process group membership changed before parent reap")


def _run_bounded_tool(
    pin: _PinnedFile,
    arguments: Sequence[str],
    *,
    context: str,
    before: Callable[[], None],
    after: Callable[[], None],
    timeout: float = TOOL_TIMEOUT_SECONDS,
    stdout_limit: int = MAX_TOOL_OUTPUT_BYTES,
    stderr_limit: int = MAX_TOOL_OUTPUT_BYTES,
    require_empty_output: bool = True,
    environment: Mapping[str, str] = EXACT_TOOL_ENVIRONMENT,
) -> tuple[dict[str, object], bytes, bytes]:
    """Run one exact pinned tool without a shell and bound its process group."""
    before()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [str(pin.path), *arguments],
            executable=str(pin.path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=dict(environment),
            close_fds=True,
            start_new_session=True,
        )
    except OSError as error:
        _fail(f"cannot start {context}: {error}")
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        _kill_process_group(process)
        _fail(f"{context} has no bounded output pipes")
    selector: selectors.BaseSelector | None = None
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    primary: BaseException | None = None
    return_code: int | None = None
    try:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)
        while streams:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                _fail(f"{context} exceeded its timeout")
            events = selector.select(min(remaining, 0.1))
            for key, _ in events:
                descriptor = key.fd
                label = streams[descriptor]
                try:
                    chunk = os.read(descriptor, READ_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    streams.pop(descriptor, None)
                    continue
                outputs[label].extend(chunk)
                if len(outputs[label]) > limits[label]:
                    _fail(f"{context} {label} exceeds its byte bound")
        _wait_for_exit_without_reaping(
            process.pid,
            deadline=started + timeout,
            context=context,
        )
    except BaseException as error:  # noqa: BLE001 - preserve and reap child.
        primary = error
    cleanup_errors: list[str] = []
    process_group_signaled = False
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process_group_signaled = True
    except (PermissionError, ProcessLookupError):
        process_group_signaled = False
    except OSError as error:
        cleanup_errors.append(f"kill process group: {error}")
    try:
        _require_process_group_leader_only(
            process.pid,
            deadline=time.monotonic() + 5,
            wait_for_signaled_members=process_group_signaled,
        )
    except RebuttalNativeProducerError as error:
        cleanup_errors.append(str(error))
    try:
        observed_return_code = process.wait(timeout=5)
        if primary is None:
            return_code = observed_return_code
    except (subprocess.TimeoutExpired, ChildProcessError) as error:
        cleanup_errors.append(f"reap process group: {error}")
    if selector is not None:
        try:
            selector.close()
        except OSError as error:
            cleanup_errors.append(f"selector close: {error}")
    for label, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        try:
            stream.close()
        except OSError as error:
            cleanup_errors.append(f"{label} close: {error}")
    if primary is not None or cleanup_errors:
        message = str(primary) if primary is not None else f"{context} cleanup failed"
        if cleanup_errors:
            message = f"{message}; " + "; ".join(cleanup_errors)
        raise RebuttalNativeProducerError(message) from primary
    if return_code is None:  # pragma: no cover - exhaustiveness.
        _fail(f"{context} completed without a return code")
    after()
    stdout = bytes(outputs["stdout"])
    stderr = bytes(outputs["stderr"])
    if return_code != 0:
        _fail(f"{context} exited with status {return_code}")
    if require_empty_output and (stdout or stderr):
        _fail(f"{context} emitted unexpected stdout or stderr")
    return (
        {
            "return_code": 0,
            "stdout_bytes": len(stdout),
            "stdout_sha256": _sha256(stdout),
            "stderr_bytes": len(stderr),
            "stderr_sha256": _sha256(stderr),
        },
        stdout,
        stderr,
    )


def _path_binding(path: Path, *, locator: str) -> dict[str, object]:
    raw = os.fsencode(path)
    return {
        "locator": locator,
        "absolute_path_recorded": False,
        "absolute_path_utf8_bytes": len(raw),
        "absolute_path_utf8_sha256": _sha256(raw),
    }


def _validate_embedded_path(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
        entry = os.lstat(absolute)
    except OSError as error:
        _fail(f"cannot resolve embedded {context} path: {error}")
    raw = os.fsencode(absolute)
    if (
        resolved != absolute
        or stat.S_ISLNK(entry.st_mode)
        or SAFE_PATH_RE.fullmatch(os.fsdecode(raw)) is None
        or any(value in raw for value in (b'"', b"\\", b"\n", b"\r", b"\0"))
    ):
        _fail(f"embedded {context} path is not canonical safe ASCII")
    return absolute


def _require_release_member_paths(
    *,
    mode: str,
    repo_root: Path,
    renderer_path: Path,
    machine_runner_path: Path,
) -> None:
    """Require revision inputs to be the exact canonical repository members."""
    if mode != MODE_REVISION:
        return
    expected_renderer = repo_root / "analysis/render_tcga_revision_rebuttal.py"
    expected_machine = repo_root / MACHINE_RUNNER_MEMBER
    if renderer_path != expected_renderer or machine_runner_path != expected_machine:
        _fail(
            "revision renderer and machine runner must be exact canonical "
            "repository member paths"
        )


def _bundle_projection(
    bundle_raw: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    parsed = _json_without_duplicates(bundle_raw, context="rebuttal derivation bundle")
    if not isinstance(parsed, dict):
        _fail("rebuttal derivation bundle must be one JSON object")
    normalized = renderer._normalize_derivation_bundle(parsed)
    if renderer._canonical_json(normalized) != bundle_raw:
        _fail("rebuttal derivation bundle is not exact canonical JSON")
    input_projection = [
        {
            "member": item["member"],
            "encoding": item["encoding"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "encoded_payload_sha256": _sha256(str(item["base64"]).encode("ascii")),
        }
        for item in normalized["canonical_inputs"]
    ]
    input_projection_sha = _sha256(_canonical_json(input_projection))
    projection = {
        "schema": normalized["schema"],
        "contract": normalized["contract"],
        "release_id": normalized["release_id"],
        "role": normalized["role"],
        "producer_protocol": normalized["producer_protocol"],
        "producer_arguments": normalized["producer_arguments"],
        "canonical_inputs": input_projection,
        "canonical_inputs_projection_sha256": input_projection_sha,
        "dependencies": normalized["dependencies"],
        "expected_output": normalized["expected_output"],
        "non_inference": normalized["non_inference"],
        "source_or_base64_payload_recorded": False,
    }
    projection["bundle_projection_sha256"] = _sha256(_canonical_json(projection))
    return normalized, projection


def _launcher_config(
    *,
    runtime: _PinnedFile,
    runtime_path: Path,
    renderer_pin: _PinnedFile,
    renderer_path: Path,
    bundle: Mapping[str, object],
) -> dict[str, object]:
    runtime_record = _mapping(
        _mapping(bundle["dependencies"], context="bundle dependencies")["runtime"],
        context="bundle runtime",
    )
    renderer_record = _mapping(
        _mapping(bundle["dependencies"], context="bundle dependencies")["renderer"],
        context="bundle renderer",
    )
    if (
        runtime_record["sha256"] != runtime.sha256
        or runtime_record["bytes"] != runtime.size
        or renderer_record["sha256"] != renderer_pin.sha256
        or renderer_record["bytes"] != renderer_pin.size
    ):
        _fail("bundle runtime or renderer anchors differ from exact launcher inputs")
    config: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "role": PDF_ID,
        "argument_count_including_argv0": 9,
        "producer_arguments": _json_clone(PRODUCER_ARGUMENTS),
        "source_fd": {
            "canonical_decimal": True,
            "minimum": 3,
            "maximum": 2**31 - 1,
            "access": "O_RDONLY; O_NONBLOCK permitted for regular file",
            "type": "regular",
            "mode": "0400",
            "owner": "effective-user-id",
            "link_count": 1,
            "minimum_bytes": 1,
            "maximum_bytes": renderer.MAX_DERIVATION_BUNDLE_BYTES,
            "seekable_and_rewound": True,
            "cloexec": "cleared-only-after-complete-validation",
        },
        "runtime": {
            **_path_binding(runtime_path, locator=str(runtime_record["locator"])),
            "bytes": runtime.size,
            "sha256": runtime.sha256,
            "mode": f"{runtime.mode:04o}",
            "owner": "effective-user-id",
            "link_count": 1,
            "pre_exec_descriptor_hash": True,
        },
        "renderer": {
            **_path_binding(renderer_path, locator=str(renderer_record["locator"])),
            "bytes": renderer_pin.size,
            "sha256": renderer_pin.sha256,
            "mode": f"{renderer_pin.mode:04o}",
            "owner": "effective-user-id",
            "link_count": 1,
            "pre_exec_descriptor_hash": True,
        },
        "cwd": "/",
        "environment": _json_clone(EXACT_LAUNCH_ENVIRONMENT),
        "environment_inherited": False,
        "shell": False,
        "process_operation": "execve-only",
        "unexpected_inherited_fds": "enumerate-/dev/fd-and-close-up-to-64",
        "stdout": "inherited-pdf-stream",
        "stderr": "inherited-and-launcher-emits-no-bytes",
        "stdin": "inherited-but-not-read-by-launcher",
        "failure_codes": {
            "64": "argument-protocol-or-fd-token",
            "65": "source-descriptor",
            "66": "cwd",
            "67": "runtime-preflight",
            "68": "renderer-preflight",
            "69": "unexpected-fd-cleanup",
            "126": "execve",
        },
    }
    config["launcher_config_sha256"] = _sha256(_canonical_json(config))
    return config


def _runtime_handoff(config: Mapping[str, object]) -> dict[str, object]:
    handoff: dict[str, object] = {
        "execve_path": "{runtime}",
        "execve_argv": [
            "{runtime}",
            "-I",
            "-S",
            "-B",
            "{renderer}",
            *PRODUCER_ARGUMENTS,
        ],
        "placeholder_bindings": {
            "{runtime}": _json_clone(config["runtime"]),
            "{renderer}": _json_clone(config["renderer"]),
            "{source_fd}": "validated-original-canonical-decimal-descriptor",
        },
        "cwd": "/",
        "environment": _json_clone(EXACT_LAUNCH_ENVIRONMENT),
        "inherit_environment": False,
        "PATH_lookup": False,
        "shell": False,
        "stdout": "inherited",
        "stderr": "inherited",
        "source_fd": _json_clone(config["source_fd"]),
    }
    handoff["runtime_handoff_sha256"] = _sha256(_canonical_json(handoff))
    return handoff


def _c_string_macro(name: str, value: str) -> str:
    if not value or any(
        character in value for character in ('"', "\\", "\n", "\r", "\0")
    ):
        _fail(f"compile-time macro {name} contains an unsafe character")
    return f'-D{name}="{value}"'


def _recipe_records() -> dict[str, object]:
    compile_recipe = [
        "{clang}",
        "-arch",
        "arm64",
        "-target",
        "arm64-apple-macos13.0",
        "--no-default-config",
        "-std=c11",
        "-Os",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-fno-ident",
        "-fno-common",
        "-fvisibility=hidden",
        "-g0",
        "-isysroot",
        "{sdk_root}",
        "-resource-dir",
        "{compiler_resource_root}",
        "-ffile-prefix-map={stage_root}=/dialect/native-producer",
        "-fdebug-prefix-map={stage_root}=/dialect/native-producer",
        "-x",
        "c",
        "{launcher_source}",
        "-c",
        "-o",
        "{object}",
        '-DDIALECT_RUNTIME_PATH="{runtime}"',
        '-DDIALECT_RUNTIME_SHA256="{runtime_sha256}"',
        "-DDIALECT_RUNTIME_BYTES={runtime_bytes}",
        "-DDIALECT_RUNTIME_MODE={runtime_mode}",
        '-DDIALECT_RENDERER_PATH="{renderer}"',
        '-DDIALECT_RENDERER_SHA256="{renderer_sha256}"',
        "-DDIALECT_RENDERER_BYTES={renderer_bytes}",
        "-DDIALECT_RENDERER_MODE={renderer_mode}",
    ]
    link_recipe = [
        "{ld}",
        "-arch",
        "arm64",
        "-syslibroot",
        "{sdk_root}",
        "-platform_version",
        "macos",
        MACOS_MINIMUM,
        "{sdk_version}",
        "-lSystem",
        "-dead_strip",
        "-no_adhoc_codesign",
        "-o",
        "{unsigned_executable}",
        "{object}",
    ]
    sign_recipe = [
        "{codesign}",
        "--force",
        "--sign",
        "-",
        "--options",
        "kill",
        "--timestamp=none",
        "--identifier",
        SIGNATURE_IDENTIFIER,
        "--verbose=0",
        "{unsigned_executable}",
    ]
    verify_recipe = [
        "{codesign}",
        "--verify",
        "--strict",
        "--verbose=0",
        "{signed_executable}",
    ]
    return {
        "compile": {
            "argv": compile_recipe,
            "argv_sha256": _sha256(_canonical_json(compile_recipe)),
        },
        "link": {
            "argv": link_recipe,
            "argv_sha256": _sha256(_canonical_json(link_recipe)),
        },
        "sign": {
            "argv": sign_recipe,
            "argv_sha256": _sha256(_canonical_json(sign_recipe)),
        },
        "verify": {
            "argv": verify_recipe,
            "argv_sha256": _sha256(_canonical_json(verify_recipe)),
        },
    }


def _normalized_recipe_invocation(
    operation: str,
    executable: Path,
    arguments: Sequence[str],
    *,
    bindings: Mapping[str, str],
) -> dict[str, object]:
    recipes = _recipe_records()
    recipe = _mapping(recipes[operation], context=f"{operation} recipe")
    normalized = _sequence(recipe["argv"], context=f"{operation} recipe argv")
    expanded: list[str] = []
    for raw_token in normalized:
        if not isinstance(raw_token, str):
            _fail(f"{operation} recipe contains a non-string token")
        token = raw_token
        for placeholder, replacement in bindings.items():
            token = token.replace(placeholder, replacement)
        if "{" in token or "}" in token:
            _fail(f"{operation} recipe has an unbound placeholder")
        expanded.append(token)
    if [str(executable), *arguments] != expanded:
        _fail(f"{operation} invocation differs from its exact normalized recipe")
    return {
        "normalized_argv": normalized,
        "normalized_argv_sha256": recipe["argv_sha256"],
    }


def _git_environment() -> dict[str, str]:
    return {
        **EXACT_TOOL_ENVIRONMENT,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run_git(
    git: _PinnedFile,
    arguments: Sequence[str],
    *,
    context: str,
    guard: Callable[[], None],
    stdout_limit: int,
) -> bytes:
    """Run bounded read-only Git plumbing under an inert configuration."""
    _, stdout, stderr = _run_bounded_tool(
        git,
        arguments,
        context=context,
        before=guard,
        after=guard,
        timeout=30.0,
        stdout_limit=stdout_limit,
        stderr_limit=MAX_TOOL_OUTPUT_BYTES,
        require_empty_output=False,
        environment=_git_environment(),
    )
    if stderr:
        _fail(f"{context} emitted unexpected stderr")
    return stdout


def _source_release_projection(
    *,
    mode: str,
    release_commit: str | None,
    release_ref: str | None,
    git: _PinnedFile,
    repo_root: Path,
    file_pins: Mapping[str, _PinnedFile],
    guard: Callable[[], None],
) -> dict[str, object]:
    if mode == MODE_SYNTHETIC:
        if release_commit is not None or release_ref is not None:
            _fail("synthetic-canary mode must not claim a release commit or ref")
        projection: dict[str, object] = {
            "status": SYNTHETIC_SOURCE_STATUS,
            "release_commit": None,
            "release_ref": None,
            "git_blob_equality": False,
            "members": [
                {
                    "member": member,
                    "bytes": file_pins[member].size,
                    "sha256": file_pins[member].sha256,
                }
                for member in RELEVANT_RELEASE_MEMBERS
            ],
            "git": {
                "locator": "xcode-git",
                "bytes": git.size,
                "sha256": git.sha256,
                "main-executable-bytes-pinned": True,
            },
        }
        projection["source_release_projection_sha256"] = _sha256(
            _canonical_json(projection)
        )
        return projection
    if (
        release_commit is None
        or COMMIT_RE.fullmatch(release_commit) is None
        or release_ref is None
        or SAFE_REF_RE.fullmatch(release_ref) is None
        or release_ref.startswith("-")
        or ".." in release_ref
        or "//" in release_ref
    ):
        _fail("revision mode requires one exact caller-pinned commit and safe ref")
    repo_raw = os.fsencode(repo_root)
    common = [
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.excludesFile=/dev/null",
        "-C",
        os.fsdecode(repo_raw),
    ]
    resolved = _run_git(
        git,
        [*common, "rev-parse", "--verify", f"{release_ref}^{{commit}}"],
        context="release-ref resolution",
        guard=guard,
        stdout_limit=128,
    )
    if resolved != f"{release_commit}\n".encode("ascii"):
        _fail("release ref does not resolve to the caller-pinned commit")
    members: list[dict[str, object]] = []
    for member in RELEVANT_RELEASE_MEMBERS:
        pin = file_pins[member]
        blob = _run_git(
            git,
            [*common, "show", f"{release_commit}:{member}"],
            context=f"release Git blob {member}",
            guard=guard,
            stdout_limit=MAX_GIT_BLOB_BYTES,
        )
        current = _pinned_bytes(pin, maximum=MAX_GIT_BLOB_BYTES, context=member)
        if blob != current:
            _fail(f"worktree file {member} differs from the pinned release blob")
        members.append(
            {
                "member": member,
                "bytes": len(blob),
                "sha256": _sha256(blob),
            }
        )
    projection: dict[str, object] = {
        "status": ("git-command-observed-listed-path-byte-equality-at-caller-commit"),
        "release_commit": release_commit,
        "release_ref": release_ref,
        "git_blob_equality": True,
        "members": members,
        "git": {
            "locator": "xcode-git",
            "bytes": git.size,
            "sha256": git.sha256,
            "main-executable-bytes-pinned": True,
        },
    }
    projection["source_release_projection_sha256"] = _sha256(
        _canonical_json(projection)
    )
    return projection


def _safe_destination_parent(destination: Path) -> tuple[Path, _ParentPin]:
    absolute = destination.absolute()
    if absolute.name in {"", ".", ".."}:
        _fail("package destination must name one new directory")
    parent = absolute.parent
    try:
        entry = os.lstat(parent)
        resolved = parent.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect package destination parent: {error}")
    if (
        resolved != parent
        or not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != os.geteuid()
        or entry.st_mode & 0o022
    ):
        _fail("package destination or parent is not a safe new authority")
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        pin = _ParentPin(
            path=parent,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            mode=stat.S_IMODE(entry.st_mode),
            uid=entry.st_uid,
        )
        if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
            _fail("package destination parent changed while opened")
        pin.revalidate(context="package destination parent")
        pin.require_absent(
            absolute.name,
            context="package destination",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return absolute, pin


def _write_new_member(path: Path, raw: bytes, *, mode: int, context: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        renderer._write_all(descriptor, raw, context=context)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != mode
            or observed.st_size != len(raw)
        ):
            _fail(f"{context} did not materialize exactly")
    finally:
        os.close(descriptor)


def _replace_owned_member(path: Path, raw: bytes, *, mode: int, context: str) -> None:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
    ):
        _fail(f"{context} is not one owned single-link stage member")
    path.chmod(0o600)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_TRUNC
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _fail(f"{context} identity changed before replacement")
        os.fchmod(descriptor, mode)
        renderer._write_all(descriptor, raw, context=context)
        os.fsync(descriptor)
        terminal = os.fstat(descriptor)
        if (
            terminal.st_size != len(raw)
            or stat.S_IMODE(terminal.st_mode) != mode
            or terminal.st_nlink != 1
        ):
            _fail(f"{context} replacement did not seal exactly")
    finally:
        os.close(descriptor)


def _read_stage_member(path: Path, *, maximum: int, context: str) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        return _read_fd(descriptor, maximum=maximum, context=context)
    finally:
        os.close(descriptor)


def _new_stage_root(
    parent: _ParentPin,
    stages: list[Path],
    *,
    label: str,
) -> Path:
    parent.revalidate(context=f"{label} stage parent before creation")
    root = Path(
        tempfile.mkdtemp(
            prefix=f".dialect-rebuttal-native-{label}-",
            dir=parent.path,
        )
    )
    stages.append(root)
    root.chmod(0o700)
    entry = os.lstat(root)
    parent.revalidate(context=f"{label} stage parent after creation")
    named = os.stat(
        root.name,
        dir_fd=parent.descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != os.geteuid()
        or stat.S_IMODE(entry.st_mode) != 0o700
        or (entry.st_dev, entry.st_ino) != (named.st_dev, named.st_ino)
    ):
        _fail(f"cannot create private {label} stage root")
    return root


def _sdk_version(sdk_root: Path) -> str:
    settings = sdk_root / "SDKSettings.json"
    try:
        raw = settings.read_bytes()
    except OSError as error:
        _fail(f"cannot read anchored SDKSettings.json: {error}")
    parsed = _json_without_duplicates(raw, context="SDKSettings.json")
    if not isinstance(parsed, dict):
        _fail("SDKSettings.json must be one object")
    version = parsed.get("Version")
    canonical = parsed.get("CanonicalName")
    if (
        not isinstance(version, str)
        or re.fullmatch(r"[1-9][0-9]{0,2}\.[0-9]{1,2}", version) is None
        or canonical != f"macosx{version}"
    ):
        _fail("SDKSettings.json has an unsupported exact macOS SDK version")
    return version


def _compile_arguments(
    *,
    root: Path,
    source: Path,
    object_path: Path,
    runtime: _PinnedFile,
    runtime_path: Path,
    renderer_pin: _PinnedFile,
    renderer_path: Path,
    sdk_root: Path,
    resource_root: Path,
) -> list[str]:
    return [
        "-arch",
        "arm64",
        "-target",
        "arm64-apple-macos13.0",
        "--no-default-config",
        "-std=c11",
        "-Os",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-fno-ident",
        "-fno-common",
        "-fvisibility=hidden",
        "-g0",
        "-isysroot",
        str(sdk_root),
        "-resource-dir",
        str(resource_root),
        f"-ffile-prefix-map={root}=/dialect/native-producer",
        f"-fdebug-prefix-map={root}=/dialect/native-producer",
        "-x",
        "c",
        str(source),
        "-c",
        "-o",
        str(object_path),
        _c_string_macro("DIALECT_RUNTIME_PATH", str(runtime_path)),
        _c_string_macro("DIALECT_RUNTIME_SHA256", runtime.sha256),
        f"-DDIALECT_RUNTIME_BYTES={runtime.size}",
        f"-DDIALECT_RUNTIME_MODE=0{runtime.mode:o}",
        _c_string_macro("DIALECT_RENDERER_PATH", str(renderer_path)),
        _c_string_macro("DIALECT_RENDERER_SHA256", renderer_pin.sha256),
        f"-DDIALECT_RENDERER_BYTES={renderer_pin.size}",
        f"-DDIALECT_RENDERER_MODE=0{renderer_pin.mode:o}",
    ]


def _link_arguments(
    *,
    object_path: Path,
    output_path: Path,
    sdk_root: Path,
    sdk_version: str,
) -> list[str]:
    return [
        "-arch",
        "arm64",
        "-syslibroot",
        str(sdk_root),
        "-platform_version",
        "macos",
        MACOS_MINIMUM,
        sdk_version,
        "-lSystem",
        "-dead_strip",
        "-no_adhoc_codesign",
        "-o",
        str(output_path),
        str(object_path),
    ]


def _build_stage(
    root: Path,
    *,
    source_raw: bytes,
    runtime: _PinnedFile,
    runtime_path: Path,
    renderer_pin: _PinnedFile,
    renderer_path: Path,
    clang: _PinnedFile,
    linker: _PinnedFile,
    codesign: _PinnedFile,
    sdk_root: Path,
    sdk_version: str,
    resource_root: Path,
    guard: Callable[[], None],
) -> dict[str, object]:
    source_path = root / AUTHORITY_MEMBER
    object_path = root / PRODUCER_MEMBER
    _write_new_member(
        source_path,
        source_raw,
        mode=0o400,
        context="launcher source stage member",
    )
    compile_arguments = _compile_arguments(
        root=root,
        source=source_path,
        object_path=object_path,
        runtime=runtime,
        runtime_path=runtime_path,
        renderer_pin=renderer_pin,
        renderer_path=renderer_path,
        sdk_root=sdk_root,
        resource_root=resource_root,
    )
    compile_recipe = _normalized_recipe_invocation(
        "compile",
        clang.path,
        compile_arguments,
        bindings={
            "{clang}": str(clang.path),
            "{stage_root}": str(root),
            "{sdk_root}": str(sdk_root),
            "{compiler_resource_root}": str(resource_root),
            "{launcher_source}": str(source_path),
            "{object}": str(object_path),
            "{runtime}": str(runtime_path),
            "{runtime_sha256}": runtime.sha256,
            "{runtime_bytes}": str(runtime.size),
            "{runtime_mode}": f"0{runtime.mode:o}",
            "{renderer}": str(renderer_path),
            "{renderer_sha256}": renderer_pin.sha256,
            "{renderer_bytes}": str(renderer_pin.size),
            "{renderer_mode}": f"0{renderer_pin.mode:o}",
        },
    )
    compile_observation, _, _ = _run_bounded_tool(
        clang,
        compile_arguments,
        context="native launcher compilation",
        before=guard,
        after=guard,
    )
    compile_observation.update(compile_recipe)
    object_raw = _read_stage_member(
        object_path,
        maximum=MAX_EXECUTABLE_BYTES,
        context="native launcher object",
    )
    source_path.chmod(0o600)
    link_arguments = _link_arguments(
        object_path=object_path,
        output_path=source_path,
        sdk_root=sdk_root,
        sdk_version=sdk_version,
    )
    link_recipe = _normalized_recipe_invocation(
        "link",
        linker.path,
        link_arguments,
        bindings={
            "{ld}": str(linker.path),
            "{sdk_root}": str(sdk_root),
            "{sdk_version}": sdk_version,
            "{unsigned_executable}": str(source_path),
            "{object}": str(object_path),
        },
    )
    link_observation, _, _ = _run_bounded_tool(
        linker,
        link_arguments,
        context="native launcher link",
        before=guard,
        after=guard,
    )
    link_observation.update(link_recipe)
    unsigned_raw = _read_stage_member(
        source_path,
        maximum=MAX_EXECUTABLE_BYTES,
        context="unsigned native launcher",
    )
    unsigned_header = _parse_thin_macho_header(
        unsigned_raw,
        context="unsigned native launcher",
    )
    if (
        unsigned_header["architecture"] != "arm64"
        or unsigned_header["cpu_subtype"] != CPU_SUBTYPE_ARM64_ALL
        or _code_signature_command_count(
            unsigned_raw, context="unsigned native launcher"
        )
        != 0
    ):
        _fail("unsigned native launcher is not unsigned generic thin arm64")
    macho_uuid = _required_macho_uuid(unsigned_raw, context="unsigned native launcher")
    sign_arguments = [
        "--force",
        "--sign",
        "-",
        "--options",
        "kill",
        "--timestamp=none",
        "--identifier",
        SIGNATURE_IDENTIFIER,
        "--verbose=0",
        str(source_path),
    ]
    sign_recipe = _normalized_recipe_invocation(
        "sign",
        codesign.path,
        sign_arguments,
        bindings={
            "{codesign}": str(codesign.path),
            "{unsigned_executable}": str(source_path),
        },
    )
    sign_observation, _, _ = _run_bounded_tool(
        codesign,
        sign_arguments,
        context="native launcher ad-hoc signing",
        before=guard,
        after=guard,
    )
    sign_observation.update(sign_recipe)
    verify_arguments = ["--verify", "--strict", "--verbose=0", str(source_path)]
    verify_recipe = _normalized_recipe_invocation(
        "verify",
        codesign.path,
        verify_arguments,
        bindings={
            "{codesign}": str(codesign.path),
            "{signed_executable}": str(source_path),
        },
    )
    verify_observation, _, _ = _run_bounded_tool(
        codesign,
        verify_arguments,
        context="native launcher code-signature verification",
        before=guard,
        after=guard,
    )
    verify_observation.update(verify_recipe)
    signed_raw = _read_stage_member(
        source_path,
        maximum=MAX_EXECUTABLE_BYTES,
        context="signed native launcher",
    )
    if signed_raw == unsigned_raw:
        _fail("ad-hoc signing did not change the native launcher bytes")
    if _required_macho_uuid(signed_raw, context="signed native launcher") != macho_uuid:
        _fail("ad-hoc signing changed the native launcher LC_UUID")
    code_directory = _parse_signed_arm64_launcher(signed_raw)
    _replace_owned_member(
        object_path,
        signed_raw,
        mode=0o500,
        context="final native launcher member",
    )
    terminal = _read_stage_member(
        object_path,
        maximum=MAX_EXECUTABLE_BYTES,
        context="final native launcher member",
    )
    if terminal != signed_raw:
        _fail("final launcher member differs from the signed build")
    return {
        "object_bytes": len(object_raw),
        "object_sha256": _sha256(object_raw),
        "unsigned_bytes": len(unsigned_raw),
        "unsigned_sha256": _sha256(unsigned_raw),
        "signed_bytes": len(signed_raw),
        "signed_sha256": _sha256(signed_raw),
        "macho_uuid": macho_uuid,
        "native_code_directory": code_directory,
        "observations": {
            "compile": compile_observation,
            "link": link_observation,
            "sign": sign_observation,
            "verify": verify_observation,
        },
    }


def _reject_unexpected_binary_private_paths(
    raw: bytes,
    *,
    runtime_path: Path,
    renderer_path: Path,
    stage_roots: Sequence[Path],
) -> None:
    scrubbed = raw.replace(os.fsencode(runtime_path), b"{runtime}").replace(
        os.fsencode(renderer_path), b"{renderer}"
    )
    forbidden = [
        *FORBIDDEN_CAPSULE_FRAGMENTS,
        b".dialect-rebuttal-native-",
        *[os.fsencode(root) for root in stage_roots],
    ]
    if any(fragment in scrubbed for fragment in forbidden):
        _fail("native launcher leaked an unexpected private or stage path")


def _authority_body(
    *,
    mode: str,
    release_id: str,
    source_bundle: _PinnedFile,
    bundle_projection: Mapping[str, object],
    source_raw: bytes,
    config: Mapping[str, object],
    runtime_handoff: Mapping[str, object],
    source_release: Mapping[str, object],
    build_a: Mapping[str, object],
    build_b: Mapping[str, object],
    producer_raw: bytes,
    code_directory: Mapping[str, object],
    toolchain: Mapping[str, object],
    expected_hashes: Mapping[str, str],
) -> dict[str, object]:
    expected_output = _mapping(
        bundle_projection["expected_output"], context="bundle expected output"
    )
    build: dict[str, object] = {
        "target": {
            "architecture": "arm64",
            "platform": "macos",
            "minimum_version": MACOS_MINIMUM,
            "sdk_version": toolchain["sdk_version"],
        },
        "environment": _json_clone(EXACT_TOOL_ENVIRONMENT),
        "inherit_environment": False,
        "cwd": "/",
        "shell": False,
        "independent_build_count": 2,
        "distinct_stage_roots_and_output_inodes": True,
        "recipes": _recipe_records(),
        "builds": [
            {"build_id": "a", **_mapping(_json_clone(build_a), context="build A")},
            {"build_id": "b", **_mapping(_json_clone(build_b), context="build B")},
        ],
        "byte_identity": {
            "object": build_a["object_sha256"] == build_b["object_sha256"],
            "unsigned": build_a["unsigned_sha256"] == build_b["unsigned_sha256"],
            "signed": build_a["signed_sha256"] == build_b["signed_sha256"],
            "native_code_directory": _json_exact(
                build_a["native_code_directory"], build_b["native_code_directory"]
            ),
        },
        "ad_hoc_signature": {
            "identifier": SIGNATURE_IDENTIFIER,
            "options": ["kill"],
            "code_directory_flags": "0x00000202",
            "timestamp": "none",
            "signer_identity_authenticated": False,
        },
    }
    build["build_projection_sha256"] = _sha256(_canonical_json(build))
    body: dict[str, object] = {
        "schema": AUTHORITY_SCHEMA,
        "contract": AUTHORITY_CONTRACT,
        "mode": mode,
        "release_id": release_id,
        "pdf_id": PDF_ID,
        "pdf_member": PDF_MEMBER,
        "status": (
            "producer-candidate-requires-external-anchor"
            if mode == MODE_REVISION
            else "synthetic-canary-only"
        ),
        "authentication": "caller-sha-anchor-only",
        "producer_protocol": PROTOCOL,
        "producer_arguments": _json_clone(PRODUCER_ARGUMENTS),
        "package_contract": _json_clone(PACKAGE_CONTRACT),
        "source_bundle": {
            "member": "rebuttal-source.bundle",
            "mode": "0400",
            "owner": "effective-user-id",
            "link_count": 1,
            "bytes": source_bundle.size,
            "sha256": source_bundle.sha256,
            "bundle_projection": _json_clone(bundle_projection),
        },
        "producer": {
            "member": PRODUCER_MEMBER,
            "mode": "0500",
            "bytes": len(producer_raw),
            "sha256": _sha256(producer_raw),
            "macho_uuid": build_a["macho_uuid"],
            "native_code_directory": _json_clone(code_directory),
        },
        "launcher_source": {
            "member": LAUNCHER_SOURCE_MEMBER,
            "encoding": "base64",
            "bytes": len(source_raw),
            "sha256": _sha256(source_raw),
            "base64": base64.b64encode(source_raw).decode("ascii"),
        },
        "launcher_config": _json_clone(config),
        "build": build,
        "toolchain": _json_clone(toolchain),
        "source_release": _json_clone(source_release),
        "runtime_handoff": _json_clone(runtime_handoff),
        "expected_output": _json_clone(expected_output),
        "caller_anchors": _json_clone(expected_hashes),
        "review_scope": REVIEW_SCOPE,
        "non_inference_limits": _json_clone(NON_INFERENCE_LIMITS),
    }
    body["manifest_body_sha256"] = _sha256(_canonical_json(body))
    return body


def _validate_expected_output(
    value: object,
    *,
    caller_anchors: Mapping[str, object],
) -> Mapping[str, object]:
    output = _mapping(value, context="expected_output")
    _expect_exact_keys(output, {"renderer_manifest", "pdf"}, context="expected_output")
    requirements = (
        (
            "renderer_manifest",
            "render-receipt.json",
            "renderer_manifest_sha256",
            MAX_BUNDLE_BYTES,
        ),
        ("pdf", PDF_MEMBER, "pdf_sha256", MAX_BUNDLE_BYTES),
    )
    for key, member, anchor, maximum in requirements:
        record = _mapping(output[key], context=f"expected_output.{key}")
        _expect_exact_keys(
            record,
            {"member", "bytes", "sha256"},
            context=f"expected_output.{key}",
        )
        if record["member"] != member:
            _fail(f"expected_output.{key} member drifted")
        _expect_positive_int(
            record["bytes"], context=f"expected_output.{key}.bytes", maximum=maximum
        )
        digest = _expect_sha256(
            record["sha256"], context=f"expected_output.{key}.sha256"
        )
        if digest != caller_anchors[anchor]:
            _fail(f"expected_output.{key} differs from its caller anchor")
    return output


def _validate_bundle_projection(
    projection: Mapping[str, object],
    *,
    release_id: str,
    caller_anchors: Mapping[str, object],
) -> Mapping[str, object]:
    _expect_exact_keys(
        projection,
        {
            "schema",
            "contract",
            "release_id",
            "role",
            "producer_protocol",
            "producer_arguments",
            "canonical_inputs",
            "canonical_inputs_projection_sha256",
            "dependencies",
            "expected_output",
            "non_inference",
            "source_or_base64_payload_recorded",
            "bundle_projection_sha256",
        },
        context="bundle_projection",
    )
    projection_sha = _expect_sha256(
        projection["bundle_projection_sha256"], context="bundle_projection_sha256"
    )
    projection_body = dict(projection)
    projection_body.pop("bundle_projection_sha256")
    if _sha256(_canonical_json(projection_body)) != projection_sha:
        _fail("bundle_projection digest is invalid")
    if (
        projection["release_id"] != release_id
        or projection["role"] != PDF_ID
        or projection["producer_protocol"] != PROTOCOL
        or not _json_exact(projection["producer_arguments"], PRODUCER_ARGUMENTS)
        or projection["source_or_base64_payload_recorded"] is not False
    ):
        _fail("bundle_projection fixed bindings drifted")
    if (
        projection["schema"] != renderer.DERIVATION_BUNDLE_SCHEMA
        or projection["contract"] != renderer.DERIVATION_BUNDLE_CONTRACT
        or not _json_exact(
            projection["non_inference"], renderer.DERIVATION_NON_INFERENCE
        )
    ):
        _fail("bundle_projection schema, contract, or non-inference drifted")
    canonical_inputs = _sequence(
        projection["canonical_inputs"], context="canonical input projection"
    )
    expected_input_members = [
        renderer.SOURCE_MEMBER,
        renderer.TEMPLATE_MEMBER,
        renderer.CONFIG_MEMBER,
    ]
    if len(canonical_inputs) != len(expected_input_members):
        _fail("canonical input projection has the wrong member count")
    for index, (expected_member, raw) in enumerate(
        zip(expected_input_members, canonical_inputs, strict=True)
    ):
        item = _mapping(raw, context=f"canonical input projection {index}")
        _expect_exact_keys(
            item,
            {"member", "encoding", "bytes", "sha256", "encoded_payload_sha256"},
            context=f"canonical input projection {index}",
        )
        if item["encoding"] != "base64" or item["member"] != expected_member:
            _fail("canonical input projection member or encoding drifted")
        _expect_positive_int(
            item["bytes"],
            context=f"canonical input projection {index}.bytes",
            maximum=MAX_BUNDLE_BYTES,
        )
        _expect_sha256(item["sha256"], context="canonical input SHA-256")
        _expect_sha256(item["encoded_payload_sha256"], context="encoded input SHA-256")
    if _sha256(_canonical_json(canonical_inputs)) != _expect_sha256(
        projection["canonical_inputs_projection_sha256"],
        context="canonical input projection digest",
    ):
        _fail("canonical input projection digest is invalid")
    dependencies = _mapping(projection["dependencies"], context="bundle dependencies")
    _expect_exact_keys(
        dependencies,
        {"fonts", "machine_runner", "renderer", "reportlab", "runtime", "tools"},
        context="bundle dependencies",
    )
    runtime = _mapping(dependencies["runtime"], context="bundle runtime")
    _expect_exact_keys(
        runtime,
        {"bytes", "locator", "python_tag", "sha256"},
        context="bundle runtime",
    )
    if runtime["python_tag"] != "3.12" or not isinstance(runtime["locator"], str):
        _fail("bundle runtime identity drifted")
    _expect_positive_int(
        runtime["bytes"], context="bundle runtime bytes", maximum=MAX_TOOL_BYTES
    )
    if runtime["sha256"] != caller_anchors["runtime_sha256"]:
        _fail("bundle runtime differs from its caller anchor")
    for key, member, anchor in (
        ("renderer", "analysis/render_tcga_revision_rebuttal.py", "renderer_sha256"),
        ("machine_runner", MACHINE_RUNNER_MEMBER, "machine_runner_sha256"),
    ):
        record = _mapping(dependencies[key], context=f"bundle {key}")
        _expect_exact_keys(
            record,
            {"bytes", "locator", "member", "sha256"},
            context=f"bundle {key}",
        )
        if (
            record["member"] != member
            or not isinstance(record["locator"], str)
            or record["sha256"] != caller_anchors[anchor]
        ):
            _fail(f"bundle {key} identity drifted")
        _expect_positive_int(
            record["bytes"], context=f"bundle {key} bytes", maximum=MAX_BUILDER_BYTES
        )
    fonts = _sequence(dependencies["fonts"], context="bundle fonts")
    expected_fonts = (
        ("regular", "system-arial-unicode", "ArialUnicodeMS"),
        ("bold", "system-arial-bold", "Arial-BoldMT"),
    )
    if len(fonts) != len(expected_fonts):
        _fail("bundle font count drifted")
    for expected, raw in zip(expected_fonts, fonts, strict=True):
        record = _mapping(raw, context="bundle font")
        _expect_exact_keys(
            record,
            {"role", "locator", "postscript_name", "bytes", "sha256"},
            context="bundle font",
        )
        if (record["role"], record["locator"], record["postscript_name"]) != expected:
            _fail("bundle font identity drifted")
        _expect_positive_int(
            record["bytes"],
            context="bundle font bytes",
            maximum=renderer.MAX_FONT_BYTES,
        )
        _expect_sha256(record["sha256"], context="bundle font SHA-256")
    reportlab = _mapping(dependencies["reportlab"], context="bundle reportlab")
    _expect_exact_keys(
        reportlab,
        {
            "locator",
            "file_count",
            "directory_count",
            "entry_count",
            "total_bytes",
            "tree_sha256",
            "bundle_bytes",
            "bundle_sha256",
        },
        context="bundle reportlab",
    )
    if reportlab["locator"] != "invoking-python-reportlab":
        _fail("bundle reportlab locator drifted")
    reportlab_files = _expect_positive_int(
        reportlab["file_count"],
        context="bundle reportlab files",
        maximum=renderer.MAX_REPORTLAB_FILES,
    )
    reportlab_directories = _expect_positive_int(
        reportlab["directory_count"],
        context="bundle reportlab directories",
        maximum=renderer.MAX_REPORTLAB_DIRECTORIES,
    )
    reportlab_entries = _expect_positive_int(
        reportlab["entry_count"],
        context="bundle reportlab entries",
        maximum=renderer.MAX_REPORTLAB_FILES + renderer.MAX_REPORTLAB_DIRECTORIES,
    )
    if reportlab_entries != reportlab_files + reportlab_directories:
        _fail("bundle reportlab entry count drifted")
    for field in ("tree_sha256", "bundle_sha256"):
        _expect_sha256(reportlab[field], context=f"bundle reportlab {field}")
    for field in ("total_bytes", "bundle_bytes"):
        _expect_positive_int(
            reportlab[field],
            context=f"bundle reportlab {field}",
            maximum=renderer.MAX_REPORTLAB_BUNDLE_BYTES,
        )
    tools = _sequence(dependencies["tools"], context="bundle tools")
    expected_tools = (
        ("pdfinfo", "homebrew-pdfinfo"),
        ("pdffonts", "homebrew-pdffonts"),
        ("pdftotext", "homebrew-pdftotext"),
    )
    if len(tools) != len(expected_tools):
        _fail("bundle tool count drifted")
    for expected, raw in zip(expected_tools, tools, strict=True):
        record = _mapping(raw, context="bundle tool")
        _expect_exact_keys(
            record,
            {"name", "locator", "bytes", "sha256"},
            context="bundle tool",
        )
        if (record["name"], record["locator"]) != expected:
            _fail("bundle tool identity drifted")
        _expect_positive_int(
            record["bytes"], context="bundle tool bytes", maximum=MAX_TOOL_BYTES
        )
        _expect_sha256(record["sha256"], context="bundle tool SHA-256")
    _validate_expected_output(
        projection["expected_output"], caller_anchors=caller_anchors
    )
    return dependencies


def _validate_launcher_config(
    config: Mapping[str, object],
    *,
    dependencies: Mapping[str, object],
    caller_anchors: Mapping[str, object],
) -> None:
    _expect_exact_keys(
        config,
        {
            "schema",
            "protocol",
            "role",
            "argument_count_including_argv0",
            "producer_arguments",
            "source_fd",
            "runtime",
            "renderer",
            "cwd",
            "environment",
            "environment_inherited",
            "shell",
            "process_operation",
            "unexpected_inherited_fds",
            "stdout",
            "stderr",
            "stdin",
            "failure_codes",
            "launcher_config_sha256",
        },
        context="launcher_config",
    )
    config_digest = _expect_sha256(
        config["launcher_config_sha256"], context="launcher_config digest"
    )
    config_body = dict(config)
    config_body.pop("launcher_config_sha256")
    if _sha256(_canonical_json(config_body)) != config_digest:
        _fail("launcher_config digest is invalid")
    fixed = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "role": PDF_ID,
        "argument_count_including_argv0": 9,
        "producer_arguments": PRODUCER_ARGUMENTS,
        "source_fd": {
            "canonical_decimal": True,
            "minimum": 3,
            "maximum": 2**31 - 1,
            "access": "O_RDONLY; O_NONBLOCK permitted for regular file",
            "type": "regular",
            "mode": "0400",
            "owner": "effective-user-id",
            "link_count": 1,
            "minimum_bytes": 1,
            "maximum_bytes": renderer.MAX_DERIVATION_BUNDLE_BYTES,
            "seekable_and_rewound": True,
            "cloexec": "cleared-only-after-complete-validation",
        },
        "cwd": "/",
        "environment": EXACT_LAUNCH_ENVIRONMENT,
        "environment_inherited": False,
        "shell": False,
        "process_operation": "execve-only",
        "unexpected_inherited_fds": "enumerate-/dev/fd-and-close-up-to-64",
        "stdout": "inherited-pdf-stream",
        "stderr": "inherited-and-launcher-emits-no-bytes",
        "stdin": "inherited-but-not-read-by-launcher",
        "failure_codes": {
            "64": "argument-protocol-or-fd-token",
            "65": "source-descriptor",
            "66": "cwd",
            "67": "runtime-preflight",
            "68": "renderer-preflight",
            "69": "unexpected-fd-cleanup",
            "126": "execve",
        },
    }
    if any(not _json_exact(config[key], expected) for key, expected in fixed.items()):
        _fail("launcher_config fixed envelope drifted")
    for key, anchor, maximum in (
        ("runtime", "runtime_sha256", MAX_TOOL_BYTES),
        ("renderer", "renderer_sha256", MAX_BUILDER_BYTES),
    ):
        record = _mapping(config[key], context=f"launcher_config {key}")
        _expect_exact_keys(
            record,
            {
                "locator",
                "absolute_path_recorded",
                "absolute_path_utf8_bytes",
                "absolute_path_utf8_sha256",
                "bytes",
                "sha256",
                "mode",
                "owner",
                "link_count",
                "pre_exec_descriptor_hash",
            },
            context=f"launcher_config {key}",
        )
        dependency = _mapping(dependencies[key], context=f"bundle dependency {key}")
        mode = record["mode"]
        if (
            record["locator"] != dependency["locator"]
            or record["absolute_path_recorded"] is not False
            or record["bytes"] != dependency["bytes"]
            or record["sha256"] != dependency["sha256"]
            or record["sha256"] != caller_anchors[anchor]
            or record["owner"] != "effective-user-id"
            or record["pre_exec_descriptor_hash"] is not True
            or not isinstance(mode, str)
            or re.fullmatch(r"[0-7]{4}", mode) is None
            or int(mode, 8) & 0o022
            or (key == "runtime" and not int(mode, 8) & 0o100)
        ):
            _fail(f"launcher_config {key} binding drifted")
        if (
            _expect_positive_int(
                record["link_count"],
                context=f"launcher_config {key} link count",
                maximum=1,
            )
            != 1
        ):
            _fail(f"launcher_config {key} link count drifted")
        _expect_positive_int(
            record["absolute_path_utf8_bytes"],
            context=f"launcher_config {key} path bytes",
            maximum=4096,
        )
        _expect_sha256(
            record["absolute_path_utf8_sha256"],
            context=f"launcher_config {key} path SHA-256",
        )
        _expect_positive_int(
            record["bytes"], context=f"launcher_config {key} bytes", maximum=maximum
        )


def _validate_toolchain(
    toolchain: Mapping[str, object], *, caller_anchors: Mapping[str, object]
) -> None:
    _expect_exact_keys(
        toolchain,
        {
            "clang",
            "linker",
            "codesign",
            "git",
            "compiler_resource_tree",
            "sdk_tree",
            "sdk_version",
            "linker_invocation",
            "codesign_invocation",
            "toolchain_projection_sha256",
        },
        context="toolchain",
    )
    digest = _expect_sha256(
        toolchain["toolchain_projection_sha256"], context="toolchain projection digest"
    )
    body = dict(toolchain)
    body.pop("toolchain_projection_sha256")
    if _sha256(_canonical_json(body)) != digest:
        _fail("toolchain projection digest is invalid")
    if (
        toolchain["linker_invocation"] != "direct-bounded-main-process"
        or toolchain["codesign_invocation"]
        != "bounded-main-path-execution; selected-fat-slice-live-mapping-not-attested"
        or not isinstance(toolchain["sdk_version"], str)
        or re.fullmatch(r"[1-9][0-9]{0,2}\.[0-9]{1,2}", toolchain["sdk_version"])
        is None
    ):
        _fail("toolchain fixed invocation or SDK version drifted")
    for key, locator, anchor in (
        ("clang", "xcode-default-toolchain-clang", "clang_sha256"),
        ("linker", "xcode-default-toolchain-ld", "linker_sha256"),
        ("git", "xcode-git", "git_sha256"),
        ("codesign", "system-codesign", "codesign_sha256"),
    ):
        record = _mapping(toolchain[key], context=f"toolchain {key}")
        _expect_exact_keys(
            record,
            {
                "locator",
                "path_recorded",
                "bytes",
                "sha256",
                "mode",
                "uid",
                "link_count",
                "binary",
            },
            context=f"toolchain {key}",
        )
        mode = record["mode"]
        if (
            record["locator"] != locator
            or record["path_recorded"] is not False
            or record["sha256"] != caller_anchors[anchor]
            or not isinstance(mode, str)
            or re.fullmatch(r"[0-7]{4}", mode) is None
            or not int(mode, 8) & 0o100
            or int(mode, 8) & 0o022
        ):
            _fail(f"toolchain {key} fixed identity drifted")
        uid = _expect_nonnegative_int(
            record["uid"], context=f"toolchain {key} uid", maximum=0xFFFFFFFF
        )
        link_count = _expect_positive_int(
            record["link_count"],
            context=f"toolchain {key} link count",
            maximum=1,
        )
        if uid != 0 or link_count != 1:
            _fail(f"toolchain {key} POSIX identity drifted")
        tool_bytes = _expect_positive_int(
            record["bytes"], context=f"toolchain {key} bytes", maximum=MAX_TOOL_BYTES
        )
        binary = _mapping(record["binary"], context=f"toolchain {key} binary")
        if key == "codesign":
            _expect_exact_keys(
                binary,
                {
                    "binary_container",
                    "fat_endianness",
                    "slice_count",
                    "slices",
                    "selected_execution_slice",
                    "selected_slice_live_mapping",
                },
                context="codesign binary",
            )
            slices = _sequence(binary["slices"], context="codesign slices")
            selected = _mapping(
                binary["selected_execution_slice"], context="codesign selected slice"
            )
            container = binary["binary_container"]
            slice_count = _expect_positive_int(
                binary["slice_count"],
                context="codesign slice count",
                maximum=MAX_FAT_SLICES,
            )
            if (
                container not in {"fat-macho32", "fat-macho64"}
                or binary["fat_endianness"] != "big"
                or slice_count != len(slices)
                or not 1 <= len(slices) <= MAX_FAT_SLICES
                or binary["selected_slice_live_mapping"] != "not-attested"
            ):
                _fail("codesign selected FAT slice identity drifted")
            header_end = 8 + len(slices) * (32 if container == "fat-macho64" else 20)
            extents: list[tuple[int, int]] = []
            architectures: set[str] = set()
            selected_candidates: list[Mapping[str, object]] = []
            for expected_index, slice_raw in enumerate(slices):
                slice_record = _mapping(
                    slice_raw,
                    context=f"codesign slice {expected_index}",
                )
                _expect_exact_keys(
                    slice_record,
                    {
                        "index",
                        "architecture",
                        "cpu_type",
                        "cpu_subtype",
                        "cpu_subtype_capabilities",
                        "alignment_exponent",
                        "offset",
                        "bytes",
                        "sha256",
                    },
                    context=f"codesign slice {expected_index}",
                )
                cpu_type = _expect_positive_int(
                    slice_record["cpu_type"],
                    context=f"codesign slice {expected_index} CPU type",
                    maximum=0x7FFFFFFF,
                )
                index = _expect_nonnegative_int(
                    slice_record["index"],
                    context=f"codesign slice {expected_index} index",
                    maximum=MAX_FAT_SLICES - 1,
                )
                cpu_subtype = _expect_nonnegative_int(
                    slice_record["cpu_subtype"],
                    context=f"codesign slice {expected_index} CPU subtype",
                    maximum=0x00FFFFFF,
                )
                subtype_capabilities = _expect_nonnegative_int(
                    slice_record["cpu_subtype_capabilities"],
                    context=(
                        f"codesign slice {expected_index} CPU subtype capabilities"
                    ),
                    maximum=0xFF000000,
                )
                alignment_exponent = _expect_nonnegative_int(
                    slice_record["alignment_exponent"],
                    context=f"codesign slice {expected_index} alignment exponent",
                    maximum=31,
                )
                offset = _expect_positive_int(
                    slice_record["offset"],
                    context=f"codesign slice {expected_index} offset",
                    maximum=tool_bytes,
                )
                size = _expect_positive_int(
                    slice_record["bytes"],
                    context=f"codesign slice {expected_index} bytes",
                    maximum=tool_bytes,
                )
                architecture = slice_record["architecture"]
                if (
                    index != expected_index
                    or architecture not in {"arm64", "x86_64"}
                    or architecture in architectures
                    or (architecture == "arm64" and cpu_type != CPU_TYPE_ARM64)
                    or (architecture == "x86_64" and cpu_type != CPU_TYPE_X86_64)
                    or subtype_capabilities & 0x00FFFFFF
                    or offset < header_end
                    or offset % (1 << alignment_exponent)
                    or offset + size > tool_bytes
                    or any(
                        not (offset + size <= left or offset >= right)
                        for left, right in extents
                    )
                ):
                    _fail(f"codesign slice {expected_index} extent or identity drifted")
                _expect_sha256(
                    slice_record["sha256"],
                    context=f"codesign slice {expected_index} SHA-256",
                )
                architectures.add(str(architecture))
                extents.append((offset, offset + size))
                if cpu_type == CPU_TYPE_ARM64 and cpu_subtype == CPU_SUBTYPE_ARM64E:
                    if subtype_capabilities != 0x80000000:
                        _fail("codesign arm64e slice capability bits drifted")
                    selected_candidates.append(slice_record)
            if len(selected_candidates) != 1 or not _json_exact(
                selected, selected_candidates[0]
            ):
                _fail("codesign selected FAT slice identity drifted")
        else:
            _expect_exact_keys(
                binary,
                {
                    "binary_container",
                    "architecture",
                    "cpu_type",
                    "cpu_subtype",
                    "cpu_subtype_capabilities",
                    "file_type",
                    "load_command_count",
                    "load_command_bytes",
                },
                context=f"toolchain {key} binary",
            )
            cpu_type = _expect_positive_int(
                binary["cpu_type"],
                context=f"toolchain {key} CPU type",
                maximum=0x7FFFFFFF,
            )
            cpu_subtype = _expect_nonnegative_int(
                binary["cpu_subtype"],
                context=f"toolchain {key} CPU subtype",
                maximum=0x00FFFFFF,
            )
            subtype_capabilities = _expect_nonnegative_int(
                binary["cpu_subtype_capabilities"],
                context=f"toolchain {key} CPU subtype capabilities",
                maximum=0xFF000000,
            )
            load_command_count = _expect_positive_int(
                binary["load_command_count"],
                context=f"toolchain {key} load command count",
                maximum=MAX_MACH_LOAD_COMMANDS,
            )
            load_command_bytes = _expect_positive_int(
                binary["load_command_bytes"],
                context=f"toolchain {key} load command bytes",
                maximum=tool_bytes,
            )
            if (
                binary["binary_container"] != "thin-macho64"
                or binary["architecture"] != "arm64"
                or cpu_type != CPU_TYPE_ARM64
                or cpu_subtype != CPU_SUBTYPE_ARM64_ALL
                or subtype_capabilities != 0
                or binary["file_type"] != "execute"
                or load_command_bytes < load_command_count * 8
            ):
                _fail(f"toolchain {key} binary identity drifted")
    for key, locator, root, anchor in (
        (
            "compiler_resource_tree",
            "xcode-clang-resource-root",
            EXPECTED_COMPILER_RESOURCE_ROOT,
            "compiler_resource_tree_sha256",
        ),
        ("sdk_tree", "xcode-macos-sdk-root", EXPECTED_SDK_ROOT, "sdk_tree_sha256"),
    ):
        record = _mapping(toolchain[key], context=f"toolchain {key}")
        _expect_exact_keys(
            record,
            {
                "locator",
                "root_path_recorded",
                "root_path_utf8_bytes",
                "root_path_utf8_sha256",
                "tree_hash_contract",
                "tree_sha256",
                "file_count",
                "directory_count",
                "symlink_count",
                "entry_count",
                "total_file_bytes",
            },
            context=f"toolchain {key}",
        )
        root_path_bytes = _expect_positive_int(
            record["root_path_utf8_bytes"],
            context=f"toolchain {key} root path bytes",
            maximum=4096,
        )
        if (
            record["locator"] != locator
            or record["root_path_recorded"] is not False
            or root_path_bytes != len(os.fsencode(root))
            or record["root_path_utf8_sha256"] != _sha256(os.fsencode(root))
            or record["tree_hash_contract"] != TREE_HASH_CONTRACT
            or record["tree_sha256"] != caller_anchors[anchor]
        ):
            _fail(f"toolchain {key} tree binding drifted")
        files = _expect_positive_int(
            record["file_count"],
            context=f"toolchain {key} files",
            maximum=MAX_TREE_FILES,
        )
        directories = _expect_positive_int(
            record["directory_count"],
            context=f"toolchain {key} directories",
            maximum=MAX_TREE_DIRECTORIES,
        )
        symlinks = _expect_nonnegative_int(
            record["symlink_count"],
            context=f"toolchain {key} symlinks",
            maximum=MAX_TREE_SYMLINKS,
        )
        entries = _expect_positive_int(
            record["entry_count"],
            context=f"toolchain {key} entries",
            maximum=MAX_TREE_ENTRIES,
        )
        if entries != files + directories + symlinks:
            _fail(f"toolchain {key} entry count is inconsistent")
        _expect_positive_int(
            record["total_file_bytes"],
            context=f"toolchain {key} total bytes",
            maximum=MAX_TREE_BYTES,
        )


def _validate_source_release(
    source_release: Mapping[str, object],
    *,
    mode: str,
    source_sha256: str,
    caller_anchors: Mapping[str, object],
) -> None:
    _expect_exact_keys(
        source_release,
        {
            "status",
            "release_commit",
            "release_ref",
            "git_blob_equality",
            "members",
            "git",
            "source_release_projection_sha256",
        },
        context="source_release",
    )
    digest = _expect_sha256(
        source_release["source_release_projection_sha256"],
        context="source_release projection digest",
    )
    body = dict(source_release)
    body.pop("source_release_projection_sha256")
    if _sha256(_canonical_json(body)) != digest:
        _fail("source_release projection digest is invalid")
    git = _mapping(source_release["git"], context="source_release Git")
    _expect_exact_keys(
        git,
        {"locator", "bytes", "sha256", "main-executable-bytes-pinned"},
        context="source_release Git",
    )
    if (
        git["locator"] != "xcode-git"
        or git["sha256"] != caller_anchors["git_sha256"]
        or git["main-executable-bytes-pinned"] is not True
    ):
        _fail("source_release Git binding drifted")
    _expect_positive_int(
        git["bytes"], context="source_release Git bytes", maximum=MAX_TOOL_BYTES
    )
    members = _sequence(source_release["members"], context="source_release members")
    member_anchors = {
        LAUNCHER_SOURCE_MEMBER: "launcher_source_sha256",
        BUILDER_MEMBER: "builder_sha256",
        BUNDLE_BUILDER_MEMBER: "bundle_builder_sha256",
        "analysis/render_tcga_revision_rebuttal.py": "renderer_sha256",
        MACHINE_RUNNER_MEMBER: "machine_runner_sha256",
    }
    if mode == MODE_SYNTHETIC:
        if (
            source_release["status"] != SYNTHETIC_SOURCE_STATUS
            or source_release["release_commit"] is not None
            or source_release["release_ref"] is not None
            or source_release["git_blob_equality"] is not False
            or len(members) != len(RELEVANT_RELEASE_MEMBERS)
        ):
            _fail("synthetic source_release semantics drifted")
        for expected_member, raw in zip(RELEVANT_RELEASE_MEMBERS, members, strict=True):
            record = _mapping(raw, context=f"source_release member {expected_member}")
            _expect_exact_keys(
                record,
                {"member", "bytes", "sha256"},
                context="source_release member",
            )
            if (
                record["member"] != expected_member
                or record["sha256"] != caller_anchors[member_anchors[expected_member]]
            ):
                _fail(f"source_release member {expected_member} binding drifted")
            _expect_positive_int(
                record["bytes"],
                context="source_release member bytes",
                maximum=MAX_GIT_BLOB_BYTES,
            )
        if source_sha256 != members[0]["sha256"]:
            _fail("synthetic launcher source differs from its listed source record")
        return
    commit = source_release["release_commit"]
    release_ref = source_release["release_ref"]
    if (
        source_release["status"]
        != "git-command-observed-listed-path-byte-equality-at-caller-commit"
        or source_release["git_blob_equality"] is not True
        or not isinstance(commit, str)
        or COMMIT_RE.fullmatch(commit) is None
        or not isinstance(release_ref, str)
        or SAFE_REF_RE.fullmatch(release_ref) is None
        or release_ref.startswith("-")
        or ".." in release_ref
        or "//" in release_ref
        or len(members) != len(RELEVANT_RELEASE_MEMBERS)
    ):
        _fail("revision source_release semantics drifted")
    for expected_member, raw in zip(RELEVANT_RELEASE_MEMBERS, members, strict=True):
        record = _mapping(raw, context=f"source_release member {expected_member}")
        _expect_exact_keys(
            record, {"member", "bytes", "sha256"}, context="source_release member"
        )
        if (
            record["member"] != expected_member
            or record["sha256"] != caller_anchors[member_anchors[expected_member]]
        ):
            _fail(f"source_release member {expected_member} binding drifted")
        _expect_positive_int(
            record["bytes"],
            context="source_release member bytes",
            maximum=MAX_GIT_BLOB_BYTES,
        )
    if source_sha256 != members[0]["sha256"]:
        _fail("launcher source differs from its release Git blob")


def _validate_build(
    build: Mapping[str, object],
    *,
    producer: Mapping[str, object],
    native_cd: Mapping[str, object],
    toolchain: Mapping[str, object],
) -> None:
    _expect_exact_keys(
        build,
        {
            "target",
            "environment",
            "inherit_environment",
            "cwd",
            "shell",
            "independent_build_count",
            "distinct_stage_roots_and_output_inodes",
            "recipes",
            "builds",
            "byte_identity",
            "ad_hoc_signature",
            "build_projection_sha256",
        },
        context="build",
    )
    digest = _expect_sha256(build["build_projection_sha256"], context="build digest")
    body = dict(build)
    body.pop("build_projection_sha256")
    if _sha256(_canonical_json(body)) != digest:
        _fail("build projection digest is invalid")
    target = {
        "architecture": "arm64",
        "platform": "macos",
        "minimum_version": MACOS_MINIMUM,
        "sdk_version": toolchain["sdk_version"],
    }
    independent_build_count = _expect_positive_int(
        build["independent_build_count"],
        context="independent build count",
        maximum=2,
    )
    if (
        not _json_exact(build["target"], target)
        or not _json_exact(build["environment"], EXACT_TOOL_ENVIRONMENT)
        or build["inherit_environment"] is not False
        or build["cwd"] != "/"
        or build["shell"] is not False
        or independent_build_count != 2
        or build["distinct_stage_roots_and_output_inodes"] is not True
        or not _json_exact(build["recipes"], _recipe_records())
        or not _json_exact(
            build["byte_identity"],
            {
                "object": True,
                "unsigned": True,
                "signed": True,
                "native_code_directory": True,
            },
        )
        or not _json_exact(
            build["ad_hoc_signature"],
            {
                "identifier": SIGNATURE_IDENTIFIER,
                "options": ["kill"],
                "code_directory_flags": "0x00000202",
                "timestamp": "none",
                "signer_identity_authenticated": False,
            },
        )
    ):
        _fail("build fixed recipe or execution envelope drifted")
    builds = _sequence(build["builds"], context="build records")
    if len(builds) != 2:
        _fail("authority must contain exactly two build records")
    normalized: list[dict[str, object]] = []
    empty_sha = _sha256(b"")
    for expected_id, raw in zip(("a", "b"), builds, strict=True):
        record = _mapping(raw, context=f"build {expected_id}")
        _expect_exact_keys(
            record,
            {
                "build_id",
                "object_bytes",
                "object_sha256",
                "unsigned_bytes",
                "unsigned_sha256",
                "signed_bytes",
                "signed_sha256",
                "macho_uuid",
                "native_code_directory",
                "observations",
            },
            context=f"build {expected_id}",
        )
        if (
            record["build_id"] != expected_id
            or record["signed_sha256"] != producer["sha256"]
            or record["signed_bytes"] != producer["bytes"]
            or record["macho_uuid"] != producer["macho_uuid"]
            or not _json_exact(record["native_code_directory"], native_cd)
        ):
            _fail(f"build {expected_id} differs from the final producer")
        for field in ("object", "unsigned", "signed"):
            _expect_positive_int(
                record[f"{field}_bytes"],
                context=f"build {expected_id} {field} bytes",
                maximum=MAX_EXECUTABLE_BYTES,
            )
            _expect_sha256(
                record[f"{field}_sha256"],
                context=f"build {expected_id} {field} SHA-256",
            )
        observations = _mapping(record["observations"], context="build observations")
        _expect_exact_keys(
            observations,
            {"compile", "link", "sign", "verify"},
            context="build observations",
        )
        for operation, observed_raw in observations.items():
            observed = _mapping(observed_raw, context=f"build observation {operation}")
            _expect_exact_keys(
                observed,
                {
                    "return_code",
                    "stdout_bytes",
                    "stdout_sha256",
                    "stderr_bytes",
                    "stderr_sha256",
                    "normalized_argv",
                    "normalized_argv_sha256",
                },
                context=f"build observation {operation}",
            )
            recipe = _mapping(
                _recipe_records()[operation], context=f"build recipe {operation}"
            )
            _expect_nonnegative_int(
                observed["return_code"],
                context=f"build observation {operation} return code",
                maximum=255,
            )
            for stream in ("stdout", "stderr"):
                _expect_nonnegative_int(
                    observed[f"{stream}_bytes"],
                    context=f"build observation {operation} {stream} bytes",
                    maximum=MAX_TOOL_OUTPUT_BYTES,
                )
            expected_observation = {
                "return_code": 0,
                "stdout_bytes": 0,
                "stdout_sha256": empty_sha,
                "stderr_bytes": 0,
                "stderr_sha256": empty_sha,
                "normalized_argv": recipe["argv"],
                "normalized_argv_sha256": recipe["argv_sha256"],
            }
            if not _json_exact(observed, expected_observation):
                _fail(f"build observation {operation} is not exact empty success")
        normalized.append(dict(record))
    a = dict(normalized[0])
    b = dict(normalized[1])
    a.pop("build_id")
    b.pop("build_id")
    if not _json_exact(a, b):
        _fail("independent build records are not exactly byte-identical")


def _normalize_authority(value: Mapping[str, object]) -> dict[str, object]:
    """Validate an exact canonical v2 capsule without trusting its producer."""
    expected_keys = {
        "schema",
        "contract",
        "mode",
        "release_id",
        "pdf_id",
        "pdf_member",
        "status",
        "authentication",
        "producer_protocol",
        "producer_arguments",
        "package_contract",
        "source_bundle",
        "producer",
        "launcher_source",
        "launcher_config",
        "build",
        "toolchain",
        "source_release",
        "runtime_handoff",
        "expected_output",
        "caller_anchors",
        "review_scope",
        "non_inference_limits",
        "manifest_body_sha256",
    }
    _expect_exact_keys(value, expected_keys, context="native producer authority")
    if value["schema"] != AUTHORITY_SCHEMA or value["contract"] != AUTHORITY_CONTRACT:
        _fail("native producer authority schema or contract drifted")
    mode = value["mode"]
    if mode not in MODES:
        _fail("native producer authority mode is invalid")
    release_id = _expect_token(value["release_id"], context="authority release_id")
    expected_status = (
        "producer-candidate-requires-external-anchor"
        if mode == MODE_REVISION
        else "synthetic-canary-only"
    )
    if (
        value["pdf_id"] != PDF_ID
        or value["pdf_member"] != PDF_MEMBER
        or value["status"] != expected_status
        or value["authentication"] != "caller-sha-anchor-only"
        or value["producer_protocol"] != PROTOCOL
        or not _json_exact(value["producer_arguments"], PRODUCER_ARGUMENTS)
        or not _json_exact(value["package_contract"], PACKAGE_CONTRACT)
        or value["review_scope"] != REVIEW_SCOPE
        or not _json_exact(value["non_inference_limits"], NON_INFERENCE_LIMITS)
    ):
        _fail("native producer authority fixed envelope drifted")
    caller_anchors = _mapping(value["caller_anchors"], context="caller_anchors")
    _expect_exact_keys(caller_anchors, CALLER_ANCHOR_KEYS, context="caller_anchors")
    for key, digest in caller_anchors.items():
        _expect_sha256(digest, context=f"caller anchor {key}")
    source_bundle = _mapping(value["source_bundle"], context="source_bundle")
    _expect_exact_keys(
        source_bundle,
        {
            "member",
            "mode",
            "owner",
            "link_count",
            "bytes",
            "sha256",
            "bundle_projection",
        },
        context="source_bundle",
    )
    if (
        source_bundle["member"] != "rebuttal-source.bundle"
        or source_bundle["mode"] != "0400"
        or source_bundle["owner"] != "effective-user-id"
    ):
        _fail("source_bundle descriptor contract drifted")
    if (
        _expect_positive_int(
            source_bundle["link_count"],
            context="source_bundle.link_count",
            maximum=1,
        )
        != 1
    ):
        _fail("source_bundle link count drifted")
    _expect_positive_int(
        source_bundle["bytes"], context="source_bundle.bytes", maximum=MAX_BUNDLE_BYTES
    )
    if (
        _expect_sha256(source_bundle["sha256"], context="source_bundle.sha256")
        != caller_anchors["source_bundle_sha256"]
    ):
        _fail("source_bundle differs from its caller anchor")
    bundle_projection = _mapping(
        source_bundle["bundle_projection"], context="bundle_projection"
    )
    dependencies = _validate_bundle_projection(
        bundle_projection,
        release_id=release_id,
        caller_anchors=caller_anchors,
    )
    producer = _mapping(value["producer"], context="producer")
    _expect_exact_keys(
        producer,
        {
            "member",
            "mode",
            "bytes",
            "sha256",
            "macho_uuid",
            "native_code_directory",
        },
        context="producer",
    )
    if producer["member"] != PRODUCER_MEMBER or producer["mode"] != "0500":
        _fail("producer member or mode drifted")
    _expect_positive_int(
        producer["bytes"], context="producer.bytes", maximum=MAX_EXECUTABLE_BYTES
    )
    _expect_sha256(producer["sha256"], context="producer.sha256")
    if (
        not isinstance(producer["macho_uuid"], str)
        or UUID_RE.fullmatch(producer["macho_uuid"]) is None
    ):
        _fail("producer Mach-O UUID is invalid")
    native_cd = _mapping(
        producer["native_code_directory"], context="producer native CodeDirectory"
    )
    _expect_exact_keys(
        native_cd,
        {
            "binary_container",
            "architecture",
            "cpu_subtype",
            "hash_type",
            "code_directory_flags",
            "code_directory_bytes",
            "cdhash",
            "code_limit",
            "code_slots",
            "page_size",
            "signature_offset",
            "signature_bytes",
        },
        context="producer native CodeDirectory",
    )
    code_directory_flags = _expect_nonnegative_int(
        native_cd["code_directory_flags"],
        context="producer CodeDirectory flags",
        maximum=0xFFFFFFFF,
    )
    if (
        native_cd["binary_container"] != "thin-macho64"
        or native_cd["architecture"] != "arm64"
        or native_cd["cpu_subtype"] != "all"
        or native_cd["hash_type"] != "sha256"
        or code_directory_flags != 0x202
        or not isinstance(native_cd["cdhash"], str)
        or re.fullmatch(r"[0-9a-f]{40}", str(native_cd["cdhash"])) is None
    ):
        _fail("producer native CodeDirectory identity drifted")
    code_limit = _expect_positive_int(
        native_cd["code_limit"],
        context="producer CodeDirectory code limit",
        maximum=MAX_EXECUTABLE_BYTES,
    )
    signature_offset = _expect_positive_int(
        native_cd["signature_offset"],
        context="producer CodeDirectory signature offset",
        maximum=MAX_EXECUTABLE_BYTES,
    )
    page_size = _expect_positive_int(
        native_cd["page_size"],
        context="producer CodeDirectory page size",
        maximum=MAX_EXECUTABLE_BYTES,
    )
    slots = _expect_positive_int(
        native_cd["code_slots"],
        context="producer CodeDirectory code slots",
        maximum=MAX_EXECUTABLE_BYTES,
    )
    _expect_positive_int(
        native_cd["code_directory_bytes"],
        context="producer CodeDirectory bytes",
        maximum=MAX_EXECUTABLE_BYTES,
    )
    signature_bytes = _expect_positive_int(
        native_cd["signature_bytes"],
        context="producer signature bytes",
        maximum=MAX_EXECUTABLE_BYTES,
    )
    if (
        code_limit != signature_offset
        or page_size != 16_384
        or slots != (code_limit + page_size - 1) // page_size
        or signature_offset + signature_bytes != producer["bytes"]
    ):
        _fail("producer CodeDirectory extents drifted")
    source = _mapping(value["launcher_source"], context="launcher_source")
    _expect_exact_keys(
        source,
        {"member", "encoding", "bytes", "sha256", "base64"},
        context="launcher_source",
    )
    if source["member"] != LAUNCHER_SOURCE_MEMBER or source["encoding"] != "base64":
        _fail("launcher_source member or encoding drifted")
    encoded_source = source["base64"]
    if not isinstance(encoded_source, str):
        _fail("launcher_source base64 must be one string")
    try:
        source_raw = base64.b64decode(encoded_source, validate=True)
    except (ValueError, TypeError) as error:
        _fail(f"launcher_source base64 is invalid: {error}")
    if len(source_raw) != _expect_positive_int(
        source["bytes"], context="launcher_source.bytes", maximum=MAX_SOURCE_BYTES
    ) or _sha256(source_raw) != _expect_sha256(
        source["sha256"], context="launcher_source.sha256"
    ):
        _fail("launcher_source decoded bytes differ from their record")
    if source["sha256"] != caller_anchors["launcher_source_sha256"]:
        _fail("launcher_source differs from its caller anchor")
    config = _mapping(value["launcher_config"], context="launcher_config")
    _validate_launcher_config(
        config,
        dependencies=dependencies,
        caller_anchors=caller_anchors,
    )
    toolchain = _mapping(value["toolchain"], context="toolchain")
    _validate_toolchain(toolchain, caller_anchors=caller_anchors)
    build = _mapping(value["build"], context="build")
    _validate_build(
        build,
        producer=producer,
        native_cd=native_cd,
        toolchain=toolchain,
    )
    source_release = _mapping(value["source_release"], context="source_release")
    _validate_source_release(
        source_release,
        mode=str(mode),
        source_sha256=str(source["sha256"]),
        caller_anchors=caller_anchors,
    )
    handoff = _mapping(value["runtime_handoff"], context="runtime_handoff")
    if not _json_exact(handoff, _runtime_handoff(config)):
        _fail("runtime_handoff differs from the exact launcher config")
    expected_output = _validate_expected_output(
        value["expected_output"], caller_anchors=caller_anchors
    )
    if not _json_exact(expected_output, bundle_projection["expected_output"]):
        _fail("top-level expected_output differs from the bundle projection")
    manifest_digest = _expect_sha256(
        value["manifest_body_sha256"], context="manifest_body_sha256"
    )
    body = dict(value)
    body.pop("manifest_body_sha256")
    if _sha256(_canonical_json(body)) != manifest_digest:
        _fail("manifest_body_sha256 does not hash the exact authority body")
    return dict(value)


def _authority_raw(value: Mapping[str, object]) -> bytes:
    normalized = _normalize_authority(value)
    raw = _canonical_json_lf(normalized)
    if len(raw) > MAX_AUTHORITY_BYTES:
        _fail("native producer authority exceeds its byte bound")
    if any(fragment in raw for fragment in FORBIDDEN_CAPSULE_FRAGMENTS):
        _fail("native producer authority contains a host-private path fragment")
    return raw


def _parse_authority_raw(raw: bytes) -> dict[str, object]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("native producer authority must end in exactly one LF")
    parsed = _json_without_duplicates(raw[:-1], context="native producer authority")
    if not isinstance(parsed, dict):
        _fail("native producer authority must be one JSON object")
    normalized = _normalize_authority(parsed)
    if _canonical_json_lf(normalized) != raw:
        _fail("native producer authority is not exact canonical JSON plus one LF")
    return normalized


def _seal_stage(root: Path, authority_raw: bytes) -> None:
    authority_path = root / AUTHORITY_MEMBER
    _replace_owned_member(
        authority_path,
        authority_raw,
        mode=0o400,
        context="native producer authority member",
    )
    producer_path = root / PRODUCER_MEMBER
    producer_path.chmod(0o500)
    root.chmod(0o500)
    root_descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        for member in PACKAGE_MEMBERS:
            descriptor = os.open(
                member,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=root_descriptor,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(root_descriptor)
    finally:
        os.close(root_descriptor)


def _directory_names(descriptor: int, *, context: str) -> list[str]:
    names: list[str] = []
    with os.scandir(descriptor) as iterator:
        for entry in iterator:
            names.append(entry.name)
            if len(names) > len(PACKAGE_MEMBERS):
                _fail(f"{context} contains extra members")
    return sorted(names)


def _pin_package(path: Path, *, context: str) -> _PackagePin:
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if (
        resolved != absolute
        or not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or entry.st_uid != os.geteuid()
        or stat.S_IMODE(entry.st_mode) != 0o500
    ):
        _fail(f"{context} root is not one canonical sealed directory")
    root_descriptor = os.open(
        absolute,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    member_descriptors: dict[str, int] = {}
    try:
        opened_root = os.fstat(root_descriptor)
        entry_identity = (
            entry.st_dev,
            entry.st_ino,
            entry.st_mtime_ns,
            stat.S_IMODE(entry.st_mode),
            entry.st_uid,
        )
        opened_identity = (
            opened_root.st_dev,
            opened_root.st_ino,
            opened_root.st_mtime_ns,
            stat.S_IMODE(opened_root.st_mode),
            opened_root.st_uid,
        )
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or stat.S_ISLNK(opened_root.st_mode)
            or opened_root.st_uid != os.geteuid()
            or stat.S_IMODE(opened_root.st_mode) != 0o500
            or opened_identity != entry_identity
        ):
            _fail(f"{context} root changed while opened")
        if _directory_names(root_descriptor, context=context) != sorted(
            PACKAGE_MEMBERS
        ):
            _fail(f"{context} has the wrong exact two-member inventory")
        member_bytes: dict[str, bytes] = {}
        member_identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        for member in PACKAGE_MEMBERS:
            descriptor = os.open(
                member,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=root_descriptor,
            )
            member_descriptors[member] = descriptor
            observed = os.fstat(descriptor)
            expected_mode = 0o500 if member == PRODUCER_MEMBER else 0o400
            maximum = (
                MAX_EXECUTABLE_BYTES
                if member == PRODUCER_MEMBER
                else MAX_AUTHORITY_BYTES
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != expected_mode
                or not 1 <= observed.st_size <= maximum
            ):
                _fail(f"{context} member {member} is not sealed exactly")
            named = os.stat(member, dir_fd=root_descriptor, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (observed.st_dev, observed.st_ino):
                _fail(f"{context} member {member} changed while opened")
            member_bytes[member] = _read_fd(
                descriptor, maximum=maximum, context=f"{context} {member}"
            )
            member_identities[member] = (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                stat.S_IMODE(observed.st_mode),
                observed.st_nlink,
                observed.st_uid,
            )
        pin = _PackagePin(
            path=absolute,
            root_descriptor=root_descriptor,
            root_identity=opened_identity,
            member_descriptors=member_descriptors,
            member_identities=member_identities,
            member_bytes=member_bytes,
        )
        pin.revalidate(context=context)
        result = pin
    except BaseException:
        for descriptor in member_descriptors.values():
            os.close(descriptor)
        os.close(root_descriptor)
        raise
    return result


def _revalidate_package_pin(pin: _PackagePin, *, context: str) -> None:
    opened_root = os.fstat(pin.root_descriptor)
    named_root = os.lstat(pin.path)
    identity = pin.root_identity
    if (
        opened_root.st_dev,
        opened_root.st_ino,
        opened_root.st_mtime_ns,
        stat.S_IMODE(opened_root.st_mode),
        opened_root.st_uid,
    ) != identity or (
        named_root.st_dev,
        named_root.st_ino,
        named_root.st_mtime_ns,
        stat.S_IMODE(named_root.st_mode),
        named_root.st_uid,
    ) != identity:
        _fail(f"{context} root identity changed")
    if _directory_names(pin.root_descriptor, context=context) != sorted(
        PACKAGE_MEMBERS
    ):
        _fail(f"{context} inventory changed")
    for member in PACKAGE_MEMBERS:
        descriptor = pin.member_descriptors[member]
        observed = os.fstat(descriptor)
        named = os.stat(member, dir_fd=pin.root_descriptor, follow_symlinks=False)
        current = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            stat.S_IMODE(observed.st_mode),
            observed.st_nlink,
            observed.st_uid,
        )
        if (
            current != pin.member_identities[member]
            or (
                named.st_dev,
                named.st_ino,
                named.st_size,
                named.st_mtime_ns,
                stat.S_IMODE(named.st_mode),
                named.st_nlink,
                named.st_uid,
            )
            != current
        ):
            _fail(f"{context} member {member} identity changed")
        maximum = (
            MAX_EXECUTABLE_BYTES if member == PRODUCER_MEMBER else MAX_AUTHORITY_BYTES
        )
        if (
            _read_fd(descriptor, maximum=maximum, context=f"{context} {member}")
            != pin.member_bytes[member]
        ):
            _fail(f"{context} member {member} bytes changed")


def _stable_package_location(
    parent: _ParentPin,
    pin: _PackagePin,
    name: str,
    *,
    context: str,
) -> Path | None:
    try:
        parent.revalidate(context=context)
        named = os.stat(
            name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        opened = os.fstat(pin.root_descriptor)
    except (OSError, RebuttalNativeProducerError):
        return None
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        return None
    return parent.path / name


def _publish_candidate(
    candidate: Path,
    destination: Path,
    parent: _ParentPin,
) -> _PackagePin:
    if candidate.parent != destination.parent or destination.parent != parent.path:
        _fail("candidate and destination must share one atomic parent")
    parent.revalidate(context="publication parent before candidate pin")
    parent.require_absent(destination.name, context="publication destination")
    candidate_pin = _pin_package(candidate, context="candidate package")
    renamed = False
    try:
        parent.revalidate(context="publication parent before rename")
        parent.require_absent(destination.name, context="publication destination")
        os.fsync(parent.descriptor)
        parent.revalidate(context="publication parent immediately before rename")
        renderer._rename_no_replace(
            parent.descriptor,
            candidate.name,
            destination.name,
        )
        renamed = True
        os.fsync(parent.descriptor)
        parent.revalidate(context="publication parent after rename")
        candidate_pin.path = destination
        candidate_pin.revalidate(context="published package")
        result = candidate_pin
    except BaseException as error:
        name = destination.name if renamed else candidate.name
        location = _stable_package_location(
            parent,
            candidate_pin,
            name,
            context="publication failure parent mapping",
        )
        cleanup_error: BaseException | None = None
        try:
            candidate_pin.close()
        except BaseException as close_error:  # noqa: BLE001 - retain primary error.
            cleanup_error = close_error
        location_label = (
            str(location) if location is not None else "ambiguous-via-held-parent-fd"
        )
        message = (
            f"{error}; materialized_path={location_label}; "
            "publication_state=preserved-after-failure; do-not-auto-delete"
        )
        if cleanup_error is not None:
            message = f"{message}; publication pin cleanup failed: {cleanup_error}"
        raise _PublicationError(
            message,
            location=location,
            renamed=renamed,
        ) from error
    return result


def _materialize_candidates(
    source_bundle_path: Path,
    parent: _ParentPin,
    *,
    runtime_path: Path,
    renderer_path: Path,
    machine_runner_path: Path,
    clang_path: Path,
    linker_path: Path,
    codesign_path: Path,
    git_path: Path,
    compiler_resource_root: Path,
    sdk_root: Path,
    release_id: str,
    mode: str,
    release_commit: str | None,
    release_ref: str | None,
    expected_source_bundle_sha256: str,
    expected_launcher_source_sha256: str,
    expected_builder_sha256: str,
    expected_bundle_builder_sha256: str,
    expected_runtime_sha256: str,
    expected_renderer_sha256: str,
    expected_machine_runner_sha256: str,
    expected_clang_sha256: str,
    expected_linker_sha256: str,
    expected_codesign_sha256: str,
    expected_git_sha256: str,
    expected_compiler_resource_tree_sha256: str,
    expected_sdk_tree_sha256: str,
    expected_renderer_manifest_sha256: str,
    expected_pdf_sha256: str,
) -> _MaterializedCandidates:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        _fail("native rebuttal producer builds require Darwin arm64")
    if mode not in MODES:
        _fail(f"mode must be one of {list(MODES)}")
    release_id = _expect_token(release_id, context="release_id")
    runtime_path = _validate_embedded_path(runtime_path, context="runtime")
    renderer_path = _validate_embedded_path(renderer_path, context="renderer")
    machine_runner_path = machine_runner_path.absolute()
    clang_path = clang_path.absolute()
    linker_path = linker_path.absolute()
    codesign_path = codesign_path.absolute()
    git_path = git_path.absolute()
    compiler_resource_root = compiler_resource_root.absolute()
    sdk_root = sdk_root.absolute()
    if (
        clang_path != EXPECTED_CLANG
        or linker_path != EXPECTED_LD
        or codesign_path != EXPECTED_CODESIGN
        or git_path != EXPECTED_GIT
        or compiler_resource_root != EXPECTED_COMPILER_RESOURCE_ROOT
        or sdk_root != EXPECTED_SDK_ROOT
    ):
        _fail("Xcode compiler, linker, resources, SDK, codesign, or Git path drifted")
    repo_root = Path(__file__).resolve(strict=True).parents[1]
    _require_release_member_paths(
        mode=mode,
        repo_root=repo_root,
        renderer_path=renderer_path,
        machine_runner_path=machine_runner_path,
    )
    fixed_paths = {
        LAUNCHER_SOURCE_MEMBER: repo_root / LAUNCHER_SOURCE_MEMBER,
        BUILDER_MEMBER: Path(__file__).resolve(strict=True),
        BUNDLE_BUILDER_MEMBER: repo_root / BUNDLE_BUILDER_MEMBER,
        "analysis/render_tcga_revision_rebuttal.py": renderer_path,
        MACHINE_RUNNER_MEMBER: machine_runner_path,
    }
    expected_hashes = {
        "source_bundle_sha256": _expect_sha256(
            expected_source_bundle_sha256, context="expected source bundle SHA-256"
        ),
        "launcher_source_sha256": _expect_sha256(
            expected_launcher_source_sha256, context="expected launcher source SHA-256"
        ),
        "builder_sha256": _expect_sha256(
            expected_builder_sha256, context="expected builder SHA-256"
        ),
        "bundle_builder_sha256": _expect_sha256(
            expected_bundle_builder_sha256,
            context="expected bundle builder SHA-256",
        ),
        "runtime_sha256": _expect_sha256(
            expected_runtime_sha256, context="expected runtime SHA-256"
        ),
        "renderer_sha256": _expect_sha256(
            expected_renderer_sha256, context="expected renderer SHA-256"
        ),
        "machine_runner_sha256": _expect_sha256(
            expected_machine_runner_sha256,
            context="expected machine runner SHA-256",
        ),
        "clang_sha256": _expect_sha256(
            expected_clang_sha256, context="expected clang SHA-256"
        ),
        "linker_sha256": _expect_sha256(
            expected_linker_sha256, context="expected linker SHA-256"
        ),
        "codesign_sha256": _expect_sha256(
            expected_codesign_sha256, context="expected codesign SHA-256"
        ),
        "git_sha256": _expect_sha256(
            expected_git_sha256, context="expected Git SHA-256"
        ),
        "compiler_resource_tree_sha256": _expect_sha256(
            expected_compiler_resource_tree_sha256,
            context="expected compiler resource tree SHA-256",
        ),
        "sdk_tree_sha256": _expect_sha256(
            expected_sdk_tree_sha256, context="expected SDK tree SHA-256"
        ),
        "renderer_manifest_sha256": _expect_sha256(
            expected_renderer_manifest_sha256,
            context="expected renderer manifest SHA-256",
        ),
        "pdf_sha256": _expect_sha256(
            expected_pdf_sha256, context="expected PDF SHA-256"
        ),
    }
    pins: dict[str, _PinnedFile] = {}
    stages: list[Path] = []
    failure: BaseException | None = None
    result: _MaterializedCandidates | None = None
    try:
        pins["source_bundle"] = _pin_file(
            source_bundle_path,
            maximum=MAX_BUNDLE_BYTES,
            context="rebuttal source bundle",
            expected_sha256=expected_hashes["source_bundle_sha256"],
            require_effective_user_owner=True,
            expected_mode=0o400,
        )
        pins[LAUNCHER_SOURCE_MEMBER] = _pin_file(
            fixed_paths[LAUNCHER_SOURCE_MEMBER],
            maximum=MAX_SOURCE_BYTES,
            context="launcher source",
            expected_sha256=expected_hashes["launcher_source_sha256"],
        )
        pins[BUILDER_MEMBER] = _pin_file(
            fixed_paths[BUILDER_MEMBER],
            maximum=MAX_BUILDER_BYTES,
            context="native producer builder",
            expected_sha256=expected_hashes["builder_sha256"],
        )
        pins[BUNDLE_BUILDER_MEMBER] = _pin_file(
            fixed_paths[BUNDLE_BUILDER_MEMBER],
            maximum=MAX_BUILDER_BYTES,
            context="rebuttal bundle builder",
            expected_sha256=expected_hashes["bundle_builder_sha256"],
        )
        pins["renderer"] = _pin_file(
            renderer_path,
            maximum=MAX_BUILDER_BYTES,
            context="rebuttal renderer",
            expected_sha256=expected_hashes["renderer_sha256"],
            require_effective_user_owner=True,
        )
        pins[MACHINE_RUNNER_MEMBER] = _pin_file(
            machine_runner_path,
            maximum=MAX_BUILDER_BYTES,
            context="machine runner",
            expected_sha256=expected_hashes["machine_runner_sha256"],
        )
        pins["runtime"] = _pin_file(
            runtime_path,
            maximum=MAX_TOOL_BYTES,
            context="Python runtime",
            expected_sha256=expected_hashes["runtime_sha256"],
            require_executable=True,
            require_effective_user_owner=True,
        )
        pins["clang"] = _pin_file(
            clang_path,
            maximum=MAX_TOOL_BYTES,
            context="Xcode clang",
            expected_sha256=expected_hashes["clang_sha256"],
            require_executable=True,
            require_root_owner=True,
        )
        pins["linker"] = _pin_file(
            linker_path,
            maximum=MAX_TOOL_BYTES,
            context="Xcode ld",
            expected_sha256=expected_hashes["linker_sha256"],
            require_executable=True,
            require_root_owner=True,
        )
        pins["codesign"] = _pin_file(
            codesign_path,
            maximum=MAX_TOOL_BYTES,
            context="system codesign",
            expected_sha256=expected_hashes["codesign_sha256"],
            require_executable=True,
            require_root_owner=True,
        )
        pins["git"] = _pin_file(
            git_path,
            maximum=MAX_TOOL_BYTES,
            context="Xcode Git",
            expected_sha256=expected_hashes["git_sha256"],
            require_executable=True,
            require_root_owner=True,
        )

        def guard() -> None:
            parent.revalidate(context="materialization parent")
            for name, pin in pins.items():
                pin.revalidate(context=name)

        guard()
        resource_projection = _require_tree_anchor(
            compiler_resource_root,
            expected_sha256=expected_hashes["compiler_resource_tree_sha256"],
            context="compiler resource tree",
        )
        sdk_projection = _require_tree_anchor(
            sdk_root,
            expected_sha256=expected_hashes["sdk_tree_sha256"],
            context="macOS SDK tree",
        )
        sdk_version = _sdk_version(sdk_root)
        bundle_raw = _pinned_bytes(
            pins["source_bundle"],
            maximum=MAX_BUNDLE_BYTES,
            context="rebuttal source bundle",
        )
        bundle, projection = _bundle_projection(bundle_raw)
        if bundle["release_id"] != release_id:
            _fail("source bundle release_id differs from the requested release")
        expected_output = _mapping(
            bundle["expected_output"], context="bundle expected_output"
        )
        manifest_output = _mapping(
            expected_output["renderer_manifest"], context="expected renderer manifest"
        )
        pdf_output = _mapping(expected_output["pdf"], context="expected PDF")
        if (
            manifest_output["sha256"] != expected_hashes["renderer_manifest_sha256"]
            or pdf_output["sha256"] != expected_hashes["pdf_sha256"]
        ):
            _fail("bundle expected outputs differ from independent caller anchors")
        config = _launcher_config(
            runtime=pins["runtime"],
            runtime_path=runtime_path,
            renderer_pin=pins["renderer"],
            renderer_path=renderer_path,
            bundle=bundle,
        )
        handoff = _runtime_handoff(config)
        release_files = {
            LAUNCHER_SOURCE_MEMBER: pins[LAUNCHER_SOURCE_MEMBER],
            BUILDER_MEMBER: pins[BUILDER_MEMBER],
            BUNDLE_BUILDER_MEMBER: pins[BUNDLE_BUILDER_MEMBER],
            "analysis/render_tcga_revision_rebuttal.py": pins["renderer"],
            MACHINE_RUNNER_MEMBER: pins[MACHINE_RUNNER_MEMBER],
        }
        source_release = _source_release_projection(
            mode=mode,
            release_commit=release_commit,
            release_ref=release_ref,
            git=pins["git"],
            repo_root=repo_root,
            file_pins=release_files,
            guard=guard,
        )
        clang_binary = _parse_thin_macho_header(
            _pinned_bytes(pins["clang"], maximum=MAX_TOOL_BYTES, context="clang"),
            context="Xcode clang",
        )
        linker_binary = _parse_thin_macho_header(
            _pinned_bytes(pins["linker"], maximum=MAX_TOOL_BYTES, context="ld"),
            context="Xcode ld",
        )
        git_binary = _parse_thin_macho_header(
            _pinned_bytes(pins["git"], maximum=MAX_TOOL_BYTES, context="Git"),
            context="Xcode Git",
        )
        for name, binary in (
            ("clang", clang_binary),
            ("linker", linker_binary),
            ("git", git_binary),
        ):
            if binary["architecture"] != "arm64":
                _fail(f"{name} must be one real thin arm64 Xcode executable")
        codesign_binary = _parse_fat_codesign(
            _pinned_bytes(pins["codesign"], maximum=MAX_TOOL_BYTES, context="codesign")
        )
        toolchain: dict[str, object] = {
            "clang": _tool_record(
                pins["clang"],
                locator="xcode-default-toolchain-clang",
                binary=clang_binary,
            ),
            "linker": _tool_record(
                pins["linker"],
                locator="xcode-default-toolchain-ld",
                binary=linker_binary,
            ),
            "codesign": _tool_record(
                pins["codesign"], locator="system-codesign", binary=codesign_binary
            ),
            "git": _tool_record(pins["git"], locator="xcode-git", binary=git_binary),
            "compiler_resource_tree": resource_projection.record(
                locator="xcode-clang-resource-root", root_path=compiler_resource_root
            ),
            "sdk_tree": sdk_projection.record(
                locator="xcode-macos-sdk-root", root_path=sdk_root
            ),
            "sdk_version": sdk_version,
            "linker_invocation": "direct-bounded-main-process",
            "codesign_invocation": (
                "bounded-main-path-execution; "
                "selected-fat-slice-live-mapping-not-attested"
            ),
        }
        toolchain["toolchain_projection_sha256"] = _sha256(_canonical_json(toolchain))
        source_raw = _pinned_bytes(
            pins[LAUNCHER_SOURCE_MEMBER],
            maximum=MAX_SOURCE_BYTES,
            context="launcher source",
        )
        stage_a = _new_stage_root(parent, stages, label="build-a")
        stage_b = _new_stage_root(parent, stages, label="build-b")
        build_a = _build_stage(
            stage_a,
            source_raw=source_raw,
            runtime=pins["runtime"],
            runtime_path=runtime_path,
            renderer_pin=pins["renderer"],
            renderer_path=renderer_path,
            clang=pins["clang"],
            linker=pins["linker"],
            codesign=pins["codesign"],
            sdk_root=sdk_root,
            sdk_version=sdk_version,
            resource_root=compiler_resource_root,
            guard=guard,
        )
        build_b = _build_stage(
            stage_b,
            source_raw=source_raw,
            runtime=pins["runtime"],
            runtime_path=runtime_path,
            renderer_pin=pins["renderer"],
            renderer_path=renderer_path,
            clang=pins["clang"],
            linker=pins["linker"],
            codesign=pins["codesign"],
            sdk_root=sdk_root,
            sdk_version=sdk_version,
            resource_root=compiler_resource_root,
            guard=guard,
        )
        for key in (
            "object_bytes",
            "object_sha256",
            "unsigned_bytes",
            "unsigned_sha256",
            "signed_bytes",
            "signed_sha256",
            "macho_uuid",
            "native_code_directory",
            "observations",
        ):
            if build_a[key] != build_b[key]:
                _fail(f"independent native builds differ at {key}")
        producer_a = _read_stage_member(
            stage_a / PRODUCER_MEMBER,
            maximum=MAX_EXECUTABLE_BYTES,
            context="build A final producer",
        )
        producer_b = _read_stage_member(
            stage_b / PRODUCER_MEMBER,
            maximum=MAX_EXECUTABLE_BYTES,
            context="build B final producer",
        )
        if producer_a != producer_b:
            _fail("independent final producer members differ")
        if (
            os.lstat(stage_a).st_ino == os.lstat(stage_b).st_ino
            or os.lstat(stage_a / PRODUCER_MEMBER).st_ino
            == os.lstat(stage_b / PRODUCER_MEMBER).st_ino
        ):
            _fail("independent builds alias a stage root or producer inode")
        _reject_unexpected_binary_private_paths(
            producer_a,
            runtime_path=runtime_path,
            renderer_path=renderer_path,
            stage_roots=(stage_a, stage_b),
        )
        terminal_resources = _require_tree_anchor(
            compiler_resource_root,
            expected_sha256=expected_hashes["compiler_resource_tree_sha256"],
            context="terminal compiler resource tree",
        )
        terminal_sdk = _require_tree_anchor(
            sdk_root,
            expected_sha256=expected_hashes["sdk_tree_sha256"],
            context="terminal macOS SDK tree",
        )
        if terminal_resources != resource_projection or terminal_sdk != sdk_projection:
            _fail("toolchain dependency trees changed across the two builds")
        terminal_release = _source_release_projection(
            mode=mode,
            release_commit=release_commit,
            release_ref=release_ref,
            git=pins["git"],
            repo_root=repo_root,
            file_pins=release_files,
            guard=guard,
        )
        if terminal_release != source_release:
            _fail("release source proof changed across the two builds")
        guard()
        authority = _authority_body(
            mode=mode,
            release_id=release_id,
            source_bundle=pins["source_bundle"],
            bundle_projection=projection,
            source_raw=source_raw,
            config=config,
            runtime_handoff=handoff,
            source_release=source_release,
            build_a=build_a,
            build_b=build_b,
            producer_raw=producer_a,
            code_directory=_mapping(
                build_a["native_code_directory"], context="native CodeDirectory"
            ),
            toolchain=toolchain,
            expected_hashes=expected_hashes,
        )
        authority_raw = _authority_raw(authority)
        _seal_stage(stage_a, authority_raw)
        _seal_stage(stage_b, authority_raw)
        pin_a = _pin_package(stage_a, context="sealed build A package")
        pin_b = _pin_package(stage_b, context="sealed build B package")
        try:
            if pin_a.member_bytes != pin_b.member_bytes:
                _fail("two sealed package candidates are not byte-identical")
            parsed_a = _parse_authority_raw(pin_a.member_bytes[AUTHORITY_MEMBER])
            parsed_b = _parse_authority_raw(pin_b.member_bytes[AUTHORITY_MEMBER])
            if parsed_a != authority or parsed_b != authority:
                _fail("sealed package authority differs from in-memory capsule")
            observed_cd = _parse_signed_arm64_launcher(
                pin_a.member_bytes[PRODUCER_MEMBER]
            )
            producer_record = _mapping(authority["producer"], context="producer")
            if (
                producer_record["sha256"]
                != _sha256(pin_a.member_bytes[PRODUCER_MEMBER])
                or producer_record["bytes"] != len(pin_a.member_bytes[PRODUCER_MEMBER])
                or producer_record["macho_uuid"]
                != _required_macho_uuid(
                    pin_a.member_bytes[PRODUCER_MEMBER],
                    context="packaged producer",
                )
                or producer_record["native_code_directory"] != observed_cd
            ):
                _fail("authority does not bind the exact final producer bytes")
            pin_a.revalidate(context="terminal build A package")
            pin_b.revalidate(context="terminal build B package")
            parent.revalidate(context="terminal materialization parent")
        finally:
            pin_a.close()
            pin_b.close()
        result = _MaterializedCandidates(
            primary=stage_a,
            independent=stage_b,
            producer_raw=producer_a,
            authority_raw=authority_raw,
            authority=authority,
        )
    except BaseException as error:  # noqa: BLE001 - preserve every stage.
        failure = error
    try:
        _close_pins(pins, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    if failure is not None:
        try:
            parent.revalidate(context="failed materialization parent mapping")
            parent_mapping_stable = True
        except RebuttalNativeProducerError:
            parent_mapping_stable = False
        locations = ",".join(str(path) for path in stages)
        if locations and parent_mapping_stable:
            message = (
                f"{failure}; materialized_paths={locations}; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
            raise RebuttalNativeProducerError(message) from failure
        if locations:
            message = (
                f"{failure}; materialized_paths=ambiguous-via-held-parent-fd; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
            raise RebuttalNativeProducerError(message) from failure
        if isinstance(failure, RebuttalNativeProducerError):
            raise failure
        raise RebuttalNativeProducerError(str(failure)) from failure
    if result is None:  # pragma: no cover - exhaustiveness.
        _fail("native producer materialization completed without candidates")
    return result


def _receipt_from_pin(
    pin: _PackagePin,
    independent: Path,
    *,
    replay_of: str | None,
) -> RebuttalNativeProducerReceipt:
    authority_raw = pin.member_bytes[AUTHORITY_MEMBER]
    authority = _parse_authority_raw(authority_raw)
    producer_raw = pin.member_bytes[PRODUCER_MEMBER]
    producer = _mapping(authority["producer"], context="receipt producer")
    if (
        producer["sha256"] != _sha256(producer_raw)
        or producer["bytes"] != len(producer_raw)
        or producer["macho_uuid"]
        != _required_macho_uuid(producer_raw, context="receipt producer")
        or producer["native_code_directory"]
        != _parse_signed_arm64_launcher(producer_raw)
    ):
        _fail("receipt package producer differs from its authority")
    code_directory = _mapping(
        producer["native_code_directory"], context="receipt CodeDirectory"
    )
    pin.revalidate(context="terminal receipt package")
    return RebuttalNativeProducerReceipt(
        package_root=str(pin.path),
        independent_build_root=str(independent),
        authority_sha256=_sha256(authority_raw),
        authority_bytes=len(authority_raw),
        producer_sha256=_sha256(producer_raw),
        producer_bytes=len(producer_raw),
        producer_cdhash=str(code_directory["cdhash"]),
        release_id=str(authority["release_id"]),
        mode=str(authority["mode"]),
        replay_of=replay_of,
        promotable=False,
    )


def build_rebuttal_native_producer(
    source_bundle: Path,
    destination: Path,
    *,
    runtime: Path,
    renderer_path: Path,
    machine_runner: Path,
    clang: Path,
    linker: Path,
    codesign: Path,
    git: Path,
    compiler_resource_root: Path,
    sdk_root: Path,
    release_id: str,
    mode: str,
    release_commit: str | None,
    release_ref: str | None,
    expected_source_bundle_sha256: str,
    expected_launcher_source_sha256: str,
    expected_builder_sha256: str,
    expected_bundle_builder_sha256: str,
    expected_runtime_sha256: str,
    expected_renderer_sha256: str,
    expected_machine_runner_sha256: str,
    expected_clang_sha256: str,
    expected_linker_sha256: str,
    expected_codesign_sha256: str,
    expected_git_sha256: str,
    expected_compiler_resource_tree_sha256: str,
    expected_sdk_tree_sha256: str,
    expected_renderer_manifest_sha256: str,
    expected_pdf_sha256: str,
) -> RebuttalNativeProducerReceipt:
    """Double-build, seal, and atomically publish one candidate package."""
    absolute, parent_pin = _safe_destination_parent(destination)
    candidate: _MaterializedCandidates | None = None
    published: Path | None = None
    published_pin: _PackagePin | None = None
    independent_pin: _PackagePin | None = None
    failure: BaseException | None = None
    receipt: RebuttalNativeProducerReceipt | None = None
    try:
        candidate = _materialize_candidates(
            source_bundle,
            parent_pin,
            runtime_path=runtime,
            renderer_path=renderer_path,
            machine_runner_path=machine_runner,
            clang_path=clang,
            linker_path=linker,
            codesign_path=codesign,
            git_path=git,
            compiler_resource_root=compiler_resource_root,
            sdk_root=sdk_root,
            release_id=release_id,
            mode=mode,
            release_commit=release_commit,
            release_ref=release_ref,
            expected_source_bundle_sha256=expected_source_bundle_sha256,
            expected_launcher_source_sha256=expected_launcher_source_sha256,
            expected_builder_sha256=expected_builder_sha256,
            expected_bundle_builder_sha256=expected_bundle_builder_sha256,
            expected_runtime_sha256=expected_runtime_sha256,
            expected_renderer_sha256=expected_renderer_sha256,
            expected_machine_runner_sha256=expected_machine_runner_sha256,
            expected_clang_sha256=expected_clang_sha256,
            expected_linker_sha256=expected_linker_sha256,
            expected_codesign_sha256=expected_codesign_sha256,
            expected_git_sha256=expected_git_sha256,
            expected_compiler_resource_tree_sha256=(
                expected_compiler_resource_tree_sha256
            ),
            expected_sdk_tree_sha256=expected_sdk_tree_sha256,
            expected_renderer_manifest_sha256=expected_renderer_manifest_sha256,
            expected_pdf_sha256=expected_pdf_sha256,
        )
        published_pin = _publish_candidate(candidate.primary, absolute, parent_pin)
        published = absolute
        independent_pin = _pin_package(
            candidate.independent, context="retained independent build"
        )
        expected_members = {
            PRODUCER_MEMBER: candidate.producer_raw,
            AUTHORITY_MEMBER: candidate.authority_raw,
        }
        if (
            published_pin.member_bytes != expected_members
            or independent_pin.member_bytes != expected_members
        ):
            _fail("published or retained package differs from materialized bytes")
        independent_pin.revalidate(context="terminal independent build")
        published_pin.revalidate(context="terminal published candidate")
        receipt = _receipt_from_pin(
            published_pin,
            candidate.independent,
            replay_of=None,
        )
        independent_pin.revalidate(context="post-receipt independent build")
        published_pin.revalidate(context="post-receipt published candidate")
    except BaseException as error:  # noqa: BLE001 - preserve all materialization.
        if (
            isinstance(error, _PublicationError)
            and error.renamed
            and error.location is not None
        ):
            published = error.location
        failure = error
    for label, pin in (
        ("independent package", independent_pin),
        ("published package", published_pin),
    ):
        if pin is None:
            continue
        try:
            pin.close()
        except BaseException as error:  # noqa: BLE001 - retain primary + cleanup.
            message = f"{label} descriptor cleanup failed: {error}"
            failure = RebuttalNativeProducerError(
                message if failure is None else f"{failure}; {message}"
            )
    parent_mapping_stable = True
    try:
        parent_pin.revalidate(context="terminal package destination parent")
    except BaseException as error:  # noqa: BLE001 - retain primary + mapping.
        parent_mapping_stable = False
        message = f"package parent mapping changed: {error}"
        failure = RebuttalNativeProducerError(
            message if failure is None else f"{failure}; {message}"
        )
    try:
        parent_pin.close()
    except BaseException as error:  # noqa: BLE001 - retain primary + cleanup.
        message = f"package parent descriptor cleanup failed: {error}"
        if failure is not None:
            message = f"{failure}; {message}"
        failure = RebuttalNativeProducerError(message)
    if failure is not None:
        paths = []
        if published is not None and parent_mapping_stable:
            paths.append(str(published))
        if candidate is not None and parent_mapping_stable:
            paths.append(str(candidate.independent))
            if published is None:
                paths.append(str(candidate.primary))
        if paths:
            message = (
                f"{failure}; materialized_paths={','.join(paths)}; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
            raise RebuttalNativeProducerError(message) from failure
        if candidate is not None:
            message = (
                f"{failure}; materialized_paths=ambiguous-via-held-parent-fd; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
            raise RebuttalNativeProducerError(message) from failure
        if isinstance(failure, RebuttalNativeProducerError):
            raise failure
        raise RebuttalNativeProducerError(str(failure)) from failure
    if receipt is None:  # pragma: no cover - exhaustiveness.
        _fail("native producer build completed without a receipt")
    return receipt


def validate_rebuttal_native_producer(
    source_bundle: Path,
    published_root: Path,
    replay_root: Path,
    *,
    expected_authority_sha256: str,
    expected_producer_sha256: str,
    runtime: Path,
    renderer_path: Path,
    machine_runner: Path,
    clang: Path,
    linker: Path,
    codesign: Path,
    git: Path,
    compiler_resource_root: Path,
    sdk_root: Path,
    release_id: str,
    mode: str,
    release_commit: str | None,
    release_ref: str | None,
    expected_source_bundle_sha256: str,
    expected_launcher_source_sha256: str,
    expected_builder_sha256: str,
    expected_bundle_builder_sha256: str,
    expected_runtime_sha256: str,
    expected_renderer_sha256: str,
    expected_machine_runner_sha256: str,
    expected_clang_sha256: str,
    expected_linker_sha256: str,
    expected_codesign_sha256: str,
    expected_git_sha256: str,
    expected_compiler_resource_tree_sha256: str,
    expected_sdk_tree_sha256: str,
    expected_renderer_manifest_sha256: str,
    expected_pdf_sha256: str,
) -> RebuttalNativeProducerReceipt:
    """Hold the original open, rebuild privately, compare, then publish replay."""
    expected_authority = _expect_sha256(
        expected_authority_sha256, context="expected authority SHA-256"
    )
    expected_producer = _expect_sha256(
        expected_producer_sha256, context="expected producer SHA-256"
    )
    absolute, parent_pin = _safe_destination_parent(replay_root)
    original: _PackagePin | None = None
    candidate: _MaterializedCandidates | None = None
    replay: Path | None = None
    replay_pin: _PackagePin | None = None
    independent_pin: _PackagePin | None = None
    failure: BaseException | None = None
    receipt: RebuttalNativeProducerReceipt | None = None
    try:
        original = _pin_package(published_root, context="original producer package")
        original_authority = original.member_bytes[AUTHORITY_MEMBER]
        original_producer = original.member_bytes[PRODUCER_MEMBER]
        if _sha256(original_authority) != expected_authority:
            _fail("original authority differs from its external caller anchor")
        if _sha256(original_producer) != expected_producer:
            _fail("original producer differs from its external caller anchor")
        parsed = _parse_authority_raw(original_authority)
        if (
            parsed["release_id"] != release_id
            or parsed["mode"] != mode
            or _mapping(parsed["source_bundle"], context="source bundle")["sha256"]
            != expected_source_bundle_sha256
        ):
            _fail("original authority differs from replay configuration")
        candidate = _materialize_candidates(
            source_bundle,
            parent_pin,
            runtime_path=runtime,
            renderer_path=renderer_path,
            machine_runner_path=machine_runner,
            clang_path=clang,
            linker_path=linker,
            codesign_path=codesign,
            git_path=git,
            compiler_resource_root=compiler_resource_root,
            sdk_root=sdk_root,
            release_id=release_id,
            mode=mode,
            release_commit=release_commit,
            release_ref=release_ref,
            expected_source_bundle_sha256=expected_source_bundle_sha256,
            expected_launcher_source_sha256=expected_launcher_source_sha256,
            expected_builder_sha256=expected_builder_sha256,
            expected_bundle_builder_sha256=expected_bundle_builder_sha256,
            expected_runtime_sha256=expected_runtime_sha256,
            expected_renderer_sha256=expected_renderer_sha256,
            expected_machine_runner_sha256=expected_machine_runner_sha256,
            expected_clang_sha256=expected_clang_sha256,
            expected_linker_sha256=expected_linker_sha256,
            expected_codesign_sha256=expected_codesign_sha256,
            expected_git_sha256=expected_git_sha256,
            expected_compiler_resource_tree_sha256=(
                expected_compiler_resource_tree_sha256
            ),
            expected_sdk_tree_sha256=expected_sdk_tree_sha256,
            expected_renderer_manifest_sha256=expected_renderer_manifest_sha256,
            expected_pdf_sha256=expected_pdf_sha256,
        )
        if (
            candidate.authority_raw != original_authority
            or candidate.producer_raw != original_producer
        ):
            _fail("independent candidate differs from the held original package")
        original.revalidate(context="original before replay publication")
        replay_pin = _publish_candidate(candidate.primary, absolute, parent_pin)
        replay = absolute
        independent_pin = _pin_package(
            candidate.independent, context="retained replay-independent package"
        )
        expected_members = {
            PRODUCER_MEMBER: candidate.producer_raw,
            AUTHORITY_MEMBER: candidate.authority_raw,
        }
        if (
            replay_pin.member_bytes != expected_members
            or independent_pin.member_bytes != expected_members
            or expected_members != original.member_bytes
        ):
            _fail("published replay or independent build differs from original")
        original.revalidate(context="terminal original package")
        replay_pin.revalidate(context="terminal replay package")
        independent_pin.revalidate(context="terminal replay-independent package")
        original.revalidate(context="second terminal original package")
        replay_pin.revalidate(context="second terminal replay package")
        receipt = _receipt_from_pin(
            replay_pin,
            candidate.independent,
            replay_of=str(published_root.absolute()),
        )
        original.revalidate(context="post-receipt original package")
        replay_pin.revalidate(context="post-receipt replay package")
        independent_pin.revalidate(context="post-receipt replay-independent package")
    except BaseException as error:  # noqa: BLE001 - preserve all roots.
        if (
            isinstance(error, _PublicationError)
            and error.renamed
            and error.location is not None
        ):
            replay = error.location
        failure = error
    cleanup_errors: list[str] = []
    for label, pin in (
        ("replay independent", independent_pin),
        ("replay", replay_pin),
        ("original", original),
    ):
        if pin is None:
            continue
        try:
            pin.close()
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(f"{label} package cleanup failed: {error}")
    parent_mapping_stable = True
    try:
        parent_pin.revalidate(context="terminal replay destination parent")
    except BaseException as error:  # noqa: BLE001
        parent_mapping_stable = False
        cleanup_errors.append(f"replay parent mapping changed: {error}")
    try:
        parent_pin.close()
    except BaseException as error:  # noqa: BLE001
        cleanup_errors.append(f"replay parent cleanup failed: {error}")
    if cleanup_errors:
        cleanup = "; ".join(cleanup_errors)
        failure = RebuttalNativeProducerError(
            cleanup if failure is None else f"{failure}; {cleanup}"
        )
    if failure is not None:
        paths = []
        if replay is not None and parent_mapping_stable:
            paths.append(str(replay))
        if candidate is not None and parent_mapping_stable:
            paths.append(str(candidate.independent))
            if replay is None:
                paths.append(str(candidate.primary))
        if paths:
            message = (
                f"{failure}; materialized_paths={','.join(paths)}; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
            raise RebuttalNativeProducerError(message) from failure
        if candidate is not None:
            message = (
                f"{failure}; materialized_paths=ambiguous-via-held-parent-fd; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
            raise RebuttalNativeProducerError(message) from failure
        if isinstance(failure, RebuttalNativeProducerError):
            raise failure
        raise RebuttalNativeProducerError(str(failure)) from failure
    if receipt is None:  # pragma: no cover - exhaustiveness.
        _fail("native producer replay completed without a receipt")
    return receipt


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--renderer-sha256", required=True)
    parser.add_argument("--machine-runner", type=Path, required=True)
    parser.add_argument("--machine-runner-sha256", required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--clang-sha256", required=True)
    parser.add_argument("--linker", type=Path, required=True)
    parser.add_argument("--linker-sha256", required=True)
    parser.add_argument("--codesign", type=Path, default=EXPECTED_CODESIGN)
    parser.add_argument("--codesign-sha256", required=True)
    parser.add_argument("--git", type=Path, default=EXPECTED_GIT)
    parser.add_argument("--git-sha256", required=True)
    parser.add_argument("--compiler-resource-root", type=Path, required=True)
    parser.add_argument("--compiler-resource-tree-sha256", required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--sdk-tree-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--builder-sha256", required=True)
    parser.add_argument("--bundle-builder-sha256", required=True)
    parser.add_argument("--renderer-manifest-sha256", required=True)
    parser.add_argument("--pdf-sha256", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--release-commit")
    parser.add_argument("--release-ref")


def _cli_kwargs(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "runtime": arguments.runtime,
        "renderer_path": arguments.renderer,
        "machine_runner": arguments.machine_runner,
        "clang": arguments.clang,
        "linker": arguments.linker,
        "codesign": arguments.codesign,
        "git": arguments.git,
        "compiler_resource_root": arguments.compiler_resource_root,
        "sdk_root": arguments.sdk_root,
        "release_id": arguments.release_id,
        "mode": arguments.mode,
        "release_commit": arguments.release_commit,
        "release_ref": arguments.release_ref,
        "expected_source_bundle_sha256": arguments.source_bundle_sha256,
        "expected_launcher_source_sha256": arguments.launcher_source_sha256,
        "expected_builder_sha256": arguments.builder_sha256,
        "expected_bundle_builder_sha256": arguments.bundle_builder_sha256,
        "expected_runtime_sha256": arguments.runtime_sha256,
        "expected_renderer_sha256": arguments.renderer_sha256,
        "expected_machine_runner_sha256": arguments.machine_runner_sha256,
        "expected_clang_sha256": arguments.clang_sha256,
        "expected_linker_sha256": arguments.linker_sha256,
        "expected_codesign_sha256": arguments.codesign_sha256,
        "expected_git_sha256": arguments.git_sha256,
        "expected_compiler_resource_tree_sha256": (
            arguments.compiler_resource_tree_sha256
        ),
        "expected_sdk_tree_sha256": arguments.sdk_tree_sha256,
        "expected_renderer_manifest_sha256": arguments.renderer_manifest_sha256,
        "expected_pdf_sha256": arguments.pdf_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    digest = subparsers.add_parser(
        "tree-digest", help="print one canonical compiler-resource or SDK tree digest"
    )
    digest.add_argument("--root", type=Path, required=True)
    digest.add_argument("--context", default="toolchain tree")
    build = subparsers.add_parser(
        "build", help="double-build and publish one candidate native producer package"
    )
    _add_common_arguments(build)
    build.add_argument("--destination", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", help="rebuild and publish one externally anchored replay"
    )
    _add_common_arguments(validate)
    validate.add_argument("--published-root", type=Path, required=True)
    validate.add_argument("--replay-root", type=Path, required=True)
    validate.add_argument("--authority-sha256", required=True)
    validate.add_argument("--producer-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run tree-digest, candidate build, or externally anchored replay mode."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "tree-digest":
        print(
            json.dumps(
                asdict(_tree_projection(arguments.root, context=arguments.context)),
                sort_keys=True,
            )
        )
        return 0
    kwargs = _cli_kwargs(arguments)
    if arguments.command == "build":
        receipt = build_rebuttal_native_producer(
            arguments.source_bundle,
            arguments.destination,
            **kwargs,
        )
    else:
        receipt = validate_rebuttal_native_producer(
            arguments.source_bundle,
            arguments.published_root,
            arguments.replay_root,
            expected_authority_sha256=arguments.authority_sha256,
            expected_producer_sha256=arguments.producer_sha256,
            **kwargs,
        )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Build and replay native machine-QA evidence for four revision PDFs.

This module is intentionally result blind.  It accepts only the exact rendered
document inventory and invokes a pinned local Poppler installation.  It does not
open manuscript sources, scientific tables, result CSVs, or any other revision
artifact.  A build is published as one immutable directory with an atomic
no-replace rename.  Validation independently repeats every native invocation in a
caller-named private tree and requires an identical canonical manifest and member
set.

The closure proves bounded PDF structure, font/image inventories, deterministic
page rendering, and native PNG integrity.  It does not prove scientific accuracy,
human visual approval, accessibility, journal acceptance, or submission status.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import resource
import selectors
import signal
import stat
import struct
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# This is a deliberately narrow executable contract.  Keep every bound explicit.
# ruff: noqa: PLR0913, S105, TRY300

MACHINE_CLOSURE_SCHEMA: Final = "dialect-revision-rendered-document-machine-closure-v1"
PRODUCER_RECEIPT_SCHEMA: Final = (
    "dialect-revision-rendered-document-machine-producer-receipt-v1"
)
MACHINE_CLOSURE_CONTRACT: Final = "four-pdf-native-poppler-replay-v1"
PDF_ORDER: Final = (
    ("clean", "manuscript-clean.pdf"),
    ("marked", "manuscript-marked.pdf"),
    ("s1", "s1-appendix.pdf"),
    ("rebuttal", "response-to-reviewers.pdf"),
)
PDF_IDS: Final = tuple(pdf_id for pdf_id, _ in PDF_ORDER)
PDF_MEMBERS: Final = tuple(member for _, member in PDF_ORDER)
TOOL_ORDER: Final = ("pdfinfo", "pdffonts", "pdfimages", "pdftoppm")
EXACT_ENVIRONMENT: Final = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
RENDER_DPI: Final = 150
EXPECTED_PNG_PIXELS_PER_METER: Final = RENDER_DPI * 10_000 // 254

# Darwin-only, fail-closed execution attestation.  Spawn, process-region, and
# signing flag values follow the Darwin headers.  CS_OPS_STATUS/CDHASH are the
# stable libSystem csops ABI values and are also verified by the real canary.
POSIX_SPAWN_START_SUSPENDED: Final = 0x0080
POSIX_SPAWN_SETSID: Final = 0x0400
POSIX_SPAWN_CLOEXEC_DEFAULT: Final = 0x4000
DARWIN_SPAWN_FLAGS: Final = (
    POSIX_SPAWN_START_SUSPENDED | POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT
)
DARWIN_P_PID: Final = 1
DARWIN_WNOHANG: Final = 0x00000001
DARWIN_WEXITED: Final = 0x00000004
DARWIN_WSTOPPED: Final = 0x00000008
DARWIN_WNOWAIT: Final = 0x00000020
DARWIN_CLD_EXITED: Final = 1
DARWIN_CLD_KILLED: Final = 2
DARWIN_CLD_DUMPED: Final = 3
DARWIN_CLD_STOPPED: Final = 5
PROC_PIDREGIONPATHINFO: Final = 8
PROC_REGIONWITHPATHINFO_SIZE: Final = 1272
PROC_REGIONINFO_SIZE: Final = 96
VNODE_PATH_OFFSET: Final = 248
MAXPATHLEN: Final = 1024
VM_PROT_EXECUTE: Final = 0x04
CS_OPS_STATUS: Final = 0
CS_OPS_CDHASH: Final = 5
CS_VALID: Final = 0x00000001
CS_INVALID_ALLOWED: Final = 0x00000020
CS_KILL: Final = 0x00000200
CS_KILLED: Final = 0x01000000
CS_DEBUGGED: Final = 0x10000000
CS_SIGNED: Final = 0x20000000
REQUIRED_CS_FLAGS: Final = CS_VALID | CS_KILL | CS_SIGNED
REJECTED_CS_FLAGS: Final = CS_INVALID_ALLOWED | CS_KILLED | CS_DEBUGGED
CS_CDHASH_LEN: Final = 20
MH_MAGIC_64: Final = 0xFEEDFACF
MH_EXECUTE: Final = 2
CPU_TYPE_ARM64: Final = 0x0100000C
LC_CODE_SIGNATURE: Final = 0x1D
CSMAGIC_CODEDIRECTORY: Final = 0xFADE0C02
CSMAGIC_EMBEDDED_SIGNATURE: Final = 0xFADE0CC0
CSSLOT_CODEDIRECTORY: Final = 0
MAX_MACH_LOAD_COMMANDS: Final = 4096
MAX_CODE_SIGNATURE_BLOBS: Final = 64
MAX_PROCESS_GROUP_MEMBERS: Final = 4096
ATTESTATION_TIMEOUT_SECONDS: Final = 5.0

MAX_PDF_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_PDF_BYTES: Final = 512 * 1024 * 1024
MAX_RAW_STREAM_BYTES: Final = 4 * 1024 * 1024
MAX_STDERR_BYTES: Final = 256 * 1024
MAX_PNG_BYTES: Final = 32 * 1024 * 1024
MAX_PAGE_EDGE_PIXELS: Final = 4096
MAX_PAGE_PIXELS: Final = 8_000_000
MAX_MEMBER_BYTES: Final = MAX_PDF_BYTES
MAX_OUTPUT_BYTES: Final = 3 * 1024 * 1024 * 1024
MAX_PAGES_PER_PDF: Final = 256
MAX_TOTAL_PAGES: Final = 512
MAX_FONTS_PER_PDF: Final = 10_000
MAX_IMAGES_PER_PDF: Final = 10_000
MAX_OUTPUT_FILES: Final = 2 + (8 * 2) + (len(PDF_ORDER) * 8) + (MAX_TOTAL_PAGES * 6)
MAX_OUTPUT_DIRECTORIES: Final = 4 + (len(PDF_ORDER) * 3)
MAX_PROCESSES: Final = len(TOOL_ORDER) + (len(PDF_ORDER) * 4) + (MAX_TOTAL_PAGES * 2)
MAX_OPEN_FILE_DESCRIPTORS: Final = 48
FD_SAFETY_HEADROOM: Final = 12
TOOL_TIMEOUT_SECONDS: Final = 60.0
RENDER_TIMEOUT_SECONDS: Final = 120.0
READ_CHUNK_BYTES: Final = 64 * 1024
US_LETTER_WIDTH_MILLIPOINTS: Final = 612_000
US_LETTER_HEIGHT_MILLIPOINTS: Final = 792_000
MANIFEST_MEMBER: Final = "machine-manifest.json"
PRODUCER_MEMBER: Final = "producer-receipt.json"
BUILDER_MEMBER: Final = (
    "analysis/build_tcga_revision_rendered_document_machine_closure.py"
)
PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
VERSION_RE: Final = re.compile(r"^[a-z]+ version ([!-~]+)$")
PAGE_SIZE_RE: Final = re.compile(
    r"^Page\s+(?:(\d+)\s+)?size:\s+"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s+x\s+"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s+pts(?:\s+.*)?$",
)
PAGE_ROT_RE: Final = re.compile(r"^Page\s+(?:(\d+)\s+)?rot:\s+(-?[0-9]+)$")
PAGE_BOX_RE: Final = re.compile(
    r"^Page\s+(?:(\d+)\s+)?(MediaBox|CropBox):\s+"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s+"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s+"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)\s+"
    r"([-+]?[0-9]+(?:\.[0-9]+)?)$",
)
FONT_RE: Final = re.compile(
    r"^(\S+)\s+"
    r"(Type 1C?|Type 3|TrueType|CID Type 0C?|CID TrueType)\s+"
    r"(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+(\d+)\s+(\d+)\s*$",
)
IMAGE_ROW_RE: Final = re.compile(r"^\s*\d+\s+\d+\s+")


class MachineClosureError(ValueError):
    """Raised when native machine-QA production or replay is invalid."""


@dataclass(frozen=True, slots=True)
class MachineClosureReceipt:
    """Summarize one published or independently replayed closure."""

    manifest_path: str
    manifest_sha256: str
    pdf_set_sha256: str
    tool_set_sha256: str
    render_set_sha256: str
    pdf_count: int
    page_count: int
    machine_pass_count: int
    replay_root: str | None


@dataclass(slots=True)
class _PinnedFile:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str

    def close(self) -> None:
        """Close the owned descriptor exactly once."""
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
        """Close the owned descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class _ProcRegionInfo(ctypes.Structure):
    _fields_ = [
        ("pri_protection", ctypes.c_uint32),
        ("pri_max_protection", ctypes.c_uint32),
        ("pri_inheritance", ctypes.c_uint32),
        ("pri_flags", ctypes.c_uint32),
        ("pri_offset", ctypes.c_uint64),
        ("pri_behavior", ctypes.c_uint32),
        ("pri_user_wired_count", ctypes.c_uint32),
        ("pri_user_tag", ctypes.c_uint32),
        ("pri_pages_resident", ctypes.c_uint32),
        ("pri_pages_shared_now_private", ctypes.c_uint32),
        ("pri_pages_swapped_out", ctypes.c_uint32),
        ("pri_pages_dirtied", ctypes.c_uint32),
        ("pri_ref_count", ctypes.c_uint32),
        ("pri_shadow_depth", ctypes.c_uint32),
        ("pri_share_mode", ctypes.c_uint32),
        ("pri_private_pages_resident", ctypes.c_uint32),
        ("pri_shared_pages_resident", ctypes.c_uint32),
        ("pri_obj_id", ctypes.c_uint32),
        ("pri_depth", ctypes.c_uint32),
        ("pri_address", ctypes.c_uint64),
        ("pri_size", ctypes.c_uint64),
    ]


class _VinfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_atime", ctypes.c_int64),
        ("vst_atimensec", ctypes.c_int64),
        ("vst_mtime", ctypes.c_int64),
        ("vst_mtimensec", ctypes.c_int64),
        ("vst_ctime", ctypes.c_int64),
        ("vst_ctimensec", ctypes.c_int64),
        ("vst_birthtime", ctypes.c_int64),
        ("vst_birthtimensec", ctypes.c_int64),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class _VnodeInfo(ctypes.Structure):
    _fields_ = [
        ("vi_stat", _VinfoStat),
        ("vi_type", ctypes.c_int32),
        ("vi_pad", ctypes.c_int32),
        ("vi_fsid", ctypes.c_int32 * 2),
    ]


class _VnodeInfoPath(ctypes.Structure):
    _fields_ = [
        ("vip_vi", _VnodeInfo),
        ("vip_path", ctypes.c_char * MAXPATHLEN),
    ]


class _ProcRegionWithPathInfo(ctypes.Structure):
    _fields_ = [
        ("prp_prinfo", _ProcRegionInfo),
        ("prp_vip", _VnodeInfoPath),
    ]


class _DarwinSigInfo(ctypes.Structure):
    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", ctypes.c_void_p),
        ("si_band", ctypes.c_long),
        ("reserved", ctypes.c_ulong * 7),
    ]


@dataclass(slots=True)
class _Production:
    manifest: dict[str, object]
    manifest_raw: bytes
    member_inventory: list[dict[str, object]]
    page_count: int


@dataclass(slots=True)
class _ProcessBudget:
    count: int = 0

    def consume(self) -> None:
        """Consume one explicitly bounded child-process slot."""
        self.count += 1
        if self.count > MAX_PROCESSES:
            _fail(f"process count exceeds the {MAX_PROCESSES}-process limit")


@dataclass(slots=True)
class _PageBudget:
    count: int = 0

    def consume(self, pages: int) -> None:
        """Reserve a document's pages before page-level work begins."""
        self.count += pages
        if self.count > MAX_TOTAL_PAGES:
            _fail(f"page count exceeds the {MAX_TOTAL_PAGES}-page aggregate limit")


def _fail(message: str) -> NoReturn:
    raise MachineClosureError(message)


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


def _expect_token(value: str, *, context: str) -> str:
    if TOKEN_RE.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase canonical token")
    return value


def _expect_sha256(value: str, *, context: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _relative_member(value: str, *, context: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or len(pure.parts) > 3
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{context} is not one bounded canonical POSIX member")
    return value


def _read_descriptor(descriptor: int, *, maximum: int, context: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, READ_CHUNK_BYTES)
        if not block:
            break
        size += len(block)
        if size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        chunks.append(block)
    return b"".join(chunks)


def _hash_descriptor(descriptor: int, *, maximum: int, context: str) -> str:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, READ_CHUNK_BYTES)
        if not block:
            break
        size += len(block)
        if size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        digest.update(block)
    return digest.hexdigest()


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
        or path_entry.st_nlink != 1
    ):
        _fail(f"{context} must be a canonical single-link regular file")
    if path_entry.st_size > maximum:
        _fail(f"{context} exceeds the {maximum}-byte limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    try:
        entry = os.fstat(descriptor)
        identity = (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns)
        named_identity = (
            path_entry.st_dev,
            path_entry.st_ino,
            path_entry.st_size,
            path_entry.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or entry.st_size > maximum
            or identity != named_identity
        ):
            _fail(f"{context} changed while it was pinned")
        digest = _hash_descriptor(descriptor, maximum=maximum, context=context)
        if os.lseek(descriptor, 0, os.SEEK_END) != entry.st_size:
            _fail(f"{context} changed while it was read")
        return _PinnedFile(
            path=absolute,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            sha256=digest,
        )
    except BaseException:
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
        if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (
            path_entry.st_dev,
            path_entry.st_ino,
        ):
            _fail(f"{context} changed while it was pinned")
        return _PinnedRoot(
            path=absolute,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            mtime_ns=entry.st_mtime_ns,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _open_root_member(
    root: _PinnedRoot,
    member: str,
    *,
    maximum: int,
    expected_size: int | None,
    context: str,
) -> _PinnedFile:
    canonical = _relative_member(member, context=f"{context} member")
    parts = PurePosixPath(canonical).parts
    directory_descriptor = os.dup(root.descriptor)
    descriptor = -1
    try:
        for part in parts[:-1]:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            child = os.open(part, flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = child
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(parts[-1], flags, dir_fd=directory_descriptor)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    finally:
        os.close(directory_descriptor)
    try:
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or entry.st_size > maximum
            or (expected_size is not None and entry.st_size != expected_size)
        ):
            _fail(f"{context} identity or size changed after inventory preflight")
        digest = _hash_descriptor(descriptor, maximum=maximum, context=context)
        if os.lseek(descriptor, 0, os.SEEK_END) != entry.st_size:
            _fail(f"{context} changed while it was read")
        return _PinnedFile(
            path=root.path / canonical,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            sha256=digest,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_file(pinned: _PinnedFile, *, context: str) -> None:
    entry = os.fstat(pinned.descriptor)
    try:
        path_entry = os.lstat(pinned.path)
        resolved = pinned.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} path disappeared after validation: {error}")
    identity = (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns)
    expected = (pinned.device, pinned.inode, pinned.size, pinned.mtime_ns)
    if (
        identity != expected
        or not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISREG(path_entry.st_mode)
        or resolved != pinned.path
        or (path_entry.st_dev, path_entry.st_ino) != (pinned.device, pinned.inode)
    ):
        _fail(f"{context} identity changed after validation")
    if (
        _hash_descriptor(
            pinned.descriptor,
            maximum=max(pinned.size, 1),
            context=context,
        )
        != pinned.sha256
    ):
        _fail(f"{context} bytes changed after validation")


def _revalidate_root(
    root: _PinnedRoot,
    *,
    context: str,
    check_mtime: bool = True,
) -> None:
    entry = os.fstat(root.descriptor)
    try:
        path_entry = os.lstat(root.path)
        resolved = root.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} path disappeared after validation: {error}")
    if (
        (entry.st_dev, entry.st_ino) != (root.device, root.inode)
        or (check_mtime and entry.st_mtime_ns != root.mtime_ns)
        or not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISDIR(path_entry.st_mode)
        or resolved != root.path
        or (path_entry.st_dev, path_entry.st_ino) != (root.device, root.inode)
    ):
        _fail(f"{context} identity changed after validation")


def _inventory_root(
    root: _PinnedRoot,
    *,
    context: str,
    maximum_entries: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    members: list[str] = []
    sizes: dict[str, int] = {}

    def walk(descriptor: int, prefix: tuple[str, ...]) -> None:
        try:
            entries = os.scandir(descriptor)
        except OSError as error:
            _fail(f"cannot list {context}: {error}")
        with entries:
            for directory_entry in entries:
                if len(members) >= maximum_entries:
                    _fail(f"{context} exceeds the {maximum_entries}-entry limit")
                name = directory_entry.name
                if name in {"", ".", ".."} or "/" in name or "\\" in name:
                    _fail(f"{context} contains a noncanonical member name")
                try:
                    entry = directory_entry.stat(follow_symlinks=False)
                except OSError as error:
                    _fail(f"cannot inspect {context} member {name!r}: {error}")
                path_parts = (*prefix, name)
                member = PurePosixPath(*path_parts).as_posix()
                if stat.S_ISREG(entry.st_mode):
                    if entry.st_nlink != 1:
                        _fail(f"{context} member {member!r} must have one hard link")
                    members.append(member)
                    sizes[member] = entry.st_size
                    continue
                if not stat.S_ISDIR(entry.st_mode) or len(path_parts) >= 3:
                    _fail(f"{context} member {member!r} has invalid type/depth")
                members.append(f"{member}/")
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                try:
                    child = os.open(name, flags, dir_fd=descriptor)
                except OSError as error:
                    _fail(f"cannot open {context} directory {member!r}: {error}")
                try:
                    walk(child, path_parts)
                finally:
                    os.close(child)

    walk(root.descriptor, ())
    return tuple(sorted(members)), sizes


def _validate_exact_root_inventory(
    root: _PinnedRoot,
    *,
    expected: Sequence[str],
    context: str,
    maximum_each: int,
    maximum_total: int,
) -> dict[str, int]:
    expected_tuple = tuple(sorted(expected))
    if len(expected_tuple) != len(set(expected_tuple)):
        _fail(f"{context} expected inventory contains duplicates")
    expected_directories = {
        PurePosixPath(*PurePosixPath(member).parts[:index]).as_posix() + "/"
        for member in expected_tuple
        for index in range(1, len(PurePosixPath(member).parts))
    }
    expected_inventory = tuple(sorted((*expected_tuple, *expected_directories)))
    actual, sizes = _inventory_root(
        root,
        context=context,
        maximum_entries=max(len(expected_inventory) + 1, 8),
    )
    if actual != expected_inventory:
        _fail(
            f"{context} inventory mismatch; expected {list(expected_inventory)}, "
            f"found {list(actual)}",
        )
    total = 0
    for member in expected_tuple:
        size = sizes[member]
        if size > maximum_each:
            _fail(f"{context} member {member!r} exceeds the per-file byte limit")
        total += size
        if total > maximum_total:
            _fail(f"{context} exceeds the aggregate byte limit")
    return sizes


def _pread_exact(
    descriptor: int,
    size: int,
    offset: int,
    *,
    context: str,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        try:
            chunk = os.pread(descriptor, min(remaining, READ_CHUNK_BYTES), position)
        except OSError as error:
            _fail(f"cannot read {context}: {error}")
        if not chunk:
            _fail(f"{context} ended before its declared size")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_png_dimensions(
    pinned: _PinnedFile,
    *,
    context: str,
) -> tuple[int, int]:
    if (
        _pread_exact(
            pinned.descriptor,
            len(PNG_SIGNATURE),
            0,
            context=context,
        )
        != PNG_SIGNATURE
    ):
        _fail(f"{context} lacks the PNG signature")
    offset = len(PNG_SIGNATURE)
    seen_ihdr = False
    seen_plte = False
    seen_idat = False
    seen_iend = False
    seen_phys = False
    idat_closed = False
    unsupported_ancillary: bytes | None = None
    width = 0
    height = 0
    color_type = 0
    expected_decoded = 0
    decoded_count = 0
    next_filter_offset = 0
    row_stride = 0
    decompressor: zlib.Decompress | None = None

    def consume_decoded(decoded: bytes) -> None:
        nonlocal decoded_count, next_filter_offset
        if decoded_count + len(decoded) > expected_decoded:
            _fail(f"{context} expands beyond its exact scanline bound")
        end = decoded_count + len(decoded)
        while next_filter_offset < end:
            filter_value = decoded[next_filter_offset - decoded_count]
            if filter_value > 4:
                _fail(f"{context} has an invalid PNG scanline filter")
            next_filter_offset += row_stride
        decoded_count = end

    while offset < pinned.size:
        header = _pread_exact(
            pinned.descriptor,
            8,
            offset,
            context=f"{context} chunk header",
        )
        length = struct.unpack(">I", header[:4])[0]
        chunk_type = header[4:]
        if any(
            byte not in range(ord("A"), ord("Z") + 1)
            and byte not in range(ord("a"), ord("z") + 1)
            for byte in chunk_type
        ) or not ord("A") <= chunk_type[2] <= ord("Z"):
            _fail(f"{context} has an invalid PNG chunk type code")
        data_offset = offset + 8
        end = data_offset + length + 4
        if end > pinned.size:
            _fail(f"{context} has a PNG chunk outside the file")
        if chunk_type in {b"acTL", b"fcTL", b"fdAT"}:
            _fail(f"{context} must not be APNG")
        if not seen_ihdr and (chunk_type != b"IHDR" or length != 13):
            _fail(f"{context} must begin with one 13-byte IHDR")
        if seen_ihdr and chunk_type == b"IHDR":
            _fail(f"{context} contains multiple IHDR chunks")
        if (
            chunk_type not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
            and 65 <= chunk_type[0] <= 90
        ):
            _fail(f"{context} contains an unknown critical PNG chunk")
        if 97 <= chunk_type[0] <= 122 and chunk_type != b"pHYs":
            unsupported_ancillary = unsupported_ancillary or chunk_type
        if chunk_type == b"pHYs":
            if seen_phys or seen_idat or length != 9:
                _fail(f"{context} has invalid pHYs ordering or size")
            seen_phys = True
        if chunk_type == b"PLTE":
            if seen_plte or seen_idat or not 3 <= length <= 768 or length % 3:
                _fail(f"{context} has invalid PLTE ordering or size")
            if color_type in {0, 4}:
                _fail(f"{context} grayscale PNG must not contain PLTE")
            seen_plte = True
        if chunk_type == b"IDAT":
            if idat_closed or not seen_ihdr:
                _fail(f"{context} has nonconsecutive or misplaced IDAT chunks")
            seen_idat = True
        elif seen_idat and chunk_type != b"IEND":
            idat_closed = True
        if chunk_type == b"IEND" and (length != 0 or not seen_idat):
            _fail(f"{context} has invalid IEND ordering or size")

        crc = zlib.crc32(chunk_type)
        remaining = length
        position = data_offset
        captured_chunk = b""
        while remaining:
            block = _pread_exact(
                pinned.descriptor,
                min(remaining, READ_CHUNK_BYTES),
                position,
                context=f"{context} chunk data",
            )
            crc = zlib.crc32(block, crc)
            if chunk_type in {b"IHDR", b"pHYs"}:
                captured_chunk += block
            elif chunk_type == b"IDAT":
                if decompressor is None:
                    decompressor = zlib.decompressobj()
                compressed = block
                try:
                    while compressed:
                        limit = min(
                            READ_CHUNK_BYTES,
                            expected_decoded - decoded_count + 1,
                        )
                        decoded = decompressor.decompress(compressed, max(limit, 1))
                        consume_decoded(decoded)
                        compressed = decompressor.unconsumed_tail
                except zlib.error as error:
                    _fail(f"{context} has invalid zlib image data: {error}")
                if decompressor.unused_data:
                    _fail(f"{context} contains data after its zlib stream")
            position += len(block)
            remaining -= len(block)
        declared_crc = struct.unpack(
            ">I",
            _pread_exact(
                pinned.descriptor,
                4,
                data_offset + length,
                context=f"{context} chunk CRC",
            ),
        )[0]
        if crc & 0xFFFFFFFF != declared_crc:
            _fail(f"{context} has an invalid PNG chunk CRC")

        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", captured_chunk[:8])
            bit_depth, color_type, compression, filtering, interlace = captured_chunk[
                8:13
            ]
            if not width or not height:
                _fail(f"{context} has zero PNG dimensions")
            if (
                bit_depth != 8
                or color_type not in {0, 2, 4, 6}
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                _fail(f"{context} has an unsupported IHDR encoding")
            if width > MAX_PAGE_EDGE_PIXELS or height > MAX_PAGE_EDGE_PIXELS:
                _fail(f"{context} exceeds the maximum page edge")
            if width * height > MAX_PAGE_PIXELS:
                _fail(f"{context} exceeds the maximum page pixel count")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
            row_stride = width * channels + 1
            expected_decoded = row_stride * height
            seen_ihdr = True
        if chunk_type == b"pHYs":
            x_resolution, y_resolution, unit = struct.unpack(">IIB", captured_chunk)
            if (
                unit != 1
                or x_resolution != EXPECTED_PNG_PIXELS_PER_METER
                or y_resolution != EXPECTED_PNG_PIXELS_PER_METER
            ):
                _fail(f"{context} pHYs does not encode the fixed 150 dpi profile")
        if chunk_type == b"IEND":
            if end != pinned.size:
                _fail(f"{context} has bytes after IEND")
            seen_iend = True
            break
        offset = end
    if (
        not seen_ihdr
        or not seen_phys
        or not seen_idat
        or not seen_iend
        or decompressor is None
    ):
        _fail(f"{context} lacks a complete PNG structure")
    try:
        consume_decoded(
            decompressor.flush(max(expected_decoded - decoded_count + 1, 1)),
        )
    except zlib.error as error:
        _fail(f"{context} has invalid terminal zlib data: {error}")
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or decoded_count != expected_decoded
        or next_filter_offset != expected_decoded
    ):
        _fail(f"{context} does not decode to its exact complete scanline geometry")
    if unsupported_ancillary is not None:
        _fail(
            f"{context} contains unsupported ancillary PNG chunk "
            f"{unsupported_ancillary.decode('ascii')}",
        )
    return width, height


def _validate_fd_headroom(owned: int) -> None:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return
    required = owned + MAX_OPEN_FILE_DESCRIPTORS + FD_SAFETY_HEADROOM
    if soft < required:
        _fail(
            f"RLIMIT_NOFILE headroom is insufficient: need {required}, have {soft}",
        )


def _canonical_existing_directory(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    try:
        entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if (
        stat.S_ISLNK(entry.st_mode)
        or not stat.S_ISDIR(entry.st_mode)
        or resolved != absolute
    ):
        _fail(f"{context} must be a canonical non-symlink directory")
    return absolute


def _canonical_new_directory(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    parent = _canonical_existing_directory(absolute.parent, context=f"{context} parent")
    if absolute.parent != parent or absolute.name in {"", ".", ".."}:
        _fail(f"{context} must have one canonical basename")
    try:
        os.lstat(absolute)
    except FileNotFoundError:
        return absolute
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    _fail(f"{context} already exists")


def _assert_distinct_roots(*paths: tuple[Path, str]) -> None:
    for index, (left, left_name) in enumerate(paths):
        for right, right_name in paths[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                _fail(f"{left_name} and {right_name} must be distinct, unnested roots")


def _pin_pdf_root(pdf_root: Path) -> tuple[object, dict[str, object]]:
    absolute = _canonical_existing_directory(pdf_root, context="PDF root")
    root = _pin_root(absolute, context="PDF root")
    try:
        _validate_exact_root_inventory(
            root,
            expected=PDF_MEMBERS,
            context="PDF root",
            maximum_each=MAX_PDF_BYTES,
            maximum_total=MAX_TOTAL_PDF_BYTES,
        )
        identity = os.fstat(root.descriptor)
        return root, {
            "absolute_path": str(absolute),
            "device": identity.st_dev,
            "inode": identity.st_ino,
            "mode": format(stat.S_IMODE(identity.st_mode), "04o"),
            "mtime_ns": identity.st_mtime_ns,
            "members": list(PDF_MEMBERS),
        }
    except BaseException:
        root.close()
        raise


def _validate_pdf_sha256_anchors(
    expected_pdf_sha256: Mapping[str, str],
) -> dict[str, str]:
    if set(expected_pdf_sha256) != set(PDF_IDS):
        _fail(f"PDF SHA-256 anchors must name exactly {list(PDF_IDS)}")
    return {
        pdf_id: _expect_sha256(
            expected_pdf_sha256[pdf_id],
            context=f"expected PDF {pdf_id} SHA-256",
        )
        for pdf_id in PDF_IDS
    }


def _pin_pdfs(
    root: object,
    *,
    expected_pdf_sha256: Mapping[str, str],
) -> dict[str, object]:
    anchors = _validate_pdf_sha256_anchors(expected_pdf_sha256)
    pins: dict[str, object] = {}
    total = 0
    try:
        for pdf_id, member in PDF_ORDER:
            expected = os.stat(
                member,
                dir_fd=root.descriptor,
                follow_symlinks=False,
            )
            pin = _open_root_member(
                root,
                member,
                maximum=MAX_PDF_BYTES,
                expected_size=expected.st_size,
                context=f"PDF {pdf_id}",
            )
            prefix = os.pread(pin.descriptor, 5, 0)
            tail = os.pread(
                pin.descriptor,
                min(1024, pin.size),
                max(pin.size - 1024, 0),
            )
            if prefix != b"%PDF-" or b"%%EOF" not in tail:
                pin.close()
                _fail(f"PDF {pdf_id} lacks the bounded PDF signature/EOF marker")
            if pin.sha256 != anchors[pdf_id]:
                pin.close()
                _fail(
                    f"PDF {pdf_id} does not match its caller-supplied SHA-256 "
                    "authority anchor",
                )
            total += pin.size
            if total > MAX_TOTAL_PDF_BYTES:
                pin.close()
                _fail(
                    f"PDF bytes exceed the {MAX_TOTAL_PDF_BYTES}-byte aggregate limit",
                )
            pins[pdf_id] = pin
        if len({pin.sha256 for pin in pins.values()}) != len(PDF_ORDER):
            _fail("the four PDF roles must contain byte-distinct documents")
        return pins
    except BaseException:
        for pin in pins.values():
            pin.close()
        raise


def _pin_dependency(path: Path, *, context: str) -> _PinnedFile:
    return _pin_file(
        path,
        maximum=4 * 1024 * 1024,
        context=context,
    )


def _validate_tool_sha256_anchors(
    expected_tool_sha256: Mapping[str, str],
) -> dict[str, str]:
    if set(expected_tool_sha256) != set(TOOL_ORDER):
        _fail(f"tool SHA-256 anchors must name exactly {list(TOOL_ORDER)}")
    return {
        name: _expect_sha256(
            expected_tool_sha256[name],
            context=f"expected Poppler tool {name} SHA-256",
        )
        for name in TOOL_ORDER
    }


def _pin_tools(
    tool_paths: Mapping[str, Path],
    *,
    expected_tool_sha256: Mapping[str, str],
) -> dict[str, _PinnedFile]:
    if set(tool_paths) != set(TOOL_ORDER):
        _fail(f"tool paths must name exactly {list(TOOL_ORDER)}")
    anchors = _validate_tool_sha256_anchors(expected_tool_sha256)
    pins: dict[str, _PinnedFile] = {}
    try:
        for name in TOOL_ORDER:
            canonical = tool_paths[name].absolute()
            if not tool_paths[name].is_absolute():
                _fail(f"Poppler tool {name} must be supplied as an absolute path")
            pin = _pin_file(
                canonical,
                maximum=16 * 1024 * 1024,
                context=f"Poppler tool {name}",
            )
            mode = stat.S_IMODE(os.fstat(pin.descriptor).st_mode)
            if mode & 0o111 == 0:
                pin.close()
                _fail(f"installed Poppler tool {name} is not executable")
            if pin.sha256 != anchors[name]:
                pin.close()
                _fail(
                    f"installed Poppler tool {name} does not match its "
                    "caller-supplied SHA-256 trust anchor",
                )
            pins[name] = pin
        return pins
    except BaseException:
        for pin in pins.values():
            pin.close()
        raise


def _revalidate_inputs(
    root: object,
    pdfs: Mapping[str, object],
    tools: Mapping[str, object],
    builder: _PinnedFile,
) -> None:
    _revalidate_root(root, context="PDF root")
    _validate_exact_root_inventory(
        root,
        expected=PDF_MEMBERS,
        context="PDF root",
        maximum_each=MAX_PDF_BYTES,
        maximum_total=MAX_TOTAL_PDF_BYTES,
    )
    for pdf_id in PDF_IDS:
        _revalidate_file(pdfs[pdf_id], context=f"PDF {pdf_id}")
    for tool_name in TOOL_ORDER:
        _revalidate_file(
            tools[tool_name],
            context=f"Poppler tool {tool_name}",
        )
    _revalidate_file(builder, context="machine closure builder")


def _pinned_bytes(pin: _PinnedFile, *, context: str) -> bytes:
    raw = os.pread(pin.descriptor, pin.size, 0)
    if len(raw) != pin.size or _sha256(raw) != pin.sha256:
        _fail(f"{context} bytes changed while deriving native provenance")
    return raw


def _parse_arm64_code_directory(pin: _PinnedFile) -> dict[str, object]:
    """Parse the one native arm64 CodeDirectory from already pinned bytes."""
    raw = _pinned_bytes(pin, context="Poppler executable")
    if len(raw) < 32:
        _fail("Poppler executable lacks a complete Mach-O header")
    try:
        (
            magic,
            cpu_type,
            _cpu_subtype,
            file_type,
            command_count,
            command_bytes,
            _flags,
            _reserved,
        ) = struct.unpack_from("<IiiIIIII", raw)
    except struct.error as error:  # pragma: no cover - guarded by size
        _fail(f"cannot parse Poppler Mach-O header: {error}")
    if magic != MH_MAGIC_64 or cpu_type != CPU_TYPE_ARM64 or file_type != MH_EXECUTE:
        _fail("Poppler executable must be a thin native arm64 Mach-O executable")
    if not 1 <= command_count <= MAX_MACH_LOAD_COMMANDS:
        _fail("Poppler Mach-O load-command count is out of bounds")
    command_end = 32 + command_bytes
    if command_end > len(raw) or command_bytes < command_count * 8:
        _fail("Poppler Mach-O load-command extent is invalid")
    code_signature: tuple[int, int] | None = None
    offset = 32
    for _ in range(command_count):
        if offset + 8 > command_end:
            _fail("Poppler Mach-O load-command table is truncated")
        command, command_size = struct.unpack_from("<II", raw, offset)
        if command_size < 8 or command_size % 8 or offset + command_size > command_end:
            _fail("Poppler Mach-O contains an invalid load command")
        if command == LC_CODE_SIGNATURE:
            if code_signature is not None or command_size != 16:
                _fail("Poppler Mach-O must contain one canonical code signature")
            data_offset, data_size = struct.unpack_from("<II", raw, offset + 8)
            code_signature = (data_offset, data_size)
        offset += command_size
    if offset != command_end or code_signature is None:
        _fail("Poppler Mach-O load commands or code signature are incomplete")
    data_offset, data_size = code_signature
    if (
        data_size < 20
        or data_offset < command_end
        or data_offset + data_size > len(raw)
    ):
        _fail("Poppler Mach-O code-signature extent is invalid")
    signature = raw[data_offset : data_offset + data_size]
    magic, signature_length, blob_count = struct.unpack_from(">III", signature)
    if magic != CSMAGIC_EMBEDDED_SIGNATURE:
        _fail("Poppler executable lacks an embedded-signature SuperBlob")
    if (
        signature_length < 12
        or signature_length > len(signature)
        or any(signature[signature_length:])
        or not 1 <= blob_count <= MAX_CODE_SIGNATURE_BLOBS
        or 12 + (blob_count * 8) > signature_length
    ):
        _fail("Poppler embedded-signature SuperBlob is malformed")
    code_directories: list[bytes] = []
    seen_slots: set[int] = set()
    index_end = 12 + (blob_count * 8)
    for index in range(blob_count):
        slot, blob_offset = struct.unpack_from(">II", signature, 12 + (index * 8))
        if slot in seen_slots:
            _fail("Poppler embedded signature contains duplicate blob slots")
        seen_slots.add(slot)
        if blob_offset < index_end or blob_offset + 8 > signature_length:
            _fail("Poppler embedded signature contains an invalid blob offset")
        blob_magic, blob_length = struct.unpack_from(">II", signature, blob_offset)
        if blob_length < 8 or blob_offset + blob_length > signature_length:
            _fail("Poppler embedded signature contains an invalid blob extent")
        if blob_magic == CSMAGIC_CODEDIRECTORY:
            if slot != CSSLOT_CODEDIRECTORY:
                _fail("alternate Poppler CodeDirectories are outside this contract")
            code_directories.append(signature[blob_offset : blob_offset + blob_length])
    if len(code_directories) != 1:
        _fail("Poppler embedded signature must contain one primary CodeDirectory")
    code_directory = code_directories[0]
    if len(code_directory) < 44:
        _fail("Poppler CodeDirectory header is truncated")
    hash_size = code_directory[36]
    hash_type = code_directory[37]
    algorithms: dict[int, tuple[str, int, Callable[[bytes], object]]] = {
        1: ("sha1", 20, hashlib.sha1),
        2: ("sha256", 32, hashlib.sha256),
        3: ("sha256-truncated", 20, hashlib.sha256),
        4: ("sha384", 48, hashlib.sha384),
    }
    algorithm = algorithms.get(hash_type)
    if algorithm is None or hash_size != algorithm[1]:
        _fail("Poppler CodeDirectory uses an unsupported hash contract")
    digest = algorithm[2](code_directory).digest()[:CS_CDHASH_LEN]
    return {
        "binary_container": "thin-macho64",
        "architecture": "arm64",
        "hash_type": algorithm[0],
        "code_directory_bytes": len(code_directory),
        "cdhash": digest.hex(),
    }


def _assert_darwin_proc_layout() -> None:
    layout = {
        "proc_regioninfo_size": ctypes.sizeof(_ProcRegionInfo),
        "proc_region_offset": _ProcRegionInfo.pri_offset.offset,
        "proc_region_address": _ProcRegionInfo.pri_address.offset,
        "proc_region_size": _ProcRegionInfo.pri_size.offset,
        "vinfo_stat_size": ctypes.sizeof(_VinfoStat),
        "vinfo_dev": _VinfoStat.vst_dev.offset,
        "vinfo_mode": _VinfoStat.vst_mode.offset,
        "vinfo_nlink": _VinfoStat.vst_nlink.offset,
        "vinfo_inode": _VinfoStat.vst_ino.offset,
        "vnode_info_size": ctypes.sizeof(_VnodeInfo),
        "vnode_path_size": ctypes.sizeof(_VnodeInfoPath),
        "vnode_path_offset": _VnodeInfoPath.vip_path.offset,
        "region_with_path_size": ctypes.sizeof(_ProcRegionWithPathInfo),
        "region_vnode_offset": _ProcRegionWithPathInfo.prp_vip.offset,
    }
    expected = {
        "proc_regioninfo_size": 96,
        "proc_region_offset": 16,
        "proc_region_address": 80,
        "proc_region_size": 88,
        "vinfo_stat_size": 136,
        "vinfo_dev": 0,
        "vinfo_mode": 4,
        "vinfo_nlink": 6,
        "vinfo_inode": 8,
        "vnode_info_size": 152,
        "vnode_path_size": 1176,
        "vnode_path_offset": 152,
        "region_with_path_size": PROC_REGIONWITHPATHINFO_SIZE,
        "region_vnode_offset": PROC_REGIONINFO_SIZE,
    }
    if layout != expected:
        _fail(f"Darwin proc-region ABI layout is unsupported: {layout}")


def _darwin_library() -> ctypes.CDLL:
    if sys.platform != "darwin":
        _fail("native execution attestation requires Darwin")
    if os.uname().machine != "arm64":
        _fail("native execution attestation requires an arm64 Darwin host")
    _assert_darwin_proc_layout()
    return ctypes.CDLL(None, use_errno=True)


def _spawn_call(status: int, *, context: str) -> None:
    if status != 0:
        _fail(f"{context}: {os.strerror(status)} (errno {status})")


def _configure_spawn_library(library: ctypes.CDLL) -> None:
    pointer = ctypes.POINTER(ctypes.c_void_p)
    library.posix_spawnattr_init.argtypes = [pointer]
    library.posix_spawnattr_init.restype = ctypes.c_int
    library.posix_spawnattr_setflags.argtypes = [pointer, ctypes.c_short]
    library.posix_spawnattr_setflags.restype = ctypes.c_int
    library.posix_spawnattr_destroy.argtypes = [pointer]
    library.posix_spawnattr_destroy.restype = ctypes.c_int
    library.posix_spawn_file_actions_init.argtypes = [pointer]
    library.posix_spawn_file_actions_init.restype = ctypes.c_int
    library.posix_spawn_file_actions_destroy.argtypes = [pointer]
    library.posix_spawn_file_actions_destroy.restype = ctypes.c_int
    library.posix_spawn_file_actions_addopen.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint16,
    ]
    library.posix_spawn_file_actions_addopen.restype = ctypes.c_int
    library.posix_spawn_file_actions_adddup2.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.posix_spawn_file_actions_adddup2.restype = ctypes.c_int
    library.posix_spawn_file_actions_addchdir_np.argtypes = [
        pointer,
        ctypes.c_char_p,
    ]
    library.posix_spawn_file_actions_addchdir_np.restype = ctypes.c_int
    library.posix_spawn_file_actions_addinherit_np.argtypes = [
        pointer,
        ctypes.c_int,
    ]
    library.posix_spawn_file_actions_addinherit_np.restype = ctypes.c_int
    library.posix_spawn.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.posix_spawn.restype = ctypes.c_int


def _destroy_spawn_state(
    library: ctypes.CDLL,
    actions: ctypes.c_void_p,
    attribute: ctypes.c_void_p,
    *,
    actions_ready: bool,
    attribute_ready: bool,
) -> list[str]:
    errors: list[str] = []
    operations = (
        (
            "file actions",
            actions_ready,
            library.posix_spawn_file_actions_destroy,
            ctypes.byref(actions),
        ),
        (
            "attributes",
            attribute_ready,
            library.posix_spawnattr_destroy,
            ctypes.byref(attribute),
        ),
    )
    for label, ready, operation, pointer in operations:
        if not ready:
            continue
        try:
            status = operation(pointer)
        except BaseException as error:  # noqa: BLE001 - attempt both cleanups.
            errors.append(f"{label}: {error}")
            continue
        if status:
            errors.append(f"{label}: {os.strerror(status)} (errno {status})")
    return errors


def _spawn_suspended_darwin(
    tool: _PinnedFile,
    arguments: Sequence[str],
    *,
    inherited_fds: Sequence[int],
    inherited_fd_binding: tuple[int, int] | None = None,
    stdout_descriptor: int,
    stderr_descriptor: int,
) -> int:
    """Spawn one new Darwin session stopped before its first user instruction."""
    if not tool.path.is_absolute():
        _fail("Poppler executable path must be absolute")
    inherited = tuple(inherited_fds)
    bindings = () if inherited_fd_binding is None else (inherited_fd_binding,)
    binding_sources = tuple(source for source, _target in bindings)
    binding_targets = tuple(target for _source, target in bindings)
    output_descriptors = {stdout_descriptor, stderr_descriptor}
    if (
        len(inherited) != len(set(inherited))
        or any(descriptor <= 2 for descriptor in inherited)
        or len(binding_sources) != len(set(binding_sources))
        or len(binding_targets) != len(set(binding_targets))
        or any(source <= 2 or target <= 2 for source, target in bindings)
        or set(inherited) & set(binding_sources)
        or set(inherited) & set(binding_targets)
        or output_descriptors & set(inherited)
        or output_descriptors & set(binding_sources)
        or output_descriptors & set(binding_targets)
        or stdout_descriptor == stderr_descriptor
        or stdout_descriptor <= 2
        or stderr_descriptor <= 2
    ):
        _fail("spawn descriptor contract is ambiguous")
    for descriptor in (
        *inherited,
        *binding_sources,
        stdout_descriptor,
        stderr_descriptor,
    ):
        try:
            os.fstat(descriptor)
        except OSError as error:  # noqa: PERF203 - exact descriptor diagnostics.
            _fail(f"spawn descriptor is not open: {error}")
    encoded_argv = [os.fsencode(tool.path)]
    for argument in arguments:
        if "\x00" in argument:
            _fail("child argument contains NUL")
        encoded_argv.append(os.fsencode(argument))
    encoded_environment = [
        f"{key}={value}".encode("ascii") for key, value in EXACT_ENVIRONMENT.items()
    ]
    argv = (ctypes.c_char_p * (len(encoded_argv) + 1))(*encoded_argv, None)
    environment = (ctypes.c_char_p * (len(encoded_environment) + 1))(
        *encoded_environment,
        None,
    )
    library = _darwin_library()
    _configure_spawn_library(library)
    attribute = ctypes.c_void_p()
    actions = ctypes.c_void_p()
    attribute_ready = False
    actions_ready = False
    process_id = 0
    spawned = ctypes.c_int()
    destruction_errors: list[str] = []
    try:
        _spawn_call(
            library.posix_spawnattr_init(ctypes.byref(attribute)),
            context="cannot initialize Darwin spawn attributes",
        )
        attribute_ready = True
        _spawn_call(
            library.posix_spawnattr_setflags(
                ctypes.byref(attribute),
                ctypes.c_short(DARWIN_SPAWN_FLAGS),
            ),
            context="cannot set fail-closed Darwin spawn flags",
        )
        _spawn_call(
            library.posix_spawn_file_actions_init(ctypes.byref(actions)),
            context="cannot initialize Darwin spawn file actions",
        )
        actions_ready = True
        _spawn_call(
            library.posix_spawn_file_actions_addopen(
                ctypes.byref(actions),
                0,
                ctypes.c_char_p(b"/dev/null"),
                os.O_RDONLY,
                0,
            ),
            context="cannot bind child stdin",
        )
        for source, target in ((stdout_descriptor, 1), (stderr_descriptor, 2)):
            _spawn_call(
                library.posix_spawn_file_actions_adddup2(
                    ctypes.byref(actions),
                    source,
                    target,
                ),
                context=f"cannot bind child descriptor {target}",
            )
        _spawn_call(
            library.posix_spawn_file_actions_addchdir_np(
                ctypes.byref(actions),
                ctypes.c_char_p(b"/"),
            ),
            context="cannot bind child working directory",
        )
        for descriptor in inherited:
            _spawn_call(
                library.posix_spawn_file_actions_addinherit_np(
                    ctypes.byref(actions),
                    descriptor,
                ),
                context="cannot inherit an exact child descriptor",
            )
        for source, target in bindings:
            if source == target:
                status = library.posix_spawn_file_actions_addinherit_np(
                    ctypes.byref(actions),
                    source,
                )
            else:
                status = library.posix_spawn_file_actions_adddup2(
                    ctypes.byref(actions),
                    source,
                    target,
                )
            _spawn_call(
                status,
                context="cannot bind an exact child descriptor number",
            )
        _spawn_call(
            library.posix_spawn(
                ctypes.byref(spawned),
                ctypes.c_char_p(encoded_argv[0]),
                ctypes.byref(actions),
                ctypes.byref(attribute),
                argv,
                environment,
            ),
            context="Darwin suspended spawn failed",
        )
        process_id = spawned.value
        if process_id <= 0:
            _fail("Darwin suspended spawn returned an invalid process id")
    finally:
        primary_error = sys.exception()
        if process_id <= 0 and spawned.value > 0:
            process_id = spawned.value
        destruction_errors.extend(
            _destroy_spawn_state(
                library,
                actions,
                attribute,
                actions_ready=actions_ready,
                attribute_ready=attribute_ready,
            ),
        )
        child_cleanup_error: BaseException | None = None
        if process_id > 0 and (primary_error is not None or destruction_errors):
            try:
                _kill_and_reap(process_id)
            except BaseException as error:  # noqa: BLE001 - unknown-child safety.
                child_cleanup_error = error
        if destruction_errors or child_cleanup_error is not None:
            details = [*destruction_errors]
            if child_cleanup_error is not None:
                details.append(f"child cleanup: {child_cleanup_error}")
            if primary_error is not None:
                details.insert(0, f"primary error: {primary_error}")
            _fail(
                "Darwin spawn state cleanup failed: " + "; ".join(details),
            )
    return process_id


def _waitid_unreaped(
    process_id: int,
    *,
    options: int,
    deadline: float,
    context: str,
) -> tuple[int, int]:
    """Observe one child state while retaining its PID against reuse."""
    if sys.platform != "darwin":
        _fail("unreaped child-state observation requires Darwin")
    layout = {
        "size": ctypes.sizeof(_DarwinSigInfo),
        "pid_offset": _DarwinSigInfo.si_pid.offset,
        "status_offset": _DarwinSigInfo.si_status.offset,
    }
    if layout != {"size": 104, "pid_offset": 12, "status_offset": 20}:
        _fail(f"Darwin siginfo ABI layout is unsupported: {layout}")
    library = _darwin_library()
    library.waitid.argtypes = [
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_DarwinSigInfo),
        ctypes.c_int,
    ]
    library.waitid.restype = ctypes.c_int
    while True:
        info = _DarwinSigInfo()
        ctypes.set_errno(0)
        result = library.waitid(
            DARWIN_P_PID,
            process_id,
            ctypes.byref(info),
            options | DARWIN_WNOHANG | DARWIN_WNOWAIT,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EINTR:
                continue
            _fail(f"{context}: {os.strerror(error_number)} (errno {error_number})")
        if info.si_pid == process_id:
            if info.si_signo != signal.SIGCHLD:
                _fail(f"{context}: waitid returned a non-SIGCHLD event")
            return info.si_code, info.si_status
        if info.si_pid != 0:
            _fail(f"{context}: waitid returned an unrelated process")
        if time.monotonic() >= deadline:
            _fail(f"{context}: timed out")
        time.sleep(0.005)


def _wait_for_suspension(process_id: int, *, deadline: float) -> tuple[int, bool]:
    code, status = _waitid_unreaped(
        process_id,
        options=DARWIN_WSTOPPED | DARWIN_WEXITED,
        deadline=deadline,
        context="cannot verify suspended child state",
    )
    if code == DARWIN_CLD_STOPPED and status == 0:
        return 0x7F, False
    if code in {DARWIN_CLD_EXITED, DARWIN_CLD_KILLED, DARWIN_CLD_DUMPED}:
        return status, True
    _fail(f"child was not suspended before attestation (code {code}, status {status})")


def _observe_terminal_unreaped(process_id: int, *, deadline: float) -> None:
    code, status = _waitid_unreaped(
        process_id,
        options=DARWIN_WEXITED,
        deadline=deadline,
        context="cannot observe bounded child terminal state without reaping",
    )
    if code not in {DARWIN_CLD_EXITED, DARWIN_CLD_KILLED, DARWIN_CLD_DUMPED}:
        _fail(f"bounded child terminal waitid event is invalid ({code}, {status})")


def _main_executable_mapping(process_id: int, tool: _PinnedFile) -> dict[str, object]:
    library = _darwin_library()
    library.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    address = 0
    for _ in range(64):
        buffer = (ctypes.c_ubyte * PROC_REGIONWITHPATHINFO_SIZE)()
        ctypes.set_errno(0)
        returned = library.proc_pidinfo(
            process_id,
            PROC_PIDREGIONPATHINFO,
            ctypes.c_uint64(address),
            ctypes.byref(buffer),
            PROC_REGIONWITHPATHINFO_SIZE,
        )
        if returned != PROC_REGIONWITHPATHINFO_SIZE:
            error_number = ctypes.get_errno()
            detail = os.strerror(error_number) if error_number else "short response"
            _fail(f"cannot attest child executable mapping: {detail}")
        raw = bytes(buffer)
        protection = struct.unpack_from("<I", raw, 0)[0]
        region_file_offset = struct.unpack_from("<Q", raw, 16)[0]
        region_address = struct.unpack_from("<Q", raw, 80)[0]
        region_size = struct.unpack_from("<Q", raw, 88)[0]
        device = struct.unpack_from("<I", raw, PROC_REGIONINFO_SIZE)[0]
        mode = struct.unpack_from("<H", raw, PROC_REGIONINFO_SIZE + 4)[0]
        link_count = struct.unpack_from("<H", raw, PROC_REGIONINFO_SIZE + 6)[0]
        inode = struct.unpack_from("<Q", raw, PROC_REGIONINFO_SIZE + 8)[0]
        path_raw = raw[VNODE_PATH_OFFSET : VNODE_PATH_OFFSET + MAXPATHLEN]
        path_bytes = path_raw.split(b"\x00", 1)[0]
        if (device, inode) == (
            tool.device & 0xFFFFFFFF,
            tool.inode,
        ) and protection & VM_PROT_EXECUTE:
            if not stat.S_ISREG(mode) or link_count != 1 or not path_bytes:
                _fail("child main executable mapping has invalid vnode metadata")
            mapping_path = os.fsdecode(path_bytes)
            return {
                "device": device,
                "inode": inode,
                "path": mapping_path,
                "mode": format(stat.S_IMODE(mode), "04o"),
                "link_count": link_count,
                "protection": protection,
                "file_offset": region_file_offset,
            }
        if region_size == 0 or region_address + region_size <= address:
            _fail("child executable region walk made no progress")
        address = region_address + region_size
    _fail("child main executable mapping does not match the pinned tool vnode")


def _csops_bytes(process_id: int, operation: int, size: int, *, context: str) -> bytes:
    library = _darwin_library()
    library.csops.argtypes = [
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.csops.restype = ctypes.c_int
    buffer = (ctypes.c_ubyte * size)()
    ctypes.set_errno(0)
    result = library.csops(process_id, operation, ctypes.byref(buffer), size)
    if result != 0:
        error_number = ctypes.get_errno()
        _fail(f"{context}: {os.strerror(error_number)} (errno {error_number})")
    return bytes(buffer)


def _attest_suspended_process(
    process_id: int,
    tool: _PinnedFile,
    expected_code_directory: Mapping[str, object],
    *,
    suspended_wait_status: int,
) -> dict[str, object]:
    mapping = _main_executable_mapping(process_id, tool)
    status_raw = _csops_bytes(
        process_id,
        CS_OPS_STATUS,
        4,
        context="cannot read child code-signing status",
    )
    code_signing_status = int.from_bytes(status_raw, sys.byteorder)
    observed_cdhash = _csops_bytes(
        process_id,
        CS_OPS_CDHASH,
        CS_CDHASH_LEN,
        context="cannot read child CodeDirectory hash",
    ).hex()
    expected_cdhash = str(expected_code_directory["cdhash"])
    if observed_cdhash != expected_cdhash:
        _fail("child CodeDirectory hash does not match the pinned tool bytes")
    if code_signing_status & REQUIRED_CS_FLAGS != REQUIRED_CS_FLAGS:
        _fail("child code-signing status lacks required validity/signature/kill flags")
    if code_signing_status & REJECTED_CS_FLAGS:
        _fail("child code-signing status contains a rejected fail-open flag")
    return {
        "protocol": "darwin-posix-spawn-suspended-main-executable-v1",
        "spawn_flags": DARWIN_SPAWN_FLAGS,
        "suspended_wait_status": suspended_wait_status,
        "code_signing_status": code_signing_status,
        "expected_code_directory": dict(expected_code_directory),
        "observed_cdhash": observed_cdhash,
        "main_executable_mapping": mapping,
        "execution_binding_scope": "main_executable",
        "non_system_dylib_closure": "not_attested",
        "same_vnode_mutation_fail_stop_assumption": (
            "invalid-signed-code-page-triggers-darwin-cs-kill"
        ),
        "other_same_vnode_mutations": "not_attested",
    }


def _reap_process(process_id: int, *, deadline: float) -> int:
    while True:
        try:
            waited, status = os.waitpid(process_id, os.WNOHANG)
        except ChildProcessError as error:
            _fail(f"cannot reap bounded child: {error}")
        except OSError as error:
            _fail(f"cannot reap bounded child: {error}")
        if waited == process_id:
            return status
        if waited != 0:
            _fail("waitpid returned an unrelated process while reaping")
        if time.monotonic() >= deadline:
            _fail("bounded child did not terminate before the reap deadline")
        time.sleep(0.005)


def _terminate_owned_process_group(process_id: int) -> None:
    """Kill and drain a child-owned group while its leader PID cannot be reused."""
    process_group_signaled = False
    try:
        os.killpg(process_id, signal.SIGKILL)
        process_group_signaled = True
    except (PermissionError, ProcessLookupError):
        # Darwin may deny or report no signal target for a group whose only
        # member is the terminal zombie leader.  The WNOWAIT-held leader still
        # owns the PID/PGID, so an exact enumeration can distinguish that state
        # from an uncontained descendant without risking a reused identifier.
        process_group_signaled = False
    except OSError as error:
        _fail(f"cannot terminate bounded child process group: {error}")
    _require_process_group_leader_only(
        process_id,
        deadline=time.monotonic() + ATTESTATION_TIMEOUT_SECONDS,
        wait_for_signaled_members=process_group_signaled,
    )


def _darwin_process_group_members(process_id: int) -> tuple[int, ...]:
    if sys.platform != "darwin":
        _fail("process-group member enumeration requires Darwin")
    library = _darwin_library()
    library.proc_listpgrppids.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.proc_listpgrppids.restype = ctypes.c_int
    pids = (ctypes.c_int * MAX_PROCESS_GROUP_MEMBERS)()
    ctypes.set_errno(0)
    count = library.proc_listpgrppids(
        process_id,
        pids,
        ctypes.sizeof(pids),
    )
    if count < 0:
        error_number = ctypes.get_errno()
        _fail(
            "cannot enumerate bounded child process group: "
            f"{os.strerror(error_number)} (errno {error_number})",
        )
    if count >= MAX_PROCESS_GROUP_MEMBERS:
        _fail("bounded child process group exceeds its member bound")
    members = tuple(pids[:count])
    if any(member <= 0 for member in members) or len(set(members)) != len(members):
        _fail("bounded child process group contains invalid or duplicate PIDs")
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
            _fail(
                "bounded child process group is not the retained leader alone "
                f"(members {members})",
            )
        time.sleep(0.005)
    if _darwin_process_group_members(process_id) != (process_id,):
        _fail("bounded child process group membership changed before parent reap")


def _kill_and_reap(
    process_id: int,
    *,
    already_reaped: bool = False,
    group_terminated: bool = False,
) -> int | None:
    if already_reaped:
        if not group_terminated:
            _fail(
                "unsafe cleanup state: child was reaped before its owned process "
                "group was terminated",
            )
        return None
    termination_errors: list[str] = []
    if not group_terminated:
        for attempt in range(2):
            try:
                _terminate_owned_process_group(process_id)
                group_terminated = True
                break
            except MachineClosureError as error:
                termination_errors.append(str(error))
                if attempt == 0:
                    try:
                        os.kill(process_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as direct_error:
                        termination_errors.append(
                            f"cannot directly terminate bounded child: {direct_error}",
                        )
        if not group_terminated:
            _fail(
                "process-group termination failed twice; retained the leader "
                "unreaped to prevent PID/PGID reuse: " + "; ".join(termination_errors),
            )
    status = _reap_process(
        process_id,
        deadline=time.monotonic() + ATTESTATION_TIMEOUT_SECONDS,
    )
    if termination_errors:
        _fail(
            "process-group termination failed before a successful contained retry: "
            + "; ".join(termination_errors),
        )
    return status


def _resume_process(process_id: int) -> None:
    try:
        os.killpg(process_id, signal.SIGCONT)
    except OSError as error:
        _fail(f"cannot resume attested child process group: {error}")


def _wait_return_code(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    _fail(f"bounded child returned a nonterminal wait status {status}")


def _run_bounded(
    tool: _PinnedFile,
    arguments: Sequence[str],
    *,
    inherited_fds: Sequence[int] = (),
    inherited_fd_binding: tuple[int, int] | None = None,
    timeout: float,
    stdout_limit: int,
    stderr_limit: int,
    budget: _ProcessBudget,
    before: Callable[[], None],
    after: Callable[[], None],
) -> tuple[int, bytes, bytes, dict[str, object]]:
    budget.consume()
    deadline = time.monotonic() + timeout
    before_completed = False
    process_id: int | None = None
    process_reaped = False
    process_group_terminated = False
    completed = False
    stdout_read = stdout_write = stderr_read = stderr_write = -1
    selector = selectors.DefaultSelector()
    streams: dict[int, tuple[str, bytearray, int]] = {}
    try:
        before()
        before_completed = True
        expected_code_directory = _parse_arm64_code_directory(tool)
        # A second full revalidation closes the parse-to-spawn mutation window.
        # CS_KILL fail-stops later mutations only when they invalidate a signed
        # executable code page; other same-vnode mutations remain unattested.
        before()
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        process_id = _spawn_suspended_darwin(
            tool,
            arguments,
            inherited_fds=inherited_fds,
            inherited_fd_binding=inherited_fd_binding,
            stdout_descriptor=stdout_write,
            stderr_descriptor=stderr_write,
        )
        os.close(stdout_write)
        stdout_write = -1
        os.close(stderr_write)
        stderr_write = -1
        suspended_status, terminal_observed = _wait_for_suspension(
            process_id,
            deadline=deadline,
        )
        if terminal_observed:
            _fail(
                "child terminated before suspended attestation "
                f"(status {suspended_status})",
            )
        attestation = _attest_suspended_process(
            process_id,
            tool,
            expected_code_directory,
            suspended_wait_status=suspended_status,
        )
        _resume_process(process_id)
        for name, descriptor, limit in (
            ("stdout", stdout_read, stdout_limit),
            ("stderr", stderr_read, stderr_limit),
        ):
            os.set_blocking(descriptor, False)
            streams[descriptor] = (name, bytearray(), limit)
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail(f"child process exceeded the {timeout:g}-second timeout")
            events = selector.select(timeout=min(remaining, 0.05))
            for key, _ in events:
                descriptor = key.fd
                name, buffer, limit = streams[descriptor]
                try:
                    block = os.read(descriptor, READ_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(descriptor)
                    continue
                buffer.extend(block)
                if len(buffer) > limit:
                    _fail(f"child {name} exceeds the {limit}-byte limit")
        stdout = bytes(streams[stdout_read][1])
        stderr = bytes(streams[stderr_read][1])
        _observe_terminal_unreaped(process_id, deadline=deadline)
        _terminate_owned_process_group(process_id)
        process_group_terminated = True
        terminal_status = _reap_process(process_id, deadline=deadline)
        process_reaped = True
        return_code = _wait_return_code(terminal_status)
        before_completed = False
        after()
        completed = True
        return return_code, stdout, stderr, attestation
    finally:
        primary_error = sys.exception()
        cleanup_errors: list[BaseException] = []
        if process_id is not None and not completed:
            try:
                _kill_and_reap(
                    process_id,
                    already_reaped=process_reaped,
                    group_terminated=process_group_terminated,
                )
            except BaseException as error:  # noqa: BLE001 - cleanup ownership.
                cleanup_errors.append(error)
        try:
            selector.close()
        except BaseException as error:  # noqa: BLE001 - cleanup ownership.
            cleanup_errors.append(error)
            try:
                selector.close()
            except BaseException as retry_error:  # noqa: BLE001
                cleanup_errors.append(retry_error)
        for descriptor in (stdout_read, stdout_write, stderr_read, stderr_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException as error:  # noqa: BLE001 - cleanup ownership.
                    cleanup_errors.append(error)
                    try:
                        os.close(descriptor)
                    except OSError as retry_error:
                        if retry_error.errno != errno.EBADF:
                            cleanup_errors.append(retry_error)
                    except BaseException as retry_error:  # noqa: BLE001
                        cleanup_errors.append(retry_error)
        if before_completed:
            try:
                after()
            except BaseException as error:  # noqa: BLE001 - cleanup ownership.
                cleanup_errors.append(error)
        if cleanup_errors:
            cleanup_detail = "; ".join(str(error) for error in cleanup_errors)
            if primary_error is not None:
                _fail(
                    f"{primary_error}; bounded child cleanup also failed: "
                    f"{cleanup_detail}",
                )
            _fail(f"bounded child cleanup failed: {cleanup_detail}")


def _open_output_directory(
    root: _PinnedRoot,
    parts: Sequence[str],
    *,
    create: bool,
    guard: Callable[[], None],
) -> int:
    guard()
    descriptor = os.dup(root.descriptor)
    try:
        for part in parts:
            if part in {"", ".", ".."} or "/" in part or "\\" in part:
                _fail("output directory contains a noncanonical component")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as error:
                    _fail(f"cannot create output directory component {part}: {error}")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                _fail(f"cannot open output directory component {part}: {error}")
            entry = os.fstat(child)
            if not stat.S_ISDIR(entry.st_mode) or stat.S_IMODE(entry.st_mode) != 0o700:
                os.close(child)
                _fail(f"output directory component {part} has invalid identity")
            os.close(descriptor)
            descriptor = child
        guard()
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _mkdirs_under(
    root: _PinnedRoot,
    member_parent: PurePosixPath,
    *,
    guard: Callable[[], None],
) -> None:
    descriptor = _open_output_directory(
        root,
        member_parent.parts,
        create=True,
        guard=guard,
    )
    os.close(descriptor)


def _write_member(
    root: _PinnedRoot,
    member: str,
    raw: bytes,
    *,
    guard: Callable[[], None],
) -> dict[str, object]:
    canonical = _relative_member(member, context="output member")
    if len(raw) > MAX_MEMBER_BYTES:
        _fail(f"output member {canonical} exceeds the member byte limit")
    pure = PurePosixPath(canonical)
    existing_files, existing_directories, existing_bytes = _walk_output(
        root,
        directory_mode=0o700,
    )
    if canonical in existing_files:
        _fail(f"output member {canonical} already exists")
    if len(existing_files) + 1 > MAX_OUTPUT_FILES:
        _fail("output tree would exceed the file count bound")
    if existing_bytes + len(raw) > MAX_OUTPUT_BYTES:
        _fail("output tree would exceed the aggregate byte bound")
    required_directories = {
        PurePosixPath(*pure.parts[:index]).as_posix()
        for index in range(1, len(pure.parts))
    }
    if len(set(existing_directories) | required_directories) > MAX_OUTPUT_DIRECTORIES:
        _fail("output tree would exceed the directory count bound")
    parent_descriptor = _open_output_directory(
        root,
        pure.parent.parts,
        create=True,
        guard=guard,
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(
            pure.name,
            flags,
            0o400,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail(f"write made no progress for output member {canonical}")
            offset += written
        os.fsync(descriptor)
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o400
            or entry.st_size != len(raw)
        ):
            _fail(f"output member {canonical} has invalid identity")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    guard()
    return {"member": canonical, "bytes": len(raw), "sha256": _sha256(raw)}


def _pin_output_member(
    root: _PinnedRoot,
    member: str,
    *,
    maximum: int,
) -> _PinnedFile:
    return _open_root_member(
        root,
        member,
        maximum=maximum,
        context=f"output member {member}",
        expected_size=None,
    )


def _normalise_argv(argv: Sequence[str], output_root: Path) -> list[str]:
    root = str(output_root)
    normalized: list[str] = []
    for token in argv:
        if token == root:
            normalized.append("{closure-root}")
        elif token.startswith(f"{root}/"):
            normalized.append(f"{{closure-root}}/{token[len(root) + 1 :]}")
        else:
            normalized.append(token)
    return normalized


def _invoke_and_record(
    *,
    name: str,
    tool_pin: _PinnedFile,
    arguments: Sequence[str],
    display_argv: Sequence[str],
    inherited_fds: Sequence[int],
    output_root: _PinnedRoot,
    raw_prefix: str,
    timeout: float,
    budget: _ProcessBudget,
    revalidate: Callable[[], None],
    output_guard: Callable[[], None],
    stdout_limit: int = MAX_RAW_STREAM_BYTES,
) -> tuple[bytes, bytes, dict[str, object], list[dict[str, object]]]:
    return_code, stdout, stderr, attestation = _run_bounded(
        tool_pin,
        arguments,
        inherited_fds=inherited_fds,
        timeout=timeout,
        stdout_limit=stdout_limit,
        stderr_limit=MAX_STDERR_BYTES,
        budget=budget,
        before=revalidate,
        after=revalidate,
    )
    stdout_member = f"{raw_prefix}.stdout"
    stderr_member = f"{raw_prefix}.stderr"
    output_members = [
        _write_member(output_root, stdout_member, stdout, guard=output_guard),
        _write_member(output_root, stderr_member, stderr, guard=output_guard),
    ]
    invocation = {
        "name": name,
        "argv": _normalise_argv(display_argv, output_root.path),
        "executable_binding": attestation,
        "descriptor_bound_input_count": len(set(inherited_fds)),
        "cwd": "/",
        "shell": False,
        "environment": dict(EXACT_ENVIRONMENT),
        "timeout_seconds": int(timeout),
        "stdout_limit_bytes": stdout_limit,
        "stderr_limit_bytes": MAX_STDERR_BYTES,
        "return_code": return_code,
        "stdout": output_members[0],
        "stderr": output_members[1],
    }
    if return_code != 0:
        _fail(f"native invocation {name} failed with status {return_code}")
    return stdout, stderr, invocation, output_members


def _ascii(raw: bytes, *, context: str) -> str:
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"{context} is not bounded ASCII: {error}")


def _single_field(text: str, label: str, *, context: str) -> str:
    values = [
        line[len(label) :].strip()
        for line in text.splitlines()
        if line.startswith(label)
    ]
    if len(values) != 1 or not values[0]:
        _fail(f"{context} must contain exactly one {label.rstrip(':')} field")
    return values[0]


def _parse_summary(raw: bytes, *, context: str) -> dict[str, object]:
    text = _ascii(raw, context=context)
    pages_raw = _single_field(text, "Pages:", context=context)
    encrypted_raw = _single_field(text, "Encrypted:", context=context)
    version = _single_field(text, "PDF version:", context=context)
    if not pages_raw.isdecimal():
        _fail(f"{context} Pages must be a positive decimal integer")
    page_count = int(pages_raw)
    if not 1 <= page_count <= MAX_PAGES_PER_PDF:
        _fail(f"{context} page count exceeds the per-PDF bound")
    encrypted_token = encrypted_raw.split(maxsplit=1)[0]
    if encrypted_token not in {"yes", "no"}:
        _fail(f"{context} Encrypted must be yes or no")
    if re.fullmatch(r"(?:1\.[0-9]|2\.0)", version) is None:
        _fail(f"{context} has an invalid PDF version")
    return {
        "page_count": page_count,
        "pdf_version": version,
        "encrypted": encrypted_token == "yes",
    }


def _millipoint(value: str, *, context: str) -> int:
    try:
        scaled = Decimal(value) * 1000
    except InvalidOperation:
        _fail(f"{context} is not a decimal point coordinate")
    if scaled != scaled.to_integral_value():
        _fail(f"{context} exceeds millipoint precision")
    integer = int(scaled)
    if abs(integer) > 10_000_000:
        _fail(f"{context} exceeds the coordinate bound")
    return integer


def _parse_pages(
    raw: bytes,
    *,
    page_count: int,
    context: str,
) -> list[dict[str, object]]:
    text = _ascii(raw, context=context)
    records: dict[int, dict[str, object]] = {
        page: {} for page in range(1, page_count + 1)
    }
    for line in text.splitlines():
        size_match = PAGE_SIZE_RE.fullmatch(line.strip())
        rotation_match = PAGE_ROT_RE.fullmatch(line.strip())
        box_match = PAGE_BOX_RE.fullmatch(line.strip())
        match = size_match or rotation_match or box_match
        if match is None:
            continue
        page_token = match.group(1)
        if page_token is None:
            if page_count != 1:
                _fail(f"{context} contains an unnumbered multi-page geometry field")
            page = 1
        else:
            page = int(page_token)
        if page not in records:
            _fail(f"{context} contains geometry for out-of-range page {page}")
        record = records[page]
        if size_match is not None:
            key = "reported_size_millipoints"
            value: object = [
                _millipoint(
                    size_match.group(2),
                    context=f"{context} page {page} width",
                ),
                _millipoint(
                    size_match.group(3),
                    context=f"{context} page {page} height",
                ),
            ]
        elif rotation_match is not None:
            key = "rotation_degrees"
            value = int(rotation_match.group(2))
        else:
            if box_match is None:  # pragma: no cover - narrowed above
                _fail("unreachable page-box parser state")
            key = (
                "media_box_millipoints"
                if box_match.group(2) == "MediaBox"
                else "crop_box_millipoints"
            )
            value = [
                _millipoint(
                    box_match.group(index),
                    context=f"{context} page {page} box",
                )
                for index in range(3, 7)
            ]
        if key in record:
            _fail(f"{context} contains duplicate {key} for page {page}")
        record[key] = value
    expected = {
        "reported_size_millipoints",
        "rotation_degrees",
        "media_box_millipoints",
        "crop_box_millipoints",
    }
    normalized: list[dict[str, object]] = []
    for page, record in records.items():
        if set(record) != expected:
            _fail(f"{context} page {page} lacks complete size/box/rotation fields")
        crop = record["crop_box_millipoints"]
        if not isinstance(crop, list):  # pragma: no cover - construction invariant
            _fail("unreachable crop-box state")
        crop_width = crop[2] - crop[0]
        crop_height = crop[3] - crop[1]
        if crop_width <= 0 or crop_height <= 0:
            _fail(f"{context} page {page} has a non-positive crop box")
        if record["reported_size_millipoints"] != [crop_width, crop_height]:
            _fail(f"{context} page {page} reported size disagrees with crop box")
        normalized.append({"page": page, **record})
    return normalized


def _parse_fonts(raw: bytes, *, context: str) -> list[dict[str, object]]:
    text = _ascii(raw, context=context)
    fonts: list[dict[str, object]] = []
    for line in text.splitlines():
        match = FONT_RE.fullmatch(line.strip())
        if match is None:
            continue
        if len(fonts) >= MAX_FONTS_PER_PDF:
            _fail(f"{context} exceeds the {MAX_FONTS_PER_PDF}-font bound")
        fonts.append(
            {
                "name": match.group(1),
                "type": match.group(2),
                "encoding": match.group(3),
                "embedded": match.group(4) == "yes",
                "subset": match.group(5) == "yes",
                "unicode_map": match.group(6) == "yes",
                "object": int(match.group(7)),
                "generation": int(match.group(8)),
            },
        )
    if not fonts:
        _fail(f"{context} contains no parseable font inventory")
    return fonts


def _parse_images(raw: bytes, *, context: str) -> list[dict[str, object]]:
    text = _ascii(raw, context=context)
    if "page" not in text.lower() or "num" not in text.lower():
        _fail(f"{context} lacks the pdfimages inventory header")
    images: list[dict[str, object]] = []
    for row in text.splitlines():
        if IMAGE_ROW_RE.match(row) is None:
            continue
        if len(images) >= MAX_IMAGES_PER_PDF:
            _fail(f"{context} exceeds the {MAX_IMAGES_PER_PDF}-image bound")
        index = len(images)
        fields = row.strip().split()
        if len(fields) < 15:
            _fail(f"{context} image row {index} is incomplete")
        try:
            page = int(fields[0])
            number = int(fields[1])
            width = int(fields[3])
            height = int(fields[4])
        except ValueError:
            _fail(f"{context} image row {index} has invalid integer fields")
        if page <= 0 or number < 0 or width <= 0 or height <= 0:
            _fail(f"{context} image row {index} has out-of-range integer fields")
        if width > 100_000 or height > 100_000:
            _fail(f"{context} image row {index} exceeds the image geometry bound")
        if any(not field.isascii() or not field.isprintable() for field in fields):
            _fail(f"{context} image row {index} contains invalid field text")
        images.append(
            {
                "page": page,
                "number": number,
                "type": fields[2],
                "width_pixels": width,
                "height_pixels": height,
                "remaining_fields": fields[5:],
            },
        )
    return images


def _expected_render_dimensions(page: Mapping[str, object]) -> tuple[int, int]:
    crop = page["crop_box_millipoints"]
    if not isinstance(crop, list) or len(crop) != 4:
        _fail("page crop box is invalid while deriving render geometry")
    width_millipoints = int(crop[2]) - int(crop[0])
    height_millipoints = int(crop[3]) - int(crop[1])
    width = (width_millipoints * RENDER_DPI + 71_999) // 72_000
    height = (height_millipoints * RENDER_DPI + 71_999) // 72_000
    rotation = int(page["rotation_degrees"]) % 360
    if rotation in {90, 270}:
        return height, width
    return width, height


def _tool_record(name: str, pin: _PinnedFile, version: str) -> dict[str, object]:
    entry = os.fstat(pin.descriptor)
    return {
        "name": name,
        "absolute_path": str(pin.path),
        "device": entry.st_dev,
        "inode": entry.st_ino,
        "mode": format(stat.S_IMODE(entry.st_mode), "04o"),
        "bytes": pin.size,
        "sha256": pin.sha256,
        "version": version,
        "native_code_directory": _parse_arm64_code_directory(pin),
    }


def _pdf_record(pdf_id: str, member: str, pin: object) -> dict[str, object]:
    entry = os.fstat(pin.descriptor)
    return {
        "pdf_id": pdf_id,
        "member": member,
        "device": entry.st_dev,
        "inode": entry.st_ino,
        "mode": format(stat.S_IMODE(entry.st_mode), "04o"),
        "bytes": pin.size,
        "sha256": pin.sha256,
    }


def _render_page(
    *,
    run: str,
    pdf_id: str,
    page: int,
    pdf_pin: object,
    tool_pin: object,
    output_root: _PinnedRoot,
    budget: _ProcessBudget,
    revalidate: Callable[[], None],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    if run == "a":
        member = f"pages/{pdf_id}/page-{page:04d}.png"
    elif run == "b":
        member = f"reproductions/{pdf_id}/page-{page:04d}.png"
    else:  # pragma: no cover - internal invariant
        _fail("invalid render run")
    display_argv = [
        str(tool_pin.path),
        "-png",
        "-r",
        str(RENDER_DPI),
        "-cropbox",
        "-f",
        str(page),
        "-l",
        str(page),
        "-singlefile",
        str(pdf_pin.path),
    ]
    arguments = [*display_argv[1:-1], f"/dev/fd/{pdf_pin.descriptor}"]
    stdout, stderr, invocation, raw_members = _invoke_and_record(
        name=f"render-{run}-{pdf_id}-page-{page:04d}",
        tool_pin=tool_pin,
        arguments=arguments,
        display_argv=display_argv,
        inherited_fds=(pdf_pin.descriptor,),
        output_root=output_root,
        raw_prefix=f"raw/{pdf_id}/render-{run}-page-{page:04d}",
        timeout=RENDER_TIMEOUT_SECONDS,
        budget=budget,
        revalidate=revalidate,
        output_guard=revalidate,
        stdout_limit=MAX_PNG_BYTES,
    )
    if not stdout or stderr:
        _fail(
            f"pdftoppm emitted invalid streams for {pdf_id} page {page} run {run}",
        )
    if len(stdout) > MAX_PNG_BYTES:
        _fail(f"render member {member} violates the PNG byte bound")
    artifact_member = _write_member(
        output_root,
        member,
        stdout,
        guard=revalidate,
    )
    revalidate()
    pin = _pin_output_member(output_root, member, maximum=MAX_PNG_BYTES)
    try:
        width, height = _parse_png_dimensions(
            pin,
            context=f"render {run} {pdf_id} page {page}",
        )
        artifact = {
            "member": member,
            "bytes": pin.size,
            "sha256": pin.sha256,
            "width_pixels": width,
            "height_pixels": height,
            "pixels_per_meter": EXPECTED_PNG_PIXELS_PER_METER,
        }
    finally:
        pin.close()
    return artifact, invocation, [*raw_members, artifact_member]


def _derive_document(
    *,
    pdf_id: str,
    member: str,
    pdf_pin: object,
    tools: Mapping[str, object],
    output_root: _PinnedRoot,
    budget: _ProcessBudget,
    page_budget: _PageBudget,
    revalidate: Callable[[], None],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    invocations: list[dict[str, object]] = []
    members: list[dict[str, object]] = []

    def invoke(
        name: str,
        tool: str,
        args: Sequence[str],
        raw_name: str,
    ) -> bytes:
        display_argv = [str(tools[tool].path), *args, str(pdf_pin.path)]
        stdout, _stderr, invocation, raw_members = _invoke_and_record(
            name=name,
            tool_pin=tools[tool],
            arguments=[*args, f"/dev/fd/{pdf_pin.descriptor}"],
            display_argv=display_argv,
            inherited_fds=(pdf_pin.descriptor,),
            output_root=output_root,
            raw_prefix=f"raw/{pdf_id}/{raw_name}",
            timeout=TOOL_TIMEOUT_SECONDS,
            budget=budget,
            revalidate=revalidate,
            output_guard=revalidate,
        )
        invocations.append(invocation)
        members.extend(raw_members)
        return stdout

    summary = _parse_summary(
        invoke(f"pdfinfo-summary-{pdf_id}", "pdfinfo", [], "pdfinfo-summary"),
        context=f"pdfinfo summary {pdf_id}",
    )
    page_count = int(summary["page_count"])
    page_budget.consume(page_count)
    page_geometry = _parse_pages(
        invoke(
            f"pdfinfo-pages-{pdf_id}",
            "pdfinfo",
            ["-f", "1", "-l", str(page_count), "-box"],
            "pdfinfo-pages",
        ),
        page_count=page_count,
        context=f"pdfinfo page geometry {pdf_id}",
    )
    fonts = _parse_fonts(
        invoke(f"pdffonts-{pdf_id}", "pdffonts", [], "pdffonts"),
        context=f"pdffonts {pdf_id}",
    )
    images = _parse_images(
        invoke(f"pdfimages-{pdf_id}", "pdfimages", ["-list"], "pdfimages"),
        context=f"pdfimages {pdf_id}",
    )
    if bool(summary["encrypted"]):
        _fail(f"PDF {pdf_id} must not be encrypted")
    if any(int(page["rotation_degrees"]) != 0 for page in page_geometry):
        _fail(f"PDF {pdf_id} must have zero page rotation")
    if any(not bool(font["embedded"]) for font in fonts):
        _fail(f"PDF {pdf_id} must have all fonts embedded")
    if any(font["type"] == "Type 3" for font in fonts):
        _fail(f"PDF {pdf_id} must not contain Type 3 fonts")
    if any(int(image["page"]) > page_count for image in images):
        _fail(f"PDF {pdf_id} image inventory references an out-of-range page")
    if pdf_id in {"clean", "marked"} and images:
        _fail(f"PDF {pdf_id} must contain zero raster images")
    crop_sizes = {
        (
            int(page["crop_box_millipoints"][2]) - int(page["crop_box_millipoints"][0]),
            int(page["crop_box_millipoints"][3]) - int(page["crop_box_millipoints"][1]),
        )
        for page in page_geometry
    }
    if len(crop_sizes) != 1:
        _fail(f"PDF {pdf_id} must have one uniform crop-box geometry")
    is_letter = crop_sizes == {
        (US_LETTER_WIDTH_MILLIPOINTS, US_LETTER_HEIGHT_MILLIPOINTS),
    }
    if pdf_id in {"clean", "marked"} and not is_letter:
        _fail(f"PDF {pdf_id} must use US Letter crop boxes on every page")

    rendered_pages: list[dict[str, object]] = []
    for page_record in page_geometry:
        page = int(page_record["page"])
        first, first_invocation, first_members = _render_page(
            run="a",
            pdf_id=pdf_id,
            page=page,
            pdf_pin=pdf_pin,
            tool_pin=tools["pdftoppm"],
            output_root=output_root,
            budget=budget,
            revalidate=revalidate,
        )
        second, second_invocation, second_members = _render_page(
            run="b",
            pdf_id=pdf_id,
            page=page,
            pdf_pin=pdf_pin,
            tool_pin=tools["pdftoppm"],
            output_root=output_root,
            budget=budget,
            revalidate=revalidate,
        )
        invocations.extend((first_invocation, second_invocation))
        members.extend((*first_members, *second_members))
        if first["sha256"] != second["sha256"] or first["bytes"] != second["bytes"]:
            _fail(f"PDF {pdf_id} page {page} renders are not byte-identical")
        expected_width, expected_height = _expected_render_dimensions(page_record)
        if (first["width_pixels"], first["height_pixels"]) != (
            expected_width,
            expected_height,
        ):
            _fail(f"PDF {pdf_id} page {page} render geometry drifted from crop box")
        rendered_pages.append(
            {
                **page_record,
                "render": first,
                "reproduction": second,
                "byte_identical": True,
            },
        )
    checks = {
        "not_encrypted": True,
        "page_count_positive": True,
        "page_geometry_complete": True,
        "zero_rotation": True,
        "font_inventory_present": True,
        "all_fonts_embedded": True,
        "no_type_3_fonts": True,
        "zero_raster_images": not images,
        "raster_policy_pass": pdf_id not in {"clean", "marked"} or not images,
        "uniform_crop_box": True,
        "us_letter": is_letter,
        "native_png_integrity": True,
        "byte_identical_double_render": True,
    }
    return (
        {
            **_pdf_record(pdf_id, member, pdf_pin),
            **summary,
            "page_geometry": rendered_pages,
            "fonts": fonts,
            "raster_images": images,
            "raster_image_count": len(images),
            "checks": checks,
        },
        invocations,
        members,
    )


def _walk_output(
    root: _PinnedRoot,
    *,
    directory_mode: int,
) -> tuple[list[str], list[str], int]:
    files: list[str] = []
    directories: list[str] = []
    total_bytes = 0
    entry_count = 0

    def visit(descriptor: int, prefix: PurePosixPath, depth: int) -> None:
        nonlocal entry_count, total_bytes
        if depth > 3:
            _fail("output tree exceeds the three-component member depth")
        try:
            entries = os.scandir(descriptor)
        except OSError as error:
            _fail(f"cannot scan output tree: {error}")
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_OUTPUT_FILES + MAX_OUTPUT_DIRECTORIES:
                    _fail("output tree exceeds the bounded entry count")
                if entry.name in {"", ".", ".."} or "/" in entry.name:
                    _fail("output tree contains an invalid member name")
                member = prefix / entry.name
                relative = member.as_posix()
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as error:
                    _fail(f"cannot inspect output member {relative}: {error}")
                if stat.S_ISLNK(info.st_mode):
                    _fail(f"output member {relative} must not be a symlink")
                if stat.S_ISDIR(info.st_mode):
                    if stat.S_IMODE(info.st_mode) != directory_mode:
                        _fail(
                            f"output directory {relative} must have mode "
                            f"{directory_mode:04o}",
                        )
                    if len(directories) >= MAX_OUTPUT_DIRECTORIES:
                        _fail("output tree exceeds the directory count bound")
                    directories.append(relative)
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                    )
                    try:
                        child = os.open(entry.name, flags, dir_fd=descriptor)
                    except OSError as error:
                        _fail(f"cannot open output directory {relative}: {error}")
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (
                            info.st_dev,
                            info.st_ino,
                        ):
                            _fail(f"output directory {relative} changed during walk")
                        visit(child, member, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    if info.st_nlink != 1:
                        _fail(
                            f"output member {relative} must have exactly one hard link",
                        )
                    if stat.S_IMODE(info.st_mode) != 0o400:
                        _fail(f"output member {relative} must have mode 0400")
                    if info.st_size > MAX_MEMBER_BYTES:
                        _fail(f"output member {relative} exceeds the file byte bound")
                    total_bytes += info.st_size
                    if total_bytes > MAX_OUTPUT_BYTES:
                        _fail("output tree exceeds the aggregate byte bound")
                    if len(files) >= MAX_OUTPUT_FILES:
                        _fail("output tree exceeds the file count bound")
                    files.append(relative)
                else:
                    _fail(f"output member {relative} has invalid type")

    root_entry = os.fstat(root.descriptor)
    if (
        not stat.S_ISDIR(root_entry.st_mode)
        or stat.S_IMODE(root_entry.st_mode) != directory_mode
    ):
        _fail(f"output root must have sealed mode {directory_mode:04o}")
    visit(root.descriptor, PurePosixPath(), 0)
    return sorted(files), sorted(directories), total_bytes


def _seal_output_tree(
    root: _PinnedRoot,
    *,
    guard: Callable[[], None],
) -> None:
    entry_count = 0

    def seal(descriptor: int, depth: int) -> None:
        nonlocal entry_count
        if depth > 3:
            _fail("output tree exceeds the sealing depth bound")
        try:
            entries = os.scandir(descriptor)
        except OSError as error:
            _fail(f"cannot scan output tree while sealing: {error}")
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_OUTPUT_FILES + MAX_OUTPUT_DIRECTORIES:
                    _fail("output tree exceeds the sealing entry bound")
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as error:
                    _fail(f"cannot inspect output member while sealing: {error}")
                if stat.S_ISREG(info.st_mode):
                    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400:
                        _fail("output file identity changed before sealing")
                    continue
                if not stat.S_ISDIR(info.st_mode):
                    _fail("output tree contains an invalid type while sealing")
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                try:
                    child = os.open(entry.name, flags, dir_fd=descriptor)
                except OSError as error:
                    _fail(f"cannot open output directory while sealing: {error}")
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (
                        info.st_dev,
                        info.st_ino,
                    ) or stat.S_IMODE(opened.st_mode) != 0o700:
                        _fail("output directory changed before sealing")
                    seal(child, depth + 1)
                    os.fchmod(child, 0o500)
                    os.fsync(child)
                finally:
                    os.close(child)

    guard()
    if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != 0o700:
        _fail("output root must be mode 0700 before sealing")
    seal(root.descriptor, 0)
    os.fchmod(root.descriptor, 0o500)
    os.fsync(root.descriptor)
    guard()
    if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != 0o500:
        _fail("output root did not retain sealed mode 0500")


def _inventory_members(
    root: _PinnedRoot,
    members: Sequence[str],
) -> list[dict[str, object]]:
    if len(members) > MAX_OUTPUT_FILES:
        _fail("member inventory exceeds the file count bound")
    inventory: list[dict[str, object]] = []
    for member in sorted(members):
        pin = _pin_output_member(root, member, maximum=MAX_MEMBER_BYTES)
        try:
            inventory.append(
                {"member": member, "bytes": pin.size, "sha256": pin.sha256},
            )
        finally:
            pin.close()
    return inventory


def _read_bound_output_member(
    root: _PinnedRoot,
    member: str,
    *,
    maximum: int,
) -> bytes:
    pin = _pin_output_member(root, member, maximum=maximum)
    try:
        return _pinned_bytes(pin, context=f"output member {member}")
    finally:
        pin.close()


def _validate_tree_against_manifest(
    root: _PinnedRoot,
    manifest_raw: bytes,
    *,
    directory_mode: int = 0o500,
) -> list[dict[str, object]]:
    if (
        _read_bound_output_member(
            root,
            MANIFEST_MEMBER,
            maximum=4 * 1024 * 1024,
        )
        != manifest_raw
    ):
        _fail("on-disk machine manifest bytes differ from the supplied manifest")
    try:
        manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"machine manifest is not canonical JSON: {error}")
    if not isinstance(manifest, dict) or _canonical_json(manifest) != manifest_raw:
        _fail("machine manifest must use canonical JSON encoding")
    if manifest.get("schema") != MACHINE_CLOSURE_SCHEMA:
        _fail("machine manifest schema is invalid")
    if manifest.get("contract") != MACHINE_CLOSURE_CONTRACT:
        _fail("machine manifest contract is invalid")
    payload_sha = manifest.get("payload_sha256")
    if not isinstance(payload_sha, str):
        _fail("machine manifest lacks its payload digest")
    unsigned = dict(manifest)
    del unsigned["payload_sha256"]
    if _sha256(_canonical_json(unsigned)) != payload_sha:
        _fail("machine manifest payload digest is invalid")
    inventory = manifest.get("member_inventory")
    if not isinstance(inventory, list):
        _fail("machine manifest member inventory must be an array")
    if len(inventory) > MAX_OUTPUT_FILES:
        _fail("machine manifest member inventory exceeds the file count bound")
    normalized: list[dict[str, object]] = []
    names: list[str] = []
    for index, record in enumerate(inventory):
        if not isinstance(record, dict) or set(record) != {"member", "bytes", "sha256"}:
            _fail(f"machine manifest member inventory[{index}] is invalid")
        member = _relative_member(str(record["member"]), context="manifest member")
        size = record["bytes"]
        digest = record["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(f"manifest member {member} has an invalid byte count")
        if not isinstance(digest, str):
            _fail(f"manifest member {member} lacks a digest")
        _expect_sha256(digest, context=f"manifest member {member} digest")
        names.append(member)
        normalized.append({"member": member, "bytes": size, "sha256": digest})
    if names != sorted(set(names)) or MANIFEST_MEMBER in names:
        _fail("machine manifest member inventory order or membership is invalid")
    actual_files, _, _ = _walk_output(root, directory_mode=directory_mode)
    expected_files = sorted([MANIFEST_MEMBER, *names])
    if actual_files != expected_files:
        _fail(
            f"machine closure inventory mismatch; expected {expected_files}, "
            f"found {actual_files}",
        )
    actual = _inventory_members(root, names)
    if actual != normalized:
        _fail("machine closure member bytes do not match the manifest inventory")
    if (
        _read_bound_output_member(
            root,
            MANIFEST_MEMBER,
            maximum=4 * 1024 * 1024,
        )
        != manifest_raw
    ):
        _fail("on-disk machine manifest changed during tree validation")
    return normalized


def _produce_into(
    pdf_root: Path,
    output_root: _PinnedRoot,
    *,
    release_id: str,
    tool_paths: Mapping[str, Path],
    expected_tool_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
    output_guard: Callable[[], None],
) -> _Production:
    _expect_token(release_id, context="release id")
    _validate_fd_headroom(0)
    pdf_root_absolute = _canonical_existing_directory(pdf_root, context="PDF root")
    output_absolute = output_root.path
    _assert_distinct_roots(
        (pdf_root_absolute, "PDF root"),
        (output_absolute, "private output root"),
    )
    output_guard()
    if stat.S_IMODE(os.fstat(output_root.descriptor).st_mode) != 0o700:
        _fail("private output root must begin in mode 0700")
    root: object | None = None
    pdfs: dict[str, object] = {}
    tools: dict[str, object] = {}
    builder: _PinnedFile | None = None
    try:
        root, root_record = _pin_pdf_root(pdf_root_absolute)
        pdf_sha256_anchors = _validate_pdf_sha256_anchors(expected_pdf_sha256)
        pdfs = _pin_pdfs(
            root,
            expected_pdf_sha256=pdf_sha256_anchors,
        )
        tool_sha256_anchors = _validate_tool_sha256_anchors(expected_tool_sha256)
        tools = _pin_tools(
            tool_paths,
            expected_tool_sha256=tool_sha256_anchors,
        )
        builder = _pin_dependency(
            Path(__file__).absolute(),
            context="machine closure builder",
        )
        _validate_fd_headroom(2 + len(pdfs) + len(tools))

        def revalidate() -> None:
            if root is None:  # pragma: no cover - closure invariant
                _fail("PDF root is not pinned")
            if builder is None:  # pragma: no cover - closure invariant
                _fail("machine closure builder is not pinned")
            _revalidate_inputs(root, pdfs, tools, builder)
            output_guard()
            _revalidate_root(
                output_root,
                context="private output root",
                check_mtime=False,
            )

        revalidate()
        budget = _ProcessBudget()
        page_budget = _PageBudget()
        invocations: list[dict[str, object]] = []
        output_members: list[dict[str, object]] = []
        tool_records: list[dict[str, object]] = []
        for name in TOOL_ORDER:
            display_argv = [str(tools[name].path), "-v"]
            stdout, stderr, invocation, raw_members = _invoke_and_record(
                name=f"{name}-version",
                tool_pin=tools[name],
                arguments=["-v"],
                display_argv=display_argv,
                inherited_fds=(),
                output_root=output_root,
                raw_prefix=f"raw/tools/{name}-version",
                timeout=TOOL_TIMEOUT_SECONDS,
                budget=budget,
                revalidate=revalidate,
                output_guard=revalidate,
            )
            invocations.append(invocation)
            output_members.extend(raw_members)
            version_text = _ascii(stdout + stderr, context=f"{name} version")
            combined = [
                line.strip() for line in version_text.splitlines() if line.strip()
            ]
            candidates = [line for line in combined if VERSION_RE.fullmatch(line)]
            if len(candidates) != 1 or not candidates[0].startswith(f"{name} version "):
                _fail(f"{name} version output is not canonical Poppler output")
            tool_records.append(_tool_record(name, tools[name], candidates[0]))

        documents: list[dict[str, object]] = []
        total_pages = 0
        for pdf_id, member in PDF_ORDER:
            document, document_invocations, document_members = _derive_document(
                pdf_id=pdf_id,
                member=member,
                pdf_pin=pdfs[pdf_id],
                tools=tools,
                output_root=output_root,
                budget=budget,
                page_budget=page_budget,
                revalidate=revalidate,
            )
            total_pages += int(document["page_count"])
            documents.append(document)
            invocations.extend(document_invocations)
            output_members.extend(document_members)
        if total_pages != page_budget.count:
            _fail("aggregate page budget accounting is inconsistent")

        pdf_records = [
            _pdf_record(pdf_id, member, pdfs[pdf_id]) for pdf_id, member in PDF_ORDER
        ]
        pdf_set_sha256 = _sha256(_canonical_json(pdf_records))
        tool_set_sha256 = _sha256(_canonical_json(tool_records))
        render_payload = [
            {
                "pdf_id": document["pdf_id"],
                "pdf_sha256": document["sha256"],
                "pages": [
                    {
                        "page": page["page"],
                        "member": page["render"]["member"],
                        "sha256": page["render"]["sha256"],
                        "bytes": page["render"]["bytes"],
                        "width_pixels": page["render"]["width_pixels"],
                        "height_pixels": page["render"]["height_pixels"],
                    }
                    for page in document["page_geometry"]
                ],
            }
            for document in documents
        ]
        render_set_sha256 = _sha256(_canonical_json(render_payload))
        builder_record = {
            "member": BUILDER_MEMBER,
            "absolute_path": str(builder.path),
            "device": builder.device,
            "inode": builder.inode,
            "mode": format(
                stat.S_IMODE(os.fstat(builder.descriptor).st_mode),
                "04o",
            ),
            "bytes": builder.size,
            "sha256": builder.sha256,
        }
        producer = {
            "schema": PRODUCER_RECEIPT_SCHEMA,
            "contract": MACHINE_CLOSURE_CONTRACT,
            "release_id": release_id,
            "pdf_root": root_record,
            "pdfs": pdf_records,
            "tools": tool_records,
            "builder": builder_record,
            "execution_contract": {
                "shell": False,
                "cwd": "/",
                "environment": dict(EXACT_ENVIRONMENT),
                "darwin_suspended_spawn": True,
                "spawn_flags": DARWIN_SPAWN_FLAGS,
                "tool_binary_contract": "thin-native-arm64-macho64",
                "tool_trust_anchor": "caller-supplied-exact-sha256-per-tool",
                "expected_tool_sha256": tool_sha256_anchors,
                "execution_binding_scope": "main_executable",
                "non_system_dylib_closure": "not_attested",
                "pdf_inputs": "descriptor_bound",
                "pdf_authority_anchor": (
                    "caller-supplied-exact-sha256-per-fixed-pdf-role"
                ),
                "expected_pdf_sha256": pdf_sha256_anchors,
                "render_output": "bounded_stdout_then_descriptor_rooted_write",
                "same_vnode_mutation_fail_stop_assumption": (
                    "invalid-signed-code-page-triggers-darwin-cs-kill"
                ),
                "other_same_vnode_mutations": "not_attested",
                "process_group_kill_on_timeout_or_overflow": True,
                "one_page_per_render_process": True,
                "double_render": True,
                "limits": {
                    "pdf_bytes_each": MAX_PDF_BYTES,
                    "pdf_bytes_total": MAX_TOTAL_PDF_BYTES,
                    "raw_stdout_bytes": MAX_RAW_STREAM_BYTES,
                    "raw_stderr_bytes": MAX_STDERR_BYTES,
                    "png_bytes_each": MAX_PNG_BYTES,
                    "output_bytes_total": MAX_OUTPUT_BYTES,
                    "pages_each": MAX_PAGES_PER_PDF,
                    "pages_total": MAX_TOTAL_PAGES,
                    "fonts_each": MAX_FONTS_PER_PDF,
                    "images_each": MAX_IMAGES_PER_PDF,
                    "files": MAX_OUTPUT_FILES,
                    "directories": MAX_OUTPUT_DIRECTORIES,
                    "processes": MAX_PROCESSES,
                    "open_file_descriptors": MAX_OPEN_FILE_DESCRIPTORS,
                    "tool_timeout_seconds": int(TOOL_TIMEOUT_SECONDS),
                    "render_timeout_seconds": int(RENDER_TIMEOUT_SECONDS),
                },
            },
            "invocation_count": budget.count,
            "invocations": invocations,
            "documents": documents,
            "pdf_set_sha256": pdf_set_sha256,
            "tool_set_sha256": tool_set_sha256,
            "render_set_sha256": render_set_sha256,
        }
        producer_raw = _canonical_json(producer)
        producer_member = _write_member(
            output_root,
            PRODUCER_MEMBER,
            producer_raw,
            guard=revalidate,
        )
        output_members.append(producer_member)
        # Derive the inventory from disk, not the accumulated claims.
        files_before_manifest, _, _ = _walk_output(
            output_root,
            directory_mode=0o700,
        )
        if len(files_before_manifest) != len(set(files_before_manifest)):
            _fail("output tree contains duplicate canonical members")
        inventory = _inventory_members(output_root, files_before_manifest)
        claimed = sorted(output_members, key=lambda record: str(record["member"]))
        if inventory != claimed:
            _fail("producer output claims do not match the exact private tree")
        manifest_unsigned = {
            "schema": MACHINE_CLOSURE_SCHEMA,
            "contract": MACHINE_CLOSURE_CONTRACT,
            "release_id": release_id,
            "pdf_order": [
                {"pdf_id": pdf_id, "member": member} for pdf_id, member in PDF_ORDER
            ],
            "producer_receipt": producer_member,
            "member_inventory": inventory,
            "pdf_set_sha256": pdf_set_sha256,
            "tool_set_sha256": tool_set_sha256,
            "render_set_sha256": render_set_sha256,
            "summary": {
                "pdf_count": len(documents),
                "page_count": total_pages,
                "machine_pass_count": len(documents),
                "raster_image_count": sum(
                    int(document["raster_image_count"]) for document in documents
                ),
                "reproducible_page_count": total_pages,
            },
            "non_inference_limits": {
                "scientific_correctness": "not inferred",
                "human_visual_approval": "not inferred",
                "journal_acceptance": "not inferred",
                "submission_status": "not inferred",
            },
        }
        manifest = {
            **manifest_unsigned,
            "payload_sha256": _sha256(_canonical_json(manifest_unsigned)),
        }
        manifest_raw = _canonical_json(manifest)
        _write_member(
            output_root,
            MANIFEST_MEMBER,
            manifest_raw,
            guard=revalidate,
        )
        revalidate()
        _seal_output_tree(output_root, guard=revalidate)
        _validate_tree_against_manifest(
            output_root,
            manifest_raw,
            directory_mode=0o500,
        )
        return _Production(
            manifest=manifest,
            manifest_raw=manifest_raw,
            member_inventory=inventory,
            page_count=total_pages,
        )
    finally:
        if builder is not None:
            builder.close()
        for tool in tools.values():
            tool.close()
        for pdf in pdfs.values():
            pdf.close()
        if root is not None:
            root.close()


def _named_directory_identity(
    parent: _PinnedRoot,
    name: str,
    *,
    context: str,
) -> tuple[int, int, int] | None:
    try:
        entry = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if not stat.S_ISDIR(entry.st_mode) or stat.S_ISLNK(entry.st_mode):
        _fail(f"{context} must be a non-symlink directory")
    return entry.st_dev, entry.st_ino, stat.S_IMODE(entry.st_mode)


def _revalidate_reserved_directory(
    parent: _PinnedRoot,
    child: _PinnedRoot,
    name: str,
    *,
    context: str,
    allowed_modes: set[int],
) -> None:
    _revalidate_root(parent, context=f"{context} parent")
    _revalidate_root(child, context=context, check_mtime=False)
    named = _named_directory_identity(parent, name, context=context)
    if named is None or named[:2] != (child.device, child.inode):
        _fail(f"{context} name or identity changed")
    if named[2] not in allowed_modes:
        _fail(f"{context} mode changed")


def _reserve_directory(
    target: Path,
    *,
    reserved_name: str,
    context: str,
) -> tuple[Path, _PinnedRoot, _PinnedRoot]:
    absolute = _canonical_new_directory(target, context=context)
    parent = _pin_root(absolute.parent, context=f"{context} parent")
    child: _PinnedRoot | None = None
    descriptor = -1
    candidate_reserved = False
    try:
        _revalidate_root(parent, context=f"{context} parent")
        if (
            _named_directory_identity(
                parent,
                absolute.name,
                context=f"{context} destination",
            )
            is not None
        ):
            _fail(f"{context} destination already exists")
        if (
            _named_directory_identity(
                parent,
                reserved_name,
                context=f"{context} candidate",
            )
            is not None
        ):
            _fail(
                "retained private candidate requires explicit review before retry: "
                f"{absolute.parent / reserved_name}",
            )
        try:
            os.mkdir(reserved_name, 0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            _fail(
                "concurrent private candidate reservation already exists: "
                f"{absolute.parent / reserved_name}",
            )
        except OSError as error:
            _fail(f"cannot reserve private candidate directory: {error}")
        candidate_reserved = True
        parent.mtime_ns = os.fstat(parent.descriptor).st_mtime_ns
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(reserved_name, flags, dir_fd=parent.descriptor)
        except OSError as error:
            _fail(f"cannot open reserved private candidate: {error}")
        entry = os.fstat(descriptor)
        child = _PinnedRoot(
            path=absolute.parent / reserved_name,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            mtime_ns=entry.st_mtime_ns,
        )
        descriptor = -1
        _revalidate_reserved_directory(
            parent,
            child,
            reserved_name,
            context=f"{context} candidate",
            allowed_modes={0o700},
        )
        return absolute, parent, child
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        if child is not None:
            child.close()
        parent.close()
        if candidate_reserved:
            message = (
                f"{error}; candidate_path={absolute.parent / reserved_name}; "
                "candidate_state=reserved-private-candidate-do-not-auto-delete; "
                "inspect identity and exact inventory before explicit removal"
            )
            raise MachineClosureError(message) from error
        raise


def _rename_no_replace(source: str, destination: str, parent_descriptor: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            function = library.renameatx_np
        except AttributeError:
            _fail(
                "platform atomic no-replace rename symbol renameatx_np is unavailable",
            )
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        try:
            function = library.renameat2
        except AttributeError:
            _fail("platform atomic no-replace rename symbol renameat2 is unavailable")
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            os.fsencode(source),
            parent_descriptor,
            os.fsencode(destination),
            1,
        )
    else:
        _fail("platform lacks a supported atomic no-replace directory rename")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        _fail("machine closure destination already exists")
    unsupported = {
        value
        for value in (
            getattr(errno, "ENOSYS", None),
            getattr(errno, "ENOTSUP", None),
            getattr(errno, "EOPNOTSUPP", None),
        )
        if value is not None
    }
    if error_number in unsupported:
        _fail(
            "filesystem/platform does not support atomic no-replace directory rename: "
            f"{os.strerror(error_number)} (errno {error_number})",
        )
    _fail(
        "atomic no-replace directory publication failed: "
        f"{os.strerror(error_number)} (errno {error_number})",
    )


def _publish_directory(
    parent: _PinnedRoot,
    stage: _PinnedRoot,
    stage_name: str,
    destination: Path,
    *,
    manifest_raw: bytes,
) -> None:
    absolute = destination.absolute()
    destination_renamed = False
    try:
        _revalidate_reserved_directory(
            parent,
            stage,
            stage_name,
            context="sealed private publication candidate",
            allowed_modes={0o500},
        )
        _validate_tree_against_manifest(stage, manifest_raw, directory_mode=0o500)
        try:
            _rename_no_replace(stage_name, absolute.name, parent.descriptor)
        except BaseException as error:
            message = (
                f"{error}; candidate_paths={stage.path}|{absolute}; "
                "candidate_state=rename-issued-outcome-ambiguous-do-not-auto-delete; "
                "inspect both names, identities, and inventories before any "
                "explicit removal"
            )
            raise MachineClosureError(message) from error
        destination_renamed = True
        os.fsync(parent.descriptor)
        parent.mtime_ns = os.fstat(parent.descriptor).st_mtime_ns
        _revalidate_root(parent, context="machine closure destination parent")
        named = _named_directory_identity(
            parent,
            absolute.name,
            context="published machine closure",
        )
        if named != (stage.device, stage.inode, 0o500):
            _fail("published machine closure destination identity changed")
        _validate_tree_against_manifest(stage, manifest_raw, directory_mode=0o500)
        os.fsync(stage.descriptor)
        named = _named_directory_identity(
            parent,
            absolute.name,
            context="published machine closure readback",
        )
        if named != (stage.device, stage.inode, 0o500):
            _fail("published machine closure destination changed during readback")
    except BaseException as error:
        if isinstance(error, MachineClosureError) and "candidate_paths=" in str(error):
            raise
        candidate = absolute if destination_renamed else stage.path
        state = (
            "destination-name-may-be-owned-or-replaced-do-not-auto-delete"
            if destination_renamed
            else "private-stage-name-may-be-owned-or-replaced-do-not-auto-delete"
        )
        message = (
            f"{error}; candidate_path={candidate}; candidate_state={state}; "
            "inspect identity and exact inventory before any explicit removal"
        )
        raise MachineClosureError(message) from error


def _receipt(
    root: Path,
    production: _Production,
    *,
    replay_root: Path | None,
) -> MachineClosureReceipt:
    summary = production.manifest["summary"]
    return MachineClosureReceipt(
        manifest_path=str(root / MANIFEST_MEMBER),
        manifest_sha256=_sha256(production.manifest_raw),
        pdf_set_sha256=str(production.manifest["pdf_set_sha256"]),
        tool_set_sha256=str(production.manifest["tool_set_sha256"]),
        render_set_sha256=str(production.manifest["render_set_sha256"]),
        pdf_count=int(summary["pdf_count"]),
        page_count=int(summary["page_count"]),
        machine_pass_count=int(summary["machine_pass_count"]),
        replay_root=str(replay_root) if replay_root is not None else None,
    )


def build_machine_closure(
    pdf_root: Path,
    destination: Path,
    *,
    release_id: str,
    tool_paths: Mapping[str, Path],
    expected_tool_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> MachineClosureReceipt:
    """Build and atomically publish one exact native machine-QA directory."""
    pdf_root_absolute = _canonical_existing_directory(pdf_root, context="PDF root")
    destination_absolute = destination.absolute()
    _assert_distinct_roots(
        (pdf_root_absolute, "PDF root"),
        (destination_absolute, "machine closure destination"),
    )
    stage_name = f".{destination_absolute.name}.private-candidate"
    absolute_destination, parent, stage = _reserve_directory(
        destination_absolute,
        reserved_name=stage_name,
        context="machine closure",
    )

    def output_guard() -> None:
        _revalidate_reserved_directory(
            parent,
            stage,
            stage_name,
            context="machine closure private candidate",
            allowed_modes={0o700, 0o500},
        )

    try:
        try:
            production = _produce_into(
                pdf_root_absolute,
                stage,
                release_id=release_id,
                tool_paths=tool_paths,
                expected_tool_sha256=expected_tool_sha256,
                expected_pdf_sha256=expected_pdf_sha256,
                output_guard=output_guard,
            )
        except BaseException as error:
            message = (
                f"{error}; candidate_path={stage.path}; "
                "candidate_state=partial-or-complete-private-candidate-do-not-"
                "auto-delete; inspect identity and exact inventory before any "
                "explicit removal"
            )
            raise MachineClosureError(message) from error
        _publish_directory(
            parent,
            stage,
            stage_name,
            absolute_destination,
            manifest_raw=production.manifest_raw,
        )
        return _receipt(absolute_destination, production, replay_root=None)
    finally:
        stage.close()
        parent.close()


def _read_anchored_manifest(
    closure_root: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    _expect_sha256(expected_manifest_sha256, context="expected manifest anchor")
    absolute = _canonical_existing_directory(
        closure_root,
        context="machine closure root",
    )
    root = _pin_root(absolute, context="machine closure root")
    try:
        if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != 0o500:
            _fail("machine closure root must be sealed mode 0500")
        pin = _pin_output_member(
            root,
            MANIFEST_MEMBER,
            maximum=4 * 1024 * 1024,
        )
        try:
            raw = _pinned_bytes(pin, context="anchored machine manifest")
            if (
                pin.sha256 != expected_manifest_sha256
                or _sha256(raw) != expected_manifest_sha256
            ):
                _fail("machine manifest does not match the independent SHA-256 anchor")
            _validate_tree_against_manifest(root, raw, directory_mode=0o500)
            if _pinned_bytes(pin, context="anchored machine manifest") != raw:
                _fail("machine manifest changed during anchored validation")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:  # pragma: no cover - checked above
                _fail(f"machine manifest is invalid: {error}")
            return raw, parsed
        finally:
            pin.close()
    finally:
        root.close()


def validate_machine_closure(
    pdf_root: Path,
    closure_root: Path,
    replay_root: Path,
    *,
    expected_manifest_sha256: str,
    release_id: str,
    tool_paths: Mapping[str, Path],
    expected_tool_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> MachineClosureReceipt:
    """Replay every producer into a separate tree and compare exact bytes."""
    closure_absolute = _canonical_existing_directory(
        closure_root,
        context="machine closure root",
    )
    replay_absolute = replay_root.absolute()
    pdf_absolute = _canonical_existing_directory(pdf_root, context="PDF root")
    _assert_distinct_roots(
        (pdf_absolute, "PDF root"),
        (closure_absolute, "machine closure root"),
        (replay_absolute, "validation replay root"),
    )
    original_raw, original_manifest = _read_anchored_manifest(
        closure_absolute,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if original_manifest.get("release_id") != release_id:
        _fail("machine manifest release id does not match replay configuration")
    _, replay_parent, replay_pin = _reserve_directory(
        replay_absolute,
        reserved_name=replay_absolute.name,
        context="validation replay",
    )
    closure_pin: _PinnedRoot | None = None

    def replay_guard() -> None:
        _revalidate_reserved_directory(
            replay_parent,
            replay_pin,
            replay_absolute.name,
            context="validation replay root",
            allowed_modes={0o700, 0o500},
        )
        if closure_pin is None:  # pragma: no cover - acquisition invariant
            _fail("machine closure root is not pinned")
        _revalidate_root(closure_pin, context="machine closure root")

    try:
        closure_pin = _pin_root(closure_absolute, context="machine closure root")
        try:
            replay = _produce_into(
                pdf_absolute,
                replay_pin,
                release_id=release_id,
                tool_paths=tool_paths,
                expected_tool_sha256=expected_tool_sha256,
                expected_pdf_sha256=expected_pdf_sha256,
                output_guard=replay_guard,
            )
        except BaseException as error:
            message = (
                f"{error}; replay_candidate_path={replay_absolute}; "
                "candidate_state=partial-or-complete-validation-replay-do-not-"
                "auto-delete; inspect identity and exact inventory before any "
                "explicit removal"
            )
            raise MachineClosureError(message) from error
        replay_guard()
        if replay.manifest_raw != original_raw:
            _fail(
                "independent native replay manifest does not match the anchored "
                "closure",
            )
        original_files, original_directories, _ = _walk_output(
            closure_pin,
            directory_mode=0o500,
        )
        replay_files, replay_directories, _ = _walk_output(
            replay_pin,
            directory_mode=0o500,
        )
        if (original_files, original_directories) != (
            replay_files,
            replay_directories,
        ):
            _fail("independent native replay tree shape does not match the closure")
        original_inventory = _inventory_members(closure_pin, original_files)
        replay_inventory = _inventory_members(replay_pin, replay_files)
        if original_inventory != replay_inventory:
            _fail("independent native replay member bytes do not match the closure")
        return _receipt(closure_absolute, replay, replay_root=replay_absolute)
    except BaseException as error:
        if isinstance(error, MachineClosureError) and "replay_candidate_path=" in str(
            error,
        ):
            raise
        message = (
            f"{error}; replay_candidate_path={replay_absolute}; "
            "candidate_state=partial-or-complete-validation-replay-do-not-"
            "auto-delete; inspect identity and exact inventory before explicit "
            "removal"
        )
        raise MachineClosureError(message) from error
    finally:
        if closure_pin is not None:
            closure_pin.close()
        replay_pin.close()
        replay_parent.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or independently replay four-PDF native machine-QA closure.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build and atomically publish closure")
    build.add_argument("--pdf-root", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    build.add_argument("--release-id", required=True)
    for pdf_id in PDF_IDS:
        build.add_argument(f"--{pdf_id}-pdf-sha256", required=True)
    for tool_name in TOOL_ORDER:
        build.add_argument(f"--{tool_name}-path", type=Path, required=True)
        build.add_argument(f"--{tool_name}-sha256", required=True)
    validate = subparsers.add_parser(
        "validate",
        help="validate anchored closure with a separate retained native replay",
    )
    validate.add_argument("--pdf-root", type=Path, required=True)
    validate.add_argument("--closure-root", type=Path, required=True)
    validate.add_argument("--replay-root", type=Path, required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    validate.add_argument("--release-id", required=True)
    for pdf_id in PDF_IDS:
        validate.add_argument(f"--{pdf_id}-pdf-sha256", required=True)
    for tool_name in TOOL_ORDER:
        validate.add_argument(f"--{tool_name}-path", type=Path, required=True)
        validate.add_argument(f"--{tool_name}-sha256", required=True)
    return parser


def _cli_tool_paths(arguments: argparse.Namespace) -> dict[str, Path]:
    return {name: getattr(arguments, f"{name}_path") for name in TOOL_ORDER}


def _cli_tool_sha256(arguments: argparse.Namespace) -> dict[str, str]:
    return {name: getattr(arguments, f"{name}_sha256") for name in TOOL_ORDER}


def _cli_pdf_sha256(arguments: argparse.Namespace) -> dict[str, str]:
    return {pdf_id: getattr(arguments, f"{pdf_id}_pdf_sha256") for pdf_id in PDF_IDS}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit build or validation command."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        receipt = build_machine_closure(
            arguments.pdf_root,
            arguments.destination,
            release_id=arguments.release_id,
            tool_paths=_cli_tool_paths(arguments),
            expected_tool_sha256=_cli_tool_sha256(arguments),
            expected_pdf_sha256=_cli_pdf_sha256(arguments),
        )
    else:
        receipt = validate_machine_closure(
            arguments.pdf_root,
            arguments.closure_root,
            arguments.replay_root,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            release_id=arguments.release_id,
            tool_paths=_cli_tool_paths(arguments),
            expected_tool_sha256=_cli_tool_sha256(arguments),
            expected_pdf_sha256=_cli_pdf_sha256(arguments),
        )
    print(_canonical_json(asdict(receipt)).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

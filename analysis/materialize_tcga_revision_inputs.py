"""Materialize the signed, canonical TCGA revision mutation inputs.

This stage is result-blind. It verifies a separately pinned D1/D2 approval,
recursively validates the immutable participant population, attests each raw MAF
against the frozen DataHub receipt, and publishes deterministic full-variant MAFs.
It never invokes a BMR provider or opens an association output.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import types
import uuid
from collections import Counter
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

# ``-I`` deliberately removes the script directory.  A materialization/validation
# child launched from the frozen implementation tree adds only that tree's src
# directory before importing DIALECT policy modules.
_BOOTSTRAP_SOURCE_ROOT = Path(__file__).parent.parent / "src"
if (
    "dialect.data.variants" not in sys.modules
    and (_BOOTSTRAP_SOURCE_ROOT / "dialect" / "data" / "variants.py").is_file()
):
    sys.path.insert(0, _BOOTSTRAP_SOURCE_ROOT.as_posix())

import pandas as pd

import dialect.data.revision_approval as approval_data
import dialect.data.tcga as tcga_data
import dialect.data.variants as variant_data
from dialect.data.revision_approval import (
    MATERIALIZE_FINAL_INPUTS_STAGE,
    STAGE_SCOPED_APPROVAL_SCHEMA,
    RevisionApproval,
    validate_revision_approval,
)
from dialect.data.tcga import (
    TCGA_CASE_LIST_RECEIPTS,
    TCGA_COHORTS,
    TCGA_DATAHUB_COMMIT,
    TCGA_DATAHUB_TREE,
    parse_tcga_sequenced_case_list,
    tcga_datahub_case_list_path,
    tcga_datahub_public_path,
)
from dialect.data.variants import TCGA_DUPLICATE_RESOLUTION_POLICY

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SCHEMA_VERSION: Final = "4.0.0"
INPUT_CONTRACT: Final = "signed-participant-axis-canonical-maf-v4"
D1_INPUT_CONTRACT: Final = "signed-participant-axis-canonical-maf-v3"
POPULATION_SCHEMA_VERSION: Final = "1.0.0"
POPULATION_CONTRACT: Final = "pinned-datahub-participant-unique-sample-axis-v1"
SERIALIZER_CONTRACT: Final = "utf8-tsv-lf-single-terminal-newline-v1"
STREAMING_CANONICALIZATION_CONTRACT: Final = (
    "sqlite-binary-key-framed-row-token-frozen-variants-equivalence-v1"
)
DECISION_ARTIFACT_SCHEMA: Final = "dialect-revision-machine-decision-v1"
D1_CONTRACT: Final = "tcga-full-variant-resolution-v1"
D2_CONTRACT: Final = "tcga-participant-population-v1"
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID_PATTERN: Final = re.compile(r"[0-9a-f]{40,64}")
_GIT_LFS_POINTER_PATTERN: Final = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size ([1-9][0-9]*)\n",
)
_MAX_GIT_LFS_POINTER_BYTES: Final = 256
_PUBLISH_CLAIM_SUFFIX: Final = ".publish-claim"
_STREAM_CHUNK_ROWS: Final = 8192
_SQLITE_FETCH_ROWS: Final = 4096
_MAX_DUPLICATE_GROUP_ROWS: Final = 100_000
_MAX_DUPLICATE_GROUP_BYTES: Final = 128 * 1024 * 1024
_SQLITE_CACHE_KIB: Final = 32 * 1024
_MIN_AVAILABLE_MEMORY_BYTES: Final = 768 * 1024 * 1024
_DISK_FIXED_SAFETY_BYTES: Final = 2 * 1024 * 1024 * 1024
_DISK_RAW_MULTIPLIER: Final = 10
_FROZEN_FILE_MODE: Final = 0o400
_FROZEN_DIRECTORY_MODE: Final = 0o500
_STAGING_FILE_MODE: Final = 0o600
_STAGING_DIRECTORY_MODE: Final = 0o700
_RUNTIME_CLOSURE_SUFFIXES: Final = (
    ".py",
    ".pyi",
    ".so",
    ".dylib",
    ".pyd",
    ".dll",
)
_SOURCE_SNAPSHOT_PATHS: Final = {
    "materializer": "implementation/analysis/materialize_tcga_revision_inputs.py",
    "population_materializer": (
        "implementation/analysis/materialize_tcga_revision_population.py"
    ),
    "canonicalizer": "implementation/src/dialect/data/variants.py",
    "tcga": "implementation/src/dialect/data/tcga.py",
    "revision_approval": "implementation/src/dialect/data/revision_approval.py",
}
_SNAPSHOT_SUPPORT_FILES: Final = {
    "implementation/src/dialect/__init__.py": (
        b'"""Isolated DIALECT policy snapshot."""\n'
    ),
    "implementation/src/dialect/data/__init__.py": (
        b'"""Isolated DIALECT data-policy snapshot."""\n'
    ),
}
_ISOLATED_BOOTSTRAP: Final = r"""
import hashlib
import json
import os
import stat
import sys
import types

root, expected, request, request_sha256, response = sys.argv[1:]
directory_flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
root_descriptor = os.open(os.path.sep, directory_flags)
for component in os.path.abspath(root).split(os.path.sep):
    if not component:
        continue
    next_descriptor = os.open(component, directory_flags, dir_fd=root_descriptor)
    os.close(root_descriptor)
    root_descriptor = next_descriptor

def stable_read(relative):
    components = relative.split("/")
    if (
        not relative
        or os.path.isabs(relative)
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise RuntimeError("snapshot source path is not a safe relative path")
    parent = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = next_descriptor
        descriptor = os.open(
            components[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
    finally:
        os.close(parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("snapshot source is not a single-link regular file")
        digest = hashlib.sha256()
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )
        if identity(before) != identity(after) or size != before.st_size:
            raise RuntimeError("snapshot source changed while opening")
        return b"".join(chunks), size, digest.hexdigest()
    finally:
        os.close(descriptor)

manifest_bytes, _, observed = stable_read("execution_manifest.json")
if observed != expected:
    raise RuntimeError("execution snapshot manifest hash mismatch")
manifest = json.loads(manifest_bytes)
source_bytes = {}
for name, record in manifest["sources"].items():
    relative = record["path"]
    path = os.path.join(root, *relative.split("/"))
    content, size, digest = stable_read(relative)
    if size != record["bytes"] or digest != record["sha256"]:
        raise RuntimeError("execution snapshot source receipt mismatch: " + name)
    source_bytes[name] = (path, content)
for relative, record in manifest["support"].items():
    _, size, digest = stable_read(relative)
    if size != record["bytes"] or digest != record["sha256"]:
        raise RuntimeError("execution snapshot support receipt mismatch: " + relative)
os.close(root_descriptor)

dialect = types.ModuleType("dialect")
dialect.__path__ = [os.path.join(root, "implementation", "src", "dialect")]
dialect.__package__ = "dialect"
data = types.ModuleType("dialect.data")
data.__path__ = [os.path.join(root, "implementation", "src", "dialect", "data")]
data.__package__ = "dialect.data"
dialect.data = data
sys.modules["dialect"] = dialect
sys.modules["dialect.data"] = data
for short, key in (
    ("tcga", "tcga"),
    ("revision_approval", "revision_approval"),
    ("variants", "canonicalizer"),
):
    name = "dialect.data." + short
    path, content = source_bytes[key]
    module = types.ModuleType(name)
    module.__file__ = path
    module.__package__ = "dialect.data"
    sys.modules[name] = module
    setattr(data, short, module)
    exec(compile(content, path, "exec", dont_inherit=True), module.__dict__)

script, content = source_bytes["materializer"]
namespace = {
    "__name__": "__main__",
    "__file__": script,
    "__package__": None,
    "__builtins__": __builtins__,
}
sys.argv = [
    script,
    "--internal-execution-snapshot-sha256", expected,
    "--internal-request", request,
    "--internal-request-sha256", request_sha256,
    "--internal-response", response,
]
exec(compile(content, script, "exec", dont_inherit=True), namespace)
""".lstrip()
_PACKAGE_NAMES: Final = ("numpy", "pandas")


class RevisionInputError(ValueError):
    """Raised when a canonical input bundle fails closed."""


@dataclass(frozen=True, slots=True)
class _StableFileState:
    """Identity and mutation-sensitive metadata for one opened regular file."""

    device: int
    inode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class _FileReceipt:
    """Exact byte count and digest consumed from one stable descriptor."""

    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _StreamingCanonicalization:
    """Bounded canonicalization output and identifier-free accounting."""

    raw_rows: int
    selected_rows: int
    output_rows: int
    multiallelic_coordinate_groups: int
    audit: object
    output_receipt: _FileReceipt
    ordered_columns_sha256: str


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/integer equality aliasing."""
    return _canonical_json(left) == _canonical_json(right)


def _open_directory_fd(path: Path, *, label: str) -> int:
    """Open an absolute directory through no-follow ancestor descriptors."""
    absolute = _absolute_unresolved(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        msg = f"Unable to open {label} without following directory symlinks: {path}"
        raise RevisionInputError(msg) from error
    return descriptor


def _open_regular_fd(path: Path, *, label: str) -> int:
    """Open a private regular file without following any symlink.

    A link count of one is part of the input contract.  Without it, an attacker
    could mutate the same inode through an unrelated pathname after the checked
    path had passed the no-follow walk.
    """
    absolute = _absolute_unresolved(path)
    parent_descriptor = _open_directory_fd(absolute.parent, label=f"{label} parent")
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        msg = f"Unable to open {label} without following symlinks: {path}"
        raise RevisionInputError(msg) from error
    finally:
        os.close(parent_descriptor)
    file_stat = os.fstat(descriptor)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        os.close(descriptor)
        msg = f"{label} must be a single-link regular file: {path}"
        raise RevisionInputError(msg)
    return descriptor


def _stable_file_state(descriptor: int, *, label: str) -> _StableFileState:
    file_stat = os.fstat(descriptor)
    if not stat.S_ISREG(file_stat.st_mode):
        msg = f"{label} must remain a regular file."
        raise RevisionInputError(msg)
    return _StableFileState(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        links=file_stat.st_nlink,
        size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        ctime_ns=file_stat.st_ctime_ns,
    )


@contextmanager
def _stable_regular_descriptor(
    path: Path,
    *,
    label: str,
) -> Iterable[tuple[int, _StableFileState]]:
    """Yield one no-follow descriptor and reject mutation during consumption."""
    descriptor = _open_regular_fd(path, label=label)
    before = _stable_file_state(descriptor, label=label)
    try:
        yield descriptor, before
    finally:
        try:
            after = _stable_file_state(descriptor, label=label)
            if after != before:
                msg = (
                    f"{label} changed while its opened descriptor was consumed: {path}"
                )
                raise RevisionInputError(msg)
        finally:
            os.close(descriptor)


def _descriptor_chunks(descriptor: int) -> Iterable[bytes]:
    """Read a descriptor from its current offset in fixed bounded chunks."""
    while True:
        chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


def _consume_descriptor_receipt(descriptor: int) -> _FileReceipt:
    digest = hashlib.sha256()
    size = 0
    for chunk in _descriptor_chunks(descriptor):
        size += len(chunk)
        digest.update(chunk)
    return _FileReceipt(bytes=size, sha256=digest.hexdigest())


def _ensure_directory_fd(path: Path, *, label: str, mode: int = 0o700) -> int:
    """Create/open an absolute directory tree using only parent descriptors."""
    absolute = _absolute_unresolved(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=mode, dir_fd=descriptor)
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        msg = f"Unable to create/open {label} without directory symlinks: {path}"
        raise RevisionInputError(msg) from error
    return descriptor


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    with _stable_regular_descriptor(path, label=label) as (descriptor, state):
        chunks = list(_descriptor_chunks(descriptor))
        content = b"".join(chunks)
        if len(content) != state.size:
            msg = f"{label} byte count changed while reading: {path}"
            raise RevisionInputError(msg)
        return content


def _regular_file_receipt(path: Path, *, label: str) -> tuple[int, str]:
    with _stable_regular_descriptor(path, label=label) as (descriptor, state):
        receipt = _consume_descriptor_receipt(descriptor)
        if receipt.bytes != state.size:
            msg = f"{label} byte count changed while hashing: {path}"
            raise RevisionInputError(msg)
        return receipt.bytes, receipt.sha256


def _sha256(path: Path) -> str:
    return _regular_file_receipt(path, label="SHA-256 input")[1]


def _sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_record(path: Path, *, display_path: str) -> dict[str, int | str]:
    size, digest = _regular_file_receipt(path, label="artifact")
    return {
        "path": display_path,
        "bytes": size,
        "sha256": digest,
    }


def _absolute_unresolved(path: str | Path) -> Path:
    return Path(os.path.abspath(path))  # noqa: PTH100


def _require_regular_file(path: Path, *, label: str) -> None:
    descriptor = _open_regular_fd(path, label=label)
    os.close(descriptor)


def _require_directory(path: Path, *, label: str) -> None:
    descriptor = _open_directory_fd(path, label=label)
    os.close(descriptor)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        msg = (
            f"{label} fields differ from the frozen schema; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )
        raise RevisionInputError(msg)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"JSON object contains duplicate key {key!r}."
            raise RevisionInputError(msg)
        result[key] = value
    return result


def _parse_json_document(raw: bytes, *, path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Invalid UTF-8 JSON manifest: {path}"
        raise RevisionInputError(msg) from error
    _reject_surrogates(parsed, label=f"JSON manifest {path}")
    if not isinstance(parsed, dict):
        msg = f"JSON manifest must contain an object: {path}"
        raise RevisionInputError(msg)
    if raw != _canonical_json(parsed) + b"\n":
        msg = f"JSON manifest is not canonical JSON with one terminal LF: {path}"
        raise RevisionInputError(msg)
    return parsed


def _reject_nonfinite_constant(value: str) -> object:
    msg = f"Non-finite JSON constant is forbidden: {value}."
    raise RevisionInputError(msg)


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = f"Non-finite JSON number is forbidden: {value}."
        raise RevisionInputError(msg)
    return parsed


def _reject_surrogates(value: object, *, label: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            msg = f"{label} contains an invalid Unicode surrogate."
            raise RevisionInputError(msg)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_surrogates(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key, label=f"{label} object key")
            _reject_surrogates(item, label=f"{label}.{key}")


def _read_json(path: Path) -> dict[str, Any]:
    raw = _read_regular_bytes(path, label="JSON manifest")
    return _parse_json_document(raw, path=path)


def _read_json_with_sha256(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    raw = _read_regular_bytes(path, label=label)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        msg = f"{label} does not match its expected SHA-256: {path}"
        raise RevisionInputError(msg)
    return _parse_json_document(raw, path=path)


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory_fd(path, label="fsync directory")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_exclusive_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    """Atomically publish descriptor-relative names without replacement."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        rename = getattr(library, "renameatx_np", None)
        if rename is None:
            msg = "renameatx_np is unavailable; exclusive publication cannot proceed."
            raise RevisionInputError(msg)
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            source_bytes,
            destination_parent_fd,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        rename = getattr(library, "renameat2", None)
        if rename is None:
            msg = "renameat2 is unavailable; exclusive publication cannot proceed."
            raise RevisionInputError(msg)
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            source_bytes,
            destination_parent_fd,
            destination_bytes,
            0x00000001,
        )
    else:
        msg = (
            "Unsupported platform for atomic exclusive input publication: "
            f"{sys.platform!r}."
        )
        raise RevisionInputError(msg)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Exclusive input publication target already exists",
            destination_name,
        )
    raise OSError(
        error_number,
        "Atomic exclusive input publication failed",
        destination_name,
    )


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically publish paths without replacement using stable parent FDs."""
    source_parent = _open_directory_fd(source.parent, label="source parent")
    try:
        destination_parent = _open_directory_fd(
            destination.parent,
            label="destination parent",
        )
        try:
            _rename_exclusive_at(
                source_parent,
                source.name,
                destination_parent,
                destination.name,
            )
        finally:
            os.close(destination_parent)
    finally:
        os.close(source_parent)


def _write_chunks_atomic(path: Path, chunks: Iterable[bytes]) -> _FileReceipt:
    """Write one file through a dirfd and publish it without replacement."""
    parent_descriptor = _open_directory_fd(path.parent, label="artifact parent")
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    published = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            _STAGING_FILE_MODE,
            dir_fd=parent_descriptor,
        )
        digest = hashlib.sha256()
        size = 0
        try:
            before = _stable_file_state(descriptor, label="staged artifact")
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    msg = "Atomic artifact chunks must be exact bytes."
                    raise RevisionInputError(msg)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        msg = f"Short write while staging artifact: {path}"
                        raise RevisionInputError(msg)
                    digest.update(view[:written])
                    size += written
                    view = view[written:]
            os.fsync(descriptor)
            after = _stable_file_state(descriptor, label="staged artifact")
            if (
                before.device != after.device
                or before.inode != after.inode
                or after.size != size
                or after.links != 1
            ):
                msg = f"Staged artifact identity changed before publication: {path}"
                raise RevisionInputError(msg)
        finally:
            os.close(descriptor)
        _rename_exclusive_at(
            parent_descriptor,
            temporary_name,
            parent_descriptor,
            path.name,
        )
        os.fsync(parent_descriptor)
        published = True
        return _FileReceipt(bytes=size, sha256=digest.hexdigest())
    finally:
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _write_bytes_atomic(path: Path, content: bytes) -> _FileReceipt:
    return _write_chunks_atomic(path, (content,))


def _write_json_atomic(path: Path, payload: object) -> _FileReceipt:
    return _write_bytes_atomic(path, _canonical_json(payload) + b"\n")


def _copy_regular_file(source: Path, destination: Path) -> _FileReceipt:
    with _stable_regular_descriptor(source, label="source artifact") as (
        descriptor,
        state,
    ):
        receipt = _write_chunks_atomic(
            destination,
            _descriptor_chunks(descriptor),
        )
        if receipt.bytes != state.size:
            msg = f"Source artifact byte count changed while copying: {source}"
            raise RevisionInputError(msg)
        return receipt


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        msg = f"{label} must be exact nonblank text."
        raise RevisionInputError(msg)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = f"{label} must be a traversal-free relative POSIX path."
        raise RevisionInputError(msg)
    return path


def _verified_file_bytes(
    root: Path,
    value: object,
    *,
    label: str,
) -> tuple[Path, bytes]:
    if not isinstance(value, dict):
        msg = f"{label} must be a file-record object."
        raise RevisionInputError(msg)
    _require_exact_keys(value, {"bytes", "path", "sha256"}, label=label)
    relative = _safe_relative_path(value["path"], label=f"{label}.path")
    digest = value["sha256"]
    size = value["bytes"]
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        msg = f"{label}.sha256 must be lowercase SHA-256."
        raise RevisionInputError(msg)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        msg = f"{label}.bytes must be a nonnegative integer."
        raise RevisionInputError(msg)
    root = _absolute_unresolved(root)
    _require_directory(root, label=f"{label} bundle root")
    path = root.joinpath(*relative.parts)
    content = _read_regular_bytes(path, label=label)
    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        msg = f"{label} does not match its byte/hash receipt: {path}"
        raise RevisionInputError(msg)
    return path, content


def _verify_file_record(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        msg = f"{label} must be a file-record object."
        raise RevisionInputError(msg)
    _require_exact_keys(value, {"bytes", "path", "sha256"}, label=label)
    relative = _safe_relative_path(value["path"], label=f"{label}.path")
    digest = value["sha256"]
    size = value["bytes"]
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        msg = f"{label}.sha256 must be lowercase SHA-256."
        raise RevisionInputError(msg)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        msg = f"{label}.bytes must be a nonnegative integer."
        raise RevisionInputError(msg)
    root = _absolute_unresolved(root)
    _require_directory(root, label=f"{label} bundle root")
    path = root.joinpath(*relative.parts)
    observed_size, observed_digest = _regular_file_receipt(path, label=label)
    if observed_size != size or observed_digest != digest:
        msg = f"{label} does not match its byte/hash receipt: {path}"
        raise RevisionInputError(msg)
    return path


def _read_verified_json_file_record(
    root: Path,
    value: object,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path, content = _verified_file_bytes(root, value, label=label)
    return path, _parse_json_document(content, path=path)


def _git_bytes(git_dir: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _validate_datahub_git_dir(git_dir: Path) -> None:
    _require_directory(git_dir, label="DataHub Git directory")
    commit = (
        _git_bytes(
            git_dir,
            ["rev-parse", f"{TCGA_DATAHUB_COMMIT}^{{commit}}"],
        )
        .decode("ascii")
        .strip()
    )
    tree = (
        _git_bytes(
            git_dir,
            ["rev-parse", f"{TCGA_DATAHUB_COMMIT}^{{tree}}"],
        )
        .decode("ascii")
        .strip()
    )
    if commit != TCGA_DATAHUB_COMMIT or tree != TCGA_DATAHUB_TREE:
        msg = "DataHub Git directory lacks the exact frozen commit/tree pair."
        raise RevisionInputError(msg)


def _parse_git_lfs_pointer(content: bytes) -> dict[str, int | str]:
    """Parse one exact canonical three-line Git LFS v1 pointer."""
    if len(content) > _MAX_GIT_LFS_POINTER_BYTES:
        msg = "Git LFS pointer exceeds the frozen maximum byte length."
        raise RevisionInputError(msg)
    match = _GIT_LFS_POINTER_PATTERN.fullmatch(content)
    if match is None:
        msg = "Mutation Git blob is not an exact canonical three-line LFS pointer."
        raise RevisionInputError(msg)
    digest = match.group(1).decode("ascii")
    size = int(match.group(2))
    return {"bytes": size, "sha256": digest}


def _git_blob_receipt(
    git_dir: Path,
    repository_path: str,
    *,
    require_lfs_pointer: bool = False,
) -> dict[str, Any]:
    object_id = (
        _git_bytes(
            git_dir,
            ["rev-parse", f"{TCGA_DATAHUB_COMMIT}:{repository_path}"],
        )
        .decode("ascii")
        .strip()
    )
    object_type = _git_bytes(git_dir, ["cat-file", "-t", object_id]).decode().strip()
    if _GIT_OBJECT_ID_PATTERN.fullmatch(object_id) is None or object_type != "blob":
        msg = f"Pinned DataHub path is not an exact Git blob: {repository_path}"
        raise RevisionInputError(msg)
    expected_size_text = (
        _git_bytes(
            git_dir,
            ["cat-file", "-s", object_id],
        )
        .decode("ascii")
        .strip()
    )
    try:
        expected_size = int(expected_size_text)
    except ValueError as error:
        msg = f"Git blob has an invalid byte count: {repository_path}"
        raise RevisionInputError(msg) from error
    if require_lfs_pointer and expected_size > _MAX_GIT_LFS_POINTER_BYTES:
        msg = (
            "Pinned mutation path is a direct blob, not a Git LFS pointer: "
            f"{repository_path}"
        )
        raise RevisionInputError(msg)
    command = ["git", f"--git-dir={git_dir}", "cat-file", "blob", object_id]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        msg = f"Unable to stream pinned Git blob: {repository_path}"
        raise RevisionInputError(msg)
    digest = hashlib.sha256()
    observed_size = 0
    pointer_content = bytearray()
    for chunk in iter(lambda: process.stdout.read(_HASH_CHUNK_BYTES), b""):
        observed_size += len(chunk)
        digest.update(chunk)
        if require_lfs_pointer:
            pointer_content.extend(chunk)
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0 or observed_size != expected_size:
        msg = (
            f"Unable to attest pinned Git blob {repository_path}; "
            f"exit={return_code}, stderr={stderr.decode(errors='replace')!r}."
        )
        raise RevisionInputError(msg)
    receipt: dict[str, Any] = {
        "object_id": object_id,
        "bytes": observed_size,
        "sha256": digest.hexdigest(),
    }
    if require_lfs_pointer:
        receipt["lfs_payload"] = _parse_git_lfs_pointer(bytes(pointer_content))
    return receipt


def _validate_cohorts(cohorts: Sequence[str] | None) -> tuple[str, ...]:
    if cohorts is None:
        return TCGA_COHORTS
    selected = tuple(cohorts)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(cohort not in TCGA_COHORTS for cohort in selected)
    ):
        msg = "Cohorts must be unique exact members of the frozen TCGA family."
        raise RevisionInputError(msg)
    selected_set = set(selected)
    return tuple(cohort for cohort in TCGA_COHORTS if cohort in selected_set)


def _require_full_cohort_family(
    cohorts: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = TCGA_COHORTS if cohorts is None else tuple(cohorts)
    if selected != TCGA_COHORTS:
        msg = (
            "Production input publication requires the exact ordered 32-cohort family."
        )
        raise RevisionInputError(msg)
    return TCGA_COHORTS


def _population_selection_policy() -> dict[str, object]:
    return {
        "analysis_unit": "one-participant-one-tumor-sample",
        "membership_source": "commit-matched-sequenced-case-list",
        "primary_sample_type_codes": sorted(
            tcga_data.PRIMARY_DISEASE_SAMPLE_TYPE_CODES,
        ),
        "singleton_rule": "retain-sole-case-list-sample-regardless-of-sample-type",
        "repeated_participant_rule": (
            "retain-exactly-one-primary-disease-sample-otherwise-fail-closed"
        ),
        "ordering": "lexicographic-sample-barcode",
        "ordered_axis_digest": "sha256-uint64be-length-framed-utf8-v1",
    }


def _module_source_path(module: object, *, label: str) -> Path:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        msg = f"Cannot locate the {label} source file."
        raise RevisionInputError(msg)
    return _absolute_unresolved(source)


def _tcga_source_path() -> Path:
    return _module_source_path(tcga_data, label="TCGA contract")


def _approval_source_path() -> Path:
    return _module_source_path(approval_data, label="approval validator")


def _population_materializer_source_path() -> Path:
    return _absolute_unresolved(
        Path(__file__).with_name("materialize_tcga_revision_population.py"),
    )


def _validate_population_cohort(
    population_root: Path,
    cohort: str,
    expected_manifest_sha256: str,
    *,
    closed_sources: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    cohort_dir = population_root / cohort
    _require_directory(cohort_dir, label=f"{cohort} population cohort directory")
    manifest_path = cohort_dir / "population_manifest.json"
    manifest = _read_json_with_sha256(
        manifest_path,
        expected_manifest_sha256,
        label=f"{cohort} population cohort manifest",
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "contract",
            "cohort",
            "source",
            "population",
            "selection_policy",
            "contract_source",
            "outputs",
        },
        label=f"{cohort} population manifest",
    )
    if (
        manifest.get("schema_version") != POPULATION_SCHEMA_VERSION
        or manifest.get("contract") != POPULATION_CONTRACT
        or manifest.get("cohort") != cohort
    ):
        msg = f"Population cohort contract is invalid: {cohort}"
        raise RevisionInputError(msg)
    source = manifest.get("source")
    population = manifest.get("population")
    selection_policy = manifest.get("selection_policy")
    contract_source = manifest.get("contract_source")
    outputs = manifest.get("outputs")
    if (
        not isinstance(source, dict)
        or not isinstance(population, dict)
        or not isinstance(selection_policy, dict)
        or not isinstance(contract_source, dict)
        or not isinstance(outputs, dict)
    ):
        msg = f"Population cohort manifest is structurally incomplete: {cohort}"
        raise RevisionInputError(msg)
    _require_exact_keys(
        source,
        {
            "repository",
            "commit",
            "tree",
            "repository_path",
            "case_list_sha256",
            "case_list_bytes",
        },
        label=f"{cohort} population source",
    )
    _require_exact_keys(
        population,
        {
            "source_sample_count",
            "selected_sample_count",
            "participant_count",
            "removed_repeat_participant_samples",
            "ordered_sample_axis_sha256",
            "lexicographically_ordered",
            "all_zero_rows_required_for_samples_without_retained_events",
        },
        label=f"{cohort} population counts",
    )
    _require_exact_keys(
        outputs,
        {"sample_axis"},
        label=f"{cohort} population outputs",
    )
    receipt = TCGA_CASE_LIST_RECEIPTS[cohort]
    common_source = {
        "repository": "https://github.com/cBioPortal/datahub",
        "commit": TCGA_DATAHUB_COMMIT,
        "tree": TCGA_DATAHUB_TREE,
        "repository_path": tcga_datahub_case_list_path(cohort).as_posix(),
    }
    if (
        any(source.get(key) != value for key, value in common_source.items())
        or not isinstance(source["case_list_sha256"], str)
        or _SHA256_PATTERN.fullmatch(source["case_list_sha256"]) is None
        or not isinstance(source["case_list_bytes"], int)
        or isinstance(source["case_list_bytes"], bool)
        or source["case_list_bytes"] <= 0
    ):
        msg = f"Population source receipt is invalid: {cohort}"
        raise RevisionInputError(msg)
    expected_population = {
        "source_sample_count": receipt.sample_count,
        "selected_sample_count": receipt.participant_count,
        "participant_count": receipt.participant_count,
        "removed_repeat_participant_samples": (
            receipt.sample_count - receipt.participant_count
        ),
        "ordered_sample_axis_sha256": (
            tcga_data.TCGA_SELECTED_SAMPLE_AXIS_SHA256[cohort]
        ),
        "lexicographically_ordered": True,
        "all_zero_rows_required_for_samples_without_retained_events": True,
    }
    if not closed_sources and (
        source["case_list_sha256"] != receipt.sha256
        or not _exact_json_equal(population, expected_population)
    ):
        msg = f"Population source/count/axis receipt is invalid: {cohort}"
        raise RevisionInputError(msg)
    if not _exact_json_equal(selection_policy, _population_selection_policy()):
        msg = f"Population selection policy is invalid: {cohort}"
        raise RevisionInputError(msg)
    if closed_sources:
        count_fields = (
            "source_sample_count",
            "selected_sample_count",
            "participant_count",
            "removed_repeat_participant_samples",
        )
        if any(
            not isinstance(population[field], int)
            or isinstance(population[field], bool)
            or population[field] < 0
            for field in count_fields
        ) or (
            population["source_sample_count"]
            != population["selected_sample_count"]
            + population["removed_repeat_participant_samples"]
            or population["selected_sample_count"] != population["participant_count"]
            or population["lexicographically_ordered"] is not True
            or population["all_zero_rows_required_for_samples_without_retained_events"]
            is not True
            or not isinstance(population["ordered_sample_axis_sha256"], str)
            or _SHA256_PATTERN.fullmatch(
                population["ordered_sample_axis_sha256"],
            )
            is None
        ):
            msg = f"Closed population accounting is invalid: {cohort}"
            raise RevisionInputError(msg)
    if closed_sources:
        source_path = _verify_file_record(
            population_root,
            contract_source,
            label=f"{cohort} closed population contract source",
        )
        if source_path != population_root / "src/dialect/data/tcga.py":
            msg = f"Population contract-source path is invalid: {cohort}"
            raise RevisionInputError(msg)
    elif contract_source != _file_record(
        _tcga_source_path(),
        display_path="src/dialect/data/tcga.py",
    ):
        msg = f"Population contract-source receipt is invalid: {cohort}"
        raise RevisionInputError(msg)
    axis_record = outputs.get("sample_axis")
    if not isinstance(axis_record, dict):
        msg = f"Population sample-axis receipt is missing: {cohort}"
        raise RevisionInputError(msg)
    expected_path = f"{cohort}/sample_axis.txt"
    if axis_record.get("path") != expected_path:
        msg = f"Population sample-axis path is invalid: {cohort}"
        raise RevisionInputError(msg)
    axis_path, axis_bytes = _verified_file_bytes(
        population_root,
        axis_record,
        label=f"{cohort} population sample axis",
    )
    if axis_path != cohort_dir / "sample_axis.txt":
        msg = f"Population sample-axis file path is invalid: {cohort}"
        raise RevisionInputError(msg)
    try:
        samples = tuple(axis_bytes.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        msg = f"Population sample axis is not UTF-8: {cohort}"
        raise RevisionInputError(msg) from error
    if (
        len(samples) != population["selected_sample_count"]
        or len(samples) != len(set(samples))
        or list(samples) != sorted(samples)
        or any(not sample or sample != sample.strip() for sample in samples)
        or _sequence_sha256(samples) != population["ordered_sample_axis_sha256"]
        or axis_bytes != ("\n".join(samples) + "\n").encode("utf-8")
    ):
        msg = f"Population sample-axis semantics are invalid: {cohort}"
        raise RevisionInputError(msg)
    return manifest, samples


def _validate_population_bundle(
    population_root: Path,
    selected_cohorts: Sequence[str],
    *,
    closed_sources: bool,
    expected_manifest_sha256: str | None = None,
    root_manifest_bytes: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    _require_directory(population_root, label="population root")
    root_path = population_root / "population_manifest.json"
    if root_manifest_bytes is None:
        manifest = (
            _read_json(root_path)
            if expected_manifest_sha256 is None
            else _read_json_with_sha256(
                root_path,
                expected_manifest_sha256,
                label="population root manifest",
            )
        )
    else:
        if (
            expected_manifest_sha256 is not None
            and hashlib.sha256(root_manifest_bytes).hexdigest()
            != expected_manifest_sha256
        ):
            msg = "Population root manifest buffer differs from its receipt."
            raise RevisionInputError(msg)
        manifest = _parse_json_document(root_manifest_bytes, path=root_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "contract",
            "source",
            "selection_policy",
            "contract_source",
            "cohorts",
            "cohort_count",
            "cohort_manifests",
            "totals",
            "generator",
        },
        label="population root manifest",
    )
    if (
        manifest.get("schema_version") != POPULATION_SCHEMA_VERSION
        or manifest.get("contract") != POPULATION_CONTRACT
    ):
        msg = "Population root manifest does not match the frozen contract."
        raise RevisionInputError(msg)
    source = manifest.get("source")
    totals = manifest.get("totals")
    selection_policy = manifest.get("selection_policy")
    if (
        not isinstance(source, dict)
        or not isinstance(totals, dict)
        or not isinstance(selection_policy, dict)
    ):
        msg = "Population root source, policy, and totals must be objects."
        raise RevisionInputError(msg)
    _require_exact_keys(
        source,
        {"repository", "commit", "tree"},
        label="population root source",
    )
    _require_exact_keys(
        totals,
        {
            "source_sample_count",
            "selected_sample_count",
            "participant_count",
            "removed_repeat_participant_samples",
        },
        label="population root totals",
    )
    expected_source = {
        "repository": "https://github.com/cBioPortal/datahub",
        "commit": TCGA_DATAHUB_COMMIT,
        "tree": TCGA_DATAHUB_TREE,
    }
    if not _exact_json_equal(source, expected_source) or not _exact_json_equal(
        selection_policy,
        _population_selection_policy(),
    ):
        msg = "Population root source or selection policy is invalid."
        raise RevisionInputError(msg)
    if closed_sources:
        contract_source_path = _verify_file_record(
            population_root,
            manifest["contract_source"],
            label="closed population contract source",
        )
        generator_path = _verify_file_record(
            population_root,
            manifest["generator"],
            label="closed population generator",
        )
        if (
            contract_source_path != population_root / "src/dialect/data/tcga.py"
            or generator_path
            != population_root / "analysis/materialize_tcga_revision_population.py"
        ):
            msg = "Population root implementation paths are invalid."
            raise RevisionInputError(msg)
    elif manifest.get("contract_source") != _file_record(
        _tcga_source_path(),
        display_path="src/dialect/data/tcga.py",
    ) or manifest.get("generator") != _file_record(
        _population_materializer_source_path(),
        display_path="analysis/materialize_tcga_revision_population.py",
    ):
        msg = "Population root implementation receipts are invalid."
        raise RevisionInputError(msg)
    cohorts = manifest.get("cohorts")
    records = manifest.get("cohort_manifests")
    if not isinstance(cohorts, list) or not isinstance(records, list):
        msg = "Population root manifest lacks cohort records."
        raise RevisionInputError(msg)
    if cohorts != [
        record.get("cohort") for record in records if isinstance(record, dict)
    ]:
        msg = "Population root cohort and manifest-record order differ."
        raise RevisionInputError(msg)
    if (
        type(manifest.get("cohort_count")) is not int
        or manifest.get("cohort_count") != len(cohorts)
        or len(cohorts) != len(set(cohorts))
    ):
        msg = "Population root cohort count is invalid."
        raise RevisionInputError(msg)
    if cohorts != list(selected_cohorts):
        msg = "Population root cohort family differs from the requested exact family."
        raise RevisionInputError(msg)
    record_by_cohort = {
        str(record["cohort"]): record for record in records if isinstance(record, dict)
    }
    axes: dict[str, tuple[str, ...]] = {}
    child_manifests: list[dict[str, Any]] = []
    for cohort in selected_cohorts:
        record = record_by_cohort[cohort]
        if set(record) != {"cohort", "manifest_sha256"}:
            msg = f"Population root cohort record is invalid: {cohort}"
            raise RevisionInputError(msg)
        digest = record["manifest_sha256"]
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            msg = f"Population cohort-manifest digest is invalid: {cohort}"
            raise RevisionInputError(msg)
        child_manifest, axes[cohort] = _validate_population_cohort(
            population_root,
            cohort,
            digest,
            closed_sources=closed_sources,
        )
        child_manifests.append(child_manifest)
    expected_totals = {
        "source_sample_count": sum(
            child["population"]["source_sample_count"] for child in child_manifests
        ),
        "selected_sample_count": sum(
            child["population"]["selected_sample_count"] for child in child_manifests
        ),
        "participant_count": sum(
            child["population"]["participant_count"] for child in child_manifests
        ),
        "removed_repeat_participant_samples": sum(
            child["population"]["removed_repeat_participant_samples"]
            for child in child_manifests
        ),
    }
    if not _exact_json_equal(manifest.get("totals"), expected_totals):
        msg = "Population root totals do not equal the verified cohort family."
        raise RevisionInputError(msg)
    return manifest, axes


def _serialize_maf_for_test(frame: pd.DataFrame) -> bytes:
    """Return the frozen pandas oracle bytes used only by equivalence tests."""
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        sep="\t",
        index=False,
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def _column_axis_sha256(columns: Sequence[object]) -> str:
    return _sequence_sha256(str(column) for column in columns)


def _json_scalar(value: object) -> object:
    """Return the exact JSON scalar used to persist one normalized cell."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        msg = f"Normalized MAF cell is not a JSON scalar: {type(value).__name__}"
        raise RevisionInputError(msg)
    if isinstance(value, float) and not math.isfinite(value):
        msg = "Normalized MAF cell is non-finite."
        raise RevisionInputError(msg)
    return value


def _decode_row_blob(blob: object, *, columns: Sequence[str]) -> list[object]:
    if not isinstance(blob, bytes):
        msg = "SQLite canonical row payload is not exact bytes."
        raise RevisionInputError(msg)
    parsed = _parse_json_document(b'{"row":' + blob + b"}\n", path=Path("row"))
    row = parsed.get("row")
    if not isinstance(row, list) or len(row) != len(columns):
        msg = "SQLite canonical row payload violates its column axis."
        raise RevisionInputError(msg)
    return row


def _row_blob(values: Sequence[object]) -> bytes:
    return _canonical_json([_json_scalar(value) for value in values])


def _configure_streaming_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "CREATE TABLE raw_rows (ordinal INTEGER PRIMARY KEY, row_blob BLOB NOT NULL)",
    )
    connection.execute(
        """CREATE TABLE normalized_rows (
        ordinal INTEGER PRIMARY KEY,
        sample TEXT NOT NULL COLLATE BINARY,
        chromosome_number INTEGER NOT NULL,
        start_position INTEGER NOT NULL,
        reference TEXT NOT NULL COLLATE BINARY,
        alternate TEXT NOT NULL COLLATE BINARY,
        row_token TEXT NOT NULL COLLATE BINARY,
        effective_newbase TEXT NOT NULL COLLATE BINARY,
        row_blob BLOB NOT NULL
        )""",
    )
    connection.execute(
        """CREATE INDEX normalized_order ON normalized_rows (
        sample COLLATE BINARY, chromosome_number, start_position,
        reference COLLATE BINARY, alternate COLLATE BINARY,
        row_token COLLATE BINARY, ordinal
        )""",
    )
    connection.execute(
        """CREATE TABLE canonical_rows (
        sample TEXT NOT NULL COLLATE BINARY,
        chromosome_number INTEGER NOT NULL,
        start_position INTEGER NOT NULL,
        reference TEXT NOT NULL COLLATE BINARY,
        alternate TEXT NOT NULL COLLATE BINARY,
        row_token TEXT NOT NULL COLLATE BINARY,
        row_blob BLOB NOT NULL
        )""",
    )
    connection.execute(
        """CREATE UNIQUE INDEX canonical_key ON canonical_rows (
        sample COLLATE BINARY, chromosome_number, start_position,
        reference COLLATE BINARY, alternate COLLATE BINARY
        )""",
    )


def _normalize_stream_chunk(
    frame: pd.DataFrame,
    *,
    frozen_canonicalizer: object,
    supplied_newbase_is_globally_complete: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Apply the signed private normalization helpers to one bounded chunk."""
    canonical = frame.copy().reset_index(drop=True)
    normalize_samples = getattr(frozen_canonicalizer, "_normalize_samples", None)
    chromosome_number = getattr(frozen_canonicalizer, "_chromosome_number", None)
    chromosome_label = getattr(frozen_canonicalizer, "_chromosome_label", None)
    normalize_positions = getattr(frozen_canonicalizer, "_normalize_positions", None)
    normalize_alleles = getattr(frozen_canonicalizer, "_normalize_alleles", None)
    validate_alternate = getattr(
        frozen_canonicalizer,
        "_validate_effective_alternate",
        None,
    )
    validate_snv = getattr(frozen_canonicalizer, "_validate_snv_end_positions", None)
    helpers = (
        normalize_samples,
        chromosome_number,
        chromosome_label,
        normalize_positions,
        normalize_alleles,
        validate_alternate,
        validate_snv,
    )
    if not all(callable(helper) for helper in helpers):
        msg = "Frozen canonicalizer lacks the signed streaming helper closure."
        raise RevisionInputError(msg)
    canonical["Tumor_Sample_Barcode"] = normalize_samples(
        canonical["Tumor_Sample_Barcode"],
    )
    chromosome_numbers = canonical["Chromosome"].map(chromosome_number)
    if chromosome_numbers.isna().any():
        msg = "Chromosome contains a value outside chromosomes 1-22, X, and Y."
        raise ValueError(msg)
    chromosome_numbers = chromosome_numbers.astype("int64")
    canonical["Chromosome"] = chromosome_numbers.map(chromosome_label)
    canonical["Start_Position"] = normalize_positions(
        canonical["Start_Position"],
        column="Start_Position",
    )
    canonical["End_Position"] = normalize_positions(
        canonical["End_Position"],
        column="End_Position",
    )
    for column in ("Reference_Allele", "Tumor_Seq_Allele2"):
        canonical[column] = normalize_alleles(canonical[column], column=column)
    canonical["Tumor_Seq_Allele1"] = normalize_alleles(
        canonical["Tumor_Seq_Allele1"],
        column="Tumor_Seq_Allele1",
    )
    if "newbase" in canonical and supplied_newbase_is_globally_complete:
        canonical["newbase"] = normalize_alleles(
            canonical["newbase"],
            column="newbase",
        )
        effective_newbases = canonical["newbase"]
    else:
        allele1 = canonical["Tumor_Seq_Allele1"]
        effective_newbases = canonical["Tumor_Seq_Allele2"].where(
            canonical["Reference_Allele"] == allele1,
            allele1,
        )
        if "newbase" in canonical:
            canonical["newbase"] = effective_newbases
    validate_alternate(canonical, effective_newbases)
    validate_snv(canonical)
    return canonical, chromosome_numbers, effective_newbases


def _parse_selected_rows_to_sqlite(  # noqa: PLR0913
    descriptor: int,
    connection: sqlite3.Connection,
    *,
    path: Path,
    selected_samples: frozenset[str],
    case_samples: frozenset[str],
    frozen_canonicalizer: object,
) -> tuple[list[str], int, int, bool]:
    """Parse a MAF once and persist only selected raw rows to disk."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    duplicate = os.dup(descriptor)
    raw_rows = 0
    selected_rows = 0
    supplied_newbase_complete = True
    try:
        with os.fdopen(duplicate, "rb") as binary:  # noqa: SIM117
            with io.TextIOWrapper(
                binary,
                encoding="utf-8",
                errors="strict",
                newline="",
            ) as text:
                reader = csv.reader(text, delimiter="\t")
                try:
                    header = next(reader)
                except StopIteration as error:
                    msg = f"Raw MAF is empty: {path}"
                    raise RevisionInputError(msg) from error
                if (
                    not header
                    or header[0] != "Hugo_Symbol"
                    or len(header) != len(set(header))
                ):
                    msg = (
                        "Raw MAF header is missing, duplicated, or noncanonical: "
                        f"{path}"
                    )
                    raise RevisionInputError(msg)
                validate_columns = getattr(
                    frozen_canonicalizer,
                    "_validate_columns",
                    None,
                )
                if not callable(validate_columns):
                    msg = "Frozen canonicalizer lacks its signed column validator."
                    raise RevisionInputError(msg)
                validate_columns(pd.DataFrame(columns=header))
                sample_index = header.index("Tumor_Sample_Barcode")
                newbase_index = header.index("newbase") if "newbase" in header else None
                pending: list[tuple[int, bytes]] = []
                for row in reader:
                    raw_rows += 1
                    if len(row) != len(header):
                        msg = (
                            f"Raw MAF row {raw_rows + 1} violates its column axis: "
                            f"{path}"
                        )
                        raise RevisionInputError(msg)
                    sample = row[sample_index]
                    if not sample or sample != sample.strip():
                        msg = (
                            "Raw MAF contains blank or padded tumor identifiers: "
                            f"{path}"
                        )
                        raise RevisionInputError(msg)
                    if sample not in case_samples:
                        msg = (
                            "Raw MAF contains a sample outside the frozen case list: "
                            f"{path}"
                        )
                        raise RevisionInputError(msg)
                    if sample not in selected_samples:
                        continue
                    selected_rows += 1
                    if newbase_index is not None and not row[newbase_index].strip():
                        supplied_newbase_complete = False
                    pending.append((selected_rows, _row_blob(row)))
                    if len(pending) >= _STREAM_CHUNK_ROWS:
                        connection.executemany(
                            "INSERT INTO raw_rows(ordinal,row_blob) VALUES (?,?)",
                            pending,
                        )
                        pending.clear()
                if pending:
                    connection.executemany(
                        "INSERT INTO raw_rows(ordinal,row_blob) VALUES (?,?)",
                        pending,
                    )
    except UnicodeDecodeError as error:
        msg = f"Raw MAF is not valid UTF-8: {path}"
        raise RevisionInputError(msg) from error
    if os.lseek(descriptor, 0, os.SEEK_CUR) < 0:
        msg = f"Raw MAF descriptor position is invalid after parsing: {path}"
        raise RevisionInputError(msg)
    if raw_rows == 0:
        msg = f"Raw MAF contains no mutation rows: {path}"
        raise RevisionInputError(msg)
    if selected_rows == 0:
        msg = f"Selected population has no MAF rows: {path}"
        raise RevisionInputError(msg)
    return header, raw_rows, selected_rows, supplied_newbase_complete


def _normalize_sqlite_rows(
    connection: sqlite3.Connection,
    *,
    columns: Sequence[str],
    frozen_canonicalizer: object,
    supplied_newbase_is_globally_complete: bool,
) -> None:
    row_sort_token = getattr(frozen_canonicalizer, "_row_sort_token", None)
    if not callable(row_sort_token):
        msg = "Frozen canonicalizer lacks its signed framed-row-token helper."
        raise RevisionInputError(msg)
    cursor = connection.execute(
        "SELECT ordinal,row_blob FROM raw_rows ORDER BY ordinal",
    )
    token_columns = sorted(columns)
    while True:
        records = cursor.fetchmany(_STREAM_CHUNK_ROWS)
        if not records:
            break
        ordinals = [int(record[0]) for record in records]
        rows = [_decode_row_blob(record[1], columns=columns) for record in records]
        frame = pd.DataFrame(rows, columns=columns, dtype=object)
        canonical, chromosome_numbers, effective_newbases = _normalize_stream_chunk(
            frame,
            frozen_canonicalizer=frozen_canonicalizer,
            supplied_newbase_is_globally_complete=(
                supplied_newbase_is_globally_complete
            ),
        )
        tokens = canonical[token_columns].apply(row_sort_token, axis="columns")
        inserts: list[tuple[object, ...]] = []
        for offset, ordinal in enumerate(ordinals):
            row = canonical.iloc[offset]
            inserts.append(
                (
                    ordinal,
                    str(row["Tumor_Sample_Barcode"]),
                    int(chromosome_numbers.iloc[offset]),
                    int(row["Start_Position"]),
                    str(row["Reference_Allele"]),
                    str(row["Tumor_Seq_Allele2"]),
                    str(tokens.iloc[offset]),
                    str(effective_newbases.iloc[offset]),
                    _row_blob([row[column] for column in columns]),
                ),
            )
        connection.executemany(
            """INSERT INTO normalized_rows(
            ordinal,sample,chromosome_number,start_position,reference,alternate,
            row_token,effective_newbase,row_blob
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            inserts,
        )
        connection.commit()


def _resolve_sqlite_groups(
    connection: sqlite3.Connection,
    *,
    columns: Sequence[str],
    frozen_canonicalizer: object,
) -> object:
    resolve_group = getattr(frozen_canonicalizer, "_resolve_duplicate_group", None)
    effective_column = getattr(
        frozen_canonicalizer,
        "_EFFECTIVE_NEWBASE_COLUMN",
        None,
    )
    audit_class = getattr(frozen_canonicalizer, "VariantResolutionAudit", None)
    policy = getattr(frozen_canonicalizer, "TCGA_DUPLICATE_RESOLUTION_POLICY", None)
    if (
        not callable(resolve_group)
        or not isinstance(effective_column, str)
        or audit_class is None
        or policy is None
    ):
        msg = "Frozen canonicalizer lacks its signed duplicate resolver closure."
        raise RevisionInputError(msg)
    query = """SELECT sample,chromosome_number,start_position,reference,alternate,
        row_token,effective_newbase,row_blob
        FROM normalized_rows
        ORDER BY sample COLLATE BINARY,chromosome_number,start_position,
        reference COLLATE BINARY,alternate COLLATE BINARY,
        row_token COLLATE BINARY,ordinal"""
    cursor = connection.execute(query)
    current_key: tuple[object, ...] | None = None
    group: list[tuple[object, ...]] = []
    group_bytes = 0
    duplicate_groups = 0
    semantic_agreement_groups = 0
    ignored_conflict_groups = 0
    effect_resolution_groups = 0
    conflict_counts: Counter[str] = Counter()
    effect_counts: Counter[str] = Counter()
    output_rows = 0

    def emit_group(records: Sequence[tuple[object, ...]]) -> None:
        nonlocal duplicate_groups
        nonlocal semantic_agreement_groups
        nonlocal ignored_conflict_groups
        nonlocal effect_resolution_groups
        nonlocal output_rows
        if not records:
            return
        key = records[0][:5]
        if len(records) == 1:
            row_blob = records[0][7]
            row_token = records[0][5]
        else:
            duplicate_groups += 1
            decoded = [
                _decode_row_blob(record[7], columns=columns) for record in records
            ]
            duplicate_frame = pd.DataFrame(decoded, columns=columns)
            duplicate_frame[effective_column] = [record[6] for record in records]
            representative, resolution = resolve_group(
                duplicate_frame,
                original_columns=list(columns),
            )
            conflict_counts.update(resolution.ignored_conflicts)
            conflict_counts.update(resolution.effect_conflicts)
            if resolution.ignored_conflicts:
                ignored_conflict_groups += 1
            if resolution.selected_mutsig_effect is not None:
                effect_resolution_groups += 1
                effect_counts.update([resolution.selected_mutsig_effect])
            if not resolution.ignored_conflicts and not resolution.effect_conflicts:
                semantic_agreement_groups += 1
            representative_row = representative.iloc[0]
            row_blob = _row_blob([representative_row[column] for column in columns])
            row_sort_token = frozen_canonicalizer._row_sort_token  # noqa: SLF001
            row_token = str(
                representative[sorted(columns)]
                .apply(
                    row_sort_token,
                    axis="columns",
                )
                .iloc[0],
            )
        connection.execute(
            """INSERT INTO canonical_rows(
            sample,chromosome_number,start_position,reference,alternate,row_token,row_blob
            ) VALUES (?,?,?,?,?,?,?)""",
            (*key, row_token, row_blob),
        )
        output_rows += 1

    while True:
        records = cursor.fetchmany(_SQLITE_FETCH_ROWS)
        if not records:
            break
        for record in records:
            key = tuple(record[:5])
            if current_key is not None and key != current_key:
                emit_group(group)
                group = []
                group_bytes = 0
            current_key = key
            group.append(tuple(record))
            group_bytes += len(record[7]) + len(record[5]) + len(record[6])
            if (
                len(group) > _MAX_DUPLICATE_GROUP_ROWS
                or group_bytes > _MAX_DUPLICATE_GROUP_BYTES
            ):
                msg = "Duplicate full-variant group exceeds the bounded resolver limit."
                raise RevisionInputError(msg)
    emit_group(group)
    connection.commit()
    input_rows = int(
        connection.execute("SELECT COUNT(*) FROM normalized_rows").fetchone()[0],
    )
    return audit_class(
        policy_version=policy.version,
        input_row_count=input_rows,
        output_row_count=output_rows,
        duplicate_group_count=duplicate_groups,
        collapsed_row_count=input_rows - output_rows,
        semantic_agreement_group_count=semantic_agreement_groups,
        ignored_conflict_group_count=ignored_conflict_groups,
        frozen_effect_resolution_group_count=effect_resolution_groups,
        resolved_conflict_groups_by_column=tuple(sorted(conflict_counts.items())),
        selected_mutsig_effect_groups=tuple(sorted(effect_counts.items())),
    )


def _canonical_tsv_chunks(
    connection: sqlite3.Connection,
    *,
    columns: Sequence[str],
) -> Iterable[bytes]:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    yield buffer.getvalue().encode("utf-8")
    query = """SELECT row_blob FROM canonical_rows
        ORDER BY sample COLLATE BINARY,chromosome_number,start_position,
        reference COLLATE BINARY,alternate COLLATE BINARY"""
    cursor = connection.execute(query)
    while True:
        records = cursor.fetchmany(_SQLITE_FETCH_ROWS)
        if not records:
            return
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        for (blob,) in records:
            writer.writerow(_decode_row_blob(blob, columns=columns))
        yield buffer.getvalue().encode("utf-8")


def _streaming_multiallelic_groups(connection: sqlite3.Connection) -> int:
    query = """SELECT COUNT(*) FROM (
        SELECT sample,chromosome_number,start_position FROM (
            SELECT DISTINCT sample,chromosome_number,start_position,
                reference,alternate FROM canonical_rows
        ) GROUP BY sample,chromosome_number,start_position HAVING COUNT(*) > 1
    )"""
    return int(connection.execute(query).fetchone()[0])


def _stream_canonicalize_maf(  # noqa: PLR0913
    raw_path: Path,
    canonical_path: Path,
    sqlite_path: Path,
    *,
    raw_copy_path: Path | None,
    expected_raw_receipt: _FileReceipt,
    selected_samples: frozenset[str],
    case_samples: frozenset[str],
    frozen_canonicalizer: object,
) -> _StreamingCanonicalization:
    """Canonicalize one MAF with bounded RAM and disk-backed deterministic order."""
    with _stable_regular_descriptor(raw_path, label="raw MAF") as (
        descriptor,
        state,
    ):
        if raw_copy_path is None:
            raw_receipt = _consume_descriptor_receipt(descriptor)
        else:
            raw_receipt = _write_chunks_atomic(
                raw_copy_path,
                _descriptor_chunks(descriptor),
            )
        if raw_receipt.bytes != state.size or raw_receipt != expected_raw_receipt:
            msg = f"Raw MAF does not match its pinned byte/hash receipt: {raw_path}"
            raise RevisionInputError(msg)
        connection = sqlite3.connect(sqlite_path)
        try:
            _configure_streaming_database(connection)
            columns, raw_rows, selected_rows, globally_complete = (
                _parse_selected_rows_to_sqlite(
                    descriptor,
                    connection,
                    path=raw_path,
                    selected_samples=selected_samples,
                    case_samples=case_samples,
                    frozen_canonicalizer=frozen_canonicalizer,
                )
            )
            if os.lseek(descriptor, 0, os.SEEK_CUR) != state.size:
                msg = f"Raw MAF parser did not consume every verified byte: {raw_path}"
                raise RevisionInputError(msg)
            connection.commit()
            _normalize_sqlite_rows(
                connection,
                columns=columns,
                frozen_canonicalizer=frozen_canonicalizer,
                supplied_newbase_is_globally_complete=globally_complete,
            )
            audit = _resolve_sqlite_groups(
                connection,
                columns=columns,
                frozen_canonicalizer=frozen_canonicalizer,
            )
            multiallelic_groups = _streaming_multiallelic_groups(connection)
            output_receipt = _write_chunks_atomic(
                canonical_path,
                _canonical_tsv_chunks(connection, columns=columns),
            )
        finally:
            connection.close()
    return _StreamingCanonicalization(
        raw_rows=raw_rows,
        selected_rows=selected_rows,
        output_rows=int(audit.output_row_count),
        multiallelic_coordinate_groups=multiallelic_groups,
        audit=audit,
        output_receipt=output_receipt,
        ordered_columns_sha256=_column_axis_sha256(columns),
    )


def _canonicalizer_source_path() -> Path:
    return _module_source_path(variant_data, label="canonicalizer")


def _materializer_source_path() -> Path:
    return _absolute_unresolved(__file__)


def _source_dependencies() -> dict[str, Path]:
    return {
        "materializer": _materializer_source_path(),
        "population_materializer": _population_materializer_source_path(),
        "canonicalizer": _canonicalizer_source_path(),
        "tcga": _tcga_source_path(),
        "revision_approval": _approval_source_path(),
    }


def _distribution_record_path(package: str) -> Path:
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError as error:
        msg = f"Required runtime distribution is unavailable: {package}"
        raise RevisionInputError(msg) from error
    candidates = [
        file
        for file in distribution.files or ()
        if PurePosixPath(str(file)).name == "RECORD"
        and ".dist-info" in PurePosixPath(str(file)).parent.name
    ]
    if len(candidates) != 1:
        msg = f"Runtime distribution has no unique RECORD snapshot: {package}"
        raise RevisionInputError(msg)
    return _absolute_unresolved(distribution.locate_file(candidates[0]))


def _distribution_runtime_closure(package: str) -> dict[str, Any]:
    """Hash the finite Python/native file closure declared by one distribution."""
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError as error:
        msg = f"Required runtime distribution is unavailable: {package}"
        raise RevisionInputError(msg) from error
    records: list[dict[str, int | str]] = []
    for entry in sorted(distribution.files or (), key=str):
        candidate = str(entry).replace(os.sep, "/")
        lower = candidate.lower()
        if not (
            lower.endswith(_RUNTIME_CLOSURE_SUFFIXES)
            or PurePosixPath(candidate).name == "RECORD"
        ):
            continue
        text = _safe_relative_path(
            candidate,
            label=f"{package} runtime distribution path",
        ).as_posix()
        path = _absolute_unresolved(distribution.locate_file(entry))
        size, digest = _regular_file_receipt(
            path,
            label=f"{package} runtime closure file {text}",
        )
        records.append({"distribution_path": text, "bytes": size, "sha256": digest})
    if not records:
        msg = f"Runtime distribution closure is empty: {package}"
        raise RevisionInputError(msg)
    return {
        "package": package,
        "version": importlib.metadata.version(package),
        "files": records,
    }


def _binary_receipt(path: Path, *, label: str) -> dict[str, int | str]:
    size, digest = _regular_file_receipt(path, label=label)
    return {"bytes": size, "sha256": digest}


def _git_executable_path() -> Path:
    executable = shutil.which("git")
    if executable is None:
        msg = "The Git executable is unavailable."
        raise RevisionInputError(msg)
    return _absolute_unresolved(Path(executable).resolve())


def _python_executable_path() -> Path:
    return _absolute_unresolved(Path(sys.executable).resolve())


def _runtime_identity() -> dict[str, Any]:
    package_records: dict[str, dict[str, object]] = {}
    for package in _PACKAGE_NAMES:
        record_path = _distribution_record_path(package)
        size, digest = _regular_file_receipt(
            record_path,
            label=f"{package} distribution RECORD",
        )
        closure = _distribution_runtime_closure(package)
        closure_bytes = _canonical_json(closure) + b"\n"
        package_records[package] = {
            "version": importlib.metadata.version(package),
            "record_bytes": size,
            "record_sha256": digest,
            "closure_files": len(closure["files"]),
            "closure_sha256": hashlib.sha256(closure_bytes).hexdigest(),
        }
    git_version = subprocess.run(
        [str(_git_executable_path()), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": _binary_receipt(
                _python_executable_path(),
                label="Python executable",
            ),
        },
        "packages": package_records,
        "git": {
            "version": git_version,
            "executable": _binary_receipt(
                _git_executable_path(),
                label="Git executable",
            ),
        },
    }


def _execution_snapshot_payload(
    source_bytes: Mapping[str, bytes],
    *,
    require_live_runtime: bool,
) -> dict[str, Any]:
    if set(source_bytes) != set(_SOURCE_SNAPSHOT_PATHS):
        msg = "Execution snapshot source-byte closure is incomplete."
        raise RevisionInputError(msg)
    return {
        "schema": "dialect-revision-isolated-execution-snapshot-v1",
        "sources": {
            name: {
                "path": _SOURCE_SNAPSHOT_PATHS[name],
                "bytes": len(source_bytes[name]),
                "sha256": hashlib.sha256(source_bytes[name]).hexdigest(),
            }
            for name in sorted(source_bytes)
        },
        "support": {
            path: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in sorted(_SNAPSHOT_SUPPORT_FILES.items())
        },
        "require_live_runtime": require_live_runtime,
        "runtime": _runtime_identity() if require_live_runtime else None,
    }


def _create_execution_snapshot(
    parent: Path,
    source_bytes: Mapping[str, bytes],
    *,
    require_live_runtime: bool,
) -> tuple[Path, str]:
    """Create one private content-addressed policy execution tree."""
    if type(require_live_runtime) is not bool:
        msg = "require_live_runtime must be an exact boolean."
        raise RevisionInputError(msg)
    payload = _execution_snapshot_payload(
        source_bytes,
        require_live_runtime=require_live_runtime,
    )
    payload_bytes = _canonical_json(payload) + b"\n"
    digest = hashlib.sha256(payload_bytes).hexdigest()
    parent_descriptor = _ensure_directory_fd(parent, label="snapshot parent")
    name = f".dialect-revision-exec-{digest[:20]}-{uuid.uuid4().hex}"
    try:
        os.mkdir(name, mode=_STAGING_DIRECTORY_MODE, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    root = parent / name
    try:
        for name_key, display_path in _SOURCE_SNAPSHOT_PATHS.items():
            destination = root.joinpath(*PurePosixPath(display_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes_atomic(destination, source_bytes[name_key])
        for display_path, content in _SNAPSHOT_SUPPORT_FILES.items():
            destination = root.joinpath(*PurePosixPath(display_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes_atomic(destination, content)
        _write_bytes_atomic(root / "execution_manifest.json", payload_bytes)
    except Exception:
        shutil.rmtree(root)
        raise
    return root, digest


def _validate_execution_snapshot(expected_sha256: str) -> None:
    """Validate the exact isolated policy/runtime snapshot executing this file."""
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        msg = "Execution snapshot SHA-256 is invalid."
        raise RevisionInputError(msg)
    root = Path(__file__).parent.parent.parent
    payload = _read_json_with_sha256(
        root / "execution_manifest.json",
        expected_sha256,
        label="execution snapshot manifest",
    )
    if not isinstance(payload, dict):
        msg = "Execution snapshot manifest must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        payload,
        {
            "schema",
            "sources",
            "support",
            "require_live_runtime",
            "runtime",
        },
        label="execution snapshot manifest",
    )
    if payload["schema"] != "dialect-revision-isolated-execution-snapshot-v1":
        msg = "Execution snapshot schema is invalid."
        raise RevisionInputError(msg)
    sources = payload["sources"]
    support = payload["support"]
    if not isinstance(sources, dict) or not isinstance(support, dict):
        msg = "Execution snapshot source/support closure is invalid."
        raise RevisionInputError(msg)
    _require_exact_keys(sources, set(_SOURCE_SNAPSHOT_PATHS), label="snapshot sources")
    _require_exact_keys(
        support,
        set(_SNAPSHOT_SUPPORT_FILES),
        label="snapshot support",
    )
    for name_key, display_path in _SOURCE_SNAPSHOT_PATHS.items():
        path = _verify_file_record(
            root,
            sources[name_key],
            label=f"snapshot {name_key}",
        )
        if path != root.joinpath(*PurePosixPath(display_path).parts):
            msg = f"Execution snapshot source path is invalid: {name_key}"
            raise RevisionInputError(msg)
    for display_path, expected_content in _SNAPSHOT_SUPPORT_FILES.items():
        record = support[display_path]
        if not isinstance(record, dict):
            msg = f"Execution snapshot support record is invalid: {display_path}"
            raise RevisionInputError(msg)
        normalized_record = {"path": display_path, **record}
        path = _verify_file_record(
            root,
            normalized_record,
            label=f"snapshot support {display_path}",
        )
        if (
            path != root.joinpath(*PurePosixPath(display_path).parts)
            or record["bytes"] != len(expected_content)
            or record["sha256"] != hashlib.sha256(expected_content).hexdigest()
        ):
            msg = f"Execution snapshot support differs: {display_path}"
            raise RevisionInputError(msg)
    require_live_runtime = payload["require_live_runtime"]
    if type(require_live_runtime) is not bool:
        msg = "Execution snapshot live-runtime flag is not an exact boolean."
        raise RevisionInputError(msg)
    if require_live_runtime and payload["runtime"] != _runtime_identity():
        msg = "Execution snapshot runtime closure differs from the live child."
        raise RevisionInputError(msg)
    if not require_live_runtime and payload["runtime"] is not None:
        msg = "Historical execution snapshot unexpectedly binds a live runtime."
        raise RevisionInputError(msg)


def _sealed_child_environment() -> dict[str, str]:
    git_parent = _git_executable_path().parent.as_posix()
    system_path = "/usr/bin:/bin:/usr/sbin:/sbin"
    return {
        "PATH": f"{git_parent}:{system_path}",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }


def _run_isolated_snapshot(
    snapshot_root: Path,
    snapshot_sha256: str,
    request: Mapping[str, object],
) -> dict[str, Any]:
    request_path = snapshot_root / "request.json"
    response_path = snapshot_root / "response.json"
    request_receipt = _write_json_atomic(request_path, dict(request))
    process = subprocess.run(
        [
            str(_python_executable_path()),
            "-I",
            "-B",
            "-c",
            _ISOLATED_BOOTSTRAP,
            str(snapshot_root),
            snapshot_sha256,
            str(request_path),
            request_receipt.sha256,
            str(response_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_sealed_child_environment(),
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        msg = f"Isolated revision-input child failed ({process.returncode}): {stderr}"
        raise RevisionInputError(msg)
    return _read_json(response_path)


def _cleanup_execution_snapshot(root: Path) -> None:
    if root.exists():
        with suppress(Exception):
            _restore_tree_owner_write(root)
        shutil.rmtree(root)


def _approval_record(approval: RevisionApproval) -> dict[str, Any]:
    return {
        "manifest_sha256": approval.manifest_sha256,
        "authorized_stage": MATERIALIZE_FINAL_INPUTS_STAGE,
        "decision_digests": {
            decision_id: approval.decision_digests[decision_id]
            for decision_id in ("D1", "D2")
        },
        "canonical_artifact_sha256": {
            decision_id: approval.decisions[decision_id].canonical_artifact.sha256
            for decision_id in ("D1", "D2")
        },
        "attestation_kind": "textual-coauthor-record-not-cryptographic-signature",
    }


def _duplicate_policy_record() -> dict[str, Any]:
    policy = asdict(TCGA_DUPLICATE_RESOLUTION_POLICY)
    return json.loads(_canonical_json(policy))


def _decision_envelope(
    decision_id: str,
    contract: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DECISION_ARTIFACT_SCHEMA,
        "decision_id": decision_id,
        "contract": contract,
        "payload": payload,
    }


def _expected_d1_artifact(
    canonicalizer_sha256: str,
    duplicate_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _decision_envelope(
        "D1",
        D1_CONTRACT,
        {
            "input_contract": D1_INPUT_CONTRACT,
            "canonicalizer_sha256": canonicalizer_sha256,
            "duplicate_resolution_policy": (
                _duplicate_policy_record()
                if duplicate_policy is None
                else dict(duplicate_policy)
            ),
        },
    )


def _expected_d2_artifact(
    population_root: Path,
    population_manifest: Mapping[str, Any],
    axes: Mapping[str, Sequence[str]],
    cohorts: Sequence[str],
    *,
    population_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    records = population_manifest["cohort_manifests"]
    cohort_digests = {
        str(record["cohort"]): str(record["manifest_sha256"]) for record in records
    }
    return _decision_envelope(
        "D2",
        D2_CONTRACT,
        {
            "population_contract": POPULATION_CONTRACT,
            "population_manifest_sha256": (
                _sha256(population_root / "population_manifest.json")
                if population_manifest_sha256 is None
                else population_manifest_sha256
            ),
            "cohorts": list(cohorts),
            "cohort_manifest_sha256": {
                cohort: cohort_digests[cohort] for cohort in cohorts
            },
            "ordered_sample_axis_sha256": {
                cohort: _sequence_sha256(axes[cohort]) for cohort in cohorts
            },
            "selected_sample_count": sum(len(axes[cohort]) for cohort in cohorts),
        },
    )


def _secure_approval(
    approval_path: Path,
    expected_approval_sha256: str,
) -> RevisionApproval:
    before = _read_regular_bytes(approval_path, label="approval manifest")
    if hashlib.sha256(before).hexdigest() != expected_approval_sha256:
        msg = "Approval manifest does not match its independent SHA-256."
        raise RevisionInputError(msg)
    approval = validate_revision_approval(
        approval_path,
        expected_approval_sha256,
        MATERIALIZE_FINAL_INPUTS_STAGE,
    )
    after = _read_regular_bytes(approval_path, label="approval manifest")
    if before != after or approval.manifest_sha256 != expected_approval_sha256:
        msg = "Approval manifest changed during validation."
        raise RevisionInputError(msg)
    if (
        approval.schema != STAGE_SCOPED_APPROVAL_SCHEMA
        or approval.allowed_stages != (MATERIALIZE_FINAL_INPUTS_STAGE,)
        or set(approval.stage_bindings) != {MATERIALIZE_FINAL_INPUTS_STAGE}
        or tuple(approval.decisions) != ("D1", "D2")
        or tuple(approval.decision_digests) != ("D1", "D2")
    ):
        msg = (
            "Canonical input materialization requires an exact stage-scoped v5 "
            "D1/D2 authority for only materialize-final-inputs."
        )
        raise RevisionInputError(msg)
    return approval


def _read_signed_decision_artifact(
    approval_path: Path,
    approval: RevisionApproval,
    decision_id: str,
) -> tuple[dict[str, Any], Path]:
    decision = approval.decisions[decision_id]
    relative = _safe_relative_path(
        decision.canonical_artifact.path,
        label=f"{decision_id} canonical artifact path",
    )
    artifact_path = approval_path.parent.joinpath(*relative.parts)
    content = _read_regular_bytes(
        artifact_path,
        label=f"{decision_id} canonical artifact",
    )
    if (
        content != decision.canonical_artifact.content
        or len(content) != decision.canonical_artifact.size_bytes
        or hashlib.sha256(content).hexdigest() != decision.canonical_artifact.sha256
    ):
        msg = f"Signed {decision_id} artifact differs from approval authority."
        raise RevisionInputError(msg)
    artifact = _parse_json_document(content, path=artifact_path)
    _require_exact_keys(
        artifact,
        {"schema", "decision_id", "contract", "payload"},
        label=f"{decision_id} machine decision",
    )
    return artifact, artifact_path


def _validate_signed_contracts(
    approval_path: Path,
    approval: RevisionApproval,
    expected_d1: Mapping[str, Any],
    expected_d2: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], Path]]:
    expected = {"D1": expected_d1, "D2": expected_d2}
    signed: dict[str, tuple[dict[str, Any], Path]] = {}
    for decision_id in ("D1", "D2"):
        artifact, artifact_path = _read_signed_decision_artifact(
            approval_path,
            approval,
            decision_id,
        )
        if not _exact_json_equal(artifact, expected[decision_id]):
            msg = (
                f"Signed {decision_id} machine decision does not bind the exact "
                "executed input policy."
            )
            raise RevisionInputError(msg)
        signed[decision_id] = artifact, artifact_path
    return signed


def _preauthorize_canonicalizer_snapshot(
    approval_path: Path,
    approval: RevisionApproval,
    canonicalizer_sha256: str,
) -> None:
    artifact, _ = _read_signed_decision_artifact(
        approval_path,
        approval,
        "D1",
    )
    payload = artifact.get("payload")
    if not isinstance(payload, dict):
        msg = "Signed D1 machine decision payload must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        payload,
        {
            "input_contract",
            "canonicalizer_sha256",
            "duplicate_resolution_policy",
        },
        label="signed D1 payload",
    )
    if (
        artifact["schema"] != DECISION_ARTIFACT_SCHEMA
        or artifact["decision_id"] != "D1"
        or artifact["contract"] != D1_CONTRACT
        or payload["input_contract"] != D1_INPUT_CONTRACT
        or payload["canonicalizer_sha256"] != canonicalizer_sha256
        or not isinstance(payload["duplicate_resolution_policy"], dict)
    ):
        msg = "Signed D1 does not preauthorize the canonicalizer snapshot."
        raise RevisionInputError(msg)


def _materialize_cohort(  # noqa: PLR0913
    cohort: str,
    *,
    raw_maf_root: Path,
    population_root: Path,
    datahub_git_dir: Path,
    staging_root: Path,
    scratch_root: Path,
    selected_axis: Sequence[str],
    approval: RevisionApproval,
    frozen_canonicalizer: object,
    duplicate_policy: Mapping[str, Any],
) -> dict[str, Any]:
    raw_source_path = raw_maf_root / f"{cohort}.maf"
    raw_output_path = staging_root / "raw_mafs" / f"{cohort}.maf"
    expected_raw_sha256 = tcga_data.TCGA_MAF_SHA256[cohort]
    repository_path = tcga_datahub_public_path(
        cohort,
        "data_mutations.txt",
    ).as_posix()
    raw_git_blob = _git_blob_receipt(
        datahub_git_dir,
        repository_path,
        require_lfs_pointer=True,
    )
    if (
        raw_git_blob["lfs_payload"]["sha256"] != expected_raw_sha256
        or type(raw_git_blob["lfs_payload"]["bytes"]) is not int
    ):
        msg = f"Raw MAF does not match the pinned DataHub receipt: {cohort}"
        raise RevisionInputError(msg)

    case_list_repository_path = tcga_datahub_case_list_path(cohort).as_posix()
    case_list_git_blob = _git_blob_receipt(
        datahub_git_dir,
        case_list_repository_path,
    )
    case_list_bytes = _git_bytes(
        datahub_git_dir,
        ["show", f"{TCGA_DATAHUB_COMMIT}:{case_list_repository_path}"],
    )
    population_child = _read_json(population_root / cohort / "population_manifest.json")
    if (
        case_list_git_blob["bytes"] != len(case_list_bytes)
        or case_list_git_blob["sha256"] != hashlib.sha256(case_list_bytes).hexdigest()
        or case_list_git_blob["sha256"] != TCGA_CASE_LIST_RECEIPTS[cohort].sha256
        or case_list_git_blob["bytes"] != population_child["source"]["case_list_bytes"]
    ):
        msg = f"Case-list blob does not match the population authority: {cohort}"
        raise RevisionInputError(msg)
    case_list_output_path = staging_root / "case_lists" / f"{cohort}.txt"
    copied_case_list = _write_bytes_atomic(case_list_output_path, case_list_bytes)
    if (
        copied_case_list.bytes != case_list_git_blob["bytes"]
        or copied_case_list.sha256 != case_list_git_blob["sha256"]
    ):
        msg = f"Case-list bytes changed while copying: {cohort}"
        raise RevisionInputError(msg)
    case_samples = parse_tcga_sequenced_case_list(case_list_bytes, cohort)
    case_sample_set = set(case_samples)
    selected_set = set(selected_axis)
    if not selected_set <= case_sample_set:
        msg = f"Selected population axis is not a subset of the case list: {cohort}"
        raise RevisionInputError(msg)

    full_key_columns = getattr(frozen_canonicalizer, "FULL_VARIANT_KEY_COLUMNS", None)
    if not isinstance(full_key_columns, tuple):
        msg = "Frozen canonicalizer does not expose its signed API."
        raise RevisionInputError(msg)
    canonical_path = staging_root / "mafs" / f"{cohort}.maf"
    sqlite_path = scratch_root / f".{cohort}.{uuid.uuid4().hex}.sqlite3"
    try:
        streamed = _stream_canonicalize_maf(
            raw_source_path,
            canonical_path,
            sqlite_path,
            raw_copy_path=raw_output_path,
            expected_raw_receipt=_FileReceipt(
                bytes=int(raw_git_blob["lfs_payload"]["bytes"]),
                sha256=expected_raw_sha256,
            ),
            selected_samples=frozenset(selected_set),
            case_samples=frozenset(case_sample_set),
            frozen_canonicalizer=frozen_canonicalizer,
        )
    finally:
        for suffix in ("", "-journal", "-shm", "-wal"):
            with suppress(FileNotFoundError):
                (Path(f"{sqlite_path}{suffix}")).unlink()
    raw_rows = streamed.raw_rows
    selected_rows = streamed.selected_rows
    canonical_rows = streamed.output_rows
    unselected_rows = raw_rows - selected_rows
    duplicate_rows_removed = selected_rows - canonical_rows
    if (
        raw_rows != unselected_rows + selected_rows
        or selected_rows != canonical_rows + duplicate_rows_removed
    ):
        msg = f"Canonical MAF row accounting does not close: {cohort}"
        raise RevisionInputError(msg)

    population_output_dir = staging_root / "population" / cohort
    resolution_audit = streamed.audit

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": INPUT_CONTRACT,
        "cohort": cohort,
        "approval": _approval_record(approval),
        "source": {
            "repository": "https://github.com/cBioPortal/datahub",
            "commit": TCGA_DATAHUB_COMMIT,
            "tree": TCGA_DATAHUB_TREE,
            "repository_path": repository_path,
            "raw_maf": _file_record(
                raw_output_path,
                display_path=f"raw_mafs/{cohort}.maf",
            ),
            "raw_git_blob": raw_git_blob,
            "case_list_repository_path": case_list_repository_path,
            "case_list_git_blob": case_list_git_blob,
            "case_list": _file_record(
                case_list_output_path,
                display_path=f"case_lists/{cohort}.txt",
            ),
        },
        "population": {
            "manifest": _file_record(
                population_output_dir / "population_manifest.json",
                display_path=f"population/{cohort}/population_manifest.json",
            ),
            "sample_axis": _file_record(
                population_output_dir / "sample_axis.txt",
                display_path=f"population/{cohort}/sample_axis.txt",
            ),
            "sample_count": len(selected_axis),
            "ordered_sample_axis_sha256": _sequence_sha256(selected_axis),
            "all_zero_rows_required_for_samples_without_retained_events": True,
        },
        "transformation": {
            "canonicalizer": _file_record(
                staging_root
                / "implementation"
                / "src"
                / "dialect"
                / "data"
                / "variants.py",
                display_path="implementation/src/dialect/data/variants.py",
            ),
            "materializer": _file_record(
                staging_root
                / "implementation"
                / "analysis"
                / "materialize_tcga_revision_inputs.py",
                display_path=(
                    "implementation/analysis/materialize_tcga_revision_inputs.py"
                ),
            ),
            "serializer": SERIALIZER_CONTRACT,
            "duplicate_resolution_policy": dict(duplicate_policy),
            "raw_maf_sha256": expected_raw_sha256,
            "sample_axis_file_sha256": _sha256(
                population_output_dir / "sample_axis.txt",
            ),
            "canonical_maf_sha256": _sha256(canonical_path),
        },
        "row_accounting": {
            "raw_rows": raw_rows,
            "unselected_sample_rows": unselected_rows,
            "selected_rows_before_deduplication": selected_rows,
            "duplicate_excess_rows_removed": duplicate_rows_removed,
            "canonical_output_rows": canonical_rows,
            "multiallelic_coordinate_groups_preserved": (
                streamed.multiallelic_coordinate_groups
            ),
            "duplicate_group_count": resolution_audit.duplicate_group_count,
            "semantic_agreement_group_count": (
                resolution_audit.semantic_agreement_group_count
            ),
            "ignored_conflict_group_count": (
                resolution_audit.ignored_conflict_group_count
            ),
            "frozen_effect_resolution_group_count": (
                resolution_audit.frozen_effect_resolution_group_count
            ),
            "resolved_conflict_groups_by_column": dict(
                resolution_audit.resolved_conflict_groups_by_column,
            ),
            "selected_mutsig_effect_groups": dict(
                resolution_audit.selected_mutsig_effect_groups,
            ),
            "unresolved_semantic_conflicts": 0,
        },
        "invariants": {
            "raw_row_identity_closed": raw_rows == unselected_rows + selected_rows,
            "deduplication_row_identity_closed": selected_rows
            == canonical_rows + duplicate_rows_removed,
            "output_full_key_duplicates": 0,
            "output_samples_outside_selected_axis": 0,
            "deterministic_order": True,
            "row_order_invariance_contract": STREAMING_CANONICALIZATION_CONTRACT,
            "association_outputs_opened": False,
        },
        "output": {
            "canonical_maf": _file_record(
                canonical_path,
                display_path=f"mafs/{cohort}.maf",
            ),
            "rows": canonical_rows,
            "ordered_columns_sha256": streamed.ordered_columns_sha256,
        },
    }
    manifest_path = staging_root / "cohorts" / f"{cohort}.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _snapshot_implementation(
    staging_root: Path,
    source_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    source_records: dict[str, dict[str, int | str]] = {}
    if set(source_bytes) != set(_SOURCE_SNAPSHOT_PATHS):
        msg = "Implementation source-byte snapshot is incomplete."
        raise RevisionInputError(msg)
    for name in _SOURCE_SNAPSHOT_PATHS:
        display_path = _SOURCE_SNAPSHOT_PATHS[name]
        destination = staging_root.joinpath(*PurePosixPath(display_path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(destination, source_bytes[name])
        source_records[name] = _file_record(
            destination,
            display_path=display_path,
        )
    runtime_snapshots: dict[str, dict[str, dict[str, int | str]]] = {}
    runtime_dir = staging_root / "implementation" / "runtime"
    runtime_dir.mkdir()
    for package in _PACKAGE_NAMES:
        record_display_path = f"implementation/runtime/{package}.RECORD"
        destination = staging_root.joinpath(
            *PurePosixPath(record_display_path).parts,
        )
        record_bytes = _read_regular_bytes(
            _distribution_record_path(package),
            label=f"{package} distribution RECORD",
        )
        _write_bytes_atomic(destination, record_bytes)
        closure_display_path = f"implementation/runtime/{package}.closure.json"
        closure_destination = staging_root.joinpath(
            *PurePosixPath(closure_display_path).parts,
        )
        _write_json_atomic(
            closure_destination,
            _distribution_runtime_closure(package),
        )
        runtime_snapshots[package] = {
            "record": _file_record(
                destination,
                display_path=record_display_path,
            ),
            "closure": _file_record(
                closure_destination,
                display_path=closure_display_path,
            ),
        }
    return {
        "sources": source_records,
        "runtime": {
            "identity": _runtime_identity(),
            "snapshots": runtime_snapshots,
        },
        "serializer": SERIALIZER_CONTRACT,
        "scientific_policy_contract": D1_INPUT_CONTRACT,
        "streaming_canonicalization_contract": (STREAMING_CANONICALIZATION_CONTRACT),
    }


def _copy_population_bundle(
    source_root: Path,
    staging_root: Path,
    cohorts: Sequence[str],
) -> None:
    destination_root = staging_root / "population"
    _copy_regular_file(
        source_root / "population_manifest.json",
        destination_root / "population_manifest.json",
    )
    for cohort in cohorts:
        destination = destination_root / cohort
        destination.mkdir()
        for filename in ("population_manifest.json", "sample_axis.txt"):
            _copy_regular_file(
                source_root / cohort / filename,
                destination / filename,
            )
    population_sources = {
        "src/dialect/data/tcga.py": _tcga_source_path(),
        "analysis/materialize_tcga_revision_population.py": (
            _population_materializer_source_path()
        ),
    }
    for display_path, source in population_sources.items():
        destination = destination_root.joinpath(
            *PurePosixPath(display_path).parts,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_regular_file(source, destination)


def _copy_signed_contracts(
    staging_root: Path,
    signed: Mapping[str, tuple[dict[str, Any], Path]],
) -> dict[str, dict[str, Any]]:
    authority_dir = staging_root / "authority"
    authority_dir.mkdir()
    copied: dict[str, dict[str, Any]] = {}
    for decision_id in ("D1", "D2"):
        content, _source = signed[decision_id]
        destination = authority_dir / f"{decision_id}.json"
        _write_bytes_atomic(destination, _canonical_json(content) + b"\n")
        copied[decision_id] = {
            "artifact": _file_record(
                destination,
                display_path=f"authority/{decision_id}.json",
            ),
            "content": content,
        }
    return copied


def _approval_artifact_receipts(approval: RevisionApproval) -> dict[str, object]:
    """Return the unique relative approval/evidence closure validated upstream."""
    receipts: dict[str, object] = {}

    def add(receipt: object, *, label: str) -> None:
        relative = _safe_relative_path(
            getattr(receipt, "path", None),
            label=f"{label} path",
        ).as_posix()
        prior = receipts.setdefault(relative, receipt)
        if (
            getattr(prior, "sha256", None) != getattr(receipt, "sha256", None)
            or getattr(prior, "size_bytes", None)
            != getattr(receipt, "size_bytes", None)
            or getattr(prior, "content", None) != getattr(receipt, "content", None)
        ):
            msg = f"Approval closure reuses {relative!r} for different bytes."
            raise RevisionInputError(msg)

    source_notice = getattr(approval, "source_notice", None)
    if source_notice is not None:
        add(source_notice.file, label="approval source notice")
    for decision_id, decision in approval.decisions.items():
        add(decision.canonical_artifact, label=f"{decision_id} artifact")
        for index, attestation in enumerate(getattr(decision, "attestations", ())):
            add(
                attestation.evidence.file,
                label=f"{decision_id} attestation {index} evidence",
            )
    return dict(sorted(receipts.items()))


def _copy_approval_closure(
    staging_root: Path,
    approval_path: Path,
    approval: RevisionApproval,
    expected_approval_sha256: str,
) -> dict[str, Any]:
    """Copy a relocatable exact approval/evidence closure into the bundle."""
    closure_root = staging_root / "authority" / "approval"
    closure_root.mkdir()
    manifest_path = closure_root / "approval_manifest.json"
    manifest_receipt = _copy_regular_file(approval_path, manifest_path)
    if manifest_receipt.sha256 != expected_approval_sha256:
        msg = "Copied approval manifest differs from its independent SHA-256."
        raise RevisionInputError(msg)
    closure_records: list[dict[str, int | str]] = []
    for relative_text, receipt in _approval_artifact_receipts(approval).items():
        relative = PurePosixPath(relative_text)
        destination = closure_root.joinpath(*relative.parts)
        if destination == manifest_path:
            msg = "Approval closure artifact collides with its bundled manifest."
            raise RevisionInputError(msg)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied = _copy_regular_file(
            approval_path.parent.joinpath(*relative.parts),
            destination,
        )
        if copied.bytes != getattr(
            receipt,
            "size_bytes",
            None,
        ) or copied.sha256 != getattr(receipt, "sha256", None):
            msg = f"Approval closure file changed while copying: {relative_text}"
            raise RevisionInputError(msg)
        closure_records.append(
            _file_record(
                destination,
                display_path=(f"authority/approval/{relative.as_posix()}"),
            ),
        )
    return {
        "manifest": _file_record(
            manifest_path,
            display_path="authority/approval/approval_manifest.json",
        ),
        "independent_manifest_sha256": expected_approval_sha256,
        "files": closure_records,
    }


def _validate_approval_closure(
    bundle_root: Path,
    value: object,
    expected_approval_sha256: str,
    *,
    external_approval_path: Path,
    require_current_execution_environment: bool,
) -> tuple[RevisionApproval, Path, tuple[str, ...]]:
    if not isinstance(value, dict):
        msg = "Bundled approval closure must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        value,
        {"manifest", "independent_manifest_sha256", "files"},
        label="approval closure",
    )
    if value["independent_manifest_sha256"] != expected_approval_sha256:
        msg = "Bundled approval closure changed its independent manifest SHA-256."
        raise RevisionInputError(msg)
    manifest_path = _verify_file_record(
        bundle_root,
        value["manifest"],
        label="bundled approval manifest",
    )
    expected_manifest_path = (
        bundle_root / "authority" / "approval" / "approval_manifest.json"
    )
    if manifest_path != expected_manifest_path:
        msg = "Bundled approval manifest path is invalid."
        raise RevisionInputError(msg)
    files = value["files"]
    if not isinstance(files, list):
        msg = "Bundled approval closure files must be a list."
        raise RevisionInputError(msg)
    observed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(files):
        path = _verify_file_record(
            bundle_root,
            record,
            label=f"approval closure file {index}",
        )
        expected_parent = bundle_root / "authority" / "approval"
        try:
            relative = path.relative_to(expected_parent).as_posix()
        except ValueError as error:
            msg = "Approval closure file escapes its co-release root."
            raise RevisionInputError(msg) from error
        if relative in observed:
            msg = f"Approval closure repeats a file: {relative}"
            raise RevisionInputError(msg)
        observed[relative] = record
    if list(observed) != sorted(observed):
        msg = "Approval closure file records are not canonically ordered."
        raise RevisionInputError(msg)
    approval = _secure_approval(manifest_path, expected_approval_sha256)
    expected_receipts = _approval_artifact_receipts(approval)
    if set(observed) != set(expected_receipts):
        msg = "Bundled approval closure is incomplete or contains extra files."
        raise RevisionInputError(msg)
    for relative, receipt in expected_receipts.items():
        record = observed[relative]
        if record["bytes"] != getattr(receipt, "size_bytes", None) or record[
            "sha256"
        ] != getattr(receipt, "sha256", None):
            msg = f"Bundled approval receipt differs from authority: {relative}"
            raise RevisionInputError(msg)
    if require_current_execution_environment:
        live_bytes = _read_regular_bytes(
            external_approval_path,
            label="current external approval manifest",
        )
        if hashlib.sha256(live_bytes).hexdigest() != expected_approval_sha256:
            msg = "Current external approval manifest differs from the bundle."
            raise RevisionInputError(msg)
        live_approval = _secure_approval(
            external_approval_path,
            expected_approval_sha256,
        )
        live_receipts = _approval_artifact_receipts(live_approval)
        if set(live_receipts) != set(expected_receipts):
            msg = "Current external approval closure differs from the bundle."
            raise RevisionInputError(msg)
        for relative, expected_receipt in expected_receipts.items():
            live_receipt = live_receipts[relative]
            if (
                getattr(live_receipt, "size_bytes", None)
                != getattr(expected_receipt, "size_bytes", None)
                or getattr(live_receipt, "sha256", None)
                != getattr(expected_receipt, "sha256", None)
                or getattr(live_receipt, "content", None)
                != getattr(expected_receipt, "content", None)
            ):
                msg = f"Current external approval file differs: {relative}"
                raise RevisionInputError(msg)
    inventory_paths = (
        "authority/approval/approval_manifest.json",
        *(f"authority/approval/{relative}" for relative in sorted(expected_receipts)),
    )
    return approval, manifest_path, inventory_paths


def _approval_closure_inventory_paths(value: object) -> tuple[str, ...]:
    """Extract the closed file axis from a pinned approval-closure claim."""
    if not isinstance(value, dict):
        msg = "Bundled approval closure must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        value,
        {"manifest", "independent_manifest_sha256", "files"},
        label="approval closure",
    )
    manifest = value["manifest"]
    files = value["files"]
    if not isinstance(manifest, dict) or not isinstance(files, list):
        msg = "Approval closure file-record family is invalid."
        raise RevisionInputError(msg)
    records = [manifest, *files]
    paths: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            msg = f"Approval closure inventory record {index} is invalid."
            raise RevisionInputError(msg)
        _require_exact_keys(
            record,
            {"path", "bytes", "sha256"},
            label=f"approval closure inventory record {index}",
        )
        path = _safe_relative_path(
            record["path"],
            label=f"approval closure inventory path {index}",
        ).as_posix()
        if not path.startswith("authority/approval/"):
            msg = "Approval closure inventory path escapes its fixed root."
            raise RevisionInputError(msg)
        paths.append(path)
    if len(paths) != len(set(paths)):
        msg = "Approval closure inventory repeats a path."
        raise RevisionInputError(msg)
    if paths[1:] != sorted(paths[1:]):
        msg = "Approval closure artifact inventory is not sorted."
        raise RevisionInputError(msg)
    return tuple(paths)


def _expected_inventory(
    cohorts: Sequence[str],
    *,
    approval_files: Iterable[str] = (),
) -> dict[str, object]:
    directories = {
        "",
        "authority",
        "case_lists",
        "cohorts",
        "implementation",
        "implementation/analysis",
        "implementation/runtime",
        "implementation/src",
        "implementation/src/dialect",
        "implementation/src/dialect/data",
        "mafs",
        "population",
        "population/analysis",
        "population/src",
        "population/src/dialect",
        "population/src/dialect/data",
        "raw_mafs",
    }
    directories.update(f"population/{cohort}" for cohort in cohorts)
    files = {
        "authority/D1.json",
        "authority/D2.json",
        "implementation/analysis/materialize_tcga_revision_inputs.py",
        "implementation/analysis/materialize_tcga_revision_population.py",
        "implementation/runtime/numpy.RECORD",
        "implementation/runtime/numpy.closure.json",
        "implementation/runtime/pandas.RECORD",
        "implementation/runtime/pandas.closure.json",
        "implementation/src/dialect/data/revision_approval.py",
        "implementation/src/dialect/data/tcga.py",
        "implementation/src/dialect/data/variants.py",
        "input_manifest.json",
        "population/analysis/materialize_tcga_revision_population.py",
        "population/population_manifest.json",
        "population/src/dialect/data/tcga.py",
    }
    files.update(approval_files)
    for approval_file in approval_files:
        path = PurePosixPath(approval_file)
        directories.update(
            PurePosixPath(*path.parts[:index]).as_posix()
            for index in range(1, len(path.parts))
        )
    for cohort in cohorts:
        files.update(
            {
                f"cohorts/{cohort}.json",
                f"case_lists/{cohort}.txt",
                f"mafs/{cohort}.maf",
                f"population/{cohort}/population_manifest.json",
                f"population/{cohort}/sample_axis.txt",
                f"raw_mafs/{cohort}.maf",
            },
        )
    return {
        "directories": sorted(directories),
        "files": sorted(files),
        "directory_mode": f"{_FROZEN_DIRECTORY_MODE:04o}",
        "file_mode": f"{_FROZEN_FILE_MODE:04o}",
    }


def _filesystem_inventory(root: Path) -> dict[str, object]:
    root = _absolute_unresolved(root)
    root_descriptor = _open_directory_fd(root, label="bundle inventory root")
    directories = [""]
    files: list[str] = []

    root_mode = stat.S_IMODE(os.fstat(root_descriptor).st_mode)
    if root_mode != _FROZEN_DIRECTORY_MODE:
        os.close(root_descriptor)
        msg = f"Bundle root directory mode must be 0500, observed {root_mode:04o}."
        raise RevisionInputError(msg)

    def walk(descriptor: int, prefix: PurePosixPath) -> None:
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."} or "/" in name or "\x00" in name:
                msg = f"Bundle inventory contains an invalid entry name: {name!r}"
                raise RevisionInputError(msg)
            entry_stat = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            relative = prefix / name
            if stat.S_ISLNK(entry_stat.st_mode):
                msg = f"Bundle inventory contains a symlink: {relative.as_posix()}"
                raise RevisionInputError(msg)
            if stat.S_ISDIR(entry_stat.st_mode):
                entry_mode = stat.S_IMODE(entry_stat.st_mode)
                if entry_mode != _FROZEN_DIRECTORY_MODE:
                    msg = (
                        "Bundle directory is writable or has a noncanonical mode: "
                        f"{relative.as_posix()} ({entry_mode:04o})"
                    )
                    raise RevisionInputError(msg)
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    directories.append(relative.as_posix())
                    walk(child_descriptor, relative)
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(entry_stat.st_mode):
                entry_mode = stat.S_IMODE(entry_stat.st_mode)
                if entry_mode != _FROZEN_FILE_MODE:
                    msg = (
                        "Bundle file is writable or has a noncanonical mode: "
                        f"{relative.as_posix()} ({entry_mode:04o})"
                    )
                    raise RevisionInputError(msg)
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened_stat = os.fstat(file_descriptor)
                    if (
                        not stat.S_ISREG(opened_stat.st_mode)
                        or opened_stat.st_nlink != 1
                    ):
                        msg = (
                            "Bundle inventory entry is not a single-link regular "
                            f"file: {relative}"
                        )
                        raise RevisionInputError(msg)
                finally:
                    os.close(file_descriptor)
                files.append(relative.as_posix())
            else:
                msg = f"Bundle inventory contains a special file: {relative.as_posix()}"
                raise RevisionInputError(msg)

    try:
        walk(root_descriptor, PurePosixPath())
    finally:
        os.close(root_descriptor)
    return {
        "directories": sorted(directories),
        "files": sorted(files),
        "directory_mode": f"{_FROZEN_DIRECTORY_MODE:04o}",
        "file_mode": f"{_FROZEN_FILE_MODE:04o}",
    }


def _set_tree_modes(root: Path, *, frozen: bool) -> None:
    """Set a closed regular-file tree to frozen or owner-writable modes."""
    root_descriptor = _open_directory_fd(root, label="bundle mode root")
    file_mode = _FROZEN_FILE_MODE if frozen else _STAGING_FILE_MODE
    directory_mode = _FROZEN_DIRECTORY_MODE if frozen else _STAGING_DIRECTORY_MODE

    def walk(descriptor: int) -> None:
        if not frozen:
            os.fchmod(descriptor, directory_mode)
        for name in sorted(os.listdir(descriptor)):
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode):
                msg = f"Refusing to change modes through bundle symlink: {name}"
                raise RevisionInputError(msg)
            if stat.S_ISDIR(entry.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    walk(child)
                    if frozen:
                        os.fchmod(child, directory_mode)
                finally:
                    os.close(child)
            elif stat.S_ISREG(entry.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened = _stable_file_state(child, label="bundle mode file")
                    if opened.links != 1:
                        msg = f"Bundle mode file is hard linked: {name}"
                        raise RevisionInputError(msg)
                    os.fchmod(child, file_mode)
                finally:
                    os.close(child)
        if frozen:
            os.fchmod(descriptor, directory_mode)

    try:
        walk(root_descriptor)
    finally:
        os.close(root_descriptor)


def _freeze_tree_read_only(root: Path) -> None:
    _set_tree_modes(root, frozen=True)


def _restore_tree_owner_write(root: Path) -> None:
    _set_tree_modes(root, frozen=False)


def _available_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available = pages * page_size
    except (OSError, TypeError, ValueError):
        available = 0
    if available <= 0 and sys.platform == "darwin":
        vm_stat = Path("/usr/bin/vm_stat")
        _require_regular_file(vm_stat, label="vm_stat executable")
        process = subprocess.run(
            [vm_stat.as_posix()],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        if process.returncode == 0:
            page_match = re.search(r"page size of (\d+) bytes", process.stdout)
            counts = {
                key: int(value)
                for key, value in re.findall(
                    r"^(Pages (?:free|inactive|speculative)):\s+(\d+)\.$",
                    process.stdout,
                    flags=re.MULTILINE,
                )
            }
            if page_match is not None:
                available = int(page_match.group(1)) * sum(counts.values())
    if available <= 0 and Path("/proc/meminfo").is_file():
        raw = _read_regular_bytes(Path("/proc/meminfo"), label="memory availability")
        match = re.search(rb"^MemAvailable:\s+([1-9][0-9]*)\s+kB$", raw, re.MULTILINE)
        if match is not None:
            available = int(match.group(1)) * 1024
    if available <= 0:
        msg = "Unable to determine conservative available memory."
        raise RevisionInputError(msg)
    return available


def _preflight_materialization_resources(
    raw_root: Path,
    output_parent: Path,
    cohorts: Sequence[str],
) -> dict[str, int]:
    raw_bytes = 0
    for cohort in cohorts:
        with _stable_regular_descriptor(
            raw_root / f"{cohort}.maf",
            label=f"{cohort} raw MAF preflight",
        ) as (_descriptor, state):
            raw_bytes += state.size
    available_memory = _available_memory_bytes()
    if available_memory < _MIN_AVAILABLE_MEMORY_BYTES:
        msg = (
            "Insufficient available memory for bounded MAF canonicalization: "
            f"need at least {_MIN_AVAILABLE_MEMORY_BYTES}, observed {available_memory}."
        )
        raise RevisionInputError(msg)
    disk = shutil.disk_usage(output_parent)
    projected_disk = raw_bytes * _DISK_RAW_MULTIPLIER + _DISK_FIXED_SAFETY_BYTES
    if disk.free < projected_disk:
        msg = (
            "Insufficient free disk for raw copies, canonical output, SQLite staging, "
            f"and safety reserve: need {projected_disk}, observed {disk.free}."
        )
        raise RevisionInputError(msg)
    return {
        "raw_bytes": raw_bytes,
        "available_memory_bytes": available_memory,
        "free_disk_bytes": disk.free,
        "projected_disk_bytes": projected_disk,
    }


def _require_binary_receipt(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        msg = f"{label} must be a binary receipt."
        raise RevisionInputError(msg)
    _require_exact_keys(value, {"bytes", "sha256"}, label=label)
    if (
        not isinstance(value["bytes"], int)
        or isinstance(value["bytes"], bool)
        or value["bytes"] <= 0
        or not isinstance(value["sha256"], str)
        or _SHA256_PATTERN.fullmatch(value["sha256"]) is None
    ):
        msg = f"{label} contains an invalid byte/hash receipt."
        raise RevisionInputError(msg)


def _validate_runtime_identity_schema(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = "Runtime identity must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(value, {"python", "packages", "git"}, label="runtime")
    python_record = value["python"]
    git_record = value["git"]
    packages = value["packages"]
    if not isinstance(python_record, dict) or not isinstance(git_record, dict):
        msg = "Runtime executable records must be objects."
        raise RevisionInputError(msg)
    if not isinstance(packages, dict):
        msg = "Runtime package records must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        python_record,
        {"implementation", "version", "executable"},
        label="runtime.python",
    )
    _require_exact_keys(
        git_record,
        {"version", "executable"},
        label="runtime.git",
    )
    if (
        not isinstance(python_record["implementation"], str)
        or not python_record["implementation"]
        or not isinstance(python_record["version"], str)
        or not python_record["version"]
        or not isinstance(git_record["version"], str)
        or not git_record["version"]
    ):
        msg = "Runtime executable identity contains blank text."
        raise RevisionInputError(msg)
    _require_binary_receipt(
        python_record["executable"],
        label="runtime.python.executable",
    )
    _require_binary_receipt(
        git_record["executable"],
        label="runtime.git.executable",
    )
    _require_exact_keys(packages, set(_PACKAGE_NAMES), label="runtime.packages")
    for package in _PACKAGE_NAMES:
        record = packages[package]
        if not isinstance(record, dict):
            msg = f"Runtime package identity is invalid: {package}"
            raise RevisionInputError(msg)
        _require_exact_keys(
            record,
            {
                "version",
                "record_bytes",
                "record_sha256",
                "closure_files",
                "closure_sha256",
            },
            label=f"runtime.packages.{package}",
        )
        if (
            not isinstance(record["version"], str)
            or not record["version"]
            or not isinstance(record["record_bytes"], int)
            or isinstance(record["record_bytes"], bool)
            or record["record_bytes"] <= 0
            or not isinstance(record["record_sha256"], str)
            or _SHA256_PATTERN.fullmatch(record["record_sha256"]) is None
            or type(record["closure_files"]) is not int
            or record["closure_files"] <= 0
            or not isinstance(record["closure_sha256"], str)
            or _SHA256_PATTERN.fullmatch(record["closure_sha256"]) is None
        ):
            msg = f"Runtime package receipt is invalid: {package}"
            raise RevisionInputError(msg)
    return value


def _validate_distribution_closure_schema(
    value: object,
    *,
    package: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{package} runtime closure must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(value, {"package", "version", "files"}, label="closure")
    files = value["files"]
    if value["package"] != package or not isinstance(value["version"], str):
        msg = f"{package} runtime closure identity is invalid."
        raise RevisionInputError(msg)
    if not isinstance(files, list) or not files:
        msg = f"{package} runtime closure files must be a nonempty list."
        raise RevisionInputError(msg)
    paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            msg = f"{package} runtime closure file {index} is invalid."
            raise RevisionInputError(msg)
        _require_exact_keys(
            record,
            {"distribution_path", "bytes", "sha256"},
            label=f"{package} runtime closure file {index}",
        )
        path = record["distribution_path"]
        if (
            not isinstance(path, str)
            or not path
            or "\x00" in path
            or "\\" in path
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
            or not isinstance(record["sha256"], str)
            or _SHA256_PATTERN.fullmatch(record["sha256"]) is None
        ):
            msg = f"{package} runtime closure file {index} receipt is invalid."
            raise RevisionInputError(msg)
        paths.append(
            _safe_relative_path(
                path,
                label=f"{package} runtime closure path {index}",
            ).as_posix(),
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        msg = f"{package} runtime closure file axis is not unique and sorted."
        raise RevisionInputError(msg)
    if sum(PurePosixPath(path).name == "RECORD" for path in paths) != 1:
        msg = f"{package} runtime closure must contain exactly one RECORD."
        raise RevisionInputError(msg)
    return value


def _validate_implementation(
    root: Path,
    value: object,
    *,
    require_current_execution_environment: bool,
) -> dict[str, tuple[Path, bytes]]:
    if not isinstance(value, dict):
        msg = "Implementation provenance must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        value,
        {
            "sources",
            "runtime",
            "serializer",
            "scientific_policy_contract",
            "streaming_canonicalization_contract",
        },
        label="implementation",
    )
    if (
        value["serializer"] != SERIALIZER_CONTRACT
        or value["scientific_policy_contract"] != D1_INPUT_CONTRACT
        or value["streaming_canonicalization_contract"]
        != STREAMING_CANONICALIZATION_CONTRACT
    ):
        msg = "Implementation serializer contract is invalid."
        raise RevisionInputError(msg)
    sources = value["sources"]
    runtime = value["runtime"]
    if not isinstance(sources, dict) or not isinstance(runtime, dict):
        msg = "Implementation source/runtime provenance is invalid."
        raise RevisionInputError(msg)
    _require_exact_keys(sources, set(_SOURCE_SNAPSHOT_PATHS), label="sources")
    _require_exact_keys(runtime, {"identity", "snapshots"}, label="runtime")
    source_snapshots: dict[str, tuple[Path, bytes]] = {}
    for name, expected_path in _SOURCE_SNAPSHOT_PATHS.items():
        path, content = _verified_file_bytes(
            root,
            sources[name],
            label=f"{name} snapshot",
        )
        if path != root.joinpath(*PurePosixPath(expected_path).parts):
            msg = f"Implementation source snapshot path is invalid: {name}"
            raise RevisionInputError(msg)
        source_snapshots[name] = (path, content)
    identity = _validate_runtime_identity_schema(runtime["identity"])
    snapshots = runtime["snapshots"]
    if not isinstance(snapshots, dict):
        msg = "Runtime snapshots must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(snapshots, set(_PACKAGE_NAMES), label="runtime snapshots")
    for package in _PACKAGE_NAMES:
        snapshot = snapshots[package]
        if not isinstance(snapshot, dict):
            msg = f"{package} runtime snapshot must be an object."
            raise RevisionInputError(msg)
        _require_exact_keys(
            snapshot,
            {"record", "closure"},
            label=f"{package} runtime snapshot",
        )
        record_path = _verify_file_record(
            root,
            snapshot["record"],
            label=f"{package} RECORD snapshot",
        )
        closure_path, closure = _read_verified_json_file_record(
            root,
            snapshot["closure"],
            label=f"{package} closure snapshot",
        )
        closure = _validate_distribution_closure_schema(
            closure,
            package=package,
        )
        expected_record_path = root / "implementation" / "runtime" / f"{package}.RECORD"
        expected_closure_path = (
            root / "implementation" / "runtime" / f"{package}.closure.json"
        )
        package_identity = identity["packages"][package]
        closure_record = next(
            record
            for record in closure["files"]
            if PurePosixPath(record["distribution_path"]).name == "RECORD"
        )
        if (
            record_path != expected_record_path
            or closure_path != expected_closure_path
            or snapshot["record"]["bytes"] != package_identity["record_bytes"]
            or snapshot["record"]["sha256"] != package_identity["record_sha256"]
            or closure_record["bytes"] != snapshot["record"]["bytes"]
            or closure_record["sha256"] != snapshot["record"]["sha256"]
            or snapshot["closure"]["sha256"] != package_identity["closure_sha256"]
            or not isinstance(closure.get("files"), list)
            or len(closure["files"]) != package_identity["closure_files"]
            or closure.get("package") != package
            or closure.get("version") != package_identity["version"]
        ):
            msg = f"Runtime package snapshot is not identity-bound: {package}"
            raise RevisionInputError(msg)
    if require_current_execution_environment:
        if identity != _runtime_identity():
            msg = "Current execution runtime differs from the materialized runtime."
            raise RevisionInputError(msg)
        for name, current_path in _source_dependencies().items():
            current_size, current_digest = _regular_file_receipt(
                current_path,
                label=f"current {name} source",
            )
            if (
                sources[name]["bytes"] != current_size
                or sources[name]["sha256"] != current_digest
            ):
                msg = f"Current execution source differs from snapshot: {name}"
                raise RevisionInputError(msg)
    return source_snapshots


def _load_canonicalizer_snapshot(content: bytes, *, path: Path) -> object:
    """Compile and execute the exact canonicalizer bytes that were verified."""
    module_name = f"_dialect_frozen_variants_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = path.as_posix()
    module.__package__ = "dialect.data"
    sys.modules[module_name] = module
    prior_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        code = compile(content, path.as_posix(), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102
    except Exception as error:
        msg = "Unable to execute the frozen canonicalizer snapshot."
        raise RevisionInputError(msg) from error
    finally:
        sys.dont_write_bytecode = prior_dont_write_bytecode
        sys.modules.pop(module_name, None)
    return module


def _publish_claim_path(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}{_PUBLISH_CLAIM_SUFFIX}")


@contextmanager
def _exclusive_publish_claim_at(
    parent_descriptor: int,
    output_name: str,
) -> Iterable[None]:
    claim_name = f".{output_name}{_PUBLISH_CLAIM_SUFFIX}"
    try:
        descriptor = os.open(
            claim_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
    except FileExistsError as error:
        msg = f"Another input publication holds the claim: {claim_name}"
        raise FileExistsError(msg) from error
    try:
        os.close(descriptor)
        yield
    finally:
        with suppress(FileNotFoundError):
            os.unlink(claim_name, dir_fd=parent_descriptor)


@contextmanager
def _exclusive_publish_claim(output_root: Path) -> Iterable[None]:
    """Compatibility wrapper around the descriptor-relative publication claim."""
    parent_descriptor = _open_directory_fd(
        output_root.parent,
        label="publication parent",
    )
    try:
        with _exclusive_publish_claim_at(parent_descriptor, output_root.name):
            yield
    finally:
        os.close(parent_descriptor)


def _directory_path_matches_fd(path: Path, descriptor: int) -> bool:
    """Return whether a path still names the exact opened directory inode."""
    try:
        observed = path.stat(follow_symlinks=False)
    except OSError:
        return False
    opened = os.fstat(descriptor)
    return (
        stat.S_ISDIR(observed.st_mode)
        and observed.st_dev == opened.st_dev
        and observed.st_ino == opened.st_ino
    )


@contextmanager
def _private_validation_scratch(
    bundle_root: Path,
    cohort: str,
) -> Iterable[Path]:
    """Yield an owner-private scratch directory on the bundle filesystem."""
    parent_path = bundle_root.parent
    parent_descriptor = _open_directory_fd(
        parent_path,
        label="validation scratch parent",
    )
    name = f".dialect-{cohort}-validate-{uuid.uuid4().hex}"
    scratch_descriptor: int | None = None
    created = False
    try:
        os.mkdir(
            name,
            mode=_STAGING_DIRECTORY_MODE,
            dir_fd=parent_descriptor,
        )
        created = True
        scratch_descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        scratch_path = parent_path / name

        def is_private_and_bound() -> bool:
            parent_stat = os.fstat(parent_descriptor)
            scratch_stat = os.fstat(scratch_descriptor)
            return (
                stat.S_ISDIR(scratch_stat.st_mode)
                and scratch_stat.st_dev == parent_stat.st_dev
                and scratch_stat.st_uid == os.geteuid()
                and stat.S_IMODE(scratch_stat.st_mode) == _STAGING_DIRECTORY_MODE
                and _directory_path_matches_fd(scratch_path, scratch_descriptor)
                and _directory_path_matches_fd(parent_path, parent_descriptor)
            )

        if not is_private_and_bound():
            msg = "Validation scratch is not owner-private on the bundle filesystem."
            raise RevisionInputError(msg)
        yield scratch_path
    finally:
        cleanup_error: RevisionInputError | None = None
        if scratch_descriptor is not None:
            scratch_path = parent_path / name
            if not is_private_and_bound():
                cleanup_error = RevisionInputError(
                    "Validation scratch path changed before descriptor-safe cleanup.",
                )
            else:
                # ``Path.iterdir`` cannot preserve this descriptor-relative walk.
                for entry_name in sorted(os.listdir(scratch_descriptor)):  # noqa: PTH208
                    entry = os.stat(
                        entry_name,
                        dir_fd=scratch_descriptor,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
                        cleanup_error = RevisionInputError(
                            "Validation scratch contains a non-private regular file.",
                        )
                        break
                    os.unlink(entry_name, dir_fd=scratch_descriptor)
                if cleanup_error is None:
                    os.fsync(scratch_descriptor)
            os.close(scratch_descriptor)
        if created and cleanup_error is None:
            os.rmdir(name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        os.close(parent_descriptor)
        if cleanup_error is not None:
            raise cleanup_error


def _materialize_tcga_revision_inputs(  # noqa: PLR0913
    raw_maf_root: str | Path,
    population_root: str | Path,
    datahub_git_dir: str | Path,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    out: str | Path,
    selected_cohorts: Sequence[str],
) -> Path:
    raw_root = _absolute_unresolved(raw_maf_root)
    population = _absolute_unresolved(population_root)
    git_dir = _absolute_unresolved(datahub_git_dir)
    approval_path = _absolute_unresolved(approval_manifest)
    output_root = _absolute_unresolved(out)
    _require_directory(raw_root, label="raw MAF root")
    approval = _secure_approval(approval_path, expected_approval_sha256)
    _validate_datahub_git_dir(git_dir)
    population_manifest, axes = _validate_population_bundle(
        population,
        selected_cohorts,
        closed_sources=False,
    )
    source_paths = _source_dependencies()
    source_bytes = {
        name: _read_regular_bytes(path, label=f"{name} implementation source")
        for name, path in source_paths.items()
    }
    canonicalizer_sha256 = hashlib.sha256(source_bytes["canonicalizer"]).hexdigest()
    frozen_canonicalizer = _load_canonicalizer_snapshot(
        source_bytes["canonicalizer"],
        path=source_paths["canonicalizer"],
    )
    frozen_policy = getattr(
        frozen_canonicalizer,
        "TCGA_DUPLICATE_RESOLUTION_POLICY",
        None,
    )
    if frozen_policy is None:
        msg = "Frozen canonicalizer lacks its duplicate-resolution policy."
        raise RevisionInputError(msg)
    frozen_policy_record = json.loads(_canonical_json(asdict(frozen_policy)))
    population_manifest_sha256 = hashlib.sha256(
        _canonical_json(population_manifest) + b"\n",
    ).hexdigest()
    signed = _validate_signed_contracts(
        approval_path,
        approval,
        _expected_d1_artifact(canonicalizer_sha256, frozen_policy_record),
        _expected_d2_artifact(
            population,
            population_manifest,
            axes,
            selected_cohorts,
            population_manifest_sha256=population_manifest_sha256,
        ),
    )

    parent_descriptor = _ensure_directory_fd(
        output_root.parent,
        label="output parent",
    )
    try:
        _preflight_materialization_resources(
            raw_root,
            output_root.parent,
            selected_cohorts,
        )
        with _exclusive_publish_claim_at(parent_descriptor, output_root.name):
            try:
                os.stat(
                    output_root.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                msg = f"Refusing to reuse existing output root: {output_root}"
                raise FileExistsError(msg)
            staging_name = f".{output_root.name}.staging-{uuid.uuid4().hex}"
            os.mkdir(
                staging_name,
                mode=_STAGING_DIRECTORY_MODE,
                dir_fd=parent_descriptor,
            )
            staging_root = output_root.parent / staging_name
            staging_descriptor = os.open(
                staging_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            published = False
            try:
                if not _directory_path_matches_fd(
                    output_root.parent,
                    parent_descriptor,
                ):
                    msg = (
                        "Output parent changed after its secure descriptor was opened."
                    )
                    raise RevisionInputError(msg)
                for name in (
                    "raw_mafs",
                    "mafs",
                    "case_lists",
                    "population",
                    "cohorts",
                ):
                    (staging_root / name).mkdir()
                scratch_root = staging_root / ".scratch"
                scratch_root.mkdir(mode=_STAGING_DIRECTORY_MODE)
                implementation = _snapshot_implementation(staging_root, source_bytes)
                _copy_population_bundle(population, staging_root, selected_cohorts)
                copied_population_manifest, copied_axes = _validate_population_bundle(
                    staging_root / "population",
                    selected_cohorts,
                    closed_sources=True,
                    expected_manifest_sha256=population_manifest_sha256,
                )
                if (
                    not _exact_json_equal(
                        copied_population_manifest,
                        population_manifest,
                    )
                    or copied_axes != axes
                ):
                    msg = "Population authority changed while its snapshot was copied."
                    raise RevisionInputError(msg)
                copied_contracts = _copy_signed_contracts(staging_root, signed)
                approval_closure = _copy_approval_closure(
                    staging_root,
                    approval_path,
                    approval,
                    expected_approval_sha256,
                )
                manifests = [
                    _materialize_cohort(
                        cohort,
                        raw_maf_root=raw_root,
                        population_root=staging_root / "population",
                        datahub_git_dir=git_dir,
                        staging_root=staging_root,
                        scratch_root=scratch_root,
                        selected_axis=copied_axes[cohort],
                        approval=approval,
                        frozen_canonicalizer=frozen_canonicalizer,
                        duplicate_policy=frozen_policy_record,
                    )
                    for cohort in selected_cohorts
                ]
                scratch_root.rmdir()
                cohort_records = [
                    {
                        "cohort": cohort,
                        "manifest": _file_record(
                            staging_root / "cohorts" / f"{cohort}.json",
                            display_path=f"cohorts/{cohort}.json",
                        ),
                    }
                    for cohort in selected_cohorts
                ]
                root_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "contract": INPUT_CONTRACT,
                    "authority": {
                        "approval": _approval_record(approval),
                        "approval_closure": approval_closure,
                        "signed_contracts": copied_contracts,
                    },
                    "source": {
                        "repository": "https://github.com/cBioPortal/datahub",
                        "commit": TCGA_DATAHUB_COMMIT,
                        "tree": TCGA_DATAHUB_TREE,
                    },
                    "population": {
                        "contract": copied_population_manifest["contract"],
                        "manifest": _file_record(
                            staging_root / "population" / "population_manifest.json",
                            display_path="population/population_manifest.json",
                        ),
                    },
                    "cohorts": list(selected_cohorts),
                    "cohort_count": len(selected_cohorts),
                    "cohort_manifests": cohort_records,
                    "totals": {
                        "raw_rows": sum(
                            manifest["row_accounting"]["raw_rows"]
                            for manifest in manifests
                        ),
                        "unselected_sample_rows": sum(
                            manifest["row_accounting"]["unselected_sample_rows"]
                            for manifest in manifests
                        ),
                        "selected_rows_before_deduplication": sum(
                            manifest["row_accounting"][
                                "selected_rows_before_deduplication"
                            ]
                            for manifest in manifests
                        ),
                        "duplicate_excess_rows_removed": sum(
                            manifest["row_accounting"]["duplicate_excess_rows_removed"]
                            for manifest in manifests
                        ),
                        "canonical_output_rows": sum(
                            manifest["row_accounting"]["canonical_output_rows"]
                            for manifest in manifests
                        ),
                        "selected_sample_count": sum(
                            len(copied_axes[cohort]) for cohort in selected_cohorts
                        ),
                    },
                    "implementation": implementation,
                    "scope": {
                        "result_blind": True,
                        "association_outputs_opened": False,
                        "providers_invoked": [],
                    },
                    "inventory": _expected_inventory(
                        selected_cohorts,
                        approval_files=(
                            str(approval_closure["manifest"]["path"]),
                            *(
                                str(record["path"])
                                for record in approval_closure["files"]
                            ),
                        ),
                    ),
                }
                manifest_path = staging_root / "input_manifest.json"
                _write_json_atomic(manifest_path, root_manifest)
                manifest_bytes = _read_regular_bytes(
                    manifest_path,
                    label="staged input manifest",
                )
                _freeze_tree_read_only(staging_root)
                _validate_materialized_input_bundle(
                    staging_root,
                    hashlib.sha256(manifest_bytes).hexdigest(),
                    approval_path,
                    expected_approval_sha256,
                    selected_cohorts,
                    require_current_execution_environment=True,
                )
                if not _directory_path_matches_fd(
                    output_root.parent,
                    parent_descriptor,
                ) or not _directory_path_matches_fd(staging_root, staging_descriptor):
                    msg = (
                        "Output parent or staging directory changed before publication."
                    )
                    raise RevisionInputError(msg)
                _rename_exclusive_at(
                    parent_descriptor,
                    staging_name,
                    parent_descriptor,
                    output_root.name,
                )
                os.fsync(parent_descriptor)
                published = True
            finally:
                os.close(staging_descriptor)
                if (
                    not published
                    and _directory_path_matches_fd(
                        output_root.parent,
                        parent_descriptor,
                    )
                    and staging_root.exists()
                ):
                    _restore_tree_owner_write(staging_root)
                    shutil.rmtree(staging_root)
        return output_root
    finally:
        os.close(parent_descriptor)


def materialize_tcga_revision_inputs(  # noqa: PLR0913
    raw_maf_root: str | Path,
    population_root: str | Path,
    datahub_git_dir: str | Path,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    out: str | Path,
    *,
    cohorts: Sequence[str] | None = None,
) -> Path:
    """Publish the exact bundle in an isolated content-addressed child."""
    selected_cohorts = _require_full_cohort_family(cohorts)
    output_root = _absolute_unresolved(out)
    source_bytes = {
        name: _read_regular_bytes(path, label=f"{name} execution source")
        for name, path in _source_dependencies().items()
    }
    snapshot_root, snapshot_sha256 = _create_execution_snapshot(
        output_root.parent,
        source_bytes,
        require_live_runtime=True,
    )
    try:
        response = _run_isolated_snapshot(
            snapshot_root,
            snapshot_sha256,
            {
                "action": "materialize",
                "raw_maf_root": _absolute_unresolved(raw_maf_root).as_posix(),
                "population_root": _absolute_unresolved(population_root).as_posix(),
                "datahub_git_dir": _absolute_unresolved(datahub_git_dir).as_posix(),
                "approval_manifest": _absolute_unresolved(
                    approval_manifest,
                ).as_posix(),
                "expected_approval_sha256": expected_approval_sha256,
                "out": output_root.as_posix(),
                "cohorts": list(selected_cohorts),
            },
        )
        if response != {"output": output_root.as_posix()}:
            msg = "Isolated materialization child returned an invalid receipt."
            raise RevisionInputError(msg)
        return output_root
    finally:
        _cleanup_execution_snapshot(snapshot_root)


def _materialize_tcga_revision_inputs_for_test(  # noqa: PLR0913
    raw_maf_root: str | Path,
    population_root: str | Path,
    datahub_git_dir: str | Path,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    out: str | Path,
    *,
    cohorts: Sequence[str],
) -> Path:
    """Materialize a deliberately non-public subset for synthetic tests only."""
    return _materialize_tcga_revision_inputs(
        raw_maf_root,
        population_root,
        datahub_git_dir,
        approval_manifest,
        expected_approval_sha256,
        out,
        _validate_cohorts(cohorts),
    )


def _require_git_blob_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{label} must be a Git-blob receipt."
        raise RevisionInputError(msg)
    _require_exact_keys(value, {"object_id", "bytes", "sha256"}, label=label)
    if (
        not isinstance(value["object_id"], str)
        or _GIT_OBJECT_ID_PATTERN.fullmatch(value["object_id"]) is None
        or not isinstance(value["bytes"], int)
        or isinstance(value["bytes"], bool)
        or value["bytes"] <= 0
        or not isinstance(value["sha256"], str)
        or _SHA256_PATTERN.fullmatch(value["sha256"]) is None
    ):
        msg = f"{label} is invalid."
        raise RevisionInputError(msg)
    return value


def _require_git_lfs_blob_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{label} must be a Git-LFS pointer receipt."
        raise RevisionInputError(msg)
    _require_exact_keys(
        value,
        {"object_id", "bytes", "sha256", "lfs_payload"},
        label=label,
    )
    if (
        not isinstance(value["object_id"], str)
        or _GIT_OBJECT_ID_PATTERN.fullmatch(value["object_id"]) is None
        or not isinstance(value["bytes"], int)
        or isinstance(value["bytes"], bool)
        or value["bytes"] <= 0
        or value["bytes"] > _MAX_GIT_LFS_POINTER_BYTES
        or not isinstance(value["sha256"], str)
        or _SHA256_PATTERN.fullmatch(value["sha256"]) is None
    ):
        msg = f"{label} Git pointer identity is invalid."
        raise RevisionInputError(msg)
    payload = value["lfs_payload"]
    if not isinstance(payload, dict):
        msg = f"{label}.lfs_payload must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(payload, {"bytes", "sha256"}, label=f"{label}.lfs_payload")
    if (
        not isinstance(payload["bytes"], int)
        or isinstance(payload["bytes"], bool)
        or payload["bytes"] <= 0
        or not isinstance(payload["sha256"], str)
        or _SHA256_PATTERN.fullmatch(payload["sha256"]) is None
    ):
        msg = f"{label} LFS payload identity is invalid."
        raise RevisionInputError(msg)
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{payload['sha256']}\n"
        f"size {payload['bytes']}\n"
    ).encode("ascii")
    if (
        len(pointer) != value["bytes"]
        or hashlib.sha256(pointer).hexdigest() != value["sha256"]
    ):
        msg = f"{label} does not reconstruct its exact canonical pointer bytes."
        raise RevisionInputError(msg)
    return value


def _validate_child_manifest(  # noqa: PLR0913
    bundle_root: Path,
    cohort: str,
    record: Mapping[str, Any],
    approval: RevisionApproval,
    axes: Mapping[str, tuple[str, ...]],
    source_snapshots: Mapping[str, tuple[Path, bytes]],
    frozen_canonicalizer: object,
    duplicate_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    _require_exact_keys(
        record,
        {"cohort", "manifest"},
        label=f"{cohort} root cohort record",
    )
    if record["cohort"] != cohort:
        msg = f"Canonical input cohort record is misbound: {cohort}"
        raise RevisionInputError(msg)
    child_path, child = _read_verified_json_file_record(
        bundle_root,
        record["manifest"],
        label=f"{cohort} canonical input manifest",
    )
    if child_path != bundle_root / "cohorts" / f"{cohort}.json":
        msg = f"Canonical input child path is invalid: {cohort}"
        raise RevisionInputError(msg)
    _require_exact_keys(
        child,
        {
            "schema_version",
            "contract",
            "cohort",
            "approval",
            "source",
            "population",
            "transformation",
            "row_accounting",
            "invariants",
            "output",
        },
        label=f"{cohort} child manifest",
    )
    if (
        child["schema_version"] != SCHEMA_VERSION
        or child["contract"] != INPUT_CONTRACT
        or child["cohort"] != cohort
        or not _exact_json_equal(child["approval"], _approval_record(approval))
    ):
        msg = f"Canonical input child authority is invalid: {cohort}"
        raise RevisionInputError(msg)
    source = child["source"]
    population = child["population"]
    transformation = child["transformation"]
    accounting = child["row_accounting"]
    invariants = child["invariants"]
    output = child["output"]
    for value, keys, label in (
        (
            source,
            {
                "repository",
                "commit",
                "tree",
                "repository_path",
                "raw_maf",
                "raw_git_blob",
                "case_list_repository_path",
                "case_list_git_blob",
                "case_list",
            },
            "source",
        ),
        (
            population,
            {
                "manifest",
                "sample_axis",
                "sample_count",
                "ordered_sample_axis_sha256",
                "all_zero_rows_required_for_samples_without_retained_events",
            },
            "population",
        ),
        (
            transformation,
            {
                "canonicalizer",
                "materializer",
                "serializer",
                "duplicate_resolution_policy",
                "raw_maf_sha256",
                "sample_axis_file_sha256",
                "canonical_maf_sha256",
            },
            "transformation",
        ),
        (
            accounting,
            {
                "raw_rows",
                "unselected_sample_rows",
                "selected_rows_before_deduplication",
                "duplicate_excess_rows_removed",
                "canonical_output_rows",
                "multiallelic_coordinate_groups_preserved",
                "duplicate_group_count",
                "semantic_agreement_group_count",
                "ignored_conflict_group_count",
                "frozen_effect_resolution_group_count",
                "resolved_conflict_groups_by_column",
                "selected_mutsig_effect_groups",
                "unresolved_semantic_conflicts",
            },
            "row accounting",
        ),
        (
            invariants,
            {
                "raw_row_identity_closed",
                "deduplication_row_identity_closed",
                "output_full_key_duplicates",
                "output_samples_outside_selected_axis",
                "deterministic_order",
                "row_order_invariance_contract",
                "association_outputs_opened",
            },
            "invariants",
        ),
        (output, {"canonical_maf", "rows", "ordered_columns_sha256"}, "output"),
    ):
        if not isinstance(value, dict):
            msg = f"{cohort} {label} must be an object."
            raise RevisionInputError(msg)
        _require_exact_keys(value, keys, label=f"{cohort} {label}")
    expected_source_identity = {
        "repository": "https://github.com/cBioPortal/datahub",
        "commit": TCGA_DATAHUB_COMMIT,
        "tree": TCGA_DATAHUB_TREE,
        "repository_path": tcga_datahub_public_path(
            cohort,
            "data_mutations.txt",
        ).as_posix(),
        "case_list_repository_path": tcga_datahub_case_list_path(cohort).as_posix(),
    }
    if any(source[key] != value for key, value in expected_source_identity.items()):
        msg = f"Canonical input source identity is invalid: {cohort}"
        raise RevisionInputError(msg)
    raw_blob = _require_git_lfs_blob_record(
        source["raw_git_blob"],
        label=f"{cohort} raw Git LFS blob",
    )
    case_blob = _require_git_blob_record(
        source["case_list_git_blob"],
        label=f"{cohort} case-list Git blob",
    )
    raw_path = _verify_file_record(
        bundle_root,
        source["raw_maf"],
        label=f"{cohort} raw MAF",
    )
    canonical_path = _verify_file_record(
        bundle_root,
        output["canonical_maf"],
        label=f"{cohort} canonical MAF",
    )
    case_list_path, case_list_bytes = _verified_file_bytes(
        bundle_root,
        source["case_list"],
        label=f"{cohort} case list",
    )
    axis_path, axis_bytes = _verified_file_bytes(
        bundle_root,
        population["sample_axis"],
        label=f"{cohort} sample axis",
    )
    population_path, population_child = _read_verified_json_file_record(
        bundle_root,
        population["manifest"],
        label=f"{cohort} population manifest",
    )
    expected_paths = {
        raw_path: bundle_root / "raw_mafs" / f"{cohort}.maf",
        canonical_path: bundle_root / "mafs" / f"{cohort}.maf",
        case_list_path: bundle_root / "case_lists" / f"{cohort}.txt",
        axis_path: bundle_root / "population" / cohort / "sample_axis.txt",
        population_path: (
            bundle_root / "population" / cohort / "population_manifest.json"
        ),
    }
    if any(observed != expected for observed, expected in expected_paths.items()):
        msg = f"Canonical input child file path is invalid: {cohort}"
        raise RevisionInputError(msg)
    expected_axis = axes[cohort]
    if (
        axis_bytes != ("\n".join(expected_axis) + "\n").encode()
        or type(population["sample_count"]) is not int
        or population["sample_count"] != len(expected_axis)
        or population["ordered_sample_axis_sha256"] != _sequence_sha256(expected_axis)
        or population["all_zero_rows_required_for_samples_without_retained_events"]
        is not True
    ):
        msg = f"Canonical input sample axis differs from population: {cohort}"
        raise RevisionInputError(msg)
    if (
        raw_blob["lfs_payload"]["bytes"] != source["raw_maf"]["bytes"]
        or raw_blob["lfs_payload"]["sha256"] != source["raw_maf"]["sha256"]
        or case_blob["bytes"] != population_child["source"]["case_list_bytes"]
        or case_blob["sha256"] != population_child["source"]["case_list_sha256"]
        or case_blob["bytes"] != source["case_list"]["bytes"]
        or case_blob["sha256"] != source["case_list"]["sha256"]
    ):
        msg = f"Canonical input Git-blob lineage is invalid: {cohort}"
        raise RevisionInputError(msg)
    canonicalizer_path, canonicalizer_bytes = source_snapshots["canonicalizer"]
    materializer_path, materializer_bytes = source_snapshots["materializer"]
    expected_transformation = {
        "canonicalizer": {
            "path": _SOURCE_SNAPSHOT_PATHS["canonicalizer"],
            "bytes": len(canonicalizer_bytes),
            "sha256": hashlib.sha256(canonicalizer_bytes).hexdigest(),
        },
        "materializer": {
            "path": _SOURCE_SNAPSHOT_PATHS["materializer"],
            "bytes": len(materializer_bytes),
            "sha256": hashlib.sha256(materializer_bytes).hexdigest(),
        },
        "serializer": SERIALIZER_CONTRACT,
        "duplicate_resolution_policy": dict(duplicate_policy),
        "raw_maf_sha256": source["raw_maf"]["sha256"],
        "sample_axis_file_sha256": population["sample_axis"]["sha256"],
        "canonical_maf_sha256": output["canonical_maf"]["sha256"],
    }
    if not _exact_json_equal(transformation, expected_transformation):
        msg = f"Canonical input transformation contract drifted: {cohort}"
        raise RevisionInputError(msg)
    if canonicalizer_path != bundle_root.joinpath(
        *PurePosixPath(_SOURCE_SNAPSHOT_PATHS["canonicalizer"]).parts,
    ) or materializer_path != bundle_root.joinpath(
        *PurePosixPath(_SOURCE_SNAPSHOT_PATHS["materializer"]).parts,
    ):
        msg = "Frozen implementation snapshot paths changed during replay."
        raise RevisionInputError(msg)
    case_samples = parse_tcga_sequenced_case_list(case_list_bytes, cohort)
    if not set(expected_axis) <= set(case_samples):
        msg = f"Canonical input population is outside its case list: {cohort}"
        raise RevisionInputError(msg)
    available_memory = _available_memory_bytes()
    if available_memory < _MIN_AVAILABLE_MEMORY_BYTES:
        msg = f"Insufficient available memory to validate {cohort} canonically."
        raise RevisionInputError(msg)
    raw_size = int(source["raw_maf"]["bytes"])
    with _private_validation_scratch(bundle_root, cohort) as scratch_root:
        required_disk = raw_size * 6 + 512 * 1024 * 1024
        free_disk = shutil.disk_usage(scratch_root).free
        if free_disk < required_disk:
            msg = (
                f"Insufficient scratch disk to validate {cohort}: "
                f"need {required_disk}, observed {free_disk}."
            )
            raise RevisionInputError(msg)
        regenerated_path = scratch_root / f"{cohort}.maf"
        sqlite_path = scratch_root / f"{cohort}.sqlite3"
        streamed = _stream_canonicalize_maf(
            raw_path,
            regenerated_path,
            sqlite_path,
            raw_copy_path=None,
            expected_raw_receipt=_FileReceipt(
                bytes=int(source["raw_maf"]["bytes"]),
                sha256=str(source["raw_maf"]["sha256"]),
            ),
            selected_samples=frozenset(expected_axis),
            case_samples=frozenset(case_samples),
            frozen_canonicalizer=frozen_canonicalizer,
        )
    resolution_audit = streamed.audit
    if (
        streamed.output_receipt.bytes != output["canonical_maf"]["bytes"]
        or streamed.output_receipt.sha256 != output["canonical_maf"]["sha256"]
    ):
        msg = f"Canonical input does not reproduce from signed authority: {cohort}"
        raise RevisionInputError(msg)
    expected_accounting = {
        "raw_rows": streamed.raw_rows,
        "unselected_sample_rows": streamed.raw_rows - streamed.selected_rows,
        "selected_rows_before_deduplication": streamed.selected_rows,
        "duplicate_excess_rows_removed": (
            streamed.selected_rows - streamed.output_rows
        ),
        "canonical_output_rows": streamed.output_rows,
        "multiallelic_coordinate_groups_preserved": (
            streamed.multiallelic_coordinate_groups
        ),
        "duplicate_group_count": resolution_audit.duplicate_group_count,
        "semantic_agreement_group_count": (
            resolution_audit.semantic_agreement_group_count
        ),
        "ignored_conflict_group_count": (resolution_audit.ignored_conflict_group_count),
        "frozen_effect_resolution_group_count": (
            resolution_audit.frozen_effect_resolution_group_count
        ),
        "resolved_conflict_groups_by_column": dict(
            resolution_audit.resolved_conflict_groups_by_column,
        ),
        "selected_mutsig_effect_groups": dict(
            resolution_audit.selected_mutsig_effect_groups,
        ),
        "unresolved_semantic_conflicts": 0,
    }
    expected_invariants = {
        "raw_row_identity_closed": True,
        "deduplication_row_identity_closed": True,
        "output_full_key_duplicates": 0,
        "output_samples_outside_selected_axis": 0,
        "deterministic_order": True,
        "row_order_invariance_contract": STREAMING_CANONICALIZATION_CONTRACT,
        "association_outputs_opened": False,
    }
    expected_output = {
        "canonical_maf": output["canonical_maf"],
        "rows": streamed.output_rows,
        "ordered_columns_sha256": streamed.ordered_columns_sha256,
    }
    if (
        not _exact_json_equal(accounting, expected_accounting)
        or not _exact_json_equal(invariants, expected_invariants)
        or not _exact_json_equal(output, expected_output)
    ):
        msg = f"Canonical input child claims do not exactly replay: {cohort}"
        raise RevisionInputError(msg)
    return child, {
        "raw_rows": streamed.raw_rows,
        "unselected_sample_rows": streamed.raw_rows - streamed.selected_rows,
        "selected_rows_before_deduplication": streamed.selected_rows,
        "duplicate_excess_rows_removed": (
            streamed.selected_rows - streamed.output_rows
        ),
        "canonical_output_rows": streamed.output_rows,
        "selected_sample_count": len(expected_axis),
    }


def _validate_materialized_input_bundle(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    selected_cohorts: Sequence[str],
    *,
    require_current_execution_environment: bool,
) -> dict[str, Any]:
    if type(require_current_execution_environment) is not bool:
        msg = "require_current_execution_environment must be an exact boolean."
        raise RevisionInputError(msg)
    if _SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None:
        msg = "Expected input-manifest SHA-256 is invalid."
        raise RevisionInputError(msg)
    bundle_root = _absolute_unresolved(root)
    _require_directory(bundle_root, label="input bundle root")
    manifest_path = bundle_root / "input_manifest.json"
    manifest = _read_json_with_sha256(
        manifest_path,
        expected_manifest_sha256,
        label="Input manifest independent SHA-256",
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "contract",
            "authority",
            "source",
            "population",
            "cohorts",
            "cohort_count",
            "cohort_manifests",
            "totals",
            "implementation",
            "scope",
            "inventory",
        },
        label="input root manifest",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["contract"] != INPUT_CONTRACT
        or not _exact_json_equal(manifest["cohorts"], list(selected_cohorts))
        or type(manifest["cohort_count"]) is not int
        or manifest["cohort_count"] != len(selected_cohorts)
    ):
        msg = "Canonical input root manifest violates its frozen family."
        raise RevisionInputError(msg)
    source = manifest["source"]
    population = manifest["population"]
    authority = manifest["authority"]
    records = manifest["cohort_manifests"]
    totals = manifest["totals"]
    for value, keys, label in (
        (source, {"repository", "commit", "tree"}, "root source"),
        (population, {"contract", "manifest"}, "root population"),
        (
            authority,
            {"approval", "approval_closure", "signed_contracts"},
            "root authority",
        ),
        (
            totals,
            {
                "raw_rows",
                "unselected_sample_rows",
                "selected_rows_before_deduplication",
                "duplicate_excess_rows_removed",
                "canonical_output_rows",
                "selected_sample_count",
            },
            "root totals",
        ),
    ):
        if not isinstance(value, dict):
            msg = f"{label} must be an object."
            raise RevisionInputError(msg)
        _require_exact_keys(value, keys, label=label)
    approval_inventory_paths = _approval_closure_inventory_paths(
        authority["approval_closure"],
    )
    expected_inventory = _expected_inventory(
        selected_cohorts,
        approval_files=approval_inventory_paths,
    )
    if not _exact_json_equal(
        manifest["inventory"],
        expected_inventory,
    ) or not _exact_json_equal(
        _filesystem_inventory(bundle_root),
        expected_inventory,
    ):
        msg = "Canonical input filesystem differs from its closed inventory."
        raise RevisionInputError(msg)
    if not _exact_json_equal(
        source,
        {
            "repository": "https://github.com/cBioPortal/datahub",
            "commit": TCGA_DATAHUB_COMMIT,
            "tree": TCGA_DATAHUB_TREE,
        },
    ):
        msg = "Canonical input root source is invalid."
        raise RevisionInputError(msg)
    source_snapshots = _validate_implementation(
        bundle_root,
        manifest["implementation"],
        require_current_execution_environment=require_current_execution_environment,
    )
    population_path, population_manifest_bytes = _verified_file_bytes(
        bundle_root,
        population["manifest"],
        label="root population manifest",
    )
    if (
        population["contract"] != POPULATION_CONTRACT
        or population_path != bundle_root / "population" / "population_manifest.json"
    ):
        msg = "Canonical input root population binding is invalid."
        raise RevisionInputError(msg)
    population_manifest, axes = _validate_population_bundle(
        bundle_root / "population",
        selected_cohorts,
        closed_sources=True,
        expected_manifest_sha256=str(population["manifest"]["sha256"]),
        root_manifest_bytes=population_manifest_bytes,
    )
    external_approval_path = _absolute_unresolved(approval_manifest)
    approval, approval_path, validated_approval_paths = _validate_approval_closure(
        bundle_root,
        authority["approval_closure"],
        expected_approval_sha256,
        external_approval_path=external_approval_path,
        require_current_execution_environment=require_current_execution_environment,
    )
    if validated_approval_paths != approval_inventory_paths:
        msg = "Approval closure inventory changed during validation."
        raise RevisionInputError(msg)
    canonicalizer_sha256 = str(
        manifest["implementation"]["sources"]["canonicalizer"]["sha256"],
    )
    _preauthorize_canonicalizer_snapshot(
        approval_path,
        approval,
        canonicalizer_sha256,
    )
    frozen_canonicalizer = _load_canonicalizer_snapshot(
        source_snapshots["canonicalizer"][1],
        path=source_snapshots["canonicalizer"][0],
    )
    frozen_policy = getattr(
        frozen_canonicalizer,
        "TCGA_DUPLICATE_RESOLUTION_POLICY",
        None,
    )
    if frozen_policy is None:
        msg = "Frozen canonicalizer lacks its duplicate-resolution policy."
        raise RevisionInputError(msg)
    frozen_policy_record = json.loads(_canonical_json(asdict(frozen_policy)))
    expected_signed = _validate_signed_contracts(
        approval_path,
        approval,
        _expected_d1_artifact(
            canonicalizer_sha256,
            frozen_policy_record,
        ),
        _expected_d2_artifact(
            bundle_root / "population",
            population_manifest,
            axes,
            selected_cohorts,
            population_manifest_sha256=str(population["manifest"]["sha256"]),
        ),
    )
    if not _exact_json_equal(authority["approval"], _approval_record(approval)):
        msg = "Canonical input approval authority is invalid."
        raise RevisionInputError(msg)
    signed_contracts = authority["signed_contracts"]
    if not isinstance(signed_contracts, dict):
        msg = "Signed input contracts must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(signed_contracts, {"D1", "D2"}, label="signed contracts")
    for decision_id in ("D1", "D2"):
        signed_record = signed_contracts[decision_id]
        if not isinstance(signed_record, dict):
            msg = f"Signed {decision_id} contract must be an object."
            raise RevisionInputError(msg)
        _require_exact_keys(
            signed_record,
            {"artifact", "content"},
            label=f"signed {decision_id}",
        )
        artifact_path, artifact_content = _read_verified_json_file_record(
            bundle_root,
            signed_record["artifact"],
            label=f"copied {decision_id} artifact",
        )
        expected_path = bundle_root / "authority" / f"{decision_id}.json"
        expected_content = expected_signed[decision_id][0]
        if (
            artifact_path != expected_path
            or not _exact_json_equal(signed_record["content"], expected_content)
            or not _exact_json_equal(artifact_content, expected_content)
        ):
            msg = f"Copied signed {decision_id} contract is invalid."
            raise RevisionInputError(msg)
    if (
        not isinstance(records, list)
        or len(records) != len(selected_cohorts)
        or [record.get("cohort") for record in records if isinstance(record, dict)]
        != list(selected_cohorts)
    ):
        msg = "Canonical input root cohort-manifest family is invalid."
        raise RevisionInputError(msg)
    observed_totals = dict.fromkeys(totals, 0)
    for cohort, record in zip(selected_cohorts, records, strict=True):
        if not isinstance(record, dict):
            msg = f"Canonical input root cohort record is invalid: {cohort}"
            raise RevisionInputError(msg)
        _, child_totals = _validate_child_manifest(
            bundle_root,
            cohort,
            record,
            approval,
            axes,
            source_snapshots,
            frozen_canonicalizer,
            expected_signed["D1"][0]["payload"]["duplicate_resolution_policy"],
        )
        for key, value in child_totals.items():
            observed_totals[key] += value
    if not _exact_json_equal(totals, observed_totals):
        msg = "Canonical input root totals do not equal replayed cohort totals."
        raise RevisionInputError(msg)
    if not _exact_json_equal(
        manifest["scope"],
        {
            "result_blind": True,
            "association_outputs_opened": False,
            "providers_invoked": [],
        },
    ):
        msg = "Canonical input bundle scope is not result-blind."
        raise RevisionInputError(msg)
    return manifest


def validate_materialized_input_bundle(
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    *,
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    """Validate in the bundle's isolated snapshot; historical mode is relocatable."""
    if type(require_current_execution_environment) is not bool:
        msg = "require_current_execution_environment must be an exact boolean."
        raise RevisionInputError(msg)
    selected_cohorts = _require_full_cohort_family(None)
    bundle_root = _absolute_unresolved(root)
    root_manifest = _read_json_with_sha256(
        bundle_root / "input_manifest.json",
        expected_manifest_sha256,
        label="input manifest bootstrap",
    )
    implementation = root_manifest.get("implementation")
    if not isinstance(implementation, dict) or not isinstance(
        implementation.get("sources"),
        dict,
    ):
        msg = "Input manifest lacks its implementation source closure."
        raise RevisionInputError(msg)
    source_bytes: dict[str, bytes] = {}
    for name, display_path in _SOURCE_SNAPSHOT_PATHS.items():
        record = implementation["sources"].get(name)
        path, content = _verified_file_bytes(
            bundle_root,
            record,
            label=f"historical {name} snapshot",
        )
        if path != bundle_root.joinpath(*PurePosixPath(display_path).parts):
            msg = f"Historical implementation source path is invalid: {name}"
            raise RevisionInputError(msg)
        source_bytes[name] = content
    if require_current_execution_environment:
        for name, live_path in _source_dependencies().items():
            live = _read_regular_bytes(live_path, label=f"live {name} source")
            if live != source_bytes[name]:
                msg = f"Current execution source differs from snapshot: {name}"
                raise RevisionInputError(msg)
    snapshot_root, snapshot_sha256 = _create_execution_snapshot(
        bundle_root.parent,
        source_bytes,
        require_live_runtime=require_current_execution_environment,
    )
    try:
        response = _run_isolated_snapshot(
            snapshot_root,
            snapshot_sha256,
            {
                "action": "validate",
                "root": bundle_root.as_posix(),
                "expected_manifest_sha256": expected_manifest_sha256,
                "approval_manifest": _absolute_unresolved(
                    approval_manifest,
                ).as_posix(),
                "expected_approval_sha256": expected_approval_sha256,
                "cohorts": list(selected_cohorts),
                "require_current_execution_environment": (
                    require_current_execution_environment
                ),
            },
        )
        manifest = response.get("manifest")
        if not isinstance(manifest, dict) or not _exact_json_equal(
            manifest,
            root_manifest,
        ):
            msg = "Isolated validation child returned an invalid manifest receipt."
            raise RevisionInputError(msg)
        return manifest
    finally:
        _cleanup_execution_snapshot(snapshot_root)


def _validate_materialized_input_bundle_for_test(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    *,
    cohorts: Sequence[str],
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    return _validate_materialized_input_bundle(
        root,
        expected_manifest_sha256,
        approval_manifest,
        expected_approval_sha256,
        _validate_cohorts(cohorts),
        require_current_execution_environment=require_current_execution_environment,
    )


def build_full_input_validation_receipt(
    root: str | Path,
    validated_manifest: Mapping[str, Any],
    expected_manifest_sha256: str,
    expected_approval_sha256: str,
) -> dict[str, Any]:
    """Build the canonical receipt only after one full root replay succeeds."""
    if (
        _SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None
        or _SHA256_PATTERN.fullmatch(expected_approval_sha256) is None
    ):
        msg = "Full-validation receipt requires exact independent SHA-256 values."
        raise RevisionInputError(msg)
    bundle_root = _absolute_unresolved(root)
    on_disk = _read_json_with_sha256(
        bundle_root / "input_manifest.json",
        expected_manifest_sha256,
        label="full-validation receipt input manifest",
    )
    if not _exact_json_equal(on_disk, dict(validated_manifest)):
        msg = "Full-validation receipt manifest is not the validated on-disk root."
        raise RevisionInputError(msg)
    implementation = on_disk.get("implementation")
    inventory = on_disk.get("inventory")
    population = on_disk.get("population")
    cohorts = on_disk.get("cohorts")
    if (
        on_disk.get("schema_version") != SCHEMA_VERSION
        or on_disk.get("contract") != INPUT_CONTRACT
        or not isinstance(implementation, dict)
        or not isinstance(inventory, dict)
        or not isinstance(population, dict)
        or not isinstance(cohorts, list)
        or type(on_disk.get("cohort_count")) is not int
        or on_disk["cohort_count"] != len(cohorts)
    ):
        msg = "Full-validation receipt source manifest is structurally invalid."
        raise RevisionInputError(msg)
    population_record = population.get("manifest")
    if not isinstance(population_record, dict):
        msg = "Full-validation receipt lacks its population manifest."
        raise RevisionInputError(msg)
    return {
        "schema": "dialect-canonical-input-full-validation-receipt-v1",
        "validation_contract": "full-streaming-canonical-replay-v1",
        "input_manifest_sha256": expected_manifest_sha256,
        "approval_manifest_sha256": expected_approval_sha256,
        "population_manifest_sha256": population_record.get("sha256"),
        "implementation_sha256": hashlib.sha256(
            _canonical_json(implementation) + b"\n",
        ).hexdigest(),
        "inventory_sha256": hashlib.sha256(
            _canonical_json(inventory) + b"\n",
        ).hexdigest(),
        "ordered_cohorts_sha256": _sequence_sha256(str(cohort) for cohort in cohorts),
        "validated_cohort_count": len(cohorts),
        "association_outputs_opened": False,
    }


def full_input_validation_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the independent hash to pass from the full parent to fresh children."""
    return hashlib.sha256(_canonical_json(dict(receipt)) + b"\n").hexdigest()


def validate_materialized_input_bundle_with_receipt(
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    *,
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    """Perform the full replay and return its child-pin-ready canonical receipt."""
    manifest = validate_materialized_input_bundle(
        root,
        expected_manifest_sha256,
        approval_manifest,
        expected_approval_sha256,
        require_current_execution_environment=require_current_execution_environment,
    )
    receipt = build_full_input_validation_receipt(
        root,
        manifest,
        expected_manifest_sha256,
        expected_approval_sha256,
    )
    return {
        "manifest": manifest,
        "receipt": receipt,
        "receipt_sha256": full_input_validation_receipt_sha256(receipt),
    }


def _validate_full_input_validation_receipt(
    manifest: Mapping[str, Any],
    receipt: object,
    expected_receipt_sha256: str,
    expected_manifest_sha256: str,
    expected_approval_sha256: str,
) -> dict[str, Any]:
    if _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None:
        msg = "Expected full-validation receipt SHA-256 is invalid."
        raise RevisionInputError(msg)
    if not isinstance(receipt, dict):
        msg = "Full-validation receipt must be an object."
        raise RevisionInputError(msg)
    _require_exact_keys(
        receipt,
        {
            "schema",
            "validation_contract",
            "input_manifest_sha256",
            "approval_manifest_sha256",
            "population_manifest_sha256",
            "implementation_sha256",
            "inventory_sha256",
            "ordered_cohorts_sha256",
            "validated_cohort_count",
            "association_outputs_opened",
        },
        label="full-validation receipt",
    )
    observed_sha256 = full_input_validation_receipt_sha256(receipt)
    implementation = manifest.get("implementation")
    inventory = manifest.get("inventory")
    population = manifest.get("population")
    cohorts = manifest.get("cohorts")
    if (
        observed_sha256 != expected_receipt_sha256
        or receipt["schema"] != "dialect-canonical-input-full-validation-receipt-v1"
        or receipt["validation_contract"] != "full-streaming-canonical-replay-v1"
        or receipt["input_manifest_sha256"] != expected_manifest_sha256
        or receipt["approval_manifest_sha256"] != expected_approval_sha256
        or not isinstance(implementation, dict)
        or not isinstance(inventory, dict)
        or not isinstance(population, dict)
        or not isinstance(population.get("manifest"), dict)
        or receipt["population_manifest_sha256"] != population["manifest"].get("sha256")
        or receipt["implementation_sha256"]
        != hashlib.sha256(_canonical_json(implementation) + b"\n").hexdigest()
        or receipt["inventory_sha256"]
        != hashlib.sha256(_canonical_json(inventory) + b"\n").hexdigest()
        or not isinstance(cohorts, list)
        or receipt["ordered_cohorts_sha256"]
        != _sequence_sha256(str(cohort) for cohort in cohorts)
        or type(receipt["validated_cohort_count"]) is not int
        or receipt["validated_cohort_count"] != len(cohorts)
        or receipt["association_outputs_opened"] is not False
    ):
        msg = "Full-validation receipt is invalid or misbound."
        raise RevisionInputError(msg)
    return receipt


def _validate_population_root_for_cohort(
    population_root: Path,
    root_manifest_bytes: bytes,
    expected_manifest_sha256: str,
    cohort: str,
    expected_cohorts: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if hashlib.sha256(root_manifest_bytes).hexdigest() != expected_manifest_sha256:
        msg = "Population root bytes differ from the selected-cohort receipt."
        raise RevisionInputError(msg)
    root = _parse_json_document(
        root_manifest_bytes,
        path=population_root / "population_manifest.json",
    )
    _require_exact_keys(
        root,
        {
            "schema_version",
            "contract",
            "source",
            "selection_policy",
            "contract_source",
            "cohorts",
            "cohort_count",
            "cohort_manifests",
            "totals",
            "generator",
        },
        label="cohort-scoped population root",
    )
    records = root["cohort_manifests"]
    if (
        root["schema_version"] != POPULATION_SCHEMA_VERSION
        or root["contract"] != POPULATION_CONTRACT
        or root["cohorts"] != list(expected_cohorts)
        or type(root["cohort_count"]) is not int
        or root["cohort_count"] != len(expected_cohorts)
        or not isinstance(records, list)
        or [record.get("cohort") for record in records if isinstance(record, dict)]
        != list(expected_cohorts)
        or not _exact_json_equal(
            root["source"],
            {
                "repository": "https://github.com/cBioPortal/datahub",
                "commit": TCGA_DATAHUB_COMMIT,
                "tree": TCGA_DATAHUB_TREE,
            },
        )
        or not _exact_json_equal(
            root["selection_policy"],
            _population_selection_policy(),
        )
    ):
        msg = "Cohort-scoped population root violates its frozen family."
        raise RevisionInputError(msg)
    record = next(
        (
            item
            for item in records
            if isinstance(item, dict) and item.get("cohort") == cohort
        ),
        None,
    )
    if not isinstance(record, dict) or set(record) != {"cohort", "manifest_sha256"}:
        msg = f"Population root lacks a valid selected cohort record: {cohort}"
        raise RevisionInputError(msg)
    digest = record["manifest_sha256"]
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        msg = f"Population selected-cohort manifest digest is invalid: {cohort}"
        raise RevisionInputError(msg)
    return _validate_population_cohort(
        population_root,
        cohort,
        digest,
        closed_sources=True,
    )


def _validate_materialized_input_cohort_binding(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    cohort: str,
    full_validation_receipt: Mapping[str, Any],
    expected_full_validation_receipt_sha256: str,
    *,
    expected_cohorts: Sequence[str],
    require_current_execution_environment: bool,
) -> dict[str, Any]:
    if type(require_current_execution_environment) is not bool:
        msg = "require_current_execution_environment must be an exact boolean."
        raise RevisionInputError(msg)
    if cohort not in expected_cohorts:
        msg = f"Cohort is absent from the expected canonical family: {cohort}"
        raise RevisionInputError(msg)
    bundle_root = _absolute_unresolved(root)
    manifest = _read_json_with_sha256(
        bundle_root / "input_manifest.json",
        expected_manifest_sha256,
        label="cohort-scoped input manifest",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != INPUT_CONTRACT
        or manifest.get("cohorts") != list(expected_cohorts)
        or type(manifest.get("cohort_count")) is not int
        or manifest["cohort_count"] != len(expected_cohorts)
    ):
        msg = "Cohort-scoped input root violates its frozen family."
        raise RevisionInputError(msg)
    receipt = _validate_full_input_validation_receipt(
        manifest,
        full_validation_receipt,
        expected_full_validation_receipt_sha256,
        expected_manifest_sha256,
        expected_approval_sha256,
    )
    implementation = manifest.get("implementation")
    authority = manifest.get("authority")
    population = manifest.get("population")
    if (
        not isinstance(implementation, dict)
        or not isinstance(authority, dict)
        or not isinstance(population, dict)
    ):
        msg = "Cohort-scoped shared authority is structurally incomplete."
        raise RevisionInputError(msg)
    _validate_implementation(
        bundle_root,
        implementation,
        require_current_execution_environment=require_current_execution_environment,
    )
    _require_exact_keys(
        authority,
        {"approval", "approval_closure", "signed_contracts"},
        label="cohort-scoped root authority",
    )
    approval, _approval_path, _approval_files = _validate_approval_closure(
        bundle_root,
        authority["approval_closure"],
        expected_approval_sha256,
        external_approval_path=_absolute_unresolved(approval_manifest),
        require_current_execution_environment=require_current_execution_environment,
    )
    if not _exact_json_equal(authority["approval"], _approval_record(approval)):
        msg = "Cohort-scoped approval authority is invalid."
        raise RevisionInputError(msg)
    population_path, population_bytes = _verified_file_bytes(
        bundle_root,
        population.get("manifest"),
        label="cohort-scoped population root",
    )
    if (
        population.get("contract") != POPULATION_CONTRACT
        or population_path != bundle_root / "population" / "population_manifest.json"
    ):
        msg = "Cohort-scoped population binding is invalid."
        raise RevisionInputError(msg)
    _population_child, axis = _validate_population_root_for_cohort(
        bundle_root / "population",
        population_bytes,
        str(population["manifest"]["sha256"]),
        cohort,
        expected_cohorts,
    )
    binding = materialized_cohort_binding(bundle_root, manifest, cohort)
    axis_path = binding["sample_axis"]["path"]
    axis_bytes = _read_regular_bytes(axis_path, label=f"{cohort} selected sample axis")
    if axis_bytes != ("\n".join(axis) + "\n").encode("utf-8"):
        msg = f"Cohort-scoped sample axis differs from population: {cohort}"
        raise RevisionInputError(msg)
    return {
        "manifest": manifest,
        "binding": binding,
        "full_validation_receipt": receipt,
        "association_outputs_opened": False,
    }


def validate_materialized_input_cohort_binding(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    cohort: str,
    full_validation_receipt: Mapping[str, Any],
    expected_full_validation_receipt_sha256: str,
    *,
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    """Validate one cohort after a separately pinned full-parent replay receipt."""
    return _validate_materialized_input_cohort_binding(
        root,
        expected_manifest_sha256,
        approval_manifest,
        expected_approval_sha256,
        cohort,
        full_validation_receipt,
        expected_full_validation_receipt_sha256,
        expected_cohorts=_require_full_cohort_family(None),
        require_current_execution_environment=require_current_execution_environment,
    )


def _validate_materialized_input_cohort_binding_for_test(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    cohort: str,
    full_validation_receipt: Mapping[str, Any],
    expected_full_validation_receipt_sha256: str,
    *,
    cohorts: Sequence[str],
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    return _validate_materialized_input_cohort_binding(
        root,
        expected_manifest_sha256,
        approval_manifest,
        expected_approval_sha256,
        cohort,
        full_validation_receipt,
        expected_full_validation_receipt_sha256,
        expected_cohorts=_validate_cohorts(cohorts),
        require_current_execution_environment=require_current_execution_environment,
    )


def materialized_cohort_binding(
    root: str | Path,
    root_manifest: Mapping[str, Any],
    cohort: str,
) -> dict[str, Any]:
    """Return no-follow-verified paths and records from a validated root manifest."""
    bundle_root = _absolute_unresolved(root)
    _require_directory(bundle_root, label="input bundle root")
    manifest_bytes = _canonical_json(root_manifest) + b"\n"
    on_disk = _read_json_with_sha256(
        bundle_root / "input_manifest.json",
        hashlib.sha256(manifest_bytes).hexdigest(),
        label="passed root manifest is not the exact manifest on disk",
    )
    if on_disk != root_manifest:
        msg = "Passed root manifest is not the exact manifest on disk."
        raise RevisionInputError(msg)
    cohorts = on_disk.get("cohorts")
    records = on_disk.get("cohort_manifests")
    if (
        not isinstance(cohorts, list)
        or not isinstance(records, list)
        or len(records) != len(cohorts)
        or [record.get("cohort") for record in records if isinstance(record, dict)]
        != cohorts
    ):
        msg = "Validated root manifest lacks cohort records."
        raise RevisionInputError(msg)
    if cohort not in cohorts:
        msg = f"Cohort is absent from the validated bundle: {cohort}"
        raise RevisionInputError(msg)
    index = cohorts.index(cohort)
    record = records[index]
    if not isinstance(record, dict):
        msg = f"Cohort manifest record is invalid: {cohort}"
        raise RevisionInputError(msg)
    _require_exact_keys(
        record,
        {"cohort", "manifest"},
        label=f"{cohort} cohort manifest record",
    )
    if record["cohort"] != cohort:
        msg = f"Cohort manifest record is cross-bound: {cohort}"
        raise RevisionInputError(msg)
    child_path, child = _read_verified_json_file_record(
        bundle_root,
        record["manifest"],
        label=f"{cohort} child manifest",
    )
    output = child.get("output")
    population = child.get("population")
    if (
        child_path != bundle_root / "cohorts" / f"{cohort}.json"
        or child.get("schema_version") != SCHEMA_VERSION
        or child.get("contract") != INPUT_CONTRACT
        or child.get("cohort") != cohort
        or not isinstance(output, dict)
        or not isinstance(population, dict)
    ):
        msg = f"Cohort manifest path is invalid: {cohort}"
        raise RevisionInputError(msg)
    bindings = {
        "child_manifest": record["manifest"],
        "canonical_maf": output.get("canonical_maf"),
        "sample_axis": population.get("sample_axis"),
        "population_manifest": population.get("manifest"),
    }
    expected_paths = {
        "child_manifest": bundle_root / "cohorts" / f"{cohort}.json",
        "canonical_maf": bundle_root / "mafs" / f"{cohort}.maf",
        "sample_axis": bundle_root / "population" / cohort / "sample_axis.txt",
        "population_manifest": (
            bundle_root / "population" / cohort / "population_manifest.json"
        ),
    }
    expected_relative_paths = {
        name: path.relative_to(bundle_root).as_posix()
        for name, path in expected_paths.items()
    }
    result: dict[str, Any] = {"cohort": cohort}
    for name, file_record in bindings.items():
        if (
            not isinstance(file_record, dict)
            or file_record.get("path") != expected_relative_paths[name]
        ):
            msg = f"{cohort} {name} is cross-bound to an unrelated cohort."
            raise RevisionInputError(msg)
        path = _verify_file_record(
            bundle_root,
            file_record,
            label=f"{cohort} {name}",
        )
        if path != expected_paths[name]:
            msg = f"{cohort} {name} is cross-bound to an unrelated cohort."
            raise RevisionInputError(msg)
        result[name] = {"path": path, "file": dict(file_record)}
    return result


validate_tcga_revision_input_bundle = validate_materialized_input_bundle


def _run_internal_execution_request(
    snapshot_sha256: str,
    request_path: Path,
    expected_request_sha256: str,
    response_path: Path,
) -> None:
    _validate_execution_snapshot(snapshot_sha256)
    request = _read_json_with_sha256(
        request_path,
        expected_request_sha256,
        label="isolated execution request",
    )
    action = request.get("action")
    if action == "materialize":
        _require_exact_keys(
            request,
            {
                "action",
                "raw_maf_root",
                "population_root",
                "datahub_git_dir",
                "approval_manifest",
                "expected_approval_sha256",
                "out",
                "cohorts",
            },
            label="isolated materialization request",
        )
        cohorts = _validate_cohorts(request["cohorts"])
        output = _materialize_tcga_revision_inputs(
            str(request["raw_maf_root"]),
            str(request["population_root"]),
            str(request["datahub_git_dir"]),
            str(request["approval_manifest"]),
            str(request["expected_approval_sha256"]),
            str(request["out"]),
            cohorts,
        )
        _write_json_atomic(response_path, {"output": output.as_posix()})
        return
    if action == "validate":
        _require_exact_keys(
            request,
            {
                "action",
                "root",
                "expected_manifest_sha256",
                "approval_manifest",
                "expected_approval_sha256",
                "cohorts",
                "require_current_execution_environment",
            },
            label="isolated validation request",
        )
        require_current = request["require_current_execution_environment"]
        if type(require_current) is not bool:
            msg = "require_current_execution_environment must be an exact boolean."
            raise RevisionInputError(msg)
        manifest = _validate_materialized_input_bundle(
            str(request["root"]),
            str(request["expected_manifest_sha256"]),
            str(request["approval_manifest"]),
            str(request["expected_approval_sha256"]),
            _validate_cohorts(request["cohorts"]),
            require_current_execution_environment=require_current,
        )
        _write_json_atomic(response_path, {"manifest": manifest})
        return
    msg = f"Unknown isolated execution action: {action!r}"
    raise RevisionInputError(msg)


def main() -> None:
    """Materialize canonical inputs from explicit, pinned local authorities."""
    if "--internal-execution-snapshot-sha256" in sys.argv:
        internal = argparse.ArgumentParser(add_help=False)
        internal.add_argument("--internal-execution-snapshot-sha256", required=True)
        internal.add_argument("--internal-request", type=Path, required=True)
        internal.add_argument("--internal-request-sha256", required=True)
        internal.add_argument("--internal-response", type=Path, required=True)
        args = internal.parse_args()
        _run_internal_execution_request(
            args.internal_execution_snapshot_sha256,
            args.internal_request,
            args.internal_request_sha256,
            args.internal_response,
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-maf-root", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--datahub-git-dir", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+")
    args = parser.parse_args()
    result = materialize_tcga_revision_inputs(
        args.raw_maf_root,
        args.population_root,
        args.datahub_git_dir,
        args.approval_manifest,
        args.expected_approval_sha256,
        args.out,
        cohorts=args.cohorts,
    )
    print(result)


if __name__ == "__main__":
    main()

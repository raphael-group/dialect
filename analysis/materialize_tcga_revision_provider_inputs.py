"""Safely rebuild the signed TCGA revision's provider inputs.

This orchestrator is intentionally result-blind.  It accepts only the independently
pinned D1/D2 input authority, validates the complete canonical 32-cohort bundle, and
runs the historical cohort pipeline in ``PREPARE_ONLY`` mode.  Association fitting is
never invoked and association-result paths are forbidden from the isolated work tree.

The work tree is resumable because CBaSE, DIG, and MutSig stages are accepted only
after their input-bound receipts reproduce.  A complete tree receives one exclusive,
canonical root manifest and is then atomically renamed to the requested output path.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import mmap
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analysis.materialize_tcga_revision_inputs import (
    materialized_cohort_binding,
    validate_materialized_input_bundle,
)
from dialect.data.revision_approval import (
    MATERIALIZE_FINAL_INPUTS_STAGE,
    STAGE_SCOPED_APPROVAL_SCHEMA,
    validate_revision_approval,
)
from dialect.data.tcga import TCGA_COHORTS


class _Digest(Protocol):
    """Minimal streaming digest interface used by framed tree hashing."""

    def update(self, data: bytes) -> None:
        """Consume one exact byte chunk."""


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
PROVIDER_INPUT_CONTRACT: Final = "signed-tcga-provider-input-rebuild-v1"
ROOT_MANIFEST_NAME: Final = "provider_input_manifest.json"
WORK_AUTHORITY_PATH: Final = PurePosixPath("_orchestration/authority.json")
CANARY_COHORT: Final = "CHOL"
MEMORY_HEAVY_COHORTS: Final = frozenset({"BRCA", "CRAD", "LGG", "SKCM", "UCEC"})
TOP_K: Final = 500
NICE_INCREMENT: Final = 10
MAX_JOBS: Final = 3
NICE_EXECUTABLE: Final = Path("/usr/bin/nice")
BASH_EXECUTABLE: Final = Path("/bin/bash")
CHILD_PYTHON_EXECUTABLE: Final = Path("/opt/anaconda3/envs/dialect/bin/python")
DIALECT_EXECUTABLE: Final = Path("/opt/anaconda3/envs/dialect/bin/dialect")
SAFE_CHILD_PATH: Final = (
    "/opt/anaconda3/envs/dialect/bin:/opt/homebrew/bin:/usr/local/bin:"
    "/usr/bin:/bin:/usr/sbin:/sbin"
)
MUTSIG_CHILD_PATH: Final = f"/opt/homebrew/bin:{SAFE_CHILD_PATH}"
MUTSIG_JAVA_HOME: Final = (
    "/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home"
)
MUTSIG_OCTAVE_ISOLATION_ARGS: Final = (
    "--no-init-all",
    "--no-history",
    "--no-gui",
)
THREAD_ENVIRONMENT: Final = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
MUTSIG_UPSTREAM_COMMIT: Final = "0109e27e70478181695f31ca8dd281bb44f0b3af"
MUTSIG_RECEIPT_FIELDS: Final = (
    "schema_version",
    "cohort",
    "upstream_commit",
    "source_tree_sha256",
    "source_file_count",
    "patch_sha256",
    "runner_sha256",
    "runtime_sha256",
    "maf_sha256",
    "sample_axis_sha256",
    "sample_axis_count",
    "lambda_sha256",
    "meta_sha256",
    "genes_sha256",
    "patients_sha256",
)
STAGE_RECEIPT_FIELDS: Final = (
    "schema_version",
    "input_sha256",
    "output_sha256",
)
COHORT_ROOT_FILES: Final = frozenset(
    {
        "bmr_pmfs.csv",
        "bmr_pmfs.dig.csv",
        "cbase_input.tsv",
        "cbase_stage_receipt.tsv",
        "count_matrix.csv",
        "dig_stage_receipt.tsv",
        "gene_level_count_matrix.csv",
        "pipeline.log",
        "sample_axis.txt",
    },
)
CBASE_OUTPUT_FILES: Final = frozenset(
    {
        "kept_mutations.csv",
        "mutation_mat.txt",
        "output_data_preparation.txt",
        "param_estimates_1.txt",
        "param_estimates_2.txt",
        "param_estimates_3.txt",
        "param_estimates_4.txt",
        "param_estimates_5.txt",
        "param_estimates_6.txt",
        "pofkgivens.txt",
        "pofkigivens.txt",
        "pofmgivens.txt",
        "pofmigivens.txt",
        "q_values.txt",
        "used_params_and_model.txt",
    },
)
MUTSIG_OUTPUT_FILES: Final = frozenset(
    {
        "persample_genes.txt",
        "persample_lambda.f32",
        "persample_meta.txt",
        "persample_patients.txt",
        "persample_receipt.tsv",
    },
)
FORBIDDEN_ASSOCIATION_FILES: Final = frozenset(
    {
        "identify_stage_receipt.tsv",
        "pairwise_interaction_results.csv",
        "single_gene_results.csv",
    },
)
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_FEATURE_PATTERN: Final = re.compile(r".+_[MN]")
_NONNEGATIVE_INTEGER_PATTERN: Final = re.compile(r"(?:0|[1-9][0-9]*)")
_ATTEMPT_FILE_PATTERN: Final = re.compile(r"[0-9a-f]{32}\.(?:json|log)")
_READBACK_FILE_PATTERN: Final = re.compile(r"[0-9a-f]{32}\.json")
_MUTSIG_STAGING_PATTERN: Final = re.compile(
    r"\.([A-Z0-9]+)\.mutsig\.[A-Za-z0-9]{6,}$",
)
_PIPELINE_TEMP_PATTERN: Final = re.compile(
    r"(?:sample_axis\.txt|(?:cbase|dig)_stage_receipt\.tsv)\.tmp\.[0-9]+",
)
_HASH_CHUNK_BYTES: Final = 1024 * 1024
TREE_HASH_CONTRACT: Final = "u64be-path-mode-content-v1"
_GIB: Final = 1024**3
_PRIOR_TASK_PEAK_RSS_BYTES: Final = round(2.083 * _GIB)
_MEMORY_HEADROOM_FACTOR: Final = 1.25
_MIN_AVAILABLE_MEMORY_FRACTION: Final = 0.33
_MIN_FREE_DISK_BYTES: Final = round(7.6 * _GIB)
_HOST_LEASE_SCHEMA: Final = "dialect-k500-same-uid-machine-lease-v1"
SAME_UID_MACHINE_LEASE_DIRECTORY: Final = Path("/tmp")
EXECUTION_SNAPSHOT_CONTRACT: Final = "provider-private-content-snapshot-v1"
EXECUTION_SNAPSHOT_RECEIPT: Final = PurePosixPath(
    "_orchestration/execution_snapshot.json",
)
RUNTIME_AUTHORITY_PATH: Final = PurePosixPath("runtime/authority.json")
RUNTIME_AUTHORITY_CONTRACT: Final = "provider-child-runtime-authority-v1"
FULL_ACCEPTANCE_CONTRACT: Final = "provider-full-acceptance-receipt-v1"
_EXECUTION_SNAPSHOT_PREFIX: Final = "execution-snapshot-"
_EXECUTION_SNAPSHOT_BUILD_PATTERN: Final = re.compile(
    r"\.execution-snapshot-build-[0-9a-f]{32}",
)
_EXECUTION_SNAPSHOT_READY_PATTERN: Final = re.compile(
    rf"{_EXECUTION_SNAPSHOT_PREFIX}[0-9a-f]{{64}}",
)


class ProviderInputError(ValueError):
    """Raised when provider-input materialization fails closed."""


@dataclass(frozen=True, slots=True)
class ProviderPaths:
    """Resolved production paths used by the provider rebuild."""

    repo_root: Path
    canonical_input_root: Path
    approval_manifest: Path
    output_root: Path
    work_root: Path
    cohort_root: Path
    mutsig_root: Path
    cbase_inputs: Path
    dig_results: Path
    pipeline: Path
    mutsig_runner: Path
    mutsig_patch: Path


@dataclass(frozen=True, slots=True)
class IndependentHashes:
    """Hashes that must arrive independently through the public CLI."""

    approval: str
    canonical_input_manifest: str
    cbase_inputs_tree: str
    dig_results: str


@dataclass(frozen=True, slots=True)
class HostResourceSnapshot:
    """One aggregate-only live host resource readback."""

    measured_at_utc: str
    logical_cores: int
    load_average_1m: float
    total_memory_bytes: int
    available_memory_bytes: int
    free_disk_bytes: int
    cpu_source: str
    memory_source: str


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Fully validated authority and source state for one rebuild."""

    paths: ProviderPaths
    hashes: IndependentHashes
    canonical_manifest: Mapping[str, Any]
    bindings: Mapping[str, Mapping[str, Any]]
    authority: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CohortBmrSemantics:
    """Validated count and scalar-PMF values needed for the K=500 gate."""

    count_features: tuple[str, ...]
    counts_by_feature: Mapping[str, tuple[int, ...]]
    cbase_pmfs: Mapping[str, Mapping[int, float]]
    dig_pmfs: Mapping[str, Mapping[int, float]]


@dataclass(frozen=True, slots=True)
class StreamedTreeRecord:
    """Bounded-memory digest and inventory metadata for one secure tree read."""

    tree_sha256: str
    file_count: int


@dataclass(frozen=True, slots=True)
class SnapshotInventoryReadback:
    """One bounded-memory full execution-snapshot readback."""

    tree_sha256: str
    file_count: int
    files: Mapping[str, Mapping[str, Any]]
    component_trees: Mapping[str, StreamedTreeRecord]


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _absolute_unresolved(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        msg = (
            f"Parent-directory traversal is forbidden in authority/output paths: {path}"
        )
        raise ProviderInputError(msg)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.absolute()


def _require_no_symlink_ancestors(path: Path, *, label: str) -> None:
    absolute = _absolute_unresolved(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            break
        if stat.S_ISLNK(current.lstat().st_mode):
            msg = f"{label} traverses a symlink: {current}"
            raise ProviderInputError(msg)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _exact_json_equal(left: object, right: object) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError, UnicodeEncodeError):
        return False


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be an independently supplied lowercase SHA-256."
        raise ProviderInputError(msg)
    return value


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
        msg = f"Unable to open {label} without directory symlinks: {path}"
        raise ProviderInputError(msg) from error
    return descriptor


def _open_regular_fd(path: Path, *, label: str) -> int:
    """Open one single-link regular file through no-follow ancestors."""
    absolute = _absolute_unresolved(path)
    parent_descriptor = _open_directory_fd(absolute.parent, label=f"{label} parent")
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        msg = f"Unable to open {label} without symlinks: {path}"
        raise ProviderInputError(msg) from error
    finally:
        os.close(parent_descriptor)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        msg = f"{label} must be a single-link regular file: {path}"
        raise ProviderInputError(msg)
    return descriptor


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    content, _ = _read_regular_bytes_with_stat(path, label=label)
    return content


def _read_regular_bytes_with_stat(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    descriptor = _open_regular_fd(path, label=label)
    opened = os.fstat(descriptor)
    captured = bytearray()
    try:
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            captured.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require_stable_open_file(
        opened,
        after,
        label=label,
        bytes_read=len(captured),
    )
    return bytes(captured), opened


def _sha256(path: Path) -> str:
    return _stream_file_record(
        path,
        display_path=path.as_posix(),
        label="SHA-256 input",
    )["sha256"]


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    observed = _stream_file_record(
        path,
        display_path=path.as_posix() if display_path is None else display_path,
        label="file receipt",
    )
    return {key: observed[key] for key in ("path", "bytes", "sha256")}


def _file_record_from_bytes(
    path: Path,
    content: bytes,
    *,
    display_path: str | None = None,
) -> dict[str, Any]:
    return {
        "path": path.as_posix() if display_path is None else display_path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"JSON object contains duplicate key {key!r}: provider authority."
            raise ProviderInputError(msg)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    msg = f"Non-finite JSON constant is forbidden: {value}."
    raise ProviderInputError(msg)


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = f"Non-finite JSON number is forbidden: {value}."
        raise ProviderInputError(msg)
    return parsed


def _reject_json_surrogates(value: object, *, label: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            msg = f"{label} contains an invalid Unicode surrogate."
            raise ProviderInputError(msg)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_json_surrogates(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_json_surrogates(key, label=f"{label} object key")
            _reject_json_surrogates(item, label=f"{label}.{key}")


def _parse_json_bytes(raw: bytes, *, path: Path) -> dict[str, Any]:
    """Parse canonical JSON from bytes already captured by a secure open."""
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except ProviderInputError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        msg = f"Invalid JSON object: {path}"
        raise ProviderInputError(msg) from error
    _reject_json_surrogates(payload, label=f"JSON object {path}")
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object: {path}"
        raise ProviderInputError(msg)
    if raw != _canonical_json(payload) + b"\n":
        msg = f"JSON is not canonical with one terminal LF: {path}"
        raise ProviderInputError(msg)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return _parse_json_bytes(
        _read_regular_bytes(path, label="JSON object"),
        path=path,
    )


def _read_json_with_sha256(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    payload, raw, _opened = _read_json_with_sha256_and_stat(
        path,
        expected_sha256,
        label=label,
    )
    return payload, raw


def _read_json_with_sha256_and_stat(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes, os.stat_result]:
    raw, opened = _read_regular_bytes_with_stat(path, label=label)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        msg = f"{label} differs from its expected SHA-256: {path}"
        raise ProviderInputError(msg)
    return _parse_json_bytes(raw, path=path), raw, opened


def _read_json_with_stat(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], os.stat_result]:
    raw, opened = _read_regular_bytes_with_stat(path, label=label)
    return _parse_json_bytes(raw, path=path), opened


def _write_json_atomic(
    path: Path,
    payload: object,
    *,
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = _canonical_json(payload)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        msg = f"Refusing to serialize a noncanonical JSON payload: {path}"
        raise ProviderInputError(msg) from error
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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
    """Atomically rename descriptor-relative names without replacement."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        rename = getattr(library, "renameatx_np", None)
        if rename is None:
            msg = "renameatx_np is unavailable; exclusive publication cannot proceed."
            raise ProviderInputError(msg)
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
            raise ProviderInputError(msg)
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
            "Unsupported platform for atomic exclusive provider publication: "
            f"{sys.platform!r}."
        )
        raise ProviderInputError(msg)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Exclusive provider publication target already exists",
            destination_name,
        )
    raise OSError(
        error_number,
        "Atomic exclusive provider publication failed",
        destination_name,
    )


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Compatibility wrapper using stable no-follow parent descriptors."""
    source_parent = _open_directory_fd(source.parent, label="rename source parent")
    try:
        destination_parent = _open_directory_fd(
            destination.parent,
            label="rename destination parent",
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


def _directory_path_matches_fd(path: Path, descriptor: int) -> bool:
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


def _sha256_text_lines(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _files_sha256(paths: Sequence[Path]) -> str:
    records = [
        _stream_file_record(
            path,
            display_path=path.name,
            label="provider output receipt artifact",
        )
        for path in paths
    ]
    if any(record["bytes"] <= 0 for record in records):
        msg = "Provider output receipt includes an empty artifact."
        raise ProviderInputError(msg)
    return _sha256_text_lines(
        [f"{record['path']}\t{record['sha256']}" for record in records],
    )


def _files_sha256_from_bytes(
    artifacts: Sequence[tuple[Path, bytes]],
) -> str:
    if any(not content for _, content in artifacts):
        msg = "Provider output receipt includes an empty artifact."
        raise ProviderInputError(msg)
    return _sha256_text_lines(
        [
            f"{path.name}\t{hashlib.sha256(content).hexdigest()}"
            for path, content in artifacts
        ],
    )


def _iter_secure_tree_files(
    root: Path,
    *,
    python_only: bool = False,
    required_directory_mode: int | None = None,
) -> Iterator[tuple[PurePosixPath, int, os.stat_result]]:
    """Yield securely opened single-link files while retaining ancestor dirfds."""
    root = _absolute_unresolved(root)
    root_descriptor = _open_directory_fd(root, label="source/input tree")

    def require_directory_mode(descriptor: int, relative: PurePosixPath) -> None:
        if required_directory_mode is None:
            return
        observed = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if observed != required_directory_mode:
            msg = (
                "Execution snapshot directory is mutable or mode-drifted: "
                f"{relative.as_posix() or '.'}"
            )
            raise ProviderInputError(msg)

    def walk(
        descriptor: int,
        prefix: PurePosixPath,
    ) -> Iterator[tuple[PurePosixPath, int, os.stat_result]]:
        require_directory_mode(descriptor, prefix)
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."} or "/" in name or "\x00" in name:
                msg = f"Source/input tree contains an invalid entry name: {name!r}"
                raise ProviderInputError(msg)
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = prefix / name
            if stat.S_ISLNK(entry.st_mode):
                msg = f"Source/input tree contains a symlink: {relative}"
                raise ProviderInputError(msg)
            if stat.S_ISDIR(entry.st_mode):
                if name == "__pycache__":
                    continue
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    yield from walk(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(entry.st_mode):
                msg = f"Source/input tree contains a special file: {relative}"
                raise ProviderInputError(msg)
            file_descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(file_descriptor)
            try:
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    msg = (
                        "Source/input tree contains a non-private regular file: "
                        f"{relative}"
                    )
                    raise ProviderInputError(msg)
                if not python_only or relative.suffix == ".py":
                    yield relative, file_descriptor, opened
            finally:
                os.close(file_descriptor)

    try:
        yield from walk(root_descriptor, PurePosixPath())
    finally:
        os.close(root_descriptor)


def _copy_fd_chunks(
    source_descriptor: int,
    destination_descriptor: int | None,
    hashers: Sequence[Any],
) -> int:
    total = 0
    while chunk := os.read(source_descriptor, _HASH_CHUNK_BYTES):
        total += len(chunk)
        for digest in hashers:
            digest.update(chunk)
        if destination_descriptor is not None:
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    msg = "Execution snapshot copy made no forward progress."
                    raise ProviderInputError(msg)
                view = view[written:]
    return total


def _normalized_snapshot_mode(mode: int) -> int:
    return 0o500 if stat.S_IMODE(mode) & 0o111 else 0o400


def _update_framed_file_prefix(
    digest: _Digest,
    relative: PurePosixPath,
    *,
    mode: int,
    size: int,
) -> None:
    encoded_path = relative.as_posix().encode()
    if size < 0:
        msg = f"Negative file size in tree-hash input: {relative}"
        raise ProviderInputError(msg)
    digest.update(struct.pack(">Q", len(encoded_path)))
    digest.update(encoded_path)
    digest.update(struct.pack(">Q", _normalized_snapshot_mode(mode)))
    digest.update(struct.pack(">Q", size))


def _require_stable_open_file(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
    bytes_read: int,
) -> None:
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or bytes_read != before.st_size
    ):
        msg = f"{label} changed while its exact open descriptor was consumed."
        raise ProviderInputError(msg)


def _stream_tree_record(
    root: Path,
    *,
    python_only: bool = False,
    required_directory_mode: int | None = None,
    required_file_modes: frozenset[int] | None = None,
) -> StreamedTreeRecord:
    digest = hashlib.sha256()
    count = 0
    for relative, descriptor, opened in _iter_secure_tree_files(
        root,
        python_only=python_only,
        required_directory_mode=required_directory_mode,
    ):
        mode = stat.S_IMODE(opened.st_mode)
        if required_file_modes is not None and mode not in required_file_modes:
            msg = f"Execution snapshot file is mutable or mode-drifted: {relative}"
            raise ProviderInputError(msg)
        _update_framed_file_prefix(
            digest,
            relative,
            mode=mode,
            size=opened.st_size,
        )
        size = _copy_fd_chunks(descriptor, None, (digest,))
        _require_stable_open_file(
            opened,
            os.fstat(descriptor),
            label=f"source/input tree file {relative}",
            bytes_read=size,
        )
        count += 1
    if count == 0:
        msg = f"Source/input directory contains no hashable files: {root}"
        raise ProviderInputError(msg)
    return StreamedTreeRecord(digest.hexdigest(), count)


def _stream_file_record(
    path: Path,
    *,
    display_path: str,
    required_mode: int | None = None,
    label: str,
) -> dict[str, Any]:
    descriptor = _open_regular_fd(path, label=label)
    try:
        opened = os.fstat(descriptor)
        mode = stat.S_IMODE(opened.st_mode)
        if required_mode is not None and mode != required_mode:
            msg = f"{label} has mutable or drifted mode: {path}"
            raise ProviderInputError(msg)
        digest = hashlib.sha256()
        size = _copy_fd_chunks(descriptor, None, (digest,))
        _require_stable_open_file(
            opened,
            os.fstat(descriptor),
            label=label,
            bytes_read=size,
        )
    finally:
        os.close(descriptor)
    return {
        "path": display_path,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "mode": mode,
    }


def _tree_sha256(root: Path, *, python_only: bool = False) -> str:
    return _stream_tree_record(root, python_only=python_only).tree_sha256


def _base_child_environment(*, path: str = SAFE_CHILD_PATH) -> dict[str, str]:
    return {
        **THREAD_ENVIRONMENT,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": path,
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TZ": "UTC",
    }


def _stream_authority_file_record(path: Path, *, label: str) -> dict[str, Any]:
    observed = _stream_file_record(
        path,
        display_path=path.as_posix(),
        label=label,
    )
    return {key: observed[key] for key in ("path", "bytes", "sha256")}


def _decode_record_sha256(value: str, *, label: str) -> str:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        msg = f"{label} does not use a SHA-256 RECORD digest."
        raise ProviderInputError(msg)
    try:
        padding = "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(encoded + padding).hex()
    except (ValueError, TypeError) as error:
        msg = f"{label} has an invalid RECORD digest."
        raise ProviderInputError(msg) from error


def _distribution_runtime_record(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    candidates = [
        file
        for file in (distribution.files or ())
        if PurePosixPath(file).name == "RECORD"
        and ".dist-info" in PurePosixPath(file).parent.name
    ]
    if len(candidates) != 1:
        msg = f"{name} distribution does not expose exactly one RECORD file."
        raise ProviderInputError(msg)
    record_relative = PurePosixPath(candidates[0].as_posix())
    base = Path(distribution.locate_file("")).absolute()
    environment_root = CHILD_PYTHON_EXECUTABLE.parent.parent
    record_path = Path(distribution.locate_file(candidates[0])).absolute()
    record_bytes = _read_regular_bytes(record_path, label=f"{name} RECORD")
    try:
        rows = list(csv.reader(io.StringIO(record_bytes.decode(), newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        msg = f"{name} distribution RECORD is not valid UTF-8 CSV."
        raise ProviderInputError(msg) from error
    files: list[dict[str, Any]] = []
    native_files: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if len(row) != 3 or not row[0]:
            msg = f"{name} RECORD row {index} is malformed."
            raise ProviderInputError(msg)
        raw_relative, recorded_hash, recorded_size = row
        if PurePosixPath(raw_relative).is_absolute() or "\x00" in raw_relative:
            msg = f"{name} RECORD row {index} has an unsafe path."
            raise ProviderInputError(msg)
        located = Path(os.path.normpath(base / raw_relative)).absolute()
        if not located.is_relative_to(environment_root):
            msg = f"{name} RECORD row {index} escapes the frozen environment."
            raise ProviderInputError(msg)
        actual = _stream_authority_file_record(
            located,
            label=f"{name} distribution file {raw_relative}",
        )
        if recorded_hash:
            expected_hash = _decode_record_sha256(
                recorded_hash,
                label=f"{name} RECORD row {index}",
            )
            if actual["sha256"] != expected_hash:
                msg = f"{name} installed file differs from RECORD: {raw_relative}"
                raise ProviderInputError(msg)
        else:
            expected_hash = None
        if recorded_size:
            if not recorded_size.isdigit() or actual["bytes"] != int(recorded_size):
                msg = f"{name} installed file size differs from RECORD: {raw_relative}"
                raise ProviderInputError(msg)
            expected_size: int | None = int(recorded_size)
        else:
            expected_size = None
        item = {
            "record_path": raw_relative,
            "record_sha256": expected_hash,
            "record_bytes": expected_size,
            "installed": actual,
        }
        files.append(item)
        lowered = located.name.casefold()
        if any(token in lowered for token in (".so", ".dylib", ".pyd", ".dll")):
            native_files.append(item)
    files.sort(key=lambda item: item["record_path"])
    native_files.sort(key=lambda item: item["record_path"])
    files_digest = hashlib.sha256(_canonical_json(files)).hexdigest()
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "record": _file_record_from_bytes(record_path, record_bytes),
        "record_relative_path": record_relative.as_posix(),
        "file_count": len(files),
        "files_sha256": files_digest,
        "files": files,
        "native_files": native_files,
    }


def _child_python_runtime_record(paths: ProviderPaths) -> dict[str, Any]:
    python_path = CHILD_PYTHON_EXECUTABLE.resolve()
    dialect_entrypoint = DIALECT_EXECUTABLE
    python_bytes = _read_regular_bytes(python_path, label="frozen child Python")
    entrypoint_bytes = _read_regular_bytes(
        dialect_entrypoint,
        label="frozen DIALECT entrypoint",
    )
    first_lines = entrypoint_bytes.splitlines()[:1]
    try:
        shebang = first_lines[0].decode()
    except (IndexError, UnicodeDecodeError) as error:
        msg = "Frozen DIALECT entrypoint has no valid interpreter shebang."
        raise ProviderInputError(msg) from error
    if (
        not shebang.startswith("#!")
        or not Path(shebang[2:]).is_absolute()
        or Path(shebang[2:]).resolve() != python_path
    ):
        msg = "Frozen DIALECT entrypoint does not use the frozen child Python."
        raise ProviderInputError(msg)
    probe = (
        "import json,platform,sys; import dialect,numpy,pandas,scipy; "
        "print(json.dumps({"
        "'dialect_file':dialect.__file__,"
        "'numpy_file':numpy.__file__,"
        "'numpy':numpy.__version__,"
        "'pandas_file':pandas.__file__,"
        "'pandas':pandas.__version__,"
        "'python':platform.python_version(),"
        "'scipy_file':scipy.__file__,"
        "'scipy':scipy.__version__,"
        "'sys_executable':sys.executable"
        "},allow_nan=False,sort_keys=True,separators=(',',':')))"
    )
    completed = subprocess.run(
        [CHILD_PYTHON_EXECUTABLE.as_posix(), "-P", "-s", "-c", probe],
        check=False,
        cwd=paths.repo_root,
        env=_base_child_environment(),
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        msg = "Frozen child Python runtime probe returned invalid JSON."
        raise ProviderInputError(msg) from error
    expected_keys = {
        "dialect_file",
        "numpy",
        "numpy_file",
        "pandas",
        "pandas_file",
        "python",
        "scipy",
        "scipy_file",
        "sys_executable",
    }
    if (
        completed.returncode != 0
        or completed.stderr
        or not isinstance(payload, dict)
        or set(payload) != expected_keys
        or any(not isinstance(payload[key], str) or not payload[key] for key in payload)
    ):
        msg = "Frozen child Python runtime probe is incomplete or failed."
        raise ProviderInputError(msg)
    observed_python = Path(payload["sys_executable"]).resolve()
    dialect_import = Path(payload["dialect_file"]).resolve()
    dialect_source_root = (paths.repo_root / "src" / "dialect").resolve()
    if observed_python != python_path:
        msg = "Frozen child Python probe resolved a different executable."
        raise ProviderInputError(msg)
    if (
        not dialect_import.is_relative_to(dialect_source_root)
        or dialect_import.is_symlink()
        or not dialect_import.is_file()
    ):
        msg = "Frozen child Python does not import DIALECT from this repository."
        raise ProviderInputError(msg)
    if (
        _read_regular_bytes(python_path, label="frozen child Python readback")
        != python_bytes
        or _read_regular_bytes(
            dialect_entrypoint,
            label="frozen DIALECT entrypoint readback",
        )
        != entrypoint_bytes
    ):
        msg = "Frozen child runtime changed during its identity probe."
        raise ProviderInputError(msg)
    dialect_import_bytes = _read_regular_bytes(
        dialect_import,
        label="DIALECT repository import",
    )
    imported_modules = {
        name: _stream_authority_file_record(
            Path(payload[f"{name}_file"]).resolve(),
            label=f"imported {name} module",
        )
        for name in ("numpy", "pandas", "scipy")
    }
    distributions = {
        name: _distribution_runtime_record(name)
        for name in ("numpy", "pandas", "scipy")
    }
    dialect_tree_sha256 = _tree_sha256(
        paths.repo_root / "src" / "dialect",
        python_only=True,
    )
    runtime = {
        "launcher": CHILD_PYTHON_EXECUTABLE.as_posix(),
        "entrypoint_shebang": shebang,
        "python_executable": _file_record_from_bytes(python_path, python_bytes),
        "dialect_entrypoint": _file_record_from_bytes(
            dialect_entrypoint,
            entrypoint_bytes,
        ),
        "dialect_import": _file_record_from_bytes(
            dialect_import,
            dialect_import_bytes,
        ),
        "dialect_tree_hash_contract": TREE_HASH_CONTRACT,
        "dialect_tree_sha256": dialect_tree_sha256,
        "imported_modules": imported_modules,
        "distributions": distributions,
        "versions": {
            "python": payload["python"],
            "numpy": payload["numpy"],
            "pandas": payload["pandas"],
            "scipy": payload["scipy"],
        },
    }
    runtime["runtime_sha256"] = hashlib.sha256(
        _canonical_json(runtime),
    ).hexdigest()
    return runtime


def _mutsig_runtime_record() -> dict[str, Any]:
    octave_name = shutil.which("octave", path=MUTSIG_CHILD_PATH)
    if octave_name is None:
        msg = "GNU Octave is unavailable on the sealed provider PATH."
        raise ProviderInputError(msg)
    octave_path = Path(octave_name).resolve()
    octave_bytes = _read_regular_bytes(octave_path, label="GNU Octave executable")
    java_path = Path(MUTSIG_JAVA_HOME) / "bin" / "java"
    java_bytes = _read_regular_bytes(java_path, label="frozen Java executable")
    completed = subprocess.run(
        [
            octave_path.as_posix(),
            *MUTSIG_OCTAVE_ISOLATION_ARGS,
            "--version",
        ],
        check=False,
        env={
            **_base_child_environment(path=MUTSIG_CHILD_PATH),
            "JAVA_HOME": MUTSIG_JAVA_HOME,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or not lines or not lines[0].strip():
        msg = "GNU Octave runtime identity could not be read on the sealed PATH."
        raise ProviderInputError(msg)
    java_completed = subprocess.run(
        [java_path.as_posix(), "-version"],
        check=False,
        env=_base_child_environment(path=MUTSIG_CHILD_PATH),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    java_lines = java_completed.stdout.splitlines()
    if java_completed.returncode != 0 or not java_lines or not java_lines[0].strip():
        msg = "Frozen Java runtime identity could not be read."
        raise ProviderInputError(msg)
    if (
        _read_regular_bytes(
            octave_path,
            label="GNU Octave executable readback",
        )
        != octave_bytes
        or _read_regular_bytes(
            java_path,
            label="frozen Java executable readback",
        )
        != java_bytes
    ):
        msg = "GNU Octave or Java changed during its identity probe."
        raise ProviderInputError(msg)
    octave_id = lines[0]
    runtime = {
        "octave": _file_record_from_bytes(octave_path, octave_bytes),
        "octave_id": octave_id,
        "java_home": MUTSIG_JAVA_HOME,
        "java_executable": _file_record_from_bytes(java_path, java_bytes),
        "java_id": java_lines[0],
    }
    runtime["runtime_sha256"] = hashlib.sha256(_canonical_json(runtime)).hexdigest()
    return runtime


def _require_hardened_mutsig_runner(path: Path, content: bytes | None = None) -> None:
    if content is None:
        try:
            runner_stat = path.lstat()
        except OSError as error:
            msg = "Pinned MutSig runner must be a regular non-symlink file."
            raise ProviderInputError(msg) from error
        if not stat.S_ISREG(runner_stat.st_mode) or runner_stat.st_nlink != 1:
            msg = "Pinned MutSig runner must be a single-link regular non-symlink file."
            raise ProviderInputError(msg)
    try:
        source = (
            _read_regular_bytes(path, label="pinned MutSig runner")
            if content is None
            else content
        ).decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "Pinned MutSig runner source could not be read as UTF-8."
        raise ProviderInputError(msg) from error
    flags = " ".join(MUTSIG_OCTAVE_ISOLATION_ARGS)
    required_invocations = (
        f'"$OCTAVE_BIN" {flags} --version',
        f'"$OCTAVE_BIN" {flags} --eval',
    )
    if any(source.count(invocation) != 1 for invocation in required_invocations):
        msg = (
            "Pinned MutSig runner must suppress all Octave startup files and "
            "history on both version and eval invocations."
        )
        raise ProviderInputError(msg)


def safe_job_cap(logical_cores: int | None = None) -> int:
    """Return the largest integer at most three and strictly below half cores."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 0)
    if not isinstance(cores, int) or isinstance(cores, bool) or cores <= 0:
        return 0
    return min(MAX_JOBS, (cores - 1) // 2)


def _validate_jobs(jobs: int, *, logical_cores: int | None = None) -> int:
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 0)
    cap = safe_job_cap(cores)
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs <= 0 or jobs > cap:
        msg = (
            f"--jobs must be between 1 and {cap} for {cores} logical cores; "
            "provider concurrency must remain strictly below half the host."
        )
        raise ProviderInputError(msg)
    return jobs


def _parse_linux_meminfo(content: str) -> tuple[int, int]:
    fields: dict[str, int] = {}
    for line in content.splitlines():
        name, separator, raw = line.partition(":")
        pieces = raw.split()
        if separator and len(pieces) == 2 and pieces[0].isdigit() and pieces[1] == "kB":
            fields[name] = int(pieces[0]) * 1024
    try:
        total = fields["MemTotal"]
        available = fields["MemAvailable"]
    except KeyError as error:
        msg = "Linux aggregate memory readback lacks MemTotal or MemAvailable."
        raise ProviderInputError(msg) from error
    if total <= 0 or not 0 <= available <= total:
        msg = "Linux aggregate memory readback is invalid."
        raise ProviderInputError(msg)
    return total, available


def _parse_darwin_memory_pressure(output: str) -> tuple[int, int]:
    total_match = re.search(r"The system has (\d+) ", output)
    free_match = re.search(r"System-wide memory free percentage: (\d+)%", output)
    if total_match is None or free_match is None:
        msg = "macOS aggregate memory readback is not parseable."
        raise ProviderInputError(msg)
    total = int(total_match.group(1))
    free_percent = int(free_match.group(1))
    if total <= 0 or not 0 <= free_percent <= 100:
        msg = "macOS aggregate memory readback is invalid."
        raise ProviderInputError(msg)
    return total, total * free_percent // 100


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            msg = f"No existing parent is available for disk readback: {path}"
            raise ProviderInputError(msg)
        candidate = candidate.parent
    return candidate


def read_host_resources(output_root: Path) -> HostResourceSnapshot:
    """Read aggregate CPU, memory, and target-filesystem state."""
    try:
        load_average = float(os.getloadavg()[0])
    except (AttributeError, OSError) as error:
        msg = "One-minute aggregate CPU load is unavailable."
        raise ProviderInputError(msg) from error
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        )
        total_memory, available_memory = _parse_darwin_memory_pressure(
            completed.stdout,
        )
        memory_source = "/usr/bin/memory_pressure -Q"
    elif sys.platform.startswith("linux"):
        total_memory, available_memory = _parse_linux_meminfo(
            Path("/proc/meminfo").read_text(encoding="utf-8"),
        )
        memory_source = "/proc/meminfo MemAvailable"
    else:
        msg = f"Unsupported platform for aggregate resource gating: {sys.platform!r}."
        raise ProviderInputError(msg)
    disk_parent = _nearest_existing_parent(output_root)
    return HostResourceSnapshot(
        measured_at_utc=_utc_now(),
        logical_cores=os.cpu_count() or 0,
        load_average_1m=load_average,
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        free_disk_bytes=shutil.disk_usage(disk_parent).free,
        cpu_source="os.getloadavg()[0]",
        memory_source=memory_source,
    )


def evaluate_host_resource_gate(
    snapshot: HostResourceSnapshot,
    *,
    jobs: int,
) -> dict[str, Any]:
    """Evaluate one live aggregate resource snapshot without process inspection."""
    reasons: list[str] = []
    safe_cap = safe_job_cap(snapshot.logical_cores)
    if jobs <= 0 or jobs > safe_cap:
        reasons.append(f"jobs={jobs} exceeds safe live cap={safe_cap}")
    half_cores = snapshot.logical_cores / 2
    projected_load = snapshot.load_average_1m + jobs
    if (
        not math.isfinite(snapshot.load_average_1m)
        or snapshot.load_average_1m < 0
        or projected_load >= half_cores
    ):
        reasons.append(
            "one-minute aggregate CPU load plus planned jobs is not below half "
            "the host",
        )
    if snapshot.total_memory_bytes <= 0 or not (
        0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes
    ):
        reasons.append("available memory is outside the physical-memory range")
    required_by_tasks = math.ceil(
        jobs * _PRIOR_TASK_PEAK_RSS_BYTES * _MEMORY_HEADROOM_FACTOR,
    )
    required_by_fraction = math.ceil(
        max(snapshot.total_memory_bytes, 0) * _MIN_AVAILABLE_MEMORY_FRACTION,
    )
    required_available = max(required_by_tasks, required_by_fraction)
    if snapshot.available_memory_bytes < required_available:
        reasons.append("available memory is below the aggregate headroom gate")
    if snapshot.free_disk_bytes < _MIN_FREE_DISK_BYTES:
        reasons.append("free disk is below the 2x historical-output gate")
    try:
        measured_at = datetime.fromisoformat(snapshot.measured_at_utc)
        timestamp_is_utc = (
            measured_at.tzinfo is not None
            and measured_at.utcoffset() == UTC.utcoffset(None)
        )
    except (TypeError, ValueError):
        timestamp_is_utc = False
    if not timestamp_is_utc or not snapshot.cpu_source or not snapshot.memory_source:
        reasons.append("resource readback provenance is incomplete")
    return {
        "passed": not reasons,
        "jobs": jobs,
        "safe_job_cap": safe_cap,
        "strict_half_core_limit": half_cores,
        "projected_load_with_planned_jobs": projected_load,
        "required_available_memory_bytes": required_available,
        "required_by_prior_rss_bytes": required_by_tasks,
        "required_by_fraction_bytes": required_by_fraction,
        "minimum_free_disk_bytes": _MIN_FREE_DISK_BYTES,
        "reasons": reasons,
    }


def _require_live_resource_gate(
    context: ProviderContext,
    *,
    jobs: int,
    label: str,
) -> None:
    snapshot = read_host_resources(context.paths.work_root)
    evaluation = evaluate_host_resource_gate(snapshot, jobs=jobs)
    record = {
        "schema_version": SCHEMA_VERSION,
        "contract": PROVIDER_INPUT_CONTRACT,
        "label": label,
        "snapshot": asdict(snapshot),
        "evaluation": evaluation,
    }
    destination = (
        context.paths.work_root / "resource_readbacks" / f"{uuid.uuid4().hex}.json"
    )
    _write_json_atomic(destination, record)
    if not evaluation["passed"]:
        msg = f"Live aggregate resource gate failed: {evaluation['reasons']}"
        raise ProviderInputError(msg)


@contextmanager
def _host_execution_lease(output_root: Path) -> Iterator[Path]:
    """Acquire the runner's same-UID machine-wide nonblocking lease."""
    lease_path = SAME_UID_MACHINE_LEASE_DIRECTORY / (f"dialect-k500-{os.getuid()}.lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lease_path, flags, 0o600)
    except OSError as error:
        msg = f"Unable to safely open the same-UID machine-wide lease: {lease_path}"
        raise ProviderInputError(msg) from error
    try:
        _require_secure_lease_file(descriptor, lease_path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            msg = (
                "Another same-UID DIALECT K=500 process holds the machine-wide "
                "resource lease."
            )
            raise ProviderInputError(msg) from error
        _require_secure_lease_file(descriptor, lease_path)
        record = {
            "schema": _HOST_LEASE_SCHEMA,
            "pid": os.getpid(),
            "output_root": output_root.as_posix(),
            "acquired_at_utc": _utc_now(),
        }
        os.ftruncate(descriptor, 0)
        os.write(descriptor, _canonical_json(record) + b"\n")
        os.fsync(descriptor)
        yield lease_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _require_secure_lease_file(descriptor: int, lease_path: Path) -> None:
    """Bind the advisory lock to one private, stable same-UID regular file."""
    descriptor_stat = os.fstat(descriptor)
    try:
        path_stat = os.lstat(lease_path)
    except OSError as error:
        msg = f"Same-UID machine-wide lease path disappeared: {lease_path}"
        raise ProviderInputError(msg) from error
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_uid != os.getuid()
        or descriptor_stat.st_nlink != 1
        or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
        or descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        msg = f"Same-UID machine-wide lease is not a stable private file: {lease_path}"
        raise ProviderInputError(msg)


def _casefold_path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in _absolute_unresolved(path).parts)


def _path_parts_contain(container: tuple[str, ...], child: tuple[str, ...]) -> bool:
    return len(container) <= len(child) and child[: len(container)] == container


def _existing_identity(path: Path) -> tuple[int, int] | None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(observed.st_mode):
        msg = f"Consumed/output path alias is a forbidden symlink: {path}"
        raise ProviderInputError(msg)
    return observed.st_dev, observed.st_ino


def _ancestor_identities(path: Path) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    candidate = _absolute_unresolved(path)
    while True:
        identity = _existing_identity(candidate)
        if identity is not None:
            identities.add(identity)
        if candidate.parent == candidate:
            return identities
        candidate = candidate.parent


def _reject_output_source_overlap(
    output: Path,
    work: Path,
    protected: Sequence[Path],
) -> None:
    for target in (output, work):
        target_parts = _casefold_path_parts(target)
        target_identity = _existing_identity(target)
        target_ancestors = _ancestor_identities(target)
        for consumed in protected:
            consumed_parts = _casefold_path_parts(consumed)
            consumed_identity = _existing_identity(consumed)
            consumed_ancestors = _ancestor_identities(consumed)
            lexical_overlap = _path_parts_contain(
                target_parts,
                consumed_parts,
            ) or _path_parts_contain(consumed_parts, target_parts)
            inode_overlap = (
                consumed_identity is not None and consumed_identity in target_ancestors
            ) or (target_identity is not None and target_identity in consumed_ancestors)
            if lexical_overlap or inode_overlap:
                msg = (
                    "Provider output/work root overlaps a consumed "
                    f"authority/source path or alias: {consumed}"
                )
                raise ProviderInputError(msg)


def _provider_paths(
    canonical_input_root: str | Path,
    approval_manifest: str | Path,
    output_root: str | Path,
    *,
    repo_root: str | Path | None,
    require_current_execution_environment: bool = True,
) -> ProviderPaths:
    repository = _absolute_unresolved(
        Path(__file__).resolve().parents[1] if repo_root is None else repo_root,
    )
    canonical = _absolute_unresolved(canonical_input_root)
    approval = _absolute_unresolved(approval_manifest)
    output = _absolute_unresolved(output_root)
    work = output.parent / f".{output.name}.provider-work"
    if output.name in {"", ".", ".."} or output.parent == output:
        msg = "Provider output root must be a narrow, named directory."
        raise ProviderInputError(msg)
    if not output.parent.is_dir() or output.parent.is_symlink():
        msg = (
            "Provider output parent must be an existing non-symlink directory: "
            f"{output.parent}"
        )
        raise ProviderInputError(msg)
    output_parent_descriptor = _open_directory_fd(
        output.parent,
        label="provider output parent",
    )
    os.close(output_parent_descriptor)
    _require_no_symlink_ancestors(repository, label="repository root")
    _require_no_symlink_ancestors(output.parent, label="provider output parent")
    protected: tuple[Path, ...] = (
        canonical,
        approval,
        repository / "src" / "dialect",
        repository / "external" / "CBaSE",
        repository / "external" / "DIGDriver" / "run" / "Pancan.genes.results.txt",
        repository / "external" / "MutSig2CV_src",
        repository / "scripts" / "run_cohort_pipeline.sh",
        repository / "scripts" / "run_mutsig_octave.sh",
        repository / "external" / "mutsig2cv_octave_dialect.patch",
        Path(__file__).resolve(),
    )
    if require_current_execution_environment:
        git_name = shutil.which("git", path=SAFE_CHILD_PATH)
        octave_name = shutil.which("octave", path=MUTSIG_CHILD_PATH)
        if git_name is None or octave_name is None:
            msg = "Git and GNU Octave must exist on the sealed provider PATH."
            raise ProviderInputError(msg)
        protected += (
            CHILD_PYTHON_EXECUTABLE.parent.parent,
            DIALECT_EXECUTABLE,
            NICE_EXECUTABLE,
            BASH_EXECUTABLE,
            Path(git_name).resolve(),
            Path(octave_name).resolve(),
            Path(MUTSIG_JAVA_HOME),
        )
    _reject_output_source_overlap(output, work, protected)
    return ProviderPaths(
        repo_root=repository,
        canonical_input_root=canonical,
        approval_manifest=approval,
        output_root=output,
        work_root=work,
        cohort_root=work / "cohorts",
        mutsig_root=work / "mutsig",
        cbase_inputs=repository / "external" / "CBaSE",
        dig_results=(
            repository / "external" / "DIGDriver" / "run" / "Pancan.genes.results.txt"
        ),
        pipeline=repository / "scripts" / "run_cohort_pipeline.sh",
        mutsig_runner=repository / "scripts" / "run_mutsig_octave.sh",
        mutsig_patch=repository / "external" / "mutsig2cv_octave_dialect.patch",
    )


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    cohort: str,
) -> dict[str, Any]:
    if binding.get("cohort") != cohort or set(binding) != {
        "cohort",
        "child_manifest",
        "canonical_maf",
        "sample_axis",
        "population_manifest",
    }:
        msg = f"Public canonical-input binding is malformed: {cohort}"
        raise ProviderInputError(msg)
    result: dict[str, Any] = {"cohort": cohort}
    for name in (
        "child_manifest",
        "canonical_maf",
        "sample_axis",
        "population_manifest",
    ):
        value = binding[name]
        if not isinstance(value, dict) or set(value) != {"path", "file"}:
            msg = f"Public canonical-input binding has invalid {name}: {cohort}"
            raise ProviderInputError(msg)
        path = value["path"]
        record = value["file"]
        if not isinstance(path, Path) or not isinstance(record, dict):
            msg = f"Public canonical-input binding has invalid {name}: {cohort}"
            raise ProviderInputError(msg)
        if set(record) != {"path", "bytes", "sha256"}:
            msg = f"Public canonical-input file receipt is malformed: {cohort}/{name}"
            raise ProviderInputError(msg)
        expected_digest = _require_sha256(
            record["sha256"],
            label=f"{cohort} canonical {name} SHA-256",
        )
        observed = _stream_file_record(
            path,
            display_path=str(record["path"]),
            label=f"canonical {cohort}/{name}",
        )
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] <= 0
            or observed["bytes"] != record["bytes"]
            or observed["sha256"] != expected_digest
        ):
            msg = f"Public canonical-input receipt changed: {cohort}/{name}"
            raise ProviderInputError(msg)
        result[name] = {"path": path, "file": dict(record)}
    return result


def _execution_contract() -> dict[str, Any]:
    return {
        "cohorts": list(TCGA_COHORTS),
        "cohort_count": len(TCGA_COHORTS),
        "top_k": TOP_K,
        "nice_increment": NICE_INCREMENT,
        "maximum_jobs": MAX_JOBS,
        "strictly_below_half_logical_cores": True,
        "serial_canary": CANARY_COHORT,
        "serial_memory_heavy_cohorts": sorted(MEMORY_HEAVY_COHORTS),
        "thread_environment": THREAD_ENVIRONMENT,
        "child_path": SAFE_CHILD_PATH,
        "child_locale": {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
        "child_python_isolation": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        },
        "private_content_addressed_execution_snapshot": (EXECUTION_SNAPSHOT_CONTRACT),
        "child_pythonpath_is_private_snapshot": True,
        "full_snapshot_validated_once_at_build_or_resume_boundary": True,
        "child_scoped_snapshot_rehashed_before_and_after_every_child": True,
        "finite_external_runtime_closure_revalidated_before_and_after_every_child": (
            True
        ),
        "mutsig_octave_isolation_args": list(MUTSIG_OCTAVE_ISOLATION_ARGS),
        "live_aggregate_resource_gate_before_every_wave": True,
    }


def _scope_contract() -> dict[str, Any]:
    return {
        "result_blind": True,
        "association_outputs_opened": False,
        "association_identify_invoked": False,
        "pipeline_mode": "PREPARE_ONLY=1",
        "providers_invoked": ["cbase", "dig", "mutsig"],
    }


def _require_exact_dict(
    value: object,
    expected_keys: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        msg = f"{label} has a partial or extra authority schema."
        raise ProviderInputError(msg)
    return value


def _validate_authority_file_record(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> str:
    record = _require_exact_dict(
        value,
        {"path", "bytes", "sha256"},
        label=label,
    )
    raw_path = record["path"]
    if not isinstance(raw_path, str):
        msg = f"{label} path is not an absolute canonical path."
        raise ProviderInputError(msg)
    record_path = Path(raw_path)
    if (
        not record_path.is_absolute()
        or ".." in record_path.parts
        or record_path.as_posix() != raw_path
        or (expected_path is not None and raw_path != expected_path.as_posix())
    ):
        msg = f"{label} path is not the exact expected authority path."
        raise ProviderInputError(msg)
    byte_count = record["bytes"]
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
    ):
        msg = f"{label} byte count is invalid."
        raise ProviderInputError(msg)
    return _require_sha256(record["sha256"], label=f"{label} SHA-256")


def _validate_distribution_runtime_authority(value: object, *, label: str) -> None:
    distribution = _require_exact_dict(
        value,
        {
            "name",
            "version",
            "record",
            "record_relative_path",
            "file_count",
            "files_sha256",
            "files",
            "native_files",
        },
        label=label,
    )
    if any(
        not isinstance(distribution[key], str)
        or not distribution[key]
        or distribution[key] != distribution[key].strip()
        for key in ("name", "version", "record_relative_path")
    ):
        msg = f"{label} identity is invalid."
        raise ProviderInputError(msg)
    _validate_authority_file_record(
        distribution["record"],
        label=f"{label} RECORD file",
    )
    files = distribution["files"]
    native_files = distribution["native_files"]
    if (
        not isinstance(files, list)
        or not isinstance(native_files, list)
        or not isinstance(distribution["file_count"], int)
        or isinstance(distribution["file_count"], bool)
        or distribution["file_count"] != len(files)
        or distribution["file_count"] <= 0
        or any(not isinstance(item, dict) for item in files)
    ):
        msg = f"{label} file inventory is invalid."
        raise ProviderInputError(msg)
    record_paths = [item.get("record_path") for item in files]
    if any(
        not isinstance(path, str) for path in record_paths
    ) or record_paths != sorted(record_paths):
        msg = f"{label} file inventory is not canonically ordered."
        raise ProviderInputError(msg)
    for item in files:
        parsed = _require_exact_dict(
            item,
            {"record_path", "record_sha256", "record_bytes", "installed"},
            label=f"{label} file",
        )
        if not isinstance(parsed["record_path"], str) or not parsed["record_path"]:
            msg = f"{label} contains an invalid RECORD path."
            raise ProviderInputError(msg)
        if parsed["record_sha256"] is not None:
            _require_sha256(
                parsed["record_sha256"],
                label=f"{label} RECORD file SHA-256",
            )
        if parsed["record_bytes"] is not None and (
            not isinstance(parsed["record_bytes"], int)
            or isinstance(parsed["record_bytes"], bool)
            or parsed["record_bytes"] < 0
        ):
            msg = f"{label} contains an invalid RECORD byte count."
            raise ProviderInputError(msg)
        _validate_authority_file_record(
            parsed["installed"],
            label=f"{label} installed file",
        )
    expected_native = [
        item
        for item in files
        if any(
            token in Path(item["installed"]["path"]).name.casefold()
            for token in (".so", ".dylib", ".pyd", ".dll")
        )
    ]
    if not _exact_json_equal(native_files, expected_native):
        msg = f"{label} native/shared file inventory is incomplete."
        raise ProviderInputError(msg)
    expected_digest = hashlib.sha256(_canonical_json(files)).hexdigest()
    if distribution["files_sha256"] != expected_digest:
        msg = f"{label} installed-file inventory digest does not reproduce."
        raise ProviderInputError(msg)


def _validate_dialect_child_authority(
    value: object,
    _paths: ProviderPaths,
    *,
    require_current_paths: bool,
) -> str:
    child = _require_exact_dict(
        value,
        {
            "launcher",
            "entrypoint_shebang",
            "python_executable",
            "dialect_entrypoint",
            "dialect_import",
            "dialect_tree_hash_contract",
            "dialect_tree_sha256",
            "imported_modules",
            "distributions",
            "versions",
            "runtime_sha256",
        },
        label="DIALECT child runtime",
    )
    python_record = _require_exact_dict(
        child["python_executable"],
        {"path", "bytes", "sha256"},
        label="DIALECT child Python executable",
    )
    _validate_authority_file_record(
        python_record,
        label="DIALECT child Python executable",
    )
    python_path = python_record["path"]
    if (
        not isinstance(child["launcher"], str)
        or not Path(child["launcher"]).is_absolute()
        or child["entrypoint_shebang"] != f"#!{python_path}"
        or (
            require_current_paths
            and child["launcher"] != CHILD_PYTHON_EXECUTABLE.as_posix()
        )
    ):
        msg = "DIALECT child launcher and interpreter shebang disagree."
        raise ProviderInputError(msg)
    _validate_authority_file_record(
        child["dialect_entrypoint"],
        label="DIALECT child entrypoint",
        expected_path=DIALECT_EXECUTABLE if require_current_paths else None,
    )
    _validate_authority_file_record(
        child["dialect_import"],
        label="DIALECT repository import",
    )
    if child["dialect_tree_hash_contract"] != TREE_HASH_CONTRACT:
        msg = "DIALECT child tree-hash contract is invalid."
        raise ProviderInputError(msg)
    _require_sha256(
        child["dialect_tree_sha256"],
        label="DIALECT child source tree SHA-256",
    )
    imported_modules = _require_exact_dict(
        child["imported_modules"],
        {"numpy", "pandas", "scipy"},
        label="DIALECT imported module authority",
    )
    distributions = _require_exact_dict(
        child["distributions"],
        {"numpy", "pandas", "scipy"},
        label="DIALECT distribution authority",
    )
    for name in ("numpy", "pandas", "scipy"):
        _validate_authority_file_record(
            imported_modules[name],
            label=f"DIALECT imported {name} module",
        )
        _validate_distribution_runtime_authority(
            distributions[name],
            label=f"DIALECT {name} distribution",
        )
    versions = _require_exact_dict(
        child["versions"],
        {"python", "numpy", "pandas", "scipy"},
        label="DIALECT child versions",
    )
    if any(
        not isinstance(versions[name], str)
        or not versions[name]
        or versions[name] != versions[name].strip()
        for name in versions
    ):
        msg = "DIALECT child version authority is invalid."
        raise ProviderInputError(msg)
    runtime_preimage = dict(child)
    runtime_preimage.pop("runtime_sha256")
    expected_runtime_hash = hashlib.sha256(
        _canonical_json(runtime_preimage),
    ).hexdigest()
    if child["runtime_sha256"] != expected_runtime_hash:
        msg = "DIALECT child runtime hash does not reproduce."
        raise ProviderInputError(msg)
    return expected_runtime_hash


def _validate_mutsig_runtime_authority(
    value: object,
    *,
    require_current_paths: bool,
) -> str:
    runtime = _require_exact_dict(
        value,
        {
            "octave",
            "octave_id",
            "java_home",
            "java_executable",
            "java_id",
            "runtime_sha256",
        },
        label="MutSig runtime",
    )
    _validate_authority_file_record(
        runtime["octave"],
        label="MutSig Octave executable",
    )
    _validate_authority_file_record(
        runtime["java_executable"],
        label="MutSig Java executable",
    )
    octave_id = runtime["octave_id"]
    java_id = runtime["java_id"]
    if (
        not isinstance(octave_id, str)
        or not octave_id
        or octave_id != octave_id.strip()
        or "\n" in octave_id
        or "\r" in octave_id
        or not isinstance(java_id, str)
        or not java_id
        or java_id != java_id.strip()
        or "\n" in java_id
        or "\r" in java_id
        or not isinstance(runtime["java_home"], str)
        or not Path(runtime["java_home"]).is_absolute()
        or (require_current_paths and runtime["java_home"] != MUTSIG_JAVA_HOME)
    ):
        msg = "MutSig runtime identity is invalid."
        raise ProviderInputError(msg)
    runtime_preimage = dict(runtime)
    runtime_preimage.pop("runtime_sha256")
    expected_runtime_hash = hashlib.sha256(
        _canonical_json(runtime_preimage),
    ).hexdigest()
    if runtime["runtime_sha256"] != expected_runtime_hash:
        msg = "MutSig runtime hash does not reproduce."
        raise ProviderInputError(msg)
    return expected_runtime_hash


def _validate_source_authority(
    value: object,
    paths: ProviderPaths,
    *,
    require_current_paths: bool,
) -> dict[str, Any]:
    file_names = {
        "orchestrator",
        "cohort_pipeline",
        "mutsig_runner",
        "mutsig_patch",
        "nice_executable",
        "bash_executable",
        "git_executable",
    }
    file_paths: dict[str, Path] = {}
    if require_current_paths:
        git_name = shutil.which("git", path=SAFE_CHILD_PATH)
        if git_name is None:
            msg = "Git is unavailable on the sealed provider PATH."
            raise ProviderInputError(msg)
        file_paths = {
            "orchestrator": Path(__file__).resolve(),
            "cohort_pipeline": paths.pipeline,
            "mutsig_runner": paths.mutsig_runner,
            "mutsig_patch": paths.mutsig_patch,
            "nice_executable": NICE_EXECUTABLE,
            "bash_executable": BASH_EXECUTABLE,
            "git_executable": Path(git_name).resolve(),
        }
    sources = _require_exact_dict(
        value,
        {*file_names, "dialect_python_tree_sha256", "python_runtime_sha256"},
        label="provider source authority",
    )
    for name in sorted(file_names):
        _validate_authority_file_record(
            sources[name],
            label=f"provider source {name}",
            expected_path=file_paths.get(name),
        )
    _require_sha256(
        sources["dialect_python_tree_sha256"],
        label="DIALECT Python source tree SHA-256",
    )
    _require_sha256(
        sources["python_runtime_sha256"],
        label="DIALECT child runtime SHA-256",
    )
    return sources


def _validate_provider_authority(
    value: object,
    paths: ProviderPaths,
    sources: Mapping[str, Any],
    *,
    require_current_paths: bool,
) -> tuple[str, str]:
    providers = _require_exact_dict(
        value,
        {"cbase", "dig", "mutsig", "dialect_child"},
        label="provider authority",
    )
    cbase = _require_exact_dict(
        providers["cbase"],
        {
            "inputs_root",
            "inputs_tree_sha256",
            "expected_inputs_tree_sha256",
        },
        label="CBaSE provider authority",
    )
    cbase_hash = _require_sha256(
        cbase["expected_inputs_tree_sha256"],
        label="provider authority CBaSE SHA-256",
    )
    if (
        not isinstance(cbase["inputs_root"], str)
        or not Path(cbase["inputs_root"]).is_absolute()
        or cbase["inputs_tree_sha256"] != cbase_hash
        or (
            require_current_paths
            and cbase["inputs_root"] != paths.cbase_inputs.as_posix()
        )
    ):
        msg = "CBaSE provider authority is internally inconsistent."
        raise ProviderInputError(msg)
    dig = _require_exact_dict(
        providers["dig"],
        {"results", "expected_results_sha256"},
        label="DIG provider authority",
    )
    dig_hash = _require_sha256(
        dig["expected_results_sha256"],
        label="provider authority DIG SHA-256",
    )
    observed_dig_hash = _validate_authority_file_record(
        dig["results"],
        label="DIG provider results",
        expected_path=paths.dig_results if require_current_paths else None,
    )
    if observed_dig_hash != dig_hash:
        msg = "DIG provider authority is internally inconsistent."
        raise ProviderInputError(msg)
    mutsig = _require_exact_dict(
        providers["mutsig"],
        {
            "upstream_commit",
            "source_tree_hash_contract",
            "source_tree_sha256",
            "source_file_count",
            "patch_sha256",
            "runner_sha256",
            "runtime",
        },
        label="MutSig provider authority",
    )
    mutsig_runtime_hash = _validate_mutsig_runtime_authority(
        mutsig["runtime"],
        require_current_paths=require_current_paths,
    )
    if (
        mutsig["upstream_commit"] != MUTSIG_UPSTREAM_COMMIT
        or mutsig["source_tree_hash_contract"] != TREE_HASH_CONTRACT
        or not isinstance(mutsig["source_file_count"], int)
        or isinstance(mutsig["source_file_count"], bool)
        or mutsig["source_file_count"] <= 0
        or mutsig["patch_sha256"] != sources["mutsig_patch"]["sha256"]
        or mutsig["runner_sha256"] != sources["mutsig_runner"]["sha256"]
        or mutsig["runtime"]["runtime_sha256"] != mutsig_runtime_hash
    ):
        msg = "MutSig provider authority does not bind its exact sources."
        raise ProviderInputError(msg)
    _require_sha256(
        mutsig["source_tree_sha256"],
        label="MutSig source tree SHA-256",
    )
    python_runtime_hash = _validate_dialect_child_authority(
        providers["dialect_child"],
        paths,
        require_current_paths=require_current_paths,
    )
    if sources["python_runtime_sha256"] != python_runtime_hash:
        msg = "DIALECT source/runtime authority hashes disagree."
        raise ProviderInputError(msg)
    return cbase_hash, dig_hash


def _validate_historical_authority_contract(
    work_authority: Mapping[str, Any],
    paths: ProviderPaths,
    *,
    require_current_paths: bool,
) -> tuple[str, str]:
    if not _exact_json_equal(work_authority["execution"], _execution_contract()):
        msg = "Provider execution authority is partial, extra, or drifted."
        raise ProviderInputError(msg)
    if not _exact_json_equal(work_authority["scope"], _scope_contract()):
        msg = "Provider result-blind scope authority is partial, extra, or drifted."
        raise ProviderInputError(msg)
    sources = _validate_source_authority(
        work_authority["sources"],
        paths,
        require_current_paths=require_current_paths,
    )
    return _validate_provider_authority(
        work_authority["providers"],
        paths,
        sources,
        require_current_paths=require_current_paths,
    )


def _runtime_authority_payload(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Return the finite external-runtime closure consumed by provider children."""
    sources = authority["sources"]
    providers = authority["providers"]
    dialect_runtime = providers["dialect_child"]
    mutsig_runtime = providers["mutsig"]["runtime"]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RUNTIME_AUTHORITY_CONTRACT,
        "tools": {
            "bash": sources["bash_executable"],
            "git": sources["git_executable"],
            "java": mutsig_runtime["java_executable"],
            "nice": sources["nice_executable"],
            "octave": mutsig_runtime["octave"],
            "python": dialect_runtime["python_executable"],
        },
        "python_runtime": dialect_runtime,
        "mutsig_runtime": mutsig_runtime,
    }


def _validate_runtime_authority_payload(
    value: object,
    context: ProviderContext,
) -> dict[str, Any]:
    payload = _require_exact_dict(
        value,
        {
            "schema_version",
            "contract",
            "tools",
            "python_runtime",
            "mutsig_runtime",
        },
        label="provider child runtime authority",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["contract"] != RUNTIME_AUTHORITY_CONTRACT
    ):
        msg = "Provider child runtime authority contract is invalid."
        raise ProviderInputError(msg)
    tools = _require_exact_dict(
        payload["tools"],
        {"bash", "git", "java", "nice", "octave", "python"},
        label="provider child tool authority",
    )
    for name in sorted(tools):
        _validate_authority_file_record(
            tools[name],
            label=f"provider child {name} executable",
        )
    _validate_dialect_child_authority(
        payload["python_runtime"],
        context.paths,
        require_current_paths=False,
    )
    _validate_mutsig_runtime_authority(
        payload["mutsig_runtime"],
        require_current_paths=False,
    )
    expected = _runtime_authority_payload(context.authority)
    if not _exact_json_equal(payload, expected):
        msg = "Provider child runtime authority differs from signed work authority."
        raise ProviderInputError(msg)
    return payload


def _stream_file_against_authority(
    record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    path = Path(str(record.get("path")))
    observed = _stream_file_record(
        path,
        display_path=path.as_posix(),
        label=label,
    )
    if observed["bytes"] != record.get("bytes") or observed["sha256"] != record.get(
        "sha256",
    ):
        msg = f"{label} changed after runtime authority capture."
        raise ProviderInputError(msg)


def _validate_current_runtime_authority(payload: Mapping[str, Any]) -> None:
    """Stream-rehash every external runtime byte admitted to a child process."""
    tools = payload["tools"]
    for name in sorted(tools):
        _stream_file_against_authority(
            tools[name],
            label=f"current provider {name} executable",
        )
    python_runtime = payload["python_runtime"]
    for name, record in sorted(python_runtime["imported_modules"].items()):
        _stream_file_against_authority(
            record,
            label=f"current imported {name} module",
        )
    for name, distribution in sorted(python_runtime["distributions"].items()):
        _stream_file_against_authority(
            distribution["record"],
            label=f"current {name} RECORD",
        )
        for item in distribution["files"]:
            _stream_file_against_authority(
                item["installed"],
                label=f"current {name} file {item['record_path']}",
            )


def _input_authority_record(
    paths: ProviderPaths,
    hashes: IndependentHashes,
    canonical_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    approval = canonical_manifest.get("authority", {}).get("approval")
    if not isinstance(approval, dict):
        msg = "Validated canonical manifest lacks its D1/D2 approval record."
        raise ProviderInputError(msg)
    decision_digests = approval.get("decision_digests")
    if not isinstance(decision_digests, dict) or set(decision_digests) != {"D1", "D2"}:
        msg = "Canonical input authority is not exactly the signed D1/D2 authority."
        raise ProviderInputError(msg)
    for decision_id in ("D1", "D2"):
        _require_sha256(
            decision_digests[decision_id],
            label=f"canonical approval {decision_id} decision digest",
        )
    if approval.get("manifest_sha256") != hashes.approval:
        msg = "Canonical input authority does not bind the independent approval hash."
        raise ProviderInputError(msg)
    if approval.get("authorized_stage") != "materialize-final-inputs":
        msg = "Provider inputs accept only materialize-final-inputs D1/D2 authority."
        raise ProviderInputError(msg)
    canonical_contract = canonical_manifest.get("contract")
    if not isinstance(canonical_contract, str) or not canonical_contract:
        msg = "Validated canonical manifest lacks its exact input contract."
        raise ProviderInputError(msg)
    canonical_manifest_path = paths.canonical_input_root / "input_manifest.json"
    return {
        "authorized_stage": "materialize-final-inputs",
        "authorized_decisions": ["D1", "D2"],
        "approval_manifest": _file_record(paths.approval_manifest),
        "expected_approval_sha256": hashes.approval,
        "decision_digests": dict(decision_digests),
        "canonical_input_root": paths.canonical_input_root.as_posix(),
        "canonical_input_manifest": _file_record(canonical_manifest_path),
        "expected_canonical_input_manifest_sha256": (hashes.canonical_input_manifest),
        "canonical_input_contract": canonical_contract,
    }


def _relocatable_input_authority_view(value: object) -> dict[str, Any]:
    authority = _require_exact_dict(
        value,
        {
            "authorized_stage",
            "authorized_decisions",
            "approval_manifest",
            "expected_approval_sha256",
            "decision_digests",
            "canonical_input_root",
            "canonical_input_manifest",
            "expected_canonical_input_manifest_sha256",
            "canonical_input_contract",
        },
        label="provider canonical input authority",
    )
    approval_hash = _validate_authority_file_record(
        authority["approval_manifest"],
        label="provider approval manifest authority",
    )
    canonical_hash = _validate_authority_file_record(
        authority["canonical_input_manifest"],
        label="provider canonical manifest authority",
    )
    decision_digests = _require_exact_dict(
        authority["decision_digests"],
        {"D1", "D2"},
        label="provider D1/D2 authority",
    )
    for decision_id in ("D1", "D2"):
        _require_sha256(
            decision_digests[decision_id],
            label=f"provider {decision_id} decision digest",
        )
    root = authority["canonical_input_root"]
    if (
        authority["authorized_stage"] != MATERIALIZE_FINAL_INPUTS_STAGE
        or authority["authorized_decisions"] != ["D1", "D2"]
        or authority["expected_approval_sha256"] != approval_hash
        or authority["expected_canonical_input_manifest_sha256"] != canonical_hash
        or not isinstance(root, str)
        or not Path(root).is_absolute()
        or not isinstance(authority["canonical_input_contract"], str)
        or not authority["canonical_input_contract"]
    ):
        msg = "Provider canonical input authority is internally inconsistent."
        raise ProviderInputError(msg)
    normalized = json.loads(_canonical_json(authority))
    normalized["approval_manifest"]["path"] = "<relocatable-approval-manifest>"
    normalized["canonical_input_root"] = "<relocatable-canonical-root>"
    normalized["canonical_input_manifest"]["path"] = "<relocatable-canonical-manifest>"
    return normalized


def _authority_record(
    paths: ProviderPaths,
    hashes: IndependentHashes,
    canonical_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_manifest_path = paths.canonical_input_root / "input_manifest.json"
    if _sha256(canonical_manifest_path) != hashes.canonical_input_manifest:
        msg = "Canonical input manifest differs from its independent CLI SHA-256."
        raise ProviderInputError(msg)
    if _sha256(paths.approval_manifest) != hashes.approval:
        msg = "Approval manifest differs from its independent CLI SHA-256."
        raise ProviderInputError(msg)
    observed_cbase_hash = _tree_sha256(paths.cbase_inputs)
    dig_record = _stream_authority_file_record(
        paths.dig_results,
        label="DIG results",
    )
    observed_dig_hash = dig_record["sha256"]
    if observed_cbase_hash != hashes.cbase_inputs_tree:
        msg = "CBaSE input tree differs from its independent CLI SHA-256."
        raise ProviderInputError(msg)
    if observed_dig_hash != hashes.dig_results:
        msg = "DIG results differ from their independent CLI SHA-256."
        raise ProviderInputError(msg)
    input_authority = _input_authority_record(paths, hashes, canonical_manifest)
    git_name = shutil.which("git", path=SAFE_CHILD_PATH)
    if git_name is None:
        msg = "Git is unavailable on the sealed provider PATH."
        raise ProviderInputError(msg)
    source_files = {
        "orchestrator": Path(__file__).resolve(),
        "cohort_pipeline": paths.pipeline,
        "mutsig_runner": paths.mutsig_runner,
        "mutsig_patch": paths.mutsig_patch,
        "nice_executable": NICE_EXECUTABLE,
        "bash_executable": BASH_EXECUTABLE,
        "git_executable": Path(git_name).resolve(),
    }
    mutsig_runner_bytes = _read_regular_bytes(
        paths.mutsig_runner,
        label="provider source mutsig_runner",
    )
    _require_hardened_mutsig_runner(
        paths.mutsig_runner,
        mutsig_runner_bytes,
    )
    sources = {
        name: _stream_authority_file_record(
            path,
            label=f"provider source {name}",
        )
        for name, path in source_files.items()
    }
    child_python_runtime = _child_python_runtime_record(paths)
    mutsig_runtime = _mutsig_runtime_record()
    mutsig_source = _stream_tree_record(
        paths.repo_root / "external" / "MutSig2CV_src",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": PROVIDER_INPUT_CONTRACT,
        "intended_output_root": paths.output_root.as_posix(),
        "authority": input_authority,
        "sources": {
            **sources,
            "dialect_python_tree_sha256": _tree_sha256(
                paths.repo_root / "src" / "dialect",
                python_only=True,
            ),
            "python_runtime_sha256": child_python_runtime["runtime_sha256"],
        },
        "providers": {
            "cbase": {
                "inputs_root": paths.cbase_inputs.as_posix(),
                "inputs_tree_sha256": observed_cbase_hash,
                "expected_inputs_tree_sha256": hashes.cbase_inputs_tree,
            },
            "dig": {
                "results": dig_record,
                "expected_results_sha256": hashes.dig_results,
            },
            "mutsig": {
                "upstream_commit": MUTSIG_UPSTREAM_COMMIT,
                "source_tree_hash_contract": TREE_HASH_CONTRACT,
                "source_tree_sha256": mutsig_source.tree_sha256,
                "source_file_count": mutsig_source.file_count,
                "patch_sha256": sources["mutsig_patch"]["sha256"],
                "runner_sha256": sources["mutsig_runner"]["sha256"],
                "runtime": mutsig_runtime,
            },
            "dialect_child": child_python_runtime,
        },
        "execution": _execution_contract(),
        "scope": _scope_contract(),
    }


def _canonical_bundle_state(
    paths: ProviderPaths,
    hashes: IndependentHashes,
    *,
    require_current_execution_environment: bool,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Any]]]:
    approval = validate_revision_approval(
        paths.approval_manifest,
        hashes.approval,
        MATERIALIZE_FINAL_INPUTS_STAGE,
    )
    if (
        approval.schema != STAGE_SCOPED_APPROVAL_SCHEMA
        or approval.allowed_stages != (MATERIALIZE_FINAL_INPUTS_STAGE,)
        or set(approval.stage_bindings) != {MATERIALIZE_FINAL_INPUTS_STAGE}
        or tuple(approval.decisions) != ("D1", "D2")
        or tuple(approval.decision_digests) != ("D1", "D2")
    ):
        msg = (
            "Provider input materialization requires an exact stage-scoped v5 "
            "D1/D2 authority for only materialize-final-inputs."
        )
        raise ProviderInputError(msg)
    canonical_manifest = validate_materialized_input_bundle(
        paths.canonical_input_root,
        hashes.canonical_input_manifest,
        paths.approval_manifest,
        hashes.approval,
        require_current_execution_environment=require_current_execution_environment,
    )
    if tuple(canonical_manifest.get("cohorts", ())) != TCGA_COHORTS:
        msg = "Provider materialization requires the exact ordered 32-cohort bundle."
        raise ProviderInputError(msg)
    bindings = {
        cohort: _validate_binding(
            materialized_cohort_binding(
                paths.canonical_input_root,
                canonical_manifest,
                cohort,
            ),
            cohort=cohort,
        )
        for cohort in TCGA_COHORTS
    }
    return canonical_manifest, bindings


def _build_context(
    paths: ProviderPaths,
    hashes: IndependentHashes,
) -> ProviderContext:
    canonical_manifest, bindings = _canonical_bundle_state(
        paths,
        hashes,
        require_current_execution_environment=True,
    )
    authority = _authority_record(paths, hashes, canonical_manifest)
    return ProviderContext(
        paths=paths,
        hashes=hashes,
        canonical_manifest=canonical_manifest,
        bindings=bindings,
        authority=authority,
    )


def _tree_digest(
    captured: Mapping[str, bytes],
    modes: Mapping[str, int] | None = None,
) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(captured.items()):
        _update_framed_file_prefix(
            digest,
            PurePosixPath(relative),
            mode=0o400 if modes is None else modes[relative],
            size=len(content),
        )
        digest.update(content)
    return digest.hexdigest()


def _write_snapshot_bytes(
    root: Path,
    relative: PurePosixPath,
    content: bytes,
    *,
    mode: int = 0o400,
) -> None:
    descriptor = _open_snapshot_destination_fd(
        root,
        relative,
        mode=mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _snapshot_file_record(
    relative: str,
    content: bytes,
    *,
    mode: int = 0o400,
) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "mode": mode,
    }


def _open_snapshot_destination_fd(
    root: Path,
    relative: PurePosixPath,
    *,
    mode: int,
) -> int:
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        msg = f"Invalid private execution snapshot path: {relative}"
        raise ProviderInputError(msg)
    parent_descriptor = _open_directory_fd(
        root,
        label="private execution snapshot root",
    )
    try:
        for part in relative.parts[:-1]:
            with suppress(FileExistsError):
                os.mkdir(part, mode=0o700, dir_fd=parent_descriptor)
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        return os.open(
            relative.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        msg = f"Unable to create private execution snapshot file: {relative}"
        raise ProviderInputError(msg) from error
    finally:
        os.close(parent_descriptor)


def _copy_open_file_to_snapshot(
    source_descriptor: int,
    opened: os.stat_result,
    destination_root: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> dict[str, Any]:
    mode = _normalized_snapshot_mode(opened.st_mode)
    destination_descriptor = _open_snapshot_destination_fd(
        destination_root,
        relative,
        mode=mode,
    )
    digest = hashlib.sha256()
    try:
        size = _copy_fd_chunks(
            source_descriptor,
            destination_descriptor,
            (digest,),
        )
        os.fsync(destination_descriptor)
        destination_stat = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_nlink != 1
            or stat.S_IMODE(destination_stat.st_mode) != mode
            or destination_stat.st_size != size
        ):
            msg = f"Execution snapshot destination is not exact/private: {relative}"
            raise ProviderInputError(msg)
    finally:
        os.close(destination_descriptor)
    _require_stable_open_file(
        opened,
        os.fstat(source_descriptor),
        label=label,
        bytes_read=size,
    )
    return {
        "path": relative.as_posix(),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "mode": mode,
    }


def _copy_file_to_snapshot(
    source: Path,
    destination_root: Path,
    relative: PurePosixPath,
    *,
    label: str,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    descriptor = _open_regular_fd(source, label=label)
    try:
        record = _copy_open_file_to_snapshot(
            descriptor,
            os.fstat(descriptor),
            destination_root,
            relative,
            label=label,
        )
    finally:
        os.close(descriptor)
    if authority is not None and (
        record["bytes"] != authority.get("bytes")
        or record["sha256"] != authority.get("sha256")
    ):
        msg = f"{label} changed after authority capture."
        raise ProviderInputError(msg)
    return record


def _copy_tree_to_snapshot(
    source_root: Path,
    destination_root: Path,
    relative_root: PurePosixPath,
    *,
    python_only: bool = False,
) -> StreamedTreeRecord:
    tree_digest = hashlib.sha256()
    count = 0
    for relative, descriptor, opened in _iter_secure_tree_files(
        source_root,
        python_only=python_only,
    ):
        _update_framed_file_prefix(
            tree_digest,
            relative,
            mode=opened.st_mode,
            size=opened.st_size,
        )
        destination_relative = relative_root / relative
        mode = _normalized_snapshot_mode(opened.st_mode)
        destination_descriptor = _open_snapshot_destination_fd(
            destination_root,
            destination_relative,
            mode=mode,
        )
        try:
            size = _copy_fd_chunks(
                descriptor,
                destination_descriptor,
                (tree_digest,),
            )
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        _require_stable_open_file(
            opened,
            os.fstat(descriptor),
            label=f"snapshot source tree {relative}",
            bytes_read=size,
        )
        count += 1
    if count == 0:
        msg = f"Execution snapshot source tree is empty: {source_root}"
        raise ProviderInputError(msg)
    return StreamedTreeRecord(tree_digest.hexdigest(), count)


def _execution_snapshot_receipt_path(context: ProviderContext) -> Path:
    return context.paths.work_root.joinpath(*EXECUTION_SNAPSHOT_RECEIPT.parts)


def _snapshot_root_from_receipt(
    context: ProviderContext,
    receipt: Mapping[str, Any],
) -> Path:
    root_name = receipt.get("root")
    if (
        not isinstance(root_name, str)
        or _EXECUTION_SNAPSHOT_READY_PATTERN.fullmatch(root_name) is None
    ):
        msg = "Execution snapshot receipt has an invalid content-addressed root."
        raise ProviderInputError(msg)
    return context.paths.work_root / "_orchestration" / root_name


def _materialize_snapshot_components(
    context: ProviderContext,
    staging: Path,
) -> dict[str, Any]:
    """Stream every mutable child input into one private staging snapshot."""
    sources = context.authority["sources"]
    providers = context.authority["providers"]
    components: dict[str, Any] = {}

    tree_specs = (
        (
            "dialect_python",
            context.paths.repo_root / "src" / "dialect",
            PurePosixPath("src/dialect"),
            True,
            sources["dialect_python_tree_sha256"],
        ),
        (
            "cbase",
            context.paths.cbase_inputs,
            PurePosixPath("external/CBaSE"),
            False,
            providers["cbase"]["inputs_tree_sha256"],
        ),
    )
    for name, live_root, relative_root, python_only, expected_digest in tree_specs:
        observed = _copy_tree_to_snapshot(
            live_root,
            staging,
            relative_root,
            python_only=python_only,
        )
        if observed.tree_sha256 != expected_digest:
            msg = f"{name} changed before secure execution snapshot."
            raise ProviderInputError(msg)
        components[name] = {
            "root": relative_root.as_posix(),
            "tree_hash_contract": TREE_HASH_CONTRACT,
            "tree_sha256": observed.tree_sha256,
            "file_count": observed.file_count,
        }

    mutsig_root = context.paths.repo_root / "external" / "MutSig2CV_src"
    mutsig_record = _copy_tree_to_snapshot(
        mutsig_root,
        staging,
        PurePosixPath("external/MutSig2CV_src"),
    )
    mutsig_authority = providers["mutsig"]
    if (
        mutsig_record.tree_sha256 != mutsig_authority["source_tree_sha256"]
        or mutsig_record.file_count != mutsig_authority["source_file_count"]
    ):
        msg = "MutSig source changed before secure execution snapshot."
        raise ProviderInputError(msg)
    components["mutsig_source"] = {
        "root": "external/MutSig2CV_src",
        "tree_hash_contract": TREE_HASH_CONTRACT,
        "tree_sha256": mutsig_record.tree_sha256,
        "file_count": mutsig_record.file_count,
        "upstream_commit": MUTSIG_UPSTREAM_COMMIT,
    }

    individual = {
        "cohort_pipeline": (
            context.paths.pipeline,
            PurePosixPath("scripts/run_cohort_pipeline.sh"),
            sources["cohort_pipeline"],
        ),
        "mutsig_runner": (
            context.paths.mutsig_runner,
            PurePosixPath("scripts/run_mutsig_octave.sh"),
            sources["mutsig_runner"],
        ),
        "mutsig_patch": (
            context.paths.mutsig_patch,
            PurePosixPath("external/mutsig2cv_octave_dialect.patch"),
            sources["mutsig_patch"],
        ),
        "dig_results": (
            context.paths.dig_results,
            PurePosixPath("external/DIGDriver/run/Pancan.genes.results.txt"),
            providers["dig"]["results"],
        ),
    }
    for name, live_path, relative, authority_record in individual.items():
        components[name] = _copy_file_to_snapshot(
            live_path,
            staging,
            relative,
            label=name,
            authority=authority_record,
        )

    cohorts: dict[str, Any] = {}
    for cohort in TCGA_COHORTS:
        binding = context.bindings[cohort]
        maf_relative = PurePosixPath("data/mafs") / f"{cohort}.maf"
        axis_relative = PurePosixPath("data/axes") / f"{cohort}.txt"
        maf_authority = binding["canonical_maf"]["file"]
        axis_authority = binding["sample_axis"]["file"]
        maf_record = _copy_file_to_snapshot(
            binding["canonical_maf"]["path"],
            staging,
            maf_relative,
            label=f"{cohort} canonical MAF",
            authority=maf_authority,
        )
        axis_record = _copy_file_to_snapshot(
            binding["sample_axis"]["path"],
            staging,
            axis_relative,
            label=f"{cohort} canonical sample axis",
            authority=axis_authority,
        )
        cohorts[cohort] = {
            "canonical_maf": {
                **maf_record,
                "authority": maf_authority,
            },
            "sample_axis": {
                **axis_record,
                "authority": axis_authority,
            },
        }
    components["canonical_inputs"] = {
        "cohorts": cohorts,
        "cohort_count": len(cohorts),
    }
    runtime_bytes = (
        _canonical_json(_runtime_authority_payload(context.authority)) + b"\n"
    )
    _write_snapshot_bytes(
        staging,
        RUNTIME_AUTHORITY_PATH,
        runtime_bytes,
        mode=0o400,
    )
    components["runtime_authority"] = _snapshot_file_record(
        RUNTIME_AUTHORITY_PATH.as_posix(),
        runtime_bytes,
    )
    return components


def _set_snapshot_read_only(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            current_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
            path.chmod(
                0o500 if current_mode & 0o111 else 0o400,
                follow_symlinks=False,
            )
        for name in directory_names:
            (current_path / name).chmod(0o500, follow_symlinks=False)
    root.chmod(0o500, follow_symlinks=False)


def _stream_snapshot_inventory(root: Path) -> SnapshotInventoryReadback:
    root_digest = hashlib.sha256()
    component_hashers = {
        "dialect_python": ("src/dialect", hashlib.sha256(), 0),
        "cbase": ("external/CBaSE", hashlib.sha256(), 0),
        "mutsig_source": ("external/MutSig2CV_src", hashlib.sha256(), 0),
    }
    files: dict[str, Mapping[str, Any]] = {}
    for relative, descriptor, opened in _iter_secure_tree_files(
        root,
        required_directory_mode=0o500,
    ):
        mode = stat.S_IMODE(opened.st_mode)
        if mode not in {0o400, 0o500}:
            msg = f"Execution snapshot file is mutable or mode-drifted: {relative}"
            raise ProviderInputError(msg)
        _update_framed_file_prefix(
            root_digest,
            relative,
            mode=mode,
            size=opened.st_size,
        )
        file_digest = hashlib.sha256()
        active_hashers: list[Any] = [root_digest, file_digest]
        for name, (prefix, component_digest, count) in component_hashers.items():
            marker = f"{prefix}/"
            if relative.as_posix().startswith(marker):
                component_relative = PurePosixPath(
                    relative.as_posix().removeprefix(marker),
                )
                _update_framed_file_prefix(
                    component_digest,
                    component_relative,
                    mode=mode,
                    size=opened.st_size,
                )
                active_hashers.append(component_digest)
                component_hashers[name] = (prefix, component_digest, count + 1)
                break
        size = _copy_fd_chunks(descriptor, None, active_hashers)
        _require_stable_open_file(
            opened,
            os.fstat(descriptor),
            label=f"execution snapshot file {relative}",
            bytes_read=size,
        )
        files[relative.as_posix()] = {
            "path": relative.as_posix(),
            "bytes": size,
            "sha256": file_digest.hexdigest(),
            "mode": mode,
        }
    if not files:
        msg = "Execution snapshot contains no files."
        raise ProviderInputError(msg)
    component_records = {
        name: StreamedTreeRecord(digest.hexdigest(), count)
        for name, (_prefix, digest, count) in component_hashers.items()
    }
    return SnapshotInventoryReadback(
        tree_sha256=root_digest.hexdigest(),
        file_count=len(files),
        files=files,
        component_trees=component_records,
    )


def _require_snapshot_build_matches_components(
    inventory: SnapshotInventoryReadback,
    components: Mapping[str, Any],
) -> None:
    for name, observed in inventory.component_trees.items():
        expected = components[name]
        if (
            expected.get("tree_hash_contract") != TREE_HASH_CONTRACT
            or expected.get("tree_sha256") != observed.tree_sha256
            or expected.get("file_count") != observed.file_count
        ):
            msg = f"Execution snapshot {name} changed during streaming copy."
            raise ProviderInputError(msg)
    for name in (
        "cohort_pipeline",
        "mutsig_runner",
        "mutsig_patch",
        "dig_results",
        "runtime_authority",
    ):
        expected = components[name]
        if not _exact_json_equal(inventory.files.get(expected["path"]), expected):
            msg = f"Execution snapshot {name} changed during streaming copy."
            raise ProviderInputError(msg)
    canonical = components["canonical_inputs"]["cohorts"]
    for cohort in TCGA_COHORTS:
        for name in ("canonical_maf", "sample_axis"):
            expected = canonical[cohort][name]
            snapshot_record = {
                key: expected[key] for key in ("path", "bytes", "sha256", "mode")
            }
            if not _exact_json_equal(
                inventory.files.get(expected["path"]),
                snapshot_record,
            ):
                msg = f"Execution snapshot {cohort}/{name} changed during copy."
                raise ProviderInputError(msg)


def _build_execution_snapshot(context: ProviderContext) -> dict[str, Any]:
    orchestration = context.paths.work_root / "_orchestration"
    staging = orchestration / f".execution-snapshot-build-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    staging_descriptor = _open_directory_fd(
        staging,
        label="execution snapshot staging root",
    )
    published = False
    try:
        components = _materialize_snapshot_components(context, staging)
        if not _directory_path_matches_fd(staging, staging_descriptor):
            msg = "Execution snapshot staging root changed during streaming copy."
            raise ProviderInputError(msg)
        _set_snapshot_read_only(staging)
        inventory = _stream_snapshot_inventory(staging)
        _require_snapshot_build_matches_components(inventory, components)
        if not _directory_path_matches_fd(staging, staging_descriptor):
            msg = "Execution snapshot staging root changed before publication."
            raise ProviderInputError(msg)
        digest = inventory.tree_sha256
        root_name = f"{_EXECUTION_SNAPSHOT_PREFIX}{digest}"
        destination = orchestration / root_name
        if os.path.lexists(destination):
            msg = "Unreceipted execution snapshot destination already exists."
            raise ProviderInputError(msg)
        _rename_exclusive(staging, destination)
        if not _directory_path_matches_fd(destination, staging_descriptor):
            msg = "Execution snapshot publication renamed a substituted directory."
            raise ProviderInputError(msg)
        published = True
    finally:
        os.close(staging_descriptor)
        if not published and staging.exists():
            staging.chmod(0o700, follow_symlinks=False)
            for current, directory_names, file_names in os.walk(staging):
                current_path = Path(current)
                for name in file_names:
                    (current_path / name).chmod(0o600, follow_symlinks=False)
                for name in directory_names:
                    (current_path / name).chmod(0o700, follow_symlinks=False)
            shutil.rmtree(staging)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "contract": EXECUTION_SNAPSHOT_CONTRACT,
        "root": root_name,
        "tree_hash_contract": TREE_HASH_CONTRACT,
        "tree_sha256": digest,
        "file_count": inventory.file_count,
        "components": components,
        "runtime_boundary": {
            "authority_file": components["runtime_authority"],
            "tools": _runtime_authority_payload(context.authority)["tools"],
        },
        "association_outputs_opened": False,
    }
    _write_json_atomic(_execution_snapshot_receipt_path(context), receipt, mode=0o444)
    return receipt


def _verify_snapshot_file(
    root: Path,
    value: Mapping[str, Any],
    *,
    label: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    relative = value.get("path")
    candidate = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        candidate is None
        or candidate.is_absolute()
        or not candidate.parts
        or candidate.as_posix() != relative
        or ".." in candidate.parts
        or "\\" in relative
        or (expected_path is not None and relative != expected_path)
    ):
        msg = f"{label} snapshot path is invalid."
        raise ProviderInputError(msg)
    if set(value) - {"path", "bytes", "sha256", "mode", "authority"}:
        msg = f"{label} snapshot receipt has extra fields."
        raise ProviderInputError(msg)
    expected_mode = value.get("mode")
    if expected_mode not in {0o400, 0o500}:
        msg = f"{label} snapshot receipt has invalid mode."
        raise ProviderInputError(msg)
    observed = _stream_file_record(
        root / relative,
        display_path=relative,
        required_mode=expected_mode,
        label=label,
    )
    if (
        not isinstance(value.get("bytes"), int)
        or isinstance(value.get("bytes"), bool)
        or value.get("bytes") < 0
        or observed["bytes"] != value.get("bytes")
        or observed["sha256"]
        != _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    ):
        msg = f"{label} snapshot receipt changed."
        raise ProviderInputError(msg)
    return observed


def _read_snapshot_runtime_authority(
    root: Path,
    record: Mapping[str, Any],
    context: ProviderContext,
) -> dict[str, Any]:
    if (
        set(record) != {"path", "bytes", "sha256", "mode"}
        or record.get(
            "path",
        )
        != RUNTIME_AUTHORITY_PATH.as_posix()
    ):
        msg = "Execution snapshot runtime-authority receipt is invalid."
        raise ProviderInputError(msg)
    raw, opened = _read_regular_bytes_with_stat(
        root.joinpath(*RUNTIME_AUTHORITY_PATH.parts),
        label="execution snapshot runtime authority",
    )
    if (
        stat.S_IMODE(opened.st_mode) != 0o400
        or record.get("mode") != 0o400
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or len(raw) != record.get("bytes")
        or hashlib.sha256(raw).hexdigest()
        != _require_sha256(
            record.get("sha256"),
            label="execution snapshot runtime authority SHA-256",
        )
    ):
        msg = "Execution snapshot runtime authority changed."
        raise ProviderInputError(msg)
    return _validate_runtime_authority_payload(
        _parse_json_bytes(
            raw,
            path=root.joinpath(*RUNTIME_AUTHORITY_PATH.parts),
        ),
        context,
    )


def _validate_execution_snapshot(
    context: ProviderContext,
    *,
    cohort: str | None = None,
    full: bool = False,
    require_current_execution_environment: bool = True,
    validate_provider_generation_sources: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Validate a snapshot with bounded I/O at full or child-local scope."""
    if (
        not isinstance(full, bool)
        or not isinstance(require_current_execution_environment, bool)
        or not isinstance(validate_provider_generation_sources, bool)
    ):
        msg = "Execution snapshot validation flags must be exact booleans."
        raise ProviderInputError(msg)
    receipt = _read_json(_execution_snapshot_receipt_path(context))
    if (
        set(receipt)
        != {
            "schema_version",
            "contract",
            "root",
            "tree_hash_contract",
            "tree_sha256",
            "file_count",
            "components",
            "runtime_boundary",
            "association_outputs_opened",
        }
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("contract") != EXECUTION_SNAPSHOT_CONTRACT
        or receipt.get("tree_hash_contract") != TREE_HASH_CONTRACT
        or receipt.get("association_outputs_opened") is not False
    ):
        msg = "Execution snapshot receipt contract is invalid."
        raise ProviderInputError(msg)
    digest = _require_sha256(
        receipt.get("tree_sha256"),
        label="execution snapshot tree SHA-256",
    )
    if receipt.get("root") != f"{_EXECUTION_SNAPSHOT_PREFIX}{digest}":
        msg = "Execution snapshot root does not equal its content address."
        raise ProviderInputError(msg)
    if (
        not isinstance(receipt.get("file_count"), int)
        or isinstance(receipt.get("file_count"), bool)
        or receipt["file_count"] <= 0
    ):
        msg = "Execution snapshot file count is invalid."
        raise ProviderInputError(msg)
    root = _snapshot_root_from_receipt(context, receipt)
    components = receipt.get("components")
    expected_component_keys = {
        "dialect_python",
        "cbase",
        "mutsig_source",
        "cohort_pipeline",
        "mutsig_runner",
        "mutsig_patch",
        "dig_results",
        "canonical_inputs",
        "runtime_authority",
    }
    if not isinstance(components, dict) or set(components) != expected_component_keys:
        msg = "Execution snapshot components are invalid."
        raise ProviderInputError(msg)

    expected_tree_authority = {
        "dialect_python": (
            "src/dialect",
            context.authority["sources"]["dialect_python_tree_sha256"],
        ),
        "cbase": (
            "external/CBaSE",
            context.authority["providers"]["cbase"]["inputs_tree_sha256"],
        ),
    }
    for name, (expected_root, expected_digest) in expected_tree_authority.items():
        value = components.get(name)
        if (
            not isinstance(value, dict)
            or set(value) != {"root", "tree_hash_contract", "tree_sha256", "file_count"}
            or value.get("root") != expected_root
            or value.get("tree_hash_contract") != TREE_HASH_CONTRACT
            or value.get("tree_sha256") != expected_digest
            or not isinstance(value.get("file_count"), int)
            or isinstance(value.get("file_count"), bool)
            or value["file_count"] <= 0
        ):
            msg = f"Execution snapshot {name} authority is invalid."
            raise ProviderInputError(msg)
    mutsig = components.get("mutsig_source")
    if (
        not isinstance(mutsig, dict)
        or set(mutsig)
        != {
            "root",
            "tree_hash_contract",
            "tree_sha256",
            "file_count",
            "upstream_commit",
        }
        or mutsig.get("root") != "external/MutSig2CV_src"
        or mutsig.get("tree_hash_contract") != TREE_HASH_CONTRACT
        or mutsig.get("upstream_commit") != MUTSIG_UPSTREAM_COMMIT
        or mutsig.get("tree_sha256")
        != context.authority["providers"]["mutsig"]["source_tree_sha256"]
        or mutsig.get("file_count")
        != context.authority["providers"]["mutsig"]["source_file_count"]
        or not isinstance(mutsig.get("file_count"), int)
        or isinstance(mutsig.get("file_count"), bool)
        or mutsig["file_count"] <= 0
    ):
        msg = "Execution snapshot MutSig source authority is invalid."
        raise ProviderInputError(msg)
    _require_sha256(
        mutsig.get("tree_sha256"),
        label="execution snapshot MutSig tree SHA-256",
    )

    individual_specs = {
        "cohort_pipeline": (
            "scripts/run_cohort_pipeline.sh",
            context.authority["sources"]["cohort_pipeline"],
        ),
        "mutsig_runner": (
            "scripts/run_mutsig_octave.sh",
            context.authority["sources"]["mutsig_runner"],
        ),
        "mutsig_patch": (
            "external/mutsig2cv_octave_dialect.patch",
            context.authority["sources"]["mutsig_patch"],
        ),
        "dig_results": (
            "external/DIGDriver/run/Pancan.genes.results.txt",
            context.authority["providers"]["dig"]["results"],
        ),
    }
    for name, (expected_path, authority_record) in individual_specs.items():
        value = components.get(name)
        if (
            not isinstance(value, dict)
            or set(value) != {"path", "bytes", "sha256", "mode"}
            or value.get("path") != expected_path
            or value.get("bytes") != authority_record.get("bytes")
            or value.get("sha256") != authority_record.get("sha256")
            or value.get("mode") not in {0o400, 0o500}
        ):
            msg = f"Execution snapshot {name} authority is invalid."
            raise ProviderInputError(msg)

    canonical = components.get("canonical_inputs")
    if (
        not isinstance(canonical, dict)
        or set(canonical) != {"cohorts", "cohort_count"}
        or canonical.get("cohort_count") != len(TCGA_COHORTS)
        or not isinstance(canonical.get("cohorts"), dict)
        or tuple(canonical["cohorts"]) != TCGA_COHORTS
    ):
        msg = "Execution snapshot canonical-input receipts are invalid."
        raise ProviderInputError(msg)
    for selected_cohort in TCGA_COHORTS:
        values = canonical["cohorts"].get(selected_cohort)
        if not isinstance(values, dict) or set(values) != {
            "canonical_maf",
            "sample_axis",
        }:
            msg = f"Execution snapshot lacks canonical cohort {selected_cohort}."
            raise ProviderInputError(msg)
        for name, expected_path in (
            ("canonical_maf", f"data/mafs/{selected_cohort}.maf"),
            ("sample_axis", f"data/axes/{selected_cohort}.txt"),
        ):
            value = values.get(name)
            expected_authority = context.bindings[selected_cohort][name]["file"]
            if (
                not isinstance(value, dict)
                or set(value) != {"path", "bytes", "sha256", "mode", "authority"}
                or value.get("path") != expected_path
                or value.get("mode") != 0o400
                or not _exact_json_equal(
                    value.get("authority"),
                    expected_authority,
                )
                or value.get("bytes") != expected_authority.get("bytes")
                or value.get("sha256") != expected_authority.get("sha256")
            ):
                msg = f"Snapshot no longer binds {selected_cohort}/{name} authority."
                raise ProviderInputError(msg)

    runtime_payload = _read_snapshot_runtime_authority(
        root,
        components["runtime_authority"],
        context,
    )
    runtime = receipt.get("runtime_boundary")
    expected_runtime = {
        "authority_file": components["runtime_authority"],
        "tools": runtime_payload["tools"],
    }
    if not isinstance(runtime, dict) or not _exact_json_equal(
        runtime,
        expected_runtime,
    ):
        msg = "Execution snapshot runtime boundary is invalid."
        raise ProviderInputError(msg)

    if full:
        inventory = _stream_snapshot_inventory(root)
        if (
            inventory.tree_sha256 != digest
            or inventory.file_count != receipt["file_count"]
        ):
            msg = "Execution snapshot full-tree content address no longer reproduces."
            raise ProviderInputError(msg)
        _require_snapshot_build_matches_components(inventory, components)
    elif cohort is not None:
        if cohort not in TCGA_COHORTS:
            msg = f"Unknown execution-snapshot cohort: {cohort}"
            raise ProviderInputError(msg)
        scoped_trees = {
            "dialect_python": "src/dialect",
        }
        if validate_provider_generation_sources:
            scoped_trees.update(
                {
                    "cbase": "external/CBaSE",
                    "mutsig_source": "external/MutSig2CV_src",
                },
            )
        for name, relative_root in scoped_trees.items():
            observed = _stream_tree_record(
                root / relative_root,
                required_directory_mode=0o500,
                required_file_modes=frozenset({0o400, 0o500}),
            )
            expected = components[name]
            if (
                observed.tree_sha256 != expected["tree_sha256"]
                or observed.file_count != expected["file_count"]
            ):
                msg = f"Execution snapshot {name} changed around child execution."
                raise ProviderInputError(msg)
        if validate_provider_generation_sources:
            for name, (expected_path, _authority) in individual_specs.items():
                _verify_snapshot_file(
                    root,
                    components[name],
                    label=f"snapshot {name}",
                    expected_path=expected_path,
                )
        for name, expected_path in (
            ("canonical_maf", f"data/mafs/{cohort}.maf"),
            ("sample_axis", f"data/axes/{cohort}.txt"),
        ):
            _verify_snapshot_file(
                root,
                canonical["cohorts"][cohort][name],
                label=f"snapshot {cohort}/{name}",
                expected_path=expected_path,
            )

    if require_current_execution_environment:
        _validate_current_runtime_authority(runtime_payload)
    return receipt, root


def _ensure_execution_snapshot(context: ProviderContext) -> dict[str, Any]:
    receipt_path = _execution_snapshot_receipt_path(context)
    if receipt_path.exists():
        receipt, _ = _validate_execution_snapshot(context, full=True)
        return receipt
    orchestration = context.paths.work_root / "_orchestration"
    for candidate in orchestration.iterdir():
        if (
            _EXECUTION_SNAPSHOT_READY_PATTERN.fullmatch(candidate.name) is not None
            and candidate.is_dir()
            and not candidate.is_symlink()
        ):
            candidate.chmod(0o700, follow_symlinks=False)
            for current, directory_names, file_names in os.walk(candidate):
                current_path = Path(current)
                for name in file_names:
                    (current_path / name).chmod(0o600, follow_symlinks=False)
                for name in directory_names:
                    (current_path / name).chmod(0o700, follow_symlinks=False)
            shutil.rmtree(candidate)
    return _build_execution_snapshot(context)


def _filesystem_inventory(root: Path) -> dict[str, list[str]]:
    root = _absolute_unresolved(root)
    root_descriptor = _open_directory_fd(root, label="provider inventory root")
    directories = [""]
    files: list[str] = []

    def walk(descriptor: int, prefix: PurePosixPath) -> None:
        for name in sorted(os.listdir(descriptor)):
            if name in {".", ".."} or "/" in name or "\x00" in name:
                msg = f"Provider tree contains an invalid entry name: {name!r}"
                raise ProviderInputError(msg)
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = prefix / name
            if stat.S_ISLNK(entry.st_mode):
                msg = f"Provider tree contains a symlink: {relative}"
                raise ProviderInputError(msg)
            if stat.S_ISDIR(entry.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    directories.append(relative.as_posix())
                    walk(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(entry.st_mode):
                msg = f"Provider tree contains a special entry: {relative}"
                raise ProviderInputError(msg)
            file_descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(file_descriptor)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    msg = f"Provider tree contains a hard-linked file: {relative}"
                    raise ProviderInputError(msg)
            finally:
                os.close(file_descriptor)
            relative_text = relative.as_posix()
            if name in FORBIDDEN_ASSOCIATION_FILES or any(
                part.startswith("id_") for part in relative.parts
            ):
                msg = f"Association identify output is forbidden: {relative_text}"
                raise ProviderInputError(msg)
            files.append(relative_text)

    try:
        walk(root_descriptor, PurePosixPath())
    finally:
        os.close(root_descriptor)
    return {"directories": sorted(directories), "files": sorted(files)}


def _is_owned_atomic_json_residue(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if len(parts) == 1:
        return (
            re.fullmatch(
                rf"\.{re.escape(ROOT_MANIFEST_NAME)}\.[0-9a-f]{{32}}\.tmp",
                parts[0],
            )
            is not None
        )
    if len(parts) == 2 and parts[0] == "resource_readbacks":
        return (
            re.fullmatch(
                r"\.[0-9a-f]{32}\.json\.[0-9a-f]{32}\.tmp",
                parts[1],
            )
            is not None
        )
    return bool(
        len(parts) == 3
        and parts[0] == "attempts"
        and parts[1] in TCGA_COHORTS
        and re.fullmatch(
            r"\.[0-9a-f]{32}\.json\.[0-9a-f]{32}\.tmp",
            parts[2],
        ),
    )


def _recover_owned_crash_residues(root: Path) -> list[str]:
    """Remove only exact pipeline/orchestrator temp names after authority replay."""
    inventory = _filesystem_inventory(root)
    staging_directories = {
        relative
        for relative in inventory["directories"]
        if len(PurePosixPath(relative).parts) == 2
        and PurePosixPath(relative).parts[0] == "mutsig"
        and (
            match := _MUTSIG_STAGING_PATTERN.fullmatch(
                PurePosixPath(relative).parts[1],
            )
        )
        and match.group(1) in TCGA_COHORTS
    }
    snapshot_build_directories = {
        relative
        for relative in inventory["directories"]
        if len(PurePosixPath(relative).parts) == 2
        and PurePosixPath(relative).parts[0] == "_orchestration"
        and _EXECUTION_SNAPSHOT_BUILD_PATTERN.fullmatch(
            PurePosixPath(relative).parts[1],
        )
        is not None
    }
    recovered: list[str] = []
    for relative in inventory["files"]:
        parts = PurePosixPath(relative).parts
        inside_staging = any(
            relative == staging or relative.startswith(f"{staging}/")
            for staging in staging_directories
        )
        pipeline_temp = bool(
            len(parts) == 3
            and parts[0] == "cohorts"
            and parts[1] in TCGA_COHORTS
            and _PIPELINE_TEMP_PATTERN.fullmatch(parts[2]),
        )
        if inside_staging:
            continue
        if pipeline_temp or _is_owned_atomic_json_residue(relative):
            (root / relative).unlink()
            recovered.append(relative)
    for relative in sorted(
        staging_directories | snapshot_build_directories,
        reverse=True,
    ):
        residue = root / relative
        if relative in snapshot_build_directories:
            for current, directory_names, file_names in os.walk(residue):
                current_path = Path(current)
                for name in file_names:
                    (current_path / name).chmod(0o600, follow_symlinks=False)
                for name in directory_names:
                    (current_path / name).chmod(0o700, follow_symlinks=False)
            residue.chmod(0o700, follow_symlinks=False)
        shutil.rmtree(residue)
        recovered.append(relative)
    return sorted(recovered)


def _require_allowed_work_inventory(root: Path) -> dict[str, list[str]]:
    inventory = _filesystem_inventory(root)
    cohorts = set(TCGA_COHORTS)
    snapshot_roots = {
        relative
        for relative in inventory["directories"]
        if len(PurePosixPath(relative).parts) == 2
        and PurePosixPath(relative).parts[0] == "_orchestration"
        and _EXECUTION_SNAPSHOT_READY_PATTERN.fullmatch(
            PurePosixPath(relative).parts[1],
        )
        is not None
    }
    if len(snapshot_roots) > 1:
        msg = "Provider work tree contains multiple execution snapshots."
        raise ProviderInputError(msg)

    def inside_snapshot(relative: str) -> bool:
        return any(
            relative == snapshot or relative.startswith(f"{snapshot}/")
            for snapshot in snapshot_roots
        )

    allowed_root_directories = {
        "_orchestration",
        "attempts",
        "cohorts",
        "mutsig",
        "resource_readbacks",
    }
    for relative in inventory["directories"]:
        parts = PurePosixPath(relative).parts
        allowed = relative == "" or relative in allowed_root_directories
        if inside_snapshot(relative):
            allowed = True
        if len(parts) == 2 and parts[0] in {"attempts", "cohorts", "mutsig"}:
            allowed = parts[1] in cohorts
        elif len(parts) == 3 and parts[0] == "cohorts":
            allowed = parts[1] in cohorts and parts[2] == "CBaSE_output"
        if not allowed:
            msg = f"Provider work tree contains an extra directory: {relative}"
            raise ProviderInputError(msg)
    for relative in inventory["files"]:
        parts = PurePosixPath(relative).parts
        allowed = relative in {
            WORK_AUTHORITY_PATH.as_posix(),
            EXECUTION_SNAPSHOT_RECEIPT.as_posix(),
            ROOT_MANIFEST_NAME,
        }
        if inside_snapshot(relative):
            allowed = True
        if len(parts) == 2 and parts[0] == "resource_readbacks":
            allowed = _READBACK_FILE_PATTERN.fullmatch(parts[1]) is not None
        elif len(parts) == 3 and parts[0] == "attempts":
            allowed = (
                parts[1] in cohorts
                and _ATTEMPT_FILE_PATTERN.fullmatch(parts[2]) is not None
            )
        elif len(parts) == 3 and parts[0] == "cohorts":
            allowed = parts[1] in cohorts and parts[2] in COHORT_ROOT_FILES
        elif len(parts) == 4 and parts[0] == "cohorts":
            allowed = (
                parts[1] in cohorts
                and parts[2] == "CBaSE_output"
                and parts[3] in CBASE_OUTPUT_FILES
            )
        elif len(parts) == 3 and parts[0] == "mutsig":
            allowed = parts[1] in cohorts and parts[2] in MUTSIG_OUTPUT_FILES
        if not allowed:
            msg = f"Provider work tree contains an extra file: {relative}"
            raise ProviderInputError(msg)
    return inventory


def _require_scoped_cohort_inventory(root: Path, cohort: str) -> None:
    cohort_inventory = _filesystem_inventory(root / "cohorts" / cohort)
    expected_cohort_files = set(COHORT_ROOT_FILES) | {
        f"CBaSE_output/{name}" for name in CBASE_OUTPUT_FILES
    }
    if (
        cohort_inventory["directories"] != ["", "CBaSE_output"]
        or set(cohort_inventory["files"]) != expected_cohort_files
    ):
        msg = f"Provider cohort closed inventory is partial or extra: {cohort}"
        raise ProviderInputError(msg)
    mutsig_inventory = _filesystem_inventory(root / "mutsig" / cohort)
    if mutsig_inventory["directories"] != [""] or set(
        mutsig_inventory["files"],
    ) != set(MUTSIG_OUTPUT_FILES):
        msg = f"Provider MutSig closed inventory is partial or extra: {cohort}"
        raise ProviderInputError(msg)


def _initialize_work_root(context: ProviderContext) -> None:
    root = context.paths.work_root
    if os.path.lexists(root):
        if root.is_symlink() or not root.is_dir():
            msg = f"Provider work root is not a non-symlink directory: {root}"
            raise ProviderInputError(msg)
        # This first pass is metadata-only: reject every symlink/special entry before
        # following the authority path, while still tolerating exact crash residues.
        _filesystem_inventory(root)
        authority_path = root / WORK_AUTHORITY_PATH
        if authority_path.is_symlink():
            msg = "Existing provider work authority is a forbidden symlink."
            raise ProviderInputError(msg)
        if not authority_path.is_file():
            msg = "Existing provider work root has partial or missing authority."
            raise ProviderInputError(msg)
        work_authority, authority_stat = _read_json_with_stat(
            authority_path,
            label="existing provider work authority",
        )
        if authority_stat.st_mode & 0o222:
            msg = "Existing provider work authority is not immutable."
            raise ProviderInputError(msg)
        if not _exact_json_equal(work_authority, context.authority):
            msg = (
                "Existing provider work root authority/source/provider hashes drifted."
            )
            raise ProviderInputError(msg)
        _recover_owned_crash_residues(root)
        _require_allowed_work_inventory(root)
        return
    initializer = root.with_name(f".{root.name}.init.{uuid.uuid4().hex}")
    published = False
    try:
        initializer.mkdir(mode=0o700)
        for name in (
            "_orchestration",
            "attempts",
            "cohorts",
            "mutsig",
            "resource_readbacks",
        ):
            (initializer / name).mkdir()
        authority_path = initializer / WORK_AUTHORITY_PATH
        _write_json_atomic(authority_path, context.authority, mode=0o444)
        _fsync_directory(authority_path.parent)
        _rename_exclusive(initializer, root)
        _fsync_directory(root.parent)
        published = True
    finally:
        if not published and initializer.exists():
            shutil.rmtree(initializer)


def _parse_exact_lines_bytes(raw: bytes, *, path: Path, label: str) -> list[str]:
    try:
        text = raw.decode()
    except UnicodeDecodeError as error:
        msg = f"{label} is not UTF-8: {path}"
        raise ProviderInputError(msg) from error
    lines = text.splitlines()
    if not lines or raw != ("\n".join(lines) + "\n").encode():
        msg = f"{label} must use LF separators and one terminal LF: {path}"
        raise ProviderInputError(msg)
    if any(not line or line != line.strip() for line in lines):
        msg = f"{label} contains a blank or padded line: {path}"
        raise ProviderInputError(msg)
    return lines


def _read_exact_lines(path: Path, *, label: str) -> list[str]:
    return _parse_exact_lines_bytes(
        _read_regular_bytes(path, label=label),
        path=path,
        label=label,
    )


def _parse_tsv_receipt_bytes(
    raw: bytes,
    *,
    path: Path,
    expected_fields: Sequence[str],
) -> dict[str, str]:
    lines = _parse_exact_lines_bytes(
        raw,
        path=path,
        label="provider stage receipt",
    )
    fields: dict[str, str] = {}
    observed_order: list[str] = []
    for line in lines:
        pieces = line.split("\t")
        if len(pieces) != 2 or not all(pieces) or pieces[0] in fields:
            msg = f"Provider stage receipt row is invalid: {path}"
            raise ProviderInputError(msg)
        observed_order.append(pieces[0])
        fields[pieces[0]] = pieces[1]
    if observed_order != list(expected_fields):
        msg = f"Provider stage receipt fields/order are invalid: {path}"
        raise ProviderInputError(msg)
    return fields


def _read_tsv_receipt(
    path: Path,
    expected_fields: Sequence[str],
) -> dict[str, str]:
    return _parse_tsv_receipt_bytes(
        _read_regular_bytes(path, label="provider stage receipt"),
        path=path,
        expected_fields=expected_fields,
    )


def _validate_cbase_projection(
    canonical_maf: Path,
    projected_path: Path,
    *,
    display_path: str,
) -> dict[str, Any]:
    """Chunk-reproduce the exact CBaSE projection without materializing the MAF."""

    class DigestWriter:
        def __init__(self) -> None:
            self.digest = hashlib.sha256()
            self.byte_count = 0

        def write(self, value: str) -> int:
            encoded = value.encode()
            self.digest.update(encoded)
            self.byte_count += len(encoded)
            return len(value)

    descriptor = _open_regular_fd(canonical_maf, label="canonical MAF projection")
    opened = os.fstat(descriptor)
    writer = DigestWriter()
    source_columns = [
        "Chromosome",
        "Start_Position",
        "Entrez_Gene_Id",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode",
    ]
    bytes_consumed = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            try:
                chunks = pd.read_csv(
                    source,
                    sep="\t",
                    low_memory=False,
                    chunksize=50_000,
                )
                observed_chunk = False
                for frame in chunks:
                    observed_chunk = True
                    frame[source_columns].to_csv(
                        writer,
                        sep="\t",
                        header=False,
                        index=False,
                        lineterminator="\n",
                    )
                if not observed_chunk:
                    msg = f"Canonical MAF contains no projection rows: {canonical_maf}"
                    raise ProviderInputError(msg)
                bytes_consumed = source.tell()
            except (KeyError, UnicodeDecodeError, ValueError) as error:
                msg = (
                    "Canonical MAF cannot be parsed for CBaSE projection: "
                    f"{canonical_maf}"
                )
                raise ProviderInputError(msg) from error
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require_stable_open_file(
        opened,
        after,
        label="canonical MAF projection",
        bytes_read=bytes_consumed,
    )
    observed = _stream_file_record(
        projected_path,
        display_path=display_path,
        label="persisted CBaSE projection",
    )
    if (
        observed["bytes"] != writer.byte_count
        or observed["sha256"] != writer.digest.hexdigest()
    ):
        msg = "CBaSE projection does not reproduce from the canonical MAF."
        raise ProviderInputError(msg)
    return {key: observed[key] for key in ("path", "bytes", "sha256")}


def _validate_count_matrix(
    count_matrix: Path,
    samples: Sequence[str],
    *,
    raw: bytes | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    content = (
        _read_regular_bytes(count_matrix, label="CBaSE count matrix")
        if raw is None
        else raw
    )
    try:
        rows = list(csv.reader(io.StringIO(content.decode(), newline="")))
    except UnicodeDecodeError as error:
        msg = f"CBaSE count matrix is not UTF-8: {count_matrix}"
        raise ProviderInputError(msg) from error
    if not rows or len(rows[0]) < 2 or any(not row for row in rows[1:]):
        msg = f"CBaSE count matrix is malformed: {count_matrix}"
        raise ProviderInputError(msg)
    features = tuple(rows[0][1:])
    if len(features) != len(set(features)) or any(
        _FEATURE_PATTERN.fullmatch(feature) is None for feature in features
    ):
        msg = f"CBaSE count matrix has an invalid feature axis: {count_matrix}"
        raise ProviderInputError(msg)
    observed = [row[0] for row in rows[1:]]
    if observed != list(samples):
        msg = "CBaSE count matrix does not preserve the signed sample axis."
        raise ProviderInputError(msg)
    expected_width = len(rows[0])
    parsed_rows: list[tuple[int, ...]] = []
    for row in rows[1:]:
        if len(row) != expected_width or any(
            _NONNEGATIVE_INTEGER_PATTERN.fullmatch(value) is None for value in row[1:]
        ):
            msg = "CBaSE count matrix values must be rectangular nonnegative integers."
            raise ProviderInputError(msg)
        values = tuple(int(value) for value in row[1:])
        if any(value > np.iinfo(np.int64).max for value in values):
            msg = "CBaSE count matrix contains an integer outside int64 range."
            raise ProviderInputError(msg)
        parsed_rows.append(values)
    counts_by_feature = {
        feature: tuple(row[position] for row in parsed_rows)
        for position, feature in enumerate(features)
    }
    return features, counts_by_feature


def _validate_pmf_table(
    path: Path,
    *,
    label: str,
    raw: bytes | None = None,
) -> dict[str, dict[int, float]]:
    content = (
        _read_regular_bytes(path, label=f"{label} PMF table") if raw is None else raw
    )
    try:
        reader = csv.reader(io.StringIO(content.decode(), newline=""))
        try:
            header = next(reader)
        except StopIteration as error:
            msg = f"{label} PMF table is empty: {path}"
            raise ProviderInputError(msg) from error
        count_keys = header[1:]
        if (
            len(header) < 2
            or len(count_keys) != len(set(count_keys))
            or any(
                _NONNEGATIVE_INTEGER_PATTERN.fullmatch(key) is None
                for key in count_keys
            )
        ):
            msg = f"{label} PMF table has invalid integer count keys: {path}"
            raise ProviderInputError(msg)
    except UnicodeDecodeError as error:
        msg = f"{label} PMF table is not UTF-8: {path}"
        raise ProviderInputError(msg) from error
    pmfs: dict[str, dict[int, float]] = {}
    seen: set[str] = set()
    for row in reader:
        if len(row) != len(header):
            msg = f"{label} PMF table is not rectangular: {path}"
            raise ProviderInputError(msg)
        feature = row[0]
        if _FEATURE_PATTERN.fullmatch(feature) is None or feature in seen:
            msg = f"{label} PMF table has an invalid feature axis: {path}"
            raise ProviderInputError(msg)
        seen.add(feature)
        probabilities: dict[int, float] = {}
        for raw_key, raw_probability in zip(count_keys, row[1:], strict=True):
            if raw_probability == "":
                continue
            try:
                probability = float(raw_probability)
            except ValueError as error:
                msg = f"{label} PMF table contains a nonnumeric value: {path}"
                raise ProviderInputError(msg) from error
            if not math.isfinite(probability) or probability < 0:
                msg = f"{label} PMF table contains an invalid probability: {path}"
                raise ProviderInputError(msg)
            probabilities[int(raw_key)] = probability
        if not probabilities or not math.isclose(
            math.fsum(probabilities.values()),
            1.0,
            rel_tol=1e-7,
            abs_tol=1e-8,
        ):
            msg = f"{label} PMF row does not sum to one: {path}/{feature}"
            raise ProviderInputError(msg)
        pmfs[feature] = probabilities
    if not pmfs:
        msg = f"{label} PMF table contains no feature rows: {path}"
        raise ProviderInputError(msg)
    return pmfs


def _reconstruct_cbase_count_matrix_bytes(
    kept_mutations: bytes,
    samples: Sequence[str],
    *,
    path: Path,
) -> tuple[bytes, bytes]:
    """Independently reproduce CBaSE's two persisted count-matrix serializations."""
    try:
        retained = pd.read_csv(
            io.BytesIO(kept_mutations),
            sep="\t",
            dtype={"sample": "string"},
            keep_default_na=False,
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        msg = f"CBaSE kept-mutation table is malformed: {path}"
        raise ProviderInputError(msg) from error
    required = {"sample", "gene", "effect"}
    if (
        not required.issubset(retained.columns)
        or len(retained.columns) != len(set(retained.columns))
        or retained["sample"].isna().any()
        or retained["gene"].isna().any()
        or retained["effect"].isna().any()
    ):
        msg = f"CBaSE kept-mutation table lacks exact count inputs: {path}"
        raise ProviderInputError(msg)
    sample_set = set(samples)
    retained_samples = set(retained["sample"].astype(str))
    outside_axis = sorted(retained_samples - sample_set)
    if outside_axis:
        msg = (
            "CBaSE kept-mutation table contains samples outside the signed axis: "
            f"{outside_axis[:5]}"
        )
        raise ProviderInputError(msg)
    counted = retained.loc[
        retained["effect"].isin(["missense", "nonsense"]),
        ["sample", "gene", "effect"],
    ].copy()
    gene_level = counted.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    gene_level = gene_level.reindex(samples, fill_value=0)
    counted["gene"] = counted["gene"] + "_" + counted["effect"].str[0].str.upper()
    effect_level = counted.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    effect_level = effect_level.reindex(samples, fill_value=0)
    return (
        effect_level.to_csv(index=True).encode(),
        gene_level.to_csv(index=True).encode(),
    )


def _validate_lambda_rates(
    path: Path,
    *,
    raw: bytes | None = None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0

    def validate_chunk(chunk: bytes) -> None:
        if len(chunk) % 4 != 0:
            msg = f"MutSig lambda tensor is not aligned float32 data: {path}"
            raise ProviderInputError(msg)
        values = np.frombuffer(chunk, dtype="<f4")
        if not np.isfinite(values).all() or (values < 0).any():
            msg = (
                "MutSig lambda tensor contains NaN, infinity, or a negative "
                f"rate: {path}"
            )
            raise ProviderInputError(msg)

    if raw is not None:
        for offset in range(0, len(raw), _HASH_CHUNK_BYTES):
            chunk = raw[offset : offset + _HASH_CHUNK_BYTES]
            validate_chunk(chunk)
            digest.update(chunk)
            total += len(chunk)
        return total, digest.hexdigest()
    descriptor = _open_regular_fd(path, label="MutSig lambda tensor")
    opened = os.fstat(descriptor)
    try:
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            validate_chunk(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require_stable_open_file(
        opened,
        after,
        label="MutSig lambda tensor",
        bytes_read=total,
    )
    return total, digest.hexdigest()


def _validate_cbase_and_dig(
    context: ProviderContext,
    cohort: str,
    canonical_maf: Path,
    sample_axis: Path,
) -> tuple[dict[str, Any], CohortBmrSemantics]:
    cohort_dir = context.paths.cohort_root / cohort
    cbase_output = cohort_dir / "CBaSE_output"
    required_root = {cohort_dir / name for name in COHORT_ROOT_FILES}
    required_cbase = {cbase_output / name for name in CBASE_OUTPUT_FILES}
    parsed_paths = {
        cohort_dir / "bmr_pmfs.csv",
        cohort_dir / "bmr_pmfs.dig.csv",
        cohort_dir / "cbase_stage_receipt.tsv",
        cohort_dir / "count_matrix.csv",
        cohort_dir / "dig_stage_receipt.tsv",
        cohort_dir / "gene_level_count_matrix.csv",
        cohort_dir / "sample_axis.txt",
        cbase_output / "kept_mutations.csv",
        cbase_output / "output_data_preparation.txt",
        cbase_output / "q_values.txt",
    }
    artifact_bytes: dict[Path, bytes] = {}
    for path in sorted(required_root | required_cbase):
        empty_allowed = path.name == "pipeline.log"
        try:
            if path in parsed_paths:
                content = _read_regular_bytes(
                    path,
                    label="required parsed provider artifact",
                )
                artifact_bytes[path] = content
                byte_count = len(content)
            else:
                byte_count = _stream_file_record(
                    path,
                    display_path=path.as_posix(),
                    label="required provider artifact",
                )["bytes"]
        except (OSError, ProviderInputError) as error:
            msg = f"Required provider artifact is missing, empty, or symlinked: {path}"
            raise ProviderInputError(msg) from error
        if byte_count == 0 and not empty_allowed:
            msg = f"Required provider artifact is missing, empty, or symlinked: {path}"
            raise ProviderInputError(msg)
    signed_axis_bytes = _read_regular_bytes(
        sample_axis,
        label=f"{cohort} signed sample axis",
    )
    samples = _parse_exact_lines_bytes(
        signed_axis_bytes,
        path=sample_axis,
        label=f"{cohort} signed sample axis",
    )
    provider_axis = cohort_dir / "sample_axis.txt"
    if artifact_bytes[provider_axis] != signed_axis_bytes:
        msg = f"Provider sample axis differs from signed authority: {cohort}"
        raise ProviderInputError(msg)
    reconstructed_effect, reconstructed_gene = _reconstruct_cbase_count_matrix_bytes(
        artifact_bytes[cbase_output / "kept_mutations.csv"],
        samples,
        path=cbase_output / "kept_mutations.csv",
    )
    if (
        artifact_bytes[cohort_dir / "count_matrix.csv"] != reconstructed_effect
        or artifact_bytes[cohort_dir / "gene_level_count_matrix.csv"]
        != reconstructed_gene
    ):
        msg = (
            "CBaSE count matrices do not reproduce byte-for-byte and semantically "
            f"from kept mutations plus the signed sample axis: {cohort}"
        )
        raise ProviderInputError(msg)
    count_features, counts_by_feature = _validate_count_matrix(
        cohort_dir / "count_matrix.csv",
        samples,
        raw=artifact_bytes[cohort_dir / "count_matrix.csv"],
    )
    cbase_pmfs = _validate_pmf_table(
        cohort_dir / "bmr_pmfs.csv",
        label="CBaSE",
        raw=artifact_bytes[cohort_dir / "bmr_pmfs.csv"],
    )
    dig_pmfs = _validate_pmf_table(
        cohort_dir / "bmr_pmfs.dig.csv",
        label="DIG",
        raw=artifact_bytes[cohort_dir / "bmr_pmfs.dig.csv"],
    )
    missing_cbase = [feature for feature in count_features if feature not in cbase_pmfs]
    if missing_cbase:
        msg = (
            "CBaSE PMFs do not cover the CBaSE count feature axis: "
            f"{cohort}/{missing_cbase[:5]}"
        )
        raise ProviderInputError(msg)
    projected_path = cohort_dir / "cbase_input.tsv"
    projection_record = _validate_cbase_projection(
        canonical_maf,
        projected_path,
        display_path=projected_path.relative_to(
            context.paths.work_root,
        ).as_posix(),
    )
    preparation_path = cbase_output / "output_data_preparation.txt"
    preparation_lines = _parse_exact_lines_bytes(
        artifact_bytes[preparation_path],
        path=preparation_path,
        label=f"{cohort} CBaSE preparation output",
    )
    if not preparation_lines[0].endswith(f"\tN_samples={len(samples)}"):
        msg = f"CBaSE persisted N does not equal the signed sample count: {cohort}"
        raise ProviderInputError(msg)

    pipeline_hash = context.authority["sources"]["cohort_pipeline"]["sha256"]
    dialect_hash = context.authority["sources"]["dialect_python_tree_sha256"]
    runtime_hash = context.authority["sources"]["python_runtime_sha256"]
    cbase_tree_hash = context.hashes.cbase_inputs_tree
    maf_hash = context.bindings[cohort]["canonical_maf"]["file"]["sha256"]
    axis_hash = context.bindings[cohort]["sample_axis"]["file"]["sha256"]
    cbase_input_hash = _sha256_text_lines(
        [
            pipeline_hash,
            maf_hash,
            axis_hash,
            dialect_hash,
            cbase_tree_hash,
            runtime_hash,
            "hg19",
        ],
    )
    cbase_outputs = [
        cohort_dir / "bmr_pmfs.csv",
        cohort_dir / "count_matrix.csv",
        cbase_output / "q_values.txt",
    ]
    cbase_output_hash = _files_sha256_from_bytes(
        [(path, artifact_bytes[path]) for path in cbase_outputs],
    )
    cbase_receipt_path = cohort_dir / "cbase_stage_receipt.tsv"
    cbase_receipt_bytes = artifact_bytes[cbase_receipt_path]
    cbase_receipt = _parse_tsv_receipt_bytes(
        cbase_receipt_bytes,
        path=cbase_receipt_path,
        expected_fields=STAGE_RECEIPT_FIELDS,
    )
    for key in ("input_sha256", "output_sha256"):
        _require_sha256(cbase_receipt[key], label=f"{cohort} CBaSE {key}")
    if (
        cbase_receipt["schema_version"] != "1"
        or cbase_receipt["input_sha256"] != cbase_input_hash
        or cbase_receipt["output_sha256"] != cbase_output_hash
    ):
        msg = f"CBaSE stage receipt is stale or misbound: {cohort}"
        raise ProviderInputError(msg)

    dig_input_hash = _sha256_text_lines(
        [
            pipeline_hash,
            maf_hash,
            axis_hash,
            hashlib.sha256(
                artifact_bytes[cohort_dir / "count_matrix.csv"],
            ).hexdigest(),
            dialect_hash,
            context.hashes.dig_results,
            runtime_hash,
            str(len(samples)),
            "hg19",
        ],
    )
    dig_output_hash = _files_sha256_from_bytes(
        [
            (
                cohort_dir / "bmr_pmfs.dig.csv",
                artifact_bytes[cohort_dir / "bmr_pmfs.dig.csv"],
            ),
        ],
    )
    dig_receipt_path = cohort_dir / "dig_stage_receipt.tsv"
    dig_receipt_bytes = artifact_bytes[dig_receipt_path]
    dig_receipt = _parse_tsv_receipt_bytes(
        dig_receipt_bytes,
        path=dig_receipt_path,
        expected_fields=STAGE_RECEIPT_FIELDS,
    )
    for key in ("input_sha256", "output_sha256"):
        _require_sha256(dig_receipt[key], label=f"{cohort} DIG {key}")
    if (
        dig_receipt["schema_version"] != "1"
        or dig_receipt["input_sha256"] != dig_input_hash
        or dig_receipt["output_sha256"] != dig_output_hash
    ):
        msg = f"DIG stage receipt is stale or misbound: {cohort}"
        raise ProviderInputError(msg)
    record = {
        "cbase": {
            "input_sha256": cbase_input_hash,
            "output_sha256": cbase_output_hash,
            "receipt": _file_record_from_bytes(
                cbase_receipt_path,
                cbase_receipt_bytes,
                display_path=cbase_receipt_path.relative_to(
                    context.paths.work_root,
                ).as_posix(),
            ),
            "projected_input": projection_record,
            "persisted_sample_count": len(samples),
        },
        "dig": {
            "input_sha256": dig_input_hash,
            "output_sha256": dig_output_hash,
            "receipt": _file_record_from_bytes(
                dig_receipt_path,
                dig_receipt_bytes,
                display_path=dig_receipt_path.relative_to(
                    context.paths.work_root,
                ).as_posix(),
            ),
        },
    }
    semantics = CohortBmrSemantics(
        count_features=count_features,
        counts_by_feature=counts_by_feature,
        cbase_pmfs=cbase_pmfs,
        dig_pmfs=dig_pmfs,
    )
    return record, semantics


def _validate_mutsig(
    context: ProviderContext,
    cohort: str,
    sample_axis: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    mutsig_dir = context.paths.mutsig_root / cohort
    artifact_bytes: dict[str, bytes] = {}
    lambda_size = 0
    lambda_hash = ""
    for name in sorted(MUTSIG_OUTPUT_FILES):
        path = mutsig_dir / name
        if name == "persample_lambda.f32":
            lambda_size, lambda_hash = _validate_lambda_rates(path)
            byte_count = lambda_size
        else:
            content = _read_regular_bytes(
                path,
                label="required parsed MutSig artifact",
            )
            artifact_bytes[name] = content
            byte_count = len(content)
        if byte_count == 0:
            msg = f"Required MutSig artifact is missing, empty, or symlinked: {path}"
            raise ProviderInputError(msg)
    receipt_path = mutsig_dir / "persample_receipt.tsv"
    receipt_bytes = artifact_bytes["persample_receipt.tsv"]
    receipt = _parse_tsv_receipt_bytes(
        receipt_bytes,
        path=receipt_path,
        expected_fields=MUTSIG_RECEIPT_FIELDS,
    )
    for key in (
        "source_tree_sha256",
        "patch_sha256",
        "runner_sha256",
        "runtime_sha256",
        "maf_sha256",
        "sample_axis_sha256",
        "lambda_sha256",
        "meta_sha256",
        "genes_sha256",
        "patients_sha256",
    ):
        _require_sha256(receipt[key], label=f"{cohort} MutSig {key}")
    sample_axis_bytes = _read_regular_bytes(
        sample_axis,
        label=f"{cohort} signed sample axis",
    )
    samples = _parse_exact_lines_bytes(
        sample_axis_bytes,
        path=sample_axis,
        label=f"{cohort} signed sample axis",
    )
    runtime_authority = (
        context.authority.get("providers", {}).get("mutsig", {}).get("runtime", {})
    )
    if not isinstance(runtime_authority, dict):
        msg = "Provider authority lacks the sealed MutSig runtime."
        raise ProviderInputError(msg)
    if (
        receipt["schema_version"] != "1"
        or receipt["cohort"] != cohort
        or receipt["upstream_commit"] != MUTSIG_UPSTREAM_COMMIT
        or receipt["source_tree_sha256"]
        != context.authority["providers"]["mutsig"]["source_tree_sha256"]
        or not receipt["source_file_count"].isdigit()
        or int(receipt["source_file_count"])
        != context.authority["providers"]["mutsig"]["source_file_count"]
        or receipt["patch_sha256"]
        != context.authority["sources"]["mutsig_patch"]["sha256"]
        or receipt["runner_sha256"]
        != context.authority["sources"]["mutsig_runner"]["sha256"]
        or receipt["runtime_sha256"] != runtime_authority.get("runtime_sha256")
        or receipt["maf_sha256"]
        != context.bindings[cohort]["canonical_maf"]["file"]["sha256"]
        or receipt["sample_axis_sha256"]
        != context.bindings[cohort]["sample_axis"]["file"]["sha256"]
        or not receipt["sample_axis_count"].isdigit()
        or int(receipt["sample_axis_count"]) != len(samples)
    ):
        msg = f"MutSig stage receipt is stale or misbound: {cohort}"
        raise ProviderInputError(msg)
    meta_path = mutsig_dir / "persample_meta.txt"
    meta_bytes = artifact_bytes["persample_meta.txt"]
    meta = _parse_tsv_receipt_bytes(
        meta_bytes,
        path=meta_path,
        expected_fields=("ng", "np", "neff"),
    )
    if not meta["ng"].isdigit() or not meta["np"].isdigit() or meta["neff"] != "2":
        msg = f"MutSig dimensions are invalid: {cohort}"
        raise ProviderInputError(msg)
    ng = int(meta["ng"])
    np_value = int(meta["np"])
    if ng <= 0 or np_value != len(samples):
        msg = f"MutSig dimensions do not equal the signed sample axis: {cohort}"
        raise ProviderInputError(msg)
    genes_path = mutsig_dir / "persample_genes.txt"
    genes_bytes = artifact_bytes["persample_genes.txt"]
    genes = _parse_exact_lines_bytes(
        genes_bytes,
        path=genes_path,
        label=f"{cohort} MutSig gene axis",
    )
    patients_path = mutsig_dir / "persample_patients.txt"
    patients_bytes = artifact_bytes["persample_patients.txt"]
    patients = _parse_exact_lines_bytes(
        patients_bytes,
        path=patients_path,
        label=f"{cohort} MutSig patient axis",
    )
    if len(genes) != ng or len(genes) != len(set(genes)) or patients != samples:
        msg = f"MutSig gene/patient axes are invalid: {cohort}"
        raise ProviderInputError(msg)
    lambda_path = mutsig_dir / "persample_lambda.f32"
    if lambda_size != ng * np_value * 2 * 4:
        msg = f"MutSig lambda tensor has the wrong byte length: {cohort}"
        raise ProviderInputError(msg)
    artifacts = {
        "lambda_sha256": lambda_path,
        "meta_sha256": meta_path,
        "genes_sha256": genes_path,
        "patients_sha256": patients_path,
    }
    artifact_content = {
        "meta_sha256": meta_bytes,
        "genes_sha256": genes_bytes,
        "patients_sha256": patients_bytes,
    }
    if receipt["lambda_sha256"] != lambda_hash or any(
        receipt[key] != hashlib.sha256(artifact_content[key]).hexdigest()
        for key in artifact_content
    ):
        msg = f"MutSig artifact hash does not match its receipt: {cohort}"
        raise ProviderInputError(msg)
    record = {
        "input": {
            "canonical_maf_sha256": receipt["maf_sha256"],
            "sample_axis_sha256": receipt["sample_axis_sha256"],
            "sample_axis_count": len(samples),
        },
        "source": {
            "upstream_commit": receipt["upstream_commit"],
            "source_tree_sha256": receipt["source_tree_sha256"],
            "source_file_count": int(receipt["source_file_count"]),
            "patch_sha256": receipt["patch_sha256"],
            "runner_sha256": receipt["runner_sha256"],
            "runtime_sha256": receipt["runtime_sha256"],
        },
        "dimensions": {"ng": ng, "np": np_value, "neff": 2},
        "receipt": _file_record_from_bytes(
            receipt_path,
            receipt_bytes,
            display_path=receipt_path.relative_to(context.paths.work_root).as_posix(),
        ),
        "artifacts": {
            "lambda": {
                "path": lambda_path.relative_to(context.paths.work_root).as_posix(),
                "bytes": lambda_size,
                "sha256": lambda_hash,
            },
            **{
                key.removesuffix("_sha256"): _file_record_from_bytes(
                    path,
                    artifact_content[key],
                    display_path=path.relative_to(
                        context.paths.work_root,
                    ).as_posix(),
                )
                for key, path in artifacts.items()
                if key != "lambda_sha256"
            },
        },
    }
    return record, tuple(genes)


def _pmf_has_observation_support(
    observed: Sequence[int],
    pmf: Mapping[int, float],
) -> bool:
    return all(
        float(pmf.get(count, 0.0)) + float(pmf.get(count - 1, 0.0)) > 0.0
        for count in set(observed)
    )


def _validate_k500_common_support(
    context: ProviderContext,
    cohort: str,
    semantics: CohortBmrSemantics,
    mutsig_genes: Sequence[str],
    *,
    expected_lambda_sha256: str | None = None,
) -> dict[str, Any]:
    gene_position = {gene: position for position, gene in enumerate(mutsig_genes)}
    if len(gene_position) != len(mutsig_genes):
        msg = f"MutSig gene axis is not unique: {cohort}"
        raise ProviderInputError(msg)
    native_common = [
        feature
        for feature in semantics.count_features
        if feature in semantics.cbase_pmfs
        and feature in semantics.dig_pmfs
        and feature.rsplit("_", 1)[0] in gene_position
    ]
    lambda_path = context.paths.mutsig_root / cohort / "persample_lambda.f32"
    sample_count = len(next(iter(semantics.counts_by_feature.values())))
    expected_bytes = len(mutsig_genes) * sample_count * 2 * 4
    descriptor = _open_regular_fd(
        lambda_path,
        label=f"{cohort} K=500 MutSig lambda tensor",
    )
    opened = os.fstat(descriptor)
    try:
        digest = hashlib.sha256()
        bytes_read = _copy_fd_chunks(descriptor, None, (digest,))
        _require_stable_open_file(
            opened,
            os.fstat(descriptor),
            label=f"{cohort} K=500 MutSig lambda tensor",
            bytes_read=bytes_read,
        )
        if bytes_read != expected_bytes:
            msg = f"MutSig lambda tensor changed dimensions before K=500 gate: {cohort}"
            raise ProviderInputError(msg)
        if (
            expected_lambda_sha256 is not None
            and digest.hexdigest() != expected_lambda_sha256
        ):
            msg = f"MutSig lambda tensor changed after receipt validation: {cohort}"
            raise ProviderInputError(msg)
        mapped = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        try:
            lambdas = np.ndarray(
                (len(mutsig_genes), sample_count, 2),
                dtype="<f4",
                buffer=mapped,
                order="F",
            )
            verified: list[str] = []
            for feature in native_common:
                observed_counts = semantics.counts_by_feature[feature]
                if not _pmf_has_observation_support(
                    observed_counts,
                    semantics.cbase_pmfs[feature],
                ) or not _pmf_has_observation_support(
                    observed_counts,
                    semantics.dig_pmfs[feature],
                ):
                    continue
                base_gene, effect = feature.rsplit("_", 1)
                effect_index = 0 if effect == "M" else 1
                observed = np.asarray(observed_counts, dtype=np.int64)
                rates = np.asarray(
                    lambdas[gene_position[base_gene], :, effect_index],
                    dtype=np.float64,
                )
                support = poisson.pmf(observed, rates) + poisson.pmf(
                    observed - 1,
                    rates,
                )
                if np.isfinite(support).all() and (support > 0).all():
                    verified.append(feature)
                    if len(verified) == TOP_K:
                        break
            del lambdas
        finally:
            mapped.close()
        _require_stable_open_file(
            opened,
            os.fstat(descriptor),
            label=f"{cohort} K=500 MutSig lambda tensor",
            bytes_read=bytes_read,
        )
    finally:
        os.close(descriptor)
    if len(verified) < TOP_K:
        msg = (
            f"Only {len(verified)} features have native, full observation support "
            f"across CBaSE, DIG, and MutSig for {cohort}; K={TOP_K} is impossible."
        )
        raise ProviderInputError(msg)
    return {
        "contract": "provider-k500-common-full-observation-support-v1",
        "required_features": TOP_K,
        "count_matrix_features": len(semantics.count_features),
        "native_common_features": len(native_common),
        "verified_full_support_lower_bound": len(verified),
        "verified_features_sha256": _sha256_text_lines(verified),
        "support_rule": "P(B=c) + P(B=c-1) > 0 for every signed sample",
        "association_results_opened": False,
    }


def validate_provider_cohort(
    context: ProviderContext,
    cohort: str,
) -> dict[str, Any]:
    """Validate all receipt-bound provider stages for one canonical cohort."""
    if cohort not in TCGA_COHORTS:
        msg = f"Unknown TCGA cohort: {cohort}"
        raise ProviderInputError(msg)
    _require_allowed_work_inventory(context.paths.work_root)
    _require_canonical_cohort_current(context, cohort)
    binding = context.bindings[cohort]
    snapshot_receipt = _execution_snapshot_receipt_path(context)
    if snapshot_receipt.exists():
        _, snapshot_root = _validate_execution_snapshot(
            context,
            cohort=cohort,
            require_current_execution_environment=False,
            validate_provider_generation_sources=False,
        )
        canonical_maf = snapshot_root / "data" / "mafs" / f"{cohort}.maf"
        sample_axis = snapshot_root / "data" / "axes" / f"{cohort}.txt"
    else:
        canonical_maf = binding["canonical_maf"]["path"]
        sample_axis = binding["sample_axis"]["path"]
    return _validate_provider_cohort_outputs(
        context,
        cohort,
        canonical_maf=canonical_maf,
        sample_axis=sample_axis,
    )


def _validate_provider_cohort_outputs(
    context: ProviderContext,
    cohort: str,
    *,
    canonical_maf: Path,
    sample_axis: Path,
) -> dict[str, Any]:
    binding = context.bindings[cohort]
    providers, semantics = _validate_cbase_and_dig(
        context,
        cohort,
        canonical_maf,
        sample_axis,
    )
    mutsig_record, mutsig_genes = _validate_mutsig(context, cohort, sample_axis)
    providers["mutsig"] = mutsig_record
    k500_support = _validate_k500_common_support(
        context,
        cohort,
        semantics,
        mutsig_genes,
        expected_lambda_sha256=mutsig_record["artifacts"]["lambda"]["sha256"],
    )
    return {
        "cohort": cohort,
        "canonical_inputs": {
            name: dict(binding[name]["file"])
            for name in (
                "child_manifest",
                "canonical_maf",
                "sample_axis",
                "population_manifest",
            )
        },
        "providers": providers,
        "k500_support": k500_support,
        "association_outputs_opened": False,
    }


def _require_canonical_cohort_current(
    context: ProviderContext,
    cohort: str,
) -> None:
    binding = context.bindings[cohort]
    for name in (
        "child_manifest",
        "canonical_maf",
        "sample_axis",
        "population_manifest",
    ):
        value = binding[name]
        path = value["path"]
        record = value["file"]
        observed = (
            _stream_file_record(
                path,
                display_path=str(record.get("path")),
                label=f"canonical {cohort}/{name}",
            )
            if isinstance(path, Path) and isinstance(record, dict)
            else {}
        )
        if (
            not isinstance(path, Path)
            or not isinstance(record, dict)
            or observed.get("sha256") != record.get("sha256")
            or observed.get("bytes") != record.get("bytes")
        ):
            msg = f"Canonical input changed after public validation: {cohort}/{name}"
            raise ProviderInputError(msg)


def _cohort_is_complete(context: ProviderContext, cohort: str) -> bool:
    _require_canonical_cohort_current(context, cohort)
    try:
        validate_provider_cohort(context, cohort)
    except (csv.Error, KeyError, OSError, OverflowError, ValueError):
        return False
    return True


def _pipeline_environment(
    context: ProviderContext,
    cohort: str,
    *,
    snapshot_root: Path | None = None,
    snapshot_receipt: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if snapshot_root is None or snapshot_receipt is None:
        observed_receipt, observed_root = _validate_execution_snapshot(
            context,
            cohort=cohort,
            require_current_execution_environment=False,
            validate_provider_generation_sources=False,
        )
        if snapshot_root is None:
            snapshot_root = observed_root
        if snapshot_receipt is None:
            snapshot_receipt = observed_receipt
    runtime = _runtime_authority_payload(context.authority)
    tools = runtime["tools"]
    python_runtime = runtime["python_runtime"]
    mutsig_runtime = runtime["mutsig_runtime"]
    runtime_file = snapshot_receipt["components"]["runtime_authority"]
    return {
        **_base_child_environment(),
        "DIALECT_PROVIDER_BASH": tools["bash"]["path"],
        "DIALECT_PROVIDER_BASH_SHA256": tools["bash"]["sha256"],
        "DIALECT_PROVIDER_CBASE_INPUTS_TREE_SHA256": (context.hashes.cbase_inputs_tree),
        "DIALECT_PROVIDER_DIALECT_TREE_SHA256": python_runtime["dialect_tree_sha256"],
        "DIALECT_PROVIDER_GIT": tools["git"]["path"],
        "DIALECT_PROVIDER_GIT_SHA256": tools["git"]["sha256"],
        "DIALECT_PROVIDER_JAVA": tools["java"]["path"],
        "DIALECT_PROVIDER_JAVA_HOME": mutsig_runtime["java_home"],
        "DIALECT_PROVIDER_JAVA_ID": mutsig_runtime["java_id"],
        "DIALECT_PROVIDER_JAVA_SHA256": tools["java"]["sha256"],
        "DIALECT_PROVIDER_MUTSIG_RUNTIME_SHA256": mutsig_runtime["runtime_sha256"],
        "DIALECT_PROVIDER_MUTSIG_SOURCE_FILE_COUNT": str(
            context.authority["providers"]["mutsig"]["source_file_count"],
        ),
        "DIALECT_PROVIDER_MUTSIG_SOURCE_TREE_SHA256": context.authority["providers"][
            "mutsig"
        ]["source_tree_sha256"],
        "DIALECT_PROVIDER_NICE": tools["nice"]["path"],
        "DIALECT_PROVIDER_NICE_SHA256": tools["nice"]["sha256"],
        "DIALECT_PROVIDER_OCTAVE": tools["octave"]["path"],
        "DIALECT_PROVIDER_OCTAVE_ID": mutsig_runtime["octave_id"],
        "DIALECT_PROVIDER_OCTAVE_SHA256": tools["octave"]["sha256"],
        "DIALECT_PROVIDER_PYTHON": tools["python"]["path"],
        "DIALECT_PROVIDER_PYTHON_RUNTIME_SHA256": python_runtime["runtime_sha256"],
        "DIALECT_PROVIDER_PYTHON_SHA256": tools["python"]["sha256"],
        "DIALECT_PROVIDER_RUNTIME_AUTHORITY_FILE": (
            snapshot_root.joinpath(*RUNTIME_AUTHORITY_PATH.parts).as_posix()
        ),
        "DIALECT_PROVIDER_RUNTIME_AUTHORITY_SHA256": runtime_file["sha256"],
        "MAF_DIR": (snapshot_root / "data" / "mafs").as_posix(),
        "MUTSIG_ROOT": context.paths.mutsig_root.as_posix(),
        "MUTSIG_SAMPLE_AXIS_FILE": (
            snapshot_root / "data" / "axes" / f"{cohort}.txt"
        ).as_posix(),
        "PREPARE_ONLY": "1",
        "PYTHONPATH": (snapshot_root / "src").as_posix(),
        "ROOT": context.paths.cohort_root.as_posix(),
        "TOP_K": str(TOP_K),
    }


def _pipeline_command(
    context: ProviderContext,
    cohort: str,
    *,
    snapshot_root: Path | None = None,
) -> list[str]:
    if snapshot_root is None:
        _, snapshot_root = _validate_execution_snapshot(
            context,
            cohort=cohort,
            require_current_execution_environment=False,
            validate_provider_generation_sources=False,
        )
    return [
        context.authority["sources"]["nice_executable"]["path"],
        "-n",
        str(NICE_INCREMENT),
        context.authority["sources"]["bash_executable"]["path"],
        (snapshot_root / "scripts" / "run_cohort_pipeline.sh").as_posix(),
        cohort,
    ]


def _record_attempt(  # noqa: PLR0913
    context: ProviderContext,
    cohort: str,
    attempt_id: str,
    command: Sequence[str],
    log_path: Path,
    *,
    started_at_utc: str,
    exit_status: int | None,
    launch_error: str | None,
    execution_snapshot: Mapping[str, Any],
    snapshot_rehashed_after_child: bool,
    snapshot_validation_error: str | None,
) -> None:
    record = {
        "schema_version": SCHEMA_VERSION,
        "contract": PROVIDER_INPUT_CONTRACT,
        "cohort": cohort,
        "attempt_id": attempt_id,
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now(),
        "command": list(command),
        "environment": {
            **THREAD_ENVIRONMENT,
            "PREPARE_ONLY": "1",
            "TOP_K": str(TOP_K),
        },
        "execution_snapshot": {
            "contract": execution_snapshot["contract"],
            "root": execution_snapshot["root"],
            "tree_sha256": execution_snapshot["tree_sha256"],
            "canonical_input": execution_snapshot["components"]["canonical_inputs"][
                "cohorts"
            ][cohort],
            "validation_scope": "cohort-inputs-and-exact-shared-runtime",
            "rehashed_after_child": snapshot_rehashed_after_child,
            "external_runtime_boundary_revalidated_after_child": (
                snapshot_rehashed_after_child
            ),
            "post_child_validation_error": snapshot_validation_error,
        },
        "exit_status": exit_status,
        "launch_error": launch_error,
        "log": _file_record(
            log_path,
            display_path=log_path.relative_to(context.paths.work_root).as_posix(),
        ),
        "association_outputs_opened": False,
    }
    destination = log_path.with_suffix(".json")
    _write_json_atomic(destination, record)


def _invoke_pipeline(context: ProviderContext, cohort: str) -> tuple[str, int]:
    _require_canonical_cohort_current(context, cohort)
    snapshot_receipt, snapshot_root = _validate_execution_snapshot(
        context,
        cohort=cohort,
    )
    attempt_id = uuid.uuid4().hex
    attempt_dir = context.paths.work_root / "attempts" / cohort
    attempt_dir.mkdir(exist_ok=True)
    log_path = attempt_dir / f"{attempt_id}.log"
    command = _pipeline_command(context, cohort, snapshot_root=snapshot_root)
    environment = _pipeline_environment(
        context,
        cohort,
        snapshot_root=snapshot_root,
        snapshot_receipt=snapshot_receipt,
    )
    started_at = _utc_now()
    exit_status: int | None = None
    launch_error: str | None = None
    snapshot_validation_error: str | None = None
    try:
        with log_path.open("xb") as log:
            completed = subprocess.run(
                command,
                cwd=snapshot_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_status = completed.returncode
    except OSError as error:
        launch_error = f"{type(error).__name__}: {error}"
        if not log_path.exists():
            log_path.touch(exist_ok=False)
    finally:
        try:
            _validate_execution_snapshot(context, cohort=cohort)
        except (OSError, ProviderInputError) as error:
            snapshot_validation_error = f"{type(error).__name__}: {error}"
        _record_attempt(
            context,
            cohort,
            attempt_id,
            command,
            log_path,
            started_at_utc=started_at,
            exit_status=exit_status,
            launch_error=launch_error,
            execution_snapshot=snapshot_receipt,
            snapshot_rehashed_after_child=snapshot_validation_error is None,
            snapshot_validation_error=snapshot_validation_error,
        )
    if snapshot_validation_error is not None:
        msg = (
            f"Execution snapshot/runtime changed while provider child ran for "
            f"{cohort}: {snapshot_validation_error}"
        )
        raise ProviderInputError(msg)
    if launch_error is not None:
        msg = f"Provider pipeline could not launch for {cohort}: {launch_error}"
        raise ProviderInputError(msg)
    if exit_status is None:
        msg = f"Provider pipeline returned no exit status for {cohort}."
        raise ProviderInputError(msg)
    return cohort, exit_status


def plan_provider_waves(
    cohorts: Sequence[str],
    *,
    jobs: int,
) -> list[tuple[str, ...]]:
    """Plan CHOL first, ordinary bounded waves next, and heavy serial waves last."""
    selected = tuple(cohorts)
    if selected != TCGA_COHORTS:
        msg = "Provider wave planning requires the exact ordered 32-cohort family."
        raise ProviderInputError(msg)
    if jobs <= 0 or jobs > MAX_JOBS:
        msg = f"Provider wave size must be between 1 and {MAX_JOBS}."
        raise ProviderInputError(msg)
    ordinary = [
        cohort
        for cohort in selected
        if cohort != CANARY_COHORT and cohort not in MEMORY_HEAVY_COHORTS
    ]
    heavy = [cohort for cohort in selected if cohort in MEMORY_HEAVY_COHORTS]
    waves: list[tuple[str, ...]] = [(CANARY_COHORT,)]
    waves.extend(
        tuple(ordinary[start : start + jobs]) for start in range(0, len(ordinary), jobs)
    )
    waves.extend((cohort,) for cohort in heavy)
    return waves


def _run_wave(
    context: ProviderContext,
    wave: Sequence[str],
    *,
    wave_number: int,
) -> None:
    _require_allowed_work_inventory(context.paths.work_root)
    pending = [cohort for cohort in wave if not _cohort_is_complete(context, cohort)]
    if not pending:
        return
    _require_live_resource_gate(
        context,
        jobs=len(pending),
        label=f"wave-{wave_number}-{'-'.join(pending)}",
    )
    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=len(pending)) as executor:
        futures = {
            executor.submit(_invoke_pipeline, context, cohort): cohort
            for cohort in pending
        }
        for future in as_completed(futures):
            cohort, return_code = future.result()
            if return_code != 0:
                failures.append((cohort, return_code))
    _require_allowed_work_inventory(context.paths.work_root)
    if failures:
        msg = f"Provider pipeline wave failed; resumable work retained: {failures}"
        raise ProviderInputError(msg)
    for cohort in pending:
        validate_provider_cohort(context, cohort)


def _inventory_file_records(
    root: Path,
    inventory: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    snapshot_roots = {
        relative
        for relative in inventory["directories"]
        if len(PurePosixPath(relative).parts) == 2
        and PurePosixPath(relative).parts[0] == "_orchestration"
        and _EXECUTION_SNAPSHOT_READY_PATTERN.fullmatch(
            PurePosixPath(relative).parts[1],
        )
        is not None
    }

    def inside_snapshot(relative: str) -> bool:
        return any(relative.startswith(f"{snapshot}/") for snapshot in snapshot_roots)

    records = []
    for relative in inventory["files"]:
        if relative == ROOT_MANIFEST_NAME or inside_snapshot(relative):
            continue
        records.append(_file_record(root / relative, display_path=relative))
    return records


def _execution_snapshot_inventory_summary(
    root: Path,
    inventory: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    receipt = _read_json(root.joinpath(*EXECUTION_SNAPSHOT_RECEIPT.parts))
    root_name = receipt.get("root")
    relative_root = f"_orchestration/{root_name}"
    snapshot_files = [
        relative
        for relative in inventory["files"]
        if relative.startswith(f"{relative_root}/")
    ]
    snapshot_directories = [
        relative
        for relative in inventory["directories"]
        if relative == relative_root or relative.startswith(f"{relative_root}/")
    ]
    if (
        not isinstance(root_name, str)
        or _EXECUTION_SNAPSHOT_READY_PATTERN.fullmatch(root_name) is None
        or len(snapshot_files) != receipt.get("file_count")
        or not snapshot_directories
    ):
        msg = "Execution snapshot closed-inventory summary is invalid."
        raise ProviderInputError(msg)
    return {
        "root": relative_root,
        "tree_hash_contract": receipt.get("tree_hash_contract"),
        "tree_sha256": receipt.get("tree_sha256"),
        "file_count": len(snapshot_files),
        "directory_count": len(snapshot_directories),
        "individual_file_receipts_omitted": True,
    }


def _require_complete_provider_layout(root: Path) -> None:
    inventory = _require_allowed_work_inventory(root)
    directories = set(inventory["directories"])
    files = set(inventory["files"])
    required_directories = {
        "",
        "_orchestration",
        "attempts",
        "cohorts",
        "mutsig",
        "resource_readbacks",
    }
    required_directories.update(f"cohorts/{cohort}" for cohort in TCGA_COHORTS)
    required_directories.update(
        f"cohorts/{cohort}/CBaSE_output" for cohort in TCGA_COHORTS
    )
    required_directories.update(f"mutsig/{cohort}" for cohort in TCGA_COHORTS)
    required_files = {
        WORK_AUTHORITY_PATH.as_posix(),
        EXECUTION_SNAPSHOT_RECEIPT.as_posix(),
    }
    for cohort in TCGA_COHORTS:
        required_files.update(f"cohorts/{cohort}/{name}" for name in COHORT_ROOT_FILES)
        required_files.update(
            f"cohorts/{cohort}/CBaSE_output/{name}" for name in CBASE_OUTPUT_FILES
        )
        required_files.update(f"mutsig/{cohort}/{name}" for name in MUTSIG_OUTPUT_FILES)
    missing_directories = sorted(required_directories - directories)
    missing_files = sorted(required_files - files)
    if missing_directories or missing_files:
        msg = (
            "Complete provider tree has missing closed-inventory entries: "
            f"directories={missing_directories[:5]}, files={missing_files[:5]}"
        )
        raise ProviderInputError(msg)


def _build_root_manifest(context: ProviderContext) -> dict[str, Any]:
    cohort_records = [
        validate_provider_cohort(context, cohort) for cohort in TCGA_COHORTS
    ]
    _require_complete_provider_layout(context.paths.work_root)
    inventory = _require_allowed_work_inventory(context.paths.work_root)
    if ROOT_MANIFEST_NAME in inventory["files"]:
        msg = "Provider root manifest already exists and cannot be overwritten."
        raise ProviderInputError(msg)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": PROVIDER_INPUT_CONTRACT,
        "completed_at_utc": _utc_now(),
        "cohorts": list(TCGA_COHORTS),
        "cohort_count": len(TCGA_COHORTS),
        "roots": {
            "cohorts": "cohorts",
            "mutsig": "mutsig",
        },
        "authority": dict(context.authority["authority"]),
        "sources": dict(context.authority["sources"]),
        "providers": dict(context.authority["providers"]),
        "execution": dict(context.authority["execution"]),
        "scope": dict(context.authority["scope"]),
        "cohort_provider_receipts": cohort_records,
        "inventory": {
            "directories": list(inventory["directories"]),
            "files": _inventory_file_records(context.paths.work_root, inventory),
            "execution_snapshot": _execution_snapshot_inventory_summary(
                context.paths.work_root,
                inventory,
            ),
            "root_manifest_excluded_from_self_inventory": ROOT_MANIFEST_NAME,
        },
    }


def _verify_manifest_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    value = manifest.get("inventory")
    if not isinstance(value, dict) or set(value) != {
        "directories",
        "files",
        "execution_snapshot",
        "root_manifest_excluded_from_self_inventory",
    }:
        msg = "Provider root manifest has an invalid closed inventory."
        raise ProviderInputError(msg)
    observed = _require_allowed_work_inventory(root)
    snapshot_summary = _execution_snapshot_inventory_summary(root, observed)
    snapshot_root = snapshot_summary["root"]
    observed_files = [
        relative
        for relative in observed["files"]
        if relative != ROOT_MANIFEST_NAME
        and not relative.startswith(f"{snapshot_root}/")
    ]
    if (
        value["directories"] != observed["directories"]
        or value["root_manifest_excluded_from_self_inventory"] != ROOT_MANIFEST_NAME
        or not _exact_json_equal(
            value["execution_snapshot"],
            snapshot_summary,
        )
        or not isinstance(value["files"], list)
        or [record.get("path") for record in value["files"] if isinstance(record, dict)]
        != observed_files
    ):
        msg = "Provider root closed inventory differs from the filesystem."
        raise ProviderInputError(msg)
    for record in value["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            msg = "Provider root inventory contains an invalid file receipt."
            raise ProviderInputError(msg)
        path = root / str(record["path"])
        observed_record = _file_record(
            path,
            display_path=str(record["path"]),
        )
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or not _exact_json_equal(observed_record, record)
        ):
            msg = f"Provider root inventory receipt changed: {record.get('path')}"
            raise ProviderInputError(msg)


def _context_for_existing_root(context: ProviderContext, root: Path) -> ProviderContext:
    paths = ProviderPaths(
        **{
            **asdict(context.paths),
            "work_root": root,
            "cohort_root": root / "cohorts",
            "mutsig_root": root / "mutsig",
        },
    )
    return ProviderContext(
        paths=paths,
        hashes=context.hashes,
        canonical_manifest=context.canonical_manifest,
        bindings=context.bindings,
        authority=context.authority,
    )


def _validate_published_root(
    context: ProviderContext,
    root: Path,
    *,
    validate_execution_snapshot_full: bool = False,
    require_current_execution_environment: bool = True,
) -> dict[str, Any]:
    manifest_path = root / ROOT_MANIFEST_NAME
    manifest, manifest_stat = _read_json_with_stat(
        manifest_path,
        label="published provider root manifest",
    )
    expected_keys = {
        "schema_version",
        "contract",
        "completed_at_utc",
        "cohorts",
        "cohort_count",
        "roots",
        "authority",
        "sources",
        "providers",
        "execution",
        "scope",
        "cohort_provider_receipts",
        "inventory",
    }
    existing_context = _context_for_existing_root(context, root)
    if validate_execution_snapshot_full:
        _validate_execution_snapshot(
            existing_context,
            full=True,
            require_current_execution_environment=(
                require_current_execution_environment
            ),
        )
    try:
        completed_at = datetime.fromisoformat(str(manifest.get("completed_at_utc")))
        completed_at_is_utc = (
            completed_at.tzinfo is not None
            and completed_at.utcoffset() == UTC.utcoffset(None)
        )
    except ValueError:
        completed_at_is_utc = False
    if (
        set(manifest) != expected_keys
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["contract"] != PROVIDER_INPUT_CONTRACT
        or manifest["cohorts"] != list(TCGA_COHORTS)
        or manifest["cohort_count"] != len(TCGA_COHORTS)
        or not _exact_json_equal(
            manifest["roots"],
            {"cohorts": "cohorts", "mutsig": "mutsig"},
        )
        or not _exact_json_equal(
            manifest["authority"],
            context.authority["authority"],
        )
        or not _exact_json_equal(
            manifest["sources"],
            context.authority["sources"],
        )
        or not _exact_json_equal(
            manifest["providers"],
            context.authority["providers"],
        )
        or not _exact_json_equal(
            manifest["execution"],
            context.authority["execution"],
        )
        or not _exact_json_equal(
            manifest["scope"],
            context.authority["scope"],
        )
        or not completed_at_is_utc
    ):
        msg = "Published provider manifest authority/source/provider contract drifted."
        raise ProviderInputError(msg)
    _verify_manifest_inventory(root, manifest)
    if manifest_stat.st_mode & 0o222:
        msg = f"Published provider root manifest is not immutable: {manifest_path}"
        raise ProviderInputError(msg)
    work_authority_path = root / WORK_AUTHORITY_PATH
    work_authority, work_authority_stat = _read_json_with_stat(
        work_authority_path,
        label="published provider work authority",
    )
    if work_authority_stat.st_mode & 0o222:
        msg = (
            f"Published provider work authority is not immutable: {work_authority_path}"
        )
        raise ProviderInputError(msg)
    if not _exact_json_equal(work_authority, context.authority):
        msg = "Published provider root contains partial or drifted work authority."
        raise ProviderInputError(msg)
    records = [
        validate_provider_cohort(existing_context, cohort) for cohort in TCGA_COHORTS
    ]
    if not _exact_json_equal(manifest["cohort_provider_receipts"], records):
        msg = "Published provider cohort receipts do not reproduce."
        raise ProviderInputError(msg)
    return manifest


def _provider_file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": path,
        "file": _file_record(path, display_path=relative),
    }


def _published_cohort_bindings(
    root: Path,
    manifest: Mapping[str, Any],
    canonical_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    receipt_by_cohort = {
        record["cohort"]: record for record in manifest["cohort_provider_receipts"]
    }
    result: dict[str, dict[str, Any]] = {}
    for cohort in TCGA_COHORTS:
        result[cohort] = _published_single_cohort_binding(
            root,
            cohort,
            receipt_by_cohort[cohort],
            canonical_bindings[cohort],
        )
    return result


def _published_single_cohort_binding(
    root: Path,
    cohort: str,
    provider_receipt: Mapping[str, Any],
    canonical_binding: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_prefix = f"cohorts/{cohort}"
    mutsig_prefix = f"mutsig/{cohort}"
    return {
        "cohort": cohort,
        "cohort_root": root / cohort_prefix,
        "mutsig_root": root / mutsig_prefix,
        "count_matrix": _provider_file_binding(
            root,
            f"{cohort_prefix}/count_matrix.csv",
        ),
        "cbase_pmfs": _provider_file_binding(
            root,
            f"{cohort_prefix}/bmr_pmfs.csv",
        ),
        "dig_pmfs": _provider_file_binding(
            root,
            f"{cohort_prefix}/bmr_pmfs.dig.csv",
        ),
        "sample_axis": _provider_file_binding(
            root,
            f"{cohort_prefix}/sample_axis.txt",
        ),
        "mutsig_lambda": _provider_file_binding(
            root,
            f"{mutsig_prefix}/persample_lambda.f32",
        ),
        "mutsig_metadata": _provider_file_binding(
            root,
            f"{mutsig_prefix}/persample_meta.txt",
        ),
        "mutsig_genes": _provider_file_binding(
            root,
            f"{mutsig_prefix}/persample_genes.txt",
        ),
        "mutsig_patients": _provider_file_binding(
            root,
            f"{mutsig_prefix}/persample_patients.txt",
        ),
        "mutsig_receipt": _provider_file_binding(
            root,
            f"{mutsig_prefix}/persample_receipt.tsv",
        ),
        "canonical_inputs": dict(canonical_binding),
        "provider_receipt": provider_receipt,
    }


def _published_scoped_cohort_binding(
    root: Path,
    snapshot_root: Path,
    cohort: str,
    provider_receipt: Mapping[str, Any],
    canonical_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose only real paths; retain non-snapshotted canonical inputs as receipts."""
    provider_binding = _published_single_cohort_binding(
        root,
        cohort,
        provider_receipt,
        {},
    )
    provider_binding["canonical_inputs"] = {
        "canonical_maf": {
            "path": snapshot_root / "data" / "mafs" / f"{cohort}.maf",
            "file": dict(canonical_binding["canonical_maf"]["file"]),
        },
        "sample_axis": {
            "path": snapshot_root / "data" / "axes" / f"{cohort}.txt",
            "file": dict(canonical_binding["sample_axis"]["file"]),
        },
    }
    provider_binding["canonical_input_receipts"] = {
        name: dict(canonical_binding[name]["file"])
        for name in ("child_manifest", "population_manifest")
    }
    return provider_binding


def _require_scoped_binding_matches_manifest_inventory(
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    cohort: str,
) -> None:
    inventory = manifest.get("inventory")
    records = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(records, list):
        msg = "Scoped provider manifest lacks its full-acceptance file receipts."
        raise ProviderInputError(msg)
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record.get("path"), str)
            or record["path"] in by_path
        ):
            msg = "Scoped provider manifest file receipts are malformed or duplicated."
            raise ProviderInputError(msg)
        by_path[record["path"]] = record
    expected_paths = {
        "count_matrix": f"cohorts/{cohort}/count_matrix.csv",
        "cbase_pmfs": f"cohorts/{cohort}/bmr_pmfs.csv",
        "dig_pmfs": f"cohorts/{cohort}/bmr_pmfs.dig.csv",
        "sample_axis": f"cohorts/{cohort}/sample_axis.txt",
        "mutsig_lambda": f"mutsig/{cohort}/persample_lambda.f32",
        "mutsig_metadata": f"mutsig/{cohort}/persample_meta.txt",
        "mutsig_genes": f"mutsig/{cohort}/persample_genes.txt",
        "mutsig_patients": f"mutsig/{cohort}/persample_patients.txt",
        "mutsig_receipt": f"mutsig/{cohort}/persample_receipt.tsv",
    }
    for name, relative in expected_paths.items():
        value = binding.get(name)
        if not isinstance(value, dict) or not _exact_json_equal(
            value.get("file"),
            by_path.get(relative),
        ):
            msg = f"Scoped provider file differs from full acceptance: {relative}"
            raise ProviderInputError(msg)


def _full_acceptance_receipt(
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    authority_payload = {
        key: manifest[key]
        for key in ("authority", "sources", "providers", "execution", "scope")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": FULL_ACCEPTANCE_CONTRACT,
        "provider_manifest_sha256": manifest_sha256,
        "execution_snapshot": manifest["inventory"]["execution_snapshot"],
        "authority_sha256": hashlib.sha256(
            _canonical_json(authority_payload),
        ).hexdigest(),
        "cohort_receipts_sha256": hashlib.sha256(
            _canonical_json(manifest["cohort_provider_receipts"]),
        ).hexdigest(),
        "full_inventory_validated": True,
        "association_outputs_opened": False,
    }


def full_acceptance_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the independent digest passed from a full parent to scoped children."""
    return hashlib.sha256(_canonical_json(dict(receipt)) + b"\n").hexdigest()


def _validate_full_acceptance_receipt(
    value: object,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    expected_digest = _require_sha256(
        expected_receipt_sha256,
        label="expected provider full-acceptance receipt SHA-256",
    )
    receipt = _require_exact_dict(
        value,
        {
            "schema_version",
            "contract",
            "provider_manifest_sha256",
            "execution_snapshot",
            "authority_sha256",
            "cohort_receipts_sha256",
            "full_inventory_validated",
            "association_outputs_opened",
        },
        label="provider full-acceptance receipt",
    )
    if full_acceptance_receipt_sha256(receipt) != expected_digest:
        msg = "Provider full-acceptance receipt differs from its independent SHA-256."
        raise ProviderInputError(msg)
    expected = _full_acceptance_receipt(manifest, manifest_sha256)
    if not _exact_json_equal(receipt, expected):
        msg = "Provider full-acceptance receipt is partial, forged, or drifted."
        raise ProviderInputError(msg)
    return receipt


def _scoped_context_from_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    work_authority: Mapping[str, Any],
) -> tuple[ProviderContext, Path]:
    paths = ProviderPaths(
        repo_root=root,
        canonical_input_root=root,
        approval_manifest=root / ROOT_MANIFEST_NAME,
        output_root=root,
        work_root=root,
        cohort_root=root / "cohorts",
        mutsig_root=root / "mutsig",
        cbase_inputs=root,
        dig_results=root / ROOT_MANIFEST_NAME,
        pipeline=root / ROOT_MANIFEST_NAME,
        mutsig_runner=root / ROOT_MANIFEST_NAME,
        mutsig_patch=root / ROOT_MANIFEST_NAME,
    )
    cbase_hash, dig_hash = _validate_historical_authority_contract(
        work_authority,
        paths,
        require_current_paths=False,
    )
    input_authority = _relocatable_input_authority_view(work_authority["authority"])
    hashes = IndependentHashes(
        approval=input_authority["expected_approval_sha256"],
        canonical_input_manifest=input_authority[
            "expected_canonical_input_manifest_sha256"
        ],
        cbase_inputs_tree=cbase_hash,
        dig_results=dig_hash,
    )
    snapshot_receipt = _read_json(root.joinpath(*EXECUTION_SNAPSHOT_RECEIPT.parts))
    snapshot_root = _snapshot_root_from_receipt(
        ProviderContext(paths, hashes, {}, {}, work_authority),
        snapshot_receipt,
    )
    cohort_receipts = manifest.get("cohort_provider_receipts")
    if (
        not isinstance(cohort_receipts, list)
        or len(cohort_receipts) != len(TCGA_COHORTS)
        or [
            record.get("cohort") if isinstance(record, dict) else None
            for record in cohort_receipts
        ]
        != list(TCGA_COHORTS)
    ):
        msg = "Provider manifest cohort receipts are incomplete or unordered."
        raise ProviderInputError(msg)
    bindings: dict[str, dict[str, Any]] = {}
    for cohort, receipt in zip(TCGA_COHORTS, cohort_receipts, strict=True):
        canonical = receipt.get("canonical_inputs")
        if not isinstance(canonical, dict) or set(canonical) != {
            "child_manifest",
            "canonical_maf",
            "sample_axis",
            "population_manifest",
        }:
            msg = f"Provider manifest canonical receipt is malformed: {cohort}"
            raise ProviderInputError(msg)
        binding: dict[str, Any] = {"cohort": cohort}
        for name, record in canonical.items():
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or not isinstance(record["path"], str)
                or not record["path"]
                or not isinstance(record["bytes"], int)
                or isinstance(record["bytes"], bool)
                or record["bytes"] <= 0
            ):
                msg = f"Provider canonical file receipt is malformed: {cohort}/{name}"
                raise ProviderInputError(msg)
            _require_sha256(
                record["sha256"],
                label=f"provider {cohort}/{name} SHA-256",
            )
            if name == "canonical_maf":
                path = snapshot_root / "data" / "mafs" / f"{cohort}.maf"
            elif name == "sample_axis":
                path = snapshot_root / "data" / "axes" / f"{cohort}.txt"
            else:
                path = None
            binding[name] = {"file": dict(record)}
            if path is not None:
                binding[name]["path"] = path
        bindings[cohort] = binding
    return (
        ProviderContext(
            paths=paths,
            hashes=hashes,
            canonical_manifest={},
            bindings=bindings,
            authority=work_authority,
        ),
        snapshot_root,
    )


def validate_materialized_provider_cohort_input(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    cohort: str,
    full_acceptance_receipt: Mapping[str, Any],
    expected_full_acceptance_receipt_sha256: str,
    *,
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    """Validate one cohort from a separately pinned full-parent receipt."""
    if not isinstance(require_current_execution_environment, bool):
        msg = "require_current_execution_environment must be a boolean."
        raise ProviderInputError(msg)
    if cohort not in TCGA_COHORTS:
        msg = f"Unknown TCGA cohort: {cohort}"
        raise ProviderInputError(msg)
    provider_root = _absolute_unresolved(root)
    expected_hash = _require_sha256(
        expected_manifest_sha256,
        label="expected provider input manifest SHA-256",
    )
    _require_no_symlink_ancestors(provider_root, label="provider input root")
    manifest_path = provider_root / ROOT_MANIFEST_NAME
    manifest, _manifest_bytes, manifest_stat = _read_json_with_sha256_and_stat(
        manifest_path,
        expected_hash,
        label="scoped provider input manifest independent SHA-256",
    )
    if manifest_stat.st_mode & 0o222:
        msg = "Published provider input manifest is not immutable."
        raise ProviderInputError(msg)
    expected_manifest_keys = {
        "schema_version",
        "contract",
        "completed_at_utc",
        "cohorts",
        "cohort_count",
        "roots",
        "authority",
        "sources",
        "providers",
        "execution",
        "scope",
        "cohort_provider_receipts",
        "inventory",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["contract"] != PROVIDER_INPUT_CONTRACT
        or manifest["cohorts"] != list(TCGA_COHORTS)
        or manifest["cohort_count"] != len(TCGA_COHORTS)
        or not _exact_json_equal(
            manifest["roots"],
            {"cohorts": "cohorts", "mutsig": "mutsig"},
        )
    ):
        msg = "Provider input manifest has an invalid scoped root contract."
        raise ProviderInputError(msg)
    accepted = _validate_full_acceptance_receipt(
        full_acceptance_receipt,
        manifest,
        expected_hash,
        expected_full_acceptance_receipt_sha256,
    )
    work_authority, authority_stat = _read_json_with_stat(
        provider_root.joinpath(*WORK_AUTHORITY_PATH.parts),
        label="scoped provider work authority",
    )
    if authority_stat.st_mode & 0o222 or any(
        not _exact_json_equal(manifest[key], work_authority.get(key))
        for key in ("authority", "sources", "providers", "execution", "scope")
    ):
        msg = "Scoped provider manifest and immutable work authority disagree."
        raise ProviderInputError(msg)
    context, snapshot_root = _scoped_context_from_manifest(
        provider_root,
        manifest,
        work_authority,
    )
    snapshot_receipt, validated_snapshot_root = _validate_execution_snapshot(
        context,
        cohort=cohort,
        require_current_execution_environment=(require_current_execution_environment),
        validate_provider_generation_sources=False,
    )
    if validated_snapshot_root != snapshot_root:
        msg = "Scoped provider execution snapshot root changed."
        raise ProviderInputError(msg)
    _require_scoped_cohort_inventory(provider_root, cohort)
    expected_receipt = manifest["cohort_provider_receipts"][TCGA_COHORTS.index(cohort)]
    final_manifest = _stream_file_record(
        manifest_path,
        display_path=ROOT_MANIFEST_NAME,
        label="scoped provider manifest final readback",
    )
    if final_manifest["sha256"] != expected_hash:
        msg = "Provider input manifest changed during scoped validation."
        raise ProviderInputError(msg)
    selected_binding = _published_scoped_cohort_binding(
        provider_root,
        snapshot_root,
        cohort,
        expected_receipt,
        context.bindings[cohort],
    )
    _require_scoped_binding_matches_manifest_inventory(
        manifest,
        selected_binding,
        cohort,
    )
    return {
        "root": provider_root,
        "cohort": cohort,
        "binding": selected_binding,
        "provider_receipt": expected_receipt,
        "execution_snapshot": {
            "root": snapshot_receipt["root"],
            "tree_sha256": snapshot_receipt["tree_sha256"],
            "validation_scope": "selected-cohort-and-exact-shared-closure",
        },
        "full_acceptance_receipt": accepted,
        "association_outputs_opened": False,
    }


def validate_materialized_provider_input_bundle(  # noqa: PLR0913
    root: str | Path,
    expected_manifest_sha256: str,
    canonical_input_root: str | Path,
    expected_canonical_manifest_sha256: str,
    approval_manifest: str | Path,
    expected_approval_sha256: str,
    *,
    require_current_execution_environment: bool = False,
) -> dict[str, Any]:
    """Replay one independently pinned, result-blind provider-input bundle.

    The returned ``cohort_bindings`` map contains absolute provider directories and
    exact ``{path, file}`` bindings for every fit input. No association output is
    opened or admitted by the closed inventory. The result also carries the full
    acceptance receipt and its child-ready independent SHA-256 pin.
    """
    if not isinstance(require_current_execution_environment, bool):
        msg = "require_current_execution_environment must be a boolean."
        raise ProviderInputError(msg)
    provider_root = _absolute_unresolved(root)
    expected_provider_hash = _require_sha256(
        expected_manifest_sha256,
        label="expected provider input manifest SHA-256",
    )
    expected_canonical_hash = _require_sha256(
        expected_canonical_manifest_sha256,
        label="expected canonical input manifest SHA-256",
    )
    expected_approval_hash = _require_sha256(
        expected_approval_sha256,
        label="expected approval manifest SHA-256",
    )
    _require_no_symlink_ancestors(provider_root, label="provider input root")
    manifest_path = provider_root / ROOT_MANIFEST_NAME
    manifest, manifest_bytes, manifest_stat = _read_json_with_sha256_and_stat(
        manifest_path,
        expected_provider_hash,
        label="Provider input manifest independent SHA-256",
    )
    if manifest_stat.st_mode & 0o222:
        msg = "Published provider input manifest is not immutable."
        raise ProviderInputError(msg)
    if (
        set(manifest)
        != {
            "schema_version",
            "contract",
            "completed_at_utc",
            "cohorts",
            "cohort_count",
            "roots",
            "authority",
            "sources",
            "providers",
            "execution",
            "scope",
            "cohort_provider_receipts",
            "inventory",
        }
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["contract"] != PROVIDER_INPUT_CONTRACT
        or manifest["cohorts"] != list(TCGA_COHORTS)
        or manifest["cohort_count"] != len(TCGA_COHORTS)
        or not _exact_json_equal(
            manifest["roots"],
            {"cohorts": "cohorts", "mutsig": "mutsig"},
        )
    ):
        msg = "Provider input manifest has an invalid root contract or layout."
        raise ProviderInputError(msg)
    _verify_manifest_inventory(provider_root, manifest)
    work_authority_path = provider_root / WORK_AUTHORITY_PATH
    work_authority, work_authority_stat = _read_json_with_stat(
        work_authority_path,
        label="published provider work authority",
    )
    if work_authority_stat.st_mode & 0o222:
        msg = "Published provider work authority is not immutable."
        raise ProviderInputError(msg)
    if (
        set(work_authority)
        != {
            "schema_version",
            "contract",
            "intended_output_root",
            "authority",
            "sources",
            "providers",
            "execution",
            "scope",
        }
        or work_authority["schema_version"] != SCHEMA_VERSION
        or work_authority["contract"] != PROVIDER_INPUT_CONTRACT
        or not isinstance(work_authority["intended_output_root"], str)
        or not Path(work_authority["intended_output_root"]).is_absolute()
        or (
            require_current_execution_environment
            and work_authority["intended_output_root"] != provider_root.as_posix()
        )
        or any(
            not _exact_json_equal(manifest[key], work_authority[key])
            for key in ("authority", "sources", "providers", "execution", "scope")
        )
    ):
        msg = "Provider input manifest and immutable work authority disagree."
        raise ProviderInputError(msg)
    paths = _provider_paths(
        canonical_input_root,
        approval_manifest,
        provider_root,
        repo_root=None,
        require_current_execution_environment=(require_current_execution_environment),
    )
    cbase_hash, dig_hash = _validate_historical_authority_contract(
        work_authority,
        paths,
        require_current_paths=require_current_execution_environment,
    )
    hashes = IndependentHashes(
        approval=expected_approval_hash,
        canonical_input_manifest=expected_canonical_hash,
        cbase_inputs_tree=cbase_hash,
        dig_results=dig_hash,
    )
    canonical_manifest, canonical_bindings = _canonical_bundle_state(
        paths,
        hashes,
        require_current_execution_environment=require_current_execution_environment,
    )
    observed_input_authority = _input_authority_record(
        paths,
        hashes,
        canonical_manifest,
    )
    input_authority_matches = (
        _exact_json_equal(work_authority["authority"], observed_input_authority)
        if require_current_execution_environment
        else _exact_json_equal(
            _relocatable_input_authority_view(work_authority["authority"]),
            _relocatable_input_authority_view(observed_input_authority),
        )
    )
    if not input_authority_matches:
        msg = "Provider bundle does not bind the supplied D1/D2 canonical authority."
        raise ProviderInputError(msg)
    authority = work_authority
    if require_current_execution_environment:
        authority = _authority_record(paths, hashes, canonical_manifest)
        if not _exact_json_equal(authority, work_authority):
            msg = "Current provider source/runtime authority differs from the bundle."
            raise ProviderInputError(msg)
    base_context = ProviderContext(
        paths=paths,
        hashes=hashes,
        canonical_manifest=canonical_manifest,
        bindings=canonical_bindings,
        authority=authority,
    )
    context = _context_for_existing_root(base_context, provider_root)
    validated_manifest = _validate_published_root(
        context,
        provider_root,
        validate_execution_snapshot_full=True,
        require_current_execution_environment=(require_current_execution_environment),
    )
    cohort_bindings = _published_cohort_bindings(
        provider_root,
        validated_manifest,
        canonical_bindings,
    )
    _verify_manifest_inventory(provider_root, validated_manifest)
    final_manifest_bytes = _read_regular_bytes(
        manifest_path,
        label="provider input manifest final readback",
    )
    if final_manifest_bytes != manifest_bytes:
        msg = "Provider input manifest changed during validation."
        raise ProviderInputError(msg)
    full_acceptance = _full_acceptance_receipt(
        validated_manifest,
        expected_provider_hash,
    )
    return {
        "root": provider_root,
        "manifest": validated_manifest,
        "manifest_file": _provider_file_binding(
            provider_root,
            ROOT_MANIFEST_NAME,
        ),
        "roots": {
            "cohorts": provider_root / "cohorts",
            "mutsig": provider_root / "mutsig",
        },
        "cohorts": list(TCGA_COHORTS),
        "cohort_bindings": cohort_bindings,
        "full_acceptance_receipt": full_acceptance,
        "full_acceptance_receipt_sha256": full_acceptance_receipt_sha256(
            full_acceptance,
        ),
        "association_outputs_opened": False,
    }


def _publish(
    context: ProviderContext,
    *,
    output_parent_fd: int | None = None,
) -> Path:
    _validate_execution_snapshot(context, full=True)
    work_manifest_path = context.paths.work_root / ROOT_MANIFEST_NAME
    if work_manifest_path.exists():
        _validate_published_root(context, context.paths.work_root)
    else:
        manifest = _build_root_manifest(context)
        _write_json_atomic(work_manifest_path, manifest, mode=0o444)
        _fsync_directory(context.paths.work_root)
        _validate_published_root(context, context.paths.work_root)
    if output_parent_fd is None:
        _rename_exclusive(context.paths.work_root, context.paths.output_root)
        _fsync_directory(context.paths.output_root.parent)
    else:
        if not _directory_path_matches_fd(
            context.paths.output_root.parent,
            output_parent_fd,
        ):
            msg = "Provider output parent changed before publication."
            raise ProviderInputError(msg)
        _rename_exclusive_at(
            output_parent_fd,
            context.paths.work_root.name,
            output_parent_fd,
            context.paths.output_root.name,
        )
        os.fsync(output_parent_fd)
    _validate_published_root(context, context.paths.output_root)
    return context.paths.output_root


def materialize_tcga_revision_provider_inputs(  # noqa: PLR0913
    canonical_input_root: str | Path,
    approval_manifest: str | Path,
    output_root: str | Path,
    expected_approval_sha256: str,
    expected_canonical_input_sha256: str,
    expected_cbase_inputs_sha256: str,
    expected_dig_results_sha256: str,
    *,
    jobs: int | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Rebuild and atomically publish all 32 signed TCGA provider-input cohorts.

    All four expected hashes are independent CLI trust anchors.  The approval is
    intentionally limited to the D1/D2 ``materialize-final-inputs`` stage; fit-stage
    authority is neither accepted nor required here.
    """
    paths = _provider_paths(
        canonical_input_root,
        approval_manifest,
        output_root,
        repo_root=repo_root,
    )
    hashes = IndependentHashes(
        approval=_require_sha256(
            expected_approval_sha256,
            label="expected approval manifest SHA-256",
        ),
        canonical_input_manifest=_require_sha256(
            expected_canonical_input_sha256,
            label="expected canonical input manifest SHA-256",
        ),
        cbase_inputs_tree=_require_sha256(
            expected_cbase_inputs_sha256,
            label="expected CBaSE input-tree SHA-256",
        ),
        dig_results=_require_sha256(
            expected_dig_results_sha256,
            label="expected DIG results SHA-256",
        ),
    )
    selected_jobs = safe_job_cap() if jobs is None else jobs
    _validate_jobs(selected_jobs)
    with _host_execution_lease(paths.output_root):
        output_parent_fd = _open_directory_fd(
            paths.output_root.parent,
            label="provider output parent",
        )
        try:
            context = _build_context(paths, hashes)
            if os.path.lexists(paths.output_root):
                if os.path.lexists(paths.work_root):
                    msg = (
                        "Both immutable output and resumable work roots exist; "
                        "refusing ambiguous provider authority."
                    )
                    raise ProviderInputError(msg)
                if paths.output_root.is_symlink() or not paths.output_root.is_dir():
                    msg = (
                        "Provider output root is not a non-symlink directory: "
                        f"{paths.output_root}"
                    )
                    raise ProviderInputError(msg)
                _validate_published_root(
                    context,
                    paths.output_root,
                    validate_execution_snapshot_full=True,
                )
                return paths.output_root
            _initialize_work_root(context)
            _ensure_execution_snapshot(context)
            _require_allowed_work_inventory(paths.work_root)
            if (paths.work_root / ROOT_MANIFEST_NAME).exists():
                return _publish(context, output_parent_fd=output_parent_fd)
            for wave_number, wave in enumerate(
                plan_provider_waves(TCGA_COHORTS, jobs=selected_jobs),
                start=1,
            ):
                _run_wave(context, wave, wave_number=wave_number)
                if wave_number == 1:
                    validate_provider_cohort(context, CANARY_COHORT)
            # Re-run the public D1/D2 + closed-canonical-inventory API at the publish
            # boundary and independently recompute every source/provider anchor.
            final_context = _build_context(paths, hashes)
            if not _exact_json_equal(final_context.authority, context.authority):
                msg = (
                    "Authority/source/provider bytes changed during provider "
                    "generation."
                )
                raise ProviderInputError(msg)
            _validate_execution_snapshot(
                final_context,
                require_current_execution_environment=False,
            )
            return _publish(
                final_context,
                output_parent_fd=output_parent_fd,
            )
        finally:
            os.close(output_parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-input-root", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--expected-canonical-input-sha256", required=True)
    parser.add_argument("--expected-cbase-inputs-sha256", required=True)
    parser.add_argument("--expected-dig-results-sha256", required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        help=(
            "General-cohort concurrency; defaults to min(3, strictly below half "
            "logical cores). Heavy cohorts always run serially."
        ),
    )
    return parser


def main() -> None:
    """Parse independent trust anchors and publish provider inputs."""
    args = _parser().parse_args()
    result = materialize_tcga_revision_provider_inputs(
        args.canonical_input_root,
        args.approval_manifest,
        args.output_root,
        args.expected_approval_sha256,
        args.expected_canonical_input_sha256,
        args.expected_cbase_inputs_sha256,
        args.expected_dig_results_sha256,
        jobs=args.jobs,
    )
    manifest_path = result / ROOT_MANIFEST_NAME
    print(
        json.dumps(
            {
                "provider_root": result.as_posix(),
                "provider_manifest_sha256": _sha256(manifest_path),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()

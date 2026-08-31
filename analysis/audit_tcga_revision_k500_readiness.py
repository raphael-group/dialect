"""Read-only pre-authority audit for the sealed TCGA K=500 revision run.

This command consolidates the mechanical prerequisites distributed across the
released runbook and its two released operational addenda.  It deliberately does
not validate or infer human approval, create a run root, acquire an execution
lease, materialize inputs, launch fitting, or open any association-result file.

The report is observational.  A passing report is not execution authority.

Run the production audit from the repository root as
``/opt/anaconda3/envs/dialect/bin/python -I -B
analysis/audit_tcga_revision_k500_readiness.py --check-live-remote``.  The
runtime gate rejects other interpreter/script/flag combinations.  In particular,
ordinary ``python -m`` loading may create ignored ``__pycache__`` bytes before
this module can execute; those bytes are outside this narrower promise of no
scientific or output-state mutation.
"""

# ruff: noqa: EM101, EM102, PTH116, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

SCHEMA: Final = "dialect-tcga-k500-pre-authority-readiness-v2"
CONTRACT: Final = (
    "read-only-no-authority-no-results-no-scientific-or-output-state-mutation-v2"
)
SOURCE_A_COMMIT: Final = "3ad05c5ad5ddd03d7922343bafa5a2748cddb20d"
SOURCE_A_LOCAL_REF: Final = "refs/heads/codex/rebuttal-k500"
SOURCE_A_TRACKING_REF: Final = "refs/remotes/origin/codex/rebuttal-k500"
SOURCE_A_REMOTE_REF: Final = "refs/heads/codex/rebuttal-k500"
EXPECTED_ORIGIN: Final = "git@github.com:raphael-group/dialect.git"
PRODUCTION_PYTHON: Final = "/opt/anaconda3/envs/dialect/bin/python"
PRODUCTION_SCRIPT: Final = "analysis/audit_tcga_revision_k500_readiness.py"
PRODUCTION_INVOCATION: Final = (
    PRODUCTION_PYTHON,
    "-I",
    "-B",
    PRODUCTION_SCRIPT,
    "--check-live-remote",
)
SAFE_PATH: Final = (
    "/opt/anaconda3/envs/dialect/bin:/opt/homebrew/bin:/usr/local/bin:"
    "/usr/bin:/bin:/usr/sbin:/sbin"
)
TCGA_COHORTS: Final = (
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "CRAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
)
_READ_CHUNK_BYTES: Final = 1024 * 1024
_GIB: Final = 1024**3
_RAW_MULTIPLIER: Final = 10
_FIXED_DISK_SAFETY_BYTES: Final = 2 * _GIB
_MAX_JOBS: Final = 3
_PRIOR_TASK_PEAK_RSS_BYTES: Final = round(2.083 * _GIB)
_MEMORY_HEADROOM_FACTOR: Final = 1.25
_MIN_AVAILABLE_MEMORY_FRACTION: Final = 0.33
_MIN_FREE_DISK_BYTES: Final = round(7.6 * _GIB)


@dataclass(frozen=True, slots=True)
class FileExpectation:
    """One exact regular-file byte expectation."""

    path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class TreeExpectation:
    """One exact source-tree digest expectation."""

    path: str
    sha256: str
    file_count: int
    python_only: bool = False


@dataclass(frozen=True, slots=True)
class HostResourceSnapshot:
    """Aggregate-only host state used by the frozen launch formula."""

    measured_at_utc: str
    logical_cores: int
    load_average_1m: float
    total_memory_bytes: int
    available_memory_bytes: int
    free_disk_bytes: int
    cpu_source: str
    memory_source: str


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Interpreter state needed to prevent incidental repository bytecode."""

    executable: str
    argv0: str
    isolated: bool
    bytecode_writes_disabled: bool


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    """Frozen paths and digests for the released pre-authority boundary."""

    source_commit: str
    local_ref: str
    tracking_ref: str
    remote_ref: str
    expected_origin: str
    git_files: tuple[FileExpectation, ...]
    worktree_files: tuple[FileExpectation, ...]
    trees: tuple[TreeExpectation, ...]
    overlays: tuple[FileExpectation, ...]
    required_directories: tuple[str, ...]
    required_executables: tuple[str, ...]
    pre_authority_absent_paths: tuple[str, ...]
    raw_maf_root: str
    output_filesystem_probe: str
    cohorts: tuple[str, ...] = TCGA_COHORTS


PRODUCTION_POLICY: Final = AuditPolicy(
    source_commit=SOURCE_A_COMMIT,
    local_ref=SOURCE_A_LOCAL_REF,
    tracking_ref=SOURCE_A_TRACKING_REF,
    remote_ref=SOURCE_A_REMOTE_REF,
    expected_origin=EXPECTED_ORIGIN,
    git_files=(
        FileExpectation(
            "src/dialect/data/revision_approval.py",
            "7f61ef82d53564219dd3f24784cfe4fe0f1752997a73912cfb44f9b637aeb0f6",
        ),
        FileExpectation(
            "analysis/materialize_tcga_revision_inputs.py",
            "1be550c49ab9d75032714eeff5c60c3fd41b168ab927cf9e224d2b442b873ba3",
        ),
        FileExpectation(
            "analysis/materialize_tcga_revision_provider_inputs.py",
            "ae7e8fced271562361fbc355278e69915b1e37520aea96b8d337d1127f799586",
        ),
        FileExpectation(
            "analysis/run_tcga_revision_k500.py",
            "0c1cfd77d04b563a9b314515dc92f1117aa28a4342dc3cff248d58398e379823",
        ),
        FileExpectation(
            "src/dialect/data/revision_fit_policy.py",
            "a76d1f18842ed9345e87be98f3dda5ee72c5ff791dacbf7ba612bd8ba40f6ae0",
        ),
        FileExpectation(
            "src/dialect/bmr/_dig_pmf.py",
            "640159c885d1eec3cb0cfd3a0d6cd0be32b71b23f698567aba9679b51bc396a4",
        ),
        FileExpectation(
            "src/dialect/bmr/dig.py",
            "403861135669831f9a38d3205b30b0aaa4e623cf86361f8064afeb2f1c50f057",
        ),
        FileExpectation(
            "scripts/run_cohort_pipeline.sh",
            "7997a48c09cc5e214fdb66073c93c9bb081e8953d982c546e2906f6ca5ea9cb7",
        ),
        FileExpectation(
            "scripts/run_mutsig_octave.sh",
            "7c4780d17006ebc822ed0dc04e23454e58530fd8e514cd96ac00c1dcdca2951e",
        ),
        FileExpectation(
            "external/mutsig2cv_octave_dialect.patch",
            "1e0aa20921983f74a4676f077860a78e51c3eb2b525d9f5b9e324fdf2d456222",
        ),
        FileExpectation(
            "external/CBaSE/NOTICE",
            "8d20a15f829f8816a30990093c2334d6017ab8e468ec62c10a6af3210372b22d",
            2279,
        ),
    ),
    worktree_files=(
        FileExpectation(
            "external/DIGDriver/run/Pancan.genes.results.txt",
            "4402b76ed39ef603f3c6ae41f0eb840456409fde8be76f27a1647d034327788c",
            14520830,
        ),
    ),
    trees=(
        TreeExpectation(
            "src/dialect",
            "552035c4f8d913e2cd27e9d86b510033a036c2bcd74321b37d389567f4add085",
            45,
            python_only=True,
        ),
        TreeExpectation(
            "external/CBaSE",
            "a5baed588d1fc31a941ce15837ac732472e93081eec89f05a4cbcad1c9fc4389",
            33,
        ),
    ),
    overlays=(
        FileExpectation(
            "research/notes/76_current_head_post_authority_k500_runbook_candidate.md",
            "933a3e9db8b27a80de5f2b6f62204200640a8bf6fde1187e5146516af8500a60",
        ),
        FileExpectation(
            "research/notes/79_current_head_k500_runbook_release_record.md",
            "15ae1e5cf82cc9123f671ede1504ddc6151ab4e2dcce9d7ff74b2173dd5bb872",
        ),
        FileExpectation(
            "research/notes/81_current_head_k500_completion_attestation_addendum_candidate.md",
            "c263578ef2f21fd206a3c4745c2ed7fe98d3271b837c19944af6cc77b9eb96d1",
            6841,
        ),
        FileExpectation(
            "research/notes/82_k500_completion_attestation_addendum_release_record.md",
            "4faaeaca9ddfd90ef8a3701233e5789d5379b75def056e6dd236f82654631626",
        ),
        FileExpectation(
            "research/notes/85_canonical_materialization_same_uid_lease_addendum_candidate.md",
            "3e49c3a1856c491cabb8c6384de5c8dda61a5bf90326727daaf7e7a4788114f3",
            9655,
        ),
        FileExpectation(
            "research/notes/88_canonical_materialization_same_uid_lease_addendum_release_record.md",
            "dc049df313fa6a2d0e588064639dad5808e20f7101ab208aaee1645bbacfa429",
            4437,
        ),
        FileExpectation(
            "research/notes/materialize_tcga_revision_inputs_with_lease.py",
            "321c60cb215b698215d2cd4da030926f218cd166310e348e9eba57172e9083af",
            18563,
        ),
        FileExpectation(
            "research/notes/attest_k500_completion.py",
            "238a7131a0e3aeb41928939840476d333d5ccfb19a081c19decdb6ea2a4d9de2",
            57923,
        ),
    ),
    required_directories=(
        "data/mafs_pancan",
        "output/tcga_revision_population_2026-08-28_v1",
        "research/sources/datahub-proof.git",
        "external/CBaSE",
    ),
    required_executables=(
        "/opt/anaconda3/envs/dialect/bin/python",
        "/opt/homebrew/bin/octave",
        "/usr/bin/java",
        "/usr/bin/git",
    ),
    pre_authority_absent_paths=(
        "output/tcga_revision_materialize_approval_2026-08-29_v12",
        "output/tcga_revision_canonical_2026-08-29_v13",
        "output/tcga_revision_providers_2026-08-29_v13",
        "output/tcga_revision_k500_2026-08-30_v1",
    ),
    raw_maf_root="data/mafs_pancan",
    output_filesystem_probe="output/tcga_revision_canonical_2026-08-29_v13",
)


class ReadinessAuditError(RuntimeError):
    """Raised when the auditor itself cannot produce trustworthy evidence."""


class _WorkspacePathAbsentError(FileNotFoundError):
    """Raised internally when a validated workspace entry is absent."""


GitRunner = Callable[[Path, Sequence[str]], bytes]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_workspace_path(value: str, *, label: str) -> PurePosixPath:
    """Return one normalized, non-escaping, repository-relative path."""
    if not value or "\x00" in value or "\\" in value:
        raise ReadinessAuditError(
            f"{label} is not a canonical workspace path: {value!r}",
        )
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {".", ".."} for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise ReadinessAuditError(
            f"{label} is not a canonical workspace path: {value!r}",
        )
    return candidate


def _validate_policy_workspace_paths(policy: AuditPolicy) -> None:
    """Reject malformed workspace paths before any Git or filesystem read."""
    path_groups: tuple[tuple[str, Sequence[str]], ...] = (
        ("git file", tuple(item.path for item in policy.git_files)),
        ("worktree file", tuple(item.path for item in policy.worktree_files)),
        ("tree", tuple(item.path for item in policy.trees)),
        ("released overlay", tuple(item.path for item in policy.overlays)),
        ("required directory", policy.required_directories),
        ("pre-authority absent path", policy.pre_authority_absent_paths),
        ("raw MAF root", (policy.raw_maf_root,)),
        ("output filesystem probe", (policy.output_filesystem_probe,)),
    )
    for label, paths in path_groups:
        for value in paths:
            _canonical_workspace_path(value, label=label)
    if not policy.cohorts or len(set(policy.cohorts)) != len(policy.cohorts):
        raise ReadinessAuditError("cohort family must be non-empty and unique")
    for cohort in policy.cohorts:
        if re.fullmatch(r"[A-Z0-9]+", cohort) is None:
            raise ReadinessAuditError(f"cohort is not a safe path stem: {cohort!r}")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_root_descriptor(root: Path, *, label: str) -> int:
    absolute = root.absolute()
    try:
        observed = os.stat(absolute, follow_symlinks=False)
        descriptor = os.open(absolute, _directory_open_flags())
    except OSError as error:
        raise ReadinessAuditError(
            f"{label} root cannot be opened: {absolute}",
        ) from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino)
    ):
        os.close(descriptor)
        raise ReadinessAuditError(f"{label} root is not a pinned directory: {absolute}")
    return descriptor


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    try:
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as error:
        raise _WorkspacePathAbsentError(label) from error
    except OSError as error:
        raise ReadinessAuditError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ReadinessAuditError(f"{label} is not a real directory")
    try:
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ReadinessAuditError(f"{label} cannot be opened") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino)
    ):
        os.close(descriptor)
        raise ReadinessAuditError(f"{label} changed while opened")
    return descriptor, opened


def _open_workspace_parent(
    root: Path,
    relative: str,
    *,
    label: str,
) -> tuple[int, str, PurePosixPath]:
    canonical = _canonical_workspace_path(relative, label=label)
    descriptor = _open_root_descriptor(root, label=label)
    try:
        traversed = PurePosixPath()
        for part in canonical.parts[:-1]:
            traversed /= part
            child, _opened = _open_child_directory(
                descriptor,
                part,
                label=f"{label} ancestor {traversed.as_posix()}",
            )
            os.close(descriptor)
            descriptor = child
        return descriptor, canonical.parts[-1], canonical
    except (OSError, ReadinessAuditError):
        os.close(descriptor)
        raise


def _open_workspace_directory(
    root: Path,
    relative: str,
    *,
    label: str,
) -> tuple[int, int, str, PurePosixPath, os.stat_result]:
    parent, name, canonical = _open_workspace_parent(root, relative, label=label)
    try:
        descriptor, opened = _open_child_directory(
            parent,
            name,
            label=f"{label} {canonical.as_posix()}",
        )
    except (OSError, ReadinessAuditError):
        os.close(parent)
        raise
    return descriptor, parent, name, canonical, opened


def _require_directory_entry_stable(
    descriptor: int,
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
    *,
    label: str,
) -> None:
    after = os.fstat(descriptor)
    try:
        after_path = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ReadinessAuditError(f"{label} changed while read") from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, key) != getattr(after, key) for key in identity_fields)
        or (after.st_dev, after.st_ino) != (after_path.st_dev, after_path.st_ino)
    ):
        raise ReadinessAuditError(f"{label} changed while read")


def _stable_file_record(root: Path, relative: str) -> dict[str, object]:
    """Hash one stable, one-link, regular, non-symlink file."""
    parent, name, canonical = _open_workspace_parent(
        root,
        relative,
        label="required file",
    )
    try:
        try:
            before_path = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except OSError as error:
            raise ReadinessAuditError(
                f"required file is unavailable: {canonical.as_posix()}",
            ) from error
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=parent)
        except OSError as error:
            raise ReadinessAuditError(
                f"required file cannot be opened: {canonical.as_posix()}",
            ) from error
    except ReadinessAuditError:
        os.close(parent)
        raise
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise ReadinessAuditError(
                "required file is not a stable one-link regular file: "
                f"{canonical.as_posix()}",
            )
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            byte_count += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_path = os.stat(name, dir_fd=parent, follow_symlinks=False)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(getattr(before, key) != getattr(after, key) for key in identity_fields)
            or (after.st_dev, after.st_ino) != (after_path.st_dev, after_path.st_ino)
            or byte_count != before.st_size
        ):
            raise ReadinessAuditError(
                f"required file changed while read: {canonical.as_posix()}",
            )
    finally:
        os.close(descriptor)
        os.close(parent)
    return {
        "bytes": byte_count,
        "path": (root.absolute() / canonical.as_posix()).as_posix(),
        "sha256": digest.hexdigest(),
    }


def _file_check(root: Path, expected: FileExpectation) -> dict[str, object]:
    try:
        observed = _stable_file_record(root, expected.path)
    except (ReadinessAuditError, _WorkspacePathAbsentError) as error:
        return {
            "passed": False,
            "path": expected.path,
            "reason": str(error),
            "expected_sha256": expected.sha256,
            "expected_bytes": expected.size_bytes,
        }
    passed = observed["sha256"] == expected.sha256 and (
        expected.size_bytes is None or observed["bytes"] == expected.size_bytes
    )
    return {
        "passed": passed,
        "path": expected.path,
        "expected_sha256": expected.sha256,
        "expected_bytes": expected.size_bytes,
        "observed_sha256": observed["sha256"],
        "observed_bytes": observed["bytes"],
    }


def _tree_record(
    root: Path,
    relative_root: str,
    *,
    python_only: bool,
) -> dict[str, object]:
    """Reproduce the provider's u64be-path-mode-content-v1 tree digest."""
    try:
        (
            root_descriptor,
            root_parent,
            root_name,
            canonical_root,
            opened_root,
        ) = _open_workspace_directory(
            root,
            relative_root,
            label="required tree",
        )
    except _WorkspacePathAbsentError as error:
        raise ReadinessAuditError(
            f"required tree is unavailable: {relative_root}",
        ) from error
    digest = hashlib.sha256()
    count = 0

    def walk(directory: int, prefix: PurePosixPath) -> None:
        nonlocal count
        before_directory = os.fstat(directory)
        try:
            names = sorted(os.listdir(directory))
        except OSError as error:
            raise ReadinessAuditError(
                f"required tree cannot be listed: {prefix.as_posix() or '.'}",
            ) from error
        for name in names:
            if name in {".", ".."} or "/" in name or "\x00" in name:
                raise ReadinessAuditError(
                    f"required tree contains invalid entry name: {name!r}",
                )
            relative = prefix / name
            try:
                observed = os.stat(
                    name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ReadinessAuditError(
                    f"required tree entry cannot be inspected: {relative}",
                ) from error
            if stat.S_ISLNK(observed.st_mode):
                raise ReadinessAuditError(f"required tree contains symlink: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                if name == "__pycache__":
                    continue
                child, opened_child = _open_child_directory(
                    directory,
                    name,
                    label=f"required tree directory {relative}",
                )
                try:
                    walk(child, relative)
                    _require_directory_entry_stable(
                        child,
                        directory,
                        name,
                        opened_child,
                        label=f"required tree directory {relative}",
                    )
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise ReadinessAuditError(
                    f"required tree contains non-private file: {relative}",
                )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                descriptor = os.open(name, flags, dir_fd=directory)
            except OSError as error:
                raise ReadinessAuditError(
                    f"required tree file cannot be opened: {relative}",
                ) from error
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    raise ReadinessAuditError(
                        f"required tree file identity changed: {relative}",
                    )
                bytes_read = 0
                if not python_only or relative.suffix == ".py":
                    encoded = relative.as_posix().encode()
                    normalized_mode = (
                        0o500 if stat.S_IMODE(opened.st_mode) & 0o111 else 0o400
                    )
                    digest.update(struct.pack(">Q", len(encoded)))
                    digest.update(encoded)
                    digest.update(struct.pack(">Q", normalized_mode))
                    digest.update(struct.pack(">Q", opened.st_size))
                    while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
                        bytes_read += len(chunk)
                        digest.update(chunk)
                after = os.fstat(descriptor)
                after_path = os.stat(
                    name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
                if (
                    opened.st_dev != after.st_dev
                    or opened.st_ino != after.st_ino
                    or opened.st_mode != after.st_mode
                    or opened.st_nlink != after.st_nlink
                    or opened.st_size != after.st_size
                    or opened.st_mtime_ns != after.st_mtime_ns
                    or opened.st_ctime_ns != after.st_ctime_ns
                    or (after.st_dev, after.st_ino)
                    != (after_path.st_dev, after_path.st_ino)
                    or (
                        (not python_only or relative.suffix == ".py")
                        and bytes_read != opened.st_size
                    )
                ):
                    raise ReadinessAuditError(
                        f"required tree file changed while read: {relative}",
                    )
            finally:
                os.close(descriptor)
            if not python_only or relative.suffix == ".py":
                count += 1

        after_directory = os.fstat(directory)
        try:
            after_names = sorted(os.listdir(directory))
        except OSError as error:
            raise ReadinessAuditError(
                f"required tree cannot be relisted: {prefix.as_posix() or '.'}",
            ) from error
        directory_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if names != after_names or any(
            getattr(before_directory, key) != getattr(after_directory, key)
            for key in directory_fields
        ):
            display = prefix.as_posix() or "."
            raise ReadinessAuditError(
                f"required tree directory changed while read: {display}",
            )

    try:
        walk(root_descriptor, PurePosixPath())
        _require_directory_entry_stable(
            root_descriptor,
            root_parent,
            root_name,
            opened_root,
            label=f"required tree {canonical_root.as_posix()}",
        )
    finally:
        os.close(root_descriptor)
        os.close(root_parent)
    if count == 0:
        raise ReadinessAuditError(
            f"required tree contains no hashable files: {canonical_root.as_posix()}",
        )
    return {
        "contract": "u64be-path-mode-content-v1",
        "file_count": count,
        "sha256": digest.hexdigest(),
    }


def _tree_check(root: Path, expected: TreeExpectation) -> dict[str, object]:
    try:
        observed = _tree_record(
            root,
            expected.path,
            python_only=expected.python_only,
        )
    except (OSError, ReadinessAuditError) as error:
        return {
            "passed": False,
            "path": expected.path,
            "reason": str(error),
            "expected_sha256": expected.sha256,
            "expected_file_count": expected.file_count,
        }
    passed = (
        observed["sha256"] == expected.sha256
        and observed["file_count"] == expected.file_count
    )
    return {
        "passed": passed,
        "path": expected.path,
        "expected_sha256": expected.sha256,
        "expected_file_count": expected.file_count,
        "observed_sha256": observed["sha256"],
        "observed_file_count": observed["file_count"],
        "contract": observed["contract"],
    }


def _default_git_runner(repo_root: Path, arguments: Sequence[str]) -> bytes:
    git = Path("/usr/bin/git")
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SAFE_PATH,
    }
    try:
        return subprocess.run(
            [git.as_posix(), "--no-pager", *arguments],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ReadinessAuditError(
            f"read-only Git command failed: {' '.join(arguments)}",
        ) from error


def _git_text(git_runner: GitRunner, root: Path, arguments: Sequence[str]) -> str:
    try:
        return git_runner(root, arguments).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ReadinessAuditError("Git metadata was not UTF-8") from error


def _audit_git(
    root: Path,
    policy: AuditPolicy,
    *,
    check_live_remote: bool,
    git_runner: GitRunner,
) -> dict[str, object]:
    active_head = _git_text(git_runner, root, ("rev-parse", "HEAD"))
    local = _git_text(git_runner, root, ("rev-parse", policy.local_ref))
    tracking = _git_text(git_runner, root, ("rev-parse", policy.tracking_ref))
    origin = _git_text(git_runner, root, ("remote", "get-url", "origin"))
    status = _git_text(
        git_runner,
        root,
        ("status", "--short", "--untracked-files=all"),
    ).splitlines()
    commit_files = []
    for expected in policy.git_files:
        payload = git_runner(
            root,
            ("show", f"{policy.source_commit}:{expected.path}"),
        )
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        commit_files.append(
            {
                "passed": observed_sha256 == expected.sha256,
                "path": expected.path,
                "expected_sha256": expected.sha256,
                "observed_sha256": observed_sha256,
            },
        )
    remote_record: dict[str, object]
    if check_live_remote:
        remote_output = _git_text(
            git_runner,
            root,
            ("ls-remote", "--exit-code", "origin", policy.remote_ref),
        )
        expected_line = f"{policy.source_commit}\t{policy.remote_ref}"
        remote_record = {
            "checked": True,
            "passed": remote_output == expected_line,
            "expected": expected_line,
            "observed": remote_output,
        }
    else:
        remote_record = {
            "checked": False,
            "passed": False,
            "reason": "live remote check omitted; rerun with --check-live-remote",
        }
    source_a_preserved = (
        local == policy.source_commit
        and tracking == policy.source_commit
        and origin == policy.expected_origin
        and bool(remote_record["passed"])
        and all(bool(record["passed"]) for record in commit_files)
    )
    return {
        "passed": (
            source_a_preserved
            and active_head == policy.source_commit
            and not status
        ),
        "preservation_passed": source_a_preserved,
        "active_launch_tree_passed": (
            active_head == policy.source_commit and not status
        ),
        "source_a_preserved": source_a_preserved,
        "active_source_a": active_head == policy.source_commit,
        "active_head": active_head,
        "clean_worktree": not status,
        "dirty_entry_count": len(status),
        "expected_source_a": policy.source_commit,
        "local_ref": {"name": policy.local_ref, "observed": local},
        "tracking_ref": {"name": policy.tracking_ref, "observed": tracking},
        "origin": {"expected": policy.expected_origin, "observed": origin},
        "live_remote": remote_record,
        "source_a_commit_files": commit_files,
    }


def _path_state(root: Path, relative: str) -> dict[str, object]:
    """Inspect only one exact directory entry; never traverse it."""
    canonical = _canonical_workspace_path(relative, label="workspace entry")
    logical = (root.absolute() / canonical.as_posix()).as_posix()
    try:
        parent, name, _canonical = _open_workspace_parent(
            root,
            relative,
            label="workspace entry",
        )
    except _WorkspacePathAbsentError:
        return {"exists": False, "kind": "absent", "path": logical}
    try:
        try:
            observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return {"exists": False, "kind": "absent", "path": logical}
    finally:
        os.close(parent)
    if stat.S_ISDIR(observed.st_mode):
        kind = "directory"
    elif stat.S_ISREG(observed.st_mode):
        kind = "regular-file"
    elif stat.S_ISLNK(observed.st_mode):
        kind = "symlink"
    else:
        kind = "special"
    return {"exists": True, "kind": kind, "path": logical}


def _required_directory_check(root: Path, relative: str) -> dict[str, object]:
    try:
        state = _path_state(root, relative)
    except ReadinessAuditError as error:
        return {
            "exists": None,
            "kind": "untrusted",
            "passed": False,
            "path": relative,
            "reason": str(error),
        }
    return {**state, "passed": state["kind"] == "directory"}


def _required_executable_check(path: str) -> dict[str, object]:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        observed = resolved.stat()
    except OSError as error:
        return {"passed": False, "path": path, "reason": str(error)}
    return {
        "passed": stat.S_ISREG(observed.st_mode) and os.access(resolved, os.X_OK),
        "path": path,
        "resolved": resolved.as_posix(),
    }


def _raw_maf_storage_check(
    root: Path,
    policy: AuditPolicy,
    *,
    free_disk_bytes: int,
) -> dict[str, object]:
    expected_names = {f"{cohort}.maf" for cohort in policy.cohorts}
    try:
        (
            raw_descriptor,
            raw_parent,
            raw_name,
            canonical_root,
            opened_root,
        ) = _open_workspace_directory(
            root,
            policy.raw_maf_root,
            label="raw MAF root",
        )
    except (ReadinessAuditError, _WorkspacePathAbsentError) as error:
        return {
            "passed": False,
            "path": policy.raw_maf_root,
            "reason": str(error),
            "contents_opened": False,
        }
    records: list[dict[str, object]] = []
    before_files: dict[str, os.stat_result] = {}
    total = 0
    stable_snapshot = True
    stability_reasons: list[str] = []
    try:
        try:
            all_names = sorted(os.listdir(raw_descriptor))  # noqa: PTH208
        except OSError as error:
            return {
                "passed": False,
                "path": policy.raw_maf_root,
                "reason": f"raw MAF root cannot be listed: {error}",
                "contents_opened": False,
            }
        observed_names = {name for name in all_names if name.endswith(".maf")}
        for name in sorted(expected_names):
            try:
                observed = os.stat(
                    name,
                    dir_fd=raw_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                records.append(
                    {"name": name, "passed": False, "reason": str(error)},
                )
                continue
            valid = stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1
            records.append(
                {"name": name, "passed": valid, "bytes": observed.st_size},
            )
            if valid:
                total += observed.st_size
                before_files[name] = observed

        try:
            after_names = sorted(os.listdir(raw_descriptor))  # noqa: PTH208
        except OSError as error:
            stable_snapshot = False
            stability_reasons.append(f"raw MAF root cannot be relisted: {error}")
            after_names = []
        if all_names != after_names:
            stable_snapshot = False
            stability_reasons.append("raw MAF directory entries changed during audit")
        file_identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        records_by_name = {str(record["name"]): record for record in records}
        for name, before in before_files.items():
            try:
                after = os.stat(
                    name,
                    dir_fd=raw_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                stable_snapshot = False
                records_by_name[name]["passed"] = False
                records_by_name[name]["reason"] = str(error)
                continue
            if any(
                getattr(before, key) != getattr(after, key)
                for key in file_identity_fields
            ):
                stable_snapshot = False
                records_by_name[name]["passed"] = False
                records_by_name[name]["reason"] = "metadata changed during audit"
        try:
            _require_directory_entry_stable(
                raw_descriptor,
                raw_parent,
                raw_name,
                opened_root,
                label=f"raw MAF root {canonical_root.as_posix()}",
            )
        except ReadinessAuditError as error:
            stable_snapshot = False
            stability_reasons.append(str(error))
    finally:
        os.close(raw_descriptor)
        os.close(raw_parent)
    required = total * _RAW_MULTIPLIER + _FIXED_DISK_SAFETY_BYTES
    names_match = observed_names == expected_names
    deficit = max(0, required - free_disk_bytes)
    passed = (
        names_match
        and stable_snapshot
        and len(records) == len(expected_names)
        and all(bool(record["passed"]) for record in records)
        and free_disk_bytes >= required
    )
    return {
        "passed": passed,
        "path": policy.raw_maf_root,
        "cohort_count": len(records),
        "names_match_exact_family": names_match,
        "metadata_snapshot_stable": stable_snapshot,
        "stability_reasons": stability_reasons,
        "raw_bytes": total,
        "raw_multiplier": _RAW_MULTIPLIER,
        "fixed_safety_bytes": _FIXED_DISK_SAFETY_BYTES,
        "required_free_bytes": required,
        "observed_free_bytes": free_disk_bytes,
        "deficit_bytes": deficit,
        "files": records,
        "contents_opened": False,
    }


def _parse_darwin_memory_pressure(output: str) -> tuple[int, int]:
    total_match = re.search(r"The system has (\d+) ", output)
    free_match = re.search(r"System-wide memory free percentage: (\d+)%", output)
    if total_match is None or free_match is None:
        raise ReadinessAuditError("macOS aggregate memory readback is not parseable")
    total = int(total_match.group(1))
    free_percent = int(free_match.group(1))
    if total <= 0 or not 0 <= free_percent <= 100:
        raise ReadinessAuditError("macOS aggregate memory readback is invalid")
    return total, total * free_percent // 100


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
        raise ReadinessAuditError(
            "Linux aggregate memory readback is incomplete",
        ) from error
    if total <= 0 or not 0 <= available <= total:
        raise ReadinessAuditError("Linux aggregate memory readback is invalid")
    return total, available


def _workspace_free_disk_bytes(root: Path, relative: str) -> int:
    """Read free bytes from a descriptor-pinned nearest workspace parent."""
    canonical = _canonical_workspace_path(
        relative,
        label="output filesystem probe",
    )
    root_descriptor = _open_root_descriptor(
        root,
        label="output filesystem probe",
    )
    descriptors = [root_descriptor]
    links: list[tuple[int, str, int, os.stat_result, str]] = []
    current = root_descriptor
    traversed = PurePosixPath()
    try:
        for part in canonical.parts:
            traversed /= part
            try:
                observed = os.stat(
                    part,
                    dir_fd=current,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                break
            except OSError as error:
                raise ReadinessAuditError(
                    f"output filesystem probe cannot inspect {traversed}",
                ) from error
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ReadinessAuditError(
                    f"output filesystem probe is not a real directory: {traversed}",
                )
            child, opened = _open_child_directory(
                current,
                part,
                label=f"output filesystem probe {traversed}",
            )
            descriptors.append(child)
            links.append((current, part, child, opened, traversed.as_posix()))
            current = child
        filesystem = os.fstatvfs(current)
        free_bytes = filesystem.f_bavail * filesystem.f_frsize
        for parent, name, child, opened, logical in reversed(links):
            _require_directory_entry_stable(
                child,
                parent,
                name,
                opened,
                label=f"output filesystem probe {logical}",
            )
        return free_bytes
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_host_resources(
    repo_root: Path,
    filesystem_probe: str,
) -> HostResourceSnapshot:
    """Read aggregate host resources without inspecting another process."""
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        )
        total, available = _parse_darwin_memory_pressure(completed.stdout)
        memory_source = "/usr/bin/memory_pressure -Q"
    elif sys.platform.startswith("linux"):
        total, available = _parse_linux_meminfo(
            Path("/proc/meminfo").read_text(encoding="utf-8"),
        )
        memory_source = "/proc/meminfo MemAvailable"
    else:
        raise ReadinessAuditError(f"unsupported resource platform: {sys.platform}")
    return HostResourceSnapshot(
        measured_at_utc=_utc_now(),
        logical_cores=os.cpu_count() or 0,
        load_average_1m=float(os.getloadavg()[0]),
        total_memory_bytes=total,
        available_memory_bytes=available,
        free_disk_bytes=_workspace_free_disk_bytes(repo_root, filesystem_probe),
        cpu_source="os.getloadavg()[0]",
        memory_source=memory_source,
    )


def evaluate_host_resource_gate(
    snapshot: HostResourceSnapshot,
    *,
    jobs: int,
) -> dict[str, object]:
    """Reproduce the runner/provider aggregate launch gate."""
    reasons: list[str] = []
    safe_cap = max(1, min(_MAX_JOBS, (max(snapshot.logical_cores, 1) - 1) // 2))
    if jobs <= 0 or jobs > safe_cap:
        reasons.append(f"jobs={jobs} exceeds safe live cap={safe_cap}")
    load_valid = (
        math.isfinite(snapshot.load_average_1m)
        and snapshot.load_average_1m >= 0
    )
    projected = snapshot.load_average_1m + jobs if load_valid else None
    half_cores = snapshot.logical_cores / 2
    if projected is None or projected >= half_cores:
        reasons.append("loadavg_1m plus planned jobs is not below half the host")
    memory_valid = snapshot.total_memory_bytes > 0 and (
        0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes
    )
    if not memory_valid:
        reasons.append("available memory is outside the physical-memory range")
    required_by_tasks = math.ceil(
        jobs * _PRIOR_TASK_PEAK_RSS_BYTES * _MEMORY_HEADROOM_FACTOR,
    )
    required_by_fraction = math.ceil(
        max(snapshot.total_memory_bytes, 0) * _MIN_AVAILABLE_MEMORY_FRACTION,
    )
    required_available = max(required_by_tasks, required_by_fraction)
    if memory_valid and snapshot.available_memory_bytes < required_available:
        reasons.append("available memory is below the aggregate headroom gate")
    if snapshot.free_disk_bytes < _MIN_FREE_DISK_BYTES:
        reasons.append("free disk is below the 2x historical-output gate")
    try:
        measured = datetime.fromisoformat(snapshot.measured_at_utc)
        timestamp_valid = (
            measured.tzinfo is not None
            and measured.utcoffset() == UTC.utcoffset(None)
        )
    except (TypeError, ValueError):
        timestamp_valid = False
    if not timestamp_valid or not snapshot.cpu_source or not snapshot.memory_source:
        reasons.append("resource readback provenance is incomplete")
    return {
        "passed": not reasons,
        "jobs": jobs,
        "safe_job_cap": safe_cap,
        "strict_half_core_limit": half_cores,
        "projected_load_with_planned_jobs": projected,
        "required_available_memory_bytes": required_available,
        "required_by_prior_rss_bytes": required_by_tasks,
        "required_by_fraction_bytes": required_by_fraction,
        "minimum_free_disk_bytes": _MIN_FREE_DISK_BYTES,
        "reasons": reasons,
    }


def _resource_snapshot_record(
    snapshot: HostResourceSnapshot,
) -> dict[str, object]:
    """Encode even a nonfinite load readback as strict canonical JSON."""
    record: dict[str, object] = asdict(snapshot)
    load = snapshot.load_average_1m
    if math.isnan(load):
        record["load_average_1m"] = None
        record["load_average_1m_state"] = "nan"
    elif load == math.inf:
        record["load_average_1m"] = None
        record["load_average_1m_state"] = "positive-infinity"
    elif load == -math.inf:
        record["load_average_1m"] = None
        record["load_average_1m_state"] = "negative-infinity"
    else:
        record["load_average_1m_state"] = "finite"
    return record


def _live_runtime_snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        executable=sys.executable,
        argv0=sys.argv[0],
        isolated=bool(sys.flags.isolated),
        bytecode_writes_disabled=bool(sys.dont_write_bytecode),
    )


def _runtime_boundary_record(
    root: Path,
    snapshot: RuntimeSnapshot,
) -> dict[str, object]:
    expected_python = Path(PRODUCTION_PYTHON).resolve(strict=False)
    observed_python = Path(snapshot.executable).resolve(strict=False)
    expected_script = (root / PRODUCTION_SCRIPT).resolve(strict=False)
    observed_script = Path(snapshot.argv0).resolve(strict=False)
    interpreter_matches = observed_python == expected_python
    script_matches = observed_script == expected_script
    passed = (
        interpreter_matches
        and script_matches
        and snapshot.isolated
        and snapshot.bytecode_writes_disabled
    )
    return {
        "passed": passed,
        "expected_invocation": list(PRODUCTION_INVOCATION),
        "observed_interpreter": observed_python.as_posix(),
        "expected_interpreter": expected_python.as_posix(),
        "interpreter_matches": interpreter_matches,
        "observed_script": observed_script.as_posix(),
        "expected_script": expected_script.as_posix(),
        "script_matches": script_matches,
        "isolated_mode": snapshot.isolated,
        "bytecode_writes_disabled": snapshot.bytecode_writes_disabled,
        "noncompliant_invocation_may_create_ignored_python_bytecode": True,
        "scope": "runtime hygiene only; never scientific or execution authority",
        "authorizes_execution": False,
    }


def audit_readiness(  # noqa: PLR0913
    repo_root: Path,
    *,
    policy: AuditPolicy = PRODUCTION_POLICY,
    check_live_remote: bool = False,
    git_runner: GitRunner = _default_git_runner,
    resource_snapshot: HostResourceSnapshot | None = None,
    runtime_snapshot: RuntimeSnapshot | None = None,
) -> dict[str, object]:
    """Build one read-only, result-blind readiness report."""
    root = repo_root.resolve(strict=True)
    _validate_policy_workspace_paths(policy)
    runtime = _runtime_boundary_record(
        root,
        runtime_snapshot or _live_runtime_snapshot(),
    )
    git = _audit_git(
        root,
        policy,
        check_live_remote=check_live_remote,
        git_runner=git_runner,
    )
    overlays = [_file_check(root, item) for item in policy.overlays]
    external_files = [_file_check(root, item) for item in policy.worktree_files]
    worktree_files = [
        _file_check(root, item) for item in (*policy.git_files, *policy.worktree_files)
    ]
    trees = [_tree_check(root, item) for item in policy.trees]
    directories = [
        _required_directory_check(root, relative)
        for relative in policy.required_directories
    ]
    executables = [
        _required_executable_check(path)
        for path in policy.required_executables
    ]
    absent_paths = []
    for relative in policy.pre_authority_absent_paths:
        try:
            state = _path_state(root, relative)
        except ReadinessAuditError as error:
            state = {
                "exists": None,
                "kind": "untrusted",
                "path": relative,
                "reason": str(error),
            }
        absent_paths.append({**state, "passed": state["exists"] is False})
    snapshot = resource_snapshot or read_host_resources(
        root,
        policy.output_filesystem_probe,
    )
    host_gates = {
        str(jobs): evaluate_host_resource_gate(snapshot, jobs=jobs)
        for jobs in (1, 3)
    }
    storage = _raw_maf_storage_check(
        root,
        policy,
        free_disk_bytes=snapshot.free_disk_bytes,
    )
    overlay_passed = all(bool(record["passed"]) for record in overlays)
    external_files_passed = all(
        bool(record["passed"]) for record in external_files
    )
    active_launch_tree_passed = (
        bool(git["passed"])
        and all(bool(record["passed"]) for record in worktree_files)
        and all(bool(record["passed"]) for record in trees)
    )
    prerequisite_passed = (
        all(bool(record["passed"]) for record in directories)
        and all(bool(record["passed"]) for record in executables)
        and all(bool(record["passed"]) for record in absent_paths)
    )
    resource_passed = all(bool(record["passed"]) for record in host_gates.values())
    mechanical_passed = (
        overlay_passed
        and bool(git["preservation_passed"])
        and bool(git["clean_worktree"])
        and bool(runtime["passed"])
        and external_files_passed
        and prerequisite_passed
        and resource_passed
        and bool(storage["passed"])
    )
    return {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "measured_at_utc": _utc_now(),
        "repo_root": root.as_posix(),
        "mechanical_preconditions": {
            "passed": mechanical_passed,
            "released_overlay_integrity": overlay_passed,
            "source_a_preserved": bool(git["preservation_passed"]),
            "current_release_b_worktree_clean": bool(git["clean_worktree"]),
            "mutation_free_runtime": bool(runtime["passed"]),
            "external_dependencies_present": external_files_passed,
            "required_inputs_and_absent_destinations": prerequisite_passed,
            "host_resource_gates": resource_passed,
            "canonical_storage_gate": bool(storage["passed"]),
        },
        "launch_tree_transition": {
            "passed": active_launch_tree_passed,
            "required_after_authority_and_storage": True,
            "required_source_commit": policy.source_commit,
            "status": (
                "active-clean-source-a"
                if active_launch_tree_passed
                else "deferred-source-a-checkout-and-fresh-readback-required"
            ),
            "authorizes_execution": False,
        },
        "git": git,
        "runtime_boundary": runtime,
        "released_overlays": overlays,
        "external_dependencies": external_files,
        "active_worktree_files": worktree_files,
        "active_worktree_trees": trees,
        "required_directories": directories,
        "required_executables": executables,
        "pre_authority_absent_paths": absent_paths,
        "host_resources": {
            "snapshot": _resource_snapshot_record(snapshot),
            "evaluations": host_gates,
            "volatile_not_reusable": True,
            "fresh_recheck_required_before_every_stage_wave_and_task": True,
        },
        "canonical_storage": {
            **storage,
            "wrapper_recheck_under_same_uid_lease_required": True,
        },
        "human_authority": {
            "evaluated": False,
            "status": "external-unverified",
            "required_first_party_approvers": [
                "Benjamin J. Raphael",
                "Uthsav Chitra",
            ],
            "authorizes_execution": False,
        },
        "scientific_scope": {
            "result_rows_opened": False,
            "fitting_launched": False,
            "roots_created_or_modified": False,
            "files_deleted": False,
            "authorizes_materialization": False,
            "authorizes_fitting": False,
            "authorizes_inspection": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--check-live-remote",
        action="store_true",
        help="Read the exact source-A ref from origin; required for a passing report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print canonical JSON and fail unless every mechanical prerequisite passes."""
    arguments = _parser().parse_args(argv)
    try:
        report = audit_readiness(
            arguments.repo_root,
            check_live_remote=arguments.check_live_remote,
        )
    except (OSError, ReadinessAuditError, subprocess.SubprocessError) as error:
        runtime = _runtime_boundary_record(
            arguments.repo_root.absolute(),
            _live_runtime_snapshot(),
        )
        report = {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "measured_at_utc": _utc_now(),
            "mechanical_preconditions": {"passed": False},
            "audit_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "runtime_boundary": runtime,
            "human_authority": {
                "evaluated": False,
                "status": "external-unverified",
                "authorizes_execution": False,
            },
            "scientific_scope": {
                "result_rows_opened": False,
                "fitting_launched": False,
                "roots_created_or_modified": False,
                "files_deleted": False,
                "authorizes_materialization": False,
                "authorizes_fitting": False,
                "authorizes_inspection": False,
            },
        }
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["mechanical_preconditions"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

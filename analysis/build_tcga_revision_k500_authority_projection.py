"""Build and verify the result-blind K=500 release-authority projection.

This module consumes only three independently anchored JSON metadata files from
the completed K=500 run.  It never opens a scientific result CSV or traverses a
run/output root.  The immutable projection lets release tooling consume the six
approved authority digests without reopening the completed run.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import secrets
import selectors
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable

PROJECTION_SCHEMA = "dialect-tcga-k500-authority-projection-v1"
ATTESTATION_SCHEMA = "1.1.0"
ATTESTATION_TYPE = "tcga-revision-k500-root-completion"
SEALED_COMPLETION_SCHEMA = "dialect-tcga-k500-sealed-completion-v1"
SEALED_COMPLETION_CONTRACT = "metadata-hash-only-whole-grid-write-once-v1"
PROVIDER_INPUT_CONTRACT = "signed-tcga-provider-input-rebuild-v1"
PROVIDER_FULL_ACCEPTANCE_CONTRACT = "provider-full-acceptance-receipt-v1"
PROVIDER_SCHEMA_VERSION = "1.0.0"
PROVIDER_TREE_HASH_CONTRACT = "u64be-path-mode-content-v1"

ATTESTATION_SCOPE_RUN = "complete frozen TCGA K=500 cohort/background grid"
ATTESTATION_VALIDATED_OPERATIONS = (
    "Frozen input/implementation contracts, task manifests, exact raw output "
    "hashes, recorded ordered feature/pair universe receipts, attempt records, "
    "and resource provenance."
)
ATTESTATION_TASK_VALIDATION_BOUNDARY = (
    "Raw CSV result files are hashed as opaque bytes and are never opened or "
    "parsed by this attestor. The attestor verifies the frozen runner's "
    "already-recorded validation receipt against each cohort contract and each "
    "current file hash."
)
ATTESTATION_PROHIBITED_OPERATIONS = (
    "No interaction call or significance determination",
    "No p-value or q-value calculation, thresholding, or summary",
    "No interaction ranking, direction summary, or biological interpretation",
)
ATTESTATION_INVENTORY_DEFINITION = (
    "Every regular file recursively present under the run root after all "
    "completion gates passed and before atomic attestation publication."
)
ATTESTATION_SELF_REFERENCE_POLICY = (
    "The completion attestation is the sole excluded path and is therefore not a "
    "member of, and cannot hash itself into, the pre-attestation inventory."
)
ATTEMPT_WINDOW_DEFINITION = (
    "Earliest recorded task invocation start through latest recorded task "
    "invocation finish; this is not a sum of task runtimes."
)
ELAPSED_RESOURCE_DEFINITION = (
    "Per-task elapsed times recorded by time.monotonic in each isolated task "
    "process; sums may exceed grid wall time because tasks overlap."
)
PEAK_RSS_RESOURCE_DEFINITION = (
    "Maximum per-task process ru_maxrss high-water mark; neither summed RSS nor "
    "concurrent whole-grid peak memory."
)
PEAK_RSS_SOURCE = "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"

SOURCE_A_COMMIT = "3ad05c5ad5ddd03d7922343bafa5a2748cddb20d"
EXPECTED_EXECUTION_SNAPSHOT_SHA256 = (
    "cf365e25e4aac18f937ce669a59923bb1d0650be6e1c4ad7fe5ccae7567373bb"
)
GENERATED_VERSION_PATH = "src/dialect/_version.py"
GENERATED_VERSION_SHA256 = (
    "7da36118d9f8d662fb369b6bea4eba4d3e69491d49a57cef129643f47f8ceee7"
)
BUILDER_PATH = "analysis/build_tcga_revision_k500_authority_projection.py"
ATTESTOR_PATH = "research/notes/attest_k500_completion.py"
ATTESTOR_BYTES = 57_923
ATTESTOR_SHA256 = "238a7131a0e3aeb41928939840476d333d5ccfb19a081c19decdb6ea2a4d9de2"

GIT_EXECUTION_PATHS = (
    "analysis/__init__.py",
    "analysis/bmr_fdr_comparison.py",
    "analysis/materialize_tcga_revision_inputs.py",
    "analysis/materialize_tcga_revision_provider_inputs.py",
    "analysis/mutsig_lambda_co.py",
    "analysis/run_tcga_revision_k500.py",
    "src/dialect/__init__.py",
    "src/dialect/api.py",
    "src/dialect/baselines/__init__.py",
    "src/dialect/baselines/discover.py",
    "src/dialect/baselines/fishers.py",
    "src/dialect/baselines/megsa.py",
    "src/dialect/baselines/runner.py",
    "src/dialect/baselines/wesme.py",
    "src/dialect/bmr/__init__.py",
    "src/dialect/bmr/_cbase_run.py",
    "src/dialect/bmr/_dig_pmf.py",
    "src/dialect/bmr/base.py",
    "src/dialect/bmr/cbase.py",
    "src/dialect/bmr/dig.py",
    "src/dialect/bmr/registry.py",
    "src/dialect/data/__init__.py",
    "src/dialect/data/cohort.py",
    "src/dialect/data/io.py",
    "src/dialect/data/revision_approval.py",
    "src/dialect/data/revision_fit_policy.py",
    "src/dialect/data/tcga.py",
    "src/dialect/data/variants.py",
    "src/dialect/models/__init__.py",
    "src/dialect/models/assembly.py",
    "src/dialect/models/gene.py",
    "src/dialect/models/interaction.py",
    "src/dialect/utils/__init__.py",
    "src/dialect/utils/identify.py",
    "src/dialect/utils/merge.py",
    "external/mutsig2cv_octave_dialect.patch",
    "scripts/run_mutsig_octave.sh",
    "scripts/run_cohort_pipeline.sh",
)

TCGA_COHORTS = (
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
BMRS = ("cbase", "dig", "mutsig")

AUTHORITY_DIGEST_FIELDS = (
    "canonical_input_manifest_sha256",
    "materialization_approval_sha256",
    "fit_approval_sha256",
    "provider_input_manifest_sha256",
    "provider_full_acceptance_receipt_sha256",
    "validated_run_authority_sha256",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_HASH_CHUNK_BYTES = 1024 * 1024

MAX_COMPLETION_ATTESTATION_BYTES = 8 * 1024 * 1024
MAX_SEALED_COMPLETION_BYTES = 4 * 1024 * 1024
MAX_RUN_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PROJECTION_BYTES = 64 * 1024
MAX_BUILDER_BYTES = 512 * 1024
MAX_GIT_EXECUTABLE_BYTES = 64 * 1024 * 1024
MAX_GIT_STDOUT_BYTES = 8 * 1024 * 1024
MAX_GIT_STDERR_BYTES = 1024 * 1024
MAX_GIT_STATUS_BYTES = 64 * 1024
MAX_GIT_SECONDS = 30.0
MAX_TRACKED_FILE_COUNT = 100_000
MAX_PINNED_TRACKED_FILE_COUNT = 8_192
MAX_TRACKED_DIRECTORY_COUNT = 8_192
MAX_SHADOW_GIT_INDEX_BYTES = 64 * 1024 * 1024
MAX_GIT_HEAD_BYTES = 4 * 1024
MAX_GIT_CONFIG_BYTES = 4 * 1024 * 1024
MAX_GIT_PACKED_REFS_BYTES = 16 * 1024 * 1024
MAX_GIT_REF_BYTES = 4 * 1024
MAX_GIT_REF_COUNT = 16_384
MAX_GIT_ADMIN_ENTRY_COUNT = 4_096
GIT_EXECUTABLE_PATH = Path("/usr/bin/git")

_GIT_CONFIG_OVERRIDES = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.ignoreCase=false",
    "-c",
    "core.precomposeUnicode=false",
    "-c",
    "core.ignoreStat=false",
    "-c",
    "core.trustctime=true",
    "-c",
    "core.checkStat=default",
    "-c",
    "core.fileMode=true",
    "-c",
    "core.symlinks=true",
    "-c",
    "core.worktree=.",
    "-c",
    "core.bare=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.sparseCheckout=false",
    "-c",
    "core.sparseCheckoutCone=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.excludesFile=/dev/null",
    "-c",
    "core.alternateRefsCommand=",
    "-c",
    "submodule.recurse=false",
    "-c",
    "fetch.recurseSubmodules=false",
)

_SEALED_GIT_ENVIRONMENT = MappingProxyType(
    {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    },
)

EXPECTED_TESTED_FAMILY = {
    "epsilon_pretest_filter": "none",
    "family": "one-complete-within-cohort-tested-pair-family",
    "feature_ranking": "descending-total-eligible-mutation-event-count",
    "marginal_effect_pretest_filter": "none",
    "pair_construction": "all-unordered-pairs-of-ordered-feature-axis",
    "provider_support": "shared-native-cbase-dig-mutsig",
    "same_base_missense_nonsense": "exclude-before-fitting-and-testing",
    "tie_break": "canonical-count-matrix-column-order",
    "top_k": 500,
}
EXPECTED_RUN_CONTRACTS = {
    "feature_policy": "descending-total-eligible-mutation-event-count",
    "observation_support_universe": "full-observation-support-common-universe-v1",
    "required_contingency_table_contract": ("observed-binary-cells-00-01-10-11-v1"),
    "required_gene_support_contract": "latent-state-union-v1",
    "required_log_odds_ratio_contract": (
        "conventional-latent-odds-00x11-over-01x10-identifiable-v2"
    ),
    "required_lrt_contract": "driver-independence-constrained-mle-v1",
    "required_lrt_nestedness_tolerance": 1e-8,
    "required_output_recomputation_atol": 1e-12,
    "required_pair_effect_identifiability_contract": (
        "full-affine-rank-relative-svd-1e-12-conservative-v1"
    ),
    "required_pair_fit_contract": (
        "deterministic-simplex-coordinate-ascent-total-kkt-v2"
    ),
    "required_pair_fit_kkt_tolerance": 1e-8,
    "required_pair_fit_max_iterations": 1000,
    "required_pair_identifiability_relative_tolerance": 1e-12,
    "required_pair_simplex_tolerance": 1e-12,
    "required_rho_contract": (
        "marshall-olkin-identifiable-finite-or-degenerate-null-v2"
    ),
    "same_base_pair_policy": "exclude-before-fitting-and-testing",
    "sample_axis_contract": (
        "count-matrix-equals-authoritative-and-mutsig-patient-axis-v2"
    ),
    "undefined_rho_lrt_tolerance": 1e-8,
}


@dataclass(frozen=True)
class K500AuthorityProjectionReceipt:
    """Digest/count-only receipt returned to public-release tooling."""

    projection_path: Path
    projection_sha256: str
    completion_attestation_sha256: str
    completion_attestation_payload_sha256: str
    sealed_completion_sha256: str
    run_manifest_sha256: str
    source_a_commit: str
    release_b_commit: str
    release_tag: str
    git_blob_count: int
    generated_file_count: int
    snapshot_file_count: int
    execution_snapshot_sha256: str
    authority_digests: MappingProxyType[str, str]
    authority_digest_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the receipt."""
        return {
            "authority_digest_count": self.authority_digest_count,
            "authority_digests": dict(self.authority_digests),
            "completion_attestation_payload_sha256": (
                self.completion_attestation_payload_sha256
            ),
            "completion_attestation_sha256": self.completion_attestation_sha256,
            "execution_snapshot_sha256": self.execution_snapshot_sha256,
            "generated_file_count": self.generated_file_count,
            "git_blob_count": self.git_blob_count,
            "projection_path": self.projection_path.as_posix(),
            "projection_sha256": self.projection_sha256,
            "release_b_commit": self.release_b_commit,
            "release_tag": self.release_tag,
            "run_manifest_sha256": self.run_manifest_sha256,
            "sealed_completion_sha256": self.sealed_completion_sha256,
            "snapshot_file_count": self.snapshot_file_count,
            "source_a_commit": self.source_a_commit,
        }


@dataclass
class _PinnedPath:
    """One absolute path held by descriptor through terminal validation."""

    path: Path
    parent_fds: list[int]
    parent_signatures: list[tuple[int, ...]]
    component_names: list[str]
    descriptor: int
    content: bytes
    signature: tuple[int, ...]
    max_bytes: int
    require_stable_parent_metadata: bool

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        max_bytes: int,
        require_single_link: bool = True,
        require_stable_parent_metadata: bool = False,
    ) -> _PinnedPath:
        """Open a private regular file without following any path symlink."""
        _require_byte_limit(max_bytes, label=f"{label} byte limit")
        absolute = _require_absolute_path(path, label=label)
        parts = absolute.parts
        if not parts or parts[0] != os.sep or len(parts) < 2:
            message = f"{label} must name a file below the filesystem root."
            raise ValueError(message)
        directory_fds = [
            os.open(
                os.sep,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            ),
        ]
        component_names: list[str] = []
        try:
            for component in parts[1:-1]:
                _require_safe_basename(component, label=label)
                parent_fd = directory_fds[-1]
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
                directory_fds.append(child)
                observed = os.fstat(child)
                if not stat.S_ISDIR(observed.st_mode):
                    message = f"{label} traverses a non-directory component."
                    raise ValueError(message)  # noqa: TRY301
                _require_entry_identity(
                    parent_fd,
                    component,
                    child,
                    label=f"{label} directory component",
                )
                component_names.append(component)
            name = parts[-1]
            _require_safe_basename(name, label=label)
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fds[-1],
            )
            try:
                observed = os.fstat(descriptor)
                if not stat.S_ISREG(observed.st_mode) or (
                    require_single_link and observed.st_nlink != 1
                ):
                    message = f"{label} must be a single-link regular file."
                    raise ValueError(message)  # noqa: TRY301
                _require_size_within_limit(
                    observed,
                    max_bytes=max_bytes,
                    label=label,
                )
                content = _read_descriptor(
                    descriptor,
                    max_bytes=max_bytes,
                    label=label,
                )
                after = os.fstat(descriptor)
                signature = _stat_signature(observed)
                if (
                    _stat_signature(after) != signature
                    or len(content) != observed.st_size
                ):
                    message = f"{label} changed during its descriptor read."
                    raise ValueError(message)  # noqa: TRY301
                _require_entry_identity(
                    directory_fds[-1],
                    name,
                    descriptor,
                    label=label,
                )
                pinned = cls(
                    path=absolute,
                    parent_fds=directory_fds,
                    parent_signatures=[
                        _stat_signature(os.fstat(directory_fd))
                        for directory_fd in directory_fds
                    ],
                    component_names=component_names,
                    descriptor=descriptor,
                    content=content,
                    signature=signature,
                    max_bytes=max_bytes,
                    require_stable_parent_metadata=require_stable_parent_metadata,
                )
                pinned.require_unchanged(label=label)
                return pinned  # noqa: TRY300
            except BaseException:
                os.close(descriptor)
                raise
        except BaseException:
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)
            raise

    def require_unchanged(self, *, label: str) -> None:
        """Replay bytes and every visible entry in the held path chain."""
        if _stat_signature(os.fstat(self.descriptor)) != self.signature:
            message = f"{label} inode changed after it was pinned."
            raise ValueError(message)
        replayed = _read_descriptor(
            self.descriptor,
            max_bytes=self.max_bytes,
            label=label,
        )
        if replayed != self.content:
            message = f"{label} bytes changed after they were pinned."
            raise ValueError(message)
        if _stat_signature(os.fstat(self.descriptor)) != self.signature:
            message = f"{label} inode changed during its pinned replay."
            raise ValueError(message)
        _require_entry_identity(
            self.parent_fds[-1],
            self.path.name,
            self.descriptor,
            label=label,
        )
        for index in range(len(self.parent_fds) - 1, 0, -1):
            _require_entry_identity(
                self.parent_fds[index - 1],
                self.component_names[index - 1],
                self.parent_fds[index],
                label=f"{label} ancestor",
            )
        terminal_replay = _read_descriptor(
            self.descriptor,
            max_bytes=self.max_bytes,
            label=label,
        )
        if terminal_replay != self.content or (
            _stat_signature(os.fstat(self.descriptor)) != self.signature
        ):
            message = f"{label} changed after its visible path checks."
            raise ValueError(message)
        _require_entry_identity(
            self.parent_fds[-1],
            self.path.name,
            self.descriptor,
            label=f"{label} terminal entry",
        )
        for index in range(len(self.parent_fds) - 1, 0, -1):
            _require_entry_identity(
                self.parent_fds[index - 1],
                self.component_names[index - 1],
                self.parent_fds[index],
                label=f"{label} terminal ancestor",
            )
        if self.require_stable_parent_metadata:
            for directory_fd, signature in zip(
                self.parent_fds,
                self.parent_signatures,
                strict=True,
            ):
                if _stat_signature(os.fstat(directory_fd)) != signature:
                    message = f"{label} ancestor metadata changed after it was pinned."
                    raise ValueError(message)

    def close(self) -> None:
        """Close the file and all held ancestor directory descriptors."""
        os.close(self.descriptor)
        for directory_fd in reversed(self.parent_fds):
            os.close(directory_fd)


@dataclass
class _PinnedDirectory:
    """Absolute directory held by descriptor through publication."""

    path: Path
    fds: list[int]
    component_names: list[str]
    signatures: list[tuple[int, ...]]
    require_stable_metadata: bool

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        require_stable_metadata: bool = True,
    ) -> _PinnedDirectory:
        """Open an absolute directory without following any path symlink."""
        absolute = _require_absolute_path(path, label=label)
        if absolute == Path(os.sep):
            message = f"{label} may not be the filesystem root."
            raise ValueError(message)
        fds = [
            os.open(
                os.sep,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            ),
        ]
        names: list[str] = []
        try:
            for component in absolute.parts[1:]:
                _require_safe_basename(component, label=label)
                parent_fd = fds[-1]
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
                fds.append(child)
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    message = f"{label} traverses a non-directory component."
                    raise ValueError(message)  # noqa: TRY301
                _require_entry_identity(
                    parent_fd,
                    component,
                    child,
                    label=label,
                )
                names.append(component)
            pinned = cls(
                path=absolute,
                fds=fds,
                component_names=names,
                signatures=[_stat_signature(os.fstat(fd)) for fd in fds],
                require_stable_metadata=require_stable_metadata,
            )
            pinned.require_unchanged(label=label)
            return pinned  # noqa: TRY300
        except BaseException:
            for directory_fd in reversed(fds):
                os.close(directory_fd)
            raise

    @property
    def descriptor(self) -> int:
        """Return the held descriptor for the requested directory."""
        return self.fds[-1]

    def require_unchanged(self, *, label: str) -> None:
        """Prove the visible path still names the held directory chain."""
        for index in range(len(self.fds) - 1, 0, -1):
            _require_entry_identity(
                self.fds[index - 1],
                self.component_names[index - 1],
                self.fds[index],
                label=f"{label} ancestor",
            )
        if self.require_stable_metadata:
            for directory_fd, signature in zip(
                self.fds[-1:],
                self.signatures[-1:],
                strict=True,
            ):
                if _stat_signature(os.fstat(directory_fd)) != signature:
                    message = f"{label} metadata changed after it was pinned."
                    raise ValueError(message)

    def close(self) -> None:
        """Close all held directory descriptors."""
        for directory_fd in reversed(self.fds):
            os.close(directory_fd)


def _require_absolute_path(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or any(
        part in {".", ".."} for part in candidate.parts
    ):
        message = f"{label} must be an absolute normalized path."
        raise ValueError(message)
    return candidate


def _require_safe_basename(name: str, *, label: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        message = f"{label} has an unsafe path component."
        raise ValueError(message)


def _stat_signature(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _require_entry_identity(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> None:
    visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if _stat_signature(visible) != _stat_signature(opened):
        message = f"{label} visible entry changed after opening."
        raise ValueError(message)


def _require_byte_limit(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{label} must be one positive integer."
        raise ValueError(message)
    return value


def _require_size_within_limit(
    observed: os.stat_result,
    *,
    max_bytes: int,
    label: str,
) -> None:
    if observed.st_size > max_bytes:
        message = f"{label} exceeds its {max_bytes}-byte safety limit."
        raise ValueError(message)


def _read_descriptor(
    descriptor: int,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    _require_byte_limit(max_bytes, label=f"{label} byte limit")
    _require_size_within_limit(
        os.fstat(descriptor),
        max_bytes=max_bytes,
        label=label,
    )
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(
        descriptor,
        min(_HASH_CHUNK_BYTES, max_bytes + 1 - total),
    ):
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            message = f"{label} grew beyond its {max_bytes}-byte safety limit."
            raise ValueError(message)
    return b"".join(chunks)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(payload: object, *, newline: bool = False) -> str:
    raw = _canonical_json(payload) + (b"\n" if newline else b"")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        message = f"{label} must be one lowercase SHA-256 digest."
        raise ValueError(message)
    return value


def _require_commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        message = f"{label} must be one full lowercase Git commit."
        raise ValueError(message)
    return value


def _require_metadata_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        message = f"{label} must be one absolute normalized path."
        raise TypeError(message)
    path = _require_absolute_path(Path(value), label=label)
    if "\x00" in value or path.as_posix() != value:
        message = f"{label} must be one absolute normalized path."
        raise ValueError(message)
    return path


def _require_integer(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{label} must be an integer greater than or equal to {minimum}."
        raise ValueError(message)
    return value


def _require_exact_dict(
    value: object,
    keys: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        message = f"{label} does not have its exact closed schema."
        raise ValueError(message)
    return value


def _parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                message = f"{label} contains duplicate JSON key {key!r}."
                raise ValueError(message)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        message = f"{label} contains non-finite JSON constant {value}."
        raise ValueError(message)

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{label} is not canonical UTF-8 JSON."
        raise ValueError(message) from error
    if not isinstance(payload, dict):
        message = f"{label} must contain one JSON object."
        raise TypeError(message)
    return payload


def _sequence_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _git_environment() -> dict[str, str]:
    return dict(_SEALED_GIT_ENVIRONMENT)


def _require_exact_git_path(path: Path) -> Path:
    exact = _require_absolute_path(path, label="Git executable")
    if exact.as_posix() != GIT_EXECUTABLE_PATH.as_posix():
        message = "Git executable must be exactly /usr/bin/git."
        raise ValueError(message)
    return exact


def _git(  # noqa: PLR0913
    git_executable: Path,
    repo_root: Path,
    *arguments: str,
    check: bool = True,
    git_descriptor: int | None = None,
    repo_descriptor: int | None = None,
    max_stdout_bytes: int = MAX_GIT_STDOUT_BYTES,
    max_stderr_bytes: int = MAX_GIT_STDERR_BYTES,
    timeout_seconds: float = MAX_GIT_SECONDS,
    git_dir: Path | None = None,
    object_directory: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    exact_git = _require_exact_git_path(git_executable)
    _require_byte_limit(max_stdout_bytes, label="Git stdout byte limit")
    _require_byte_limit(max_stderr_bytes, label="Git stderr byte limit")
    timeout = _require_positive_number(
        timeout_seconds,
        label="Git execution timeout",
    )
    if (git_dir is None) != (object_directory is None):
        message = "Shadow Git execution requires both admin and object directories."
        raise ValueError(message)
    if (git_descriptor is None) != (repo_descriptor is None):
        message = "Git execution requires both executable and repository descriptors."
        raise ValueError(message)
    executable: str | None = None
    cwd = repo_root.as_posix()
    pass_fds: tuple[int, ...] = ()
    preexec_fn = None
    if git_descriptor is not None and repo_descriptor is not None:
        pass_fds = (git_descriptor, repo_descriptor)
        descriptor_root = Path("/proc/self/fd")
        if descriptor_root.is_dir():
            executable = f"{descriptor_root.as_posix()}/{git_descriptor}"
            cwd = f"{descriptor_root.as_posix()}/{repo_descriptor}"
        elif hasattr(os, "fchdir"):
            cwd = None

            def enter_pinned_repository() -> None:
                os.fchdir(repo_descriptor)

            preexec_fn = enter_pinned_repository
        else:
            message = "Descriptor-anchored Git execution is unavailable."
            raise RuntimeError(message)
    shadow_arguments: list[str] = []
    environment = _git_environment()
    if git_dir is not None and object_directory is not None:
        shadow = _require_absolute_path(git_dir, label="shadow Git directory")
        objects = _require_absolute_path(
            object_directory,
            label="Git object directory",
        )
        shadow_arguments.append(f"--git-dir={shadow.as_posix()}")
        environment["GIT_OBJECT_DIRECTORY"] = objects.as_posix()
    command = [
        exact_git.as_posix(),
        "--no-pager",
        "--no-replace-objects",
        "--no-optional-locks",
        "--work-tree=.",
        *shadow_arguments,
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
        pass_fds=pass_fds,
        preexec_fn=preexec_fn,  # noqa: PLW1509
        start_new_session=True,
    )
    stdout, stderr, returncode = _collect_bounded_process_output(
        process,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        timeout_seconds=timeout,
    )
    completed = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    if check:
        completed.check_returncode()
    return completed


def _collect_bounded_process_output(
    process: subprocess.Popen[bytes],
    *,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    timeout_seconds: float,
) -> tuple[bytes, bytes, int]:
    """Drain both Git pipes concurrently without exceeding byte ceilings."""
    if process.stdout is None or process.stderr is None:
        message = "Bounded Git execution requires stdout and stderr pipes."
        raise RuntimeError(message)
    streams = {
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    limits = {
        "stdout": max_stdout_bytes,
        "stderr": max_stderr_bytes,
    }
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
                name = key.data
                stream = streams[name]
                remaining = limits[name] + 1 - len(buffers[name])
                try:
                    chunk = os.read(
                        stream.fileno(),
                        min(_HASH_CHUNK_BYTES, remaining),
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
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as error:
            message = "Git process could not be reaped after forced termination."
            raise RuntimeError(message) from error
        raise
    finally:
        selector.close()
        for stream in streams.values():
            if not stream.closed:
                stream.close()


def _raise_git_output_limit(name: str, limit: int) -> NoReturn:
    message = f"Git {name} exceeded its {limit}-byte safety limit."
    raise ValueError(message)


def _raise_git_timeout(timeout_seconds: float) -> NoReturn:
    message = f"Git execution exceeded its {timeout_seconds:g}-second safety timeout."
    raise TimeoutError(message)


def _kill_git_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill Git and every descendant retained in its private process group."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if process.poll() is None:
            with suppress(PermissionError, ProcessLookupError):
                process.kill()


def _require_nonexecuting_git_config(raw: bytes) -> None:
    """Reject effective Git configuration that can run content filters."""
    if not raw.endswith(b"\0"):
        message = "Effective Git configuration output is not NUL terminated."
        raise ValueError(message)
    fields = raw[:-1].split(b"\0")
    if len(fields) % 2 != 0:
        message = "Effective Git configuration output is malformed."
        raise ValueError(message)
    for index in range(1, len(fields), 2):
        try:
            key = fields[index].decode("ascii").lower()
        except UnicodeDecodeError as error:
            message = "Effective Git configuration contains a non-ASCII key."
            raise ValueError(message) from error
        if re.fullmatch(r"filter\..+\.(?:clean|smudge|process)", key) is not None:
            message = "Effective Git configuration contains an executable filter."
            raise ValueError(message)


def _require_normal_git_index(raw: bytes) -> set[str]:
    """Reject sparse, assume-unchanged, unmerged, or abnormal index entries."""
    if raw and not raw.endswith(b"\0"):
        message = "Git index-flag output is not NUL terminated."
        raise ValueError(message)
    paths: set[str] = set()
    for record in raw.removesuffix(b"\0").split(b"\0") if raw else ():
        if len(record) < 3 or record[:2] != b"H ":
            message = "Git index contains a non-normal tracked entry flag."
            raise ValueError(message)
        path = _decode_git_path(record[2:], label="Git index-flag path")
        if path in paths:
            message = "Git index-flag output contains a duplicate path."
            raise ValueError(message)
        paths.add(path)
    return paths


def _require_nul_records(raw: bytes, *, label: str) -> tuple[bytes, ...]:
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        message = f"{label} is not NUL terminated."
        raise ValueError(message)
    records = tuple(raw[:-1].split(b"\0"))
    if any(not record for record in records):
        message = f"{label} contains an empty record."
        raise ValueError(message)
    return records


def _decode_git_path(raw: bytes, *, label: str) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        message = f"{label} is not UTF-8."
        raise ValueError(message) from error
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(
            part in {"", ".", ".."} or part.casefold() == ".git" for part in path.parts
        )
    ):
        message = f"{label} is not a safe normalized worktree path."
        raise ValueError(message)
    return value


def _parse_release_tree(raw: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in _require_nul_records(raw, label="release-B tree output"):
        try:
            header, raw_path = record.split(b"\t", 1)
            raw_mode, raw_type, raw_oid = header.split(b" ")
            mode = raw_mode.decode("ascii")
            object_type = raw_type.decode("ascii")
            oid = raw_oid.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            message = "Release-B tree output contains a malformed record."
            raise ValueError(message) from error
        path = _decode_git_path(raw_path, label="release-B tree path")
        if mode == "160000" or object_type == "commit":
            message = "Release B is not exactly clean: Git submodules are unsupported."
            raise ValueError(message)
        if (
            mode not in {"100644", "100755", "120000"}
            or object_type != "blob"
            or re.fullmatch(r"[0-9a-f]{40}", oid) is None
            or path in entries
        ):
            message = "Release-B tree output is outside the closed file contract."
            raise ValueError(message)
        entries[path] = (mode, oid)
    if not entries or len(entries) > MAX_TRACKED_FILE_COUNT:
        message = "Release-B tree has an invalid tracked-file count."
        raise ValueError(message)
    return entries


def _parse_index_entries(raw: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in _require_nul_records(raw, label="Git index-stage output"):
        try:
            header, raw_path = record.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = header.split(b" ")
            mode = raw_mode.decode("ascii")
            oid = raw_oid.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            message = "Git index-stage output contains a malformed record."
            raise ValueError(message) from error
        path = _decode_git_path(raw_path, label="Git index path")
        if (
            stage != "0"
            or mode not in {"100644", "100755", "120000"}
            or re.fullmatch(r"[0-9a-f]{40}", oid) is None
            or path in entries
        ):
            message = "Git index contains an unmerged or invalid entry."
            raise ValueError(message)
        entries[path] = (mode, oid)
    return entries


@dataclass
class _PinnedWorktreeSnapshot:
    """Descriptor-held tracked paths and parents spanning a clean proof."""

    directory_descriptors: dict[tuple[str, ...], int]
    directory_signatures: dict[tuple[str, ...], tuple[int, ...]]
    file_descriptors: dict[str, int]
    file_signatures: dict[str, tuple[int, ...]]
    symlink_signatures: dict[str, tuple[int, ...]]
    symlink_targets: dict[str, str]

    @classmethod
    def open(
        cls,
        root_descriptor: int,
        entries: dict[str, tuple[str, str]],
    ) -> _PinnedWorktreeSnapshot:
        """Pin metadata only; Git performs the opaque content comparison."""
        if len(entries) > MAX_PINNED_TRACKED_FILE_COUNT:
            message = "Release-B tracked-file count exceeds its pinning safety limit."
            raise ValueError(message)
        parent_parts: set[tuple[str, ...]] = {()}
        for path in entries:
            parts = PurePosixPath(path).parts
            parent_parts.update(tuple(parts[:index]) for index in range(1, len(parts)))
        if len(parent_parts) > MAX_TRACKED_DIRECTORY_COUNT:
            message = "Release-B tracked-parent count exceeds its safety limit."
            raise ValueError(message)
        directories = {(): os.dup(root_descriptor)}
        files: dict[str, int] = {}
        try:
            for parts in sorted(
                parent_parts - {()},
                key=lambda item: (len(item), item),
            ):
                parent = directories[parts[:-1]]
                child = os.open(
                    parts[-1],
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent,
                )
                try:
                    if not stat.S_ISDIR(os.fstat(child).st_mode):
                        message = "Release B tracked path traverses a non-directory."
                        raise ValueError(message)  # noqa: TRY301
                    _require_entry_identity(
                        parent,
                        parts[-1],
                        child,
                        label="release-B tracked parent",
                    )
                except BaseException:
                    os.close(child)
                    raise
                directories[parts] = child

            symlink_signatures: dict[str, tuple[int, ...]] = {}
            symlink_targets: dict[str, str] = {}
            for path, (mode, _oid) in entries.items():
                parts = PurePosixPath(path).parts
                parent = directories[tuple(parts[:-1])]
                name = parts[-1]
                if mode == "120000":
                    observed = os.stat(
                        name,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISLNK(observed.st_mode) or observed.st_nlink != 1:
                        message = (
                            "Release B is not exactly clean: tracked symlink "
                            "changed type or link count."
                        )
                        raise ValueError(message)  # noqa: TRY301
                    symlink_signatures[path] = _stat_signature(observed)
                    symlink_targets[path] = os.readlink(name, dir_fd=parent)
                    continue
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent,
                )
                try:
                    observed = os.fstat(descriptor)
                    expected_executable = mode == "100755"
                    if (
                        not stat.S_ISREG(observed.st_mode)
                        or observed.st_nlink != 1
                        or bool(observed.st_mode & 0o111) != expected_executable
                    ):
                        message = (
                            "Release B is not exactly clean: tracked file changed "
                            "type, mode, or link count."
                        )
                        raise ValueError(message)  # noqa: TRY301
                    _require_entry_identity(
                        parent,
                        name,
                        descriptor,
                        label="release-B tracked file",
                    )
                except BaseException:
                    os.close(descriptor)
                    raise
                files[path] = descriptor
            snapshot = cls(
                directory_descriptors=directories,
                directory_signatures={
                    parts: _stat_signature(os.fstat(descriptor))
                    for parts, descriptor in directories.items()
                },
                file_descriptors=files,
                file_signatures={
                    path: _stat_signature(os.fstat(descriptor))
                    for path, descriptor in files.items()
                },
                symlink_signatures=symlink_signatures,
                symlink_targets=symlink_targets,
            )
            snapshot.require_unchanged()
            return snapshot  # noqa: TRY300
        except BaseException:
            for descriptor in files.values():
                os.close(descriptor)
            for descriptor in reversed(tuple(directories.values())):
                os.close(descriptor)
            raise

    def require_unchanged(self) -> None:
        """Replay every tracked inode and parent directory identity."""
        for parts, descriptor in self.directory_descriptors.items():
            if (
                _stat_signature(os.fstat(descriptor))
                != self.directory_signatures[parts]
            ):
                message = (
                    "Release-B tracked-parent metadata changed during clean proof."
                )
                raise ValueError(message)
            if parts:
                _require_entry_identity(
                    self.directory_descriptors[parts[:-1]],
                    parts[-1],
                    descriptor,
                    label="release-B tracked parent",
                )
        for path, descriptor in self.file_descriptors.items():
            parts = PurePosixPath(path).parts
            if _stat_signature(os.fstat(descriptor)) != self.file_signatures[path]:
                message = "Release-B tracked file changed during clean proof."
                raise ValueError(message)
            _require_entry_identity(
                self.directory_descriptors[tuple(parts[:-1])],
                parts[-1],
                descriptor,
                label="release-B tracked file",
            )
        for path, signature in self.symlink_signatures.items():
            parts = PurePosixPath(path).parts
            parent = self.directory_descriptors[tuple(parts[:-1])]
            observed = os.stat(
                parts[-1],
                dir_fd=parent,
                follow_symlinks=False,
            )
            if (
                _stat_signature(observed) != signature
                or os.readlink(parts[-1], dir_fd=parent) != self.symlink_targets[path]
            ):
                message = "Release-B tracked symlink changed during clean proof."
                raise ValueError(message)

    def close(self) -> None:
        """Close every held worktree file and directory descriptor."""
        for descriptor in self.file_descriptors.values():
            os.close(descriptor)
        for descriptor in reversed(tuple(self.directory_descriptors.values())):
            os.close(descriptor)


@dataclass
class _PinnedRefTree:
    """Small descriptor-held snapshot of the repository's file ref store."""

    root: _PinnedDirectory
    directory_descriptors: dict[tuple[str, ...], int]
    directory_signatures: dict[tuple[str, ...], tuple[int, ...]]
    file_descriptors: dict[tuple[str, ...], int]
    file_signatures: dict[tuple[str, ...], tuple[int, ...]]
    file_contents: dict[tuple[str, ...], bytes]

    @classmethod
    def open(cls, path: Path) -> _PinnedRefTree:
        """Pin every loose ref without following links or reading large files."""
        root = _PinnedDirectory.open(path, label="release Git refs directory")
        directories = {(): os.dup(root.descriptor)}
        files: dict[tuple[str, ...], int] = {}
        contents: dict[tuple[str, ...], bytes] = {}
        pending = [()]
        entry_count = 0
        try:
            while pending:
                parts = pending.pop()
                parent = directories[parts]
                with os.scandir(parent) as iterator:
                    for entry in iterator:
                        entry_count += 1
                        if entry_count > MAX_GIT_REF_COUNT:
                            message = (
                                "Git ref-store entry count exceeds its safety limit."
                            )
                            raise ValueError(message)  # noqa: TRY301
                        name = entry.name
                        _require_safe_basename(name, label="Git ref entry")
                        child_parts = (*parts, name)
                        observed = os.stat(
                            name,
                            dir_fd=parent,
                            follow_symlinks=False,
                        )
                        if stat.S_ISDIR(observed.st_mode):
                            child = os.open(
                                name,
                                os.O_RDONLY
                                | getattr(os, "O_DIRECTORY", 0)
                                | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_CLOEXEC", 0),
                                dir_fd=parent,
                            )
                            try:
                                _require_entry_identity(
                                    parent,
                                    name,
                                    child,
                                    label="Git ref directory",
                                )
                            except BaseException:
                                os.close(child)
                                raise
                            directories[child_parts] = child
                            pending.append(child_parts)
                        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                            descriptor = os.open(
                                name,
                                os.O_RDONLY
                                | getattr(os, "O_NONBLOCK", 0)
                                | getattr(os, "O_NOFOLLOW", 0)
                                | getattr(os, "O_CLOEXEC", 0),
                                dir_fd=parent,
                            )
                            try:
                                opened = os.fstat(descriptor)
                                if (
                                    not stat.S_ISREG(opened.st_mode)
                                    or opened.st_nlink != 1
                                ):
                                    message = (
                                        "Git loose ref changed type or link count."
                                    )
                                    raise ValueError(message)  # noqa: TRY301
                                _require_size_within_limit(
                                    opened,
                                    max_bytes=MAX_GIT_REF_BYTES,
                                    label="Git loose ref",
                                )
                                content = _read_descriptor(
                                    descriptor,
                                    max_bytes=MAX_GIT_REF_BYTES,
                                    label="Git loose ref",
                                )
                                if _stat_signature(os.fstat(descriptor)) != (
                                    _stat_signature(opened)
                                ):
                                    message = "Git loose ref changed while opening."
                                    raise ValueError(message)  # noqa: TRY301
                                _require_entry_identity(
                                    parent,
                                    name,
                                    descriptor,
                                    label="Git loose ref",
                                )
                            except BaseException:
                                os.close(descriptor)
                                raise
                            files[child_parts] = descriptor
                            contents[child_parts] = content
                        else:
                            message = (
                                "Git ref store contains a special or linked entry."
                            )
                            raise ValueError(message)  # noqa: TRY301
            snapshot = cls(
                root=root,
                directory_descriptors=directories,
                directory_signatures={
                    parts: _stat_signature(os.fstat(descriptor))
                    for parts, descriptor in directories.items()
                },
                file_descriptors=files,
                file_signatures={
                    parts: _stat_signature(os.fstat(descriptor))
                    for parts, descriptor in files.items()
                },
                file_contents=contents,
            )
            snapshot.require_unchanged()
            return snapshot  # noqa: TRY300
        except BaseException:
            for descriptor in files.values():
                os.close(descriptor)
            for descriptor in reversed(tuple(directories.values())):
                os.close(descriptor)
            root.close()
            raise

    def require_unchanged(self) -> None:
        """Replay loose-ref bytes and every directory identity."""
        for parts, descriptor in self.directory_descriptors.items():
            if (
                _stat_signature(os.fstat(descriptor))
                != self.directory_signatures[parts]
            ):
                message = "Git loose-ref directory changed during source proof."
                raise ValueError(message)
            if parts:
                _require_entry_identity(
                    self.directory_descriptors[parts[:-1]],
                    parts[-1],
                    descriptor,
                    label="Git loose-ref directory",
                )
        for parts, descriptor in self.file_descriptors.items():
            if (
                _stat_signature(os.fstat(descriptor)) != self.file_signatures[parts]
                or _read_descriptor(
                    descriptor,
                    max_bytes=MAX_GIT_REF_BYTES,
                    label="Git loose ref",
                )
                != self.file_contents[parts]
            ):
                message = "Git loose ref changed during source proof."
                raise ValueError(message)
            _require_entry_identity(
                self.directory_descriptors[parts[:-1]],
                parts[-1],
                descriptor,
                label="Git loose ref",
            )
        self.root.require_unchanged(label="release Git refs directory")

    def close(self) -> None:
        """Close every loose-ref descriptor."""
        for descriptor in self.file_descriptors.values():
            os.close(descriptor)
        for descriptor in reversed(tuple(self.directory_descriptors.values())):
            os.close(descriptor)
        self.root.close()


def _bounded_directory_names(
    descriptor: int,
    *,
    max_entries: int,
    label: str,
) -> set[str]:
    """Return bounded names from one already pinned directory descriptor."""
    names: set[str] = set()
    with os.scandir(descriptor) as iterator:
        for entry in iterator:
            if len(names) >= max_entries:
                message = f"{label} entry count exceeds its safety limit."
                raise ValueError(message)
            _require_safe_basename(entry.name, label=label)
            names.add(entry.name)
    return names


@dataclass
class _PinnedGitState:
    """Git index, refs, and control files held through terminal source proof."""

    git_directory: _PinnedDirectory
    files: tuple[_PinnedPath, ...]
    refs: _PinnedRefTree

    @classmethod
    def open(cls, repo_root: Path) -> _PinnedGitState:
        """Pin the supported ordinary SHA-1 file-backed repository state."""
        git_path = repo_root / ".git"
        git_directory = _PinnedDirectory.open(
            git_path,
            label="release Git admin directory",
        )
        files: list[_PinnedPath] = []
        refs: _PinnedRefTree | None = None
        try:
            admin_names = _bounded_directory_names(
                git_directory.descriptor,
                max_entries=MAX_GIT_ADMIN_ENTRY_COUNT,
                label="release Git admin directory",
            )
            if (
                "commondir" in admin_names
                or "reftable" in admin_names
                or any(name.startswith("sharedindex.") for name in admin_names)
            ):
                message = "Release Git admin layout is outside the closed contract."
                raise ValueError(message)  # noqa: TRY301
            for name, limit in (
                ("HEAD", MAX_GIT_HEAD_BYTES),
                ("config", MAX_GIT_CONFIG_BYTES),
                ("index", MAX_SHADOW_GIT_INDEX_BYTES),
            ):
                files.append(
                    _PinnedPath.open(
                        git_path / name,
                        label=f"release Git {name}",
                        max_bytes=limit,
                    ),
                )
            for name, limit in (
                ("config.worktree", MAX_GIT_CONFIG_BYTES),
                ("packed-refs", MAX_GIT_PACKED_REFS_BYTES),
            ):
                if name in admin_names:
                    files.append(
                        _PinnedPath.open(
                            git_path / name,
                            label=f"release Git {name}",
                            max_bytes=limit,
                        ),
                    )
            refs = _PinnedRefTree.open(git_path / "refs")
            state = cls(
                git_directory=git_directory,
                files=tuple(files),
                refs=refs,
            )
            state.require_unchanged()
            return state  # noqa: TRY300
        except BaseException:
            if refs is not None:
                refs.close()
            for item in reversed(files):
                item.close()
            git_directory.close()
            raise

    def require_unchanged(self) -> None:
        """Replay every held Git administrative input."""
        for item in self.files:
            item.require_unchanged(label=f"release Git {item.path.name}")
        self.refs.require_unchanged()
        self.git_directory.require_unchanged(label="release Git admin directory")

    def close(self) -> None:
        """Close every held Git administrative descriptor."""
        self.refs.close()
        for item in reversed(self.files):
            item.close()
        self.git_directory.close()


def _write_private_shadow_file(
    directory_descriptor: int,
    name: str,
    content: bytes,
) -> None:
    """Create one exact private shadow-Git control file without replacement."""
    _require_safe_basename(name, label="shadow Git file")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o400,
        dir_fd=directory_descriptor,
    )
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written < 1:
                message = "Shadow Git control-file write made no progress."
                raise OSError(message)
            offset += written
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o400
            or observed.st_size != len(content)
        ):
            message = "Shadow Git control file is not exact and private."
            raise ValueError(message)
        _require_entry_identity(
            directory_descriptor,
            name,
            descriptor,
            label="shadow Git control file",
        )
    finally:
        os.close(descriptor)


def _require_shadow_git_clean(
    run_git: Callable[..., subprocess.CompletedProcess[bytes]],
    *,
    release_b: str,
    object_directory: Path,
) -> None:
    """Compare the worktree with release B using a private inert Git admin dir."""
    with tempfile.TemporaryDirectory(prefix="dialect-k500-shadow-git-") as temporary:
        shadow_path = Path(temporary).resolve(strict=True)
        shadow_path.chmod(0o700)
        shadow = _PinnedDirectory.open(
            shadow_path,
            label="private shadow Git directory",
            require_stable_metadata=False,
        )
        index: _PinnedPath | None = None
        try:
            child_descriptors: dict[str, int] = {}
            try:
                for name in ("info", "objects", "refs"):
                    os.mkdir(name, mode=0o700, dir_fd=shadow.descriptor)
                    child = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=shadow.descriptor,
                    )
                    try:
                        os.fchmod(child, 0o700)
                        _require_entry_identity(
                            shadow.descriptor,
                            name,
                            child,
                            label="shadow Git directory entry",
                        )
                    except BaseException:
                        os.close(child)
                        raise
                    child_descriptors[name] = child
                _write_private_shadow_file(
                    shadow.descriptor,
                    "HEAD",
                    f"{release_b}\n".encode("ascii"),
                )
                _write_private_shadow_file(
                    shadow.descriptor,
                    "config",
                    b"[core]\n\trepositoryformatversion = 0\n\tbare = false\n",
                )
                _write_private_shadow_file(
                    child_descriptors["info"],
                    "exclude",
                    b"# private shadow excludes only the explicit pathspec\n",
                )
            finally:
                for descriptor in child_descriptors.values():
                    os.close(descriptor)
            run_git(
                "read-tree",
                "--reset",
                release_b,
                git_dir=shadow_path,
                object_directory=object_directory,
            )
            index = _PinnedPath.open(
                shadow_path / "index",
                label="shadow Git index",
                max_bytes=MAX_SHADOW_GIT_INDEX_BYTES,
            )
            untracked_ignore_controls = run_git(
                "ls-files",
                "--others",
                "-z",
                "--",
                ".gitignore",
                ":(glob,top)**/.gitignore",
                ":(exclude,top).git",
                max_stdout_bytes=MAX_GIT_STATUS_BYTES,
                git_dir=shadow_path,
                object_directory=object_directory,
            ).stdout
            if untracked_ignore_controls:
                message = (
                    "Release B contains an untracked worktree ignore-control file."
                )
                raise ValueError(message)
            status_output = run_git(
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
                "--",
                ".",
                ":(exclude,top).git",
                max_stdout_bytes=MAX_GIT_STATUS_BYTES,
                git_dir=shadow_path,
                object_directory=object_directory,
            ).stdout
            if status_output:
                message = "Release B is not exactly clean under the inert Git proof."
                raise ValueError(message)
            index.require_unchanged(label="shadow Git index")
            shadow.require_unchanged(label="private shadow Git directory")
        finally:
            if index is not None:
                index.close()
            shadow.close()


def _require_release_tag(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("-")
        or value.endswith("/")
        or ".." in value
        or "@{" in value
        or "//" in value
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) is None
    ):
        message = "release tag has an unsafe or noncanonical name."
        raise ValueError(message)
    return value


def _anchored_git_receipt(run_manifest: dict[str, Any]) -> dict[str, Any]:
    git = _require_exact_dict(
        run_manifest.get("git"),
        {"dirty", "executable", "head", "status", "version"},
        label="anchored K500 run Git receipt",
    )
    receipt = _require_exact_dict(
        git["executable"],
        {"bytes", "path", "sha256"},
        label="anchored K500 run Git-executable receipt",
    )
    authority = run_manifest.get("revision_authority")
    provider = authority.get("provider_input") if isinstance(authority, dict) else None
    provider_git = (
        provider.get("git_executable") if isinstance(provider, dict) else None
    )
    if provider_git != receipt:
        message = "Run and provider authority do not share one Git executable receipt."
        raise ValueError(message)
    _require_integer(
        receipt["bytes"],
        label="anchored Git-executable byte count",
        minimum=1,
    )
    receipt_path = _require_metadata_path(
        receipt["path"],
        label="anchored Git-executable path",
    )
    if receipt_path != GIT_EXECUTABLE_PATH:
        message = "Anchored Git executable must be exactly /usr/bin/git."
        raise ValueError(message)
    _require_sha256(receipt["sha256"], label="anchored Git-executable digest")
    return dict(receipt)


def _source_boundary(  # noqa: PLR0913
    repo_root: Path,
    git_executable: Path,
    *,
    release_b_commit: str,
    release_tag: str,
    builder: _PinnedPath,
    anchored_git_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    exact_git = _require_exact_git_path(git_executable)
    root = _PinnedDirectory.open(repo_root, label="release repository root")
    git_file: _PinnedPath | None = None
    object_directory: _PinnedDirectory | None = None
    try:
        object_directory = _PinnedDirectory.open(
            repo_root / ".git/objects",
            label="release Git object directory",
        )
        git_file = _PinnedPath.open(
            exact_git,
            label="Git executable",
            max_bytes=MAX_GIT_EXECUTABLE_BYTES,
            require_single_link=False,
            require_stable_parent_metadata=True,
        )
        if anchored_git_receipt != {
            "bytes": len(git_file.content),
            "path": git_file.path.as_posix(),
            "sha256": hashlib.sha256(git_file.content).hexdigest(),
        }:
            message = "Git executable differs from the independently anchored run."
            raise ValueError(message)

        def run_git(
            *arguments: str,
            check: bool = True,
            max_stdout_bytes: int = MAX_GIT_STDOUT_BYTES,
            git_dir: Path | None = None,
            object_directory: Path | None = None,
        ) -> subprocess.CompletedProcess[bytes]:
            return _git(
                exact_git,
                repo_root,
                *arguments,
                check=check,
                git_descriptor=git_file.descriptor,
                repo_descriptor=root.descriptor,
                max_stdout_bytes=max_stdout_bytes,
                git_dir=git_dir,
                object_directory=object_directory,
            )

        source_a = _require_commit(SOURCE_A_COMMIT, label="source A")
        release_b = _require_commit(release_b_commit, label="release B")
        tag = _require_release_tag(release_tag)
        object_format = run_git("rev-parse", "--show-object-format").stdout.strip()
        if object_format != b"sha1":
            message = "Release repository must use the exact SHA-1 object format."
            raise ValueError(message)

        def read_config_snapshot() -> tuple[bytes, bytes]:
            names = run_git(
                "config",
                "--null",
                "--show-origin",
                "--name-only",
                "--includes",
                "--list",
            ).stdout
            _require_nonexecuting_git_config(names)
            values = run_git(
                "config",
                "--null",
                "--show-origin",
                "--includes",
                "--list",
            ).stdout
            return names, values

        def require_clean_release() -> tuple[_PinnedWorktreeSnapshot, _PinnedGitState]:
            config_before = read_config_snapshot()
            tree_before = run_git(
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                release_b,
            ).stdout
            tree_entries = _parse_release_tree(tree_before)
            index_before = run_git("ls-files", "--stage", "-z").stdout
            index_entries = _parse_index_entries(index_before)
            flags_before = run_git("ls-files", "-v", "-z").stdout
            flagged_paths = _require_normal_git_index(flags_before)
            if index_entries != tree_entries or flagged_paths != set(tree_entries):
                message = "Release B is not exactly clean: index differs from commit."
                raise ValueError(message)
            _require_shadow_git_clean(
                run_git,
                release_b=release_b,
                object_directory=object_directory.path,
            )
            worktree = _PinnedWorktreeSnapshot.open(
                root.descriptor,
                tree_entries,
            )
            git_state: _PinnedGitState | None = None
            try:
                git_state = _PinnedGitState.open(repo_root)
                _require_shadow_git_clean(
                    run_git,
                    release_b=release_b,
                    object_directory=object_directory.path,
                )
                config_after = read_config_snapshot()
                tree_after = run_git(
                    "ls-tree",
                    "-r",
                    "-z",
                    "--full-tree",
                    release_b,
                ).stdout
                index_after = run_git("ls-files", "--stage", "-z").stdout
                flags_after = run_git("ls-files", "-v", "-z").stdout
                _require_normal_git_index(flags_after)
                if (
                    config_after != config_before
                    or tree_after != tree_before
                    or index_after != index_before
                    or flags_after != flags_before
                ):
                    message = (
                        "Git configuration, tree, or index changed during clean proof."
                    )
                    raise ValueError(message)  # noqa: TRY301
                worktree.require_unchanged()
                git_state.require_unchanged()
                return worktree, git_state  # noqa: TRY300
            except BaseException:
                if git_state is not None:
                    git_state.close()
                worktree.close()
                raise

        for commit, label in ((source_a, "source A"), (release_b, "release B")):
            resolved = (
                run_git(
                    "rev-parse",
                    "--verify",
                    f"{commit}^{{commit}}",
                )
                .stdout.decode("ascii")
                .strip()
            )
            if resolved != commit:
                message = f"{label} did not resolve to its exact commit."
                raise ValueError(message)
        head = (
            run_git("rev-parse", "HEAD")
            .stdout.decode(
                "ascii",
            )
            .strip()
        )
        if head != release_b:
            message = (
                "Live repository HEAD is not the independently supplied release B."
            )
            raise ValueError(message)
        initial_worktree, initial_git_state = require_clean_release()
        try:
            initial_worktree.require_unchanged()
            initial_git_state.require_unchanged()
        finally:
            initial_git_state.close()
            initial_worktree.close()
        tag_result = run_git(
            "rev-parse",
            "--verify",
            f"refs/tags/{tag}^{{commit}}",
            check=False,
        )
        if tag_result.returncode != 0:
            message = "Release tag does not resolve to a commit."
            raise ValueError(message)
        tag_commit = tag_result.stdout.decode("ascii").strip()
        if tag_commit != release_b:
            message = "Release tag does not resolve to exact release B."
            raise ValueError(message)
        ancestor = run_git(
            "merge-base",
            "--is-ancestor",
            source_a,
            release_b,
            check=False,
        )
        if ancestor.returncode != 0:
            message = "Source A is not an ancestor of release B."
            raise ValueError(message)

        source_hashes: dict[str, str] = {}
        if len(GIT_EXECUTION_PATHS) != len(set(GIT_EXECUTION_PATHS)):
            message = "Frozen Git execution-path inventory contains duplicates."
            raise RuntimeError(message)
        for path in GIT_EXECUTION_PATHS:
            parsed = PurePosixPath(path)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or parsed.as_posix() != path
            ):
                message = f"Frozen Git execution path is unsafe: {path}"
                raise RuntimeError(message)
            source_bytes = run_git("cat-file", "blob", f"{source_a}:{path}").stdout
            release_bytes = run_git("cat-file", "blob", f"{release_b}:{path}").stdout
            if release_bytes != source_bytes:
                message = f"Release B changed frozen K500 execution blob {path}."
                raise ValueError(message)
            source_hashes[path] = hashlib.sha256(source_bytes).hexdigest()
        for commit in (source_a, release_b):
            observed_generated = run_git(
                "ls-tree",
                "--name-only",
                commit,
                "--",
                GENERATED_VERSION_PATH,
            )
            if observed_generated.stdout:
                message = f"Generated version path is unexpectedly tracked at {commit}."
                raise ValueError(message)
        snapshot = {**source_hashes, GENERATED_VERSION_PATH: GENERATED_VERSION_SHA256}
        snapshot_sha256 = _json_sha256(snapshot)
        if snapshot_sha256 != EXPECTED_EXECUTION_SNAPSHOT_SHA256:
            message = "Frozen 39-file execution snapshot does not match its trust root."
            raise ValueError(message)

        builder_blob = run_git(
            "cat-file",
            "blob",
            f"{release_b}:{BUILDER_PATH}",
        ).stdout
        if builder_blob != builder.content:
            message = "Live projection builder differs from its release-B Git blob."
            raise ValueError(message)
        builder_record = {
            "bytes": len(builder.content),
            "path": BUILDER_PATH,
            "sha256": hashlib.sha256(builder.content).hexdigest(),
        }
        source = {
            "execution_snapshot_sha256": snapshot_sha256,
            "generated_file_count": 1,
            "generated_path": GENERATED_VERSION_PATH,
            "generated_sha256": GENERATED_VERSION_SHA256,
            "git_executable": dict(anchored_git_receipt),
            "git_blob_count": len(GIT_EXECUTION_PATHS),
            "release_b_commit": release_b,
            "release_tag": tag,
            "snapshot_file_count": len(snapshot),
            "source_a_commit": source_a,
        }
        terminal_worktree, terminal_git_state = require_clean_release()
        try:
            terminal_head = (
                run_git(
                    "rev-parse",
                    "HEAD",
                )
                .stdout.decode("ascii")
                .strip()
            )
            terminal_tag = (
                run_git(
                    "rev-parse",
                    "--verify",
                    f"refs/tags/{tag}^{{commit}}",
                )
                .stdout.decode("ascii")
                .strip()
            )
            if terminal_head != release_b or terminal_tag != release_b:
                message = (
                    "Release-B HEAD, tag, or clean state changed during source proof."
                )
                raise ValueError(message)
            root.require_unchanged(label="release repository root")
            object_directory.require_unchanged(label="release Git object directory")
            git_file.require_unchanged(label="Git executable")
            builder.require_unchanged(label="projection builder")
            terminal_worktree.require_unchanged()
            terminal_git_state.require_unchanged()
            return {"builder": builder_record, "source": source}, snapshot
        finally:
            terminal_git_state.close()
            terminal_worktree.close()
    finally:
        if git_file is not None:
            git_file.close()
        if object_directory is not None:
            object_directory.close()
        root.close()


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        message = f"{label} must be one nonempty string."
        raise ValueError(message)
    return value


def _require_positive_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        message = f"{label} must be one finite positive number."
        raise ValueError(message)
    return float(value)


def _validate_attested_resource_usage(
    value: object,
    *,
    coordinate: str,
) -> tuple[float, int]:
    usage = _require_exact_dict(
        value,
        {"elapsed_seconds", "peak_rss"},
        label=f"completion-attestation task resource usage {coordinate}",
    )
    elapsed = _require_positive_number(
        usage["elapsed_seconds"],
        label=f"completion-attestation elapsed seconds {coordinate}",
    )
    peak = _require_exact_dict(
        usage["peak_rss"],
        {"bytes", "native_unit", "native_value", "platform", "source"},
        label=f"completion-attestation peak RSS {coordinate}",
    )
    native_value = _require_integer(
        peak["native_value"],
        label=f"completion-attestation native peak RSS {coordinate}",
        minimum=1,
    )
    peak_bytes = _require_integer(
        peak["bytes"],
        label=f"completion-attestation peak RSS bytes {coordinate}",
        minimum=1,
    )
    platform = _require_nonempty_string(
        peak["platform"],
        label=f"completion-attestation RSS platform {coordinate}",
    )
    if platform == "darwin":
        expected_unit = "bytes"
        multiplier = 1
    elif platform.startswith("linux"):
        expected_unit = "KiB"
        multiplier = 1024
    else:
        message = f"Completion-attestation RSS platform is invalid: {coordinate}."
        raise ValueError(message)
    if (
        peak["native_unit"] != expected_unit
        or peak_bytes != native_value * multiplier
        or peak["source"] != PEAK_RSS_SOURCE
    ):
        message = f"Completion-attestation peak RSS is invalid: {coordinate}."
        raise ValueError(message)
    return elapsed, peak_bytes


def _validate_attested_contracts(
    value: object,
    sealed_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = value
    if not isinstance(records, list) or len(records) != len(TCGA_COHORTS):
        message = "Completion-attestation contract grid is invalid."
        raise ValueError(message)
    result: dict[str, dict[str, Any]] = {}
    for cohort, record, sealed in zip(
        TCGA_COHORTS,
        records,
        sealed_manifest["contracts"],
        strict=True,
    ):
        receipt = _require_exact_dict(
            record,
            {
                "cohort",
                "contract_path",
                "contract_sha256",
                "features",
                "ordered_features_sha256",
                "ordered_pair_sha256",
                "pairs_per_background",
                "same_base_pairs_excluded",
                "samples",
            },
            label="completion-attestation contract receipt",
        )
        features = _require_integer(
            receipt["features"],
            label=f"completion-attestation feature count {cohort}",
            minimum=1,
        )
        pairs = _require_integer(
            receipt["pairs_per_background"],
            label=f"completion-attestation pair count {cohort}",
            minimum=0,
        )
        excluded = _require_integer(
            receipt["same_base_pairs_excluded"],
            label=f"completion-attestation same-base exclusion count {cohort}",
            minimum=0,
        )
        _require_integer(
            receipt["samples"],
            label=f"completion-attestation sample count {cohort}",
            minimum=1,
        )
        for key in (
            "contract_sha256",
            "ordered_features_sha256",
            "ordered_pair_sha256",
        ):
            _require_sha256(
                receipt[key],
                label=f"completion-attestation {key} {cohort}",
            )
        if (
            receipt["cohort"] != cohort
            or receipt["contract_path"] != f"contracts/{cohort}.json"
            or receipt["contract_sha256"] != sealed["contract_sha256"]
            or features != 500
            or pairs + excluded != features * (features - 1) // 2
        ):
            message = f"Completion-attestation contract receipt is invalid: {cohort}."
            raise ValueError(message)
        result[cohort] = receipt
    return result


def _validate_attested_tasks(
    value: object,
    sealed_manifest: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    records = value
    coordinates = tuple((cohort, bmr) for cohort in TCGA_COHORTS for bmr in BMRS)
    if not isinstance(records, list) or len(records) != len(coordinates):
        message = "Completion-attestation task grid is invalid."
        raise ValueError(message)
    elapsed_by_bmr = {bmr: [] for bmr in BMRS}
    peak_by_bmr = {bmr: [] for bmr in BMRS}
    for (cohort, bmr), record, sealed in zip(
        coordinates,
        records,
        sealed_manifest["tasks"],
        strict=True,
    ):
        coordinate = f"{cohort}/{bmr}"
        receipt = _require_exact_dict(
            record,
            {
                "bmr",
                "cohort",
                "completed_at_utc",
                "contract_sha256",
                "resource_usage",
                "task_manifest_path",
                "task_manifest_sha256",
                "validation",
            },
            label="completion-attestation task receipt",
        )
        _require_nonempty_string(
            receipt["completed_at_utc"],
            label=f"completion-attestation completion timestamp {coordinate}",
        )
        _require_sha256(
            receipt["contract_sha256"],
            label=f"completion-attestation task contract digest {coordinate}",
        )
        _require_sha256(
            receipt["task_manifest_sha256"],
            label=f"completion-attestation task-manifest digest {coordinate}",
        )
        contract = contracts[cohort]
        validation = _require_exact_dict(
            receipt["validation"],
            {
                "features",
                "ordered_features_sha256",
                "ordered_pair_sha256",
                "pairs",
                "pairwise_sha256",
                "single_gene_sha256",
            },
            label=f"completion-attestation task validation {coordinate}",
        )
        _require_integer(
            validation["features"],
            label=f"completion-attestation task features {coordinate}",
            minimum=1,
        )
        _require_integer(
            validation["pairs"],
            label=f"completion-attestation task pairs {coordinate}",
            minimum=0,
        )
        for key in (
            "ordered_features_sha256",
            "ordered_pair_sha256",
            "pairwise_sha256",
            "single_gene_sha256",
        ):
            _require_sha256(
                validation[key],
                label=f"completion-attestation task {key} {coordinate}",
            )
        if (
            receipt["cohort"] != cohort
            or receipt["bmr"] != bmr
            or receipt["task_manifest_path"]
            != f"tasks/{cohort}/{bmr}/task_manifest.json"
            or receipt["task_manifest_sha256"] != sealed["task_manifest"]["sha256"]
            or receipt["contract_sha256"] != sealed["contract_sha256"]
            or receipt["contract_sha256"] != contract["contract_sha256"]
            or validation["features"] != contract["features"]
            or validation["pairs"] != contract["pairs_per_background"]
            or validation["ordered_features_sha256"]
            != contract["ordered_features_sha256"]
            or validation["ordered_pair_sha256"] != contract["ordered_pair_sha256"]
            or validation["pairwise_sha256"]
            != sealed["pairwise_interaction_results"]["sha256"]
            or validation["single_gene_sha256"]
            != sealed["single_gene_results"]["sha256"]
        ):
            message = f"Completion-attestation task receipt is invalid: {coordinate}."
            raise ValueError(message)
        elapsed, peak = _validate_attested_resource_usage(
            receipt["resource_usage"],
            coordinate=coordinate,
        )
        elapsed_by_bmr[bmr].append(elapsed)
        peak_by_bmr[bmr].append(peak)
    return elapsed_by_bmr, peak_by_bmr


def _validate_attested_attempts(value: object, *, task_count: int) -> None:
    attempts = _require_exact_dict(
        value,
        {
            "attempt_records",
            "earliest_started_at_utc",
            "exit_status_counts",
            "latest_finished_at_utc",
            "observed_window_definition",
            "successful_task_coordinates",
        },
        label="completion-attestation attempt summary",
    )
    attempt_count = _require_integer(
        attempts["attempt_records"],
        label="completion-attestation attempt count",
        minimum=task_count,
    )
    successful = _require_integer(
        attempts["successful_task_coordinates"],
        label="completion-attestation successful task count",
        minimum=0,
    )
    _require_nonempty_string(
        attempts["earliest_started_at_utc"],
        label="completion-attestation earliest attempt timestamp",
    )
    _require_nonempty_string(
        attempts["latest_finished_at_utc"],
        label="completion-attestation latest attempt timestamp",
    )
    counts = attempts["exit_status_counts"]
    if not isinstance(counts, dict) or not counts:
        message = "Completion-attestation exit-status counts are invalid."
        raise ValueError(message)
    count_total = 0
    for code, count in counts.items():
        if not isinstance(code, str) or re.fullmatch(r"-?[0-9]+", code) is None:
            message = "Completion-attestation exit-status key is invalid."
            raise ValueError(message)
        count_total += _require_integer(
            count,
            label="completion-attestation exit-status count",
            minimum=1,
        )
    if (
        successful != task_count
        or counts.get("0", 0) < task_count
        or count_total != attempt_count
        or attempts["observed_window_definition"] != ATTEMPT_WINDOW_DEFINITION
    ):
        message = "Completion-attestation attempt summary is inconsistent."
        raise ValueError(message)


def _validate_attested_resources(
    value: object,
    *,
    runner_resource_policy: object,
    elapsed_by_bmr: dict[str, list[float]],
    peak_by_bmr: dict[str, list[int]],
) -> None:
    resources = _require_exact_dict(
        value,
        {"runner_resource_policy", "task_elapsed_seconds", "task_peak_rss_bytes"},
        label="completion-attestation resource summary",
    )
    elapsed = [item for bmr in BMRS for item in elapsed_by_bmr[bmr]]
    peaks = [item for bmr in BMRS for item in peak_by_bmr[bmr]]
    expected_elapsed = {
        "definition": ELAPSED_RESOURCE_DEFINITION,
        "maximum": max(elapsed),
        "median": statistics.median(elapsed),
        "minimum": min(elapsed),
        "sum": math.fsum(elapsed),
        "sum_by_background": {bmr: math.fsum(elapsed_by_bmr[bmr]) for bmr in BMRS},
    }
    expected_peak = {
        "definition": PEAK_RSS_RESOURCE_DEFINITION,
        "maximum": max(peaks),
        "maximum_by_background": {bmr: max(peak_by_bmr[bmr]) for bmr in BMRS},
        "source": PEAK_RSS_SOURCE,
    }
    if (
        resources["runner_resource_policy"] != runner_resource_policy
        or resources["task_elapsed_seconds"] != expected_elapsed
        or resources["task_peak_rss_bytes"] != expected_peak
    ):
        message = "Completion-attestation resource summary is inconsistent."
        raise ValueError(message)


def _known_inventory_receipts(
    sealed_manifest: dict[str, Any],
    *,
    sealed_sha256: str,
    sealed_bytes: int,
    run_sha256: str,
    run_bytes: int,
) -> dict[str, dict[str, Any]]:
    receipts = {
        "run_manifest.json": {"bytes": run_bytes, "sha256": run_sha256},
        "sealed_completion_manifest.json": {
            "bytes": sealed_bytes,
            "sha256": sealed_sha256,
        },
    }
    for contract in sealed_manifest["contracts"]:
        receipts[f"contracts/{contract['cohort']}.json"] = {
            "bytes": contract["bytes"],
            "sha256": contract["file_sha256"],
        }
    for task in sealed_manifest["tasks"]:
        root = f"tasks/{task['cohort']}/{task['bmr']}"
        receipts[f"{root}/pairwise_interaction_results.csv"] = task[
            "pairwise_interaction_results"
        ]
        receipts[f"{root}/single_gene_results.csv"] = task["single_gene_results"]
        receipts[f"{root}/task_manifest.json"] = task["task_manifest"]
    return receipts


def _validate_attested_inventory(
    value: object,
    *,
    known_receipts: dict[str, dict[str, Any]],
) -> None:
    inventory = _require_exact_dict(
        value,
        {
            "definition",
            "excluded_paths",
            "file_count",
            "files",
            "records_sha256",
            "self_reference_policy",
            "total_bytes",
        },
        label="completion-attestation inventory",
    )
    inventory_files = inventory["files"]
    if not isinstance(inventory_files, list):
        message = "Completion-attestation inventory files are invalid."
        raise TypeError(message)
    inventory_by_path: dict[str, dict[str, Any]] = {}
    inventory_total = 0
    for record in inventory_files:
        item = _require_exact_dict(
            record,
            {"bytes", "mtime_ns", "path", "sha256"},
            label="completion-attestation inventory file",
        )
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).as_posix() != path
            or path == "completion_attestation.json"
            or path in inventory_by_path
        ):
            message = "Completion-attestation inventory contains an unsafe path."
            raise ValueError(message)
        _require_integer(
            item["bytes"],
            label="completion-attestation inventory byte count",
            minimum=0,
        )
        _require_integer(
            item["mtime_ns"],
            label="completion-attestation inventory mtime",
            minimum=1,
        )
        _require_sha256(
            item["sha256"],
            label="completion-attestation inventory file digest",
        )
        inventory_by_path[path] = item
        inventory_total += item["bytes"]
    _require_integer(
        inventory["file_count"],
        label="completion-attestation inventory file count",
        minimum=len(known_receipts),
    )
    _require_integer(
        inventory["total_bytes"],
        label="completion-attestation inventory byte count",
        minimum=1,
    )
    if (
        inventory["definition"] != ATTESTATION_INVENTORY_DEFINITION
        or inventory["excluded_paths"] != ["completion_attestation.json"]
        or inventory["self_reference_policy"] != ATTESTATION_SELF_REFERENCE_POLICY
        or inventory["file_count"] != len(inventory_files)
        or inventory["total_bytes"] != inventory_total
        or inventory["records_sha256"] != _json_sha256(inventory_files)
        or list(inventory_by_path) != sorted(inventory_by_path)
    ):
        message = "Completion-attestation inventory summary is invalid."
        raise ValueError(message)
    for path, expected in known_receipts.items():
        observed = inventory_by_path.get(path)
        if (
            observed is None
            or {
                "bytes": observed["bytes"],
                "sha256": observed["sha256"],
            }
            != expected
        ):
            message = f"Completion-attestation inventory lost sealed receipt {path}."
            raise ValueError(message)


def _validate_attestation(  # noqa: PLR0913
    payload: dict[str, Any],
    *,
    sealed_sha256: str,
    sealed_bytes: int,
    run_sha256: str,
    run_bytes: int,
    source_snapshot: dict[str, str],
    sealed_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
) -> str:
    keys = {
        "attempts",
        "attestation_payload_sha256",
        "attestation_type",
        "completion",
        "contracts",
        "created_at_utc",
        "frozen_run",
        "generator",
        "pre_attestation_inventory",
        "resources",
        "schema_version",
        "scope",
        "sealed_completion",
        "status",
        "tasks",
    }
    _require_exact_dict(payload, keys, label="completion attestation")
    payload_digest = _require_sha256(
        payload["attestation_payload_sha256"],
        label="completion-attestation payload digest",
    )
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "attestation_payload_sha256"
    }
    if _json_sha256(unsigned) != payload_digest:
        message = "Completion-attestation internal payload digest does not verify."
        raise ValueError(message)
    if (
        payload["schema_version"] != ATTESTATION_SCHEMA
        or payload["attestation_type"] != ATTESTATION_TYPE
        or payload["status"] != "complete"
    ):
        message = "Completion-attestation identity or status is invalid."
        raise ValueError(message)
    _require_nonempty_string(
        payload["created_at_utc"],
        label="completion-attestation creation timestamp",
    )
    generator = _require_exact_dict(
        payload["generator"],
        {"bytes", "path", "sha256"},
        label="completion-attestation generator",
    )
    _require_integer(
        generator["bytes"],
        label="completion-attestation generator byte count",
        minimum=1,
    )
    _require_sha256(
        generator["sha256"],
        label="completion-attestation generator digest",
    )
    if generator != {
        "bytes": ATTESTOR_BYTES,
        "path": ATTESTOR_PATH,
        "sha256": ATTESTOR_SHA256,
    }:
        message = "Completion attestation was not created by the released attestor."
        raise ValueError(message)
    scope = _require_exact_dict(
        payload["scope"],
        {
            "interpretation",
            "prohibited_operations",
            "run",
            "task_validation_boundary",
            "validated_operations",
        },
        label="completion-attestation scope",
    )
    prohibited = scope["prohibited_operations"]
    if (
        scope["run"] != ATTESTATION_SCOPE_RUN
        or scope["interpretation"] != "strictly non-inferential"
        or scope["task_validation_boundary"] != ATTESTATION_TASK_VALIDATION_BOUNDARY
        or scope["validated_operations"] != ATTESTATION_VALIDATED_OPERATIONS
        or prohibited != list(ATTESTATION_PROHIBITED_OPERATIONS)
    ):
        message = "Completion attestation is not result-blind."
        raise ValueError(message)
    frozen = _require_exact_dict(
        payload["frozen_run"],
        {
            "analysis",
            "implementation_sha256",
            "mutsig_root",
            "outer_git_clean",
            "outer_git_head",
            "run_manifest_path",
            "run_manifest_sha256",
            "run_root",
            "run_schema_version",
            "source_root",
            "top_k",
        },
        label="completion-attestation frozen run",
    )
    _require_integer(
        frozen["top_k"],
        label="completion-attestation frozen top K",
        minimum=1,
    )
    if (
        frozen["analysis"] != "tcga-revision-k500"
        or frozen["top_k"] != 500
        or frozen["outer_git_head"] != SOURCE_A_COMMIT
        or frozen["outer_git_clean"] is not True
        or frozen["run_manifest_path"] != "run_manifest.json"
        or frozen["run_manifest_sha256"] != run_sha256
        or frozen["implementation_sha256"] != source_snapshot
        or frozen["run_schema_version"] != run_manifest["schema_version"]
        or frozen["source_root"] != run_manifest["source_root"]
        or frozen["mutsig_root"] != run_manifest["mutsig_root"]
    ):
        message = "Completion attestation does not bind the exact frozen run/source."
        raise ValueError(message)
    _require_metadata_path(frozen["run_root"], label="attested K500 run root")
    sealed = _require_exact_dict(
        payload["sealed_completion"],
        {"bytes", "path", "sha256"},
        label="completion-attestation sealed-completion receipt",
    )
    _require_integer(
        sealed["bytes"],
        label="completion-attestation sealed-completion byte count",
        minimum=1,
    )
    _require_sha256(
        sealed["sha256"],
        label="completion-attestation sealed-completion digest",
    )
    if sealed != {
        "bytes": sealed_bytes,
        "path": "sealed_completion_manifest.json",
        "sha256": sealed_sha256,
    }:
        message = "Completion attestation does not bind the supplied runner seal."
        raise ValueError(message)
    contracts = _validate_attested_contracts(
        payload["contracts"],
        sealed_manifest,
    )
    elapsed_by_bmr, peak_by_bmr = _validate_attested_tasks(
        payload["tasks"],
        sealed_manifest,
        contracts,
    )
    task_count = len(TCGA_COHORTS) * len(BMRS)
    _validate_attested_attempts(payload["attempts"], task_count=task_count)
    _validate_attested_resources(
        payload["resources"],
        runner_resource_policy=run_manifest["resource_policy"],
        elapsed_by_bmr=elapsed_by_bmr,
        peak_by_bmr=peak_by_bmr,
    )
    completion = _require_exact_dict(
        payload["completion"],
        {
            "backgrounds_expected",
            "candidate_pairs_all_tasks",
            "candidate_pairs_per_background",
            "cohorts_expected",
            "cohorts_validated",
            "features_per_task",
            "ordered_sample_memberships_across_cohorts",
            "same_base_pairs_excluded_per_background",
            "tasks_expected",
            "tasks_validated",
        },
        label="completion-attestation completion summary",
    )
    for key, value in completion.items():
        _require_integer(
            value,
            label=f"completion-attestation summary {key}",
            minimum=0,
        )
    expected_completion = {
        "backgrounds_expected": len(BMRS),
        "candidate_pairs_all_tasks": sum(
            record["pairs_per_background"] for record in contracts.values()
        )
        * len(BMRS),
        "candidate_pairs_per_background": sum(
            record["pairs_per_background"] for record in contracts.values()
        ),
        "cohorts_expected": len(TCGA_COHORTS),
        "cohorts_validated": len(TCGA_COHORTS),
        "features_per_task": 500,
        "ordered_sample_memberships_across_cohorts": sum(
            record["samples"] for record in contracts.values()
        ),
        "same_base_pairs_excluded_per_background": sum(
            record["same_base_pairs_excluded"] for record in contracts.values()
        ),
        "tasks_expected": task_count,
        "tasks_validated": task_count,
    }
    if completion != expected_completion:
        message = "Completion attestation does not certify the exact 32-by-3 grid."
        raise ValueError(message)
    _validate_attested_inventory(
        payload["pre_attestation_inventory"],
        known_receipts=_known_inventory_receipts(
            sealed_manifest,
            sealed_sha256=sealed_sha256,
            sealed_bytes=sealed_bytes,
            run_sha256=run_sha256,
            run_bytes=run_bytes,
        ),
    )
    return payload_digest


def _validate_authority(authority: object) -> tuple[dict[str, Any], dict[str, str]]:
    value = _require_exact_dict(
        authority,
        {
            "canonical_input_root",
            "configured",
            "expected_canonical_input_sha256",
            "expected_fit_approval_sha256",
            "expected_input_approval_sha256",
            "fit_approval_manifest",
            "input_approval_manifest",
            "provider_input",
        },
        label="validated run authority",
    )
    provider = _require_exact_dict(
        value["provider_input"],
        {
            "association_outputs_opened",
            "cohort_provider_receipts_sha256",
            "contract",
            "expected_manifest_sha256",
            "full_acceptance_receipt",
            "full_acceptance_receipt_sha256",
            "git_executable",
            "manifest",
            "root",
        },
        label="validated provider authority",
    )
    full_acceptance = _require_exact_dict(
        provider["full_acceptance_receipt"],
        {
            "association_outputs_opened",
            "authority_sha256",
            "cohort_receipts_sha256",
            "contract",
            "execution_snapshot",
            "full_inventory_validated",
            "provider_manifest_sha256",
            "schema_version",
        },
        label="provider full-acceptance receipt",
    )
    canonical_root = _require_metadata_path(
        value["canonical_input_root"],
        label="canonical-input root",
    )
    _require_metadata_path(
        value["input_approval_manifest"],
        label="materialization approval manifest",
    )
    _require_metadata_path(
        value["fit_approval_manifest"],
        label="fit approval manifest",
    )
    provider_root = _require_metadata_path(
        provider["root"],
        label="provider-input root",
    )
    if provider_root == canonical_root:
        message = "Canonical-input and provider-input roots must be distinct."
        raise ValueError(message)
    for record_name in ("git_executable", "manifest"):
        record = _require_exact_dict(
            provider[record_name],
            {"bytes", "path", "sha256"},
            label=f"provider {record_name} record",
        )
        _require_sha256(record["sha256"], label=f"provider {record_name} digest")
        _require_integer(
            record["bytes"],
            label=f"provider {record_name} byte count",
            minimum=1,
        )
        _require_metadata_path(
            record["path"],
            label=f"provider {record_name} path",
        )
    canonical_digest = _require_sha256(
        value["expected_canonical_input_sha256"],
        label="canonical-input manifest digest",
    )
    materialization_digest = _require_sha256(
        value["expected_input_approval_sha256"],
        label="materialization approval digest",
    )
    fit_digest = _require_sha256(
        value["expected_fit_approval_sha256"],
        label="fit approval digest",
    )
    provider_manifest_digest = _require_sha256(
        provider["expected_manifest_sha256"],
        label="provider-input manifest digest",
    )
    provider_acceptance_digest = _require_sha256(
        provider["full_acceptance_receipt_sha256"],
        label="provider full-acceptance digest",
    )
    provider_cohort_digest = _require_sha256(
        provider["cohort_provider_receipts_sha256"],
        label="provider cohort-receipt sequence digest",
    )
    _require_sha256(
        full_acceptance["authority_sha256"],
        label="provider full-acceptance authority digest",
    )
    full_acceptance_cohort_digest = _require_sha256(
        full_acceptance["cohort_receipts_sha256"],
        label="provider full-acceptance cohort-receipt digest",
    )
    execution_snapshot = _require_exact_dict(
        full_acceptance["execution_snapshot"],
        {
            "directory_count",
            "file_count",
            "individual_file_receipts_omitted",
            "root",
            "tree_hash_contract",
            "tree_sha256",
        },
        label="provider full-acceptance execution snapshot",
    )
    _require_integer(
        execution_snapshot["directory_count"],
        label="provider execution-snapshot directory count",
        minimum=1,
    )
    _require_integer(
        execution_snapshot["file_count"],
        label="provider execution-snapshot file count",
        minimum=1,
    )
    _require_sha256(
        execution_snapshot["tree_sha256"],
        label="provider execution-snapshot tree digest",
    )
    snapshot_root = execution_snapshot["root"]
    tree_contract = execution_snapshot["tree_hash_contract"]
    if (
        not isinstance(snapshot_root, str)
        or snapshot_root
        != (f"_orchestration/execution-snapshot-{execution_snapshot['tree_sha256']}")
        or PurePosixPath(snapshot_root).is_absolute()
        or ".." in PurePosixPath(snapshot_root).parts
        or tree_contract != PROVIDER_TREE_HASH_CONTRACT
        or execution_snapshot["individual_file_receipts_omitted"] is not True
    ):
        message = "Provider full-acceptance execution snapshot is invalid."
        raise ValueError(message)
    if (
        value["configured"] is not True
        or provider["association_outputs_opened"] is not False
        or provider["contract"] != PROVIDER_INPUT_CONTRACT
        or provider["manifest"]["sha256"] != provider_manifest_digest
        or Path(provider["manifest"]["path"])
        != provider_root / "provider_input_manifest.json"
        or full_acceptance["contract"] != PROVIDER_FULL_ACCEPTANCE_CONTRACT
        or full_acceptance["schema_version"] != PROVIDER_SCHEMA_VERSION
        or full_acceptance["provider_manifest_sha256"] != provider_manifest_digest
        or provider_cohort_digest != full_acceptance_cohort_digest
        or full_acceptance["full_inventory_validated"] is not True
        or full_acceptance["association_outputs_opened"] is not False
        or _json_sha256(full_acceptance, newline=True) != provider_acceptance_digest
    ):
        message = "Validated run authority is not internally cross-bound."
        raise ValueError(message)
    digests = {
        "canonical_input_manifest_sha256": canonical_digest,
        "fit_approval_sha256": fit_digest,
        "materialization_approval_sha256": materialization_digest,
        "provider_full_acceptance_receipt_sha256": provider_acceptance_digest,
        "provider_input_manifest_sha256": provider_manifest_digest,
        "validated_run_authority_sha256": _json_sha256(value),
    }
    if set(digests) != set(AUTHORITY_DIGEST_FIELDS):
        message = "Authority projection does not contain exactly six digests."
        raise RuntimeError(message)
    return value, digests


def _validate_seal_and_run(
    sealed: dict[str, Any],
    run: dict[str, Any],
    *,
    run_sha256: str,
    run_bytes: int,
    source_snapshot: dict[str, str],
) -> dict[str, str]:
    _require_exact_dict(
        sealed,
        {
            "analysis",
            "authority",
            "bmrs",
            "cohorts",
            "contract",
            "contracts",
            "downstream_binding",
            "grid",
            "result_rows_opened",
            "run_manifest",
            "schema",
            "tasks",
            "top_k",
        },
        label="sealed-completion manifest",
    )
    run_keys = {
        "analysis",
        "bmrs",
        "cohorts",
        "created_at_utc",
        "feature_policy",
        "git",
        "implementation_sha256",
        "mutsig_root",
        "observation_support_universe",
        "required_contingency_table_contract",
        "required_gene_support_contract",
        "required_log_odds_ratio_contract",
        "required_lrt_contract",
        "required_lrt_nestedness_tolerance",
        "required_output_recomputation_atol",
        "required_pair_effect_identifiability_contract",
        "required_pair_fit_contract",
        "required_pair_fit_kkt_tolerance",
        "required_pair_fit_max_iterations",
        "required_pair_identifiability_relative_tolerance",
        "required_pair_simplex_tolerance",
        "required_rho_contract",
        "resource_policy",
        "revision_authority",
        "same_base_pair_policy",
        "sample_axis_contract",
        "schema_version",
        "signed_tested_family",
        "source_root",
        "tested_family_implementation",
        "top_k",
        "undefined_rho_lrt_tolerance",
    }
    _require_exact_dict(run, run_keys, label="K500 run manifest")
    git = _require_exact_dict(
        run["git"],
        {"dirty", "executable", "head", "status", "version"},
        label="K500 run Git receipt",
    )
    git_executable = _require_exact_dict(
        git["executable"],
        {"bytes", "path", "sha256"},
        label="K500 run Git-executable receipt",
    )
    _require_integer(
        git_executable["bytes"],
        label="K500 run Git-executable byte count",
        minimum=1,
    )
    _require_sha256(
        git_executable["sha256"],
        label="K500 run Git-executable digest",
    )
    _require_metadata_path(
        git_executable["path"],
        label="K500 run Git-executable path",
    )
    grid = _require_exact_dict(
        sealed["grid"],
        {"ordered_coordinates_sha256", "task_count"},
        label="sealed-completion grid",
    )
    _require_integer(
        grid["task_count"],
        label="sealed-completion task count",
        minimum=1,
    )
    _require_sha256(
        grid["ordered_coordinates_sha256"],
        label="sealed-completion coordinate digest",
    )
    run_receipt = _require_exact_dict(
        sealed["run_manifest"],
        {"bytes", "sha256"},
        label="sealed-completion run-manifest receipt",
    )
    _require_integer(
        run_receipt["bytes"],
        label="sealed-completion run-manifest byte count",
        minimum=1,
    )
    _require_sha256(
        run_receipt["sha256"],
        label="sealed-completion run-manifest digest",
    )
    downstream = _require_exact_dict(
        sealed["downstream_binding"],
        {"field", "stage"},
        label="sealed-completion downstream binding",
    )
    coordinates = tuple(f"{cohort}/{bmr}" for cohort in TCGA_COHORTS for bmr in BMRS)
    _require_integer(sealed["top_k"], label="sealed-completion top K", minimum=1)
    _require_integer(run["top_k"], label="K500 run top K", minimum=1)
    contract_receipts = sealed["contracts"]
    task_receipts = sealed["tasks"]
    if not isinstance(contract_receipts, list) or not isinstance(task_receipts, list):
        message = "Sealed-completion contract/task receipts are invalid."
        raise TypeError(message)
    observed_contract_cohorts: list[str] = []
    contract_sha256_by_cohort: dict[str, str] = {}
    for receipt in contract_receipts:
        contract_receipt = _require_exact_dict(
            receipt,
            {"bytes", "cohort", "contract_sha256", "file_sha256"},
            label="sealed-completion contract receipt",
        )
        _require_integer(
            contract_receipt["bytes"],
            label="sealed-completion contract byte count",
            minimum=1,
        )
        _require_sha256(
            contract_receipt["contract_sha256"],
            label="sealed-completion contract digest",
        )
        _require_sha256(
            contract_receipt["file_sha256"],
            label="sealed-completion contract-file digest",
        )
        observed_contract_cohorts.append(contract_receipt["cohort"])
        contract_sha256_by_cohort[contract_receipt["cohort"]] = contract_receipt[
            "contract_sha256"
        ]
    observed_task_coordinates: list[str] = []
    for receipt in task_receipts:
        task_receipt = _require_exact_dict(
            receipt,
            {
                "bmr",
                "cohort",
                "consumed_input_sha256",
                "contract_sha256",
                "pairwise_interaction_results",
                "single_gene_results",
                "task_manifest",
            },
            label="sealed-completion task receipt",
        )
        for key in (
            "pairwise_interaction_results",
            "single_gene_results",
            "task_manifest",
        ):
            file_receipt = _require_exact_dict(
                task_receipt[key],
                {"bytes", "sha256"},
                label=f"sealed-completion {key} receipt",
            )
            _require_integer(
                file_receipt["bytes"],
                label=f"sealed-completion {key} byte count",
                minimum=1,
            )
            _require_sha256(
                file_receipt["sha256"],
                label=f"sealed-completion {key} digest",
            )
        _require_sha256(
            task_receipt["contract_sha256"],
            label="sealed-completion task contract digest",
        )
        consumed = task_receipt["consumed_input_sha256"]
        if not isinstance(consumed, dict) or not consumed:
            message = "Sealed-completion consumed-input receipt is invalid."
            raise TypeError(message)
        for key, digest in consumed.items():
            if not isinstance(key, str) or not key:
                message = "Sealed-completion consumed-input key is invalid."
                raise ValueError(message)
            _require_sha256(
                digest,
                label="sealed-completion consumed-input digest",
            )
        if (
            task_receipt["cohort"] in contract_sha256_by_cohort
            and task_receipt["contract_sha256"]
            != contract_sha256_by_cohort[task_receipt["cohort"]]
        ):
            message = "Sealed-completion task lost its cohort-contract binding."
            raise ValueError(message)
        observed_task_coordinates.append(
            f"{task_receipt['cohort']}/{task_receipt['bmr']}",
        )
    if (
        sealed["schema"] != SEALED_COMPLETION_SCHEMA
        or sealed["contract"] != SEALED_COMPLETION_CONTRACT
        or sealed["analysis"] != "tcga-revision-k500"
        or sealed["top_k"] != 500
        or sealed["cohorts"] != list(TCGA_COHORTS)
        or sealed["bmrs"] != list(BMRS)
        or sealed["result_rows_opened"] is not False
        or downstream
        != {
            "field": "upstream_result_manifest_sha256",
            "stage": "inspect-tcga-k500",
        }
        or grid["task_count"] != len(coordinates)
        or grid["ordered_coordinates_sha256"] != _sequence_sha256(coordinates)
        or observed_contract_cohorts != list(TCGA_COHORTS)
        or observed_task_coordinates != list(coordinates)
        or run_receipt != {"bytes": run_bytes, "sha256": run_sha256}
        or run["analysis"] != "tcga-revision-k500"
        or run["schema_version"] != "3.0.0"
        or run["top_k"] != 500
        or run["cohorts"] != list(TCGA_COHORTS)
        or run["bmrs"] != list(BMRS)
        or run["implementation_sha256"] != source_snapshot
        or run["signed_tested_family"] != EXPECTED_TESTED_FAMILY
        or run["tested_family_implementation"] != EXPECTED_TESTED_FAMILY
        or any(run[key] != value for key, value in EXPECTED_RUN_CONTRACTS.items())
        or git["head"] != SOURCE_A_COMMIT
        or git["dirty"] is not False
        or git["status"] != []
    ):
        message = "Seal/run metadata does not bind the exact result-blind K500 grid."
        raise ValueError(message)
    run_authority, digests = _validate_authority(run["revision_authority"])
    if sealed["authority"] != run_authority:
        message = "Sealed completion and run manifest authority records differ."
        raise ValueError(message)
    provider = run_authority["provider_input"]
    provider_root = Path(provider["root"])
    source_root = _require_metadata_path(
        run["source_root"],
        label="K500 run source root",
    )
    mutsig_root = _require_metadata_path(
        run["mutsig_root"],
        label="K500 run MutSig root",
    )
    if (
        source_root != provider_root / "cohorts"
        or mutsig_root != provider_root / "mutsig"
        or git_executable != provider["git_executable"]
    ):
        message = "K500 run paths or Git receipt lost the provider authority."
        raise ValueError(message)
    return digests


def _projection_payload(  # noqa: PLR0913
    *,
    source_boundary: dict[str, Any],
    completion_attestation_sha256: str,
    completion_attestation_payload_sha256: str,
    sealed_completion_sha256: str,
    run_manifest_sha256: str,
    authority_digests: dict[str, str],
) -> dict[str, Any]:
    return {
        "binding": {
            "completion_attestation_payload_sha256": (
                completion_attestation_payload_sha256
            ),
            "completion_attestation_sha256": completion_attestation_sha256,
            "run_manifest_sha256": run_manifest_sha256,
            "sealed_completion_sha256": sealed_completion_sha256,
        },
        "builder": source_boundary["builder"],
        "revision_authority": authority_digests,
        "schema": PROJECTION_SCHEMA,
        "source": source_boundary["source"],
    }


def _publish_no_replace(
    output_path: Path,
    content: bytes,
    *,
    protected: tuple[_PinnedPath, ...],
) -> _PinnedPath:
    output = _require_absolute_path(output_path, label="projection output")
    _require_safe_basename(output.name, label="projection output")
    if len(content) > MAX_PROJECTION_BYTES:
        message = (
            "Canonical K500 authority projection exceeds its "
            f"{MAX_PROJECTION_BYTES}-byte safety limit."
        )
        raise ValueError(message)
    parent = _PinnedDirectory.open(
        output.parent,
        label="projection output parent",
        require_stable_metadata=False,
    )
    temporary_name = f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    temporary_created = False
    temporary_fd: int | None = None
    published: _PinnedPath | None = None
    published_transferred = False
    destination_linked = False
    try:
        if os.path.lexists(output):
            message = f"Refusing to replace existing projection: {output}"
            raise FileExistsError(message)
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent.descriptor,
        )
        temporary_created = True
        offset = 0
        while offset < len(content):
            written = os.write(temporary_fd, content[offset:])
            if written <= 0:
                message = "Projection staging write made no forward progress."
                raise OSError(message)
            offset += written
        os.fchmod(temporary_fd, 0o444)
        os.fsync(temporary_fd)
        staged = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 1
            or staged.st_size != len(content)
        ):
            message = "Projection staging file is not one private regular file."
            raise ValueError(message)
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=parent.descriptor,
                dst_dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            destination_linked = True
        except OSError as error:
            if error.errno == errno.EEXIST:
                message = f"Refusing to replace racing projection: {output}"
                raise FileExistsError(message) from error
            raise
        linked = os.fstat(temporary_fd)
        if not stat.S_ISREG(linked.st_mode) or linked.st_nlink != 2:
            visible_destination = os.stat(
                output.name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if (
                visible_destination.st_dev == linked.st_dev
                and visible_destination.st_ino == linked.st_ino
            ):
                os.unlink(output.name, dir_fd=parent.descriptor)
                destination_linked = False
            message = "Projection staging link count changed during publication."
            raise ValueError(message)
        os.unlink(temporary_name, dir_fd=parent.descriptor)
        temporary_created = False
        after_unlink = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(after_unlink.st_mode)
            or after_unlink.st_nlink != 1
            or stat.S_IMODE(after_unlink.st_mode) != 0o444
        ):
            message = "Published projection did not close to one private link."
            raise ValueError(message)
        if (
            _read_descriptor(
                temporary_fd,
                max_bytes=MAX_PROJECTION_BYTES,
                label="projection staging file",
            )
            != content
        ):
            message = "Projection staging bytes changed during publication."
            raise ValueError(message)
        os.fsync(parent.descriptor)
        published = _PinnedPath.open(
            output,
            label="published projection",
            max_bytes=MAX_PROJECTION_BYTES,
        )
        if (
            published.content != content
            or stat.S_IMODE(os.fstat(published.descriptor).st_mode) != 0o444
        ):
            message = "Published projection bytes differ from canonical payload."
            raise ValueError(message)
        for item in protected:
            item.require_unchanged(label=f"protected input {item.path.name}")
        parent.require_unchanged(label="projection output parent")
        published.require_unchanged(label="published projection")
        published_transferred = True
        return published
    finally:
        if published is not None and not published_transferred:
            published.close()
        if (
            destination_linked
            and not published_transferred
            and temporary_fd is not None
        ):
            with suppress(FileNotFoundError):
                visible_destination = os.stat(
                    output.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                opened_destination = os.fstat(temporary_fd)
                if (
                    visible_destination.st_dev == opened_destination.st_dev
                    and visible_destination.st_ino == opened_destination.st_ino
                ):
                    os.unlink(output.name, dir_fd=parent.descriptor)
                    os.fsync(parent.descriptor)
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent.descriptor)
        parent.close()


def _receipt(
    projection_path: Path,
    projection: dict[str, Any],
    raw: bytes,
) -> K500AuthorityProjectionReceipt:
    source = projection["source"]
    binding = projection["binding"]
    digests = projection["revision_authority"]
    return K500AuthorityProjectionReceipt(
        projection_path=projection_path,
        projection_sha256=hashlib.sha256(raw).hexdigest(),
        completion_attestation_sha256=binding["completion_attestation_sha256"],
        completion_attestation_payload_sha256=(
            binding["completion_attestation_payload_sha256"]
        ),
        sealed_completion_sha256=binding["sealed_completion_sha256"],
        run_manifest_sha256=binding["run_manifest_sha256"],
        source_a_commit=source["source_a_commit"],
        release_b_commit=source["release_b_commit"],
        release_tag=source["release_tag"],
        git_blob_count=source["git_blob_count"],
        generated_file_count=source["generated_file_count"],
        snapshot_file_count=source["snapshot_file_count"],
        execution_snapshot_sha256=source["execution_snapshot_sha256"],
        authority_digests=MappingProxyType(dict(digests)),
        authority_digest_count=len(digests),
    )


def build_k500_authority_projection(  # noqa: PLR0913
    *,
    completion_attestation_path: Path,
    expected_completion_attestation_sha256: str,
    sealed_completion_path: Path,
    expected_sealed_completion_sha256: str,
    run_manifest_path: Path,
    expected_run_manifest_sha256: str,
    repo_root: Path,
    release_b_commit: str,
    release_tag: str,
    output_path: Path,
    git_executable: Path = GIT_EXECUTABLE_PATH,
) -> K500AuthorityProjectionReceipt:
    """Build one immutable result-blind projection from anchored metadata."""
    expected_attestation = _require_sha256(
        expected_completion_attestation_sha256,
        label="expected completion-attestation digest",
    )
    expected_seal = _require_sha256(
        expected_sealed_completion_sha256,
        label="expected sealed-completion digest",
    )
    expected_run = _require_sha256(
        expected_run_manifest_sha256,
        label="expected run-manifest digest",
    )
    opened_inputs: list[_PinnedPath] = []
    try:
        opened_inputs.append(
            _PinnedPath.open(
                completion_attestation_path,
                label="completion attestation",
                max_bytes=MAX_COMPLETION_ATTESTATION_BYTES,
            ),
        )
        opened_inputs.append(
            _PinnedPath.open(
                sealed_completion_path,
                label="sealed completion",
                max_bytes=MAX_SEALED_COMPLETION_BYTES,
            ),
        )
        opened_inputs.append(
            _PinnedPath.open(
                run_manifest_path,
                label="run manifest",
                max_bytes=MAX_RUN_MANIFEST_BYTES,
            ),
        )
        opened_inputs.append(
            _PinnedPath.open(
                Path(__file__).absolute(),
                label="projection builder",
                max_bytes=MAX_BUILDER_BYTES,
            ),
        )
        inputs = tuple(opened_inputs)
        attestation_file, seal_file, run_file, builder = inputs
        observed = tuple(
            hashlib.sha256(item.content).hexdigest() for item in inputs[:3]
        )
        if observed != (expected_attestation, expected_seal, expected_run):
            message = "One or more independently anchored metadata digests differ."
            raise ValueError(message)
        attestation = _parse_json(
            attestation_file.content,
            label="completion attestation",
        )
        seal = _parse_json(seal_file.content, label="sealed completion")
        run = _parse_json(run_file.content, label="run manifest")
        for label, pinned, payload in (
            ("completion attestation", attestation_file, attestation),
            ("sealed completion", seal_file, seal),
            ("run manifest", run_file, run),
        ):
            if pinned.content != _canonical_json(payload) + b"\n":
                message = f"{label} is not exact canonical newline-terminated JSON."
                raise ValueError(message)
        git_receipt = _anchored_git_receipt(run)
        source_boundary, source_snapshot = _source_boundary(
            _require_absolute_path(repo_root, label="release repository root"),
            _require_absolute_path(git_executable, label="Git executable"),
            release_b_commit=release_b_commit,
            release_tag=release_tag,
            builder=builder,
            anchored_git_receipt=git_receipt,
        )
        authority_digests = _validate_seal_and_run(
            seal,
            run,
            run_sha256=expected_run,
            run_bytes=len(run_file.content),
            source_snapshot=source_snapshot,
        )
        payload_digest = _validate_attestation(
            attestation,
            sealed_sha256=expected_seal,
            sealed_bytes=len(seal_file.content),
            run_sha256=expected_run,
            run_bytes=len(run_file.content),
            source_snapshot=source_snapshot,
            sealed_manifest=seal,
            run_manifest=run,
        )
        projection = _projection_payload(
            source_boundary=source_boundary,
            completion_attestation_sha256=expected_attestation,
            completion_attestation_payload_sha256=payload_digest,
            sealed_completion_sha256=expected_seal,
            run_manifest_sha256=expected_run,
            authority_digests=authority_digests,
        )
        content = _canonical_json(projection) + b"\n"
        published = _publish_no_replace(output_path, content, protected=inputs)
        try:
            final_boundary, final_snapshot = _source_boundary(
                _require_absolute_path(repo_root, label="release repository root"),
                _require_absolute_path(git_executable, label="Git executable"),
                release_b_commit=release_b_commit,
                release_tag=release_tag,
                builder=builder,
                anchored_git_receipt=git_receipt,
            )
            if final_boundary != source_boundary or final_snapshot != source_snapshot:
                message = (
                    "Release source boundary changed during projection publication."
                )
                raise ValueError(message)
            for item in inputs:
                item.require_unchanged(
                    label=f"terminal protected input {item.path.name}",
                )
            published.require_unchanged(label="terminal published projection")
            return _receipt(published.path, projection, published.content)
        finally:
            published.close()
    finally:
        for item in reversed(opened_inputs):
            item.close()


def _validate_projection_schema(payload: dict[str, Any]) -> None:
    _require_exact_dict(
        payload,
        {"binding", "builder", "revision_authority", "schema", "source"},
        label="K500 authority projection",
    )
    if payload["schema"] != PROJECTION_SCHEMA:
        message = "K500 authority projection schema is invalid."
        raise ValueError(message)
    builder = _require_exact_dict(
        payload["builder"],
        {"bytes", "path", "sha256"},
        label="projection builder record",
    )
    if builder["path"] != BUILDER_PATH:
        message = "Projection builder record is invalid."
        raise ValueError(message)
    _require_integer(builder["bytes"], label="projection builder bytes", minimum=1)
    _require_sha256(builder["sha256"], label="projection builder digest")
    source = _require_exact_dict(
        payload["source"],
        {
            "execution_snapshot_sha256",
            "generated_file_count",
            "generated_path",
            "generated_sha256",
            "git_executable",
            "git_blob_count",
            "release_b_commit",
            "release_tag",
            "snapshot_file_count",
            "source_a_commit",
        },
        label="projection source",
    )
    for key in ("generated_file_count", "git_blob_count", "snapshot_file_count"):
        _require_integer(
            source[key],
            label=f"projection source {key}",
            minimum=1,
        )
    git_receipt = _require_exact_dict(
        source["git_executable"],
        {"bytes", "path", "sha256"},
        label="projection Git-executable receipt",
    )
    _require_integer(
        git_receipt["bytes"],
        label="projection Git-executable byte count",
        minimum=1,
    )
    projected_git_path = _require_metadata_path(
        git_receipt["path"],
        label="projection Git-executable path",
    )
    if projected_git_path != GIT_EXECUTABLE_PATH:
        message = "Projection Git executable must be exactly /usr/bin/git."
        raise ValueError(message)
    _require_sha256(
        git_receipt["sha256"],
        label="projection Git-executable digest",
    )
    if (
        source["source_a_commit"] != SOURCE_A_COMMIT
        or source["git_blob_count"] != len(GIT_EXECUTION_PATHS)
        or source["generated_file_count"] != 1
        or source["snapshot_file_count"] != len(GIT_EXECUTION_PATHS) + 1
        or source["generated_path"] != GENERATED_VERSION_PATH
        or source["generated_sha256"] != GENERATED_VERSION_SHA256
        or source["execution_snapshot_sha256"] != EXPECTED_EXECUTION_SNAPSHOT_SHA256
    ):
        message = "Projection source closure is invalid."
        raise ValueError(message)
    _require_commit(source["release_b_commit"], label="projection release B")
    _require_release_tag(source["release_tag"])
    binding = _require_exact_dict(
        payload["binding"],
        {
            "completion_attestation_payload_sha256",
            "completion_attestation_sha256",
            "run_manifest_sha256",
            "sealed_completion_sha256",
        },
        label="projection metadata binding",
    )
    for key, value in binding.items():
        _require_sha256(value, label=f"projection binding {key}")
    digests = _require_exact_dict(
        payload["revision_authority"],
        set(AUTHORITY_DIGEST_FIELDS),
        label="six-digest revision authority projection",
    )
    for key, value in digests.items():
        _require_sha256(value, label=f"authority digest {key}")


def validate_k500_authority_projection(
    projection_path: Path,
    *,
    expected_projection_sha256: str,
    repo_root: Path,
    git_executable: Path = GIT_EXECUTABLE_PATH,
) -> K500AuthorityProjectionReceipt:
    """Validate an anchored projection without reopening any run metadata."""
    expected = _require_sha256(
        expected_projection_sha256,
        label="expected K500 authority-projection digest",
    )
    projection_file = _PinnedPath.open(
        projection_path,
        label="K500 authority projection",
        max_bytes=MAX_PROJECTION_BYTES,
    )
    builder: _PinnedPath | None = None
    try:
        if stat.S_IMODE(os.fstat(projection_file.descriptor).st_mode) != 0o444:
            message = "K500 authority projection is not immutable read-only metadata."
            raise ValueError(message)
        builder = _PinnedPath.open(
            Path(__file__).absolute(),
            label="projection builder",
            max_bytes=MAX_BUILDER_BYTES,
        )
        if hashlib.sha256(projection_file.content).hexdigest() != expected:
            message = "K500 authority projection differs from its independent digest."
            raise ValueError(message)
        projection = _parse_json(
            projection_file.content,
            label="K500 authority projection",
        )
        if projection_file.content != _canonical_json(projection) + b"\n":
            message = "K500 authority projection is not exact canonical JSON."
            raise ValueError(message)
        _validate_projection_schema(projection)
        boundary, _ = _source_boundary(
            _require_absolute_path(repo_root, label="release repository root"),
            _require_absolute_path(git_executable, label="Git executable"),
            release_b_commit=projection["source"]["release_b_commit"],
            release_tag=projection["source"]["release_tag"],
            builder=builder,
            anchored_git_receipt=projection["source"]["git_executable"],
        )
        if (
            projection["builder"] != boundary["builder"]
            or projection["source"] != boundary["source"]
        ):
            message = "Projection source/builder no longer matches release B."
            raise ValueError(message)
        projection_file.require_unchanged(label="K500 authority projection")
        builder.require_unchanged(label="projection builder")
        return _receipt(projection_file.path, projection, projection_file.content)
    finally:
        if builder is not None:
            builder.close()
        projection_file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--completion-attestation", type=Path, required=True)
    build.add_argument("--expected-completion-attestation-sha256", required=True)
    build.add_argument("--sealed-completion", type=Path, required=True)
    build.add_argument("--expected-sealed-completion-sha256", required=True)
    build.add_argument("--run-manifest", type=Path, required=True)
    build.add_argument("--expected-run-manifest-sha256", required=True)
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--release-b-commit", required=True)
    build.add_argument("--release-tag", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--git-executable", type=Path, default=GIT_EXECUTABLE_PATH)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--projection", type=Path, required=True)
    validate.add_argument("--expected-projection-sha256", required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--git-executable", type=Path, default=GIT_EXECUTABLE_PATH)
    return parser


def main() -> None:
    """Run the result-blind projection builder or validator."""
    args = _parser().parse_args()
    if args.command == "build":
        receipt = build_k500_authority_projection(
            completion_attestation_path=args.completion_attestation,
            expected_completion_attestation_sha256=(
                args.expected_completion_attestation_sha256
            ),
            sealed_completion_path=args.sealed_completion,
            expected_sealed_completion_sha256=args.expected_sealed_completion_sha256,
            run_manifest_path=args.run_manifest,
            expected_run_manifest_sha256=args.expected_run_manifest_sha256,
            repo_root=args.repo_root,
            release_b_commit=args.release_b_commit,
            release_tag=args.release_tag,
            output_path=args.output,
            git_executable=args.git_executable,
        )
    else:
        receipt = validate_k500_authority_projection(
            args.projection,
            expected_projection_sha256=args.expected_projection_sha256,
            repo_root=args.repo_root,
            git_executable=args.git_executable,
        )
    sys.stdout.write(f"{json.dumps(receipt.as_dict(), indent=2, sort_keys=True)}\n")


if __name__ == "__main__":
    main()

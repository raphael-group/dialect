"""Run the frozen, native-support TCGA K=500 revision analysis.

This runner is deliberately independent of the historical ``id_*`` output trees.
For each TCGA cohort it selects one shared, deterministic set of 500 mutation-event
features that is natively supported by CBaSE, DIG, and the per-sample MutSig lambda
tensor.  It then fits the exact same pair family under all three backgrounds while
excluding same-base-gene ``_M:_N`` pairs before fitting.

Scientific output is task-atomic and never overwritten.  A completed
``cohort/background`` task is resumed only after its manifest, hashes, ordered feature
axis, and complete ordered pair universe validate.  Failed and interrupted attempts
remain isolated under ``attempts`` / ``work`` for diagnosis.

The current repository must expose the required LRT, pair-fit, and observation-support
contracts. These intentional launch gates prevent expensive K=500 runs from using the
historical statistic, an uncertified optimizer, or silently unsupported observations.

Launch only after the LRT and signed canonical-MAF binding gates are implemented
and reviewed::

    PYTHONPATH=src /opt/anaconda3/envs/dialect/bin/python \
      -m analysis.run_tcga_revision_k500 \
      --provider-input-root <immutable-provider-root> \
      --expected-provider-input-manifest-sha256 <independent-sha256> \
      --canonical-input-root <immutable-canonical-root> \
      --expected-canonical-input-sha256 <independent-sha256> \
      --input-approval-manifest <d1-d2-approval.json> \
      --expected-input-approval-sha256 <independent-sha256> \
      --fit-approval-manifest <d1-d6-approval.json> \
      --expected-fit-approval-sha256 <independent-sha256> \
      --output-root <fresh-output-root> --jobs 2 --mutsig-jobs 1 --nice 10
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import re
import resource
import shutil
import stat as stat_module
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import poisson

from analysis.materialize_tcga_revision_inputs import (
    INPUT_CONTRACT as CANONICAL_INPUT_CONTRACT,
)
from analysis.materialize_tcga_revision_inputs import (
    build_full_input_validation_receipt,
    full_input_validation_receipt_sha256,
    materialized_cohort_binding,
    validate_materialized_input_bundle,
    validate_materialized_input_cohort_binding,
)
from analysis.materialize_tcga_revision_provider_inputs import (
    CHILD_PYTHON_EXECUTABLE as PROVIDER_CHILD_PYTHON_EXECUTABLE,
)
from analysis.materialize_tcga_revision_provider_inputs import (
    PROVIDER_INPUT_CONTRACT,
    full_acceptance_receipt_sha256,
    validate_materialized_provider_cohort_input,
    validate_materialized_provider_input_bundle,
)
from analysis.mutsig_lambda_co import (
    PRODUCTION_POISSON_NORMALIZATION,
    PRODUCTION_POISSON_STORAGE_CONTRACT,
    PRODUCTION_POISSON_SUPPORT_CONTRACT,
    PRODUCTION_POISSON_SUPPORT_RULE,
    PRODUCTION_POISSON_TAIL_TOLERANCE,
    build_native_poisson_pmfs,
    build_poisson_support_contract,
    estimate_native_poisson_pmf_storage,
    validate_poisson_support_contract,
)
from dialect.data.revision_approval import (
    APPROVAL_SCHEMA,
    DECISION_IDS,
    FIT_SEALED_TCGA_K500_STAGE,
    MATERIALIZE_FINAL_INPUTS_STAGE,
    STAGE_MINIMUM_DECISIONS,
    STAGE_SCOPED_APPROVAL_SCHEMA,
    DecisionApproval,
    RevisionApproval,
    validate_revision_approval,
)
from dialect.data.revision_fit_policy import (
    D4ImplementationContract,
    EffectIdentifiabilityImplementationContract,
    NumericalImplementationContract,
    RevisionFitPolicy,
    TestedFamilyPolicy,
    validate_revision_fit_policy,
)
from dialect.models import gene as gene_module
from dialect.models import interaction as interaction_module
from dialect.models.gene import (
    MARGINAL_FIT_BRACKET_WIDTH_TOL,
    MARGINAL_FIT_CONTRACT,
    MARGINAL_FIT_FIXED_POINT_TOL,
    MARGINAL_FIT_FLAT_TIE_BREAK,
    MARGINAL_FIT_KKT_TOL,
    MARGINAL_FIT_MAX_ITER,
    Gene,
)
from dialect.models.interaction import Interaction
from dialect.utils.identify import (
    SINGLE_GENE_COUNT_CONTRACT,
    SINGLE_GENE_RESULT_COLUMNS,
    create_single_gene_results,
    estimate_pi_for_each_gene,
    estimate_taus_for_each_interaction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import TextIO

# Version 3 is intentionally incompatible with observed-kmax-truncated MutSig PMFs.
SCHEMA_VERSION = "3.0.0"
TOP_K = 500
BMRS = ("cbase", "dig", "mutsig")
TESTED_FAMILY_FEATURE_RANKING = "descending-total-eligible-mutation-event-count"
TESTED_FAMILY_TIE_BREAK = "canonical-count-matrix-column-order"
TESTED_FAMILY_PROVIDER_SUPPORT = "shared-native-cbase-dig-mutsig"
TESTED_FAMILY_PAIR_CONSTRUCTION = "all-unordered-pairs-of-ordered-feature-axis"
TESTED_FAMILY_SAME_BASE_POLICY = "exclude-before-fitting-and-testing"
TESTED_FAMILY_NO_PRETEST_FILTER = "none"
TESTED_FAMILY_SCOPE = "one-complete-within-cohort-tested-pair-family"
REQUIRED_TESTED_FAMILY = TestedFamilyPolicy(
    top_k=TOP_K,
    feature_ranking=TESTED_FAMILY_FEATURE_RANKING,
    tie_break=TESTED_FAMILY_TIE_BREAK,
    provider_support=TESTED_FAMILY_PROVIDER_SUPPORT,
    pair_construction=TESTED_FAMILY_PAIR_CONSTRUCTION,
    same_base_missense_nonsense=TESTED_FAMILY_SAME_BASE_POLICY,
    epsilon_pretest_filter=TESTED_FAMILY_NO_PRETEST_FILTER,
    marginal_effect_pretest_filter=TESTED_FAMILY_NO_PRETEST_FILTER,
    family=TESTED_FAMILY_SCOPE,
)
SEALED_COMPLETION_SCHEMA = "dialect-tcga-k500-sealed-completion-v1"
SEALED_COMPLETION_CONTRACT = "metadata-hash-only-whole-grid-write-once-v1"
SEALED_COMPLETION_NAME = "sealed_completion_manifest.json"
MEMORY_HEAVY_MUTSIG_COHORTS = {"BRCA", "CRAD", "LGG", "SKCM", "UCEC"}
CANARY_COHORT = "CHOL"
REQUIRED_LRT_CONTRACT = "driver-independence-constrained-mle-v1"
REQUIRED_PAIR_FIT_CONTRACT = "deterministic-simplex-coordinate-ascent-total-kkt-v2"
REQUIRED_PAIR_FIT_KKT_TOL = 1e-8
REQUIRED_PAIR_FIT_MAX_ITER = 1000
REQUIRED_PAIR_SIMPLEX_TOL = 1e-12
REQUIRED_LRT_NESTEDNESS_TOL = 1e-8
REQUIRED_OUTPUT_RECOMPUTATION_ATOL = 1e-12
REQUIRED_PAIR_IDENTIFIABILITY_RTOL = 1e-12
REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT = (
    "full-affine-rank-relative-svd-1e-12-conservative-v1"
)
REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS = "full-affine-rank"
REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS = "rank-deficient"
REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS = "rank-not-certified-underflow"
REQUIRED_NONIDENTIFIED_EFFECT_BLANK_FIELDS = (
    "Tau_1X",
    "Tau_X1",
    "Rho",
    "Log Odds Ratio",
    "Wald Statistic",
)
REQUIRED_RHO_CONTRACT = "marshall-olkin-identifiable-finite-or-degenerate-null-v2"
REQUIRED_UNDEFINED_RHO_LRT_TOL = 1e-8
REQUIRED_CONTINGENCY_TABLE_CONTRACT = "observed-binary-cells-00-01-10-11-v1"
REQUIRED_LOG_ODDS_RATIO_CONTRACT = (
    "conventional-latent-odds-00x11-over-01x10-identifiable-v2"
)
OBSERVATION_SUPPORT_UNIVERSE = "full-observation-support-common-universe-v1"
REQUIRED_GENE_SUPPORT_CONTRACT = "latent-state-union-v1"
SAMPLE_AXIS_CONTRACT = "count-matrix-equals-authoritative-and-mutsig-patient-axis-v2"
REQUIRED_D4_IMPLEMENTATION = D4ImplementationContract(
    lrt_contract=REQUIRED_LRT_CONTRACT,
    numerical_implementation=NumericalImplementationContract(
        marginal_fit_contract=MARGINAL_FIT_CONTRACT,
        marginal_fit_max_iterations=MARGINAL_FIT_MAX_ITER,
        marginal_fit_total_kkt_tolerance=MARGINAL_FIT_KKT_TOL,
        marginal_fit_bracket_width_tolerance=MARGINAL_FIT_BRACKET_WIDTH_TOL,
        marginal_fit_fixed_point_tolerance=MARGINAL_FIT_FIXED_POINT_TOL,
        marginal_fit_flat_likelihood_tie_break=MARGINAL_FIT_FLAT_TIE_BREAK,
        pair_fit_contract=REQUIRED_PAIR_FIT_CONTRACT,
        pair_fit_max_iterations=REQUIRED_PAIR_FIT_MAX_ITER,
        pair_fit_total_kkt_tolerance=REQUIRED_PAIR_FIT_KKT_TOL,
        pair_simplex_tolerance=REQUIRED_PAIR_SIMPLEX_TOL,
        lrt_nestedness_tolerance=REQUIRED_LRT_NESTEDNESS_TOL,
        effect_identifiability=EffectIdentifiabilityImplementationContract(
            contract=REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT,
            relative_tolerance=REQUIRED_PAIR_IDENTIFIABILITY_RTOL,
            status_vocabulary=(
                REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
                REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS,
                REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS,
            ),
            identified_status=REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
            nonidentified_statuses=(
                REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS,
                REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS,
            ),
            nonidentified_effect_blank_fields=(
                REQUIRED_NONIDENTIFIED_EFFECT_BLANK_FIELDS
            ),
        ),
        rho_contract=REQUIRED_RHO_CONTRACT,
        undefined_rho_lrt_tolerance=REQUIRED_UNDEFINED_RHO_LRT_TOL,
        log_odds_ratio_contract=REQUIRED_LOG_ODDS_RATIO_CONTRACT,
    ),
)


class SealedFitError(RuntimeError):
    """Stable result-blind failure surfaced by the sealed fitting boundary."""

    def __init__(self, code: str, phase: str, row_index: int | None = None) -> None:
        """Build a generic error that never includes scientific values."""
        self.code = code
        self.phase = phase
        self.row_index = row_index
        suffix = "" if row_index is None else f" row_index={row_index}"
        super().__init__(f"sealed-fit-error code={code} phase={phase}{suffix}")


def _sealed_error(
    code: str,
    phase: str,
    row_index: int | None = None,
) -> SealedFitError:
    return SealedFitError(code, phase, row_index)


MUTSIG_RECEIPT_SCHEMA_VERSION = "1"
MUTSIG_UPSTREAM_COMMIT = "0109e27e70478181695f31ca8dd281bb44f0b3af"
MUTSIG_MAF_BINDING_STATUS = "unverified-canonical-maf-stop-ship"
MUTSIG_MAF_BINDING_REQUIREMENT = (
    "bind receipt maf_sha256 to the signed canonical MAF/population manifest"
)
MUTSIG_PATCH_PATH = Path("external/mutsig2cv_octave_dialect.patch")
MUTSIG_RUNNER_PATH = Path("scripts/run_mutsig_octave.sh")
MUTSIG_NATIVE_FWRITE_ENDIAN_CONTRACT = (
    "octave-native-fwrite-little-endian-host-required-v1"
)
MUTSIG_TENSOR_LAYOUT_CANARY = "nonuniform-2x3x2-fortran-gene-patient-effect-m0-n1-v1"
MUTSIG_RECEIPT_KEYS = frozenset(
    {
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
    },
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
FEATURE_PATTERN = re.compile(r".+_[MN]")
MUTSIG_EFFECT_INDEX = {"M": 0, "N": 1}
THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
INTERNAL_TASK_ENV = "DIALECT_K500_ORCHESTRATED_TASK"
RUNNER_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE_ROOT = RUNNER_REPO_ROOT / "src"
SAFE_CHILD_PATH = (
    "/opt/anaconda3/envs/dialect/bin:/opt/homebrew/bin:/usr/local/bin:"
    "/usr/bin:/bin:/usr/sbin:/sbin"
)
SEALED_TASK_ENVIRONMENT = {
    **THREAD_LIMIT_ENV,
    INTERNAL_TASK_ENV: "1",
    "KMP_DUPLICATE_LIB_OK": "True",
    "KMP_INIT_AT_FORK": "FALSE",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": SAFE_CHILD_PATH,
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": os.pathsep.join(
        (RUNNER_REPO_ROOT.as_posix(), RUNNER_SOURCE_ROOT.as_posix()),
    ),
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
    "DIALECT_K500_REPO_ROOT": RUNNER_REPO_ROOT.as_posix(),
    "DIALECT_K500_SOURCE_ROOT": RUNNER_SOURCE_ROOT.as_posix(),
    "TZ": "UTC",
}
PAIRWISE_COLUMNS = (
    "Gene A",
    "Gene B",
    "Tau_00",
    "Tau_10",
    "Tau_01",
    "Tau_11",
    "_00_",
    "_10_",
    "_01_",
    "_11_",
    "Tau_1X",
    "Tau_X1",
    "Effect Identifiability",
    "Rho",
    "Log Odds Ratio",
    "Likelihood Ratio",
    "Wald Statistic",
    "Fit Algorithm",
    "Fit Converged",
    "Fit Iterations",
    "Fit Last LL Gain",
    "Fit Fixed-Point Residual",
    "Fit KKT Residual",
    "Pair Fit Contract",
    "Null Log Likelihood",
    "Alternative Log Likelihood",
    "LRT Contract",
)
EXECUTED_LOCAL_PYTHON_MODULES: tuple[tuple[str, Path], ...] = (
    ("analysis", Path("analysis/__init__.py")),
    ("analysis.bmr_fdr_comparison", Path("analysis/bmr_fdr_comparison.py")),
    (
        "analysis.materialize_tcga_revision_inputs",
        Path("analysis/materialize_tcga_revision_inputs.py"),
    ),
    (
        "analysis.materialize_tcga_revision_provider_inputs",
        Path("analysis/materialize_tcga_revision_provider_inputs.py"),
    ),
    ("analysis.mutsig_lambda_co", Path("analysis/mutsig_lambda_co.py")),
    (
        "analysis.run_tcga_revision_k500",
        Path("analysis/run_tcga_revision_k500.py"),
    ),
    ("dialect", Path("src/dialect/__init__.py")),
    ("dialect._version", Path("src/dialect/_version.py")),
    ("dialect.api", Path("src/dialect/api.py")),
    ("dialect.baselines", Path("src/dialect/baselines/__init__.py")),
    ("dialect.baselines.discover", Path("src/dialect/baselines/discover.py")),
    ("dialect.baselines.fishers", Path("src/dialect/baselines/fishers.py")),
    ("dialect.baselines.megsa", Path("src/dialect/baselines/megsa.py")),
    ("dialect.baselines.runner", Path("src/dialect/baselines/runner.py")),
    ("dialect.baselines.wesme", Path("src/dialect/baselines/wesme.py")),
    ("dialect.bmr", Path("src/dialect/bmr/__init__.py")),
    ("dialect.bmr._cbase_run", Path("src/dialect/bmr/_cbase_run.py")),
    ("dialect.bmr._dig_pmf", Path("src/dialect/bmr/_dig_pmf.py")),
    ("dialect.bmr.base", Path("src/dialect/bmr/base.py")),
    ("dialect.bmr.cbase", Path("src/dialect/bmr/cbase.py")),
    ("dialect.bmr.dig", Path("src/dialect/bmr/dig.py")),
    ("dialect.bmr.registry", Path("src/dialect/bmr/registry.py")),
    ("dialect.data", Path("src/dialect/data/__init__.py")),
    ("dialect.data.cohort", Path("src/dialect/data/cohort.py")),
    ("dialect.data.io", Path("src/dialect/data/io.py")),
    (
        "dialect.data.revision_approval",
        Path("src/dialect/data/revision_approval.py"),
    ),
    (
        "dialect.data.revision_fit_policy",
        Path("src/dialect/data/revision_fit_policy.py"),
    ),
    ("dialect.data.tcga", Path("src/dialect/data/tcga.py")),
    ("dialect.data.variants", Path("src/dialect/data/variants.py")),
    ("dialect.models", Path("src/dialect/models/__init__.py")),
    ("dialect.models.assembly", Path("src/dialect/models/assembly.py")),
    ("dialect.models.gene", Path("src/dialect/models/gene.py")),
    ("dialect.models.interaction", Path("src/dialect/models/interaction.py")),
    ("dialect.utils", Path("src/dialect/utils/__init__.py")),
    ("dialect.utils.identify", Path("src/dialect/utils/identify.py")),
    ("dialect.utils.merge", Path("src/dialect/utils/merge.py")),
)
NON_PYTHON_EXECUTION_SOURCE_FILES = (
    MUTSIG_PATCH_PATH,
    MUTSIG_RUNNER_PATH,
    Path("scripts/run_cohort_pipeline.sh"),
)
SOURCE_FILES = (
    *(path for _, path in EXECUTED_LOCAL_PYTHON_MODULES),
    *NON_PYTHON_EXECUTION_SOURCE_FILES,
)
_HASH_CHUNK_BYTES = 1024 * 1024
_GIB = 1024**3
MAX_GENERAL_JOBS = 3
REQUIRED_NICE_INCREMENT = 10
PRIOR_TASK_PEAK_RSS_BYTES = round(2.083 * _GIB)
MEMORY_HEADROOM_FACTOR = 1.25
MIN_AVAILABLE_MEMORY_FRACTION = 0.33
MIN_FREE_DISK_BYTES = round(7.6 * _GIB)
STAGING_DIRECTORY_MODE = 0o700
STAGING_FILE_MODE = 0o600
SAME_UID_MACHINE_LEASE_SCHEMA = "dialect-k500-same-uid-machine-lease-v1"
SAME_UID_MACHINE_LEASE_DIRECTORY = Path("/tmp")


@dataclass(frozen=True)
class RunPaths:
    """Input and isolated output roots for a revision run."""

    source_root: Path
    mutsig_root: Path
    output_root: Path
    canonical_input_root: Path | None = None
    input_approval_manifest: Path | None = None
    expected_input_approval_sha256: str | None = None
    fit_approval_manifest: Path | None = None
    expected_fit_approval_sha256: str | None = None
    expected_canonical_input_sha256: str | None = None
    provider_input_root: Path | None = None
    expected_provider_input_manifest_sha256: str | None = None


@dataclass(frozen=True)
class ScientificFileSnapshot:
    """Immutable bytes and descriptor stat captured by one stable file read."""

    label: str
    path: Path
    content: bytes
    stat_result: os.stat_result

    def record(self) -> dict[str, Any]:
        """Return the public receipt derived from these exact captured bytes."""
        return {
            "path": self.path.as_posix(),
            "bytes": len(self.content),
            "ctime_ns": self.stat_result.st_ctime_ns,
            "device": self.stat_result.st_dev,
            "inode": self.stat_result.st_ino,
            "mtime_ns": self.stat_result.st_mtime_ns,
            "mode": self.stat_result.st_mode,
            "nlink": self.stat_result.st_nlink,
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "uid": self.stat_result.st_uid,
        }


@dataclass(frozen=True)
class CohortScientificSnapshot:
    """One parse-once scientific snapshot used by all contract computations."""

    files: Mapping[str, ScientificFileSnapshot]
    counts: pd.DataFrame
    authoritative_samples: tuple[str, ...]
    cbase_pmfs: Mapping[str, Mapping[int, float]]
    dig_pmfs: Mapping[str, Mapping[int, float]]
    mutsig_metadata: Mapping[str, int]
    mutsig_genes: tuple[str, ...]
    mutsig_patients: tuple[str, ...]
    mutsig_lambdas: np.ndarray
    mutsig_receipt: Mapping[str, str]


@dataclass(frozen=True)
class Task:
    """One atomic cohort/background inference task."""

    cohort: str
    bmr: str


@dataclass(frozen=True)
class HostResourceSnapshot:
    """One live, aggregate-only host resource readback."""

    measured_at_utc: str
    logical_cores: int
    load_average_1m: float
    total_memory_bytes: int
    available_memory_bytes: int
    free_disk_bytes: int
    cpu_source: str
    memory_source: str


@contextmanager
def _same_uid_machine_execution_lease(output_root: Path) -> Iterator[Path]:
    """Hold one same-UID machine-wide lease across preflight or execution.

    The per-process job cap is insufficient if the same workstation account launches
    two orchestrators with different output roots. This secure 0600 advisory lease
    serializes that account without claiming cross-user exclusion; aggregate resource
    gates still account for load from every user.
    """
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
        msg = f"Unable to open the K=500 same-UID machine lease safely: {lease_path}"
        raise RuntimeError(msg) from error
    try:
        _require_secure_lease_file(descriptor, lease_path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            msg = (
                "Another DIALECT K=500 process for this UID already holds the "
                f"machine-wide resource lease: {lease_path}"
            )
            raise RuntimeError(msg) from error
        _require_secure_lease_file(descriptor, lease_path)
        lease_record = {
            "schema": SAME_UID_MACHINE_LEASE_SCHEMA,
            "pid": os.getpid(),
            "output_root": output_root.absolute().as_posix(),
            "acquired_at_utc": _utc_now(),
        }
        payload = _canonical_json(lease_record) + b"\n"
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
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
        msg = f"Same-UID machine lease path disappeared: {lease_path}"
        raise RuntimeError(msg) from error
    if (
        not stat_module.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_uid != os.getuid()
        or descriptor_stat.st_nlink != 1
        or stat_module.S_IMODE(descriptor_stat.st_mode) != 0o600
        or descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        msg = f"Same-UID machine lease is not a stable private file: {lease_path}"
        raise RuntimeError(msg)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _task_resource_usage(started: float) -> dict[str, Any]:
    """Return normalized, explicitly sourced resource usage for this task process."""
    elapsed_seconds = time.monotonic() - started
    native_peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        native_unit = "bytes"
        peak_rss_bytes = native_peak_rss
    elif sys.platform.startswith("linux"):
        native_unit = "KiB"
        peak_rss_bytes = native_peak_rss * 1024
    else:
        msg = f"Unsupported ru_maxrss unit convention on platform {sys.platform!r}."
        raise RuntimeError(msg)
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss": {
            "bytes": peak_rss_bytes,
            "native_value": native_peak_rss,
            "native_unit": native_unit,
            "platform": sys.platform,
            "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        },
    }


def _validate_task_resource_usage(manifest: dict[str, Any], task_dir: Path) -> None:
    """Fail closed when task resource provenance is absent or internally invalid."""
    usage = manifest.get("resource_usage")
    if not isinstance(usage, dict):
        msg = f"Task manifest lacks resource usage provenance: {task_dir}"
        raise TypeError(msg)
    elapsed_seconds = usage.get("elapsed_seconds")
    peak_rss = usage.get("peak_rss")
    if (
        not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0
        or not isinstance(peak_rss, dict)
    ):
        msg = f"Task manifest has invalid elapsed-time/RSS provenance: {task_dir}"
        raise ValueError(msg)
    native_value = peak_rss.get("native_value")
    peak_rss_bytes = peak_rss.get("bytes")
    platform = peak_rss.get("platform")
    native_unit = peak_rss.get("native_unit")
    if platform == "darwin":
        expected_unit = "bytes"
        multiplier = 1
    elif isinstance(platform, str) and platform.startswith("linux"):
        expected_unit = "KiB"
        multiplier = 1024
    else:
        msg = f"Task manifest has unsupported RSS platform provenance: {task_dir}"
        raise ValueError(msg)
    if (
        not isinstance(native_value, int)
        or isinstance(native_value, bool)
        or native_value <= 0
        or not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes != native_value * multiplier
        or native_unit != expected_unit
        or peak_rss.get("source")
        != "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    ):
        msg = f"Task manifest has invalid peak-RSS provenance: {task_dir}"
        raise ValueError(msg)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256(path: Path) -> str:
    digest, _, _ = _hash_secure_regular_with_stat(path, label="SHA-256 input")
    return digest


def _file_record(path: Path) -> dict[str, Any]:
    digest, observed, byte_count = _hash_secure_regular_with_stat(
        path,
        label="file-record input",
    )
    return {
        "path": path.as_posix(),
        "bytes": byte_count,
        "mtime_ns": observed.st_mtime_ns,
        "sha256": digest,
    }


def _sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _rename_exclusive_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename descriptor-relative names without replacement."""
    _require_safe_basename(source_name, label="exclusive rename source")
    _require_safe_basename(destination_name, label="exclusive rename destination")
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        rename = getattr(library, "renameatx_np", None)
        if rename is None:
            msg = (
                "renameatx_np is unavailable; exclusive task publication cannot "
                "proceed."
            )
            raise RuntimeError(msg)
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
            msg = "renameat2 is unavailable; exclusive task publication cannot proceed."
            raise RuntimeError(msg)
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
        msg = f"Unsupported platform for exclusive task publication: {sys.platform!r}."
        raise RuntimeError(msg)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Exclusive task publication target already exists",
            destination_name,
        )
    raise OSError(
        error_number,
        "Atomic exclusive task publication failed",
        destination_name,
    )


def _write_all(descriptor: int, content: bytes, *, label: str) -> None:
    """Write exact bytes, rejecting short or stalled descriptor writes."""
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            msg = f"Short write while staging {label}."
            raise OSError(msg)
        view = view[written:]


def _write_bytes_atomic_at(
    parent_fd: int,
    name: str,
    content: bytes,
    *,
    label: str,
) -> None:
    """Publish one single-link regular file by dirfd, fsync, and exact readback."""
    _require_safe_basename(name, label=label)
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    published = False
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            STAGING_FILE_MODE,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = f"Staged {label} must be a single-link regular file."
            raise ValueError(msg)
        _write_all(descriptor, content, label=label)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or not stat_module.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_size != len(content)
        ):
            msg = f"Staged {label} identity changed before publication."
            raise ValueError(msg)
        _require_regular_entry_identity(
            parent_fd,
            temporary_name,
            descriptor,
            label=f"staged {label}",
        )
        os.close(descriptor)
        descriptor = None
        _rename_exclusive_at(parent_fd, temporary_name, parent_fd, name)
        os.fsync(parent_fd)
        published = True
        observed = _read_regular_at(parent_fd, name, label=label)
        if observed != content:
            msg = f"Published {label} failed exact descriptor readback."
            raise ValueError(msg)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)


def _write_json_atomic(path: Path, payload: object) -> None:
    """Publish canonical JSON without replacement through a stable parent dirfd."""
    content = _canonical_json(payload) + b"\n"
    parent_fd = _ensure_secure_directory(path.parent, label="JSON artifact parent")
    try:
        _write_bytes_atomic_at(
            parent_fd,
            path.name,
            content,
            label=f"JSON artifact {path.name}",
        )
        _require_directory_path_identity(
            path.parent,
            parent_fd,
            label="JSON artifact parent",
        )
    finally:
        os.close(parent_fd)


def _fsync_directory(path: Path) -> None:
    """Fsync one no-follow directory and ensure its path still reaches that inode."""
    descriptor = _open_secure_directory(path, label="fsync directory")
    try:
        os.fsync(descriptor)
        _require_directory_path_identity(path, descriptor, label="fsync directory")
    finally:
        os.close(descriptor)


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Publish a directory via stable parent dirfds with identity revalidation."""
    source_parent = _open_secure_directory(source.parent, label="rename source parent")
    try:
        destination_parent = _open_secure_directory(
            destination.parent,
            label="rename destination parent",
        )
        try:
            source_fd = _open_directory_at(
                source_parent,
                source.name,
                label="rename staged directory",
            )
            try:
                _require_directory_entry_identity(
                    source_parent,
                    source.name,
                    source_fd,
                    label="rename staged directory",
                )
                _require_directory_path_identity(
                    source.parent,
                    source_parent,
                    label="rename source parent",
                )
                _require_directory_path_identity(
                    destination.parent,
                    destination_parent,
                    label="rename destination parent",
                )
                _rename_exclusive_at(
                    source_parent,
                    source.name,
                    destination_parent,
                    destination.name,
                )
                _require_directory_entry_identity(
                    destination_parent,
                    destination.name,
                    source_fd,
                    label="published task directory",
                )
                os.fsync(destination_parent)
                _require_directory_path_identity(
                    source.parent,
                    source_parent,
                    label="rename source parent",
                )
                _require_directory_path_identity(
                    destination.parent,
                    destination_parent,
                    label="rename destination parent",
                )
            finally:
                os.close(source_fd)
        finally:
            os.close(destination_parent)
    finally:
        os.close(source_parent)


def _open_secure_directory(path: Path, *, label: str) -> int:
    """Open an absolute directory through a descriptor-relative no-symlink walk."""
    absolute = Path(os.path.abspath(path))  # noqa: PTH100
    if not absolute.anchor:
        msg = f"{label} must be absolute."
        raise ValueError(msg)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not no_follow or not directory or os.open not in os.supports_dir_fd:
        msg = "Secure descriptor-relative directory reads are unavailable."
        raise RuntimeError(msg)
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(absolute.anchor, flags)
    opened = False
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat_module.S_ISDIR(os.fstat(descriptor).st_mode):
            msg = f"{label} must be a directory: {path}"
            raise ValueError(msg)
        opened = True
    finally:
        if not opened:
            os.close(descriptor)
    return descriptor


def _require_safe_basename(name: str, *, label: str) -> str:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        msg = f"{label} must be a single safe basename."
        raise ValueError(msg)
    return name


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not no_follow or not directory or os.open not in os.supports_dir_fd:
        msg = "Secure descriptor-relative directory operations are unavailable."
        raise RuntimeError(msg)
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_directory_at(parent_fd: int, name: str, *, label: str) -> int:
    """Open one no-follow child directory relative to a pinned parent."""
    _require_safe_basename(name, label=label)
    descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    if not stat_module.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        msg = f"{label} must remain a directory."
        raise ValueError(msg)
    return descriptor


def _ensure_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
    mode: int = STAGING_DIRECTORY_MODE,
) -> int:
    """Create/open one child directory without following or replacing an entry."""
    _require_safe_basename(name, label=label)
    try:
        return _open_directory_at(parent_fd, name, label=label)
    except FileNotFoundError:
        with suppress(FileExistsError):
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
        descriptor = _open_directory_at(parent_fd, name, label=label)
        os.fsync(parent_fd)
        return descriptor


def _create_directory_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
    mode: int = STAGING_DIRECTORY_MODE,
) -> int:
    """Create one fresh child directory and bind its entry to the opened inode."""
    _require_safe_basename(name, label=label)
    os.mkdir(name, mode=mode, dir_fd=parent_fd)
    descriptor = _open_directory_at(parent_fd, name, label=label)
    try:
        _require_directory_entry_identity(
            parent_fd,
            name,
            descriptor,
            label=label,
        )
        os.fsync(parent_fd)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _ensure_secure_directory(path: Path, *, label: str) -> int:
    """Create/open an absolute directory tree through pinned no-follow ancestors."""
    absolute = Path(os.path.abspath(path))  # noqa: PTH100
    flags = _directory_flags()
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(
                        component,
                        mode=STAGING_DIRECTORY_MODE,
                        dir_fd=descriptor,
                    )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        os.close(descriptor)
        msg = f"Unable to create/open {label} without directory symlinks: {path}"
        raise RuntimeError(msg) from error
    return descriptor


def _require_directory_entry_identity(
    parent_fd: int,
    name: str,
    directory_fd: int,
    *,
    label: str,
) -> None:
    """Require a parent entry to remain the directory held by ``directory_fd``."""
    _require_safe_basename(name, label=label)
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        msg = f"{label} disappeared from its pinned parent."
        raise ValueError(msg) from error
    opened = os.fstat(directory_fd)
    if (
        not stat_module.S_ISDIR(entry.st_mode)
        or not stat_module.S_ISDIR(opened.st_mode)
        or entry.st_dev != opened.st_dev
        or entry.st_ino != opened.st_ino
    ):
        msg = f"{label} inode identity changed under its pinned parent."
        raise ValueError(msg)


def _require_directory_path_identity(
    path: Path,
    directory_fd: int,
    *,
    label: str,
) -> None:
    """Require the current absolute no-follow path to resolve to the pinned inode."""
    try:
        current_fd = _open_secure_directory(path, label=f"{label} path revalidation")
    except OSError as error:
        msg = f"{label} path identity cannot be revalidated: {path}"
        raise ValueError(msg) from error
    try:
        current = os.fstat(current_fd)
        pinned = os.fstat(directory_fd)
        if (
            current.st_dev != pinned.st_dev
            or current.st_ino != pinned.st_ino
            or not stat_module.S_ISDIR(current.st_mode)
            or not stat_module.S_ISDIR(pinned.st_mode)
        ):
            msg = f"{label} path identity changed during publication: {path}"
            raise ValueError(msg)
    finally:
        os.close(current_fd)


def _directory_entry_exists(parent_fd: int, name: str, *, label: str) -> bool:
    """Check one destination entry without following it; reject non-directories."""
    _require_safe_basename(name, label=label)
    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat_module.S_ISDIR(observed.st_mode):
        msg = f"{label} exists but is not a directory."
        raise ValueError(msg)
    return True


def _require_regular_entry_identity(
    parent_fd: int,
    name: str,
    descriptor: int,
    *,
    label: str,
) -> None:
    """Require one parent entry to remain the opened single-link regular file."""
    _require_safe_basename(name, label=label)
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        msg = f"{label} disappeared from its pinned directory."
        raise ValueError(msg) from error
    opened = os.fstat(descriptor)
    if (
        not stat_module.S_ISREG(entry.st_mode)
        or not stat_module.S_ISREG(opened.st_mode)
        or entry.st_nlink != 1
        or opened.st_nlink != 1
        or entry.st_dev != opened.st_dev
        or entry.st_ino != opened.st_ino
    ):
        msg = f"{label} must remain one single-link regular file."
        raise ValueError(msg)


def _stable_regular_stat_signature(observed: os.stat_result) -> tuple[object, ...]:
    """Return every mutation-relevant stat field that a read must preserve."""
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_uid,
        observed.st_gid,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
        getattr(observed, "st_birthtime", None),
        getattr(observed, "st_flags", None),
        getattr(observed, "st_gen", None),
        getattr(observed, "st_blocks", None),
        getattr(observed, "st_blksize", None),
    )


def _read_pinned_regular_replay(
    descriptor: int,
    anchored_stat: os.stat_result,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Replay exact bytes and stable metadata through one already pinned file."""
    before = os.fstat(descriptor)
    if (
        not stat_module.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or _stable_regular_stat_signature(before)
        != _stable_regular_stat_signature(anchored_stat)
    ):
        msg = f"{label} changed before its pinned descriptor replay."
        raise ValueError(msg)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
        chunks.append(chunk)
    content = b"".join(chunks)
    after = os.fstat(descriptor)
    if (
        _stable_regular_stat_signature(after) != _stable_regular_stat_signature(before)
        or len(content) != before.st_size
    ):
        msg = f"{label} changed during its pinned descriptor replay."
        raise ValueError(msg)
    return content, after


@dataclass(frozen=True)
class _PinnedRegularSnapshot:
    """One held task artifact and its exact pre-validation state."""

    descriptor: int
    content: bytes
    stat: os.stat_result


@contextmanager
def _pinned_task_output_snapshot(
    directory_fd: int,
) -> Iterator[dict[str, _PinnedRegularSnapshot]]:
    """Pin the exact closed three-file task bundle across final validation."""
    names = (
        "single_gene_results.csv",
        "pairwise_interaction_results.csv",
        "task_manifest.json",
    )
    if set(os.listdir(directory_fd)) != set(names):
        raise _sealed_error("output-inventory-invalid", "validate-output")  # noqa: EM101
    snapshots: dict[str, _PinnedRegularSnapshot] = {}
    try:
        for name in names:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            try:
                anchored_stat = os.fstat(descriptor)
                _require_regular_entry_identity(
                    directory_fd,
                    name,
                    descriptor,
                    label=f"pinned final task artifact {name}",
                )
                content, replayed_stat = _read_pinned_regular_replay(
                    descriptor,
                    anchored_stat,
                    label=f"pinned final task artifact {name}",
                )
            except Exception:
                os.close(descriptor)
                raise
            snapshots[name] = _PinnedRegularSnapshot(
                descriptor=descriptor,
                content=content,
                stat=replayed_stat,
            )
        yield snapshots
    finally:
        for snapshot in snapshots.values():
            os.close(snapshot.descriptor)


def _replay_pinned_task_output_snapshot(
    directory_fd: int,
    snapshots: Mapping[str, _PinnedRegularSnapshot],
) -> None:
    """Rebind validated task bytes to unchanged, still-visible file entries."""
    expected_names = set(snapshots)
    if set(os.listdir(directory_fd)) != expected_names:
        msg = "Final task artifact inventory changed after validation."
        raise ValueError(msg)
    for name, snapshot in snapshots.items():
        content, replayed_stat = _read_pinned_regular_replay(
            snapshot.descriptor,
            snapshot.stat,
            label=f"final task artifact {name}",
        )
        if content != snapshot.content:
            msg = f"Final task artifact {name} bytes changed after validation."
            raise ValueError(msg)
        _require_regular_entry_identity(
            directory_fd,
            name,
            snapshot.descriptor,
            label=f"visible final task artifact {name}",
        )
        visible_stat = os.fstat(snapshot.descriptor)
        if _stable_regular_stat_signature(
            visible_stat,
        ) != _stable_regular_stat_signature(replayed_stat):
            msg = f"Final task artifact {name} changed during visible-entry binding."
            raise ValueError(msg)
    if set(os.listdir(directory_fd)) != expected_names:
        msg = "Final task artifact inventory changed during visible-entry binding."
        raise ValueError(msg)


def _unlink_single_link_regular_at(parent_fd: int, name: str, *, label: str) -> None:
    """Remove one staged regular file only if it has no alternate hard link."""
    _require_safe_basename(name, label=label)
    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat_module.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        msg = f"Refusing to remove non-private staged {label}."
        raise ValueError(msg)
    os.unlink(name, dir_fd=parent_fd)


def _require_empty_directory(directory_fd: int, *, label: str) -> None:
    if os.listdir(directory_fd):
        msg = f"{label} contains an unknown entry."
        raise ValueError(msg)


def _read_regular_at_with_stat(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read one stable single-link file from an already verified directory."""
    if not name or "/" in name or name in {".", ".."}:
        msg = f"{label} has an invalid basename."
        raise ValueError(msg)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = f"{label} must be a single-link regular file."
            raise ValueError(msg)
        chunks = []
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mode != before.st_mode
            or after.st_nlink != 1
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or len(content) != before.st_size
        ):
            msg = f"{label} changed during its stable descriptor read."
            raise ValueError(msg)
        return content, before
    finally:
        os.close(descriptor)


def _hash_regular_at_with_stat(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[str, os.stat_result, int]:
    """Stream-hash one stable single-link regular file by descriptor."""
    if not name or "/" in name or name in {".", ".."}:
        msg = f"{label} has an invalid basename."
        raise ValueError(msg)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = f"{label} must be a single-link regular file."
            raise ValueError(msg)
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mode != before.st_mode
            or after.st_nlink != 1
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or byte_count != before.st_size
        ):
            msg = f"{label} changed during its stable descriptor hash."
            raise ValueError(msg)
        return digest.hexdigest(), before, byte_count
    finally:
        os.close(descriptor)


def _read_regular_at(directory_fd: int, name: str, *, label: str) -> bytes:
    """Read one nonlinked regular file from an already verified directory."""
    content, _ = _read_regular_at_with_stat(directory_fd, name, label=label)
    return content


def _read_secure_regular_with_stat(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    parent_fd = _open_secure_directory(path.parent, label=f"{label} parent")
    try:
        return _read_regular_at_with_stat(parent_fd, path.name, label=label)
    finally:
        os.close(parent_fd)


def _read_secure_regular_bytes(path: Path, *, label: str) -> bytes:
    content, _ = _read_secure_regular_with_stat(path, label=label)
    return content


def _read_visible_regular_with_stat(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read a stable file and prove the persistent path still names its inode."""
    _require_safe_basename(path.name, label=label)
    parent_fd = _open_secure_directory(path.parent, label=f"{label} parent")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
                msg = f"{label} must be a single-link regular file."
                raise ValueError(msg)
            chunks = []
            while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
                chunks.append(chunk)
            content = b"".join(chunks)
            after = os.fstat(descriptor)
            if (
                _stable_regular_stat_signature(after)
                != _stable_regular_stat_signature(before)
                or len(content) != before.st_size
            ):
                msg = f"{label} changed during its visible-entry readback."
                raise ValueError(msg)
            _require_regular_entry_identity(
                parent_fd,
                path.name,
                descriptor,
                label=label,
            )
            _require_directory_path_identity(
                path.parent,
                parent_fd,
                label=f"{label} parent",
            )
            visible = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if _stable_regular_stat_signature(
                visible,
            ) != _stable_regular_stat_signature(before):
                msg = f"{label} visible path changed during readback."
                raise ValueError(msg)
            return content, before
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _hash_secure_regular_with_stat(
    path: Path,
    *,
    label: str,
) -> tuple[str, os.stat_result, int]:
    parent_fd = _open_secure_directory(path.parent, label=f"{label} parent")
    try:
        return _hash_regular_at_with_stat(parent_fd, path.name, label=label)
    finally:
        os.close(parent_fd)


def _parse_json_bytes(raw: bytes, *, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        msg = f"Invalid authoritative JSON document: {path}"
        raise ValueError(msg) from error
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object: {path}"
        raise TypeError(msg)
    _reject_json_surrogates(payload, label=f"authoritative JSON {path}")
    if raw != _canonical_json(payload) + b"\n":
        msg = f"Authoritative JSON is not canonical with one terminal LF: {path}"
        raise ValueError(msg)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    raw = _read_secure_regular_bytes(path, label="authoritative JSON")
    return _parse_json_bytes(raw, path=path)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            msg = f"Duplicate JSON object key: {key!r}."
            raise ValueError(msg)
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    msg = f"Non-finite JSON constant is forbidden: {value}."
    raise ValueError(msg)


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = f"JSON float overflows finite range: {value}."
        raise ValueError(msg)
    return parsed


def _reject_json_surrogates(value: object, *, label: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            msg = f"{label} contains a Unicode surrogate."
            raise ValueError(msg)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_json_surrogates(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_json_surrogates(key, label=f"{label} object key")
            _reject_json_surrogates(item, label=f"{label}.{key}")


def _revision_authority_values(paths: RunPaths) -> tuple[object, ...]:
    return (
        paths.provider_input_root,
        paths.expected_provider_input_manifest_sha256,
        paths.canonical_input_root,
        paths.input_approval_manifest,
        paths.expected_input_approval_sha256,
        paths.fit_approval_manifest,
        paths.expected_fit_approval_sha256,
        paths.expected_canonical_input_sha256,
    )


def _revision_authority_is_configured(paths: RunPaths) -> bool:
    values = _revision_authority_values(paths)
    if any(value is not None for value in values) and any(
        value is None for value in values
    ):
        msg = "Revision input authority arguments must be supplied together."
        raise ValueError(msg)
    return all(value is not None for value in values)


def _revision_authority_manifest_record(paths: RunPaths) -> dict[str, Any]:
    if not _revision_authority_is_configured(paths):
        return {"configured": False}
    provider = _validated_provider_bundle(paths)
    return {
        "configured": True,
        "provider_input": _provider_root_receipt(paths, provider),
        "canonical_input_root": paths.canonical_input_root.as_posix(),
        "input_approval_manifest": paths.input_approval_manifest.as_posix(),
        "expected_input_approval_sha256": paths.expected_input_approval_sha256,
        "fit_approval_manifest": paths.fit_approval_manifest.as_posix(),
        "expected_fit_approval_sha256": paths.expected_fit_approval_sha256,
        "expected_canonical_input_sha256": paths.expected_canonical_input_sha256,
    }


def _revision_authority_cli_args(paths: RunPaths) -> list[str]:
    if not _revision_authority_is_configured(paths):
        return []
    return [
        "--provider-input-root",
        paths.provider_input_root.as_posix(),
        "--expected-provider-input-manifest-sha256",
        str(paths.expected_provider_input_manifest_sha256),
        "--canonical-input-root",
        paths.canonical_input_root.as_posix(),
        "--input-approval-manifest",
        paths.input_approval_manifest.as_posix(),
        "--expected-input-approval-sha256",
        str(paths.expected_input_approval_sha256),
        "--fit-approval-manifest",
        paths.fit_approval_manifest.as_posix(),
        "--expected-fit-approval-sha256",
        str(paths.expected_fit_approval_sha256),
        "--expected-canonical-input-sha256",
        str(paths.expected_canonical_input_sha256),
    ]


def _require_lowercase_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        msg = f"{label} must be a lowercase SHA-256."
        raise ValueError(msg)
    return value


@lru_cache(maxsize=4)
def _validated_input_bundle(
    root: Path,
    expected_manifest_sha256: str,
    approval_manifest: Path,
    expected_approval_sha256: str,
) -> dict[str, Any]:
    manifest = validate_materialized_input_bundle(
        root,
        expected_manifest_sha256,
        approval_manifest,
        expected_approval_sha256,
        require_current_execution_environment=True,
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


@lru_cache(maxsize=4)
def _validated_provider_bundle_cached(  # noqa: PLR0913
    provider_root: Path,
    expected_provider_manifest_sha256: str,
    canonical_input_root: Path,
    expected_canonical_manifest_sha256: str,
    input_approval_manifest: Path,
    expected_input_approval_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable provider bundle against independent trust anchors."""
    return validate_materialized_provider_input_bundle(
        provider_root,
        expected_provider_manifest_sha256,
        canonical_input_root,
        expected_canonical_manifest_sha256,
        input_approval_manifest,
        expected_input_approval_sha256,
        require_current_execution_environment=True,
    )


def _validated_provider_bundle(
    paths: RunPaths,
    *,
    fresh: bool = False,
) -> dict[str, Any]:
    """Return the sole production provider authority and reject path substitution."""
    if not _revision_authority_is_configured(paths):
        msg = "Production provider validation requires complete revision authority."
        raise ValueError(msg)
    provider_root = paths.provider_input_root
    expected_provider = paths.expected_provider_input_manifest_sha256
    canonical_root = paths.canonical_input_root
    expected_canonical = paths.expected_canonical_input_sha256
    approval_manifest = paths.input_approval_manifest
    expected_approval = paths.expected_input_approval_sha256
    if (
        provider_root is None
        or expected_provider is None
        or canonical_root is None
        or expected_canonical is None
        or approval_manifest is None
        or expected_approval is None
    ):
        msg = "Revision authority changed during provider validation."
        raise RuntimeError(msg)
    if fresh:
        _validated_provider_bundle_cached.cache_clear()
    bundle = _validated_provider_bundle_cached(
        provider_root,
        _require_lowercase_sha256(
            expected_provider,
            label="expected provider input manifest SHA-256",
        ),
        canonical_root,
        _require_lowercase_sha256(
            expected_canonical,
            label="expected canonical input manifest SHA-256",
        ),
        approval_manifest,
        _require_lowercase_sha256(
            expected_approval,
            label="expected input approval manifest SHA-256",
        ),
    )
    if (
        set(bundle)
        != {
            "root",
            "manifest",
            "manifest_file",
            "roots",
            "cohorts",
            "cohort_bindings",
            "full_acceptance_receipt",
            "full_acceptance_receipt_sha256",
            "association_outputs_opened",
        }
        or bundle["association_outputs_opened"] is not False
        or bundle["cohorts"] != list(TCGA_COHORTS)
        or bundle["root"] != provider_root
        or bundle["roots"]
        != {"cohorts": paths.source_root, "mutsig": paths.mutsig_root}
    ):
        msg = "Validated provider bundle does not match the derived runner paths."
        raise ValueError(msg)
    return bundle


def _provider_root_receipt(
    paths: RunPaths,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the independently pinned root receipt for JSON provenance."""
    manifest_binding = bundle.get("manifest_file")
    manifest = bundle.get("manifest")
    full_acceptance = bundle.get("full_acceptance_receipt")
    full_acceptance_sha256 = bundle.get("full_acceptance_receipt_sha256")
    if (
        not isinstance(manifest_binding, dict)
        or not isinstance(manifest, dict)
        or not isinstance(full_acceptance, dict)
        or not isinstance(full_acceptance_sha256, str)
    ):
        msg = "Validated provider bundle lacks its root manifest receipt."
        raise TypeError(msg)
    if (
        set(full_acceptance)
        != {
            "association_outputs_opened",
            "authority_sha256",
            "cohort_receipts_sha256",
            "contract",
            "execution_snapshot",
            "full_inventory_validated",
            "provider_manifest_sha256",
            "schema_version",
        }
        or full_acceptance.get("contract") != "provider-full-acceptance-receipt-v1"
        or full_acceptance.get("full_inventory_validated") is not True
        or full_acceptance.get("association_outputs_opened") is not False
        or not isinstance(full_acceptance.get("execution_snapshot"), dict)
    ):
        msg = "Validated provider full-acceptance receipt has an invalid schema."
        raise ValueError(msg)
    path = manifest_binding.get("path")
    record = manifest_binding.get("file")
    if not isinstance(path, Path) or not isinstance(record, dict):
        msg = "Validated provider manifest binding has an invalid schema."
        raise TypeError(msg)
    expected_hash = _require_lowercase_sha256(
        paths.expected_provider_input_manifest_sha256,
        label="expected provider input manifest SHA-256",
    )
    if full_acceptance.get("provider_manifest_sha256") != expected_hash:
        msg = "Provider full-acceptance receipt lost its manifest trust anchor."
        raise ValueError(msg)
    validated_full_acceptance_sha256 = _require_lowercase_sha256(
        full_acceptance_sha256,
        label="provider full-acceptance receipt SHA-256",
    )
    if (
        full_acceptance_receipt_sha256(full_acceptance)
        != validated_full_acceptance_sha256
    ):
        msg = "Provider full-acceptance receipt lost its independent digest."
        raise ValueError(msg)
    for key in ("authority_sha256", "cohort_receipts_sha256"):
        _require_lowercase_sha256(
            full_acceptance.get(key),
            label=f"provider full-acceptance {key}",
        )
    if (
        set(record) != {"path", "bytes", "sha256"}
        or record.get("sha256") != expected_hash
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or record["bytes"] <= 0
        or manifest.get("contract") != PROVIDER_INPUT_CONTRACT
    ):
        msg = "Validated provider root receipt is inconsistent with its trust anchor."
        raise ValueError(msg)
    sources = manifest.get("sources")
    git_executable = (
        sources.get("git_executable") if isinstance(sources, dict) else None
    )
    if not isinstance(git_executable, dict) or set(git_executable) != {
        "bytes",
        "path",
        "sha256",
    }:
        msg = "Validated provider root lacks an exact Git executable receipt."
        raise TypeError(msg)
    _read_frozen_record_bytes(git_executable, label="provider-authorized Git")
    cohort_receipts_sha256 = _json_sha256(
        manifest.get("cohort_provider_receipts"),
    )
    if full_acceptance.get("cohort_receipts_sha256") != cohort_receipts_sha256:
        msg = "Provider full acceptance lost its cohort-receipt sequence."
        raise ValueError(msg)
    return {
        "contract": PROVIDER_INPUT_CONTRACT,
        "root": bundle["root"].as_posix(),
        "expected_manifest_sha256": expected_hash,
        "manifest": {
            "path": path.as_posix(),
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        },
        "cohort_provider_receipts_sha256": cohort_receipts_sha256,
        "full_acceptance_receipt": full_acceptance,
        "full_acceptance_receipt_sha256": validated_full_acceptance_sha256,
        "git_executable": dict(git_executable),
        "association_outputs_opened": False,
    }


def _provider_file_record(binding: object, *, label: str) -> dict[str, Any]:
    """Normalize one public provider file binding to an absolute file receipt."""
    if not isinstance(binding, dict) or set(binding) != {"path", "file"}:
        msg = f"Validated provider {label} binding has an invalid schema."
        raise TypeError(msg)
    path = binding["path"]
    record = binding["file"]
    if (
        not isinstance(path, Path)
        or not isinstance(record, dict)
        or set(record) != {"path", "bytes", "sha256"}
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] <= 0
    ):
        msg = f"Validated provider {label} receipt has an invalid schema."
        raise TypeError(msg)
    return {
        "path": path.as_posix(),
        "bytes": record["bytes"],
        "sha256": _require_lowercase_sha256(
            record["sha256"],
            label=f"validated provider {label} SHA-256",
        ),
    }


def _provider_cohort_binding(
    paths: RunPaths,
    cohort: str,
    *,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one cohort binding from the validated closed provider root."""
    validated = _validated_provider_bundle(paths) if bundle is None else bundle
    bindings = validated.get("cohort_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(TCGA_COHORTS):
        msg = "Validated provider bundle has an invalid cohort binding set."
        raise ValueError(msg)
    binding = bindings.get(cohort)
    if not isinstance(binding, dict) or binding.get("cohort") != cohort:
        msg = f"Validated provider bundle lacks the cohort binding for {cohort}."
        raise ValueError(msg)
    return binding


@lru_cache(maxsize=4)
def _validated_input_approval(
    manifest: Path,
    expected_sha256: str,
) -> RevisionApproval:
    approval = validate_revision_approval(
        manifest,
        expected_sha256,
        MATERIALIZE_FINAL_INPUTS_STAGE,
    )
    _require_materialize_stage_binding(approval)
    return approval


@lru_cache(maxsize=4)
def _validated_fit_approval(
    manifest: Path,
    expected_sha256: str,
) -> RevisionApproval:
    approval = validate_revision_approval(
        manifest,
        expected_sha256,
        FIT_SEALED_TCGA_K500_STAGE,
    )
    _require_stage_scoped_fit_approval(approval)
    return approval


def _require_stage_scoped_fit_approval(
    approval: RevisionApproval,
) -> None:
    """Require production v5 fit authority with exactly the D1-D6 scope."""
    required_decisions = STAGE_MINIMUM_DECISIONS[FIT_SEALED_TCGA_K500_STAGE]
    if (
        approval.schema != STAGE_SCOPED_APPROVAL_SCHEMA
        or tuple(approval.decisions) != required_decisions
    ):
        msg = (
            "Production K=500 fitting requires the stage-scoped v5 fit approval "
            "with exactly D1-D6."
        )
        raise ValueError(msg)


def _signed_tested_family_record(paths: RunPaths) -> dict[str, Any] | None:
    """Validate and serialize the signed D5 family before run initialization."""
    if not _revision_authority_is_configured(paths):
        return None
    fit_manifest = paths.fit_approval_manifest
    expected_fit_sha256 = paths.expected_fit_approval_sha256
    if fit_manifest is None or expected_fit_sha256 is None:
        msg = "Fit approval authority disappeared during family validation."
        raise RuntimeError(msg)
    approval = _validated_fit_approval(fit_manifest, expected_fit_sha256)
    _require_fit_stage_binding(approval, paths)
    policy = validate_revision_fit_policy(
        approval,
        expected_d4_implementation=REQUIRED_D4_IMPLEMENTATION,
        expected_tested_family=REQUIRED_TESTED_FAMILY,
    )
    if policy.d5.tested_family != REQUIRED_TESTED_FAMILY:
        msg = "Signed D5 tested family does not match the runner implementation."
        raise ValueError(msg)
    _require_d3_runtime_contract(policy, paths)
    return asdict(policy.d5.tested_family)


def _fit_policy_record(policy: RevisionFitPolicy) -> dict[str, Any]:
    receipts = {
        decision_id: {
            "decision_id": receipt.decision_id,
            "contract": receipt.contract,
            "decision_digest": receipt.decision_digest,
            "canonical_artifact_path": receipt.canonical_artifact_path,
            "canonical_artifact_sha256": receipt.canonical_artifact_sha256,
            "canonical_artifact_size_bytes": receipt.canonical_artifact_size_bytes,
            "payload_sha256": receipt.payload_sha256,
        }
        for decision_id, receipt in policy.receipts.items()
    }
    return {
        "d3": asdict(policy.d3),
        "d4": asdict(policy.d4),
        "d5": asdict(policy.d5),
        "d6": asdict(policy.d6),
        "receipts": receipts,
    }


def _decision_reauthorization_record(
    decision: DecisionApproval,
) -> dict[str, Any]:
    """Return stage-independent semantics that must match on reauthorization."""
    return {
        "decision_id": decision.decision_id,
        "disposition": decision.disposition,
        "exact_resolution": decision.exact_resolution,
        "canonical_artifact": {
            "sha256": decision.canonical_artifact.sha256,
            "size_bytes": decision.canonical_artifact.size_bytes,
            "content": decision.canonical_artifact.content,
        },
        "execution_owner": decision.execution_owner,
        "claim_owner": decision.claim_owner,
        "rerun_or_reuse_consequence": decision.rerun_or_reuse_consequence,
        "permitted_claims": decision.permitted_claims,
        "forbidden_claims": decision.forbidden_claims,
    }


def _require_materialize_stage_binding(
    approval: RevisionApproval | dict[str, Any],
) -> dict[str, str]:
    """Require a materialize-only envelope bound to the exact D1/D2 artifacts."""
    if isinstance(approval, RevisionApproval):
        allowed_stages: object = approval.allowed_stages
        bindings: object = approval.stage_bindings
        try:
            d1_sha256 = approval.decisions["D1"].canonical_artifact.sha256
            d2_sha256 = approval.decisions["D2"].canonical_artifact.sha256
        except (AttributeError, KeyError) as exc:
            msg = "Materialize approval lacks an exact D1/D2 artifact binding."
            raise TypeError(msg) from exc
        expected_allowed: object = (MATERIALIZE_FINAL_INPUTS_STAGE,)
    else:
        allowed_stages = approval.get("allowed_stages")
        bindings = approval.get("stage_bindings")
        decisions = approval.get("decisions")
        if not isinstance(decisions, list):
            msg = "Materialize approval lacks an exact D1/D2 decision sequence."
            raise TypeError(msg)
        selected: dict[str, str] = {}
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            decision_id = decision.get("decision_id")
            if decision_id not in {"D1", "D2"}:
                continue
            artifact = decision.get("canonical_artifact")
            if (
                decision_id in selected
                or not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256"}
            ):
                msg = "Materialize approval lacks an exact D1/D2 artifact binding."
                raise TypeError(msg)
            selected[decision_id] = artifact["sha256"]
        if set(selected) != {"D1", "D2"}:
            msg = "Materialize approval lacks an exact D1/D2 artifact binding."
            raise TypeError(msg)
        d1_sha256 = selected["D1"]
        d2_sha256 = selected["D2"]
        expected_allowed = [MATERIALIZE_FINAL_INPUTS_STAGE]
    if (
        allowed_stages != expected_allowed
        or not isinstance(bindings, Mapping)
        or set(bindings) != {MATERIALIZE_FINAL_INPUTS_STAGE}
    ):
        msg = "Input approval must grant only the materialize-final-inputs stage."
        raise ValueError(msg)
    expected = {
        "d1_canonical_artifact_sha256": _require_lowercase_sha256(
            d1_sha256,
            label="materialize approval D1 artifact SHA-256",
        ),
        "d2_canonical_artifact_sha256": _require_lowercase_sha256(
            d2_sha256,
            label="materialize approval D2 artifact SHA-256",
        ),
    }
    raw = bindings.get(MATERIALIZE_FINAL_INPUTS_STAGE)
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        msg = "Input approval stage binding does not match the exact D1/D2 artifacts."
        raise ValueError(msg)
    return expected


def _require_fit_stage_binding(
    approval: RevisionApproval | dict[str, Any],
    paths: RunPaths,
) -> dict[str, str]:
    """Bind fit authority to the two independently pinned input roots."""
    if isinstance(approval, RevisionApproval):
        _require_stage_scoped_fit_approval(approval)
        allowed_stages: object = approval.allowed_stages
        bindings: object = approval.stage_bindings
    else:
        allowed_stages = approval.get("allowed_stages")
        bindings = approval.get("stage_bindings")
    expected_allowed: object = (
        (FIT_SEALED_TCGA_K500_STAGE,)
        if isinstance(approval, RevisionApproval)
        else [FIT_SEALED_TCGA_K500_STAGE]
    )
    if (
        allowed_stages != expected_allowed
        or not isinstance(bindings, Mapping)
        or set(bindings) != {FIT_SEALED_TCGA_K500_STAGE}
    ):
        msg = "Fit approval must grant only the sealed-fit stage."
        raise ValueError(msg)
    raw = bindings.get(FIT_SEALED_TCGA_K500_STAGE)
    expected = {
        "canonical_input_manifest_sha256": _require_lowercase_sha256(
            paths.expected_canonical_input_sha256,
            label="expected canonical input manifest SHA-256",
        ),
        "provider_input_manifest_sha256": _require_lowercase_sha256(
            paths.expected_provider_input_manifest_sha256,
            label="expected provider input manifest SHA-256",
        ),
    }
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        msg = (
            "Fit approval stage binding does not match the independently pinned "
            "canonical and provider input roots."
        )
        raise ValueError(msg)
    return expected


def _canonical_input_binding(paths: RunPaths, cohort: str) -> dict[str, Any] | None:
    """Derive one verified canonical-MAF binding from independently pinned inputs."""
    if not _revision_authority_is_configured(paths):
        return None
    root = paths.canonical_input_root
    input_approval_path = paths.input_approval_manifest
    expected_input_approval = paths.expected_input_approval_sha256
    fit_approval_path = paths.fit_approval_manifest
    expected_fit_approval = paths.expected_fit_approval_sha256
    expected_root = paths.expected_canonical_input_sha256
    if (
        root is None
        or input_approval_path is None
        or expected_input_approval is None
        or fit_approval_path is None
        or expected_fit_approval is None
        or expected_root is None
    ):
        msg = "Revision authority configuration changed during validation."
        raise RuntimeError(msg)
    input_approval = _validated_input_approval(
        input_approval_path,
        expected_input_approval,
    )
    fit_approval = _validated_fit_approval(
        fit_approval_path,
        expected_fit_approval,
    )
    _require_fit_stage_binding(fit_approval, paths)
    fit_policy = validate_revision_fit_policy(
        fit_approval,
        expected_d4_implementation=REQUIRED_D4_IMPLEMENTATION,
        expected_tested_family=REQUIRED_TESTED_FAMILY,
    )
    if fit_policy.d5.tested_family != REQUIRED_TESTED_FAMILY:
        msg = "Signed D5 tested family does not match the executed K=500 family."
        raise ValueError(msg)
    _require_d3_runtime_contract(fit_policy, paths)
    validated_input = _validated_input_bundle(
        root,
        _require_lowercase_sha256(
            expected_root,
            label="expected canonical input manifest SHA-256",
        ),
        input_approval_path,
        expected_input_approval,
    )
    if set(validated_input) != {"manifest", "receipt", "receipt_sha256"}:
        msg = "Canonical full-validation result has an invalid closed schema."
        raise TypeError(msg)
    root_manifest = validated_input["manifest"]
    full_validation_receipt = validated_input["receipt"]
    full_validation_receipt_hash = _require_lowercase_sha256(
        validated_input["receipt_sha256"],
        label="canonical full-validation receipt SHA-256",
    )
    if not isinstance(root_manifest, dict) or not isinstance(
        full_validation_receipt,
        dict,
    ):
        msg = "Canonical full-validation result is structurally invalid."
        raise TypeError(msg)
    for decision_id in ("D1", "D2"):
        input_decision = input_approval.decisions[decision_id]
        fit_decision = fit_approval.decisions[decision_id]
        if _decision_reauthorization_record(
            input_decision,
        ) != _decision_reauthorization_record(fit_decision):
            msg = (
                "Fit approval does not exactly reauthorize input decision "
                f"{decision_id}."
            )
            raise ValueError(msg)
    cohort_binding = materialized_cohort_binding(root, root_manifest, cohort)
    provider_binding = _provider_cohort_binding(paths, cohort)
    child_path = cohort_binding["child_manifest"]["path"]
    canonical_path = cohort_binding["canonical_maf"]["path"]
    axis_path = cohort_binding["sample_axis"]["path"]
    population_path = cohort_binding["population_manifest"]["path"]
    child_record = _file_record(child_path)
    canonical_record = _file_record(canonical_path)
    axis_record = _file_record(axis_path)
    population_record = _file_record(population_path)
    provider_axis_path = provider_binding["sample_axis"]["path"]
    if (
        not isinstance(provider_axis_path, Path)
        or provider_axis_path.is_symlink()
        or not provider_axis_path.is_file()
        or _read_secure_regular_bytes(
            provider_axis_path,
            label=f"provider {cohort} sample axis",
        )
        != _read_secure_regular_bytes(
            axis_path,
            label=f"canonical {cohort} sample axis",
        )
    ):
        msg = f"Provider sample axis does not equal the signed axis for {cohort}."
        raise ValueError(msg)
    provider_canonical = provider_binding.get("canonical_inputs")
    canonical_names = {
        "child_manifest": child_record,
        "canonical_maf": canonical_record,
        "sample_axis": axis_record,
        "population_manifest": population_record,
    }
    if not isinstance(provider_canonical, dict) or any(
        _provider_file_record(provider_canonical.get(name), label=f"{cohort} {name}")[
            "sha256"
        ]
        != record["sha256"]
        for name, record in canonical_names.items()
    ):
        msg = f"Provider bundle does not bind the canonical inputs for {cohort}."
        raise ValueError(msg)
    return {
        "status": "verified",
        "contract": CANONICAL_INPUT_CONTRACT,
        "input_approval": {
            "manifest": _file_record(input_approval_path),
            "manifest_sha256": input_approval.manifest_sha256,
            "authorized_stage": MATERIALIZE_FINAL_INPUTS_STAGE,
            "decision_digests": {
                decision_id: input_approval.decision_digests[decision_id]
                for decision_id in ("D1", "D2")
            },
        },
        "fit_approval": {
            "manifest": _file_record(fit_approval_path),
            "manifest_sha256": fit_approval.manifest_sha256,
            "authorized_stage": FIT_SEALED_TCGA_K500_STAGE,
            "decision_digests": {
                decision_id: fit_approval.decision_digests[decision_id]
                for decision_id in ("D1", "D2", "D3", "D4", "D5", "D6")
            },
        },
        "fit_policy": _fit_policy_record(fit_policy),
        "input_manifest": _file_record(root / "input_manifest.json"),
        "full_validation": {
            "receipt": full_validation_receipt,
            "receipt_sha256": full_validation_receipt_hash,
        },
        "cohort_manifest": child_record,
        "canonical_maf": canonical_record,
        "authoritative_sample_axis": axis_record,
        "population_manifest": population_record,
        "canonical_maf_path": canonical_path.as_posix(),
        "authoritative_sample_axis_path": axis_path.as_posix(),
        "population_manifest_path": population_path.as_posix(),
    }


def _prime_parent_revision_authority(
    paths: RunPaths,
    cohort: str,
) -> dict[str, Any]:
    """Perform one closed-family replay for the parent orchestration boundary."""
    _validated_input_bundle.cache_clear()
    _validated_provider_bundle_cached.cache_clear()
    _validated_input_approval.cache_clear()
    _validated_fit_approval.cache_clear()
    provider = _validated_provider_bundle(paths)
    binding = _canonical_input_binding(paths, cohort)
    if binding is None:
        msg = "Production orchestration lost its complete revision authority."
        raise RuntimeError(msg)
    return _verify_provider_stage_authority(
        paths,
        cohort,
        binding,
        bundle=provider,
    )


def _verify_exact_file_record(
    record: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        msg = f"Frozen {label} receipt must be an object."
        raise TypeError(msg)
    observed_path = record.get("path")
    if (
        not isinstance(observed_path, str)
        or Path(observed_path) != expected_path
        or record.get("sha256") != expected_sha256
    ):
        msg = f"Frozen {label} receipt does not match its independent anchor."
        raise ValueError(msg)
    _verify_file_record(record)
    return record


def _approval_manifest_decisions(  # noqa: PLR0913
    authority: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    expected_stage: str,
    expected_decision_ids: tuple[str, ...],
    allowed_schemas: tuple[str, ...],
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(authority, dict) or set(authority) != {
        "manifest",
        "manifest_sha256",
        "authorized_stage",
        "decision_digests",
    }:
        msg = f"Frozen {label} approval receipt has an invalid schema."
        raise TypeError(msg)
    if (
        authority["manifest_sha256"] != expected_sha256
        or authority["authorized_stage"] != expected_stage
    ):
        msg = f"Frozen {label} approval does not match its CLI authority."
        raise ValueError(msg)
    _verify_exact_file_record(
        authority["manifest"],
        expected_path=expected_path,
        expected_sha256=expected_sha256,
        label=f"{label} approval manifest",
    )
    manifest = _read_json(expected_path)
    schema = manifest.get("schema")
    allowed_stages = manifest.get("allowed_stages")
    decisions = manifest.get("decisions")
    if schema not in allowed_schemas:
        msg = f"Pinned {label} approval has an invalid schema."
        raise ValueError(msg)
    if schema == APPROVAL_SCHEMA:
        canonical_decision_ids = DECISION_IDS
        valid_stage_envelope = allowed_stages == [expected_stage]
    elif schema == STAGE_SCOPED_APPROVAL_SCHEMA:
        minimum_decisions = STAGE_MINIMUM_DECISIONS.get(expected_stage)
        if minimum_decisions is None or expected_decision_ids != minimum_decisions:
            msg = f"Pinned {label} approval expected decision scope is invalid."
            raise ValueError(msg)
        canonical_decision_ids = expected_decision_ids
        valid_stage_envelope = allowed_stages == [expected_stage]
    else:
        msg = f"Pinned {label} approval expects an unsupported schema."
        raise ValueError(msg)
    if not valid_stage_envelope:
        msg = f"Pinned {label} approval has an invalid stage envelope."
        raise ValueError(msg)
    if not isinstance(decisions, list) or [
        decision.get("decision_id") if isinstance(decision, dict) else None
        for decision in decisions
    ] != list(canonical_decision_ids):
        msg = f"Pinned {label} approval has a noncanonical decision sequence."
        raise ValueError(msg)
    decision_by_id = {decision["decision_id"]: decision for decision in decisions}
    observed = {
        decision_id: _json_sha256(decision_by_id[decision_id])
        for decision_id in expected_decision_ids
    }
    expected = authority["decision_digests"]
    if not isinstance(expected, dict) or expected != observed:
        msg = f"Pinned {label} approval decision digests changed."
        raise ValueError(msg)
    return manifest, observed


def _parse_relative_authority_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        msg = f"{label} must be a normalized relative POSIX path."
        raise ValueError(msg)
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        msg = f"{label} must be a normalized relative POSIX path."
        raise ValueError(msg)
    return value


def _verify_fit_policy_receipts(
    policy: object,
    *,
    fit_manifest: dict[str, Any],
    fit_manifest_path: Path,
    fit_decision_digests: dict[str, str],
) -> None:
    if not isinstance(policy, dict) or set(policy) != {
        "d3",
        "d4",
        "d5",
        "d6",
        "receipts",
    }:
        msg = "Frozen fit-policy record has an invalid schema."
        raise TypeError(msg)
    d5 = policy["d5"]
    if (
        not isinstance(d5, dict)
        or d5.get("tested_family") != asdict(REQUIRED_TESTED_FAMILY)
        or "family" in d5
    ):
        msg = "Frozen D5 policy does not bind the executed K=500 tested family."
        raise ValueError(msg)
    receipts = policy["receipts"]
    if not isinstance(receipts, dict) or set(receipts) != {"D3", "D4", "D5", "D6"}:
        msg = "Frozen fit-policy receipts must cover exactly D3-D6."
        raise TypeError(msg)
    decisions = fit_manifest["decisions"]
    decision_by_id = {decision["decision_id"]: decision for decision in decisions}
    expected_receipt_keys = {
        "decision_id",
        "contract",
        "decision_digest",
        "canonical_artifact_path",
        "canonical_artifact_sha256",
        "canonical_artifact_size_bytes",
        "payload_sha256",
    }
    for decision_id, receipt in receipts.items():
        if not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys:
            msg = f"Frozen {decision_id} fit-policy receipt has an invalid schema."
            raise TypeError(msg)
        decision = decision_by_id[decision_id]
        artifact = decision.get("canonical_artifact")
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            msg = f"Pinned {decision_id} fit-policy artifact binding is invalid."
            raise TypeError(msg)
        relative = _parse_relative_authority_path(
            artifact["path"],
            label=f"{decision_id} fit-policy artifact path",
        )
        artifact_path = fit_manifest_path.parent.joinpath(
            *PurePosixPath(relative).parts,
        )
        if (
            receipt["decision_id"] != decision_id
            or receipt["decision_digest"] != fit_decision_digests[decision_id]
            or receipt["canonical_artifact_path"] != relative
            or receipt["canonical_artifact_sha256"] != artifact["sha256"]
        ):
            msg = f"Frozen {decision_id} fit-policy receipt changed."
            raise ValueError(msg)
        _verify_exact_file_record(
            {
                "path": artifact_path.as_posix(),
                "bytes": receipt["canonical_artifact_size_bytes"],
                "sha256": receipt["canonical_artifact_sha256"],
            },
            expected_path=artifact_path,
            expected_sha256=receipt["canonical_artifact_sha256"],
            label=f"{decision_id} fit-policy artifact",
        )
        envelope = _read_json(artifact_path)
        if (
            envelope.get("decision_id") != decision_id
            or envelope.get("contract") != receipt["contract"]
            or _json_sha256(envelope.get("payload")) != receipt["payload_sha256"]
        ):
            msg = f"Pinned {decision_id} fit-policy payload changed."
            raise ValueError(msg)


def _same_file_identity(first: object, second: object) -> bool:
    return (
        isinstance(first, dict)
        and isinstance(second, dict)
        and all(
            first.get(key) == second.get(key) for key in ("path", "bytes", "sha256")
        )
    )


def _verify_embedded_file_binding(
    embedded: object,
    frozen: object,
    *,
    root: Path,
    label: str,
) -> None:
    if not isinstance(embedded, dict) or set(embedded) != {"path", "bytes", "sha256"}:
        msg = f"Pinned {label} embedded receipt has an invalid schema."
        raise TypeError(msg)
    relative = _parse_relative_authority_path(
        embedded["path"],
        label=f"{label} embedded path",
    )
    expected_path = root.joinpath(*PurePosixPath(relative).parts)
    if (
        not isinstance(frozen, dict)
        or frozen.get("path") != expected_path.as_posix()
        or frozen.get("bytes") != embedded["bytes"]
        or frozen.get("sha256") != embedded["sha256"]
    ):
        msg = f"Pinned {label} embedded receipt differs from the frozen contract."
        raise ValueError(msg)


def _verify_canonical_cohort_receipts(
    paths: RunPaths,
    cohort: str,
    authority: dict[str, Any],
) -> None:
    root = paths.canonical_input_root
    expected_root_sha256 = paths.expected_canonical_input_sha256
    approval_manifest = paths.input_approval_manifest
    expected_approval_sha256 = paths.expected_input_approval_sha256
    if (
        root is None
        or expected_root_sha256 is None
        or approval_manifest is None
        or expected_approval_sha256 is None
    ):
        msg = "Canonical input authority disappeared during cohort verification."
        raise RuntimeError(msg)
    input_manifest_path = root / "input_manifest.json"
    _verify_exact_file_record(
        authority.get("input_manifest"),
        expected_path=input_manifest_path,
        expected_sha256=expected_root_sha256,
        label="canonical input root manifest",
    )
    full_validation = authority.get("full_validation")
    if (
        not isinstance(full_validation, dict)
        or set(full_validation) != {"receipt", "receipt_sha256"}
        or not isinstance(full_validation.get("receipt"), dict)
    ):
        msg = "Frozen canonical full-validation receipt has an invalid schema."
        raise TypeError(msg)
    full_validation_sha256 = _require_lowercase_sha256(
        full_validation["receipt_sha256"],
        label="canonical full-validation receipt SHA-256",
    )
    validated_scope = validate_materialized_input_cohort_binding(
        root,
        expected_root_sha256,
        approval_manifest,
        expected_approval_sha256,
        cohort,
        full_validation["receipt"],
        full_validation_sha256,
        require_current_execution_environment=False,
    )
    if (
        not isinstance(validated_scope, dict)
        or set(validated_scope)
        != {
            "manifest",
            "binding",
            "full_validation_receipt",
            "association_outputs_opened",
        }
        or validated_scope.get("full_validation_receipt") != full_validation["receipt"]
        or validated_scope.get("association_outputs_opened") is not False
    ):
        msg = f"Canonical scoped validation did not bind {cohort}."
        raise ValueError(msg)
    scoped_binding = validated_scope.get("binding")
    if not isinstance(scoped_binding, dict) or scoped_binding.get("cohort") != cohort:
        msg = f"Canonical scoped binding is invalid for {cohort}."
        raise TypeError(msg)
    expected_records = {
        "child_manifest": authority.get("cohort_manifest"),
        "canonical_maf": authority.get("canonical_maf"),
        "sample_axis": authority.get("authoritative_sample_axis"),
        "population_manifest": authority.get("population_manifest"),
    }
    for name, frozen in expected_records.items():
        observed = scoped_binding.get(name)
        if (
            not isinstance(frozen, dict)
            or not isinstance(observed, dict)
            or set(observed) != {"path", "file"}
            or not isinstance(observed.get("path"), Path)
            or observed["path"].as_posix() != frozen.get("path")
            or not isinstance(observed.get("file"), dict)
            or observed["file"].get("bytes") != frozen.get("bytes")
            or observed["file"].get("sha256") != frozen.get("sha256")
        ):
            msg = f"Frozen {cohort} canonical {name} receipt is invalid."
            raise ValueError(msg)


def _verify_provider_cohort_receipts(
    paths: RunPaths,
    cohort: str,
    authority: dict[str, Any],
    provenance: dict[str, Any],
    contract_inputs: object,
) -> None:
    root = paths.provider_input_root
    canonical_root = paths.canonical_input_root
    expected_root_sha256 = paths.expected_provider_input_manifest_sha256
    root_receipt = provenance.get("root_receipt")
    if (
        root is None
        or canonical_root is None
        or expected_root_sha256 is None
        or not isinstance(root_receipt, dict)
    ):
        msg = "Provider input authority disappeared during cohort verification."
        raise RuntimeError(msg)
    full_acceptance = root_receipt.get("full_acceptance_receipt")
    full_acceptance_sha256 = root_receipt.get(
        "full_acceptance_receipt_sha256",
    )
    validated_scope = validate_materialized_provider_cohort_input(
        root,
        expected_root_sha256,
        cohort,
        full_acceptance,
        full_acceptance_sha256,
        require_current_execution_environment=True,
    )
    expected_scope_keys = {
        "association_outputs_opened",
        "binding",
        "cohort",
        "execution_snapshot",
        "full_acceptance_receipt",
        "provider_receipt",
        "root",
    }
    if (
        not isinstance(validated_scope, dict)
        or set(validated_scope) != expected_scope_keys
        or validated_scope.get("root") != root
        or validated_scope.get("cohort") != cohort
        or validated_scope.get("full_acceptance_receipt") != full_acceptance
        or validated_scope.get("association_outputs_opened") is not False
        or validated_scope.get("provider_receipt") != provenance.get("cohort_receipt")
    ):
        msg = f"Narrow provider validation did not bind the frozen {cohort} scope."
        raise ValueError(msg)
    snapshot = validated_scope.get("execution_snapshot")
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"root", "tree_sha256", "validation_scope"}
        or snapshot.get("validation_scope")
        != "selected-cohort-and-exact-shared-closure"
    ):
        msg = "Narrow provider execution-snapshot receipt is invalid."
        raise ValueError(msg)
    narrow_binding = validated_scope.get("binding")
    expected_binding_names = {
        "cohort",
        "cohort_root",
        "mutsig_root",
        "count_matrix",
        "cbase_pmfs",
        "dig_pmfs",
        "sample_axis",
        "mutsig_lambda",
        "mutsig_metadata",
        "mutsig_genes",
        "mutsig_patients",
        "mutsig_receipt",
        "canonical_inputs",
        "canonical_input_receipts",
        "provider_receipt",
    }
    if not isinstance(narrow_binding, dict) or set(narrow_binding) != (
        expected_binding_names
    ):
        msg = f"Narrow provider {cohort} binding has an invalid closed schema."
        raise TypeError(msg)
    if (
        root_receipt.get("root") != root.as_posix()
        or root_receipt.get("expected_manifest_sha256") != expected_root_sha256
        or root_receipt.get("contract") != PROVIDER_INPUT_CONTRACT
        or root_receipt.get("association_outputs_opened") is not False
    ):
        msg = "Frozen provider root receipt differs from its independent authority."
        raise ValueError(msg)
    manifest_path = root / "provider_input_manifest.json"
    _verify_exact_file_record(
        root_receipt.get("manifest"),
        expected_path=manifest_path,
        expected_sha256=expected_root_sha256,
        label="provider input root manifest",
    )
    manifest = _read_json(manifest_path)
    sources = manifest.get("sources")
    embedded_git = sources.get("git_executable") if isinstance(sources, dict) else None
    frozen_git = root_receipt.get("git_executable")
    if (
        not isinstance(embedded_git, dict)
        or not isinstance(frozen_git, dict)
        or embedded_git != frozen_git
    ):
        msg = "Pinned provider Git executable receipt changed."
        raise ValueError(msg)
    _read_frozen_record_bytes(frozen_git, label="provider-authorized Git")
    cohort_receipts = manifest.get("cohort_provider_receipts")
    if not isinstance(cohort_receipts, list) or _json_sha256(
        cohort_receipts,
    ) != root_receipt.get("cohort_provider_receipts_sha256"):
        msg = "Pinned provider root cohort-receipt sequence changed."
        raise ValueError(msg)
    selected = [
        receipt
        for receipt in cohort_receipts
        if isinstance(receipt, dict) and receipt.get("cohort") == cohort
    ]
    if len(selected) != 1 or selected[0] != provenance.get("cohort_receipt"):
        msg = f"Pinned provider root does not select the frozen {cohort} receipt."
        raise ValueError(msg)
    canonical = selected[0].get("canonical_inputs")
    canonical_frozen = {
        "child_manifest": authority.get("cohort_manifest"),
        "canonical_maf": authority.get("canonical_maf"),
        "sample_axis": authority.get("authoritative_sample_axis"),
        "population_manifest": authority.get("population_manifest"),
    }
    if not isinstance(canonical, dict) or set(canonical) != set(canonical_frozen):
        msg = f"Provider {cohort} receipt lost its canonical-input hash binding."
        raise ValueError(msg)
    for name, frozen in canonical_frozen.items():
        _verify_embedded_file_binding(
            canonical.get(name),
            frozen,
            root=canonical_root,
            label=f"provider {cohort} canonical {name}",
        )
    narrow_canonical = narrow_binding.get("canonical_inputs")
    narrow_canonical_receipts = narrow_binding.get("canonical_input_receipts")
    if (
        not isinstance(narrow_canonical, dict)
        or set(narrow_canonical) != {"canonical_maf", "sample_axis"}
        or not isinstance(narrow_canonical_receipts, dict)
        or set(narrow_canonical_receipts) != {"child_manifest", "population_manifest"}
    ):
        msg = f"Narrow provider canonical binding is invalid for {cohort}."
        raise TypeError(msg)
    for name in ("canonical_maf", "sample_axis"):
        raw_observed = narrow_canonical.get(name)
        observed = _provider_file_record(
            raw_observed,
            label=f"narrow provider canonical {cohort}/{name}",
        )
        frozen = canonical_frozen[name]
        if not isinstance(raw_observed, dict):
            msg = f"Narrow provider canonical {cohort}/{name} is invalid."
            raise TypeError(msg)
        _verify_embedded_file_binding(
            raw_observed.get("file"),
            frozen,
            root=canonical_root,
            label=f"narrow provider canonical {cohort}/{name}",
        )
        if (
            not isinstance(frozen, dict)
            or observed.get("bytes") != frozen.get("bytes")
            or observed.get("sha256") != frozen.get("sha256")
        ):
            msg = f"Narrow provider canonical {cohort}/{name} changed."
            raise ValueError(msg)
    for name in ("child_manifest", "population_manifest"):
        observed = narrow_canonical_receipts.get(name)
        frozen = canonical_frozen[name]
        _verify_embedded_file_binding(
            observed,
            frozen,
            root=canonical_root,
            label=f"narrow provider canonical receipt {cohort}/{name}",
        )
    provider_files = provenance.get("files")
    expected_paths = {
        "count_matrix": paths.source_root / cohort / "count_matrix.csv",
        "cbase_pmfs": paths.source_root / cohort / "bmr_pmfs.csv",
        "dig_pmfs": paths.source_root / cohort / "bmr_pmfs.dig.csv",
        "sample_axis": paths.source_root / cohort / "sample_axis.txt",
        "mutsig_lambda": paths.mutsig_root / cohort / "persample_lambda.f32",
        "mutsig_metadata": paths.mutsig_root / cohort / "persample_meta.txt",
        "mutsig_genes": paths.mutsig_root / cohort / "persample_genes.txt",
        "mutsig_patients": paths.mutsig_root / cohort / "persample_patients.txt",
        "mutsig_receipt": paths.mutsig_root / cohort / "persample_receipt.tsv",
    }
    if not isinstance(provider_files, dict) or set(provider_files) != set(
        expected_paths,
    ):
        msg = f"Frozen provider {cohort} file set is invalid."
        raise TypeError(msg)
    for name in expected_paths:
        if not _same_file_identity(
            _provider_file_record(
                narrow_binding.get(name),
                label=f"narrow provider {cohort}/{name}",
            ),
            provider_files[name],
        ):
            msg = f"Narrow provider {cohort}/{name} differs from the task contract."
            raise ValueError(msg)
    for name, expected_path in expected_paths.items():
        record = provider_files[name]
        if not isinstance(record, dict):
            msg = f"Frozen provider {cohort}/{name} receipt is invalid."
            raise TypeError(msg)
        _verify_exact_file_record(
            record,
            expected_path=expected_path,
            expected_sha256=str(record.get("sha256")),
            label=f"provider {cohort}/{name}",
        )
    if not isinstance(contract_inputs, dict):
        msg = f"Frozen {cohort} input contract is invalid."
        raise TypeError(msg)
    mutsig_inputs = contract_inputs.get("mutsig", {})
    mutsig_files = (
        mutsig_inputs.get("files", {}) if isinstance(mutsig_inputs, dict) else {}
    )
    input_bindings = {
        "count_matrix": contract_inputs.get("counts"),
        "cbase_pmfs": contract_inputs.get("cbase"),
        "dig_pmfs": contract_inputs.get("dig"),
        "sample_axis": contract_inputs.get("sample_axis"),
        "mutsig_lambda": mutsig_files.get("lambda"),
        "mutsig_metadata": mutsig_files.get("metadata"),
        "mutsig_genes": mutsig_files.get("genes"),
        "mutsig_patients": mutsig_files.get("patients"),
        "mutsig_receipt": mutsig_files.get("receipt"),
    }
    if any(
        not _same_file_identity(provider_files[name], binding)
        for name, binding in input_bindings.items()
    ):
        msg = f"Frozen provider {cohort} file receipts differ from task inputs."
        raise ValueError(msg)


def _verify_frozen_cohort_authority(
    paths: RunPaths,
    contract: dict[str, Any],
) -> None:
    """Rehash only one task's frozen authorities without a whole-family replay."""
    if not _revision_authority_is_configured(paths):
        msg = "Production cohort verification requires complete revision authority."
        raise ValueError(msg)
    cohort = contract.get("cohort")
    authority = contract.get("revision_input_authority")
    provenance = contract.get("provider_input_provenance")
    if (
        not isinstance(cohort, str)
        or cohort not in TCGA_COHORTS
        or not isinstance(authority, dict)
        or not isinstance(provenance, dict)
    ):
        msg = "Frozen production contract lacks cohort-local revision authority."
        raise TypeError(msg)
    input_manifest_path = paths.input_approval_manifest
    fit_manifest_path = paths.fit_approval_manifest
    expected_input_sha256 = paths.expected_input_approval_sha256
    expected_fit_sha256 = paths.expected_fit_approval_sha256
    if (
        input_manifest_path is None
        or fit_manifest_path is None
        or expected_input_sha256 is None
        or expected_fit_sha256 is None
    ):
        msg = "Approval authority disappeared during cohort verification."
        raise RuntimeError(msg)
    input_manifest, _ = _approval_manifest_decisions(
        authority.get("input_approval"),
        expected_path=input_manifest_path,
        expected_sha256=expected_input_sha256,
        expected_stage=MATERIALIZE_FINAL_INPUTS_STAGE,
        expected_decision_ids=("D1", "D2"),
        allowed_schemas=(STAGE_SCOPED_APPROVAL_SCHEMA,),
        label="input",
    )
    _require_materialize_stage_binding(input_manifest)
    fit_manifest, fit_digests = _approval_manifest_decisions(
        authority.get("fit_approval"),
        expected_path=fit_manifest_path,
        expected_sha256=expected_fit_sha256,
        expected_stage=FIT_SEALED_TCGA_K500_STAGE,
        expected_decision_ids=("D1", "D2", "D3", "D4", "D5", "D6"),
        allowed_schemas=(STAGE_SCOPED_APPROVAL_SCHEMA,),
        label="fit",
    )
    _require_fit_stage_binding(fit_manifest, paths)
    _verify_fit_policy_receipts(
        authority.get("fit_policy"),
        fit_manifest=fit_manifest,
        fit_manifest_path=fit_manifest_path,
        fit_decision_digests=fit_digests,
    )
    _verify_canonical_cohort_receipts(paths, cohort, authority)
    _verify_provider_cohort_receipts(
        paths,
        cohort,
        authority,
        provenance,
        contract.get("inputs"),
    )


def _verify_provider_stage_authority(
    paths: RunPaths,
    cohort: str,
    binding: dict[str, Any],
    *,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validated = _validated_provider_bundle(paths) if bundle is None else bundle
    cohort_binding = _provider_cohort_binding(paths, cohort, bundle=validated)
    expected_keys = {
        "cohort",
        "cohort_root",
        "mutsig_root",
        "count_matrix",
        "cbase_pmfs",
        "dig_pmfs",
        "sample_axis",
        "mutsig_lambda",
        "mutsig_metadata",
        "mutsig_genes",
        "mutsig_patients",
        "mutsig_receipt",
        "canonical_inputs",
        "provider_receipt",
    }
    if (
        set(cohort_binding) != expected_keys
        or cohort_binding["cohort_root"] != paths.source_root / cohort
        or cohort_binding["mutsig_root"] != paths.mutsig_root / cohort
    ):
        msg = f"Validated provider cohort layout was substituted for {cohort}."
        raise ValueError(msg)
    provider_receipt = cohort_binding["provider_receipt"]
    if (
        not isinstance(provider_receipt, dict)
        or provider_receipt.get("cohort") != cohort
        or provider_receipt.get("association_outputs_opened") is not False
    ):
        msg = f"Validated provider cohort receipt is invalid for {cohort}."
        raise ValueError(msg)
    canonical_inputs = cohort_binding["canonical_inputs"]
    expected_canonical_hashes = {
        "child_manifest": binding["cohort_manifest"]["sha256"],
        "canonical_maf": binding["canonical_maf"]["sha256"],
        "sample_axis": binding["authoritative_sample_axis"]["sha256"],
        "population_manifest": binding["population_manifest"]["sha256"],
    }
    if not isinstance(canonical_inputs, dict) or any(
        _provider_file_record(canonical_inputs.get(name), label=f"{cohort} {name}")[
            "sha256"
        ]
        != expected_hash
        for name, expected_hash in expected_canonical_hashes.items()
    ):
        msg = f"Provider cohort receipt does not bind signed inputs for {cohort}."
        raise ValueError(msg)
    files = {
        name: _provider_file_record(cohort_binding[name], label=f"{cohort} {name}")
        for name in (
            "count_matrix",
            "cbase_pmfs",
            "dig_pmfs",
            "sample_axis",
            "mutsig_lambda",
            "mutsig_metadata",
            "mutsig_genes",
            "mutsig_patients",
            "mutsig_receipt",
        )
    }
    return {
        "contract": PROVIDER_INPUT_CONTRACT,
        "root_receipt": _provider_root_receipt(paths, validated),
        "cohort": cohort,
        "cohort_receipt": provider_receipt,
        "files": files,
        "association_outputs_opened": False,
    }


def _base_gene(feature: str) -> str:
    return feature.rsplit("_", 1)[0]


def iter_tested_pairs(features: Sequence[str]) -> Iterator[tuple[str, str]]:
    """Yield the frozen ordered pair universe, omitting same-base-gene pairs."""
    for feature_a, feature_b in combinations(features, 2):
        if _base_gene(feature_a) != _base_gene(feature_b):
            yield feature_a, feature_b


def _pair_contract(features: Sequence[str]) -> dict[str, Any]:
    same_base_excluded = 0
    digest = hashlib.sha256()
    row_count = 0
    for feature_a, feature_b in combinations(features, 2):
        if _base_gene(feature_a) == _base_gene(feature_b):
            same_base_excluded += 1
            continue
        encoded = f"{feature_a}\t{feature_b}".encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        row_count += 1
    return {
        "ordered_pair_sha256": digest.hexdigest(),
        "row_count": row_count,
        "same_base_pairs_excluded": same_base_excluded,
        "unfiltered_row_count": math.comb(len(features), 2),
    }


def _implemented_tested_family(top_k: int) -> TestedFamilyPolicy:
    """Describe the exact family produced by this runner for one requested K."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        msg = "Tested-family top_k must be a positive integer."
        raise ValueError(msg)
    return TestedFamilyPolicy(
        top_k=top_k,
        feature_ranking=TESTED_FAMILY_FEATURE_RANKING,
        tie_break=TESTED_FAMILY_TIE_BREAK,
        provider_support=TESTED_FAMILY_PROVIDER_SUPPORT,
        pair_construction=TESTED_FAMILY_PAIR_CONSTRUCTION,
        same_base_missense_nonsense=TESTED_FAMILY_SAME_BASE_POLICY,
        epsilon_pretest_filter=TESTED_FAMILY_NO_PRETEST_FILTER,
        marginal_effect_pretest_filter=TESTED_FAMILY_NO_PRETEST_FILTER,
        family=TESTED_FAMILY_SCOPE,
    )


def select_common_features(  # noqa: PLR0913
    counts: pd.DataFrame,
    *,
    cbase_features: set[str],
    dig_features: set[str],
    mutsig_genes: set[str],
    fully_supported_features: set[str] | None = None,
    top_k: int = TOP_K,
) -> tuple[list[str], dict[str, int]]:
    """Select one shared count-ranked native-support feature axis.

    Ranking is descending cohort mutation-event count. Equal totals preserve the
    original count-matrix column order, reproducing the historical CBaSE/DIG feature
    ranking while making the order immutable through the count-matrix hash.
    """
    native = {
        feature
        for feature in counts.columns
        if feature in cbase_features
        and feature in dig_features
        and FEATURE_PATTERN.fullmatch(feature) is not None
        and _base_gene(feature) in mutsig_genes
        and (fully_supported_features is None or feature in fully_supported_features)
    }
    totals = {str(feature): int(counts[feature].sum()) for feature in native}
    ordinal = {
        str(feature): position for position, feature in enumerate(counts.columns)
    }
    ordered = sorted(native, key=lambda feature: (-totals[feature], ordinal[feature]))
    if len(ordered) < top_k:
        msg = (
            f"Only {len(ordered)} features have native support in all three BMRs; "
            f"K={top_k} is impossible."
        )
        raise ValueError(msg)
    selected = ordered[:top_k]
    return selected, {feature: totals[feature] for feature in selected}


def _require_tested_family_contract(
    contract: dict[str, Any],
    *,
    require_signed_k500: bool,
) -> None:
    """Cross-check family metadata against the actual selection and pair universe."""
    if contract.get("schema_version") != SCHEMA_VERSION:
        msg = "Cohort contract schema version is incompatible with this runner."
        raise ValueError(msg)
    top_k = contract.get("top_k")
    implemented = _implemented_tested_family(top_k)
    if contract.get("tested_family") != asdict(implemented):
        msg = "Cohort contract does not describe the runner's exact tested family."
        raise ValueError(msg)
    features = contract.get("features")
    if not isinstance(features, list) or len(features) != top_k:
        msg = "Cohort contract feature axis does not contain exactly top_k features."
        raise ValueError(msg)
    feature_policy = contract.get("feature_policy")
    if not isinstance(feature_policy, dict) or feature_policy != {
        "feature_ranking": implemented.feature_ranking,
        "mutsig_cbase_feature_fallback": False,
        "observation_support": OBSERVATION_SUPPORT_UNIVERSE,
        "provider_support": implemented.provider_support,
        "tie_break": implemented.tie_break,
    }:
        msg = "Cohort contract feature selection does not match the tested family."
        raise ValueError(msg)
    expected_pair_policy = {
        "epsilon_pretest_filter": implemented.epsilon_pretest_filter,
        "marginal_effect_pretest_filter": implemented.marginal_effect_pretest_filter,
        "pair_construction": implemented.pair_construction,
        "same_base_missense_nonsense": implemented.same_base_missense_nonsense,
        **_pair_contract(features),
    }
    if contract.get("pair_policy") != expected_pair_policy:
        msg = "Cohort contract pair construction does not match the tested family."
        raise ValueError(msg)
    if tuple(BMRS) != ("cbase", "dig", "mutsig"):
        msg = "Runner BMR provider order does not match tested-family support."
        raise RuntimeError(msg)
    mutsig_pmf = contract.get("mutsig_pmf_contract")
    if not isinstance(mutsig_pmf, dict):
        msg = "Cohort contract lacks the closed MutSig PMF support contract."
        raise TypeError(msg)
    max_native_lambda = mutsig_pmf.get("max_selected_native_lambda")
    observed_kmax = mutsig_pmf.get("observed_kmax")
    if (
        not isinstance(max_native_lambda, (int, float))
        or isinstance(max_native_lambda, bool)
        or not isinstance(observed_kmax, int)
        or isinstance(observed_kmax, bool)
    ):
        msg = "Cohort MutSig PMF support coordinates are invalid."
        raise TypeError(msg)
    canonical_mutsig_pmf = validate_poisson_support_contract(
        mutsig_pmf,
        max_native_lambda=float(max_native_lambda),
        observed_kmax=observed_kmax,
    )
    if (
        canonical_mutsig_pmf["contract"] != PRODUCTION_POISSON_SUPPORT_CONTRACT
        or canonical_mutsig_pmf["support_rule"] != PRODUCTION_POISSON_SUPPORT_RULE
        or canonical_mutsig_pmf["normalization"] != PRODUCTION_POISSON_NORMALIZATION
        or canonical_mutsig_pmf["tail_tolerance"] != PRODUCTION_POISSON_TAIL_TOLERANCE
        or canonical_mutsig_pmf["effect_pages"] != MUTSIG_EFFECT_INDEX
    ):
        msg = "Runner MutSig PMF constants drifted from the production contract."
        raise RuntimeError(msg)
    samples = contract.get("samples")
    sample_count = samples.get("count") if isinstance(samples, dict) else None
    storage = contract.get("mutsig_pmf_storage_contract")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or not isinstance(storage, dict)
    ):
        msg = "Cohort contract lacks its MutSig PMF storage estimate."
        raise TypeError(msg)
    expected_storage = estimate_native_poisson_pmf_storage(
        len(features),
        sample_count,
        canonical_mutsig_pmf["inclusive_support_k"],
    )
    if (
        storage != expected_storage
        or storage.get("contract") != PRODUCTION_POISSON_STORAGE_CONTRACT
    ):
        msg = "Cohort MutSig PMF storage estimate drifted from its frozen axes."
        raise ValueError(msg)
    mutsig_inputs = contract.get("inputs", {}).get("mutsig", {})
    tensor_encoding = (
        mutsig_inputs.get("tensor_encoding")
        if isinstance(mutsig_inputs, dict)
        else None
    )
    if tensor_encoding != _mutsig_tensor_encoding_record(read_only=True):
        msg = "Cohort MutSig native-endian tensor encoding contract drifted."
        raise ValueError(msg)
    if require_signed_k500:
        authority = contract.get("revision_input_authority")
        fit_policy = (
            authority.get("fit_policy") if isinstance(authority, dict) else None
        )
        d5 = fit_policy.get("d5") if isinstance(fit_policy, dict) else None
        if (
            implemented != REQUIRED_TESTED_FAMILY
            or not isinstance(d5, dict)
            or d5.get("tested_family") != asdict(REQUIRED_TESTED_FAMILY)
        ):
            msg = "Production cohort does not bind signed D5 to the executed family."
            raise ValueError(msg)


def _parse_counts_csv(raw: bytes, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), index_col=0)
    if frame.empty or frame.shape[1] == 0:
        msg = f"Count matrix must have samples and features: {label}"
        raise ValueError(msg)
    frame.index = pd.Index([str(value) for value in frame.index])
    frame.columns = pd.Index([str(value) for value in frame.columns])
    if not frame.index.is_unique or not frame.columns.is_unique:
        msg = f"Count matrix axes must be unique: {label}"
        raise ValueError(msg)
    invalid = [
        feature
        for feature in frame.columns
        if FEATURE_PATTERN.fullmatch(feature) is None
    ]
    if invalid:
        msg = f"Count matrix contains invalid mutation-event IDs: {invalid[:5]}"
        raise ValueError(msg)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        msg = f"Count matrix values must be finite nonnegative integers: {label}"
        raise ValueError(msg)
    counts = numeric.astype(np.int64)
    for block in counts._mgr.blocks:  # noqa: SLF001
        block.values.flags.writeable = False  # noqa: PD011
    return counts


def _read_counts(path: Path) -> pd.DataFrame:
    return _parse_counts_csv(
        _read_secure_regular_bytes(path, label="count matrix"),
        label=path.as_posix(),
    )


def _parse_strict_pmfs_csv(
    raw: bytes,
    *,
    label: str,
) -> dict[str, dict[int, float]]:
    """Load exact integer-keyed PMFs without assuming contiguous count support."""
    frame = pd.read_csv(io.BytesIO(raw), index_col=0)
    frame.index = pd.Index([str(value) for value in frame.index])
    if frame.empty or not frame.index.is_unique:
        msg = f"BMR PMF table must contain a unique feature axis: {label}"
        raise ValueError(msg)
    try:
        count_keys = [int(str(column)) for column in frame.columns]
    except ValueError as error:
        msg = f"BMR PMF columns must be integer count keys: {label}"
        raise ValueError(msg) from error
    if any(key < 0 for key in count_keys) or len(count_keys) != len(set(count_keys)):
        msg = f"BMR PMF count keys must be unique nonnegative integers: {label}"
        raise ValueError(msg)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    finite_or_nan = np.isfinite(values) | np.isnan(values)
    if not finite_or_nan.all() or (np.nan_to_num(values, nan=0.0) < 0).any():
        msg = f"BMR PMF values must be finite, nonnegative, or padding NaN: {label}"
        raise ValueError(msg)
    row_sums = np.nansum(values, axis=1)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        msg = f"Every BMR PMF row must have positive mass: {label}"
        raise ValueError(msg)
    pmfs: dict[str, dict[int, float]] = {}
    for position, feature in enumerate(frame.index):
        pmf = {
            key: float(value)
            for key, value in zip(count_keys, values[position], strict=True)
            if not np.isnan(value)
        }
        total = sum(pmf.values())
        pmfs[feature] = {key: value / total for key, value in pmf.items()}
    return pmfs


def _load_strict_pmfs(path: Path) -> dict[str, dict[int, float]]:
    return _parse_strict_pmfs_csv(
        _read_secure_regular_bytes(path, label="BMR PMFs"),
        label=path.as_posix(),
    )


def _parse_canonical_utf8_lines(
    raw: bytes,
    *,
    path: Path,
    label: str,
) -> list[str]:
    """Parse one canonical UTF-8/LF sidecar from already captured bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"{label} is not valid UTF-8: {path}"
        raise ValueError(msg) from error
    values = text.splitlines()
    if raw != ("\n".join(values) + "\n").encode():
        msg = f"{label} must use LF separators and one terminal newline: {path}"
        raise ValueError(msg)
    return values


def _read_canonical_utf8_lines(path: Path, *, label: str) -> list[str]:
    return _parse_canonical_utf8_lines(
        _read_secure_regular_bytes(path, label=label),
        path=path,
        label=label,
    )


def _parse_mutsig_metadata(raw: bytes, *, path: Path) -> dict[str, int]:
    """Parse MutSig dimensions from the snapshot rather than reopening a path."""
    fields: dict[str, int] = {}
    for line in _parse_canonical_utf8_lines(
        raw,
        path=path,
        label="MutSig metadata",
    ):
        pieces = line.split("\t")
        if len(pieces) != 2 or pieces[0] in fields or not pieces[1].isdigit():
            msg = f"Invalid MutSig metadata row in {path}"
            raise ValueError(msg)
        fields[pieces[0]] = int(pieces[1])
    if set(fields) != {"ng", "np", "neff"} or fields["neff"] != 2:
        msg = f"MutSig metadata must contain positive ng/np and neff=2: {path}"
        raise ValueError(msg)
    if fields["ng"] <= 0 or fields["np"] <= 0:
        msg = f"MutSig ng and np must be positive: {path}"
        raise ValueError(msg)
    return fields


def _read_mutsig_metadata(path: Path) -> dict[str, int]:
    return _parse_mutsig_metadata(
        _read_secure_regular_bytes(path, label="MutSig metadata"),
        path=path,
    )


def _parse_axis(
    raw: bytes,
    expected: int,
    *,
    path: Path,
    label: str,
) -> list[str]:
    """Parse one unique MutSig axis from already captured bytes."""
    values = _parse_canonical_utf8_lines(
        raw,
        path=path,
        label=f"MutSig {label} axis",
    )
    if len(values) != expected or any(
        not value or value != value.strip() for value in values
    ):
        msg = f"MutSig {label} axis does not match metadata: {path}"
        raise ValueError(msg)
    if len(values) != len(set(values)):
        msg = f"MutSig {label} axis contains duplicate identifiers: {path}"
        raise ValueError(msg)
    return values


def _read_axis(path: Path, expected: int, *, label: str) -> list[str]:
    return _parse_axis(
        _read_secure_regular_bytes(path, label=f"MutSig {label} axis"),
        expected,
        path=path,
        label=label,
    )


def _parse_authoritative_sample_axis(raw: bytes, *, path: Path) -> list[str]:
    """Parse the canonical materialized tumor axis from snapshot bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"Authoritative sample axis is not UTF-8: {path}"
        raise ValueError(msg) from error
    values = text.splitlines()
    canonical = ("\n".join(values) + "\n").encode()
    if raw != canonical:
        msg = (
            "Authoritative sample axis must use UTF-8, LF separators, and one "
            f"terminal newline: {path}"
        )
        raise ValueError(msg)
    if not values or any(not value or value != value.strip() for value in values):
        msg = f"Authoritative sample axis contains a blank or padded ID: {path}"
        raise ValueError(msg)
    if len(values) != len(set(values)):
        msg = f"Authoritative sample axis contains duplicate IDs: {path}"
        raise ValueError(msg)
    if values != sorted(values):
        msg = f"Authoritative sample axis must be lexicographically ordered: {path}"
        raise ValueError(msg)
    return values


def _read_authoritative_sample_axis(path: Path) -> list[str]:
    """Read the materialized axis in its canonical UTF-8/LF byte convention."""
    return _parse_authoritative_sample_axis(
        _read_secure_regular_bytes(path, label="MutSig axis"),
        path=path,
    )


def _parse_mutsig_receipt(raw: bytes, *, path: Path) -> dict[str, str]:
    """Parse the closed MutSig receipt from already captured bytes."""
    fields: dict[str, str] = {}
    for line in _parse_canonical_utf8_lines(
        raw,
        path=path,
        label="MutSig receipt",
    ):
        pieces = line.split("\t")
        if len(pieces) != 2 or not pieces[0] or not pieces[1] or pieces[0] in fields:
            msg = f"Invalid MutSig receipt row in {path}"
            raise ValueError(msg)
        fields[pieces[0]] = pieces[1]
    if set(fields) != MUTSIG_RECEIPT_KEYS:
        missing = sorted(MUTSIG_RECEIPT_KEYS - set(fields))
        unexpected = sorted(set(fields) - MUTSIG_RECEIPT_KEYS)
        msg = (
            f"MutSig receipt has the wrong fields: {path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
        raise ValueError(msg)
    return fields


def _read_mutsig_receipt(path: Path) -> dict[str, str]:
    """Read the receipt published last by the tracked MutSig runner."""
    return _parse_mutsig_receipt(
        _read_secure_regular_bytes(path, label="MutSig receipt"),
        path=path,
    )


def _require_little_endian_mutsig_platform() -> dict[str, str]:
    """Reject hosts that cannot interpret the producer's native-endian fwrite."""
    if sys.byteorder != "little":
        msg = (
            "MutSig production input is native-endian Octave fwrite output; "
            "this runner requires sys.byteorder == 'little'."
        )
        raise RuntimeError(msg)
    return {
        "contract": MUTSIG_NATIVE_FWRITE_ENDIAN_CONTRACT,
        "producer_fwrite_byte_order": "native",
        "required_consumer_sys_byteorder": "little",
        "observed_consumer_sys_byteorder": sys.byteorder,
    }


def _reshape_mutsig_lambda_bytes(
    raw: bytes,
    dimensions: Mapping[str, int],
    *,
    path: Path,
) -> np.ndarray:
    """Return one read-only ``<f4`` Fortran gene/patient/effect tensor view."""
    ng = dimensions["ng"]
    patient_count = dimensions["np"]
    effect_count = dimensions["neff"]
    expected_bytes = ng * patient_count * effect_count * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        msg = (
            f"MutSig lambda tensor has {len(raw)} bytes; metadata requires "
            f"{expected_bytes}: {path}"
        )
        raise ValueError(msg)
    tensor = np.frombuffer(raw, dtype="<f4").reshape(
        (ng, patient_count, effect_count),
        order="F",
    )
    tensor.flags.writeable = False
    if tensor.dtype != np.dtype("<f4") or not tensor.flags.f_contiguous:
        msg = "MutSig lambda tensor lost its little-endian Fortran layout."
        raise RuntimeError(msg)
    return tensor


def _require_mutsig_tensor_layout_canary() -> str:
    """Exercise a nonuniform tensor that detects C-order and M/N page swaps."""
    expected = np.empty((2, 3, 2), dtype="<f4", order="F")
    for gene_position in range(2):
        for patient_position in range(3):
            for effect_position in range(2):
                expected[gene_position, patient_position, effect_position] = (
                    100 * gene_position + 10 * patient_position + effect_position + 0.25
                )
    parsed = _reshape_mutsig_lambda_bytes(
        expected.tobytes(order="F"),
        {"ng": 2, "np": 3, "neff": 2},
        path=Path("<internal-layout-canary>"),
    )
    if (
        MUTSIG_EFFECT_INDEX != {"M": 0, "N": 1}
        or not np.array_equal(parsed, expected)
        or float(parsed[1, 2, MUTSIG_EFFECT_INDEX["M"]]) != 120.25
        or float(parsed[1, 2, MUTSIG_EFFECT_INDEX["N"]]) != 121.25
        or parsed.flags.writeable
    ):
        msg = "MutSig nonuniform tensor layout canary failed."
        raise RuntimeError(msg)
    return MUTSIG_TENSOR_LAYOUT_CANARY


def _mutsig_tensor_encoding_record(*, read_only: bool) -> dict[str, Any]:
    """Return the exact native-fwrite interpretation bound into each contract."""
    return {
        **_require_little_endian_mutsig_platform(),
        "dtype": "<f4",
        "effect_pages": dict(MUTSIG_EFFECT_INDEX),
        "layout_canary": _require_mutsig_tensor_layout_canary(),
        "order": "Fortran-(gene,patient,effect)",
        "read_only": read_only,
    }


def _freeze_pmfs(
    pmfs: dict[str, dict[int, float]],
) -> Mapping[str, Mapping[int, float]]:
    """Freeze parsed PMFs so every later stage consumes the same objects."""
    return MappingProxyType(
        {
            feature: MappingProxyType(dict(probabilities))
            for feature, probabilities in pmfs.items()
        },
    )


def _snapshot_scientific_file(path: Path, *, label: str) -> ScientificFileSnapshot:
    """Capture exactly one stable descriptor read for one scientific input."""
    content, observed = _read_secure_regular_with_stat(path, label=label)
    return ScientificFileSnapshot(label, path, content, observed)


def _build_cohort_scientific_snapshot(
    *,
    count_path: Path,
    sample_axis_path: Path,
    cbase_path: Path,
    dig_path: Path,
    mutsig_dir: Path,
) -> CohortScientificSnapshot:
    """Read and parse the complete cohort scientific input set exactly once."""
    _require_little_endian_mutsig_platform()
    _require_mutsig_tensor_layout_canary()
    paths = {
        "counts": (count_path, "count matrix"),
        "sample_axis": (sample_axis_path, "authoritative sample axis"),
        "cbase": (cbase_path, "CBaSE PMFs"),
        "dig": (dig_path, "DIG PMFs"),
        "mutsig_metadata": (
            mutsig_dir / "persample_meta.txt",
            "MutSig metadata",
        ),
        "mutsig_genes": (
            mutsig_dir / "persample_genes.txt",
            "MutSig gene axis",
        ),
        "mutsig_patients": (
            mutsig_dir / "persample_patients.txt",
            "MutSig patient axis",
        ),
        "mutsig_lambda": (
            mutsig_dir / "persample_lambda.f32",
            "MutSig lambda tensor",
        ),
        "mutsig_receipt": (
            mutsig_dir / "persample_receipt.tsv",
            "MutSig receipt",
        ),
    }
    files = {
        name: _snapshot_scientific_file(path, label=label)
        for name, (path, label) in paths.items()
    }
    metadata_file = files["mutsig_metadata"]
    metadata = _parse_mutsig_metadata(
        metadata_file.content,
        path=metadata_file.path,
    )
    genes_file = files["mutsig_genes"]
    genes = _parse_axis(
        genes_file.content,
        metadata["ng"],
        path=genes_file.path,
        label="gene",
    )
    patients_file = files["mutsig_patients"]
    patients = _parse_axis(
        patients_file.content,
        metadata["np"],
        path=patients_file.path,
        label="patient",
    )
    lambda_file = files["mutsig_lambda"]
    lambdas = _reshape_mutsig_lambda_bytes(
        lambda_file.content,
        metadata,
        path=lambda_file.path,
    )
    counts_file = files["counts"]
    sample_axis_file = files["sample_axis"]
    cbase_file = files["cbase"]
    dig_file = files["dig"]
    receipt_file = files["mutsig_receipt"]
    return CohortScientificSnapshot(
        files=MappingProxyType(files),
        counts=_parse_counts_csv(
            counts_file.content,
            label=counts_file.path.as_posix(),
        ),
        authoritative_samples=tuple(
            _parse_authoritative_sample_axis(
                sample_axis_file.content,
                path=sample_axis_file.path,
            ),
        ),
        cbase_pmfs=_freeze_pmfs(
            _parse_strict_pmfs_csv(
                cbase_file.content,
                label=cbase_file.path.as_posix(),
            ),
        ),
        dig_pmfs=_freeze_pmfs(
            _parse_strict_pmfs_csv(
                dig_file.content,
                label=dig_file.path.as_posix(),
            ),
        ),
        mutsig_metadata=MappingProxyType(metadata),
        mutsig_genes=tuple(genes),
        mutsig_patients=tuple(patients),
        mutsig_lambdas=lambdas,
        mutsig_receipt=MappingProxyType(
            _parse_mutsig_receipt(receipt_file.content, path=receipt_file.path),
        ),
    )


def _stable_stat_coordinates(observed: os.stat_result) -> tuple[int, ...]:
    """Return stat fields that bind a snapshot to the persistent path object."""
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_uid,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _verify_scientific_snapshot_paths(snapshot: CohortScientificSnapshot) -> None:
    """Reject any persistent path replacement after snapshot-based computation."""
    for name, frozen in snapshot.files.items():
        current, observed = _read_visible_regular_with_stat(
            frozen.path,
            label=f"final scientific snapshot readback {name}",
        )
        observed_stat = _stable_stat_coordinates(observed)
        frozen_stat = _stable_stat_coordinates(frozen.stat_result)
        if current != frozen.content or observed_stat != frozen_stat:
            msg = (
                "Scientific input changed or was replaced after its immutable "
                f"snapshot: {frozen.path}"
            )
            raise ValueError(msg)


def _validate_mutsig_receipt(  # noqa: PLR0913
    mutsig_dir: Path,
    dimensions: dict[str, int],
    artifact_records: dict[str, dict[str, Any]],
    *,
    receipt: Mapping[str, str],
    receipt_record: dict[str, Any],
    authoritative_axis_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate input/source provenance and every receipt-bound sidecar."""
    receipt_path = mutsig_dir / "persample_receipt.tsv"
    if receipt["schema_version"] != MUTSIG_RECEIPT_SCHEMA_VERSION:
        msg = f"Unsupported MutSig receipt schema in {receipt_path}"
        raise ValueError(msg)
    if receipt["cohort"] != mutsig_dir.name:
        msg = f"MutSig receipt cohort does not match its directory: {receipt_path}"
        raise ValueError(msg)
    if receipt["upstream_commit"] != MUTSIG_UPSTREAM_COMMIT:
        msg = (
            f"MutSig receipt does not pin the required upstream commit: {receipt_path}"
        )
        raise ValueError(msg)
    if receipt["sample_axis_sha256"] != authoritative_axis_sha256:
        msg = (
            "MutSig receipt sample_axis_sha256 does not match the authoritative "
            f"sample_axis.txt: {receipt_path}"
        )
        raise ValueError(msg)

    sha256_fields = {key for key in MUTSIG_RECEIPT_KEYS if key.endswith("_sha256")}
    for key in sha256_fields:
        if re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None:
            msg = f"MutSig receipt {key} is not a lowercase SHA-256: {receipt_path}"
            raise ValueError(msg)
    if not receipt["sample_axis_count"].isdigit() or (
        int(receipt["sample_axis_count"]) != dimensions["np"]
    ):
        msg = (
            f"MutSig receipt sample-axis count does not match metadata: {receipt_path}"
        )
        raise ValueError(msg)
    if (
        not receipt["source_file_count"].isdigit()
        or int(receipt["source_file_count"]) <= 0
    ):
        msg = f"MutSig receipt source-file count is invalid: {receipt_path}"
        raise ValueError(msg)

    repo_root = Path(__file__).resolve().parents[1]
    expected_source_hashes = {
        "patch_sha256": _sha256(repo_root / MUTSIG_PATCH_PATH),
        "runner_sha256": _sha256(repo_root / MUTSIG_RUNNER_PATH),
    }
    for key, expected in expected_source_hashes.items():
        if receipt[key] != expected:
            msg = (
                f"MutSig receipt {key} does not match the tracked source: "
                f"{receipt_path}"
            )
            raise ValueError(msg)

    receipt_artifact_keys = {
        "lambda": "lambda_sha256",
        "metadata": "meta_sha256",
        "genes": "genes_sha256",
        "patients": "patients_sha256",
    }
    for artifact, key in receipt_artifact_keys.items():
        if receipt[key] != artifact_records[artifact]["sha256"]:
            msg = f"MutSig receipt hash does not match {artifact}: {receipt_path}"
            raise ValueError(msg)
    return dict(receipt), receipt_record


def _mutsig_contract(
    snapshot: CohortScientificSnapshot,
    mutsig_dir: Path,
    count_samples: Sequence[str],
    *,
    authoritative_axis_sha256: str,
) -> tuple[set[str], dict[str, Any]]:
    fields = dict(snapshot.mutsig_metadata)
    genes = snapshot.mutsig_genes
    patients = snapshot.mutsig_patients
    ordered_count_samples = [str(sample) for sample in count_samples]
    patient_position = {patient: position for position, patient in enumerate(patients)}
    count_sample_set = set(ordered_count_samples)
    patient_set = set(patients)
    missing = [sample for sample in ordered_count_samples if sample not in patient_set]
    extra = [patient for patient in patients if patient not in count_sample_set]
    if ordered_count_samples != list(patients):
        msg = (
            "Count-matrix sample axis must exactly equal the ordered MutSig patient "
            f"axis; missing_in_mutsig={missing[:5]}, extra_in_mutsig={extra[:5]}, "
            f"same_set={count_sample_set == patient_set}."
        )
        raise ValueError(msg)
    mapping = [
        f"{sample}\t{patient_position[sample]}" for sample in ordered_count_samples
    ]
    ordered_axis_sha256 = _sequence_sha256(ordered_count_samples)
    artifact_records = {
        "lambda": snapshot.files["mutsig_lambda"].record(),
        "metadata": snapshot.files["mutsig_metadata"].record(),
        "genes": snapshot.files["mutsig_genes"].record(),
        "patients": snapshot.files["mutsig_patients"].record(),
    }
    receipt, receipt_record = _validate_mutsig_receipt(
        mutsig_dir,
        fields,
        artifact_records,
        receipt=snapshot.mutsig_receipt,
        receipt_record=snapshot.files["mutsig_receipt"].record(),
        authoritative_axis_sha256=authoritative_axis_sha256,
    )
    return set(genes), {
        "dimensions": fields,
        "receipt": {
            "schema_version": receipt["schema_version"],
            "upstream_commit": receipt["upstream_commit"],
            "source_tree_sha256": receipt["source_tree_sha256"],
            "source_file_count": int(receipt["source_file_count"]),
            "patch_sha256": receipt["patch_sha256"],
            "runner_sha256": receipt["runner_sha256"],
            "runtime_sha256": receipt["runtime_sha256"],
            "maf_sha256": receipt["maf_sha256"],
            "canonical_maf_binding": {
                "status": MUTSIG_MAF_BINDING_STATUS,
                "required_before_production": MUTSIG_MAF_BINDING_REQUIREMENT,
            },
            "sample_axis_sha256": receipt["sample_axis_sha256"],
            "sample_axis_count": int(receipt["sample_axis_count"]),
        },
        "sample_mapping": {
            "contract": SAMPLE_AXIS_CONTRACT,
            "cohort_samples": len(ordered_count_samples),
            "matched_samples": len(ordered_count_samples),
            "extra_mutsig_samples": 0,
            "cohort_mean_fallback_samples": 0,
            "exact_order_match": True,
            "ordered_count_ids_sha256": ordered_axis_sha256,
            "ordered_mutsig_ids_sha256": ordered_axis_sha256,
            "ordered_mapping_sha256": _sequence_sha256(mapping),
        },
        "tensor_encoding": _mutsig_tensor_encoding_record(
            read_only=not snapshot.mutsig_lambdas.flags.writeable,
        ),
        "files": {**artifact_records, "receipt": receipt_record},
    }


def _shared_pmf_has_observation_support(
    observed: np.ndarray,
    pmf: dict[int, float],
) -> bool:
    return all(
        float(pmf.get(int(count), 0.0)) + float(pmf.get(int(count) - 1, 0.0)) > 0.0
        for count in np.unique(observed)
    )


def build_full_support_universe(  # noqa: PLR0913
    counts: pd.DataFrame,
    *,
    cbase_pmfs: Mapping[str, Mapping[int, float]],
    dig_pmfs: Mapping[str, Mapping[int, float]],
    mutsig_lambdas: np.ndarray,
    mutsig_genes: Sequence[str],
    mutsig_patients: Sequence[str],
) -> tuple[set[str], dict[str, Any]]:
    """Return features with native, full observation support in all three BMRs."""
    if not np.isfinite(mutsig_lambdas).all() or (mutsig_lambdas < 0).any():
        msg = "MutSig lambda values must be finite and nonnegative."
        raise ValueError(msg)
    if len(mutsig_genes) != len(set(mutsig_genes)):
        msg = "MutSig gene identifiers must be unique."
        raise ValueError(msg)
    if len(mutsig_patients) != len(set(mutsig_patients)):
        msg = "MutSig patient identifiers must be unique."
        raise ValueError(msg)
    gene_position = {gene: position for position, gene in enumerate(mutsig_genes)}
    patient_position = {
        patient: position for position, patient in enumerate(mutsig_patients)
    }
    missing_samples = [
        str(sample) for sample in counts.index if str(sample) not in patient_position
    ]
    if missing_samples:
        msg = f"MutSig lacks count-matrix samples: {missing_samples[:5]}"
        raise ValueError(msg)
    sample_positions = np.array(
        [patient_position[str(sample)] for sample in counts.index],
        dtype=np.int64,
    )

    eligible: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    provider_reason_counts = {
        provider: {"missing_native_background": 0, "zero_observation_support": 0}
        for provider in BMRS
    }
    for feature in counts.columns:
        base, effect = feature.rsplit("_", 1)
        reasons: list[dict[str, str]] = []
        if feature not in cbase_pmfs:
            reasons.append(
                {"provider": "cbase", "reason": "missing_native_background"},
            )
        if feature not in dig_pmfs:
            reasons.append(
                {"provider": "dig", "reason": "missing_native_background"},
            )
        if base not in gene_position or effect not in MUTSIG_EFFECT_INDEX:
            reasons.append(
                {"provider": "mutsig", "reason": "missing_native_background"},
            )
        if not reasons:
            observed = counts[feature].to_numpy(dtype=np.int64)
            if not _shared_pmf_has_observation_support(
                observed,
                cbase_pmfs[feature],
            ):
                reasons.append(
                    {"provider": "cbase", "reason": "zero_observation_support"},
                )
            if not _shared_pmf_has_observation_support(observed, dig_pmfs[feature]):
                reasons.append(
                    {"provider": "dig", "reason": "zero_observation_support"},
                )
            lambda_values = mutsig_lambdas[
                gene_position[base],
                sample_positions,
                MUTSIG_EFFECT_INDEX[effect],
            ]
            mutsig_support = poisson.pmf(observed, lambda_values) + poisson.pmf(
                observed - 1,
                lambda_values,
            )
            if not np.isfinite(mutsig_support).all() or not (mutsig_support > 0).all():
                reasons.append(
                    {"provider": "mutsig", "reason": "zero_observation_support"},
                )
        if reasons:
            exclusions.append({"feature": feature, "reasons": reasons})
            for reason in reasons:
                provider_reason_counts[reason["provider"]][reason["reason"]] += 1
        else:
            eligible.add(feature)

    report = {
        "contract": OBSERVATION_SUPPORT_UNIVERSE,
        "support_rule": "P(B=c) + P(B=c-1) > 0 for every cohort sample",
        "count_matrix_features": len(counts.columns),
        "eligible_features": len(eligible),
        "excluded_features": exclusions,
        "excluded_feature_count": len(exclusions),
        "excluded_features_sha256": _json_sha256(exclusions),
        "provider_reason_counts": provider_reason_counts,
    }
    return eligible, report


def audit_background_support(
    counts: pd.DataFrame,
    features: Sequence[str],
    pmfs: dict[str, Any],
) -> dict[str, Any]:
    """Audit exact observed-count support and its tested-pair implications.

    A feature/sample observation is supported when at least one latent driver state
    can explain it, i.e. ``P(B=c) + P(B=c-1) > 0``. Pair support is the intersection
    of its two feature masks. The compact hashes bind every ordered feature and pair
    mask without writing a large boolean matrix to the manifest.
    """
    n_samples = len(counts)
    mask_bytes = (n_samples + 7) // 8
    sample_zero_counts = np.zeros(n_samples, dtype=np.int64)
    feature_masks: list[int] = []
    feature_mask_bytes: list[bytes] = []
    zero_by_feature: dict[str, int] = {}
    feature_digest = hashlib.sha256()

    for feature in features:
        background = pmfs.get(feature)
        if background is None:
            msg = f"Support audit is missing PMFs for {feature}."
            raise ValueError(msg)
        per_sample = (
            background if isinstance(background, list) else [background] * n_samples
        )
        if len(per_sample) != n_samples:
            msg = f"Support audit PMF/sample length mismatch for {feature}."
            raise ValueError(msg)
        observed = counts[feature].to_numpy(dtype=np.int64)
        unsupported = np.fromiter(
            (
                float(pmf.get(int(count), 0.0)) + float(pmf.get(int(count) - 1, 0.0))
                == 0.0
                for count, pmf in zip(observed, per_sample, strict=True)
            ),
            dtype=bool,
            count=n_samples,
        )
        sample_zero_counts += unsupported
        zero_count = int(unsupported.sum())
        if zero_count:
            zero_by_feature[feature] = zero_count
        packed = np.packbits(unsupported, bitorder="little").tobytes()
        if len(packed) != mask_bytes:
            msg = "Packed support mask has an unexpected byte length."
            raise ValueError(msg)
        feature_mask_bytes.append(packed)
        feature_masks.append(int.from_bytes(packed, "little"))
        encoded = feature.encode("utf-8")
        feature_digest.update(len(encoded).to_bytes(8, "big"))
        feature_digest.update(encoded)
        feature_digest.update(packed)

    pair_digest = hashlib.sha256()
    excluded_histogram: dict[int, int] = {}
    unique_pair_masks: set[int] = set()
    pair_count = 0
    for index_a, index_b in combinations(range(len(features)), 2):
        if _base_gene(features[index_a]) == _base_gene(features[index_b]):
            continue
        mask = feature_masks[index_a] | feature_masks[index_b]
        excluded = mask.bit_count()
        excluded_histogram[excluded] = excluded_histogram.get(excluded, 0) + 1
        unique_pair_masks.add(mask)
        pair_digest.update(mask.to_bytes(mask_bytes, "little"))
        pair_count += 1

    feature_mask_count = len(set(feature_masks))
    full_pairs = excluded_histogram.get(0, 0)
    return {
        "support_rule": "P(B=c) + P(B=c-1) > 0 exactly",
        "feature_sample_observations": len(features) * n_samples,
        "zero_support_feature_samples": int(sample_zero_counts.sum()),
        "features_with_zero_support": len(zero_by_feature),
        "samples_with_any_zero_support": int(np.sum(sample_zero_counts > 0)),
        "max_zero_support_features_per_sample": int(
            sample_zero_counts.max(initial=0),
        ),
        "zero_support_by_feature": zero_by_feature,
        "ordered_feature_masks_sha256": feature_digest.hexdigest(),
        "unique_feature_masks": feature_mask_count,
        "pairs": {
            "tested": pair_count,
            "full_sample_support": full_pairs,
            "with_excluded_samples": pair_count - full_pairs,
            "excluded_sample_count_histogram": {
                str(key): excluded_histogram[key] for key in sorted(excluded_histogram)
            },
            "max_excluded_samples": max(excluded_histogram, default=0),
            "unique_effective_sample_masks": len(unique_pair_masks),
            "ordered_effective_masks_sha256": pair_digest.hexdigest(),
        },
        "inference_implication": {
            "all_features_have_full_sample_support": not any(feature_masks),
            "all_features_share_one_effective_sample_mask": feature_mask_count == 1,
            "global_feature_pi_is_exact_pair_null_on_full_sample_set": (
                not any(feature_masks)
            ),
            "pair_specific_effective_mask_refit_required": feature_mask_count > 1,
        },
    }


def _require_full_observation_support(contract: dict[str, Any]) -> None:
    universe = contract.get("full_support_universe", {})
    exclusions = universe.get("excluded_features", [])
    if universe.get("contract") != OBSERVATION_SUPPORT_UNIVERSE:
        msg = "Cohort contract does not use the frozen full-support universe."
        raise ValueError(msg)
    if universe.get("excluded_features_sha256") != _json_sha256(exclusions):
        msg = "Full-support exclusion list hash does not match the cohort contract."
        raise ValueError(msg)
    expected_pairs = contract["pair_policy"]["row_count"]
    for bmr in BMRS:
        audit = contract["observed_count_support_audit"][bmr]
        if (
            audit["zero_support_feature_samples"] != 0
            or audit["pairs"]["full_sample_support"] != expected_pairs
            or audit["pairs"]["with_excluded_samples"] != 0
            or not audit["inference_implication"][
                "global_feature_pi_is_exact_pair_null_on_full_sample_set"
            ]
        ):
            msg = (
                f"Selected {bmr} feature universe lacks full observation support; "
                "global single-gene constrained-null marginals are not valid for "
                "every tested pair."
            )
            raise ValueError(msg)


def _require_exact_sample_axis(contract: dict[str, Any]) -> None:
    """Require one authoritative ordered tumor axis across all backgrounds."""
    samples = contract.get("samples", {})
    count = samples.get("count")
    if (
        samples.get("contract") != SAMPLE_AXIS_CONTRACT
        or not samples.get("exact_order_match")
        or samples.get("extra_mutsig_samples") != 0
        or samples.get("cohort_mean_fallback_samples") != 0
        or samples.get("cohort_samples") != count
        or samples.get("authoritative_samples") != count
        or samples.get("matched_samples") != count
        or samples.get("ordered_ids_sha256") != samples.get("ordered_count_ids_sha256")
        or samples.get("ordered_ids_sha256") != samples.get("ordered_mutsig_ids_sha256")
        or samples.get("ordered_ids_sha256")
        != samples.get("ordered_authoritative_ids_sha256")
    ):
        msg = "Cohort contract does not bind one exact ordered sample axis."
        raise ValueError(msg)


def _require_canonical_mutsig_maf_binding(contract: dict[str, Any]) -> None:
    """Block production until the receipt MAF hash has an approved authority."""
    binding = (
        contract.get("inputs", {})
        .get("mutsig", {})
        .get("receipt", {})
        .get("canonical_maf_binding", {})
    )
    if binding.get("status") != "verified":
        requirement = binding.get(
            "required_before_production",
            MUTSIG_MAF_BINDING_REQUIREMENT,
        )
        msg = f"K=500 launch blocked by MutSig MAF provenance stop-ship: {requirement}."
        raise RuntimeError(msg)


def build_cohort_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Build a deterministic fail-closed contract for one cohort."""
    if top_k == TOP_K:
        _frozen_python_executable()
    if top_k == TOP_K and not _revision_authority_is_configured(paths):
        msg = "Production K=500 contracts require the pinned provider bundle."
        raise ValueError(msg)
    canonical_binding = _canonical_input_binding(paths, cohort)
    if canonical_binding is None or top_k != TOP_K:
        cohort_dir = paths.source_root / cohort
        count_path = cohort_dir / "count_matrix.csv"
        sample_axis_path = cohort_dir / "sample_axis.txt"
        cbase_path = cohort_dir / "bmr_pmfs.csv"
        dig_path = cohort_dir / "bmr_pmfs.dig.csv"
        mutsig_dir = paths.mutsig_root / cohort
    else:
        provider_binding = _provider_cohort_binding(paths, cohort)
        count_path = provider_binding["count_matrix"]["path"]
        sample_axis_path = provider_binding["sample_axis"]["path"]
        cbase_path = provider_binding["cbase_pmfs"]["path"]
        dig_path = provider_binding["dig_pmfs"]["path"]
        mutsig_dir = provider_binding["mutsig_root"]
    snapshot = _build_cohort_scientific_snapshot(
        count_path=count_path,
        sample_axis_path=sample_axis_path,
        cbase_path=cbase_path,
        dig_path=dig_path,
        mutsig_dir=mutsig_dir,
    )
    counts = snapshot.counts
    ordered_count_samples = [str(sample) for sample in counts.index]
    authoritative_samples = list(snapshot.authoritative_samples)
    if ordered_count_samples != authoritative_samples:
        msg = (
            "Count-matrix sample axis must exactly equal authoritative "
            f"sample_axis.txt for {cohort}."
        )
        raise ValueError(msg)
    authoritative_axis_sha256 = snapshot.files["sample_axis"].record()["sha256"]
    cbase_pmfs = snapshot.cbase_pmfs
    dig_pmfs = snapshot.dig_pmfs
    cbase_features = set(cbase_pmfs)
    dig_features = set(dig_pmfs)
    mutsig_genes, mutsig = _mutsig_contract(
        snapshot,
        mutsig_dir,
        ordered_count_samples,
        authoritative_axis_sha256=authoritative_axis_sha256,
    )
    provider_provenance = None
    if canonical_binding is not None:
        receipt = mutsig["receipt"]
        if receipt["maf_sha256"] != canonical_binding["canonical_maf"]["sha256"]:
            msg = f"MutSig receipt does not bind the signed canonical MAF: {cohort}"
            raise ValueError(msg)
        if (
            receipt["sample_axis_sha256"]
            != canonical_binding["authoritative_sample_axis"]["sha256"]
        ):
            msg = f"MutSig receipt does not bind the signed sample axis: {cohort}"
            raise ValueError(msg)
        provider_provenance = _verify_provider_stage_authority(
            paths,
            cohort,
            canonical_binding,
        )
        receipt["canonical_maf_binding"] = {
            "status": "verified",
            "contract": canonical_binding["contract"],
            "input_approval_manifest_sha256": canonical_binding["input_approval"][
                "manifest_sha256"
            ],
            "fit_approval_manifest_sha256": canonical_binding["fit_approval"][
                "manifest_sha256"
            ],
            "input_manifest_sha256": canonical_binding["input_manifest"]["sha256"],
            "cohort_manifest_sha256": canonical_binding["cohort_manifest"]["sha256"],
            "canonical_maf_sha256": canonical_binding["canonical_maf"]["sha256"],
        }
    fully_supported_features, support_universe = build_full_support_universe(
        counts,
        cbase_pmfs=cbase_pmfs,
        dig_pmfs=dig_pmfs,
        mutsig_lambdas=snapshot.mutsig_lambdas,
        mutsig_genes=snapshot.mutsig_genes,
        mutsig_patients=snapshot.mutsig_patients,
    )
    features, totals = select_common_features(
        counts,
        cbase_features=cbase_features,
        dig_features=dig_features,
        mutsig_genes=mutsig_genes,
        fully_supported_features=fully_supported_features,
        top_k=top_k,
    )
    selected_counts = counts.loc[:, features]
    selected_observed_kmax = int(selected_counts.to_numpy().max(initial=0))
    selected_native_rates = _selected_mutsig_native_rates(
        snapshot.mutsig_lambdas,
        snapshot.mutsig_genes,
        snapshot.mutsig_patients,
        ordered_count_samples,
        features,
    )
    mutsig_pmf_contract = build_poisson_support_contract(
        _max_selected_native_lambda(selected_native_rates),
        selected_observed_kmax,
    )
    mutsig_pmf_storage_contract = estimate_native_poisson_pmf_storage(
        len(features),
        len(selected_counts),
        mutsig_pmf_contract["inclusive_support_k"],
    )
    selected_pmfs: dict[str, Mapping[str, Any]] = {
        "cbase": MappingProxyType(
            {feature: cbase_pmfs[feature] for feature in features},
        ),
        "dig": MappingProxyType(
            {feature: dig_pmfs[feature] for feature in features},
        ),
        "mutsig": MappingProxyType(
            build_native_poisson_pmfs(
                selected_native_rates,
                mutsig_pmf_contract,
            ),
        ),
    }
    support_audit = {}
    for bmr, pmfs in selected_pmfs.items():
        support_audit[bmr] = audit_background_support(
            selected_counts,
            features,
            pmfs,
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "top_k": top_k,
        "tested_family": asdict(_implemented_tested_family(top_k)),
        "feature_policy": {
            "feature_ranking": TESTED_FAMILY_FEATURE_RANKING,
            "mutsig_cbase_feature_fallback": False,
            "observation_support": OBSERVATION_SUPPORT_UNIVERSE,
            "provider_support": TESTED_FAMILY_PROVIDER_SUPPORT,
            "tie_break": TESTED_FAMILY_TIE_BREAK,
        },
        "pair_policy": {
            "epsilon_pretest_filter": TESTED_FAMILY_NO_PRETEST_FILTER,
            "marginal_effect_pretest_filter": TESTED_FAMILY_NO_PRETEST_FILTER,
            "pair_construction": TESTED_FAMILY_PAIR_CONSTRUCTION,
            "same_base_missense_nonsense": TESTED_FAMILY_SAME_BASE_POLICY,
            **_pair_contract(features),
        },
        "samples": {
            "count": len(counts),
            "authoritative_samples": len(authoritative_samples),
            "ordered_ids_sha256": _sequence_sha256(ordered_count_samples),
            "ordered_authoritative_ids_sha256": _sequence_sha256(
                authoritative_samples,
            ),
            **mutsig["sample_mapping"],
        },
        "features": features,
        "feature_totals": totals,
        "cutoff_tie": {
            "total": totals[features[-1]],
            "eligible_features": sum(
                feature in fully_supported_features
                and int(counts[feature].sum()) == totals[features[-1]]
                for feature in counts.columns
            ),
            "selected_features": sum(
                total == totals[features[-1]] for total in totals.values()
            ),
        },
        "ordered_features_sha256": _sequence_sha256(features),
        "native_support": {
            "cbase_features": len(cbase_features),
            "dig_features": len(dig_features),
            "mutsig_genes": len(mutsig_genes),
            "common_features": sum(
                feature in cbase_features
                and feature in dig_features
                and _base_gene(feature) in mutsig_genes
                for feature in counts.columns
            ),
            "full_observation_support_common_features": len(
                fully_supported_features,
            ),
            "selected_features": len(features),
        },
        "full_support_universe": support_universe,
        "mutsig_pmf_contract": mutsig_pmf_contract,
        "mutsig_pmf_storage_contract": mutsig_pmf_storage_contract,
        "observed_count_support_audit": support_audit,
        "inputs": {
            "counts": snapshot.files["counts"].record(),
            "sample_axis": snapshot.files["sample_axis"].record(),
            "cbase": snapshot.files["cbase"].record(),
            "dig": snapshot.files["dig"].record(),
            "mutsig": mutsig,
        },
    }
    if canonical_binding is not None:
        contract["revision_input_authority"] = canonical_binding
        contract["provider_input_provenance"] = provider_provenance
    _require_tested_family_contract(
        contract,
        require_signed_k500=top_k == TOP_K,
    )
    _require_exact_sample_axis(contract)
    _require_full_observation_support(contract)
    _verify_scientific_snapshot_paths(snapshot)
    return contract


def _contract_path(paths: RunPaths, cohort: str) -> Path:
    return paths.output_root / "contracts" / f"{cohort}.json"


def _task_dir(paths: RunPaths, task: Task) -> Path:
    return paths.output_root / "tasks" / task.cohort / task.bmr


def _ensure_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    current = build_cohort_contract(paths, cohort, top_k=top_k)
    _verify_contract_scientific_records(current)
    path = _contract_path(paths, cohort)
    if path.exists():
        existing = _read_json(path)
        if existing != current:
            msg = f"Input drift detected for frozen cohort contract: {cohort}"
            raise ValueError(msg)
    else:
        _write_json_atomic(path, current)
    return current


def _verify_file_record(record: dict[str, Any]) -> None:
    _read_frozen_record_bytes(record, label="input")


def _read_frozen_record_bytes(record: object, *, label: str) -> bytes:
    allowed_schemas = (
        {"bytes", "path", "sha256"},
        {"bytes", "mtime_ns", "path", "sha256"},
        {
            "bytes",
            "ctime_ns",
            "device",
            "inode",
            "mode",
            "mtime_ns",
            "nlink",
            "path",
            "sha256",
            "uid",
        },
    )
    if not isinstance(record, dict) or set(record) not in allowed_schemas:
        msg = f"Frozen {label} has an invalid file-record schema."
        raise TypeError(msg)
    path = Path(str(record["path"]))
    expected_bytes = record["bytes"]
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        msg = f"Frozen input byte length changed after preflight: {path}"
        raise ValueError(msg)
    expected_sha256 = _require_lowercase_sha256(
        record["sha256"],
        label=f"{path} frozen SHA-256",
    )
    content, observed = _read_visible_regular_with_stat(
        path,
        label=f"frozen {label}",
    )
    if len(content) != expected_bytes:
        msg = f"Frozen {label} byte length changed after preflight: {path}"
        raise ValueError(msg)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        msg = f"Frozen {label} hash changed after preflight: {path}"
        raise ValueError(msg)
    if "inode" in record:
        expected_stat = (
            record["device"],
            record["inode"],
            record["mode"],
            record["nlink"],
            record["uid"],
            record["bytes"],
            record["mtime_ns"],
            record["ctime_ns"],
        )
        if expected_stat != _stable_stat_coordinates(observed):
            msg = f"Frozen {label} path identity changed after preflight: {path}"
            raise ValueError(msg)
    return content


def _verify_contract_scientific_records(contract: Mapping[str, Any]) -> None:
    """Perform the final path readback immediately before contract publication."""
    inputs = contract.get("inputs")
    mutsig = inputs.get("mutsig") if isinstance(inputs, dict) else None
    mutsig_files = mutsig.get("files") if isinstance(mutsig, dict) else None
    if (
        not isinstance(inputs, dict)
        or not isinstance(mutsig_files, dict)
        or set(mutsig_files) != {"genes", "lambda", "metadata", "patients", "receipt"}
    ):
        msg = "Cohort contract lacks the complete scientific input record set."
        raise TypeError(msg)
    records = {
        "counts": inputs.get("counts"),
        "sample_axis": inputs.get("sample_axis"),
        "cbase": inputs.get("cbase"),
        "dig": inputs.get("dig"),
        **{f"mutsig_{name}": record for name, record in mutsig_files.items()},
    }
    for name, record in records.items():
        _read_frozen_record_bytes(record, label=f"pre-publication {name}")


def _selected_mutsig_native_rates(
    lambdas: np.ndarray,
    genes: Sequence[str],
    patients: Sequence[str],
    sample_ids: Sequence[str],
    features: Sequence[str],
) -> dict[str, np.ndarray]:
    """Select exact float32 MutSig rates on the frozen feature and sample axes."""
    if lambdas.dtype != np.dtype("<f4"):
        msg = "MutSig lambda tensor is not frozen little-endian float32."
        raise TypeError(msg)
    if not np.isfinite(lambdas).all() or (lambdas < 0).any():
        msg = "MutSig lambda tensor contains an invalid native rate."
        raise ValueError(msg)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    patient_index = {patient: index for index, patient in enumerate(patients)}
    try:
        sample_positions = [patient_index[str(sample)] for sample in sample_ids]
    except KeyError:
        msg = "Frozen MutSig patient axis does not cover the count matrix."
        raise ValueError(msg) from None

    rates_by_feature: dict[str, np.ndarray] = {}
    for feature in features:
        base, effect = feature.rsplit("_", 1)
        if base not in gene_index or effect not in MUTSIG_EFFECT_INDEX:
            msg = "Frozen MutSig gene axis does not cover the feature contract."
            raise ValueError(msg)
        rates = np.asarray(
            lambdas[
                gene_index[base],
                sample_positions,
                MUTSIG_EFFECT_INDEX[effect],
            ],
            dtype="<f4",
        )
        rates.flags.writeable = False
        rates_by_feature[feature] = rates
    return rates_by_feature


def _max_selected_native_lambda(rates_by_feature: Mapping[str, np.ndarray]) -> float:
    """Return the exact binary32 maximum as an exactly representable Python float."""
    if not rates_by_feature:
        msg = "MutSig support requires at least one selected native feature."
        raise ValueError(msg)
    return max(
        float(rates.max(initial=np.float32(0))) for rates in rates_by_feature.values()
    )


def _mutsig_pmfs_from_frozen_bytes(
    inputs: dict[str, Any],
    counts: pd.DataFrame,
    features: list[str],
    pmf_contract: Mapping[str, object],
) -> dict[str, Any]:
    if inputs.get("tensor_encoding") != _mutsig_tensor_encoding_record(
        read_only=True,
    ):
        msg = "Frozen MutSig tensor encoding contract is invalid."
        raise ValueError(msg)
    files = inputs.get("files")
    if not isinstance(files, dict) or set(files) != {
        "genes",
        "lambda",
        "metadata",
        "patients",
        "receipt",
    }:
        msg = "Frozen MutSig input receipt has an invalid file set."
        raise TypeError(msg)
    metadata_raw = _read_frozen_record_bytes(files["metadata"], label="MutSig metadata")
    genes_raw = _read_frozen_record_bytes(files["genes"], label="MutSig gene axis")
    patients_raw = _read_frozen_record_bytes(
        files["patients"],
        label="MutSig patient axis",
    )
    lambda_raw = _read_frozen_record_bytes(files["lambda"], label="MutSig lambda")
    receipt_raw = _read_frozen_record_bytes(files["receipt"], label="MutSig receipt")
    metadata_path = Path(str(files["metadata"]["path"]))
    genes_path = Path(str(files["genes"]["path"]))
    patients_path = Path(str(files["patients"]["path"]))
    lambda_path = Path(str(files["lambda"]["path"]))
    receipt_path = Path(str(files["receipt"]["path"]))
    metadata = _parse_mutsig_metadata(metadata_raw, path=metadata_path)
    genes = _parse_axis(
        genes_raw,
        metadata["ng"],
        path=genes_path,
        label="gene",
    )
    patients = _parse_axis(
        patients_raw,
        metadata["np"],
        path=patients_path,
        label="patient",
    )
    _parse_mutsig_receipt(receipt_raw, path=receipt_path)
    lambdas = _reshape_mutsig_lambda_bytes(
        lambda_raw,
        metadata,
        path=lambda_path,
    )
    rates_by_feature = _selected_mutsig_native_rates(
        lambdas,
        genes,
        patients,
        [str(sample) for sample in counts.index],
        features,
    )
    observed_kmax = int(counts.loc[:, features].to_numpy().max(initial=0))
    validate_poisson_support_contract(
        pmf_contract,
        max_native_lambda=_max_selected_native_lambda(rates_by_feature),
        observed_kmax=observed_kmax,
    )
    return build_native_poisson_pmfs(rates_by_feature, pmf_contract)


def _load_frozen_scientific_inputs(
    contract: dict[str, Any],
    bmr: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = contract.get("inputs")
    features = contract.get("features")
    if (
        not isinstance(inputs, dict)
        or not isinstance(features, list)
        or bmr not in BMRS
    ):
        msg = "Frozen scientific input contract is invalid."
        raise TypeError(msg)
    count_raw = _read_frozen_record_bytes(inputs.get("counts"), label="count matrix")
    counts = _parse_counts_csv(count_raw, label="frozen count matrix")
    if bmr in {"cbase", "dig"}:
        pmf_raw = _read_frozen_record_bytes(inputs.get(bmr), label=f"{bmr} PMFs")
        all_pmfs = _parse_strict_pmfs_csv(pmf_raw, label=f"frozen {bmr} PMFs")
        try:
            pmfs = {feature: all_pmfs[feature] for feature in features}
        except KeyError:
            msg = "Frozen PMFs do not cover the feature contract."
            raise ValueError(msg) from None
    else:
        mutsig_inputs = inputs.get("mutsig")
        if not isinstance(mutsig_inputs, dict):
            msg = "Frozen MutSig input contract is invalid."
            raise TypeError(msg)
        pmf_contract = contract.get("mutsig_pmf_contract")
        if not isinstance(pmf_contract, dict):
            msg = "Frozen MutSig input contract lacks its PMF support contract."
            raise TypeError(msg)
        pmfs = _mutsig_pmfs_from_frozen_bytes(
            mutsig_inputs,
            counts,
            features,
            pmf_contract,
        )
    return counts, pmfs


def _consumed_input_hashes(contract: dict[str, Any], bmr: str) -> dict[str, str]:
    inputs = contract["inputs"]
    consumed = {
        "counts": _require_lowercase_sha256(
            inputs["counts"]["sha256"],
            label="consumed count-matrix SHA-256",
        ),
    }
    if bmr in {"cbase", "dig"}:
        consumed[bmr] = _require_lowercase_sha256(
            inputs[bmr]["sha256"],
            label=f"consumed {bmr} PMF SHA-256",
        )
    else:
        consumed.update(
            {
                f"mutsig_{name}": _require_lowercase_sha256(
                    record["sha256"],
                    label=f"consumed MutSig {name} SHA-256",
                )
                for name, record in inputs["mutsig"]["files"].items()
            },
        )
    return consumed


def _load_verified_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int,
) -> dict[str, Any]:
    path = _contract_path(paths, cohort)
    if not path.exists():
        if top_k == TOP_K:
            msg = f"Production child lacks its parent-frozen cohort contract: {path}"
            raise FileNotFoundError(msg)
        return _ensure_contract(paths, cohort, top_k=top_k)
    contract = _read_json(path)
    if contract.get("top_k") != top_k or contract.get("cohort") != cohort:
        msg = f"Frozen contract coordinates do not match task: {path}"
        raise ValueError(msg)
    _require_tested_family_contract(
        contract,
        require_signed_k500=top_k == TOP_K,
    )
    _require_exact_sample_axis(contract)
    _require_full_observation_support(contract)
    # Rehash only this cohort's authority before opening any frozen input path.
    if top_k == TOP_K:
        _verify_frozen_cohort_authority(paths, contract)
    inputs = contract["inputs"]
    _verify_file_record(inputs["counts"])
    _verify_file_record(inputs["sample_axis"])
    _verify_file_record(inputs["cbase"])
    _verify_file_record(inputs["dig"])
    for record in inputs["mutsig"]["files"].values():
        _verify_file_record(record)
    return contract


def _source_snapshot(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): _sha256(repo_root / path) for path in SOURCE_FILES}


def _d3_source_contract_bytes(payload: object, *, newline: bool) -> bytes:
    """Match the fit-authority builder's UTF-8 canonicalization exactly."""
    content = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return content + (b"\n" if newline else b"")


def _require_d3_runtime_contract(
    policy: RevisionFitPolicy,
    paths: RunPaths,
) -> dict[str, Any]:
    """Bind signed D3-v2 support and source receipts to this exact clean runner."""
    support = policy.d3.mutsig_support
    binding = policy.d3.implementation_binding
    if support is None or binding is None:
        msg = "Production K=500 fitting requires the signed D3-v2 MutSig contract."
        raise ValueError(msg)
    tensor = _mutsig_tensor_encoding_record(read_only=True)
    if (
        tensor.get("dtype") != "<f4"
        or tensor.get("effect_pages") != MUTSIG_EFFECT_INDEX
        or tensor.get("order") != "Fortran-(gene,patient,effect)"
        or tensor.get("read_only") is not True
        or tensor.get("layout_canary") != MUTSIG_TENSOR_LAYOUT_CANARY
    ):
        msg = "Executed MutSig tensor semantics differ from signed native D3 support."
        raise RuntimeError(msg)
    expected_support = {
        "dtype": tensor["dtype"],
        "effect_pages": tensor["effect_pages"],
        "fallback_or_floor": "none",
        "lambda_dtype": "native-binary32",
        "layout_canary": tensor["layout_canary"],
        "normalization": PRODUCTION_POISSON_NORMALIZATION,
        "order": tensor["order"],
        "predecessor_proof": "required-when-tail-endpoint-binds",
        "read_only": tensor["read_only"],
        "storage_contract": PRODUCTION_POISSON_STORAGE_CONTRACT,
        "support_contract": PRODUCTION_POISSON_SUPPORT_CONTRACT,
        "support_rule": PRODUCTION_POISSON_SUPPORT_RULE,
        "tail_tolerance": PRODUCTION_POISSON_TAIL_TOLERANCE,
    }
    if asdict(support) != expected_support:
        msg = "Signed D3 MutSig support differs from the executed native-tail contract."
        raise ValueError(msg)

    source_files = _source_snapshot(RUNNER_REPO_ROOT)
    runner_path = Path(__file__).resolve().relative_to(RUNNER_REPO_ROOT).as_posix()
    source_snapshot_sha256 = hashlib.sha256(
        _d3_source_contract_bytes(source_files, newline=False),
    ).hexdigest()
    provider = _validated_provider_bundle(paths)
    provider_receipt = _provider_root_receipt(paths, provider)
    git = _git_snapshot(RUNNER_REPO_ROOT, provider_receipt["git_executable"])
    if git["dirty"]:
        msg = "Signed D3 source authority requires the exact clean reviewed Git tree."
        raise ValueError(msg)
    expected_binding = {
        "reviewed_scientific_commit": git["head"],
        "runner_path": runner_path,
        "runner_sha256": source_files[runner_path],
        "source_file_count": len(source_files),
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    observed_binding = asdict(binding)
    for key, expected in expected_binding.items():
        if observed_binding.get(key) != expected:
            msg = f"Signed D3 implementation binding drifted for {key}."
            raise ValueError(msg)

    source_contract = {
        "association_results_read": False,
        "d4_canonical_artifact_sha256": policy.receipts["D4"].canonical_artifact_sha256,
        "d5_canonical_artifact_sha256": policy.receipts["D5"].canonical_artifact_sha256,
        "mutsig_minimal_tail_contract": expected_support,
        "reviewed_scientific_commit": git["head"],
        "runner": {
            "path": runner_path,
            "sha256": source_files[runner_path],
        },
        "schema": "dialect-revision-k500-scientific-source-v1",
        "source_file_count": len(source_files),
        "source_files": source_files,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    source_contract_sha256 = hashlib.sha256(
        _d3_source_contract_bytes(source_contract, newline=True),
    ).hexdigest()
    if binding.source_contract_sha256 != source_contract_sha256:
        msg = "Signed D3 source-contract digest differs from the live source closure."
        raise ValueError(msg)
    return {
        "implementation_binding": observed_binding,
        "mutsig_support": expected_support,
        "source_contract_sha256": source_contract_sha256,
        "tensor_encoding": tensor,
    }


def _require_pinned_import_roots(implementation: object) -> None:
    """Bind every imported local execution module to its exact frozen source."""
    if not isinstance(implementation, dict):
        msg = "Frozen implementation receipt is unavailable for import validation."
        raise TypeError(msg)
    for directory, label in (
        (RUNNER_REPO_ROOT, "runner repository root"),
        (RUNNER_REPO_ROOT / "analysis", "runner analysis package"),
        (RUNNER_SOURCE_ROOT, "runner source root"),
        (RUNNER_SOURCE_ROOT / "dialect", "runner DIALECT package"),
    ):
        descriptor = _open_secure_directory(directory, label=label)
        os.close(descriptor)

    expected_source_keys = {path.as_posix() for path in SOURCE_FILES}
    if set(implementation) != expected_source_keys:
        msg = "Frozen implementation receipt has incomplete or extra source coverage."
        raise RuntimeError(msg)

    for module_name, relative in EXECUTED_LOCAL_PYTHON_MODULES:
        module = sys.modules.get(module_name)
        if module is None and module_name == "analysis.run_tcga_revision_k500":
            candidate = sys.modules.get(__name__)
            specification = getattr(candidate, "__spec__", None)
            if getattr(specification, "name", None) == module_name:
                module = candidate
        if module is None:
            msg = f"Required local execution module was not imported: {module_name}."
            raise RuntimeError(msg)
        raw_path = getattr(module, "__file__", None)
        expected_path = RUNNER_REPO_ROOT / relative
        if not isinstance(raw_path, str) or Path(raw_path).absolute() != expected_path:
            msg = f"{module_name} was imported outside the pinned repository tree."
            raise RuntimeError(msg)
        expected_sha256 = implementation.get(relative.as_posix())
        if (
            not isinstance(expected_sha256, str)
            or _sha256(expected_path) != expected_sha256
        ):
            msg = f"{module_name} differs from the frozen implementation receipt."
            raise RuntimeError(msg)

    loaded_roots = {
        "analysis": RUNNER_REPO_ROOT / "analysis",
        "dialect": RUNNER_SOURCE_ROOT / "dialect",
    }
    for module_name, module in tuple(sys.modules.items()):
        package = module_name.partition(".")[0]
        if package not in loaded_roots:
            continue
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        path = Path(str(raw_path)).absolute()
        root = loaded_roots[package]
        if path == root or not path.is_relative_to(root):
            msg = f"Loaded {package} module escaped its pinned source root."
            raise RuntimeError(msg)


def _git_snapshot(repo_root: Path, git_record: object) -> dict[str, Any]:
    """Read repository state with the exact provider-authorized Git binary."""
    if not isinstance(git_record, dict) or set(git_record) != {
        "bytes",
        "path",
        "sha256",
    }:
        msg = "Git source-state checks require an exact provider runtime receipt."
        raise TypeError(msg)
    _read_frozen_record_bytes(git_record, label="provider-authorized Git")
    git_path = Path(git_record["path"])
    git_environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SAFE_CHILD_PATH,
    }
    version = subprocess.run(
        [git_path.as_posix(), "--version"],
        env=git_environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        [git_path.as_posix(), "--no-pager", "rev-parse", "HEAD"],
        cwd=repo_root,
        env=git_environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            git_path.as_posix(),
            "--no-pager",
            "status",
            "--short",
            "--untracked-files=all",
        ],
        cwd=repo_root,
        env=git_environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    _read_frozen_record_bytes(git_record, label="provider-authorized Git")
    return {
        "head": head,
        "dirty": bool(status),
        "status": status,
        "executable": dict(git_record),
        "version": version,
    }


def _verify_recorded_source_hashes(paths: RunPaths) -> dict[str, str]:
    manifest_path = paths.output_root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    recorded = manifest.get("implementation_sha256")
    if not isinstance(recorded, dict):
        msg = "Run manifest does not contain implementation source hashes."
        raise TypeError(msg)
    repo_root = Path(__file__).resolve().parents[1]
    current = _source_snapshot(repo_root)
    if recorded != current:
        msg = "Inference implementation changed after the frozen run was initialized."
        raise ValueError(msg)
    return current


def _verify_run_implementation(paths: RunPaths) -> dict[str, str]:
    current = _verify_recorded_source_hashes(paths)
    manifest = _read_json(paths.output_root / "run_manifest.json")
    repo_root = Path(__file__).resolve().parents[1]
    authority = manifest.get("revision_authority")
    provider = authority.get("provider_input") if isinstance(authority, dict) else None
    git_record = provider.get("git_executable") if isinstance(provider, dict) else None
    repo_state = _git_snapshot(repo_root, git_record)
    recorded_git = manifest.get("git", {})
    if (
        not isinstance(recorded_git, dict)
        or recorded_git.get("dirty") is not False
        or repo_state["dirty"]
        or repo_state["head"] != recorded_git.get("head")
        or repo_state["executable"] != recorded_git.get("executable")
        or repo_state["version"] != recorded_git.get("version")
    ):
        msg = "Production inference requires the clean Git HEAD pinned by the run."
        raise ValueError(msg)
    return current


def _initialize_run(paths: RunPaths, *, allow_dirty: bool) -> dict[str, Any]:
    _frozen_python_executable()
    repo_root = Path(__file__).resolve().parents[1]
    revision_authority = _revision_authority_manifest_record(paths)
    signed_tested_family = _signed_tested_family_record(paths)
    resource_policy = {
        "default_jobs": safe_default_jobs(),
        "default_mutsig_jobs": min(2, safe_default_jobs()),
        "maximum_general_jobs": MAX_GENERAL_JOBS,
        "serial_canary_cohort": CANARY_COHORT,
        "memory_heavy_mutsig_cohorts": sorted(MEMORY_HEAVY_MUTSIG_COHORTS),
        "memory_heavy_mutsig_jobs": 1,
        "required_child_process_nice_increment": REQUIRED_NICE_INCREMENT,
        "sealed_child_environment": SEALED_TASK_ENVIRONMENT,
        "prior_task_peak_rss_bytes": PRIOR_TASK_PEAK_RSS_BYTES,
        "memory_headroom_factor": MEMORY_HEADROOM_FACTOR,
        "minimum_available_memory_fraction": MIN_AVAILABLE_MEMORY_FRACTION,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "live_readback_before_every_wave": True,
        "live_readback_before_every_task": True,
        "aggregate_cpu_rule": (
            "finite nonnegative loadavg_1m + planned jobs < logical_cores / 2"
        ),
    }
    expected = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "tcga-revision-k500",
        "top_k": TOP_K,
        "cohorts": list(TCGA_COHORTS),
        "bmrs": list(BMRS),
        "source_root": paths.source_root.as_posix(),
        "mutsig_root": paths.mutsig_root.as_posix(),
        "feature_policy": TESTED_FAMILY_FEATURE_RANKING,
        "same_base_pair_policy": TESTED_FAMILY_SAME_BASE_POLICY,
        "signed_tested_family": signed_tested_family,
        "tested_family_implementation": asdict(REQUIRED_TESTED_FAMILY),
        "required_lrt_contract": REQUIRED_LRT_CONTRACT,
        "required_pair_fit_contract": REQUIRED_PAIR_FIT_CONTRACT,
        "required_pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
        "required_pair_fit_max_iterations": REQUIRED_PAIR_FIT_MAX_ITER,
        "required_pair_simplex_tolerance": REQUIRED_PAIR_SIMPLEX_TOL,
        "required_lrt_nestedness_tolerance": REQUIRED_LRT_NESTEDNESS_TOL,
        "required_output_recomputation_atol": REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
        "required_pair_identifiability_relative_tolerance": (
            REQUIRED_PAIR_IDENTIFIABILITY_RTOL
        ),
        "required_pair_effect_identifiability_contract": (
            REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT
        ),
        "required_rho_contract": REQUIRED_RHO_CONTRACT,
        "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
        "required_contingency_table_contract": REQUIRED_CONTINGENCY_TABLE_CONTRACT,
        "required_log_odds_ratio_contract": REQUIRED_LOG_ODDS_RATIO_CONTRACT,
        "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
        "required_gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
        "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
        "revision_authority": revision_authority,
        "resource_policy": resource_policy,
    }
    manifest_path = paths.output_root / "run_manifest.json"
    provider_authority = revision_authority.get("provider_input")
    git_record = (
        provider_authority.get("git_executable")
        if isinstance(provider_authority, dict)
        else None
    )
    git = _git_snapshot(repo_root, git_record)
    if not allow_dirty and git["dirty"]:
        msg = (
            "Production K=500 inference requires a clean Git tree; commit or "
            "otherwise resolve every tracked/untracked change before launch."
        )
        raise RuntimeError(msg)
    if paths.output_root.exists():
        if not manifest_path.is_file():
            msg = (
                f"Output root already exists without this runner's manifest; refusing "
                f"to reuse or overwrite it: {paths.output_root}"
            )
            raise FileExistsError(msg)
        manifest = _read_json(manifest_path)
        for key, value in expected.items():
            if manifest.get(key) != value:
                msg = f"Run manifest mismatch for {key!r}; use a fresh output root."
                raise ValueError(msg)
        if manifest.get("implementation_sha256") != _source_snapshot(repo_root):
            msg = "Run implementation hashes changed; use a fresh output root."
            raise ValueError(msg)
        recorded_git = manifest.get("git", {})
        if not allow_dirty and (
            not isinstance(recorded_git, dict)
            or recorded_git.get("dirty") is not False
            or recorded_git.get("head") != git["head"]
        ):
            msg = "Run was not initialized from the current clean Git HEAD."
            raise RuntimeError(msg)
        _require_pinned_import_roots(manifest.get("implementation_sha256"))
        return manifest

    paths.output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        **expected,
        "created_at_utc": _utc_now(),
        "git": git,
        "implementation_sha256": _source_snapshot(repo_root),
    }
    _require_pinned_import_roots(manifest["implementation_sha256"])
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _require_corrected_lrt() -> tuple[str, str, str, str, str, str, str]:
    actual = getattr(interaction_module, "LRT_CONTRACT", None)
    if actual != REQUIRED_LRT_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.LRT_CONTRACT must be "
            f"{REQUIRED_LRT_CONTRACT!r}, found {actual!r}. Implement and test the "
            "constrained independence-null MLE before running this grid."
        )
        raise RuntimeError(msg)
    pair_fit_actual = getattr(interaction_module, "PAIR_FIT_CONTRACT", None)
    if pair_fit_actual != REQUIRED_PAIR_FIT_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.PAIR_FIT_CONTRACT "
            f"must be {REQUIRED_PAIR_FIT_CONTRACT!r}, found {pair_fit_actual!r}."
        )
        raise RuntimeError(msg)
    pair_fit_kkt_actual = getattr(interaction_module, "PAIR_FIT_KKT_TOL", None)
    if pair_fit_kkt_actual != REQUIRED_PAIR_FIT_KKT_TOL:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.PAIR_FIT_KKT_TOL "
            f"must be {REQUIRED_PAIR_FIT_KKT_TOL!r}, found "
            f"{pair_fit_kkt_actual!r}."
        )
        raise RuntimeError(msg)
    pair_fit_max_iter_actual = getattr(
        interaction_module,
        "PAIR_FIT_MAX_ITER",
        None,
    )
    if pair_fit_max_iter_actual != REQUIRED_PAIR_FIT_MAX_ITER:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.PAIR_FIT_MAX_ITER "
            f"must be {REQUIRED_PAIR_FIT_MAX_ITER!r}, found "
            f"{pair_fit_max_iter_actual!r}."
        )
        raise RuntimeError(msg)
    pair_simplex_actual = getattr(interaction_module, "PAIR_SIMPLEX_TOL", None)
    if pair_simplex_actual != REQUIRED_PAIR_SIMPLEX_TOL:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.PAIR_SIMPLEX_TOL "
            f"must be {REQUIRED_PAIR_SIMPLEX_TOL!r}, found "
            f"{pair_simplex_actual!r}."
        )
        raise RuntimeError(msg)
    lrt_nestedness_actual = getattr(
        interaction_module,
        "LRT_NESTEDNESS_TOL",
        None,
    )
    if lrt_nestedness_actual != REQUIRED_LRT_NESTEDNESS_TOL:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.LRT_NESTEDNESS_TOL must be "
            f"{REQUIRED_LRT_NESTEDNESS_TOL!r}, found "
            f"{lrt_nestedness_actual!r}."
        )
        raise RuntimeError(msg)
    identifiability_rtol_actual = getattr(
        interaction_module,
        "PAIR_IDENTIFIABILITY_RTOL",
        None,
    )
    if identifiability_rtol_actual != REQUIRED_PAIR_IDENTIFIABILITY_RTOL:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.PAIR_IDENTIFIABILITY_RTOL must be "
            f"{REQUIRED_PAIR_IDENTIFIABILITY_RTOL!r}, found "
            f"{identifiability_rtol_actual!r}."
        )
        raise RuntimeError(msg)
    effect_identifiability_actual = getattr(
        interaction_module,
        "PAIR_EFFECT_IDENTIFIABILITY_CONTRACT",
        None,
    )
    if effect_identifiability_actual != REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction."
            "PAIR_EFFECT_IDENTIFIABILITY_CONTRACT must be "
            f"{REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT!r}, found "
            f"{effect_identifiability_actual!r}."
        )
        raise RuntimeError(msg)
    effect_statuses_actual = (
        getattr(interaction_module, "PAIR_EFFECT_IDENTIFIED_STATUS", None),
        getattr(interaction_module, "PAIR_EFFECT_RANK_DEFICIENT_STATUS", None),
        getattr(interaction_module, "PAIR_EFFECT_UNDERFLOW_STATUS", None),
    )
    effect_statuses_required = (
        REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
        REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS,
        REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS,
    )
    if effect_statuses_actual != effect_statuses_required:
        msg = "K=500 launch blocked: pair effect-identifiability statuses drifted."
        raise RuntimeError(msg)
    rho_actual = getattr(interaction_module, "RHO_CONTRACT", None)
    if rho_actual != REQUIRED_RHO_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.RHO_CONTRACT must "
            f"be {REQUIRED_RHO_CONTRACT!r}, found {rho_actual!r}."
        )
        raise RuntimeError(msg)
    undefined_rho_lrt_tol_actual = getattr(
        interaction_module,
        "UNDEFINED_RHO_LRT_TOL",
        None,
    )
    if undefined_rho_lrt_tol_actual != REQUIRED_UNDEFINED_RHO_LRT_TOL:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.UNDEFINED_RHO_LRT_TOL must be "
            f"{REQUIRED_UNDEFINED_RHO_LRT_TOL!r}, found "
            f"{undefined_rho_lrt_tol_actual!r}."
        )
        raise RuntimeError(msg)
    contingency_actual = getattr(
        interaction_module,
        "CONTINGENCY_TABLE_CONTRACT",
        None,
    )
    if contingency_actual != REQUIRED_CONTINGENCY_TABLE_CONTRACT:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.CONTINGENCY_TABLE_CONTRACT must be "
            f"{REQUIRED_CONTINGENCY_TABLE_CONTRACT!r}, found "
            f"{contingency_actual!r}."
        )
        raise RuntimeError(msg)
    log_odds_actual = getattr(
        interaction_module,
        "LOG_ODDS_RATIO_CONTRACT",
        None,
    )
    if log_odds_actual != REQUIRED_LOG_ODDS_RATIO_CONTRACT:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.LOG_ODDS_RATIO_CONTRACT must be "
            f"{REQUIRED_LOG_ODDS_RATIO_CONTRACT!r}, found {log_odds_actual!r}."
        )
        raise RuntimeError(msg)
    gene_support_actual = getattr(gene_module, "OBSERVATION_SUPPORT_CONTRACT", None)
    if gene_support_actual != REQUIRED_GENE_SUPPORT_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.gene.OBSERVATION_SUPPORT_CONTRACT "
            f"must be {REQUIRED_GENE_SUPPORT_CONTRACT!r}, found "
            f"{gene_support_actual!r}."
        )
        raise RuntimeError(msg)
    return (
        str(actual),
        str(pair_fit_actual),
        str(effect_identifiability_actual),
        str(rho_actual),
        str(gene_support_actual),
        str(contingency_actual),
        str(log_odds_actual),
    )


def _task_pmfs(  # noqa: PLR0913
    paths: RunPaths,
    task: Task,
    counts: pd.DataFrame,
    features: list[str],
    *,
    contract: dict[str, Any] | None = None,
    mutsig_pmf_contract: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if contract is not None:
        frozen_counts, pmfs = _load_frozen_scientific_inputs(contract, task.bmr)
        if (
            list(frozen_counts.index) != list(counts.index)
            or list(features) != list(contract.get("features", []))
            or not frozen_counts.loc[:, features].equals(counts.loc[:, features])
        ):
            msg = "Frozen task inputs differ from the requested scientific axes."
            raise ValueError(msg)
        return pmfs
    if _revision_authority_is_configured(paths):
        provider_binding = _provider_cohort_binding(paths, task.cohort)
        cohort_dir = provider_binding["cohort_root"]
        cbase_path = cohort_dir / "bmr_pmfs.csv"
        dig_path = cohort_dir / "bmr_pmfs.dig.csv"
    else:
        cohort_dir = paths.source_root / task.cohort
        cbase_path = cohort_dir / "bmr_pmfs.csv"
        dig_path = cohort_dir / "bmr_pmfs.dig.csv"
    if task.bmr in {"cbase", "dig"}:
        pmf_path = cbase_path if task.bmr == "cbase" else dig_path
        all_pmfs = _load_strict_pmfs(pmf_path)
        missing = [feature for feature in features if feature not in all_pmfs]
        if missing:
            msg = f"{task.bmr} lost native feature support: {missing[:5]}"
            raise ValueError(msg)
        return {feature: all_pmfs[feature] for feature in features}

    if mutsig_pmf_contract is not None:
        msg = (
            "Path-based MutSig PMF construction is disabled; use the frozen "
            "cohort input contract."
        )
        raise RuntimeError(msg)
    msg = "Production MutSig PMFs require a frozen cohort input contract."
    raise TypeError(msg)


def _build_genes(
    counts: pd.DataFrame,
    features: Sequence[str],
    pmfs: dict[str, Any],
) -> dict[str, Gene]:
    genes = {
        feature: Gene(
            name=feature,
            samples=counts.index,
            counts=counts[feature].to_numpy(),
            bmr_pmf=pmfs[feature],
        )
        for feature in features
    }
    if list(genes) != list(features):
        msg = "Gene construction did not preserve the frozen feature order."
        raise ValueError(msg)
    return genes


def _pairwise_record(interaction: Interaction) -> dict[str, Any]:
    taus = (
        interaction.tau_00,
        interaction.tau_01,
        interaction.tau_10,
        interaction.tau_11,
    )
    contingency = interaction.compute_contingency_table()
    likelihood_ratio = interaction.likelihood_ratio
    if likelihood_ratio is None:
        likelihood_ratio = interaction.compute_likelihood_ratio(taus)
    effect_identifiability = interaction.effect_identifiability_status()
    effect_identifiable = (
        effect_identifiability == REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS
    )
    return {
        "Gene A": interaction.gene_a.name,
        "Gene B": interaction.gene_b.name,
        "Tau_00": interaction.tau_00,
        "Tau_10": interaction.tau_10,
        "Tau_01": interaction.tau_01,
        "Tau_11": interaction.tau_11,
        "_00_": contingency[0, 0],
        "_10_": contingency[1, 0],
        "_01_": contingency[0, 1],
        "_11_": contingency[1, 1],
        "Tau_1X": (
            interaction.tau_10 + interaction.tau_11 if effect_identifiable else None
        ),
        "Tau_X1": (
            interaction.tau_01 + interaction.tau_11 if effect_identifiable else None
        ),
        "Effect Identifiability": effect_identifiability,
        "Rho": interaction.compute_rho_for_direction(taus, likelihood_ratio),
        "Log Odds Ratio": (
            interaction.compute_log_odds_ratio(taus) if effect_identifiable else None
        ),
        "Likelihood Ratio": likelihood_ratio,
        "Wald Statistic": (
            interaction.compute_wald_statistic(taus) if effect_identifiable else None
        ),
        "Fit Algorithm": interaction.fit_algorithm,
        "Fit Converged": interaction.fit_converged,
        "Fit Iterations": interaction.fit_iterations,
        "Fit Last LL Gain": interaction.fit_last_log_likelihood_gain,
        "Fit Fixed-Point Residual": interaction.fit_fixed_point_residual,
        "Fit KKT Residual": interaction.fit_kkt_residual,
        "Pair Fit Contract": REQUIRED_PAIR_FIT_CONTRACT,
        "Null Log Likelihood": interaction.null_log_likelihood,
        "Alternative Log Likelihood": interaction.alternative_log_likelihood,
        "LRT Contract": REQUIRED_LRT_CONTRACT,
    }


def _single_gene_results_bytes(genes: Sequence[Gene]) -> bytes:
    """Render the established single-gene CSV schema entirely in memory."""
    buffer = io.StringIO(newline="")
    create_single_gene_results(
        list(genes),
        buffer,  # type: ignore[arg-type]
        cbase_phi_vals_present=False,
    )
    return buffer.getvalue().encode("utf-8")


def _write_pairwise_results_to_handle(
    handle: TextIO,
    genes: dict[str, Gene],
    features: Sequence[str],
) -> int:
    """Fit and stream the pair table through an already pinned text handle."""
    rows = 0
    writer = csv.DictWriter(handle, fieldnames=PAIRWISE_COLUMNS)
    writer.writeheader()
    for feature_a, feature_b in iter_tested_pairs(features):
        interaction = Interaction(genes[feature_a], genes[feature_b])
        estimate_taus_for_each_interaction([interaction])
        writer.writerow(_pairwise_record(interaction))
        rows += 1
    return rows


def _write_pairwise_results_at(
    directory_fd: int,
    name: str,
    genes: dict[str, Gene],
    features: Sequence[str],
) -> int:
    """Stream a pair table to a private temp file and publish it by dirfd."""
    _require_safe_basename(name, label="pairwise output")
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            STAGING_FILE_MODE,
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = "Staged pairwise output must be a single-link regular file."
            raise ValueError(msg)
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
            closefd=False,
        ) as handle:
            rows = _write_pairwise_results_to_handle(handle, genes, features)
            handle.flush()
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or not stat_module.S_ISREG(after.st_mode)
            or after.st_nlink != 1
        ):
            msg = "Staged pairwise output identity changed before publication."
            raise ValueError(msg)
        _require_regular_entry_identity(
            directory_fd,
            temporary_name,
            descriptor,
            label="staged pairwise output",
        )
        os.close(descriptor)
        descriptor = None
        _rename_exclusive_at(directory_fd, temporary_name, directory_fd, name)
        os.fsync(directory_fd)
        published = True
        return rows
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)


def _write_pairwise_results(
    path: Path,
    genes: dict[str, Gene],
    features: Sequence[str],
) -> int:
    """Compatibility wrapper over the descriptor-relative pairwise writer."""
    parent_fd = _open_secure_directory(path.parent, label="pairwise output parent")
    try:
        rows = _write_pairwise_results_at(parent_fd, path.name, genes, features)
        _require_directory_path_identity(
            path.parent,
            parent_fd,
            label="pairwise output parent",
        )
        return rows
    finally:
        os.close(parent_fd)


def _read_task_output_snapshot(
    task_dir: Path,
    *,
    require_manifest: bool,
    directory_fd: int | None = None,
) -> tuple[bytes, bytes, bytes | None]:
    """Read a closed task inventory from stable no-follow descriptors."""
    expected_names = {
        "pairwise_interaction_results.csv",
        "single_gene_results.csv",
    }
    if require_manifest:
        expected_names.add("task_manifest.json")
    owned_descriptor = directory_fd is None
    if directory_fd is None:
        directory_fd = _open_secure_directory(task_dir, label="task output")
    elif not stat_module.S_ISDIR(os.fstat(directory_fd).st_mode):
        raise _sealed_error("output-directory-invalid", "validate-output")  # noqa: EM101
    try:
        if set(os.listdir(directory_fd)) != expected_names:  # noqa: PTH208
            raise _sealed_error(  # noqa: TRY301
                "output-inventory-invalid",  # noqa: EM101
                "validate-output",
            )
        single_bytes = _read_regular_at(
            directory_fd,
            "single_gene_results.csv",
            label="single-gene output",
        )
        pairwise_bytes = _read_regular_at(
            directory_fd,
            "pairwise_interaction_results.csv",
            label="pairwise output",
        )
        manifest_bytes = (
            _read_regular_at(
                directory_fd,
                "task_manifest.json",
                label="task manifest",
            )
            if require_manifest
            else None
        )
    except SealedFitError:
        raise
    except Exception:  # noqa: BLE001
        raise _sealed_error(
            "output-secure-read-failed",  # noqa: EM101
            "validate-output",
        ) from None
    finally:
        if owned_descriptor:
            os.close(directory_fd)
    return single_bytes, pairwise_bytes, manifest_bytes


def _numeric_equal(actual: float, expected: float) -> bool:
    if np.isnan(actual) or np.isnan(expected):
        return False
    if np.isinf(actual) or np.isinf(expected):
        return actual == expected
    return bool(
        np.isclose(
            actual,
            expected,
            rtol=0,
            atol=REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
        ),
    )


def _pair_simplex_is_valid(taus: Sequence[float]) -> bool:
    """Return whether four finite probabilities satisfy the frozen simplex bound."""
    values = np.asarray(taus, dtype=float)
    return bool(
        values.shape == (4,)
        and np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(
            values.sum(),
            1.0,
            rtol=0,
            atol=REQUIRED_PAIR_SIMPLEX_TOL,
        ),
    )


def _validate_single_gene_mle_receipt(row: pd.Series, gene: Gene) -> None:
    """Replay and validate every D4-v3 marginal-fit receipt field."""
    gene.validate_mle_fit()
    required = REQUIRED_D4_IMPLEMENTATION.numerical_implementation
    if (
        required.marginal_fit_contract != MARGINAL_FIT_CONTRACT
        or required.marginal_fit_max_iterations != MARGINAL_FIT_MAX_ITER
        or required.marginal_fit_total_kkt_tolerance != MARGINAL_FIT_KKT_TOL
        or required.marginal_fit_bracket_width_tolerance
        != MARGINAL_FIT_BRACKET_WIDTH_TOL
        or required.marginal_fit_fixed_point_tolerance != MARGINAL_FIT_FIXED_POINT_TOL
        or required.marginal_fit_flat_likelihood_tie_break
        != MARGINAL_FIT_FLAT_TIE_BREAK
    ):
        msg = "Runner marginal-fit constants drifted from its D4-v3 contract."
        raise RuntimeError(msg)
    try:
        actual_iterations = float(row["MLE Iterations"])
        actual_bracket_width = float(row["MLE Bracket Width"])
        actual_fixed_point = float(row["MLE Fixed-Point Residual"])
        actual_kkt = float(row["MLE KKT Residual"])
    except (TypeError, ValueError) as error:
        msg = f"Gene {gene.name} has non-numeric marginal-fit receipts."
        raise ValueError(msg) from error
    expected_diagnostics = (
        (actual_bracket_width, gene.mle_bracket_width),
        (actual_fixed_point, gene.mle_fixed_point_residual),
        (actual_kkt, gene.mle_kkt_residual),
    )
    if (
        row["MLE Algorithm"] != required.marginal_fit_contract
        or not np.isfinite(actual_iterations)
        or not actual_iterations.is_integer()
        or not 0 <= actual_iterations <= required.marginal_fit_max_iterations
        or int(actual_iterations) != gene.mle_iterations
        or not np.isfinite(actual_bracket_width)
        or not 0
        <= actual_bracket_width
        <= required.marginal_fit_bracket_width_tolerance
        or not np.isfinite(actual_fixed_point)
        or not 0 <= actual_fixed_point <= required.marginal_fit_fixed_point_tolerance
        or not np.isfinite(actual_kkt)
        or not 0 <= actual_kkt <= required.marginal_fit_total_kkt_tolerance
        or any(
            expected is None or actual != float(expected)
            for actual, expected in expected_diagnostics
        )
    ):
        msg = f"Gene {gene.name} has invalid marginal-fit receipts."
        raise ValueError(msg)


def _validate_single_gene_output(
    raw: bytes,
    contract: dict[str, Any],
    counts: pd.DataFrame,
    pmfs: dict[str, Any],
    genes: dict[str, Gene] | None = None,
) -> int:
    """Recompute every single-gene scientific column from frozen inputs."""
    try:
        single = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except Exception:  # noqa: BLE001
        raise _sealed_error("single-csv-invalid", "validate-single") from None  # noqa: EM101
    if tuple(single.columns) != SINGLE_GENE_RESULT_COLUMNS:
        raise _sealed_error("single-schema-invalid", "validate-single")  # noqa: EM101
    if len(single) != len(contract["features"]):
        raise _sealed_error(
            "single-row-count-invalid",  # noqa: EM101
            "validate-single",
        )
    for row_index, feature in enumerate(contract["features"]):
        try:
            row = single.iloc[row_index]
            if row["Gene Name"] != feature:
                raise ValueError  # noqa: TRY301
            gene = (
                genes[feature]
                if genes is not None
                else Gene(
                    name=feature,
                    samples=counts.index,
                    counts=counts[feature].to_numpy(),
                    bmr_pmf=pmfs[feature],
                )
            )
            if not gene.mle_converged:
                gene.estimate_pi_with_mle()
            _validate_single_gene_mle_receipt(row, gene)
            expected_pi = float(gene.pi)
            expected_lor = gene.compute_log_odds_ratio(expected_pi)
            expected_lrt = gene.compute_likelihood_ratio(expected_pi)
            expected_observed = int(counts[feature].sum())
            expected_passenger = gene.calculate_expected_total_mutations()
            expected_difference = expected_observed - expected_passenger
            actual_values = {
                "Pi": float(row["Pi"]),
                "Log Odds Ratio": float(row["Log Odds Ratio"]),
                "Likelihood Ratio": float(row["Likelihood Ratio"]),
                "Expected Mutations": float(row["Expected Mutations"]),
                "Obs. - Exp. Mutations": float(row["Obs. - Exp. Mutations"]),
                "MLE Log Likelihood": float(row["MLE Log Likelihood"]),
            }
            actual_observed = float(row["Observed Mutations"])
            expected_values = {
                "Pi": expected_pi,
                "Log Odds Ratio": expected_lor,
                "Likelihood Ratio": expected_lrt,
                "Expected Mutations": expected_passenger,
                "Obs. - Exp. Mutations": expected_difference,
                "MLE Log Likelihood": float(gene.mle_log_likelihood),
            }
            if any(
                not _numeric_equal(actual_values[name], expected)
                for name, expected in expected_values.items()
            ) or (
                not np.isfinite(actual_observed)
                or not actual_observed.is_integer()
                or int(actual_observed) != expected_observed
            ):
                raise ValueError  # noqa: TRY301
            if (
                str(row["MLE Converged"]).lower() != "true"
                or row["Single-Gene LRT Status"] != gene.likelihood_ratio_status
                or row["LRT Contract"] != REQUIRED_LRT_CONTRACT
                or row["Single-Gene Count Contract"] != SINGLE_GENE_COUNT_CONTRACT
            ):
                raise ValueError  # noqa: TRY301
        except Exception:  # noqa: BLE001, PERF203
            raise _sealed_error(
                "single-row-invalid",  # noqa: EM101
                "validate-single",
                row_index,
            ) from None
    return len(single)


def _validate_pairwise_rho(
    raw_rho: str,
    csv_order_taus: Sequence[float],
    likelihood_ratio: float,
    pair: tuple[str, str],
    *,
    effect_identifiable: bool,
) -> None:
    """Recompute rho and enforce its finite-or-degenerate-null contract."""
    if not effect_identifiable:
        if raw_rho != "":
            msg = f"Rank-deficient effect must not report rho for pair {pair}."
            raise ValueError(msg)
        return
    tau_00, tau_10, tau_01, tau_11 = csv_order_taus
    expected_rho = interaction_module.compute_marshall_olkin_rho(
        [tau_00, tau_01, tau_10, tau_11],
    )
    if expected_rho is None:
        if raw_rho != "" or likelihood_ratio > REQUIRED_UNDEFINED_RHO_LRT_TOL:
            msg = f"Invalid undefined-rho boundary result for pair {pair}."
            raise ValueError(msg)
        return
    try:
        actual_rho = float(raw_rho)
    except (TypeError, ValueError) as error:
        msg = f"Missing or non-numeric rho for pair {pair}."
        raise ValueError(msg) from error
    if (
        not np.isfinite(actual_rho)
        or abs(actual_rho) > 1 + REQUIRED_PAIR_SIMPLEX_TOL
        or actual_rho != expected_rho
    ):
        msg = f"Reported rho does not match fitted taus for pair {pair}."
        raise ValueError(msg)


def _validate_pairwise_output_impl(
    raw: bytes,
    contract: dict[str, Any],
    counts: pd.DataFrame,
    genes: dict[str, Gene],
) -> int:
    if _sequence_sha256(str(sample) for sample in counts.index) != contract["samples"][
        "ordered_ids_sha256"
    ] or len(counts) != int(contract["samples"]["count"]):
        msg = "Frozen count matrix no longer matches the contracted sample axis."
        raise ValueError(msg)
    n_samples = len(counts)
    mutation_masks = {
        feature: int.from_bytes(
            np.packbits(
                counts[feature].to_numpy(dtype=np.int64) > 0,
                bitorder="little",
            ).tobytes(),
            "little",
        )
        for feature in contract["features"]
    }
    expected_pairs = iter_tested_pairs(contract["features"])
    digest = hashlib.sha256()
    row_count = 0
    with io.StringIO(raw.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIRWISE_COLUMNS:
            msg = "Unexpected pairwise result schema."
            raise ValueError(msg)
        for expected in expected_pairs:
            row = next(reader, None)
            if row is None:
                msg = "Pairwise output ended before the frozen pair universe."
                raise ValueError(msg)
            actual = (row["Gene A"], row["Gene B"])
            if actual != expected:
                msg = (
                    f"Pair universe/order mismatch at row {row_count}: "
                    f"expected={expected}, actual={actual}"
                )
                raise ValueError(msg)
            if _base_gene(actual[0]) == _base_gene(actual[1]):
                msg = f"Same-base pair escaped the exclusion policy: {actual}"
                raise ValueError(msg)
            try:
                taus = [
                    float(row[column])
                    for column in ("Tau_00", "Tau_10", "Tau_01", "Tau_11")
                ]
                likelihood_ratio = float(row["Likelihood Ratio"])
                contingency = [
                    float(row[column]) for column in ("_00_", "_10_", "_01_", "_11_")
                ]
                fit_iterations = float(row["Fit Iterations"])
                fit_last_gain = float(row["Fit Last LL Gain"])
                fit_fixed_point_residual = float(row["Fit Fixed-Point Residual"])
                fit_kkt_residual = float(row["Fit KKT Residual"])
                null_log_likelihood = float(row["Null Log Likelihood"])
                alternative_log_likelihood = float(
                    row["Alternative Log Likelihood"],
                )
            except (TypeError, ValueError) as error:
                msg = f"Non-numeric fitted result for pair {actual}: {error}"
                raise ValueError(msg) from error
            if not _pair_simplex_is_valid(taus):
                msg = f"Invalid fitted tau simplex for pair {actual}: {taus}"
                raise ValueError(msg)
            model_taus = [taus[0], taus[2], taus[1], taus[3]]
            interaction = Interaction(genes[actual[0]], genes[actual[1]])
            expected_likelihood_ratio = interaction.compute_likelihood_ratio(
                model_taus,
            )
            expected_alternative_log_likelihood = interaction.alternative_log_likelihood
            expected_null_log_likelihood = interaction.null_log_likelihood
            if (
                expected_alternative_log_likelihood is None
                or expected_null_log_likelihood is None
            ):
                msg = f"Pair likelihood recomputation failed for {actual}."
                raise RuntimeError(msg)
            expected_fixed_point_residual, expected_kkt_residual = (
                interaction.compute_fit_certificates(model_taus)
            )
            effect_identifiability = interaction.effect_identifiability_status()
            effect_identifiable = (
                effect_identifiability == REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS
            )
            if row["Effect Identifiability"] != effect_identifiability:
                msg = f"Invalid effect-identifiability status for pair {actual}."
                raise ValueError(msg)
            deterministic_refit: Interaction | None = None
            if not effect_identifiable:
                deterministic_refit = Interaction(
                    genes[actual[0]],
                    genes[actual[1]],
                )
                deterministic_refit.estimate_tau_with_coordinate_ascent()
                deterministic_csv_taus = [
                    float(deterministic_refit.tau_00),
                    float(deterministic_refit.tau_10),
                    float(deterministic_refit.tau_01),
                    float(deterministic_refit.tau_11),
                ]
                if taus != deterministic_csv_taus:
                    msg = (
                        "Nonidentifiable pair violates deterministic tie-break: "
                        f"{actual}."
                    )
                    raise ValueError(msg)
            if not np.isfinite(likelihood_ratio) or likelihood_ratio < 0:
                msg = f"Invalid likelihood ratio for pair {actual}: {likelihood_ratio}"
                raise ValueError(msg)
            _validate_pairwise_rho(
                row["Rho"],
                taus,
                likelihood_ratio,
                actual,
                effect_identifiable=effect_identifiable,
            )
            tau_00, tau_10, tau_01, tau_11 = taus
            if effect_identifiable:
                try:
                    tau_1x = float(row["Tau_1X"])
                    tau_x1 = float(row["Tau_X1"])
                except (TypeError, ValueError) as error:
                    msg = f"Missing identifiable tau marginals for pair {actual}."
                    raise ValueError(msg) from error
                if (
                    not np.isfinite(tau_1x)
                    or not np.isfinite(tau_x1)
                    or tau_1x != tau_10 + tau_11
                    or tau_x1 != tau_01 + tau_11
                ):
                    msg = f"Invalid reported tau marginals for pair {actual}."
                    raise ValueError(msg)
            elif row["Tau_1X"] != "" or row["Tau_X1"] != "":
                msg = f"Rank-deficient tau marginals must be blank for pair {actual}."
                raise ValueError(msg)
            log_odds_defined = effect_identifiable and all(tau > 0 for tau in taus)
            if log_odds_defined:
                expected_log_odds = float(
                    math.log(tau_00)
                    + math.log(tau_11)
                    - math.log(tau_01)
                    - math.log(tau_10),
                )
                expected_wald = float(
                    expected_log_odds
                    * math.exp(-0.5 * float(logsumexp(-np.log(taus)))),
                )
                try:
                    actual_log_odds = float(row["Log Odds Ratio"])
                    actual_wald = float(row["Wald Statistic"])
                except (TypeError, ValueError) as error:
                    msg = f"Missing conventional LOR/Wald statistic for pair {actual}."
                    raise ValueError(msg) from error
                if (
                    not np.isfinite(actual_log_odds)
                    or not np.isfinite(actual_wald)
                    or actual_log_odds != expected_log_odds
                    or actual_wald != expected_wald
                ):
                    msg = (
                        f"Conventional LOR/Wald orientation mismatch for pair {actual}."
                    )
                    raise ValueError(msg)
            elif row["Log Odds Ratio"] != "" or row["Wald Statistic"] != "":
                msg = f"Undefined LOR/Wald statistic must be blank for pair {actual}."
                raise ValueError(msg)
            if (
                row["Fit Algorithm"] != REQUIRED_PAIR_FIT_CONTRACT
                or row["Fit Converged"].strip().lower() != "true"
                or not np.isfinite(fit_iterations)
                or fit_iterations < 0
                or not fit_iterations.is_integer()
                or fit_iterations > REQUIRED_PAIR_FIT_MAX_ITER
                or (fit_iterations == 0 and fit_last_gain != 0)
                or (
                    deterministic_refit is not None
                    and fit_iterations != deterministic_refit.fit_iterations
                )
                or not np.isfinite(fit_last_gain)
                or fit_last_gain < 0
                or (
                    deterministic_refit is not None
                    and not _numeric_equal(
                        fit_last_gain,
                        float(deterministic_refit.fit_last_log_likelihood_gain),
                    )
                )
                or not np.isfinite(fit_fixed_point_residual)
                or fit_fixed_point_residual < 0
                or fit_fixed_point_residual > REQUIRED_PAIR_FIT_KKT_TOL
                or not np.isfinite(fit_kkt_residual)
                or fit_kkt_residual < 0
                or fit_kkt_residual > REQUIRED_PAIR_FIT_KKT_TOL
                or not _numeric_equal(
                    alternative_log_likelihood,
                    expected_alternative_log_likelihood,
                )
                or not _numeric_equal(
                    null_log_likelihood,
                    expected_null_log_likelihood,
                )
                or not _numeric_equal(
                    likelihood_ratio,
                    expected_likelihood_ratio,
                )
                or not _numeric_equal(
                    fit_fixed_point_residual,
                    expected_fixed_point_residual,
                )
                or not _numeric_equal(
                    fit_kkt_residual,
                    expected_kkt_residual,
                )
                or row["Pair Fit Contract"] != REQUIRED_PAIR_FIT_CONTRACT
                or row["LRT Contract"] != REQUIRED_LRT_CONTRACT
            ):
                msg = f"Invalid convergence/LRT provenance for pair {actual}."
                raise ValueError(msg)
            if (
                not np.isfinite(contingency).all()
                or any(
                    value < 0 or not float(value).is_integer() for value in contingency
                )
                or int(sum(contingency)) != int(contract["samples"]["count"])
            ):
                msg = f"Invalid contingency table for pair {actual}: {contingency}"
                raise ValueError(msg)
            mask_a = mutation_masks[actual[0]]
            mask_b = mutation_masks[actual[1]]
            n_11 = (mask_a & mask_b).bit_count()
            n_10 = mask_a.bit_count() - n_11
            n_01 = mask_b.bit_count() - n_11
            n_00 = n_samples - n_10 - n_01 - n_11
            expected_contingency = [n_00, n_10, n_01, n_11]
            if contingency != expected_contingency:
                msg = (
                    f"Observed-cell semantics do not match the count matrix for pair "
                    f"{actual}: expected={expected_contingency}, actual={contingency}"
                )
                raise ValueError(msg)
            encoded = f"{actual[0]}\t{actual[1]}".encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            row_count += 1
        extra = next(reader, None)
        if extra is not None:
            msg = f"Pairwise output contains extra row after frozen universe: {extra}"
            raise ValueError(msg)
    expected_count = int(contract["pair_policy"]["row_count"])
    if row_count != expected_count:
        msg = f"Pair row count {row_count} does not match contract {expected_count}."
        raise ValueError(msg)
    expected_hash = contract["pair_policy"]["ordered_pair_sha256"]
    if digest.hexdigest() != expected_hash:
        msg = "Pairwise output pair-universe hash does not match the cohort contract."
        raise ValueError(msg)
    return row_count


def _validate_pairwise_output(
    raw: bytes,
    contract: dict[str, Any],
    counts: pd.DataFrame,
    genes: dict[str, Gene],
) -> int:
    try:
        return _validate_pairwise_output_impl(raw, contract, counts, genes)
    except SealedFitError:
        raise
    except Exception:  # noqa: BLE001
        raise _sealed_error("pair-output-invalid", "validate-pair") from None  # noqa: EM101


def validate_task_output(  # noqa: PLR0913
    task_dir: Path,
    contract: dict[str, Any],
    *,
    require_manifest: bool = True,
    bmr: str | None = None,
    scientific_inputs: tuple[pd.DataFrame, dict[str, Any]] | None = None,
    directory_fd: int | None = None,
    pinned_snapshot: Mapping[str, _PinnedRegularSnapshot] | None = None,
) -> dict[str, Any]:
    """Validate a complete task directory against its frozen cohort contract."""
    _require_exact_sample_axis(contract)
    _require_full_observation_support(contract)
    if pinned_snapshot is None:
        single_raw, pairwise_raw, manifest_raw = _read_task_output_snapshot(
            task_dir,
            require_manifest=require_manifest,
            directory_fd=directory_fd,
        )
    else:
        expected_snapshot_names = {
            "single_gene_results.csv",
            "pairwise_interaction_results.csv",
            "task_manifest.json",
        }
        if not require_manifest or set(pinned_snapshot) != expected_snapshot_names:
            error = _sealed_error(
                "output-inventory-invalid",
                "validate-output",
            )
            raise error
        single_raw = pinned_snapshot["single_gene_results.csv"].content
        pairwise_raw = pinned_snapshot["pairwise_interaction_results.csv"].content
        manifest_raw = pinned_snapshot["task_manifest.json"].content
    manifest = None
    if manifest_raw is not None:
        try:
            manifest = _parse_json_bytes(
                manifest_raw,
                path=task_dir / "task_manifest.json",
            )
        except Exception:  # noqa: BLE001
            raise _sealed_error(
                "task-manifest-invalid",  # noqa: EM101
                "validate-output",
            ) from None
        manifest_bmr = manifest.get("bmr")
        if bmr is not None and manifest_bmr != bmr:
            raise _sealed_error(
                "task-coordinate-invalid",  # noqa: EM101
                "validate-output",
            )
        bmr = manifest_bmr if isinstance(manifest_bmr, str) else None
    if bmr not in BMRS:
        raise _sealed_error("task-bmr-missing", "validate-output")  # noqa: EM101
    try:
        counts, pmfs = (
            scientific_inputs
            if scientific_inputs is not None
            else _load_frozen_scientific_inputs(contract, bmr)
        )
    except Exception:  # noqa: BLE001
        raise _sealed_error(
            "scientific-input-invalid",  # noqa: EM101
            "validate-output",
        ) from None
    genes = _build_genes(counts, contract["features"], pmfs)
    estimate_pi_for_each_gene(list(genes.values()))
    feature_count = _validate_single_gene_output(
        single_raw,
        contract,
        counts,
        pmfs,
        genes,
    )
    row_count = _validate_pairwise_output(pairwise_raw, contract, counts, genes)
    validation = {
        "features": feature_count,
        "ordered_features_sha256": _sequence_sha256(contract["features"]),
        "pairs": row_count,
        "ordered_pair_sha256": contract["pair_policy"]["ordered_pair_sha256"],
        "single_gene_sha256": hashlib.sha256(single_raw).hexdigest(),
        "pairwise_sha256": hashlib.sha256(pairwise_raw).hexdigest(),
    }
    if require_manifest:
        if manifest is None:
            raise _sealed_error(
                "task-manifest-missing",  # noqa: EM101
                "validate-output",
            )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            msg = f"Task manifest schema version is incompatible: {task_dir}"
            raise ValueError(msg)
        if manifest.get("exit_status") != 0:
            msg = f"Completed task does not record exit_status=0: {task_dir}"
            raise ValueError(msg)
        _validate_task_resource_usage(manifest, task_dir)
        expected_provenance = {
            "lrt_contract": REQUIRED_LRT_CONTRACT,
            "pair_fit_contract": REQUIRED_PAIR_FIT_CONTRACT,
            "pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
            "pair_fit_max_iterations": REQUIRED_PAIR_FIT_MAX_ITER,
            "pair_simplex_tolerance": REQUIRED_PAIR_SIMPLEX_TOL,
            "lrt_nestedness_tolerance": REQUIRED_LRT_NESTEDNESS_TOL,
            "output_recomputation_atol": REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
            "pair_identifiability_relative_tolerance": (
                REQUIRED_PAIR_IDENTIFIABILITY_RTOL
            ),
            "pair_effect_identifiability_contract": (
                REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT
            ),
            "rho_contract": REQUIRED_RHO_CONTRACT,
            "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
            "contingency_table_contract": REQUIRED_CONTINGENCY_TABLE_CONTRACT,
            "log_odds_ratio_contract": REQUIRED_LOG_ODDS_RATIO_CONTRACT,
            "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
            "gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
            "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
        }
        if any(
            manifest.get(key) != value for key, value in expected_provenance.items()
        ):
            msg = f"Task statistical-contract provenance is invalid: {task_dir}"
            raise ValueError(msg)
        expected_provider_receipt = (
            contract.get("provider_input_provenance", {}).get("root_receipt")
            if isinstance(contract.get("provider_input_provenance"), dict)
            else None
        )
        if manifest.get("provider_input_root_receipt") != expected_provider_receipt:
            msg = f"Task provider-root provenance is invalid: {task_dir}"
            raise ValueError(msg)
        if manifest.get("mutsig_pmf_contract") != contract.get(
            "mutsig_pmf_contract",
        ):
            msg = f"Task MutSig PMF provenance is invalid: {task_dir}"
            raise ValueError(msg)
        if manifest.get("mutsig_pmf_storage_contract") != contract.get(
            "mutsig_pmf_storage_contract",
        ):
            msg = f"Task MutSig PMF storage provenance is invalid: {task_dir}"
            raise ValueError(msg)
        if manifest.get("consumed_input_sha256") != _consumed_input_hashes(
            contract,
            bmr,
        ):
            raise _sealed_error(
                "consumed-input-receipt-invalid",  # noqa: EM101
                "validate-output",
            )
        niceness = manifest.get("niceness")
        if (
            not isinstance(niceness, dict)
            or set(niceness) != {"requested_increment", "resulting_process_nice"}
            or not isinstance(niceness["requested_increment"], int)
            or isinstance(niceness["requested_increment"], bool)
            or niceness["requested_increment"] < 0
            or not isinstance(niceness["resulting_process_nice"], int)
            or isinstance(niceness["resulting_process_nice"], bool)
            or niceness["resulting_process_nice"] < niceness["requested_increment"]
            or (
                contract.get("top_k") == TOP_K
                and niceness["requested_increment"] != REQUIRED_NICE_INCREMENT
            )
        ):
            msg = f"Task niceness provenance is invalid: {task_dir}"
            raise ValueError(msg)
        if manifest.get("contract_sha256") != _json_sha256(contract):
            msg = f"Task contract hash mismatch: {task_dir}"
            raise ValueError(msg)
        if manifest.get("validation") != validation:
            msg = f"Task validation record drifted from its outputs: {task_dir}"
            raise ValueError(msg)
    return validation


def execute_task(
    paths: RunPaths,
    task: Task,
    *,
    nice_increment: int = 10,
    top_k: int = TOP_K,
    expected_contract_sha256: str | None = None,
) -> str:
    """Execute one task in an isolated staging directory and atomically publish it."""
    if task.cohort not in TCGA_COHORTS or task.bmr not in BMRS:
        msg = f"Invalid task: {task}"
        raise ValueError(msg)
    if nice_increment < 0:
        msg = "Task niceness increment must be nonnegative."
        raise ValueError(msg)
    if top_k == TOP_K:
        if nice_increment != REQUIRED_NICE_INCREMENT:
            msg = (
                "Production tasks require the frozen niceness increment "
                f"{REQUIRED_NICE_INCREMENT}; observed {nice_increment}."
            )
            raise ValueError(msg)
        if expected_contract_sha256 is None:
            msg = "Production tasks require the orchestrator's frozen contract hash."
            raise ValueError(msg)
        if not _revision_authority_is_configured(paths):
            msg = "Production tasks require the pinned provider bundle authority."
            raise ValueError(msg)
        _frozen_python_executable()
        _require_internal_task_environment()
        child_run_manifest = _read_json(paths.output_root / "run_manifest.json")
        _require_pinned_import_roots(
            child_run_manifest.get("implementation_sha256"),
        )
        _require_live_resource_gate(
            paths,
            jobs=1,
            label=f"task-start-{task.cohort}-{task.bmr}",
        )
    (
        lrt_contract,
        pair_fit_contract,
        effect_identifiability_contract,
        rho_contract,
        gene_support_contract,
        contingency_table_contract,
        log_odds_ratio_contract,
    ) = _require_corrected_lrt()
    implementation_sha256 = _verify_run_implementation(paths) if top_k == TOP_K else {}
    resulting_nice = (
        os.nice(nice_increment)
        if nice_increment > 0
        else os.getpriority(os.PRIO_PROCESS, 0)
    )
    contract = _load_verified_contract(paths, task.cohort, top_k=top_k)
    if top_k == TOP_K:
        _require_canonical_mutsig_maf_binding(contract)
    contract_sha256 = _json_sha256(contract)
    if (
        expected_contract_sha256 is not None
        and contract_sha256 != expected_contract_sha256
    ):
        msg = f"Orchestrator/task contract hash mismatch for {task.cohort}."
        raise ValueError(msg)
    final_dir = _task_dir(paths, task)
    attempt_id = uuid.uuid4().hex
    work_dir = paths.output_root / "work" / task.cohort / f"{task.bmr}.{attempt_id}"
    output_root_fd = _open_secure_directory(paths.output_root, label="task output root")
    tasks_fd: int | None = None
    task_cohort_fd: int | None = None
    work_root_fd: int | None = None
    work_cohort_fd: int | None = None
    staging_fd: int | None = None
    try:
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="task output root",
        )
        tasks_fd = _ensure_directory_at(
            output_root_fd,
            "tasks",
            label="task publication root",
        )
        task_cohort_fd = _ensure_directory_at(
            tasks_fd,
            task.cohort,
            label="task publication cohort",
        )
        work_root_fd = _ensure_directory_at(
            output_root_fd,
            "work",
            label="task staging root",
        )
        work_cohort_fd = _ensure_directory_at(
            work_root_fd,
            task.cohort,
            label="task staging cohort",
        )
        if _directory_entry_exists(
            task_cohort_fd,
            task.bmr,
            label="completed task destination",
        ):
            final_fd = _open_directory_at(
                task_cohort_fd,
                task.bmr,
                label="completed task destination",
            )
            try:
                with _pinned_task_output_snapshot(final_fd) as final_snapshot:
                    _require_directory_entry_identity(
                        task_cohort_fd,
                        task.bmr,
                        final_fd,
                        label="completed task destination",
                    )
                    validate_task_output(
                        final_dir,
                        contract,
                        bmr=task.bmr,
                        directory_fd=final_fd,
                        pinned_snapshot=final_snapshot,
                    )
                    _require_directory_path_identity(
                        paths.output_root,
                        output_root_fd,
                        label="task output root",
                    )
                    _require_directory_path_identity(
                        final_dir.parent,
                        task_cohort_fd,
                        label="task publication cohort",
                    )
                    _require_directory_entry_identity(
                        task_cohort_fd,
                        task.bmr,
                        final_fd,
                        label="completed task destination",
                    )
                    _replay_pinned_task_output_snapshot(final_fd, final_snapshot)
                    _require_directory_entry_identity(
                        task_cohort_fd,
                        task.bmr,
                        final_fd,
                        label="completed task destination after artifact replay",
                    )
                for descriptor in (
                    work_cohort_fd,
                    work_root_fd,
                    task_cohort_fd,
                    tasks_fd,
                    output_root_fd,
                ):
                    if descriptor is not None:
                        os.close(descriptor)
                return "already-complete"
            finally:
                os.close(final_fd)
        staging_fd = _create_directory_at(
            work_cohort_fd,
            work_dir.name,
            label="fresh task staging directory",
        )
        _require_directory_path_identity(
            work_dir.parent,
            work_cohort_fd,
            label="task staging cohort",
        )
    except Exception:
        if staging_fd is not None:
            os.close(staging_fd)
        for descriptor in (
            work_cohort_fd,
            work_root_fd,
            task_cohort_fd,
            tasks_fd,
            output_root_fd,
        ):
            if descriptor is not None:
                os.close(descriptor)
        raise

    started = time.monotonic()
    published = False
    phase = "consume-inputs"
    failure_error: SealedFitError | None = None
    try:
        counts, pmfs = _load_frozen_scientific_inputs(contract, task.bmr)
        features = [str(feature) for feature in contract["features"]]
        counts = counts.loc[:, features]
        observed_kmax = int(counts.to_numpy().max(initial=0))
        if observed_kmax != contract["mutsig_pmf_contract"]["observed_kmax"]:
            msg = "Selected observed count support changed after preflight."
            raise ValueError(msg)  # noqa: TRY301
        if list(pmfs) != features:
            msg = "BMR PMFs do not preserve the exact frozen feature order."
            raise ValueError(msg)  # noqa: TRY301
        phase = "fit-single-gene"
        genes = _build_genes(counts, features, pmfs)
        estimate_pi_for_each_gene(genes.values())
        _write_bytes_atomic_at(
            staging_fd,
            "single_gene_results.csv",
            _single_gene_results_bytes(list(genes.values())),
            label="single-gene output",
        )
        phase = "fit-pairwise"
        written = _write_pairwise_results_at(
            staging_fd,
            "pairwise_interaction_results.csv",
            genes,
            features,
        )
        if written != contract["pair_policy"]["row_count"]:
            msg = "Writer did not emit the complete frozen pair universe."
            raise ValueError(msg)  # noqa: TRY301
        phase = "validate-output"
        validation = validate_task_output(
            work_dir,
            contract,
            require_manifest=False,
            bmr=task.bmr,
            scientific_inputs=(counts, pmfs),
            directory_fd=staging_fd,
        )
        resource_usage = _task_resource_usage(started)
        provider_root_receipt = (
            contract["provider_input_provenance"]["root_receipt"]
            if top_k == TOP_K
            else None
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cohort": task.cohort,
            "bmr": task.bmr,
            "top_k": top_k,
            "contract_sha256": contract_sha256,
            "native_support_only": True,
            "mutsig_cbase_feature_fallback": False,
            "mutsig_pmf_contract": contract["mutsig_pmf_contract"],
            "mutsig_pmf_storage_contract": contract["mutsig_pmf_storage_contract"],
            "same_base_pairs_excluded_before_fit": True,
            "lrt_contract": lrt_contract,
            "pair_fit_contract": pair_fit_contract,
            "pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
            "pair_fit_max_iterations": REQUIRED_PAIR_FIT_MAX_ITER,
            "pair_simplex_tolerance": REQUIRED_PAIR_SIMPLEX_TOL,
            "lrt_nestedness_tolerance": REQUIRED_LRT_NESTEDNESS_TOL,
            "output_recomputation_atol": REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
            "pair_identifiability_relative_tolerance": (
                REQUIRED_PAIR_IDENTIFIABILITY_RTOL
            ),
            "pair_effect_identifiability_contract": (effect_identifiability_contract),
            "rho_contract": rho_contract,
            "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
            "contingency_table_contract": contingency_table_contract,
            "log_odds_ratio_contract": log_odds_ratio_contract,
            "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
            "gene_support_contract": gene_support_contract,
            "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
            "exit_status": 0,
            "completed_at_utc": _utc_now(),
            "niceness": {
                "requested_increment": nice_increment,
                "resulting_process_nice": resulting_nice,
            },
            "provider_input_root_receipt": provider_root_receipt,
            "resource_usage": resource_usage,
            "implementation_sha256": implementation_sha256,
            "consumed_input_sha256": _consumed_input_hashes(contract, task.bmr),
            "validation": validation,
        }
        _require_directory_entry_identity(
            work_cohort_fd,
            work_dir.name,
            staging_fd,
            label="task staging directory before manifest",
        )
        _write_bytes_atomic_at(
            staging_fd,
            "task_manifest.json",
            _canonical_json(manifest) + b"\n",
            label="task manifest",
        )
        os.fsync(staging_fd)
        validate_task_output(
            work_dir,
            contract,
            bmr=task.bmr,
            scientific_inputs=(counts, pmfs),
            directory_fd=staging_fd,
        )
        if top_k == TOP_K:
            # Replay the closed bundle after fitting, immediately before visibility.
            _verify_frozen_cohort_authority(paths, contract)
        phase = "publish-output"
        _require_directory_entry_identity(
            work_cohort_fd,
            work_dir.name,
            staging_fd,
            label="task staging directory before publication",
        )
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="task output root",
        )
        _require_directory_path_identity(
            work_dir.parent,
            work_cohort_fd,
            label="task staging cohort",
        )
        _require_directory_path_identity(
            final_dir.parent,
            task_cohort_fd,
            label="task publication cohort",
        )
        _rename_exclusive_at(
            work_cohort_fd,
            work_dir.name,
            task_cohort_fd,
            task.bmr,
        )
        published = True
        _require_directory_entry_identity(
            task_cohort_fd,
            task.bmr,
            staging_fd,
            label="published task directory",
        )
        os.fsync(task_cohort_fd)
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="task output root",
        )
        _require_directory_path_identity(
            final_dir.parent,
            task_cohort_fd,
            label="task publication cohort",
        )
        with _pinned_task_output_snapshot(staging_fd) as final_snapshot:
            validate_task_output(
                final_dir,
                contract,
                bmr=task.bmr,
                scientific_inputs=(counts, pmfs),
                directory_fd=staging_fd,
                pinned_snapshot=final_snapshot,
            )
            # Rebind the exact bytes validated above to the still-visible task only
            # after all path checks, then replay each held file before returning.
            _require_directory_path_identity(
                paths.output_root,
                output_root_fd,
                label="task output root after final validation",
            )
            _require_directory_path_identity(
                paths.output_root / "tasks",
                tasks_fd,
                label="task publication root after final validation",
            )
            _require_directory_path_identity(
                final_dir.parent,
                task_cohort_fd,
                label="task publication cohort after final validation",
            )
            _require_directory_entry_identity(
                task_cohort_fd,
                task.bmr,
                staging_fd,
                label="published task directory after final validation",
            )
            _replay_pinned_task_output_snapshot(staging_fd, final_snapshot)
            _require_directory_entry_identity(
                task_cohort_fd,
                task.bmr,
                staging_fd,
                label="published task directory after artifact replay",
            )
    except Exception as error:  # noqa: BLE001
        safe_error = (
            error
            if isinstance(error, SealedFitError)
            else SealedFitError("task-execution-failed", phase)
        )
        failure = {
            "schema_version": SCHEMA_VERSION,
            "cohort": task.cohort,
            "bmr": task.bmr,
            "attempt_id": attempt_id,
            "exit_status": 1,
            "failed_at_utc": _utc_now(),
            "resource_usage": _task_resource_usage(started),
            "failure": {
                "code": safe_error.code,
                "phase": safe_error.phase,
                "row_index": safe_error.row_index,
            },
        }
        if not published:
            try:
                _require_directory_entry_identity(
                    work_cohort_fd,
                    work_dir.name,
                    staging_fd,
                    label="failed task staging directory",
                )
                for result_name in (
                    "single_gene_results.csv",
                    "pairwise_interaction_results.csv",
                    "task_manifest.json",
                ):
                    _unlink_single_link_regular_at(
                        staging_fd,
                        result_name,
                        label=result_name,
                    )
                _require_empty_directory(
                    staging_fd,
                    label="Failed task staging directory",
                )
                _write_bytes_atomic_at(
                    staging_fd,
                    "failure_manifest.json",
                    _canonical_json(failure) + b"\n",
                    label="failure manifest",
                )
                os.fsync(staging_fd)
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "Unable to publish a trusted failure manifest.",
                )
        failure_error = SealedFitError(
            safe_error.code,
            safe_error.phase,
            safe_error.row_index,
        )
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        for descriptor in (
            work_cohort_fd,
            work_root_fd,
            task_cohort_fd,
            tasks_fd,
            output_root_fd,
        ):
            if descriptor is not None:
                os.close(descriptor)
    if failure_error is not None:
        raise failure_error
    return "completed"


def safe_default_jobs(logical_cores: int | None = None) -> int:
    """Return at most three jobs and below half when the host has at least 3 cores."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    return max(1, min(MAX_GENERAL_JOBS, (cores - 1) // 2))


def _parse_darwin_memory_pressure(output: str) -> tuple[int, int]:
    total_match = re.search(r"The system has (\d+) ", output)
    free_match = re.search(r"System-wide memory free percentage: (\d+)%", output)
    if total_match is None or free_match is None:
        msg = "Could not parse macOS memory_pressure aggregate readback."
        raise RuntimeError(msg)
    total = int(total_match.group(1))
    free_percent = int(free_match.group(1))
    if total <= 0 or not 0 <= free_percent <= 100:
        msg = "macOS memory_pressure returned invalid aggregate values."
        raise RuntimeError(msg)
    return total, total * free_percent // 100


def _parse_linux_meminfo(content: str) -> tuple[int, int]:
    fields: dict[str, int] = {}
    for line in content.splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        pieces = raw_value.split()
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1] == "kB":
            fields[name] = int(pieces[0]) * 1024
    try:
        total = fields["MemTotal"]
        available = fields["MemAvailable"]
    except KeyError as error:
        msg = "Linux /proc/meminfo lacks MemTotal or MemAvailable."
        raise RuntimeError(msg) from error
    if total <= 0 or not 0 <= available <= total:
        msg = "Linux /proc/meminfo returned invalid aggregate values."
        raise RuntimeError(msg)
    return total, available


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            msg = f"No existing parent is available for disk readback: {path}"
            raise FileNotFoundError(msg)
        candidate = parent
    return candidate


def read_host_resources(output_root: Path) -> HostResourceSnapshot:
    """Read aggregate host CPU, memory, core, and filesystem capacity."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        )
        total_memory, available_memory = _parse_darwin_memory_pressure(
            result.stdout,
        )
        memory_source = "/usr/bin/memory_pressure -Q"
    elif sys.platform.startswith("linux"):
        total_memory, available_memory = _parse_linux_meminfo(
            Path("/proc/meminfo").read_text(encoding="utf-8"),
        )
        memory_source = "/proc/meminfo MemAvailable"
    else:
        msg = f"Unsupported platform for live memory gating: {sys.platform!r}."
        raise RuntimeError(msg)
    disk_parent = _nearest_existing_parent(output_root)
    load_average = float(os.getloadavg()[0])
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
    """Evaluate the frozen half-machine launch policy against one readback."""
    reasons: list[str] = []
    if snapshot.logical_cores <= 0:
        reasons.append("logical core count must be positive")
    half_cores = snapshot.logical_cores / 2
    load_is_valid = (
        math.isfinite(snapshot.load_average_1m) and snapshot.load_average_1m >= 0
    )
    projected_load = snapshot.load_average_1m + jobs if load_is_valid else None
    if not load_is_valid or projected_load is None or projected_load >= half_cores:
        reasons.append(
            "one-minute aggregate CPU load plus planned jobs is not below half "
            "the host",
        )
    if snapshot.total_memory_bytes <= 0:
        reasons.append("total memory must be positive")
    if not 0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes:
        reasons.append("available memory is outside the physical-memory range")
    if snapshot.free_disk_bytes < 0:
        reasons.append("free disk cannot be negative")
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

    required_by_prior_rss = math.ceil(
        jobs * PRIOR_TASK_PEAK_RSS_BYTES * MEMORY_HEADROOM_FACTOR,
    )
    required_by_fraction = math.ceil(
        max(snapshot.total_memory_bytes, 0) * MIN_AVAILABLE_MEMORY_FRACTION,
    )
    required_available = max(required_by_prior_rss, required_by_fraction)
    safe_cap = safe_default_jobs(max(snapshot.logical_cores, 1))
    if jobs <= 0 or jobs > safe_cap:
        reasons.append(f"jobs={jobs} exceeds safe live cap={safe_cap}")
    if (
        snapshot.total_memory_bytes > 0
        and 0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes
        and snapshot.available_memory_bytes < required_available
    ):
        reasons.append(
            "available memory is below the prior-RSS/fraction headroom gate",
        )
    if snapshot.free_disk_bytes < MIN_FREE_DISK_BYTES:
        reasons.append("free disk is below the 2x historical-output gate")
    return {
        "passed": not reasons,
        "jobs": jobs,
        "safe_job_cap": safe_cap,
        "strict_half_core_limit": half_cores,
        "projected_load_with_planned_jobs": projected_load,
        "required_available_memory_bytes": required_available,
        "required_by_prior_rss_bytes": required_by_prior_rss,
        "required_by_fraction_bytes": required_by_fraction,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "reasons": reasons,
    }


def _require_live_resource_gate(
    paths: RunPaths,
    *,
    jobs: int,
    label: str,
) -> None:
    snapshot = read_host_resources(paths.output_root)
    evaluation = evaluate_host_resource_gate(snapshot, jobs=jobs)
    snapshot_record = asdict(snapshot)
    load_state = "finite"
    if math.isnan(snapshot.load_average_1m):
        load_state = "nan"
    elif snapshot.load_average_1m == math.inf:
        load_state = "positive-infinity"
    elif snapshot.load_average_1m == -math.inf:
        load_state = "negative-infinity"
    if load_state != "finite":
        snapshot_record["load_average_1m"] = None
    snapshot_record["load_average_1m_state"] = load_state
    record = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "snapshot": snapshot_record,
        "evaluation": evaluation,
    }
    record_path = paths.output_root / "resource_readbacks" / f"{uuid.uuid4().hex}.json"
    _write_json_atomic(record_path, record)
    if (
        snapshot.total_memory_bytes > 0
        and 0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes
    ):
        available_memory = (
            f"{snapshot.available_memory_bytes / snapshot.total_memory_bytes:.0%}"
        )
    else:
        available_memory = "invalid"
    projected_load = evaluation["projected_load_with_planned_jobs"]
    projected_load_label = (
        f"{projected_load:.2f}"
        if isinstance(projected_load, (int, float))
        else "invalid"
    )
    print(
        f"resource gate {label}: jobs={jobs}, "
        f"projected_load={projected_load_label}, "
        f"available_memory={available_memory}, "
        f"free_disk={snapshot.free_disk_bytes / _GIB:.1f} GiB",
    )
    if not evaluation["passed"]:
        msg = f"Live resource gate failed: {evaluation['reasons']}"
        raise RuntimeError(msg)


def _require_internal_task_environment() -> None:
    """Require the exact deterministic, sealed child launch environment."""
    observed = dict(os.environ)
    if observed != SEALED_TASK_ENVIRONMENT:
        missing = sorted(set(SEALED_TASK_ENVIRONMENT).difference(observed))
        unexpected = sorted(set(observed).difference(SEALED_TASK_ENVIRONMENT))
        mismatched = sorted(
            name
            for name in set(observed).intersection(SEALED_TASK_ENVIRONMENT)
            if observed[name] != SEALED_TASK_ENVIRONMENT[name]
        )
        msg = (
            "Production internal tasks require the orchestrator's exact sealed "
            f"environment; missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )
        raise RuntimeError(msg)


def _frozen_python_executable() -> Path:
    """Require the runner to use the interpreter pinned by provider authority."""
    try:
        expected = PROVIDER_CHILD_PYTHON_EXECUTABLE.resolve(strict=True)
        observed = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        msg = "Unable to resolve the frozen production Python executable."
        raise RuntimeError(msg) from error
    if not expected.is_file() or observed != expected:
        msg = (
            "Production K=500 inference must run under the provider-pinned Python "
            f"executable {expected}; observed {observed}."
        )
        raise RuntimeError(msg)
    return expected


def _validate_cli_resource_options(
    *,
    jobs: int,
    mutsig_jobs: int,
    nice_increment: int,
    logical_cores: int | None = None,
) -> None:
    """Fail closed on resource-related public CLI overrides."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    safe_cap = safe_default_jobs(cores)
    if jobs <= 0:
        msg = "--jobs must be positive."
        raise ValueError(msg)
    if jobs > safe_cap:
        msg = (
            f"--jobs {jobs} exceeds the safe cap {safe_cap} for {cores} logical cores."
        )
        raise ValueError(msg)
    mutsig_cap = min(2, safe_cap)
    if mutsig_jobs <= 0 or mutsig_jobs > mutsig_cap:
        msg = (
            f"--mutsig-jobs must be between 1 and {mutsig_cap} for "
            f"{cores} logical cores."
        )
        raise ValueError(msg)
    if nice_increment != REQUIRED_NICE_INCREMENT:
        msg = (
            f"--nice must equal the frozen production increment "
            f"{REQUIRED_NICE_INCREMENT}."
        )
        raise ValueError(msg)


def _parse_cohorts(value: str | None) -> list[str]:
    if value is None:
        return list(TCGA_COHORTS)
    cohorts = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(cohorts) - set(TCGA_COHORTS))
    if unknown or not cohorts or len(cohorts) != len(set(cohorts)):
        msg = f"Invalid or duplicate TCGA cohort selection: {unknown or cohorts}"
        raise ValueError(msg)
    return cohorts


def _record_attempt(  # noqa: PLR0913
    paths: RunPaths,
    task: Task,
    attempt_id: str,
    command: list[str],
    return_code: int,
    log_path: Path,
    started_at: str,
) -> None:
    record = {
        "schema_version": SCHEMA_VERSION,
        "cohort": task.cohort,
        "bmr": task.bmr,
        "attempt_id": attempt_id,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "command": command,
        "exit_status": return_code,
        "log": _file_record(log_path),
    }
    path = (
        paths.output_root / "attempts" / task.cohort / task.bmr / f"{attempt_id}.json"
    )
    _write_json_atomic(path, record)


def _invoke_task(paths: RunPaths, task: Task, nice_increment: int) -> tuple[Task, int]:
    attempt_id = uuid.uuid4().hex
    attempt_dir = paths.output_root / "attempts" / task.cohort / task.bmr
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / f"{attempt_id}.log"
    contract_sha256 = _json_sha256(_read_json(_contract_path(paths, task.cohort)))
    command = [
        _frozen_python_executable().as_posix(),
        "-P",
        "-s",
        "-m",
        "analysis.run_tcga_revision_k500",
        "--output-root",
        paths.output_root.as_posix(),
        "--internal-cohort",
        task.cohort,
        "--internal-bmr",
        task.bmr,
        "--internal-contract-sha256",
        contract_sha256,
        "--nice",
        str(nice_increment),
    ]
    command.extend(_revision_authority_cli_args(paths))
    env = dict(SEALED_TASK_ENVIRONMENT)
    started_at = _utc_now()
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=RUNNER_REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    _record_attempt(
        paths,
        task,
        attempt_id,
        command,
        completed.returncode,
        log_path,
        started_at,
    )
    return task, completed.returncode


def _metadata_task_receipt(
    task_dir: Path,
    contract: dict[str, Any],
    task: Task,
) -> dict[str, Any]:
    """Validate and return task metadata/hashes without parsing scientific rows."""
    expected_names = {
        "pairwise_interaction_results.csv",
        "single_gene_results.csv",
        "task_manifest.json",
    }
    directory_fd = _open_secure_directory(task_dir, label="published task")
    try:
        if set(os.listdir(directory_fd)) != expected_names:  # noqa: PTH208
            msg = "Published task inventory is not closed."
            raise ValueError(msg)
        single_bytes = _read_regular_at(
            directory_fd,
            "single_gene_results.csv",
            label="single-gene output",
        )
        pairwise_bytes = _read_regular_at(
            directory_fd,
            "pairwise_interaction_results.csv",
            label="pairwise output",
        )
        manifest_bytes = _read_regular_at(
            directory_fd,
            "task_manifest.json",
            label="task manifest",
        )
    finally:
        os.close(directory_fd)
    manifest = _parse_json_bytes(
        manifest_bytes,
        path=task_dir / "task_manifest.json",
    )
    expected_manifest_keys = {
        "bmr",
        "cohort",
        "completed_at_utc",
        "consumed_input_sha256",
        "contingency_table_contract",
        "contract_sha256",
        "exit_status",
        "gene_support_contract",
        "implementation_sha256",
        "log_odds_ratio_contract",
        "lrt_contract",
        "output_recomputation_atol",
        "mutsig_cbase_feature_fallback",
        "mutsig_pmf_contract",
        "mutsig_pmf_storage_contract",
        "native_support_only",
        "niceness",
        "observation_support_universe",
        "pair_effect_identifiability_contract",
        "pair_fit_contract",
        "pair_fit_kkt_tolerance",
        "pair_fit_max_iterations",
        "pair_identifiability_relative_tolerance",
        "pair_simplex_tolerance",
        "lrt_nestedness_tolerance",
        "provider_input_root_receipt",
        "resource_usage",
        "rho_contract",
        "same_base_pairs_excluded_before_fit",
        "sample_axis_contract",
        "schema_version",
        "top_k",
        "undefined_rho_lrt_tolerance",
        "validation",
    }
    if set(manifest) != expected_manifest_keys:
        msg = "Published task manifest has an invalid closed schema."
        raise ValueError(msg)
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or set(validation) != {
        "features",
        "ordered_features_sha256",
        "ordered_pair_sha256",
        "pairs",
        "pairwise_sha256",
        "single_gene_sha256",
    }:
        msg = "Published task validation receipt has an invalid schema."
        raise ValueError(msg)
    expected_feature_hash = _sequence_sha256(contract["features"])
    expected_provider_receipt = (
        contract.get("provider_input_provenance", {}).get("root_receipt")
        if isinstance(contract.get("provider_input_provenance"), dict)
        else None
    )
    expected_provenance = {
        "contingency_table_contract": REQUIRED_CONTINGENCY_TABLE_CONTRACT,
        "gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
        "log_odds_ratio_contract": REQUIRED_LOG_ODDS_RATIO_CONTRACT,
        "lrt_contract": REQUIRED_LRT_CONTRACT,
        "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
        "pair_effect_identifiability_contract": (
            REQUIRED_PAIR_EFFECT_IDENTIFIABILITY_CONTRACT
        ),
        "pair_fit_contract": REQUIRED_PAIR_FIT_CONTRACT,
        "pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
        "pair_fit_max_iterations": REQUIRED_PAIR_FIT_MAX_ITER,
        "pair_identifiability_relative_tolerance": (REQUIRED_PAIR_IDENTIFIABILITY_RTOL),
        "pair_simplex_tolerance": REQUIRED_PAIR_SIMPLEX_TOL,
        "lrt_nestedness_tolerance": REQUIRED_LRT_NESTEDNESS_TOL,
        "output_recomputation_atol": REQUIRED_OUTPUT_RECOMPUTATION_ATOL,
        "rho_contract": REQUIRED_RHO_CONTRACT,
        "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
        "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
    }
    niceness = manifest.get("niceness")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("cohort") != task.cohort
        or manifest.get("bmr") != task.bmr
        or manifest.get("top_k") != contract.get("top_k")
        or manifest.get("exit_status") != 0
        or manifest.get("native_support_only") is not True
        or manifest.get("mutsig_cbase_feature_fallback") is not False
        or manifest.get("mutsig_pmf_contract") != contract.get("mutsig_pmf_contract")
        or manifest.get("mutsig_pmf_storage_contract")
        != contract.get("mutsig_pmf_storage_contract")
        or manifest.get("same_base_pairs_excluded_before_fit") is not True
        or any(manifest.get(key) != value for key, value in expected_provenance.items())
        or manifest.get("contract_sha256") != _json_sha256(contract)
        or manifest.get("provider_input_root_receipt") != expected_provider_receipt
        or manifest.get("consumed_input_sha256")
        != _consumed_input_hashes(contract, task.bmr)
        or not isinstance(manifest.get("implementation_sha256"), dict)
        or not isinstance(niceness, dict)
        or set(niceness) != {"requested_increment", "resulting_process_nice"}
        or not isinstance(niceness.get("requested_increment"), int)
        or isinstance(niceness.get("requested_increment"), bool)
        or niceness["requested_increment"] < 0
        or not isinstance(niceness.get("resulting_process_nice"), int)
        or isinstance(niceness.get("resulting_process_nice"), bool)
        or niceness["resulting_process_nice"] < niceness["requested_increment"]
        or (
            contract.get("top_k") == TOP_K
            and niceness["requested_increment"] != REQUIRED_NICE_INCREMENT
        )
        or validation.get("features") != len(contract["features"])
        or validation.get("ordered_features_sha256") != expected_feature_hash
        or validation.get("pairs") != contract["pair_policy"]["row_count"]
        or validation.get("ordered_pair_sha256")
        != contract["pair_policy"]["ordered_pair_sha256"]
        or validation.get("single_gene_sha256")
        != hashlib.sha256(single_bytes).hexdigest()
        or validation.get("pairwise_sha256")
        != hashlib.sha256(pairwise_bytes).hexdigest()
    ):
        msg = "Published task metadata/hash receipt does not match its contract."
        raise ValueError(msg)
    _validate_task_resource_usage(manifest, task_dir)
    return {
        "bmr": task.bmr,
        "cohort": task.cohort,
        "consumed_input_sha256": manifest["consumed_input_sha256"],
        "contract_sha256": manifest["contract_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "pairwise_interaction_results": {
            "bytes": len(pairwise_bytes),
            "sha256": hashlib.sha256(pairwise_bytes).hexdigest(),
        },
        "provider_input_root_receipt": manifest["provider_input_root_receipt"],
        "single_gene_results": {
            "bytes": len(single_bytes),
            "sha256": hashlib.sha256(single_bytes).hexdigest(),
        },
        "task_manifest": {
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    }


def _metadata_task_state(
    task_dir: Path,
    contract: dict[str, Any],
    task: Task,
) -> str:
    """Return completion state after metadata/hash-only task validation."""
    _metadata_task_receipt(task_dir, contract, task)
    return "complete"


def _require_completion_run_manifest(
    paths: RunPaths,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the frozen run/root anchors without replaying scientific inputs."""
    if not _revision_authority_is_configured(paths):
        msg = "Sealed completion requires the complete revision authority."
        raise ValueError(msg)
    canonical_root = paths.canonical_input_root
    input_approval = paths.input_approval_manifest
    fit_approval = paths.fit_approval_manifest
    provider_root = paths.provider_input_root
    if (
        canonical_root is None
        or input_approval is None
        or fit_approval is None
        or provider_root is None
    ):
        msg = "Sealed completion authority disappeared after validation."
        raise RuntimeError(msg)
    authority = manifest.get("revision_authority")
    expected_authority_keys = {
        "canonical_input_root",
        "configured",
        "expected_canonical_input_sha256",
        "expected_fit_approval_sha256",
        "expected_input_approval_sha256",
        "fit_approval_manifest",
        "input_approval_manifest",
        "provider_input",
    }
    if not isinstance(authority, dict) or set(authority) != expected_authority_keys:
        msg = "Run manifest revision authority has an invalid closed schema."
        raise ValueError(msg)
    provider = authority.get("provider_input")
    expected_provider_keys = {
        "association_outputs_opened",
        "cohort_provider_receipts_sha256",
        "contract",
        "expected_manifest_sha256",
        "full_acceptance_receipt",
        "full_acceptance_receipt_sha256",
        "git_executable",
        "manifest",
        "root",
    }
    provider_manifest = provider.get("manifest") if isinstance(provider, dict) else None
    provider_git = (
        provider.get("git_executable") if isinstance(provider, dict) else None
    )
    provider_full_acceptance = (
        provider.get("full_acceptance_receipt") if isinstance(provider, dict) else None
    )
    provider_full_acceptance_sha256 = (
        provider.get("full_acceptance_receipt_sha256")
        if isinstance(provider, dict)
        else None
    )
    expected_provider_manifest = provider_root / "provider_input_manifest.json"
    if (
        manifest.get("analysis") != "tcga-revision-k500"
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("top_k") != TOP_K
        or manifest.get("cohorts") != list(TCGA_COHORTS)
        or manifest.get("bmrs") != list(BMRS)
        or manifest.get("signed_tested_family") != asdict(REQUIRED_TESTED_FAMILY)
        or manifest.get("tested_family_implementation")
        != asdict(REQUIRED_TESTED_FAMILY)
        or not isinstance(manifest.get("implementation_sha256"), dict)
        or not manifest["implementation_sha256"]
        or not isinstance(manifest.get("git"), dict)
        or manifest["git"].get("dirty") is not False
        or authority.get("configured") is not True
        or authority.get("canonical_input_root") != canonical_root.as_posix()
        or authority.get("input_approval_manifest") != input_approval.as_posix()
        or authority.get("expected_input_approval_sha256")
        != paths.expected_input_approval_sha256
        or authority.get("fit_approval_manifest") != fit_approval.as_posix()
        or authority.get("expected_fit_approval_sha256")
        != paths.expected_fit_approval_sha256
        or authority.get("expected_canonical_input_sha256")
        != paths.expected_canonical_input_sha256
        or not isinstance(provider, dict)
        or set(provider) != expected_provider_keys
        or provider.get("contract") != PROVIDER_INPUT_CONTRACT
        or provider.get("root") != provider_root.as_posix()
        or provider.get("expected_manifest_sha256")
        != paths.expected_provider_input_manifest_sha256
        or provider.get("association_outputs_opened") is not False
        or not isinstance(provider_full_acceptance, dict)
        or provider_full_acceptance.get("contract")
        != "provider-full-acceptance-receipt-v1"
        or provider_full_acceptance.get("provider_manifest_sha256")
        != paths.expected_provider_input_manifest_sha256
        or provider_full_acceptance.get("full_inventory_validated") is not True
        or provider_full_acceptance.get("association_outputs_opened") is not False
        or not isinstance(provider_full_acceptance_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", provider_full_acceptance_sha256) is None
        or full_acceptance_receipt_sha256(provider_full_acceptance)
        != provider_full_acceptance_sha256
        or not isinstance(provider_git, dict)
        or set(provider_git) != {"bytes", "path", "sha256"}
        or not isinstance(provider_manifest, dict)
        or set(provider_manifest) != {"bytes", "path", "sha256"}
        or provider_manifest.get("path") != expected_provider_manifest.as_posix()
        or provider_manifest.get("sha256")
        != paths.expected_provider_input_manifest_sha256
    ):
        msg = "Run manifest does not match the independently pinned completion roots."
        raise ValueError(msg)
    _read_frozen_record_bytes(provider_git, label="completion provider-authorized Git")
    return authority


def _require_closed_completion_layout(paths: RunPaths) -> None:
    """Require exactly the canonical contract/task coordinates before sealing."""
    contracts_fd = _open_secure_directory(
        paths.output_root / "contracts",
        label="completion contracts",
    )
    try:
        expected_contracts = {f"{cohort}.json" for cohort in TCGA_COHORTS}
        if set(os.listdir(contracts_fd)) != expected_contracts:  # noqa: PTH208
            msg = "Completion contract inventory is not the canonical 32-cohort set."
            raise ValueError(msg)
    finally:
        os.close(contracts_fd)
    tasks_fd = _open_secure_directory(
        paths.output_root / "tasks",
        label="completion tasks",
    )
    try:
        if set(os.listdir(tasks_fd)) != set(TCGA_COHORTS):  # noqa: PTH208
            msg = "Completion task inventory is not the canonical cohort set."
            raise ValueError(msg)
    finally:
        os.close(tasks_fd)
    for cohort in TCGA_COHORTS:
        cohort_fd = _open_secure_directory(
            paths.output_root / "tasks" / cohort,
            label=f"completion task cohort {cohort}",
        )
        try:
            if set(os.listdir(cohort_fd)) != set(BMRS):  # noqa: PTH208
                msg = f"Completion task inventory is incomplete for {cohort}."
                raise ValueError(msg)
        finally:
            os.close(cohort_fd)


def _publish_sealed_completion(
    paths: RunPaths,
    payload: dict[str, Any],
    *,
    output_root_fd: int | None = None,
) -> dict[str, Any]:
    """Publish completion last by pinned dirfd, fsync, and exact readback."""
    path = paths.output_root / SEALED_COMPLETION_NAME
    expected = _canonical_json(payload) + b"\n"
    owned_descriptor = output_root_fd is None
    if output_root_fd is None:
        output_root_fd = _open_secure_directory(
            paths.output_root,
            label="sealed completion output root",
        )
    completion_fd: int | None = None
    try:
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="sealed completion output root",
        )
        try:
            _write_bytes_atomic_at(
                output_root_fd,
                SEALED_COMPLETION_NAME,
                expected,
                label="sealed completion manifest",
            )
        except FileExistsError:
            observed = _read_regular_at(
                output_root_fd,
                SEALED_COMPLETION_NAME,
                label="sealed completion manifest",
            )
            if observed != expected:
                msg = "Existing sealed completion manifest differs from the whole grid."
                raise ValueError(msg) from None
        os.fsync(output_root_fd)
        completion_fd = os.open(
            SEALED_COMPLETION_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=output_root_fd,
        )
        anchored_stat = os.fstat(completion_fd)
        _require_regular_entry_identity(
            output_root_fd,
            SEALED_COMPLETION_NAME,
            completion_fd,
            label="sealed completion manifest before final readback",
        )
        observed = _read_regular_at(
            output_root_fd,
            SEALED_COMPLETION_NAME,
            label="sealed completion manifest",
        )
        if observed != expected:
            msg = "Sealed completion bytes changed during exclusive publication."
            raise ValueError(msg)
        # Replay the held inode only after the path and visible-entry checks below.
        # This closes the same-inode mutation interval after the first readback.
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="sealed completion output root after final readback",
        )
        _require_regular_entry_identity(
            output_root_fd,
            SEALED_COMPLETION_NAME,
            completion_fd,
            label="sealed completion manifest after final readback",
        )
        replayed, _ = _read_pinned_regular_replay(
            completion_fd,
            anchored_stat,
            label="sealed completion manifest final pinned replay",
        )
        if replayed != expected:
            msg = "Sealed completion bytes changed after its final path checks."
            raise ValueError(msg)
        result = {
            "bytes": len(replayed),
            "path": path.as_posix(),
            "sha256": hashlib.sha256(replayed).hexdigest(),
        }
        _require_regular_entry_identity(
            output_root_fd,
            SEALED_COMPLETION_NAME,
            completion_fd,
            label="sealed completion manifest after final pinned replay",
        )
        return result
    finally:
        if completion_fd is not None:
            os.close(completion_fd)
        if owned_descriptor:
            os.close(output_root_fd)


def _finalize_sealed_completion(
    paths: RunPaths,
    cohorts: Sequence[str],
) -> dict[str, Any]:
    """Seal the grid while pinning and revalidating the output-root inode."""
    output_root_fd = _open_secure_directory(
        paths.output_root,
        label="completion output root",
    )
    try:
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="completion output root",
        )
        result = _finalize_sealed_completion_at(paths, cohorts, output_root_fd)
        _require_directory_path_identity(
            paths.output_root,
            output_root_fd,
            label="completion output root",
        )
        return result
    finally:
        os.close(output_root_fd)


def _finalize_sealed_completion_at(
    paths: RunPaths,
    cohorts: Sequence[str],
    output_root_fd: int,
) -> dict[str, Any]:
    """Build metadata receipts and publish through one pinned output-root dirfd."""
    if len(cohorts) != len(TCGA_COHORTS) or set(cohorts) != set(TCGA_COHORTS):
        msg = "A cohort subset cannot publish a whole-grid completion manifest."
        raise ValueError(msg)
    _require_closed_completion_layout(paths)
    run_manifest_path = paths.output_root / "run_manifest.json"
    run_manifest_raw = _read_secure_regular_bytes(
        run_manifest_path,
        label="completion run manifest",
    )
    run_manifest = _parse_json_bytes(run_manifest_raw, path=run_manifest_path)
    authority = _require_completion_run_manifest(paths, run_manifest)
    implementation_sha256 = run_manifest["implementation_sha256"]
    provider_receipt = authority["provider_input"]
    contract_receipts = []
    task_receipts = []
    coordinates = []
    for cohort in TCGA_COHORTS:
        contract_path = _contract_path(paths, cohort)
        contract_raw = _read_secure_regular_bytes(
            contract_path,
            label=f"completion contract {cohort}",
        )
        contract = _parse_json_bytes(contract_raw, path=contract_path)
        if contract.get("cohort") != cohort or contract.get("top_k") != TOP_K:
            msg = f"Completion contract coordinates are invalid for {cohort}."
            raise ValueError(msg)
        contract_sha256 = _json_sha256(contract)
        contract_receipts.append(
            {
                "bytes": len(contract_raw),
                "cohort": cohort,
                "contract_sha256": contract_sha256,
                "file_sha256": hashlib.sha256(contract_raw).hexdigest(),
            },
        )
        for bmr in BMRS:
            task = Task(cohort, bmr)
            receipt = _metadata_task_receipt(_task_dir(paths, task), contract, task)
            if (
                receipt.pop("implementation_sha256") != implementation_sha256
                or receipt.pop("provider_input_root_receipt") != provider_receipt
                or receipt["contract_sha256"] != contract_sha256
            ):
                msg = "Completed task does not bind the frozen run/root authority."
                raise ValueError(msg)
            coordinates.append(f"{cohort}/{bmr}")
            task_receipts.append(receipt)
    payload = {
        "analysis": "tcga-revision-k500",
        "authority": authority,
        "bmrs": list(BMRS),
        "cohorts": list(TCGA_COHORTS),
        "contract": SEALED_COMPLETION_CONTRACT,
        "contracts": contract_receipts,
        "downstream_binding": {
            "field": "upstream_result_manifest_sha256",
            "stage": "inspect-tcga-k500",
        },
        "grid": {
            "ordered_coordinates_sha256": _sequence_sha256(coordinates),
            "task_count": len(task_receipts),
        },
        "result_rows_opened": False,
        "run_manifest": {
            "bytes": len(run_manifest_raw),
            "sha256": hashlib.sha256(run_manifest_raw).hexdigest(),
        },
        "schema": SEALED_COMPLETION_SCHEMA,
        "tasks": task_receipts,
        "top_k": TOP_K,
    }
    manifest_file = _publish_sealed_completion(
        paths,
        payload,
        output_root_fd=output_root_fd,
    )
    return {"manifest": payload, "manifest_file": manifest_file}


def _status(paths: RunPaths, cohorts: Sequence[str]) -> dict[str, Any]:
    """Return aggregate manifest/hash-only state without inspecting result rows."""
    records: dict[str, dict[str, str]] = {}
    counts = {"complete": 0, "pending": 0, "invalid": 0}
    for cohort in cohorts:
        contract_path = _contract_path(paths, cohort)
        contract = None
        contract_invalid = False
        if os.path.lexists(contract_path):
            try:
                contract = _read_json(contract_path)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                contract_invalid = True
        records[cohort] = {}
        for bmr in BMRS:
            task = Task(cohort, bmr)
            task_dir = _task_dir(paths, task)
            if not os.path.lexists(task_dir):
                state = "pending"
            elif contract is None or contract_invalid:
                state = "invalid"
            else:
                try:
                    state = _metadata_task_state(task_dir, contract, task)
                except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
                    state = "invalid"
            records[cohort][bmr] = state
            counts[state] += 1
    return {"counts": counts, "tasks": records}


def _run_task_batch(
    paths: RunPaths,
    tasks: Sequence[Task],
    *,
    jobs: int,
    nice_increment: int,
) -> int:
    failures = 0
    if not tasks:
        return failures
    if jobs <= 0:
        msg = "Task-batch concurrency must be positive."
        raise ValueError(msg)
    for start in range(0, len(tasks), jobs):
        wave = tasks[start : start + jobs]
        wave_jobs = len(wave)
        wave_number = start // jobs + 1
        _require_live_resource_gate(
            paths,
            jobs=wave_jobs,
            label="batch-{}-{}".format(
                wave_number,
                "-".join(f"{task.cohort}/{task.bmr}" for task in wave),
            ),
        )
        with ThreadPoolExecutor(max_workers=wave_jobs) as executor:
            futures = {
                executor.submit(_invoke_task, paths, task, nice_increment): task
                for task in wave
            }
            for future in as_completed(futures):
                task, return_code = future.result()
                if return_code == 0:
                    print(f"complete {task.cohort}/{task.bmr}")
                else:
                    failures += 1
                    print(
                        f"failed {task.cohort}/{task.bmr}: exit={return_code}",
                        file=sys.stderr,
                    )
    return failures


def _run_serial_canaries(
    paths: RunPaths,
    tasks: Sequence[Task],
    *,
    nice_increment: int,
) -> int:
    for position, task in enumerate(tasks, start=1):
        _require_live_resource_gate(
            paths,
            jobs=1,
            label=f"canary-{position}-{task.cohort}-{task.bmr}",
        )
        completed_task, return_code = _invoke_task(paths, task, nice_increment)
        if return_code != 0:
            print(
                f"failed {completed_task.cohort}/{completed_task.bmr}: "
                f"exit={return_code}",
                file=sys.stderr,
            )
            return 1
        print(f"complete {completed_task.cohort}/{completed_task.bmr}")
    return 0


def _require_validated_canary_outputs(paths: RunPaths) -> None:
    """Require all frozen CHOL outputs before launching a non-canary task."""
    task_dirs = [_task_dir(paths, Task(CANARY_COHORT, bmr)) for bmr in BMRS]
    missing = [task_dir for task_dir in task_dirs if not task_dir.is_dir()]
    if missing:
        msg = (
            f"Validated {CANARY_COHORT} canaries for all backgrounds are required "
            "before any non-canary production task; include CHOL in this run or "
            "resume from a run with complete validated CHOL outputs. "
            f"Missing: {[path.as_posix() for path in missing]}"
        )
        raise RuntimeError(msg)
    try:
        contract = _load_verified_contract(paths, CANARY_COHORT, top_k=TOP_K)
        for task_dir in task_dirs:
            validate_task_output(task_dir, contract)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        msg = (
            f"Validated {CANARY_COHORT} canaries for all backgrounds are required "
            "before any non-canary production task; include CHOL in this run or "
            "resume from a run with complete validated CHOL outputs."
        )
        raise RuntimeError(msg) from error


def _orchestrate(  # noqa: PLR0913
    paths: RunPaths,
    cohorts: Sequence[str],
    *,
    jobs: int,
    mutsig_jobs: int,
    nice_increment: int,
    preflight_only: bool,
) -> int:
    if not cohorts:
        msg = "Production orchestration requires at least one cohort."
        raise ValueError(msg)
    _prime_parent_revision_authority(paths, cohorts[0])
    _initialize_run(paths, allow_dirty=preflight_only)
    contracts = {}
    for cohort in cohorts:
        contracts[cohort] = _ensure_contract(paths, cohort)
        print(f"preflight {cohort}: valid K={TOP_K} contract")
    if preflight_only:
        _verify_recorded_source_hashes(paths)
        try:
            _require_corrected_lrt()
        except RuntimeError as error:
            print(f"preflight launch gate: {error}", file=sys.stderr)
            return 2
        print("preflight launch gate: corrected LRT contract present")
        try:
            for contract in contracts.values():
                _require_canonical_mutsig_maf_binding(contract)
        except RuntimeError as error:
            print(f"preflight launch gate: {error}", file=sys.stderr)
            return 2
        return 0

    _require_corrected_lrt()
    for contract in contracts.values():
        _require_canonical_mutsig_maf_binding(contract)
    pending = []
    for cohort in cohorts:
        contract = _read_json(_contract_path(paths, cohort))
        for bmr in BMRS:
            task = Task(cohort, bmr)
            task_dir = _task_dir(paths, task)
            if task_dir.exists():
                validate_task_output(task_dir, contract)
                print(f"skip {cohort}/{bmr}: validated complete")
            else:
                pending.append(task)
    if not pending:
        if len(cohorts) == len(TCGA_COHORTS) and set(cohorts) == set(TCGA_COHORTS):
            _prime_parent_revision_authority(paths, TCGA_COHORTS[0])
            completion = _finalize_sealed_completion(paths, cohorts)
            print(
                f"sealed completion {completion['manifest_file']['sha256']}",
            )
        print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
        return 0

    canaries = [task for task in pending if task.cohort == CANARY_COHORT]
    non_canaries = [task for task in pending if task.cohort != CANARY_COHORT]
    canary_failures = _run_serial_canaries(
        paths,
        canaries,
        nice_increment=nice_increment,
    )
    if canary_failures:
        print("canary failed; full K=500 grid remains gated", file=sys.stderr)
        print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
        return 1
    if non_canaries:
        _require_validated_canary_outputs(paths)

    cohort_level = [task for task in non_canaries if task.bmr != "mutsig"]
    light_mutsig = [
        task
        for task in non_canaries
        if task.bmr == "mutsig" and task.cohort not in MEMORY_HEAVY_MUTSIG_COHORTS
    ]
    heavy_mutsig = [
        task
        for task in non_canaries
        if task.bmr == "mutsig" and task.cohort in MEMORY_HEAVY_MUTSIG_COHORTS
    ]
    failures = canary_failures + _run_task_batch(
        paths,
        cohort_level,
        jobs=jobs,
        nice_increment=nice_increment,
    )
    failures += _run_task_batch(
        paths,
        light_mutsig,
        jobs=mutsig_jobs,
        nice_increment=nice_increment,
    )
    # The five largest MutSig tensors/selected-feature PMF sets run serially on the
    # 24 GB development host. This sacrifices some wall time to avoid swap pressure.
    failures += _run_task_batch(
        paths,
        heavy_mutsig,
        jobs=1,
        nice_increment=nice_increment,
    )
    if (
        failures == 0
        and len(cohorts) == len(TCGA_COHORTS)
        and set(cohorts) == set(TCGA_COHORTS)
    ):
        _prime_parent_revision_authority(paths, TCGA_COHORTS[0])
        completion = _finalize_sealed_completion(paths, cohorts)
        print(f"sealed completion {completion['manifest_file']['sha256']}")
    print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
    return int(failures > 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provider-input-root", type=Path)
    parser.add_argument("--expected-provider-input-manifest-sha256")
    parser.add_argument("--canonical-input-root", type=Path)
    parser.add_argument("--input-approval-manifest", type=Path)
    parser.add_argument("--expected-input-approval-sha256")
    parser.add_argument("--fit-approval-manifest", type=Path)
    parser.add_argument("--expected-fit-approval-sha256")
    parser.add_argument("--expected-canonical-input-sha256")
    parser.add_argument(
        "--cohorts",
        help="Comma-separated TCGA cohort subset; default is the canonical 32.",
    )
    parser.add_argument("--jobs", type=int, default=safe_default_jobs())
    parser.add_argument(
        "--mutsig-jobs",
        type=int,
        default=min(2, safe_default_jobs()),
    )
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--internal-cohort",
        choices=TCGA_COHORTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-bmr",
        choices=BMRS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-contract-sha256", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Validate, resume, or execute the frozen revision grid."""
    args = _parser().parse_args()
    provider_input_root = (
        None
        if args.provider_input_root is None
        else args.provider_input_root.absolute()
    )
    unconfigured_provider_root = (
        args.output_root.absolute().parent / ".provider-input-unconfigured"
    )
    derived_provider_root = provider_input_root or unconfigured_provider_root
    paths = RunPaths(
        source_root=derived_provider_root / "cohorts",
        mutsig_root=derived_provider_root / "mutsig",
        output_root=args.output_root.absolute(),
        canonical_input_root=(
            None
            if args.canonical_input_root is None
            else args.canonical_input_root.absolute()
        ),
        input_approval_manifest=(
            None
            if args.input_approval_manifest is None
            else args.input_approval_manifest.absolute()
        ),
        expected_input_approval_sha256=args.expected_input_approval_sha256,
        fit_approval_manifest=(
            None
            if args.fit_approval_manifest is None
            else args.fit_approval_manifest.absolute()
        ),
        expected_fit_approval_sha256=args.expected_fit_approval_sha256,
        expected_canonical_input_sha256=args.expected_canonical_input_sha256,
        provider_input_root=provider_input_root,
        expected_provider_input_manifest_sha256=(
            args.expected_provider_input_manifest_sha256
        ),
    )
    if args.nice < 0:
        msg = "--nice must be nonnegative."
        raise ValueError(msg)
    internal_values = (
        args.internal_cohort,
        args.internal_bmr,
        args.internal_contract_sha256,
    )
    if any(value is not None for value in internal_values):
        if any(value is None for value in internal_values):
            msg = "All internal task coordinates and the contract hash are required."
            raise ValueError(msg)
        if not _revision_authority_is_configured(paths):
            msg = "Production internal tasks require the complete revision authority."
            raise ValueError(msg)
        execute_task(
            paths,
            Task(args.internal_cohort, args.internal_bmr),
            nice_increment=args.nice,
            expected_contract_sha256=args.internal_contract_sha256,
        )
        return

    cohorts = _parse_cohorts(args.cohorts)
    if args.status:
        if not paths.output_root.exists():
            msg = f"Run output root does not exist: {paths.output_root}"
            raise FileNotFoundError(msg)
        print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
        return
    if not _revision_authority_is_configured(paths):
        msg = (
            "K=500 preflight and execution require the complete independently pinned "
            "revision authority."
        )
        raise ValueError(msg)
    _validate_cli_resource_options(
        jobs=args.jobs,
        mutsig_jobs=args.mutsig_jobs,
        nice_increment=args.nice,
    )
    with _same_uid_machine_execution_lease(paths.output_root):
        exit_status = _orchestrate(
            paths,
            cohorts,
            jobs=args.jobs,
            mutsig_jobs=args.mutsig_jobs,
            nice_increment=args.nice,
            preflight_only=args.preflight_only,
        )
    raise SystemExit(exit_status)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

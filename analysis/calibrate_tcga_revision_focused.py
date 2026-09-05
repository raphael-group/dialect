"""Run the result-blind affirmative calibration for the focused K=500 analysis.

The primary gate covers MutSig in all 32 cohorts.  CBaSE and DIG are evaluated in
five predeclared cohorts as descriptive sensitivity checks only.  Every cell is
generated from fitted independent DIALECT marginals and measures rejection on 32
disjoint K=500-axis pairs.  Each primary cohort/pair/alpha endpoint receives a
simultaneous one-sided exact-binomial upper confidence bound.  This is finite-
scenario evidence, not a proof of uniform p-value or false-discovery-rate control.
Repeated profile-LRT fits use a bounded batch kernel with unchanged scalar-model
reconciliation at log-domain, test-decision, and rank-decision boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from analysis import run_tcga_revision_focused as focused_runner
from analysis import run_tcga_revision_k500 as core
from analysis.calibration_batch import fit_gene_pairs_batched
from dialect.data.tcga import TCGA_COHORTS
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
CONFIG_PATH: Final = Path(__file__).with_name("tcga_revision_calibration_config.json")
RUN_CONTRACT: Final = "focused-parametric-null-calibration-run-v3"
TASK_CONTRACT: Final = "focused-parametric-null-calibration-cell-v3"
SUMMARY_CONTRACT: Final = "focused-parametric-null-calibration-summary-v3"
TASK_DATA_NAME: Final = "calibration_arrays.npz"
TASK_MANIFEST_NAME: Final = "task_manifest.json"
RUN_MANIFEST_NAME: Final = "run_manifest.json"
SUMMARY_NAME: Final = "calibration_summary.json"
SUMMARY_TABLE_NAME: Final = "calibration_cells.csv"
PRIMARY_ROLE: Final = "primary-gate"
DESCRIPTIVE_ROLE: Final = "descriptive-only"
MARGINAL_SCREEN: Final = "marginal_lrt"
ENDPOINT_ACCEPTED: Final = "pass"
ENDPOINT_REJECTED: Final = "fail"
GATE_NOT_APPLICABLE: Final = "not-applicable"
FIT_KERNEL_CONTRACT: Final = (
    "bounded-batched-profile-lrt-with-scalar-boundary-reconciliation-v1"
)
REPLICATE_CHUNK_RULE: Final = "max(1,min(512,128000//sample-count))"
THREAD_ENV: Final = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SUMMARY_COLUMNS: Final = (
    "cohort",
    "provider",
    "role",
    "screen",
    "sentinel_pair_index",
    "threshold",
    "events",
    "trials",
    "rate",
    "reportable_trials",
    "nonreportable_trials",
    "gate_endpoint",
    "exact_binomial_familywise_error",
    "exact_binomial_endpoint_count",
    "bonferroni_endpoint_error",
    "clopper_pearson_upper_bound",
    "acceptance_upper_bound",
    "endpoint_gate_pass",
)


@dataclass(frozen=True, slots=True)
class ProtocolCell:
    """One immutable calibration coordinate and its inferential role."""

    cohort: str
    provider: str
    role: str


@dataclass(frozen=True, slots=True)
class Cell:
    """One fitted cohort/provider independence generator."""

    cohort: str
    provider: str
    features: tuple[str, ...]
    samples: tuple[str, ...]
    pmfs: Mapping[str, Any]
    pi: np.ndarray
    source_task_manifest: Mapping[str, int | str]
    single_gene_input: Mapping[str, int | str]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _result_blindness_receipt() -> dict[str, Any]:
    """Disclose integrity hashing without implying pairwise bytes stay unopened."""
    return {
        "observed_pair_statistics_parsed_or_inspected": False,
        "pairwise_files_integrity_hashed": True,
        "pairwise_hash_use": "run-integrity-validation-only",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        msg = f"Required regular file is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _protocol_cells(config: Mapping[str, Any]) -> tuple[ProtocolCell, ...]:
    """Return the exact ordered primary and descriptive calibration cells."""
    cells = config["cells"]
    primary = cells["primary_gate"]
    descriptive = cells["descriptive"]
    result = tuple(
        ProtocolCell(cohort, provider, str(primary["role"]))
        for cohort in primary["cohorts"]
        for provider in primary["providers"]
    ) + tuple(
        ProtocolCell(cohort, provider, str(descriptive["role"]))
        for cohort in descriptive["cohorts"]
        for provider in descriptive["providers"]
    )
    coordinates = {(cell.cohort, cell.provider) for cell in result}
    if len(coordinates) != len(result):
        msg = "Calibration protocol contains duplicate cohort/provider cells."
        raise ValueError(msg)
    return result


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cells = config.get("cells", {})
    primary = cells.get("primary_gate", {})
    descriptive = cells.get("descriptive", {})
    marginal = config.get("marginal_lrt", {})
    gate = config.get("affirmative_gate", {})
    reporting = config.get("reporting_candidates", {})
    resources = config.get("resources", {})
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("contract") != "focused-result-blind-null-calibration-v3"
        or config.get("seed") != 20260903
        or primary.get("role") != PRIMARY_ROLE
        or primary.get("cohorts") != list(TCGA_COHORTS)
        or primary.get("providers") != ["mutsig"]
        or descriptive.get("role") != DESCRIPTIVE_ROLE
        or descriptive.get("cohorts")
        != ["CHOL", "LAML", "PAAD", "SKCM", "UCEC"]
        or descriptive.get("providers") != ["cbase", "dig"]
        or marginal.get("pair_selection")
        != "32-disjoint-pairs-spanning-the-K500-feature-rank-axis"
        or marginal.get("sentinel_pair_count") != 32
        or marginal.get("replicates_per_cell") != 10_000
        or marginal.get("replicate_rng")
        != "sha256-cell-seed-and-sentinel-pair-index-v1"
        or marginal.get("fit_kernel") != FIT_KERNEL_CONTRACT
        or marginal.get("replicate_chunk_rule") != REPLICATE_CHUNK_RULE
        or marginal.get("alphas") != [0.01, 0.05]
        or gate.get("provider") != "mutsig"
        or gate.get("endpoint_unit") != "cohort-sentinel-pair-alpha"
        or gate.get("method")
        != (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        )
        or gate.get("familywise_error") != 0.05
        or gate.get("endpoint_count")
        != len(TCGA_COHORTS) * int(marginal.get("sentinel_pair_count", 0)) * 2
        or gate.get("acceptance_upper_bounds")
        != {"0.01": 0.02, "0.05": 0.07}
        or reporting.get("test") != "chi-square-one-df-profile-lrt"
        or reporting.get("primary_adjustment") != "benjamini-yekutieli"
        or reporting.get("primary_q_threshold") != 0.01
        or reporting.get("sensitivity_adjustment") != "benjamini-hochberg"
        or reporting.get("sensitivity_q_threshold") != 0.01
        or reporting.get("thresholds_selected_from_observed_pairs") is not False
        or reporting.get("interpretation")
        != "finite-scenario-stress-not-formal-uniform-FDR-proof"
        or resources
        != {
            "observed_logical_cpus": 14,
            "max_outer_cells": 3,
            "max_mutsig_cells": 1,
            "mutsig_fit_workers": 5,
            "descriptive_fit_workers": 1,
            "max_total_fit_workers": 7,
            "blas_threads_per_fit_worker": 1,
            "nice_increment": 10,
            "overwrite_outputs": False,
        }
    ):
        msg = "Focused calibration configuration violates its frozen v3 contract."
        raise ValueError(msg)
    protocol = _protocol_cells(config)
    if (
        len(protocol) != 42
        or sum(cell.role == PRIMARY_ROLE for cell in protocol) != 32
        or sum(cell.role == DESCRIPTIVE_ROLE for cell in protocol) != 10
    ):
        msg = "Focused calibration configuration has an invalid cell inventory."
        raise ValueError(msg)
    return config


def _frozen_resource_observation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the portable frozen half-machine receipt."""
    resources = config["resources"]
    logical_cpus = int(resources["observed_logical_cpus"])
    max_outer_cells = int(resources["max_outer_cells"])
    max_mutsig_cells = int(resources["max_mutsig_cells"])
    mutsig_fit_workers = int(resources["mutsig_fit_workers"])
    descriptive_fit_workers = int(resources["descriptive_fit_workers"])
    max_total_fit_workers = int(resources["max_total_fit_workers"])
    half_logical_cpu_limit = logical_cpus // 2
    scheduled_fit_worker_limit = (
        max_mutsig_cells * mutsig_fit_workers
        + (max_outer_cells - max_mutsig_cells) * descriptive_fit_workers
    )
    if (
        logical_cpus < 2
        or max_outer_cells < 1
        or not 1 <= max_mutsig_cells <= max_outer_cells
        or mutsig_fit_workers < 1
        or descriptive_fit_workers < 1
        or int(resources["blas_threads_per_fit_worker"]) != 1
        or scheduled_fit_worker_limit != max_total_fit_workers
        or max_total_fit_workers > half_logical_cpu_limit
    ):
        msg = (
            "Calibration runtime no longer satisfies the frozen half-machine "
            "resource contract."
        )
        raise RuntimeError(msg)
    return {
        "logical_cpus_observed": logical_cpus,
        "logical_cpu_source": "os.cpu_count()",
        "half_logical_cpu_limit": half_logical_cpu_limit,
        "scheduled_fit_worker_limit": scheduled_fit_worker_limit,
        "half_machine_limit_satisfied": True,
    }


def _preflight_runtime_resources(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed before execution if this host differs from the frozen host."""
    observation = _frozen_resource_observation(config)
    if os.cpu_count() != observation["logical_cpus_observed"]:
        msg = (
            "Calibration execution host differs from the frozen 14-logical-CPU "
            "resource contract."
        )
        raise RuntimeError(msg)
    return observation


def _fit_worker_count(config: Mapping[str, Any], provider: str) -> int:
    """Return the frozen inner fit-worker count for one provider."""
    key = "mutsig_fit_workers" if provider == "mutsig" else "descriptive_fit_workers"
    return int(config["resources"][key])


def _worker_topology(
    config: Mapping[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Return the exact worker topology used by one calibration cell."""
    resources = config["resources"]
    observation = _frozen_resource_observation(config)
    fit_workers = _fit_worker_count(config, provider)
    return {
        **observation,
        "max_outer_cells": int(resources["max_outer_cells"]),
        "max_mutsig_cells": int(resources["max_mutsig_cells"]),
        "fit_workers": fit_workers,
        "fit_execution": (
            "spawned-process-pool" if fit_workers > 1 else "task-main-process"
        ),
        "max_total_fit_workers": int(resources["max_total_fit_workers"]),
        "blas_threads_per_fit_worker": int(
            resources["blas_threads_per_fit_worker"],
        ),
        "thread_environment": dict(THREAD_ENV),
    }


def _normalized_peak_rss(
    native_value: int,
    *,
    source: str,
    semantics: str,
) -> dict[str, Any]:
    """Normalize one getrusage RSS value without changing its semantics."""
    if sys.platform == "darwin":
        native_unit = "bytes"
        multiplier = 1
    elif sys.platform.startswith("linux"):
        native_unit = "KiB"
        multiplier = 1024
    else:
        msg = f"Unsupported ru_maxrss unit convention on platform {sys.platform!r}."
        raise RuntimeError(msg)
    return {
        "bytes": native_value * multiplier,
        "native_value": native_value,
        "native_unit": native_unit,
        "platform": sys.platform,
        "source": source,
        "semantics": semantics,
    }


def _rusage_record(
    usage: resource.struct_rusage,
    *,
    children: bool,
) -> dict[str, Any]:
    """Return an honest self or terminated-child getrusage record."""
    if children:
        source = "resource.getrusage(resource.RUSAGE_CHILDREN)"
        semantics = "maximum-over-terminated-children-not-additive"
    else:
        source = "resource.getrusage(resource.RUSAGE_SELF)"
        semantics = "task-process-maximum-resident-set-size"
    return {
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "peak_rss": _normalized_peak_rss(
            int(usage.ru_maxrss),
            source=f"{source}.ru_maxrss",
            semantics=semantics,
        ),
    }


def _validate_rusage_record(
    record: object,
    *,
    children: bool,
    require_positive_rss: bool,
) -> None:
    """Validate one explicit self or child resource receipt."""
    if not isinstance(record, dict):
        msg = "Calibration resource receipt is missing."
        raise TypeError(msg)
    user = record.get("user_cpu_seconds")
    system = record.get("system_cpu_seconds")
    peak = record.get("peak_rss")
    if (
        not isinstance(user, (int, float))
        or isinstance(user, bool)
        or not math.isfinite(user)
        or user < 0
        or not isinstance(system, (int, float))
        or isinstance(system, bool)
        or not math.isfinite(system)
        or system < 0
        or not isinstance(peak, dict)
    ):
        msg = "Calibration resource receipt has invalid CPU/RSS values."
        raise ValueError(msg)
    native_value = peak.get("native_value")
    byte_count = peak.get("bytes")
    platform = peak.get("platform")
    if platform == "darwin":
        expected_unit = "bytes"
        multiplier = 1
    elif isinstance(platform, str) and platform.startswith("linux"):
        expected_unit = "KiB"
        multiplier = 1024
    else:
        msg = "Calibration resource receipt has unsupported RSS provenance."
        raise ValueError(msg)
    source_root = (
        "resource.getrusage(resource.RUSAGE_CHILDREN)"
        if children
        else "resource.getrusage(resource.RUSAGE_SELF)"
    )
    expected_semantics = (
        "maximum-over-terminated-children-not-additive"
        if children
        else "task-process-maximum-resident-set-size"
    )
    if (
        not isinstance(native_value, int)
        or isinstance(native_value, bool)
        or native_value < int(require_positive_rss)
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count != native_value * multiplier
        or peak.get("native_unit") != expected_unit
        or peak.get("source") != f"{source_root}.ru_maxrss"
        or peak.get("semantics") != expected_semantics
    ):
        msg = "Calibration resource receipt has invalid RSS provenance."
        raise ValueError(msg)


def _validate_calibration_resource_usage(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    provider: str,
) -> None:
    """Validate explicit worker topology plus separate self/child telemetry."""
    expected_topology = _worker_topology(config, provider=provider)
    if manifest.get("worker_topology") != expected_topology:
        msg = "Calibration task worker topology violates its resource contract."
        raise ValueError(msg)
    usage = manifest.get("resource_usage")
    if not isinstance(usage, dict):
        msg = "Calibration task lacks explicit resource usage."
        raise TypeError(msg)
    self_record = usage.get("self")
    child_record = usage.get("terminated_children")
    fit_workers = int(expected_topology["fit_workers"])
    _validate_rusage_record(
        self_record,
        children=False,
        require_positive_rss=True,
    )
    _validate_rusage_record(
        child_record,
        children=True,
        require_positive_rss=fit_workers > 1,
    )
    if (
        not isinstance(self_record, dict)
        or usage.get("user_cpu_seconds") != self_record.get("user_cpu_seconds")
        or usage.get("system_cpu_seconds") != self_record.get("system_cpu_seconds")
        or usage.get("peak_rss", {}).get("bytes")
        != self_record.get("peak_rss", {}).get("bytes")
    ):
        msg = "Calibration task legacy self-usage fields disagree with their receipt."
        raise ValueError(msg)


def _seed(root_seed: int, cohort: str, provider: str) -> int:
    digest = hashlib.sha256(
        _canonical_json([root_seed, "focused-calibration-v3", cohort, provider]),
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _pmf_arrays(pmf: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray]:
    items = sorted((int(key), float(value)) for key, value in pmf.items())
    if (
        not items
        or len({key for key, _ in items}) != len(items)
        or any(key < 0 for key, _ in items)
        or any(not np.isfinite(value) or value < 0 for _, value in items)
    ):
        msg = "Invalid background PMF."
        raise ValueError(msg)
    keys = np.asarray([key for key, value in items if value > 0], dtype=np.int64)
    probabilities = np.asarray(
        [value for _, value in items if value > 0],
        dtype=np.float64,
    )
    if not len(keys) or not math.isclose(
        float(probabilities.sum()),
        1.0,
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        msg = "Background PMF does not sum to one."
        raise ValueError(msg)
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return keys, cumulative


def _prepare_samplers(
    cell: Cell,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]:
    prepared = []
    for feature in cell.features:
        background = cell.pmfs[feature]
        if isinstance(background, dict):
            prepared.append((_pmf_arrays(background),))
        else:
            if len(background) != len(cell.samples):
                msg = "Sample-specific PMF axis differs from the sample axis."
                raise ValueError(msg)
            prepared.append(tuple(_pmf_arrays(pmf) for pmf in background))
    return tuple(prepared)


def _sample_background(
    rng: np.random.Generator,
    sampler: tuple[tuple[np.ndarray, np.ndarray], ...],
    sample_count: int,
) -> np.ndarray:
    if len(sampler) == 1:
        keys, cumulative = sampler[0]
        indices = np.searchsorted(cumulative, rng.random(sample_count), side="right")
        return keys[np.minimum(indices, len(keys) - 1)]
    if len(sampler) != sample_count:
        msg = "Prepared sample-specific PMF axis differs from the sample axis."
        raise ValueError(msg)
    uniforms = rng.random(sample_count)
    result = np.empty(sample_count, dtype=np.int64)
    for index, ((keys, cumulative), value) in enumerate(
        zip(sampler, uniforms, strict=True),
    ):
        position = min(
            int(np.searchsorted(cumulative, value, side="right")),
            len(keys) - 1,
        )
        result[index] = keys[position]
    return result


def _simulate_features(
    cell: Cell,
    indices: Sequence[int],
    samplers: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.empty((len(cell.samples), len(indices)), dtype=np.int64)
    for column, index in enumerate(indices):
        background = _sample_background(rng, samplers[index], len(cell.samples))
        driver = rng.binomial(1, float(cell.pi[index]), size=len(cell.samples))
        counts[:, column] = background + driver
    return counts


def _pair_genes(
    cell: Cell,
    indices: tuple[int, int],
    counts: np.ndarray,
    gene_type: type[Gene],
) -> tuple[Gene, Gene]:
    """Construct one simulated pair on the frozen sample/background axes."""
    genes = []
    for column, index in enumerate(indices):
        feature = cell.features[index]
        genes.append(
            gene_type(
                name=feature,
                samples=cell.samples,
                counts=counts[:, column],
                bmr_pmf=cell.pmfs[feature],
            ),
        )
    if len(genes) != 2:
        msg = "Calibration requires exactly two simulated genes per pair."
        raise ValueError(msg)
    return genes[0], genes[1]


def _fit_pair_with_gene_type(
    cell: Cell,
    indices: tuple[int, int],
    counts: np.ndarray,
    gene_type: type[Gene],
) -> tuple[float, bool]:
    """Return the profile LRT and whether its dependence effect is reportable."""
    genes = list(_pair_genes(cell, indices, counts, gene_type))
    for gene in genes:
        gene.estimate_pi_with_mle()
    interaction = Interaction(*genes)
    interaction.estimate_tau_with_coordinate_ascent()
    statistic = float(interaction.likelihood_ratio)
    if not np.isfinite(statistic) or statistic < -1e-10:
        msg = "Calibration pair produced an invalid profile LRT."
        raise ValueError(msg)
    status = interaction.effect_identifiability_status()
    valid_statuses = {
        core.REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
        core.REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS,
        core.REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS,
    }
    if status not in valid_statuses:
        msg = "Calibration pair produced an invalid identifiability status."
        raise ValueError(msg)
    return (
        max(statistic, 0.0),
        status == core.REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS,
    )


def _fit_pair(
    cell: Cell,
    indices: tuple[int, int],
    counts: np.ndarray,
) -> tuple[float, bool]:
    """Fit one pair through the unmodified production model classes."""
    return _fit_pair_with_gene_type(cell, indices, counts, Gene)


def _fit_pair_scalar_reference(
    cell: Cell,
    indices: tuple[int, int],
    counts: np.ndarray,
) -> tuple[float, bool]:
    """Fit one pair through the unmodified production Gene implementation."""
    return _fit_pair(cell, indices, counts)


def _effective_log_p_values(
    likelihood_ratios: np.ndarray,
    reportable: np.ndarray,
) -> np.ndarray:
    """Apply the production stable log-p and nonidentifiable-pair policy."""
    values = np.asarray(likelihood_ratios, dtype=np.float64)
    mask = np.asarray(reportable)
    if (
        values.shape != mask.shape
        or mask.dtype != np.dtype(bool)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        msg = "Calibration LRT/reportability arrays are invalid."
        raise ValueError(msg)
    result = np.zeros(values.shape, dtype=np.float64)
    result[mask] = np.log(2.0) + log_ndtr(-np.sqrt(values[mask]))
    if np.isnan(result).any() or np.isposinf(result).any() or (result > 0).any():
        msg = "Stable calibration chi-square log-survival evaluation failed."
        raise ValueError(msg)
    return result


def _effective_p_values(
    likelihood_ratios: np.ndarray,
    reportable: np.ndarray,
) -> np.ndarray:
    """Return display p-values derived from the stable production log-p path."""
    return np.exp(_effective_log_p_values(likelihood_ratios, reportable))


def _sentinel_pairs(features: Sequence[str], count: int = 32) -> np.ndarray:
    positions = {feature: index for index, feature in enumerate(features)}
    pairs = list(core.iter_tested_pairs(features))
    targets = np.linspace(0, len(pairs) - 1, count, dtype=np.int64)
    selected = []
    used: set[int] = set()
    for target in targets:
        for offset in range(len(pairs)):
            index = (int(target) + offset) % len(pairs)
            feature_a, feature_b = pairs[index]
            candidate = (positions[feature_a], positions[feature_b])
            if candidate[0] not in used and candidate[1] not in used:
                selected.append(candidate)
                used.update(candidate)
                break
        else:
            msg = "Could not select the prespecified disjoint sentinel pair axis."
            raise RuntimeError(msg)
    return np.asarray(selected, dtype=np.int32)


_PAIR_WORKER_CELL: Cell | None = None
_PAIR_WORKER_SAMPLERS: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...] | None = (
    None
)
_PAIR_WORKER_SEED: int | None = None


def _pair_seed(cell_seed: int, pair_index: int) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            [cell_seed, "sentinel-pair-independent-replicates-v1", pair_index],
        ),
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _narrow_simulation_cell(
    cell: Cell,
    sentinel_pairs: np.ndarray,
) -> tuple[Cell, np.ndarray]:
    """Return a worker payload containing only the disjoint sentinel features."""
    selected = [int(value) for value in sentinel_pairs.ravel()]
    if len(set(selected)) != len(selected):
        msg = "Calibration sentinel pairs must remain feature-disjoint."
        raise ValueError(msg)
    features = tuple(cell.features[index] for index in selected)
    narrowed = Cell(
        cohort=cell.cohort,
        provider=cell.provider,
        features=features,
        samples=cell.samples,
        pmfs={feature: cell.pmfs[feature] for feature in features},
        pi=cell.pi[np.asarray(selected, dtype=np.int64)].copy(),
        source_task_manifest=cell.source_task_manifest,
        single_gene_input=cell.single_gene_input,
    )
    local_pairs = np.arange(len(selected), dtype=np.int32).reshape(-1, 2)
    return narrowed, local_pairs


def _initialize_pair_worker(cell: Cell, cell_seed: int) -> None:
    global _PAIR_WORKER_CELL  # noqa: PLW0603
    global _PAIR_WORKER_SAMPLERS  # noqa: PLW0603
    global _PAIR_WORKER_SEED  # noqa: PLW0603
    _PAIR_WORKER_CELL = cell
    _PAIR_WORKER_SAMPLERS = _prepare_samplers(cell)
    _PAIR_WORKER_SEED = cell_seed


def _replicate_chunk_size(sample_count: int) -> int:
    """Bound the batch working set while preserving replicate order."""
    if sample_count < 1:
        msg = "Calibration batch requires a nonempty sample axis."
        raise ValueError(msg)
    return max(1, min(512, 128_000 // sample_count))


def _simulate_pair_worker(
    task: tuple[int, int, int, int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate and fit every replicate for one sentinel pair."""
    pair_index, index_a, index_b, replicates = task
    cell = _PAIR_WORKER_CELL
    samplers = _PAIR_WORKER_SAMPLERS
    cell_seed = _PAIR_WORKER_SEED
    if cell is None or samplers is None or cell_seed is None:
        msg = "Calibration pair worker was not initialized."
        raise RuntimeError(msg)
    indices = (index_a, index_b)
    rng = np.random.default_rng(_pair_seed(cell_seed, pair_index))
    likelihood_ratios = np.empty(replicates, dtype=np.float64)
    reportable = np.empty(replicates, dtype=bool)
    scalar_fallback = np.empty(replicates, dtype=bool)
    chunk_size = _replicate_chunk_size(len(cell.samples))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        pairs = [
            _pair_genes(
                cell,
                indices,
                _simulate_features(cell, indices, samplers, rng),
                Gene,
            )
            for _replicate in range(start, stop)
        ]
        fitted = fit_gene_pairs_batched(pairs)
        likelihood_ratios[start:stop] = fitted.likelihood_ratio
        reportable[start:stop] = fitted.reportable
        scalar_fallback[start:stop] = fitted.scalar_fallback
    return pair_index, likelihood_ratios, reportable, scalar_fallback


def _simulate_null_arrays(
    *,
    cell: Cell,
    sentinel_pairs: np.ndarray,
    replicates: int,
    cell_seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic pair-parallel calibration arrays."""
    if replicates < 1 or workers < 1 or workers > len(sentinel_pairs):
        msg = "Calibration pair simulation dimensions are invalid."
        raise ValueError(msg)
    narrowed, local_pairs = _narrow_simulation_cell(cell, sentinel_pairs)
    tasks = tuple(
        (pair_index, int(indices[0]), int(indices[1]), replicates)
        for pair_index, indices in enumerate(local_pairs)
    )
    if workers == 1:
        _initialize_pair_worker(narrowed, cell_seed)
        results = [_simulate_pair_worker(task) for task in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(
            processes=workers,
            initializer=_initialize_pair_worker,
            initargs=(narrowed, cell_seed),
        ) as pool:
            results = pool.map(_simulate_pair_worker, tasks, chunksize=1)
    likelihood_ratios = np.empty(
        (replicates, len(sentinel_pairs)),
        dtype=np.float64,
    )
    reportable = np.empty((replicates, len(sentinel_pairs)), dtype=bool)
    scalar_fallback = np.empty((replicates, len(sentinel_pairs)), dtype=bool)
    observed_indices = []
    for pair_index, pair_lrt, pair_reportable, pair_fallback in results:
        observed_indices.append(pair_index)
        likelihood_ratios[:, pair_index] = pair_lrt
        reportable[:, pair_index] = pair_reportable
        scalar_fallback[:, pair_index] = pair_fallback
    if observed_indices != list(range(len(sentinel_pairs))):
        msg = "Calibration pair workers returned an invalid pair axis."
        raise RuntimeError(msg)
    return likelihood_ratios, reportable, scalar_fallback


def _validate_single_gene_source(
    run_root: Path,
    cohort: str,
    provider: str,
    contract: Mapping[str, Any],
) -> tuple[dict[str, int | str], dict[str, int | str]]:
    """Validate and return the task-manifest and single-gene source records."""
    task_root = run_root / "tasks" / cohort / provider
    contract_sha256 = hashlib.sha256(_canonical_json(contract)).hexdigest()
    manifest = focused_runner._validate_completed_task(  # noqa: SLF001
        task_root,
        contract_sha256=contract_sha256,
        cohort=cohort,
        provider=provider,
        pairwise_rows=int(contract["pair_policy"]["row_count"]),
    )
    source = task_root / "single_gene_results.csv"
    expected = manifest.get("outputs", {}).get(source.name, {})
    actual = _file_record(source, relative_to=run_root)
    if (
        expected.get("path") != source.name
        or expected.get("bytes") != actual["bytes"]
        or expected.get("sha256") != actual["sha256"]
    ):
        msg = f"Single-gene source changed before calibration: {cohort}/{provider}"
        raise ValueError(msg)
    return (
        _file_record(task_root / "task_manifest.json", relative_to=run_root),
        actual,
    )


def _load_cell(run_root: Path, provider_root: Path, cohort: str, provider: str) -> Cell:
    contract = json.loads(
        (run_root / "contracts" / f"{cohort}.json").read_text(encoding="utf-8"),
    )
    paths = focused_runner._paths(provider_root, run_root)  # noqa: SLF001
    current = core.build_cohort_contract(paths, cohort, top_k=500)
    if contract != current:
        msg = f"Frozen cohort contract drifted before calibration: {cohort}"
        raise ValueError(msg)
    source_manifest, single_gene_input = _validate_single_gene_source(
        run_root,
        cohort,
        provider,
        contract,
    )
    counts, pmfs = core._load_frozen_scientific_inputs(contract, provider)  # noqa: SLF001
    single_path = run_root / "tasks" / cohort / provider / "single_gene_results.csv"
    single_raw = single_path.read_bytes()
    if (
        len(single_raw) != single_gene_input["bytes"]
        or hashlib.sha256(single_raw).hexdigest() != single_gene_input["sha256"]
    ):
        msg = (
            "Single-gene source changed while loading calibration: "
            f"{cohort}/{provider}"
        )
        raise ValueError(msg)
    single = pd.read_csv(
        BytesIO(single_raw),
        float_precision="round_trip",
    )
    features = tuple(contract["features"])
    if (
        tuple(single.columns) != core.SINGLE_GENE_RESULT_COLUMNS
        or single["Gene Name"].tolist() != list(features)
    ):
        msg = f"Single-gene feature axis differs from the contract: {cohort}/{provider}"
        raise ValueError(msg)
    pi = single["Pi"].to_numpy(dtype=np.float64)
    if not np.isfinite(pi).all() or (pi < 0).any() or (pi > 1).any():
        msg = f"Invalid fitted marginal pi values: {cohort}/{provider}"
        raise ValueError(msg)
    return Cell(
        cohort=cohort,
        provider=provider,
        features=features,
        samples=tuple(str(sample) for sample in counts.index),
        pmfs=pmfs,
        pi=pi,
        source_task_manifest=source_manifest,
        single_gene_input=single_gene_input,
    )


def _task_root(output_root: Path, cohort: str, provider: str) -> Path:
    return output_root / "tasks" / cohort / provider


def _validate_external_roots(
    run_root: Path,
    provider_root: Path,
) -> tuple[str, str]:
    """Validate the exact production completion and provider roots."""
    provider_manifest = focused_runner._load_provider_manifest(provider_root)  # noqa: SLF001
    if provider_manifest.get("cohorts") != list(TCGA_COHORTS):
        msg = "Calibration provider root does not cover the canonical 32 cohorts."
        raise ValueError(msg)
    completion_path = run_root / "completion_manifest.json"
    if not completion_path.is_file() or completion_path.is_symlink():
        msg = "Calibration production completion manifest is missing or unsafe."
        raise FileNotFoundError(msg)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_coordinates = {
        (cohort, provider) for cohort in TCGA_COHORTS for provider in core.BMRS
    }
    records = {
        (str(item.get("cohort")), str(item.get("provider"))): item.get("manifest", {})
        for item in completion.get("tasks", [])
    }
    run_manifest_path = run_root / "run_manifest.json"
    if not run_manifest_path.is_file() or run_manifest_path.is_symlink():
        msg = "Calibration production run manifest is missing or unsafe."
        raise FileNotFoundError(msg)
    run_record = completion.get("run_manifest", {})
    if (
        completion.get("schema_version") != focused_runner.SCHEMA_VERSION
        or completion.get("contract") != focused_runner.COMPLETION_CONTRACT
        or completion.get("config_sha256")
        != focused_runner._sha256(focused_runner.CONFIG_PATH)  # noqa: SLF001
        or completion.get("cohorts") != list(TCGA_COHORTS)
        or completion.get("task_count") != len(expected_coordinates)
        or set(records) != expected_coordinates
        or run_record.get("path") != "run_manifest.json"
        or run_record.get("bytes") != run_manifest_path.stat().st_size
        or run_record.get("sha256") != _sha256(run_manifest_path)
    ):
        msg = "Calibration requires the complete validated 32-by-3 production grid."
        raise ValueError(msg)
    for (cohort, provider), record in records.items():
        path = run_root / "tasks" / cohort / provider / "task_manifest.json"
        if (
            record.get("path") != f"tasks/{cohort}/{provider}/task_manifest.json"
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            msg = f"Production task receipt changed: {cohort}/{provider}"
            raise ValueError(msg)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    provider_record = run_manifest.get("provider_manifest", {})
    provider_path = provider_root / "provider_manifest.json"
    if not provider_path.is_file() or provider_path.is_symlink():
        msg = "Calibration provider manifest is missing or unsafe."
        raise FileNotFoundError(msg)
    if (
        run_manifest.get("contract") != focused_runner.RUN_CONTRACT
        or provider_record.get("path") != "provider_manifest.json"
        or provider_record.get("bytes") != provider_path.stat().st_size
        or provider_record.get("sha256") != _sha256(provider_path)
    ):
        msg = "Production run is bound to a different provider root."
        raise ValueError(msg)
    return _sha256(completion_path), _sha256(provider_path)


def _run_manifest_payload(
    *,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_external_roots(run_root, provider_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RUN_CONTRACT,
        "config": _file_record(CONFIG_PATH, relative_to=CONFIG_PATH.parent.parent),
        "run_completion": _file_record(
            run_root / "completion_manifest.json",
            relative_to=run_root,
        ),
        "provider_manifest": _file_record(
            provider_root / "provider_manifest.json",
            relative_to=provider_root,
        ),
        "cells": [
            {
                "cohort": cell.cohort,
                "provider": cell.provider,
                "role": cell.role,
            }
            for cell in _protocol_cells(config)
        ],
        "resource_contract": dict(config["resources"]),
        "runtime_resource_observation": _frozen_resource_observation(config),
        "thread_environment": dict(THREAD_ENV),
        "result_blindness": _result_blindness_receipt(),
    }


def _ensure_run_root(
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _load_config()
    _preflight_runtime_resources(config)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tasks").mkdir(exist_ok=True)
    manifest = _run_manifest_payload(
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    content = _canonical_json(manifest) + b"\n"
    path = output_root / RUN_MANIFEST_NAME
    if path.exists():
        if path.read_bytes() != content:
            msg = "Calibration run root is bound to different inputs."
            raise ValueError(msg)
    else:
        _write_atomic(path, content)
    return config


def _validate_run_manifest(
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    expected = _run_manifest_payload(
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    path = output_root / RUN_MANIFEST_NAME
    if path.read_bytes() != _canonical_json(expected) + b"\n":
        msg = "Calibration run manifest is invalid or its external inputs changed."
        raise ValueError(msg)
    return (
        str(expected["run_completion"]["sha256"]),
        str(expected["provider_manifest"]["sha256"]),
    )


def _validate_task(  # noqa: PLR0913
    task_root: Path,
    config: Mapping[str, Any],
    *,
    cohort: str,
    provider: str,
    role: str,
    run_completion_sha256: str,
    run_root: Path,
) -> dict[str, Any]:
    manifest = json.loads((task_root / TASK_MANIFEST_NAME).read_text(encoding="utf-8"))
    data = task_root / TASK_DATA_NAME
    marginal = config["marginal_lrt"]
    contract = json.loads(
        (run_root / "contracts" / f"{cohort}.json").read_text(encoding="utf-8"),
    )
    expected_worker_topology = _worker_topology(config, provider=provider)
    sample_count = contract.get("samples", {}).get("count")
    fit_kernel = manifest.get("fit_kernel")
    fallback_count = (
        fit_kernel.get("scalar_fallback_count")
        if isinstance(fit_kernel, dict)
        else None
    )
    expected_fit_kernel = {
        "contract": marginal["fit_kernel"],
        "replicate_chunk_rule": marginal["replicate_chunk_rule"],
        "replicate_chunk_size": (
            _replicate_chunk_size(sample_count)
            if isinstance(sample_count, int) and not isinstance(sample_count, bool)
            else None
        ),
        "scalar_fallback_count": fallback_count,
    }
    source_manifest, single_gene_input = _validate_single_gene_source(
        run_root,
        cohort,
        provider,
        contract,
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != TASK_CONTRACT
        or manifest.get("cohort") != cohort
        or manifest.get("provider") != provider
        or manifest.get("role") != role
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("run_completion_sha256") != run_completion_sha256
        or manifest.get("seed") != _seed(int(config["seed"]), cohort, provider)
        or manifest.get("marginal_replicates")
        != int(marginal["replicates_per_cell"])
        or manifest.get("sentinel_pair_count")
        != int(marginal["sentinel_pair_count"])
        or manifest.get("alphas") != marginal["alphas"]
        or manifest.get("replicate_rng") != marginal["replicate_rng"]
        or not isinstance(fit_kernel, dict)
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or not isinstance(fallback_count, int)
        or isinstance(fallback_count, bool)
        or not 0
        <= fallback_count
        <= int(marginal["replicates_per_cell"])
        * int(marginal["sentinel_pair_count"])
        or fit_kernel != expected_fit_kernel
        or manifest.get("worker_topology") != expected_worker_topology
        or manifest.get("source_task_manifest") != source_manifest
        or manifest.get("single_gene_input") != single_gene_input
        or manifest.get("output", {}).get("path") != TASK_DATA_NAME
        or manifest.get("output", {}).get("bytes") != data.stat().st_size
        or manifest.get("output", {}).get("sha256") != _sha256(data)
        or {path.name for path in task_root.iterdir()}
        != {TASK_DATA_NAME, TASK_MANIFEST_NAME}
    ):
        msg = f"Calibration task failed manifest validation: {task_root}"
        raise ValueError(msg)
    core._validate_task_resource_usage(manifest, task_root)  # noqa: SLF001
    _validate_calibration_resource_usage(
        manifest,
        config,
        provider=provider,
    )
    replicates = int(marginal["replicates_per_cell"])
    pair_count = int(marginal["sentinel_pair_count"])
    expected_pairs = _sentinel_pairs(contract["features"], pair_count)
    with np.load(data, allow_pickle=False) as arrays:
        if (
            set(arrays.files)
            != {
                "marginal_lrt",
                "marginal_reportable",
                "scalar_fallback",
                "sentinel_pairs",
            }
            or arrays["marginal_lrt"].shape != (replicates, pair_count)
            or arrays["marginal_reportable"].shape != (replicates, pair_count)
            or arrays["marginal_reportable"].dtype != np.dtype(bool)
            or arrays["scalar_fallback"].shape != (replicates, pair_count)
            or arrays["scalar_fallback"].dtype != np.dtype(bool)
            or arrays["sentinel_pairs"].shape != (pair_count, 2)
            or not np.array_equal(arrays["sentinel_pairs"], expected_pairs)
            or not np.isfinite(arrays["marginal_lrt"]).all()
            or (arrays["marginal_lrt"] < 0).any()
            or manifest.get("marginal_reportable_count")
            != int(arrays["marginal_reportable"].sum())
            or fallback_count != int(arrays["scalar_fallback"].sum())
        ):
            msg = f"Calibration task arrays failed validation: {task_root}"
            raise ValueError(msg)
    return manifest


def _find_protocol_cell(
    config: Mapping[str, Any],
    cohort: str,
    provider: str,
) -> ProtocolCell:
    matches = [
        cell
        for cell in _protocol_cells(config)
        if (cell.cohort, cell.provider) == (cohort, provider)
    ]
    if len(matches) != 1:
        msg = "Internal calibration cell is outside the frozen protocol."
        raise ValueError(msg)
    return matches[0]


def run_task(  # noqa: PLR0913
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    cohort: str,
    provider: str,
    nice_increment: int,
) -> str:
    """Run and publish one prespecified cohort/provider calibration cell."""
    config = _load_config()
    protocol_cell = _find_protocol_cell(config, cohort, provider)
    run_completion_sha256, _ = _validate_run_manifest(
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
    )
    task_root = _task_root(output_root, cohort, provider)
    if task_root.exists():
        _validate_task(
            task_root,
            config,
            cohort=cohort,
            provider=provider,
            role=protocol_cell.role,
            run_completion_sha256=run_completion_sha256,
            run_root=run_root,
        )
        return "already-complete"
    _preflight_runtime_resources(config)
    if nice_increment:
        os.nice(nice_increment)
    os.environ.update(THREAD_ENV)
    started = time.monotonic()
    cell = _load_cell(run_root, provider_root, cohort, provider)
    pair_count = int(config["marginal_lrt"]["sentinel_pair_count"])
    sentinel_pairs = _sentinel_pairs(cell.features, pair_count)
    marginal_replicates = int(config["marginal_lrt"]["replicates_per_cell"])
    fit_workers = _fit_worker_count(config, provider)
    with threadpool_limits(
        limits=int(config["resources"]["blas_threads_per_fit_worker"]),
    ):
        (
            marginal_lrt,
            marginal_reportable,
            scalar_fallback,
        ) = _simulate_null_arrays(
            cell=cell,
            sentinel_pairs=sentinel_pairs,
            replicates=marginal_replicates,
            cell_seed=_seed(int(config["seed"]), cohort, provider),
            workers=fit_workers,
        )

    task_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{provider}.", dir=task_root.parent))
    data_path = staging / TASK_DATA_NAME
    with data_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            marginal_lrt=marginal_lrt,
            marginal_reportable=marginal_reportable,
            scalar_fallback=scalar_fallback,
            sentinel_pairs=sentinel_pairs,
        )
        handle.flush()
        os.fsync(handle.fileno())
    self_usage = _rusage_record(
        resource.getrusage(resource.RUSAGE_SELF),
        children=False,
    )
    child_usage = _rusage_record(
        resource.getrusage(resource.RUSAGE_CHILDREN),
        children=True,
    )
    resource_usage = core._task_resource_usage(started)  # noqa: SLF001
    resource_usage.update(
        {
            "user_cpu_seconds": self_usage["user_cpu_seconds"],
            "system_cpu_seconds": self_usage["system_cpu_seconds"],
            "self": self_usage,
            "terminated_children": child_usage,
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": TASK_CONTRACT,
        "cohort": cohort,
        "provider": provider,
        "role": protocol_cell.role,
        "config_sha256": _sha256(CONFIG_PATH),
        "run_completion_sha256": run_completion_sha256,
        "seed": _seed(int(config["seed"]), cohort, provider),
        "marginal_replicates": marginal_replicates,
        "sentinel_pair_count": len(sentinel_pairs),
        "alphas": config["marginal_lrt"]["alphas"],
        "replicate_rng": config["marginal_lrt"]["replicate_rng"],
        "fit_kernel": {
            "contract": config["marginal_lrt"]["fit_kernel"],
            "replicate_chunk_rule": config["marginal_lrt"][
                "replicate_chunk_rule"
            ],
            "replicate_chunk_size": _replicate_chunk_size(len(cell.samples)),
            "scalar_fallback_count": int(scalar_fallback.sum()),
        },
        "worker_topology": _worker_topology(config, provider=provider),
        "marginal_reportable_count": int(marginal_reportable.sum()),
        "source_task_manifest": dict(cell.source_task_manifest),
        "single_gene_input": dict(cell.single_gene_input),
        "resource_usage": resource_usage,
        "output": _file_record(data_path, relative_to=staging),
    }
    _write_atomic(staging / TASK_MANIFEST_NAME, _canonical_json(manifest) + b"\n")
    if task_root.exists():
        msg = f"Calibration task output appeared during execution: {task_root}"
        raise FileExistsError(msg)
    staging.replace(task_root)
    _validate_task(
        task_root,
        config,
        cohort=cohort,
        provider=provider,
        role=protocol_cell.role,
        run_completion_sha256=run_completion_sha256,
        run_root=run_root,
    )
    return "complete"


def _task_command(  # noqa: PLR0913
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    cohort: str,
    provider: str,
    nice_increment: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "analysis.calibrate_tcga_revision_focused",
        "--run-root",
        run_root.as_posix(),
        "--provider-root",
        provider_root.as_posix(),
        "--output-root",
        output_root.as_posix(),
        "--internal-cohort",
        cohort,
        "--internal-provider",
        provider,
        "--nice",
        str(nice_increment),
    ]


def _run_protocol(  # noqa: PLR0913
    tasks: Sequence[ProtocolCell],
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    jobs: int,
    max_mutsig_cells: int,
    mutsig_fit_workers: int,
    descriptive_fit_workers: int,
    max_total_fit_workers: int,
    nice_increment: int,
) -> None:
    """Run mixed protocol cells under total and MutSig concurrency caps."""
    if not tasks:
        return
    maximum_scheduled_fit_workers = (
        max_mutsig_cells * mutsig_fit_workers
        + (jobs - max_mutsig_cells) * descriptive_fit_workers
    )
    if (
        jobs < 1
        or max_mutsig_cells < 1
        or max_mutsig_cells > jobs
        or mutsig_fit_workers < 1
        or descriptive_fit_workers < 1
        or maximum_scheduled_fit_workers > max_total_fit_workers
    ):
        msg = "Calibration scheduler concurrency limits are invalid."
        raise ValueError(msg)
    environment = os.environ.copy()
    environment.update(THREAD_ENV)

    def invoke(cell: ProtocolCell) -> None:
        subprocess.run(
            _task_command(
                run_root=run_root,
                provider_root=provider_root,
                output_root=output_root,
                cohort=cell.cohort,
                provider=cell.provider,
                nice_increment=nice_increment,
            ),
            check=True,
            env=environment,
        )

    mutsig_remaining = iter(cell for cell in tasks if cell.provider == "mutsig")
    descriptive_remaining = iter(
        cell for cell in tasks if cell.provider != "mutsig"
    )
    next_mutsig = next(mutsig_remaining, None)
    next_descriptive = next(descriptive_remaining, None)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures: dict[Any, ProtocolCell] = {}

        def fill_slots() -> None:
            nonlocal next_descriptive, next_mutsig
            active_mutsig = sum(
                cell.provider == "mutsig" for cell in futures.values()
            )
            while (
                len(futures) < jobs
                and next_mutsig is not None
                and active_mutsig < max_mutsig_cells
            ):
                cell = next_mutsig
                next_mutsig = next(mutsig_remaining, None)
                futures[executor.submit(invoke, cell)] = cell
                active_mutsig += 1
            while len(futures) < jobs and next_descriptive is not None:
                cell = next_descriptive
                next_descriptive = next(descriptive_remaining, None)
                futures[executor.submit(invoke, cell)] = cell

        fill_slots()
        while futures:
            future = next(as_completed(futures))
            cell = futures.pop(future)
            try:
                future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            print(f"calibration complete {cell.cohort}/{cell.provider}", flush=True)
            fill_slots()


def run_all(
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    jobs: int,
    nice_increment: int,
) -> None:
    """Run all frozen cells with at most one MutSig worker and three total jobs."""
    config = _ensure_run_root(run_root, provider_root, output_root)
    resources = config["resources"]
    if jobs < 1 or jobs > int(resources["max_outer_cells"]):
        msg = "Calibration --jobs exceeds the frozen resource contract."
        raise ValueError(msg)
    if nice_increment != int(resources["nice_increment"]):
        msg = "Calibration --nice differs from the frozen resource contract."
        raise ValueError(msg)
    protocol = _protocol_cells(config)
    _run_protocol(
        protocol,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        jobs=jobs,
        max_mutsig_cells=int(resources["max_mutsig_cells"]),
        mutsig_fit_workers=int(resources["mutsig_fit_workers"]),
        descriptive_fit_workers=int(resources["descriptive_fit_workers"]),
        max_total_fit_workers=int(resources["max_total_fit_workers"]),
        nice_increment=nice_increment,
    )


def _clopper_pearson_upper_bound(
    *,
    successes: int,
    trials: int,
    endpoint_error: float,
) -> float:
    """Return one exact one-sided Clopper-Pearson binomial upper bound."""
    if (
        trials < 1
        or successes < 0
        or successes > trials
        or not 0 < endpoint_error < 1
    ):
        msg = "Exact-binomial gate parameters are invalid."
        raise ValueError(msg)
    if successes == trials:
        return 1.0
    upper = float(
        beta.ppf(
            1.0 - endpoint_error,
            successes + 1,
            trials - successes,
        ),
    )
    if not np.isfinite(upper) or not successes / trials <= upper <= 1:
        msg = "Exact-binomial upper bound is invalid."
        raise RuntimeError(msg)
    return upper


def _gate_fields(
    *,
    successes: int,
    trials: int,
    alpha: float,
    config: Mapping[str, Any],
) -> dict[str, float | int | str]:
    """Evaluate one primary MutSig calibration endpoint affirmatively."""
    if successes < 0 or successes > trials or trials < 1:
        msg = "Calibration endpoint counts are invalid."
        raise ValueError(msg)
    gate = config["affirmative_gate"]
    familywise_error = float(gate["familywise_error"])
    endpoint_count = int(gate["endpoint_count"])
    endpoint_error = familywise_error / endpoint_count
    upper = _clopper_pearson_upper_bound(
        successes=successes,
        trials=trials,
        endpoint_error=endpoint_error,
    )
    acceptance = float(gate["acceptance_upper_bounds"][f"{alpha:.2f}"])
    return {
        "exact_binomial_familywise_error": familywise_error,
        "exact_binomial_endpoint_count": endpoint_count,
        "bonferroni_endpoint_error": endpoint_error,
        "clopper_pearson_upper_bound": upper,
        "acceptance_upper_bound": acceptance,
        "endpoint_gate_pass": (
            ENDPOINT_ACCEPTED if upper <= acceptance else ENDPOINT_REJECTED
        ),
    }


def _summary_frame(
    *,
    output_root: Path,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    run_completion_sha256, _ = _validate_run_manifest(
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
    )
    rows: list[dict[str, Any]] = []
    task_manifests = []
    nonreportable_fit_count = 0
    for cell in _protocol_cells(config):
        task_root = _task_root(output_root, cell.cohort, cell.provider)
        manifest = _validate_task(
            task_root,
            config,
            cohort=cell.cohort,
            provider=cell.provider,
            role=cell.role,
            run_completion_sha256=run_completion_sha256,
            run_root=run_root,
        )
        task_manifest_record = _file_record(
            task_root / TASK_MANIFEST_NAME,
            relative_to=output_root,
        )
        task_manifest_record.update(
            {
                "cohort": cell.cohort,
                "provider": cell.provider,
                "role": cell.role,
                "fit_kernel": manifest["fit_kernel"],
                "worker_topology": manifest["worker_topology"],
                "resource_usage": manifest["resource_usage"],
            },
        )
        task_manifests.append(task_manifest_record)
        with np.load(task_root / TASK_DATA_NAME, allow_pickle=False) as arrays:
            reportable = arrays["marginal_reportable"]
            log_p_values = _effective_log_p_values(
                arrays["marginal_lrt"],
                reportable,
            )
            nonreportable_fit_count += int(reportable.size - reportable.sum())
            for pair_index in range(log_p_values.shape[1]):
                pair_log_p_values = log_p_values[:, pair_index]
                pair_reportable = reportable[:, pair_index]
                reportable_trials = int(pair_reportable.sum())
                nonreportable_trials = int(
                    pair_reportable.size - reportable_trials,
                )
                for raw_alpha in config["marginal_lrt"]["alphas"]:
                    alpha = float(raw_alpha)
                    successes = int((pair_log_p_values <= np.log(alpha)).sum())
                    trials = int(pair_log_p_values.size)
                    row: dict[str, Any] = {
                        "cohort": cell.cohort,
                        "provider": cell.provider,
                        "role": cell.role,
                        "screen": MARGINAL_SCREEN,
                        "sentinel_pair_index": pair_index,
                        "threshold": alpha,
                        "events": successes,
                        "trials": trials,
                        "rate": successes / trials,
                        "reportable_trials": reportable_trials,
                        "nonreportable_trials": nonreportable_trials,
                        "gate_endpoint": cell.role == PRIMARY_ROLE,
                        "exact_binomial_familywise_error": "",
                        "exact_binomial_endpoint_count": "",
                        "bonferroni_endpoint_error": "",
                        "clopper_pearson_upper_bound": "",
                        "acceptance_upper_bound": "",
                        "endpoint_gate_pass": GATE_NOT_APPLICABLE,
                    }
                    if cell.role == PRIMARY_ROLE:
                        row.update(
                            _gate_fields(
                                successes=successes,
                                trials=trials,
                                alpha=alpha,
                                config=config,
                            ),
                        )
                    rows.append(row)
    return (
        pd.DataFrame(rows, columns=SUMMARY_COLUMNS),
        task_manifests,
        nonreportable_fit_count,
    )


def _summary_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _validated_gate_rows(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate the pair-resolved endpoint inventory and return gate rows."""
    if tuple(frame.columns) != SUMMARY_COLUMNS:
        msg = "Calibration summary table columns are invalid."
        raise ValueError(msg)
    pair_count = int(config["marginal_lrt"]["sentinel_pair_count"])
    replicates = int(config["marginal_lrt"]["replicates_per_cell"])
    alphas = tuple(float(value) for value in config["marginal_lrt"]["alphas"])
    expected = {
        (cell.cohort, cell.provider, cell.role, pair_index, alpha)
        for cell in _protocol_cells(config)
        for pair_index in range(pair_count)
        for alpha in alphas
    }
    observed = list(
        zip(
            frame["cohort"].astype(str),
            frame["provider"].astype(str),
            frame["role"].astype(str),
            frame["sentinel_pair_index"].astype(int),
            frame["threshold"].astype(float),
            strict=True,
        ),
    )
    primary_mask = frame["role"].eq(PRIMARY_ROLE)
    descriptive_mask = frame["role"].eq(DESCRIPTIVE_ROLE)
    descriptive_blank_gate_fields = (
        "exact_binomial_familywise_error",
        "exact_binomial_endpoint_count",
        "bonferroni_endpoint_error",
        "clopper_pearson_upper_bound",
        "acceptance_upper_bound",
    )
    if (
        len(observed) != len(expected)
        or set(observed) != expected
        or not frame["screen"].eq(MARGINAL_SCREEN).all()
        or not frame["trials"].eq(replicates).all()
        or not (frame["reportable_trials"] + frame["nonreportable_trials"])
        .eq(frame["trials"])
        .all()
        or not frame.loc[primary_mask, "gate_endpoint"].all()
        or frame.loc[descriptive_mask, "gate_endpoint"].any()
        or not frame.loc[
            descriptive_mask,
            "endpoint_gate_pass",
        ].eq(GATE_NOT_APPLICABLE).all()
        or not frame.loc[
            descriptive_mask,
            descriptive_blank_gate_fields,
        ].eq("").to_numpy().all()
    ):
        msg = "Calibration pair-resolved endpoint inventory is invalid."
        raise ValueError(msg)
    gate_rows = frame.loc[primary_mask]
    gate = config["affirmative_gate"]
    endpoint_count = int(gate["endpoint_count"])
    endpoint_error = float(gate["familywise_error"]) / endpoint_count
    gate_field_names = (
        "exact_binomial_familywise_error",
        "exact_binomial_endpoint_count",
        "bonferroni_endpoint_error",
        "clopper_pearson_upper_bound",
        "acceptance_upper_bound",
        "endpoint_gate_pass",
    )
    if (
        len(gate_rows) != endpoint_count
        or not gate_rows["endpoint_gate_pass"].isin(
            [ENDPOINT_ACCEPTED, ENDPOINT_REJECTED],
        ).all()
        or not gate_rows["exact_binomial_familywise_error"].eq(
            float(gate["familywise_error"]),
        ).all()
        or not gate_rows["exact_binomial_endpoint_count"].eq(endpoint_count).all()
        or not gate_rows["bonferroni_endpoint_error"].eq(endpoint_error).all()
        or not gate_rows.apply(
            lambda row: row["acceptance_upper_bound"]
            == float(gate["acceptance_upper_bounds"][f"{row['threshold']:.2f}"]),
            axis=1,
        ).all()
    ):
        msg = "Calibration exact-binomial gate endpoint inventory is invalid."
        raise ValueError(msg)
    for row in gate_rows.to_dict(orient="records"):
        expected_fields = _gate_fields(
            successes=int(row["events"]),
            trials=int(row["trials"]),
            alpha=float(row["threshold"]),
            config=config,
        )
        if any(row[name] != expected_fields[name] for name in gate_field_names):
            msg = "Calibration exact-binomial endpoint differs from its counts."
            raise ValueError(msg)
    return gate_rows


def _summary_payload(  # noqa: PLR0913
    *,
    output_root: Path,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    task_manifests: list[dict[str, Any]],
    nonreportable_fit_count: int,
) -> dict[str, Any]:
    run_completion_sha256, provider_manifest_sha256 = _validate_run_manifest(
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
    )
    gate_rows = _validated_gate_rows(frame, config)
    passed = int(gate_rows["endpoint_gate_pass"].eq(ENDPOINT_ACCEPTED).sum())
    gate = config["affirmative_gate"]
    endpoint_error = float(gate["familywise_error"]) / int(gate["endpoint_count"])
    reporting = config["reporting_candidates"]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": SUMMARY_CONTRACT,
        "config_sha256": _sha256(CONFIG_PATH),
        "cell_count": len(_protocol_cells(config)),
        "primary_gate_cell_count": sum(
            cell.role == PRIMARY_ROLE for cell in _protocol_cells(config)
        ),
        "descriptive_cell_count": sum(
            cell.role == DESCRIPTIVE_ROLE for cell in _protocol_cells(config)
        ),
        "marginal_endpoint_count": len(frame),
        "primary_gate_endpoint_count": len(gate_rows),
        "primary_gate_passed_endpoint_count": passed,
        "overall_gate_pass": passed == len(gate_rows),
        "gate_provider": gate["provider"],
        "gate_endpoint_unit": gate["endpoint_unit"],
        "gate_method": gate["method"],
        "exact_binomial_familywise_error": gate["familywise_error"],
        "exact_binomial_endpoint_count": gate["endpoint_count"],
        "bonferroni_endpoint_error": endpoint_error,
        "clopper_pearson_confidence_level": 1.0 - endpoint_error,
        "acceptance_upper_bounds": gate["acceptance_upper_bounds"],
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "nonreportable_fit_count": nonreportable_fit_count,
        "primary_adjustment": reporting["primary_adjustment"],
        "primary_q_candidate": reporting["primary_q_threshold"],
        "sensitivity_adjustment": reporting["sensitivity_adjustment"],
        "sensitivity_q_candidate": reporting["sensitivity_q_threshold"],
        "interpretation": reporting["interpretation"],
        "reporting_rule_selected": False,
        "resource_contract": dict(config["resources"]),
        "runtime_resource_observation": _frozen_resource_observation(config),
        "thread_environment": dict(THREAD_ENV),
        "resource_usage_interpretation": {
            "self_and_terminated_child_cpu_seconds_reported_separately": True,
            "terminated_child_peak_rss": (
                "maximum-over-terminated-children-not-additive"
            ),
        },
        "result_blindness": _result_blindness_receipt(),
        "run_completion_sha256": run_completion_sha256,
        "provider_manifest_sha256": provider_manifest_sha256,
        "run_manifest": _file_record(
            output_root / RUN_MANIFEST_NAME,
            relative_to=output_root,
        ),
        "task_manifests": task_manifests,
        "table": _file_record(
            output_root / SUMMARY_TABLE_NAME,
            relative_to=output_root,
        ),
    }


def summarize(
    *,
    output_root: Path,
    run_root: Path,
    provider_root: Path,
) -> Path:
    """Summarize the affirmative gate without selecting a reporting rule."""
    table_path = output_root / SUMMARY_TABLE_NAME
    summary_path = output_root / SUMMARY_NAME
    if table_path.exists() or summary_path.exists():
        msg = "Refusing to overwrite calibration summary artifacts."
        raise FileExistsError(msg)
    config = _load_config()
    frame, task_manifests, nonreportable_fit_count = _summary_frame(
        output_root=output_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    _write_atomic(table_path, _summary_csv_bytes(frame))
    summary = _summary_payload(
        output_root=output_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
        frame=frame,
        task_manifests=task_manifests,
        nonreportable_fit_count=nonreportable_fit_count,
    )
    _write_atomic(summary_path, _canonical_json(summary) + b"\n")
    validate_summary(
        output_root,
        run_root=run_root,
        provider_root=provider_root,
    )
    return summary_path


def validate_summary(
    output_root: Path,
    *,
    run_root: Path,
    provider_root: Path,
) -> dict[str, Any]:
    """Recompute and validate the complete calibration tree and gate decision."""
    config = _load_config()
    expected_frame, task_manifests, nonreportable_fit_count = _summary_frame(
        output_root=output_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    table_path = output_root / SUMMARY_TABLE_NAME
    if table_path.read_bytes() != _summary_csv_bytes(expected_frame):
        msg = "Calibration summary table differs from recomputed task evidence."
        raise ValueError(msg)
    expected = _summary_payload(
        output_root=output_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
        frame=expected_frame,
        task_manifests=task_manifests,
        nonreportable_fit_count=nonreportable_fit_count,
    )
    summary = json.loads(
        (output_root / SUMMARY_NAME).read_text(encoding="utf-8"),
    )
    if summary != expected or not isinstance(summary.get("overall_gate_pass"), bool):
        msg = "Calibration summary or affirmative gate decision is invalid."
        raise ValueError(msg)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--validate-summary", action="store_true")
    parser.add_argument("--internal-cohort")
    parser.add_argument("--internal-provider", choices=core.BMRS)
    return parser


def main() -> None:
    """Run, resume, summarize, or validate focused calibration."""
    args = _parser().parse_args()
    output_root = args.output_root.absolute()
    run_root = args.run_root.resolve()
    provider_root = args.provider_root.resolve()
    if args.summarize and args.validate_summary:
        msg = "Choose either --summarize or --validate-summary."
        raise ValueError(msg)
    if args.summarize:
        print(
            summarize(
                output_root=output_root,
                run_root=run_root,
                provider_root=provider_root,
            ),
        )
        return
    if args.validate_summary:
        validate_summary(
            output_root,
            run_root=run_root,
            provider_root=provider_root,
        )
        print(output_root / SUMMARY_NAME)
        return
    if (args.internal_cohort is None) != (args.internal_provider is None):
        msg = "Internal cohort and provider must be supplied together."
        raise ValueError(msg)
    if args.internal_cohort is not None:
        print(
            run_task(
                run_root=run_root,
                provider_root=provider_root,
                output_root=output_root,
                cohort=args.internal_cohort,
                provider=args.internal_provider,
                nice_increment=args.nice,
            ),
        )
        return
    run_all(
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        jobs=args.jobs,
        nice_increment=args.nice,
    )


if __name__ == "__main__":
    main()

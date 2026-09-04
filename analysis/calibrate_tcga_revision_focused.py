"""Run the result-blind affirmative calibration for the focused K=500 analysis.

The primary gate covers MutSig in all 32 cohorts.  CBaSE and DIG are evaluated in
five predeclared cohorts as descriptive sensitivity checks only.  Every cell is
generated from fitted independent DIALECT marginals and measures rejection on 64
disjoint K=500-axis pairs.  This is finite-scenario evidence, not a proof of uniform
p-value or false-discovery-rate control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from scipy.stats import chi2

from analysis import run_tcga_revision_focused as focused_runner
from analysis import run_tcga_revision_k500 as core
from dialect.data.tcga import TCGA_COHORTS
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
CONFIG_PATH: Final = Path(__file__).with_name("tcga_revision_calibration_config.json")
RUN_CONTRACT: Final = "focused-parametric-null-calibration-run-v2"
TASK_CONTRACT: Final = "focused-parametric-null-calibration-cell-v2"
SUMMARY_CONTRACT: Final = "focused-parametric-null-calibration-summary-v2"
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
    "threshold",
    "events",
    "trials",
    "rate",
    "reportable_trials",
    "nonreportable_trials",
    "gate_endpoint",
    "hoeffding_familywise_error",
    "hoeffding_endpoint_count",
    "hoeffding_margin",
    "hoeffding_upper_bound",
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
        or config.get("contract") != "focused-result-blind-null-calibration-v2"
        or primary.get("role") != PRIMARY_ROLE
        or primary.get("cohorts") != list(TCGA_COHORTS)
        or primary.get("providers") != ["mutsig"]
        or descriptive.get("role") != DESCRIPTIVE_ROLE
        or descriptive.get("cohorts")
        != ["CHOL", "LAML", "PAAD", "SKCM", "UCEC"]
        or descriptive.get("providers") != ["cbase", "dig"]
        or marginal.get("pair_selection")
        != "64-disjoint-pairs-spanning-the-K500-feature-rank-axis"
        or marginal.get("sentinel_pair_count") != 64
        or marginal.get("replicates_per_cell") != 1000
        or marginal.get("alphas") != [0.01, 0.05]
        or gate.get("provider") != "mutsig"
        or gate.get("method")
        != "simultaneous-one-sided-hoeffding-upper-bound"
        or gate.get("familywise_error") != 0.05
        or gate.get("endpoint_count") != len(TCGA_COHORTS) * 2
        or gate.get("acceptance_upper_bounds")
        != {"0.01": 0.02, "0.05": 0.06}
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
            "max_jobs": 3,
            "max_mutsig_jobs": 1,
            "threads_per_job": 1,
            "nice_increment": 10,
            "overwrite_outputs": False,
        }
    ):
        msg = "Focused calibration configuration violates its frozen v2 contract."
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


def _seed(root_seed: int, cohort: str, provider: str) -> int:
    digest = hashlib.sha256(
        _canonical_json([root_seed, "focused-calibration-v2", cohort, provider]),
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


def _fit_pair(
    cell: Cell,
    indices: tuple[int, int],
    counts: np.ndarray,
) -> tuple[float, bool]:
    """Return the profile LRT and whether its dependence effect is reportable."""
    genes = []
    for column, index in enumerate(indices):
        feature = cell.features[index]
        gene = Gene(
            name=feature,
            samples=cell.samples,
            counts=counts[:, column],
            bmr_pmf=cell.pmfs[feature],
        )
        gene.estimate_pi_with_mle()
        genes.append(gene)
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


def _effective_p_values(
    likelihood_ratios: np.ndarray,
    reportable: np.ndarray,
) -> np.ndarray:
    """Apply the production nonidentifiable-pair p=1 policy."""
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
    result = np.ones(values.shape, dtype=np.float64)
    result[mask] = chi2.sf(values[mask], df=1)
    return result


def _sentinel_pairs(features: Sequence[str], count: int = 64) -> np.ndarray:
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
        "observed_pair_statistics_opened": False,
    }


def _ensure_run_root(
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _load_config()
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
    replicates = int(marginal["replicates_per_cell"])
    pair_count = int(marginal["sentinel_pair_count"])
    expected_pairs = _sentinel_pairs(contract["features"], pair_count)
    with np.load(data, allow_pickle=False) as arrays:
        if (
            set(arrays.files)
            != {"marginal_lrt", "marginal_reportable", "sentinel_pairs"}
            or arrays["marginal_lrt"].shape != (replicates, pair_count)
            or arrays["marginal_reportable"].shape != (replicates, pair_count)
            or arrays["marginal_reportable"].dtype != np.dtype(bool)
            or arrays["sentinel_pairs"].shape != (pair_count, 2)
            or not np.array_equal(arrays["sentinel_pairs"], expected_pairs)
            or not np.isfinite(arrays["marginal_lrt"]).all()
            or (arrays["marginal_lrt"] < 0).any()
            or manifest.get("marginal_reportable_count")
            != int(arrays["marginal_reportable"].sum())
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
    if nice_increment:
        os.nice(nice_increment)
    started = time.monotonic()
    cell = _load_cell(run_root, provider_root, cohort, provider)
    samplers = _prepare_samplers(cell)
    rng = np.random.default_rng(_seed(int(config["seed"]), cohort, provider))
    pair_count = int(config["marginal_lrt"]["sentinel_pair_count"])
    sentinel_pairs = _sentinel_pairs(cell.features, pair_count)
    marginal_replicates = int(config["marginal_lrt"]["replicates_per_cell"])
    marginal_lrt = np.empty(
        (marginal_replicates, len(sentinel_pairs)),
        dtype=np.float64,
    )
    marginal_reportable = np.empty(
        (marginal_replicates, len(sentinel_pairs)),
        dtype=bool,
    )
    for replicate in range(marginal_replicates):
        for pair_index, raw_indices in enumerate(sentinel_pairs):
            indices = (int(raw_indices[0]), int(raw_indices[1]))
            counts = _simulate_features(cell, indices, samplers, rng)
            statistic, reportable = _fit_pair(cell, indices, counts)
            marginal_lrt[replicate, pair_index] = statistic
            marginal_reportable[replicate, pair_index] = reportable

    task_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{provider}.", dir=task_root.parent))
    data_path = staging / TASK_DATA_NAME
    with data_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            marginal_lrt=marginal_lrt,
            marginal_reportable=marginal_reportable,
            sentinel_pairs=sentinel_pairs,
        )
        handle.flush()
        os.fsync(handle.fileno())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    resource_usage = core._task_resource_usage(started)  # noqa: SLF001
    resource_usage.update(
        {
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
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


def _run_batch(  # noqa: PLR0913
    tasks: Sequence[ProtocolCell],
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    jobs: int,
    nice_increment: int,
) -> None:
    if not tasks:
        return
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

    remaining = iter(tasks)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures: dict[Any, ProtocolCell] = {}
        for _ in range(jobs):
            cell = next(remaining, None)
            if cell is None:
                break
            futures[executor.submit(invoke, cell)] = cell
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
            next_cell = next(remaining, None)
            if next_cell is not None:
                futures[executor.submit(invoke, next_cell)] = next_cell


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
    if jobs < 1 or jobs > int(resources["max_jobs"]):
        msg = "Calibration --jobs exceeds the frozen resource contract."
        raise ValueError(msg)
    if nice_increment != int(resources["nice_increment"]):
        msg = "Calibration --nice differs from the frozen resource contract."
        raise ValueError(msg)
    protocol = _protocol_cells(config)
    primary = tuple(cell for cell in protocol if cell.provider == "mutsig")
    descriptive = tuple(cell for cell in protocol if cell.provider != "mutsig")
    _run_batch(
        primary,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        jobs=int(resources["max_mutsig_jobs"]),
        nice_increment=nice_increment,
    )
    _run_batch(
        descriptive,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        jobs=jobs,
        nice_increment=nice_increment,
    )


def _hoeffding_margin(
    *,
    trials: int,
    endpoint_count: int,
    familywise_error: float,
) -> float:
    """Return the simultaneous one-sided Hoeffding union-bound margin."""
    if trials < 1 or endpoint_count < 1 or not 0 < familywise_error < 1:
        msg = "Hoeffding gate parameters are invalid."
        raise ValueError(msg)
    return math.sqrt(math.log(endpoint_count / familywise_error) / (2 * trials))


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
    margin = _hoeffding_margin(
        trials=trials,
        endpoint_count=int(gate["endpoint_count"]),
        familywise_error=float(gate["familywise_error"]),
    )
    upper = min(1.0, successes / trials + margin)
    acceptance = float(gate["acceptance_upper_bounds"][f"{alpha:.2f}"])
    return {
        "hoeffding_familywise_error": float(gate["familywise_error"]),
        "hoeffding_endpoint_count": int(gate["endpoint_count"]),
        "hoeffding_margin": margin,
        "hoeffding_upper_bound": upper,
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
) -> tuple[pd.DataFrame, list[dict[str, int | str]], int]:
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
        _validate_task(
            task_root,
            config,
            cohort=cell.cohort,
            provider=cell.provider,
            role=cell.role,
            run_completion_sha256=run_completion_sha256,
            run_root=run_root,
        )
        task_manifests.append(
            _file_record(task_root / TASK_MANIFEST_NAME, relative_to=output_root),
        )
        with np.load(task_root / TASK_DATA_NAME, allow_pickle=False) as arrays:
            reportable = arrays["marginal_reportable"]
            p_values = _effective_p_values(arrays["marginal_lrt"], reportable)
            reportable_trials = int(reportable.sum())
            nonreportable_trials = int(reportable.size - reportable_trials)
            nonreportable_fit_count += nonreportable_trials
            for raw_alpha in config["marginal_lrt"]["alphas"]:
                alpha = float(raw_alpha)
                successes = int((p_values <= alpha).sum())
                trials = int(p_values.size)
                row: dict[str, Any] = {
                    "cohort": cell.cohort,
                    "provider": cell.provider,
                    "role": cell.role,
                    "screen": MARGINAL_SCREEN,
                    "threshold": alpha,
                    "events": successes,
                    "trials": trials,
                    "rate": successes / trials,
                    "reportable_trials": reportable_trials,
                    "nonreportable_trials": nonreportable_trials,
                    "gate_endpoint": cell.role == PRIMARY_ROLE,
                    "hoeffding_familywise_error": "",
                    "hoeffding_endpoint_count": "",
                    "hoeffding_margin": "",
                    "hoeffding_upper_bound": "",
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


def _summary_payload(  # noqa: PLR0913
    *,
    output_root: Path,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    task_manifests: list[dict[str, int | str]],
    nonreportable_fit_count: int,
) -> dict[str, Any]:
    run_completion_sha256, provider_manifest_sha256 = _validate_run_manifest(
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
    )
    gate_rows = frame.loc[frame["gate_endpoint"]]
    if (
        len(gate_rows) != int(config["affirmative_gate"]["endpoint_count"])
        or not gate_rows["endpoint_gate_pass"].isin(
            [ENDPOINT_ACCEPTED, ENDPOINT_REJECTED],
        ).all()
    ):
        msg = "Calibration gate endpoint inventory is invalid."
        raise ValueError(msg)
    passed = int(gate_rows["endpoint_gate_pass"].eq(ENDPOINT_ACCEPTED).sum())
    gate = config["affirmative_gate"]
    trials = int(gate_rows["trials"].iloc[0])
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
        "gate_method": gate["method"],
        "hoeffding_familywise_error": gate["familywise_error"],
        "acceptance_upper_bounds": gate["acceptance_upper_bounds"],
        "hoeffding_margin": _hoeffding_margin(
            trials=trials,
            endpoint_count=int(gate["endpoint_count"]),
            familywise_error=float(gate["familywise_error"]),
        ),
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

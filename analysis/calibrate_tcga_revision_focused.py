"""Run the prespecified focused parametric-null calibration.

Each cohort/provider cell is generated from its fitted independent DIALECT
marginals.  The workflow measures marginal profile-LRT rejection rates on 64
deterministic K=500-axis pairs and complete-null BH family events on the first 30
matched features.  It is a finite-scenario stress test, not a proof of uniform FDR
control, and it never reads observed pair statistics or selects a reporting threshold.
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, chi2

from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import run_tcga_revision_k500 as core
from analysis.run_tcga_revision_focused import _paths
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
CONFIG_PATH: Final = Path(__file__).with_name("tcga_revision_calibration_config.json")
RUN_CONTRACT: Final = "focused-parametric-null-calibration-run-v1"
TASK_CONTRACT: Final = "focused-parametric-null-calibration-cell-v1"
SUMMARY_CONTRACT: Final = "focused-parametric-null-calibration-summary-v1"
TASK_DATA_NAME: Final = "calibration_arrays.npz"
TASK_MANIFEST_NAME: Final = "task_manifest.json"
RUN_MANIFEST_NAME: Final = "run_manifest.json"
SUMMARY_NAME: Final = "calibration_summary.json"
SUMMARY_TABLE_NAME: Final = "calibration_cells.csv"
THREAD_ENV: Final = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class Cell:
    """One fitted cohort/provider independence generator."""

    cohort: str
    provider: str
    features: tuple[str, ...]
    samples: tuple[str, ...]
    pmfs: Mapping[str, Any]
    pi: np.ndarray


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


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cells = config.get("cells", {})
    marginal = config.get("marginal_lrt", {})
    family = config.get("complete_null_family", {})
    reporting = config.get("reporting_candidates", {})
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("contract") != "focused-result-blind-null-calibration-v1"
        or cells.get("cohorts") != ["CHOL", "LAML", "PAAD", "SKCM", "UCEC"]
        or cells.get("providers") != list(core.BMRS)
        or marginal.get("replicates_per_cell") != 1000
        or marginal.get("alphas") != [0.01, 0.05, 0.1]
        or family.get("top_k") != 30
        or family.get("replicates_per_cell") != 250
        or family.get("candidate_q_values") != [0.1, 0.2]
        or reporting.get("primary_q_threshold") != 0.1
        or reporting.get("sensitivity_q_threshold") != 0.2
        or reporting.get("thresholds_selected_from_observed_pairs") is not False
    ):
        msg = "Focused calibration configuration violates its frozen contract."
        raise ValueError(msg)
    return config


def _seed(root_seed: int, cohort: str, provider: str) -> int:
    digest = hashlib.sha256(
        _canonical_json([root_seed, "focused-calibration", cohort, provider]),
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
) -> float:
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
    return max(statistic, 0.0)


def _fit_family(
    cell: Cell,
    feature_indices: Sequence[int],
    counts: np.ndarray,
) -> np.ndarray:
    genes: dict[str, Gene] = {}
    for column, index in enumerate(feature_indices):
        feature = cell.features[index]
        gene = Gene(
            name=feature,
            samples=cell.samples,
            counts=counts[:, column],
            bmr_pmf=cell.pmfs[feature],
        )
        gene.estimate_pi_with_mle()
        genes[feature] = gene
    values = []
    for feature_a, feature_b in core.iter_tested_pairs(tuple(genes)):
        interaction = Interaction(genes[feature_a], genes[feature_b])
        interaction.estimate_tau_with_coordinate_ascent()
        statistic = float(interaction.likelihood_ratio)
        if not np.isfinite(statistic) or statistic < -1e-10:
            msg = "Calibration family produced an invalid profile LRT."
            raise ValueError(msg)
        values.append(max(statistic, 0.0))
    return np.asarray(values, dtype=np.float64)


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


def _load_cell(run_root: Path, provider_root: Path, cohort: str, provider: str) -> Cell:
    contract = json.loads(
        (run_root / "contracts" / f"{cohort}.json").read_text(encoding="utf-8"),
    )
    paths = _paths(provider_root, run_root)
    current = core.build_cohort_contract(paths, cohort, top_k=500)
    if contract != current:
        msg = f"Frozen cohort contract drifted before calibration: {cohort}"
        raise ValueError(msg)
    counts, pmfs = core._load_frozen_scientific_inputs(contract, provider)  # noqa: SLF001
    single = pd.read_csv(
        run_root / "tasks" / cohort / provider / "single_gene_results.csv",
        float_precision="round_trip",
    )
    features = tuple(contract["features"])
    if single["Gene Name"].tolist() != list(features):
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
    )


def _task_root(output_root: Path, cohort: str, provider: str) -> Path:
    return output_root / "tasks" / cohort / provider


def _run_completion_sha256(output_root: Path, config: Mapping[str, Any]) -> str:
    manifest = json.loads(
        (output_root / RUN_MANIFEST_NAME).read_text(encoding="utf-8"),
    )
    run_completion = manifest.get("run_completion", {})
    provider_manifest = manifest.get("provider_manifest", {})
    digest = run_completion.get("sha256")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != RUN_CONTRACT
        or manifest.get("config", {}).get("sha256") != _sha256(CONFIG_PATH)
        or manifest.get("cohorts") != config["cells"]["cohorts"]
        or manifest.get("providers") != list(core.BMRS)
        or manifest.get("observed_pair_statistics_opened") is not False
        or run_completion.get("path") != "completion_manifest.json"
        or not isinstance(run_completion.get("bytes"), int)
        or run_completion["bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or provider_manifest.get("path") != "provider_manifest.json"
        or not isinstance(provider_manifest.get("bytes"), int)
        or provider_manifest["bytes"] <= 0
        or not isinstance(provider_manifest.get("sha256"), str)
        or len(provider_manifest["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in provider_manifest["sha256"]
        )
    ):
        msg = "Calibration run manifest is invalid."
        raise ValueError(msg)
    return digest


def _validate_task(
    task_root: Path,
    config: Mapping[str, Any],
    *,
    cohort: str,
    provider: str,
    run_completion_sha256: str,
) -> dict[str, Any]:
    manifest = json.loads((task_root / TASK_MANIFEST_NAME).read_text(encoding="utf-8"))
    data = task_root / TASK_DATA_NAME
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != TASK_CONTRACT
        or manifest.get("cohort") != cohort
        or manifest.get("provider") != provider
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("run_completion_sha256") != run_completion_sha256
        or manifest.get("seed") != _seed(int(config["seed"]), cohort, provider)
        or manifest.get("marginal_replicates")
        != int(config["marginal_lrt"]["replicates_per_cell"])
        or manifest.get("sentinel_pair_count") != 64
        or manifest.get("family_replicates")
        != int(config["complete_null_family"]["replicates_per_cell"])
        or manifest.get("family_top_k")
        != int(config["complete_null_family"]["top_k"])
        or manifest.get("q_values")
        != config["complete_null_family"]["candidate_q_values"]
        or manifest.get("output", {}).get("path") != TASK_DATA_NAME
        or manifest.get("output", {}).get("bytes") != data.stat().st_size
        or manifest.get("output", {}).get("sha256") != _sha256(data)
    ):
        msg = f"Calibration task failed manifest validation: {task_root}"
        raise ValueError(msg)
    core._validate_task_resource_usage(manifest, task_root)  # noqa: SLF001
    with np.load(data, allow_pickle=False) as arrays:
        expected = {
            "marginal_lrt",
            "family_rejections",
            "family_min_p",
            "sentinel_pairs",
        }
        marginal_replicates = int(config["marginal_lrt"]["replicates_per_cell"])
        family_replicates = int(config["complete_null_family"]["replicates_per_cell"])
        if (
            set(arrays.files) != expected
            or arrays["marginal_lrt"].shape != (marginal_replicates, 64)
            or arrays["family_rejections"].shape != (family_replicates, 2)
            or arrays["family_min_p"].shape != (family_replicates,)
            or arrays["sentinel_pairs"].shape != (64, 2)
            or not np.isfinite(arrays["marginal_lrt"]).all()
            or (arrays["marginal_lrt"] < 0).any()
            or not np.isfinite(arrays["family_min_p"]).all()
            or (arrays["family_min_p"] < 0).any()
            or (arrays["family_min_p"] > 1).any()
            or (arrays["family_rejections"] < 0).any()
            or (
                arrays["family_rejections"]
                > (
                    int(config["complete_null_family"]["top_k"])
                    * (int(config["complete_null_family"]["top_k"]) - 1)
                    // 2
                )
            ).any()
        ):
            msg = f"Calibration task arrays failed validation: {task_root}"
            raise ValueError(msg)
    return manifest


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
    task_root = _task_root(output_root, cohort, provider)
    run_completion_sha256 = _sha256(run_root / "completion_manifest.json")
    if task_root.exists():
        _validate_task(
            task_root,
            config,
            cohort=cohort,
            provider=provider,
            run_completion_sha256=run_completion_sha256,
        )
        return "already-complete"
    if nice_increment:
        os.nice(nice_increment)
    started = time.monotonic()
    cell = _load_cell(run_root, provider_root, cohort, provider)
    samplers = _prepare_samplers(cell)
    rng = np.random.default_rng(_seed(int(config["seed"]), cohort, provider))
    sentinel_pairs = _sentinel_pairs(cell.features)
    marginal_replicates = int(config["marginal_lrt"]["replicates_per_cell"])
    marginal_lrt = np.empty(
        (marginal_replicates, len(sentinel_pairs)),
        dtype=np.float64,
    )
    for replicate in range(marginal_replicates):
        for pair_index, raw_indices in enumerate(sentinel_pairs):
            indices = (int(raw_indices[0]), int(raw_indices[1]))
            counts = _simulate_features(cell, indices, samplers, rng)
            marginal_lrt[replicate, pair_index] = _fit_pair(cell, indices, counts)

    family_k = int(config["complete_null_family"]["top_k"])
    family_indices = tuple(range(family_k))
    q_values = tuple(
        float(value)
        for value in config["complete_null_family"]["candidate_q_values"]
    )
    family_replicates = int(config["complete_null_family"]["replicates_per_cell"])
    family_rejections = np.empty((family_replicates, len(q_values)), dtype=np.int32)
    family_min_p = np.empty(family_replicates, dtype=np.float64)
    for replicate in range(family_replicates):
        counts = _simulate_features(cell, family_indices, samplers, rng)
        lrt = _fit_family(cell, family_indices, counts)
        p_values = chi2.sf(lrt, df=1)
        adjusted = postprocess.benjamini_hochberg(p_values)
        family_min_p[replicate] = float(p_values.min(initial=1.0))
        for q_index, threshold in enumerate(q_values):
            family_rejections[replicate, q_index] = int((adjusted <= threshold).sum())

    task_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{provider}.", dir=task_root.parent))
    data_path = staging / TASK_DATA_NAME
    with data_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            marginal_lrt=marginal_lrt,
            family_rejections=family_rejections,
            family_min_p=family_min_p,
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
        "config_sha256": _sha256(CONFIG_PATH),
        "run_completion_sha256": run_completion_sha256,
        "seed": _seed(int(config["seed"]), cohort, provider),
        "marginal_replicates": marginal_replicates,
        "sentinel_pair_count": len(sentinel_pairs),
        "family_replicates": family_replicates,
        "family_top_k": family_k,
        "q_values": list(q_values),
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
        run_completion_sha256=run_completion_sha256,
    )
    return "complete"


def _ensure_run_root(
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _load_config()
    cohorts = tuple(config["cells"]["cohorts"])
    postprocess._validate_completion(run_root, cohorts)  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tasks").mkdir(exist_ok=True)
    manifest = {
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
        "cohorts": list(cohorts),
        "providers": list(core.BMRS),
        "observed_pair_statistics_opened": False,
    }
    content = _canonical_json(manifest) + b"\n"
    path = output_root / RUN_MANIFEST_NAME
    if path.exists():
        if path.read_bytes() != content:
            msg = "Calibration run root is bound to different inputs."
            raise ValueError(msg)
    else:
        _write_atomic(path, content)
    return config


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


def run_all(
    *,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    jobs: int,
    nice_increment: int,
) -> None:
    """Run all frozen calibration cells with bounded low-priority workers."""
    config = _ensure_run_root(run_root, provider_root, output_root)
    resources = config["resources"]
    if jobs < 1 or jobs > int(resources["max_jobs"]):
        msg = "Calibration --jobs exceeds the frozen resource contract."
        raise ValueError(msg)
    if nice_increment != int(resources["nice_increment"]):
        msg = "Calibration --nice differs from the frozen resource contract."
        raise ValueError(msg)
    tasks = [
        (cohort, provider)
        for cohort in config["cells"]["cohorts"]
        for provider in config["cells"]["providers"]
    ]
    environment = os.environ.copy()
    environment.update(THREAD_ENV)

    def invoke(task: tuple[str, str]) -> None:
        cohort, provider = task
        subprocess.run(
            _task_command(
                run_root=run_root,
                provider_root=provider_root,
                output_root=output_root,
                cohort=cohort,
                provider=provider,
                nice_increment=nice_increment,
            ),
            check=True,
            env=environment,
        )

    remaining = iter(tasks)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures: dict[Any, tuple[str, str]] = {}
        for _ in range(jobs):
            task = next(remaining, None)
            if task is None:
                break
            futures[executor.submit(invoke, task)] = task
        while futures:
            future = next(as_completed(futures))
            cohort, provider = futures.pop(future)
            try:
                future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            print(f"calibration complete {cohort}/{provider}", flush=True)
            task = next(remaining, None)
            if task is not None:
                futures[executor.submit(invoke, task)] = task


def _interval(successes: int, trials: int) -> tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    return lower, upper


def summarize(*, output_root: Path) -> Path:
    """Summarize prespecified inflation tests without selecting a new rule."""
    table_path = output_root / SUMMARY_TABLE_NAME
    summary_path = output_root / SUMMARY_NAME
    if table_path.exists() or summary_path.exists():
        msg = "Refusing to overwrite calibration summary artifacts."
        raise FileExistsError(msg)
    config = _load_config()
    run_completion_sha256 = _run_completion_sha256(output_root, config)
    marginal_tests = len(config["cells"]["cohorts"]) * len(core.BMRS) * len(
        config["marginal_lrt"]["alphas"],
    )
    family_tests = len(config["cells"]["cohorts"]) * len(core.BMRS) * len(
        config["complete_null_family"]["candidate_q_values"],
    )
    rows = []
    task_manifests = []
    detected = False
    for cohort in config["cells"]["cohorts"]:
        for provider in core.BMRS:
            task_root = _task_root(output_root, cohort, provider)
            _validate_task(
                task_root,
                config,
                cohort=cohort,
                provider=provider,
                run_completion_sha256=run_completion_sha256,
            )
            task_manifests.append(
                _file_record(
                    task_root / TASK_MANIFEST_NAME,
                    relative_to=output_root,
                ),
            )
            with np.load(task_root / TASK_DATA_NAME, allow_pickle=False) as arrays:
                p_values = chi2.sf(arrays["marginal_lrt"], df=1)
                for alpha in config["marginal_lrt"]["alphas"]:
                    successes = int((p_values <= float(alpha)).sum())
                    trials = int(p_values.size)
                    p_inflation = float(
                        binomtest(
                            successes,
                            trials,
                            float(alpha),
                            alternative="greater",
                        ).pvalue,
                    )
                    adjusted = min(1.0, p_inflation * marginal_tests)
                    detected |= adjusted < 0.05
                    lower, upper = _interval(successes, trials)
                    rows.append(
                        {
                            "cohort": cohort,
                            "provider": provider,
                            "screen": "marginal_lrt",
                            "threshold": float(alpha),
                            "events": successes,
                            "replicates": trials,
                            "rate": successes / trials,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "bonferroni_p": adjusted,
                            "inflation_detected": adjusted < 0.05,
                        },
                    )
                family = arrays["family_rejections"]
                for index, q_value in enumerate(
                    config["complete_null_family"]["candidate_q_values"],
                ):
                    events = int((family[:, index] > 0).sum())
                    trials = len(family)
                    p_inflation = float(
                        binomtest(
                            events,
                            trials,
                            float(q_value),
                            alternative="greater",
                        ).pvalue,
                    )
                    adjusted = min(1.0, p_inflation * family_tests)
                    detected |= adjusted < 0.05
                    lower, upper = _interval(events, trials)
                    rows.append(
                        {
                            "cohort": cohort,
                            "provider": provider,
                            "screen": "complete_null_bh_family",
                            "threshold": float(q_value),
                            "events": events,
                            "replicates": trials,
                            "rate": events / trials,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "bonferroni_p": adjusted,
                            "inflation_detected": adjusted < 0.05,
                        },
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(table_path, index=False, lineterminator="\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "contract": SUMMARY_CONTRACT,
        "config_sha256": _sha256(CONFIG_PATH),
        "cell_count": len(config["cells"]["cohorts"]) * len(core.BMRS),
        "marginal_test_count": marginal_tests,
        "family_test_count": family_tests,
        "detected_inflation": detected,
        "retain_chi_square_bh_candidates": not detected,
        "primary_q_candidate": config["reporting_candidates"]["primary_q_threshold"],
        "sensitivity_q_candidate": config["reporting_candidates"][
            "sensitivity_q_threshold"
        ],
        "interpretation": config["reporting_candidates"]["interpretation"],
        "reporting_rule_selected": False,
        "run_manifest": _file_record(
            output_root / RUN_MANIFEST_NAME,
            relative_to=output_root,
        ),
        "task_manifests": task_manifests,
        "table": _file_record(table_path, relative_to=output_root),
    }
    _write_atomic(summary_path, _canonical_json(summary) + b"\n")
    return summary_path


def validate_summary(output_root: Path) -> dict[str, Any]:
    """Validate the complete calibration tree and its immutable summary."""
    config = _load_config()
    summary_path = output_root / SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_tasks = [
        (cohort, provider)
        for cohort in config["cells"]["cohorts"]
        for provider in core.BMRS
    ]
    records = {
        str(record.get("path")): record
        for record in summary.get("task_manifests", [])
    }
    table_path = output_root / SUMMARY_TABLE_NAME
    table_record = summary.get("table", {})
    run_path = output_root / RUN_MANIFEST_NAME
    run_record = summary.get("run_manifest", {})
    run_completion_sha256 = _run_completion_sha256(output_root, config)
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("contract") != SUMMARY_CONTRACT
        or summary.get("config_sha256") != _sha256(CONFIG_PATH)
        or summary.get("cell_count") != len(expected_tasks)
        or summary.get("marginal_test_count")
        != len(expected_tasks) * len(config["marginal_lrt"]["alphas"])
        or summary.get("family_test_count")
        != len(expected_tasks)
        * len(config["complete_null_family"]["candidate_q_values"])
        or summary.get("primary_q_candidate")
        != config["reporting_candidates"]["primary_q_threshold"]
        or summary.get("sensitivity_q_candidate")
        != config["reporting_candidates"]["sensitivity_q_threshold"]
        or summary.get("reporting_rule_selected") is not False
        or not isinstance(summary.get("detected_inflation"), bool)
        or not isinstance(summary.get("retain_chi_square_bh_candidates"), bool)
        or summary.get("retain_chi_square_bh_candidates")
        == summary.get("detected_inflation")
        or table_record.get("path") != SUMMARY_TABLE_NAME
        or table_record.get("bytes") != table_path.stat().st_size
        or table_record.get("sha256") != _sha256(table_path)
        or run_record.get("path") != RUN_MANIFEST_NAME
        or run_record.get("bytes") != run_path.stat().st_size
        or run_record.get("sha256") != _sha256(run_path)
        or len(records) != len(expected_tasks)
    ):
        msg = "Calibration summary or receipt inventory is invalid."
        raise ValueError(msg)
    for cohort, provider in expected_tasks:
        task_root = _task_root(output_root, cohort, provider)
        _validate_task(
            task_root,
            config,
            cohort=cohort,
            provider=provider,
            run_completion_sha256=run_completion_sha256,
        )
        relative = f"tasks/{cohort}/{provider}/{TASK_MANIFEST_NAME}"
        record = records.get(relative, {})
        manifest_path = output_root / relative
        if (
            record.get("bytes") != manifest_path.stat().st_size
            or record.get("sha256") != _sha256(manifest_path)
        ):
            msg = f"Calibration task receipt changed: {cohort}/{provider}"
            raise ValueError(msg)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--provider-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--internal-cohort")
    parser.add_argument("--internal-provider", choices=core.BMRS)
    return parser


def main() -> None:
    """Run, resume, or summarize focused calibration."""
    args = _parser().parse_args()
    output_root = args.output_root.absolute()
    if args.summarize:
        print(summarize(output_root=output_root))
        return
    if args.run_root is None or args.provider_root is None:
        msg = "--run-root and --provider-root are required for calibration execution."
        raise ValueError(msg)
    run_root = args.run_root.resolve()
    provider_root = args.provider_root.resolve()
    if (args.internal_cohort is None) != (args.internal_provider is None):
        msg = "Internal cohort and provider must be supplied together."
        raise ValueError(msg)
    if args.internal_cohort is not None:
        config = _load_config()
        if (
            args.internal_cohort not in config["cells"]["cohorts"]
            or args.internal_provider not in core.BMRS
        ):
            msg = "Internal calibration cell is outside the frozen protocol."
            raise ValueError(msg)
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

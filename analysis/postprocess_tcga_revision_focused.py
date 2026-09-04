"""Derive provider-specific p- and q-values for the focused K=500 grid.

The fit tests dependence once per unordered pair.  This stage retains the complete
within-cohort pair family for each BMR and emits both Benjamini--Yekutieli (BY) and
Benjamini--Hochberg (BH) adjusted values.  Direction is a descriptive annotation from
the fitted Marshall--Olkin rho, never a separate test.  This module does not select a
threshold or combine providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from analysis import run_tcga_revision_focused as focused_runner
from analysis import run_tcga_revision_k500 as core
from analysis.prepare_tcga_revision_focused import _parse_cohorts
from analysis.run_tcga_revision_k500 import BMRS, PAIRWISE_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION: Final = "1.0.0"
DERIVATION_CONTRACT: Final = "focused-provider-complete-family-log-by-bh-v4"
ROOT_CONTRACT: Final = "focused-32x3-provider-inference-v4"
RESULT_NAME: Final = "provider_inference.csv"
COHORT_MANIFEST_NAME: Final = "cohort_manifest.json"
ROOT_MANIFEST_NAME: Final = "postprocess_manifest.json"
LOG_MIN_POSITIVE_FLOAT: Final = float(np.log(np.nextafter(0.0, 1.0)))
PROBABILITY_REPRESENTATION: Final = (
    "natural-log-inference-with-smallest-positive-float-clipped-display"
)
FIT_DIAGNOSTIC_FIELDS: Final = (
    "fit_converged",
    "fit_iterations",
    "fit_last_ll_gain",
    "fit_fixed_point_residual",
    "fit_kkt_residual",
)
RAW_FIT_DIAGNOSTIC_COLUMNS: Final = {
    "fit_converged": "Fit Converged",
    "fit_iterations": "Fit Iterations",
    "fit_last_ll_gain": "Fit Last LL Gain",
    "fit_fixed_point_residual": "Fit Fixed-Point Residual",
    "fit_kkt_residual": "Fit KKT Residual",
}


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


def _require_regular_file(path: Path, *, label: str) -> None:
    """Reject missing files and any path whose final component is a symlink."""
    if not path.is_file() or path.is_symlink():
        msg = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(msg)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    _require_regular_file(path, label="Required regular file")
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


def _validate_log_probabilities(log_values: np.ndarray, *, label: str) -> np.ndarray:
    """Return one-dimensional log probabilities in ``[-inf, 0]``."""
    values = np.asarray(log_values, dtype=np.float64)
    if (
        values.ndim != 1
        or np.isnan(values).any()
        or np.isposinf(values).any()
        or (values > 0).any()
    ):
        msg = f"{label} requires one-dimensional log probabilities in [-inf, 0]."
        raise ValueError(msg)
    return values


def log_benjamini_hochberg(log_p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted log p-values for one complete finite family."""
    values = _validate_log_probabilities(log_p_values, label="BH")
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = (
        ranked
        + np.log(float(count))
        - np.log(np.arange(1, count + 1, dtype=np.float64))
    )
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 0.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _by_from_log_bh(log_bh_adjusted: np.ndarray) -> np.ndarray:
    """Return BY log q-values from BH log q-values for the same family."""
    adjusted = _validate_log_probabilities(log_bh_adjusted, label="BY")
    if len(adjusted) == 0:
        return adjusted.copy()
    harmonic = float(
        np.sum(1.0 / np.arange(1, len(adjusted) + 1, dtype=np.float64)),
    )
    return np.minimum(adjusted + np.log(harmonic), 0.0)


def log_benjamini_yekutieli(log_p_values: np.ndarray) -> np.ndarray:
    """Return BY-adjusted log p-values for one complete finite family."""
    return _by_from_log_bh(log_benjamini_hochberg(log_p_values))


def _display_probabilities(log_values: np.ndarray) -> np.ndarray:
    """Exponentiate log probabilities without ever emitting a literal zero."""
    values = _validate_log_probabilities(log_values, label="Display conversion")
    return np.exp(np.maximum(values, LOG_MIN_POSITIVE_FLOAT))


def _log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Convert ordinary probabilities to log space, preserving exact-zero inputs."""
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        msg = "Probabilities must be a one-dimensional finite p-value family in [0, 1]."
        raise ValueError(msg)
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    positive = values > 0
    result[positive] = np.log(values[positive])
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return positive-display BH values; inference itself is performed in log space."""
    return _display_probabilities(log_benjamini_hochberg(_log_probabilities(p_values)))


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """Return positive-display BY values; inference itself is performed in log space."""
    return _display_probabilities(log_benjamini_yekutieli(_log_probabilities(p_values)))


def _log_chi_square_one_df_survival(statistics: np.ndarray) -> np.ndarray:
    """Return the stable log survival function for a chi-square(1) statistic."""
    values = np.asarray(statistics, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        msg = "Chi-square statistics must be one-dimensional, finite, and nonnegative."
        raise ValueError(msg)
    result = np.log(2.0) + log_ndtr(-np.sqrt(values))
    if np.isnan(result).any() or np.isposinf(result).any() or (result > 0).any():
        msg = "Stable chi-square log-survival evaluation failed."
        raise ValueError(msg)
    return result


def _direction(rho: pd.Series) -> pd.Series:
    values = pd.to_numeric(rho, errors="coerce").to_numpy(dtype=np.float64)
    labels = np.full(len(values), "unavailable", dtype=object)
    labels[np.isfinite(values) & (values < 0)] = "ME"
    labels[np.isfinite(values) & (values > 0)] = "CO"
    labels[np.isfinite(values) & (values == 0)] = "neutral"
    return pd.Series(labels, index=rho.index, dtype="string")


def _provider_statistics(
    likelihood_ratio: np.ndarray,
    rho: pd.Series,
    identifiability: pd.Series,
) -> dict[str, np.ndarray | pd.Series]:
    """Apply the complete-family policy to one provider's fitted statistics."""
    statistics = np.asarray(likelihood_ratio, dtype=np.float64)
    effects = identifiability.astype("string")
    allowed_identifiability = {
        "full-affine-rank",
        "rank-deficient",
        "rank-not-certified-underflow",
    }
    if (
        statistics.ndim != 1
        or len(statistics) != len(rho)
        or len(statistics) != len(effects)
        or not np.isfinite(statistics).all()
        or (statistics < -1e-10).any()
        or not effects.isin(allowed_identifiability).all()
    ):
        msg = "Invalid fitted statistics for provider inference."
        raise ValueError(msg)
    statistics = np.maximum(statistics, 0.0)
    reportable = effects.eq("full-affine-rank").to_numpy()
    log_p_values = _log_chi_square_one_df_survival(statistics)
    log_p_values[~reportable] = 0.0
    log_bh_q_values = log_benjamini_hochberg(log_p_values)
    log_by_q_values = _by_from_log_bh(log_bh_q_values)
    numeric_rho = pd.to_numeric(rho, errors="coerce")
    rho_missing = rho.isna().to_numpy()
    numeric_rho_values = numeric_rho.to_numpy(dtype=float)
    finite_rho = np.isfinite(numeric_rho_values)
    if ((~rho_missing & ~finite_rho) | np.isinf(numeric_rho_values)).any():
        msg = "A fitted rho must be finite or genuinely missing."
        raise ValueError(msg)
    invalid_missing_rho = (
        reportable
        & ~finite_rho
        & (statistics > core.REQUIRED_UNDEFINED_RHO_LRT_TOL)
    )
    if invalid_missing_rho.any():
        msg = "A full-rank positive-LRT fitted effect has no finite rho."
        raise ValueError(msg)
    if (finite_rho & ~reportable).any():
        msg = "A non-full-rank fitted effect must not report rho."
        raise ValueError(msg)
    finite_reportable_rho = numeric_rho.loc[reportable & finite_rho].to_numpy(
        dtype=float,
    )
    if (np.abs(finite_reportable_rho) > 1 + core.REQUIRED_PAIR_SIMPLEX_TOL).any():
        msg = "A fitted rho lies outside its valid range."
        raise ValueError(msg)
    reportable_rho = numeric_rho.where(reportable)
    return {
        "likelihood_ratio": statistics,
        "log_p_value": log_p_values,
        "p_value": _display_probabilities(log_p_values),
        "log_by_q_value": log_by_q_values,
        "by_q_value": _display_probabilities(log_by_q_values),
        "log_bh_q_value": log_bh_q_values,
        "bh_q_value": _display_probabilities(log_bh_q_values),
        "rho": reportable_rho,
        "direction": _direction(reportable_rho),
        "effect_identifiability": effects,
        "effect_reportable": reportable,
    }


def _validated_fit_diagnostics(  # noqa: PLR0913
    *,
    fit_converged: pd.Series,
    fit_iterations: pd.Series,
    fit_last_ll_gain: pd.Series,
    fit_fixed_point_residual: pd.Series,
    fit_kkt_residual: pd.Series,
    label: str,
) -> dict[str, np.ndarray]:
    """Return canonical pair-fit receipts after strict type and range checks."""
    if not pd.api.types.is_bool_dtype(fit_converged.dtype):
        msg = f"Pair-fit convergence receipts are not boolean: {label}"
        raise ValueError(msg)
    converged = fit_converged.to_numpy(dtype=np.bool_)
    if not converged.all():
        msg = f"Pair-fit convergence receipts contain an unaccepted fit: {label}"
        raise ValueError(msg)

    if not pd.api.types.is_integer_dtype(fit_iterations.dtype):
        msg = f"Pair-fit iteration receipts are not integers: {label}"
        raise ValueError(msg)
    iterations = fit_iterations.to_numpy(dtype=np.int64)
    if (
        (iterations < 0).any()
        or (iterations > core.REQUIRED_PAIR_FIT_MAX_ITER).any()
    ):
        msg = f"Pair-fit iteration receipts are outside the frozen range: {label}"
        raise ValueError(msg)

    numeric_series = {
        "last log-likelihood gain": fit_last_ll_gain,
        "fixed-point residual": fit_fixed_point_residual,
        "KKT residual": fit_kkt_residual,
    }
    numeric: dict[str, np.ndarray] = {}
    for name, series in numeric_series.items():
        if (
            not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            msg = f"Pair-fit {name} receipts are not real-valued: {label}"
            raise ValueError(msg)
        values = series.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            msg = f"Pair-fit {name} receipts are invalid: {label}"
            raise ValueError(msg)
        numeric[name] = values

    gains = numeric["last log-likelihood gain"]
    fixed_point = numeric["fixed-point residual"]
    kkt = numeric["KKT residual"]
    if (gains[iterations == 0] != 0).any():
        msg = f"Zero-iteration pair fits must have zero last gain: {label}"
        raise ValueError(msg)
    if (
        (fixed_point > core.REQUIRED_PAIR_FIT_KKT_TOL).any()
        or (kkt > core.REQUIRED_PAIR_FIT_KKT_TOL).any()
    ):
        msg = f"Pair-fit certificate residual exceeds the frozen tolerance: {label}"
        raise ValueError(msg)
    return {
        "fit_converged": converged,
        "fit_iterations": iterations,
        "fit_last_ll_gain": gains,
        "fit_fixed_point_residual": fixed_point,
        "fit_kkt_residual": kkt,
    }


def _raw_fit_diagnostics(
    frame: pd.DataFrame,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    """Validate and copy the five public diagnostics from one raw pair table."""
    return _validated_fit_diagnostics(
        **{
            field: frame[column]
            for field, column in RAW_FIT_DIAGNOSTIC_COLUMNS.items()
        },
        label=label,
    )


def provider_fit_diagnostics(
    frame: pd.DataFrame,
    provider: str,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    """Validate one provider's embedded public pair-fit diagnostic columns."""
    return _validated_fit_diagnostics(
        **{
            field: frame[f"{provider}_{field}"]
            for field in FIT_DIAGNOSTIC_FIELDS
        },
        label=label,
    )


def result_columns() -> tuple[str, ...]:
    """Return the exact provider-inference table schema."""
    columns = ["gene_a", "gene_b"]
    for provider in BMRS:
        columns.extend(
            [
                f"{provider}_likelihood_ratio",
                f"{provider}_log_p_value",
                f"{provider}_p_value",
                f"{provider}_log_by_q_value",
                f"{provider}_by_q_value",
                f"{provider}_log_bh_q_value",
                f"{provider}_bh_q_value",
                f"{provider}_rho",
                f"{provider}_direction",
                f"{provider}_effect_identifiability",
                f"{provider}_effect_reportable",
                *(f"{provider}_{field}" for field in FIT_DIAGNOSTIC_FIELDS),
            ],
        )
    return tuple(columns)


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    """Compare two serialized columns without conflating missing numeric values."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return np.array_equal(
            left.to_numpy(dtype=np.float64),
            right.to_numpy(dtype=np.float64),
            equal_nan=True,
        )
    return left.astype("string").fillna("<NA>").equals(
        right.astype("string").fillna("<NA>"),
    )


def validate_inference_frame(frame: pd.DataFrame, *, cohort: str) -> None:
    """Recompute every inferential annotation over each complete provider family."""
    if (
        tuple(frame.columns) != result_columns()
        or frame[["gene_a", "gene_b"]].duplicated().any()
        or frame[["gene_a", "gene_b"]].isna().any().any()
    ):
        msg = f"Invalid postprocessed inference schema or pair axis: {cohort}"
        raise ValueError(msg)
    for provider in BMRS:
        diagnostics = provider_fit_diagnostics(
            frame,
            provider,
            label=f"{cohort}/{provider}",
        )
        expected = _provider_statistics(
            frame[f"{provider}_likelihood_ratio"].to_numpy(dtype=np.float64),
            frame[f"{provider}_rho"],
            frame[f"{provider}_effect_identifiability"],
        )
        for name, values in expected.items():
            column = f"{provider}_{name}"
            if not _same_values(frame[column], pd.Series(values, index=frame.index)):
                msg = (
                    "Postprocessed inference does not reproduce the complete-family "
                    f"policy: {cohort}/{provider}/{name}"
                )
                raise ValueError(msg)
        for name, values in diagnostics.items():
            column = f"{provider}_{name}"
            if not _same_values(frame[column], pd.Series(values, index=frame.index)):
                msg = (
                    "Postprocessed fit diagnostics are not canonical: "
                    f"{cohort}/{provider}/{name}"
                )
                raise ValueError(msg)


def _read_provider(run_root: Path, cohort: str, provider: str) -> pd.DataFrame:
    task_root = run_root / "tasks" / cohort / provider
    contract_path = run_root / "contracts" / f"{cohort}.json"
    manifest_path = task_root / "task_manifest.json"
    if task_root.is_symlink():
        msg = f"Task output directory is unsafe: {cohort}/{provider}"
        raise ValueError(msg)
    _require_regular_file(contract_path, label="Focused cohort contract")
    _require_regular_file(manifest_path, label="Focused task manifest")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sha256 = hashlib.sha256(_canonical_json(contract)).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = task_root / "pairwise_interaction_results.csv"
    source_record = manifest.get("outputs", {}).get(source.name, {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != focused_runner.TASK_CONTRACT
        or manifest.get("cohort") != cohort
        or manifest.get("provider") != provider
        or manifest.get("top_k") != 500
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("config_sha256")
        != _sha256(focused_runner.CONFIG_PATH)
        or manifest.get("pairwise_rows") != contract["pair_policy"]["row_count"]
        or set(manifest.get("outputs", {}))
        != {"pairwise_interaction_results.csv", "single_gene_results.csv"}
        or {path.name for path in task_root.iterdir()}
        != {
            "pairwise_interaction_results.csv",
            "single_gene_results.csv",
            "task_manifest.json",
        }
        or source_record.get("path") != source.name
        or not source.is_file()
        or source.is_symlink()
        or source_record.get("bytes") != source.stat().st_size
        or source_record.get("sha256") != _sha256(source)
    ):
        msg = f"Task output is not bound by its manifest: {cohort}/{provider}"
        raise ValueError(msg)
    frame = pd.read_csv(source, float_precision="round_trip")
    if tuple(frame.columns) != PAIRWISE_COLUMNS:
        msg = f"Unexpected pairwise schema: {cohort}/{provider}"
        raise ValueError(msg)
    if len(frame) != contract["pair_policy"]["row_count"]:
        msg = f"Invalid profile likelihood ratios: {cohort}/{provider}"
        raise ValueError(msg)
    expected_pairs = list(core.iter_tested_pairs(contract["features"]))
    observed_pairs = list(
        zip(frame["Gene A"].astype(str), frame["Gene B"].astype(str), strict=True),
    )
    if observed_pairs != expected_pairs:
        msg = f"Pair axis differs from the frozen cohort contract: {cohort}/{provider}"
        raise ValueError(msg)
    statistics = _provider_statistics(
        frame["Likelihood Ratio"].to_numpy(dtype=np.float64),
        frame["Rho"],
        frame["Effect Identifiability"],
    )
    fit_diagnostics = _raw_fit_diagnostics(
        frame,
        label=f"{cohort}/{provider}",
    )
    result = pd.DataFrame(
        {
            "gene_a": frame["Gene A"].astype("string"),
            "gene_b": frame["Gene B"].astype("string"),
            **{
                f"{provider}_{name}": value
                for name, value in statistics.items()
            },
            **{
                f"{provider}_{name}": value
                for name, value in fit_diagnostics.items()
            },
        },
    )
    if tuple(result.columns) != tuple(
        column
        for column in result_columns()
        if column in {"gene_a", "gene_b"} or column.startswith(f"{provider}_")
    ):
        msg = f"Internal provider inference schema drifted: {cohort}/{provider}"
        raise RuntimeError(msg)
    return result


def _validate_completion(
    run_root: Path,
    cohorts: Sequence[str],
) -> Path:
    completion = run_root / "completion_manifest.json"
    _require_regular_file(completion, label="Focused completion manifest")
    payload = json.loads(completion.read_text(encoding="utf-8"))
    expected_tasks = {(cohort, provider) for cohort in cohorts for provider in BMRS}
    observed_tasks = {
        (str(task.get("cohort")), str(task.get("provider")))
        for task in payload.get("tasks", [])
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != focused_runner.COMPLETION_CONTRACT
        or payload.get("config_sha256") != _sha256(focused_runner.CONFIG_PATH)
        or payload.get("task_count") != len(payload.get("tasks", []))
        or len(observed_tasks) != len(payload.get("tasks", []))
        or not set(cohorts) <= set(payload.get("cohorts", []))
        or not expected_tasks <= observed_tasks
    ):
        msg = "Focused K=500 completion manifest does not cover the requested grid."
        raise ValueError(msg)
    for task in payload["tasks"]:
        coordinate = (str(task.get("cohort")), str(task.get("provider")))
        if coordinate not in expected_tasks:
            continue
        record = task.get("manifest", {})
        expected_relative = (
            f"tasks/{coordinate[0]}/{coordinate[1]}/task_manifest.json"
        )
        if record.get("path") != expected_relative:
            msg = (
                "Completion task manifest path is invalid: "
                f"{coordinate[0]}/{coordinate[1]}"
            )
            raise ValueError(msg)
        path = run_root / expected_relative
        _require_regular_file(path, label="Focused completed-task manifest")
        if (
            record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            msg = f"Completion task manifest changed: {coordinate[0]}/{coordinate[1]}"
            raise ValueError(msg)
    return completion


def _valid_diagnostics(value: object, pair_count: int) -> bool:
    """Return whether one cohort's effect-status counters are exhaustive."""
    if not isinstance(value, dict) or set(value) != set(BMRS):
        return False
    for provider in BMRS:
        record = value.get(provider)
        if not isinstance(record, dict):
            return False
        rank_counts = [
            record.get("full_affine_rank_count"),
            record.get("rank_deficient_count"),
            record.get("rank_not_certified_underflow_count"),
        ]
        clipping_counts = [
            record.get("p_display_clipped_count"),
            record.get("by_display_clipped_count"),
            record.get("bh_display_clipped_count"),
        ]
        if (
            any(not isinstance(item, int) or item < 0 for item in rank_counts)
            or sum(rank_counts) != pair_count
            or any(
                not isinstance(item, int) or not 0 <= item <= pair_count
                for item in clipping_counts
            )
        ):
            return False
    return True


def _validate_raw_source_binding(
    *,
    manifest: dict[str, Any],
    frame: pd.DataFrame,
    run_root: Path,
    cohort: str,
) -> None:
    """Bind derived pair/statistic/effect columns to every raw provider output."""
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(BMRS):
        msg = f"Focused derived sources are incomplete: {cohort}"
        raise ValueError(msg)
    for provider in BMRS:
        source = (
            run_root
            / "tasks"
            / cohort
            / provider
            / "pairwise_interaction_results.csv"
        )
        if sources[provider] != _file_record(source, relative_to=run_root):
            msg = f"Focused derived source changed: {cohort}/{provider}"
            raise ValueError(msg)
        expected = _read_provider(run_root, cohort, provider)
        for column in expected.columns:
            if not _same_values(frame[column], expected[column]):
                msg = (
                    "Focused derived values differ from the raw provider output: "
                    f"{cohort}/{provider}/{column}"
                )
                raise ValueError(msg)


def validate_derived_root(
    output_root: Path,
    cohorts: Sequence[str],
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Validate receipt-bound derived tables before downstream consumption."""
    if output_root.is_symlink():
        msg = f"Focused postprocess root is unsafe: {output_root}"
        raise ValueError(msg)
    root_path = output_root / ROOT_MANIFEST_NAME
    _require_regular_file(root_path, label="Focused postprocess root manifest")
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    payload_cohorts = payload.get("cohorts")
    if (
        not isinstance(payload_cohorts, list)
        or any(not isinstance(item, str) or not item for item in payload_cohorts)
        or len(set(payload_cohorts)) != len(payload_cohorts)
    ):
        msg = "Focused postprocess cohort inventory is invalid."
        raise ValueError(msg)
    root_cohorts = tuple(payload_cohorts)
    records = {
        str(record.get("path")): record
        for record in payload.get("cohort_manifests", [])
    }
    expected_manifest_paths = {
        f"{cohort}/{COHORT_MANIFEST_NAME}" for cohort in root_cohorts
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != ROOT_CONTRACT
        or payload.get("effective_p_policy")
        != "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        or payload.get("probability_representation")
        != PROBABILITY_REPRESENTATION
        or payload.get("multiplicity")
        != {
            "primary": "benjamini-yekutieli",
            "nominal_sensitivity": "benjamini-hochberg",
        }
        or payload.get("cohort_count") != len(root_cohorts)
        or payload.get("provider_family_count")
        != len(root_cohorts) * len(BMRS)
        or not isinstance(payload.get("pair_count_per_provider"), int)
        or payload.get("pair_count_per_provider", -1) < 0
        or payload.get("reporting_threshold_selected") is not False
        or not set(cohorts) <= set(root_cohorts)
        or len(records) != len(payload.get("cohort_manifests", []))
        or set(records) != expected_manifest_paths
        or {path.name for path in output_root.iterdir()}
        != {ROOT_MANIFEST_NAME, *root_cohorts}
    ):
        msg = "Focused postprocess root manifest is invalid."
        raise ValueError(msg)
    if run_root is not None:
        completion_path = _validate_completion(run_root, root_cohorts)
        completion_record = payload.get("run_completion", {})
        if (
            completion_record.get("path") != completion_path.name
            or completion_record.get("bytes") != completion_path.stat().st_size
            or completion_record.get("sha256") != _sha256(completion_path)
        ):
            msg = "Focused postprocess root is bound to a different production run."
            raise ValueError(msg)
    validated_pair_count = 0
    for cohort in root_cohorts:
        relative_manifest = f"{cohort}/{COHORT_MANIFEST_NAME}"
        record = records.get(relative_manifest, {})
        cohort_root = output_root / cohort
        manifest_path = output_root / relative_manifest
        result_path = cohort_root / RESULT_NAME
        if not cohort_root.is_dir() or cohort_root.is_symlink():
            msg = f"Focused derived cohort directory is unsafe: {cohort}"
            raise ValueError(msg)
        _require_regular_file(manifest_path, label="Focused cohort manifest")
        _require_regular_file(result_path, label="Focused provider inference table")
        if (
            record.get("bytes") != manifest_path.stat().st_size
            or record.get("sha256") != _sha256(manifest_path)
        ):
            msg = f"Focused cohort manifest changed: {cohort}"
            raise ValueError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = manifest.get("output", {})
        pair_count = manifest.get("pair_count")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != DERIVATION_CONTRACT
            or manifest.get("cohort") != cohort
            or manifest.get("providers") != list(BMRS)
            or manifest.get("family")
            != "all-matched-unordered-pairs-excluding-same-base-M:N"
            or not isinstance(manifest.get("sources"), dict)
            or set(manifest.get("sources", {})) != set(BMRS)
            or not isinstance(pair_count, int)
            or pair_count < 0
            or manifest.get("multiplicity")
            != {
                "primary": "provider-specific-BY-over-complete-within-cohort-family",
                "nominal_sensitivity": (
                    "provider-specific-BH-over-complete-within-cohort-family"
                ),
            }
            or manifest.get("non_full_rank")
            != "retain-in-family-with-p-one-and-no-directional-effect"
            or manifest.get("probability_representation")
            != PROBABILITY_REPRESENTATION
            or manifest.get("reporting_threshold_selected") is not False
            or not _valid_diagnostics(manifest.get("diagnostics"), pair_count)
            or output.get("path") != f"{cohort}/{RESULT_NAME}"
            or output.get("bytes") != result_path.stat().st_size
            or output.get("sha256") != _sha256(result_path)
            or {path.name for path in cohort_root.iterdir()}
            != {RESULT_NAME, COHORT_MANIFEST_NAME}
        ):
            msg = f"Focused derived cohort output is invalid: {cohort}"
            raise ValueError(msg)
        frame = pd.read_csv(result_path, float_precision="round_trip")
        if len(frame) != pair_count:
            msg = f"Focused derived pair family is incomplete: {cohort}"
            raise ValueError(msg)
        validate_inference_frame(frame, cohort=cohort)
        validated_pair_count += pair_count
        if run_root is not None:
            _validate_raw_source_binding(
                manifest=manifest,
                frame=frame,
                run_root=run_root,
                cohort=cohort,
            )
    if payload["pair_count_per_provider"] != validated_pair_count:
        msg = "Focused postprocess aggregate pair count is invalid."
        raise ValueError(msg)
    return payload


def derive_cohort(run_root: Path, cohort: str, output_root: Path) -> dict[str, Any]:
    """Validate and derive all three provider families for one cohort."""
    frames = [_read_provider(run_root, cohort, provider) for provider in BMRS]
    pair_axis = frames[0].loc[:, ["gene_a", "gene_b"]]
    for provider, frame in zip(BMRS[1:], frames[1:], strict=True):
        if not pair_axis.equals(frame.loc[:, ["gene_a", "gene_b"]]):
            msg = f"Provider pair axes differ for {cohort}/{provider}."
            raise ValueError(msg)
    combined = pair_axis.copy()
    for frame in frames:
        combined = pd.concat([combined, frame.iloc[:, 2:]], axis=1)

    cohort_root = output_root / cohort
    cohort_root.mkdir(parents=True)
    result_path = cohort_root / RESULT_NAME
    combined.to_csv(result_path, index=False, lineterminator="\n")
    sources = {
        provider: _file_record(
            run_root
            / "tasks"
            / cohort
            / provider
            / "pairwise_interaction_results.csv",
            relative_to=run_root,
        )
        for provider in BMRS
    }
    diagnostics = {}
    for provider, frame in zip(BMRS, frames, strict=True):
        effects = frame[f"{provider}_effect_identifiability"].astype("string")
        log_p = frame[f"{provider}_log_p_value"].to_numpy(dtype=float)
        log_by = frame[f"{provider}_log_by_q_value"].to_numpy(dtype=float)
        log_bh = frame[f"{provider}_log_bh_q_value"].to_numpy(dtype=float)
        diagnostics[provider] = {
            "full_affine_rank_count": int(effects.eq("full-affine-rank").sum()),
            "rank_deficient_count": int(effects.eq("rank-deficient").sum()),
            "rank_not_certified_underflow_count": int(
                effects.eq("rank-not-certified-underflow").sum(),
            ),
            "p_display_clipped_count": int(
                (log_p < LOG_MIN_POSITIVE_FLOAT).sum(),
            ),
            "by_display_clipped_count": int(
                (log_by < LOG_MIN_POSITIVE_FLOAT).sum(),
            ),
            "bh_display_clipped_count": int(
                (log_bh < LOG_MIN_POSITIVE_FLOAT).sum(),
            ),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": DERIVATION_CONTRACT,
        "cohort": cohort,
        "pair_count": len(combined),
        "providers": list(BMRS),
        "family": "all-matched-unordered-pairs-excluding-same-base-M:N",
        "multiplicity": {
            "primary": "provider-specific-BY-over-complete-within-cohort-family",
            "nominal_sensitivity": (
                "provider-specific-BH-over-complete-within-cohort-family"
            ),
        },
        "direction": "rho-sign-after-nondirectional-profile-LRT",
        "non_full_rank": "retain-in-family-with-p-one-and-no-directional-effect",
        "probability_representation": PROBABILITY_REPRESENTATION,
        "diagnostics": diagnostics,
        "reporting_threshold_selected": False,
        "sources": sources,
        "output": _file_record(result_path, relative_to=output_root),
    }
    _write_atomic(
        cohort_root / COHORT_MANIFEST_NAME,
        _canonical_json(manifest) + b"\n",
    )
    return manifest


def derive(
    *,
    run_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> Path:
    """Publish a no-replace, fully receipt-bound provider inference tree."""
    completion = _validate_completion(run_root, cohorts)
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite postprocess root: {output_root}"
        raise FileExistsError(msg)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    manifests = [derive_cohort(run_root, cohort, staging) for cohort in cohorts]
    root = {
        "schema_version": SCHEMA_VERSION,
        "contract": ROOT_CONTRACT,
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "probability_representation": PROBABILITY_REPRESENTATION,
        "multiplicity": {
            "primary": "benjamini-yekutieli",
            "nominal_sensitivity": "benjamini-hochberg",
        },
        "run_completion": _file_record(completion, relative_to=run_root),
        "cohorts": list(cohorts),
        "cohort_count": len(cohorts),
        "provider_family_count": len(cohorts) * len(BMRS),
        "pair_count_per_provider": sum(
            int(manifest["pair_count"]) for manifest in manifests
        ),
        "reporting_threshold_selected": False,
        "cohort_manifests": [
            _file_record(
                staging / cohort / COHORT_MANIFEST_NAME,
                relative_to=staging,
            )
            for cohort in cohorts
        ],
    }
    _write_atomic(
        staging / ROOT_MANIFEST_NAME,
        _canonical_json(root) + b"\n",
    )
    staging.replace(output_root)
    validate_derived_root(output_root, cohorts, run_root=run_root)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohorts")
    return parser


def main() -> None:
    """Run focused provider-specific postprocessing."""
    args = _parser().parse_args()
    derive(
        run_root=args.run_root.resolve(),
        output_root=args.output_root.absolute(),
        cohorts=_parse_cohorts(args.cohorts),
    )


if __name__ == "__main__":
    main()

"""Build the focused DIALECT revision tables, runtime summary, and Figure 6.

This stage consumes only the validated matched K=500 fit, provider-specific
postprocessing, prespecified calibration, and frozen global reporting rule.  It
does not choose thresholds or modify scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import freeze_tcga_revision_reporting_rule as rule_module
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import run_tcga_revision_k500 as core
from analysis.prepare_tcga_revision_focused import validate_provider_root
from dialect.data.tcga import TCGA_COHORTS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
REPORT_CONTRACT: Final = "focused-revision-reporting-artifacts-v3"
HIGH_BURDEN_QUANTILE: Final = 0.99
EXPECTED_TUMOR_COUNT: Final = 10_433
FOCAL_BURDEN_COHORT: Final = "UCEC"
BURDEN_COUNT_MAX: Final = 1_000_000.0
BURDEN_LOG1P_MAX: Final = float(np.log1p(BURDEN_COUNT_MAX))
BURDEN_BIN_COUNT: Final = 24
BURDEN_BIN_COLUMNS: Final = (
    "cohort",
    "provider",
    "observed_log1p_bin_lower",
    "observed_log1p_bin_upper",
    "expected_log1p_bin_lower",
    "expected_log1p_bin_upper",
    "tumor_count",
)
COHORT_BURDEN_HISTOGRAM_COLUMNS: Final = (
    "cohort",
    "total_nonsynonymous_snv_events",
    "tumor_count",
)
BURDEN_SUMMARY_COLUMNS: Final = (
    "tumors",
    "burden_median",
    "burden_q25",
    "burden_q75",
    "burden_p90",
    "burden_p95",
    "burden_max",
    "high_burden_fraction",
)
BURDEN_SOURCE_POLICY: Final = (
    "cohort-burden-histogram-and-fixed-aggregate-figure-bins-only"
)
OVERLAP_COLUMNS: Final = (
    "cohort",
    "direction",
    "adjustment",
    "q_threshold",
    "mutsig_primary_rejection_count",
    "cbase_descriptive_crossing_count",
    "dig_descriptive_crossing_count",
    "mutsig_rejection_cbase_concordant_crossing_count",
    "mutsig_rejection_cbase_discordant_crossing_count",
    "mutsig_rejection_dig_concordant_crossing_count",
    "mutsig_rejection_dig_discordant_crossing_count",
)
RUNTIME_COLUMNS: Final = (
    "cohort",
    "provider",
    "pairwise_rows",
    "elapsed_seconds",
    "user_cpu_seconds",
    "system_cpu_seconds",
    "peak_rss_bytes",
)
FIT_DIAGNOSTIC_COLUMNS: Final = (
    "scope",
    "pairwise_rows",
    "converged_rows",
    "nonconverged_rows",
    "iterations_min",
    "iterations_median",
    "iterations_p95",
    "iterations_max",
    "minimum_last_ll_gain",
    "maximum_last_ll_gain",
    "maximum_fixed_point_residual",
    "maximum_kkt_residual",
    "full_affine_rank_rows",
    "rank_deficient_rows",
    "rank_not_certified_underflow_rows",
)
PROVIDER_LABELS: Final = {
    "cbase": "CBaSE",
    "dig": "DIG",
    "mutsig": "MutSig",
}
PROVIDER_COLORS: Final = {
    "cbase": "#6B7280",
    "dig": "#0072B2",
    "mutsig": "#D55E00",
}
ADJUSTMENT_COLUMNS: Final = {
    "benjamini-yekutieli": "by_q_value",
    "benjamini-hochberg": "bh_q_value",
}
LOG_ADJUSTMENT_COLUMNS: Final = {
    "benjamini-yekutieli": "log_by_q_value",
    "benjamini-hochberg": "log_bh_q_value",
}
ADJUSTMENT_LABELS: Final = {
    "benjamini-yekutieli": "BY",
    "benjamini-hochberg": "BH",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rule(
    rule_path: Path,
    calibration_root: Path,
    postprocess_root: Path,
) -> dict[str, Any]:
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    primary_q = rule.get("primary_q_threshold")
    sensitivity_q = rule.get("sensitivity_q_threshold")
    gate = rule.get("calibration_gate", {})
    gate_pass = gate.get("overall_gate_pass") if isinstance(gate, dict) else None
    expected_status = (
        rule_module.REPORTABLE_STATUS
        if gate_pass is True
        else rule_module.WITHHELD_STATUS
    )
    if (
        rule.get("schema_version") != SCHEMA_VERSION
        or rule.get("contract") != rule_module.RULE_CONTRACT
        or rule.get("scope")
        != "one-identical-rule-across-all-32-tcga-pan-cancer-atlas-cohorts"
        or rule.get("primary_provider") != "mutsig"
        or rule.get("continuity_provider") != "cbase"
        or rule.get("supplementary_providers") != ["dig"]
        or rule.get("test") != "chi-square-one-df-profile-lrt"
        or rule.get("effective_p_policy")
        != "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        or rule.get("multiplicity")
        != "provider-specific-complete-within-cohort-family"
        or rule.get("primary_adjustment") != "benjamini-yekutieli"
        or rule.get("sensitivity_adjustment") != "benjamini-hochberg"
        or not isinstance(primary_q, (int, float))
        or not isinstance(sensitivity_q, (int, float))
        or float(primary_q) != 0.01
        or float(sensitivity_q) != 0.01
        or rule.get("threshold_comparison") != "inclusive-less-than-or-equal"
        or rule.get("direction")
        != "primary-provider-rho-sign-after-nondirectional-rejection"
        or rule.get("direction_unavailable")
        != "retain-nondirectional-rejection-exclude-from-me-co-lists"
        or rule.get("thresholds_selected_from_observed_pairs") is not False
        or rule.get("calibration_interpretation")
        != "finite-scenario-stress-not-formal-uniform-FDR-proof"
        or not isinstance(gate_pass, bool)
        or gate.get("provider") != "mutsig"
        or gate.get("endpoint_unit") != "cohort-sentinel-pair-alpha"
        or gate.get("method")
        != (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        )
        or gate.get("endpoint_count") != 2_048
        or gate.get("familywise_error") != 0.05
        or gate.get("acceptance_upper_bounds")
        != {"0.01": 0.02, "0.05": 0.07}
        or rule.get("inference_status") != expected_status
        or (
            rule.get("inference_status") == rule_module.REPORTABLE_STATUS
            and rule.get("withheld_reason") is not None
        )
        or (
            rule.get("inference_status") == rule_module.WITHHELD_STATUS
            and rule.get("withheld_reason")
            != "prespecified-finite-scenario-calibration-gate-failed"
        )
        or rule.get("calibration_summary_sha256")
        != _sha256(calibration_root / calibration.SUMMARY_NAME)
        or rule.get("postprocess_manifest_sha256")
        != _sha256(postprocess_root / postprocess.ROOT_MANIFEST_NAME)
    ):
        msg = "Frozen reporting rule is invalid or bound to different inputs."
        raise ValueError(msg)
    return rule


def _require_reportable_rule(  # noqa: PLR0913
    *,
    calibration_root: Path,
    postprocess_root: Path,
    rule_path: Path,
    run_root: Path,
    provider_root: Path,
    action: str,
) -> dict[str, Any]:
    """Validate the affirmative gate before any association-capable validator."""
    gate_summary = rule_module.load_calibration_gate(calibration_root)
    if not rule_module.calibration_gate_pass(gate_summary):
        msg = f"Association-level {action} is withheld: calibration gate failed."
        raise RuntimeError(msg)

    if not rule_path.is_file():
        msg = f"Frozen reporting rule is missing: {rule_path}"
        raise FileNotFoundError(msg)
    if rule_path.is_symlink():
        msg = f"Frozen reporting rule is unsafe: {rule_path}"
        raise ValueError(msg)
    rule_preview = json.loads(rule_path.read_text(encoding="utf-8"))
    if not isinstance(rule_preview, dict):
        msg = "Frozen reporting rule must be a JSON object."
        raise TypeError(msg)
    preview_gate = rule_preview.get("calibration_gate")
    if not isinstance(preview_gate, dict):
        msg = "Frozen reporting rule lacks a calibration gate object."
        raise TypeError(msg)
    preview_gate_pass = preview_gate.get("overall_gate_pass")
    if not isinstance(preview_gate_pass, bool):
        msg = "Frozen reporting rule lacks an exact boolean calibration gate."
        raise TypeError(msg)
    if not preview_gate_pass:
        reason = rule_preview.get(
            "withheld_reason",
            "prespecified-calibration-gate-failure",
        )
        msg = f"Association-level {action} is withheld: {reason}"
        raise RuntimeError(msg)
    if rule_preview.get("inference_status") != rule_module.REPORTABLE_STATUS:
        msg = f"Association-level {action} is withheld: rule is not reportable."
        raise RuntimeError(msg)

    validated_summary = calibration.validate_summary(
        calibration_root,
        run_root=run_root,
        provider_root=provider_root,
    )
    if rule_module.calibration_gate_pass(validated_summary) is not True:
        msg = "Calibration gate changed during complete validation."
        raise RuntimeError(msg)
    rule = _load_rule(rule_path, calibration_root, postprocess_root)
    if rule["inference_status"] != rule_module.REPORTABLE_STATUS:
        msg = f"Association-level {action} is withheld: rule is not reportable."
        raise RuntimeError(msg)
    return rule


def _q_column(provider: str, adjustment: str) -> str:
    """Return the explicit provider q-value column for one named adjustment."""
    try:
        suffix = ADJUSTMENT_COLUMNS[adjustment]
    except KeyError as error:
        msg = f"Unsupported multiplicity adjustment: {adjustment}"
        raise ValueError(msg) from error
    return f"{provider}_{suffix}"


def _log_q_column(provider: str, adjustment: str) -> str:
    """Return the inferential log-q column for one named adjustment."""
    try:
        suffix = LOG_ADJUSTMENT_COLUMNS[adjustment]
    except KeyError as error:
        msg = f"Unsupported multiplicity adjustment: {adjustment}"
        raise ValueError(msg) from error
    return f"{provider}_{suffix}"


def _threshold_crossing(
    frame: pd.DataFrame,
    provider: str,
    adjustment: str,
    threshold: float,
) -> np.ndarray:
    """Return inclusive threshold crossings using inferential log q-values."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        msg = "A q-value threshold must be finite and lie in (0, 1]."
        raise ValueError(msg)
    return (
        frame[_log_q_column(provider, adjustment)].to_numpy(dtype=np.float64)
        <= math.log(threshold)
    )


def _threshold_label(adjustment: str, threshold: float) -> str:
    """Return one compact label derived from the frozen reporting rule."""
    try:
        label = ADJUSTMENT_LABELS[adjustment]
    except KeyError as error:
        msg = f"Unsupported multiplicity adjustment: {adjustment}"
        raise ValueError(msg) from error
    return f"{label} q <= {threshold:g}"


def _decision_prefix(provider: str, analysis: str) -> str:
    """Name exported decisions without promoting descriptive providers to inference."""
    if provider not in core.BMRS:
        msg = f"Unsupported background provider: {provider}"
        raise ValueError(msg)
    if analysis == "primary":
        if provider == "mutsig":
            return "mutsig_primary_rejection"
        return f"{provider}_descriptive_primary_rule_crossing"
    if analysis == "sensitivity":
        return f"{provider}_descriptive_sensitivity_rule_crossing"
    msg = f"Unsupported reporting analysis: {analysis}"
    raise ValueError(msg)


def summary_columns() -> tuple[str, ...]:
    """Return the exact public cohort-summary schema."""
    columns = [
        "cohort",
        "primary_adjustment",
        "primary_q_threshold",
        "sensitivity_adjustment",
        "sensitivity_q_threshold",
        "tumors",
        "selected_features",
        "tested_pairs",
        "same_base_pairs_excluded",
        "unfiltered_pair_count",
        "burden_median",
        "burden_q25",
        "burden_q75",
        "burden_p90",
        "burden_p95",
        "burden_max",
        "high_burden_fraction",
    ]
    for provider in core.BMRS:
        for label in ("primary", "sensitivity"):
            prefix = _decision_prefix(provider, label)
            columns.extend(
                [
                    f"{prefix}_total",
                    f"{prefix}_me",
                    f"{prefix}_co",
                    f"{prefix}_direction_unavailable",
                ],
            )
    return tuple(columns)


def top_primary_columns() -> tuple[str, ...]:
    """Return the exact public top-pair schema."""
    columns = [
        "cohort",
        "primary_adjustment",
        "primary_q_threshold",
        "direction",
        *postprocess.result_columns(),
    ]
    for provider in core.BMRS:
        prefix = _decision_prefix(provider, "primary")
        columns.extend(
            [
                prefix,
                f"{prefix}_direction_concordant",
                f"{prefix}_direction_discordant",
                f"{prefix}_direction_unavailable",
            ],
        )
    columns.extend(
        [
            "descriptive_direction_concordant_provider_count",
            "descriptive_direction_discordant_provider_count",
        ],
    )
    return tuple(columns)


def report_csv_columns() -> dict[str, tuple[str, ...]]:
    """Return every exact CSV schema allowed in the public report."""
    return {
        "cohort_burden_histogram.csv": COHORT_BURDEN_HISTOGRAM_COLUMNS,
        "figure6_burden_bins.csv": BURDEN_BIN_COLUMNS,
        "table_s5.csv": summary_columns(),
        "provider_overlap.csv": OVERLAP_COLUMNS,
        "top_primary_pairs.csv": top_primary_columns(),
        "runtime_summary.csv": RUNTIME_COLUMNS,
        "fit_diagnostics_summary.csv": FIT_DIAGNOSTIC_COLUMNS,
    }


def _calibration_gate_maxima(calibration_table: pd.DataFrame) -> pd.DataFrame:
    """Return each cohort's worst pair-resolved primary-gate endpoint."""
    required = {
        "cohort",
        "provider",
        "screen",
        "sentinel_pair_index",
        "threshold",
        "gate_endpoint",
        "clopper_pearson_upper_bound",
        "acceptance_upper_bound",
    }
    if not required.issubset(calibration_table.columns):
        msg = "Calibration table lacks pair-resolved gate columns."
        raise ValueError(msg)
    gate_mask = calibration_table["gate_endpoint"].astype("boolean").fillna(
        value=False,
    )
    gate = calibration_table.loc[gate_mask].copy()
    if gate.empty:
        msg = "Calibration table lacks primary gate endpoints."
        raise ValueError(msg)
    gate["sentinel_pair_index"] = gate["sentinel_pair_index"].astype(int)
    gate["threshold"] = gate["threshold"].astype(float)
    gate["clopper_pearson_upper_bound"] = pd.to_numeric(
        gate["clopper_pearson_upper_bound"],
        errors="raise",
    )
    gate["acceptance_upper_bound"] = pd.to_numeric(
        gate["acceptance_upper_bound"],
        errors="raise",
    )
    expected_pair_axis = set(range(32))
    rows = []
    for (cohort, threshold), group in gate.groupby(
        ["cohort", "threshold"],
        sort=True,
    ):
        acceptance = group["acceptance_upper_bound"].unique()
        if (
            not group["provider"].eq("mutsig").all()
            or not group["screen"].eq("marginal_lrt").all()
            or group["sentinel_pair_index"].duplicated().any()
            or set(group["sentinel_pair_index"]) != expected_pair_axis
            or len(acceptance) != 1
            or not np.isfinite(
                group["clopper_pearson_upper_bound"].to_numpy(dtype=float),
            ).all()
        ):
            msg = "Calibration worst-case summary has an incomplete endpoint family."
            raise ValueError(msg)
        rows.append(
            {
                "cohort": str(cohort),
                "threshold": float(threshold),
                "maximum_clopper_pearson_upper_bound": float(
                    group["clopper_pearson_upper_bound"].max(),
                ),
                "acceptance_upper_bound": float(acceptance[0]),
            },
        )
    return pd.DataFrame(rows)


def _read_inference(postprocess_root: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(
        postprocess_root / cohort / postprocess.RESULT_NAME,
        float_precision="round_trip",
    )
    postprocess.validate_inference_frame(frame, cohort=cohort)
    return frame


def _read_counts(provider_root: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(
        provider_root / "cohorts" / cohort / "count_matrix.csv",
        index_col=0,
    )
    values = frame.to_numpy(dtype=np.float64)
    if (
        frame.empty
        or not frame.index.is_unique
        or not frame.columns.is_unique
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        msg = f"Invalid focused count matrix: {cohort}"
        raise ValueError(msg)
    return frame


def _burden_values(
    provider_root: Path,
    cohorts: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        cohort: _read_counts(provider_root, cohort).sum(axis=1).to_numpy(dtype=float)
        for cohort in cohorts
    }


def _cohort_burden_histogram(
    values: Mapping[str, np.ndarray],
    cohorts: Sequence[str],
) -> pd.DataFrame:
    """Aggregate integer tumor burdens without retaining a sample axis."""
    if set(values) != set(cohorts):
        msg = "Cannot build an exact cohort burden histogram."
        raise ValueError(msg)
    rows: list[dict[str, int | str]] = []
    for cohort in cohorts:
        burdens = np.asarray(values[cohort])
        if (
            burdens.ndim != 1
            or len(burdens) == 0
            or not np.issubdtype(burdens.dtype, np.number)
            or not np.isfinite(burdens).all()
            or (burdens < 0).any()
            or not np.equal(burdens, np.floor(burdens)).all()
        ):
            msg = f"Invalid burdens for aggregate cohort histogram: {cohort}"
            raise ValueError(msg)
        burden_bins, tumor_counts = np.unique(
            burdens.astype(np.int64),
            return_counts=True,
        )
        rows.extend(
            {
                "cohort": cohort,
                "total_nonsynonymous_snv_events": int(burden),
                "tumor_count": int(tumor_count),
            }
            for burden, tumor_count in zip(
                burden_bins,
                tumor_counts,
                strict=True,
            )
        )
    return pd.DataFrame(rows, columns=COHORT_BURDEN_HISTOGRAM_COLUMNS)


def _burden_values_from_histogram(
    histogram: pd.DataFrame,
    cohorts: Sequence[str],
) -> dict[str, np.ndarray]:
    """Reconstruct each cohort's burden multiset from its aggregate histogram."""
    if (
        tuple(histogram.columns) != COHORT_BURDEN_HISTOGRAM_COLUMNS
        or histogram.empty
        or histogram.isna().any().any()
        or not pd.api.types.is_integer_dtype(
            histogram["total_nonsynonymous_snv_events"],
        )
        or not pd.api.types.is_integer_dtype(histogram["tumor_count"])
    ):
        msg = "Aggregate cohort burden histogram violates its exact schema."
        raise ValueError(msg)
    burden_bins = histogram["total_nonsynonymous_snv_events"].to_numpy(
        dtype=np.int64,
    )
    tumor_counts = histogram["tumor_count"].to_numpy(dtype=np.int64)
    cohort_axis = histogram["cohort"].tolist()
    observed_blocks = [
        cohort
        for index, cohort in enumerate(cohort_axis)
        if index == 0 or cohort != cohort_axis[index - 1]
    ]
    if (
        observed_blocks != list(cohorts)
        or (burden_bins < 0).any()
        or (tumor_counts < 1).any()
        or int(tumor_counts.sum()) != EXPECTED_TUMOR_COUNT
    ):
        msg = "Aggregate cohort burden histogram has invalid coordinates or counts."
        raise ValueError(msg)
    values: dict[str, np.ndarray] = {}
    for cohort in cohorts:
        selected = histogram["cohort"].eq(cohort).to_numpy()
        selected_bins = burden_bins[selected]
        selected_counts = tumor_counts[selected]
        if len(selected_bins) == 0 or not (np.diff(selected_bins) > 0).all():
            msg = f"Aggregate burden bins are not strictly ordered: {cohort}"
            raise ValueError(msg)
        values[cohort] = np.repeat(selected_bins, selected_counts).astype(float)
    return values


def _burden_summary_fields(
    burdens: np.ndarray,
    high_burden_threshold: float,
) -> dict[str, int | float]:
    """Compute every Table S5 burden field from one cohort burden multiset."""
    return {
        "tumors": len(burdens),
        "burden_median": float(np.median(burdens)),
        "burden_q25": float(np.quantile(burdens, 0.25)),
        "burden_q75": float(np.quantile(burdens, 0.75)),
        "burden_p90": float(np.quantile(burdens, 0.90)),
        "burden_p95": float(np.quantile(burdens, 0.95)),
        "burden_max": float(np.max(burdens)),
        "high_burden_fraction": float((burdens >= high_burden_threshold).mean()),
    }


def _validate_burden_histogram_summary(
    histogram: pd.DataFrame,
    summary: pd.DataFrame,
    manifest_threshold: object,
) -> float:
    """Bind every reported burden statistic to the public aggregate histogram."""
    cohorts = tuple(TCGA_COHORTS)
    values = _burden_values_from_histogram(histogram, cohorts)
    threshold = _high_burden_threshold(values)
    if (
        not isinstance(manifest_threshold, (int, float))
        or isinstance(manifest_threshold, bool)
        or not math.isfinite(float(manifest_threshold))
        or float(manifest_threshold) != threshold
        or summary["cohort"].tolist() != list(cohorts)
    ):
        msg = "Aggregate burden histogram does not reproduce the manifest threshold."
        raise ValueError(msg)
    expected = pd.DataFrame(
        [
            {
                "cohort": cohort,
                **_burden_summary_fields(values[cohort], threshold),
            }
            for cohort in cohorts
        ],
    )
    expected_values = expected.loc[:, BURDEN_SUMMARY_COLUMNS].to_numpy(dtype=float)
    reported_values = summary.loc[:, BURDEN_SUMMARY_COLUMNS].to_numpy(dtype=float)
    if not np.array_equal(reported_values, expected_values):
        msg = "Table S5 burden summary differs from its aggregate histogram."
        raise ValueError(msg)
    return threshold


def _high_burden_threshold(values: Mapping[str, np.ndarray]) -> float:
    pooled = np.concatenate(tuple(values.values()))
    if len(pooled) != EXPECTED_TUMOR_COUNT or not np.isfinite(pooled).all():
        msg = "Cannot define the pooled high-burden threshold."
        raise ValueError(msg)
    return float(np.quantile(pooled, HIGH_BURDEN_QUANTILE, method="higher"))


def _cohort_pair_counts(run_root: Path, cohort: str) -> dict[str, int]:
    """Read and reconcile aggregate pair counts from one frozen cohort contract."""
    path = run_root / "contracts" / f"{cohort}.json"
    if not path.is_file() or path.is_symlink():
        msg = f"Focused cohort contract is missing or unsafe: {path}"
        raise ValueError(msg)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        msg = f"Focused cohort contract must be a JSON object: {cohort}"
        raise TypeError(msg)
    features = contract.get("features")
    pair_policy = contract.get("pair_policy")
    if (
        contract.get("cohort") != cohort
        or contract.get("top_k") != 500
        or not isinstance(features, list)
        or len(features) != 500
        or len(set(features)) != len(features)
        or not all(isinstance(feature, str) and feature for feature in features)
        or not isinstance(pair_policy, dict)
    ):
        msg = f"Focused cohort contract has invalid pair coordinates: {cohort}"
        raise ValueError(msg)
    expected = core._pair_contract(features)  # noqa: SLF001
    if any(pair_policy.get(key) != value for key, value in expected.items()):
        msg = f"Focused cohort pair counts do not reconcile: {cohort}"
        raise ValueError(msg)
    tested = expected["row_count"]
    excluded = expected["same_base_pairs_excluded"]
    unfiltered = expected["unfiltered_row_count"]
    if tested + excluded != unfiltered or unfiltered != math.comb(len(features), 2):
        msg = f"Focused cohort pair universe is inconsistent: {cohort}"
        raise ValueError(msg)
    return {
        "selected_features": len(features),
        "tested_pairs": tested,
        "same_base_pairs_excluded": excluded,
        "unfiltered_pair_count": unfiltered,
    }


def _cohort_summary_row(  # noqa: PLR0913
    *,
    cohort: str,
    frame: pd.DataFrame,
    burdens: np.ndarray,
    high_burden_threshold: float,
    primary_adjustment: str,
    primary_q: float,
    sensitivity_adjustment: str,
    sensitivity_q: float,
    pair_counts: Mapping[str, int],
) -> dict[str, int | float | str]:
    expected_pair_keys = {
        "selected_features",
        "tested_pairs",
        "same_base_pairs_excluded",
        "unfiltered_pair_count",
    }
    if (
        set(pair_counts) != expected_pair_keys
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in pair_counts.values()
        )
        or pair_counts["tested_pairs"] != len(frame)
        or pair_counts["tested_pairs"] + pair_counts["same_base_pairs_excluded"]
        != pair_counts["unfiltered_pair_count"]
    ):
        msg = f"Reported pair counts do not match the fitted family: {cohort}"
        raise ValueError(msg)
    row: dict[str, int | float | str] = {
        "cohort": cohort,
        "primary_adjustment": ADJUSTMENT_LABELS[primary_adjustment],
        "primary_q_threshold": primary_q,
        "sensitivity_adjustment": ADJUSTMENT_LABELS[sensitivity_adjustment],
        "sensitivity_q_threshold": sensitivity_q,
        **pair_counts,
        **_burden_summary_fields(burdens, high_burden_threshold),
    }
    for provider in core.BMRS:
        directions = frame[f"{provider}_direction"].astype("string")
        for label, adjustment, threshold in (
            ("primary", primary_adjustment, primary_q),
            ("sensitivity", sensitivity_adjustment, sensitivity_q),
        ):
            crossing = _threshold_crossing(frame, provider, adjustment, threshold)
            prefix = _decision_prefix(provider, label)
            row[f"{prefix}_total"] = int(crossing.sum())
            row[f"{prefix}_me"] = int(
                (crossing & directions.eq("ME").to_numpy()).sum(),
            )
            row[f"{prefix}_co"] = int(
                (crossing & directions.eq("CO").to_numpy()).sum(),
            )
            row[f"{prefix}_direction_unavailable"] = int(
                (
                    crossing
                    & ~directions.isin(["ME", "CO"]).to_numpy()
                ).sum(),
            )
    return row


def _top_primary_pairs(
    frame: pd.DataFrame,
    *,
    cohort: str,
    primary_adjustment: str,
    primary_q: float,
    per_direction: int = 10,
) -> pd.DataFrame:
    provider = "mutsig"
    primary_log_column = _log_q_column(provider, primary_adjustment)
    rejected = _threshold_crossing(
        frame,
        provider,
        primary_adjustment,
        primary_q,
    )
    parts = []
    for direction in ("ME", "CO"):
        selected = frame.loc[
            rejected & frame[f"{provider}_direction"].eq(direction),
        ].copy()
        selected["absolute_mutsig_rho"] = selected["mutsig_rho"].abs()
        selected = selected.sort_values(
            [primary_log_column, "mutsig_log_p_value", "absolute_mutsig_rho"],
            ascending=[True, True, False],
            kind="stable",
        ).head(per_direction)
        selected.insert(0, "direction", direction)
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    result.insert(0, "cohort", cohort)
    result.insert(1, "primary_adjustment", ADJUSTMENT_LABELS[primary_adjustment])
    result.insert(2, "primary_q_threshold", primary_q)
    for provider in core.BMRS:
        crossing = _threshold_crossing(
            result,
            provider,
            primary_adjustment,
            primary_q,
        )
        provider_direction = result[f"{provider}_direction"].astype("string")
        directional = provider_direction.isin(["ME", "CO"])
        concordant = crossing & provider_direction.eq(result["direction"])
        discordant = crossing & directional & ~concordant
        prefix = _decision_prefix(provider, "primary")
        result[prefix] = crossing
        result[f"{prefix}_direction_concordant"] = concordant
        result[f"{prefix}_direction_discordant"] = discordant
        result[f"{prefix}_direction_unavailable"] = crossing & ~directional
    descriptive_providers = tuple(
        provider for provider in core.BMRS if provider != "mutsig"
    )
    result["descriptive_direction_concordant_provider_count"] = sum(
        result[
            f"{_decision_prefix(provider, 'primary')}_direction_concordant"
        ].astype(np.int8)
        for provider in descriptive_providers
    )
    result["descriptive_direction_discordant_provider_count"] = sum(
        result[
            f"{_decision_prefix(provider, 'primary')}_direction_discordant"
        ].astype(np.int8)
        for provider in descriptive_providers
    )
    return result.drop(columns="absolute_mutsig_rho")


def _overlap_rows(
    frame: pd.DataFrame,
    *,
    cohort: str,
    primary_adjustment: str,
    primary_q: float,
) -> list[dict[str, int | float | str]]:
    rows = []
    for direction in ("ME", "CO"):
        crossings = {
            provider: _threshold_crossing(
                frame,
                provider,
                primary_adjustment,
                primary_q,
            )
            for provider in core.BMRS
        }
        directions = {
            provider: frame[f"{provider}_direction"].astype("string")
            for provider in core.BMRS
        }
        masks = {
            provider: crossings[provider]
            & directions[provider].eq(direction).to_numpy()
            for provider in core.BMRS
        }
        mutsig = masks["mutsig"]
        cbase_opposite = (
            crossings["cbase"]
            & directions["cbase"].isin(["ME", "CO"]).to_numpy()
            & ~directions["cbase"].eq(direction).to_numpy()
        )
        dig_opposite = (
            crossings["dig"]
            & directions["dig"].isin(["ME", "CO"]).to_numpy()
            & ~directions["dig"].eq(direction).to_numpy()
        )

        rows.append(
            {
                "cohort": cohort,
                "direction": direction,
                "adjustment": ADJUSTMENT_LABELS[primary_adjustment],
                "q_threshold": primary_q,
                "mutsig_primary_rejection_count": int(masks["mutsig"].sum()),
                "cbase_descriptive_crossing_count": int(masks["cbase"].sum()),
                "dig_descriptive_crossing_count": int(masks["dig"].sum()),
                "mutsig_rejection_cbase_concordant_crossing_count": int(
                    (mutsig & masks["cbase"]).sum(),
                ),
                "mutsig_rejection_cbase_discordant_crossing_count": int(
                    (mutsig & cbase_opposite).sum(),
                ),
                "mutsig_rejection_dig_concordant_crossing_count": int(
                    (mutsig & masks["dig"]).sum(),
                ),
                "mutsig_rejection_dig_discordant_crossing_count": int(
                    (mutsig & dig_opposite).sum(),
                ),
            },
        )
    return rows


def _runtime_rows(run_root: Path, cohorts: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for cohort in cohorts:
        for provider in core.BMRS:
            path = run_root / "tasks" / cohort / provider / "task_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            usage = manifest["resource_usage"]
            rows.append(
                {
                    "cohort": cohort,
                    "provider": provider,
                    "pairwise_rows": manifest["pairwise_rows"],
                    "elapsed_seconds": usage["elapsed_seconds"],
                    "user_cpu_seconds": usage["user_cpu_seconds"],
                    "system_cpu_seconds": usage["system_cpu_seconds"],
                    "peak_rss_bytes": usage["peak_rss"]["bytes"],
                },
            )
    return rows


def _fit_diagnostic_rows(
    run_root: Path,
    cohorts: Sequence[str],
) -> list[dict[str, int | float | str]]:
    """Summarize every production pair-fit certificate without loading all rows."""
    columns = [
        "Fit Converged",
        "Fit Iterations",
        "Fit Last LL Gain",
        "Fit Fixed-Point Residual",
        "Fit KKT Residual",
        "Effect Identifiability",
    ]
    iterations: dict[str, list[np.ndarray]] = {provider: [] for provider in core.BMRS}
    aggregates = {
        provider: {
            "rows": 0,
            "converged": 0,
            "minimum_gain": np.inf,
            "maximum_gain": -np.inf,
            "maximum_fixed_point": 0.0,
            "maximum_kkt": 0.0,
            "full_rank": 0,
            "rank_deficient": 0,
            "rank_underflow": 0,
        }
        for provider in core.BMRS
    }
    for cohort in cohorts:
        for provider in core.BMRS:
            path = (
                run_root
                / "tasks"
                / cohort
                / provider
                / "pairwise_interaction_results.csv"
            )
            aggregate = aggregates[provider]
            for chunk in pd.read_csv(
                path,
                usecols=columns,
                chunksize=100_000,
                float_precision="round_trip",
            ):
                chunk_iterations = chunk["Fit Iterations"].to_numpy(dtype=np.int32)
                gains = chunk["Fit Last LL Gain"].to_numpy(dtype=float)
                fixed_point = chunk["Fit Fixed-Point Residual"].to_numpy(dtype=float)
                kkt = chunk["Fit KKT Residual"].to_numpy(dtype=float)
                convergence = chunk["Fit Converged"]
                effects = chunk["Effect Identifiability"].astype("string")
                if (
                    not convergence.isin([True, False]).all()
                    or (chunk_iterations < 0).any()
                    or not np.isfinite(gains).all()
                    or (gains < -1e-12).any()
                    or not np.isfinite(fixed_point).all()
                    or (fixed_point < 0).any()
                    or not np.isfinite(kkt).all()
                    or (kkt < 0).any()
                    or not effects.isin(
                        [
                            "full-affine-rank",
                            "rank-deficient",
                            "rank-not-certified-underflow",
                        ],
                    ).all()
                ):
                    msg = f"Invalid production fit diagnostics: {cohort}/{provider}"
                    raise ValueError(msg)
                converged = convergence.to_numpy(dtype=bool)
                iterations[provider].append(chunk_iterations)
                aggregate["rows"] += len(chunk)
                aggregate["converged"] += int(converged.sum())
                aggregate["minimum_gain"] = min(
                    float(aggregate["minimum_gain"]),
                    float(gains.min()),
                )
                aggregate["maximum_gain"] = max(
                    float(aggregate["maximum_gain"]),
                    float(gains.max()),
                )
                aggregate["maximum_fixed_point"] = max(
                    float(aggregate["maximum_fixed_point"]),
                    float(fixed_point.max()),
                )
                aggregate["maximum_kkt"] = max(
                    float(aggregate["maximum_kkt"]),
                    float(kkt.max()),
                )
                aggregate["full_rank"] += int(effects.eq("full-affine-rank").sum())
                aggregate["rank_deficient"] += int(effects.eq("rank-deficient").sum())
                aggregate["rank_underflow"] += int(
                    effects.eq("rank-not-certified-underflow").sum(),
                )

    def summarize(
        scope: str,
        providers: Sequence[str],
    ) -> dict[str, int | float | str]:
        values = np.concatenate(
            [array for provider in providers for array in iterations[provider]],
        )
        selected = [aggregates[provider] for provider in providers]
        rows = sum(int(item["rows"]) for item in selected)
        converged = sum(int(item["converged"]) for item in selected)
        return {
            "scope": scope,
            "pairwise_rows": rows,
            "converged_rows": converged,
            "nonconverged_rows": rows - converged,
            "iterations_min": int(values.min()),
            "iterations_median": float(np.quantile(values, 0.5)),
            "iterations_p95": float(np.quantile(values, 0.95)),
            "iterations_max": int(values.max()),
            "minimum_last_ll_gain": min(
                float(item["minimum_gain"]) for item in selected
            ),
            "maximum_last_ll_gain": max(
                float(item["maximum_gain"]) for item in selected
            ),
            "maximum_fixed_point_residual": max(
                float(item["maximum_fixed_point"]) for item in selected
            ),
            "maximum_kkt_residual": max(
                float(item["maximum_kkt"]) for item in selected
            ),
            "full_affine_rank_rows": sum(int(item["full_rank"]) for item in selected),
            "rank_deficient_rows": sum(
                int(item["rank_deficient"]) for item in selected
            ),
            "rank_not_certified_underflow_rows": sum(
                int(item["rank_underflow"]) for item in selected
            ),
        }

    return [
        summarize("all", core.BMRS),
        *(summarize(provider, (provider,)) for provider in core.BMRS),
    ]


def _pmf_mean(pmf: Mapping[int, float]) -> float:
    return float(sum(int(key) * float(value) for key, value in pmf.items()))


def _expected_selected_burden(
    *,
    run_root: Path,
    cohort: str,
    provider: str,
) -> tuple[np.ndarray, np.ndarray]:
    contract = json.loads(
        (run_root / "contracts" / f"{cohort}.json").read_text(encoding="utf-8"),
    )
    counts, pmfs = core._load_frozen_scientific_inputs(contract, provider)  # noqa: SLF001
    features = tuple(contract["features"])
    selected = counts.loc[:, features]
    task_root = run_root / "tasks" / cohort / provider
    single_path = task_root / "single_gene_results.csv"
    task_manifest = json.loads(
        (task_root / "task_manifest.json").read_text(encoding="utf-8"),
    )
    single_record = task_manifest.get("outputs", {}).get(single_path.name, {})
    if (
        single_record.get("path") != single_path.name
        or single_record.get("bytes") != single_path.stat().st_size
        or single_record.get("sha256") != _sha256(single_path)
    ):
        msg = (
            "Single-event result is not bound by its task manifest: "
            f"{cohort}/{provider}"
        )
        raise ValueError(msg)
    single = pd.read_csv(single_path, float_precision="round_trip")
    if single["Gene Name"].tolist() != list(features):
        msg = f"Single-event axis changed: {cohort}/{provider}"
        raise ValueError(msg)
    pi = single["Pi"].to_numpy(dtype=float)
    expected = np.zeros(len(selected), dtype=float)
    for feature, fitted_pi in zip(features, pi, strict=True):
        background = pmfs[feature]
        if isinstance(background, dict):
            expected += _pmf_mean(background) + fitted_pi
        else:
            if len(background) != len(selected):
                msg = f"Sample-specific PMF axis changed: {cohort}/{provider}"
                raise ValueError(msg)
            expected += np.fromiter(
                (_pmf_mean(item) for item in background),
                dtype=float,
                count=len(selected),
            ) + fitted_pi
    return selected.sum(axis=1).to_numpy(dtype=float), expected


def _aggregate_burden_bins(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    provider: str,
) -> pd.DataFrame:
    """Aggregate tumor burdens into one fixed, non-sample-level 2D grid."""
    observed_values = np.asarray(observed, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    if (
        provider not in core.BMRS
        or observed_values.ndim != 1
        or expected_values.shape != observed_values.shape
        or not np.isfinite(observed_values).all()
        or not np.isfinite(expected_values).all()
        or (observed_values < 0).any()
        or (expected_values < 0).any()
    ):
        msg = f"Invalid burden values for aggregate Figure 6 bins: {provider}"
        raise ValueError(msg)
    observed_log = np.log1p(observed_values)
    expected_log = np.log1p(expected_values)
    if (
        (observed_log > BURDEN_LOG1P_MAX).any()
        or (expected_log > BURDEN_LOG1P_MAX).any()
    ):
        msg = "Figure 6 burden exceeds the frozen aggregate-bin domain."
        raise ValueError(msg)
    edges = np.linspace(0.0, BURDEN_LOG1P_MAX, BURDEN_BIN_COUNT + 1)
    histogram, observed_edges, expected_edges = np.histogram2d(
        observed_log,
        expected_log,
        bins=(edges, edges),
    )
    rows = []
    for observed_index, expected_index in np.argwhere(histogram > 0):
        rows.append(
            {
                "cohort": FOCAL_BURDEN_COHORT,
                "provider": provider,
                "observed_log1p_bin_lower": observed_edges[observed_index],
                "observed_log1p_bin_upper": observed_edges[observed_index + 1],
                "expected_log1p_bin_lower": expected_edges[expected_index],
                "expected_log1p_bin_upper": expected_edges[expected_index + 1],
                "tumor_count": int(histogram[observed_index, expected_index]),
            },
        )
    return pd.DataFrame(rows, columns=BURDEN_BIN_COLUMNS)


def _figure6_burden_bins(run_root: Path) -> pd.DataFrame:
    """Return fixed aggregate bins underlying Figure 6 panel A."""
    observed_axis: np.ndarray | None = None
    frames = []
    for provider in core.BMRS:
        observed, expected = _expected_selected_burden(
            run_root=run_root,
            cohort=FOCAL_BURDEN_COHORT,
            provider=provider,
        )
        if observed_axis is None:
            observed_axis = observed
        elif not np.array_equal(
            observed_axis,
            observed,
        ):
            msg = "Observed selected burden differs between providers."
            raise ValueError(msg)
        frames.append(_aggregate_burden_bins(observed, expected, provider=provider))
    if observed_axis is None:
        msg = "Figure 6 aggregate burden source could not be constructed."
        raise RuntimeError(msg)
    return pd.concat(frames, ignore_index=True)


def _plot_figure6(  # noqa: PLR0913
    *,
    burden_bins: pd.DataFrame,
    summary: pd.DataFrame,
    overlap: pd.DataFrame,
    calibration_table: pd.DataFrame,
    primary_adjustment: str,
    primary_q: float,
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "font.family": "Arial",
            "font.size": 9,
            "figure.dpi": 150,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )
    figure = plt.figure(figsize=(7.5, 8.75), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1.15, 1.0),
        height_ratios=(1.0, 1.0, 1.0),
    )
    ax_b = figure.add_subplot(grid[:, 0])
    ax_a = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_d = figure.add_subplot(grid[2, 1])

    ax = ax_a
    for provider in core.BMRS:
        selected_bins = burden_bins.loc[burden_bins["provider"].eq(provider)]
        observed = np.exp(
            (
                selected_bins["observed_log1p_bin_lower"]
                + selected_bins["observed_log1p_bin_upper"]
            )
            / 2,
        )
        expected = np.exp(
            (
                selected_bins["expected_log1p_bin_lower"]
                + selected_bins["expected_log1p_bin_upper"]
            )
            / 2,
        )
        counts = selected_bins["tumor_count"].to_numpy(dtype=float)
        ax.scatter(
            observed,
            expected,
            s=8 + 7 * np.sqrt(counts),
            alpha=0.55,
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
        )
    maximum = float(max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot([1, maximum], [1, maximum], color="#111827", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed selected events per tumor + 1")
    ax.set_ylabel("Model-expected events per tumor + 1")
    ax.set_title("A  UCEC burden across background models", loc="left")
    ax.legend(frameon=False)

    ax = ax_b
    ordered = summary.sort_values(
        f"{_decision_prefix('mutsig', 'primary')}_co",
        ascending=True,
    )
    positions = np.arange(len(ordered))
    offsets = {"cbase": -0.22, "dig": 0.0, "mutsig": 0.22}
    provider_counts = []
    for provider in core.BMRS:
        counts = ordered[
            f"{_decision_prefix(provider, 'primary')}_co"
        ].to_numpy(dtype=float)
        provider_counts.append(counts)
        ax.scatter(
            np.log10(counts + 1),
            positions + offsets[provider],
            s=24,
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
        )
    ax.set_yticks(positions, ordered["cohort"])
    maximum_count = max(float(values.max()) for values in provider_counts)
    candidate_ticks = np.asarray([0, 1, 10, 100, 1_000, 10_000, 100_000])
    count_ticks = candidate_ticks[candidate_ticks <= maximum_count]
    if len(count_ticks) == 0 or count_ticks[-1] < maximum_count:
        count_ticks = np.append(count_ticks, int(np.ceil(maximum_count)))
    ax.set_xticks(
        np.log10(count_ticks + 1),
        [f"{value:,}" for value in count_ticks],
    )
    threshold_label = _threshold_label(primary_adjustment, primary_q)
    ax.set_xlabel(f"CO-direction pairs at {threshold_label}")
    ax.set_title(
        "B  Primary MutSig rejections and descriptive crossings",
        loc="left",
    )
    ax.grid(axis="x", alpha=0.2)

    ax = ax_c
    marginal = calibration_table.loc[
        calibration_table["screen"].eq("marginal_lrt"),
    ]
    cell_rates = marginal.groupby(
        ["cohort", "provider", "threshold"],
        as_index=False,
    )["rate"].mean()
    for provider in core.BMRS:
        selected = cell_rates.loc[cell_rates["provider"].eq(provider)]
        for threshold, group in selected.groupby("threshold"):
            x = np.full(len(group), float(threshold))
            ax.scatter(
                x,
                group["rate"],
                s=22,
                alpha=0.75,
                color=PROVIDER_COLORS[provider],
            )
        means = selected.groupby("threshold", as_index=False)["rate"].mean()
        ax.plot(
            means["threshold"],
            means["rate"],
            color=PROVIDER_COLORS[provider],
            label=f"{PROVIDER_LABELS[provider]} cohort mean",
        )
    gate_maxima = _calibration_gate_maxima(calibration_table)
    ax.scatter(
        gate_maxima["threshold"],
        gate_maxima["maximum_clopper_pearson_upper_bound"],
        marker="^",
        facecolors="none",
        edgecolors=PROVIDER_COLORS["mutsig"],
        s=22,
        label="MutSig cohort worst pair",
    )
    acceptance = gate_maxima.groupby("threshold", as_index=False)[
        "acceptance_upper_bound"
    ].first()
    ax.scatter(
        acceptance["threshold"],
        acceptance["acceptance_upper_bound"],
        marker="_",
        color="#111827",
        s=70,
        linewidths=1.4,
        label="Acceptance bound",
    )
    calibration_values = [
        float(cell_rates["rate"].max()),
        float(gate_maxima["maximum_clopper_pearson_upper_bound"].max()),
        float(gate_maxima["acceptance_upper_bound"].max()),
    ]
    limits = [0, max(0.06, max(calibration_values) * 1.08)]
    ax.plot(limits, limits, color="#111827", linewidth=0.8, linestyle="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Nominal p-value threshold")
    ax.set_ylabel("Rejection rate / upper bound")
    ax.set_title("C  Fitted-null calibration; cohort worst pair", loc="left")
    ax.legend(frameon=False, ncols=2, loc="upper left")

    ax = ax_d
    aggregate = overlap.groupby("direction", sort=False)[
        [
            "mutsig_primary_rejection_count",
            "mutsig_rejection_cbase_concordant_crossing_count",
            "mutsig_rejection_dig_concordant_crossing_count",
            "mutsig_rejection_cbase_discordant_crossing_count",
            "mutsig_rejection_dig_discordant_crossing_count",
        ]
    ].sum()
    directions = ["ME", "CO"]
    positions = np.arange(len(directions), dtype=float)
    widths = 0.24
    series = (
        (
            "mutsig_primary_rejection_count",
            "Primary MutSig rejections",
            PROVIDER_COLORS["mutsig"],
        ),
        (
            "mutsig_rejection_cbase_concordant_crossing_count",
            "With descriptive CBaSE crossing, same direction",
            PROVIDER_COLORS["cbase"],
        ),
        (
            "mutsig_rejection_dig_concordant_crossing_count",
            "With descriptive DIG crossing, same direction",
            PROVIDER_COLORS["dig"],
        ),
    )
    for offset, (column, label, color) in zip(
        (-widths, 0.0, widths),
        series,
        strict=True,
    ):
        values = aggregate.reindex(directions, fill_value=0)[column].to_numpy()
        ax.bar(positions + offset, values, width=widths, color=color, label=label)
    cbase_discordant = int(
        aggregate["mutsig_rejection_cbase_discordant_crossing_count"].sum(),
    )
    dig_discordant = int(
        aggregate["mutsig_rejection_dig_discordant_crossing_count"].sum(),
    )
    ax.text(
        0.02,
        0.98,
        (
            "Opposite-direction crossings among primary MutSig rejections: "
            f"CBaSE {cbase_discordant:,}; DIG {dig_discordant:,}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.set_xticks(positions, ["Mutual exclusivity", "Co-occurrence"])
    ax.set_ylabel(f"Pairs across {len(summary)} cohorts ({threshold_label})")
    ax.set_title("D  Direction-concordant provider overlap", loc="left")
    ax.margins(y=0.35)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    ax.grid(axis="y", alpha=0.2)

    figure.savefig(
        output,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        msg = "Figure 6 rendering failed."
        raise RuntimeError(msg)


def _report_csv_frames(
    *,
    run_root: Path,
    provider_root: Path,
    postprocess_root: Path,
    rule: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], float]:
    """Recompute every public report CSV from its receipt-bound source."""
    cohorts = tuple(TCGA_COHORTS)
    raw_burdens = _burden_values(provider_root, cohorts)
    burden_histogram = _cohort_burden_histogram(raw_burdens, cohorts)
    burdens = _burden_values_from_histogram(burden_histogram, cohorts)
    burden_threshold = _high_burden_threshold(burdens)
    primary_adjustment = str(rule["primary_adjustment"])
    sensitivity_adjustment = str(rule["sensitivity_adjustment"])
    primary_q = float(rule["primary_q_threshold"])
    sensitivity_q = float(rule["sensitivity_q_threshold"])
    summary_rows = []
    overlap_rows = []
    top_frames = []
    for cohort in cohorts:
        frame = _read_inference(postprocess_root, cohort)
        pair_counts = _cohort_pair_counts(run_root, cohort)
        summary_rows.append(
            _cohort_summary_row(
                cohort=cohort,
                frame=frame,
                burdens=burdens[cohort],
                high_burden_threshold=burden_threshold,
                primary_adjustment=primary_adjustment,
                primary_q=primary_q,
                sensitivity_adjustment=sensitivity_adjustment,
                sensitivity_q=sensitivity_q,
                pair_counts=pair_counts,
            ),
        )
        overlap_rows.extend(
            _overlap_rows(
                frame,
                cohort=cohort,
                primary_adjustment=primary_adjustment,
                primary_q=primary_q,
            ),
        )
        top_frames.append(
            _top_primary_pairs(
                frame,
                cohort=cohort,
                primary_adjustment=primary_adjustment,
                primary_q=primary_q,
            ),
        )
    top_pairs = pd.concat(top_frames, ignore_index=True)
    if tuple(top_pairs.columns) != top_primary_columns():
        msg = "Generated top-pair table violates its frozen public schema."
        raise RuntimeError(msg)
    frames = {
        "cohort_burden_histogram.csv": burden_histogram,
        "figure6_burden_bins.csv": _figure6_burden_bins(run_root),
        "table_s5.csv": pd.DataFrame(summary_rows, columns=summary_columns()),
        "provider_overlap.csv": pd.DataFrame(overlap_rows, columns=OVERLAP_COLUMNS),
        "top_primary_pairs.csv": top_pairs,
        "runtime_summary.csv": pd.DataFrame(
            _runtime_rows(run_root, cohorts),
            columns=RUNTIME_COLUMNS,
        ),
        "fit_diagnostics_summary.csv": pd.DataFrame(
            _fit_diagnostic_rows(run_root, cohorts),
            columns=FIT_DIAGNOSTIC_COLUMNS,
        ),
    }
    expected_schemas = report_csv_columns()
    if any(
        tuple(frame.columns) != expected_schemas[name]
        for name, frame in frames.items()
    ):
        msg = "Generated report CSV violates its exact public schema."
        raise RuntimeError(msg)
    return frames, burden_threshold


def _table_s5_tex(
    summary: pd.DataFrame,
    *,
    primary_adjustment: str,
    primary_q: float,
    sensitivity_adjustment: str,
    sensitivity_q: float,
) -> str:
    """Render the deterministic Table S5 LaTeX source."""
    burden_columns = [
        "cohort",
        "tumors",
        "selected_features",
        "tested_pairs",
        "same_base_pairs_excluded",
        "unfiltered_pair_count",
        "burden_median",
        "burden_q25",
        "burden_q75",
        "burden_p95",
        "burden_max",
        "high_burden_fraction",
    ]
    burden_labels = [
        "Cohort",
        "Tumors",
        "Selected features",
        "Tested pairs",
        "Same-base pairs excluded",
        "Unfiltered pairs",
        "Median",
        "Q25",
        "Q75",
        "P95",
        "Max",
        "High fraction",
    ]
    burden_table = summary.loc[:, burden_columns].set_axis(burden_labels, axis=1)
    primary_label = _threshold_label(primary_adjustment, primary_q)
    sensitivity_label = _threshold_label(sensitivity_adjustment, sensitivity_q)
    call_rows = [
        {
            "Cohort": row["cohort"],
            "Background": PROVIDER_LABELS[provider],
            "Primary-rule interpretation": (
                "primary MutSig rejection"
                if provider == "mutsig"
                else "descriptive threshold crossing"
            ),
            f"{primary_label} decisions total": row[
                f"{_decision_prefix(provider, 'primary')}_total"
            ],
            f"{primary_label} decisions ME": row[
                f"{_decision_prefix(provider, 'primary')}_me"
            ],
            f"{primary_label} decisions CO": row[
                f"{_decision_prefix(provider, 'primary')}_co"
            ],
            f"{primary_label} decisions direction unavailable": row[
                f"{_decision_prefix(provider, 'primary')}_direction_unavailable"
            ],
            f"{sensitivity_label} sensitivity crossings total": row[
                f"{_decision_prefix(provider, 'sensitivity')}_total"
            ],
            f"{sensitivity_label} sensitivity crossings ME": row[
                f"{_decision_prefix(provider, 'sensitivity')}_me"
            ],
            f"{sensitivity_label} sensitivity crossings CO": row[
                f"{_decision_prefix(provider, 'sensitivity')}_co"
            ],
            f"{sensitivity_label} sensitivity crossings direction unavailable": row[
                f"{_decision_prefix(provider, 'sensitivity')}_direction_unavailable"
            ],
        }
        for row in summary.to_dict(orient="records")
        for provider in core.BMRS
    ]
    call_table = pd.DataFrame(call_rows)
    return (
        "\\textbf{A. Cohort and mutation-burden summary}\\par\n"
        + burden_table.to_latex(index=False, float_format="%.3f", escape=True)
        + "\n\\medskip\n"
        + (
            "\\textbf{B. Primary MutSig rejections and descriptive background "
            "crossings by fitted direction}\\par\n"
        )
        + call_table.to_latex(index=False, escape=True, longtable=True)
    )


def validate_report(  # noqa: PLR0913
    output_root: Path,
    *,
    run_root: Path | None = None,
    provider_root: Path | None = None,
    postprocess_root: Path | None = None,
    calibration_root: Path | None = None,
    rule_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable reporting tree and its external input bindings."""
    external_inputs = (
        run_root,
        provider_root,
        postprocess_root,
        calibration_root,
        rule_path,
    )
    if not all(path is not None for path in external_inputs):
        msg = "All external reporting inputs are required before report access."
        raise ValueError(msg)
    rule = _require_reportable_rule(
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        rule_path=rule_path,
        run_root=run_root,
        provider_root=provider_root,
        action="report validation",
    )
    manifest_path = output_root / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_outputs = {
        "cohort_burden_histogram.csv",
        "figure6_burden_bins.csv",
        "table_s5.csv",
        "provider_overlap.csv",
        "top_primary_pairs.csv",
        "runtime_summary.csv",
        "fit_diagnostics_summary.csv",
        "table_s5.tex",
        "figure6.pdf",
    }
    records = manifest.get("outputs", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != REPORT_CONTRACT
        or manifest.get("cohorts") != list(TCGA_COHORTS)
        or manifest.get("primary_provider") != "mutsig"
        or manifest.get("inference_status") != rule_module.REPORTABLE_STATUS
        or manifest.get("effective_p_policy")
        != "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        or manifest.get("primary_adjustment") != "benjamini-yekutieli"
        or manifest.get("sensitivity_adjustment") != "benjamini-hochberg"
        or manifest.get("primary_q_threshold") != 0.01
        or manifest.get("sensitivity_q_threshold") != 0.01
        or manifest.get("provider_overlap")
        != "direction-concordant-descriptive-only-not-an-inferential-vote"
        or manifest.get("threshold_decision_scale") != "natural-log-q-values"
        or manifest.get("probability_representation")
        != postprocess.PROBABILITY_REPRESENTATION
        or manifest.get("sample_level_rows_included") is not False
        or manifest.get("burden_source_policy") != BURDEN_SOURCE_POLICY
        or set(manifest.get("inputs", {}))
        != {
            "run_completion",
            "provider_manifest",
            "postprocess_manifest",
            "calibration_summary",
            "reporting_rule",
        }
        or manifest.get("high_burden_definition", {}).get("pooled_tumor_count")
        != EXPECTED_TUMOR_COUNT
        or manifest.get("high_burden_definition", {}).get("source")
        != "cohort_burden_histogram.csv"
        or set(records) != expected_outputs
        or {path.name for path in output_root.iterdir()}
        != {*expected_outputs, "report_manifest.json"}
    ):
        msg = "Focused reporting manifest or inventory is invalid."
        raise ValueError(msg)
    for name in expected_outputs:
        path = output_root / name
        record = records[name]
        if (
            record.get("path") != name
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            msg = f"Focused reporting output changed: {name}"
            raise ValueError(msg)
    validate_provider_root(provider_root, TCGA_COHORTS)
    postprocess.validate_derived_root(
        postprocess_root,
        TCGA_COHORTS,
        run_root=run_root,
    )
    input_paths = {
        "run_completion": (
            run_root / "completion_manifest.json",
            run_root,
        ),
        "provider_manifest": (
            provider_root / "provider_manifest.json",
            provider_root,
        ),
        "postprocess_manifest": (
            postprocess_root / postprocess.ROOT_MANIFEST_NAME,
            postprocess_root,
        ),
        "calibration_summary": (
            calibration_root / calibration.SUMMARY_NAME,
            calibration_root,
        ),
    }
    for name, (path, root) in input_paths.items():
        if manifest["inputs"][name] != _file_record(path, relative_to=root):
            msg = f"Focused report is bound to a different input: {name}"
            raise ValueError(msg)
    expected_rule = {
        "path": rule_path.name,
        "bytes": rule_path.stat().st_size,
        "sha256": _sha256(rule_path),
    }
    if manifest["inputs"]["reporting_rule"] != expected_rule:
        msg = "Focused report is bound to a different reporting rule."
        raise ValueError(msg)
    burden_histogram = pd.read_csv(output_root / "cohort_burden_histogram.csv")
    summary = pd.read_csv(output_root / "table_s5.csv")
    overlap = pd.read_csv(output_root / "provider_overlap.csv")
    top_pairs = pd.read_csv(output_root / "top_primary_pairs.csv")
    runtime = pd.read_csv(output_root / "runtime_summary.csv")
    fit_diagnostics = pd.read_csv(output_root / "fit_diagnostics_summary.csv")
    figure_burden = pd.read_csv(output_root / "figure6_burden_bins.csv")
    frames = {
        "cohort_burden_histogram.csv": burden_histogram,
        "figure6_burden_bins.csv": figure_burden,
        "table_s5.csv": summary,
        "provider_overlap.csv": overlap,
        "top_primary_pairs.csv": top_pairs,
        "runtime_summary.csv": runtime,
        "fit_diagnostics_summary.csv": fit_diagnostics,
    }
    expected_schemas = report_csv_columns()
    if any(
        tuple(frame.columns) != expected_schemas[name]
        for name, frame in frames.items()
    ):
        msg = "Focused reporting CSV violates its exact public schema."
        raise ValueError(msg)
    histogram_burden_threshold = _validate_burden_histogram_summary(
        burden_histogram,
        summary,
        manifest.get("high_burden_definition", {}).get("threshold"),
    )
    expected_frames, expected_burden_threshold = _report_csv_frames(
        run_root=run_root,
        provider_root=provider_root,
        postprocess_root=postprocess_root,
        rule=rule,
    )
    if any(
        (output_root / name).read_bytes() != _csv_bytes(expected)
        for name, expected in expected_frames.items()
    ):
        msg = "Focused reporting CSV differs from exact source recomputation."
        raise ValueError(msg)
    expected_table_tex = _table_s5_tex(
        expected_frames["table_s5.csv"],
        primary_adjustment=str(rule["primary_adjustment"]),
        primary_q=float(rule["primary_q_threshold"]),
        sensitivity_adjustment=str(rule["sensitivity_adjustment"]),
        sensitivity_q=float(rule["sensitivity_q_threshold"]),
    ).encode()
    if (output_root / "table_s5.tex").read_bytes() != expected_table_tex:
        msg = "Focused Table S5 LaTeX differs from exact source recomputation."
        raise ValueError(msg)
    with tempfile.TemporaryDirectory(
        prefix="dialect-report-verify-",
        dir=output_root.parent,
    ) as temporary:
        expected_figure = Path(temporary) / "figure6.pdf"
        calibration_table = pd.read_csv(
            calibration_root / calibration.SUMMARY_TABLE_NAME,
            float_precision="round_trip",
        )
        _plot_figure6(
            burden_bins=expected_frames["figure6_burden_bins.csv"],
            summary=expected_frames["table_s5.csv"],
            overlap=expected_frames["provider_overlap.csv"],
            calibration_table=calibration_table,
            primary_adjustment=str(rule["primary_adjustment"]),
            primary_q=float(rule["primary_q_threshold"]),
            output=expected_figure,
        )
        if (output_root / "figure6.pdf").read_bytes() != expected_figure.read_bytes():
            msg = "Focused Figure 6 PDF differs from exact source recomputation."
            raise ValueError(msg)
    if (
        manifest.get("high_burden_definition", {}).get("threshold")
        != expected_burden_threshold
        or histogram_burden_threshold != expected_burden_threshold
    ):
        msg = "Focused high-burden threshold differs from source recomputation."
        raise ValueError(msg)
    expected_overlap_columns = list(OVERLAP_COLUMNS)
    overlap_count_columns = expected_overlap_columns[4:]
    overlap_schema_valid = overlap.columns.tolist() == expected_overlap_columns
    overlap_counts = (
        overlap.loc[:, overlap_count_columns].to_numpy(dtype=float)
        if overlap_schema_valid
        else np.empty((0, len(overlap_count_columns)), dtype=float)
    )
    expected_overlap_cohorts = np.repeat(TCGA_COHORTS, 2).tolist()
    expected_overlap_directions = ["ME", "CO"] * len(TCGA_COHORTS)
    figure_burden_schema_valid = (
        figure_burden.columns.tolist() == list(BURDEN_BIN_COLUMNS)
    )
    figure_burden_numeric_columns = list(BURDEN_BIN_COLUMNS[2:])
    figure_burden_values = (
        figure_burden.loc[:, figure_burden_numeric_columns].to_numpy(dtype=float)
        if figure_burden_schema_valid
        else np.empty((0, len(figure_burden_numeric_columns)), dtype=float)
    )
    figure_counts = (
        figure_burden["tumor_count"].to_numpy(dtype=float)
        if figure_burden_schema_valid
        else np.empty(0, dtype=float)
    )
    expected_focal_tumors = int(
        summary.set_index("cohort").loc[FOCAL_BURDEN_COHORT, "tumors"],
    )
    expected_pair_counts = pd.DataFrame(
        [
            {"cohort": cohort, **_cohort_pair_counts(run_root, cohort)}
            for cohort in TCGA_COHORTS
        ],
    )
    pair_count_columns = [
        "selected_features",
        "tested_pairs",
        "same_base_pairs_excluded",
        "unfiltered_pair_count",
    ]
    reported_pair_counts = summary.loc[:, pair_count_columns].to_numpy(dtype=float)
    contract_pair_counts = expected_pair_counts.loc[
        :,
        pair_count_columns,
    ].to_numpy(dtype=float)
    decision_count_columns = [
        column
        for column in summary.columns
        if column.endswith(("_total", "_me", "_co", "_direction_unavailable"))
    ]
    decision_counts = summary.loc[:, decision_count_columns].to_numpy(dtype=float)
    decision_decompositions_valid = all(
        np.array_equal(
            summary[f"{_decision_prefix(provider, label)}_total"].to_numpy(),
            summary[
                [
                    f"{_decision_prefix(provider, label)}_me",
                    f"{_decision_prefix(provider, label)}_co",
                    f"{_decision_prefix(provider, label)}_direction_unavailable",
                ]
            ].sum(axis=1).to_numpy(),
        )
        for provider in core.BMRS
        for label in ("primary", "sensitivity")
    )
    summary_by_cohort = summary.set_index("cohort")
    overlap_summary_counts_valid = all(
        int(row.mutsig_primary_rejection_count)
        == int(
            summary_by_cohort.loc[
                row.cohort,
                f"mutsig_primary_rejection_{row.direction.casefold()}",
            ],
        )
        and int(row.cbase_descriptive_crossing_count)
        == int(
            summary_by_cohort.loc[
                row.cohort,
                f"cbase_descriptive_primary_rule_crossing_{row.direction.casefold()}",
            ],
        )
        and int(row.dig_descriptive_crossing_count)
        == int(
            summary_by_cohort.loc[
                row.cohort,
                f"dig_descriptive_primary_rule_crossing_{row.direction.casefold()}",
            ],
        )
        for row in overlap.itertuples(index=False)
    )
    runtime_pairs = runtime.pivot_table(
        index="cohort",
        columns="provider",
        values="pairwise_rows",
        aggfunc="first",
    ).reindex(index=TCGA_COHORTS, columns=core.BMRS)
    top_group_sizes = top_pairs.groupby(["cohort", "direction"]).size()
    top_boolean_columns = [
        column
        for column in top_pairs.columns
        if column.endswith(
            (
                "_rejection",
                "_crossing",
                "_direction_concordant",
                "_direction_discordant",
                "_direction_unavailable",
            ),
        )
    ]
    forbidden_axis_tokens = ("sample", "patient", "cohort_row", "tumor_row")
    if (
        len(summary) != len(TCGA_COHORTS)
        or summary["cohort"].tolist() != list(TCGA_COHORTS)
        or not summary["primary_adjustment"].eq("BY").all()
        or not summary["primary_q_threshold"].eq(0.01).all()
        or not summary["sensitivity_adjustment"].eq("BH").all()
        or not summary["sensitivity_q_threshold"].eq(0.01).all()
        or int(summary["tumors"].sum()) != EXPECTED_TUMOR_COUNT
        or not np.array_equal(reported_pair_counts, contract_pair_counts)
        or not np.isfinite(decision_counts).all()
        or (decision_counts < 0).any()
        or not np.equal(decision_counts, np.floor(decision_counts)).all()
        or not decision_decompositions_valid
        or not overlap_summary_counts_valid
        or (
            decision_counts
            > summary["tested_pairs"].to_numpy(dtype=float)[:, np.newaxis]
        ).any()
        or len(overlap) != len(TCGA_COHORTS) * 2
        or not overlap_schema_valid
        or overlap["cohort"].tolist() != expected_overlap_cohorts
        or overlap["direction"].tolist() != expected_overlap_directions
        or not overlap["adjustment"].eq("BY").all()
        or not overlap["q_threshold"].eq(0.01).all()
        or not np.isfinite(overlap_counts).all()
        or (overlap_counts < 0).any()
        or not np.equal(overlap_counts, np.floor(overlap_counts)).all()
        or (
            overlap["mutsig_rejection_cbase_concordant_crossing_count"]
            + overlap["mutsig_rejection_cbase_discordant_crossing_count"]
            > overlap["mutsig_primary_rejection_count"]
        ).any()
        or (
            overlap["mutsig_rejection_dig_concordant_crossing_count"]
            + overlap["mutsig_rejection_dig_discordant_crossing_count"]
            > overlap["mutsig_primary_rejection_count"]
        ).any()
        or len(runtime) != len(TCGA_COHORTS) * len(core.BMRS)
        or runtime.duplicated(subset=["cohort", "provider"]).any()
        or runtime_pairs.isna().any().any()
        or not np.equal(
            runtime_pairs.to_numpy(dtype=float),
            summary["tested_pairs"].to_numpy(dtype=float)[:, np.newaxis],
        ).all()
        or fit_diagnostics["scope"].tolist() != ["all", *core.BMRS]
        or int(fit_diagnostics.iloc[0]["pairwise_rows"])
        != int(runtime["pairwise_rows"].sum())
        or int(fit_diagnostics.iloc[0]["nonconverged_rows"]) != 0
        or int(fit_diagnostics.iloc[0]["full_affine_rank_rows"])
        + int(fit_diagnostics.iloc[0]["rank_deficient_rows"])
        + int(fit_diagnostics.iloc[0]["rank_not_certified_underflow_rows"])
        != int(fit_diagnostics.iloc[0]["pairwise_rows"])
        or not top_pairs["cohort"].isin(TCGA_COHORTS).all()
        or not top_pairs["direction"].isin(["ME", "CO"]).all()
        or not top_pairs["primary_adjustment"].eq("BY").all()
        or not top_pairs["primary_q_threshold"].eq(0.01).all()
        or top_pairs[["cohort", "gene_a", "gene_b"]].duplicated().any()
        or not top_pairs["mutsig_primary_rejection"].isin([True]).all()
        or not top_pairs["mutsig_direction"].eq(top_pairs["direction"]).all()
        or (top_group_sizes > 10).any()
        or any(
            not top_pairs[column].isin([True, False]).all()
            for column in top_boolean_columns
        )
        or not figure_burden_schema_valid
        or set(figure_burden["cohort"]) != {FOCAL_BURDEN_COHORT}
        or set(figure_burden["provider"]) != set(core.BMRS)
        or figure_burden.duplicated(
            subset=list(BURDEN_BIN_COLUMNS[:-1]),
        ).any()
        or not np.isfinite(figure_burden_values).all()
        or (figure_burden_values < 0).any()
        or (figure_counts < 1).any()
        or not np.equal(figure_counts, np.floor(figure_counts)).all()
        or not all(
            int(
                figure_burden.loc[
                    figure_burden["provider"].eq(provider),
                    "tumor_count",
                ].sum(),
            )
            == expected_focal_tumors
            for provider in core.BMRS
        )
        or not (
            figure_burden["observed_log1p_bin_lower"]
            < figure_burden["observed_log1p_bin_upper"]
        ).all()
        or not (
            figure_burden["expected_log1p_bin_lower"]
            < figure_burden["expected_log1p_bin_upper"]
        ).all()
        or (figure_burden["observed_log1p_bin_upper"] > BURDEN_LOG1P_MAX).any()
        or (figure_burden["expected_log1p_bin_upper"] > BURDEN_LOG1P_MAX).any()
        or any(
            token in column.casefold()
            for column in figure_burden.columns
            for token in forbidden_axis_tokens
        )
        or (output_root / "figure6.pdf").read_bytes()[:5] != b"%PDF-"
    ):
        msg = "Focused reporting tables or PDF failed dimensional validation."
        raise ValueError(msg)
    return manifest


def build_report(  # noqa: PLR0913
    *,
    run_root: Path,
    provider_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    rule_path: Path,
    output_root: Path,
) -> Path:
    """Build all result-dependent reporting artifacts once."""
    cohorts = tuple(TCGA_COHORTS)
    rule = _require_reportable_rule(
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        rule_path=rule_path,
        run_root=run_root,
        provider_root=provider_root,
        action="reporting",
    )
    validate_provider_root(provider_root, cohorts)
    postprocess.validate_derived_root(postprocess_root, cohorts, run_root=run_root)
    postprocess._validate_completion(run_root, cohorts)  # noqa: SLF001
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite reporting root: {output_root}"
        raise FileExistsError(msg)

    outputs, burden_threshold = _report_csv_frames(
        run_root=run_root,
        provider_root=provider_root,
        postprocess_root=postprocess_root,
        rule=rule,
    )
    primary_adjustment = str(rule["primary_adjustment"])
    sensitivity_adjustment = str(rule["sensitivity_adjustment"])
    primary_q = float(rule["primary_q_threshold"])
    sensitivity_q = float(rule["sensitivity_q_threshold"])
    figure6_burden_bins = outputs["figure6_burden_bins.csv"]
    summary = outputs["table_s5.csv"]
    overlap = outputs["provider_overlap.csv"]
    calibration_table = pd.read_csv(
        calibration_root / calibration.SUMMARY_TABLE_NAME,
        float_precision="round_trip",
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    for name, frame in outputs.items():
        frame.to_csv(staging / name, index=False, lineterminator="\n")

    (staging / "table_s5.tex").write_text(
        _table_s5_tex(
            summary,
            primary_adjustment=primary_adjustment,
            primary_q=primary_q,
            sensitivity_adjustment=sensitivity_adjustment,
            sensitivity_q=sensitivity_q,
        ),
        encoding="utf-8",
    )
    figure_path = staging / "figure6.pdf"
    _plot_figure6(
        burden_bins=figure6_burden_bins,
        summary=summary,
        overlap=overlap,
        calibration_table=calibration_table,
        primary_adjustment=primary_adjustment,
        primary_q=primary_q,
        output=figure_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": REPORT_CONTRACT,
        "cohorts": list(cohorts),
        "primary_provider": "mutsig",
        "inference_status": rule["inference_status"],
        "effective_p_policy": rule["effective_p_policy"],
        "primary_adjustment": primary_adjustment,
        "primary_q_threshold": primary_q,
        "sensitivity_adjustment": sensitivity_adjustment,
        "sensitivity_q_threshold": sensitivity_q,
        "provider_overlap": (
            "direction-concordant-descriptive-only-not-an-inferential-vote"
        ),
        "threshold_decision_scale": "natural-log-q-values",
        "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
        "sample_level_rows_included": False,
        "burden_source_policy": BURDEN_SOURCE_POLICY,
        "high_burden_definition": {
            "measure": "pre-K total nonsynonymous SNV event count per tumor",
            "reference": "pooled 10,433-tumor 32-cohort analysis population",
            "pooled_tumor_count": EXPECTED_TUMOR_COUNT,
            "quantile": HIGH_BURDEN_QUANTILE,
            "threshold": burden_threshold,
            "source": "cohort_burden_histogram.csv",
            "comparison": "greater-than-or-equal",
            "interpretation": (
                "descriptive high-burden fraction, not a clinical hypermutator label"
            ),
        },
        "inputs": {
            "run_completion": _file_record(
                run_root / "completion_manifest.json",
                relative_to=run_root,
            ),
            "provider_manifest": _file_record(
                provider_root / "provider_manifest.json",
                relative_to=provider_root,
            ),
            "postprocess_manifest": _file_record(
                postprocess_root / postprocess.ROOT_MANIFEST_NAME,
                relative_to=postprocess_root,
            ),
            "calibration_summary": _file_record(
                calibration_root / calibration.SUMMARY_NAME,
                relative_to=calibration_root,
            ),
            "reporting_rule": {
                "path": rule_path.name,
                "bytes": rule_path.stat().st_size,
                "sha256": _sha256(rule_path),
            },
        },
        "outputs": {
            path.name: _file_record(path, relative_to=staging)
            for path in sorted(staging.iterdir())
            if path.is_file()
        },
    }
    _write_atomic(staging / "report_manifest.json", _canonical_json(manifest) + b"\n")
    staging.replace(output_root)
    validate_report(
        output_root,
        run_root=run_root,
        provider_root=provider_root,
        postprocess_root=postprocess_root,
        calibration_root=calibration_root,
        rule_path=rule_path,
    )
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--reporting-rule", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    """Build focused reporting artifacts from frozen inputs."""
    args = _parser().parse_args()
    print(
        build_report(
            run_root=args.run_root.resolve(),
            provider_root=args.provider_root.resolve(),
            postprocess_root=args.postprocess_root.resolve(),
            calibration_root=args.calibration_root.resolve(),
            rule_path=args.reporting_rule.resolve(),
            output_root=args.output_root.absolute(),
        ),
    )


if __name__ == "__main__":
    main()

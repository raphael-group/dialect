"""Summarize the frozen focused K=500 reporting rule and PAAD/LAML results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import report_tcga_revision_focused as reporting
from analysis.prepare_tcga_revision_focused import _parse_cohorts
from analysis.run_tcga_revision_k500 import BMRS

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION: Final = "1.0.0"
DIAGNOSTIC_CONTRACT: Final = "focused-provider-result-diagnostics-v3"
FOCAL_COHORTS: Final = ("LAML", "PAAD")


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


def _provider_record(  # noqa: PLR0913
    frame: pd.DataFrame,
    *,
    cohort: str,
    provider: str,
    analysis: str,
    adjustment: str,
    threshold: float,
) -> dict[str, int | float | str | bool]:
    """Return fixed-rule crossings and positive-display clipping diagnostics."""
    log_p_values = frame[f"{provider}_log_p_value"].to_numpy(dtype=np.float64)
    p_values = frame[f"{provider}_p_value"].to_numpy(dtype=np.float64)
    log_q_values = frame[
        reporting._log_q_column(provider, adjustment)  # noqa: SLF001
    ].to_numpy(dtype=np.float64)
    q_values = frame[reporting._q_column(provider, adjustment)].to_numpy(  # noqa: SLF001
        dtype=np.float64,
    )
    directions = frame[f"{provider}_direction"].astype("string")
    identifiability = frame[f"{provider}_effect_identifiability"].astype("string")
    crossing = reporting._threshold_crossing(  # noqa: SLF001
        frame,
        provider,
        adjustment,
        threshold,
    )
    interpretation = (
        "primary-MutSig-rejection"
        if provider == "mutsig" and analysis == "primary"
        else "descriptive-threshold-crossing"
    )
    return {
        "cohort": cohort,
        "provider": provider,
        "analysis": analysis,
        "evidence_role": (
            "primary-inference"
            if provider == "mutsig" and analysis == "primary"
            else "descriptive"
        ),
        "adjustment": reporting.ADJUSTMENT_LABELS[adjustment],
        "q_threshold": threshold,
        "interpretation": interpretation,
        "pair_count": len(frame),
        "min_log_p_value": float(log_p_values.min()),
        "min_display_p_value": float(p_values.min()),
        "min_log_q_value": float(log_q_values.min()),
        "min_display_q_value": float(q_values.min()),
        "p_display_clipped_count": int(
            (log_p_values < postprocess.LOG_MIN_POSITIVE_FLOAT).sum(),
        ),
        "q_display_clipped_count": int(
            (log_q_values < postprocess.LOG_MIN_POSITIVE_FLOAT).sum(),
        ),
        "all_q_values_one": bool((q_values == 1).all()),
        "non_full_rank_count": int(identifiability.ne("full-affine-rank").sum()),
        "rank_not_certified_underflow_count": int(
            identifiability.eq("rank-not-certified-underflow").sum(),
        ),
        "decision_count": int(crossing.sum()),
        "me_direction_decision_count": int(
            (crossing & directions.eq("ME").to_numpy()).sum(),
        ),
        "co_direction_decision_count": int(
            (crossing & directions.eq("CO").to_numpy()).sum(),
        ),
        "unavailable_direction_count": int(
            (crossing & ~directions.isin(["ME", "CO"]).to_numpy()).sum(),
        ),
    }


def _focal_top_pairs(  # noqa: PLR0913
    frame: pd.DataFrame,
    *,
    cohort: str,
    primary_adjustment: str,
    primary_q: float,
    sensitivity_adjustment: str,
    sensitivity_q: float,
) -> pd.DataFrame:
    """Return a fixed-size MutSig-ranked audit table without tuning a threshold."""
    primary_column = reporting._log_q_column(  # noqa: SLF001
        "mutsig",
        primary_adjustment,
    )
    ranked = frame.assign(
        _absolute_mutsig_rho=frame["mutsig_rho"].abs().fillna(-1.0),
    ).sort_values(
        [primary_column, "mutsig_log_p_value", "_absolute_mutsig_rho"],
        ascending=[True, True, False],
        kind="stable",
    )
    result = ranked.head(50).drop(columns="_absolute_mutsig_rho").copy()
    result.insert(0, "mutsig_primary_rank", np.arange(1, len(result) + 1))
    result.insert(0, "cohort", cohort)
    result.insert(
        1,
        "primary_adjustment",
        reporting.ADJUSTMENT_LABELS[primary_adjustment],
    )
    result.insert(2, "primary_q_threshold", primary_q)
    result.insert(
        3,
        "sensitivity_adjustment",
        reporting.ADJUSTMENT_LABELS[sensitivity_adjustment],
    )
    result.insert(4, "sensitivity_q_threshold", sensitivity_q)
    mutsig_direction = result["mutsig_direction"].astype("string")
    for provider in BMRS:
        provider_direction = result[f"{provider}_direction"].astype("string")
        for label, adjustment, threshold in (
            ("primary", primary_adjustment, primary_q),
            ("sensitivity", sensitivity_adjustment, sensitivity_q),
        ):
            crossing = reporting._threshold_crossing(  # noqa: SLF001
                result,
                provider,
                adjustment,
                threshold,
            )
            prefix = reporting._decision_prefix(provider, label)  # noqa: SLF001
            result[prefix] = crossing
            result[f"{prefix}_direction_concordant"] = (
                crossing
                & mutsig_direction.isin(["ME", "CO"])
                & provider_direction.eq(mutsig_direction)
            )
            result[f"{prefix}_direction_discordant"] = (
                crossing
                & mutsig_direction.isin(["ME", "CO"])
                & provider_direction.isin(["ME", "CO"])
                & provider_direction.ne(mutsig_direction)
            )
    return result


def diagnose(  # noqa: PLR0913
    *,
    run_root: Path,
    provider_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    rule_path: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> Path:
    """Write fixed-rule counts, directional overlap, and focal audit tables."""
    rule = reporting._require_reportable_rule(  # noqa: SLF001
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        rule_path=rule_path,
        run_root=run_root,
        provider_root=provider_root,
        action="diagnostics",
    )
    source_manifest = postprocess_root / postprocess.ROOT_MANIFEST_NAME
    if not source_manifest.is_file():
        msg = "Focused postprocess manifest is missing."
        raise FileNotFoundError(msg)
    postprocess.validate_derived_root(postprocess_root, cohorts, run_root=run_root)
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite diagnostic root: {output_root}"
        raise FileExistsError(msg)

    primary_adjustment = str(rule["primary_adjustment"])
    sensitivity_adjustment = str(rule["sensitivity_adjustment"])
    primary_q = float(rule["primary_q_threshold"])
    sensitivity_q = float(rule["sensitivity_q_threshold"])
    count_records = []
    overlap_records = []
    focal_tables = []
    for cohort in cohorts:
        frame = reporting._read_inference(postprocess_root, cohort)  # noqa: SLF001
        for analysis, adjustment, threshold in (
            ("primary", primary_adjustment, primary_q),
            ("nominal_sensitivity", sensitivity_adjustment, sensitivity_q),
        ):
            count_records.extend(
                [
                    _provider_record(
                        frame,
                        cohort=cohort,
                        provider=provider,
                        analysis=analysis,
                        adjustment=adjustment,
                        threshold=threshold,
                    )
                    for provider in BMRS
                ],
            )
        overlap_records.extend(
            reporting._overlap_rows(  # noqa: SLF001
                frame,
                cohort=cohort,
                primary_adjustment=primary_adjustment,
                primary_q=primary_q,
            ),
        )
        if cohort in FOCAL_COHORTS:
            focal_tables.append(
                _focal_top_pairs(
                    frame,
                    cohort=cohort,
                    primary_adjustment=primary_adjustment,
                    primary_q=primary_q,
                    sensitivity_adjustment=sensitivity_adjustment,
                    sensitivity_q=sensitivity_q,
                ),
            )

    counts = pd.DataFrame(count_records)
    overlap = pd.DataFrame(overlap_records)
    focal = (
        pd.concat(focal_tables, ignore_index=True)
        if focal_tables
        else pd.DataFrame()
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    counts_path = staging / "cohort_provider_counts.csv"
    overlap_path = staging / "provider_overlap_counts.csv"
    focal_path = staging / "paad_laml_top_pairs.csv"
    counts.to_csv(counts_path, index=False, lineterminator="\n")
    overlap.to_csv(overlap_path, index=False, lineterminator="\n")
    focal.to_csv(focal_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": DIAGNOSTIC_CONTRACT,
        "postprocess_manifest_sha256": _sha256(source_manifest),
        "calibration_summary_sha256": _sha256(
            calibration_root / calibration.SUMMARY_NAME,
        ),
        "reporting_rule_sha256": _sha256(rule_path),
        "inference_status": rule["inference_status"],
        "effective_p_policy": rule["effective_p_policy"],
        "cohorts": list(cohorts),
        "primary_adjustment": primary_adjustment,
        "primary_q_threshold": primary_q,
        "sensitivity_adjustment": sensitivity_adjustment,
        "sensitivity_q_threshold": sensitivity_q,
        "focal_cohorts": list(FOCAL_COHORTS),
        "interpretation": (
            "direction-concordant-provider-overlap-is-descriptive-not-a-vote"
        ),
        "threshold_decision_scale": "natural-log-q-values",
        "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (counts_path, overlap_path, focal_path)
        },
    }
    _write_atomic(
        staging / "diagnostic_manifest.json",
        _canonical_json(manifest) + b"\n",
    )
    staging.replace(output_root)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--reporting-rule", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohorts")
    return parser


def main() -> None:
    """Run focused result diagnostics."""
    args = _parser().parse_args()
    diagnose(
        run_root=args.run_root.resolve(),
        provider_root=args.provider_root.resolve(),
        postprocess_root=args.postprocess_root.resolve(),
        calibration_root=args.calibration_root.resolve(),
        rule_path=args.reporting_rule.resolve(),
        output_root=args.output_root.absolute(),
        cohorts=_parse_cohorts(args.cohorts),
    )


if __name__ == "__main__":
    main()

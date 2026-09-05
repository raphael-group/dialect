"""Freeze the global DIALECT rule after prespecified two-stage calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Final

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import calibrate_tcga_revision_focused_confirmation as confirmation
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis.prepare_tcga_revision_focused import CONFIG_PATH, _load_config
from dialect.data.tcga import TCGA_COHORTS

SCHEMA_VERSION: Final = "1.0.0"
RULE_CONTRACT: Final = "focused-global-reporting-rule-v4"
RULE_NAME: Final = "reporting_rule.json"
REPORTABLE_STATUS: Final = "reportable"
WITHHELD_STATUS: Final = "withheld"


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


def _write_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        msg = f"Refusing to overwrite frozen reporting rule: {path}"
        raise FileExistsError(msg)
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


def _load_summary_gate(summary_path: Path, *, label: str) -> dict[str, object]:
    """Read one summary gate without opening any association-level artifact."""
    if not summary_path.is_file():
        msg = f"{label} summary is missing: {summary_path}"
        raise FileNotFoundError(msg)
    if summary_path.is_symlink():
        msg = f"{label} summary is unsafe: {summary_path}"
        raise ValueError(msg)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        msg = f"{label} summary must be a JSON object."
        raise TypeError(msg)
    calibration_gate_pass(summary)
    return summary


def load_calibration_gate(calibration_root: Path) -> dict[str, object]:
    """Read the preserved stage-one summary gate without association access."""
    return _load_summary_gate(
        calibration_root / calibration.SUMMARY_NAME,
        label="Calibration",
    )


def load_confirmation_gate(confirmation_root: Path) -> dict[str, object]:
    """Read the composite confirmation summary gate without association access."""
    return _load_summary_gate(
        confirmation_root / confirmation.SUMMARY_NAME,
        label="Calibration confirmation",
    )


def calibration_gate_pass(summary: dict[str, object]) -> bool:
    """Return an exact boolean gate decision or fail closed."""
    gate_pass = summary.get("overall_gate_pass")
    if not isinstance(gate_pass, bool):
        msg = "Calibration summary lacks the affirmative overall gate decision."
        raise TypeError(msg)
    return gate_pass


def freeze_rule(  # noqa: PLR0913
    *,
    calibration_root: Path,
    confirmation_root: Path,
    postprocess_root: Path,
    output_path: Path,
    run_root: Path,
    provider_root: Path,
) -> Path:
    """Write one global rule without using observed result counts or identities.

    A failed affirmative calibration gate produces an immutable withheld rule instead
    of silently selecting a different threshold.  Association-level reporting must
    reject that status.
    """
    config = _load_config()
    calibration_config = calibration._load_config()  # noqa: SLF001
    stage1_preview = load_calibration_gate(calibration_root)
    if calibration_gate_pass(stage1_preview) is not False:
        msg = "The preserved stage-one calibration must retain its failed decision."
        raise ValueError(msg)
    confirmation_preview = load_confirmation_gate(confirmation_root)
    stage1_summary = calibration.validate_summary(
        calibration_root,
        run_root=run_root,
        provider_root=provider_root,
    )
    confirmation_summary = confirmation.load_validated_composite_gate(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=confirmation_root,
    )
    if stage1_preview != stage1_summary:
        msg = "Stage-one calibration changed during complete validation."
        raise RuntimeError(msg)
    if confirmation_preview != confirmation_summary:
        msg = "Calibration confirmation changed during complete validation."
        raise RuntimeError(msg)
    gate_pass = calibration_gate_pass(confirmation_summary)
    if stage1_summary.get("reporting_rule_selected") is not False or (
        confirmation_summary.get("reporting_rule_selected") is not False
    ):
        msg = "Calibration must remain result-blind and must not select a rule."
        raise ValueError(msg)
    candidates = calibration_config["reporting_candidates"]
    expected_candidates = {
        "test": "chi-square-one-df-profile-lrt",
        "primary_adjustment": "benjamini-yekutieli",
        "primary_q_threshold": 0.01,
        "sensitivity_adjustment": "benjamini-hochberg",
        "sensitivity_q_threshold": 0.01,
        "thresholds_selected_from_observed_pairs": False,
        "interpretation": "finite-scenario-stress-not-formal-uniform-FDR-proof",
    }
    if any(candidates.get(key) != value for key, value in expected_candidates.items()):
        msg = "Calibration reporting candidates violate the frozen global rule."
        raise ValueError(msg)
    expected_summary = {
        "gate_provider": "mutsig",
        "gate_endpoint_unit": "cohort-sentinel-pair-alpha",
        "gate_method": (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        ),
        "exact_binomial_familywise_error": 0.05,
        "exact_binomial_endpoint_count": 2_048,
        "acceptance_upper_bounds": {"0.01": 0.02, "0.05": 0.07},
        "primary_adjustment": "benjamini-yekutieli",
        "primary_q_candidate": 0.01,
        "sensitivity_adjustment": "benjamini-hochberg",
        "sensitivity_q_candidate": 0.01,
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
    }
    if any(stage1_summary.get(key) != value for key, value in expected_summary.items()):
        msg = "Calibration summary violates the prespecified reporting candidates."
        raise ValueError(msg)
    expected_confirmation = {
        "source_calibration_overall_gate_pass": False,
        "endpoint_count": 2_048,
        "stage1_familywise_error": 0.025,
        "stage1_endpoint_count": 2_048,
        "stage1_selected_endpoint_count": 1,
        "stage2_familywise_error": 0.025,
        "stage2_endpoint_count": 1,
        "stage2_replicates_per_selected_endpoint": 100_000,
        "stage1_and_stage2_counts_pooled": False,
        "additional_confirmation_stage_permitted": False,
        "total_familywise_error": 0.05,
        "overall_rule": (
            "all-unselected-endpoints-pass-stage1-and-all-selected-endpoints-"
            "pass-stage2"
        ),
        "interpretation": ("finite-scenario-stress-not-formal-uniform-FDR-proof"),
    }
    if (
        stage1_summary.get("overall_gate_pass") is not False
        or stage1_summary.get("primary_gate_endpoint_count") != 2_048
        or stage1_summary.get("primary_gate_passed_endpoint_count") != 2_047
        or any(
            confirmation_summary.get(key) != value
            for key, value in expected_confirmation.items()
        )
        or confirmation_summary.get("composite_overall_gate_pass") is not gate_pass
    ):
        msg = "Composite calibration confirmation violates the frozen protocol."
        raise ValueError(msg)
    if gate_pass:
        postprocess.validate_derived_root(
            postprocess_root,
            TCGA_COHORTS,
            run_root=run_root,
        )
        postprocess_manifest_sha256: str | None = _sha256(
            postprocess_root / postprocess.ROOT_MANIFEST_NAME,
        )
    else:
        # A negative gate is a terminal scientific outcome, not permission to
        # derive or even require association-level outputs.  Keep the withheld
        # receipt independent of a postprocessing tree so the gate can stop the
        # workflow exactly where intended.
        postprocess_manifest_sha256 = None
    inference_status = REPORTABLE_STATUS if gate_pass else WITHHELD_STATUS
    rule = {
        "schema_version": SCHEMA_VERSION,
        "contract": RULE_CONTRACT,
        "analysis_config_sha256": _sha256(CONFIG_PATH),
        "calibration_config_sha256": _sha256(calibration.CONFIG_PATH),
        "calibration_summary_sha256": _sha256(
            calibration_root / calibration.SUMMARY_NAME,
        ),
        "calibration_confirmation_config_sha256": _sha256(
            confirmation.CONFIG_PATH,
        ),
        "calibration_confirmation_summary_sha256": _sha256(
            confirmation_root / confirmation.SUMMARY_NAME,
        ),
        "calibration_confirmation_final_table_sha256": _sha256(
            confirmation_root / confirmation.FINAL_TABLE_NAME,
        ),
        "postprocess_manifest_sha256": postprocess_manifest_sha256,
        "scope": "one-identical-rule-across-all-32-tcga-pan-cancer-atlas-cohorts",
        "test": candidates["test"],
        "effective_p_policy": stage1_summary["effective_p_policy"],
        "multiplicity": "provider-specific-complete-within-cohort-family",
        "primary_adjustment": candidates["primary_adjustment"],
        "sensitivity_adjustment": candidates["sensitivity_adjustment"],
        "primary_provider": config["analysis"]["primary_provider"],
        "continuity_provider": config["analysis"]["continuity_provider"],
        "supplementary_providers": config["analysis"]["supplementary_providers"],
        "primary_q_threshold": candidates["primary_q_threshold"],
        "sensitivity_q_threshold": candidates["sensitivity_q_threshold"],
        "threshold_comparison": "inclusive-less-than-or-equal",
        "direction": "primary-provider-rho-sign-after-nondirectional-rejection",
        "direction_unavailable": (
            "retain-nondirectional-rejection-exclude-from-me-co-lists"
        ),
        "provider_overlap": "descriptive-only-not-an-inferential-vote",
        "me_presentation": "primary-MutSig-with-CBaSE-continuity-comparison",
        "co_presentation": "primary-MutSig-with-CBaSE-and-DIG-sensitivity",
        "thresholds_selected_from_observed_pairs": candidates[
            "thresholds_selected_from_observed_pairs"
        ],
        "calibration_gate": {
            "provider": stage1_summary.get("gate_provider"),
            "endpoint_unit": stage1_summary.get("gate_endpoint_unit"),
            "method": "two-stage-alpha-spending-exact-binomial-confirmation",
            "endpoint_count": confirmation_summary.get("endpoint_count"),
            "total_familywise_error": confirmation_summary.get(
                "total_familywise_error",
            ),
            "acceptance_upper_bounds": stage1_summary.get(
                "acceptance_upper_bounds",
            ),
            "source_calibration_overall_gate_pass": stage1_summary.get(
                "overall_gate_pass",
            ),
            "source_calibration_passed_endpoint_count": stage1_summary.get(
                "primary_gate_passed_endpoint_count",
            ),
            "stage1_familywise_error": confirmation_summary.get(
                "stage1_familywise_error",
            ),
            "stage1_selected_endpoint_count": confirmation_summary.get(
                "stage1_selected_endpoint_count",
            ),
            "stage2_familywise_error": confirmation_summary.get(
                "stage2_familywise_error",
            ),
            "stage2_endpoint_count": confirmation_summary.get(
                "stage2_endpoint_count",
            ),
            "stage2_replicates_per_selected_endpoint": confirmation_summary.get(
                "stage2_replicates_per_selected_endpoint",
            ),
            "stage1_and_stage2_counts_pooled": confirmation_summary.get(
                "stage1_and_stage2_counts_pooled",
            ),
            "additional_confirmation_stage_permitted": confirmation_summary.get(
                "additional_confirmation_stage_permitted",
            ),
            "overall_gate_pass": gate_pass,
        },
        "inference_status": inference_status,
        "withheld_reason": (
            None
            if gate_pass
            else "prespecified-two-stage-calibration-confirmation-failed"
        ),
        "claim_scope": "finite-scenario-calibrated-nominal-inference",
        "calibration_interpretation": candidates["interpretation"],
    }
    _write_once(output_path, _canonical_json(rule) + b"\n")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Freeze the calibrated global reporting rule."""
    args = _parser().parse_args()
    print(
        freeze_rule(
            calibration_root=args.calibration_root.resolve(),
            confirmation_root=args.confirmation_root.resolve(),
            postprocess_root=args.postprocess_root.resolve(),
            run_root=args.run_root.resolve(),
            provider_root=args.provider_root.resolve(),
            output_path=args.output.absolute(),
        ),
    )


if __name__ == "__main__":
    main()

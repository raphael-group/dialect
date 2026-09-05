"""Freeze the global DIALECT reporting rule after prespecified calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Final

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis.prepare_tcga_revision_focused import CONFIG_PATH, _load_config
from dialect.data.tcga import TCGA_COHORTS

SCHEMA_VERSION: Final = "1.0.0"
RULE_CONTRACT: Final = "focused-global-reporting-rule-v3"
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


def load_calibration_gate(calibration_root: Path) -> dict[str, object]:
    """Read only the calibration summary needed to enforce the access gate.

    This deliberately precedes the complete calibration validator: the latter
    verifies raw fit receipts and may therefore read pairwise fit outputs.  A
    negative, missing, or malformed gate must stop before that can happen.
    """
    summary_path = calibration_root / calibration.SUMMARY_NAME
    if not summary_path.is_file():
        msg = f"Calibration summary is missing: {summary_path}"
        raise FileNotFoundError(msg)
    if summary_path.is_symlink():
        msg = f"Calibration summary is unsafe: {summary_path}"
        raise ValueError(msg)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        msg = "Calibration summary must be a JSON object."
        raise TypeError(msg)
    calibration_gate_pass(summary)
    return summary


def calibration_gate_pass(summary: dict[str, object]) -> bool:
    """Return an exact boolean gate decision or fail closed."""
    gate_pass = summary.get("overall_gate_pass")
    if not isinstance(gate_pass, bool):
        msg = "Calibration summary lacks the affirmative overall gate decision."
        raise TypeError(msg)
    return gate_pass


def freeze_rule(
    *,
    calibration_root: Path,
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
    summary = load_calibration_gate(calibration_root)
    gate_pass = calibration_gate_pass(summary)
    if gate_pass:
        summary = calibration.validate_summary(
            calibration_root,
            run_root=run_root,
            provider_root=provider_root,
        )
        if calibration_gate_pass(summary) is not True:
            msg = "Calibration gate changed during complete validation."
            raise RuntimeError(msg)
    if summary.get("reporting_rule_selected") is not False:
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
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        msg = "Calibration summary violates the prespecified reporting candidates."
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
        "postprocess_manifest_sha256": postprocess_manifest_sha256,
        "scope": "one-identical-rule-across-all-32-tcga-pan-cancer-atlas-cohorts",
        "test": candidates["test"],
        "effective_p_policy": summary["effective_p_policy"],
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
            "provider": summary.get("gate_provider"),
            "endpoint_unit": summary.get("gate_endpoint_unit"),
            "method": summary.get("gate_method"),
            "endpoint_count": summary.get("exact_binomial_endpoint_count"),
            "familywise_error": summary.get("exact_binomial_familywise_error"),
            "acceptance_upper_bounds": summary.get("acceptance_upper_bounds"),
            "overall_gate_pass": gate_pass,
        },
        "inference_status": inference_status,
        "withheld_reason": (
            None
            if gate_pass
            else "prespecified-finite-scenario-calibration-gate-failed"
        ),
        "claim_scope": "finite-scenario-calibrated-nominal-inference",
        "calibration_interpretation": candidates["interpretation"],
    }
    _write_once(output_path, _canonical_json(rule) + b"\n")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
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
            postprocess_root=args.postprocess_root.resolve(),
            run_root=args.run_root.resolve(),
            provider_root=args.provider_root.resolve(),
            output_path=args.output.absolute(),
        ),
    )


if __name__ == "__main__":
    main()

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
RULE_CONTRACT: Final = "focused-global-reporting-rule-v1"
RULE_NAME: Final = "reporting_rule.json"


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


def freeze_rule(
    *,
    calibration_root: Path,
    postprocess_root: Path,
    output_path: Path,
) -> Path:
    """Write one global rule without using observed result counts or identities."""
    config = _load_config()
    calibration_config = calibration._load_config()  # noqa: SLF001
    summary = calibration.validate_summary(calibration_root)
    postprocess.validate_derived_root(postprocess_root, TCGA_COHORTS)
    if summary.get("detected_inflation"):
        msg = (
            "Prespecified calibration detected inflation; the chi-square/BH candidates "
            "cannot be frozen as the reporting rule."
        )
        raise RuntimeError(msg)
    if summary.get("retain_chi_square_bh_candidates") is not True:
        msg = "Calibration did not retain the prespecified reporting candidates."
        raise RuntimeError(msg)
    candidates = calibration_config["reporting_candidates"]
    rule = {
        "schema_version": SCHEMA_VERSION,
        "contract": RULE_CONTRACT,
        "analysis_config_sha256": _sha256(CONFIG_PATH),
        "calibration_config_sha256": _sha256(calibration.CONFIG_PATH),
        "calibration_summary_sha256": _sha256(
            calibration_root / calibration.SUMMARY_NAME,
        ),
        "postprocess_manifest_sha256": _sha256(
            postprocess_root / postprocess.ROOT_MANIFEST_NAME,
        ),
        "scope": "one-identical-rule-across-all-32-cancer-types",
        "test": "chi-square-one-df-profile-lrt",
        "multiplicity": "provider-specific-BH-complete-within-cohort-family",
        "primary_provider": config["analysis"]["primary_provider"],
        "continuity_provider": config["analysis"]["continuity_provider"],
        "supplementary_providers": config["analysis"]["supplementary_providers"],
        "primary_q_threshold": candidates["primary_q_threshold"],
        "sensitivity_q_threshold": candidates["sensitivity_q_threshold"],
        "threshold_comparison": "inclusive-less-than-or-equal",
        "direction": "primary-provider-rho-sign-after-nondirectional-testing",
        "direction_unavailable": (
            "retain-nondirectional-rejection-exclude-from-me-co-lists"
        ),
        "provider_overlap": "descriptive-only-not-an-inferential-vote",
        "me_presentation": "primary-MutSig-with-CBaSE-continuity-comparison",
        "co_presentation": "primary-MutSig-with-CBaSE-and-DIG-sensitivity",
        "thresholds_selected_from_observed_pairs": False,
        "claim_scope": "finite-scenario-calibrated-nominal-inference",
    }
    _write_once(output_path, _canonical_json(rule) + b"\n")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Freeze the calibrated global reporting rule."""
    args = _parser().parse_args()
    print(
        freeze_rule(
            calibration_root=args.calibration_root.resolve(),
            postprocess_root=args.postprocess_root.resolve(),
            output_path=args.output.absolute(),
        ),
    )


if __name__ == "__main__":
    main()

"""Focused inference-v2 contract tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import report_tcga_revision_focused as reporting

if TYPE_CHECKING:
    from pathlib import Path


def test_by_and_bh_are_explicit_complete_family_adjustments() -> None:
    p_values = np.asarray([0.01, 0.04, 0.03, 0.002], dtype=np.float64)

    bh = postprocess.benjamini_hochberg(p_values)
    by = postprocess.benjamini_yekutieli(p_values)

    assert bh == pytest.approx([0.02, 0.04, 0.04, 0.008])
    assert by == pytest.approx(np.asarray([0.02, 0.04, 0.04, 0.008]) * (25 / 12))
    assert np.all(by >= bh)


def test_non_full_rank_pairs_remain_in_family_at_p_one() -> None:
    statistics = postprocess._provider_statistics(  # noqa: SLF001
        np.asarray([1_000.0, 1_000.0, 3.841458820694124]),
        pd.Series([-0.9, 0.9, 0.2]),
        pd.Series(
            [
                "rank-deficient",
                "rank-not-certified-underflow",
                "full-affine-rank",
            ],
        ),
    )

    p_values = np.asarray(statistics["p_value"])
    bh = np.asarray(statistics["bh_q_value"])
    assert p_values[:2].tolist() == [1.0, 1.0]
    assert p_values[2] == pytest.approx(0.05)
    assert bh[2] == pytest.approx(0.15)
    assert np.asarray(statistics["effect_reportable"]).tolist() == [False, False, True]
    assert pd.Series(statistics["direction"]).tolist() == [
        "unavailable",
        "unavailable",
        "CO",
    ]
    assert pd.Series(statistics["rho"]).iloc[:2].isna().all()


def test_report_builder_stops_before_association_reads_for_withheld_rule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reporting, "validate_provider_root", lambda *_args: {})
    monkeypatch.setattr(
        reporting.postprocess,
        "validate_derived_root",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        reporting.calibration,
        "validate_summary",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        reporting,
        "_load_rule",
        lambda *_args: {
            "inference_status": reporting.rule_module.WITHHELD_STATUS,
            "withheld_reason": "prespecified-gate-failed",
        },
    )

    def fail_if_read(*_args, **_kwargs):
        msg = "association values were read"
        raise AssertionError(msg)

    monkeypatch.setattr(reporting, "_burden_values", fail_if_read)
    monkeypatch.setattr(reporting, "_read_inference", fail_if_read)

    with pytest.raises(RuntimeError, match="reporting is withheld"):
        reporting.build_report(
            run_root=tmp_path / "run",
            provider_root=tmp_path / "providers",
            postprocess_root=tmp_path / "postprocess",
            calibration_root=tmp_path / "calibration",
            rule_path=tmp_path / "rule.json",
            output_root=tmp_path / "report",
        )


def test_labels_are_derived_from_named_adjustment() -> None:
    assert (
        reporting._threshold_label("benjamini-yekutieli", 0.01)  # noqa: SLF001
        == "BY q <= 0.01"
    )
    assert (
        reporting._threshold_label("benjamini-hochberg", 0.01)  # noqa: SLF001
        == "BH q <= 0.01"
    )

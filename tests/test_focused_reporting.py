"""Tests for focused result-dependent revision reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import report_tcga_revision_focused as reporting

if TYPE_CHECKING:
    from pathlib import Path


def _inference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_a": ["A_M", "B_M", "C_M", "D_M"],
            "gene_b": ["B_M", "C_M", "D_M", "E_M"],
            "cbase_likelihood_ratio": [10.0, 9.0, 2.0, 0.1],
            "cbase_log_p_value": np.log([0.001, 0.0015, 0.003, 0.9]),
            "cbase_p_value": [0.001, 0.0015, 0.003, 0.9],
            "cbase_log_by_q_value": np.log([0.005, 0.006, 0.5, 1.0]),
            "cbase_by_q_value": [0.005, 0.006, 0.5, 1.0],
            "cbase_log_bh_q_value": np.log([0.004, 0.005, 0.009, 1.0]),
            "cbase_bh_q_value": [0.004, 0.005, 0.009, 1.0],
            "cbase_rho": [-0.5, -0.4, -0.2, 0.1],
            "cbase_direction": ["ME", "ME", "ME", "CO"],
            "cbase_effect_identifiability": ["full-affine-rank"] * 4,
            "cbase_effect_reportable": [True] * 4,
            "dig_likelihood_ratio": [9.0, 8.0, 1.0, 0.2],
            "dig_log_p_value": np.log([0.001, 0.002, 0.015, 0.7]),
            "dig_p_value": [0.001, 0.002, 0.015, 0.7],
            "dig_log_by_q_value": np.log([0.006, 0.007, 0.8, 1.0]),
            "dig_by_q_value": [0.006, 0.007, 0.8, 1.0],
            "dig_log_bh_q_value": np.log([0.005, 0.006, 0.02, 1.0]),
            "dig_bh_q_value": [0.005, 0.006, 0.02, 1.0],
            "dig_rho": [-0.4, 0.3, -0.1, 0.2],
            "dig_direction": ["ME", "CO", "ME", "CO"],
            "dig_effect_identifiability": ["full-affine-rank"] * 4,
            "dig_effect_reportable": [True] * 4,
            "mutsig_likelihood_ratio": [15.0, 12.0, 3.0, 0.1],
            "mutsig_log_p_value": np.log([0.0001, 0.001, 0.002, 0.8]),
            "mutsig_p_value": [0.0001, 0.001, 0.002, 0.8],
            "mutsig_log_by_q_value": np.log([0.001, 0.004, 0.4, 0.9]),
            "mutsig_by_q_value": [0.001, 0.004, 0.4, 0.9],
            "mutsig_log_bh_q_value": np.log([0.0005, 0.003, 0.008, 0.8]),
            "mutsig_bh_q_value": [0.0005, 0.003, 0.008, 0.8],
            "mutsig_rho": [-0.7, 0.6, -0.3, 0.2],
            "mutsig_direction": ["ME", "CO", "ME", "CO"],
            "mutsig_effect_identifiability": ["full-affine-rank"] * 4,
            "mutsig_effect_reportable": [True] * 4,
        },
    )


def test_high_burden_threshold_uses_frozen_pooled_population() -> None:
    values = {"A": np.arange(reporting.EXPECTED_TUMOR_COUNT, dtype=float)}
    assert reporting._high_burden_threshold(values) == 10_328  # noqa: SLF001

    with pytest.raises(ValueError, match="pooled high-burden"):
        reporting._high_burden_threshold({"A": np.arange(10)})  # noqa: SLF001


def test_cohort_summary_uses_global_rule_for_every_provider() -> None:
    row = reporting._cohort_summary_row(  # noqa: SLF001
        cohort="TEST",
        frame=_inference_frame(),
        burdens=np.asarray([1, 2, 3, 100], dtype=float),
        high_burden_threshold=100,
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
        sensitivity_adjustment="benjamini-hochberg",
        sensitivity_q=0.01,
    )
    assert row["tested_pairs"] == 4
    assert row["high_burden_fraction"] == 0.25
    assert row["mutsig_primary_rejection_total"] == 2
    assert row["mutsig_primary_rejection_me"] == 1
    assert row["mutsig_primary_rejection_co"] == 1
    assert row["mutsig_primary_rejection_direction_unavailable"] == 0
    assert row["cbase_descriptive_primary_rule_crossing_total"] == 2
    assert row["dig_descriptive_primary_rule_crossing_total"] == 2
    assert row["mutsig_descriptive_sensitivity_rule_crossing_total"] == 3


def test_threshold_decisions_use_log_q_not_clipped_display_values() -> None:
    frame = _inference_frame()
    frame["mutsig_by_q_value"] = 1.0

    crossing = reporting._threshold_crossing(  # noqa: SLF001
        frame,
        "mutsig",
        "benjamini-yekutieli",
        0.01,
    )

    assert crossing.tolist() == [True, True, False, False]


def test_primary_pair_ranking_and_overlap_are_descriptive() -> None:
    frame = _inference_frame()
    top = reporting._top_primary_pairs(  # noqa: SLF001
        frame,
        cohort="TEST",
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
    )
    assert top[["gene_a", "gene_b"]].to_numpy().tolist() == [
        ["A_M", "B_M"],
        ["B_M", "C_M"],
    ]
    assert top["descriptive_direction_concordant_provider_count"].tolist() == [2, 1]
    assert top["descriptive_direction_discordant_provider_count"].tolist() == [0, 1]

    overlap = reporting._overlap_rows(  # noqa: SLF001
        frame,
        cohort="TEST",
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
    )
    assert overlap == [
        {
            "cohort": "TEST",
            "direction": "ME",
            "adjustment": "BY",
            "q_threshold": 0.01,
            "mutsig_primary_rejection_count": 1,
            "cbase_descriptive_crossing_count": 2,
            "dig_descriptive_crossing_count": 1,
            "mutsig_rejection_cbase_concordant_crossing_count": 1,
            "mutsig_rejection_cbase_discordant_crossing_count": 0,
            "mutsig_rejection_dig_concordant_crossing_count": 1,
            "mutsig_rejection_dig_discordant_crossing_count": 0,
        },
        {
            "cohort": "TEST",
            "direction": "CO",
            "adjustment": "BY",
            "q_threshold": 0.01,
            "mutsig_primary_rejection_count": 1,
            "cbase_descriptive_crossing_count": 0,
            "dig_descriptive_crossing_count": 1,
            "mutsig_rejection_cbase_concordant_crossing_count": 0,
            "mutsig_rejection_cbase_discordant_crossing_count": 1,
            "mutsig_rejection_dig_concordant_crossing_count": 1,
            "mutsig_rejection_dig_discordant_crossing_count": 0,
        },
    ]


def test_pmf_mean_preserves_native_integer_support() -> None:
    assert reporting._pmf_mean({0: 0.5, 2: 0.25, 4: 0.25}) == 1.5  # noqa: SLF001


def test_figure_burden_bins_reconcile_observed_axis_without_sample_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_by_provider = {"cbase": 2.0, "dig": 3.0, "mutsig": 4.0}

    def fake_expected(
        *,
        run_root: Path,
        cohort: str,
        provider: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert run_root == tmp_path
        assert cohort == reporting.FOCAL_BURDEN_COHORT
        return np.asarray([1.0, 5.0]), np.full(2, expected_by_provider[provider])

    monkeypatch.setattr(reporting, "_expected_selected_burden", fake_expected)
    source = reporting._figure6_burden_bins(tmp_path)  # noqa: SLF001

    assert source.columns.tolist() == list(reporting.BURDEN_BIN_COLUMNS)
    assert set(source["provider"]) == set(reporting.core.BMRS)
    assert source.groupby("provider")["tumor_count"].sum().eq(2).all()
    assert not any(
        token in column.casefold()
        for column in source.columns
        for token in ("sample", "patient", "cohort_row", "tumor_row")
    )


def test_figure6_uses_v2_calibration_and_directional_overlap(tmp_path: Path) -> None:
    burden = pd.concat(
        [
            reporting._aggregate_burden_bins(  # noqa: SLF001
                np.asarray([1.0, 3.0]),
                np.asarray(expected),
                provider=provider,
            )
            for provider, expected in {
                "cbase": [1.0, 2.0],
                "dig": [1.5, 2.5],
                "mutsig": [0.8, 2.8],
            }.items()
        ],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        {
            "cohort": ["A", "B"],
            "cbase_descriptive_primary_rule_crossing_co": [2, 1],
            "dig_descriptive_primary_rule_crossing_co": [1, 1],
            "mutsig_primary_rejection_co": [1, 0],
        },
    )
    overlap = pd.DataFrame(
        {
            "direction": ["ME", "CO", "ME", "CO"],
            "mutsig_primary_rejection_count": [2, 1, 1, 0],
            "mutsig_rejection_cbase_concordant_crossing_count": [1, 1, 1, 0],
            "mutsig_rejection_dig_concordant_crossing_count": [2, 0, 0, 0],
            "mutsig_rejection_cbase_discordant_crossing_count": [0, 0, 0, 0],
            "mutsig_rejection_dig_discordant_crossing_count": [0, 1, 1, 0],
        },
    )
    calibration = pd.DataFrame(
        [
            {
                "provider": provider,
                "screen": "marginal_lrt",
                "threshold": threshold,
                "rate": threshold / 2,
                "gate_endpoint": provider == "mutsig",
                "hoeffding_upper_bound": (
                    threshold * 0.8 if provider == "mutsig" else np.nan
                ),
                "acceptance_upper_bound": (
                    {0.01: 0.02, 0.05: 0.06}[threshold]
                    if provider == "mutsig"
                    else np.nan
                ),
            }
            for provider in reporting.core.BMRS
            for threshold in (0.01, 0.05)
        ],
    )
    output = tmp_path / "figure6.pdf"

    reporting._plot_figure6(  # noqa: SLF001
        burden_bins=burden,
        summary=summary,
        overlap=overlap,
        calibration_table=calibration,
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
        output=output,
    )

    assert output.read_bytes().startswith(b"%PDF-")


def test_fit_diagnostics_cover_every_certificate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reporting.core, "BMRS", ("cbase",))
    task = tmp_path / "tasks" / "TEST" / "cbase"
    task.mkdir(parents=True)
    pd.DataFrame(
        {
            "Fit Converged": [True, True, True],
            "Fit Iterations": [0, 2, 4],
            "Fit Last LL Gain": [0.0, 1e-12, 2e-12],
            "Fit Fixed-Point Residual": [0.0, 1e-10, 2e-10],
            "Fit KKT Residual": [0.0, 1e-9, 2e-9],
            "Effect Identifiability": [
                "full-affine-rank",
                "rank-deficient",
                "rank-not-certified-underflow",
            ],
        },
    ).to_csv(task / "pairwise_interaction_results.csv", index=False)

    rows = reporting._fit_diagnostic_rows(tmp_path, ("TEST",))  # noqa: SLF001

    assert rows == [
        {
            "scope": "all",
            "pairwise_rows": 3,
            "converged_rows": 3,
            "nonconverged_rows": 0,
            "iterations_min": 0,
            "iterations_median": 2.0,
            "iterations_p95": 3.8,
            "iterations_max": 4,
            "minimum_last_ll_gain": 0.0,
            "maximum_last_ll_gain": 2e-12,
            "maximum_fixed_point_residual": 2e-10,
            "maximum_kkt_residual": 2e-9,
            "full_affine_rank_rows": 1,
            "rank_deficient_rows": 1,
            "rank_not_certified_underflow_rows": 1,
        },
        {
            "scope": "cbase",
            "pairwise_rows": 3,
            "converged_rows": 3,
            "nonconverged_rows": 0,
            "iterations_min": 0,
            "iterations_median": 2.0,
            "iterations_p95": 3.8,
            "iterations_max": 4,
            "minimum_last_ll_gain": 0.0,
            "maximum_last_ll_gain": 2e-12,
            "maximum_fixed_point_residual": 2e-10,
            "maximum_kkt_residual": 2e-9,
            "full_affine_rank_rows": 1,
            "rank_deficient_rows": 1,
            "rank_not_certified_underflow_rows": 1,
        },
    ]


def test_report_validator_binds_inventory_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "report"
    root.mkdir()
    tumors = [reporting.EXPECTED_TUMOR_COUNT - 31, *([1] * 31)]
    pd.DataFrame(
        {
            "cohort": list(reporting.TCGA_COHORTS),
            "primary_adjustment": ["BY"] * len(reporting.TCGA_COHORTS),
            "primary_q_threshold": [0.01] * len(reporting.TCGA_COHORTS),
            "sensitivity_adjustment": ["BH"] * len(reporting.TCGA_COHORTS),
            "sensitivity_q_threshold": [0.01] * len(reporting.TCGA_COHORTS),
            "tumors": tumors,
        },
    ).to_csv(root / "table_s5.csv", index=False)
    pd.DataFrame(
        {
            "cohort": np.repeat(reporting.TCGA_COHORTS, 2),
            "direction": ["ME", "CO"] * len(reporting.TCGA_COHORTS),
            "adjustment": ["BY"] * (len(reporting.TCGA_COHORTS) * 2),
            "q_threshold": [0.01] * (len(reporting.TCGA_COHORTS) * 2),
            "mutsig_primary_rejection_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
            "cbase_descriptive_crossing_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
            "dig_descriptive_crossing_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
            "mutsig_rejection_cbase_concordant_crossing_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
            "mutsig_rejection_cbase_discordant_crossing_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
            "mutsig_rejection_dig_concordant_crossing_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
            "mutsig_rejection_dig_discordant_crossing_count": np.zeros(
                len(reporting.TCGA_COHORTS) * 2,
                dtype=int,
            ),
        },
    ).to_csv(root / "provider_overlap.csv", index=False)
    pd.DataFrame(
        {
            "cohort": np.repeat(
                reporting.TCGA_COHORTS,
                len(reporting.core.BMRS),
                ),
                "provider": list(reporting.core.BMRS) * len(reporting.TCGA_COHORTS),
                "pairwise_rows": np.ones(
                    len(reporting.TCGA_COHORTS) * len(reporting.core.BMRS),
                    dtype=int,
                ),
            },
        ).to_csv(root / "runtime_summary.csv", index=False)
    pairwise_rows = len(reporting.TCGA_COHORTS) * len(reporting.core.BMRS)
    pd.DataFrame(
        {
            "scope": ["all", *reporting.core.BMRS],
            "pairwise_rows": [pairwise_rows, *([len(reporting.TCGA_COHORTS)] * 3)],
            "nonconverged_rows": [0, 0, 0, 0],
            "full_affine_rank_rows": [
                pairwise_rows,
                *([len(reporting.TCGA_COHORTS)] * 3),
            ],
            "rank_deficient_rows": [0, 0, 0, 0],
            "rank_not_certified_underflow_rows": [0, 0, 0, 0],
        },
    ).to_csv(root / "fit_diagnostics_summary.csv", index=False)
    pd.DataFrame(columns=["cohort", "gene_a", "gene_b"]).to_csv(
        root / "top_primary_pairs.csv",
        index=False,
    )
    (root / "table_s5.tex").write_text("table\n", encoding="utf-8")
    (root / "figure6.pdf").write_bytes(b"%PDF-synthetic\n")
    focal_index = list(reporting.TCGA_COHORTS).index(reporting.FOCAL_BURDEN_COHORT)
    focal_tumors = tumors[focal_index]
    pd.DataFrame(
        {
            "cohort": [reporting.FOCAL_BURDEN_COHORT] * len(reporting.core.BMRS),
            "provider": list(reporting.core.BMRS),
            "observed_log10_plus_one_lower": [0.0] * len(reporting.core.BMRS),
            "observed_log10_plus_one_upper": [0.25] * len(reporting.core.BMRS),
            "expected_log10_plus_one_lower": [0.0] * len(reporting.core.BMRS),
            "expected_log10_plus_one_upper": [0.25] * len(reporting.core.BMRS),
            "tumor_count": [focal_tumors] * len(reporting.core.BMRS),
        },
    ).to_csv(root / "figure6_burden_bins.csv", index=False)
    names = {
        "figure6_burden_bins.csv",
        "table_s5.csv",
        "provider_overlap.csv",
        "top_primary_pairs.csv",
        "runtime_summary.csv",
        "fit_diagnostics_summary.csv",
        "table_s5.tex",
        "figure6.pdf",
    }
    manifest = {
        "schema_version": reporting.SCHEMA_VERSION,
        "contract": reporting.REPORT_CONTRACT,
        "cohorts": list(reporting.TCGA_COHORTS),
        "primary_provider": "mutsig",
        "inference_status": reporting.rule_module.REPORTABLE_STATUS,
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "primary_adjustment": "benjamini-yekutieli",
        "primary_q_threshold": 0.01,
        "sensitivity_adjustment": "benjamini-hochberg",
        "sensitivity_q_threshold": 0.01,
        "provider_overlap": (
            "direction-concordant-descriptive-only-not-an-inferential-vote"
        ),
        "threshold_decision_scale": "natural-log-q-values",
        "probability_representation": reporting.postprocess.PROBABILITY_REPRESENTATION,
        "sample_level_rows_included": False,
        "burden_source_policy": "fixed-aggregate-bins-and-cohort-summaries-only",
        "inputs": {
            "run_completion": {},
            "provider_manifest": {},
            "postprocess_manifest": {},
            "calibration_summary": {},
            "reporting_rule": {},
        },
        "high_burden_definition": {
            "pooled_tumor_count": reporting.EXPECTED_TUMOR_COUNT,
        },
        "outputs": {
            name: reporting._file_record(root / name, relative_to=root)  # noqa: SLF001
            for name in names
        },
    }
    (root / "report_manifest.json").write_bytes(
        reporting._canonical_json(manifest) + b"\n",  # noqa: SLF001
    )
    assert reporting.validate_report(root)["contract"] == reporting.REPORT_CONTRACT

    (root / "figure6.pdf").write_bytes(b"%PDF-tampered\n")
    with pytest.raises(ValueError, match="output changed"):
        reporting.validate_report(root)

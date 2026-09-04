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
            "cbase_p_value": [0.001, 0.02, 0.03, 0.9],
            "cbase_q_value": [0.01, 0.2, 0.05, 1.0],
            "cbase_rho": [-0.5, 0.4, -0.2, 0.1],
            "cbase_direction": ["ME", "CO", "ME", "CO"],
            "dig_p_value": [0.002, 0.01, 0.6, 0.7],
            "dig_q_value": [0.02, 0.08, 0.8, 0.9],
            "dig_rho": [-0.4, 0.3, -0.1, 0.2],
            "dig_direction": ["ME", "CO", "ME", "CO"],
            "mutsig_p_value": [0.0001, 0.004, 0.3, 0.8],
            "mutsig_q_value": [0.001, 0.04, 0.4, 0.9],
            "mutsig_rho": [-0.7, 0.6, -0.3, 0.2],
            "mutsig_direction": ["ME", "CO", "ME", "CO"],
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
        primary_q=0.1,
        sensitivity_q=0.2,
    )
    assert row["tested_pairs"] == 4
    assert row["high_burden_fraction"] == 0.25
    assert row["mutsig_primary_total"] == 2
    assert row["mutsig_primary_me"] == 1
    assert row["mutsig_primary_co"] == 1
    assert row["mutsig_primary_direction_unavailable"] == 0
    assert row["cbase_primary_total"] == 2
    assert row["dig_primary_total"] == 2


def test_primary_pair_ranking_and_overlap_are_descriptive() -> None:
    frame = _inference_frame()
    top = reporting._top_primary_pairs(  # noqa: SLF001
        frame,
        cohort="TEST",
        primary_q=0.1,
    )
    assert top[["gene_a", "gene_b"]].to_numpy().tolist() == [
        ["A_M", "B_M"],
        ["B_M", "C_M"],
    ]
    assert top["provider_support"].tolist() == [3, 2]

    overlap = reporting._overlap_rows(  # noqa: SLF001
        frame,
        cohort="TEST",
        primary_q=0.1,
    )
    assert overlap == [
        {
            "cohort": "TEST",
            "direction": "ME",
            "q_threshold": 0.1,
            "cbase": 2,
            "dig": 1,
            "mutsig": 1,
            "at_least_one": 2,
            "at_least_two": 1,
            "all_three": 1,
            "mutsig_and_cbase": 1,
        },
        {
            "cohort": "TEST",
            "direction": "CO",
            "q_threshold": 0.1,
            "cbase": 0,
            "dig": 1,
            "mutsig": 1,
            "at_least_one": 1,
            "at_least_two": 1,
            "all_three": 0,
            "mutsig_and_cbase": 0,
        },
    ]


def test_pmf_mean_preserves_native_integer_support() -> None:
    assert reporting._pmf_mean({0: 0.5, 2: 0.25, 4: 0.25}) == 1.5  # noqa: SLF001


def test_cohort_burden_source_is_deidentified_and_complete() -> None:
    values = {
        cohort: np.asarray([index], dtype=float)
        for index, cohort in enumerate(reporting.TCGA_COHORTS)
    }
    source = reporting._cohort_burden_source(values)  # noqa: SLF001

    assert source["cohort"].tolist() == list(reporting.TCGA_COHORTS)
    assert source["cohort_row"].tolist() == [1] * len(reporting.TCGA_COHORTS)
    assert not any("sample" in column.casefold() for column in source.columns)


def test_figure_burden_source_reconciles_observed_axis(
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
    source = reporting._figure6_burden_source(tmp_path)  # noqa: SLF001

    assert source["cohort_row"].tolist() == [1, 2]
    assert source["observed_selected_event_count"].tolist() == [1.0, 5.0]
    assert source["mutsig_model_expected_selected_event_count"].tolist() == [4.0, 4.0]


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
                "full-affine-rank",
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
            "full_affine_rank_rows": 2,
            "rank_deficient_rows": 1,
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
            "full_affine_rank_rows": 2,
            "rank_deficient_rows": 1,
        },
    ]


def test_report_validator_binds_inventory_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "report"
    root.mkdir()
    tumors = [reporting.EXPECTED_TUMOR_COUNT - 31, *([1] * 31)]
    pd.DataFrame(
        {"cohort": list(reporting.TCGA_COHORTS), "tumors": tumors},
    ).to_csv(root / "table_s5.csv", index=False)
    pd.DataFrame(
        {
            "cohort": np.repeat(reporting.TCGA_COHORTS, 2),
            "direction": ["ME", "CO"] * len(reporting.TCGA_COHORTS),
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
            "cohort": np.repeat(
                reporting.TCGA_COHORTS,
                tumors,
            ),
            "cohort_row": np.concatenate(
                [np.arange(1, count + 1) for count in tumors],
            ),
            "pre_k_total_nonsynonymous_snv_event_count": np.zeros(
                reporting.EXPECTED_TUMOR_COUNT,
            ),
        },
    ).to_csv(root / "cohort_burden_source.csv", index=False)
    pd.DataFrame(
        {
            "cohort": [reporting.FOCAL_BURDEN_COHORT] * focal_tumors,
            "cohort_row": np.arange(1, focal_tumors + 1),
            "observed_selected_event_count": np.zeros(focal_tumors),
            "cbase_model_expected_selected_event_count": np.zeros(focal_tumors),
            "dig_model_expected_selected_event_count": np.zeros(focal_tumors),
            "mutsig_model_expected_selected_event_count": np.zeros(focal_tumors),
        },
    ).to_csv(root / "figure6_burden_source.csv", index=False)
    names = {
        "cohort_burden_source.csv",
        "figure6_burden_source.csv",
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

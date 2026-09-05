"""Tests for focused result-dependent revision reporting."""

from __future__ import annotations

import inspect
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import report_tcga_revision_focused as reporting

if TYPE_CHECKING:
    from pathlib import Path


def _inference_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
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
    for provider in reporting.core.BMRS:
        frame[f"{provider}_fit_converged"] = True
        frame[f"{provider}_fit_iterations"] = np.asarray(
            [0, 1, 2, 3],
            dtype=np.int64,
        )
        frame[f"{provider}_fit_last_ll_gain"] = [0.0, 1e-12, 2e-12, 3e-12]
        frame[f"{provider}_fit_fixed_point_residual"] = [
            0.0,
            1e-10,
            2e-10,
            3e-10,
        ]
        frame[f"{provider}_fit_kkt_residual"] = [0.0, 1e-9, 2e-9, 3e-9]
    return frame.loc[:, reporting.postprocess.result_columns()]


def test_high_burden_threshold_uses_frozen_pooled_population() -> None:
    values = {"A": np.arange(reporting.EXPECTED_TUMOR_COUNT, dtype=float)}
    assert reporting._high_burden_threshold(values) == 10_328  # noqa: SLF001

    with pytest.raises(ValueError, match="pooled high-burden"):
        reporting._high_burden_threshold({"A": np.arange(10)})  # noqa: SLF001


def test_cohort_burden_histogram_reconstructs_every_summary_field(
    tmp_path: Path,
) -> None:
    cohorts = tuple(reporting.TCGA_COHORTS)
    first_count = reporting.EXPECTED_TUMOR_COUNT - len(cohorts) + 1
    values = {
        cohorts[0]: np.concatenate(
            (
                np.asarray([1.0, 9.0]),
                np.full(first_count - 2, 7.0),
            ),
        ),
        **{cohort: np.asarray([7.0]) for cohort in cohorts[1:]},
    }
    histogram = reporting._cohort_burden_histogram(values, cohorts)  # noqa: SLF001
    reconstructed = reporting._burden_values_from_histogram(  # noqa: SLF001
        histogram,
        cohorts,
    )
    threshold = reporting._high_burden_threshold(reconstructed)  # noqa: SLF001
    summary = pd.DataFrame(
        [
            {
                "cohort": cohort,
                **reporting._burden_summary_fields(  # noqa: SLF001
                    reconstructed[cohort],
                    threshold,
                ),
            }
            for cohort in cohorts
        ],
    )

    assert histogram.columns.tolist() == list(
        reporting.COHORT_BURDEN_HISTOGRAM_COLUMNS,
    )
    assert histogram["cohort"].drop_duplicates().tolist() == list(cohorts)
    assert sum(map(len, reconstructed.values())) == reporting.EXPECTED_TUMOR_COUNT
    summary_path = tmp_path / "table_s5.csv"
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    decoded_summary = reporting._read_report_csv(summary_path)  # noqa: SLF001
    assert np.array_equal(
        decoded_summary.loc[:, reporting.BURDEN_SUMMARY_COLUMNS].to_numpy(
            dtype=float,
        ),
        summary.loc[:, reporting.BURDEN_SUMMARY_COLUMNS].to_numpy(dtype=float),
    )
    assert (
        reporting._validate_burden_histogram_summary(  # noqa: SLF001
            histogram,
            decoded_summary,
            threshold,
        )
        == threshold
    )

    tampered = decoded_summary.copy()
    tampered.loc[0, "high_burden_fraction"] = np.nextafter(
        tampered.loc[0, "high_burden_fraction"],
        np.inf,
    )
    tampered.to_csv(summary_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="burden summary"):
        reporting._validate_burden_histogram_summary(  # noqa: SLF001
            histogram,
            reporting._read_report_csv(summary_path),  # noqa: SLF001
            threshold,
        )
    with pytest.raises(ValueError, match="manifest threshold"):
        reporting._validate_burden_histogram_summary(  # noqa: SLF001
            histogram,
            summary,
            threshold + 1,
        )

    disordered = pd.concat(
        [histogram.iloc[[1, 0]], histogram.iloc[2:]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        reporting._burden_values_from_histogram(  # noqa: SLF001
            disordered,
            cohorts,
        )


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
        pair_counts={
            "selected_features": 4,
            "tested_pairs": 4,
            "same_base_pairs_excluded": 2,
            "unfiltered_pair_count": 6,
        },
    )
    assert row["tested_pairs"] == 4
    assert row["same_base_pairs_excluded"] == 2
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


def test_burden_panel_labels_fitted_model_expectation() -> None:
    burden = pd.concat(
        [
            reporting._aggregate_burden_bins(  # noqa: SLF001
                np.asarray([1.0]),
                np.asarray([1.0]),
                provider=provider,
            )
            for provider in reporting.core.BMRS
        ],
        ignore_index=True,
    )
    figure, axis = reporting.plt.subplots()
    try:
        reporting._plot_burden_panel(axis, burden)  # noqa: SLF001
        assert axis.get_ylabel() == "Model-expected selected events (+1)"
    finally:
        reporting.plt.close(figure)


def test_figure6_uses_calibration_and_directional_overlap(tmp_path: Path) -> None:
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
                "cohort": cohort,
                "provider": provider,
                "screen": "marginal_lrt",
                "sentinel_pair_index": pair_index,
                "threshold": threshold,
                "rate": threshold / 2 + pair_index / 100_000,
                "gate_endpoint": provider == "mutsig",
                "clopper_pearson_upper_bound": (
                    threshold * 0.8 + pair_index / 100_000
                    if provider == "mutsig"
                    else np.nan
                ),
                "acceptance_upper_bound": (
                    {0.01: 0.02, 0.05: 0.07}[threshold]
                    if provider == "mutsig"
                    else np.nan
                ),
            }
            for cohort in ("A", "B")
            for provider in reporting.core.BMRS
            for pair_index in range(32)
            for threshold in (0.01, 0.05)
        ],
    )
    confirmation = pd.DataFrame(
        [
            {
                **dict.fromkeys(reporting.confirmation.FINAL_TABLE_COLUMNS, ""),
                "endpoint_id": f"endpoint-{index:04d}",
                "provider": "mutsig",
                "threshold": threshold,
                "acceptance_upper_bound": {0.01: 0.02, 0.05: 0.07}[threshold],
                "final_clopper_pearson_upper_bound": threshold,
                "final_pass": reporting.confirmation.ENDPOINT_ACCEPTED,
            }
            for index, threshold in enumerate([0.01] * 1_024 + [0.05] * 1_024)
        ],
        columns=reporting.confirmation.FINAL_TABLE_COLUMNS,
    )
    output = tmp_path / "figure6.pdf"

    reporting._plot_figure6(  # noqa: SLF001
        burden_bins=burden,
        summary=summary,
        overlap=overlap,
        calibration_table=calibration,
        confirmation_table=confirmation,
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
        output=output,
    )

    assert output.read_bytes().startswith(b"%PDF-")
    if pdfinfo := shutil.which("pdfinfo"):
        info = subprocess.run(  # noqa: S603
            [pdfinfo, output],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "Page size:       540 x 594 pts" in info
    if pdffonts := shutil.which("pdffonts"):
        fonts = subprocess.run(  # noqa: S603
            [pdffonts, output],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "Arial" in fonts
        assert "Type 3" not in fonts


def test_figure6_layout_helpers_keep_clean_ticks_and_panel_order() -> None:
    summary = pd.DataFrame(
        {
            "cohort": ["LOW", "TIE_B", "TIE_A", "HIGH"],
            "cbase_descriptive_primary_rule_crossing_co": [0, 4, 4, 123_071],
            "dig_descriptive_primary_rule_crossing_co": [1, 2, 4, 110_000],
            "mutsig_primary_rejection_co": [0, 4, 3, 0],
        },
    )
    ordered = reporting._ordered_cohort_summary(summary)  # noqa: SLF001

    assert reporting.FIGURE6_PANEL_ORDER == ("A", "B", "C", "D")
    assert reporting.FIGURE6_SIZE_INCHES == (7.5, 8.25)
    assert ordered["cohort"].tolist() == ["LOW", "TIE_A", "TIE_B", "HIGH"]
    assert reporting._co_call_count_ticks(123_071).tolist() == [  # noqa: SLF001
        0,
        1,
        10,
        100,
        1_000,
        10_000,
        100_000,
    ]


def test_figure6_labels_descriptive_threshold_crossings_precisely() -> None:
    source = inspect.getsource(reporting._plot_cohort_panel)  # noqa: SLF001
    assert "CO-direction crossings by cohort" in source
    assert "Pairs crossing {publication_threshold} with CO direction" in source
    assert '"CO calls' not in source


def test_figure6_overlap_matrix_preserves_direction_and_provider_meaning() -> None:
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

    matrix, discordant = reporting._overlap_panel_values(overlap)  # noqa: SLF001

    assert matrix.tolist() == [[3, 2, 2], [1, 1, 0]]
    assert discordant == (0, 2)


def test_table_s5_tex_uses_readable_split_panels_and_compact_headers() -> None:
    summary = pd.DataFrame(
        [
            {
                "cohort": "UCEC",
                "tumors": 517,
                "selected_features": 500,
                "tested_pairs": 124_748,
                "same_base_pairs_excluded": 2,
                "unfiltered_pair_count": 124_750,
                "burden_median": 63.0,
                "burden_q25": 40.0,
                "burden_q75": 384.0,
                "burden_p90": 1520.4,
                "burden_p95": 7070.6,
                "burden_max": 23_753.0,
                "high_burden_fraction": 0.0967,
                **{
                    f"{reporting._decision_prefix(provider, analysis)}_{suffix}": (  # noqa: SLF001
                        1 if provider == "mutsig" and suffix == "total" else 0
                    )
                    for provider in reporting.core.BMRS
                    for analysis in ("primary", "sensitivity")
                    for suffix in ("total", "me", "co", "direction_unavailable")
                },
            },
        ],
    )

    source = reporting._table_s5_tex(  # noqa: SLF001
        summary,
        high_burden_threshold=1974.0,
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
        sensitivity_adjustment="benjamini-hochberg",
        sensitivity_q=0.01,
    )

    assert "A1. Analysis inventory" in source
    assert "A2. Preselection nonsynonymous-SNV burden" in source
    assert "Selected events" in source
    assert "Cohort abbreviations." in source
    assert "UCEC, uterine corpus endometrial carcinoma" in source
    assert "Counts are threshold crossings within each cohort's complete" in source
    assert "BMR, background mutation rate" in source
    assert "MutSig/BY is primary; MutSig/BH is nominal sensitivity" in source
    assert "at or above 1974, the pooled 99th percentile across 10,433 tumors" in source
    assert source.count(r"\clearpage") == 2
    assert source.count(r"\begin{longtable}") == 1
    assert r"BY $q\leq 0.01$" in source
    assert r"BH $q\leq 0.01$" in source
    assert "Primary-rule interpretation" not in source
    assert "decisions direction unavailable" not in source
    assert not any(line.endswith((" ", "\t")) for line in source.splitlines())
    assert max(map(len, source.splitlines())) < 180


def test_calibration_gate_maxima_selects_worst_pair_per_cohort() -> None:
    frame = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "provider": "mutsig",
                "screen": "marginal_lrt",
                "sentinel_pair_index": pair_index,
                "threshold": threshold,
                "gate_endpoint": True,
                "clopper_pearson_upper_bound": threshold + pair_index / 10_000,
                "acceptance_upper_bound": {0.01: 0.02, 0.05: 0.07}[threshold],
            }
            for cohort in ("A", "B")
            for threshold in (0.01, 0.05)
            for pair_index in range(32)
        ],
    )

    maxima = reporting._calibration_gate_maxima(frame)  # noqa: SLF001

    assert len(maxima) == 4
    assert maxima["maximum_clopper_pearson_upper_bound"].tolist() == pytest.approx(
        [0.0131, 0.0531, 0.0131, 0.0531],
    )
    with pytest.raises(ValueError, match="incomplete endpoint family"):
        reporting._calibration_gate_maxima(frame.iloc[:-1])  # noqa: SLF001


def test_fit_diagnostics_cover_every_certificate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    providers = ("cbase", "dig")
    monkeypatch.setattr(reporting.core, "BMRS", providers)
    postprocess_root = tmp_path / "postprocess"
    effects = [
        "full-affine-rank",
        "rank-deficient",
        "rank-not-certified-underflow",
    ]
    values = {
        "cbase": {
            "fit_iterations": [0, 2, 4],
            "fit_last_ll_gain": [0.0, 2e-12, 4e-12],
            "fit_fixed_point_residual": [0.0, 2e-10, 4e-10],
            "fit_kkt_residual": [0.0, 2e-9, 4e-9],
        },
        "dig": {
            "fit_iterations": [1, 3, 5],
            "fit_last_ll_gain": [1e-12, 3e-12, 5e-12],
            "fit_fixed_point_residual": [1e-10, 3e-10, 5e-10],
            "fit_kkt_residual": [1e-9, 3e-9, 5e-9],
        },
    }
    for cohort, indices in (("FIRST", (0, 1)), ("SECOND", (2,))):
        cohort_root = postprocess_root / cohort
        cohort_root.mkdir(parents=True)
        frame: dict[str, object] = {}
        for provider in providers:
            frame[f"{provider}_fit_converged"] = [True] * len(indices)
            for field, provider_values in values[provider].items():
                frame[f"{provider}_{field}"] = [
                    provider_values[index] for index in indices
                ]
            frame[f"{provider}_effect_identifiability"] = [
                effects[index] for index in indices
            ]
        pd.DataFrame(frame).to_csv(
            cohort_root / reporting.postprocess.RESULT_NAME,
            index=False,
        )

    rows = reporting._fit_diagnostic_rows(  # noqa: SLF001
        postprocess_root,
        ("FIRST", "SECOND"),
    )
    by_scope = {str(row["scope"]): row for row in rows}

    assert list(by_scope) == ["all", "cbase", "dig"]
    assert by_scope["all"]["pairwise_rows"] == 6
    assert by_scope["all"]["iterations_min"] == 0
    assert by_scope["all"]["iterations_median"] == 2.5
    assert by_scope["all"]["iterations_p95"] == 4.75
    assert by_scope["all"]["iterations_max"] == 5
    assert by_scope["all"]["minimum_last_ll_gain"] == 0.0
    assert by_scope["all"]["maximum_last_ll_gain"] == 5e-12
    assert by_scope["all"]["maximum_fixed_point_residual"] == 5e-10
    assert by_scope["all"]["maximum_kkt_residual"] == 5e-9
    assert by_scope["all"]["full_affine_rank_rows"] == 2
    assert by_scope["all"]["rank_deficient_rows"] == 2
    assert by_scope["all"]["rank_not_certified_underflow_rows"] == 2
    assert by_scope["cbase"]["iterations_median"] == 2.0
    assert by_scope["cbase"]["iterations_p95"] == 3.8
    assert by_scope["dig"]["iterations_median"] == 3.0
    assert by_scope["dig"]["iterations_p95"] == 4.8
    assert all(row["converged_rows"] == row["pairwise_rows"] for row in rows)
    assert all(row["nonconverged_rows"] == 0 for row in rows)

    tampered = pd.read_csv(
        postprocess_root / "SECOND" / reporting.postprocess.RESULT_NAME,
    )
    tampered.loc[0, "dig_fit_kkt_residual"] = (
        reporting.core.REQUIRED_PAIR_FIT_KKT_TOL * 2
    )
    tampered.to_csv(
        postprocess_root / "SECOND" / reporting.postprocess.RESULT_NAME,
        index=False,
    )
    with pytest.raises(ValueError, match="exceeds the frozen tolerance"):
        reporting._fit_diagnostic_rows(  # noqa: SLF001
            postprocess_root,
            ("FIRST", "SECOND"),
        )


def test_report_validator_binds_inventory_and_bytes(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "report"
    root.mkdir()
    run_root = tmp_path / "run"
    provider_root = tmp_path / "providers"
    postprocess_root = tmp_path / "postprocess"
    calibration_root = tmp_path / "calibration"
    confirmation_root = tmp_path / "confirmation"
    for external_root in (
        run_root,
        provider_root,
        postprocess_root,
        calibration_root,
        confirmation_root,
    ):
        external_root.mkdir()
    features = [f"G{index}_M" for index in range(500)]
    pair_policy = reporting.core._pair_contract(features)  # noqa: SLF001
    contracts_root = run_root / "contracts"
    contracts_root.mkdir()
    for cohort in reporting.TCGA_COHORTS:
        (contracts_root / f"{cohort}.json").write_text(
            reporting.json.dumps(
                {
                    "cohort": cohort,
                    "top_k": 500,
                    "features": features,
                    "pair_policy": pair_policy,
                },
            ),
            encoding="utf-8",
        )
    external_paths = {
        "run_completion": run_root / "completion_manifest.json",
        "provider_manifest": provider_root / "provider_manifest.json",
        "postprocess_manifest": (
            postprocess_root / reporting.postprocess.ROOT_MANIFEST_NAME
        ),
        "calibration_summary": (calibration_root / reporting.calibration.SUMMARY_NAME),
        "calibration_confirmation_summary": (
            confirmation_root / reporting.confirmation.SUMMARY_NAME
        ),
        "calibration_confirmation_final_table": (
            confirmation_root / reporting.confirmation.FINAL_TABLE_NAME
        ),
    }
    for path in external_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    (calibration_root / reporting.calibration.SUMMARY_TABLE_NAME).write_text(
        "placeholder\n",
        encoding="utf-8",
    )
    rule_path = tmp_path / "reporting_rule.json"
    rule_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        reporting,
        "_require_reportable_rule",
        lambda **_kwargs: {
            "inference_status": "reportable",
            "primary_adjustment": "benjamini-yekutieli",
            "primary_q_threshold": 0.01,
            "sensitivity_adjustment": "benjamini-hochberg",
            "sensitivity_q_threshold": 0.01,
        },
    )
    monkeypatch.setattr(reporting, "validate_provider_root", lambda *_args: {})
    monkeypatch.setattr(
        reporting.postprocess,
        "validate_derived_root",
        lambda *_args, **_kwargs: {},
    )
    tumors = [reporting.EXPECTED_TUMOR_COUNT - 31, *([1] * 31)]
    summary_rows = []
    for cohort, tumor_count in zip(reporting.TCGA_COHORTS, tumors, strict=True):
        row = dict.fromkeys(reporting.summary_columns(), 0)
        row.update(
            {
                "cohort": cohort,
                "primary_adjustment": "BY",
                "primary_q_threshold": 0.01,
                "sensitivity_adjustment": "BH",
                "sensitivity_q_threshold": 0.01,
                "tumors": tumor_count,
                "selected_features": 500,
                "tested_pairs": pair_policy["row_count"],
                "same_base_pairs_excluded": pair_policy["same_base_pairs_excluded"],
                "unfiltered_pair_count": pair_policy["unfiltered_row_count"],
                "high_burden_fraction": 1.0,
            },
        )
        summary_rows.append(row)
    pd.DataFrame(
        summary_rows,
        columns=reporting.summary_columns(),
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
            "pairwise_rows": np.full(
                len(reporting.TCGA_COHORTS) * len(reporting.core.BMRS),
                pair_policy["row_count"],
                dtype=np.int64,
            ),
            "elapsed_seconds": 0.0,
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
        },
        columns=reporting.RUNTIME_COLUMNS,
    ).to_csv(root / "runtime_summary.csv", index=False)
    provider_pairwise_rows = len(reporting.TCGA_COHORTS) * pair_policy["row_count"]
    pairwise_rows = provider_pairwise_rows * len(reporting.core.BMRS)
    fit_rows = []
    for scope, row_count in (
        ("all", pairwise_rows),
        *((provider, provider_pairwise_rows) for provider in reporting.core.BMRS),
    ):
        row = dict.fromkeys(reporting.FIT_DIAGNOSTIC_COLUMNS, 0)
        row.update(
            {
                "scope": scope,
                "pairwise_rows": row_count,
                "converged_rows": row_count,
                "full_affine_rank_rows": row_count,
            },
        )
        fit_rows.append(row)
    pd.DataFrame(
        fit_rows,
        columns=reporting.FIT_DIAGNOSTIC_COLUMNS,
    ).to_csv(root / "fit_diagnostics_summary.csv", index=False)
    pd.DataFrame(columns=reporting.top_primary_columns()).to_csv(
        root / "top_primary_pairs.csv",
        index=False,
    )
    (root / "figure6.pdf").write_bytes(b"%PDF-synthetic\n")
    focal_index = list(reporting.TCGA_COHORTS).index(reporting.FOCAL_BURDEN_COHORT)
    focal_tumors = tumors[focal_index]
    pd.DataFrame(
        {
            "cohort": [reporting.FOCAL_BURDEN_COHORT] * len(reporting.core.BMRS),
            "provider": list(reporting.core.BMRS),
            "observed_log1p_bin_lower": [0.0] * len(reporting.core.BMRS),
            "observed_log1p_bin_upper": [0.25] * len(reporting.core.BMRS),
            "expected_log1p_bin_lower": [0.0] * len(reporting.core.BMRS),
            "expected_log1p_bin_upper": [0.25] * len(reporting.core.BMRS),
            "tumor_count": [focal_tumors] * len(reporting.core.BMRS),
        },
    ).to_csv(root / "figure6_burden_bins.csv", index=False)
    pd.DataFrame(
        {
            "cohort": reporting.TCGA_COHORTS,
            "total_nonsynonymous_snv_events": np.zeros(
                len(reporting.TCGA_COHORTS),
                dtype=np.int64,
            ),
            "tumor_count": tumors,
        },
        columns=reporting.COHORT_BURDEN_HISTOGRAM_COLUMNS,
    ).to_csv(root / "cohort_burden_histogram.csv", index=False)
    expected_frames = {
        name: pd.read_csv(root / name) for name in reporting.report_csv_columns()
    }
    (root / "table_s5.tex").write_text(
        reporting._table_s5_tex(  # noqa: SLF001
            expected_frames["table_s5.csv"],
            high_burden_threshold=0.0,
            primary_adjustment="benjamini-yekutieli",
            primary_q=0.01,
            sensitivity_adjustment="benjamini-hochberg",
            sensitivity_q=0.01,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reporting,
        "_report_csv_frames",
        lambda **_kwargs: (expected_frames, 0.0),
    )
    monkeypatch.setattr(
        reporting,
        "_plot_figure6",
        lambda *, output, **_kwargs: output.write_bytes(b"%PDF-synthetic\n"),
    )
    names = {
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
        "burden_source_policy": reporting.BURDEN_SOURCE_POLICY,
        "inputs": {
            "run_completion": reporting._file_record(  # noqa: SLF001
                external_paths["run_completion"],
                relative_to=run_root,
            ),
            "provider_manifest": reporting._file_record(  # noqa: SLF001
                external_paths["provider_manifest"],
                relative_to=provider_root,
            ),
            "postprocess_manifest": reporting._file_record(  # noqa: SLF001
                external_paths["postprocess_manifest"],
                relative_to=postprocess_root,
            ),
            "calibration_summary": reporting._file_record(  # noqa: SLF001
                external_paths["calibration_summary"],
                relative_to=calibration_root,
            ),
            "calibration_confirmation_summary": reporting._file_record(  # noqa: SLF001
                external_paths["calibration_confirmation_summary"],
                relative_to=confirmation_root,
            ),
            "calibration_confirmation_final_table": reporting._file_record(  # noqa: SLF001
                external_paths["calibration_confirmation_final_table"],
                relative_to=confirmation_root,
            ),
            "reporting_rule": {
                "path": rule_path.name,
                "bytes": rule_path.stat().st_size,
                "sha256": reporting._sha256(rule_path),  # noqa: SLF001
            },
        },
        "high_burden_definition": {
            "pooled_tumor_count": reporting.EXPECTED_TUMOR_COUNT,
            "threshold": 0.0,
            "source": "cohort_burden_histogram.csv",
        },
        "outputs": {
            name: reporting._file_record(root / name, relative_to=root)  # noqa: SLF001
            for name in names
        },
    }
    (root / "report_manifest.json").write_bytes(
        reporting._canonical_json(manifest) + b"\n",  # noqa: SLF001
    )
    validate_kwargs = {
        "run_root": run_root,
        "provider_root": provider_root,
        "postprocess_root": postprocess_root,
        "calibration_root": calibration_root,
        "confirmation_root": confirmation_root,
        "rule_path": rule_path,
    }
    assert reporting.validate_report(root, **validate_kwargs)["contract"] == (
        reporting.REPORT_CONTRACT
    )

    (root / "figure6.pdf").write_bytes(b"%PDF-tampered\n")
    with pytest.raises(ValueError, match="output changed"):
        reporting.validate_report(root, **validate_kwargs)

    (root / "figure6.pdf").write_bytes(b"%PDF-synthetic\n")
    leaked = pd.read_csv(root / "table_s5.csv")
    leaked["barcode"] = "forbidden-row-axis"
    leaked.to_csv(root / "table_s5.csv", index=False)
    manifest["outputs"]["table_s5.csv"] = reporting._file_record(  # noqa: SLF001
        root / "table_s5.csv",
        relative_to=root,
    )
    (root / "report_manifest.json").write_bytes(
        reporting._canonical_json(manifest) + b"\n",  # noqa: SLF001
    )
    with pytest.raises(ValueError, match="exact public schema"):
        reporting.validate_report(root, **validate_kwargs)

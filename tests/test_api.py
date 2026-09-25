"""Tests for the public dialect.api surface (estimate_bmr / identify_interactions)."""

import pandas as pd
import pytest

from dialect import (
    IdentifyResult,
    SampleAxisError,
    api,
    estimate_bmr,
    identify_interactions,
)
from dialect.bmr.base import BMRResult
from dialect.models.gene import (
    MARGINAL_FIT_BRACKET_WIDTH_TOL,
    MARGINAL_FIT_CONTRACT,
    MARGINAL_FIT_FIXED_POINT_TOL,
    MARGINAL_FIT_KKT_TOL,
)
from dialect.models.interaction import (
    PAIR_EFFECT_IDENTIFIABILITY_CONTRACT,
    PAIR_IDENTIFIABILITY_RTOL,
)


def _write_cohort(tmp_path):
    counts = pd.DataFrame(
        {
            "TP53_M": [1, 0, 1, 0, 2, 1, 0, 1],
            "KRAS_M": [0, 1, 0, 1, 0, 0, 1, 0],
            "EGFR_M": [1, 1, 0, 0, 1, 0, 0, 1],
        },
        index=[f"s{i}" for i in range(8)],
    )
    counts.rename_axis("sample").to_csv(tmp_path / "count_matrix.csv")
    pmf = [0.7, 0.2, 0.08, 0.02]
    bmr = pd.DataFrame([pmf] * len(counts.columns), index=counts.columns)
    bmr.index.name = "gene"
    bmr.to_csv(tmp_path / "bmr_pmfs.csv")
    return tmp_path / "count_matrix.csv", tmp_path / "bmr_pmfs.csv"


def test_top_level_reexports_match_api():
    assert estimate_bmr is api.estimate_bmr
    assert identify_interactions is api.identify_interactions
    assert IdentifyResult is api.IdentifyResult
    assert SampleAxisError is api.SampleAxisError


def test_identify_interactions_returns_frames(tmp_path):
    counts, bmr = _write_cohort(tmp_path)

    result = api.identify_interactions(counts, bmr, tmp_path, top_k=10)

    assert isinstance(result, api.IdentifyResult)
    assert result.out_dir == tmp_path
    assert set(result.single_gene["Gene Name"]) == {"TP53_M", "KRAS_M", "EGFR_M"}
    assert result.single_gene["Expected Mutations"].tolist() == pytest.approx(
        [3.36] * 3,
    )
    assert result.single_gene["Obs. - Exp. Mutations"].tolist() == pytest.approx(
        [
            2.64,
            -0.36,
            0.64,
        ],
    )
    assert "CBaSE Pos. Sel. Phi" not in result.single_gene
    assert "CBaSE Pos. Sel. P-Val" not in result.single_gene
    assert set(result.single_gene["MLE Algorithm"]) == {MARGINAL_FIT_CONTRACT}
    assert result.single_gene["MLE Converged"].all()
    assert (
        result.single_gene["MLE Bracket Width"] <= MARGINAL_FIT_BRACKET_WIDTH_TOL
    ).all()
    assert (
        result.single_gene["MLE Fixed-Point Residual"] <= MARGINAL_FIT_FIXED_POINT_TOL
    ).all()
    assert (result.single_gene["MLE KKT Residual"] <= MARGINAL_FIT_KKT_TOL).all()
    assert {"Gene A", "Gene B", "Rho"} <= set(result.pairwise.columns)
    assert set(result.pairwise["Effect Identifiability Contract"]) == {
        PAIR_EFFECT_IDENTIFIABILITY_CONTRACT,
    }
    assert set(result.pairwise["Effect Identifiability Relative Tolerance"]) == {
        PAIR_IDENTIFIABILITY_RTOL,
    }
    assert len(result.pairwise) == 3  # 3 choose 2 pairs
    assert (tmp_path / "single_gene_results.csv").exists()
    assert (tmp_path / "pairwise_interaction_results.csv").exists()


def test_estimate_bmr_routes_to_provider(tmp_path):
    (tmp_path / "cohort.maf").write_text("dummy\n")
    pd.DataFrame(
        {
            "GENE": ["G"],
            "ALPHA": [100.0],
            "THETA": [0.3],
            "Pi_MIS": [0.04],
            "Pi_NONS": [0.002],
        },
    ).to_csv(tmp_path / "dig.results.txt", sep="\t", index=False)

    result = api.estimate_bmr(
        tmp_path / "cohort.maf",
        tmp_path,
        provider="dig",
        dig_results=str(tmp_path / "dig.results.txt"),
        n_samples=10,
    )

    assert isinstance(result, BMRResult)
    assert result.provider == "dig"


def _write_same_base_counts(path):
    counts = pd.DataFrame(
        {
            "TP53_M": [1, 0, 1, 1, 0, 1],
            "TP53_N": [0, 1, 0, 0, 1, 0],
            "KRAS_M": [1, 1, 0, 0, 0, 1],
            "IDH1_M": [0, 0, 1, 1, 0, 0],
        },
        index=[f"S{i}" for i in range(6)],
    )
    counts.to_csv(path)


def test_compare_methods_explicit_axis_excludes_same_base_pairs(tmp_path, monkeypatch):
    from dialect.baselines import runner  # noqa: PLC0415

    seen = {}

    def fake_fisher(interactions):
        seen["pairs"] = [(i.gene_a.name, i.gene_b.name) for i in interactions]
        return {
            i.name: {"me_pval": 0.5, "co_pval": 0.5, "me_qval": 0.5, "co_qval": 0.5}
            for i in interactions
        }

    def unavailable(*_args, **_kwargs):
        raise RuntimeError

    monkeypatch.setattr(runner, "run_fishers_exact_analysis", fake_fisher)
    for name in ("run_discover_analysis", "run_megsa_analysis", "run_wesme_analysis"):
        monkeypatch.setattr(runner, name, unavailable)
    counts = tmp_path / "counts.csv"
    _write_same_base_counts(counts)

    api.compare_methods(
        counts,
        tmp_path / "out",
        top_k=3,
        features=["IDH1_M", "TP53_M", "TP53_N"],
        exclude_same_base_pairs=True,
    )

    assert seen["pairs"] == [("IDH1_M", "TP53_M"), ("IDH1_M", "TP53_N")]
    written = pd.read_csv(tmp_path / "out/comparison_pairwise_interaction_results.csv")
    assert len(written) == 2


def test_compare_methods_rejects_mismatched_explicit_axis(tmp_path):
    counts = tmp_path / "counts.csv"
    _write_same_base_counts(counts)

    with pytest.raises(ValueError, match="does not match"):
        api.compare_methods(counts, tmp_path / "out", top_k=2, features=["TP53_M"])
    with pytest.raises(ValueError, match="absent from the count matrix"):
        api.compare_methods(
            counts,
            tmp_path / "out",
            top_k=2,
            features=["TP53_M", "X_M"],
        )


def test_discover_q_values_are_rebased_over_the_tested_family():
    from statsmodels.stats.multitest import multipletests  # noqa: PLC0415

    from dialect.baselines.runner import _rebase_discover_q_values  # noqa: PLC0415

    frame = pd.DataFrame(
        {
            "Discover ME P-Val": [0.01, 0.02, 0.5],
            "Discover CO P-Val": [0.9, 0.001, 0.2],
            "Discover ME Q-Val": [0.0, 0.0, 0.0],
            "Discover CO Q-Val": [0.0, 0.0, 0.0],
        },
    )

    rebased = _rebase_discover_q_values(frame)

    for direction in ("ME", "CO"):
        p_values = frame[f"Discover {direction} P-Val"]
        expected = multipletests(p_values, method="fdr_bh")[1]
        assert rebased[f"Discover {direction} Q-Val"].tolist() == expected.tolist()

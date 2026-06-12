"""Tests for the public dialect.api surface (estimate_bmr / identify_interactions)."""

import pandas as pd

from dialect import IdentifyResult, api, estimate_bmr, identify_interactions
from dialect.bmr.base import BMRResult


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


def test_identify_interactions_returns_frames(tmp_path):
    counts, bmr = _write_cohort(tmp_path)

    result = api.identify_interactions(counts, bmr, tmp_path, top_k=10)

    assert isinstance(result, api.IdentifyResult)
    assert result.out_dir == tmp_path
    assert set(result.single_gene["Gene Name"]) == {"TP53_M", "KRAS_M", "EGFR_M"}
    assert {"Gene A", "Gene B", "Rho"} <= set(result.pairwise.columns)
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

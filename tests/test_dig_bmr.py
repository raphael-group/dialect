"""Tests for the DIG -> DIALECT background-PMF adapter."""

import pandas as pd
import pytest

from dialect.utils.dig_bmr import dig_results_to_bmr_pmfs


def _write_dig(tmp_path, **cols) -> str:
    f = tmp_path / "dig.results.txt"
    pd.DataFrame(cols).to_csv(f, sep="\t", index=False)
    return str(f)


def test_adapter_produces_normalized_per_effect_pmfs(tmp_path) -> None:
    """Each (gene, effect) row is a valid PMF summing to 1, with _M and _N rows."""
    f = _write_dig(
        tmp_path,
        GENE=["SHORT", "LONG"],
        ALPHA=[50.0, 300.0],
        THETA=[0.3, 0.3],
        Pi_MIS=[0.04, 0.04],
        Pi_NONS=[0.002, 0.002],
    )
    out = str(tmp_path / "bmr_pmfs.csv")
    dig_results_to_bmr_pmfs(f, n_samples=36, out=out, max_count=10)
    b = pd.read_csv(out, index_col=0)

    assert set(b.index) == {"SHORT_M", "SHORT_N", "LONG_M", "LONG_N"}
    assert (b.sum(axis=1).round(6) == 1.0).all()


def test_higher_background_gene_has_more_passenger_mass(tmp_path) -> None:
    """A higher-ALPHA gene puts more per-sample mass off zero (the TTN effect)."""
    f = _write_dig(
        tmp_path,
        GENE=["SHORT", "LONG"],
        ALPHA=[50.0, 300.0],
        THETA=[0.3, 0.3],
        Pi_MIS=[0.04, 0.04],
        Pi_NONS=[0.002, 0.002],
    )
    out = str(tmp_path / "bmr_pmfs.csv")
    dig_results_to_bmr_pmfs(f, n_samples=36, out=out, max_count=10)
    b = pd.read_csv(out, index_col=0)

    assert b.loc["LONG_M", "1"] > b.loc["SHORT_M", "1"]
    # Nonsense (rarer than missense) keeps more mass at zero.
    assert b.loc["SHORT_N", "0"] > b.loc["SHORT_M", "0"]


def test_missing_columns_raise(tmp_path) -> None:
    f = _write_dig(tmp_path, GENE=["X"], ALPHA=[1.0])
    with pytest.raises(ValueError, match="missing required columns"):
        dig_results_to_bmr_pmfs(f, n_samples=10, out=str(tmp_path / "o.csv"))

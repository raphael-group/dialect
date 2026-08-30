"""Tests for the DIG -> DIALECT background-PMF adapter."""

import pandas as pd
import pytest
from scipy.stats import nbinom

from dialect.bmr import dig_results_to_bmr_pmfs


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
    assert b.loc["SHORT_M", "9"] + b.loc["SHORT_M", "10"] > 0


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


def test_tail_bound_can_require_support_beyond_fifty(tmp_path) -> None:
    """The emitted native NB tail, not a fixed cap, determines shared support."""
    alpha = 1000.0
    theta = 0.5
    pi_mis = 1.0
    n_samples = 10
    tail_eps = 1e-7
    f = _write_dig(
        tmp_path,
        GENE=["HIGH"],
        ALPHA=[alpha],
        THETA=[theta],
        Pi_MIS=[pi_mis],
        Pi_NONS=[0.0],
    )
    out = str(tmp_path / "bmr_pmfs.csv")

    dig_results_to_bmr_pmfs(
        f,
        n_samples=n_samples,
        out=out,
        tail_eps=tail_eps,
    )

    b = pd.read_csv(out, index_col=0)
    distribution = nbinom(
        alpha / n_samples,
        1.0 / (1.0 + theta * pi_mis),
    )
    expected_kmax = int(distribution.isf(tail_eps))
    observed_kmax = int(b.columns[-1])
    assert expected_kmax > 50
    assert observed_kmax == expected_kmax
    assert distribution.sf(observed_kmax) <= tail_eps
    assert distribution.sf(observed_kmax - 1) > tail_eps


def test_missing_columns_raise(tmp_path) -> None:
    f = _write_dig(tmp_path, GENE=["X"], ALPHA=[1.0])
    with pytest.raises(ValueError, match="missing required columns"):
        dig_results_to_bmr_pmfs(f, n_samples=10, out=str(tmp_path / "o.csv"))


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("n_samples", True, "positive integer"),
        ("n_samples", 10.0, "positive integer"),
        ("max_count", -1, "nonnegative integer"),
        ("max_count", True, "nonnegative integer"),
        ("tail_eps", 0.0, "strictly between"),
        ("tail_eps", 1.0, "strictly between"),
        ("tail_eps", float("nan"), "strictly between"),
    ],
)
def test_converter_rejects_invalid_support_contract(
    tmp_path,
    argument,
    value,
    message,
) -> None:
    f = _write_dig(
        tmp_path,
        GENE=["G"],
        ALPHA=[100.0],
        THETA=[0.3],
        Pi_MIS=[0.04],
        Pi_NONS=[0.002],
    )
    arguments = {
        "dig_results": f,
        "n_samples": 10,
        "out": str(tmp_path / "o.csv"),
        "max_count": None,
        "tail_eps": 1e-7,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        dig_results_to_bmr_pmfs(**arguments)


def test_converter_rejects_duplicate_gene_identifiers(tmp_path) -> None:
    f = _write_dig(
        tmp_path,
        GENE=["G", "G"],
        ALPHA=[100.0, 100.0],
        THETA=[0.3, 0.3],
        Pi_MIS=[0.04, 0.04],
        Pi_NONS=[0.002, 0.002],
    )
    with pytest.raises(ValueError, match="duplicate gene"):
        dig_results_to_bmr_pmfs(f, n_samples=10, out=str(tmp_path / "o.csv"))


def test_converter_rejects_when_no_native_effect_distribution_is_valid(
    tmp_path,
) -> None:
    f = _write_dig(
        tmp_path,
        GENE=["G"],
        ALPHA=[100.0],
        THETA=[0.3],
        Pi_MIS=[float("nan")],
        Pi_NONS=[-0.1],
    )
    with pytest.raises(ValueError, match="no valid gene-effect"):
        dig_results_to_bmr_pmfs(f, n_samples=10, out=str(tmp_path / "o.csv"))

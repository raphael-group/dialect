"""Tests for the pluggable BMRProvider abstraction."""

import pandas as pd
import pytest

from dialect.bmr import (
    BMRProvider,
    BMRResult,
    CBaSEProvider,
    DIGProvider,
    available,
    get_provider,
)


def test_registry_lists_and_resolves() -> None:
    assert set(available()) >= {"cbase", "dig"}
    assert isinstance(get_provider("cbase"), CBaSEProvider)
    with pytest.raises(ValueError, match="Unknown BMR provider"):
        get_provider("nope")


def test_providers_satisfy_protocol() -> None:
    # runtime_checkable Protocol: both backends are structurally BMRProviders.
    assert isinstance(CBaSEProvider(), BMRProvider)
    assert isinstance(DIGProvider("x.txt", 10), BMRProvider)


def test_cbase_provider_load(tmp_path) -> None:
    pd.DataFrame(
        {0: [0.9, 0.8], 1: [0.1, 0.2]},
        index=["TP53_M", "TP53_N"],
    ).rename_axis("gene").to_csv(tmp_path / "bmr_pmfs.csv")
    pd.DataFrame(
        {"TP53_M": [0, 1], "TP53_N": [0, 0]},
        index=["s1", "s2"],
    ).rename_axis("sample").to_csv(tmp_path / "count_matrix.csv")

    result = CBaSEProvider().load(str(tmp_path))

    assert isinstance(result, BMRResult)
    assert result.provider == "cbase"
    assert result.pmfs["TP53_M"] == {0: 0.9, 1: 0.1}
    assert list(result.counts.columns) == ["TP53_M", "TP53_N"]


def test_dig_provider_estimate(tmp_path) -> None:
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
    pd.DataFrame({"G_M": [0, 1]}, index=["s1", "s2"]).rename_axis("sample").to_csv(
        tmp_path / "count_matrix.csv",
    )

    result = DIGProvider(str(tmp_path / "dig.results.txt"), n_samples=10).estimate(
        str(tmp_path / "cohort.maf"),
        str(tmp_path),
    )

    assert isinstance(result, BMRResult)
    assert result.provider == "dig"
    assert {"G_M", "G_N"} <= set(result.pmfs)
    assert abs(sum(result.pmfs["G_M"].values()) - 1.0) < 1e-6

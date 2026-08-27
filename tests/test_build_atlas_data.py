"""Tests for the immutable Atlas K=100 release builder."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis.build_atlas_data import (
    BASELINE_COLUMN_MAP,
    DIALECT_FIELDS,
    DISCOVER_SOURCE_SHA256,
    Source,
    _count_ranked_features,
    _load_valid_count_matrix,
    _load_valid_pmfs,
    _require_exact_tested_features,
    build_release,
    dialect_payload,
)

if TYPE_CHECKING:
    from pathlib import Path


def _dialect_frame() -> pd.DataFrame:
    genes = ["A_M", "B_M", "C_N"]
    directions = [
        ((0.45, 0.25, 0.25, 0.05), 4.0),
        ((0.6, 0.1, 0.1, 0.2), -1.0),
        ((0.25, 0.25, 0.25, 0.25), 0.0),
    ]
    rows = []
    for (gene_a, gene_b), (taus, lrt) in zip(
        combinations(genes, 2),
        directions,
        strict=True,
    ):
        tau00, tau10, tau01, tau11 = taus
        denominator = (
            (tau00 + tau01) * (tau10 + tau11) * (tau00 + tau10) * (tau01 + tau11)
        ) ** 0.5
        rho = (tau11 * tau00 - tau01 * tau10) / denominator
        log_odds = math.log((tau00 * tau11) / (tau01 * tau10))
        wald = log_odds / math.sqrt(
            (1 / tau01) + (1 / tau10) + (1 / tau00) + (1 / tau11),
        )
        observed = (1, 1, 1, 1) if gene_b == "B_M" else (1, 0, 1, 2)
        rows.append(
            {
                "Gene A": gene_a,
                "Gene B": gene_b,
                "Tau_00": tau00,
                "Tau_10": tau10,
                "Tau_01": tau01,
                "Tau_11": tau11,
                "_00_": observed[0],
                "_10_": observed[1],
                "_01_": observed[2],
                "_11_": observed[3],
                "Tau_1X": tau10 + tau11,
                "Tau_X1": tau01 + tau11,
                "Rho": rho,
                "Log Odds Ratio": log_odds,
                "Likelihood Ratio": lrt,
                "Wald Statistic": wald,
            },
        )
    return pd.DataFrame(rows)


def _baseline_frame() -> pd.DataFrame:
    rows = []
    for gene_a, gene_b in combinations(["A_M", "B_M", "C_N"], 2):
        row: dict[str, object] = {}
        for source in BASELINE_COLUMN_MAP:
            if source == "Gene A":
                row[source] = gene_a
            elif source == "Gene B":
                row[source] = gene_b
            elif source == "MEGSA S-Score (LRT)":
                row[source] = 2.0
            else:
                row[source] = 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def _write_fixture(tmp_path: Path) -> tuple[Source, Path, Path]:
    cohort = tmp_path / "source" / "CHOL"
    cohort.mkdir(parents=True)
    count_path = cohort / "count_matrix.csv"
    pd.DataFrame(
        {
            "A_M": [1, 0, 1, 0],
            "B_M": [0, 1, 1, 0],
            "C_N": [0, 0, 1, 0],
        },
        index=["s1", "s2", "s3", "s4"],
    ).to_csv(count_path)
    pmfs = pd.DataFrame({"0": [1.0, 1.0, 1.0]}, index=["A_M", "B_M", "C_N"])
    pmfs.to_csv(cohort / "bmr_pmfs.csv", index_label="gene")
    pmfs.to_csv(cohort / "bmr_pmfs.dig.csv", index_label="gene")
    for bmr in ("cbase", "dig", "mutsig"):
        result_dir = cohort / f"id_{bmr}"
        result_dir.mkdir()
        _dialect_frame().to_csv(
            result_dir / "pairwise_interaction_results.csv",
            index=False,
        )

    baseline_root = tmp_path / "baselines"
    baseline_dir = baseline_root / "TCGA" / "CHOL"
    baseline_dir.mkdir(parents=True)
    comparison_path = baseline_dir / "comparison_pairwise_interaction_results.csv"
    _baseline_frame().to_csv(comparison_path, index=False)
    comparison_hash = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    count_hash = hashlib.sha256(count_path.read_bytes()).hexdigest()
    baseline_metadata = {
        "source_gene_k": 100,
        "cohort": {"id": "TCGA__CHOL"},
        "input": {"sha256": count_hash},
        "top_features": ["A_M", "B_M", "C_N"],
        "method_coverage": {"fixture": {"status": "complete"}},
        "artifacts": {"comparison": {"sha256": comparison_hash}},
    }
    metadata_path = baseline_dir / "metadata.json"
    metadata_path.write_text(json.dumps(baseline_metadata))
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    (baseline_root / "manifest.json").write_text(
        json.dumps(
            {
                "release_id": "fixture-baselines",
                "source_gene_k": 100,
                "release_seed": 20260826,
                "cohort_count": 1,
                "provenance": {
                    "discover": {
                        "version": "0.9.6",
                        "python_source_sha256": DISCOVER_SOURCE_SHA256,
                    },
                },
                "cohorts": [
                    {
                        "id": "TCGA__CHOL",
                        "input": baseline_metadata["input"],
                        "top_features": baseline_metadata["top_features"],
                        "method_coverage": baseline_metadata["method_coverage"],
                        "artifacts": {
                            "comparison": {"sha256": comparison_hash},
                            "metadata": {"sha256": metadata_hash},
                        },
                    },
                ],
            },
        ),
    )
    drivers = tmp_path / "drivers.tsv"
    pd.DataFrame({"Hugo Symbol": ["A", "C"]}).to_csv(
        drivers,
        sep="\t",
        index=False,
    )
    mutsig_root = tmp_path / "mutsig"
    mutsig_cohort = mutsig_root / "CHOL"
    mutsig_cohort.mkdir(parents=True)
    (mutsig_cohort / "persample_genes.txt").write_text("A\nB\nC\n")
    (mutsig_cohort / "persample_patients.txt").write_text("s1\ns2\ns3\ns4\n")
    (mutsig_cohort / "persample_meta.txt").write_text("ng\t3\nnp\t4\nneff\t2\n")
    (mutsig_cohort / "persample_lambda.f32").write_bytes(
        np.zeros(3 * 4 * 2, dtype="<f4").tobytes(),
    )
    return Source("TCGA", cohort.parent, mutsig_root), baseline_root, drivers


def test_dialect_payload_uses_unified_bh_and_preserves_negative_lrt(
    tmp_path: Path,
) -> None:
    result = tmp_path / "pairwise.csv"
    _dialect_frame().to_csv(result, index=False)

    payload = dialect_payload(result, n_samples=4, label="fixture")
    fields = {field: index for index, field in enumerate(DIALECT_FIELDS)}
    rows = payload["rows"]

    assert [row[fields["direction"]] for row in rows] == ["ME", "CO", "neutral"]
    assert rows[1][fields["lrt"]] == -1.0
    assert rows[1][fields["p"]] == 1.0
    assert rows[1][fields["q"]] == 1.0
    assert rows[0][fields["observed_both"]] == 1
    assert rows[0][fields["observed_b_only"]] == 1
    assert rows[0][fields["observed_a_only"]] == 1
    assert rows[0][fields["observed_neither"]] == 1
    assert payload["summary"]["negative_lrt_count"] == 1


def test_build_release_writes_a_hashed_complete_cohort(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    out = tmp_path / "release"

    manifest = build_release(
        out=out,
        baseline_root=baseline_root,
        sources=(source,),
        drivers_path=drivers,
        generated_at="2026-08-26T00:00:00Z",
        require_committed_sources=False,
    )

    assert manifest["immutable"] is True
    assert manifest["coverage"] == {
        "cohorts": 1,
        "cohort_ids_sha256": hashlib.sha256(
            b'["TCGA__CHOL"]',
        ).hexdigest(),
        "studies": {"TCGA": 1},
        "samples": 4,
        "dialect_tables": 3,
        "baseline_tables": 1,
        "mutsig_cbase_fallback_feature_instances": 0,
        "mutsig_pair_rows_with_cbase_fallback": 0,
    }
    index = json.loads((out / "index.json").read_text())
    record = index["cohorts"][0]
    assert record["cancer"] == "Cholangiocarcinoma"
    data_path = out / record["data_file"]
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == record["data_sha256"]
    cohort = json.loads(data_path.read_text())
    assert cohort["drivers"] == ["A", "C"]
    assert set(cohort["models"]) == {"cbase", "dig", "mutsig"}
    assert len(cohort["models"]["cbase"]["rows"]) == 3
    assert len(cohort["baselines"]["rows"]) == 3
    assert cohort["testing_universes"]["summary"] == {
        "common_dialect_features": 3,
        "union_dialect_features": 3,
        "common_dialect_pairs": 3,
        "baseline_pairs_shared_with_all_dialect": 3,
        "mutsig_pairs_with_cbase_fallback": 0,
        "common_dialect_pairs_with_mutsig_fallback": 0,
    }
    assert record["testing_universe"]["model_features"] == {
        "cbase": 3,
        "dig": 3,
        "mutsig": 3,
    }


def test_build_release_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    out = tmp_path / "release"
    out.mkdir()
    (out / "sentinel").write_text("published")

    with pytest.raises(FileExistsError, match="already exists"):
        build_release(
            out=out,
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


def test_build_release_rejects_a_complete_but_stale_lower_k_table(
    tmp_path: Path,
) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    cbase_path = source.root / "CHOL" / "id_cbase" / "pairwise_interaction_results.csv"
    _dialect_frame().iloc[:1].to_csv(cbase_path, index=False)
    out = tmp_path / "release"

    with pytest.raises(ValueError, match="count-ranked K=100"):
        build_release(
            out=out,
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )
    assert not out.exists()


def test_dialect_payload_rejects_an_incomplete_pair_universe(tmp_path: Path) -> None:
    result = tmp_path / "bad.csv"
    _dialect_frame().iloc[:2].to_csv(result, index=False)

    with pytest.raises(ValueError, match="incomplete pair universe"):
        dialect_payload(result, n_samples=4, label="bad")


def test_raw_inputs_allow_colon_source_ids_but_published_rows_do_not(
    tmp_path: Path,
) -> None:
    raw_feature = "Em:AC008101.5_M"
    counts_path = tmp_path / "count_matrix.csv"
    pd.DataFrame({raw_feature: [0, 1]}, index=["s1", "s2"]).to_csv(counts_path)
    pmf_path = tmp_path / "bmr_pmfs.csv"
    pd.DataFrame({"0": [1.0]}, index=[raw_feature]).to_csv(
        pmf_path,
        index_label="gene",
    )

    assert list(_load_valid_count_matrix(counts_path, label="raw").columns) == [
        raw_feature,
    ]
    assert list(_load_valid_pmfs(pmf_path, label="raw").index) == [raw_feature]

    result = tmp_path / "published.csv"
    frame = _dialect_frame()
    frame.loc[frame["Gene A"] == "A_M", "Gene A"] = raw_feature
    frame.to_csv(result, index=False)
    with pytest.raises(ValueError, match="unsafe published gene-effect IDs"):
        dialect_payload(result, n_samples=4, label="published")


def test_count_ranked_features_enforces_provider_specific_exact_set() -> None:
    counts = pd.DataFrame(
        {
            "A_M": [2, 0],
            "B_M": [1, 1],
            "C_M": [0, 2],
            "D_M": [3, 0],
        },
    )
    expected = _count_ranked_features(
        counts,
        {"A_M", "B_M", "C_M", "D_M"},
        "cbase",
        k=3,
    )

    assert expected == ["D_M", "A_M", "B_M"]
    with pytest.raises(ValueError, match="not the exact count-ranked"):
        _require_exact_tested_features(
            {"D_M", "A_M", "C_M"},
            expected,
            label="fixture/cbase",
        )


def test_build_release_rejects_fractional_count_matrix_value(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    count_path = source.root / "CHOL" / "count_matrix.csv"
    counts = pd.read_csv(count_path, index_col=0).astype(float)
    counts.loc["s1", "A_M"] = 0.5
    counts.to_csv(count_path)

    with pytest.raises(ValueError, match="finite nonnegative integers"):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


def test_count_matrix_rejects_duplicate_sample_axis(tmp_path: Path) -> None:
    count_path = tmp_path / "count_matrix.csv"
    pd.DataFrame({"A_M": [0, 1]}, index=["s1", "s1"]).to_csv(count_path)

    with pytest.raises(ValueError, match="duplicate sample identifier"):
        _load_valid_count_matrix(count_path, label="duplicate")


def test_build_release_rejects_malformed_pmf_mass(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    pmf_path = source.root / "CHOL" / "bmr_pmfs.csv"
    pmfs = pd.read_csv(pmf_path, index_col=0)
    pmfs.loc["A_M", "0"] = 0.5
    pmfs.to_csv(pmf_path, index_label="gene")

    with pytest.raises(ValueError, match="PMF row probability mass differs from 1"):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


def test_build_release_rejects_negative_pmf_probability(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    pmf_path = source.root / "CHOL" / "bmr_pmfs.dig.csv"
    pmfs = pd.read_csv(pmf_path, index_col=0)
    pmfs.loc["A_M", "0"] = -1.0
    pmfs.to_csv(pmf_path, index_label="gene")

    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


def test_build_release_rejects_truncated_mutsig_lambda(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    lambda_path = source.mutsig_root / "CHOL" / "persample_lambda.f32"
    lambda_path.write_bytes(lambda_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="MutSig lambda byte length"):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


def test_build_release_rejects_nonfinite_mutsig_lambda(tmp_path: Path) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    lambda_path = source.mutsig_root / "CHOL" / "persample_lambda.f32"
    values = np.zeros(3 * 4 * 2, dtype="<f4")
    values[7] = np.nan
    lambda_path.write_bytes(values.tobytes())

    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("ng\t3\nnp\t4\nneff\t2\nextra\t1\n", "fields must be exactly"),
        ("ng\t3\nnp\t4\nneff\t3\n", "neff must equal 2"),
    ],
)
def test_build_release_rejects_invalid_mutsig_metadata(
    tmp_path: Path,
    metadata: str,
    message: str,
) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    metadata_path = source.mutsig_root / "CHOL" / "persample_meta.txt"
    metadata_path.write_text(metadata)

    with pytest.raises(ValueError, match=message):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )


def test_build_release_rejects_mutsig_axis_dimension_mismatch(
    tmp_path: Path,
) -> None:
    source, baseline_root, drivers = _write_fixture(tmp_path)
    genes_path = source.mutsig_root / "CHOL" / "persample_genes.txt"
    genes_path.write_text("A\nB\n")

    with pytest.raises(ValueError, match="gene axis length"):
        build_release(
            out=tmp_path / "release",
            baseline_root=baseline_root,
            sources=(source,),
            drivers_path=drivers,
            require_committed_sources=False,
        )

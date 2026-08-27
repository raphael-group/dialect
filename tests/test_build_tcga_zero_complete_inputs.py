import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.build_tcga_zero_complete_inputs import build_zero_complete_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    source_root = root / "source"
    mutsig_root = root / "mutsig"
    source_dir = source_root / "CHOL"
    mutsig_dir = mutsig_root / "CHOL"
    source_dir.mkdir(parents=True)
    mutsig_dir.mkdir(parents=True)

    counts = pd.DataFrame(
        {
            "A_M": [2, 0],
            "B_N": [0, 1],
        },
        index=pd.Index(["s2", "s1"], name="sample"),
    )
    counts.to_csv(source_dir / "count_matrix.csv")
    cbase = pd.DataFrame(
        [[0.8, 0.2, 0.0], [0.9, 0.1, 0.0]],
        index=pd.Index(["A_M", "B_N"], name="feature"),
    )
    cbase.to_csv(source_dir / "bmr_pmfs.csv")
    (mutsig_dir / "persample_patients.txt").write_text(
        "s1\ns2\ns3\n",
        encoding="utf-8",
    )

    dig_results = root / "dig.results.txt"
    pd.DataFrame(
        {
            "GENE": ["A", "B"],
            "ALPHA": [2.0, 3.0],
            "THETA": [0.5, 0.25],
            "Pi_MIS": [0.75, 0.6],
            "Pi_NONS": [0.25, 0.4],
        },
    ).to_csv(dig_results, sep="\t", index=False)
    return source_root, mutsig_root, dig_results


def test_build_restores_exact_axis_and_emits_auditable_manifests(tmp_path):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    output_root = tmp_path / "corrected"
    source_count = source_root / "CHOL" / "count_matrix.csv"
    source_cbase = source_root / "CHOL" / "bmr_pmfs.csv"
    count_before = source_count.read_bytes()
    cbase_before = source_cbase.read_bytes()

    result = build_zero_complete_inputs(
        source_root,
        mutsig_root,
        dig_results,
        output_root,
        cbase_mode="copy",
    )

    assert result == output_root.resolve()
    corrected = pd.read_csv(output_root / "CHOL" / "count_matrix.csv", index_col=0)
    assert corrected.index.tolist() == ["s1", "s2", "s3"]
    assert corrected.columns.tolist() == ["A_M", "B_N"]
    assert corrected.loc["s1"].tolist() == [0, 1]
    assert corrected.loc["s2"].tolist() == [2, 0]
    assert corrected.loc["s3"].tolist() == [0, 0]
    assert corrected.sum().tolist() == [2, 1]
    assert int(corrected.to_numpy().max()) == 2

    assert source_count.read_bytes() == count_before
    assert source_cbase.read_bytes() == cbase_before
    assert (output_root / "CHOL" / "bmr_pmfs.csv").read_bytes() == cbase_before
    dig_pmfs = pd.read_csv(output_root / "CHOL" / "bmr_pmfs.dig.csv", index_col=0)
    assert {"A_M", "A_N", "B_M", "B_N"} == set(dig_pmfs.index)
    assert dig_pmfs.shape[1] >= 3

    cohort_manifest = json.loads(
        (output_root / "CHOL" / "input_manifest.json").read_text(),
    )
    root_manifest = json.loads((output_root / "input_manifest.json").read_text())
    assert cohort_manifest["contract"] == "exact-ordered-mutsig-zero-complete-v1"
    assert cohort_manifest["sample_axis"]["old_n"] == 2
    assert cohort_manifest["sample_axis"]["new_n"] == 3
    assert cohort_manifest["sample_axis"]["inserted_ids"] == ["s3"]
    assert cohort_manifest["sample_axis"]["output_zero_row_count"] == 1
    assert cohort_manifest["count_matrix"]["grand_total"] == 3
    assert cohort_manifest["count_matrix"]["max_count"] == 2
    assert (
        cohort_manifest["count_matrix"]["input_ordered_features_sha256"]
        == cohort_manifest["count_matrix"]["output_ordered_features_sha256"]
    )
    assert (
        cohort_manifest["count_matrix"]["input_feature_totals_sha256"]
        == cohort_manifest["count_matrix"]["output_feature_totals_sha256"]
    )
    assert cohort_manifest["count_matrix"]["input_grand_total"] == 3
    assert cohort_manifest["count_matrix"]["output_grand_total"] == 3
    assert cohort_manifest["count_matrix"]["input_max_count"] == 2
    assert cohort_manifest["count_matrix"]["output_max_count"] == 2
    assert cohort_manifest["dig"] == {
        "converter": "dialect.bmr._dig_pmf.dig_results_to_bmr_pmfs",
        "max_count": 2,
        "n_samples": 3,
        "tail_eps": 1e-7,
    }
    assert cohort_manifest["materialization"]["cbase_pmfs"] == "copy"
    assert (
        cohort_manifest["inputs"]["cbase_pmfs"]["sha256"]
        == cohort_manifest["outputs"]["cbase_pmfs"]["sha256"]
        == _sha256(source_cbase)
    )
    assert len(cohort_manifest["outputs"]["dig_pmfs"]["sha256"]) == 64
    assert len(cohort_manifest["count_matrix"]["feature_totals_sha256"]) == 64
    assert len(cohort_manifest["git"]["head"]) == 40
    assert root_manifest["cohorts"] == ["CHOL"]
    assert root_manifest["sample_totals"] == {
        "inserted_count": 1,
        "new_n": 3,
        "old_n": 2,
        "output_zero_row_count": 1,
    }
    assert root_manifest["cohort_manifests"][0]["manifest_sha256"] == _sha256(
        output_root / "CHOL" / "input_manifest.json",
    )


def test_manifests_are_deterministic_across_output_roots(tmp_path):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_zero_complete_inputs(
        source_root,
        mutsig_root,
        dig_results,
        first,
        cbase_mode="copy",
    )
    build_zero_complete_inputs(
        source_root,
        mutsig_root,
        dig_results,
        second,
        cbase_mode="copy",
    )

    assert (first / "CHOL" / "input_manifest.json").read_bytes() == (
        second / "CHOL" / "input_manifest.json"
    ).read_bytes()
    assert (first / "input_manifest.json").read_bytes() == (
        second / "input_manifest.json"
    ).read_bytes()


def test_existing_output_is_never_reused(tmp_path):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    output_root = tmp_path / "corrected"
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing output root"):
        build_zero_complete_inputs(
            source_root,
            mutsig_root,
            dig_results,
            output_root,
        )

    assert marker.read_text(encoding="utf-8") == "do not replace"


@pytest.mark.parametrize("duplicate_axis", ["mutsig", "counts"])
def test_duplicate_sample_identifiers_fail_closed(tmp_path, duplicate_axis):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    if duplicate_axis == "mutsig":
        (mutsig_root / "CHOL" / "persample_patients.txt").write_text(
            "s1\ns2\ns2\n",
            encoding="utf-8",
        )
    else:
        counts = pd.read_csv(source_root / "CHOL" / "count_matrix.csv", index_col=0)
        counts.index = ["s1", "s1"]
        counts.to_csv(source_root / "CHOL" / "count_matrix.csv")
    output_root = tmp_path / "corrected"

    with pytest.raises(ValueError, match="duplicate"):
        build_zero_complete_inputs(
            source_root,
            mutsig_root,
            dig_results,
            output_root,
        )

    assert not output_root.exists()


def test_count_samples_must_be_a_subset_of_mutsig_axis(tmp_path):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    counts_path = source_root / "CHOL" / "count_matrix.csv"
    counts = pd.read_csv(counts_path, index_col=0)
    counts.index = ["s1", "outside"]
    counts.to_csv(counts_path)
    output_root = tmp_path / "corrected"

    with pytest.raises(ValueError, match="absent from the MutSig patient axis"):
        build_zero_complete_inputs(
            source_root,
            mutsig_root,
            dig_results,
            output_root,
        )

    assert not output_root.exists()


def test_preexisting_zero_row_fails_closed(tmp_path):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    counts_path = source_root / "CHOL" / "count_matrix.csv"
    counts = pd.read_csv(counts_path, index_col=0)
    counts.loc["s2"] = 0
    counts.to_csv(counts_path)
    output_root = tmp_path / "corrected"

    with pytest.raises(ValueError, match="already contains all-zero rows"):
        build_zero_complete_inputs(
            source_root,
            mutsig_root,
            dig_results,
            output_root,
        )

    assert not output_root.exists()


@pytest.mark.parametrize("bad_value", [-1, 0.5, "not-a-count"])
def test_invalid_count_values_fail_closed(tmp_path, bad_value):
    source_root, mutsig_root, dig_results = _write_fixture(tmp_path)
    counts_path = source_root / "CHOL" / "count_matrix.csv"
    counts = pd.read_csv(counts_path, index_col=0)
    counts = counts.astype(object)
    counts.loc["s2", "A_M"] = bad_value
    counts.to_csv(counts_path)
    output_root = tmp_path / "corrected"

    with pytest.raises(ValueError, match="Count matrix"):
        build_zero_complete_inputs(
            source_root,
            mutsig_root,
            dig_results,
            output_root,
        )

    assert not output_root.exists()

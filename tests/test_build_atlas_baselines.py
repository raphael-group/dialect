"""Focused contract tests for the immutable Atlas baseline release builder."""

from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest

from analysis.build_atlas_baselines import (
    EXPECTED_COHORT_COUNT,
    EXPECTED_COLUMNS,
    PROBABILITY_COLUMNS,
    CohortSpec,
    SourceSpec,
    canonical_sources,
    discover_cohorts,
    expected_pair_universe,
    stable_cohort_seed,
    top_features,
    validate_comparison_frame,
)


def _write_count_matrix(path, features=("A_M", "B_M", "C_N")):
    frame = pd.DataFrame(
        {feature: [index + 1, 0, index + 1] for index, feature in enumerate(features)},
        index=["S1", "S2", "S3"],
    )
    frame.to_csv(path)


def _valid_comparison(features):
    rows = []
    for gene_a, gene_b in combinations(features, 2):
        row = dict.fromkeys(EXPECTED_COLUMNS, 0.5)
        row["Gene A"] = gene_a
        row["Gene B"] = gene_b
        row["MEGSA S-Score (LRT)"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_canonical_source_lock_is_exactly_71(tmp_path):
    sources = canonical_sources(tmp_path)

    assert [source.study for source in sources] == [
        "TCGA",
        "MSK-IMPACT",
        "MSK-CHORD",
    ]
    assert sum(source.expected_cohorts for source in sources) == EXPECTED_COHORT_COUNT


def test_discover_cohorts_uses_only_locked_source_roots(tmp_path):
    tcga = tmp_path / "pancan"
    msk = tmp_path / "impact"
    legacy = tmp_path / "CHOL"
    for root, cohorts in ((tcga, ("ACC", "CHOL")), (msk, ("Breast",))):
        for cohort in cohorts:
            cohort_dir = root / cohort
            cohort_dir.mkdir(parents=True)
            _write_count_matrix(cohort_dir / "count_matrix.csv")
    legacy.mkdir()
    _write_count_matrix(legacy / "count_matrix.csv")

    cohorts = discover_cohorts(
        (SourceSpec("TCGA", tcga, 2), SourceSpec("MSK", msk, 1)),
        expected_total=3,
    )

    assert cohorts == [
        CohortSpec("TCGA", "ACC", tcga / "ACC/count_matrix.csv"),
        CohortSpec("TCGA", "CHOL", tcga / "CHOL/count_matrix.csv"),
        CohortSpec("MSK", "Breast", msk / "Breast/count_matrix.csv"),
    ]
    assert all(cohort.count_matrix.parent != legacy for cohort in cohorts)


def test_discover_cohorts_fails_on_source_count_drift(tmp_path):
    root = tmp_path / "pancan"
    cohort_dir = root / "ACC"
    cohort_dir.mkdir(parents=True)
    _write_count_matrix(cohort_dir / "count_matrix.csv")

    with pytest.raises(RuntimeError, match="expected 2, found 1"):
        discover_cohorts(
            (SourceSpec("TCGA", root, 2),),
            expected_total=2,
        )


def test_stable_cohort_seed_is_repeatable_and_study_qualified():
    first = stable_cohort_seed(20260826, "TCGA__CHOL")

    assert first == stable_cohort_seed(20260826, "TCGA__CHOL")
    assert first != stable_cohort_seed(20260826, "MSK-CHORD__CHOL")
    assert first != stable_cohort_seed(1, "TCGA__CHOL")
    assert 0 <= first < 2**32


def test_top_features_matches_stable_total_count_order():
    counts = pd.DataFrame(
        {
            "first_tie": [1, 0],
            "second_tie": [0, 1],
            "largest": [2, 1],
        },
    )

    selected = top_features(counts, k=2)

    assert selected == ["largest", "first_tie"]
    assert expected_pair_universe(selected) == {("first_tie", "largest")}


def test_validate_comparison_accepts_full_exact_pair_universe():
    features = ["A_M", "B_M", "C_N"]
    frame = _valid_comparison(features)

    validate_comparison_frame(frame, features)


def test_validate_comparison_rejects_missing_discover_schema():
    features = ["A_M", "B_M", "C_N"]
    frame = _valid_comparison(features).drop(columns="Discover ME P-Val")

    with pytest.raises(ValueError, match=r"missing=.*Discover ME P-Val"):
        validate_comparison_frame(frame, features)


def test_validate_comparison_rejects_duplicate_unordered_pair():
    features = ["A_M", "B_M", "C_N"]
    frame = _valid_comparison(features)
    frame.loc[1, ["Gene A", "Gene B"]] = ["B_M", "A_M"]

    with pytest.raises(ValueError, match="duplicate unordered"):
        validate_comparison_frame(frame, features)


@pytest.mark.parametrize("column", PROBABILITY_COLUMNS)
def test_validate_comparison_rejects_probability_outside_unit_interval(column):
    features = ["A_M", "B_M", "C_N"]
    frame = _valid_comparison(features)
    frame.loc[0, column] = 1.01

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_comparison_frame(frame, features)

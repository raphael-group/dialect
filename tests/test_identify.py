"""Focused tests for DIALECT result serialization."""

from __future__ import annotations

import pandas as pd
import pytest

from dialect.models.gene import Gene
from dialect.utils.identify import (
    SINGLE_GENE_CBASE_ANNOTATION_COLUMNS,
    SINGLE_GENE_COUNT_CONTRACT,
    SINGLE_GENE_RESULT_COLUMN_SEMANTICS,
    SINGLE_GENE_RESULT_COLUMNS,
    SINGLE_GENE_RESULT_COLUMNS_WITH_CBASE,
    create_single_gene_results,
)


def _heterogeneous_fitted_gene() -> Gene:
    """Return a fitted gene with unequal sample-specific passenger expectations."""
    gene = Gene(
        name="G_M",
        samples=["s0", "s1"],
        counts=[1, 2],
        bmr_pmf=[
            {0: 0.5, 1: 0.5},
            {0: 0.25, 1: 0.25, 2: 0.5},
        ],
    )
    gene.estimate_pi_with_mle()
    return gene


def test_single_gene_output_uses_cohort_total_expected_passengers(tmp_path) -> None:
    """Expected and Obs-Exp values share the observed cohort-total scale."""
    output = tmp_path / "single_gene_results.csv"

    create_single_gene_results(
        [_heterogeneous_fitted_gene()],
        str(output),
        cbase_phi_vals_present=False,
    )

    result = pd.read_csv(output)
    assert tuple(result.columns) == SINGLE_GENE_RESULT_COLUMNS
    row = result.iloc[0]
    assert row["Observed Mutations"] == 3
    assert row["Expected Mutations"] == pytest.approx(1.75)
    assert row["Obs. - Exp. Mutations"] == pytest.approx(1.25)
    assert row["Single-Gene Count Contract"] == SINGLE_GENE_COUNT_CONTRACT
    assert not set(SINGLE_GENE_CBASE_ANNOTATION_COLUMNS) & set(result.columns)


def test_single_gene_output_emits_both_cbase_annotations_together(tmp_path) -> None:
    """CBaSE phi and p-value are one optional schema block, never half-present."""
    output = tmp_path / "single_gene_results.csv"
    gene = _heterogeneous_fitted_gene()
    gene.cbase_phi = 1.25
    gene.cbase_p = 0.03

    create_single_gene_results(
        [gene],
        str(output),
        cbase_phi_vals_present=True,
    )

    result = pd.read_csv(output)
    assert tuple(result.columns) == SINGLE_GENE_RESULT_COLUMNS_WITH_CBASE
    assert result.loc[0, "CBaSE Pos. Sel. Phi"] == pytest.approx(1.25)
    assert result.loc[0, "CBaSE Pos. Sel. P-Val"] == pytest.approx(0.03)


def test_empty_single_gene_output_preserves_exact_schema(tmp_path) -> None:
    """An empty result still serializes the complete non-CBaSE header."""
    output = tmp_path / "single_gene_results.csv"

    create_single_gene_results([], str(output), cbase_phi_vals_present=False)

    result = pd.read_csv(output)
    assert result.empty
    assert tuple(result.columns) == SINGLE_GENE_RESULT_COLUMNS


def test_single_gene_schema_defines_every_column_semantic() -> None:
    """Every required and optional output column has one explicit meaning."""
    assert set(SINGLE_GENE_RESULT_COLUMN_SEMANTICS) == set(
        SINGLE_GENE_RESULT_COLUMNS_WITH_CBASE,
    )

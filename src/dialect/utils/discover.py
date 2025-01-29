"""TODO: Add docstring."""

import pandas as pd
from discover import DiscoverMatrix, PairwiseDiscoverResult, pairwise_discover_test


def create_mutation_matrix_from_cnt_df(
    cnt_df: pd.DataFrame,
    top_genes: list,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    top_gene_names = [gene.name for gene in top_genes]
    mutation_matrix = cnt_df[top_gene_names] > 0
    return mutation_matrix.astype(int)


def run_discover_for_interaction(
    events: DiscoverMatrix,
    interaction_type: str,
) -> PairwiseDiscoverResult:
    """TODO: Add docstring."""
    if interaction_type not in {"me", "co"}:
        msg = "interaction_type must be 'me' or 'co'"
        raise ValueError(msg)

    alternative = "less" if interaction_type == "me" else "greater"
    return pairwise_discover_test(events, alternative=alternative, fdr_method="BH")


def extract_discover_results(
    interactions: list,
    me_results: PairwiseDiscoverResult,
    co_results: PairwiseDiscoverResult,
) -> dict:
    """TODO: Add docstring."""
    results = {}

    for interaction in interactions:
        gene_a = interaction.gene_a.name
        gene_b = interaction.gene_b.name
        me_pval = (
            me_results.pvalues.loc[gene_a, gene_b]
            if not pd.isna(me_results.pvalues.loc[gene_a, gene_b])
            else me_results.pvalues.loc[gene_b, gene_a]
        )
        co_pval = (
            co_results.pvalues.loc[gene_a, gene_b]
            if not pd.isna(co_results.pvalues.loc[gene_a, gene_b])
            else co_results.pvalues.loc[gene_b, gene_a]
        )
        me_qval = (
            me_results.qvalues.loc[gene_a, gene_b]
            if not pd.isna(me_results.qvalues.loc[gene_a, gene_b])
            else me_results.qvalues.loc[gene_b, gene_a]
        )
        co_qval = (
            co_results.qvalues.loc[gene_a, gene_b]
            if not pd.isna(co_results.qvalues.loc[gene_a, gene_b])
            else co_results.qvalues.loc[gene_b, gene_a]
        )

        results[interaction.name] = {
            "me_pval": me_pval,
            "co_pval": co_pval,
            "me_qval": me_qval,
            "co_qval": co_qval,
        }

    return results


def run_discover_analysis(
    cnt_df: pd.DataFrame,
    top_genes: list,
    interactions: list,
) -> dict:
    """TODO: Add docstring."""
    mutation_matrix = create_mutation_matrix_from_cnt_df(
        cnt_df,
        top_genes,
    ).T

    events = DiscoverMatrix(mutation_matrix)
    me_results = run_discover_for_interaction(events, interaction_type="me")
    co_results = run_discover_for_interaction(events, interaction_type="co")

    return extract_discover_results(
        interactions,
        me_results,
        co_results,
    )

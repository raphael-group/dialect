import logging
import pandas as pd
from discover import DiscoverMatrix, pairwise_discover_test

# TODO: Create essential and verbose logging info for all methods


def create_mutation_matrix_from_cnt_df(cnt_df, top_genes):
    """
    Creates a binary mutation matrix from the count matrix (cnt_df) using top genes.

    Args:
        cnt_df (pd.DataFrame): Original count matrix with genes as columns and samples as rows.
        top_genes (list): List of top-k Gene objects.

    Returns:
        pd.DataFrame: Binary mutation matrix (genes x samples).
    """
    top_gene_names = [gene.name for gene in top_genes]
    mutation_matrix = cnt_df[top_gene_names] > 0  # Binarize mutation data
    return mutation_matrix.astype(int)


def run_discover_for_interaction(events, interaction_type):
    """
    Runs DISCOVER for a specific interaction type.

    Args:
        events (DiscoverMatrix): DISCOVER matrix object.
        interaction_type (str): 'me' for mutual exclusivity, 'co' for co-occurrence.

    Returns:
        pd.DataFrame: DISCOVER result object for the specified interaction type.
    """
    if interaction_type not in {"me", "co"}:
        raise ValueError("interaction_type must be 'me' or 'co'")

    alternative = "less" if interaction_type == "me" else "greater"
    return pairwise_discover_test(events, alternative=alternative, fdr_method="BH")


def extract_discover_results(interactions, me_results, co_results):
    """
    Extracts DISCOVER results for interaction pairs.

    Args:
        interactions (list): List of Interaction objects.
        me_results: Mutual exclusivity DISCOVER results.
        co_results: Co-occurrence DISCOVER results.

    Returns:
        dict: Dictionary where keys are gene pairs (tuples of Gene names),
              and values are dicts with 'me_qval' and 'co_qval' keys.
    """
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


def run_discover_analysis(cnt_df, top_genes, interactions):
    """
    Main function to run DISCOVER analysis.

    Args:
        cnt_df (pd.DataFrame): Original count matrix with genes as columns and samples as rows.
        top_genes (list): List of top-k Gene objects.
        interactions (list): List of Interaction objects.

    Returns:
        dict: Dictionary of DISCOVER results with gene pairs as keys.
    """
    logging.info("Running DISCOVER analysis...")

    # Step 1: Create binary mutation matrix from cnt_df.
    mutation_matrix = create_mutation_matrix_from_cnt_df(
        cnt_df, top_genes
    ).T  # Transpose since DISCOVER expects genes as rows

    # Step 2: Run DISCOVER analysis
    events = DiscoverMatrix(mutation_matrix)
    me_results = run_discover_for_interaction(events, interaction_type="me")
    co_results = run_discover_for_interaction(events, interaction_type="co")

    # Step 3: Extract results for the specified interactions
    discover_results = extract_discover_results(
        interactions,
        me_results,
        co_results,
    )
    logging.info("Finished running DISCOVER analysis.")
    return discover_results

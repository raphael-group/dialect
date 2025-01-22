import pandas as pd
import logging

from dialect.utils.helpers import (
    check_file_exists,
    initialize_gene_objects,
    initialize_interaction_objects,
)
from dialect.utils.fishers import run_fishers_exact_analysis
from dialect.utils.discover import run_discover_analysis
from dialect.utils.megsa import run_megsa_analysis
from dialect.utils.wesme import run_wesme_analysis
from dialect.models.gene import Gene


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def results_to_dataframe(results, me_pcol, co_pcol, me_qcol, co_qcol):
    """
    Converts a results dictionary into a pandas DataFrame.

    Args:
        results (dict): A dictionary where keys are "gene_a:gene_b" strings
                        and values are dictionaries with "me_qval" and "co_qval".

    Returns:
        pd.DataFrame: A DataFrame with columns 'gene_a', 'gene_b', 'me_qval', and 'co_qval'.
    """
    return pd.DataFrame(
        [
            {
                "Gene A": key.split(":")[0],
                "Gene B": key.split(":")[1],
                me_pcol: vals["me_pval"],
                co_pcol: vals["co_pval"],
                me_qcol: vals["me_qval"],
                co_qcol: vals["co_qval"],
            }
            for key, vals in results.items()
        ]
    )


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def run_comparison_methods(
    cnt_mtx,
    out,
    k,
    gene_level,
):
    logging.info("Running comparison methods")
    check_file_exists(cnt_mtx)
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)

    if k <= 0:
        raise ValueError("k must be a positive integer")

    # TODO: integrate this logic into helper function
    genes = []
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].values
        genes.append(
            Gene(
                name=gene_name,
                samples=cnt_df.index,
                counts=counts,
                bmr_pmf=None,
            )
        )
    top_genes, interactions = initialize_interaction_objects(k, genes)

    logging.info("Running Fisher's exact test...")
    # TODO: modify run_fisher_exact_analysis to directly return a dataframe
    fisher_results = run_fishers_exact_analysis(interactions)
    fisher_df = results_to_dataframe(
        fisher_results,
        "Fisher's ME P-Val",
        "Fisher's CO P-Val",
        "Fisher's ME Q-Val",
        "Fisher's CO Q-Val",
    )

    logging.info("Running DISCOVER...")
    # TODO: modify run_discover_analysis to directly return a dataframe
    discover_results = run_discover_analysis(cnt_df, top_genes, interactions)
    discover_df = results_to_dataframe(
        discover_results,
        "Discover ME P-Val",
        "Discover CO P-Val",
        "Discover ME Q-Val",
        "Discover CO Q-Val",
    )

    logging.info("Running MEGSA...")
    megsa_df = run_megsa_analysis(cnt_df, interactions)

    logging.info("Running WeSME/WeSCO...")
    wesme_df = run_wesme_analysis(cnt_df, out, interactions)

    merged_df = pd.merge(
        fisher_df, discover_df, on=["Gene A", "Gene B"], how="inner"
    )
    merged_df = pd.merge(
        merged_df, megsa_df, on=["Gene A", "Gene B"], how="inner"
    )
    merged_df = pd.merge(
        merged_df, wesme_df, on=["Gene A", "Gene B"], how="inner"
    )
    comparison_interaction_fout = f"{out}/comparison_interaction_results.csv"
    if gene_level:
        comparison_interaction_fout = (
            f"{out}/gene_level_comparison_interaction_results.csv"
        )
    merged_df.to_csv(comparison_interaction_fout, index=False)
    logging.info(f"Comparison results saved to {comparison_interaction_fout}")
    logging.info("Finished running comparison methods")

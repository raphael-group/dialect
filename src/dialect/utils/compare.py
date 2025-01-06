import logging

from dialect.utils.helpers import *
from dialect.utils.fishers import run_fishers_exact_analysis
from dialect.utils.discover import run_discover_analysis
from dialect.utils.megsa import run_megsa_analysis
from dialect.utils.wesme import run_wesme_analysis


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def results_to_dataframe(results, me_col, co_col):
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
                me_col: qvals["me_qval"],
                co_col: qvals["co_qval"],
            }
            for key, qvals in results.items()
        ]
    )


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def run_comparison_methods(cnt_mtx, bmr_pmfs, out, k):
    logging.info("Running comparison methods")
    verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)
    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)

    if k <= 0:
        raise ValueError("k must be a positive integer")

    genes = initialize_gene_objects(cnt_df, bmr_dict)
    top_genes, interactions = initialize_interaction_objects(k, genes.values())

    logging.info("Running Fisher's exact test...")
    # TODO: modify run_fisher_exact_analysis to directly return a dataframe
    fisher_results = run_fishers_exact_analysis(interactions)
    fisher_df = results_to_dataframe(
        fisher_results, "Fisher's ME Q-Val", "Fisher's CO Q-Val"
    )

    logging.info("Running DISCOVER...")
    # TODO: modify run_discover_analysis to directly return a dataframe
    discover_results = run_discover_analysis(cnt_df, top_genes, interactions)
    discover_df = results_to_dataframe(
        discover_results, "Discover ME Q-Val", "Discover CO Q-Val"
    )

    logging.info("Running MEGSA...")
    megsa_df = run_megsa_analysis(cnt_df, interactions)

    logging.info("Running WeSME/WeSCO...")
    wesme_df = run_wesme_analysis(cnt_df, out, interactions)

    # TODO: Implement SELECT Analysis

    merged_df = pd.merge(fisher_df, discover_df, on=["Gene A", "Gene B"], how="inner")
    merged_df = pd.merge(merged_df, megsa_df, on=["Gene A", "Gene B"], how="inner")
    merged_df = pd.merge(merged_df, wesme_df, on=["Gene A", "Gene B"], how="inner")
    comparison_interaction_fout = f"{out}/comparison_interaction_results.csv"
    merged_df.to_csv(comparison_interaction_fout, index=False)
    logging.info(f"Comparison results saved to {comparison_interaction_fout}")
    logging.info("Finished running comparison methods")

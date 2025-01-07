import pandas as pd


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def merge_pairwise_interaction_results(dialect_results, alt_results, out):
    dialect_results_df = pd.read_csv(dialect_results)
    alt_results_df = pd.read_csv(alt_results)
    merged_df = pd.merge(
        dialect_results_df,
        alt_results_df,
        on=["Gene A", "Gene B"],
        how="inner",
    )
    comparison_interaction_fout = f"{out}/complete_pairwise_ixn_results.csv"
    merged_df.to_csv(comparison_interaction_fout, index=False)

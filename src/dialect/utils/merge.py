"""TODO: Add docstring."""


import pandas as pd


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def merge_pairwise_interaction_results(
    dialect_results: str,
    alt_results: str,
    out: str,
) -> None:
    """TODO: Add docstring."""
    dialect_results_df = pd.read_csv(dialect_results)
    alt_results_df = pd.read_csv(alt_results)
    merged_df = dialect_results_df.merge(
        alt_results_df,
        on=["Gene A", "Gene B"],
        how="inner",
    )
    comparison_interaction_fout = f"{out}/complete_pairwise_interaction_results.csv"
    merged_df.to_csv(comparison_interaction_fout, index=False)

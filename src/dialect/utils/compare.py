"""TODO: Add docstring."""

import pandas as pd

from dialect.models.gene import Gene
from dialect.utils.discover import run_discover_analysis
from dialect.utils.fishers import run_fishers_exact_analysis
from dialect.utils.helpers import (
    check_file_exists,
    initialize_interaction_objects,
)
from dialect.utils.megsa import run_megsa_analysis
from dialect.utils.wesme import run_wesme_analysis


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def results_to_dataframe(
    results: dict,
    me_pcol: str,
    co_pcol: str,
    me_qcol: str,
    co_qcol: str,
) -> pd.DataFrame:
    """TODO: Add docstring."""
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
        ],
    )


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def run_comparison_methods(
    cnt_mtx: str,
    out: str,
    k: int,
    gene_level: str,
) -> None:
    """TODO: Add docstring."""
    check_file_exists(cnt_mtx)
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)

    if k <= 0:
        msg = "k must be a positive integer"
        raise ValueError(msg)

    genes = []
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].to_numpy()
        genes.append(
            Gene(
                name=gene_name,
                samples=cnt_df.index,
                counts=counts,
                bmr_pmf=None,
            ),
        )
    top_genes, interactions = initialize_interaction_objects(k, genes)

    fisher_results = run_fishers_exact_analysis(interactions)
    fisher_df = results_to_dataframe(
        fisher_results,
        "Fisher's ME P-Val",
        "Fisher's CO P-Val",
        "Fisher's ME Q-Val",
        "Fisher's CO Q-Val",
    )

    discover_results = run_discover_analysis(cnt_df, top_genes, interactions)
    discover_df = results_to_dataframe(
        discover_results,
        "Discover ME P-Val",
        "Discover CO P-Val",
        "Discover ME Q-Val",
        "Discover CO Q-Val",
    )

    megsa_df = run_megsa_analysis(cnt_df, interactions)

    wesme_df = run_wesme_analysis(cnt_df, out, interactions)

    merged_df = fisher_df.merge(
        discover_df,
        on=["Gene A", "Gene B"],
        how="inner",
    )
    merged_df = merged_df.merge(
        megsa_df,
        on=["Gene A", "Gene B"],
        how="inner",
    )
    merged_df = merged_df.merge(
        wesme_df,
        on=["Gene A", "Gene B"],
        how="inner",
    )
    comparison_interaction_fout = f"{out}/comparison_interaction_results.csv"
    if gene_level:
        comparison_interaction_fout = (
            f"{out}/gene_level_comparison_interaction_results.csv"
        )
    merged_df.to_csv(comparison_interaction_fout, index=False)

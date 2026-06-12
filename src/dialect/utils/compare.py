"""Run alternative ME/CO methods (Fisher, DISCOVER, MEGSA, WeSME) for benchmarking.

Each method runs independently; if one is unavailable (e.g. DISCOVER not installed
or R missing for MEGSA) it is skipped with a logged warning rather than failing the
whole comparison step.
"""

import logging

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
    is_gene_level: bool,
) -> None:
    """Run each comparison method independently; skip any that are unavailable."""
    check_file_exists(cnt_mtx)
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)

    if k <= 0:
        msg = "k must be a positive integer"
        raise ValueError(msg)

    genes = [
        Gene(
            name=gene_name,
            samples=cnt_df.index,
            counts=cnt_df[gene_name].to_numpy(),
            bmr_pmf=None,
        )
        for gene_name in cnt_df.columns
    ]
    top_genes, interactions = initialize_interaction_objects(k, genes)

    method_dfs = []

    def _run(label: str, fn) -> None:
        try:
            method_dfs.append(fn())
            logging.info("Comparison method '%s' completed.", label)
        except Exception:  # noqa: BLE001 - one method failing must not sink the rest
            logging.exception(
                "Comparison method '%s' skipped (unavailable/failed).",
                label,
            )

    _run(
        "Fisher's Exact",
        lambda: results_to_dataframe(
            run_fishers_exact_analysis(interactions),
            "Fisher's ME P-Val",
            "Fisher's CO P-Val",
            "Fisher's ME Q-Val",
            "Fisher's CO Q-Val",
        ),
    )
    _run(
        "DISCOVER",
        lambda: results_to_dataframe(
            run_discover_analysis(cnt_df, top_genes, interactions),
            "Discover ME P-Val",
            "Discover CO P-Val",
            "Discover ME Q-Val",
            "Discover CO Q-Val",
        ),
    )
    _run("MEGSA", lambda: run_megsa_analysis(cnt_df, interactions))
    _run("WeSME", lambda: run_wesme_analysis(cnt_df, out, interactions))

    if not method_dfs:
        msg = "All comparison methods failed; no output written."
        raise RuntimeError(msg)

    merged_df = method_dfs[0]
    for df in method_dfs[1:]:
        merged_df = merged_df.merge(df, on=["Gene A", "Gene B"], how="inner")

    comparison_interaction_fout = f"{out}/comparison_pairwise_interaction_results.csv"
    if is_gene_level:
        comparison_interaction_fout = (
            f"{out}/gene_level_comparison_pairwise_interaction_results.csv"
        )
    merged_df.to_csv(comparison_interaction_fout, index=False)
    logging.info(
        "Wrote comparison results (%d methods) to %s",
        len(method_dfs),
        comparison_interaction_fout,
    )

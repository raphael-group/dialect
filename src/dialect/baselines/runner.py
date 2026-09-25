"""Run alternative ME/CO methods (Fisher, DISCOVER, MEGSA, WeSME) for benchmarking.

Each method runs independently; if one is unavailable (e.g. DISCOVER not installed
or R missing for MEGSA) it is skipped with a logged warning rather than failing the
whole comparison step.
"""

import logging
from collections.abc import Callable, Sequence
from itertools import combinations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from dialect.baselines.discover import run_discover_analysis
from dialect.baselines.fishers import run_fishers_exact_analysis
from dialect.baselines.megsa import run_megsa_analysis
from dialect.baselines.wesme import run_wesme_analysis
from dialect.data.io import check_file_exists
from dialect.models.assembly import initialize_interaction_objects
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

logger = logging.getLogger(__name__)


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


def _base_gene(feature: str) -> str:
    return feature.rsplit("_", 1)[0]


def _explicit_interactions(
    genes: list[Gene],
    features: Sequence[str],
    *,
    exclude_same_base_pairs: bool,
) -> tuple[list[Gene], list[Interaction]]:
    """Build interactions over an explicit ordered feature axis."""
    by_name = {gene.name: gene for gene in genes}
    ordered = [str(feature) for feature in features]
    if len(ordered) < 2 or len(ordered) != len(set(ordered)):  # noqa: PLR2004
        msg = "features must list at least two unique count-matrix columns"
        raise ValueError(msg)
    missing = [feature for feature in ordered if feature not in by_name]
    if missing:
        msg = f"features are absent from the count matrix: {missing[:5]}"
        raise ValueError(msg)
    top_genes = [by_name[feature] for feature in ordered]
    interactions = [
        Interaction(gene_a, gene_b)
        for gene_a, gene_b in combinations(top_genes, 2)
        if not (
            exclude_same_base_pairs
            and _base_gene(gene_a.name) == _base_gene(gene_b.name)
        )
    ]
    return top_genes, interactions


def _select_interactions(
    genes: list[Gene],
    k: int,
    features: Sequence[str] | None,
    *,
    exclude_same_base_pairs: bool,
) -> tuple[list[Gene], list[Interaction]]:
    """Choose the tested axis: historical top-``k`` or an explicit feature list."""
    if features is None and not exclude_same_base_pairs:
        return initialize_interaction_objects(k, genes)
    if features is None:
        top_genes, _ = initialize_interaction_objects(k, genes)
        features = [gene.name for gene in top_genes]
    return _explicit_interactions(
        genes,
        features,
        exclude_same_base_pairs=exclude_same_base_pairs,
    )


def _rebase_discover_q_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute DISCOVER BH q-values over exactly the tested pair family.

    DISCOVER adjusts over every pair of its input matrix. When the tested family is
    a strict subset (same-base pairs excluded), each direction is re-adjusted with
    Benjamini-Hochberg over the emitted p-values so its family matches the others.
    """
    for direction in ("ME", "CO"):
        frame[f"Discover {direction} Q-Val"] = multipletests(
            frame[f"Discover {direction} P-Val"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
    return frame


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def run_comparison_methods(  # noqa: PLR0913
    cnt_mtx: str,
    out: str,
    k: int,
    is_gene_level: bool,
    *,
    features: Sequence[str] | None = None,
    exclude_same_base_pairs: bool = False,
) -> None:
    """Run each comparison method independently; skip any that are unavailable.

    By default the tested features are the top ``k`` by total count. ``features``
    instead fixes the exact ordered axis (``k`` must then equal its length), and
    ``exclude_same_base_pairs`` drops missense/nonsense pairs of one gene before any
    method forms its multiple-testing family. Methods that weight samples (WeSME)
    still read the full count matrix.
    """
    check_file_exists(cnt_mtx)
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)

    if k <= 0:
        msg = "k must be a positive integer"
        raise ValueError(msg)
    if features is not None and len(features) != k:
        msg = f"k={k} does not match the {len(features)} explicit features"
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
    top_genes, interactions = _select_interactions(
        genes,
        k,
        features,
        exclude_same_base_pairs=exclude_same_base_pairs,
    )
    rebase_discover = exclude_same_base_pairs

    method_dfs = []

    def _run(label: str, fn: Callable[[], pd.DataFrame]) -> None:
        try:
            method_dfs.append(fn())
            logger.info("Comparison method '%s' completed.", label)
        except Exception:  # defensive: a failed method must not sink the others
            logger.exception(
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

    def _discover() -> pd.DataFrame:
        frame = results_to_dataframe(
            run_discover_analysis(cnt_df, top_genes, interactions),
            "Discover ME P-Val",
            "Discover CO P-Val",
            "Discover ME Q-Val",
            "Discover CO Q-Val",
        )
        return _rebase_discover_q_values(frame) if rebase_discover else frame

    _run("DISCOVER", _discover)
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
    logger.info(
        "Wrote comparison results (%d methods) to %s",
        len(method_dfs),
        comparison_interaction_fout,
    )

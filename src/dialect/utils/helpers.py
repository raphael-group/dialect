"""Cohort-assembly helpers: Gene / Interaction construction.

The pure I/O helpers now live in :mod:`dialect.data.io` and are re-exported here for
backward compatibility (legacy ``from dialect.utils.helpers import ...`` call sites).
New code should import them from :mod:`dialect.data.io` directly.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING

from dialect.data.io import (
    check_file_exists,
    load_bmr_pmfs,
    load_cnt_mtx_and_bmr_pmfs,
    load_likely_passenger_genes,
    load_putative_driver_genes,
    read_cbase_results_file,
    verify_cnt_mtx_and_bmr_pmfs,
)
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "check_file_exists",
    "initialize_gene_objects",
    "initialize_interaction_objects",
    "load_bmr_pmfs",
    "load_cnt_mtx_and_bmr_pmfs",
    "load_likely_passenger_genes",
    "load_putative_driver_genes",
    "read_cbase_results_file",
    "verify_cnt_mtx_and_bmr_pmfs",
]

logger = logging.getLogger(__name__)


def initialize_gene_objects(cnt_df: pd.DataFrame, bmr_dict: dict) -> dict:
    """Build Gene objects, skipping genes the BMR provider does not cover.

    Different BMR providers model slightly different gene sets (e.g. DIG cannot fit
    a few genes that CBaSE can). Genes present in the count matrix but absent from
    the background model are dropped with a warning rather than hard-failing, so
    DIALECT is robust to the choice of BMR provider.
    """
    genes = {}
    missing = []
    for gene_name in cnt_df.columns:
        bmr_pmf_arr = bmr_dict.get(gene_name)
        if bmr_pmf_arr is None:
            missing.append(gene_name)
            continue
        bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
        genes[gene_name] = Gene(
            name=gene_name,
            samples=cnt_df.index,
            counts=cnt_df[gene_name].to_numpy(),
            bmr_pmf=bmr_pmf,
        )
    if missing:
        logger.warning(
            "Dropped %d/%d genes with no background PMF from the BMR provider "
            "(e.g. %s).",
            len(missing),
            len(cnt_df.columns),
            ", ".join(missing[:5]),
        )
    return genes


def initialize_interaction_objects(k: int, genes: list) -> tuple:
    """Build pairwise Interaction objects over the top-``k`` genes by total count."""
    interactions = []
    top_genes = sorted(genes, key=lambda x: sum(x.counts), reverse=True)[:k]
    for gene_a, gene_b in combinations(top_genes, 2):
        interactions.append(Interaction(gene_a, gene_b))
    return top_genes, interactions

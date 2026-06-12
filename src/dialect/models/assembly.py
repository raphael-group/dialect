"""Assemble model objects (Gene / Interaction) from a cohort's raw data.

This is the ``data`` -> ``models`` boundary: it consumes a cohort's raw count
matrix + background PMFs and constructs the :class:`~dialect.models.gene.Gene` and
:class:`~dialect.models.interaction.Interaction` objects the EM operates on, plus
the optional CBaSE positive-selection annotation. It lives in ``models`` (not
``data``) because it builds model objects.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

if TYPE_CHECKING:
    import pandas as pd

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


def save_cbase_stats_to_gene_objects(genes: dict, cbase_stats: pd.DataFrame) -> bool:
    """Annotate Gene objects with CBaSE positive-selection phi/p.

    Returns ``True`` if the annotation was applied, ``False`` if no CBaSE stats were
    provided (so callers can drop the now-empty phi column from their output).
    """
    if cbase_stats is None or cbase_stats.empty:
        return False

    missense_gene_to_positive_selection_phi = {
        f"{row['gene']}_M": row["phi_m_pos_or_p(m=0|s)"]
        for _, row in cbase_stats.iterrows()
    }
    missense_gene_to_positive_selection_p = {
        f"{row['gene']}_M": row["p_phi_m_pos"] for _, row in cbase_stats.iterrows()
    }

    nonsense_gene_to_positive_selection_phi = {
        f"{row['gene']}_N": row["phi_k_pos_or_p(k=0|s)"]
        for _, row in cbase_stats.iterrows()
    }
    nonsense_gene_to_positive_selection_p = {
        f"{row['gene']}_N": row["p_phi_k_pos"] for _, row in cbase_stats.iterrows()
    }

    gene_to_positive_selection_phi = {
        **missense_gene_to_positive_selection_phi,
        **nonsense_gene_to_positive_selection_phi,
    }
    gene_to_positive_select_p = {
        **missense_gene_to_positive_selection_p,
        **nonsense_gene_to_positive_selection_p,
    }

    for name, gene in genes.items():
        if name not in gene_to_positive_selection_phi:
            msg = f"Gene {name} not found in the CBaSE results file."
            raise ValueError(msg)
        gene.cbase_phi = gene_to_positive_selection_phi[name]
        gene.cbase_p = gene_to_positive_select_p[name]

    return True

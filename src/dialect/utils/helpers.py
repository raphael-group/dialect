"""TODO: Add docstring."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


def verify_cnt_mtx_and_bmr_pmfs(cnt_mtx: str, bmr_pmfs: str) -> None:
    """TODO: Add docstring."""
    check_file_exists(cnt_mtx)
    check_file_exists(bmr_pmfs)


def check_file_exists(fn: str) -> None:
    """TODO: Add docstring."""
    if not Path(fn).exists():
        msg = f"File not found: {fn}"
        raise FileNotFoundError(msg)


def load_bmr_pmfs(bmr_pmfs: str) -> dict:
    """TODO: Add docstring."""
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")
    return {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}


def load_cnt_mtx_and_bmr_pmfs(cnt_mtx: str, bmr_pmfs: str) -> tuple:
    """TODO: Add docstring."""
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_dict = load_bmr_pmfs(bmr_pmfs)
    return cnt_df, bmr_dict


def initialize_gene_objects(cnt_df: pd.DataFrame, bmr_dict: dict) -> dict:
    """TODO: Add docstring."""
    genes = {}
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].to_numpy()
        bmr_pmf_arr = bmr_dict.get(gene_name)
        if bmr_pmf_arr is None:
            msg = f"No BMR PMF found for gene {gene_name}"
            raise ValueError(msg)
        bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
        genes[gene_name] = Gene(
            name=gene_name,
            samples=cnt_df.index,
            counts=counts,
            bmr_pmf=bmr_pmf,
        )
    return genes


def initialize_interaction_objects(k: int, genes: list) -> tuple:
    """TODO: Add docstring."""
    interactions = []
    top_genes = sorted(genes, key=lambda x: sum(x.counts), reverse=True)[:k]
    for gene_a, gene_b in combinations(top_genes, 2):
        interactions.append(Interaction(gene_a, gene_b))
    return top_genes, interactions


def read_cbase_results_file(cbase_stats_fn: str) -> pd.DataFrame | None:
    """TODO: Add docstring."""
    if cbase_stats_fn is None:
        return None

    try:
        cbase_stats_df = pd.read_csv(cbase_stats_fn, sep="\t", skiprows=1)
    except FileNotFoundError:
        return None

    return cbase_stats_df

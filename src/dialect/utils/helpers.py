"""TODO: Add docstring."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

logger = logging.getLogger(__name__)
_PMF_SUM_TOL = 1e-6


def load_likely_passenger_genes(likely_passenger_dir: Path) -> set:
    """TODO: Add docstring."""
    subtype_to_likely_passengers = {}
    for likely_passenger_fn in likely_passenger_dir.iterdir():
        if likely_passenger_fn.suffix == ".md":
            continue
        subtype = likely_passenger_fn.stem
        likely_passengers = pd.read_csv(
            likely_passenger_fn, header=None, names=["Gene"],
        )["Gene"]
        subtype_to_likely_passengers[subtype] = set(likely_passengers)
    return subtype_to_likely_passengers


def load_putative_driver_genes(driver_filepath: Path) -> set:
    """TODO: Add docstring."""
    driver_df = pd.read_csv(driver_filepath, sep="\t", index_col=0)
    return set(driver_df.index + "_M") | set(driver_df.index + "_N")


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
    """Load per-gene background PMFs, stripping NaN padding and renormalizing.

    Each row is expected to be a proper PMF summing to 1. Aggressive tail
    truncation (e.g. a coarse CBaSE THRESHOLD) can leave a row summing to < 1;
    such rows are renormalized and a warning is logged, because every downstream
    likelihood/EM step assumes a normalized P(B).
    """
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")
    pmfs = {}
    n_renormalized = 0
    for key, raw in bmr_dict.items():
        pmf = [x for x in raw if not np.isnan(x)]
        total = sum(pmf)
        if total <= 0:
            msg = f"BMR PMF for {key} sums to {total}; cannot normalize."
            raise ValueError(msg)
        if abs(total - 1.0) > _PMF_SUM_TOL:
            pmf = [x / total for x in pmf]
            n_renormalized += 1
        pmfs[key] = pmf
    if n_renormalized:
        logger.warning(
            "Renormalized %d/%d BMR PMFs that did not sum to 1 "
            "(likely tail truncation); check the BMR threshold.",
            n_renormalized,
            len(bmr_dict),
        )
    return pmfs


def load_cnt_mtx_and_bmr_pmfs(cnt_mtx: str, bmr_pmfs: str) -> tuple:
    """TODO: Add docstring."""
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_dict = load_bmr_pmfs(bmr_pmfs)
    return cnt_df, bmr_dict


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

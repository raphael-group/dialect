"""I/O for DIALECT's data contract: count matrices and background PMFs.

The data layer is the base of the dependency DAG -- it has no internal DIALECT
imports, so every higher layer may depend on it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_PMF_SUM_TOL = 1e-6


def check_file_exists(fn: str) -> None:
    """Raise :class:`FileNotFoundError` if ``fn`` does not exist."""
    if not Path(fn).exists():
        msg = f"File not found: {fn}"
        raise FileNotFoundError(msg)


def verify_cnt_mtx_and_bmr_pmfs(cnt_mtx: str, bmr_pmfs: str) -> None:
    """Verify the count-matrix and BMR-PMF files both exist."""
    check_file_exists(cnt_mtx)
    check_file_exists(bmr_pmfs)


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
    """Load a count matrix (DataFrame) and its per-gene BMR PMFs (dict)."""
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_dict = load_bmr_pmfs(bmr_pmfs)
    return cnt_df, bmr_dict


def read_cbase_results_file(cbase_stats_fn: str | None) -> pd.DataFrame | None:
    """Read a CBaSE ``q_values.txt`` selection-stats file, or ``None``."""
    if cbase_stats_fn is None:
        return None
    try:
        return pd.read_csv(cbase_stats_fn, sep="\t", skiprows=1)
    except FileNotFoundError:
        return None


def load_likely_passenger_genes(likely_passenger_dir: Path) -> dict[str, set]:
    """Load per-subtype likely-passenger gene sets from a directory of files."""
    subtype_to_likely_passengers = {}
    for likely_passenger_fn in likely_passenger_dir.iterdir():
        if likely_passenger_fn.suffix == ".md":
            continue
        subtype = likely_passenger_fn.stem
        genes = pd.read_csv(likely_passenger_fn, header=None, names=["Gene"])["Gene"]
        subtype_to_likely_passengers[subtype] = set(genes)
    return subtype_to_likely_passengers


def load_putative_driver_genes(driver_filepath: Path) -> set:
    """Load putative driver genes (suffixed ``_M``/``_N``) from a TSV."""
    driver_df = pd.read_csv(driver_filepath, sep="\t", index_col=0)
    return set(driver_df.index + "_M") | set(driver_df.index + "_N")

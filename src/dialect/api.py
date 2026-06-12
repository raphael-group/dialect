"""DIALECT's public Python API.

Stable, importable entry points for the two core operations -- estimating a
background mutation rate and identifying mutual-exclusivity / co-occurrence
interactions -- plus the dataclass they return. This is the surface the CLI,
agents, notebooks, and the web backend all call into; everything below it
(``models``, ``bmr``, ``data``) is an implementation detail.

Example:
    >>> from dialect import api
    >>> bmr = api.estimate_bmr("data/mafs/BRCA.maf", "out/BRCA", provider="cbase")
    >>> result = api.identify_interactions(
    ...     "out/BRCA/count_matrix.csv", "out/BRCA/bmr_pmfs.csv", "out/BRCA",
    ... )
    >>> result.pairwise.sort_values("Rho").head()  # strongest mutual exclusivity
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from dialect.bmr import get_provider
from dialect.data.io import read_cbase_results_file
from dialect.utils.identify import identify_pairwise_interactions

if TYPE_CHECKING:
    from dialect.bmr.base import BMRResult

__all__ = ["IdentifyResult", "estimate_bmr", "identify_interactions"]

_DEFAULT_TOP_K = 100


def estimate_bmr(
    maf: str | Path,
    out_dir: str | Path,
    *,
    provider: str = "cbase",
    reference: str = "hg19",
    **provider_kwargs: object,
) -> BMRResult:
    """Estimate a background mutation rate for a cohort MAF.

    Thin, typed wrapper over the BMR provider registry: resolves ``provider``,
    runs it on ``maf``, and returns the :class:`~dialect.bmr.base.BMRResult`
    (per-gene PMFs + count matrix), also written under ``out_dir`` as
    ``bmr_pmfs.csv`` / ``count_matrix.csv``.

    Args:
        maf: path to a TCGA-style MAF for one cohort.
        out_dir: directory for the provider's outputs (created if needed).
        provider: registered provider name; see ``dialect.bmr.available()``.
        reference: genome build passed to the provider (CBaSE: ``hg19``/``hg38``).
        **provider_kwargs: forwarded to the provider constructor (e.g.
            ``threshold`` for CBaSE; ``dig_results`` + ``n_samples`` for DIG).

    Returns:
        The provider's :class:`BMRResult`.
    """
    prov = get_provider(provider, **provider_kwargs)
    return prov.estimate(str(maf), str(out_dir), reference=reference)


@dataclass(frozen=True)
class IdentifyResult:
    """Structured result of :func:`identify_interactions`.

    Attributes:
        single_gene: per-gene driver estimates (``Pi`` and related statistics).
        pairwise: per-pair interaction statistics (``Rho``, ``Tau_*``, ...).
        out_dir: directory the CSVs were written to.
    """

    single_gene: pd.DataFrame
    pairwise: pd.DataFrame
    out_dir: Path


def identify_interactions(
    counts: str | Path,
    bmr_pmfs: str | Path,
    out_dir: str | Path,
    *,
    top_k: int = _DEFAULT_TOP_K,
    cbase_stats: str | Path | pd.DataFrame | None = None,
) -> IdentifyResult:
    """Identify mutual-exclusivity / co-occurrence interactions for a cohort.

    Runs DIALECT's EM over a background-corrected count matrix: estimates each
    gene's driver rate ``pi``, then fits the bivariate-Bernoulli ``tau`` for the
    top-``top_k`` genes' pairs. Writes ``single_gene_results.csv`` and
    ``pairwise_interaction_results.csv`` under ``out_dir`` and returns them.

    Args:
        counts: path to the cohort ``count_matrix.csv`` (BMR-agnostic counts).
        bmr_pmfs: path to the matching ``bmr_pmfs.csv`` (per-gene background PMFs).
        out_dir: output directory (created if needed).
        top_k: number of top genes (by total count) to pair.
        cbase_stats: optional CBaSE selection stats -- a path to ``q_values.txt``
            or an already-loaded DataFrame -- used to annotate genes with
            positive-selection phi/p; ignored if None.

    Returns:
        An :class:`IdentifyResult` with the single-gene and pairwise frames.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = (
        cbase_stats
        if isinstance(cbase_stats, pd.DataFrame) or cbase_stats is None
        else read_cbase_results_file(str(cbase_stats))
    )
    identify_pairwise_interactions(
        cnt_mtx=str(counts),
        bmr_pmfs=str(bmr_pmfs),
        out=str(out),
        k=top_k,
        cbase_stats=stats,
    )
    return IdentifyResult(
        single_gene=pd.read_csv(out / "single_gene_results.csv"),
        pairwise=pd.read_csv(out / "pairwise_interaction_results.csv"),
        out_dir=out,
    )

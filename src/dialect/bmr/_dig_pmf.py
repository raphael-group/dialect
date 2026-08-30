"""Convert DIGDriver gene-model output into DIALECT background PMFs.

DIGDriver's gene model (``DigDriver.py geneDriver``) emits, per gene, a
negative-binomial (NB) background for the gene's somatic SNV count plus
per-effect fractions ``Pi_MIS``, ``Pi_NONS``, ... . DIALECT instead consumes a
*per-sample* background count PMF ``P(B=k)`` per gene, split by effect
(``_M`` missense / ``_N`` nonsense) -- the same contract CBaSE produces in
``bmr_pmfs.csv``. This module bridges the two so DIG can serve as an alternative
BMR provider for the BMR-sensitivity analysis reviewers asked for.

Derivation (Gamma-Poisson / negative-binomial):

- DIG models a gene's *cohort* SNV count as Gamma-Poisson with shape ``ALPHA``
  and scale ``THETA``  =>  count ~ ``NB(n=ALPHA, p=1/(1+THETA))`` with mean
  ``ALPHA*THETA`` (this is exactly how DIG forms ``EXP_SNV = ALPHA*THETA*Pi_SUM``).
- Thinning to one effect class ``e`` (each SNV is class ``e`` w.p. ``Pi_e``)
  scales the Gamma scale by ``Pi_e``  =>  class-``e`` cohort count
  ~ ``NB(ALPHA, 1/(1+THETA*Pi_e))`` with mean ``ALPHA*THETA*Pi_e = EXP_e``.
- The cohort count is the sum over ``N`` i.i.d. samples; by infinite
  divisibility of the NB, the *per-sample* class-``e`` count
  ~ ``NB(ALPHA/N, 1/(1+THETA*Pi_e))`` with mean ``EXP_e / N``.

This yields ``P(B_i = k)`` per (gene, effect), matching DIALECT's input format.
The per-sample mean is tiny (<< 1), so the result is robust to the NB-vs-Poisson
choice; the NB form is used to preserve DIG's overdispersion.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import nbinom

# DIALECT effect suffix -> DIG per-effect fraction column.
_EFFECTS = {"_M": "Pi_MIS", "_N": "Pi_NONS"}
_REQUIRED = {"GENE", "ALPHA", "THETA", *(_EFFECTS.values())}


class _DiscreteTail(Protocol):
    """Subset of a frozen SciPy discrete distribution used for support search."""

    def isf(self, q: float) -> float:
        """Return the inverse survival-function endpoint."""

    def sf(self, k: int) -> float:
        """Return the upper-tail probability beyond ``k``."""


def _minimal_tail_endpoint(
    distribution: _DiscreteTail,
    tail_eps: float,
) -> int:
    """Return the smallest integer endpoint with upper tail at most ``tail_eps``."""
    raw_endpoint = float(distribution.isf(tail_eps))
    if not math.isfinite(raw_endpoint) or raw_endpoint < 0:
        msg = "DIG negative-binomial tail endpoint is nonfinite or negative"
        raise ValueError(msg)
    endpoint = int(raw_endpoint)
    while float(distribution.sf(endpoint)) > tail_eps:
        endpoint += 1
    while endpoint > 0 and float(distribution.sf(endpoint - 1)) <= tail_eps:
        endpoint -= 1
    if not math.isfinite(float(distribution.sf(endpoint))):
        msg = "DIG negative-binomial upper-tail probability is nonfinite"
        raise ValueError(msg)
    return endpoint


def _nb_params(
    alpha: float,
    theta: float,
    pi: float,
    n_samples: int,
) -> tuple[float, float]:
    """Per-sample NB (n, p) for one effect class (ALPHA split by N, effect-thinned)."""
    return alpha / n_samples, 1.0 / (1.0 + theta * pi)


def _validate_support_contract(
    n_samples: int,
    max_count: int | None,
    tail_eps: float,
) -> tuple[int, int, float]:
    """Validate and normalize the finite-support arguments."""
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, Integral)
        or n_samples <= 0
    ):
        msg = "n_samples must be a positive integer"
        raise ValueError(msg)
    if max_count is not None and (
        isinstance(max_count, (bool, np.bool_))
        or not isinstance(max_count, Integral)
        or max_count < 0
    ):
        msg = "max_count must be a nonnegative integer or None"
        raise ValueError(msg)
    if (
        isinstance(tail_eps, (bool, np.bool_))
        or not isinstance(tail_eps, Real)
        or not math.isfinite(float(tail_eps))
        or not 0.0 < float(tail_eps) < 1.0
    ):
        msg = "tail_eps must be finite and strictly between zero and one"
        raise ValueError(msg)
    return int(n_samples), int(max_count or 0), float(tail_eps)


def _validate_gene_axis(df: pd.DataFrame) -> None:
    """Reject ambiguous gene identifiers before constructing effect keys."""
    raw_genes = df["GENE"]
    if raw_genes.isna().any():
        msg = "DIG results contain a missing gene identifier"
        raise ValueError(msg)
    genes = raw_genes.astype(str)
    if any(not gene or gene != gene.strip() for gene in genes):
        msg = "DIG gene identifiers must be nonempty and free of outer whitespace"
        raise ValueError(msg)
    if genes.duplicated().any():
        msg = "DIG results contain duplicate gene identifiers"
        raise ValueError(msg)


def _native_parameter(value: object, *, column: str, gene: str) -> float:
    """Return one finite native DIG parameter or fail closed."""
    if isinstance(value, (bool, np.bool_)):
        msg = f"DIG {column} must be numeric for gene {gene}"
        raise TypeError(msg)
    try:
        parameter = float(value)
    except (TypeError, ValueError) as error:
        msg = f"DIG {column} must be numeric for gene {gene}"
        raise ValueError(msg) from error
    if not math.isfinite(parameter):
        msg = f"DIG {column} must be finite for gene {gene}"
        raise ValueError(msg)
    return parameter


def _collect_parameters(
    df: pd.DataFrame,
    n_samples: int,
    max_count: int,
    tail_eps: float,
) -> tuple[dict[str, tuple[float, float, float]], int]:
    """Collect native effect parameters and the shared minimal support endpoint."""
    params: dict[str, tuple[float, float, float]] = {}
    endpoint = max(1, max_count)
    for _, row in df.iterrows():
        gene = str(row["GENE"])
        alpha = _native_parameter(row["ALPHA"], column="ALPHA", gene=gene)
        theta = _native_parameter(row["THETA"], column="THETA", gene=gene)
        if alpha <= 0:
            msg = f"DIG ALPHA must be positive for gene {gene}"
            raise ValueError(msg)
        if theta <= 0:
            msg = f"DIG THETA must be positive for gene {gene}"
            raise ValueError(msg)
        for suffix, pi_col in _EFFECTS.items():
            pi = _native_parameter(row[pi_col], column=pi_col, gene=gene)
            if not 0.0 <= pi <= 1.0:
                msg = f"DIG {pi_col} must be between zero and one for gene {gene}"
                raise ValueError(msg)
            key = f"{gene}{suffix}"
            params[key] = (alpha, theta, pi)
            if pi > 0:
                distribution = nbinom(*_nb_params(alpha, theta, pi, n_samples))
                endpoint = max(
                    endpoint,
                    _minimal_tail_endpoint(distribution, tail_eps),
                )
    if not params:
        msg = "DIG results contain no valid gene-effect background distributions"
        raise ValueError(msg)
    return params, endpoint


def _build_normalized_pmfs(
    params: dict[str, tuple[float, float, float]],
    n_samples: int,
    endpoint: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Evaluate and normalize every native effect distribution on shared support."""
    counts = np.arange(endpoint + 1)
    pmfs: dict[str, np.ndarray] = {}
    for key, (alpha, theta, pi) in params.items():
        if pi <= 0:
            pmf = np.zeros(endpoint + 1)
            pmf[0] = 1.0
        else:
            pmf = nbinom(*_nb_params(alpha, theta, pi, n_samples)).pmf(counts)
            total = float(math.fsum(float(value) for value in pmf))
            if not math.isfinite(total) or total <= 0:
                msg = (
                    f"DIG native probabilities do not have positive finite mass: {key}"
                )
                raise ValueError(msg)
            pmf = pmf / total
        pmfs[key] = pmf
    return pmfs, counts


def dig_results_to_bmr_pmfs(
    dig_results: str,
    n_samples: int,
    out: str,
    *,
    max_count: int | None = None,
    tail_eps: float = 1e-7,
) -> str:
    """Write a DIALECT ``bmr_pmfs.csv`` from a DIG geneDriver results file.

    :param dig_results: path to ``*.results.txt`` from ``DigDriver.py geneDriver``.
    :param n_samples: number of tumor samples in the cohort DIG was run on.
    :param out: output path for the DIALECT-format ``bmr_pmfs.csv``.
    :param max_count: if given, ensure the PMF support covers at least this count
        (e.g. the max observed count in the cohort's count matrix).
    :param tail_eps: maximum omitted upper-tail mass before normalization.

    All effects share the inclusive support ``0..K``, where ``K`` covers
    ``max_count`` and every effect-specific negative-binomial ``tail_eps``
    quantile. No fixed count ceiling or probability floor is applied.
    """
    df = pd.read_csv(dig_results, sep="\t")
    missing = _REQUIRED - set(df.columns)
    if missing:
        msg = f"DIG results missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    n_samples, maximum, tail_eps = _validate_support_contract(
        n_samples,
        max_count,
        tail_eps,
    )
    _validate_gene_axis(df)
    params, endpoint = _collect_parameters(df, n_samples, maximum, tail_eps)
    pmfs, counts = _build_normalized_pmfs(params, n_samples, endpoint)

    mat = pd.DataFrame.from_dict(pmfs, orient="index", columns=list(counts))
    mat.index.name = "gene"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out)
    return out

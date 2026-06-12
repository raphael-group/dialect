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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

# DIALECT effect suffix -> DIG per-effect fraction column.
_EFFECTS = {"_M": "Pi_MIS", "_N": "Pi_NONS"}
_REQUIRED = {"GENE", "ALPHA", "THETA", *(_EFFECTS.values())}
_KMAX_CAP = 50  # per-sample passenger counts never realistically exceed this


def _nb_params(
    alpha: float,
    theta: float,
    pi: float,
    n_samples: int,
) -> tuple[float, float]:
    """Per-sample NB (n, p) for one effect class (ALPHA split by N, effect-thinned)."""
    return alpha / n_samples, 1.0 / (1.0 + theta * pi)


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
    :param tail_eps: truncate each PMF where the survival drops below this.
    """
    df = pd.read_csv(dig_results, sep="\t")
    missing = _REQUIRED - set(df.columns)
    if missing:
        msg = f"DIG results missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    if n_samples <= 0:
        msg = "n_samples must be a positive integer"
        raise ValueError(msg)

    # Pass 1: collect per-(gene, effect) NB params and find the global support size.
    params: dict[str, tuple[float, float, float]] = {}
    kmax = max_count or 1
    for _, row in df.iterrows():
        alpha, theta = float(row["ALPHA"]), float(row["THETA"])
        if not (np.isfinite(alpha) and np.isfinite(theta) and alpha > 0 and theta > 0):
            continue
        for suffix, pi_col in _EFFECTS.items():
            pi = float(row[pi_col])
            key = f"{row['GENE']}{suffix}"
            params[key] = (alpha, theta, pi)
            if pi > 0:
                nb = nbinom(*_nb_params(alpha, theta, pi, n_samples))
                kmax = max(kmax, min(int(nb.ppf(1 - tail_eps)), _KMAX_CAP))

    # Pass 2: build each PMF over the shared support 0..kmax and renormalize.
    counts = np.arange(kmax + 1)
    pmfs: dict[str, np.ndarray] = {}
    for key, (alpha, theta, pi) in params.items():
        if pi <= 0:
            pmf = np.zeros(kmax + 1)
            pmf[0] = 1.0
        else:
            pmf = nbinom(*_nb_params(alpha, theta, pi, n_samples)).pmf(counts)
            total = pmf.sum()
            if total > 0:
                pmf = pmf / total
        pmfs[key] = pmf

    mat = pd.DataFrame.from_dict(pmfs, orient="index", columns=list(counts))
    mat.index.name = "gene"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out)
    return out

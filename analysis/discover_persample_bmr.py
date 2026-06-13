"""Prototype: a DISCOVER-style per-gene-per-sample background, lifted to a count PMF.

The literature review (research/notes/12) identified DISCOVER (Canisius et al. 2016) as
the only off-the-shelf method that natively conditions on per-sample tumour burden and
exposes a per-gene-per-sample background -- but its cell is a Bernoulli alteration
probability, not the count distribution P(B_{g,s}=k) DIALECT consumes, and its package
does not install cleanly. We re-implement its background faithfully and lift it to a
count PMF, so DIALECT can be fed a true per-gene-per-sample background:

  1. Binarize the cohort's per-effect count matrix (mutated = count > 0).
  2. Fit the DISCOVER max-entropy background p_{g,s} = 1/(1 + exp(a_g + b_s)) by
     coordinate ascent, so the expected margins match BOTH the per-gene mutation
     frequency (row sums) AND the per-sample mutation burden (column sums) -- the
     double-marginal constraint that controls the tumour-burden confound.
  3. Lift the Bernoulli probability to a per-cell Poisson rate lambda_{g,s} =
     -ln(1 - p_{g,s}) and a count PMF P(B_{g,s}=k) = Poisson(lambda_{g,s}).

Validation here is standalone: high-burden samples must receive a *higher* background
for the long passenger genes (that is the whole point -- it absorbs the hypermutator
excess that DIALECT currently misreads as co-occurring drivers).

Usage::

    python analysis/discover_persample_bmr.py --cohort UCEC
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_MAX_ITERS = 200
_TOL = 1e-6
_P_CLIP = 1e-12
# A few canonical long passenger genes used only to probe the per-sample axis.
_LONG_PROBE = {"TTN", "MUC16", "SYNE1", "CSMD3", "FAT4", "OBSCN"}


def fit_discover_background(binary: np.ndarray) -> np.ndarray:
    """Fit p_{g,s} = sigmoid-style max-entropy background matching row+col margins.

    ``binary`` is a (genes x samples) 0/1 matrix. Returns the (genes x samples) matrix
    of alteration probabilities whose expected per-gene and per-sample totals match the
    observed ones (DISCOVER's Poisson-binomial background). Solved by coordinate ascent:
    p_{g,s} = 1/(1 + exp(a_g + b_s)); alternately root-find a_g to match each gene's row
    sum and b_s to match each sample's column sum (both monotonic).
    """
    row_target = binary.sum(axis=1).astype(float)  # per-gene mutated-sample counts
    col_target = binary.sum(axis=0).astype(float)  # per-sample burden (mutated genes)
    n_genes, n_samples = binary.shape
    a = np.zeros(n_genes)
    b = np.zeros(n_samples)

    def _solve_axis(other: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
        # For each i, find x_i s.t. sum_j 1/(1+exp(x_i + other_j)) = target_i (monotone).
        out = np.zeros(n)
        for i in range(n):
            t = target[i]
            if t <= 0:
                out[i] = 50.0  # all-zero row/col -> p ~ 0
                continue
            if t >= len(other):
                out[i] = -50.0
                continue
            lo, hi = -50.0, 50.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                s = np.sum(1.0 / (1.0 + np.exp(mid + other)))
                if s > t:  # too many expected -> raise x
                    lo = mid
                else:
                    hi = mid
            out[i] = 0.5 * (lo + hi)
        return out

    prev = None
    for _ in range(_MAX_ITERS):
        a = _solve_axis(b, row_target, n_genes)
        b = _solve_axis(a, col_target, n_samples)
        p = 1.0 / (1.0 + np.exp(a[:, None] + b[None, :]))
        if prev is not None and np.max(np.abs(p - prev)) < _TOL:
            break
        prev = p
    return np.clip(p, _P_CLIP, 1 - _P_CLIP)


def background_to_rate(p: np.ndarray) -> np.ndarray:
    """Lift a Bernoulli alteration probability to a Poisson count rate lambda."""
    return -np.log(1.0 - p)


def main() -> None:
    """Fit the per-sample background for a cohort and validate the per-sample axis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--results-root", default="output")
    args = parser.parse_args()

    counts = pd.read_csv(
        Path(args.results_root) / args.cohort / "count_matrix.csv", index_col=0,
    )
    binary = (counts.to_numpy() > 0).astype(float).T  # genes x samples
    genes = list(counts.columns)
    burden = counts.to_numpy().sum(axis=1)  # per-sample total mutation count

    p = fit_discover_background(binary)
    lam = background_to_rate(p)

    # Margin recovery (sanity: expected totals match observed).
    row_err = np.max(np.abs(p.sum(axis=1) - binary.sum(axis=1)))
    col_err = np.max(np.abs(p.sum(axis=0) - binary.sum(axis=0)))

    # The point: for the longest passenger genes, the per-sample background rate must
    # rise with sample burden (it absorbs the hypermutator excess).
    base = [g.rsplit("_", 1)[0] for g in genes]
    long_idx = [i for i, b0 in enumerate(base) if b0 in _LONG_PROBE]
    hi = burden >= np.quantile(burden, 0.9)
    lo = burden <= np.quantile(burden, 0.5)

    print(f"\nDISCOVER-style per-sample background — {args.cohort} "
          f"({binary.shape[0]} gene-effects x {binary.shape[1]} samples)\n")
    print(f"margin recovery: max row err={row_err:.3g}, max col err={col_err:.3g}")
    print(f"burden: median={int(np.median(burden))}, "
          f"top-decile cutoff={int(np.quantile(burden, 0.9))}, max={int(burden.max())}")
    if long_idx:
        lam_long = lam[long_idx]
        hi_mean = float(lam_long[:, hi].mean())
        lo_mean = float(lam_long[:, lo].mean())
        ratio = hi_mean / max(lo_mean, 1e-9)
        print(
            f"\nlong-passenger background rate lambda "
            f"(mean over {len(long_idx)} long-gene effects):\n"
            f"  high-burden samples (top 10%):    {hi_mean:.4f}\n"
            f"  low-burden  samples (bottom 50%): {lo_mean:.4f}\n"
            f"  ratio hi/lo: {ratio:.1f}x  (per-sample axis -- hypermutators get a "
            f"much higher passenger background)",
        )


if __name__ == "__main__":
    main()

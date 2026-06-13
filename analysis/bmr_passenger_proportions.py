"""Likely-passenger proportion among DIALECT's top ME/CO pairs, per BMR provider.

Faithful to the paper's ranking (``stats/ranking.py``): the epsilon-threshold filter
(both ``Tau_1X``, ``Tau_X1`` > eps), the rho-sign split (ME rho<0 ranked by Rho asc; CO
rho>0 ranked by Likelihood Ratio desc), and NO significance test (the paper ranks). Adds
reviewer R1-5's intra-gene (``GENE_M:GENE_N`` same-gene) exclusion.

"Suspicious" = the paper's per-cohort event-level likely passengers
(``data/event_level_likely_passengers/<COHORT>.txt`` = top-100 most-mutated non-OncoKB
genes). The headline metric is the paper's: the AUC (over top-1..K ranked pairs) of the
fraction of genes that are likely passengers -- lower is better.

Usage::

    python analysis/bmr_passenger_proportions.py                # every cohort in output/
    python analysis/bmr_passenger_proportions.py --cohort BRCA   # one cohort
    python analysis/bmr_passenger_proportions.py --csv out.csv   # also dump tidy rows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dialect.stats.thresholds import compute_epsilon_threshold

_BMRS = ("cbase", "dig", "mutsig")
_K = 100  # AUC horizon (top-1..K ranked pairs), matching the paper


def _base(gene_effect: str) -> str:
    """Strip the _M/_N effect suffix to the gene symbol (for intra-gene detection)."""
    return gene_effect.rsplit("_", 1)[0]


def rank_and_filter(
    df: pd.DataFrame,
    eps: float,
    direction: str,
    *,
    exclude_intra: bool = True,
) -> pd.DataFrame:
    """Apply the paper's eps-filter + rho-sign split + ranking (+ intra-gene drop)."""
    out = df.copy()
    n_intra = 0
    if exclude_intra:
        intra = out["Gene A"].map(_base) == out["Gene B"].map(_base)
        n_intra = int(intra.sum())
        out = out[~intra]
    out = out[(out["Tau_1X"] > eps) & (out["Tau_X1"] > eps)]
    if direction == "ME":
        out = out[out["Rho"] < 0].sort_values("Rho", ascending=True)
    else:
        out = out[out["Rho"] > 0].sort_values("Likelihood Ratio", ascending=False)
    out.attrs["n_intra_dropped"] = n_intra
    return out


def passenger_proportion(ranked: pd.DataFrame, passengers: set, k: int) -> float:
    """Fraction of gene slots in the top-k ranked pairs that are likely passengers."""
    head = ranked.head(k)
    genes = head["Gene A"].tolist() + head["Gene B"].tolist()
    return 0.0 if not genes else sum(g in passengers for g in genes) / len(genes)


def proportion_auc(ranked: pd.DataFrame, passengers: set, horizon: int = _K) -> float:
    """Paper's metric: AUC of the passenger-proportion curve over top-1..horizon."""
    props = [passenger_proportion(ranked, passengers, k) for k in range(1, horizon + 1)]
    return float(np.trapezoid(props, dx=1) / horizon)


def load_passengers(cohort: str, root: Path) -> set:
    """Per-cohort event-level likely-passenger set (top non-OncoKB genes)."""
    fn = root.parent / "data" / "event_level_likely_passengers" / f"{cohort}.txt"
    if not fn.exists():
        return set()
    return {line.strip() for line in fn.read_text().splitlines() if line.strip()}


def evaluate_cohort(cohort: str, root: Path) -> list[dict]:
    """One row per (cohort, BMR) with ME/CO pair counts + passenger proportions."""
    cohort_dir = root / cohort
    cnt_fn = cohort_dir / "count_matrix.csv"
    if not cnt_fn.exists():
        return []
    n_samples = pd.read_csv(cnt_fn, index_col=0).shape[0]
    eps = compute_epsilon_threshold(n_samples)
    passengers = load_passengers(cohort, root)

    rows = []
    for bmr in _BMRS:
        pw_fn = cohort_dir / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if not pw_fn.exists():
            continue
        df = pd.read_csv(pw_fn)
        me = rank_and_filter(df, eps, "ME")
        co = rank_and_filter(df, eps, "CO")
        rows.append({
            "cohort": cohort,
            "bmr": bmr,
            "n_samples": n_samples,
            "eps": round(eps, 5),
            "intra_dropped": me.attrs["n_intra_dropped"],
            "n_ME": len(me),
            "ME_pass_top10": round(passenger_proportion(me, passengers, 10), 3),
            "ME_pass_auc": round(proportion_auc(me, passengers), 3),
            "n_CO": len(co),
            "CO_pass_top10": round(passenger_proportion(co, passengers, 10), 3),
            "CO_pass_auc": round(proportion_auc(co, passengers), 3),
        })
    return rows


def main() -> None:
    """Print (and optionally dump) the per-cohort/BMR passenger-proportion table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=None, help="One cohort; default = all.")
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--csv", default=None, help="Optional tidy-CSV output path.")
    args = parser.parse_args()
    root = Path(args.results_root)

    cohorts = (
        [args.cohort]
        if args.cohort
        else sorted(d.name for d in root.iterdir() if (d / "count_matrix.csv").exists())
    )
    rows = [r for c in cohorts for r in evaluate_cohort(c, root)]
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nLikely-passenger proportion among top ME/CO pairs "
          "(paper ranking + eps-filter + intra-gene exclusion)\n")
    print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

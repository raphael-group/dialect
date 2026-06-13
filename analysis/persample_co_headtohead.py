"""Head-to-head: does a per-sample COUNT background collapse the spurious CO?

The fix needs a per-gene-per-sample background *count* distribution P(B_{g,s}=k).
No tool exposes one (DISCOVER gives a per-sample presence probability; MutSigCV2
computes a per-(gene,patient) rate but does not expose it -- research/notes/12). The
self-contained route is the RANK-1 / MADGiC factorization (= what MutSigCV2 does
internally): take CBaSE's per-gene passenger rate E[B_g] and scale it by each
sample's relative mutation burden, then form a per-sample Poisson count PMF:

    lambda_{g,s} = E[B_g] * (burden_s / mean_burden)
    P(B_{g,s}=k) = Poisson(lambda_{g,s})

This sums to each sample's burden by construction (sum_g lambda_{g,s} = burden_s), so
a hypermutator's long-gene excess lands near its per-sample expected count and is
absorbed into B instead of being misread as a co-occurring driver.

We then run DIALECT's EM (now sample-indexed) with this per-sample background and
compare its FDR-significant CO calls + suspicious-gene fraction against the stock
cohort-level CBaSE run. Usage::

    python analysis/persample_co_headtohead.py --cohort LUAD -k 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analysis.bmr_fdr_comparison import call_significant, summarize
from dialect.data.io import load_bmr_pmfs
from dialect.models.assembly import initialize_interaction_objects
from dialect.models.gene import Gene
from dialect.utils.identify import (
    create_pairwise_results,
    estimate_pi_for_each_gene,
    estimate_taus_for_each_interaction,
)

_FDR = 0.05


def per_gene_rate(pmf: list) -> float:
    """E[B_g] = sum_k k * P(B_g=k) from a per-gene background PMF (list over k)."""
    return float(sum(k * p for k, p in enumerate(pmf)))


def build_persample_genes(
    counts: pd.DataFrame,
    bmr_pmfs: dict,
    top_k: int,
) -> dict:
    """Build top-k Gene objects with burden-scaled per-sample Poisson backgrounds."""
    burden = counts.to_numpy().sum(axis=1).astype(float)
    mean_burden = burden.mean()
    r_s = burden / mean_burden  # per-sample relative burden multiplier
    kmax = int(counts.to_numpy().max())
    ks = np.arange(kmax + 1)

    totals = counts.sum(axis=0).sort_values(ascending=False)
    top_genes = [g for g in totals.index if g in bmr_pmfs][:top_k]

    genes = {}
    for gene_name in top_genes:
        lam_g = per_gene_rate(bmr_pmfs[gene_name])
        lam_gs = np.clip(lam_g * r_s, 1e-9, None)  # per-sample rate
        # Per-sample Poisson count PMF over 0..kmax (renormalized on truncation).
        pmf_matrix = poisson.pmf(ks[None, :], lam_gs[:, None])
        pmf_matrix /= pmf_matrix.sum(axis=1, keepdims=True)
        per_sample_pmfs = [dict(enumerate(row)) for row in pmf_matrix]
        genes[gene_name] = Gene(
            name=gene_name,
            samples=counts.index,
            counts=counts[gene_name].to_numpy(),
            bmr_pmf=per_sample_pmfs,
        )
    return genes


def run_persample_identify(cohort: str, root: Path, top_k: int) -> pd.DataFrame:
    """Run DIALECT's EM with the per-sample background; return the pairwise frame."""
    counts = pd.read_csv(root / cohort / "count_matrix.csv", index_col=0)
    bmr_pmfs = load_bmr_pmfs(str(root / cohort / "bmr_pmfs.csv"))

    genes = build_persample_genes(counts, bmr_pmfs, top_k)
    out = root / cohort / "id_persample"
    out.mkdir(parents=True, exist_ok=True)
    single_fout = str(out / "single_gene_results.csv")
    pairwise_fout = str(out / "pairwise_interaction_results.csv")

    estimate_pi_for_each_gene(genes.values(), single_fout)
    _, interactions = initialize_interaction_objects(top_k, genes.values())
    estimate_taus_for_each_interaction(interactions)
    create_pairwise_results(interactions, pairwise_fout)
    return pd.read_csv(pairwise_fout)


def main() -> None:
    """Run the per-sample-vs-CBaSE CO head-to-head for a cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=100)
    parser.add_argument("--results-root", default="output")
    args = parser.parse_args()

    root = Path(args.results_root)
    persample = run_persample_identify(args.cohort, root, args.top_k)
    rows = [summarize(args.cohort, "persample", call_significant(persample, _FDR))]
    for bmr in ("cbase", "dig"):
        fn = root / args.cohort / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if fn.exists():
            called = call_significant(pd.read_csv(fn), _FDR)
            rows.append(summarize(args.cohort, bmr, called))

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\nCO head-to-head -- {args.cohort} (BH q<{_FDR})\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

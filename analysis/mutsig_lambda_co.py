"""DIALECT with the PROPER per-(gene,sample) MutSig BMR (patched-source lambda dump).

Reads ``persample_lambda.f32`` -- the per-(gene,patient,effect) expected background
that MutSig2CV computes *internally* (per-patient, per-category hypergeometric means),
dumped by our Octave-patched source (scripts/run_mutsig_octave.sh) -- and builds a
per-sample Poisson count PMF  P(B_{g,s}=k) = Poisson(lambda_{g,s,eff})  for DIALECT.

This replaces the scalar-f_p reconstruction in ``mutsig_persample_co.py``: lambda here
varies by sample AND sequence context (a POLE/MSI sample's excess flows through its
own per-category counts), which the single per-patient scalar could not represent.

Usage::

    python analysis/mutsig_lambda_co.py --cohort UCEC \
        --results-root output --mutsig-root output/mutsigsrc -k 100
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
_EFF = {"M": 0, "N": 1}  # page index into the lambda dump


def load_lambda(mutsig_dir: Path) -> tuple:
    """Read the raw per-(gene,patient,effect) lambda dump + its gene/patient labels."""
    meta = dict(
        line.split("\t")
        for line in (mutsig_dir / "persample_meta.txt").read_text().splitlines()
        if line.strip()
    )
    ng, npat, neff = int(meta["ng"]), int(meta["np"]), int(meta["neff"])
    lam = np.fromfile(mutsig_dir / "persample_lambda.f32", dtype="<f4")
    lam = lam.reshape((ng, npat, neff), order="F")  # column-major from Octave fwrite
    genes = (mutsig_dir / "persample_genes.txt").read_text().split()
    patients = (mutsig_dir / "persample_patients.txt").read_text().split()
    return lam, genes, patients


def build_lambda_pmfs(
    gene_effects: list,
    samples: pd.Index,
    mutsig_dir: Path,
    cbase_pmfs: dict,
    kmax: int,
) -> dict:
    """Per-(gene_effect, sample) Poisson PMFs from lambda; CBaSE-fallback if absent."""
    lam, genes, patients = load_lambda(mutsig_dir)
    gidx = {g: i for i, g in enumerate(genes)}
    pidx = {p: i for i, p in enumerate(patients)}
    col = [pidx.get(str(s), -1) for s in samples]  # mutsig patient column per sample
    ks = np.arange(kmax + 1)

    out = {}
    for ge in gene_effects:
        base, eff = ge.rsplit("_", 1)
        gi = gidx.get(base)
        if gi is None or eff not in _EFF:
            if ge in cbase_pmfs:  # gene/effect not modelled by MutSig -> cohort CBaSE
                out[ge] = dict(enumerate(cbase_pmfs[ge]))
            continue
        lam_g = lam[gi, :, _EFF[eff]]
        gene_mean = float(lam_g.mean()) if lam_g.size else 0.0
        lam_s = np.array([lam_g[c] if c >= 0 else gene_mean for c in col], dtype=float)
        lam_s = np.clip(lam_s, 1e-12, None)
        pmf = poisson.pmf(ks[None, :], lam_s[:, None])
        pmf /= pmf.sum(axis=1, keepdims=True)
        out[ge] = [dict(enumerate(row)) for row in pmf]
    return out


def _build_genes(counts: pd.DataFrame, pmfs: dict, top_k: int) -> dict:
    totals = counts.sum(axis=0).sort_values(ascending=False)
    top = [g for g in totals.index if g in pmfs][:top_k]
    return {
        ge: Gene(
            name=ge,
            samples=counts.index,
            counts=counts[ge].to_numpy(),
            bmr_pmf=pmfs[ge],
        )
        for ge in top
    }


def run(cohort: str, root: Path, mutsig_root: Path, top_k: int, suffix: str) -> Path:
    """Run DIALECT with the proper MutSig per-sample lambda BMR; write pairwise CSV."""
    cohort_dir = root / cohort
    counts = pd.read_csv(cohort_dir / "count_matrix.csv", index_col=0)
    kmax = int(counts.to_numpy().max())
    cbase = load_bmr_pmfs(str(cohort_dir / "bmr_pmfs.csv"))
    pmfs = build_lambda_pmfs(
        list(counts.columns), counts.index, mutsig_root / cohort, cbase, kmax,
    )
    genes = _build_genes(counts, pmfs, top_k)

    out = cohort_dir / f"id_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    estimate_pi_for_each_gene(genes.values())
    _, interactions = initialize_interaction_objects(top_k, genes.values())
    estimate_taus_for_each_interaction(interactions)
    create_pairwise_results(interactions, str(out / "pairwise_interaction_results.csv"))
    return out / "pairwise_interaction_results.csv"


def main() -> None:
    """Run DIALECT with the proper MutSig lambda BMR and compare to other BMRs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--mutsig-root", default="output/mutsigsrc")
    parser.add_argument("-k", "--top-k", type=int, default=100)
    parser.add_argument("--suffix", default="mutsiglam")
    args = parser.parse_args()
    root = Path(args.results_root)

    pw = run(args.cohort, root, Path(args.mutsig_root), args.top_k, args.suffix)
    called = call_significant(pd.read_csv(pw), _FDR)
    rows = [summarize(args.cohort, args.suffix, called)]
    for bmr in ("mutsig", "cbase", "dig"):
        fn = root / args.cohort / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if fn.exists():
            other = call_significant(pd.read_csv(fn), _FDR)
            rows.append(summarize(args.cohort, bmr, other))
    pd.set_option("display.width", 200)
    print(f"\nProper MutSig lambda BMR vs others -- {args.cohort} (BH q<{_FDR})\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

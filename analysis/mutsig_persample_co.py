"""Compute MutSigCV per-(gene,sample) BMR PMFs and run DIALECT with them.

Implements Route D (research/notes/12 + the MutSig extraction workflow): reconstruct
MutSigCV's per-(gene,patient) background from an existing compiled MutSig2CV run
(``output/<cohort>_mutsig/results.mat`` + ``patient_counts_and_rates.txt``) -- no
MATLAB -- and feed it to DIALECT's now sample-indexed EM.

MutSigCV factorizes the rate as mu_{g,p,eff} = (x_g/X_g) * f_c(eff) * f_p(p), where
x_g/X_g = ``G.mutrate`` is the bagel covariate-pooled per-gene base rate and
f_p = mu_p/mu_tot is the PER-PATIENT burden factor (the per-sample axis a cohort
background cannot express). The per-(gene,patient,effect) background expected count is
lambda_{g,p,eff} = (N_eff[g] / n_patients) * mutrate[g] * f_p(p), and we emit a
per-sample Poisson PMF P(B_{g,p,eff}=k) = Poisson(lambda) (the beta-binomial
overdispersion of MutSig's hyge2pdf is a documented refinement; Poisson(N*mu) is the
workflow-sanctioned first cut and isolates the per-sample axis we are testing).

Effect channels: missense -> DIALECT ``_M`` (Nmis); nonsense+splice -> ``_N``
(Nnon+Nspl). Genes absent from / NaN in results.mat fall back to the cohort CBaSE PMF
(shared across samples) so every tested gene still has a background.

Usage::

    python analysis/mutsig_persample_co.py --cohort CHOL -k 100      # run DIALECT
    python analysis/mutsig_persample_co.py --cohort CHOL --validate  # axis check only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
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


def _load_mutsig_bagel(cohort: str, root: Path) -> dict:
    """Per-gene bagel rate + per-effect coverage from results.mat (x_g/X_g, Nmis...)."""
    g = sio.loadmat(
        root / f"{cohort}_mutsig" / "results.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["G"]
    gene = np.atleast_1d(g.gene)
    return {
        str(gene[i]): {
            "mutrate": float(np.atleast_1d(g.mutrate)[i]),
            "Nmis": float(np.atleast_1d(g.Nmis)[i]),
            "Nnon": float(np.atleast_1d(g.Nnon)[i]),
            "Nspl": float(np.atleast_1d(g.Nspl)[i]),
        }
        for i in range(len(gene))
    }


def _patient_factor(cohort: str, root: Path) -> dict:
    """f_p(p) = mu_p / mu_tot per patient name (the per-sample burden factor)."""
    pr = pd.read_csv(
        root / f"{cohort}_mutsig" / "patient_counts_and_rates.txt", sep="\t",
    )
    mu_tot = pr["n_tot"].sum() / pr["N_tot"].sum()
    return {
        name: (n / cov) / mu_tot
        for name, n, cov in zip(pr["name"], pr["n_tot"], pr["N_tot"], strict=False)
    }


def mutsig_persample_pmfs(
    cohort: str,
    root: Path,
    gene_effects: list,
    samples: pd.Index,
    kmax: int,
) -> dict:
    """Per-(gene_effect, sample) MutSigCV Poisson background PMFs, CBaSE-fallback."""
    bagel = _load_mutsig_bagel(cohort, root)
    fp_by_name = _patient_factor(cohort, root)
    f_p = np.array([fp_by_name.get(s, 1.0) for s in samples])
    n_pat = len(samples)
    cbase = load_bmr_pmfs(str(root / cohort / "bmr_pmfs.csv"))
    ks = np.arange(kmax + 1)

    out = {}
    for ge in gene_effects:
        base, eff = ge.rsplit("_", 1)
        info = bagel.get(base)
        if info is None or not np.isfinite(info["mutrate"]) or info["mutrate"] <= 0:
            if ge in cbase:  # fall back to the cohort CBaSE per-gene PMF (shared)
                out[ge] = dict(enumerate(cbase[ge]))
            continue
        n_eff = info["Nmis"] if eff == "M" else info["Nnon"] + info["Nspl"]
        base_lambda = (n_eff / n_pat) * info["mutrate"]  # per-patient expected bg count
        lam = np.clip(base_lambda * f_p, 1e-12, None)
        pmf_matrix = poisson.pmf(ks[None, :], lam[:, None])
        pmf_matrix /= pmf_matrix.sum(axis=1, keepdims=True)
        out[ge] = [dict(enumerate(row)) for row in pmf_matrix]
    return out


def _build_genes(counts: pd.DataFrame, pmfs: dict, top_k: int) -> dict:
    totals = counts.sum(axis=0).sort_values(ascending=False)
    top = [g for g in totals.index if g in pmfs][:top_k]
    genes = {}
    for ge in top:
        genes[ge] = Gene(
            name=ge,
            samples=counts.index,
            counts=counts[ge].to_numpy(),
            bmr_pmf=pmfs[ge],  # per-sample list OR a shared dict (CBaSE fallback)
        )
    return genes


def validate(cohort: str, root: Path) -> None:
    """Check the per-sample axis: long-gene background rises with patient burden."""
    counts = pd.read_csv(root / cohort / "count_matrix.csv", index_col=0)
    ges = list(counts.columns)
    kmax = int(counts.to_numpy().max())
    pmfs = mutsig_persample_pmfs(cohort, root, ges, counts.index, kmax)
    fp = _patient_factor(cohort, root)
    f_p = np.array([fp.get(s, 1.0) for s in counts.index])
    hi, lo = f_p >= np.quantile(f_p, 0.9), f_p <= np.quantile(f_p, 0.5)

    def mean_e_b(ge: str, mask: np.ndarray) -> float:
        per_sample = pmfs.get(ge)
        if not isinstance(per_sample, list):
            return float("nan")
        e = [sum(k * p for k, p in per_sample[i].items()) for i in np.where(mask)[0]]
        return float(np.mean(e)) if e else float("nan")

    probes = [g for g in ("TTN_M", "MUC16_M", "CSMD3_M", "OBSCN_M") if g in pmfs]
    print(f"\nMutSigCV per-sample BMR axis check -- {cohort}\n")
    for ge in probes:
        print(f"  {ge}: E[B] hi-burden={mean_e_b(ge, hi):.4f}  "
              f"lo-burden={mean_e_b(ge, lo):.4f}")


def run(cohort: str, root: Path, top_k: int) -> None:
    """Run DIALECT with the MutSigCV per-sample BMR and compare CO to CBaSE/DIG."""
    counts = pd.read_csv(root / cohort / "count_matrix.csv", index_col=0)
    kmax = int(counts.to_numpy().max())
    pmfs = mutsig_persample_pmfs(cohort, root, list(counts.columns), counts.index, kmax)
    genes = _build_genes(counts, pmfs, top_k)

    out = root / cohort / "id_mutsig"
    out.mkdir(parents=True, exist_ok=True)
    estimate_pi_for_each_gene(genes.values(), str(out / "single_gene_results.csv"))
    _, interactions = initialize_interaction_objects(top_k, genes.values())
    estimate_taus_for_each_interaction(interactions)
    create_pairwise_results(interactions, str(out / "pairwise_interaction_results.csv"))

    rows = [summarize(
        cohort, "mutsig",
        call_significant(pd.read_csv(out / "pairwise_interaction_results.csv"), _FDR),
    )]
    for bmr in ("cbase", "dig"):
        fn = root / cohort / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if fn.exists():
            rows.append(summarize(cohort, bmr, call_significant(pd.read_csv(fn), _FDR)))
    pd.set_option("display.width", 200)
    print(f"\nCO head-to-head (MutSigCV per-sample BMR) -- {cohort} (BH q<{_FDR})\n")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    """CLI entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=100)
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root = Path(args.results_root)
    if args.validate:
        validate(args.cohort, root)
    else:
        run(args.cohort, root, args.top_k)


if __name__ == "__main__":
    main()

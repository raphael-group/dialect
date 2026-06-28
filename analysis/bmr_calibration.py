"""Show WHY a per-gene BMR (CBaSE/Dig) is miscalibrated under hypermutation.

For each tumor sample we compare its OBSERVED total mutation count (over the analyzed
gene-effects) to the EXPECTED background total under each BMR model:

  - CBaSE / Dig give ONE background distribution per gene, shared across all samples, so
    the expected background total is ~CONSTANT across samples -- it cannot track the
    orders-of-magnitude per-sample burden variation in a hypermutated cohort (UCEC
    MSI/POLE vs CN-low). The excess observed counts in hypermutators are therefore
    mis-read as DRIVER signal -> inflated co-occurrence.
  - The sample-specific MutSigCV2 BMR gives a per-(gene,sample) Poisson mean (lambda),
    so the expected background scales with each sample's burden and tracks observed.

Headline readout: Pearson r(observed, expected) across samples -- ~0 for the per-gene
models (flat), high for the sample-specific model. Plus per-gene observed-vs-expected
summed over all samples.

Usage:  python -m analysis.bmr_calibration --cohort UCEC
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.bmr_overcorrection_check import load_lambda

PANCAN = Path("output/pancan")
MUTSIG = Path("output/mutsigsrc")


def cbase_expected(pmf_fn: Path) -> pd.Series:
    """E[B] per gene-effect = sum_c c * P(count=c) from a CBaSE/Dig bmr_pmfs.csv."""
    df = pd.read_csv(pmf_fn, index_col=0)
    counts = df.columns.astype(int).to_numpy(dtype=float)
    p = df.to_numpy(dtype=float)
    p[~np.isfinite(p)] = 0.0  # guard against malformed tail entries
    rowsum = p.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    p = p / rowsum  # PMFs should sum to 1; renormalize defensively
    return pd.Series(p @ counts, index=df.index)


def mutsig_expected_per_sample(cohort: str, gene_effects: list[str],
                               samples: list[str]) -> pd.Series | None:
    """Sum of per-(gene,sample) lambda over the analyzed gene-effects, per sample."""
    root = MUTSIG / cohort
    if not (root / "persample_lambda.f32").exists():
        return None
    lam, gidx = load_lambda(cohort)
    pat_txt = (root / "persample_patients.txt").read_text().splitlines()
    pats = [p.strip() for p in pat_txt]
    pidx = {p: i for i, p in enumerate(pats)}
    # align samples (count_matrix index) to lambda patient columns
    keep = [s for s in samples if s in pidx]
    cols = [pidx[s] for s in keep]
    total = np.zeros(len(keep))
    for ge in gene_effects:
        base, suf = ge.rsplit("_", 1)
        if base not in gidx:
            continue
        e = 0 if suf == "M" else 1
        total += lam[gidx[base], cols, e]
    return pd.Series(total, index=keep)


def main() -> None:
    """Compute + plot observed-vs-expected calibration for one cohort."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="UCEC")
    p.add_argument("--out", default="figures/bmr_calibration")
    args = p.parse_args()
    c = args.cohort
    cdir = PANCAN / c

    cm = pd.read_csv(cdir / "count_matrix.csv", index_col=0)
    samples = cm.index.tolist()
    gene_effects = cm.columns.tolist()
    observed = cm.sum(axis=1)  # observed total per sample

    expected = {}
    cb = cbase_expected(cdir / "bmr_pmfs.csv").reindex(gene_effects).fillna(0)
    expected["CBaSE"] = pd.Series(float(cb.sum()), index=samples)  # constant per sample
    dig_fn = cdir / "bmr_pmfs.dig.csv"
    if dig_fn.exists():
        dg = cbase_expected(dig_fn).reindex(gene_effects).fillna(0)
        expected["Dig"] = pd.Series(float(dg.sum()), index=samples)
    ms = mutsig_expected_per_sample(c, gene_effects, samples)
    if ms is not None:
        expected["MutSigCV2"] = ms

    print(f"\n=== {c}: observed vs expected background (N={len(samples)} samples) ===")
    print(f"observed total/sample: median={observed.median():.0f}  "
          f"min={observed.min():.0f}  max={observed.max():.0f}  "
          f"(fold range {observed.max()/max(observed.min(),1):.0f}x)\n")
    print(f"{'BMR model':12} {'type':16} {'pearson_r':>10} {'exp/sample':>22}")
    rows = []
    for name, exp in expected.items():
        aligned_obs = observed.reindex(exp.index)
        if float(exp.std(ddof=0)) == 0:
            r = 0.0
            kind = "per-gene (constant)"
            espan = f"{exp.iloc[0]:.0f} (flat)"
        else:
            r = float(np.corrcoef(aligned_obs.to_numpy(), exp.to_numpy())[0, 1])
            kind = "sample-specific"
            espan = f"{exp.min():.0f}-{exp.max():.0f}"
        print(f"{name:12} {kind:16} {r:10.3f} {espan:>22}")
        rows.append({
            "cohort": c, "bmr": name, "kind": kind, "pearson_r": round(r, 3),
            "exp_min": round(float(exp.min()), 1),
            "exp_max": round(float(exp.max()), 1),
            "obs_median": float(observed.median()), "obs_max": float(observed.max()),
        })

    # plot: observed (x) vs expected (y) per sample
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    colors = {"CBaSE": "#7f7f7f", "Dig": "#1f77b4", "MutSigCV2": "#d62728"}
    lim = max(observed.max(), *(float(e.max()) for e in expected.values())) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="#bbb", lw=1, label="obs = exp")
    for name, exp in expected.items():
        ax.scatter(observed.reindex(exp.index), exp, s=14, alpha=0.6,
                   color=colors.get(name, "#333"), label=f"{name}", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("observed mutations per sample", fontsize=12)
    ax.set_ylabel("expected background per sample", fontsize=12)
    ax.set_title(f"{c}: per-gene BMR cannot track per-sample burden", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{c}_calibration.png", dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(out / f"{c}_calibration.csv", index=False)
    print(f"\nwrote {out}/{c}_calibration.png + .csv")


if __name__ == "__main__":
    main()

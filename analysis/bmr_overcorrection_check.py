"""Quantify per-sample-MutSig OVER-correction of real CO in low-TMB cohorts.

Three reproducible checks behind the burden-aware BMR recommendation (rebuttal note 14):

  1. ``laml-lambda``  -- for AML driver genes, the per-sample MutSig background's
     observed/expected ratio. A driver whose obs/exp ~ 1 is being explained away as
     passenger background (over-correction); DNMT3A is the culprit in LAML.
  2. ``lowtmb-scan``  -- for the lowest-TMB cohorts, the top-CO pairs recovered under
     CBaSE / DIG / per-sample MutSig, to show which REAL pairs MutSig erases.
  3. ``tmb-co``       -- per-cohort median TMB vs #CO pairs under each BMR, the
     monotone pattern motivating a burden-aware switch.

Usage::

    python -m analysis.bmr_overcorrection_check laml-lambda
    python -m analysis.bmr_overcorrection_check lowtmb-scan
    python -m analysis.bmr_overcorrection_check tmb-co --csv /tmp/tmb_co.csv
    python -m analysis.bmr_overcorrection_check all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bmr_passenger_proportions import rank_and_filter
from dialect.stats.thresholds import compute_epsilon_threshold

PANCAN = Path("output/pancan")
MUTSIG = Path("output/mutsigsrc")
AML_DRIVERS = ["DNMT3A", "FLT3", "IDH1", "IDH2", "NPM1", "NRAS", "KRAS",
               "PTPN11", "TET2", "RUNX1", "TP53", "CEBPA", "WT1"]
LOWTMB_COHORTS = ["LAML", "LGG", "PRAD", "PAAD", "THCA", "TGCT", "UVM", "PCPG"]


def _base(ge: str) -> str:
    return ge.rsplit("_", 1)[0]


def load_lambda(cohort: str) -> tuple[np.ndarray, dict]:
    """Return (lambda[ng,np,neff], gene->index) for a cohort per-sample MutSig dump."""
    root = MUTSIG / cohort
    meta = dict(line.split("\t") for line in
                (root / "persample_meta.txt").read_text().split("\n") if "\t" in line)
    ng, np_, neff = int(meta["ng"]), int(meta["np"]), int(meta["neff"])
    genes = [g.strip() for g in (root / "persample_genes.txt").read_text().splitlines()]
    lam = np.fromfile(root / "persample_lambda.f32", dtype="<f4").reshape(
        (ng, np_, neff), order="F")
    return lam, {g: i for i, g in enumerate(genes)}


def laml_lambda(cohort: str = "LAML") -> pd.DataFrame:
    """Obs vs expected-background for driver gene-effects under per-sample MutSig."""
    lam, gidx = load_lambda(cohort)
    cm = pd.read_csv(PANCAN / cohort / "count_matrix.csv", index_col=0)
    rows = []
    for g in AML_DRIVERS:
        if g not in gidx:
            continue
        i = gidx[g]
        for e, enm in enumerate(["_M", "_N"]):
            col = f"{g}{enm}"
            if col not in cm.columns:
                continue
            obs_tot = int(cm[col].sum())
            obs_samp = int((cm[col] > 0).sum())
            exp_bg = float(lam[i, :, e].sum())
            pct_bg = round(100 * min(exp_bg / obs_tot, 1), 1) if obs_tot else 0.0
            rows.append({
                "gene_effect": col,
                "obs_samples": obs_samp,
                "obs_total": obs_tot,
                "exp_bg_total": round(exp_bg, 3),
                "obs_over_exp": round(obs_tot / exp_bg, 2) if exp_bg > 0 else np.inf,
                "pct_explained_as_bg": pct_bg,
            })
    return pd.DataFrame(rows).sort_values("obs_over_exp")


def top_co(cohort: str, bmr: str, k: int = 6) -> list[str]:
    """Top-k co-occurring base-gene pairs for a cohort/BMR after eps + rank filter."""
    d = PANCAN / cohort
    fn = d / f"id_{bmr}" / "pairwise_interaction_results.csv"
    if not fn.exists():
        return []
    n = pd.read_csv(d / "count_matrix.csv", index_col=0).shape[0]
    eps = compute_epsilon_threshold(n)
    co = rank_and_filter(pd.read_csv(fn), eps, "CO").head(k)
    return [f"{_base(a)}:{_base(b)}"
            for a, b in zip(co["Gene A"], co["Gene B"], strict=False)]


def lowtmb_scan() -> None:
    """Print top-CO pairs per BMR for each low-TMB cohort."""
    for c in LOWTMB_COHORTS:
        if not (PANCAN / c / "count_matrix.csv").exists():
            continue
        print(f"=== {c} ===")
        for bmr in ("cbase", "dig", "mutsig"):
            print(f"  {bmr:7}: {', '.join(top_co(c, bmr)) or '(none)'}")
        print()


def tmb_co() -> pd.DataFrame:
    """Per-cohort median TMB and #CO pairs under each BMR."""
    rows = []
    for d in sorted(PANCAN.iterdir()):
        cm_fn = d / "count_matrix.csv"
        if not cm_fn.exists():
            continue
        cm = pd.read_csv(cm_fn, index_col=0)
        n = cm.shape[0]
        eps = compute_epsilon_threshold(n)
        med_tmb = round(float(cm.sum(axis=1).median()), 1)
        rec = {"cohort": d.name, "N": n, "medTMB": med_tmb}
        for bmr in ("cbase", "dig", "mutsig"):
            fn = d / f"id_{bmr}" / "pairwise_interaction_results.csv"
            rec[f"CO_{bmr}"] = (len(rank_and_filter(pd.read_csv(fn), eps, "CO"))
                               if fn.exists() else None)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("medTMB")


def main() -> None:
    """Run the requested over-correction check(s) and print the tables."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("check", choices=["laml-lambda", "lowtmb-scan", "tmb-co", "all"])
    p.add_argument("--csv", default=None)
    args = p.parse_args()
    pd.set_option("display.width", 200)

    if args.check in ("laml-lambda", "all"):
        print("\n[1] LAML per-sample MutSig background obs/exp (over-correction)\n")
        print(laml_lambda().to_string(index=False))
        print("\n  -> DNMT3A's background EXPLAINS most of its observed mutations"
              " (obs/exp~1):\n     the per-sample model treats a canonical AML driver"
              " as passenger.")
    if args.check in ("lowtmb-scan", "all"):
        print("\n[2] Top-CO pairs in low-TMB cohorts: CBaSE vs DIG vs MutSig\n")
        lowtmb_scan()
    if args.check in ("tmb-co", "all"):
        print("\n[3] Median TMB vs #CO pairs per BMR (burden-aware switch)\n")
        df = tmb_co()
        print(df.to_string(index=False))
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

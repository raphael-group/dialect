"""Export the DIALECT 69-cohort x 3-BMR ME/CO networks to JSON for the web atlas.

For every cohort (TCGA + MSK) and every BMR (CBaSE/Dig/MutSigCV2), apply the paper's
eps-filter + intra-gene exclusion + rho-sign ranking, take the top-K mutually-exclusive
(rho<0) and top-K co-occurring (rho>0) gene-effect pairs, and emit a single compact
bundle (atlas.json, all three BMRs) that drives the Observable Framework site in
../dialect-atlas.

Usage:  python -m analysis.build_atlas_data [--out ../dialect-atlas/src/data] [--k 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.bmr_passenger_proportions import rank_and_filter
from dialect.stats.thresholds import compute_epsilon_threshold

BMRS = ("cbase", "dig", "mutsig")
BMR_LABEL = {"cbase": "CBaSE", "dig": "Dig", "mutsig": "MutSigCV2"}
SOURCES = [
    ("TCGA", Path("output/pancan")),
    ("MSK-IMPACT", Path("output/msk/IMPACT2026")),
    ("MSK-CHORD", Path("output/msk/CHORD2024")),
]


def _base(ge: str) -> str:
    return ge.rsplit("_", 1)[0]


def load_drivers() -> set[str]:
    """Return the set of OncoKB cancer-gene Hugo symbols."""
    df = pd.read_csv("data/references/OncoKB_Cancer_Gene_List.tsv", sep="\t")
    return set(df["Hugo Symbol"].astype(str))


def direction_payload(df: pd.DataFrame, eps: float, direction: str, k: int) -> dict:
    """Top-k edges (gene-effect rows w/ base labels + stats) for one direction."""
    ranked = rank_and_filter(df, eps, direction)
    n_total = len(ranked)
    edges = []
    for _, r in ranked.head(k).iterrows():
        a, b = _base(r["Gene A"]), _base(r["Gene B"])
        if a == b:
            continue
        edges.append({
            "a": a, "b": b, "ga": r["Gene A"], "gb": r["Gene B"],
            "rho": round(float(r["Rho"]), 4),
            "lrt": round(float(r["Likelihood Ratio"]), 3),
            "tau11": round(float(r["Tau_11"]), 4),
            "ta": round(float(r["Tau_1X"]), 4),
            "tb": round(float(r["Tau_X1"]), 4),
            "n11": int(r["_11_"]), "n10": int(r["_10_"]),
            "n01": int(r["_01_"]), "n00": int(r["_00_"]),
        })
    return {"n_total": n_total, "edges": edges}


def cohort_payload(study: str, cohort: str, cdir: Path, k: int,
                   drivers: set[str]) -> dict | None:
    """Build the full atlas record (all 3 BMRs, ME+CO edges, drivers) for one cohort."""
    cm_fn = cdir / "count_matrix.csv"
    if not cm_fn.exists():
        return None
    cm = pd.read_csv(cm_fn, index_col=0)
    n = cm.shape[0]
    eps = compute_epsilon_threshold(n)
    rec = {
        "id": f"{study}__{cohort}", "study": study, "cohort": cohort,
        "n_samples": n, "median_tmb": round(float(cm.sum(axis=1).median()), 1),
        "eps": round(eps, 5), "bmrs": {},
    }
    for bmr in BMRS:
        fn = cdir / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if not fn.exists():
            continue
        df = pd.read_csv(fn)
        rec["bmrs"][bmr] = {
            "ME": direction_payload(df, eps, "ME", k),
            "CO": direction_payload(df, eps, "CO", k),
        }
    genes = {e[g] for b in rec["bmrs"].values()
             for d in b.values() for e in d["edges"] for g in ("a", "b")}
    rec["drivers"] = sorted(g for g in genes if g in drivers)
    return rec


def main() -> None:
    """Export atlas.json (all cohorts x BMRs x ME/CO) for the web atlas."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="../dialect-atlas/src/data")
    p.add_argument("--k", type=int, default=50)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drivers = load_drivers()

    cohorts = []
    for study, root in SOURCES:
        if not root.exists():
            continue
        for cdir in sorted(root.iterdir()):
            if not cdir.is_dir():
                continue
            rec = cohort_payload(study, cdir.name, cdir, args.k, drivers)
            if rec and rec["bmrs"]:
                cohorts.append(rec)

    cohorts.sort(key=lambda r: (r["study"], r["cohort"]))
    (out / "atlas.json").write_text(json.dumps({
        "bmrs": list(BMRS), "bmr_label": BMR_LABEL, "cohorts": cohorts,
    }, separators=(",", ":")))
    tot = sum(c["n_samples"] for c in cohorts)
    size_kb = (out / "atlas.json").stat().st_size // 1024
    print(f"wrote atlas.json ({size_kb} KB) with {len(cohorts)} cohorts to {out}")
    print(f"  studies: {sorted({c['study'] for c in cohorts})}; total samples: {tot}")


if __name__ == "__main__":
    main()

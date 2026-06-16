"""Consolidate DIALECT ME/CO networks across all cohorts into one structured JSON.

For each cohort (TCGA pancan + MSK-IMPACT2026 + MSK-CHORD2024) and each BMR, apply
the paper's eps-filter + intra-gene exclusion + rho-sign ranking, take the top-K ME
(rho<0) and top-K CO (rho>0) pairs, collapse _M/_N to base genes, and record which
BMRs support each (pair, direction) within the cohort. Emits one record per cohort
with its consensus ME and CO edge lists, annotated by supporting-BMR set and rank.

Output: /tmp/consensus_networks.json  (list[cohort-record]) for downstream lit search.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.bmr_passenger_proportions import rank_and_filter
from dialect.stats.thresholds import compute_epsilon_threshold

_BMRS = ("cbase", "dig", "mutsig")
TOP_K = 10

# (label, results-root, is the cohort dir directly under root?)
SOURCES = [
    ("TCGA", Path("output/pancan")),
    ("MSK-IMPACT2026", Path("output/msk/IMPACT2026")),
    ("MSK-CHORD2024", Path("output/msk/CHORD2024")),
]


def _base(ge: str) -> str:
    return ge.rsplit("_", 1)[0]


def _pairkey(a: str, b: str) -> tuple[str, str]:
    x, y = _base(a), _base(b)
    return (x, y) if x <= y else (y, x)


def cohort_record(study: str, cohort: str, cdir: Path) -> dict | None:
    """Build the consensus ME/CO record for one cohort (None if it has no data)."""
    cm_fn = cdir / "count_matrix.csv"
    if not cm_fn.exists():
        return None
    cm = pd.read_csv(cm_fn, index_col=0)
    n = cm.shape[0]
    med_tmb = float(cm.sum(axis=1).median())
    eps = compute_epsilon_threshold(n)

    # pair -> {direction: {bmr -> best(rank, rho, lr)}}
    me_pairs: dict = {}
    co_pairs: dict = {}
    for bmr in _BMRS:
        fn = cdir / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if not fn.exists():
            continue
        df = pd.read_csv(fn)
        for direction, store in (("ME", me_pairs), ("CO", co_pairs)):
            ranked = (rank_and_filter(df, eps, direction)
                      .head(TOP_K).reset_index(drop=True))
            for rank, r in ranked.iterrows():
                key = _pairkey(r["Gene A"], r["Gene B"])
                if key[0] == key[1]:  # intra-gene already dropped, belt-and-braces
                    continue
                rec = store.setdefault(key, {"bmrs": {}})
                rec["bmrs"][bmr] = {
                    "rank": int(rank) + 1,
                    "rho": round(float(r["Rho"]), 4),
                    "lr": round(float(r["Likelihood Ratio"]), 4),
                }

    def fmt(store: dict) -> list[dict]:
        out = []
        for (a, b), rec in store.items():
            bmrs = rec["bmrs"]
            best_rank = min(v["rank"] for v in bmrs.values())
            out.append({
                "pair": f"{a}:{b}",
                "gene_a": a,
                "gene_b": b,
                "supported_by": sorted(bmrs.keys()),
                "n_bmr": len(bmrs),
                "best_rank": best_rank,
                "per_bmr": bmrs,
            })
        # most-supported, then best-ranked first
        out.sort(key=lambda d: (-d["n_bmr"], d["best_rank"]))
        return out

    return {
        "study": study,
        "cohort": cohort,
        "n_samples": n,
        "median_tmb": round(med_tmb, 1),
        "epsilon": round(eps, 5),
        "ME": fmt(me_pairs),
        "CO": fmt(co_pairs),
    }


def main() -> None:
    """Write the consolidated consensus-network JSON across all sources."""
    records = []
    for study, root in SOURCES:
        if not root.exists():
            continue
        for cdir in sorted(root.iterdir()):
            if not cdir.is_dir():
                continue
            rec = cohort_record(study, cdir.name, cdir)
            if rec:
                records.append(rec)

    out = Path("/tmp/consensus_networks.json")
    out.write_text(json.dumps(records, indent=2))

    # summary
    n_pairs = sum(len(r["ME"]) + len(r["CO"]) for r in records)
    uniq = set()
    for r in records:
        for d in ("ME", "CO"):
            for p in r[d]:
                uniq.add((p["pair"], d))
    print(f"cohorts: {len(records)}")
    print(f"total (cohort,pair,dir) edges in top-{TOP_K}: {n_pairs}")
    print(f"unique (pair,direction) across all cohorts: {len(uniq)}")
    print(f"wrote {out}")
    print(f"\n{'study':16} {'cohort':28} {'N':>6} {'medTMB':>7} {'#ME':>5} {'#CO':>5}")
    for r in records:
        print(f"{r['study']:16} {r['cohort']:28} {r['n_samples']:6d} "
              f"{r['median_tmb']:7.1f} {len(r['ME']):5d} {len(r['CO']):5d}")


if __name__ == "__main__":
    main()

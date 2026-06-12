"""Benchmark DIALECT's driver calls across background-mutation-rate (BMR) providers.

For each cohort, run ``dialect identify`` twice -- once with the CBaSE background and
once with the DIG background -- holding the (BMR-agnostic) count matrix fixed, so only
``P(B)`` changes. Report, per cohort and BMR, the driver rate of TTN (the canonical
long-gene false positive) and how many known long/recurrently-mutated passenger genes
appear in the top-20 genes by estimated driver rate ``pi``.

This is the BMR-sensitivity analysis: a miscalibrated background lets long passengers
masquerade as drivers; a better background suppresses them while leaving real driver
biology unchanged.

Prerequisites per cohort ``T``: ``output/T/{count_matrix.csv,bmr_pmfs.csv}`` from
``dialect generate -m data/mafs/T.maf -o output/T`` (CBaSE), plus a DIG geneDriver
results file (the pretrained Pancan map works for all cohorts).

Usage::

    python analysis/bmr_sensitivity.py \
        --cohorts CHOL LAML PRAD LUAD BRCA UCEC \
        --dig-results external/DIGDriver/run/Pancan.genes.results.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from dialect.bmr import dig_results_to_bmr_pmfs

# Known long / recurrently-mutated passenger genes (the classic BMR false positives).
LONG_PASSENGERS = {
    "TTN", "MUC16", "MUC4", "MUC5B", "RYR1", "RYR2", "RYR3", "LRP1B", "CSMD1",
    "CSMD2", "CSMD3", "FLG", "FLG2", "DNAH5", "DNAH8", "DNAH9", "DNAH11", "DNAH17",
    "PCLO", "OBSCN", "SYNE1", "SYNE2", "FAT1", "FAT3", "FAT4", "USH2A", "NEB",
    "HMCN1", "SPTA1", "AHNAK", "AHNAK2", "ADGRV1", "PKHD1L1", "XIRP2", "DST",
    "MACF1", "PLEC", "NEFH", "RP1", "ZFHX4", "HRNR", "ANK3", "COL6A3",
}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def evaluate(cohort: str, n_samples: int, dig_results: str, dialect: str) -> list[dict]:
    """Run both BMRs for one cohort and return per-BMR summary rows."""
    out = Path("output") / cohort
    counts = out / "count_matrix.csv"
    if not (out / "bmr_pmfs.csv").exists():
        print(f"{cohort}: no CBaSE bmr_pmfs.csv in {out}; run dialect generate first.")
        return []

    # DIG background: per-sample PMFs with support up to the cohort's max observed
    # count (covers hypermutators -> avoids the degenerate-EM tau error on e.g. UCEC).
    max_count = int(pd.read_csv(counts, index_col=0).to_numpy().max())
    dig_results_to_bmr_pmfs(
        dig_results, n_samples, str(out / "bmr_pmfs.dig.csv"), max_count=max_count,
    )

    rows = []
    for bmr, pmf in [("cbase", "bmr_pmfs.csv"), ("dig", "bmr_pmfs.dig.csv")]:
        odir = out / f"id_{bmr}"
        _run([
            dialect, "identify", "-c", str(counts), "-b", str(out / pmf),
            "-o", str(odir), "-k", "100",
        ])
        sg = pd.read_csv(odir / "single_gene_results.csv")
        sg["base"] = sg["Gene Name"].str.rsplit("_", n=1).str[0]
        top20 = sg.sort_values("Pi", ascending=False).head(20)
        ttn = sg.loc[sg["base"] == "TTN", "Pi"]
        rows.append({
            "cohort": cohort,
            "N": n_samples,
            "bmr": bmr,
            "TTN_pi": round(float(ttn.max()), 3) if len(ttn) else None,
            "long_passengers_top20": len(set(top20["base"]) & LONG_PASSENGERS),
        })
    return rows


def main() -> None:
    """Run the BMR-sensitivity benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--dig-results", required=True)
    parser.add_argument("--dialect", default="dialect", help="path to the dialect CLI")
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        help="sample count per cohort (parallel to --cohorts); "
        "inferred from each count_matrix.csv if omitted",
    )
    args = parser.parse_args()

    samples = args.samples or [
        pd.read_csv(Path("output") / c / "count_matrix.csv", index_col=0).shape[0]
        for c in args.cohorts
    ]
    rows: list[dict] = []
    for cohort, n in zip(args.cohorts, samples, strict=True):
        rows.extend(evaluate(cohort, n, args.dig_results, args.dialect))

    if not rows:
        sys.exit("No results; check that `dialect generate` was run per cohort.")
    df = pd.DataFrame(rows)
    table = df.pivot_table(
        index="cohort",
        columns="bmr",
        values=["TTN_pi", "long_passengers_top20"],
        aggfunc="first",
    )
    print(table.reindex(args.cohorts).to_string())
    df.to_csv("output/bmr_sensitivity_summary.csv", index=False)
    print("\nsaved output/bmr_sensitivity_summary.csv")


if __name__ == "__main__":
    main()

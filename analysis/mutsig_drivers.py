"""DIALECT single-gene putative drivers under the MutSigCV per-sample BMR.

Builds the per-(gene_effect, sample) MutSigCV background (analysis.mutsig_persample_co),
runs DIALECT's single-gene EM (pi per gene-effect), and calls the single-gene driver
test: lambda_LR = 2*(ll(pi) - ll(0)) ~ chi^2_1 under H0 (no driver), p from chi2.sf,
then BH-FDR across the gene-effects. A gene-effect with q < FDR is a putative driver.

Events stay separated by effect: GENE_M (missense, Nmis coverage) and GENE_N
(nonsense+splice, Nnon+Nspl coverage) are distinct Gene objects with distinct per-sample
PMFs -- never conflated into one gene. Reports MutSig vs CBaSE side by side and flags
the long-gene / FLAGS artifacts, to see whether the per-sample BMR cleans the candidate
set. Usage::

    python analysis/mutsig_drivers.py --cohort BRCA -k 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from analysis.mutsig_persample_co import _build_genes, mutsig_persample_pmfs
from dialect.utils.identify import (
    create_single_gene_results,
    estimate_pi_for_each_gene,
)

_FDR = 0.05

# Long / hypermutator-prone passenger genes repeatedly flagged as artifacts (notes/11).
FLAGS = {
    "TTN", "MUC16", "MUC4", "MUC5B", "MUC17", "RYR1", "RYR2", "RYR3", "DNAH5",
    "DNAH8", "DNAH9", "DNAH11", "DNAH17", "SYNE1", "SYNE2", "OBSCN", "HERC1",
    "HERC2", "ASPM", "ASH1L", "CSMD1", "CSMD3", "FAT1", "FAT2", "FAT3", "FAT4",
    "PCLO", "NEB", "FLG", "LRP1B", "USH2A", "MACF1", "PKHD1", "SPTA1", "APOB",
    "XIRP2", "HMCN1", "DST", "RELN", "KIAA1109", "VPS13C", "CENPE", "KIAA1210",
    "SACS", "ANK2", "SCN10A", "MXRA5", "HYDIN", "CUBN", "COL6A3", "GPR98",
    "ADGRV1", "TENM1", "TG", "ZFHX4", "CSMD2", "DNAH2", "TPTE",
}


def _is_suspicious(gene_effect: str) -> bool:
    """True if the gene part (before _M/_N) is a known long-gene/FLAGS artifact."""
    return gene_effect.rsplit("_", 1)[0] in FLAGS


def _call_drivers(df: pd.DataFrame, fdr: float) -> pd.DataFrame:
    """Per-gene chi2.sf(LRT,1) -> BH q; flag suspicious; q<fdr marks a driver."""
    out = df.rename(columns={"Gene Name": "gene", "Likelihood Ratio": "LRT"}).copy()
    out["p"] = chi2.sf(out["LRT"].clip(lower=0), df=1)
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    out["susp"] = out["gene"].map(_is_suspicious)
    out["driver"] = out["q"] < fdr
    return out.sort_values("LRT", ascending=False)


def compute_mutsig_single_gene(cohort: str, root: Path, top_k: int) -> Path:
    """Build MutSig per-sample genes, estimate pi, write single_gene_results.csv."""
    counts = pd.read_csv(root / cohort / "count_matrix.csv", index_col=0)
    kmax = int(counts.to_numpy().max())
    pmfs = mutsig_persample_pmfs(cohort, root, list(counts.columns), counts.index, kmax)
    genes = _build_genes(counts, pmfs, top_k)

    out = root / cohort / "id_mutsig"
    out.mkdir(parents=True, exist_ok=True)
    fout = out / "single_gene_results.csv"
    estimate_pi_for_each_gene(genes.values())  # no cache file -> estimate from scratch
    create_single_gene_results(genes.values(), str(fout), cbase_phi_vals_present=False)
    return fout


def _report(df: pd.DataFrame, name: str) -> None:
    drivers = df[df["driver"]]
    n_susp = int(drivers["susp"].sum())
    print(f"=== {name}: {len(drivers)} putative drivers (single-gene q<{_FDR}), "
          f"{n_susp} suspicious ===")
    for _, r in drivers.iterrows():
        flag = "   <-- SUSPICIOUS" if r["susp"] else ""
        print(f"  {r['gene']:<16} pi={r['pi']:.3f}  LRT={r['LRT']:7.1f}  "
              f"q={r['q']:.1e}{flag}")
    print()


def main() -> None:
    """Compute + report MutSig vs CBaSE single-gene putative drivers for a cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=100)
    parser.add_argument("--results-root", default="output")
    args = parser.parse_args()
    root = Path(args.results_root)

    fout = compute_mutsig_single_gene(args.cohort, root, args.top_k)
    m = _call_drivers(pd.read_csv(fout).rename(columns={"Pi": "pi"}), _FDR)
    print(f"\nDIALECT single-gene putative drivers -- {args.cohort} "
          f"(BH q<{_FDR}, top-{args.top_k})\n")
    _report(m, "MutSig (per-sample)")

    cbase_fn = root / args.cohort / "id_cbase" / "single_gene_results.csv"
    if cbase_fn.exists():
        c = _call_drivers(pd.read_csv(cbase_fn).rename(columns={"Pi": "pi"}), _FDR)
        _report(c, "CBaSE (per-gene)")


if __name__ == "__main__":
    main()

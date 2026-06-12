"""Principled-FDR BMR-provider comparison + a suspicious-gene false-positive rate.

This replaces DIALECT's ad-hoc, asymmetric ME/CO significance thresholds (the epsilon
cutoff + per-method ``*_SIGNIFICANCE_THRESHOLDS``) with a single, symmetric test:

  - DIALECT already reports, per gene pair, its likelihood-ratio statistic
    ``lambda_LR = -2[ l(tau_null) - l(tau_hat) ]``, where ``tau_null`` is the
    independence model (a 1-constraint *interior* point of the tau-simplex).
    Under H0 of no interaction, ``lambda_LR ~ chi^2_1``, so the proper p-value is
    ``p = chi2.sf(lambda_LR, df=1)``.
  - We Benjamini-Hochberg adjust the p-values across all tested pairs (a single,
    unified family for ME and CO) and call an interaction significant at ``q < Q``.
  - Mutual exclusivity vs co-occurrence is read from the *sign* of Marshall-Olkin
    ``rho`` (rho < 0 -> ME, rho > 0 -> CO) -- one test, direction from the effect,
    instead of two different thresholds.

Validation readout: of the FDR-significant interactions, what fraction involve a
"suspicious" gene -- a FLAGS / MutSig covariate-inflated artifact (long, late-
replicating, low-expression genes whose somatic-count inflation is a background-model
failure, not biology). A well-calibrated BMR should yield a *lower* suspicious-gene
fraction at the same FDR while preserving the real driver interactions.

Usage::

    python analysis/bmr_fdr_comparison.py \
        --cohorts CHOL LAML PRAD LUAD BRCA UCEC \
        --bmrs cbase dig --fdr 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------------- #
# Suspicious genes: the canonical FLAGS (Shyr et al. 2014, BMC Med Genomics) and other
# long / late-replicating / low-expression genes that MutSig-style covariate models
# down-weight. Deliberately EXCLUDES established drivers (so real biology is not
# penalized). A pair is "suspicious" if EITHER gene's base symbol is in this set.
# ---------------------------------------------------------------------------------- #
SUSPICIOUS_GENES = frozenset({
    # FLAGS core
    "TTN", "MUC16", "MUC4", "MUC5B", "MUC2", "MUC6", "MUC12", "MUC17", "MUC19",
    "MUC5AC", "SYNE1", "SYNE2", "NEB", "OBSCN", "FSIP2", "GPR98", "ADGRV1",
    "RYR1", "RYR2", "RYR3", "FAT1", "FAT2", "FAT3", "FAT4", "USH2A", "PKHD1",
    "PKHD1L1", "DNAH1", "DNAH2", "DNAH3", "DNAH5", "DNAH6", "DNAH7", "DNAH8",
    "DNAH9", "DNAH10", "DNAH11", "DNAH17", "DNAH14", "HMCN1", "HMCN2", "FLG",
    "FLG2", "AHNAK", "AHNAK2", "MACF1", "PLEC", "DST", "SPTA1", "SPTBN5",
    "CSMD1", "CSMD2", "CSMD3", "LRP1B", "LRP2", "CUBN", "PCLO", "PCDH15",
    # large/recurrent co-occurrence artifacts seen empirically (long genes)
    "HERC1", "HERC2", "ANK2", "ANK3", "ASPM", "ASH1L", "XIRP2", "ZFHX4",
    "NBEA", "RIMS2", "DYNC1H1", "DNAH12", "COL6A3", "COL11A1",
    "NEFH", "RP1", "HRNR", "VPS13B", "VPS13C", "TENM1", "TENM3",
    "KIAA1109", "FCGBP", "RELN", "DCHS2", "PKD1L1", "PKD1L2",
})


def _base(gene: str) -> str:
    """Strip DIALECT's ``_M`` / ``_N`` effect suffix to get the gene symbol."""
    return gene.rsplit("_", 1)[0]


def _is_suspicious_pair(gene_a: str, gene_b: str) -> bool:
    return _base(gene_a) in SUSPICIOUS_GENES or _base(gene_b) in SUSPICIOUS_GENES


def call_significant(pairwise: pd.DataFrame, fdr: float) -> pd.DataFrame:
    """Add p-value, BH q-value, direction, and significance to a pairwise table."""
    df = pairwise.copy()
    lrt = df["Likelihood Ratio"].clip(lower=0).to_numpy()
    df["p_value"] = chi2.sf(lrt, df=1)
    df["q_value"] = multipletests(df["p_value"], method="fdr_bh")[1]
    df["direction"] = np.where(df["Rho"] < 0, "ME", "CO")
    df["significant"] = df["q_value"] < fdr
    df["suspicious"] = [
        _is_suspicious_pair(a, b)
        for a, b in zip(df["Gene A"], df["Gene B"], strict=False)
    ]
    return df


def _susp_frac(sub: pd.DataFrame) -> float:
    return round(float(sub["suspicious"].mean()), 3) if len(sub) else 0.0


def summarize(cohort: str, bmr: str, df: pd.DataFrame) -> dict:
    """Per (cohort, BMR) significance + suspicious-gene-fraction summary row.

    Breaks the suspicious fraction down by direction: real ME biology should stay
    clean while the CO calls are where the long-gene artifacts accumulate.
    """
    sig = df[df["significant"]]
    me = sig[sig["direction"] == "ME"]
    co = sig[sig["direction"] == "CO"]
    n_sig = len(sig)
    return {
        "cohort": cohort,
        "bmr": bmr,
        "n_sig": n_sig,
        "n_ME": len(me),
        "ME_susp": _susp_frac(me),
        "n_CO": len(co),
        "CO_susp": _susp_frac(co),
        "susp_frac": round(float(sig["suspicious"].mean()), 3) if n_sig else 0.0,
    }


def main() -> None:
    """Run the principled-FDR BMR comparison across cohorts and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--bmrs", nargs="+", default=["cbase", "dig"])
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--out", default="output/bmr_fdr_comparison.csv")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows: list[dict] = []
    for cohort in args.cohorts:
        for bmr in args.bmrs:
            fn = root / cohort / f"id_{bmr}" / "pairwise_interaction_results.csv"
            if not fn.exists():
                print(f"skip {cohort}/{bmr}: {fn} not found")
                continue
            called = call_significant(pd.read_csv(fn), args.fdr)
            rows.append(summarize(cohort, bmr, called))

    if not rows:
        msg = "no result files found; run `dialect identify` per cohort/BMR first"
        raise SystemExit(msg)

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"\nPrincipled-FDR BMR comparison (BH q < {args.fdr})\n")
    print(summary.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

"""Proof-of-principle fix: a burden-aware test collapses the spurious co-occurrence.

For every gene pair, compare two significance tests over the SAME data:

  - DIALECT (per-gene background only): p = chi2.sf(lambda_LR, df=1) on DIALECT's
    likelihood-ratio statistic.
  - Burden-aware: logistic  mutated_B ~ mutated_A + log(sample_burden);  the p-value
    of the mutated_A coefficient tests whether A and B co-occur BEYOND what each
    sample's total mutation burden already explains.

Both are BH-FDR'd across all pairs (q<0.05). We report, per cohort, the number of
significant CO calls and the FLAGS/MutSig "suspicious"-gene fraction under each test.
The burden-aware test is a test-level proxy for the principled fix (a per-sample
background B_s); collapsing the spurious-CO count + suspicious fraction shows the
direction works before building it into DIALECT's EM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from analysis.bmr_fdr_comparison import SUSPICIOUS_GENES, _base, call_significant
from analysis.confirm_co_burden_confound import _logit_pvalue

_FDR = 0.05


def _suspicious(gene_a: str, gene_b: str) -> bool:
    return _base(gene_a) in SUSPICIOUS_GENES or _base(gene_b) in SUSPICIOUS_GENES


def burden_aware_calls(
    pairwise: pd.DataFrame,
    binary: pd.DataFrame,
    burden: np.ndarray,
) -> pd.DataFrame:
    """Burden-adjusted co-occurrence p-value + BH-FDR for every pair."""
    log_burden = np.log1p(burden).reshape(-1, 1)
    pvals = []
    for gene_a, gene_b in zip(pairwise["Gene A"], pairwise["Gene B"], strict=False):
        if gene_a not in binary.columns or gene_b not in binary.columns:
            pvals.append(np.nan)
            continue
        a = binary[gene_a].to_numpy(dtype=float)
        b = binary[gene_b].to_numpy(dtype=float)
        if b.sum() in (0, len(b)) or a.sum() in (0, len(a)):
            pvals.append(np.nan)
            continue
        p = _logit_pvalue(b, np.column_stack([a, log_burden]))
        pvals.append(np.nan if p is None else p)
    out = pairwise.copy()
    out["p_burden"] = pvals
    mask = out["p_burden"].notna()
    out["q_burden"] = np.nan
    out.loc[mask, "q_burden"] = multipletests(
        out.loc[mask, "p_burden"], method="fdr_bh",
    )[1]
    return out


def _co_stats(df: pd.DataFrame, sig_mask: pd.Series, co_mask: pd.Series) -> dict:
    sig_co = df[sig_mask & co_mask]
    n = len(sig_co)
    susp = sum(
        _suspicious(a, b)
        for a, b in zip(sig_co["Gene A"], sig_co["Gene B"], strict=False)
    )
    return {"n_CO": n, "CO_susp": round(susp / n, 3) if n else 0.0}


def analyze(cohort: str, root: Path) -> dict:
    """Compare DIALECT vs burden-aware CO calls for one cohort (CBaSE)."""
    counts = pd.read_csv(root / cohort / "count_matrix.csv", index_col=0)
    binary = counts > 0
    burden = counts.to_numpy().sum(axis=1)
    pairwise = pd.read_csv(
        root / cohort / "id_cbase" / "pairwise_interaction_results.csv",
    )

    dia = call_significant(pairwise, _FDR)
    co_dir = dia["direction"] == "CO"
    dia_stats = _co_stats(dia, dia["significant"], co_dir)

    ba = burden_aware_calls(pairwise, binary, burden)
    ba_sig = ba["q_burden"] < _FDR
    ba_stats = _co_stats(dia, ba_sig.fillna(value=False), co_dir)

    return {
        "cohort": cohort,
        "DIALECT_n_CO": dia_stats["n_CO"],
        "DIALECT_CO_susp": dia_stats["CO_susp"],
        "burden_aware_n_CO": ba_stats["n_CO"],
        "burden_aware_CO_susp": ba_stats["CO_susp"],
        "CO_reduction": round(1 - ba_stats["n_CO"] / dia_stats["n_CO"], 3)
        if dia_stats["n_CO"]
        else None,
    }


def main() -> None:
    """Run the burden-aware-vs-DIALECT CO comparison and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--out", default="output/burden_aware_co_demo.csv")
    args = parser.parse_args()

    root = Path(args.results_root)
    summary = pd.DataFrame([analyze(c, root) for c in args.cohorts])
    pd.set_option("display.width", 200)
    print("\nDIALECT CO vs burden-aware CO (BH q<0.05, CBaSE)\n")
    print(summary.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

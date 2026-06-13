"""Confirm DIALECT's co-occurrence calls are a per-sample tumor-burden confound.

Three model-free diagnostics on the observed count matrix, per cohort, over the gene
pairs DIALECT calls significantly co-occurring (principled BH-FDR, see
``bmr_fdr_comparison``). No model re-runs, no parameter fits beyond plain GLMs --
purely a test of whether the CO signal is explained by sample burden.

1. BURDEN-CONDITIONING. For each significant CO pair fit two logistic regressions of
   one gene's mutation indicator: ``B ~ A`` and ``B ~ A + log(sample_burden)``. Report
   the fraction of pairs whose A<->B association is significant marginally (p<0.05) but
   vanishes (p>=0.05) once burden is included -- i.e. burden explains the co-occurrence.

2. HYPERMUTATOR LEVERAGE. Recompute the raw pairwise co-occurrence odds ratio with and
   without the top-decile-burden samples; report how far the median CO odds ratio
   collapses when the hypermutators are dropped.

3. LONG-GENE DEGREE. Correlate each gene's CO degree (number of significant CO
   partners) with its total mutation count (a size/burden proxy). A strong positive
   correlation means high-count genes co-occur with everything -- the shared-cause
   fingerprint.

Usage::

    python analysis/confirm_co_burden_confound.py --cohorts CHOL BRCA LUAD UCEC
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from analysis.bmr_fdr_comparison import call_significant

_SIG_ALPHA = 0.05
_TOP_BURDEN_DECILE = 0.10
_MAX_PAIRS = 1500  # subsample significant-CO pairs per cohort for the GLM test


def _logit_pvalue(y: np.ndarray, x_cols: np.ndarray) -> float | None:
    """p-value of the FIRST predictor's coefficient in a logistic GLM (or None)."""
    design = sm.add_constant(x_cols, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            res = sm.Logit(y, design).fit(disp=0, maxiter=50)
    except Exception:  # noqa: BLE001 - GLM fit has many numeric failure modes
        return None
    return float(res.pvalues[1])


def burden_conditioning(
    binary: pd.DataFrame,
    burden: np.ndarray,
    co_pairs: list[tuple[str, str]],
    rng: np.random.Generator,
) -> dict:
    """Fraction of CO pairs whose A<->B link is explained away by sample burden."""
    if len(co_pairs) > _MAX_PAIRS:
        idx = rng.choice(len(co_pairs), _MAX_PAIRS, replace=False)
        co_pairs = [co_pairs[i] for i in idx]
    log_burden = np.log1p(burden).reshape(-1, 1)
    n_marg_sig = explained_away = tested = 0
    for gene_a, gene_b in co_pairs:
        if gene_a not in binary.columns or gene_b not in binary.columns:
            continue
        a = binary[gene_a].to_numpy(dtype=float)
        b = binary[gene_b].to_numpy(dtype=float)
        if b.sum() in (0, len(b)) or a.sum() in (0, len(a)):
            continue
        p_marg = _logit_pvalue(b, a.reshape(-1, 1))
        p_full = _logit_pvalue(b, np.column_stack([a, log_burden]))
        if p_marg is None or p_full is None:
            continue
        tested += 1
        if p_marg < _SIG_ALPHA:
            n_marg_sig += 1
            if p_full >= _SIG_ALPHA:
                explained_away += 1
    return {
        "pairs_tested": tested,
        "marginally_assoc": n_marg_sig,
        "explained_by_burden": explained_away,
        "frac_explained": round(explained_away / n_marg_sig, 3) if n_marg_sig else None,
    }


def _median_log_or(binary: pd.DataFrame, pairs: list[tuple[str, str]]) -> float:
    """Median Haldane-corrected log odds ratio over the given pairs."""
    log_ors = []
    for gene_a, gene_b in pairs:
        if gene_a not in binary.columns or gene_b not in binary.columns:
            continue
        a = binary[gene_a].to_numpy(dtype=bool)
        b = binary[gene_b].to_numpy(dtype=bool)
        n11 = float((a & b).sum()) + 0.5
        n10 = float((a & ~b).sum()) + 0.5
        n01 = float((~a & b).sum()) + 0.5
        n00 = float((~a & ~b).sum()) + 0.5
        log_ors.append(np.log((n11 * n00) / (n10 * n01)))
    return float(np.median(log_ors)) if log_ors else float("nan")


def hypermutator_leverage(
    binary: pd.DataFrame,
    burden: np.ndarray,
    co_pairs: list[tuple[str, str]],
) -> dict:
    """Median CO odds ratio with vs without the top-decile-burden samples."""
    cutoff = np.quantile(burden, 1 - _TOP_BURDEN_DECILE)
    keep = burden < cutoff
    full = _median_log_or(binary, co_pairs)
    dropped = _median_log_or(binary.loc[keep], co_pairs)
    return {
        "median_logOR_full": round(full, 3),
        "median_logOR_drop_top10pct": round(dropped, 3),
        "collapse_frac": round(1 - dropped / full, 3) if full else None,
    }


def long_gene_degree(
    counts: pd.DataFrame,
    co_pairs: list[tuple[str, str]],
) -> dict:
    """Spearman corr of a gene's CO-degree with its total mutation count."""
    degree: dict[str, int] = {}
    for gene_a, gene_b in co_pairs:
        degree[gene_a] = degree.get(gene_a, 0) + 1
        degree[gene_b] = degree.get(gene_b, 0) + 1
    if len(degree) < 3:
        return {"spearman_degree_vs_count": None, "n_genes": len(degree)}
    genes = [g for g in degree if g in counts.columns]
    deg = [degree[g] for g in genes]
    tot = [int(counts[g].sum()) for g in genes]
    rho, _ = spearmanr(deg, tot)
    return {"spearman_degree_vs_count": round(float(rho), 3), "n_genes": len(genes)}


def analyze(cohort: str, root: Path, rng: np.random.Generator) -> dict:
    """Run all three diagnostics for one cohort (CBaSE result set)."""
    counts = pd.read_csv(root / cohort / "count_matrix.csv", index_col=0)
    binary = counts > 0
    burden = counts.to_numpy().sum(axis=1)
    pairwise = pd.read_csv(
        root / cohort / "id_cbase" / "pairwise_interaction_results.csv",
    )
    called = call_significant(pairwise, _SIG_ALPHA)
    co = called[called["significant"] & (called["direction"] == "CO")]
    co_pairs = list(zip(co["Gene A"], co["Gene B"], strict=False))
    row = {"cohort": cohort, "n_sig_CO": len(co_pairs)}
    row.update(burden_conditioning(binary, burden, co_pairs, rng))
    row.update(hypermutator_leverage(binary, burden, co_pairs))
    row.update(long_gene_degree(counts, co_pairs))
    return row


def main() -> None:
    """Run the CO-confound confirmation across cohorts and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--out", default="output/co_burden_confound.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    root = Path(args.results_root)
    rows = [analyze(c, root, rng) for c in args.cohorts]
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\nCO-as-burden-confound confirmation (significant CO pairs, CBaSE)\n")
    print(summary.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

"""Run MEGSA (Mutually Exclusive Gene Set Analysis) via an Rscript subprocess.

MEGSA's likelihood is implemented in ``external/MEGSA/MEGSA.R``. We invoke it
through ``Rscript`` rather than ``rpy2`` to avoid R/rpy2 ABI coupling: the only
runtime requirement is a working ``Rscript`` on PATH. The R driver computes the
MEGSA S statistic (a likelihood-ratio test) for each requested gene pair.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

_MEGSA_R = Path(__file__).resolve().parents[3] / "external" / "MEGSA" / "MEGSA.R"

# R driver: source MEGSA.R and compute funEstimate(...)$S for each gene pair.
_DRIVER = r"""
args  <- commandArgs(trailingOnly = TRUE)
source(args[1])
mat   <- as.matrix(read.csv(args[2], row.names = 1, check.names = FALSE))
pairs <- read.csv(args[3], stringsAsFactors = FALSE, check.names = FALSE)
s_scores <- numeric(nrow(pairs))
for (i in seq_len(nrow(pairs))) {
    sub <- mat[, c(pairs$A[i], pairs$B[i]), drop = FALSE]
    s_scores[i] <- tryCatch(funEstimate(sub, tol = 1e-7)$S,
                            error = function(e) NA_real_)
}
out <- data.frame(A = pairs$A, B = pairs$B, S = s_scores, check.names = FALSE)
write.csv(out, args[4], row.names = FALSE)
"""


def run_megsa_analysis(cnt_df: pd.DataFrame, interactions: list) -> pd.DataFrame:
    """Compute MEGSA S-score / p-value / q-value for each interaction pair."""
    binary = (cnt_df > 0).astype(int)
    pairs = [(ixn.gene_a.name, ixn.gene_b.name) for ixn in interactions]
    genes = sorted({gene for pair in pairs for gene in pair})

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        binary[genes].to_csv(tdp / "matrix.csv")
        pd.DataFrame(pairs, columns=["A", "B"]).to_csv(tdp / "pairs.csv", index=False)
        (tdp / "driver.R").write_text(_DRIVER)
        out_fn = tdp / "out.csv"
        subprocess.run(
            [
                "Rscript",
                str(tdp / "driver.R"),
                str(_MEGSA_R),
                str(tdp / "matrix.csv"),
                str(tdp / "pairs.csv"),
                str(out_fn),
            ],
            check=True,
        )
        res = pd.read_csv(out_fn)

    results_df = res.rename(
        columns={"A": "Gene A", "B": "Gene B", "S": "MEGSA S-Score (LRT)"},
    )
    results_df["MEGSA P-Val"] = 0.5 * chi2.sf(results_df["MEGSA S-Score (LRT)"], df=1)
    results_df["MEGSA Q-Val"] = multipletests(
        results_df["MEGSA P-Val"].fillna(1.0),
        method="fdr_bh",
    )[1]
    return results_df

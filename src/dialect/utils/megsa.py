"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def load_megsa_r_code(file_path: str) -> SignatureTranslatedAnonymousPackage:
    """TODO: Add docstring."""
    with Path(file_path).open() as r_file:
        r_code = r_file.read()
    return SignatureTranslatedAnonymousPackage(r_code, "MEGSA")


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def run_megsa_analysis(cnt_df: pd.DataFrame, interactions: list) -> pd.DataFrame:
    """TODO: Add docstring."""
    cnt_df = (cnt_df > 0).astype(int)
    pandas2ri.activate()
    current_dir = Path(__file__).resolve().parent
    relative_path = "../../../external/MEGSA/MEGSA.R"
    file_path = Path(current_dir) / relative_path
    megsa_package = load_megsa_r_code(file_path)
    results = []
    for interaction in interactions:
        gene_a, gene_b = interaction.gene_a.name, interaction.gene_b.name
        gene_pair_matrix = cnt_df[[gene_a, gene_b]]
        r_result = megsa_package.funEstimate(
            mutationMat=pandas2ri.py2rpy(gene_pair_matrix),
            tol=1e-7,
        )
        s_score = r_result.rx2("S")[0]
        p_val = 0.5 * chi2.sf(s_score, df=1)
        results.append(
            {
                "Gene A": gene_a,
                "Gene B": gene_b,
                "MEGSA S-Score (LRT)": s_score,
                "MEGSA P-Val": p_val,
            },
        )
    results_df = pd.DataFrame(results)
    q_values = multipletests(results_df["MEGSA P-Val"], method="fdr_bh")[1]
    results_df["MEGSA Q-Val"] = q_values
    return results_df

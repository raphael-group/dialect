import os
import logging
import pandas as pd
from scipy.stats import chi2
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def load_megsa_r_code(file_path):
    with open(file_path, "r") as r_file:
        r_code = r_file.read()
    return SignatureTranslatedAnonymousPackage(r_code, "MEGSA")


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def run_megsa_analysis(cnt_df, interactions):
    logging.info("Running MEGSA analysis...")
    cnt_df = (cnt_df > 0).astype(int)  # binarize cnt_df
    pandas2ri.activate()
    current_dir = os.path.dirname(__file__)  # Directory of megsa.py
    relative_path = "../../../external/MEGSA/MEGSA.R"  # Relative path to MEGSA.R
    file_path = os.path.abspath(os.path.join(current_dir, relative_path))
    megsa_package = load_megsa_r_code(file_path)
    results = []
    for interaction in interactions:
        gene_a, gene_b = interaction.gene_a.name, interaction.gene_b.name
        gene_pair_matrix = cnt_df[[gene_a, gene_b]]  # subset to current pair
        r_result = megsa_package.funEstimate(
            mutationMat=pandas2ri.py2rpy(gene_pair_matrix), tol=1e-7
        )
        S_score = r_result.rx2("S")[0]  # LRT statistic
        p_val = 0.5 * chi2.sf(S_score, df=1)
        results.append(
            {
                "Gene A": gene_a,
                "Gene B": gene_b,
                "MEGSA S-Score (LRT)": S_score,
                "MEGSA P-Val": p_val,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df

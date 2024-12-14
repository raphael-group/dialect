"""
This script computes, analyzes, and visualizes log-likelihood curves for genes
based on count matrix and BMR PMFs, exploring the presence of a single optimum.
"""

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from dialect.utils.identify import *
from dialect.models.gene import Gene

# ---------------------------------------------------------------------------- #
#                                   CONSTANTS                                  #
# ---------------------------------------------------------------------------- #

CNT_MTX_FN = "/Users/work/Research/dialect/dialect_latest/output/count_matrix.csv"
BMR_PMFS_FN = "/Users/work/Research/dialect/dialect_latest/output/bmr_pmfs.csv"


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def is_single_optimum(log_likelihoods):
    """
    Check if the log-likelihood function has a single optimum (convex/concave).

    :param log_likelihoods: (list of float) Log-likelihood values.
    :return: (bool) True if there is a single optimum, False otherwise.
    """
    first_diff = np.diff(log_likelihoods)  # first derivative
    sign_changes = np.diff(np.sign(first_diff))  # sign changes in first derivative
    return np.count_nonzero(sign_changes != 0) <= 1


def explore_log_likelihoods_for_single_gene_estimation(cnt_df, bmr_dict):
    for gene_name in cnt_df.columns:
        gene = Gene(
            name=gene_name,
            counts=cnt_df[gene_name].values,
            bmr_pmf={
                i: bmr_dict[gene_name][i] for i in range(len(bmr_dict[gene_name]))
            },
        )
        pi_values = np.linspace(0, 0.99, 100)
        log_likelihoods = [gene.compute_log_likelihood(pi) for pi in pi_values]
        if not is_single_optimum(log_likelihoods):
            print(f"The log-likelihood function for gene {gene_name} is not convex.")
            break


def visualize_log_likelihood_plot_for_gene(cnt_df, bmr_dict, gene_name):
    if gene_name not in cnt_df.columns or gene_name not in bmr_dict:
        raise ValueError(f"Gene {gene_name} not found in count matrix or BMR PMFs.")

    gene = Gene(
        name=gene_name,
        counts=cnt_df[gene_name].values,
        bmr_pmf={i: bmr_dict[gene_name][i] for i in range(len(bmr_dict[gene_name]))},
    )
    pi_values = np.linspace(0, 0.99, 100)
    log_likelihoods = [gene.compute_log_likelihood(pi) for pi in pi_values]
    plt.plot(pi_values, log_likelihoods, label=gene_name)
    plt.title(f"Log-Likelihood for {gene_name}")
    plt.xlabel("pi")
    plt.ylabel("Log-Likelihood")
    plt.show()


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(CNT_MTX_FN, BMR_PMFS_FN)
    explore_log_likelihoods_for_single_gene_estimation(cnt_df, bmr_dict)
    visualize_log_likelihood_plot_for_gene(cnt_df, bmr_dict, gene_name="TP53_M")

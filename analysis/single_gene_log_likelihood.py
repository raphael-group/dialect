"""
This script computes, analyzes, and visualizes log-likelihood curves for genes
based on count matrix and BMR PMFs, exploring the presence of a single optimum.
"""

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

import logging
import numpy as np
import matplotlib.pyplot as plt
from dialect.utils.identify import *
from dialect.models.gene import Gene


# TODO: extract and refactor user interaction methods
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
    logging.info("Exploring log-likelihood curves for single-gene estimation.")
    logging.info(
        f"Evaluating which of {len(cnt_df.columns)} genes have non-convex log likelihood curves..."
    )
    for gene_name in cnt_df.columns:
        gene = Gene(
            name=gene_name,
            samples=cnt_df.index,
            counts=cnt_df[gene_name].values,
            bmr_pmf={
                i: bmr_dict[gene_name][i] for i in range(len(bmr_dict[gene_name]))
            },
        )
        pi_values = np.linspace(0, 0.99, 100)
        log_likelihoods = [gene.compute_log_likelihood(pi) for pi in pi_values]
        if not is_single_optimum(log_likelihoods):
            print(f"The log-likelihood function for gene {gene_name} is not convex.")

    logging.info(
        "Exploration of log-likelihood curves for single-gene estimation complete."
    )


def visualize_log_likelihood_plot_for_gene(cnt_df, bmr_dict, gene_name):
    if gene_name not in cnt_df.columns or gene_name not in bmr_dict:
        raise ValueError(f"Gene {gene_name} not found in count matrix or BMR PMFs.")

    gene = Gene(
        name=gene_name,
        samples=cnt_df.index,
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
    cnt_mtx_path = input("Enter the path to the count matrix file: ").strip()
    bmr_pmfs_path = input("Enter the path to the BMR PMFs file: ").strip()

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)
    check_convexity = (
        input(
            "Would you like to check if all genes have convex log-likelihood curves? (yes/no): "
        )
        .strip()
        .lower()
    )
    if check_convexity in ["yes", "y"]:
        explore_log_likelihoods_for_single_gene_estimation(cnt_df, bmr_dict)

    print("\nType 'exit' to quit the program at any time.")
    print("Note: Gene names should end with '_M' (missense) or '_N' (nonsense).")

    while True:
        gene_name = input("Enter the name of the gene to visualize: ").strip()

        if gene_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        if gene_name not in cnt_df.columns:
            print(
                f"Gene '{gene_name}' does not exist in the count matrix. "
                "Please ensure the name ends with '_M' or '_N'. Try again."
            )
        else:
            # Visualize the log-likelihood plot
            try:
                visualize_log_likelihood_plot_for_gene(cnt_df, bmr_dict, gene_name)
            except Exception as e:
                print(f"An error occurred: {e}")

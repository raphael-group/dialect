"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #

import contextlib
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dialect.models.gene import Gene
from dialect.utils.helpers import load_cnt_mtx_and_bmr_pmfs


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def is_single_optimum(log_likelihoods: list) -> bool:
    """TODO: Add docstring."""
    first_diff = np.diff(log_likelihoods)
    sign_changes = np.diff(np.sign(first_diff))
    return np.count_nonzero(sign_changes != 0) <= 1


def explore_log_likelihoods_for_single_gene_estimation(
    cnt_df: pd.DataFrame,
    bmr_dict: dict,
) -> None:
    """TODO: Add docstring."""
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
            pass


def visualize_log_likelihood_plot_for_gene(
    cnt_df: pd.DataFrame,
    bmr_dict: dict,
    gene_name: str,
) -> None:
    """TODO: Add docstring."""
    if gene_name not in cnt_df.columns or gene_name not in bmr_dict:
        msg = f"Gene {gene_name} not found in count matrix or BMR PMFs."
        raise ValueError(msg)

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


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    cnt_mtx_path = input("Enter the path to the count matrix file: ").strip()
    bmr_pmfs_path = input("Enter the path to the BMR PMFs file: ").strip()

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)
    check_convexity = (
        input(
            "Would you like to check if all genes have convex log-likelihood curves? "
            "(yes/no): ",
        )
        .strip()
        .lower()
    )
    if check_convexity in ["yes", "y"]:
        explore_log_likelihoods_for_single_gene_estimation(cnt_df, bmr_dict)

    while True:
        gene_name = input("Enter the name of the gene to visualize: ").strip()

        if gene_name.lower() == "exit":
            break

        if gene_name not in cnt_df.columns:
            pass
        else:
            with contextlib.suppress(Exception):
                visualize_log_likelihood_plot_for_gene(cnt_df, bmr_dict, gene_name)


if __name__ == "__main__":
    main()

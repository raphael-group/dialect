"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #

import logging

import numpy as np

from dialect.utils.helpers import initialize_gene_objects
from dialect.utils.identify import load_cnt_mtx_and_bmr_pmfs
from dialect.utils.simulate import simulate_interaction_pair


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def calculate_tau_recovery_error(
    true_taus: list,
    estimated_taus: list,
) -> float:
    """TODO: Add docstring."""
    true_taus = np.array(true_taus)
    estimated_taus = np.array(estimated_taus)
    error = np.abs(true_taus - estimated_taus)
    return error.mean()


def count_false_positives(
    true_taus: list,
    estimated_taus: list,
    threshold: float = 1e-6,
) -> int:
    """TODO: Add docstring."""
    true_zeros = np.array(true_taus) < threshold
    false_positives = np.logical_and(true_zeros, np.array(estimated_taus) >= threshold)
    return false_positives.sum()


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    cnt_mtx_path = input("Enter the path to the count matrix file: ").strip()
    bmr_pmfs_path = input("Enter the path to the BMR PMFs file: ").strip()

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)

    gene_objects = initialize_gene_objects(cnt_df, bmr_dict)

    while True:
        gene_a_name = input("Enter the name of Gene A: ").strip()
        if gene_a_name.lower() == "exit":
            break
        if gene_a_name not in gene_objects:
            continue

        gene_b_name = input("Enter the name of Gene B: ").strip()
        if gene_b_name.lower() == "exit":
            break
        if gene_b_name not in gene_objects:
            continue

        try:
            tau_00 = float(input("Enter the value for tau_00 (e.g., 0.8): ").strip())
            tau_01 = float(input("Enter the value for tau_01 (e.g., 0.1): ").strip())
            tau_10 = float(input("Enter the value for tau_10 (e.g., 0.1): ").strip())
            tau_11 = float(input("Enter the value for tau_11 (e.g., 0.0): ").strip())
        except ValueError:
            continue

        if not np.isclose(sum([tau_00, tau_01, tau_10, tau_11]), 1):
            continue

        gene_a = gene_objects[gene_a_name]
        gene_b = gene_objects[gene_b_name]

        nsamples = len(cnt_df)
        iterations = 100
        threshold = 1e-6

        total_recovery_error = 0.0
        total_false_positives = 0

        for _ in range(iterations):
            interaction = simulate_interaction_pair(
                gene_a.bmr_pmf,
                gene_b.bmr_pmf,
                tau_01,
                tau_10,
                tau_11,
                nsamples,
            )

            try:
                interaction.estimate_tau_with_em_from_scratch()
            except (ValueError, RuntimeError):
                continue

            estimated_taus = (
                interaction.tau_00,
                interaction.tau_01,
                interaction.tau_10,
                interaction.tau_11,
            )

            true_taus = (tau_00, tau_01, tau_10, tau_11)
            total_recovery_error += calculate_tau_recovery_error(
                true_taus,
                estimated_taus,
            )
            total_false_positives += count_false_positives(
                true_taus,
                estimated_taus,
                threshold,
            )

        average_recovery_error = total_recovery_error / iterations
        average_false_positive_rate = total_false_positives / (
            iterations * len(true_taus)
        )
        logging.info(
            "Average recover error: %f\nAverage false positive rate: %f",
            average_recovery_error,
            average_false_positive_rate,
        )

if __name__ == "__main__":
    main()

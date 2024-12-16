"""
This script analyzes tau recovery error in simulated interactions over multiple iterations
to evaluate the model's accuracy in estimating tau parameters. The user provides the count
matrix, BMR PMFs, gene names, and tau parameters to generate simulations and computes
the recovery error and misestimation rates.
"""

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import logging
import numpy as np
from dialect.utils.simulate import simulate_interaction_pair
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.identify import load_cnt_mtx_and_bmr_pmfs


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def initialize_gene_objects(cnt_df, bmr_dict):
    """
    Create a dictionary mapping gene names to Gene objects.
    """
    gene_objects = {}
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].values
        bmr_pmf = {i: bmr_dict[gene_name][i] for i in range(len(bmr_dict[gene_name]))}
        gene_objects[gene_name] = Gene(
            name=gene_name, samples=cnt_df.index, counts=counts, bmr_pmf=bmr_pmf
        )
    logging.info(f"Initialized {len(gene_objects)} Gene objects.")
    return gene_objects


def calculate_tau_recovery_error(true_taus, estimated_taus):
    """
    Calculate and return the tau recovery error.

    :param true_taus (tuple): True tau values used in simulation.
    :param estimated_taus (tuple): Tau values estimated by the model.
    :return (float): Mean absolute error between true and estimated tau values.
    """
    true_taus = np.array(true_taus)
    estimated_taus = np.array(estimated_taus)
    error = np.abs(true_taus - estimated_taus)
    return error.mean()


def count_false_positives(true_taus, estimated_taus, threshold=1e-6):
    """
    Count the number of false positives where tau values are misestimated as non-zero.

    :param true_taus (tuple): True tau values used in simulation.
    :param estimated_taus (tuple): Tau values estimated by the model.
    :param threshold (float): Threshold below which values are considered zero.
    :return (int): Count of false positives.
    """
    true_zeros = np.array(true_taus) < threshold
    false_positives = np.logical_and(true_zeros, np.array(estimated_taus) >= threshold)
    return false_positives.sum()


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Tau Recovery Over Iterations Analysis Script")

    # Prompt for count matrix and BMR PMFs file
    cnt_mtx_path = input("Enter the path to the count matrix file: ").strip()
    bmr_pmfs_path = input("Enter the path to the BMR PMFs file: ").strip()

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)

    # Initialize gene objects
    gene_objects = initialize_gene_objects(cnt_df, bmr_dict)

    print("\nType 'exit' to quit the program at any time.")
    while True:
        # Prompt for gene A
        gene_a_name = input("Enter the name of Gene A: ").strip()
        if gene_a_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        if gene_a_name not in gene_objects:
            print(f"Gene '{gene_a_name}' does not exist. Try again.")
            continue

        # Prompt for gene B
        gene_b_name = input("Enter the name of Gene B: ").strip()
        if gene_b_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        if gene_b_name not in gene_objects:
            print(f"Gene '{gene_b_name}' does not exist. Try again.")
            continue

        # Prompt for tau parameters
        try:
            tau_00 = float(input("Enter the value for tau_00 (e.g., 0.8): ").strip())
            tau_01 = float(input("Enter the value for tau_01 (e.g., 0.1): ").strip())
            tau_10 = float(input("Enter the value for tau_10 (e.g., 0.1): ").strip())
            tau_11 = float(input("Enter the value for tau_11 (e.g., 0.0): ").strip())
        except ValueError:
            print("Invalid tau value entered. Please enter numeric values.")
            continue

        # Ensure tau parameters sum to 1
        if not np.isclose(sum([tau_00, tau_01, tau_10, tau_11]), 1):
            print("Tau parameters must sum to 1. Please re-enter.")
            continue

        # Get Gene objects
        gene_a = gene_objects[gene_a_name]
        gene_b = gene_objects[gene_b_name]

        # Simulation settings
        nsamples = len(cnt_df)
        iterations = 1000
        threshold = 1e-6

        # Metrics for analysis
        total_recovery_error = 0.0
        total_false_positives = 0

        print(f"\nRunning {iterations} simulations...")

        for i in range(iterations):
            # Simulate the interaction
            interaction = simulate_interaction_pair(
                gene_a.bmr_pmf, gene_b.bmr_pmf, tau_01, tau_10, tau_11, nsamples
            )

            # Estimate tau parameters using the model
            try:
                interaction.estimate_tau_with_em_from_scratch()
            except Exception as e:
                logging.warning(f"Simulation {i+1} failed: {e}")
                continue

            # Extract estimated taus
            estimated_taus = (
                interaction.tau_00,
                interaction.tau_01,
                interaction.tau_10,
                interaction.tau_11,
            )

            # Calculate recovery error and false positives
            true_taus = (tau_00, tau_01, tau_10, tau_11)
            total_recovery_error += calculate_tau_recovery_error(
                true_taus, estimated_taus
            )
            total_false_positives += count_false_positives(
                true_taus, estimated_taus, threshold
            )

        # Compute averages
        average_recovery_error = total_recovery_error / iterations
        average_false_positive_rate = total_false_positives / (
            iterations * len(true_taus)
        )

        # Print results
        print("\n=== Tau Recovery Analysis Results ===")
        print(f"True Tau Values: {true_taus}")
        print(f"Average Tau Recovery Error: {average_recovery_error:.6f}")
        print(
            f"False Positive Rate (Non-Zero Misestimation): {average_false_positive_rate:.6f}"
        )
        print("=====================================\n")

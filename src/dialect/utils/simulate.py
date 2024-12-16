import numpy as np
import pandas as pd

from itertools import combinations
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
# TODO: move to universal function and use across simulate/identify/generate
def load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
    return cnt_df, bmr_dict


def simulate_single_gene_passengers(bmr_pmf, nsamples):
    """
    Simulate passenger mutation counts for a single gene, ensuring that adding a driver mutation
    does not exceed the maximum count in the background mutation rate (BMR PMF).

    :param bmr_pmf (dict): Background mutation rate probability mass function (PMF).
    :param nsamples (int): Number of samples to simulate.
    :return np.ndarray: Passenger mutation counts for each sample.
    """
    if not np.isclose(sum(bmr_pmf.values()), 1.0):
        raise ValueError("Background mutation rates (bmr_pmf) must sum to 1.")

    # Exclude the last mutation count to leave space for driver mutation addition
    max_count = max(bmr_pmf.keys())
    adjusted_pmf = {k: v for k, v in bmr_pmf.items() if k < max_count}
    adjusted_pmf_sum = sum(adjusted_pmf.values())
    adjusted_pmf = {k: v / adjusted_pmf_sum for k, v in adjusted_pmf.items()}
    return np.random.choice(
        list(adjusted_pmf.keys()), nsamples, p=list(adjusted_pmf.values())
    )


def simulate_single_gene_drivers(nsamples, pi):
    """
    Simulate driver mutations for a single gene.

    :param nsamples (int): Number of samples to simulate.
    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :return np.ndarray: Driver mutation counts (0 or 1) for each sample.
    """
    if not (0 <= pi <= 1):
        raise ValueError("Driver mutation rate pi must be between 0 and 1.")
    return np.random.binomial(1, pi, size=nsamples)


def simulate_pairwise_gene_drivers(nsamples, tau_01, tau_10, tau_11):
    """
    Simulate driver mutations for an interacting gene pair.

    :param nsamples (int): Number of samples to simulate.
    :param tau_01 (float): Probability (D=0, D'=1).
    :param tau_10 (float): Probability (D=1, D'=0).
    :param tau_11 (float): Probability (D=1, D'=1).
    :return tuple: Driver mutation counts for gene_a and gene_b.
    """
    gene_a_drivers = np.zeros(nsamples)
    gene_b_drivers = np.zeros(nsamples)
    rnd = np.random.uniform(size=nsamples)

    both_mutations = rnd < tau_11
    only_gene_b_mutations = (rnd >= tau_11) & (rnd < tau_11 + tau_01)
    only_gene_a_mutations = (rnd >= tau_11 + tau_01) & (rnd < tau_11 + tau_01 + tau_10)

    gene_a_drivers[both_mutations | only_gene_a_mutations] = 1
    gene_b_drivers[both_mutations | only_gene_b_mutations] = 1

    return gene_a_drivers, gene_b_drivers


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def simulate_single_gene(bmr_pmf, nsamples, pi):
    """
    Simulate somatic mutation counts for a single gene.

    The somatic mutation count is the sum of passenger mutations (from the BMR PMF)
    and driver mutations (simulated using a Bernoulli process with rate pi).

    :param bmr_pmf (list): Background mutation rate probability mass function (PMF).
    :param nsamples (int): Number of samples to simulate.
    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :return np.ndarray: Simulated somatic mutation counts for each sample.
    """
    passenger_counts = simulate_single_gene_passengers(bmr_pmf, nsamples)
    driver_counts = simulate_single_gene_drivers(nsamples, pi)
    somatic_mutations = (passenger_counts + driver_counts).astype(int)
    simulated_gene = Gene(
        name="SimulatedGene",
        samples=range(nsamples),
        counts=somatic_mutations,
        bmr_pmf=bmr_pmf,
    )
    return simulated_gene


def simulate_interaction_pair(gene_a_pmf, gene_b_pmf, tau_01, tau_10, tau_11, nsamples):
    """
    Simulate an interaction between two genes.

    :param gene_a_pmf (list): Background mutation rate PMF for gene A.
    :param gene_b_pmf (list): Background mutation rate PMF for gene B.
    :param tau_01 (float): Probability of (D=0, D'=1).
    :param tau_10 (float): Probability of (D=1, D'=0).
    :param tau_11 (float): Probability of (D=1, D'=1).
    :param nsamples (int): Number of samples to simulate.
    :return tuple: Simulated Gene objects for gene A and gene B.
    """
    gene_a_passengers = simulate_single_gene_passengers(gene_a_pmf, nsamples)
    gene_b_passengers = simulate_single_gene_passengers(gene_b_pmf, nsamples)

    gene_a_drivers, gene_b_drivers = simulate_pairwise_gene_drivers(
        nsamples, tau_01, tau_10, tau_11
    )

    gene_a_counts = (gene_a_passengers + gene_a_drivers).astype(int)
    gene_b_counts = (gene_b_passengers + gene_b_drivers).astype(int)

    simulated_gene_a = Gene(
        name="SimulatedGeneA",
        samples=range(nsamples),
        counts=gene_a_counts,
        bmr_pmf=gene_a_pmf,
    )
    simulated_gene_b = Gene(
        name="SimulatedGeneB",
        samples=range(nsamples),
        counts=gene_b_counts,
        bmr_pmf=gene_b_pmf,
    )

    simulated_interaction = Interaction(
        gene_a=simulated_gene_a, gene_b=simulated_gene_b
    )

    return simulated_interaction


# TODO: test usage
def simulate_mutation_matrix(
    cnt_mtx_filename,
    bmr_pmfs_filename,
    nsamples=1000,
    num_pairs=25,
    num_passengers=450,
    tau_range=(0.1, 0.2),
):
    """
    Simulate a mutation matrix with predefined characteristics.

    :param cnt_mtx_filename (str): Path to the count matrix CSV file.
    :param bmr_pmfs_filename (str): Path to the BMR PMFs CSV file.
    :param nsamples (int): Number of samples (rows in the mutation matrix).
    :param num_pairs (int): Number of mutually exclusive pairs to simulate.
    :param num_passengers (int): Number of passenger genes to simulate.
    :param tau_range (tuple): Range for tau values for mutually exclusive pairs.
    :return: tuple (mutation_matrix DataFrame, expected_results dict)
    """
    # Load data
    _, bmr_pmfs_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_filename, bmr_pmfs_filename)

    # Select representative high and low BMR genes
    lowest_bmr_gene = min(bmr_pmfs_dict, key=lambda k: sum(bmr_pmfs_dict[k]))
    highest_bmr_gene = max(bmr_pmfs_dict, key=lambda k: sum(bmr_pmfs_dict[k]))

    mutation_matrix = []
    gene_labels = []
    expected_results = {"positive": [], "negative": []}

    # Simulate mutually exclusive pairs
    for i in range(num_pairs):
        high_bmr_driver = f"HIGH_BMR_DRIVER_{i}"
        low_bmr_driver = f"LOW_BMR_DRIVER_{i}"
        bmr_pmfs_dict[high_bmr_driver] = bmr_pmfs_dict[highest_bmr_gene]
        bmr_pmfs_dict[low_bmr_driver] = bmr_pmfs_dict[lowest_bmr_gene]

        tau_10 = tau_01 = np.random.uniform(*tau_range)
        tau_11 = 0  # No co-occurrence for mutually exclusive pairs

        gene_a_mutations, gene_b_mutations = simulate_interaction_pair(
            bmr_pmfs_dict[high_bmr_driver],
            bmr_pmfs_dict[low_bmr_driver],
            tau_01,
            tau_10,
            tau_11,
            nsamples,
        )
        mutation_matrix.append(gene_a_mutations)
        mutation_matrix.append(gene_b_mutations)
        gene_labels.extend([high_bmr_driver, low_bmr_driver])
        expected_results["positive"].append((high_bmr_driver, low_bmr_driver))

    # Simulate passenger genes
    for i in range(num_passengers):
        passenger_gene = f"HIGH_BMR_PASSENGER_{i}"
        bmr_pmfs_dict[passenger_gene] = bmr_pmfs_dict[highest_bmr_gene]
        passenger_mutations = simulate_single_gene_passengers(
            bmr_pmfs_dict[passenger_gene], nsamples
        )
        mutation_matrix.append(passenger_mutations)
        gene_labels.append(passenger_gene)

    # Add negative pairs (all non-pairwise combinations that are not in positive pairs)
    all_genes = [gene for pair in expected_results["positive"] for gene in pair] + [
        f"HIGH_BMR_PASSENGER_{i}" for i in range(num_passengers)
    ]
    negative_pairs = list(
        set(combinations(all_genes, 2)) - set(expected_results["positive"])
    )
    expected_results["negative"] = negative_pairs

    mutation_matrix = np.array(mutation_matrix).T  # Convert to samples x genes
    mutation_matrix_df = pd.DataFrame(mutation_matrix, columns=gene_labels)

    return mutation_matrix_df, expected_results

import os
import json
import logging
import numpy as np
from scipy.stats import binom

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


# TODO: ADD SUPPORT FOR SINGLE GENE CREATE
# TODO: ADD SUPPORT FOR SINGLE GENE EVALUATE


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def generate_bmr_pmf(length, mu, threshold=1e-50):
    """
    Generate the binomial mutation rate (BMR) PMF for a given gene length and mutation rate.

    Parameters:
    - length: int, the length of the gene in nucleotides.
    - mu: float, the per nucleotide mutation rate.
    - threshold: float, the PMF value threshold for inclusion in the distribution.

    Returns:
    - dict: Keys are counts (0 to length) and values are the PMF for those counts, filtered by the threshold.
    """
    pmf_values = binom.pmf(range(length + 1), n=length, p=mu)
    bmr_pmf_arr = [pmf for pmf in pmf_values if pmf >= threshold]
    return bmr_pmf_arr


def simulate_single_gene_passenger_mutations(bmr_pmf, nsamples):
    """
    Simulate passenger mutation counts for a single gene.

    :param bmr_pmf (dict): Background mutation rate categorical distribution.
    :param nsamples (int): Number of samples to simulate.
    :return np.ndarray: Passenger mutation counts across samples.
    """
    if not np.isclose(sum(bmr_pmf.values()), 1.0, atol=1e-6):
        raise ValueError("Background mutation rates (bmr_pmf) must sum to 1.")

    # normalize bmr pmf values
    bmr_pmf = {k: v / sum(bmr_pmf.values()) for k, v in bmr_pmf.items()}
    return np.random.choice(
        list(bmr_pmf.keys()),
        nsamples,
        p=list(bmr_pmf.values()),
    )


def simulate_single_gene_driver_mutations(pi, nsamples):
    """
    Simulate driver mutations for a single gene.

    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :param nsamples (int): Number of samples to simulate.
    :return np.ndarray: Binary driver mutation counts across samples.
    """
    if not (0 <= pi <= 1):
        raise ValueError("Driver mutation rate pi must be between 0 and 1.")
    return np.random.binomial(1, pi, size=nsamples)


def simulate_single_gene_somatic_mutations(bmr_pmf_arr, pi, nsamples):
    """
    Simulate somatic mutation counts for a single gene.

    The somatic mutation count is the sum of passenger and driver mutations.

    :param bmr_pmf (list): Background mutation rate probability mass function.
    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :param nsamples (int): Number of samples to simulate.
    :return np.ndarray: Simulated somatic mutation counts for each sample.
    """
    bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
    passenger_mutations = simulate_single_gene_passenger_mutations(
        bmr_pmf,
        nsamples,
    )
    driver_mutations = simulate_single_gene_driver_mutations(pi, nsamples)
    somatic_mutations = (passenger_mutations + driver_mutations).astype(int)
    simulated_gene = Gene(
        name="SimulatedGene",
        samples=range(nsamples),
        counts=somatic_mutations,
        bmr_pmf=bmr_pmf_arr,
    )
    return simulated_gene


def simulate_pairwise_gene_driver_mutations(tau_01, tau_10, tau_11, nsamples):
    """
    Simulate driver mutations for a pair of genes.

    :param tau_01 (float): Probability (D=0, D'=1).
    :param tau_10 (float): Probability (D=1, D'=0).
    :param tau_11 (float): Probability (D=1, D'=1).
    :param nsamples (int): Number of samples to simulate.
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


def simulate_pairwise_gene_somatic_mutations(
    gene_a_pmf,
    gene_b_pmf,
    tau_01,
    tau_10,
    tau_11,
    nsamples,
):
    """
    Simulate somatic mutations between two genes.

    :param gene_a_pmf (dict): Background mutation rate PMF for gene A.
    :param gene_b_pmf (dict): Background mutation rate PMF for gene B.
    :param tau_01 (float): Probability of (D=0, D'=1).
    :param tau_10 (float): Probability of (D=1, D'=0).
    :param tau_11 (float): Probability of (D=1, D'=1).
    :param nsamples (int): Number of samples to simulate.
    :return tuple: Simulated Gene objects for gene A and gene B.
    """
    gene_a_passenger_mutations = simulate_single_gene_passenger_mutations(
        gene_a_pmf, nsamples
    )
    gene_b_passenger_mutations = simulate_single_gene_passenger_mutations(
        gene_b_pmf, nsamples
    )
    gene_a_driver_mutations, gene_b_driver_mutations = (
        simulate_pairwise_gene_driver_mutations(tau_01, tau_10, tau_11, nsamples)
    )
    gene_a_somatic_mutations = (
        gene_a_passenger_mutations + gene_a_driver_mutations
    ).astype(int)
    gene_b_somatic_mutations = (
        gene_b_passenger_mutations + gene_b_driver_mutations
    ).astype(int)

    simulated_gene_a = Gene(
        name="SimulatedGeneA",
        samples=range(nsamples),
        counts=gene_a_somatic_mutations,
        bmr_pmf=gene_a_pmf,
    )
    simulated_gene_b = Gene(
        name="SimulatedGeneB",
        samples=range(nsamples),
        counts=gene_b_somatic_mutations,
        bmr_pmf=gene_b_pmf,
    )

    simulated_interaction = Interaction(
        gene_a=simulated_gene_a, gene_b=simulated_gene_b
    )

    return simulated_interaction


# ------------------------------------------------------------------------------------ #
#                               SIMULATE CREATE FUNCTIONS                              #
# ------------------------------------------------------------------------------------ #


# ------------------------------------ SINGLE GENE ----------------------------------- #
def create_single_gene_simulation(
    pi,
    num_samples,
    num_simulations,
    length,
    mu,
    out,
    seed,
):
    """
    Create a single gene simulation.

    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :param num_samples (int): Number of samples to simulate.
    :param num_simulations (int): Number of simulations to run.
    :param bmr (str): Path to the BMR file.
    :param gene (str): Gene name.
    :param out (str): Output path for the simulation results.
    :param seed (int): Random seed for reproducibility.
    """
    logging.info("Creating single gene simulation")
    np.random.seed(seed)
    os.makedirs(out, exist_ok=True)

    logging.info(
        f"Simulating {num_simulations} single gene somatic mutations with "
        f"{num_samples} samples and driver mutation rate pi={pi}"
    )

    bmr_pmf_arr = generate_bmr_pmf(length, mu)
    simulated_genes = []
    for _ in range(num_simulations):
        simulated_gene = simulate_single_gene_somatic_mutations(
            bmr_pmf_arr,
            pi,
            num_samples,
        )
        simulated_genes.append(simulated_gene.counts)

    counts_array = np.array(simulated_genes)
    np.save(os.path.join(out, "single_gene_simulated_data.npy"), counts_array)
    logging.info(
        f"Saved single gene simulation data to {out}/single_gene_simulated_data.npy"
    )

    params = {
        "pi": pi,
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "seed": seed,
        "length": length,
        "mu": mu,
        "bmr_pmf": bmr_pmf_arr,
    }
    with open(os.path.join(out, "single_gene_simulation_parameters.json"), "w") as f:
        json.dump(params, f, indent=4)
    logging.info(
        f"Saved simulation parameters to {out}/single_gene_simulation_parameters.json"
    )


# ----------------------------------- PAIR OF GENES ---------------------------------- #


# -------------------------------------- MATRIX -------------------------------------- #


# ------------------------------------------------------------------------------------ #
#                              SIMULATE EVALUATE FUNCTIONS                             #
# ------------------------------------------------------------------------------------ #


# ------------------------------------ SINGLE GENE ----------------------------------- #
def evaluate_single_gene_simulation(
    params,
    data,
    out,
):
    """
    Evaluate the single gene simulation.

    :param params_path (str): Path to the JSON file containing simulation parameters.
    :param data_path (str): Path to the .npy file containing simulation data.
    :param out (str): Output path for evaluation results.
    """
    logging.info("Evaluating DIALECT on simulated data for a single gene")
    os.makedirs(out, exist_ok=True)

    try:
        with open(params, "r") as f:
            params = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read parameters file: {params}")
        raise e

    pi = params.get("pi")
    num_samples = params.get("num_samples")
    bmr_pmf_arr = params.get("bmr_pmf")
    bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
    logging.info(f"Loaded parameters: {params}")

    try:
        data = np.load(data)
    except Exception as e:
        logging.error(f"Failed to read simulation data file: {data}")
        raise e
    logging.info(f"Loaded simulation data from {data}. Shape: {data.shape}")

    est_pi_vals = []
    for i, row in enumerate(data):
        simulated_gene = Gene(
            name=f"SimulatedGene_{i}",
            samples=range(num_samples),
            counts=row,
            bmr_pmf=bmr_pmf,
        )
        simulated_gene.estimate_pi_with_em_from_scratch()
        est_pi_vals.append(simulated_gene.pi)

    est_pi_fout = os.path.join(out, "estimated_pi_values.npy")
    np.save(est_pi_fout, np.array(est_pi_vals))
    deviations = [abs(est - pi) for est in est_pi_vals]
    results = {
        "true_pi": pi,
        "mean_estimated_pi": np.mean(est_pi_vals),
        "std_estimated_pi": np.std(est_pi_vals),
        "mean_deviation": np.mean(deviations),
        "std_deviation": np.std(deviations),
    }
    results_path = os.path.join(out, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


# ----------------------------------- PAIR OF GENES ---------------------------------- #


# -------------------------------------- MATRIX -------------------------------------- #
# def simulate_mutation_matrix(
#     cnt_mtx_filename,
#     bmr_pmfs_filename,
#     nsamples=1000,
#     num_pairs=25,
#     num_passengers=450,
#     tau_range=(0.1, 0.2),
# ):
#     """
#     Simulate a mutation matrix with predefined characteristics.

#     :param cnt_mtx_filename (str): Path to the count matrix CSV file.
#     :param bmr_pmfs_filename (str): Path to the BMR PMFs CSV file.
#     :param nsamples (int): Number of samples (rows in the mutation matrix).
#     :param num_pairs (int): Number of mutually exclusive pairs to simulate.
#     :param num_passengers (int): Number of passenger genes to simulate.
#     :param tau_range (tuple): Range for tau values for mutually exclusive pairs.
#     :return: tuple (mutation_matrix DataFrame, expected_results dict)
#     """
#     # Load data
#     _, bmr_pmfs_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_filename, bmr_pmfs_filename)

#     # Select representative high and low BMR genes
#     lowest_bmr_gene = min(bmr_pmfs_dict, key=lambda k: sum(bmr_pmfs_dict[k]))
#     highest_bmr_gene = max(bmr_pmfs_dict, key=lambda k: sum(bmr_pmfs_dict[k]))

#     mutation_matrix = []
#     gene_labels = []
#     expected_results = {"positive": [], "negative": []}

#     # Simulate mutually exclusive pairs
#     for i in range(num_pairs):
#         high_bmr_driver = f"HIGH_BMR_DRIVER_{i}"
#         low_bmr_driver = f"LOW_BMR_DRIVER_{i}"
#         bmr_pmfs_dict[high_bmr_driver] = bmr_pmfs_dict[highest_bmr_gene]
#         bmr_pmfs_dict[low_bmr_driver] = bmr_pmfs_dict[lowest_bmr_gene]

#         tau_10 = tau_01 = np.random.uniform(*tau_range)
#         tau_11 = 0  # No co-occurrence for mutually exclusive pairs

#         gene_a_mutations, gene_b_mutations = simulate_interaction_pair(
#             bmr_pmfs_dict[high_bmr_driver],
#             bmr_pmfs_dict[low_bmr_driver],
#             tau_01,
#             tau_10,
#             tau_11,
#             nsamples,
#         )
#         mutation_matrix.append(gene_a_mutations)
#         mutation_matrix.append(gene_b_mutations)
#         gene_labels.extend([high_bmr_driver, low_bmr_driver])
#         expected_results["positive"].append((high_bmr_driver, low_bmr_driver))

#     # Simulate passenger genes
#     for i in range(num_passengers):
#         passenger_gene = f"HIGH_BMR_PASSENGER_{i}"
#         bmr_pmfs_dict[passenger_gene] = bmr_pmfs_dict[highest_bmr_gene]
#         passenger_mutations = simulate_single_gene_passengers(
#             bmr_pmfs_dict[passenger_gene], nsamples
#         )
#         mutation_matrix.append(passenger_mutations)
#         gene_labels.append(passenger_gene)

#     # Add negative pairs (all non-pairwise combinations that are not in positive pairs)
#     all_genes = [gene for pair in expected_results["positive"] for gene in pair] + [
#         f"HIGH_BMR_PASSENGER_{i}" for i in range(num_passengers)
#     ]
#     negative_pairs = list(
#         set(combinations(all_genes, 2)) - set(expected_results["positive"])
#     )
#     expected_results["negative"] = negative_pairs

#     mutation_matrix = np.array(mutation_matrix).T  # Convert to samples x genes
#     mutation_matrix_df = pd.DataFrame(mutation_matrix, columns=gene_labels)

#     return mutation_matrix_df, expected_results
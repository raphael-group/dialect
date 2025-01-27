import os
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import binom

from dialect.utils.helpers import load_cnt_mtx_and_bmr_pmfs
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.postprocessing import compute_epsilon_threshold
from dialect.utils.plotting import plot_mtx_sim_pr_curve


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
        samples=[f"S{i}" for i in range(nsamples)],
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
        samples=[f"S{i}" for i in range(nsamples)],
        counts=gene_a_somatic_mutations,
        bmr_pmf=gene_a_pmf,
    )
    simulated_gene_b = Gene(
        name="SimulatedGeneB",
        samples=[f"S{i}" for i in range(nsamples)],
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
def create_pair_gene_simulation(
    tau_10,
    tau_01,
    tau_11,
    num_samples,
    num_simulations,
    length_a,
    mu_a,
    length_b,
    mu_b,
    out,
    seed,
):
    np.random.seed(seed)
    bmr_pmf_arr_a = generate_bmr_pmf(length_a, mu_a)
    bmr_pmf_arr_b = generate_bmr_pmf(length_b, mu_b)
    bmr_pmf_a = {i: bmr_pmf_arr_a[i] for i in range(len(bmr_pmf_arr_a))}
    bmr_pmf_b = {i: bmr_pmf_arr_b[i] for i in range(len(bmr_pmf_arr_b))}
    simulated_interactions = []
    for _ in range(num_simulations):
        simulated_interaction = simulate_pairwise_gene_somatic_mutations(
            bmr_pmf_a,
            bmr_pmf_b,
            tau_01,
            tau_10,
            tau_11,
            num_samples,
        )
        simulated_interactions.append(simulated_interaction)

    counts_array = np.array(
        [
            np.stack((interaction.gene_a.counts, interaction.gene_b.counts), axis=-1)
            for interaction in simulated_interactions
        ]
    )
    np.save(os.path.join(out, "pair_gene_simulated_data.npy"), counts_array)

    params = {
        "tau_10": tau_10,
        "tau_01": tau_01,
        "tau_11": tau_11,
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "length_a": length_a,
        "mu_a": mu_a,
        "length_b": length_b,
        "mu_b": mu_b,
        "bmr_pmf_a": bmr_pmf_arr_a,
        "bmr_pmf_b": bmr_pmf_arr_b,
    }
    with open(os.path.join(out, "pair_gene_simulation_parameters.json"), "w") as f:
        json.dump(params, f, indent=4)


# -------------------------------------- MATRIX -------------------------------------- #
def create_matrix_simulation(
    cnt_mtx_fn,
    bmr_pmfs_fn,
    driver_genes_fn,
    dout,
    num_likely_passengers,
    num_me_pairs,
    num_co_pairs,
    num_samples,
    ixn_strength,
    seed=42,
):
    np.random.seed(seed)
    os.makedirs(dout, exist_ok=True)

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_fn, bmr_pmfs_fn)

    drivers_arr = pd.read_csv(driver_genes_fn, sep="\t", index_col=0).index
    drivers_set = set(drivers_arr + "_M") | set(drivers_arr + "_N")

    likely_passengers_df = cnt_df.drop(drivers_set, axis=1, errors="ignore")
    likely_passengers = list(
        likely_passengers_df.sum(axis=0)
        .sort_values(ascending=False)
        .head(num_likely_passengers)
        .index
    )

    drivers_df = cnt_df[[col for col in drivers_set if col in cnt_df.columns]]
    drivers = list(
        drivers_df.sum(axis=0)
        .sort_values(ascending=False)
        .head(2 * (num_me_pairs + num_co_pairs))
        .index
    )
    np.random.shuffle(drivers)

    me_pairs = [(drivers[i], drivers[i + 1]) for i in range(0, 2 * num_me_pairs, 2)]
    co_pairs = [
        (drivers[i], drivers[i + 1])
        for i in range(2 * num_me_pairs, 2 * (num_me_pairs + num_co_pairs), 2)
    ]

    simulated_counts = {}

    def arr_to_dict(pmf_array):
        return {i: pmf_array[i] for i in range(len(pmf_array))}

    for gene_a, gene_b in me_pairs:
        bmr_pmf_a = arr_to_dict(bmr_dict[gene_a])
        bmr_pmf_b = arr_to_dict(bmr_dict[gene_b])

        tau_01 = ixn_strength
        tau_10 = ixn_strength
        tau_11 = 0.0

        interaction = simulate_pairwise_gene_somatic_mutations(
            gene_a_pmf=bmr_pmf_a,
            gene_b_pmf=bmr_pmf_b,
            tau_01=tau_01,
            tau_10=tau_10,
            tau_11=tau_11,
            nsamples=num_samples,
        )
        simulated_counts[gene_a] = interaction.gene_a.counts
        simulated_counts[gene_b] = interaction.gene_b.counts

    for gene_a, gene_b in co_pairs:
        bmr_pmf_a = arr_to_dict(bmr_dict[gene_a])
        bmr_pmf_b = arr_to_dict(bmr_dict[gene_b])

        tau_11 = ixn_strength
        tau_01 = 0.0
        tau_10 = 0.0

        interaction = simulate_pairwise_gene_somatic_mutations(
            gene_a_pmf=bmr_pmf_a,
            gene_b_pmf=bmr_pmf_b,
            tau_01=tau_01,
            tau_10=tau_10,
            tau_11=tau_11,
            nsamples=num_samples,
        )
        simulated_counts[gene_a] = interaction.gene_a.counts
        simulated_counts[gene_b] = interaction.gene_b.counts

    for likely_passenger in likely_passengers:
        simulated_gene = simulate_single_gene_somatic_mutations(
            bmr_pmf_arr=bmr_dict[likely_passenger],
            pi=0.0,
            nsamples=num_samples,
        )
        simulated_counts[likely_passenger] = simulated_gene.counts

    all_genes_order = drivers + likely_passengers

    final_array = []
    for g in all_genes_order:
        final_array.append(simulated_counts[g])
    final_array = np.array(final_array).T

    sim_df = pd.DataFrame(
        final_array,
        columns=all_genes_order,
        index=[f"S{i}" for i in range(num_samples)],
    )
    matrix_out_fn = os.path.join(dout, "count_matrix.csv")
    sim_df.index.name = "sample"
    sim_df.to_csv(matrix_out_fn, index=True)
    logging.info(f"Saved simulated matrix to {matrix_out_fn}")

    info = {
        "ME Pairs": me_pairs,
        "CO Pairs": co_pairs,
        "Likely Passengers": likely_passengers,
        "num_samples": num_samples,
        "ME tau_01, tau_10": ixn_strength,
        "CO tau_11": ixn_strength,
    }
    gt_out_fn = os.path.join(dout, "matrix_simulation_info.json")
    with open(gt_out_fn, "w") as f:
        json.dump(info, f, indent=4)
    logging.info(f"Saved ground truth interactions to {gt_out_fn}")
    logging.info("Matrix simulation completed.")


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
            samples=[f"S{i}" for i in range(num_samples)],
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
def evaluate_pair_gene_simulation(
    params,
    data,
    out,
):
    """
    Evaluate DIALECT on simulated data for a pair of genes.

    1. Loads the simulation parameters (true tau values).
    2. Loads the simulated data (pair_gene_simulated_data.npy).
    3. Builds Interaction objects for each simulation and runs
       estimate_tau_with_em_from_scratch().
    4. Compares the estimated tau values with the ground truth.
    5. Saves evaluation statistics to output JSON.

    :param params_path: str, path to the JSON file containing simulation parameters.
    :param data_path: str, path to the .npy file containing simulated pair data.
    :param out: str, path to the directory where results (JSON) will be saved.
    """
    logging.info("Evaluating DIALECT on simulated data for a pair of genes")
    os.makedirs(out, exist_ok=True)

    try:
        with open(params, "r") as f:
            params = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read parameters file: {params}")
        raise e

    true_tau_10 = params.get("tau_10")
    true_tau_01 = params.get("tau_01")
    true_tau_11 = params.get("tau_11")
    num_samples = params.get("num_samples")
    num_simulations = params.get("num_simulations")

    bmr_pmf_arr_a = params.get("bmr_pmf_a")
    bmr_pmf_arr_b = params.get("bmr_pmf_b")
    bmr_pmf_a = {i: bmr_pmf_arr_a[i] for i in range(len(bmr_pmf_arr_a))}
    bmr_pmf_b = {i: bmr_pmf_arr_b[i] for i in range(len(bmr_pmf_arr_b))}

    logging.info(f"Loaded parameters:\n{json.dumps(params, indent=4)}")

    data = np.load(data)

    est_tau_00_vals = []
    est_tau_01_vals = []
    est_tau_10_vals = []
    est_tau_11_vals = []

    for i in range(num_simulations):
        gene_a_counts = data[i, :, 0]
        gene_b_counts = data[i, :, 1]

        gene_a = Gene(
            name=f"SimulatedGeneA_{i}",
            samples=[f"S{i}" for i in range(num_samples)],
            counts=gene_a_counts,
            bmr_pmf=bmr_pmf_a,
        )
        gene_b = Gene(
            name=f"SimulatedGeneB_{i}",
            samples=[f"S{i}" for i in range(num_samples)],
            counts=gene_b_counts,
            bmr_pmf=bmr_pmf_b,
        )

        interaction = Interaction(gene_a=gene_a, gene_b=gene_b)
        interaction.estimate_tau_with_em_from_scratch()

        est_tau_00_vals.append(interaction.tau_00)
        est_tau_01_vals.append(interaction.tau_01)
        est_tau_10_vals.append(interaction.tau_10)
        est_tau_11_vals.append(interaction.tau_11)

    dev_01 = [abs(est - true_tau_01) for est in est_tau_01_vals]
    dev_10 = [abs(est - true_tau_10) for est in est_tau_10_vals]
    dev_11 = [abs(est - true_tau_11) for est in est_tau_11_vals]

    results = {
        "true_tau_01": true_tau_01,
        "true_tau_10": true_tau_10,
        "true_tau_11": true_tau_11,
        "mean_est_tau_01": float(np.mean(est_tau_01_vals)),
        "std_est_tau_01": float(np.std(est_tau_01_vals)),
        "mean_dev_01": float(np.mean(dev_01)),
        "std_dev_01": float(np.std(dev_01)),
        "mean_est_tau_10": float(np.mean(est_tau_10_vals)),
        "std_est_tau_10": float(np.std(est_tau_10_vals)),
        "mean_dev_10": float(np.mean(dev_10)),
        "std_dev_10": float(np.std(dev_10)),
        "mean_est_tau_11": float(np.mean(est_tau_11_vals)),
        "std_est_tau_11": float(np.std(est_tau_11_vals)),
        "mean_dev_11": float(np.mean(dev_11)),
        "std_dev_11": float(np.std(dev_11)),
    }

    results_path = os.path.join(out, "pair_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Saved pair evaluation results to {results_path}")


# -------------------------------------- MATRIX -------------------------------------- #
def get_ground_truth_labels(df, simulation_info, ixn_type):
    true_ixn_pairs = set()
    key = f"{ixn_type} Pairs"  # "ME_pairs" or "CO_pairs"
    for g1, g2 in simulation_info.get(key, []):
        true_ixn_pairs.add((g1, g2))
        true_ixn_pairs.add((g2, g1))

    labels = []
    for _, row in df.iterrows():
        pair = (row["Gene A"], row["Gene B"])
        labels.append(1 if pair in true_ixn_pairs else 0)

    return np.array(labels)


def get_method_scores(df, num_samples, ixn_type):
    dialect_rho = df["Rho"].astype(float).values
    tau_1X = df["Tau_1X"].astype(float).values
    tau_X1 = df["Tau_X1"].astype(float).values
    epsilon = compute_epsilon_threshold(num_samples)
    mask_no_interaction = (tau_1X < epsilon) | (tau_X1 < epsilon)
    dialect_rho[mask_no_interaction] = 0.0

    dialect_rho = -dialect_rho if ixn_type == "ME" else dialect_rho

    if ixn_type == "ME":
        fishers_pval = df["Fisher's ME P-Val"].astype(float).values
        discover_pval = df["Discover ME P-Val"].astype(float).values
        megsa_s = df["MEGSA S-Score (LRT)"].astype(float).values
        wesme_pval = df["WeSME P-Val"].astype(float).values

        fishers_score = -np.log10(fishers_pval + 1e-300)
        discover_score = -np.log10(discover_pval + 1e-300)
        wesme_score = -np.log10(wesme_pval + 1e-300)

        methods = {
            "DIALECT": dialect_rho,
            "Fisher's Exact Test": fishers_score,
            "DISCOVER": discover_score,
            "MEGSA": megsa_s,
            "WeSME": wesme_score,
        }

    else:
        fishers_pval = df["Fisher's CO P-Val"].astype(float).values
        discover_pval = df["Discover CO P-Val"].astype(float).values
        wesco_pval = df["WeSCO P-Val"].astype(float).values

        fishers_score = -np.log10(fishers_pval + 1e-300)
        discover_score = -np.log10(discover_pval + 1e-300)
        wesco_score = -np.log10(wesco_pval + 1e-300)

        methods = {
            "DIALECT": dialect_rho,
            "Fisher's Exact Test": fishers_score,
            "DISCOVER": discover_score,
            "WeSCO": wesco_score,
        }

    return methods


def evaluate_matrix_simulation(
    results_fn,
    simulation_info_fn,
    dout,
    ixn_type,
):
    df = pd.read_csv(results_fn)
    with open(simulation_info_fn, "r") as f:
        gt = json.load(f)
    num_samples = gt.get("num_samples")

    y_true = get_ground_truth_labels(df, gt, ixn_type)
    methods = get_method_scores(df, num_samples, ixn_type)
    fout = os.path.join(dout, f"{ixn_type}_pr_curve.png")

    plot_mtx_sim_pr_curve(
        methods,
        y_true,
        fout,
        ixn_type,
    )

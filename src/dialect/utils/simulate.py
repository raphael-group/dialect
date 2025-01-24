import os
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import binom

from dialect.utils.helpers import load_cnt_mtx_and_bmr_pmfs
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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
    cnt_mtx_filename,
    driver_genes_filename,
    decoy_genes_filename,
    bmr_pmfs_filename,
    out,
    num_samples,
    num_me_pairs,
    num_co_pairs,
    decoy_gene_count,
    seed=42,
):
    """
    Creates a single simulated mutation matrix with the following steps:

    1. Read files:
       - cnt_mtx_filename: real data count matrix (samples x genes) or (genes x samples).
         We'll assume columns=genes, rows=samples in this example.
       - driver_genes_filename: list of known driver genes (text file, one gene per line).
       - decoy_genes_filename: list of 'decoy' genes (sorted by highest mutation frequency).
       - bmr_pmfs_filename: JSON or other file with a dict: { gene_name: [bmr_pmf_array], ... }

    2. From the driver genes, pick the top 2*(num_me_pairs + num_co_pairs)
       based on total mutation counts in the real count matrix.
       Randomly shuffle them and assign the first 'num_me_pairs' pairs as ME,
       and the next 'num_co_pairs' pairs as CO.

    3. For each pair (A,B):
       - If ME, sample tau_01, tau_10, set tau_11 ~ 0 or near 0
       - If CO, sample tau_11 to be larger, etc.
       - Simulate pairwise somatic counts using your code:
         simulate_pairwise_gene_somatic_mutations(...)

    4. Read 'decoy_gene_count' decoy genes from the decoy_genes_filename,
       each simulated as a single gene with pi=0.

    5. Combine everything into a final matrix (num_samples x total_genes).
       Return (or save) the matrix as well as the “ground-truth” pairs.
    """

    np.random.seed(seed)
    os.makedirs(out, exist_ok=True)
    driver_gene_symbols = pd.read_csv(
        driver_genes_filename, sep="\t", index_col=0
    ).index.tolist()
    driver_genes = set(
        [driver + "_M" for driver in driver_gene_symbols]
        + [driver + "_N" for driver in driver_gene_symbols]
    )
    with open(decoy_genes_filename, "r") as f:
        decoy_genes = [line.strip() for line in f if line.strip()]

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_filename, bmr_pmfs_filename)

    driver_counts = []
    for gene in driver_genes:
        if gene in cnt_df.columns:
            total_cnt = cnt_df[gene].sum()
            driver_counts.append((gene, total_cnt))
    driver_counts.sort(key=lambda x: x[1], reverse=True)

    needed_drivers = 2 * (num_me_pairs + num_co_pairs)
    top_drivers = [x[0] for x in driver_counts[:needed_drivers]]
    np.random.shuffle(top_drivers)

    me_driver_pairs = []
    co_driver_pairs = []

    idx = 0
    for _ in range(num_me_pairs):
        pair = (top_drivers[idx], top_drivers[idx + 1])
        me_driver_pairs.append(pair)
        idx += 2
    for _ in range(num_co_pairs):
        pair = (top_drivers[idx], top_drivers[idx + 1])
        co_driver_pairs.append(pair)
        idx += 2

    simulated_counts = {}

    def arr_to_dict(pmf_array):
        return {i: pmf_array[i] for i in range(len(pmf_array))}

    for gene_a, gene_b in me_driver_pairs:
        bmr_pmf_a = arr_to_dict(bmr_dict[gene_a])
        bmr_pmf_b = arr_to_dict(bmr_dict[gene_b])

        tau_01 = np.random.uniform(0.1, 0.2)
        tau_10 = np.random.uniform(0.1, 0.2)
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

    for gene_a, gene_b in co_driver_pairs:
        bmr_pmf_a = arr_to_dict(bmr_dict[gene_a])
        bmr_pmf_b = arr_to_dict(bmr_dict[gene_b])

        tau_11 = np.random.uniform(0.1, 0.2)
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

    chosen_decoys = decoy_genes[:decoy_gene_count]
    for decoy_gene in chosen_decoys:
        bmr_decoy_arr = bmr_dict[decoy_gene]
        simulated_gene = simulate_single_gene_somatic_mutations(
            bmr_pmf_arr=bmr_decoy_arr,
            pi=0.0,
            nsamples=num_samples,
        )
        simulated_counts[decoy_gene] = simulated_gene.counts

    me_genes = [g for pair in me_driver_pairs for g in pair]
    co_genes = [g for pair in co_driver_pairs for g in pair]
    used_driver_genes = me_genes + co_genes
    all_genes_order = used_driver_genes + chosen_decoys

    final_array = []
    for g in all_genes_order:
        final_array.append(simulated_counts[g])
    final_array = np.array(final_array).T

    sim_df = pd.DataFrame(
        final_array,
        columns=all_genes_order,
        index=[f"S{i}" for i in range(num_samples)],
    )
    matrix_out_fn = os.path.join(out, "simulated_matrix.csv")
    sim_df.index.name = "sample"
    sim_df.to_csv(matrix_out_fn, index=True)
    logging.info(f"Saved simulated matrix to {matrix_out_fn}")

    ground_truth = {
        "ME_pairs": me_driver_pairs,
        "CO_pairs": co_driver_pairs,
        "all_genes_order": all_genes_order,
        "num_samples": num_samples,
    }
    gt_out_fn = os.path.join(out, "ground_truth_interactions.json")
    with open(gt_out_fn, "w") as f:
        json.dump(ground_truth, f, indent=4)
    logging.info(f"Saved ground truth interactions to {gt_out_fn}")
    logging.info("Matrix simulation completed.")
    return sim_df, ground_truth

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


def evaluate_matrix_simulation(
    merged_results_file,
    ground_truth_file,
    out_png,
):
    """
    Reads a merged results file containing pairwise interaction metrics,
    as well as a ground-truth JSON listing which pairs are true ME vs. not.
    Produces a Precision-Recall curve for each method, labeling true ME pairs
    as positives and everything else as negatives.

    :param merged_results_file: str, path to CSV with columns:
        - GeneA, GeneB
        - Rho (DIALECT)
        - Fisher's ME P-Val
        - DISCOVER ME P-Val
        - MEGSA S-Score
        - WeSME P-Val
        - Tau_1X
        - Tau_X1
    :param ground_truth_file: str, path to JSON with keys like "ME_pairs" (list of pairs)
                              and possibly "CO_pairs", etc.
    :param out_png: str, path to output .png file containing the PR curves plot.
    """

    df = pd.read_csv(merged_results_file)
    with open(ground_truth_file, "r") as f:
        gt = json.load(f)

    true_me_pairs = set()
    for g1, g2 in gt.get("ME_pairs", []):
        true_me_pairs.add((g1, g2))
        true_me_pairs.add((g2, g1))

    y_true = []
    for i, row in df.iterrows():
        pair = (row["Gene A"], row["Gene B"])
        if pair in true_me_pairs:
            y_true.append(1)
        else:
            y_true.append(0)
    y_true = np.array(y_true)

    dialect_rho = df["Rho"].values.astype(float)
    dialect_rho = -dialect_rho
    tau_1X = df["Tau_1X"].values.astype(float)
    tau_X1 = df["Tau_X1"].values.astype(float)
    mask_no_interaction = (tau_1X < 0.01) | (tau_X1 < 0.01)
    dialect_rho[mask_no_interaction] = 0.0

    fishers_pval = df["Fisher's CO P-Val"].values.astype(float)
    fishers_score = -np.log10(fishers_pval + 1e-300)  # small offset to avoid log(0)

    discover_pval = df["Discover ME P-Val"].values.astype(float)
    discover_score = -np.log10(discover_pval + 1e-300)

    megsa_s = df["MEGSA S-Score (LRT)"].values.astype(float)

    wesme_pval = df["WeSME P-Val"].values.astype(float)
    wesme_score = -np.log10(wesme_pval + 1e-300)

    methods = {
        "DIALECT": dialect_rho,
        "Fisher's Exact Test": fishers_score,
        "DISCOVER": discover_score,
        "MEGSA": megsa_s,
        "WeSME": wesme_score,
    }

    plt.figure(figsize=(7, 6))
    for method_name, scores in methods.items():
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{method_name} (AP={ap:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall for ME Identification", fontsize=14)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved PR curve plot to {out_png}")

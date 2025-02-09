"""TODO: Add docstring."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, truncnorm
from sklearn.metrics import average_precision_score

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.helpers import load_cnt_mtx_and_bmr_pmfs
from dialect.utils.plotting import (
    draw_auc_vs_factor_curve,
    draw_average_simulation_precision_recall_curve,
    draw_concat_simulation_precision_recall_curve,
    draw_hit_curve,
)
from dialect.utils.postprocessing import (
    compute_epsilon_threshold,
    generate_top_ranked_co_interaction_tables,
    generate_top_ranked_me_interaction_tables,
)

ME_METHODS = [
    "DIALECT (Rho)",
    "DISCOVER",
    "Fisher's Exact Test",
    "WeSME",
    "MEGSA",
]

CO_METHODS = [
    "DIALECT (LRT)",
    "DISCOVER",
    "Fisher's Exact Test",
    "WeSCO",
]


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def generate_bmr_pmf(length: int, mu: float, threshold: float = 1e-50) -> list:
    """TODO: Add docstring."""
    pmf_values = binom.pmf(range(length + 1), n=length, p=mu)
    return [pmf for pmf in pmf_values if pmf >= threshold]


def simulate_single_gene_passenger_mutations(
    bmr_pmf: dict,
    nsamples: int,
) -> np.ndarray:
    """TODO: Add docstring."""
    if not np.isclose(sum(bmr_pmf.values()), 1.0, atol=1e-6):
        msg = "Background mutation rates (bmr_pmf) must sum to 1."
        raise ValueError(msg)

    bmr_pmf = {k: v / sum(bmr_pmf.values()) for k, v in bmr_pmf.items()}
    rng = np.random.default_rng()
    return rng.choice(
        list(bmr_pmf.keys()),
        nsamples,
        p=list(bmr_pmf.values()),
    )


def simulate_single_gene_driver_mutations(pi: float, nsamples: int) -> np.ndarray:
    """TODO: Add docstring."""
    if not (0 <= pi <= 1):
        msg = "Driver mutation rate pi must be between 0 and 1."
        raise ValueError(msg)
    rng = np.random.default_rng()
    return rng.binomial(1, pi, size=nsamples)


def simulate_single_gene_somatic_mutations(
    bmr_pmf_arr: list,
    pi: float,
    nsamples: int,
) -> Gene:
    """TODO: Add docstring."""
    bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
    passenger_mutations = simulate_single_gene_passenger_mutations(
        bmr_pmf,
        nsamples,
    )
    driver_mutations = simulate_single_gene_driver_mutations(pi, nsamples)
    somatic_mutations = (passenger_mutations + driver_mutations).astype(int)
    return Gene(
        name="SimulatedGene",
        samples=[f"S{i}" for i in range(nsamples)],
        counts=somatic_mutations,
        bmr_pmf=bmr_pmf_arr,
    )


def simulate_pairwise_gene_driver_mutations(
    tau_01: float,
    tau_10: float,
    tau_11: float,
    nsamples: int,
    driver_proportion: float,
) -> tuple:
    """TODO: Add docstring."""
    gene_a_drivers = np.zeros(nsamples)
    gene_b_drivers = np.zeros(nsamples)
    rng = np.random.default_rng()
    n_drivers = int(nsamples * driver_proportion)
    if n_drivers:
        rnd = rng.uniform(size=n_drivers)
        both_mutations = rnd < tau_11
        only_gene_b_mutations = (rnd >= tau_11) & (rnd < tau_11 + tau_01)
        only_gene_a_mutations = (rnd >= tau_11 + tau_01) & (
            rnd < tau_11 + tau_01 + tau_10
        )
        idx = np.arange(n_drivers)
        gene_a_drivers[idx[both_mutations | only_gene_a_mutations]] = 1
        gene_b_drivers[idx[both_mutations | only_gene_b_mutations]] = 1
    return gene_a_drivers, gene_b_drivers


def simulate_pairwise_gene_somatic_mutations(
    gene_a_pmf: dict,
    gene_b_pmf: dict,
    tau_01: float,
    tau_10: float,
    tau_11: float,
    nsamples: int,
    driver_proportion: float,
) -> Interaction:
    """TODO: Add docstring."""
    gene_a_passenger_mutations = simulate_single_gene_passenger_mutations(
        gene_a_pmf,
        nsamples,
    )
    gene_b_passenger_mutations = simulate_single_gene_passenger_mutations(
        gene_b_pmf,
        nsamples,
    )
    gene_a_driver_mutations, gene_b_driver_mutations = (
        simulate_pairwise_gene_driver_mutations(
            tau_01,
            tau_10,
            tau_11,
            nsamples,
            driver_proportion,
        )
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

    return Interaction(
        gene_a=simulated_gene_a,
        gene_b=simulated_gene_b,
    )


# ------------------------------------------------------------------------------------ #
#                               SIMULATE CREATE FUNCTIONS                              #
# ------------------------------------------------------------------------------------ #


# ------------------------------------ SINGLE GENE ----------------------------------- #
def create_single_gene_simulation(
    pi: float,
    num_samples: int,
    num_simulations: int,
    length: int,
    mu: float,
    out: str,
    seed: int,
) -> None:
    """TODO: Add docstring."""
    np.random.default_rng(seed)
    dout = Path(out)
    dout.mkdir(parents=True, exist_ok=True)

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
    np.save(
        Path(dout) / "single_gene_simulated_data.npy",
        counts_array,
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
    param_out = Path(dout) / "single_gene_simulation_parameters.json"
    with param_out.open("w") as f:
        json.dump(params, f, indent=4)


# ----------------------------------- PAIR OF GENES ---------------------------------- #
def create_pair_gene_simulation(
    tau_10: float,
    tau_01: float,
    tau_11: float,
    num_samples: int,
    num_simulations: int,
    length_a: int,
    mu_a: float,
    length_b: int,
    mu_b: float,
    out: str,
    driver_proportion: float,
    seed: int,
) -> None:
    """TODO: Add docstring."""
    np.random.default_rng(seed)
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
            driver_proportion,
        )
        simulated_interactions.append(simulated_interaction)

    counts_array = np.array(
        [
            np.stack((interaction.gene_a.counts, interaction.gene_b.counts), axis=-1)
            for interaction in simulated_interactions
        ],
    )
    data_out = Path(out) / "pair_gene_simulated_data.npy"
    np.save(data_out, counts_array)

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
    param_out = Path(out) / "pair_gene_simulation_parameters.json"
    with param_out.open("w") as f:
        json.dump(params, f, indent=4)


# -------------------------------------- MATRIX -------------------------------------- #
def create_matrix_simulation(
    cnt_mtx_fn: str,
    bmr_pmfs_fn: str,
    driver_genes_fn: str,
    dout: str,
    num_likely_passengers: int,
    num_me_pairs: int,
    num_co_pairs: int,
    num_samples: int,
    tau_uv_low: float,
    tau_uv_high: float,
    driver_proportion: float,
    seed: int = 42,
) -> None:
    """TODO: Add docstring."""
    rng = np.random.default_rng(seed)
    dir_out = Path(dout)
    dir_out.mkdir(parents=True, exist_ok=True)
    tau_midpoint = (tau_uv_low + tau_uv_high) / 2
    tau_std_dev = (tau_uv_high - tau_uv_low) / 6
    a, b = (
        (tau_uv_low - tau_midpoint) / tau_std_dev,
        (tau_uv_high - tau_midpoint) / tau_std_dev,
    )

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_fn, bmr_pmfs_fn)

    drivers_arr = pd.read_csv(driver_genes_fn, sep="\t", index_col=0).index
    drivers_set = set(drivers_arr + "_M") | set(drivers_arr + "_N")

    likely_passengers_df = cnt_df.drop(drivers_set, axis=1, errors="ignore")
    likely_passengers = list(
        likely_passengers_df.sum(axis=0)
        .sort_values(ascending=False)
        .head(num_likely_passengers)
        .index,
    )

    drivers_df = cnt_df[[col for col in drivers_set if col in cnt_df.columns]]
    drivers = list(
        drivers_df.sum(axis=0)
        .sort_values(ascending=False)
        .head(2 * (num_me_pairs + num_co_pairs))
        .index,
    )
    rng.shuffle(drivers)

    me_pairs = [(drivers[i], drivers[i + 1]) for i in range(0, 2 * num_me_pairs, 2)]
    co_pairs = [
        (drivers[i], drivers[i + 1])
        for i in range(2 * num_me_pairs, 2 * (num_me_pairs + num_co_pairs), 2)
    ]

    simulated_counts = {}

    def arr_to_dict(pmf_array: list) -> dict:
        return {i: pmf_array[i] for i in range(len(pmf_array))}

    for gene_a, gene_b in me_pairs:
        bmr_pmf_a = arr_to_dict(bmr_dict[gene_a])
        bmr_pmf_b = arr_to_dict(bmr_dict[gene_b])

        tau_01 = truncnorm.rvs(
            a,
            b,
            loc=tau_midpoint,
            scale=tau_std_dev,
            random_state=rng,
        )
        tau_10 = truncnorm.rvs(
            a,
            b,
            loc=tau_midpoint,
            scale=tau_std_dev,
            random_state=rng,
        )
        tau_11 = 0.0

        interaction = simulate_pairwise_gene_somatic_mutations(
            gene_a_pmf=bmr_pmf_a,
            gene_b_pmf=bmr_pmf_b,
            tau_01=tau_01,
            tau_10=tau_10,
            tau_11=tau_11,
            nsamples=num_samples,
            driver_proportion=driver_proportion,
        )
        simulated_counts[gene_a] = interaction.gene_a.counts
        simulated_counts[gene_b] = interaction.gene_b.counts

    for gene_a, gene_b in co_pairs:
        bmr_pmf_a = arr_to_dict(bmr_dict[gene_a])
        bmr_pmf_b = arr_to_dict(bmr_dict[gene_b])

        tau_11 = truncnorm.rvs(
            a,
            b,
            loc=tau_midpoint,
            scale=tau_std_dev,
            random_state=rng,
        )
        tau_01 = 0.0
        tau_10 = 0.0

        interaction = simulate_pairwise_gene_somatic_mutations(
            gene_a_pmf=bmr_pmf_a,
            gene_b_pmf=bmr_pmf_b,
            tau_01=tau_01,
            tau_10=tau_10,
            tau_11=tau_11,
            nsamples=num_samples,
            driver_proportion=driver_proportion,
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

    final_array = [simulated_counts[g] for g in all_genes_order]
    final_array = np.array(final_array).T

    sim_df = pd.DataFrame(
        final_array,
        columns=all_genes_order,
        index=[f"S{i}" for i in range(num_samples)],
    )
    matrix_out_fn = dir_out / "count_matrix.csv"
    sim_df.index.name = "sample"
    nonzero_gene_mut_cnt_sim_df = sim_df.loc[:, sim_df.sum() != 0]
    nonzero_gene_mut_cnt_sim_df.to_csv(matrix_out_fn, index=True)

    info = {
        "ME Pairs": me_pairs,
        "CO Pairs": co_pairs,
        "Likely Passengers": likely_passengers,
        "num_samples": num_samples,
        "tau_low": tau_uv_low,
        "tau_high": tau_uv_high,
        "driver_proportion": driver_proportion,
    }
    gt_out_fn = dir_out / "matrix_simulation_info.json"
    with gt_out_fn.open("w") as f:
        json.dump(info, f, indent=4)


# ------------------------------------------------------------------------------------ #
#                              SIMULATE EVALUATE FUNCTIONS                             #
# ------------------------------------------------------------------------------------ #


# ------------------------------------ SINGLE GENE ----------------------------------- #
def evaluate_single_gene_simulation(
    params: str,
    data: str,
    out: str,
) -> None:
    """TODO: Add docstring."""
    dout = Path(out)
    dout.mkdir(parents=True, exist_ok=True)
    params_path = Path(params)

    with params_path.open() as f:
        params = json.load(f)

    pi = params.get("pi")
    num_samples = params.get("num_samples")
    bmr_pmf_arr = params.get("bmr_pmf")
    bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}

    data = np.load(data)

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

    est_pi_fout = dout / "estimated_pi_values.npy"
    np.save(est_pi_fout, np.array(est_pi_vals))
    deviations = [abs(est - pi) for est in est_pi_vals]
    results = {
        "true_pi": pi,
        "mean_estimated_pi": np.mean(est_pi_vals),
        "std_estimated_pi": np.std(est_pi_vals),
        "mean_deviation": np.mean(deviations),
        "std_deviation": np.std(deviations),
    }
    results_path = dout / "evaluation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=4)


# ----------------------------------- PAIR OF GENES ---------------------------------- #
def evaluate_pair_gene_simulation(
    params: str,
    data: str,
    out: str,
) -> None:
    """TODO: Add docstring."""
    dout = Path(out)
    dout.mkdir(parents=True, exist_ok=True)
    params_path = Path(params)

    with params_path.open() as f:
        params = json.load(f)

    true_tau_10 = params.get("tau_10")
    true_tau_01 = params.get("tau_01")
    true_tau_11 = params.get("tau_11")
    num_samples = params.get("num_samples")
    num_simulations = params.get("num_simulations")

    bmr_pmf_arr_a = params.get("bmr_pmf_a")
    bmr_pmf_arr_b = params.get("bmr_pmf_b")
    bmr_pmf_a = {i: bmr_pmf_arr_a[i] for i in range(len(bmr_pmf_arr_a))}
    bmr_pmf_b = {i: bmr_pmf_arr_b[i] for i in range(len(bmr_pmf_arr_b))}

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
    results_path = dout / "pair_evaluation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=4)


# -------------------------------------- MATRIX -------------------------------------- #
def get_ground_truth_labels(
    df: pd.DataFrame,
    simulation_info: dict,
    ixn_type: str,
) -> np.ndarray:
    """TODO: Add docstring."""
    true_ixn_pairs = set()
    key = f"{ixn_type} Pairs"
    for g1, g2 in simulation_info.get(key, []):
        true_ixn_pairs.add((g1, g2))
        true_ixn_pairs.add((g2, g1))

    labels = []
    for _, row in df.iterrows():
        pair = (row["Gene A"], row["Gene B"])
        labels.append(1 if pair in true_ixn_pairs else 0)

    return np.array(labels)


def get_method_scores(df: pd.DataFrame, num_samples: int, ixn_type: str) -> dict:
    """TODO: Add docstring."""
    dialect_rho = df["Rho"].astype(float).to_numpy()
    dialect_lrt = df["Likelihood Ratio"].astype(float).to_numpy()
    tau_1x = df["Tau_1X"].astype(float).to_numpy()
    tau_x1 = df["Tau_X1"].astype(float).to_numpy()
    epsilon = compute_epsilon_threshold(num_samples)
    mask_no_interaction = (tau_1x < epsilon) | (tau_x1 < epsilon)
    dialect_rho[mask_no_interaction] = 0.0
    dialect_lrt[mask_no_interaction] = 0.0

    if ixn_type == "ME":
        dialect_rho = -dialect_rho
        mask_co_interaction = df["Rho"] > 0
        dialect_lrt[mask_co_interaction] = 0.0
    else:
        mask_me_interaction = df["Rho"] < 0
        dialect_lrt[mask_me_interaction] = 0.0

    if ixn_type == "ME":
        fishers_pval = df["Fisher's ME P-Val"].astype(float).to_numpy()
        discover_pval = df["Discover ME P-Val"].astype(float).to_numpy()
        megsa_s = df["MEGSA S-Score (LRT)"].astype(float).to_numpy()
        wesme_pval = df["WeSME P-Val"].astype(float).to_numpy()

        fishers_score = -np.log10(fishers_pval + 1e-300)
        discover_score = -np.log10(discover_pval + 1e-300)
        wesme_score = -np.log10(wesme_pval + 1e-300)

        method_to_scores = {
            "DIALECT (Rho)": dialect_rho,
            "DIALECT (LRT)": dialect_lrt,
            "Fisher's Exact Test": fishers_score,
            "DISCOVER": discover_score,
            "WeSME": wesme_score,
            "MEGSA": megsa_s,
        }

    else:
        fishers_pval = df["Fisher's CO P-Val"].astype(float).to_numpy()
        discover_pval = df["Discover CO P-Val"].astype(float).to_numpy()
        wesco_pval = df["WeSCO P-Val"].astype(float).to_numpy()

        fishers_score = -np.log10(fishers_pval + 1e-300)
        discover_score = -np.log10(discover_pval + 1e-300)
        wesco_score = -np.log10(wesco_pval + 1e-300)

        method_to_scores = {
            "DIALECT (Rho)": dialect_rho,
            "DIALECT (LRT)": dialect_lrt,
            "Fisher's Exact Test": fishers_score,
            "DISCOVER": discover_score,
            "WeSCO": wesco_score,
        }

    return method_to_scores

def evaluate_auc_vs_driver_proportion(
    nruns: int,
    results_dir: Path,
    out: Path,
    ixn_type: str,
) -> None:
    """TODO: Add docstring."""
    methods = ME_METHODS if ixn_type == "ME" else CO_METHODS
    method_to_auprc_vals = {}
    driver_proportions = np.arange(0.05, 1.01, 0.05)
    for driver_proportion in driver_proportions:
        formatted_dp = f"{driver_proportion:.1f}DP"
        all_y_true = []
        all_method_scores = []
        for i in range(1, nruns + 1):
            results_base_dir = Path(
                re.sub(
                    r"/[\d\.]+DP/",
                    f"/{formatted_dp}/",
                    str(results_dir),
                ),
            )
            results_fn = (
                results_base_dir / f"R{i}" / "complete_pairwise_ixn_results.csv"
            )
            simulation_info_fn = (
                results_base_dir / f"R{i}" / "matrix_simulation_info.json"
            )
            results_df = pd.read_csv(results_fn)
            with simulation_info_fn.open() as f:
                gt = json.load(f)
            num_samples = gt.get("num_samples")
            all_y_true.append(get_ground_truth_labels(results_df, gt, ixn_type))
            all_method_scores.append(
                get_method_scores(results_df, num_samples, ixn_type),
            )
        for _, method_name in enumerate(methods):
            y_true_concat = []
            scores_concat = []
            for y_true, methods_dict in zip(all_y_true, all_method_scores):
                scores = methods_dict[method_name]
                y_true_concat.append(y_true)
                scores_concat.append(scores)
            y_true_concat = np.concatenate(y_true_concat)
            scores_concat = np.concatenate(scores_concat)
            auc_val = average_precision_score(y_true_concat, scores_concat)
            if method_name not in method_to_auprc_vals:
                method_to_auprc_vals[method_name] = []
            method_to_auprc_vals[method_name].append(auc_val)
    draw_auc_vs_factor_curve(
        driver_proportions,
        method_to_auprc_vals,
        out / f"{ixn_type}_auc_vs_driver_proportion_curve",
        xlabel="Driver Proportion",
    )

def evaluate_auc_vs_num_samples(
    nruns: int,
    results_dir: Path,
    out: Path,
    ixn_type: str,
) -> None:
    """TODO: Add docstring."""
    methods = ME_METHODS if ixn_type == "ME" else CO_METHODS
    method_to_auprc_vals = {}
    num_samples_list = np.arange(50, 1851, 50)
    for num_samples in num_samples_list:
        formatted_ns = f"NS{num_samples}"
        all_y_true = []
        all_method_scores = []
        for i in range(1, nruns + 1):
            results_base_dir = Path(
                re.sub(
                    r"/NS[\d\.]+/",
                    f"/{formatted_ns}/",
                    str(results_dir),
                ),
            )
            results_fn = (
                results_base_dir / f"R{i}" / "complete_pairwise_ixn_results.csv"
            )
            simulation_info_fn = (
                results_base_dir / f"R{i}" / "matrix_simulation_info.json"
            )
            results_df = pd.read_csv(results_fn)
            with simulation_info_fn.open() as f:
                gt = json.load(f)
            all_y_true.append(get_ground_truth_labels(results_df, gt, ixn_type))
            all_method_scores.append(
                get_method_scores(results_df, num_samples, ixn_type),
            )
        for _, method_name in enumerate(methods):
            y_true_concat = []
            scores_concat = []
            for y_true, methods_dict in zip(all_y_true, all_method_scores):
                scores = methods_dict[method_name]
                y_true_concat.append(y_true)
                scores_concat.append(scores)
            y_true_concat = np.concatenate(y_true_concat)
            scores_concat = np.concatenate(scores_concat)
            auc_val = average_precision_score(y_true_concat, scores_concat)
            if method_name not in method_to_auprc_vals:
                method_to_auprc_vals[method_name] = []
            method_to_auprc_vals[method_name].append(auc_val)
    draw_auc_vs_factor_curve(
        num_samples_list,
        method_to_auprc_vals,
        out / f"{ixn_type}_auc_vs_num_samples_curve",
        xlabel="Number of Samples",
    )

def evaluate_matrix_simulation(
    results_dir: Path,
    out: Path,
    nruns: int,
    ixn_type: str,
) -> None:
    """TODO: Add docstring."""
    all_y_true = []
    all_method_scores = []
    true_me_pairs = []
    true_co_pairs = []
    top_ranked_me_tables = []
    top_ranked_co_tables = []
    num_genes = None
    for i in range(1, nruns + 1):
        results_fn = results_dir / f"R{i}" / "complete_pairwise_ixn_results.csv"
        simulation_info_fn = results_dir / f"R{i}" / "matrix_simulation_info.json"
        results_df = pd.read_csv(results_fn)
        with simulation_info_fn.open() as f:
            gt = json.load(f)
        num_samples = gt.get("num_samples")
        num_genes = len(
            gt.get("ME Pairs") + gt.get("CO Pairs") + gt.get("Likely Passengers"),
        )
        all_y_true.append(get_ground_truth_labels(results_df, gt, ixn_type))
        all_method_scores.append(get_method_scores(results_df, num_samples, ixn_type))

        true_me_pairs.append(gt.get("ME Pairs", []))
        top_ranked_me_tables.append(
            generate_top_ranked_me_interaction_tables(
                results_df=results_df,
                num_pairs=1_000,
                num_samples=num_samples,
                methods=ME_METHODS,
            ),
        )
        true_co_pairs.append(gt.get("CO Pairs", []))
        top_ranked_co_tables.append(
            generate_top_ranked_co_interaction_tables(
                results_df=results_df,
                num_pairs=1_000,
                num_samples=num_samples,
                methods=CO_METHODS,
            ),
        )

    methods = ME_METHODS if ixn_type == "ME" else CO_METHODS

    draw_average_simulation_precision_recall_curve(
        all_method_scores,
        all_y_true,
        out / f"{ixn_type}_average_pr_curve",
        methods=methods,
    )

    draw_concat_simulation_precision_recall_curve(
        all_method_scores,
        all_y_true,
        out / f"{ixn_type}_concat_pr_curve",
        methods=methods,
    )

    if ixn_type == "ME":
        true_pairs = true_me_pairs
        top_ranked_tables = top_ranked_me_tables
    else:
        true_pairs = true_co_pairs
        top_ranked_tables = top_ranked_co_tables

    draw_hit_curve(
        true_pairs,
        top_ranked_tables,
        methods=methods,
        total_pairs=num_genes * (num_genes - 1) // 2,
        fout=out / f"{ixn_type}_hit_curve",
    )

    evaluate_auc_vs_driver_proportion(
        nruns,
        results_dir,
        out,
        ixn_type,
    )

    evaluate_auc_vs_num_samples(
        nruns,
        results_dir,
        out,
        ixn_type,
    )

import os
import sys
import numpy as np
import pandas as pd
import importlib.resources
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import binom
from itertools import combinations

os.environ["DIALECT_PROJECT_DIRECTORY"] = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


def generate_bmr_pmfs(gene_size, max_x=5, mu=1e-6):
    bmr_pmfs = [binom.pmf(x, gene_size, mu) for x in range(max_x)]
    return bmr_pmfs


def generate_pairwise_driver_mutations(nsamples, tau_10, tau_01, tau_11):
    gene_a_driver_mutations = np.zeros(nsamples)
    gene_b_driver_mutations = np.zeros(nsamples)
    rnd = np.random.uniform(size=nsamples)

    both_mutations = rnd < tau_11
    only_gene_b_mutations = (rnd >= tau_11) & (rnd < tau_11 + tau_01)
    only_gene_a_mutations = (rnd >= tau_11 + tau_01) & (rnd < tau_11 + tau_01 + tau_10)

    gene_a_driver_mutations[both_mutations | only_gene_a_mutations] = 1
    gene_b_driver_mutations[both_mutations | only_gene_b_mutations] = 1

    return gene_a_driver_mutations, gene_b_driver_mutations


def calculate_fisher_results(vals, results, nruns, alpha_threshold=0.05):
    fisher_results = {}
    for val in vals:
        fishers_pvals = [x[1] for x in results["fishers"][val]]
        fisher_results[val] = sum(1 for x in fishers_pvals if x < alpha_threshold) / nruns
    return fisher_results


def calculate_dialect_results(
    vals, results, nruns, llr_threshold=3, interaction_type="me"
):  # TODO: TUNE LLR THRESHOLD
    dialect_results = {}
    for val in vals:
        dialect_llrs = [x[2] for x in results["dialect"][val]]
        dialect_log_odds = [x[3] for x in results["dialect"][val]]
        if interaction_type == "me":
            dialect_results[val] = (
                sum(
                    1 for x, y in zip(dialect_llrs, dialect_log_odds) if x > llr_threshold and y > 0
                )
                / nruns
            )
        else:
            dialect_results[val] = (
                sum(
                    1 for x, y in zip(dialect_llrs, dialect_log_odds) if x > llr_threshold and y < 0
                )
                / nruns
            )
    return dialect_results


def run_simulation(
    nruns,
    tau_10,
    tau_01,
    tau_11,
    interaction_type,
    variable_param,
    variable_values,
    gene_a_size,
    gene_b_size,
    fixed_nsamples=1000,
):
    fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(fishers_dir)
    sys.path.append(dialect_dir)
    fishers_module = importlib.import_module("run_fishers")
    dialect_module = importlib.import_module("core")

    gene_a_bmr_pmfs = generate_bmr_pmfs(gene_a_size)  # P(x = 0) = ~0.99
    gene_a_bmr_pmfs = gene_a_bmr_pmfs / sum(gene_a_bmr_pmfs)

    results = {"fishers": {}, "dialect": {}}
    for variable_value in variable_values:
        results["fishers"][variable_value] = []
        results["dialect"][variable_value] = []

        if variable_param == "gene_b_nonzero_bmr":
            gene_b_bmr_pmfs = generate_bmr_pmfs(gene_b_size, mu=variable_value)
            gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)
            nsamples = fixed_nsamples
        elif variable_param == "num_samples":
            nsamples = variable_value
            gene_b_bmr_pmfs = generate_bmr_pmfs(gene_b_size)  # P(x = 0) = ~0.99
            gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)
        elif variable_param == "driver_subset_size":
            gene_b_size = 1e5
            gene_b_bmr_pmfs = generate_bmr_pmfs(gene_b_size, mu=5e-6)  # P(x = 0) = ~0.60
            gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)
            nsamples = fixed_nsamples

        for _ in range(nruns):
            gene_a_passenger_mutations = np.random.choice(
                len(gene_a_bmr_pmfs), nsamples, p=gene_a_bmr_pmfs
            )
            gene_b_passenger_mutations = np.random.choice(
                len(gene_b_bmr_pmfs), nsamples, p=gene_b_bmr_pmfs
            )

            if variable_param == "driver_subset_size":
                gene_a_driver_mutations = np.concatenate(
                    [
                        np.random.choice([0, 1], variable_value, p=[1 - tau_10, tau_10]),
                        np.zeros(nsamples - variable_value),
                    ]
                )
                gene_b_driver_mutations = np.zeros(nsamples)
            else:
                gene_a_driver_mutations, gene_b_driver_mutations = (
                    generate_pairwise_driver_mutations(nsamples, tau_10, tau_01, tau_11)
                )

            # combine to get total somatic mutations for each gene then binarize
            gene_a_somatic_mutations = (
                gene_a_passenger_mutations + gene_a_driver_mutations
            ).astype(int)
            gene_b_somatic_mutations = (
                gene_b_passenger_mutations + gene_b_driver_mutations
            ).astype(int)
            gene_a_binarized_mutations = np.array(
                [1 if x > 0 else 0 for x in gene_a_somatic_mutations]
            )
            gene_b_binarized_mutations = np.array(
                [1 if x > 0 else 0 for x in gene_b_somatic_mutations]
            )

            # run fishers exact test and store results
            fishers_result = fishers_module.fishers_exact_test(
                gene_a_binarized_mutations, gene_b_binarized_mutations, interaction_type
            )
            results["fishers"][variable_value].append(fishers_result)

            # run dialect and store results
            gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
                gene_a_somatic_mutations, gene_a_bmr_pmfs
            )
            gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
                gene_b_somatic_mutations, gene_b_bmr_pmfs
            )
            dialect_result = dialect_module.dialect_pairwise(
                gene_a_pi,
                gene_b_pi,
                gene_a_log_likelihood,
                gene_b_log_likelihood,
                gene_a_somatic_mutations,
                gene_b_somatic_mutations,
                gene_a_bmr_pmfs,
                gene_b_bmr_pmfs,
            )
            results["dialect"][variable_value].append(dialect_result)

    fisher_results = calculate_fisher_results(variable_values, results, nruns)
    dialect_results = calculate_dialect_results(
        variable_values,
        results,
        nruns,
        llr_threshold=1,
        interaction_type=interaction_type if variable_param != "driver_subset_size" else "me",
    )

    return fisher_results, dialect_results


def generate_somatic_mutations(nsamples, tau, gene_a_bmr_pmfs, gene_b_bmr_pmfs):
    # Generate passenger mutations
    gene_a_passenger_mutations = np.random.choice(len(gene_a_bmr_pmfs), nsamples, p=gene_a_bmr_pmfs)
    gene_b_passenger_mutations = np.random.choice(len(gene_b_bmr_pmfs), nsamples, p=gene_b_bmr_pmfs)

    # Generate driver mutations
    gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
        nsamples, tau, tau, 0
    )

    # Combine to get total somatic mutations
    gene_a_somatic_mutations = (gene_a_passenger_mutations + gene_a_driver_mutations).astype(int)
    gene_b_somatic_mutations = (gene_b_passenger_mutations + gene_b_driver_mutations).astype(int)

    return gene_a_somatic_mutations, gene_b_somatic_mutations


def calculate_llr(nsamples, nruns, tau_values, gene_a_size, gene_b_size):
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")

    gene_a_bmr_pmfs = generate_bmr_pmfs(gene_a_size)
    gene_a_bmr_pmfs = gene_a_bmr_pmfs / sum(gene_a_bmr_pmfs)
    gene_b_bmr_pmfs = generate_bmr_pmfs(gene_b_size)
    gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)

    llr_results = {tau: [] for tau in tau_values}

    for tau in tau_values:
        for _ in range(nruns):
            gene_a_somatic_mutations, gene_b_somatic_mutations = generate_somatic_mutations(
                nsamples, tau, gene_a_bmr_pmfs, gene_b_bmr_pmfs
            )

            gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
                gene_a_somatic_mutations, gene_a_bmr_pmfs
            )
            gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
                gene_b_somatic_mutations, gene_b_bmr_pmfs
            )
            _, _, llr, _ = dialect_module.dialect_pairwise(
                gene_a_pi,
                gene_b_pi,
                gene_a_log_likelihood,
                gene_b_log_likelihood,
                gene_a_somatic_mutations,
                gene_b_somatic_mutations,
                gene_a_bmr_pmfs,
                gene_b_bmr_pmfs,
            )

            llr_results[tau].append(llr)

    return llr_results


def plot_llr_distribution(llr_results, tau_values):
    plt.figure(figsize=(12, 8))
    for tau in tau_values:
        plt.hist(llr_results[tau], bins=50, alpha=0.5, label=f"tau={tau}")
    plt.xlabel("Log Likelihood Ratio (LLR)")
    plt.ylabel("Frequency")
    plt.title("LLR Distribution for Different tau Values")
    plt.legend()
    plt.show()


def generate_null_distribution(nsamples, nruns, gene_a_bmr_pmfs, gene_b_bmr_pmfs):
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")
    null_llrs = []
    null_log_odds = []

    for _ in range(nruns):
        gene_a_passenger_mutations = np.random.choice(
            len(gene_a_bmr_pmfs), nsamples, p=gene_a_bmr_pmfs
        )
        gene_b_passenger_mutations = np.random.choice(
            len(gene_b_bmr_pmfs), nsamples, p=gene_b_bmr_pmfs
        )

        gene_a_somatic_mutations = gene_a_passenger_mutations.astype(int)
        gene_b_somatic_mutations = gene_b_passenger_mutations.astype(int)

        gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
            gene_a_somatic_mutations, gene_a_bmr_pmfs
        )
        gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
            gene_b_somatic_mutations, gene_b_bmr_pmfs
        )
        _, _, llr, log_odds = dialect_module.dialect_pairwise(
            gene_a_pi,
            gene_b_pi,
            gene_a_log_likelihood,
            gene_b_log_likelihood,
            gene_a_somatic_mutations,
            gene_b_somatic_mutations,
            gene_a_bmr_pmfs,
            gene_b_bmr_pmfs,
        )

        null_llrs.append(llr)
        null_log_odds.append(log_odds)

    return null_llrs, null_log_odds


def determine_llr_threshold(null_llrs, false_positive_rate=0.05):
    return np.percentile(null_llrs, 100 * (1 - false_positive_rate))


def assess_significance(real_llr, llr_threshold):
    return real_llr > llr_threshold


# def tune_significance_threshold():
#     dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
#     sys.path.append(dialect_dir)
#     dialect_module = importlib.import_module("core")
#     fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
#     sys.path.append(fishers_dir)
#     fishers_module = importlib.import_module("run_fishers")

#     brca_cnt_mtx_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/BRCA/BRCA/BRCA_cbase_cnt_mtx.csv"
#     brca_bmr_pmfs_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/BRCA/BRCA/BRCA_cbase_bmr_pmfs.csv"

#     brca_cnt_mtx_df = pd.read_csv(brca_cnt_mtx_fn, index_col=0)
#     brca_bmr_pmfs_df = pd.read_csv(brca_bmr_pmfs_fn, index_col=0)
#     brca_bmrs_pmfs_dict = brca_bmr_pmfs_df.T.to_dict(orient="list")
#     brca_bmrs_pmfs_dict = {
#         key: [x for x in brca_bmrs_pmfs_dict[key] if not np.isnan(x)] for key in brca_bmrs_pmfs_dict
#     }

#     gene_a = "AKT1_M"
#     gene_b = "TTN_M"


#     # nsamples = 1000
#     # nruns = 100
#     gene_a_size = 1e4
#     gene_b_size = 1e6
#     # tau_values = np.array([0])  # np.arange(0.1, 0.125, 0.025)

#     # llr_results = calculate_llr(nsamples, nruns, tau_values, gene_a_size, gene_b_size)
#     # plot_llr_distribution(llr_results, tau_values)
#     # Generate null distribution for a given pair of genes
#     nsamples = brca_cnt_mtx_df.shape[0]  # 1000
#     nruns = 1000
#     gene_a_bmr_pmfs = np.array(brca_bmrs_pmfs_dict[gene_a])  # generate_bmr_pmfs(gene_a_size)
#     gene_a_bmr_pmfs = gene_a_bmr_pmfs / sum(gene_a_bmr_pmfs)
#     gene_b_bmr_pmfs = np.array(brca_bmrs_pmfs_dict[gene_b])  # generate_bmr_pmfs(gene_b_size)
#     gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)
#     print(f"Gene {gene_a} Counts: {brca_cnt_mtx_df[gene_a].sum()}")
#     print(f"Gene {gene_b} Counts: {brca_cnt_mtx_df[gene_b].sum()}")

#     # sample from gene_a_bmr_pmfs and gene_b_bmr_pmfs
#     gene_a_sim_passengers = np.random.choice(
#         len(gene_a_bmr_pmfs), nsamples, p=gene_a_bmr_pmfs
#     ).sum()
#     gene_b_sim_passengers = np.random.choice(
#         len(gene_b_bmr_pmfs), nsamples, p=gene_b_bmr_pmfs
#     ).sum()
#     print(f"Gene {gene_a} Simulated Passengers: {gene_a_sim_passengers}")
#     print(f"Gene {gene_b} Simulated Passengers: {gene_b_sim_passengers}")
#     quit()

#     null_llrs = generate_null_distribution(nsamples, nruns, gene_a_bmr_pmfs, gene_b_bmr_pmfs)

#     # Determine the LLR threshold for a 0.1% false positive rate
#     llr_threshold = determine_llr_threshold(null_llrs, false_positive_rate=0.001)

#     # Calculate the LLR for the real data
#     gene_a_somatic_mutations = brca_cnt_mtx_df[gene_a]
#     gene_b_somatic_mutations = brca_cnt_mtx_df[gene_b]
#     gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
#         gene_a_somatic_mutations, gene_a_bmr_pmfs
#     )
#     gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
#         gene_b_somatic_mutations, gene_b_bmr_pmfs
#     )
#     _, _, real_llr, _ = dialect_module.dialect_pairwise(
#         gene_a_pi,
#         gene_b_pi,
#         gene_a_log_likelihood,
#         gene_b_log_likelihood,
#         gene_a_somatic_mutations,
#         gene_b_somatic_mutations,
#         gene_a_bmr_pmfs,
#         gene_b_bmr_pmfs,
#     )

#     gene_a_binarized_mutations = np.array([1 if x > 0 else 0 for x in gene_a_somatic_mutations])
#     gene_b_binarized_mutations = np.array([1 if x > 0 else 0 for x in gene_b_somatic_mutations])
#     fishers_result = fishers_module.fishers_exact_test(
#         gene_a_binarized_mutations, gene_b_binarized_mutations, "co"
#     )
#     print(f"Fishers Result: {fishers_result}")

#     # # Assess the significance of the real LLR
#     is_significant = assess_significance(real_llr, llr_threshold)

#     print(f"Real LLR: {real_llr}")
#     print(f"LLR Threshold: {llr_threshold}")
#     print(f"Is Significant: {is_significant}")


def run_simulation_with_false_positive_analysis(
    nsamples, nruns, tau_10, tau_01, gene_a_bmr_pmfs, gene_b_bmr_pmfs, llr_threshold
):
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")
    fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
    sys.path.append(fishers_dir)
    fishers_module = importlib.import_module("run_fishers")

    false_positives_dialect = 0
    false_positives_fishers = 0

    for _ in range(nruns):
        gene_a_passenger_mutations = np.random.choice(
            len(gene_a_bmr_pmfs), nsamples, p=gene_a_bmr_pmfs
        )
        gene_b_passenger_mutations = np.random.choice(
            len(gene_b_bmr_pmfs), nsamples, p=gene_b_bmr_pmfs
        )

        gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
            nsamples, tau_10, tau_01, 0
        )

        gene_a_somatic_mutations = (gene_a_passenger_mutations + gene_a_driver_mutations).astype(
            int
        )
        gene_b_somatic_mutations = (gene_b_passenger_mutations + gene_b_driver_mutations).astype(
            int
        )
        gene_a_binarized_mutations = np.array([1 if x > 0 else 0 for x in gene_a_somatic_mutations])
        gene_b_binarized_mutations = np.array([1 if x > 0 else 0 for x in gene_b_somatic_mutations])

        # Dialect method
        gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
            gene_a_somatic_mutations, gene_a_bmr_pmfs
        )
        gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
            gene_b_somatic_mutations, gene_b_bmr_pmfs
        )
        _, _, real_llr, _ = dialect_module.dialect_pairwise(
            gene_a_pi,
            gene_b_pi,
            gene_a_log_likelihood,
            gene_b_log_likelihood,
            gene_a_somatic_mutations,
            gene_b_somatic_mutations,
            gene_a_bmr_pmfs,
            gene_b_bmr_pmfs,
        )

        if real_llr > llr_threshold:
            false_positives_dialect += 1

        # Fishers method
        fishers_result = fishers_module.fishers_exact_test(
            gene_a_binarized_mutations, gene_b_binarized_mutations, "me"
        )[1]
        if fishers_result < 0.05:
            false_positives_fishers += 1

    return false_positives_dialect / nruns, false_positives_fishers / nruns


def plot_fpr_vs_gene_a_pi_driver_and_passenger():
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")
    fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
    sys.path.append(fishers_dir)
    fishers_module = importlib.import_module("run_fishers")

    brca_cnt_mtx_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/AML/AML_cbase_cnt_mtx.csv"
    brca_bmr_pmfs_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/AML/AML_cbase_bmr_pmfs.csv"

    brca_cnt_mtx_df = pd.read_csv(brca_cnt_mtx_fn, index_col=0)
    brca_bmr_pmfs_df = pd.read_csv(brca_bmr_pmfs_fn, index_col=0)
    brca_bmrs_pmfs_dict = brca_bmr_pmfs_df.T.to_dict(orient="list")
    brca_bmrs_pmfs_dict = {
        key: [x for x in brca_bmrs_pmfs_dict[key] if not np.isnan(x)] for key in brca_bmrs_pmfs_dict
    }

    gene_a = "DNMT3A_M"
    gene_b = "TTN_M"
    gene_a_bmr_pmfs = np.array(brca_bmrs_pmfs_dict[gene_a])
    gene_a_bmr_pmfs = gene_a_bmr_pmfs / sum(gene_a_bmr_pmfs)
    gene_b_bmr_pmfs = np.array(brca_bmrs_pmfs_dict[gene_b])
    gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)

    # Parameters
    nsamples = 1000
    nruns = 100
    tau_01 = 0.0

    tau_10_values = np.arange(0.05, 0.35, 0.05)
    false_positive_rates_dialect = []
    false_positive_rates_fishers = []

    # Generate null distribution and determine LLR threshold
    null_llrs = generate_null_distribution(10 * nsamples, nruns, gene_a_bmr_pmfs, gene_b_bmr_pmfs)
    llr_threshold = determine_llr_threshold(null_llrs, false_positive_rate=0.01)
    for tau_10 in tau_10_values:
        # Run simulation to assess false positives
        false_positive_rate_dialect, false_positive_rate_fishers = (
            run_simulation_with_false_positive_analysis(
                nsamples, nruns, tau_10, tau_01, gene_a_bmr_pmfs, gene_b_bmr_pmfs, llr_threshold
            )
        )

        false_positive_rates_dialect.append(false_positive_rate_dialect)
        false_positive_rates_fishers.append(false_positive_rate_fishers)

        print(f"tau_10: {tau_10}, LLR Threshold: {llr_threshold}")
        print(f"False Positive Rate (Dialect): {false_positive_rate_dialect}")
        print(f"False Positive Rate (Fishers): {false_positive_rate_fishers}")

    # Plot the false positive rates
    plt.figure(figsize=(10, 6))
    plt.plot(
        tau_10_values,
        false_positive_rates_dialect,
        label="Dialect",
        marker="o",
        color="green",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        tau_10_values,
        false_positive_rates_fishers,
        label="Fishers",
        marker="x",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel(r"$\tau_{10}$", fontsize=18)
    plt.ylabel("False Positive Rate", fontsize=18)
    plt.title(r"FPR vs. $\tau_{10}$", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sensitivity_vs_tau():
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")
    fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
    sys.path.append(fishers_dir)
    fishers_module = importlib.import_module("run_fishers")

    aml_cnt_mtx_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/AML/AML_cbase_cnt_mtx.csv"
    aml_bmr_pmfs_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/AML/AML_cbase_bmr_pmfs.csv"

    aml_cnt_mtx_df = pd.read_csv(aml_cnt_mtx_fn, index_col=0)
    aml_bmr_pmfs_df = pd.read_csv(aml_bmr_pmfs_fn, index_col=0)
    aml_bmrs_pmfs_dict = aml_bmr_pmfs_df.T.to_dict(orient="list")
    aml_bmrs_pmfs_dict = {
        key: [x for x in aml_bmrs_pmfs_dict[key] if not np.isnan(x)] for key in aml_bmrs_pmfs_dict
    }

    gene_a = "DNMT3A_M"
    gene_b = "IDH2_M"
    gene_a_bmr_pmfs = np.array(aml_bmrs_pmfs_dict[gene_a])
    gene_a_bmr_pmfs = gene_a_bmr_pmfs / sum(gene_a_bmr_pmfs)
    gene_b_bmr_pmfs = np.array(aml_bmrs_pmfs_dict[gene_b])
    gene_b_bmr_pmfs = gene_b_bmr_pmfs / sum(gene_b_bmr_pmfs)

    # Parameters
    nsamples = 1000
    nruns = 300
    tau_11 = 0.0

    tau_values = np.arange(0.025, 0.125, 0.025)
    sensitivities_dialect = []
    sensitivities_fishers = []

    # Generate null distribution and determine LLR threshold
    null_llrs = generate_null_distribution(10 * nsamples, nruns, gene_a_bmr_pmfs, gene_b_bmr_pmfs)
    llr_threshold = determine_llr_threshold(null_llrs, false_positive_rate=0.01)

    for tau in tau_values:
        tau_10 = tau
        tau_01 = tau

        # Run simulation to assess sensitivity
        true_positive_dialect = 0
        true_positive_fishers = 0

        for _ in range(nruns):
            # Generate driver and passenger mutations
            gene_a_passenger_mutations = np.random.choice(
                len(gene_a_bmr_pmfs), nsamples, p=gene_a_bmr_pmfs
            )
            gene_b_passenger_mutations = np.random.choice(
                len(gene_b_bmr_pmfs), nsamples, p=gene_b_bmr_pmfs
            )

            gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
                nsamples, tau_10, tau_01, tau_11
            )

            gene_a_somatic_mutations = (
                gene_a_passenger_mutations + gene_a_driver_mutations
            ).astype(int)
            gene_b_somatic_mutations = (
                gene_b_passenger_mutations + gene_b_driver_mutations
            ).astype(int)
            gene_a_binarized_mutations = np.array(
                [1 if x > 0 else 0 for x in gene_a_somatic_mutations]
            )
            gene_b_binarized_mutations = np.array(
                [1 if x > 0 else 0 for x in gene_b_somatic_mutations]
            )

            # Dialect method
            gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
                gene_a_somatic_mutations, gene_a_bmr_pmfs
            )
            gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
                gene_b_somatic_mutations, gene_b_bmr_pmfs
            )
            _, _, real_llr, _ = dialect_module.dialect_pairwise(
                gene_a_pi,
                gene_b_pi,
                gene_a_log_likelihood,
                gene_b_log_likelihood,
                gene_a_somatic_mutations,
                gene_b_somatic_mutations,
                gene_a_bmr_pmfs,
                gene_b_bmr_pmfs,
            )

            if real_llr > llr_threshold:
                true_positive_dialect += 1

            # Fishers method
            fishers_result = fishers_module.fishers_exact_test(
                gene_a_binarized_mutations, gene_b_binarized_mutations, "me"
            )[1]
            if fishers_result < 0.05:
                true_positive_fishers += 1

        sensitivity_dialect = true_positive_dialect / nruns
        sensitivity_fishers = true_positive_fishers / nruns

        sensitivities_dialect.append(sensitivity_dialect)
        sensitivities_fishers.append(sensitivity_fishers)

        print(f"tau: {tau}, LLR Threshold: {llr_threshold}")
        print(f"Sensitivity (Dialect): {sensitivity_dialect}")
        print(f"Sensitivity (Fishers): {sensitivity_fishers}")

    # Plot the sensitivities
    plt.figure(figsize=(10, 6))
    plt.plot(
        tau_values,
        sensitivities_dialect,
        label="Dialect",
        marker="o",
        color="green",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        tau_values,
        sensitivities_fishers,
        label="Fishers",
        marker="x",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel("Sensitivity", fontsize=18)
    plt.title(r"Sensitivity vs. $\tau$", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()


def read_bmrs():
    # Load BMRs for DNMT3A_M and TTN_M
    aml_bmr_pmfs_fn = "/Users/ahmed/workspace/research/dialect/results/tcga_pancan_atlas_2018/AML/AML_cbase_bmr_pmfs.csv"
    aml_bmr_df = pd.read_csv(aml_bmr_pmfs_fn, index_col=0)
    aml_bmrs_pmfs_dict = aml_bmr_df.T.to_dict(orient="list")
    aml_bmrs_pmfs_dict = {
        key: [x for x in aml_bmrs_pmfs_dict[key] if not np.isnan(x)] for key in aml_bmrs_pmfs_dict
    }
    aml_bmrs_pmfs_dict = {
        key: [x / sum(aml_bmrs_pmfs_dict[key]) for x in aml_bmrs_pmfs_dict[key]]
        for key in aml_bmrs_pmfs_dict
    }

    # Assign BMRs to DRVR and PSNGR genes
    drvr_bmr_pmfs = aml_bmrs_pmfs_dict["DNMT3A_M"]
    psngr_bmr_pmfs = aml_bmrs_pmfs_dict["TTN_M"]
    return drvr_bmr_pmfs, psngr_bmr_pmfs


def small_matrix_simulation():
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")
    fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
    sys.path.append(fishers_dir)
    fishers_module = importlib.import_module("run_fishers")

    drvr_bmr_pmfs, psngr_bmr_pmfs = read_bmrs()

    # Define gene names
    drvr_genes = [f"DRVR_{i}" for i in range(1, 51)]
    psngr_genes = [f"PSNGR_{i}" for i in range(1, 51)]
    all_genes = drvr_genes + psngr_genes

    # Define interaction types
    mut_excl_pairs = [(f"DRVR_{2*i+1}", f"DRVR_{2*i+2}") for i in range(10)]
    co_occ_pairs = [(f"DRVR_{2*i+21}", f"DRVR_{2*i+22}") for i in range(10)]
    indep_drvr_genes = [f"DRVR_{i}" for i in range(41, 51)]

    tau_values = [0.025, 0.05, 0.10, 0.15, 0.20]
    nsamples = 1000

    mutation_matrix = pd.DataFrame(0, index=range(nsamples), columns=all_genes)

    for i, (gene_a, gene_b) in enumerate(mut_excl_pairs):
        tau_10 = tau_01 = tau_values[i % 5]
        tau_11 = 0
        gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
            nsamples, tau_10, tau_01, tau_11
        )
        mutation_matrix[gene_a] += gene_a_driver_mutations
        mutation_matrix[gene_b] += gene_b_driver_mutations

    for i, (gene_a, gene_b) in enumerate(co_occ_pairs):
        tau_11 = tau_values[i % 5]
        tau_10 = tau_01 = 0
        gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
            nsamples, tau_10, tau_01, tau_11
        )
        mutation_matrix[gene_a] += gene_a_driver_mutations
        mutation_matrix[gene_b] += gene_b_driver_mutations

    for i, gene in enumerate(indep_drvr_genes):
        tau = tau_values[i % 5]
        gene_driver_mutations = np.random.binomial(1, tau, nsamples)
        mutation_matrix[gene] += gene_driver_mutations

    # Assign BMRs to all genes
    for gene in drvr_genes:
        bmr_pmfs = np.array(drvr_bmr_pmfs)
        mutation_matrix[gene] += np.random.choice(len(bmr_pmfs), nsamples, p=bmr_pmfs)

    for gene in psngr_genes:
        bmr_pmfs = np.array(psngr_bmr_pmfs)
        mutation_matrix[gene] += np.random.choice(len(bmr_pmfs), nsamples, p=bmr_pmfs)

    mutation_matrix = mutation_matrix.astype(int)

    # Generate null distributions and determine LLR thresholds
    null_llrs_drvr_drvr = generate_null_distribution(nsamples, 100, drvr_bmr_pmfs, drvr_bmr_pmfs)
    llr_threshold_drvr_drvr = determine_llr_threshold(
        null_llrs_drvr_drvr, false_positive_rate=0.001
    )

    null_llrs_drvr_psngr = generate_null_distribution(nsamples, 100, drvr_bmr_pmfs, psngr_bmr_pmfs)
    llr_threshold_drvr_psngr = determine_llr_threshold(
        null_llrs_drvr_psngr, false_positive_rate=0.001
    )

    null_llrs_psngr_psngr = generate_null_distribution(
        nsamples, 100, psngr_bmr_pmfs, psngr_bmr_pmfs
    )
    llr_threshold_psngr_psngr = determine_llr_threshold(
        null_llrs_psngr_psngr, false_positive_rate=0.001
    )

    print(f"LLR Threshold (DRVR-DRVR): {llr_threshold_drvr_drvr}")
    print(f"LLR Threshold (DRVR-PSNGR): {llr_threshold_drvr_psngr}")
    print(f"LLR Threshold (PSNGR-PSNGR): {llr_threshold_psngr_psngr}")

    def determine_interaction_type(gene_a, gene_b):
        if (gene_a, gene_b) in mut_excl_pairs or (gene_b, gene_a) in mut_excl_pairs:
            return "me"
        elif (gene_a, gene_b) in co_occ_pairs or (gene_b, gene_a) in co_occ_pairs:
            return "co"
        else:
            return "None"

    results_dialect = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    results_fishers = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    gene_pairs = list(combinations(all_genes, 2))
    for gene_a, gene_b in tqdm(gene_pairs):
        interaction_type = determine_interaction_type(gene_a, gene_b)
        gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
            mutation_matrix[gene_a], drvr_bmr_pmfs if "DRVR" in gene_a else psngr_bmr_pmfs
        )
        gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
            mutation_matrix[gene_b], drvr_bmr_pmfs if "DRVR" in gene_b else psngr_bmr_pmfs
        )
        _, _, llr, _ = dialect_module.dialect_pairwise(
            gene_a_pi,
            gene_b_pi,
            gene_a_log_likelihood,
            gene_b_log_likelihood,
            mutation_matrix[gene_a],
            mutation_matrix[gene_b],
            drvr_bmr_pmfs if "DRVR" in gene_a else psngr_bmr_pmfs,
            drvr_bmr_pmfs if "DRVR" in gene_b else psngr_bmr_pmfs,
        )

        if "DRVR" in gene_a and "DRVR" in gene_b:
            threshold = llr_threshold_drvr_drvr
        elif ("DRVR" in gene_a and "PSNGR" in gene_b) or ("PSNGR" in gene_a and "DRVR" in gene_b):
            threshold = llr_threshold_drvr_psngr
        else:
            threshold = llr_threshold_psngr_psngr

        if llr > threshold:
            if interaction_type == "me" or interaction_type == "co":
                results_dialect["TP"] += 1
            else:
                results_dialect["FP"] += 1
        else:
            if interaction_type == "me" or interaction_type == "co":
                results_dialect["FN"] += 1
            else:
                results_dialect["TN"] += 1

        # Fishers
        gene_a_binarized = np.array([1 if x > 0 else 0 for x in mutation_matrix[gene_a]])
        gene_b_binarized = np.array([1 if x > 0 else 0 for x in mutation_matrix[gene_b]])
        fishers_result = fishers_module.fishers_exact_test(
            gene_a_binarized, gene_b_binarized, "me"
        )[1]

        if fishers_result < 0.05:
            if interaction_type == "me" or interaction_type == "co":
                results_fishers["TP"] += 1
            else:
                results_fishers["FP"] += 1
        else:
            if interaction_type == "me" or interaction_type == "co":
                results_fishers["FN"] += 1
            else:
                results_fishers["TN"] += 1

    # Compute sensitivity and false positive rate for Dialect
    sensitivity_dialect = results_dialect["TP"] / (results_dialect["TP"] + results_dialect["FN"])
    false_positive_rate_dialect = results_dialect["FP"] / (
        results_dialect["FP"] + results_dialect["TN"]
    )

    # Compute sensitivity and false positive rate for Fishers
    sensitivity_fishers = results_fishers["TP"] / (results_fishers["TP"] + results_fishers["FN"])
    false_positive_rate_fishers = results_fishers["FP"] / (
        results_fishers["FP"] + results_fishers["TN"]
    )

    # Plot results
    labels = ["Sensitivity", "False Positive Rate"]
    values_dialect = [sensitivity_dialect, false_positive_rate_dialect]
    values_fishers = [sensitivity_fishers, false_positive_rate_fishers]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35  # width of the bars

    x = np.arange(len(labels))
    rects1 = ax.bar(x - width / 2, values_dialect, width, label="Dialect", color="green")
    rects2 = ax.bar(x + width / 2, values_fishers, width, label="Fishers", color="blue")

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel("Metric")
    ax.set_ylabel("Rate")
    ax.set_title("Sensitivity and False Positive Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Function to add labels above bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(round(height, 2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


def main():
    # plot_fpr_vs_gene_a_pi_driver_and_passenger()
    # plot_sensitivity_vs_tau()
    small_matrix_simulation()
    # tune_significance_threshold()
    # variable_param = "num_samples"
    # variable_values = [500, 1000, 1500, 2000, 2500, 3000]
    # variable_param = "gene_b_nonzero_bmr"
    # variable_values = np.linspace(1e-6, 1e-5, 5)
    # nruns = 100
    # tau_10 = 0.05
    # tau_01 = 0.05
    # tau_11 = 0
    # interaction_type = "me"
    # gene_a_size = 1e4
    # gene_b_size = 1e4
    # fisher_results, dialect_results = run_simulation(
    #     nruns,
    #     tau_10,
    #     tau_01,
    #     tau_11,
    #     interaction_type,
    #     variable_param,
    #     variable_values,
    #     gene_a_size,
    #     gene_b_size,
    # )
    # print(fisher_results)
    # print(dialect_results)


def plot_observed_and_expected_mutations(cancer_subtype, top_k=20, sampling_count=100):
    base_path = "/Users/work/workspace/research/dialect/results/tcga_pancan_atlas_2018"
    cnt_mtx_fn = os.path.join(base_path, cancer_subtype, f"{cancer_subtype}_cbase_cnt_mtx.csv")
    bmr_fn = os.path.join(base_path, cancer_subtype, f"{cancer_subtype}_cbase_bmr_pmfs.csv")
    
    cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
    bmr_df = pd.read_csv(bmr_fn, index_col=0)
    bmrs_pmfs_dict = bmr_df.T.to_dict(orient="list")
    bmrs_pmfs_dict = {
        key: [x for x in bmrs_pmfs_dict[key] if not np.isnan(x)] for key in bmrs_pmfs_dict
    }
    bmrs_pmfs_dict = {
        key: [x / sum(bmrs_pmfs_dict[key]) for x in bmrs_pmfs_dict[key]]
        for key in bmrs_pmfs_dict
    }

    top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[:top_k]
    observed_counts_per_gene = cnt_mtx_df.sum(axis=0)

    nsamples = cnt_mtx_df.shape[0]

    # Generate expected counts by sampling multiple times and averaging
    expected_counts_per_gene = {gene: 0 for gene in top_k_genes}

    for _ in range(sampling_count):
        simulated_passengers_per_gene = {
            gene: np.random.choice(len(bmrs_pmfs_dict[gene]), nsamples, p=bmrs_pmfs_dict[gene])
            for gene in top_k_genes
        }
        for gene in top_k_genes:
            expected_counts_per_gene[gene] += simulated_passengers_per_gene[gene].sum()

    # Average the expected counts
    expected_counts_per_gene = {gene: count / sampling_count for gene, count in expected_counts_per_gene.items()}

    observed_counts = [observed_counts_per_gene[gene] for gene in top_k_genes]
    expected_counts = [expected_counts_per_gene[gene] for gene in top_k_genes]

    x = np.arange(len(top_k_genes))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(
        x - width / 2,
        observed_counts,
        width,
        label="Observed",
        alpha=0.8,
        color="blue",
    )
    bars2 = ax.bar(
        x + width / 2,
        expected_counts,
        width,
        label="Expected",
        alpha=0.8,
        color="red",
    )

    # Increase label sizes
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("Genes", fontsize=18)
    ax.set_ylabel("Mutation Counts", fontsize=18)
    ax.set_title(f"Observed & Expected Mutation Counts per Gene w/ CBaSE BMRs ({cancer_subtype})", fontsize=20)

    # Set the x-ticks and x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(top_k_genes, rotation=45, ha="right")

    # Add legends for observed and expected counts
    custom_lines = [
        plt.Line2D([0], [0], color="blue", lw=4, label="Observed"),
        plt.Line2D([0], [0], color="red", lw=4, label="Expected")
    ]
    ax.legend(handles=custom_lines, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"figures/{cancer_subtype}_observed_vs_expected.svg")




def plot_observed_and_expected_mutations_by_gene_type(cancer_subtype, driver_genes, passenger_genes, sampling_count=100):
    base_path = "/Users/work/workspace/research/dialect/results/tcga_pancan_atlas_2018"
    cnt_mtx_fn = os.path.join(base_path, cancer_subtype, f"{cancer_subtype}_cbase_cnt_mtx.csv")
    bmr_fn = os.path.join(base_path, cancer_subtype, f"{cancer_subtype}_cbase_bmr_pmfs.csv")
    
    cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
    bmr_df = pd.read_csv(bmr_fn, index_col=0)
    bmrs_pmfs_dict = bmr_df.T.to_dict(orient="list")
    bmrs_pmfs_dict = {
        key: [x for x in bmrs_pmfs_dict[key] if not np.isnan(x)] for key in bmrs_pmfs_dict
    }
    bmrs_pmfs_dict = {
        key: [x / sum(bmrs_pmfs_dict[key]) for x in bmrs_pmfs_dict[key]]
        for key in bmrs_pmfs_dict
    }

    all_genes = driver_genes + passenger_genes
    observed_counts_per_gene = cnt_mtx_df.sum(axis=0)

    nsamples = cnt_mtx_df.shape[0]

    # Generate expected counts by sampling multiple times and averaging
    expected_counts_per_gene = {gene: 0 for gene in all_genes}

    for _ in range(sampling_count):
        simulated_passengers_per_gene = {
            gene: np.random.choice(len(bmrs_pmfs_dict[gene]), nsamples, p=bmrs_pmfs_dict[gene])
            for gene in all_genes
        }
        for gene in all_genes:
            expected_counts_per_gene[gene] += simulated_passengers_per_gene[gene].sum()

    # Average the expected counts
    expected_counts_per_gene = {gene: count / sampling_count for gene, count in expected_counts_per_gene.items()}

    observed_counts = [observed_counts_per_gene[gene] for gene in all_genes]
    expected_counts = [expected_counts_per_gene[gene] for gene in all_genes]

    x = np.arange(len(all_genes))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(
        x - width / 2,
        observed_counts,
        width,
        label="Observed",
        alpha=0.8,
        color="blue",
    )
    bars2 = ax.bar(
        x + width / 2,
        expected_counts,
        width,
        label="Expected",
        alpha=0.8,
        color="red",
    )

    # Increase label sizes
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlabel("Genes", fontsize=18)
    ax.set_ylabel("Mutation Counts", fontsize=18)
    ax.set_title(f"Observed & Expected Mutation Counts per Gene w/ CBaSE BMRs ({cancer_subtype})", fontsize=20)

    # Set the x-ticks and x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(all_genes, rotation=45, ha="right")

    # Add legends for observed and expected counts
    custom_lines = [
        plt.Line2D([0], [0], color="blue", lw=4, label="Observed"),
        plt.Line2D([0], [0], color="red", lw=4, label="Expected")
    ]
    ax.legend(handles=custom_lines, loc="upper right")

    plt.tight_layout()
    plt.show()


def new_small_matrix_simulation(driver_genes, passenger_genes, cancer_subtype):
    dialect_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "dialect")
    sys.path.append(dialect_dir)
    dialect_module = importlib.import_module("core")
    fishers_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers")
    sys.path.append(fishers_dir)
    fishers_module = importlib.import_module("run_fishers")

    base_path = "/Users/work/workspace/research/dialect/results/tcga_pancan_atlas_2018"
    bmr_fn = os.path.join(base_path, cancer_subtype, f"{cancer_subtype}_cbase_bmr_pmfs.csv")
    bmr_df = pd.read_csv(bmr_fn, index_col=0)
    bmrs_pmfs_dict = bmr_df.T.to_dict(orient="list")
    bmrs_pmfs_dict = {
        key: [x for x in bmrs_pmfs_dict[key] if not np.isnan(x)] for key in bmrs_pmfs_dict
    }
    bmrs_pmfs_dict = {
        key: [x / sum(bmrs_pmfs_dict[key]) for x in bmrs_pmfs_dict[key]]
        for key in bmrs_pmfs_dict
    }

    drvr_bmr_pmfs = {gene: bmrs_pmfs_dict[gene] for gene in driver_genes}
    psngr_bmr_pmfs = {gene: bmrs_pmfs_dict[gene] for gene in passenger_genes}

    # Initialize mutation matrix
    nsamples = 1000
    mutation_matrix = pd.DataFrame(0, index=range(nsamples), columns=driver_genes + passenger_genes)

    # Generate mutations for driver genes
    mut_excl_pairs = [(driver_genes[0], driver_genes[1])]
    co_occ_pairs = [(driver_genes[2], driver_genes[3])]
    print('Mutually exclusive pairs:', mut_excl_pairs)
    print('Co-occurring pairs:', co_occ_pairs)

    # Mutually exclusive pair
    tau_10 = tau_01 = 0.05
    tau_11 = 0
    gene_a, gene_b = mut_excl_pairs[0]
    gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
        nsamples, tau_10, tau_01, tau_11
    )
    mutation_matrix[gene_a] += gene_a_driver_mutations
    mutation_matrix[gene_b] += gene_b_driver_mutations

    # Co-occurring pair
    tau_11 = 0.05
    tau_10 = tau_01 = 0
    gene_a, gene_b = co_occ_pairs[0]
    gene_a_driver_mutations, gene_b_driver_mutations = generate_pairwise_driver_mutations(
        nsamples, tau_10, tau_01, tau_11
    )
    mutation_matrix[gene_a] += gene_a_driver_mutations
    mutation_matrix[gene_b] += gene_b_driver_mutations

    # Generate mutations for passenger genes
    for gene in passenger_genes:
        bmr_pmfs = np.array(psngr_bmr_pmfs[gene])
        mutation_matrix[gene] += np.random.choice(len(bmr_pmfs), nsamples, p=bmr_pmfs)

    mutation_matrix = mutation_matrix.astype(int)

    # Generate null distributions and determine LLR thresholds
    null_llrs_drvr_drvr = generate_null_distribution(nsamples, 100, drvr_bmr_pmfs[driver_genes[0]], drvr_bmr_pmfs[driver_genes[1]])
    llr_threshold_drvr_drvr = determine_llr_threshold(null_llrs_drvr_drvr, false_positive_rate=0.01)

    null_llrs_drvr_psngr = generate_null_distribution(nsamples, 100, drvr_bmr_pmfs[driver_genes[0]], psngr_bmr_pmfs[passenger_genes[0]])
    llr_threshold_drvr_psngr = determine_llr_threshold(null_llrs_drvr_psngr, false_positive_rate=0.01)

    null_llrs_psngr_psngr = generate_null_distribution(nsamples, 100, psngr_bmr_pmfs[passenger_genes[0]], psngr_bmr_pmfs[passenger_genes[1]])
    llr_threshold_psngr_psngr = determine_llr_threshold(null_llrs_psngr_psngr, false_positive_rate=0.01)

    print(f"LLR Threshold (DRVR-DRVR): {llr_threshold_drvr_drvr}")
    print(f"LLR Threshold (DRVR-PSNGR): {llr_threshold_drvr_psngr}")
    print(f"LLR Threshold (PSNGR-PSNGR): {llr_threshold_psngr_psngr}")

    def determine_interaction_type(gene_a, gene_b):
        if (gene_a, gene_b) in mut_excl_pairs or (gene_b, gene_a) in mut_excl_pairs:
            return "me"
        elif (gene_a, gene_b) in co_occ_pairs or (gene_b, gene_a) in co_occ_pairs:
            return "co"
        else:
            return "None"

    results_list = []

    gene_pairs = list(combinations(driver_genes + passenger_genes, 2))
    for gene_a, gene_b in tqdm(gene_pairs):
        interaction_type = determine_interaction_type(gene_a, gene_b)
        gene_a_pi, gene_a_log_likelihood, _ = dialect_module.dialect_singleton(
            mutation_matrix[gene_a], drvr_bmr_pmfs[gene_a] if gene_a in driver_genes else psngr_bmr_pmfs[gene_a]
        )
        gene_b_pi, gene_b_log_likelihood, _ = dialect_module.dialect_singleton(
            mutation_matrix[gene_b], drvr_bmr_pmfs[gene_b] if gene_b in driver_genes else psngr_bmr_pmfs[gene_b]
        )
        _, _, llr, log_odds = dialect_module.dialect_pairwise(
            gene_a_pi,
            gene_b_pi,
            gene_a_log_likelihood,
            gene_b_log_likelihood,
            mutation_matrix[gene_a],
            mutation_matrix[gene_b],
            drvr_bmr_pmfs[gene_a] if gene_a in driver_genes else psngr_bmr_pmfs[gene_a],
            drvr_bmr_pmfs[gene_b] if gene_b in driver_genes else psngr_bmr_pmfs[gene_b],
        )

        if gene_a in driver_genes and gene_b in driver_genes:
            threshold = llr_threshold_drvr_drvr
        elif (gene_a in driver_genes and gene_b in passenger_genes) or (gene_b in driver_genes and gene_a in passenger_genes):
            threshold = llr_threshold_drvr_psngr
        else:
            threshold = llr_threshold_psngr_psngr

        if log_odds > 0:
            if llr > threshold:
                if interaction_type == "me":
                    dialect_classification = "TP"
                else:
                    dialect_classification = "FP"
            else:
                if interaction_type == "me":
                    dialect_classification = "FN"
                else:
                    dialect_classification = "TN"
        else:
            if llr > threshold:
                if interaction_type == "co":
                    dialect_classification = "TP"
                else:
                    dialect_classification = "FP"
            else:
                if interaction_type == "co":
                    dialect_classification = "FN"
                else:
                    dialect_classification = "TN"

        # Fishers
        gene_a_binarized = np.array([1 if x > 0 else 0 for x in mutation_matrix[gene_a]])
        gene_b_binarized = np.array([1 if x > 0 else 0 for x in mutation_matrix[gene_b]])
        me_fisher_result = fishers_module.fishers_exact_test(
            gene_a_binarized, gene_b_binarized, "me"
        )[1]
        co_fisher_result = fishers_module.fishers_exact_test(
            gene_a_binarized, gene_b_binarized, "co"
        )[1]

        fisher_me_classification = "TP" if me_fisher_result < 0.05 and interaction_type == "me" else \
                                    "FP" if me_fisher_result < 0.05 and interaction_type == "None" else \
                                    "FN" if me_fisher_result >= 0.05 and interaction_type == "me" else "TN"
        
        fisher_co_classification = "TP" if co_fisher_result < 0.05 and interaction_type == "co" else \
                                    "FP" if co_fisher_result < 0.05 and interaction_type == "None" else \
                                    "FN" if co_fisher_result >= 0.05 and interaction_type == "co" else "TN"

        results_list.append([gene_a, gene_b, llr, threshold, log_odds, me_fisher_result, co_fisher_result,
                             dialect_classification, fisher_me_classification, fisher_co_classification])

    results_df = pd.DataFrame(results_list, columns=["gene_a", "gene_b", "dialect_llr",
                                                     "dialect_threshold", "dialect_log_odds",
                                                     "fisher_me_pval", "fisher_co_pval",
                                                     "dialect_classification",
                                                     "fisher_me_classification", "fisher_co_classification"])
    results_df.to_csv("simulation_results.csv", index=False)
    return results_df


brca_driver_genes = ['PIK3CA_M', 'TP53_M', 'TP53_N', 'CDH1_N']
brca_passenger_genes = ['TTN_M', 'MUC16_M', 'RYR2_M', 'FLG_M', 'SYNE1_M', 'HMCN1_M', 'USH2A_M',
                        'DMD_M', 'ZFHX4_M', 'OBSCN_M', 'SYNE2_M', 'MUC17_M', 'KMT2C_M',
                        'SPTA1_M', 'CSMD3_M', 'FAT3_M', 'RYR3_M', 'HUWE1_M', 'DST_M', 'CACNA1E_M']

if __name__ == "__main__":
    dialect_sensitivities, dialect_fprs = [], []
    fisher_sensitivities, fisher_fprs = [], []
    for _ in tqdm(range(10)):
        results_df = new_small_matrix_simulation(brca_driver_genes, brca_passenger_genes, 'BRCA')
        dialect_tp = results_df[(results_df['dialect_classification'] == 'TP')]
        dialect_fn = results_df[(results_df['dialect_classification'] == 'FN')]
        dialect_fp = results_df[(results_df['dialect_classification'] == 'FP')]
        dialect_tn = results_df[(results_df['dialect_classification'] == 'TN')]
        dialect_sensitivity = dialect_tp.shape[0] / (dialect_tp.shape[0] + dialect_fn.shape[0])
        dialect_fpr = dialect_fp.shape[0] / (dialect_fp.shape[0] + dialect_tn.shape[0])

        fisher_tp = results_df[(results_df['fisher_me_classification'] == 'TP') | (results_df['fisher_co_classification'] == 'TP')]
        fisher_fn = results_df[(results_df['fisher_me_classification'] == 'FN') & (results_df['fisher_co_classification'] == 'FN')]
        fisher_fp = results_df[(results_df['fisher_me_classification'] == 'FP') | (results_df['fisher_co_classification'] == 'FP')]
        fisher_tn = results_df[(results_df['fisher_me_classification'] == 'TN') & (results_df['fisher_co_classification'] == 'TN')]
        fisher_sensitivity = fisher_tp.shape[0] / (fisher_tp.shape[0] + fisher_fn.shape[0])
        fisher_fpr = fisher_fp.shape[0] / (fisher_fp.shape[0] + fisher_tn.shape[0])

        dialect_sensitivities.append(dialect_sensitivity)
        dialect_fprs.append(dialect_fpr)
        fisher_sensitivities.append(fisher_sensitivity)
        fisher_fprs.append(fisher_fpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dialect_fprs, dialect_sensitivities, label='Dialect', marker='o')
    plt.plot(fisher_fprs, fisher_sensitivities, label='Fisher\'s', marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity vs. False Positive Rate')
    plt.legend()
    plt.show()


    # plot_observed_and_expected_mutations_by_gene_type("BRCA", brca_driver_genes, brca_passenger_genes, sampling_count=100)

    # for cancer_subtype in ['AML', 'BRCA', 'LUAD', 'LUSC', 'GBM', 'UCEC']:
        # plot_observed_and_expected_mutations(cancer_subtype, top_k=20, sampling_count=100)
    # main()


# def run_simulation(cnt_mtx_fn, bmr_fn, dout, top_k=20):
#     assert os.path.exists(cnt_mtx_fn), f"File not found: {cnt_mtx_fn}"
#     assert os.path.exists(bmr_fn), f"File not found: {bmr_fn}"
#     cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
#     bmr_df = pd.read_csv(bmr_fn, index_col=0)
#     top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[:top_k]

#     ###### MANUAL FOR AML ######
#     drivers = [
#         "DNMT3A_M",
#         "IDH2_M",
#         "FLT3_M",
#         "IDH1_M",
#         "TP53_M",
#         "NRAS_M",
#         "RUNX1_M",
#         "PTPN11_M",
#         "SMC1A_M",
#         "KIT_M",
#     ]
#     passengers = ["TTN_M", "MUC16_M"]

#     # create a real sums dictionary of the counts of the drivers from the cnt_mtx_df
#     total_cnts_dict = cnt_mtx_df.sum(axis=0)

#     bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
#     bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
#     bmr_dict = {key: [x / sum(bmr_dict[key]) for x in bmr_dict[key]] for key in bmr_dict}
#     driver_bmr_muts = {
#         key: np.random.choice(len(bmr_dict[key]), cnt_mtx_df.shape[0], p=bmr_dict[key])
#         for key in drivers
#     }
#     # passenger_bmr_muts = {
#     #     key: np.random.choice(len(bmr_dict[key]), cnt_mtx_df.shape[0], p=bmr_dict[key])
#     #     for key in passengers
#     # }
#     exp_driver_probs = {
#         key: int(total_cnts_dict[key] - driver_bmr_muts[key].sum()) / cnt_mtx_df.shape[0]
#         for key in driver_bmr_muts
#     }
#     print(exp_driver_probs)

#     # Generate counts for the passengers and drivers
#     simulation_genes = ["DNMT3A_M", "IDH2_M", "SMC1A_M", "KIT_M", "TTN_M", "MUC16_M"]
#     driver_probs = {
#         "DNMT3A_M": 0.25,
#         "IDH2_M": 0.15,
#         "SMC1A_M": 0.05,
#         "KIT_M": 0.05,
#         "TTN_M": 0,
#         "MUC16_M": 0,
#     }
#     pair_one = ["DNMT3A_M", "IDH2_M"]
#     pair_two = ["SMC1A_M", "KIT_M"]
#     nsamples = 1000
#     dnmt_drivers, idh_drivers = generate_pairwise_driver_mutations(  # perfect me
#         nsamples, driver_probs[pair_one[0]], driver_probs[pair_one[1]], 0
#     )
#     smc_drivers, kit_drivers = generate_pairwise_driver_mutations(  # perfect co
#         nsamples, 0, 0, driver_probs[pair_two[0]]
#     )
#     dnmt_psngrs = np.random.choice(len(bmr_dict["DNMT3A_M"]), nsamples, p=bmr_dict["DNMT3A_M"])
#     idh_psngrs = np.random.choice(len(bmr_dict["IDH1_M"]), nsamples, p=bmr_dict["IDH1_M"])
#     smc_psngrs = np.random.choice(len(bmr_dict["SMC1A_M"]), nsamples, p=bmr_dict["SMC1A_M"])
#     kit_psngrs = np.random.choice(len(bmr_dict["KIT_M"]), nsamples, p=bmr_dict["KIT_M"])
#     ttn_psngrs = np.random.choice(len(bmr_dict["TTN_M"]), nsamples, p=bmr_dict["TTN_M"])
#     muc_psngrs = np.random.choice(len(bmr_dict["MUC16_M"]), nsamples, p=bmr_dict["MUC16_M"])

#     dnm_muts = dnmt_drivers + dnmt_psngrs
#     idh_muts = idh_drivers + idh_psngrs
#     smc_muts = smc_drivers + smc_psngrs
#     kit_muts = kit_drivers + kit_psngrs
#     ttn_muts = ttn_psngrs
#     muc_muts = muc_psngrs

#     data = {
#         "DNMT3A_M": dnm_muts,
#         "IDH2_M": idh_muts,
#         "SMC1A_M": smc_muts,
#         "KIT_M": kit_muts,
#         "TTN_M": ttn_muts,
#         "MUC16_M": muc_muts,
#     }
#     sim_cnt_mtx = pd.DataFrame(data).astype(int)
#     sim_fout = os.path.join(dout, "sim_cnt_mtx.csv")
#     sim_cnt_mtx.to_csv(sim_fout, index=True)
#     return sim_fout

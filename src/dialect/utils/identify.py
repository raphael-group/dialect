import logging
import numpy as np
import pandas as pd

from itertools import combinations
from dialect.utils.helpers import *
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
def verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    check_file_exists(cnt_mtx)
    check_file_exists(bmr_pmfs)


def load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
    return cnt_df, bmr_dict


def create_single_gene_table(genes, output_path):
    """
    Create a table of single-gene test results and save it to a CSV file.

    :param genes: (list) A list of Gene objects.
    :param output_path: (str) Path to save the CSV file.
    """
    results = []
    for gene in genes:
        log_odds_ratio = gene.compute_log_odds_ratio(gene.pi)
        likelihood_ratio = gene.compute_likelihood_ratio(gene.pi)
        observed_mutations = sum(gene.counts)
        expected_mutations = gene.calculate_expected_mutations()
        obs_minus_exp_mutations = observed_mutations - expected_mutations

        results.append(
            {
                "Gene Name": gene.name,
                "Pi": gene.pi,
                "Log Odds Ratio": log_odds_ratio,
                "Likelihood Ratio": likelihood_ratio,
                "Observed Mutations": observed_mutations,
                "Expected Mutations": expected_mutations,
                "Obs. - Exp. Mutations": obs_minus_exp_mutations,
            }
        )
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Single-gene results saved to {output_path}")


def create_pairwise_results_table(interactions, output_path):
    """
    Create a table of pairwise interaction test results and save it to a CSV file.

    :param interactions: (list) A list of Interaction objects.
    :param output_path: (str) Path to save the CSV file.
    """
    results = []
    for interaction in interactions:
        taus = (
            interaction.tau_00,
            interaction.tau_01,
            interaction.tau_10,
            interaction.tau_11,
        )
        rho = interaction.compute_rho(taus)
        log_odds_ratio = interaction.compute_log_odds_ratio(taus)
        likelihood_ratio = interaction.compute_likelihood_ratio(taus)
        cm = interaction.compute_contingency_table()

        results.append(
            {
                "Gene A": interaction.gene_a.name,
                "Gene B": interaction.gene_b.name,
                "Tau_00": interaction.tau_00,
                "Tau_10": interaction.tau_10,
                "Tau_01": interaction.tau_01,
                "Tau_11": interaction.tau_11,
                "_00_": cm[0, 0],
                "_10_": cm[1, 0],
                "_01_": cm[0, 1],
                "_11_": cm[1, 1],
                "Tau_1X": interaction.tau_10 + interaction.tau_11,
                "Tau_X1": interaction.tau_01 + interaction.tau_11,
                "Rho": rho,
                "Log Odds Ratio": log_odds_ratio,
                "Likelihood Ratio": likelihood_ratio,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Pairwise results saved to {output_path}")


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def identify_pairwise_interactions(cnt_mtx, bmr_pmfs, out, k):
    """
    Main function to identify pairwise interactions between genetic drivers in tumors using DIALECT.
    ! Work in Progress

    @param cnt_mtx (str): Path to the input count matrix file containing mutation data.
    @param bmr_pmfs (str): Path to the file with background mutation rate (BMR) distributions.
    @param out (str): Directory where outputs will be saved.
    @param k (int): Top k genes according to count of mutations will be used.
    """
    logging.info("Identifying pairwise interactions using DIALECT")
    verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)
    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)

    if k <= 0:
        logging.error("k must be a positive integer")
        raise ValueError("k must be a positive integer")

    genes = []
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].values
        bmr_pmf_arr = bmr_dict.get(gene_name, None)
        if bmr_pmf_arr is None:
            raise ValueError(f"No BMR PMF found for gene {gene_name}")
        bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
        genes.append(Gene(name=gene_name, counts=counts, bmr_pmf=bmr_pmf))
    logging.info(f"Initialized {len(genes)} Gene objects.")

    logging.info("Running EM to estimate pi for single genes...")
    for gene in genes:
        gene.estimate_pi_with_em_from_scratch()
        logging.info(f"Estimated pi of {gene.pi} for gene {gene.name}")
    logging.info("Finished estimating pi for single genes.")

    interactions = []
    top_genes = sorted(genes, key=lambda x: sum(x.counts), reverse=True)[:k]
    for gene_a, gene_b in combinations(top_genes, 2):
        interactions.append(Interaction(gene_a, gene_b))
    logging.info(f"Initialized {len(interactions)} Interaction objects.")

    logging.info("Running EM to estimate pi for pairwise interactions...")
    for interaction in interactions:
        interaction.estimate_tau_with_em_from_scratch()
        logging.info(
            f"Estimated tau_00={interaction.tau_00}, tau_01={interaction.tau_01}, tau_10={interaction.tau_10}, tau_11={interaction.tau_11} for interaction {interaction.name}"
        )
    logging.info("Finished estimating tau for pairwise interactions.")

    logging.info("Creating single-gene results table...")
    create_single_gene_table(genes, f"{out}/single_gene_results.csv")
    logging.info("Finished creating single-gene results table.")

    # TODO: Check log likelihood plots for pairwise interactions
    # ? Are the plots convex; do we need multiple EM initializations
    logging.info("Creating pairwise interaction results table...")
    create_pairwise_results_table(
        interactions, f"{out}/pairwise_interaction_results.csv"
    )
    logging.info("Finished creating pairwise interaction results table.")
    # TODO: Add methods to get set of co-occurring samples

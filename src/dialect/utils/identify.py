import logging
import pandas as pd

from dialect.utils.helpers import *


# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
def verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    check_file_exists(cnt_mtx)
    check_file_exists(bmr_pmfs)


def create_single_gene_results(genes, output_path):
    """
    Create a table of single-gene test results and save it to a CSV file.

    :param genes: (list) A list of Gene objects.
    :param output_path: (str) Path to save the CSV file.
    """
    logging.info("Creating single-gene results table...")
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
    logging.info("Finished creating single-gene results table.")


def create_pairwise_results(interactions, output_path):
    """
    Create a table of pairwise interaction test results and save it to a CSV file.

    :param interactions: (list) A list of Interaction objects.
    :param output_path: (str) Path to save the CSV file.
    """
    logging.info("Creating pairwise interaction results table...")
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
        fisher_me_pval, fisher_co_pval = interaction.compute_fisher_pvalues()

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
                "Fisher's ME P-Val": fisher_me_pval,
                "Fisher's CO P-Val": fisher_co_pval,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Pairwise results saved to {output_path}")
    logging.info("Finished creating pairwise interaction results table.")


def estimate_pi_for_each_gene(genes, single_gene_output_file=None):
    """
    Set pi values from file if it exists; otherwise, estimate using EM.
    """
    logging.info("Running EM to estimate pi for single genes...")
    pi_from_file = {}
    print(single_gene_output_file)
    print(os.path.exists(single_gene_output_file))
    if single_gene_output_file and os.path.exists(single_gene_output_file):
        try:
            pi_from_file = (
                pd.read_csv(single_gene_output_file)
                .set_index("Gene Name")["Pi"]
                .to_dict()
            )
        except Exception as e:
            logging.warning(f"Error reading {single_gene_output_file}: {e}")

    for gene in genes:
        if gene.name in pi_from_file:
            gene.pi = pi_from_file[gene.name]
            logging.info(f"Loaded pi={gene.pi} for {gene.name} from file")
        else:
            gene.estimate_pi_with_em_from_scratch()
            logging.info(f"Estimated pi={gene.pi} for {gene.name}")
    logging.info("Finished estimating pi for single genes.")


def estimate_taus_for_each_interaction(interactions):
    logging.info("Running EM to estimate pi for pairwise interactions...")
    for interaction in interactions:
        interaction.estimate_tau_with_em_from_scratch()
        logging.info(
            f"Estimated tau_00={interaction.tau_00}, tau_01={interaction.tau_01}, tau_10={interaction.tau_10}, tau_11={interaction.tau_11} for interaction {interaction.name}"
        )
    logging.info("Finished estimating tau for pairwise interactions.")


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
        raise ValueError("k must be a positive integer")

    genes = initialize_gene_objects(cnt_df, bmr_dict)
    estimate_pi_for_each_gene(genes.values(), f"{out}/single_gene_results.csv")
    interactions = initialize_interaction_objects(k, genes.values())
    estimate_taus_for_each_interaction(interactions)
    create_single_gene_results(genes.values(), f"{out}/single_gene_results.csv")

    # TODO: Check log likelihood plots for pairwise interactions
    # ? Are the plots convex; do we need multiple EM initializations
    create_pairwise_results(interactions, f"{out}/pairwise_interaction_results.csv")

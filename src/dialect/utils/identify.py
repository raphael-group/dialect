import logging
import pandas as pd

from dialect.utils.helpers import *

# TODO: Create essential and verbose logging info for all methods


# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
def save_cbase_stats_to_gene_objects(genes, cbase_stats):
    """
    Save CBaSE Phi statistics to gene objects by updating their attributes.

    :param genes (dict): A dictionary of gene objects keyed by their names.
    :param cbase_stats (pd.DataFrame): DataFrame containing CBaSE result statistics.
    :return (bool): True if successful, False otherwise.
    """
    logging.info("Saving CBaSE phi statistic to gene objects...")

    if cbase_stats is None or cbase_stats.empty:
        logging.info(
            "No CBaSE result file provided or the file is empty. "
            "Please provide a valid path to the file with the --cbase_stats flag."
        )
        return False

    missense_gene_to_positive_selection_phi = {
        f"{row['gene']}_M": row["phi_m_pos_or_p(m=0|s)"]
        for _, row in cbase_stats.iterrows()
    }

    nonsense_gene_to_positive_selection_phi = {
        f"{row['gene']}_N": row["phi_k_pos_or_p(k=0|s)"]
        for _, row in cbase_stats.iterrows()
    }

    gene_to_positive_selection_phi = {
        **missense_gene_to_positive_selection_phi,
        **nonsense_gene_to_positive_selection_phi,
    }

    for name, gene in genes.items():
        if name not in gene_to_positive_selection_phi:
            raise ValueError(f"Gene {name} not found in the CBaSE results file.")
        gene.cbase_phi = gene_to_positive_selection_phi[name]

    logging.info("Finished saving CBaSE phi statistic to gene objects.")
    return True


def create_single_gene_results(genes, output_path, cbase_phi_vals_present):
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
        cbase_phi = gene.cbase_phi

        results.append(
            {
                "Gene Name": gene.name,
                "Pi": gene.pi,
                "Log Odds Ratio": log_odds_ratio,
                "Likelihood Ratio": likelihood_ratio,
                "Observed Mutations": observed_mutations,
                "Expected Mutations": expected_mutations,
                "Obs. - Exp. Mutations": obs_minus_exp_mutations,
                "CBaSE Pos. Sel. Phi": cbase_phi,
            }
        )
    results_df = pd.DataFrame(results)
    if not cbase_phi_vals_present:
        results_df = results_df.drop(columns=["CBaSE Pos. Sel. Phi"])
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
def identify_pairwise_interactions(
    cnt_mtx,
    bmr_pmfs,
    out,
    k,
    cbase_stats,  # TODO separate this out from here
):
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

    # TODO: change these names to be more concise: dialect_gene_results.csv
    # TODO: separate out CBaSE stats into file CBaSE_gene_results.csv
    # TODO: create functionality to run MutSigCV and save results into MutSigCV_gene_results.csv
    # TODO: rename pairwise to be DIALECT_ixn_results.csv
    single_gene_fout = f"{out}/single_gene_results.csv"
    pairwise_interaction_fout = f"{out}/pairwise_interaction_results.csv"

    genes = initialize_gene_objects(cnt_df, bmr_dict)
    estimate_pi_for_each_gene(genes.values(), single_gene_fout)
    _, interactions = initialize_interaction_objects(k, genes.values())
    estimate_taus_for_each_interaction(interactions)

    cbase_phi_vals_present = save_cbase_stats_to_gene_objects(genes, cbase_stats)
    create_single_gene_results(genes.values(), single_gene_fout, cbase_phi_vals_present)
    create_pairwise_results(interactions, pairwise_interaction_fout)

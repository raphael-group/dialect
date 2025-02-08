"""TODO: Add docstring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dialect.utils.helpers import (
    initialize_gene_objects,
    initialize_interaction_objects,
    load_cnt_mtx_and_bmr_pmfs,
    verify_cnt_mtx_and_bmr_pmfs,
)


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def save_cbase_stats_to_gene_objects(genes: dict, cbase_stats: pd.DataFrame) -> bool:
    """TODO: Add docstring."""
    if cbase_stats is None or cbase_stats.empty:
        return False

    missense_gene_to_positive_selection_phi = {
        f"{row['gene']}_M": row["phi_m_pos_or_p(m=0|s)"]
        for _, row in cbase_stats.iterrows()
    }
    missense_gene_to_positive_selection_p = {
        f"{row['gene']}_M": row["p_phi_m_pos"] for _, row in cbase_stats.iterrows()
    }

    nonsense_gene_to_positive_selection_phi = {
        f"{row['gene']}_N": row["phi_k_pos_or_p(k=0|s)"]
        for _, row in cbase_stats.iterrows()
    }
    nonsense_gene_to_positive_selection_p = {
        f"{row['gene']}_N": row["p_phi_k_pos"] for _, row in cbase_stats.iterrows()
    }

    gene_to_positive_selection_phi = {
        **missense_gene_to_positive_selection_phi,
        **nonsense_gene_to_positive_selection_phi,
    }
    gene_to_positive_select_p = {
        **missense_gene_to_positive_selection_p,
        **nonsense_gene_to_positive_selection_p,
    }

    for name, gene in genes.items():
        if name not in gene_to_positive_selection_phi:
            msg = f"Gene {name} not found in the CBaSE results file."
            raise ValueError(
                msg,
            )
        gene.cbase_phi = gene_to_positive_selection_phi[name]
        gene.cbase_p = gene_to_positive_select_p[name]

    return True


def create_single_gene_results(
    genes: list,
    output_path: str,
    cbase_phi_vals_present: bool,
) -> None:
    """TODO: Add docstring."""
    results = []
    for gene in genes:
        log_odds_ratio = gene.compute_log_odds_ratio(gene.pi)
        likelihood_ratio = gene.compute_likelihood_ratio(gene.pi)
        observed_mutations = sum(gene.counts)
        expected_mutations = gene.calculate_expected_mutations()
        obs_minus_exp_mutations = observed_mutations - expected_mutations
        cbase_phi = gene.cbase_phi
        cbase_p = gene.cbase_p

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
                "CBaSE Pos. Sel. P-Val": cbase_p,
            },
        )
    results_df = pd.DataFrame(results)
    if not cbase_phi_vals_present:
        results_df = results_df.drop(columns=["CBaSE Pos. Sel. Phi"])
    results_df.to_csv(output_path, index=False)


def create_pairwise_results(interactions: list, output_path: str) -> None:
    """TODO: Add docstring."""
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
        wald_statistic = interaction.compute_wald_statistic(taus)
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
                "Wald Statistic": wald_statistic,
            },
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def estimate_pi_for_each_gene(
    genes: list,
    single_gene_output_file: str | None = None,
) -> None:
    """TODO: Add docstring."""
    pi_from_file = {}
    if single_gene_output_file and Path(single_gene_output_file).exists():
        try:
            pi_from_file = (
                pd.read_csv(single_gene_output_file)
                .set_index("Gene Name")["Pi"]
                .to_dict()
            )
        except FileNotFoundError:
            sys.exit(1)

    for gene in genes:
        if gene.name in pi_from_file:
            gene.pi = pi_from_file[gene.name]
        else:
            gene.estimate_pi_with_em_from_scratch()


def estimate_taus_for_each_interaction(interactions: list) -> None:
    """TODO: Add docstring."""
    for interaction in interactions:
        interaction.estimate_tau_with_em_from_scratch()


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def identify_pairwise_interactions(
    cnt_mtx: str,
    bmr_pmfs: str,
    out: str,
    k: int,
    cbase_stats: pd.DataFrame | None,
) -> None:
    """TODO: Add docstring."""
    verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)
    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)

    if k <= 0:
        msg = "k must be a positive integer"
        raise ValueError(msg)

    single_gene_fout = f"{out}/single_gene_results.csv"
    pairwise_interaction_fout = f"{out}/pairwise_interaction_results.csv"

    genes = initialize_gene_objects(cnt_df, bmr_dict)
    estimate_pi_for_each_gene(genes.values(), single_gene_fout)
    _, interactions = initialize_interaction_objects(k, genes.values())
    estimate_taus_for_each_interaction(interactions)

    cbase_phi_vals_present = save_cbase_stats_to_gene_objects(
        genes,
        cbase_stats,
    )
    create_single_gene_results(
        genes.values(),
        single_gene_fout,
        cbase_phi_vals_present,
    )
    create_pairwise_results(interactions, pairwise_interaction_fout)

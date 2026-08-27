"""TODO: Add docstring."""

from __future__ import annotations

import pandas as pd

from dialect.data.cohort import MutationCohort
from dialect.models.assembly import (
    initialize_gene_objects,
    initialize_interaction_objects,
    save_cbase_stats_to_gene_objects,
)
from dialect.models.interaction import LRT_CONTRACT, PAIR_FIT_CONTRACT


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
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
                "MLE Converged": gene.mle_converged,
                "MLE Iterations": gene.mle_iterations,
                "MLE Log Likelihood": gene.mle_log_likelihood,
                "Single-Gene LRT Status": gene.likelihood_ratio_status,
                "LRT Contract": LRT_CONTRACT,
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
        log_odds_ratio = interaction.compute_log_odds_ratio(taus)
        wald_statistic = interaction.compute_wald_statistic(taus)
        likelihood_ratio = interaction.likelihood_ratio
        if likelihood_ratio is None:
            likelihood_ratio = interaction.compute_likelihood_ratio(taus)
        rho = interaction.compute_rho_for_direction(taus, likelihood_ratio)
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
                "Fit Algorithm": interaction.fit_algorithm,
                "Fit Converged": interaction.fit_converged,
                "Fit Iterations": interaction.fit_iterations,
                "Fit Last LL Gain": interaction.fit_last_log_likelihood_gain,
                "Fit Fixed-Point Residual": interaction.fit_fixed_point_residual,
                "Fit KKT Residual": interaction.fit_kkt_residual,
                "Pair Fit Contract": PAIR_FIT_CONTRACT,
                "Null Log Likelihood": interaction.null_log_likelihood,
                "Alternative Log Likelihood": interaction.alternative_log_likelihood,
                "LRT Contract": LRT_CONTRACT,
            },
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def estimate_pi_for_each_gene(
    genes: list,
    single_gene_output_file: str | None = None,
) -> None:
    """Fit every constrained-null marginal MLE against the current inputs.

    Historical CSV values cannot be reused safely because they do not identify the
    likelihood/support contract or the count/BMR inputs that produced them.
    """
    del single_gene_output_file
    for gene in genes:
        gene.estimate_pi_with_mle()


def estimate_taus_for_each_interaction(interactions: list) -> None:
    """Fit each alternative and propagate every support/convergence failure."""
    for interaction in interactions:
        interaction.estimate_tau_with_coordinate_ascent()
        interaction.verify_taus_are_valid(
            [
                interaction.tau_00,
                interaction.tau_01,
                interaction.tau_10,
                interaction.tau_11,
            ],
        )


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
    cohort = MutationCohort.from_files(cnt_mtx, bmr_pmfs)

    if k <= 0:
        msg = "k must be a positive integer"
        raise ValueError(msg)

    single_gene_fout = f"{out}/single_gene_results.csv"
    pairwise_interaction_fout = f"{out}/pairwise_interaction_results.csv"

    genes = initialize_gene_objects(cohort.counts, cohort.bmr_pmfs)
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

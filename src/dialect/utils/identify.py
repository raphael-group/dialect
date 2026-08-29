"""TODO: Add docstring."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import pandas as pd

from dialect.data.cohort import MutationCohort
from dialect.models.assembly import (
    initialize_gene_objects,
    initialize_interaction_objects,
    save_cbase_stats_to_gene_objects,
)
from dialect.models.gene import MARGINAL_FIT_CONTRACT
from dialect.models.interaction import (
    LRT_CONTRACT,
    PAIR_EFFECT_IDENTIFIABILITY_CONTRACT,
    PAIR_EFFECT_IDENTIFIED_STATUS,
    PAIR_FIT_CONTRACT,
    PAIR_IDENTIFIABILITY_RTOL,
)

SINGLE_GENE_COUNT_CONTRACT: Final = "cohort-total-observed-and-passenger-expected-v1"

_SINGLE_GENE_RESULT_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "Gene Name",
    "Pi",
    "Log Odds Ratio",
    "Likelihood Ratio",
    "Observed Mutations",
    "Expected Mutations",
    "Obs. - Exp. Mutations",
)
SINGLE_GENE_CBASE_ANNOTATION_COLUMNS: Final[tuple[str, str]] = (
    "CBaSE Pos. Sel. Phi",
    "CBaSE Pos. Sel. P-Val",
)
_SINGLE_GENE_RESULT_SUFFIX_COLUMNS: Final[tuple[str, ...]] = (
    "MLE Algorithm",
    "MLE Converged",
    "MLE Iterations",
    "MLE Bracket Width",
    "MLE Fixed-Point Residual",
    "MLE KKT Residual",
    "MLE Log Likelihood",
    "Single-Gene LRT Status",
    "LRT Contract",
    "Single-Gene Count Contract",
)
SINGLE_GENE_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    _SINGLE_GENE_RESULT_PREFIX_COLUMNS + _SINGLE_GENE_RESULT_SUFFIX_COLUMNS
)
SINGLE_GENE_RESULT_COLUMNS_WITH_CBASE: Final[tuple[str, ...]] = (
    _SINGLE_GENE_RESULT_PREFIX_COLUMNS
    + SINGLE_GENE_CBASE_ANNOTATION_COLUMNS
    + _SINGLE_GENE_RESULT_SUFFIX_COLUMNS
)
SINGLE_GENE_RESULT_COLUMN_SEMANTICS: Final = MappingProxyType(
    {
        "Gene Name": "exact gene-effect feature identifier",
        "Pi": "constrained maximum-likelihood driver probability",
        "Log Odds Ratio": "log(Pi / (1 - Pi)) with infinite boundary values",
        "Likelihood Ratio": "single-gene passenger-null likelihood-ratio statistic",
        "Observed Mutations": "cohort total sum_i C_i",
        "Expected Mutations": "cohort total sum_i sum_k k P(B_i = k)",
        "Obs. - Exp. Mutations": (
            "Observed Mutations minus Expected Mutations on the cohort-total scale"
        ),
        "CBaSE Pos. Sel. Phi": "optional CBaSE positive-selection phi annotation",
        "CBaSE Pos. Sel. P-Val": "optional CBaSE positive-selection p-value",
        "MLE Algorithm": "machine-readable constrained marginal-fit contract",
        "MLE Converged": "whether the constrained marginal MLE converged",
        "MLE Iterations": "number of signed-score bisection iterations",
        "MLE Bracket Width": "final signed-score bracket width",
        "MLE Fixed-Point Residual": "stable mixture fixed-point residual",
        "MLE KKT Residual": "total-log-likelihood score KKT residual",
        "MLE Log Likelihood": "maximized single-gene log likelihood",
        "Single-Gene LRT Status": "finite or boundary-null LRT status",
        "LRT Contract": "machine-readable likelihood-ratio-test contract",
        "Single-Gene Count Contract": SINGLE_GENE_COUNT_CONTRACT,
    },
)


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def create_single_gene_results(
    genes: list,
    output_path: str,
    cbase_phi_vals_present: bool,
) -> None:
    """Write one row per gene under the explicit single-gene result schema.

    ``Observed Mutations``, ``Expected Mutations``, and their difference are all
    cohort totals. The optional CBaSE phi and p-value columns are emitted together
    only when CBaSE annotations were applied.
    """
    results = []
    for gene in genes:
        gene.validate_mle_fit()
        log_odds_ratio = gene.compute_log_odds_ratio(gene.pi)
        likelihood_ratio = gene.compute_likelihood_ratio(gene.pi)
        observed_mutations = sum(gene.counts)
        expected_mutations = gene.calculate_expected_total_mutations()
        obs_minus_exp_mutations = observed_mutations - expected_mutations
        row = {
            "Gene Name": gene.name,
            "Pi": gene.pi,
            "Log Odds Ratio": log_odds_ratio,
            "Likelihood Ratio": likelihood_ratio,
            "Observed Mutations": observed_mutations,
            "Expected Mutations": expected_mutations,
            "Obs. - Exp. Mutations": obs_minus_exp_mutations,
            "MLE Algorithm": gene.mle_algorithm,
            "MLE Converged": gene.mle_converged,
            "MLE Iterations": gene.mle_iterations,
            "MLE Bracket Width": gene.mle_bracket_width,
            "MLE Fixed-Point Residual": gene.mle_fixed_point_residual,
            "MLE KKT Residual": gene.mle_kkt_residual,
            "MLE Log Likelihood": gene.mle_log_likelihood,
            "Single-Gene LRT Status": gene.likelihood_ratio_status,
            "LRT Contract": LRT_CONTRACT,
            "Single-Gene Count Contract": SINGLE_GENE_COUNT_CONTRACT,
        }
        if row["MLE Algorithm"] != MARGINAL_FIT_CONTRACT:
            msg = f"Gene {gene.name} has a drifting serialized marginal-fit contract."
            raise ValueError(msg)
        if cbase_phi_vals_present:
            row.update(
                {
                    "CBaSE Pos. Sel. Phi": gene.cbase_phi,
                    "CBaSE Pos. Sel. P-Val": gene.cbase_p,
                },
            )
        results.append(row)

    columns = (
        SINGLE_GENE_RESULT_COLUMNS_WITH_CBASE
        if cbase_phi_vals_present
        else SINGLE_GENE_RESULT_COLUMNS
    )
    results_df = pd.DataFrame(results, columns=columns)
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
        likelihood_ratio = interaction.likelihood_ratio
        if likelihood_ratio is None:
            likelihood_ratio = interaction.compute_likelihood_ratio(taus)
        effect_identifiability = interaction.effect_identifiability_status()
        effect_identifiable = effect_identifiability == PAIR_EFFECT_IDENTIFIED_STATUS
        log_odds_ratio = (
            interaction.compute_log_odds_ratio(taus) if effect_identifiable else None
        )
        wald_statistic = (
            interaction.compute_wald_statistic(taus) if effect_identifiable else None
        )
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
                "Tau_1X": (
                    interaction.tau_10 + interaction.tau_11
                    if effect_identifiable
                    else None
                ),
                "Tau_X1": (
                    interaction.tau_01 + interaction.tau_11
                    if effect_identifiable
                    else None
                ),
                "Effect Identifiability": effect_identifiability,
                "Effect Identifiability Contract": (
                    PAIR_EFFECT_IDENTIFIABILITY_CONTRACT
                ),
                "Effect Identifiability Relative Tolerance": (
                    PAIR_IDENTIFIABILITY_RTOL
                ),
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

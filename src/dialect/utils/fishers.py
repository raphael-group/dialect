import logging
from statsmodels.stats.multitest import multipletests


def run_fishers_exact_analysis(interactions):
    """
    Computes Fisher's exact test q-values for mutual exclusivity and co-occurrence
    across all interaction pairs.

    :param interactions: List of Interaction objects.
    :return: Dictionary where keys are interaction names, and values are dicts
             with 'me_qval' and 'co_qval'.
    """
    logging.info("Computing Fisher's q-values...")

    # Step 1: Compute p-values and store interaction names
    interaction_names = []
    me_pvals = []
    co_pvals = []

    for interaction in interactions:
        interaction_names.append(interaction.name)
        me_pval, co_pval = interaction.compute_fisher_pvalues()
        me_pvals.append(me_pval)
        co_pvals.append(co_pval)

    # Step 2: Apply BH correction
    me_qvals = multipletests(me_pvals, method="fdr_bh")[1]
    co_qvals = multipletests(co_pvals, method="fdr_bh")[1]

    # Step 3: Create the final dictionary with q-values
    fisher_results = {
        name: {"me_qval": me_qvals[i], "co_qval": co_qvals[i]}
        for i, name in enumerate(interaction_names)
    }

    logging.info("Finished computing Fisher's q-values.")
    return fisher_results

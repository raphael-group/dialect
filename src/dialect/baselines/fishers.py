"""TODO: Add docstring."""

from statsmodels.stats.multitest import multipletests


def run_fishers_exact_analysis(interactions: list) -> dict:
    """TODO: Add docstring."""
    interaction_names = []
    me_pvals = []
    co_pvals = []

    for interaction in interactions:
        interaction_names.append(interaction.name)
        me_pval, co_pval = interaction.compute_fisher_pvalues()
        me_pvals.append(me_pval)
        co_pvals.append(co_pval)

    me_qvals = multipletests(me_pvals, method="fdr_bh")[1]
    co_qvals = multipletests(co_pvals, method="fdr_bh")[1]

    return {
        name: {
            "me_pval": me_pvals[i],
            "co_pval": co_pvals[i],
            "me_qval": me_qvals[i],
            "co_qval": co_qvals[i],
        }
        for i, name in enumerate(interaction_names)
    }

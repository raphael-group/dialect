"""TODO: Add docstring."""

from math import sqrt
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# ------------------------------------------------------------------------------------ #
#                                       CONSTANTS                                      #
# ------------------------------------------------------------------------------------ #
ME_METHOD_RANKING_CRITERIA = {
    "DIALECT (Rho)": ("Rho", "ascending"),
    "DIALECT (LRT)": ("Likelihood Ratio", "descending"),
    "DIALECT (LOR)": ("Log Odds Ratio", "descending"),
    "DIALECT (Wald)": ("Wald Statistic", "descending"),
    "DISCOVER": ("Discover ME P-Val", "ascending"),
    "Fisher's Exact Test": ("Fisher's ME P-Val", "ascending"),
    "MEGSA": ("MEGSA S-Score (LRT)", "descending"),
    "WeSME": ("WeSME P-Val", "ascending"),
}

CO_METHOD_RANKING_CRITERIA = {
    "DIALECT (Rho)": ("Rho", "descending"),
    "DIALECT (LRT)": ("Likelihood Ratio", "descending"),
    "DIALECT (LOR)": ("Log Odds Ratio", "ascending"),
    "DIALECT (Wald)": ("Wald Statistic", "ascending"),
    "DISCOVER": ("Discover CO P-Val", "ascending"),
    "Fisher's Exact Test": ("Fisher's CO P-Val", "ascending"),
    "WeSCO": ("WeSCO P-Val", "ascending"),
}

ME_METHOD_SIGNIFICANCE_THRESOLDS = {
    "DISCOVER": ("Discover ME Q-Val", 0.01),
    "Fisher's Exact Test": ("Fisher's ME Q-Val", 0.01),
    "MEGSA": ("MEGSA P-Val", 1e-3),
    "WeSME": ("WeSME Q-Val", 0.01),
}

CO_METHOD_SIGNIFICANCE_THRESOLDS = {
    "DISCOVER": ("Discover CO Q-Val", 0.01),
    "Fisher's Exact Test": ("Fisher's CO Q-Val", 0.01),
    "WeSCO": ("WeSCO Q-Val", 0.01),
}

# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def compute_epsilon_threshold(num_samples: int, alpha: float = 0.001) -> float:
    r"""Compute the epsilon threshold for the one-sided normal approximation CI.

    We solve the equation:

    .. math::

        \\max\\Bigl(\\epsilon - z_{(1 - \\alpha)}
        \\sqrt{\\frac{\\epsilon (1 - \\epsilon)}{n}}, 0\\Bigr) = 0

    for \\(\\epsilon \\in [0, 1]\\), where:
    - \\(n\\) = ``num_samples``,
    - \\(\\alpha\\) is the one-sided significance level,
    - \\(z_{(1 - \\alpha)}\\) is critical value from the standard normal distribution
      at the \\((1 - \\alpha)\\) quantile.

    **Parameters**:
      :param num_samples: (int) Number of samples (\\(n\\))
      :param alpha: (float) Significance level for the one-sided interval
                    (default 0.001 for a 99.9% one-sided CI)

    **Returns**:
      :return: (float) The marginal driver mutation rate threshold \\(\\epsilon\\) s.t.
               the lower bound of the one-sided normal CI is exactly 0.
    """
    z = norm.ppf(1 - alpha)

    def f(eps: float) -> float:
        return eps - z * sqrt(eps * (1 - eps) / num_samples)

    return brentq(f, 1e-6, 1.0)


def generate_top_ranked_co_interaction_tables(
    results_df: pd.DataFrame,
    num_pairs: int,
    num_samples: int,
    methods: list,
    dialect_thresholds_fn: Path,
) -> dict:
    """TODO: Add docstring."""
    dialect_thresholds_df = pd.read_csv(dialect_thresholds_fn, index_col=0)
    method_to_top_ranked_co_interaction_table = {}
    method_to_num_significant_co_pairs = {}
    for method in methods:
        if method not in CO_METHOD_RANKING_CRITERIA:
            continue
        rank_metric, sort_order = CO_METHOD_RANKING_CRITERIA[method]
        significance_metric, threshold = CO_METHOD_SIGNIFICANCE_THRESOLDS.get(
            method,
            (None, None),
        )
        if method == "DIALECT (LRT)":
            significance_metric = "Likelihood Ratio"
            threshold = dialect_thresholds_df.sort_values(
                by="CO_LRT_Threshold",
                ascending=False,
            ).iloc[0]["CO_LRT_Threshold"]
        top_ranked_co_interaction_table = results_df.sort_values(
            by=rank_metric,
            ascending=sort_order == "ascending",
        )
        if method in {
            "DIALECT (Rho)",
            "DIALECT (LRT)",
            "DIALECT (LOR)",
            "DIALECT (Wald)",
        }:
            epsilon = compute_epsilon_threshold(num_samples)
            top_ranked_co_interaction_table = top_ranked_co_interaction_table[
                (top_ranked_co_interaction_table["Tau_1X"] > epsilon)
                & (top_ranked_co_interaction_table["Tau_X1"] > epsilon)
                & (top_ranked_co_interaction_table["Rho"] > 0)
            ]

        if method == "DIALECT (LRT)":
            method_top_ranked_co_interaction_table = top_ranked_co_interaction_table[
                ["Gene A", "Gene B", rank_metric]
            ].copy()
            method_top_ranked_co_interaction_table["Significant"] = (
                method_top_ranked_co_interaction_table[significance_metric] > threshold
            )
            method_to_num_significant_co_pairs[method] = (
                method_top_ranked_co_interaction_table["Significant"].sum()
            )
        else:
            method_top_ranked_co_interaction_table = top_ranked_co_interaction_table[
                ["Gene A", "Gene B", rank_metric, significance_metric]
            ].copy()
            method_top_ranked_co_interaction_table["Significant"] = (
                method_top_ranked_co_interaction_table[significance_metric] < threshold
            )
            method_to_num_significant_co_pairs[method] = (
                method_top_ranked_co_interaction_table["Significant"].sum()
            )

        method_to_top_ranked_co_interaction_table[method] = (
            method_top_ranked_co_interaction_table.head(num_pairs)
        )

    return method_to_top_ranked_co_interaction_table, method_to_num_significant_co_pairs

def generate_top_ranked_me_interaction_tables(
    results_df: pd.DataFrame,
    num_pairs: int,
    num_samples: int,
    methods: list,
    dialect_thresholds_fn: Path,
) -> dict:
    """TODO: Add docstring."""
    dialect_thresholds_df = pd.read_csv(dialect_thresholds_fn, index_col=0)
    method_to_top_ranked_me_interaction_table = {}
    method_to_num_significant_me_pairs = {}
    for method in methods:
        if method not in ME_METHOD_RANKING_CRITERIA:
            continue
        rank_metric, sort_order = ME_METHOD_RANKING_CRITERIA[method]
        significance_metric, threshold = ME_METHOD_SIGNIFICANCE_THRESOLDS.get(
            method,
            (None, None),
        )
        if method == "DIALECT (Rho)":
            significance_metric = "Rho"
            threshold = dialect_thresholds_df.sort_values(
                by="ME_Rho_Threshold",
                ascending=True,
            ).iloc[0]["ME_Rho_Threshold"]
        top_ranked_me_interaction_table = results_df.sort_values(
            by=rank_metric,
            ascending=sort_order == "ascending",
        )
        if method in {
            "DIALECT (Rho)",
            "DIALECT (LRT)",
            "DIALECT (LOR)",
            "DIALECT (Wald)",
        }:
            epsilon = compute_epsilon_threshold(num_samples)
            top_ranked_me_interaction_table = top_ranked_me_interaction_table[
                (top_ranked_me_interaction_table["Tau_1X"] > epsilon)
                & (top_ranked_me_interaction_table["Tau_X1"] > epsilon)
                & (top_ranked_me_interaction_table["Rho"] < 0)
            ]

        if rank_metric == significance_metric:
            method_top_ranked_me_interaction_table = top_ranked_me_interaction_table[
                ["Gene A", "Gene B", rank_metric]
            ].copy()
        else:
            method_top_ranked_me_interaction_table = top_ranked_me_interaction_table[
                ["Gene A", "Gene B", rank_metric, significance_metric]
            ].copy()
        method_top_ranked_me_interaction_table["Significant"] = (
            method_top_ranked_me_interaction_table[significance_metric] < threshold
        )
        method_to_num_significant_me_pairs[method] = (
            method_top_ranked_me_interaction_table["Significant"].sum()
        )
        method_to_top_ranked_me_interaction_table[method] = (
            method_top_ranked_me_interaction_table.head(num_pairs)
        )

    return method_to_top_ranked_me_interaction_table, method_to_num_significant_me_pairs

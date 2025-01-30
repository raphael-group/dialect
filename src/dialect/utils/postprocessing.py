"""TODO: Add docstring."""

from math import sqrt

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# ------------------------------------------------------------------------------------ #
#                                       CONSTANTS                                      #
# ------------------------------------------------------------------------------------ #
MIN_DRIVER_COUNT = 10
PVALUE_THRESHOLD = 1.0

ME_COLUMN_MAP = {
    "DIALECT": "Rho",
    "DISCOVER": "Discover ME P-Val",
    "Fisher's Exact Test": "Fisher's ME P-Val",
    "MEGSA": "MEGSA S-Score (LRT)",
    "WeSME": "WeSME P-Val",
}

CO_COLUMN_MAP = {
    "DIALECT": "Rho",
    "DISCOVER": "Discover CO P-Val",
    "Fisher's Exact Test": "Fisher's CO P-Val",
    "MEGSA": None,
    "WeSME": "WeSCO P-Val",
}


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def get_sort_column(method: str, is_me: bool) -> str:
    """TODO: Add docstring."""
    if is_me:
        return ME_COLUMN_MAP.get(method)
    return CO_COLUMN_MAP.get(method)

def filter_by_dialect(
    top_ranking_pairs: pd.DataFrame,
    num_samples: int,
    is_me: bool,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    epsilon = compute_epsilon_threshold(num_samples)
    filtered_pairs = top_ranking_pairs[
        (top_ranking_pairs["Tau_1X"] > epsilon)
        & (top_ranking_pairs["Tau_X1"] > epsilon)
    ]
    return (
        filtered_pairs[filtered_pairs["Rho"] < 0]
        if is_me
        else filtered_pairs[filtered_pairs["Rho"] > 0]
    )


def filter_by_method(
    top_ranking_pairs: pd.DataFrame,
    method: str,
    is_me: bool,
    num_samples: int,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    if method == "MEGSA" and not is_me:
        return None

    if method == "DIALECT":
        top_ranking_pairs = filter_by_dialect(top_ranking_pairs, num_samples, is_me)

    elif method == "MEGSA":
        top_ranking_pairs = top_ranking_pairs[
            top_ranking_pairs["MEGSA S-Score (LRT)"] > 0
        ]

    elif method == "DISCOVER":
        if is_me:
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["Discover ME P-Val"] < PVALUE_THRESHOLD
            ]
        else:
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["Discover CO P-Val"] < PVALUE_THRESHOLD
            ]

    elif method == "Fisher's Exact Test":
        if is_me:
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["Fisher's ME P-Val"] < PVALUE_THRESHOLD
            ]
        else:
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["Fisher's CO P-Val"] < PVALUE_THRESHOLD
            ]

    elif method == "WeSME":
        if is_me:
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["WeSME P-Val"] < PVALUE_THRESHOLD
            ]
        else:
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["WeSCO P-Val"] < PVALUE_THRESHOLD
            ]

    return top_ranking_pairs


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


def get_top_ranked_pairs_by_method(
    results_df: pd.DataFrame,
    method: str,
    is_me: bool,
    num_pairs: int,
    num_samples: int,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    sort_col = get_sort_column(method, is_me)
    if sort_col is None:
        return None

    if method == "DIALECT":
        ascending = is_me
    elif method == "MEGSA":
        ascending = False
    else:
        ascending = True

    top_ranking_pairs = results_df.sort_values(
        by=sort_col,
        ascending=ascending,
    )
    top_ranking_pairs = filter_by_method(
        top_ranking_pairs,
        method,
        is_me,
        num_samples,
    )
    if top_ranking_pairs is None or top_ranking_pairs.empty:
        return None
    top_ranking_pairs = top_ranking_pairs.head(num_pairs)
    return top_ranking_pairs[["Gene A", "Gene B", sort_col]]


def generate_top_ranking_tables(
    results_df: pd.DataFrame,
    is_me: bool,
    num_pairs: int,
    num_samples: int,
) -> dict:
    """TODO: Add docstring."""
    methods = ["DIALECT", "DISCOVER", "Fisher's Exact Test", "MEGSA", "WeSME"]
    tables = {}
    for method in methods:
        top_df = get_top_ranked_pairs_by_method(
            results_df=results_df,
            method=method,
            is_me=is_me,
            num_pairs=num_pairs,
            num_samples=num_samples,
        )
        top_df = (
            pd.DataFrame(columns=["Gene A", "Gene B"]) if top_df is None else top_df
        )
        tables[method] = top_df

    return tables

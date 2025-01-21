import pandas as pd

# ---------------------------------------------------------------------------- #
#                                   CONSTANTS                                  #
# ---------------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def get_sort_column(method, is_me):
    """
    Returns the column name to sort on, depending on method and whether
    we're doing ME or CO. Returns None if method doesn't apply to the
    chosen meco (e.g. MEGSA for co-occurrence).
    """
    if is_me:
        return ME_COLUMN_MAP.get(method, None)
    else:
        return CO_COLUMN_MAP.get(method, None)


def filter_by_method(
    top_ranking_pairs,
    method,
    is_me,
    num_samples,
):
    """
    Applies method-specific filters to the top_ranking_pairs DataFrame,
    depending on whether we're seeking ME or CO.
    Returns the filtered DataFrame or None if not applicable.
    """
    if method == "MEGSA" and not is_me:
        return None

    if method == "DIALECT":
        epsilon = MIN_DRIVER_COUNT / num_samples
        # only keep pairs w/ both driver marginals > epsilon
        top_ranking_pairs = top_ranking_pairs[
            (top_ranking_pairs["Tau_1X"] > epsilon)
            & (top_ranking_pairs["Tau_X1"] > epsilon)
        ]
        if is_me:
            top_ranking_pairs = top_ranking_pairs[top_ranking_pairs["Rho"] < 0]
        else:
            top_ranking_pairs = top_ranking_pairs[top_ranking_pairs["Rho"] > 0]

    elif method == "MEGSA":
        # For ME, keep only S-Score > 0
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


def get_top_ranked_pairs_by_method(
    results_df,
    method,
    is_me,
    num_pairs,
    num_samples,
):
    """
    Given a results_df containing all methods' results, a method name,
    and whether we are looking at ME or CO, returns the top num_pairs
    after applying the appropriate filters/sorting.
    Returns None if not applicable (e.g. MEGSA + CO).
    """
    sort_col = get_sort_column(method, is_me)
    if sort_col is None:
        return None

    if method == "DIALECT":
        # sort rho ascending for ME and descending for CO
        # negative rho values indicate mutual exclusivity
        ascending = is_me
    elif method == "MEGSA":
        # MEGSA uses LRT scores, which you sort descending
        ascending = False
    else:
        # all other methods have p-values that you sort ascending
        ascending = True

    top_ranking_pairs = results_df.sort_values(
        by=sort_col, ascending=ascending
    )
    top_ranking_pairs = filter_by_method(
        top_ranking_pairs, method, is_me, num_samples
    )
    if top_ranking_pairs is None or top_ranking_pairs.empty:
        return None
    top_ranking_pairs = top_ranking_pairs.head(num_pairs)
    top_ranking_pairs = top_ranking_pairs[["Gene A", "Gene B", sort_col]]
    return top_ranking_pairs


def generate_top_ranking_tables(
    results_df,
    is_me,
    num_pairs,
    num_samples,
):
    """
    Generates a dictionary of top-ranked dataframes for each method w/ ME or CO
    return: dict { method_name : DataFrame or None }
    """
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
        tables[method] = top_df

    return tables

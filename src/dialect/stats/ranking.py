"""Rank and threshold per-method ME / CO interaction tables for reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from dialect.stats.constants import (
    CO_METHOD_RANKING_CRITERIA,
    CO_METHOD_SIGNIFICANCE_THRESHOLDS,
    ME_METHOD_RANKING_CRITERIA,
    ME_METHOD_SIGNIFICANCE_THRESHOLDS,
)
from dialect.stats.thresholds import compute_epsilon_threshold

if TYPE_CHECKING:
    from pathlib import Path


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
        significance_metric, threshold = CO_METHOD_SIGNIFICANCE_THRESHOLDS.get(
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
        significance_metric, threshold = ME_METHOD_SIGNIFICANCE_THRESHOLDS.get(
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

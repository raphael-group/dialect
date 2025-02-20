"""TODO: Add docstring."""

from __future__ import annotations

import math

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import colors
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score, precision_recall_curve
from upsetplot import UpSet, from_contents

# ------------------------------------------------------------------------------------ #
#                                   MODULE CONSTANTS                                   #
# ------------------------------------------------------------------------------------ #
font_path = "/u/ashuaibi/.fonts/cmuserif.ttf"
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
FONT_FAMILY = font_prop.get_name()
plt.rcParams["font.family"] = FONT_FAMILY
FONT_SCALE = 1.5

PUTATIVE_DRIVER_COLOR = "#A3C1DA"
PUTATIVE_PASSENGER_COLOR = "#D7D7D7"
LIKELY_PASSENGER_COLOR = "#FFB3B3"


# ------------------------------------------------------------------------------------ #
#                                 NETWORK VISUALIZATION                                #
# ------------------------------------------------------------------------------------ #
def draw_single_me_or_co_interaction_network(
    edges: np.ndarray,
    significant_nodes: set,
    putative_drivers: set,
    likely_passengers: set,
    method: str,
    fout: str,
    figsize: tuple = (4, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""

    def _get_bounding_box(color: tuple, is_significant: bool) -> dict:
        return {
            "facecolor": color,
            "edgecolor": "black",
            "boxstyle": "round,pad=0.25",
            "linewidth": 2 * font_scale if is_significant else font_scale / 2,
        }

    def _draw_label(
        x: float,
        y: float,
        label: str,
        color: tuple,
        is_significant: bool,
    ) -> None:
        ax.text(
            x,
            y,
            label,
            bbox=_get_bounding_box(color, is_significant),
            ha="center",
            va="center",
            fontsize=font_scale * 8,
        )


    def _get_node_colors(graph: nx.Graph) -> list:
        return [
            LIKELY_PASSENGER_COLOR
            if node in likely_passengers
            else PUTATIVE_DRIVER_COLOR
            if node in putative_drivers
            else PUTATIVE_PASSENGER_COLOR
            for node in graph.nodes
        ]

    plt.rcParams["font.serif"] = FONT_FAMILY

    graph = nx.Graph()
    graph.add_edges_from(edges)
    node_colors = _get_node_colors(graph)
    pos = nx.spring_layout(graph, k=2.5, iterations=100, seed=42)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    nx.draw(
        graph,
        pos,
        ax=ax,
        node_color="none",
        edge_color="steelblue",
        with_labels=False,
    )
    for i, (node, (x, y)) in enumerate(pos.items()):
        _draw_label(x, y, node, node_colors[i], node in significant_nodes)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.title(method, fontsize=font_scale * 10)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close(fig)


def draw_single_me_and_co_interaction_network(
    me_edges: np.ndarray,
    co_edges: np.ndarray,
    putative_drivers: set,
    likely_passengers: set,
    method: str,
    fout: str,
    figsize: tuple = (4, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""

    def _get_bounding_box(color: tuple) -> dict:
        return {
            "facecolor": color,
            "edgecolor": "black",
            "boxstyle": "round,pad=0.25",
        }

    def _draw_label(x: float, y: float, label: str, color: tuple) -> None:
        ax.text(
            x,
            y,
            label,
            bbox=_get_bounding_box(color),
            ha="center",
            va="center",
            fontsize=font_scale * 8,
        )

    def _get_node_colors(graph: nx.Graph) -> list:
        return [
            LIKELY_PASSENGER_COLOR
            if node in likely_passengers
            else PUTATIVE_DRIVER_COLOR
            if node in putative_drivers
            else PUTATIVE_PASSENGER_COLOR
            for node in graph.nodes
        ]

    graph = nx.Graph()
    for a, b in me_edges:
        graph.add_edge(a, b, interaction_type="ME")
    for a, b in co_edges:
        graph.add_edge(a, b, interaction_type="CO")

    node_colors = _get_node_colors(graph)
    pos = nx.spring_layout(graph, k=2.5, iterations=100, seed=42)

    plt.rcParams["font.serif"] = FONT_FAMILY

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    me_edge_list = [
        (u, v) for u, v, d in graph.edges(data=True) if d["interaction_type"] == "ME"
    ]
    co_edge_list = [
        (u, v) for u, v, d in graph.edges(data=True) if d["interaction_type"] == "CO"
    ]

    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=me_edge_list, edge_color="green")
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=co_edge_list, edge_color="blue")

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="none",
        edgecolors="none",
    )

    for i, (node, (x, y)) in enumerate(pos.items()):
        _draw_label(x, y, node, node_colors[i])

    plt.title(method, fontsize=font_scale * 10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------------------------ #
#                                PRECISION-RECALL CURVES                               #
# ------------------------------------------------------------------------------------ #
def draw_simulation_precision_recall_curve(
    methods: dict,
    y_true: np.ndarray,
    fout: str,
    figsize: tuple = (5, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY

    plt.figure(figsize=figsize)
    for method_name, scores in methods.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(
            recall,
            precision,
            label=f"{method_name} (AUC={ap:.3f})",
            linewidth=font_scale * 2,
            alpha=0.75,
        )
    random_auc = sum(y_true) / len(y_true)
    plt.axhline(
        y=random_auc,
        color="gray",
        label=f"Baseline (AUC={random_auc:.3f})",
        linewidth=font_scale * 2,
        alpha=0.75,
    )
    plt.xlabel("Recall", fontsize=font_scale * 10)
    plt.ylabel("Precision", fontsize=font_scale * 10)

    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)

    plt.gca().patch.set_alpha(0)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=font_scale * 6,
        frameon=True,
        facecolor="none",
        edgecolor="black",
    )

    plt.tight_layout()
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close()


def draw_average_simulation_precision_recall_curve(
    all_method_scores: list,
    all_y_true: list,
    fout: str,
    methods: list,
    figsize: tuple = (5, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.figure(figsize=figsize)

    fixed_recall_points = np.linspace(0, 1, 10001)
    for idx, method_name in enumerate(methods):
        single_curve_precisions = []
        auprc_vals = []
        for y_true, methods_dict in zip(all_y_true, all_method_scores):
            scores = methods_dict[method_name]
            scores = np.nan_to_num(scores, nan=0)
            single_precision, single_recall, _ = precision_recall_curve(y_true, scores)

            interp_func = interp1d(
                single_recall,
                single_precision,
                kind="linear",
                bounds_error=False,
                fill_value=(1, 0),
            )
            interp_precision = interp_func(fixed_recall_points)
            single_curve_precisions.append(interp_precision)
            auprc_vals.append(average_precision_score(y_true, scores))

        precision_arr = np.array(single_curve_precisions)
        precision_mean = np.mean(precision_arr, axis=0)
        precision_std_dev = np.std(precision_arr, axis=0)
        plt.plot(
            fixed_recall_points,
            precision_mean,
            label=f"{method_name} (AUC={np.mean(auprc_vals):.3f})",
            linewidth=font_scale * 2,
            alpha=0.75,
            color=f"C{idx}",
        )
        plt.fill_between(
            fixed_recall_points,
            np.clip(precision_mean - precision_std_dev, 0, 1),
            np.clip(precision_mean + precision_std_dev, 0, 1),
            color=f"C{idx}",
            alpha=0.2,
        )

    total_positives = sum(np.sum(y_true) for y_true in all_y_true)
    total_samples = sum(len(y_true) for y_true in all_y_true)
    baseline = total_positives / total_samples if total_samples > 0 else 0

    plt.axhline(
        y=baseline,
        color="gray",
        label=f"Baseline (AUC={baseline:.3f})",
        linewidth=font_scale * 2,
        alpha=0.75,
    )

    plt.ylim(-0.01, 1.01)
    plt.xlabel("Recall", fontsize=font_scale * 10)
    plt.ylabel("Precision", fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)
    leg = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=font_scale * 6,
        frameon=True,
        facecolor="none",
        edgecolor="black",
    )
    leg.get_frame().set_edgecolor("k")
    plt.gca().patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close()


def draw_concat_simulation_precision_recall_curve(
    all_method_scores: list,
    all_y_true: list,
    fout: str,
    methods: list,
    figsize: tuple = (5, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.figure(figsize=figsize)

    for idx, method_name in enumerate(methods):
        y_true_concat = []
        scores_concat = []
        for y_true, methods_dict in zip(all_y_true, all_method_scores):
            scores = methods_dict[method_name]
            scores = np.nan_to_num(scores, nan=0)
            y_true_concat.append(y_true)
            scores_concat.append(scores)
        y_true_concat = np.concatenate(y_true_concat)
        scores_concat = np.concatenate(scores_concat)

        precision, recall, _ = precision_recall_curve(y_true_concat, scores_concat)
        auc_val = average_precision_score(y_true_concat, scores_concat)
        plt.plot(
            recall,
            precision,
            label=f"{method_name} (AUC={auc_val:.3f})",
            linewidth=font_scale * 2,
            alpha=0.75,
            color=f"C{idx}",
        )

    total_positives = sum(np.sum(y_true) for y_true in all_y_true)
    total_samples = sum(len(y_true) for y_true in all_y_true)
    baseline = total_positives / total_samples if total_samples > 0 else 0

    plt.axhline(
        y=baseline,
        color="gray",
        label=f"Baseline (AUC={baseline:.3f})",
        linewidth=font_scale * 2,
        alpha=0.75,
    )

    plt.ylim(-0.01, 1.01)
    plt.xlabel("Recall", fontsize=font_scale * 10)
    plt.ylabel("Precision", fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)
    leg = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=font_scale * 6,
        frameon=True,
        facecolor="none",
        edgecolor="black",
    )
    leg.get_frame().set_edgecolor("k")
    plt.gca().patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close()


def draw_auprc_hit_curve(
    true_pair_lists: list,
    top_ranked_tables: list,
    methods: list,
    total_pairs: int,
    fout: str,
    max_k: int = 1000,
    figsize: tuple = (5, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.figure(figsize=figsize)

    def get_recalls_at_k(
        true_pairs: set,
        top_ranked_table: pd.DataFrame,
        num_positive: int,
        max_k: int,
    ) -> list:
        recalls_at_k = []
        for k in range(max_k):
            top_k_pairs_set = {
                tuple(sorted(pair))
                for pair in top_ranked_table.head(k)[["Gene A", "Gene B"]].to_numpy()
            }
            recalls_at_k.append(len(top_k_pairs_set & true_pairs) / num_positive)
        return recalls_at_k

    method_to_recalls_at_k_lists = {}
    for _, (true_pairs, top_ranked_table) in enumerate(
        zip(true_pair_lists, top_ranked_tables),
    ):
        true_pairs_set = {tuple(sorted(pair)) for pair in true_pairs}
        num_positive = len(true_pairs)
        for method in methods:
            if method not in method_to_recalls_at_k_lists:
                method_to_recalls_at_k_lists[method] = []
            method_top_ranked_table = top_ranked_table[method]
            method_recalls_at_k = get_recalls_at_k(
                true_pairs_set,
                method_top_ranked_table,
                num_positive,
                max_k,
            )
            method_to_recalls_at_k_lists[method].append(method_recalls_at_k)
    for method, recall_at_k_lists in method_to_recalls_at_k_lists.items():
        mean_recall_at_k = np.mean(recall_at_k_lists, axis=0)
        plt.plot(
            np.arange(1, len(mean_recall_at_k) + 1),
            mean_recall_at_k,
            label=method,
            linewidth=font_scale * 2,
            alpha=0.75,
        )

    k_values = np.arange(max_k)
    baseline_recall = k_values / total_pairs
    plt.plot(
        k_values,
        baseline_recall,
        color="gray",
        label="Baseline",
        linewidth=font_scale * 2,
        alpha=0.75,
    )

    plt.ylim(-0.01, 1.01)
    plt.ylabel("Recall@K", fontsize=font_scale * 10)
    plt.xlabel("Top K Ranked Interactions", fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)
    leg = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=font_scale * 6,
        frameon=True,
        facecolor="none",
        edgecolor="black",
    )
    leg.get_frame().set_edgecolor("k")
    plt.gca().patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close()


def draw_auc_vs_factor_curve(
    x: list,
    method_to_avg_auprc_vals: dict,
    fout: str,
    xlabel: str,
    figsize: tuple = (5, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.figure(figsize=figsize)

    for method, avg_auprc_vals in method_to_avg_auprc_vals.items():
        plt.plot(
            x,
            avg_auprc_vals,
            label=method,
            linewidth=font_scale * 2,
            alpha=0.75,
            marker="o",
            markersize=font_scale * 3,
        )

    plt.ylabel("AUPRC", fontsize=font_scale * 10)
    plt.xlabel(xlabel, fontsize=font_scale * 10)
    plt.ylim(0, 1)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)
    leg = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=font_scale * 6,
        frameon=True,
        facecolor="none",
        edgecolor="black",
    )
    leg.get_frame().set_edgecolor("k")
    plt.gca().patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(f"{fout}.png", dpi=300, transparent=True)
    plt.savefig(f"{fout}.svg", dpi=300, transparent=True)
    plt.close()


# ------------------------------------------------------------------------------------ #
#                                    CBASE ANALYSIS                                    #
# ------------------------------------------------------------------------------------ #
def draw_cbase_likely_passenger_proportion_barplot(
    subtype_to_likely_passenger_proportion: dict,
    out_fn: str,
    num_genes: int,
    figsize: tuple = (4, 6),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY

    plt.figure(figsize=figsize)
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=False, right=False)

    sorted_subtype_to_proportion = sorted(
        subtype_to_likely_passenger_proportion.items(),
        key=lambda item: item[1],
    )
    subtypes, proportions = zip(*sorted_subtype_to_proportion)
    plt.barh(
        subtypes,
        proportions,
        color="black",
        edgecolor="black",
        alpha=0.75,
    )
    plt.xlim(0, 1)
    plt.xlabel(
        f"Proportion of Likely Passengers\nin Top {num_genes} Genes",
        fontsize=font_scale * 10,
    )
    plt.ylabel("Subtype", fontsize=font_scale * 10)

    plt.tight_layout()
    plt.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    plt.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close()


def draw_cbase_top_likely_passenger_upset(
    subtype_to_likely_passenger_gene_overlap: dict,
    out_fn: str,
    min_mutation_count: int = 4,
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    gene_to_subtypes = {}
    for subtype, genes in subtype_to_likely_passenger_gene_overlap.items():
        for gene in genes:
            if gene not in gene_to_subtypes:
                gene_to_subtypes[gene] = set()
            gene_to_subtypes[gene].add(subtype)
    high_freq_gene_to_subtypes = {
        gene: subtypes
        for gene, subtypes in gene_to_subtypes.items()
        if len(subtypes) >= min_mutation_count
    }

    data_df = from_contents(high_freq_gene_to_subtypes)
    upset = UpSet(data_df, totals_plot_elements=0, element_size=font_scale * 20)

    subplots = upset.plot()
    subplots["intersections"].set_ylabel("Number of Subtypes", fontsize=font_scale * 10)
    subplots["matrix"].set_ylabel("Likely Passengers", fontsize=font_scale * 10)

    plt.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    plt.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close()


# ------------------------------------------------------------------------------------ #
#                            GENE OBSERVED EXPECTED ANALYSIS                           #
# ------------------------------------------------------------------------------------ #
def draw_gene_expected_and_observed_mutations_barplot(
    results_df: pd.DataFrame,
    likely_passenger_genes: set,
    putative_driver_genes: set,
    out_fn: str,
    num_genes: int = 5,
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    top_likely_passenger_df = results_df[
        results_df["Gene Name"].isin(likely_passenger_genes)
    ].nlargest(num_genes, "Observed Mutations")
    top_putative_driver_df = results_df[
        results_df["Gene Name"].isin(putative_driver_genes)
    ].nlargest(num_genes, "Observed Mutations")
    all_genes_df = pd.concat(
        [top_likely_passenger_df, top_putative_driver_df],
    ).sort_values("Observed Mutations", ascending=False)

    gene_labels = [
        f"{gene} *" if gene in putative_driver_genes else gene
        for gene in all_genes_df["Gene Name"]
    ]

    figwidth = num_genes
    fig, ax = plt.subplots(figsize=(figwidth, font_scale * 3))
    x = np.arange(len(all_genes_df))
    bar_width = 0.4

    ax.bar(
        x - bar_width / 2,
        all_genes_df["Observed Mutations"],
        width=bar_width,
        color="black",
        label="Observed",
    )
    ax.bar(
        x + bar_width / 2,
        all_genes_df["Expected Mutations"],
        width=bar_width,
        color="gray",
        label="Expected",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        gene_labels,
        rotation=30,
        ha="right",
        fontsize=font_scale * 8,
        color="black",
    )
    ax.set_xlabel("Gene", fontsize=font_scale * 10)
    ax.set_ylabel("Mutation Count", fontsize=font_scale * 10)
    ax.legend(fontsize=font_scale * 8)

    plt.tight_layout()
    plt.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    plt.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close()


# ------------------------------------------------------------------------------------ #
#                        TOP RANKED LIKELY PASSENGER PROPORTION                        #
# ------------------------------------------------------------------------------------ #
def draw_likely_passenger_gene_proportion_violinplot(
    method_to_subtype_to_passenger_proportion: dict,
    out_fn: str,
    figsize: tuple = (4.8, 3),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY

    methods = list(method_to_subtype_to_passenger_proportion.keys())
    values = [
        list(method_to_subtype_to_passenger_proportion[method].values())
        for method in methods
    ]

    x_positions = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=figsize)
    vp = ax.violinplot(
        values,
        positions=x_positions,
        showextrema=True,
        showmedians=True,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods)
    ax.set_xlabel("Method", fontsize=font_scale * 10)
    ax.set_ylabel(
        "Likely Passenger\nProportion AUC",
        fontsize=font_scale * 10,
    )

    for body in vp["bodies"]:
        body.set_facecolor("lightslategray")
        body.set_edgecolor("darkslategray")
        body.set_alpha(0.8)

    plt.setp(vp["cmedians"], color="maroon", linewidth=font_scale)
    plt.setp(vp["cmins"], color="slategray", linewidth=font_scale)
    plt.setp(vp["cmaxes"], color="slategray", linewidth=font_scale)
    plt.setp(vp["cbars"], color="slategray", linewidth=font_scale)

    ax.minorticks_on()
    ax.tick_params(axis="both", direction="in", length=font_scale * 4, width=font_scale)
    ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    ax.tick_params(axis="y", which="minor", left=True, right=True)
    ax.tick_params(axis="x", which="major", top=False, bottom=True)
    ax.tick_params(axis="y", which="major", left=True, right=True)
    ax.patch.set_alpha(0)

    plt.tight_layout()
    plt.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    plt.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close()


def draw_likely_passenger_proportion_hit_curve(
    method_to_likely_passenger_proportions: dict,
    out_fn: str,
    figsize: tuple = (5, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.figure(figsize=figsize)
    for method, proportions in method_to_likely_passenger_proportions.items():
        plt.plot(
            np.arange(1, len(proportions) + 1),
            proportions,
            label=method,
            linewidth=font_scale * 2,
            alpha=0.75,
        )
    plt.ylim(-0.01, 1.01)
    plt.ylabel("Likely Passenger\nProportion @ K", fontsize=font_scale * 10)
    plt.xlabel("Top K Ranked Pairs", fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)
    leg = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=font_scale * 6,
        frameon=True,
        facecolor="none",
        edgecolor="black",
    )
    leg.get_frame().set_edgecolor("k")
    plt.gca().patch.set_alpha(0)
    plt.tight_layout()
    plt.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    plt.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close()


# ------------------------------------------------------------------------------------ #
#                            MUTATION FREQUENCY DISTRIBUTION                           #
# ------------------------------------------------------------------------------------ #
def draw_sample_mutation_count_subtype_histograms(
    subtype_to_sample_mutation_counts: dict,
    subtype_avg_sample_mutation_count: float,
    out_fn: str,
    figsize: tuple = (5, 6),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY

    n_subtypes = len(subtype_to_sample_mutation_counts)
    ncols = 2

    nrows = math.ceil(n_subtypes / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if n_subtypes > 1 else [axes]

    default_color = "lightslategray"
    for i, (subtype, sample_mutation_counts) in enumerate(
        subtype_to_sample_mutation_counts.items(),
    ):
        ax = axes[i]
        ax.hist(
            sample_mutation_counts,
            color=default_color,
            alpha=0.8,
            edgecolor="black",
            bins=10,
            log=True,
        )
        ax.axvline(
            subtype_avg_sample_mutation_count,
            color="red",
            linewidth=font_scale,
            linestyle="--",
        )
        ax.set_ylim(1, 1e3)
        ax.set_title(subtype, fontsize=font_scale * 10)

    plt.tight_layout()
    fig.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    fig.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close(fig)


def draw_gene_mutation_variability_hexbin_plots(
    subtype_to_gene_mean_mutation_counts: dict,
    subtype_to_gene_std_dev_mutation_counts: dict,
    out_fn: str,
    subtypes: list,
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    ncols = len(subtype_to_gene_mean_mutation_counts)
    nrows = 1
    figsize = (font_scale * ncols, font_scale * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    for i, subtype in enumerate(subtypes):
        ax = axes[i]
        means = subtype_to_gene_mean_mutation_counts[subtype]
        stds = subtype_to_gene_std_dev_mutation_counts[subtype]

        ax.hexbin(
            means,
            stds,
            norm=colors.LogNorm(),
            gridsize=int(font_scale * 10),
            cmap="cividis",
        )
        ax.set_title(subtype, fontsize=font_scale * 10)
    plt.tight_layout()
    fig.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    fig.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close(fig)


def draw_single_subtype_background_mutation_distribution(
    subtype: str,
    exp_bkgd_muts: list,
    out_fn: str,
    font_scale: float = FONT_SCALE,
    figsize: type = (4, 4),
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    fig = plt.figure(figsize=figsize)
    plt.hist(
        exp_bkgd_muts,
        color="lightslategray",
        alpha=0.8,
        edgecolor="black",
        log=True,
    )
    plt.title(subtype, fontsize=font_scale * 10)
    plt.xlabel("Expected Passenger Mutations\nPer Sample", fontsize=font_scale * 10)
    plt.ylabel("Number of Genes", fontsize=font_scale * 10)

    plt.tight_layout()
    fig.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    fig.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close(fig)


def draw_all_subtypes_background_mutation_distribution(
    exp_bkgd_muts: list,
    out_fn: str,
    font_scale: float = FONT_SCALE,
    figsize: type = (6, 4),
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    fig = plt.figure(figsize=figsize)
    plt.hist(
        exp_bkgd_muts,
        color="lightslategray",
        alpha=0.8,
        edgecolor="black",
        log=True,
    )
    plt.xlabel(
        "Expected Per Sample Passenger\nMutations of a Gene",
        fontsize=font_scale * 10,
    )
    plt.ylabel("Number of Genes", fontsize=font_scale * 10)

    plt.tight_layout()
    fig.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    fig.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close(fig)

def draw_mutation_variability_scatterplot(
    subtype_to_coef_of_variation: dict,
    out_fn: str,
    font_scale: float = FONT_SCALE,
    figsize: tuple = (12, 3),
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    fig = plt.figure(figsize=figsize)
    sorted_subtypes = sorted(
        subtype_to_coef_of_variation.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    subtypes, coef_of_variations = zip(*sorted_subtypes)
    plt.scatter(
        x=subtypes,
        y=coef_of_variations,
        color="black",
        s=font_scale * 40,
    )
    plt.ylim(0, 23)
    plt.xlabel("Subtype", fontsize=font_scale * 10)
    plt.ylabel("Mutation Count\nCoefficient of Variation", fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8, rotation=45)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.gca().tick_params(axis="x", which="minor", top=False, bottom=False)
    plt.gca().tick_params(axis="y", which="minor", left=True, right=True)
    plt.gca().tick_params(axis="x", which="major", top=True, bottom=True)
    plt.gca().tick_params(axis="y", which="major", left=True, right=True)
    plt.tight_layout()
    fig.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    fig.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close(fig)

"""TODO: Add docstring."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from upsetplot import UpSet, from_contents

# ------------------------------------------------------------------------------------ #
#                                   MODULE CONSTANTS                                   #
# ------------------------------------------------------------------------------------ #
FONT_FAMILY = "CMU Serif"
FONT_STYLE = "serif"
FONT_SCALE = 1.5

PUTATIVE_DRIVER_COLOR = "#A3C1DA"
PUTATIVE_PASSENGER_COLOR = "#D7D7D7"
LIKELY_PASSENGER_COLOR = "#FFB3B3"


# ------------------------------------------------------------------------------------ #
#                                 NETWORK VISUALIZATION                                #
# ------------------------------------------------------------------------------------ #
def draw_single_interaction_network(
    edges: np.ndarray,
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
            fontfamily=FONT_STYLE,
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
    plt.rcParams["font.family"] = FONT_STYLE

    graph = nx.Graph()
    graph.add_edges_from(edges)
    node_colors = _get_node_colors(graph)
    pos = nx.spring_layout(graph, k=2.5, iterations=100, seed=42)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    nx.draw(graph, pos, ax=ax, node_color="none", with_labels=False)
    for i, (node, (x, y)) in enumerate(pos.items()):
        _draw_label(x, y, node, node_colors[i])
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.title(method, fontsize=font_scale * 10)
    ax.set_aspect("equal")
    ax.axis("off")
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
    plt.rcParams["font.family"] = FONT_STYLE

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
    plt.rcParams["font.family"] = FONT_STYLE

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
    figsize: tuple = (6, 4),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.rcParams["font.family"] = FONT_STYLE

    methods = list(method_to_subtype_to_passenger_proportion.keys())
    values = [
        list(method_to_subtype_to_passenger_proportion[method].values())
        for method in methods
    ]
    methods = [
        method.replace("Fisher's Exact Test", "Fisher's\nExact Test")
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
    ax.set_ylabel("Likely Passenger\nProportion", fontsize=font_scale * 10)

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


# ------------------------------------------------------------------------------------ #
#                            MUTATION FREQUENCY DISTRIBUTION                           #
# ------------------------------------------------------------------------------------ #
def draw_sample_mutation_count_subtype_histograms(
    subtype_to_sample_mutation_counts: dict,
    subtype_avg_sample_mutation_count: float,
    xlim_mapping: dict,
    out_fn: str,
    figsize: tuple = (5, 6),
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    plt.rcParams["font.serif"] = FONT_FAMILY
    plt.rcParams["font.family"] = FONT_STYLE

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
            log=True,
        )
        ax.axvline(
            subtype_avg_sample_mutation_count,
            color="red",
            linewidth=font_scale,
            linestyle="--",
        )
        ax.set_xlim(0, xlim_mapping[subtype])
        ax.set_ylim(1, 1e3)
        ax.set_title(subtype, fontsize=font_scale * 10)

    plt.tight_layout()
    fig.savefig(f"{out_fn}.png", dpi=300, transparent=True)
    fig.savefig(f"{out_fn}.svg", dpi=300, transparent=True)
    plt.close(fig)


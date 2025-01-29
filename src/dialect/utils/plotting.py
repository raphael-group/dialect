"""TODO: Add docstring."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, FancyBboxPatch, Patch
from plotnine import (
    aes,
    element_text,
    geom_boxplot,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    position_jitter,
    scale_color_manual,
    scale_shape_manual,
    theme,
    theme_tufte,
    ylim,
)
from sklearn.metrics import average_precision_score, precision_recall_curve
from upsetplot import UpSet, from_contents

from dialect.utils.postprocessing import generate_top_ranking_tables

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# ------------------------------------------------------------------------------------ #
#                              SET DEFAULT PLOTTING STYLE                              #
# ------------------------------------------------------------------------------------ #
rcParams["text.usetex"] = True
rcParams["font.family"] = "sans-serif"
rcParams["font.serif"] = ["Computer Modern"]
rcParams["font.sans-serif"] = ["Computer Modern"]


DEFAULT_GENE_COLOR = "#D3D3D3"
DECOY_GENE_COLOR = "#FFB3B3"
DRIVER_GENE_COLOR = "#A3C1DA"
EDGE_COLOR = "black"
EPSILON_MUTATION_COUNT = 10
PVALUE_THRESHOLD = 1


def set_dynamic_font_sizes(
    figsize: tuple,
    base_label_size: int = 24,
    base_tick_size: int = 20,
) -> None:
    """TODO: Add docstring."""
    scale_factor = (figsize[0] * figsize[1]) / (10 * 8)
    label_size = base_label_size * scale_factor
    tick_size = base_tick_size * scale_factor
    rcParams["axes.labelsize"] = label_size
    rcParams["xtick.labelsize"] = tick_size
    rcParams["ytick.labelsize"] = tick_size


COLOR_MAPPING = {
    "UCEC": "lightcoral",
    "SKCM": "lightcoral",
    "CRAD": "moccasin",
    "STAD": "moccasin",
    "LUAD": "khaki",
    "LUSC": "khaki",
}

XLIM_MAPPING = {
    "UCEC": 32000,
    "SKCM": 32000,
    "CRAD": 10000,
    "STAD": 10000,
    "LUAD": 2000,
    "LUSC": 2000,
}


def apply_tufte_style(ax: Axes) -> None:
    """TODO: Add docstring."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))
    ax.tick_params(axis="both", which="major", labelsize=20)


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def draw_network_plot(
    ax: Axes,
    graph: nx.Graph,
    pos: dict,
    labels: dict | None = None,
    node_colors: list | None = None,
) -> None:
    """TODO: Add docstring."""
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=EDGE_COLOR, width=2)

    if node_colors is None:
        node_colors = [DEFAULT_GENE_COLOR] * len(graph.nodes)

    for node, (x, y), color in zip(graph.nodes, pos.values(), node_colors):
        label = labels[node] if labels else str(node)
        bbox = FancyBboxPatch(
            (x - 0.08, y),
            0.16,
            0,
            boxstyle=BoxStyle.Round(pad=0.08),
            edgecolor=EDGE_COLOR,
            facecolor=color,
            linewidth=1,
        )
        ax.add_patch(bbox)
        ax.text(
            x,
            y,
            label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
            bbox={"facecolor": "none", "edgecolor": "none"},
        )
    ax.set_aspect("equal")
    ax.axis("off")


def plot_subtype_histogram(
    ax: Axes,
    subtype: str,
    data_series: pd.Series,
    avg_across_others: float,
) -> None:
    """TODO: Add docstring."""
    color = COLOR_MAPPING[subtype]
    xlim = XLIM_MAPPING[subtype]

    ax.hist(
        data_series,
        bins=20,
        range=(0, xlim),
        color=color,
        alpha=0.8,
        edgecolor="black",
        log=True,
    )
    ax.set_ylim(1, 1000)

    if 0 < avg_across_others < xlim:
        ax.axvline(
            avg_across_others,
            color="red",
            linewidth=2,
            linestyle="--",
            label="Avg. Sample Mutation Count Across Other Subtypes",
        )

    apply_tufte_style(ax)

    ax.set_xlim(0, xlim)
    ax.set_title(subtype, fontsize=28, pad=5)


# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def plot_decoy_gene_fractions(
    data_filepath: str,
    num_pairs: int,
    is_me: bool,
    out_dir: str,
) -> None:
    """TODO: Add docstring."""
    data_df = pd.read_csv(data_filepath)
    if not is_me:
        data_df = data_df[data_df["Method"] != "MEGSA"]

    subtypes = data_df["Subtype"].unique()
    colors = [
        "green",
        "blue",
        "red",
        "purple",
        "yellow",
        "orange",
        "black",
        "brown",
    ]
    shapes = ["o", "s", "^", "D"]

    color_shape_combinations = list(product(colors, shapes))
    color_shape_mapping = {
        subtypes[i]: color_shape_combinations[i] for i in range(len(subtypes))
    }
    color_mapping = {
        subtype: combo[0] for subtype, combo in color_shape_mapping.items()
    }
    shape_mapping = {
        subtype: combo[1] for subtype, combo in color_shape_mapping.items()
    }
    ixn_type = "ME" if is_me else "CO"

    plot = (
        ggplot(
            data_df,
            aes(x="Method", y="Fraction", color="Subtype", shape="Subtype"),
        )
        + geom_boxplot(
            aes(group="Method"),
            alpha=0.5,
            outlier_alpha=0,
            show_legend=False,
        )
        + geom_point(
            position=position_jitter(width=0.25),
            size=5,
            alpha=0.75,
            show_legend=True,
        )
        + scale_color_manual(values=color_mapping)
        + scale_shape_manual(values=shape_mapping)
        + labs(
            title=f"Proportion of Top-Ranked {ixn_type} Pairs w/ Likely Passengers",
            x="Method",
            y=f"Proportion of Top {num_pairs} {ixn_type} Pairs with Decoy Genes",
            color="Subtype",
            shape="Subtype",
        )
        + theme_tufte(base_family="Computer Modern")
        + theme(
            figure_size=(12, 8),
            plot_title=element_text(size=20, weight="bold"),
            axis_title=element_text(size=18),
            axis_text=element_text(size=16),
            legend_title=element_text(size=14, hjust=0.5),
            legend_text=element_text(size=12),
            legend_position="bottom",
            legend_box="horizontal",
        )
        + guides(
            color=guide_legend(title="Subtypes", ncol=11),
            shape=guide_legend(title="Subtypes", ncol=11),
        )
        + ylim(0, 1)
    )

    dout = Path(out_dir)
    dout.mkdir(parents=True, exist_ok=True)
    plot.save(
        f"{out_dir}/{ixn_type}_decoy_gene_fractions_boxplot.svg",
        dpi=300,
    )


def draw_network_gridplot_across_methods(
    num_edges: int,
    subtype: str,
    driver_genes: set,
    decoy_genes: set,
    results_df: pd.DataFrame,
    num_samples: int,
    is_me: bool,
    dout: str,
) -> None:
    """TODO: Add docstring."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    suptitle = (
        f"Top 10 Ranked ME Pairs in {subtype}"
        if is_me
        else f"Top 10 Ranked CO Pairs in {subtype}"
    )
    fig.suptitle(suptitle, fontsize=42, y=0.999)
    top_tables = generate_top_ranking_tables(
        results_df=results_df,
        is_me=is_me,
        num_pairs=num_edges,
        num_samples=num_samples,
    )
    for idx, (method, top_ranking_pairs) in enumerate(top_tables.items()):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(method, fontsize=36)
        edges = (
            []
            if top_ranking_pairs is None
            else top_ranking_pairs[["Gene A", "Gene B"]].to_numpy()
        )
        graph = nx.Graph()
        graph.add_edges_from(edges)

        if graph.number_of_edges() == 0:
            ax.text(
                0.5,
                0.5,
                "No Interactions Identified",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=32,
                color="dimgray",
            )
            ax.axis("off")
            continue

        node_colors = []
        for node in graph.nodes:
            if node in decoy_genes:
                node_colors.append(DECOY_GENE_COLOR)
            elif node in driver_genes:
                node_colors.append(DRIVER_GENE_COLOR)
            else:
                node_colors.append(DEFAULT_GENE_COLOR)

        min_edges = 2
        if graph.number_of_nodes() == min_edges:
            pos = nx.spring_layout(graph, seed=42)
        else:
            pos = nx.circular_layout(graph)

        labels = {node: node for node in graph.nodes}
        draw_network_plot(ax, graph, pos, labels, node_colors)

    plt.subplots_adjust(hspace=6, wspace=2)
    legend_ax = axes[1, 2]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=DRIVER_GENE_COLOR,
                markersize=50,
                linestyle="None",
                label="Driver Gene",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=DEFAULT_GENE_COLOR,
                markersize=50,
                linestyle="None",
                label="Passenger Gene",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=DECOY_GENE_COLOR,
                markersize=50,
                linestyle="None",
                label="Decoy Passenger Gene",
            ),
            plt.Line2D(
                [0],
                [0],
                linestyle="-",
                color="black",
                linewidth=4,
                label="Mutual Exclusivity",
            ),
        ],
        labelspacing=2,
        borderpad=1.5,
        loc="center",
        fontsize=24,
    )

    plt.tight_layout(pad=0.5)
    ranking_type = "ME" if is_me else "CO"
    fout = f"{dout}/{subtype}_{ranking_type}_network_plots_across_methods.png"
    plt.savefig(fout, dpi=300)
    plt.close(fig)


def plot_sample_mutation_count_subtype_histograms(
    subtype_sample_mut_counts: dict,
    avg_mut_cnt: float,
    fout: str,
) -> None:
    """TODO: Add docstring."""
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))
    axes = axes.flatten()

    for i, (subtype, data_series) in enumerate(
        subtype_sample_mut_counts.items(),
    ):
        ax = axes[i]
        plot_subtype_histogram(ax, subtype, data_series, avg_mut_cnt)

    plt.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.1)

    fig.text(
        0.5,
        0.06,
        "Total Sample Mutation Count",
        ha="center",
        va="center",
        fontsize=28,
    )
    fig.text(
        0.06,
        0.5,
        "Number of Samples",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=28,
    )

    legend_handles = [
        Patch(
            facecolor="lightcoral",
            edgecolor="black",
            label="Mut Cnt: 0-30k",
        ),
        Patch(facecolor="moccasin", edgecolor="black", label="Mut Cnt: 0-0k"),
        Patch(facecolor="khaki", edgecolor="black", label="Mut Cnt: 0-2k"),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            linestyle="--",
            label="Avg. Sample Mutation Count Across Other Subtypes",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=24,
        frameon=False,
    )

    dout = Path(fout).parent
    dout.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        fout,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_cbase_driver_decoy_gene_fractions(
    subtype_decoy_gene_fractions: dict,
    fout: str,
) -> None:
    """TODO: Add docstring."""
    data_df = pd.DataFrame(
        list(subtype_decoy_gene_fractions.items()),
        columns=["Subtype", "Decoy Fraction"],
    )

    data_df["Decoy Fraction"] = data_df["Decoy Fraction"].replace(0, 0.001)
    data_df = data_df.sort_values("Decoy Fraction", ascending=False)

    sns.set_context("talk", rc={"axes.linewidth": 0.8, "grid.linewidth": 0.5})

    plt.figure(figsize=(6, 10))
    ax = sns.barplot(
        data=data_df,
        y="Subtype",
        x="Decoy Fraction",
        palette="Greys_r",
        edgecolor="black",
        hue="Subtype",
        legend=False,
    )
    apply_tufte_style(ax)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    for container, label in zip(ax.containers, data_df["Subtype"]):
        ax.bar_label(
            container,
            labels=[label] * len(container),
            fontsize=14,
            label_type="edge",
            padding=3,
        )

    plt.title(
        "Fraction of Likely Passenger Genes in Top 50",
        fontsize=20,
        pad=10,
    )
    plt.xlabel("Likely Passenger Fraction", fontsize=18)
    plt.tight_layout()

    plt.savefig(fout, transparent=True)


def plot_cbase_top_decoy_genes_upset(
    subtype_to_high_ranked_decoys: dict,
    high_ranked_decoy_freqs: dict,
    top_n: int,
    fout: str,
) -> None:
    """TODO: Add docstring."""
    top_genes = sorted(
        high_ranked_decoy_freqs,
        key=high_ranked_decoy_freqs.get,
        reverse=True,
    )[:top_n]
    contents = {}
    for gene in top_genes:
        subtypes_with_gene = [
            subtype
            for subtype, decoys in subtype_to_high_ranked_decoys.items()
            if gene in decoys
        ]
        contents[gene] = set(subtypes_with_gene)

    data_df = from_contents(contents)

    upset = UpSet(data_df, totals_plot_elements=0, element_size=40)
    plt.figure(figsize=(16, 8))
    subplots = upset.plot()
    subplots["intersections"].set_ylabel("Number of Subtypes")
    subplots["matrix"].set_ylabel("Likely Passengers")
    plt.savefig(fout, transparent=True)
    plt.close()


def plot_cbase_driver_and_passenger_mutation_counts(
    decoys: set,
    drivers: set,
    res_df: pd.DataFrame,
    subtype: str,
) -> None:
    """TODO: Add docstring."""
    decoy_df = res_df[res_df["Gene Name"].isin(decoys)]
    top5_decoys = decoy_df.nlargest(5, "Observed Mutations")
    driver_df = res_df[res_df["Gene Name"].isin(drivers)]
    top5_drivers = driver_df.nlargest(5, "Observed Mutations")
    top10 = pd.concat([top5_decoys, top5_drivers], ignore_index=True)
    top10 = top10.sort_values("Observed Mutations", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    apply_tufte_style(ax)
    x = np.arange(len(top10))
    bar_width = 0.4
    ax.bar(
        x - bar_width / 2,
        top10["Observed Mutations"],
        width=bar_width,
        color="black",
        label="Observed",
    )
    ax.bar(
        x + bar_width / 2,
        top10["Expected Mutations"],
        width=bar_width,
        color="slategray",
        label="Expected",
    )
    ax.set_xlabel("Gene", fontsize=14)
    ax.set_ylabel("Mutation Count", fontsize=14)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xticklabels(top10["Gene Name"], rotation=25, ha="right")
    for label, gene in zip(ax.get_xticklabels(), top10["Gene Name"]):
        if gene in drivers:
            label.set_bbox(
                {
                    "facecolor": "#9999FF",
                    "alpha": 0.35,
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.3",
                },
            )
        else:
            label.set_bbox(
                {
                    "facecolor": "#FF9999",
                    "alpha": 0.35,
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.3",
                },
            )
    driver_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#9999FF", alpha=0.35)
    passenger_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#FF9999",
        alpha=0.35,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles += [driver_patch, passenger_patch]
    labels += ["Putative Driver", "Likely Passenger"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
    )
    plt.tight_layout()
    plt.savefig(
        f"figures/cbase_obs_exp_plots/{subtype}_cbase_driver_vs_passenger_counts.png",
    )


def plot_mtx_sim_pr_curve(
    methods: dict,
    y_true: np.ndarray,
    fout: str,
) -> None:
    """TODO: Add docstring."""
    figsize = (10, 8)
    plt.figure(figsize=figsize)
    set_dynamic_font_sizes(figsize=figsize)
    for method_name, scores in methods.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{method_name} (AUC={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(fout, dpi=300)
    plt.close()

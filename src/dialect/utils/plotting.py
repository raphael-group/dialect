"""TODO: Add docstring."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator
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

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
#                                  CBASE BMR ANALYSIS                                  #
# ------------------------------------------------------------------------------------ #
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


# ------------------------------------------------------------------------------------ #
#                        TOP RANKED LIKELY PASSENGER PROPORTION                        #
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


# ------------------------------------------------------------------------------------ #
#                            MUTATION FREQUENCY DISTRIBUTION                           #
# ------------------------------------------------------------------------------------ #
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


# ------------------------------------------------------------------------------------ #
#                              SET DEFAULT PLOTTING STYLE                              #
# ------------------------------------------------------------------------------------ #
rcParams["text.usetex"] = True
rcParams["font.family"] = "sans-serif"
rcParams["font.serif"] = ["Computer Modern"]
rcParams["font.sans-serif"] = ["Computer Modern"]


EDGE_COLOR = "black"
EPSILON_MUTATION_COUNT = 10
PVALUE_THRESHOLD = 1


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

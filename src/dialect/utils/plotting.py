import os
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, BoxStyle, Patch

from dialect.utils.postprocessing import generate_top_ranking_tables
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    geom_point,
    theme,
    theme_tufte,
    element_text,
    labs,
    scale_shape_manual,
    scale_color_manual,
    guides,
    guide_legend,
    position_jitter,
    ylim,
)

# ---------------------------------------------------------------------------- #
#                          SET DEFAULT PLOTTING STYLE                          #
# ---------------------------------------------------------------------------- #
rcParams["font.family"] = "sans-serif"
rcParams["font.serif"] = ["Computer Modern"]
rcParams["font.sans-serif"] = ["Computer Modern"]
rcParams["text.usetex"] = True  # Optional for LaTeX-like rendering


DEFAULT_GENE_COLOR = "#D3D3D3"  # Lighter light gray for generic genes
DECOY_GENE_COLOR = "#FFB3B3"  # Pastel red for decoy genes
DRIVER_GENE_COLOR = "#A3C1DA"  # Blue-gray for driver genes
EDGE_COLOR = "black"
EPSILON_MUTATION_COUNT = 10  # minimum count of mutations
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


def apply_tufte_style(ax):
    """
    Hides top/right spines, moves left/bottom outward, larger tick labels.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))
    ax.tick_params(axis="both", which="major", labelsize=20)


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def draw_network_plot(ax, G, pos, labels=None, node_colors=None):
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE_COLOR, width=2)

    if node_colors is None:
        node_colors = [DEFAULT_GENE_COLOR] * len(G.nodes)

    for node, (x, y), color in zip(G.nodes, pos.values(), node_colors):
        label = labels[node] if labels else str(node)
        bbox = FancyBboxPatch(
            (x - 0.08, y),
            0.16,  # width
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
            bbox=dict(facecolor="none", edgecolor="none"),
        )
    ax.set_aspect("equal")
    ax.axis("off")


def plot_subtype_histogram(ax, subtype, data_series, avg_across_others):
    """
    Plots a histogram of mutation counts (x-axis) for 'subtype' on log-scale Y.
    Draws a red dashed vertical line at 'avg_across_others', if within x-lim.
    """
    color = COLOR_MAPPING[subtype]
    xlim = XLIM_MAPPING[subtype]

    ax.hist(
        data_series,
        bins=20,
        range=(0, xlim),
        # ylim
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


# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
def plot_decoy_gene_fractions(
    data_filepath,
    num_pairs,
    is_me,
    out_dir,
):

    df = pd.read_csv(data_filepath)
    if not is_me:  # remove MEGSA from plot for CO analysis
        df = df[df["Method"] != "MEGSA"]

    subtypes = df["Subtype"].unique()
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
    shapes = ["o", "s", "^", "D"]  # Circle, square, triangle, diamond

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
            df,
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
        + scale_color_manual(values=color_mapping)  # Custom color mapping
        + scale_shape_manual(values=shape_mapping)  # Custom shape mapping
        + labs(
            title=f"Proportion of Top-Ranked {ixn_type} Pairs with Decoy Genes by Method",
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

    # Step 4: Save Plot
    os.makedirs(out_dir, exist_ok=True)
    plot.save(f"{out_dir}/{ixn_type}_decoy_gene_fractions_boxplot.svg", dpi=300)
    print(
        f"Plot saved to {out_dir}/{ixn_type}_decoy_gene_fractions_boxplot.svg"
    )


def draw_network_gridplot_across_methods(
    num_edges,
    subtype,
    driver_genes,
    decoy_genes,
    results_df,
    num_samples,
    is_me,
):
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
            else top_ranking_pairs[["Gene A", "Gene B"]].values
        )
        G = nx.Graph()
        G.add_edges_from(edges)

        if G.number_of_edges() == 0:
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
        for node in G.nodes:
            if node in decoy_genes:
                node_colors.append(DECOY_GENE_COLOR)
            elif node in driver_genes:
                node_colors.append(DRIVER_GENE_COLOR)
            else:
                node_colors.append(DEFAULT_GENE_COLOR)

        if G.number_of_nodes() == 2:
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = nx.circular_layout(G)

        labels = {node: node for node in G.nodes}
        draw_network_plot(ax, G, pos, labels, node_colors)

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
    fout = f"figures/{subtype}_{ranking_type}_network_plots_across_methods.png"
    plt.savefig(fout, dpi=300)


def plot_sample_mutation_count_subtype_histograms(
    subtype_sample_mut_counts,
    avg_mut_cnt,
    fout,
):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))
    axes = axes.flatten()

    for i, (subtype, data_series) in enumerate(
        subtype_sample_mut_counts.items()
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
            facecolor="lightcoral", edgecolor="black", label="Mut Cnt: 0–30k"
        ),
        Patch(facecolor="moccasin", edgecolor="black", label="Mut Cnt: 0–10k"),
        Patch(facecolor="khaki", edgecolor="black", label="Mut Cnt: 0–2k"),
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

    os.makedirs(os.path.dirname(fout), exist_ok=True)
    fig.savefig(
        fout,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    print(f"3×2 histogram figure saved as SVG to: {fout}")


def plot_cbase_driver_decoy_gene_fractions(subtype_decoy_gene_fractions, fout):
    df = pd.DataFrame(
        list(subtype_decoy_gene_fractions.items()),
        columns=["Subtype", "Decoy Fraction"],
    )

    df["Decoy Fraction"] = df["Decoy Fraction"].replace(0, 0.001)
    df.sort_values("Decoy Fraction", ascending=False, inplace=True)

    sns.set_context("talk", rc={"axes.linewidth": 0.8, "grid.linewidth": 0.5})

    plt.figure(figsize=(6, 10))
    ax = sns.barplot(
        data=df,
        y="Subtype",
        x="Decoy Fraction",
        palette="Greys_r",
        edgecolor="black",
        hue="Subtype",
        legend=False,
        # edges="black",
    )
    apply_tufte_style(ax)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    for container, label in zip(ax.containers, df["Subtype"]):
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

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, BoxStyle
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
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Computer Modern"]
rcParams["text.usetex"] = True  # Optional for LaTeX-like rendering


DEFAULT_GENE_COLOR = "#D3D3D3"  # Lighter light gray for generic genes
DECOY_GENE_COLOR = "#FFB3B3"  # Pastel red for decoy genes
DRIVER_GENE_COLOR = "#A3C1DA"  # Blue-gray for driver genes
EDGE_COLOR = "black"
EPSILON_MUTATION_COUNT = 10  # minimum count of mutations
PVALUE_THRESHOLD = 1


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
    colors = ["green", "blue", "red", "purple", "yellow", "orange", "black", "brown"]
    shapes = ["o", "s", "^", "D"]  # Circle, square, triangle, diamond

    color_shape_combinations = list(product(colors, shapes))
    color_shape_mapping = {subtypes[i]: color_shape_combinations[i] for i in range(len(subtypes))}
    color_mapping = {subtype: combo[0] for subtype, combo in color_shape_mapping.items()}
    shape_mapping = {subtype: combo[1] for subtype, combo in color_shape_mapping.items()}
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
    print(f"Plot saved to {out_dir}/{ixn_type}_decoy_gene_fractions_boxplot.svg")


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
        f"Top 10 Ranked ME Pairs in {subtype}" if is_me else f"Top 10 Ranked CO Pairs in {subtype}"
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
        edges = [] if top_ranking_pairs is None else top_ranking_pairs[["Gene A", "Gene B"]].values
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
                [0], [0], linestyle="-", color="black", linewidth=4, label="Mutual Exclusivity"
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

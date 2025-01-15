import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, BoxStyle
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
            bbox=dict(facecolor="none", edgecolor="none"),
        )
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
def plot_decoy_gene_fractions(data_filepath, out_dir):
    df = pd.read_csv(data_filepath)

    subtypes = df["Subtype"].unique()
    colors = ["green", "blue", "red", "purple", "yellow", "orange", "black", "brown"]
    shapes = ["o", "s", "^", "D"]  # Circle, square, triangle, diamond

    color_shape_combinations = list(product(colors, shapes))
    color_shape_mapping = {subtypes[i]: color_shape_combinations[i] for i in range(len(subtypes))}
    color_mapping = {subtype: combo[0] for subtype, combo in color_shape_mapping.items()}
    shape_mapping = {subtype: combo[1] for subtype, combo in color_shape_mapping.items()}

    # Create the Plot
    plot = (
        ggplot(
            df,
            aes(x="Method", y="Fraction", color="Subtype", shape="Subtype"),
        )
        + geom_boxplot(
            aes(group="Method"),
            alpha=0.5,
            outlier_alpha=0,
            # TODO set y limit to 1
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
            title="Proportion of Top-Ranked Pairs with Decoy Genes by Method",
            x="Method",
            y="Proportion of Top 100 Pairs with Decoy Genes",
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
    )

    # Step 4: Save Plot
    os.makedirs(out_dir, exist_ok=True)
    plot.save(f"{out_dir}/decoy_gene_fractions_boxplot.svg", dpi=300)
    print(f"Plot saved to {out_dir}/decoy_gene_fractions_boxplot.svg")


def draw_network_gridplot_across_methods(
    top_k,
    subtype,
    driver_genes,
    decoy_genes,
    results_df,
    num_samples,
):
    # TODO: move this to shared location to use across this and decoy gene plot
    methods = {
        "DIALECT": "Rho",
        "DISCOVER": "Discover ME P-Val",
        "Fisher's Exact Test": "Fisher's ME P-Val",
        "MEGSA": "MEGSA S-Score (LRT)",
        "WeSME": "WeSME P-Val",
    }

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(f"Top 10 Ranked ME Pairs in {subtype}", fontsize=42, y=0.999)
    for idx, (method, col) in enumerate(methods.items()):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(method, fontsize=36)
        top_ranking_pairs = results_df.sort_values(
            col, ascending=col != "MEGSA S-Score (LRT)"
        ).head(top_k)

        if method == "DIALECT":
            # i want to set the epsilon proportionally to the number of samples
            epsilon = EPSILON_MUTATION_COUNT / num_samples
            top_ranking_pairs = top_ranking_pairs[
                (top_ranking_pairs["Rho"] < 0)
                & (top_ranking_pairs["Tau_1X"] > epsilon)
                & (top_ranking_pairs["Tau_X1"] > epsilon)
            ]
        elif method == "MEGSA":
            top_ranking_pairs = top_ranking_pairs[top_ranking_pairs["MEGSA S-Score (LRT)"] > 0]
        elif method == "DISCOVER":
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["Discover ME P-Val"] < PVALUE_THRESHOLD
            ]
        elif method == "Fisher's Exact Test":
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["Fisher's ME P-Val"] < PVALUE_THRESHOLD
            ]
        elif method == "WeSME":
            top_ranking_pairs = top_ranking_pairs[
                top_ranking_pairs["WeSME P-Val"] < PVALUE_THRESHOLD
            ]

        edges = top_ranking_pairs[["Gene A", "Gene B"]].values
        G = nx.Graph()
        G.add_edges_from(edges)

        node_colors = []
        for node in G.nodes:
            if node in decoy_genes:
                node_colors.append(DECOY_GENE_COLOR)
            elif node in driver_genes:
                node_colors.append(DRIVER_GENE_COLOR)
            else:
                node_colors.append(DEFAULT_GENE_COLOR)

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
            ),  # Driver gene
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=DEFAULT_GENE_COLOR,
                markersize=50,
                linestyle="None",
                label="Passenger Gene",
            ),  # Default gene
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=DECOY_GENE_COLOR,
                markersize=50,
                linestyle="None",
                label="Decoy Passenger Gene",
            ),  # Decoy gene
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
        fontsize=24,  # Larger font size
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(f"figures/{subtype}_network_plots_across_methods.png", dpi=300)

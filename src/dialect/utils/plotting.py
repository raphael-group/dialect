import os
import pandas as pd

from itertools import product
from matplotlib import rcParams
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


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def plotnine_decoy_gene_fractions(data_filepath, out_dir):
    df = pd.read_csv(data_filepath)

    subtypes = df["Subtype"].unique()
    colors = ["green", "blue", "red", "purple", "yellow", "orange", "black", "brown"]
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

    # Create the Plot
    plot = (
        ggplot(df, aes(x="Method", y="Fraction", color="Subtype", shape="Subtype"))
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
    plot.save(f"{out_dir}/decoy_gene_fractions_boxplot_plotnine.svg", dpi=300)
    print(f"Plot saved to {out_dir}/decoy_gene_fractions_boxplot_plotnine.svg")

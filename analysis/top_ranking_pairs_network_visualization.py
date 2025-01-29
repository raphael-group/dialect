"""TODO: Add docstring."""

import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from dialect.utils.plotting import draw_network_gridplot_across_methods


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def build_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
    parser = ArgumentParser(description="Decoy Gene Analysis")
    parser.add_argument(
        "-n",
        "--num_edges",
        type=int,
        default=10,
        help="Number of top ranking pairs to visualize",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Directory with results for all subtypes",
    )
    parser.add_argument(
        "-dvr",
        "--driver_genes_fn",
        type=str,
        default="data/references/OncoKB_Cancer_Gene_List.tsv",
        help="File with driver genes",
    )
    parser.add_argument(
        "-d",
        "--decoy_genes_dir",
        type=str,
        default="data/decoy_genes",
        help="Directory with all decoy gene files",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="figures/network_plots",
        help="Output directory",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--me",
        action="store_true",
        help="Perform analysis for mutual exclusivity",
    )
    group.add_argument(
        "--co",
        action="store_true",
        help="Perform analysis for co-occurrence",
    )
    return parser


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_argument_parser()
    args = parser.parse_args()

    drvr_df = pd.read_csv(args.driver_genes_fn, sep="\t", index_col=0)
    driver_genes = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    subtypes = os.listdir(args.results_dir)
    for subtype in subtypes:
        results_fn = (
            Path(args.results_dir) / subtype / "complete_pairwise_ixn_results.csv"
        )
        cnt_mtx_fn = Path(args.results_dir) / subtype / "count_matrix.csv"
        decoy_genes_fn = Path(args.decoy_genes_dir) / f"{subtype}_decoy_genes.txt"
        if not results_fn.exists() or not decoy_genes_fn.exists():
            continue
        results_df = pd.read_csv(results_fn)
        decoy_genes = set(
            pd.read_csv(decoy_genes_fn, header=None, names=["Gene"])["Gene"],
        )
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        draw_network_gridplot_across_methods(
            args.num_edges,
            subtype,
            driver_genes,
            decoy_genes,
            results_df,
            num_samples,
            args.me,
            args.out,
        )


if __name__ == "__main__":
    main()

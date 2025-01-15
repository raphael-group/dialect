import os
import logging
import pandas as pd

from argparse import ArgumentParser
from dialect.utils.plotting import draw_network_gridplot_across_methods


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def build_argument_parser():
    parser = ArgumentParser(description="Decoy Gene Analysis")
    parser.add_argument(
        "-k",
        "--top_k",
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
        required=True,
        help="File with driver genes",
    )
    parser.add_argument(
        "-d",
        "--decoy_genes_dir",
        type=str,
        required=True,
        help="Directory with all decoy gene files",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="Output directory",
    )
    return parser


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()

    drvr_df = pd.read_csv(args.driver_genes_fn, sep="\t", index_col=0)
    driver_genes = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    subtypes = os.listdir(args.results_dir)
    for subtype in subtypes:
        results_fn = os.path.join(args.results_dir, subtype, "complete_pairwise_ixn_results.csv")
        cnt_mtx_fn = os.path.join(args.results_dir, subtype, "count_matrix.csv")
        decoy_genes_fn = os.path.join(args.decoy_genes_dir, f"{subtype}_decoy_genes.txt")
        if not os.path.exists(results_fn) or not os.path.exists(decoy_genes_fn):
            logging.info(f"Skipping {subtype} since input files not found")
            continue
        results_df = pd.read_csv(results_fn)
        decoy_genes = set(pd.read_csv(decoy_genes_fn, header=None, names=["Gene"])["Gene"])
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        draw_network_gridplot_across_methods(
            args.top_k,
            subtype,
            driver_genes,
            decoy_genes,
            results_df,
            num_samples,
        )

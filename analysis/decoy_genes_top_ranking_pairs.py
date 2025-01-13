import os
import logging
import pandas as pd

from argparse import ArgumentParser
from dialect.utils.plotting import plot_decoy_gene_fractions


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def build_argument_parser():
    parser = ArgumentParser(description="Decoy Gene Analysis")
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Number of top ranking pairs to analyze",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Directory with results for all subtypes",
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
#                                MAIN FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
def compute_decoy_gene_fraction_across_methods(ixn_res_df, decoy_genes, k):
    if ixn_res_df.empty:
        raise ValueError("Input DataFrame is empty")

    methods = {
        "DIALECT": "Rho",
        "DISCOVER": "Discover ME Q-Val",
        "Fisher's Exact Test": "Fisher's ME Q-Val",
        "MEGSA": "MEGSA S-Score (LRT)",
        "WeSME": "WeSME Q-Val",
    }

    fractions = {}
    for method, column in methods.items():
        top_ranking = ixn_res_df.sort_values(
            column, ascending=column != "MEGSA S-Score (LRT)"
        ).head(k)

        pairs_with_at_least_one_decoy_gene = (
            top_ranking["Gene A"].isin(decoy_genes)
            | top_ranking["Gene B"].isin(decoy_genes)
        ).sum()
        fractions[method] = pairs_with_at_least_one_decoy_gene / top_ranking.shape[0]

    return fractions


def compute_decoy_gene_fractions_across_subtypes(results_dir, decoy_genes_dir, top_k):
    subtypes = os.listdir(results_dir)
    subtype_decoy_gene_fractions = {}
    for subtype in subtypes:
        results_fn = os.path.join(
            results_dir, subtype, "complete_pairwise_ixn_results.csv"
        )
        decoy_genes_fn = os.path.join(decoy_genes_dir, f"{subtype}_decoy_genes.txt")
        if not os.path.exists(results_fn) or not os.path.exists(decoy_genes_fn):
            logging.info(f"Skipping {subtype} since input files not found")
            continue
        ixn_res_df = pd.read_csv(results_fn)
        decoy_genes = set(
            pd.read_csv(decoy_genes_fn, header=None, names=["Gene"])["Gene"]
        )
        subtype_decoy_gene_fractions[subtype] = (
            compute_decoy_gene_fraction_across_methods(
                ixn_res_df,
                decoy_genes,
                k=top_k,
            )
        )

    return subtype_decoy_gene_fractions


def save_output(subtype_decoy_gene_fractions, fout):
    gene_fraction_data = [
        {"Subtype": subtype, "Method": method, "Fraction": fraction}
        for subtype, fractions in subtype_decoy_gene_fractions.items()
        for method, fraction in fractions.items()
    ]
    df = pd.DataFrame(gene_fraction_data)
    df.to_csv(fout, index=False)


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()

    subtype_decoy_gene_fractions = compute_decoy_gene_fractions_across_subtypes(
        args.results_dir,
        args.decoy_genes_dir,
        args.top_k,
    )
    fout = os.path.join(args.out, "decoy_gene_fractions_by_method.csv")
    save_output(subtype_decoy_gene_fractions, fout)
    plot_decoy_gene_fractions(fout, args.out)

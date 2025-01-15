import os
import logging
import pandas as pd

from argparse import ArgumentParser
from dialect.utils.plotting import plot_decoy_gene_fractions

EPSILON_MUTATION_COUNT = 10
PVALUE_THRESHOLD = 1  # Treshold for other methods


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
def compute_decoy_gene_fraction_across_methods(
    ixn_res_df,
    decoy_genes,
    num_samples,
    k,
):
    if ixn_res_df.empty:
        raise ValueError("Input DataFrame is empty")

    methods = {
        "DIALECT": "Rho",
        "DISCOVER": "Discover ME P-Val",
        "Fisher's Exact Test": "Fisher's ME P-Val",
        "MEGSA": "MEGSA S-Score (LRT)",
        "WeSME": "WeSME P-Val",
    }

    fractions = {}
    for method, column in methods.items():
        top_ranking_pairs = ixn_res_df.sort_values(
            column, ascending=column != "MEGSA S-Score (LRT)"
        ).head(k)

        # TODO: UNIFY THIS INTO A HELPER FUNCTION TO USE ACROSS ANALYSIS MODULES
        if method == "DIALECT":
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

        pairs_with_at_least_one_decoy_gene = (
            top_ranking_pairs["Gene A"].isin(decoy_genes)
            | top_ranking_pairs["Gene B"].isin(decoy_genes)
        ).sum()
        if top_ranking_pairs.shape[0] == 0:
            fractions[method] = 0
        else:
            fractions[method] = pairs_with_at_least_one_decoy_gene / top_ranking_pairs.shape[0]

    return fractions


def compute_decoy_gene_fractions_across_subtypes(
    results_dir,
    decoy_genes_dir,
    top_k,
):
    subtypes = os.listdir(results_dir)
    subtype_decoy_gene_fractions = {}
    for subtype in subtypes:
        results_fn = os.path.join(results_dir, subtype, "complete_pairwise_ixn_results.csv")
        cnt_mtx_fn = os.path.join(args.results_dir, subtype, "count_matrix.csv")
        decoy_genes_fn = os.path.join(decoy_genes_dir, f"{subtype}_decoy_genes.txt")
        if not os.path.exists(results_fn) or not os.path.exists(decoy_genes_fn):
            logging.info(f"Skipping {subtype} since input files not found")
            continue
        ixn_res_df = pd.read_csv(results_fn)
        decoy_genes = set(pd.read_csv(decoy_genes_fn, header=None, names=["Gene"])["Gene"])
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        subtype_decoy_gene_fractions[subtype] = compute_decoy_gene_fraction_across_methods(
            ixn_res_df,
            decoy_genes,
            num_samples,
            k=top_k,
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

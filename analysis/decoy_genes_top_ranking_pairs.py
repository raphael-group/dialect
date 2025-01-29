"""TODO: Add docstring."""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from dialect.utils.plotting import plot_decoy_gene_fractions
from dialect.utils.postprocessing import generate_top_ranking_tables

EPSILON_MUTATION_COUNT = 10
PVALUE_THRESHOLD = 1

# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def build_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
    parser = ArgumentParser(description="Decoy Gene Analysis")
    parser.add_argument(
        "-n",
        "--num_pairs",
        type=int,
        default=10,
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
        default="data/decoy_genes",
        help="Directory with all decoy gene files",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="output/RESULTS",
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


def compute_prop_pairs_with_at_least_one_decoy(
    decoy_genes: set,
    top_ranking_pairs: pd.DataFrame,
) -> float:
    """TODO: Add docstring."""
    pairs_with_at_least_one_decoy_gene = (
        top_ranking_pairs["Gene A"].isin(decoy_genes)
        | top_ranking_pairs["Gene B"].isin(decoy_genes)
    ).sum()
    total_pairs = top_ranking_pairs.shape[0]
    return pairs_with_at_least_one_decoy_gene / total_pairs


def compute_prop_unique_decoy_genes_in_top_pairs(
    decoy_genes: set,
    top_ranking_pairs: pd.DataFrame,
) -> float:
    """TODO: Add docstring."""
    total_unique_genes = set(
        top_ranking_pairs["Gene A"].tolist() + top_ranking_pairs["Gene B"].tolist(),
    )
    total_unique_decoy_genes = len(total_unique_genes.intersection(decoy_genes))
    return total_unique_decoy_genes / len(total_unique_genes)


def compute_prop_decoy_genes_in_top_pairs(
    decoy_genes: set,
    top_ranking_pairs: pd.DataFrame,
) -> float:
    """TODO: Add docstring."""
    all_genes_list = (
        top_ranking_pairs["Gene A"].tolist() + top_ranking_pairs["Gene B"].tolist()
    )
    total_decoy_genes = len([x for x in all_genes_list if x in decoy_genes])
    return total_decoy_genes / len(all_genes_list)


# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def compute_decoy_gene_fraction_across_methods(
    ixn_res_df: pd.DataFrame,
    decoy_genes: set,
    num_samples: int,
    num_pairs: int,
    ixn_type: str,
    comp_scheme: int = 3,
):
    """TODO: Add docstring."""
    if ixn_res_df.empty:
        msg = "Input DataFrame is empty"
        raise ValueError(msg)

    top_tables = generate_top_ranking_tables(
        results_df=ixn_res_df,
        ixn_type=ixn_type,
        num_pairs=num_pairs,
        num_samples=num_samples,
    )
    proportions = {}
    for method, top_ranking_pairs in top_tables.items():
        if top_ranking_pairs is None or top_ranking_pairs.empty:
            decoy_gene_proportion = 0
        elif comp_scheme == 1:
            decoy_gene_proportion = compute_prop_pairs_with_at_least_one_decoy(
                decoy_genes,
                top_ranking_pairs,
            )
        elif comp_scheme == 2:
            decoy_gene_proportion = compute_prop_unique_decoy_genes_in_top_pairs(
                decoy_genes,
                top_ranking_pairs,
            )
        else:
            decoy_gene_proportion = compute_prop_decoy_genes_in_top_pairs(
                decoy_genes,
                top_ranking_pairs,
            )
        proportions[method] = decoy_gene_proportion

    return proportions


def compute_decoy_gene_fractions_across_subtypes(
    results_dir: str,
    decoy_genes_dir: str,
    num_pairs: int,
    ixn_type: str,
) -> dict:
    """TODO: Add docstring."""
    subtypes = os.listdir(results_dir)
    subtype_decoy_gene_fractions = {}
    for subtype in subtypes:
        results_fn = Path(results_dir) / subtype / "complete_pairwise_ixn_results.csv"
        cnt_mtx_fn = Path(results_dir) / subtype / "count_matrix.csv"
        decoy_genes_fn = Path(decoy_genes_dir) / f"{subtype}_decoy_genes.txt"
        if not results_fn.exists() or not decoy_genes_fn.exists():
            continue
        ixn_res_df = pd.read_csv(results_fn)
        decoy_genes = set(
            pd.read_csv(decoy_genes_fn, header=None, names=["Gene"])["Gene"],
        )
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        subtype_decoy_gene_fractions[subtype] = (
            compute_decoy_gene_fraction_across_methods(
                ixn_res_df,
                decoy_genes,
                num_samples,
                num_pairs,
                ixn_type,
            )
        )

    return subtype_decoy_gene_fractions


def save_output(subtype_decoy_gene_fractions: dict, fout: str) -> None:
    """TODO: Add docstring."""
    gene_fraction_data = [
        {"Subtype": subtype, "Method": method, "Fraction": fraction}
        for subtype, fractions in subtype_decoy_gene_fractions.items()
        for method, fraction in fractions.items()
    ]
    results_df = pd.DataFrame(gene_fraction_data)
    results_df.to_csv(fout, index=False)


def main() -> None:
    """TODO: Add docstring."""
    parser = build_argument_parser()
    args = parser.parse_args()

    ixn_type = "ME" if args.me else "CO"
    subtype_decoy_gene_fractions = compute_decoy_gene_fractions_across_subtypes(
        args.results_dir,
        args.decoy_genes_dir,
        args.num_pairs,
        ixn_type,
    )
    fout = Path(args.out) / f"{ixn_type}_decoy_gene_fractions_by_method.csv"
    save_output(subtype_decoy_gene_fractions, fout)
    plot_decoy_gene_fractions(
        fout,
        args.num_pairs,
        args.me,
        args.out,
    )


if __name__ == "__main__":
    main()

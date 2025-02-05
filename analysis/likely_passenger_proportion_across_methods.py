"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from dialect.utils.helpers import load_likely_passenger_genes
from dialect.utils.plotting import draw_likely_passenger_gene_proportion_violinplot
from dialect.utils.postprocessing import (
    generate_top_ranked_co_interaction_tables,
    generate_top_ranked_me_interaction_tables,
)

# ------------------------------------------------------------------------------------ #
#                                       CONSTANTS                                      #
# ------------------------------------------------------------------------------------ #
ME_METHODS = ["DIALECT", "DISCOVER", "Fisher's Exact Test", "MEGSA", "WeSME"]
CO_METHODS = ["DIALECT", "DISCOVER", "Fisher's Exact Test", "WeSME"]


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def build_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_pairs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-lp",
        "--likely_passenger_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        required=True,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-me",
        "--mutual_exclusivity",
        action="store_true",
    )
    group.add_argument(
        "-co",
        "--cooccurrence",
        action="store_true",
    )
    return parser


def compute_num_likely_passengers_in_top_ranked_pairs(
    likely_passengers: set,
    top_ranking_pairs: pd.DataFrame,
) -> float:
    """TODO: Add docstring."""
    all_genes_list = (
        top_ranking_pairs["Gene A"].tolist() + top_ranking_pairs["Gene B"].tolist()
    )
    total_likely_passengers = len([x for x in all_genes_list if x in likely_passengers])
    return (
        0 if not len(all_genes_list) else total_likely_passengers / len(all_genes_list)
    )


def compute_likely_passenger_proportions(
    results_dir: Path,
    subtype_to_likely_passengers: dict,
    num_pairs: int,
    methods: list,
    generate_top_ranked_interaction_table: callable,
) -> dict:
    """TODO: Add docstring."""
    method_to_subtype_to_likely_passenger_proportion = {}
    for subtype_dir in results_dir.iterdir():
        subtype = subtype_dir.name
        cnt_mtx_fn = subtype_dir / "count_matrix.csv"
        if not cnt_mtx_fn.exists():
            continue
        results_df = pd.read_csv(subtype_dir / "complete_pairwise_ixn_results.csv")
        likely_passengers = subtype_to_likely_passengers[subtype]
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        method_to_top_ranked_interaction_table = generate_top_ranked_interaction_table(
            results_df=results_df,
            num_pairs=num_pairs,
            num_samples=num_samples,
            methods=methods,
        )
        for (
            method,
            top_ranked_me_interaction_table,
        ) in method_to_top_ranked_interaction_table.items():
            if method not in method_to_subtype_to_likely_passenger_proportion:
                method_to_subtype_to_likely_passenger_proportion[method] = {}
            likely_passenger_proportion = (
                compute_num_likely_passengers_in_top_ranked_pairs(
                    likely_passengers,
                    top_ranked_me_interaction_table,
                )
            )
            method_to_subtype_to_likely_passenger_proportion[method][subtype] = (
                likely_passenger_proportion
            )
    return method_to_subtype_to_likely_passenger_proportion


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_argument_parser()
    args = parser.parse_args()
    subtype_to_likely_passengers = load_likely_passenger_genes(
        args.likely_passenger_dir,
    )
    if args.mutual_exclusivity:
        method_to_subtype_to_proportion = compute_likely_passenger_proportions(
            results_dir=args.results_dir,
            subtype_to_likely_passengers=subtype_to_likely_passengers,
            num_pairs=args.num_pairs,
            methods=ME_METHODS,
            generate_top_ranked_interaction_table=generate_top_ranked_me_interaction_tables,
        )
    else:  # args.cooccurrence
        method_to_subtype_to_proportion = compute_likely_passenger_proportions(
            results_dir=args.results_dir,
            subtype_to_likely_passengers=subtype_to_likely_passengers,
            num_pairs=args.num_pairs,
            methods=CO_METHODS,
            generate_top_ranked_interaction_table=generate_top_ranked_co_interaction_tables,
        )
    draw_likely_passenger_gene_proportion_violinplot(
        method_to_subtype_to_proportion,
        out_fn=args.out_dir / "likely_passenger_proportion_violinplot",
    )


if __name__ == "__main__":
    main()

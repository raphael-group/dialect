"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from dialect.utils.plotting import (
    draw_single_me_and_co_interaction_network,
    draw_single_me_or_co_interaction_network,
)
from dialect.utils.postprocessing import (
    generate_top_ranked_co_interaction_tables,
    generate_top_ranked_me_interaction_tables,
)


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
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--driver_genes_fn",
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
        "--out",
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
    group.add_argument(
        "-both",
        "--both",
        action="store_true",
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
    putative_drivers = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    subtypes = [subtype.name for subtype in args.results_dir.iterdir()]
    num_edges = args.num_edges // 2 if args.both else args.num_edges
    for subtype in subtypes:
        results_fn = args.results_dir / subtype / "complete_pairwise_ixn_results.csv"
        cnt_mtx_fn = args.results_dir / subtype / "count_matrix.csv"
        likely_passenger_fn = args.likely_passenger_dir / f"{subtype}.txt"
        if not results_fn.exists() or not likely_passenger_fn.exists():
            continue
        results_df = pd.read_csv(results_fn)
        likely_passengers = set(
            pd.read_csv(likely_passenger_fn, header=None, names=["Gene"])["Gene"],
        )
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]

        top_ranked_me_interactions_by_method = (
            generate_top_ranked_me_interaction_tables(
                results_df=results_df,
                num_pairs=num_edges,
                num_samples=num_samples,
                methods=[
                    "DIALECT",
                    "DISCOVER",
                    "Fisher's Exact Test",
                    "MEGSA",
                    "WeSME",
                ],
            )
        )
        top_ranked_co_interactions_by_method = (
            generate_top_ranked_co_interaction_tables(
                results_df=results_df,
                num_pairs=num_edges,
                num_samples=num_samples,
                methods=[
                    "DIALECT",
                    "DISCOVER",
                    "Fisher's Exact Test",
                    "WeSME",
                ],
            )
        )
        for method in top_ranked_co_interactions_by_method:
            top_ranked_me_pairs = top_ranked_me_interactions_by_method[method]
            top_ranked_co_pairs = top_ranked_co_interactions_by_method[method]
            top_ranked_pairs = (
                top_ranked_me_pairs if args.mutual_exclusivity else top_ranked_co_pairs
            )
            fout = f"{args.out}/{subtype}_{method}_network"
            if args.mutual_exclusivity or args.cooccurrence:
                draw_single_me_or_co_interaction_network(
                    edges=top_ranked_pairs[["Gene A", "Gene B"]].to_numpy(),
                    putative_drivers=putative_drivers,
                    likely_passengers=likely_passengers,
                    method=method,
                    fout=fout,
                )
            else:
                draw_single_me_and_co_interaction_network(
                    me_edges=top_ranked_me_pairs[["Gene A", "Gene B"]].to_numpy(),
                    co_edges=top_ranked_co_pairs[["Gene A", "Gene B"]].to_numpy(),
                    putative_drivers=putative_drivers,
                    likely_passengers=likely_passengers,
                    method=method,
                    fout=fout,
                )


if __name__ == "__main__":
    main()

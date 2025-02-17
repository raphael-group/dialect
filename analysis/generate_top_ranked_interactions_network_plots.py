"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.plotting import (
    draw_single_me_or_co_interaction_network,
)
from dialect.utils.postprocessing import (
    generate_top_ranked_co_interaction_tables,
    generate_top_ranked_me_interaction_tables,
)

ME_METHODS = ["DIALECT (Rho)", "DISCOVER", "Fisher's Exact Test", "MEGSA", "WeSME"]
CO_METHODS = ["DIALECT (LRT)", "DISCOVER", "Fisher's Exact Test", "WeSCO"]


def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        add_num_pairs=True,
        add_analysis_type=True,
        add_driver_genes_fn=True,
        add_likely_passenger_dir=True,
    )
    args = parser.parse_args()

    drvr_df = pd.read_csv(args.driver_genes_fn, sep="\t", index_col=0)
    putative_drivers = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    subtypes = [subtype.name for subtype in args.results_dir.iterdir()]
    num_edges = args.num_pairs // 2 if args.analysis_type == "BOTH" else args.num_pairs
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

        top_ranked_me_interactions_by_method, method_to_num_significant_me_pairs = (
            generate_top_ranked_me_interaction_tables(
                results_df=results_df,
                num_pairs=num_edges,
                num_samples=num_samples,
                methods=ME_METHODS,
            )
        )
        top_ranked_co_interactions_by_method, method_to_num_significant_co_pairs = (
            generate_top_ranked_co_interaction_tables(
                results_df=results_df,
                num_pairs=num_edges,
                num_samples=num_samples,
                methods=CO_METHODS,
            )
        )
        if args.analysis_type == "ME":
            for method in top_ranked_me_interactions_by_method:
                top_ranked_me_pairs = top_ranked_me_interactions_by_method[method]
                dout = Path(f"{args.out_dir}/{subtype}")
                dout.mkdir(parents=True, exist_ok=True)
                fout = f"{args.out_dir}/{subtype}/{method}_network"
                significant_pairs = top_ranked_me_pairs[
                    top_ranked_me_pairs["Significant"]
                ]
                significant_nodes = (
                    {}
                    if significant_pairs.empty
                    else set(significant_pairs["Gene A"])
                    | set(
                        significant_pairs["Gene B"],
                    )
                )
                draw_single_me_or_co_interaction_network(
                    edges=top_ranked_me_pairs[["Gene A", "Gene B"]].to_numpy(),
                    significant_nodes=significant_nodes,
                    putative_drivers=putative_drivers,
                    likely_passengers=likely_passengers,
                    method=method,
                    fout=fout,
                )
        elif args.analysis_type == "CO":
            for method in top_ranked_co_interactions_by_method:
                top_ranked_co_pairs = top_ranked_co_interactions_by_method[method]
                dout = Path(f"{args.out_dir}/{subtype}")
                dout.mkdir(parents=True, exist_ok=True)
                fout = f"{args.out_dir}/{subtype}/{method}_network"
                significant_pairs = top_ranked_co_pairs[
                    top_ranked_co_pairs["Significant"]
                ]
                significant_nodes = (
                    {}
                    if significant_pairs.empty
                    else set(significant_pairs["Gene A"])
                    | set(
                        significant_pairs["Gene B"],
                    )
                )
                draw_single_me_or_co_interaction_network(
                    edges=top_ranked_co_pairs[["Gene A", "Gene B"]].to_numpy(),
                    significant_nodes=significant_nodes,
                    putative_drivers=putative_drivers,
                    likely_passengers=likely_passengers,
                    method=method,
                    fout=fout,
                )

if __name__ == "__main__":
    main()

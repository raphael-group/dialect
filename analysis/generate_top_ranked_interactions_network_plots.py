"""TODO: Add docstring."""

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
        results_dir_required=True,
        out_dir_required=True,
        add_driver_genes_fn=True,
        add_likely_passenger_dir=True,
        add_analysis_type=True,
        add_num_pairs=True,
        driver_genes_required=True,
        likely_passenger_required=True,
    )
    args = parser.parse_args()

    drvr_df = pd.read_csv(args.driver_genes_fn, sep="\t", index_col=0)
    putative_drivers = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    subtypes = [subtype.name for subtype in args.results_dir.iterdir()]
    num_edges = args.num_pairs // 2 if args.analysis_type == "both" else args.num_pairs
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
                methods=ME_METHODS,
            )
        )
        top_ranked_co_interactions_by_method = (
            generate_top_ranked_co_interaction_tables(
                results_df=results_df,
                num_pairs=num_edges,
                num_samples=num_samples,
                methods=CO_METHODS,
            )
        )
        if args.analysis_type == "mutual_exclusivity":
            for method in top_ranked_me_interactions_by_method:
                top_ranked_me_pairs = top_ranked_me_interactions_by_method[method]
                fout = f"{args.out_dir}/{subtype}_{method}_network"
                draw_single_me_or_co_interaction_network(
                    edges=top_ranked_me_pairs[["Gene A", "Gene B"]].to_numpy(),
                    putative_drivers=putative_drivers,
                    likely_passengers=likely_passengers,
                    method=method,
                    fout=fout,
                )
        elif args.analysis_type == "cooccurrence":
            for method in top_ranked_co_interactions_by_method:
                top_ranked_co_pairs = top_ranked_co_interactions_by_method[method]
                fout = f"{args.out_dir}/{subtype}_{method}_network"
                draw_single_me_or_co_interaction_network(
                    edges=top_ranked_co_pairs[["Gene A", "Gene B"]].to_numpy(),
                    putative_drivers=putative_drivers,
                    likely_passengers=likely_passengers,
                    method=method,
                    fout=fout,
                )

if __name__ == "__main__":
    main()

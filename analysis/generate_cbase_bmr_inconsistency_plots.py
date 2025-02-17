"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.helpers import (
    load_likely_passenger_genes,
    load_putative_driver_genes,
)
from dialect.utils.plotting import (
    draw_cbase_likely_passenger_proportion_barplot,
    draw_cbase_top_likely_passenger_upset,
    draw_gene_expected_and_observed_mutations_barplot,
)


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def load_all_subtype_single_gene_results(results_dir: Path, num_genes: int) -> dict:
    """TODO: Add docstring."""
    subtype_to_results_df = {}
    for subtype_dir in results_dir.iterdir():
        subtype = subtype_dir.stem
        subtype_results_fn = subtype_dir / "single_gene_results.csv"
        subtype_results_df = pd.read_csv(
            subtype_results_fn,
        )
        sorted_subtype_results_df = subtype_results_df.sort_values(
            by="CBaSE Pos. Sel. Phi",
            ascending=False,
        )
        subtype_to_results_df[subtype] = sorted_subtype_results_df.head(num_genes)

    return subtype_to_results_df


def compute_subtype_likely_passenger_proportion(
    subtype_to_likely_passengers: dict,
    subtype_to_cbase_top_ranked_genes: dict,
) -> None:
    """TODO: Add docstring."""
    subtype_to_likely_passenger_gene_overlap = {}
    subtype_to_likely_passenger_proportion = {}
    for (
        subtype,
        cbase_top_ranked_results,
    ) in subtype_to_cbase_top_ranked_genes.items():
        cbase_top_ranked_genes = set(cbase_top_ranked_results["Gene Name"])
        likely_passengers = subtype_to_likely_passengers[subtype]
        likely_passenger_gene_overlap = cbase_top_ranked_genes.intersection(
            likely_passengers,
        )
        likely_passenger_proportion = len(likely_passenger_gene_overlap) / len(
            cbase_top_ranked_genes,
        )
        subtype_to_likely_passenger_gene_overlap[subtype] = (
            likely_passenger_gene_overlap
        )
        subtype_to_likely_passenger_proportion[subtype] = likely_passenger_proportion
    return (
        subtype_to_likely_passenger_gene_overlap,
        subtype_to_likely_passenger_proportion,
    )


def compute_subtype_putative_driver_gene_overlap(
    putative_drivers: set,
    subtype_to_cbase_top_ranked_genes: dict,
) -> None:
    """TODO: Add docstring."""
    subtype_to_driver_gene_overlap = {}
    for (
        subtype,
        cbase_top_ranked_results,
    ) in subtype_to_cbase_top_ranked_genes.items():
        cbase_top_ranked_genes = set(cbase_top_ranked_results["Gene Name"])
        driver_gene_overlap = cbase_top_ranked_genes.intersection(putative_drivers)
        subtype_to_driver_gene_overlap[subtype] = driver_gene_overlap
    return subtype_to_driver_gene_overlap


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        add_num_genes=True,
        add_likely_passenger_dir=True,
        add_putative_driver_gene_fn=True,
    )
    args = parser.parse_args()
    putative_drivers = load_putative_driver_genes(args.putative_driver_gene_fn)
    subtype_to_likely_passengers = load_likely_passenger_genes(
        args.likely_passenger_dir,
    )
    subtype_to_cbase_top_ranked_results = load_all_subtype_single_gene_results(
        args.results_dir,
        args.num_genes,
    )
    subtype_to_likely_passenger_gene_overlap, subtype_to_likely_passenger_proportion = (
        compute_subtype_likely_passenger_proportion(
            subtype_to_likely_passengers,
            subtype_to_cbase_top_ranked_results,
        )
    )
    subtype_to_putative_driver_gene_overlap = (
        compute_subtype_putative_driver_gene_overlap(
            putative_drivers,
            subtype_to_cbase_top_ranked_results,
        )
    )

    draw_cbase_top_likely_passenger_upset(
        subtype_to_likely_passenger_gene_overlap,
        out_fn=args.out_dir / "cbase_likely_passenger_upset",
    )

    draw_cbase_likely_passenger_proportion_barplot(
        subtype_to_likely_passenger_proportion=subtype_to_likely_passenger_proportion,
        out_fn=args.out_dir / "cbase_likely_passenger_proportion",
        num_genes=args.num_genes,
    )

    for (
        subtype,
        cbase_top_ranked_results,
    ) in subtype_to_cbase_top_ranked_results.items():
        likely_passenger_genes = subtype_to_likely_passengers[subtype]
        putative_driver_genes = subtype_to_putative_driver_gene_overlap[subtype]
        draw_gene_expected_and_observed_mutations_barplot(
            cbase_top_ranked_results,
            likely_passenger_genes,
            putative_driver_genes,
            out_fn=args.out_dir / "cbase_obs_exp_plots" / f"{subtype}",
        )


if __name__ == "__main__":
    main()

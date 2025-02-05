"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.plotting import draw_sample_mutation_count_subtype_histograms


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def compute_avg_sample_mutation_count(results_dir: Path) -> pd.Series:
    """TODO: Add docstring."""
    subtype_sample_mutation_count_sums = []
    for subtype in results_dir.iterdir():
        subtype_cnt_mtx_fn = subtype / "count_matrix.csv"
        if not subtype_cnt_mtx_fn.exists():
            continue
        cnt_mtx_df = pd.read_csv(subtype_cnt_mtx_fn, index_col=0)
        subtype_sample_mutation_count_sums.append(cnt_mtx_df.sum(axis=1))
    return pd.concat(subtype_sample_mutation_count_sums).mean()


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        results_dir_required=True,
        out_dir_required=True,
        add_subtypes=True,
    )
    args = parser.parse_args()

    subtype_avg_sample_mutation_count = compute_avg_sample_mutation_count(
        args.results_dir,
    )

    subtypes = [s.strip() for s in args.subtypes.split(",") if s.strip()]
    xlim_mapping = {}
    subtype_to_sample_mutation_counts = {}
    for subtype in subtypes:
        subtype_cnt_mtx_fn = Path(args.results_dir) / subtype / "count_matrix.csv"
        sample_mutation_counts = pd.read_csv(
            subtype_cnt_mtx_fn,
            index_col=0,
        ).sum(axis=1)
        subtype_to_sample_mutation_counts[subtype] = sample_mutation_counts
        xlim_mapping[subtype] = sample_mutation_counts.max() * 1.01

    out_fn = Path(args.out_dir) / "sample_mutation_count_subtype_histograms"
    draw_sample_mutation_count_subtype_histograms(
        subtype_to_sample_mutation_counts,
        subtype_avg_sample_mutation_count,
        xlim_mapping,
        out_fn,
    )


if __name__ == "__main__":
    main()

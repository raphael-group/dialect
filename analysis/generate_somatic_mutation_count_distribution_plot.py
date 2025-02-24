"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.plotting import draw_all_subtypes_mutation_distribution


def get_all_subtype_somatic_mutation_counts(results_dir: Path) -> list:
    """TODO: Add docstring."""
    somatic_mutation_counts = []
    for subtype_dir in results_dir.iterdir():
        subtype = subtype_dir.name
        if subtype != "BRCA":
            continue
        cnt_mtx_fn = subtype_dir / "count_matrix.csv"
        if not cnt_mtx_fn.exists():
            continue
        cnt_df = pd.read_csv(cnt_mtx_fn, index_col=0)
        somatic_mutation_counts.extend(cnt_df.sum(axis=0).tolist())
    return somatic_mutation_counts


def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser()
    args = parser.parse_args()
    somatic_mutation_counts = get_all_subtype_somatic_mutation_counts(args.results_dir)
    xlimit = 25
    draw_all_subtypes_mutation_distribution(
        [x for x in somatic_mutation_counts if x < xlimit],
        out_fn=args.out_dir / "all_subtypes_somatic_mutation_count_distribution",
        xlabel="Somatic Mutation Count per Gene-Sample Event",
        ylabel="Frequency",
        xlim=(0, xlimit),
        figsize=(5, 4),
    )


if __name__ == "__main__":
    main()

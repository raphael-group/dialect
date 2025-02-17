"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.plotting import draw_gene_mutation_variability_hexbin_plots


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def compute_gene_mean_and_std_dev_counts(results_dir: Path, subtypes: set) -> tuple:
    """TODO: Add docstring."""
    subtype_to_gene_mean_mutation_counts = {}
    subtype_to_gene_std_dev_mutation_counts = {}
    for subtype_dir in results_dir.iterdir():
        subtype_cnt_mtx_fn = subtype_dir / "count_matrix.csv"
        subtype = subtype_dir.name
        if subtype not in subtypes or not subtype_cnt_mtx_fn.exists():
            continue
        cnt_mtx_df = pd.read_csv(subtype_cnt_mtx_fn, index_col=0)
        gene_mean_mutation_counts = cnt_mtx_df.mean(axis=0)
        gene_std_dev_mutation_counts = cnt_mtx_df.std(axis=0)
        lower_mean = gene_mean_mutation_counts.quantile(0.01)
        upper_mean = gene_mean_mutation_counts.quantile(0.99)
        lower_std = gene_std_dev_mutation_counts.quantile(0.01)
        upper_std = gene_std_dev_mutation_counts.quantile(0.99)
        valid_genes = gene_mean_mutation_counts.index[
            (gene_mean_mutation_counts >= lower_mean)
            & (gene_mean_mutation_counts <= upper_mean)
            & (gene_std_dev_mutation_counts >= lower_std)
            & (gene_std_dev_mutation_counts <= upper_std)
        ]
        gene_mean_mutation_counts = gene_mean_mutation_counts.loc[valid_genes]
        gene_std_dev_mutation_counts = gene_std_dev_mutation_counts.loc[valid_genes]
        subtype_to_gene_mean_mutation_counts[subtype] = gene_mean_mutation_counts
        subtype_to_gene_std_dev_mutation_counts[subtype] = gene_std_dev_mutation_counts
    return subtype_to_gene_mean_mutation_counts, subtype_to_gene_std_dev_mutation_counts


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(add_subtypes=True)
    args = parser.parse_args()
    subtypes = [s.strip() for s in args.subtypes.split(",") if s.strip()]
    subtype_to_gene_mean_mutation_counts, subtype_to_gene_std_dev_mutation_counts = (
        compute_gene_mean_and_std_dev_counts(
            args.results_dir,
            set(subtypes),
        )
    )
    out_fn = Path(args.out_dir) / "gene_mutation_variability_subtype_hexbin_plots"
    draw_gene_mutation_variability_hexbin_plots(
        subtype_to_gene_mean_mutation_counts,
        subtype_to_gene_std_dev_mutation_counts,
        out_fn,
        subtypes,
    )


if __name__ == "__main__":
    main()

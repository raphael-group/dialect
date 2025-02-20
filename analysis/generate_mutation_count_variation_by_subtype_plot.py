"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.plotting import draw_mutation_variability_scatterplot


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def get_coefficients_of_variation(results_dir: Path) -> tuple:
    """TODO: Add docstring."""
    subtype_to_coef_of_variation = {}
    for subtype_dir in results_dir.iterdir():
        subtype = subtype_dir.name
        cnt_mtx_fn = subtype_dir / "count_matrix.csv"
        if not cnt_mtx_fn.exists():
            continue
        cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
        cnt_mtx_arr = cnt_mtx_df.to_numpy().flatten()
        coef_of_variation = cnt_mtx_arr.std() / cnt_mtx_arr.mean()
        subtype_to_coef_of_variation[subtype] = coef_of_variation
    return subtype_to_coef_of_variation

# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser()
    args = parser.parse_args()
    subtype_to_coef_of_variation = get_coefficients_of_variation(args.results_dir)
    out_fn = args.out_dir / "subtype_mutation_variability_scatterplot"
    draw_mutation_variability_scatterplot(
        subtype_to_coef_of_variation,
        out_fn,
    )


if __name__ == "__main__":
    main()

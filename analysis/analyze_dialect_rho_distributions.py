"""TODO: Add docstring."""

from pathlib import Path

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def get_me_threshold(rho_values: pd.Series, fdr:float=0.05) -> float:
    """TODO: Add docstring."""
    return rho_values.quantile(fdr)

def get_co_threshold(lrt_values: pd.Series, fdr:float=0.05) -> float:
    """TODO: Add docstring."""
    return lrt_values.quantile(1 - fdr)

# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser()
    args = parser.parse_args()
    sim_dir = Path("output/simulations")
    sim_type_name = "NS{}/0ME_0CO_200P/1.0DP/0.05TL_0.10TH"

    subtype_to_significance_thresholds = {}
    for subtype_dir in args.results_dir.iterdir():
        subtype = subtype_dir.name
        cnt_mtx_fn = subtype_dir / "count_matrix.csv"
        if not cnt_mtx_fn.exists():
            continue
        num_samples = pd.read_csv(cnt_mtx_fn).shape[0]
        sim_subtype_dir = sim_dir / subtype / sim_type_name.format(num_samples)
        if not sim_subtype_dir.is_dir():
            continue
        rho_values = []
        lrt_values = []
        for i in range(1, 101):
            results_dir = sim_subtype_dir / f"R{i}"
            results_fn = results_dir / "pairwise_interaction_results.csv"
            if not results_fn.exists():
                continue
            results_df = pd.read_csv(results_fn)
            me_results_df = results_df[results_df["Rho"] < 0]
            co_results_df = results_df[results_df["Rho"] > 0]
            rho_values.extend(me_results_df["Rho"].tolist())
            lrt_values.extend(co_results_df["Likelihood Ratio"].tolist())
        me_threshold = get_me_threshold(pd.Series(rho_values))
        co_threshold = get_co_threshold(pd.Series(lrt_values))
        subtype_to_significance_thresholds[subtype] = {
            "ME_Rho_Threshold": me_threshold,
            "CO_LRT_Threshold": co_threshold,
        }
    thresholds_df = pd.DataFrame.from_dict(subtype_to_significance_thresholds,
                                           orient="index")
    thresholds_df = thresholds_df.reset_index().rename(columns={"index": "subtype"})
    out_fn = args.out_dir / "dialect_significance_thresholds.csv"
    thresholds_df.to_csv(out_fn, index=False)


if __name__ == "__main__":
    main()

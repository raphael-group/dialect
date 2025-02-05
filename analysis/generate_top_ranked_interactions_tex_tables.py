"""TODO: Add docstring."""

import os

import numpy as np
import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.postprocessing import (
    generate_top_ranked_co_interaction_tables,
    generate_top_ranked_me_interaction_tables,
)

FLOAT_LOW_THRESHOLD = 0.01
MANTISSA_MAX_THRESHOLD = 10
ME_METHODS = ["DIALECT", "DISCOVER", "Fisher's Exact Test", "MEGSA", "WeSME"]
CO_METHODS = ["DIALECT", "DISCOVER", "Fisher's Exact Test", "WeSCO"]


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def format_float(value: float) -> str:
    """TODO: Add docstring."""
    sign = "-" if value < 0 else ""
    abs_val = abs(value)

    if abs_val == 0:
        return "0"
    if abs_val >= FLOAT_LOW_THRESHOLD:
        val_str = f"{abs_val:.2f}".rstrip("0").rstrip(".")
        return sign + val_str
    exponent = int(np.floor(np.log10(abs_val)))
    mantissa = abs_val / (10**exponent)
    while mantissa >= MANTISSA_MAX_THRESHOLD:
        mantissa /= 10
        exponent += 1
    while mantissa < 1:
        mantissa *= 10
        exponent -= 1

    mantissa_str = f"{mantissa:.2g}"
    return f"{sign}{mantissa_str}x10^{{{exponent}}}"


def escape_gene_name(gene_name: str) -> str:
    r"""Convert '_M' -> '\_M' and '_N' -> '\_N' so LaTeX can render underscores."""
    return gene_name.replace("_M", r"\_M").replace("_N", r"\_N")


def _build_subtable_latex_(
    methods_list: list,
    top_pairs_by_method: dict,
    metric_labels: dict,
    ixn_type: str,
) -> str:
    """TODO: Add docstring."""
    num_methods = len(methods_list)
    col_spec = "||".join(["cc"] * num_methods)

    lines = []
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")

    header_parts = [r"\multicolumn{2}{c}{" + method + "}" for method in methods_list]
    lines.append(" & ".join(header_parts) + r" \\ \hline")

    subheader_parts = []
    for method in methods_list:
        col_name = f"{ixn_type} Gene Pair"
        subheader_parts.append(col_name)
        subheader_parts.append(metric_labels[method])
    lines.append(" & ".join(subheader_parts) + r" \\ \hline")

    max_rows = max(len(top_pairs_by_method[m]) for m in methods_list)
    for i in range(max_rows):
        row_entries = []
        for method in methods_list:
            pairs_list = top_pairs_by_method[method]
            if i < len(pairs_list):
                pair_str, val_str = pairs_list[i]
                row_entries.append(pair_str)
                row_entries.append(val_str)
            else:
                row_entries.append("")
                row_entries.append("")
        lines.append(" & ".join(row_entries) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def create_final_table(
    subtype: str,
    top_pairs_by_method: dict,
    num_pairs: int,
    ixn_type: str,
) -> str:
    """Create one big table environment with two "sub-tables" across methods.

    - First row (3 methods): DIALECT, DISCOVER, Fisher's Exact Test
    - Second row (2 methods): MEGSA, WeSME
    """
    metric_labels = {
        "DIALECT": r"$\rho$",
        "DISCOVER": "p-value",
        "Fisher's Exact Test": "p-value",
        "MEGSA": "S-Score",
        "WeSME": "p-value",
        "WeSCO": "p-value",
    }

    top_row_methods = ["DIALECT", "DISCOVER", "Fisher's Exact Test"]
    bottom_row_methods = ["MEGSA"]
    if ixn_type == "ME":
        bottom_row_methods.append("WeSME")
    else:
        bottom_row_methods.append("WeSCO")

    top_subtable = _build_subtable_latex_(
        top_row_methods,
        top_pairs_by_method,
        metric_labels,
        ixn_type,
    )
    bottom_subtable = _build_subtable_latex_(
        bottom_row_methods,
        top_pairs_by_method,
        metric_labels,
        ixn_type,
    )

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")

    lines.append(top_subtable)
    lines.append(r"\vspace{-0.2cm}")
    lines.append(bottom_subtable)

    caption_str = rf"Top {num_pairs} ranked ME pairs across methods in {subtype}"
    lines.append(rf"\caption{{{caption_str}}}")
    lines.append(r"\label{tab:" + subtype + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        results_dir_required=True,
        out_dir_required=True,
        putative_driver_required=True,
        likely_passenger_required=True,
        add_num_pairs=True,
        add_analysis_type=True,
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subtypes = os.listdir(args.results_dir)
    for subtype in subtypes:
        results_fn = args.results_dir / subtype / "complete_pairwise_ixn_results.csv"
        cnt_mtx_fn = args.results_dir / subtype / "count_matrix.csv"
        if not results_fn.exists() or not cnt_mtx_fn.exists():
            continue
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        results_df = pd.read_csv(results_fn)
        if args.analysis_type == "mutual_exclusivity":
            top_tables = generate_top_ranked_me_interaction_tables(
                results_df=results_df,
                num_pairs=args.num_pairs,
                num_samples=num_samples,
                methods=ME_METHODS,
            )
        else:
            top_tables = generate_top_ranked_co_interaction_tables(
                results_df=results_df,
                num_pairs=args.num_pairs,
                num_samples=num_samples,
                methods=CO_METHODS,
            )
        top_pairs_by_method = {}
        for method_name, method_df in top_tables.items():
            if method_df is None or method_df.empty:
                top_pairs_by_method[method_name] = []
                continue
            pairs_list = []
            metric_col_name = method_df.columns[-1]
            for _, row in method_df.iterrows():
                gene_a = escape_gene_name(str(row["Gene A"]))
                gene_b = escape_gene_name(str(row["Gene B"]))
                metric_val = row[metric_col_name]
                val_str = format_float(metric_val) if metric_val is not None else "N/A"
                interaction_str = f"{gene_a}:{gene_b}"
                pairs_list.append((interaction_str, val_str))
            if method_name == "WeSME" and args.analysis_type == "cooccurrence":
                adjusted_method_name = "WeSCO"
                top_pairs_by_method[adjusted_method_name] = pairs_list
            else:
                top_pairs_by_method[method_name] = pairs_list

        latex_str = create_final_table(
            subtype,
            top_pairs_by_method,
            args.num_pairs,
            "ME" if args.analysis_type == "mutual_exclusivity" else "CO",
        )
        if args.analysis_type == "mutual_exclusivity":
            fout = args.out_dir / f"table_of_top_me_pairs_{subtype}.tex"
        else:
            fout = args.out_dir / f"table_of_top_co_pairs_{subtype}.tex"
        with fout.open("w") as f:
            f.write(latex_str)


if __name__ == "__main__":
    main()

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
ME_METHODS = ["DIALECT (Rho)", "DISCOVER", "Fisher's Exact Test", "MEGSA", "WeSME"]
CO_METHODS = ["DIALECT (LRT)", "DISCOVER", "Fisher's Exact Test", "WeSCO"]

METHOD_TO_METRIC = {
    "DIALECT (Rho)": r"$\rho$",
    "DIALECT (LRT)": "LRT",
    "DISCOVER": "q-value",
    "Fisher's Exact Test": "q-value",
    "MEGSA": "p-value",
    "WeSME": "p-value",
    "WeSCO": "p-value",
}

METHOD_TO_ME_METRIC = {
    "DIALECT (Rho)": "Rho",
    "DISCOVER": "Discover ME Q-Val",
    "Fisher's Exact Test": "Fisher's ME Q-Val",
    "MEGSA": "MEGSA P-Val",
    "WeSME": "WeSME P-Val",
}

METHOD_TO_CO_METRICS = {
    "DIALECT (LRT)": "Likelihood Ratio",
    "DISCOVER": "Discover CO Q-Val",
    "Fisher's Exact Test": "Fisher's CO Q-Val",
    "WeSCO": "WeSCO P-Val",
}

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
    lines.append(r"\renewcommand{\arraystretch}{1}")
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


def _write_caption(ixn_type: str, method_to_num_sig_pairs: dict) -> str:
    if ixn_type == "ME":
        ixn_label = "mutually exclusive"
        indicator_text = (
            "More negative $\\rho$ values, lower p-values, and higher likelihood ratio "
            "test (LRT) scores indicate stronger mutual exclusivity."
        )
    elif ixn_type == "CO":
        ixn_label = "co-occurring"
        indicator_text = (
            "More positive $\\rho$ values, lower p-values, and higher likelihood ratio "
            "test (LRT) scores indicate stronger co-occurrence."
        )
    else:
        ixn_label = "interactions"
        indicator_text = ""

    sig_pairs_list = [
        f"{method}: {num}" for method, num in method_to_num_sig_pairs.items()
    ]
    sig_pairs_str = ", ".join(sig_pairs_list)

    return (
        f"\\textbf{{Top ranked {ixn_label} interactions in UCEC across methods.}} "
        f"$10$ {ixn_label} gene pairs most highly ranked by \\OurMethod{{}}, DISCOVER, "
        f"Fisher's Exact Test, MEGSA, and WeSME on TCGA endometrial cancer. "
        f"{indicator_text} "
        "Significance thresholds applied were: Fisher's Exact Test"
        " (FDR threshold of 0.01), "
        "DISCOVER (Benjamini-Hochberg correction with a"
        " maximum FDR threshold of 0.01), "
        "MEGSA (p-value threshold of 1 \\times 10$^{-3}$), and WeSME"
        "(FDR-corrected at 0.01), "
        "while DIALECT did not threshold by significance. "
        f"The total number of significant pairs identified were: {sig_pairs_str}."
    )


# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def create_final_table(
    subtype: str,
    top_pairs_by_method: dict,
    ixn_type: str,
    method_to_num_sig_pairs: dict,
) -> str:
    """Create one big table environment with two "sub-tables" across methods.

    - First row (3 methods): DIALECT, DISCOVER, Fisher's Exact Test
    - Second row (2 methods): MEGSA, WeSME
    """
    if ixn_type == "ME":
        top_row_methods = ["DIALECT (Rho)", "DISCOVER", "Fisher's Exact Test"]
        bottom_row_methods = ["MEGSA", "WeSME"]
    else:
        top_row_methods = ["DIALECT (LRT)", "DISCOVER", "Fisher's Exact Test"]
        bottom_row_methods = ["Fisher's Exact Test", "WeSCO"]

    top_subtable = _build_subtable_latex_(
        top_row_methods,
        top_pairs_by_method,
        METHOD_TO_METRIC,
        ixn_type,
    )
    bottom_subtable = _build_subtable_latex_(
        bottom_row_methods,
        top_pairs_by_method,
        METHOD_TO_METRIC,
        ixn_type,
    )

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")

    lines.append(top_subtable)
    lines.append(r"\vspace{-0.2cm}")
    lines.append(bottom_subtable)

    caption = _write_caption(ixn_type, method_to_num_sig_pairs)
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:" + subtype + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        add_num_pairs=True,
        add_analysis_type=True,
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subtypes = os.listdir(args.results_dir)
    for subtype in subtypes:
        subtype_dir = args.results_dir / subtype
        results_fn = subtype_dir / "complete_pairwise_ixn_results.csv"
        cnt_mtx_fn = subtype_dir / "count_matrix.csv"
        if not results_fn.exists() or not cnt_mtx_fn.exists():
            continue
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        results_df = pd.read_csv(results_fn)
        if args.analysis_type == "ME":
            top_tables, method_to_num_sig_pairs = (
                generate_top_ranked_me_interaction_tables(
                    results_df=results_df,
                    num_pairs=args.num_pairs,
                    num_samples=num_samples,
                    methods=ME_METHODS,
                )
            )
        else:
            top_tables, method_to_num_sig_pairs = (
                generate_top_ranked_co_interaction_tables(
                    results_df=results_df,
                    num_pairs=args.num_pairs,
                    num_samples=num_samples,
                    methods=CO_METHODS,
                )
            )
        top_pairs_by_method = {}
        for method_name, method_df in top_tables.items():
            if method_df is None or method_df.empty:
                top_pairs_by_method[method_name] = []
                continue
            pairs_list = []
            metric_col_name = (
                METHOD_TO_ME_METRIC[method_name]
                if args.analysis_type == "ME"
                else METHOD_TO_CO_METRICS[method_name]
            )
            for _, row in method_df.iterrows():
                gene_a = escape_gene_name(str(row["Gene A"]))
                gene_b = escape_gene_name(str(row["Gene B"]))
                metric_val = row[metric_col_name]
                val_str = format_float(metric_val) if metric_val is not None else "N/A"
                interaction_str = f"{gene_a}:{gene_b}"
                pairs_list.append((interaction_str, val_str))
            if method_name == "WeSME" and args.analysis_type == "CO":
                adjusted_method_name = "WeSCO"
                top_pairs_by_method[adjusted_method_name] = pairs_list
            else:
                top_pairs_by_method[method_name] = pairs_list

        latex_str = create_final_table(
            subtype,
            top_pairs_by_method,
            args.analysis_type,
            method_to_num_sig_pairs,
        )
        out_dir = args.out_dir / subtype
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fn = out_dir / f"table_of_{args.analysis_type}_top_pairs.tex"
        with out_fn.open("w") as f:
            f.write(latex_str)


if __name__ == "__main__":
    main()

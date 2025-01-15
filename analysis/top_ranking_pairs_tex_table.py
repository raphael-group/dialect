import os
import logging
import pandas as pd
import numpy as np
from argparse import ArgumentParser

EPSILON_THRESHOLD = 0.05
PVALUE_THRESHOLD = 1


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def build_argument_parser():
    """
    We do NOT modify this parser, as requested.
    """
    parser = ArgumentParser(description="Generate LaTeX tables for top ME pairs across methods.")
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=10,
        help="Number of top ranking pairs to visualize",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="Directory with results for all subtypes",
    )
    parser.add_argument(
        "-dvr",
        "--driver_genes_fn",
        type=str,
        required=True,
        help="File with driver genes",
    )
    parser.add_argument(
        "-d",
        "--decoy_genes_dir",
        type=str,
        required=True,
        help="Directory with all decoy gene files",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="Output directory for tex files",
    )
    return parser


def format_float(value):
    """
    Handles positive and negative floats:
      - If |value| >= 0.01, display with up to 2 decimals (no trailing zeros).
      - If |value| < 0.01 (and not zero), use base-10 notation: e.g. 3x10^{-3}.
      - Preserve a '-' sign if the original value is negative.
      - Ensures the mantissa stays in the range [1, 10) and uses two significant digits
        to avoid outputs like '1e+01x10^{-4}'.
    """
    sign = "-" if value < 0 else ""
    abs_val = abs(value)

    if abs_val == 0:
        return "0"
    elif abs_val >= 0.01:
        val_str = f"{abs_val:.2f}".rstrip("0").rstrip(".")
        return sign + val_str
    else:
        exponent = int(np.floor(np.log10(abs_val)))
        mantissa = abs_val / (10**exponent)
        # Adjust if mantissa >= 10 or < 1
        while mantissa >= 10:
            mantissa /= 10
            exponent += 1
        while mantissa < 1:
            mantissa *= 10
            exponent -= 1

        # Keep two significant digits for the mantissa
        mantissa_str = f"{mantissa:.2g}"
        return f"{sign}{mantissa_str}x10^{{{exponent}}}"


def escape_gene_name(gene_name):
    """
    Convert '_M' -> '\_M' and '_N' -> '\_N' so LaTeX can render underscores.
    """
    return gene_name.replace("_M", r"\_M").replace("_N", r"\_N")


def build_subtable_latex(methods_list, top_pairs_by_method, metric_labels):
    """
    Builds a LaTeX fragment for the given set of methods in a single row of columns.
    E.g., ["DIALECT","DISCOVER","Fisher's Exact Test"] -> a 3-method tabular.
    """
    num_methods = len(methods_list)
    col_spec = "||".join(["cc"] * num_methods)

    lines = []
    lines.append(r"\renewcommand{\arraystretch}{1.2}")  # Slightly bigger spacing
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")

    header_parts = []
    for method in methods_list:
        header_parts.append(r"\multicolumn{2}{c}{" + method + "}")
    lines.append(" & ".join(header_parts) + r" \\ \hline")

    subheader_parts = []
    for method in methods_list:
        subheader_parts.append("ME Gene Pair")
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


# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
def create_final_table(subtype, top_pairs_by_method, top_k):
    """
    Creates one big table environment with two "sub-tables":
     - First row (3 methods): DIALECT, DISCOVER, Fisher's Exact Test
     - Second row (2 methods): MEGSA, WeSME
    We place them in consecutive tabulars with minimal spacing.
    """
    metric_labels = {
        "DIALECT": r"$\rho$",
        "DISCOVER": "p-value",
        "Fisher's Exact Test": "p-value",
        "MEGSA": "S-Score",
        "WeSME": "p-value",
    }

    top_row_methods = ["DIALECT", "DISCOVER", "Fisher's Exact Test"]
    bottom_row_methods = ["MEGSA", "WeSME"]

    top_subtable = build_subtable_latex(top_row_methods, top_pairs_by_method, metric_labels)
    bottom_subtable = build_subtable_latex(bottom_row_methods, top_pairs_by_method, metric_labels)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")

    # Top sub-table (DIALECT, DISCOVER, FET)
    lines.append(top_subtable)
    # Tiny vertical gap
    lines.append(r"\vspace{-0.2cm}")
    # Bottom sub-table (MEGSA, WeSME)
    lines.append(bottom_subtable)

    lines.append(rf"\caption{{Top {top_k} ranked ME pairs across methods in {subtype}}}")
    lines.append(r"\label{tab:" + subtype + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    methods_info = {
        "DIALECT": {
            "column": "Rho",
            "ascending": True,
        },
        "DISCOVER": {
            "column": "Discover ME P-Val",
            "ascending": True,
        },
        "Fisher's Exact Test": {
            "column": "Fisher's ME P-Val",
            "ascending": True,
        },
        "MEGSA": {
            "column": "MEGSA S-Score (LRT)",
            "ascending": False,
        },
        "WeSME": {
            "column": "WeSME P-Val",
            "ascending": True,
        },
    }

    subtypes = os.listdir(args.results_dir)
    for subtype in subtypes:
        results_fn = os.path.join(args.results_dir, subtype, "complete_pairwise_ixn_results.csv")
        if not os.path.exists(results_fn):
            logging.info(f"Skipping {subtype}: file not found.")
            continue
        results_df = pd.read_csv(results_fn)

        top_pairs_by_method = {}
        for method_name, info in methods_info.items():
            col_name = info["column"]
            ascending = info["ascending"]

            if col_name not in results_df.columns:
                logging.warning(
                    f"Column {col_name} not found in {results_fn} for {subtype}. Skipping {method_name}."
                )
                top_pairs_by_method[method_name] = []
                continue

            # 1) Subset as per the method's rules
            method_df = results_df.copy()

            if method_name == "DIALECT":
                method_df = method_df[method_df["Rho"] < 0]
                method_df = method_df[
                    (method_df["Tau_1X"] > EPSILON_THRESHOLD)
                    & (method_df["Tau_X1"] > EPSILON_THRESHOLD)
                ]

            if method_name == "MEGSA":
                method_df = method_df[method_df["MEGSA S-Score (LRT)"] > 0]

            if method_name == "DISCOVER":
                method_df = method_df[method_df["Discover ME P-Val"] < PVALUE_THRESHOLD]

            if method_name == "Fisher's Exact Test":
                method_df = method_df[method_df["Fisher's ME P-Val"] < PVALUE_THRESHOLD]

            if method_name == "WeSME":
                method_df = method_df[method_df["WeSME P-Val"] < PVALUE_THRESHOLD]

            # 2) Sort by the column, then pick top K
            if not method_df.empty:
                method_df = method_df.sort_values(col_name, ascending=ascending).head(args.top_k)
            else:
                top_pairs_by_method[method_name] = []
                continue

            # 3) Construct the final list of (ME Gene Pair, value_str)
            pairs_list = []
            for _, row in method_df.iterrows():
                geneA = escape_gene_name(str(row["Gene A"]))
                geneB = escape_gene_name(str(row["Gene B"]))

                interaction_str = f"{geneA}:{geneB}"
                val = row[col_name]
                val_str = format_float(val)
                pairs_list.append((interaction_str, val_str))

            top_pairs_by_method[method_name] = pairs_list

        # Build the final LaTeX table
        latex_str = create_final_table(subtype, top_pairs_by_method, args.top_k)

        # Save to file
        out_filename = os.path.join(args.out, f"table_of_top_pairs_{subtype}.tex")
        with open(out_filename, "w") as f:
            f.write(latex_str)

        logging.info(f"Wrote LaTeX table for {subtype} to {out_filename}")

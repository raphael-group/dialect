import os
import logging
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from dialect.utils.postprocessing import generate_top_ranking_tables


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def build_argument_parser():
    """
    We do NOT modify this parser, as requested.
    """
    parser = ArgumentParser(description="Generate LaTeX tables for top pairs across methods.")
    parser.add_argument(
        "-n",
        "--num_pairs",
        type=int,
        default=10,
        help="Number of top ranking pairs to include in table",
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
        default="data/references/OncoKB_Cancer_Gene_List.tsv",
        help="File with driver genes",
    )
    parser.add_argument(
        "-d",
        "--decoy_genes_dir",
        type=str,
        default="data/decoy_genes",
        help="Directory with all decoy gene files",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="tables",
        help="Output directory for tex files",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--me",
        action="store_true",
        help="Perform analysis for mutual exclusivity",
    )
    group.add_argument(
        "--co",
        action="store_true",
        help="Perform analysis for co-occurrence",
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


def build_subtable_latex(methods_list, top_pairs_by_method, metric_labels, is_me):
    """
    Builds a LaTeX fragment for the given set of methods in a single row of columns.
    E.g., ["DIALECT","DISCOVER","Fisher's Exact Test"] -> a 3-method tabular.
    """
    num_methods = len(methods_list)
    col_spec = "||".join(["cc"] * num_methods)

    lines = []
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")

    header_parts = []
    for method in methods_list:
        header_parts.append(r"\multicolumn{2}{c}{" + method + "}")
    lines.append(" & ".join(header_parts) + r" \\ \hline")

    subheader_parts = []
    for method in methods_list:
        col_name = "ME Gene Pair" if is_me else "CO Gene Pair"
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


# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
def create_final_table(subtype, top_pairs_by_method, num_pairs, is_me):
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
        "WeSCO": "p-value",
    }

    top_row_methods = ["DIALECT", "DISCOVER", "Fisher's Exact Test"]
    bottom_row_methods = ["MEGSA"]
    if is_me:
        bottom_row_methods.append("WeSME")
    else:
        bottom_row_methods.append("WeSCO")

    top_subtable = build_subtable_latex(top_row_methods, top_pairs_by_method, metric_labels, is_me)
    bottom_subtable = build_subtable_latex(
        bottom_row_methods, top_pairs_by_method, metric_labels, is_me
    )

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")

    lines.append(top_subtable)
    lines.append(r"\vspace{-0.2cm}")
    lines.append(bottom_subtable)

    lines.append(rf"\caption{{Top {num_pairs} ranked ME pairs across methods in {subtype}}}")
    lines.append(r"\label{tab:" + subtype + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    subtypes = os.listdir(args.results_dir)
    for subtype in subtypes:
        results_fn = os.path.join(args.results_dir, subtype, "complete_pairwise_ixn_results.csv")
        cnt_mtx_fn = os.path.join(args.results_dir, subtype, "count_matrix.csv")
        if not os.path.exists(results_fn) or not os.path.exists(cnt_mtx_fn):
            logging.info(f"Skipping {subtype}: file not found.")
            continue
        num_samples = pd.read_csv(cnt_mtx_fn, index_col=0).shape[0]
        results_df = pd.read_csv(results_fn)
        top_tables = generate_top_ranking_tables(
            results_df=results_df,
            is_me=args.me,
            num_pairs=args.num_pairs,
            num_samples=num_samples,
        )
        top_pairs_by_method = {}
        for method_name, method_df in top_tables.items():
            if method_df is None or method_df.empty:
                logging.info(f"No top pairs for method: {method_name}")
                top_pairs_by_method[method_name] = []
                continue
            pairs_list = []
            metric_col_name = method_df.columns[-1]
            for _, row in method_df.iterrows():
                geneA = escape_gene_name(str(row["Gene A"]))
                geneB = escape_gene_name(str(row["Gene B"]))
                metric_val = row[metric_col_name]
                val_str = format_float(metric_val) if metric_val is not None else "N/A"
                interaction_str = f"{geneA}:{geneB}"
                pairs_list.append((interaction_str, val_str))
            if method_name == "WeSME" and args.co:
                method_name = "WeSCO"
            top_pairs_by_method[method_name] = pairs_list

        latex_str = create_final_table(subtype, top_pairs_by_method, args.num_pairs, args.me)
        if args.me:
            fout = os.path.join(args.out, f"table_of_top_me_pairs_{subtype}.tex")
        else:
            fout = os.path.join(args.out, f"table_of_top_co_pairs_{subtype}.tex")
        with open(fout, "w") as f:
            f.write(latex_str)

        logging.info(f"Wrote LaTeX table for {subtype} to {fout}")

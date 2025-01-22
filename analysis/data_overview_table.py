"""Generate a LaTeX table summarizing TCGA subtypes, samples, and mutations."""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd


def build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the script.

    Returns:
        argparse.ArgumentParser: The argument parser for the script.

    """
    parser = argparse.ArgumentParser(description="Generate TCGA subtype LaTeX table.")
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help=(
            "Directory with subfolders for each subtype, "
            "each containing count_matrix.csv"
        ),
    )
    parser.add_argument(
        "--study_abbrev_fn",
        type=str,
        default="data/references/TCGA_Study_Abbreviations.csv",
        help="CSV with columns: 'Study Abbreviation', 'Study Name'",
    )
    parser.add_argument(
        "--pancancer_counts_fn",
        type=str,
        default="data/references/TCGA_Pancancer_Sample_Counts.csv",
        help=(
            "CSV with columns: 'Subtype' (matching study abbreviation), "
            "'Number of Samples'"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="tables",
        help="Output LaTeX table file",
    )
    return parser


def create_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Create a LaTeX table from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data for the table.
        caption (str): Caption for the LaTeX table.
        label (str): Label for the LaTeX table.

    Returns:
        str: LaTeX formatted table as a string.

    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{6cm}c||cc||cc}")
    lines.append(r"\hline")
    lines.append(
        r"\multicolumn{2}{|c||}{\textbf{Cancer Type}} & "
        r"\multicolumn{2}{c||}{\textbf{Number of Samples}} & "
        r"\multicolumn{2}{c|}{\textbf{Number of Mutations by Type}} \\",
    )
    lines.append(r"\hline\hline")
    lines.append(
        r"\textbf{Study Name} & \textbf{Abbreviation} & "
        r"\textbf{Total} & \textbf{Mutated} & \textbf{Missense} & \textbf{Nonsense} \\",
    )
    lines.append(r"\hline")
    for _, row in df.iterrows():
        study_name = str(row["Study Name"])
        abbreviation = str(row["Study Abbreviation"])
        total = str(row["Number of Samples"])
        mutated = str(row["# Mutated Samples"])
        missense = str(row["# Missense"])
        nonsense = str(row["# Nonsense"])

        lines.append(
            f"{study_name} & {abbreviation} & {total} & {mutated} & "
            f"{missense} & {nonsense} \\\\",
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    if not Path(args.study_abbrev_fn).exists():
        logging.error("File not found: %s", args.study_abbrev_fn)
    if not Path(args.pancancer_counts_fn).exists():
        logging.error("File not found: %s", args.pancancer_counts_fn)
        sys.exit(1)
    abbrev_df = pd.read_csv(args.study_abbrev_fn)
    pancancer_df = pd.read_csv(args.pancancer_counts_fn)
    pancancer_df["Number of Samples"] = pancancer_df["Number of Samples"].astype(int)
    abbrev_df_renamed = abbrev_df.rename(columns={"Study Abbreviation": "Subtype"})
    ref_df = abbrev_df_renamed.merge(pancancer_df, on="Subtype", how="inner")
    subtypes = sorted(os.listdir(args.results_dir))
    results_list = []

    for stype in subtypes:
        stype_path = Path(args.results_dir) / stype
        if not stype_path.is_dir():
            continue
        cnt_mtx_fn = stype_path / "count_matrix.csv"
        if not cnt_mtx_fn.exists():
            logging.warning(
                "No count_matrix.csv found for subtype %s. Skipping.",
                stype,
            )
            continue
        cnt_df = pd.read_csv(cnt_mtx_fn)
        num_mut_samples = cnt_df.shape[0]
        miss_cols = [c for c in cnt_df.columns if c.endswith("_M")]
        missense_total = cnt_df[miss_cols].to_numpy().sum() if miss_cols else 0
        nons_cols = [c for c in cnt_df.columns if c.endswith("_N")]
        nonsense_total = cnt_df[nons_cols].to_numpy().sum() if nons_cols else 0
        results_list.append(
            {
                "Subtype": stype,
                "# Mutated Samples": int(num_mut_samples),
                "# Missense": int(missense_total),
                "# Nonsense": int(nonsense_total),
            },
        )
    results_df = pd.DataFrame(results_list)
    final_df = ref_df.merge(results_df, on="Subtype", how="inner")
    final_df["Number of Samples"] = final_df["Number of Samples"].astype(int)
    final_df = final_df.rename(columns={"Subtype": "Study Abbreviation"})
    final_df = final_df[
        [
            "Study Name",
            "Study Abbreviation",
            "Number of Samples",
            "# Mutated Samples",
            "# Missense",
            "# Nonsense",
        ]
    ]
    final_df = final_df.sort_values(by="Study Abbreviation")
    latex_str = create_latex_table(
        df=final_df,
        caption="Summary of TCGA Subtypes, Samples, and Mutations",
        label="tab:tcga_subtypes",
    )

    fout = Path(args.out) / "tcga_data_overview_table.tex"
    with fout.open("w") as f:
        f.write(latex_str)

    logging.info("Wrote LaTeX table to %s", fout)

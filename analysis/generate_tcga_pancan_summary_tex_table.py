"""TODO: Add docstring."""

import os

import pandas as pd
from dialect.utils.argument_parser import build_analysis_argument_parser


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def create_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """TODO: Add docstring."""
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


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        add_study_abbrev_fn=True,
        add_pancancer_counts_fn=True,
    )
    args = parser.parse_args()
    abbrev_df = pd.read_csv(args.study_abbrev_fn)
    pancancer_df = pd.read_csv(args.pancancer_counts_fn)
    pancancer_df["Number of Samples"] = pancancer_df["Number of Samples"].astype(int)
    abbrev_df_renamed = abbrev_df.rename(columns={"Study Abbreviation": "Subtype"})
    ref_df = abbrev_df_renamed.merge(pancancer_df, on="Subtype", how="inner")
    subtypes = sorted(os.listdir(args.results_dir))
    results_list = []

    for stype in subtypes:
        stype_path = args.results_dir / stype
        if not stype_path.is_dir():
            continue
        cnt_mtx_fn = stype_path / "count_matrix.csv"
        if not cnt_mtx_fn.exists():
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

    fout = args.out_dir / "tcga_data_overview_table.tex"
    with fout.open("w") as f:
        f.write(latex_str)


if __name__ == "__main__":
    main()

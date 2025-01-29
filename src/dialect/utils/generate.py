"""TODO: Add docstring."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dialect.utils.helpers import check_file_exists


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def convert_maf_to_cbase_input_file(maf: str, dout: str) -> str:
    """TODO: Add docstring."""
    maf_df = pd.read_csv(maf, sep="\t", low_memory=False)
    maf_df = maf_df.rename(
        columns={
            "Chromosome": "CHROM",
            "Start_Position": "POS",
            "Entrez_Gene_Id": "ID",
            "Reference_Allele": "REF",
            "Tumor_Seq_Allele2": "ALT",
            "Tumor_Sample_Barcode": "SAMPLE_BARCODE",
        },
    )[["CHROM", "POS", "ID", "REF", "ALT", "SAMPLE_BARCODE"]]
    fout = Path(dout) / "cbase_input.tsv"
    maf_df.to_csv(fout, sep="\t", header=False, index=False)
    return fout


def generate_bmr_using_cbase(
    maf: str,
    out: str,
    reference: str,
    threshold: str,
) -> None:
    """TODO: Add docstring."""
    cbase_input_fn = convert_maf_to_cbase_input_file(maf, out)

    cbase_output_dir = Path(out) / "CBaSE_output"
    cbase_output_dir.mkdir(exist_ok=True)

    cbase_params_script = Path("external") / "CBaSE" / "CBaSE_params_v1.2.py"
    cbase_params_script = cbase_params_script.resolve()

    cbase_qvals_script = Path("external") / "CBaSE" / "CBaSE_qvals_v1.2.py"
    cbase_qvals_script = cbase_qvals_script.resolve()

    cbase_auxiliary_dir = Path("external") / "CBaSE" / "auxiliary"
    cbase_auxiliary_dir = cbase_auxiliary_dir.resolve()

    try:
        cbase_params_cmd = [
            "python",
            str(cbase_params_script),
            str(cbase_input_fn),
            "1",
            str(reference),
            "3",
            "0",
            str(out),
            str(cbase_auxiliary_dir),
            str(cbase_output_dir),
        ]
        subprocess.run(cbase_params_cmd, check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)

    try:
        cbase_qvals_cmd = [
            "python",
            cbase_qvals_script,
            out,
            cbase_output_dir,
            threshold,
        ]
        subprocess.run(cbase_qvals_cmd, check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


def generate_counts_from_cbase_output(out: str) -> None:
    """TODO: Add docstring."""
    cbase_kept_mutations_fn = Path(out) / "CBaSE_output" / "kept_mutations.csv"

    mut_df = pd.read_csv(cbase_kept_mutations_fn, sep="\t")
    mut_df = mut_df[mut_df["effect"].isin(["missense", "nonsense"])]
    gene_level_df = mut_df.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    gene_level_df.to_csv(
        Path(out) / "gene_level_count_matrix.csv",
        index=True,
    )
    mut_df["gene"] = mut_df["gene"] + "_" + mut_df["effect"].str[0].str.upper()
    mut_df = mut_df.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    mut_df.to_csv(
        Path(out) / "count_matrix.csv",
        index=True,
    )


def generate_bmr_files_from_cbase_output(out: str) -> None:
    """TODO: Add docstring."""
    mis_bmr_fn = Path(out) / "CBaSE_output" / "pofmigivens.txt"
    non_bmr_fn = Path(out) / "CBaSE_output" / "pofkigivens.txt"

    all_dfs = []
    for fn, suffix in zip([mis_bmr_fn, non_bmr_fn], ["_M", "_N"]):
        with fn.open() as f:
            lines = f.readlines()
            file_length = len(lines)
            max_cols = np.max([len(line.split("\t")) for line in lines])

        column_names = ["gene", *list(range(max_cols))]
        pmf_df = pd.read_csv(
            fn,
            sep="\t",
            names=column_names,
            skiprows=range(0, file_length, 2),
        )

        # Modify gene names with the appropriate suffix
        pmf_df["gene"] = pmf_df["gene"].str.rsplit("_", n=1).str[0] + suffix

        pmf_df = pmf_df.set_index("gene", drop=True)
        all_dfs.append(pmf_df)

    pmf_df = pd.concat(all_dfs)
    pmf_df.to_csv(Path(out) / "bmr_pmfs.csv", index=True)


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def generate_bmr_and_counts(
    maf: str,
    out: str,
    reference: str,
    threshold: str,
) -> None:
    """TODO: Add docstring."""
    check_file_exists(maf)
    generate_bmr_using_cbase(maf, out, reference, threshold)
    generate_bmr_files_from_cbase_output(out)
    generate_counts_from_cbase_output(out)

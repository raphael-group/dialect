import os
import logging
import subprocess
import numpy as np
import pandas as pd

from dialect.utils.helpers import *

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


def convert_maf_to_CBaSE_input_file(maf, out):
    """
    Converts a MAF file to the CBaSE accepted VCF input format with sample barcode in the last column.
    + Renames columns to align with CBaSE requirements.
    + Saves the reformatted data to the specified output directory.

    @param maf (str): Path to the original MAF file.
        ! 'Chromosome' column should use single values (e.g., 1, 2, X), not prefixed values (e.g., chr1).
    @param out (str): Directory where the reformatted file will be saved.

    @returns (str): Path to the saved CBaSE input file.
    """
    logging.info(f"Converting MAF file: {maf} to CBaSE input file format.")
    df = pd.read_csv(maf, sep="\t", low_memory=False)
    df = df.rename(
        columns={
            "Chromosome": "CHROM",
            "Start_Position": "POS",
            "Entrez_Gene_Id": "ID",
            "Reference_Allele": "REF",
            "Tumor_Seq_Allele2": "ALT",
            "Tumor_Sample_Barcode": "SAMPLE_BARCODE",
        }
    )[["CHROM", "POS", "ID", "REF", "ALT", "SAMPLE_BARCODE"]]
    out_fn = os.path.join(out, "cbase_input.tsv")
    df.to_csv(out_fn, sep="\t", header=False, index=False)
    logging.info(f"CBaSE input file saved to: {out_fn}")
    return out_fn


def generate_bmr_using_CBaSE(maf, out, reference):
    """
    Generates background mutation rate (BMR) distributions and count matrix using the CBaSE method.

    @param maf (str): Path to the input MAF file.
    @param out (str): Directory where outputs and intermediate files will be saved.
    @param reference (str): Genome reference build to use (e.g., hg19 or hg38).
    Raises subprocess.CalledProcessError if any CBaSE script fails.
    """

    logging.info(
        f"Generating BMR and count matrix for MAF file: {maf} using CBaSE method."
    )
    CBaSE_input_fn = convert_maf_to_CBaSE_input_file(maf, out)

    # make a folder for CBaSE intermediate files
    CBaSE_output_dir = os.path.join(out, "CBaSE_output")
    os.makedirs(CBaSE_output_dir, exist_ok=True)

    logging.info(f"Running CBaSE method on input file: {CBaSE_input_fn}")

    CBaSE_params_script = os.path.abspath(
        os.path.join("external", "CBaSE_params_v1.2.py")
    )
    CBaSE_qvals_script = os.path.abspath(
        os.path.join("external", "CBaSE_qvals_v1.2.py")
    )
    CBaSE_auxiliary_dir = os.path.abspath(os.path.join("external", "auxiliary"))

    try:
        cbase_params_cmd = [
            "python",
            CBaSE_params_script,
            CBaSE_input_fn,
            "1",
            reference,
            "3",
            "0",
            out,
            CBaSE_auxiliary_dir,
            CBaSE_output_dir,
        ]
        subprocess.run(cbase_params_cmd, check=True)
        logging.info("CBaSE parameter script completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"CBaSE parameter script failed with error: {e}")
        raise

    try:
        logging.info("Running CBaSE q-value script.")
        cbase_qvals_cmd = [
            "python",
            CBaSE_qvals_script,
            out,
            CBaSE_output_dir,
        ]
        subprocess.run(cbase_qvals_cmd, check=True)
        logging.info("CBaSE q-value script completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"CBaSE q-value script failed with error: {e}")
        raise


def generate_counts_from_CBaSE_output(out):
    """
    Generates a count matrix from CBaSE output, focusing on retained missense and nonsense mutations.

    @param out (str): Directory containing the CBaSE output files.
    Processes mutations, groups them by gene and effect, and creates a pivoted count matrix.
    The resulting matrix is saved as 'count_matrix.csv' in the specified output directory.
    TODO: validate that only missense and nonsense should be kept.
    ? shouldn't we include synonymous mutations in count matrix?
    """
    logging.info(
        "Generating count matrix from CBaSE output of retained missense and nonsense mutations."
    )
    CBaSE_kept_mutations_fn = os.path.join(
        out, "CBaSE_output", "output_kept_mutations.csv"
    )

    df = pd.read_csv(CBaSE_kept_mutations_fn, sep="\t")
    df = df[df["effect"].isin(["missense", "nonsense"])]
    df["gene"] = df["gene"] + "_" + df["effect"].str[0].str.upper()
    df = df.pivot_table(index="gene", columns="sample", aggfunc="size", fill_value=0).T
    df.to_csv(os.path.join(out, "count_matrix.csv"), index=True)
    logging.info(f"Count matrix saved to: {os.path.join(out, 'count_matrix.csv')}")


def generate_bmr_files_from_CBaSE_output(out):
    """
    Generates BMR PMF files from CBaSE output for missense and nonsense mutations.

    @param out (str): Directory containing the CBaSE output files.
    Processes the BMR probability files ('pofmigivens.txt' for missense and 'pofkigivens.txt' for nonsense),
    appends mutation type suffixes to gene names ('_M' for missense, '_N' for nonsense),
    and saves the combined probability mass functions as 'bmr_pmfs.csv' in the output directory.
    """
    logging.info("Generating BMR PMF files from CBaSE output.")
    mis_bmr_fn = os.path.join(out, "CBaSE_output", "pofmigivens_output.txt")
    non_bmr_fn = os.path.join(out, "CBaSE_output", "pofkigivens_output.txt")

    all_dfs = []
    for fn, suffix in zip([mis_bmr_fn, non_bmr_fn], ["_M", "_N"]):
        with open(fn, "r") as f:
            lines = f.readlines()
            file_length = len(lines)
            max_cols = np.max([len(line.split("\t")) for line in lines])

        column_names = ["gene"] + list(range(0, max_cols))
        df = pd.read_csv(
            fn, sep="\t", names=column_names, skiprows=range(0, file_length, 2)
        )

        # Modify gene names with the appropriate suffix
        df["gene"] = df["gene"].str.rsplit("_", n=1).str[0] + suffix

        df.set_index("gene", drop=True, inplace=True)
        all_dfs.append(df)

    df = pd.concat(all_dfs)
    df.to_csv(os.path.join(out, "bmr_pmfs.csv"), index=True)
    logging.info(f"CBaSE BMR PMFs saved to: {os.path.join(out, 'bmr_pmfs.csv')}")


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #


def generate_bmr_and_counts(maf, out, reference):
    """
    Main function to generate background mutation rate (BMR) distributions and a count matrix.

    @param maf (str): Path to the input MAF file.
    @param out (str): Directory where outputs will be saved.
    @param reference (str): Genome reference build to use (e.g., hg19 or hg38).
    Validates the input file, creates necessary directories, and orchestrates BMR generation and count matrix creation.
    """
    logging.info(f"Generating BMR and count matrix for MAF file: {maf}")
    check_file_exists(maf)
    # generate_bmr_using_CBaSE(maf, out, reference)
    generate_bmr_files_from_CBaSE_output(out)
    generate_counts_from_CBaSE_output(out)
    logging.info("BMR and count matrix generation completed successfully.")

import os
import subprocess
import pandas as pd

import logging


def check_file_exists(maf):
    """Check if the input file exists."""
    logging.info(f"Validating input file: {maf}")
    if not os.path.exists(maf):
        raise FileNotFoundError(f"File not found: {maf}")


def convert_maf_to_CBaSE_input_file(maf, out):
    "Convert MAF to CBaSE input file format."
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


def generate_using_CBaSE(maf, out, reference):
    """
    Use the CBaSE method to generate background mutation rate (BMR) distributions and create a count matrix.
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


def generate_bmr_and_counts(maf, out, reference):
    """
    Generate background mutation rate (BMR) distributions and create a count matrix.

    This function uses an external method to generate the BMR distributions based on the provided
    mutation annotation format (MAF) file. It then creates a count matrix according to the mutations
    used by the BMR method.

    Args:
        maf (str): Path to the mutation annotation format (MAF) file.
        out (str): Path to the output file where the results will be saved.
        reference (str): Path to the reference genome file.

    Returns:
        None
    """
    logging.info(f"Generating BMR and count matrix for MAF file: {maf}")
    check_file_exists(maf)
    os.makedirs(out, exist_ok=True)
    generate_using_CBaSE(maf, out, reference)


# TODO: convert the following into simpler functions to create BMR file and count matrix file

# def generate_cnt_mtx_and_bmr(cbase_dout, subtype, subtype_dout):
#     """Generate count matrix and BMR distributions."""
#     logging.info("Building count matrix and BMR tables.")
#     mutations_csv = os.path.join(cbase_dout, f"{subtype}_kept_mutations.csv")
#     cnt_mtx_df = build_cnt_mtx(mutations_csv)  # Assumes this function exists

#     mis_bmr_fn = os.path.join(cbase_dout, f"pofmigivens_{subtype}.txt")
#     non_bmr_fn = os.path.join(cbase_dout, f"pofkigivens_{subtype}.txt")
#     bmr_df = build_bmr_table(mis_bmr_fn, non_bmr_fn)  # Assumes this function exists

#     # Save outputs
#     cnt_mtx_fn = os.path.join(subtype_dout, f"{subtype}_cbase_cnt_mtx.csv")
#     bmr_fn = os.path.join(subtype_dout, f"{subtype}_cbase_bmr_pmfs.csv")
#     cnt_mtx_df.to_csv(cnt_mtx_fn, index=False)
#     bmr_df.to_csv(bmr_fn, index=False)

#     logging.info(f"Count matrix saved to: {cnt_mtx_fn}")
#     logging.info(f"BMR table saved to: {bmr_fn}")


# def build_cnt_mtx(mutations_fn):
#     df = pd.read_csv(mutations_fn, sep="\t")
#     df = df[df["effect"].isin(["missense", "nonsense"])]
#     df["gene"] = df["gene"] + "_" + df["effect"].str[0].str.upper()
#     return df.pivot_table(
#         index="gene", columns="sample", aggfunc="size", fill_value=0
#     ).T


# def build_bmr_table(mis_bmr_fn, non_bmr_fn):
#     all_dfs = []
#     for fn, suffix in zip([mis_bmr_fn, non_bmr_fn], ["_M", "_N"]):
#         with open(fn, "r") as f:
#             lines = f.readlines()
#             file_length = len(lines)
#             max_cols = np.max([len(line.split("\t")) for line in lines])

#         column_names = ["gene"] + list(range(0, max_cols))
#         df = pd.read_csv(
#             fn, sep="\t", names=column_names, skiprows=range(0, file_length, 2)
#         )

#         # Modify gene names with the appropriate suffix
#         df["gene"] = df["gene"].str.rsplit("_", n=1).str[0] + suffix

#         df.set_index("gene", drop=True, inplace=True)
#         df.index.name = None
#         all_dfs.append(df)

#     # Concatenate all dataframes and return
#     return pd.concat(all_dfs)

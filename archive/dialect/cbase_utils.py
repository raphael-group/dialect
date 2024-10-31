import numpy as np
import pandas as pd


def convert_maf_to_vcf(maf_fn):
    df = pd.read_csv(maf_fn, sep="\t", low_memory=False)

    return df.rename(  # rename to match cbase required format
        columns={
            "Chromosome": "CHROM",
            "Start_Position": "POS",
            "Entrez_Gene_Id": "ID",
            "Reference_Allele": "REF",
            "Tumor_Seq_Allele2": "ALT",
            "Tumor_Sample_Barcode": "SAMPLE_BARCODE",
        }
    )[["CHROM", "POS", "ID", "REF", "ALT", "SAMPLE_BARCODE"]]


def build_cnt_mtx(mutations_fn):
    df = pd.read_csv(mutations_fn, sep="\t")
    df = df[df["effect"].isin(["missense", "nonsense"])]
    df["gene"] = df["gene"] + "_" + df["effect"].str[0].str.upper()
    return df.pivot_table(index="gene", columns="sample", aggfunc="size", fill_value=0).T


def build_bmr_table(mis_bmr_fn, non_bmr_fn):
    all_dfs = []
    for fn, suffix in zip([mis_bmr_fn, non_bmr_fn], ["_M", "_N"]):
        with open(fn, "r") as f:
            lines = f.readlines()
            file_length = len(lines)
            max_cols = np.max([len(line.split("\t")) for line in lines])

        column_names = ["gene"] + list(range(0, max_cols))
        df = pd.read_csv(fn, sep="\t", names=column_names, skiprows=range(0, file_length, 2))

        # Modify gene names with the appropriate suffix
        df["gene"] = df["gene"].str.rsplit("_", n=1).str[0] + suffix

        df.set_index("gene", drop=True, inplace=True)
        df.index.name = None
        all_dfs.append(df)

    # Concatenate all dataframes and return
    return pd.concat(all_dfs)

import pandas as pd

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-maf", required=True)
    parser.add_argument("-fout", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    df = pd.read_csv(args.maf, sep="\t", low_memory=False)

    pancan_columns = [
        "Chromosome",
        "Start_Position",
        "Entrez_Gene_Id",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode",
    ]
    lawrence_columns = [
        "chr",
        "pos",
        "gene",
        "ref_allele",
        "newbase",
        "patient",
    ]

    if all(column in df.columns for column in pancan_columns):
        selected_columns = pancan_columns
    else:
        selected_columns = lawrence_columns
    df_out = df[selected_columns]
    df_out.columns = [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "SAMPLE_BARCODE",
    ]  # rename to match cbase vcf format
    df_out.to_csv(args.fout, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()

import os
import logging
import pandas as pd

from argparse import ArgumentParser


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def build_argument_parser():
    parser = ArgumentParser(description="Identify decoy genes")
    parser.add_argument(
        "-c", "--cnt", required=True, help="Path to the count matrix file"
    )
    parser.add_argument(
        "-d", "--drvr", required=True, help="Path to the driver genes file"
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Rank cutoff for high mutation frequency",
    )
    parser.add_argument("-s", "--subtype", required=True, help="Name of cancer subtype")
    parser.add_argument(
        "-o", "--out", required=True, help="Path to the output directory"
    )
    return parser


# ---------------------------------------------------------------------------- #
#                                MAIN FUNCTIONS                                #
# ---------------------------------------------------------------------------- #
def identify_decoy_genes(cnt_df, driver_genes, k, fout):
    logging.info("Identifying decoy genes")
    gene_mutation_counts = cnt_df.sum(axis=0).sort_values(ascending=False)
    top_k_genes = gene_mutation_counts.head(k).index
    decoy_genes = [gene for gene in top_k_genes if gene not in driver_genes]
    with open(fout, "w") as f:
        f.write("\n".join(decoy_genes))


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    cnt_df = pd.read_csv(args.cnt, index_col=0)
    drvr_df = pd.read_csv(args.drvr, sep="\t", index_col=0)
    driver_genes = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    os.makedirs(args.out, exist_ok=True)
    fout = os.path.join(args.out, f"{args.subtype}_decoy_genes.txt")
    identify_decoy_genes(
        cnt_df=cnt_df,
        driver_genes=driver_genes,
        k=args.top_k,
        fout=fout,
    )

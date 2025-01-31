"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def build_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
    parser = ArgumentParser(description="Identify decoy genes")
    parser.add_argument(
        "-c",
        "--cnt",
        required=True,
        help="Path to the count matrix file",
    )
    parser.add_argument(
        "-d",
        "--drvr",
        required=True,
        help="Path to the driver genes file",
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
        "-o",
        "--out",
        required=True,
        help="Path to the output directory",
    )
    return parser


# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def identify_likely_passenger_genes(
    cnt_df: pd.DataFrame,
    driver_genes: list,
    k: int,
    fout: Path,
) -> None:
    """TODO: Add docstring."""
    decoy_genes_df = cnt_df.drop(driver_genes, axis=1, errors="ignore")
    top_decoy_genes = (
        decoy_genes_df.sum(axis=0).sort_values(ascending=False).head(k).index
    )
    with fout.open("w") as f:
        f.write("\n".join(top_decoy_genes))


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    cnt_df = pd.read_csv(args.cnt, index_col=0)
    drvr_df = pd.read_csv(args.drvr, sep="\t", index_col=0)
    driver_genes = set(drvr_df.index + "_M") | set(drvr_df.index + "_N")
    dout = Path(args.out)
    dout.mkdir(parents=True, exist_ok=True)
    fout = dout / f"{args.subtype}.txt"
    identify_likely_passenger_genes(
        cnt_df=cnt_df,
        driver_genes=driver_genes,
        k=args.top_k,
        fout=fout,
    )

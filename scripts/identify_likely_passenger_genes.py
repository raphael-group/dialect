"""TODO: Add docstring."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def build_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
    parser = ArgumentParser()
    parser.add_argument("-s", "--subtype", required=True)
    parser.add_argument("-k", "--top_k", type=int, default=100)
    parser.add_argument("-g", "--gene_level", action="store_true")
    parser.add_argument("-o", "--out_dir", type=Path, required=True)
    parser.add_argument("-c", "--cnt_mtx_fn", type=Path, required=True)
    parser.add_argument("-d", "--putative_drivers_fn", type=Path, required=True)
    return parser


# ------------------------------------------------------------------------------------ #
#                                    MAIN FUNCTIONS                                    #
# ------------------------------------------------------------------------------------ #
def identify_likely_passenger_genes(
    cnt_df: pd.DataFrame,
    driver_genes: set,
    k: int,
    out_fn: Path,
) -> None:
    """TODO: Add docstring."""
    decoy_genes_df = cnt_df.drop(driver_genes, axis=1, errors="ignore")
    top_decoy_genes = (
        decoy_genes_df.sum(axis=0).sort_values(ascending=False).head(k).index
    )
    with out_fn.open("w") as f:
        f.write("\n".join(top_decoy_genes))


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    cnt_mtx_df = pd.read_csv(args.cnt_mtx_fn, index_col=0)
    putative_drivers_df = pd.read_csv(args.putative_drivers_fn, sep="\t", index_col=0)
    driver_genes = (
        set(putative_drivers_df.index)
        if args.gene_level
        else set(putative_drivers_df.index + "_M")
        | set(putative_drivers_df.index + "_N")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    identify_likely_passenger_genes(
        cnt_df=cnt_mtx_df,
        driver_genes=driver_genes,
        k=args.top_k,
        out_fn=args.out_dir / f"{args.subtype}.txt",
    )

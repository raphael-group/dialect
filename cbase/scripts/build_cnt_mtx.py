import pandas as pd

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-mis", action="store_true")
    group.add_argument("-non", action="store_true")
    parser.add_argument("-fout", required=True)
    parser.add_argument("-mut_fn", required=True)
    return parser.parse_args()


def main():
    """
    Generate a count matrix from a cbase mutation file.
    Subset mutation file by missense or nonsense mutations.
    Drops genes with no mutations.
    """
    args = get_args()
    df = pd.read_csv(args.mut_fn, sep="\t")
    if args.mis:
        df = df[df["effect"] == "missense"]
    elif args.non:
        df = df[df["effect"] == "nonsense"]
    pivot_df = df.pivot_table(
        index="gene", columns="sample", aggfunc="size", fill_value=0
    )
    pivot_df.T.to_csv(args.fout)


if __name__ == "__main__":
    main()

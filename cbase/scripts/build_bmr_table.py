import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", required=True)
    parser.add_argument("-d", required=True)
    parser.add_argument("-mtype", required=True)
    parser.add_argument("-mtx_fn", required=True)
    parser.add_argument("-bmr_fn", required=True)
    return parser.parse_args()


def create_bmr_pmf_table(bmr_fn):
    with open(bmr_fn, "r") as f:
        lines = f.readlines()
        file_length = len(lines)
        max_cols = np.max([len(line.split("\t")) for line in lines])
    column_names = ["gene"] + list(range(0, max_cols))
    df = pd.read_csv(  # nans are generated to fill in empty cols
        bmr_fn, sep="\t", names=column_names, skiprows=range(0, file_length, 2)
    )
    df["gene"] = df["gene"].str.rsplit("_", n=1).str[0]
    df.set_index("gene", drop=True, inplace=True)
    df.index.name = None
    return df


def create_obs_exp_table(cnt_df, pmf_dict):
    obs_mut_cnts = cnt_df.sum(axis=0).to_dict()  # calculate observed mutation counts
    exp_mut_cnts = {}  # calculate expected mutation counts
    for gene in cnt_df.columns:
        gene_pmf = pmf_dict[gene]
        ex_sum = cnt_df.shape[0] * sum([i * x for i, x in enumerate(gene_pmf)])
        exp_mut_cnts[gene] = ex_sum
    # create final obs-exp table
    obs_exp_df = pd.DataFrame(
        {
            "obs": [obs_mut_cnts[gene] for gene in cnt_df.columns],
            "exp": [exp_mut_cnts[gene] for gene in cnt_df.columns],
            "obs - exp": [
                obs_mut_cnts[gene] - exp_mut_cnts[gene] for gene in cnt_df.columns
            ],
        },
        index=cnt_df.columns,
    ).sort_values(by="obs - exp", ascending=False)
    return obs_exp_df


def main():
    args = get_args()

    ### step 1: read mutation cnt mtx
    cnt_df = pd.read_csv(args.mtx_fn)
    cnt_df.set_index("sample", inplace=True)

    ### step 2: read cbase bmr file, create df, and save
    bmr_df = create_bmr_pmf_table(args.bmr_fn)
    bmr_df.to_csv(os.path.join(args.d, f"{args.c}_{args.mtype}_bmr.csv"))

    ### step 3: generate dict from df and generate obs-exp table
    pmf_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    pmf_dict = {  # to drop nans caused by empty cols from df creation
        key: [x for x in pmf_dict[key] if not np.isnan(x)] for key in pmf_dict
    }
    obs_exp_df = create_obs_exp_table(cnt_df, pmf_dict)
    obs_exp_df.to_csv(os.path.join(args.d, f"{args.c}_{args.mtype}_obs_exp_table.csv"))


if __name__ == "__main__":
    main()

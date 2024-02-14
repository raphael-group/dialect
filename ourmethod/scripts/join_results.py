import os
import pandas as pd

from scipy.stats import chi2
from argparse import ArgumentParser

# TODO: CHANGE PATH FOR SERVER
OUR_DOUT = "/Users/ahmed/workspace/research/lvsi/ourmethod/out/{}/{}/{}"
THEIR_DOUT = "/Users/ahmed/workspace/research/lvsi/{}/out/{}/{}"


# Helper Functions
def get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", dest="cancer", required=True)
    parser.add_argument("-d", dest="dataset", required=True)
    parser.add_argument("-bmr", dest="bmr", required=True)
    return parser.parse_args()


def join_cbase_results(cancer, bmr, our_dout, their_dout):
    # Open files
    mis_df_name = os.path.join(our_dout, "{}_final_mis_single.csv".format(cancer))
    non_df_name = os.path.join(our_dout, "{}_final_non_single.csv".format(cancer))
    their_df_name = os.path.join(their_dout, "q_values_{}.txt".format(cancer))
    mis_obs_exp_df_name = os.path.join(
        their_dout, "{}_mis_obs_exp_table.csv".format(cancer)
    )
    non_obs_exp_df_name = os.path.join(
        their_dout, "{}_non_obs_exp_table.csv".format(cancer)
    )
    mis_df = pd.read_csv(mis_df_name)
    non_df = pd.read_csv(non_df_name)
    mis_obs_exp_df = pd.read_csv(mis_obs_exp_df_name, index_col=0)
    non_obs_exp_df = pd.read_csv(non_obs_exp_df_name, index_col=0)
    their_df = pd.read_csv(their_df_name, skiprows=[0], sep="\t")

    # Create p-vals & ranks from LLR values
    mis_df.insert(4, "pval", chi2.sf(mis_df["llr"], 3))
    mis_df = mis_df.sort_values(by="pval")
    mis_df = mis_df.reset_index(drop=True)
    mis_df.insert(5, "rank", mis_df["pval"].rank(method="min"))

    non_df.insert(4, "pval", chi2.sf(non_df["llr"], 3))
    non_df = non_df.sort_values(by="pval")
    non_df = non_df.reset_index(drop=True)
    non_df.insert(5, "rank", non_df["pval"].rank(method="min"))

    # Subset & create ranks from their_dfs
    mis_cols_to_keep = ["gene", "q_phi_m_neg", "q_phi_m_pos"]
    non_cols_to_keep = ["gene", "q_phi_k_neg", "q_phi_k_pos"]

    their_mis_df = their_df[mis_cols_to_keep]
    their_mis_df = their_mis_df.sort_values(by="q_phi_m_pos")
    their_mis_df = their_mis_df.reset_index(drop=True)
    their_mis_df["pos_rank"] = their_mis_df["q_phi_m_pos"].rank(method="min")
    their_mis_df = their_mis_df.sort_values(by="q_phi_m_neg")
    their_mis_df = their_mis_df.reset_index(drop=True)
    their_mis_df["neg_rank"] = their_mis_df["q_phi_m_neg"].rank(method="min")

    their_non_df = their_df[non_cols_to_keep]
    their_non_df = their_non_df.sort_values(by="q_phi_k_pos")
    their_non_df = their_non_df.reset_index(drop=True)
    their_non_df["pos_rank"] = their_non_df["q_phi_k_pos"].rank(method="min")
    their_non_df = their_non_df.sort_values(by="q_phi_k_neg")
    their_non_df = their_non_df.reset_index(drop=True)
    their_non_df["neg_rank"] = their_non_df["q_phi_k_neg"].rank(method="min")

    # Join their and our dfs
    # rename mis_obs_exp_df index to 'gene'
    mis_obs_exp_df = mis_obs_exp_df.rename_axis("gene").reset_index()
    non_obs_exp_df = non_obs_exp_df.rename_axis("gene").reset_index()
    joined_mis_df = pd.merge(mis_df, their_mis_df, on="gene", how="inner")
    joined_mis_df = pd.merge(joined_mis_df, mis_obs_exp_df, on="gene", how="inner")
    joined_non_df = pd.merge(non_df, their_non_df, on="gene", how="inner")
    joined_non_df = pd.merge(joined_non_df, non_obs_exp_df, on="gene", how="inner")

    # Save output
    joined_mis_df.to_csv(
        os.path.join(our_dout, "{}_{}_joined_mis_single.csv".format(cancer, bmr))
    )
    joined_non_df.to_csv(
        os.path.join(our_dout, "{}_{}_joined_non_single.csv".format(cancer, bmr))
    )


def join_dig_results(cancer, bmr, our_dout, their_dout):
    # Open files
    mis_df_name = os.path.join(our_dout, "{}_final_mis_single.csv".format(cancer))
    non_df_name = os.path.join(our_dout, "{}_final_non_single.csv".format(cancer))
    their_df_name = os.path.join(their_dout, "{}.results.txt".format(cancer))
    mis_df = pd.read_csv(mis_df_name)
    non_df = pd.read_csv(non_df_name)
    their_df = pd.read_csv(their_df_name, sep="\t")

    # Create p-vals & ranks from LLR values
    mis_df.insert(4, "pval", chi2.sf(mis_df["llr"], 3))
    mis_df = mis_df.sort_values(by="pval")
    mis_df = mis_df.reset_index(drop=True)
    mis_df.insert(5, "rank", mis_df["pval"].rank(method="min"))

    non_df.insert(4, "pval", chi2.sf(non_df["llr"], 3))
    non_df = non_df.sort_values(by="pval")
    non_df = non_df.reset_index(drop=True)
    non_df.insert(5, "rank", non_df["pval"].rank(method="min"))

    # Subset & create ranks from their_dfs
    mis_cols_to_keep = ["GENE", "PVAL_MIS_BURDEN", "PVAL_MIS_BURDEN_SAMPLE"]
    non_cols_to_keep = ["GENE", "PVAL_NONS_BURDEN", "PVAL_NONS_BURDEN_SAMPLE"]

    their_mis_df = their_df[mis_cols_to_keep]
    their_mis_df = their_mis_df.sort_values(by="PVAL_MIS_BURDEN")
    their_mis_df = their_mis_df.reset_index(drop=True)
    their_mis_df["d-rank"] = their_mis_df["PVAL_MIS_BURDEN"].rank(method="min")
    their_mis_df = their_mis_df.rename(columns={"GENE": "gene"})

    their_non_df = their_df[non_cols_to_keep]
    their_non_df = their_non_df.sort_values(by="PVAL_NONS_BURDEN")
    their_non_df = their_non_df.reset_index(drop=True)
    their_non_df["d-rank"] = their_non_df["PVAL_NONS_BURDEN"].rank(method="min")
    their_non_df = their_non_df.rename(columns={"GENE": "gene"})

    # Join their and our dfs
    joined_mis_df = pd.merge(mis_df, their_mis_df, on="gene", how="inner")
    joined_non_df = pd.merge(non_df, their_non_df, on="gene", how="inner")

    # Save output
    joined_mis_df.to_csv("{}_{}_joined_mis_single.csv".format(cancer, bmr))
    joined_non_df.to_csv("{}_{}_joined_non_single.csv".format(cancer, bmr))


def join_mutsig_results(cancer, bmr, our_dout, their_dout):
    mis_df_name = os.path.join(our_dout, "{}_final_mis_single.csv".format(cancer))
    non_df_name = os.path.join(our_dout, "{}_final_non_single.csv".format(cancer))
    ind_df_name = os.path.join(our_dout, "{}_final_ind_single.csv".format(cancer))
    their_df_name = os.path.join(their_dout, "sig_genes.txt")
    mis_df = pd.read_csv(mis_df_name)
    non_df = pd.read_csv(non_df_name)
    ind_df = pd.read_csv(ind_df_name)
    their_df = pd.read_csv(their_df_name, sep="\t")

    # Create p-vals & ranks from LLR values
    mis_df.insert(4, "pval", chi2.sf(mis_df["llr"], 3))
    mis_df = mis_df.sort_values(by="pval")
    mis_df = mis_df.reset_index(drop=True)
    mis_df.insert(5, "rank", mis_df["pval"].rank(method="min"))

    non_df.insert(4, "pval", chi2.sf(non_df["llr"], 3))
    non_df = non_df.sort_values(by="pval")
    non_df = non_df.reset_index(drop=True)
    non_df.insert(5, "rank", non_df["pval"].rank(method="min"))

    ind_df.insert(4, "pval", chi2.sf(ind_df["llr"], 3))
    ind_df = ind_df.sort_values(by="pval")
    ind_df = ind_df.reset_index(drop=True)
    ind_df.insert(5, "rank", ind_df["pval"].rank(method="min"))

    # Subset & create ranks from their_dfs
    mis_cols_to_keep = ["rank", "gene", "p", "q"]
    non_cols_to_keep = ["rank", "gene", "p", "q"]
    ind_cols_to_keep = ["rank", "gene", "p", "q"]

    their_mis_df = their_df[mis_cols_to_keep]
    their_mis_df = their_mis_df.rename(columns={"rank": "m-rank"})

    their_non_df = their_df[non_cols_to_keep]
    their_non_df = their_non_df.rename(columns={"rank": "m-rank"})

    their_ind_df = their_df[ind_cols_to_keep]
    their_ind_df = their_ind_df.rename(columns={"rank": "m-rank"})

    # Join their and our dfs
    joined_mis_df = pd.merge(mis_df, their_mis_df, on="gene", how="inner")
    joined_non_df = pd.merge(non_df, their_non_df, on="gene", how="inner")
    joined_ind_df = pd.merge(ind_df, their_ind_df, on="gene", how="inner")

    # Save output
    joined_mis_df.to_csv("{}_mis_{}.csv".format(cancer, bmr))
    joined_non_df.to_csv("{}_non_{}.csv".format(cancer, bmr))
    joined_ind_df.to_csv("{}_ind_{}.csv".format(cancer, bmr))


def main():
    args = get_args()
    our_dout = OUR_DOUT.format(args.bmr, args.dataset, args.cancer)
    their_dout = THEIR_DOUT.format(args.bmr, args.dataset, args.cancer)
    if args.bmr == "cbase":
        join_cbase_results(args.cancer, args.bmr, our_dout, their_dout)
    elif args.bmr == "dig":
        join_dig_results(args.cancer, args.bmr, our_dout, their_dout)
    elif args.bmr == "mutsig":
        join_mutsig_results(args.cancer, args.bmr, our_dout, their_dout)


if __name__ == "__main__":
    main()

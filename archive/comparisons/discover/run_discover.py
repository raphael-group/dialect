import os
import discover
import numpy as np
import pandas as pd

from itertools import combinations
from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("cnt_mtx_fn", help="Path to the count matrix file")
    parser.add_argument("dout", help="Path to the output directory")
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--feature_level", default="mutation", choices=["gene", "mutation"])
    return parser


def discover_all_pairs(cnt_mtx_df, interaction_type):
    if interaction_type not in {"me", "co"}:
        raise ValueError("interaction_type must be 'me' or 'co'")
    C = cnt_mtx_df.to_numpy()
    X = np.where(C, 1, 0)  # Binarize count matrix
    X_df = pd.DataFrame(X.T)  # genes by samples
    X_df.index = cnt_mtx_df.columns
    X_df.columns = cnt_mtx_df.index
    events = discover.DiscoverMatrix(X_df)
    if interaction_type == "me":
        return discover.pairwise_discover_test(events, alternative="less")
    else:
        return discover.pairwise_discover_test(events, alternative="greater")


def create_result_lists(result, pairs):
    pvals, qvals = [], []
    for gene_a, gene_b in pairs:
        pval = (
            result.pvalues.loc[gene_a, gene_b]
            if not np.isnan(result.pvalues.loc[gene_a, gene_b])
            else result.pvalues.loc[gene_b, gene_a]
        )
        pvals.append(pval)
        qval = (
            result.qvalues.loc[gene_a, gene_b]
            if not np.isnan(result.qvalues.loc[gene_a, gene_b])
            else result.qvalues.loc[gene_b, gene_a]
        )
        qvals.append(qval)
    return pvals, qvals


def run_discover():
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.exists(args.cnt_mtx_fn), "File not found: %s" % args.cnt_mtx_fn
    cnt_mtx_df = pd.read_csv(args.cnt_mtx_fn, index_col=0)
    if args.feature_level == "mutation":
        top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[: args.top_k]
    else:  # change cnt_mtx to reflect gene level features
        cnt_mtx_df = cnt_mtx_df.groupby(lambda x: x.split("_")[0], axis=1).sum()
        top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[: args.top_k]
    cnt_mtx_df = cnt_mtx_df[top_k_genes]

    me_result_mutex = discover_all_pairs(cnt_mtx_df, interaction_type="me")
    co_result_mutex = discover_all_pairs(cnt_mtx_df, interaction_type="co")

    pairs = list(combinations(top_k_genes, 2))
    me_pvals, me_qvals = create_result_lists(me_result_mutex, pairs)
    co_pvals, co_qvals = create_result_lists(co_result_mutex, pairs)

    results_df = pd.DataFrame(
        {
            "gene_a": [gene_a for gene_a, _ in pairs],
            "gene_b": [gene_b for _, gene_b in pairs],
            "me_pval": me_pvals,
            "me_qval": me_qvals,
            "co_pval": co_pvals,
            "co_qval": co_qvals,
        },
        columns=["gene_a", "gene_b", "me_pval", "me_qval", "co_pval", "co_qval"],
    )
    results_df.to_csv(
        os.path.join(args.dout, "discover_" + args.feature_level + "_results.csv"), index=False
    )


if __name__ == "__main__":
    run_discover()

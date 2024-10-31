import pandas as pd

from tqdm import tqdm
from scipy import stats
from typing import Literal, List
from itertools import combinations
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import multipletests


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("cnt_mtx_fn", help="Path to the count matrix file")
    parser.add_argument("dout", help="Path to the output directory")
    parser.add_argument("--top_k", default=100, type=int)
    return parser


def fishers_exact_test(
    gene_a_binarized_mutations: List[int],
    gene_b_binarized_mutations: List[int],
    interaction_type: Literal["me", "co"],
):
    if interaction_type not in {"me", "co"}:
        raise ValueError("interaction_type must be 'me' or 'co'")

    alternative = "less" if interaction_type == "me" else "greater"
    contingency_table = confusion_matrix(
        gene_a_binarized_mutations, gene_b_binarized_mutations, labels=[1, 0]
    )
    oddsratio, pvalue = stats.fisher_exact(contingency_table, alternative=alternative)

    return oddsratio, pvalue


def run_fishers_all_pairs(cnt_mtx_df, gene_pairs, interaction_type):
    param_estimates = {}
    for gene_a, gene_b in tqdm(gene_pairs):
        gene_a_binarized_mutations = cnt_mtx_df[gene_a] > 0
        gene_b_binarized_mutations = cnt_mtx_df[gene_b] > 0
        oddsratio, pvalue = fishers_exact_test(
            gene_a_binarized_mutations, gene_b_binarized_mutations, interaction_type
        )
        param_estimates[(gene_a, gene_b)] = pvalue
    gene_a_col = [gene_a for gene_a, _ in param_estimates.keys()]
    gene_b_col = [gene_b for _, gene_b in param_estimates.keys()]
    pval = list(param_estimates.values())
    qval = multipletests(pval, alpha=0.05, method="fdr_bh")[1]
    return pd.DataFrame(
        {
            "gene_a": gene_a_col,
            "gene_b": gene_b_col,
            f"{interaction_type}_pval": pval,
            f"{interaction_type}_qval": qval,
        }
    )


def run_fishers():
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.exists(args.cnt_mtx_fn), f"File not found: {args.cnt_mtx_fn}"
    cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
    top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[:top_k]
    gene_pairs = list(combinations(top_k_genes, 2))
    me_results_df = run_fishers_all_genes(cnt_mtx_df, args.top_k, interaction_type="me")
    co_results_df = run_fishers_all_genes(cnt_mtx_df, args.top_k, interaction_type="co")
    results_df = pd.merge(me_results_df, co_results_df, on=["gene_a", "gene_b"])
    results_df.to_csv(os.path.join(args.dout, "fishers_results.csv"))


if __name__ == "__main__":
    run_fishers()

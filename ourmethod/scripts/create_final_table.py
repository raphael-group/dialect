import os
import discover
import numpy as np
import pandas as pd
import discover.datasets

from scipy import stats
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix


# Helper Functions
def get_args():
    parser = ArgumentParser()

    parser.add_argument("--count_mtx_fn", dest="count_mtx", required=True)
    parser.add_argument("--cancer", dest="cancer", required=True)
    parser.add_argument("--orig_discover_fout", dest="discover_fout", required=False)

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--single", action="store_true")
    model_group.add_argument("--pair", action="store_true")

    in_group = parser.add_mutually_exclusive_group(required=True)
    in_group.add_argument("--single_fn", dest="single_fn")
    in_group.add_argument("--pair_fn", dest="pair_fn")

    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--single_fout", dest="single_fout")
    out_group.add_argument("--pair_fout", dest="pair_fout")


    return parser.parse_args()


def get_univariate_columns(single_df, count_mtx_df):
    mutation_counts_series = count_mtx_df.sum()
    mutation_counts_dict = {
        x: y for x, y in zip(mutation_counts_series.index, mutation_counts_series)
    }
    mutation_counts = np.array([mutation_counts_dict[x] for x in single_df["gene"]])
    mutation_freqs = np.array(
        [np.count_nonzero(count_mtx_df[x]) for x in single_df["gene"]]
    )
    mutation_freqs = mutation_freqs.astype(float)  # for float division in python 2
    mutation_fracs = mutation_freqs / mutation_counts
    return mutation_counts, mutation_freqs, mutation_fracs


def run_original_discover(pair_df, count_mtx_df, fout):
    C = count_mtx_df.to_numpy()
    X = np.where(C, 1, 0)  # Binarize count matrix
    X_df = pd.DataFrame(X.T)  # genes by samples
    X_df.index = count_mtx_df.columns
    X_df.columns = count_mtx_df.index
    events = discover.DiscoverMatrix(X_df)
    subset = set(pair_df["gene_1"].tolist() + pair_df["gene_2"].tolist())
    subset = set([x.split('_')[0] for x in subset])
    me_result_mutex = discover.pairwise_discover_test(events[subset])
    me_result_mutex.significant_pairs(0.05).to_csv('{}_me_top_pairs.csv'.format(fout))
    me_result_mutex.qvalues.to_csv('{}_me_qvalues.csv'.format(fout))
    #co_result_mutex = discover.pairwise_discover_test(events, alternative="greater")
    #co_result_mutex.significant_pairs().to_csv('{}_co_result_mutex.csv'.format(cancer))

def calculate_discover(pair_df, count_mtx_df):
    C = count_mtx_df.to_numpy()
    X = np.where(C, 1, 0)  # Binarize count matrix
    X_df = pd.DataFrame(X.T)  # genes by samples
    X_df.index = count_mtx_df.columns
    X_df.columns = count_mtx_df.index
    events = discover.DiscoverMatrix(X_df)
    subset = set(pair_df["gene_1"].tolist() + pair_df["gene_2"].tolist())
    me_result_mutex = discover.pairwise_discover_test(events[subset])
    co_result_mutex = discover.pairwise_discover_test(
        events[subset], alternative="greater"
    )
    print(me_result_mutex.significant_pairs(0.05))
    me_pvals, me_qvals, co_pvals, co_qvals = [], [], [], []
    for gene_1, gene_2 in zip(pair_df["gene_1"], pair_df["gene_2"]):
        me_pval = (
            me_result_mutex.pvalues.loc[gene_1, gene_2]
            if not np.isnan(me_result_mutex.pvalues.loc[gene_1, gene_2])
            else me_result_mutex.pvalues.loc[gene_2, gene_1]
        )
        me_qval = (
            me_result_mutex.qvalues.loc[gene_1, gene_2]
            if not np.isnan(me_result_mutex.qvalues.loc[gene_1, gene_2])
            else me_result_mutex.qvalues.loc[gene_2, gene_1]
        )
        me_pvals.append(me_pval)
        me_qvals.append(me_qval)

        co_pval = (
            co_result_mutex.pvalues.loc[gene_1, gene_2]
            if not np.isnan(co_result_mutex.pvalues.loc[gene_1, gene_2])
            else co_result_mutex.pvalues.loc[gene_2, gene_1]
        )
        co_qval = (
            co_result_mutex.qvalues.loc[gene_1, gene_2]
            if not np.isnan(co_result_mutex.qvalues.loc[gene_1, gene_2])
            else co_result_mutex.qvalues.loc[gene_2, gene_1]
        )
        co_pvals.append(co_pval)
        co_qvals.append(co_qval)
    return me_pvals, me_qvals, co_pvals, co_qvals


def calculate_fishers(pair_df, count_mtx_df):
    fishers_exact_me_pvals = []
    fishers_exact_co_pvals = []
    contingency_matrix_vals = []
    for gene_1, gene_2 in zip(pair_df["gene_1"], pair_df["gene_2"]):
        gene_1_counts = count_mtx_df[gene_1].copy()
        gene_2_counts = count_mtx_df[gene_2].copy()
        gene_1_counts[gene_1_counts > 1] = 1
        gene_2_counts[gene_2_counts > 1] = 1
        gene_1_counts[gene_1_counts.isnull()] = 0
        gene_2_counts[gene_2_counts.isnull()] = 0
        cross_tab = confusion_matrix(gene_1_counts, gene_2_counts, labels=[1, 0])
        _, me_pval = stats.fisher_exact(cross_tab, alternative="less")
        _, co_pval = stats.fisher_exact(cross_tab, alternative="greater")
        fishers_exact_me_pvals.append(me_pval)
        fishers_exact_co_pvals.append(co_pval)
        contingency_matrix = np.array(cross_tab).flatten()
        contingency_matrix_vals.append(contingency_matrix)
    return fishers_exact_me_pvals, fishers_exact_co_pvals, contingency_matrix_vals


def calculate_overlapping_samples(pair_df, count_mtx_df):
    overlapping_samples_list = []
    sample_mut_counts = count_mtx_df.sum(axis=1)
    sample_mut_count_dict = {
        s[s.find("-") + 1 :]: count
        for s, count in zip([str(x) for x in sample_mut_counts.index], sample_mut_counts)
    }
    for gene_1, gene_2 in zip(pair_df["gene_1"], pair_df["gene_2"]):
        gene_1_mutations = count_mtx_df[count_mtx_df[gene_1] > 0]
        both_mutations = gene_1_mutations[gene_1_mutations[gene_2] > 0]
        if len(both_mutations) <= 8:
            overlapping_samples = both_mutations.index.tolist()
            overlapping_samples = [s[s.find("-") + 1 :] for s in [str(x) for x in overlapping_samples]]
            mut_counts = [sample_mut_count_dict[s] for s in overlapping_samples]
            sorted_pairs = sorted(
                zip(overlapping_samples, mut_counts),
                key=lambda pair: pair[1],
                reverse=True,
            )
            desc_sorted_overlapping_samples = [key for key, _ in sorted_pairs]
        else:
            desc_sorted_overlapping_samples = []
        overlapping_samples_list.append(", ".join(desc_sorted_overlapping_samples))
    return overlapping_samples_list


def get_pairs_subset(pair_df, top_k=10000):
    top_me = pair_df.sort_values(by="mei", ascending=False).head(top_k)
    top_co = pair_df.sort_values(by="coi", ascending=False).head(top_k)
    subset_df = pd.concat([top_me, top_co]).drop_duplicates()
    return subset_df


def main():
    args = get_args()
    count_mtx_df = pd.read_csv(args.count_mtx, index_col=0)

    if args.single:
        single_df = pd.read_csv(args.single_fn, sep="\t", index_col=0)
        mutation_counts, mutation_freqs, mutation_fracs = get_univariate_columns(
            single_df, count_mtx_df
        )
        single_df = single_df.assign(
            mut_count=mutation_counts, mut_freq=mutation_freqs, mut_frac=mutation_fracs
        )
        single_pi_vals = {x: y for x, y in zip(single_df["gene"], single_df["pi"])}
        single_df.sort_values(by="llr", ascending=False, inplace=True)
        single_df.to_csv(args.single_fout, index=False)
    else:  # pair
        pair_df = pd.read_csv(args.pair_fn, sep="\t", index_col=0)
        mis_singleton_fn = os.path.join(
            os.path.dirname(args.pair_fn),
            "{}_mis_single_pi_vals.csv".format(args.cancer),
        )
        mis_singleton_df = pd.read_csv(mis_singleton_fn, sep="\t", index_col=0)
        non_singleton_fn = os.path.join(
            os.path.dirname(args.pair_fn),
            "{}_non_single_pi_vals.csv".format(args.cancer),
        )
        non_singleton_df = pd.read_csv(non_singleton_fn, sep="\t", index_col=0)
        mis_singleton_df["gene"] = ["{}_M".format(x) for x in mis_singleton_df["gene"]]
        non_singleton_df["gene"] = ["{}_N".format(x) for x in non_singleton_df["gene"]]
        singleton_df = pd.concat([mis_singleton_df, non_singleton_df], axis=0)
        singleton_pi_vals = dict(zip(singleton_df["gene"], singleton_df["pi"]))
        pi_x = np.array([singleton_pi_vals[x] for x in pair_df["gene_1"]])
        pi_y = np.array([singleton_pi_vals[y] for y in pair_df["gene_2"]])
        
        # to run discover on the genes when combined by mutation type  
        new_df = pd.DataFrame()
        prefixes = set(col.split('_')[0] for col in count_mtx_df.columns)
        for prefix in prefixes:
            cols_to_combine = [col for col in count_mtx_df.columns if col.startswith(prefix + '_')]
            combined_col = count_mtx_df[cols_to_combine].sum(axis=1)
            new_df[prefix] = combined_col
        run_original_discover(pair_df, new_df, args.discover_fout)
        
        discover = calculate_discover(pair_df, count_mtx_df)
        me_fishers, co_fishers, cont_mats = calculate_fishers(pair_df, count_mtx_df)
        d_me_pvals, d_me_qvals, d_co_pvals, d_co_qvals = discover
        overlap_samples = calculate_overlapping_samples(pair_df, count_mtx_df)
        cont_11_vals, cont_10_vals, cont_01_vals, cont_00_vals = (
            list(t) for t in zip(*cont_mats)
        )
        pair_df["_00_"] = cont_00_vals
        pair_df["_10_"] = cont_10_vals
        pair_df["_01_"] = cont_01_vals
        pair_df["_11_"] = cont_11_vals
        # pair_df["pi_11/(pi_x*pi_y)"] = pair_df["pi_11"] / (pi_x * pi_y)
        pair_df["pi_11/((pi_11+pi_10)(pi_11+pi_01))"] = pair_df["pi_11"] / (
            (pair_df["pi_11"] + pair_df["pi_10"])
            * (pair_df["pi_11"] + pair_df["pi_01"])
        )
        # pair_df["(pi_10+pi_01)/((pi_x*(1-pi_y))+(pi_y*(1-pi_x)))"] = (
        #     pair_df["pi_10"] + pair_df["pi_01"]
        # ) / ((pi_x * (1 - pi_y)) + (pi_y * (1 - pi_x)))
        
        epsilon = 1e-6  # A small constant
        pair_df["log-odds"] = np.log(
            ((pair_df["pi_11"] + epsilon) * (pair_df["pi_00"] + epsilon)) /
            ((pair_df["pi_10"] + epsilon) * (pair_df["pi_01"] + epsilon))
        )
        pair_df["fp_me"] = me_fishers
        pair_df["fp_co"] = co_fishers
        pair_df["dp_me"] = d_me_pvals
        pair_df["dq_me"] = d_me_qvals
        pair_df["dp_co"] = d_co_pvals
        pair_df["dq_co"] = d_co_qvals
        pair_df["overlap"] = overlap_samples
        pair_df.to_csv(args.pair_fout, index=False)


if __name__ == "__main__":
    main()

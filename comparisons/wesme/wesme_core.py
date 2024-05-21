import os
import pandas as pd
from tqdm import tqdm
from time import time

from wesme_utils import *
from argparse import ArgumentParser
from statsmodels.stats.multitest import multipletests


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("dout", help="Path to the output directory")
    parser.add_argument(
        "rnum", type=int, help="Number of instances to generate for the random sampling"
    )
    # Require the user to either provide a maf_fn or a mut_fn
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--maf_fn", help="Path to the input MAF file")
    group.add_argument("--mut_fn", help="Path to the mutation list file")

    parser.add_argument("--all", action="store_true", help="Use all ks in weighted sampling")
    parser.add_argument("--top_k", default=100, type=int, help="Top k instances to use in sampling")

    return parser


def compute_sample_weights(mut_fn, dout):
    mut_list_dic, samples = read_mut_list(mut_fn, samples=True)
    mut_count = count_muts(mut_list_dic, samples)
    mut_freq = compute_sample_freq(mut_count)
    write_dic(mut_freq, os.path.join(dout, "sample_mut_freqs.txt"))


def run_weighted_sampling(mut_fn, freq_fn, rnum, dout, use_all=False):
    mut_list_dic, samples = read_mut_list(mut_fn, samples=True)
    freq_dic = read_dic(freq_fn)
    freqs = [float(freq_dic[s]) for s in samples]

    nsamples = len(samples)
    all_ks = range(nsamples) if use_all else comp_all_ks(mut_list_dic)
    cur_wrs_dir = os.path.join(dout, "weighted_sampling")
    if not os.path.exists(cur_wrs_dir):
        os.makedirs(cur_wrs_dir)
    for k in all_ks:
        if k == 0:
            continue
        ws_file = open(os.path.join(cur_wrs_dir, "_".join(["wr", str(k), str(rnum)]) + ".txt"), "w")
        for i in range(rnum):
            wsamples = np.random.choice(range(nsamples), k, p=freqs, replace=False)
            ws_file.write("%s\n" % ",".join([str(idx) for idx in wsamples]))
        ws_file.close()


def compute_pairwise_pvalues(mut_fn, dout, top_k):
    mut_list_dic, samples = read_mut_list(mut_fn, samples=True)
    gene_ks = dict([(gene, len(mut_list_dic[gene])) for gene in mut_list_dic])
    nsamples = len(samples)

    cur_wrs_dir = os.path.join(dout, "weighted_sampling")
    kfiles = os.listdir(cur_wrs_dir)
    ws_k_cover_dic = {}
    for kf in kfiles:
        k = int(kf.split(".")[0].split("_")[1])
        ws_k_cover_dic[k], samples = read_mut_list(os.path.join(cur_wrs_dir, kf), genes=False)

    ws_me_pv_dic, ws_me_ex_cover_sizes_dic = {}, {}
    ws_co_pv_dic, ws_co_ex_cover_sizes_dic = {}, {}
    fisher_co_pv_dic, jaccard_dic = {}, {}  # hypergeom pvals, jaccard index
    genes = list(mut_list_dic.keys())
    genes = sorted(genes, key=lambda x: gene_ks[x], reverse=True)[:top_k]
    for i in tqdm(range(len(genes))):
        gene1 = genes[i]
        for j in range(i + 1, len(genes)):
            gene2 = genes[j]
            gene_pair = (gene1, gene2)
            covers = [mut_list_dic[gene1], mut_list_dic[gene2]]
            start_time = time()
            ws_me_pv, ws_me_ex_cover_sizes_dic = compute_me_co_pv_ws(
                covers,
                ws_k_cover_dic,
                max_pair_num=10**5,
                ws_ex_cover_sizes_dic=ws_me_ex_cover_sizes_dic,
            )
            ws_co_pv, ws_co_ex_cover_sizes_dic = compute_me_co_pv_ws(
                covers,
                ws_k_cover_dic,
                max_pair_num=10**5,
                ws_ex_cover_sizes_dic=ws_co_ex_cover_sizes_dic,
                me_co="co",
            )
            ws_me_pv_dic[gene_pair] = ws_me_pv  # WeSME pvalue
            ws_co_pv_dic[gene_pair] = ws_co_pv  # WeSCO pvalue
            jaccard_dic[gene_pair] = compute_jaccard(mut_list_dic[gene1], mut_list_dic[gene2])
            fisher_co_pv_dic[gene_pair] = compute_co_pv_hypergeom(
                mut_list_dic[gene1], mut_list_dic[gene2], nsamples
            )

    ws_me_qv = multipletests(list(ws_me_pv_dic.values()), method="fdr_bh")[1]
    ws_co_qv = multipletests(list(ws_co_pv_dic.values()), method="fdr_bh")[1]
    df = pd.DataFrame(
        {
            "gene_a": [g[0] for g in ws_me_pv_dic.keys()],
            "gene_b": [g[1] for g in ws_me_pv_dic.keys()],
            "ws_me_pv": list(ws_me_pv_dic.values()),
            "ws_me_qv": ws_me_qv,
            "ws_co_pv": list(ws_co_pv_dic.values()),
            "ws_co_qv": ws_co_qv,
            "jaccard": list(jaccard_dic.values()),
            "fisher_co_pv": list(fisher_co_pv_dic.values()),
        }
    )
    fout = os.path.join(dout, "wext_pairwise_pvalues.csv")  # Define output file path
    df.to_csv(fout, index=False)  # Write DataFrame to CSV file


def run_wesme(cnt_mtx_df, dout, top_k):
    wesme_dout = os.path.join(dout, "wesme_out")
    if not os.path.exists(wesme_dout):
        os.makedirs(wesme_dout)
    mut_fn = convert_cnt_mtx_to_mut_list(cnt_mtx_df, wesme_dout)
    compute_sample_weights(mut_fn, wesme_dout)
    freqs_fn = os.path.join(
        wesme_dout, "sample_mut_freqs.txt"
    )  # produced by compute_sample_weights
    run_weighted_sampling(
        mut_fn, freqs_fn, 100, wesme_dout, False
    )  # TODO  MODIFY RNUM FROM INPUT AND args.all
    compute_pairwise_pvalues(mut_fn, wesme_dout, top_k)


def main():
    parser = get_parser()
    args = parser.parse_args()
    mut_fn = convert_maf_to_mut_list(args.maf_fn, args.dout)
    compute_sample_weights(mut_fn, args.dout)
    freqs_fn = os.path.join(args.dout, "sample_mut_freqs.txt")  # produced by compute_sample_weights
    run_weighted_sampling(mut_fn, freqs_fn, args.rnum, args.dout, args.all)
    compute_pairwise_pvalues(mut_fn, args.dout, args.top_k)


if __name__ == "__main__":
    main()

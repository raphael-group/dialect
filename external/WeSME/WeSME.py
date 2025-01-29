import bisect
import os
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def count_muts(mut_list_dic, samples):
    nsamples = len(samples)
    counts = [0 for i in range(nsamples)]
    for gene in mut_list_dic:
        for i in mut_list_dic[gene]:
            counts[i] += 1
    return dict(zip(*[samples, counts]))


def compute_sample_freq(count_dic):
    total = float(sum(count_dic.values()))
    return {sample: count_dic[sample] / total for sample in count_dic}


def comp_all_ks(mut_dic):
    return {len(x) for x in mut_dic.values()}


def read_dic(filename):
    lines = open(filename).readlines()
    data_dic = {}
    for l in lines:
        tkns = l.split()
        if len(tkns) > 2:
            sys.stderr.write("more than two columns")
            return {}
        data_dic[tkns[0]] = tkns[1]
    return data_dic


def read_mut_list(filename, sep=",", samples=False, genes=True):
    if genes is True:
        temp_dic = read_dic(filename)
        rel_dic_list = {}
        if samples is True:
            slist = temp_dic["samples"].split(sep)
            del temp_dic["samples"]
        else:
            slist = []
        for g in temp_dic:
            if len(temp_dic) == 0:
                continue
            rel_dic_list[g] = [int(x) for x in temp_dic[g].split(sep)]
        return rel_dic_list, slist
    if genes is False:
        idx_list = []
        lines = open(filename).readlines()
        if samples is True:
            slist = lines[0].strip().split(sep)
        else:
            idx_list.append([int(x) for x in lines[0].strip().split(sep)])
            slist = []
        for l in lines[1:]:
            idx_list.append([int(x) for x in l.strip().split(sep)])
        return idx_list, slist
    return None


def read_mut_matrix(filename, mw=3, mutsig_file=None):
    lines = open(filename).readlines()
    genes = []
    samples = lines[0].split()[1:]
    data_dic = {}

    if mutsig_file is not None:
        mutsig = pd.read_table(mutsig_file, sep=" ", index_col=0).to_dict()["mutsig_score"]
        mutsig_data_dic = {}
        for g in data_dic:
            mutsig_data_dic[g] = [(mutsig[g] + 1) * x for x in data_dic[g]]
        data_dic = mutsig_data_dic

    elif mw > 0:
        weight_dic = {"N": 0, "A": 1, "D": 1, "M": mw, "AM": mw + 1, "DM": mw + 1}
        for l in lines[1:]:
            tkns = l.split()
            data_dic[tkns[0]] = [float(weight_dic[x]) for x in tkns[1:]]
            genes.append(tkns[0])
    else:
        for l in lines[1:]:
            tkns = l.split()
            data_dic[tkns[0]] = list(tkns[1:])
            genes.append(tkns[0])

    return genes, samples, data_dic


def write_mut_matrix(genes, samples, data_dic, filename) -> None:
    f = open(filename, "w")
    f.write("gene\t{}\n".format("\t".join(samples)))
    for g in genes:
        f.write(f"{g}\t")
        f.write("{}\n".format("\t".join([str(x) for x in data_dic[g]])))
    f.close()


def write_mut_list(rel_dic, filename, samples=None, sep=",") -> None:
    list_dic = {}
    for g in rel_dic:
        covers = mut_ex.misc.get_positives(rel_dic[g])
        if len(covers) == 0:
            continue
        list_dic[g] = sep.join([str(x) for x in covers])

    if samples is None:
        write_dic(list_dic, filename)
    else:
        write_dic(list_dic, filename, "samples\t" + ",".join(samples))


def write_alt_list(list_dic, filename, samples=None, sep=",") -> None:
    for g in list_dic:
        list_dic[g] = sep.join([str(x) for x in list_dic[g]])

    if samples is None:
        write_dic(list_dic, filename)
    else:
        write_dic(list_dic, filename, "samples\t" + ",".join(samples))


def write_dic(any_dic, filename, labels=None) -> None:
    f = open(filename, "w")
    if labels is not None:
        f.write(f"{labels}\n")
    for x in any_dic:
        f.write(f"{x!s}\t{any_dic[x]!s}\n")
    f.close()


def create_type_idx(samples, types, sample_type_dic):
    type_idx_dic = {}
    for ty in types:
        type_idx_dic[ty] = filter(lambda i: sample_type_dic[samples[i]] == ty, range(len(samples)))

    return type_idx_dic


def compute_jaccard(cover1, cover2):
    union_cov = set(cover1).union(cover2)
    cross_cov = set(cover1).intersection(cover2)
    return len(cross_cov) / float(len(union_cov))


def compute_me_pv_hypergeom(cover1, cover2, nsamples):
    param = (len(set(cover1).intersection(cover2)), nsamples, len(cover1), len(cover2))
    return ss.hypergeom.cdf(*param)


def compute_co_pv_hypergeom(cover1, cover2, nsamples):
    param = (
        len(cover2) - len(set(cover1).intersection(cover2)),
        nsamples,
        nsamples - len(cover1),
        len(cover2),
    )
    return ss.hypergeom.cdf(*param)


def compute_hypergeom(cover1, cover2, nsamples):
    param = (len(set(cover1).intersection(cover2)), nsamples, len(cover1), len(cover2))
    return ss.hypergeom.pmf(*param)


def compute_ex_cover(covers):
    ex_cov = set()
    union_cov = set()
    cross_cov = set()
    for cov in covers:
        ex_cov = ex_cov.symmetric_difference(set(cov).difference(cross_cov))
        cross_cov = cross_cov.union(union_cov.intersection(cov))
        union_cov = union_cov.union(cov)
    return ex_cov


def compute_rank(cover_size, ws_cover_sizes, me_co="me", ordered=False):
    if me_co == "me":
        if ordered:
            less = bisect.bisect_left(ws_cover_sizes, cover_size)
        else:
            less = len(filter(lambda c: c < cover_size, ws_cover_sizes))
        rank = len(ws_cover_sizes) - less
    elif me_co == "co":
        if ordered:
            lesseq = bisect.bisect_right(ws_cover_sizes, cover_size)
        else:
            lesseq = len(filter(lambda c: c <= cover_size, ws_cover_sizes))
        rank = lesseq
    return rank


def choose_random_tuples(ws_num, tsize, tnum, slack=0.3, max_attempts=100):
    rtuples = set()
    attempts = 0

    while len(rtuples) < tnum and attempts < max_attempts:
        needed = tnum - len(rtuples)
        cur_num = min(int(needed * (1 + slack)), ws_num)
        if cur_num == 0:
            break

        ris = np.random.choice(ws_num, size=(tsize, cur_num), replace=True)
        new_tuples = [tuple(row) for row in ris.T if len(set(row)) == tsize]
        rtuples.update(new_tuples)
        attempts += 1

    return list(rtuples)[:tnum]


def compute_ws_cover_sizes(cover_sizes, ws_k_cover_dic, tuple_num, ws_ex_cover_sizes_dic):
    cover_sizes_key = gen_key(cover_sizes)
    if (
        cover_sizes_key not in ws_ex_cover_sizes_dic
        or len(ws_ex_cover_sizes_dic[cover_sizes_key]) < tuple_num
    ):
        ws_num = len(ws_k_cover_dic[cover_sizes[0]])
        tsize = len(cover_sizes)
        random_indices = choose_random_tuples(
            ws_num,
            len(cover_sizes),
            tuple_num,
        )
        ws_k_covers = [
            ws_k_cover_dic[k] for k in cover_sizes_key
        ]
        # random_covers = [
        # [ws_k_covers[i][rtuple[i]] for i in range(tsize)] for rtuple in random_indices
        # ]
        random_covers = [
            [
                ws_k_covers[i][rtuple[i]]
                for i in range(tsize)
                if i < len(ws_k_covers) and rtuple[i] < len(ws_k_covers[i])
            ]
            for rtuple in random_indices
        ]
        ws_ex_cover_sizes = [
            len(compute_ex_cover(random_covers[i])) for i in range(len(random_covers))
        ]
        ws_ex_cover_sizes.sort()
        ws_ex_cover_sizes_dic[cover_sizes_key] = ws_ex_cover_sizes
    else:
        ws_ex_cover_sizes = ws_ex_cover_sizes_dic[cover_sizes_key]

    return ws_ex_cover_sizes, ws_ex_cover_sizes_dic


def compute_me_co_pv_ws(
    covers,
    ws_k_cover_dic,
    me_co="me",
    min_rank=100,
    init_pair_num=10**3,
    max_pair_num=10**6,
    ws_ex_cover_sizes_dic=None,
    max_attempts=10,
):
    if ws_ex_cover_sizes_dic is None:
        ws_ex_cover_sizes_dic = {}
    cover_sizes = [len(c) for c in covers]
    ex_cover_size = len(compute_ex_cover(covers))
    ws_cover_sizes, ws_ex_cover_sizes_dic = compute_ws_cover_sizes(
        cover_sizes,
        ws_k_cover_dic,
        init_pair_num,
        ws_ex_cover_sizes_dic,
    )

    attempts = 0
    ws_rank, pair_num = 0, 0
    while ws_rank < min_rank and pair_num < max_pair_num and attempts < max_attempts:
        ws_cover_sizes, ws_ex_cover_sizes_dic = compute_ws_cover_sizes(
            cover_sizes,
            ws_k_cover_dic,
            pair_num * 10,
            ws_ex_cover_sizes_dic,
        )
        ws_rank = compute_rank(ex_cover_size, ws_cover_sizes, me_co, ordered=True)
        pair_num = len(ws_cover_sizes)
        attempts += 1

    ws_pv = ws_rank / float(pair_num)
    return ws_pv, ws_ex_cover_sizes_dic


def bipartite_double_edge_swap(G, genes, samples, nswap=1, max_tries=1e75):
    if G.is_directed():
        msg = "double_edge_swap() not defined for directed graphs."
        raise nx.NetworkXError(msg)
    if nswap > max_tries:
        msg = "Number of swaps > number of tries allowed."
        raise nx.NetworkXError(msg)
    if len(G) < 4:
        msg = "Graph has less than four nodes."
        raise nx.NetworkXError(msg)
    n = 0
    swapcount = 0

    gkeys, gdegrees = zip(*G.degree(genes).items())
    gcdf = nx.utils.cumulative_distribution(gdegrees)

    pkeys, pdegrees = zip(*G.degree(samples).items())
    pcdf = nx.utils.cumulative_distribution(pdegrees)

    while swapcount < nswap:
        gi = nx.utils.discrete_sequence(1, cdistribution=gcdf)
        pi = nx.utils.discrete_sequence(1, cdistribution=pcdf)

        gene1 = gkeys[gi[0]]
        sample1 = pkeys[pi[0]]

        sample2 = random.choice(list(G[gene1]))
        gene2 = random.choice(list(G[sample1]))

        if (gene1 not in G[sample1]) and (gene2 not in G[sample2]):
            G.add_edge(gene1, sample1)
            G.add_edge(gene2, sample2)

            G.remove_edge(gene1, sample2)
            G.remove_edge(gene2, sample1)
            swapcount += 1
        if n >= max_tries:
            e = (
                f"Maximum number of swap attempts ({n}) exceeded "
                + f"before desired swaps achieved ({nswap})."
            )
            raise nx.NetworkXAlgorithmError(e)
        n += 1
        if n % 10000 == 0:
            pass
    return G


def permute_mut_graph(G, genes, samples, Q=100):
    H = G.copy()
    bipartite_double_edge_swap(H, genes, samples, nswap=Q * len(G.edges()))
    return H


def construct_mut_graph_per_type(mut_dic, cancers, type_idx_dic):
    mut_graphs = {}
    for cancer in cancers:
        mut_graphs[cancer] = nx.Graph()
    for gene in mut_dic:
        mut_sam_idxs = misc.get_positives(mut_dic[gene])
        for cancer in cancers:
            edges = [(gene, s) for s in set(type_idx_dic[cancer]).intersection(mut_sam_idxs)]
            mut_graphs[cancer].add_edges_from(edges)
    return mut_graphs


def construct_mut_graph_per_type_from_list(mut_list_dic, cancers, type_idx_dic):
    mut_graphs = {}
    for cancer in cancers:
        mut_graphs[cancer] = nx.Graph()
    for gene in mut_list_dic:
        mut_sam_idxs = mut_list_dic[gene]
        for cancer in cancers:
            edges = [(gene, s) for s in set(type_idx_dic[cancer]).intersection(mut_sam_idxs)]
            mut_graphs[cancer].add_edges_from(edges)
    return mut_graphs


def construct_mut_dic_from_graphs(graphs, genes, nsamples):
    mut_dic = {}
    for gene in genes:
        mut_dic[gene] = [0 for i in range(nsamples)]
        for cancer in graphs:
            if gene not in graphs[cancer]:
                continue
            for i in graphs[cancer].neighbors(gene):
                mut_dic[gene][i] = 1
    return mut_dic


def get_positives(in_list):
    return filter(lambda i: in_list[i] > 0, range(len(in_list)))


def gen_key(alist):
    alist = list(alist)
    alist.sort()
    return tuple(alist)


def gen_uniq_key(alist):
    alist = list(set(alist))
    alist.sort()
    return tuple(alist)


def convert_cnt_mtx_to_mut_list(cnt_mtx_df, dout):
    genes = cnt_mtx_df.index.tolist()
    samples = cnt_mtx_df.columns.tolist()
    idx_list = [np.flatnonzero(cnt_mtx_df.loc[g].values) for g in genes]
    fout = os.path.join(dout, "mut_list.txt")
    fout_file = open(fout, "w")
    fout_file.write("samples\t{}\n".format(",".join(samples)))
    for g, idxs in zip(genes, idx_list):
        fout_file.write("{}\t{}\n".format(g, ",".join([str(i) for i in idxs])))
    fout_file.close()
    return fout


def convert_maf_to_mut_list(maf_fn, dout):
    df = pd.read_csv(maf_fn, sep="\t")
    df = df[df["Consequence"].isin(["missense_variant", "nonsense_variant"])]
    df["Hugo_Symbol"] = df["Hugo_Symbol"] + "_" + df["Consequence"].str[0].str.upper()
    cnt_mtx = df.pivot_table(
        index="Hugo_Symbol",
        columns="Tumor_Sample_Barcode",
        aggfunc="size",
        fill_value=0,
    ).T
    return convert_cnt_mtx_to_mut_list(cnt_mtx, dout)


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def compute_sample_weights(mut_fn, dout) -> None:
    mut_list_dic, samples = read_mut_list(mut_fn, samples=True)
    mut_count = count_muts(mut_list_dic, samples)
    mut_freq = compute_sample_freq(mut_count)
    write_dic(mut_freq, os.path.join(dout, "sample_mut_freqs.txt"))


def run_weighted_sampling(mut_fn, freq_fn, rnum, dout, use_all=False) -> None:
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
        for _i in range(rnum):
            wsamples = np.random.choice(range(nsamples), k, p=freqs, replace=False)
            ws_file.write("{}\n".format(",".join([str(idx) for idx in wsamples])))
        ws_file.close()


def compute_pairwise_pvalues(mut_fn, dout, gene_pairs):
    mut_list_dic, samples = read_mut_list(mut_fn, samples=True)
    {gene: len(mut_list_dic[gene]) for gene in mut_list_dic}
    nsamples = len(samples)

    cur_wrs_dir = os.path.join(dout, "weighted_sampling")
    kfiles = os.listdir(cur_wrs_dir)
    ws_k_cover_dic = {}
    for kf in kfiles:
        k = int(kf.split(".")[0].split("_")[1])
        ws_k_cover_dic[k], samples = read_mut_list(os.path.join(cur_wrs_dir, kf), genes=False)

    ws_me_pv_dic, ws_me_ex_cover_sizes_dic = {}, {}
    ws_co_pv_dic, ws_co_ex_cover_sizes_dic = {}, {}
    fisher_co_pv_dic, jaccard_dic = {}, {}
    for gene_pair in tqdm(gene_pairs):
        gene_a, gene_b = gene_pair
        covers = [mut_list_dic[gene_a], mut_list_dic[gene_b]]
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
        ws_me_pv_dic[gene_pair] = ws_me_pv
        ws_co_pv_dic[gene_pair] = ws_co_pv
        jaccard_dic[gene_pair] = compute_jaccard(mut_list_dic[gene_a], mut_list_dic[gene_b])
        fisher_co_pv_dic[gene_pair] = compute_co_pv_hypergeom(
            mut_list_dic[gene_a],
            mut_list_dic[gene_b],
            nsamples,
        )

    ws_me_qv = multipletests(list(ws_me_pv_dic.values()), method="fdr_bh")[1]
    ws_co_qv = multipletests(list(ws_co_pv_dic.values()), method="fdr_bh")[1]
    return pd.DataFrame(
        {
            "Gene A": [g[0] for g in ws_me_pv_dic],
            "Gene B": [g[1] for g in ws_me_pv_dic],
            "WeSME P-Val": list(ws_me_pv_dic.values()),
            "WeSCO P-Val": list(ws_co_pv_dic.values()),
            "WeSME Q-Val": ws_me_qv,
            "WeSCO Q-Val": ws_co_qv,
        },
    )

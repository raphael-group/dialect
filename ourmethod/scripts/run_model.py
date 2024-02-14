import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import combinations
from argparse import ArgumentParser

# Set current directory as working directory to import models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

# Constants
TOP_K = 50
MAX_ITER = 10
TOL = 1e-5
ZERO_MUT_CASE_EPS = 0  # when c_i = 0, use this for p(c_i - 1)

# Helper Functions
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-bmr_d", required=True)
    parser.add_argument("-mtx_d", required=True)
    parser.add_argument("-dout", required=True)
    parser.add_argument("-c", required=True)

    mut_type_group = parser.add_mutually_exclusive_group(required=False)
    mut_type_group.add_argument("-mis", action="store_true")
    mut_type_group.add_argument("-non", action="store_true")
    mut_type_group.add_argument("-ind", action="store_true")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("-single", action="store_true")
    model_group.add_argument("-pair", action="store_true")
    return parser


def em_single(c, pi, prob):
    lls = []
    for t in range(MAX_ITER):
        bg = np.array([prob[c_i] if c_i < len(prob) else 0 for c_i in c])
        dv = np.array([prob[c_i - 1] if 0 < c_i < len(prob) else 0 for c_i in c])
        total_prob = (1 - pi) * bg + pi * dv
        z_0 = (1 - pi) * bg / total_prob
        z_1 = pi * dv / total_prob
        ll_prev = log_likelihood(pi, bg, dv)
        pi = np.mean(z_1)
        ll = log_likelihood(pi, bg, dv)
        lls.append(ll)
        if np.abs(ll - ll_prev) < TOL:
            break
    return pi, bg, dv, lls


def log_likelihood(pi, bg, dv):
    return np.sum(np.log((1 - pi) * bg + pi * dv))


def singleton_em(cnt_df, bmr_df):
    # pi initializations
    min_pi, max_pi, num_pis = 1e-6, 0.2, 5
    pis = np.array([np.array([1 - p, p]) for p in np.linspace(min_pi, max_pi, num_pis)])

    # CBaSE setup
    pmf_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    pmf_dict = {  # to drop nans caused by empty cols from df creation
        key: [x for x in pmf_dict[key] if not np.isnan(x)] for key in pmf_dict
    }
    genes = cnt_df.columns  # less genes in cnt_df b/c genes w/o mutations dropped
    param_estimates = {}
    for gene in tqdm(genes):
        gene_pmf = pmf_dict[gene]
        gene_counts = cnt_df[gene].values
        results = []
        for pi in pis:
            pi, bg, dv, lls = em_single(gene_counts, pi[1], gene_pmf)
            ll = log_likelihood(pi, bg, dv)
            llr = 2 * (ll - log_likelihood(0, bg, dv))
            results.append((pi, ll, llr))
            # mm = SMM(gene_counts, pi, gene_pmf)
            # pi, log_likelihood, log_likelihood_ratio = mm.fit()
            # results.append((pi, log_likelihood, log_likelihood_ratio))
        max_index = np.argmax([x[1] for x in results])
        max_result = results[max_index]
        param_estimates[gene] = max_result

    # MutSigCV
    # genes = cnt_df.columns  # less genes in cnt_df b/c genes w/o mutations dropped
    # param_estimates = {}
    # for gene in tqdm(genes):
    #     sample_pmfs = bmr_df[gene]
    #     gene_counts = {
    #         sample: cnt for sample, cnt in zip(cnt_df.index, cnt_df[gene].values)
    #     }
    #     pmf_c = np.array([sample_pmfs[s][c] for s, c in gene_counts.items()])
    #     pmf_c_min_one = np.array(
    #         [
    #             sample_pmfs[s][c - 1] if c > 0 else ZERO_MUT_CASE_EPS
    #             for s, c in gene_counts.items()
    #         ]
    #     )
    #     results = []
    #     for pi in pis:
    #         mm = SMM(pmf_c, pmf_c_min_one, pi, 2, TOL, MAX_ITER)
    #         pi, log_likelihood, log_likelihood_ratio = mm.fit()
    #         results.append((pi, log_likelihood, log_likelihood_ratio))
    #     max_index = np.argmax([x[1] for x in results])
    #     max_result = results[max_index]
    #     param_estimates[gene] = max_result

    return param_estimates


def get_pair_pis(init_count, pi_1, pi_2, min_pi_00=0.1):
    pi_00 = np.linspace(min_pi_00, 1, init_count)
    pi_init_random = np.array([(1 - p) / 3 for p in pi_00]).reshape(-1, 1) * np.array(
        [0, 1, 1, 1]
    ) + np.column_stack((pi_00, np.zeros((init_count, 3))))
    pi_init_custom = np.array(
        [
            [
                1 - np.max([pi_1, pi_2]),
                0 if pi_2 < pi_1 else pi_2 - pi_1,
                0 if pi_1 < pi_2 else pi_1 - pi_2,
                np.min([pi_1, pi_2]),
            ],  # co-occurrence initialization
            [1 - (pi_1 + pi_2), pi_2, pi_1, 0],  # mutual exclusivity initialization
            [
                (1 - pi_1) * (1 - pi_2),
                (1 - pi_1) * pi_2,
                pi_1 * (1 - pi_2),
                pi_1 * pi_2,
            ],  # independence initialization
        ]
    )
    return np.vstack((pi_init_random, pi_init_custom))


def em_pair(iter, tol, c, c_p, pi_00, pi_01, pi_10, pi_11, prob, prob_p):
    #bg = np.array([prob[c_i] for c_i in c])
    #bg_p = np.array([prob_p[c_i] for c_i in c_p])
    #dv = np.array([prob[c_i - 1] if c_i > 0 else 0 for c_i in c])
    #dv_p = np.array([prob_p[c_i - 1] if c_i > 0 else 0 for c_i in c_p])

    bg = np.array([prob[c_i] if c_i < len(prob) else 0 for c_i in c])
    bg_p = np.array([prob_p[c_i] if c_i < len(prob_p) else 0 for c_i in c_p])
    dv = np.array([prob[c_i - 1] if 0 < c_i < len(prob) else 0 for c_i in c])
    dv_p = np.array([prob_p[c_i - 1] if 0 < c_i < len(prob_p) else 0 for c_i in c_p])


    for _ in range(iter):
        total_prob = (
            pi_00 * bg * bg_p
            + pi_01 * bg * dv_p
            + pi_10 * dv * bg_p
            + pi_11 * dv * dv_p
        )
        z_00 = pi_00 * bg * bg_p / total_prob
        z_01 = pi_01 * bg * dv_p / total_prob
        z_10 = pi_10 * dv * bg_p / total_prob
        z_11 = pi_11 * dv * dv_p / total_prob
        ll_prev = log_likelihood_pair(pi_00, pi_01, pi_10, pi_11, bg, bg_p, dv, dv_p)
        pi_00 = np.mean(z_00, axis=0)
        pi_01 = np.mean(z_01, axis=0)
        pi_10 = np.mean(z_10, axis=0)
        pi_11 = np.mean(z_11, axis=0)
        ll = log_likelihood_pair(pi_00, pi_01, pi_10, pi_11, bg, bg_p, dv, dv_p)
        if np.abs(ll - ll_prev) < tol:
           break
    return pi_00, pi_01, pi_10, pi_11, bg, bg_p, dv, dv_p


def log_likelihood_pair(pi_00, pi_01, pi_10, pi_11, bg, bg_p, dv, dv_p):
    likelihoods = (
        pi_00 * bg * bg_p + pi_01 * bg * dv_p + pi_10 * dv * bg_p + pi_11 * dv * dv_p
    )
    log_likelihood = np.sum(np.log(likelihoods))
    return log_likelihood


def process_gene_pair(
    gene_1,
    gene_2,
    singleton_dict,
    cnt_df,
    pmf_dict,
    iterations,
    threshold,
):
    pis = get_pair_pis(2, singleton_dict[gene_1], singleton_dict[gene_2])
    g1c = cnt_df[gene_1].values
    g2c = cnt_df[gene_2].values
    prob = pmf_dict[gene_1]
    prob_p = pmf_dict[gene_2]
    results = []
    
    # bg = np.array([prob[c_i] for c_i in g1c])
    # bg_p = np.array([prob_p[c_i] for c_i in g2c])
    # dv = np.array([prob[c_i - 1] if c_i > 0 else 0 for c_i in g1c])
    # dv_p = np.array([prob_p[c_i - 1] if c_i > 0 else 0 for c_i in g2c])
   
    # def neg_log_likelihood(pi):
    #     pi_00, pi_01, pi_10, pi_11 = pi
    #     likelihoods = (
    #         pi_00 * bg * bg_p + pi_01 * bg * dv_p + pi_10 * dv * bg_p + pi_11 * dv * dv_p
    #     )
    #     log_likelihood = np.sum(np.log(likelihoods))
    #     return -log_likelihood 

    # pi = np.array([0.25, 0.25, 0.25, 0.25])
    # alpha = 1e-300  # necessary to define nonzero bounds
    # cons = [{"type": "eq", "fun": lambda x: 1 - np.sum(x)}]
    # bnds = [(alpha, 1 - alpha)] * 4
    # output = minimize(
    #     neg_log_likelihood,
    #     pi,
    #     method="SLSQP",
    #     bounds=bnds,
    #     constraints=cons,
    # )
    # print(gene_1, gene_2)
    # print(output["x"])
    # quit()

    for pi in pis:
        pi_00, pi_01, pi_10, pi_11, bg, bg_p, dv, dv_p = em_pair(
            iterations, threshold, g1c, g2c, pi[0], pi[1], pi[2], pi[3], prob, prob_p
        )
        #bg = np.array([prob[c_i] for c_i in g1c])
        #bg_p = np.array([prob_p[c_i] for c_i in g2c])
        #dv = np.array([prob[c_i - 1] if c_i > 0 else 0 for c_i in g1c])
        #dv_p = np.array([prob_p[c_i - 1] if c_i > 0 else 0 for c_i in g2c])

        bg = np.array([prob[c_i] if c_i < len(prob) else 0 for c_i in g1c])
        bg_p = np.array([prob_p[c_i] if c_i < len(prob_p) else 0 for c_i in g2c])
        dv = np.array([prob[c_i - 1] if 0 < c_i < len(prob) else 0 for c_i in g1c])
        dv_p = np.array([prob_p[c_i - 1] if 0 < c_i < len(prob_p) else 0 for c_i in g2c])

        ll = log_likelihood_pair(pi_00, pi_01, pi_10, pi_11, bg, bg_p, dv, dv_p)
        pi_null = pis[-1]
        ll_null = log_likelihood_pair(
            pi_null[0], pi_null[1], pi_null[2], pi_null[3], bg, bg_p, dv, dv_p
        )
        llr = 2 * (ll - ll_null)
        results.append(([pi_00, pi_01, pi_10, pi_11], ll, llr))
    max_index = np.argmax([x[-1] for x in results]) 
    max_result = results[max_index]
    return max_result


# def process_gene_pair_wrapper(args):
#     gene_1, gene_2, singleton_dict, cnt_df, pmf_dict = args
#     return gene_1, gene_2, process_gene_pair(gene_1, gene_2, singleton_dict, cnt_df, pmf_dict, 100, 1e-100)

# def bivariate_em(cnt_df, mis_bmr_df, non_bmr_df, pairs, singleton_dict):
#     mis_pmf_dict = mis_bmr_df.T.to_dict(orient="list")
#     mis_pmf_dict = {key: [x for x in mis_pmf_dict[key] if not np.isnan(x)] for key in mis_pmf_dict}
#     non_pmf_dict = non_bmr_df.T.to_dict(orient="list")
#     non_pmf_dict = {key: [x for x in non_pmf_dict[key] if not np.isnan(x)] for key in non_pmf_dict}
#     pmf_dict = {**mis_pmf_dict, **non_pmf_dict}
#     param_estimates = {}
#     pool_args = [(gene_1, gene_2, singleton_dict, cnt_df, pmf_dict) for gene_1, gene_2 in pairs]
#     num_processes = 10
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         results = list(tqdm(pool.imap(process_gene_pair_wrapper, pool_args), total=len(pairs)))
#     for gene_1, gene_2, result in results:
#         param_estimates[(gene_1, gene_2)] = result
#     return param_estimates


def bivariate_em(cnt_df, mis_bmr_df, non_bmr_df, pairs, singleton_dict):
    mis_pmf_dict = mis_bmr_df.T.to_dict(orient="list")
    mis_pmf_dict = {  # to drop nans caused by empty cols from df creation
        key: [x for x in mis_pmf_dict[key] if not np.isnan(x)] for key in mis_pmf_dict
    }
    non_pmf_dict = non_bmr_df.T.to_dict(orient="list")
    non_pmf_dict = {  # to drop nans caused by empty cols from df creation
        key: [x for x in non_pmf_dict[key] if not np.isnan(x)] for key in non_pmf_dict
    }
    pmf_dict = {**mis_pmf_dict, **non_pmf_dict}
    param_estimates = {}
    for gene_1, gene_2 in tqdm(pairs):
        pi, ll, llr = process_gene_pair(
            gene_1, gene_2, singleton_dict, cnt_df, pmf_dict, MAX_ITER, TOL
        )
        # for i in range (4):
        #     if pi[i] == 0: pi[i] = alpha
        param_estimates[(gene_1, gene_2)] = (pi, ll, llr)
        #max_result = [[0, 0, 0, 0], 0, 0]
        #pi_hat, it, eps = max_result[0], 10, 1e-5
        #g1c_bin = [1 if x > 0 else 0 for x in cnt_df[gene_1].values]
        #g2c_bin = [1 if x > 0 else 0 for x in cnt_df[gene_2].values]
        #cont_table_bool = [bool(x) for x in pd.crosstab(g1c_bin, g2c_bin).to_numpy().flatten()]
        #pi_hat_bool = [bool(x) for x in pi_hat]
        #while any(x ^ y for x, y in zip(cont_table_bool, pi_hat_bool)):
        #    if it > 3000: break
        #    max_result = process_gene_pair(
        #        gene_1, gene_2, singleton_dict, cnt_df, pmf_dict, it, eps
        #    )
        #    param_estimates[(gene_1, gene_2)] = max_result
        #    pi_hat, it, eps = max_result[0], 3 * it, eps / 1e10
        #    pi_hat_bool = [bool(x) for x in pi_hat]
    return param_estimates


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.single:
        if not (args.mis or args.non or args.ind):
            parser.error("One of '-mis', '-non', or '-ind' must also be specified.")
        mut_type = "mis" if args.mis else "non" if args.non else "ind"
        cnt_fn = os.path.join(args.mtx_d, f"{args.c}_{mut_type}_cnt_mtx.csv")
        cnt_df = pd.read_csv(cnt_fn, index_col=0)
        cnt_df = cnt_df.astype(int)
        bmr_fn = os.path.join(args.bmr_d, f"{args.c}_{mut_type}_bmr.csv")
        bmr_df = pd.read_csv(bmr_fn, index_col=0)

        ############# MutSigCV ##################
        # bmr_dict = {}
        # for _, row in bmr_df.iterrows():
        #     gene, sample, pmfs = row["gene"], row["sample"], row["pmfs"]
        #     pmfs_list = list(map(float, pmfs[1:-1].split(",")))
        #     if gene not in bmr_dict:
        #         bmr_dict[gene] = {}
        #     bmr_dict[gene][sample] = pmfs_list
        # results = singleton_em(cnt_df, bmr_dict)  # keys: genes, vals: (pi, ll, llr)
        #########################################

        # CBaSE
        results = singleton_em(cnt_df, bmr_df)  # keys: genes, vals: (pi, ll, llr)

        genes, pis = results.keys(), [x[0] for x in results.values()]
        lls, llrs = [x[1] for x in results.values()], [x[2] for x in results.values()]
        df = pd.DataFrame({"gene": genes, "pi": pis, "ll": lls, "llr": llrs})
        single_fout = os.path.join(args.dout, f"{args.c}_{mut_type}_single_pi_vals.csv")
        df.to_csv(single_fout, sep="\t")
    else:  # paired gene model
        # read mis dfs and rename columns
        mis_cnt_fn = os.path.join(args.mtx_d, f"{args.c}_mis_cnt_mtx.csv")
        mis_cnt_df = pd.read_csv(mis_cnt_fn, index_col=0)
        mis_bmr_fn = os.path.join(args.bmr_d, f"{args.c}_mis_bmr.csv")
        mis_bmr_df = pd.read_csv(mis_bmr_fn, index_col=0)
        mis_cnt_df.columns = [f"{x}_M" for x in mis_cnt_df.columns]
        mis_bmr_df.index = [f"{x}_M" for x in mis_bmr_df.index]

        # read non dfs and rename columns
        non_cnt_fn = os.path.join(args.mtx_d, f"{args.c}_non_cnt_mtx.csv")
        non_cnt_df = pd.read_csv(non_cnt_fn, index_col=0)
        non_bmr_fn = os.path.join(args.bmr_d, f"{args.c}_non_bmr.csv")
        non_bmr_df = pd.read_csv(non_bmr_fn, index_col=0)
        non_cnt_df.columns = [f"{x}_N" for x in non_cnt_df.columns]
        non_bmr_df.index = [f"{x}_N" for x in non_bmr_df.index]

        # concatenate mis/non cnt dfs, fill nans, and save
        cnt_df = pd.concat([mis_cnt_df, non_cnt_df], axis=1)
        cnt_df.fillna(0, inplace=True)
        cnt_df = cnt_df.astype(int)
        cnt_df.to_csv(os.path.join(args.dout, f"{args.c}_joint_cnt_mtx.csv"))

        # get singleton pi vals
        mis_singleton_fn = os.path.join(args.dout, f"{args.c}_mis_single_pi_vals.csv")
        non_singleton_fn = os.path.join(args.dout, f"{args.c}_non_single_pi_vals.csv")
        mis_singleton_df = pd.read_csv(mis_singleton_fn, sep="\t", index_col=0)
        non_singleton_df = pd.read_csv(non_singleton_fn, sep="\t", index_col=0)
        mis_singleton_df["gene"] = [f"{x}_M" for x in mis_singleton_df["gene"]]
        non_singleton_df["gene"] = [f"{x}_N" for x in non_singleton_df["gene"]]
        singleton_df = pd.concat([mis_singleton_df, non_singleton_df], axis=0)
        singleton_dict = dict(zip(singleton_df["gene"], singleton_df["pi"]))

        # create pairs and run bivariate em
        top_k_genes = cnt_df.sum(axis=0).sort_values(ascending=False).index[:TOP_K]
        #top_k_genes = ['RYR2_M', 'DNAH10_M']
        gene_pairs = list(combinations(top_k_genes, 2))
        results = bivariate_em(
            cnt_df, mis_bmr_df, non_bmr_df, gene_pairs, singleton_dict
        )

        # save results
        pair_gene_fout = os.path.join(args.dout, f"{args.c}_pair_pi_vals.csv")
        g1, g2 = [x[0] for x in results.keys()], [x[1] for x in results.keys()]
        pi_00 = [x[0][0] for x in results.values()]
        pi_10 = [x[0][1] for x in results.values()]
        pi_01 = [x[0][2] for x in results.values()]
        pi_11 = [x[0][3] for x in results.values()]
        lls, llrs = [x[1] for x in results.values()], [x[2] for x in results.values()]
        df = pd.DataFrame(
            {
                "gene_1": g1,
                "gene_2": g2,
                "pi_00": pi_00,
                "pi_10": pi_10,
                "pi_01": pi_01,
                "pi_11": pi_11,
                "ll": lls,
                "llr": llrs,
            }
        )
        df.to_csv(pair_gene_fout, sep="\t")


if __name__ == "__main__":
    main()

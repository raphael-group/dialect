#!/usr/bin/env python
import os
import sys
import subprocess
import importlib.resources

from tqdm import tqdm
from itertools import combinations
from argparse import ArgumentParser

from utils import *
from cbase_utils import *

# from wesme.core import run_wesme


def dialect_singleton(somatic_mutations, bmr_pmfs):
    pi_inits = generate_pi_inits(
        init_count=10, min_pi=0, max_pi=0.5
    )  # Generate initial guesses for 'pi'
    # Construct background and driver mixtures from BMR PMFs and somatic mutation indices
    background_mixture = np.array(
        [bmr_pmfs[c_i] if c_i < len(bmr_pmfs) else 0 for c_i in somatic_mutations]
    )
    driver_mixture = np.array(
        [bmr_pmfs[c_i - 1] if 0 < c_i < len(bmr_pmfs) else 0 for c_i in somatic_mutations]
    )
    results = []
    for pi_init in pi_inits:
        pi = em_single(
            pi_init, background_mixture, driver_mixture
        )  # Run EM algorithm for each initial 'pi'
        null_pi = 0  # Null hypothesis of no driver mutations
        est_log_likelihood = log_likelihood(
            pi, background_mixture, driver_mixture
        )  # Calculate estimated log likelihood
        null_log_likelihood = log_likelihood(
            null_pi, background_mixture, driver_mixture
        )  # Log likelihood of null hypothesis
        log_likelihood_ratio = 2 * (
            est_log_likelihood - null_log_likelihood
        )  # Compute log likelihood ratio
        results.append((pi, est_log_likelihood, log_likelihood_ratio))
    optimal_param_vals = max(results, key=lambda x: x[-1])  # sort by log likelihood ratio
    return optimal_param_vals


def dialect_pairwise(
    gene_a_pi,
    gene_b_pi,
    gene_a_log_likelihood,
    gene_b_log_likelihood,
    gene_a_somatic_mutations,
    gene_b_somatic_mutations,
    gene_a_bmr_pmfs,
    gene_b_bmr_pmfs,
):
    # Generate initial tau values for pairwise model
    tau_inits = generate_pairwise_tau_inits(10, gene_a_pi, gene_b_pi)
    gene_a_background_mixture = np.array(
        [
            gene_a_bmr_pmfs[c_i] if c_i < len(gene_a_bmr_pmfs) else 0
            for c_i in gene_a_somatic_mutations
        ]
    )
    gene_a_driver_mixture = np.array(
        [
            gene_a_bmr_pmfs[c_i - 1] if 0 < c_i < len(gene_a_bmr_pmfs) else 0
            for c_i in gene_a_somatic_mutations
        ]
    )
    gene_b_background_mixture = np.array(
        [
            gene_b_bmr_pmfs[c_i] if c_i < len(gene_b_bmr_pmfs) else 0
            for c_i in gene_b_somatic_mutations
        ]
    )
    gene_b_driver_mixture = np.array(
        [
            gene_b_bmr_pmfs[c_i - 1] if 0 < c_i < len(gene_b_bmr_pmfs) else 0
            for c_i in gene_b_somatic_mutations
        ]
    )
    results = []
    for tau_init in tau_inits:
        tau_no_drivers, tau_driver_b_only, tau_driver_a_only, tau_both_drivers = em_pair(
            tau_init,
            gene_a_background_mixture,
            gene_b_background_mixture,
            gene_a_driver_mixture,
            gene_b_driver_mixture,
        )
        est_log_likelihood = log_likelihood_pair(
            tau_no_drivers,
            tau_driver_b_only,
            tau_driver_a_only,
            tau_both_drivers,
            gene_a_background_mixture,
            gene_b_background_mixture,
            gene_a_driver_mixture,
            gene_b_driver_mixture,
        )
        null_log_likelihood = gene_a_log_likelihood + gene_b_log_likelihood
        log_likelihood_ratio = 2 * (est_log_likelihood - null_log_likelihood)
        results.append(
            (
                [
                    tau_no_drivers,
                    tau_driver_b_only,
                    tau_driver_a_only,
                    tau_both_drivers,
                ],
                est_log_likelihood,
                log_likelihood_ratio,
            )
        )
    max_result = max(
        results, key=lambda x: x[-1]
    )  # select the result with the maximum log likelihood ratio
    log_odds = calculate_log_odds(max_result[0])
    max_result = (*max_result, log_odds)
    return max_result


def run_dialect_singleton_all_genes(cnt_df, bmr_df):
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
    param_estimates = {}
    for gene in tqdm(cnt_df.columns):
        somatic_mutations = cnt_df[gene]
        bmr_pmfs = bmr_dict[gene]
        optimal_params = dialect_singleton(somatic_mutations, bmr_pmfs)
        param_estimates[gene] = optimal_params
    genes, pis = param_estimates.keys(), [x[0] for x in param_estimates.values()]
    lls, llrs = [x[1] for x in param_estimates.values()], [x[2] for x in param_estimates.values()]
    df = pd.DataFrame({"gene": genes, "pi": pis, "ll": lls, "llr": llrs})
    return df


def run_dialect_pairwise_all_pairs(single_df, cnt_df, bmr_df, gene_pairs):
    # convert single_df to dict with gene as key and pi, ll, llr as value
    single_dict = single_df.set_index("gene").T.to_dict()
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
    param_estimates = {}
    for gene_a, gene_b in tqdm(gene_pairs):
        optimal_params = dialect_pairwise(
            single_dict[gene_a]["pi"],
            single_dict[gene_b]["pi"],
            single_dict[gene_a]["ll"],
            single_dict[gene_b]["ll"],
            cnt_df[gene_a],
            cnt_df[gene_b],
            bmr_dict[gene_a],
            bmr_dict[gene_b],
        )
        param_estimates[(gene_a, gene_b)] = optimal_params
    genes, taus = param_estimates.keys(), [x[0] for x in param_estimates.values()]
    lls, llrs, log_odds = (
        [x[1] for x in param_estimates.values()],
        [x[2] for x in param_estimates.values()],
        [x[3] for x in param_estimates.values()],
    )
    gene_a, gene_b = zip(*genes)
    tau_no_drivers, tau_driver_b_only, tau_driver_a_only, tau_both_drivers = zip(*taus)
    df = pd.DataFrame(
        {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "tau_00": tau_no_drivers,
            "tau_01": tau_driver_b_only,
            "tau_10": tau_driver_a_only,
            "tau_11": tau_both_drivers,
            "ll": lls,
            "llr": llrs,
            "log_odds": log_odds,
        }
    )
    return df


def generate_bmr_and_cnt_mtx(maf_fn, dout, method):
    assert os.path.exists(maf_fn), f"File not found: {maf_fn}"
    maf_basename = os.path.basename(maf_fn)
    subtype = os.path.splitext(maf_basename)[0]
    subtype_dout = os.path.join(dout, subtype)
    os.makedirs(subtype_dout, exist_ok=True)

    if method == "cbase":
        subtype_cbase_dout = os.path.join(subtype_dout, "cbase_out")
        os.makedirs(subtype_cbase_dout, exist_ok=True)
        # create cbase vcf file
        vcf_df = convert_maf_to_vcf(maf_fn)
        vcf_fn = os.path.join(subtype_cbase_dout, f"{subtype}_cbase.vcf")
        vcf_df.to_csv(vcf_fn, sep="\t", header=False, index=False)

        # prepare cbase script commands
        cbase_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "bmrs", "cbase")
        cbase_params_script_path = os.path.join(cbase_dir, "cbase_params_v1.2.py")
        cbase_qvals_script_path = os.path.join(cbase_dir, "cbase_qvals_v1.2.py")
        cbase_aux_directory = os.path.join(cbase_dir, "auxiliary")
        cbase_params_cmd = f"python {cbase_params_script_path} {vcf_fn} 1 hg19 3 0 {subtype} {cbase_aux_directory} {subtype_cbase_dout}"
        cbase_qvals_cmd = f"python {cbase_qvals_script_path} {subtype} {subtype_cbase_dout}"

        # run cbase commands
        subprocess.run(cbase_params_cmd, shell=True, check=True)
        subprocess.run(cbase_qvals_cmd, shell=True, check=True)

        cnt_mtx_df = build_cnt_mtx(
            os.path.join(subtype_cbase_dout, f"{subtype}_kept_mutations.csv")
        )
        mis_bmr_fn = os.path.join(subtype_cbase_dout, f"pofmigivens_{subtype}.txt")
        non_bmr_fn = os.path.join(subtype_cbase_dout, f"pofkigivens_{subtype}.txt")
        bmr_df = build_bmr_table(mis_bmr_fn, non_bmr_fn)

        cnt_mtx_df.to_csv(os.path.join(subtype_dout, f"{subtype}_cbase_cnt_mtx.csv"))
        bmr_df.to_csv(os.path.join(subtype_dout, f"{subtype}_cbase_bmr_pmfs.csv"))


def analyze_interactions(cnt_mtx_fn, bmr_fn, dout, top_k):
    assert os.path.exists(cnt_mtx_fn), f"File not found: {cnt_mtx_fn}"
    assert os.path.exists(bmr_fn), f"File not found: {bmr_fn}"
    cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
    bmr_df = pd.read_csv(bmr_fn, index_col=0)

    # run singleton analysis
    singleton_results_df = run_dialect_singleton_all_genes(cnt_mtx_df, bmr_df)
    singleton_results_df.to_csv(os.path.join(dout, "single_genes.csv"))

    # run pairwise analysis
    top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[:top_k]
    gene_pairs = list(combinations(top_k_genes, 2))
    pairwise_results_fn = run_dialect_pairwise_all_pairs(
        singleton_results_df, cnt_mtx_df, bmr_df, gene_pairs
    )
    pairwise_results_fn.to_csv(os.path.join(dout, "pairwise_genes.csv"))


def run_comparison_method(cnt_mtx_fn, dout, top_k, method, feature_level):
    cnt_mtx_df = pd.read_csv(cnt_mtx_fn, index_col=0)
    top_k_genes = cnt_mtx_df.sum(axis=0).sort_values(ascending=False).index[:top_k]
    gene_pairs = list(combinations(top_k_genes, 2))
    if method == "fishers":
        fishers_dir = os.path.join(
            os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "fishers"
        )
        sys.path.append(fishers_dir)
        fishers_module = importlib.import_module("run_fishers")
        me_results_df = fishers_module.run_fishers_all_pairs(
            cnt_mtx_df, gene_pairs, interaction_type="me"
        )
        co_results_df = fishers_module.run_fishers_all_pairs(
            cnt_mtx_df, gene_pairs, interaction_type="co"
        )
        results_df = pd.merge(me_results_df, co_results_df, on=["gene_a", "gene_b"])
        results_df.to_csv(os.path.join(dout, "fishers_results.csv"), index=False)
        sys.path.remove(fishers_dir)
    elif method == "discover":
        discover_dir = os.path.join(
            os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "discover"
        )
        conda_activate_command = "source activate discover"
        discover_script_path = os.path.join(discover_dir, "run_discover.py")
        discover_command = f"python {discover_script_path} {cnt_mtx_fn} {dout} --top_k {top_k} --feature_level {feature_level}"
        full_command = f"{conda_activate_command} && {discover_command}"

        process = subprocess.Popen(
            full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        print(stdout.decode(), stderr.decode())
    elif method == "wesme":
        wesme_dir = os.path.join(os.environ["DIALECT_PROJECT_DIRECTORY"], "comparisons", "wesme")
        sys.path.append(wesme_dir)  # Add the parent directory of wesme
        wesme_module = importlib.import_module("wesme_core")
        wesme_module.run_wesme(cnt_mtx_df, dout, top_k)
        pass


def run_dialect():
    os.environ["DIALECT_PROJECT_DIRECTORY"] = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    parser = get_parser()
    args = parser.parse_args()
    if args.command == "generate":
        generate_bmr_and_cnt_mtx(args.maf_fn, args.dout, args.method)
    elif args.command == "analyze":
        analyze_interactions(args.cnt_mtx_fn, args.bmr_fn, args.dout, args.top_k)
    elif args.command == "compare":
        run_comparison_method(
            args.cnt_mtx_fn, args.dout, args.top_k, args.method, args.feature_level
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    run_dialect()

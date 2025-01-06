import os
import sys

# TODO: find a cleaner way to import wesme code from external
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
external_dir = os.path.join(project_root, "external")
sys.path.append(external_dir)

from WeSME.WeSME import *


def run_wesme_analysis(cnt_df, out, interactions):
    print("Running WeSME analysis...")
    cnt_df = cnt_df.T  # transpose the count matrix
    wesme_dout = os.path.join(out, "WeSME_output")
    os.makedirs(wesme_dout, exist_ok=True)
    mut_fn = convert_cnt_mtx_to_mut_list(cnt_df, wesme_dout)
    compute_sample_weights(mut_fn, wesme_dout)
    freqs_fn = os.path.join(wesme_dout, "sample_mut_freqs.txt")
    run_weighted_sampling(mut_fn, freqs_fn, 100, wesme_dout, False)
    gene_pairs = [(ixn.gene_a.name, ixn.gene_b.name) for ixn in interactions]
    wesme_results_df = compute_pairwise_pvalues(mut_fn, wesme_dout, gene_pairs)
    return wesme_results_df

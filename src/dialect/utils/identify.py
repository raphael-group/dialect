import logging
import numpy as np
import pandas as pd

from dialect.utils.helpers import *
from dialect.models.gene import Gene

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def identify_pairwise_interactions(cnt_mtx, bmr_pmfs, out, k):
    """
    Main function to identify pairwise interactions between genetic drivers in tumors using DIALECT.
    ! Work in Progress

    @param cnt_mtx (str): Path to the input count matrix file containing mutation data.
    @param bmr_pmfs (str): Path to the file with background mutation rate (BMR) distributions.
    @param out (str): Directory where outputs will be saved.
    @param k (int): Top k genes according to count of mutations will be used.
    """
    logging.info("Identifying pairwise interactions using DIALECT")
    check_file_exists(cnt_mtx)
    check_file_exists(bmr_pmfs)

    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}

    if k <= 0:
        logging.error("k must be a positive integer")
        raise ValueError("k must be a positive integer")

    genes = []
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].values
        bmr_pmf_arr = bmr_dict.get(gene_name, None)
        if bmr_pmf_arr is None:
            raise ValueError(f"No BMR PMF found for gene {gene_name}")
        bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
        genes.append(Gene(name=gene_name, counts=counts, bmr_pmf=bmr_pmf))
    logging.info(f"Initialized {len(genes)} Gene objects.")

    # log info on about to run pi estimation on single genes
    logging.info("Running EM to estimate pi for single genes...")
    for gene in genes:
        gene.estimate_pi_with_em_from_scratch()
        logging.info(f"Estimated pi of {gene.pi} for gene {gene.name}")
    logging.info("Finished estimating pi for single genes.")

    logging.info("Implementation in progress.")
    # ! Continue Implementation Here. Steps:
    # TODO: Call interaction methods on top pairs
    # sort genes by sum of counts variable, pick top X, create interaction objects, run EM on them

import logging
import numpy as np
import pandas as pd

from collections import defaultdict

from dialect.utils.helpers import *
from dialect.models.gene import Gene

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def identify_pairwise_interactions(maf, bmr, out, k):
    """
    Main function to identify pairwise interactions between genetic drivers in tumors using DIALECT.
    ! Work in Progress

    @param maf (str): Path to the input MAF (Mutation Annotation Format) file containing mutation data.
    @param bmr (str): Path to the file with background mutation rate (BMR) distributions.
    @param out (str): Directory where outputs will be saved.
    @param k (int): Top k genes according to count of mutations will be used.
    """
    logging.info("Identifying pairwise interactions using DIALECT")
    check_file_exists(maf)
    check_file_exists(bmr)

    maf_df = pd.read_csv(maf, index_col=0)
    bmr_df = pd.read_csv(bmr, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}

    if k <= 0:
        logging.error("k must be a positive integer")
        raise ValueError("k must be a positive integer")

    genes = {}
    for gene_name in maf_df.columns:
        counts = maf_df[gene_name].values
        bmr_pmf_arr = bmr_dict.get(gene_name, None)
        if bmr_pmf_arr is None:
            raise ValueError(f"No BMR PMF found for gene {gene_name}")
        bmr_pmf = defaultdict(
            lambda: 0, {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
        )
        genes[gene_name] = Gene(name=gene_name, counts=counts, bmr_pmf=bmr_pmf)
    logging.info(f"Initialized {len(genes)} Gene objects.")

    logging.info("Implementation in progress.")
    # ! Continue Implementation Here. Steps:
    # TODO: Create class for pairwise gene interactions
    # TODO: Add and implement EM algorithm in Gene class
    # TODO: Implement 3-5 tests for each method in Gene class
    # TODO: Implement Pomegranate Mixture Model in Gene class
    # TODO: Implement other metrics in Gene class (KL, MI, etc.)

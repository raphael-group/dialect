import os
import logging
import numpy as np
import pandas as pd

from itertools import combinations
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


def check_file_exists(fn):
    """
    Checks if the specified file exists.

    @param fn (str): Path to the file to validate.
    Raises FileNotFoundError if the file does not exist.
    """
    logging.info(f"Validating input file: {fn}")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"File not found: {fn}")


def load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
    return cnt_df, bmr_dict


def initialize_gene_objects(cnt_df, bmr_dict):
    genes = {}
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].values
        bmr_pmf_arr = bmr_dict.get(gene_name, None)
        if bmr_pmf_arr is None:
            raise ValueError(f"No BMR PMF found for gene {gene_name}")
        bmr_pmf = {i: bmr_pmf_arr[i] for i in range(len(bmr_pmf_arr))}
        genes[gene_name] = Gene(
            name=gene_name, samples=cnt_df.index, counts=counts, bmr_pmf=bmr_pmf
        )
    logging.info(f"Initialized {len(genes)} Gene objects.")
    return genes


def initialize_interaction_objects(k, genes):
    interactions = []
    # TODO move logic to get top genes to a helper function to use across different scripts
    top_genes = sorted(genes, key=lambda x: sum(x.counts), reverse=True)[:k]
    for gene_a, gene_b in combinations(top_genes, 2):
        interactions.append(Interaction(gene_a, gene_b))
    logging.info(f"Initialized {len(interactions)} Interaction objects.")
    return top_genes, interactions


def read_cbase_results_file(args):
    cbase_stats = None
    if not args.cbase_stats is None:
        logging.info(f"Reading CBaSE q-values file: {args.cbase_stats}")
        cbase_stats = pd.read_csv(args.cbase_stats, sep="\t", skiprows=1)
    return cbase_stats

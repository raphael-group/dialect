import os
import logging
import numpy as np
import pandas as pd

from itertools import combinations
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction

# TODO: Create essential and verbose logging info for all methods


def verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    check_file_exists(cnt_mtx)
    check_file_exists(bmr_pmfs)


def check_file_exists(fn):
    """
    Checks if the specified file exists.

    @param fn (str): Path to the file to validate.
    Raises FileNotFoundError if the file does not exist.
    """
    logging.info(f"Validating input file: {fn}")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"File not found: {fn}")


def load_bmr_pmfs(bmr_pmfs):
    bmr_df = pd.read_csv(bmr_pmfs, index_col=0)
    bmr_dict = bmr_df.T.to_dict(orient="list")  # key: gene, value: list of pmf values
    bmr_dict = {key: [x for x in bmr_dict[key] if not np.isnan(x)] for key in bmr_dict}
    return bmr_dict


def load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs):
    cnt_df = pd.read_csv(cnt_mtx, index_col=0)
    bmr_dict = load_bmr_pmfs(bmr_pmfs)
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


def read_cbase_results_file(cbase_stats_fn):
    """
    Reads the CBaSE q-values file and returns it as a DataFrame.

    :param cbase_stats_fn: Path to the CBaSE q-values file.
    :type cbase_stats_fn: str or None
    :return: DataFrame containing the CBaSE q-values or None if not provided.
    :rtype: pandas.DataFrame or None
    """
    if cbase_stats_fn is None:
        logging.info("No CBaSE q-values file provided.")
        return None

    logging.info(f"Reading CBaSE q-values file: {cbase_stats_fn}")
    try:
        cbase_stats_df = pd.read_csv(cbase_stats_fn, sep="\t", skiprows=1)
        logging.info("Successfully read the CBaSE q-values file.")
        logging.verbose(f"CBaSE file shape: {cbase_stats_df.shape}")
        logging.verbose(f"CBaSE file preview:\n{cbase_stats_df.head()}")
        return cbase_stats_df
    except Exception as e:
        logging.error(f"Failed to read CBaSE q-values file: {e}")
        raise

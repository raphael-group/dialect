import os
import logging
import numpy as np
import pandas as pd


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

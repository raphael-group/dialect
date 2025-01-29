"""TODO: Add docstring."""

import sys
from pathlib import Path

import pandas as pd

current_dir = Path(__file__).parent
project_root = (current_dir / "../../../").resolve()
external_dir = project_root / "external"
sys.path.append(str(external_dir))

from WeSME.WeSME import (
    compute_pairwise_pvalues,
    compute_sample_weights,
    convert_cnt_mtx_to_mut_list,
    run_weighted_sampling,
)


def run_wesme_analysis(
    cnt_df: pd.DataFrame,
    out: str,
    interactions: list,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    cnt_df = cnt_df.T
    wesme_dout = Path(out) / "WeSME_output"
    wesme_dout.mkdir(parents=True, exist_ok=True)
    mut_fn = convert_cnt_mtx_to_mut_list(cnt_df, wesme_dout)
    compute_sample_weights(mut_fn, wesme_dout)
    freqs_fn = wesme_dout / "sample_mut_freqs.txt"
    run_weighted_sampling(mut_fn, freqs_fn, 100, wesme_dout)
    gene_pairs = [(ixn.gene_a.name, ixn.gene_b.name) for ixn in interactions]
    return compute_pairwise_pvalues(mut_fn, wesme_dout, gene_pairs)

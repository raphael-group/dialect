"""TODO: Add docstring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def run_wesme_analysis(
    cnt_df: pd.DataFrame,
    out: str,
    interactions: list,
) -> pd.DataFrame:
    """TODO: Add docstring."""
    external_dir = (Path(__file__).parent / "../../../external").resolve()
    if str(external_dir) not in sys.path:
        sys.path.append(str(external_dir))
    # WeSME is vendored in external/ (put on sys.path above), not an installed
    # package; this is the single sanctioned dynamic-import site (no static import
    # of external/ anywhere else in the package).
    from WeSME.WeSME import (  # noqa: PLC0415
        compute_pairwise_pvalues,
        compute_sample_weights,
        convert_cnt_mtx_to_mut_list,
        run_weighted_sampling,
    )

    cnt_df = cnt_df.T
    wesme_dout = Path(out) / "WeSME_output"
    wesme_dout.mkdir(parents=True, exist_ok=True)
    mut_fn = convert_cnt_mtx_to_mut_list(cnt_df, wesme_dout)
    compute_sample_weights(mut_fn, wesme_dout)
    freqs_fn = wesme_dout / "sample_mut_freqs.txt"
    run_weighted_sampling(mut_fn, freqs_fn, 100, wesme_dout)
    gene_pairs = [(ixn.gene_a.name, ixn.gene_b.name) for ixn in interactions]
    return compute_pairwise_pvalues(mut_fn, wesme_dout, gene_pairs)

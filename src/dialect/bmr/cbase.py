"""CBaSE background-mutation-rate provider.

Wraps the vendored CBaSE (Weghorn & Sunyaev, *Nat. Genet.* 2017) behind the
:class:`~dialect.bmr.base.BMRProvider` contract: the subprocess invocation, temp
files, and path anchoring are fully hidden behind ``.estimate()`` (the pysam
"wrapper-is-the-API" pattern).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dialect.bmr._cbase_run import generate_bmr_and_counts
from dialect.bmr.base import BMRResult
from dialect.data.io import load_bmr_pmfs, read_cbase_results_file


class CBaSEProvider:
    """Background model from CBaSE's empirical-Bayes per-gene count PMFs."""

    name = "cbase"

    def __init__(self, threshold: str = "1e-100") -> None:
        """`threshold` is CBaSE's PMF tail-truncation cutoff."""
        self.threshold = threshold

    def estimate(
        self,
        maf_path: str,
        out_dir: str,
        *,
        reference: str = "hg19",
    ) -> BMRResult:
        """Run CBaSE on ``maf_path`` and return the background model."""
        generate_bmr_and_counts(maf_path, out_dir, reference, self.threshold)
        return self.load(out_dir)

    def load(self, out_dir: str) -> BMRResult:
        """Build a :class:`BMRResult` from an existing ``generate`` output dir."""
        out = Path(out_dir)
        pmf_arrays = load_bmr_pmfs(str(out / "bmr_pmfs.csv"))
        pmfs = {gene: dict(enumerate(arr)) for gene, arr in pmf_arrays.items()}
        counts = pd.read_csv(out / "count_matrix.csv", index_col=0)
        q_values = out / "CBaSE_output" / "q_values.txt"
        selection = (
            read_cbase_results_file(str(q_values)) if q_values.exists() else None
        )
        return BMRResult(
            pmfs=pmfs,
            counts=counts,
            selection=selection,
            provider=self.name,
        )

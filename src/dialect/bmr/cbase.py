"""CBaSE background-mutation-rate provider.

Wraps the vendored CBaSE (Weghorn & Sunyaev, *Nat. Genet.* 2017) behind the
:class:`~dialect.bmr.base.BMRProvider` contract: the subprocess invocation, temp
files, and path anchoring are fully hidden behind ``.estimate()`` (the pysam
"wrapper-is-the-API" pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from dialect.bmr._cbase_run import generate_bmr_and_counts
from dialect.bmr.base import BMRResult
from dialect.data.io import load_bmr_pmfs, read_cbase_results_file

if TYPE_CHECKING:
    from collections.abc import Sequence


class CBaSEProvider:
    """Background model from CBaSE's empirical-Bayes per-gene count PMFs."""

    name = "cbase"

    def __init__(
        self,
        threshold: str = "1e-100",
        *,
        n_samples: int | None = None,
        sample_ids: Sequence[str] | str | Path | None = None,
        pmf_only: bool = False,
    ) -> None:
        """Configure CBaSE's PMF cutoff and optional exact complete-cohort axis.

        Args:
            threshold: CBaSE's PMF tail-truncation cutoff.
            n_samples: Optional assertion for the complete cohort size. It must equal
                the length of ``sample_ids``; supplying it without ``sample_ids`` is
                rejected because zero-event rows could not be constructed.
            sample_ids: Exact ordered sample identifiers, or a UTF-8 path containing
                one identifier per line. The generated count matrices are reindexed
                to this axis. ``None`` preserves CBaSE's retained-mutation sample
                inference; DIALECT uses that same inferred axis for its count rows.
            pmf_only: Generate passenger-count PMFs for observed genes without
                running CBaSE's separate simulated gene-selection q-value stage.
        """
        self.threshold = threshold
        self.n_samples = n_samples
        self.sample_ids = sample_ids
        self.pmf_only = pmf_only

    def estimate(
        self,
        maf_path: str,
        out_dir: str,
        *,
        reference: str = "hg19",
    ) -> BMRResult:
        """Run CBaSE on ``maf_path`` and return the background model."""
        options = {
            "n_samples": self.n_samples,
            "sample_ids": self.sample_ids,
        }
        if self.pmf_only:
            options["pmf_only"] = True
        generate_bmr_and_counts(
            maf_path,
            out_dir,
            reference,
            self.threshold,
            **options,
        )
        return self.load(out_dir)

    def load(self, out_dir: str) -> BMRResult:
        """Build a :class:`BMRResult` from an existing ``generate`` output dir."""
        out = Path(out_dir)
        pmf_arrays = load_bmr_pmfs(str(out / "bmr_pmfs.csv"))
        pmfs = {gene: dict(enumerate(arr)) for gene, arr in pmf_arrays.items()}
        counts = pd.read_csv(
            out / "count_matrix.csv",
            index_col=0,
            dtype={0: "string"},
            keep_default_na=False,
        )
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

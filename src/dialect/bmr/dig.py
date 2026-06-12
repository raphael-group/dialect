"""DIG (DIGDriver) background-mutation-rate provider.

Promotes :func:`dialect.utils.dig_bmr.dig_results_to_bmr_pmfs` to the
:class:`~dialect.bmr.base.BMRProvider` contract. DIG's gene model emits per-gene
negative-binomial parameters; we convert them to DIALECT's per-sample PMFs (see
``dialect.utils.dig_bmr`` for the Gamma-Poisson derivation).

The provider is configured with a pre-computed DIG ``geneDriver`` results file
(DIG runs in its own conda env). Running DIG end-to-end from a raw MAF additionally
requires DIG's annotation pipeline (hg19 fasta + ``DigPreprocess``); that is tracked
as a follow-up and intentionally not inlined here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dialect.bmr.base import BMRResult
from dialect.utils.dig_bmr import dig_results_to_bmr_pmfs
from dialect.utils.helpers import check_file_exists, load_bmr_pmfs


class DIGProvider:
    """Background model from a DIGDriver gene-model results file."""

    name = "dig"

    def __init__(self, dig_results: str, n_samples: int) -> None:
        """`dig_results` is a ``DigDriver.py geneDriver`` ``*.results.txt`` for the cohort."""
        self.dig_results = dig_results
        self.n_samples = n_samples

    def estimate(
        self,
        maf_path: str,
        out_dir: str,
        *,
        reference: str = "hg19",  # noqa: ARG002 - baked into DIG's pretrained model
    ) -> BMRResult:
        """Convert the configured DIG results into a :class:`BMRResult`.

        The count matrix is background-agnostic; it is read from ``out_dir`` if a prior
        ``generate`` produced it, otherwise left empty (DIALECT can be pointed at any
        matching ``count_matrix.csv``).
        """
        check_file_exists(maf_path)
        check_file_exists(self.dig_results)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bmr_csv = out / "bmr_pmfs.dig.csv"
        dig_results_to_bmr_pmfs(self.dig_results, self.n_samples, str(bmr_csv))
        pmf_arrays = load_bmr_pmfs(str(bmr_csv))
        pmfs = {gene: dict(enumerate(arr)) for gene, arr in pmf_arrays.items()}
        counts_path = out / "count_matrix.csv"
        counts = (
            pd.read_csv(counts_path, index_col=0)
            if counts_path.exists()
            else pd.DataFrame()
        )
        return BMRResult(pmfs=pmfs, counts=counts, provider=self.name)

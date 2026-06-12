"""The in-memory cohort: a cohort's raw observed mutation data.

:class:`MutationCohort` is the raw-data object the rest of DIALECT builds on -- the
samples-by-gene count matrix plus the per-gene background PMFs, and nothing derived.
It lives in the ``data`` base layer, so it holds no model objects, no EM estimates,
and no annotations; those are assembled from a cohort in ``models`` (see
:mod:`dialect.models.assembly`). ``data`` never imports ``models``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dialect.data.io import load_cnt_mtx_and_bmr_pmfs, verify_cnt_mtx_and_bmr_pmfs

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class MutationCohort:
    """A cohort's raw observed mutation data (no derived state).

    Attributes:
        counts: samples x ``GENE_M``/``GENE_N`` integer somatic-count matrix.
        bmr_pmfs: per-gene background PMFs ``P(B=k)`` (gene -> list over counts).
    """

    counts: pd.DataFrame
    bmr_pmfs: dict

    @property
    def samples(self) -> pd.Index:
        """The cohort's sample index (the rows of :attr:`counts`)."""
        return self.counts.index

    @classmethod
    def from_files(cls, cnt_mtx: str, bmr_pmfs: str) -> MutationCohort:
        """Load a cohort from a ``count_matrix.csv`` + ``bmr_pmfs.csv`` pair."""
        verify_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)
        counts, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx, bmr_pmfs)
        return cls(counts=counts, bmr_pmfs=bmr_dict)

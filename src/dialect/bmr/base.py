"""The ``BMRProvider`` contract every background-mutation-rate backend satisfies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BMRResult:
    """A cohort's background-mutation-rate estimate in DIALECT's input contract.

    Attributes:
        pmfs: mapping ``"GENE_M"`` / ``"GENE_N"`` -> ``{count: probability}``, the
            per-sample background count PMF (summing to 1) for each gene and effect.
        counts: samples x gene-effect somatic count matrix (background-agnostic).
        selection: optional per-gene selection statistics (e.g. CBaSE phi / p-values).
        provider: name of the provider that produced this result.
    """

    pmfs: dict[str, dict[int, float]]
    counts: pd.DataFrame
    selection: pd.DataFrame | None = None
    provider: str = ""


@runtime_checkable
class BMRProvider(Protocol):
    """Estimates a per-gene background-mutation-rate model for a cohort.

    Implementations may wrap an external tool (CBaSE), a pretrained model (DIG), or a
    regression (dNdScv); they all return the same :class:`BMRResult` contract, so the
    background model becomes a swappable, benchmarkable axis.
    """

    name: str

    def estimate(
        self,
        maf_path: str,
        out_dir: str,
        *,
        reference: str = "hg19",
    ) -> BMRResult:
        """Estimate the background model for ``maf_path``, writing artifacts to ``out_dir``."""
        ...

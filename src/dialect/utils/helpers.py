"""Backward-compat shim.

Cohort I/O now lives in :mod:`dialect.data.io`; Gene/Interaction assembly now lives
in :mod:`dialect.models.assembly`. Both are re-exported here for legacy callers
(``from dialect.utils.helpers import ...``). New code should import from those
modules directly.
"""

from __future__ import annotations

from dialect.data.io import (
    check_file_exists,
    load_bmr_pmfs,
    load_cnt_mtx_and_bmr_pmfs,
    load_likely_passenger_genes,
    load_putative_driver_genes,
    read_cbase_results_file,
    verify_cnt_mtx_and_bmr_pmfs,
)
from dialect.models.assembly import (
    initialize_gene_objects,
    initialize_interaction_objects,
)

__all__ = [
    "check_file_exists",
    "initialize_gene_objects",
    "initialize_interaction_objects",
    "load_bmr_pmfs",
    "load_cnt_mtx_and_bmr_pmfs",
    "load_likely_passenger_genes",
    "load_putative_driver_genes",
    "read_cbase_results_file",
    "verify_cnt_mtx_and_bmr_pmfs",
]

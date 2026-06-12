"""Backward-compat shim: CBaSE run/extraction now lives in :mod:`dialect.bmr`.

Use ``dialect.bmr.cbase.CBaSEProvider`` (or :mod:`dialect.bmr._cbase_run`) in new
code. This module re-exports the public functions for legacy call sites.
"""

from __future__ import annotations

from dialect.bmr._cbase_run import (
    convert_maf_to_cbase_input_file,
    generate_bmr_and_counts,
    generate_bmr_files_from_cbase_output,
    generate_bmr_using_cbase,
    generate_counts_from_cbase_output,
)

__all__ = [
    "convert_maf_to_cbase_input_file",
    "generate_bmr_and_counts",
    "generate_bmr_files_from_cbase_output",
    "generate_bmr_using_cbase",
    "generate_counts_from_cbase_output",
]

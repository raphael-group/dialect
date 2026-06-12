"""Backward-compat shim: the DIG->PMF conversion now lives in :mod:`dialect.bmr`.

Import from :mod:`dialect.bmr._dig_pmf` (or use ``dialect.bmr.dig.DIGProvider``) in
new code. This module re-exports the public function for legacy call sites.
"""

from __future__ import annotations

from dialect.bmr._dig_pmf import dig_results_to_bmr_pmfs

__all__ = ["dig_results_to_bmr_pmfs"]

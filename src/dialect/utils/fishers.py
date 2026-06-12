"""Backward-compat shim: Fisher's-exact baseline now lives in dialect.baselines."""

from __future__ import annotations

from dialect.baselines.fishers import run_fishers_exact_analysis

__all__ = ["run_fishers_exact_analysis"]

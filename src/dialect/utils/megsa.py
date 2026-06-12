"""Backward-compat shim: MEGSA baseline now lives in dialect.baselines."""

from __future__ import annotations

from dialect.baselines.megsa import run_megsa_analysis

__all__ = ["run_megsa_analysis"]

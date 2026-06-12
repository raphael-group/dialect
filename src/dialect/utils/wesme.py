"""Backward-compat shim: WeSME baseline now lives in dialect.baselines."""

from __future__ import annotations

from dialect.baselines.wesme import run_wesme_analysis

__all__ = ["run_wesme_analysis"]

"""Backward-compat shim: DISCOVER baseline now lives in dialect.baselines."""

from __future__ import annotations

from dialect.baselines.discover import run_discover_analysis

__all__ = ["run_discover_analysis"]

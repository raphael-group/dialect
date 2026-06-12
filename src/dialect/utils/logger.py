"""Backward-compat shim: logging configuration now lives in :mod:`dialect.config`."""

from __future__ import annotations

from dialect.config import VERBOSE_LEVEL, configure_logging, verbose

__all__ = ["VERBOSE_LEVEL", "configure_logging", "verbose"]

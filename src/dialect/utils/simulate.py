"""Backward-compat shim: simulation/benchmarking now lives in dialect.experiments.

Import from ``dialect.experiments.simulate`` in new code. Attribute access here
forwards to that module so legacy ``dialect.utils.simulate`` call sites (the CLI
``simulate`` command and the research scripts in ``analysis/``) keep working.
"""

from __future__ import annotations

from dialect.experiments import simulate as _simulate


def __getattr__(name: str) -> object:
    """Forward any attribute access to the new dialect.experiments.simulate module."""
    return getattr(_simulate, name)

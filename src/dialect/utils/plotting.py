"""Backward-compat shim: plotting now lives in :mod:`dialect.viz.plotting`.

Import from ``dialect.viz`` (or ``dialect.viz.plotting``) in new code. Attribute
access here forwards to that module so legacy ``dialect.utils.plotting`` call sites
(the research scripts in ``analysis/`` and ``utils/simulate.py``) keep working.
"""

from __future__ import annotations

from dialect.viz import plotting as _plotting


def __getattr__(name: str) -> object:
    """Forward any attribute access to the new dialect.viz.plotting module."""
    return getattr(_plotting, name)

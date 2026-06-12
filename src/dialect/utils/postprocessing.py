"""Backward-compat shim: post-processing now lives in :mod:`dialect.stats`.

Ranking moved to ``dialect.stats.ranking``, the epsilon threshold to
``dialect.stats.thresholds``, and the per-method constants to
``dialect.stats.constants``. Re-exported here for legacy callers.
"""

from __future__ import annotations

from dialect.stats.ranking import (
    generate_top_ranked_co_interaction_tables,
    generate_top_ranked_me_interaction_tables,
)
from dialect.stats.thresholds import compute_epsilon_threshold

__all__ = [
    "compute_epsilon_threshold",
    "generate_top_ranked_co_interaction_tables",
    "generate_top_ranked_me_interaction_tables",
]

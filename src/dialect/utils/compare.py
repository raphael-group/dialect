"""Backward-compat shim: the comparison runner now lives in dialect.baselines.runner.

Use ``dialect.baselines.runner`` in new code. Re-exported here for legacy callers
(``dialect.utils.compare`` / ``dialect.utils.run_comparison_methods``).
"""

from __future__ import annotations

from dialect.baselines.runner import results_to_dataframe, run_comparison_methods

__all__ = ["results_to_dataframe", "run_comparison_methods"]

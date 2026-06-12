"""Statistical post-processing for DIALECT results: thresholds, ranking, simulation.

A pure-statistics layer that consumes result frames (and model value objects) and
produces significance thresholds and ranked ME/CO tables. It imports only ``data``
and ``models``; never ``bmr``, ``baselines``, ``viz``, ``api``, or ``cli``.
"""

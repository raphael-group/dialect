"""Simulation + benchmarking experiments (research orchestration).

This is top-of-stack orchestration code: it generates synthetic cohorts, runs
DIALECT and the baselines over them, evaluates the results, and renders figures.
It therefore sits *above* the pure layers and may import ``data``, ``models``,
``stats``, ``viz``, and ``baselines`` -- which is why it lives here rather than in
``stats`` (``stats`` must not import ``viz``).
"""

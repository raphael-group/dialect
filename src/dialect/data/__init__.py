"""DIALECT's data layer: the I/O and in-memory data contract.

This is the base of the dependency DAG -- it has no internal DIALECT imports, so
every higher layer (``bmr``, ``baselines``, ``models``, ``stats``, ``api``) may
depend on it.
"""

"""Alternative ME/CO methods (Fisher, DISCOVER, MEGSA, WeSME) for benchmarking.

These are adapter/runner modules that wrap external tools or reference statistics so
DIALECT's calls can be compared against them. They consume cohort data (``data``) and
model value objects (``models``), and invoke vendored tools in ``external/`` only via
subprocess / guarded ``sys.path`` -- never by importing ``external`` as a package.
"""

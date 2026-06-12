"""DIALECT: mutual exclusivity & co-occurrence between cancer drivers, BMR-aware.

DIALECT is an EM latent-variable model that decomposes each observed somatic count
into a passenger background (from a pluggable background-mutation-rate provider) and
a latent driver indicator, then ranks gene pairs for mutual exclusivity and
co-occurrence. Import the core operations directly::

    from dialect import estimate_bmr, identify_interactions

See :mod:`dialect.api` for the full surface.
"""

from __future__ import annotations

from dialect.api import IdentifyResult, estimate_bmr, identify_interactions

try:
    from dialect._version import __version__
except ImportError:  # pragma: no cover - version file is generated at build time
    __version__ = "0+unknown"

__all__ = [
    "IdentifyResult",
    "__version__",
    "estimate_bmr",
    "identify_interactions",
]

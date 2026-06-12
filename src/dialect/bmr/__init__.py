"""Pluggable background-mutation-rate (BMR) providers for DIALECT.

DIALECT conditions every likelihood on a per-gene, per-effect background count PMF
``P(B=k)``. This package makes the background model a first-class, swappable axis
(``cbase`` / ``dig`` / ...), which is both cleaner architecture and the direct answer
to reviewers asking how robust the ME/CO calls are to the choice of BMR model.
"""

from dialect.bmr.base import BMRProvider, BMRResult
from dialect.bmr.cbase import CBaSEProvider
from dialect.bmr.dig import DIGProvider
from dialect.bmr.registry import available, get_provider

__all__ = [
    "BMRProvider",
    "BMRResult",
    "CBaSEProvider",
    "DIGProvider",
    "available",
    "get_provider",
]

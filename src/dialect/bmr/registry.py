"""Name -> BMRProvider lookup so ``--bmr cbase|dig`` resolves by string.

New providers register here without touching existing code (Open/Closed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialect.bmr.cbase import CBaSEProvider
from dialect.bmr.dig import DIGProvider

if TYPE_CHECKING:
    from dialect.bmr.base import BMRProvider

_PROVIDERS: dict[str, type] = {
    "cbase": CBaSEProvider,
    "dig": DIGProvider,
}


def available() -> list[str]:
    """Return the registered BMR-provider names."""
    return sorted(_PROVIDERS)


def get_provider(name: str, **kwargs: object) -> BMRProvider:
    """Instantiate a BMR provider by name (extra kwargs go to its constructor)."""
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError:
        msg = f"Unknown BMR provider {name!r}; available: {available()}"
        raise ValueError(msg) from None
    return provider_cls(**kwargs)

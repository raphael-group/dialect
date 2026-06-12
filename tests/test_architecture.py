"""Architecture guard: enforce the layered import DAG so it cannot silently regress.

DIALECT's intended dependency direction is one-way, ``bmr -> data`` (with ``data`` as a
dependency-free base layer). The providers in :mod:`dialect.bmr` used to import *upward*
into :mod:`dialect.utils`; this test pins the de-inverted layering in place so a future
edit that re-introduces an upward edge fails CI instead of rotting the architecture.

As more of the re-layout lands (baselines/, stats/, viz/, cli/, api), extend
``ALLOWED_INTERNAL_PREFIXES`` with each newly-cleaned layer's permitted dependencies.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "dialect"

# package -> internal dialect.* prefixes its modules may import from. A package may
# always import itself; "data" is the base layer and imports nothing internal.
ALLOWED_INTERNAL_PREFIXES = {
    "data": {"dialect.data"},
    "bmr": {"dialect.bmr", "dialect.data"},
    # models may build on the data layer and depend on the BMRProvider *port*
    # (dialect.bmr.base), but never on concrete providers or higher layers.
    "models": {"dialect.models", "dialect.data", "dialect.bmr.base"},
    # baselines wrap external tools / reference stats; they consume cohort data
    # and model value objects, but never bmr/stats/viz/api/cli.
    "baselines": {"dialect.baselines", "dialect.data", "dialect.models"},
    # stats is pure statistics over result frames + models; never bmr/baselines/viz.
    "stats": {"dialect.stats", "dialect.data", "dialect.models"},
    # viz renders figures from cohort data + stats result frames; never bmr/baselines.
    "viz": {"dialect.viz", "dialect.data", "dialect.stats"},
}


def _internal_imports(py_file: Path) -> set[str]:
    """Return the set of ``dialect.*`` modules imported by ``py_file``."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import (within the same package) -> not upward.
            if node.module and node.level == 0 and node.module.startswith("dialect"):
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("dialect"):
                    imported.add(alias.name)
    return imported


def _is_allowed(module: str, allowed: set[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in allowed)


@pytest.mark.parametrize("package", sorted(ALLOWED_INTERNAL_PREFIXES))
def test_package_only_imports_allowed_layers(package: str) -> None:
    allowed = ALLOWED_INTERNAL_PREFIXES[package]
    violations: list[str] = []
    for py_file in (SRC / package).rglob("*.py"):
        for module in _internal_imports(py_file):
            if not _is_allowed(module, allowed):
                rel = py_file.relative_to(SRC.parent)
                violations.append(f"{rel} imports {module}")
    assert not violations, (
        f"dialect.{package} may only import {sorted(allowed)}; upward/illegal edges:\n"
        + "\n".join(sorted(violations))
    )

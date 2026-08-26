"""Build the immutable K=100 competing-method release for the DIALECT Atlas.

The release is generated only from the canonical 71 cohort count matrices. It never
reads a pre-existing comparison CSV, so the legacy ``output/CHOL`` K=30 artifact cannot
enter the release. Fisher, DISCOVER, MEGSA, and WeSME/WeSCO are run through DIALECT's
public comparison API; the build fails closed if any method is unavailable or emits an
incomplete result.

DISCOVER is an optional dependency. Supply the documented DISCOVER Python package on
``PYTHONPATH`` before invoking this module.

Usage::

    PYTHONPATH=/path/to/DISCOVER/python \
      python -m analysis.build_atlas_baselines --jobs 4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SCHEMA_VERSION = "1.0"
SOURCE_GENE_K = 100
RELEASE_SEED = 20260826
RELEASE_ID = "dialect-atlas-baselines-k100"
EXPECTED_COHORT_COUNT = 71

GENE_COLUMNS = ("Gene A", "Gene B")
METHOD_COLUMNS: dict[str, tuple[str, ...]] = {
    "fisher": (
        "Fisher's ME P-Val",
        "Fisher's CO P-Val",
        "Fisher's ME Q-Val",
        "Fisher's CO Q-Val",
    ),
    "discover": (
        "Discover ME P-Val",
        "Discover CO P-Val",
        "Discover ME Q-Val",
        "Discover CO Q-Val",
    ),
    "megsa": (
        "MEGSA S-Score (LRT)",
        "MEGSA P-Val",
        "MEGSA Q-Val",
    ),
    "wesme": ("WeSME P-Val", "WeSME Q-Val"),
    "wesco": ("WeSCO P-Val", "WeSCO Q-Val"),
}
EXPECTED_COLUMNS = (
    *GENE_COLUMNS,
    *METHOD_COLUMNS["fisher"],
    *METHOD_COLUMNS["discover"],
    *METHOD_COLUMNS["megsa"],
    "WeSME P-Val",
    "WeSCO P-Val",
    "WeSME Q-Val",
    "WeSCO Q-Val",
)
PROBABILITY_COLUMNS = tuple(
    column for column in EXPECTED_COLUMNS if column.endswith(("P-Val", "Q-Val"))
)
METHOD_CONTRACT = {
    "fisher": {
        "directions": ["ME", "CO"],
        "call_rule": "direction-specific BH q < 0.01",
        "columns": list(METHOD_COLUMNS["fisher"]),
    },
    "discover": {
        "directions": ["ME", "CO"],
        "call_rule": "direction-specific BH q < 0.01",
        "columns": list(METHOD_COLUMNS["discover"]),
    },
    "megsa": {
        "directions": ["ME"],
        "call_rule": "p < 0.001",
        "columns": list(METHOD_COLUMNS["megsa"]),
    },
    "wesme": {
        "directions": ["ME"],
        "call_rule": "BH q < 0.01",
        "columns": list(METHOD_COLUMNS["wesme"]),
    },
    "wesco": {
        "directions": ["CO"],
        "call_rule": "BH q < 0.01",
        "columns": list(METHOD_COLUMNS["wesco"]),
    },
}


@dataclass(frozen=True)
class SourceSpec:
    """One canonical Atlas study root and its locked cohort count."""

    study: str
    root: Path
    expected_cohorts: int


@dataclass(frozen=True)
class CohortSpec:
    """A canonical cohort count matrix selected for baseline generation."""

    study: str
    cohort: str
    count_matrix: Path

    @property
    def cohort_id(self) -> str:
        """Return the stable study-qualified cohort identifier."""
        return f"{self.study}__{self.cohort}"


@dataclass(frozen=True)
class ReleaseContext:
    """Paths and locked parameters shared by independent cohort workers."""

    staging_root: Path
    repo_root: Path
    published_root: Path
    k: int
    release_seed: int


def canonical_sources(repo_root: Path) -> tuple[SourceSpec, ...]:
    """Return the three locked Atlas study roots (32 + 34 + 5 cohorts)."""
    return (
        SourceSpec("TCGA", repo_root / "output/pancan", 32),
        SourceSpec("MSK-IMPACT", repo_root / "output/msk/IMPACT2026", 34),
        SourceSpec("MSK-CHORD", repo_root / "output/msk/CHORD2024", 5),
    )


def discover_cohorts(
    sources: Sequence[SourceSpec],
    *,
    expected_total: int = EXPECTED_COHORT_COUNT,
) -> list[CohortSpec]:
    """Enumerate and validate the exact canonical cohort set."""
    cohorts: list[CohortSpec] = []
    for source in sources:
        if not source.root.is_dir():
            msg = f"Missing canonical cohort root: {source.root}"
            raise FileNotFoundError(msg)
        cohort_dirs = sorted(
            path
            for path in source.root.iterdir()
            if path.is_dir() and (path / "count_matrix.csv").is_file()
        )
        if len(cohort_dirs) != source.expected_cohorts:
            msg = (
                f"{source.study} cohort lock failed: expected "
                f"{source.expected_cohorts}, found {len(cohort_dirs)} under "
                f"{source.root}"
            )
            raise RuntimeError(msg)
        cohorts.extend(
            CohortSpec(source.study, path.name, path / "count_matrix.csv")
            for path in cohort_dirs
        )

    locked_total = sum(source.expected_cohorts for source in sources)
    if locked_total != expected_total or len(cohorts) != expected_total:
        msg = (
            "Atlas cohort lock failed: expected exactly "
            f"{expected_total}, found {len(cohorts)}"
        )
        raise RuntimeError(msg)
    cohort_ids = [cohort.cohort_id for cohort in cohorts]
    if len(cohort_ids) != len(set(cohort_ids)):
        msg = "Atlas cohort lock failed: duplicate study-qualified cohort IDs"
        raise RuntimeError(msg)
    return cohorts


def stable_cohort_seed(release_seed: int, cohort_id: str) -> int:
    """Derive a stable NumPy legacy-RNG seed from the release and cohort IDs."""
    payload = f"{release_seed}:{cohort_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, files: Iterable[Path] | None = None) -> str:
    """Hash relative paths and bytes for a deterministic file-tree digest."""
    selected = sorted(
        files
        if files is not None
        else (path for path in root.rglob("*") if path.is_file()),
    )
    digest = hashlib.sha256()
    for path in selected:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    """Prefer a portable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_count_matrix(path: Path) -> pd.DataFrame:
    """Load and minimally validate one baseline input matrix."""
    frame = pd.read_csv(path, index_col=0)
    if frame.empty or frame.shape[1] < 2:
        msg = f"Count matrix must contain samples and at least two features: {path}"
        raise ValueError(msg)
    if not frame.columns.is_unique:
        msg = f"Count matrix has duplicate feature names: {path}"
        raise ValueError(msg)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        msg = f"Count matrix contains non-finite or negative values: {path}"
        raise ValueError(msg)
    return numeric


def top_features(counts: pd.DataFrame, k: int = SOURCE_GENE_K) -> list[str]:
    """Match DIALECT's stable top-k-by-total-count feature selection exactly."""
    totals = counts.sum(axis=0)
    ordered = sorted(counts.columns, key=lambda feature: totals[feature], reverse=True)
    return [str(feature) for feature in ordered[: min(k, len(ordered))]]


def expected_pair_universe(features: Sequence[str]) -> set[tuple[str, str]]:
    """Return the unordered pair universe implied by the selected features."""
    return {tuple(sorted(pair)) for pair in combinations(features, 2)}


def validate_comparison_frame(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> None:
    """Fail closed unless a comparison table exactly matches the release contract."""
    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(EXPECTED_COLUMNS))
        msg = (
            "Incomplete or drifted comparison schema; "
            f"missing={missing}, extra={extra}, order={list(actual_columns)}"
        )
        raise ValueError(msg)
    if frame.isna().any(axis=None):
        null_columns = frame.columns[frame.isna().any()].tolist()
        msg = f"Comparison table contains null values in columns: {null_columns}"
        raise ValueError(msg)

    expected_pairs = expected_pair_universe(features)
    if len(frame) != len(expected_pairs):
        msg = f"Expected {len(expected_pairs)} rows, found {len(frame)}"
        raise ValueError(msg)

    observed_pairs: list[tuple[str, str]] = []
    pair_rows = frame.loc[:, list(GENE_COLUMNS)].itertuples(index=False, name=None)
    for gene_a, gene_b in pair_rows:
        a, b = str(gene_a), str(gene_b)
        if a == b:
            msg = f"Self-pair found in comparison table: {a}:{b}"
            raise ValueError(msg)
        observed_pairs.append(tuple(sorted((a, b))))
    if len(observed_pairs) != len(set(observed_pairs)):
        msg = "Comparison table contains duplicate unordered gene pairs"
        raise ValueError(msg)
    observed_set = set(observed_pairs)
    if observed_set != expected_pairs:
        missing_pairs = sorted(expected_pairs - observed_set)[:5]
        extra_pairs = sorted(observed_set - expected_pairs)[:5]
        msg = (
            "Comparison pair universe does not match top features; "
            f"missing_examples={missing_pairs}, extra_examples={extra_pairs}"
        )
        raise ValueError(msg)

    numeric_columns = [
        column for column in EXPECTED_COLUMNS if column not in GENE_COLUMNS
    ]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        msg = "Comparison table contains non-finite numeric values"
        raise ValueError(msg)
    probabilities = numeric.loc[:, list(PROBABILITY_COLUMNS)]
    outside_unit_interval = (probabilities < 0) | (probabilities > 1)
    if outside_unit_interval.any(axis=None):
        bad_columns = probabilities.columns[outside_unit_interval.any()].tolist()
        msg = f"Probability/q-value columns outside [0, 1]: {bad_columns}"
        raise ValueError(msg)


def _method_coverage() -> dict[str, dict[str, object]]:
    """Return explicit complete coverage for the strict 17-column schema."""
    return {
        method: {"status": "complete", **contract}
        for method, contract in METHOD_CONTRACT.items()
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, human-auditable JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_comparison(count_matrix: Path, output_dir: Path, k: int, seed: int) -> None:
    """Invoke the existing public pipeline after seeding its legacy NumPy RNG."""
    np.random.seed(seed)  # noqa: NPY002 -- vendored WeSME consumes NumPy's global RNG.
    from dialect import api  # noqa: PLC0415 -- keep worker imports after RNG/env setup.

    api.compare_methods(count_matrix, output_dir, top_k=k, gene_level=False)


def _cohort_metadata(
    cohort: CohortSpec,
    counts: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    context: ReleaseContext,
) -> dict[str, Any]:
    """Create one validated cohort's metadata payload."""
    seed = stable_cohort_seed(context.release_seed, cohort.cohort_id)
    comparison = output_dir / "comparison_pairwise_interaction_results.csv"
    data_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "source_gene_k": SOURCE_GENE_K,
        "release_seed": context.release_seed,
        "cohort": {
            "id": cohort.cohort_id,
            "study": cohort.study,
            "name": cohort.cohort,
            "n_samples": int(counts.shape[0]),
            "n_input_features": int(counts.shape[1]),
            "n_tested_features": len(features),
            "expected_pair_count": len(expected_pair_universe(features)),
        },
        "rng": {
            "library": "numpy.random legacy global RNG",
            "seed": seed,
            "derivation": "uint32(sha256('<release_seed>:<study>__<cohort>')[0:4])",
        },
        "top_features": features,
        "method_coverage": _method_coverage(),
        "input": {
            "path": _relative_or_absolute(cohort.count_matrix, context.repo_root),
            "sha256": sha256_file(cohort.count_matrix),
        },
        "artifacts": {
            "comparison": {
                "path": _relative_or_absolute(
                    context.published_root
                    / cohort.study
                    / cohort.cohort
                    / "comparison_pairwise_interaction_results.csv",
                    context.repo_root,
                ),
                "sha256": sha256_file(comparison),
                "rows": len(expected_pair_universe(features)),
            },
            "data_tree_sha256": sha256_tree(output_dir, data_files),
            "data_file_count": len(data_files),
        },
    }


def _read_comparison(path: Path, cohort_id: str) -> pd.DataFrame:
    """Require and read the comparison artifact emitted by the existing runner."""
    if not path.is_file():
        msg = f"Comparison pipeline wrote no result for {cohort_id}"
        raise RuntimeError(msg)
    return pd.read_csv(path)


def _generate_one_cohort(
    cohort: CohortSpec,
    context: ReleaseContext,
) -> dict[str, Any]:
    """Generate, validate, and atomically publish one cohort inside staging."""
    study_root = context.staging_root / cohort.study
    study_root.mkdir(parents=True, exist_ok=True)
    target = study_root / cohort.cohort
    if target.exists():
        msg = f"Duplicate cohort output target: {target}"
        raise FileExistsError(msg)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cohort.cohort}.", dir=study_root))
    try:
        counts = load_count_matrix(cohort.count_matrix)
        features = top_features(counts, context.k)
        seed = stable_cohort_seed(context.release_seed, cohort.cohort_id)
        _run_comparison(cohort.count_matrix, temporary, context.k, seed)
        comparison = temporary / "comparison_pairwise_interaction_results.csv"
        frame = _read_comparison(comparison, cohort.cohort_id)
        validate_comparison_frame(frame, features)
        metadata = _cohort_metadata(
            cohort,
            counts,
            features,
            temporary,
            context,
        )
        _write_json(temporary / "metadata.json", metadata)
        metadata["artifacts"]["metadata"] = {
            "path": _relative_or_absolute(
                context.published_root / cohort.study / cohort.cohort / "metadata.json",
                context.repo_root,
            ),
            "sha256": sha256_file(temporary / "metadata.json"),
        }
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def _package_version(name: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _discover_provenance() -> dict[str, Any]:
    """Require DISCOVER and hash its Python source, including PYTHONPATH installs."""
    spec = importlib.util.find_spec("discover")
    if spec is None or spec.origin is None:
        msg = (
            "DISCOVER is required for the immutable baseline release but is not "
            "importable. Supply the official Python package through "
            "PYTHONPATH before running this generator."
        )
        raise RuntimeError(msg)
    module = importlib.import_module("discover")
    origin = Path(spec.origin).resolve()
    if spec.submodule_search_locations:
        source_root = Path(next(iter(spec.submodule_search_locations))).resolve()
        source_files = sorted(source_root.rglob("*.py"))
        source_hash = sha256_tree(source_root, source_files)
    else:
        source_root = origin.parent
        source_hash = sha256_file(origin)
    version = getattr(module, "__version__", None) or _package_version("discover")
    return {
        "version": str(version) if version is not None else None,
        "origin": str(origin),
        "python_source_root": str(source_root),
        "python_source_sha256": source_hash,
    }


def _git_value(repo_root: Path, arguments: Sequence[str]) -> str | None:
    """Read one Git provenance value without making Git state changes."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _command_value(command: Sequence[str]) -> str | None:
    """Return combined standard output for a read-only version command."""
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return (result.stdout or result.stderr).strip()


def collect_provenance(repo_root: Path) -> dict[str, Any]:
    """Collect software versions and exact baseline implementation source hashes."""
    source_paths = (
        Path("analysis/build_atlas_baselines.py"),
        Path("src/dialect/api.py"),
        Path("src/dialect/baselines/runner.py"),
        Path("src/dialect/baselines/fishers.py"),
        Path("src/dialect/baselines/discover.py"),
        Path("src/dialect/baselines/megsa.py"),
        Path("src/dialect/baselines/wesme.py"),
        Path("src/dialect/models/assembly.py"),
        Path("src/dialect/models/interaction.py"),
        Path("external/MEGSA/MEGSA.R"),
        Path("external/WeSME/WeSME.py"),
    )
    missing_sources = [
        path for path in source_paths if not (repo_root / path).is_file()
    ]
    if missing_sources:
        msg = f"Missing baseline implementation sources: {missing_sources}"
        raise FileNotFoundError(msg)
    git_status = _git_value(
        repo_root,
        ["status", "--porcelain", "--untracked-files=no"],
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            package: _package_version(package)
            for package in (
                "dialect",
                "numpy",
                "pandas",
                "scipy",
                "statsmodels",
                "networkx",
            )
        },
        "git": {
            "commit": _git_value(repo_root, ["rev-parse", "HEAD"]),
            "dirty_tracked_files": bool(git_status),
        },
        "source_files": {
            path.as_posix(): sha256_file(repo_root / path) for path in source_paths
        },
        "discover": _discover_provenance(),
        "rscript": _command_value(["Rscript", "--version"]),
    }


def _root_manifest(
    cohorts: Sequence[CohortSpec],
    cohort_metadata: Sequence[dict[str, Any]],
    sources: Sequence[SourceSpec],
    release_seed: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the root immutable-release manifest."""
    metadata_by_id = {item["cohort"]["id"]: item for item in cohort_metadata}
    entries = []
    for cohort in cohorts:
        metadata = metadata_by_id[cohort.cohort_id]
        entries.append(
            {
                **metadata["cohort"],
                "seed": metadata["rng"]["seed"],
                "top_features": metadata["top_features"],
                "method_coverage": metadata["method_coverage"],
                "input": metadata["input"],
                "artifacts": metadata["artifacts"],
            },
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "source_gene_k": SOURCE_GENE_K,
        "release_seed": release_seed,
        "cohort_count": len(entries),
        "source_counts": {source.study: source.expected_cohorts for source in sources},
        "expected_columns": list(EXPECTED_COLUMNS),
        "probability_columns": list(PROBABILITY_COLUMNS),
        "method_contract": METHOD_CONTRACT,
        "provenance": provenance,
        "cohorts": entries,
    }


def _require_complete_release(metadata: Sequence[dict[str, Any]]) -> None:
    """Fail unless every locked cohort completed before manifest publication."""
    if len(metadata) != EXPECTED_COHORT_COUNT:
        msg = f"Expected {EXPECTED_COHORT_COUNT} completed cohorts, got {len(metadata)}"
        raise RuntimeError(msg)


def build_release(
    repo_root: Path,
    output_root: Path,
    *,
    jobs: int,
    release_seed: int,
) -> Path:
    """Generate all 71 cohorts in staging and atomically publish the release."""
    if jobs <= 0:
        msg = "jobs must be a positive integer"
        raise ValueError(msg)
    sources = canonical_sources(repo_root)
    cohorts = discover_cohorts(sources)
    provenance = collect_provenance(repo_root)
    if output_root.exists():
        msg = f"Refusing to overwrite immutable release directory: {output_root}"
        raise FileExistsError(msg)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent),
    )
    context = ReleaseContext(
        staging_root=staging_root,
        repo_root=repo_root,
        published_root=output_root,
        k=SOURCE_GENE_K,
        release_seed=release_seed,
    )
    metadata: list[dict[str, Any]] = []
    try:
        if jobs == 1:
            for index, cohort in enumerate(cohorts, start=1):
                metadata.append(
                    _generate_one_cohort(
                        cohort,
                        context,
                    ),
                )
                print(f"[{index}/{len(cohorts)}] complete: {cohort.cohort_id}")
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                future_to_cohort = {
                    executor.submit(
                        _generate_one_cohort,
                        cohort,
                        context,
                    ): cohort
                    for cohort in cohorts
                }
                for index, future in enumerate(as_completed(future_to_cohort), start=1):
                    cohort = future_to_cohort[future]
                    metadata.append(future.result())
                    print(f"[{index}/{len(cohorts)}] complete: {cohort.cohort_id}")

        _require_complete_release(metadata)
        manifest = _root_manifest(
            cohorts,
            metadata,
            sources,
            release_seed,
            provenance,
        )
        _write_json(staging_root / "manifest.json", manifest)
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return output_root


def main() -> None:
    """Parse release options and build the complete immutable K=100 baseline set."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent cohort worker processes (default: 1).",
    )
    parser.add_argument(
        "--release-seed",
        type=int,
        default=RELEASE_SEED,
        help=f"Release RNG seed (default: {RELEASE_SEED}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "output/atlas_baselines/k100",
        help="Immutable release directory; must not already exist.",
    )
    args = parser.parse_args()

    mpl_cache = Path(tempfile.gettempdir()) / "dialect-atlas-baseline-mpl-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    result = build_release(
        repo_root,
        args.out.resolve(),
        jobs=args.jobs,
        release_seed=args.release_seed,
    )
    print(f"Published immutable baseline release: {result}")


if __name__ == "__main__":
    main()

"""Validate and attest the focused revision provenance boundary.

The production fitter predates the final release code and intentionally remains
immutable.  This module wraps its v1 receipts with a public, path-sanitized v2
attestation without rewriting any production artifact or parsing association
values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from analysis import prepare_tcga_revision_focused as preparation
from analysis import run_tcga_revision_focused as runner
from analysis import run_tcga_revision_k500 as core

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
FIT_ATTESTATION_CONTRACT: Final = "focused-fit-source-runtime-attestation-v2"
PUBLIC_COHORT_CONTRACT: Final = "focused-public-cohort-contract-projection-v1"
FIT_ATTESTATION_NAME: Final = "fit_execution_attestation.json"
PRODUCTION_FIT_COMMIT: Final = "b23a9fc4f32fd3df6d145a655fec3df221ab8b04"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TCGA_SAMPLE_ID = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9-]+)?")
_MACHINE_FILE_KEYS = frozenset(
    {"ctime_ns", "device", "inode", "mode", "mtime_ns", "nlink", "uid"},
)
_MACHINE_PATH_KEYS = frozenset({"path", "root"})
_FOCUSED_FIT_SOURCES = (
    Path("analysis/prepare_tcga_revision_focused.py"),
    Path("analysis/run_tcga_revision_focused.py"),
    Path("analysis/tcga_revision_config.json"),
)
_GENERATED_UNTRACKED_FIT_SOURCES = frozenset({Path("src/dialect/_version.py")})
FIT_SOURCE_FILES: Final = tuple(
    sorted(
        (
            {*core.SOURCE_FILES, *_FOCUSED_FIT_SOURCES}
            - _GENERATED_UNTRACKED_FIT_SOURCES
        ),
        key=lambda path: path.as_posix(),
    ),
)
RELEASE_PIPELINE_FILES: Final = (
    Path("analysis/build_tcga_revision_focused_release.py"),
    Path("analysis/calibrate_tcga_revision_focused.py"),
    Path("analysis/focused_revision_provenance.py"),
    Path("analysis/freeze_tcga_revision_reporting_rule.py"),
    Path("analysis/postprocess_tcga_revision_focused.py"),
    Path("analysis/report_tcga_revision_focused.py"),
    Path("analysis/tcga_revision_calibration_config.json"),
)
RUNTIME_DISTRIBUTIONS: Final = ("numpy", "pandas", "scipy", "scikit-learn")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        msg = f"Required provenance input is missing or unsafe: {path}"
        raise ValueError(msg)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _record_matches(record: object, path: Path, *, expected_path: str) -> bool:
    return (
        isinstance(record, dict)
        and record.get("path") == expected_path
        and record.get("bytes") == path.stat().st_size
        and record.get("sha256") == _sha256_path(path)
        and path.is_file()
        and not path.is_symlink()
    )


def _write_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        msg = f"Refusing to overwrite provenance artifact: {path}"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"Expected a JSON object: {path}"
        raise TypeError(msg)
    return value


def _require_complete_coordinates(
    records: Sequence[Mapping[str, object]],
    cohorts: Sequence[str],
) -> dict[tuple[str, str], Mapping[str, object]]:
    expected = {(cohort, provider) for cohort in cohorts for provider in core.BMRS}
    indexed: dict[tuple[str, str], Mapping[str, object]] = {}
    for record in records:
        coordinate = (str(record.get("cohort")), str(record.get("provider")))
        if coordinate in indexed:
            msg = f"Duplicate raw task coordinate: {coordinate}"
            raise ValueError(msg)
        indexed[coordinate] = record
    if set(indexed) != expected:
        msg = "Raw completion does not contain the exact cohort/provider grid."
        raise ValueError(msg)
    return indexed


def validate_raw_chain(
    *,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    cohorts: Sequence[str],
) -> dict[str, Any]:
    """Validate the complete raw receipt chain without parsing association values."""
    cohorts = tuple(cohorts)
    preparation.validate_input_root(input_root, cohorts)
    provider_manifest = preparation.validate_provider_root(provider_root, cohorts)
    input_path = input_root / "input_manifest.json"
    provider_path = provider_root / "provider_manifest.json"
    if not _record_matches(
        provider_manifest.get("input_manifest"),
        input_path,
        expected_path="input_manifest.json",
    ):
        msg = "Provider manifest is not bound to the supplied input manifest."
        raise ValueError(msg)

    run_path = run_root / "run_manifest.json"
    run_manifest = _load_json(run_path)
    if (
        run_manifest.get("schema_version") != runner.SCHEMA_VERSION
        or run_manifest.get("contract") != runner.RUN_CONTRACT
        or run_manifest.get("cohorts") != list(cohorts)
        or run_manifest.get("providers") != list(core.BMRS)
        or run_manifest.get("top_k") != 500
        or run_manifest.get("config_sha256") != runner._sha256(runner.CONFIG_PATH)  # noqa: SLF001
        or not _record_matches(
            run_manifest.get("provider_manifest"),
            provider_path,
            expected_path="provider_manifest.json",
        )
    ):
        msg = "Focused raw run manifest is not bound to the supplied inputs."
        raise ValueError(msg)

    completion_path = run_root / "completion_manifest.json"
    completion = _load_json(completion_path)
    tasks = completion.get("tasks")
    if (
        completion.get("schema_version") != runner.SCHEMA_VERSION
        or completion.get("contract") != runner.COMPLETION_CONTRACT
        or completion.get("cohorts") != list(cohorts)
        or not isinstance(tasks, list)
        or completion.get("task_count") != len(tasks)
        or not _record_matches(
            completion.get("run_manifest"),
            run_path,
            expected_path="run_manifest.json",
        )
    ):
        msg = "Focused completion manifest is not bound to its raw run."
        raise ValueError(msg)
    indexed = _require_complete_coordinates(tasks, cohorts)

    contract_records = []
    task_records = []
    for cohort in cohorts:
        contract_path = run_root / "contracts" / f"{cohort}.json"
        contract = _load_json(contract_path)
        if (
            contract.get("cohort") != cohort
            or contract.get("top_k") != 500
            or contract.get("focused_config_sha256")
            != run_manifest.get("config_sha256")
            or len(contract.get("features", [])) != 500
            or contract.get("pair_policy", {}).get("row_count") is None
        ):
            msg = f"Raw cohort contract is invalid: {cohort}"
            raise ValueError(msg)
        contract_sha256 = _sha256_bytes(_canonical_json(contract))
        contract_records.append(
            {
                **_file_record(contract_path, relative_to=run_root),
                "canonical_sha256": contract_sha256,
            },
        )
        for provider in core.BMRS:
            task_root = run_root / "tasks" / cohort / provider
            manifest = runner._validate_completed_task(  # noqa: SLF001
                task_root,
                contract_sha256=contract_sha256,
                cohort=cohort,
                provider=provider,
                pairwise_rows=int(contract["pair_policy"]["row_count"]),
            )
            manifest_path = task_root / "task_manifest.json"
            completion_record = indexed[(cohort, provider)].get("manifest")
            relative = f"tasks/{cohort}/{provider}/task_manifest.json"
            if not _record_matches(
                completion_record,
                manifest_path,
                expected_path=relative,
            ):
                msg = f"Completion task record changed: {cohort}/{provider}"
                raise ValueError(msg)
            task_records.append(
                {
                    "cohort": cohort,
                    "provider": provider,
                    "manifest": _file_record(manifest_path, relative_to=run_root),
                    "outputs": deepcopy(manifest["outputs"]),
                },
            )
    return {
        "input_manifest": _file_record(input_path, relative_to=input_root),
        "provider_manifest": _file_record(provider_path, relative_to=provider_root),
        "run_manifest": _file_record(run_path, relative_to=run_root),
        "completion_manifest": _file_record(
            completion_path,
            relative_to=run_root,
        ),
        "cohort_contracts": contract_records,
        "task_manifests": task_records,
    }


def _sanitize_contract_value(value: object, *, logical_path: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        sanitized = {}
        for key, child in value.items():
            if key in _MACHINE_FILE_KEYS:
                continue
            if isinstance(child, str) and (
                key in _MACHINE_PATH_KEYS or key.endswith(("_path", "_root"))
            ):
                sanitized[key] = "input-record://" + "/".join(logical_path)
            else:
                sanitized[key] = _sanitize_contract_value(
                    child,
                    logical_path=(*logical_path, str(key)),
                )
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_contract_value(child, logical_path=(*logical_path, str(index)))
            for index, child in enumerate(value)
        ]
    return value


def _contains_sample_identifier(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"sample_ids", "patient_ids"} or _contains_sample_identifier(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sample_identifier(child) for child in value)
    return isinstance(value, str) and _TCGA_SAMPLE_ID.search(value) is not None


def public_cohort_contract(contract_path: Path) -> dict[str, Any]:
    """Return a scientific contract projection without host paths or identifiers."""
    contract = _load_json(contract_path)
    cohort = str(contract.get("cohort", ""))
    if not cohort:
        msg = f"Cohort contract lacks a cohort: {contract_path}"
        raise ValueError(msg)
    raw = contract_path.read_bytes()
    projection = _sanitize_contract_value(contract, logical_path=(cohort,))
    serialized = _canonical_json(projection)
    if (
        b"/Users/" in serialized
        or b"/home/" in serialized
        or _contains_sample_identifier(projection)
    ):
        msg = f"Public cohort contract leaks a host path or identifier axis: {cohort}"
        raise ValueError(msg)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": PUBLIC_COHORT_CONTRACT,
        "cohort": cohort,
        "source_contract": {
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
            "canonical_sha256": _sha256_bytes(_canonical_json(contract)),
        },
        "projection": projection,
    }


def _git(*args: str, repository_root: Path, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def _commit(value: str, *, label: str) -> str:
    if _COMMIT.fullmatch(value) is None:
        msg = f"{label} must be a lowercase full Git commit."
        raise ValueError(msg)
    return value


def _git_blob_record(
    repository_root: Path,
    commit: str,
    relative: Path,
) -> dict[str, int | str]:
    content = _git(
        "show",
        f"{commit}:{relative.as_posix()}",
        repository_root=repository_root,
        text=False,
    )
    if not isinstance(content, bytes):
        raise TypeError
    return {
        "path": relative.as_posix(),
        "bytes": len(content),
        "sha256": _sha256_bytes(content),
    }


def _source_records(
    repository_root: Path,
    commit: str,
    paths: Sequence[Path],
) -> list[dict[str, int | str]]:
    return [
        _git_blob_record(repository_root, commit, path)
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]


def _git_blob_exists(repository_root: Path, commit: str, relative: Path) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{relative.as_posix()}"],
            cwd=repository_root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _runtime_record(executable: Path) -> dict[str, Any]:
    executable = executable.resolve(strict=True)
    if not executable.is_file() or executable.is_symlink():
        msg = f"Fit runtime is not a stable regular file: {executable}"
        raise ValueError(msg)
    if executable != Path(sys.executable).resolve(strict=True):
        msg = "Runtime attestation must execute under the supplied Python binary."
        raise ValueError(msg)
    packages = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        packages[distribution] = importlib.metadata.version(distribution)
    return {
        "scope": "post-run-runtime-readback-not-process-memory-attestation",
        "python": {
            "basename": executable.name,
            "bytes": executable.stat().st_size,
            "sha256": _sha256_path(executable),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "packages": packages,
        "thread_environment": dict(sorted(runner.THREAD_ENV.items())),
    }


def _source_boundary(
    *,
    repository_root: Path,
    fit_commit: str,
    release_commit: str,
) -> dict[str, Any]:
    fit_commit = _commit(fit_commit, label="fit commit")
    release_commit = _commit(release_commit, label="release commit")
    head = str(
        _git("rev-parse", "HEAD", repository_root=repository_root),
    ).strip()
    status = str(
        _git(
            "status",
            "--porcelain",
            repository_root=repository_root,
        ),
    )
    if head != release_commit or status:
        msg = "Release attestation requires the requested clean HEAD."
        raise ValueError(msg)
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", fit_commit, release_commit],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        msg = "Fit commit must be an ancestor of the release commit."
        raise ValueError(msg)
    for generated in _GENERATED_UNTRACKED_FIT_SOURCES:
        if _git_blob_exists(
            repository_root,
            fit_commit,
            generated,
        ) or _git_blob_exists(repository_root, release_commit, generated):
            msg = f"Generated fit-source exception became tracked: {generated}"
            raise ValueError(msg)
    fit_sources = _source_records(repository_root, fit_commit, FIT_SOURCE_FILES)
    release_fit_sources = _source_records(
        repository_root,
        release_commit,
        FIT_SOURCE_FILES,
    )
    if fit_sources != release_fit_sources:
        msg = "Raw-fit source bytes differ between fit and release commits."
        raise ValueError(msg)
    release_pipeline = _source_records(
        repository_root,
        release_commit,
        RELEASE_PIPELINE_FILES,
    )
    fit_tree = str(
        _git("rev-parse", f"{fit_commit}^{{tree}}", repository_root=repository_root),
    ).strip()
    release_tree = str(
        _git(
            "rev-parse",
            f"{release_commit}^{{tree}}",
            repository_root=repository_root,
        ),
    ).strip()
    return {
        "fit_source_commit": fit_commit,
        "fit_source_tree": fit_tree,
        "release_source_commit": release_commit,
        "release_source_tree": release_tree,
        "fit_is_ancestor_of_release": True,
        "fit_source_files": fit_sources,
        "excluded_generated_fit_sources": [
            {
                "path": path.as_posix(),
                "reason": (
                    "setuptools-scm generated module absent from both Git commits; "
                    "not independently attributable to the completed task runtime"
                ),
            }
            for path in sorted(
                _GENERATED_UNTRACKED_FIT_SOURCES,
                key=lambda item: item.as_posix(),
            )
        ],
        "raw_fit_sources_unchanged_at_release": True,
        "release_pipeline_files": release_pipeline,
        "repository": "raphael-group/dialect",
    }


def build_fit_attestation(  # noqa: PLR0913
    *,
    repository_root: Path,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    cohorts: Sequence[str],
    fit_commit: str,
    release_commit: str,
    runtime_executable: Path,
    output_path: Path,
) -> Path:
    """Build a write-once public attestation around the immutable raw fit."""
    if fit_commit != PRODUCTION_FIT_COMMIT:
        msg = f"Focused production fit commit must be {PRODUCTION_FIT_COMMIT}."
        raise ValueError(msg)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": FIT_ATTESTATION_CONTRACT,
        "scope": (
            "post-run-source-runtime-and-receipt-reconstruction; "
            "not-loaded-process-memory-attestation"
        ),
        "source": _source_boundary(
            repository_root=repository_root,
            fit_commit=fit_commit,
            release_commit=release_commit,
        ),
        "runtime": _runtime_record(runtime_executable),
        "raw_chain": validate_raw_chain(
            input_root=input_root,
            provider_root=provider_root,
            run_root=run_root,
            cohorts=cohorts,
        ),
        "privacy": {
            "raw_tumor_level_inputs_included": False,
            "sample_identifiers_included": False,
            "restricted_mutsig_source_included": False,
        },
    }
    _write_once(output_path, _canonical_json(payload) + b"\n")
    return output_path


def validate_fit_attestation(  # noqa: PLR0913
    attestation_path: Path,
    *,
    repository_root: Path,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    cohorts: Sequence[str],
    fit_commit: str,
    release_commit: str,
    runtime_executable: Path,
) -> dict[str, Any]:
    """Revalidate an attestation against every external provenance root."""
    if fit_commit != PRODUCTION_FIT_COMMIT:
        msg = f"Focused production fit commit must be {PRODUCTION_FIT_COMMIT}."
        raise ValueError(msg)
    payload = _load_json(attestation_path)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != FIT_ATTESTATION_CONTRACT
        or payload.get("source")
        != _source_boundary(
            repository_root=repository_root,
            fit_commit=fit_commit,
            release_commit=release_commit,
        )
        or payload.get("runtime") != _runtime_record(runtime_executable)
        or payload.get("raw_chain")
        != validate_raw_chain(
            input_root=input_root,
            provider_root=provider_root,
            run_root=run_root,
            cohorts=cohorts,
        )
        or payload.get("privacy")
        != {
            "raw_tumor_level_inputs_included": False,
            "sample_identifiers_included": False,
            "restricted_mutsig_source_included": False,
        }
    ):
        msg = "Fit source/runtime attestation is invalid or bound elsewhere."
        raise ValueError(msg)
    return payload


def validate_sha256(value: object, *, label: str) -> str:
    """Return one validated lowercase SHA-256 value."""
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        msg = f"{label} must be a lowercase SHA-256 digest."
        raise ValueError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--provider-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--fit-commit", default=PRODUCTION_FIT_COMMIT)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--runtime-executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    """Build one write-once fit attestation from validated receipt roots."""
    args = _parser().parse_args()
    output = build_fit_attestation(
        repository_root=args.repository_root.resolve(),
        input_root=args.input_root.resolve(),
        provider_root=args.provider_root.resolve(),
        run_root=args.run_root.resolve(),
        cohorts=preparation.TCGA_COHORTS,
        fit_commit=args.fit_commit,
        release_commit=args.release_commit,
        runtime_executable=args.runtime_executable,
        output_path=args.output.absolute(),
    )
    print(output)


if __name__ == "__main__":
    main()

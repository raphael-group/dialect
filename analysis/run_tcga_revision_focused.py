"""Run the focused matched K=500 TCGA grid.

This is the direct execution path for the coauthor-approved revision contract.  It
retains the existing native-support, corrected-LRT, deterministic fitting, atomic
publication, and validation code while replacing the unrelated human-attestation
state machine with one checked-in analysis configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from analysis import run_tcga_revision_k500 as core
from analysis.prepare_tcga_revision_focused import (
    CONFIG_PATH,
    PROVIDER_CONTRACT,
    THREAD_ENV,
    _load_config,
    _parse_cohorts,
    validate_provider_root,
)
from dialect.data.tcga import TCGA_COHORTS

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION: Final = "1.0.0"
RUN_CONTRACT: Final = "focused-matched-k500-grid-v1"
TASK_CONTRACT: Final = "focused-corrected-profile-lrt-task-v1"
COMPLETION_CONTRACT: Final = "focused-32x3-k500-completion-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        msg = f"Required regular file is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_provider_manifest(provider_root: Path) -> dict[str, Any]:
    manifest_path = provider_root / "provider_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != PROVIDER_CONTRACT
        or manifest.get("cohort_count") != len(manifest.get("cohorts", []))
    ):
        msg = "Provider root does not satisfy the focused input contract."
        raise ValueError(msg)
    return validate_provider_root(provider_root, tuple(manifest["cohorts"]))


def _paths(provider_root: Path, output_root: Path) -> core.RunPaths:
    return core.RunPaths(
        source_root=provider_root / "cohorts",
        mutsig_root=provider_root / "mutsig",
        output_root=output_root,
        focused_config_sha256=_sha256(CONFIG_PATH),
    )


def _contract_path(output_root: Path, cohort: str) -> Path:
    return output_root / "contracts" / f"{cohort}.json"


def _task_path(output_root: Path, cohort: str, provider: str) -> Path:
    return output_root / "tasks" / cohort / provider


def _ensure_run_root(
    provider_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> None:
    config = _load_config()
    provider_manifest = _load_provider_manifest(provider_root)
    if not set(cohorts) <= set(provider_manifest["cohorts"]):
        msg = "Requested cohorts are missing from the provider root."
        raise ValueError(msg)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "contracts").mkdir(exist_ok=True)
    (output_root / "tasks").mkdir(exist_ok=True)
    (output_root / "work").mkdir(exist_ok=True)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": RUN_CONTRACT,
        "config": _file_record(CONFIG_PATH, relative_to=CONFIG_PATH.parent.parent),
        "config_sha256": _sha256(CONFIG_PATH),
        "provider_manifest": _file_record(
            provider_root / "provider_manifest.json",
            relative_to=provider_root,
        ),
        "cohorts": list(cohorts),
        "providers": config["analysis"]["providers"],
        "top_k": config["analysis"]["top_k"],
        "resources": config["execution"],
    }
    content = _canonical_json(run_manifest) + b"\n"
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != content:
            msg = "Existing run root is bound to a different focused contract."
            raise ValueError(msg)
    else:
        _write_atomic(manifest_path, content)


def _ensure_cohort_contract(
    paths: core.RunPaths,
    cohort: str,
) -> dict[str, Any]:
    contract = core.build_cohort_contract(paths, cohort, top_k=500)
    content = _canonical_json(contract) + b"\n"
    path = _contract_path(paths.output_root, cohort)
    if path.exists():
        if path.read_bytes() != content:
            msg = f"Frozen cohort contract changed: {cohort}"
            raise ValueError(msg)
    else:
        _write_atomic(path, content)
    return contract


def _validate_completed_task(
    task_root: Path,
    *,
    contract_sha256: str,
    cohort: str,
    provider: str,
    pairwise_rows: int,
) -> dict[str, Any]:
    manifest_path = task_root / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"pairwise_interaction_results.csv", "single_gene_results.csv"}
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != TASK_CONTRACT
        or manifest.get("cohort") != cohort
        or manifest.get("provider") != provider
        or manifest.get("top_k") != 500
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("single_gene_rows") != 500
        or manifest.get("pairwise_rows") != pairwise_rows
        or set(manifest.get("outputs", {})) != expected
        or {path.name for path in task_root.iterdir()}
        != {*expected, "task_manifest.json"}
    ):
        msg = f"Completed task manifest is invalid: {task_root}"
        raise ValueError(msg)
    for name in expected:
        record = manifest["outputs"][name]
        path = task_root / name
        if (
            record.get("path") != name
            or not path.is_file()
            or path.is_symlink()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            msg = f"Completed task output changed: {path}"
            raise ValueError(msg)
    core._validate_task_resource_usage(manifest, task_root)  # noqa: SLF001
    return manifest


def _resource_usage(started: float) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    normalized = core._task_resource_usage(started)  # noqa: SLF001
    normalized.update(
        {
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
        },
    )
    return normalized


def run_task(
    *,
    provider_root: Path,
    output_root: Path,
    cohort: str,
    provider: str,
    nice_increment: int,
) -> str:
    """Fit, validate, and atomically publish one cohort/provider task."""
    if cohort not in TCGA_COHORTS or provider not in core.BMRS:
        msg = f"Invalid task coordinates: {cohort}/{provider}"
        raise ValueError(msg)
    paths = _paths(provider_root, output_root)
    contract = _ensure_cohort_contract(paths, cohort)
    contract_sha256 = hashlib.sha256(_canonical_json(contract)).hexdigest()
    final_root = _task_path(output_root, cohort, provider)
    if final_root.exists():
        _validate_completed_task(
            final_root,
            contract_sha256=contract_sha256,
            cohort=cohort,
            provider=provider,
            pairwise_rows=int(contract["pair_policy"]["row_count"]),
        )
        return "already-complete"

    started = time.monotonic()
    if nice_increment:
        os.nice(nice_increment)
    work_parent = output_root / "work" / cohort
    work_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{provider}.", dir=work_parent))
    try:
        counts, pmfs = core._load_frozen_scientific_inputs(  # noqa: SLF001
            contract,
            provider,
        )
        features = list(contract["features"])
        counts = counts.loc[:, features]
        genes = core._build_genes(counts, features, pmfs)  # noqa: SLF001
        core.estimate_pi_for_each_gene(genes.values())
        single_path = staging_root / "single_gene_results.csv"
        pairwise_path = staging_root / "pairwise_interaction_results.csv"
        _write_atomic(
            single_path,
            core._single_gene_results_bytes(list(genes.values())),  # noqa: SLF001
        )
        written = core._write_pairwise_results(  # noqa: SLF001
            pairwise_path,
            genes,
            features,
        )
        if written != contract["pair_policy"]["row_count"]:
            msg = f"Pair writer emitted an incomplete family: {cohort}/{provider}"
            raise RuntimeError(msg)  # noqa: TRY301
        single_raw = single_path.read_bytes()
        pairwise_raw = pairwise_path.read_bytes()
        single_rows = core._validate_single_gene_output(  # noqa: SLF001
            single_raw,
            contract,
            counts,
            pmfs,
            genes,
        )
        pairwise_rows = core._validate_pairwise_output_impl(  # noqa: SLF001
            pairwise_raw,
            contract,
            counts,
            genes,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "contract": TASK_CONTRACT,
            "cohort": cohort,
            "provider": provider,
            "top_k": 500,
            "contract_sha256": contract_sha256,
            "config_sha256": _sha256(CONFIG_PATH),
            "single_gene_rows": single_rows,
            "pairwise_rows": pairwise_rows,
            "resource_usage": _resource_usage(started),
            "outputs": {
                single_path.name: _file_record(single_path, relative_to=staging_root),
                pairwise_path.name: _file_record(
                    pairwise_path,
                    relative_to=staging_root,
                ),
            },
        }
        _write_atomic(
            staging_root / "task_manifest.json",
            _canonical_json(manifest) + b"\n",
        )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            msg = f"Task output appeared during fitting: {final_root}"
            raise FileExistsError(msg)  # noqa: TRY301
        staging_root.replace(final_root)
        _validate_completed_task(
            final_root,
            contract_sha256=contract_sha256,
            cohort=cohort,
            provider=provider,
            pairwise_rows=int(contract["pair_policy"]["row_count"]),
        )
    except BaseException as error:
        if staging_root.exists():
            failure = {
                "schema_version": SCHEMA_VERSION,
                "contract": "focused-failed-task-evidence-v1",
                "cohort": cohort,
                "provider": provider,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_atomic(
                staging_root / "failure.json",
                _canonical_json(failure) + b"\n",
            )
            print(f"retained failed task evidence: {staging_root}", flush=True)
        raise
    return "complete"


def _task_command(
    *,
    provider_root: Path,
    output_root: Path,
    cohort: str,
    provider: str,
    nice_increment: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "analysis.run_tcga_revision_focused",
        "--provider-root",
        provider_root.as_posix(),
        "--output-root",
        output_root.as_posix(),
        "--internal-cohort",
        cohort,
        "--internal-provider",
        provider,
        "--nice",
        str(nice_increment),
    ]


def _run_batch(
    tasks: Sequence[tuple[str, str]],
    *,
    provider_root: Path,
    output_root: Path,
    jobs: int,
    nice_increment: int,
) -> None:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)

    def invoke(task: tuple[str, str]) -> None:
        cohort, provider = task
        subprocess.run(
            _task_command(
                provider_root=provider_root,
                output_root=output_root,
                cohort=cohort,
                provider=provider,
                nice_increment=nice_increment,
            ),
            check=True,
            env=environment,
        )

    if jobs == 1:
        for cohort, provider in tasks:
            invoke((cohort, provider))
            print(f"complete {cohort}/{provider}", flush=True)
        return

    remaining = iter(tasks)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures: dict[Any, tuple[str, str]] = {}
        for _ in range(jobs):
            task = next(remaining, None)
            if task is None:
                break
            futures[executor.submit(invoke, task)] = task
        while futures:
            future = next(as_completed(futures))
            cohort, provider = futures.pop(future)
            try:
                future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            print(f"complete {cohort}/{provider}", flush=True)
            task = next(remaining, None)
            if task is not None:
                futures[executor.submit(invoke, task)] = task


def _status(
    *,
    provider_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> dict[str, Any]:
    paths = _paths(provider_root, output_root)
    states = []
    completed = 0
    for cohort in cohorts:
        contract = _ensure_cohort_contract(paths, cohort)
        contract_sha256 = hashlib.sha256(_canonical_json(contract)).hexdigest()
        for provider in core.BMRS:
            task_root = _task_path(output_root, cohort, provider)
            state = "pending"
            if task_root.exists():
                _validate_completed_task(
                    task_root,
                    contract_sha256=contract_sha256,
                    cohort=cohort,
                    provider=provider,
                    pairwise_rows=int(contract["pair_policy"]["row_count"]),
                )
                state = "complete"
                completed += 1
            states.append({"cohort": cohort, "provider": provider, "state": state})
    return {
        "completed": completed,
        "total": len(cohorts) * len(core.BMRS),
        "tasks": states,
    }


def _finalize_completion(
    *,
    provider_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> Path:
    status = _status(
        provider_root=provider_root,
        output_root=output_root,
        cohorts=cohorts,
    )
    if status["completed"] != status["total"]:
        msg = "Cannot finalize an incomplete focused K=500 grid."
        raise RuntimeError(msg)
    records = []
    for task in status["tasks"]:
        task_root = _task_path(
            output_root,
            str(task["cohort"]),
            str(task["provider"]),
        )
        records.append(
            {
                "cohort": task["cohort"],
                "provider": task["provider"],
                "manifest": _file_record(
                    task_root / "task_manifest.json",
                    relative_to=output_root,
                ),
            },
        )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "contract": COMPLETION_CONTRACT,
        "config_sha256": _sha256(CONFIG_PATH),
        "run_manifest": _file_record(
            output_root / "run_manifest.json",
            relative_to=output_root,
        ),
        "cohorts": list(cohorts),
        "task_count": len(records),
        "tasks": records,
    }
    path = output_root / "completion_manifest.json"
    content = _canonical_json(completion) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            msg = "Completion manifest differs from the validated task grid."
            raise ValueError(msg)
    else:
        _write_atomic(path, content)
    return path


def orchestrate(  # noqa: PLR0913
    *,
    provider_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
    jobs: int,
    nice_increment: int,
    preflight_only: bool,
) -> None:
    """Preflight, resume, and complete the requested matched grid."""
    config = _load_config()
    if not 1 <= jobs <= config["execution"]["max_general_jobs"]:
        msg = "--jobs exceeds the focused revision resource contract."
        raise ValueError(msg)
    if nice_increment != config["execution"]["nice_increment"]:
        msg = "--nice differs from the focused revision resource contract."
        raise ValueError(msg)
    _ensure_run_root(provider_root, output_root, cohorts)
    paths = _paths(provider_root, output_root)
    for cohort in cohorts:
        _ensure_cohort_contract(paths, cohort)
        print(f"preflight {cohort}: valid matched K=500 contract", flush=True)
    if preflight_only:
        return

    canary = [("CHOL", provider) for provider in core.BMRS if "CHOL" in cohorts]
    if canary:
        _run_batch(
            canary,
            provider_root=provider_root,
            output_root=output_root,
            jobs=1,
            nice_increment=nice_increment,
        )
    others = [cohort for cohort in cohorts if cohort != "CHOL"]
    cohort_level = [
        (cohort, provider)
        for cohort in others
        for provider in ("cbase", "dig")
    ]
    mutsig = [(cohort, "mutsig") for cohort in others]
    _run_batch(
        cohort_level,
        provider_root=provider_root,
        output_root=output_root,
        jobs=jobs,
        nice_increment=nice_increment,
    )
    _run_batch(
        mutsig,
        provider_root=provider_root,
        output_root=output_root,
        jobs=1,
        nice_increment=nice_increment,
    )
    completion = _finalize_completion(
        provider_root=provider_root,
        output_root=output_root,
        cohorts=cohorts,
    )
    print(f"completion {completion} sha256={_sha256(completion)}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohorts")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--internal-cohort", choices=TCGA_COHORTS)
    parser.add_argument("--internal-provider", choices=core.BMRS)
    return parser


def main() -> None:
    """Run one internal task or the focused grid orchestrator."""
    args = _parser().parse_args()
    provider_root = args.provider_root.resolve()
    output_root = args.output_root.absolute()
    if (args.internal_cohort is None) != (args.internal_provider is None):
        msg = "Internal cohort and provider must be supplied together."
        raise ValueError(msg)
    if args.internal_cohort is not None:
        result = run_task(
            provider_root=provider_root,
            output_root=output_root,
            cohort=args.internal_cohort,
            provider=args.internal_provider,
            nice_increment=args.nice,
        )
        print(result)
        return
    cohorts = _parse_cohorts(args.cohorts)
    _ensure_run_root(provider_root, output_root, cohorts)
    if args.status:
        print(
            json.dumps(
                _status(
                    provider_root=provider_root,
                    output_root=output_root,
                    cohorts=cohorts,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return
    orchestrate(
        provider_root=provider_root,
        output_root=output_root,
        cohorts=cohorts,
        jobs=args.jobs,
        nice_increment=args.nice,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()

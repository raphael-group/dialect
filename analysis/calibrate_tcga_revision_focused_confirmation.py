"""Confirm only failed focused-calibration endpoints with fresh null replicates.

This module implements the frozen second phase of a result-blind two-stage
calibration protocol.  Stage 1 fully revalidates the existing 10,000-replicate
v4 calibration tree, recomputes all 2,048 primary MutSig Clopper--Pearson bounds
at ``gamma1 / 2,048``, and selects every failure.  Stage 2 generates 100,000
fresh, independent fitted-null replicates for each selected endpoint and tests
it at ``gamma2 / M``, where ``M`` is the frozen number selected at stage 1.
The stages are never pooled, there is no third stage, and no observed pairwise
association statistic is parsed or inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis import calibrate_tcga_revision_focused as stage1_calibration
from analysis import run_tcga_revision_k500 as core

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
CONFIG_PATH: Final = Path(__file__).with_name(
    "tcga_revision_calibration_confirmation_config.json",
)
RUN_CONTRACT: Final = "focused-null-calibration-confirmation-run-v1"
TASK_CONTRACT: Final = "focused-null-calibration-confirmation-endpoint-v1"
SUMMARY_CONTRACT: Final = "focused-null-calibration-confirmation-summary-v1"
RUN_MANIFEST_NAME: Final = "run_manifest.json"
TASK_MANIFEST_NAME: Final = "task_manifest.json"
TASK_DATA_NAME: Final = "confirmation_arrays.npz"
SUMMARY_NAME: Final = "confirmation_summary.json"
FINAL_TABLE_NAME: Final = "confirmation_final_endpoints.csv"
FINAL_CELLS_NAME: Final = FINAL_TABLE_NAME
ENDPOINT_ACCEPTED: Final = "pass"
ENDPOINT_REJECTED: Final = "fail"
STAGE2_NOT_APPLICABLE: Final = "not-applicable"
FINAL_TABLE_COLUMNS: Final = (
    "endpoint_id",
    "cohort",
    "provider",
    "sentinel_pair_index",
    "threshold",
    "acceptance_upper_bound",
    "stage1_events",
    "stage1_trials",
    "stage1_rate",
    "stage1_reportable_trials",
    "stage1_nonreportable_trials",
    "stage1_familywise_error",
    "stage1_endpoint_count",
    "stage1_endpoint_error",
    "stage1_clopper_pearson_upper_bound",
    "stage1_pass",
    "selected_for_stage2",
    "stage2_events",
    "stage2_trials",
    "stage2_rate",
    "stage2_reportable_trials",
    "stage2_nonreportable_trials",
    "stage2_familywise_error",
    "stage2_endpoint_count",
    "stage2_endpoint_error",
    "stage2_clopper_pearson_upper_bound",
    "stage2_pass",
    "final_evidence_stage",
    "final_clopper_pearson_upper_bound",
    "final_pass",
)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One source-calibration endpoint selected for confirmation."""

    cohort: str
    provider: str
    sentinel_pair_index: int
    threshold: float

    @property
    def endpoint_id(self) -> str:
        """Return a stable filesystem-safe coordinate identifier."""
        alpha = round(self.threshold * 100)
        return (
            f"{self.cohort}__{self.provider}__sentinel-"
            f"{self.sentinel_pair_index:02d}__alpha-{alpha:03d}"
        )


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


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        msg = f"Required regular file is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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


def _result_blindness_receipt() -> dict[str, Any]:
    return {
        "observed_pair_statistics_parsed_or_inspected": False,
        "production_pairwise_files_integrity_hashed": True,
        "production_pairwise_hash_use": "run-integrity-validation-only",
        "source_calibration_arrays_recomputed": True,
        "stage1_and_stage2_replicates_pooled": False,
        "third_stage_permitted": False,
    }


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = config.get("source_calibration", {})
    gate = config.get("two_stage_gate", {})
    stage1 = gate.get("stage1", {})
    stage2 = gate.get("stage2", {})
    simulation = config.get("simulation", {})
    resources = config.get("resources", {})
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("contract")
        != "focused-result-blind-null-calibration-confirmation-v1"
        or config.get("seed") != 20260904
        or source
        != {
            "contract": stage1_calibration.SUMMARY_CONTRACT,
            "provider": "mutsig",
            "endpoint_unit": "cohort-sentinel-pair-alpha",
            "endpoint_count": 2_048,
            "replicates_per_endpoint": 10_000,
            "sentinel_pair_count": 32,
            "alphas": [0.01, 0.05],
        }
        or gate.get("total_familywise_error") != 0.05
        or stage1
        != {
            "familywise_error": 0.025,
            "endpoint_count": 2_048,
            "selection_rule": (
                "select-every-endpoint-whose-recomputed-one-sided-clopper-"
                "pearson-upper-bound-exceeds-its-acceptance-bound"
            ),
        }
        or stage2
        != {
            "familywise_error": 0.025,
            "replicates_per_selected_endpoint": 100_000,
            "endpoint_count_rule": "number-of-stage1-selected-endpoints",
            "decision_rule": (
                "one-sided-clopper-pearson-upper-bound-at-gamma2-divided-by-"
                "selected-endpoint-count"
            ),
        }
        or gate.get("acceptance_upper_bounds") != {"0.01": 0.02, "0.05": 0.07}
        or gate.get("overall_rule")
        != (
            "all-unselected-endpoints-pass-stage1-and-all-selected-endpoints-"
            "pass-stage2"
        )
        or simulation
        != {
            "generator": (
                "fitted-independent-dialect-marginals-from-the-source-"
                "calibration-cell"
            ),
            "sentinel_pair_source": (
                "same-indexed-pair-from-source-32-disjoint-K500-axis"
            ),
            "replicate_rng": (
                "sha256-confirmation-root-seed-endpoint-coordinate-and-shard-"
                "index-v1"
            ),
            "replicate_shards": 5,
            "replicate_shard_rule": (
                "five-equal-contiguous-20000-replicate-shards-concatenated-in-"
                "shard-index-order"
            ),
            "fit_kernel": stage1_calibration.FIT_KERNEL_CONTRACT,
            "replicate_chunk_rule": stage1_calibration.REPLICATE_CHUNK_RULE,
            "effective_p_policy": (
                "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
            ),
            "pooling_with_stage1": False,
            "additional_stage_after_stage2": False,
        }
        or resources
        != {
            "observed_logical_cpus": 14,
            "max_outer_endpoints": 1,
            "fit_workers_per_endpoint": 5,
            "max_total_fit_workers": 5,
            "blas_threads_per_fit_worker": 1,
            "nice_increment": 10,
            "overwrite_outputs": False,
        }
    ):
        msg = "Confirmation configuration violates its frozen v1 contract."
        raise ValueError(msg)
    if not math.isclose(
        float(stage1["familywise_error"]) + float(stage2["familywise_error"]),
        float(gate["total_familywise_error"]),
    ):
        msg = "Confirmation familywise-error allocation is invalid."
        raise ValueError(msg)
    return config


def _resource_observation(config: Mapping[str, Any]) -> dict[str, Any]:
    resources = config["resources"]
    logical_cpus = int(resources["observed_logical_cpus"])
    workers = int(resources["max_total_fit_workers"])
    half_limit = logical_cpus // 2
    if (
        logical_cpus < 2
        or int(resources["max_outer_endpoints"]) != 1
        or int(resources["fit_workers_per_endpoint"]) != workers
        or workers < 1
        or workers > half_limit
        or int(resources["blas_threads_per_fit_worker"]) != 1
    ):
        msg = "Confirmation runtime violates its frozen half-machine contract."
        raise RuntimeError(msg)
    return {
        "logical_cpus_observed": logical_cpus,
        "logical_cpu_source": "os.cpu_count()",
        "half_logical_cpu_limit": half_limit,
        "scheduled_fit_worker_limit": workers,
        "half_machine_limit_satisfied": True,
    }


def _preflight_runtime_resources(config: Mapping[str, Any]) -> dict[str, Any]:
    observation = _resource_observation(config)
    if os.cpu_count() != observation["logical_cpus_observed"]:
        msg = (
            "Confirmation execution host differs from the frozen 14-logical-"
            "CPU contract."
        )
        raise RuntimeError(msg)
    return observation


def _endpoint_from_row(row: Mapping[str, Any]) -> Endpoint:
    return Endpoint(
        cohort=str(row["cohort"]),
        provider=str(row["provider"]),
        sentinel_pair_index=int(row["sentinel_pair_index"]),
        threshold=float(row["threshold"]),
    )


def _stage1_frame(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fully validate v4, then recompute every stricter stage-1 bound."""
    source_summary = stage1_calibration.validate_summary(
        calibration_root,
        run_root=run_root,
        provider_root=provider_root,
    )
    source_config = stage1_calibration._load_config()  # noqa: SLF001
    source_frame, _, _ = stage1_calibration._summary_frame(  # noqa: SLF001
        output_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        config=source_config,
    )
    gate_rows = stage1_calibration._validated_gate_rows(  # noqa: SLF001
        source_frame,
        source_config,
    ).copy()
    source = config["source_calibration"]
    stage1 = config["two_stage_gate"]["stage1"]
    endpoint_count = int(stage1["endpoint_count"])
    endpoint_error = float(stage1["familywise_error"]) / endpoint_count
    if (
        source_summary.get("contract") != source["contract"]
        or source_summary.get("primary_gate_endpoint_count") != endpoint_count
        or len(gate_rows) != endpoint_count
        or not gate_rows["provider"].eq(source["provider"]).all()
        or not gate_rows["trials"].eq(source["replicates_per_endpoint"]).all()
        or set(gate_rows["threshold"].astype(float)) != set(source["alphas"])
    ):
        msg = "Source calibration does not match the confirmation protocol."
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    for source_row in gate_rows.to_dict(orient="records"):
        endpoint = _endpoint_from_row(source_row)
        upper = stage1_calibration._clopper_pearson_upper_bound(  # noqa: SLF001
            successes=int(source_row["events"]),
            trials=int(source_row["trials"]),
            endpoint_error=endpoint_error,
        )
        acceptance = float(
            config["two_stage_gate"]["acceptance_upper_bounds"][
                f"{endpoint.threshold:.2f}"
            ],
        )
        passed = upper <= acceptance
        rows.append(
            {
                "endpoint_id": endpoint.endpoint_id,
                "cohort": endpoint.cohort,
                "provider": endpoint.provider,
                "sentinel_pair_index": endpoint.sentinel_pair_index,
                "threshold": endpoint.threshold,
                "acceptance_upper_bound": acceptance,
                "stage1_events": int(source_row["events"]),
                "stage1_trials": int(source_row["trials"]),
                "stage1_rate": int(source_row["events"])
                / int(source_row["trials"]),
                "stage1_reportable_trials": int(source_row["reportable_trials"]),
                "stage1_nonreportable_trials": int(
                    source_row["nonreportable_trials"],
                ),
                "stage1_familywise_error": float(stage1["familywise_error"]),
                "stage1_endpoint_count": endpoint_count,
                "stage1_endpoint_error": endpoint_error,
                "stage1_clopper_pearson_upper_bound": upper,
                "stage1_pass": (
                    ENDPOINT_ACCEPTED if passed else ENDPOINT_REJECTED
                ),
                "selected_for_stage2": not passed,
            },
        )
    frame = pd.DataFrame(rows)
    if (
        len(frame) != endpoint_count
        or frame["endpoint_id"].duplicated().any()
        or not (
            frame["stage1_reportable_trials"]
            + frame["stage1_nonreportable_trials"]
        )
        .eq(frame["stage1_trials"])
        .all()
    ):
        msg = "Recomputed stage-1 endpoint inventory is invalid."
        raise ValueError(msg)
    return frame, source_summary


def _selected_endpoints(frame: pd.DataFrame) -> tuple[Endpoint, ...]:
    selected = frame.loc[frame["selected_for_stage2"]]
    return tuple(_endpoint_from_row(row) for row in selected.to_dict(orient="records"))


def _endpoint_payload(endpoint: Endpoint) -> dict[str, Any]:
    return {
        "endpoint_id": endpoint.endpoint_id,
        "cohort": endpoint.cohort,
        "provider": endpoint.provider,
        "sentinel_pair_index": endpoint.sentinel_pair_index,
        "threshold": endpoint.threshold,
    }


def _endpoint_seed(root_seed: int, endpoint: Endpoint) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            [
                root_seed,
                "focused-calibration-confirmation-endpoint-v1",
                endpoint.cohort,
                endpoint.provider,
                endpoint.sentinel_pair_index,
                f"{endpoint.threshold:.2f}",
            ],
        ),
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _shard_seed(endpoint_seed: int, shard_index: int) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            [endpoint_seed, "confirmation-independent-replicate-shard-v1", shard_index],
        ),
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _source_records(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
) -> dict[str, Any]:
    return {
        "calibration_config": _file_record(
            stage1_calibration.CONFIG_PATH,
            relative_to=stage1_calibration.CONFIG_PATH.parent.parent,
        ),
        "calibration_run_manifest": _file_record(
            calibration_root / stage1_calibration.RUN_MANIFEST_NAME,
            relative_to=calibration_root,
        ),
        "calibration_table": _file_record(
            calibration_root / stage1_calibration.SUMMARY_TABLE_NAME,
            relative_to=calibration_root,
        ),
        "calibration_summary": _file_record(
            calibration_root / stage1_calibration.SUMMARY_NAME,
            relative_to=calibration_root,
        ),
        "production_completion": _file_record(
            run_root / "completion_manifest.json",
            relative_to=run_root,
        ),
        "provider_manifest": _file_record(
            provider_root / "provider_manifest.json",
            relative_to=provider_root,
        ),
    }


def _run_manifest_payload(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stage1_frame, source_summary = _stage1_frame(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    selected = _selected_endpoints(stage1_frame)
    stage1 = config["two_stage_gate"]["stage1"]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": RUN_CONTRACT,
        "config": _file_record(CONFIG_PATH, relative_to=CONFIG_PATH.parent.parent),
        "source_records": _source_records(
            calibration_root=calibration_root,
            run_root=run_root,
            provider_root=provider_root,
        ),
        "source_calibration_overall_gate_pass": source_summary["overall_gate_pass"],
        "stage1": {
            "familywise_error": stage1["familywise_error"],
            "endpoint_count": stage1["endpoint_count"],
            "endpoint_error": float(stage1["familywise_error"])
            / int(stage1["endpoint_count"]),
            "selection_rule": stage1["selection_rule"],
            "selected_endpoint_count": len(selected),
            "selected_endpoints": [
                _endpoint_payload(endpoint) for endpoint in selected
            ],
        },
        "stage2": dict(config["two_stage_gate"]["stage2"]),
        "resource_contract": dict(config["resources"]),
        "runtime_resource_observation": _resource_observation(config),
        "thread_environment": dict(stage1_calibration.THREAD_ENV),
        "result_blindness": _result_blindness_receipt(),
    }


def _ensure_run_root(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_config()
    _preflight_runtime_resources(config)
    payload = _run_manifest_payload(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tasks").mkdir(exist_ok=True)
    path = output_root / RUN_MANIFEST_NAME
    content = _canonical_json(payload) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            msg = "Confirmation run root is bound to different source evidence."
            raise ValueError(msg)
    else:
        _write_atomic(path, content)
    return config, payload


def _validate_run_manifest(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _run_manifest_payload(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    path = output_root / RUN_MANIFEST_NAME
    if path.read_bytes() != _canonical_json(expected) + b"\n":
        msg = "Confirmation run manifest or its source evidence is invalid."
        raise ValueError(msg)
    return expected


def _manifest_endpoints(manifest: Mapping[str, Any]) -> tuple[Endpoint, ...]:
    raw = manifest.get("stage1", {}).get("selected_endpoints", [])
    if not isinstance(raw, list):
        msg = "Confirmation run manifest selected endpoints are invalid."
        raise TypeError(msg)
    endpoints = tuple(_endpoint_from_row(item) for item in raw)
    if (
        len(endpoints) != manifest.get("stage1", {}).get("selected_endpoint_count")
        or len({endpoint.endpoint_id for endpoint in endpoints}) != len(endpoints)
        or [_endpoint_payload(endpoint) for endpoint in endpoints] != raw
    ):
        msg = "Confirmation run manifest selected endpoint inventory is invalid."
        raise ValueError(msg)
    return endpoints


def _find_endpoint(manifest: Mapping[str, Any], endpoint_id: str) -> Endpoint:
    matches = [
        endpoint
        for endpoint in _manifest_endpoints(manifest)
        if endpoint.endpoint_id == endpoint_id
    ]
    if len(matches) != 1:
        msg = "Confirmation task endpoint is outside the frozen stage-1 selection."
        raise ValueError(msg)
    return matches[0]


def _task_root(output_root: Path, endpoint: Endpoint) -> Path:
    return output_root / "tasks" / endpoint.endpoint_id


def _worker_topology(config: Mapping[str, Any]) -> dict[str, Any]:
    resources = config["resources"]
    return {
        **_resource_observation(config),
        "max_outer_endpoints": int(resources["max_outer_endpoints"]),
        "fit_workers": int(resources["fit_workers_per_endpoint"]),
        "fit_execution": "spawned-independent-replicate-shards",
        "max_total_fit_workers": int(resources["max_total_fit_workers"]),
        "blas_threads_per_fit_worker": int(
            resources["blas_threads_per_fit_worker"],
        ),
        "thread_environment": dict(stage1_calibration.THREAD_ENV),
    }


def _endpoint_shard_worker(
    task: tuple[stage1_calibration.Cell, np.ndarray, int, int, int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    cell, sentinel_pair, endpoint_seed, shard_index, replicates = task
    os.environ.update(stage1_calibration.THREAD_ENV)
    with threadpool_limits(limits=1):
        likelihood_ratio, reportable, fallback = (
            stage1_calibration._simulate_null_arrays(  # noqa: SLF001
                cell=cell,
                sentinel_pairs=sentinel_pair,
                replicates=replicates,
                cell_seed=_shard_seed(endpoint_seed, shard_index),
                workers=1,
            )
        )
    return (
        shard_index,
        likelihood_ratio[:, 0],
        reportable[:, 0],
        fallback[:, 0],
    )


def _simulate_endpoint_arrays(
    *,
    cell: stage1_calibration.Cell,
    sentinel_pair: np.ndarray,
    endpoint_seed: int,
    replicates: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        sentinel_pair.shape != (1, 2)
        or replicates < 1
        or workers < 1
        or replicates % workers != 0
    ):
        msg = "Confirmation endpoint simulation dimensions are invalid."
        raise ValueError(msg)
    shard_size = replicates // workers
    tasks = tuple(
        (cell, sentinel_pair, endpoint_seed, shard_index, shard_size)
        for shard_index in range(workers)
    )
    if workers == 1:
        results = [_endpoint_shard_worker(tasks[0])]
    else:
        with multiprocessing.get_context("spawn").Pool(processes=workers) as pool:
            results = pool.map(_endpoint_shard_worker, tasks, chunksize=1)
    if [result[0] for result in results] != list(range(workers)):
        msg = "Confirmation replicate shards returned an invalid shard axis."
        raise RuntimeError(msg)
    return tuple(  # type: ignore[return-value]
        np.concatenate([result[array_index] for result in results])
        for array_index in (1, 2, 3)
    )


def _validate_task_resource_usage(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    task_root: Path,
) -> None:
    if manifest.get("worker_topology") != _worker_topology(config):
        msg = "Confirmation task worker topology violates its resource contract."
        raise ValueError(msg)
    core._validate_task_resource_usage(manifest, task_root)  # noqa: SLF001
    usage = manifest.get("resource_usage")
    if not isinstance(usage, dict):
        msg = "Confirmation task lacks a resource-usage receipt."
        raise TypeError(msg)
    self_record = usage.get("self")
    child_record = usage.get("terminated_children")
    stage1_calibration._validate_rusage_record(  # noqa: SLF001
        self_record,
        children=False,
        require_positive_rss=True,
    )
    stage1_calibration._validate_rusage_record(  # noqa: SLF001
        child_record,
        children=True,
        require_positive_rss=True,
    )
    if (
        not isinstance(self_record, dict)
        or usage.get("user_cpu_seconds") != self_record.get("user_cpu_seconds")
        or usage.get("system_cpu_seconds") != self_record.get("system_cpu_seconds")
        or usage.get("peak_rss", {}).get("bytes")
        != self_record.get("peak_rss", {}).get("bytes")
    ):
        msg = "Confirmation task self-usage fields disagree with their receipt."
        raise ValueError(msg)


def _validate_task(  # noqa: PLR0913
    task_root: Path,
    *,
    endpoint: Endpoint,
    run_manifest_sha256: str,
    run_root: Path,
    provider_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = task_root / TASK_MANIFEST_NAME
    data_path = task_root / TASK_DATA_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage2 = config["two_stage_gate"]["stage2"]
    simulation = config["simulation"]
    replicates = int(stage2["replicates_per_selected_endpoint"])
    cell = stage1_calibration._load_cell(  # noqa: SLF001
        run_root,
        provider_root,
        endpoint.cohort,
        endpoint.provider,
    )
    all_pairs = stage1_calibration._sentinel_pairs(  # noqa: SLF001
        cell.features,
        int(config["source_calibration"]["sentinel_pair_count"]),
    )
    expected_pair = all_pairs[endpoint.sentinel_pair_index]
    expected_seed = _endpoint_seed(int(config["seed"]), endpoint)
    expected_source_manifest, expected_single_gene = (
        stage1_calibration._validate_single_gene_source(  # noqa: SLF001
            run_root,
            endpoint.cohort,
            endpoint.provider,
            json.loads(
                (run_root / "contracts" / f"{endpoint.cohort}.json").read_text(
                    encoding="utf-8",
                ),
            ),
        )
    )
    fit_kernel = manifest.get("fit_kernel")
    fallback_count = (
        fit_kernel.get("scalar_fallback_count")
        if isinstance(fit_kernel, dict)
        else None
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != TASK_CONTRACT
        or manifest.get("endpoint") != _endpoint_payload(endpoint)
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("run_manifest_sha256") != run_manifest_sha256
        or manifest.get("seed") != expected_seed
        or manifest.get("replicates") != replicates
        or manifest.get("replicate_rng") != simulation["replicate_rng"]
        or manifest.get("replicate_shards") != simulation["replicate_shards"]
        or manifest.get("replicate_shard_rule") != simulation["replicate_shard_rule"]
        or manifest.get("shard_seeds")
        != [
            _shard_seed(expected_seed, index)
            for index in range(int(simulation["replicate_shards"]))
        ]
        or manifest.get("sentinel_pair") != expected_pair.tolist()
        or manifest.get("sentinel_features")
        != [cell.features[int(index)] for index in expected_pair]
        or manifest.get("source_task_manifest") != expected_source_manifest
        or manifest.get("single_gene_input") != expected_single_gene
        or not isinstance(fit_kernel, dict)
        or not isinstance(fallback_count, int)
        or isinstance(fallback_count, bool)
        or fit_kernel
        != {
            "contract": simulation["fit_kernel"],
            "replicate_chunk_rule": simulation["replicate_chunk_rule"],
            "replicate_chunk_size": stage1_calibration._replicate_chunk_size(  # noqa: SLF001
                len(cell.samples),
            ),
            "scalar_fallback_count": fallback_count,
        }
        or manifest.get("output", {}).get("path") != TASK_DATA_NAME
        or manifest.get("output", {}).get("bytes") != data_path.stat().st_size
        or manifest.get("output", {}).get("sha256") != _sha256(data_path)
        or {path.name for path in task_root.iterdir()}
        != {TASK_MANIFEST_NAME, TASK_DATA_NAME}
    ):
        msg = f"Confirmation task failed manifest validation: {task_root}"
        raise ValueError(msg)
    _validate_task_resource_usage(manifest, config, task_root)
    with np.load(data_path, allow_pickle=False) as arrays:
        if (
            set(arrays.files)
            != {
                "likelihood_ratio",
                "reportable",
                "scalar_fallback",
                "sentinel_pair",
            }
            or arrays["likelihood_ratio"].shape != (replicates,)
            or arrays["reportable"].shape != (replicates,)
            or arrays["reportable"].dtype != np.dtype(bool)
            or arrays["scalar_fallback"].shape != (replicates,)
            or arrays["scalar_fallback"].dtype != np.dtype(bool)
            or arrays["sentinel_pair"].shape != (2,)
            or not np.array_equal(arrays["sentinel_pair"], expected_pair)
            or not np.isfinite(arrays["likelihood_ratio"]).all()
            or (arrays["likelihood_ratio"] < 0).any()
            or manifest.get("reportable_count")
            != int(arrays["reportable"].sum())
            or fallback_count != int(arrays["scalar_fallback"].sum())
        ):
            msg = f"Confirmation task arrays failed validation: {task_root}"
            raise ValueError(msg)
    return manifest


def run_task(  # noqa: PLR0913
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    endpoint_id: str,
    nice_increment: int,
) -> str:
    """Run one frozen selected endpoint with 100,000 fresh replicates."""
    config, run_manifest = _ensure_run_root(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
    )
    if nice_increment != int(config["resources"]["nice_increment"]):
        msg = "Confirmation --nice differs from the frozen resource contract."
        raise ValueError(msg)
    endpoint = _find_endpoint(run_manifest, endpoint_id)
    run_manifest_sha256 = _sha256(output_root / RUN_MANIFEST_NAME)
    task_root = _task_root(output_root, endpoint)
    if task_root.exists():
        _validate_task(
            task_root,
            endpoint=endpoint,
            run_manifest_sha256=run_manifest_sha256,
            run_root=run_root,
            provider_root=provider_root,
            config=config,
        )
        return "already-complete"
    _preflight_runtime_resources(config)
    if nice_increment:
        os.nice(nice_increment)
    os.environ.update(stage1_calibration.THREAD_ENV)
    started = time.monotonic()
    cell = stage1_calibration._load_cell(  # noqa: SLF001
        run_root,
        provider_root,
        endpoint.cohort,
        endpoint.provider,
    )
    sentinel_pairs = stage1_calibration._sentinel_pairs(  # noqa: SLF001
        cell.features,
        int(config["source_calibration"]["sentinel_pair_count"]),
    )
    sentinel_pair = sentinel_pairs[
        endpoint.sentinel_pair_index : endpoint.sentinel_pair_index + 1
    ]
    endpoint_seed = _endpoint_seed(int(config["seed"]), endpoint)
    replicates = int(
        config["two_stage_gate"]["stage2"]["replicates_per_selected_endpoint"],
    )
    workers = int(config["resources"]["fit_workers_per_endpoint"])
    with threadpool_limits(
        limits=int(config["resources"]["blas_threads_per_fit_worker"]),
    ):
        likelihood_ratio, reportable, scalar_fallback = _simulate_endpoint_arrays(
            cell=cell,
            sentinel_pair=sentinel_pair,
            endpoint_seed=endpoint_seed,
            replicates=replicates,
            workers=workers,
        )

    task_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".endpoint.", dir=task_root.parent))
    data_path = staging / TASK_DATA_NAME
    with data_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            likelihood_ratio=likelihood_ratio,
            reportable=reportable,
            scalar_fallback=scalar_fallback,
            sentinel_pair=sentinel_pair[0],
        )
        handle.flush()
        os.fsync(handle.fileno())
    self_usage = stage1_calibration._rusage_record(  # noqa: SLF001
        resource.getrusage(resource.RUSAGE_SELF),
        children=False,
    )
    child_usage = stage1_calibration._rusage_record(  # noqa: SLF001
        resource.getrusage(resource.RUSAGE_CHILDREN),
        children=True,
    )
    resource_usage = core._task_resource_usage(started)  # noqa: SLF001
    resource_usage.update(
        {
            "user_cpu_seconds": self_usage["user_cpu_seconds"],
            "system_cpu_seconds": self_usage["system_cpu_seconds"],
            "self": self_usage,
            "terminated_children": child_usage,
        },
    )
    source_manifest, single_gene_input = (
        stage1_calibration._validate_single_gene_source(  # noqa: SLF001
            run_root,
            endpoint.cohort,
            endpoint.provider,
            json.loads(
                (run_root / "contracts" / f"{endpoint.cohort}.json").read_text(
                    encoding="utf-8",
                ),
            ),
        )
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": TASK_CONTRACT,
        "endpoint": _endpoint_payload(endpoint),
        "config_sha256": _sha256(CONFIG_PATH),
        "run_manifest_sha256": run_manifest_sha256,
        "seed": endpoint_seed,
        "replicates": replicates,
        "replicate_rng": config["simulation"]["replicate_rng"],
        "replicate_shards": config["simulation"]["replicate_shards"],
        "replicate_shard_rule": config["simulation"]["replicate_shard_rule"],
        "shard_seeds": [
            _shard_seed(endpoint_seed, index)
            for index in range(int(config["simulation"]["replicate_shards"]))
        ],
        "sentinel_pair": sentinel_pair[0].tolist(),
        "sentinel_features": [cell.features[int(index)] for index in sentinel_pair[0]],
        "fit_kernel": {
            "contract": config["simulation"]["fit_kernel"],
            "replicate_chunk_rule": config["simulation"]["replicate_chunk_rule"],
            "replicate_chunk_size": stage1_calibration._replicate_chunk_size(  # noqa: SLF001
                len(cell.samples),
            ),
            "scalar_fallback_count": int(scalar_fallback.sum()),
        },
        "worker_topology": _worker_topology(config),
        "source_task_manifest": source_manifest,
        "single_gene_input": single_gene_input,
        "reportable_count": int(reportable.sum()),
        "resource_usage": resource_usage,
        "output": _file_record(data_path, relative_to=staging),
    }
    _write_atomic(staging / TASK_MANIFEST_NAME, _canonical_json(manifest) + b"\n")
    if task_root.exists():
        msg = f"Confirmation task output appeared during execution: {task_root}"
        raise FileExistsError(msg)
    staging.replace(task_root)
    _validate_task(
        task_root,
        endpoint=endpoint,
        run_manifest_sha256=run_manifest_sha256,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    return "complete"


def _task_command(  # noqa: PLR0913
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    endpoint: Endpoint,
    nice_increment: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "analysis.calibrate_tcga_revision_focused_confirmation",
        "--calibration-root",
        calibration_root.as_posix(),
        "--run-root",
        run_root.as_posix(),
        "--provider-root",
        provider_root.as_posix(),
        "--output-root",
        output_root.as_posix(),
        "--internal-endpoint-id",
        endpoint.endpoint_id,
        "--nice",
        str(nice_increment),
    ]


def run_all(  # noqa: PLR0913
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    jobs: int,
    nice_increment: int,
) -> None:
    """Run every selected endpoint serially, with five fit workers each."""
    config, manifest = _ensure_run_root(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
    )
    if jobs != int(config["resources"]["max_outer_endpoints"]):
        msg = "Confirmation --jobs differs from the frozen serial scheduler."
        raise ValueError(msg)
    if nice_increment != int(config["resources"]["nice_increment"]):
        msg = "Confirmation --nice differs from the frozen resource contract."
        raise ValueError(msg)
    environment = os.environ.copy()
    environment.update(stage1_calibration.THREAD_ENV)
    for endpoint in _manifest_endpoints(manifest):
        subprocess.run(
            _task_command(
                calibration_root=calibration_root,
                run_root=run_root,
                provider_root=provider_root,
                output_root=output_root,
                endpoint=endpoint,
                nice_increment=nice_increment,
            ),
            check=True,
            env=environment,
        )
        print(f"confirmation complete {endpoint.endpoint_id}", flush=True)


def _stage2_evidence(
    *,
    endpoint: Endpoint,
    selected_count: int,
    task_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stage2 = config["two_stage_gate"]["stage2"]
    if selected_count < 1:
        msg = "Stage-2 evidence requires at least one selected endpoint."
        raise ValueError(msg)
    endpoint_error = float(stage2["familywise_error"]) / selected_count
    with np.load(task_root / TASK_DATA_NAME, allow_pickle=False) as arrays:
        reportable = arrays["reportable"]
        log_p_values = stage1_calibration._effective_log_p_values(  # noqa: SLF001
            arrays["likelihood_ratio"],
            reportable,
        )
        events = int((log_p_values <= np.log(endpoint.threshold)).sum())
        trials = int(log_p_values.size)
        reportable_trials = int(reportable.sum())
    upper = stage1_calibration._clopper_pearson_upper_bound(  # noqa: SLF001
        successes=events,
        trials=trials,
        endpoint_error=endpoint_error,
    )
    acceptance = float(
        config["two_stage_gate"]["acceptance_upper_bounds"][
            f"{endpoint.threshold:.2f}"
        ],
    )
    return {
        "stage2_events": events,
        "stage2_trials": trials,
        "stage2_rate": events / trials,
        "stage2_reportable_trials": reportable_trials,
        "stage2_nonreportable_trials": trials - reportable_trials,
        "stage2_familywise_error": float(stage2["familywise_error"]),
        "stage2_endpoint_count": selected_count,
        "stage2_endpoint_error": endpoint_error,
        "stage2_clopper_pearson_upper_bound": upper,
        "stage2_pass": (
            ENDPOINT_ACCEPTED if upper <= acceptance else ENDPOINT_REJECTED
        ),
    }


def _final_frame(  # noqa: PLR0913
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    stage1_frame, _ = _stage1_frame(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        config=config,
    )
    selected = _manifest_endpoints(run_manifest)
    selected_by_id = {endpoint.endpoint_id: endpoint for endpoint in selected}
    run_manifest_sha256 = _sha256(output_root / RUN_MANIFEST_NAME)
    task_records: list[dict[str, Any]] = []
    stage2_by_id: dict[str, dict[str, Any]] = {}
    for endpoint in selected:
        task_root = _task_root(output_root, endpoint)
        manifest = _validate_task(
            task_root,
            endpoint=endpoint,
            run_manifest_sha256=run_manifest_sha256,
            run_root=run_root,
            provider_root=provider_root,
            config=config,
        )
        record = _file_record(
            task_root / TASK_MANIFEST_NAME,
            relative_to=output_root,
        )
        record.update({"endpoint": manifest["endpoint"]})
        task_records.append(record)
        stage2_by_id[endpoint.endpoint_id] = _stage2_evidence(
            endpoint=endpoint,
            selected_count=len(selected),
            task_root=task_root,
            config=config,
        )

    rows = []
    for stage1_row in stage1_frame.to_dict(orient="records"):
        endpoint_id = str(stage1_row["endpoint_id"])
        selected_for_stage2 = bool(stage1_row["selected_for_stage2"])
        if selected_for_stage2 != (endpoint_id in selected_by_id):
            msg = "Frozen selection differs from recomputed stage-1 failures."
            raise ValueError(msg)
        row = dict(stage1_row)
        if selected_for_stage2:
            evidence = stage2_by_id[endpoint_id]
            row.update(evidence)
            row.update(
                {
                    "final_evidence_stage": "stage2",
                    "final_clopper_pearson_upper_bound": evidence[
                        "stage2_clopper_pearson_upper_bound"
                    ],
                    "final_pass": evidence["stage2_pass"],
                },
            )
        else:
            row.update(
                {
                    "stage2_events": "",
                    "stage2_trials": "",
                    "stage2_rate": "",
                    "stage2_reportable_trials": "",
                    "stage2_nonreportable_trials": "",
                    "stage2_familywise_error": "",
                    "stage2_endpoint_count": "",
                    "stage2_endpoint_error": "",
                    "stage2_clopper_pearson_upper_bound": "",
                    "stage2_pass": STAGE2_NOT_APPLICABLE,
                    "final_evidence_stage": "stage1",
                    "final_clopper_pearson_upper_bound": row[
                        "stage1_clopper_pearson_upper_bound"
                    ],
                    "final_pass": row["stage1_pass"],
                },
            )
        rows.append(row)
    frame = pd.DataFrame(rows, columns=FINAL_TABLE_COLUMNS)
    if (
        len(frame) != int(config["source_calibration"]["endpoint_count"])
        or frame["endpoint_id"].duplicated().any()
        or int(frame["selected_for_stage2"].sum()) != len(selected)
        or not frame.loc[
            ~frame["selected_for_stage2"],
            "stage1_pass",
        ].eq(ENDPOINT_ACCEPTED).all()
        or not frame["final_pass"].isin(
            [ENDPOINT_ACCEPTED, ENDPOINT_REJECTED],
        ).all()
    ):
        msg = "Composite confirmation endpoint table is invalid."
        raise ValueError(msg)
    return frame, task_records


def _summary_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _summary_payload(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    frame: pd.DataFrame,
    task_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = frame.loc[frame["selected_for_stage2"]]
    stage2_passed = int(selected["stage2_pass"].eq(ENDPOINT_ACCEPTED).sum())
    final_passed = int(frame["final_pass"].eq(ENDPOINT_ACCEPTED).sum())
    stage1 = config["two_stage_gate"]["stage1"]
    stage2 = config["two_stage_gate"]["stage2"]
    selected_count = len(selected)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": SUMMARY_CONTRACT,
        "config_sha256": _sha256(CONFIG_PATH),
        "source_calibration_overall_gate_pass": run_manifest[
            "source_calibration_overall_gate_pass"
        ],
        "endpoint_count": len(frame),
        "stage1_familywise_error": stage1["familywise_error"],
        "stage1_endpoint_count": stage1["endpoint_count"],
        "stage1_endpoint_error": float(stage1["familywise_error"])
        / int(stage1["endpoint_count"]),
        "stage1_selected_endpoint_count": selected_count,
        "stage2_familywise_error": stage2["familywise_error"],
        "stage2_endpoint_count": selected_count,
        "stage2_endpoint_error": (
            float(stage2["familywise_error"]) / selected_count
            if selected_count
            else None
        ),
        "stage2_replicates_per_selected_endpoint": stage2[
            "replicates_per_selected_endpoint"
        ],
        "stage2_passed_endpoint_count": stage2_passed,
        "final_passed_endpoint_count": final_passed,
        "overall_gate_pass": final_passed == len(frame),
        "composite_overall_gate_pass": final_passed == len(frame),
        "overall_rule": config["two_stage_gate"]["overall_rule"],
        "total_familywise_error": config["two_stage_gate"][
            "total_familywise_error"
        ],
        "stage1_and_stage2_counts_pooled": False,
        "additional_confirmation_stage_permitted": False,
        "reporting_rule_selected": False,
        "interpretation": "finite-scenario-stress-not-formal-uniform-FDR-proof",
        "resource_contract": dict(config["resources"]),
        "runtime_resource_observation": _resource_observation(config),
        "thread_environment": dict(stage1_calibration.THREAD_ENV),
        "result_blindness": _result_blindness_receipt(),
        "source_records": run_manifest["source_records"],
        "run_manifest": _file_record(
            output_root / RUN_MANIFEST_NAME,
            relative_to=output_root,
        ),
        "task_manifests": list(task_records),
        "final_table": _file_record(
            output_root / FINAL_TABLE_NAME,
            relative_to=output_root,
        ),
    }


def summarize(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> Path:
    """Write the final endpoint table and composite confirmation receipt once."""
    table_path = output_root / FINAL_TABLE_NAME
    summary_path = output_root / SUMMARY_NAME
    if table_path.exists() or summary_path.exists():
        msg = "Refusing to overwrite confirmation summary artifacts."
        raise FileExistsError(msg)
    config = _load_config()
    run_manifest = _validate_run_manifest(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
    )
    frame, task_records = _final_frame(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
        run_manifest=run_manifest,
    )
    _write_atomic(table_path, _summary_csv_bytes(frame))
    payload = _summary_payload(
        output_root=output_root,
        config=config,
        run_manifest=run_manifest,
        frame=frame,
        task_records=task_records,
    )
    _write_atomic(summary_path, _canonical_json(payload) + b"\n")
    validate_summary(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
    )
    return summary_path


def validate_summary(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute the complete two-stage tree and validate its final decision."""
    config = _load_config()
    run_manifest = _validate_run_manifest(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
    )
    frame, task_records = _final_frame(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        config=config,
        run_manifest=run_manifest,
    )
    table_path = output_root / FINAL_TABLE_NAME
    if table_path.read_bytes() != _summary_csv_bytes(frame):
        msg = "Confirmation final table differs from recomputed evidence."
        raise ValueError(msg)
    expected = _summary_payload(
        output_root=output_root,
        config=config,
        run_manifest=run_manifest,
        frame=frame,
        task_records=task_records,
    )
    summary = json.loads((output_root / SUMMARY_NAME).read_text(encoding="utf-8"))
    if (
        summary != expected
        or not isinstance(summary.get("overall_gate_pass"), bool)
        or summary.get("overall_gate_pass")
        is not summary.get("composite_overall_gate_pass")
    ):
        msg = "Confirmation summary or composite gate decision is invalid."
        raise ValueError(msg)
    return summary


def load_validated_composite_gate(
    *,
    calibration_root: Path,
    run_root: Path,
    provider_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Return the fully recomputed confirmation summary for downstream gates."""
    return validate_summary(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--validate-summary", action="store_true")
    parser.add_argument("--internal-endpoint-id")
    return parser


def main() -> None:
    """Run, resume, summarize, or validate the two-stage confirmation."""
    args = _parser().parse_args()
    calibration_root = args.calibration_root.resolve()
    run_root = args.run_root.resolve()
    provider_root = args.provider_root.resolve()
    output_root = args.output_root.absolute()
    if args.summarize and args.validate_summary:
        msg = "Choose either --summarize or --validate-summary."
        raise ValueError(msg)
    if args.internal_endpoint_id and (args.summarize or args.validate_summary):
        msg = "An internal endpoint cannot be combined with a summary action."
        raise ValueError(msg)
    if args.summarize:
        print(
            summarize(
                calibration_root=calibration_root,
                run_root=run_root,
                provider_root=provider_root,
                output_root=output_root,
            ),
        )
        return
    if args.validate_summary:
        validate_summary(
            calibration_root=calibration_root,
            run_root=run_root,
            provider_root=provider_root,
            output_root=output_root,
        )
        print(output_root / SUMMARY_NAME)
        return
    if args.internal_endpoint_id:
        print(
            run_task(
                calibration_root=calibration_root,
                run_root=run_root,
                provider_root=provider_root,
                output_root=output_root,
                endpoint_id=args.internal_endpoint_id,
                nice_increment=args.nice,
            ),
        )
        return
    run_all(
        calibration_root=calibration_root,
        run_root=run_root,
        provider_root=provider_root,
        output_root=output_root,
        jobs=args.jobs,
        nice_increment=args.nice,
    )


if __name__ == "__main__":
    main()

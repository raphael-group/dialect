"""Focused pair-resolved affirmative calibration v3 contract tests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import calibrate_tcga_revision_focused as calibration

if TYPE_CHECKING:
    from pathlib import Path


def _resource_usage(*, spawned_children: bool) -> dict[str, object]:
    self_peak = calibration._normalized_peak_rss(  # noqa: SLF001
        1,
        source="resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        semantics="task-process-maximum-resident-set-size",
    )
    child_peak = calibration._normalized_peak_rss(  # noqa: SLF001
        int(spawned_children),
        source="resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss",
        semantics="maximum-over-terminated-children-not-additive",
    )
    return {
        "elapsed_seconds": 1.0,
        "peak_rss": self_peak,
        "user_cpu_seconds": 0.5,
        "system_cpu_seconds": 0.1,
        "self": {
            "user_cpu_seconds": 0.5,
            "system_cpu_seconds": 0.1,
            "peak_rss": self_peak,
        },
        "terminated_children": {
            "user_cpu_seconds": 0.4 if spawned_children else 0.0,
            "system_cpu_seconds": 0.05 if spawned_children else 0.0,
            "peak_rss": child_peak,
        },
    }


def test_protocol_has_exact_primary_and_descriptive_cells() -> None:
    config = calibration._load_config()  # noqa: SLF001
    cells = calibration._protocol_cells(config)  # noqa: SLF001

    assert len(cells) == 42
    assert len({(cell.cohort, cell.provider) for cell in cells}) == 42
    primary_coordinates = [
        (cell.cohort, cell.provider)
        for cell in cells
        if cell.role == calibration.PRIMARY_ROLE
    ]
    assert primary_coordinates == [
        (cohort, "mutsig") for cohort in calibration.TCGA_COHORTS
    ]
    assert {
        (cell.cohort, cell.provider)
        for cell in cells
        if cell.role == calibration.DESCRIPTIVE_ROLE
    } == {
        (cohort, provider)
        for cohort in ("CHOL", "LAML", "PAAD", "SKCM", "UCEC")
        for provider in ("cbase", "dig")
    }
    assert config["marginal_lrt"] == {
        "pair_selection": "32-disjoint-pairs-spanning-the-K500-feature-rank-axis",
        "sentinel_pair_count": 32,
        "replicates_per_cell": 10_000,
        "replicate_rng": "sha256-cell-seed-and-sentinel-pair-index-v1",
        "fit_kernel": (
            "bounded-batched-profile-lrt-with-scalar-boundary-reconciliation-v1"
        ),
        "replicate_chunk_rule": "max(1,min(512,128000//sample-count))",
        "alphas": [0.01, 0.05],
    }
    assert config["affirmative_gate"] == {
        "provider": "mutsig",
        "endpoint_unit": "cohort-sentinel-pair-alpha",
        "method": (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        ),
        "familywise_error": 0.05,
        "endpoint_count": 2_048,
        "acceptance_upper_bounds": {"0.01": 0.02, "0.05": 0.07},
    }
    assert config["resources"] == {
        "observed_logical_cpus": 14,
        "max_outer_cells": 3,
        "max_mutsig_cells": 1,
        "mutsig_fit_workers": 5,
        "descriptive_fit_workers": 1,
        "max_total_fit_workers": 7,
        "blas_threads_per_fit_worker": 1,
        "nice_increment": 10,
        "overwrite_outputs": False,
    }
    assert calibration._frozen_resource_observation(config) == {  # noqa: SLF001
        "logical_cpus_observed": 14,
        "logical_cpu_source": "os.cpu_count()",
        "half_logical_cpu_limit": 7,
        "scheduled_fit_worker_limit": 7,
        "half_machine_limit_satisfied": True,
    }


def test_effective_p_values_set_nonreportable_fits_to_one() -> None:
    lrt = np.asarray([[100.0, 100.0], [0.0, 4.0]])
    reportable = np.asarray([[True, False], [False, True]])

    observed = calibration._effective_p_values(lrt, reportable)  # noqa: SLF001

    assert observed[0, 0] < 1e-20
    assert observed[0, 1] == 1.0
    assert observed[1, 0] == 1.0
    assert 0 < observed[1, 1] < 0.1


def test_result_blindness_receipt_discloses_integrity_hashing() -> None:
    assert calibration._result_blindness_receipt() == {  # noqa: SLF001
        "observed_pair_statistics_parsed_or_inspected": False,
        "pairwise_files_integrity_hashed": True,
        "pairwise_hash_use": "run-integrity-validation-only",
    }


def test_resource_receipts_validate_portably_but_new_execution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = calibration._load_config()  # noqa: SLF001
    monkeypatch.setattr(calibration.os, "cpu_count", lambda: 14)
    produced_topology = calibration._worker_topology(  # noqa: SLF001
        config,
        provider="mutsig",
    )
    calibration._preflight_runtime_resources(config)  # noqa: SLF001

    monkeypatch.setattr(calibration.os, "cpu_count", lambda: 8)
    assert (
        calibration._worker_topology(config, provider="mutsig")  # noqa: SLF001
        == produced_topology
    )
    calibration._validate_calibration_resource_usage(  # noqa: SLF001
        {
            "worker_topology": produced_topology,
            "resource_usage": _resource_usage(spawned_children=True),
        },
        config,
        provider="mutsig",
    )
    with pytest.raises(RuntimeError, match="execution host differs"):
        calibration._preflight_runtime_resources(config)  # noqa: SLF001


@pytest.mark.parametrize(
    ("status", "expected_label"),
    [
        (calibration.core.REQUIRED_PAIR_EFFECT_IDENTIFIED_STATUS, "reportable"),
        (calibration.core.REQUIRED_PAIR_EFFECT_RANK_DEFICIENT_STATUS, "blocked"),
        (calibration.core.REQUIRED_PAIR_EFFECT_UNDERFLOW_STATUS, "blocked"),
    ],
)
def test_pair_fit_records_actual_identifiability_policy(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_label: str,
) -> None:
    class FakeGene:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def estimate_pi_with_mle(self) -> None:
            pass

    class FakeInteraction:
        likelihood_ratio = 25.0

        def __init__(self, *_genes: object) -> None:
            pass

        def estimate_tau_with_coordinate_ascent(self) -> None:
            pass

        def effect_identifiability_status(self) -> str:
            return status

    monkeypatch.setattr(calibration, "Gene", FakeGene)
    monkeypatch.setattr(calibration, "Interaction", FakeInteraction)
    cell = calibration.Cell(
        cohort="TEST",
        provider="mutsig",
        features=("A_M", "B_M"),
        samples=("S1", "S2"),
        pmfs={"A_M": {0: 1.0}, "B_M": {0: 1.0}},
        pi=np.asarray([0.1, 0.2]),
        source_task_manifest={},
        single_gene_input={},
    )

    statistic, reportable = calibration._fit_pair(  # noqa: SLF001
        cell,
        (0, 1),
        np.zeros((2, 2), dtype=np.int64),
    )

    assert statistic == 25.0
    assert reportable is (expected_label == "reportable")

def test_pair_parallel_simulation_is_worker_count_invariant() -> None:
    samples = tuple(f"S{index}" for index in range(30))
    features = ("A_M", "B_M", "C_M", "D_M")
    pmfs = {
        feature: [{0: 0.98, 1: 0.02} for _ in samples] for feature in features
    }
    cell = calibration.Cell(
        cohort="SYNTHETIC",
        provider="mutsig",
        features=features,
        samples=samples,
        pmfs=pmfs,
        pi=np.asarray([0.05, 0.08, 0.03, 0.1]),
        source_task_manifest={},
        single_gene_input={},
    )
    pairs = np.asarray([[0, 1], [2, 3]], dtype=np.int32)

    serial = calibration._simulate_null_arrays(  # noqa: SLF001
        cell=cell,
        sentinel_pairs=pairs,
        replicates=4,
        cell_seed=12345,
        workers=1,
    )
    parallel = calibration._simulate_null_arrays(  # noqa: SLF001
        cell=cell,
        sentinel_pairs=pairs,
        replicates=4,
        cell_seed=12345,
        workers=2,
    )

    assert np.array_equal(serial[0], parallel[0])
    assert np.array_equal(serial[1], parallel[1])
    assert np.array_equal(serial[2], parallel[2])


@pytest.mark.parametrize("background", ["shared", "sample-specific"])
def test_batched_worker_matches_seeded_scalar_simulation(background: str) -> None:
    samples = tuple(f"S{index}" for index in range(50))
    features = ("A_M", "B_M", "C_M", "D_M")
    shared_pmf = {0: 0.97, 1: 0.03}
    pmfs = {
        feature: (
            [dict(shared_pmf) for _sample in samples]
            if background == "sample-specific"
            else dict(shared_pmf)
        )
        for feature in features
    }
    cell = calibration.Cell(
        cohort="SYNTHETIC",
        provider="mutsig" if background == "sample-specific" else "cbase",
        features=features,
        samples=samples,
        pmfs=pmfs,
        pi=np.asarray([0.05, 0.08, 0.03, 0.1]),
        source_task_manifest={},
        single_gene_input={},
    )
    pairs = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
    narrowed, local_pairs = calibration._narrow_simulation_cell(  # noqa: SLF001
        cell,
        pairs,
    )
    samplers = calibration._prepare_samplers(narrowed)  # noqa: SLF001
    expected_lrt = np.empty((6, 2), dtype=float)
    expected_reportable = np.empty((6, 2), dtype=bool)
    for pair_index, indices in enumerate(local_pairs):
        rng = np.random.default_rng(calibration._pair_seed(4481, pair_index))  # noqa: SLF001
        pair = (int(indices[0]), int(indices[1]))
        for replicate in range(6):
            counts = calibration._simulate_features(  # noqa: SLF001
                narrowed,
                pair,
                samplers,
                rng,
            )
            (
                expected_lrt[replicate, pair_index],
                expected_reportable[replicate, pair_index],
            ) = calibration._fit_pair_scalar_reference(  # noqa: SLF001
                narrowed,
                pair,
                counts,
            )

    observed_lrt, observed_reportable, scalar_fallback = (
        calibration._simulate_null_arrays(  # noqa: SLF001
            cell=cell,
            sentinel_pairs=pairs,
            replicates=6,
            cell_seed=4481,
            workers=1,
        )
    )

    np.testing.assert_allclose(observed_lrt, expected_lrt, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(observed_reportable, expected_reportable)
    observed_p = calibration._effective_p_values(  # noqa: SLF001
        observed_lrt,
        observed_reportable,
    )
    expected_p = calibration._effective_p_values(  # noqa: SLF001
        expected_lrt,
        expected_reportable,
    )
    for alpha in (0.01, 0.05):
        np.testing.assert_array_equal(observed_p <= alpha, expected_p <= alpha)
    assert scalar_fallback.shape == observed_lrt.shape
    assert scalar_fallback.dtype == np.dtype(bool)


def test_exact_binomial_gate_is_pair_resolved_affirmative_and_simultaneous() -> None:
    config = calibration._load_config()  # noqa: SLF001
    endpoint_error = 0.05 / 2_048
    zero_event_upper = calibration._clopper_pearson_upper_bound(  # noqa: SLF001
        successes=0,
        trials=10_000,
        endpoint_error=endpoint_error,
    )

    assert zero_event_upper == pytest.approx(1 - endpoint_error ** (1 / 10_000))
    assert (
        calibration._clopper_pearson_upper_bound(  # noqa: SLF001
            successes=10_000,
            trials=10_000,
            endpoint_error=endpoint_error,
        )
        == 1.0
    )
    accepted = calibration._gate_fields(  # noqa: SLF001
        successes=145,
        trials=10_000,
        alpha=0.01,
        config=config,
    )
    rejected = calibration._gate_fields(  # noqa: SLF001
        successes=146,
        trials=10_000,
        alpha=0.01,
        config=config,
    )
    assert accepted["bonferroni_endpoint_error"] == endpoint_error
    assert accepted["clopper_pearson_upper_bound"] == pytest.approx(
        0.019976185363449205,
    )
    assert accepted["endpoint_gate_pass"] == calibration.ENDPOINT_ACCEPTED
    assert rejected["clopper_pearson_upper_bound"] == pytest.approx(
        0.020092442991776503,
    )
    assert rejected["endpoint_gate_pass"] == calibration.ENDPOINT_REJECTED
    accepted_at_five_percent = calibration._gate_fields(  # noqa: SLF001
        successes=598,
        trials=10_000,
        alpha=0.05,
        config=config,
    )
    rejected_at_five_percent = calibration._gate_fields(  # noqa: SLF001
        successes=599,
        trials=10_000,
        alpha=0.05,
        config=config,
    )
    assert accepted_at_five_percent["clopper_pearson_upper_bound"] == pytest.approx(
        0.0699842679396794,
    )
    assert (
        accepted_at_five_percent["endpoint_gate_pass"]
        == calibration.ENDPOINT_ACCEPTED
    )
    assert rejected_at_five_percent["clopper_pearson_upper_bound"] == pytest.approx(
        0.07009167516093866,
    )
    assert (
        rejected_at_five_percent["endpoint_gate_pass"]
        == calibration.ENDPOINT_REJECTED
    )


def test_single_gene_source_hash_is_checked_against_task_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    task_root = run_root / "tasks" / "CHOL" / "mutsig"
    task_root.mkdir(parents=True)
    single = task_root / "single_gene_results.csv"
    single.write_bytes(b"Gene Name,Pi\nA_M,0.1\n")
    digest = hashlib.sha256(single.read_bytes()).hexdigest()
    manifest = {
        "outputs": {
            "single_gene_results.csv": {
                "path": "single_gene_results.csv",
                "bytes": single.stat().st_size,
                "sha256": digest,
            },
        },
    }
    (task_root / "task_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibration.focused_runner,
        "_validate_completed_task",
        lambda *_args, **_kwargs: manifest,
    )
    contract = {"pair_policy": {"row_count": 1}}

    calibration._validate_single_gene_source(  # noqa: SLF001
        run_root,
        "CHOL",
        "mutsig",
        contract,
    )
    single.write_bytes(b"Gene Name,Pi\nA_M,0.2\n")

    with pytest.raises(ValueError, match="Single-gene source changed"):
        calibration._validate_single_gene_source(  # noqa: SLF001
            run_root,
            "CHOL",
            "mutsig",
            contract,
        )


def test_task_arrays_require_boolean_reportability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = calibration._load_config()  # noqa: SLF001
    run_root = tmp_path / "run"
    contract_root = run_root / "contracts"
    contract_root.mkdir(parents=True)
    features = [f"G{index}_M" for index in range(500)]
    (contract_root / "ACC.json").write_text(
        json.dumps({"features": features, "samples": {"count": 100}}),
        encoding="utf-8",
    )
    task_root = tmp_path / "calibration" / "tasks" / "ACC" / "mutsig"
    task_root.mkdir(parents=True)
    data_path = task_root / calibration.TASK_DATA_NAME
    sentinel = calibration._sentinel_pairs(features)  # noqa: SLF001
    np.savez_compressed(
        data_path,
        marginal_lrt=np.zeros((10_000, 32)),
        marginal_reportable=np.ones((10_000, 32), dtype=bool),
        scalar_fallback=np.zeros((10_000, 32), dtype=bool),
        sentinel_pairs=sentinel,
    )
    source_manifest = {"path": "tasks/ACC/mutsig/task_manifest.json"}
    single_input = {"path": "tasks/ACC/mutsig/single_gene_results.csv"}
    monkeypatch.setattr(
        calibration,
        "_validate_single_gene_source",
        lambda *_args: (source_manifest, single_input),
    )
    monkeypatch.setattr(
        calibration.core,
        "_validate_task_resource_usage",
        lambda *_args: None,
    )
    manifest = {
        "schema_version": calibration.SCHEMA_VERSION,
        "contract": calibration.TASK_CONTRACT,
        "cohort": "ACC",
        "provider": "mutsig",
        "role": calibration.PRIMARY_ROLE,
        "config_sha256": calibration._sha256(calibration.CONFIG_PATH),  # noqa: SLF001
        "run_completion_sha256": "a" * 64,
        "seed": calibration._seed(int(config["seed"]), "ACC", "mutsig"),  # noqa: SLF001
        "marginal_replicates": 10_000,
        "sentinel_pair_count": 32,
        "alphas": [0.01, 0.05],
        "replicate_rng": "sha256-cell-seed-and-sentinel-pair-index-v1",
        "fit_kernel": {
            "contract": (
                "bounded-batched-profile-lrt-with-scalar-boundary-"
                "reconciliation-v1"
            ),
            "replicate_chunk_rule": "max(1,min(512,128000//sample-count))",
            "replicate_chunk_size": 512,
            "scalar_fallback_count": 0,
        },
        "worker_topology": calibration._worker_topology(  # noqa: SLF001
            config,
            provider="mutsig",
        ),
        "marginal_reportable_count": 320_000,
        "source_task_manifest": source_manifest,
        "single_gene_input": single_input,
        "resource_usage": _resource_usage(spawned_children=True),
        "output": calibration._file_record(data_path, relative_to=task_root),  # noqa: SLF001
    }
    (task_root / calibration.TASK_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    calibration._validate_task(  # noqa: SLF001
        task_root,
        config,
        cohort="ACC",
        provider="mutsig",
        role=calibration.PRIMARY_ROLE,
        run_completion_sha256="a" * 64,
        run_root=run_root,
    )

    np.savez_compressed(
        data_path,
        marginal_lrt=np.zeros((10_000, 32)),
        marginal_reportable=np.ones((10_000, 32), dtype=np.int8),
        scalar_fallback=np.zeros((10_000, 32), dtype=bool),
        sentinel_pairs=sentinel,
    )
    manifest["output"] = calibration._file_record(  # noqa: SLF001
        data_path,
        relative_to=task_root,
    )
    (task_root / calibration.TASK_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="arrays failed validation"):
        calibration._validate_task(  # noqa: SLF001
            task_root,
            config,
            cohort="ACC",
            provider="mutsig",
            role=calibration.PRIMARY_ROLE,
            run_completion_sha256="a" * 64,
            run_root=run_root,
        )


def test_summary_frame_keeps_localized_pair_endpoints_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = calibration._load_config()  # noqa: SLF001
    output_root = tmp_path / "calibration"
    task_root = output_root / "tasks" / "TEST" / "mutsig"
    task_root.mkdir(parents=True)
    (task_root / calibration.TASK_MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
    np.savez_compressed(
        task_root / calibration.TASK_DATA_NAME,
        marginal_lrt=np.asarray([[100.0, 0.0]] * 3),
        marginal_reportable=np.ones((3, 2), dtype=bool),
        scalar_fallback=np.zeros((3, 2), dtype=bool),
        sentinel_pairs=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
    )
    cell = calibration.ProtocolCell("TEST", "mutsig", calibration.PRIMARY_ROLE)
    monkeypatch.setattr(calibration, "_protocol_cells", lambda _config: (cell,))
    monkeypatch.setattr(
        calibration,
        "_validate_run_manifest",
        lambda **_kwargs: ("a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        calibration,
        "_validate_task",
        lambda *_args, **_kwargs: {
            "fit_kernel": {},
            "worker_topology": {},
            "resource_usage": {},
        },
    )

    frame, task_manifests, nonreportable = calibration._summary_frame(  # noqa: SLF001
        output_root=output_root,
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        config=config,
    )

    assert frame["sentinel_pair_index"].tolist() == [0, 0, 1, 1]
    assert frame["threshold"].tolist() == [0.01, 0.05, 0.01, 0.05]
    assert frame["events"].tolist() == [3, 3, 0, 0]
    assert frame["trials"].tolist() == [3, 3, 3, 3]
    assert len(task_manifests) == 1
    assert nonreportable == 0


def test_overall_gate_ignores_descriptive_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = calibration._load_config()  # noqa: SLF001
    output_root = tmp_path / "calibration"
    output_root.mkdir()
    (output_root / calibration.RUN_MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
    (output_root / calibration.SUMMARY_TABLE_NAME).write_text(
        "table\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibration,
        "_validate_run_manifest",
        lambda **_kwargs: ("a" * 64, "b" * 64),
    )
    rows = []
    for cell in calibration._protocol_cells(config):  # noqa: SLF001
        for pair_index in range(32):
            for alpha in (0.01, 0.05):
                successes = 0 if cell.role == calibration.PRIMARY_ROLE else 10_000
                row = {
                    "cohort": cell.cohort,
                    "provider": cell.provider,
                    "role": cell.role,
                    "screen": calibration.MARGINAL_SCREEN,
                    "sentinel_pair_index": pair_index,
                    "threshold": alpha,
                    "events": successes,
                    "trials": 10_000,
                    "rate": successes / 10_000,
                    "reportable_trials": 10_000,
                    "nonreportable_trials": 0,
                    "gate_endpoint": cell.role == calibration.PRIMARY_ROLE,
                    "exact_binomial_familywise_error": "",
                    "exact_binomial_endpoint_count": "",
                    "bonferroni_endpoint_error": "",
                    "clopper_pearson_upper_bound": "",
                    "acceptance_upper_bound": "",
                    "endpoint_gate_pass": calibration.GATE_NOT_APPLICABLE,
                }
                if cell.role == calibration.PRIMARY_ROLE:
                    row.update(
                        calibration._gate_fields(  # noqa: SLF001
                            successes=successes,
                            trials=10_000,
                            alpha=alpha,
                            config=config,
                        ),
                    )
                rows.append(row)
    gate_rows = pd.DataFrame(rows, columns=calibration.SUMMARY_COLUMNS)
    descriptive = gate_rows.loc[gate_rows["role"].eq(calibration.DESCRIPTIVE_ROLE)]
    assert len(descriptive) == 640
    assert not descriptive["gate_endpoint"].any()
    assert descriptive["endpoint_gate_pass"].eq(
        calibration.GATE_NOT_APPLICABLE,
    ).all()
    for field in (
        "exact_binomial_familywise_error",
        "exact_binomial_endpoint_count",
        "bonferroni_endpoint_error",
        "clopper_pearson_upper_bound",
        "acceptance_upper_bound",
    ):
        assert descriptive[field].eq("").all()

    summary = calibration._summary_payload(  # noqa: SLF001
        output_root=output_root,
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        config=config,
        frame=gate_rows,
        task_manifests=[],
        nonreportable_fit_count=0,
    )

    assert summary["overall_gate_pass"] is True
    assert summary["primary_gate_endpoint_count"] == 2_048
    assert summary["marginal_endpoint_count"] == 2_688
    first_gate_index = gate_rows.index[gate_rows["gate_endpoint"]][0]
    gate_rows.loc[first_gate_index, "events"] = 146
    gate_rows.loc[first_gate_index, "rate"] = 0.0146
    failed_fields = calibration._gate_fields(  # noqa: SLF001
        successes=146,
        trials=10_000,
        alpha=float(gate_rows.loc[first_gate_index, "threshold"]),
        config=config,
    )
    for name, value in failed_fields.items():
        gate_rows.loc[first_gate_index, name] = value
    summary = calibration._summary_payload(  # noqa: SLF001
        output_root=output_root,
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        config=config,
        frame=gate_rows,
        task_manifests=[],
        nonreportable_fit_count=0,
    )
    assert summary["overall_gate_pass"] is False


def test_scheduler_runs_mutsig_serially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = calibration._load_config()  # noqa: SLF001
    captured: dict[str, object] = {}
    monkeypatch.setattr(calibration, "_ensure_run_root", lambda *_args: config)

    def capture(
        tasks: tuple[calibration.ProtocolCell, ...],
        **kwargs: object,
    ) -> None:
        captured.update(tasks=tasks, **kwargs)

    monkeypatch.setattr(calibration, "_run_protocol", capture)

    calibration.run_all(
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_root=tmp_path / "calibration",
        jobs=3,
        nice_increment=10,
    )

    tasks = captured["tasks"]
    assert isinstance(tasks, tuple)
    assert len(tasks) == 42
    assert sum(cell.provider == "mutsig" for cell in tasks) == 32
    assert captured["jobs"] == 3
    assert captured["max_mutsig_cells"] == 1
    assert captured["mutsig_fit_workers"] == 5
    assert captured["descriptive_fit_workers"] == 1
    assert captured["max_total_fit_workers"] == 7


def test_mixed_scheduler_caps_one_mutsig_and_two_descriptive_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = (
        calibration.ProtocolCell("M1", "mutsig", calibration.PRIMARY_ROLE),
        calibration.ProtocolCell("M2", "mutsig", calibration.PRIMARY_ROLE),
        calibration.ProtocolCell("D1", "cbase", calibration.DESCRIPTIVE_ROLE),
        calibration.ProtocolCell("D2", "dig", calibration.DESCRIPTIVE_ROLE),
        calibration.ProtocolCell("D3", "cbase", calibration.DESCRIPTIVE_ROLE),
        calibration.ProtocolCell("D4", "dig", calibration.DESCRIPTIVE_ROLE),
    )
    lock = threading.Lock()
    active = {"mutsig": 0, "descriptive": 0}
    maxima = {"mutsig": 0, "descriptive": 0, "total": 0, "fit_workers": 0}

    def fake_run(command: list[str], **_kwargs: object) -> None:
        provider = command[command.index("--internal-provider") + 1]
        role = "mutsig" if provider == "mutsig" else "descriptive"
        with lock:
            active[role] += 1
            maxima[role] = max(maxima[role], active[role])
            maxima["total"] = max(maxima["total"], sum(active.values()))
            active_fit_workers = active["mutsig"] * 5 + active["descriptive"]
            maxima["fit_workers"] = max(maxima["fit_workers"], active_fit_workers)
        time.sleep(0.02)
        with lock:
            active[role] -= 1

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    calibration._run_protocol(  # noqa: SLF001
        tasks,
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_root=tmp_path / "calibration",
        jobs=3,
        max_mutsig_cells=1,
        mutsig_fit_workers=5,
        descriptive_fit_workers=1,
        max_total_fit_workers=7,
        nice_increment=10,
    )

    assert maxima == {
        "mutsig": 1,
        "descriptive": 2,
        "total": 3,
        "fit_workers": 7,
    }

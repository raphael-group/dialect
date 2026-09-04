"""Focused affirmative calibration v2 contract tests."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import calibrate_tcga_revision_focused as calibration

if TYPE_CHECKING:
    from pathlib import Path


def test_protocol_has_exact_primary_and_descriptive_cells() -> None:
    config = calibration._load_config()  # noqa: SLF001
    cells = calibration._protocol_cells(config)  # noqa: SLF001

    assert len(cells) == 42
    assert len({(cell.cohort, cell.provider) for cell in cells}) == 42
    primary_cohorts = [
        cell.cohort for cell in cells if cell.role == calibration.PRIMARY_ROLE
    ]
    assert primary_cohorts == list(calibration.TCGA_COHORTS)
    assert {
        (cell.cohort, cell.provider)
        for cell in cells
        if cell.role == calibration.DESCRIPTIVE_ROLE
    } == {
        (cohort, provider)
        for cohort in ("CHOL", "LAML", "PAAD", "SKCM", "UCEC")
        for provider in ("cbase", "dig")
    }
    assert config["affirmative_gate"]["endpoint_count"] == 64


def test_effective_p_values_set_nonreportable_fits_to_one() -> None:
    lrt = np.asarray([[100.0, 100.0], [0.0, 4.0]])
    reportable = np.asarray([[True, False], [False, True]])

    observed = calibration._effective_p_values(lrt, reportable)  # noqa: SLF001

    assert observed[0, 0] < 1e-20
    assert observed[0, 1] == 1.0
    assert observed[1, 0] == 1.0
    assert 0 < observed[1, 1] < 0.1


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


def test_hoeffding_gate_is_affirmative_and_simultaneous() -> None:
    config = calibration._load_config()  # noqa: SLF001
    margin = calibration._hoeffding_margin(  # noqa: SLF001
        trials=64_000,
        endpoint_count=64,
        familywise_error=0.05,
    )

    assert margin == pytest.approx(0.00747632479737792)
    assert calibration._gate_fields(  # noqa: SLF001
        successes=768,
        trials=64_000,
        alpha=0.01,
        config=config,
    )["endpoint_gate_pass"] == calibration.ENDPOINT_ACCEPTED
    assert calibration._gate_fields(  # noqa: SLF001
        successes=832,
        trials=64_000,
        alpha=0.01,
        config=config,
    )["endpoint_gate_pass"] == calibration.ENDPOINT_REJECTED


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
        json.dumps({"features": features}),
        encoding="utf-8",
    )
    task_root = tmp_path / "calibration" / "tasks" / "ACC" / "mutsig"
    task_root.mkdir(parents=True)
    data_path = task_root / calibration.TASK_DATA_NAME
    sentinel = calibration._sentinel_pairs(features)  # noqa: SLF001
    np.savez_compressed(
        data_path,
        marginal_lrt=np.zeros((1000, 64)),
        marginal_reportable=np.ones((1000, 64), dtype=bool),
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
        "marginal_replicates": 1000,
        "sentinel_pair_count": 64,
        "alphas": [0.01, 0.05],
        "marginal_reportable_count": 64_000,
        "source_task_manifest": source_manifest,
        "single_gene_input": single_input,
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
        marginal_lrt=np.zeros((1000, 64)),
        marginal_reportable=np.ones((1000, 64), dtype=np.int8),
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
    gate_rows = pd.DataFrame(
        {
            "gate_endpoint": [True] * 64 + [False],
            "endpoint_gate_pass": [calibration.ENDPOINT_ACCEPTED] * 64
            + [calibration.ENDPOINT_REJECTED],
            "trials": [64_000] * 65,
        },
    )

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
    gate_rows.loc[0, "endpoint_gate_pass"] = calibration.ENDPOINT_REJECTED
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
    batches: list[tuple[tuple[calibration.ProtocolCell, ...], int]] = []
    monkeypatch.setattr(calibration, "_ensure_run_root", lambda *_args: config)

    def capture(
        tasks: tuple[calibration.ProtocolCell, ...],
        **kwargs: object,
    ) -> None:
        batches.append((tasks, int(kwargs["jobs"])))

    monkeypatch.setattr(calibration, "_run_batch", capture)

    calibration.run_all(
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_root=tmp_path / "calibration",
        jobs=3,
        nice_increment=10,
    )

    assert len(batches) == 2
    assert len(batches[0][0]) == 32
    assert {cell.provider for cell in batches[0][0]} == {"mutsig"}
    assert batches[0][1] == 1
    assert len(batches[1][0]) == 10
    assert "mutsig" not in {cell.provider for cell in batches[1][0]}
    assert batches[1][1] == 3

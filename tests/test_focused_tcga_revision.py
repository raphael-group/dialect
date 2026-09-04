"""Contract tests for the focused K=500 revision workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import freeze_tcga_revision_reporting_rule as reporting_rule
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import prepare_tcga_revision_focused as preparation
from analysis import run_tcga_revision_focused as runner

if TYPE_CHECKING:
    from pathlib import Path


def _minimal_provider_root(root: Path) -> tuple[Path, dict[str, object]]:
    cohort_root = root / "cohorts" / "CHOL"
    mutsig_root = root / "mutsig" / "CHOL"
    cohort_root.mkdir(parents=True)
    mutsig_root.mkdir(parents=True)
    for name in preparation.REQUIRED_PROVIDER_FILES:
        (cohort_root / name).write_text(f"{name}\n", encoding="utf-8")
    for name in preparation.REQUIRED_MUTSIG_FILES:
        (mutsig_root / name).write_text(f"{name}\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": preparation.SCHEMA_VERSION,
        "contract": preparation.PROVIDER_CONTRACT,
        "config_sha256": preparation._sha256(preparation.CONFIG_PATH),  # noqa: SLF001
        "cohorts": ["CHOL"],
        "cohort_count": 1,
        "records": [preparation._provider_record(root, "CHOL")],  # noqa: SLF001
    }
    path = root / "provider_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path, manifest


def test_focused_config_freezes_scientific_hierarchy() -> None:
    config = preparation._load_config()  # noqa: SLF001

    analysis = config["analysis"]
    reporting = config["reporting"]
    assert analysis["cohort_count"] == 32
    assert analysis["participant_count"] == 10433
    assert analysis["top_k"] == 500
    assert analysis["providers"] == ["cbase", "dig", "mutsig"]
    assert analysis["primary_provider"] == "mutsig"
    assert analysis["continuity_provider"] == "cbase"
    assert analysis["supplementary_providers"] == ["dig"]
    assert analysis["provider_overlap"] == "descriptive-only"
    assert analysis["epsilon_prefilter"] == "none"
    assert analysis["probability_floor_or_provider_fallback"] == "none"
    assert analysis["test"] == "nondirectional-two-sided-one-df-profile-lrt"
    assert analysis["direction"] == "marshall-olkin-rho-sign-after-testing"
    assert reporting["multiplicity"] == "freeze-after-calibration"
    assert reporting["primary_q_threshold"] == "freeze-after-calibration"


def test_focused_cohort_selection_retains_canonical_order() -> None:
    assert preparation._parse_cohorts("PAAD,CHOL,LAML") == (  # noqa: SLF001
        "CHOL",
        "LAML",
        "PAAD",
    )
    with pytest.raises(ValueError, match="unique"):
        preparation._parse_cohorts("CHOL,CHOL")  # noqa: SLF001


def test_run_root_is_bound_to_config_and_provider_manifest(tmp_path) -> None:
    provider_root = tmp_path / "providers"
    provider_path, _ = _minimal_provider_root(provider_root)
    output_root = tmp_path / "run"

    runner._ensure_run_root(provider_root, output_root, ("CHOL",))  # noqa: SLF001

    manifest = json.loads(
        (output_root / "run_manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest["contract"] == runner.RUN_CONTRACT
    assert manifest["config_sha256"] == hashlib.sha256(
        preparation.CONFIG_PATH.read_bytes(),
    ).hexdigest()
    assert manifest["provider_manifest"]["sha256"] == hashlib.sha256(
        provider_path.read_bytes(),
    ).hexdigest()


def test_run_root_rejects_provider_manifest_drift(tmp_path) -> None:
    provider_root = tmp_path / "providers"
    provider_path, base = _minimal_provider_root(provider_root)
    output_root = tmp_path / "run"
    runner._ensure_run_root(provider_root, output_root, ("CHOL",))  # noqa: SLF001
    provider_path.write_text(
        json.dumps({**base, "unexpected": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different focused contract"):
        runner._ensure_run_root(  # noqa: SLF001
            provider_root,
            output_root,
            ("CHOL",),
        )


def test_input_root_validation_detects_canonical_maf_drift(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    maf = root / "mafs" / "CHOL.maf"
    axis = root / "population" / "CHOL" / "sample_axis.txt"
    maf.parent.mkdir(parents=True)
    axis.parent.mkdir(parents=True)
    maf.write_bytes(b"maf\n")
    axis.write_text("sample-1\n", encoding="utf-8")
    manifest = {
        "schema_version": preparation.SCHEMA_VERSION,
        "contract": preparation.INPUT_CONTRACT,
        "config_sha256": preparation._sha256(preparation.CONFIG_PATH),  # noqa: SLF001
        "cohorts": ["CHOL"],
        "cohort_count": 1,
        "participant_count": 1,
        "cohort_records": [
            {
                "cohort": "CHOL",
                "sample_count": 1,
                "sample_axis_sha256": preparation._sequence_sha256(["sample-1"]),  # noqa: SLF001
                "canonical_maf": {
                    "path": "mafs/CHOL.maf",
                    "bytes": maf.stat().st_size,
                    "sha256": preparation._sha256(maf),  # noqa: SLF001
                },
            },
        ],
    }
    (root / "input_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    preparation.validate_input_root(root, ("CHOL",))
    maf.write_bytes(b"changed\n")

    with pytest.raises(ValueError, match="canonical input changed"):
        preparation.validate_input_root(root, ("CHOL",))


def test_provider_root_validation_detects_artifact_drift(tmp_path: Path) -> None:
    root = tmp_path / "providers"
    cohort_root = root / "cohorts" / "CHOL"
    mutsig_root = root / "mutsig" / "CHOL"
    cohort_root.mkdir(parents=True)
    mutsig_root.mkdir(parents=True)
    for name in preparation.REQUIRED_PROVIDER_FILES:
        (cohort_root / name).write_text(f"{name}\n", encoding="utf-8")
    for name in preparation.REQUIRED_MUTSIG_FILES:
        (mutsig_root / name).write_text(f"{name}\n", encoding="utf-8")
    manifest = {
        "schema_version": preparation.SCHEMA_VERSION,
        "contract": preparation.PROVIDER_CONTRACT,
        "config_sha256": preparation._sha256(preparation.CONFIG_PATH),  # noqa: SLF001
        "cohorts": ["CHOL"],
        "cohort_count": 1,
        "records": [preparation._provider_record(root, "CHOL")],  # noqa: SLF001
    }
    (root / "provider_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    preparation.validate_provider_root(root, ("CHOL",))
    (cohort_root / preparation.REQUIRED_PROVIDER_FILES[0]).write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider artifacts changed"):
        preparation.validate_provider_root(root, ("CHOL",))


def test_focused_runner_rejects_excess_parallelism(tmp_path) -> None:
    with pytest.raises(ValueError, match="resource contract"):
        runner.orchestrate(
            provider_root=tmp_path / "providers",
            output_root=tmp_path / "run",
            cohorts=("CHOL",),
            jobs=4,
            nice_increment=10,
            preflight_only=True,
        )


def test_completed_task_manifest_binds_coordinate_and_resource_units(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    outputs = {}
    for name in ("pairwise_interaction_results.csv", "single_gene_results.csv"):
        path = task_root / name
        path.write_text(f"{name}\n", encoding="utf-8")
        outputs[name] = runner._file_record(path, relative_to=task_root)  # noqa: SLF001
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "contract": runner.TASK_CONTRACT,
        "cohort": "CHOL",
        "provider": "cbase",
        "top_k": 500,
        "contract_sha256": "contract",
        "config_sha256": runner._sha256(preparation.CONFIG_PATH),  # noqa: SLF001
        "single_gene_rows": 500,
        "pairwise_rows": 1,
        "resource_usage": {
            "elapsed_seconds": 1.0,
            "peak_rss": {
                "bytes": 1024,
                "native_value": 1,
                "native_unit": "KiB",
                "platform": "linux",
                "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            },
        },
        "outputs": outputs,
    }
    (task_root / "task_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    runner._validate_completed_task(  # noqa: SLF001
        task_root,
        contract_sha256="contract",
        cohort="CHOL",
        provider="cbase",
        pairwise_rows=1,
    )
    with pytest.raises(ValueError, match="manifest is invalid"):
        runner._validate_completed_task(  # noqa: SLF001
            task_root,
            contract_sha256="contract",
            cohort="CHOL",
            provider="dig",
            pairwise_rows=1,
        )


def test_focused_bh_uses_the_complete_family() -> None:
    observed = postprocess.benjamini_hochberg(
        np.array([0.01, 0.04, 0.03, 0.002], dtype=np.float64),
    )

    assert observed == pytest.approx([0.02, 0.04, 0.04, 0.008])


def test_direction_is_assigned_only_from_fitted_rho() -> None:
    observed = postprocess._direction(  # noqa: SLF001
        pd.Series([-0.2, 0.0, 0.4, np.nan]),
    )

    assert observed.tolist() == ["ME", "neutral", "CO", "unavailable"]


def test_focused_bh_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite p-value family"):
        postprocess.benjamini_hochberg(np.array([0.1, np.nan]))


def test_serial_canary_stops_before_the_next_provider(tmp_path, monkeypatch) -> None:
    observed = []

    def fail_first(command, **_kwargs):
        observed.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", fail_first)

    with pytest.raises(subprocess.CalledProcessError):
        runner._run_batch(  # noqa: SLF001
            [("CHOL", "cbase"), ("CHOL", "dig")],
            provider_root=tmp_path / "providers",
            output_root=tmp_path / "run",
            jobs=1,
            nice_increment=10,
        )

    assert len(observed) == 1
    assert "cbase" in observed[0]


def test_calibration_candidates_are_frozen_before_result_inspection() -> None:
    config = calibration._load_config()  # noqa: SLF001

    assert config["cells"]["cohorts"] == ["CHOL", "LAML", "PAAD", "SKCM", "UCEC"]
    assert config["reporting_candidates"] == {
        "primary_q_threshold": 0.1,
        "sensitivity_q_threshold": 0.2,
        "thresholds_selected_from_observed_pairs": False,
        "retain_chi_square_bh_only_without_detected_inflation": True,
        "interpretation": "finite-scenario-stress-not-formal-uniform-FDR-proof",
    }


def test_calibration_sentinel_pairs_are_disjoint_and_span_axis() -> None:
    features = tuple(f"G{index}_M" for index in range(500))
    pairs = calibration._sentinel_pairs(features)  # noqa: SLF001

    assert pairs.shape == (64, 2)
    assert len(set(pairs.ravel().tolist())) == 128
    assert int(pairs.min()) == 0
    assert int(pairs.max()) >= 490


def _write_calibration_task(
    task_root: Path,
    *,
    cohort: str = "CHOL",
    provider: str = "cbase",
    run_completion_sha256: str = "a" * 64,
) -> dict[str, object]:
    config = calibration._load_config()  # noqa: SLF001
    task_root.mkdir(parents=True)
    data_path = task_root / calibration.TASK_DATA_NAME
    np.savez_compressed(
        data_path,
        marginal_lrt=np.zeros((1000, 64), dtype=np.float64),
        family_rejections=np.zeros((250, 2), dtype=np.int32),
        family_min_p=np.ones(250, dtype=np.float64),
        sentinel_pairs=np.arange(128, dtype=np.int32).reshape(64, 2),
    )
    manifest: dict[str, object] = {
        "schema_version": calibration.SCHEMA_VERSION,
        "contract": calibration.TASK_CONTRACT,
        "cohort": cohort,
        "provider": provider,
        "config_sha256": calibration._sha256(calibration.CONFIG_PATH),  # noqa: SLF001
        "run_completion_sha256": run_completion_sha256,
        "seed": calibration._seed(int(config["seed"]), cohort, provider),  # noqa: SLF001
        "marginal_replicates": 1000,
        "sentinel_pair_count": 64,
        "family_replicates": 250,
        "family_top_k": 30,
        "q_values": [0.1, 0.2],
        "resource_usage": {
            "elapsed_seconds": 1.0,
            "peak_rss": {
                "bytes": 1024,
                "native_value": 1,
                "native_unit": "KiB",
                "platform": "linux",
                "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            },
            "user_cpu_seconds": 0.5,
            "system_cpu_seconds": 0.1,
        },
        "output": {
            "path": calibration.TASK_DATA_NAME,
            "bytes": data_path.stat().st_size,
            "sha256": calibration._sha256(data_path),  # noqa: SLF001
        },
    }
    (task_root / calibration.TASK_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest


def test_calibration_task_validation_binds_coordinates(tmp_path: Path) -> None:
    config = calibration._load_config()  # noqa: SLF001
    task_root = tmp_path / "task"
    manifest = _write_calibration_task(task_root)

    calibration._validate_task(  # noqa: SLF001
        task_root,
        config,
        cohort="CHOL",
        provider="cbase",
        run_completion_sha256="a" * 64,
    )
    manifest["provider"] = "dig"
    (task_root / calibration.TASK_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest validation"):
        calibration._validate_task(  # noqa: SLF001
            task_root,
            config,
            cohort="CHOL",
            provider="cbase",
            run_completion_sha256="a" * 64,
        )


def test_calibration_task_validation_rejects_other_run(tmp_path: Path) -> None:
    config = calibration._load_config()  # noqa: SLF001
    task_root = tmp_path / "task"
    _write_calibration_task(task_root)

    with pytest.raises(ValueError, match="manifest validation"):
        calibration._validate_task(  # noqa: SLF001
            task_root,
            config,
            cohort="CHOL",
            provider="cbase",
            run_completion_sha256="b" * 64,
        )


def test_postprocess_root_validation_detects_table_drift(tmp_path: Path) -> None:
    output_root = tmp_path / "postprocess"
    cohort_root = output_root / "CHOL"
    cohort_root.mkdir(parents=True)
    result_path = cohort_root / postprocess.RESULT_NAME
    result_path.write_text("gene_a,gene_b\nA_M,B_M\n", encoding="utf-8")
    cohort_manifest = {
        "schema_version": postprocess.SCHEMA_VERSION,
        "contract": postprocess.DERIVATION_CONTRACT,
        "cohort": "CHOL",
        "providers": list(postprocess.BMRS),
        "output": postprocess._file_record(  # noqa: SLF001
            result_path,
            relative_to=output_root,
        ),
    }
    cohort_manifest_path = cohort_root / postprocess.COHORT_MANIFEST_NAME
    cohort_manifest_path.write_bytes(
        postprocess._canonical_json(cohort_manifest) + b"\n",  # noqa: SLF001
    )
    root_manifest = {
        "schema_version": postprocess.SCHEMA_VERSION,
        "contract": postprocess.ROOT_CONTRACT,
        "cohorts": ["CHOL"],
        "cohort_count": 1,
        "provider_family_count": 3,
        "cohort_manifests": [
            postprocess._file_record(  # noqa: SLF001
                cohort_manifest_path,
                relative_to=output_root,
            ),
        ],
    }
    (output_root / postprocess.ROOT_MANIFEST_NAME).write_bytes(
        postprocess._canonical_json(root_manifest) + b"\n",  # noqa: SLF001
    )

    postprocess.validate_derived_root(output_root, ("CHOL",))
    result_path.write_text("gene_a,gene_b\nA_M,C_M\n", encoding="utf-8")

    with pytest.raises(ValueError, match="derived cohort output"):
        postprocess.validate_derived_root(output_root, ("CHOL",))


def test_reporting_rule_freezes_prespecified_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_root = tmp_path / "calibration"
    postprocess_root = tmp_path / "postprocess"
    calibration_root.mkdir()
    postprocess_root.mkdir()
    (calibration_root / calibration.SUMMARY_NAME).write_text("{}\n", encoding="utf-8")
    (postprocess_root / postprocess.ROOT_MANIFEST_NAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibration,
        "validate_summary",
        lambda _root: {
            "detected_inflation": False,
            "retain_chi_square_bh_candidates": True,
        },
    )
    monkeypatch.setattr(postprocess, "validate_derived_root", lambda *_args: {})
    output = tmp_path / "reporting_rule.json"

    reporting_rule.freeze_rule(
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        output_path=output,
    )

    rule = json.loads(output.read_text(encoding="utf-8"))
    assert rule["primary_provider"] == "mutsig"
    assert rule["primary_q_threshold"] == 0.1
    assert rule["sensitivity_q_threshold"] == 0.2
    assert rule["scope"] == "one-identical-rule-across-all-32-cancer-types"
    assert (
        rule["direction_unavailable"]
        == "retain-nondirectional-rejection-exclude-from-me-co-lists"
    )
    assert rule["thresholds_selected_from_observed_pairs"] is False

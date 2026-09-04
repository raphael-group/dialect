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
    cells = calibration._protocol_cells(config)  # noqa: SLF001

    assert config["cells"] == {
        "primary_gate": {
            "role": calibration.PRIMARY_ROLE,
            "cohorts": list(calibration.TCGA_COHORTS),
            "providers": ["mutsig"],
        },
        "descriptive": {
            "role": calibration.DESCRIPTIVE_ROLE,
            "cohorts": ["CHOL", "LAML", "PAAD", "SKCM", "UCEC"],
            "providers": ["cbase", "dig"],
        },
    }
    assert len(cells) == 42
    assert len({(cell.cohort, cell.provider) for cell in cells}) == 42
    assert [
        (cell.cohort, cell.provider)
        for cell in cells
        if cell.role == calibration.PRIMARY_ROLE
    ] == [(cohort, "mutsig") for cohort in calibration.TCGA_COHORTS]
    assert {
        (cell.cohort, cell.provider)
        for cell in cells
        if cell.role == calibration.DESCRIPTIVE_ROLE
    } == {
        (cohort, provider)
        for cohort in ("CHOL", "LAML", "PAAD", "SKCM", "UCEC")
        for provider in ("cbase", "dig")
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
    assert config["reporting_candidates"] == {
        "test": "chi-square-one-df-profile-lrt",
        "primary_adjustment": "benjamini-yekutieli",
        "primary_q_threshold": 0.01,
        "sensitivity_adjustment": "benjamini-hochberg",
        "sensitivity_q_threshold": 0.01,
        "thresholds_selected_from_observed_pairs": False,
        "interpretation": "finite-scenario-stress-not-formal-uniform-FDR-proof",
    }


def test_calibration_sentinel_pairs_are_disjoint_and_span_axis() -> None:
    features = tuple(f"G{index}_M" for index in range(500))
    pairs = calibration._sentinel_pairs(features)  # noqa: SLF001

    assert pairs.shape == (32, 2)
    assert len(set(pairs.ravel().tolist())) == 64
    assert int(pairs.min()) == 0
    assert int(pairs.max()) >= 490


def _write_calibration_task(
    task_root: Path,
    *,
    run_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = calibration._load_config()  # noqa: SLF001
    cohort = "CHOL"
    provider = "cbase"
    features = [f"G{index}_M" for index in range(500)]
    contract_root = run_root / "contracts"
    contract_root.mkdir(parents=True, exist_ok=True)
    (contract_root / f"{cohort}.json").write_text(
        json.dumps({"features": features, "samples": {"count": 100}}),
        encoding="utf-8",
    )
    task_root.mkdir(parents=True)
    data_path = task_root / calibration.TASK_DATA_NAME
    np.savez_compressed(
        data_path,
        marginal_lrt=np.zeros((10_000, 32), dtype=np.float64),
        marginal_reportable=np.ones((10_000, 32), dtype=bool),
        scalar_fallback=np.zeros((10_000, 32), dtype=bool),
        sentinel_pairs=calibration._sentinel_pairs(features),  # noqa: SLF001
    )
    source_task_manifest: dict[str, object] = {
        "path": f"tasks/{cohort}/{provider}/task_manifest.json",
        "bytes": 101,
        "sha256": "b" * 64,
    }
    single_gene_input: dict[str, object] = {
        "path": f"tasks/{cohort}/{provider}/single_gene_results.csv",
        "bytes": 202,
        "sha256": "c" * 64,
    }
    self_peak = calibration._normalized_peak_rss(  # noqa: SLF001
        1,
        source="resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        semantics="task-process-maximum-resident-set-size",
    )
    child_peak = calibration._normalized_peak_rss(  # noqa: SLF001
        0,
        source="resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss",
        semantics="maximum-over-terminated-children-not-additive",
    )
    manifest: dict[str, object] = {
        "schema_version": calibration.SCHEMA_VERSION,
        "contract": calibration.TASK_CONTRACT,
        "cohort": cohort,
        "provider": provider,
        "role": calibration.DESCRIPTIVE_ROLE,
        "config_sha256": calibration._sha256(calibration.CONFIG_PATH),  # noqa: SLF001
        "run_completion_sha256": "a" * 64,
        "seed": calibration._seed(int(config["seed"]), cohort, provider),  # noqa: SLF001
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
            provider=provider,
        ),
        "marginal_reportable_count": 320_000,
        "source_task_manifest": source_task_manifest,
        "single_gene_input": single_gene_input,
        "resource_usage": {
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
                "user_cpu_seconds": 0.0,
                "system_cpu_seconds": 0.0,
                "peak_rss": child_peak,
            },
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
    return manifest, source_task_manifest, single_gene_input


def test_calibration_task_validation_binds_coordinates(tmp_path: Path) -> None:
    config = calibration._load_config()  # noqa: SLF001
    task_root = tmp_path / "task"
    run_root = tmp_path / "run"
    manifest, source_task_manifest, single_gene_input = _write_calibration_task(
        task_root,
        run_root=run_root,
    )

    def validate_source(*_args: object) -> tuple[dict[str, object], dict[str, object]]:
        return source_task_manifest, single_gene_input

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            calibration,
            "_validate_single_gene_source",
            validate_source,
        )
        calibration._validate_task(  # noqa: SLF001
            task_root,
            config,
            cohort="CHOL",
            provider="cbase",
            role=calibration.DESCRIPTIVE_ROLE,
            run_completion_sha256="a" * 64,
            run_root=run_root,
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
                role=calibration.DESCRIPTIVE_ROLE,
                run_completion_sha256="a" * 64,
                run_root=run_root,
            )


def test_calibration_task_validation_rejects_other_run(tmp_path: Path) -> None:
    config = calibration._load_config()  # noqa: SLF001
    task_root = tmp_path / "task"
    run_root = tmp_path / "run"
    _, source_task_manifest, single_gene_input = _write_calibration_task(
        task_root,
        run_root=run_root,
    )

    def validate_source(*_args: object) -> tuple[dict[str, object], dict[str, object]]:
        return source_task_manifest, single_gene_input

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            calibration,
            "_validate_single_gene_source",
            validate_source,
        )
        with pytest.raises(ValueError, match="manifest validation"):
            calibration._validate_task(  # noqa: SLF001
                task_root,
                config,
                cohort="CHOL",
                provider="cbase",
                role=calibration.DESCRIPTIVE_ROLE,
                run_completion_sha256="b" * 64,
                run_root=run_root,
            )


def test_postprocess_root_validation_detects_table_drift(tmp_path: Path) -> None:
    output_root = tmp_path / "postprocess"
    cohort_root = output_root / "CHOL"
    cohort_root.mkdir(parents=True)
    result_path = cohort_root / postprocess.RESULT_NAME
    frame = pd.DataFrame({"gene_a": ["A_M"], "gene_b": ["B_M"]})
    for provider in postprocess.BMRS:
        statistics = postprocess._provider_statistics(  # noqa: SLF001
            np.asarray([1.0]),
            pd.Series([-0.1]),
            pd.Series(["full-affine-rank"]),
        )
        for name, values in statistics.items():
            frame[f"{provider}_{name}"] = values
    frame.to_csv(result_path, index=False)
    diagnostics = {
        provider: {
            "full_affine_rank_count": 1,
            "rank_deficient_count": 0,
            "rank_not_certified_underflow_count": 0,
            "p_display_clipped_count": 0,
            "by_display_clipped_count": 0,
            "bh_display_clipped_count": 0,
        }
        for provider in postprocess.BMRS
    }
    cohort_manifest = {
        "schema_version": postprocess.SCHEMA_VERSION,
        "contract": postprocess.DERIVATION_CONTRACT,
        "cohort": "CHOL",
        "providers": list(postprocess.BMRS),
        "sources": {provider: {} for provider in postprocess.BMRS},
        "pair_count": 1,
        "family": "all-matched-unordered-pairs-excluding-same-base-M:N",
        "multiplicity": {
            "primary": "provider-specific-BY-over-complete-within-cohort-family",
            "nominal_sensitivity": (
                "provider-specific-BH-over-complete-within-cohort-family"
            ),
        },
        "non_full_rank": "retain-in-family-with-p-one-and-no-directional-effect",
        "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
        "reporting_threshold_selected": False,
        "diagnostics": diagnostics,
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
        "pair_count_per_provider": 1,
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "probability_representation": postprocess.PROBABILITY_REPRESENTATION,
        "multiplicity": {
            "primary": "benjamini-yekutieli",
            "nominal_sensitivity": "benjamini-hochberg",
        },
        "reporting_threshold_selected": False,
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
    root_manifest["pair_count_per_provider"] = 2
    (output_root / postprocess.ROOT_MANIFEST_NAME).write_bytes(
        postprocess._canonical_json(root_manifest) + b"\n",  # noqa: SLF001
    )
    with pytest.raises(ValueError, match="aggregate pair count"):
        postprocess.validate_derived_root(output_root, ("CHOL",))
    root_manifest["pair_count_per_provider"] = 1
    (output_root / postprocess.ROOT_MANIFEST_NAME).write_bytes(
        postprocess._canonical_json(root_manifest) + b"\n",  # noqa: SLF001
    )
    frame.loc[0, "gene_b"] = "C_M"
    frame.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="derived cohort output"):
        postprocess.validate_derived_root(output_root, ("CHOL",))


def test_reporting_rule_freezes_prespecified_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_root = tmp_path / "calibration"
    postprocess_root = tmp_path / "postprocess"
    run_root = tmp_path / "run"
    provider_root = tmp_path / "providers"
    calibration_root.mkdir()
    postprocess_root.mkdir()
    run_root.mkdir()
    provider_root.mkdir()
    summary = {
        "overall_gate_pass": True,
        "reporting_rule_selected": False,
        "gate_provider": "mutsig",
        "gate_endpoint_unit": "cohort-sentinel-pair-alpha",
        "gate_method": (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        ),
        "exact_binomial_familywise_error": 0.05,
        "exact_binomial_endpoint_count": 2_048,
        "acceptance_upper_bounds": {"0.01": 0.02, "0.05": 0.07},
        "primary_adjustment": "benjamini-yekutieli",
        "primary_q_candidate": 0.01,
        "sensitivity_adjustment": "benjamini-hochberg",
        "sensitivity_q_candidate": 0.01,
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
    }
    (calibration_root / calibration.SUMMARY_NAME).write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (postprocess_root / postprocess.ROOT_MANIFEST_NAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        calibration,
        "validate_summary",
        lambda *_args, **_kwargs: summary,
    )
    monkeypatch.setattr(
        postprocess,
        "validate_derived_root",
        lambda *_args, **_kwargs: {},
    )
    output = tmp_path / "reporting_rule.json"

    reporting_rule.freeze_rule(
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        run_root=run_root,
        provider_root=provider_root,
        output_path=output,
    )

    rule = json.loads(output.read_text(encoding="utf-8"))
    assert rule["primary_provider"] == "mutsig"
    assert rule["primary_adjustment"] == "benjamini-yekutieli"
    assert rule["primary_q_threshold"] == 0.01
    assert rule["sensitivity_adjustment"] == "benjamini-hochberg"
    assert rule["sensitivity_q_threshold"] == 0.01
    assert rule["inference_status"] == reporting_rule.REPORTABLE_STATUS
    assert rule["calibration_gate"] == {
        "provider": "mutsig",
        "endpoint_unit": "cohort-sentinel-pair-alpha",
        "method": (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        ),
        "endpoint_count": 2_048,
        "familywise_error": 0.05,
        "acceptance_upper_bounds": {"0.01": 0.02, "0.05": 0.07},
        "overall_gate_pass": True,
    }
    assert (
        rule["scope"]
        == "one-identical-rule-across-all-32-tcga-pan-cancer-atlas-cohorts"
    )
    assert (
        rule["direction"]
        == "primary-provider-rho-sign-after-nondirectional-rejection"
    )
    assert (
        rule["direction_unavailable"]
        == "retain-nondirectional-rejection-exclude-from-me-co-lists"
    )
    assert rule["thresholds_selected_from_observed_pairs"] is False


def test_failed_gate_freezes_withheld_rule_without_association_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_root = tmp_path / "calibration"
    postprocess_root = tmp_path / "postprocess"
    calibration_root.mkdir()
    postprocess_root.mkdir()
    summary = {
        "overall_gate_pass": False,
        "reporting_rule_selected": False,
        "gate_provider": "mutsig",
        "gate_endpoint_unit": "cohort-sentinel-pair-alpha",
        "gate_method": (
            "pair-resolved-simultaneous-one-sided-exact-binomial-"
            "clopper-pearson-with-bonferroni"
        ),
        "exact_binomial_familywise_error": 0.05,
        "exact_binomial_endpoint_count": 2_048,
        "acceptance_upper_bounds": {"0.01": 0.02, "0.05": 0.07},
        "primary_adjustment": "benjamini-yekutieli",
        "primary_q_candidate": 0.01,
        "sensitivity_adjustment": "benjamini-hochberg",
        "sensitivity_q_candidate": 0.01,
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
    }
    (calibration_root / calibration.SUMMARY_NAME).write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (postprocess_root / postprocess.ROOT_MANIFEST_NAME).write_text(
        "{}\n",
        encoding="utf-8",
    )

    def fail_if_association_accessed(*_args, **_kwargs):
        msg = "association table was accessed"
        raise AssertionError(msg)

    path_type = type(tmp_path)
    original_open = path_type.open

    def guarded_open(path, *args, **kwargs):
        if path.name in {
            postprocess.RESULT_NAME,
            "pairwise_interaction_results.csv",
        }:
            msg = f"association file was opened: {path}"
            raise AssertionError(msg)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "open", guarded_open)
    monkeypatch.setattr(calibration, "validate_summary", fail_if_association_accessed)
    monkeypatch.setattr(
        postprocess,
        "validate_derived_root",
        fail_if_association_accessed,
    )
    output = tmp_path / "withheld-rule.json"
    reporting_rule.freeze_rule(
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_path=output,
    )

    rule = json.loads(output.read_text(encoding="utf-8"))
    assert rule["calibration_gate"]["overall_gate_pass"] is False
    assert rule["inference_status"] == reporting_rule.WITHHELD_STATUS


@pytest.mark.parametrize(
    ("summary", "write_summary", "error"),
    [
        ({}, True, TypeError),
        ({"overall_gate_pass": "false"}, True, TypeError),
        ({}, False, FileNotFoundError),
    ],
)
def test_freeze_rejects_missing_or_malformed_gate_before_association_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, object],
    write_summary: bool,  # noqa: FBT001
    error: type[Exception],
) -> None:
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    if write_summary:
        (calibration_root / calibration.SUMMARY_NAME).write_text(
            json.dumps(summary),
            encoding="utf-8",
        )

    def fail_if_association_accessed(*_args, **_kwargs):
        msg = "association table was accessed"
        raise AssertionError(msg)

    monkeypatch.setattr(calibration, "validate_summary", fail_if_association_accessed)
    monkeypatch.setattr(
        postprocess,
        "validate_derived_root",
        fail_if_association_accessed,
    )
    with pytest.raises(error):
        reporting_rule.freeze_rule(
            calibration_root=calibration_root,
            postprocess_root=tmp_path / "postprocess",
            run_root=tmp_path / "run",
            provider_root=tmp_path / "providers",
            output_path=tmp_path / "reporting-rule.json",
        )

"""Result-blind two-stage focused calibration confirmation tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import calibrate_tcga_revision_focused as stage1
from analysis import calibrate_tcga_revision_focused_confirmation as confirmation

if TYPE_CHECKING:
    from pathlib import Path


def _source_gate_rows(*, failures: set[tuple[str, int, float]]) -> pd.DataFrame:
    rows = []
    for cohort in confirmation.stage1_calibration.TCGA_COHORTS:
        for pair_index in range(32):
            for threshold in (0.01, 0.05):
                coordinate = (cohort, pair_index, threshold)
                events = 700 if coordinate in failures else 0
                rows.append(
                    {
                        "cohort": cohort,
                        "provider": "mutsig",
                        "role": stage1.PRIMARY_ROLE,
                        "screen": stage1.MARGINAL_SCREEN,
                        "sentinel_pair_index": pair_index,
                        "threshold": threshold,
                        "events": events,
                        "trials": 10_000,
                        "rate": events / 10_000,
                        "reportable_trials": 9_900,
                        "nonreportable_trials": 100,
                        "gate_endpoint": True,
                        "exact_binomial_familywise_error": 0.05,
                        "exact_binomial_endpoint_count": 2_048,
                        "bonferroni_endpoint_error": 0.05 / 2_048,
                        "clopper_pearson_upper_bound": 0.0,
                        "acceptance_upper_bound": (
                            0.02 if threshold == 0.01 else 0.07
                        ),
                        "endpoint_gate_pass": stage1.ENDPOINT_ACCEPTED,
                    },
                )
    return pd.DataFrame(rows, columns=stage1.SUMMARY_COLUMNS)


def _patch_source_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    rows: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        confirmation.stage1_calibration,
        "validate_summary",
        lambda *_args, **_kwargs: {
            "contract": stage1.SUMMARY_CONTRACT,
            "primary_gate_endpoint_count": 2_048,
            "overall_gate_pass": False,
        },
    )
    monkeypatch.setattr(
        confirmation.stage1_calibration,
        "_load_config",
        lambda: {"source": "frozen"},
    )
    monkeypatch.setattr(
        confirmation.stage1_calibration,
        "_summary_frame",
        lambda **_kwargs: (rows, [], 0),
    )
    monkeypatch.setattr(
        confirmation.stage1_calibration,
        "_validated_gate_rows",
        lambda frame, _config: frame,
    )


def test_config_freezes_two_stage_error_split_and_half_machine_limit() -> None:
    config = confirmation._load_config()  # noqa: SLF001

    assert config["seed"] == 20260904
    assert config["two_stage_gate"]["stage1"] == {
        "familywise_error": 0.025,
        "endpoint_count": 2_048,
        "selection_rule": (
            "select-every-endpoint-whose-recomputed-one-sided-clopper-pearson-"
            "upper-bound-exceeds-its-acceptance-bound"
        ),
    }
    assert config["two_stage_gate"]["stage2"] == {
        "familywise_error": 0.025,
        "replicates_per_selected_endpoint": 100_000,
        "endpoint_count_rule": "number-of-stage1-selected-endpoints",
        "decision_rule": (
            "one-sided-clopper-pearson-upper-bound-at-gamma2-divided-by-"
            "selected-endpoint-count"
        ),
    }
    assert config["simulation"]["pooling_with_stage1"] is False
    assert config["simulation"]["additional_stage_after_stage2"] is False
    assert confirmation._resource_observation(config) == {  # noqa: SLF001
        "logical_cpus_observed": 14,
        "logical_cpu_source": "os.cpu_count()",
        "half_logical_cpu_limit": 7,
        "scheduled_fit_worker_limit": 5,
        "half_machine_limit_satisfied": True,
    }


def test_stage1_recomputes_stricter_bounds_and_selects_every_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failures = {("UVM", 0, 0.05), ("CHOL", 12, 0.01)}
    rows = _source_gate_rows(failures=failures)
    _patch_source_recomputation(monkeypatch, rows)
    config = confirmation._load_config()  # noqa: SLF001

    frame, _ = confirmation._stage1_frame(  # noqa: SLF001
        calibration_root=tmp_path / "calibration",
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        config=config,
    )

    expected_error = 0.025 / 2_048
    assert frame["stage1_endpoint_error"].eq(expected_error).all()
    assert set(frame.loc[frame["selected_for_stage2"], "endpoint_id"]) == {
        "UVM__mutsig__sentinel-00__alpha-005",
        "CHOL__mutsig__sentinel-12__alpha-001",
    }
    assert frame.loc[~frame["selected_for_stage2"], "stage1_pass"].eq(
        confirmation.ENDPOINT_ACCEPTED,
    ).all()


def test_endpoint_and_shard_seeds_are_fresh_and_coordinate_separated() -> None:
    first = confirmation.Endpoint("UVM", "mutsig", 0, 0.05)
    second = confirmation.Endpoint("UVM", "mutsig", 0, 0.01)
    first_seed = confirmation._endpoint_seed(20260904, first)  # noqa: SLF001

    assert first.endpoint_id == "UVM__mutsig__sentinel-00__alpha-005"
    assert first_seed != confirmation._endpoint_seed(20260904, second)  # noqa: SLF001
    assert first_seed != stage1._seed(20260903, "UVM", "mutsig")  # noqa: SLF001
    assert len(
        {
            confirmation._shard_seed(first_seed, index)  # noqa: SLF001
            for index in range(5)
        },
    ) == 5


def test_stage2_uses_only_fresh_counts_and_gamma2_over_m(tmp_path: Path) -> None:
    endpoint = confirmation.Endpoint("UVM", "mutsig", 0, 0.05)
    task_root = tmp_path / endpoint.endpoint_id
    task_root.mkdir()
    likelihood_ratio = np.zeros(100_000, dtype=np.float64)
    likelihood_ratio[:5_000] = 100.0
    reportable = np.ones(100_000, dtype=bool)
    reportable[5_000:5_100] = False
    with (task_root / confirmation.TASK_DATA_NAME).open("wb") as handle:
        np.savez_compressed(
            handle,
            likelihood_ratio=likelihood_ratio,
            reportable=reportable,
            scalar_fallback=np.zeros(100_000, dtype=bool),
            sentinel_pair=np.asarray([0, 1], dtype=np.int32),
        )

    evidence = confirmation._stage2_evidence(  # noqa: SLF001
        endpoint=endpoint,
        selected_count=2,
        task_root=task_root,
        config=confirmation._load_config(),  # noqa: SLF001
    )

    assert evidence["stage2_events"] == 5_000
    assert evidence["stage2_trials"] == 100_000
    assert evidence["stage2_endpoint_count"] == 2
    assert evidence["stage2_endpoint_error"] == 0.025 / 2
    assert evidence["stage2_pass"] == confirmation.ENDPOINT_ACCEPTED


def test_run_manifest_freezes_selection_and_is_write_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = confirmation._load_config()  # noqa: SLF001
    endpoint = confirmation.Endpoint("UVM", "mutsig", 0, 0.05)
    payload = {
        "schema_version": confirmation.SCHEMA_VERSION,
        "contract": confirmation.RUN_CONTRACT,
        "stage1": {
            "selected_endpoint_count": 1,
            "selected_endpoints": [confirmation._endpoint_payload(endpoint)],  # noqa: SLF001
        },
    }
    monkeypatch.setattr(confirmation, "_load_config", lambda: config)
    monkeypatch.setattr(confirmation, "_preflight_runtime_resources", lambda _c: {})
    monkeypatch.setattr(confirmation, "_run_manifest_payload", lambda **_k: payload)
    output_root = tmp_path / "confirmation"

    _, observed = confirmation._ensure_run_root(  # noqa: SLF001
        calibration_root=tmp_path / "calibration",
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_root=output_root,
    )
    first = (output_root / confirmation.RUN_MANIFEST_NAME).read_bytes()
    confirmation._ensure_run_root(  # noqa: SLF001
        calibration_root=tmp_path / "calibration",
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_root=output_root,
    )

    assert observed == payload
    assert (output_root / confirmation.RUN_MANIFEST_NAME).read_bytes() == first
    payload["stage1"]["selected_endpoint_count"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="different source evidence"):
        confirmation._ensure_run_root(  # noqa: SLF001
            calibration_root=tmp_path / "calibration",
            run_root=tmp_path / "run",
            provider_root=tmp_path / "providers",
            output_root=output_root,
        )


def test_composite_summary_exposes_one_canonical_boolean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = {
        "overall_gate_pass": True,
        "composite_overall_gate_pass": True,
    }
    monkeypatch.setattr(confirmation, "validate_summary", lambda **_kwargs: expected)

    observed = confirmation.load_validated_composite_gate(
        calibration_root=tmp_path / "calibration",
        run_root=tmp_path / "run",
        provider_root=tmp_path / "providers",
        output_root=tmp_path / "confirmation",
    )

    assert observed is expected


def test_confirmation_protocol_note_forbids_pooling_and_third_stage() -> None:
    note = confirmation.CONFIG_PATH.with_name(
        "tcga_revision_calibration_confirmation_protocol.md",
    ).read_text(encoding="utf-8")
    assert "Stage-1 and stage-2 counts are not pooled" in note
    assert "no third stage" in note
    assert "never parsed or" in note


def test_config_json_is_canonical_contract_source() -> None:
    raw = json.loads(confirmation.CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw == confirmation._load_config()  # noqa: SLF001
    assert confirmation.FINAL_CELLS_NAME == confirmation.FINAL_TABLE_NAME

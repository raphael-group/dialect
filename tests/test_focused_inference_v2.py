"""Focused inference-v2 contract tests."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from analysis import diagnose_tcga_revision_focused as diagnosis
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import report_tcga_revision_focused as reporting

if TYPE_CHECKING:
    from pathlib import Path


def _guard_association_file_access(
    monkeypatch: pytest.MonkeyPatch,
    sample_path: Path,
) -> None:
    path_type = type(sample_path)
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


def test_by_and_bh_are_explicit_complete_family_adjustments() -> None:
    p_values = np.asarray([0.01, 0.04, 0.03, 0.002], dtype=np.float64)

    bh = postprocess.benjamini_hochberg(p_values)
    by = postprocess.benjamini_yekutieli(p_values)

    assert bh == pytest.approx([0.02, 0.04, 0.04, 0.008])
    assert by == pytest.approx(np.asarray([0.02, 0.04, 0.04, 0.008]) * (25 / 12))
    assert np.all(by >= bh)


def test_non_full_rank_pairs_remain_in_family_at_p_one() -> None:
    statistics = postprocess._provider_statistics(  # noqa: SLF001
        np.asarray([1_000.0, 1_000.0, 3.841458820694124]),
        pd.Series([np.nan, np.nan, 0.2]),
        pd.Series(
            [
                "rank-deficient",
                "rank-not-certified-underflow",
                "full-affine-rank",
            ],
        ),
    )

    p_values = np.asarray(statistics["p_value"])
    bh = np.asarray(statistics["bh_q_value"])
    assert p_values[:2].tolist() == [1.0, 1.0]
    assert p_values[2] == pytest.approx(0.05)
    assert bh[2] == pytest.approx(0.15)
    assert np.asarray(statistics["effect_reportable"]).tolist() == [False, False, True]
    assert pd.Series(statistics["direction"]).tolist() == [
        "unavailable",
        "unavailable",
        "CO",
    ]
    assert pd.Series(statistics["rho"]).iloc[:2].isna().all()


def test_log_space_tail_and_adjustments_never_emit_zero() -> None:
    statistics = postprocess._provider_statistics(  # noqa: SLF001
        np.asarray([2_000.0, 1.0]),
        pd.Series([0.5, -0.1]),
        pd.Series(["full-affine-rank", "full-affine-rank"]),
    )

    assert np.isfinite(np.asarray(statistics["log_p_value"])).all()
    assert np.asarray(statistics["log_p_value"])[0] < -1_000
    for column in ("p_value", "by_q_value", "bh_q_value"):
        values = np.asarray(statistics[column])
        assert (values > 0).all()
        assert np.isfinite(values).all()


def test_full_rank_degenerate_rho_is_unavailable_only_for_negligible_lrt() -> None:
    statistics = postprocess._provider_statistics(  # noqa: SLF001
        np.asarray([0.0]),
        pd.Series([np.nan]),
        pd.Series(["full-affine-rank"]),
    )

    assert np.asarray(statistics["p_value"]).tolist() == [1.0]
    assert pd.Series(statistics["direction"]).tolist() == ["unavailable"]
    with pytest.raises(ValueError, match="positive-LRT"):
        postprocess._provider_statistics(  # noqa: SLF001
            np.asarray([postprocess.core.REQUIRED_UNDEFINED_RHO_LRT_TOL * 2]),
            pd.Series([np.nan]),
            pd.Series(["full-affine-rank"]),
        )


@pytest.mark.parametrize("rho", [np.inf, -np.inf, "not-a-number"])
def test_nonfinite_or_malformed_rho_is_never_treated_as_missing(rho: object) -> None:
    with pytest.raises(ValueError, match="finite or genuinely missing"):
        postprocess._provider_statistics(  # noqa: SLF001
            np.asarray([0.0]),
            pd.Series([rho]),
            pd.Series(["full-affine-rank"]),
        )


def test_semantic_validator_recomputes_complete_family_q_values() -> None:
    providers = []
    for provider in postprocess.BMRS:
        values = postprocess._provider_statistics(  # noqa: SLF001
            np.asarray([10.0, 1.0]),
            pd.Series([-0.5, 0.5]),
            pd.Series(["full-affine-rank", "full-affine-rank"]),
        )
        providers.append(
            pd.DataFrame(
                {f"{provider}_{name}": value for name, value in values.items()},
            ),
        )
    frame = pd.concat(
        [
            pd.DataFrame({"gene_a": ["A_M", "C_M"], "gene_b": ["B_M", "D_M"]}),
            *providers,
        ],
        axis=1,
    )

    postprocess.validate_inference_frame(frame, cohort="TEST")
    frame.loc[0, "mutsig_log_by_q_value"] = frame.loc[
        0,
        "mutsig_log_bh_q_value",
    ]
    frame.loc[0, "mutsig_by_q_value"] = frame.loc[0, "mutsig_bh_q_value"]
    with pytest.raises(ValueError, match="complete-family policy"):
        postprocess.validate_inference_frame(frame, cohort="TEST")


def test_raw_source_binding_rejects_derived_statistic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    pair_axis = pd.DataFrame({"gene_a": ["A_M"], "gene_b": ["B_M"]})
    expected_by_provider = {}
    derived = pair_axis.copy()
    sources = {}
    for provider in postprocess.BMRS:
        source = (
            run_root
            / "tasks"
            / "TEST"
            / provider
            / "pairwise_interaction_results.csv"
        )
        source.parent.mkdir(parents=True)
        source.write_text(f"source-{provider}\n", encoding="utf-8")
        sources[provider] = postprocess._file_record(  # noqa: SLF001
            source,
            relative_to=run_root,
        )
        statistics = postprocess._provider_statistics(  # noqa: SLF001
            np.asarray([1.0]),
            pd.Series([-0.2]),
            pd.Series(["full-affine-rank"]),
        )
        expected = pd.DataFrame(
            {
                "gene_a": ["A_M"],
                "gene_b": ["B_M"],
                **{f"{provider}_{name}": value for name, value in statistics.items()},
            },
        )
        expected_by_provider[provider] = expected
        derived = pd.concat([derived, expected.iloc[:, 2:]], axis=1)

    monkeypatch.setattr(
        postprocess,
        "_read_provider",
        lambda _root, _cohort, provider: expected_by_provider[provider],
    )
    manifest = {"sources": sources}
    postprocess._validate_raw_source_binding(  # noqa: SLF001
        manifest=manifest,
        frame=derived,
        run_root=run_root,
        cohort="TEST",
    )

    derived.loc[0, "dig_likelihood_ratio"] = 2.0
    with pytest.raises(ValueError, match="differ from the raw provider output"):
        postprocess._validate_raw_source_binding(  # noqa: SLF001
            manifest=manifest,
            frame=derived,
            run_root=run_root,
            cohort="TEST",
        )


def test_provider_reader_rejects_pair_axis_drift_even_with_updated_receipt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    contract_root = run_root / "contracts"
    task_root = run_root / "tasks" / "TEST" / "cbase"
    contract_root.mkdir(parents=True)
    task_root.mkdir(parents=True)
    contract = {
        "features": ["A_M", "B_M"],
        "pair_policy": {"row_count": 1},
    }
    (contract_root / "TEST.json").write_bytes(
        postprocess._canonical_json(contract) + b"\n",  # noqa: SLF001
    )
    pairwise = pd.DataFrame(
        {
            column: pd.Series([None], dtype="object")
            for column in postprocess.PAIRWISE_COLUMNS
        },
    )
    pairwise.loc[0, "Gene A"] = "A_M"
    pairwise.loc[0, "Gene B"] = "B_M"
    pairwise.loc[0, "Likelihood Ratio"] = 1.0
    pairwise.loc[0, "Rho"] = -0.2
    pairwise.loc[0, "Effect Identifiability"] = "full-affine-rank"
    source = task_root / "pairwise_interaction_results.csv"
    pairwise.to_csv(source, index=False)
    (task_root / "single_gene_results.csv").write_text(
        "placeholder\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": postprocess.SCHEMA_VERSION,
        "contract": postprocess.focused_runner.TASK_CONTRACT,
        "cohort": "TEST",
        "provider": "cbase",
        "top_k": 500,
        "contract_sha256": hashlib.sha256(
            postprocess._canonical_json(contract),  # noqa: SLF001
        ).hexdigest(),
        "config_sha256": postprocess._sha256(  # noqa: SLF001
            postprocess.focused_runner.CONFIG_PATH,
        ),
        "pairwise_rows": 1,
        "outputs": {
            source.name: postprocess._file_record(  # noqa: SLF001
                source,
                relative_to=task_root,
            ),
            "single_gene_results.csv": {},
        },
    }
    manifest_path = task_root / "task_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    observed_pairs = postprocess._read_provider(  # noqa: SLF001
        run_root,
        "TEST",
        "cbase",
    )[["gene_a", "gene_b"]].to_numpy()
    assert observed_pairs.tolist() == [["A_M", "B_M"]]

    pairwise.loc[0, "Gene B"] = "C_M"
    pairwise.to_csv(source, index=False)
    manifest["outputs"][source.name] = postprocess._file_record(  # noqa: SLF001
        source,
        relative_to=task_root,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Pair axis differs"):
        postprocess._read_provider(run_root, "TEST", "cbase")  # noqa: SLF001


@pytest.mark.parametrize(
    ("gate_value", "write_summary", "error", "message"),
    [
        (False, True, RuntimeError, "reporting is withheld"),
        (None, False, FileNotFoundError, "Calibration summary is missing"),
        ("false", True, TypeError, "affirmative overall gate decision"),
    ],
)
def test_report_builder_stops_before_association_reads_for_closed_gate(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate_value: object,
    write_summary: bool,  # noqa: FBT001
    error: type[Exception],
    message: str,
) -> None:
    _guard_association_file_access(monkeypatch, tmp_path)
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    if write_summary:
        (calibration_root / reporting.calibration.SUMMARY_NAME).write_text(
            json.dumps({"overall_gate_pass": gate_value}),
            encoding="utf-8",
        )

    def fail_if_read(*_args, **_kwargs):
        msg = "association values were read"
        raise AssertionError(msg)

    monkeypatch.setattr(reporting, "validate_provider_root", fail_if_read)
    monkeypatch.setattr(reporting.postprocess, "validate_derived_root", fail_if_read)
    monkeypatch.setattr(reporting.calibration, "validate_summary", fail_if_read)
    monkeypatch.setattr(reporting, "_load_rule", fail_if_read)
    monkeypatch.setattr(reporting, "_burden_values", fail_if_read)
    monkeypatch.setattr(reporting, "_read_inference", fail_if_read)

    with pytest.raises(error, match=message):
        reporting.build_report(
            run_root=tmp_path / "run",
            provider_root=tmp_path / "providers",
            postprocess_root=tmp_path / "postprocess",
            calibration_root=calibration_root,
            rule_path=tmp_path / "rule.json",
            output_root=tmp_path / "report",
        )


@pytest.mark.parametrize(
    ("gate_value", "write_summary", "error", "message"),
    [
        (False, True, RuntimeError, "diagnostics is withheld"),
        (None, False, FileNotFoundError, "Calibration summary is missing"),
        ("false", True, TypeError, "affirmative overall gate decision"),
    ],
)
def test_diagnostics_stop_before_association_reads_for_closed_gate(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate_value: object,
    write_summary: bool,  # noqa: FBT001
    error: type[Exception],
    message: str,
) -> None:
    _guard_association_file_access(monkeypatch, tmp_path)
    postprocess_root = tmp_path / "postprocess"
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    if write_summary:
        (calibration_root / reporting.calibration.SUMMARY_NAME).write_text(
            json.dumps({"overall_gate_pass": gate_value}),
            encoding="utf-8",
        )

    def fail_if_read(*_args, **_kwargs):
        msg = "association values were read"
        raise AssertionError(msg)

    monkeypatch.setattr(diagnosis.postprocess, "validate_derived_root", fail_if_read)
    monkeypatch.setattr(diagnosis.calibration, "validate_summary", fail_if_read)
    monkeypatch.setattr(diagnosis.reporting, "_load_rule", fail_if_read)
    monkeypatch.setattr(diagnosis.reporting, "_read_inference", fail_if_read)

    with pytest.raises(error, match=message):
        diagnosis.diagnose(
            run_root=tmp_path / "run",
            provider_root=tmp_path / "providers",
            postprocess_root=postprocess_root,
            calibration_root=calibration_root,
            rule_path=tmp_path / "rule.json",
            output_root=tmp_path / "diagnostics",
            cohorts=("CHOL",),
        )


@pytest.mark.parametrize(
    ("gate_value", "write_summary", "error"),
    [
        (False, True, RuntimeError),
        (None, False, FileNotFoundError),
        ({"not": "a boolean"}, True, TypeError),
    ],
)
def test_report_validator_checks_gate_before_report_or_association_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gate_value: object,
    write_summary: bool,  # noqa: FBT001
    error: type[Exception],
) -> None:
    _guard_association_file_access(monkeypatch, tmp_path)
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    if write_summary:
        (calibration_root / reporting.calibration.SUMMARY_NAME).write_text(
            json.dumps({"overall_gate_pass": gate_value}),
            encoding="utf-8",
        )

    def fail_if_read(*_args, **_kwargs):
        msg = "association values were read"
        raise AssertionError(msg)

    monkeypatch.setattr(reporting.calibration, "validate_summary", fail_if_read)
    monkeypatch.setattr(reporting, "_load_rule", fail_if_read)
    monkeypatch.setattr(reporting, "validate_provider_root", fail_if_read)
    monkeypatch.setattr(reporting.postprocess, "validate_derived_root", fail_if_read)

    with pytest.raises(error):
        reporting.validate_report(
            tmp_path / "report",
            run_root=tmp_path / "run",
            provider_root=tmp_path / "providers",
            postprocess_root=tmp_path / "postprocess",
            calibration_root=calibration_root,
            rule_path=tmp_path / "rule.json",
        )


def test_report_validator_requires_gate_inputs_before_report_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="required before report access"):
        reporting.validate_report(tmp_path / "report")


def test_labels_are_derived_from_named_adjustment() -> None:
    assert (
        reporting._threshold_label("benjamini-yekutieli", 0.01)  # noqa: SLF001
        == "BY q <= 0.01"
    )
    assert (
        reporting._threshold_label("benjamini-hochberg", 0.01)  # noqa: SLF001
        == "BH q <= 0.01"
    )
    assert reporting._decision_prefix("mutsig", "primary") == (  # noqa: SLF001
        "mutsig_primary_rejection"
    )
    assert reporting._decision_prefix("cbase", "primary") == (  # noqa: SLF001
        "cbase_descriptive_primary_rule_crossing"
    )
    assert reporting._decision_prefix("mutsig", "sensitivity") == (  # noqa: SLF001
        "mutsig_descriptive_sensitivity_rule_crossing"
    )

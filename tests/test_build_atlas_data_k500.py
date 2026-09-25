"""Contract tests for the K=500 focused-grid Atlas release assembler."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import build_atlas_data_k500 as k500
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis.build_atlas_baselines import feature_sequence_sha256
from analysis.build_atlas_data import default_release_id

FEATURES = ["TP53_M", "KRAS_M", "TP53_N", "IDH1_M"]
PAIRS = [
    ("TP53_M", "KRAS_M"),
    ("TP53_M", "IDH1_M"),
    ("KRAS_M", "TP53_N"),
    ("KRAS_M", "IDH1_M"),
    ("TP53_N", "IDH1_M"),
]
LRT = {
    "mutsig": [30.0, 0.5, 12.0, 0.0, 3.0],
    "cbase": [25.0, 0.1, 0.0, 9.0, 1.0],
    "dig": [2.0, 0.2, 0.3, 0.4, 0.0],
}
RHO = {
    "mutsig": [-0.6, 0.1, 0.4, np.nan, -0.2],
    "cbase": [-0.5, 0.05, np.nan, 0.3, -0.1],
    "dig": [-0.1, 0.02, 0.03, 0.04, np.nan],
}
IDENTIFIABILITY = {
    provider: [
        "full-affine-rank" if np.isfinite(value) else "rank-deficient"
        for value in values
    ]
    for provider, values in RHO.items()
}
# mutsig row 3 has LRT 0 so a missing rho is admissible even when full rank.
IDENTIFIABILITY["mutsig"][3] = "full-affine-rank"


def _sha(path: Path) -> str:
    return k500.sha256_file(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _counts() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.integers(0, 3, size=(8, len(FEATURES) + 1)),
        columns=[*FEATURES, "OTHER_M"],
        index=[f"S{i}" for i in range(8)],
    )
    frame.index.name = "sample"
    return frame


def _task(provider: str, counts: pd.DataFrame) -> pd.DataFrame:
    binary = counts[FEATURES] > 0
    rows = []
    for (a, b), lrt, rho, effect in zip(
        PAIRS,
        LRT[provider],
        RHO[provider],
        IDENTIFIABILITY[provider],
        strict=True,
    ):
        both = int((binary[a] & binary[b]).sum())
        a_only = int((binary[a] & ~binary[b]).sum())
        b_only = int((~binary[a] & binary[b]).sum())
        rows.append(
            {
                "Gene A": a,
                "Gene B": b,
                "Tau_00": 0.7,
                "Tau_10": 0.1,
                "Tau_01": 0.15,
                "Tau_11": 0.05,
                "_00_": len(counts) - both - a_only - b_only,
                "_10_": a_only,
                "_01_": b_only,
                "_11_": both,
                "Rho": rho,
                "Log Odds Ratio": np.nan,
                "Likelihood Ratio": lrt,
                "Wald Statistic": np.nan,
                "Effect Identifiability": effect,
                "Fit Converged": True,
                "Fit Iterations": 3,
                "Fit Last LL Gain": 1e-12,
                "Fit Fixed-Point Residual": 0.0,
                "Fit KKT Residual": 0.0,
                "Null Log Likelihood": -10.0,
                "Alternative Log Likelihood": -10.0 + lrt / 2,
            },
        )
    return pd.DataFrame(rows)


def _inference(tasks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"gene_a": [a for a, _ in PAIRS], "gene_b": [b for _, b in PAIRS]},
    )
    for provider in postprocess.BMRS:
        task = tasks[provider]
        statistics = postprocess._provider_statistics(  # noqa: SLF001
            task["Likelihood Ratio"].to_numpy(dtype=np.float64),
            task["Rho"],
            task["Effect Identifiability"],
        )
        diagnostics = postprocess._raw_fit_diagnostics(task, label=provider)  # noqa: SLF001
        for name, values in {**statistics, **diagnostics}.items():
            frame[f"{provider}_{name}"] = (
                values.to_numpy() if isinstance(values, pd.Series) else values
            )
    return frame[list(postprocess.result_columns())]


def _crossings(frame: pd.DataFrame, provider: str, adjustment: str) -> dict[str, int]:
    crossing = frame[f"{provider}_log_{adjustment}_q_value"] <= math.log(0.01)
    direction = frame[f"{provider}_direction"].astype(str)
    return {
        "total": int(crossing.sum()),
        "me": int((crossing & direction.eq("ME")).sum()),
        "co": int((crossing & direction.eq("CO")).sum()),
        "direction_unavailable": int(
            (crossing & ~direction.isin(["ME", "CO"])).sum(),
        ),
    }


def _table_s5_row(frame: pd.DataFrame, n_samples: int) -> dict[str, object]:
    row: dict[str, object] = {
        "cohort": "CHOL",
        "tumors": n_samples,
        "tested_pairs": len(PAIRS),
        "same_base_pairs_excluded": 1,
    }
    for provider in k500.BMRS:
        for adjustment, analysis in (("by", "primary"), ("bh", "sensitivity")):
            prefix = (
                "mutsig_primary_rejection"
                if provider == "mutsig" and analysis == "primary"
                else f"{provider}_descriptive_{analysis}_rule_crossing"
            )
            for key, value in _crossings(frame, provider, adjustment).items():
                row[f"{prefix}_{key}"] = value
    return row


@pytest.fixture
def grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> k500.FocusedInputs:
    """Write one complete, internally consistent synthetic focused grid."""
    monkeypatch.setattr(k500, "TOP_K", len(FEATURES))
    root = tmp_path / "inputs"
    provider_root = root / "providers"
    run_root = root / "run"
    post_root = root / "post"
    counts = _counts()
    count_path = provider_root / "cohorts/CHOL/count_matrix.csv"
    count_path.parent.mkdir(parents=True)
    counts.to_csv(count_path)
    _write_json(
        run_root / "contracts/CHOL.json",
        {
            "top_k": len(FEATURES),
            "features": FEATURES,
            "ordered_features_sha256": feature_sequence_sha256(FEATURES),
            "inputs": {"counts": {"sha256": _sha(count_path)}},
            "pair_policy": {"row_count": len(PAIRS)},
        },
    )

    tasks = {provider: _task(provider, counts) for provider in k500.BMRS}
    records = []
    for provider, task in tasks.items():
        task_root = run_root / "tasks/CHOL" / provider
        task_root.mkdir(parents=True)
        path = task_root / "pairwise_interaction_results.csv"
        task.to_csv(path, index=False)
        manifest_path = task_root / "task_manifest.json"
        _write_json(
            manifest_path,
            {
                "top_k": len(FEATURES),
                "outputs": {path.name: {"sha256": _sha(path)}},
            },
        )
        records.append(
            {
                "cohort": "CHOL",
                "provider": provider,
                "manifest": {
                    "path": f"tasks/CHOL/{provider}/task_manifest.json",
                    "sha256": _sha(manifest_path),
                },
            },
        )
    _write_json(
        run_root / "completion_manifest.json",
        {
            "contract": k500.COMPLETION_CONTRACT,
            "cohorts": ["CHOL"],
            "task_count": 3,
            "tasks": records,
        },
    )
    _write_json(run_root / "run_manifest.json", {"contract": "test"})

    inference = _inference(tasks)
    inference_path = post_root / "CHOL" / postprocess.RESULT_NAME
    inference_path.parent.mkdir(parents=True)
    inference.to_csv(inference_path, index=False)
    cohort_manifest = post_root / "CHOL/cohort_manifest.json"
    _write_json(
        cohort_manifest,
        {
            "cohort": "CHOL",
            "output": {"sha256": _sha(inference_path)},
            "sources": {
                provider: {
                    "path": f"tasks/CHOL/{provider}/pairwise_interaction_results.csv",
                    "sha256": _sha(
                        run_root
                        / f"tasks/CHOL/{provider}/pairwise_interaction_results.csv",
                    ),
                }
                for provider in k500.BMRS
            },
        },
    )
    _write_json(
        post_root / "postprocess_manifest.json",
        {
            "contract": k500.POSTPROCESS_CONTRACT,
            "cohorts": ["CHOL"],
            "cohort_manifests": [
                {"path": "CHOL/cohort_manifest.json", "sha256": _sha(cohort_manifest)},
            ],
        },
    )
    confirmation = root / "confirmation_summary.json"
    _write_json(confirmation, {"composite_overall_gate_pass": True})
    rule_path = root / "rule.json"
    _write_json(
        rule_path,
        {
            "contract": k500.RULE_CONTRACT,
            "inference_status": "reportable",
            "primary_provider": "mutsig",
            "primary_adjustment": "benjamini-yekutieli",
            "sensitivity_adjustment": "benjamini-hochberg",
            "primary_q_threshold": 0.01,
            "sensitivity_q_threshold": 0.01,
            "threshold_comparison": "inclusive-less-than-or-equal",
            "multiplicity": "provider-specific-complete-within-cohort-family",
            "effective_p_policy": (
                "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
            ),
            "direction": "primary-provider-rho-sign-after-nondirectional-rejection",
            "direction_unavailable": (
                "retain-nondirectional-rejection-exclude-from-me-co-lists"
            ),
            "provider_overlap": "descriptive-only-not-an-inferential-vote",
            "thresholds_selected_from_observed_pairs": False,
            "calibration_gate": {"overall_gate_pass": True},
            "calibration_interpretation": "finite-scenario",
            "claim_scope": "finite-scenario-calibrated-nominal-inference",
            "test": "chi-square-one-df-profile-lrt",
            "postprocess_manifest_sha256": _sha(
                post_root / "postprocess_manifest.json",
            ),
            "calibration_confirmation_summary_sha256": _sha(confirmation),
        },
    )
    report_root = root / "report"
    _write_json(
        report_root / "report_manifest.json", {"inference_status": "reportable"},
    )
    pd.DataFrame([_table_s5_row(inference, len(counts))]).to_csv(
        report_root / "table_s5.csv",
        index=False,
    )

    baseline_root = root / "baselines"
    comparison_dir = baseline_root / "TCGA/CHOL"
    comparison_dir.mkdir(parents=True)
    rows = []
    for index, (a, b) in enumerate(reversed(PAIRS)):  # stored out of pair order
        row = {"Gene A": b, "Gene B": a}  # and in the opposite orientation
        for position, (_, source) in enumerate(k500.BASELINE_COLUMNS):
            row[source] = (
                1.5
                if source == "MEGSA S-Score (LRT)"
                else ((index + 1) / 10 + position / 1000)
            )
        rows.append(row)
    comparison = comparison_dir / "comparison_pairwise_interaction_results.csv"
    pd.DataFrame(rows).to_csv(comparison, index=False)
    metadata = comparison_dir / "metadata.json"
    _write_json(
        metadata,
        {
            "cohort": {"id": "TCGA__CHOL"},
            "source_gene_k": len(FEATURES),
            "input": {"sha256": _sha(count_path)},
            "feature_axis": {
                "ordered_features_sha256": feature_sequence_sha256(FEATURES),
                "exclude_same_base_pairs": True,
            },
            "rng": {"seed": 11},
        },
    )
    _write_json(
        baseline_root / "manifest.json",
        {
            "release_id": k500.BASELINE_RELEASE_ID,
            "source_gene_k": len(FEATURES),
            "cohort_count": 1,
            "profile": {"exclude_same_base_pairs": True},
            "provenance": {
                "source_files": {
                    "analysis/build_atlas_baselines.py": _sha(
                        Path("analysis/build_atlas_baselines.py"),
                    ),
                },
                "discover": {"version": "0.9.6"},
            },
            "cohorts": [
                {
                    "id": "TCGA__CHOL",
                    "artifacts": {
                        "comparison": {"sha256": _sha(comparison)},
                        "metadata": {"sha256": _sha(metadata)},
                    },
                },
            ],
        },
    )
    drivers = root / "drivers.tsv"
    drivers.write_text("Hugo Symbol\nTP53\nKRAS\n")
    return k500.FocusedInputs(
        provider_root=provider_root,
        run_root=run_root,
        postprocess_root=post_root,
        rule_path=rule_path,
        confirmation_summary=confirmation,
        report_root=report_root,
        release_receipt=None,
        baseline_root=baseline_root,
        drivers_path=drivers,
        expected_cohorts=1,
    )


def _build(inputs: k500.FocusedInputs, out: Path) -> dict[str, object]:
    return k500.build_release(
        out=out,
        release_id="k500-test",
        inputs=inputs,
        generated_at="2026-09-25T00:00:00Z",
        require_committed_sources=False,
    )


def _table(release: Path, name: str) -> dict[str, np.ndarray]:
    cohort = json.loads((release / "cohorts/TCGA__CHOL/cohort.json").read_text())
    record = cohort["tables"][name]
    raw = gzip.decompress(
        (release / "cohorts/TCGA__CHOL" / record["file"]).read_bytes(),
    )
    return k500.decode_table(raw, record["rows"], record["columns"])


def test_encode_table_aligns_columns_and_round_trips() -> None:
    spec = (("flag", "uint8"), ("value", "float64"), ("index", "uint16"))
    columns = {
        "flag": np.array([1, 0, 1]),
        "value": np.array([0.5, np.nan, -2.0]),
        "index": np.array([3, 2, 1]),
    }

    raw, layout = k500.encode_table(columns, spec)
    decoded = k500.decode_table(raw, 3, layout)

    assert [column["name"] for column in layout] == ["value", "index", "flag"]
    assert all(
        column["offset"] % k500.NUMPY_DTYPES[column["dtype"]].itemsize == 0
        for column in layout
    )
    assert np.array_equal(decoded["value"], columns["value"], equal_nan=True)
    assert decoded["index"].tolist() == [3, 2, 1]
    with pytest.raises(ValueError, match="rows"):
        k500.encode_table({**columns, "index": np.array([1])}, spec)


def test_decision_bits_are_inclusive_on_the_log_scale() -> None:
    log_q = np.array([math.log(0.01), np.nextafter(math.log(0.01), 0), math.log(0.2)])

    bits = k500.decision_bits(log_q)
    primary = 1 << k500.DECISION_THRESHOLDS.index(0.01)

    assert bool(bits[0] & primary)
    assert not bits[1] & primary
    assert bits[1] & (1 << k500.DECISION_THRESHOLDS.index(0.05))
    assert bits[2] == 0


def test_direction_ranks_follow_the_revision_report_order() -> None:
    me, co, na = (k500.DIRECTION_CODES[label] for label in ("ME", "CO", "unavailable"))
    direction = np.array([me, me, co, na, me])
    log_q = np.array([-5.0, -5.0, -1.0, -9.0, -7.0])
    log_p = np.array([-6.0, -6.0, -2.0, -9.0, -8.0])
    rho = np.array([-0.2, -0.9, 0.4, 0.0, -0.1])

    ranks = k500.direction_ranks(direction, log_q, log_p, rho)

    assert ranks.tolist() == [3, 2, 1, 0, 1]


def test_build_release_publishes_the_sealed_family_exactly(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    release = tmp_path / "k500-test"

    manifest = _build(grid, release)

    assert manifest["coverage"]["cohorts"] == 1
    assert k500.verify_release(release) == {"cohorts": 1, "tables": 5}
    inference = pd.read_csv(
        grid.postprocess_root / "CHOL" / postprocess.RESULT_NAME,
        float_precision="round_trip",
    )
    pairs = _table(release, "pairs")
    cohort = json.loads((release / "cohorts/TCGA__CHOL/cohort.json").read_text())
    assert [
        (cohort["features"][a], cohort["features"][b])
        for a, b in zip(pairs["a"], pairs["b"], strict=True)
    ] == PAIRS
    for provider in k500.BMRS:
        table = _table(release, provider)
        for name in ("lrt", "log_by_q", "by_q", "log_bh_q", "rho"):
            source = {
                "lrt": "likelihood_ratio",
                "log_by_q": "log_by_q_value",
                "by_q": "by_q_value",
                "log_bh_q": "log_bh_q_value",
                "rho": "rho",
            }[name]
            assert np.array_equal(
                table[name],
                inference[f"{provider}_{source}"].to_numpy(dtype=float),
                equal_nan=True,
            )
        labels = [cohort["enums"]["direction"][code] for code in table["direction"]]
        assert labels == inference[f"{provider}_direction"].astype(str).tolist()
    baselines = _table(release, "baselines")
    # Stored reversed, so pair i maps back to source row (n - 1 - i).
    assert baselines["fisher_me_p"].tolist() == pytest.approx(
        [(len(PAIRS) - i) / 10 for i in range(len(PAIRS))],
    )
    index = json.loads((release / "index.json").read_text())
    assert (
        index["cohorts"][0]["model_summaries"]["mutsig"]["by_q_le_0_01"]["total"]
        == (_crossings(inference, "mutsig", "by")["total"])
    )
    assert set(cohort["drivers"]) == {"TP53", "KRAS"}
    with pytest.raises(FileExistsError):
        _build(grid, release)


def test_build_release_rejects_disagreement_with_table_s5(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    table = pd.read_csv(grid.report_root / "table_s5.csv")
    table.loc[0, "mutsig_primary_rejection_total"] += 1
    table.to_csv(grid.report_root / "table_s5.csv", index=False)

    with pytest.raises(ValueError, match="published Table S5"):
        _build(grid, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_build_release_rejects_a_task_changed_after_completion(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    path = grid.run_root / "tasks/CHOL/dig/pairwise_interaction_results.csv"
    path.write_text(path.read_text().replace("0.4,", "0.41,", 1))

    with pytest.raises(ValueError, match="changed since completion"):
        _build(grid, tmp_path / "release")


def test_build_release_withholds_inference_when_the_gate_failed(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    rule = json.loads(grid.rule_path.read_text())
    rule["calibration_gate"]["overall_gate_pass"] = False
    grid.rule_path.write_text(json.dumps(rule))

    with pytest.raises(RuntimeError, match="withheld"):
        _build(grid, tmp_path / "release")


def test_build_release_rejects_a_drifted_reporting_rule(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    rule = json.loads(grid.rule_path.read_text())
    rule["primary_adjustment"] = "benjamini-hochberg"
    grid.rule_path.write_text(json.dumps(rule))

    with pytest.raises(ValueError, match="frozen revision contract"):
        _build(grid, tmp_path / "release")


def test_build_release_rejects_baselines_on_another_axis(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    path = grid.baseline_root / "TCGA/CHOL/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["feature_axis"]["ordered_features_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="not bound to the focused contract"):
        _build(grid, tmp_path / "release")


def test_verify_release_detects_a_tampered_table(
    grid: k500.FocusedInputs,
    tmp_path: Path,
) -> None:
    release = tmp_path / "k500-test"
    _build(grid, release)
    shard = release / "cohorts/TCGA__CHOL/dialect-mutsig.bin.gz"
    raw = bytearray(gzip.decompress(shard.read_bytes()))
    raw[0] ^= 1
    shard.write_bytes(gzip.compress(bytes(raw), mtime=0))

    with pytest.raises(ValueError, match="compressed hash mismatch"):
        k500.verify_release(release)


def test_default_release_id_is_k_and_utc_date() -> None:
    from datetime import UTC, datetime  # noqa: PLC0415

    moment = datetime(2026, 9, 25, 23, 59, tzinfo=UTC)

    assert default_release_id(500, moment) == "k500-2026-09-25"
    assert default_release_id(100, moment) == "k100-2026-09-25"

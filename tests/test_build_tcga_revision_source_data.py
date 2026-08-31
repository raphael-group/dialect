from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import build_tcga_revision_source_data as source_data
from dialect.stats.revision_inference import adjust_q_values as production_q_values

_DIGEST = "a" * 64
_EVIDENCE_SHA256 = "9" * 64
_FEATURES = ("A_M", "B_M", "C_M")
_PAIRS = (("A_M", "B_M"), ("A_M", "C_M"), ("B_M", "C_M"))


@pytest.fixture
def writable_tmp_path(tmp_path: Path):
    def thaw() -> None:
        for root, directories, files in os.walk(tmp_path):
            root_path = Path(root)
            root_path.chmod(0o700)
            for directory in directories:
                (root_path / directory).chmod(0o700)
            for file_name in files:
                (root_path / file_name).chmod(0o600)

    yield tmp_path.resolve()
    thaw()


def _canonical(value: object) -> bytes:
    return source_data._canonical_json(value) + b"\n"  # noqa: SLF001


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _resize_frozen_file(path: Path, size_bytes: int) -> None:
    path.parent.chmod(0o700)
    path.chmod(0o600)
    with path.open("r+b") as stream:
        stream.truncate(size_bytes)
    path.chmod(0o400)
    path.parent.chmod(0o500)


def _pair_sha256(pairs=_PAIRS) -> str:
    return source_data._sequence_sha256(  # noqa: SLF001
        [f"{gene_a}\t{gene_b}" for gene_a, gene_b in pairs],
    )


def _implementation(*, omit: str | None = None) -> dict[str, object]:
    paths = {
        "analysis/postprocess_tcga_revision_k500.py",
        "analysis/run_tcga_revision_k500.py",
        "src/dialect/data/revision_approval.py",
        "src/dialect/data/revision_fit_policy.py",
        "src/dialect/models/interaction.py",
        "src/dialect/stats/revision_inference.py",
    }
    if omit is not None:
        paths.remove(omit)
    file_hashes = {path: _sha(path.encode()) for path in sorted(paths)}
    return {
        "files": file_hashes,
        "combined_sha256": _sha(source_data._canonical_json(file_hashes)),  # noqa: SLF001
    }


def _policy_receipts() -> dict[str, dict[str, object]]:
    return {
        decision_id: {
            "canonical_artifact_path": f"{decision_id.lower()}.json",
            "canonical_artifact_sha256": _sha(_decision_artifact(decision_id)),
            "canonical_artifact_size_bytes": len(_decision_artifact(decision_id)),
            "contract": f"{decision_id.lower()}-contract-v1",
            "decision_digest": _sha(f"decision-{decision_id}".encode()),
            "decision_id": decision_id,
            "payload_sha256": _sha(
                source_data._canonical_json(_decision_payload(decision_id)),  # noqa: SLF001
            ),
        }
        for decision_id in (f"D{index}" for index in range(3, 7))
    }


def _root_approvals(*, certified: bool = False) -> dict[str, object]:
    return {
        "input": {"path": "/synthetic/input.json", "sha256": "2" * 64},
        "fit": {"path": "/synthetic/fit.json", "sha256": "3" * 64},
        "inspect": {
            "path": "/synthetic/inspect.json",
            "sha256": "4" * 64,
            "authorized_stage": "inspect-tcga-k500",
        },
        "calibration": (
            {
                "path": "/synthetic/calibration.json",
                "sha256": "5" * 64,
                "schema": source_data.STAGE_SCOPED_APPROVAL_SCHEMA_V6,
                "authorized_stage": "calibration",
                "decision_digests": {
                    f"D{index}": str(index) * 64 for index in range(1, 7)
                },
            }
            if certified
            else None
        ),
    }


def _direction_policy() -> dict[str, object]:
    return {
        "provider_rule": "rho-negative-me-positive-co-zero-neutral",
        "undefined_rho_rule": "unavailable",
        "consensus_rule": "unanimous-me-or-co-else-not-unanimous",
        "reporting_layer": "descriptive-post-rejection",
        "directional_fdr_control": False,
    }


def _tested_family() -> dict[str, object]:
    return {
        "top_k": 3,
        "feature_ranking": "descending-total-eligible-mutation-event-count",
        "tie_break": "canonical-count-matrix-column-order",
        "provider_support": "shared-native-cbase-dig-mutsig",
        "pair_construction": "all-unordered-pairs-of-ordered-feature-axis",
        "same_base_missense_nonsense": "exclude-before-fitting-and-testing",
        "epsilon_pretest_filter": "none",
        "marginal_effect_pretest_filter": "none",
        "family": "one-complete-within-cohort-tested-pair-family",
    }


def _multiplicity_manifest() -> dict[str, object]:
    return {
        "computed_methods": ["by", "bh"],
        "primary_method": "by",
        "primary_q_threshold": 0.01,
        "primary_reporting_layer": "confirmatory-conditional-on-valid-marginals",
        "sensitivity_method": "bh",
        "sensitivity_q_threshold": 0.01,
        "sensitivity_reporting_layer": "nominal-sensitivity",
        "descriptive_methods": ["by", "bh"],
        "descriptive_q_threshold": 0.05,
        "descriptive_reporting_layer": "descriptive",
        "threshold_comparison": "inclusive-less-than-or-equal",
    }


def _d3_policy() -> dict[str, object]:
    return {
        "primary_provider": "cbase",
        "sensitivity_providers": ["dig", "mutsig"],
        "all_three_conjunction_role": "secondary",
        "burden_dependent_switching": False,
        "rationale": "synthetic signed policy fixture",
        "mutsig_support": {"support_contract": "synthetic-native-support"},
        "implementation_binding": {"runner_sha256": "a" * 64},
    }


def _d5_policy() -> dict[str, object]:
    multiplicity = _multiplicity_manifest()
    multiplicity.pop("computed_methods")
    return {
        "conjunction": {
            "mode": "nondirectional-max-p-iut",
            "component_order": list(source_data.BMRS),
            "valid_component_statuses": list(
                source_data.VALID_COMPONENT_STATUSES,
            ),
            "p_value_combiner": "max(p_cbase,p_dig,p_mutsig)",
            "invalid_component": "fail-cohort-conjunction-no-p-value",
            "missing_component": "fail-cohort-conjunction-no-p-value",
            "sign_discordance": "retain-max-p-direction-not-unanimous",
            "effect_unidentifiable": "retain-valid-p-direction-unavailable",
            "direction_affects_p_or_q": False,
        },
        "direction_annotation": _direction_policy(),
        "tested_family": _tested_family(),
        "multiplicity": multiplicity,
        "component_failure_semantics": (
            "task-abort-no-published-row-no-p-one-substitution"
        ),
    }


def _decision_payload(decision_id: str) -> dict[str, object]:
    return {
        "D3": _d3_policy(),
        "D4": {"contract": "synthetic-d4"},
        "D5": _d5_policy(),
        "D6": {"contract": "synthetic-d6"},
    }.get(decision_id, {"value": decision_id})


def _decision_artifact(decision_id: str) -> bytes:
    return _canonical(
        {
            "contract": f"{decision_id.lower()}-contract-v1",
            "decision_id": decision_id,
            "payload": _decision_payload(decision_id),
            "schema": "synthetic-machine-decision-v1",
        },
    )


def _fit_decision_records() -> list[dict[str, object]]:
    return [
        {
            "decision_id": decision_id,
            "contract": f"{decision_id.lower()}-contract-v1",
            "canonical_artifact_sha256": _sha(_decision_artifact(decision_id)),
            "canonical_artifact_size_bytes": len(_decision_artifact(decision_id)),
            "payload_sha256": _sha(
                source_data._canonical_json(_decision_payload(decision_id)),  # noqa: SLF001
            ),
        }
        for decision_id in (f"D{index}" for index in range(1, 7))
    ]


def _rows(*, pair_order=_PAIRS, certified: bool = False) -> bytes:
    pair_order = tuple(pair_order)
    p_value = 0.001 if certified else 0.5
    serialized_p = format(p_value, ".17g")
    by_q = format(
        float(
            source_data._adjust_q_values(  # noqa: SLF001
                [p_value] * len(pair_order),
                method="by",
            )[0],
        ),
        ".17g",
    )
    bh_q = format(
        float(
            source_data._adjust_q_values(  # noqa: SLF001
                [p_value] * len(pair_order),
                method="bh",
            )[0],
        ),
        ".17g",
    )
    eligible = certified
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=source_data.OUTPUT_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    pair_sha256 = _pair_sha256()
    provider_sources = dict(
        zip(source_data.BMRS, ("a" * 64, "b" * 64, "c" * 64), strict=True),
    )
    for gene_a, gene_b in pair_order:
        row = {
            "schema": source_data.POSTPROCESS_COHORT_SCHEMA,
            "derivation_contract": source_data.DERIVATION_CONTRACT,
            "d5_contract": source_data.D5_CONTRACT,
            "cohort": "TEST",
            "gene_a": gene_a,
            "gene_b": gene_b,
            "conjunction_p_value": serialized_p,
            "consensus_direction": "unanimous-me",
            "d3_conjunction_role": "secondary",
            "by_q_value": by_q,
            "bh_q_value": bh_q,
            "marginal_validity_status": "certified" if certified else "absent",
            "conditional_by_inferential_eligible": str(eligible).lower(),
            "by_q_le_0_01": str(float(by_q) <= 0.01).lower(),
            "conditional_by_q_le_0_01_reportable": str(
                float(by_q) <= 0.01 and eligible,
            ).lower(),
            "bh_q_le_0_01_nominal": str(float(bh_q) <= 0.01).lower(),
            "by_q_le_0_05_descriptive": str(float(by_q) <= 0.05).lower(),
            "bh_q_le_0_05_descriptive": str(float(bh_q) <= 0.05).lower(),
            "cohort_contract_sha256": "d" * 64,
            "ordered_features_sha256": "e" * 64,
            "ordered_pair_sha256": pair_sha256,
        }
        for provider in source_data.BMRS:
            row[f"{provider}_component_status"] = "valid-profile-lrt"
            row[f"{provider}_p_value"] = serialized_p
            row[f"{provider}_direction"] = "me"
            row[f"{provider}_effect_identifiability"] = "full-affine-rank"
            row[f"{provider}_source_sha256"] = provider_sources[provider]
        writer.writerow(row)
    return stream.getvalue().encode()


def _mutate_rows(raw: bytes, **updates: str) -> bytes:
    reader = csv.DictReader(io.StringIO(raw.decode(), newline=""))
    rows = list(reader)
    for row in rows:
        row.update(updates)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=source_data.OUTPUT_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _cohort_manifest(
    csv_bytes: bytes,
    implementation: dict[str, object],
    *,
    certified: bool = False,
) -> dict[str, object]:
    provider_sources = dict(
        zip(source_data.BMRS, ("a" * 64, "b" * 64, "c" * 64), strict=True),
    )
    return {
        "axis": {
            "feature_count": 3,
            "pair_count": 3,
            "ordered_features_sha256": "e" * 64,
            "ordered_pair_sha256": _pair_sha256(),
            "cohort_contract_sha256": "d" * 64,
        },
        "cohort": "TEST",
        "complete_family_required": True,
        "component_failure_semantics": (
            "task-abort-no-published-row-no-p-one-substitution"
        ),
        "component_order": list(source_data.BMRS),
        "components": {
            provider: {
                "pairwise_sha256": provider_sources[provider],
                "raw_schema": list(source_data.RAW_PAIRWISE_COLUMNS),
            }
            for provider in source_data.BMRS
        },
        "d3_provider_hierarchy": {**_d3_policy(), "synthetic_qa_default": False},
        "d5_contract": source_data.D5_CONTRACT,
        "derivation_contract": source_data.DERIVATION_CONTRACT,
        "direction": _direction_policy(),
        "implementation": implementation,
        "marginal_validity": {
            "status": "certified" if certified else "absent",
            "conditional_by_inferential_eligible": certified,
            "evidence_id": "synthetic-certified-evidence" if certified else None,
            "artifact_sha256": _EVIDENCE_SHA256 if certified else None,
            "correction_selection_affected": False,
            "q_values_affected": False,
        },
        "multiplicity": _multiplicity_manifest(),
        "output": {
            "name": source_data.POSTPROCESS_CSV_NAME,
            "bytes": len(csv_bytes),
            "rows": 3,
            "sha256": _sha(csv_bytes),
            "columns": list(source_data.OUTPUT_COLUMNS),
        },
        "p_value_combiner": "max(p_cbase,p_dig,p_mutsig)",
        "pair_filtering": False,
        "pair_ranking": False,
        "production_authority": {
            "grid_authority_sha256": "placeholder",
            "sealed_completion_sha256": "1" * 64,
            "canonical_input_manifest_sha256": "c" * 64,
            "provider_input_manifest_sha256": "d" * 64,
            "input_approval_sha256": "2" * 64,
            "fit_approval_sha256": "3" * 64,
            "inspect_approval_sha256": "4" * 64,
            "calibration_approval_sha256": "5" * 64 if certified else None,
            "marginal_validity_evidence_sha256": (
                _EVIDENCE_SHA256 if certified else None
            ),
            "cohort": "TEST",
            "d5_policy_receipt": _policy_receipts()["D5"],
            "task_bindings": [
                {
                    "provider": provider,
                    "contract_sha256": "d" * 64,
                    "pairwise_sha256": provider_sources[provider],
                    "single_gene_sha256": _sha(f"single-{provider}".encode()),
                    "task_manifest_sha256": _sha(f"task-{provider}".encode()),
                }
                for provider in source_data.BMRS
            ],
        },
        "production_eligible": True,
        "schema": source_data.POSTPROCESS_COHORT_SCHEMA,
        "tested_family": _tested_family(),
        "valid_component_statuses": list(source_data.VALID_COMPONENT_STATUSES),
    }


def _approval(config: source_data.SourceDataBuildConfig) -> SimpleNamespace:
    decisions = {}
    for decision_id in source_data.DECISION_IDS:
        content = _decision_artifact(decision_id)
        decisions[decision_id] = SimpleNamespace(
            decision_id=decision_id,
            canonical_artifact=SimpleNamespace(
                path=f"release/{decision_id.lower()}.json",
                sha256=_sha(content),
                size_bytes=len(content),
                content=content,
            ),
        )
    return SimpleNamespace(
        schema=source_data.STAGE_SCOPED_APPROVAL_SCHEMA_V6,
        allowed_stages=(source_data.RELEASE_STAGE,),
        decisions=decisions,
        decision_digests={},
        stage_bindings={
            source_data.RELEASE_STAGE: {
                "canonical_input_manifest_sha256": (
                    config.expected_canonical_input_sha256
                ),
                "provider_input_manifest_sha256": (
                    config.expected_provider_input_sha256
                ),
                "upstream_result_manifest_sha256": (
                    config.expected_postprocess_release_sha256
                ),
            },
        },
        manifest_sha256=config.expected_release_approval_sha256,
    )


def _valid_approval(config: source_data.SourceDataBuildConfig) -> SimpleNamespace:
    approval = _approval(config)
    approval.decision_digests = {
        decision: _sha(decision.encode()) for decision in source_data.DECISION_IDS
    }
    return approval


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit_implementation: str | None = None,
    certified: bool = False,
) -> tuple[source_data.SourceDataBuildConfig, Path, bytes]:
    monkeypatch.setattr(source_data, "TCGA_COHORTS", ("TEST",))
    monkeypatch.setattr(source_data, "TOP_K", 3)
    root = tmp_path / "postprocess"
    cohort_root = root / "TEST"
    cohort_root.mkdir(parents=True)
    csv_bytes = _rows(certified=certified)
    implementation = _implementation(omit=omit_implementation)
    manifest = _cohort_manifest(csv_bytes, implementation, certified=certified)
    (cohort_root / source_data.POSTPROCESS_CSV_NAME).write_bytes(csv_bytes)
    manifest_path = cohort_root / source_data.POSTPROCESS_COHORT_MANIFEST_NAME
    manifest_path.write_bytes(_canonical(manifest))

    authority = {
        "analysis": "tcga-revision-k500",
        "approvals": _root_approvals(certified=certified),
        "bmrs": list(source_data.BMRS),
        "cohorts": ["TEST"],
        "contract": source_data.POSTPROCESS_AUTHORITY_CONTRACT,
        "contracts": [
            {
                "cohort": "TEST",
                "contract_sha256": "d" * 64,
                "file_sha256": "f" * 64,
            },
        ],
        "d3_conjunction_role": "secondary",
        "fit_decisions": _fit_decision_records(),
        "fit_policy": {
            "d3": _d3_policy(),
            "d4": {"contract": "synthetic-d4"},
            "d5": _d5_policy(),
            "d6": {"contract": "synthetic-d6"},
            "receipts": _policy_receipts(),
        },
        "marginal_validity_evidence": {
            "path": "/synthetic/evidence.json" if certified else None,
            "sha256": _EVIDENCE_SHA256 if certified else None,
            "evidence_id": "synthetic-certified-evidence" if certified else None,
            "status": "certified" if certified else "absent",
        },
        "roots": {
            "run_output_root": "/synthetic/run",
            "canonical_input_root": "/synthetic/canonical",
            "canonical_input_manifest_sha256": "c" * 64,
            "provider_input_root": "/synthetic/provider",
            "provider_input_manifest_sha256": "d" * 64,
        },
        "schema": source_data.POSTPROCESS_AUTHORITY_SCHEMA,
        "sealed_completion": {
            "name": "sealed_completion_manifest.json",
            "sha256": "1" * 64,
            "task_count": 3,
        },
        "top_k": 3,
    }
    authority_bytes = _canonical(authority)
    authority_sha256 = _sha(authority_bytes)
    (root / source_data.POSTPROCESS_AUTHORITY_NAME).write_bytes(authority_bytes)
    manifest["production_authority"]["grid_authority_sha256"] = authority_sha256
    manifest_path.write_bytes(_canonical(manifest))
    manifest_bytes = manifest_path.read_bytes()
    publication_binding = _sha(
        source_data._canonical_json(  # noqa: SLF001
            {
                "authority_sha256": authority_sha256,
                "cohort": "TEST",
                "csv_sha256": _sha(csv_bytes),
                "manifest_sha256": _sha(manifest_bytes),
                "row_count": 3,
            },
        ),
    )
    release = {
        "analysis": "tcga-revision-k500",
        "authority_receipt": {
            "name": source_data.POSTPROCESS_AUTHORITY_NAME,
            "sha256": authority_sha256,
        },
        "bmrs": list(source_data.BMRS),
        "cohorts": ["TEST"],
        "contract": source_data.POSTPROCESS_RELEASE_CONTRACT,
        "grid_authority_sha256": authority_sha256,
        "marginal_validity_evidence_sha256": (_EVIDENCE_SHA256 if certified else None),
        "outputs": [
            {
                "cohort": "TEST",
                "csv_sha256": _sha(csv_bytes),
                "directory": "TEST",
                "manifest_sha256": _sha(manifest_bytes),
                "publication_binding_sha256": publication_binding,
                "rows": 3,
            },
        ],
        "schema": source_data.POSTPROCESS_RELEASE_SCHEMA,
        "sealed_completion_sha256": "1" * 64,
        "top_k": 3,
    }
    release_bytes = _canonical(release)
    (root / source_data.POSTPROCESS_RELEASE_MANIFEST_NAME).write_bytes(release_bytes)
    for file_path in root.rglob("*"):
        if file_path.is_file():
            file_path.chmod(0o400)
    cohort_root.chmod(0o500)
    root.chmod(0o500)

    config = source_data.SourceDataBuildConfig(
        postprocess_root=root,
        release_approval_manifest=tmp_path / "approval.json",
        expected_postprocess_release_sha256=_sha(release_bytes),
        expected_postprocess_authority_sha256=authority_sha256,
        expected_postprocess_implementation_sha256=str(
            implementation["combined_sha256"],
        ),
        expected_sealed_completion_sha256="1" * 64,
        expected_canonical_input_sha256="c" * 64,
        expected_provider_input_sha256="d" * 64,
        expected_release_approval_sha256="f" * 64,
        expected_marginal_validity_evidence_sha256=(
            _EVIDENCE_SHA256 if certified else None
        ),
    )
    return config, root, csv_bytes


def test_build_publishes_exact_frozen_source_data(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, csv_bytes = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    output = writable_tmp_path / "source-data"

    receipt = source_data.build_source_data_release(config, output)

    assert receipt.cohort_count == 1
    assert receipt.total_rows == 3
    assert (output / "cohorts" / "TEST.csv").read_bytes() == csv_bytes
    manifest_bytes = (output / source_data.SOURCE_DATA_MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    assert receipt.manifest_sha256 == _sha(manifest_bytes)
    assert manifest["production_eligible"] is True
    assert manifest["authority"]["postprocess_release_manifest_sha256"] == (
        config.expected_postprocess_release_sha256
    )
    assert manifest["dataset"]["total_rows"] == 3
    assert (output.stat().st_mode & 0o777) == 0o500
    assert ((output / "cohorts" / "TEST.csv").stat().st_mode & 0o777) == 0o400


def test_release_approval_fails_before_postprocess_root_opens(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = source_data.SourceDataBuildConfig(
        postprocess_root=writable_tmp_path / "missing",
        release_approval_manifest=writable_tmp_path / "approval.json",
        expected_postprocess_release_sha256=_DIGEST,
        expected_postprocess_authority_sha256=_DIGEST,
        expected_postprocess_implementation_sha256=_DIGEST,
        expected_sealed_completion_sha256=_DIGEST,
        expected_canonical_input_sha256=_DIGEST,
        expected_provider_input_sha256=_DIGEST,
        expected_release_approval_sha256=_DIGEST,
        expected_marginal_validity_evidence_sha256=None,
    )
    opened = False

    def open_spy(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError

    monkeypatch.setattr(source_data, "_open_frozen_directory", open_spy)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("blocked")),
    )

    with pytest.raises(source_data.SourceDataBuildError, match=r"before.*row access"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )
    assert opened is False


def test_wrong_release_digest_fails_before_csv_parser(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    config = replace(config, expected_postprocess_release_sha256="0" * 64)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    parsed = False

    def parse_spy(*_args, **_kwargs):
        nonlocal parsed
        parsed = True
        raise AssertionError

    monkeypatch.setattr(source_data, "_validate_csv_stream", parse_spy)

    with pytest.raises(source_data.SourceDataBuildError, match="release manifest"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )
    assert parsed is False


@pytest.mark.parametrize("mutation", ["authority-anchor", "cohort-manifest"])
def test_metadata_authentication_failures_precede_row_parsing(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    if mutation == "authority-anchor":
        config = replace(config, expected_postprocess_authority_sha256="0" * 64)
    else:
        manifest_path = root / "TEST" / source_data.POSTPROCESS_COHORT_MANIFEST_NAME
        manifest_path.chmod(0o600)
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        manifest_path.chmod(0o400)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    parsed = False

    def parse_spy(*_args, **_kwargs):
        nonlocal parsed
        parsed = True
        raise AssertionError

    monkeypatch.setattr(source_data, "_validate_csv_stream", parse_spy)

    with pytest.raises(source_data.SourceDataBuildError):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )
    assert parsed is False


@pytest.mark.parametrize(
    ("member_kind", "target_label"),
    [
        ("release", "postprocess release manifest"),
        ("authority", "postprocess authority"),
        ("cohort", "TEST postprocess manifest"),
    ],
)
def test_sparse_oversize_postprocess_metadata_fails_before_read_or_parse(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_kind: str,
    target_label: str,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    target = {
        "release": root / source_data.POSTPROCESS_RELEASE_MANIFEST_NAME,
        "authority": root / source_data.POSTPROCESS_AUTHORITY_NAME,
        "cohort": (root / "TEST" / source_data.POSTPROCESS_COHORT_MANIFEST_NAME),
    }[member_kind]
    _resize_frozen_file(
        target,
        source_data._MAX_POSTPROCESS_METADATA_BYTES + 1,  # noqa: SLF001
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original_read = source_data._read_stable_descriptor  # noqa: SLF001
    original_parse = source_data._parse_canonical_json  # noqa: SLF001
    read_labels: list[str] = []
    parsed_labels: list[str] = []

    def read_spy(descriptor, identity, *, label):
        read_labels.append(label)
        return original_read(descriptor, identity, label=label)

    def parse_spy(raw, *, label):
        parsed_labels.append(label)
        return original_parse(raw, label=label)

    monkeypatch.setattr(source_data, "_read_stable_descriptor", read_spy)
    monkeypatch.setattr(source_data, "_parse_canonical_json", parse_spy)
    output = writable_tmp_path / "source-data"
    before = _open_descriptor_count()

    with pytest.raises(source_data.SourceDataBuildError, match=r"bounded.*size"):
        source_data.build_source_data_release(config, output)

    assert target_label not in read_labels
    assert target_label not in parsed_labels
    assert _open_descriptor_count() == before
    assert not output.exists()
    assert list(writable_tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_sparse_oversize_postprocess_csv_fails_before_hash_or_parser(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    target = root / "TEST" / source_data.POSTPROCESS_CSV_NAME
    _resize_frozen_file(
        target,
        source_data._MAX_SOURCE_DATA_COHORT_CSV_BYTES + 1,  # noqa: SLF001
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original_hash = source_data._hash_stable_descriptor  # noqa: SLF001
    hashed_labels: list[str] = []

    def hash_spy(descriptor, identity, *, label):
        hashed_labels.append(label)
        return original_hash(descriptor, identity, label=label)

    def forbid_csv_parser(*_args, **_kwargs):
        msg = "oversize postprocess CSV reached semantic parsing"
        raise AssertionError(msg)

    monkeypatch.setattr(source_data, "_hash_stable_descriptor", hash_spy)
    monkeypatch.setattr(source_data, "_validate_csv_descriptor", forbid_csv_parser)
    monkeypatch.setattr(source_data, "_validate_csv_rows", forbid_csv_parser)
    monkeypatch.setattr(source_data, "_validate_csv_stream", forbid_csv_parser)
    output = writable_tmp_path / "source-data"
    before = _open_descriptor_count()

    with pytest.raises(source_data.SourceDataBuildError, match=r"bounded.*size"):
        source_data.build_source_data_release(config, output)

    assert "TEST source CSV" not in hashed_labels
    assert _open_descriptor_count() == before
    assert not output.exists()
    assert list(writable_tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_incomplete_postprocess_source_closure_fails(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(
        writable_tmp_path,
        monkeypatch,
        omit_implementation="src/dialect/models/interaction.py",
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )

    with pytest.raises(source_data.SourceDataBuildError, match="source closure"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )


def test_duplicate_pair_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source_data, "TOP_K", 3)
    raw = _rows(pair_order=(_PAIRS[0], _PAIRS[0], _PAIRS[2]))
    manifest = _cohort_manifest(raw, _implementation())

    with pytest.raises(source_data.SourceDataBuildError, match="pair axis"):
        source_data._validate_csv_rows(  # noqa: SLF001
            raw,
            cohort="TEST",
            manifest=manifest,
        )


def test_reordered_pairs_change_ordered_axis_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_data, "TOP_K", 3)
    raw = _rows(pair_order=tuple(reversed(_PAIRS)))
    manifest = _cohort_manifest(raw, _implementation())

    observed, rows = source_data._validate_csv_rows(  # noqa: SLF001
        raw,
        cohort="TEST",
        manifest=manifest,
    )

    assert rows == 3
    assert observed != manifest["axis"]["ordered_pair_sha256"]


def test_input_mutation_before_rename_leaves_destination_absent(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, csv_bytes = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original = source_data._validate_pinned_output_tree  # noqa: SLF001
    mutated = False

    def mutate_after_frozen_validation(*args, **kwargs) -> None:
        nonlocal mutated
        original(*args, **kwargs)
        if kwargs["frozen"] and not mutated:
            source_csv = root / "TEST" / source_data.POSTPROCESS_CSV_NAME
            source_csv.chmod(0o600)
            source_csv.write_bytes(csv_bytes.replace(b"0.5", b"0.4", 1))
            mutated = True

    monkeypatch.setattr(
        source_data,
        "_validate_pinned_output_tree",
        mutate_after_frozen_validation,
    )
    output = writable_tmp_path / "source-data"

    with pytest.raises(source_data.SourceDataBuildError, match="changed"):
        source_data.build_source_data_release(config, output)
    assert output.exists() is False


def test_source_mutation_after_atomic_commit_does_not_obscure_success(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, csv_bytes = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original = source_data._rename_no_replace  # noqa: SLF001

    def rename_then_mutate(parent_fd: int, source: str, destination: str) -> None:
        original(parent_fd, source, destination)
        source_csv = root / "TEST" / source_data.POSTPROCESS_CSV_NAME
        source_csv.chmod(0o600)
        source_csv.write_bytes(csv_bytes.replace(b"0.5", b"0.4", 1))

    monkeypatch.setattr(source_data, "_rename_no_replace", rename_then_mutate)
    output = writable_tmp_path / "source-data"

    receipt = source_data.build_source_data_release(config, output)

    assert receipt.output_root == output.as_posix()
    assert output.is_dir()
    assert (output / source_data.SOURCE_DATA_MANIFEST_NAME).is_file()


def test_existing_destination_is_never_replaced(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    output = writable_tmp_path / "source-data"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    copied = False

    def copy_spy(*_args, **_kwargs):
        nonlocal copied
        copied = True
        raise AssertionError

    monkeypatch.setattr(source_data, "_write_file_from_pin", copy_spy)

    with pytest.raises(FileExistsError):
        source_data.build_source_data_release(config, output)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert copied is False


def test_output_cannot_be_published_inside_immutable_source_root(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )

    with pytest.raises(source_data.SourceDataBuildError, match="outside"):
        source_data.build_source_data_release(config, root / "nested-output")


def test_production_config_has_no_arbitrary_scientific_input_seam() -> None:
    names = {item.name for item in fields(source_data.SourceDataBuildConfig)}

    assert names == {
        "postprocess_root",
        "release_approval_manifest",
        "expected_postprocess_release_sha256",
        "expected_postprocess_authority_sha256",
        "expected_postprocess_implementation_sha256",
        "expected_sealed_completion_sha256",
        "expected_canonical_input_sha256",
        "expected_provider_input_sha256",
        "expected_release_approval_sha256",
        "expected_marginal_validity_evidence_sha256",
    }


def test_data_dictionary_is_exact_and_ordered() -> None:
    dictionary = source_data._data_dictionary()  # noqa: SLF001

    assert [column["name"] for column in dictionary["columns"]] == list(
        source_data.OUTPUT_COLUMNS,
    )
    assert len({column["name"] for column in dictionary["columns"]}) == len(
        source_data.OUTPUT_COLUMNS,
    )


def test_release_approval_must_be_singleton_v6_before_root_open(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    approval = _valid_approval(config)
    approval.schema = "dialect-revision-coauthor-approval-v5"
    opened = False

    def open_spy(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError

    monkeypatch.setattr(source_data, "_open_frozen_directory", open_spy)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: approval,
    )

    with pytest.raises(source_data.SourceDataBuildError, match="singleton v6"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )
    assert opened is False


def test_source_csv_is_descriptor_pinned_without_retaining_bytes(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(root, flags)
    try:
        pin = source_data._open_cohort(root_fd, "TEST")  # noqa: SLF001
        try:
            assert pin.csv_file.content is None
            assert pin.csv_file.size_bytes > 0
            assert pin.manifest_file.content is not None
        finally:
            source_data._close_cohort(pin)  # noqa: SLF001
    finally:
        os.close(root_fd)


def test_root_semantic_contract_must_match_cohort_axis(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    authority = json.loads(
        (root / source_data.POSTPROCESS_AUTHORITY_NAME).read_bytes(),
    )
    manifest = json.loads(
        (root / "TEST" / source_data.POSTPROCESS_COHORT_MANIFEST_NAME).read_bytes(),
    )
    authority["contracts"][0]["contract_sha256"] = "0" * 64

    with pytest.raises(source_data.SourceDataBuildError, match="inconsistent"):
        source_data._validate_cohort_production_authority(  # noqa: SLF001
            manifest["production_authority"],
            root_authority=authority,
            config=config,
            cohort="TEST",
            axis=manifest["axis"],
            components=manifest["components"],
        )


def test_task_bindings_must_cover_all_three_providers(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    authority = json.loads(
        (root / source_data.POSTPROCESS_AUTHORITY_NAME).read_bytes(),
    )
    manifest = json.loads(
        (root / "TEST" / source_data.POSTPROCESS_COHORT_MANIFEST_NAME).read_bytes(),
    )
    manifest["production_authority"]["task_bindings"].pop()

    with pytest.raises(source_data.SourceDataBuildError, match="task bindings"):
        source_data._validate_cohort_production_authority(  # noqa: SLF001
            manifest["production_authority"],
            root_authority=authority,
            config=config,
            cohort="TEST",
            axis=manifest["axis"],
            components=manifest["components"],
        )


def test_reversed_duplicate_pair_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_data, "TOP_K", 3)
    raw = _rows(pair_order=(_PAIRS[0], tuple(reversed(_PAIRS[0])), _PAIRS[2]))
    manifest = _cohort_manifest(raw, _implementation())

    with pytest.raises(source_data.SourceDataBuildError, match="pair axis"):
        source_data._validate_csv_rows(  # noqa: SLF001
            raw,
            cohort="TEST",
            manifest=manifest,
        )


def test_incomplete_pair_family_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_data, "TOP_K", 3)
    raw = _rows(pair_order=_PAIRS[:2])
    manifest = _cohort_manifest(raw, _implementation())

    with pytest.raises(source_data.SourceDataBuildError, match="incomplete"):
        source_data._validate_csv_rows(  # noqa: SLF001
            raw,
            cohort="TEST",
            manifest=manifest,
        )


def test_hardlinked_source_csv_is_rejected(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    root.chmod(0o700)
    cohort_root = root / "TEST"
    cohort_root.chmod(0o700)
    source_csv = cohort_root / source_data.POSTPROCESS_CSV_NAME
    os.link(source_csv, writable_tmp_path / "attacker-link.csv")
    cohort_root.chmod(0o500)
    root.chmod(0o500)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )

    with pytest.raises(source_data.SourceDataBuildError, match="single-link"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )


def test_publication_binding_is_recomputed(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    root.chmod(0o700)
    release_path = root / source_data.POSTPROCESS_RELEASE_MANIFEST_NAME
    release_path.chmod(0o600)
    release = json.loads(release_path.read_bytes())
    release["outputs"][0]["publication_binding_sha256"] = "0" * 64
    release_bytes = _canonical(release)
    release_path.write_bytes(release_bytes)
    release_path.chmod(0o400)
    root.chmod(0o500)
    config = replace(
        config,
        expected_postprocess_release_sha256=_sha(release_bytes),
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )

    with pytest.raises(source_data.SourceDataBuildError, match="publication binding"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )


def test_staging_directory_substitution_never_publishes(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original = source_data._require_directory_entry_identity  # noqa: SLF001
    substituted = False

    def substitute(directory_fd, name, expected, *, label) -> None:
        nonlocal substituted
        if label == "staged source-data root" and not substituted:
            os.rename(
                name,
                f"{name}.attacker-held",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir(name, mode=0o500, dir_fd=directory_fd)
            substituted = True
        original(directory_fd, name, expected, label=label)

    monkeypatch.setattr(
        source_data,
        "_require_directory_entry_identity",
        substitute,
    )
    output = writable_tmp_path / "source-data"

    with pytest.raises(source_data.SourceDataBuildError, match="identity changed"):
        source_data.build_source_data_release(config, output)
    assert output.exists() is False


def test_cohort_directory_substitution_never_publishes(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original = source_data._validate_pinned_output_tree  # noqa: SLF001
    substituted = False

    def substitute_after_first_frozen(root_fd, cohorts_fd, **kwargs) -> None:
        nonlocal substituted
        original(root_fd, cohorts_fd, **kwargs)
        if kwargs["frozen"] and not substituted:
            os.fchmod(root_fd, 0o700)
            os.rename(
                source_data.COHORT_DIRECTORY_NAME,
                "cohorts.attacker-held",
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            os.mkdir(source_data.COHORT_DIRECTORY_NAME, mode=0o500, dir_fd=root_fd)
            os.fchmod(root_fd, 0o500)
            substituted = True

    monkeypatch.setattr(
        source_data,
        "_validate_pinned_output_tree",
        substitute_after_first_frozen,
    )
    output = writable_tmp_path / "source-data"

    with pytest.raises(source_data.SourceDataBuildError, match="identity changed"):
        source_data.build_source_data_release(config, output)
    assert output.exists() is False


def test_synthetic_hierarchy_cannot_enter_production(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, csv_bytes = _fixture(writable_tmp_path, monkeypatch)
    authority = json.loads(
        (root / source_data.POSTPROCESS_AUTHORITY_NAME).read_bytes(),
    )
    manifest = _cohort_manifest(csv_bytes, _implementation())
    manifest["d3_provider_hierarchy"]["synthetic_qa_default"] = True

    with pytest.raises(source_data.SourceDataBuildError, match="D3 production"):
        source_data._validate_cohort_policy(  # noqa: SLF001
            manifest,
            root_authority=authority,
            config=config,
            cohort="TEST",
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"cbase_component_status": "synthetic-default"}, "component semantics"),
        (
            {
                "cbase_component_status": "valid-degenerate-null-p-one",
                "cbase_p_value": "1",
                "cbase_direction": "unavailable",
                "cbase_effect_identifiability": "rank-deficient",
            },
            "component semantics",
        ),
        ({"cbase_direction": "unavailable"}, "component semantics"),
        ({"consensus_direction": "discordant"}, "direction or reporting"),
        ({"conditional_by_inferential_eligible": "true"}, "direction or reporting"),
    ],
)
def test_row_semantic_substitutions_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
    match: str,
) -> None:
    monkeypatch.setattr(source_data, "TOP_K", 3)
    raw = _mutate_rows(_rows(), **updates)
    manifest = _cohort_manifest(raw, _implementation())

    with pytest.raises(source_data.SourceDataBuildError, match=match):
        source_data._validate_csv_rows(  # noqa: SLF001
            raw,
            cohort="TEST",
            manifest=manifest,
        )


def test_coherent_but_wrong_complete_family_q_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_data, "TOP_K", 3)
    raw = _mutate_rows(_rows(), by_q_value="0.75")
    manifest = _cohort_manifest(raw, _implementation())

    with pytest.raises(source_data.SourceDataBuildError, match="q-values"):
        source_data._validate_csv_rows(  # noqa: SLF001
            raw,
            cohort="TEST",
            manifest=manifest,
        )


def test_release_must_reauthorize_exact_fit_decision_artifacts(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    approval = _valid_approval(config)
    changed = _canonical(
        {
            "contract": "d5-contract-v1",
            "decision_id": "D5",
            "payload": {"changed": True},
            "schema": "synthetic-machine-decision-v1",
        },
    )
    approval.decisions["D5"].canonical_artifact = SimpleNamespace(
        path="release/d5.json",
        sha256=_sha(changed),
        size_bytes=len(changed),
        content=changed,
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: approval,
    )

    with pytest.raises(source_data.SourceDataBuildError, match="reauthorize D5"):
        source_data.build_source_data_release(
            config,
            writable_tmp_path / "output",
        )


@pytest.mark.parametrize("mutation", ["receipt", "policy"])
def test_fit_policy_must_link_to_fit_decision_artifacts(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    authority = json.loads(
        (root / source_data.POSTPROCESS_AUTHORITY_NAME).read_bytes(),
    )
    if mutation == "receipt":
        authority["fit_policy"]["receipts"]["D5"]["payload_sha256"] = "0" * 64
    else:
        authority["fit_policy"]["d4"]["changed"] = True

    with pytest.raises(source_data.SourceDataBuildError, match="policy linkage"):
        source_data._validate_root_fit_policy(authority)  # noqa: SLF001


def test_root_and_cohort_marginal_states_must_match(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, csv_bytes = _fixture(writable_tmp_path, monkeypatch)
    authority = json.loads(
        (root / source_data.POSTPROCESS_AUTHORITY_NAME).read_bytes(),
    )
    authority["marginal_validity_evidence"]["status"] = "invalid"
    manifest = _cohort_manifest(csv_bytes, _implementation())

    with pytest.raises(source_data.SourceDataBuildError, match="state is inconsistent"):
        source_data._validate_cohort_policy(  # noqa: SLF001
            manifest,
            root_authority=authority,
            config=config,
            cohort="TEST",
        )


def test_certified_evidence_path_builds_and_reports_conditional_flags(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(
        writable_tmp_path,
        monkeypatch,
        certified=True,
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    output = writable_tmp_path / "certified-source-data"

    receipt = source_data.build_source_data_release(config, output)

    assert receipt.total_rows == 3
    manifest = json.loads((output / source_data.SOURCE_DATA_MANIFEST_NAME).read_bytes())
    assert (
        manifest["authority"]["marginal_validity_evidence_sha256"] == _EVIDENCE_SHA256
    )
    with (output / "cohorts" / "TEST.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["marginal_validity_status"] == "certified"
    assert row["conditional_by_inferential_eligible"] == "true"
    assert row["conditional_by_q_le_0_01_reportable"] == "true"


@pytest.mark.parametrize("mutation", ["root-status", "cohort-eligibility"])
def test_certified_evidence_inconsistencies_are_rejected(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config, root, csv_bytes = _fixture(
        writable_tmp_path,
        monkeypatch,
        certified=True,
    )
    authority = json.loads(
        (root / source_data.POSTPROCESS_AUTHORITY_NAME).read_bytes(),
    )
    manifest = _cohort_manifest(csv_bytes, _implementation(), certified=True)
    if mutation == "root-status":
        authority["marginal_validity_evidence"]["status"] = "absent"
        with pytest.raises(source_data.SourceDataBuildError, match="production roots"):
            source_data._validate_authority(config, authority)  # noqa: SLF001
    else:
        manifest["marginal_validity"]["conditional_by_inferential_eligible"] = False
        with pytest.raises(source_data.SourceDataBuildError, match="validity gate"):
            source_data._validate_cohort_policy(  # noqa: SLF001
                manifest,
                root_authority=authority,
                config=config,
                cohort="TEST",
            )


def test_data_dictionary_matches_nonnullable_production_encoding() -> None:
    dictionary = source_data._data_dictionary()  # noqa: SLF001
    columns = {column["name"]: column for column in dictionary["columns"]}

    assert tuple(columns) == source_data.OUTPUT_COLUMNS
    assert all(column["nullable"] is False for column in columns.values())
    for field_name in (
        "cbase_p_value",
        "dig_p_value",
        "mutsig_p_value",
        "conjunction_p_value",
        "by_q_value",
        "bh_q_value",
    ):
        assert columns[field_name]["data_type"] == "number"
    assert columns["conditional_by_inferential_eligible"]["data_type"] == "boolean"
    assert columns["consensus_direction"]["data_type"] == "string"


def test_builder_implementation_inventory_and_digest_sensitivity(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation_path = writable_tmp_path / "revision_approval.py"
    implementation_path.write_bytes(b"first")
    monkeypatch.setattr(
        source_data.revision_approval_module,
        "__file__",
        implementation_path.as_posix(),
    )
    first = source_data._builder_implementation()  # noqa: SLF001
    implementation_path.write_bytes(b"second")
    second = source_data._builder_implementation()  # noqa: SLF001

    assert set(first["files"]) == {
        "analysis/build_tcga_revision_source_data.py",
        "src/dialect/data/revision_approval.py",
    }
    assert (
        first["files"]["src/dialect/data/revision_approval.py"]
        != second["files"]["src/dialect/data/revision_approval.py"]
    )
    assert first["combined_sha256"] != second["combined_sha256"]


def test_builder_implementation_drift_before_commit_is_rejected(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    implementation_path = writable_tmp_path / "revision_approval.py"
    implementation_path.write_bytes(b"first")
    monkeypatch.setattr(
        source_data.revision_approval_module,
        "__file__",
        implementation_path.as_posix(),
    )
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original = source_data._validate_pinned_output_tree  # noqa: SLF001
    mutated = False

    def mutate_after_freeze(*args, **kwargs) -> None:
        nonlocal mutated
        original(*args, **kwargs)
        if kwargs["frozen"] and not mutated:
            implementation_path.write_bytes(b"second")
            mutated = True

    monkeypatch.setattr(
        source_data,
        "_validate_pinned_output_tree",
        mutate_after_freeze,
    )
    output = writable_tmp_path / "source-data"

    with pytest.raises(
        source_data.SourceDataBuildError,
        match="implementation changed",
    ):
        source_data.build_source_data_release(config, output)
    assert output.exists() is False


def test_frozen_output_file_inodes_are_fsynced_before_rename(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    pinned: set[int] = set()
    synced: set[int] = set()
    original_pin = source_data._pin_output_file  # noqa: SLF001
    original_fsync = source_data.os.fsync
    original_rename = source_data._rename_no_replace  # noqa: SLF001

    def pin_spy(*args, **kwargs):
        pin = original_pin(*args, **kwargs)
        pinned.add(pin.descriptor)
        return pin

    def fsync_spy(descriptor: int) -> None:
        synced.add(descriptor)
        original_fsync(descriptor)

    def rename_spy(parent_fd: int, source: str, destination: str) -> None:
        assert pinned
        assert pinned <= synced
        original_rename(parent_fd, source, destination)

    monkeypatch.setattr(source_data, "_pin_output_file", pin_spy)
    monkeypatch.setattr(source_data.os, "fsync", fsync_spy)
    monkeypatch.setattr(source_data, "_rename_no_replace", rename_spy)

    source_data.build_source_data_release(
        config,
        writable_tmp_path / "source-data",
    )


def test_output_pin_hash_is_bracketed_by_live_entry_identity(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = writable_tmp_path / "staging"
    directory.mkdir(mode=0o700)
    member = directory / source_data.README_NAME
    content = b"opaque support bytes\n"
    member.write_bytes(content)
    member.chmod(0o600)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    pin = source_data._pin_output_file(  # noqa: SLF001
        directory_fd,
        source_data.README_NAME,
        expected_sha256=_sha(content),
        expected_size=len(content),
    )
    original_hash = source_data._hash_stable_descriptor  # noqa: SLF001
    swapped = False

    def hash_then_swap(descriptor, identity, *, label):
        nonlocal swapped
        digest = original_hash(descriptor, identity, label=label)
        if not swapped:
            member.rename(writable_tmp_path / "attacker-held-output")
            member.write_bytes(content)
            member.chmod(0o600)
            swapped = True
        return digest

    monkeypatch.setattr(source_data, "_hash_stable_descriptor", hash_then_swap)
    try:
        with pytest.raises(source_data.SourceDataBuildError, match="identity changed"):
            source_data._validate_output_pin(  # noqa: SLF001
                directory_fd,
                pin,
                frozen=False,
            )
    finally:
        os.close(pin.descriptor)
        os.close(directory_fd)


def test_staged_output_fifo_is_rejected_without_blocking(
    writable_tmp_path: Path,
) -> None:
    directory = writable_tmp_path / "staging-fifo"
    directory.mkdir(mode=0o700)
    os.mkfifo(directory / source_data.README_NAME, mode=0o600)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        with pytest.raises(source_data.SourceDataBuildError, match="Cannot pin"):
            source_data._pin_output_file(  # noqa: SLF001
                directory_fd,
                source_data.README_NAME,
                expected_sha256=_sha(b"opaque"),
                expected_size=6,
            )
    finally:
        os.close(directory_fd)


def test_output_tree_final_sweep_rejects_post_member_inventory_change(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original_validate = source_data._validate_output_pin  # noqa: SLF001
    mutated = False

    def validate_then_add(directory_fd, pin, *, frozen):
        nonlocal mutated
        original_validate(directory_fd, pin, frozen=frozen)
        if pin.name == "TEST.csv" and not frozen and not mutated:
            source_data._write_file(directory_fd, "extra.csv", b"opaque")  # noqa: SLF001
            mutated = True

    monkeypatch.setattr(source_data, "_validate_output_pin", validate_then_add)
    output = writable_tmp_path / "source-data"

    with pytest.raises(source_data.SourceDataBuildError, match="inventory"):
        source_data.build_source_data_release(config, output)
    assert mutated is True
    assert output.exists() is False


def test_final_input_sweep_rejects_last_hash_inventory_change(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    original_hash = source_data._hash_stable_descriptor  # noqa: SLF001
    target = f"TEST/{source_data.POSTPROCESS_COHORT_MANIFEST_NAME}"
    target_hashes = 0

    def hash_then_add(descriptor, identity, *, label):
        nonlocal target_hashes
        digest = original_hash(descriptor, identity, label=label)
        if label == target:
            target_hashes += 1
            if target_hashes == 2:
                cohort_root = root / "TEST"
                cohort_root.chmod(0o700)
                extra = cohort_root / "late-extra.bin"
                extra.write_bytes(b"opaque")
                extra.chmod(0o400)
                cohort_root.chmod(0o500)
        return digest

    monkeypatch.setattr(source_data, "_hash_stable_descriptor", hash_then_add)
    output = writable_tmp_path / "source-data"

    with pytest.raises(source_data.SourceDataBuildError):
        source_data.build_source_data_release(config, output)
    assert target_hashes == 2
    assert output.exists() is False


def test_publication_parent_fstat_failure_does_not_leak_descriptor(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    output = writable_tmp_path / "source-data"
    publication_parent = output.parent.resolve()
    original_open = source_data.os.open
    original_fstat = source_data.os.fstat
    parent_descriptor: int | None = None

    def open_spy(path, *args, **kwargs):
        nonlocal parent_descriptor
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == publication_parent and kwargs.get("dir_fd") is None:
            parent_descriptor = descriptor
        return descriptor

    def fstat_spy(descriptor):
        if descriptor == parent_descriptor:
            msg = "synthetic publication-parent fstat failure"
            raise OSError(msg)
        return original_fstat(descriptor)

    monkeypatch.setattr(source_data.os, "open", open_spy)
    monkeypatch.setattr(source_data.os, "fstat", fstat_spy)
    before = _open_descriptor_count()

    with pytest.raises(source_data.SourceDataBuildError, match="publication parent"):
        source_data.build_source_data_release(config, output)
    assert _open_descriptor_count() == before


def test_final_published_root_path_identity_is_revalidated(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _root, _csv = _fixture(writable_tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    output = writable_tmp_path / "source-data"
    original_parent_check = source_data._require_parent  # noqa: SLF001
    checks = 0

    def check_then_replace(lexical, resolved, descriptor, expected):
        nonlocal checks
        original_parent_check(lexical, resolved, descriptor, expected)
        checks += 1
        if checks == 3:
            os.rename(
                output.name,
                "attacker-held-published-root",
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.mkdir(output.name, mode=0o500, dir_fd=descriptor)

    monkeypatch.setattr(source_data, "_require_parent", check_then_replace)

    with pytest.raises(source_data.SourceDataBuildError, match="identity changed"):
        source_data.build_source_data_release(config, output)
    assert checks == 3
    assert output.is_dir()
    assert (writable_tmp_path / "attacker-held-published-root").is_dir()


@pytest.mark.parametrize("method", ["bh", "by"])
def test_independent_q_replay_matches_production_algorithm(method: str) -> None:
    values = [0.7, 0.01, 0.2, 0.01, 1.0, 0.0]
    expected = production_q_values(values, method=method)
    observed = source_data._adjust_q_values(values, method=method)  # noqa: SLF001

    assert [format(float(value), ".17g") for value in expected] == [
        format(value, ".17g") for value in observed
    ]


def test_identical_inputs_produce_identical_manifests(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = writable_tmp_path / "first"
    second_root = writable_tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, _root, _csv = _fixture(first_root, monkeypatch)
    second, _root, _csv = _fixture(second_root, monkeypatch)
    approvals = {
        first.expected_release_approval_sha256: _valid_approval(first),
        second.expected_release_approval_sha256: _valid_approval(second),
    }
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda _path, digest, _stage: approvals[digest],
    )

    source_data.build_source_data_release(first, first_root / "release")
    source_data.build_source_data_release(second, second_root / "release")

    assert (
        first_root / "release" / source_data.SOURCE_DATA_MANIFEST_NAME
    ).read_bytes() == (
        second_root / "release" / source_data.SOURCE_DATA_MANIFEST_NAME
    ).read_bytes()


def _published_source_data_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, source_data.SourceDataReleaseReceipt]:
    config, _postprocess_root, _csv = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        source_data,
        "validate_revision_approval",
        lambda *_args, **_kwargs: _valid_approval(config),
    )
    output = tmp_path / "source-data"
    receipt = source_data.build_source_data_release(config, output)
    return output, receipt


def _rewrite_published_manifest(
    root: Path,
    mutate,
) -> str:
    manifest_path = root / source_data.SOURCE_DATA_MANIFEST_NAME
    root.chmod(0o700)
    manifest_path.chmod(0o600)
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    raw = _canonical(manifest)
    manifest_path.write_bytes(raw)
    manifest_path.chmod(0o400)
    root.chmod(0o500)
    return _sha(raw)


def _open_descriptor_count() -> int:
    fd_root = Path("/proc/self/fd")
    if not fd_root.is_dir():
        fd_root = Path("/dev/fd")
    return sum(1 for _entry in fd_root.iterdir())


def test_result_blind_release_validator_accepts_exact_opaque_tree(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, build_receipt = _published_source_data_fixture(
        writable_tmp_path,
        monkeypatch,
    )

    def forbid_csv_parser(*_args, **_kwargs):
        msg = "source-data release validation parsed CSV rows"
        raise AssertionError(msg)

    monkeypatch.setattr(source_data, "_validate_csv_descriptor", forbid_csv_parser)
    monkeypatch.setattr(source_data, "_validate_csv_rows", forbid_csv_parser)
    monkeypatch.setattr(source_data, "_validate_csv_stream", forbid_csv_parser)

    receipt = source_data.validate_source_data_release(
        root,
        build_receipt.manifest_sha256,
    )

    expected_files = (
        root / source_data.SOURCE_DATA_MANIFEST_NAME,
        root / source_data.DATA_DICTIONARY_NAME,
        root / source_data.README_NAME,
        root / source_data.COHORT_DIRECTORY_NAME / "TEST.csv",
    )
    assert receipt == source_data.SourceDataValidationReceipt(
        source_data_root=root.as_posix(),
        manifest_sha256=build_receipt.manifest_sha256,
        file_count=4,
        cohort_count=1,
        total_bytes=sum(path.stat().st_size for path in expected_files),
        total_rows=3,
    )


def test_production_source_data_inventory_is_exactly_35_files() -> None:
    assert len(source_data.TCGA_COHORTS) == 32
    assert 3 + len(source_data.TCGA_COHORTS) == 35


def test_malformed_manifest_anchor_opens_nothing(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    opened = False

    def open_spy(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError

    monkeypatch.setattr(source_data.os, "open", open_spy)

    with pytest.raises(source_data.SourceDataBuildError, match="lowercase SHA-256"):
        source_data.validate_source_data_release(root, "not-a-digest")
    assert opened is False


def test_wrong_manifest_anchor_opens_only_root_and_manifest(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    original_open = source_data.os.open
    opened: list[str] = []

    def open_spy(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(source_data.os, "open", open_spy)

    with pytest.raises(source_data.SourceDataBuildError, match="independent"):
        source_data.validate_source_data_release(root, "0" * 64)
    assert opened == [root.as_posix(), source_data.SOURCE_DATA_MANIFEST_NAME]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../TEST.csv",
        "/synthetic/TEST.csv",
        "cohorts/../TEST.csv",
        "cohorts//TEST.csv",
    ],
)
def test_manifest_member_path_traversal_fails_before_cohort_open(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    root, _receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    digest = _rewrite_published_manifest(
        root,
        lambda manifest: manifest["dataset"]["cohort_files"][0].__setitem__(
            "path",
            unsafe_path,
        ),
    )
    original_open = source_data.os.open
    opened: list[str] = []

    def open_spy(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(source_data.os, "open", open_spy)

    with pytest.raises(source_data.SourceDataBuildError, match="cohort receipt"):
        source_data.validate_source_data_release(root, digest)
    assert source_data.COHORT_DIRECTORY_NAME not in opened
    assert "TEST.csv" not in opened


@pytest.mark.parametrize(
    ("location", "mutation"),
    [
        ("root", "extra"),
        ("root", "missing"),
        ("cohorts", "extra"),
        ("cohorts", "missing"),
    ],
)
def test_release_validator_rejects_extra_or_missing_tree_members(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    mutation: str,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    directory = root if location == "root" else root / source_data.COHORT_DIRECTORY_NAME
    directory.chmod(0o700)
    if mutation == "extra":
        extra = directory / "extra.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o400)
    else:
        name = source_data.README_NAME if location == "root" else "TEST.csv"
        (directory / name).rename(writable_tmp_path / f"held-{location}")
    directory.chmod(0o500)

    with pytest.raises(source_data.SourceDataBuildError, match="inventory"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


def test_release_validator_rejects_ancestor_symlink(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    alias = writable_tmp_path.parent / f"{writable_tmp_path.name}-alias"
    alias.symlink_to(writable_tmp_path, target_is_directory=True)
    aliased_root = alias / root.name

    with pytest.raises(source_data.SourceDataBuildError, match="symlinked"):
        source_data.validate_source_data_release(
            aliased_root,
            receipt.manifest_sha256,
        )


def test_release_validator_rejects_lexical_path_traversal_before_open(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    traversing = root.parent / ".." / root.parent.name / root.name
    opened = False

    def open_spy(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError

    monkeypatch.setattr(source_data.os, "open", open_spy)

    with pytest.raises(source_data.SourceDataBuildError, match="traversal-free"):
        source_data.validate_source_data_release(
            traversing,
            receipt.manifest_sha256,
        )
    assert opened is False


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_release_validator_rejects_nonregular_cohort_member_without_blocking(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    cohorts = root / source_data.COHORT_DIRECTORY_NAME
    member = cohorts / "TEST.csv"
    held = writable_tmp_path / f"held-{kind}.csv"
    cohorts.chmod(0o700)
    member.rename(held)
    if kind == "symlink":
        member.symlink_to(held)
    elif kind == "fifo":
        os.mkfifo(member, mode=0o400)
    else:
        member.mkdir(mode=0o400)
    cohorts.chmod(0o500)

    with pytest.raises(source_data.SourceDataBuildError):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


def test_release_validator_rejects_manifest_fifo_without_blocking(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    manifest = root / source_data.SOURCE_DATA_MANIFEST_NAME
    root.chmod(0o700)
    manifest.rename(writable_tmp_path / "held-manifest.json")
    os.mkfifo(manifest, mode=0o400)
    root.chmod(0o500)

    with pytest.raises(source_data.SourceDataBuildError):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


def test_release_validator_rejects_hardlinked_cohort_member(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    member = root / source_data.COHORT_DIRECTORY_NAME / "TEST.csv"
    os.link(member, writable_tmp_path / "attacker-hardlink.csv")

    with pytest.raises(source_data.SourceDataBuildError, match="single-link"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


def test_release_validator_rejects_same_byte_descriptor_path_swap(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    member = root / source_data.COHORT_DIRECTORY_NAME / "TEST.csv"
    replacement_bytes = member.read_bytes()
    original_hash = source_data._hash_stable_descriptor  # noqa: SLF001
    swapped = False

    def hash_then_swap(descriptor, identity, *, label):
        nonlocal swapped
        digest = original_hash(descriptor, identity, label=label)
        if label == "source-data cohort file TEST.csv" and not swapped:
            cohorts = member.parent
            cohorts.chmod(0o700)
            member.rename(writable_tmp_path / "attacker-held.csv")
            member.write_bytes(replacement_bytes)
            member.chmod(0o400)
            cohorts.chmod(0o500)
            swapped = True
        return digest

    monkeypatch.setattr(source_data, "_hash_stable_descriptor", hash_then_swap)

    with pytest.raises(source_data.SourceDataBuildError, match="identity changed"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


def test_release_validator_rejects_root_rename_and_recreation(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    original_validate = source_data._validate_source_data_release_manifest  # noqa: SLF001
    moved = False

    def validate_then_replace(manifest):
        nonlocal moved
        plan = original_validate(manifest)
        if not moved:
            root.rename(writable_tmp_path / "attacker-held-root")
            root.mkdir(mode=0o500)
            moved = True
        return plan

    monkeypatch.setattr(
        source_data,
        "_validate_source_data_release_manifest",
        validate_then_replace,
    )

    with pytest.raises(source_data.SourceDataBuildError, match="root changed"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


def test_release_validator_rejects_live_builder_drift(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    drifted = source_data._builder_implementation()  # noqa: SLF001
    drifted["files"] = dict(drifted["files"])
    drifted["files"]["src/dialect/data/revision_approval.py"] = "0" * 64
    drifted["combined_sha256"] = _sha(
        source_data._canonical_json(drifted["files"]),  # noqa: SLF001
    )
    monkeypatch.setattr(source_data, "_builder_implementation", lambda: drifted)

    with pytest.raises(source_data.SourceDataBuildError, match="drifted"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


@pytest.mark.parametrize("member", [source_data.README_NAME, "TEST.csv"])
def test_release_validator_rejects_member_byte_or_size_mismatch(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: str,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    if member == source_data.README_NAME:
        directory = root
        path = root / member
        replacement = path.read_bytes().replace(b"DIALECT", b"D1ALECT", 1)
    else:
        directory = root / source_data.COHORT_DIRECTORY_NAME
        path = directory / member
        replacement = path.read_bytes() + b"x"
    directory.chmod(0o700)
    path.chmod(0o600)
    path.write_bytes(replacement)
    path.chmod(0o400)
    directory.chmod(0o500)

    with pytest.raises(source_data.SourceDataBuildError, match="receipt"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


@pytest.mark.parametrize("member_kind", ["support", "cohort"])
def test_release_validator_rejects_malformed_declared_size_before_member_open(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_kind: str,
) -> None:
    root, _receipt = _published_source_data_fixture(
        writable_tmp_path,
        monkeypatch,
    )

    def mutate(manifest):
        if member_kind == "support":
            manifest["supporting_files"]["readme"]["bytes"] = "not-an-integer"
        else:
            manifest["dataset"]["cohort_files"][0]["bytes"] = False

    digest = _rewrite_published_manifest(root, mutate)
    original_open = source_data.os.open
    opened: list[str] = []

    def open_spy(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(source_data.os, "open", open_spy)
    before = _open_descriptor_count()

    with pytest.raises(source_data.SourceDataBuildError, match="positive integer"):
        source_data.validate_source_data_release(root, digest)

    assert opened == [root.as_posix(), source_data.SOURCE_DATA_MANIFEST_NAME]
    assert _open_descriptor_count() == before
    assert list(writable_tmp_path.glob(f".{root.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("member_kind", "error_pattern"),
    [("support", "receipt size"), ("cohort", "bounded.*size")],
)
def test_release_validator_rejects_sparse_oversize_member_before_hashing(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_kind: str,
    error_pattern: str,
) -> None:
    root, receipt = _published_source_data_fixture(
        writable_tmp_path,
        monkeypatch,
    )
    if member_kind == "support":
        target = root / source_data.README_NAME
        _resize_frozen_file(target, target.stat().st_size + 1)
        digest = receipt.manifest_sha256
        target_label = f"source-data support file {source_data.README_NAME}"
    else:
        target = root / source_data.COHORT_DIRECTORY_NAME / "TEST.csv"
        oversize = source_data._MAX_SOURCE_DATA_COHORT_CSV_BYTES + 1  # noqa: SLF001
        _resize_frozen_file(target, oversize)
        digest = _rewrite_published_manifest(
            root,
            lambda manifest: manifest["dataset"]["cohort_files"][0].__setitem__(
                "bytes",
                oversize,
            ),
        )
        target_label = "source-data cohort file TEST.csv"

    original_hash = source_data._hash_stable_descriptor  # noqa: SLF001
    hashed_labels: list[str] = []

    def hash_spy(descriptor, identity, *, label):
        hashed_labels.append(label)
        return original_hash(descriptor, identity, label=label)

    def forbid_csv_parser(*_args, **_kwargs):
        msg = "source-data release validation parsed CSV rows"
        raise AssertionError(msg)

    monkeypatch.setattr(source_data, "_hash_stable_descriptor", hash_spy)
    monkeypatch.setattr(source_data, "_validate_csv_descriptor", forbid_csv_parser)
    monkeypatch.setattr(source_data, "_validate_csv_rows", forbid_csv_parser)
    monkeypatch.setattr(source_data, "_validate_csv_stream", forbid_csv_parser)
    before = _open_descriptor_count()

    with pytest.raises(source_data.SourceDataBuildError, match=error_pattern):
        source_data.validate_source_data_release(root, digest)

    assert target_label not in hashed_labels
    assert _open_descriptor_count() == before
    assert list(writable_tmp_path.glob(f".{root.name}.*.tmp")) == []


def test_release_validator_rejects_noncanonical_nan_manifest_without_member_open(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    manifest_path = root / source_data.SOURCE_DATA_MANIFEST_NAME
    root.chmod(0o700)
    manifest_path.chmod(0o600)
    raw = manifest_path.read_bytes().replace(b'"total_rows":3', b'"total_rows":NaN')
    assert b"NaN" in raw
    manifest_path.write_bytes(raw)
    manifest_path.chmod(0o400)
    root.chmod(0o500)
    original_open = source_data.os.open
    opened: list[str] = []

    def open_spy(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(source_data.os, "open", open_spy)

    with pytest.raises(source_data.SourceDataBuildError, match="not valid JSON"):
        source_data.validate_source_data_release(root, _sha(raw))
    assert source_data.COHORT_DIRECTORY_NAME not in opened
    assert "TEST.csv" not in opened


def test_release_validator_final_boundary_rechecks_live_member_path(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    member = root / source_data.README_NAME
    replacement = member.read_bytes()
    original_live = source_data._require_live_builder_implementation  # noqa: SLF001
    calls = 0

    def mutate_on_second_builder_check(expected):
        nonlocal calls
        original_live(expected)
        calls += 1
        if calls == 2:
            root.chmod(0o700)
            member.rename(writable_tmp_path / "attacker-held-readme.md")
            member.write_bytes(replacement)
            member.chmod(0o400)
            root.chmod(0o500)

    monkeypatch.setattr(
        source_data,
        "_require_live_builder_implementation",
        mutate_on_second_builder_check,
    )

    with pytest.raises(source_data.SourceDataBuildError, match="changed"):
        source_data.validate_source_data_release(root, receipt.manifest_sha256)


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_release_validator_closes_all_descriptors(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    root, receipt = _published_source_data_fixture(writable_tmp_path, monkeypatch)
    before = _open_descriptor_count()
    if outcome == "failure":
        with pytest.raises(source_data.SourceDataBuildError):
            source_data.validate_source_data_release(root, "0" * 64)
    else:
        source_data.validate_source_data_release(root, receipt.manifest_sha256)
    assert _open_descriptor_count() == before


def test_open_cohort_oserror_is_wrapped_without_descriptor_leak(
    writable_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config, root, _csv = _fixture(writable_tmp_path, monkeypatch)
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    original_inventory = source_data._require_inventory  # noqa: SLF001

    def inventory_error(directory_fd, expected, *, label):
        if label == "postprocess cohort TEST":
            msg = "synthetic inventory failure"
            raise OSError(msg)
        return original_inventory(directory_fd, expected, label=label)

    monkeypatch.setattr(source_data, "_require_inventory", inventory_error)
    before = _open_descriptor_count()
    try:
        with pytest.raises(
            source_data.SourceDataBuildError,
            match="postprocess cohort TEST",
        ):
            source_data._open_cohort(root_fd, "TEST")  # noqa: SLF001
        assert _open_descriptor_count() == before
    finally:
        os.close(root_fd)


def test_cli_preserves_legacy_build_flags_and_adds_explicit_modes() -> None:
    build_flags = [
        "--postprocess-root",
        "/synthetic/postprocess",
        "--release-approval-manifest",
        "/synthetic/approval.json",
        "--expected-postprocess-release-sha256",
        "1" * 64,
        "--expected-postprocess-authority-sha256",
        "2" * 64,
        "--expected-postprocess-implementation-sha256",
        "3" * 64,
        "--expected-sealed-completion-sha256",
        "4" * 64,
        "--expected-canonical-input-sha256",
        "5" * 64,
        "--expected-provider-input-sha256",
        "6" * 64,
        "--expected-release-approval-sha256",
        "7" * 64,
        "--output-root",
        "/synthetic/output",
    ]

    assert source_data._parse_cli_arguments(build_flags).command == "build"  # noqa: SLF001
    assert (
        source_data._parse_cli_arguments(  # noqa: SLF001
            ["build", *build_flags],
        ).command
        == "build"
    )
    validate_args = source_data._parse_cli_arguments(  # noqa: SLF001
        [
            "validate",
            "--source-data-root",
            "/synthetic/source-data",
            "--expected-manifest-sha256",
            "8" * 64,
        ],
    )
    assert validate_args.command == "validate"
    assert validate_args.expected_manifest_sha256 == "8" * 64

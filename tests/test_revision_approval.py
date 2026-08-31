from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from dialect.data import revision_approval as approval_module
from dialect.data.revision_approval import (
    APPROVAL_SCHEMA,
    APPROVERS,
    CALIBRATION_STAGE,
    COMPARATORS_STAGE,
    DECISION_ALLOWED_STAGES,
    DECISION_IDS,
    DEFERRED_DISPOSITION,
    EVIDENCE_LINE_SCHEMA,
    FIT_SEALED_TCGA_K500_STAGE,
    INSPECT_TCGA_K500_STAGE,
    MATERIALIZE_FINAL_INPUTS_STAGE,
    MSK_STAGE,
    NO_GO_DISPOSITION,
    RELEASE_STAGE,
    REVISION_STAGES,
    SOURCE_NOTICE_KINDS,
    STAGE_BINDING_KEYS,
    STAGE_MINIMUM_DECISIONS,
    STAGE_MINIMUM_DECISIONS_V6,
    STAGE_SCOPED_APPROVAL_SCHEMA,
    STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    STAGE_SCOPED_EVIDENCE_LINE_SCHEMA,
    STAGE_SCOPED_EVIDENCE_LINE_SCHEMA_V6,
    RevisionApprovalError,
    canonical_decision_evidence_line,
    compute_decision_authority_sha256,
    validate_revision_approval,
)

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 8, 28, 18, 0, 0, tzinfo=timezone.utc)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _attested_at(approver: str) -> str:
    return f"2026-08-28T12:00:0{APPROVERS.index(approver)}Z"


def _stage_bindings(decision_sha256: str) -> dict[str, dict[str, str]]:
    bindings = {}
    for stage in REVISION_STAGES:
        bindings[stage] = {
            key: _sha256(f"{stage}:{key}".encode()) for key in STAGE_BINDING_KEYS[stage]
        }
    bindings[MATERIALIZE_FINAL_INPUTS_STAGE] = {
        "d1_canonical_artifact_sha256": decision_sha256,
        "d2_canonical_artifact_sha256": decision_sha256,
    }
    return bindings


def _payload(tmp_path: Path) -> dict[str, object]:
    source_content = b"coauthor meeting approval notice\n"
    decision_content = b'{"contract":"canonical decision matrix"}\n'
    source_sha256 = _sha256(source_content)
    decision_sha256 = _sha256(decision_content)
    (tmp_path / "source-notice.txt").write_bytes(source_content)
    (tmp_path / "decision-record.txt").write_bytes(decision_content)
    stage_bindings = _stage_bindings(decision_sha256)
    decisions = [
        {
            "allowed_stages": list(DECISION_ALLOWED_STAGES[decision_id]),
            "attestations": [],
            "canonical_artifact": {
                "path": "decision-record.txt",
                "sha256": decision_sha256,
            },
            "claim_owner": f"claim owner for {decision_id}",
            "decision_id": decision_id,
            "disposition": "go",
            "exact_resolution": f"exact resolution for {decision_id}",
            "execution_owner": f"execution owner for {decision_id}",
            "forbidden_claims": [f"forbidden claim for {decision_id}"],
            "manifest_allowed_stages": list(REVISION_STAGES),
            "manifest_stage_bindings": copy.deepcopy(stage_bindings),
            "permitted_claims": [f"permitted claim for {decision_id}"],
            "recorded_by": "Ahmed Shuaibi",
            "rerun_or_reuse_consequence": f"rerun consequence for {decision_id}",
        }
        for decision_id in DECISION_IDS
    ]
    _write_evidence_and_attestations(tmp_path, decisions)
    return {
        "allowed_stages": list(REVISION_STAGES),
        "decisions": decisions,
        "schema": APPROVAL_SCHEMA,
        "stage_bindings": stage_bindings,
        "source_notice": {
            "file": {
                "path": "source-notice.txt",
                "sha256": source_sha256,
            },
            "kind": "coauthor-authored-email",
            "locator": "Message-ID <dialect-revision-approval@example.org>",
        },
    }


def _write_evidence_and_attestations(
    tmp_path: Path,
    decisions: list[dict[str, object]],
) -> None:
    evidence_paths = {
        "Benjamin J. Raphael": "benjamin-evidence.txt",
        "Uthsav Chitra": "uthsav-evidence.txt",
    }
    evidence_locators = {
        "Benjamin J. Raphael": "Message-ID <benjamin-approval@example.org>",
        "Uthsav Chitra": "Message-ID <uthsav-approval@example.org>",
    }
    evidence_receipts: dict[str, tuple[str, str]] = {}
    for approver in APPROVERS:
        lines = [
            f"From: {approver}",
            "Exact DIALECT revision authority records:",
            *(
                canonical_decision_evidence_line(
                    decision,
                    tmp_path,
                    approver,
                    _attested_at(approver),
                )
                for decision in decisions
            ),
        ]
        content = ("\n".join(lines) + "\n").encode("ascii")
        path = evidence_paths[approver]
        (tmp_path / path).write_bytes(content)
        evidence_receipts[approver] = (path, _sha256(content))

    for decision in decisions:
        artifact_sha256 = decision["canonical_artifact"]["sha256"]
        authority_sha256 = compute_decision_authority_sha256(decision, tmp_path)
        disposition = decision["disposition"]
        decision["attestations"] = [
            {
                "approver": approver,
                "attested_at_utc": _attested_at(approver),
                "attested_disposition": disposition,
                "canonical_artifact_sha256": artifact_sha256,
                "decision_authority_sha256": authority_sha256,
                "evidence": {
                    "file": {
                        "path": evidence_receipts[approver][0],
                        "sha256": evidence_receipts[approver][1],
                    },
                    "kind": "coauthor-authored-email",
                    "locator": evidence_locators[approver],
                },
            }
            for approver in APPROVERS
        ]


def _stage_scoped_payload(
    tmp_path: Path,
    *,
    stage: str = MATERIALIZE_FINAL_INPUTS_STAGE,
    approval_schema: str = STAGE_SCOPED_APPROVAL_SCHEMA,
) -> dict[str, object]:
    """Return a stage-scoped manifest with its schema's exact decisions."""
    payload = _payload(tmp_path)
    schema_contracts = {
        STAGE_SCOPED_APPROVAL_SCHEMA: (STAGE_MINIMUM_DECISIONS, "v5"),
        STAGE_SCOPED_APPROVAL_SCHEMA_V6: (STAGE_MINIMUM_DECISIONS_V6, "v6"),
    }
    stage_minima, schema_version = schema_contracts[approval_schema]
    decision_ids = stage_minima[stage]
    payload["schema"] = approval_schema
    payload["allowed_stages"] = [stage]
    payload["stage_bindings"] = {stage: payload["stage_bindings"][stage]}
    decisions = [
        next(
            decision
            for decision in payload["decisions"]
            if decision["decision_id"] == decision_id
        )
        for decision_id in decision_ids
    ]
    for decision in decisions:
        decision["allowed_stages"] = [stage]
        decision["manifest_allowed_stages"] = [stage]
        decision["manifest_stage_bindings"] = copy.deepcopy(
            payload["stage_bindings"],
        )
    evidence_paths = {
        APPROVERS[0]: f"benjamin-evidence-{schema_version}.txt",
        APPROVERS[1]: f"uthsav-evidence-{schema_version}.txt",
    }
    evidence_receipts: dict[str, tuple[str, str]] = {}
    for approver in APPROVERS:
        lines = [
            canonical_decision_evidence_line(
                decision,
                tmp_path,
                approver,
                _attested_at(approver),
                approval_schema=approval_schema,
            )
            for decision in decisions
        ]
        content = ("\n".join(lines) + "\n").encode("ascii")
        path = evidence_paths[approver]
        (tmp_path / path).write_bytes(content)
        evidence_receipts[approver] = (path, _sha256(content))
    for decision in decisions:
        artifact_sha256 = decision["canonical_artifact"]["sha256"]
        authority_sha256 = compute_decision_authority_sha256(
            decision,
            tmp_path,
            approval_schema=approval_schema,
        )
        decision["attestations"] = [
            {
                "approver": approver,
                "attested_at_utc": _attested_at(approver),
                "attested_disposition": decision["disposition"],
                "canonical_artifact_sha256": artifact_sha256,
                "decision_authority_sha256": authority_sha256,
                "evidence": {
                    "file": {
                        "path": evidence_receipts[approver][0],
                        "sha256": evidence_receipts[approver][1],
                    },
                    "kind": "coauthor-authored-machine-record",
                    "locator": (
                        f"Message-ID <{approver}-{schema_version}@example.org>"
                    ),
                },
            }
            for approver in APPROVERS
        ]
    payload["decisions"] = decisions
    payload["source_notice"] = copy.deepcopy(
        decisions[0]["attestations"][0]["evidence"],
    )
    return payload


def _replace_evidence_bytes(
    tmp_path: Path,
    payload: dict[str, object],
    approver: str,
    content: bytes,
) -> None:
    attestation = next(
        item
        for item in payload["decisions"][0]["attestations"]
        if item["approver"] == approver
    )
    relative_path = attestation["evidence"]["file"]["path"]
    (tmp_path / relative_path).write_bytes(content)
    receipt_sha256 = _sha256(content)
    for decision in payload["decisions"]:
        for item in decision["attestations"]:
            if item["approver"] == approver:
                item["evidence"]["file"]["sha256"] = receipt_sha256
    source_file = payload["source_notice"]["file"]
    if source_file["path"] == relative_path:
        source_file["sha256"] = receipt_sha256


def _set_manifest_allowed_stages(
    tmp_path: Path,
    payload: dict[str, object],
    stages: list[str],
) -> None:
    payload["allowed_stages"] = list(stages)
    payload["stage_bindings"] = {
        stage: payload["stage_bindings"][stage] for stage in stages
    }
    for decision in payload["decisions"]:
        decision["manifest_allowed_stages"] = list(stages)
        decision["manifest_stage_bindings"] = copy.deepcopy(
            payload["stage_bindings"],
        )
    _write_evidence_and_attestations(tmp_path, payload["decisions"])


def _write_manifest(tmp_path: Path, payload: object) -> tuple[Path, str]:
    path = tmp_path / "approval.json"
    content = _canonical_bytes(payload)
    path.write_bytes(content)
    return path, _sha256(content)


def _validate(
    tmp_path: Path,
    payload: object,
    *,
    stage: str = RELEASE_STAGE,
):
    path, digest = _write_manifest(tmp_path, payload)
    return validate_revision_approval(path, digest, stage, now=NOW)


def _set_decision_disposition(
    tmp_path: Path,
    payload: dict[str, object],
    decision_index: int,
    disposition: str,
) -> None:
    decision = payload["decisions"][decision_index]
    decision["disposition"] = disposition
    for attestation in decision["attestations"]:
        attestation["attested_disposition"] = disposition
    _write_evidence_and_attestations(tmp_path, payload["decisions"])


@pytest.mark.parametrize(
    "target",
    ["approval.json", "decision-record.txt", "benjamin-evidence.txt"],
)
def test_approval_authority_rejects_hardlinked_inputs(
    tmp_path: Path,
    target: str,
) -> None:
    payload = _payload(tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    linked = tmp_path / target
    os.link(linked, tmp_path / f"{target}.attacker-link")

    with pytest.raises(RevisionApprovalError, match="single-link regular file"):
        validate_revision_approval(path, digest, RELEASE_STAGE, now=NOW)


def test_approval_authority_rejects_same_descriptor_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    original = approval_module._read_descriptor  # noqa: SLF001

    def mutate_after_read(file_fd: int, read_path: Path, label: str) -> bytes:
        content = original(file_fd, read_path, label)
        if label == "approval manifest":
            with read_path.open("ab") as handle:
                handle.write(b"attacker-byte")
        return content

    monkeypatch.setattr(approval_module, "_read_descriptor", mutate_after_read)

    with pytest.raises(RevisionApprovalError, match="changed while its secure"):
        validate_revision_approval(path, digest, RELEASE_STAGE, now=NOW)


def test_approval_authority_rejects_short_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    original = approval_module._read_descriptor  # noqa: SLF001

    def truncate_read(file_fd: int, read_path: Path, label: str) -> bytes:
        content = original(file_fd, read_path, label)
        return content[:-1] if label == "approval manifest" else content

    monkeypatch.setattr(approval_module, "_read_descriptor", truncate_read)

    with pytest.raises(RevisionApprovalError, match="changed while its secure"):
        validate_revision_approval(path, digest, RELEASE_STAGE, now=NOW)


@pytest.mark.parametrize("stage", REVISION_STAGES)
def test_valid_manifest_authorizes_each_frozen_stage(tmp_path, stage):
    payload = _payload(tmp_path)

    approval = _validate(tmp_path, payload, stage=stage)

    manifest_content = (tmp_path / "approval.json").read_bytes()
    assert approval.manifest_sha256 == _sha256(manifest_content)
    assert approval.schema == APPROVAL_SCHEMA
    assert approval.allowed_stages == REVISION_STAGES
    assert tuple(approval.decisions) == DECISION_IDS
    assert tuple(approval.decision_digests) == DECISION_IDS
    assert all(len(digest) == 64 for digest in approval.decision_digests.values())
    assert approval.source_notice.file.path == "source-notice.txt"
    assert approval.source_notice.kind == "coauthor-authored-email"
    assert approval.source_notice.locator.startswith("Message-ID")
    assert approval.decisions["D1"].attestations[0].approver == ("Benjamin J. Raphael")
    assert approval.decisions["D1"].attestations[1].approver == "Uthsav Chitra"
    assert approval.decisions["D1"].attestations[0].evidence.file.sha256 == _sha256(
        (tmp_path / "benjamin-evidence.txt").read_bytes(),
    )
    assert EVIDENCE_LINE_SCHEMA.encode() in (
        approval.decisions["D1"].attestations[0].evidence.file.content
    )
    assert approval.decisions["D1"].decision_authority_sha256 == (
        approval.decisions["D1"].attestations[0].decision_authority_sha256
    )
    assert approval.decisions["D1"].canonical_artifact.content == (
        b'{"contract":"canonical decision matrix"}\n'
    )
    assert approval.decisions["D1"].canonical_artifact.size_bytes == len(
        approval.decisions["D1"].canonical_artifact.content,
    )
    assert set(STAGE_MINIMUM_DECISIONS[stage]).issubset(approval.decisions)


def test_stage_scoped_manifest_authorizes_only_exact_d1_d2(tmp_path):
    payload = _stage_scoped_payload(tmp_path)

    approval = _validate(
        tmp_path,
        payload,
        stage=MATERIALIZE_FINAL_INPUTS_STAGE,
    )

    assert approval.schema == STAGE_SCOPED_APPROVAL_SCHEMA
    assert approval.allowed_stages == (MATERIALIZE_FINAL_INPUTS_STAGE,)
    assert tuple(approval.decisions) == ("D1", "D2")
    assert tuple(approval.decision_digests) == ("D1", "D2")
    for decision in approval.decisions.values():
        assert decision.allowed_stages == (MATERIALIZE_FINAL_INPUTS_STAGE,)
        assert STAGE_SCOPED_EVIDENCE_LINE_SCHEMA.encode() in (
            decision.attestations[0].evidence.file.content
        )


@pytest.mark.parametrize("stage", REVISION_STAGES)
def test_v5_stage_scoped_schema_uses_historical_minimum_for_every_stage(
    tmp_path,
    stage,
):
    payload = _stage_scoped_payload(tmp_path, stage=stage)

    approval = _validate(tmp_path, payload, stage=stage)

    assert approval.schema == STAGE_SCOPED_APPROVAL_SCHEMA
    assert approval.allowed_stages == (stage,)
    assert tuple(approval.decisions) == STAGE_MINIMUM_DECISIONS[stage]
    assert tuple(approval.decision_digests) == STAGE_MINIMUM_DECISIONS[stage]


@pytest.mark.parametrize("stage", REVISION_STAGES)
def test_v6_stage_scoped_schema_uses_v6_minimum_for_every_stage(tmp_path, stage):
    payload = _stage_scoped_payload(
        tmp_path,
        stage=stage,
        approval_schema=STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    )

    approval = _validate(tmp_path, payload, stage=stage)

    assert approval.schema == STAGE_SCOPED_APPROVAL_SCHEMA_V6
    assert approval.allowed_stages == (stage,)
    assert tuple(approval.decisions) == STAGE_MINIMUM_DECISIONS_V6[stage]
    assert tuple(approval.decision_digests) == STAGE_MINIMUM_DECISIONS_V6[stage]
    assert STAGE_SCOPED_EVIDENCE_LINE_SCHEMA_V6.encode() in (
        next(iter(approval.decisions.values())).attestations[0].evidence.file.content
    )


@pytest.mark.parametrize("attack", ["missing", "extra", "reordered"])
def test_stage_scoped_manifest_rejects_decision_set_drift(tmp_path, attack):
    payload = _stage_scoped_payload(tmp_path)
    if attack == "missing":
        payload["decisions"].pop()
    elif attack == "extra":
        extra = copy.deepcopy(payload["decisions"][-1])
        extra["decision_id"] = "D3"
        payload["decisions"].append(extra)
    else:
        payload["decisions"].reverse()

    with pytest.raises(
        RevisionApprovalError,
        match=r"must contain exactly|ordered exactly",
    ):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_evidence_rejects_d3_injection(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    path = tmp_path / "benjamin-evidence-v5.txt"
    content = path.read_bytes()
    injected = (
        f"{STAGE_SCOPED_EVIDENCE_LINE_SCHEMA}\t{APPROVERS[0]}\tD3\tgo\t"
        f"{_attested_at(APPROVERS[0])}\t{'0' * 64}\t{'1' * 64}\n"
    ).encode("ascii")
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content + injected)

    with pytest.raises(RevisionApprovalError, match="exactly the stage-scoped"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_evidence_rejects_v4_marker_injection(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    path = tmp_path / "benjamin-evidence-v5.txt"
    content = path.read_bytes()
    injected = (
        f"{EVIDENCE_LINE_SCHEMA}\t{APPROVERS[0]}\tD3\tgo\t"
        f"{_attested_at(APPROVERS[0])}\t{'0' * 64}\t{'1' * 64}\n"
    ).encode("ascii")
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content + injected)

    with pytest.raises(RevisionApprovalError, match="must use evidence schema"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_manifest_rejects_split_evidence_files(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    original = tmp_path / "benjamin-evidence-v5.txt"
    split = tmp_path / "benjamin-evidence-v5-split.txt"
    content = original.read_bytes()
    split.write_bytes(content)
    d2_attestation = payload["decisions"][1]["attestations"][0]
    d2_attestation["evidence"]["file"] = {
        "path": split.name,
        "sha256": _sha256(content),
    }

    with pytest.raises(RevisionApprovalError, match="one identical complete evidence"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_manifest_rejects_cross_marker_timestamps(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    original = _attested_at(APPROVERS[0])
    changed = "2026-08-28T12:01:00Z"
    path = tmp_path / "benjamin-evidence-v5.txt"
    content = path.read_bytes()
    d2_line = content.splitlines()[1]
    changed_line = d2_line.replace(original.encode(), changed.encode())
    _replace_evidence_bytes(
        tmp_path,
        payload,
        APPROVERS[0],
        content.replace(d2_line, changed_line),
    )
    payload["decisions"][1]["attestations"][0]["attested_at_utc"] = changed

    with pytest.raises(
        RevisionApprovalError,
        match="one identical attestation timestamp",
    ):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


@pytest.mark.parametrize(
    "prose",
    [
        b"I also adopt D3-D10.\n",
        b"\nI also adopt D3-D10.\n",
        b"\n",
        b"From: Benjamin J. Raphael\n",
    ],
)
def test_stage_scoped_manifest_rejects_any_nonmarker_evidence_bytes(
    tmp_path,
    prose,
):
    payload = _stage_scoped_payload(tmp_path)
    path = tmp_path / "benjamin-evidence-v5.txt"
    content = path.read_bytes()
    attacked = prose + content if prose.startswith(b"From:") else content + prose
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], attacked)

    with pytest.raises(RevisionApprovalError, match="evidence bytes must equal only"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_manifest_rejects_non_machine_record_evidence_kind(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    for decision in payload["decisions"]:
        decision["attestations"][0]["evidence"]["kind"] = "signed-document"

    with pytest.raises(RevisionApprovalError, match="machine-record bytes"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_manifest_rejects_separate_source_notice(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    content = (tmp_path / "benjamin-evidence-v5.txt").read_bytes()
    separate = tmp_path / "separate-source-notice.txt"
    separate.write_bytes(content)
    payload["source_notice"]["file"] = {
        "path": separate.name,
        "sha256": _sha256(content),
    }

    with pytest.raises(RevisionApprovalError, match="exact first approver"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_manifest_rejects_multiple_allowed_stages(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    payload["allowed_stages"].append(FIT_SEALED_TCGA_K500_STAGE)

    with pytest.raises(RevisionApprovalError, match="requires exactly one"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_scoped_materialization_cannot_authorize_fit(tmp_path):
    payload = _stage_scoped_payload(tmp_path)
    path, digest = _write_manifest(tmp_path, payload)

    with pytest.raises(RevisionApprovalError, match="does not authorize stage"):
        validate_revision_approval(
            path,
            digest,
            FIT_SEALED_TCGA_K500_STAGE,
            now=NOW,
        )


def test_returned_authority_is_deeply_immutable(tmp_path):
    approval = _validate(tmp_path, _payload(tmp_path))

    with pytest.raises(FrozenInstanceError):
        approval.manifest_sha256 = "0" * 64
    with pytest.raises(TypeError):
        approval.decisions["D1"] = approval.decisions["D2"]
    with pytest.raises(TypeError):
        approval.decision_digests["D1"] = "0" * 64
    with pytest.raises(FrozenInstanceError):
        approval.decisions["D1"].exact_resolution = "changed"
    with pytest.raises(FrozenInstanceError):
        approval.decisions["D1"].canonical_artifact.content = b"changed"


def test_evidence_helpers_round_trip_separate_multi_decision_emails(tmp_path):
    payload = _payload(tmp_path)

    approval = _validate(tmp_path, payload)

    for approver_index, approver in enumerate(APPROVERS):
        evidence = approval.decisions["D1"].attestations[approver_index].evidence
        marker_lines = [
            line
            for line in evidence.file.content.decode().splitlines()
            if line.startswith(f"{EVIDENCE_LINE_SCHEMA}\t")
        ]
        expected_lines = [
            canonical_decision_evidence_line(
                decision,
                tmp_path,
                approver,
                _attested_at(approver),
            )
            for decision in payload["decisions"]
        ]
        assert marker_lines == expected_lines
        assert len(marker_lines) == len(DECISION_IDS)
        for decision, marker_line in zip(
            approval.decisions.values(),
            marker_lines,
            strict=True,
        ):
            assert decision.decision_authority_sha256 in marker_line


def test_authority_helpers_accept_read_only_decision_mappings(tmp_path):
    payload = _payload(tmp_path)
    decision = MappingProxyType(payload["decisions"][0])

    digest = compute_decision_authority_sha256(decision, tmp_path)
    line = canonical_decision_evidence_line(
        decision,
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    )

    assert digest in line


@pytest.mark.parametrize(
    "field",
    [
        "exact_resolution",
        "execution_owner",
        "claim_owner",
        "rerun_or_reuse_consequence",
        "permitted_claims",
        "forbidden_claims",
    ],
)
def test_coauthor_authority_digest_rejects_operational_field_substitution(
    tmp_path,
    field,
):
    payload = _payload(tmp_path)
    decision = payload["decisions"][0]
    if field.endswith("_claims"):
        decision[field] = [f"substituted {field}"]
    else:
        decision[field] = f"substituted {field}"

    with pytest.raises(RevisionApprovalError, match="does not bind every operational"):
        _validate(tmp_path, payload)


def test_recorded_by_is_manifest_pinned_but_outside_coauthor_authority(tmp_path):
    payload = _payload(tmp_path)
    before = compute_decision_authority_sha256(payload["decisions"][0], tmp_path)
    payload["decisions"][0]["recorded_by"] = "Independent recorder"

    approval = _validate(tmp_path, payload)

    assert approval.decisions["D1"].recorded_by == "Independent recorder"
    assert approval.decisions["D1"].decision_authority_sha256 == before


def test_attestation_authority_digest_substitution_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["decision_authority_sha256"] = "0" * 64

    with pytest.raises(RevisionApprovalError, match="does not bind every operational"):
        _validate(tmp_path, payload)


def test_coauthor_authority_digest_rejects_disposition_substitution(tmp_path):
    payload = _payload(tmp_path)
    decision = payload["decisions"][0]
    decision["disposition"] = NO_GO_DISPOSITION
    for attestation in decision["attestations"]:
        attestation["attested_disposition"] = NO_GO_DISPOSITION

    with pytest.raises(RevisionApprovalError, match="does not bind every operational"):
        _validate(tmp_path, payload)


def test_coauthor_authority_digest_binds_canonical_artifact_path(tmp_path):
    payload = _payload(tmp_path)
    content = (tmp_path / "decision-record.txt").read_bytes()
    (tmp_path / "substituted-decision-record.txt").write_bytes(content)
    payload["decisions"][0]["canonical_artifact"]["path"] = (
        "substituted-decision-record.txt"
    )

    with pytest.raises(RevisionApprovalError, match="does not bind every operational"):
        _validate(tmp_path, payload)


def test_evidence_authority_digest_line_drift_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    line = canonical_decision_evidence_line(
        payload["decisions"][0],
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    ).encode()
    drifted = line[:-1] + (b"0" if line[-1:] != b"0" else b"1")
    content = (tmp_path / "benjamin-evidence.txt").read_bytes().replace(line, drifted)
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content)

    with pytest.raises(RevisionApprovalError, match="exactly one canonical evidence"):
        _validate(tmp_path, payload)


def test_duplicate_evidence_marker_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    line = canonical_decision_evidence_line(
        payload["decisions"][0],
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    ).encode()
    content = (
        (tmp_path / "benjamin-evidence.txt")
        .read_bytes()
        .replace(
            line,
            line + b"\n" + line,
        )
    )
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content)

    with pytest.raises(RevisionApprovalError, match="duplicate or conflicting"):
        _validate(tmp_path, payload)


def test_conflicting_evidence_marker_for_same_decision_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    line = canonical_decision_evidence_line(
        payload["decisions"][0],
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    ).encode()
    conflicting = line[:-1] + (b"0" if line[-1:] != b"0" else b"1")
    content = (
        (tmp_path / "benjamin-evidence.txt")
        .read_bytes()
        .replace(
            line,
            line + b"\n" + conflicting,
        )
    )
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content)

    with pytest.raises(RevisionApprovalError, match="duplicate or conflicting"):
        _validate(tmp_path, payload)


def test_embedded_evidence_marker_substring_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    line = canonical_decision_evidence_line(
        payload["decisions"][0],
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    ).encode()
    content = (
        (tmp_path / "benjamin-evidence.txt")
        .read_bytes()
        .replace(
            line,
            b"> quoted: " + line,
        )
    )
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content)

    with pytest.raises(RevisionApprovalError, match="embeds the evidence schema"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda content: b"\xff" + content, "unambiguous UTF-8"),
        (lambda content: content.replace(b"\n", b"\r\n"), "without CR or NUL"),
        (lambda content: b"\x00" + content, "without CR or NUL"),
    ],
)
def test_evidence_encoding_attacks_are_rejected(tmp_path, mutation, expected):
    payload = _payload(tmp_path)
    content = mutation((tmp_path / "benjamin-evidence.txt").read_bytes())
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content)

    with pytest.raises(RevisionApprovalError, match=expected):
        _validate(tmp_path, payload)


def test_evidence_marker_order_attack_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    first = canonical_decision_evidence_line(
        payload["decisions"][0],
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    ).encode()
    second = canonical_decision_evidence_line(
        payload["decisions"][1],
        tmp_path,
        APPROVERS[0],
        _attested_at(APPROVERS[0]),
    ).encode()
    content = (tmp_path / "benjamin-evidence.txt").read_bytes()
    content = content.replace(first, b"TEMPORARY-MARKER", 1)
    content = content.replace(second, first, 1)
    content = content.replace(b"TEMPORARY-MARKER", second, 1)
    _replace_evidence_bytes(tmp_path, payload, APPROVERS[0], content)

    with pytest.raises(RevisionApprovalError, match="canonical approver and D1-D10"):
        _validate(tmp_path, payload)


def test_historical_stage_minima_remain_unchanged_for_v4_and_v5():
    expected = tuple(f"D{index}" for index in range(1, 7))
    assert STAGE_MINIMUM_DECISIONS[FIT_SEALED_TCGA_K500_STAGE] == expected
    assert STAGE_MINIMUM_DECISIONS[INSPECT_TCGA_K500_STAGE] == expected
    assert STAGE_MINIMUM_DECISIONS[CALIBRATION_STAGE] == ("D4", "D5", "D6")


def test_v4_calibration_still_authorizes_with_d1_no_go(tmp_path):
    payload = _payload(tmp_path)
    _set_manifest_allowed_stages(tmp_path, payload, [CALIBRATION_STAGE])
    _set_decision_disposition(tmp_path, payload, 0, NO_GO_DISPOSITION)

    approval = _validate(tmp_path, payload, stage=CALIBRATION_STAGE)

    assert approval.schema == APPROVAL_SCHEMA
    assert approval.decisions["D1"].disposition == NO_GO_DISPOSITION
    assert all(
        approval.decisions[decision_id].disposition == "go"
        for decision_id in STAGE_MINIMUM_DECISIONS[CALIBRATION_STAGE]
    )


def test_v5_calibration_still_uses_exact_d4_through_d6(tmp_path):
    payload = _stage_scoped_payload(tmp_path, stage=CALIBRATION_STAGE)

    approval = _validate(tmp_path, payload, stage=CALIBRATION_STAGE)

    assert approval.schema == STAGE_SCOPED_APPROVAL_SCHEMA
    assert tuple(approval.decisions) == ("D4", "D5", "D6")
    assert STAGE_SCOPED_EVIDENCE_LINE_SCHEMA.encode() in (
        approval.decisions["D4"].attestations[0].evidence.file.content
    )


def test_v6_calibration_uses_exact_d1_through_d6(tmp_path):
    payload = _stage_scoped_payload(
        tmp_path,
        stage=CALIBRATION_STAGE,
        approval_schema=STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    )

    approval = _validate(tmp_path, payload, stage=CALIBRATION_STAGE)

    assert approval.schema == STAGE_SCOPED_APPROVAL_SCHEMA_V6
    assert tuple(approval.decisions) == tuple(f"D{index}" for index in range(1, 7))
    assert STAGE_SCOPED_EVIDENCE_LINE_SCHEMA_V6.encode() in (
        approval.decisions["D1"].attestations[0].evidence.file.content
    )


def test_v6_calibration_rejects_d4_through_d6_only(tmp_path):
    payload = _stage_scoped_payload(
        tmp_path,
        stage=CALIBRATION_STAGE,
        approval_schema=STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    )
    payload["decisions"] = payload["decisions"][3:]
    path, digest = _write_manifest(tmp_path, payload)

    with pytest.raises(RevisionApprovalError, match="must contain exactly"):
        validate_revision_approval(
            path,
            digest,
            CALIBRATION_STAGE,
            now=NOW,
        )


def test_v6_calibration_rejects_v5_evidence_marker_replay(tmp_path):
    payload = _stage_scoped_payload(
        tmp_path,
        stage=CALIBRATION_STAGE,
        approval_schema=STAGE_SCOPED_APPROVAL_SCHEMA_V6,
    )
    for approver in APPROVERS:
        attestation = next(
            item
            for item in payload["decisions"][0]["attestations"]
            if item["approver"] == approver
        )
        path = tmp_path / attestation["evidence"]["file"]["path"]
        attacked = path.read_bytes().replace(
            STAGE_SCOPED_EVIDENCE_LINE_SCHEMA_V6.encode(),
            STAGE_SCOPED_EVIDENCE_LINE_SCHEMA.encode(),
        )
        _replace_evidence_bytes(tmp_path, payload, approver, attacked)

    with pytest.raises(RevisionApprovalError, match="must use evidence schema"):
        _validate(tmp_path, payload, stage=CALIBRATION_STAGE)


def test_stage_minimum_decision_matrix_is_exact():
    d1_to_d5 = tuple(f"D{index}" for index in range(1, 6))
    d1_to_d6 = (*d1_to_d5, "D6")
    assert dict(STAGE_MINIMUM_DECISIONS) == {
        MATERIALIZE_FINAL_INPUTS_STAGE: ("D1", "D2"),
        FIT_SEALED_TCGA_K500_STAGE: d1_to_d6,
        INSPECT_TCGA_K500_STAGE: d1_to_d6,
        CALIBRATION_STAGE: ("D4", "D5", "D6"),
        COMPARATORS_STAGE: ("D4", "D5", "D8"),
        MSK_STAGE: (*d1_to_d5, "D9"),
        RELEASE_STAGE: DECISION_IDS,
    }
    assert dict(STAGE_MINIMUM_DECISIONS_V6) == {
        **STAGE_MINIMUM_DECISIONS,
        CALIBRATION_STAGE: d1_to_d6,
    }


def test_manifest_must_explicitly_authorize_requested_stage(tmp_path):
    payload = _payload(tmp_path)
    _set_manifest_allowed_stages(
        tmp_path,
        payload,
        [MATERIALIZE_FINAL_INPUTS_STAGE],
    )
    path, digest = _write_manifest(tmp_path, payload)

    with pytest.raises(RevisionApprovalError, match="does not authorize stage"):
        validate_revision_approval(
            path,
            digest,
            FIT_SEALED_TCGA_K500_STAGE,
            now=NOW,
        )


def test_recorder_cannot_expand_top_level_stage_envelope(tmp_path):
    payload = _payload(tmp_path)
    _set_manifest_allowed_stages(
        tmp_path,
        payload,
        [MATERIALIZE_FINAL_INPUTS_STAGE],
    )
    payload["allowed_stages"] = [
        MATERIALIZE_FINAL_INPUTS_STAGE,
        FIT_SEALED_TCGA_K500_STAGE,
    ]

    with pytest.raises(RevisionApprovalError, match="stage_bindings"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_stage_envelope_changes_each_coauthor_authority_digest(tmp_path):
    payload = _payload(tmp_path)
    original = compute_decision_authority_sha256(payload["decisions"][0], tmp_path)
    payload["decisions"][0]["manifest_allowed_stages"] = [
        MATERIALIZE_FINAL_INPUTS_STAGE,
    ]
    payload["decisions"][0]["manifest_stage_bindings"] = {
        MATERIALIZE_FINAL_INPUTS_STAGE: payload["stage_bindings"][
            MATERIALIZE_FINAL_INPUTS_STAGE
        ],
    }

    changed = compute_decision_authority_sha256(payload["decisions"][0], tmp_path)

    assert changed != original


def test_fit_root_binding_changes_each_coauthor_authority_digest(tmp_path) -> None:
    payload = _payload(tmp_path)
    decision = payload["decisions"][0]
    original = compute_decision_authority_sha256(decision, tmp_path)
    decision["manifest_stage_bindings"][FIT_SEALED_TCGA_K500_STAGE][
        "canonical_input_manifest_sha256"
    ] = "0" * 64

    changed = compute_decision_authority_sha256(decision, tmp_path)

    assert changed != original


def test_fit_evidence_cannot_be_replayed_against_other_input_roots(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["stage_bindings"][FIT_SEALED_TCGA_K500_STAGE][
        "provider_input_manifest_sha256"
    ] = "0" * 64
    for decision in payload["decisions"]:
        decision["manifest_stage_bindings"][FIT_SEALED_TCGA_K500_STAGE][
            "provider_input_manifest_sha256"
        ] = "0" * 64

    with pytest.raises(RevisionApprovalError, match="does not bind every operational"):
        _validate(tmp_path, payload, stage=FIT_SEALED_TCGA_K500_STAGE)


@pytest.mark.parametrize("attack", ["missing", "unknown"])
def test_stage_binding_keys_are_exact(tmp_path, attack) -> None:
    payload = _payload(tmp_path)
    binding = payload["stage_bindings"][FIT_SEALED_TCGA_K500_STAGE]
    if attack == "missing":
        binding.pop("provider_input_manifest_sha256")
    else:
        binding["ambient_root_sha256"] = "0" * 64

    with pytest.raises(RevisionApprovalError, match="invalid keys"):
        _validate(tmp_path, payload, stage=FIT_SEALED_TCGA_K500_STAGE)


def test_materialize_binding_must_match_verified_d1_d2_artifacts(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["stage_bindings"][MATERIALIZE_FINAL_INPUTS_STAGE][
        "d1_canonical_artifact_sha256"
    ] = "0" * 64
    for decision in payload["decisions"]:
        decision["manifest_stage_bindings"] = copy.deepcopy(
            payload["stage_bindings"],
        )
    _write_evidence_and_attestations(tmp_path, payload["decisions"])

    with pytest.raises(RevisionApprovalError, match="exact verified D1 and D2"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


@pytest.mark.parametrize("disposition", [NO_GO_DISPOSITION, DEFERRED_DISPOSITION])
def test_signed_no_go_or_deferred_nonrequired_decision_is_preserved(
    tmp_path,
    disposition,
):
    payload = _payload(tmp_path)
    _set_decision_disposition(tmp_path, payload, 6, disposition)
    _set_manifest_allowed_stages(
        tmp_path,
        payload,
        [stage for stage in REVISION_STAGES if stage != RELEASE_STAGE],
    )

    approval = _validate(
        tmp_path,
        payload,
        stage=MATERIALIZE_FINAL_INPUTS_STAGE,
    )

    assert approval.decisions["D7"].disposition == disposition
    assert all(
        attestation.attested_disposition == disposition
        for attestation in approval.decisions["D7"].attestations
    )


@pytest.mark.parametrize("disposition", [NO_GO_DISPOSITION, DEFERRED_DISPOSITION])
def test_required_no_go_or_deferred_decision_cannot_authorize_stage(
    tmp_path,
    disposition,
):
    payload = _payload(tmp_path)
    _set_decision_disposition(tmp_path, payload, 0, disposition)

    with pytest.raises(RevisionApprovalError, match="blocked by a no-go or deferred"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_required_no_go_with_blocked_stages_removed_still_fails_requested_stage(
    tmp_path,
):
    payload = _payload(tmp_path)
    _set_decision_disposition(tmp_path, payload, 0, NO_GO_DISPOSITION)
    blocked_stages = set(DECISION_ALLOWED_STAGES["D1"])
    _set_manifest_allowed_stages(
        tmp_path,
        payload,
        [stage for stage in REVISION_STAGES if stage not in blocked_stages],
    )

    with pytest.raises(RevisionApprovalError, match="does not authorize stage"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_unknown_decision_disposition_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    decision = payload["decisions"][6]
    decision["disposition"] = "maybe"
    for attestation in decision["attestations"]:
        attestation["attested_disposition"] = "maybe"

    with pytest.raises(RevisionApprovalError, match="disposition must be one of"):
        _validate(tmp_path, payload)


def test_unknown_requested_stage_is_rejected_before_parsing(tmp_path):
    path, digest = _write_manifest(tmp_path, _payload(tmp_path))

    with pytest.raises(RevisionApprovalError, match="Unknown revision stage"):
        validate_revision_approval(path, digest, "look-at-results", now=NOW)


@pytest.mark.parametrize("expected", ["A" * 64, "0" * 63, "g" * 64])
def test_expected_manifest_sha_must_be_lowercase_hex(tmp_path, expected):
    path, _digest = _write_manifest(tmp_path, _payload(tmp_path))

    with pytest.raises(RevisionApprovalError, match="lowercase hexadecimal"):
        validate_revision_approval(path, expected, RELEASE_STAGE, now=NOW)


def test_manifest_hash_drift_is_rejected(tmp_path):
    path, _digest = _write_manifest(tmp_path, _payload(tmp_path))

    with pytest.raises(RevisionApprovalError, match="manifest SHA-256 mismatch"):
        validate_revision_approval(path, "0" * 64, RELEASE_STAGE, now=NOW)


def test_duplicate_json_keys_are_rejected(tmp_path):
    payload = _payload(tmp_path)
    content = _canonical_bytes(payload)
    duplicate = content.replace(
        b'{"allowed_stages":',
        b'{"schema":"duplicate","allowed_stages":',
        1,
    )
    path = tmp_path / "approval.json"
    path.write_bytes(duplicate)

    with pytest.raises(RevisionApprovalError, match="Duplicate JSON key: 'schema'"):
        validate_revision_approval(path, _sha256(duplicate), RELEASE_STAGE, now=NOW)


def test_nonfinite_json_constants_are_rejected(tmp_path):
    payload = _payload(tmp_path)
    content = _canonical_bytes(payload).replace(
        b'{"allowed_stages":',
        b'{"not_a_number":NaN,"allowed_stages":',
        1,
    )
    path = tmp_path / "approval.json"
    path.write_bytes(content)

    with pytest.raises(RevisionApprovalError, match="Non-finite JSON constant"):
        validate_revision_approval(path, _sha256(content), RELEASE_STAGE, now=NOW)


def test_numeric_overflow_is_rejected_as_nonfinite_json(tmp_path):
    payload = _payload(tmp_path)
    content = _canonical_bytes(payload).replace(
        b'{"allowed_stages":',
        b'{"not_a_number":1e999,"allowed_stages":',
        1,
    )
    path = tmp_path / "approval.json"
    path.write_bytes(content)

    with pytest.raises(RevisionApprovalError, match="Non-finite JSON number"):
        validate_revision_approval(path, _sha256(content), RELEASE_STAGE, now=NOW)


def test_escaped_lone_surrogate_is_rejected_as_invalid_unicode(tmp_path):
    payload = _payload(tmp_path)
    content = _canonical_bytes(payload).replace(
        b'{"allowed_stages":',
        b'{"invalid_unicode":"\\ud800","allowed_stages":',
        1,
    )
    path = tmp_path / "approval.json"
    path.write_bytes(content)

    with pytest.raises(RevisionApprovalError, match="invalid Unicode scalar"):
        validate_revision_approval(path, _sha256(content), RELEASE_STAGE, now=NOW)


def test_noncanonical_json_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    content = json.dumps(payload, indent=2, sort_keys=False).encode()
    path = tmp_path / "approval.json"
    path.write_bytes(content)

    with pytest.raises(RevisionApprovalError, match="not canonical JSON"):
        validate_revision_approval(path, _sha256(content), RELEASE_STAGE, now=NOW)


@pytest.mark.parametrize(
    ("container", "unknown_key"),
    [("top", "notes"), ("decision", "signature"), ("attestation", "email")],
)
def test_unknown_schema_keys_are_rejected(tmp_path, container, unknown_key):
    payload = _payload(tmp_path)
    if container == "top":
        payload[unknown_key] = "unexpected"
    elif container == "decision":
        payload["decisions"][0][unknown_key] = "unexpected"
    else:
        payload["decisions"][0]["attestations"][0][unknown_key] = "unexpected"

    with pytest.raises(RevisionApprovalError, match=r"unknown=\["):
        _validate(tmp_path, payload)


def test_missing_decision_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"].pop()

    with pytest.raises(RevisionApprovalError, match="exactly D1-D10"):
        _validate(tmp_path, payload)


def test_duplicate_decision_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"][1] = copy.deepcopy(payload["decisions"][0])

    with pytest.raises(RevisionApprovalError, match="Duplicate decision record: D1"):
        _validate(tmp_path, payload)


def test_decisions_must_use_canonical_d1_to_d10_order(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"][0], payload["decisions"][1] = (
        payload["decisions"][1],
        payload["decisions"][0],
    )

    with pytest.raises(RevisionApprovalError, match="ordered exactly D1-D10"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "swapped"])
def test_exact_named_attestations_are_required(tmp_path, mutation):
    payload = _payload(tmp_path)
    attestations = payload["decisions"][0]["attestations"]
    if mutation == "duplicate":
        attestations[1] = copy.deepcopy(attestations[0])
    elif mutation == "missing":
        attestations.pop()
    else:
        attestations.reverse()

    expected = "exactly Benjamin" if mutation == "missing" else "must be ordered"
    with pytest.raises(RevisionApprovalError, match=expected):
        _validate(tmp_path, payload)


def test_attestation_must_explicitly_match_decision_disposition(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["attested_disposition"] = (
        NO_GO_DISPOSITION
    )

    with pytest.raises(RevisionApprovalError, match="must equal the decision"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize(
    "kind",
    ["meeting-transcript", "conversational-summary", "recorder-inference"],
)
def test_source_notice_rejects_inferred_or_transcribed_approval(tmp_path, kind):
    payload = _payload(tmp_path)
    payload["source_notice"]["kind"] = kind

    with pytest.raises(RevisionApprovalError, match="first-party coauthor evidence"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize(
    "kind",
    ["meeting-transcript", "conversational-summary", "recorder-inference"],
)
def test_attestation_rejects_inferred_or_conversational_evidence(tmp_path, kind):
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["evidence"]["kind"] = kind

    with pytest.raises(RevisionApprovalError, match="first-party coauthor evidence"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize("kind", sorted(SOURCE_NOTICE_KINDS))
def test_each_first_party_source_notice_kind_is_accepted(tmp_path, kind):
    payload = _payload(tmp_path)
    payload["source_notice"]["kind"] = kind

    approval = _validate(tmp_path, payload)

    assert approval.source_notice.kind == kind


def test_source_notice_locator_must_be_nonblank(tmp_path):
    payload = _payload(tmp_path)
    payload["source_notice"]["locator"] = " "

    with pytest.raises(RevisionApprovalError, match="nonblank exact string"):
        _validate(tmp_path, payload)


def test_attestation_must_bind_verified_canonical_artifact(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["canonical_artifact_sha256"] = "0" * 64

    with pytest.raises(RevisionApprovalError, match="verified decision artifact"):
        _validate(tmp_path, payload)


def test_email_approvals_require_separate_evidence_files(tmp_path):
    payload = _payload(tmp_path)
    attestations = payload["decisions"][0]["attestations"]
    attestations[1]["evidence"]["file"] = copy.deepcopy(
        attestations[0]["evidence"]["file"],
    )

    with pytest.raises(RevisionApprovalError, match="separate evidence files"):
        _validate(tmp_path, payload)


def test_coauthor_approvals_require_distinct_evidence_locators(tmp_path):
    payload = _payload(tmp_path)
    attestations = payload["decisions"][0]["attestations"]
    attestations[1]["evidence"]["kind"] = attestations[0]["evidence"]["kind"]
    attestations[1]["evidence"]["locator"] = attestations[0]["evidence"]["locator"]

    with pytest.raises(RevisionApprovalError, match="distinct evidence locators"):
        _validate(tmp_path, payload)


def test_stage_scoped_coauthor_records_reject_one_path_with_two_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _stage_scoped_payload(tmp_path)
    shared_path = "benjamin-evidence-v5.txt"
    second_content = (tmp_path / "uthsav-evidence-v5.txt").read_bytes()
    for decision in payload["decisions"]:
        decision["attestations"][1]["evidence"]["file"]["path"] = shared_path

    original = approval_module._read_regular_file  # noqa: SLF001

    def swap_snapshot(
        path: Path,
        label: str,
        *,
        root: Path | None = None,
    ) -> bytes:
        if path.name == shared_path and ".attestations[1].evidence.file" in label:
            return second_content
        return original(path, label, root=root)

    monkeypatch.setattr(approval_module, "_read_regular_file", swap_snapshot)

    with pytest.raises(RevisionApprovalError, match="distinct relative paths"):
        _validate(tmp_path, payload, stage=MATERIALIZE_FINAL_INPUTS_STAGE)


def test_validator_rejects_inconsistent_receipts_for_one_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(tmp_path)
    source_content = (tmp_path / "source-notice.txt").read_bytes()
    shared_path = "benjamin-evidence.txt"
    payload["source_notice"]["file"]["path"] = shared_path
    original = approval_module._read_regular_file  # noqa: SLF001

    def swap_snapshot(
        path: Path,
        label: str,
        *,
        root: Path | None = None,
    ) -> bytes:
        if path.name == shared_path and label == "manifest.source_notice.file":
            return source_content
        return original(path, label, root=root)

    monkeypatch.setattr(approval_module, "_read_regular_file", swap_snapshot)

    with pytest.raises(RevisionApprovalError, match="inconsistent immutable receipts"):
        _validate(tmp_path, payload)


def test_validator_rejects_manifest_path_reused_for_another_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(tmp_path)
    source_content = (tmp_path / "source-notice.txt").read_bytes()
    payload["source_notice"]["file"]["path"] = "approval.json"
    original = approval_module._read_regular_file  # noqa: SLF001

    def swap_snapshot(
        path: Path,
        label: str,
        *,
        root: Path | None = None,
    ) -> bytes:
        if path.name == "approval.json" and label == "manifest.source_notice.file":
            return source_content
        return original(path, label, root=root)

    monkeypatch.setattr(approval_module, "_read_regular_file", swap_snapshot)

    with pytest.raises(RevisionApprovalError, match="inconsistent immutable receipts"):
        _validate(tmp_path, payload)


def test_shared_signed_document_supports_distinct_signature_locators(tmp_path):
    payload = _payload(tmp_path)
    attestations = payload["decisions"][0]["attestations"]
    shared_content = (
        "Signed DIALECT decision authority\n"
        + "\n".join(
            canonical_decision_evidence_line(
                decision,
                tmp_path,
                approver,
                _attested_at(approver),
            )
            for approver in APPROVERS
            for decision in payload["decisions"]
        )
        + "\n"
    ).encode("ascii")
    (tmp_path / "shared-signed-evidence.txt").write_bytes(shared_content)
    shared_file = {
        "path": "shared-signed-evidence.txt",
        "sha256": _sha256(shared_content),
    }
    for index, attestation in enumerate(attestations, start=1):
        attestation["evidence"]["kind"] = "signed-document"
        attestation["evidence"]["locator"] = f"signature-field-{index}"
        attestation["evidence"]["file"] = copy.deepcopy(shared_file)

    approval = _validate(tmp_path, payload)

    d1_attestations = approval.decisions["D1"].attestations
    assert d1_attestations[0].evidence.file.sha256 == (
        d1_attestations[1].evidence.file.sha256
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-28T12:00:00.1Z",
        "2026-08-28T12:00:00+00:00",
        "2026-08-28T12:00:00",
        "2026-02-30T12:00:00Z",
    ],
)
def test_attestation_timestamp_must_be_valid_whole_second_z_utc(
    tmp_path,
    timestamp,
):
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["attested_at_utc"] = timestamp

    with pytest.raises(
        RevisionApprovalError,
        match=r"UTC timestamp|whole-second UTC",
    ):
        _validate(tmp_path, payload)


def test_attestation_timestamp_is_bound_by_exact_evidence_marker(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["attested_at_utc"] = (
        "2026-08-28T12:01:00Z"
    )

    with pytest.raises(RevisionApprovalError, match="exactly one canonical evidence"):
        _validate(tmp_path, payload)


def test_evidence_marker_rejects_noncanonical_timestamp(tmp_path) -> None:
    payload = _payload(tmp_path)
    approver = APPROVERS[0]
    line = canonical_decision_evidence_line(
        payload["decisions"][0],
        tmp_path,
        approver,
        _attested_at(approver),
    ).encode()
    malformed = line.replace(
        _attested_at(approver).encode(),
        b"2026-08-28T12:00:00.0Z",
    )
    content = (
        (tmp_path / "benjamin-evidence.txt")
        .read_bytes()
        .replace(
            line,
            malformed,
        )
    )
    _replace_evidence_bytes(tmp_path, payload, approver, content)

    with pytest.raises(RevisionApprovalError, match="whole-second UTC"):
        _validate(tmp_path, payload)


def test_evidence_helper_requires_exact_whole_second_timestamp(tmp_path) -> None:
    payload = _payload(tmp_path)

    with pytest.raises(RevisionApprovalError, match="whole-second UTC"):
        canonical_decision_evidence_line(
            payload["decisions"][0],
            tmp_path,
            APPROVERS[0],
            "2026-08-28T12:00:00+00:00",
        )


def test_future_attestation_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    payload["decisions"][0]["attestations"][0]["attested_at_utc"] = (
        "2026-08-28T18:00:01Z"
    )

    with pytest.raises(RevisionApprovalError, match="is in the future"):
        _validate(tmp_path, payload)


def test_naive_validation_clock_is_rejected(tmp_path):
    path, digest = _write_manifest(tmp_path, _payload(tmp_path))

    with pytest.raises(RevisionApprovalError, match="timezone-aware"):
        validate_revision_approval(
            path,
            digest,
            RELEASE_STAGE,
            now=datetime(2026, 8, 28, 18, 0, 0),  # noqa: DTZ001
        )


@pytest.mark.parametrize(
    "field",
    [
        "exact_resolution",
        "execution_owner",
        "claim_owner",
        "rerun_or_reuse_consequence",
        "recorded_by",
    ],
)
def test_decision_narrative_fields_must_be_nonblank_exact_text(tmp_path, field):
    payload = _payload(tmp_path)
    payload["decisions"][0][field] = "  "

    with pytest.raises(RevisionApprovalError, match="nonblank exact string"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize("field", ["permitted_claims", "forbidden_claims"])
def test_claim_lists_must_be_nonempty(tmp_path, field):
    payload = _payload(tmp_path)
    payload["decisions"][0][field] = []

    with pytest.raises(RevisionApprovalError, match="at least one claim"):
        _validate(tmp_path, payload)


def test_same_claim_cannot_be_both_permitted_and_forbidden(tmp_path):
    payload = _payload(tmp_path)
    claim = payload["decisions"][0]["permitted_claims"][0]
    payload["decisions"][0]["forbidden_claims"] = [claim]

    with pytest.raises(RevisionApprovalError, match="permits and forbids"):
        _validate(tmp_path, payload)


def test_duplicate_top_level_stage_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    payload["allowed_stages"].append(RELEASE_STAGE)

    with pytest.raises(RevisionApprovalError, match="duplicate stages"):
        _validate(tmp_path, payload)


def test_missing_decision_stage_authority_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    stages = payload["decisions"][0]["allowed_stages"]
    stages.remove(RELEASE_STAGE)

    with pytest.raises(RevisionApprovalError, match="frozen decision matrix"):
        _validate(tmp_path, payload)


def test_unknown_stage_authority_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    payload["allowed_stages"].append("publish-unreviewed-results")

    with pytest.raises(RevisionApprovalError, match="unknown stages"):
        _validate(tmp_path, payload)


@pytest.mark.parametrize(
    "malicious_path",
    [
        "/absolute/source-notice.txt",
        "../source-notice.txt",
        "./source-notice.txt",
        ".",
        "..\\source-notice.txt",
        "source\nnotice.txt",
        "source\tnotice.txt",
    ],
)
def test_artifact_paths_must_be_normalized_relative_and_nontraversing(
    tmp_path,
    malicious_path,
):
    payload = _payload(tmp_path)
    payload["source_notice"]["file"]["path"] = malicious_path

    with pytest.raises(RevisionApprovalError, match="relative POSIX path"):
        _validate(tmp_path, payload)


def test_artifact_hash_drift_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    (tmp_path / "decision-record.txt").write_bytes(b"drifted decision record\n")

    with pytest.raises(RevisionApprovalError, match="SHA-256 mismatch"):
        _validate(tmp_path, payload)


def test_empty_canonical_artifact_is_rejected(tmp_path):
    payload = _payload(tmp_path)
    (tmp_path / "decision-record.txt").write_bytes(b"")

    with pytest.raises(RevisionApprovalError, match="nonempty evidence or contract"):
        _validate(tmp_path, payload)


def test_artifact_sha_must_be_lowercase(tmp_path):
    payload = _payload(tmp_path)
    artifact = payload["decisions"][0]["canonical_artifact"]
    artifact["sha256"] = artifact["sha256"].upper()

    with pytest.raises(RevisionApprovalError, match="lowercase hexadecimal"):
        _validate(tmp_path, payload)


def test_final_artifact_symlink_is_rejected_even_when_bytes_match(tmp_path):
    payload = _payload(tmp_path)
    source_path = tmp_path / "source-notice.txt"
    target = tmp_path / "source-notice-target.txt"
    target.write_bytes(source_path.read_bytes())
    source_path.unlink()
    source_path.symlink_to(target)

    with pytest.raises(RevisionApprovalError, match="must not contain symlinks"):
        _validate(tmp_path, payload)


def test_parent_artifact_symlink_is_rejected_even_when_bytes_match(tmp_path):
    payload = _payload(tmp_path)
    target_dir = tmp_path / "real-notice"
    target_dir.mkdir()
    content = b"coauthor meeting approval notice\n"
    (target_dir / "source.txt").write_bytes(content)
    (tmp_path / "notice-link").symlink_to(target_dir, target_is_directory=True)
    payload["source_notice"]["file"] = {
        "path": "notice-link/source.txt",
        "sha256": _sha256(content),
    }

    with pytest.raises(RevisionApprovalError, match="must not contain symlinks"):
        _validate(tmp_path, payload)


def test_manifest_symlink_is_rejected(tmp_path):
    path, digest = _write_manifest(tmp_path, _payload(tmp_path))
    link = tmp_path / "approval-link.json"
    link.symlink_to(path)

    with pytest.raises(RevisionApprovalError, match="must not contain symlinks"):
        validate_revision_approval(link, digest, RELEASE_STAGE, now=NOW)


def test_manifest_parent_symlink_is_rejected(tmp_path):
    real_dir = tmp_path / "real-approval"
    real_dir.mkdir()
    path, digest = _write_manifest(real_dir, _payload(real_dir))
    link_dir = tmp_path / "approval-link"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(RevisionApprovalError, match="must not contain symlinks"):
        validate_revision_approval(
            link_dir / path.name,
            digest,
            RELEASE_STAGE,
            now=NOW,
        )

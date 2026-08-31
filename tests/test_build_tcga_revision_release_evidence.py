"""Adversarial tests for the result-blind release-evidence closure."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from analysis import build_tcga_revision_artifact_registry as registry
from analysis import build_tcga_revision_release_evidence as closure


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii") + b"\n"


def _write(root: Path, member: str, raw: bytes) -> Path:
    path = root / member
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


@dataclass(frozen=True)
class _Fixture:
    registry_path: Path
    registry_sha256: str
    renderer_root: Path
    rendered_output_root: Path
    gate_root: Path
    source_root: Path
    primary_source: Path
    calibration_source: Path


def _make_fixture(
    tmp_path: Path,
    *,
    source_size_delta: int = 0,
    extra_source_role: bool = False,
    duplicate_receipt_id: bool = False,
    traversal_receipt_id: bool = False,
) -> _Fixture:
    renderer_root = tmp_path / "renderer"
    output_root = tmp_path / "rendered-output"
    gate_root = tmp_path / "gate-receipts"
    source_root = tmp_path / "source-data"
    renderer_root.mkdir()
    output_root.mkdir()
    gate_root.mkdir()
    source_root.mkdir()

    registry_builder = Path(registry.__file__).read_bytes()
    _write(
        renderer_root,
        "analysis/build_tcga_revision_artifact_registry.py",
        registry_builder,
    )
    renderer_raw = b'print("opaque renderer fixture")\n'
    renderer_member = "analysis/render_interaction_summary.py"
    _write(renderer_root, renderer_member, renderer_raw)
    output_raw = b"% result-blind rendered table fixture\n"
    output_member = "rendered/interaction_summary/table.tex"
    _write(output_root, output_member, output_raw)

    gate_raw = {
        "K500": b"\xffopaque-k500-receipt\x00",
        "CAL": b"\xfeopaque-calibration-receipt\x00",
        "COAUTH": b"\xfdopaque-coauthor-receipt\x00",
    }
    receipt_ids = {
        "K500": "k500-receipt",
        "CAL": "cal-receipt",
        "COAUTH": "coauthor-receipt",
    }
    if duplicate_receipt_id:
        receipt_ids["CAL"] = receipt_ids["K500"]
        gate_raw["CAL"] = gate_raw["K500"]
    if traversal_receipt_id:
        receipt_ids["K500"] = "receipts/../k500-receipt"

    for gate, receipt_id in receipt_ids.items():
        if ".." not in receipt_id:
            _write(gate_root, receipt_id, gate_raw[gate])

    primary_raw = b"\xff,\x00,opaque-source-not-parsed\n"
    calibration_raw = b"\xfeopaque-calibration-source\x00"
    primary = _write(source_root, "source-data/primary.csv", primary_raw)
    calibration = _write(
        source_root,
        "source-data/calibration.bin",
        calibration_raw,
    )
    source_records: list[dict[str, object]] = [
        {
            "source_id": "primary-family",
            "release_member": "source-data/primary.csv",
            "role": "primary",
            "sha256": _sha256(primary_raw),
            "bytes": len(primary_raw) + source_size_delta,
        },
        {
            "source_id": "calibration-evidence",
            "release_member": "source-data/calibration.bin",
            "role": "calibration",
            "sha256": _sha256(calibration_raw),
            "bytes": len(calibration_raw),
        },
    ]
    if extra_source_role:
        runtime_raw = b"opaque-runtime-source"
        _write(source_root, "source-data/runtime.bin", runtime_raw)
        source_records.append(
            {
                "source_id": "runtime-evidence",
                "release_member": "source-data/runtime.bin",
                "role": "runtime",
                "sha256": _sha256(runtime_raw),
                "bytes": len(runtime_raw),
            },
        )

    ledger = [
        {
            "gate": gate,
            "receipt_id": receipt_ids[gate],
            "sha256": _sha256(gate_raw[gate]),
        }
        for gate in ("K500", "CAL", "COAUTH")
    ]
    artifacts: list[dict[str, object]] = []
    ledger_by_gate = {record["gate"]: record for record in ledger}
    for spec in registry.ARTIFACT_SPECS:
        receipts = [
            dict(ledger_by_gate[gate])
            for gate in spec.required_gates
            if gate in ledger_by_gate
        ]
        if spec.semantic_id == "interaction_summary":
            artifacts.append(
                {
                    "semantic_id": spec.semantic_id,
                    "status": "ready",
                    "gate_receipts": receipts,
                    "source_data": source_records,
                    "renderer": {
                        "script": renderer_member,
                        "sha256": _sha256(renderer_raw),
                    },
                    "outputs": [
                        {
                            "output_id": "interaction-summary-tex",
                            "release_member": output_member,
                            "media_type": "application/x-tex",
                            "sha256": _sha256(output_raw),
                            "bytes": len(output_raw),
                        },
                    ],
                },
            )
            continue
        unsatisfied = [
            gate
            for gate in registry.GATE_ORDER
            if gate in spec.required_gates and gate not in ledger_by_gate
        ]
        reason_code = (
            "required_gate_not_satisfied"
            if unsatisfied
            else "coauthor_decision_to_omit"
        )
        artifacts.append(
            {
                "semantic_id": spec.semantic_id,
                "status": "omitted",
                "gate_receipts": receipts,
                "omission": {
                    "reason_code": reason_code,
                    "unsatisfied_gates": unsatisfied,
                },
            },
        )

    reconciliation = {
        "schema": registry.RECONCILIATION_INPUT_SCHEMA,
        "release": {
            "release_id": "tcga-k500-revision-release",
            "postprocess_release_sha256": "1" * 64,
            "source_data_manifest_sha256": "2" * 64,
        },
        "gate_ledger": ledger,
        "artifacts": artifacts,
    }
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_raw = _canonical(reconciliation)
    reconciliation_path.write_bytes(reconciliation_raw)
    registry_path = tmp_path / "artifact_registry.json"
    receipt = registry.build_artifact_registry(
        reconciliation_path,
        renderer_root,
        output_root,
        registry_path,
        expected_reconciliation_sha256=_sha256(reconciliation_raw),
    )
    return _Fixture(
        registry_path=registry_path,
        registry_sha256=receipt.manifest_sha256,
        renderer_root=renderer_root,
        rendered_output_root=output_root,
        gate_root=gate_root,
        source_root=source_root,
        primary_source=primary,
        calibration_source=calibration,
    )


def _build(fixture: _Fixture, destination: Path) -> closure.ReleaseEvidenceReceipt:
    return closure.build_release_evidence_closure(
        fixture.registry_path,
        fixture.renderer_root,
        fixture.rendered_output_root,
        fixture.gate_root,
        fixture.source_root,
        destination,
        expected_artifact_registry_sha256=fixture.registry_sha256,
    )


def test_build_and_validate_close_every_declared_opaque_byte(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "release_evidence.json"
    receipt = _build(fixture, destination)

    raw = destination.read_bytes()
    value = json.loads(raw)
    assert raw == _canonical(value)
    assert receipt.manifest_sha256 == _sha256(raw)
    assert receipt.gate_receipt_count == 3
    assert receipt.source_member_count == 2
    assert receipt.ready_count == 1
    assert receipt.omitted_count == len(registry.ARTIFACT_SPECS) - 1
    assert value["trust_model"] == closure.TRUST_MODEL
    assert value["artifact_registry"]["sha256"] == fixture.registry_sha256
    assert [record["member"] for record in value["gate_receipts"]] == [
        "k500-receipt",
        "cal-receipt",
        "coauthor-receipt",
    ]
    registry_value = json.loads(fixture.registry_path.read_bytes())
    expected_gate_bindings = {
        gate: sorted(
            artifact["semantic_id"]
            for artifact in registry_value["artifacts"]
            if any(
                receipt["gate"] == gate
                for receipt in artifact["gate_receipts"]
            )
        )
        for gate in ("K500", "CAL", "COAUTH")
    }
    assert {
        record["gate"]: record["artifacts"]
        for record in value["gate_receipts"]
    } == expected_gate_bindings
    registry_artifacts = {
        artifact["semantic_id"]: artifact
        for artifact in registry_value["artifacts"]
    }
    for artifact in value["artifacts"]:
        declared = registry_artifacts[artifact["semantic_id"]]
        assert artifact["required_gates"] == declared["required_gates"]
        assert artifact["satisfied_gates"] == [
            receipt["gate"] for receipt in declared["gate_receipts"]
        ]
        assert artifact["receipt_ids"] == [
            receipt["receipt_id"] for receipt in declared["gate_receipts"]
        ]
        assert artifact["required_source_roles"] == declared["required_source_roles"]
    assert {
        (binding["semantic_id"], binding["role"])
        for source in value["source_data"]
        for binding in source["bindings"]
    } == {
        ("interaction_summary", "primary"),
        ("interaction_summary", "calibration"),
    }
    payload = dict(value)
    declared = payload.pop("closure_payload_sha256")
    assert declared == _sha256(_canonical(payload)[:-1])

    validated = closure.validate_release_evidence_closure(
        destination,
        fixture.registry_path,
        fixture.renderer_root,
        fixture.rendered_output_root,
        fixture.gate_root,
        fixture.source_root,
        expected_closure_sha256=receipt.manifest_sha256,
        expected_artifact_registry_sha256=fixture.registry_sha256,
    )
    assert validated == receipt


def test_registry_is_validated_before_evidence_roots_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    observed: list[str] = []
    native = closure.artifact_registry.validate_artifact_registry
    pin_root = closure._pin_root  # noqa: SLF001

    def validate(*args, **kwargs):
        observed.append("registry")
        return native(*args, **kwargs)

    def pin(path, *, context):
        observed.append(context)
        return pin_root(path, context=context)

    monkeypatch.setattr(
        closure.artifact_registry,
        "validate_artifact_registry",
        validate,
    )
    monkeypatch.setattr(closure, "_pin_root", pin)
    _build(fixture, tmp_path / "closure.json")
    assert observed[0] == "registry"
    assert observed.index("registry") < observed.index("gate-receipt root")
    assert observed.index("registry") < observed.index("source-data root")


def test_bad_registry_anchor_fails_before_missing_evidence_roots(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    with pytest.raises(closure.ReleaseEvidenceError, match="registry validation"):
        closure.build_release_evidence_closure(
            fixture.registry_path,
            fixture.renderer_root,
            fixture.rendered_output_root,
            tmp_path / "does-not-exist-gates",
            tmp_path / "does-not-exist-sources",
            tmp_path / "closure.json",
            expected_artifact_registry_sha256="0" * 64,
        )


def test_bad_closure_anchor_fails_before_evidence_roots_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    _build(fixture, destination)
    root_opened = False

    def reject_root_open(*_args, **_kwargs):
        nonlocal root_opened
        root_opened = True
        raise AssertionError

    monkeypatch.setattr(closure, "_pin_root", reject_root_open)
    with pytest.raises(closure.ReleaseEvidenceError, match="independent anchor"):
        closure.validate_release_evidence_closure(
            destination,
            fixture.registry_path,
            fixture.renderer_root,
            fixture.rendered_output_root,
            fixture.gate_root,
            fixture.source_root,
            expected_closure_sha256="0" * 64,
            expected_artifact_registry_sha256=fixture.registry_sha256,
        )
    assert not root_opened


@pytest.mark.parametrize("root_name", ["gate", "source"])
def test_exact_inventory_rejects_extraneous_member(
    tmp_path: Path,
    root_name: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    root = fixture.gate_root if root_name == "gate" else fixture.source_root
    _write(root, "unexpected.bin", b"not declared")
    with pytest.raises(closure.ReleaseEvidenceError, match="inventory mismatch"):
        _build(fixture, tmp_path / "closure.json")


def test_gate_receipt_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.gate_root / "cal-receipt").write_bytes(b"changed")
    with pytest.raises(closure.ReleaseEvidenceError, match="SHA-256 mismatch"):
        _build(fixture, tmp_path / "closure.json")


def test_source_digest_mismatch_is_rejected_without_parsing_rows(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.primary_source.write_bytes(b"changed opaque bytes")
    with pytest.raises(closure.ReleaseEvidenceError, match="SHA-256 mismatch"):
        _build(fixture, tmp_path / "closure.json")


def test_source_declared_byte_count_is_enforced(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, source_size_delta=1)
    with pytest.raises(closure.ReleaseEvidenceError, match="byte-count mismatch"):
        _build(fixture, tmp_path / "closure.json")


def test_ready_artifact_cannot_smuggle_an_extra_source_role(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, extra_source_role=True)
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match="source roles are not exact",
    ):
        _build(fixture, tmp_path / "closure.json")


def test_two_gates_cannot_reuse_one_receipt_identity(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, duplicate_receipt_id=True)
    with pytest.raises(closure.ReleaseEvidenceError, match="one unique receipt member"):
        _build(fixture, tmp_path / "closure.json")


def test_gate_receipt_path_traversal_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, traversal_receipt_id=True)
    with pytest.raises(closure.ReleaseEvidenceError, match="escapes"):
        _build(fixture, tmp_path / "closure.json")


def test_symlinked_source_member_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(fixture.primary_source.read_bytes())
    fixture.primary_source.unlink()
    fixture.primary_source.symlink_to(external)
    with pytest.raises(closure.ReleaseEvidenceError, match="symlink"):
        _build(fixture, tmp_path / "closure.json")


def test_hardlinked_source_member_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    os.link(fixture.primary_source, tmp_path / "outside-hardlink.bin")
    with pytest.raises(closure.ReleaseEvidenceError, match="single-link"):
        _build(fixture, tmp_path / "closure.json")


def test_existing_destination_is_preserved(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    destination.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        _build(fixture, destination)
    assert destination.read_bytes() == b"preserve me"


def test_source_mutation_at_link_boundary_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    native = closure._publish_no_replace  # noqa: SLF001

    def mutate_then_publish(destination, parent_fd, raw, *, boundary_check):
        fixture.primary_source.write_bytes(b"link-boundary mutation")
        return native(
            destination,
            parent_fd,
            raw,
            boundary_check=boundary_check,
        )

    monkeypatch.setattr(closure, "_publish_no_replace", mutate_then_publish)
    with pytest.raises(closure.ReleaseEvidenceError, match="changed"):
        _build(fixture, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".closure.json.staging-*"))


def test_source_mutation_immediately_after_link_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    native_link = os.link

    def link_then_mutate(*args, **kwargs):
        result = native_link(*args, **kwargs)
        fixture.primary_source.write_bytes(b"post-link source mutation")
        return result

    monkeypatch.setattr(closure.os, "link", link_then_mutate)
    with pytest.raises(closure.ReleaseEvidenceError, match="changed"):
        _build(fixture, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".closure.json.staging-*"))


def test_post_link_readback_failure_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    native_open = os.open

    def fail_published_open(path, flags, *args, **kwargs):
        if path == destination.name and kwargs.get("dir_fd") is not None:
            raise OSError
        return native_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(closure.os, "open", fail_published_open)
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match="published closure destination cannot be pinned",
    ):
        _build(fixture, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".closure.json.staging-*"))


def test_destination_parent_replacement_after_link_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    publish_root = tmp_path / "publish"
    moved_root = tmp_path / "publish-moved"
    publish_root.mkdir()
    destination = publish_root / "closure.json"
    native_link = os.link

    def link_then_replace_parent(*args, **kwargs):
        result = native_link(*args, **kwargs)
        publish_root.rename(moved_root)
        publish_root.mkdir()
        return result

    monkeypatch.setattr(closure.os, "link", link_then_replace_parent)
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match="destination parent changed",
    ):
        _build(fixture, destination)
    assert not destination.exists()
    assert not (moved_root / "closure.json").exists()
    assert not list(moved_root.glob(".closure.json.staging-*"))


def test_destination_parent_replacement_before_link_rolls_back_staging(
    tmp_path: Path,
) -> None:
    publish_root = tmp_path / "publish"
    moved_root = tmp_path / "publish-moved"
    publish_root.mkdir()
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        publish_root / "closure.json",
        registry_path=tmp_path / "registry.json",
    )

    def replace_parent() -> None:
        publish_root.rename(moved_root)
        publish_root.mkdir()

    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="destination parent changed",
        ):
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=replace_parent,
            )
    finally:
        os.close(parent_fd)
    assert not destination.exists()
    assert not (moved_root / "closure.json").exists()
    assert not list(moved_root.glob(".closure.json.staging-*"))


def test_destination_parent_replacement_during_final_digest_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_root = tmp_path / "publish"
    moved_root = tmp_path / "publish-moved"
    publish_root.mkdir()
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        publish_root / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    digest_calls = 0
    native_digest = closure._digest_descriptor  # noqa: SLF001

    def digest_then_replace_parent(descriptor: int):
        nonlocal digest_calls
        result = native_digest(descriptor)
        digest_calls += 1
        if digest_calls == 4:
            publish_root.rename(moved_root)
            publish_root.mkdir()
        return result

    monkeypatch.setattr(closure, "_digest_descriptor", digest_then_replace_parent)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="destination parent changed",
        ):
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    assert digest_calls == 4
    assert not destination.exists()
    assert not (moved_root / "closure.json").exists()
    assert not list(moved_root.glob(".closure.json.staging-*"))


def test_same_inode_corruption_after_staging_unlink_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    native_unlink = os.unlink
    native_open = os.open

    def unlink_then_corrupt(path, *args, **kwargs):
        result = native_unlink(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".closure.json.staging-"):
            parent_fd = kwargs["dir_fd"]
            os.chmod(destination.name, 0o600, dir_fd=parent_fd)
            descriptor = native_open(destination.name, os.O_WRONLY, dir_fd=parent_fd)
            try:
                size = os.fstat(descriptor).st_size
                os.write(descriptor, b"X" * size)
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
        return result

    monkeypatch.setattr(closure.os, "unlink", unlink_then_corrupt)
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match="does not match the staged file",
    ):
        _build(fixture, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".closure.json.staging-*"))


def test_validator_rejects_canonical_tampering_even_with_new_file_anchor(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    _build(fixture, destination)
    value = json.loads(destination.read_bytes())
    value["release"]["release_id"] = "tampered-release"
    payload = dict(value)
    payload.pop("closure_payload_sha256")
    value["closure_payload_sha256"] = _sha256(_canonical(payload)[:-1])
    tampered = _canonical(value)
    destination.chmod(0o600)
    destination.write_bytes(tampered)
    destination.chmod(0o400)

    with pytest.raises(closure.ReleaseEvidenceError, match="does not match live"):
        closure.validate_release_evidence_closure(
            destination,
            fixture.registry_path,
            fixture.renderer_root,
            fixture.rendered_output_root,
            fixture.gate_root,
            fixture.source_root,
            expected_closure_sha256=_sha256(tampered),
            expected_artifact_registry_sha256=fixture.registry_sha256,
        )


def test_cli_help_exposes_build_and_validate_commands() -> None:
    parser = closure._parser()  # noqa: SLF001
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["build", "--help"])
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["validate", "--help"])

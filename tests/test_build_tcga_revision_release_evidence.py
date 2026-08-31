"""Adversarial tests for the result-blind release-evidence closure."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pytest

from analysis import build_tcga_revision_artifact_registry as registry
from analysis import build_tcga_revision_release_evidence as closure


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


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
            if any(receipt["gate"] == gate for receipt in artifact["gate_receipts"])
        )
        for gate in ("K500", "CAL", "COAUTH")
    }
    assert {
        record["gate"]: record["artifacts"] for record in value["gate_receipts"]
    } == expected_gate_bindings
    registry_artifacts = {
        artifact["semantic_id"]: artifact for artifact in registry_value["artifacts"]
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
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match=r"expected entry limit|unexpected entry",
    ):
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
    fixture.primary_source.write_bytes(b"\x00" * fixture.primary_source.stat().st_size)
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


def test_absolute_metadata_fifo_is_rejected_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = tmp_path / "metadata.json"
    os.mkfifo(metadata)
    native_open = closure.os.open

    def guarded_open(path: str | Path, *args: object, **kwargs: object) -> int:
        if path == metadata:
            message = "FIFO must be rejected by metadata preflight"
            raise AssertionError(message)
        return native_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(closure.os, "open", guarded_open)
    with pytest.raises(closure.ReleaseEvidenceError, match="regular file"):
        closure._pin_absolute_file(metadata, context="synthetic metadata")  # noqa: SLF001


def test_absolute_metadata_open_swap_to_fifo_is_nonblocking_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = tmp_path / "metadata.json"
    backup = tmp_path / "metadata.before-fifo.json"
    metadata.write_bytes(b"{}\n")
    native_open = closure.os.open
    native_fstat = closure.os.fstat
    opened: list[int] = []
    swapped = False

    def swapping_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == metadata and not swapped:
            assert flags & os.O_NONBLOCK
            assert flags & os.O_CLOEXEC
            metadata.rename(backup)
            os.mkfifo(metadata)
            swapped = True
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == metadata:
            opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(closure.os, "open", swapping_open)
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match="changed while it was pinned",
    ):
        closure._pin_absolute_file(metadata, context="synthetic metadata")  # noqa: SLF001
    assert swapped
    assert backup.read_bytes() == b"{}\n"
    assert metadata.is_fifo()
    assert len(opened) == 1
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        native_fstat(opened[0])


def test_inventory_to_member_open_fifo_swap_is_nonblocking_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    raw = b"opaque evidence\n"
    member = "member.bin"
    member_path = _write(evidence_root, member, raw)
    backup = evidence_root / "member.before-fifo.bin"
    root = closure._pin_root(evidence_root, context="evidence root")  # noqa: SLF001
    closure._require_exact_inventory(  # noqa: SLF001
        root,
        (member,),
        context="evidence root",
    )
    native_open = closure.os.open
    native_fstat = closure.os.fstat
    opened: list[int] = []
    swapped = False

    def swapping_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == member and dir_fd is not None and not swapped:
            assert flags & os.O_NONBLOCK
            assert flags & os.O_CLOEXEC
            member_path.rename(backup)
            os.mkfifo(member_path)
            swapped = True
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == member and dir_fd is not None:
            opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(closure.os, "open", swapping_open)
    try:
        with pytest.raises(closure.ReleaseEvidenceError, match="single-link regular"):
            closure._pin_member(  # noqa: SLF001
                root,
                member,
                context="evidence member",
                expected_sha256=_sha256(raw),
                expected_size=len(raw),
            )
    finally:
        root.close()
    assert swapped
    assert backup.read_bytes() == raw
    assert member_path.is_fifo()
    assert len(opened) == 1
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        native_fstat(opened[0])


def test_oversized_metadata_is_rejected_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = tmp_path / "oversized.json"
    metadata.write_bytes(b"12345")
    monkeypatch.setattr(closure, "_MAX_METADATA_BYTES", 4)

    def forbidden_open(*_args: object, **_kwargs: object) -> int:
        message = "oversized metadata must fail before open"
        raise AssertionError(message)

    monkeypatch.setattr(closure.os, "open", forbidden_open)
    with pytest.raises(closure.ReleaseEvidenceError, match="metadata size limit"):
        closure._pin_absolute_file(metadata, context="synthetic metadata")  # noqa: SLF001


def test_oversized_member_is_rejected_after_bounded_open_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    member = "oversized.bin"
    raw = b"12345"
    _write(evidence_root, member, raw)
    root = closure._pin_root(evidence_root, context="evidence root")  # noqa: SLF001
    monkeypatch.setattr(closure, "_MAX_EVIDENCE_MEMBER_BYTES", 4)
    native_open = closure.os.open
    native_fstat = closure.os.fstat
    opened: list[int] = []

    def tracking_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == member and dir_fd is not None:
            assert flags & os.O_NONBLOCK
            opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(closure.os, "open", tracking_open)
    try:
        with pytest.raises(closure.ReleaseEvidenceError, match="evidence size limit"):
            closure._pin_member(  # noqa: SLF001
                root,
                member,
                context="evidence member",
                expected_sha256=_sha256(raw),
                expected_size=None,
            )
    finally:
        root.close()
    assert len(opened) == 1
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        native_fstat(opened[0])


def test_digest_growth_reads_only_declared_maximum_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "growing.bin"
    raw = b"abc"
    evidence.write_bytes(raw)
    descriptor = os.open(evidence, os.O_RDONLY)
    native_read = closure.os.read
    requested: list[int] = []
    grew = False

    def growing_read(candidate: int, size: int) -> bytes:
        nonlocal grew
        if candidate == descriptor:
            requested.append(size)
            if not grew:
                with evidence.open("ab") as stream:
                    stream.write(b"d")
                grew = True
        return native_read(candidate, size)

    monkeypatch.setattr(closure.os, "read", growing_read)
    try:
        with pytest.raises(closure.ReleaseEvidenceError, match="read bound"):
            closure._digest_descriptor(  # noqa: SLF001
                descriptor,
                maximum=len(raw),
                context="growing evidence",
            )
    finally:
        os.close(descriptor)
    assert requested == [len(raw) + 1]


def test_expected_inventory_rejects_member_count_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(closure, "_MAX_INVENTORY_MEMBERS", 2)
    with pytest.raises(closure.ReleaseEvidenceError, match="2-member limit"):
        closure._prepare_expected_inventory(  # noqa: SLF001
            ("first.bin", "second.bin", "third.bin"),
            context="synthetic inventory",
        )


def test_expected_inventory_rejects_component_depth_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(closure, "_MAX_INVENTORY_DEPTH", 2)
    with pytest.raises(closure.ReleaseEvidenceError, match="2-component depth limit"):
        closure._prepare_expected_inventory(  # noqa: SLF001
            ("one/two/three.bin",),
            context="synthetic inventory",
        )


def test_many_sibling_inventory_directories_use_bounded_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    members = tuple(f"directory-{index:03d}/member.bin" for index in range(128))
    for member in members:
        _write(evidence_root, member, b"opaque\n")
    root = closure._pin_root(evidence_root, context="evidence root")  # noqa: SLF001
    native_dup = closure.os.dup
    native_open = closure.os.open
    native_close = closure.os.close
    live: set[int] = set()
    maximum_live = 0

    def track(descriptor: int) -> int:
        nonlocal maximum_live
        live.add(descriptor)
        maximum_live = max(maximum_live, len(live))
        return descriptor

    def tracking_dup(descriptor: int) -> int:
        return track(native_dup(descriptor))

    def tracking_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        return track(native_open(path, flags, mode, dir_fd=dir_fd))

    def tracking_close(descriptor: int) -> None:
        live.discard(descriptor)
        native_close(descriptor)

    monkeypatch.setattr(closure.os, "dup", tracking_dup)
    monkeypatch.setattr(closure.os, "open", tracking_open)
    monkeypatch.setattr(closure.os, "close", tracking_close)
    try:
        closure._require_exact_inventory(  # noqa: SLF001
            root,
            members,
            context="evidence root",
        )
        assert not live
        assert maximum_live <= 2
    finally:
        root.close()


def test_scandir_base_exception_closes_inventory_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InventoryInterrupted(BaseException):
        pass

    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    _write(evidence_root, "member.bin", b"opaque\n")
    root = closure._pin_root(evidence_root, context="evidence root")  # noqa: SLF001
    expected = closure._prepare_expected_inventory(  # noqa: SLF001
        ("member.bin",),
        context="evidence root",
    )
    native_dup = closure.os.dup
    native_fstat = closure.os.fstat
    native_scandir = closure.os.scandir
    opened: list[int] = []
    iterator_closed = False

    class InterruptingIterator:
        def __init__(self, descriptor: int) -> None:
            self._iterator = native_scandir(descriptor)

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            _exception_type: object,
            _exception: object,
            _traceback: object,
        ) -> None:
            nonlocal iterator_closed
            iterator_closed = True
            self._iterator.close()

        def __iter__(self) -> InterruptingIterator:
            return self

        def __next__(self) -> os.DirEntry[str]:
            raise InventoryInterrupted

    def tracking_dup(descriptor: int) -> int:
        duplicate = native_dup(descriptor)
        opened.append(duplicate)
        return duplicate

    monkeypatch.setattr(closure.os, "dup", tracking_dup)
    monkeypatch.setattr(closure.os, "scandir", InterruptingIterator)
    try:
        with pytest.raises(InventoryInterrupted):
            closure._enumerate_tree(  # noqa: SLF001
                root,
                expected=expected,
                context="evidence root",
            )
    finally:
        root.close()
    assert iterator_closed
    assert len(opened) == 1
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        native_fstat(opened[0])


def test_existing_destination_is_preserved(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    destination.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        _build(fixture, destination)
    assert destination.read_bytes() == b"preserve me"


def test_source_mutation_before_rename_retains_private_stage(
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
    with pytest.raises(closure.ReleaseEvidenceError, match="changed") as captured:
        _build(fixture, destination)
    assert not destination.exists()
    assert "candidate_path=" in str(captured.value)
    assert "private-stage-name-may-be-owned-or-replaced" in str(captured.value)
    assert len(list(tmp_path.glob(".closure.json.private-*"))) == 1


def test_pre_rename_failure_retains_one_stage_and_blocks_retry(
    tmp_path: Path,
) -> None:
    publish_root = tmp_path / "publish"
    publish_root.mkdir()
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        publish_root / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    raw = b'{"valid":true}\n'

    def fail_before_rename() -> None:
        message = "synthetic pre-rename failure"
        raise closure.ReleaseEvidenceError(message)

    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="synthetic pre-rename failure",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                raw,
                boundary_check=fail_before_rename,
            )
        message = str(captured.value)
        assert f"expected_sha256={_sha256(raw)}" in message
        assert f"expected_bytes={len(raw)}" in message
        assert "private-stage-name-may-be-owned-or-replaced" in message
        stages = list(publish_root.glob(".closure.json.private-*"))
        assert len(stages) == 1
        assert stages[0].stat().st_mode & 0o777 == 0o400
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="retained private closure stage requires explicit review",
        ):
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                raw,
                boundary_check=lambda: None,
            )
        assert list(publish_root.glob(".closure.json.private-*")) == stages
    finally:
        os.close(parent_fd)


def test_post_scan_stage_reservation_race_preserves_unowned_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    candidate = tmp_path / ".closure.json.private-candidate"
    competitor = b"competitor-stage\n"
    native_open = closure.os.open
    injected = False

    def racing_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal injected
        if path == candidate.name and not injected:
            injected = True
            descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
            try:
                closure.os.write(descriptor, competitor)
            finally:
                closure.os.close(descriptor)
            raise FileExistsError(
                closure.errno.EEXIST,
                closure.os.strerror(closure.errno.EEXIST),
                path,
            )
        return native_open(path, flags, mode, dir_fd=dir_fd)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(closure.os, "open", racing_open)
    monkeypatch.setattr(closure.os, "unlink", forbidden_unlink)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="appeared after the retained-stage preflight",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
        message = str(captured.value)
        assert f"candidate_path={candidate}" in message
        assert "expected_sha256=unknown" in message
        assert "expected_bytes=unknown" in message
        assert "private-stage-name-not-proven-owned" in message
        assert candidate.read_bytes() == competitor
        assert list(tmp_path.glob(".closure.json.private-*")) == [candidate]
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="retained private closure stage requires explicit review",
        ):
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
        assert candidate.read_bytes() == competitor
        assert list(tmp_path.glob(".closure.json.private-*")) == [candidate]
    finally:
        os.close(parent_fd)


def test_source_mutation_immediately_after_rename_retains_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    native_rename = closure._rename_no_replace  # noqa: SLF001

    def rename_then_mutate(source: str, target: str, parent_fd: int) -> None:
        native_rename(source, target, parent_fd)
        fixture.primary_source.write_bytes(b"post-link source mutation")

    monkeypatch.setattr(closure, "_rename_no_replace", rename_then_mutate)
    with pytest.raises(closure.ReleaseEvidenceError, match="changed") as captured:
        _build(fixture, destination)
    assert destination.exists()
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert not list(tmp_path.glob(".closure.json.private-*"))


def test_post_rename_readback_failure_retains_publication(
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
    ) as captured:
        _build(fixture, destination)
    assert destination.exists()
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert not list(tmp_path.glob(".closure.json.private-*"))


def test_published_destination_fifo_swap_is_nonblocking_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    owned_output = tmp_path / "owned-closure.json"
    raw = b'{"valid":true}\n'
    native_open = closure.os.open
    native_fstat = closure.os.fstat
    opened: list[int] = []
    swapped = False

    def swapping_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == destination.name and dir_fd is not None and not swapped:
            assert flags & os.O_NONBLOCK
            assert flags & os.O_CLOEXEC
            destination.rename(owned_output)
            os.mkfifo(destination)
            swapped = True
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == destination.name and dir_fd is not None:
            opened.append(descriptor)
        return descriptor

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(closure.os, "open", swapping_open)
    monkeypatch.setattr(closure.os, "unlink", forbidden_unlink)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="does not match the staged file",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                raw,
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    assert swapped
    assert destination.is_fifo()
    assert owned_output.read_bytes() == raw
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert not list(tmp_path.glob(".closure.json.private-*"))
    assert len(opened) == 1
    with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
        native_fstat(opened[0])


def test_published_destination_oversized_swap_is_rejected_before_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    owned_output = tmp_path / "owned-closure.json"
    raw = b'{"valid":true}\n'
    competitor = raw + b"oversized"
    native_open = closure.os.open
    native_digest = closure._digest_descriptor  # noqa: SLF001
    swapped = False

    def swapping_open(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == destination.name and dir_fd is not None and not swapped:
            destination.rename(owned_output)
            destination.write_bytes(competitor)
            swapped = True
        return native_open(path, flags, mode, dir_fd=dir_fd)

    def guarded_digest(
        descriptor: int,
        *,
        maximum: int,
        context: str,
    ) -> tuple[str, int, os.stat_result]:
        assert context != "published closure destination"
        return native_digest(descriptor, maximum=maximum, context=context)

    monkeypatch.setattr(closure.os, "open", swapping_open)
    monkeypatch.setattr(closure, "_digest_descriptor", guarded_digest)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="does not match the staged file",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                raw,
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    assert swapped
    assert destination.read_bytes() == competitor
    assert owned_output.read_bytes() == raw
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)


def test_published_destination_growth_uses_expected_size_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    raw = b'{"valid":true}\n'
    native_digest = closure._digest_descriptor  # noqa: SLF001
    published_maxima: list[int] = []
    grown = False

    def growing_digest(
        descriptor: int,
        *,
        maximum: int,
        context: str,
    ) -> tuple[str, int, os.stat_result]:
        nonlocal grown
        if context == "published closure destination":
            published_maxima.append(maximum)
            if not grown:
                grown = True
                destination.chmod(0o600)
                with destination.open("ab") as stream:
                    stream.write(b"x")
        return native_digest(descriptor, maximum=maximum, context=context)

    monkeypatch.setattr(closure, "_digest_descriptor", growing_digest)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="read bound",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                raw,
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    assert published_maxima == [len(raw)]
    assert destination.read_bytes() == raw + b"x"
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)


def test_destination_parent_replacement_after_rename_retains_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    publish_root = tmp_path / "publish"
    moved_root = tmp_path / "publish-moved"
    publish_root.mkdir()
    destination = publish_root / "closure.json"
    native_rename = closure._rename_no_replace  # noqa: SLF001

    def rename_then_replace_parent(source: str, target: str, parent_fd: int) -> None:
        native_rename(source, target, parent_fd)
        publish_root.rename(moved_root)
        publish_root.mkdir()

    monkeypatch.setattr(closure, "_rename_no_replace", rename_then_replace_parent)
    with pytest.raises(
        closure.ReleaseEvidenceError,
        match="destination parent changed",
    ) as captured:
        _build(fixture, destination)
    assert not destination.exists()
    assert (moved_root / "closure.json").exists()
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert not list(moved_root.glob(".closure.json.private-*"))


def test_destination_parent_replacement_before_rename_retains_staging(
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
        ) as captured:
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
    assert "private-stage-name-may-be-owned-or-replaced" in str(captured.value)
    assert len(list(moved_root.glob(".closure.json.private-*"))) == 1


def test_destination_parent_replacement_during_final_digest_retains_destination(
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

    def digest_then_replace_parent(
        descriptor: int,
        *,
        maximum: int,
        context: str,
    ) -> tuple[str, int, os.stat_result]:
        nonlocal digest_calls
        result = native_digest(descriptor, maximum=maximum, context=context)
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
        ) as captured:
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
    assert (moved_root / "closure.json").exists()
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert not list(moved_root.glob(".closure.json.private-*"))


def test_competitor_at_atomic_rename_survives_and_stage_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    competitor = b"competitor-owned\n"
    native_rename = closure._rename_no_replace  # noqa: SLF001

    def race(source: str, target: str, directory_fd: int) -> None:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, competitor)
        finally:
            os.close(descriptor)
        native_rename(source, target, directory_fd)

    monkeypatch.setattr(closure, "_rename_no_replace", race)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="destination may already exist",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    assert destination.read_bytes() == competitor
    message = str(captured.value)
    assert f"candidate_path={destination}" in message
    assert "alternate_candidate_path=" in message
    assert "destination-or-private-stage-names-may-be-owned-or-replaced" in message
    assert len(list(tmp_path.glob(".closure.json.private-*"))) == 1


def test_rename_then_raise_reports_both_candidates_without_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    native_rename = closure._rename_no_replace  # noqa: SLF001

    def rename_then_raise(source: str, target: str, directory_fd: int) -> None:
        native_rename(source, target, directory_fd)
        message = "synthetic ambiguous rename return"
        raise OSError(message)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(closure, "_rename_no_replace", rename_then_raise)
    monkeypatch.setattr(closure.os, "unlink", forbidden_unlink)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="synthetic ambiguous rename return",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    message = str(captured.value)
    assert f"candidate_path={destination}" in message
    assert "alternate_candidate_path=" in message
    assert ".closure.json.private-" in message
    assert "destination-or-private-stage-names-may-be-owned-or-replaced" in message
    assert destination.exists()
    assert not list(tmp_path.glob(".closure.json.private-*"))


def test_post_rename_failure_never_unlinks_and_leaks_no_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    destination = tmp_path / "closure.json"
    opened_descriptors: list[int] = []
    native_open = closure.os.open
    native_fstat = closure.os.fstat
    native_rename = closure._rename_no_replace  # noqa: SLF001

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = native_open(*args, **kwargs)  # type: ignore[arg-type]
        opened_descriptors.append(descriptor)
        return descriptor

    def rename_then_mutate(source: str, target: str, parent_fd: int) -> None:
        native_rename(source, target, parent_fd)
        fixture.primary_source.write_bytes(b"post-rename source mutation")

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(closure.os, "open", tracking_open)
    monkeypatch.setattr(closure, "_rename_no_replace", rename_then_mutate)
    monkeypatch.setattr(closure.os, "unlink", forbidden_unlink)
    with pytest.raises(closure.ReleaseEvidenceError, match="changed") as captured:
        _build(fixture, destination)
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert destination.exists()
    assert opened_descriptors
    for descriptor in opened_descriptors:
        with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
            native_fstat(descriptor)


def test_missing_atomic_rename_symbol_retains_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )

    class MissingRenameLibrary:
        pass

    monkeypatch.setattr(
        closure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: MissingRenameLibrary(),
    )
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="rename symbol",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    message = str(captured.value)
    assert f"candidate_path={tmp_path / '.closure.json.private-candidate'}" in message
    assert "alternate_candidate_path=" not in message
    assert "private-stage-name-may-be-owned-or-replaced" in message
    assert len(list(tmp_path.glob(".closure.json.private-*"))) == 1


def test_unsupported_atomic_rename_error_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )

    class UnsupportedRename:
        def __call__(self, *_args: object) -> int:
            closure.ctypes.set_errno(closure.errno.ENOTSUP)
            return -1

    class UnsupportedRenameLibrary:
        renameatx_np = UnsupportedRename()
        renameat2 = UnsupportedRename()

    monkeypatch.setattr(
        closure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: UnsupportedRenameLibrary(),
    )
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="does not support atomic no-replace rename",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                b'{"valid":true}\n',
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    message = str(captured.value)
    assert f"candidate_path={destination}" in message
    assert "alternate_candidate_path=" in message
    assert "destination-or-private-stage-names-may-be-owned-or-replaced" in message
    assert len(list(tmp_path.glob(".closure.json.private-*"))) == 1


def test_post_syscall_eio_reports_both_candidates_and_retains_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination, parent_fd = closure._ensure_destination(  # noqa: SLF001
        tmp_path / "closure.json",
        registry_path=tmp_path / "registry.json",
    )
    candidate = tmp_path / ".closure.json.private-candidate"
    raw = b'{"valid":true}\n'

    class FailedRename:
        def __call__(self, *_args: object) -> int:
            closure.ctypes.set_errno(closure.errno.EIO)
            return -1

    class FailedRenameLibrary:
        renameatx_np = FailedRename()
        renameat2 = FailedRename()

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(
        closure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: FailedRenameLibrary(),
    )
    monkeypatch.setattr(closure.os, "unlink", forbidden_unlink)
    try:
        with pytest.raises(
            closure.ReleaseEvidenceError,
            match="publication failed",
        ) as captured:
            closure._publish_no_replace(  # noqa: SLF001
                destination,
                parent_fd,
                raw,
                boundary_check=lambda: None,
            )
    finally:
        os.close(parent_fd)
    message = str(captured.value)
    assert f"candidate_path={destination}" in message
    assert f"alternate_candidate_path={candidate}" in message
    assert "destination-or-private-stage-names-may-be-owned-or-replaced" in message
    assert not destination.exists()
    assert candidate.read_bytes() == raw


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


def test_destination_base_exception_closes_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "closure.json"
    native_open = closure.os.open
    native_stat = closure.os.stat
    native_fstat = closure.os.fstat
    opened: list[int] = []

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = native_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    def interrupting_stat(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        if kwargs.get("dir_fd") is not None and path == destination.name:
            raise KeyboardInterrupt
        return native_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(closure.os, "open", tracking_open)
    monkeypatch.setattr(closure.os, "stat", interrupting_stat)
    with pytest.raises(KeyboardInterrupt):
        closure._ensure_destination(  # noqa: SLF001
            destination,
            registry_path=tmp_path / "registry.json",
        )
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
            native_fstat(descriptor)

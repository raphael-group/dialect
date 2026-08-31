from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from analysis import build_tcga_revision_artifact_registry as artifact_registry
from analysis import build_tcga_revision_document_reconciliation as reconciliation

# Tests intentionally probe private, security-relevant seams.

_RELEASE_ID = "synthetic-k500-revision"
_PDF_SHA256 = "d" * 64
_STALE_LITERAL = "STALE-SCIENTIFIC-CLAIM"
_STALE_MARKER_ID = "stale-scientific-claim"
_POSTPROCESS_SHA256 = "1" * 64
_SOURCE_MANIFEST_SHA256 = "2" * 64


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical(value: object) -> bytes:
    return _json_bytes(value) + b"\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _array(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _receipt(gate: str) -> dict[str, str]:
    return {
        "gate": gate,
        "receipt_id": f"{gate.lower()}-synthetic-receipt",
        "sha256": _sha256(f"receipt:{gate}".encode()),
    }


def _marker(document_id: str, placement_id: str, boundary: str) -> str:
    marker = f"RECONCILIATION-TARGET:{placement_id}:{boundary}"
    if document_id == "rebuttal":
        return f"<!-- {marker} -->"
    return f"% {marker}"


def _placement(case: SyntheticCase, placement_id: str) -> dict[str, object]:
    for raw in _array(case.input_value["placements"]):
        record = _mapping(raw)
        if record["placement_id"] == placement_id:
            return record
    message = f"missing placement {placement_id}"
    raise AssertionError(message)


def _artifact_placement_id(semantic_id: str) -> str:
    return f"artifact-{semantic_id}"


@dataclass(slots=True)
class SyntheticCase:
    """Hold one entirely synthetic reconciliation fixture."""

    root: Path
    renderer_root: Path
    rendered_output_root: Path
    registry_path: Path
    anchor_path: Path
    input_path: Path
    output_path: Path
    registry_value: dict[str, object]
    input_value: dict[str, object]
    contents: dict[str, str]
    prefixes: dict[str, str]

    def refresh_registry(self) -> None:
        """Rewrite the self-digested synthetic artifact registry."""
        self.registry_value.pop("registry_payload_sha256", None)
        self.registry_value["registry_payload_sha256"] = _sha256(
            _json_bytes(self.registry_value),
        )
        _write_json(self.registry_path, self.registry_value)

    def refresh_documents(self) -> None:
        """Render synthetic marked documents and refresh their anchor."""
        placements = [_mapping(item) for item in _array(self.input_value["placements"])]
        for document_id, _role in reconciliation.DOCUMENT_ORDER:
            parts = [self.prefixes[document_id]]
            for placement in placements:
                if placement["document_id"] != document_id:
                    continue
                placement_id = cast("str", placement["placement_id"])
                parts.extend(
                    (
                        f"{_marker(document_id, placement_id, 'BEGIN')}\n",
                        self.contents[placement_id],
                        f"{_marker(document_id, placement_id, 'END')}\n",
                    ),
                )
            member = cast(
                "str",
                {"main": "main.tex", "s1": "s1.tex", "rebuttal": "rebuttal.md"}[
                    document_id
                ],
            )
            (self.root / member).write_text("".join(parts), encoding="utf-8")
        self.reanchor_documents()

    def reanchor_documents(self) -> None:
        """Rewrite the exact three-member document anchor."""
        members = {"main": "main.tex", "s1": "s1.tex", "rebuttal": "rebuttal.md"}
        documents: list[dict[str, object]] = []
        for document_id, role in reconciliation.DOCUMENT_ORDER:
            member = members[document_id]
            raw = (self.root / member).read_bytes()
            documents.append(
                {
                    "document_id": document_id,
                    "role": role,
                    "member": member,
                    "bytes": len(raw),
                    "sha256": _sha256(raw),
                },
            )
        _write_json(
            self.anchor_path,
            {
                "schema": reconciliation.DOCUMENT_ANCHOR_SCHEMA,
                "documents": documents,
            },
        )

    def refresh_input(self) -> None:
        """Bind and rewrite the canonical reconciliation input."""
        binding = _mapping(self.input_value["binding"])
        binding["artifact_registry_sha256"] = _sha256(
            self.registry_path.read_bytes(),
        )
        binding["document_anchor_sha256"] = _sha256(self.anchor_path.read_bytes())
        _write_json(self.input_path, self.input_value)

    def write_all(self) -> None:
        """Write every synthetic input in dependency order."""
        self.refresh_registry()
        self.refresh_documents()
        self.refresh_input()


def _registry_artifact(
    spec: artifact_registry.ArtifactSpec,
    *,
    omitted: bool,
) -> dict[str, object]:
    base: dict[str, object] = {
        "semantic_id": spec.semantic_id,
        "title": spec.title,
        "kind": spec.kind,
        "required_gates": list(spec.required_gates),
        "required_source_roles": list(spec.required_source_roles),
        "source_requirement": spec.source_requirement,
        "status": "omitted" if omitted else "ready",
        "gate_receipts": [_receipt(gate) for gate in spec.required_gates],
    }
    claims = [claim.as_record() for claim in spec.claims]
    if omitted:
        base.update(
            {
                "omission": {
                    "reason_code": "coauthor_decision_to_omit",
                    "reason": (
                        "All required gates have declared receipts and the "
                        "coauthors elected to omit this artifact from the release."
                    ),
                    "unsatisfied_gates": [],
                },
                "planned_claims": claims,
            },
        )
    else:
        if spec.source_requirement == "none":
            sources: list[dict[str, object]] = []
        elif spec.source_requirement == "upstream-manifest":
            sources = [
                {
                    "source_id": "postprocess-release-manifest",
                    "release_member": "postprocess/release_manifest.json",
                    "role": "provenance",
                    "sha256": _POSTPROCESS_SHA256,
                    "bytes": 123,
                },
                {
                    "source_id": "source-data-manifest",
                    "release_member": "source-data/source_data_manifest.json",
                    "role": "provenance",
                    "sha256": _SOURCE_MANIFEST_SHA256,
                    "bytes": 456,
                },
            ]
        else:
            sources = [
                {
                    "source_id": f"{spec.semantic_id}-{role}",
                    "release_member": (
                        f"source-data/never-open/{spec.semantic_id}.{role}.csv"
                    ),
                    "role": role,
                    "sha256": _sha256(
                        f"source:{spec.semantic_id}:{role}".encode(),
                    ),
                    "bytes": 123,
                }
                for role in spec.required_source_roles
            ]
        sources.sort(key=lambda record: cast("str", record["source_id"]))
        media_type, suffix = {
            "figure": ("application/pdf", ".pdf"),
            "table": ("application/pdf", ".pdf"),
            "supplementary-data": ("text/csv", ".csv"),
            "provenance-record": ("application/json", ".json"),
        }[spec.kind]
        # These deliberately nonexistent row-bearing members prove this verifier
        # treats registry source-data metadata as opaque and never opens them.
        base.update(
            {
                "source_data": sources,
                "renderer": {
                    "script": f"analysis/render_{spec.semantic_id}.py",
                    "sha256": "b" * 64,
                    "bytes": 789,
                },
                "outputs": [
                    {
                        "output_id": f"{spec.semantic_id}-opaque-output",
                        "release_member": (
                            f"rendered/{spec.semantic_id}/artifact{suffix}"
                        ),
                        "media_type": media_type,
                        "sha256": "c" * 64,
                        "bytes": 456,
                    },
                ],
                "claims": claims,
            },
        )
    return base


def _materialize_registry_artifact_bytes(
    registry_value: dict[str, object],
    *,
    renderer_root: Path,
    rendered_output_root: Path,
) -> None:
    for raw_artifact in _array(registry_value["artifacts"]):
        artifact = _mapping(raw_artifact)
        if artifact["status"] != "ready":
            continue
        semantic_id = cast("str", artifact["semantic_id"])
        renderer = _mapping(artifact["renderer"])
        renderer_member = cast("str", renderer["script"])
        renderer_raw = f"# synthetic renderer for {semantic_id}\n".encode()
        renderer_path = renderer_root / renderer_member
        renderer_path.parent.mkdir(parents=True, exist_ok=True)
        renderer_path.write_bytes(renderer_raw)
        renderer["sha256"] = _sha256(renderer_raw)
        renderer["bytes"] = len(renderer_raw)
        for raw_output in _array(artifact["outputs"]):
            output = _mapping(raw_output)
            output_member = cast("str", output["release_member"])
            output_raw = f"synthetic opaque output for {semantic_id}\n".encode()
            output_path = rendered_output_root / output_member
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(output_raw)
            output["sha256"] = _sha256(output_raw)
            output["bytes"] = len(output_raw)


def _make_case(
    tmp_path: Path,
    *,
    mode: str = "final",
    omitted_ids: frozenset[str] = frozenset(),
) -> SyntheticCase:
    document_root = tmp_path / "documents"
    document_root.mkdir()
    renderer_root = tmp_path / "renderers"
    rendered_output_root = tmp_path / "rendered-outputs"
    renderer_root.mkdir()
    rendered_output_root.mkdir()
    artifact_builder = Path(artifact_registry.__file__).read_bytes()
    artifact_builder_member = (
        renderer_root / "analysis/build_tcga_revision_artifact_registry.py"
    )
    artifact_builder_member.parent.mkdir(parents=True)
    artifact_builder_member.write_bytes(artifact_builder)
    registry_value: dict[str, object] = {
        "schema": artifact_registry.ARTIFACT_REGISTRY_SCHEMA,
        "contract": artifact_registry.ARTIFACT_REGISTRY_CONTRACT,
        "trust_model": artifact_registry.TRUST_MODEL,
        "release": {
            "release_id": _RELEASE_ID,
            "postprocess_release_sha256": _POSTPROCESS_SHA256,
            "source_data_manifest_sha256": _SOURCE_MANIFEST_SHA256,
        },
        "builder": {
            "script": "analysis/build_tcga_revision_artifact_registry.py",
            "sha256": _sha256(artifact_builder),
            "bytes": len(artifact_builder),
        },
        "gate_catalog": [dict(record) for record in artifact_registry.GATE_CATALOG],
        "gate_ledger": [_receipt(gate) for gate in artifact_registry.GATE_ORDER],
        "artifact_catalog_sha256": artifact_registry.artifact_catalog_sha256(),
        "artifacts": [
            _registry_artifact(spec, omitted=spec.semantic_id in omitted_ids)
            for spec in artifact_registry.ARTIFACT_SPECS
        ],
    }
    _materialize_registry_artifact_bytes(
        registry_value,
        renderer_root=renderer_root,
        rendered_output_root=rendered_output_root,
    )
    contents: dict[str, str] = {}
    placements: list[dict[str, object]] = []
    artifact_placement_ids: list[str] = []
    representation_gate = {
        "figure": "FIG",
        "table": "TABLE",
        "supplementary-data": "REL",
        "provenance-record": "REL",
    }
    placement_kind = {"provenance-record": "provenance"}
    for index, spec in enumerate(artifact_registry.ARTIFACT_SPECS):
        placement_id = _artifact_placement_id(spec.semantic_id)
        artifact_placement_ids.append(placement_id)
        content = f"Synthetic placement for {spec.semantic_id}.\n"
        contents[placement_id] = content
        omitted = spec.semantic_id in omitted_ids
        gate_set = {*spec.required_gates, representation_gate[spec.kind]}
        placements.append(
            {
                "placement_id": placement_id,
                "document_id": "main" if index % 2 == 0 else "s1",
                "kind": placement_kind.get(spec.kind, spec.kind),
                "status": "omitted" if omitted else "ready",
                "required_gates": [
                    gate
                    for gate in reconciliation.DOCUMENT_GATE_ORDER
                    if gate in gate_set
                ],
                "artifact_claims": [
                    {"artifact_id": spec.semantic_id, "claim_id": claim.claim_id}
                    for claim in spec.claims
                ],
                "content_sha256": _sha256(content.encode()),
                "page_location": {
                    "pdf_sha256": _PDF_SHA256,
                    "page": index + 1,
                    "line_start": 1,
                    "line_end": 2,
                },
                "omission": (
                    {
                        "reason_code": "coauthor_decision_to_omit",
                        "receipt_id": "coauthor-omission-receipt",
                        "sha256": _sha256(b"coauthor omission"),
                        "forbidden_token_ids": [_STALE_MARKER_ID],
                    }
                    if omitted
                    else None
                ),
            },
        )
    response_ids: list[str] = []
    for reviewer_item_id in reconciliation.REVIEWER_ITEM_ORDER:
        placement_id = f"response-{reviewer_item_id}"
        response_ids.append(placement_id)
        content = f"Synthetic response for {reviewer_item_id}.\n"
        contents[placement_id] = content
        placements.append(
            {
                "placement_id": placement_id,
                "document_id": "rebuttal",
                "kind": "response",
                "status": "ready",
                "required_gates": ["COAUTH"],
                "artifact_claims": [],
                "content_sha256": _sha256(content.encode()),
                "page_location": None,
                "omission": None,
            },
        )
    placements.sort(key=lambda record: cast("str", record["placement_id"]))
    reviewer_items: list[dict[str, object]] = []
    for index, (reviewer_item_id, response_id) in enumerate(
        zip(reconciliation.REVIEWER_ITEM_ORDER, response_ids, strict=True),
    ):
        reviewer_items.append(
            {
                "reviewer_item_id": reviewer_item_id,
                "response_placement_id": response_id,
                "target_placement_ids": (
                    sorted(artifact_placement_ids) if index == 0 else []
                ),
                "artifact_ids": (
                    [spec.semantic_id for spec in artifact_registry.ARTIFACT_SPECS]
                    if index == 0
                    else []
                ),
            },
        )
    input_value: dict[str, object] = {
        "schema": reconciliation.RECONCILIATION_INPUT_SCHEMA,
        "mode": mode,
        "binding": {
            "release_id": _RELEASE_ID,
            "artifact_registry_sha256": "0" * 64,
            "document_anchor_sha256": "0" * 64,
        },
        "gate_ledger": [_receipt(gate) for gate in reconciliation.DOCUMENT_GATE_ORDER],
        "reviewer_items": reviewer_items,
        "placements": placements,
        "forbidden_tokens": [
            {
                "token_id": _STALE_MARKER_ID,
                "literal": _STALE_LITERAL,
                "document_ids": list(reconciliation.DOCUMENT_IDS),
            },
        ],
    }
    case = SyntheticCase(
        root=document_root,
        renderer_root=renderer_root,
        rendered_output_root=rendered_output_root,
        registry_path=tmp_path / "artifact_registry.json",
        anchor_path=tmp_path / "document_anchor.json",
        input_path=tmp_path / "document_reconciliation_input.json",
        output_path=tmp_path / "document_reconciliation.json",
        registry_value=registry_value,
        input_value=input_value,
        contents=contents,
        prefixes={
            "main": "Synthetic manuscript.\n",
            "s1": "Synthetic supporting information.\n",
            "rebuttal": "Synthetic response to reviewers.\n",
        },
    )
    case.write_all()
    return case


def _build(
    case: SyntheticCase,
    *,
    destination: Path | None = None,
) -> reconciliation.DocumentReconciliationReceipt:
    return reconciliation.build_document_reconciliation(
        case.input_path,
        case.registry_path,
        case.renderer_root,
        case.rendered_output_root,
        case.anchor_path,
        case.root,
        destination or case.output_path,
        expected_reconciliation_sha256=_sha256(case.input_path.read_bytes()),
        expected_artifact_registry_sha256=_sha256(case.registry_path.read_bytes()),
        expected_document_anchor_sha256=_sha256(case.anchor_path.read_bytes()),
    )


def _validate(
    case: SyntheticCase,
    manifest: Path,
) -> reconciliation.DocumentReconciliationReceipt:
    return reconciliation.validate_document_reconciliation(
        manifest,
        case.registry_path,
        case.renderer_root,
        case.rendered_output_root,
        case.anchor_path,
        case.root,
        expected_manifest_sha256=_sha256(manifest.read_bytes()),
        expected_artifact_registry_sha256=_sha256(case.registry_path.read_bytes()),
        expected_document_anchor_sha256=_sha256(case.anchor_path.read_bytes()),
    )


def test_final_manifest_is_canonical_complete_read_only_and_validatable(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)

    receipt = _build(case)

    raw = case.output_path.read_bytes()
    manifest = _mapping(json.loads(raw))
    assert raw == _canonical(manifest)
    assert stat.S_IMODE(case.output_path.stat().st_mode) == 0o400
    assert receipt.manifest_sha256 == _sha256(raw)
    assert receipt.placement_count == len(artifact_registry.ARTIFACT_SPECS) + len(
        reconciliation.REVIEWER_ITEM_ORDER,
    )
    summary = _mapping(manifest["summary"])
    assert summary["artifact_count"] == 13
    assert summary["reviewer_item_count"] == 27
    assert summary["document_count"] == 3
    payload = copy.deepcopy(manifest)
    declared = payload.pop("manifest_payload_sha256")
    assert declared == _sha256(_json_bytes(payload))
    assert b"source-data/never-open" not in raw
    assert b"Synthetic placement" not in raw
    assert _validate(case, case.output_path) == receipt


def test_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    _build(case)
    before = case.output_path.read_bytes()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="destination already exists",
    ):
        _build(case)

    assert case.output_path.read_bytes() == before


def test_symlinked_destination_parent_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    publish_parent = tmp_path / "publish"
    publish_parent.mkdir()
    alias = tmp_path / "publish-alias"
    alias.symlink_to(publish_parent, target_is_directory=True)

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="canonical non-symlink directory",
    ):
        _build(case, destination=alias / "reconciliation.json")


def test_publish_race_preserves_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    competitor = b"competing destination\n"
    real_link = reconciliation.os.link

    def racing_link(source: object, destination: object, **kwargs: object) -> None:
        destination_parent = cast("int", kwargs["dst_dir_fd"])
        descriptor = os.open(
            destination,  # type: ignore[arg-type]
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_parent,
        )
        try:
            os.write(descriptor, competitor)
        finally:
            os.close(descriptor)
        real_link(source, destination, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reconciliation.os, "link", racing_link)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="destination already exists",
    ):
        _build(case)

    assert case.output_path.read_bytes() == competitor
    assert not list(tmp_path.glob(".document_reconciliation.json.private-*"))


def test_publish_swap_is_detected_without_deleting_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    competitor = b"swapped destination\n"
    real_link = reconciliation.os.link

    def swapping_link(source: object, destination: object, **kwargs: object) -> None:
        real_link(source, destination, **kwargs)  # type: ignore[arg-type]
        destination_parent = cast("int", kwargs["dst_dir_fd"])
        os.unlink(destination, dir_fd=destination_parent)  # type: ignore[arg-type]
        descriptor = os.open(
            destination,  # type: ignore[arg-type]
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_parent,
        )
        try:
            os.write(descriptor, competitor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(reconciliation.os, "link", swapping_link)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="identity does not match staging",
    ):
        _build(case)

    assert case.output_path.read_bytes() == competitor
    assert not list(tmp_path.glob(".document_reconciliation.json.private-*"))


def test_document_swap_at_publish_boundary_rolls_back_own_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    real_link = reconciliation.os.link

    def swapping_document(
        source: object,
        destination: object,
        **kwargs: object,
    ) -> None:
        real_link(source, destination, **kwargs)  # type: ignore[arg-type]
        main = case.root / "main.tex"
        original = main.read_bytes()
        main.rename(case.root / "main.before-swap.tex")
        main.write_bytes(original)

    monkeypatch.setattr(reconciliation.os, "link", swapping_document)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match=r"document (root identity|member 0 path) changed after validation",
    ):
        _build(case)

    assert not case.output_path.exists()
    assert not list(tmp_path.glob(".document_reconciliation.json.private-*"))


def test_destination_parent_replacement_is_detected_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    publish_parent = tmp_path / "publish"
    publish_parent.mkdir()
    destination = publish_parent / "reconciliation.json"
    moved_parent = tmp_path / "publish-moved"
    real_link = reconciliation.os.link

    def replacing_parent(
        source: object,
        target: object,
        **kwargs: object,
    ) -> None:
        publish_parent.rename(moved_parent)
        publish_parent.mkdir()
        real_link(source, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(reconciliation.os, "link", replacing_parent)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="destination parent changed during publication",
    ):
        _build(case, destination=destination)

    assert not destination.exists()
    assert not (moved_parent / destination.name).exists()
    assert not list(moved_parent.glob(".reconciliation.json.private-*"))


def test_destination_parent_replacement_after_link_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    publish_parent = tmp_path / "publish"
    publish_parent.mkdir()
    destination = publish_parent / "reconciliation.json"
    moved_parent = tmp_path / "publish-moved"
    real_link = reconciliation.os.link

    def replacing_parent(
        source: object,
        target: object,
        **kwargs: object,
    ) -> None:
        real_link(source, target, **kwargs)  # type: ignore[arg-type]
        publish_parent.rename(moved_parent)
        publish_parent.mkdir()

    monkeypatch.setattr(reconciliation.os, "link", replacing_parent)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="destination parent changed during publication",
    ):
        _build(case, destination=destination)

    assert not destination.exists()
    assert not (moved_parent / destination.name).exists()
    assert not list(moved_parent.glob(".reconciliation.json.private-*"))


@pytest.mark.parametrize(
    "root_attribute",
    ["root", "renderer_root", "rendered_output_root"],
)
def test_input_root_replacement_after_link_rolls_back_own_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_attribute: str,
) -> None:
    case = _make_case(tmp_path)
    root = cast("Path", getattr(case, root_attribute))
    moved_root = tmp_path / f"{root.name}-moved"
    real_link = reconciliation.os.link

    def replacing_root(
        source: object,
        target: object,
        **kwargs: object,
    ) -> None:
        real_link(source, target, **kwargs)  # type: ignore[arg-type]
        root.rename(moved_root)
        root.mkdir()

    monkeypatch.setattr(reconciliation.os, "link", replacing_root)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match=r"(document|renderer|rendered-output) root identity changed",
    ):
        _build(case)

    assert not case.output_path.exists()
    assert not list(tmp_path.glob(".document_reconciliation.json.private-*"))


def test_input_root_replacement_after_final_digest_is_detected_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    moved_root = tmp_path / "rendered-outputs-moved"
    real_verify = reconciliation._verify_published_destination  # noqa: SLF001
    call_count = 0

    def replacing_after_digest(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        real_verify(*args, **kwargs)  # type: ignore[arg-type]
        call_count += 1
        if call_count == 3:
            case.rendered_output_root.rename(moved_root)
            case.rendered_output_root.mkdir()

    monkeypatch.setattr(
        reconciliation,
        "_verify_published_destination",
        replacing_after_digest,
    )
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="rendered-output root identity changed",
    ):
        _build(case)

    assert call_count == 3
    assert not case.output_path.exists()
    assert not list(tmp_path.glob(".document_reconciliation.json.private-*"))


def test_destination_parent_replacement_after_final_digest_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    publish_parent = tmp_path / "publish"
    publish_parent.mkdir()
    moved_parent = tmp_path / "publish-moved"
    destination = publish_parent / "reconciliation.json"
    real_verify = reconciliation._verify_published_destination  # noqa: SLF001
    call_count = 0

    def replacing_after_digest(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        real_verify(*args, **kwargs)  # type: ignore[arg-type]
        call_count += 1
        if call_count == 3:
            publish_parent.rename(moved_parent)
            publish_parent.mkdir()

    monkeypatch.setattr(
        reconciliation,
        "_verify_published_destination",
        replacing_after_digest,
    )
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="destination parent changed during publication",
    ):
        _build(case, destination=destination)

    assert call_count == 3
    assert not destination.exists()
    assert not (moved_parent / destination.name).exists()
    assert not list(moved_parent.glob(".reconciliation.json.private-*"))


def test_draft_allows_scoped_pending_placeholder(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    response_id = f"response-{reconciliation.REVIEWER_ITEM_ORDER[0]}"
    response = _placement(case, response_id)
    response["status"] = "pending"
    response["content_sha256"] = None
    case.contents[response_id] = "**[COAUTH: pending]**\n"
    case.refresh_documents()
    case.refresh_input()

    receipt = _build(case)

    assert receipt.pending_count == 1
    assert receipt.mode == "draft"


def test_final_rejects_pending_placement(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    response_id = f"response-{reconciliation.REVIEWER_ITEM_ORDER[0]}"
    response = _placement(case, response_id)
    response["status"] = "pending"
    response["content_sha256"] = None
    case.contents[response_id] = "**[COAUTH: pending]**\n"
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="final reconciliation cannot contain pending",
    ):
        _build(case)


def test_artifact_anchor_failure_precedes_document_root_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    real_pin_root = reconciliation._pin_root  # noqa: SLF001

    def forbidden_open(path: Path, *, context: str) -> object:
        if context == "document root":
            message = "document root was opened"
            raise AssertionError(message)
        return real_pin_root(path, context=context)

    monkeypatch.setattr(reconciliation, "_pin_root", forbidden_open)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="artifact registry SHA-256",
    ):
        reconciliation.build_document_reconciliation(
            case.input_path,
            case.registry_path,
            case.renderer_root,
            case.rendered_output_root,
            case.anchor_path,
            case.root,
            case.output_path,
            expected_reconciliation_sha256=_sha256(case.input_path.read_bytes()),
            expected_artifact_registry_sha256="0" * 64,
            expected_document_anchor_sha256=_sha256(case.anchor_path.read_bytes()),
        )


def test_reconciliation_binding_failure_precedes_document_root_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    binding = _mapping(case.input_value["binding"])
    binding["artifact_registry_sha256"] = "0" * 64
    _write_json(case.input_path, case.input_value)
    real_pin_root = reconciliation._pin_root  # noqa: SLF001

    def forbidden_open(path: Path, *, context: str) -> object:
        if context == "document root":
            message = "document root was opened"
            raise AssertionError(message)
        return real_pin_root(path, context=context)

    monkeypatch.setattr(reconciliation, "_pin_root", forbidden_open)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match=r"does not bind.*artifact registry",
    ):
        _build(case)


@pytest.mark.parametrize("operation", ["build", "validate"])
def test_native_registry_failure_precedes_document_root_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    case = _make_case(tmp_path)
    if operation == "validate":
        _build(case)
    first = _mapping(_array(case.registry_value["artifacts"])[0])
    output = _mapping(_array(first["outputs"])[0])
    (case.rendered_output_root / cast("str", output["release_member"])).unlink()
    real_pin_root = reconciliation._pin_root  # noqa: SLF001

    def forbidden_open(path: Path, *, context: str) -> object:
        if context == "document root":
            message = "document root was opened"
            raise AssertionError(message)
        return real_pin_root(path, context=context)

    def run_operation() -> None:
        if operation == "build":
            _build(case)
        else:
            _validate(case, case.output_path)

    monkeypatch.setattr(reconciliation, "_pin_root", forbidden_open)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="native artifact registry validation failed",
    ):
        run_operation()


@pytest.mark.parametrize(
    ("root_attribute", "message"),
    [
        ("root", "outside document root"),
        ("renderer_root", "outside renderer root"),
        ("rendered_output_root", "outside rendered-output root"),
    ],
)
def test_destination_must_be_outside_every_input_root(
    tmp_path: Path,
    root_attribute: str,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    root = cast("Path", getattr(case, root_attribute))
    parent = root / "nested-publication"
    parent.mkdir()

    with pytest.raises(reconciliation.DocumentReconciliationError, match=message):
        _build(case, destination=parent / "reconciliation.json")


def test_symlinked_renderer_root_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    alias = tmp_path / "renderer-alias"
    alias.symlink_to(case.renderer_root, target_is_directory=True)

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="renderer root must be a canonical non-symlink directory",
    ):
        reconciliation.build_document_reconciliation(
            case.input_path,
            case.registry_path,
            alias,
            case.rendered_output_root,
            case.anchor_path,
            case.root,
            case.output_path,
            expected_reconciliation_sha256=_sha256(case.input_path.read_bytes()),
            expected_artifact_registry_sha256=_sha256(case.registry_path.read_bytes()),
            expected_document_anchor_sha256=_sha256(case.anchor_path.read_bytes()),
        )


def test_row_bearing_registry_members_are_never_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    real_open_member = artifact_registry._open_member_descriptor  # noqa: SLF001

    def guarded_open_member(
        root: artifact_registry._PinnedRoot,
        member: str,
    ) -> int:
        assert not member.startswith("source-data/")
        return real_open_member(root, member)

    monkeypatch.setattr(
        artifact_registry,
        "_open_member_descriptor",
        guarded_open_member,
    )

    _build(case)
    assert (
        "source_data_root_path"
        not in inspect.signature(
            reconciliation.build_document_reconciliation,
        ).parameters
    )


def _remove_claim(case: SyntheticCase) -> None:
    first = artifact_registry.ARTIFACT_SPECS[0]
    claims = _array(
        _placement(case, _artifact_placement_id(first.semantic_id))["artifact_claims"],
    )
    claims.pop()


def _duplicate_claim(case: SyntheticCase) -> None:
    first, second = artifact_registry.ARTIFACT_SPECS[:2]
    claim = {
        "artifact_id": first.semantic_id,
        "claim_id": first.claims[0].claim_id,
    }
    claims = _array(
        _placement(case, _artifact_placement_id(second.semantic_id))["artifact_claims"],
    )
    claims.insert(0, claim)


def _remove_reviewer_item(case: SyntheticCase) -> None:
    _array(case.input_value["reviewer_items"]).pop()


def _reuse_response(case: SyntheticCase) -> None:
    reviewers = [_mapping(item) for item in _array(case.input_value["reviewer_items"])]
    reviewers[1]["response_placement_id"] = reviewers[0]["response_placement_id"]


def _break_target(case: SyntheticCase) -> None:
    reviewer = _mapping(_array(case.input_value["reviewer_items"])[0])
    reviewer["target_placement_ids"] = ["missing-target"]


def _disconnect_reviewer_artifacts(case: SyntheticCase) -> None:
    reviewer = _mapping(_array(case.input_value["reviewer_items"])[0])
    first = artifact_registry.ARTIFACT_SPECS[0]
    reviewer["target_placement_ids"] = [
        _artifact_placement_id(first.semantic_id),
    ]


def _clear_reviewer_artifact_links(case: SyntheticCase) -> None:
    for raw in _array(case.input_value["reviewer_items"]):
        _mapping(raw)["artifact_ids"] = []


def _remove_artifact_placement_gates(case: SyntheticCase) -> None:
    first = artifact_registry.ARTIFACT_SPECS[0]
    placement = _placement(case, _artifact_placement_id(first.semantic_id))
    placement["required_gates"] = []


def _change_artifact_placement_kind(case: SyntheticCase) -> None:
    first = artifact_registry.ARTIFACT_SPECS[0]
    placement = _placement(case, _artifact_placement_id(first.semantic_id))
    placement["kind"] = "table"


def _remove_page_location(case: SyntheticCase) -> None:
    first = artifact_registry.ARTIFACT_SPECS[0]
    _placement(case, _artifact_placement_id(first.semantic_id))["page_location"] = None


def _remove_figure_gate(case: SyntheticCase) -> None:
    ledger = _array(case.input_value["gate_ledger"])
    ledger[:] = [record for record in ledger if _mapping(record)["gate"] != "FIG"]


def _reverse_gate_ledger(case: SyntheticCase) -> None:
    _array(case.input_value["gate_ledger"]).reverse()


def _remove_s1_placements(case: SyntheticCase) -> None:
    for raw in _array(case.input_value["placements"]):
        placement = _mapping(raw)
        if placement["document_id"] == "s1":
            placement["document_id"] = "main"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_remove_claim, "artifact claim closure failed"),
        (_duplicate_claim, "mapped by both"),
        (_remove_reviewer_item, "all 27 reviewer items"),
        (_reuse_response, "multiple reviewer items"),
        (_break_target, "target_placement_ids is invalid"),
        (_disconnect_reviewer_artifacts, "not carried by its target placements"),
        (_clear_reviewer_artifact_links, "reviewer-item artifact closure failed"),
        (_remove_artifact_placement_gates, "required_gates contradicts"),
        (_change_artifact_placement_kind, "kind contradicts"),
        (_remove_page_location, "lacks page/line evidence"),
        (_remove_figure_gate, "is ready but lacks gates"),
        (_reverse_gate_ledger, "gate_ledger is not canonically ordered"),
        (_remove_s1_placements, "placements in main, S1, and rebuttal"),
    ],
)
def test_reconciliation_contract_rejects_incomplete_or_duplicate_mapping(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    cast("object", mutate)(case)  # type: ignore[operator]
    case.refresh_input()

    with pytest.raises(reconciliation.DocumentReconciliationError, match=message):
        _build(case)


def test_valid_omission_is_registry_bound_and_accounted(tmp_path: Path) -> None:
    omitted_id = artifact_registry.ARTIFACT_SPECS[0].semantic_id
    case = _make_case(tmp_path, omitted_ids=frozenset({omitted_id}))

    receipt = _build(case)

    assert receipt.omitted_count == 1
    manifest = _mapping(json.loads(case.output_path.read_bytes()))
    assert _mapping(manifest["artifact_registry"])["omitted_count"] == 1


@pytest.mark.parametrize(
    "corruption",
    ["reason", "ready-placement", "omitted-ready-artifact", "token-scope", "no-claims"],
)
def test_invalid_omission_is_rejected(tmp_path: Path, corruption: str) -> None:
    first = artifact_registry.ARTIFACT_SPECS[0]
    if corruption == "omitted-ready-artifact":
        case = _make_case(tmp_path)
        placement = _placement(case, _artifact_placement_id(first.semantic_id))
        placement["status"] = "omitted"
        placement["omission"] = {
            "reason_code": "coauthor_decision_to_omit",
            "receipt_id": "coauthor-omission-receipt",
            "sha256": _sha256(b"coauthor omission"),
            "forbidden_token_ids": [_STALE_MARKER_ID],
        }
        message = "cannot omit a ready artifact"
    else:
        case = _make_case(
            tmp_path,
            omitted_ids=frozenset({first.semantic_id}),
        )
        placement = _placement(case, _artifact_placement_id(first.semantic_id))
        omission = _mapping(placement["omission"])
        if corruption == "reason":
            omission["reason_code"] = "release_scope_exclusion"
            message = "reason contradicts"
        elif corruption == "ready-placement":
            placement["status"] = "ready"
            placement["omission"] = None
            message = "maps an omitted artifact as ready"
        elif corruption == "token-scope":
            token = _mapping(_array(case.input_value["forbidden_tokens"])[0])
            token["document_ids"] = ["s1"]
            message = "token does not cover its document"
        else:
            placement["artifact_claims"] = []
            message = "must own at least one artifact claim"
    case.refresh_input()

    with pytest.raises(reconciliation.DocumentReconciliationError, match=message):
        _build(case)


def test_ready_block_rejects_unresolved_placeholder(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first_id = _artifact_placement_id(artifact_registry.ARTIFACT_SPECS[0].semantic_id)
    case.contents[first_id] = "**[K500: pending]**\n"
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="contains an unresolved placeholder",
    ):
        _build(case)


def test_ready_block_rejects_whitespace_only_content(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first_id = _artifact_placement_id(artifact_registry.ARTIFACT_SPECS[0].semantic_id)
    case.contents[first_id] = " \n"
    _placement(case, first_id)["content_sha256"] = _sha256(b" \n")
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="has empty final content",
    ):
        _build(case)


def test_ready_block_rejects_generic_todo_placeholder(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first_id = _artifact_placement_id(artifact_registry.ARTIFACT_SPECS[0].semantic_id)
    content = "TODO: replace with final scientific prose.\n"
    case.contents[first_id] = content
    _placement(case, first_id)["content_sha256"] = _sha256(content.encode())
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="contains an unresolved placeholder",
    ):
        _build(case)


def test_placeholder_outside_target_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.prefixes["main"] += "% K500 gate: pending\n"
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="placeholder outside a target",
    ):
        _build(case)


def test_forbidden_stale_token_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.prefixes["s1"] += f"{_STALE_LITERAL}\n"
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="forbidden stale token",
    ):
        _build(case)


def test_missing_and_extra_marked_targets_are_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first_id = _artifact_placement_id(artifact_registry.ARTIFACT_SPECS[0].semantic_id)
    block = (
        f"{_marker('main', first_id, 'BEGIN')}\n"
        f"{case.contents[first_id]}"
        f"{_marker('main', first_id, 'END')}\n"
    )
    main = case.root / "main.tex"
    main.write_text(
        main.read_text(encoding="utf-8").replace(block, ""),
        encoding="utf-8",
    )
    with main.open("a", encoding="utf-8") as stream:
        stream.write(
            "% RECONCILIATION-TARGET:extra-target:BEGIN\n"
            "Extra.\n"
            "% RECONCILIATION-TARGET:extra-target:END\n",
        )
    case.reanchor_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="target closure failed",
    ):
        _build(case)


def test_noncanonical_marker_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first_id = _artifact_placement_id(artifact_registry.ARTIFACT_SPECS[0].semantic_id)
    main = case.root / "main.tex"
    canonical = _marker("main", first_id, "BEGIN")
    main.write_text(
        main.read_text(encoding="utf-8").replace(canonical, f"{canonical} trailing"),
        encoding="utf-8",
    )
    case.reanchor_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="noncanonical target marker",
    ):
        _build(case)


def test_content_digest_drift_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    first_id = _artifact_placement_id(artifact_registry.ARTIFACT_SPECS[0].semantic_id)
    case.contents[first_id] = "Changed but otherwise complete prose.\n"
    case.refresh_documents()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="content digest does not match",
    ):
        _build(case)


def test_document_drift_from_anchor_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    main = case.root / "main.tex"
    with main.open("a", encoding="utf-8") as stream:
        stream.write("Unanchored mutation.\n")

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="differs from its independent anchor",
    ):
        _build(case)


def test_document_symlink_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    main = case.root / "main.tex"
    real_main = tmp_path / "real-main.tex"
    main.rename(real_main)
    main.symlink_to(real_main)

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="cannot open document main",
    ):
        _build(case)


def test_document_hardlink_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    os.link(case.root / "main.tex", tmp_path / "hardlink-main.tex")

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="exactly one hard link",
    ):
        _build(case)


@pytest.mark.parametrize("member", ["../main.tex", "main//tex", "a/../main.tex"])
def test_document_anchor_rejects_noncanonical_or_traversing_member(
    tmp_path: Path,
    member: str,
) -> None:
    case = _make_case(tmp_path)
    anchor = _mapping(json.loads(case.anchor_path.read_bytes()))
    first = _mapping(_array(anchor["documents"])[0])
    first["member"] = member
    _write_json(case.anchor_path, anchor)
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match=r"(canonical POSIX|safe relative POSIX)",
    ):
        _build(case)


def test_noncanonical_reconciliation_input_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case.input_path.write_text(
        json.dumps(case.input_value, indent=2, sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="not canonical JSON",
    ):
        _build(case)


def test_registry_payload_tampering_is_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    release = _mapping(case.registry_value["release"])
    release["unbound_field"] = "tampered"
    _write_json(case.registry_path, case.registry_value)
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="native artifact registry validation failed",
    ):
        _build(case)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("contract", "wrong contract"),
        ("trust", "explicit trust model"),
        ("gate-catalog", "frozen gate catalog"),
        ("builder", "renderer script"),
        ("extra-ready-key", "non-closed schema"),
        ("missing-source-role", "missing artifact-specific source roles"),
        ("renderer", "analysis Python module"),
        ("output", "unsupported media type or suffix"),
        ("gate-receipt", "contradicts the global ledger"),
    ],
)
def test_redigested_registry_metadata_must_match_native_contract(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    artifacts = [_mapping(item) for item in _array(case.registry_value["artifacts"])]
    first = artifacts[0]
    if corruption == "contract":
        case.registry_value["contract"] = "different-contract"
    elif corruption == "trust":
        case.registry_value["trust_model"] = {"source_data": "changed"}
    elif corruption == "gate-catalog":
        _array(case.registry_value["gate_catalog"]).pop()
    elif corruption == "builder":
        _mapping(case.registry_value["builder"])["bytes"] = 1
    elif corruption == "extra-ready-key":
        first["unexpected"] = True
    elif corruption == "missing-source-role":
        _array(first["source_data"]).pop()
    elif corruption == "renderer":
        _mapping(first["renderer"])["script"] = "render.py"
    elif corruption == "output":
        _mapping(_array(first["outputs"])[0])["media_type"] = "text/csv"
    else:
        _mapping(_array(first["gate_receipts"])[0])["sha256"] = "e" * 64
    case.refresh_registry()
    case.refresh_input()

    with pytest.raises(reconciliation.DocumentReconciliationError, match=message):
        _build(case)


def test_redigested_registry_omission_must_match_native_reason(
    tmp_path: Path,
) -> None:
    first = artifact_registry.ARTIFACT_SPECS[0]
    case = _make_case(tmp_path, omitted_ids=frozenset({first.semantic_id}))
    artifact = _mapping(_array(case.registry_value["artifacts"])[0])
    _mapping(artifact["omission"])["reason"] = "Noncanonical explanation."
    case.refresh_registry()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match=r"omission\.reason is not canonical",
    ):
        _build(case)


def test_registry_builder_must_match_live_native_builder_after_native_validation(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    replacement = b"# synthetic but non-live registry builder\n"
    builder_path = (
        case.renderer_root / "analysis/build_tcga_revision_artifact_registry.py"
    )
    builder_path.write_bytes(replacement)
    builder = _mapping(case.registry_value["builder"])
    builder["sha256"] = _sha256(replacement)
    builder["bytes"] = len(replacement)
    case.refresh_registry()
    case.refresh_input()

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="does not bind the live native builder",
    ):
        _build(case)


@pytest.mark.parametrize("operation", ["build", "validate"])
def test_later_metadata_pin_failure_closes_earlier_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    case = _make_case(tmp_path)
    if operation == "validate":
        _build(case)
    real_pin = reconciliation._pin_file  # noqa: SLF001
    first_descriptor: int | None = None
    call_count = 0

    def failing_pin(path: Path, *, maximum: int, context: str) -> object:
        nonlocal call_count, first_descriptor
        call_count += 1
        if call_count == 2:
            message = "synthetic pin failure"
            raise reconciliation.DocumentReconciliationError(message)
        pinned = real_pin(path, maximum=maximum, context=context)
        if first_descriptor is None:
            first_descriptor = pinned.descriptor
        return pinned

    def run_operation() -> None:
        if operation == "build":
            _build(case)
        else:
            _validate(case, case.output_path)

    monkeypatch.setattr(reconciliation, "_pin_file", failing_pin)
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="synthetic pin failure",
    ):
        run_operation()

    assert first_descriptor is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(first_descriptor)


def test_manifest_self_digest_and_canonical_fields_are_revalidated(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    _build(case)
    manifest = _mapping(json.loads(case.output_path.read_bytes()))
    summary = _mapping(manifest["summary"])
    summary["artifact_count"] = 12
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = _sha256(_json_bytes(manifest))
    tampered = tmp_path / "tampered_manifest.json"
    _write_json(tampered, manifest)

    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="manifest field 'summary' is not canonical",
    ):
        _validate(case, tampered)


def test_manifest_validation_detects_output_root_swap_after_final_native_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _build(case)
    moved_root = tmp_path / "rendered-outputs-moved-during-validation"
    real_validate = reconciliation._validate_native_artifact_registry  # noqa: SLF001
    call_count = 0

    def swapping_validation(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        receipt = real_validate(*args, **kwargs)  # type: ignore[arg-type]
        call_count += 1
        if call_count == 2:
            case.rendered_output_root.rename(moved_root)
            case.rendered_output_root.mkdir()
        return receipt

    monkeypatch.setattr(
        reconciliation,
        "_validate_native_artifact_registry",
        swapping_validation,
    )
    with pytest.raises(
        reconciliation.DocumentReconciliationError,
        match="rendered-output root identity changed",
    ):
        _validate(case, case.output_path)

    assert call_count == 2

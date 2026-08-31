"""Synthetic adversarial tests for rendered-document evidence closure."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from matplotlib.figure import Figure

from analysis import build_tcga_revision_rendered_document_evidence as evidence

# Adversarial tests intentionally exercise private pinning/publication seams.
# ruff: noqa: D101, D102, EM101, S603, SLF001, TRY003

_RELEASE_ID = "dialect-revision-final"
_BASELINE_SHA = hashlib.sha256(b"baseline source snapshot").hexdigest()
_REVISED_SHA = hashlib.sha256(b"revised source snapshot").hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _mapping(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value)


def _array(value: object) -> list[object]:
    return cast("list[object]", value)


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


@functools.lru_cache(maxsize=8)
def _png(width: int = 1275, height: int = 1650) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + bytes(width * 3)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(scanline * height))
        + _chunk(b"IEND", b"")
    )


@dataclass(slots=True)
class SyntheticCase:
    root: Path
    plan_path: Path
    reconciliation_path: Path
    registry_path: Path
    release_evidence_path: Path
    document_anchor_path: Path
    source_snapshot_path: Path
    derivation_path: Path
    machine_path: Path
    visual_path: Path
    pdf_root: Path
    png_root: Path
    renderer_root: Path
    rendered_output_root: Path
    gate_root: Path
    source_root: Path
    document_root: Path
    output_path: Path
    mode: str = "final"

    def pdf_digest(self, pdf_id: str) -> str:
        member = evidence.PDF_MEMBER_BY_ID[pdf_id]
        return _sha((self.pdf_root / member).read_bytes())

    def png_member(self, pdf_id: str, page: int = 1) -> str:
        return f"pages/{pdf_id}/page-{page:04d}.png"

    def png_digest(self, pdf_id: str, page: int = 1) -> str:
        return _sha((self.png_root / self.png_member(pdf_id, page)).read_bytes())

    def render_set(self) -> tuple[list[dict[str, object]], str]:
        documents: list[dict[str, object]] = []
        for pdf_id, member in evidence.PDF_ORDER:
            pdf_raw = (self.pdf_root / member).read_bytes()
            png_member = self.png_member(pdf_id)
            png_raw = (self.png_root / png_member).read_bytes()
            documents.append(
                {
                    "pdf_id": pdf_id,
                    "pdf_role": evidence.PDF_ROLE_BY_ID[pdf_id],
                    "pdf_member": member,
                    "pdf_sha256": _sha(pdf_raw),
                    "pdf_bytes": len(pdf_raw),
                    "page_count": 1,
                    "pages": [
                        {
                            "page": 1,
                            "png_member": png_member,
                            "png_sha256": _sha(png_raw),
                            "png_bytes": len(png_raw),
                        },
                    ],
                },
            )
        return documents, _sha(
            _canonical(
                {
                    "render_settings": evidence.RENDER_SETTINGS,
                    "documents": documents,
                },
            ),
        )

    def rewrite_reconciliation(
        self,
        *,
        reconciliation_mode: str | None = None,
        pending_count: int = 0,
    ) -> None:
        mode = reconciliation_mode or self.mode
        value = {
            "schema": evidence._reconciliation.DOCUMENT_RECONCILIATION_SCHEMA,
            "mode": mode,
            "release_id": _RELEASE_ID,
            "inputs": {
                "artifact_registry_sha256": _sha(self.registry_path.read_bytes()),
                "document_anchor_sha256": _sha(
                    self.document_anchor_path.read_bytes(),
                ),
            },
            "artifact_registry": {"ready_count": 13, "omitted_count": 0},
            "placements": [
                {
                    "placement_id": "main-placement",
                    "document_id": "main",
                    "page_location": {
                        "pdf_sha256": self.pdf_digest("clean"),
                        "page": 1,
                    },
                },
                {
                    "placement_id": "rebuttal-placement",
                    "document_id": "rebuttal",
                    "page_location": None,
                },
                {
                    "placement_id": "s1-placement",
                    "document_id": "s1",
                    "page_location": {
                        "pdf_sha256": self.pdf_digest("s1"),
                        "page": 1,
                    },
                },
            ],
            "summary": {
                "placement_count": 3,
                "ready_count": 3 - pending_count,
                "omitted_count": 0,
                "pending_count": pending_count,
            },
        }
        _write_json(self.reconciliation_path, value)

    def rewrite_snapshot(self) -> None:
        value = {
            "schema": evidence.SOURCE_SNAPSHOT_ANCHOR_SCHEMA,
            "release_id": _RELEASE_ID,
            "document_anchor_sha256": _sha(self.document_anchor_path.read_bytes()),
            "snapshots": [
                {
                    "kind": "baseline",
                    "snapshot_id": "baseline-source",
                    "sha256": _BASELINE_SHA,
                    "bytes": 101,
                },
                {
                    "kind": "revised",
                    "snapshot_id": "revised-source",
                    "sha256": _REVISED_SHA,
                    "bytes": 202,
                },
            ],
            "clean_marked_bindings": [
                {
                    "pdf_id": pdf_id,
                    "baseline_snapshot_sha256": _BASELINE_SHA,
                    "revised_snapshot_sha256": _REVISED_SHA,
                }
                for pdf_id in ("clean", "marked")
            ],
            "non_inference_limits": evidence.NON_INFERENCE_LIMITS,
        }
        _write_json(self.source_snapshot_path, value)

    def rewrite_derivation(self, *, unproven: frozenset[str] = frozenset()) -> None:
        shared = {
            "candidate_manifest_sha256": _sha(b"candidate manifest"),
            "candidate_manifest_bytes": 111,
            "baseline_snapshot_sha256": _BASELINE_SHA,
            "revised_snapshot_sha256": _REVISED_SHA,
            "accepted_roundtrip_sha256": _sha(b"accepted roundtrip"),
            "accepted_roundtrip_bytes": 112,
            "declined_roundtrip_sha256": _sha(b"declined roundtrip"),
            "declined_roundtrip_bytes": 113,
        }
        documents: list[dict[str, object]] = []
        for pdf_id, _ in evidence.PDF_ORDER:
            if pdf_id in unproven:
                documents.append(
                    {
                        "pdf_id": pdf_id,
                        "source_document_id": evidence.SOURCE_DOCUMENT_BY_PDF_ID[
                            pdf_id
                        ],
                        "status": "unproven",
                        "evidence": None,
                    },
                )
                continue
            rebuild_sha = self.pdf_digest(pdf_id)
            if pdf_id in {"clean", "marked"}:
                record: dict[str, object] = {
                    **shared,
                    "source_to_pdf_receipt_sha256": _sha(
                        f"{pdf_id} build receipt".encode(),
                    ),
                    "source_to_pdf_receipt_bytes": 114,
                    "rebuild_a_sha256": rebuild_sha,
                    "rebuild_b_sha256": rebuild_sha,
                }
                if pdf_id == "marked":
                    record.update(
                        {
                            "latexdiff_receipt_sha256": _sha(b"latexdiff receipt"),
                            "latexdiff_receipt_bytes": 115,
                        },
                    )
            else:
                record = {
                    "approved_manifest_sha256": _sha(
                        f"{pdf_id} approved manifest".encode(),
                    ),
                    "approved_manifest_bytes": 116,
                    "source_to_pdf_receipt_sha256": _sha(
                        f"{pdf_id} build receipt".encode(),
                    ),
                    "source_to_pdf_receipt_bytes": 117,
                    "external_pdf_qa_receipt_sha256": _sha(
                        f"{pdf_id} external QA".encode(),
                    ),
                    "external_pdf_qa_receipt_bytes": 118,
                    "rebuild_a_sha256": rebuild_sha,
                    "rebuild_b_sha256": rebuild_sha,
                }
            documents.append(
                {
                    "pdf_id": pdf_id,
                    "source_document_id": evidence.SOURCE_DOCUMENT_BY_PDF_ID[pdf_id],
                    "status": "attested",
                    "evidence": record,
                },
            )
        value = {
            "schema": evidence.DERIVATION_EVIDENCE_SCHEMA,
            "mode": self.mode,
            "release_id": _RELEASE_ID,
            "bindings": {
                "document_reconciliation_sha256": _sha(
                    self.reconciliation_path.read_bytes(),
                ),
                "release_evidence_sha256": _sha(
                    self.release_evidence_path.read_bytes(),
                ),
                "source_snapshot_anchor_sha256": _sha(
                    self.source_snapshot_path.read_bytes(),
                ),
            },
            "nonpromotable_pdf_sha256": [],
            "documents": documents,
            "non_inference_limits": evidence.NON_INFERENCE_LIMITS,
        }
        _write_json(self.derivation_path, value)

    def rewrite_machine(
        self,
        *,
        encrypted_id: str | None = None,
        type3_id: str | None = None,
    ) -> None:
        _, render_set_sha = self.render_set()
        documents: list[dict[str, object]] = []
        for pdf_id, member in evidence.PDF_ORDER:
            pdf_raw = (self.pdf_root / member).read_bytes()
            png_member = self.png_member(pdf_id)
            png_raw = (self.png_root / png_member).read_bytes()
            encrypted = pdf_id == encrypted_id
            font_type = "Type 3" if pdf_id == type3_id else "Type 1"
            issues = []
            if encrypted:
                issues.append("encrypted")
            if font_type == "Type 3":
                issues.append("type-3-font")
            documents.append(
                {
                    "pdf_id": pdf_id,
                    "pdf_member": member,
                    "pdf_sha256": _sha(pdf_raw),
                    "pdf_bytes": len(pdf_raw),
                    "pdf_version": "1.7",
                    "page_count": 1,
                    "encrypted": encrypted,
                    "raster_image_count": 0,
                    "pdfinfo_sha256": _sha(f"pdfinfo {pdf_id}".encode()),
                    "pdfinfo_bytes": 200,
                    "pdffonts_sha256": _sha(f"pdffonts {pdf_id}".encode()),
                    "pdffonts_bytes": 201,
                    "pdfimages_sha256": _sha(f"pdfimages {pdf_id}".encode()),
                    "pdfimages_bytes": 202,
                    "pages": [
                        {
                            "page": 1,
                            "media_box_millipoints": [0, 0, 612_000, 792_000],
                            "crop_box_millipoints": [0, 0, 612_000, 792_000],
                            "width_millipoints": 612_000,
                            "height_millipoints": 792_000,
                            "rotation_degrees": 0,
                            "png_member": png_member,
                            "png_sha256": _sha(png_raw),
                            "png_bytes": len(png_raw),
                            "rendered_width_pixels": 1275,
                            "rendered_height_pixels": 1650,
                            "render_a_sha256": _sha(png_raw),
                            "render_a_bytes": len(png_raw),
                            "render_b_sha256": _sha(png_raw),
                            "render_b_bytes": len(png_raw),
                        },
                    ],
                    "fonts": [
                        {
                            "name": "SyntheticFont",
                            "type": font_type,
                            "encoding": "Custom",
                            "embedded": True,
                            "subset": True,
                            "unicode": True,
                        },
                    ],
                    "status": "fail" if issues else "pass",
                    "issues": sorted(issues),
                },
            )
        value = {
            "schema": evidence.MACHINE_EVIDENCE_SCHEMA,
            "mode": self.mode,
            "release_id": _RELEASE_ID,
            "producer": {
                "producer_receipt_sha256": _sha(b"machine producer receipt"),
                "producer_receipt_bytes": 300,
                "tools": [
                    {
                        "name": name,
                        "absolute_path": f"/synthetic/bin/{name}",
                        "sha256": _sha(name.encode()),
                        "bytes": 301,
                        "version": "synthetic-1.0",
                    }
                    for name in ("pdfinfo", "pdffonts", "pdfimages", "pdftoppm")
                ],
            },
            "tool_contract": evidence.MACHINE_TOOL_CONTRACT,
            "render_settings": evidence.RENDER_SETTINGS,
            "render_set_sha256": render_set_sha,
            "documents": documents,
            "non_inference_limits": evidence.NON_INFERENCE_LIMITS,
        }
        _write_json(self.machine_path, value)

    def rewrite_visual(
        self,
        *,
        pending_id: str | None = None,
        fail_id: str | None = None,
    ) -> None:
        _, render_set_sha = self.render_set()
        reviewer_id = "synthetic-reviewer"
        reviewer_role = "independent-visual-reviewer"
        documents: list[dict[str, object]] = []
        for pdf_id, _ in evidence.PDF_ORDER:
            decision = "pass"
            issue_codes: list[str] = []
            if pdf_id == pending_id:
                decision = "pending"
                issue_codes = ["review-pending"]
            elif pdf_id == fail_id:
                decision = "fail"
                issue_codes = ["clipping"]
            documents.append(
                {
                    "pdf_id": pdf_id,
                    "pdf_sha256": self.pdf_digest(pdf_id),
                    "page_count": 1,
                    "pages": [
                        {
                            "reviewer_id": reviewer_id,
                            "reviewer_role": reviewer_role,
                            "pdf_role": evidence.PDF_ROLE_BY_ID[pdf_id],
                            "page": 1,
                            "pdf_sha256": self.pdf_digest(pdf_id),
                            "png_member": self.png_member(pdf_id),
                            "png_sha256": self.png_digest(pdf_id),
                            "render_set_sha256": render_set_sha,
                            "decision": decision,
                            "issue_codes": issue_codes,
                        },
                    ],
                },
            )
        value = {
            "schema": evidence.VISUAL_QA_RECEIPT_SCHEMA,
            "mode": self.mode,
            "release_id": _RELEASE_ID,
            "review_kind": "human",
            "review_id": "all-page-review",
            "reviewer_id": reviewer_id,
            "reviewer_role": reviewer_role,
            "reviewed_at_utc": "2026-08-31T12:00:00Z",
            "independent_review": True,
            "criteria": evidence.VISUAL_REVIEW_CRITERIA,
            "render_set_sha256": render_set_sha,
            "documents": documents,
            "non_inference_limits": evidence.NON_INFERENCE_LIMITS,
        }
        _write_json(self.visual_path, value)

    def rewrite_plan(self) -> None:
        value = {
            "schema": evidence.RENDERED_DOCUMENT_INPUT_SCHEMA,
            "mode": self.mode,
            "release_id": _RELEASE_ID,
            "bindings": {
                "document_reconciliation_sha256": _sha(
                    self.reconciliation_path.read_bytes(),
                ),
                "artifact_registry_sha256": _sha(self.registry_path.read_bytes()),
                "release_evidence_sha256": _sha(
                    self.release_evidence_path.read_bytes(),
                ),
                "document_anchor_sha256": _sha(
                    self.document_anchor_path.read_bytes(),
                ),
                "source_snapshot_anchor_sha256": _sha(
                    self.source_snapshot_path.read_bytes(),
                ),
                "derivation_evidence_sha256": _sha(
                    self.derivation_path.read_bytes(),
                ),
                "machine_evidence_sha256": _sha(self.machine_path.read_bytes()),
                "visual_qa_receipt_sha256": _sha(self.visual_path.read_bytes()),
            },
            "render_settings": evidence.RENDER_SETTINGS,
            "documents": [
                {
                    "pdf_id": pdf_id,
                    "pdf_member": member,
                    "pdf_role": evidence.PDF_ROLE_BY_ID[pdf_id],
                    "source_document_id": evidence.SOURCE_DOCUMENT_BY_PDF_ID[pdf_id],
                    "output_prefix_template": f"pages/{pdf_id}/page-{{page:04d}}",
                    "page_count": 1,
                    "page_size_millipoints": {
                        "width": 612_000,
                        "height": 792_000,
                    },
                }
                for pdf_id, member in evidence.PDF_ORDER
            ],
            "non_inference_limits": evidence.NON_INFERENCE_LIMITS,
        }
        _write_json(self.plan_path, value)

    def rewrite_all(self) -> None:
        self.rewrite_reconciliation()
        self.rewrite_snapshot()
        self.rewrite_derivation()
        self.rewrite_machine()
        self.rewrite_visual()
        self.rewrite_plan()

    def refresh_from_derivation(self) -> None:
        """Rewrite downstream anchored metadata after derivation changes."""
        self.rewrite_plan()

    def config(self) -> evidence.RenderedDocumentEvidenceInputs:
        return evidence.RenderedDocumentEvidenceInputs(
            plan_path=self.plan_path,
            reconciliation_path=self.reconciliation_path,
            artifact_registry_path=self.registry_path,
            renderer_root=self.renderer_root,
            rendered_output_root=self.rendered_output_root,
            release_evidence_path=self.release_evidence_path,
            gate_receipt_root=self.gate_root,
            source_data_root=self.source_root,
            document_anchor_path=self.document_anchor_path,
            document_root=self.document_root,
            source_snapshot_anchor_path=self.source_snapshot_path,
            derivation_evidence_path=self.derivation_path,
            machine_evidence_path=self.machine_path,
            visual_qa_receipt_path=self.visual_path,
            pdf_root=self.pdf_root,
            png_root=self.png_root,
            expected_plan_sha256=_sha(self.plan_path.read_bytes()),
            expected_reconciliation_sha256=_sha(
                self.reconciliation_path.read_bytes(),
            ),
            expected_artifact_registry_sha256=_sha(
                self.registry_path.read_bytes(),
            ),
            expected_release_evidence_sha256=_sha(
                self.release_evidence_path.read_bytes(),
            ),
            expected_document_anchor_sha256=_sha(
                self.document_anchor_path.read_bytes(),
            ),
            expected_source_snapshot_anchor_sha256=_sha(
                self.source_snapshot_path.read_bytes(),
            ),
            expected_derivation_evidence_sha256=_sha(
                self.derivation_path.read_bytes(),
            ),
            expected_machine_evidence_sha256=_sha(self.machine_path.read_bytes()),
            expected_visual_qa_receipt_sha256=_sha(self.visual_path.read_bytes()),
        )


def _make_case(tmp_path: Path, *, mode: str = "final") -> SyntheticCase:
    pdf_root = tmp_path / "pdfs"
    png_root = tmp_path / "renders"
    pdf_root.mkdir(parents=True)
    png_root.mkdir()
    for pdf_id, member in evidence.PDF_ORDER:
        (pdf_root / member).write_bytes(
            f"%PDF-1.7\nsynthetic {pdf_id}\n%%EOF\n".encode(),
        )
        png_path = png_root / f"pages/{pdf_id}/page-0001.png"
        png_path.parent.mkdir(parents=True)
        png_path.write_bytes(_png())
    roots = {
        name: tmp_path / name
        for name in (
            "renderer-root",
            "rendered-output-root",
            "gate-root",
            "source-root",
            "document-root",
        )
    }
    for root in roots.values():
        root.mkdir()
    case = SyntheticCase(
        root=tmp_path,
        plan_path=tmp_path / "plan.json",
        reconciliation_path=tmp_path / "reconciliation.json",
        registry_path=tmp_path / "registry.json",
        release_evidence_path=tmp_path / "release-evidence.json",
        document_anchor_path=tmp_path / "document-anchor.json",
        source_snapshot_path=tmp_path / "source-snapshot.json",
        derivation_path=tmp_path / "derivation.json",
        machine_path=tmp_path / "machine.json",
        visual_path=tmp_path / "visual.json",
        pdf_root=pdf_root,
        png_root=png_root,
        renderer_root=roots["renderer-root"],
        rendered_output_root=roots["rendered-output-root"],
        gate_root=roots["gate-root"],
        source_root=roots["source-root"],
        document_root=roots["document-root"],
        output_path=tmp_path / "rendered-document-evidence.json",
        mode=mode,
    )
    _write_json(case.registry_path, {"synthetic": "registry"})
    _write_json(case.release_evidence_path, {"synthetic": "release evidence"})
    _write_json(case.document_anchor_path, {"synthetic": "document anchor"})
    case.rewrite_all()
    return case


@pytest.fixture(autouse=True)
def _native_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate_reconciliation(
        manifest_path: Path,
        *_args: object,
        expected_manifest_sha256: str,
        **_kwargs: object,
    ) -> evidence._reconciliation.DocumentReconciliationReceipt:
        value = json.loads(manifest_path.read_text(encoding="ascii"))
        pending = int(_mapping(value["summary"])["pending_count"])
        return evidence._reconciliation.DocumentReconciliationReceipt(
            manifest_path=str(manifest_path.absolute()),
            manifest_sha256=expected_manifest_sha256,
            mode=str(value["mode"]),
            placement_count=len(_array(value["placements"])),
            ready_count=len(_array(value["placements"])) - pending,
            omitted_count=0,
            pending_count=pending,
        )

    def validate_release(
        closure_path: Path,
        *_args: object,
        expected_closure_sha256: str,
        **_kwargs: object,
    ) -> evidence._release_evidence.ReleaseEvidenceReceipt:
        return evidence._release_evidence.ReleaseEvidenceReceipt(
            manifest_path=str(closure_path.absolute()),
            manifest_sha256=expected_closure_sha256,
            gate_receipt_count=9,
            source_member_count=35,
            ready_count=13,
            omitted_count=0,
        )

    monkeypatch.setattr(
        evidence._reconciliation,
        "validate_document_reconciliation",
        validate_reconciliation,
    )
    monkeypatch.setattr(
        evidence._release_evidence,
        "validate_release_evidence_closure",
        validate_release,
    )


def _build(case: SyntheticCase) -> evidence.RenderedDocumentEvidenceReceipt:
    return evidence.build_rendered_document_evidence(case.config(), case.output_path)


def test_draft_build_and_independent_validation_close_exact_mechanical_chain(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    receipt = _build(case)
    assert receipt.mode == "draft"
    assert receipt.pdf_count == 4
    assert receipt.page_count == 4
    assert receipt.machine_attested_pass_count == 4
    assert receipt.visual_pass_page_count == 4
    assert receipt.promotable is False
    manifest = json.loads(case.output_path.read_text(encoding="ascii"))
    assert manifest["summary"]["png_count"] == 4
    assert (
        manifest["render_set"]["sha256"] == manifest["visual_qa"]["render_set_sha256"]
    )
    assert manifest["non_inference_limits"] == evidence.NON_INFERENCE_LIMITS
    assert [item["pdf_id"] for item in manifest["documents"]] == list(
        evidence.PDF_IDS,
    )
    assert manifest["summary"]["promotion_blockers"] == [
        "native-derivation-producer-closure-not-validated",
        "native-machine-producer-closure-not-validated",
        "human-reviewer-identity-not-authenticated",
    ]
    assert manifest["promotion_policy"] == evidence.PROMOTION_POLICY
    assert manifest["public_release_relationship"] == (
        evidence.PUBLIC_RELEASE_RELATIONSHIP
    )
    assert manifest["derivation_evidence"]["reference_validation"] == (
        "unverified-external-references"
    )
    assert manifest["machine_evidence"]["reference_validation"] == (
        "unverified-external-references"
    )
    assert all(
        item["status"] == "attested"
        for item in manifest["derivation_evidence"]["documents"]
    )
    validated = evidence.validate_rendered_document_evidence(
        case.config(),
        case.output_path,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    assert validated == receipt


def test_final_complete_wrapper_attestations_remain_nonpromotable(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="native derivation and machine producer closures",
    ):
        _build(case)


def test_native_upstream_validators_run_before_pdf_or_png_root_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    events: list[str] = []

    def stop(*_args: object, **_kwargs: object) -> object:
        events.append("reconciliation")
        raise evidence._reconciliation.DocumentReconciliationError("stop")

    real_pin_root = evidence._pin_root

    def record_root(path: Path, *, context: str) -> object:
        events.append(context)
        return real_pin_root(path, context=context)

    monkeypatch.setattr(
        evidence._reconciliation,
        "validate_document_reconciliation",
        stop,
    )
    monkeypatch.setattr(evidence, "_pin_root", record_root)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="native document"):
        _build(case)
    assert events == ["reconciliation"]


def test_release_evidence_validation_precedes_rendered_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    opened: list[str] = []

    def stop(*_args: object, **_kwargs: object) -> object:
        raise evidence._release_evidence.ReleaseEvidenceError("stop")

    def record(*args: object, **kwargs: object) -> object:
        opened.append(str(args[0]))
        return evidence._PinnedFile(*args, **kwargs)

    monkeypatch.setattr(
        evidence._release_evidence,
        "validate_release_evidence_closure",
        stop,
    )
    monkeypatch.setattr(evidence, "_open_root_member", record)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="release-evidence",
    ):
        _build(case)
    assert opened == []


def test_native_validator_calls_use_exact_paths_and_digest_keywords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    reconciliation_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    release_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def reconciliation_spy(
        *args: object,
        **kwargs: object,
    ) -> evidence._reconciliation.DocumentReconciliationReceipt:
        reconciliation_calls.append((args, kwargs))
        value = json.loads(case.reconciliation_path.read_text(encoding="ascii"))
        return evidence._reconciliation.DocumentReconciliationReceipt(
            manifest_path=str(case.reconciliation_path.absolute()),
            manifest_sha256=str(kwargs["expected_manifest_sha256"]),
            mode=str(value["mode"]),
            placement_count=3,
            ready_count=3,
            omitted_count=0,
            pending_count=0,
        )

    def release_spy(
        *args: object,
        **kwargs: object,
    ) -> evidence._release_evidence.ReleaseEvidenceReceipt:
        release_calls.append((args, kwargs))
        return evidence._release_evidence.ReleaseEvidenceReceipt(
            manifest_path=str(case.release_evidence_path.absolute()),
            manifest_sha256=str(kwargs["expected_closure_sha256"]),
            gate_receipt_count=9,
            source_member_count=35,
            ready_count=13,
            omitted_count=0,
        )

    monkeypatch.setattr(
        evidence._reconciliation,
        "validate_document_reconciliation",
        reconciliation_spy,
    )
    monkeypatch.setattr(
        evidence._release_evidence,
        "validate_release_evidence_closure",
        release_spy,
    )
    _build(case)
    expected_reconciliation_args = (
        case.reconciliation_path,
        case.registry_path,
        case.renderer_root,
        case.rendered_output_root,
        case.document_anchor_path,
        case.document_root,
    )
    expected_reconciliation_kwargs = {
        "expected_manifest_sha256": _sha(case.reconciliation_path.read_bytes()),
        "expected_artifact_registry_sha256": _sha(case.registry_path.read_bytes()),
        "expected_document_anchor_sha256": _sha(
            case.document_anchor_path.read_bytes(),
        ),
    }
    expected_release_args = (
        case.release_evidence_path,
        case.registry_path,
        case.renderer_root,
        case.rendered_output_root,
        case.gate_root,
        case.source_root,
    )
    expected_release_kwargs = {
        "expected_closure_sha256": _sha(case.release_evidence_path.read_bytes()),
        "expected_artifact_registry_sha256": _sha(case.registry_path.read_bytes()),
    }
    assert reconciliation_calls
    assert release_calls
    assert all(
        args == expected_reconciliation_args
        and kwargs == expected_reconciliation_kwargs
        for args, kwargs in reconciliation_calls
    )
    assert all(
        args == expected_release_args and kwargs == expected_release_kwargs
        for args, kwargs in release_calls
    )


def test_native_receipt_paths_digests_and_counts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    original_reconciliation = evidence._reconciliation.validate_document_reconciliation

    def wrong_reconciliation(
        *_args: object,
        expected_manifest_sha256: str,
        **_kwargs: object,
    ) -> evidence._reconciliation.DocumentReconciliationReceipt:
        return evidence._reconciliation.DocumentReconciliationReceipt(
            manifest_path=str((tmp_path / "wrong.json").absolute()),
            manifest_sha256=expected_manifest_sha256,
            mode="final",
            placement_count=3,
            ready_count=3,
            omitted_count=0,
            pending_count=0,
        )

    monkeypatch.setattr(
        evidence._reconciliation,
        "validate_document_reconciliation",
        wrong_reconciliation,
    )
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="wrong manifest path",
    ):
        _build(case)
    monkeypatch.setattr(
        evidence._reconciliation,
        "validate_document_reconciliation",
        original_reconciliation,
    )

    def wrong_release(
        closure_path: Path,
        *_args: object,
        expected_closure_sha256: str,
        **_kwargs: object,
    ) -> evidence._release_evidence.ReleaseEvidenceReceipt:
        return evidence._release_evidence.ReleaseEvidenceReceipt(
            manifest_path=str(closure_path.absolute()),
            manifest_sha256=expected_closure_sha256,
            gate_receipt_count=-1,
            source_member_count=35,
            ready_count=13,
            omitted_count=0,
        )

    case = _make_case(tmp_path / "release")
    monkeypatch.setattr(
        evidence._release_evidence,
        "validate_release_evidence_closure",
        wrong_release,
    )
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="invalid count"):
        _build(case)


def test_draft_is_never_promotable_and_may_retain_pending_review(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    case.rewrite_derivation(unproven=frozenset({"marked", "rebuttal"}))
    case.rewrite_visual(pending_id="marked")
    case.rewrite_plan()
    receipt = _build(case)
    assert receipt.mode == "draft"
    assert receipt.promotable is False
    manifest = json.loads(case.output_path.read_text(encoding="ascii"))
    assert manifest["summary"]["visual_pending_page_count"] == 1


def test_draft_can_bind_final_reconciliation_without_becoming_promotable(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    case.rewrite_reconciliation(reconciliation_mode="final")
    case.rewrite_derivation()
    case.rewrite_plan()
    assert _build(case).promotable is False


def test_draft_may_record_absent_human_visual_receipt(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    plan = json.loads(case.plan_path.read_text(encoding="ascii"))
    _mapping(plan["bindings"])["visual_qa_receipt_sha256"] = None
    _write_json(case.plan_path, plan)
    config = replace(
        case.config(),
        visual_qa_receipt_path=None,
        expected_visual_qa_receipt_sha256=None,
    )
    receipt = _build_with_config(case, config)
    assert receipt.promotable is False
    manifest = json.loads(case.output_path.read_text(encoding="ascii"))
    assert manifest["visual_qa"] is None
    assert manifest["summary"]["visual_pending_page_count"] == 4
    assert "human-visual-qa-receipt-absent" in manifest["summary"]["promotion_blockers"]


def test_final_requires_human_visual_receipt_binding(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    plan = json.loads(case.plan_path.read_text(encoding="ascii"))
    _mapping(plan["bindings"])["visual_qa_receipt_sha256"] = None
    _write_json(case.plan_path, plan)
    config = replace(
        case.config(),
        visual_qa_receipt_path=None,
        expected_visual_qa_receipt_sha256=None,
    )
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="requires a visual-QA receipt binding",
    ):
        _build_with_config(case, config)


def test_final_rejects_draft_or_pending_reconciliation(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    case.rewrite_reconciliation(reconciliation_mode="draft")
    case.rewrite_derivation()
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="final plan"):
        _build(case)
    case.rewrite_reconciliation(pending_count=1)
    case.rewrite_derivation()
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="zero-pending"):
        _build(case)


@pytest.mark.parametrize("pdf_id", evidence.PDF_IDS)
def test_final_rejects_unproven_document_derivation(
    tmp_path: Path,
    pdf_id: str,
) -> None:
    case = _make_case(tmp_path)
    case.rewrite_derivation(unproven=frozenset({pdf_id}))
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="cannot remain"):
        _build(case)


def test_marked_requires_latexdiff_and_identical_rebuilds(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    value = json.loads(case.derivation_path.read_text(encoding="ascii"))
    marked = _mapping(_array(value["documents"])[1])
    marked_evidence = _mapping(marked["evidence"])
    marked_evidence.pop("latexdiff_receipt_sha256")
    _write_json(case.derivation_path, value)
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="invalid keys"):
        _build(case)
    case.rewrite_derivation()
    value = json.loads(case.derivation_path.read_text(encoding="ascii"))
    marked = _mapping(_array(value["documents"])[1])
    _mapping(marked["evidence"])["rebuild_b_sha256"] = "f" * 64
    _write_json(case.derivation_path, value)
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="byte-identical"):
        _build(case)


def test_main_and_s1_page_locations_bind_exact_pdf_and_page_bounds(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    value = json.loads(case.reconciliation_path.read_text(encoding="ascii"))
    main = _mapping(_array(value["placements"])[0])
    _mapping(main["page_location"])["pdf_sha256"] = "f" * 64
    _write_json(case.reconciliation_path, value)
    case.rewrite_derivation()
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="exact clean PDF"):
        _build(case)
    case.rewrite_reconciliation()
    value = json.loads(case.reconciliation_path.read_text(encoding="ascii"))
    s1 = _mapping(_array(value["placements"])[2])
    _mapping(s1["page_location"])["page"] = 2
    _write_json(case.reconciliation_path, value)
    case.rewrite_derivation()
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="outside the s1"):
        _build(case)


@pytest.mark.parametrize("pdf_id", evidence.PDF_IDS)
def test_exact_pdf_inventory_rejects_missing_or_extra(
    tmp_path: Path,
    pdf_id: str,
) -> None:
    case = _make_case(tmp_path)
    (case.pdf_root / evidence.PDF_MEMBER_BY_ID[pdf_id]).unlink()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="inventory mismatch",
    ):
        _build(case)


def test_exact_render_tree_rejects_extra_empty_directory_and_extra_file(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    (case.png_root / "pages/extra").mkdir()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="inventory mismatch",
    ):
        _build(case)
    (case.png_root / "pages/extra").rmdir()
    (case.png_root / "pages/clean/extra.png").write_bytes(_png())
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="inventory mismatch",
    ):
        _build(case)


def test_missing_page_and_noncanonical_flat_page_are_rejected(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    page = case.png_root / case.png_member("clean")
    page.unlink()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="inventory mismatch",
    ):
        _build(case)
    page.write_bytes(_png())
    (case.png_root / "clean-1.png").write_bytes(_png())
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="inventory mismatch",
    ):
        _build(case)


def test_png_structure_rejects_bad_crc_apng_and_wrong_dimensions(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    page = case.png_root / case.png_member("clean")
    raw = bytearray(page.read_bytes())
    raw[29] ^= 1
    page.write_bytes(raw)
    case.rewrite_derivation()
    case.rewrite_machine()
    case.rewrite_visual()
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="CRC"):
        _build(case)
    page.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1275, 1650, 8, 2, 0, 0, 0))
        + _chunk(b"acTL", struct.pack(">II", 1, 0))
        + _chunk(b"IEND", b""),
    )
    case.rewrite_derivation()
    case.rewrite_machine()
    case.rewrite_visual()
    case.rewrite_plan()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="must not be APNG",
    ):
        _build(case)
    page.write_bytes(_png(width=100, height=100))
    case.rewrite_derivation()
    case.rewrite_machine()
    case.rewrite_visual()
    case.rewrite_plan()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="does not match PNG IHDR",
    ):
        _build(case)


@pytest.mark.parametrize(
    ("case_name", "raw", "message"),
    [
        (
            "corrupt-zlib",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", 1275, 1650, 8, 2, 0, 0, 0),
            )
            + _chunk(b"IDAT", b"not-a-zlib-stream")
            + _chunk(b"IEND", b""),
            "invalid zlib",
        ),
        (
            "invalid-ihdr",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", 1275, 1650, 16, 2, 0, 0, 0),
            )
            + _chunk(b"IDAT", zlib.compress(b""))
            + _chunk(b"IEND", b""),
            "unsupported IHDR",
        ),
        (
            "invalid-filter",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(b"\x05\x00\x00\x00"))
            + _chunk(b"IEND", b""),
            "scanline filter",
        ),
        (
            "short-geometry",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", 1275, 1650, 8, 2, 0, 0, 0),
            )
            + _chunk(b"IDAT", zlib.compress(b"\x00" + bytes(1275 * 3)))
            + _chunk(b"IEND", b""),
            "scanline geometry",
        ),
        (
            "unknown-critical",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", 1275, 1650, 8, 2, 0, 0, 0),
            )
            + _chunk(b"ABCD", b"")
            + _chunk(b"IDAT", zlib.compress(b""))
            + _chunk(b"IEND", b""),
            "unknown critical",
        ),
        (
            "invalid-type-code",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", 1275, 1650, 8, 2, 0, 0, 0),
            )
            + _chunk(b"ABcD", b"")
            + _chunk(b"IDAT", zlib.compress(b""))
            + _chunk(b"IEND", b""),
            "chunk type code",
        ),
        (
            "malformed-trns",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"tRNS", b"\x00")
            + _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
            + _chunk(b"IEND", b""),
            "unsupported ancillary.*tRNS",
        ),
        (
            "malformed-gama",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"gAMA", b"\x00\x01\x02")
            + _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
            + _chunk(b"IEND", b""),
            "unsupported ancillary.*gAMA",
        ),
        (
            "malformed-phys",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"pHYs", b"\x00" * 8)
            + _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
            + _chunk(b"IEND", b""),
            "invalid pHYs",
        ),
        (
            "wrong-phys-rounding",
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + _chunk(b"pHYs", struct.pack(">IIB", 5906, 5906, 1))
            + _chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
            + _chunk(b"IEND", b""),
            "fixed 150 dpi profile",
        ),
    ],
)
def test_png_decoder_rejects_malformed_streams(
    tmp_path: Path,
    case_name: str,
    raw: bytes,
    message: str,
) -> None:
    case = _make_case(tmp_path / case_name)
    (case.png_root / case.png_member("clean")).write_bytes(raw)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match=message):
        _build(case)


def test_png_decoder_rejects_nonconsecutive_idat(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    compressed = zlib.compress((b"\x00" + bytes(1275 * 3)) * 1650)
    midpoint = len(compressed) // 2
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1275, 1650, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed[:midpoint])
        + _chunk(b"tEXt", b"key\x00value")
        + _chunk(b"IDAT", compressed[midpoint:])
        + _chunk(b"IEND", b"")
    )
    (case.png_root / case.png_member("clean")).write_bytes(raw)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="nonconsecutive",
    ):
        _build(case)


def test_png_decoder_accepts_exact_optional_150_dpi_phys_chunk(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    original = _png()
    ihdr_end = 8 + 4 + 4 + 13 + 4
    raw = (
        original[:ihdr_end]
        + _chunk(
            b"pHYs",
            struct.pack(
                ">IIB",
                evidence._EXPECTED_PNG_PIXELS_PER_METER,
                evidence._EXPECTED_PNG_PIXELS_PER_METER,
                1,
            ),
        )
        + original[ihdr_end:]
    )
    assert evidence._EXPECTED_PNG_PIXELS_PER_METER == 5905
    (case.png_root / case.png_member("clean")).write_bytes(raw)
    case.rewrite_machine()
    case.rewrite_visual()
    case.rewrite_plan()
    assert _build(case).promotable is False


def test_real_pdftoppm_synthetic_canary_matches_png_contract(tmp_path: Path) -> None:
    executable = shutil.which("pdftoppm")
    if executable is None:
        pytest.skip("pdftoppm is unavailable")
    pdf_root = tmp_path / "pdf-root"
    png_root = tmp_path / "png-root"
    pdf_root.mkdir()
    png_root.mkdir()
    pdf_path = pdf_root / evidence.PDF_MEMBER_BY_ID["clean"]
    figure = Figure(figsize=(8.5, 11))
    figure.text(0.5, 0.5, "synthetic page", ha="center", va="center")
    figure.savefig(pdf_path, format="pdf")
    relative_prefix = str(evidence.RENDER_SETTINGS["output_prefix_template"]).format(
        pdf_id="clean",
        page=1,
    )
    output_prefix = png_root / relative_prefix
    output_prefix.parent.mkdir(parents=True)
    argv = [
        str(token).format(
            page=1,
            pdftoppm_absolute_path=str(Path(executable).resolve()),
            pdf_absolute_path=str(pdf_path.resolve()),
            output_absolute_prefix=str(output_prefix.absolute()),
        )
        for token in evidence.RENDER_SETTINGS["argv_template"]
    ]
    assert argv[0] == str(Path(executable).resolve())
    assert argv[-2] == str(pdf_path.resolve())
    assert argv[-1] == str(output_prefix.absolute())
    assert pdf_path.parent == pdf_root
    assert output_prefix.parent.is_relative_to(png_root)
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(evidence.RENDER_SETTINGS["cwd"]),
        env={
            str(key): str(value)
            for key, value in evidence.RENDER_SETTINGS["environment"].items()
        },
    )
    assert completed.returncode == 0, completed.stderr
    expected_member = str(evidence.RENDER_SETTINGS["output_member_template"]).format(
        pdf_id="clean",
        page=1,
    )
    png_path = png_root / expected_member
    assert png_path == Path(f"{output_prefix}.png")
    assert png_path.is_file()
    pinned = evidence._pin_file(
        png_path,
        maximum=evidence._MAX_PNG_BYTES,
        context="synthetic Poppler canary",
        capture=False,
    )
    try:
        assert evidence._parse_png_dimensions(
            pinned,
            context="synthetic Poppler canary",
        ) == (1275, 1650)
    finally:
        pinned.close()
    raw = png_path.read_bytes()
    offset = len(b"\x89PNG\r\n\x1a\n")
    phys: tuple[int, int, int] | None = None
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        data = raw[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs":
            phys = struct.unpack(">IIB", data)
        offset += 12 + length
    assert phys == (
        evidence._EXPECTED_PNG_PIXELS_PER_METER,
        evidence._EXPECTED_PNG_PIXELS_PER_METER,
        1,
    )
    assert phys == (5905, 5905, 1)


def test_rendered_pins_do_not_capture_full_pdf_or_png_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    original = evidence._assert_render_signatures
    observed = False

    def inspect(
        pdf_pins: dict[str, evidence._PinnedFile],
        png_pins: dict[str, evidence._PinnedFile],
    ) -> dict[str, tuple[int, int]]:
        nonlocal observed
        assert all(pin.raw is None for pin in (*pdf_pins.values(), *png_pins.values()))
        observed = True
        return original(pdf_pins, png_pins)

    monkeypatch.setattr(evidence, "_assert_render_signatures", inspect)
    _build(case)
    assert observed


def test_pdf_roles_must_bind_byte_distinct_documents(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    clean = case.pdf_root / evidence.PDF_MEMBER_BY_ID["clean"]
    marked = case.pdf_root / evidence.PDF_MEMBER_BY_ID["marked"]
    marked.write_bytes(clean.read_bytes())
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="byte-distinct"):
        _build(case)


def test_cropbox_render_contract_accepts_inner_letter_cropbox(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    value = json.loads(case.machine_path.read_text(encoding="ascii"))
    clean = _mapping(_array(value["documents"])[0])
    page = _mapping(_array(clean["pages"])[0])
    page["media_box_millipoints"] = [0, 0, 620_000, 800_000]
    page["crop_box_millipoints"] = [4_000, 4_000, 616_000, 796_000]
    _write_json(case.machine_path, value)
    case.rewrite_plan()
    assert "-cropbox" in evidence.RENDER_SETTINGS["argv_template"]
    assert _build(case).promotable is False


def test_draft_rotated_page_uses_swapped_render_geometry(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    page_path = case.png_root / case.png_member("clean")
    page_path.write_bytes(_png(width=1650, height=1275))
    case.rewrite_machine()
    value = json.loads(case.machine_path.read_text(encoding="ascii"))
    clean = _mapping(_array(value["documents"])[0])
    page = _mapping(_array(clean["pages"])[0])
    page["rotation_degrees"] = 90
    page["rendered_width_pixels"] = 1650
    page["rendered_height_pixels"] = 1275
    clean["status"] = "fail"
    clean["issues"] = ["rotated-page"]
    _write_json(case.machine_path, value)
    case.rewrite_visual()
    case.rewrite_plan()
    receipt = _build(case)
    assert receipt.promotable is False
    manifest = json.loads(case.output_path.read_text(encoding="ascii"))
    assert _mapping(_array(manifest["documents"])[0])["machine_attestation_issues"] == [
        "rotated-page",
    ]


def test_machine_evidence_requires_two_identical_renders(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    value = json.loads(case.machine_path.read_text(encoding="ascii"))
    clean = _mapping(_array(value["documents"])[0])
    page = _mapping(_array(clean["pages"])[0])
    page["render_b_sha256"] = "f" * 64
    _write_json(case.machine_path, value)
    case.rewrite_plan()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="two byte-identical",
    ):
        _build(case)


@pytest.mark.parametrize(
    "absolute_path",
    ["/opt/../bin/pdftoppm", "//host/bin/pdftoppm", "/"],
)
def test_machine_tool_path_must_be_canonical_absolute_file(
    tmp_path: Path,
    absolute_path: str,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    value = json.loads(case.machine_path.read_text(encoding="ascii"))
    producer = _mapping(value["producer"])
    pdftoppm = _mapping(_array(producer["tools"])[3])
    pdftoppm["absolute_path"] = absolute_path
    _write_json(case.machine_path, value)
    case.rewrite_plan()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="not canonical absolute POSIX",
    ):
        _build(case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("encrypted", "failing machine checks"),
        ("type3", "failing machine checks"),
    ],
)
def test_final_machine_failures_are_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    case.rewrite_machine(
        encrypted_id="clean" if mutation == "encrypted" else None,
        type3_id="clean" if mutation == "type3" else None,
    )
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match=message):
        _build(case)


def test_clean_and_marked_require_us_letter_and_zero_raster_images(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    plan = json.loads(case.plan_path.read_text(encoding="ascii"))
    clean_plan = _mapping(_array(plan["documents"])[0])
    _mapping(clean_plan["page_size_millipoints"])["width"] = 595_000
    _write_json(case.plan_path, plan)
    config = case.config()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="status/issues"):
        evidence.build_rendered_document_evidence(config, case.output_path)
    case.rewrite_plan()
    machine = json.loads(case.machine_path.read_text(encoding="ascii"))
    clean_machine = _mapping(_array(machine["documents"])[0])
    clean_machine["raster_image_count"] = 1
    clean_machine["status"] = "fail"
    clean_machine["issues"] = ["raster-image-present"]
    _write_json(case.machine_path, machine)
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="failing machine"):
        _build(case)


def test_visual_receipt_binds_render_set_reviewer_pdf_and_png(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    for field, value, message in (
        ("render_set_sha256", "f" * 64, "render set"),
        ("reviewer_id", "different-reviewer", "human reviewer"),
        ("pdf_sha256", "f" * 64, "reviewed PDF"),
        ("png_sha256", "f" * 64, "exact PNG"),
    ):
        case.rewrite_visual()
        visual = json.loads(case.visual_path.read_text(encoding="ascii"))
        page = _mapping(
            _array(_mapping(_array(visual["documents"])[0])["pages"])[0],
        )
        page[field] = value
        _write_json(case.visual_path, visual)
        case.rewrite_plan()
        with pytest.raises(evidence.RenderedDocumentEvidenceError, match=message):
            _build(case)


def test_visual_receipt_timestamp_must_be_a_real_utc_date(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    value = json.loads(case.visual_path.read_text(encoding="ascii"))
    value["reviewed_at_utc"] = "2026-02-31T12:00:00Z"
    _write_json(case.visual_path, value)
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="real UTC date"):
        _build(case)


def test_final_rejects_pending_or_failed_visual_page(tmp_path: Path) -> None:
    for decision in ("pending", "fail"):
        case = _make_case(tmp_path / decision)
        if decision == "pending":
            case.rewrite_visual(pending_id="clean")
        else:
            case.rewrite_visual(fail_id="clean")
        case.rewrite_plan()
        with pytest.raises(
            evidence.RenderedDocumentEvidenceError,
            match="pass every page",
        ):
            _build(case)


def test_source_snapshot_and_clean_marked_candidate_bindings_are_exact(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    snapshot = json.loads(case.source_snapshot_path.read_text(encoding="ascii"))
    marked = _mapping(_array(snapshot["clean_marked_bindings"])[1])
    marked["baseline_snapshot_sha256"] = "f" * 64
    _write_json(case.source_snapshot_path, snapshot)
    case.rewrite_derivation()
    case.rewrite_plan()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="baseline snapshot",
    ):
        _build(case)


def test_known_working_pdf_digest_is_nonpromotable(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    derivation = json.loads(case.derivation_path.read_text(encoding="ascii"))
    derivation["nonpromotable_pdf_sha256"] = [case.pdf_digest("clean")]
    _write_json(case.derivation_path, derivation)
    case.rewrite_plan()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="nonpromotable"):
        _build(case)


def test_plan_non_inference_and_fixed_render_settings_cannot_drift(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    plan = json.loads(case.plan_path.read_text(encoding="ascii"))
    _mapping(plan["render_settings"])["dpi"] = 120
    _write_json(case.plan_path, plan)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="fixed render"):
        _build(case)
    case.rewrite_plan()
    plan = json.loads(case.plan_path.read_text(encoding="ascii"))
    _mapping(plan["non_inference_limits"])["submission_approval"] = "inferred"
    _write_json(case.plan_path, plan)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="limits drifted"):
        _build(case)


@pytest.mark.parametrize("page", [1, 9, 10, 100])
def test_single_page_render_template_emits_exact_canonical_name(page: int) -> None:
    settings = evidence.RENDER_SETTINGS
    prefix = str(settings["output_prefix_template"]).format(
        pdf_id="clean",
        page=page,
    )
    member = str(settings["output_member_template"]).format(
        pdf_id="clean",
        page=page,
    )
    argv = [
        str(token).format(
            page=page,
            pdftoppm_absolute_path="/synthetic/bin/pdftoppm",
            pdf_absolute_path="/synthetic/pdf-root/manuscript-clean.pdf",
            output_absolute_prefix=f"/synthetic/png-root/{prefix}",
        )
        for token in settings["argv_template"]
    ]
    assert prefix == f"pages/clean/page-{page:04d}"
    assert member == f"{prefix}.png"
    assert argv[argv.index("-f") + 1] == str(page)
    assert argv[argv.index("-l") + 1] == str(page)
    assert "-singlefile" in argv
    assert "-cropbox" in argv


def test_metadata_and_render_members_reject_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    original = case.machine_path
    alias = tmp_path / "machine-alias.json"
    alias.symlink_to(original)
    config = case.config()
    config = replace(config, machine_evidence_path=alias)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="non-symlink"):
        _build_with_config(case, config)
    alias.unlink()
    page = case.png_root / case.png_member("clean")
    hardlink = tmp_path / "page-hardlink.png"
    os.link(page, hardlink)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="one hard link"):
        _build(case)


def _build_with_config(
    case: SyntheticCase,
    config: evidence.RenderedDocumentEvidenceInputs,
) -> evidence.RenderedDocumentEvidenceReceipt:
    return evidence.build_rendered_document_evidence(config, case.output_path)


def test_oversize_metadata_is_rejected_before_read(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    with case.machine_path.open("r+b") as handle:
        handle.truncate(evidence._MAX_METADATA_BYTES + 1)
    config = case.config()
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="byte limit"):
        _build_with_config(case, config)
    assert not list(tmp_path.glob(".rendered-document-evidence.json.private-*"))


@pytest.mark.parametrize("kind", ["pdf", "png"])
def test_sparse_oversize_render_member_is_rejected_before_rendered_hash_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    case = _make_case(tmp_path)
    if kind == "pdf":
        target = case.pdf_root / evidence.PDF_MEMBER_BY_ID["clean"]
    else:
        target = case.png_root / case.png_member("clean")
    with target.open("r+b") as handle:
        handle.truncate(
            evidence._MAX_PDF_BYTES + 1
            if kind == "pdf"
            else evidence._MAX_PNG_BYTES + 1,
        )
    original_hash = evidence._hash_descriptor
    rendered_hash_contexts: list[str] = []
    pread_called = False

    def record_hash(descriptor: int, *, maximum: int, context: str) -> str:
        if context.startswith(("PDF ", "PNG ")):
            rendered_hash_contexts.append(context)
        return original_hash(descriptor, maximum=maximum, context=context)

    def record_pread(*args: object, **kwargs: object) -> bytes:
        nonlocal pread_called
        _ = (args, kwargs)
        pread_called = True
        return b""

    monkeypatch.setattr(evidence, "_hash_descriptor", record_hash)
    monkeypatch.setattr(evidence, "_pread_exact", record_pread)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="per-file byte limit",
    ):
        _build(case)
    assert rendered_hash_contexts == []
    assert pread_called is False


def test_aggregate_png_cap_is_rejected_before_rendered_hash_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    actual_total = sum(
        (case.png_root / case.png_member(pdf_id)).stat().st_size
        for pdf_id in evidence.PDF_IDS
    )
    monkeypatch.setattr(evidence, "_MAX_TOTAL_PNG_BYTES", actual_total - 1)
    original_hash = evidence._hash_descriptor
    rendered_hash_contexts: list[str] = []
    pread_called = False

    def record_hash(descriptor: int, *, maximum: int, context: str) -> str:
        if context.startswith(("PDF ", "PNG ")):
            rendered_hash_contexts.append(context)
        return original_hash(descriptor, maximum=maximum, context=context)

    def record_pread(*args: object, **kwargs: object) -> bytes:
        nonlocal pread_called
        _ = (args, kwargs)
        pread_called = True
        return b""

    monkeypatch.setattr(evidence, "_hash_descriptor", record_hash)
    monkeypatch.setattr(evidence, "_pread_exact", record_pread)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="aggregate byte limit",
    ):
        _build(case)
    assert rendered_hash_contexts == []
    assert pread_called is False


def test_expected_render_leaf_rejects_fifo_before_open(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    target = case.png_root / case.png_member("clean")
    target.unlink()
    os.mkfifo(target)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="invalid type"):
        _build(case)


def test_destination_is_no_replace_and_outside_input_roots(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    case.output_path.write_bytes(b"existing")
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="already exists"):
        _build(case)
    case.output_path.unlink()
    inside = case.pdf_root / "evidence.json"
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="outside every"):
        evidence.build_rendered_document_evidence(case.config(), inside)


def test_pre_rename_failure_retains_private_stage_instead_of_racy_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    calls = 0
    original = evidence._PreparedEvidence.revalidate

    def fail_after_stage(self: evidence._PreparedEvidence) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise evidence.RenderedDocumentEvidenceError("synthetic race")
        original(self)

    monkeypatch.setattr(evidence._PreparedEvidence, "revalidate", fail_after_stage)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="synthetic race",
    ) as captured:
        _build(case)
    assert "candidate_path=" in str(captured.value)
    assert "expected_sha256=" in str(captured.value)
    assert "expected_bytes=" in str(captured.value)
    assert "private-stage-name-may-be-owned-or-replaced" in str(captured.value)
    assert not case.output_path.exists()
    stages = list(tmp_path.glob(".rendered-document-evidence.json.private-*"))
    assert len(stages) == 1
    assert stages[0].stat().st_mode & 0o777 == 0o400
    monkeypatch.setattr(evidence._PreparedEvidence, "revalidate", original)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="retained private stage requires explicit review",
    ):
        _build(case)
    assert len(list(tmp_path.glob(".rendered-document-evidence.json.private-*"))) == 1


def test_post_rename_failure_preserves_owned_destination_without_name_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    calls = 0
    original = evidence._PreparedEvidence.revalidate

    def fail_after_rename(self: evidence._PreparedEvidence) -> None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise evidence.RenderedDocumentEvidenceError("post-rename race")
        original(self)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("failure cleanup must never unlink a mutable name")

    monkeypatch.setattr(evidence._PreparedEvidence, "revalidate", fail_after_rename)
    monkeypatch.setattr(evidence.os, "unlink", forbidden_unlink)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="post-rename race",
    ) as captured:
        _build(case)
    assert "candidate_path=" in str(captured.value)
    assert "expected_sha256=" in str(captured.value)
    assert "expected_bytes=" in str(captured.value)
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)
    assert case.output_path.exists()
    assert case.output_path.stat().st_mode & 0o777 == 0o400
    assert not list(tmp_path.glob(".rendered-document-evidence.json.private-*"))


def test_competitor_destination_at_atomic_rename_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    original = evidence._rename_no_replace
    competitor = b"competitor-owned"

    def race(source: str, destination: str, directory_descriptor: int) -> None:
        case.output_path.write_bytes(competitor)
        original(source, destination, directory_descriptor)

    monkeypatch.setattr(evidence, "_rename_no_replace", race)
    with pytest.raises(evidence.RenderedDocumentEvidenceError, match="already exists"):
        _build(case)
    assert case.output_path.read_bytes() == competitor
    assert len(list(tmp_path.glob(".rendered-document-evidence.json.private-*"))) == 1


def test_partial_stage_failure_reports_unknown_candidate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")
    original_write = evidence.os.write
    calls = 0

    def partial_then_fail(descriptor: int, raw: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, raw[: max(1, len(raw) // 2)])
        raise OSError("synthetic partial write failure")

    monkeypatch.setattr(evidence.os, "write", partial_then_fail)
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="synthetic partial write failure",
    ) as captured:
        _build(case)
    message = str(captured.value)
    assert "expected_sha256=unknown" in message
    assert "expected_bytes=unknown" in message
    assert "private-stage-name-may-be-owned-or-replaced" in message
    assert len(list(tmp_path.glob(".rendered-document-evidence.json.private-*"))) == 1


def test_missing_atomic_rename_symbol_fails_closed_with_retained_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path, mode="draft")

    class MissingRenameLibrary:
        pass

    monkeypatch.setattr(
        evidence.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: MissingRenameLibrary(),
    )
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="rename symbol",
    ) as captured:
        _build(case)
    assert "candidate_path=" in str(captured.value)
    assert len(list(tmp_path.glob(".rendered-document-evidence.json.private-*"))) == 1


def test_validation_rejects_manifest_tamper_and_is_read_only(tmp_path: Path) -> None:
    case = _make_case(tmp_path, mode="draft")
    receipt = _build(case)
    before = case.output_path.stat()
    with pytest.raises(
        evidence.RenderedDocumentEvidenceError,
        match="independent anchor",
    ):
        evidence.validate_rendered_document_evidence(
            case.config(),
            case.output_path,
            expected_manifest_sha256="f" * 64,
        )
    validated = evidence.validate_rendered_document_evidence(
        case.config(),
        case.output_path,
        expected_manifest_sha256=receipt.manifest_sha256,
    )
    after = case.output_path.stat()
    assert validated == receipt
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def test_cli_help_paths_are_available() -> None:
    script = Path(evidence.__file__)
    for argv in (["--help"], ["build", "--help"], ["validate", "--help"]):
        result = subprocess.run(
            [sys.executable, str(script), *argv],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "evidence" in result.stdout.lower()


def test_repeated_failures_do_not_leak_file_descriptors(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    fd_root = Path("/dev/fd")
    if not fd_root.exists():
        pytest.skip("descriptor inventory is unavailable")
    before = len(list(fd_root.iterdir()))
    plan = json.loads(case.plan_path.read_text(encoding="ascii"))
    _mapping(plan["bindings"])["machine_evidence_sha256"] = "f" * 64
    _write_json(case.plan_path, plan)
    config = case.config()
    for _ in range(10):
        with pytest.raises(evidence.RenderedDocumentEvidenceError):
            _build_with_config(case, config)
    after = len(list(fd_root.iterdir()))
    assert after <= before + 2


def test_low_descriptor_limit_subprocess_fails_headroom_preflight() -> None:
    script = """
import resource
from analysis import build_tcga_revision_rendered_document_evidence as evidence

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
soft = 80 if hard == resource.RLIM_INFINITY else min(80, hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
try:
    evidence._validate_fd_headroom(
        20,
        gate_receipt_count=9,
        source_member_count=35,
    )
except evidence.RenderedDocumentEvidenceError as error:
    print(error)
else:
    raise SystemExit("headroom preflight unexpectedly passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "RLIMIT_NOFILE headroom" in result.stdout

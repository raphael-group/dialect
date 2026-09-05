"""Tests for the focused submission document-manifest producer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis import build_tcga_revision_focused_document_manifest as documents
from analysis import build_tcga_revision_focused_release as release

if TYPE_CHECKING:
    from pathlib import Path


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    document_root = tmp_path / "documents"
    document_root.mkdir()
    for name in release.REQUIRED_DOCUMENTS:
        (document_root / name).write_text(
            f"final {name}\n",
            encoding="utf-8",
        )
    report_manifest = tmp_path / "report_manifest.json"
    report_manifest.write_text('{"complete":true}\n', encoding="utf-8")
    return document_root, report_manifest


def test_build_and_verify_exact_document_manifest(tmp_path: Path) -> None:
    document_root, report_manifest = _fixture(tmp_path)

    path = documents.build_manifest(
        document_root=document_root,
        report_manifest=report_manifest,
    )
    first_bytes = path.read_bytes()

    assert documents.verify_manifest(
        document_root=document_root,
        report_manifest=report_manifest,
    ) == path
    value = json.loads(first_bytes)
    assert value["schema_version"] == release.SCHEMA_VERSION
    assert value["contract"] == release.DOCUMENT_CONTRACT
    assert set(value["outputs"]) == release.REQUIRED_DOCUMENTS
    assert value["inputs"]["report_manifest"]["path"] == "report_manifest.json"
    assert first_bytes.endswith(b"\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )


def test_verify_rejects_document_mutation(tmp_path: Path) -> None:
    document_root, report_manifest = _fixture(tmp_path)
    documents.build_manifest(
        document_root=document_root,
        report_manifest=report_manifest,
    )

    (document_root / "manuscript.tex").write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer matches"):
        documents.verify_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )


def test_build_rejects_extra_files_and_wrong_report_name(tmp_path: Path) -> None:
    document_root, report_manifest = _fixture(tmp_path)
    (document_root / "unexpected.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the required files"):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )

    (document_root / "unexpected.txt").unlink()
    wrong_name = report_manifest.with_name("other.json")
    report_manifest.rename(wrong_name)
    with pytest.raises(ValueError, match=r"named report_manifest\.json"):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=wrong_name,
        )


def test_build_rejects_symlinked_document(tmp_path: Path) -> None:
    document_root, report_manifest = _fixture(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("replacement\n", encoding="utf-8")
    manuscript = document_root / "manuscript.tex"
    manuscript.unlink()
    manuscript.symlink_to(target)

    with pytest.raises(ValueError, match="without symlinks"):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )

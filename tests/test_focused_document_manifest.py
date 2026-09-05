"""Tests for the focused submission document-manifest producer."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from analysis import build_tcga_revision_focused_document_manifest as documents
from analysis import build_tcga_revision_focused_release as release

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_DOCUMENTS = {
    "Fig1.tif",
    "Fig2.tif",
    "S1_Table.csv",
    "S1_Table.pdf",
    "S1_Table.tex",
    "cover_letter.pdf",
    "manuscript.pdf",
    "manuscript.tex",
    "marked_manuscript.pdf",
    "rebuttal.md",
    "response_to_reviewers.pdf",
    "supporting_information.pdf",
    "supporting_information.tex",
}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    document_root = tmp_path / "documents"
    document_root.mkdir()
    table = (
        ",".join(release.reporting.report_csv_columns()["table_s5.csv"]) + "\n"
    ).encode()
    for name in release.REQUIRED_DOCUMENTS:
        if name in {"Fig1.tif", "Fig2.tif"}:
            content = b"II*\x00\x08\x00\x00\x00"
        elif name == "S1_Table.csv":
            content = table
        else:
            content = f"final {name}\n".encode()
        (document_root / name).write_bytes(content)
    report_table = tmp_path / "table_s5.csv"
    report_table.write_bytes(table)
    report_manifest = tmp_path / "report_manifest.json"
    report_manifest.write_text(
        json.dumps(
            {
                "outputs": {
                    "table_s5.csv": {
                        "path": "table_s5.csv",
                        "bytes": len(table),
                        "sha256": hashlib.sha256(table).hexdigest(),
                    },
                },
            },
        )
        + "\n",
        encoding="utf-8",
    )
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
    assert release.DOCUMENT_CONTRACT == "focused-submission-document-set-v2"
    assert value["contract"] == release.DOCUMENT_CONTRACT
    assert release.REQUIRED_DOCUMENTS == _EXPECTED_DOCUMENTS
    assert set(value["outputs"]) == _EXPECTED_DOCUMENTS
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


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("Fig1.tif", b"not-a-tiff", "bounded classic TIFF"),
        (
            "Fig2.tif",
            b"II*\x00TCGA-AB-1234",
            "TCGA sample barcode",
        ),
        (
            "S1_Table.csv",
            b"cohort,sample_id\nACC,TCGA-AB-1234\n",
            "privacy contract",
        ),
    ],
)
def test_build_rejects_unsafe_portal_artifacts(
    tmp_path: Path,
    name: str,
    content: bytes,
    message: str,
) -> None:
    document_root, report_manifest = _fixture(tmp_path)
    (document_root / name).write_bytes(content)

    with pytest.raises(ValueError, match=message):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )


def test_build_rejects_oversized_tiff(tmp_path: Path) -> None:
    document_root, report_manifest = _fixture(tmp_path)
    (document_root / "Fig1.tif").write_bytes(
        b"II*\x00" + b"0" * release._PORTAL_TIFF_LIMIT_BYTES,  # noqa: SLF001
    )

    with pytest.raises(ValueError, match="size ceiling"):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )


def test_build_binds_s1_table_to_report_bytes(tmp_path: Path) -> None:
    document_root, report_manifest = _fixture(tmp_path)
    with (document_root / "S1_Table.csv").open("ab") as handle:
        handle.write(b"ACC\n")

    with pytest.raises(ValueError, match="byte-identical"):
        documents.build_manifest(
            document_root=document_root,
            report_manifest=report_manifest,
        )

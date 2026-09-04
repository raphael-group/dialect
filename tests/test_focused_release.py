"""Tests for the focused immutable submission release."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from analysis import build_tcga_revision_focused_release as release

if TYPE_CHECKING:
    from pathlib import Path


def test_archive_is_deterministic_and_receipt_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    members = [
        release._bytes_member("README.md", b"release\n"),  # noqa: SLF001
        release._file_member("results/source.csv", source),  # noqa: SLF001
    ]
    commit = "a" * 40
    manifest = release._manifest(members, commit)  # noqa: SLF001
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    release._write_archive(first, members, manifest)  # noqa: SLF001
    release._write_archive(second, members, manifest)  # noqa: SLF001

    assert first.read_bytes() == second.read_bytes()
    assert release.verify_archive(first)["source_commit"] == commit

    receipt = {
        "schema_version": release.SCHEMA_VERSION,
        "contract": release.RECEIPT_CONTRACT,
        "archive": {
            "path": first.name,
            "bytes": first.stat().st_size,
            "sha256": release._sha256_path(first),  # noqa: SLF001
        },
        "release_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "source_commit": commit,
        "member_count": len(members),
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(release._canonical_json(receipt) + b"\n")  # noqa: SLF001
    assert release.verify_release(first, receipt_path)["source_commit"] == commit


def test_archive_verifier_rejects_payload_drift(tmp_path: Path) -> None:
    original = [release._bytes_member("data.txt", b"original\n")]  # noqa: SLF001
    manifest = release._manifest(original, "b" * 40)  # noqa: SLF001
    changed = [release._bytes_member("data.txt", b"changed\n")]  # noqa: SLF001
    archive = tmp_path / "changed.tar.gz"
    release._write_archive(archive, changed, manifest)  # noqa: SLF001
    with pytest.raises(ValueError, match="digest differs"):
        release.verify_archive(archive)


def test_document_plan_requires_final_submission_files(tmp_path: Path) -> None:
    (tmp_path / "manuscript.pdf").write_bytes(b"%PDF-manuscript")
    (tmp_path / "optional_figure.tif").write_bytes(b"TIFF")
    with pytest.raises(ValueError, match="missing required files"):
        release._document_members(tmp_path)  # noqa: SLF001


def test_document_plan_includes_exact_decision_letter_artifacts(tmp_path: Path) -> None:
    for name in release.REQUIRED_DOCUMENTS:
        (tmp_path / name).write_text(f"final {name}\n", encoding="utf-8")

    members = release._document_members(tmp_path)  # noqa: SLF001

    assert {member.name for member in members} == {
        f"documents/{name}" for name in release.REQUIRED_DOCUMENTS
    }

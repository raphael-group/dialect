from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import scripts.download_cbioportal_mafs as downloader


def test_study_url_is_bound_to_the_pinned_commit():
    assert downloader.study_url("CRAD") == (
        "https://media.githubusercontent.com/media/cBioPortal/datahub/"
        "64392efc82b38655f67188a4e95e44ca22e030c0/"
        "public/coadread_tcga_pan_can_atlas_2018/data_mutations.txt"
    )


def test_maf_validation_requires_header_size_and_exact_digest(tmp_path, monkeypatch):
    maf = tmp_path / "CHOL.maf"
    payload = b"Hugo_Symbol\tTumor_Sample_Barcode\nIDH1\tTCGA-AA-0001-01\n"
    maf.write_bytes(payload)
    monkeypatch.setattr(downloader, "MIN_MAF_BYTES", 1)
    monkeypatch.setattr(
        downloader,
        "TCGA_MAF_SHA256",
        {"CHOL": hashlib.sha256(payload).hexdigest()},
    )

    assert downloader.maf_validation_error(maf, "CHOL") is None

    maf.write_bytes(b"not-a-maf\n")
    assert "Hugo_Symbol" in downloader.maf_validation_error(maf, "CHOL")


def test_maf_validation_returns_clean_reason_for_invalid_utf8(tmp_path, monkeypatch):
    maf = tmp_path / "CHOL.maf"
    maf.write_bytes(b"Hugo_Symbol\xff\n")
    monkeypatch.setattr(downloader, "MIN_MAF_BYTES", 1)

    assert downloader.maf_validation_error(maf, "CHOL") == (
        "file is not valid UTF-8"
    )


def test_download_refuses_to_overwrite_a_mismatched_existing_file(
    tmp_path,
    monkeypatch,
):
    payload = b"Hugo_Symbol\nexisting\n"
    out = tmp_path / "CHOL.maf"
    out.write_bytes(payload)
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setattr(downloader, "MIN_MAF_BYTES", 1)
    monkeypatch.setattr(downloader, "TCGA_MAF_SHA256", {"CHOL": "0" * 64})

    run_mock = Mock()
    monkeypatch.setattr(downloader.subprocess, "run", run_mock)

    cohort, status = downloader.download("CHOL")

    assert cohort == "CHOL"
    assert status.startswith("FAIL (existing file refused: SHA-256 mismatch")
    assert out.read_bytes() == payload
    run_mock.assert_not_called()


def test_download_validates_then_atomically_installs_pinned_bytes(
    tmp_path,
    monkeypatch,
):
    payload = b"Hugo_Symbol\tTumor_Sample_Barcode\nIDH1\tTCGA-AA-0001-01\n"
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setattr(downloader, "MIN_MAF_BYTES", 1)
    monkeypatch.setattr(
        downloader,
        "TCGA_MAF_SHA256",
        {"CHOL": hashlib.sha256(payload).hexdigest()},
    )

    def fake_run(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    cohort, status = downloader.download("CHOL")

    assert cohort == "CHOL"
    assert status.startswith("ok (pinned")
    assert (tmp_path / "CHOL.maf").read_bytes() == payload
    assert list(tmp_path.glob("*.partial")) == []


def test_download_never_overwrites_destination_that_appears_mid_download(
    tmp_path,
    monkeypatch,
):
    payload = b"Hugo_Symbol\tTumor_Sample_Barcode\nIDH1\tTCGA-AA-0001-01\n"
    competing_payload = b"created-by-another-process\n"
    destination = tmp_path / "CHOL.maf"
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setattr(downloader, "MIN_MAF_BYTES", 1)
    monkeypatch.setattr(
        downloader,
        "TCGA_MAF_SHA256",
        {"CHOL": hashlib.sha256(payload).hexdigest()},
    )

    def fake_run(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(payload)
        destination.write_bytes(competing_payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    cohort, status = downloader.download("CHOL")

    assert cohort == "CHOL"
    assert status == "FAIL (destination appeared during download; refused)"
    assert destination.read_bytes() == competing_payload
    assert list(tmp_path.glob("*.partial")) == []


def test_main_returns_nonzero_when_any_frozen_download_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_cbioportal_mafs.py", "CHOL"])
    monkeypatch.setattr(
        downloader,
        "download",
        lambda cohort: (cohort, "FAIL (test receipt mismatch)"),
    )

    with pytest.raises(SystemExit) as error:
        downloader.main()

    assert error.value.code == 1

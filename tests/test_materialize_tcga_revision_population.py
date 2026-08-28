from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest

import dialect.data.tcga as tcga_module
from analysis import materialize_tcga_revision_population as materializer
from dialect.data.tcga import TCGACaseListReceipt, tcga_datahub_case_list_path

GIT = shutil.which("git")


def _case_list(sample_ids: tuple[str, ...]) -> bytes:
    joined = "\t".join(sample_ids)
    description = f"Samples with mutation data ({len(sample_ids)} samples)"
    return (
        "cancer_study_identifier: chol_tcga_pan_can_atlas_2018\n"
        "stable_id: chol_tcga_pan_can_atlas_2018_sequenced\n"
        "case_list_name: Samples with mutation data\n"
        f"case_list_description: {description}\n"
        "case_list_category: all_cases_with_mutation_data\n"
        f"case_list_ids: {joined}\n"
    ).encode()


def _sequence_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _git(command: list[str], *, cwd: Path) -> str:
    assert GIT is not None
    return subprocess.run(  # noqa: S603
        [GIT, *command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepare_source(tmp_path: Path, monkeypatch) -> tuple[Path, bytes, tuple[str, ...]]:
    repo = tmp_path / "datahub"
    repo.mkdir()
    _git(["init", "-q"], cwd=repo)
    _git(["config", "user.name", "test"], cwd=repo)
    _git(["config", "user.email", "test@example.com"], cwd=repo)
    sample_ids = (
        "TCGA-ZZ-0002-06",
        "TCGA-AA-0001-02",
        "TCGA-ZZ-0002-01",
    )
    selected = ("TCGA-AA-0001-02", "TCGA-ZZ-0002-01")
    content = _case_list(sample_ids)
    source = repo / tcga_datahub_case_list_path("CHOL")
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    _git(["add", "."], cwd=repo)
    _git(["commit", "-qm", "fixture"], cwd=repo)
    commit = _git(["rev-parse", "HEAD"], cwd=repo)
    tree = _git(["rev-parse", "HEAD^{tree}"], cwd=repo)
    monkeypatch.setattr(materializer, "TCGA_DATAHUB_COMMIT", commit)
    monkeypatch.setattr(materializer, "TCGA_DATAHUB_TREE", tree)
    receipt = TCGACaseListReceipt(
        hashlib.sha256(content).hexdigest(),
        participant_count=2,
        sample_count=3,
    )
    receipts = MappingProxyType({"CHOL": receipt})
    axis_receipts = MappingProxyType({"CHOL": _sequence_sha256(selected)})
    monkeypatch.setattr(materializer, "TCGA_CASE_LIST_RECEIPTS", receipts)
    monkeypatch.setattr(
        materializer,
        "TCGA_SELECTED_SAMPLE_AXIS_SHA256",
        axis_receipts,
    )
    monkeypatch.setattr(tcga_module, "TCGA_CASE_LIST_RECEIPTS", receipts)
    monkeypatch.setattr(
        tcga_module,
        "TCGA_SELECTED_SAMPLE_AXIS_SHA256",
        axis_receipts,
    )
    return repo / ".git", content, selected


def test_materializer_publishes_exact_axis_and_aggregate_only_manifests(
    tmp_path,
    monkeypatch,
):
    git_dir, _content, selected = _prepare_source(tmp_path, monkeypatch)
    output = tmp_path / "population"

    result = materializer.materialize_tcga_revision_population(
        git_dir,
        output,
        cohorts=["CHOL"],
    )

    assert result == output
    assert (output / "CHOL" / "sample_axis.txt").read_text().splitlines() == list(
        selected,
    )
    cohort_manifest_text = (output / "CHOL" / "population_manifest.json").read_text()
    root_manifest_text = (output / "population_manifest.json").read_text()
    assert "TCGA-" not in cohort_manifest_text
    assert "TCGA-" not in root_manifest_text
    cohort_manifest = json.loads(cohort_manifest_text)
    root_manifest = json.loads(root_manifest_text)
    assert cohort_manifest["population"]["source_sample_count"] == 3
    assert cohort_manifest["population"]["selected_sample_count"] == 2
    assert cohort_manifest["population"]["removed_repeat_participant_samples"] == 1
    expected_policy = {
        "analysis_unit": "one-participant-one-tumor-sample",
        "membership_source": "commit-matched-sequenced-case-list",
        "ordered_axis_digest": "sha256-uint64be-length-framed-utf8-v1",
        "ordering": "lexicographic-sample-barcode",
        "primary_sample_type_codes": ["01", "03", "09"],
        "repeated_participant_rule": (
            "retain-exactly-one-primary-disease-sample-otherwise-fail-closed"
        ),
        "singleton_rule": (
            "retain-sole-case-list-sample-regardless-of-sample-type"
        ),
    }
    assert cohort_manifest["selection_policy"] == expected_policy
    assert root_manifest["selection_policy"] == expected_policy
    assert "git_dir_basename" not in root_manifest["source"]
    tcga_source = Path(tcga_module.__file__).resolve()
    tcga_source_sha256 = hashlib.sha256(tcga_source.read_bytes()).hexdigest()
    expected_contract_source = {
        "bytes": tcga_source.stat().st_size,
        "path": "src/dialect/data/tcga.py",
        "sha256": tcga_source_sha256,
    }
    assert cohort_manifest["contract_source"] == expected_contract_source
    assert root_manifest["contract_source"] == expected_contract_source
    assert root_manifest["totals"] == {
        "participant_count": 2,
        "removed_repeat_participant_samples": 1,
        "selected_sample_count": 2,
        "source_sample_count": 3,
    }
    assert list(tmp_path.glob(".population.publish-claim")) == []


def test_materializer_refuses_output_reuse(tmp_path, monkeypatch):
    git_dir, _content, _selected = _prepare_source(tmp_path, monkeypatch)
    output = tmp_path / "population"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        materializer.materialize_tcga_revision_population(
            git_dir,
            output,
            cohorts=["CHOL"],
        )

    assert list(tmp_path.glob(".population.publish-claim")) == []


def test_materializer_refuses_a_broken_output_symlink(tmp_path, monkeypatch):
    git_dir, _content, _selected = _prepare_source(tmp_path, monkeypatch)
    missing_target = tmp_path / "missing-target"
    output = tmp_path / "population"
    output.symlink_to(missing_target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        materializer.materialize_tcga_revision_population(
            git_dir,
            output,
            cohorts=["CHOL"],
        )

    assert output.is_symlink()
    assert not missing_target.exists()
    assert list(tmp_path.glob(".population.publish-claim")) == []


def test_materializer_rejects_tree_mismatch_before_staging(tmp_path, monkeypatch):
    git_dir, _content, _selected = _prepare_source(tmp_path, monkeypatch)
    monkeypatch.setattr(materializer, "TCGA_DATAHUB_TREE", "0" * 40)
    output = tmp_path / "population"

    with pytest.raises(ValueError, match="commit/tree pair"):
        materializer.materialize_tcga_revision_population(
            git_dir,
            output,
            cohorts=["CHOL"],
        )

    assert not output.exists()
    assert list(tmp_path.glob(".population.staging-*")) == []


def test_materializer_cleans_staging_after_blob_receipt_failure(tmp_path, monkeypatch):
    git_dir, content, _selected = _prepare_source(tmp_path, monkeypatch)
    bad_receipts = MappingProxyType(
        {
            "CHOL": TCGACaseListReceipt(
                hashlib.sha256(content + b"changed").hexdigest(),
                participant_count=2,
                sample_count=3,
            ),
        },
    )
    monkeypatch.setattr(tcga_module, "TCGA_CASE_LIST_RECEIPTS", bad_receipts)
    output = tmp_path / "population"

    with pytest.raises(ValueError, match="case-list SHA-256 mismatch"):
        materializer.materialize_tcga_revision_population(
            git_dir,
            output,
            cohorts=["CHOL"],
        )

    assert not output.exists()
    assert list(tmp_path.glob(".population.staging-*")) == []
    assert list(tmp_path.glob(".population.publish-claim")) == []


def test_materializer_refuses_an_existing_publish_claim(tmp_path, monkeypatch):
    git_dir, _content, _selected = _prepare_source(tmp_path, monkeypatch)
    output = tmp_path / "population"
    claim = tmp_path / ".population.publish-claim"
    claim.write_bytes(b"")

    with pytest.raises(FileExistsError, match="holds the claim"):
        materializer.materialize_tcga_revision_population(
            git_dir,
            output,
            cohorts=["CHOL"],
        )

    assert not output.exists()
    assert claim.exists()
    assert list(tmp_path.glob(".population.staging-*")) == []

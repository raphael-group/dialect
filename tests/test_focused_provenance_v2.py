"""Tests for the focused revision provenance-v2 boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from analysis import focused_revision_provenance as provenance
from analysis import prepare_tcga_revision_focused as preparation
from analysis import run_tcga_revision_focused as runner

if TYPE_CHECKING:
    from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_release_pipeline_inventory_binds_every_calibration_dependency() -> None:
    expected = (
        provenance.Path("analysis/build_tcga_revision_focused_document_manifest.py"),
        provenance.Path("analysis/build_tcga_revision_focused_release.py"),
        provenance.Path("analysis/calibrate_tcga_revision_focused.py"),
        provenance.Path("analysis/calibration_batch.py"),
        provenance.Path("analysis/diagnose_tcga_revision_focused.py"),
        provenance.Path("analysis/focused_revision_provenance.py"),
        provenance.Path("analysis/freeze_tcga_revision_reporting_rule.py"),
        provenance.Path("analysis/postprocess_tcga_revision_focused.py"),
        provenance.Path("analysis/report_tcga_revision_focused.py"),
        provenance.Path("analysis/tcga_revision_calibration_config.json"),
    )
    assert expected == provenance.RELEASE_PIPELINE_FILES


def _raw_chain_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "input"
    provider_root = tmp_path / "provider"
    run_root = tmp_path / "run"
    input_root.mkdir()
    provider_root.mkdir()
    (run_root / "contracts").mkdir(parents=True)

    input_path = input_root / "input_manifest.json"
    input_path.write_text('{"contract":"input"}\n', encoding="utf-8")
    provider_manifest = {
        "input_manifest": _record(input_path, relative_to=input_root),
    }
    provider_path = provider_root / "provider_manifest.json"
    provider_path.write_text(json.dumps(provider_manifest) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        preparation,
        "validate_input_root",
        lambda *_args: {"contract": "input"},
    )
    monkeypatch.setattr(
        preparation,
        "validate_provider_root",
        lambda *_args: provider_manifest,
    )

    run_manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "contract": runner.RUN_CONTRACT,
        "cohorts": ["CHOL"],
        "providers": ["cbase", "dig", "mutsig"],
        "top_k": 500,
        "config_sha256": runner._sha256(runner.CONFIG_PATH),  # noqa: SLF001
        "provider_manifest": _record(provider_path, relative_to=provider_root),
    }
    run_path = run_root / "run_manifest.json"
    run_path.write_text(json.dumps(run_manifest) + "\n", encoding="utf-8")
    contract = {
        "cohort": "CHOL",
        "top_k": 500,
        "focused_config_sha256": run_manifest["config_sha256"],
        "features": [f"G{index}_M" for index in range(500)],
        "pair_policy": {"row_count": 1},
    }
    contract_path = run_root / "contracts" / "CHOL.json"
    contract_path.write_bytes(provenance._canonical_json(contract) + b"\n")  # noqa: SLF001

    completion_tasks = []
    manifests = {}
    for provider in ("cbase", "dig", "mutsig"):
        task_root = run_root / "tasks" / "CHOL" / provider
        task_root.mkdir(parents=True)
        manifest = {
            "outputs": {
                "pairwise_interaction_results.csv": {
                    "path": "pairwise_interaction_results.csv",
                    "bytes": 10,
                    "sha256": "a" * 64,
                },
                "single_gene_results.csv": {
                    "path": "single_gene_results.csv",
                    "bytes": 5,
                    "sha256": "b" * 64,
                },
            },
        }
        manifest_path = task_root / "task_manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        manifests[provider] = manifest
        completion_tasks.append(
            {
                "cohort": "CHOL",
                "provider": provider,
                "manifest": _record(manifest_path, relative_to=run_root),
            },
        )

    def validate_task(task_root: Path, **_kwargs):
        return manifests[task_root.name]

    monkeypatch.setattr(runner, "_validate_completed_task", validate_task)
    completion = {
        "schema_version": runner.SCHEMA_VERSION,
        "contract": runner.COMPLETION_CONTRACT,
        "cohorts": ["CHOL"],
        "task_count": 3,
        "run_manifest": _record(run_path, relative_to=run_root),
        "tasks": completion_tasks,
    }
    (run_root / "completion_manifest.json").write_text(
        json.dumps(completion) + "\n",
        encoding="utf-8",
    )
    return input_root, provider_root, run_root


def test_raw_chain_binds_input_provider_run_completion_and_exact_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root, provider_root, run_root = _raw_chain_fixture(tmp_path, monkeypatch)

    evidence = provenance.validate_raw_chain(
        input_root=input_root,
        provider_root=provider_root,
        run_root=run_root,
        cohorts=("CHOL",),
    )

    assert len(evidence["cohort_contracts"]) == 1
    assert len(evidence["task_manifests"]) == 3
    assert evidence["completion_manifest"]["path"] == "completion_manifest.json"

    completion_path = run_root / "completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["tasks"].pop()
    completion["task_count"] = 2
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact cohort/provider grid"):
        provenance.validate_raw_chain(
            input_root=input_root,
            provider_root=provider_root,
            run_root=run_root,
            cohorts=("CHOL",),
        )


@pytest.mark.parametrize(
    "content",
    [
        b'{"contract":"first","contract":"second"}\n',
        b'{"contract":NaN}\n',
        b'{"contract":Infinity}\n',
    ],
)
def test_source_json_loader_rejects_ambiguous_extensions(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(content)

    with pytest.raises(ValueError, match=r"duplicate key|numeric constant"):
        provenance._load_json(path)  # noqa: SLF001


def test_public_contract_removes_host_identity_but_preserves_source_digests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "CHOL.json"
    contract = {
        "cohort": "CHOL",
        "samples": {"count": 36, "ordered_ids_sha256": "a" * 64},
        "inputs": {
            "counts": {
                "path": "/Users/private/run/count_matrix.csv",
                "bytes": 100,
                "sha256": "b" * 64,
                "device": 8,
                "inode": 99,
                "mtime_ns": 123,
            },
        },
        "provider": {
            "root": "/private/provider-root",
            "canonical_maf_path": "/Volumes/restricted/CHOL.maf",
        },
    }
    path.write_bytes(provenance._canonical_json(contract) + b"\n")  # noqa: SLF001

    public = provenance.public_cohort_contract(path)
    serialized = provenance._canonical_json(public)  # noqa: SLF001

    assert public["source_contract"]["sha256"] == _sha256(path)
    assert (
        public["source_contract"]["canonical_sha256"]
        == hashlib.sha256(
            provenance._canonical_json(contract),  # noqa: SLF001
        ).hexdigest()
    )
    assert b"/Users/" not in serialized
    assert b"/private/provider-root" not in serialized
    assert b"/Volumes/restricted" not in serialized
    assert b"inode" not in serialized
    assert public["projection"]["inputs"]["counts"]["path"].startswith(
        "input-record://",
    )


def test_public_contract_rejects_literal_tcga_sample_identifier(tmp_path: Path) -> None:
    path = tmp_path / "CHOL.json"
    path.write_text(
        json.dumps({"cohort": "CHOL", "sample_ids": ["TCGA-AA-1234-01A"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identifier axis"):
        provenance.public_cohort_contract(path)


def test_runtime_record_is_bound_to_executing_python(tmp_path: Path) -> None:
    other_python = tmp_path / "python"
    other_python.write_bytes(b"not-the-running-interpreter")

    with pytest.raises(ValueError, match="supplied Python binary"):
        provenance._runtime_record(other_python)  # noqa: SLF001

    runtime = provenance._runtime_record(  # noqa: SLF001
        provenance.Path(sys.executable),
    )
    assert runtime["python"]["sha256"] == _sha256(
        provenance.Path(sys.executable).resolve(),
    )


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["/usr/bin/git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_source_boundary_rejects_fit_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "test")
    _git(repository, "config", "user.email", "test@example.com")
    fit_source = repository / "fit.py"
    fit_source.write_text("FIT = 1\n", encoding="utf-8")
    _git(repository, "add", "fit.py")
    _git(repository, "commit", "-m", "fit")
    fit_commit = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(provenance, "FIT_SOURCE_FILES", (provenance.Path("fit.py"),))
    monkeypatch.setattr(provenance, "RELEASE_PIPELINE_FILES", ())

    boundary = provenance._source_boundary(  # noqa: SLF001
        repository_root=repository,
        fit_commit=fit_commit,
        release_commit=fit_commit,
    )
    assert boundary["raw_fit_sources_unchanged_at_release"] is True

    untracked = repository / "untracked.txt"
    untracked.write_text("not attested\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean HEAD"):
        provenance._source_boundary(  # noqa: SLF001
            repository_root=repository,
            fit_commit=fit_commit,
            release_commit=fit_commit,
        )
    untracked.unlink()

    fit_source.write_text("FIT = 2\n", encoding="utf-8")
    _git(repository, "add", "fit.py")
    _git(repository, "commit", "-m", "release")
    release_commit = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="differ between fit and release"):
        provenance._source_boundary(  # noqa: SLF001
            repository_root=repository,
            fit_commit=fit_commit,
            release_commit=release_commit,
        )


def test_attestation_validator_rejects_nonproduction_fit_commit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=provenance.PRODUCTION_FIT_COMMIT):
        provenance.validate_fit_attestation(
            tmp_path / "missing.json",
            repository_root=tmp_path,
            input_root=tmp_path,
            provider_root=tmp_path,
            run_root=tmp_path,
            cohorts=("CHOL",),
            fit_commit="a" * 40,
            release_commit="b" * 40,
            runtime_executable=provenance.Path(sys.executable),
        )

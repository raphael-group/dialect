"""Tests for the result-blind K=500 authority projection."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from analysis import build_tcga_revision_k500_authority_projection as projection

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ProjectionFixture:
    """Synthetic Git and metadata boundary used by projection tests."""

    repo: Path
    git: Path
    source_a: str
    release_b: str
    release_tag: str
    attestation: Path
    seal: Path
    run: Path
    output: Path
    attestation_sha256: str
    seal_sha256: str
    run_sha256: str
    authority_digests: dict[str, str]

    def build(self, **overrides: Any) -> projection.K500AuthorityProjectionReceipt:
        """Build with all independently anchored fixture values."""
        arguments: dict[str, Any] = {
            "completion_attestation_path": self.attestation,
            "expected_completion_attestation_sha256": self.attestation_sha256,
            "sealed_completion_path": self.seal,
            "expected_sealed_completion_sha256": self.seal_sha256,
            "run_manifest_path": self.run,
            "expected_run_manifest_sha256": self.run_sha256,
            "repo_root": self.repo,
            "release_b_commit": self.release_b,
            "release_tag": self.release_tag,
            "output_path": self.output,
            "git_executable": self.git,
        }
        arguments.update(overrides)
        return projection.build_k500_authority_projection(**arguments)


def _git(repo: Path, git: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [git.as_posix(), *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: object) -> bytes:
    raw = projection._canonical_json(value) + b"\n"  # noqa: SLF001
    path.write_bytes(raw)
    return raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _descriptor_count() -> int | None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.is_dir():
        return None
    return len(list(descriptor_root.iterdir()))


def _make_sparse(path: Path, *, size: int, mode: int = 0o600) -> None:
    with path.open("wb") as stream:
        stream.truncate(size)
    path.chmod(mode)


def _add_synthetic_submodule(
    fixture: ProjectionFixture,
    tmp_path: Path,
    *,
    release_tag: str,
) -> tuple[Path, str]:
    submodule_source = tmp_path / f"{release_tag}-source"
    submodule_source.mkdir()
    _git(submodule_source, fixture.git, "init", "-q")
    _git(
        submodule_source,
        fixture.git,
        "config",
        "user.email",
        "synthetic@example.test",
    )
    _git(
        submodule_source,
        fixture.git,
        "config",
        "user.name",
        "Synthetic Test",
    )
    tracked = submodule_source / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    _git(submodule_source, fixture.git, "add", ".")
    _git(submodule_source, fixture.git, "commit", "-q", "-m", "frozen")
    submodule = fixture.repo / "vendor/synthetic-submodule"
    _git(
        fixture.repo,
        fixture.git,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        submodule_source.as_posix(),
        "vendor/synthetic-submodule",
    )
    _git(fixture.repo, fixture.git, "add", ".")
    _git(
        fixture.repo,
        fixture.git,
        "commit",
        "-q",
        "-m",
        "add synthetic submodule",
    )
    release_b = _git(fixture.repo, fixture.git, "rev-parse", "HEAD")
    _git(fixture.repo, fixture.git, "tag", release_tag)
    return submodule, release_b


def _rewrite_metadata_chain(
    fixture: ProjectionFixture,
    *,
    mutate_run: Callable[[dict[str, Any]], None] | None = None,
    mutate_seal: Callable[[dict[str, Any]], None] | None = None,
    mutate_attestation: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, str, str]:
    run = json.loads(fixture.run.read_bytes())
    if mutate_run is not None:
        mutate_run(run)
    run_raw = _write_json(fixture.run, run)

    seal = json.loads(fixture.seal.read_bytes())
    seal["authority"] = run["revision_authority"]
    seal["run_manifest"] = {"bytes": len(run_raw), "sha256": _sha256(run_raw)}
    if mutate_seal is not None:
        mutate_seal(seal)
    seal_raw = _write_json(fixture.seal, seal)

    attestation = json.loads(fixture.attestation.read_bytes())
    unsigned = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_payload_sha256"
    }
    unsigned["frozen_run"]["run_manifest_sha256"] = _sha256(run_raw)
    unsigned["sealed_completion"] = {
        "bytes": len(seal_raw),
        "path": "sealed_completion_manifest.json",
        "sha256": _sha256(seal_raw),
    }
    for record in unsigned["pre_attestation_inventory"]["files"]:
        if record["path"] == "sealed_completion_manifest.json":
            record["bytes"] = len(seal_raw)
            record["sha256"] = _sha256(seal_raw)
        elif record["path"] == "run_manifest.json":
            record["bytes"] = len(run_raw)
            record["sha256"] = _sha256(run_raw)
    inventory = unsigned["pre_attestation_inventory"]
    inventory["total_bytes"] = sum(record["bytes"] for record in inventory["files"])
    inventory["records_sha256"] = projection._json_sha256(  # noqa: SLF001
        inventory["files"],
    )
    if mutate_attestation is not None:
        mutate_attestation(unsigned)
    attestation_raw = _write_json(
        fixture.attestation,
        {
            **unsigned,
            "attestation_payload_sha256": projection._json_sha256(  # noqa: SLF001
                unsigned,
            ),
        },
    )
    return _sha256(attestation_raw), _sha256(seal_raw), _sha256(run_raw)


def _source_snapshot(source_files: dict[str, bytes]) -> dict[str, str]:
    return {
        **{name: _sha256(raw) for name, raw in source_files.items()},
        projection.GENERATED_VERSION_PATH: projection.GENERATED_VERSION_SHA256,
    }


def _authority(git: Path) -> tuple[dict[str, Any], dict[str, str]]:
    git_raw = git.read_bytes()
    git_receipt = {
        "bytes": len(git_raw),
        "path": git.as_posix(),
        "sha256": _sha256(git_raw),
    }
    provider_manifest_sha256 = "d" * 64
    full_acceptance = {
        "association_outputs_opened": False,
        "authority_sha256": "a" * 64,
        "cohort_receipts_sha256": "b" * 64,
        "contract": projection.PROVIDER_FULL_ACCEPTANCE_CONTRACT,
        "execution_snapshot": {
            "directory_count": 2,
            "file_count": 3,
            "individual_file_receipts_omitted": True,
            "root": f"_orchestration/execution-snapshot-{'c' * 64}",
            "tree_hash_contract": projection.PROVIDER_TREE_HASH_CONTRACT,
            "tree_sha256": "c" * 64,
        },
        "full_inventory_validated": True,
        "provider_manifest_sha256": provider_manifest_sha256,
        "schema_version": projection.PROVIDER_SCHEMA_VERSION,
    }
    full_acceptance_sha256 = projection._json_sha256(  # noqa: SLF001
        full_acceptance,
        newline=True,
    )
    authority = {
        "canonical_input_root": "/synthetic/canonical",
        "configured": True,
        "expected_canonical_input_sha256": "1" * 64,
        "expected_fit_approval_sha256": "3" * 64,
        "expected_input_approval_sha256": "2" * 64,
        "fit_approval_manifest": "/synthetic/fit/approval.json",
        "input_approval_manifest": "/synthetic/materialize/approval.json",
        "provider_input": {
            "association_outputs_opened": False,
            "cohort_provider_receipts_sha256": "b" * 64,
            "contract": projection.PROVIDER_INPUT_CONTRACT,
            "expected_manifest_sha256": provider_manifest_sha256,
            "full_acceptance_receipt": full_acceptance,
            "full_acceptance_receipt_sha256": full_acceptance_sha256,
            "git_executable": git_receipt,
            "manifest": {
                "bytes": 123,
                "path": "/synthetic/provider/provider_input_manifest.json",
                "sha256": provider_manifest_sha256,
            },
            "root": "/synthetic/provider",
        },
    }
    digests = {
        "canonical_input_manifest_sha256": "1" * 64,
        "fit_approval_sha256": "3" * 64,
        "materialization_approval_sha256": "2" * 64,
        "provider_full_acceptance_receipt_sha256": full_acceptance_sha256,
        "provider_input_manifest_sha256": provider_manifest_sha256,
        "validated_run_authority_sha256": projection._json_sha256(  # noqa: SLF001
            authority,
        ),
    }
    return authority, digests


def _run_manifest(
    *,
    source_a: str,
    snapshot: dict[str, str],
    authority: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": "tcga-revision-k500",
        "bmrs": list(projection.BMRS),
        "cohorts": list(projection.TCGA_COHORTS),
        "created_at_utc": "2026-08-30T00:00:00+00:00",
        "feature_policy": "descending-total-eligible-mutation-event-count",
        "git": {
            "dirty": False,
            "executable": dict(authority["provider_input"]["git_executable"]),
            "head": source_a,
            "status": [],
            "version": "git version synthetic",
        },
        "implementation_sha256": snapshot,
        "mutsig_root": "/synthetic/provider/mutsig",
        "observation_support_universe": "full-observation-support-common-universe-v1",
        "required_contingency_table_contract": ("observed-binary-cells-00-01-10-11-v1"),
        "required_gene_support_contract": "latent-state-union-v1",
        "required_log_odds_ratio_contract": (
            "conventional-latent-odds-00x11-over-01x10-identifiable-v2"
        ),
        "required_lrt_contract": "driver-independence-constrained-mle-v1",
        "required_lrt_nestedness_tolerance": 1e-8,
        "required_output_recomputation_atol": 1e-12,
        "required_pair_effect_identifiability_contract": (
            "full-affine-rank-relative-svd-1e-12-conservative-v1"
        ),
        "required_pair_fit_contract": (
            "deterministic-simplex-coordinate-ascent-total-kkt-v2"
        ),
        "required_pair_fit_kkt_tolerance": 1e-8,
        "required_pair_fit_max_iterations": 1000,
        "required_pair_identifiability_relative_tolerance": 1e-12,
        "required_pair_simplex_tolerance": 1e-12,
        "required_rho_contract": (
            "marshall-olkin-identifiable-finite-or-degenerate-null-v2"
        ),
        "resource_policy": {"jobs": 2},
        "revision_authority": authority,
        "same_base_pair_policy": "exclude-before-fitting-and-testing",
        "sample_axis_contract": (
            "count-matrix-equals-authoritative-and-mutsig-patient-axis-v2"
        ),
        "schema_version": "3.0.0",
        "signed_tested_family": dict(projection.EXPECTED_TESTED_FAMILY),
        "source_root": "/synthetic/provider/cohorts",
        "tested_family_implementation": dict(projection.EXPECTED_TESTED_FAMILY),
        "top_k": 500,
        "undefined_rho_lrt_tolerance": 1e-8,
    }


def _seal(
    *,
    authority: dict[str, Any],
    run_raw: bytes,
) -> dict[str, Any]:
    coordinates = tuple(
        f"{cohort}/{bmr}"
        for cohort in projection.TCGA_COHORTS
        for bmr in projection.BMRS
    )
    return {
        "analysis": "tcga-revision-k500",
        "authority": authority,
        "bmrs": list(projection.BMRS),
        "cohorts": list(projection.TCGA_COHORTS),
        "contract": projection.SEALED_COMPLETION_CONTRACT,
        "contracts": [
            {
                "bytes": 1,
                "cohort": cohort,
                "contract_sha256": "4" * 64,
                "file_sha256": "5" * 64,
            }
            for cohort in projection.TCGA_COHORTS
        ],
        "downstream_binding": {
            "field": "upstream_result_manifest_sha256",
            "stage": "inspect-tcga-k500",
        },
        "grid": {
            "ordered_coordinates_sha256": projection._sequence_sha256(  # noqa: SLF001
                coordinates,
            ),
            "task_count": len(coordinates),
        },
        "result_rows_opened": False,
        "run_manifest": {"bytes": len(run_raw), "sha256": _sha256(run_raw)},
        "schema": projection.SEALED_COMPLETION_SCHEMA,
        "tasks": [
            {
                "bmr": bmr,
                "cohort": cohort,
                "consumed_input_sha256": {"counts": "6" * 64},
                "contract_sha256": "4" * 64,
                "pairwise_interaction_results": {
                    "bytes": 1,
                    "sha256": "7" * 64,
                },
                "single_gene_results": {"bytes": 1, "sha256": "8" * 64},
                "task_manifest": {"bytes": 1, "sha256": "9" * 64},
            }
            for cohort in projection.TCGA_COHORTS
            for bmr in projection.BMRS
        ],
        "top_k": 500,
    }


def _attestation(
    *,
    source_a: str,
    snapshot: dict[str, str],
    run_raw: bytes,
    seal_raw: bytes,
) -> dict[str, Any]:
    seal = json.loads(seal_raw)
    contracts = [
        {
            "cohort": cohort,
            "contract_path": f"contracts/{cohort}.json",
            "contract_sha256": "4" * 64,
            "features": 500,
            "ordered_features_sha256": "a" * 64,
            "ordered_pair_sha256": "b" * 64,
            "pairs_per_background": 124_749,
            "same_base_pairs_excluded": 1,
            "samples": 1,
        }
        for cohort in projection.TCGA_COHORTS
    ]
    resource_usage = {
        "elapsed_seconds": 1.0,
        "peak_rss": {
            "bytes": 1024,
            "native_unit": "bytes",
            "native_value": 1024,
            "platform": "darwin",
            "source": projection.PEAK_RSS_SOURCE,
        },
    }
    tasks = [
        {
            "bmr": bmr,
            "cohort": cohort,
            "completed_at_utc": "2026-08-30T00:30:00+00:00",
            "contract_sha256": "4" * 64,
            "resource_usage": resource_usage,
            "task_manifest_path": f"tasks/{cohort}/{bmr}/task_manifest.json",
            "task_manifest_sha256": "9" * 64,
            "validation": {
                "features": 500,
                "ordered_features_sha256": "a" * 64,
                "ordered_pair_sha256": "b" * 64,
                "pairs": 124_749,
                "pairwise_sha256": "7" * 64,
                "single_gene_sha256": "8" * 64,
            },
        }
        for cohort in projection.TCGA_COHORTS
        for bmr in projection.BMRS
    ]
    inventory_receipts = projection._known_inventory_receipts(  # noqa: SLF001
        seal,
        sealed_sha256=_sha256(seal_raw),
        sealed_bytes=len(seal_raw),
        run_sha256=_sha256(run_raw),
        run_bytes=len(run_raw),
    )
    inventory_files = [
        {
            **receipt,
            "mtime_ns": 1,
            "path": path,
        }
        for path, receipt in sorted(inventory_receipts.items())
    ]
    task_count = len(projection.TCGA_COHORTS) * len(projection.BMRS)
    pairs_per_background = sum(record["pairs_per_background"] for record in contracts)
    base = {
        "attempts": {
            "attempt_records": task_count,
            "earliest_started_at_utc": "2026-08-30T00:00:00+00:00",
            "exit_status_counts": {"0": task_count},
            "latest_finished_at_utc": "2026-08-30T00:30:00+00:00",
            "observed_window_definition": projection.ATTEMPT_WINDOW_DEFINITION,
            "successful_task_coordinates": task_count,
        },
        "attestation_type": projection.ATTESTATION_TYPE,
        "completion": {
            "backgrounds_expected": len(projection.BMRS),
            "candidate_pairs_all_tasks": pairs_per_background * len(projection.BMRS),
            "candidate_pairs_per_background": pairs_per_background,
            "cohorts_expected": len(projection.TCGA_COHORTS),
            "cohorts_validated": len(projection.TCGA_COHORTS),
            "features_per_task": 500,
            "ordered_sample_memberships_across_cohorts": len(
                projection.TCGA_COHORTS,
            ),
            "same_base_pairs_excluded_per_background": len(
                projection.TCGA_COHORTS,
            ),
            "tasks_expected": task_count,
            "tasks_validated": task_count,
        },
        "contracts": contracts,
        "created_at_utc": "2026-08-30T01:00:00+00:00",
        "frozen_run": {
            "analysis": "tcga-revision-k500",
            "implementation_sha256": snapshot,
            "mutsig_root": "/synthetic/provider/mutsig",
            "outer_git_clean": True,
            "outer_git_head": source_a,
            "run_manifest_path": "run_manifest.json",
            "run_manifest_sha256": _sha256(run_raw),
            "run_root": "/synthetic/run",
            "run_schema_version": "3.0.0",
            "source_root": "/synthetic/provider/cohorts",
            "top_k": 500,
        },
        "generator": {
            "bytes": projection.ATTESTOR_BYTES,
            "path": projection.ATTESTOR_PATH,
            "sha256": projection.ATTESTOR_SHA256,
        },
        "pre_attestation_inventory": {
            "definition": projection.ATTESTATION_INVENTORY_DEFINITION,
            "excluded_paths": ["completion_attestation.json"],
            "file_count": len(inventory_files),
            "files": inventory_files,
            "records_sha256": projection._json_sha256(  # noqa: SLF001
                inventory_files,
            ),
            "self_reference_policy": projection.ATTESTATION_SELF_REFERENCE_POLICY,
            "total_bytes": sum(record["bytes"] for record in inventory_files),
        },
        "resources": {
            "runner_resource_policy": {"jobs": 2},
            "task_elapsed_seconds": {
                "definition": projection.ELAPSED_RESOURCE_DEFINITION,
                "maximum": 1.0,
                "median": 1.0,
                "minimum": 1.0,
                "sum": float(task_count),
                "sum_by_background": {
                    bmr: float(len(projection.TCGA_COHORTS)) for bmr in projection.BMRS
                },
            },
            "task_peak_rss_bytes": {
                "definition": projection.PEAK_RSS_RESOURCE_DEFINITION,
                "maximum": 1024,
                "maximum_by_background": dict.fromkeys(projection.BMRS, 1024),
                "source": projection.PEAK_RSS_SOURCE,
            },
        },
        "schema_version": projection.ATTESTATION_SCHEMA,
        "scope": {
            "interpretation": "strictly non-inferential",
            "prohibited_operations": list(
                projection.ATTESTATION_PROHIBITED_OPERATIONS,
            ),
            "run": projection.ATTESTATION_SCOPE_RUN,
            "task_validation_boundary": (
                projection.ATTESTATION_TASK_VALIDATION_BOUNDARY
            ),
            "validated_operations": projection.ATTESTATION_VALIDATED_OPERATIONS,
        },
        "sealed_completion": {
            "bytes": len(seal_raw),
            "path": "sealed_completion_manifest.json",
            "sha256": _sha256(seal_raw),
        },
        "status": "complete",
        "tasks": tasks,
    }
    return {
        **base,
        "attestation_payload_sha256": projection._json_sha256(base),  # noqa: SLF001
    }


@pytest.fixture
def projection_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ProjectionFixture:
    git = projection.GIT_EXECUTABLE_PATH
    assert git.is_file()
    repo = tmp_path / "outer" / "inner" / "repo"
    repo.mkdir(parents=True)
    _git(repo, git, "init", "-q")
    _git(repo, git, "config", "user.email", "synthetic@example.test")
    _git(repo, git, "config", "user.name", "Synthetic Test")

    source_files = {
        "exec/model.py": b"MODEL = 'frozen'\n",
        "scripts/run.sh": b"#!/bin/sh\nexit 0\n",
    }
    for name, raw in source_files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    _git(repo, git, "add", ".")
    _git(repo, git, "commit", "-q", "-m", "source a")
    source_a = _git(repo, git, "rev-parse", "HEAD")

    builder_path = repo / projection.BUILDER_PATH
    builder_path.parent.mkdir(parents=True, exist_ok=True)
    builder_path.write_bytes(Path(projection.__file__).read_bytes())
    (repo / "release.txt").write_text("release b\n", encoding="utf-8")
    _git(repo, git, "add", ".")
    _git(repo, git, "commit", "-q", "-m", "release b")
    release_b = _git(repo, git, "rev-parse", "HEAD")
    release_tag = "k500-test-v1"
    _git(repo, git, "tag", release_tag)

    monkeypatch.setattr(projection, "SOURCE_A_COMMIT", source_a)
    monkeypatch.setattr(projection, "GIT_EXECUTION_PATHS", tuple(source_files))
    snapshot = _source_snapshot(source_files)
    monkeypatch.setattr(
        projection,
        "EXPECTED_EXECUTION_SNAPSHOT_SHA256",
        projection._json_sha256(snapshot),  # noqa: SLF001
    )

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    authority, digests = _authority(git)
    run_payload = _run_manifest(
        source_a=source_a,
        snapshot=snapshot,
        authority=authority,
    )
    run = metadata / "run_manifest.json"
    run_raw = _write_json(run, run_payload)
    seal_payload = _seal(authority=authority, run_raw=run_raw)
    seal = metadata / "sealed_completion_manifest.json"
    seal_raw = _write_json(seal, seal_payload)
    attestation_payload = _attestation(
        source_a=source_a,
        snapshot=snapshot,
        run_raw=run_raw,
        seal_raw=seal_raw,
    )
    attestation = metadata / "completion_attestation.json"
    attestation_raw = _write_json(attestation, attestation_payload)
    output_parent = tmp_path / "release"
    output_parent.mkdir()
    return ProjectionFixture(
        repo=repo,
        git=git,
        source_a=source_a,
        release_b=release_b,
        release_tag=release_tag,
        attestation=attestation,
        seal=seal,
        run=run,
        output=output_parent / "k500_authority_projection.json",
        attestation_sha256=_sha256(attestation_raw),
        seal_sha256=_sha256(seal_raw),
        run_sha256=_sha256(run_raw),
        authority_digests=digests,
    )


def test_build_and_validate_projection_without_reopening_run_metadata(
    projection_fixture: ProjectionFixture,
) -> None:
    receipt = projection_fixture.build()

    assert receipt.projection_path == projection_fixture.output
    assert receipt.source_a_commit == projection_fixture.source_a
    assert receipt.release_b_commit == projection_fixture.release_b
    assert receipt.release_tag == projection_fixture.release_tag
    assert receipt.git_blob_count == 2
    assert receipt.generated_file_count == 1
    assert receipt.snapshot_file_count == 3
    assert receipt.authority_digest_count == 6
    assert dict(receipt.authority_digests) == projection_fixture.authority_digests
    payload = json.loads(projection_fixture.output.read_bytes())
    assert set(payload) == {
        "binding",
        "builder",
        "revision_authority",
        "schema",
        "source",
    }
    assert set(payload["revision_authority"]) == set(
        projection.AUTHORITY_DIGEST_FIELDS,
    )

    projection_fixture.attestation.unlink()
    projection_fixture.seal.unlink()
    projection_fixture.run.unlink()
    validated = projection.validate_k500_authority_projection(
        projection_fixture.output,
        expected_projection_sha256=receipt.projection_sha256,
        repo_root=projection_fixture.repo,
        git_executable=projection_fixture.git,
    )
    assert validated == receipt


def test_builder_is_deterministic_and_never_replaces(
    projection_fixture: ProjectionFixture,
) -> None:
    first = projection_fixture.build()
    frozen = projection_fixture.output.read_bytes()

    with pytest.raises(FileExistsError, match="replace"):
        projection_fixture.build()

    assert projection_fixture.output.read_bytes() == frozen
    assert first.projection_sha256 == _sha256(frozen)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("expected_completion_attestation_sha256", "0" * 64),
        ("expected_sealed_completion_sha256", "0" * 64),
        ("expected_run_manifest_sha256", "0" * 64),
    ],
)
def test_builder_requires_every_independent_metadata_anchor(
    projection_fixture: ProjectionFixture,
    argument: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="anchored metadata"):
        projection_fixture.build(**{argument: value})
    assert not projection_fixture.output.exists()


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "fifo"])
def test_builder_rejects_nonprivate_or_special_metadata_inputs(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
    attack: str,
) -> None:
    original = projection_fixture.seal
    attacked = tmp_path / "attacked-seal.json"
    if attack == "symlink":
        attacked.symlink_to(original)
    elif attack == "hardlink":
        attacked.hardlink_to(original)
    else:
        os.mkfifo(attacked)

    with pytest.raises((OSError, ValueError)):
        projection_fixture.build(sealed_completion_path=attacked)
    assert not projection_fixture.output.exists()


def test_builder_rejects_ancestor_symlink(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked-metadata"
    linked.symlink_to(projection_fixture.seal.parent, target_is_directory=True)
    attacked = linked / projection_fixture.seal.name

    with pytest.raises(OSError, match=r".+"):
        projection_fixture.build(sealed_completion_path=attacked)
    assert not projection_fixture.output.exists()


@pytest.mark.parametrize(
    ("path_attribute", "size_limit"),
    [
        ("attestation", projection.MAX_COMPLETION_ATTESTATION_BYTES),
        ("seal", projection.MAX_SEALED_COMPLETION_BYTES),
        ("run", projection.MAX_RUN_MANIFEST_BYTES),
    ],
)
def test_builder_rejects_oversized_sparse_metadata_before_read_or_parse(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
    path_attribute: str,
    size_limit: int,
) -> None:
    attacked = getattr(projection_fixture, path_attribute)
    _make_sparse(attacked, size=size_limit + 1)
    attacked_stat = attacked.stat()
    attacked_identity = (attacked_stat.st_dev, attacked_stat.st_ino)
    original_read = projection._read_descriptor  # noqa: SLF001

    def guarded_read(
        descriptor: int,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == attacked_identity:
            pytest.fail("oversized sparse metadata reached a descriptor read")
        return original_read(descriptor, max_bytes=max_bytes, label=label)

    def reject_parse(raw: bytes, *, label: str) -> dict[str, Any]:
        _ = raw, label
        pytest.fail("metadata parsing began before all byte limits passed")

    monkeypatch.setattr(projection, "_read_descriptor", guarded_read)
    monkeypatch.setattr(projection, "_parse_json", reject_parse)
    before = _descriptor_count()
    for _ in range(5):
        with pytest.raises(ValueError, match="safety limit"):
            projection_fixture.build()
    after = _descriptor_count()

    assert before is None or after == before
    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_builder_rejects_oversized_sparse_builder_before_read_or_parse(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacked = tmp_path / "oversized-projection-builder.py"
    _make_sparse(attacked, size=projection.MAX_BUILDER_BYTES + 1, mode=0o444)
    attacked_stat = attacked.stat()
    attacked_identity = (attacked_stat.st_dev, attacked_stat.st_ino)
    original_read = projection._read_descriptor  # noqa: SLF001

    def guarded_read(
        descriptor: int,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == attacked_identity:
            pytest.fail("oversized sparse builder reached a descriptor read")
        return original_read(descriptor, max_bytes=max_bytes, label=label)

    def reject_parse(raw: bytes, *, label: str) -> dict[str, Any]:
        _ = raw, label
        pytest.fail("metadata parsing began before the builder byte limit passed")

    monkeypatch.setattr(projection, "__file__", attacked.as_posix())
    monkeypatch.setattr(projection, "_read_descriptor", guarded_read)
    monkeypatch.setattr(projection, "_parse_json", reject_parse)
    before = _descriptor_count()
    for _ in range(5):
        with pytest.raises(ValueError, match="safety limit"):
            projection_fixture.build()
    after = _descriptor_count()

    assert before is None or after == before
    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_validator_rejects_oversized_sparse_projection_before_read_or_parse(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacked = projection_fixture.output
    _make_sparse(attacked, size=projection.MAX_PROJECTION_BYTES + 1, mode=0o444)
    attacked_stat = attacked.stat()
    attacked_identity = (attacked_stat.st_dev, attacked_stat.st_ino)
    original_read = projection._read_descriptor  # noqa: SLF001

    def guarded_read(
        descriptor: int,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == attacked_identity:
            pytest.fail("oversized sparse projection reached a descriptor read")
        return original_read(descriptor, max_bytes=max_bytes, label=label)

    def reject_parse(raw: bytes, *, label: str) -> dict[str, Any]:
        _ = raw, label
        pytest.fail("oversized projection reached JSON parsing")

    monkeypatch.setattr(projection, "_read_descriptor", guarded_read)
    monkeypatch.setattr(projection, "_parse_json", reject_parse)
    before = _descriptor_count()
    for _ in range(5):
        with pytest.raises(ValueError, match="safety limit"):
            projection.validate_k500_authority_projection(
                attacked,
                expected_projection_sha256="0" * 64,
                repo_root=projection_fixture.repo,
                git_executable=projection_fixture.git,
            )
    after = _descriptor_count()

    assert before is None or after == before
    assert list(attacked.parent.iterdir()) == [attacked]


def test_builder_rejects_cross_consistent_authority_edits(
    projection_fixture: ProjectionFixture,
) -> None:
    run = json.loads(projection_fixture.run.read_bytes())
    seal = json.loads(projection_fixture.seal.read_bytes())
    run["revision_authority"]["expected_fit_approval_sha256"] = "9" * 64
    seal["authority"] = run["revision_authority"]
    run_raw = _write_json(projection_fixture.run, run)
    seal["run_manifest"] = {"bytes": len(run_raw), "sha256": _sha256(run_raw)}
    seal_raw = _write_json(projection_fixture.seal, seal)
    attestation = json.loads(projection_fixture.attestation.read_bytes())
    unsigned = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_payload_sha256"
    }
    unsigned["frozen_run"]["run_manifest_sha256"] = _sha256(run_raw)
    unsigned["sealed_completion"] = {
        "bytes": len(seal_raw),
        "path": "sealed_completion_manifest.json",
        "sha256": _sha256(seal_raw),
    }
    attestation = {
        **unsigned,
        "attestation_payload_sha256": projection._json_sha256(unsigned),  # noqa: SLF001
    }
    _write_json(projection_fixture.attestation, attestation)

    with pytest.raises(ValueError, match="anchored metadata"):
        projection_fixture.build()


def test_builder_rejects_mismatched_seal_and_run_authority(
    projection_fixture: ProjectionFixture,
) -> None:
    seal = json.loads(projection_fixture.seal.read_bytes())
    seal["authority"]["expected_fit_approval_sha256"] = "9" * 64
    seal_raw = _write_json(projection_fixture.seal, seal)

    with pytest.raises(ValueError, match="authority records differ"):
        projection_fixture.build(
            expected_sealed_completion_sha256=_sha256(seal_raw),
        )


def test_builder_rejects_attestation_task_receipts_not_bound_to_seal(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(attestation: dict[str, Any]) -> None:
        attestation["tasks"][0]["task_manifest_sha256"] = "a" * 64

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_attestation=mutate,
    )

    with pytest.raises(ValueError, match="task receipt is invalid"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )
    assert not projection_fixture.output.exists()


def test_builder_rejects_cross_consistent_run_contract_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(run: dict[str, Any]) -> None:
        run["required_lrt_contract"] = "obsolete-lrt-contract"

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_run=mutate,
    )

    with pytest.raises(ValueError, match="exact result-blind K500 grid"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )
    assert not projection_fixture.output.exists()


def test_builder_rejects_provider_acceptance_cohort_digest_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(run: dict[str, Any]) -> None:
        run["revision_authority"]["provider_input"][
            "cohort_provider_receipts_sha256"
        ] = "a" * 64

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_run=mutate,
    )

    with pytest.raises(ValueError, match="internally cross-bound"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )
    assert not projection_fixture.output.exists()


def test_builder_rejects_provider_snapshot_content_address_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(run: dict[str, Any]) -> None:
        provider = run["revision_authority"]["provider_input"]
        acceptance = provider["full_acceptance_receipt"]
        acceptance["execution_snapshot"]["root"] = (
            f"_orchestration/execution-snapshot-{'e' * 64}"
        )
        provider["full_acceptance_receipt_sha256"] = projection._json_sha256(  # noqa: SLF001
            acceptance,
            newline=True,
        )

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_run=mutate,
    )

    with pytest.raises(ValueError, match="execution snapshot is invalid"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )
    assert not projection_fixture.output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_pairs_all_tasks", 1, "exact 32-by-3 grid"),
        ("tasks_validated", 95, "exact 32-by-3 grid"),
    ],
)
def test_builder_rejects_attestation_completion_summary_drift(
    projection_fixture: ProjectionFixture,
    field: str,
    value: int,
    message: str,
) -> None:
    def mutate(attestation: dict[str, Any]) -> None:
        attestation["completion"][field] = value

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_attestation=mutate,
    )

    with pytest.raises(ValueError, match=message):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )


def test_builder_rejects_attestation_inventory_output_hash_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(attestation: dict[str, Any]) -> None:
        inventory = attestation["pre_attestation_inventory"]
        for record in inventory["files"]:
            if record["path"] == ("tasks/ACC/cbase/pairwise_interaction_results.csv"):
                record["sha256"] = "a" * 64
                break
        inventory["records_sha256"] = projection._json_sha256(  # noqa: SLF001
            inventory["files"],
        )

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_attestation=mutate,
    )

    with pytest.raises(ValueError, match="lost sealed receipt"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )


def test_builder_rejects_attestation_resource_summary_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(attestation: dict[str, Any]) -> None:
        attestation["resources"]["task_elapsed_seconds"]["sum"] += 1

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_attestation=mutate,
    )

    with pytest.raises(ValueError, match="resource summary is inconsistent"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )


def test_builder_rejects_abbreviated_attestation_scope(
    projection_fixture: ProjectionFixture,
) -> None:
    def mutate(attestation: dict[str, Any]) -> None:
        attestation["scope"]["task_validation_boundary"] = "opaque file hashes only"

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_attestation=mutate,
    )

    with pytest.raises(ValueError, match="not result-blind"):
        projection_fixture.build(
            expected_completion_attestation_sha256=attestation_sha,
            expected_sealed_completion_sha256=seal_sha,
            expected_run_manifest_sha256=run_sha,
        )


def test_builder_rejects_release_b_execution_blob_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    path = projection_fixture.repo / projection.GIT_EXECUTION_PATHS[0]
    path.write_text("MODEL = 'changed'\n", encoding="utf-8")
    _git(projection_fixture.repo, projection_fixture.git, "add", ".")
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "commit",
        "-q",
        "-m",
        "changed execution blob",
    )
    changed_b = _git(
        projection_fixture.repo,
        projection_fixture.git,
        "rev-parse",
        "HEAD",
    )
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "tag",
        "changed-release",
    )

    with pytest.raises(ValueError, match="changed frozen K500 execution blob"):
        projection_fixture.build(
            release_b_commit=changed_b,
            release_tag="changed-release",
        )


def test_builder_rejects_generated_version_tracked_at_release_b(
    projection_fixture: ProjectionFixture,
) -> None:
    generated = projection_fixture.repo / projection.GENERATED_VERSION_PATH
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("VERSION = 'bad'\n", encoding="utf-8")
    _git(projection_fixture.repo, projection_fixture.git, "add", ".")
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "commit",
        "-q",
        "-m",
        "tracked generated version",
    )
    changed_b = _git(
        projection_fixture.repo,
        projection_fixture.git,
        "rev-parse",
        "HEAD",
    )
    _git(projection_fixture.repo, projection_fixture.git, "tag", "generated-release")

    with pytest.raises(ValueError, match="unexpectedly tracked"):
        projection_fixture.build(
            release_b_commit=changed_b,
            release_tag="generated-release",
        )


def test_builder_requires_clean_exact_tagged_release_head(
    projection_fixture: ProjectionFixture,
) -> None:
    (projection_fixture.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build()

    (projection_fixture.repo / "untracked.txt").unlink()
    with pytest.raises(ValueError, match="tag does not resolve"):
        projection_fixture.build(release_tag="missing-tag")


def test_builder_rejects_unanchored_git_executable(
    projection_fixture: ProjectionFixture,
) -> None:
    fake_git = projection_fixture.repo.parent / "fake-git"
    fake_git.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_git.chmod(0o500)

    with pytest.raises(ValueError, match="exactly /usr/bin/git"):
        projection_fixture.build(git_executable=fake_git)
    assert not projection_fixture.output.exists()


def test_validator_rejects_unanchored_git_executable(
    projection_fixture: ProjectionFixture,
) -> None:
    receipt = projection_fixture.build()
    fake_git = projection_fixture.repo.parent / "validator-fake-git"
    fake_git.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_git.chmod(0o500)

    with pytest.raises(ValueError, match="exactly /usr/bin/git"):
        projection.validate_k500_authority_projection(
            projection_fixture.output,
            expected_projection_sha256=receipt.projection_sha256,
            repo_root=projection_fixture.repo,
            git_executable=fake_git,
        )


def test_builder_rejects_oversized_sparse_git_before_read_or_execution(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacked = tmp_path / "synthetic-usr-bin-git"
    _make_sparse(
        attacked,
        size=projection.MAX_GIT_EXECUTABLE_BYTES + 1,
        mode=0o555,
    )
    attacked_stat = attacked.stat()
    attacked_identity = (attacked_stat.st_dev, attacked_stat.st_ino)
    anchored_receipt = {
        "bytes": projection.MAX_GIT_EXECUTABLE_BYTES + 1,
        "path": attacked.as_posix(),
        "sha256": "e" * 64,
    }

    def mutate_run(run: dict[str, Any]) -> None:
        run["git"]["executable"] = dict(anchored_receipt)
        run["revision_authority"]["provider_input"]["git_executable"] = dict(
            anchored_receipt,
        )

    attestation_sha, seal_sha, run_sha = _rewrite_metadata_chain(
        projection_fixture,
        mutate_run=mutate_run,
    )
    original_read = projection._read_descriptor  # noqa: SLF001

    def guarded_read(
        descriptor: int,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == attacked_identity:
            pytest.fail("oversized sparse Git executable reached a descriptor read")
        return original_read(descriptor, max_bytes=max_bytes, label=label)

    def reject_git_execution(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        pytest.fail("oversized Git executable reached subprocess execution")

    monkeypatch.setattr(projection, "GIT_EXECUTABLE_PATH", attacked)
    monkeypatch.setattr(projection, "_read_descriptor", guarded_read)
    monkeypatch.setattr(projection.subprocess, "Popen", reject_git_execution)
    before = _descriptor_count()
    for _ in range(5):
        with pytest.raises(ValueError, match="safety limit"):
            projection_fixture.build(
                expected_completion_attestation_sha256=attestation_sha,
                expected_sealed_completion_sha256=seal_sha,
                expected_run_manifest_sha256=run_sha,
                git_executable=attacked,
            )
    after = _descriptor_count()

    assert before is None or after == before
    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_git_invocations_ignore_poisoned_environment_and_path(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poisoned_bin = tmp_path / "poisoned-bin"
    poisoned_bin.mkdir()
    marker = tmp_path / "poisoned-git-ran"
    poisoned_git = poisoned_bin / "git"
    poisoned_git.write_text(
        f"#!/bin/sh\ntouch '{marker.as_posix()}'\nexit 91\n",
        encoding="utf-8",
    )
    poisoned_git.chmod(0o555)
    for key, value in {
        "GIT_CONFIG_GLOBAL": (tmp_path / "poisoned.gitconfig").as_posix(),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_NOSYSTEM": "0",
        "GIT_CONFIG_VALUE_0": poisoned_git.as_posix(),
        "GIT_DIR": (tmp_path / "poisoned-git-dir").as_posix(),
        "GIT_NO_LAZY_FETCH": "0",
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_OPTIONAL_LOCKS": "1",
        "GIT_WORK_TREE": (tmp_path / "poisoned-worktree").as_posix(),
        "PATH": poisoned_bin.as_posix(),
    }.items():
        monkeypatch.setenv(key, value)

    original_popen = subprocess.Popen
    observed_calls = 0

    def guarded_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        nonlocal observed_calls
        command = args[0]
        assert command[:5] == [
            projection.GIT_EXECUTABLE_PATH.as_posix(),
            "--no-pager",
            "--no-replace-objects",
            "--no-optional-locks",
            "--work-tree=.",
        ]
        config_start = 5
        expected_environment = dict(projection._SEALED_GIT_ENVIRONMENT)  # noqa: SLF001
        if command[config_start].startswith("--git-dir="):
            config_start += 1
            expected_environment["GIT_OBJECT_DIRECTORY"] = (
                projection_fixture.repo / ".git/objects"
            ).as_posix()
        config_end = config_start + len(
            projection._GIT_CONFIG_OVERRIDES,  # noqa: SLF001
        )
        assert (
            list(projection._GIT_CONFIG_OVERRIDES)  # noqa: SLF001
            == command[config_start:config_end]
        )
        assert kwargs["env"] == expected_environment
        observed_calls += 1
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(projection.subprocess, "Popen", guarded_popen)
    receipt = projection_fixture.build()

    assert observed_calls > 0
    assert receipt.release_b_commit == projection_fixture.release_b
    assert not marker.exists()


def test_repository_local_fsmonitor_cannot_execute(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "fsmonitor-executed"
    fsmonitor = tmp_path / "synthetic-fsmonitor"
    fsmonitor.write_text(
        f"#!/bin/sh\ntouch '{marker.as_posix()}'\nexit 0\n",
        encoding="utf-8",
    )
    fsmonitor.chmod(0o555)
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "config",
        "core.fsmonitor",
        fsmonitor.as_posix(),
    )

    receipt = projection_fixture.build()

    assert receipt.release_b_commit == projection_fixture.release_b
    assert not marker.exists()


def test_repository_local_worktree_cannot_hide_dirty_release(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
) -> None:
    alternate_worktree = tmp_path / "empty-alternate-worktree"
    alternate_worktree.mkdir()
    (projection_fixture.repo / "untracked-release-file").write_text(
        "dirty\n",
        encoding="utf-8",
    )
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "config",
        "core.worktree",
        alternate_worktree.as_posix(),
    )

    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()


def test_repository_info_exclude_cannot_hide_dirty_release(
    projection_fixture: ProjectionFixture,
) -> None:
    hidden_name = "hidden-by-original-info-exclude"
    info_exclude = projection_fixture.repo / ".git/info/exclude"
    info_exclude.write_text(f"/{hidden_name}\n", encoding="utf-8")
    (projection_fixture.repo / hidden_name).write_text("dirty\n", encoding="utf-8")
    assert (
        _git(
            projection_fixture.repo,
            projection_fixture.git,
            "ls-files",
            "--others",
            "--exclude-standard",
        )
        == ""
    )

    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()


@pytest.mark.parametrize("location", ["root", "nested"])
def test_untracked_gitignore_cannot_hide_dirty_release(
    projection_fixture: ProjectionFixture,
    location: str,
) -> None:
    parent = projection_fixture.repo
    if location == "nested":
        parent /= "untracked-directory"
        parent.mkdir()
    (parent / ".gitignore").write_text("*\n", encoding="utf-8")
    (parent / "hidden-dirty-file").write_text("dirty\n", encoding="utf-8")
    assert (
        _git(
            projection_fixture.repo,
            projection_fixture.git,
            "status",
            "--porcelain=v1",
        )
        == ""
    )

    with pytest.raises(ValueError, match="untracked worktree ignore-control"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()


def test_git_config_value_change_is_detected_across_clean_proof(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "config",
        "synthetic.guard",
        "before",
    )
    original_git = projection._git  # noqa: SLF001
    attacked = False

    def change_config_after_value_snapshot(git, repo, *arguments, **kwargs):
        nonlocal attacked
        result = original_git(git, repo, *arguments, **kwargs)
        if (
            not attacked
            and arguments[:1] == ("config",)
            and "--name-only" not in arguments
        ):
            attacked = True
            _git(
                projection_fixture.repo,
                projection_fixture.git,
                "config",
                "synthetic.guard",
                "after",
            )
        return result

    monkeypatch.setattr(projection, "_git", change_config_after_value_snapshot)

    with pytest.raises(ValueError, match=r"configuration.*changed"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()


def test_repository_local_filemode_cannot_hide_tracked_mode_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    tracked = projection_fixture.repo / "exec/model.py"
    tracked.chmod(0o755)
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "config",
        "core.fileMode",
        "false",
    )

    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()


def test_case_colliding_untracked_path_cannot_hide_from_clean_proof(
    projection_fixture: ProjectionFixture,
) -> None:
    nested = projection_fixture.repo / "nested"
    nested.mkdir()
    tracked = nested / "Foo"
    tracked.write_text("frozen\n", encoding="utf-8")
    _git(projection_fixture.repo, projection_fixture.git, "add", ".")
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "commit",
        "-q",
        "-m",
        "add case-sensitive tracked path",
    )
    changed_b = _git(
        projection_fixture.repo,
        projection_fixture.git,
        "rev-parse",
        "HEAD",
    )
    changed_tag = "case-collision-release"
    _git(projection_fixture.repo, projection_fixture.git, "tag", changed_tag)
    colliding = nested / "foo"
    colliding.write_text("untracked\n", encoding="utf-8")
    if tracked.samefile(colliding):
        pytest.skip("case-colliding paths are unavailable on this filesystem")
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "config",
        "core.ignorecase",
        "true",
    )

    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build(
            release_b_commit=changed_b,
            release_tag=changed_tag,
        )

    assert not projection_fixture.output.exists()


def test_repository_clean_filter_is_rejected_before_execution(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
) -> None:
    attributes = projection_fixture.repo / ".gitattributes"
    filtered = projection_fixture.repo / "filtered.txt"
    attributes.write_text("filtered.txt filter=evil\n", encoding="utf-8")
    filtered.write_text("frozen\n", encoding="utf-8")
    _git(projection_fixture.repo, projection_fixture.git, "add", ".")
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "commit",
        "-q",
        "-m",
        "add synthetic filter attribute",
    )
    changed_b = _git(
        projection_fixture.repo,
        projection_fixture.git,
        "rev-parse",
        "HEAD",
    )
    changed_tag = "filter-attack-release"
    _git(projection_fixture.repo, projection_fixture.git, "tag", changed_tag)
    marker = tmp_path / "clean-filter-executed"
    clean_filter = tmp_path / "synthetic-clean-filter"
    clean_filter.write_text(
        f"#!/bin/sh\ntouch '{marker.as_posix()}'\ncat\n",
        encoding="utf-8",
    )
    clean_filter.chmod(0o555)
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "config",
        "filter.evil.clean",
        clean_filter.as_posix(),
    )
    filtered.write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="executable filter"):
        projection_fixture.build(
            release_b_commit=changed_b,
            release_tag=changed_tag,
        )

    assert not marker.exists()
    assert not projection_fixture.output.exists()


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_non_normal_index_flags_cannot_hide_tracked_drift(
    projection_fixture: ProjectionFixture,
    flag: str,
) -> None:
    tracked_relative = "exec/model.py"
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "update-index",
        flag,
        tracked_relative,
    )
    (projection_fixture.repo / tracked_relative).write_text(
        "MODEL = 'hidden drift'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-normal tracked entry"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()


@pytest.mark.parametrize("attack", ["dirty", "moved"])
def test_release_clean_check_rejects_dirty_or_moved_submodule(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
    attack: str,
) -> None:
    changed_tag = f"submodule-{attack}"
    submodule, changed_b = _add_synthetic_submodule(
        projection_fixture,
        tmp_path,
        release_tag=changed_tag,
    )
    if attack == "dirty":
        (submodule / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    else:
        submodule.rename(tmp_path / "moved-synthetic-submodule")

    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build(
            release_b_commit=changed_b,
            release_tag=changed_tag,
        )

    assert not projection_fixture.output.exists()


def test_git_untracked_output_is_bounded_without_descriptor_leak(
    projection_fixture: ProjectionFixture,
) -> None:
    for index in range(400):
        name = f"untracked-{index:04d}-{'x' * 180}"
        (projection_fixture.repo / name).touch()
    before = _descriptor_count()

    with pytest.raises(ValueError, match="Git stdout exceeded"):
        projection_fixture.build()

    after = _descriptor_count()
    assert before is None or after == before
    assert not projection_fixture.output.exists()


def test_git_error_output_is_bounded_without_descriptor_leak(
    projection_fixture: ProjectionFixture,
) -> None:
    invalid_object = "f" * 4096
    before = _descriptor_count()

    with pytest.raises(ValueError, match="Git stderr exceeded"):
        projection._git(  # noqa: SLF001
            projection_fixture.git,
            projection_fixture.repo,
            "cat-file",
            "blob",
            invalid_object,
            check=False,
            max_stderr_bytes=512,
        )

    after = _descriptor_count()
    assert before is None or after == before


def test_git_fifo_hang_is_timed_out_reaped_and_does_not_publish(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = projection_fixture.repo / ".git/index"
    index.unlink()
    os.mkfifo(index)
    original_git = projection._git  # noqa: SLF001

    def quick_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        kwargs["timeout_seconds"] = 0.1
        return original_git(*args, **kwargs)

    monkeypatch.setattr(projection, "_git", quick_timeout)
    before = _descriptor_count()

    with pytest.raises(TimeoutError, match="safety timeout"):
        projection_fixture.build()

    after = _descriptor_count()
    assert before is None or after == before
    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_bounded_collector_timeout_kills_blocked_descendant_process_group(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "blocked-child-fifo"
    os.mkfifo(fifo)
    ready = tmp_path / "blocked-child-ready"
    child_code = (
        "import os,sys,time; "
        "fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NONBLOCK); "
        "open(sys.argv[2],'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    leader_code = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],"
        "sys.argv[3]]); child.wait()"
    )
    before = _descriptor_count()
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-c",
            leader_code,
            child_code,
            fifo.as_posix(),
            ready.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    with pytest.raises(TimeoutError, match="safety timeout"):
        projection._collect_bounded_process_output(  # noqa: SLF001
            process,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
            timeout_seconds=0.5,
        )

    after = _descriptor_count()
    assert before is None or after == before
    assert ready.is_file()
    with pytest.raises(OSError, match=os.strerror(errno.ENXIO)):
        os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)


def test_process_group_cleanup_tolerates_permission_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PermissionDeniedProcess:
        pid = 91_919

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise PermissionError

    def deny_group_kill(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(projection.os, "killpg", deny_group_kill)

    projection._kill_git_process_group(PermissionDeniedProcess())  # type: ignore[arg-type]  # noqa: SLF001


def test_git_replace_ref_cannot_alter_source_blob_verification(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
) -> None:
    alternate_blob = (
        subprocess.run(  # noqa: S603
            [projection_fixture.git.as_posix(), "hash-object", "-w", "--stdin"],
            cwd=projection_fixture.repo,
            check=True,
            capture_output=True,
            input=b"MODEL = 'replacement attack'\n",
        )
        .stdout.decode("ascii")
        .strip()
    )
    replacement_index = tmp_path / "replacement.index"
    replacement_environment = {
        **os.environ,
        "GIT_INDEX_FILE": replacement_index.as_posix(),
    }
    replacement_environment.pop("GIT_NO_REPLACE_OBJECTS", None)
    subprocess.run(  # noqa: S603
        [projection_fixture.git.as_posix(), "read-tree", projection_fixture.source_a],
        cwd=projection_fixture.repo,
        env=replacement_environment,
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603
        [
            projection_fixture.git.as_posix(),
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{alternate_blob},exec/model.py",
        ],
        cwd=projection_fixture.repo,
        env=replacement_environment,
        check=True,
        capture_output=True,
    )
    alternate_tree = (
        subprocess.run(  # noqa: S603
            [projection_fixture.git.as_posix(), "write-tree"],
            cwd=projection_fixture.repo,
            env=replacement_environment,
            check=True,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )
    replacement_commit = (
        subprocess.run(  # noqa: S603
            [projection_fixture.git.as_posix(), "commit-tree", alternate_tree],
            cwd=projection_fixture.repo,
            env=replacement_environment,
            check=True,
            capture_output=True,
            input=b"replacement attack\n",
        )
        .stdout.decode("ascii")
        .strip()
    )
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "replace",
        projection_fixture.source_a,
        replacement_commit,
    )
    replaced_bytes = subprocess.run(  # noqa: S603
        [
            projection_fixture.git.as_posix(),
            "cat-file",
            "blob",
            f"{projection_fixture.source_a}:exec/model.py",
        ],
        cwd=projection_fixture.repo,
        env=replacement_environment,
        check=True,
        capture_output=True,
    ).stdout
    assert replaced_bytes == b"MODEL = 'replacement attack'\n"

    receipt = projection_fixture.build()

    assert receipt.execution_snapshot_sha256 == (
        projection.EXPECTED_EXECUTION_SNAPSHOT_SHA256
    )


def test_builder_rejects_release_b_builder_blob_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    builder = projection_fixture.repo / projection.BUILDER_PATH
    builder.write_text("# substituted release builder\n", encoding="utf-8")
    _git(projection_fixture.repo, projection_fixture.git, "add", ".")
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "commit",
        "-q",
        "-m",
        "substituted builder",
    )
    changed_b = _git(
        projection_fixture.repo,
        projection_fixture.git,
        "rev-parse",
        "HEAD",
    )
    _git(projection_fixture.repo, projection_fixture.git, "tag", "builder-drift")

    with pytest.raises(ValueError, match="builder differs"):
        projection_fixture.build(
            release_b_commit=changed_b,
            release_tag="builder-drift",
        )


def test_builder_requires_source_a_to_be_release_b_ancestor(
    projection_fixture: ProjectionFixture,
) -> None:
    tree = _git(
        projection_fixture.repo,
        projection_fixture.git,
        "rev-parse",
        f"{projection_fixture.release_b}^{{tree}}",
    )
    completed = subprocess.run(  # noqa: S603
        [projection_fixture.git.as_posix(), "commit-tree", tree],
        cwd=projection_fixture.repo,
        check=True,
        capture_output=True,
        input=b"orphan release\n",
    )
    orphan = completed.stdout.decode("ascii").strip()
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "update-ref",
        "refs/heads/orphan-release",
        orphan,
    )
    _git(
        projection_fixture.repo,
        projection_fixture.git,
        "switch",
        "-q",
        "orphan-release",
    )
    _git(projection_fixture.repo, projection_fixture.git, "tag", "orphan-release")

    with pytest.raises(ValueError, match="not an ancestor"):
        projection_fixture.build(
            release_b_commit=orphan,
            release_tag="orphan-release",
        )


def test_validator_rejects_projection_schema_drift_even_with_new_outer_hash(
    projection_fixture: ProjectionFixture,
) -> None:
    projection_fixture.build()
    payload = json.loads(projection_fixture.output.read_bytes())
    payload["extra"] = True
    projection_fixture.output.chmod(0o600)
    raw = _write_json(projection_fixture.output, payload)
    projection_fixture.output.chmod(0o444)

    with pytest.raises(ValueError, match="closed schema"):
        projection.validate_k500_authority_projection(
            projection_fixture.output,
            expected_projection_sha256=_sha256(raw),
            repo_root=projection_fixture.repo,
            git_executable=projection_fixture.git,
        )


def test_validator_rejects_six_digest_projection_drift(
    projection_fixture: ProjectionFixture,
) -> None:
    projection_fixture.build()
    payload = json.loads(projection_fixture.output.read_bytes())
    payload["revision_authority"].pop("fit_approval_sha256")
    projection_fixture.output.chmod(0o600)
    raw = _write_json(projection_fixture.output, payload)
    projection_fixture.output.chmod(0o444)

    with pytest.raises(ValueError, match="six-digest"):
        projection.validate_k500_authority_projection(
            projection_fixture.output,
            expected_projection_sha256=_sha256(raw),
            repo_root=projection_fixture.repo,
            git_executable=projection_fixture.git,
        )


def test_validator_rejects_noncanonical_projection_even_with_new_outer_hash(
    projection_fixture: ProjectionFixture,
) -> None:
    projection_fixture.build()
    payload = json.loads(projection_fixture.output.read_bytes())
    projection_fixture.output.chmod(0o600)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    projection_fixture.output.write_bytes(raw)
    projection_fixture.output.chmod(0o444)

    with pytest.raises(ValueError, match="exact canonical JSON"):
        projection.validate_k500_authority_projection(
            projection_fixture.output,
            expected_projection_sha256=_sha256(raw),
            repo_root=projection_fixture.repo,
            git_executable=projection_fixture.git,
        )


def test_builder_rejects_noncanonical_anchored_attestation(
    projection_fixture: ProjectionFixture,
) -> None:
    payload = json.loads(projection_fixture.attestation.read_bytes())
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    projection_fixture.attestation.write_bytes(raw)

    with pytest.raises(ValueError, match="not exact canonical"):
        projection_fixture.build(
            expected_completion_attestation_sha256=_sha256(raw),
        )


def test_publication_rejects_oversized_projection_before_staging(
    projection_fixture: ProjectionFixture,
) -> None:
    before = _descriptor_count()
    for _ in range(5):
        with pytest.raises(ValueError, match="safety limit"):
            projection._publish_no_replace(  # noqa: SLF001
                projection_fixture.output,
                b"x" * (projection.MAX_PROJECTION_BYTES + 1),
                protected=(),
            )
    after = _descriptor_count()

    assert before is None or after == before
    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_publication_preserves_racing_destination(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_link = projection.os.link
    attacker = b"attacker-owned\n"

    def racing_link(src, dst, **kwargs):
        destination_fd = os.open(
            dst,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        try:
            os.write(destination_fd, attacker)
        finally:
            os.close(destination_fd)
        return original_link(src, dst, **kwargs)

    monkeypatch.setattr(projection.os, "link", racing_link)

    with pytest.raises(FileExistsError, match="racing"):
        projection_fixture.build()
    assert projection_fixture.output.read_bytes() == attacker


def test_post_publication_input_mutation_is_detected_and_projection_preserved(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = projection._publish_no_replace  # noqa: SLF001

    def mutate_then_publish(output_path, content, *, protected):
        result = original_publish(output_path, content, protected=protected)
        projection_fixture.seal.write_bytes(
            projection_fixture.seal.read_bytes() + b" ",
        )
        return result

    monkeypatch.setattr(projection, "_publish_no_replace", mutate_then_publish)

    with pytest.raises(ValueError, match="changed"):
        projection_fixture.build()
    assert projection_fixture.output.is_file()


def test_post_publication_output_parent_swap_is_detected(
    projection_fixture: ProjectionFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = projection._publish_no_replace  # noqa: SLF001
    moved_parent = tmp_path / "moved-release"

    def publish_then_swap(output_path, content, *, protected):
        result = original_publish(output_path, content, protected=protected)
        output_path.parent.rename(moved_parent)
        output_path.parent.mkdir()
        return result

    monkeypatch.setattr(projection, "_publish_no_replace", publish_then_swap)

    with pytest.raises(ValueError, match=r"ancestor|metadata changed"):
        projection_fixture.build()
    assert not projection_fixture.output.exists()
    assert (moved_parent / projection_fixture.output.name).is_file()


def test_publication_detects_extra_staging_hardlink(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_link = projection.os.link
    extra_name = "attacker-staging-link"

    def hardlink_then_publish(src, dst, **kwargs):
        original_link(
            src,
            extra_name,
            src_dir_fd=kwargs["src_dir_fd"],
            dst_dir_fd=kwargs["dst_dir_fd"],
            follow_symlinks=False,
        )
        return original_link(src, dst, **kwargs)

    monkeypatch.setattr(projection.os, "link", hardlink_then_publish)

    with pytest.raises(ValueError, match="link count"):
        projection_fixture.build()


def test_publication_rejects_mode_drift_before_visible_readback(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_link = projection.os.link

    def link_then_chmod(src, dst, **kwargs):
        result = original_link(src, dst, **kwargs)
        os.chmod(dst, 0o600, dir_fd=kwargs["dst_dir_fd"])
        return result

    monkeypatch.setattr(projection.os, "link", link_then_chmod)

    with pytest.raises(ValueError, match="private link"):
        projection_fixture.build()
    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_publication_staging_readback_failure_removes_owned_destination(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = projection._read_descriptor  # noqa: SLF001

    def corrupt_staging_readback(
        descriptor: int,
        *,
        max_bytes: int,
        label: str,
    ) -> bytes:
        observed = original_read(descriptor, max_bytes=max_bytes, label=label)
        if label == "projection staging file":
            return observed + b"corrupt"
        return observed

    monkeypatch.setattr(projection, "_read_descriptor", corrupt_staging_readback)

    with pytest.raises(ValueError, match="staging bytes changed"):
        projection_fixture.build()

    assert not projection_fixture.output.exists()
    assert list(projection_fixture.output.parent.iterdir()) == []


def test_validator_rejects_writable_projection(
    projection_fixture: ProjectionFixture,
) -> None:
    receipt = projection_fixture.build()
    projection_fixture.output.chmod(0o600)

    with pytest.raises(ValueError, match="immutable read-only"):
        projection.validate_k500_authority_projection(
            projection_fixture.output,
            expected_projection_sha256=receipt.projection_sha256,
            repo_root=projection_fixture.repo,
            git_executable=projection_fixture.git,
        )


def test_source_boundary_rejects_repository_rename_and_restore(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = projection._git  # noqa: SLF001
    moved = projection_fixture.repo.with_name("temporarily-moved-repo")
    attacked = False

    def rename_after_git(*args, **kwargs):
        nonlocal attacked
        result = original_git(*args, **kwargs)
        if not attacked:
            attacked = True
            projection_fixture.repo.rename(moved)
            projection_fixture.repo.mkdir()
            projection_fixture.repo.rmdir()
            moved.rename(projection_fixture.repo)
        return result

    monkeypatch.setattr(projection, "_git", rename_after_git)

    with pytest.raises(ValueError, match="metadata changed"):
        projection_fixture.build()
    assert not projection_fixture.output.exists()


def test_source_boundary_rejects_in_place_worktree_drift_after_clean_check(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = projection._git  # noqa: SLF001
    attacked = False

    def mutate_after_shadow_status(git, repo, *arguments, **kwargs):
        nonlocal attacked
        result = original_git(git, repo, *arguments, **kwargs)
        if not attacked and arguments[:1] == ("status",) and kwargs.get("git_dir"):
            attacked = True
            target = projection_fixture.repo / projection.GIT_EXECUTION_PATHS[0]
            target.write_text("MODEL = 'racing worktree drift'\n", encoding="utf-8")
        return result

    monkeypatch.setattr(projection, "_git", mutate_after_shadow_status)

    with pytest.raises(ValueError, match="exactly clean"):
        projection_fixture.build()
    assert not projection_fixture.output.exists()


def test_source_boundary_rejects_nested_untracked_after_final_status(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = projection._git  # noqa: SLF001
    shadow_status_calls = 0

    def add_nested_untracked_after_final_status(git, repo, *arguments, **kwargs):
        nonlocal shadow_status_calls
        result = original_git(git, repo, *arguments, **kwargs)
        if arguments[:1] == ("status",) and kwargs.get("git_dir"):
            shadow_status_calls += 1
            if shadow_status_calls == 2:
                target = projection_fixture.repo / "exec/untracked-after-status"
                target.write_text("racing untracked path\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        projection,
        "_git",
        add_nested_untracked_after_final_status,
    )

    with pytest.raises(ValueError, match="tracked-parent metadata changed"):
        projection_fixture.build()

    assert shadow_status_calls == 2
    assert not projection_fixture.output.exists()


def test_source_boundary_holds_tracked_file_after_terminal_clean_proof(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = projection._git  # noqa: SLF001
    head_reads = 0

    def mutate_after_terminal_clean(git, repo, *arguments, **kwargs):
        nonlocal head_reads
        result = original_git(git, repo, *arguments, **kwargs)
        if arguments == ("rev-parse", "HEAD"):
            head_reads += 1
            if head_reads == 2:
                target = projection_fixture.repo / projection.GIT_EXECUTION_PATHS[0]
                target.write_text("MODEL = 'post-proof drift'\n", encoding="utf-8")
        return result

    monkeypatch.setattr(projection, "_git", mutate_after_terminal_clean)

    with pytest.raises(ValueError, match="tracked file changed"):
        projection_fixture.build()

    assert head_reads == 2
    assert not projection_fixture.output.exists()


@pytest.mark.parametrize("target", ["index", "head", "tag"])
def test_source_boundary_holds_git_state_after_terminal_clean_proof(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    original_git = projection._git  # noqa: SLF001
    head_reads = 0

    def mutate_git_state_after_terminal_clean(git, repo, *arguments, **kwargs):
        nonlocal head_reads
        result = original_git(git, repo, *arguments, **kwargs)
        if arguments == ("rev-parse", "HEAD"):
            head_reads += 1
            if head_reads == 2:
                if target == "index":
                    _git(
                        projection_fixture.repo,
                        projection_fixture.git,
                        "update-index",
                        "--assume-unchanged",
                        projection.GIT_EXECUTION_PATHS[0],
                    )
                elif target == "head":
                    (projection_fixture.repo / ".git/HEAD").write_text(
                        f"{projection_fixture.source_a}\n",
                        encoding="ascii",
                    )
                else:
                    tag_path = (
                        projection_fixture.repo
                        / ".git/refs/tags"
                        / projection_fixture.release_tag
                    )
                    tag_path.write_text(
                        f"{projection_fixture.source_a}\n",
                        encoding="ascii",
                    )
        return result

    monkeypatch.setattr(
        projection,
        "_git",
        mutate_git_state_after_terminal_clean,
    )

    with pytest.raises(
        ValueError,
        match=r"release Git|Git loose ref|HEAD, tag, or clean state changed",
    ):
        projection_fixture.build()

    assert head_reads == 2
    assert not projection_fixture.output.exists()


def test_source_boundary_anchors_repo_across_parent_ancestor_rename(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = projection._git  # noqa: SLF001
    fixture_root = projection_fixture.repo.parent
    moved = fixture_root.with_name("temporarily-moved-fixture-root")
    attacked = False

    def rename_during_git(*args, **kwargs):
        nonlocal attacked
        if not attacked:
            attacked = True
            fixture_root.rename(moved)
            fixture_root.mkdir()
            try:
                return original_git(*args, **kwargs)
            finally:
                fixture_root.rmdir()
                moved.rename(fixture_root)
        return original_git(*args, **kwargs)

    monkeypatch.setattr(projection, "_git", rename_during_git)

    receipt = projection_fixture.build()
    assert receipt.release_b_commit == projection_fixture.release_b
    assert receipt.execution_snapshot_sha256 == (
        projection.EXPECTED_EXECUTION_SNAPSHOT_SHA256
    )


def test_source_boundary_anchors_repo_across_outer_ancestor_rename(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git = projection._git  # noqa: SLF001
    outer_ancestor = projection_fixture.repo.parents[2]
    moved = outer_ancestor.with_name(f"{outer_ancestor.name}-temporarily-moved")
    attacked = False

    def rename_during_git(*args, **kwargs):
        nonlocal attacked
        if not attacked:
            attacked = True
            outer_ancestor.rename(moved)
            outer_ancestor.mkdir()
            try:
                return original_git(*args, **kwargs)
            finally:
                outer_ancestor.rmdir()
                moved.rename(outer_ancestor)
        return original_git(*args, **kwargs)

    monkeypatch.setattr(projection, "_git", rename_during_git)

    receipt = projection_fixture.build()
    assert receipt.release_b_commit == projection_fixture.release_b


def test_failed_ancestor_open_does_not_leak_descriptors(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path("/dev/fd").is_dir():
        pytest.skip("descriptor inventory is unavailable")
    original_identity = projection._require_entry_identity  # noqa: SLF001

    def reject_directory(parent_fd, name, descriptor, *, label):
        original_identity(parent_fd, name, descriptor, label=label)
        if "directory component" in label:
            message = "synthetic directory rejection"
            raise ValueError(message)

    monkeypatch.setattr(projection, "_require_entry_identity", reject_directory)
    before = len(list(Path("/dev/fd").iterdir()))
    for _ in range(20):
        with pytest.raises(ValueError, match="synthetic directory"):
            projection._PinnedPath.open(  # noqa: SLF001
                projection_fixture.seal,
                label="synthetic input",
                max_bytes=projection.MAX_SEALED_COMPLETION_BYTES,
            )
    after = len(list(Path("/dev/fd").iterdir()))
    assert after == before


def test_partial_input_open_failure_does_not_leak_descriptors(
    projection_fixture: ProjectionFixture,
) -> None:
    if not Path("/dev/fd").is_dir():
        pytest.skip("descriptor inventory is unavailable")
    missing = projection_fixture.seal.with_name("missing-seal.json")
    before = len(list(Path("/dev/fd").iterdir()))
    for _ in range(20):
        with pytest.raises(FileNotFoundError):
            projection_fixture.build(sealed_completion_path=missing)
    after = len(list(Path("/dev/fd").iterdir()))
    assert after == before


def test_noncanonical_git_path_does_not_leak_input_descriptors(
    projection_fixture: ProjectionFixture,
) -> None:
    if not Path("/dev/fd").is_dir():
        pytest.skip("descriptor inventory is unavailable")
    missing_git = projection_fixture.git.with_name("missing-git")
    before = len(list(Path("/dev/fd").iterdir()))
    for _ in range(20):
        with pytest.raises(ValueError, match="exactly /usr/bin/git"):
            projection_fixture.build(git_executable=missing_git)
    after = len(list(Path("/dev/fd").iterdir()))
    assert after == before


def test_post_link_publication_failure_does_not_leak_descriptors(
    projection_fixture: ProjectionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path("/dev/fd").is_dir():
        pytest.skip("descriptor inventory is unavailable")
    protected = projection._PinnedPath.open(  # noqa: SLF001
        projection_fixture.attestation,
        label="synthetic protected input",
        max_bytes=projection.MAX_COMPLETION_ATTESTATION_BYTES,
    )
    original = projection._PinnedPath.require_unchanged  # noqa: SLF001

    def reject_protected(self, *, label):
        if label.startswith("protected input"):
            message = "synthetic post-link rejection"
            raise ValueError(message)
        return original(self, label=label)

    monkeypatch.setattr(
        projection._PinnedPath,  # noqa: SLF001
        "require_unchanged",
        reject_protected,
    )
    try:
        before = len(list(Path("/dev/fd").iterdir()))
        for index in range(20):
            output = projection_fixture.output.with_name(f"projection-{index}.json")
            with pytest.raises(ValueError, match="synthetic post-link"):
                projection._publish_no_replace(  # noqa: SLF001
                    output,
                    b"{}\n",
                    protected=(protected,),
                )
        after = len(list(Path("/dev/fd").iterdir()))
        assert after == before
    finally:
        protected.close()


def test_repeated_failed_validation_does_not_leak_descriptors(
    projection_fixture: ProjectionFixture,
) -> None:
    if not Path("/dev/fd").is_dir():
        pytest.skip("descriptor inventory is unavailable")
    receipt = projection_fixture.build()
    before = len(list(Path("/dev/fd").iterdir()))
    for _ in range(20):
        with pytest.raises(ValueError, match="independent digest"):
            projection.validate_k500_authority_projection(
                projection_fixture.output,
                expected_projection_sha256="0" * 64,
                repo_root=projection_fixture.repo,
                git_executable=projection_fixture.git,
            )
    after = len(list(Path("/dev/fd").iterdir()))
    assert after == before
    assert receipt.projection_sha256 == _sha256(projection_fixture.output.read_bytes())


def test_released_source_a_execution_snapshot_reproduces_exactly() -> None:
    repo = Path(__file__).resolve().parents[1]
    git = projection.GIT_EXECUTABLE_PATH
    assert git.is_file()
    source_hashes = {
        path: _sha256(
            subprocess.run(  # noqa: S603
                [
                    git.as_posix(),
                    "cat-file",
                    "blob",
                    f"{projection.SOURCE_A_COMMIT}:{path}",
                ],
                cwd=repo,
                env=projection._git_environment(),  # noqa: SLF001
                check=True,
                capture_output=True,
            ).stdout,
        )
        for path in projection.GIT_EXECUTION_PATHS
    }
    snapshot = {
        **source_hashes,
        projection.GENERATED_VERSION_PATH: projection.GENERATED_VERSION_SHA256,
    }

    assert len(projection.GIT_EXECUTION_PATHS) == 38
    assert len(snapshot) == 39
    assert (
        projection._json_sha256(snapshot)  # noqa: SLF001
        == projection.EXPECTED_EXECUTION_SNAPSHOT_SHA256
    )

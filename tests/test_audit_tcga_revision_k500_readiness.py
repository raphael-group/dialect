"""Tests for the read-only K=500 pre-authority readiness auditor."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from analysis import audit_tcga_revision_k500_readiness as readiness

if TYPE_CHECKING:
    from collections.abc import Sequence


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(path: Path, content: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755 if executable else 0o644)


def _snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    records = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        observed = path.lstat()
        digest = _sha256(path.read_bytes()) if path.is_file() else "directory"
        records[relative] = (digest, observed.st_mode, observed.st_size)
    return records


class FakeGit:
    """Deterministic, read-only Git metadata endpoint for synthetic audits."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        commit: str,
        origin: str,
        files: dict[str, bytes],
        active_head: str | None = None,
        status: bytes = b"",
        remote_commit: str | None = None,
    ) -> None:
        """Build one configurable synthetic Git endpoint."""
        self.commit = commit
        self.origin = origin
        self.files = files
        self.active_head = active_head or commit
        self.status = status
        self.remote_commit = remote_commit or commit
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, _root: Path, arguments: Sequence[str]) -> bytes:
        """Return exact metadata for one expected read-only Git command."""
        args = tuple(arguments)
        self.calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return f"{self.active_head}\n".encode()
        if args[:1] == ("rev-parse",):
            return f"{self.commit}\n".encode()
        if args == ("remote", "get-url", "origin"):
            return f"{self.origin}\n".encode()
        if args == ("status", "--short", "--untracked-files=all"):
            return self.status
        if args[:1] == ("show",):
            _commit, separator, path = args[1].partition(":")
            assert separator == ":"
            return self.files[path]
        if args[:3] == ("ls-remote", "--exit-code", "origin"):
            return f"{self.remote_commit}\t{args[3]}\n".encode()
        raise AssertionError(f"unexpected Git call: {args}")


def _synthetic_policy(
    tmp_path: Path,
) -> tuple[readiness.AuditPolicy, FakeGit, readiness.HostResourceSnapshot]:
    commit = "a" * 40
    origin = "git@example.invalid:group/project.git"
    git_payloads = {
        "src/example.py": b"VALUE = 1\n",
        "scripts/run.sh": b"#!/bin/sh\nexit 0\n",
    }
    for relative, content in git_payloads.items():
        _write(tmp_path / relative, content, executable=relative.endswith(".sh"))
    dependency = b"immutable dependency\n"
    _write(tmp_path / "external/dependency.bin", dependency)
    overlay = b"released runbook overlay\n"
    _write(tmp_path / "notes/released.md", overlay)
    _write(tmp_path / "source_tree/module.py", b"X = 1\n")
    _write(tmp_path / "source_tree/runner.sh", b"#!/bin/sh\n", executable=True)
    raw_root = tmp_path / "raw"
    _write(raw_root / "AAA.maf", b"12345")
    _write(raw_root / "BBB.maf", b"1234567")
    (tmp_path / "population").mkdir()
    (tmp_path / "datahub.git").mkdir()
    executable = tmp_path / "tools/python"
    _write(executable, b"#!/bin/sh\n", executable=True)
    tree_record = readiness._tree_record(  # noqa: SLF001
        tmp_path,
        "source_tree",
        python_only=False,
    )
    policy = readiness.AuditPolicy(
        source_commit=commit,
        local_ref="refs/heads/source-a",
        tracking_ref="refs/remotes/origin/source-a",
        remote_ref="refs/heads/source-a",
        expected_origin=origin,
        git_files=tuple(
            readiness.FileExpectation(path, _sha256(content), len(content))
            for path, content in git_payloads.items()
        ),
        worktree_files=(
            readiness.FileExpectation(
                "external/dependency.bin",
                _sha256(dependency),
                len(dependency),
            ),
        ),
        trees=(
            readiness.TreeExpectation(
                "source_tree",
                str(tree_record["sha256"]),
                int(tree_record["file_count"]),
            ),
        ),
        overlays=(
            readiness.FileExpectation(
                "notes/released.md",
                _sha256(overlay),
                len(overlay),
            ),
        ),
        required_directories=("raw", "population", "datahub.git"),
        required_executables=(executable.as_posix(),),
        pre_authority_absent_paths=(
            "output/approval",
            "output/canonical",
            "output/providers",
            "output/k500",
        ),
        raw_maf_root="raw",
        output_filesystem_probe="output/canonical",
        cohorts=("AAA", "BBB"),
    )
    raw_bytes = 12
    required_disk = raw_bytes * 10 + 2 * 1024**3
    snapshot = readiness.HostResourceSnapshot(
        measured_at_utc="2026-08-31T05:00:00+00:00",
        logical_cores=16,
        load_average_1m=1.0,
        total_memory_bytes=32 * 1024**3,
        available_memory_bytes=20 * 1024**3,
        free_disk_bytes=max(required_disk + 1024, 20 * 1024**3),
        cpu_source="synthetic aggregate load",
        memory_source="synthetic aggregate memory",
    )
    return policy, FakeGit(commit=commit, origin=origin, files=git_payloads), snapshot


def _runtime_snapshot(root: Path) -> readiness.RuntimeSnapshot:
    return readiness.RuntimeSnapshot(
        executable=readiness.PRODUCTION_PYTHON,
        argv0=(root / readiness.PRODUCTION_SCRIPT).as_posix(),
        isolated=True,
        bytecode_writes_disabled=True,
    )


def test_complete_synthetic_audit_passes_without_mutating_any_file(
    tmp_path: Path,
) -> None:
    """A closed mechanical boundary passes but never claims authority."""
    policy, git, snapshot = _synthetic_policy(tmp_path)
    before = _snapshot(tmp_path)

    report = readiness.audit_readiness(
        tmp_path,
        policy=policy,
        check_live_remote=True,
        git_runner=git,
        resource_snapshot=snapshot,
        runtime_snapshot=_runtime_snapshot(tmp_path),
    )

    assert report["mechanical_preconditions"] == {
        "passed": True,
        "released_overlay_integrity": True,
        "source_a_preserved": True,
        "current_release_b_worktree_clean": True,
        "mutation_free_runtime": True,
        "external_dependencies_present": True,
        "required_inputs_and_absent_destinations": True,
        "host_resource_gates": True,
        "canonical_storage_gate": True,
    }
    assert report["launch_tree_transition"]["passed"] is True
    assert report["human_authority"] == {
        "evaluated": False,
        "status": "external-unverified",
        "required_first_party_approvers": [
            "Benjamin J. Raphael",
            "Uthsav Chitra",
        ],
        "authorizes_execution": False,
    }
    assert report["scientific_scope"] == {
        "result_rows_opened": False,
        "fitting_launched": False,
        "roots_created_or_modified": False,
        "files_deleted": False,
        "authorizes_materialization": False,
        "authorizes_fitting": False,
        "authorizes_inspection": False,
    }
    assert report["canonical_storage"]["contents_opened"] is False
    assert report["runtime_boundary"]["passed"] is True
    assert report["runtime_boundary"]["authorizes_execution"] is False
    assert report["runtime_boundary"][
        "noncompliant_invocation_may_create_ignored_python_bytecode"
    ] is True
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("mutation", "failed_key"),
    [
        ("dirty", "current_release_b_worktree_clean"),
        ("remote-drift", "source_a_preserved"),
        ("overlay-drift", "released_overlay_integrity"),
        ("commit-source-drift", "source_a_preserved"),
        ("external-drift", "external_dependencies_present"),
        ("destination-exists", "required_inputs_and_absent_destinations"),
        ("disk-deficit", "canonical_storage_gate"),
        ("host-load", "host_resource_gates"),
        ("runtime", "mutation_free_runtime"),
    ],
)
def test_each_mechanical_boundary_fails_closed(
    tmp_path: Path,
    mutation: str,
    failed_key: str,
) -> None:
    """Every independent drift leaves the report non-authorizing and red."""
    policy, git, snapshot = _synthetic_policy(tmp_path)
    runtime = _runtime_snapshot(tmp_path)
    if mutation == "dirty":
        git.status = b"?? unexpected.txt\n"
    elif mutation == "remote-drift":
        git.remote_commit = "c" * 40
    elif mutation == "overlay-drift":
        _write(tmp_path / "notes/released.md", b"changed\n")
    elif mutation == "commit-source-drift":
        git.files["src/example.py"] = b"VALUE = 2\n"
    elif mutation == "external-drift":
        _write(tmp_path / "external/dependency.bin", b"changed\n")
    elif mutation == "destination-exists":
        (tmp_path / "output/approval").mkdir(parents=True)
    elif mutation == "disk-deficit":
        snapshot = replace(snapshot, free_disk_bytes=1024)
    elif mutation == "host-load":
        snapshot = replace(snapshot, load_average_1m=7.0)
    elif mutation == "runtime":
        runtime = replace(
            runtime,
            isolated=False,
            bytecode_writes_disabled=False,
        )
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(mutation)

    report = readiness.audit_readiness(
        tmp_path,
        policy=policy,
        check_live_remote=True,
        git_runner=git,
        resource_snapshot=snapshot,
        runtime_snapshot=runtime,
    )

    assert report["mechanical_preconditions"]["passed"] is False
    assert report["mechanical_preconditions"][failed_key] is False
    assert report["human_authority"]["authorizes_execution"] is False
    assert report["scientific_scope"]["result_rows_opened"] is False


@pytest.mark.parametrize("mutation", ["active-b", "source-drift", "tree-drift"])
def test_deferred_launch_tree_drift_does_not_overstate_launch_readiness(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Release-B auditing separates pre-authority readiness from checkout A."""
    policy, git, snapshot = _synthetic_policy(tmp_path)
    if mutation == "active-b":
        git.active_head = "b" * 40
    elif mutation == "source-drift":
        _write(tmp_path / "src/example.py", b"VALUE = 2\n")
    elif mutation == "tree-drift":
        _write(tmp_path / "source_tree/module.py", b"X = 2\n")

    report = readiness.audit_readiness(
        tmp_path,
        policy=policy,
        check_live_remote=True,
        git_runner=git,
        resource_snapshot=snapshot,
        runtime_snapshot=_runtime_snapshot(tmp_path),
    )

    assert report["mechanical_preconditions"]["passed"] is True
    assert report["git"]["source_a_preserved"] is True
    assert report["git"]["active_source_a"] is (mutation != "active-b")
    assert report["launch_tree_transition"] == {
        "passed": False,
        "required_after_authority_and_storage": True,
        "required_source_commit": policy.source_commit,
        "status": "deferred-source-a-checkout-and-fresh-readback-required",
        "authorizes_execution": False,
    }


def test_omitted_live_remote_is_an_explicit_stop(
    tmp_path: Path,
) -> None:
    """A cached local ref never substitutes for the mandatory live readback."""
    policy, git, snapshot = _synthetic_policy(tmp_path)

    report = readiness.audit_readiness(
        tmp_path,
        policy=policy,
        check_live_remote=False,
        git_runner=git,
        resource_snapshot=snapshot,
        runtime_snapshot=_runtime_snapshot(tmp_path),
    )

    assert report["git"]["live_remote"] == {
        "checked": False,
        "passed": False,
        "reason": "live remote check omitted; rerun with --check-live-remote",
    }
    assert report["git"]["source_a_preserved"] is False
    assert report["mechanical_preconditions"]["passed"] is False
    assert not any(call[:1] == ("ls-remote",) for call in git.calls)


@pytest.mark.parametrize(
    "field",
    [
        "git_files",
        "worktree_files",
        "trees",
        "overlays",
        "required_directories",
        "pre_authority_absent_paths",
        "raw_maf_root",
        "output_filesystem_probe",
    ],
)
def test_every_policy_workspace_path_is_validated_before_git(
    tmp_path: Path,
    field: str,
) -> None:
    """No malformed policy path can escape the repository or reach Git."""
    policy, git, snapshot = _synthetic_policy(tmp_path)
    escaping = "../outside"
    if field == "git_files":
        policy = replace(
            policy,
            git_files=(replace(policy.git_files[0], path=escaping),),
        )
    elif field == "worktree_files":
        policy = replace(
            policy,
            worktree_files=(replace(policy.worktree_files[0], path=escaping),),
        )
    elif field == "trees":
        policy = replace(
            policy,
            trees=(replace(policy.trees[0], path=escaping),),
        )
    elif field == "overlays":
        policy = replace(
            policy,
            overlays=(replace(policy.overlays[0], path=escaping),),
        )
    elif field == "required_directories":
        policy = replace(policy, required_directories=(escaping,))
    elif field == "pre_authority_absent_paths":
        policy = replace(policy, pre_authority_absent_paths=(escaping,))
    elif field == "raw_maf_root":
        policy = replace(policy, raw_maf_root=escaping)
    elif field == "output_filesystem_probe":
        policy = replace(policy, output_filesystem_probe=escaping)
    else:  # pragma: no cover - protects the parametrization itself
        raise AssertionError(field)

    with pytest.raises(readiness.ReadinessAuditError, match="canonical workspace"):
        readiness.audit_readiness(
            tmp_path,
            policy=policy,
            check_live_remote=True,
            git_runner=git,
            resource_snapshot=snapshot,
            runtime_snapshot=_runtime_snapshot(tmp_path),
        )

    assert git.calls == []


@pytest.mark.parametrize(
    "relative",
    ["", "/absolute", "a/../outside", "a/./b", "a//b", "a/", "a\\b"],
)
def test_workspace_paths_must_be_canonical(relative: str) -> None:
    """Absolute, escaping, and merely equivalent spellings are all rejected."""
    with pytest.raises(readiness.ReadinessAuditError):
        readiness._canonical_workspace_path(  # noqa: SLF001
            relative,
            label="synthetic",
        )


def test_ancestor_symlinks_cannot_satisfy_workspace_checks(
    tmp_path: Path,
) -> None:
    """Final-component O_NOFOLLOW is not allowed to hide a linked ancestor."""
    outside = tmp_path / "outside"
    overlay = b"released bytes\n"
    _write(outside / "released.md", overlay)
    _write(outside / "tree/module.py", b"X = 1\n")
    _write(outside / "raw/AAA.maf", b"123")
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    file_record = readiness._file_check(  # noqa: SLF001
        tmp_path,
        readiness.FileExpectation(
            "linked/released.md",
            _sha256(overlay),
            len(overlay),
        ),
    )
    tree_record = readiness._tree_check(  # noqa: SLF001
        tmp_path,
        readiness.TreeExpectation("linked/tree", "0" * 64, 1),
    )
    policy = readiness.AuditPolicy(
        source_commit="a" * 40,
        local_ref="refs/heads/source-a",
        tracking_ref="refs/remotes/origin/source-a",
        remote_ref="refs/heads/source-a",
        expected_origin="git@example.invalid:group/project.git",
        git_files=(),
        worktree_files=(),
        trees=(),
        overlays=(),
        required_directories=(),
        required_executables=(),
        pre_authority_absent_paths=(),
        raw_maf_root="linked/raw",
        output_filesystem_probe="output/canonical",
        cohorts=("AAA",),
    )
    raw_record = readiness._raw_maf_storage_check(  # noqa: SLF001
        tmp_path,
        policy,
        free_disk_bytes=10 * 1024**3,
    )
    directory_record = readiness._required_directory_check(  # noqa: SLF001
        tmp_path,
        "linked/raw",
    )

    assert file_record["passed"] is False
    assert tree_record["passed"] is False
    assert raw_record["passed"] is False
    assert directory_record["passed"] is False
    assert "ancestor linked" in str(file_record["reason"])
    assert "ancestor linked" in str(tree_record["reason"])
    assert "ancestor linked" in str(raw_record["reason"])
    assert "ancestor linked" in str(directory_record["reason"])
    with pytest.raises(readiness.ReadinessAuditError, match="real directory"):
        readiness._workspace_free_disk_bytes(  # noqa: SLF001
            tmp_path,
            "linked/output",
        )


def test_raw_maf_gate_never_opens_maf_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical storage arithmetic is metadata-only over the exact family."""
    policy, _git, snapshot = _synthetic_policy(tmp_path)
    original_open = os.open

    def guarded_open(
        path: str | bytes | os.PathLike[str],
        *args: object,
        **kwargs: object,
    ) -> int:
        if os.fspath(path).endswith(".maf"):
            raise AssertionError("raw MAF content was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    record = readiness._raw_maf_storage_check(  # noqa: SLF001
        tmp_path,
        policy,
        free_disk_bytes=snapshot.free_disk_bytes,
    )

    assert record["passed"] is True
    assert record["raw_bytes"] == 12
    assert record["required_free_bytes"] == 12 * 10 + 2 * 1024**3
    assert record["contents_opened"] is False


def test_raw_maf_gate_rejects_extra_or_linked_family_members(
    tmp_path: Path,
) -> None:
    """Name drift and non-private input files fail before materialization."""
    policy, _git, snapshot = _synthetic_policy(tmp_path)
    _write(tmp_path / "raw/EXTRA.maf", b"extra")
    with (tmp_path / "raw/AAA.maf").open("rb") as source:
        assert source.read() == b"12345"
    os.link(tmp_path / "raw/BBB.maf", tmp_path / "raw/BBB-copy")

    record = readiness._raw_maf_storage_check(  # noqa: SLF001
        tmp_path,
        policy,
        free_disk_bytes=snapshot.free_disk_bytes,
    )

    assert record["passed"] is False
    assert record["names_match_exact_family"] is False
    assert next(item for item in record["files"] if item["name"] == "BBB.maf")[
        "passed"
    ] is False


def test_exact_output_entry_is_never_traversed_or_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a result-looking subtree is inspected only at its root entry."""
    output = tmp_path / "output/k500"
    result = output / "tasks/CHOL/cbase/pairwise_interaction_results.csv"
    _write(result, b"forbidden row payload\n")
    original_open = os.open
    original_scandir = os.scandir

    def guarded_open(
        path: str | bytes | os.PathLike[str],
        *args: object,
        **kwargs: object,
    ) -> int:
        if "pairwise_interaction_results.csv" in os.fspath(path):
            raise AssertionError("association result was opened")
        return original_open(path, *args, **kwargs)

    def guarded_scandir(
        path: str | bytes | os.PathLike[str],
    ) -> os.ScandirIterator[str]:
        if Path(path).is_relative_to(output):
            raise AssertionError("K500 output tree was traversed")
        return original_scandir(path)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "scandir", guarded_scandir)

    state = readiness._path_state(tmp_path, "output/k500")  # noqa: SLF001

    assert state == {
        "exists": True,
        "kind": "directory",
        "path": output.as_posix(),
    }


def test_stable_file_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    """Released overlay hashes cannot be redirected or multiply linked."""
    original = tmp_path / "original"
    _write(original, b"content")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(original)

    with pytest.raises(readiness.ReadinessAuditError):
        readiness._stable_file_record(tmp_path, "symlink")  # noqa: SLF001

    hardlink = tmp_path / "hardlink"
    os.link(original, hardlink)
    with pytest.raises(readiness.ReadinessAuditError):
        readiness._stable_file_record(tmp_path, "original")  # noqa: SLF001


def test_tree_digest_uses_provider_framing_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    """The source-tree receipt matches the frozen framing contract exactly."""
    payload = b"abc"
    _write(tmp_path / "tree/a.py", payload)
    encoded = b"a.py"
    expected = hashlib.sha256(
        len(encoded).to_bytes(8, "big")
        + encoded
        + (0o400).to_bytes(8, "big")
        + len(payload).to_bytes(8, "big")
        + payload,
    ).hexdigest()

    record = readiness._tree_record(  # noqa: SLF001
        tmp_path,
        "tree",
        python_only=False,
    )

    assert record == {
        "contract": "u64be-path-mode-content-v1",
        "file_count": 1,
        "sha256": expected,
    }
    (tmp_path / "tree/__pycache__").symlink_to(tmp_path / "tree/a.py")
    with pytest.raises(readiness.ReadinessAuditError):
        readiness._tree_record(  # noqa: SLF001
            tmp_path,
            "tree",
            python_only=False,
        )


@pytest.mark.parametrize(
    ("jobs", "passed"),
    [(1, True), (3, True), (4, False)],
)
def test_resource_gate_matches_strict_half_host_policy(
    jobs: int,
    passed: bool,  # noqa: FBT001
) -> None:
    """Concurrency never exceeds three or the strict half-core boundary."""
    snapshot = readiness.HostResourceSnapshot(
        measured_at_utc="2026-08-31T05:00:00+00:00",
        logical_cores=14,
        load_average_1m=3.9,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        free_disk_bytes=20 * 1024**3,
        cpu_source="aggregate load",
        memory_source="aggregate memory",
    )

    record = readiness.evaluate_host_resource_gate(snapshot, jobs=jobs)

    assert record["passed"] is passed
    assert record["safe_job_cap"] == 3
    assert record["strict_half_core_limit"] == 7
    assert math.isclose(
        float(record["required_by_fraction_bytes"]),
        math.ceil(24 * 1024**3 * 0.33),
    )


def test_three_jobs_fail_at_equality_with_half_host() -> None:
    """The load rule is strict; equality is not rounded into a pass."""
    snapshot = readiness.HostResourceSnapshot(
        measured_at_utc="2026-08-31T05:00:00+00:00",
        logical_cores=14,
        load_average_1m=4.0,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        free_disk_bytes=20 * 1024**3,
        cpu_source="aggregate load",
        memory_source="aggregate memory",
    )

    record = readiness.evaluate_host_resource_gate(snapshot, jobs=3)

    assert record["passed"] is False
    assert record["projected_load_with_planned_jobs"] == 7


@pytest.mark.parametrize(
    ("load", "state"),
    [
        (math.nan, "nan"),
        (math.inf, "positive-infinity"),
        (-math.inf, "negative-infinity"),
    ],
)
def test_nonfinite_resource_readback_remains_canonical_json(
    load: float,
    state: str,
) -> None:
    """A corrupt aggregate load fails closed without breaking JSON emission."""
    snapshot = readiness.HostResourceSnapshot(
        measured_at_utc="2026-08-31T05:00:00+00:00",
        logical_cores=14,
        load_average_1m=load,
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        free_disk_bytes=20 * 1024**3,
        cpu_source="aggregate load",
        memory_source="aggregate memory",
    )

    record = readiness._resource_snapshot_record(snapshot)  # noqa: SLF001

    assert record["load_average_1m"] is None
    assert record["load_average_1m_state"] == state
    json.dumps(record, allow_nan=False)
    assert readiness.evaluate_host_resource_gate(snapshot, jobs=1)["passed"] is False


def test_isolated_no_bytecode_cli_prefix_creates_no_bytes(
    tmp_path: Path,
) -> None:
    """The mandated interpreter prefix is mutation-free from a pristine copy."""
    source = Path(readiness.__file__).resolve()
    copy_root = tmp_path / "copy"
    init_bytes = (source.parent / "__init__.py").read_bytes()
    _write(copy_root / "analysis/__init__.py", init_bytes)
    _write(copy_root / readiness.PRODUCTION_SCRIPT, source.read_bytes())
    before = _snapshot(copy_root)
    command = [*readiness.PRODUCTION_INVOCATION[:-1], "--help"]

    completed = subprocess.run(  # noqa: S603
        command,
        cwd=copy_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert readiness.PRODUCTION_INVOCATION[1:3] == ("-I", "-B")
    assert _snapshot(copy_root) == before
    assert not list(copy_root.rglob("*.pyc"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("executable", "/usr/bin/python3"),
        ("argv0", "analysis/another_auditor.py"),
        ("isolated", False),
        ("bytecode_writes_disabled", False),
    ],
)
def test_runtime_boundary_requires_every_production_property(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """No partial approximation of the production interpreter contract passes."""
    snapshot = replace(_runtime_snapshot(tmp_path), **{field: value})

    record = readiness._runtime_boundary_record(  # noqa: SLF001
        tmp_path,
        snapshot,
    )

    assert record["passed"] is False
    assert record["authorizes_execution"] is False


def test_main_prints_json_and_propagates_red_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The CLI has no output-path option and returns nonzero for a red report."""
    report = {
        "mechanical_preconditions": {"passed": False},
        "human_authority": {"authorizes_execution": False},
    }

    def fake_audit(
        _root: Path,
        *,
        check_live_remote: bool,
    ) -> dict[str, object]:
        assert check_live_remote is False
        return report

    monkeypatch.setattr(readiness, "audit_readiness", fake_audit)

    assert readiness.main(("--repo-root", tmp_path.as_posix())) == 1
    assert json.loads(capsys.readouterr().out) == report


def test_main_converts_audit_failure_to_non_authorizing_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """An unreadable prerequisite cannot produce a traceback-only ambiguity."""

    def fail_audit(
        _root: Path,
        *,
        check_live_remote: bool,
    ) -> dict[str, object]:
        assert check_live_remote is True
        raise readiness.ReadinessAuditError("synthetic read failure")

    monkeypatch.setattr(readiness, "audit_readiness", fail_audit)

    exit_status = readiness.main(
        (
            "--repo-root",
            tmp_path.as_posix(),
            "--check-live-remote",
        ),
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_status == 2
    assert report["mechanical_preconditions"]["passed"] is False
    assert report["human_authority"]["authorizes_execution"] is False
    assert report["scientific_scope"]["result_rows_opened"] is False
    assert report["audit_error"]["type"] == "ReadinessAuditError"


def test_production_policy_pins_every_released_overlay() -> None:
    """The tracked command cannot silently omit one operational overlay."""
    observed = {item.path: item.sha256 for item in readiness.PRODUCTION_POLICY.overlays}
    assert observed == {
        "research/notes/76_current_head_post_authority_k500_runbook_candidate.md": (
            "933a3e9db8b27a80de5f2b6f62204200640a8bf6fde1187e5146516af8500a60"
        ),
        "research/notes/79_current_head_k500_runbook_release_record.md": (
            "15ae1e5cf82cc9123f671ede1504ddc6151ab4e2dcce9d7ff74b2173dd5bb872"
        ),
        (
            "research/notes/81_current_head_k500_completion_"
            "attestation_addendum_candidate.md"
        ): (
            "c263578ef2f21fd206a3c4745c2ed7fe98d3271b837c19944af6cc77b9eb96d1"
        ),
        "research/notes/82_k500_completion_attestation_addendum_release_record.md": (
            "4faaeaca9ddfd90ef8a3701233e5789d5379b75def056e6dd236f82654631626"
        ),
        (
            "research/notes/85_canonical_materialization_same_uid_"
            "lease_addendum_candidate.md"
        ): (
            "3e49c3a1856c491cabb8c6384de5c8dda61a5bf90326727daaf7e7a4788114f3"
        ),
        (
            "research/notes/88_canonical_materialization_same_uid_"
            "lease_addendum_release_record.md"
        ): (
            "dc049df313fa6a2d0e588064639dad5808e20f7101ab208aaee1645bbacfa429"
        ),
        "research/notes/materialize_tcga_revision_inputs_with_lease.py": (
            "321c60cb215b698215d2cd4da030926f218cd166310e348e9eba57172e9083af"
        ),
        "research/notes/attest_k500_completion.py": (
            "238a7131a0e3aeb41928939840476d333d5ccfb19a081c19decdb6ea2a4d9de2"
        ),
    }


def test_production_policy_preserves_exact_source_and_family() -> None:
    """Source A and the complete ordered TCGA family are immutable constants."""
    policy = readiness.PRODUCTION_POLICY
    assert policy.source_commit == readiness.SOURCE_A_COMMIT
    assert policy.cohorts == readiness.TCGA_COHORTS
    assert len(policy.cohorts) == 32
    assert policy.cohorts[4] == "CHOL"
    assert policy.pre_authority_absent_paths == (
        "output/tcga_revision_materialize_approval_2026-08-29_v12",
        "output/tcga_revision_canonical_2026-08-29_v13",
        "output/tcga_revision_providers_2026-08-29_v13",
        "output/tcga_revision_k500_2026-08-30_v1",
    )

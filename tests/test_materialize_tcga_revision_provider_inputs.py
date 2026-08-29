import copy
import hashlib
import os
import re
import subprocess
import sys
import threading
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import materialize_tcga_revision_provider_inputs as provider

# Focused contract tests intentionally exercise fail-closed private seams.
# ruff: noqa: SLF001


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _write(path: Path, contents: bytes = b"x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _record(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _digest(path.read_bytes()),
    }


def _authority_record(path: Path, character: str) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": 1,
        "sha256": character * 64,
    }


def _paths(tmp_path: Path) -> provider.ProviderPaths:
    work = tmp_path / ".provider.provider-work"
    return provider.ProviderPaths(
        repo_root=tmp_path / "repo",
        canonical_input_root=tmp_path / "canonical",
        approval_manifest=tmp_path / "approval.json",
        output_root=tmp_path / "provider",
        work_root=work,
        cohort_root=work / "cohorts",
        mutsig_root=work / "mutsig",
        cbase_inputs=tmp_path / "repo/external/CBaSE",
        dig_results=tmp_path / "repo/external/DIG/results.txt",
        pipeline=tmp_path / "repo/scripts/run_cohort_pipeline.sh",
        mutsig_runner=tmp_path / "repo/scripts/run_mutsig_octave.sh",
        mutsig_patch=tmp_path / "repo/external/mutsig.patch",
    )


def _synthetic_published_authority(
    paths: provider.ProviderPaths,
    hashes: provider.IndependentHashes,
    canonical_manifest: dict[str, object],
) -> dict[str, object]:
    source_files = {
        "orchestrator": Path(provider.__file__).resolve(),
        "cohort_pipeline": paths.pipeline,
        "mutsig_runner": paths.mutsig_runner,
        "mutsig_patch": paths.mutsig_patch,
        "nice_executable": provider.NICE_EXECUTABLE,
        "bash_executable": provider.BASH_EXECUTABLE,
        "git_executable": Path("/usr/bin/git"),
    }
    characters = dict(zip(source_files, "1234567", strict=True))
    sources = {
        name: _authority_record(path, characters[name])
        for name, path in source_files.items()
    }
    python_versions = {
        "python": "3.12.13",
        "numpy": "2.1.1",
        "pandas": "2.2.3",
        "scipy": "1.14.1",
    }
    python_executable = provider.CHILD_PYTHON_EXECUTABLE
    octave_id = "GNU Octave, version 9.1.0"
    java_id = 'openjdk version "11.0.24"'

    def distribution_record(name: str, character: str) -> dict[str, object]:
        installed = _authority_record(
            Path(f"/opt/anaconda3/envs/dialect/lib/{name}.py"),
            character,
        )
        files = [
            {
                "record_path": f"{name}.py",
                "record_sha256": character * 64,
                "record_bytes": 1,
                "installed": installed,
            },
        ]
        return {
            "name": name,
            "version": python_versions[name],
            "record": _authority_record(
                Path(
                    f"/opt/anaconda3/envs/dialect/lib/{name}.dist-info/RECORD",
                ),
                character,
            ),
            "record_relative_path": f"{name}.dist-info/RECORD",
            "file_count": 1,
            "files_sha256": _digest(provider._canonical_json(files)),
            "files": files,
            "native_files": [],
        }

    dialect_child = {
        "launcher": python_executable.as_posix(),
        "entrypoint_shebang": f"#!{python_executable}",
        "python_executable": _authority_record(python_executable, "8"),
        "dialect_entrypoint": _authority_record(provider.DIALECT_EXECUTABLE, "9"),
        "dialect_import": _authority_record(
            paths.repo_root / "src/dialect/__init__.py",
            "a",
        ),
        "dialect_tree_hash_contract": provider.TREE_HASH_CONTRACT,
        "dialect_tree_sha256": "b" * 64,
        "imported_modules": {
            name: _authority_record(
                Path(f"/opt/anaconda3/envs/dialect/lib/{name}/__init__.py"),
                character,
            )
            for name, character in zip(
                ("numpy", "pandas", "scipy"),
                "cde",
                strict=True,
            )
        },
        "distributions": {
            name: distribution_record(name, character)
            for name, character in zip(
                ("numpy", "pandas", "scipy"),
                "cde",
                strict=True,
            )
        },
        "versions": python_versions,
    }
    dialect_child["runtime_sha256"] = _digest(
        provider._canonical_json(dialect_child),
    )
    mutsig_runtime = {
        "octave": _authority_record(Path("/opt/homebrew/bin/octave"), "f"),
        "octave_id": octave_id,
        "java_home": provider.MUTSIG_JAVA_HOME,
        "java_executable": _authority_record(
            Path(provider.MUTSIG_JAVA_HOME) / "bin/java",
            "0",
        ),
        "java_id": java_id,
    }
    mutsig_runtime["runtime_sha256"] = _digest(
        provider._canonical_json(mutsig_runtime),
    )
    sources.update(
        {
            "dialect_python_tree_sha256": dialect_child["dialect_tree_sha256"],
            "python_runtime_sha256": dialect_child["runtime_sha256"],
        },
    )
    providers = {
        "cbase": {
            "inputs_root": paths.cbase_inputs.as_posix(),
            "inputs_tree_sha256": hashes.cbase_inputs_tree,
            "expected_inputs_tree_sha256": hashes.cbase_inputs_tree,
        },
        "dig": {
            "results": {
                "path": paths.dig_results.as_posix(),
                "bytes": 1,
                "sha256": hashes.dig_results,
            },
            "expected_results_sha256": hashes.dig_results,
        },
        "mutsig": {
            "upstream_commit": provider.MUTSIG_UPSTREAM_COMMIT,
            "source_tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "source_tree_sha256": "f" * 64,
            "source_file_count": 1,
            "patch_sha256": sources["mutsig_patch"]["sha256"],
            "runner_sha256": sources["mutsig_runner"]["sha256"],
            "runtime": mutsig_runtime,
        },
        "dialect_child": dialect_child,
    }
    return {
        "schema_version": provider.SCHEMA_VERSION,
        "contract": provider.PROVIDER_INPUT_CONTRACT,
        "intended_output_root": paths.output_root.as_posix(),
        "authority": provider._input_authority_record(
            paths,
            hashes,
            canonical_manifest,
        ),
        "sources": sources,
        "providers": providers,
        "execution": provider._execution_contract(),
        "scope": provider._scope_contract(),
    }


def _synthetic_published_bundle(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "provider"
    _initialize_layout(root)
    canonical_root = tmp_path / "canonical"
    approval_path = _write(tmp_path / "approval.json", b'{"signed":true}\n')
    canonical_manifest_path = _write(
        canonical_root / "input_manifest.json",
        b'{"fixture":"canonical"}\n',
    )
    hashes = provider.IndependentHashes(
        approval=_digest(approval_path.read_bytes()),
        canonical_input_manifest=_digest(canonical_manifest_path.read_bytes()),
        cbase_inputs_tree="c" * 64,
        dig_results="d" * 64,
    )
    canonical_manifest = {
        "contract": "synthetic-canonical-input-v1",
        "authority": {
            "approval": {
                "authorized_stage": "materialize-final-inputs",
                "manifest_sha256": hashes.approval,
                "decision_digests": {"D1": "1" * 64, "D2": "2" * 64},
            },
        },
    }
    paths = provider._provider_paths(
        canonical_root,
        approval_path,
        root,
        repo_root=None,
    )
    work_authority = _synthetic_published_authority(
        paths,
        hashes,
        canonical_manifest,
    )
    manifest = {
        "schema_version": provider.SCHEMA_VERSION,
        "contract": provider.PROVIDER_INPUT_CONTRACT,
        "completed_at_utc": "2026-08-28T00:00:00+00:00",
        "cohorts": list(provider.TCGA_COHORTS),
        "cohort_count": len(provider.TCGA_COHORTS),
        "roots": {"cohorts": "cohorts", "mutsig": "mutsig"},
        "authority": work_authority["authority"],
        "sources": work_authority["sources"],
        "providers": work_authority["providers"],
        "execution": work_authority["execution"],
        "scope": work_authority["scope"],
        "cohort_provider_receipts": [
            {"cohort": cohort} for cohort in provider.TCGA_COHORTS
        ],
        "inventory": {},
    }
    bundle = SimpleNamespace(
        root=root,
        canonical_root=canonical_root,
        approval_path=approval_path,
        hashes=hashes,
        canonical_manifest=canonical_manifest,
        work_authority=work_authority,
        manifest=manifest,
    )
    _republish_synthetic_bundle(bundle, work_authority)
    return bundle


def _republish_synthetic_bundle(
    bundle: SimpleNamespace,
    work_authority: dict[str, object],
) -> str:
    authority_path = bundle.root / provider.WORK_AUTHORITY_PATH
    manifest_path = bundle.root / provider.ROOT_MANIFEST_NAME
    for path in (authority_path, manifest_path):
        if path.exists():
            path.unlink()
    provider._write_json_atomic(authority_path, work_authority, mode=0o444)
    manifest = copy.deepcopy(bundle.manifest)
    for key in ("authority", "sources", "providers", "execution", "scope"):
        manifest[key] = work_authority[key]
    inventory = provider._require_allowed_work_inventory(bundle.root)
    manifest["inventory"] = {
        "directories": inventory["directories"],
        "files": [
            provider._file_record(
                authority_path,
                display_path=provider.WORK_AUTHORITY_PATH.as_posix(),
            ),
        ],
        "execution_snapshot": {
            "root": "_orchestration/execution-snapshot-" + "e" * 64,
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": "e" * 64,
            "file_count": 1,
            "directory_count": 1,
            "individual_file_receipts_omitted": True,
        },
        "root_manifest_excluded_from_self_inventory": provider.ROOT_MANIFEST_NAME,
    }
    provider._write_json_atomic(manifest_path, manifest, mode=0o444)
    bundle.work_authority = work_authority
    bundle.manifest = manifest
    return _digest(manifest_path.read_bytes())


def _context(
    tmp_path: Path,
    *,
    bindings: dict[str, dict[str, object]] | None = None,
) -> provider.ProviderContext:
    paths = _paths(tmp_path)
    hashes = provider.IndependentHashes(
        approval="a" * 64,
        canonical_input_manifest="b" * 64,
        cbase_inputs_tree="c" * 64,
        dig_results="d" * 64,
    )
    runtime_record = {
        "path": (tmp_path / "runtime/tool").as_posix(),
        "bytes": 1,
        "sha256": "7" * 64,
    }
    authority = {
        "authority": {},
        "sources": {
            "cohort_pipeline": {"sha256": "1" * 64},
            "mutsig_patch": {"sha256": "2" * 64},
            "mutsig_runner": {"sha256": "3" * 64},
            "dialect_python_tree_sha256": "4" * 64,
            "python_runtime_sha256": "5" * 64,
            "bash_executable": runtime_record,
            "git_executable": runtime_record,
            "nice_executable": runtime_record,
        },
        "providers": {
            "mutsig": {
                "source_tree_sha256": "8" * 64,
                "source_file_count": 1,
                "runtime": {
                    "octave": runtime_record,
                    "octave_id": "GNU Octave, version 9.1.0",
                    "java_home": "/runtime/java",
                    "java_executable": runtime_record,
                    "java_id": "openjdk version 11",
                    "runtime_sha256": "6" * 64,
                },
            },
            "dialect_child": {
                "python_executable": runtime_record,
                "dialect_tree_sha256": "4" * 64,
                "runtime_sha256": "5" * 64,
            },
        },
        "execution": {},
        "scope": {},
    }
    return provider.ProviderContext(
        paths=paths,
        hashes=hashes,
        canonical_manifest={},
        bindings={} if bindings is None else bindings,
        authority=authority,
    )


def _initialize_layout(root: Path) -> None:
    root.mkdir()
    for name in (
        "_orchestration",
        "attempts",
        "cohorts",
        "mutsig",
        "resource_readbacks",
    ):
        (root / name).mkdir()


def _synthetic_full_acceptance_manifest() -> dict[str, object]:
    return {
        "authority": {},
        "sources": {},
        "providers": {},
        "execution": {},
        "scope": {},
        "cohort_provider_receipts": [],
        "inventory": {
            "execution_snapshot": {
                "root": "_orchestration/execution-snapshot-" + "a" * 64,
                "tree_sha256": "a" * 64,
            },
        },
    }


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":-Infinity}\n',
        b'{"x":1e999}\n',
        b'{"x":' + (b"9" * 5000) + b"}\n",
        b'{"x":"\\ud800"}\n',
        b'{"x":1,"x":2}\n',
        b'{"x":1}',
        b'{"x":1}\n\n',
        b'{ "x":1}\n',
        b'{"z":1,"a":2}\n',
    ],
)
def test_provider_json_parser_rejects_ambiguous_or_noncanonical_authority(
    tmp_path: Path,
    raw: bytes,
) -> None:
    manifest = _write(tmp_path / "authority.json", raw)
    with pytest.raises(provider.ProviderInputError):
        provider._read_json(manifest)


def test_provider_json_parser_accepts_exact_finite_canonical_object(
    tmp_path: Path,
) -> None:
    manifest = _write(
        tmp_path / "authority.json",
        b'{"a":[true,null,1.5],"z":"\\u00e9"}\n',
    )
    assert provider._read_json(manifest) == {
        "a": [True, None, 1.5],
        "z": "\u00e9",
    }


def test_provider_json_hash_and_parse_share_one_descriptor_during_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b'{"authority":"original"}\n'
    manifest = _write(tmp_path / "authority.json", original)
    replacement = _write(
        tmp_path / "replacement.json",
        b'{"authority":"replacement"}\n',
    )
    secure_open = provider._open_regular_fd
    raced = False

    def open_then_replace(path: Path, *, label: str) -> int:
        nonlocal raced
        descriptor = secure_open(path, label=label)
        if path == manifest and not raced:
            raced = True
            replacement.replace(manifest)
        return descriptor

    monkeypatch.setattr(provider, "_open_regular_fd", open_then_replace)
    payload, consumed = provider._read_json_with_sha256(
        manifest,
        _digest(original),
        label="raced authority",
    )

    assert payload == {"authority": "original"}
    assert consumed == original
    assert manifest.read_bytes() == b'{"authority":"replacement"}\n'


def test_source_snapshot_inventory_rejects_hardlinks_and_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    original = _write(source / "original.py", b"VALUE = 1\n")
    hardlink = source / "hardlink.py"
    os.link(original, hardlink)
    with pytest.raises(provider.ProviderInputError, match="non-private"):
        provider._stream_tree_record(source)

    hardlink.unlink()
    alias = source / "alias.py"
    alias.symlink_to(original)
    with pytest.raises(provider.ProviderInputError, match="symlink"):
        provider._stream_tree_record(source)


def test_snapshot_file_receipt_rejects_parent_traversal(tmp_path: Path) -> None:
    outside = _write(tmp_path / "outside", b"secret\n")
    record = {
        "path": "../outside",
        "bytes": outside.stat().st_size,
        "sha256": _digest(outside.read_bytes()),
    }
    with pytest.raises(provider.ProviderInputError, match="path is invalid"):
        provider._verify_snapshot_file(tmp_path / "snapshot", record, label="escape")


def test_execution_snapshot_is_content_addressed_immutable_and_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _initialize_layout(context.paths.work_root)
    captured: dict[str, bytes] = {
        "src/dialect/__init__.py": b"VERSION = 1\n",
        "external/CBaSE/input.txt": b"cbase\n",
        "external/MutSig2CV_src/source.m": b"mutsig\n",
        "scripts/run_cohort_pipeline.sh": b"#!/bin/sh\n",
        "scripts/run_mutsig_octave.sh": b"#!/bin/sh\n",
        "external/mutsig2cv_octave_dialect.patch": b"patch\n",
        "external/DIGDriver/run/Pancan.genes.results.txt": b"dig\n",
        provider.RUNTIME_AUTHORITY_PATH.as_posix(): b"{}\n",
    }
    modes = {
        relative: 0o500 if relative.startswith("scripts/") else 0o400
        for relative in captured
    }
    source_records = {
        name: provider._snapshot_file_record(
            relative,
            captured[relative],
            mode=modes[relative],
        )
        for name, relative in {
            "cohort_pipeline": "scripts/run_cohort_pipeline.sh",
            "mutsig_runner": "scripts/run_mutsig_octave.sh",
            "mutsig_patch": "external/mutsig2cv_octave_dialect.patch",
        }.items()
    }
    dig_record = provider._snapshot_file_record(
        "external/DIGDriver/run/Pancan.genes.results.txt",
        captured["external/DIGDriver/run/Pancan.genes.results.txt"],
        mode=0o400,
    )
    bindings: dict[str, dict[str, object]] = {}
    cohort_receipts: dict[str, object] = {}
    for cohort in provider.TCGA_COHORTS:
        bindings[cohort] = {}
        cohort_receipts[cohort] = {}
        for name, relative in (
            ("canonical_maf", f"data/mafs/{cohort}.maf"),
            ("sample_axis", f"data/axes/{cohort}.txt"),
        ):
            content = f"{cohort}/{name}\n".encode()
            captured[relative] = content
            modes[relative] = 0o400
            authority = {
                "path": f"signed/{cohort}/{name}",
                "bytes": len(content),
                "sha256": _digest(content),
            }
            bindings[cohort][name] = {"file": authority}
            cohort_receipts[cohort][name] = {
                **provider._snapshot_file_record(relative, content),
                "authority": authority,
            }
    components = {
        "dialect_python": {
            "root": "src/dialect",
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": provider._tree_digest(
                {"__init__.py": captured["src/dialect/__init__.py"]},
            ),
            "file_count": 1,
        },
        "cbase": {
            "root": "external/CBaSE",
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": provider._tree_digest(
                {"input.txt": captured["external/CBaSE/input.txt"]},
            ),
            "file_count": 1,
        },
        "mutsig_source": {
            "root": "external/MutSig2CV_src",
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": provider._tree_digest(
                {"source.m": captured["external/MutSig2CV_src/source.m"]},
            ),
            "file_count": 1,
            "upstream_commit": provider.MUTSIG_UPSTREAM_COMMIT,
        },
        **source_records,
        "dig_results": dig_record,
        "canonical_inputs": {
            "cohorts": cohort_receipts,
            "cohort_count": len(provider.TCGA_COHORTS),
        },
        "runtime_authority": provider._snapshot_file_record(
            provider.RUNTIME_AUTHORITY_PATH.as_posix(),
            captured[provider.RUNTIME_AUTHORITY_PATH.as_posix()],
        ),
    }
    context = replace(context, bindings=bindings)

    def materialize(_context, staging):
        for relative, content in sorted(captured.items()):
            provider._write_snapshot_bytes(
                staging,
                provider.PurePosixPath(relative),
                content,
                mode=modes[relative],
            )
        return components

    monkeypatch.setattr(
        provider,
        "_materialize_snapshot_components",
        materialize,
    )
    monkeypatch.setattr(
        provider,
        "_runtime_authority_payload",
        lambda _authority: {"tools": {}},
    )

    receipt = provider._build_execution_snapshot(context)
    snapshot_root = context.paths.work_root / "_orchestration" / receipt["root"]
    assert receipt["root"] == (
        f"{provider._EXECUTION_SNAPSHOT_PREFIX}{provider._tree_digest(captured, modes)}"
    )
    assert snapshot_root.stat().st_mode & 0o777 == 0o500
    assert (snapshot_root / "scripts/run_cohort_pipeline.sh").stat().st_mode & (
        0o777
    ) == 0o500
    assert (
        provider._stream_snapshot_inventory(snapshot_root).tree_sha256
        == (receipt["tree_sha256"])
    )

    target = snapshot_root / "data/mafs/CHOL.maf"
    target.chmod(0o600)
    target.write_bytes(b"changed\n")
    target.chmod(0o400)
    assert (
        provider._stream_snapshot_inventory(snapshot_root).tree_sha256
        != (receipt["tree_sha256"])
    )

    snapshot_root.chmod(0o700)
    for current, directory_names, file_names in os.walk(snapshot_root):
        current_path = Path(current)
        for name in file_names:
            (current_path / name).chmod(0o600)
        for name in directory_names:
            (current_path / name).chmod(0o700)


def test_framed_tree_digest_rejects_legacy_cross_file_boundary_collision() -> None:
    one_file = {"a": b"bc\0d"}
    two_files = {"a": b"b", "c": b"d"}

    def legacy_digest(tree: dict[str, bytes]) -> str:
        digest = hashlib.sha256()
        for relative, content in sorted(tree.items()):
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(content)
        return digest.hexdigest()

    assert legacy_digest(one_file) == legacy_digest(two_files)
    assert provider._tree_digest(one_file) != provider._tree_digest(two_files)


def test_snapshot_copy_peak_memory_is_bounded_for_large_sparse_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    sentinel_size = 64 * 1024 * 1024
    with source.open("xb") as handle:
        handle.seek(sentinel_size - 1)
        handle.write(b"x")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    tracemalloc.start()
    record = provider._copy_file_to_snapshot(
        source,
        snapshot,
        provider.PurePosixPath("large/source.bin"),
        label="large streaming sentinel",
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert record["bytes"] == sentinel_size
    assert (snapshot / "large/source.bin").stat().st_size == sentinel_size
    assert peak < 8 * 1024 * 1024


def test_scoped_public_validator_never_reads_unrelated_cohort_or_full_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "provider"
    root.mkdir()
    root.joinpath(*provider.WORK_AUTHORITY_PATH.parts).parent.mkdir(
        parents=True,
    )
    shared_contract: dict[str, object] = {}
    receipts = [{"cohort": cohort} for cohort in provider.TCGA_COHORTS]
    manifest = {
        "schema_version": provider.SCHEMA_VERSION,
        "contract": provider.PROVIDER_INPUT_CONTRACT,
        "completed_at_utc": "2026-08-28T00:00:00+00:00",
        "cohorts": list(provider.TCGA_COHORTS),
        "cohort_count": len(provider.TCGA_COHORTS),
        "roots": {"cohorts": "cohorts", "mutsig": "mutsig"},
        "authority": shared_contract,
        "sources": shared_contract,
        "providers": shared_contract,
        "execution": shared_contract,
        "scope": shared_contract,
        "cohort_provider_receipts": receipts,
        "inventory": {},
    }
    manifest_path = root / provider.ROOT_MANIFEST_NAME
    provider._write_json_atomic(manifest_path, manifest, mode=0o444)
    provider._write_json_atomic(
        root.joinpath(*provider.WORK_AUTHORITY_PATH.parts),
        {
            key: value
            for key, value in manifest.items()
            if key in {"authority", "sources", "providers", "execution", "scope"}
        },
        mode=0o444,
    )
    unrelated = root / "cohorts/BRCA/huge-unrelated-sentinel.bin"
    unrelated.parent.mkdir(parents=True)
    with unrelated.open("xb") as handle:
        handle.seek(512 * 1024 * 1024 - 1)
        handle.write(b"x")
    snapshot_root = root / "_orchestration/snapshot"
    snapshot_root.mkdir(parents=True)
    context = _context(tmp_path, bindings={"CHOL": {}})
    acceptance = {"accepted": True}
    validation_calls: list[dict[str, object]] = []
    secure_open = provider._open_regular_fd

    def guarded_open(path: Path, *, label: str) -> int:
        if path == unrelated:
            pytest.fail("scoped validation opened an unrelated cohort sentinel")
        return secure_open(path, label=label)

    def scoped_snapshot(_context, **kwargs):
        validation_calls.append(kwargs)
        return {"root": "snapshot", "tree_sha256": "a" * 64}, snapshot_root

    def forbidden(*_args, **_kwargs):
        pytest.fail("scoped validation invoked a full-family validator")

    monkeypatch.setattr(provider, "_open_regular_fd", guarded_open)
    monkeypatch.setattr(
        provider,
        "_validate_full_acceptance_receipt",
        lambda *_args: acceptance,
    )
    monkeypatch.setattr(
        provider,
        "_scoped_context_from_manifest",
        lambda *_args: (context, snapshot_root),
    )
    monkeypatch.setattr(provider, "_validate_execution_snapshot", scoped_snapshot)
    monkeypatch.setattr(provider, "_require_scoped_cohort_inventory", lambda *_: None)
    monkeypatch.setattr(provider, "_validate_provider_cohort_outputs", forbidden)
    monkeypatch.setattr(
        provider,
        "_published_scoped_cohort_binding",
        lambda *_args: {"cohort": "CHOL"},
    )
    monkeypatch.setattr(
        provider,
        "_require_scoped_binding_matches_manifest_inventory",
        lambda *_args: None,
    )
    monkeypatch.setattr(provider, "_verify_manifest_inventory", forbidden)
    monkeypatch.setattr(provider, "_stream_snapshot_inventory", forbidden)
    monkeypatch.setattr(provider, "_published_cohort_bindings", forbidden)

    result = provider.validate_materialized_provider_cohort_input(
        root,
        _digest(manifest_path.read_bytes()),
        "CHOL",
        acceptance,
        "f" * 64,
    )

    assert validation_calls == [
        {
            "cohort": "CHOL",
            "require_current_execution_environment": False,
            "validate_provider_generation_sources": False,
        },
    ]
    assert result["binding"] == {"cohort": "CHOL"}
    assert result["association_outputs_opened"] is False


def test_scoped_binding_does_not_invent_paths_for_receipt_only_canonical_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider"
    for relative in (
        "cohorts/CHOL/count_matrix.csv",
        "cohorts/CHOL/bmr_pmfs.csv",
        "cohorts/CHOL/bmr_pmfs.dig.csv",
        "cohorts/CHOL/sample_axis.txt",
        "mutsig/CHOL/persample_lambda.f32",
        "mutsig/CHOL/persample_meta.txt",
        "mutsig/CHOL/persample_genes.txt",
        "mutsig/CHOL/persample_patients.txt",
        "mutsig/CHOL/persample_receipt.tsv",
    ):
        _write(root / relative)
    snapshot_root = root / "_orchestration" / f"execution-snapshot-{'a' * 64}"
    canonical = {
        name: {
            "file": {
                "path": f"canonical/CHOL/{name}.json",
                "bytes": index + 1,
                "sha256": str(index) * 64,
            },
        }
        for index, name in enumerate(
            (
                "child_manifest",
                "canonical_maf",
                "sample_axis",
                "population_manifest",
            ),
            start=1,
        )
    }

    binding = provider._published_scoped_cohort_binding(
        root,
        snapshot_root,
        "CHOL",
        {"cohort": "CHOL"},
        canonical,
    )

    assert set(binding["canonical_inputs"]) == {"canonical_maf", "sample_axis"}
    assert binding["canonical_inputs"]["canonical_maf"]["path"] == (
        snapshot_root / "data/mafs/CHOL.maf"
    )
    assert set(binding["canonical_input_receipts"]) == {
        "child_manifest",
        "population_manifest",
    }
    assert all(
        isinstance(receipt["path"], str)
        for receipt in binding["canonical_input_receipts"].values()
    )
    provider_fields = (
        "count_matrix",
        "cbase_pmfs",
        "dig_pmfs",
        "sample_axis",
        "mutsig_lambda",
        "mutsig_metadata",
        "mutsig_genes",
        "mutsig_patients",
        "mutsig_receipt",
    )
    manifest = {
        "inventory": {
            "files": [binding[name]["file"] for name in provider_fields],
        },
    }
    provider._require_scoped_binding_matches_manifest_inventory(
        manifest,
        binding,
        "CHOL",
    )
    cross_wired = copy.deepcopy(binding)
    cross_wired["cbase_pmfs"]["file"] = binding["dig_pmfs"]["file"]
    with pytest.raises(provider.ProviderInputError, match="differs"):
        provider._require_scoped_binding_matches_manifest_inventory(
            manifest,
            cross_wired,
            "CHOL",
        )


def test_scoped_validator_end_to_end_reads_only_selected_accepted_files(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "provider"
    canonical_root = tmp_path / "canonical"
    approval = _write(tmp_path / "approval.json", b'{"signed":true}\n')
    canonical_manifest_file = _write(
        canonical_root / "input_manifest.json",
        b'{"fixture":"canonical"}\n',
    )
    hashes = provider.IndependentHashes(
        approval=_digest(approval.read_bytes()),
        canonical_input_manifest=_digest(canonical_manifest_file.read_bytes()),
        cbase_inputs_tree="c" * 64,
        dig_results="d" * 64,
    )
    canonical_manifest = {
        "contract": "synthetic-canonical-input-v1",
        "authority": {
            "approval": {
                "authorized_stage": "materialize-final-inputs",
                "manifest_sha256": hashes.approval,
                "decision_digests": {"D1": "1" * 64, "D2": "2" * 64},
            },
        },
    }
    paths = _paths(tmp_path)
    work_authority = _synthetic_published_authority(
        paths,
        hashes,
        canonical_manifest,
    )
    work_authority["intended_output_root"] = root.as_posix()

    selected_files: dict[str, Path] = {}
    for relative in sorted(
        {
            *(f"cohorts/CHOL/{name}" for name in provider.COHORT_ROOT_FILES),
            *(
                f"cohorts/CHOL/CBaSE_output/{name}"
                for name in provider.CBASE_OUTPUT_FILES
            ),
            *(f"mutsig/CHOL/{name}" for name in provider.MUTSIG_OUTPUT_FILES),
        },
    ):
        selected_files[relative] = _write(root / relative, f"{relative}\n".encode())
    unrelated_provider = root / "cohorts/BRCA/huge-unrelated.bin"
    unrelated_provider.parent.mkdir(parents=True)
    with unrelated_provider.open("xb") as handle:
        handle.seek(512 * 1024 * 1024 - 1)
        handle.write(b"x")

    dialect_bytes = b"VERSION = 1\n"
    dialect_digest = provider._tree_digest({"__init__.py": dialect_bytes})
    dialect_child = work_authority["providers"]["dialect_child"]
    dialect_child["dialect_tree_sha256"] = dialect_digest
    dialect_child_preimage = {
        key: value for key, value in dialect_child.items() if key != "runtime_sha256"
    }
    dialect_child["runtime_sha256"] = _digest(
        provider._canonical_json(dialect_child_preimage),
    )
    work_authority["sources"]["dialect_python_tree_sha256"] = dialect_digest
    work_authority["sources"]["python_runtime_sha256"] = dialect_child["runtime_sha256"]
    runtime_payload = provider._runtime_authority_payload(work_authority)
    runtime_bytes = provider._canonical_json(runtime_payload) + b"\n"

    maf_bytes = b"selected canonical maf\n"
    axis_bytes = b"s1\n"
    snapshot_canonical: dict[str, dict[str, dict[str, object]]] = {}
    cohort_receipts: list[dict[str, object]] = []
    for index, cohort in enumerate(provider.TCGA_COHORTS, start=1):
        maf_content = maf_bytes if cohort == "CHOL" else f"maf-{cohort}\n".encode()
        axis_content = axis_bytes if cohort == "CHOL" else f"axis-{cohort}\n".encode()
        canonical_records = {
            "child_manifest": {
                "path": f"canonical/{cohort}/input_manifest.json",
                "bytes": 1,
                "sha256": f"{index % 10}" * 64,
            },
            "canonical_maf": {
                "path": f"canonical/{cohort}/{cohort}.maf",
                "bytes": len(maf_content),
                "sha256": _digest(maf_content),
            },
            "sample_axis": {
                "path": f"canonical/{cohort}/sample_axis.txt",
                "bytes": len(axis_content),
                "sha256": _digest(axis_content),
            },
            "population_manifest": {
                "path": f"canonical/{cohort}/population_manifest.json",
                "bytes": 1,
                "sha256": f"{(index + 1) % 10}" * 64,
            },
        }
        snapshot_canonical[cohort] = {
            "canonical_maf": {
                "path": f"data/mafs/{cohort}.maf",
                "bytes": len(maf_content),
                "sha256": _digest(maf_content),
                "mode": 0o400,
                "authority": canonical_records["canonical_maf"],
            },
            "sample_axis": {
                "path": f"data/axes/{cohort}.txt",
                "bytes": len(axis_content),
                "sha256": _digest(axis_content),
                "mode": 0o400,
                "authority": canonical_records["sample_axis"],
            },
        }
        cohort_receipts.append(
            {
                "cohort": cohort,
                "canonical_inputs": canonical_records,
            },
        )

    snapshot_digest = "e" * 64
    snapshot_name = f"{provider._EXECUTION_SNAPSHOT_PREFIX}{snapshot_digest}"
    snapshot_root = root / "_orchestration" / snapshot_name
    _write(snapshot_root / "src/dialect/__init__.py", dialect_bytes)
    _write(snapshot_root / "data/mafs/CHOL.maf", maf_bytes)
    _write(snapshot_root / "data/axes/CHOL.txt", axis_bytes)
    _write(
        snapshot_root.joinpath(*provider.RUNTIME_AUTHORITY_PATH.parts),
        runtime_bytes,
    )
    unrelated_snapshot = snapshot_root / "data/mafs/BRCA.maf"
    with unrelated_snapshot.open("xb") as handle:
        handle.seek(512 * 1024 * 1024 - 1)
        handle.write(b"x")
    for candidate in snapshot_root.rglob("*"):
        candidate.chmod(0o500 if candidate.is_dir() else 0o400)
    snapshot_root.chmod(0o500)

    sources = work_authority["sources"]
    providers = work_authority["providers"]

    def snapshot_component(
        relative: str,
        authority: dict[str, object],
    ) -> dict[str, object]:
        return {
            "path": relative,
            "bytes": authority["bytes"],
            "sha256": authority["sha256"],
            "mode": 0o400,
        }

    runtime_record = provider._snapshot_file_record(
        provider.RUNTIME_AUTHORITY_PATH.as_posix(),
        runtime_bytes,
    )
    components = {
        "dialect_python": {
            "root": "src/dialect",
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": dialect_digest,
            "file_count": 1,
        },
        "cbase": {
            "root": "external/CBaSE",
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": hashes.cbase_inputs_tree,
            "file_count": 1,
        },
        "mutsig_source": {
            "root": "external/MutSig2CV_src",
            "tree_hash_contract": provider.TREE_HASH_CONTRACT,
            "tree_sha256": providers["mutsig"]["source_tree_sha256"],
            "file_count": providers["mutsig"]["source_file_count"],
            "upstream_commit": provider.MUTSIG_UPSTREAM_COMMIT,
        },
        "cohort_pipeline": snapshot_component(
            "scripts/run_cohort_pipeline.sh",
            sources["cohort_pipeline"],
        ),
        "mutsig_runner": snapshot_component(
            "scripts/run_mutsig_octave.sh",
            sources["mutsig_runner"],
        ),
        "mutsig_patch": snapshot_component(
            "external/mutsig2cv_octave_dialect.patch",
            sources["mutsig_patch"],
        ),
        "dig_results": snapshot_component(
            "external/DIGDriver/run/Pancan.genes.results.txt",
            providers["dig"]["results"],
        ),
        "canonical_inputs": {
            "cohorts": snapshot_canonical,
            "cohort_count": len(provider.TCGA_COHORTS),
        },
        "runtime_authority": runtime_record,
    }
    snapshot_receipt = {
        "schema_version": provider.SCHEMA_VERSION,
        "contract": provider.EXECUTION_SNAPSHOT_CONTRACT,
        "root": snapshot_name,
        "tree_hash_contract": provider.TREE_HASH_CONTRACT,
        "tree_sha256": snapshot_digest,
        "file_count": 5,
        "components": components,
        "runtime_boundary": {
            "authority_file": runtime_record,
            "tools": runtime_payload["tools"],
        },
        "association_outputs_opened": False,
    }
    provider._write_json_atomic(
        root.joinpath(*provider.EXECUTION_SNAPSHOT_RECEIPT.parts),
        snapshot_receipt,
        mode=0o444,
    )
    provider._write_json_atomic(
        root.joinpath(*provider.WORK_AUTHORITY_PATH.parts),
        work_authority,
        mode=0o444,
    )
    selected_binding_paths = (
        "cohorts/CHOL/count_matrix.csv",
        "cohorts/CHOL/bmr_pmfs.csv",
        "cohorts/CHOL/bmr_pmfs.dig.csv",
        "cohorts/CHOL/sample_axis.txt",
        "mutsig/CHOL/persample_lambda.f32",
        "mutsig/CHOL/persample_meta.txt",
        "mutsig/CHOL/persample_genes.txt",
        "mutsig/CHOL/persample_patients.txt",
        "mutsig/CHOL/persample_receipt.tsv",
    )
    manifest = {
        "schema_version": provider.SCHEMA_VERSION,
        "contract": provider.PROVIDER_INPUT_CONTRACT,
        "completed_at_utc": "2026-08-28T00:00:00+00:00",
        "cohorts": list(provider.TCGA_COHORTS),
        "cohort_count": len(provider.TCGA_COHORTS),
        "roots": {"cohorts": "cohorts", "mutsig": "mutsig"},
        "authority": work_authority["authority"],
        "sources": sources,
        "providers": providers,
        "execution": work_authority["execution"],
        "scope": work_authority["scope"],
        "cohort_provider_receipts": cohort_receipts,
        "inventory": {
            "directories": [],
            "files": [
                provider._file_record(root / relative, display_path=relative)
                for relative in selected_binding_paths
            ],
            "execution_snapshot": {
                "root": f"_orchestration/{snapshot_name}",
                "tree_hash_contract": provider.TREE_HASH_CONTRACT,
                "tree_sha256": snapshot_digest,
                "file_count": 5,
                "directory_count": 6,
                "individual_file_receipts_omitted": True,
            },
            "root_manifest_excluded_from_self_inventory": (provider.ROOT_MANIFEST_NAME),
        },
    }
    manifest_path = root / provider.ROOT_MANIFEST_NAME
    provider._write_json_atomic(manifest_path, manifest, mode=0o444)
    manifest_sha256 = _digest(manifest_path.read_bytes())
    acceptance = provider._full_acceptance_receipt(manifest, manifest_sha256)
    acceptance_sha256 = provider.full_acceptance_receipt_sha256(acceptance)
    secure_open = provider._open_regular_fd
    opened_paths: set[Path] = set()

    def guarded_open(path: Path, *, label: str) -> int:
        if path in {unrelated_provider, unrelated_snapshot}:
            pytest.fail("scoped validator opened an unrelated large sentinel")
        opened_paths.add(path)
        return secure_open(path, label=label)

    monkeypatch.setattr(provider, "_open_regular_fd", guarded_open)
    with pytest.raises(provider.ProviderInputError, match="independent SHA-256"):
        provider.validate_materialized_provider_cohort_input(
            root,
            manifest_sha256,
            "CHOL",
            acceptance,
            "0" * 64,
        )
    validated = provider.validate_materialized_provider_cohort_input(
        root,
        manifest_sha256,
        "CHOL",
        acceptance,
        acceptance_sha256,
    )

    expected_opened = {
        *(root / relative for relative in selected_binding_paths),
        snapshot_root / "data/mafs/CHOL.maf",
        snapshot_root / "data/axes/CHOL.txt",
    }
    assert expected_opened <= opened_paths
    assert (
        validated["provider_receipt"]
        == cohort_receipts[provider.TCGA_COHORTS.index("CHOL")]
    )

    count_matrix = root / "cohorts/CHOL/count_matrix.csv"
    original_count = count_matrix.read_bytes()
    count_matrix.write_bytes(b"mutated selected count matrix\n")
    with pytest.raises(provider.ProviderInputError, match="full acceptance"):
        provider.validate_materialized_provider_cohort_input(
            root,
            manifest_sha256,
            "CHOL",
            acceptance,
            acceptance_sha256,
        )
    count_matrix.write_bytes(original_count)

    cbase = root / "cohorts/CHOL/bmr_pmfs.csv"
    dig = root / "cohorts/CHOL/bmr_pmfs.dig.csv"
    cbase_bytes, dig_bytes = cbase.read_bytes(), dig.read_bytes()
    cbase.write_bytes(dig_bytes)
    dig.write_bytes(cbase_bytes)
    with pytest.raises(provider.ProviderInputError, match="full acceptance"):
        provider.validate_materialized_provider_cohort_input(
            root,
            manifest_sha256,
            "CHOL",
            acceptance,
            acceptance_sha256,
        )


def test_safe_job_cap_is_strictly_below_half_and_at_most_three() -> None:
    assert provider.safe_job_cap(2) == 0
    assert provider.safe_job_cap(3) == 1
    assert provider.safe_job_cap(4) == 1
    assert provider.safe_job_cap(5) == 2
    assert provider.safe_job_cap(64) == 3
    with pytest.raises(provider.ProviderInputError, match="strictly below half"):
        provider._validate_jobs(1, logical_cores=2)
    with pytest.raises(provider.ProviderInputError, match="between 1 and 2"):
        provider._validate_jobs(3, logical_cores=5)


def test_pipeline_tree_hash_helpers_match_provider_validator(tmp_path: Path) -> None:
    pipeline = (
        Path(provider.__file__).resolve().parents[1]
        / "scripts"
        / "run_cohort_pipeline.sh"
    ).read_text(encoding="utf-8")
    matched = re.search(
        r"tree_sha256\(\) \{\n  run_python -c '\n(.*?)\n' "
        r'"\$1" "\$2"\n\}',
        pipeline,
        flags=re.DOTALL,
    )
    assert matched is not None
    helper = matched.group(1)

    tree = tmp_path / "tree"
    _write(tree / "a.py", b"a = 1\n")
    _write(tree / "nested/b.py", b"b = 2\n")
    _write(tree / "nested/data.txt", b"payload\n")
    _write(tree / "__pycache__/ignored.pyc", b"cache\n")

    observed_python = subprocess.run(  # noqa: S603
        [sys.executable, "-c", helper, str(tree), "python-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_all = subprocess.run(  # noqa: S603
        [sys.executable, "-c", helper, str(tree), "all-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert observed_python == provider._tree_sha256(tree, python_only=True)
    assert observed_all == provider._tree_sha256(tree)


def test_wave_plan_has_chol_canary_and_serial_heavy_tail() -> None:
    waves = provider.plan_provider_waves(provider.TCGA_COHORTS, jobs=3)
    assert waves[0] == ("CHOL",)
    assert all(len(wave) <= 3 for wave in waves)
    assert waves[-5:] == [
        (cohort,)
        for cohort in provider.TCGA_COHORTS
        if cohort in provider.MEMORY_HEAVY_COHORTS
    ]
    flattened = [cohort for wave in waves for cohort in wave]
    assert len(flattened) == 32
    assert set(flattened) == set(provider.TCGA_COHORTS)
    for wave in waves:
        if len(wave) > 1:
            assert not set(wave) & provider.MEMORY_HEAVY_COHORTS


def test_parallel_wave_validates_only_after_all_transient_staging_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    _initialize_layout(paths.work_root)
    bindings = {
        cohort: {
            "canonical_maf": {"path": tmp_path / f"canonical/{cohort}.maf"},
            "sample_axis": {"path": tmp_path / f"canonical/{cohort}.txt"},
        }
        for cohort in ("ACC", "BLCA")
    }
    context = replace(_context(tmp_path, bindings=bindings), paths=paths)
    snapshot_root = tmp_path / "execution-snapshot"
    (snapshot_root / "scripts").mkdir(parents=True)
    snapshot_receipt = {
        "contract": provider.EXECUTION_SNAPSHOT_CONTRACT,
        "root": f"{provider._EXECUTION_SNAPSHOT_PREFIX}{'a' * 64}",
        "tree_sha256": "a" * 64,
        "components": {
            "runtime_authority": {"sha256": "9" * 64},
            "canonical_inputs": {
                "cohorts": {
                    cohort: {
                        "canonical_maf": {"sha256": "b" * 64},
                        "sample_axis": {"sha256": "c" * 64},
                    }
                    for cohort in ("ACC", "BLCA")
                },
            },
        },
    }
    barrier = threading.Barrier(2)
    transient = paths.mutsig_root / ".ACC.mutsig.synthetic"
    validated: list[str] = []

    def run(command, **_kwargs):
        cohort = command[-1]
        if cohort == "ACC":
            transient.mkdir()
            barrier.wait(timeout=2)
            time.sleep(0.05)
            transient.rmdir()
        else:
            barrier.wait(timeout=2)
        return SimpleNamespace(returncode=0)

    def validate(_context, cohort):
        assert not transient.exists()
        validated.append(cohort)
        return {"cohort": cohort}

    monkeypatch.setattr(provider.subprocess, "run", run)
    monkeypatch.setattr(provider, "_cohort_is_complete", lambda *_args: False)
    monkeypatch.setattr(
        provider,
        "_require_canonical_cohort_current",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        provider,
        "_require_live_resource_gate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(provider, "validate_provider_cohort", validate)
    monkeypatch.setattr(
        provider,
        "_validate_execution_snapshot",
        lambda *_args, **_kwargs: (snapshot_receipt, snapshot_root),
    )
    provider._run_wave(context, ("ACC", "BLCA"), wave_number=2)
    assert validated == ["ACC", "BLCA"]


def test_pipeline_records_and_rejects_post_child_snapshot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _initialize_layout(context.paths.work_root)
    snapshot_root = tmp_path / "execution-snapshot"
    (snapshot_root / "scripts").mkdir(parents=True)
    receipt = {
        "contract": provider.EXECUTION_SNAPSHOT_CONTRACT,
        "root": f"{provider._EXECUTION_SNAPSHOT_PREFIX}{'a' * 64}",
        "tree_sha256": "a" * 64,
        "components": {
            "runtime_authority": {"sha256": "9" * 64},
            "canonical_inputs": {
                "cohorts": {
                    "CHOL": {
                        "canonical_maf": {"sha256": "b" * 64},
                        "sample_axis": {"sha256": "c" * 64},
                    },
                },
            },
        },
    }
    validations = 0

    def validate(*_args, **_kwargs):
        nonlocal validations
        validations += 1
        if validations == 2:
            msg = "snapshot mutated"
            raise provider.ProviderInputError(msg)
        return receipt, snapshot_root

    monkeypatch.setattr(provider, "_validate_execution_snapshot", validate)
    monkeypatch.setattr(
        provider,
        "_require_canonical_cohort_current",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        provider.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(provider.ProviderInputError, match="changed while"):
        provider._invoke_pipeline(context, "CHOL")

    attempt_receipts = list(
        (context.paths.work_root / "attempts/CHOL").glob("*.json"),
    )
    assert validations == 2
    assert len(attempt_receipts) == 1
    attempt = provider._read_json(attempt_receipts[0])
    assert attempt["execution_snapshot"]["rehashed_after_child"] is False
    assert (
        "snapshot mutated"
        in attempt["execution_snapshot"]["post_child_validation_error"]
    )


@pytest.mark.parametrize(
    ("artifact_name", "invalid_bytes"),
    [
        ("count_matrix.csv", b"\xff\xfe"),
        ("bmr_pmfs.csv", b"feature," + (b"9" * 5000) + b"\nA_M,1\n"),
    ],
    ids=("invalid-utf8", "oversized-count-key"),
)
def test_resume_regenerates_malformed_provider_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    invalid_bytes: bytes,
) -> None:
    context = _context(tmp_path)
    _initialize_layout(context.paths.work_root)
    artifact = _write(
        context.paths.cohort_root / "CHOL" / artifact_name,
        invalid_bytes,
    )
    invoked: list[str] = []

    def validate(_context, _cohort):
        if artifact_name == "count_matrix.csv":
            provider._validate_count_matrix(artifact, ["s1"])
        else:
            provider._validate_pmf_table(artifact, label="synthetic")
        return {"cohort": "CHOL"}

    def invoke(_context, cohort):
        invoked.append(cohort)
        if artifact_name == "count_matrix.csv":
            artifact.write_bytes(b"sample,A_M\ns1,0\n")
        else:
            artifact.write_bytes(b"feature,0\nA_M,1\n")
        return cohort, 0

    monkeypatch.setattr(provider, "validate_provider_cohort", validate)
    monkeypatch.setattr(provider, "_invoke_pipeline", invoke)
    monkeypatch.setattr(
        provider,
        "_require_canonical_cohort_current",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        provider,
        "_require_live_resource_gate",
        lambda *_args, **_kwargs: None,
    )

    provider._run_wave(context, ("CHOL",), wave_number=1)

    assert invoked == ["CHOL"]


def test_completeness_probe_never_downgrades_canonical_authority_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)

    def reject_canonical(*_args):
        msg = "canonical authority changed"
        raise provider.ProviderInputError(msg)

    monkeypatch.setattr(
        provider,
        "_require_canonical_cohort_current",
        reject_canonical,
    )
    with pytest.raises(provider.ProviderInputError, match="canonical authority"):
        provider._cohort_is_complete(context, "CHOL")


def test_live_gate_uses_aggregate_cpu_memory_disk_and_job_cap() -> None:
    healthy = provider.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=16,
        load_average_1m=2.0,
        total_memory_bytes=64 * provider._GIB,
        available_memory_bytes=48 * provider._GIB,
        free_disk_bytes=100 * provider._GIB,
        cpu_source="aggregate-load",
        memory_source="aggregate-memory",
    )
    assert provider.evaluate_host_resource_gate(healthy, jobs=3)["passed"] is True

    pressured = replace(
        healthy,
        load_average_1m=8.0,
        available_memory_bytes=1 * provider._GIB,
        free_disk_bytes=1 * provider._GIB,
    )
    evaluation = provider.evaluate_host_resource_gate(pressured, jobs=4)
    assert evaluation["passed"] is False
    assert len(evaluation["reasons"]) == 4


def test_live_gate_reserves_planned_jobs_below_half_host() -> None:
    snapshot = provider.HostResourceSnapshot(
        measured_at_utc="2026-08-28T00:00:00+00:00",
        logical_cores=16,
        load_average_1m=5.0,
        total_memory_bytes=64 * provider._GIB,
        available_memory_bytes=48 * provider._GIB,
        free_disk_bytes=100 * provider._GIB,
        cpu_source="aggregate-load",
        memory_source="aggregate-memory",
    )

    evaluation = provider.evaluate_host_resource_gate(snapshot, jobs=3)

    assert evaluation["passed"] is False
    assert evaluation["projected_load_with_planned_jobs"] == 8.0
    assert evaluation["strict_half_core_limit"] == 8.0
    assert evaluation["reasons"] == [
        "one-minute aggregate CPU load plus planned jobs is not below half the host",
    ]


def test_pipeline_environment_is_sealed_and_uses_public_canonical_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maf = _write(tmp_path / "canonical/mafs/CHOL.maf")
    axis = _write(tmp_path / "canonical/population/CHOL/sample_axis.txt", b"s1\n")
    context = _context(
        tmp_path,
        bindings={
            "CHOL": {
                "canonical_maf": {"path": maf},
                "sample_axis": {"path": axis},
            },
        },
    )
    hostile_root = tmp_path / "hostile"
    hostile = {
        "BASH_ENV": (hostile_root / "unset-prepare-only.sh").as_posix(),
        "DYLD_INSERT_LIBRARIES": (hostile_root / "injected.dylib").as_posix(),
        "JAVA_HOME": (hostile_root / "java").as_posix(),
        "LD_PRELOAD": (hostile_root / "injected.so").as_posix(),
        "OCTAVE_BIN": (hostile_root / "fake-octave").as_posix(),
        "PATH": (hostile_root / "path").as_posix(),
        "PYTHONPATH": (hostile_root / "python").as_posix(),
        "PYTHONNOUSERSITE": "0",
        "PYTHONSAFEPATH": "0",
        "SKIP_MUTSIG": "1",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    snapshot_root = tmp_path / "private-snapshot"
    snapshot_receipt = {
        "components": {
            "runtime_authority": {"sha256": "9" * 64},
        },
    }
    environment = provider._pipeline_environment(
        context,
        "CHOL",
        snapshot_root=snapshot_root,
        snapshot_receipt=snapshot_receipt,
    )
    assert not (
        set(hostile) - {"PATH", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONSAFEPATH"}
    ) & set(environment)
    assert environment["PATH"] == provider.SAFE_CHILD_PATH
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PREPARE_ONLY"] == "1"
    assert environment["TOP_K"] == "500"
    assert environment["MAF_DIR"] == (snapshot_root / "data/mafs").as_posix()
    assert (
        environment["MUTSIG_SAMPLE_AXIS_FILE"]
        == (snapshot_root / "data/axes/CHOL.txt").as_posix()
    )
    assert environment["PYTHONPATH"] == (snapshot_root / "src").as_posix()
    assert environment["ROOT"] == context.paths.cohort_root.as_posix()
    assert environment["MUTSIG_ROOT"] == context.paths.mutsig_root.as_posix()
    assert environment["DIALECT_PROVIDER_CBASE_INPUTS_TREE_SHA256"] == "c" * 64
    assert environment["DIALECT_PROVIDER_MUTSIG_SOURCE_TREE_SHA256"] == "8" * 64
    assert environment["DIALECT_PROVIDER_MUTSIG_SOURCE_FILE_COUNT"] == "1"
    assert {name: environment[name] for name in provider.THREAD_ENVIRONMENT} == (
        provider.THREAD_ENVIRONMENT
    )
    assert provider._pipeline_command(
        context,
        "CHOL",
        snapshot_root=snapshot_root,
    ) == [
        context.authority["sources"]["nice_executable"]["path"],
        "-n",
        "10",
        context.authority["sources"]["bash_executable"]["path"],
        (snapshot_root / "scripts/run_cohort_pipeline.sh").as_posix(),
        "CHOL",
    ]


def test_child_runtime_probe_binds_frozen_entrypoint_and_repo_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    child_python = _write(tmp_path / "runtime/python", b"python-binary\n")
    dialect_import = _write(paths.repo_root / "src/dialect/__init__.py")
    entrypoint = _write(
        tmp_path / "runtime/dialect",
        f"#!{child_python}\n".encode(),
    )
    imported = {
        name: _write(tmp_path / f"runtime/{name}/__init__.py", name.encode())
        for name in ("numpy", "pandas", "scipy")
    }
    payload = {
        "dialect_file": dialect_import.as_posix(),
        "numpy": "2.0",
        "numpy_file": imported["numpy"].as_posix(),
        "pandas": "3.0",
        "pandas_file": imported["pandas"].as_posix(),
        "python": "3.12.4",
        "scipy": "4.0",
        "scipy_file": imported["scipy"].as_posix(),
        "sys_executable": child_python.as_posix(),
    }
    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=(provider._canonical_json(payload).decode() + "\n"),
            stderr="",
        )

    monkeypatch.setattr(provider, "CHILD_PYTHON_EXECUTABLE", child_python)
    monkeypatch.setattr(provider, "DIALECT_EXECUTABLE", entrypoint)
    monkeypatch.setattr(provider.subprocess, "run", run)
    monkeypatch.setattr(
        provider,
        "_distribution_runtime_record",
        lambda name: {"name": name, "fixture": True},
    )
    monkeypatch.setattr(provider, "_tree_sha256", lambda *_args, **_kwargs: "f" * 64)
    record = provider._child_python_runtime_record(paths)

    assert observed["command"][:3] == [child_python.as_posix(), "-P", "-s"]
    assert observed["environment"] == provider._base_child_environment()
    assert record["dialect_import"]["sha256"] == _digest(
        dialect_import.read_bytes(),
    )
    runtime_preimage = {
        key: value for key, value in record.items() if key != "runtime_sha256"
    }
    assert record["runtime_sha256"] == _digest(
        provider._canonical_json(runtime_preimage),
    )

    payload["dialect_file"] = _write(tmp_path / "outside/dialect.py").as_posix()
    with pytest.raises(provider.ProviderInputError, match="this repository"):
        provider._child_python_runtime_record(paths)


def test_mutsig_runtime_probe_uses_the_runner_java_home_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    octave = _write(tmp_path / "octave", b"octave-binary\n")
    java_home = tmp_path / "java"
    java = _write(java_home / "bin/java", b"java-binary\n")
    observed: list[tuple[object, object]] = []

    def run(command, **kwargs):
        observed.append((command, kwargs["env"]))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GNU Octave, version 9.1.0\n"
                if command[0] == octave.as_posix()
                else 'openjdk version "11.0.24"\n'
            ),
        )

    monkeypatch.setattr(provider.shutil, "which", lambda *_args, **_kwargs: octave)
    monkeypatch.setattr(provider, "MUTSIG_JAVA_HOME", java_home.as_posix())
    monkeypatch.setattr(provider.subprocess, "run", run)
    record = provider._mutsig_runtime_record()

    assert observed[0][0] == [
        octave.as_posix(),
        "--no-init-all",
        "--no-history",
        "--no-gui",
        "--version",
    ]
    assert observed[0][1]["JAVA_HOME"] == java_home.as_posix()
    assert observed[0][1]["PATH"] == provider.MUTSIG_CHILD_PATH
    assert observed[1][0] == [java.as_posix(), "-version"]
    runtime_preimage = {
        key: value for key, value in record.items() if key != "runtime_sha256"
    }
    assert record["runtime_sha256"] == _digest(
        provider._canonical_json(runtime_preimage),
    )


def test_mutsig_runner_preflight_requires_startup_and_history_suppression(
    tmp_path: Path,
) -> None:
    runner = (
        Path(provider.__file__).resolve().parents[1] / "scripts/run_mutsig_octave.sh"
    )
    provider._require_hardened_mutsig_runner(runner)

    weakened = _write(
        tmp_path / "run_mutsig_octave.sh",
        runner.read_bytes().replace(b"--no-init-all", b"", 1),
    )
    with pytest.raises(provider.ProviderInputError, match="startup files"):
        provider._require_hardened_mutsig_runner(weakened)

    symlink = tmp_path / "runner-symlink"
    symlink.symlink_to(runner)
    with pytest.raises(provider.ProviderInputError, match="non-symlink"):
        provider._require_hardened_mutsig_runner(symlink)


def test_public_bundle_api_is_used_for_all_32_no_derived_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cohorts": list(provider.TCGA_COHORTS),
        "authority": {
            "approval": {
                "manifest_sha256": "a" * 64,
                "decision_digests": {"D1": "1" * 64, "D2": "2" * 64},
            },
        },
    }
    calls: list[tuple[object, ...]] = []

    def validate(*args, **kwargs):
        calls.append((*args, kwargs))
        return manifest

    def bind(_root, passed_manifest, cohort):
        assert passed_manifest is manifest
        result = {"cohort": cohort}
        for name in (
            "child_manifest",
            "canonical_maf",
            "sample_axis",
            "population_manifest",
        ):
            path = _write(tmp_path / f"bindings/{cohort}/{name}", name.encode())
            result[name] = {
                "path": path,
                "file": _record(path, f"signed/{cohort}/{name}"),
            }
        return result

    monkeypatch.setattr(provider, "validate_materialized_input_bundle", validate)
    monkeypatch.setattr(
        provider,
        "validate_revision_approval",
        lambda *_args: SimpleNamespace(
            schema=provider.STAGE_SCOPED_APPROVAL_SCHEMA,
            allowed_stages=(provider.MATERIALIZE_FINAL_INPUTS_STAGE,),
            stage_bindings={provider.MATERIALIZE_FINAL_INPUTS_STAGE: {}},
            decisions={"D1": object(), "D2": object()},
            decision_digests={"D1": "1" * 64, "D2": "2" * 64},
        ),
    )
    monkeypatch.setattr(provider, "materialized_cohort_binding", bind)
    monkeypatch.setattr(provider, "_authority_record", lambda *_args: {"ok": True})
    hashes = provider.IndependentHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    context = provider._build_context(paths, hashes)

    assert len(context.bindings) == 32
    assert calls == [
        (
            paths.canonical_input_root,
            "b" * 64,
            paths.approval_manifest,
            "a" * 64,
            {"require_current_execution_environment": True},
        ),
    ]


def test_provider_context_rejects_coauthorized_extra_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    hashes = provider.IndependentHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    monkeypatch.setattr(
        provider,
        "validate_revision_approval",
        lambda *_args: SimpleNamespace(
            schema=provider.STAGE_SCOPED_APPROVAL_SCHEMA,
            allowed_stages=(
                provider.MATERIALIZE_FINAL_INPUTS_STAGE,
                "fit-sealed-tcga-k500",
            ),
            stage_bindings={
                provider.MATERIALIZE_FINAL_INPUTS_STAGE: {},
                "fit-sealed-tcga-k500": {},
            },
            decisions={"D1": object(), "D2": object()},
            decision_digests={"D1": "1" * 64, "D2": "2" * 64},
        ),
    )
    monkeypatch.setattr(
        provider,
        "validate_materialized_input_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical bundle must not be opened after overbroad approval",
        ),
    )

    with pytest.raises(provider.ProviderInputError, match="stage-scoped v5"):
        provider._canonical_bundle_state(
            paths,
            hashes,
            require_current_execution_environment=True,
        )


def test_provider_context_rejects_historical_v4_overattestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    hashes = provider.IndependentHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    monkeypatch.setattr(
        provider,
        "validate_revision_approval",
        lambda *_args: SimpleNamespace(
            schema="dialect-revision-coauthor-approval-v4",
            allowed_stages=(provider.MATERIALIZE_FINAL_INPUTS_STAGE,),
            stage_bindings={provider.MATERIALIZE_FINAL_INPUTS_STAGE: {}},
            decisions={f"D{index}": object() for index in range(1, 11)},
            decision_digests={f"D{index}": "1" * 64 for index in range(1, 11)},
        ),
    )
    monkeypatch.setattr(
        provider,
        "validate_materialized_input_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical bundle must not be opened after v4 overattestation",
        ),
    )

    with pytest.raises(provider.ProviderInputError, match="stage-scoped v5"):
        provider._canonical_bundle_state(
            paths,
            hashes,
            require_current_execution_environment=True,
        )


def test_work_inventory_rejects_association_outputs_extras_and_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    _initialize_layout(root)
    (root / "cohorts/CHOL").mkdir()
    _write(root / "cohorts/CHOL/pairwise_interaction_results.csv")
    with pytest.raises(provider.ProviderInputError, match="Association identify"):
        provider._require_allowed_work_inventory(root)
    (root / "cohorts/CHOL/pairwise_interaction_results.csv").unlink()
    _write(root / "surprise.txt")
    with pytest.raises(provider.ProviderInputError, match="extra file"):
        provider._require_allowed_work_inventory(root)
    (root / "surprise.txt").unlink()
    (root / "alias").symlink_to(root / "cohorts")
    with pytest.raises(provider.ProviderInputError, match="symlink"):
        provider._require_allowed_work_inventory(root)


def test_work_inventory_rejects_hard_linked_output_before_pipeline_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _initialize_layout(context.paths.work_root)
    sentinel = _write(tmp_path / "protected-canonical-sentinel.maf", b"protected\n")
    linked_output = context.paths.cohort_root / "CHOL/count_matrix.csv"
    linked_output.parent.mkdir()
    provider.os.link(sentinel, linked_output)

    def forbidden_launch(*_args, **_kwargs):
        pytest.fail("pipeline launched after a hard-linked provider output")

    monkeypatch.setattr(provider, "_invoke_pipeline", forbidden_launch)
    with pytest.raises(provider.ProviderInputError, match="hard-linked"):
        provider._run_wave(context, ("CHOL",), wave_number=1)

    assert sentinel.read_bytes() == b"protected\n"


def test_hard_linked_root_manifest_is_never_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    _initialize_layout(root)
    sentinel = _write(
        tmp_path / "external-association-result.json",
        b'{"association":"secret"}\n',
    )
    manifest_path = root / provider.ROOT_MANIFEST_NAME
    provider.os.link(sentinel, manifest_path)
    context = _context(tmp_path)
    original_read_json = provider._read_json
    original_sha256 = provider._sha256

    def guarded_read_json(path, **kwargs):
        if path == manifest_path:
            pytest.fail("hard-linked external manifest was opened as JSON")
        return original_read_json(path, **kwargs)

    def guarded_sha256(path):
        if path == manifest_path:
            pytest.fail("hard-linked external manifest was opened for hashing")
        return original_sha256(path)

    monkeypatch.setattr(provider, "_read_json", guarded_read_json)
    with pytest.raises(provider.ProviderInputError, match="single-link"):
        provider._validate_published_root(context, root)

    monkeypatch.setattr(provider, "_sha256", guarded_sha256)
    with pytest.raises(provider.ProviderInputError, match="single-link"):
        provider.validate_materialized_provider_input_bundle(
            root,
            "a" * 64,
            tmp_path / "canonical",
            "b" * 64,
            tmp_path / "approval.json",
            "c" * 64,
        )

    assert sentinel.read_bytes() == b'{"association":"secret"}\n'


def test_published_validation_rejects_authority_symlink_before_opening_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    _initialize_layout(root)
    context = _context(tmp_path)
    sentinel = _write(tmp_path / "external-association-sentinel.json", b"{}\n")
    authority_path = root / provider.WORK_AUTHORITY_PATH
    authority_path.symlink_to(sentinel)
    manifest = {
        "schema_version": provider.SCHEMA_VERSION,
        "contract": provider.PROVIDER_INPUT_CONTRACT,
        "completed_at_utc": "2026-08-28T00:00:00+00:00",
        "cohorts": list(provider.TCGA_COHORTS),
        "cohort_count": len(provider.TCGA_COHORTS),
        "roots": {"cohorts": "cohorts", "mutsig": "mutsig"},
        "authority": context.authority["authority"],
        "sources": context.authority["sources"],
        "providers": context.authority["providers"],
        "execution": context.authority["execution"],
        "scope": context.authority["scope"],
        "cohort_provider_receipts": [],
        "inventory": {
            "directories": [],
            "files": [],
            "root_manifest_excluded_from_self_inventory": (provider.ROOT_MANIFEST_NAME),
        },
    }
    provider._write_json_atomic(
        root / provider.ROOT_MANIFEST_NAME,
        manifest,
        mode=0o444,
    )
    original_read_json = provider._read_json

    def guarded_read(path, **kwargs):
        if path == authority_path:
            pytest.fail("authority symlink target would have been opened")
        return original_read_json(path, **kwargs)

    monkeypatch.setattr(provider, "_read_json", guarded_read)
    monkeypatch.setattr(provider, "_verify_manifest_inventory", lambda *_args: None)
    with pytest.raises(provider.ProviderInputError, match="symlink"):
        provider._validate_published_root(context, root)


def test_resume_rejects_authority_symlink_before_opening_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _initialize_layout(context.paths.work_root)
    sentinel = _write(tmp_path / "external-association-sentinel.json", b"{}\n")
    authority_path = context.paths.work_root / provider.WORK_AUTHORITY_PATH
    authority_path.symlink_to(sentinel)
    original_read_json = provider._read_json

    def guarded_read(path, **kwargs):
        if path == authority_path:
            pytest.fail("authority symlink target would have been opened")
        return original_read_json(path, **kwargs)

    monkeypatch.setattr(provider, "_read_json", guarded_read)
    with pytest.raises(provider.ProviderInputError, match="symlink"):
        provider._initialize_work_root(context)


def test_resume_rejects_symlinked_authority_parent_before_opening_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _initialize_layout(context.paths.work_root)
    external = tmp_path / "external-authority"
    _write(
        external / "authority.json",
        provider._canonical_json(context.authority) + b"\n",
    )
    (context.paths.work_root / "_orchestration").rmdir()
    (context.paths.work_root / "_orchestration").symlink_to(external)
    authority_path = context.paths.work_root / provider.WORK_AUTHORITY_PATH
    original_read_json = provider._read_json

    def guarded_read(path, **kwargs):
        if path == authority_path:
            pytest.fail("symlinked authority parent would have been followed")
        return original_read_json(path, **kwargs)

    monkeypatch.setattr(provider, "_read_json", guarded_read)
    with pytest.raises(provider.ProviderInputError, match="symlink"):
        provider._initialize_work_root(context)


def test_deterministic_work_root_cannot_alias_canonical_authority(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / ".provider.provider-work"
    with pytest.raises(provider.ProviderInputError, match="overlaps"):
        provider._provider_paths(
            canonical,
            tmp_path / "approval.json",
            tmp_path / "provider",
            repo_root=tmp_path / "repo",
        )


def test_parent_traversal_cannot_alias_final_or_derived_work_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "q").mkdir()
    with pytest.raises(provider.ProviderInputError, match="traversal"):
        provider._provider_paths(
            tmp_path / "canonical",
            tmp_path / "approval.json",
            tmp_path / "q/../canonical/provider",
            repo_root=tmp_path / "repo",
        )
    with pytest.raises(provider.ProviderInputError, match="traversal"):
        provider._provider_paths(
            tmp_path / "q/../.provider.provider-work",
            tmp_path / "approval.json",
            tmp_path / "provider",
            repo_root=tmp_path / "repo",
        )


def test_resume_requires_exact_immutable_work_authority(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    provider._initialize_work_root(context)
    authority_mode = (
        (context.paths.work_root / provider.WORK_AUTHORITY_PATH).stat().st_mode
    )
    assert authority_mode & 0o222 == 0

    drifted = replace(context, authority={"different": True})
    with pytest.raises(provider.ProviderInputError, match="hashes drifted"):
        provider._initialize_work_root(drifted)

    authority_path = context.paths.work_root / provider.WORK_AUTHORITY_PATH
    authority_path.chmod(0o644)
    residue = _write(
        context.paths.work_root / "cohorts/CHOL/sample_axis.txt.tmp.123",
        b"do-not-clean\n",
    )
    with pytest.raises(provider.ProviderInputError, match="not immutable"):
        provider._initialize_work_root(context)
    assert residue.read_bytes() == b"do-not-clean\n"


def test_resume_recovers_only_exact_owned_crash_residues_after_authority(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    context.paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    provider._initialize_work_root(context)
    residue_paths = [
        context.paths.work_root / f".{provider.ROOT_MANIFEST_NAME}.{'a' * 32}.tmp",
        context.paths.work_root / f"resource_readbacks/.{'b' * 32}.json.{'c' * 32}.tmp",
        context.paths.work_root / f"attempts/CHOL/.{'d' * 32}.json.{'e' * 32}.tmp",
        context.paths.work_root / "cohorts/CHOL/sample_axis.txt.tmp.123",
        context.paths.work_root / "cohorts/CHOL/cbase_stage_receipt.tsv.tmp.123",
    ]
    for residue in residue_paths:
        _write(residue, b"partial\n")
    staging = context.paths.work_root / "mutsig/.CHOL.mutsig.A1b2C3"
    _write(staging / "partial-provider-artifact", b"partial\n")

    provider._initialize_work_root(context)

    assert all(not residue.exists() for residue in residue_paths)
    assert not staging.exists()


def test_resume_never_cleans_residue_under_drifted_authority(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    provider._initialize_work_root(context)
    residue = _write(
        context.paths.work_root / "cohorts/CHOL/dig_stage_receipt.tsv.tmp.123",
        b"partial\n",
    )

    with pytest.raises(provider.ProviderInputError, match="hashes drifted"):
        provider._initialize_work_root(replace(context, authority={"drifted": True}))

    assert residue.read_bytes() == b"partial\n"


def test_exclusive_rename_never_replaces_racing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write(source / "source.txt", b"source\n")
    _write(destination / "destination.txt", b"destination\n")

    with pytest.raises(FileExistsError, match="target already exists"):
        provider._rename_exclusive(source, destination)

    assert (source / "source.txt").read_bytes() == b"source\n"
    assert (destination / "destination.txt").read_bytes() == b"destination\n"


def test_exclusive_rename_atomically_publishes_absent_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write(source / "source.txt", b"source\n")

    provider._rename_exclusive(source, destination)

    assert not source.exists()
    assert (destination / "source.txt").read_bytes() == b"source\n"


def test_immutable_manifest_is_read_only_at_exclusive_atomic_visibility(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / provider.ROOT_MANIFEST_NAME
    provider._write_json_atomic(manifest, {"value": 1}, mode=0o444)

    assert manifest.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        provider._write_json_atomic(manifest, {"value": 2}, mode=0o444)
    assert manifest.read_bytes() == b'{"value":1}\n'


def test_same_uid_machine_wide_lease_is_nonblocking_across_outputs(
    tmp_path: Path,
) -> None:
    with provider._host_execution_lease(tmp_path / "first") as lease_path:
        assert lease_path == provider.SAME_UID_MACHINE_LEASE_DIRECTORY / (
            f"dialect-k500-{provider.os.getuid()}.lock"
        )
        with (
            pytest.raises(provider.ProviderInputError, match="same-UID"),
            provider._host_execution_lease(tmp_path / "second"),
        ):
            pytest.fail("second same-UID machine-wide lease was acquired")


def test_same_uid_machine_wide_lease_identity_ignores_tmpdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TMPDIR", (tmp_path / "attacker-selected-temp").as_posix())

    with provider._host_execution_lease(tmp_path / "provider") as lease_path:
        assert lease_path.parent == provider.SAME_UID_MACHINE_LEASE_DIRECTORY


def test_same_uid_machine_wide_lease_rejects_insecure_or_replaced_file(
    tmp_path: Path,
) -> None:
    lease_path = _write(tmp_path / "lease", b"lease\n")
    lease_path.chmod(0o644)
    descriptor = provider.os.open(lease_path, provider.os.O_RDWR)
    try:
        with pytest.raises(provider.ProviderInputError, match="stable private file"):
            provider._require_secure_lease_file(descriptor, lease_path)

        lease_path.chmod(0o600)
        replacement = _write(tmp_path / "replacement", b"replacement\n")
        replacement.chmod(0o600)
        replacement.replace(lease_path)
        with pytest.raises(provider.ProviderInputError, match="stable private file"):
            provider._require_secure_lease_file(descriptor, lease_path)
    finally:
        provider.os.close(descriptor)


def test_closed_inventory_rejects_post_manifest_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "work"
    _initialize_layout(root)
    authority = _write(root / provider.WORK_AUTHORITY_PATH, b"{}\n")
    inventory = provider._require_allowed_work_inventory(root)
    snapshot_summary = {
        "root": "_orchestration/execution-snapshot-" + "e" * 64,
        "tree_hash_contract": provider.TREE_HASH_CONTRACT,
        "tree_sha256": "e" * 64,
        "file_count": 1,
        "directory_count": 1,
        "individual_file_receipts_omitted": True,
    }
    monkeypatch.setattr(
        provider,
        "_execution_snapshot_inventory_summary",
        lambda *_args: snapshot_summary,
    )
    manifest = {
        "inventory": {
            "directories": inventory["directories"],
            "files": [
                provider._file_record(
                    authority,
                    display_path=provider.WORK_AUTHORITY_PATH.as_posix(),
                ),
            ],
            "execution_snapshot": snapshot_summary,
            "root_manifest_excluded_from_self_inventory": provider.ROOT_MANIFEST_NAME,
        },
    }
    provider._verify_manifest_inventory(root, manifest)
    _write(root / "resource_readbacks/00000000000000000000000000000000.json")
    with pytest.raises(
        provider.ProviderInputError,
        match="differs from the filesystem",
    ):
        provider._verify_manifest_inventory(root, manifest)


def test_mutsig_receipt_binds_canonical_maf_axis_sources_and_artifacts(
    tmp_path: Path,
) -> None:
    axis = _write(tmp_path / "canonical/axis.txt", b"s1\n")
    maf = _write(tmp_path / "canonical/CHOL.maf", b"maf\n")
    binding = {
        "canonical_maf": {"path": maf, "file": {"sha256": _digest(maf.read_bytes())}},
        "sample_axis": {"path": axis, "file": {"sha256": _digest(axis.read_bytes())}},
    }
    context = _context(tmp_path, bindings={"CHOL": binding})
    mutsig = context.paths.mutsig_root / "CHOL"
    lambda_path = _write(mutsig / "persample_lambda.f32", b"\0" * 8)
    meta = _write(mutsig / "persample_meta.txt", b"ng\t1\nnp\t1\nneff\t2\n")
    genes = _write(mutsig / "persample_genes.txt", b"G\n")
    patients = _write(mutsig / "persample_patients.txt", b"s1\n")
    values = {
        "schema_version": "1",
        "cohort": "CHOL",
        "upstream_commit": provider.MUTSIG_UPSTREAM_COMMIT,
        "source_tree_sha256": "8" * 64,
        "source_file_count": "1",
        "patch_sha256": "2" * 64,
        "runner_sha256": "3" * 64,
        "runtime_sha256": "6" * 64,
        "maf_sha256": _digest(maf.read_bytes()),
        "sample_axis_sha256": _digest(axis.read_bytes()),
        "sample_axis_count": "1",
        "lambda_sha256": _digest(lambda_path.read_bytes()),
        "meta_sha256": _digest(meta.read_bytes()),
        "genes_sha256": _digest(genes.read_bytes()),
        "patients_sha256": _digest(patients.read_bytes()),
    }
    receipt = mutsig / "persample_receipt.tsv"
    receipt.write_text(
        "".join(f"{name}\t{values[name]}\n" for name in provider.MUTSIG_RECEIPT_FIELDS),
        encoding="utf-8",
    )
    mutsig_record, _genes = provider._validate_mutsig(context, "CHOL", axis)
    assert mutsig_record["dimensions"] == {
        "ng": 1,
        "np": 1,
        "neff": 2,
    }
    values["runtime_sha256"] = "7" * 64
    receipt.write_text(
        "".join(f"{name}\t{values[name]}\n" for name in provider.MUTSIG_RECEIPT_FIELDS),
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="stale or misbound"):
        provider._validate_mutsig(context, "CHOL", axis)
    values["runtime_sha256"] = "6" * 64
    values["maf_sha256"] = "f" * 64
    receipt.write_text(
        "".join(f"{name}\t{values[name]}\n" for name in provider.MUTSIG_RECEIPT_FIELDS),
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="stale or misbound"):
        provider._validate_mutsig(context, "CHOL", axis)
    values["maf_sha256"] = _digest(maf.read_bytes())
    lambda_path.write_bytes(np.array([np.nan, 0], dtype="<f4").tobytes())
    values["lambda_sha256"] = _digest(lambda_path.read_bytes())
    receipt.write_text(
        "".join(f"{name}\t{values[name]}\n" for name in provider.MUTSIG_RECEIPT_FIELDS),
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="NaN"):
        provider._validate_mutsig(context, "CHOL", axis)


def test_k500_gate_requires_common_native_full_observation_support(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    features = tuple(f"G{index}_M" for index in range(provider.TOP_K))
    genes = tuple(f"G{index}" for index in range(provider.TOP_K))
    supported_pmf = {0: 0.5, 1: 0.5}
    cbase_pmfs = dict.fromkeys(features, supported_pmf)
    dig_pmfs = dict.fromkeys(features, supported_pmf)
    semantics = provider.CohortBmrSemantics(
        count_features=features,
        counts_by_feature=dict.fromkeys(features, (1,)),
        cbase_pmfs=cbase_pmfs,
        dig_pmfs=dig_pmfs,
    )
    lambda_path = context.paths.mutsig_root / "CHOL/persample_lambda.f32"
    _write(
        lambda_path,
        np.zeros((provider.TOP_K, 1, 2), dtype="<f4").tobytes(order="F"),
    )

    support = provider._validate_k500_common_support(
        context,
        "CHOL",
        semantics,
        genes,
    )
    assert support["verified_full_support_lower_bound"] == provider.TOP_K

    dig_pmfs = dict(dig_pmfs)
    dig_pmfs[features[-1]] = {2: 1.0}
    unsupported = replace(semantics, dig_pmfs=dig_pmfs)
    with pytest.raises(provider.ProviderInputError, match="Only 499 features"):
        provider._validate_k500_common_support(
            context,
            "CHOL",
            unsupported,
            genes,
        )


def test_cbase_and_dig_receipts_recompute_from_signed_inputs(tmp_path: Path) -> None:
    axis = _write(tmp_path / "canonical/axis.txt", b"s1\n")
    maf = _write(
        tmp_path / "canonical/CHOL.maf",
        (
            b"Chromosome\tStart_Position\tEntrez_Gene_Id\tReference_Allele\t"
            b"Tumor_Seq_Allele2\tTumor_Sample_Barcode\n"
            b"1\t10\t1\tA\tC\ts1\n"
        ),
    )
    bindings = {
        "CHOL": {
            "canonical_maf": {
                "path": maf,
                "file": {"sha256": _digest(maf.read_bytes())},
            },
            "sample_axis": {
                "path": axis,
                "file": {"sha256": _digest(axis.read_bytes())},
            },
        },
    }
    context = _context(tmp_path, bindings=bindings)
    cohort = context.paths.cohort_root / "CHOL"
    cbase_output = cohort / "CBaSE_output"
    for name in provider.COHORT_ROOT_FILES:
        _write(cohort / name)
    (cohort / "pipeline.log").write_bytes(b"")
    (cohort / "sample_axis.txt").write_bytes(axis.read_bytes())
    (cohort / "cbase_input.tsv").write_bytes(
        b"1\t10\t1\tA\tC\ts1\n",
    )
    (cohort / "count_matrix.csv").write_text(
        "sample,A_M\ns1,1\n",
        encoding="utf-8",
    )
    valid_pmf = b"feature,0,1\nA_M,0.9,0.1\n"
    (cohort / "bmr_pmfs.csv").write_bytes(valid_pmf)
    (cohort / "bmr_pmfs.dig.csv").write_bytes(valid_pmf)
    for name in provider.CBASE_OUTPUT_FILES:
        _write(cbase_output / name)
    (cbase_output / "kept_mutations.csv").write_bytes(
        b"sample\tgene\teffect\ns1\tA\tmissense\n",
    )
    (cohort / "gene_level_count_matrix.csv").write_bytes(
        b"sample,A\ns1,1\n",
    )
    (cbase_output / "output_data_preparation.txt").write_text(
        "contract\tN_samples=1\n",
        encoding="utf-8",
    )

    cbase_input_hash = provider._sha256_text_lines(
        [
            "1" * 64,
            _digest(maf.read_bytes()),
            _digest(axis.read_bytes()),
            "4" * 64,
            "c" * 64,
            "5" * 64,
            "hg19",
        ],
    )
    cbase_output_hash = provider._files_sha256(
        [
            cohort / "bmr_pmfs.csv",
            cohort / "count_matrix.csv",
            cbase_output / "q_values.txt",
        ],
    )
    (cohort / "cbase_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{cbase_input_hash}\n"
        f"output_sha256\t{cbase_output_hash}\n",
        encoding="utf-8",
    )
    dig_input_hash = provider._sha256_text_lines(
        [
            "1" * 64,
            _digest(maf.read_bytes()),
            _digest(axis.read_bytes()),
            _digest((cohort / "count_matrix.csv").read_bytes()),
            "4" * 64,
            "d" * 64,
            "5" * 64,
            "1",
            "hg19",
        ],
    )
    dig_output_hash = provider._files_sha256([cohort / "bmr_pmfs.dig.csv"])
    (cohort / "dig_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{dig_input_hash}\n"
        f"output_sha256\t{dig_output_hash}\n",
        encoding="utf-8",
    )

    validated, _semantics = provider._validate_cbase_and_dig(
        context,
        "CHOL",
        maf,
        axis,
    )
    assert validated["cbase"]["persisted_sample_count"] == 1
    (cohort / "bmr_pmfs.csv").write_bytes(b"feature,0,1\nB_M,0.9,0.1\n")
    disjoint_cbase_output_hash = provider._files_sha256(
        [
            cohort / "bmr_pmfs.csv",
            cohort / "count_matrix.csv",
            cbase_output / "q_values.txt",
        ],
    )
    (cohort / "cbase_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{cbase_input_hash}\n"
        f"output_sha256\t{disjoint_cbase_output_hash}\n",
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="do not cover"):
        provider._validate_cbase_and_dig(context, "CHOL", maf, axis)
    (cohort / "bmr_pmfs.csv").write_bytes(valid_pmf)
    (cohort / "cbase_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{cbase_input_hash}\n"
        f"output_sha256\t{cbase_output_hash}\n",
        encoding="utf-8",
    )
    (cohort / "bmr_pmfs.dig.csv").write_bytes(b"feature,0\nA_M,nan\n")
    invalid_dig_output_hash = provider._files_sha256(
        [cohort / "bmr_pmfs.dig.csv"],
    )
    (cohort / "dig_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{dig_input_hash}\n"
        f"output_sha256\t{invalid_dig_output_hash}\n",
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="invalid probability"):
        provider._validate_cbase_and_dig(context, "CHOL", maf, axis)
    (cohort / "bmr_pmfs.dig.csv").write_bytes(valid_pmf)
    (cohort / "count_matrix.csv").write_text(
        "sample,A_M\ns1,2\n",
        encoding="utf-8",
    )
    forged_cbase_output_hash = provider._files_sha256(
        [
            cohort / "bmr_pmfs.csv",
            cohort / "count_matrix.csv",
            cbase_output / "q_values.txt",
        ],
    )
    (cohort / "cbase_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{cbase_input_hash}\n"
        f"output_sha256\t{forged_cbase_output_hash}\n",
        encoding="utf-8",
    )
    forged_dig_input_hash = provider._sha256_text_lines(
        [
            "1" * 64,
            _digest(maf.read_bytes()),
            _digest(axis.read_bytes()),
            _digest((cohort / "count_matrix.csv").read_bytes()),
            "4" * 64,
            "d" * 64,
            "5" * 64,
            "1",
            "hg19",
        ],
    )
    (cohort / "dig_stage_receipt.tsv").write_text(
        "schema_version\t1\n"
        f"input_sha256\t{forged_dig_input_hash}\n"
        f"output_sha256\t{dig_output_hash}\n",
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="do not reproduce"):
        provider._validate_cbase_and_dig(context, "CHOL", maf, axis)


def test_stage_receipt_rejects_duplicate_reordered_or_partial_authority(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.tsv"
    receipt.write_text(
        "schema_version\t1\noutput_sha256\t"
        + "a" * 64
        + "\ninput_sha256\t"
        + "b" * 64
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(provider.ProviderInputError, match="fields/order"):
        provider._read_tsv_receipt(receipt, provider.STAGE_RECEIPT_FIELDS)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("full_inventory_validated", 1),
        ("association_outputs_opened", 0),
    ],
)
def test_full_acceptance_receipt_rejects_bool_integer_substitution(
    field: str,
    forged: int,
) -> None:
    manifest = _synthetic_full_acceptance_manifest()
    manifest_sha256 = "b" * 64
    receipt = provider._full_acceptance_receipt(manifest, manifest_sha256)
    receipt[field] = forged

    with pytest.raises(provider.ProviderInputError, match="forged"):
        provider._validate_full_acceptance_receipt(
            receipt,
            manifest,
            manifest_sha256,
            provider.full_acceptance_receipt_sha256(receipt),
        )


def test_full_acceptance_receipt_requires_independent_digest() -> None:
    manifest = _synthetic_full_acceptance_manifest()
    manifest_sha256 = "b" * 64
    receipt = provider._full_acceptance_receipt(manifest, manifest_sha256)
    parent_digest = provider.full_acceptance_receipt_sha256(receipt)

    with pytest.raises(provider.ProviderInputError, match="independent SHA-256"):
        provider._validate_full_acceptance_receipt(
            receipt,
            manifest,
            manifest_sha256,
            "0" * 64,
        )

    self_issued = copy.deepcopy(receipt)
    self_issued["cohort_receipts_sha256"] = "c" * 64
    with pytest.raises(provider.ProviderInputError, match="independent SHA-256"):
        provider._validate_full_acceptance_receipt(
            self_issued,
            manifest,
            manifest_sha256,
            parent_digest,
        )
    with pytest.raises(provider.ProviderInputError, match="forged"):
        provider._validate_full_acceptance_receipt(
            self_issued,
            manifest,
            manifest_sha256,
            provider.full_acceptance_receipt_sha256(self_issued),
        )

    partial = dict(receipt)
    partial.pop("authority_sha256")
    with pytest.raises(provider.ProviderInputError, match="partial or extra"):
        provider._validate_full_acceptance_receipt(
            partial,
            manifest,
            manifest_sha256,
            provider.full_acceptance_receipt_sha256(partial),
        )


def test_scoped_provider_validator_requires_receipt_digest_argument(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="expected_full_acceptance_receipt_sha256"):
        provider.validate_materialized_provider_cohort_input(  # type: ignore[call-arg]
            tmp_path / "provider",
            "a" * 64,
            "CHOL",
            {},
        )


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_public_provider_validators_require_exact_boolean_runtime_flag(
    tmp_path: Path,
    invalid: object,
) -> None:
    with pytest.raises(provider.ProviderInputError, match="must be a boolean"):
        provider.validate_materialized_provider_cohort_input(
            tmp_path / "provider",
            "a" * 64,
            "CHOL",
            {},
            "b" * 64,
            require_current_execution_environment=invalid,  # type: ignore[arg-type]
        )
    with pytest.raises(provider.ProviderInputError, match="must be a boolean"):
        provider.validate_materialized_provider_input_bundle(
            tmp_path / "provider",
            "a" * 64,
            tmp_path / "canonical",
            "b" * 64,
            tmp_path / "approval.json",
            "c" * 64,
            require_current_execution_environment=invalid,  # type: ignore[arg-type]
        )


def test_public_provider_bundle_api_requires_independent_manifest_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _synthetic_published_bundle(tmp_path)
    canonical_bindings = {
        cohort: {"cohort": cohort} for cohort in provider.TCGA_COHORTS
    }
    observed: dict[str, object] = {}

    def canonical_state(_paths, _hashes, **kwargs):
        observed["require_current"] = kwargs["require_current_execution_environment"]
        return bundle.canonical_manifest, canonical_bindings

    monkeypatch.setattr(provider, "_canonical_bundle_state", canonical_state)
    monkeypatch.setattr(provider, "_verify_manifest_inventory", lambda *_args: None)
    monkeypatch.setattr(
        provider,
        "_validate_published_root",
        lambda *_args, **_kwargs: bundle.manifest,
    )
    monkeypatch.setattr(
        provider,
        "_published_cohort_bindings",
        lambda *_args: {"CHOL": {"cohort": "CHOL"}},
    )
    validated = provider.validate_materialized_provider_input_bundle(
        bundle.root,
        _digest((bundle.root / provider.ROOT_MANIFEST_NAME).read_bytes()),
        bundle.canonical_root,
        bundle.hashes.canonical_input_manifest,
        bundle.approval_path,
        bundle.hashes.approval,
    )
    assert observed["require_current"] is False
    assert validated["root"] == bundle.root
    assert validated["cohort_bindings"] == {"CHOL": {"cohort": "CHOL"}}
    assert validated["full_acceptance_receipt_sha256"] == (
        provider.full_acceptance_receipt_sha256(
            validated["full_acceptance_receipt"],
        )
    )

    with pytest.raises(provider.ProviderInputError, match="independent SHA-256"):
        provider.validate_materialized_provider_input_bundle(
            bundle.root,
            "f" * 64,
            bundle.canonical_root,
            bundle.hashes.canonical_input_manifest,
            bundle.approval_path,
            bundle.hashes.approval,
        )

    monkeypatch.setattr(
        provider,
        "_authority_record",
        lambda *_args: {"drifted": True},
    )
    monkeypatch.setattr(
        provider,
        "_validate_historical_authority_contract",
        lambda *_args, **_kwargs: (
            bundle.hashes.cbase_inputs_tree,
            bundle.hashes.dig_results,
        ),
    )
    monkeypatch.setattr(
        provider,
        "_input_authority_record",
        lambda *_args: bundle.manifest["authority"],
    )
    with pytest.raises(provider.ProviderInputError, match="Current provider"):
        provider.validate_materialized_provider_input_bundle(
            bundle.root,
            _digest((bundle.root / provider.ROOT_MANIFEST_NAME).read_bytes()),
            bundle.canonical_root,
            bundle.hashes.canonical_input_manifest,
            bundle.approval_path,
            bundle.hashes.approval,
            require_current_execution_environment=True,
        )


@pytest.mark.parametrize(
    ("path", "value", "operation"),
    [
        (("scope", "association_identify_invoked"), True, "set"),
        (("scope", "association_outputs_opened"), 0, "set"),
        (("execution", "top_k"), 499, "set"),
        (("execution", "strictly_below_half_logical_cores"), 1, "set"),
        (("execution", "maximum_jobs"), 4, "set"),
        (("authority", "decision_digests", "D1"), "f" * 64, "set"),
        (("authority", "canonical_input_contract"), "forged-contract", "set"),
        (("sources", "extra_source"), "forged", "set"),
        (("sources", "orchestrator"), None, "delete"),
        (("providers", "cbase", "extra_field"), True, "set"),
        (("providers", "mutsig", "runtime"), None, "delete"),
    ],
    ids=(
        "association-scope",
        "false-to-zero",
        "top-k",
        "true-to-one",
        "jobs",
        "decision-digest",
        "canonical-contract",
        "source-extra",
        "source-partial",
        "provider-extra",
        "provider-partial",
    ),
)
def test_public_provider_bundle_historical_replay_rejects_semantic_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
    operation: str,
) -> None:
    bundle = _synthetic_published_bundle(tmp_path)
    work_authority = copy.deepcopy(bundle.work_authority)
    target = work_authority
    for name in path[:-1]:
        nested = target[name]
        assert isinstance(nested, dict)
        target = nested
    if operation == "delete":
        del target[path[-1]]
    else:
        target[path[-1]] = value
    manifest_hash = _republish_synthetic_bundle(bundle, work_authority)
    canonical_bindings = {
        cohort: {"cohort": cohort} for cohort in provider.TCGA_COHORTS
    }
    monkeypatch.setattr(
        provider,
        "_canonical_bundle_state",
        lambda *_args, **_kwargs: (
            bundle.canonical_manifest,
            canonical_bindings,
        ),
    )
    monkeypatch.setattr(provider, "_verify_manifest_inventory", lambda *_args: None)
    monkeypatch.setattr(
        provider,
        "_validate_published_root",
        lambda *_args, **_kwargs: bundle.manifest,
    )

    with pytest.raises(provider.ProviderInputError):
        provider.validate_materialized_provider_input_bundle(
            bundle.root,
            manifest_hash,
            bundle.canonical_root,
            bundle.hashes.canonical_input_manifest,
            bundle.approval_path,
            bundle.hashes.approval,
        )


def test_all_four_independent_hashes_are_mandatory(tmp_path: Path) -> None:
    with pytest.raises(provider.ProviderInputError, match="independently supplied"):
        provider.materialize_tcga_revision_provider_inputs(
            tmp_path / "canonical",
            tmp_path / "approval.json",
            tmp_path / "provider",
            "not-a-hash",
            "b" * 64,
            "c" * 64,
            "d" * 64,
            jobs=1,
        )

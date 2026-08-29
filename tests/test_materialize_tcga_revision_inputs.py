from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import analysis.materialize_tcga_revision_inputs as materializer
from dialect.data.tcga import TCGACaseListReceipt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _lfs_pointer(payload: bytes) -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{_sha256_bytes(payload)}\n"
        f"size {len(payload)}\n"
    ).encode("ascii")


def _sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_record(path: Path, display_path: str) -> dict[str, int | str]:
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _thaw(root: Path) -> None:
    materializer._restore_tree_owner_write(root)  # noqa: SLF001


def _freeze(root: Path) -> None:
    materializer._freeze_tree_read_only(root)  # noqa: SLF001


def _freeze_modes_without_following_symlinks(root: Path) -> None:
    for directory, subdirectories, files in os.walk(
        root,
        topdown=False,
        followlinks=False,
    ):
        base = Path(directory)
        for name in files:
            path = base / name
            if not path.is_symlink():
                path.chmod(materializer._FROZEN_FILE_MODE)  # noqa: SLF001
        for name in subdirectories:
            path = base / name
            if not path.is_symlink():
                path.chmod(materializer._FROZEN_DIRECTORY_MODE)  # noqa: SLF001
        base.chmod(materializer._FROZEN_DIRECTORY_MODE)  # noqa: SLF001


def test_exclusive_publish_never_replaces_racing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_bytes(b"source\n")
    (destination / "racer.txt").write_bytes(b"racer\n")

    with pytest.raises(FileExistsError, match="target already exists"):
        materializer._rename_exclusive(source, destination)  # noqa: SLF001

    assert (source / "source.txt").read_bytes() == b"source\n"
    assert (destination / "racer.txt").read_bytes() == b"racer\n"


def test_exclusive_publish_atomically_moves_absent_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "source.txt").write_bytes(b"source\n")

    materializer._rename_exclusive(source, destination)  # noqa: SLF001

    assert not source.exists()
    assert (destination / "source.txt").read_bytes() == b"source\n"


def test_atomic_write_failure_removes_unpublished_temporary_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"

    def failing_chunks() -> Any:
        yield b"partial"
        msg = "injected stream failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="injected stream failure"):
        materializer._write_chunks_atomic(  # noqa: SLF001
            destination,
            failing_chunks(),
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_dirfd_publication_stays_on_opened_parent_after_path_swap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "source"
    source.mkdir()
    (source / "payload.txt").write_bytes(b"verified\n")
    parent_descriptor = materializer._open_directory_fd(  # noqa: SLF001
        parent,
        label="test parent",
    )
    moved = tmp_path / "opened-parent"
    try:
        parent.rename(moved)
        parent.mkdir()
        materializer._rename_exclusive_at(  # noqa: SLF001
            parent_descriptor,
            "source",
            parent_descriptor,
            "published",
        )
    finally:
        materializer.os.close(parent_descriptor)

    assert (moved / "published/payload.txt").read_bytes() == b"verified\n"
    assert not (parent / "published").exists()


def test_secure_output_parent_creation_rejects_symlink_ancestor(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(materializer.RevisionInputError, match="symlink"):
        materializer._ensure_directory_fd(  # noqa: SLF001
            alias / "nested",
            label="test output parent",
        )

    assert not (target / "nested").exists()


def test_json_hash_and_parse_use_same_open_descriptor_during_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _canonical_bytes({"authority": "verified"})
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(original)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(_canonical_bytes({"authority": "raced"}))
    secure_open = materializer._open_regular_fd  # noqa: SLF001
    swapped = False

    def racing_open(path: Path, *, label: str) -> int:
        nonlocal swapped
        descriptor = secure_open(path, label=label)
        if path == manifest and not swapped:
            swapped = True
            replacement.replace(manifest)
        return descriptor

    monkeypatch.setattr(materializer, "_open_regular_fd", racing_open)
    parsed = materializer._read_json_with_sha256(  # noqa: SLF001
        manifest,
        _sha256_bytes(original),
        label="racing manifest",
    )

    assert parsed == {"authority": "verified"}
    assert json.loads(manifest.read_bytes()) == {"authority": "raced"}


def test_stable_descriptor_rejects_in_place_mutation_after_first_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"a" * (materializer._HASH_CHUNK_BYTES + 8))  # noqa: SLF001
    original = materializer._descriptor_chunks  # noqa: SLF001
    raced = False

    def racing_chunks(descriptor: int) -> Any:
        nonlocal raced
        for chunk in original(descriptor):
            if not raced:
                raced = True
                with path.open("r+b") as handle:
                    handle.seek(0)
                    handle.write(b"b")
                    handle.flush()
                    os.fsync(handle.fileno())
            yield chunk

    monkeypatch.setattr(materializer, "_descriptor_chunks", racing_chunks)

    with pytest.raises(materializer.RevisionInputError, match="changed while"):
        materializer._read_regular_bytes(path, label="racing input")  # noqa: SLF001


def test_stable_copy_rejects_short_descriptor_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"complete bytes")

    monkeypatch.setattr(
        materializer,
        "_descriptor_chunks",
        lambda descriptor: iter((os.read(descriptor, 4),)),
    )

    with pytest.raises(materializer.RevisionInputError, match="byte count"):
        materializer._copy_regular_file(source, destination)  # noqa: SLF001


def test_freeze_normalizes_every_file_and_directory_mode(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "artifact.txt").write_bytes(b"immutable\n")

    _freeze(root)
    inventory = materializer._filesystem_inventory(root)  # noqa: SLF001

    assert stat.S_IMODE(root.stat().st_mode) == 0o500
    assert stat.S_IMODE(nested.stat().st_mode) == 0o500
    assert stat.S_IMODE((nested / "artifact.txt").stat().st_mode) == 0o400
    assert inventory["directory_mode"] == "0500"
    assert inventory["file_mode"] == "0400"


@pytest.mark.parametrize(
    ("target", "mode", "message"),
    [
        ("file", 0o600, "Bundle file is writable"),
        ("directory", 0o700, "Bundle directory is writable"),
    ],
)
def test_inventory_rejects_writable_bundle_entries(
    tmp_path: Path,
    target: str,
    mode: int,
    message: str,
) -> None:
    root = tmp_path / "bundle"
    nested = root / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "artifact.txt"
    artifact.write_bytes(b"immutable\n")
    _freeze(root)
    (artifact if target == "file" else nested).chmod(mode)

    with pytest.raises(materializer.RevisionInputError, match=message):
        materializer._filesystem_inventory(root)  # noqa: SLF001


def test_canonicalizer_executes_exact_verified_bytes_not_path_readback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "variants.py"
    path.write_text("VALUE = 'mutable-path'\n", encoding="utf-8")
    verified = b"VALUE = 'verified-buffer'\n"

    module = materializer._load_canonicalizer_snapshot(  # noqa: SLF001
        verified,
        path=path,
    )

    assert module.VALUE == "verified-buffer"


def test_isolated_bootstrap_rejects_source_tamper_before_policy_execution(
    tmp_path: Path,
) -> None:
    source_bytes = {
        name: path.read_bytes()
        for name, path in materializer._source_dependencies().items()  # noqa: SLF001
    }
    snapshot, digest = materializer._create_execution_snapshot(  # noqa: SLF001
        tmp_path,
        source_bytes,
        require_live_runtime=False,
    )
    materializer_path = snapshot.joinpath(
        *materializer._SOURCE_SNAPSHOT_PATHS["materializer"].split("/"),  # noqa: SLF001
    )
    materializer_path.write_bytes(materializer_path.read_bytes() + b"# tamper\n")
    try:
        with pytest.raises(
            materializer.RevisionInputError,
            match="source receipt mismatch",
        ):
            materializer._run_isolated_snapshot(  # noqa: SLF001
                snapshot,
                digest,
                {"action": "must-not-execute"},
            )
    finally:
        materializer._cleanup_execution_snapshot(snapshot)  # noqa: SLF001


def test_canonical_inventory_rejects_hard_linked_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    source = root / "source.txt"
    source.write_bytes(b"same inode\n")
    materializer.os.link(source, root / "alias.txt")

    with pytest.raises(
        materializer.RevisionInputError,
        match=r"single-link|hard linked",
    ):
        _freeze(root)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":-Infinity}\n',
        b'{"x":1e999}\n',
        b'{"x":"\\ud800"}\n',
    ],
)
def test_json_manifest_parser_rejects_nonfinite_numbers_and_surrogates(
    tmp_path: Path,
    raw: bytes,
) -> None:
    with pytest.raises(materializer.RevisionInputError):
        materializer._parse_json_document(  # noqa: SLF001
            raw,
            path=tmp_path / "manifest.json",
        )


def test_git_lfs_pointer_parser_accepts_exact_real_shape() -> None:
    digest = "a" * 64
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\n"
        "size 12176450\n"
    ).encode("ascii")

    assert len(pointer) == 133
    assert materializer._parse_git_lfs_pointer(pointer) == {  # noqa: SLF001
        "bytes": 12176450,
        "sha256": digest,
    }


def _synthetic_lfs_pointer(  # noqa: PLR0913
    *,
    line_ending: bytes = b"\n",
    algorithm: bytes = b"sha256",
    digest: bytes = b"a" * 64,
    size: bytes = b"10",
    extra_line: bytes | None = None,
    terminal_line_ending: bool = True,
) -> bytes:
    lines = [
        b"version https://git-lfs.github.com/spec/v1",
        b"oid " + algorithm + b":" + digest,
        b"size " + size,
    ]
    if extra_line is not None:
        lines.append(extra_line)
    pointer = line_ending.join(lines) + line_ending
    if terminal_line_ending:
        return pointer
    return pointer[: -len(line_ending)]


@pytest.mark.parametrize(
    "pointer",
    [
        b"direct mutation bytes\n",
        b"version https://git-lfs.github.com/spec/v1\n",
        _synthetic_lfs_pointer(line_ending=b"\r\n"),
        _synthetic_lfs_pointer(digest=b"A" * 64),
        _synthetic_lfs_pointer(algorithm=b"sha512"),
        _synthetic_lfs_pointer(size=b"010"),
        _synthetic_lfs_pointer(size=b"0"),
        _synthetic_lfs_pointer(extra_line=b"ext-extra-line true"),
        _synthetic_lfs_pointer(terminal_line_ending=False),
    ],
)
def test_git_lfs_pointer_parser_rejects_direct_or_malformed_bytes(
    pointer: bytes,
) -> None:
    with pytest.raises(materializer.RevisionInputError, match="LFS pointer"):
        materializer._parse_git_lfs_pointer(pointer)  # noqa: SLF001


def _mock_pinned_git_blob(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    object_id = "1" * 40

    def fake_git_bytes(_git_dir: Path, arguments: Sequence[str]) -> bytes:
        if arguments[0] == "rev-parse":
            return f"{object_id}\n".encode()
        if arguments[:2] == ["cat-file", "-t"]:
            return b"blob\n"
        if arguments[:2] == ["cat-file", "-s"]:
            return f"{len(content)}\n".encode()
        msg = f"unexpected synthetic Git arguments: {arguments}"
        raise AssertionError(msg)

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(content)
            self.stderr = io.BytesIO()

        @staticmethod
        def wait() -> int:
            return 0

        @staticmethod
        def kill() -> None:
            return None

    monkeypatch.setattr(materializer, "_git_bytes", fake_git_bytes)
    monkeypatch.setattr(
        materializer.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )


def test_git_blob_receipt_binds_pointer_and_declared_lfs_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic payload identity"
    pointer = _lfs_pointer(payload)
    _mock_pinned_git_blob(monkeypatch, pointer)

    receipt = materializer._git_blob_receipt(  # noqa: SLF001
        tmp_path,
        "study/data_mutations.txt",
        require_lfs_pointer=True,
    )

    assert receipt == {
        "object_id": "1" * 40,
        "bytes": len(pointer),
        "sha256": _sha256_bytes(pointer),
        "lfs_payload": {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        },
    }


def test_git_blob_receipt_rejects_direct_mutation_blob_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_blob = b"x" * (materializer._MAX_GIT_LFS_POINTER_BYTES + 1)  # noqa: SLF001
    _mock_pinned_git_blob(monkeypatch, direct_blob)
    monkeypatch.setattr(
        materializer.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("direct payload must not be streamed"),
    )

    with pytest.raises(materializer.RevisionInputError, match="direct blob"):
        materializer._git_blob_receipt(  # noqa: SLF001
            tmp_path,
            "study/data_mutations.txt",
            require_lfs_pointer=True,
        )


def test_lfs_receipt_rejects_declared_payload_tamper() -> None:
    payload = b"payload"
    pointer = _lfs_pointer(payload)
    receipt = {
        "object_id": "1" * 40,
        "bytes": len(pointer),
        "sha256": _sha256_bytes(pointer),
        "lfs_payload": {
            "bytes": len(payload),
            "sha256": "0" * 64,
        },
    }

    with pytest.raises(materializer.RevisionInputError, match="reconstruct"):
        materializer._require_git_lfs_blob_record(  # noqa: SLF001
            receipt,
            label="synthetic raw Git LFS blob",
        )


def _maf_rows() -> list[dict[str, str]]:
    base = {
        "Hugo_Symbol": "GENE1",
        "Entrez_Gene_Id": "1",
        "Chromosome": "1",
        "Start_Position": "10",
        "End_Position": "10",
        "Variant_Classification": "Missense_Mutation",
        "Variant_Type": "SNP",
        "Reference_Allele": "A",
        "Tumor_Seq_Allele1": "A",
        "Tumor_Seq_Allele2": "C",
        "NCBI_Build": "GRCh37",
        "Tumor_Sample_Barcode": "S1",
    }
    duplicate = {**base, "Hugo_Symbol": "GENE2", "Entrez_Gene_Id": "2"}
    multiallelic = {**base, "Tumor_Seq_Allele2": "G"}
    unselected = {
        **base,
        "Start_Position": "20",
        "End_Position": "20",
        "Tumor_Seq_Allele2": "T",
        "Tumor_Sample_Barcode": "S2",
    }
    return [base, duplicate, multiallelic, unselected]


def _write_maf(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_population(
    population_root: Path,
    axis: tuple[str, ...],
    receipt: TCGACaseListReceipt,
    case_list_bytes: int,
) -> None:
    cohort_dir = population_root / "CHOL"
    cohort_dir.mkdir(parents=True)
    axis_path = cohort_dir / "sample_axis.txt"
    axis_path.write_bytes(("\n".join(axis) + "\n").encode())
    contract_source = materializer._file_record(  # noqa: SLF001
        materializer._tcga_source_path(),  # noqa: SLF001
        display_path="src/dialect/data/tcga.py",
    )
    generator = materializer._file_record(  # noqa: SLF001
        materializer._population_materializer_source_path(),  # noqa: SLF001
        display_path="analysis/materialize_tcga_revision_population.py",
    )
    population_counts = {
        "source_sample_count": receipt.sample_count,
        "selected_sample_count": receipt.participant_count,
        "participant_count": receipt.participant_count,
        "removed_repeat_participant_samples": (
            receipt.sample_count - receipt.participant_count
        ),
        "ordered_sample_axis_sha256": _sequence_sha256(axis),
        "lexicographically_ordered": True,
        "all_zero_rows_required_for_samples_without_retained_events": True,
    }
    child = {
        "schema_version": materializer.POPULATION_SCHEMA_VERSION,
        "contract": materializer.POPULATION_CONTRACT,
        "cohort": "CHOL",
        "source": {
            "repository": "https://github.com/cBioPortal/datahub",
            "commit": materializer.TCGA_DATAHUB_COMMIT,
            "tree": materializer.TCGA_DATAHUB_TREE,
            "repository_path": materializer.tcga_datahub_case_list_path(
                "CHOL",
            ).as_posix(),
            "case_list_sha256": receipt.sha256,
            "case_list_bytes": case_list_bytes,
        },
        "population": population_counts,
        "selection_policy": materializer._population_selection_policy(),  # noqa: SLF001
        "contract_source": contract_source,
        "outputs": {
            "sample_axis": _file_record(axis_path, "CHOL/sample_axis.txt"),
        },
    }
    child_path = cohort_dir / "population_manifest.json"
    child_path.write_bytes(_canonical_bytes(child))
    root = {
        "schema_version": materializer.POPULATION_SCHEMA_VERSION,
        "contract": materializer.POPULATION_CONTRACT,
        "source": {
            "repository": "https://github.com/cBioPortal/datahub",
            "commit": materializer.TCGA_DATAHUB_COMMIT,
            "tree": materializer.TCGA_DATAHUB_TREE,
        },
        "selection_policy": materializer._population_selection_policy(),  # noqa: SLF001
        "contract_source": contract_source,
        "cohorts": ["CHOL"],
        "cohort_count": 1,
        "cohort_manifests": [
            {"cohort": "CHOL", "manifest_sha256": _sha256(child_path)},
        ],
        "totals": {
            "source_sample_count": population_counts["source_sample_count"],
            "selected_sample_count": population_counts["selected_sample_count"],
            "participant_count": population_counts["participant_count"],
            "removed_repeat_participant_samples": population_counts[
                "removed_repeat_participant_samples"
            ],
        },
        "generator": generator,
    }
    (population_root / "population_manifest.json").write_bytes(
        _canonical_bytes(root),
    )


def _artifact_receipt(path: Path) -> SimpleNamespace:
    content = path.read_bytes()
    return SimpleNamespace(
        path=path.name,
        sha256=_sha256_bytes(content),
        size_bytes=len(content),
        content=content,
    )


def _set_artifact(inputs: SimpleNamespace, decision_id: str, value: object) -> None:
    path = inputs.approval_path.parent / f"{decision_id}.json"
    path.write_bytes(_canonical_bytes(value))
    inputs.approval.decisions[decision_id].canonical_artifact = _artifact_receipt(path)


@pytest.fixture
def synthetic_inputs(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    raw_root = tmp_path / "raw"
    population_root = tmp_path / "population"
    raw_root.mkdir()
    raw_path = raw_root / "CHOL.maf"
    _write_maf(raw_path, _maf_rows())
    axis = ("S1", "S3")
    case_samples = ("S1", "S2", "S3")
    case_content = b"synthetic-case-list\n"
    receipt = TCGACaseListReceipt(
        _sha256_bytes(case_content),
        participant_count=len(axis),
        sample_count=len(case_samples),
    )
    monkeypatch.setattr(
        materializer,
        "TCGA_CASE_LIST_RECEIPTS",
        MappingProxyType({"CHOL": receipt}),
    )
    monkeypatch.setattr(
        materializer.tcga_data,
        "TCGA_MAF_SHA256",
        MappingProxyType({"CHOL": _sha256(raw_path)}),
    )
    monkeypatch.setattr(
        materializer.tcga_data,
        "TCGA_SELECTED_SAMPLE_AXIS_SHA256",
        MappingProxyType({"CHOL": _sequence_sha256(axis)}),
    )
    _write_population(population_root, axis, receipt, len(case_content))
    monkeypatch.setattr(
        materializer,
        "parse_tcga_sequenced_case_list",
        lambda _content, _cohort: case_samples,
    )

    def fake_git_bytes(_git_dir: Path, arguments: Sequence[str]) -> bytes:
        if arguments[0] == "rev-parse":
            if arguments[1].endswith("^{tree}"):
                return f"{materializer.TCGA_DATAHUB_TREE}\n".encode()
            return f"{materializer.TCGA_DATAHUB_COMMIT}\n".encode()
        return case_content

    def fake_git_blob(
        _git_dir: Path,
        repository_path: str,
        *,
        require_lfs_pointer: bool = False,
    ) -> dict[str, Any]:
        is_case_list = repository_path.endswith("cases_sequenced.txt")
        content = case_content if is_case_list else raw_path.read_bytes()
        if require_lfs_pointer:
            pointer = _lfs_pointer(content)
            return {
                "object_id": "1" * 40,
                "bytes": len(pointer),
                "sha256": _sha256_bytes(pointer),
                "lfs_payload": {
                    "bytes": len(content),
                    "sha256": _sha256_bytes(content),
                },
            }
        return {
            "object_id": "2" * 40,
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }

    monkeypatch.setattr(materializer, "_git_bytes", fake_git_bytes)
    monkeypatch.setattr(materializer, "_git_blob_receipt", fake_git_blob)
    runtime_records: dict[str, Path] = {}
    runtime_closures: dict[str, dict[str, object]] = {}
    for package in materializer._PACKAGE_NAMES:  # noqa: SLF001
        record_path = tmp_path / f"{package}.RECORD"
        record_path.write_bytes(f"{package}-record\n".encode())
        runtime_records[package] = record_path
        record_bytes = record_path.read_bytes()
        runtime_closures[package] = {
            "package": package,
            "version": "test-1",
            "files": [
                {
                    "distribution_path": f"{package}-test.dist-info/RECORD",
                    "bytes": len(record_bytes),
                    "sha256": _sha256_bytes(record_bytes),
                },
                {
                    "distribution_path": f"{package}/__init__.py",
                    "bytes": 1,
                    "sha256": "e" * 64,
                },
            ],
        }

    def fake_runtime_identity() -> dict[str, object]:
        packages: dict[str, object] = {}
        for package in materializer._PACKAGE_NAMES:  # noqa: SLF001
            record = runtime_records[package].read_bytes()
            closure_bytes = _canonical_bytes(runtime_closures[package])
            packages[package] = {
                "version": "test-1",
                "record_bytes": len(record),
                "record_sha256": _sha256_bytes(record),
                "closure_files": 2,
                "closure_sha256": _sha256_bytes(closure_bytes),
            }
        return {
            "python": {
                "implementation": "CPython",
                "version": "test",
                "executable": {"bytes": 1, "sha256": "a" * 64},
            },
            "packages": packages,
            "git": {
                "version": "git version test",
                "executable": {"bytes": 1, "sha256": "d" * 64},
            },
        }

    monkeypatch.setattr(
        materializer,
        "_distribution_record_path",
        lambda package: runtime_records[package],
    )
    monkeypatch.setattr(
        materializer,
        "_distribution_runtime_closure",
        lambda package: runtime_closures[package],
    )
    monkeypatch.setattr(materializer, "_runtime_identity", fake_runtime_identity)
    monkeypatch.setattr(
        materializer,
        "_available_memory_bytes",
        lambda: 8 * 1024 * 1024 * 1024,
    )
    git_dir = tmp_path / "datahub.git"
    git_dir.mkdir()
    approval_path = tmp_path / "approval.json"
    approval_content = _canonical_bytes({"synthetic": "approval"})
    approval_path.write_bytes(approval_content)
    approval = SimpleNamespace(
        schema=materializer.STAGE_SCOPED_APPROVAL_SCHEMA,
        manifest_sha256=_sha256_bytes(approval_content),
        allowed_stages=(materializer.MATERIALIZE_FINAL_INPUTS_STAGE,),
        stage_bindings={materializer.MATERIALIZE_FINAL_INPUTS_STAGE: {}},
        decision_digests=MappingProxyType({"D1": "b" * 64, "D2": "c" * 64}),
        decisions={
            "D1": SimpleNamespace(canonical_artifact=None),
            "D2": SimpleNamespace(canonical_artifact=None),
        },
    )
    inputs = SimpleNamespace(
        raw_root=raw_root,
        raw_path=raw_path,
        population_root=population_root,
        git_dir=git_dir,
        approval_path=approval_path,
        approval=approval,
        approval_sha256=_sha256_bytes(approval_content),
        axis=axis,
    )
    population_manifest = json.loads(
        (population_root / "population_manifest.json").read_text(),
    )
    axes = {"CHOL": axis}
    _set_artifact(
        inputs,
        "D1",
        materializer._expected_d1_artifact(  # noqa: SLF001
            _sha256(materializer._canonicalizer_source_path()),  # noqa: SLF001
        ),
    )
    _set_artifact(
        inputs,
        "D2",
        materializer._expected_d2_artifact(  # noqa: SLF001
            population_root,
            population_manifest,
            axes,
            ("CHOL",),
        ),
    )

    def fake_approval(
        _path: Path,
        _expected_sha256: str,
        _required_stage: str,
    ) -> SimpleNamespace:
        return approval

    monkeypatch.setattr(materializer, "validate_revision_approval", fake_approval)
    return inputs


def test_secure_approval_rejects_coauthorized_extra_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "approval.json"
    path.write_bytes(b'{"approval":true}\n')
    digest = _sha256(path)
    coauthorized = SimpleNamespace(
        schema=materializer.STAGE_SCOPED_APPROVAL_SCHEMA,
        manifest_sha256=digest,
        allowed_stages=(
            materializer.MATERIALIZE_FINAL_INPUTS_STAGE,
            "fit-sealed-tcga-k500",
        ),
        stage_bindings={
            materializer.MATERIALIZE_FINAL_INPUTS_STAGE: {},
            "fit-sealed-tcga-k500": {},
        },
        decisions={"D1": object(), "D2": object()},
        decision_digests={"D1": "1" * 64, "D2": "2" * 64},
    )
    monkeypatch.setattr(
        materializer,
        "validate_revision_approval",
        lambda *_args: coauthorized,
    )

    with pytest.raises(materializer.RevisionInputError, match="stage-scoped v5"):
        materializer._secure_approval(path, digest)  # noqa: SLF001


def test_secure_approval_rejects_historical_v4_overattestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "approval.json"
    path.write_bytes(b'{"approval":true}\n')
    digest = _sha256(path)
    historical = SimpleNamespace(
        schema=materializer.approval_data.APPROVAL_SCHEMA,
        manifest_sha256=digest,
        allowed_stages=(materializer.MATERIALIZE_FINAL_INPUTS_STAGE,),
        stage_bindings={materializer.MATERIALIZE_FINAL_INPUTS_STAGE: {}},
        decisions={f"D{index}": object() for index in range(1, 11)},
        decision_digests={f"D{index}": "1" * 64 for index in range(1, 11)},
    )
    monkeypatch.setattr(
        materializer,
        "validate_revision_approval",
        lambda *_args: historical,
    )

    with pytest.raises(materializer.RevisionInputError, match="stage-scoped v5"):
        materializer._secure_approval(path, digest)  # noqa: SLF001


def test_resource_preflight_rejects_low_available_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "CHOL.maf").write_bytes(b"x")
    monkeypatch.setattr(
        materializer,
        "_available_memory_bytes",
        lambda: materializer._MIN_AVAILABLE_MEMORY_BYTES - 1,  # noqa: SLF001
    )

    with pytest.raises(materializer.RevisionInputError, match="Insufficient available"):
        materializer._preflight_materialization_resources(  # noqa: SLF001
            raw,
            tmp_path,
            ("CHOL",),
        )


def test_resource_preflight_rejects_projected_staging_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "CHOL.maf").write_bytes(b"x" * 128)
    monkeypatch.setattr(
        materializer,
        "_available_memory_bytes",
        lambda: 8 * 1024 * 1024 * 1024,
    )
    monkeypatch.setattr(
        materializer.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    with pytest.raises(materializer.RevisionInputError, match="Insufficient free disk"):
        materializer._preflight_materialization_resources(  # noqa: SLF001
            raw,
            tmp_path,
            ("CHOL",),
        )


def test_validation_scratch_is_private_on_bundle_filesystem(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with materializer._private_validation_scratch(  # noqa: SLF001
        bundle,
        "CHOL",
    ) as scratch:
        scratch_stat = scratch.stat()
        parent_stat = bundle.parent.stat()
        assert scratch.parent == bundle.parent
        assert scratch_stat.st_dev == parent_stat.st_dev
        assert scratch_stat.st_uid == os.geteuid()
        assert stat.S_IMODE(scratch_stat.st_mode) == 0o700
        (scratch / "bounded.sqlite3").write_bytes(b"scratch\n")

    assert not scratch.exists()


def _materialize(inputs: SimpleNamespace, output: Path) -> Path:
    return materializer._materialize_tcga_revision_inputs_for_test(  # noqa: SLF001
        inputs.raw_root,
        inputs.population_root,
        inputs.git_dir,
        inputs.approval_path,
        inputs.approval_sha256,
        output,
        cohorts=["CHOL"],
    )


def _validate(inputs: SimpleNamespace, output: Path, **kwargs: Any) -> dict[str, Any]:
    return materializer._validate_materialized_input_bundle_for_test(  # noqa: SLF001
        output,
        _sha256(output / "input_manifest.json"),
        inputs.approval_path,
        inputs.approval_sha256,
        cohorts=["CHOL"],
        **kwargs,
    )


def _rewrite_root_child(output: Path, mutate: Any) -> None:
    _thaw(output)
    child_path = output / "cohorts" / "CHOL.json"
    child = json.loads(child_path.read_text())
    mutate(child)
    child_path.write_bytes(_canonical_bytes(child))
    root_path = output / "input_manifest.json"
    root = json.loads(root_path.read_text())
    root["cohort_manifests"][0]["manifest"] = _file_record(
        child_path,
        "cohorts/CHOL.json",
    )
    root_path.write_bytes(_canonical_bytes(root))
    _freeze(output)


def test_materializer_publishes_and_replays_closed_bundle(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    root = _validate(synthetic_inputs, output)
    canonical = pd.read_csv(output / "mafs" / "CHOL.maf", sep="\t", dtype=str)
    binding = materializer.materialized_cohort_binding(output, root, "CHOL")

    assert len(canonical) == 2
    assert set(canonical["Tumor_Seq_Allele2"]) == {"C", "G"}
    assert (
        root["authority"]["signed_contracts"]["D1"]["content"]["contract"]
        == materializer.D1_CONTRACT
    )
    assert binding["canonical_maf"]["path"] == output / "mafs" / "CHOL.maf"
    assert binding["sample_axis"]["path"] == (
        output / "population" / "CHOL" / "sample_axis.txt"
    )
    assert binding["canonical_maf"]["file"]["sha256"] == _sha256(
        output / "mafs" / "CHOL.maf",
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o500
    assert stat.S_IMODE((output / "mafs/CHOL.maf").stat().st_mode) == 0o400
    assert root["implementation"]["scientific_policy_contract"] == (
        materializer.D1_INPUT_CONTRACT
    )
    assert root["implementation"]["streaming_canonicalization_contract"] == (
        materializer.STREAMING_CANONICALIZATION_CONTRACT
    )
    signed_d1 = root["authority"]["signed_contracts"]["D1"]["content"]
    assert signed_d1["payload"]["input_contract"] == materializer.D1_INPUT_CONTRACT
    assert set(signed_d1["payload"]) == {
        "input_contract",
        "canonicalizer_sha256",
        "duplicate_resolution_policy",
    }


def test_materializer_failure_cleans_staging_tree_and_publication_claim(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle"

    def fail_cohort(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        msg = "injected materialization failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(materializer, "_materialize_cohort", fail_cohort)

    with pytest.raises(RuntimeError, match="injected materialization failure"):
        _materialize(synthetic_inputs, output)

    assert not output.exists()
    assert not materializer._publish_claim_path(output).exists()  # noqa: SLF001
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_cohort_binding_uses_pinned_full_receipt_without_reading_unrelated_large_file(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    manifest_sha256 = _sha256(output / "input_manifest.json")
    root = _validate(synthetic_inputs, output)
    full_validator_calls = 0

    def validated_full_parent(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal full_validator_calls
        full_validator_calls += 1
        return root

    monkeypatch.setattr(
        materializer,
        "validate_materialized_input_bundle",
        validated_full_parent,
    )
    parent = materializer.validate_materialized_input_bundle_with_receipt(
        output,
        manifest_sha256,
        synthetic_inputs.approval_path,
        synthetic_inputs.approval_sha256,
    )
    receipt = parent["receipt"]
    receipt_sha256 = parent["receipt_sha256"]
    assert full_validator_calls == 1
    _thaw(output)
    unrelated = output / "mafs" / "UCEC.maf"
    with unrelated.open("wb") as handle:
        handle.seek(1024 * 1024 * 1024 - 1)
        handle.write(b"\0")
    _freeze(output)
    secure_open = materializer._open_regular_fd  # noqa: SLF001

    def sentinel(path: Path, *, label: str) -> int:
        if path == unrelated:
            pytest.fail("cohort validation read an unrelated 1 GiB sparse MAF")
        return secure_open(path, label=label)

    monkeypatch.setattr(materializer, "_open_regular_fd", sentinel)
    validated = materializer._validate_materialized_input_cohort_binding_for_test(  # noqa: SLF001
        output,
        manifest_sha256,
        synthetic_inputs.approval_path,
        synthetic_inputs.approval_sha256,
        "CHOL",
        receipt,
        receipt_sha256,
        cohorts=["CHOL"],
    )

    assert validated["binding"]["canonical_maf"]["path"] == (
        output / "mafs" / "CHOL.maf"
    )
    assert validated["association_outputs_opened"] is False


def test_cohort_binding_rejects_canonical_maf_cross_bound_to_unrelated_cohort(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    _thaw(output)
    unrelated = output / "mafs" / "UCEC.maf"
    unrelated.write_bytes((output / "mafs" / "CHOL.maf").read_bytes())
    child_path = output / "cohorts" / "CHOL.json"
    child = json.loads(child_path.read_text())
    child["output"]["canonical_maf"] = _file_record(
        unrelated,
        "mafs/UCEC.maf",
    )
    child_path.write_bytes(_canonical_bytes(child))
    root_path = output / "input_manifest.json"
    root = json.loads(root_path.read_text())
    root["cohort_manifests"][0]["manifest"] = _file_record(
        child_path,
        "cohorts/CHOL.json",
    )
    root_path.write_bytes(_canonical_bytes(root))
    _freeze(output)
    secure_open = materializer._open_regular_fd  # noqa: SLF001

    def sentinel(path: Path, *, label: str) -> int:
        if path == unrelated:
            pytest.fail("cross-bound unrelated cohort must not be opened")
        return secure_open(path, label=label)

    monkeypatch.setattr(materializer, "_open_regular_fd", sentinel)

    with pytest.raises(materializer.RevisionInputError, match="cross-bound"):
        materializer.materialized_cohort_binding(output, root, "CHOL")


def test_cohort_binding_rejects_unpinned_or_tampered_full_receipt(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    manifest_sha256 = _sha256(output / "input_manifest.json")
    root = _validate(synthetic_inputs, output)
    receipt = materializer.build_full_input_validation_receipt(
        output,
        root,
        manifest_sha256,
        synthetic_inputs.approval_sha256,
    )
    receipt_sha256 = materializer.full_input_validation_receipt_sha256(receipt)
    changed = {**receipt, "validated_cohort_count": 99}

    with pytest.raises(materializer.RevisionInputError, match="misbound"):
        materializer._validate_materialized_input_cohort_binding_for_test(  # noqa: SLF001
            output,
            manifest_sha256,
            synthetic_inputs.approval_path,
            synthetic_inputs.approval_sha256,
            "CHOL",
            changed,
            receipt_sha256,
            cohorts=["CHOL"],
        )


def test_public_materializer_forbids_subset_publication(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    with pytest.raises(materializer.RevisionInputError, match="32-cohort"):
        materializer.materialize_tcga_revision_inputs(
            synthetic_inputs.raw_root,
            synthetic_inputs.population_root,
            synthetic_inputs.git_dir,
            synthetic_inputs.approval_path,
            synthetic_inputs.approval_sha256,
            tmp_path / "bundle",
            cohorts=["CHOL"],
        )


def test_materializer_rejects_d1_policy_substitution(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    wrong = materializer._expected_d1_artifact(  # noqa: SLF001
        _sha256(materializer._canonicalizer_source_path()),  # noqa: SLF001
    )
    wrong["payload"]["duplicate_resolution_policy"] = {"substituted": True}
    _set_artifact(synthetic_inputs, "D1", wrong)

    with pytest.raises(materializer.RevisionInputError, match="exact executed"):
        _materialize(synthetic_inputs, tmp_path / "bundle")


def test_materializer_rejects_d2_population_substitution(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    path = synthetic_inputs.approval_path.parent / "D2.json"
    wrong = json.loads(path.read_text())
    wrong["payload"]["population_manifest_sha256"] = "0" * 64
    _set_artifact(synthetic_inputs, "D2", wrong)

    with pytest.raises(materializer.RevisionInputError, match="exact executed"):
        _materialize(synthetic_inputs, tmp_path / "bundle")


def test_validator_rejects_unknown_filesystem_artifact(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    _thaw(output)
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    _freeze(output)

    with pytest.raises(materializer.RevisionInputError, match="closed inventory"):
        _validate(synthetic_inputs, output)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda child: child.update({"unknown": True}), "frozen schema"),
        (
            lambda child: child["invariants"].update(
                {"association_outputs_opened": True},
            ),
            "claims do not exactly replay",
        ),
        (
            lambda child: child["output"].update({"rows": 999}),
            "claims do not exactly replay",
        ),
    ],
)
def test_validator_rejects_closed_schema_and_claim_tamper(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    _rewrite_root_child(output, mutate)

    with pytest.raises(materializer.RevisionInputError, match=message):
        _validate(synthetic_inputs, output)


def test_validator_rejects_lfs_payload_lineage_tamper(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")

    def mutate(child: dict[str, Any]) -> None:
        child["source"]["raw_git_blob"]["lfs_payload"]["bytes"] += 1

    _rewrite_root_child(output, mutate)

    with pytest.raises(materializer.RevisionInputError, match="canonical pointer"):
        _validate(synthetic_inputs, output)


def test_immutable_validation_does_not_require_current_checkout(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    original = materializer._source_dependencies()  # noqa: SLF001
    drifted = tmp_path / "drifted.py"
    drifted.write_text("# changed current checkout\n", encoding="utf-8")
    monkeypatch.setattr(
        materializer,
        "_source_dependencies",
        lambda: {**original, "materializer": drifted},
    )

    _validate(synthetic_inputs, output)
    with pytest.raises(materializer.RevisionInputError, match="differs from snapshot"):
        _validate(
            synthetic_inputs,
            output,
            require_current_execution_environment=True,
        )


def test_historical_validation_uses_relocatable_bundled_approval_closure(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    moved = tmp_path / "moved-approval"
    moved.mkdir()
    for name in ("approval.json", "D1.json", "D2.json"):
        (synthetic_inputs.approval_path.parent / name).rename(moved / name)

    root = _validate(synthetic_inputs, output)

    assert (
        root["authority"]["approval_closure"]["independent_manifest_sha256"]
        == synthetic_inputs.approval_sha256
    )
    assert root["authority"]["approval_closure"]["manifest"]["path"] == (
        "authority/approval/approval_manifest.json"
    )


def test_live_validation_requires_current_external_approval(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    synthetic_inputs.approval_path.rename(tmp_path / "approval-moved.json")

    with pytest.raises(materializer.RevisionInputError, match="approval"):
        _validate(
            synthetic_inputs,
            output,
            require_current_execution_environment=True,
        )


def test_validator_requires_exact_boolean_environment_flag(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")

    with pytest.raises(materializer.RevisionInputError, match="exact boolean"):
        materializer._validate_materialized_input_bundle_for_test(  # noqa: SLF001
            output,
            _sha256(output / "input_manifest.json"),
            synthetic_inputs.approval_path,
            synthetic_inputs.approval_sha256,
            cohorts=["CHOL"],
            require_current_execution_environment=1,  # type: ignore[arg-type]
        )


def test_validator_rejects_symlinked_authority_artifact(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    _thaw(output)
    artifact = output / "authority" / "D1.json"
    target = tmp_path / "outside.json"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(target)
    _freeze_modes_without_following_symlinks(output)

    with pytest.raises(materializer.RevisionInputError, match="symlink"):
        _validate(synthetic_inputs, output)


def test_validator_rejects_symlinked_authority_ancestor(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    _thaw(output)
    authority = output / "authority"
    moved = tmp_path / "authority-real"
    authority.rename(moved)
    authority.symlink_to(moved, target_is_directory=True)
    _freeze_modes_without_following_symlinks(output)

    with pytest.raises(materializer.RevisionInputError, match="symlink"):
        _validate(synthetic_inputs, output)


def test_validator_requires_independently_pinned_root_hash(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")

    with pytest.raises(materializer.RevisionInputError, match="independent SHA"):
        materializer._validate_materialized_input_bundle_for_test(  # noqa: SLF001
            output,
            "0" * 64,
            synthetic_inputs.approval_path,
            synthetic_inputs.approval_sha256,
            cohorts=["CHOL"],
        )


def test_materializer_is_row_order_invariant(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _materialize(synthetic_inputs, tmp_path / "first")
    expected = (first / "mafs" / "CHOL.maf").read_bytes()
    _write_maf(synthetic_inputs.raw_path, _maf_rows()[::-1])
    monkeypatch.setattr(
        materializer.tcga_data,
        "TCGA_MAF_SHA256",
        MappingProxyType({"CHOL": _sha256(synthetic_inputs.raw_path)}),
    )
    second = _materialize(synthetic_inputs, tmp_path / "second")

    assert (second / "mafs" / "CHOL.maf").read_bytes() == expected


def test_validator_rejects_canonical_maf_tamper(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    _thaw(output)
    with (output / "mafs" / "CHOL.maf").open("ab") as handle:
        handle.write(b"tamper")
    _freeze(output)

    with pytest.raises(materializer.RevisionInputError, match="byte/hash receipt"):
        _validate(synthetic_inputs, output)


def test_validator_rejects_runtime_native_closure_tamper(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    closure = output / "implementation" / "runtime" / "numpy.closure.json"
    _thaw(output)
    closure.write_bytes(closure.read_bytes() + b"tamper\n")
    _freeze(output)

    with pytest.raises(materializer.RevisionInputError, match="byte/hash receipt"):
        _validate(synthetic_inputs, output)


def test_validator_rejects_bundled_approval_evidence_closure_tamper(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    artifact = output / "authority" / "approval" / "D1.json"
    _thaw(output)
    artifact.write_bytes(artifact.read_bytes() + b"tamper\n")
    _freeze(output)

    with pytest.raises(materializer.RevisionInputError, match="byte/hash receipt"):
        _validate(synthetic_inputs, output)


def test_binding_rejects_manifest_not_on_disk(
    synthetic_inputs: SimpleNamespace,
    tmp_path: Path,
) -> None:
    output = _materialize(synthetic_inputs, tmp_path / "bundle")
    root = _validate(synthetic_inputs, output)
    changed = {**root, "cohort_count": 99}

    with pytest.raises(materializer.RevisionInputError, match="exact manifest"):
        materializer.materialized_cohort_binding(output, changed, "CHOL")


def test_raw_maf_reader_preserves_literal_na_and_numeric_ids(tmp_path: Path) -> None:
    rows = [
        {**_maf_rows()[0], "Tumor_Sample_Barcode": "NA"},
        {**_maf_rows()[0], "Tumor_Sample_Barcode": "001"},
    ]

    _assert_streaming_matches_frozen_frame(
        tmp_path,
        rows,
        selected_samples=frozenset({"NA", "001"}),
    )


def _assert_streaming_matches_frozen_frame(
    tmp_path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    selected_samples: frozenset[str],
) -> None:
    suffix = len(list(tmp_path.iterdir()))
    raw_path = tmp_path / f"raw-{suffix}.maf"
    output_path = tmp_path / f"canonical-{suffix}.maf"
    sqlite_path = tmp_path / f"stream-{suffix}.sqlite3"
    _write_maf(raw_path, rows)
    raw_bytes = raw_path.read_bytes()
    streamed = materializer._stream_canonicalize_maf(  # noqa: SLF001
        raw_path,
        output_path,
        sqlite_path,
        raw_copy_path=None,
        expected_raw_receipt=materializer._FileReceipt(  # noqa: SLF001
            bytes=len(raw_bytes),
            sha256=_sha256_bytes(raw_bytes),
        ),
        selected_samples=selected_samples,
        case_samples=frozenset({str(row["Tumor_Sample_Barcode"]) for row in rows}),
        frozen_canonicalizer=materializer.variant_data,
    )
    selected = pd.DataFrame(rows, dtype=object)
    selected = selected.loc[
        selected["Tumor_Sample_Barcode"].isin(selected_samples),
    ].copy()
    expected, audit = (
        materializer.variant_data.canonicalize_tcga_full_variants_with_audit(
            selected,
        )
    )

    assert output_path.read_bytes() == materializer._serialize_maf_for_test(  # noqa: SLF001
        expected,
    )
    assert streamed.audit == audit
    assert streamed.raw_rows == len(rows)
    assert streamed.selected_rows == len(selected)
    assert streamed.output_rows == len(expected)


def test_sqlite_streaming_is_exact_across_chunk_boundaries_and_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materializer, "_STREAM_CHUNK_ROWS", 1)
    rows = _maf_rows()

    _assert_streaming_matches_frozen_frame(
        tmp_path,
        rows,
        selected_samples=frozenset({"S1", "S3"}),
    )
    _assert_streaming_matches_frozen_frame(
        tmp_path,
        rows[::-1],
        selected_samples=frozenset({"S1", "S3"}),
    )


def test_sqlite_streaming_preserves_global_newbase_fallback_across_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materializer, "_STREAM_CHUNK_ROWS", 1)
    rows = [dict(row) for row in _maf_rows()]
    rows[0]["newbase"] = "c"
    rows[1]["newbase"] = ""
    rows[2]["newbase"] = "g"
    rows[3]["newbase"] = ""

    _assert_streaming_matches_frozen_frame(
        tmp_path,
        rows,
        selected_samples=frozenset({"S1", "S3"}),
    )


def test_streaming_production_path_never_uses_whole_file_pandas_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materializer.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("streaming path called pandas.read_csv"),
    )

    _assert_streaming_matches_frozen_frame(
        tmp_path,
        _maf_rows(),
        selected_samples=frozenset({"S1", "S3"}),
    )


def test_large_synthetic_streaming_peak_rss_stays_bounded(tmp_path: Path) -> None:
    script = tmp_path / "rss_probe.py"
    work = tmp_path / "rss-work"
    work.mkdir()
    script.write_text(
        """
import csv
import hashlib
import resource
import sys
from pathlib import Path

import analysis.materialize_tcga_revision_inputs as materializer

root = Path(sys.argv[1])
raw = root / "synthetic.maf"
canonical = root / "canonical.maf"
database = root / "canonical.sqlite3"
columns = [
    "Hugo_Symbol", "Entrez_Gene_Id", "Chromosome", "Start_Position",
    "End_Position", "Variant_Classification", "Variant_Type",
    "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2",
    "NCBI_Build", "Tumor_Sample_Barcode",
]
digest = hashlib.sha256()
size = 0
with raw.open("wb") as binary:
    class Sink:
        def write(self, text):
            global size
            encoded = text.encode("utf-8")
            digest.update(encoded)
            size += len(encoded)
            binary.write(encoded)
            return len(text)
    writer = csv.writer(Sink(), delimiter="\t", lineterminator="\\n")
    writer.writerow(columns)
    for position in range(1, 300001):
        writer.writerow([
            "GENE", "1", str((position % 22) + 1), str(position), str(position),
            "Missense_Mutation", "SNP", "A", "A", "C", "GRCh37", "S1",
        ])
streamed = materializer._stream_canonicalize_maf(
    raw,
    canonical,
    database,
    raw_copy_path=None,
    expected_raw_receipt=materializer._FileReceipt(size, digest.hexdigest()),
    selected_samples=frozenset({"S1"}),
    case_samples=frozenset({"S1"}),
    frozen_canonicalizer=materializer.variant_data,
)
if streamed.output_rows != 300000:
    raise SystemExit("unexpected output row count")
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_bytes = peak if sys.platform == "darwin" else peak * 1024
print(f"{size} {peak_bytes}")
""".lstrip(),
        encoding="utf-8",
    )

    repo_root = Path(__file__).parent.parent
    current_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = (
        repo_root.as_posix()
        if current_pythonpath is None
        else os.pathsep.join((repo_root.as_posix(), current_pythonpath))
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(script), str(work)],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        text=True,
        cwd=repo_root,
    )
    raw_bytes, peak_bytes = map(int, completed.stdout.strip().split())

    assert raw_bytes > 16 * 1024 * 1024
    assert peak_bytes < 512 * 1024 * 1024

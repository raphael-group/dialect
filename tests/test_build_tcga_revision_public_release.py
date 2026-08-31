"""Adversarial tests for the immutable public-release archive boundary."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, Self

import pytest

from analysis import build_tcga_revision_public_release as public_release

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return public_release._canonical_json(value) + b"\n"  # noqa: SLF001


class _AdversarialScandir:
    """Yield synthetic names and fail if a scanner reads past its claimed cap."""

    def __init__(self, name_at: Callable[[int], str], *, maximum_reads: int) -> None:
        self._name_at = name_at
        self._maximum_reads = maximum_reads
        self.reads = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> SimpleNamespace:
        self.reads += 1
        if self.reads > self._maximum_reads:
            message = "bounded directory scanner read past its explicit cap"
            raise AssertionError(message)
        return SimpleNamespace(name=self._name_at(self.reads))


def _minimal_plan(anchors: Mapping[str, str] | None = None) -> dict[str, object]:
    return {
        "release": {
            "release_id": "synthetic-release",
            "version": "1.0.0",
            "archive_name": "synthetic-release.tar",
            "receipt_name": "synthetic-release.tar.receipt.json",
            "source_commit_a": public_release.SOURCE_COMMIT_A,
            "release_commit_b": "b" * 40,
            "source_tag": "synthetic-v1",
        },
        "anchors": dict(
            anchors
            or {
                key: f"{index + 100:064x}"
                for index, key in enumerate(sorted(public_release._ANCHOR_KEYS))  # noqa: SLF001
            },
        ),
        "approvals": [
            {
                "kind": kind,
                "status": "ready",
                "receipt_id": f"{kind}-receipt",
                "sha256": f"{index + 300:064x}",
            }
            for index, kind in enumerate(("license-review", "public-boundary"))
        ],
        "documents": [
            {
                "document_id": document_id,
                "disposition": "exclude",
                "release_member": None,
                "reason": "No rendered-document visual-QA contract exists.",
            }
            for document_id in ("main", "s1", "rebuttal")
        ],
        "source_dispositions": [],
    }


def _release_parts(
    extra_payloads: Mapping[str, bytes] | None = None,
) -> tuple[dict[str, public_release._ArchiveEntry], bytes, bytes]:
    source_manifest_raw = _canonical({"synthetic": "source-data-manifest"})
    registry_raw = _canonical({"synthetic": "artifact-registry"})
    closure_raw = _canonical({"synthetic": "release-evidence"})
    projection_raw = _canonical({"synthetic": "K500-authority-projection"})
    anchors = {
        key: f"{index + 100:064x}"
        for index, key in enumerate(sorted(public_release._ANCHOR_KEYS))  # noqa: SLF001
    }
    anchors.update(
        {
            "source_data_manifest_sha256": _sha256(source_manifest_raw),
            "artifact_registry_sha256": _sha256(registry_raw),
            "release_evidence_sha256": _sha256(closure_raw),
            "k500_authority_projection_sha256": _sha256(projection_raw),
        },
    )
    plan = _minimal_plan(anchors)
    plan_raw = _canonical(plan)
    payloads = {
        public_release.BUILDER_MEMBER: b"synthetic public-release builder\n",
        public_release.PUBLIC_PLAN_MEMBER: plan_raw,
        public_release.PUBLIC_REGISTRY_MEMBER: registry_raw,
        public_release.PUBLIC_CLOSURE_MEMBER: closure_raw,
        public_release.PUBLIC_PROJECTION_MEMBER: projection_raw,
        f"source-data/{public_release.source_data.SOURCE_DATA_MANIFEST_NAME}": (
            source_manifest_raw
        ),
        "source-data/opaque.bin": b"\x00opaque synthetic fixture\xff",
        **(extra_payloads or {}),
    }
    entries: dict[str, public_release._ArchiveEntry] = {
        member: public_release._MemoryEntry(  # noqa: SLF001
            member=member,
            raw=raw,
            role="synthetic-fixture",
            origin={"kind": "synthetic-test"},
        )
        for member, raw in payloads.items()
    }
    plan_file = SimpleNamespace(sha256=_sha256(plan_raw), size_bytes=len(plan_raw))
    source_receipt = SimpleNamespace(
        source_data_root="/synthetic/source-data",
        manifest_sha256=anchors["source_data_manifest_sha256"],
        file_count=public_release.SOURCE_DATA_FILE_COUNT,
        cohort_count=public_release.SOURCE_DATA_COHORT_COUNT,
        total_bytes=0,
        total_rows=0,
    )
    projection_receipt = SimpleNamespace(
        projection_path="/synthetic/k500-authority-projection.json",
        projection_sha256=anchors["k500_authority_projection_sha256"],
        completion_attestation_sha256=anchors["completion_attestation_sha256"],
        completion_attestation_payload_sha256="3" * 64,
        sealed_completion_sha256=anchors["sealed_completion_sha256"],
        run_manifest_sha256="5" * 64,
        source_a_commit=public_release.SOURCE_COMMIT_A,
        release_b_commit="b" * 40,
        release_tag="synthetic-v1",
        git_blob_count=38,
        generated_file_count=1,
        snapshot_file_count=39,
        execution_snapshot_sha256=public_release.EXECUTION_SNAPSHOT_SHA256,
        authority_digests={
            key: f"{index + 400:064x}"
            for index, key in enumerate(
                public_release.k500_authority_projection.AUTHORITY_DIGEST_FIELDS,
            )
        },
        authority_digest_count=6,
    )
    manifest_raw, checksums_raw = public_release._manifest_and_checksums(  # noqa: SLF001
        plan_file=plan_file,
        plan=plan,
        entries=entries,
        source_lineage={},
        artifact_summary={},
        source_receipt=source_receipt,
        dependency_inventory_sha256="e" * 64,
        dependency_boundaries=[],
        gate_receipts=[],
        k500_projection_receipt=projection_receipt,
    )
    return entries, manifest_raw, checksums_raw


def _prepared_release(
    extra_payloads: Mapping[str, bytes] | None = None,
) -> SimpleNamespace:
    entries, manifest_raw, checksums_raw = _release_parts(extra_payloads)
    return SimpleNamespace(
        entries=tuple(entries.values()),
        manifest_raw=manifest_raw,
        checksums_raw=checksums_raw,
    )


def _publication_case(
    tmp_path: Path,
) -> tuple[SimpleNamespace, Path, Path, Path]:
    prepared = _prepared_release()
    destination = (tmp_path / "destination").resolve()
    destination.mkdir()
    archive_path = destination / "synthetic-release.tar"
    receipt_path = destination / "synthetic-release.tar.receipt.json"
    prepared.config = SimpleNamespace(
        destination_archive=archive_path,
        destination_receipt=receipt_path,
    )
    prepared.plan = _minimal_plan()
    prepared.plan_file = SimpleNamespace(sha256="c" * 64)
    return prepared, destination, archive_path, receipt_path


def _populate_unrelated_entries(root: Path, count: int) -> None:
    for index in range(count):
        (root / f"unrelated-{index:04d}").write_bytes(b"")


def _write_release_archive(path: Path, prepared: SimpleNamespace | None = None) -> None:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        public_release._write_archive(  # noqa: SLF001
            descriptor,
            prepared if prepared is not None else _prepared_release(),
        )
    finally:
        os.close(descriptor)


def _pin(path: Path) -> public_release._PinnedFile:
    return public_release._pin_absolute_file(  # noqa: SLF001
        path.resolve(),
        context="synthetic archive",
    )


def _verify(path: Path, manifest_raw: bytes) -> public_release._VerifiedArchive:
    pinned = _pin(path)
    try:
        return public_release._verify_archive_descriptor(  # noqa: SLF001
            pinned,
            expected_archive_sha256=pinned.sha256,
            expected_manifest_sha256=_sha256(manifest_raw),
        )
    finally:
        pinned.close()


def _tar_bytes(
    members: Sequence[tuple[tarfile.TarInfo, bytes]],
    *,
    tar_format: int = tarfile.USTAR_FORMAT,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output,
        mode="w",
        format=tar_format,
        encoding="ascii",
        errors="strict",
    ) as archive:
        for info, raw in members:
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def _regular_info(name: str) -> tarfile.TarInfo:
    return public_release._tar_info(name, 0)  # noqa: SLF001


def _raw_release_archive(
    *,
    entries: Mapping[str, public_release._ArchiveEntry] | None = None,
    manifest_raw: bytes | None = None,
    checksums_raw: bytes | None = None,
    mutate_info: Callable[[tarfile.TarInfo], None] | None = None,
) -> bytes:
    default_entries, default_manifest, default_checksums = _release_parts()
    selected_entries = default_entries if entries is None else dict(entries)
    selected_manifest = default_manifest if manifest_raw is None else manifest_raw
    selected_checksums = default_checksums if checksums_raw is None else checksums_raw
    raws = {
        **{
            member: entry.raw
            for member, entry in selected_entries.items()
            if isinstance(entry, public_release._MemoryEntry)  # noqa: SLF001
        },
        public_release.ARCHIVE_MANIFEST_MEMBER: selected_manifest,
        public_release.ARCHIVE_CHECKSUM_MEMBER: selected_checksums,
    }
    members: list[tuple[tarfile.TarInfo, bytes]] = []
    for member, raw in sorted(raws.items()):
        info = _regular_info(member)
        if mutate_info is not None:
            mutate_info(info)
        members.append((info, raw))
    return _tar_bytes(members)


def _write_raw(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o400)


def _first_tar_end_offset(raw: bytes) -> int:
    offset = 0
    zero = b"\0" * public_release.TAR_BLOCK_BYTES
    while raw[offset : offset + public_release.TAR_BLOCK_BYTES] != zero:
        header = raw[offset : offset + public_release.TAR_BLOCK_BYTES]
        info = tarfile.TarInfo.frombuf(header, "ascii", "strict")
        offset += public_release.TAR_BLOCK_BYTES
        offset += info.size + (-info.size) % public_release.TAR_BLOCK_BYTES
    return offset


def _manifest_with(
    manifest_raw: bytes,
    mutate: Callable[[dict[str, object]], None],
) -> bytes:
    manifest = json.loads(manifest_raw)
    mutate(manifest)
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = _sha256(
        public_release._canonical_json(payload),  # noqa: SLF001
    )
    return _canonical(manifest)


def _checksums(
    entries: Mapping[str, public_release._ArchiveEntry],
    manifest_raw: bytes,
    *,
    include: Sequence[str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> bytes:
    digests = {member: entry.sha256 for member, entry in entries.items()}
    digests[public_release.ARCHIVE_MANIFEST_MEMBER] = _sha256(manifest_raw)
    selected = sorted(digests if include is None else include)
    replacements = overrides or {}
    return "".join(
        f"{replacements.get(member, digests[member])}  {member}\n"
        for member in selected
    ).encode("ascii")


def test_writer_is_deterministic_canonical_ustar_and_verifies(tmp_path: Path) -> None:
    prepared = _prepared_release()
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    _write_release_archive(first, prepared)
    _write_release_archive(second, prepared)

    first_raw = first.read_bytes()
    assert first_raw == second.read_bytes()
    assert len(first_raw) % public_release.TAR_RECORD_BYTES == 0
    assert first_raw[-(2 * public_release.TAR_BLOCK_BYTES) :] == (
        b"\0" * (2 * public_release.TAR_BLOCK_BYTES)
    )

    verified = _verify(first, prepared.manifest_raw)
    assert verified.archive_sha256 == _sha256(first_raw)
    assert verified.manifest_sha256 == _sha256(prepared.manifest_raw)
    assert verified.release_id == "synthetic-release"
    assert verified.member_count == len(prepared.entries) + 2

    with tarfile.open(first, mode="r:") as archive:
        infos = archive.getmembers()
    assert [info.name for info in infos] == sorted(info.name for info in infos)
    assert all(info.mode == public_release.ARCHIVE_FILE_MODE for info in infos)
    assert all(info.uid == info.gid == info.mtime == 0 for info in infos)
    assert all(info.uname == info.gname == info.linkname == "" for info in infos)
    assert all(info.isfile() and not info.pax_headers for info in infos)


@pytest.mark.parametrize(
    "member",
    [
        "./payload.bin",
        "payload/../payload.bin",
        "../payload.bin",
        "/payload.bin",
        "payload\\item.bin",
        "payload//item.bin",
    ],
)
def test_verifier_rejects_unsafe_or_normalized_aliases(
    tmp_path: Path,
    member: str,
) -> None:
    info = _regular_info(member)
    archive_path = tmp_path / "unsafe.tar"
    _write_raw(archive_path, _tar_bytes([(info, b"unsafe")]))
    pinned = _pin(archive_path)
    try:
        with pytest.raises(public_release.PublicReleaseError):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        pinned.close()


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.GNUTYPE_SPARSE,
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.DIRTYPE,
    ],
)
def test_verifier_rejects_extended_sparse_and_special_headers(
    tmp_path: Path,
    member_type: bytes,
) -> None:
    info = _regular_info("unsafe-header")
    info.type = member_type
    if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
        info.linkname = "target"
    archive_path = tmp_path / "unsafe-header.tar"
    _write_raw(archive_path, _tar_bytes([(info, b"")]))
    pinned = _pin(archive_path)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match=r"immutable USTAR|canonical",
        ):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        pinned.close()


def test_verifier_rejects_gnu_regular_header(tmp_path: Path) -> None:
    archive_path = tmp_path / "gnu.tar"
    _write_raw(
        archive_path,
        _tar_bytes(
            [(_regular_info("regular"), b"payload")],
            tar_format=tarfile.GNU_FORMAT,
        ),
    )
    pinned = _pin(archive_path)
    try:
        with pytest.raises(public_release.PublicReleaseError, match="canonical USTAR"):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        pinned.close()


def test_verifier_rejects_duplicate_members(tmp_path: Path) -> None:
    info_a = _regular_info("duplicate")
    info_b = _regular_info("duplicate")
    archive_path = tmp_path / "duplicate.tar"
    _write_raw(
        archive_path,
        _tar_bytes([(info_a, b"first"), (info_b, b"second")]),
    )
    pinned = _pin(archive_path)
    try:
        with pytest.raises(public_release.PublicReleaseError, match="duplicate"):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        pinned.close()


@pytest.mark.parametrize("suffix", [b"x", b"\0" * tarfile.RECORDSIZE])
def test_verifier_rejects_trailing_or_concatenated_bytes(
    tmp_path: Path,
    suffix: bytes,
) -> None:
    prepared = _prepared_release()
    original = tmp_path / "original.tar"
    _write_release_archive(original, prepared)
    damaged = tmp_path / "damaged.tar"
    _write_raw(damaged, original.read_bytes() + suffix)

    pinned = _pin(damaged)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match=r"trailing|concatenated|noncanonical",
        ):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256=_sha256(prepared.manifest_raw),
            )
    finally:
        pinned.close()


def test_verifier_rejects_concatenated_archive(tmp_path: Path) -> None:
    prepared = _prepared_release()
    original = tmp_path / "original.tar"
    _write_release_archive(original, prepared)
    damaged = tmp_path / "concatenated.tar"
    raw = original.read_bytes()
    _write_raw(damaged, raw + raw)

    pinned = _pin(damaged)
    try:
        with pytest.raises(public_release.PublicReleaseError, match="concatenated"):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256=_sha256(prepared.manifest_raw),
            )
    finally:
        pinned.close()


def test_verifier_requires_two_zero_end_blocks(tmp_path: Path) -> None:
    prepared = _prepared_release()
    original = tmp_path / "original.tar"
    _write_release_archive(original, prepared)
    raw = bytearray(original.read_bytes())
    second_end_block = _first_tar_end_offset(raw) + public_release.TAR_BLOCK_BYTES
    raw[second_end_block] = 1
    damaged = tmp_path / "one-end-block.tar"
    _write_raw(damaged, bytes(raw))

    pinned = _pin(damaged)
    try:
        with pytest.raises(public_release.PublicReleaseError, match="only one TAR end"):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256=_sha256(prepared.manifest_raw),
            )
    finally:
        pinned.close()


def test_verifier_rejects_nonzero_member_padding(tmp_path: Path) -> None:
    prepared = _prepared_release()
    original = tmp_path / "original.tar"
    _write_release_archive(original, prepared)
    raw = bytearray(original.read_bytes())
    first = tarfile.TarInfo.frombuf(
        bytes(raw[: public_release.TAR_BLOCK_BYTES]),
        "ascii",
        "strict",
    )
    assert first.size % public_release.TAR_BLOCK_BYTES != 0
    padding_offset = public_release.TAR_BLOCK_BYTES + first.size
    raw[padding_offset] = 1
    damaged = tmp_path / "nonzero-padding.tar"
    _write_raw(damaged, bytes(raw))

    pinned = _pin(damaged)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match=r"nonzero.*padding",
        ):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256=_sha256(prepared.manifest_raw),
            )
    finally:
        pinned.close()


def test_archive_member_limit_accepts_4096_and_rejects_4097(tmp_path: Path) -> None:
    def payloads_for_archive_member_count(count: int) -> dict[str, bytes]:
        default_entries, _, _ = _release_parts()
        default_count = len(default_entries) + 2
        return {
            f"payload/member-{index:04d}.bin": b""
            for index in range(count - default_count)
        }

    accepted = _prepared_release(
        payloads_for_archive_member_count(public_release.MAX_ARCHIVE_MEMBERS),
    )
    accepted_path = tmp_path / "accepted-limit.tar"
    _write_release_archive(accepted_path, accepted)
    assert _verify(accepted_path, accepted.manifest_raw).member_count == 4096

    rejected = _prepared_release(
        payloads_for_archive_member_count(public_release.MAX_ARCHIVE_MEMBERS + 1),
    )
    rejected_path = tmp_path / "rejected-limit.tar"
    _write_release_archive(rejected_path, rejected)
    pinned = _pin(rejected_path)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="member-count limit",
        ):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256=pinned.sha256,
                expected_manifest_sha256=_sha256(rejected.manifest_raw),
            )
    finally:
        pinned.close()


def test_verifier_rejects_manifest_that_does_not_close_payloads(tmp_path: Path) -> None:
    entries, manifest_raw, _ = _release_parts()

    def omit_payload(manifest: dict[str, object]) -> None:
        members = manifest["members"]
        assert isinstance(members, list)
        members.pop()
        inventory = manifest["inventory"]
        assert isinstance(inventory, dict)
        inventory["payload_member_count"] = len(members)
        inventory["archive_member_count"] = len(members) + 2
        inventory["payload_members_sha256"] = _sha256(
            public_release._canonical_json(members),  # noqa: SLF001
        )

    bad_manifest = _manifest_with(manifest_raw, omit_payload)
    checksums = _checksums(entries, bad_manifest)
    archive_path = tmp_path / "open-manifest.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=bad_manifest,
            checksums_raw=checksums,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match=r"close.*inventory"):
        _verify(archive_path, bad_manifest)


def test_verifier_rejects_incomplete_checksum_coverage(tmp_path: Path) -> None:
    entries, manifest_raw, _ = _release_parts()
    include = [
        public_release.ARCHIVE_MANIFEST_MEMBER,
        public_release.BUILDER_MEMBER,
    ]
    checksums = _checksums(entries, manifest_raw, include=include)
    archive_path = tmp_path / "open-checksums.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="coverage"):
        _verify(archive_path, manifest_raw)


def test_verifier_rejects_sha256sums_self_coverage(tmp_path: Path) -> None:
    entries, manifest_raw, checksums_raw = _release_parts()
    records = checksums_raw.decode("ascii").splitlines()
    records.append(f"{'0' * 64}  {public_release.ARCHIVE_CHECKSUM_MEMBER}")
    checksums = ("\n".join(sorted(records, key=lambda line: line[66:])) + "\n").encode(
        "ascii",
    )
    archive_path = tmp_path / "self-covered-checksums.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="closed policy"):
        _verify(archive_path, manifest_raw)


def test_verifier_rejects_checksum_contradiction(tmp_path: Path) -> None:
    entries, manifest_raw, _ = _release_parts()
    checksums = _checksums(
        entries,
        manifest_raw,
        overrides={public_release.BUILDER_MEMBER: "0" * 64},
    )
    archive_path = tmp_path / "contradictory-checksums.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="contradicts"):
        _verify(archive_path, manifest_raw)


def test_verifier_rejects_manifest_payload_digest_mismatch(tmp_path: Path) -> None:
    entries, manifest_raw, _ = _release_parts()
    manifest = json.loads(manifest_raw)
    manifest["manifest_payload_sha256"] = "0" * 64
    bad_manifest = _canonical(manifest)
    checksums = _checksums(entries, bad_manifest)
    archive_path = tmp_path / "bad-manifest-digest.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=bad_manifest,
            checksums_raw=checksums,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="payload digest"):
        _verify(archive_path, bad_manifest)


def test_verifier_cross_binds_nested_manifest_authorities(tmp_path: Path) -> None:
    entries, manifest_raw, _ = _release_parts()

    def change_builder(manifest: dict[str, object]) -> None:
        manifest["builder"]["sha256"] = "0" * 64

    def change_k500_binding(manifest: dict[str, object]) -> None:
        manifest["k500_authority"]["completion_attestation_sha256"] = "0" * 64

    def omit_document(manifest: dict[str, object]) -> None:
        manifest["documents"].pop()

    for index, (mutate, match) in enumerate(
        (
            (change_builder, "builder is not bound"),
            (change_k500_binding, "K500 authority binding"),
            (omit_document, "account for main, S1, and rebuttal"),
        ),
    ):
        bad_manifest = _manifest_with(manifest_raw, mutate)
        archive_path = tmp_path / f"bad-nested-binding-{index}.tar"
        _write_raw(
            archive_path,
            _raw_release_archive(
                entries=entries,
                manifest_raw=bad_manifest,
                checksums_raw=_checksums(entries, bad_manifest),
            ),
        )
        with pytest.raises(public_release.PublicReleaseError, match=match):
            _verify(archive_path, bad_manifest)


@pytest.mark.parametrize(
    ("member", "expected_result"),
    [
        ("a" * 100, "accepted"),
        ("a" * 101, "rejected"),
        (f"{'p' * 155}/{'n' * 100}", "accepted"),
        (f"{'p' * 156}/name", "rejected"),
        (f"prefix/{'n' * 101}", "rejected"),
    ],
)
def test_ustar_representability_is_field_accurate(
    member: str,
    expected_result: str,
) -> None:
    if expected_result == "accepted":
        assert public_release._canonical_member(member, context="member") == member  # noqa: SLF001
    else:
        with pytest.raises(public_release.PublicReleaseError, match="USTAR"):
            public_release._canonical_member(member, context="member")  # noqa: SLF001


@pytest.mark.parametrize(
    "member",
    [
        "private/source.csv",
        "raw/source.csv",
        "research/private-note.md",
        "private",
        "raw",
        "research",
        "rebuttal",
        "documents",
        public_release.RESTRICTED_EXECUTION_PATH,
    ],
)
def test_public_members_reject_private_or_raw_namespaces(member: str) -> None:
    with pytest.raises(public_release.PublicReleaseError, match="forbidden"):
        public_release._validate_public_member(member, context="member")  # noqa: SLF001


def test_code_allowlist_rejects_exact_forbidden_namespace_leaf() -> None:
    with pytest.raises(public_release.PublicReleaseError, match="forbidden"):
        public_release._normalize_code_paths(  # noqa: SLF001
            ["research"],
            execution_paths=[],
            mode="draft",
        )


def test_code_allowlist_rejects_restricted_execution_patch() -> None:
    with pytest.raises(public_release.PublicReleaseError, match="forbidden"):
        public_release._normalize_code_paths(  # noqa: SLF001
            [public_release.RESTRICTED_EXECUTION_PATH],
            execution_paths=[],
            mode="draft",
        )


def test_verifier_rejects_restricted_execution_patch_archive_member(
    tmp_path: Path,
) -> None:
    entries, manifest_raw, checksums_raw = _release_parts(
        {public_release.RESTRICTED_EXECUTION_PATH: b"restricted patch bytes"},
    )
    archive_path = tmp_path / "restricted-execution-member.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums_raw,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="restricted"):
        _verify(archive_path, manifest_raw)


def test_final_entry_inventory_rejects_restricted_bytes_under_allowed_alias() -> None:
    restricted_raw = b"restricted patch bytes under an allowed alias"
    entries: dict[str, public_release._ArchiveEntry] = {
        "analysis/allowed_renderer.py": public_release._MemoryEntry(  # noqa: SLF001
            member="analysis/allowed_renderer.py",
            raw=restricted_raw,
            role="release-code",
            origin={"kind": "synthetic-test"},
        ),
    }
    with pytest.raises(public_release.PublicReleaseError, match="alias nonpublic"):
        public_release._require_no_nonpublic_entry_bytes(  # noqa: SLF001
            entries,
            {_sha256(restricted_raw)},
        )


def test_outer_hash_is_checked_before_any_tar_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "not-a-tar.tar"
    _write_raw(archive_path, b"not a tar archive")
    pinned = _pin(archive_path)

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(public_release, "_pread_exact", forbidden_read)
    try:
        with pytest.raises(public_release.PublicReleaseError, match="outer SHA-256"):
            public_release._verify_archive_descriptor(  # noqa: SLF001
                pinned,
                expected_archive_sha256="0" * 64,
                expected_manifest_sha256="0" * 64,
            )
    finally:
        pinned.close()


@pytest.mark.parametrize(
    "invalid_field",
    [
        "archive-anchor",
        "manifest-anchor",
        "doi",
        "locator",
        "archive-path",
        "destination-path",
        "same-path",
    ],
)
def test_verify_download_rejects_malformed_inputs_before_any_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: str,
) -> None:
    archive_path = (tmp_path / "download.tar").resolve()
    destination = (tmp_path / "readback.json").resolve()
    arguments: dict[str, object] = {
        "downloaded_archive_path": archive_path,
        "expected_archive_sha256": "a" * 64,
        "expected_manifest_sha256": "b" * 64,
        "doi": "10.1234/synthetic.1",
        "locator": "https://example.org/records/1",
        "destination_receipt": destination,
    }
    replacements = {
        "archive-anchor": ("expected_archive_sha256", "invalid"),
        "manifest-anchor": ("expected_manifest_sha256", "invalid"),
        "doi": ("doi", "invalid"),
        "locator": ("locator", "http://example.org/records/1"),
        "archive-path": ("downloaded_archive_path", Path("relative.tar")),
        "destination-path": ("destination_receipt", Path("relative.json")),
        "same-path": ("destination_receipt", archive_path),
    }
    field_name, replacement = replacements[invalid_field]
    arguments[field_name] = replacement

    def forbidden_open(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(public_release, "_pin_absolute_file", forbidden_open)
    monkeypatch.setattr(public_release, "_destination_parent", forbidden_open)
    with pytest.raises(public_release.PublicReleaseError):
        public_release.verify_download(**arguments)


def test_wrong_outer_hash_hashes_archive_but_never_parses_or_opens_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_release()
    archive_path = (tmp_path / "download.tar").resolve()
    _write_release_archive(archive_path, prepared)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(public_release, "_pread_exact", forbidden)
    monkeypatch.setattr(public_release, "_destination_parent", forbidden)
    with pytest.raises(public_release.PublicReleaseError, match="outer SHA-256"):
        public_release.verify_download(
            archive_path,
            expected_archive_sha256="0" * 64,
            expected_manifest_sha256=_sha256(prepared.manifest_raw),
            doi="10.1234/synthetic.1",
            locator="https://example.org/records/1",
            destination_receipt=(tmp_path / "readback.json").resolve(),
        )


def test_verifier_rejects_noncanonical_crlf_checksums(tmp_path: Path) -> None:
    entries, manifest_raw, checksums_raw = _release_parts()
    archive_path = tmp_path / "crlf-checksums.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums_raw.replace(b"\n", b"\r\n"),
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="canonical ASCII"):
        _verify(archive_path, manifest_raw)


def test_verify_download_publishes_canonical_readback_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_release()
    archive_path = (tmp_path / "download.tar").resolve()
    destination = (tmp_path / "portal-readback.json").resolve()
    _write_release_archive(archive_path, prepared)
    archive_sha256 = _sha256(archive_path.read_bytes())
    manifest_sha256 = _sha256(prepared.manifest_raw)

    def extraction_api_is_forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(public_release.tarfile, "open", extraction_api_is_forbidden)

    receipt = public_release.verify_download(
        archive_path,
        expected_archive_sha256=archive_sha256,
        expected_manifest_sha256=manifest_sha256,
        doi="10.1234/synthetic.1",
        locator="https://example.org/records/1?download=1",
        destination_receipt=destination,
    )

    raw = destination.read_bytes()
    payload = json.loads(raw)
    assert raw == _canonical(payload)
    assert destination.stat().st_mode & 0o777 == 0o400
    assert receipt.archive_sha256 == archive_sha256
    assert receipt.manifest_sha256 == manifest_sha256
    assert receipt.destination_sha256 == _sha256(raw)
    assert payload["verification"] == {
        "outer_sha256_checked_before_tar": True,
        "streamed": True,
        "extracted": False,
        "unsafe_members_rejected": True,
        "manifest_checksum_inventory_closed": True,
    }

    before = destination.read_bytes()
    with pytest.raises(public_release.PublicReleaseError, match="refusing to replace"):
        public_release.verify_download(
            archive_path,
            expected_archive_sha256=archive_sha256,
            expected_manifest_sha256=manifest_sha256,
            doi="10.1234/synthetic.1",
            locator="https://example.org/records/1",
            destination_receipt=destination,
        )
    assert destination.read_bytes() == before


def test_destination_preflight_refuses_existing_file_without_mutation(
    tmp_path: Path,
) -> None:
    destination = (tmp_path / "existing.json").resolve()
    destination.write_bytes(b"do not replace")
    with pytest.raises(public_release.PublicReleaseError, match="refusing to replace"):
        public_release._destination_parent(  # noqa: SLF001
            (destination,),
            forbidden_roots=(),
            context="synthetic publication",
        )
    assert destination.read_bytes() == b"do not replace"


def test_no_replace_rename_loses_race_without_overwriting(tmp_path: Path) -> None:
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    staging = "staging"
    destination = "destination"
    (tmp_path / staging).write_bytes(b"ours")
    (tmp_path / destination).write_bytes(b"competitor")
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="refusing to replace",
        ):
            public_release._rename_staged_no_replace(  # noqa: SLF001
                parent,
                staging,
                destination,
                context="synthetic publication",
            )
    finally:
        parent.close()
    assert (tmp_path / destination).read_bytes() == b"competitor"
    assert (tmp_path / staging).read_bytes() == b"ours"


def test_portal_publication_loses_race_and_retains_private_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "receipt.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    original_rename = public_release._rename_staged_no_replace  # noqa: SLF001

    def racing_rename(
        pinned_parent: public_release._PinnedRoot,
        staging_name: str,
        destination_name: str,
        *,
        context: str,
    ) -> None:
        destination.write_bytes(b"competitor")
        original_rename(
            pinned_parent,
            staging_name,
            destination_name,
            context=context,
        )

    monkeypatch.setattr(public_release, "_rename_staged_no_replace", racing_rename)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="refusing to replace",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"ours\n",
                revalidate_input=lambda: None,
            )
    finally:
        parent.close()
    assert destination.read_bytes() == b"competitor"
    stages = list(tmp_path.glob(".receipt.json.private-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"ours\n"
    assert "private-stage-name-may-be-owned-or-replaced" in str(captured.value)


def test_portal_publication_retains_private_stage_after_parent_replacement(
    tmp_path: Path,
) -> None:
    destination_root = (tmp_path / "destination").resolve()
    moved_root = (tmp_path / "moved-destination").resolve()
    destination_root.mkdir()
    parent = public_release._pin_root(destination_root, context="destination")  # noqa: SLF001
    replaced = False

    def replace_parent_once() -> None:
        nonlocal replaced
        if replaced:
            return
        destination_root.rename(moved_root)
        destination_root.mkdir()
        replaced = True

    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match=r"parent.*changed",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination_root / "receipt.json",
                b"synthetic receipt\n",
                revalidate_input=replace_parent_once,
            )
    finally:
        parent.close()
    assert list(destination_root.iterdir()) == []
    stages = list(moved_root.glob(".receipt.json.private-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"synthetic receipt\n"
    assert "candidate_parent_identity=" in str(captured.value)


def test_fifo_and_directory_are_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = (tmp_path / "pipe").resolve()
    os.mkfifo(fifo)
    with pytest.raises(public_release.PublicReleaseError, match="regular file"):
        public_release._pin_absolute_file(fifo, context="FIFO")  # noqa: SLF001
    with pytest.raises(public_release.PublicReleaseError, match="regular file"):
        public_release._pin_absolute_file(tmp_path.resolve(), context="directory")  # noqa: SLF001

    root = public_release._pin_root(tmp_path.resolve(), context="root")  # noqa: SLF001
    try:
        with pytest.raises(public_release.PublicReleaseError, match="regular file"):
            public_release._pin_member(root, "pipe", context="FIFO member")  # noqa: SLF001
    finally:
        root.close()


def test_oversize_absolute_metadata_is_rejected_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "oversize.json").resolve()
    with path.open("wb") as stream:
        stream.truncate(public_release.MAX_METADATA_BYTES + 1)

    def forbidden_digest(_descriptor: int) -> None:
        message = "oversize metadata must not be hashed"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_digest_descriptor", forbidden_digest)
    with pytest.raises(public_release.PublicReleaseError, match="exceeds"):
        public_release._pin_absolute_file(  # noqa: SLF001
            path,
            context="metadata",
            maximum=public_release.MAX_METADATA_BYTES,
        )


def test_authenticated_member_size_is_checked_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "member.bin"
    with path.open("wb") as stream:
        stream.truncate(1024 * 1024)
    root = public_release._pin_root(tmp_path.resolve(), context="root")  # noqa: SLF001

    def forbidden_digest(_descriptor: int) -> None:
        message = "wrong-size member must not be hashed"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_digest_descriptor", forbidden_digest)
    try:
        with pytest.raises(public_release.PublicReleaseError, match="size differs"):
            public_release._pin_expected_member(  # noqa: SLF001
                root,
                {
                    "member": path.name,
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
                context="authenticated member",
            )
    finally:
        root.close()


def test_invalid_authenticated_member_digest_is_rejected_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = public_release._pin_root(tmp_path.resolve(), context="root")  # noqa: SLF001

    def forbidden_open(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(public_release, "_pin_member", forbidden_open)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="lowercase SHA-256",
        ):
            public_release._pin_expected_member(  # noqa: SLF001
                root,
                {"member": "member.bin", "bytes": 1, "sha256": "invalid"},
                context="authenticated member",
            )
    finally:
        root.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("mode", 0o644),
        ("uid", 1),
        ("gid", 1),
        ("mtime", 1),
        ("uname", "owner"),
        ("gname", "group"),
        ("linkname", "unused-target"),
    ],
)
def test_verifier_rejects_noncanonical_regular_header_metadata(
    tmp_path: Path,
    attribute: str,
    value: int | str,
) -> None:
    entries, manifest_raw, checksums_raw = _release_parts()

    def mutate(info: tarfile.TarInfo) -> None:
        setattr(info, attribute, value)

    archive_path = tmp_path / f"bad-{attribute}.tar"
    _write_raw(
        archive_path,
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums_raw,
            mutate_info=mutate,
        ),
    )
    with pytest.raises(public_release.PublicReleaseError, match="immutable USTAR"):
        _verify(archive_path, manifest_raw)


@pytest.mark.parametrize("field_offset", [329, 337])
def test_verifier_rejects_nonzero_regular_device_fields(
    tmp_path: Path,
    field_offset: int,
) -> None:
    entries, manifest_raw, checksums_raw = _release_parts()
    raw = bytearray(
        _raw_release_archive(
            entries=entries,
            manifest_raw=manifest_raw,
            checksums_raw=checksums_raw,
        ),
    )
    raw[field_offset : field_offset + 8] = tarfile.itn(
        1,
        8,
        tarfile.USTAR_FORMAT,
    )
    raw[148:156] = b" " * 8
    raw[148:156] = tarfile.itn(
        sum(raw[: public_release.TAR_BLOCK_BYTES]),
        8,
        tarfile.USTAR_FORMAT,
    )
    archive_path = tmp_path / f"bad-device-field-{field_offset}.tar"
    _write_raw(archive_path, bytes(raw))
    with pytest.raises(public_release.PublicReleaseError, match="immutable USTAR"):
        _verify(archive_path, manifest_raw)


def _anchored_plan(*, mode: str) -> dict[str, object]:
    execution_paths = [
        {"path": path, "sha256": f"{index + 1:064x}"}
        for index, path in enumerate(
            sorted(public_release.k500_authority_projection.GIT_EXECUTION_PATHS),
        )
    ]
    final = mode == "final"
    return {
        "schema": public_release.PLAN_SCHEMA,
        "contract": public_release.PLAN_CONTRACT,
        "mode": mode,
        "release": {
            "release_id": "synthetic-release",
            "version": "1.0.0",
            "archive_name": "synthetic-release.tar",
            "receipt_name": "synthetic-release.tar.receipt.json",
            "source_commit_a": public_release.SOURCE_COMMIT_A,
            "release_commit_b": "b" * 40,
            "source_tag": "synthetic-v1",
        },
        "anchors": {
            key: f"{index + 100:064x}"
            for index, key in enumerate(sorted(public_release._ANCHOR_KEYS))  # noqa: SLF001
        },
        "approvals": [
            {
                "kind": kind,
                "status": "ready" if final else "pending",
                "receipt_id": f"{kind}-receipt" if final else None,
                "sha256": f"{index + 300:064x}" if final else None,
            }
            for index, kind in enumerate(("license-review", "public-boundary"))
        ],
        "execution": {
            "generated_version_sha256": public_release.GENERATED_VERSION_SHA256,
            "paths": execution_paths if final else [],
        },
        "code_paths": sorted(
            set(public_release.REQUIRED_CODE_PATHS)
            | (
                {str(record["path"]) for record in execution_paths}
                - public_release.RESTRICTED_EXECUTION_PATHS
            ),
        )
        if final
        else [],
        "source_dispositions": [],
        "documents": [
            {
                "document_id": document_id,
                "disposition": "exclude" if final else "pending",
                "release_member": None,
                "reason": "No rendered-document and visual-QA anchor exists.",
            }
            for document_id in ("main", "s1", "rebuttal")
        ],
    }


def _write_plan(path: Path, value: Mapping[str, object]) -> str:
    raw = _canonical(value)
    path.write_bytes(raw)
    return _sha256(raw)


def test_plan_audit_is_result_blind_and_supports_draft_and_final(
    tmp_path: Path,
) -> None:
    draft_path = (tmp_path / "draft.json").resolve()
    draft_sha = _write_plan(draft_path, _anchored_plan(mode="draft"))
    draft = public_release.audit_public_release_plan(
        draft_path,
        expected_plan_sha256=draft_sha,
    )
    assert draft.mode == "draft"
    assert draft.pending_count == 5
    assert draft.ready_to_publish is False

    final_path = (tmp_path / "final.json").resolve()
    final_sha = _write_plan(final_path, _anchored_plan(mode="final"))
    final = public_release.audit_public_release_plan(
        final_path,
        expected_plan_sha256=final_sha,
    )
    assert final.mode == "final"
    assert final.pending_count == 0
    assert final.ready_to_publish is True


def test_final_plan_requires_explicit_zero_document_byte_boundary(
    tmp_path: Path,
) -> None:
    plan = _anchored_plan(mode="final")
    plan["documents"][0]["disposition"] = "pending"
    path = (tmp_path / "invalid.json").resolve()
    digest = _write_plan(path, plan)
    with pytest.raises(public_release.PublicReleaseError, match="pending document"):
        public_release.audit_public_release_plan(
            path,
            expected_plan_sha256=digest,
        )


def test_final_plan_rejects_same_count_execution_path_substitution(
    tmp_path: Path,
) -> None:
    plan = _anchored_plan(mode="final")
    plan["execution"]["paths"][0]["path"] = "analysis/substituted.py"
    plan["execution"]["paths"].sort(key=lambda record: record["path"])
    plan["code_paths"] = sorted(
        set(public_release.REQUIRED_CODE_PATHS)
        | (
            {record["path"] for record in plan["execution"]["paths"]}
            - public_release.RESTRICTED_EXECUTION_PATHS
        ),
    )
    path = (tmp_path / "substituted.json").resolve()
    digest = _write_plan(path, plan)
    with pytest.raises(public_release.PublicReleaseError, match="frozen K500"):
        public_release.audit_public_release_plan(
            path,
            expected_plan_sha256=digest,
        )


def test_exact_code_closure_rejects_every_extra_or_missing_tracked_path() -> None:
    plan = _anchored_plan(mode="final")
    renderer = {"analysis/render_public.py"}
    expected = (
        set(public_release.REQUIRED_CODE_PATHS)
        | renderer
        | (
            {str(record["path"]) for record in plan["execution"]["paths"]}
            - public_release.RESTRICTED_EXECUTION_PATHS
        )
    )
    code = dict.fromkeys(expected, b"synthetic")
    public_release._require_exact_code_closure(code, plan, renderer)  # noqa: SLF001
    with pytest.raises(public_release.PublicReleaseError, match="exact execution"):
        public_release._require_exact_code_closure(  # noqa: SLF001
            {**code, "docs/unselected.txt": b"unselected"},
            plan,
            renderer,
        )
    missing = dict(code)
    missing.pop(next(iter(expected)))
    with pytest.raises(public_release.PublicReleaseError, match="exact execution"):
        public_release._require_exact_code_closure(  # noqa: SLF001
            missing,
            plan,
            renderer,
        )


def test_restricted_execution_patch_is_verified_but_never_returned_as_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _anchored_plan(mode="final")
    blobs: dict[str, bytes] = {}
    for record in plan["execution"]["paths"]:
        member = str(record["path"])
        raw = f"synthetic Git blob for {member}\n".encode()
        blobs[member] = raw
        record["sha256"] = _sha256(raw)
    for member in plan["code_paths"]:
        blobs.setdefault(member, f"synthetic Git blob for {member}\n".encode())

    git_member_reads: list[tuple[str, str]] = []

    def fake_git_require(
        _repository: object,
        _git_executable: object,
        arguments: Sequence[str],
        *,
        context: str,
        max_stdout_bytes: int,
    ) -> bytes:
        del context, max_stdout_bytes
        if arguments[-1].startswith("refs/tags/"):
            return f"{plan['release']['release_commit_b']}\n".encode()
        return f"{arguments[-1].removesuffix('^{commit}')}\n".encode()

    def fake_git_member(
        _repository: object,
        _git_executable: object,
        commit: str,
        member: str,
    ) -> bytes:
        git_member_reads.append((commit, member))
        return blobs[member]

    monkeypatch.setattr(public_release, "_git_require", fake_git_require)
    monkeypatch.setattr(
        public_release,
        "_git_command",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(public_release, "_git_member_absent", lambda *_args: None)
    monkeypatch.setattr(public_release, "_git_commit_member", fake_git_member)
    monkeypatch.setattr(
        public_release,
        "_pin_absolute_file",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        public_release,
        "_read_descriptor",
        lambda *_args, **_kwargs: blobs[public_release.BUILDER_MEMBER],
    )

    code_bytes, lineage = public_release._validate_git_lineage(  # noqa: SLF001
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        plan,
    )

    restricted = public_release.RESTRICTED_EXECUTION_PATH
    assert restricted not in code_bytes
    assert git_member_reads.count((plan["release"]["source_commit_a"], restricted)) == 1
    assert (
        git_member_reads.count((plan["release"]["release_commit_b"], restricted)) == 1
    )
    assert lineage["restricted_execution_paths"] == [
        {
            "path": restricted,
            "dependency_id": public_release.RESTRICTED_EXECUTION_DEPENDENCY_ID,
            "bytes": len(blobs[restricted]),
            "sha256": _sha256(blobs[restricted]),
            "verified_at_source_a_and_release_b": True,
            "included_in_public_release": False,
        },
    ]


def test_restricted_execution_verification_cross_binds_dependency_policy() -> None:
    raw = b"synthetic restricted patch"
    verification = [
        {
            "path": public_release.RESTRICTED_EXECUTION_PATH,
            "dependency_id": public_release.RESTRICTED_EXECUTION_DEPENDENCY_ID,
            "bytes": len(raw),
            "sha256": _sha256(raw),
            "verified_at_source_a_and_release_b": True,
            "included_in_public_release": False,
        },
    ]
    record = {
        "dependency_class": "mutsig_patch",
        "identity": {
            "source": "synthetic restricted patch",
            "patch_bytes": len(raw),
            "patch_sha256": _sha256(raw),
        },
        "license_id": "LicenseRef-Broad-MutSig2CV",
        "license_status": "restricted",
        "redistribution": "exclude",
        "included_in_public_release": False,
        "unresolved": ["Written redistribution permission is unresolved."],
    }
    records = {public_release.RESTRICTED_EXECUTION_DEPENDENCY_ID: record}
    public_release._validate_restricted_execution_boundary(  # noqa: SLF001
        records,
        verification,
    )

    record["identity"]["patch_sha256"] = "0" * 64
    with pytest.raises(public_release.PublicReleaseError, match="contradicts"):
        public_release._validate_restricted_execution_boundary(  # noqa: SLF001
            records,
            verification,
        )
    record["identity"]["patch_sha256"] = _sha256(raw)
    record["redistribution"] = "include"
    with pytest.raises(public_release.PublicReleaseError, match="contradicts"):
        public_release._validate_restricted_execution_boundary(  # noqa: SLF001
            records,
            verification,
        )


def test_excluded_evidence_bytes_are_denylisted_even_under_an_alias() -> None:
    records = [
        {"member": "evidence/selected.csv", "sha256": "a" * 64},
        {"member": "evidence/excluded.csv", "sha256": "b" * 64},
    ]
    dispositions = {
        ("evidence-source", "evidence/selected.csv"): {"disposition": "include"},
        ("evidence-source", "evidence/excluded.csv"): {"disposition": "exclude"},
    }
    assert public_release._excluded_evidence_source_hashes(  # noqa: SLF001
        records,
        dispositions,
    ) == {"b" * 64}
    with pytest.raises(public_release.PublicReleaseError, match="forbidden"):
        public_release._require_includable_source(  # noqa: SLF001
            {
                "dependency_ids": [public_release.INCLUDED_DEPENDENCY_ID],
            },
            digest="b" * 64,
            excluded_hashes={"b" * 64},
        )


def _projection_config(tmp_path: Path, projection_sha256: str) -> SimpleNamespace:
    return SimpleNamespace(
        k500_authority_projection_path=(tmp_path / "projection.json").resolve(),
        expected_k500_authority_projection_sha256=projection_sha256,
    )


def _projection_receipt(
    tmp_path: Path,
    plan: Mapping[str, object],
) -> public_release.k500_authority_projection.K500AuthorityProjectionReceipt:
    anchors = plan["anchors"]
    release = plan["release"]
    return public_release.k500_authority_projection.K500AuthorityProjectionReceipt(
        projection_path=(tmp_path / "projection.json").resolve(),
        projection_sha256=anchors["k500_authority_projection_sha256"],
        completion_attestation_sha256=anchors["completion_attestation_sha256"],
        completion_attestation_payload_sha256="1" * 64,
        sealed_completion_sha256=anchors["sealed_completion_sha256"],
        run_manifest_sha256="2" * 64,
        source_a_commit=release["source_commit_a"],
        release_b_commit=release["release_commit_b"],
        release_tag=release["source_tag"],
        git_blob_count=38,
        generated_file_count=1,
        snapshot_file_count=39,
        execution_snapshot_sha256=public_release.EXECUTION_SNAPSHOT_SHA256,
        authority_digests=MappingProxyType(
            {
                key: f"{index + 400:064x}"
                for index, key in enumerate(
                    public_release.k500_authority_projection.AUTHORITY_DIGEST_FIELDS,
                )
            },
        ),
        authority_digest_count=6,
    )


def test_projection_receipt_cross_binding_is_closed(tmp_path: Path) -> None:
    plan = _anchored_plan(mode="final")
    projection_sha = plan["anchors"]["k500_authority_projection_sha256"]
    config = _projection_config(tmp_path, projection_sha)
    receipt = _projection_receipt(tmp_path, plan)
    public_release._require_projection_receipt(config, plan, receipt)  # noqa: SLF001

    for field, replacement in (
        ("projection_path", (tmp_path / "other-projection.json").resolve()),
        ("sealed_completion_sha256", "f" * 64),
        ("snapshot_file_count", 38),
        ("execution_snapshot_sha256", "e" * 64),
    ):
        invalid = SimpleNamespace(**vars(receipt))
        setattr(invalid, field, replacement)
        with pytest.raises(public_release.PublicReleaseError, match="contradicts"):
            public_release._require_projection_receipt(  # noqa: SLF001
                config,
                plan,
                invalid,
            )


def test_projection_authority_cross_binds_source_data_inputs(tmp_path: Path) -> None:
    plan = _anchored_plan(mode="final")
    receipt = _projection_receipt(tmp_path, plan)
    source_manifest = {
        "authority": {
            key: receipt.authority_digests[key]
            for key in (
                "canonical_input_manifest_sha256",
                "provider_input_manifest_sha256",
            )
        },
    }
    public_release._require_projection_source_authority(  # noqa: SLF001
        source_manifest,
        receipt,
    )
    source_manifest["authority"]["provider_input_manifest_sha256"] = "f" * 64
    with pytest.raises(public_release.PublicReleaseError, match="authority differ"):
        public_release._require_projection_source_authority(  # noqa: SLF001
            source_manifest,
            receipt,
        )


def test_projection_cross_binds_the_pinned_git_executable() -> None:
    pinned = public_release._pin_absolute_file(  # noqa: SLF001
        public_release.GIT_EXECUTABLE,
        context="Git executable",
        maximum=public_release.MAX_GIT_EXECUTABLE_BYTES,
        require_single_link=False,
    )
    try:
        projection = {
            "source": {
                "git_executable": {
                    "path": public_release.GIT_EXECUTABLE.as_posix(),
                    "bytes": pinned.size_bytes,
                    "sha256": pinned.sha256,
                },
            },
        }
        public_release._require_projection_git_executable(  # noqa: SLF001
            projection,
            pinned,
        )
        projection["source"]["git_executable"]["sha256"] = "0" * 64
        with pytest.raises(public_release.PublicReleaseError, match="pinned Git"):
            public_release._require_projection_git_executable(  # noqa: SLF001
                projection,
                pinned,
            )
    finally:
        pinned.close()


def test_native_validator_receipts_bind_exact_supplied_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _anchored_plan(mode="final")
    anchors = plan["anchors"]
    config = SimpleNamespace(
        source_data_root=(tmp_path / "source-data").resolve(),
        artifact_registry_path=(tmp_path / "artifact-registry.json").resolve(),
        release_evidence_path=(tmp_path / "release-evidence.json").resolve(),
        renderer_root=(tmp_path / "renderers").resolve(),
        rendered_output_root=(tmp_path / "rendered-output").resolve(),
        gate_receipt_root=(tmp_path / "gate-receipts").resolve(),
        evidence_source_root=(tmp_path / "evidence-source").resolve(),
        document_reconciliation_path=(tmp_path / "reconciliation.json").resolve(),
        document_anchor_path=(tmp_path / "document-anchor.json").resolve(),
        document_root=(tmp_path / "documents").resolve(),
        k500_authority_projection_path=(tmp_path / "projection.json").resolve(),
        expected_k500_authority_projection_sha256=anchors[
            "k500_authority_projection_sha256"
        ],
        repository_root=tmp_path.resolve(),
    )
    receipts = {
        "source": SimpleNamespace(
            source_data_root=str(config.source_data_root),
            manifest_sha256=anchors["source_data_manifest_sha256"],
            file_count=35,
            cohort_count=32,
            total_bytes=1,
            total_rows=1,
        ),
        "registry": SimpleNamespace(
            manifest_path=str(config.artifact_registry_path),
            manifest_sha256=anchors["artifact_registry_sha256"],
            ready_count=1,
            omitted_count=12,
        ),
        "closure": SimpleNamespace(
            manifest_path=str(config.release_evidence_path),
            manifest_sha256=anchors["release_evidence_sha256"],
            gate_receipt_count=4,
            source_member_count=1,
            ready_count=1,
            omitted_count=12,
        ),
        "document": SimpleNamespace(
            manifest_path=str(config.document_reconciliation_path),
            manifest_sha256=anchors["document_reconciliation_sha256"],
            mode="final",
            placement_count=1,
            ready_count=1,
            omitted_count=12,
            pending_count=0,
        ),
        "projection": _projection_receipt(tmp_path, plan),
    }
    monkeypatch.setattr(
        public_release.source_data,
        "validate_source_data_release",
        lambda *_args, **_kwargs: receipts["source"],
    )
    monkeypatch.setattr(
        public_release.artifact_registry,
        "validate_artifact_registry",
        lambda *_args, **_kwargs: receipts["registry"],
    )
    monkeypatch.setattr(
        public_release.release_evidence,
        "validate_release_evidence_closure",
        lambda *_args, **_kwargs: receipts["closure"],
    )
    monkeypatch.setattr(
        public_release.document_reconciliation,
        "validate_document_reconciliation",
        lambda *_args, **_kwargs: receipts["document"],
    )
    monkeypatch.setattr(
        public_release.k500_authority_projection,
        "validate_k500_authority_projection",
        lambda *_args, **_kwargs: receipts["projection"],
    )

    public_release._run_native_validators(config, plan)  # noqa: SLF001
    for receipt_name, field_name in (
        ("source", "source_data_root"),
        ("registry", "manifest_path"),
        ("closure", "manifest_path"),
        ("document", "manifest_path"),
    ):
        valid = receipts[receipt_name]
        receipts[receipt_name] = SimpleNamespace(**vars(valid))
        setattr(receipts[receipt_name], field_name, str(tmp_path / "other"))
        with pytest.raises(
            public_release.PublicReleaseError,
            match=r"contradicts|closed",
        ):
            public_release._run_native_validators(config, plan)  # noqa: SLF001
        receipts[receipt_name] = valid


def test_git_commands_use_the_pinned_absolute_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []
    monkeypatch.setenv("PATH", "/poisoned/bin")

    def capture(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        observed.extend((arguments, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(public_release.subprocess, "Popen", capture)
    monkeypatch.setattr(
        public_release,
        "_collect_bounded_git_output",
        lambda *_args, **_kwargs: (b"", b"", 0),
    )
    repository = public_release._pin_root(  # noqa: SLF001
        tmp_path.resolve(),
        context="repository",
    )
    git_executable = public_release._pin_absolute_file(  # noqa: SLF001
        public_release.GIT_EXECUTABLE,
        context="Git executable",
        maximum=public_release.MAX_GIT_EXECUTABLE_BYTES,
        require_single_link=False,
    )
    try:
        public_release._git_command(  # noqa: SLF001
            repository,
            git_executable,
            ["status"],
        )
    finally:
        git_executable.close()
        repository.close()
    arguments, kwargs = observed
    environment = kwargs["env"]
    assert arguments[:5] == [
        public_release.GIT_EXECUTABLE.as_posix(),
        "--no-pager",
        "--no-replace-objects",
        "--no-optional-locks",
        "--work-tree=.",
    ]
    assert list(public_release._GIT_CONFIG_OVERRIDES) == arguments[5:-1]  # noqa: SLF001
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_SYSTEM"] == "/dev/null"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert set(kwargs["pass_fds"]) == {
        git_executable.descriptor,
        repository.descriptor,
    }
    assert kwargs["cwd"] is None or str(kwargs["cwd"]).startswith("/proc/self/fd/")
    assert kwargs["preexec_fn"] is not None or kwargs["executable"] is not None
    assert kwargs["start_new_session"] is True


def test_exact_git_show_command_has_a_reachable_stdout_ceiling() -> None:
    repository = public_release._pin_root(  # noqa: SLF001
        Path.cwd().resolve(),
        context="repository",
    )
    git_executable = public_release._pin_absolute_file(  # noqa: SLF001
        public_release.GIT_EXECUTABLE,
        context="Git executable",
        maximum=public_release.MAX_GIT_EXECUTABLE_BYTES,
        require_single_link=False,
    )
    try:
        with pytest.raises(public_release.PublicReleaseError, match="stdout exceeded"):
            public_release._git_command(  # noqa: SLF001
                repository,
                git_executable,
                ["show", f"{public_release.SOURCE_COMMIT_A}:README.md"],
                max_stdout_bytes=1,
            )
    finally:
        git_executable.close()
        repository.close()


def test_bounded_git_output_reaps_process_after_stdout_limit() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time;os.write(1,b'x'*4096);time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    with pytest.raises(public_release.PublicReleaseError, match="stdout exceeded"):
        public_release._collect_bounded_git_output(  # noqa: SLF001
            process,
            max_stdout_bytes=128,
            max_stderr_bytes=128,
            timeout_seconds=2.0,
        )
    assert process.poll() is not None


def test_bounded_git_output_times_out_and_reaps_process() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    with pytest.raises(public_release.PublicReleaseError, match="timeout"):
        public_release._collect_bounded_git_output(  # noqa: SLF001
            process,
            max_stdout_bytes=128,
            max_stderr_bytes=128,
            timeout_seconds=0.05,
        )
    assert process.poll() is not None


def test_descriptor_anchored_git_execution_uses_the_pinned_repository() -> None:
    repository = public_release._pin_root(  # noqa: SLF001
        Path.cwd().resolve(),
        context="repository",
    )
    git_executable = public_release._pin_absolute_file(  # noqa: SLF001
        public_release.GIT_EXECUTABLE,
        context="Git executable",
        maximum=public_release.MAX_GIT_EXECUTABLE_BYTES,
        require_single_link=False,
    )
    try:
        completed = public_release._git_command(  # noqa: SLF001
            repository,
            git_executable,
            ["rev-parse", "--show-toplevel"],
        )
    finally:
        git_executable.close()
        repository.close()
    assert completed.returncode == 0
    assert completed.stdout.strip()


def test_destination_cannot_be_nested_in_any_input_root(tmp_path: Path) -> None:
    input_root = (tmp_path / "document-root").resolve()
    destination_parent = input_root / "nested"
    destination_parent.mkdir(parents=True)
    with pytest.raises(public_release.PublicReleaseError, match="inside an input root"):
        public_release._destination_parent(  # noqa: SLF001
            ((destination_parent / "release.tar").resolve(),),
            forbidden_roots=(input_root,),
            context="public release",
        )


def test_dual_publication_is_no_replace_and_final_readback_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        receipt = public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    assert receipt.archive_sha256 == _sha256(archive_path.read_bytes())
    assert receipt.receipt_sha256 == _sha256(receipt_path.read_bytes())
    assert archive_path.stat().st_mode & 0o777 == 0o400
    assert receipt_path.stat().st_mode & 0o777 == 0o400
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        (archive_path.name, receipt_path.name),
    )
    payload = json.loads(receipt_path.read_bytes())
    assert payload["contract"] == (
        "sequential-per-member-atomic-no-replace-public-release-v1"
    )
    assert payload["publication"] == {
        "pair_atomic": False,
        "publication_order": ["archive", "receipt"],
        "archive_atomic_no_replace": True,
        "receipt_atomic_no_replace": True,
        "partial_publication_retained_for_explicit_reconciliation": True,
        "input_revalidated_after_archive_write": True,
        "submission_package_created": False,
    }


def test_portal_rename_then_raise_reports_ambiguous_names_without_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    original_rename = public_release._rename_staged_no_replace  # noqa: SLF001

    def rename_then_raise(
        pinned_parent: public_release._PinnedRoot,
        staging_name: str,
        destination_name: str,
        *,
        context: str,
    ) -> None:
        original_rename(
            pinned_parent,
            staging_name,
            destination_name,
            context=context,
        )
        message = "synthetic exception after successful rename"
        raise OSError(message)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "failure cleanup must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_rename_staged_no_replace", rename_then_raise)
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="destination-or-private-stage-name",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
    finally:
        parent.close()
    message = str(captured.value)
    assert str(destination) in message
    assert ".readback.json.private-" in message
    assert "expected_sha256=" in message
    assert "expected_bytes=18" in message
    assert destination.read_bytes() == b"synthetic receipt\n"
    assert not list(tmp_path.glob(".readback.json.private-*"))


def test_portal_partial_write_retains_unknown_candidate_and_blocks_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    original_write_all = public_release._write_all  # noqa: SLF001

    def partial_write(descriptor: int, raw: bytes) -> None:
        assert os.write(descriptor, raw[:3]) == 3
        message = "synthetic partial write"
        raise OSError(message)

    monkeypatch.setattr(public_release, "_write_all", partial_write)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="synthetic partial write",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
        message = str(captured.value)
        assert "expected_sha256=unknown" in message
        assert "expected_bytes=unknown" in message
        stages = list(tmp_path.glob(".readback.json.private-*"))
        assert len(stages) == 1
        assert stages[0].read_bytes() == b"syn"
        monkeypatch.setattr(public_release, "_write_all", original_write_all)
        with pytest.raises(
            public_release.PublicReleaseError,
            match="retained private stage requires explicit review",
        ):
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
        assert list(tmp_path.glob(".readback.json.private-*")) == stages
    finally:
        parent.close()


def test_exact_root_inventory_stops_on_first_unexpected_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = public_release._pin_root(tmp_path.resolve(), context="synthetic root")  # noqa: SLF001
    entries = _AdversarialScandir(
        lambda _read: "unexpected",
        maximum_reads=1,
    )
    monkeypatch.setattr(public_release.os, "scandir", lambda _fd: entries)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="unexpected inventory member",
        ):
            public_release._require_exact_root_inventory(  # noqa: SLF001
                root,
                {"expected"},
                context="synthetic exact root",
            )
    finally:
        root.close()
    assert entries.reads == 1


def test_exact_root_inventory_stops_on_first_duplicate_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = public_release._pin_root(tmp_path.resolve(), context="synthetic root")  # noqa: SLF001
    entries = _AdversarialScandir(lambda _read: "a", maximum_reads=2)
    monkeypatch.setattr(public_release.os, "scandir", lambda _fd: entries)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="duplicate inventory member",
        ):
            public_release._require_exact_root_inventory(  # noqa: SLF001
                root,
                {"a", "b"},
                context="synthetic exact root",
            )
    finally:
        root.close()
    assert entries.reads == 2


def test_exact_root_inventory_reports_missing_member_after_stream_end(
    tmp_path: Path,
) -> None:
    root = public_release._pin_root(tmp_path.resolve(), context="synthetic root")  # noqa: SLF001
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="missing inventory members",
        ):
            public_release._require_exact_root_inventory(  # noqa: SLF001
                root,
                {"expected"},
                context="synthetic exact root",
            )
    finally:
        root.close()


def test_publication_stage_preflight_stops_at_bounded_directory_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    entries = _AdversarialScandir(
        lambda read: f"unrelated-{read}",
        maximum_reads=public_release.PUBLICATION_DIRECTORY_SCAN_CAP,
    )
    monkeypatch.setattr(public_release.os, "scandir", lambda _fd: entries)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="exceeds the bounded directory policy",
        ):
            public_release._require_no_retained_stages(  # noqa: SLF001
                parent,
                ("readback.json",),
                context="portal readback",
                reserved_entries=0,
            )
    finally:
        parent.close()
    assert entries.reads == public_release.PUBLICATION_DIRECTORY_SCAN_CAP


@pytest.mark.parametrize(
    "reserved_entries",
    [True, -1, public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES + 1, 1.5],
)
def test_publication_stage_preflight_rejects_invalid_reserved_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reserved_entries: object,
) -> None:
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001

    def forbidden_scan(_descriptor: int) -> NoReturn:
        message = "invalid capacity must fail before directory scanning"
        raise AssertionError(message)

    monkeypatch.setattr(public_release.os, "scandir", forbidden_scan)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="reserved_entries is outside policy",
        ):
            public_release._require_no_retained_stages(  # noqa: SLF001
                parent,
                ("readback.json",),
                context="portal readback",
                reserved_entries=reserved_entries,  # type: ignore[arg-type]
            )
    finally:
        parent.close()


def test_portal_capacity_boundary_succeeds_with_one_reserved_slot(
    tmp_path: Path,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    _populate_unrelated_entries(
        tmp_path,
        public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES - 1,
    )
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    try:
        digest = public_release._publish_portal_readback(  # noqa: SLF001
            parent,
            destination,
            b"synthetic receipt\n",
            revalidate_input=lambda: None,
        )
    finally:
        parent.close()
    assert digest == _sha256(b"synthetic receipt\n")
    assert destination.read_bytes() == b"synthetic receipt\n"
    assert (
        sum(1 for _entry in tmp_path.iterdir())
        == public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES
    )


def test_portal_capacity_one_over_fails_before_stage_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    _populate_unrelated_entries(
        tmp_path,
        public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES,
    )
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001

    def forbidden_create(*_args: object, **_kwargs: object) -> NoReturn:
        message = "over-capacity portal publication must not create a stage"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_create_staging", forbidden_create)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="reserving 1 publication slots",
        ):
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
    finally:
        parent.close()
    assert not destination.exists()
    assert not list(tmp_path.glob(".readback.json.private-*"))


def test_dual_capacity_boundary_succeeds_with_two_reserved_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    _populate_unrelated_entries(
        destination,
        public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES - 2,
    )
    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        receipt = public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    assert receipt.archive_sha256 == _sha256(archive_path.read_bytes())
    assert receipt.receipt_sha256 == _sha256(receipt_path.read_bytes())
    assert (
        sum(1 for _entry in destination.iterdir())
        == public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES
    )


def test_dual_capacity_one_over_fails_before_stage_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    _populate_unrelated_entries(
        destination,
        public_release.MAX_PUBLICATION_DIRECTORY_ENTRIES - 1,
    )
    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)

    def forbidden_create(*_args: object, **_kwargs: object) -> NoReturn:
        message = "over-capacity dual publication must not create a stage"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_create_staging", forbidden_create)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="reserving 2 publication slots",
        ):
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    assert not archive_path.exists()
    assert not receipt_path.exists()
    assert not list(destination.glob(".*.private-candidate"))


def test_retained_stage_preflight_stops_after_first_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001

    class BoundedEntries:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> BoundedEntries:
            return self

        def __next__(self) -> SimpleNamespace:
            if hasattr(self, "returned"):
                message = "scan continued after the first retained stage"
                raise AssertionError(message)
            self.returned = True
            return SimpleNamespace(name=".readback.json.private-retained")

    monkeypatch.setattr(public_release.os, "scandir", lambda _fd: BoundedEntries())
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="retained private stage requires explicit review",
        ):
            public_release._require_no_retained_stages(  # noqa: SLF001
                parent,
                ("readback.json",),
                context="portal readback",
                reserved_entries=0,
            )
    finally:
        parent.close()


def test_deterministic_private_stage_allows_one_concurrent_reservation(
    tmp_path: Path,
) -> None:
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    barrier = threading.Barrier(2)

    def reserve() -> tuple[str, str]:
        barrier.wait()
        try:
            stage, descriptor, _identity = public_release._create_staging(  # noqa: SLF001
                parent,
                "readback.json",
            )
        except public_release.PublicReleaseError as error:
            return "blocked", str(error)
        os.close(descriptor)
        return "reserved", stage

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(lambda _index: reserve(), range(2)))
    finally:
        parent.close()
    reserved = [outcome for outcome in outcomes if outcome[0] == "reserved"]
    blocked = [outcome for outcome in outcomes if outcome[0] == "blocked"]
    assert len(reserved) == 1
    assert len(blocked) == 1
    assert reserved[0][1] == ".readback.json.private-candidate"
    assert "retained private stage requires explicit review" in blocked[0][1]
    assert "private-stage-name-may-be-owned-or-replaced" in blocked[0][1]
    assert "expected_sha256=unknown" in blocked[0][1]
    assert sorted(path.name for path in tmp_path.iterdir()) == [reserved[0][1]]


def test_missing_atomic_rename_symbol_retains_verified_portal_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001

    class MissingRenameLibrary:
        pass

    monkeypatch.setattr(public_release.sys, "platform", "darwin")
    monkeypatch.setattr(
        public_release.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: MissingRenameLibrary(),
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="rename symbol renameatx_np is unavailable",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
    finally:
        parent.close()
    assert "expected_sha256=unknown" not in str(captured.value)
    stages = list(tmp_path.glob(".readback.json.private-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"synthetic receipt\n"


def test_unsupported_atomic_rename_filesystem_retains_verified_portal_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001

    class UnsupportedRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            return -1

    library = SimpleNamespace(renameatx_np=UnsupportedRename())
    monkeypatch.setattr(public_release.sys, "platform", "darwin")
    monkeypatch.setattr(
        public_release.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: library,
    )
    monkeypatch.setattr(
        public_release.ctypes,
        "get_errno",
        lambda: public_release.errno.ENOTSUP,
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="does not support atomic no-replace rename",
        ):
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
    finally:
        parent.close()
    assert not destination.exists()
    assert len(list(tmp_path.glob(".readback.json.private-*"))) == 1


def test_portal_eio_after_issued_rename_is_ambiguous_and_never_deletes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001

    class EioRename:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            destination.write_bytes(b"competitor")
            return -1

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "EIO cleanup must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release.sys, "platform", "darwin")
    monkeypatch.setattr(
        public_release.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameatx_np=EioRename()),
    )
    monkeypatch.setattr(
        public_release.ctypes,
        "get_errno",
        lambda: public_release.errno.EIO,
    )
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="issued syscall outcome is treated as ambiguous",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=lambda: None,
            )
    finally:
        parent.close()
    stage = tmp_path / ".readback.json.private-candidate"
    assert destination.read_bytes() == b"competitor"
    assert stage.read_bytes() == b"synthetic receipt\n"
    message = str(captured.value)
    assert f"candidate_paths={destination}|{stage}" in message
    assert "destination-or-private-stage-name-may-be-owned-or-replaced" in message


def test_dual_publication_retains_archive_and_receipt_stage_on_receipt_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)
    original_rename = public_release._rename_staged_no_replace  # noqa: SLF001
    calls = 0

    def receipt_race(
        parent: public_release._PinnedRoot,
        staging_name: str,
        destination_name: str,
        *,
        context: str,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            receipt_path.write_bytes(b"competitor receipt")
        original_rename(
            parent,
            staging_name,
            destination_name,
            context=context,
        )

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "partial publication must never roll back by name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release, "_rename_staged_no_replace", receipt_race)
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="refusing to replace public release receipt",
        ) as captured:
            public_release._publish_release(prepared, parent)  # noqa: SLF001
        assert archive_path.exists()
        assert receipt_path.read_bytes() == b"competitor receipt"
        archive_stages = list(destination.glob(".synthetic-release.tar.private-*"))
        receipt_stages = list(
            destination.glob(".synthetic-release.tar.receipt.json.private-*"),
        )
        assert archive_stages == []
        assert len(receipt_stages) == 1
        message = str(captured.value)
        assert "candidate_label=archive" in message
        assert "destination-name-may-be-owned-or-replaced" in message
        assert "candidate_label=receipt" in message
        assert "private-stage-name-may-be-owned-or-replaced" in message
        assert f"intended_destination={receipt_path}" in message
        monkeypatch.setattr(
            public_release,
            "_rename_staged_no_replace",
            original_rename,
        )
        with pytest.raises(
            public_release.PublicReleaseError,
            match="retained private stage requires explicit review",
        ):
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()


def test_dual_receipt_eio_is_ambiguous_and_preserves_archive_competitor_and_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)

    class EioOnReceiptRename:
        argtypes: object = None
        restype: object = None

        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            source_fd: int,
            source_name: bytes,
            destination_fd: int,
            destination_name: bytes,
            _flags: int,
        ) -> int:
            self.calls += 1
            if self.calls == 1:
                os.rename(
                    os.fsdecode(source_name),
                    os.fsdecode(destination_name),
                    src_dir_fd=source_fd,
                    dst_dir_fd=destination_fd,
                )
                return 0
            receipt_path.write_bytes(b"competitor receipt")
            return -1

    rename = EioOnReceiptRename()

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "dual EIO cleanup must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release.sys, "platform", "darwin")
    monkeypatch.setattr(
        public_release.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameatx_np=rename),
    )
    monkeypatch.setattr(
        public_release.ctypes,
        "get_errno",
        lambda: public_release.errno.EIO,
    )
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="issued syscall outcome is treated as ambiguous",
        ) as captured:
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    receipt_stage = destination / (
        ".synthetic-release.tar.receipt.json.private-candidate"
    )
    assert archive_path.exists()
    assert receipt_path.read_bytes() == b"competitor receipt"
    assert receipt_stage.exists()
    assert not (destination / ".synthetic-release.tar.private-candidate").exists()
    message = str(captured.value)
    assert "candidate_label=archive" in message
    assert (
        f"candidate_label=receipt,candidate_paths={receipt_path}|{receipt_stage}"
        in message
    )
    assert "destination-or-private-stage-name-may-be-owned-or-replaced" in message


def test_dual_receipt_rename_then_raise_retains_both_destinations_as_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)
    original_rename = public_release._rename_staged_no_replace  # noqa: SLF001
    calls = 0

    def receipt_rename_then_raise(
        parent: public_release._PinnedRoot,
        staging_name: str,
        destination_name: str,
        *,
        context: str,
    ) -> None:
        nonlocal calls
        calls += 1
        original_rename(
            parent,
            staging_name,
            destination_name,
            context=context,
        )
        if calls == 2:
            message = "synthetic exception after successful receipt rename"
            raise OSError(message)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "ambiguous pair publication must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(
        public_release,
        "_rename_staged_no_replace",
        receipt_rename_then_raise,
    )
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="synthetic exception after successful receipt rename",
        ) as captured:
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    assert archive_path.exists()
    assert receipt_path.exists()
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        (archive_path.name, receipt_path.name),
    )
    message = str(captured.value)
    assert "candidate_label=archive" in message
    assert "candidate_label=receipt" in message
    assert (
        "candidate_state=destination-or-private-stage-name-may-be-owned-or-"
        "replaced-do-not-auto-delete"
    ) in message
    assert str(receipt_path) in message
    assert ".synthetic-release.tar.receipt.json.private-" in message


def test_dual_pre_rename_failure_retains_both_fully_verified_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    calls = 0

    def fail_before_rename(_value: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            message = "synthetic pre-rename failure"
            raise public_release.PublicReleaseError(message)

    monkeypatch.setattr(public_release, "_revalidate_prepared", fail_before_rename)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="synthetic pre-rename failure",
        ) as captured:
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    assert not archive_path.exists()
    assert not receipt_path.exists()
    assert len(list(destination.glob(".synthetic-release.tar.private-*"))) == 1
    assert (
        len(
            list(
                destination.glob(
                    ".synthetic-release.tar.receipt.json.private-*",
                ),
            ),
        )
        == 1
    )
    message = str(captured.value)
    assert message.count("expected_sha256=") == 2
    assert "expected_sha256=unknown" not in message


def test_dual_snapshot_cap_failure_retains_published_names_and_stops_scanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    monkeypatch.setattr(public_release, "_revalidate_prepared", lambda _value: None)
    original_scandir = public_release.os.scandir
    saturated = _AdversarialScandir(
        lambda read: f"attacker-entry-{read}",
        maximum_reads=public_release.PUBLICATION_DIRECTORY_SCAN_CAP,
    )
    scans = 0

    def selected_scandir(descriptor: int) -> object:
        nonlocal scans
        scans += 1
        if scans == 2:
            return saturated
        return original_scandir(descriptor)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "bounded snapshot failure must never unlink published names"
        raise AssertionError(message)

    monkeypatch.setattr(public_release.os, "scandir", selected_scandir)
    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="publication directory exceeds the bounded policy",
        ) as captured:
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    monkeypatch.setattr(public_release.os, "scandir", original_scandir)
    assert saturated.reads == public_release.PUBLICATION_DIRECTORY_SCAN_CAP
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        (archive_path.name, receipt_path.name),
    )
    message = str(captured.value)
    assert (
        message.count(
            "destination-name-may-be-owned-or-replaced-do-not-auto-delete",
        )
        == 2
    )


def test_dual_publication_fails_closed_on_final_archive_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, destination, archive_path, receipt_path = _publication_case(tmp_path)
    calls = 0

    def swap_during_final_input_validation(_value: object) -> None:
        nonlocal calls
        calls += 1
        if calls != 5:
            return
        decoy = destination / "archive-decoy"
        decoy.write_bytes(archive_path.read_bytes())
        decoy.chmod(0o400)
        decoy.replace(archive_path)

    monkeypatch.setattr(
        public_release,
        "_revalidate_prepared",
        swap_during_final_input_validation,
    )

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "final readback failure must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)
    parent = public_release._destination_parent(  # noqa: SLF001
        (archive_path, receipt_path),
        forbidden_roots=(),
        context="public release",
    )
    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="final readback",
        ) as captured:
            public_release._publish_release(prepared, parent)  # noqa: SLF001
    finally:
        parent.close()
    assert archive_path.read_bytes()
    assert receipt_path.read_bytes()
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        (archive_path.name, receipt_path.name),
    )
    assert (
        str(captured.value).count(
            "destination-name-may-be-owned-or-replaced-do-not-auto-delete",
        )
        == 2
    )


def test_portal_readback_fails_closed_on_final_receipt_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (tmp_path / "readback.json").resolve()
    parent = public_release._pin_root(tmp_path.resolve(), context="destination")  # noqa: SLF001
    calls = 0

    def swap_during_final_input_validation() -> None:
        nonlocal calls
        calls += 1
        if calls != 3:
            return
        decoy = tmp_path / "receipt-decoy"
        decoy.write_bytes(destination.read_bytes())
        decoy.chmod(0o400)
        decoy.replace(destination)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        message = "portal final readback failure must never unlink a mutable name"
        raise AssertionError(message)

    monkeypatch.setattr(public_release.os, "unlink", forbidden_unlink)

    try:
        with pytest.raises(
            public_release.PublicReleaseError,
            match="final readback",
        ) as captured:
            public_release._publish_portal_readback(  # noqa: SLF001
                parent,
                destination,
                b"synthetic receipt\n",
                revalidate_input=swap_during_final_input_validation,
            )
    finally:
        parent.close()
    assert destination.read_bytes() == b"synthetic receipt\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]
    assert "destination-name-may-be-owned-or-replaced" in str(captured.value)

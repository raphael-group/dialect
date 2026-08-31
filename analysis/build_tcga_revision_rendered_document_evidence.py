"""Audit exact rendered-document and all-page visual-QA wrapper evidence.

This result-blind boundary starts from an independently hash-anchored document
reconciliation.  Its native validator must succeed before any rendered PDF or PNG
byte is opened.  The builder then closes an exact four-PDF inventory, the exact
one-PNG-per-page render inventory, machine-reported PDF/font/encryption/page-size
checks, baseline/revised source-snapshot anchors, and a separate independently
anchored human visual-review receipt.

PDF bytes receive only bounded signature/EOF checks, while PNGs are structurally
parsed and their bounded scanline streams are decoded.  Machine evidence and human
review are recorded, not recreated or inferred here.  In particular, this contract
does not establish scientific correctness, general text legibility, NAAS
compliance, reviewer identity, coauthor approval, portal upload/readback, or
submission approval.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import errno
import hashlib
import importlib
import json
import os
import re
import resource
import stat
import struct
import sys
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_reconciliation = importlib.import_module(
    "analysis.build_tcga_revision_document_reconciliation",
)
_release_evidence = importlib.import_module(
    "analysis.build_tcga_revision_release_evidence",
)

# Error messages intentionally expose exact schema paths.  This module has a large
# validation surface because publication and validation share one sealed path.
# ruff: noqa: PLR0913

RENDERED_DOCUMENT_INPUT_SCHEMA: Final = (
    "dialect-revision-rendered-document-evidence-input-v1"
)
SOURCE_SNAPSHOT_ANCHOR_SCHEMA: Final = (
    "dialect-revision-document-source-snapshot-anchor-v1"
)
DERIVATION_EVIDENCE_SCHEMA: Final = (
    "dialect-revision-rendered-document-derivation-evidence-v1"
)
MACHINE_EVIDENCE_SCHEMA: Final = (
    "dialect-revision-rendered-document-machine-evidence-v1"
)
VISUAL_QA_RECEIPT_SCHEMA: Final = "dialect-revision-rendered-document-visual-qa-v1"
RENDERED_DOCUMENT_EVIDENCE_SCHEMA: Final = (
    "dialect-revision-rendered-document-evidence-v1"
)
RENDERED_DOCUMENT_EVIDENCE_CONTRACT: Final = "four-pdf-all-page-draft-wrapper-audit-v1"
PROMOTION_POLICY: Final = (
    "nonpromotable-until-native-derivation-and-machine-producer-closures"
)
PUBLIC_RELEASE_RELATIONSHIP: Final = (
    "separate-from-intentional-zero-document-public-release-boundary"
)

TRUST_MODEL: Final = {
    "document_reconciliation": (
        "independently SHA-256 anchored and natively validated before any rendered "
        "PDF or PNG byte is opened"
    ),
    "release_evidence": (
        "independently SHA-256 anchored and natively validated against the same "
        "artifact registry before any rendered PDF or PNG byte is opened"
    ),
    "source_snapshots": (
        "independently anchored baseline and revised source snapshots are bound by "
        "digest; their source bytes are not opened here"
    ),
    "derivation_evidence": (
        "independently anchored wrapper-level candidate/build receipts bind exact "
        "PDF digests; their underlying tools and receipt payloads are not opened or "
        "natively re-derived here"
    ),
    "rendered_bytes": (
        "exact PDF and one-PNG-per-page inventories are descriptor pinned; PDFs "
        "receive bounded signature/EOF checks and PNGs receive bounded structural, "
        "CRC, zlib, scanline-geometry, and filter-byte validation"
    ),
    "machine_evidence": (
        "independently anchored pdfinfo/pdffonts/pdfimages-style claims are "
        "reconciled to exact "
        "PDF and PNG digests; this builder does not invoke or independently validate "
        "the upstream PDF tools or re-derive PDF semantics"
    ),
    "human_review": (
        "a final-ready successor requires a separate independently anchored human "
        "receipt covering every exact PDF/page/PNG digest; draft audits may record "
        "it as absent, and self-declared reviewer fields do not authenticate a person"
    ),
}

NON_INFERENCE_LIMITS: Final = {
    "scientific_correctness": "not inferred",
    "general_text_legibility": "not inferred",
    "naas_compliance": "not inferred",
    "submission_approval": "not inferred",
    "human_identity": "not authenticated",
    "coauthor_approval": "not inferred",
    "portal_upload_or_readback": "not inferred",
}

PDF_ORDER: Final = (
    ("clean", "manuscript-clean.pdf"),
    ("marked", "manuscript-marked.pdf"),
    ("s1", "s1-appendix.pdf"),
    ("rebuttal", "response-to-reviewers.pdf"),
)
PDF_IDS: Final = tuple(record[0] for record in PDF_ORDER)
PDF_MEMBERS: Final = tuple(record[1] for record in PDF_ORDER)
PDF_MEMBER_BY_ID: Final = dict(PDF_ORDER)
PDF_ROLE_BY_ID: Final = {
    "clean": "clean revised manuscript PDF",
    "marked": "marked manuscript PDF",
    "s1": "S1 Appendix PDF",
    "rebuttal": "response to reviewers PDF",
}
SOURCE_DOCUMENT_BY_PDF_ID: Final = {
    "clean": "main",
    "marked": None,
    "s1": "s1",
    "rebuttal": "rebuttal",
}

RENDER_SETTINGS: Final = {
    "engine": "pdftoppm",
    "invocation_granularity": "one-page-per-process",
    "cwd": "/",
    "shell": False,
    "inherit_environment": False,
    "environment": {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
    "path_policy": "absolute-pinned-input-and-output-under-distinct-roots",
    "argv_template": [
        "{pdftoppm_absolute_path}",
        "-png",
        "-r",
        "150",
        "-cropbox",
        "-f",
        "{page}",
        "-l",
        "{page}",
        "-singlefile",
        "{pdf_absolute_path}",
        "{output_absolute_prefix}",
    ],
    "dpi": 150,
    "format": "png",
    "first_page": 1,
    "output_prefix_template": "pages/{pdf_id}/page-{page:04d}",
    "output_member_template": "pages/{pdf_id}/page-{page:04d}.png",
}

MACHINE_TOOL_CONTRACT: Final = {
    "pdf_metadata_tool": "pdfinfo",
    "font_inventory_tool": "pdffonts",
    "image_inventory_tool": "pdfimages",
    "page_size_unit": "millipoint",
    "required_checks": [
        "not-encrypted",
        "page-count",
        "page-size",
        "zero-rotation",
        "font-inventory-present",
        "all-fonts-embedded",
        "no-type-3-fonts",
    ],
}

VISUAL_REVIEW_CRITERIA: Final = [
    "page-render-present",
    "no-obvious-render-failure",
    "no-clipping",
    "no-overlap",
]

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_PDF_VERSION_RE: Final = re.compile(r"(?:1\.[4-9]|2\.0)")
_UTC_RE: Final = re.compile(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
)
_READ_CHUNK_BYTES: Final = 1024 * 1024
_MAX_METADATA_BYTES: Final = 2 * 1024 * 1024
_MAX_VISUAL_RECEIPT_BYTES: Final = 256 * 1024
_MAX_PDF_BYTES: Final = 128 * 1024 * 1024
_MAX_TOTAL_PDF_BYTES: Final = 512 * 1024 * 1024
_MAX_PNG_BYTES: Final = 32 * 1024 * 1024
_MAX_TOTAL_PNG_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_PAGES_PER_PDF: Final = 256
_MAX_TOTAL_PAGES: Final = 512
_MAX_PAGE_EDGE_PIXELS: Final = 4096
_MAX_PAGE_PIXELS: Final = 8_000_000
_MAX_PNG_MEMBERS: Final = _MAX_TOTAL_PAGES
_MAX_PNG_ROOT_ENTRIES: Final = _MAX_PNG_MEMBERS + 1 + len(PDF_ORDER)
# Poppler/libpng records integer pixels per metre by truncating dpi / 0.0254.
_EXPECTED_PNG_PIXELS_PER_METER: Final = int(RENDER_SETTINGS["dpi"]) * 10_000 // 254
_LATE_METADATA_DESCRIPTOR_COUNT: Final = 5
_NATIVE_ARTIFACT_REGISTRY_PEAK_DESCRIPTORS: Final = 16
_NATIVE_RECONCILIATION_PEAK_DESCRIPTORS: Final = 27
_NATIVE_RELEASE_FIXED_PEAK_DESCRIPTORS: Final = 21
_PUBLICATION_DESCRIPTOR_COUNT: Final = 2
_FD_SAFETY_HEADROOM: Final = 8
_PDF_SIGNATURE: Final = b"%PDF-"
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_BUILDER_MEMBER: Final = "analysis/build_tcga_revision_rendered_document_evidence.py"


class RenderedDocumentEvidenceError(ValueError):
    """Raised when the rendered-document wrapper audit is invalid."""


@dataclass(frozen=True, slots=True)
class RenderedDocumentEvidenceInputs:
    """Name every input path and its independently supplied digest anchor."""

    plan_path: Path
    reconciliation_path: Path
    artifact_registry_path: Path
    renderer_root: Path
    rendered_output_root: Path
    release_evidence_path: Path
    gate_receipt_root: Path
    source_data_root: Path
    document_anchor_path: Path
    document_root: Path
    source_snapshot_anchor_path: Path
    derivation_evidence_path: Path
    machine_evidence_path: Path
    visual_qa_receipt_path: Path | None
    pdf_root: Path
    png_root: Path
    expected_plan_sha256: str
    expected_reconciliation_sha256: str
    expected_artifact_registry_sha256: str
    expected_release_evidence_sha256: str
    expected_document_anchor_sha256: str
    expected_source_snapshot_anchor_sha256: str
    expected_derivation_evidence_sha256: str
    expected_machine_evidence_sha256: str
    expected_visual_qa_receipt_sha256: str | None


@dataclass(frozen=True, slots=True)
class RenderedDocumentEvidenceReceipt:
    """Summarize a published or independently validated evidence manifest."""

    manifest_path: str
    manifest_sha256: str
    mode: str
    pdf_count: int
    page_count: int
    machine_attested_pass_count: int
    visual_pass_page_count: int
    promotable: bool


@dataclass(slots=True)
class _PinnedFile:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    raw: bytes | None

    def close(self) -> None:
        """Close the descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _PinnedRoot:
    path: Path
    descriptor: int
    device: int
    inode: int
    mtime_ns: int

    def close(self) -> None:
        """Close the descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _PreparedEvidence:
    config: RenderedDocumentEvidenceInputs
    manifest: dict[str, object]
    metadata: list[tuple[_PinnedFile, str]]
    roots: list[tuple[_PinnedRoot, str]]
    rendered: list[tuple[_PinnedFile, str]]
    expected_pdf_members: tuple[str, ...]
    expected_png_members: tuple[str, ...]

    def revalidate(self) -> None:
        """Revalidate the full native and byte-level chain."""
        _validate_native_reconciliation(self.config)
        _validate_native_release_evidence(self.config)
        for pinned, context in self.metadata:
            _revalidate_file(pinned, context=context)
        for root, context in self.roots:
            _revalidate_root(root, context=context)
        _validate_exact_root_inventory(
            self.roots[0][0],
            expected=self.expected_pdf_members,
            context="PDF root",
            maximum_each=_MAX_PDF_BYTES,
            maximum_total=_MAX_TOTAL_PDF_BYTES,
        )
        _validate_exact_root_inventory(
            self.roots[1][0],
            expected=self.expected_png_members,
            context="PNG root",
            maximum_each=_MAX_PNG_BYTES,
            maximum_total=_MAX_TOTAL_PNG_BYTES,
        )
        for pinned, context in self.rendered:
            _revalidate_file(pinned, context=context)

    def close(self) -> None:
        """Close all owned descriptors."""
        for pinned, _ in self.rendered:
            pinned.close()
        for root, _ in self.roots:
            root.close()
        for pinned, _ in self.metadata:
            pinned.close()


def _fail(message: str) -> NoReturn:
    raise RenderedDocumentEvidenceError(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        _fail(f"value is not canonical-JSON encodable: {error}")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object")
    return value


def _expect_sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        _fail(f"{context} must be an array")
    return value


def _expect_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{context} has invalid keys; expected {sorted(expected)}, "
            f"found {sorted(actual)}",
        )


def _expect_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{context} must be a nonempty string")
    if not value.isascii() or any(ord(character) < 32 for character in value):
        _fail(f"{context} must be printable ASCII")
    return value


def _expect_token(value: object, *, context: str) -> str:
    token = _expect_string(value, context=context)
    if _TOKEN_RE.fullmatch(token) is None:
        _fail(f"{context} is not a canonical token")
    return token


def _expect_sha256(value: object, *, context: str) -> str:
    digest = _expect_string(value, context=context)
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _expect_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{context} must be a boolean")
    return value


def _expect_nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{context} must be a nonnegative integer")
    return value


def _expect_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer")
    return value


def _expect_positive_int(value: object, *, context: str) -> int:
    result = _expect_nonnegative_int(value, context=context)
    if result == 0:
        _fail(f"{context} must be positive")
    return result


def _expect_mode(value: object, *, context: str) -> str:
    mode = _expect_string(value, context=context)
    if mode not in {"draft", "final"}:
        _fail(f"{context} must be 'draft' or 'final'")
    return mode


def _expect_relative_member(value: object, *, context: str) -> str:
    member = _expect_string(value, context=context)
    if "\\" in member or any(ord(character) == 127 for character in member):
        _fail(f"{context} is not a canonical POSIX member")
    pure = PurePosixPath(member)
    if (
        pure.is_absolute()
        or pure.as_posix() != member
        or len(pure.parts) > 3
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{context} must be one safe bounded POSIX member")
    return member


def _read_descriptor(descriptor: int, *, maximum: int, context: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _hash_descriptor(descriptor: int, *, maximum: int, context: str) -> str:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        digest.update(chunk)
    return digest.hexdigest()


def _pin_file(
    path: Path,
    *,
    maximum: int,
    context: str,
    capture: bool = True,
) -> _PinnedFile:
    absolute = path.absolute()
    try:
        path_entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if (
        stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISREG(path_entry.st_mode)
        or resolved != absolute
    ):
        _fail(f"{context} must be a canonical non-symlink regular file")
    if path_entry.st_nlink != 1:
        _fail(f"{context} must have exactly one hard link")
    if path_entry.st_size > maximum:
        _fail(f"{context} exceeds the {maximum}-byte limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode):
            _fail(f"{context} must be a regular file")
        if entry.st_nlink != 1:
            _fail(f"{context} must have exactly one hard link")
        if entry.st_size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        identity = (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns)
        path_identity = (
            path_entry.st_dev,
            path_entry.st_ino,
            path_entry.st_size,
            path_entry.st_mtime_ns,
        )
        if identity != path_identity:
            _fail(f"{context} changed while it was pinned")
        raw = (
            _read_descriptor(descriptor, maximum=maximum, context=context)
            if capture
            else None
        )
        digest = (
            _sha256(raw)
            if raw is not None
            else _hash_descriptor(descriptor, maximum=maximum, context=context)
        )
        if os.lseek(descriptor, 0, os.SEEK_END) != entry.st_size:
            _fail(f"{context} changed while it was read")
        return _PinnedFile(
            path=absolute,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            sha256=digest,
            raw=raw,
        )
    except Exception:
        os.close(descriptor)
        raise


def _pin_root(path: Path, *, context: str) -> _PinnedRoot:
    absolute = path.absolute()
    try:
        path_entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect {context}: {error}")
    if (
        stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISDIR(path_entry.st_mode)
        or resolved != absolute
    ):
        _fail(f"{context} must be a canonical non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISDIR(entry.st_mode):
            _fail(f"{context} must be a directory")
        if (entry.st_dev, entry.st_ino) != (path_entry.st_dev, path_entry.st_ino):
            _fail(f"{context} changed while it was pinned")
        return _PinnedRoot(
            path=absolute,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            mtime_ns=entry.st_mtime_ns,
        )
    except Exception:
        os.close(descriptor)
        raise


def _open_root_member(
    root: _PinnedRoot,
    member: str,
    *,
    maximum: int,
    expected_size: int,
    context: str,
) -> _PinnedFile:
    canonical = _expect_relative_member(member, context=f"{context} member")
    parts = PurePosixPath(canonical).parts
    directory_descriptor = os.dup(root.descriptor)
    try:
        for part in parts[:-1]:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(parts[-1], flags, dir_fd=directory_descriptor)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    finally:
        os.close(directory_descriptor)
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode):
            _fail(f"{context} must be a regular file")
        if entry.st_nlink != 1:
            _fail(f"{context} must have exactly one hard link")
        if entry.st_size > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        if entry.st_size != expected_size:
            _fail(f"{context} size changed after inventory preflight")
        digest = _hash_descriptor(descriptor, maximum=maximum, context=context)
        if os.lseek(descriptor, 0, os.SEEK_END) != entry.st_size:
            _fail(f"{context} changed while it was read")
        return _PinnedFile(
            path=root.path / canonical,
            descriptor=descriptor,
            device=entry.st_dev,
            inode=entry.st_ino,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            sha256=digest,
            raw=None,
        )
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_file(pinned: _PinnedFile, *, context: str) -> None:
    entry = os.fstat(pinned.descriptor)
    try:
        path_entry = os.lstat(pinned.path)
        resolved = pinned.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} path disappeared after validation: {error}")
    identity = (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns)
    expected = (pinned.device, pinned.inode, pinned.size, pinned.mtime_ns)
    if (
        identity != expected
        or not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISREG(path_entry.st_mode)
        or resolved != pinned.path
        or (path_entry.st_dev, path_entry.st_ino) != (pinned.device, pinned.inode)
    ):
        _fail(f"{context} identity changed after validation")
    if (
        _hash_descriptor(
            pinned.descriptor,
            maximum=max(pinned.size, 1),
            context=context,
        )
        != pinned.sha256
    ):
        _fail(f"{context} bytes changed after validation")


def _revalidate_root(root: _PinnedRoot, *, context: str) -> None:
    entry = os.fstat(root.descriptor)
    try:
        path_entry = os.lstat(root.path)
        resolved = root.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} path disappeared after validation: {error}")
    if (
        (entry.st_dev, entry.st_ino, entry.st_mtime_ns)
        != (root.device, root.inode, root.mtime_ns)
        or not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISDIR(path_entry.st_mode)
        or resolved != root.path
        or (path_entry.st_dev, path_entry.st_ino) != (root.device, root.inode)
    ):
        _fail(f"{context} identity changed after validation")


def _parse_canonical_json(pinned: _PinnedFile, *, context: str) -> Mapping[str, object]:
    if pinned.raw is None:
        _fail(f"{context} was not captured as bounded metadata")
    try:
        text = pinned.raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"{context} must be ASCII canonical JSON: {error}")

    def reject_constant(value: str) -> NoReturn:
        _fail(f"{context} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        _fail(f"{context} is not valid JSON: {error}")
    if pinned.raw != _canonical_json(value) + b"\n":
        _fail(f"{context} is not canonical JSON with one trailing newline")
    return _expect_mapping(value, context=context)


def _validate_exact_root_inventory(
    root: _PinnedRoot,
    *,
    expected: Sequence[str],
    context: str,
    maximum_each: int,
    maximum_total: int,
) -> dict[str, int]:
    expected_tuple = tuple(sorted(expected))
    if len(expected_tuple) != len(set(expected_tuple)):
        _fail(f"{context} expected inventory contains duplicates")
    expected_directories = {
        PurePosixPath(*PurePosixPath(member).parts[:index]).as_posix() + "/"
        for member in expected_tuple
        for index in range(1, len(PurePosixPath(member).parts))
    }
    expected_inventory = tuple(sorted((*expected_tuple, *expected_directories)))
    actual, sizes = _inventory_root(root, context=context)
    if actual != expected_inventory:
        _fail(
            f"{context} inventory mismatch; expected {list(expected_inventory)}, "
            f"found {list(actual)}",
        )
    total = 0
    for member in expected_tuple:
        size = sizes[member]
        if size > maximum_each:
            _fail(f"{context} member {member!r} exceeds the per-file byte limit")
        total += size
        if total > maximum_total:
            _fail(f"{context} exceeds the aggregate byte limit")
    return sizes


def _inventory_root(
    root: _PinnedRoot,
    *,
    context: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    members: list[str] = []
    sizes: dict[str, int] = {}

    def walk(descriptor: int, prefix: tuple[str, ...]) -> None:
        try:
            entries = os.scandir(descriptor)
        except OSError as error:
            _fail(f"cannot list {context}: {error}")
        with entries:
            for directory_entry in entries:
                name = directory_entry.name
                if name in {"", ".", ".."} or "/" in name or "\\" in name:
                    _fail(f"{context} contains a noncanonical member name")
                try:
                    entry = directory_entry.stat(follow_symlinks=False)
                except OSError as error:
                    _fail(f"cannot inspect {context} member {name!r}: {error}")
                path_parts = (*prefix, name)
                member = PurePosixPath(*path_parts).as_posix()
                if stat.S_ISREG(entry.st_mode):
                    if entry.st_nlink != 1:
                        _fail(f"{context} member {member!r} must have one hard link")
                    members.append(member)
                    sizes[member] = entry.st_size
                    if len(members) > _MAX_PNG_ROOT_ENTRIES:
                        _fail(f"{context} contains too many members")
                    continue
                if not stat.S_ISDIR(entry.st_mode) or len(path_parts) >= 3:
                    _fail(f"{context} member {member!r} has an invalid type/depth")
                members.append(f"{member}/")
                if len(members) > _MAX_PNG_ROOT_ENTRIES:
                    _fail(f"{context} contains too many members")
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                try:
                    child = os.open(name, flags, dir_fd=descriptor)
                except OSError as error:
                    _fail(f"cannot open {context} directory {member!r}: {error}")
                try:
                    walk(child, path_parts)
                finally:
                    os.close(child)

    walk(root.descriptor, ())
    return tuple(sorted(members)), sizes


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_path_topology(
    config: RenderedDocumentEvidenceInputs,
    *,
    destination: Path | None,
) -> None:
    if (config.visual_qa_receipt_path is None) != (
        config.expected_visual_qa_receipt_sha256 is None
    ):
        _fail("visual-QA receipt path and digest must be supplied together")
    roots = tuple(
        path.absolute()
        for path in (
            config.renderer_root,
            config.rendered_output_root,
            config.gate_receipt_root,
            config.source_data_root,
            config.document_root,
            config.pdf_root,
            config.png_root,
        )
    )
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if _paths_overlap(first, second):
                _fail("all input roots must be pairwise distinct and non-nested")
    input_files = tuple(
        path
        for path in (
            config.plan_path,
            config.reconciliation_path,
            config.artifact_registry_path,
            config.release_evidence_path,
            config.document_anchor_path,
            config.source_snapshot_anchor_path,
            config.derivation_evidence_path,
            config.machine_evidence_path,
            config.visual_qa_receipt_path,
        )
        if path is not None
    )
    absolute_files = tuple(path.absolute() for path in input_files)
    if len(set(absolute_files)) != len(absolute_files):
        _fail("metadata inputs must be distinct files")
    for path in absolute_files:
        if any(path == root or root in path.parents for root in roots):
            _fail("metadata inputs must be outside every input root")
    if destination is not None:
        absolute_destination = destination.absolute()
        if absolute_destination in absolute_files:
            _fail("destination must differ from every metadata input")
        if any(
            absolute_destination == root or root in absolute_destination.parents
            for root in roots
        ):
            _fail("destination must be outside every input root")


def _validate_independent_digest(
    pinned: _PinnedFile,
    expected: str,
    *,
    context: str,
) -> None:
    if pinned.sha256 != _expect_sha256(expected, context=f"expected {context} SHA-256"):
        _fail(f"{context} SHA-256 does not match its independent anchor")


def _open_descriptor_count() -> int:
    device_fds = Path("/dev/fd")
    candidate = device_fds if device_fds.exists() else Path("/proc/self/fd")
    try:
        return len(tuple(candidate.iterdir()))
    except OSError as error:
        _fail(f"cannot establish the current open-descriptor count: {error}")


def _validate_fd_headroom(
    page_count: int,
    *,
    gate_receipt_count: int,
    source_member_count: int,
) -> None:
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        return
    current = _open_descriptor_count()
    native_peak = max(
        _NATIVE_RECONCILIATION_PEAK_DESCRIPTORS,
        _NATIVE_RELEASE_FIXED_PEAK_DESCRIPTORS
        + gate_receipt_count
        + source_member_count,
    )
    required = (
        current
        + _LATE_METADATA_DESCRIPTOR_COUNT
        + page_count
        + len(PDF_ORDER)
        + 2  # pinned PDF and PNG roots
        + native_peak
        + _PUBLICATION_DESCRIPTOR_COUNT
        + _FD_SAFETY_HEADROOM
    )
    if required > soft_limit:
        _fail(
            "insufficient RLIMIT_NOFILE headroom for descriptor-pinned rendered "
            f"evidence: need at least {required}, soft limit is {soft_limit}",
        )


def _normalize_plan(value: Mapping[str, object]) -> dict[str, object]:
    context = "rendered-document input"
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "release_id",
            "bindings",
            "render_settings",
            "documents",
            "non_inference_limits",
        },
        context=context,
    )
    if value["schema"] != RENDERED_DOCUMENT_INPUT_SCHEMA:
        _fail(f"{context} has the wrong schema")
    mode = _expect_mode(value["mode"], context=f"{context}.mode")
    release_id = _expect_token(value["release_id"], context=f"{context}.release_id")
    if value["render_settings"] != RENDER_SETTINGS:
        _fail(f"{context}.render_settings is not the fixed render contract")
    if value["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail(f"{context}.non_inference_limits drifted")
    bindings = _expect_mapping(value["bindings"], context=f"{context}.bindings")
    binding_keys = {
        "document_reconciliation_sha256",
        "artifact_registry_sha256",
        "release_evidence_sha256",
        "document_anchor_sha256",
        "source_snapshot_anchor_sha256",
        "derivation_evidence_sha256",
        "machine_evidence_sha256",
        "visual_qa_receipt_sha256",
    }
    _expect_keys(bindings, binding_keys, context=f"{context}.bindings")
    normalized_bindings: dict[str, str | None] = {}
    for key in sorted(binding_keys):
        if key == "visual_qa_receipt_sha256" and bindings[key] is None:
            if mode == "final":
                _fail(f"final {context} requires a visual-QA receipt binding")
            normalized_bindings[key] = None
        else:
            normalized_bindings[key] = _expect_sha256(
                bindings[key],
                context=f"{context}.bindings.{key}",
            )
    documents = _expect_sequence(value["documents"], context=f"{context}.documents")
    if len(documents) != len(PDF_ORDER):
        _fail(f"{context}.documents must contain exactly four PDFs")
    normalized_documents: list[dict[str, object]] = []
    total_pages = 0
    for index, (expected_id, expected_member) in enumerate(PDF_ORDER):
        document_context = f"{context}.documents[{index}]"
        document = _expect_mapping(documents[index], context=document_context)
        _expect_keys(
            document,
            {
                "pdf_id",
                "pdf_member",
                "pdf_role",
                "source_document_id",
                "output_prefix_template",
                "page_count",
                "page_size_millipoints",
            },
            context=document_context,
        )
        if document["pdf_id"] != expected_id:
            _fail(f"{document_context}.pdf_id is not in the exact PDF order")
        if document["pdf_member"] != expected_member:
            _fail(f"{document_context}.pdf_member is not the exact PDF member")
        if document["pdf_role"] != PDF_ROLE_BY_ID[expected_id]:
            _fail(f"{document_context}.pdf_role is not the fixed PLOS role")
        if document["source_document_id"] != SOURCE_DOCUMENT_BY_PDF_ID[expected_id]:
            _fail(f"{document_context}.source_document_id has the wrong derivation")
        expected_output_prefix = f"pages/{expected_id}/page-{{page:04d}}"
        if document["output_prefix_template"] != expected_output_prefix:
            _fail(
                f"{document_context}.output_prefix_template is not the canonical "
                "single-page output prefix",
            )
        page_count = _expect_positive_int(
            document["page_count"],
            context=f"{document_context}.page_count",
        )
        if page_count > _MAX_PAGES_PER_PDF:
            _fail(f"{document_context}.page_count exceeds {_MAX_PAGES_PER_PDF}")
        page_size = _expect_mapping(
            document["page_size_millipoints"],
            context=f"{document_context}.page_size_millipoints",
        )
        _expect_keys(
            page_size,
            {"width", "height"},
            context=f"{document_context}.page_size_millipoints",
        )
        width = _expect_positive_int(
            page_size["width"],
            context=f"{document_context}.page_size_millipoints.width",
        )
        height = _expect_positive_int(
            page_size["height"],
            context=f"{document_context}.page_size_millipoints.height",
        )
        if not 72_000 <= width <= 2_000_000 or not 72_000 <= height <= 2_000_000:
            _fail(f"{document_context}.page_size_millipoints is implausible")
        total_pages += page_count
        normalized_documents.append(
            {
                "pdf_id": expected_id,
                "pdf_member": expected_member,
                "pdf_role": PDF_ROLE_BY_ID[expected_id],
                "source_document_id": SOURCE_DOCUMENT_BY_PDF_ID[expected_id],
                "output_prefix_template": expected_output_prefix,
                "page_count": page_count,
                "page_size_millipoints": {"width": width, "height": height},
            },
        )
    if total_pages > _MAX_TOTAL_PAGES:
        _fail(f"{context}.documents exceeds the total page limit")
    return {
        "schema": RENDERED_DOCUMENT_INPUT_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "bindings": normalized_bindings,
        "render_settings": RENDER_SETTINGS,
        "documents": normalized_documents,
        "non_inference_limits": NON_INFERENCE_LIMITS,
    }


def _normalize_source_snapshot_anchor(
    value: Mapping[str, object],
    *,
    release_id: str,
    document_anchor_sha256: str,
) -> dict[str, object]:
    context = "source-snapshot anchor"
    _expect_keys(
        value,
        {
            "schema",
            "release_id",
            "document_anchor_sha256",
            "snapshots",
            "clean_marked_bindings",
            "non_inference_limits",
        },
        context=context,
    )
    if value["schema"] != SOURCE_SNAPSHOT_ANCHOR_SCHEMA:
        _fail(f"{context} has the wrong schema")
    if value["release_id"] != release_id:
        _fail(f"{context}.release_id does not match the plan")
    if value["document_anchor_sha256"] != document_anchor_sha256:
        _fail(f"{context} does not bind the independently anchored document anchor")
    if value["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail(f"{context}.non_inference_limits drifted")
    snapshots = _expect_sequence(value["snapshots"], context=f"{context}.snapshots")
    if len(snapshots) != 2:
        _fail(f"{context}.snapshots must contain baseline and revised")
    normalized_snapshots: list[dict[str, object]] = []
    digests: dict[str, str] = {}
    for index, kind in enumerate(("baseline", "revised")):
        item_context = f"{context}.snapshots[{index}]"
        item = _expect_mapping(snapshots[index], context=item_context)
        _expect_keys(
            item,
            {"kind", "snapshot_id", "sha256", "bytes"},
            context=item_context,
        )
        if item["kind"] != kind:
            _fail(f"{item_context}.kind is not in baseline/revised order")
        digest = _expect_sha256(item["sha256"], context=f"{item_context}.sha256")
        digests[kind] = digest
        normalized_snapshots.append(
            {
                "kind": kind,
                "snapshot_id": _expect_token(
                    item["snapshot_id"],
                    context=f"{item_context}.snapshot_id",
                ),
                "sha256": digest,
                "bytes": _expect_positive_int(
                    item["bytes"],
                    context=f"{item_context}.bytes",
                ),
            },
        )
    if digests["baseline"] == digests["revised"]:
        _fail(f"{context} baseline and revised snapshots must be distinct")
    bindings = _expect_sequence(
        value["clean_marked_bindings"],
        context=f"{context}.clean_marked_bindings",
    )
    expected_ids = ("clean", "marked")
    if len(bindings) != len(expected_ids):
        _fail(f"{context}.clean_marked_bindings must bind exactly clean and marked")
    normalized_bindings: list[dict[str, str]] = []
    for index, expected_id in enumerate(expected_ids):
        item_context = f"{context}.clean_marked_bindings[{index}]"
        item = _expect_mapping(bindings[index], context=item_context)
        _expect_keys(
            item,
            {
                "pdf_id",
                "baseline_snapshot_sha256",
                "revised_snapshot_sha256",
            },
            context=item_context,
        )
        if item["pdf_id"] != expected_id:
            _fail(f"{item_context}.pdf_id is not in clean/marked order")
        if item["baseline_snapshot_sha256"] != digests["baseline"]:
            _fail(f"{item_context} does not bind the baseline snapshot")
        if item["revised_snapshot_sha256"] != digests["revised"]:
            _fail(f"{item_context} does not bind the revised snapshot")
        normalized_bindings.append(
            {
                "pdf_id": expected_id,
                "baseline_snapshot_sha256": digests["baseline"],
                "revised_snapshot_sha256": digests["revised"],
            },
        )
    return {
        "schema": SOURCE_SNAPSHOT_ANCHOR_SCHEMA,
        "release_id": release_id,
        "document_anchor_sha256": document_anchor_sha256,
        "snapshots": normalized_snapshots,
        "clean_marked_bindings": normalized_bindings,
        "non_inference_limits": NON_INFERENCE_LIMITS,
    }


def _provenance_digest(
    record: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    return _expect_sha256(record[key], context=f"{context}.{key}")


def _provenance_bytes(
    record: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> int:
    return _expect_positive_int(record[key], context=f"{context}.{key}")


def _normalize_derivation_evidence(
    value: Mapping[str, object],
    *,
    mode: str,
    release_id: str,
    reconciliation_sha256: str,
    release_evidence_sha256: str,
    source_snapshot_anchor_sha256: str,
    source_snapshot: Mapping[str, object],
    pdf_pins: Mapping[str, _PinnedFile],
) -> dict[str, object]:
    context = "derivation evidence"
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "release_id",
            "bindings",
            "nonpromotable_pdf_sha256",
            "documents",
            "non_inference_limits",
        },
        context=context,
    )
    if value["schema"] != DERIVATION_EVIDENCE_SCHEMA:
        _fail(f"{context} has the wrong schema")
    if value["mode"] != mode or value["release_id"] != release_id:
        _fail(f"{context} mode/release_id does not match the plan")
    if value["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail(f"{context}.non_inference_limits drifted")
    bindings = _expect_mapping(value["bindings"], context=f"{context}.bindings")
    _expect_keys(
        bindings,
        {
            "document_reconciliation_sha256",
            "release_evidence_sha256",
            "source_snapshot_anchor_sha256",
        },
        context=f"{context}.bindings",
    )
    required_bindings = {
        "document_reconciliation_sha256": reconciliation_sha256,
        "release_evidence_sha256": release_evidence_sha256,
        "source_snapshot_anchor_sha256": source_snapshot_anchor_sha256,
    }
    if dict(bindings) != required_bindings:
        _fail(f"{context}.bindings do not close the upstream provenance chain")
    nonpromotable_values = _expect_sequence(
        value["nonpromotable_pdf_sha256"],
        context=f"{context}.nonpromotable_pdf_sha256",
    )
    nonpromotable = [
        _expect_sha256(
            digest,
            context=f"{context}.nonpromotable_pdf_sha256[{index}]",
        )
        for index, digest in enumerate(nonpromotable_values)
    ]
    if nonpromotable != sorted(set(nonpromotable)):
        _fail(f"{context}.nonpromotable_pdf_sha256 must be uniquely sorted")
    snapshot_values = _expect_sequence(
        source_snapshot["snapshots"],
        context="normalized source snapshots",
    )
    snapshots = {
        str(_expect_mapping(item, context="source snapshot")["kind"]): str(
            _expect_mapping(item, context="source snapshot")["sha256"],
        )
        for item in snapshot_values
    }
    documents = _expect_sequence(value["documents"], context=f"{context}.documents")
    if len(documents) != len(PDF_ORDER):
        _fail(f"{context}.documents must cover exactly four PDFs")
    normalized: list[dict[str, object]] = []
    shared_candidate: dict[str, object] | None = None
    for index, (pdf_id, _) in enumerate(PDF_ORDER):
        item_context = f"{context}.documents[{index}]"
        item = _expect_mapping(documents[index], context=item_context)
        _expect_keys(
            item,
            {"pdf_id", "source_document_id", "status", "evidence"},
            context=item_context,
        )
        if item["pdf_id"] != pdf_id:
            _fail(f"{item_context}.pdf_id is not in exact PDF order")
        if item["source_document_id"] != SOURCE_DOCUMENT_BY_PDF_ID[pdf_id]:
            _fail(f"{item_context}.source_document_id has the wrong derivation")
        status = _expect_string(item["status"], context=f"{item_context}.status")
        if status not in {"attested", "unproven"}:
            _fail(f"{item_context}.status must be attested or unproven")
        if status == "unproven":
            if item["evidence"] is not None:
                _fail(f"{item_context}.evidence must be null while unproven")
            if mode == "final":
                _fail(f"final {item_context} cannot remain unproven")
            normalized.append(
                {
                    "pdf_id": pdf_id,
                    "source_document_id": SOURCE_DOCUMENT_BY_PDF_ID[pdf_id],
                    "status": "unproven",
                    "evidence": None,
                },
            )
            continue
        evidence = _expect_mapping(item["evidence"], context=f"{item_context}.evidence")
        if pdf_id in {"clean", "marked"}:
            expected_keys = {
                "candidate_manifest_sha256",
                "candidate_manifest_bytes",
                "baseline_snapshot_sha256",
                "revised_snapshot_sha256",
                "accepted_roundtrip_sha256",
                "accepted_roundtrip_bytes",
                "declined_roundtrip_sha256",
                "declined_roundtrip_bytes",
                "source_to_pdf_receipt_sha256",
                "source_to_pdf_receipt_bytes",
                "rebuild_a_sha256",
                "rebuild_b_sha256",
            }
            if pdf_id == "marked":
                expected_keys |= {
                    "latexdiff_receipt_sha256",
                    "latexdiff_receipt_bytes",
                }
            _expect_keys(evidence, expected_keys, context=f"{item_context}.evidence")
            candidate_shared = {
                "candidate_manifest_sha256": _provenance_digest(
                    evidence,
                    "candidate_manifest_sha256",
                    context=f"{item_context}.evidence",
                ),
                "candidate_manifest_bytes": _provenance_bytes(
                    evidence,
                    "candidate_manifest_bytes",
                    context=f"{item_context}.evidence",
                ),
                "baseline_snapshot_sha256": _provenance_digest(
                    evidence,
                    "baseline_snapshot_sha256",
                    context=f"{item_context}.evidence",
                ),
                "revised_snapshot_sha256": _provenance_digest(
                    evidence,
                    "revised_snapshot_sha256",
                    context=f"{item_context}.evidence",
                ),
                "accepted_roundtrip_sha256": _provenance_digest(
                    evidence,
                    "accepted_roundtrip_sha256",
                    context=f"{item_context}.evidence",
                ),
                "accepted_roundtrip_bytes": _provenance_bytes(
                    evidence,
                    "accepted_roundtrip_bytes",
                    context=f"{item_context}.evidence",
                ),
                "declined_roundtrip_sha256": _provenance_digest(
                    evidence,
                    "declined_roundtrip_sha256",
                    context=f"{item_context}.evidence",
                ),
                "declined_roundtrip_bytes": _provenance_bytes(
                    evidence,
                    "declined_roundtrip_bytes",
                    context=f"{item_context}.evidence",
                ),
            }
            if (
                candidate_shared["baseline_snapshot_sha256"] != snapshots["baseline"]
                or candidate_shared["revised_snapshot_sha256"] != snapshots["revised"]
            ):
                _fail(f"{item_context} does not bind baseline/revised snapshots")
            if shared_candidate is None:
                shared_candidate = candidate_shared
            elif shared_candidate != candidate_shared:
                _fail("clean and marked do not bind identical candidate provenance")
            normalized_evidence: dict[str, object] = {
                **candidate_shared,
                "source_to_pdf_receipt_sha256": _provenance_digest(
                    evidence,
                    "source_to_pdf_receipt_sha256",
                    context=f"{item_context}.evidence",
                ),
                "source_to_pdf_receipt_bytes": _provenance_bytes(
                    evidence,
                    "source_to_pdf_receipt_bytes",
                    context=f"{item_context}.evidence",
                ),
                "rebuild_a_sha256": _provenance_digest(
                    evidence,
                    "rebuild_a_sha256",
                    context=f"{item_context}.evidence",
                ),
                "rebuild_b_sha256": _provenance_digest(
                    evidence,
                    "rebuild_b_sha256",
                    context=f"{item_context}.evidence",
                ),
            }
            if pdf_id == "marked":
                normalized_evidence.update(
                    {
                        "latexdiff_receipt_sha256": _provenance_digest(
                            evidence,
                            "latexdiff_receipt_sha256",
                            context=f"{item_context}.evidence",
                        ),
                        "latexdiff_receipt_bytes": _provenance_bytes(
                            evidence,
                            "latexdiff_receipt_bytes",
                            context=f"{item_context}.evidence",
                        ),
                    },
                )
        else:
            _expect_keys(
                evidence,
                {
                    "approved_manifest_sha256",
                    "approved_manifest_bytes",
                    "source_to_pdf_receipt_sha256",
                    "source_to_pdf_receipt_bytes",
                    "external_pdf_qa_receipt_sha256",
                    "external_pdf_qa_receipt_bytes",
                    "rebuild_a_sha256",
                    "rebuild_b_sha256",
                },
                context=f"{item_context}.evidence",
            )
            normalized_evidence = {
                key: (
                    _provenance_bytes(evidence, key, context=f"{item_context}.evidence")
                    if key.endswith("_bytes")
                    else _provenance_digest(
                        evidence,
                        key,
                        context=f"{item_context}.evidence",
                    )
                )
                for key in sorted(evidence)
            }
        pdf_digest = pdf_pins[pdf_id].sha256
        if mode == "final" and pdf_digest in nonpromotable:
            _fail(f"{item_context} matches a known nonpromotable working PDF")
        if (
            normalized_evidence["rebuild_a_sha256"] != pdf_digest
            or normalized_evidence["rebuild_b_sha256"] != pdf_digest
        ):
            _fail(f"{item_context} lacks two byte-identical PDF rebuilds")
        normalized.append(
            {
                "pdf_id": pdf_id,
                "source_document_id": SOURCE_DOCUMENT_BY_PDF_ID[pdf_id],
                "status": "attested",
                "reference_validation": "unverified-external-references",
                "evidence": normalized_evidence,
            },
        )
    return {
        "schema": DERIVATION_EVIDENCE_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "bindings": required_bindings,
        "nonpromotable_pdf_sha256": nonpromotable,
        "documents": normalized,
        "non_inference_limits": NON_INFERENCE_LIMITS,
    }


def _expected_png_member(pdf_id: str, page: int) -> str:
    return f"pages/{pdf_id}/page-{page:04d}.png"


def _expected_pixel_dimension(millipoints: int) -> int:
    numerator = millipoints * int(RENDER_SETTINGS["dpi"])
    return (numerator + 71_999) // 72_000


def _normalize_font_record(value: object, *, context: str) -> dict[str, object]:
    record = _expect_mapping(value, context=context)
    _expect_keys(
        record,
        {"name", "type", "encoding", "embedded", "subset", "unicode"},
        context=context,
    )
    return {
        "name": _expect_string(record["name"], context=f"{context}.name"),
        "type": _expect_string(record["type"], context=f"{context}.type"),
        "encoding": _expect_string(
            record["encoding"],
            context=f"{context}.encoding",
        ),
        "embedded": _expect_bool(record["embedded"], context=f"{context}.embedded"),
        "subset": _expect_bool(record["subset"], context=f"{context}.subset"),
        "unicode": _expect_bool(record["unicode"], context=f"{context}.unicode"),
    }


def _normalize_page_box(value: object, *, context: str) -> list[int]:
    coordinates = _expect_sequence(value, context=context)
    if len(coordinates) != 4:
        _fail(f"{context} must contain x0, y0, x1, and y1")
    normalized = [
        _expect_int(coordinate, context=f"{context}[{index}]")
        for index, coordinate in enumerate(coordinates)
    ]
    if any(abs(coordinate) > 2_000_000 for coordinate in normalized):
        _fail(f"{context} coordinate is outside the bounded page geometry")
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        _fail(f"{context} has nonpositive geometry")
    return normalized


def _derive_machine_issues(
    *,
    pdf_id: str,
    encrypted: bool,
    raster_image_count: int,
    page_records: Sequence[Mapping[str, object]],
    expected_size: Mapping[str, object],
    fonts: Sequence[Mapping[str, object]],
) -> list[str]:
    issues: set[str] = set()
    if encrypted:
        issues.add("encrypted")
    if pdf_id in {"clean", "marked"} and raster_image_count:
        issues.add("raster-image-present")
    if pdf_id in {"clean", "marked"} and expected_size != {
        "width": 612_000,
        "height": 792_000,
    }:
        issues.add("not-us-letter")
    for page in page_records:
        if page["rotation_degrees"] != 0:
            issues.add("rotated-page")
        if (
            page["width_millipoints"] != expected_size["width"]
            or page["height_millipoints"] != expected_size["height"]
        ):
            issues.add("page-size-mismatch")
    if not fonts:
        issues.add("font-inventory-empty")
    if any(not bool(font["embedded"]) for font in fonts):
        issues.add("font-not-embedded")
    if any(font["type"] == "Type 3" for font in fonts):
        issues.add("type-3-font")
    return sorted(issues)


def _normalize_machine_producer(value: object) -> dict[str, object]:
    context = "machine evidence.producer"
    producer = _expect_mapping(value, context=context)
    _expect_keys(
        producer,
        {"producer_receipt_sha256", "producer_receipt_bytes", "tools"},
        context=context,
    )
    tools_value = _expect_sequence(producer["tools"], context=f"{context}.tools")
    expected_names = ("pdfinfo", "pdffonts", "pdfimages", "pdftoppm")
    if len(tools_value) != len(expected_names):
        _fail(
            f"{context}.tools must bind pdfinfo, pdffonts, pdfimages, and pdftoppm",
        )
    tools: list[dict[str, object]] = []
    for index, name in enumerate(expected_names):
        tool_context = f"{context}.tools[{index}]"
        tool = _expect_mapping(tools_value[index], context=tool_context)
        _expect_keys(
            tool,
            {"name", "absolute_path", "sha256", "bytes", "version"},
            context=tool_context,
        )
        if tool["name"] != name:
            _fail(f"{tool_context}.name is not in the fixed tool order")
        absolute_path = _expect_string(
            tool["absolute_path"],
            context=f"{tool_context}.absolute_path",
        )
        parsed_path = PurePosixPath(absolute_path)
        if (
            parsed_path.root != "/"
            or parsed_path.anchor != "/"
            or not parsed_path.name
            or parsed_path.as_posix() != absolute_path
            or any(part in {".", ".."} for part in parsed_path.parts)
            or "\\" in absolute_path
        ):
            _fail(f"{tool_context}.absolute_path is not canonical absolute POSIX")
        tools.append(
            {
                "name": name,
                "absolute_path": absolute_path,
                "sha256": _expect_sha256(
                    tool["sha256"],
                    context=f"{tool_context}.sha256",
                ),
                "bytes": _expect_positive_int(
                    tool["bytes"],
                    context=f"{tool_context}.bytes",
                ),
                "version": _expect_string(
                    tool["version"],
                    context=f"{tool_context}.version",
                ),
            },
        )
    return {
        "producer_receipt_sha256": _expect_sha256(
            producer["producer_receipt_sha256"],
            context=f"{context}.producer_receipt_sha256",
        ),
        "producer_receipt_bytes": _expect_positive_int(
            producer["producer_receipt_bytes"],
            context=f"{context}.producer_receipt_bytes",
        ),
        "tools": tools,
    }


def _normalize_machine_evidence(
    value: Mapping[str, object],
    *,
    mode: str,
    release_id: str,
    plan_documents: Sequence[Mapping[str, object]],
    pdf_pins: Mapping[str, _PinnedFile],
    png_pins: Mapping[str, _PinnedFile],
    png_dimensions: Mapping[str, tuple[int, int]],
    render_set_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    context = "machine evidence"
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "release_id",
            "producer",
            "tool_contract",
            "render_settings",
            "render_set_sha256",
            "documents",
            "non_inference_limits",
        },
        context=context,
    )
    if value["schema"] != MACHINE_EVIDENCE_SCHEMA:
        _fail(f"{context} has the wrong schema")
    if value["mode"] != mode or value["release_id"] != release_id:
        _fail(f"{context} mode/release_id does not match the plan")
    if value["tool_contract"] != MACHINE_TOOL_CONTRACT:
        _fail(f"{context}.tool_contract drifted")
    if value["render_settings"] != RENDER_SETTINGS:
        _fail(f"{context}.render_settings drifted")
    if value["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail(f"{context}.non_inference_limits drifted")
    producer = _normalize_machine_producer(value["producer"])
    if value["render_set_sha256"] != render_set_sha256:
        _fail(f"{context}.render_set_sha256 does not bind exact rendered bytes")
    records = _expect_sequence(value["documents"], context=f"{context}.documents")
    if len(records) != len(PDF_ORDER):
        _fail(f"{context}.documents must contain exactly four PDFs")
    normalized_documents: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for index, plan_document in enumerate(plan_documents):
        item_context = f"{context}.documents[{index}]"
        item = _expect_mapping(records[index], context=item_context)
        _expect_keys(
            item,
            {
                "pdf_id",
                "pdf_member",
                "pdf_sha256",
                "pdf_bytes",
                "pdf_version",
                "page_count",
                "encrypted",
                "raster_image_count",
                "pdfinfo_sha256",
                "pdfinfo_bytes",
                "pdffonts_sha256",
                "pdffonts_bytes",
                "pdfimages_sha256",
                "pdfimages_bytes",
                "pages",
                "fonts",
                "status",
                "issues",
            },
            context=item_context,
        )
        pdf_id = str(plan_document["pdf_id"])
        pdf_member = str(plan_document["pdf_member"])
        if item["pdf_id"] != pdf_id or item["pdf_member"] != pdf_member:
            _fail(f"{item_context} does not match the exact PDF plan")
        pdf_pin = pdf_pins[pdf_id]
        if item["pdf_sha256"] != pdf_pin.sha256 or item["pdf_bytes"] != pdf_pin.size:
            _fail(f"{item_context} does not match exact PDF bytes")
        pdf_version = _expect_string(
            item["pdf_version"],
            context=f"{item_context}.pdf_version",
        )
        if _PDF_VERSION_RE.fullmatch(pdf_version) is None:
            _fail(f"{item_context}.pdf_version is unsupported")
        page_count = int(plan_document["page_count"])
        if item["page_count"] != page_count:
            _fail(f"{item_context}.page_count does not match the plan")
        encrypted = _expect_bool(
            item["encrypted"],
            context=f"{item_context}.encrypted",
        )
        raster_image_count = _expect_nonnegative_int(
            item["raster_image_count"],
            context=f"{item_context}.raster_image_count",
        )
        page_values = _expect_sequence(item["pages"], context=f"{item_context}.pages")
        if len(page_values) != page_count:
            _fail(f"{item_context}.pages does not cover every page")
        expected_size = _expect_mapping(
            plan_document["page_size_millipoints"],
            context=f"plan document {pdf_id} page size",
        )
        normalized_pages: list[dict[str, object]] = []
        for page_index in range(1, page_count + 1):
            page_context = f"{item_context}.pages[{page_index - 1}]"
            page = _expect_mapping(page_values[page_index - 1], context=page_context)
            _expect_keys(
                page,
                {
                    "page",
                    "media_box_millipoints",
                    "crop_box_millipoints",
                    "width_millipoints",
                    "height_millipoints",
                    "rotation_degrees",
                    "png_member",
                    "png_sha256",
                    "png_bytes",
                    "rendered_width_pixels",
                    "rendered_height_pixels",
                    "render_a_sha256",
                    "render_a_bytes",
                    "render_b_sha256",
                    "render_b_bytes",
                },
                context=page_context,
            )
            if page["page"] != page_index:
                _fail(f"{page_context}.page is not sequential")
            media_box = _normalize_page_box(
                page["media_box_millipoints"],
                context=f"{page_context}.media_box_millipoints",
            )
            crop_box = _normalize_page_box(
                page["crop_box_millipoints"],
                context=f"{page_context}.crop_box_millipoints",
            )
            if (
                crop_box[0] < media_box[0]
                or crop_box[1] < media_box[1]
                or crop_box[2] > media_box[2]
                or crop_box[3] > media_box[3]
            ):
                _fail(f"{page_context}.crop_box_millipoints leaves the media box")
            png_member = _expected_png_member(pdf_id, page_index)
            if page["png_member"] != png_member:
                _fail(f"{page_context}.png_member is not the fixed render member")
            png_pin = png_pins[png_member]
            if (
                page["png_sha256"] != png_pin.sha256
                or page["png_bytes"] != png_pin.size
            ):
                _fail(f"{page_context} does not match exact PNG bytes")
            if (
                page["render_a_sha256"] != png_pin.sha256
                or page["render_b_sha256"] != png_pin.sha256
                or page["render_a_bytes"] != png_pin.size
                or page["render_b_bytes"] != png_pin.size
            ):
                _fail(f"{page_context} lacks two byte-identical render reproductions")
            width = _expect_positive_int(
                page["width_millipoints"],
                context=f"{page_context}.width_millipoints",
            )
            height = _expect_positive_int(
                page["height_millipoints"],
                context=f"{page_context}.height_millipoints",
            )
            if (
                width != crop_box[2] - crop_box[0]
                or height != crop_box[3] - crop_box[1]
            ):
                _fail(f"{page_context} dimensions do not match the crop box")
            rotation = _expect_nonnegative_int(
                page["rotation_degrees"],
                context=f"{page_context}.rotation_degrees",
            )
            if rotation not in {0, 90, 180, 270}:
                _fail(f"{page_context}.rotation_degrees is invalid")
            expected_width_pixels = _expected_pixel_dimension(width)
            expected_height_pixels = _expected_pixel_dimension(height)
            if rotation in {90, 270}:
                expected_width_pixels, expected_height_pixels = (
                    expected_height_pixels,
                    expected_width_pixels,
                )
            actual_width_pixels, actual_height_pixels = png_dimensions[png_member]
            if (
                actual_width_pixels != expected_width_pixels
                or actual_height_pixels != expected_height_pixels
            ):
                _fail(f"{page_context} PDF page size does not match PNG IHDR")
            if page["rendered_width_pixels"] != expected_width_pixels:
                _fail(f"{page_context}.rendered_width_pixels is inconsistent")
            if page["rendered_height_pixels"] != expected_height_pixels:
                _fail(f"{page_context}.rendered_height_pixels is inconsistent")
            normalized_pages.append(
                {
                    "page": page_index,
                    "media_box_millipoints": media_box,
                    "crop_box_millipoints": crop_box,
                    "width_millipoints": width,
                    "height_millipoints": height,
                    "rotation_degrees": rotation,
                    "png_member": png_member,
                    "png_sha256": png_pin.sha256,
                    "png_bytes": png_pin.size,
                    "rendered_width_pixels": expected_width_pixels,
                    "rendered_height_pixels": expected_height_pixels,
                    "render_a_sha256": png_pin.sha256,
                    "render_a_bytes": png_pin.size,
                    "render_b_sha256": png_pin.sha256,
                    "render_b_bytes": png_pin.size,
                },
            )
        font_values = _expect_sequence(item["fonts"], context=f"{item_context}.fonts")
        fonts = [
            _normalize_font_record(font, context=f"{item_context}.fonts[{font_index}]")
            for font_index, font in enumerate(font_values)
        ]
        font_keys = [
            (str(font["name"]), str(font["type"]), str(font["encoding"]))
            for font in fonts
        ]
        if font_keys != sorted(font_keys) or len(font_keys) != len(set(font_keys)):
            _fail(f"{item_context}.fonts must be uniquely sorted")
        issues = _derive_machine_issues(
            pdf_id=pdf_id,
            encrypted=encrypted,
            raster_image_count=raster_image_count,
            page_records=normalized_pages,
            expected_size=expected_size,
            fonts=fonts,
        )
        expected_status = "pass" if not issues else "fail"
        if item["status"] != expected_status or item["issues"] != issues:
            _fail(f"{item_context} status/issues do not match machine evidence")
        if mode == "final" and issues:
            _fail(f"final {item_context} has failing machine checks: {issues}")
        normalized_document = {
            "pdf_id": pdf_id,
            "pdf_member": pdf_member,
            "pdf_sha256": pdf_pin.sha256,
            "pdf_bytes": pdf_pin.size,
            "pdf_version": pdf_version,
            "page_count": page_count,
            "encrypted": encrypted,
            "raster_image_count": raster_image_count,
            "pdfinfo_sha256": _expect_sha256(
                item["pdfinfo_sha256"],
                context=f"{item_context}.pdfinfo_sha256",
            ),
            "pdfinfo_bytes": _expect_positive_int(
                item["pdfinfo_bytes"],
                context=f"{item_context}.pdfinfo_bytes",
            ),
            "pdffonts_sha256": _expect_sha256(
                item["pdffonts_sha256"],
                context=f"{item_context}.pdffonts_sha256",
            ),
            "pdffonts_bytes": _expect_positive_int(
                item["pdffonts_bytes"],
                context=f"{item_context}.pdffonts_bytes",
            ),
            "pdfimages_sha256": _expect_sha256(
                item["pdfimages_sha256"],
                context=f"{item_context}.pdfimages_sha256",
            ),
            "pdfimages_bytes": _expect_positive_int(
                item["pdfimages_bytes"],
                context=f"{item_context}.pdfimages_bytes",
            ),
            "pages": normalized_pages,
            "fonts": fonts,
            "status": expected_status,
            "issues": issues,
        }
        normalized_documents.append(normalized_document)
        summaries.append(
            {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "pdf_sha256": pdf_pin.sha256,
                "pdf_bytes": pdf_pin.size,
                "page_count": page_count,
                "page_size_millipoints": dict(expected_size),
                "machine_attestation_status": expected_status,
                "machine_attestation_issues": issues,
                "font_count": len(fonts),
                "font_inventory_sha256": _sha256(_canonical_json(fonts)),
                "pages": [
                    {
                        "page": page["page"],
                        "png_member": page["png_member"],
                        "png_sha256": page["png_sha256"],
                        "png_bytes": page["png_bytes"],
                    }
                    for page in normalized_pages
                ],
            },
        )
    return (
        {
            "schema": MACHINE_EVIDENCE_SCHEMA,
            "mode": mode,
            "release_id": release_id,
            "producer": producer,
            "tool_contract": MACHINE_TOOL_CONTRACT,
            "render_settings": RENDER_SETTINGS,
            "render_set_sha256": render_set_sha256,
            "documents": normalized_documents,
            "non_inference_limits": NON_INFERENCE_LIMITS,
        },
        summaries,
    )


def _normalize_visual_receipt(
    value: Mapping[str, object],
    *,
    mode: str,
    release_id: str,
    document_summaries: Sequence[dict[str, object]],
    render_set_sha256: str,
) -> tuple[dict[str, object], dict[str, int | str]]:
    context = "visual-QA receipt"
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "release_id",
            "review_kind",
            "review_id",
            "reviewer_id",
            "reviewer_role",
            "reviewed_at_utc",
            "independent_review",
            "criteria",
            "render_set_sha256",
            "documents",
            "non_inference_limits",
        },
        context=context,
    )
    if value["schema"] != VISUAL_QA_RECEIPT_SCHEMA:
        _fail(f"{context} has the wrong schema")
    if value["mode"] != mode or value["release_id"] != release_id:
        _fail(f"{context} mode/release_id does not match the plan")
    if value["review_kind"] != "human":
        _fail(f"{context}.review_kind must be human")
    if not _expect_bool(
        value["independent_review"],
        context=f"{context}.independent_review",
    ):
        _fail(f"{context}.independent_review must be true")
    if value["criteria"] != VISUAL_REVIEW_CRITERIA:
        _fail(f"{context}.criteria drifted")
    if value["render_set_sha256"] != render_set_sha256:
        _fail(f"{context}.render_set_sha256 does not bind exact rendered bytes")
    if value["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail(f"{context}.non_inference_limits drifted")
    review_id = _expect_token(value["review_id"], context=f"{context}.review_id")
    reviewer_id = _expect_token(
        value["reviewer_id"],
        context=f"{context}.reviewer_id",
    )
    reviewer_role = _expect_token(
        value["reviewer_role"],
        context=f"{context}.reviewer_role",
    )
    reviewed_at = _expect_string(
        value["reviewed_at_utc"],
        context=f"{context}.reviewed_at_utc",
    )
    if _UTC_RE.fullmatch(reviewed_at) is None:
        _fail(f"{context}.reviewed_at_utc must be second-precision UTC")
    try:
        dt.datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.UTC,
        )
    except ValueError as error:
        _fail(f"{context}.reviewed_at_utc is not a real UTC date: {error}")
    values = _expect_sequence(value["documents"], context=f"{context}.documents")
    if len(values) != len(document_summaries):
        _fail(f"{context}.documents must cover exactly four PDFs")
    normalized_documents: list[dict[str, object]] = []
    pass_pages = 0
    pending_pages = 0
    fail_pages = 0
    for index, summary in enumerate(document_summaries):
        item_context = f"{context}.documents[{index}]"
        item = _expect_mapping(values[index], context=item_context)
        _expect_keys(
            item,
            {"pdf_id", "pdf_sha256", "page_count", "pages"},
            context=item_context,
        )
        if item["pdf_id"] != summary["pdf_id"]:
            _fail(f"{item_context}.pdf_id is not in exact PDF order")
        if item["pdf_sha256"] != summary["pdf_sha256"]:
            _fail(f"{item_context}.pdf_sha256 does not match exact PDF bytes")
        if item["page_count"] != summary["page_count"]:
            _fail(f"{item_context}.page_count does not match machine evidence")
        expected_pages = _expect_sequence(
            summary["pages"],
            context=f"document summary {summary['pdf_id']} pages",
        )
        pages = _expect_sequence(item["pages"], context=f"{item_context}.pages")
        if len(pages) != len(expected_pages):
            _fail(f"{item_context}.pages does not cover every rendered page")
        normalized_pages: list[dict[str, object]] = []
        for page_index, expected_page_raw in enumerate(expected_pages, start=1):
            page_context = f"{item_context}.pages[{page_index - 1}]"
            page = _expect_mapping(pages[page_index - 1], context=page_context)
            expected_page = _expect_mapping(
                expected_page_raw,
                context=f"expected {page_context}",
            )
            _expect_keys(
                page,
                {
                    "reviewer_id",
                    "reviewer_role",
                    "pdf_role",
                    "page",
                    "pdf_sha256",
                    "png_member",
                    "png_sha256",
                    "render_set_sha256",
                    "decision",
                    "issue_codes",
                },
                context=page_context,
            )
            if (
                page["reviewer_id"] != reviewer_id
                or page["reviewer_role"] != reviewer_role
            ):
                _fail(f"{page_context} does not bind the human reviewer and role")
            if page["pdf_role"] != PDF_ROLE_BY_ID[str(summary["pdf_id"])]:
                _fail(f"{page_context}.pdf_role is not the fixed PDF role")
            if page["pdf_sha256"] != summary["pdf_sha256"]:
                _fail(f"{page_context}.pdf_sha256 does not bind the reviewed PDF")
            if page["render_set_sha256"] != render_set_sha256:
                _fail(f"{page_context} does not bind the reviewed render set")
            if page["page"] != page_index:
                _fail(f"{page_context}.page is not sequential")
            if (
                page["png_member"] != expected_page["png_member"]
                or page["png_sha256"] != expected_page["png_sha256"]
            ):
                _fail(f"{page_context} does not bind the exact PNG digest")
            decision = _expect_string(
                page["decision"],
                context=f"{page_context}.decision",
            )
            if decision not in {"pass", "pending", "fail"}:
                _fail(f"{page_context}.decision is invalid")
            raw_issue_codes = _expect_sequence(
                page["issue_codes"],
                context=f"{page_context}.issue_codes",
            )
            issue_codes = [
                _expect_token(
                    code,
                    context=f"{page_context}.issue_codes[{issue_index}]",
                )
                for issue_index, code in enumerate(raw_issue_codes)
            ]
            if issue_codes != sorted(set(issue_codes)):
                _fail(f"{page_context}.issue_codes must be uniquely sorted")
            if decision == "pass" and issue_codes:
                _fail(f"{page_context} passing review cannot retain issue codes")
            if decision != "pass" and not issue_codes:
                _fail(f"{page_context} non-passing review requires issue codes")
            if decision == "pass":
                pass_pages += 1
            elif decision == "pending":
                pending_pages += 1
            else:
                fail_pages += 1
            normalized_pages.append(
                {
                    "page": page_index,
                    "reviewer_id": reviewer_id,
                    "reviewer_role": reviewer_role,
                    "pdf_role": PDF_ROLE_BY_ID[str(summary["pdf_id"])],
                    "pdf_sha256": summary["pdf_sha256"],
                    "png_member": expected_page["png_member"],
                    "png_sha256": expected_page["png_sha256"],
                    "render_set_sha256": render_set_sha256,
                    "decision": decision,
                    "issue_codes": issue_codes,
                },
            )
        normalized_documents.append(
            {
                "pdf_id": summary["pdf_id"],
                "pdf_sha256": summary["pdf_sha256"],
                "page_count": summary["page_count"],
                "pages": normalized_pages,
            },
        )
    if mode == "final" and (pending_pages or fail_pages):
        _fail("final visual-QA receipt must pass every page")
    overall_status = "pass"
    if fail_pages:
        overall_status = "fail"
    elif pending_pages:
        overall_status = "pending"
    normalized = {
        "schema": VISUAL_QA_RECEIPT_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "review_kind": "human",
        "review_id": review_id,
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "reviewed_at_utc": reviewed_at,
        "independent_review": True,
        "criteria": VISUAL_REVIEW_CRITERIA,
        "render_set_sha256": render_set_sha256,
        "documents": normalized_documents,
        "non_inference_limits": NON_INFERENCE_LIMITS,
    }
    return (
        normalized,
        {
            "status": overall_status,
            "pass_page_count": pass_pages,
            "pending_page_count": pending_pages,
            "fail_page_count": fail_pages,
        },
    )


def _validate_plan_bindings(
    plan: Mapping[str, object],
    config: RenderedDocumentEvidenceInputs,
) -> None:
    bindings = _expect_mapping(plan["bindings"], context="plan bindings")
    expected = {
        "document_reconciliation_sha256": config.expected_reconciliation_sha256,
        "artifact_registry_sha256": config.expected_artifact_registry_sha256,
        "release_evidence_sha256": config.expected_release_evidence_sha256,
        "document_anchor_sha256": config.expected_document_anchor_sha256,
        "source_snapshot_anchor_sha256": (
            config.expected_source_snapshot_anchor_sha256
        ),
        "derivation_evidence_sha256": config.expected_derivation_evidence_sha256,
        "machine_evidence_sha256": config.expected_machine_evidence_sha256,
        "visual_qa_receipt_sha256": config.expected_visual_qa_receipt_sha256,
    }
    for key, raw_expected in expected.items():
        if key == "visual_qa_receipt_sha256" and raw_expected is None:
            if bindings[key] is not None:
                _fail(f"rendered-document input unexpectedly binds {key}")
            continue
        expected_digest = _expect_sha256(raw_expected, context=f"expected {key}")
        if bindings[key] != expected_digest:
            _fail(f"rendered-document input does not bind expected {key}")


def _validate_native_reconciliation(
    config: RenderedDocumentEvidenceInputs,
) -> _reconciliation.DocumentReconciliationReceipt:
    try:
        receipt = _reconciliation.validate_document_reconciliation(
            config.reconciliation_path,
            config.artifact_registry_path,
            config.renderer_root,
            config.rendered_output_root,
            config.document_anchor_path,
            config.document_root,
            expected_manifest_sha256=config.expected_reconciliation_sha256,
            expected_artifact_registry_sha256=(
                config.expected_artifact_registry_sha256
            ),
            expected_document_anchor_sha256=config.expected_document_anchor_sha256,
        )
    except (_reconciliation.DocumentReconciliationError, OSError) as error:
        _fail(f"native document reconciliation validation failed: {error}")
    if receipt.manifest_path != str(config.reconciliation_path.absolute()):
        _fail("native document reconciliation returned the wrong manifest path")
    if receipt.manifest_sha256 != config.expected_reconciliation_sha256:
        _fail("native document reconciliation returned the wrong manifest digest")
    return receipt


def _validate_native_release_evidence(
    config: RenderedDocumentEvidenceInputs,
) -> _release_evidence.ReleaseEvidenceReceipt:
    try:
        receipt = _release_evidence.validate_release_evidence_closure(
            config.release_evidence_path,
            config.artifact_registry_path,
            config.renderer_root,
            config.rendered_output_root,
            config.gate_receipt_root,
            config.source_data_root,
            expected_closure_sha256=config.expected_release_evidence_sha256,
            expected_artifact_registry_sha256=(
                config.expected_artifact_registry_sha256
            ),
        )
    except (_release_evidence.ReleaseEvidenceError, OSError) as error:
        _fail(f"native release-evidence validation failed: {error}")
    if receipt.manifest_path != str(config.release_evidence_path.absolute()):
        _fail("native release-evidence validation returned the wrong manifest path")
    if receipt.manifest_sha256 != config.expected_release_evidence_sha256:
        _fail("native release-evidence validation returned the wrong manifest digest")
    for field in (
        receipt.gate_receipt_count,
        receipt.source_member_count,
        receipt.ready_count,
        receipt.omitted_count,
    ):
        if isinstance(field, bool) or not isinstance(field, int) or field < 0:
            _fail("native release-evidence validation returned an invalid count")
    return receipt


def _reconciliation_summary(
    value: Mapping[str, object],
    receipt: _reconciliation.DocumentReconciliationReceipt,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    context = "document reconciliation manifest"
    if value.get("schema") != _reconciliation.DOCUMENT_RECONCILIATION_SCHEMA:
        _fail(f"{context} has the wrong schema")
    mode = _expect_mode(value.get("mode"), context=f"{context}.mode")
    release_id = _expect_token(
        value.get("release_id"),
        context=f"{context}.release_id",
    )
    inputs = _expect_mapping(value.get("inputs"), context=f"{context}.inputs")
    document_anchor_sha = _expect_sha256(
        inputs.get("document_anchor_sha256"),
        context=f"{context}.inputs.document_anchor_sha256",
    )
    summary = _expect_mapping(value.get("summary"), context=f"{context}.summary")
    pending_count = _expect_nonnegative_int(
        summary.get("pending_count"),
        context=f"{context}.summary.pending_count",
    )
    placement_count = _expect_nonnegative_int(
        summary.get("placement_count"),
        context=f"{context}.summary.placement_count",
    )
    ready_count = _expect_nonnegative_int(
        summary.get("ready_count"),
        context=f"{context}.summary.ready_count",
    )
    omitted_count = _expect_nonnegative_int(
        summary.get("omitted_count"),
        context=f"{context}.summary.omitted_count",
    )
    if (
        receipt.manifest_sha256 != expected_sha256
        or receipt.mode != mode
        or receipt.pending_count != pending_count
        or receipt.placement_count != placement_count
        or receipt.ready_count != ready_count
        or receipt.omitted_count != omitted_count
    ):
        _fail("native reconciliation receipt contradicts its validated manifest")
    registry = _expect_mapping(
        value.get("artifact_registry"),
        context=f"{context}.artifact_registry",
    )
    registry_ready = _expect_nonnegative_int(
        registry.get("ready_count"),
        context=f"{context}.artifact_registry.ready_count",
    )
    registry_omitted = _expect_nonnegative_int(
        registry.get("omitted_count"),
        context=f"{context}.artifact_registry.omitted_count",
    )
    page_locations: list[dict[str, object]] = []
    placements = _expect_sequence(
        value.get("placements"),
        context=f"{context}.placements",
    )
    for index, raw_placement in enumerate(placements):
        placement = _expect_mapping(
            raw_placement,
            context=f"{context}.placements[{index}]",
        )
        document_id = placement.get("document_id")
        location_raw = placement.get("page_location")
        if document_id not in {"main", "s1"} or location_raw is None:
            continue
        location = _expect_mapping(
            location_raw,
            context=f"{context}.placements[{index}].page_location",
        )
        page_locations.append(
            {
                "placement_id": _expect_token(
                    placement.get("placement_id"),
                    context=f"{context}.placements[{index}].placement_id",
                ),
                "document_id": document_id,
                "pdf_sha256": _expect_sha256(
                    location.get("pdf_sha256"),
                    context=f"{context}.placements[{index}].page_location.pdf_sha256",
                ),
                "page": _expect_positive_int(
                    location.get("page"),
                    context=f"{context}.placements[{index}].page_location.page",
                ),
            },
        )
    page_locations.sort(key=lambda record: str(record["placement_id"]))
    return {
        "schema": _reconciliation.DOCUMENT_RECONCILIATION_SCHEMA,
        "manifest_path": receipt.manifest_path,
        "sha256": expected_sha256,
        "mode": mode,
        "release_id": release_id,
        "document_anchor_sha256": document_anchor_sha,
        "placement_count": receipt.placement_count,
        "ready_count": receipt.ready_count,
        "omitted_count": receipt.omitted_count,
        "pending_count": receipt.pending_count,
        "artifact_registry_sha256": _expect_sha256(
            _expect_mapping(value.get("inputs"), context=f"{context}.inputs").get(
                "artifact_registry_sha256",
            ),
            context=f"{context}.inputs.artifact_registry_sha256",
        ),
        "artifact_ready_count": registry_ready,
        "artifact_omitted_count": registry_omitted,
        "page_locations": page_locations,
    }


def _builder_record(pinned: _PinnedFile) -> dict[str, object]:
    return {
        "script": _BUILDER_MEMBER,
        "sha256": pinned.sha256,
        "bytes": pinned.size,
    }


def _receipt(
    path: Path,
    manifest: Mapping[str, object],
    raw: bytes,
) -> RenderedDocumentEvidenceReceipt:
    summary = _expect_mapping(manifest["summary"], context="manifest.summary")
    return RenderedDocumentEvidenceReceipt(
        manifest_path=str(path.absolute()),
        manifest_sha256=_sha256(raw),
        mode=str(manifest["mode"]),
        pdf_count=int(summary["pdf_count"]),
        page_count=int(summary["page_count"]),
        machine_attested_pass_count=int(summary["machine_attested_pass_count"]),
        visual_pass_page_count=int(summary["visual_pass_page_count"]),
        promotable=bool(summary["promotable"]),
    )


def _pread_exact(
    descriptor: int,
    size: int,
    offset: int,
    *,
    context: str,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        try:
            chunk = os.pread(descriptor, min(remaining, _READ_CHUNK_BYTES), position)
        except OSError as error:
            _fail(f"cannot read {context}: {error}")
        if not chunk:
            _fail(f"{context} ended before its declared size")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_png_dimensions(
    pinned: _PinnedFile,
    *,
    context: str,
) -> tuple[int, int]:
    if (
        _pread_exact(
            pinned.descriptor,
            len(_PNG_SIGNATURE),
            0,
            context=context,
        )
        != _PNG_SIGNATURE
    ):
        _fail(f"{context} lacks the PNG signature")
    offset = len(_PNG_SIGNATURE)
    seen_ihdr = False
    seen_plte = False
    seen_idat = False
    seen_iend = False
    seen_phys = False
    idat_closed = False
    unsupported_ancillary: bytes | None = None
    width = 0
    height = 0
    color_type = 0
    expected_decoded = 0
    decoded_count = 0
    next_filter_offset = 0
    row_stride = 0
    decompressor: zlib.Decompress | None = None

    def consume_decoded(decoded: bytes) -> None:
        nonlocal decoded_count, next_filter_offset
        if decoded_count + len(decoded) > expected_decoded:
            _fail(f"{context} expands beyond its exact scanline bound")
        end = decoded_count + len(decoded)
        while next_filter_offset < end:
            filter_value = decoded[next_filter_offset - decoded_count]
            if filter_value > 4:
                _fail(f"{context} has an invalid PNG scanline filter")
            next_filter_offset += row_stride
        decoded_count = end

    while offset < pinned.size:
        header = _pread_exact(
            pinned.descriptor,
            8,
            offset,
            context=f"{context} chunk header",
        )
        length = struct.unpack(">I", header[:4])[0]
        chunk_type = header[4:]
        if any(
            byte not in range(ord("A"), ord("Z") + 1)
            and byte not in range(ord("a"), ord("z") + 1)
            for byte in chunk_type
        ) or not ord("A") <= chunk_type[2] <= ord("Z"):
            _fail(f"{context} has an invalid PNG chunk type code")
        data_offset = offset + 8
        end = data_offset + length + 4
        if end > pinned.size:
            _fail(f"{context} has a PNG chunk outside the file")
        if chunk_type in {b"acTL", b"fcTL", b"fdAT"}:
            _fail(f"{context} must not be APNG")
        if not seen_ihdr and (chunk_type != b"IHDR" or length != 13):
            _fail(f"{context} must begin with one 13-byte IHDR")
        if seen_ihdr and chunk_type == b"IHDR":
            _fail(f"{context} contains multiple IHDR chunks")
        if (
            chunk_type not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
            and 65 <= chunk_type[0] <= 90
        ):
            _fail(f"{context} contains an unknown critical PNG chunk")
        if 97 <= chunk_type[0] <= 122 and chunk_type != b"pHYs":
            unsupported_ancillary = unsupported_ancillary or chunk_type
        if chunk_type == b"pHYs":
            if seen_phys or seen_idat or length != 9:
                _fail(f"{context} has invalid pHYs ordering or size")
            seen_phys = True
        if chunk_type == b"PLTE":
            if seen_plte or seen_idat or not 3 <= length <= 768 or length % 3:
                _fail(f"{context} has invalid PLTE ordering or size")
            if color_type in {0, 4}:
                _fail(f"{context} grayscale PNG must not contain PLTE")
            seen_plte = True
        if chunk_type == b"IDAT":
            if idat_closed or not seen_ihdr:
                _fail(f"{context} has nonconsecutive or misplaced IDAT chunks")
            seen_idat = True
        elif seen_idat and chunk_type != b"IEND":
            idat_closed = True
        if chunk_type == b"IEND" and (length != 0 or not seen_idat):
            _fail(f"{context} has invalid IEND ordering or size")

        crc = zlib.crc32(chunk_type)
        remaining = length
        position = data_offset
        captured_chunk = b""
        while remaining:
            block = _pread_exact(
                pinned.descriptor,
                min(remaining, _READ_CHUNK_BYTES),
                position,
                context=f"{context} chunk data",
            )
            crc = zlib.crc32(block, crc)
            if chunk_type in {b"IHDR", b"pHYs"}:
                captured_chunk += block
            elif chunk_type == b"IDAT":
                if decompressor is None:
                    decompressor = zlib.decompressobj()
                compressed = block
                try:
                    while compressed:
                        limit = min(
                            _READ_CHUNK_BYTES,
                            expected_decoded - decoded_count + 1,
                        )
                        decoded = decompressor.decompress(compressed, max(limit, 1))
                        consume_decoded(decoded)
                        compressed = decompressor.unconsumed_tail
                except zlib.error as error:
                    _fail(f"{context} has invalid zlib image data: {error}")
                if decompressor.unused_data:
                    _fail(f"{context} contains data after its zlib stream")
            position += len(block)
            remaining -= len(block)
        declared_crc = struct.unpack(
            ">I",
            _pread_exact(
                pinned.descriptor,
                4,
                data_offset + length,
                context=f"{context} chunk CRC",
            ),
        )[0]
        if crc & 0xFFFFFFFF != declared_crc:
            _fail(f"{context} has an invalid PNG chunk CRC")

        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", captured_chunk[:8])
            bit_depth, color_type, compression, filtering, interlace = captured_chunk[
                8:13
            ]
            if not width or not height:
                _fail(f"{context} has zero PNG dimensions")
            if (
                bit_depth != 8
                or color_type not in {0, 2, 4, 6}
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                _fail(f"{context} has an unsupported IHDR encoding")
            if width > _MAX_PAGE_EDGE_PIXELS or height > _MAX_PAGE_EDGE_PIXELS:
                _fail(f"{context} exceeds the maximum page edge")
            if width * height > _MAX_PAGE_PIXELS:
                _fail(f"{context} exceeds the maximum page pixel count")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
            row_stride = width * channels + 1
            expected_decoded = row_stride * height
            seen_ihdr = True
        if chunk_type == b"pHYs":
            x_resolution, y_resolution, unit = struct.unpack(">IIB", captured_chunk)
            if (
                unit != 1
                or x_resolution != _EXPECTED_PNG_PIXELS_PER_METER
                or y_resolution != _EXPECTED_PNG_PIXELS_PER_METER
            ):
                _fail(f"{context} pHYs does not encode the fixed 150 dpi profile")
        if chunk_type == b"IEND":
            if end != pinned.size:
                _fail(f"{context} has bytes after IEND")
            seen_iend = True
            break
        offset = end
    if not seen_ihdr or not seen_idat or not seen_iend or decompressor is None:
        _fail(f"{context} lacks a complete PNG structure")
    try:
        consume_decoded(
            decompressor.flush(max(expected_decoded - decoded_count + 1, 1)),
        )
    except zlib.error as error:
        _fail(f"{context} has invalid terminal zlib data: {error}")
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or decoded_count != expected_decoded
        or next_filter_offset != expected_decoded
    ):
        _fail(f"{context} does not decode to its exact complete scanline geometry")
    if unsupported_ancillary is not None:
        _fail(
            f"{context} contains unsupported ancillary PNG chunk "
            f"{unsupported_ancillary.decode('ascii')}",
        )
    return width, height


def _assert_render_signatures(
    pdf_pins: Mapping[str, _PinnedFile],
    png_pins: Mapping[str, _PinnedFile],
) -> dict[str, tuple[int, int]]:
    for pdf_id, pinned in pdf_pins.items():
        prefix = _pread_exact(
            pinned.descriptor,
            min(len(_PDF_SIGNATURE), pinned.size),
            0,
            context=f"PDF {pdf_id} prefix",
        )
        tail_size = min(1024, pinned.size)
        tail = _pread_exact(
            pinned.descriptor,
            tail_size,
            pinned.size - tail_size,
            context=f"PDF {pdf_id} tail",
        )
        if prefix != _PDF_SIGNATURE or b"%%EOF" not in tail:
            _fail(f"PDF {pdf_id} lacks the bounded PDF signature/EOF marker")
    if len({pinned.sha256 for pinned in pdf_pins.values()}) != len(PDF_ORDER):
        _fail("the four PDF roles must have byte-distinct documents")
    return {
        member: _parse_png_dimensions(pinned, context=f"PNG {member}")
        for member, pinned in png_pins.items()
    }


def _render_set(
    plan_documents: Sequence[Mapping[str, object]],
    pdf_pins: Mapping[str, _PinnedFile],
    png_pins: Mapping[str, _PinnedFile],
) -> tuple[list[dict[str, object]], str]:
    documents: list[dict[str, object]] = []
    for plan_document in plan_documents:
        pdf_id = str(plan_document["pdf_id"])
        page_count = int(plan_document["page_count"])
        pdf_pin = pdf_pins[pdf_id]
        documents.append(
            {
                "pdf_id": pdf_id,
                "pdf_role": PDF_ROLE_BY_ID[pdf_id],
                "pdf_member": PDF_MEMBER_BY_ID[pdf_id],
                "pdf_sha256": pdf_pin.sha256,
                "pdf_bytes": pdf_pin.size,
                "page_count": page_count,
                "pages": [
                    {
                        "page": page,
                        "png_member": _expected_png_member(pdf_id, page),
                        "png_sha256": png_pins[
                            _expected_png_member(pdf_id, page)
                        ].sha256,
                        "png_bytes": png_pins[_expected_png_member(pdf_id, page)].size,
                    }
                    for page in range(1, page_count + 1)
                ],
            },
        )
    digest_payload = {
        "render_settings": RENDER_SETTINGS,
        "documents": documents,
    }
    return documents, _sha256(_canonical_json(digest_payload))


def _validate_reconciliation_page_locations(
    reconciliation_summary: Mapping[str, object],
    document_summaries: Sequence[Mapping[str, object]],
    *,
    mode: str,
) -> dict[str, object]:
    summary_by_id = {str(item["pdf_id"]): item for item in document_summaries}
    locations = _expect_sequence(
        reconciliation_summary["page_locations"],
        context="reconciliation page locations",
    )
    normalized: list[dict[str, object]] = []
    represented: set[str] = set()
    for index, raw_location in enumerate(locations):
        location = _expect_mapping(
            raw_location,
            context=f"reconciliation page locations[{index}]",
        )
        document_id = str(location["document_id"])
        pdf_id = "clean" if document_id == "main" else "s1"
        represented.add(document_id)
        pdf_summary = summary_by_id[pdf_id]
        if location["pdf_sha256"] != pdf_summary["pdf_sha256"]:
            _fail(
                f"reconciliation placement {location['placement_id']} does not bind "
                f"the exact {pdf_id} PDF",
            )
        if int(location["page"]) > int(pdf_summary["page_count"]):
            _fail(
                f"reconciliation placement {location['placement_id']} page is outside "
                f"the {pdf_id} PDF",
            )
        normalized.append(dict(location))
    if mode == "final" and represented != {"main", "s1"}:
        _fail("final reconciliation must locate both main and S1 rendered PDFs")
    return {
        "placement_count": len(normalized),
        "bindings_sha256": _sha256(_canonical_json(normalized)),
    }


def _validate_global_inode_uniqueness(
    metadata: Sequence[tuple[_PinnedFile, str]],
    rendered: Sequence[tuple[_PinnedFile, str]],
) -> None:
    by_identity: dict[tuple[int, int], str] = {}
    for pinned, context in (*metadata, *rendered):
        identity = (pinned.device, pinned.inode)
        prior = by_identity.get(identity)
        if prior is not None:
            _fail(f"{context} aliases input {prior}")
        by_identity[identity] = context


def _metadata_specs(
    config: RenderedDocumentEvidenceInputs,
) -> tuple[tuple[Path, str, str, int], ...]:
    required = [
        (
            config.plan_path,
            "rendered-document input",
            config.expected_plan_sha256,
            _MAX_METADATA_BYTES,
        ),
        (
            config.source_snapshot_anchor_path,
            "source-snapshot anchor",
            config.expected_source_snapshot_anchor_sha256,
            _MAX_METADATA_BYTES,
        ),
        (
            config.derivation_evidence_path,
            "derivation evidence",
            config.expected_derivation_evidence_sha256,
            _MAX_METADATA_BYTES,
        ),
        (
            config.machine_evidence_path,
            "machine evidence",
            config.expected_machine_evidence_sha256,
            _MAX_METADATA_BYTES,
        ),
    ]
    if (
        config.visual_qa_receipt_path is not None
        and config.expected_visual_qa_receipt_sha256 is not None
    ):
        required.append(
            (
                config.visual_qa_receipt_path,
                "visual-QA receipt",
                config.expected_visual_qa_receipt_sha256,
                _MAX_VISUAL_RECEIPT_BYTES,
            ),
        )
    return tuple(required)


@contextlib.contextmanager
def _prepare_evidence(
    config: RenderedDocumentEvidenceInputs,
    *,
    destination: Path | None,
) -> Iterator[_PreparedEvidence]:
    _validate_path_topology(config, destination=destination)
    metadata: list[tuple[_PinnedFile, str]] = []
    roots: list[tuple[_PinnedRoot, str]] = []
    rendered: list[tuple[_PinnedFile, str]] = []
    prepared: _PreparedEvidence | None = None
    try:
        metadata_by_context: dict[str, _PinnedFile] = {}
        for path, context, expected_sha, maximum in _metadata_specs(config):
            pinned = _pin_file(path, maximum=maximum, context=context)
            metadata.append((pinned, context))
            metadata_by_context[context] = pinned
            _validate_independent_digest(pinned, expected_sha, context=context)
        plan = _normalize_plan(
            _parse_canonical_json(
                metadata_by_context["rendered-document input"],
                context="rendered-document input",
            ),
        )
        _validate_plan_bindings(plan, config)
        # Both native upstream validators run before either rendered-byte root opens.
        native_receipt = _validate_native_reconciliation(config)
        release_receipt = _validate_native_release_evidence(config)
        _validate_fd_headroom(
            sum(
                int(_expect_mapping(item, context="plan document")["page_count"])
                for item in _expect_sequence(
                    plan["documents"],
                    context="normalized plan documents",
                )
            ),
            gate_receipt_count=release_receipt.gate_receipt_count,
            source_member_count=release_receipt.source_member_count,
        )

        for path, context, expected_sha in (
            (
                config.reconciliation_path,
                "document reconciliation manifest",
                config.expected_reconciliation_sha256,
            ),
            (
                config.artifact_registry_path,
                "artifact registry",
                config.expected_artifact_registry_sha256,
            ),
            (
                config.release_evidence_path,
                "release-evidence closure",
                config.expected_release_evidence_sha256,
            ),
            (
                config.document_anchor_path,
                "document anchor",
                config.expected_document_anchor_sha256,
            ),
            (Path(__file__), "live rendered-document builder", ""),
        ):
            pinned = _pin_file(path, maximum=_MAX_METADATA_BYTES, context=context)
            metadata.append((pinned, context))
            metadata_by_context[context] = pinned
            if expected_sha:
                _validate_independent_digest(pinned, expected_sha, context=context)
        reconciliation_value = _parse_canonical_json(
            metadata_by_context["document reconciliation manifest"],
            context="document reconciliation manifest",
        )
        reconciliation_summary = _reconciliation_summary(
            reconciliation_value,
            native_receipt,
            expected_sha256=config.expected_reconciliation_sha256,
        )
        mode = str(plan["mode"])
        release_id = str(plan["release_id"])
        if mode == "final" and reconciliation_summary["mode"] != "final":
            _fail("final plan requires a final document reconciliation")
        if mode == "draft" and reconciliation_summary["mode"] not in {
            "draft",
            "final",
        }:
            _fail("draft plan has an invalid document reconciliation mode")
        if reconciliation_summary["release_id"] != release_id:
            _fail("plan release_id does not match native document reconciliation")
        if (
            reconciliation_summary["document_anchor_sha256"]
            != config.expected_document_anchor_sha256
        ):
            _fail("document reconciliation does not bind the expected document anchor")
        if mode == "final" and reconciliation_summary["pending_count"] != 0:
            _fail("final evidence requires a final zero-pending reconciliation")
        if release_receipt.manifest_sha256 != config.expected_release_evidence_sha256:
            _fail("native release-evidence receipt returned the wrong digest")
        if (
            reconciliation_summary["artifact_registry_sha256"]
            != config.expected_artifact_registry_sha256
        ):
            _fail("document reconciliation binds a different artifact registry")
        if (
            release_receipt.ready_count
            != reconciliation_summary["artifact_ready_count"]
            or release_receipt.omitted_count
            != reconciliation_summary["artifact_omitted_count"]
        ):
            _fail("release evidence and document reconciliation disagree on artifacts")
        release_summary = {
            "schema": _release_evidence.CLOSURE_SCHEMA,
            "manifest_path": release_receipt.manifest_path,
            "sha256": release_receipt.manifest_sha256,
            "gate_receipt_count": release_receipt.gate_receipt_count,
            "source_member_count": release_receipt.source_member_count,
            "ready_count": release_receipt.ready_count,
            "omitted_count": release_receipt.omitted_count,
            "artifact_registry_sha256": config.expected_artifact_registry_sha256,
        }

        source_snapshot = _normalize_source_snapshot_anchor(
            _parse_canonical_json(
                metadata_by_context["source-snapshot anchor"],
                context="source-snapshot anchor",
            ),
            release_id=release_id,
            document_anchor_sha256=config.expected_document_anchor_sha256,
        )
        pdf_root = _pin_root(config.pdf_root, context="PDF root")
        png_root = _pin_root(config.png_root, context="PNG root")
        roots.extend(((pdf_root, "PDF root"), (png_root, "PNG root")))
        plan_documents = _expect_sequence(
            plan["documents"],
            context="normalized plan documents",
        )
        expected_png_members = tuple(
            _expected_png_member(str(document["pdf_id"]), page)
            for raw_document in plan_documents
            for document in [_expect_mapping(raw_document, context="plan document")]
            for page in range(1, int(document["page_count"]) + 1)
        )
        pdf_sizes = _validate_exact_root_inventory(
            pdf_root,
            expected=PDF_MEMBERS,
            context="PDF root",
            maximum_each=_MAX_PDF_BYTES,
            maximum_total=_MAX_TOTAL_PDF_BYTES,
        )
        png_sizes = _validate_exact_root_inventory(
            png_root,
            expected=expected_png_members,
            context="PNG root",
            maximum_each=_MAX_PNG_BYTES,
            maximum_total=_MAX_TOTAL_PNG_BYTES,
        )
        pdf_pins: dict[str, _PinnedFile] = {}
        for pdf_id, member in PDF_ORDER:
            pinned = _open_root_member(
                pdf_root,
                member,
                maximum=_MAX_PDF_BYTES,
                expected_size=pdf_sizes[member],
                context=f"PDF {pdf_id}",
            )
            rendered.append((pinned, f"PDF {pdf_id}"))
            pdf_pins[pdf_id] = pinned
        if sum(pin.size for pin in pdf_pins.values()) > _MAX_TOTAL_PDF_BYTES:
            _fail("four-PDF inventory exceeds the total PDF byte limit")
        png_pins: dict[str, _PinnedFile] = {}
        for member in expected_png_members:
            pinned = _open_root_member(
                png_root,
                member,
                maximum=_MAX_PNG_BYTES,
                expected_size=png_sizes[member],
                context=f"PNG {member}",
            )
            rendered.append((pinned, f"PNG {member}"))
            png_pins[member] = pinned
        if sum(pin.size for pin in png_pins.values()) > _MAX_TOTAL_PNG_BYTES:
            _fail("page-render inventory exceeds the total PNG byte limit")
        png_dimensions = _assert_render_signatures(pdf_pins, png_pins)
        _validate_global_inode_uniqueness(metadata, rendered)
        normalized_plan_documents = [
            _expect_mapping(document, context="plan document")
            for document in plan_documents
        ]
        render_set, render_set_sha256 = _render_set(
            normalized_plan_documents,
            pdf_pins,
            png_pins,
        )
        machine, document_summaries = _normalize_machine_evidence(
            _parse_canonical_json(
                metadata_by_context["machine evidence"],
                context="machine evidence",
            ),
            mode=mode,
            release_id=release_id,
            plan_documents=normalized_plan_documents,
            pdf_pins=pdf_pins,
            png_pins=png_pins,
            png_dimensions=png_dimensions,
            render_set_sha256=render_set_sha256,
        )
        derivation = _normalize_derivation_evidence(
            _parse_canonical_json(
                metadata_by_context["derivation evidence"],
                context="derivation evidence",
            ),
            mode=mode,
            release_id=release_id,
            reconciliation_sha256=config.expected_reconciliation_sha256,
            release_evidence_sha256=config.expected_release_evidence_sha256,
            source_snapshot_anchor_sha256=(
                config.expected_source_snapshot_anchor_sha256
            ),
            source_snapshot=source_snapshot,
            pdf_pins=pdf_pins,
        )
        page_location_summary = _validate_reconciliation_page_locations(
            reconciliation_summary,
            document_summaries,
            mode=mode,
        )
        machine_attested_pass_count = sum(
            summary["machine_attestation_status"] == "pass"
            for summary in document_summaries
        )
        page_count = sum(int(summary["page_count"]) for summary in document_summaries)
        visual_pin = metadata_by_context.get("visual-QA receipt")
        if visual_pin is None:
            visual = None
            visual_summary: dict[str, int | str] = {
                "status": "pending",
                "pass_page_count": 0,
                "pending_page_count": page_count,
                "fail_page_count": 0,
            }
        else:
            visual, visual_summary = _normalize_visual_receipt(
                _parse_canonical_json(
                    visual_pin,
                    context="visual-QA receipt",
                ),
                mode=mode,
                release_id=release_id,
                document_summaries=document_summaries,
                render_set_sha256=render_set_sha256,
            )
        # The wrapper-level derivation and machine receipts contain independently
        # anchored declarations, but this builder does not open their referenced
        # receipts/raw outputs or invoke their producers.  Until a separate native
        # producer-evidence closure exists, this boundary must not turn those string
        # claims into a final-ready/promotable decision.
        promotable = False
        promotion_blockers = [
            "native-derivation-producer-closure-not-validated",
            "native-machine-producer-closure-not-validated",
            "human-reviewer-identity-not-authenticated",
        ]
        if visual is None:
            promotion_blockers.append("human-visual-qa-receipt-absent")
        if mode == "final":
            _fail(
                "final evidence requires separately supplied native derivation and "
                "machine producer closures; wrapper attestations are nonpromotable",
            )
        if visual is None:
            for summary in document_summaries:
                summary["visual_status"] = "pending"
        else:
            for summary, visual_document_raw in zip(
                document_summaries,
                _expect_sequence(visual["documents"], context="visual documents"),
                strict=True,
            ):
                visual_document = _expect_mapping(
                    visual_document_raw,
                    context="visual document",
                )
                visual_pages = _expect_sequence(
                    visual_document["pages"],
                    context="visual document pages",
                )
                decisions = [
                    str(_expect_mapping(page, context="visual page")["decision"])
                    for page in visual_pages
                ]
                summary["visual_status"] = (
                    "fail"
                    if "fail" in decisions
                    else "pending"
                    if "pending" in decisions
                    else "pass"
                )
        manifest: dict[str, object] = {
            "schema": RENDERED_DOCUMENT_EVIDENCE_SCHEMA,
            "contract": RENDERED_DOCUMENT_EVIDENCE_CONTRACT,
            "promotion_policy": PROMOTION_POLICY,
            "public_release_relationship": PUBLIC_RELEASE_RELATIONSHIP,
            "trust_model": TRUST_MODEL,
            "non_inference_limits": NON_INFERENCE_LIMITS,
            "mode": mode,
            "release_id": release_id,
            "inputs": {
                "plan_sha256": config.expected_plan_sha256,
                "document_reconciliation_sha256": (
                    config.expected_reconciliation_sha256
                ),
                "artifact_registry_sha256": config.expected_artifact_registry_sha256,
                "release_evidence_sha256": config.expected_release_evidence_sha256,
                "document_anchor_sha256": config.expected_document_anchor_sha256,
                "source_snapshot_anchor_sha256": (
                    config.expected_source_snapshot_anchor_sha256
                ),
                "derivation_evidence_sha256": (
                    config.expected_derivation_evidence_sha256
                ),
                "machine_evidence_sha256": config.expected_machine_evidence_sha256,
                "visual_qa_receipt_sha256": (config.expected_visual_qa_receipt_sha256),
            },
            "document_reconciliation": reconciliation_summary,
            "release_evidence": release_summary,
            "source_snapshot_anchor": source_snapshot,
            "derivation_evidence": {
                "schema": derivation["schema"],
                "sha256": config.expected_derivation_evidence_sha256,
                "reference_validation": "unverified-external-references",
                "documents": derivation["documents"],
            },
            "render_settings": RENDER_SETTINGS,
            "render_set": {
                "sha256": render_set_sha256,
                "documents": render_set,
            },
            "reconciliation_page_locations": page_location_summary,
            "machine_evidence": {
                "schema": machine["schema"],
                "sha256": config.expected_machine_evidence_sha256,
                "tool_contract": MACHINE_TOOL_CONTRACT,
                "producer": machine["producer"],
                "render_set_sha256": render_set_sha256,
                "reference_validation": "unverified-external-references",
            },
            "visual_qa": (
                None
                if visual is None
                else {
                    "schema": visual["schema"],
                    "sha256": config.expected_visual_qa_receipt_sha256,
                    "review_id": visual["review_id"],
                    "reviewer_id": visual["reviewer_id"],
                    "reviewer_role": visual["reviewer_role"],
                    "reviewed_at_utc": visual["reviewed_at_utc"],
                    "render_set_sha256": render_set_sha256,
                    **visual_summary,
                }
            ),
            "documents": document_summaries,
            "summary": {
                "pdf_count": len(document_summaries),
                "page_count": page_count,
                "png_count": len(png_pins),
                "machine_attested_pass_count": machine_attested_pass_count,
                "machine_attested_fail_count": (
                    len(document_summaries) - machine_attested_pass_count
                ),
                "visual_pass_page_count": visual_summary["pass_page_count"],
                "visual_pending_page_count": visual_summary["pending_page_count"],
                "visual_fail_page_count": visual_summary["fail_page_count"],
                "promotable": promotable,
                "promotion_blockers": promotion_blockers,
            },
            "builder": _builder_record(
                metadata_by_context["live rendered-document builder"],
            ),
        }
        manifest["manifest_payload_sha256"] = _sha256(_canonical_json(manifest))
        prepared = _PreparedEvidence(
            config=config,
            manifest=manifest,
            metadata=metadata,
            roots=roots,
            rendered=rendered,
            expected_pdf_members=PDF_MEMBERS,
            expected_png_members=expected_png_members,
        )
        prepared.revalidate()
        yield prepared
    finally:
        if prepared is not None:
            prepared.close()
        else:
            for pinned, _ in rendered:
                pinned.close()
            for root, _ in roots:
                root.close()
            for pinned, _ in metadata:
                pinned.close()


def _ensure_new_destination(destination: Path) -> tuple[Path, int]:
    absolute = destination.absolute()
    if absolute.name in {"", ".", ".."} or absolute.parent == absolute:
        _fail("destination must name one new regular file")
    try:
        parent_entry = os.lstat(absolute.parent)
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect destination parent: {error}")
    if (
        stat.S_ISLNK(parent_entry.st_mode)
        or not stat.S_ISDIR(parent_entry.st_mode)
        or resolved_parent != absolute.parent
    ):
        _fail("destination parent must be a canonical non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = os.open(absolute.parent, flags)
    except OSError as error:
        _fail(f"cannot open destination parent: {error}")
    try:
        pinned_parent = os.fstat(parent_descriptor)
        if (parent_entry.st_dev, parent_entry.st_ino) != (
            pinned_parent.st_dev,
            pinned_parent.st_ino,
        ):
            _fail("destination parent changed while it was pinned")
        try:
            os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return absolute, parent_descriptor
        _fail("rendered-document evidence destination already exists")
    except Exception:
        os.close(parent_descriptor)
        raise


def _revalidate_destination_parent(absolute: Path, descriptor: int) -> None:
    try:
        path_entry = os.lstat(absolute.parent)
        resolved = absolute.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"destination parent disappeared during publication: {error}")
    pinned = os.fstat(descriptor)
    if (
        stat.S_ISLNK(path_entry.st_mode)
        or not stat.S_ISDIR(path_entry.st_mode)
        or resolved != absolute.parent
        or (path_entry.st_dev, path_entry.st_ino) != (pinned.st_dev, pinned.st_ino)
    ):
        _fail("destination parent changed during publication")


def _verify_published(
    absolute: Path,
    parent_descriptor: int,
    *,
    expected_identity: tuple[int, int],
    expected_links: int,
    raw: bytes,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        _fail(f"cannot open published rendered-document evidence: {error}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
            or (before.st_dev, before.st_ino) != expected_identity
            or before.st_nlink != expected_links
            or before.st_size != len(raw)
        ):
            _fail("published rendered-document evidence identity is invalid")
        if (
            _read_descriptor(
                descriptor,
                maximum=max(len(raw), 1),
                context="published rendered-document evidence",
            )
            != raw
        ):
            _fail("published rendered-document evidence bytes do not match")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        ):
            _fail("published rendered-document evidence changed during readback")
        named = os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != expected_identity
            or named.st_nlink != expected_links
            or stat.S_IMODE(named.st_mode) != 0o400
        ):
            _fail("published rendered-document evidence name changed during readback")
        _revalidate_destination_parent(absolute, parent_descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(
    source: str,
    destination: str,
    directory_descriptor: int,
) -> None:
    """Atomically rename one staged name without replacing an existing name."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            function = library.renameatx_np
        except AttributeError:
            _fail(
                "platform atomic no-replace rename symbol renameatx_np is unavailable",
            )
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            directory_descriptor,
            os.fsencode(source),
            directory_descriptor,
            os.fsencode(destination),
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        try:
            function = library.renameat2
        except AttributeError:
            _fail("platform atomic no-replace rename symbol renameat2 is unavailable")
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            directory_descriptor,
            os.fsencode(source),
            directory_descriptor,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    else:
        _fail("platform lacks a supported atomic no-replace rename")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        _fail("rendered-document evidence destination already exists")
    unsupported_errors = {
        number
        for number in (
            getattr(errno, "ENOSYS", None),
            getattr(errno, "ENOTSUP", None),
            getattr(errno, "EOPNOTSUPP", None),
        )
        if number is not None
    }
    if error_number in unsupported_errors:
        _fail(
            "filesystem/platform does not support atomic no-replace rename: "
            f"{os.strerror(error_number)} (errno {error_number})",
        )
    _fail(
        "atomic no-replace publication failed: "
        f"{os.strerror(error_number)} (errno {error_number})",
    )


def _publish_no_replace(
    destination: Path,
    raw: bytes,
    *,
    revalidate: Callable[[], None],
) -> None:
    absolute, parent_descriptor = _ensure_new_destination(destination)
    staging_prefix = f".{absolute.name}.private-"
    try:
        entries = os.scandir(parent_descriptor)
    except OSError as error:
        os.close(parent_descriptor)
        _fail(f"cannot inspect destination parent for retained stages: {error}")
    retained_name: str | None = None
    try:
        with entries:
            for entry in entries:
                if entry.name.startswith(staging_prefix):
                    retained_name = entry.name
                    break
    except OSError as error:
        os.close(parent_descriptor)
        _fail(f"cannot scan destination parent for retained stages: {error}")
    if retained_name is not None:
        os.close(parent_descriptor)
        _fail(
            "retained private stage requires explicit review before retry: "
            f"{absolute.parent / retained_name}",
        )
    staging_name = f"{staging_prefix}{uuid.uuid4().hex}"
    descriptor = -1
    staging_present = False
    destination_renamed = False
    stage_verified = False
    staged_identity: tuple[int, int] | None = None
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(staging_name, flags, 0o600, dir_fd=parent_descriptor)
        staging_present = True
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("staged rendered-document evidence write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        if (
            _read_descriptor(
                descriptor,
                maximum=max(len(raw), 1),
                context="staged rendered-document evidence",
            )
            != raw
        ):
            _fail("staged rendered-document evidence bytes do not match")
        staged = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o400
            or staged.st_nlink != 1
            or staged.st_size != len(raw)
        ):
            _fail("staged rendered-document evidence identity is invalid")
        staged_identity = (staged.st_dev, staged.st_ino)
        stage_verified = True
        revalidate()
        _revalidate_destination_parent(absolute, parent_descriptor)
        _rename_no_replace(staging_name, absolute.name, parent_descriptor)
        staging_present = False
        destination_renamed = True
        os.fsync(parent_descriptor)
        _verify_published(
            absolute,
            parent_descriptor,
            expected_identity=staged_identity,
            expected_links=1,
            raw=raw,
        )
        revalidate()
        for _ in range(2):
            _verify_published(
                absolute,
                parent_descriptor,
                expected_identity=staged_identity,
                expected_links=1,
                raw=raw,
            )
            revalidate()
        _revalidate_destination_parent(absolute, parent_descriptor)
        _verify_published(
            absolute,
            parent_descriptor,
            expected_identity=staged_identity,
            expected_links=1,
            raw=raw,
        )
    except Exception as error:
        candidate_path: Path | None = None
        candidate_state = "none"
        if destination_renamed:
            candidate_path = absolute
            candidate_state = (
                "destination-name-may-be-owned-or-replaced-do-not-auto-delete"
            )
        elif staging_present:
            candidate_path = absolute.parent / staging_name
            candidate_state = (
                "private-stage-name-may-be-owned-or-replaced-do-not-auto-delete"
            )
        if candidate_path is None:
            raise
        expected_sha256 = _sha256(raw) if stage_verified else "unknown"
        expected_bytes: int | str = len(raw) if stage_verified else "unknown"
        diagnostic = (
            f"{error}; candidate_path={candidate_path}; "
            f"expected_sha256={expected_sha256}; expected_bytes={expected_bytes}; "
            f"candidate_state={candidate_state}; inspect identity and bytes before "
            "any explicit review/removal"
        )
        raise RenderedDocumentEvidenceError(diagnostic) from error
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            # Never unlink by name during failure cleanup: neither POSIX unlinkat nor
            # Python exposes an inode-conditional unlink.  After an adversarial name
            # swap, stat-then-unlink could delete an unrelated file.  A fully written
            # immutable destination or private staging file is therefore retained on
            # failure and reported by its deterministic name pattern.
            os.close(parent_descriptor)


def build_rendered_document_evidence(
    config: RenderedDocumentEvidenceInputs,
    destination: Path,
) -> RenderedDocumentEvidenceReceipt:
    """Build and atomically publish one nonpromotable wrapper-audit manifest."""
    with _prepare_evidence(config, destination=destination) as prepared:
        raw = _canonical_json(prepared.manifest) + b"\n"
        _publish_no_replace(destination, raw, revalidate=prepared.revalidate)
        return _receipt(destination, prepared.manifest, raw)


def validate_rendered_document_evidence(
    config: RenderedDocumentEvidenceInputs,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> RenderedDocumentEvidenceReceipt:
    """Validate one independently anchored evidence manifest and every input."""
    manifest_file = _pin_file(
        manifest_path,
        maximum=_MAX_METADATA_BYTES,
        context="rendered-document evidence manifest",
    )
    try:
        _validate_independent_digest(
            manifest_file,
            expected_manifest_sha256,
            context="rendered-document evidence manifest",
        )
        manifest = _parse_canonical_json(
            manifest_file,
            context="rendered-document evidence manifest",
        )
        with _prepare_evidence(config, destination=None) as prepared:
            if manifest != prepared.manifest:
                _fail("rendered-document evidence manifest is not canonical for inputs")
            prepared.revalidate()
            _revalidate_file(
                manifest_file,
                context="rendered-document evidence manifest",
            )
            return _receipt(manifest_path, manifest, manifest_file.raw)
    finally:
        manifest_file.close()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--artifact-registry", type=Path, required=True)
    parser.add_argument("--renderer-root", type=Path, required=True)
    parser.add_argument("--rendered-output-root", type=Path, required=True)
    parser.add_argument("--release-evidence", type=Path, required=True)
    parser.add_argument("--gate-receipt-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--document-anchor", type=Path, required=True)
    parser.add_argument("--document-root", type=Path, required=True)
    parser.add_argument("--source-snapshot-anchor", type=Path, required=True)
    parser.add_argument("--derivation-evidence", type=Path, required=True)
    parser.add_argument("--machine-evidence", type=Path, required=True)
    parser.add_argument(
        "--visual-qa-receipt",
        type=Path,
        help="optional only for a nonpromotable draft audit",
    )
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--png-root", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-reconciliation-sha256", required=True)
    parser.add_argument("--expected-artifact-registry-sha256", required=True)
    parser.add_argument("--expected-release-evidence-sha256", required=True)
    parser.add_argument("--expected-document-anchor-sha256", required=True)
    parser.add_argument("--expected-source-snapshot-anchor-sha256", required=True)
    parser.add_argument("--expected-derivation-evidence-sha256", required=True)
    parser.add_argument("--expected-machine-evidence-sha256", required=True)
    parser.add_argument(
        "--expected-visual-qa-receipt-sha256",
        help="required exactly when --visual-qa-receipt is supplied",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a no-replace evidence manifest")
    _add_common_arguments(build)
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate",
        help="validate an independently anchored evidence manifest",
    )
    _add_common_arguments(validate)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> RenderedDocumentEvidenceInputs:
    return RenderedDocumentEvidenceInputs(
        plan_path=args.plan.absolute(),
        reconciliation_path=args.reconciliation.absolute(),
        artifact_registry_path=args.artifact_registry.absolute(),
        renderer_root=args.renderer_root.absolute(),
        rendered_output_root=args.rendered_output_root.absolute(),
        release_evidence_path=args.release_evidence.absolute(),
        gate_receipt_root=args.gate_receipt_root.absolute(),
        source_data_root=args.source_data_root.absolute(),
        document_anchor_path=args.document_anchor.absolute(),
        document_root=args.document_root.absolute(),
        source_snapshot_anchor_path=args.source_snapshot_anchor.absolute(),
        derivation_evidence_path=args.derivation_evidence.absolute(),
        machine_evidence_path=args.machine_evidence.absolute(),
        visual_qa_receipt_path=(
            None
            if args.visual_qa_receipt is None
            else args.visual_qa_receipt.absolute()
        ),
        pdf_root=args.pdf_root.absolute(),
        png_root=args.png_root.absolute(),
        expected_plan_sha256=args.expected_plan_sha256,
        expected_reconciliation_sha256=args.expected_reconciliation_sha256,
        expected_artifact_registry_sha256=args.expected_artifact_registry_sha256,
        expected_release_evidence_sha256=args.expected_release_evidence_sha256,
        expected_document_anchor_sha256=args.expected_document_anchor_sha256,
        expected_source_snapshot_anchor_sha256=(
            args.expected_source_snapshot_anchor_sha256
        ),
        expected_derivation_evidence_sha256=(args.expected_derivation_evidence_sha256),
        expected_machine_evidence_sha256=args.expected_machine_evidence_sha256,
        expected_visual_qa_receipt_sha256=args.expected_visual_qa_receipt_sha256,
    )


def main() -> None:
    """Run the rendered-document evidence CLI."""
    args = _parser().parse_args()
    config = _config_from_args(args)
    if args.command == "build":
        receipt = build_rendered_document_evidence(config, args.output.absolute())
    else:
        receipt = validate_rendered_document_evidence(
            config,
            args.manifest.absolute(),
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(_canonical_json(asdict(receipt)).decode("ascii"))


if __name__ == "__main__":
    main()

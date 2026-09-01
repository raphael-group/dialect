"""Verify an authenticated all-page visual-QA receipt with OpenSSH sshsig.

This is a deliberately isolated promotion seam.  It validates one canonical v2
visual-QA receipt, one independently SHA-256-anchored OpenSSH allowed-signers
authority, and one independently anchored Ed25519 sshsig.  The signature covers a
fixed domain separator followed by the exact canonical receipt bytes.  Verification
uses one independently pinned ``ssh-keygen`` executable with an exact principal and
namespace, a minimal environment, no shell, bounded streams, and a fixed timeout.

A successful result proves that the receipt has a valid signature under the single
anchored Ed25519 public key.  It does *not* establish current key custody or
freshness, the signer's human identity, actual page inspection, scientific
correctness, or promotion.  Those authorities remain outside this verifier.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

VISUAL_QA_RECEIPT_SCHEMA: Final = "dialect-revision-rendered-document-visual-qa-v2"
VISUAL_QA_VERIFICATION_SCHEMA: Final = (
    "dialect-revision-rendered-document-visual-qa-verification-v1"
)
VISUAL_QA_VERIFICATION_CONTRACT: Final = "canonical-v2-ed25519-sshsig-authentication-v1"
SSHSIG_NAMESPACE: Final = "dialect-revision-rendered-document-visual-qa-v2"
DOMAIN_SEPARATOR: Final = b"DIALECT-RENDERED-DOCUMENT-VISUAL-QA-V2\n"
ATTESTATION_STATEMENT: Final = (
    "I attest that I inspected every bound rendered page and found none of the "
    "listed visual defects."
)
REVIEW_KIND: Final = "human-visual-qa-attestation"
REVIEWER_ROLE: Final = "independent-visual-reviewer"
VISUAL_REVIEW_CRITERIA: Final = [
    "page-render-present",
    "no-obvious-render-failure",
    "no-clipping",
    "no-overlap",
]
NON_INFERENCE_LIMITS: Final = {
    "human_identity_to_key_binding": "caller-authority-not-inferred",
    "actual_visual_inspection": (
        "caller-authority-signed-claim-not-independently-observed"
    ),
    "current_private_key_control_or_signature_freshness": "not-inferred",
    "reviewer_independence": "signed-claim-not-independently-verified",
    "review_timestamp_accuracy": "signed-claim-not-independently-verified",
    "upstream_manifests_and_pdf_png_bytes": "not-opened-or-verified-here",
    "independent_anchor_provenance": "caller-authority-not-inferred",
    "scientific_correctness": "not-inferred",
    "coauthor_approval": "not-inferred",
    "journal_submission_or_acceptance": "not-inferred",
    "loaded_python_and_ssh_keygen_dependency_closure": "not-native-attested",
    "detached_process_descendants": "outside-process-group-containment",
}
PDF_ORDER: Final = (
    ("clean", "manuscript-clean.pdf", "clean revised manuscript PDF"),
    ("marked", "manuscript-marked.pdf", "marked manuscript PDF"),
    ("s1", "s1-appendix.pdf", "S1 Appendix PDF"),
    ("rebuttal", "response-to-reviewers.pdf", "response to reviewers PDF"),
)

EXACT_ENVIRONMENT: Final = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
TOOL_TIMEOUT_SECONDS: Final = 10.0
MAX_RECEIPT_BYTES: Final = 2 * 1024 * 1024
MAX_AUTHORITY_BYTES: Final = 16 * 1024
MAX_SIGNATURE_BYTES: Final = 32 * 1024
MAX_TOOL_BYTES: Final = 32 * 1024 * 1024
MAX_PAYLOAD_BYTES: Final = len(DOMAIN_SEPARATOR) + MAX_RECEIPT_BYTES
MAX_STDOUT_BYTES: Final = 4096
MAX_STDERR_BYTES: Final = 4096
MAX_PAGES_PER_PDF: Final = 256
MAX_TOTAL_PAGES: Final = 512
MAX_PDF_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_PDF_BYTES: Final = 512 * 1024 * 1024
MAX_PNG_BYTES: Final = 32 * 1024 * 1024
MAX_TOTAL_PNG_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_PAGE_EDGE_PIXELS: Final = 4096
MAX_PAGE_PIXELS: Final = 8_000_000
MAX_OUTPUT_MEMBER_BYTES: Final = MAX_PAYLOAD_BYTES
MAX_OUTPUT_BYTES: Final = (
    MAX_RECEIPT_BYTES
    + MAX_AUTHORITY_BYTES
    + MAX_SIGNATURE_BYTES
    + MAX_PAYLOAD_BYTES
    + MAX_STDOUT_BYTES
    + MAX_STDERR_BYTES
    + MAX_RECEIPT_BYTES
)
MAX_OUTPUT_FILES: Final = 7
READ_CHUNK_BYTES: Final = 64 * 1024

RECEIPT_MEMBER: Final = "visual-qa-receipt.json"
AUTHORITY_MEMBER: Final = "allowed-signers"
SIGNATURE_MEMBER: Final = "visual-qa.sig"
PAYLOAD_MEMBER: Final = "signed-payload.bin"
STDOUT_MEMBER: Final = "ssh-keygen.stdout"
STDERR_MEMBER: Final = "ssh-keygen.stderr"
MANIFEST_MEMBER: Final = "verification-manifest.json"
OUTPUT_MEMBERS: Final = (
    AUTHORITY_MEMBER,
    PAYLOAD_MEMBER,
    RECEIPT_MEMBER,
    SIGNATURE_MEMBER,
    STDERR_MEMBER,
    STDOUT_MEMBER,
    MANIFEST_MEMBER,
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_PRINCIPAL_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9.@_+-]{2,127}")
_UTC_RE: Final = re.compile(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
)
_BEGIN_SIGNATURE: Final = "-----BEGIN SSH SIGNATURE-----"
_END_SIGNATURE: Final = "-----END SSH SIGNATURE-----"
_ARMOR_WIDTH: Final = 70


class VisualQaVerificationError(ValueError):
    """Raised when signed visual-QA authentication is invalid."""


@dataclass(frozen=True, slots=True)
class VisualQaVerificationInputs:
    """Name every independently anchored verification input."""

    receipt_path: Path
    allowed_signers_path: Path
    signature_path: Path
    ssh_keygen_path: Path
    expected_receipt_sha256: str
    expected_allowed_signers_sha256: str
    expected_signature_sha256: str
    expected_ssh_keygen_sha256: str
    expected_principal: str


@dataclass(frozen=True, slots=True)
class VisualQaVerificationReceipt:
    """Summarize one published verification or one retained replay."""

    manifest_path: str
    manifest_sha256: str
    visual_qa_receipt_sha256: str
    allowed_signers_sha256: str
    signature_sha256: str
    payload_sha256: str
    key_fingerprint: str
    release_id: str
    page_count: int
    authentication_status: str
    promotable: bool
    replay_root: str | None


@dataclass(slots=True)
class _PinnedFile:
    path: Path
    descriptor: int
    device: int
    inode: int
    mode: int
    nlink: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str
    parent_authority: tuple[_DirectoryAuthority, ...]

    def close(self) -> None:
        """Close the owned descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _PinnedRoot:
    path: Path
    descriptor: int
    device: int
    inode: int
    mode: int

    def close(self) -> None:
        """Close the owned descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _PinnedMember:
    member: str
    descriptor: int
    identity: tuple[int, ...]
    raw: bytes

    def close(self) -> None:
        """Close the owned descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(frozen=True, slots=True)
class _DirectoryAuthority:
    path: Path
    device: int
    inode: int
    mode: int
    uid: int
    gid: int


@dataclass(frozen=True, slots=True)
class _Authority:
    principal: str
    public_key_blob: bytes
    public_key_base64: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _SshSignature:
    public_key_blob: bytes
    namespace: str
    hash_algorithm: str
    signature_algorithm: str


@dataclass(frozen=True, slots=True)
class _Production:
    manifest: dict[str, object]
    manifest_raw: bytes
    receipt_sha256: str
    authority_sha256: str
    signature_sha256: str
    payload_sha256: str
    fingerprint: str
    release_id: str
    page_count: int


def _fail(message: str) -> NoReturn:
    raise VisualQaVerificationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


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
    observed = set(value)
    if observed != expected:
        _fail(
            f"{context} keys differ: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}",
        )


def _expect_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        _fail(f"{context} must be a string")
    return value


def _expect_token(value: object, *, context: str) -> str:
    text = _expect_string(value, context=context)
    if _TOKEN_RE.fullmatch(text) is None:
        _fail(f"{context} is not a canonical token")
    return text


def _expect_principal(value: object, *, context: str) -> str:
    text = _expect_string(value, context=context)
    if _PRINCIPAL_RE.fullmatch(text) is None or any(
        marker in text for marker in (",", "*", "?", "[", "]", "!")
    ):
        _fail(f"{context} is not one exact OpenSSH principal")
    return text


def _expect_sha256(value: object, *, context: str) -> str:
    text = _expect_string(value, context=context)
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{context} must be one lowercase SHA-256 digest")
    return text


def _expect_positive_int(value: object, *, maximum: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer")
    if value <= 0 or value > maximum:
        _fail(f"{context} is outside the bound 1..{maximum}")
    return value


def _json_no_duplicates(raw: bytes, *, context: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                _fail(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> NoReturn:
        _fail(f"{context} contains non-finite number {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{context} is not valid UTF-8 JSON: {error}")


def _stable_pdf_projection(
    documents: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "pdf_id": document["pdf_id"],
            "pdf_member": document["pdf_member"],
            "pdf_bytes": document["pdf_bytes"],
            "pdf_sha256": document["pdf_sha256"],
        }
        for document in documents
    ]


def _machine_render_projection(
    documents: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "pdf_id": document["pdf_id"],
            "pdf_sha256": document["pdf_sha256"],
            "pages": [
                {
                    "page": page["page"],
                    "member": page["png_member"],
                    "sha256": page["png_sha256"],
                    "bytes": page["png_bytes"],
                    "width_pixels": page["width_pixels"],
                    "height_pixels": page["height_pixels"],
                }
                for page in document["pages"]  # type: ignore[union-attr]
            ],
        }
        for document in documents
    ]


def _normalize_visual_receipt(raw: bytes) -> tuple[dict[str, object], int]:
    parsed = _expect_mapping(
        _json_no_duplicates(raw, context="visual-QA receipt"),
        context="visual-QA receipt",
    )
    _expect_keys(
        parsed,
        {
            "schema",
            "mode",
            "release_id",
            "review_kind",
            "review_id",
            "reviewer_principal",
            "reviewer_role",
            "reviewed_at_utc",
            "independent_review",
            "criteria",
            "attestation_statement",
            "derivation_manifest_sha256",
            "machine_manifest_sha256",
            "rebuttal_renderer_manifest_sha256",
            "pdf_set_sha256",
            "render_set_sha256",
            "documents",
            "non_inference_limits",
        },
        context="visual-QA receipt",
    )
    if parsed["schema"] != VISUAL_QA_RECEIPT_SCHEMA:
        _fail("visual-QA receipt has the wrong schema")
    if parsed["mode"] != "final":
        _fail("visual-QA receipt mode must be final")
    release_id = _expect_token(parsed["release_id"], context="release id")
    if parsed["review_kind"] != REVIEW_KIND:
        _fail("visual-QA receipt has the wrong review kind")
    review_id = _expect_token(parsed["review_id"], context="review id")
    principal = _expect_principal(
        parsed["reviewer_principal"],
        context="reviewer principal",
    )
    if parsed["reviewer_role"] != REVIEWER_ROLE:
        _fail("visual-QA receipt has the wrong reviewer role")
    reviewed_at = _expect_string(parsed["reviewed_at_utc"], context="reviewed at")
    if _UTC_RE.fullmatch(reviewed_at) is None:
        _fail("visual-QA receipt reviewed_at_utc must be second-precision UTC")
    try:
        dt.datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.UTC,
        )
    except ValueError as error:
        _fail(f"visual-QA receipt reviewed_at_utc is not a real date: {error}")
    if parsed["independent_review"] is not True:
        _fail("visual-QA receipt independent_review must be true")
    if parsed["criteria"] != VISUAL_REVIEW_CRITERIA:
        _fail("visual-QA receipt criteria drifted")
    if parsed["attestation_statement"] != ATTESTATION_STATEMENT:
        _fail("visual-QA receipt attestation statement drifted")
    if parsed["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail("visual-QA receipt non-inference limits drifted")
    derivation_sha = _expect_sha256(
        parsed["derivation_manifest_sha256"],
        context="derivation manifest SHA-256",
    )
    machine_sha = _expect_sha256(
        parsed["machine_manifest_sha256"],
        context="machine manifest SHA-256",
    )
    renderer_sha = _expect_sha256(
        parsed["rebuttal_renderer_manifest_sha256"],
        context="rebuttal renderer manifest SHA-256",
    )
    if len({derivation_sha, machine_sha, renderer_sha}) != 3:
        _fail("visual-QA receipt upstream manifest anchors must be distinct")
    pdf_set_sha = _expect_sha256(parsed["pdf_set_sha256"], context="PDF set SHA-256")
    render_set_sha = _expect_sha256(
        parsed["render_set_sha256"],
        context="render set SHA-256",
    )
    raw_documents = _expect_sequence(parsed["documents"], context="documents")
    if len(raw_documents) != len(PDF_ORDER):
        _fail("visual-QA receipt must contain exactly four documents")
    documents: list[dict[str, object]] = []
    total_pages = 0
    total_pdf_bytes = 0
    total_png_bytes = 0
    for index, (pdf_id, pdf_member, pdf_role) in enumerate(PDF_ORDER):
        context = f"documents[{index}]"
        document = _expect_mapping(raw_documents[index], context=context)
        _expect_keys(
            document,
            {
                "pdf_id",
                "pdf_member",
                "pdf_role",
                "pdf_sha256",
                "pdf_bytes",
                "page_count",
                "pages",
            },
            context=context,
        )
        if (
            document["pdf_id"] != pdf_id
            or document["pdf_member"] != pdf_member
            or document["pdf_role"] != pdf_role
        ):
            _fail(f"{context} is not in the exact fixed document order")
        pdf_sha = _expect_sha256(
            document["pdf_sha256"],
            context=f"{context}.pdf_sha256",
        )
        pdf_bytes = _expect_positive_int(
            document["pdf_bytes"],
            maximum=MAX_PDF_BYTES,
            context=f"{context}.pdf_bytes",
        )
        total_pdf_bytes += pdf_bytes
        if total_pdf_bytes > MAX_TOTAL_PDF_BYTES:
            _fail("visual-QA receipt exceeds the aggregate PDF byte bound")
        page_count = _expect_positive_int(
            document["page_count"],
            maximum=MAX_PAGES_PER_PDF,
            context=f"{context}.page_count",
        )
        raw_pages = _expect_sequence(document["pages"], context=f"{context}.pages")
        if len(raw_pages) != page_count:
            _fail(f"{context}.pages does not cover every page")
        pages: list[dict[str, object]] = []
        for page_number, raw_page in enumerate(raw_pages, start=1):
            page_context = f"{context}.pages[{page_number - 1}]"
            page = _expect_mapping(raw_page, context=page_context)
            _expect_keys(
                page,
                {
                    "page",
                    "png_member",
                    "png_sha256",
                    "png_bytes",
                    "width_pixels",
                    "height_pixels",
                    "decision",
                    "issue_codes",
                },
                context=page_context,
            )
            if page["page"] != page_number:
                _fail(f"{page_context}.page is not sequential")
            expected_png = f"pages/{pdf_id}/page-{page_number:04d}.png"
            if page["png_member"] != expected_png:
                _fail(f"{page_context}.png_member is not canonical")
            png_sha = _expect_sha256(
                page["png_sha256"],
                context=f"{page_context}.png_sha256",
            )
            png_bytes = _expect_positive_int(
                page["png_bytes"],
                maximum=MAX_PNG_BYTES,
                context=f"{page_context}.png_bytes",
            )
            total_png_bytes += png_bytes
            if total_png_bytes > MAX_TOTAL_PNG_BYTES:
                _fail("visual-QA receipt exceeds the aggregate PNG byte bound")
            width = _expect_positive_int(
                page["width_pixels"],
                maximum=MAX_PAGE_EDGE_PIXELS,
                context=f"{page_context}.width_pixels",
            )
            height = _expect_positive_int(
                page["height_pixels"],
                maximum=MAX_PAGE_EDGE_PIXELS,
                context=f"{page_context}.height_pixels",
            )
            if width * height > MAX_PAGE_PIXELS:
                _fail(f"{page_context} exceeds the page-pixel bound")
            if page["decision"] != "pass" or page["issue_codes"] != []:
                _fail(f"{page_context} must be pass with no issue codes")
            pages.append(
                {
                    "page": page_number,
                    "png_member": expected_png,
                    "png_sha256": png_sha,
                    "png_bytes": png_bytes,
                    "width_pixels": width,
                    "height_pixels": height,
                    "decision": "pass",
                    "issue_codes": [],
                },
            )
        total_pages += page_count
        if total_pages > MAX_TOTAL_PAGES:
            _fail("visual-QA receipt exceeds the aggregate page bound")
        documents.append(
            {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "pdf_role": pdf_role,
                "pdf_sha256": pdf_sha,
                "pdf_bytes": pdf_bytes,
                "page_count": page_count,
                "pages": pages,
            },
        )
    if len({str(document["pdf_sha256"]) for document in documents}) != len(PDF_ORDER):
        _fail("visual-QA receipt must bind four byte-distinct PDFs")
    if _sha256(_canonical_json(_stable_pdf_projection(documents))) != pdf_set_sha:
        _fail("visual-QA receipt PDF set digest is not self-consistent")
    if (
        _sha256(_canonical_json(_machine_render_projection(documents)))
        != render_set_sha
    ):
        _fail("visual-QA receipt render set digest is not self-consistent")
    normalized = {
        "schema": VISUAL_QA_RECEIPT_SCHEMA,
        "mode": "final",
        "release_id": release_id,
        "review_kind": REVIEW_KIND,
        "review_id": review_id,
        "reviewer_principal": principal,
        "reviewer_role": REVIEWER_ROLE,
        "reviewed_at_utc": reviewed_at,
        "independent_review": True,
        "criteria": VISUAL_REVIEW_CRITERIA,
        "attestation_statement": ATTESTATION_STATEMENT,
        "derivation_manifest_sha256": derivation_sha,
        "machine_manifest_sha256": machine_sha,
        "rebuttal_renderer_manifest_sha256": renderer_sha,
        "pdf_set_sha256": pdf_set_sha,
        "render_set_sha256": render_set_sha,
        "documents": documents,
        "non_inference_limits": NON_INFERENCE_LIMITS,
    }
    if _canonical_json(normalized) != raw:
        _fail("visual-QA receipt must use exact canonical JSON encoding")
    return normalized, total_pages


def _canonical_existing_file(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve {context}: {error}")
    if resolved != absolute:
        _fail(f"{context} must be an absolute canonical path without symlinks")
    return absolute


def _stat_identity(entry: os.stat_result) -> tuple[int, ...]:
    return (
        entry.st_dev,
        entry.st_ino,
        entry.st_mode,
        entry.st_nlink,
        entry.st_uid,
        entry.st_gid,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns,
    )


def _pin_executable_parent_authority(path: Path) -> tuple[_DirectoryAuthority, ...]:
    parents = [path.parent, *path.parent.parents]
    authority: list[_DirectoryAuthority] = []
    for parent in parents:
        try:
            entry = parent.stat(follow_symlinks=False)
        except OSError as error:
            _fail(f"cannot inspect ssh-keygen path authority {parent}: {error}")
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != 0:
            _fail("every ssh-keygen path ancestor must be a root-owned directory")
        if stat.S_IMODE(entry.st_mode) & 0o022:
            _fail("ssh-keygen path ancestors must not be group/world writable")
        authority.append(
            _DirectoryAuthority(
                path=parent,
                device=entry.st_dev,
                inode=entry.st_ino,
                mode=entry.st_mode,
                uid=entry.st_uid,
                gid=entry.st_gid,
            ),
        )
    return tuple(authority)


def _revalidate_executable_parent_authority(
    authority: Sequence[_DirectoryAuthority],
) -> None:
    if not authority:
        _fail("ssh-keygen path lacks pinned root-owned ancestor authority")
    for expected in authority:
        try:
            entry = expected.path.stat(follow_symlinks=False)
        except OSError as error:
            _fail(f"cannot revalidate ssh-keygen path authority: {error}")
        if (
            not stat.S_ISDIR(entry.st_mode)
            or entry.st_dev != expected.device
            or entry.st_ino != expected.inode
            or entry.st_mode != expected.mode
            or entry.st_uid != expected.uid
            or entry.st_gid != expected.gid
            or entry.st_uid != 0
            or stat.S_IMODE(entry.st_mode) & 0o022
        ):
            _fail("ssh-keygen path authority changed or became writable")


def _pin_file(
    path: Path,
    *,
    expected_sha256: str,
    maximum: int,
    context: str,
    executable: bool = False,
) -> tuple[_PinnedFile, bytes]:
    expected = _expect_sha256(expected_sha256, context=f"expected {context} SHA-256")
    absolute = _canonical_existing_file(path, context=context)
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
        before = os.fstat(descriptor)
        named = absolute.stat(follow_symlinks=False)
        if _stat_identity(before) != _stat_identity(named):
            _fail(f"{context} path and descriptor identities differ")
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            _fail(f"{context} must be a linked regular file")
        if before.st_size <= 0 or before.st_size > maximum:
            _fail(f"{context} exceeds its byte bound")
        if executable:
            if before.st_uid != 0:
                _fail(f"{context} must be owned by root")
            if stat.S_IMODE(before.st_mode) & 0o7022:
                _fail(f"{context} must not be privileged or group/world writable")
            if not stat.S_IMODE(before.st_mode) & 0o111:
                _fail(f"{context} must be executable")
        chunks: list[bytes] = []
        total = 0
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor,
                min(READ_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
            if not chunk:
                _fail(f"{context} ended before its recorded size")
            total += len(chunk)
            if total > maximum:
                _fail(f"{context} exceeds its byte bound")
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(before):
            _fail(f"{context} changed while it was read")
        digest = _sha256(raw)
        if digest != expected:
            _fail(f"{context} does not match its independent SHA-256 anchor")
        parent_authority = (
            _pin_executable_parent_authority(absolute) if executable else ()
        )
        return (
            _PinnedFile(
                path=absolute,
                descriptor=descriptor,
                device=before.st_dev,
                inode=before.st_ino,
                mode=before.st_mode,
                nlink=before.st_nlink,
                uid=before.st_uid,
                gid=before.st_gid,
                size=before.st_size,
                mtime_ns=before.st_mtime_ns,
                ctime_ns=before.st_ctime_ns,
                sha256=digest,
                parent_authority=parent_authority,
            ),
            raw,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_pin(pin: _PinnedFile, *, context: str) -> None:
    if pin.parent_authority:
        _revalidate_executable_parent_authority(pin.parent_authority)
    try:
        descriptor_entry = os.fstat(pin.descriptor)
        named_entry = pin.path.stat(follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot revalidate {context}: {error}")
    expected = (
        pin.device,
        pin.inode,
        pin.mode,
        pin.nlink,
        pin.uid,
        pin.gid,
        pin.size,
        pin.mtime_ns,
        pin.ctime_ns,
    )
    if _stat_identity(descriptor_entry) != expected:
        _fail(f"{context} descriptor identity changed")
    if _stat_identity(named_entry) != _stat_identity(descriptor_entry):
        _fail(f"{context} named identity changed")
    digest = hashlib.sha256()
    offset = 0
    while offset < pin.size:
        chunk = os.pread(
            pin.descriptor,
            min(READ_CHUNK_BYTES, pin.size - offset),
            offset,
        )
        if not chunk:
            _fail(f"{context} ended early during byte revalidation")
        digest.update(chunk)
        offset += len(chunk)
    if digest.hexdigest() != pin.sha256:
        _fail(f"{context} bytes changed after their SHA-256 pin")
    if _stat_identity(os.fstat(pin.descriptor)) != expected:
        _fail(f"{context} identity changed during byte revalidation")
    if pin.parent_authority:
        _revalidate_executable_parent_authority(pin.parent_authority)


def _read_ssh_string(raw: bytes, offset: int, *, context: str) -> tuple[bytes, int]:
    if offset + 4 > len(raw):
        _fail(f"{context} is truncated before an SSH string length")
    length = struct.unpack_from(">I", raw, offset)[0]
    start = offset + 4
    end = start + length
    if end > len(raw):
        _fail(f"{context} contains a truncated SSH string")
    return raw[start:end], end


def _parse_ed25519_key_blob(raw: bytes, *, context: str) -> None:
    algorithm, offset = _read_ssh_string(raw, 0, context=context)
    key, offset = _read_ssh_string(raw, offset, context=context)
    if algorithm != b"ssh-ed25519" or len(key) != 32 or offset != len(raw):
        _fail(f"{context} must be one canonical Ed25519 public-key blob")


def _fingerprint(key_blob: bytes) -> str:
    encoded = base64.b64encode(hashlib.sha256(key_blob).digest()).decode("ascii")
    return f"SHA256:{encoded.rstrip('=')}"


def _parse_authority(raw: bytes, *, expected_principal: str) -> _Authority:
    principal = _expect_principal(expected_principal, context="expected principal")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"allowed-signers authority is not ASCII: {error}")
    if "\r" in text or "\x00" in text or not text.endswith("\n"):
        _fail("allowed-signers authority must be canonical LF-terminated ASCII")
    if text.count("\n") != 1:
        _fail("allowed-signers authority must contain exactly one line")
    fields = text[:-1].split(" ")
    if len(fields) != 4 or any(not field for field in fields):
        _fail("allowed-signers authority must have one canonical four-field record")
    authority_principal, option, algorithm, encoded = fields
    expected_option = f'namespaces="{SSHSIG_NAMESPACE}"'
    if authority_principal != principal:
        _fail("allowed-signers authority does not contain the exact principal")
    if option != expected_option:
        _fail("allowed-signers authority must restrict the exact sshsig namespace")
    if algorithm != "ssh-ed25519":
        _fail("allowed-signers authority must contain only an Ed25519 key")
    try:
        key_blob = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        _fail(f"allowed-signers authority key is not canonical base64: {error}")
    if base64.b64encode(key_blob).decode("ascii") != encoded:
        _fail("allowed-signers authority key base64 is not canonical")
    _parse_ed25519_key_blob(key_blob, context="allowed-signers authority key")
    return _Authority(
        principal=principal,
        public_key_blob=key_blob,
        public_key_base64=encoded,
        fingerprint=_fingerprint(key_blob),
    )


def _parse_signature(raw: bytes, *, authority: _Authority) -> _SshSignature:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"sshsig armor is not ASCII: {error}")
    if "\r" in text or "\x00" in text or not text.endswith("\n"):
        _fail("sshsig armor must be canonical LF-terminated ASCII")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != _BEGIN_SIGNATURE or lines[-1] != _END_SIGNATURE:
        _fail("sshsig armor markers are invalid")
    encoded = "".join(lines[1:-1])
    if (
        not encoded
        or any(len(line) != _ARMOR_WIDTH for line in lines[1:-2])
        or len(lines[-2]) > _ARMOR_WIDTH
    ):
        _fail("sshsig armor line wrapping is not canonical")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        _fail(f"sshsig armor payload is not valid base64: {error}")
    canonical_lines = [
        encoded[index : index + _ARMOR_WIDTH]
        for index in range(0, len(encoded), _ARMOR_WIDTH)
    ]
    canonical = (
        _BEGIN_SIGNATURE
        + "\n"
        + "\n".join(canonical_lines)
        + "\n"
        + _END_SIGNATURE
        + "\n"
    )
    if canonical != text:
        _fail("sshsig armor is not canonical")
    if not decoded.startswith(b"SSHSIG"):
        _fail("sshsig payload lacks the SSHSIG magic")
    offset = len(b"SSHSIG")
    if offset + 4 > len(decoded):
        _fail("sshsig payload lacks its version")
    version = struct.unpack_from(">I", decoded, offset)[0]
    offset += 4
    if version != 1:
        _fail("sshsig payload version must be 1")
    public_key, offset = _read_ssh_string(decoded, offset, context="sshsig payload")
    namespace_raw, offset = _read_ssh_string(decoded, offset, context="sshsig payload")
    reserved, offset = _read_ssh_string(decoded, offset, context="sshsig payload")
    hash_algorithm_raw, offset = _read_ssh_string(
        decoded,
        offset,
        context="sshsig payload",
    )
    signature_blob, offset = _read_ssh_string(
        decoded,
        offset,
        context="sshsig payload",
    )
    if offset != len(decoded):
        _fail("sshsig payload contains trailing bytes")
    _parse_ed25519_key_blob(public_key, context="sshsig public key")
    if public_key != authority.public_key_blob:
        _fail("sshsig public key differs from the anchored authority key")
    try:
        namespace = namespace_raw.decode("ascii")
        hash_algorithm = hash_algorithm_raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"sshsig namespace or hash algorithm is not ASCII: {error}")
    if namespace != SSHSIG_NAMESPACE:
        _fail("sshsig namespace differs from the exact protocol namespace")
    if reserved:
        _fail("sshsig reserved field must be empty")
    if hash_algorithm != "sha512":
        _fail("sshsig hash algorithm must be sha512")
    signature_algorithm_raw, signature_offset = _read_ssh_string(
        signature_blob,
        0,
        context="sshsig signature blob",
    )
    signature_value, signature_offset = _read_ssh_string(
        signature_blob,
        signature_offset,
        context="sshsig signature blob",
    )
    if (
        signature_algorithm_raw != b"ssh-ed25519"
        or len(signature_value) != 64
        or signature_offset != len(signature_blob)
    ):
        _fail("sshsig signature blob must contain one canonical Ed25519 signature")
    return _SshSignature(
        public_key_blob=public_key,
        namespace=namespace,
        hash_algorithm=hash_algorithm,
        signature_algorithm="ssh-ed25519",
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as error:
        if error.errno != errno.ESRCH:
            _fail(f"cannot terminate ssh-keygen process group: {error}")


def _run_bounded(
    executable: Path,
    arguments: Sequence[str],
    payload: bytes,
    *,
    inherited_fds: Sequence[int],
) -> tuple[int, bytes, bytes]:
    if len(payload) > MAX_PAYLOAD_BYTES:
        _fail("sshsig stdin payload exceeds its byte bound")
    argv = [str(executable), *arguments]
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=dict(EXACT_ENVIRONMENT),
            shell=False,
            close_fds=True,
            pass_fds=tuple(inherited_fds),
            start_new_session=True,
        )
    except OSError as error:
        _fail(f"cannot execute pinned ssh-keygen: {error}")
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _kill_process_group(process)
        process.wait()
        _fail("ssh-keygen pipes were not created")
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    written = 0
    deadline = time.monotonic() + TOOL_TIMEOUT_SECONDS
    for stream in (process.stdin, process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("pinned ssh-keygen verification timed out")
            events = selector.select(min(remaining, 0.1))
            if not events and process.poll() is not None:
                # Drain any final readable bytes before declaring an incomplete pipe.
                events = selector.select(0)
                if not events:
                    for key in list(selector.get_map().values()):
                        if key.data == "stdin":
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    if selector.get_map():
                        _fail("ssh-keygen exited before its output pipes reached EOF")
            for key, _ in events:
                stream = key.fileobj
                if key.data == "stdin":
                    if written == len(payload):
                        selector.unregister(stream)
                        stream.close()
                        continue
                    try:
                        count = os.write(stream.fileno(), payload[written:])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    if count <= 0:
                        _fail("ssh-keygen stdin write made no progress")
                    written += count
                    if written == len(payload):
                        selector.unregister(stream)
                        stream.close()
                else:
                    try:
                        chunk = os.read(stream.fileno(), READ_CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    target = stdout if key.data == "stdout" else stderr
                    limit = (
                        MAX_STDOUT_BYTES if key.data == "stdout" else MAX_STDERR_BYTES
                    )
                    target.extend(chunk)
                    if len(target) > limit:
                        _fail(f"ssh-keygen {key.data} exceeds its byte bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _fail("pinned ssh-keygen verification timed out")
        return_code = process.wait(timeout=remaining)
        if written != len(payload):
            _fail("ssh-keygen did not consume the complete bounded stdin payload")
        return return_code, bytes(stdout), bytes(stderr)
    finally:
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        _kill_process_group(process)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as error:
            _fail(f"ssh-keygen process group did not terminate: {error}")


def _canonical_new_directory(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve {context} parent: {error}")
    if parent != absolute.parent:
        _fail(f"{context} parent must be an absolute canonical directory")
    if absolute.exists() or absolute.is_symlink():
        _fail(f"{context} destination already exists")
    return absolute


def _pin_root(path: Path, *, context: str) -> _PinnedRoot:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open {context}: {error}")
    entry = os.fstat(descriptor)
    if not stat.S_ISDIR(entry.st_mode):
        os.close(descriptor)
        _fail(f"{context} must be a directory")
    return _PinnedRoot(
        path=path,
        descriptor=descriptor,
        device=entry.st_dev,
        inode=entry.st_ino,
        mode=entry.st_mode,
    )


def _revalidate_root(root: _PinnedRoot, *, context: str) -> None:
    try:
        entry = os.fstat(root.descriptor)
    except OSError as error:
        _fail(f"cannot revalidate {context}: {error}")
    if (
        not stat.S_ISDIR(entry.st_mode)
        or entry.st_dev != root.device
        or entry.st_ino != root.inode
        or stat.S_IMODE(entry.st_mode) != stat.S_IMODE(root.mode)
    ):
        _fail(f"{context} identity or mode changed")
    try:
        named = root.path.stat(follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot revalidate named {context}: {error}")
    if (
        not stat.S_ISDIR(named.st_mode)
        or (named.st_dev, named.st_ino) != (root.device, root.inode)
        or stat.S_IMODE(named.st_mode) != stat.S_IMODE(root.mode)
    ):
        _fail(f"named {context} identity or mode changed")


def _reserve_stage(
    destination: Path,
    *,
    context: str,
) -> tuple[Path, _PinnedRoot, _PinnedRoot, str]:
    absolute = _canonical_new_directory(destination, context=context)
    parent = _pin_root(absolute.parent, context=f"{context} parent")
    stage_name = f".{absolute.name}.private-candidate-{uuid.uuid4().hex}"
    reserved = False
    stage: _PinnedRoot | None = None
    try:
        os.mkdir(stage_name, 0o700, dir_fd=parent.descriptor)
        reserved = True
        stage = _pin_root(
            absolute.parent / stage_name,
            context=f"{context} candidate",
        )
        _revalidate_root(parent, context=f"{context} parent")
        _revalidate_root(stage, context=f"{context} candidate")
    except OSError as error:
        message = f"cannot reserve {context} private candidate: {error}"
    except VisualQaVerificationError as error:
        message = str(error)
    else:
        return absolute, parent, stage, stage_name
    if stage is not None:
        stage.close()
    parent.close()
    if reserved:
        message = (
            f"{message}; candidate_path={absolute.parent / stage_name}; "
            "candidate_state=reserved-private-candidate-do-not-auto-delete"
        )
    raise VisualQaVerificationError(message)


def _write_member(root: _PinnedRoot, member: str, raw: bytes) -> dict[str, object]:
    if member not in OUTPUT_MEMBERS:
        _fail(f"output member {member!r} is outside the exact inventory")
    if len(raw) > MAX_OUTPUT_MEMBER_BYTES:
        _fail(f"output member {member} exceeds its byte bound")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(member, flags, 0o400, dir_fd=root.descriptor)
    except OSError as error:
        _fail(f"cannot create output member {member}: {error}")
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                _fail(f"write made no progress for output member {member}")
            offset += count
        os.fsync(descriptor)
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o400
            or entry.st_size != len(raw)
        ):
            _fail(f"output member {member} has invalid identity")
    finally:
        os.close(descriptor)
    return {"member": member, "bytes": len(raw), "sha256": _sha256(raw)}


def _open_member(root: _PinnedRoot, member: str, *, maximum: int) -> tuple[int, bytes]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(member, flags, dir_fd=root.descriptor)
    except OSError as error:
        _fail(f"cannot open output member {member}: {error}")
    try:
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) != 0o400
            or entry.st_size < 0
            or entry.st_size > maximum
        ):
            _fail(f"output member {member} has invalid identity or size")
        raw = bytearray()
        offset = 0
        while offset < entry.st_size:
            chunk = os.pread(
                descriptor,
                min(READ_CHUNK_BYTES, entry.st_size - offset),
                offset,
            )
            if not chunk:
                _fail(f"output member {member} ended early")
            raw.extend(chunk)
            offset += len(chunk)
        if _stat_identity(os.fstat(descriptor)) != _stat_identity(entry):
            _fail(f"output member {member} changed while read")
        return descriptor, bytes(raw)
    except BaseException:
        os.close(descriptor)
        raise


def _pin_tree_member(
    root: _PinnedRoot,
    member: str,
    *,
    maximum: int,
) -> _PinnedMember:
    descriptor, raw = _open_member(root, member, maximum=maximum)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(member, dir_fd=root.descriptor, follow_symlinks=False)
        if _stat_identity(opened) != _stat_identity(named):
            _fail(f"output member {member} named identity differs from its descriptor")
        return _PinnedMember(
            member=member,
            descriptor=descriptor,
            identity=_stat_identity(opened),
            raw=raw,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_tree_member(root: _PinnedRoot, pin: _PinnedMember) -> None:
    try:
        opened = os.fstat(pin.descriptor)
        named = os.stat(pin.member, dir_fd=root.descriptor, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot revalidate output member {pin.member}: {error}")
    if _stat_identity(opened) != pin.identity or _stat_identity(named) != pin.identity:
        _fail(f"output member {pin.member} identity changed during validation")
    digest = hashlib.sha256()
    offset = 0
    while offset < len(pin.raw):
        chunk = os.pread(
            pin.descriptor,
            min(READ_CHUNK_BYTES, len(pin.raw) - offset),
            offset,
        )
        if not chunk:
            _fail(f"output member {pin.member} ended during revalidation")
        digest.update(chunk)
        offset += len(chunk)
    if digest.hexdigest() != _sha256(pin.raw):
        _fail(f"output member {pin.member} bytes changed during validation")
    if _stat_identity(os.fstat(pin.descriptor)) != pin.identity:
        _fail(f"output member {pin.member} changed while it was rehashed")


def _walk_tree(root: _PinnedRoot, *, directory_mode: int) -> list[str]:
    _revalidate_root(root, context="verification tree")
    if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != directory_mode:
        _fail(f"verification tree root must have mode {directory_mode:04o}")
    try:
        entries = list(os.scandir(root.descriptor))
    except OSError as error:
        _fail(f"cannot scan verification tree: {error}")
    if len(entries) > MAX_OUTPUT_FILES:
        _fail("verification tree exceeds the file-count bound")
    members: list[str] = []
    total = 0
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o400
        ):
            _fail(f"verification tree member {entry.name} has invalid identity")
        total += info.st_size
        if total > MAX_OUTPUT_BYTES:
            _fail("verification tree exceeds the aggregate byte bound")
        members.append(entry.name)
    if sorted(members) != sorted(OUTPUT_MEMBERS):
        _fail("verification tree has the wrong exact member inventory")
    return sorted(members)


def _seal_tree(root: _PinnedRoot) -> None:
    if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != 0o700:
        _fail("private verification candidate must begin mode 0700")
    os.fchmod(root.descriptor, 0o500)
    os.fsync(root.descriptor)
    root.mode = (root.mode & ~0o7777) | 0o500
    _walk_tree(root, directory_mode=0o500)


def _rename_no_replace(source: str, destination: str, parent_descriptor: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            function = library.renameatx_np
        except AttributeError:
            _fail("platform lacks renameatx_np atomic no-replace publication")
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        flag = 0x00000004
    elif sys.platform.startswith("linux"):
        try:
            function = library.renameat2
        except AttributeError:
            _fail("platform lacks renameat2 atomic no-replace publication")
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        flag = 1
    else:
        _fail("platform lacks a supported atomic no-replace directory rename")
    function.restype = ctypes.c_int
    result = function(
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(destination),
        flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        _fail("verification destination already exists")
    _fail(
        "atomic no-replace verification publication failed: "
        f"{os.strerror(error_number)} (errno {error_number})",
    )


def _file_record(member: str, raw: bytes) -> dict[str, object]:
    return {"member": member, "bytes": len(raw), "sha256": _sha256(raw)}


def _validate_file_record(
    value: object,
    *,
    member: str,
    raw: bytes,
    context: str,
) -> None:
    record = _expect_mapping(value, context=context)
    _expect_keys(record, {"member", "bytes", "sha256"}, context=context)
    if record != _file_record(member, raw):
        _fail(f"{context} does not bind the exact output member")


def _expected_success_stdout(authority: _Authority) -> bytes:
    return (
        f'Good "{SSHSIG_NAMESPACE}" signature for {authority.principal} with '
        f"ED25519 key {authority.fingerprint}\n"
    ).encode("ascii")


def _produce_into(config: VisualQaVerificationInputs, root: _PinnedRoot) -> _Production:
    receipt_pin: _PinnedFile | None = None
    authority_pin: _PinnedFile | None = None
    signature_pin: _PinnedFile | None = None
    tool_pin: _PinnedFile | None = None
    authority_fd = -1
    signature_fd = -1
    try:
        receipt_pin, receipt_raw = _pin_file(
            config.receipt_path,
            expected_sha256=config.expected_receipt_sha256,
            maximum=MAX_RECEIPT_BYTES,
            context="visual-QA receipt",
        )
        authority_pin, authority_raw = _pin_file(
            config.allowed_signers_path,
            expected_sha256=config.expected_allowed_signers_sha256,
            maximum=MAX_AUTHORITY_BYTES,
            context="allowed-signers authority",
        )
        signature_pin, signature_raw = _pin_file(
            config.signature_path,
            expected_sha256=config.expected_signature_sha256,
            maximum=MAX_SIGNATURE_BYTES,
            context="sshsig signature",
        )
        tool_pin, _ = _pin_file(
            config.ssh_keygen_path,
            expected_sha256=config.expected_ssh_keygen_sha256,
            maximum=MAX_TOOL_BYTES,
            context="ssh-keygen executable",
            executable=True,
        )
        normalized, page_count = _normalize_visual_receipt(receipt_raw)
        principal = _expect_principal(
            config.expected_principal,
            context="expected principal",
        )
        if normalized["reviewer_principal"] != principal:
            _fail("visual-QA receipt principal differs from the expected principal")
        authority = _parse_authority(authority_raw, expected_principal=principal)
        signature = _parse_signature(signature_raw, authority=authority)
        payload = DOMAIN_SEPARATOR + receipt_raw
        if len(payload) > MAX_PAYLOAD_BYTES:
            _fail("domain-separated canonical payload exceeds its byte bound")

        input_members = [
            _write_member(root, RECEIPT_MEMBER, receipt_raw),
            _write_member(root, AUTHORITY_MEMBER, authority_raw),
            _write_member(root, SIGNATURE_MEMBER, signature_raw),
            _write_member(root, PAYLOAD_MEMBER, payload),
        ]
        authority_fd, authority_copy = _open_member(
            root,
            AUTHORITY_MEMBER,
            maximum=MAX_AUTHORITY_BYTES,
        )
        signature_fd, signature_copy = _open_member(
            root,
            SIGNATURE_MEMBER,
            maximum=MAX_SIGNATURE_BYTES,
        )
        if authority_copy != authority_raw or signature_copy != signature_raw:
            _fail("descriptor-pinned sshsig inputs differ from anchored bytes")
        for pin, context in (
            (receipt_pin, "visual-QA receipt"),
            (authority_pin, "allowed-signers authority"),
            (signature_pin, "sshsig signature"),
            (tool_pin, "ssh-keygen executable"),
        ):
            _revalidate_pin(pin, context=context)
        arguments = [
            "-Y",
            "verify",
            "-f",
            f"/dev/fd/{authority_fd}",
            "-I",
            principal,
            "-n",
            SSHSIG_NAMESPACE,
            "-s",
            f"/dev/fd/{signature_fd}",
        ]
        return_code, stdout, stderr = _run_bounded(
            tool_pin.path,
            arguments,
            payload,
            inherited_fds=(authority_fd, signature_fd),
        )
        for pin, context in (
            (receipt_pin, "visual-QA receipt"),
            (authority_pin, "allowed-signers authority"),
            (signature_pin, "sshsig signature"),
            (tool_pin, "ssh-keygen executable"),
        ):
            _revalidate_pin(pin, context=context)
        if return_code != 0:
            _fail(
                "pinned ssh-keygen rejected the signed receipt with status "
                f"{return_code}",
            )
        if stderr:
            _fail("pinned ssh-keygen emitted unexpected stderr on success")
        if stdout != _expected_success_stdout(authority):
            _fail("pinned ssh-keygen success output is not the exact expected record")
        stdout_record = _write_member(root, STDOUT_MEMBER, stdout)
        stderr_record = _write_member(root, STDERR_MEMBER, stderr)
        unsigned: dict[str, object] = {
            "schema": VISUAL_QA_VERIFICATION_SCHEMA,
            "contract": VISUAL_QA_VERIFICATION_CONTRACT,
            "release_id": normalized["release_id"],
            "receipt": {
                **_file_record(RECEIPT_MEMBER, receipt_raw),
                "schema": VISUAL_QA_RECEIPT_SCHEMA,
                "review_id": normalized["review_id"],
                "reviewed_at_utc": normalized["reviewed_at_utc"],
                "page_count": page_count,
                "derivation_manifest_sha256": normalized["derivation_manifest_sha256"],
                "machine_manifest_sha256": normalized["machine_manifest_sha256"],
                "rebuttal_renderer_manifest_sha256": normalized[
                    "rebuttal_renderer_manifest_sha256"
                ],
                "pdf_set_sha256": normalized["pdf_set_sha256"],
                "render_set_sha256": normalized["render_set_sha256"],
            },
            "authority": {
                **_file_record(AUTHORITY_MEMBER, authority_raw),
                "anchor_kind": "independently-supplied-sha256",
                "principal": principal,
                "namespace": SSHSIG_NAMESPACE,
                "key_algorithm": "ssh-ed25519",
                "public_key_blob_sha256": _sha256(authority.public_key_blob),
                "key_fingerprint": authority.fingerprint,
            },
            "signature": {
                **_file_record(SIGNATURE_MEMBER, signature_raw),
                "format": "openssh-sshsig-v1",
                "namespace": signature.namespace,
                "hash_algorithm": signature.hash_algorithm,
                "signature_algorithm": signature.signature_algorithm,
                "public_key_blob_sha256": _sha256(signature.public_key_blob),
            },
            "payload": {
                **_file_record(PAYLOAD_MEMBER, payload),
                "construction": "fixed-domain-separator-plus-canonical-v2-receipt",
                "domain_separator_ascii": DOMAIN_SEPARATOR.decode("ascii"),
            },
            "tool": {
                "absolute_path": str(tool_pin.path),
                "bytes": tool_pin.size,
                "sha256": tool_pin.sha256,
                "owner_uid": tool_pin.uid,
                "mode": format(stat.S_IMODE(tool_pin.mode), "04o"),
            },
            "invocation": {
                "argv": [
                    str(tool_pin.path),
                    "-Y",
                    "verify",
                    "-f",
                    "{descriptor-pinned-allowed-signers}",
                    "-I",
                    principal,
                    "-n",
                    SSHSIG_NAMESPACE,
                    "-s",
                    "{descriptor-pinned-signature}",
                ],
                "cwd": "/",
                "shell": False,
                "environment": EXACT_ENVIRONMENT,
                "process_containment": {
                    "new_session": True,
                    "same_process_group_kill_on_exit": True,
                    "detached_descendants": "not-contained",
                },
                "timeout_seconds": int(TOOL_TIMEOUT_SECONDS),
                "stdin_limit_bytes": MAX_PAYLOAD_BYTES,
                "stdout_limit_bytes": MAX_STDOUT_BYTES,
                "stderr_limit_bytes": MAX_STDERR_BYTES,
                "return_code": 0,
                "stdout": stdout_record,
                "stderr": stderr_record,
            },
            "authentication": {
                "status": "verified",
                "claim": "valid-ed25519-signature-under-the-anchored-public-key",
                "principal": principal,
                "namespace": SSHSIG_NAMESPACE,
                "key_fingerprint": authority.fingerprint,
            },
            "promotion": {
                "promotable": False,
                "authority": "none-this-is-an-isolated-authentication-seam",
            },
            "non_inference_limits": NON_INFERENCE_LIMITS,
            "input_members": sorted(
                input_members,
                key=lambda item: str(item["member"]),
            ),
            "payload_sha256": _sha256(payload),
        }
        unsigned["manifest_body_sha256"] = _sha256(_canonical_json(unsigned))
        manifest_raw = _canonical_json(unsigned)
        _write_member(root, MANIFEST_MEMBER, manifest_raw)
        return _Production(
            manifest=unsigned,
            manifest_raw=manifest_raw,
            receipt_sha256=receipt_pin.sha256,
            authority_sha256=authority_pin.sha256,
            signature_sha256=signature_pin.sha256,
            payload_sha256=_sha256(payload),
            fingerprint=authority.fingerprint,
            release_id=str(normalized["release_id"]),
            page_count=page_count,
        )
    finally:
        if authority_fd >= 0:
            os.close(authority_fd)
        if signature_fd >= 0:
            os.close(signature_fd)
        for pin in (receipt_pin, authority_pin, signature_pin, tool_pin):
            if pin is not None:
                pin.close()


def _read_member(root: _PinnedRoot, member: str, *, maximum: int) -> bytes:
    descriptor, raw = _open_member(root, member, maximum=maximum)
    os.close(descriptor)
    return raw


def _validate_tree_payload(
    raw_by_member: Mapping[str, bytes],
    manifest_raw: bytes,
) -> dict[str, object]:
    if raw_by_member[MANIFEST_MEMBER] != manifest_raw:
        _fail("verification manifest changed")
    parsed = _expect_mapping(
        _json_no_duplicates(manifest_raw, context="verification manifest"),
        context="verification manifest",
    )
    if _canonical_json(parsed) != manifest_raw:
        _fail("verification manifest must use canonical JSON encoding")
    _expect_keys(
        parsed,
        {
            "schema",
            "contract",
            "release_id",
            "receipt",
            "authority",
            "signature",
            "payload",
            "tool",
            "invocation",
            "authentication",
            "promotion",
            "non_inference_limits",
            "input_members",
            "payload_sha256",
            "manifest_body_sha256",
        },
        context="verification manifest",
    )
    if parsed["schema"] != VISUAL_QA_VERIFICATION_SCHEMA:
        _fail("verification manifest has the wrong schema")
    if parsed["contract"] != VISUAL_QA_VERIFICATION_CONTRACT:
        _fail("verification manifest has the wrong contract")
    signed_payload_sha = _expect_sha256(
        parsed["payload_sha256"],
        context="signed payload SHA-256",
    )
    manifest_body_sha = parsed["manifest_body_sha256"]
    _expect_sha256(
        manifest_body_sha,
        context="verification manifest body SHA-256",
    )
    unsigned = dict(parsed)
    del unsigned["manifest_body_sha256"]
    if _sha256(_canonical_json(unsigned)) != manifest_body_sha:
        _fail("verification manifest body digest is invalid")

    receipt_raw = raw_by_member[RECEIPT_MEMBER]
    authority_raw = raw_by_member[AUTHORITY_MEMBER]
    signature_raw = raw_by_member[SIGNATURE_MEMBER]
    signed_payload = raw_by_member[PAYLOAD_MEMBER]
    stdout = raw_by_member[STDOUT_MEMBER]
    stderr = raw_by_member[STDERR_MEMBER]
    normalized, page_count = _normalize_visual_receipt(receipt_raw)
    principal = str(normalized["reviewer_principal"])
    authority = _parse_authority(authority_raw, expected_principal=principal)
    signature = _parse_signature(signature_raw, authority=authority)
    if signed_payload != DOMAIN_SEPARATOR + receipt_raw:
        _fail("published signed payload does not bind the exact canonical receipt")
    if _sha256(signed_payload) != signed_payload_sha:
        _fail("verification manifest signed-payload digest is invalid")
    if stdout != _expected_success_stdout(authority) or stderr:
        _fail("published ssh-keygen output does not record one exact success")
    if parsed["release_id"] != normalized["release_id"]:
        _fail("verification manifest release id differs from the signed receipt")
    if parsed["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail("verification manifest non-inference limits drifted")
    if parsed["promotion"] != {
        "promotable": False,
        "authority": "none-this-is-an-isolated-authentication-seam",
    }:
        _fail("verification manifest promotion boundary drifted")

    receipt_record = _expect_mapping(parsed["receipt"], context="manifest receipt")
    _expect_keys(
        receipt_record,
        {
            "member",
            "bytes",
            "sha256",
            "schema",
            "review_id",
            "reviewed_at_utc",
            "page_count",
            "derivation_manifest_sha256",
            "machine_manifest_sha256",
            "rebuttal_renderer_manifest_sha256",
            "pdf_set_sha256",
            "render_set_sha256",
        },
        context="manifest receipt",
    )
    expected_receipt_record = {
        **_file_record(RECEIPT_MEMBER, receipt_raw),
        "schema": VISUAL_QA_RECEIPT_SCHEMA,
        "review_id": normalized["review_id"],
        "reviewed_at_utc": normalized["reviewed_at_utc"],
        "page_count": page_count,
        "derivation_manifest_sha256": normalized["derivation_manifest_sha256"],
        "machine_manifest_sha256": normalized["machine_manifest_sha256"],
        "rebuttal_renderer_manifest_sha256": normalized[
            "rebuttal_renderer_manifest_sha256"
        ],
        "pdf_set_sha256": normalized["pdf_set_sha256"],
        "render_set_sha256": normalized["render_set_sha256"],
    }
    if receipt_record != expected_receipt_record:
        _fail("verification manifest receipt record drifted")

    authority_record = _expect_mapping(
        parsed["authority"],
        context="manifest authority",
    )
    expected_authority_record = {
        **_file_record(AUTHORITY_MEMBER, authority_raw),
        "anchor_kind": "independently-supplied-sha256",
        "principal": principal,
        "namespace": SSHSIG_NAMESPACE,
        "key_algorithm": "ssh-ed25519",
        "public_key_blob_sha256": _sha256(authority.public_key_blob),
        "key_fingerprint": authority.fingerprint,
    }
    if authority_record != expected_authority_record:
        _fail("verification manifest authority record drifted")

    signature_record = _expect_mapping(
        parsed["signature"],
        context="manifest signature",
    )
    expected_signature_record = {
        **_file_record(SIGNATURE_MEMBER, signature_raw),
        "format": "openssh-sshsig-v1",
        "namespace": signature.namespace,
        "hash_algorithm": signature.hash_algorithm,
        "signature_algorithm": signature.signature_algorithm,
        "public_key_blob_sha256": _sha256(signature.public_key_blob),
    }
    if signature_record != expected_signature_record:
        _fail("verification manifest signature record drifted")

    payload_record = _expect_mapping(parsed["payload"], context="manifest payload")
    if payload_record != {
        **_file_record(PAYLOAD_MEMBER, signed_payload),
        "construction": "fixed-domain-separator-plus-canonical-v2-receipt",
        "domain_separator_ascii": DOMAIN_SEPARATOR.decode("ascii"),
    }:
        _fail("verification manifest signed-payload record drifted")
    input_members = _expect_sequence(
        parsed["input_members"],
        context="manifest input members",
    )
    expected_input_members = sorted(
        [
            _file_record(RECEIPT_MEMBER, receipt_raw),
            _file_record(AUTHORITY_MEMBER, authority_raw),
            _file_record(SIGNATURE_MEMBER, signature_raw),
            _file_record(PAYLOAD_MEMBER, signed_payload),
        ],
        key=lambda item: str(item["member"]),
    )
    if input_members != expected_input_members:
        _fail("verification manifest input-member inventory drifted")

    authentication = _expect_mapping(
        parsed["authentication"],
        context="manifest authentication",
    )
    if authentication != {
        "status": "verified",
        "claim": "valid-ed25519-signature-under-the-anchored-public-key",
        "principal": principal,
        "namespace": SSHSIG_NAMESPACE,
        "key_fingerprint": authority.fingerprint,
    }:
        _fail("verification manifest authentication record drifted")
    tool = _expect_mapping(parsed["tool"], context="manifest tool")
    _expect_keys(
        tool,
        {"absolute_path", "bytes", "sha256", "owner_uid", "mode"},
        context="manifest tool",
    )
    tool_path_text = _expect_string(
        tool["absolute_path"],
        context="manifest tool absolute path",
    )
    tool_path = Path(tool_path_text)
    if (
        not tool_path.is_absolute()
        or ".." in tool_path.parts
        or str(tool_path) != tool_path_text
    ):
        _fail("verification manifest tool path is not canonical absolute syntax")
    _expect_positive_int(
        tool["bytes"],
        maximum=MAX_TOOL_BYTES,
        context="manifest tool bytes",
    )
    _expect_sha256(tool["sha256"], context="manifest tool SHA-256")
    if tool["owner_uid"] != 0:
        _fail("verification manifest tool owner must be root")
    tool_mode = _expect_string(tool["mode"], context="manifest tool mode")
    if re.fullmatch(r"[0-7]{4}", tool_mode) is None:
        _fail("verification manifest tool mode is not canonical octal")
    mode_value = int(tool_mode, 8)
    if mode_value & 0o7022 or not mode_value & 0o111:
        _fail(
            "verification manifest tool mode is privileged, writable, or "
            "non-executable",
        )
    invocation = _expect_mapping(parsed["invocation"], context="manifest invocation")
    _expect_keys(
        invocation,
        {
            "argv",
            "cwd",
            "shell",
            "environment",
            "process_containment",
            "timeout_seconds",
            "stdin_limit_bytes",
            "stdout_limit_bytes",
            "stderr_limit_bytes",
            "return_code",
            "stdout",
            "stderr",
        },
        context="manifest invocation",
    )
    stdout_record = _expect_mapping(
        invocation.get("stdout"),
        context="manifest invocation stdout",
    )
    stderr_record = _expect_mapping(
        invocation.get("stderr"),
        context="manifest invocation stderr",
    )
    _validate_file_record(
        stdout_record,
        member=STDOUT_MEMBER,
        raw=stdout,
        context="manifest invocation stdout",
    )
    _validate_file_record(
        stderr_record,
        member=STDERR_MEMBER,
        raw=stderr,
        context="manifest invocation stderr",
    )
    expected_argv = [
        tool_path_text,
        "-Y",
        "verify",
        "-f",
        "{descriptor-pinned-allowed-signers}",
        "-I",
        principal,
        "-n",
        SSHSIG_NAMESPACE,
        "-s",
        "{descriptor-pinned-signature}",
    ]
    if (
        invocation["argv"] != expected_argv
        or invocation["cwd"] != "/"
        or invocation["shell"] is not False
        or invocation["environment"] != EXACT_ENVIRONMENT
        or invocation["process_containment"]
        != {
            "new_session": True,
            "same_process_group_kill_on_exit": True,
            "detached_descendants": "not-contained",
        }
        or invocation["timeout_seconds"] != int(TOOL_TIMEOUT_SECONDS)
        or invocation["stdin_limit_bytes"] != MAX_PAYLOAD_BYTES
        or invocation["stdout_limit_bytes"] != MAX_STDOUT_BYTES
        or invocation["stderr_limit_bytes"] != MAX_STDERR_BYTES
        or invocation["return_code"] != 0
    ):
        _fail("verification manifest invocation policy drifted")
    return dict(parsed)


def _pin_validated_tree(
    root: _PinnedRoot,
    manifest_raw: bytes,
) -> tuple[dict[str, object], list[_PinnedMember]]:
    members = _walk_tree(root, directory_mode=0o500)
    if members != sorted(OUTPUT_MEMBERS):
        _fail("verification tree member inventory drifted")
    pins: list[_PinnedMember] = []
    try:
        for member in OUTPUT_MEMBERS:
            maximum = (
                MAX_RECEIPT_BYTES
                if member in {RECEIPT_MEMBER, MANIFEST_MEMBER}
                else MAX_OUTPUT_MEMBER_BYTES
            )
            pins.append(_pin_tree_member(root, member, maximum=maximum))
        raw_by_member = {pin.member: pin.raw for pin in pins}
        result = _validate_tree_payload(raw_by_member, manifest_raw)
        _revalidate_root(root, context="verification tree")
        for pin in pins:
            _revalidate_tree_member(root, pin)
        if _walk_tree(root, directory_mode=0o500) != members:
            _fail("verification tree inventory changed during validation")
        _revalidate_root(root, context="verification tree")
        for pin in pins:
            _revalidate_tree_member(root, pin)
    except BaseException:
        for pin in pins:
            pin.close()
        raise
    else:
        return result, pins


def _validate_tree(root: _PinnedRoot, manifest_raw: bytes) -> dict[str, object]:
    result, pins = _pin_validated_tree(root, manifest_raw)
    try:
        return result
    finally:
        for pin in pins:
            pin.close()


def _publish(
    destination: Path,
    parent: _PinnedRoot,
    stage: _PinnedRoot,
    stage_name: str,
    production: _Production,
) -> None:
    renamed = False
    try:
        _revalidate_root(parent, context="verification destination parent")
        _revalidate_root(stage, context="sealed verification candidate")
        _seal_tree(stage)
        _validate_tree(stage, production.manifest_raw)
        _revalidate_root(parent, context="verification destination parent")
        _rename_no_replace(stage_name, destination.name, parent.descriptor)
        renamed = True
        stage.path = destination
        os.fsync(parent.descriptor)
        _revalidate_root(parent, context="verification destination parent")
        _revalidate_root(stage, context="published verification destination")
        try:
            resolved = destination.resolve(strict=True)
        except OSError as error:
            _fail(f"cannot resolve published verification destination: {error}")
        if resolved != destination:
            _fail("published verification destination is no longer canonical")
        named = os.stat(
            destination.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != (stage.device, stage.inode)
            or stat.S_IMODE(named.st_mode) != 0o500
        ):
            _fail("published verification destination identity changed")
        _validate_tree(stage, production.manifest_raw)
    except BaseException as error:
        if isinstance(error, VisualQaVerificationError) and "candidate_paths=" in str(
            error,
        ):
            raise
        state = (
            "published-destination-may-exist-do-not-auto-delete"
            if renamed
            else "private-candidate-may-exist-do-not-auto-delete"
        )
        message = (
            f"{error}; candidate_paths={stage.path}|{destination}; "
            f"candidate_state={state}"
        )
        raise VisualQaVerificationError(message) from error


def _receipt(
    root: Path,
    production: _Production,
    *,
    replay_root: Path | None,
) -> VisualQaVerificationReceipt:
    return VisualQaVerificationReceipt(
        manifest_path=str(root / MANIFEST_MEMBER),
        manifest_sha256=_sha256(production.manifest_raw),
        visual_qa_receipt_sha256=production.receipt_sha256,
        allowed_signers_sha256=production.authority_sha256,
        signature_sha256=production.signature_sha256,
        payload_sha256=production.payload_sha256,
        key_fingerprint=production.fingerprint,
        release_id=production.release_id,
        page_count=production.page_count,
        authentication_status="verified",
        promotable=False,
        replay_root=None if replay_root is None else str(replay_root),
    )


def build_visual_qa_verification(
    config: VisualQaVerificationInputs,
    destination: Path,
) -> VisualQaVerificationReceipt:
    """Verify and atomically publish one sealed authentication record."""
    absolute, parent, stage, stage_name = _reserve_stage(
        destination,
        context="visual-QA verification",
    )
    try:
        try:
            production = _produce_into(config, stage)
        except BaseException as error:
            message = (
                f"{error}; candidate_path={stage.path}; "
                "candidate_state=partial-private-candidate-do-not-auto-delete"
            )
            raise VisualQaVerificationError(message) from error
        _publish(absolute, parent, stage, stage_name, production)
        return _receipt(absolute, production, replay_root=None)
    finally:
        stage.close()
        parent.close()


def _read_anchored_verification(
    verification_root: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Path, bytes, dict[str, object], _PinnedRoot, list[_PinnedMember]]:
    expected = _expect_sha256(
        expected_manifest_sha256,
        context="expected verification manifest SHA-256",
    )
    absolute = verification_root.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve verification root: {error}")
    if resolved != absolute:
        _fail("verification root must be an absolute canonical directory")
    root = _pin_root(absolute, context="verification root")
    try:
        if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != 0o500:
            _fail("verification root must be sealed mode 0500")
        raw = _read_member(root, MANIFEST_MEMBER, maximum=MAX_RECEIPT_BYTES)
        if _sha256(raw) != expected:
            _fail("verification manifest differs from its independent SHA-256 anchor")
        manifest, pins = _pin_validated_tree(root, raw)
    except BaseException:
        root.close()
        raise
    else:
        return absolute, raw, manifest, root, pins


def validate_visual_qa_verification(
    config: VisualQaVerificationInputs,
    verification_root: Path,
    replay_root: Path,
    *,
    expected_manifest_sha256: str,
) -> VisualQaVerificationReceipt:
    """Re-run authentication into a distinct, retained, sealed replay tree."""
    (
        original_root,
        original_raw,
        original,
        original_pin,
        original_members,
    ) = _read_anchored_verification(
        verification_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    replay_absolute = replay_root.absolute()
    parent: _PinnedRoot | None = None
    stage: _PinnedRoot | None = None
    published = False
    try:
        if (
            replay_absolute == original_root
            or replay_absolute in original_root.parents
            or original_root in replay_absolute.parents
        ):
            _fail("verification root and replay root must be disjoint")
        receipt_record = _expect_mapping(
            original["receipt"],
            context="anchored verification receipt",
        )
        if receipt_record["sha256"] != config.expected_receipt_sha256:
            _fail("anchored verification manifest does not bind the replay receipt")
        absolute, parent, stage, stage_name = _reserve_stage(
            replay_absolute,
            context="visual-QA verification replay",
        )
        _revalidate_root(original_pin, context="anchored verification root")
        try:
            production = _produce_into(config, stage)
        except BaseException as error:
            message = (
                f"{error}; replay_candidate_path={stage.path}; "
                "candidate_state=partial-private-replay-do-not-auto-delete"
            )
            raise VisualQaVerificationError(message) from error
        _revalidate_root(original_pin, context="anchored verification root")
        for pin in original_members:
            _revalidate_tree_member(original_pin, pin)
        if production.manifest_raw != original_raw:
            _fail("independent visual-QA verification replay manifest differs")
        original_by_member = {pin.member: pin.raw for pin in original_members}
        for member in OUTPUT_MEMBERS:
            maximum = (
                MAX_RECEIPT_BYTES
                if member == MANIFEST_MEMBER
                else MAX_OUTPUT_MEMBER_BYTES
            )
            if original_by_member[member] != _read_member(
                stage,
                member,
                maximum=maximum,
            ):
                _fail(f"private verification replay member {member} differs")
        _revalidate_root(original_pin, context="anchored verification root")
        for pin in original_members:
            _revalidate_tree_member(original_pin, pin)
        _publish(absolute, parent, stage, stage_name, production)
        published = True
        replay_pin = _pin_root(absolute, context="published verification replay")
        try:
            _revalidate_root(original_pin, context="anchored verification root")
            for pin in original_members:
                _revalidate_tree_member(original_pin, pin)
            for member in OUTPUT_MEMBERS:
                maximum = (
                    MAX_RECEIPT_BYTES
                    if member == MANIFEST_MEMBER
                    else MAX_OUTPUT_MEMBER_BYTES
                )
                if original_by_member[member] != _read_member(
                    replay_pin,
                    member,
                    maximum=maximum,
                ):
                    _fail(f"retained verification replay member {member} differs")
        finally:
            replay_pin.close()
        return _receipt(original_root, production, replay_root=absolute)
    except BaseException as error:
        if isinstance(error, VisualQaVerificationError) and any(
            marker in str(error)
            for marker in ("replay_candidate_path=", "candidate_paths=")
        ):
            raise
        candidate = (
            replay_absolute if published else (stage.path if stage else replay_absolute)
        )
        state = (
            "published-replay-may-exist-do-not-auto-delete"
            if published
            else "partial-private-replay-do-not-auto-delete"
        )
        message = f"{error}; replay_candidate_path={candidate}; candidate_state={state}"
        raise VisualQaVerificationError(message) from error
    finally:
        if stage is not None:
            stage.close()
        if parent is not None:
            parent.close()
        for pin in original_members:
            pin.close()
        original_pin.close()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--ssh-keygen", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-allowed-signers-sha256", required=True)
    parser.add_argument("--expected-signature-sha256", required=True)
    parser.add_argument("--expected-ssh-keygen-sha256", required=True)
    parser.add_argument("--expected-principal", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build",
        help="verify and publish a sealed signed visual-QA record",
    )
    _add_common_arguments(build)
    build.add_argument("--destination", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate",
        help="independently replay a sealed signed visual-QA record",
    )
    _add_common_arguments(validate)
    validate.add_argument("--verification-root", type=Path, required=True)
    validate.add_argument("--replay-root", type=Path, required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    return parser


def _cli_config(arguments: argparse.Namespace) -> VisualQaVerificationInputs:
    return VisualQaVerificationInputs(
        receipt_path=arguments.receipt.absolute(),
        allowed_signers_path=arguments.allowed_signers.absolute(),
        signature_path=arguments.signature.absolute(),
        ssh_keygen_path=arguments.ssh_keygen.absolute(),
        expected_receipt_sha256=arguments.expected_receipt_sha256,
        expected_allowed_signers_sha256=arguments.expected_allowed_signers_sha256,
        expected_signature_sha256=arguments.expected_signature_sha256,
        expected_ssh_keygen_sha256=arguments.expected_ssh_keygen_sha256,
        expected_principal=arguments.expected_principal,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run an explicit build or retained independent replay."""
    arguments = _parser().parse_args(argv)
    config = _cli_config(arguments)
    if arguments.command == "build":
        receipt = build_visual_qa_verification(config, arguments.destination)
    else:
        receipt = validate_visual_qa_verification(
            config,
            arguments.verification_root,
            arguments.replay_root,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
        )
    print(_canonical_json(asdict(receipt)).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

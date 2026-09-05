"""Render and replay one deterministic response-to-reviewers PDF.

The public entry points consume only caller-SHA-anchored Markdown, JSON layout
and document configuration, two TrueType fonts, one ReportLab package tree, and
four native executables.  Every child receives authority through inherited file
descriptors; no shell, raw TeX, HTML, filters, includes, images, or network input
is accepted.  Two independent ReportLab passes must be byte-identical before a
candidate can be published.

Publication is an atomic, no-replace rename of a sealed directory.  Validation
rebuilds the complete directory in a separately retained replay root and requires
the canonical manifest to be byte-identical to its independent anchor.

This is a packaging and rendering boundary, not scientific or submission
authority.  Main-executable attestation is inherited from the Darwin rendered-
document machine-closure runner.  Python stdlib/dylibs and native dylib closure
remain explicitly unattested, and machine checks do not replace page-complete
human visual review.

The CLI also accepts the fixed four-role derivation argument shape.  That mode
reads one canonical bundle only from an inherited read-only descriptor,
regenerates the PDF through this same production path, requires the exact
expected renderer manifest and PDF hashes, and emits only PDF bytes on stdout.
A separately reviewed thin-arm64 launcher and authority receipt are still
required before the four-role closure can treat this as a real producer.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import fcntl
import hashlib
import html
import io
import json
import os
import re
import resource
import stat
import struct
import sys
import tempfile
import types
import unicodedata
import urllib.parse
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# The implementation is deliberately narrow and fail closed.  Complexity comes
# from authority/publishing boundaries rather than document semantics.
# ruff: noqa: COM812, PERF203, PLC0415, PLR0913, SIM115, SLF001, TRY300

SCHEMA: Final = "dialect-revision-rebuttal-render-v1"
CONTRACT: Final = "descriptor-rooted-reportlab-invariant-double-build-v1"
TEMPLATE_SCHEMA: Final = "dialect-revision-rebuttal-template-v1"
CONFIG_SCHEMA: Final = "dialect-revision-rebuttal-config-v1"
DERIVATION_BUNDLE_SCHEMA: Final = "dialect-revision-rebuttal-derivation-bundle-v1"
DERIVATION_BUNDLE_CONTRACT: Final = (
    "canonical-inputs-pinned-renderer-fresh-pdf-derivation-v1"
)
DERIVATION_PROTOCOL: Final = "dialect-pdf-derivation-fd-protocol-v1"
DERIVATION_ROLE: Final = "rebuttal"
MANIFEST_MEMBER: Final = "render-receipt.json"
PDF_MEMBER: Final = "response-to-reviewers.pdf"
SOURCE_MEMBER: Final = "source.canonical.md"
TEMPLATE_MEMBER: Final = "template.canonical.json"
CONFIG_MEMBER: Final = "config.canonical.json"
MEMBER_ORDER: Final = (
    SOURCE_MEMBER,
    TEMPLATE_MEMBER,
    CONFIG_MEMBER,
    PDF_MEMBER,
)
TOOL_ORDER: Final = ("python", "pdfinfo", "pdffonts", "pdftotext")
MACHINE_RUNNER_MEMBER: Final = (
    "analysis/build_tcga_revision_rendered_document_machine_closure.py"
)

MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
MAX_PRIVACY_CLASSIFIER_BYTES: Final = 16 * 1024 * 1024
MAX_JSON_BYTES: Final = 128 * 1024
MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_DERIVATION_BUNDLE_BYTES: Final = 8 * 1024 * 1024
MAX_FONT_BYTES: Final = 32 * 1024 * 1024
MAX_MACHINE_RUNNER_BYTES: Final = 8 * 1024 * 1024
MAX_PDF_BYTES: Final = 32 * 1024 * 1024
MAX_REPORTLAB_FILES: Final = 512
MAX_REPORTLAB_DIRECTORIES: Final = 64
MAX_REPORTLAB_ENTRIES: Final = 768
MAX_REPORTLAB_DEPTH: Final = 8
MAX_REPORTLAB_BYTES: Final = 32 * 1024 * 1024
MAX_REPORTLAB_BUNDLE_BYTES: Final = 32 * 1024 * 1024
MAX_PAGES: Final = 256
MAX_SOURCE_LINES: Final = 20_000
MAX_BLOCKS: Final = 5_000
MAX_QUOTES: Final = 1_000
MAX_RESPONSES: Final = 1_000
MAX_REVIEWERS: Final = 128
MAX_LIST_ITEMS: Final = 5_000
MAX_LEDGER_ENTRIES: Final = 12_000
MAX_STDERR_BYTES: Final = 256 * 1024
TOOL_TIMEOUT_SECONDS: Final = 120.0
MAX_FDS: Final = 40
READ_CHUNK_BYTES: Final = 64 * 1024
REPORTLAB_INVARIANT_EPOCH: Final = 946684800  # 2000-01-01 00:00:00 UTC

DERIVATION_LOCATORS: Final = {
    "renderer": "current-renderer",
    "machine_runner": "renderer-sibling-machine-runner",
    "runtime": "invoking-python",
    "reportlab": "invoking-python-reportlab",
    "regular_font": "system-arial-unicode",
    "bold_font": "system-arial-bold",
    "pdfinfo": "homebrew-pdfinfo",
    "pdffonts": "homebrew-pdffonts",
    "pdftotext": "homebrew-pdftotext",
}
DERIVATION_FIXED_LOCATOR_PATHS: Final = {
    "system-arial-unicode": Path(
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    ),
    "system-arial-bold": Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    "homebrew-pdfinfo": Path("/opt/homebrew/bin/pdfinfo"),
    "homebrew-pdffonts": Path("/opt/homebrew/bin/pdffonts"),
    "homebrew-pdftotext": Path("/opt/homebrew/bin/pdftotext"),
}
DERIVATION_NON_INFERENCE: Final = {
    "adapter_source_review": "not-inferred",
    "ambient_same_uid_filesystem_containment": "not-provided",
    "decoded_canonical_input_private_paths": (
        "rejected-as-utf8-text-without-recursive-decoding"
    ),
    "child_tool_and_dylib_closure": "not-attested",
    "coauthor_or_submission_approval": "not-inferred",
    "human_visual_approval": "required-separately",
    "journal_acceptance_or_upload": "not-inferred",
    "loaded_python_code_identity": "path-bytes-pinned-after-bootstrap-only",
    "native_adapter_authority": "not-provided-by-this-bundle",
    "pre_rendered_pdf_member": "absent",
    "producer_pdf_input": "none",
    "recursive_content_classification": "not-provided",
    "scientific_accuracy": "not-inferred",
}

SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
SOURCE_BEGIN_RE: Final = re.compile(
    r"<!-- SOURCE-COMMENT:([a-z0-9][a-z0-9._-]{1,127}):BEGIN -->",
)
SOURCE_END_RE: Final = re.compile(
    r"<!-- SOURCE-COMMENT:([a-z0-9][a-z0-9._-]{1,127}):END -->",
)
HEADING_RE: Final = re.compile(r"^(#{1,4}) ([^#].*)$")
LIST_RE: Final = re.compile(r"^(?:[-*] |([1-9][0-9]*)[.)] )(.*)$")
GENERIC_PLACEHOLDER_RE: Final = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:TODO|TBD|FIXME)(?![A-Za-z0-9_])",
)
GATE_MARKER_RE: Final = re.compile(
    r"\[(?:K500|CAL|COAUTH|COMP|MSK|SIM|NONE|FIG|TABLE|REL)"
    r"(?:\+(?:K500|CAL|COAUTH|COMP|MSK|SIM|NONE|FIG|TABLE|REL))*\s*:",
    re.IGNORECASE,
)
OTHER_PENDING_RE: Final = re.compile(
    r"(?im)(?:RECONCILIATION-(?:PENDING|TARGET):|DIALECT[-_ ]GATE:|"
    r"^\s*%\s*[A-Z0-9]+(?:\+[A-Z0-9]+)*\s+gate:)",
)
RESULT_LOCATION_SENTINEL_RE: Final = re.compile(
    r"\[\[(?:RESULT|LOCATION)",
    re.IGNORECASE,
)
RAW_TEX_RE: Final = re.compile(
    r"\\(?:input|include|includeonly|write|openin|openout|read|special|"
    r"immediate|usepackage|documentclass|catcode|csname|newread|newwrite)\b",
    re.IGNORECASE,
)
GENERIC_TEX_COMMAND_RE: Final = re.compile(r"\\[A-Za-z@]+")
INLINE_MATH_RE: Final = re.compile(r"\$([^$\n]+)\$")
RAW_HTML_RE: Final = re.compile(
    r"(?:</?[A-Za-z][^>\n]*>|<!--.*?-->|<!DOCTYPE\b[^>]*>|"
    r"<!\[CDATA\[.*?\]\]>|<\?.*?\?>)",
    re.IGNORECASE | re.DOTALL,
)
HTML_OPENER_RE: Final = re.compile(
    r"(?:<(?:!|\?|/)|<[A-Za-z][A-Za-z0-9-]*(?:\s|/|>))",
)
HTML_ENTITY_RE: Final = re.compile(r"&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
IMAGE_RE: Final = re.compile(r"!\[[^\]]*\]\([^)]*\)")
REFERENCE_IMAGE_RE: Final = re.compile(r"!\[[^\]\n]*\](?!\()")
UNSAFE_LINK_RE: Final = re.compile(
    r"(?i)(?<!!)\[[^\]\n]+\]\((?!https?://)[^)\n]+\)",
)
LOCAL_PATH_RE: Final = re.compile(
    r"(?im)(?:file://|(?<![A-Za-z0-9_.<-])(?:"
    r"\.\.?[/\\]|~(?:[A-Za-z0-9._-]+)?[/\\]|[A-Za-z]:[/\\]|"
    r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})[/\\]|"
    r"%[A-Za-z_][A-Za-z0-9_]*%[/\\]|"
    r"/(?!/)|\\\\)(?=\S))",
)
ANGLE_LOCAL_PATH_RE: Final = re.compile(
    r"(?i)<(?:"
    r"\.\.?[/\\]|"
    r"~(?:[A-Za-z0-9._-]+)?[/\\]|"
    r"[A-Za-z]:[/\\]|"
    r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})[/\\]|"
    r"%[A-Za-z_][A-Za-z0-9_]*%[/\\]|"
    r"\\\\|"
    r"[/\\](?:"
    r"(?:Users|private|tmp|var|etc|opt|usr|Library|System|root|home)(?:[/\\]|>)|"
    r"[^/\\<>\s]+[/\\]|"
    r"[^/\\<>\s]+>"
    r")"
    r")",
)
PERCENT_ESCAPE_RE: Final = re.compile(r"%[0-9A-Fa-f]{2}")
PUBLIC_AUTHORITY_AT_RE: Final = re.compile(
    r"(?:localhost|"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?|"
    r"(?:[0-9]{1,3}\.){3}[0-9]{1,3})"
    r"(?::[0-9]{1,5})?",
    re.IGNORECASE,
)
PUBLIC_URL_TAIL_CHARS: Final = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "-._~!$&'*+;=:@%/?#()",
)
INLINE_LINK_AT_RE: Final = re.compile(
    r"\[([^\]\n]+)\]\((https?://[^()\s]+)\)",
    re.IGNORECASE,
)
PDF_SIGNATURE: Final = b"%PDF-"
PDF_EOF: Final = b"%%EOF\n"
FORBIDDEN_PDF_TOKENS: Final = (
    b"/Encrypt",
    b"/JavaScript",
    b"/OpenAction",
    b"/Launch",
    b"/EmbeddedFile",
    b"/AcroForm",
    b"/XFA",
)
FONT_ROW_RE: Final = re.compile(
    r"^(\S+)\s+(Type 1C?|Type 3|TrueType|CID Type 0C?|CID TrueType)\s+"
    r"(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+(\d+)\s+(\d+)\s*$",
)
PDF_FONTS_HEADER: Final = (
    "name                                 type              encoding         "
    "emb sub uni object ID"
)
PDF_FONTS_SEPARATOR: Final = (
    "------------------------------------ ----------------- ---------------- "
    "--- --- --- ---------"
)


class RebuttalRenderError(ValueError):
    """Raised when deterministic rebuttal rendering or replay fails."""


class _NoImageModule(types.ModuleType):
    """Fail closed if ReportLab unexpectedly attempts any Pillow operation."""

    def __getattr__(self, name: str) -> NoReturn:
        _fail(f"image operation {name!r} is outside the rebuttal renderer contract")


@dataclass(frozen=True, slots=True)
class RebuttalRenderReceipt:
    """Summarize one published render or independent replay."""

    manifest_path: str
    manifest_sha256: str
    pdf_path: str
    pdf_sha256: str
    page_count: int
    quote_count: int
    response_count: int
    replay_root: str | None


@dataclass(frozen=True, slots=True)
class _Block:
    kind: str
    text: str
    level: int = 0
    marker: str = ""


@dataclass(frozen=True, slots=True)
class _MarkdownAudit:
    blocks: tuple[_Block, ...]
    quotes: tuple[tuple[str, str], ...]
    reviewer_count: int
    response_count: int


@dataclass(frozen=True, slots=True)
class _ReportLabBundle:
    raw: bytes
    tree_sha256: str
    file_count: int
    directory_count: int
    entry_count: int
    total_bytes: int
    records: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _Production:
    manifest: dict[str, object]
    manifest_raw: bytes
    pdf_raw: bytes
    page_count: int
    quote_count: int
    response_count: int


@dataclass(slots=True)
class _MachineAuthority:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str
    module: object

    def close(self) -> None:
        """Close the caller-anchored helper descriptor exactly once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _Snapshot:
    descriptor: int
    size: int
    sha256: str

    def close(self) -> None:
        """Close one unlinked, read-only derivation-input snapshot once."""
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _LocalProcessBudget:
    """Enforce the renderer's exact two-build plus three-QA process count."""

    count: int = 0
    maximum: int = 5

    def consume(self) -> None:
        """Consume one exact renderer process slot."""
        self.count += 1
        if self.count > self.maximum:
            _fail(
                f"render process count exceeds the exact {self.maximum}-process budget"
            )

    def assert_complete(self) -> None:
        """Require every expected child and no additional child."""
        if self.count != self.maximum:
            message = (
                f"render process count is {self.count}; expected exactly {self.maximum}"
            )
            _fail(
                message,
            )


def _fail(message: str) -> NoReturn:
    raise RebuttalRenderError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        _fail(f"cannot encode canonical JSON: {error}")


def _require_manifest_size(raw: bytes) -> bytes:
    if len(raw) > MAX_MANIFEST_BYTES:
        _fail(f"canonical manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit")
    return raw


def _expect_sha256(value: str, *, context: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _expect_token(value: object, *, context: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        _fail(f"{context} must be a canonical token")
    return value


def _validate_unicode(
    value: str,
    *,
    context: str,
    allowed_controls: frozenset[str] = frozenset(),
) -> None:
    for character in value:
        point = ord(character)
        category = unicodedata.category(character)
        if character in allowed_controls:
            continue
        if (
            category in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            or (character.isspace() and character != " ")
            or 0xFDD0 <= point <= 0xFDEF
            or point & 0xFFFF in {0xFFFE, 0xFFFF}
        ):
            _fail(f"{context} contains a forbidden Unicode control or noncharacter")


def _expect_string(value: object, *, context: str, maximum: int = 500) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != unicodedata.normalize("NFC", value)
        or len(value) > maximum
    ):
        _fail(f"{context} must be a nonempty canonical single-line string")
    _validate_unicode(value, context=context)
    return value


def _expect_bounded_int(
    value: object,
    *,
    context: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        _fail(f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def _decode_canonical_base64(
    value: object,
    *,
    context: str,
    maximum: int,
) -> bytes:
    encoded = _expect_string(
        value,
        context=f"{context}.base64",
        maximum=((maximum + 2) // 3) * 4,
    )
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as error:
        _fail(f"{context}.base64 is not strict base64: {error}")
    if len(raw) > maximum:
        _fail(f"{context} exceeds its decoded byte limit")
    if base64.b64encode(raw).decode("ascii") != encoded:
        _fail(f"{context}.base64 is not canonical")
    return raw


def _normalize_derivation_input(
    value: object,
    *,
    member: str,
    maximum: int,
) -> tuple[dict[str, object], bytes]:
    context = f"derivation bundle input {member}"
    record = _expect_keys(
        value,
        {"member", "encoding", "bytes", "sha256", "base64"},
        context=context,
    )
    if record["member"] != member or record["encoding"] != "base64":
        _fail(f"{context} member or encoding is invalid")
    raw = _decode_canonical_base64(record["base64"], context=context, maximum=maximum)
    _reject_private_paths_in_canonical_member(raw, member=member)
    size = _expect_bounded_int(
        record["bytes"],
        context=f"{context}.bytes",
        minimum=1,
        maximum=maximum,
    )
    digest = _expect_sha256(record["sha256"], context=f"{context}.sha256")
    if len(raw) != size or _sha256(raw) != digest:
        _fail(f"{context} does not match its size and SHA-256 anchors")
    return (
        {
            "member": member,
            "encoding": "base64",
            "bytes": size,
            "sha256": digest,
            "base64": base64.b64encode(raw).decode("ascii"),
        },
        raw,
    )


def _normalize_derivation_file_anchor(
    value: object,
    *,
    context: str,
    locator: str,
    maximum: int,
    member: str | None = None,
) -> dict[str, object]:
    keys = {"locator", "bytes", "sha256"}
    if member is not None:
        keys.add("member")
    record = _expect_keys(value, keys, context=context)
    if record["locator"] != locator:
        _fail(f"{context}.locator is invalid")
    if member is not None and record["member"] != member:
        _fail(f"{context}.member is invalid")
    size = _expect_bounded_int(
        record["bytes"],
        context=f"{context}.bytes",
        minimum=1,
        maximum=maximum,
    )
    digest = _expect_sha256(record["sha256"], context=f"{context}.sha256")
    normalized: dict[str, object] = {
        "locator": locator,
        "bytes": size,
        "sha256": digest,
    }
    if member is not None:
        normalized["member"] = member
    return normalized


def _normalize_derivation_bundle(value: object) -> dict[str, object]:
    """Validate one canonical rebuttal source bundle with no PDF input member."""
    record = _expect_keys(
        value,
        {
            "schema",
            "contract",
            "release_id",
            "role",
            "producer_protocol",
            "producer_arguments",
            "canonical_inputs",
            "dependencies",
            "expected_output",
            "non_inference",
        },
        context="derivation bundle",
    )
    if (
        record["schema"] != DERIVATION_BUNDLE_SCHEMA
        or record["contract"] != DERIVATION_BUNDLE_CONTRACT
        or record["role"] != DERIVATION_ROLE
        or record["producer_protocol"] != DERIVATION_PROTOCOL
    ):
        _fail("derivation bundle schema, contract, role, or protocol is invalid")
    release_id = _expect_token(record["release_id"], context="bundle.release_id")
    expected_arguments = [
        "--dialect-derivation-protocol",
        DERIVATION_PROTOCOL,
        "--pdf-id",
        DERIVATION_ROLE,
        "--source-fd",
        "{source_fd}",
        "--pdf-output",
        "stdout",
    ]
    if record["producer_arguments"] != expected_arguments:
        _fail("derivation bundle producer_arguments drifted")

    raw_inputs = record["canonical_inputs"]
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 3:
        _fail("derivation bundle canonical_inputs must contain exactly three members")
    normalized_inputs: list[dict[str, object]] = []
    decoded: dict[str, bytes] = {}
    for index, (member, maximum) in enumerate(
        (
            (SOURCE_MEMBER, MAX_SOURCE_BYTES),
            (TEMPLATE_MEMBER, MAX_JSON_BYTES),
            (CONFIG_MEMBER, MAX_JSON_BYTES),
        ),
    ):
        normalized, raw = _normalize_derivation_input(
            raw_inputs[index],
            member=member,
            maximum=maximum,
        )
        normalized_inputs.append(normalized)
        decoded[member] = raw
    source = decoded[SOURCE_MEMBER]
    template_raw = decoded[TEMPLATE_MEMBER]
    config_raw = decoded[CONFIG_MEMBER]
    if _canonicalize_markdown(source) != source:
        _fail("bundle Markdown input is not canonical")
    template = _normalize_template(
        _json_without_duplicates(template_raw, context="bundle template")
    )
    config = _normalize_config(
        _json_without_duplicates(config_raw, context="bundle config")
    )
    if _canonical_json(template) != template_raw:
        _fail("bundle template input is not canonical JSON")
    if _canonical_json(config) != config_raw:
        _fail("bundle config input is not canonical JSON")
    if config["release_id"] != release_id:
        _fail("bundle config release_id does not match the bundle")
    audit = _parse_markdown(source)
    _validate_title_binding(audit, config)

    dependencies = _expect_keys(
        record["dependencies"],
        {"renderer", "machine_runner", "runtime", "tools", "fonts", "reportlab"},
        context="bundle.dependencies",
    )
    renderer_anchor = _normalize_derivation_file_anchor(
        dependencies["renderer"],
        context="bundle.dependencies.renderer",
        locator=DERIVATION_LOCATORS["renderer"],
        maximum=MAX_SOURCE_BYTES,
        member="analysis/render_tcga_revision_rebuttal.py",
    )
    runner_anchor = _normalize_derivation_file_anchor(
        dependencies["machine_runner"],
        context="bundle.dependencies.machine_runner",
        locator=DERIVATION_LOCATORS["machine_runner"],
        maximum=MAX_MACHINE_RUNNER_BYTES,
        member=MACHINE_RUNNER_MEMBER,
    )
    runtime = _expect_keys(
        dependencies["runtime"],
        {"locator", "python_tag", "bytes", "sha256"},
        context="bundle.dependencies.runtime",
    )
    if runtime["locator"] != DERIVATION_LOCATORS["runtime"]:
        _fail("bundle.dependencies.runtime.locator is invalid")
    python_tag = _expect_string(
        runtime["python_tag"],
        context="bundle.dependencies.runtime.python_tag",
        maximum=8,
    )
    if re.fullmatch(r"3\.[0-9]{1,2}", python_tag) is None:
        _fail("bundle.dependencies.runtime.python_tag is invalid")
    runtime_anchor = {
        "locator": DERIVATION_LOCATORS["runtime"],
        "python_tag": python_tag,
        "bytes": _expect_bounded_int(
            runtime["bytes"],
            context="bundle.dependencies.runtime.bytes",
            minimum=1,
            maximum=128 * 1024 * 1024,
        ),
        "sha256": _expect_sha256(
            runtime["sha256"], context="bundle.dependencies.runtime.sha256"
        ),
    }

    tools = dependencies["tools"]
    if not isinstance(tools, list) or len(tools) != len(TOOL_ORDER) - 1:
        _fail("bundle.dependencies.tools has the wrong cardinality")
    normalized_tools: list[dict[str, object]] = []
    for index, name in enumerate(TOOL_ORDER[1:]):
        context = f"bundle.dependencies.tools[{index}]"
        item = _expect_keys(
            tools[index], {"name", "locator", "bytes", "sha256"}, context=context
        )
        if item["name"] != name:
            _fail(f"{context}.name is not in fixed tool order")
        anchor = _normalize_derivation_file_anchor(
            {key: item[key] for key in ("locator", "bytes", "sha256")},
            context=context,
            locator=DERIVATION_LOCATORS[name],
            maximum=128 * 1024 * 1024,
        )
        normalized_tools.append({"name": name, **anchor})

    fonts = dependencies["fonts"]
    if not isinstance(fonts, list) or len(fonts) != 2:
        _fail("bundle.dependencies.fonts must contain two roles")
    normalized_fonts: list[dict[str, object]] = []
    for index, role in enumerate(("regular", "bold")):
        context = f"bundle.dependencies.fonts[{index}]"
        item = _expect_keys(
            fonts[index],
            {"role", "locator", "bytes", "sha256", "postscript_name"},
            context=context,
        )
        if item["role"] != role:
            _fail(f"{context}.role is not in fixed font order")
        anchor = _normalize_derivation_file_anchor(
            {key: item[key] for key in ("locator", "bytes", "sha256")},
            context=context,
            locator=DERIVATION_LOCATORS[f"{role}_font"],
            maximum=MAX_FONT_BYTES,
        )
        postscript_name = _expect_string(
            item["postscript_name"],
            context=f"{context}.postscript_name",
            maximum=127,
        )
        _reject_private_paths(postscript_name, context=f"{context}.postscript_name")
        normalized_fonts.append(
            {"role": role, **anchor, "postscript_name": postscript_name}
        )
    if normalized_fonts[0]["sha256"] == normalized_fonts[1]["sha256"]:
        _fail("bundle font roles must bind distinct bytes")

    reportlab = _expect_keys(
        dependencies["reportlab"],
        {
            "locator",
            "tree_sha256",
            "file_count",
            "directory_count",
            "entry_count",
            "total_bytes",
            "bundle_sha256",
            "bundle_bytes",
        },
        context="bundle.dependencies.reportlab",
    )
    if reportlab["locator"] != DERIVATION_LOCATORS["reportlab"]:
        _fail("bundle.dependencies.reportlab.locator is invalid")
    normalized_reportlab = {
        "locator": DERIVATION_LOCATORS["reportlab"],
        "tree_sha256": _expect_sha256(
            reportlab["tree_sha256"],
            context="bundle.dependencies.reportlab.tree_sha256",
        ),
        "file_count": _expect_bounded_int(
            reportlab["file_count"],
            context="bundle.dependencies.reportlab.file_count",
            minimum=1,
            maximum=MAX_REPORTLAB_FILES,
        ),
        "directory_count": _expect_bounded_int(
            reportlab["directory_count"],
            context="bundle.dependencies.reportlab.directory_count",
            minimum=1,
            maximum=MAX_REPORTLAB_DIRECTORIES,
        ),
        "entry_count": _expect_bounded_int(
            reportlab["entry_count"],
            context="bundle.dependencies.reportlab.entry_count",
            minimum=1,
            maximum=MAX_REPORTLAB_ENTRIES,
        ),
        "total_bytes": _expect_bounded_int(
            reportlab["total_bytes"],
            context="bundle.dependencies.reportlab.total_bytes",
            minimum=1,
            maximum=MAX_REPORTLAB_BYTES,
        ),
        "bundle_sha256": _expect_sha256(
            reportlab["bundle_sha256"],
            context="bundle.dependencies.reportlab.bundle_sha256",
        ),
        "bundle_bytes": _expect_bounded_int(
            reportlab["bundle_bytes"],
            context="bundle.dependencies.reportlab.bundle_bytes",
            minimum=1,
            maximum=MAX_REPORTLAB_BUNDLE_BYTES,
        ),
    }
    if normalized_reportlab["entry_count"] < normalized_reportlab["file_count"]:
        _fail("bundle ReportLab entry count is smaller than its file count")

    expected_output = _expect_keys(
        record["expected_output"],
        {"renderer_manifest", "pdf"},
        context="bundle.expected_output",
    )
    expected_manifest = _expect_keys(
        expected_output["renderer_manifest"],
        {"member", "bytes", "sha256"},
        context="bundle.expected_output.renderer_manifest",
    )
    if expected_manifest["member"] != MANIFEST_MEMBER:
        _fail("bundle expected renderer manifest member is invalid")
    normalized_manifest = {
        "member": MANIFEST_MEMBER,
        "bytes": _expect_bounded_int(
            expected_manifest["bytes"],
            context="bundle.expected_output.renderer_manifest.bytes",
            minimum=1,
            maximum=MAX_MANIFEST_BYTES,
        ),
        "sha256": _expect_sha256(
            expected_manifest["sha256"],
            context="bundle.expected_output.renderer_manifest.sha256",
        ),
    }
    expected_pdf = _expect_keys(
        expected_output["pdf"],
        {"member", "bytes", "sha256"},
        context="bundle.expected_output.pdf",
    )
    if expected_pdf["member"] != PDF_MEMBER:
        _fail("bundle expected PDF member is invalid")
    normalized_pdf = {
        "member": PDF_MEMBER,
        "bytes": _expect_bounded_int(
            expected_pdf["bytes"],
            context="bundle.expected_output.pdf.bytes",
            minimum=len(PDF_SIGNATURE) + len(PDF_EOF),
            maximum=MAX_PDF_BYTES,
        ),
        "sha256": _expect_sha256(
            expected_pdf["sha256"], context="bundle.expected_output.pdf.sha256"
        ),
    }
    if record["non_inference"] != DERIVATION_NON_INFERENCE:
        _fail("derivation bundle non-inference limits drifted")
    return {
        "schema": DERIVATION_BUNDLE_SCHEMA,
        "contract": DERIVATION_BUNDLE_CONTRACT,
        "release_id": release_id,
        "role": DERIVATION_ROLE,
        "producer_protocol": DERIVATION_PROTOCOL,
        "producer_arguments": expected_arguments,
        "canonical_inputs": normalized_inputs,
        "dependencies": {
            "renderer": renderer_anchor,
            "machine_runner": runner_anchor,
            "runtime": runtime_anchor,
            "tools": normalized_tools,
            "fonts": normalized_fonts,
            "reportlab": normalized_reportlab,
        },
        "expected_output": {
            "renderer_manifest": normalized_manifest,
            "pdf": normalized_pdf,
        },
        "non_inference": dict(DERIVATION_NON_INFERENCE),
    }


def _expect_keys(
    value: object,
    keys: set[str],
    *,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{context} must have exactly keys {sorted(keys)}")
    return value


def _decode_utf8(raw: bytes, *, context: str) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{context} must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        _fail(f"{context} is not strict UTF-8: {error}")
    if "\x00" in text:
        _fail(f"{context} contains NUL")
    return text


def _canonicalize_markdown(raw: bytes) -> bytes:
    text = _decode_utf8(raw, context="source Markdown")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    _validate_unicode(
        text,
        context="source Markdown",
        allowed_controls=frozenset({"\n", "\t"}),
    )
    canonical_lines: list[str] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if any(ord(character) < 32 and character != "\t" for character in line):
            _fail(f"source Markdown line {line_number} contains a control character")
        if re.search(r" {2,}$", line) is not None:
            _fail("source Markdown hard line breaks are unsupported")
        canonical_lines.append(line.rstrip(" \t"))
    while canonical_lines and not canonical_lines[-1]:
        canonical_lines.pop()
    if not canonical_lines:
        _fail("source Markdown is empty")
    canonical = ("\n".join(canonical_lines) + "\n").encode("utf-8")
    if len(canonical) > MAX_SOURCE_BYTES:
        _fail(f"canonical source exceeds the {MAX_SOURCE_BYTES}-byte limit")
    return canonical


def _json_without_duplicates(raw: bytes, *, context: str) -> object:
    text = _decode_utf8(raw, context=context)

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                _fail(f"{context} duplicates JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs)
    except json.JSONDecodeError as error:
        _fail(f"{context} is not valid JSON: {error}")


def _bounded_number(
    value: object,
    *,
    context: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be numeric")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        _fail(f"{context} must be in [{minimum:g}, {maximum:g}]")
    return numeric


def _normalize_color(value: object, *, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"#[0-9A-F]{6}", value) is None:
        _fail(f"{context} must be an uppercase six-digit RGB color")
    return value


def _normalize_template(value: object) -> dict[str, object]:
    record = _expect_keys(
        value,
        {"schema", "page_size", "margins", "type", "colors"},
        context="template",
    )
    if record["schema"] != TEMPLATE_SCHEMA or record["page_size"] != "letter":
        _fail("template schema or page size is unsupported")
    margins = _expect_keys(
        record["margins"],
        {"top", "right", "bottom", "left"},
        context="template.margins",
    )
    normalized_margins = {
        key: _bounded_number(
            margins[key],
            context=f"template.margins.{key}",
            minimum=36,
            maximum=108,
        )
        for key in ("top", "right", "bottom", "left")
    }
    type_record = _expect_keys(
        record["type"],
        {"title", "heading_2", "heading_3", "heading_4", "body", "leading"},
        context="template.type",
    )
    type_bounds = {
        "title": (18, 30),
        "heading_2": (13, 22),
        "heading_3": (10, 18),
        "heading_4": (9, 15),
        "body": (9, 13),
        "leading": (12, 20),
    }
    normalized_type = {
        key: _bounded_number(
            type_record[key],
            context=f"template.type.{key}",
            minimum=type_bounds[key][0],
            maximum=type_bounds[key][1],
        )
        for key in type_bounds
    }
    if normalized_type["leading"] <= normalized_type["body"]:
        _fail("template body leading must exceed body size")
    colors = _expect_keys(
        record["colors"],
        {"ink", "muted", "accent", "quote_background", "rule"},
        context="template.colors",
    )
    normalized_colors = {
        key: _normalize_color(colors[key], context=f"template.colors.{key}")
        for key in ("ink", "muted", "accent", "quote_background", "rule")
    }
    return {
        "schema": TEMPLATE_SCHEMA,
        "page_size": "letter",
        "margins": normalized_margins,
        "type": normalized_type,
        "colors": normalized_colors,
    }


def _normalize_config(value: object) -> dict[str, object]:
    record = _expect_keys(
        value,
        {
            "schema",
            "release_id",
            "manuscript_id",
            "manuscript_title",
            "response_title",
            "authors",
            "source_date_epoch",
        },
        context="config",
    )
    if record["schema"] != CONFIG_SCHEMA:
        _fail("config schema is unsupported")
    release_id = _expect_token(record["release_id"], context="config.release_id")
    authors_raw = record["authors"]
    if not isinstance(authors_raw, list) or not 1 <= len(authors_raw) <= 32:
        _fail("config.authors must contain 1-32 authors")
    authors = [
        _expect_string(author, context=f"config.authors[{index}]", maximum=120)
        for index, author in enumerate(authors_raw)
    ]
    epoch = record["source_date_epoch"]
    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch != REPORTLAB_INVARIANT_EPOCH
    ):
        _fail("config.source_date_epoch must equal ReportLab's fixed invariant epoch")
    normalized = {
        "schema": CONFIG_SCHEMA,
        "release_id": release_id,
        "manuscript_id": _expect_string(
            record["manuscript_id"],
            context="config.manuscript_id",
            maximum=120,
        ),
        "manuscript_title": _expect_string(
            record["manuscript_title"],
            context="config.manuscript_title",
            maximum=500,
        ),
        "response_title": _expect_string(
            record["response_title"],
            context="config.response_title",
            maximum=160,
        ),
        "authors": authors,
        "source_date_epoch": epoch,
    }
    for key, item in normalized.items():
        if isinstance(item, str):
            _reject_private_paths(item, context=f"config.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                _reject_private_paths(
                    str(nested),
                    context=f"config.{key}[{index}]",
                )
    return normalized


def _redact_public_urls(classifier: str) -> str:
    """Redact conservative public URLs without swallowing path-like suffixes."""
    output: list[str] = []
    retained = 0
    for scheme in re.finditer(r"(?i)https?://", classifier):
        if scheme.start() < retained:
            continue
        authority = PUBLIC_AUTHORITY_AT_RE.match(classifier, scheme.end())
        if authority is None:
            continue
        candidate = classifier[scheme.start() : authority.end()]
        try:
            split = urllib.parse.urlsplit(candidate)
            port = split.port
        except ValueError:
            continue
        if (
            split.scheme.lower() not in {"http", "https"}
            or split.hostname is None
            or split.username is not None
            or split.password is not None
            or (port is not None and not 1 <= port <= 65535)
        ):
            continue
        end = authority.end()
        if end < len(classifier) and classifier[end] in "/?#":
            scan = end
            depth = 0
            last_balanced = end
            while scan < len(classifier):
                character = classifier[scan]
                if character not in PUBLIC_URL_TAIL_CHARS:
                    break
                if character == "(":
                    depth += 1
                elif character == ")":
                    if depth == 0:
                        break
                    depth -= 1
                scan += 1
                if depth == 0:
                    last_balanced = scan
            end = last_balanced
        output.append(classifier[retained : scheme.start()])
        output.append("[public-url]")
        retained = end
    output.append(classifier[retained:])
    return "".join(output)


def _reject_private_paths(value: str, *, context: str) -> None:
    """Reject host-private path forms from every byte that can be published."""
    classifier = unicodedata.normalize("NFKC", value).translate(
        str.maketrans(
            {
                "\N{DIVISION SLASH}": "/",
                "\N{FRACTION SLASH}": "/",
                "\N{REVERSE SOLIDUS OPERATOR}": "\\",
            },
        ),
    )
    if len(classifier.encode("utf-8")) > MAX_PRIVACY_CLASSIFIER_BYTES:
        _fail(f"{context} exceeds the bounded privacy-classifier size")
    classifier = _redact_public_urls(classifier)
    if PERCENT_ESCAPE_RE.search(classifier) is not None:
        _fail(
            f"{context} contains a private absolute, home, file, traversal, "
            "or percent-encoded path",
        )
    private_patterns = (
        LOCAL_PATH_RE,
        ANGLE_LOCAL_PATH_RE,
        re.compile(
            r"(?i)(?<![A-Za-z0-9.<])[/\\](?:Users|private|tmp|var|etc|opt|usr|"
            r"Library|System|root|home)(?:\b|[/\\])",
        ),
        re.compile(
            r"(?i)(?<![A-Za-z0-9.<])[/\\][A-Za-z0-9._~-]+"
            r"(?:[/\\][A-Za-z0-9._~@+\-]+)+",
        ),
        re.compile(
            r"(?i)(?<![A-Za-z0-9_.-])(?:research|output|tmp|temp|\.codex|"
            r"\.agents|\.git|\.cache)[/\\]",
        ),
        re.compile(
            r"(?i)(?<![A-Za-z0-9_.-])(?:data|results|attachments?|uploads?)[/\\]"
            r"(?:[^/\\\s]+[/\\])*[^/\\\s]+\.[A-Za-z0-9]{1,12}"
            r"(?![A-Za-z0-9])",
        ),
        re.compile(
            r"(?i)(?<![A-Za-z0-9_.-])(?:data|results|attachments?|uploads?)"
            r"[/\\]\.[^/\\\s]+",
        ),
    )
    if any(pattern.search(classifier) is not None for pattern in private_patterns):
        _fail(f"{context} contains a private absolute, home, file, or traversal path")


def _reject_private_paths_in_canonical_member(raw: bytes, *, member: str) -> None:
    """Scan one decoded canonical member as UTF-8 text without recursive decoding."""
    text = _decode_utf8(raw, context=f"decoded canonical input {member}")
    _reject_private_paths(text, context=f"decoded canonical input {member}")


def _reject_unresolved_or_unsafe(markdown: str) -> None:
    classifier = unicodedata.normalize("NFKC", markdown)
    checks = (
        (GENERIC_PLACEHOLDER_RE, "TODO/TBD/FIXME placeholder"),
        (GATE_MARKER_RE, "unresolved DIALECT gate marker"),
        (OTHER_PENDING_RE, "unresolved reconciliation or gate marker"),
        (RAW_TEX_RE, "raw TeX file or execution primitive"),
        (IMAGE_RE, "Markdown image"),
        (REFERENCE_IMAGE_RE, "Markdown reference image"),
        (UNSAFE_LINK_RE, "non-HTTP(S) Markdown link"),
    )
    for pattern, label in checks:
        if pattern.search(classifier) is not None:
            _fail(f"source Markdown contains a forbidden {label}")
    without_supported_symbols = classifier.replace(r"$\rho$", "").replace(
        r"$\tau$",
        "",
    )
    if GENERIC_TEX_COMMAND_RE.search(without_supported_symbols) is not None:
        _fail("source Markdown contains a raw TeX command")
    if (
        RAW_HTML_RE.search(classifier) is not None
        or HTML_OPENER_RE.search(classifier) is not None
        or HTML_ENTITY_RE.search(classifier) is not None
    ):
        _fail("source Markdown contains raw HTML or entity syntax")


def _reject_result_location_sentinels(markdown: str) -> None:
    """Reject revision placeholders everywhere, including literal quotations."""
    classifier = unicodedata.normalize("NFKC", markdown)
    if RESULT_LOCATION_SENTINEL_RE.search(classifier) is not None:
        _fail("source Markdown contains an unresolved RESULT/LOCATION sentinel")


def _reject_unsupported_block_line(line: str) -> None:
    """Reject block syntax that the deliberately narrow parser cannot preserve."""
    stripped = line.lstrip()
    compact = re.sub(r"[ \t]", "", stripped)
    if (
        len(compact) >= 3
        and len(set(compact)) == 1
        and compact[0] in {"-", "_", "*"}
        and line != "---"
    ):
        _fail("source Markdown contains an unsupported thematic break")
    if re.match(r"^#{1,6}\t", stripped):
        _fail("source Markdown tab-delimited headings are unsupported")
    if re.match(r"^#{5,}(?:\s|$)", stripped):
        _fail("source Markdown heading levels 5 and 6 are unsupported")
    if line.startswith(("+ ", "    ", "\t")):
        _fail("source Markdown contains an unsupported list or indented code block")
    list_shaped = re.match(
        r"^(?:[-+*](?:[ \t]|$)|[0-9]+[.)](?:[ \t]|$))",
        stripped,
    )
    if list_shaped is not None and LIST_RE.fullmatch(line) is None:
        _fail("source Markdown contains an unsupported or nested list marker")
    if line != stripped and stripped.startswith(("#", ">", "```", "~~~")):
        _fail("source Markdown contains an unsupported indented block marker")
    if re.fullmatch(r"=+", line):
        _fail("source Markdown setext headings are unsupported")
    if line.startswith(("```", "~~~")):
        _fail("source Markdown contains a code fence outside a reviewer quotation")
    if stripped.startswith("|"):
        _fail("source Markdown tables are outside the renderer contract")
    if re.match(r"^\[[^\]\n]+\]:", stripped):
        _fail("source Markdown reference-link definitions are unsupported")


def _parse_inline_parts(text: str) -> tuple[tuple[str, str, str], ...]:
    """Parse a small nonnested inline grammar shared by rendering and QA."""
    if "\n" in text or "\r" in text:
        _fail("inline Markdown must be a single logical line")
    parts: list[tuple[str, str, str]] = []
    plain: list[str] = []

    def flush() -> None:
        if plain:
            parts.append(("plain", "".join(plain), ""))
            plain.clear()

    cursor = 0
    while cursor < len(text):
        if text.startswith(r"\$", cursor):
            plain.append("$")
            cursor += 2
            continue
        character = text[cursor]
        if character == "\\":
            _fail("source Markdown contains an unsupported escape or TeX command")
        if character == "`":
            end = text.find("`", cursor + 1)
            if end < 0 or end == cursor + 1:
                _fail("source Markdown contains an unmatched or empty code span")
            flush()
            parts.append(("code", text[cursor + 1 : end], ""))
            cursor = end + 1
            continue
        if text.startswith("**", cursor):
            end = text.find("**", cursor + 2)
            if end < 0 or end == cursor + 2:
                _fail("source Markdown contains unmatched or empty bold markup")
            content = text[cursor + 2 : end]
            if any(marker in content for marker in ("*", "`", "$", "\\", "[", "]")):
                _fail("source Markdown contains nested or ambiguous bold markup")
            flush()
            parts.append(("bold", content, ""))
            cursor = end + 2
            continue
        if character == "*":
            _fail(
                "italic Markdown requires an unattested italic font and is unsupported",
            )
        if character == "$":
            end = text.find("$", cursor + 1)
            if end < 0 or end == cursor + 1:
                _fail("source Markdown contains malformed or empty inline math")
            expression = text[cursor + 1 : end]
            if "$" in expression:
                _fail("source Markdown contains nested inline math")
            if expression in {r"\rho", r"\tau"}:
                visible = {r"\rho": "\N{GREEK SMALL LETTER RHO}", r"\tau": "τ"}[
                    expression
                ]
            elif re.fullmatch(r"[A-Za-z0-9 .,:;=<>+\-*/()%]+", expression):
                visible = expression
            else:
                _fail("source Markdown contains unsupported TeX-style inline math")
            flush()
            parts.append(("math", visible, ""))
            cursor = end + 1
            continue
        if character == "[":
            link = INLINE_LINK_AT_RE.match(text, cursor)
            if link is not None:
                label, target = link.groups()
                if any(marker in label for marker in ("[", "]", "*", "`", "$", "\\")):
                    _fail("source Markdown contains nested or ambiguous link markup")
                flush()
                parts.append(("link", label, target))
                cursor = link.end()
                continue
            closing = text.find("](", cursor + 1)
            if closing >= 0:
                _fail("source Markdown contains unsupported or ambiguous link markup")
            if re.search(r"\][ \t]*\[", text[cursor:]) is not None:
                _fail("source Markdown reference links are unsupported")
        if character == "_" and (cursor == 0 or not text[cursor - 1].isalnum()):
            delimiter = "__" if text.startswith("__", cursor) else "_"
            end = text.find(delimiter, cursor + len(delimiter))
            if end > cursor + len(delimiter) and (
                end + len(delimiter) == len(text)
                or not text[end + len(delimiter)].isalnum()
            ):
                _fail("underscore emphasis Markdown is unsupported")
        plain.append(character)
        cursor += 1
    flush()
    return tuple(parts)


def _validate_inline_text(text: str) -> None:
    _parse_inline_parts(text)


def _parse_markdown(canonical: bytes) -> _MarkdownAudit:
    text = canonical.decode("utf-8")
    _reject_result_location_sentinels(text)
    _reject_private_paths(text, context="canonical source Markdown")
    lines = text[:-1].split("\n")
    if len(lines) > MAX_SOURCE_LINES:
        _fail(f"source Markdown exceeds the {MAX_SOURCE_LINES}-line limit")
    blocks: list[_Block] = []
    quotes: list[tuple[str, str]] = []
    quote_ids: set[str] = set()
    list_item_count = 0
    reviewer_count = 0
    reviewer_numbers: list[int] = []
    in_reviewer_section = False
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if not line:
            cursor += 1
            continue
        begin = SOURCE_BEGIN_RE.fullmatch(line)
        if begin is not None:
            if not in_reviewer_section:
                _fail("reviewer quotations must appear under a 'Reviewer #N' section")
            quote_id = begin.group(1)
            if quote_id in quote_ids:
                _fail(f"source comment id {quote_id!r} is duplicated")
            if cursor + 2 >= len(lines) or lines[cursor + 1] != "```text":
                _fail(f"source comment {quote_id!r} must contain one text fence")
            cursor += 2
            quote_lines: list[str] = []
            while cursor < len(lines) and lines[cursor] != "```":
                if SOURCE_BEGIN_RE.fullmatch(lines[cursor]) or SOURCE_END_RE.fullmatch(
                    lines[cursor],
                ):
                    _fail(f"source comment {quote_id!r} nests a marker")
                quote_lines.append(lines[cursor])
                cursor += 1
            if cursor >= len(lines):
                _fail(f"source comment {quote_id!r} has no closing fence")
            cursor += 1
            if cursor >= len(lines) or SOURCE_END_RE.fullmatch(lines[cursor]) is None:
                _fail(f"source comment {quote_id!r} has no exact end marker")
            end = SOURCE_END_RE.fullmatch(lines[cursor])
            if end is None or end.group(1) != quote_id:
                _fail(f"source comment {quote_id!r} has a mismatched end marker")
            if not quote_lines or not quote_lines[0] or not quote_lines[-1]:
                _fail(
                    f"source comment {quote_id!r} must not have blank edge lines",
                )
            quote = "\n".join(quote_lines)
            quote_ids.add(quote_id)
            quotes.append((quote_id, quote))
            blocks.append(_Block("quote", quote, marker=quote_id))
            cursor += 1
            continue
        if SOURCE_END_RE.fullmatch(line) is not None:
            _fail("source Markdown contains an unmatched source-comment end marker")
        _reject_unsupported_block_line(line)
        if re.fullmatch(r"=+", line) or (
            line == "---" and cursor > 0 and bool(lines[cursor - 1])
        ):
            _fail("source Markdown setext headings are unsupported")
        heading = HEADING_RE.fullmatch(line)
        if heading is not None:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if not title:
                _fail("source Markdown contains an empty heading")
            if re.search(r"\s#+$", title) is not None:
                _fail("source Markdown closing heading hashes are unsupported")
            if level == 2 and re.fullmatch(r"Reviewer #[1-9][0-9]*", title):
                reviewer_count += 1
                reviewer_numbers.append(int(title.removeprefix("Reviewer #")))
                in_reviewer_section = True
            elif level == 2:
                in_reviewer_section = False
            blocks.append(_Block("heading", title, level=level))
            cursor += 1
            continue
        if line == "---":
            blocks.append(_Block("rule", ""))
            cursor += 1
            continue
        if line.startswith(">"):
            if not in_reviewer_section:
                _fail("reviewer quotations must appear under a 'Reviewer #N' section")
            quote_lines = []
            while cursor < len(lines) and lines[cursor].startswith(">"):
                current = lines[cursor]
                quote_lines.append(
                    current[2:] if current.startswith("> ") else current[1:]
                )
                cursor += 1
            if cursor < len(lines) and lines[cursor]:
                _fail("block quotations require a blank line before the response")
            quote_number = len(quotes) + 1
            quote_id = f"blockquote-{quote_number}"
            while quote_id in quote_ids:
                quote_number += 1
                quote_id = f"blockquote-{quote_number}"
            if not quote_lines or not quote_lines[0] or not quote_lines[-1]:
                _fail("source Markdown contains an empty-edged block quotation")
            quote = "\n".join(quote_lines)
            quote_ids.add(quote_id)
            quotes.append((quote_id, quote))
            blocks.append(_Block("quote", quote, marker=quote_id))
            continue
        list_match = LIST_RE.fullmatch(line)
        if list_match is not None:
            ordered = list_match.group(1) is not None
            marker = "ordered" if ordered else "bullet"
            items: list[str] = []
            while cursor < len(lines):
                match = LIST_RE.fullmatch(lines[cursor])
                if match is None or (match.group(1) is not None) != ordered:
                    break
                if ordered and int(match.group(1)) != len(items) + 1:
                    _fail("ordered lists must start at 1 and remain sequential")
                item_parts = [match.group(2).strip()]
                if not item_parts[0]:
                    _fail("source Markdown contains an empty list item")
                cursor += 1
                while (
                    cursor < len(lines)
                    and lines[cursor]
                    and lines[cursor].startswith(("  ", "\t"))
                    and LIST_RE.fullmatch(lines[cursor]) is None
                ):
                    raw_continuation = lines[cursor]
                    continuation = raw_continuation.strip()
                    _reject_unsupported_block_line(raw_continuation)
                    if (
                        not continuation
                        or raw_continuation.startswith(("    ", "\t"))
                        or LIST_RE.fullmatch(continuation)
                        or continuation.startswith(("+ ", ">", "```", "~~~", "|"))
                        or HEADING_RE.fullmatch(continuation)
                        or SOURCE_BEGIN_RE.fullmatch(continuation)
                        or SOURCE_END_RE.fullmatch(continuation)
                    ):
                        _fail(
                            "list continuation contains an unsupported or "
                            "nested construct"
                        )
                    item_parts.append(continuation)
                    cursor += 1
                items.append(" ".join(item_parts))
                list_item_count += 1
                if list_item_count > MAX_LIST_ITEMS:
                    _fail(
                        f"source Markdown exceeds the {MAX_LIST_ITEMS}-list-item limit",
                    )
            if cursor < len(lines) and lines[cursor]:
                _fail("lists require a blank line before the following block")
            blocks.append(_Block("list", "\n".join(items), marker=marker))
            continue
        paragraph = [line.strip()]
        cursor += 1
        while cursor < len(lines):
            candidate = lines[cursor]
            if (
                not candidate
                or HEADING_RE.fullmatch(candidate)
                or SOURCE_BEGIN_RE.fullmatch(candidate)
                or SOURCE_END_RE.fullmatch(candidate)
                or LIST_RE.fullmatch(candidate)
                or candidate == "---"
                or candidate.startswith((">", "```", "~~~"))
            ):
                break
            if candidate.lstrip().startswith("|"):
                _fail("source Markdown tables are outside the renderer contract")
            _reject_unsupported_block_line(candidate)
            paragraph.append(candidate.strip())
            cursor += 1
        blocks.append(_Block("paragraph", " ".join(paragraph)))

    if len(blocks) > MAX_BLOCKS:
        _fail(f"source Markdown exceeds the {MAX_BLOCKS}-block limit")
    if len(quotes) > MAX_QUOTES:
        _fail(f"source Markdown exceeds the {MAX_QUOTES}-quote limit")
    if reviewer_count > MAX_REVIEWERS:
        _fail(f"source Markdown exceeds the {MAX_REVIEWERS}-reviewer limit")
    if not blocks or blocks[0].kind != "heading" or blocks[0].level != 1:
        _fail("source Markdown must begin with one level-one heading")
    if sum(block.kind == "heading" and block.level == 1 for block in blocks) != 1:
        _fail("source Markdown must contain exactly one level-one heading")
    if reviewer_count < 1:
        _fail("source Markdown must contain at least one 'Reviewer #N' section")
    if reviewer_numbers != list(range(1, len(reviewer_numbers) + 1)):
        _fail("Reviewer #N sections must be unique, ordered, and sequential")
    if not quotes:
        _fail("source Markdown must contain at least one reviewer quotation")
    reviewer_starts = [
        index
        for index, block in enumerate(blocks)
        if block.kind == "heading"
        and block.level == 2
        and re.fullmatch(r"Reviewer #[1-9][0-9]*", block.text)
    ]
    for section_index, start in enumerate(reviewer_starts):
        end = (
            reviewer_starts[section_index + 1]
            if section_index + 1 < len(reviewer_starts)
            else len(blocks)
        )
        if not any(block.kind == "quote" for block in blocks[start + 1 : end]):
            _fail("every Reviewer #N section must contain a reviewer quotation")

    nonquote_text = "\n".join(block.text for block in blocks if block.kind != "quote")
    _reject_unresolved_or_unsafe(nonquote_text)
    for block in blocks:
        if block.kind == "heading" and block.level == 1:
            continue
        if block.kind in {"heading", "paragraph"}:
            _validate_inline_text(block.text)
        elif block.kind == "list":
            for item in block.text.split("\n"):
                _validate_inline_text(item)

    response_count = 0
    for index, block in enumerate(blocks):
        if block.kind != "quote":
            continue
        found = False
        for following in blocks[index + 1 :]:
            if following.kind in {"quote", "rule"} or following.kind == "heading":
                break
            if following.kind in {"paragraph", "list"} and following.text.strip():
                found = True
                break
        if not found:
            _fail(f"reviewer quotation {block.marker!r} has no response body")
        response_count += 1
        if response_count > MAX_RESPONSES:
            _fail(f"source Markdown exceeds the {MAX_RESPONSES}-response limit")
    return _MarkdownAudit(
        blocks=tuple(blocks),
        quotes=tuple(quotes),
        reviewer_count=reviewer_count,
        response_count=response_count,
    )


def _validate_title_binding(
    audit: _MarkdownAudit,
    config: Mapping[str, object],
) -> None:
    source_title = audit.blocks[0]
    if source_title.text != config["response_title"]:
        _fail("source level-one heading must equal config.response_title exactly")
    if (
        "*" in source_title.text
        or "`" in source_title.text
        or "\\" in source_title.text
        or "_" in source_title.text
        or "[" in source_title.text
        or "]" in source_title.text
        or "](" in source_title.text
        or HTML_ENTITY_RE.search(source_title.text) is not None
        or re.search(r"\$[^$\n]+\$", source_title.text) is not None
    ):
        _fail("source level-one heading and response title must be literal text")


def _safe_inline(text: str) -> str:
    """Convert a deliberately tiny inline Markdown subset to ReportLab markup."""
    rendered: list[str] = []
    for kind, content, target in _parse_inline_parts(text):
        safe_content = html.escape(content, quote=False)
        if kind in {"plain", "math"}:
            rendered.append(safe_content)
        elif kind in {"code", "bold"}:
            rendered.append(f"<font name='DialectBold'>{safe_content}</font>")
        elif kind == "link":
            rendered.append(
                f"{safe_content} ({html.escape(target, quote=False)})",
            )
        else:  # pragma: no cover - parser exhaustiveness
            _fail(f"unsupported inline part kind {kind!r}")
    return "".join(rendered)


def _plain_visible(text: str) -> str:
    """Return the plain text expected after the supported inline transforms."""
    return "".join(
        f"{content} ({target})" if kind == "link" else content
        for kind, content, target in _parse_inline_parts(text)
    )


def _visible_text_ledger(
    audit: _MarkdownAudit,
    config: Mapping[str, object],
) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []

    def add(block_id: str, kind: str, text: str, *, literal: bool = False) -> None:
        normalized = _normalized_text(text if literal else _plain_visible(text))
        if not normalized:
            _fail(f"visible text ledger entry {block_id!r} is empty")
        if len(records) >= MAX_LEDGER_ENTRIES:
            _fail(
                f"visible text ledger exceeds the {MAX_LEDGER_ENTRIES}-entry limit",
            )
        records.append(
            {
                "block_id": block_id,
                "kind": kind,
                "sha256": _sha256(normalized.encode("utf-8")),
                "text": normalized,
            },
        )

    add(
        "config:response-title",
        "title",
        str(config["response_title"]),
        literal=True,
    )
    add(
        "config:manuscript-id",
        "metadata",
        str(config["manuscript_id"]),
        literal=True,
    )
    add(
        "config:manuscript-title",
        "metadata",
        str(config["manuscript_title"]),
        literal=True,
    )
    add(
        "config:authors",
        "metadata",
        ", ".join(str(author) for author in config["authors"]),
        literal=True,
    )
    for index, block in enumerate(audit.blocks):
        if block.kind == "heading" and block.level == 1:
            continue
        if block.kind in {"heading", "paragraph"}:
            add(
                f"block:{index}",
                block.kind,
                block.text,
            )
        elif block.kind == "quote":
            add(f"block:{index}:reviewer-label", "label", "REVIEWER COMMENT")
            add(f"block:{index}", "quote", block.text, literal=True)
            add(f"block:{index}:response-label", "label", "RESPONSE")
        elif block.kind == "list":
            for item_index, item in enumerate(block.text.split("\n")):
                marker = str(item_index + 1) if block.marker == "ordered" else "•"
                add(
                    f"block:{index}:item:{item_index}",
                    "list-item",
                    f"{marker} {item}",
                )
    return tuple(records)


def _strip_pdf_decorations(
    extracted: str,
    config: Mapping[str, object],
    *,
    page_count: int,
) -> str:
    pages = extracted.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if len(pages) != page_count:
        _fail("pdftotext page separators do not match pdfinfo page count")
    cleaned_pages: list[str] = []
    manuscript_id = str(config["manuscript_id"])
    for page_number, page in enumerate(pages, start=1):
        footer = f"{manuscript_id} Page {page_number}"
        retained: list[str] = []
        matches = 0
        for line in page.splitlines():
            if _normalized_text(line) == footer:
                matches += 1
            else:
                retained.append(line)
        if matches != 1:
            _fail(f"page {page_number} does not contain one exact footer decoration")
        cleaned_pages.append("\n".join(retained))
    return "\n".join(cleaned_pages)


def _verify_visible_text_ledger(
    extracted: str,
    ledger: Sequence[Mapping[str, str]],
) -> None:
    """Require the complete visible body to equal the ordered source ledger."""
    normalized = _normalized_text(extracted)
    expected = " ".join(record["text"] for record in ledger)
    if normalized != expected:
        _fail("rendered visible body differs from the complete ordered text ledger")


def _footer_width_fits(
    manuscript_width: float,
    page_label_width: float,
    usable_width: float,
) -> bool:
    """Return whether the two footer labels retain the fixed 24-point gap."""
    return manuscript_width + page_label_width + 24 <= usable_width


def _render_pdf_bytes(
    canonical_source: bytes,
    template: Mapping[str, object],
    config: Mapping[str, object],
    *,
    regular_font_path: str,
    bold_font_path: str,
) -> bytes:
    """Render canonical inputs.  Called only inside the isolated child."""
    # Imports are intentionally local: the orchestration interpreter does not
    # need ReportLab, and the child prepends a descriptor-backed package bundle.
    from reportlab import rl_config  # type: ignore[import-not-found]
    from reportlab.lib import colors  # type: ignore[import-not-found]
    from reportlab.lib.enums import TA_LEFT  # type: ignore[import-not-found]
    from reportlab.lib.pagesizes import letter  # type: ignore[import-not-found]
    from reportlab.lib.styles import ParagraphStyle  # type: ignore[import-not-found]
    from reportlab.pdfbase import pdfmetrics  # type: ignore[import-not-found]
    from reportlab.pdfbase.ttfonts import TTFont  # type: ignore[import-not-found]
    from reportlab.pdfgen import (
        canvas as canvas_module,  # type: ignore[import-not-found]
    )
    from reportlab.platypus import (  # type: ignore[import-not-found]
        BaseDocTemplate,
        Frame,
        HRFlowable,
        ListFlowable,
        ListItem,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
    )

    audit = _parse_markdown(canonical_source)
    _validate_title_binding(audit, config)
    rl_config.invariant = 1
    rl_config.useA85 = 0
    pdfmetrics.registerFont(TTFont("DialectRegular", regular_font_path, validate=1))
    pdfmetrics.registerFont(TTFont("DialectBold", bold_font_path, validate=1))

    margins = template["margins"]
    type_record = template["type"]
    color_record = template["colors"]
    if (
        not isinstance(margins, dict)
        or not isinstance(type_record, dict)
        or not isinstance(
            color_record,
            dict,
        )
    ):
        _fail("normalized template lost its object structure")
    ink = colors.HexColor(str(color_record["ink"]))
    muted = colors.HexColor(str(color_record["muted"]))
    accent = colors.HexColor(str(color_record["accent"]))
    quote_background = colors.HexColor(str(color_record["quote_background"]))
    rule = colors.HexColor(str(color_record["rule"]))
    body_size = float(type_record["body"])
    leading = float(type_record["leading"])

    styles = {
        "title": ParagraphStyle(
            "Title",
            fontName="DialectBold",
            fontSize=float(type_record["title"]),
            leading=float(type_record["title"]) * 1.15,
            textColor=ink,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="DialectRegular",
            fontSize=10,
            leading=14,
            textColor=muted,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            fontName="DialectBold",
            fontSize=float(type_record["heading_2"]),
            leading=float(type_record["heading_2"]) * 1.22,
            textColor=ink,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            fontName="DialectBold",
            fontSize=float(type_record["heading_3"]),
            leading=float(type_record["heading_3"]) * 1.25,
            textColor=ink,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "Heading4",
            fontName="DialectBold",
            fontSize=float(type_record["heading_4"]),
            leading=float(type_record["heading_4"]) * 1.25,
            textColor=ink,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="DialectRegular",
            fontSize=body_size,
            leading=leading,
            textColor=ink,
            alignment=TA_LEFT,
            spaceAfter=7,
            splitLongWords=True,
        ),
        "response": ParagraphStyle(
            "Response",
            fontName="DialectRegular",
            fontSize=body_size,
            leading=leading,
            textColor=ink,
            leftIndent=11,
            rightIndent=2,
            borderColor=accent,
            borderWidth=0.7,
            borderPadding=(0, 0, 0, 9),
            spaceAfter=7,
            splitLongWords=True,
        ),
        "quote": ParagraphStyle(
            "Quote",
            fontName="DialectRegular",
            fontSize=max(body_size - 0.5, 8.5),
            leading=max(leading - 1, 11.5),
            textColor=ink,
            backColor=quote_background,
            borderColor=rule,
            borderWidth=0.5,
            borderPadding=9,
            leftIndent=5,
            rightIndent=5,
            spaceAfter=9,
            splitLongWords=True,
        ),
        "label": ParagraphStyle(
            "Label",
            fontName="DialectBold",
            fontSize=8,
            leading=10,
            textColor=accent,
            spaceBefore=4,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "list": ParagraphStyle(
            "List",
            parent=None,
            fontName="DialectRegular",
            fontSize=body_size,
            leading=leading,
            textColor=ink,
            leftIndent=4,
        ),
    }

    output = io.BytesIO()
    left = float(margins["left"])
    right = float(margins["right"])
    top = float(margins["top"])
    bottom = float(margins["bottom"])
    manuscript_footer_width = pdfmetrics.stringWidth(
        str(config["manuscript_id"]),
        "DialectRegular",
        8,
    )
    page_footer_width = pdfmetrics.stringWidth(f"Page {MAX_PAGES}", "DialectRegular", 8)
    if not _footer_width_fits(
        manuscript_footer_width,
        page_footer_width,
        letter[0] - left - right,
    ):
        _fail("manuscript_id cannot fit beside the bounded page footer")
    document = BaseDocTemplate(
        output,
        pagesize=letter,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title=str(config["response_title"]),
        author=", ".join(str(author) for author in config["authors"]),
        subject=str(config["manuscript_title"]),
        creator="DIALECT deterministic rebuttal renderer v1",
        showBoundary=0,
    )
    frame = Frame(
        left,
        bottom,
        letter[0] - left - right,
        letter[1] - top - bottom,
        id="body",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    def decorate_page(canvas: object, _document: object) -> None:
        page_canvas = canvas
        page_canvas.saveState()
        page_canvas.setTitle(str(config["response_title"]))
        page_canvas.setAuthor(", ".join(str(author) for author in config["authors"]))
        page_canvas.setSubject(str(config["manuscript_title"]))
        page_canvas.setCreator("DIALECT deterministic rebuttal renderer v1")
        page_canvas.setFont("DialectRegular", 8)
        page_canvas.setFillColor(muted)
        page_canvas.drawString(left, 30, str(config["manuscript_id"]))
        page_canvas.drawRightString(
            letter[0] - right, 30, f"Page {page_canvas.getPageNumber()}"
        )
        page_canvas.setStrokeColor(rule)
        page_canvas.setLineWidth(0.5)
        page_canvas.line(left, letter[1] - 36, letter[0] - right, letter[1] - 36)
        page_canvas.restoreState()

    document.addPageTemplates(
        [PageTemplate(id="response", frames=[frame], onPage=decorate_page)]
    )
    story: list[object] = []
    story.append(
        Paragraph(
            html.escape(str(config["response_title"]), quote=False),
            styles["title"],
        ),
    )
    story.append(
        Paragraph(
            html.escape(str(config["manuscript_id"]), quote=False),
            styles["subtitle"],
        ),
    )
    story.append(
        Paragraph(
            html.escape(str(config["manuscript_title"]), quote=False),
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            html.escape(
                ", ".join(str(author) for author in config["authors"]),
                quote=False,
            ),
            styles["subtitle"],
        ),
    )
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.8, color=accent, spaceAfter=12))

    after_quote = False
    reviewer_seen = 0
    for block in audit.blocks:
        if block.kind == "heading":
            if block.level == 1:
                continue
            if block.level == 2 and re.fullmatch(r"Reviewer #[1-9][0-9]*", block.text):
                reviewer_seen += 1
                if reviewer_seen > 1:
                    story.append(PageBreak())
            style = styles[f"h{block.level}"]
            story.append(Paragraph(_safe_inline(block.text), style))
            after_quote = False
        elif block.kind == "quote":
            story.append(Paragraph("REVIEWER COMMENT", styles["label"]))
            quote_markup = "<br/>".join(
                html.escape(line, quote=False) for line in block.text.split("\n")
            )
            story.append(Paragraph(quote_markup, styles["quote"]))
            story.append(Paragraph("RESPONSE", styles["label"]))
            after_quote = True
        elif block.kind == "paragraph":
            story.append(
                Paragraph(
                    _safe_inline(block.text),
                    styles["response"] if after_quote else styles["body"],
                ),
            )
        elif block.kind == "list":
            items = [
                ListItem(Paragraph(_safe_inline(item), styles["list"]))
                for item in block.text.split("\n")
            ]
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if block.marker == "ordered" else "bullet",
                    start="1" if block.marker == "ordered" else None,
                    leftIndent=23 if after_quote else 18,
                    bulletFontName="DialectRegular",
                    bulletFontSize=body_size,
                    spaceAfter=7,
                ),
            )
        elif block.kind == "rule":
            story.append(
                HRFlowable(
                    width="100%", thickness=0.5, color=rule, spaceBefore=5, spaceAfter=7
                )
            )
            after_quote = False
        else:  # pragma: no cover - parser exhaustiveness
            _fail(f"unsupported parsed block kind {block.kind!r}")

    def deterministic_canvas(*args: object, **kwargs: object) -> object:
        kwargs["invariant"] = 1
        kwargs["pageCompression"] = 1
        kwargs["initialFontName"] = "DialectRegular"
        kwargs["initialFontSize"] = body_size
        kwargs["initialLeading"] = leading
        return canvas_module.Canvas(*args, **kwargs)

    document.build(story, canvasmaker=deterministic_canvas)
    raw = output.getvalue()
    if len(raw) > MAX_PDF_BYTES:
        _fail(f"rendered PDF exceeds the {MAX_PDF_BYTES}-byte limit")
    return raw


def _read_fd(descriptor: int, *, maximum: int, context: str) -> bytes:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        _fail(f"cannot seek {context}: {error}")
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            block = os.read(descriptor, READ_CHUNK_BYTES)
        except OSError as error:
            _fail(f"cannot read {context}: {error}")
        if not block:
            break
        total += len(block)
        if total > maximum:
            _fail(f"{context} exceeds the {maximum}-byte limit")
        chunks.append(block)
    return b"".join(chunks)


def _write_all(descriptor: int, raw: bytes, *, context: str) -> None:
    written = 0
    while written < len(raw):
        try:
            count = os.write(descriptor, raw[written:])
        except OSError as error:
            _fail(f"cannot write {context}: {error}")
        if count <= 0:
            _fail(f"cannot make progress while writing {context}")
        written += count


def _close_owned_file(
    handle: object,
    *,
    context: str,
    primary_error: BaseException | None,
) -> None:
    try:
        handle.close()
    except BaseException as error:  # noqa: BLE001 - preserve both failures.
        message = f"{context} cleanup failed: {error}"
        if primary_error is not None:
            message = f"{primary_error}; {message}"
        raise RebuttalRenderError(message) from primary_error


def _install_no_image_pillow_stub() -> tuple[types.ModuleType, _NoImageModule]:
    """Install the minimal import-only PIL surface required by ReportLab."""
    if "PIL" in sys.modules or "PIL.Image" in sys.modules:
        _fail("image support was loaded before the fail-closed renderer boundary")
    pillow_stub = types.ModuleType("PIL")
    pillow_stub.__path__ = []  # type: ignore[attr-defined]
    image_stub = _NoImageModule("PIL.Image")
    pillow_stub.Image = image_stub  # type: ignore[attr-defined]
    sys.modules["PIL"] = pillow_stub
    sys.modules["PIL.Image"] = image_stub
    return pillow_stub, image_stub


REPORTLAB_BLOCKED_OPTIONAL_MODULES: Final = (
    "_rl_accel",
    "_rl_renderPM",
    "_renderPM",
    "sgmlop",
    "pyRXP",
    "pyRXPU",
    "uharfbuzz",
    "rlbidi",
    "pyphen",
)


def _install_reportlab_import_guards() -> tuple[types.ModuleType, types.ModuleType]:
    """Preseed inert customization modules and fail-closed optional natives."""
    guarded = (
        "reportlab_mods",
        "reportlab_settings",
        *REPORTLAB_BLOCKED_OPTIONAL_MODULES,
    )
    occupied = [name for name in guarded if name in sys.modules]
    if occupied:
        _fail(f"ReportLab import guard names were already occupied: {occupied}")
    mods_stub = types.ModuleType("reportlab_mods")
    settings_stub = types.ModuleType("reportlab_settings")
    settings_stub.T1SearchPath = []  # type: ignore[attr-defined]
    settings_stub.TTFSearchPath = []  # type: ignore[attr-defined]
    settings_stub.CMapSearchPath = []  # type: ignore[attr-defined]
    sys.modules["reportlab_mods"] = mods_stub
    sys.modules["reportlab_settings"] = settings_stub
    for name in REPORTLAB_BLOCKED_OPTIONAL_MODULES:
        sys.modules[name] = None
    return mods_stub, settings_stub


def _audit_reportlab_runtime(
    reportlab_bundle: str,
    mods_stub: types.ModuleType,
    settings_stub: types.ModuleType,
) -> None:
    """Prove the imported ReportLab surface stayed pure Python and descriptor-bound."""
    from importlib.machinery import ExtensionFileLoader

    if (
        sys.modules.get("reportlab_mods") is not mods_stub
        or sys.modules.get("reportlab_settings") is not settings_stub
        or getattr(settings_stub, "T1SearchPath", None) != []
        or getattr(settings_stub, "TTFSearchPath", None) != []
        or getattr(settings_stub, "CMapSearchPath", None) != []
    ):
        _fail("ReportLab customization guards were replaced during rendering")
    if any(
        sys.modules.get(name, object()) is not None
        for name in REPORTLAB_BLOCKED_OPTIONAL_MODULES
    ):
        _fail("ReportLab loaded an optional native accelerator or parser")
    configuration = sys.modules.get("reportlab.rl_config")
    if configuration is None or any(
        getattr(configuration, name, None) != []
        for name in ("T1SearchPath", "TTFSearchPath", "CMapSearchPath")
    ):
        _fail("ReportLab effective font and CMap search paths are not empty")
    accelerator = sys.modules.get("reportlab.lib.rl_accel")
    if accelerator is None:
        _fail("ReportLab pure-Python accelerator facade was not imported")
    c_functions = getattr(accelerator, "_c_funcs", None)
    python_functions = getattr(accelerator, "_py_funcs", None)
    exports = getattr(accelerator, "__all__", None)
    if (
        c_functions != {}
        or not isinstance(python_functions, dict)
        or not isinstance(
            exports,
            list,
        )
    ):
        _fail("ReportLab accelerator inventory is not the pure-Python fallback")
    if not exports or any(
        name not in python_functions
        or python_functions[name] is None
        or getattr(accelerator, name, None) is not python_functions[name]
        for name in exports
    ):
        _fail("ReportLab accelerator exports are not bound to pure-Python functions")
    for name, module in tuple(sys.modules.items()):
        if name != "reportlab" and not name.startswith("reportlab."):
            continue
        if module is None:
            _fail(f"loaded ReportLab module {name!r} is missing")
        module_path = getattr(module, "__file__", None)
        loader = getattr(module, "__loader__", None)
        if (
            not isinstance(module_path, str)
            or not module_path.startswith(f"{reportlab_bundle}/reportlab/")
            or module_path.endswith((".so", ".dylib"))
            or isinstance(loader, ExtensionFileLoader)
        ):
            _fail(f"ReportLab module {name!r} escaped the pure-Python bundle")


def _internal_render(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-fd", required=True, type=int)
    parser.add_argument("--template-fd", required=True, type=int)
    parser.add_argument("--config-fd", required=True, type=int)
    parser.add_argument("--regular-font-fd", required=True, type=int)
    parser.add_argument("--bold-font-fd", required=True, type=int)
    parser.add_argument("--reportlab-bundle-fd", required=True, type=int)
    parser.add_argument("--source-date-epoch", required=True, type=int)
    parsed = parser.parse_args(arguments)
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        _fail("internal renderer requires isolated, no-site, no-bytecode Python")
    if sys.pycache_prefix != "/dev/null/dialect-rebuttal-pycache":
        _fail("internal renderer has an unexpected pycache prefix")
    # Darwin injects this CoreFoundation variable even when posix_spawn receives
    # an otherwise exact environment. Validate its narrow system form, then
    # remove it before importing the renderer dependency or consuming inputs.
    cf_encoding = os.environ.pop("__CF_USER_TEXT_ENCODING", None)
    if (
        cf_encoding is not None
        and re.fullmatch(
            r"0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+:0x[0-9A-Fa-f]+",
            cf_encoding,
        )
        is None
    ):
        _fail("internal renderer received a malformed Darwin encoding variable")
    if os.environ != {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}:
        _fail("internal renderer environment is not exact")
    source = _canonicalize_markdown(
        _read_fd(
            parsed.source_fd, maximum=MAX_SOURCE_BYTES, context="source descriptor"
        ),
    )
    template = _normalize_template(
        _json_without_duplicates(
            _read_fd(
                parsed.template_fd,
                maximum=MAX_JSON_BYTES,
                context="template descriptor",
            ),
            context="template",
        ),
    )
    config = _normalize_config(
        _json_without_duplicates(
            _read_fd(
                parsed.config_fd, maximum=MAX_JSON_BYTES, context="config descriptor"
            ),
            context="config",
        ),
    )
    if parsed.source_date_epoch != config["source_date_epoch"]:
        _fail("internal source-date epoch does not match canonical config")
    reportlab_bundle = f"/dev/fd/{parsed.reportlab_bundle_fd}"
    sys.path.insert(0, reportlab_bundle)
    mods_stub, settings_stub = _install_reportlab_import_guards()
    # reportlab.lib.utils imports PIL.Image unconditionally, although this
    # renderer rejects every image construct and never calls its image helpers.
    # Supply an import-only module: any unexpected image access fails instead
    # of loading Pillow's unattested native extension closure.
    pillow_stub, image_stub = _install_no_image_pillow_stub()
    regular_font = f"/dev/fd/{parsed.regular_font_fd}"
    bold_font = f"/dev/fd/{parsed.bold_font_fd}"
    rendered = _render_pdf_bytes(
        source,
        template,
        config,
        regular_font_path=regular_font,
        bold_font_path=bold_font,
    )
    imported = sys.modules.get("reportlab")
    imported_path = getattr(imported, "__file__", "")
    if not isinstance(imported_path, str) or not imported_path.startswith(
        f"{reportlab_bundle}/reportlab/",
    ):
        _fail("ReportLab was not imported from the descriptor-backed bundle")
    if (
        sys.modules.get("PIL") is not pillow_stub
        or sys.modules.get("PIL.Image") is not image_stub
        or any(
            name == "_imaging" or (name.startswith("PIL.") and name != "PIL.Image")
            for name in sys.modules
        )
    ):
        _fail("renderer loaded a Pillow or native imaging module")
    _audit_reportlab_runtime(reportlab_bundle, mods_stub, settings_stub)
    _write_all(1, rendered, context="rendered PDF stdout")
    return 0


def _bounded_directory_names(
    directory: int | Path | str,
    *,
    maximum: int,
    context: str,
) -> list[str]:
    if maximum <= 0:
        _fail(f"{context} enumeration bound must be positive")
    names: list[str] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > maximum:
                    _fail(f"{context} exceeds the {maximum}-entry bound")
    except OSError as error:
        _fail(f"cannot enumerate {context}: {error}")
    return sorted(names)


def _close_descriptor(
    descriptor: int,
    *,
    context: str,
    primary_error: BaseException | None,
) -> None:
    try:
        os.close(descriptor)
    except BaseException as error:  # noqa: BLE001 - preserve both failures.
        message = f"{context} descriptor cleanup failed: {error}"
        if primary_error is not None:
            message = f"{primary_error}; {message}"
        raise RebuttalRenderError(message) from primary_error


def _inventory_reportlab(root: Path) -> _ReportLabBundle:
    absolute = root.absolute()
    try:
        root_lstat = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect ReportLab package root: {error}")
    if (
        stat.S_ISLNK(root_lstat.st_mode)
        or not stat.S_ISDIR(root_lstat.st_mode)
        or resolved != absolute
        or absolute.name != "reportlab"
    ):
        _fail("ReportLab root must be a canonical directory named 'reportlab'")
    records: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    total = 0
    scanned_directory_count = 0
    scanned_entry_count = 0
    seen_identities: set[tuple[int, int]] = set()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(absolute, directory_flags)

    def visit(
        directory_descriptor: int,
        relative_parts: tuple[str, ...],
        depth: int,
    ) -> None:
        nonlocal scanned_directory_count, scanned_entry_count, total
        directory_before = os.fstat(directory_descriptor)
        if depth > MAX_REPORTLAB_DEPTH:
            _fail(f"ReportLab tree exceeds depth {MAX_REPORTLAB_DEPTH}")
        scanned_directory_count += 1
        if scanned_directory_count > MAX_REPORTLAB_DIRECTORIES:
            _fail(
                "ReportLab tree exceeds the "
                f"{MAX_REPORTLAB_DIRECTORIES}-directory bound",
            )
        names: list[str] = []
        try:
            with os.scandir(directory_descriptor) as entries:
                for item in entries:
                    scanned_entry_count += 1
                    if scanned_entry_count > MAX_REPORTLAB_ENTRIES:
                        _fail(
                            "ReportLab tree exceeds the "
                            f"{MAX_REPORTLAB_ENTRIES}-entry bound",
                        )
                    names.append(item.name)
        except OSError as error:
            _fail(f"cannot enumerate ReportLab directory: {error}")
        for name in sorted(names):
            if name in {"", ".", ".."} or "/" in name or "\\" in name:
                _fail("ReportLab tree contains a noncanonical entry name")
            if not relative_parts and name in {
                "local_rl_mods",
                "local_rl_mods.py",
                "local_rl_settings",
                "local_rl_settings.py",
            }:
                _fail("ReportLab bundle contains a local customization module")
            try:
                entry = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                _fail(f"cannot inspect ReportLab entry {name!r}: {error}")
            if stat.S_ISLNK(entry.st_mode):
                _fail("ReportLab tree contains a symlink")
            child_parts = (*relative_parts, name)
            if stat.S_ISDIR(entry.st_mode):
                if name == "__pycache__":
                    continue
                child = os.open(name, directory_flags, dir_fd=directory_descriptor)
                try:
                    pinned_directory = os.fstat(child)
                    expected_directory = (
                        entry.st_dev,
                        entry.st_ino,
                        entry.st_size,
                        entry.st_mtime_ns,
                    )
                    if (
                        pinned_directory.st_dev,
                        pinned_directory.st_ino,
                        pinned_directory.st_size,
                        pinned_directory.st_mtime_ns,
                    ) != expected_directory:
                        _fail("ReportLab directory changed while pinned")
                    visit(child, child_parts, depth + 1)
                    after_directory = os.fstat(child)
                    named_directory = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        after_directory.st_dev,
                        after_directory.st_ino,
                        after_directory.st_size,
                        after_directory.st_mtime_ns,
                    ) != expected_directory or (
                        named_directory.st_dev,
                        named_directory.st_ino,
                        named_directory.st_size,
                        named_directory.st_mtime_ns,
                    ) != expected_directory:
                        _fail("ReportLab directory changed during traversal")
                finally:
                    _close_descriptor(
                        child,
                        context="ReportLab child directory",
                        primary_error=sys.exception(),
                    )
                continue
            if not stat.S_ISREG(entry.st_mode):
                _fail("ReportLab tree contains a non-regular entry")
            suffix = PurePosixPath(name).suffix
            if suffix in {".pyc", ".so", ".dylib"}:
                continue
            if entry.st_nlink != 1:
                _fail("ReportLab tree contains a non-single-link file")
            identity = (entry.st_dev, entry.st_ino)
            if identity in seen_identities:
                _fail("ReportLab package aliases one member inode")
            seen_identities.add(identity)
            file_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            relative = PurePosixPath(*child_parts).as_posix()
            descriptor = os.open(name, file_flags, dir_fd=directory_descriptor)
            try:
                pinned = os.fstat(descriptor)
                expected_identity = (
                    entry.st_dev,
                    entry.st_ino,
                    entry.st_size,
                    entry.st_mtime_ns,
                )
                if (
                    not stat.S_ISREG(pinned.st_mode)
                    or pinned.st_nlink != 1
                    or (
                        pinned.st_dev,
                        pinned.st_ino,
                        pinned.st_size,
                        pinned.st_mtime_ns,
                    )
                    != expected_identity
                ):
                    _fail(f"ReportLab member {relative!r} changed while pinned")
                raw = _read_fd(
                    descriptor,
                    maximum=MAX_REPORTLAB_BYTES - total,
                    context=f"ReportLab member {relative}",
                )
                named_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    os.fstat(descriptor).st_size != len(raw)
                    or (
                        named_after.st_dev,
                        named_after.st_ino,
                        named_after.st_size,
                        named_after.st_mtime_ns,
                    )
                    != expected_identity
                ):
                    _fail(f"ReportLab member {relative!r} changed while read")
            finally:
                _close_descriptor(
                    descriptor,
                    context="ReportLab member",
                    primary_error=sys.exception(),
                )
            total += len(raw)
            if total > MAX_REPORTLAB_BYTES:
                _fail("ReportLab tree exceeds its aggregate byte limit")
            member = f"reportlab/{relative}"
            records.append(
                {"member": member, "sha256": _sha256(raw), "size": len(raw)},
            )
            payloads.append((member, raw))
            if len(records) > MAX_REPORTLAB_FILES:
                _fail("ReportLab tree exceeds its file-count limit")
        directory_after = os.fstat(directory_descriptor)
        if (
            directory_after.st_dev,
            directory_after.st_ino,
            directory_after.st_size,
            directory_after.st_mtime_ns,
        ) != (
            directory_before.st_dev,
            directory_before.st_ino,
            directory_before.st_size,
            directory_before.st_mtime_ns,
        ):
            _fail("ReportLab directory changed during bounded enumeration")

    try:
        pinned_root = os.fstat(root_descriptor)
        if (pinned_root.st_dev, pinned_root.st_ino) != (
            root_lstat.st_dev,
            root_lstat.st_ino,
        ):
            _fail("ReportLab root changed while pinned")
        visit(root_descriptor, (), 0)
        named_after = os.lstat(absolute)
        current_root = os.fstat(root_descriptor)
        expected_root = (
            root_lstat.st_dev,
            root_lstat.st_ino,
            root_lstat.st_size,
            root_lstat.st_mtime_ns,
        )
        if (
            current_root.st_dev,
            current_root.st_ino,
            current_root.st_size,
            current_root.st_mtime_ns,
        ) != expected_root or (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_size,
            named_after.st_mtime_ns,
        ) != expected_root:
            _fail("ReportLab root changed during traversal")
    finally:
        _close_descriptor(
            root_descriptor,
            context="ReportLab root",
            primary_error=sys.exception(),
        )
    if not records or "reportlab/__init__.py" not in {
        str(record["member"]) for record in records
    }:
        _fail("ReportLab tree lacks reportlab/__init__.py")
    included_directories: set[str] = set()
    for record in records:
        parts = PurePosixPath(str(record["member"])).parts[:-1]
        included_directories.update(
            PurePosixPath(*parts[:depth]).as_posix()
            for depth in range(1, len(parts) + 1)
        )
    directory_count = len(included_directories)
    entry_count = len(records) + directory_count
    tree_sha256 = _sha256(_canonical_json(records))
    archive = io.BytesIO()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as bundle:
        for member, raw in payloads:
            info = zipfile.ZipInfo(member, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100400 << 16
            bundle.writestr(info, raw)
    bundle_raw = archive.getvalue()
    if len(bundle_raw) > MAX_REPORTLAB_BUNDLE_BYTES:
        _fail("ReportLab descriptor bundle exceeds its byte limit")
    return _ReportLabBundle(
        raw=bundle_raw,
        tree_sha256=tree_sha256,
        file_count=len(records),
        directory_count=directory_count,
        entry_count=entry_count,
        total_bytes=total,
        records=tuple(records),
    )


def reportlab_dependency_digest(root: Path) -> dict[str, object]:
    """Return the canonical caller anchor for one pure-Python ReportLab tree."""
    bundle = _inventory_reportlab(root)
    return {
        "schema": "dialect-reportlab-pure-python-tree-anchor-v1",
        "tree_sha256": bundle.tree_sha256,
        "file_count": bundle.file_count,
        "directory_count": bundle.directory_count,
        "entry_count": bundle.entry_count,
        "total_bytes": bundle.total_bytes,
        "excluded": ["__pycache__", "*.pyc", "*.so", "*.dylib"],
    }


def _machine_runner_path() -> Path:
    return _builder_path().with_name(
        "build_tcga_revision_rendered_document_machine_closure.py"
    )


def _load_machine_authority(
    path: Path,
    *,
    expected_sha256: str,
) -> _MachineAuthority:
    """Hash a fixed helper before compiling its pinned descriptor bytes."""
    expected = _expect_sha256(
        expected_sha256,
        context="expected machine-runner SHA-256",
    )
    absolute = path.absolute()
    try:
        named = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect machine runner: {error}")
    if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or named.st_size > MAX_MACHINE_RUNNER_BYTES
        or resolved != absolute
    ):
        _fail("machine runner must be a canonical single-link regular file")
    descriptor = os.open(
        absolute,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        pinned = os.fstat(descriptor)
        identity = (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
        )
        if (
            pinned.st_dev,
            pinned.st_ino,
            pinned.st_size,
            pinned.st_mtime_ns,
        ) != identity:
            _fail("machine runner changed while it was pinned")
        raw = _read_fd(
            descriptor,
            maximum=MAX_MACHINE_RUNNER_BYTES,
            context="machine runner",
        )
        digest = _sha256(raw)
        named_after = os.lstat(absolute)
        if (
            os.fstat(descriptor).st_size != len(raw)
            or (
                named_after.st_dev,
                named_after.st_ino,
                named_after.st_size,
                named_after.st_mtime_ns,
            )
            != identity
        ):
            _fail("machine runner changed while it was read")
        if digest != expected:
            _fail("machine runner does not match its caller SHA-256 anchor")
        module_name = f"_dialect_pinned_machine_runner_{digest}"
        missing_module = object()
        previous_module = sys.modules.get(module_name, missing_module)
        module = types.ModuleType(module_name)
        module.__file__ = f"/dev/fd/{descriptor}"
        module.__package__ = "analysis"
        code = compile(
            raw,
            f"<pinned-machine-runner:{digest}>",
            "exec",
            dont_inherit=True,
            optimize=0,
        )
        sys.modules[module_name] = module
        try:
            exec(code, module.__dict__)  # noqa: S102 - caller-hash-pinned source.
        finally:
            if sys.modules.get(module_name) is module:
                if previous_module is missing_module:
                    sys.modules.pop(module_name)
                else:
                    sys.modules[module_name] = previous_module
        required_attributes = (
            "_pin_file",
            "_run_bounded",
            "DARWIN_SPAWN_FLAGS",
            "REQUIRED_CS_FLAGS",
            "REJECTED_CS_FLAGS",
            "VM_PROT_EXECUTE",
        )
        if any(not hasattr(module, attribute) for attribute in required_attributes):
            _fail("machine runner lacks the fixed private adapter surface")
        return _MachineAuthority(
            path=absolute,
            descriptor=descriptor,
            device=pinned.st_dev,
            inode=pinned.st_ino,
            size=pinned.st_size,
            mtime_ns=pinned.st_mtime_ns,
            sha256=digest,
            module=module,
        )
    except BaseException as error:
        _close_descriptor(
            descriptor,
            context="machine runner",
            primary_error=error,
        )
        raise


def _raise_machine_failure(context: str, error: BaseException) -> NoReturn:
    if isinstance(error, RebuttalRenderError):
        raise error
    message = f"{context}: {error}"
    raise RebuttalRenderError(message) from error


def _pin_inputs(
    source: Path,
    template: Path,
    config: Path,
    regular_font: Path,
    bold_font: Path,
    builder: Path,
    tool_paths: Mapping[str, Path],
    expected_machine_runner_sha256: str,
) -> dict[str, object]:
    pins: dict[str, object] = {}
    specifications = (
        ("source", source, MAX_SOURCE_BYTES),
        ("template", template, MAX_JSON_BYTES),
        ("config", config, MAX_JSON_BYTES),
        ("regular_font", regular_font, MAX_FONT_BYTES),
        ("bold_font", bold_font, MAX_FONT_BYTES),
        ("builder", builder, MAX_SOURCE_BYTES),
    )
    try:
        authority = _load_machine_authority(
            _machine_runner_path(),
            expected_sha256=expected_machine_runner_sha256,
        )
        pins["machine_runner"] = authority
        machine = authority.module
        for name, path, maximum in specifications:
            _revalidate_pinned_file(authority, context="machine runner")
            pins[name] = machine._pin_file(
                path, maximum=maximum, context=name.replace("_", " ")
            )
            _revalidate_pinned_file(authority, context="machine runner")
        if set(tool_paths) != set(TOOL_ORDER):
            _fail(f"tool paths must contain exactly {list(TOOL_ORDER)}")
        for name in TOOL_ORDER:
            _revalidate_pinned_file(authority, context="machine runner")
            pins[f"tool:{name}"] = machine._pin_file(
                tool_paths[name],
                maximum=128 * 1024 * 1024,
                context=f"{name} executable",
            )
            _revalidate_pinned_file(authority, context="machine runner")
        identities: dict[tuple[int, int], str] = {}
        for name, pin in pins.items():
            identity = (pin.device, pin.inode)
            if identity in identities:
                _fail(f"{name} aliases {identities[identity]}")
            identities[identity] = name
        return pins
    except BaseException as error:  # noqa: BLE001 - close every acquired pin.
        _close_pins(pins, primary_error=error)
        _raise_machine_failure("cannot pin renderer inputs", error)


def _resolve_fixed_derivation_locator(locator: str) -> Path:
    if locator not in DERIVATION_FIXED_LOCATOR_PATHS:
        _fail(f"unknown fixed derivation locator {locator!r}")
    try:
        return DERIVATION_FIXED_LOCATOR_PATHS[locator].resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve fixed derivation locator {locator!r}: {error}")


def _derive_reportlab_path(runtime: Path, python_tag: str) -> Path:
    candidate = (
        runtime.parent.parent
        / "lib"
        / f"python{python_tag}"
        / "site-packages"
        / "reportlab"
    ).absolute()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve invoking-Python ReportLab locator: {error}")
    if resolved != candidate:
        _fail("invoking-Python ReportLab locator is not canonical")
    return resolved


def _pin_derivation_dependencies(
    bundle: Mapping[str, object],
) -> tuple[dict[str, object], Path]:
    dependencies = bundle["dependencies"]
    runtime_record = dependencies["runtime"]
    python_tag = str(runtime_record["python_tag"])
    running_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_tag != running_tag:
        _fail("derivation bundle Python tag does not match the invoking runtime")
    try:
        runtime = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve the invoking Python runtime: {error}")
    reportlab_root = _derive_reportlab_path(runtime, python_tag)
    fonts = {str(item["role"]): item for item in dependencies["fonts"]}
    tools = {str(item["name"]): item for item in dependencies["tools"]}
    tool_paths = {
        "python": runtime,
        **{
            name: _resolve_fixed_derivation_locator(str(tools[name]["locator"]))
            for name in TOOL_ORDER[1:]
        },
    }
    specifications = (
        (
            "regular_font",
            _resolve_fixed_derivation_locator(str(fonts["regular"]["locator"])),
            MAX_FONT_BYTES,
        ),
        (
            "bold_font",
            _resolve_fixed_derivation_locator(str(fonts["bold"]["locator"])),
            MAX_FONT_BYTES,
        ),
        ("builder", _builder_path(), MAX_SOURCE_BYTES),
    )
    pins: dict[str, object] = {}
    try:
        authority = _load_machine_authority(
            _machine_runner_path(),
            expected_sha256=str(dependencies["machine_runner"]["sha256"]),
        )
        pins["machine_runner"] = authority
        machine = authority.module
        for name, path, maximum in specifications:
            _revalidate_pinned_file(authority, context="machine runner")
            pins[name] = machine._pin_file(
                path, maximum=maximum, context=name.replace("_", " ")
            )
            _revalidate_pinned_file(authority, context="machine runner")
        for name in TOOL_ORDER:
            _revalidate_pinned_file(authority, context="machine runner")
            pins[f"tool:{name}"] = machine._pin_file(
                tool_paths[name],
                maximum=128 * 1024 * 1024,
                context=f"{name} executable",
            )
            _revalidate_pinned_file(authority, context="machine runner")
        identities: dict[tuple[int, int], str] = {}
        for name, pin in pins.items():
            identity = (pin.device, pin.inode)
            if identity in identities:
                _fail(f"{name} aliases {identities[identity]}")
            identities[identity] = name
        return pins, reportlab_root
    except BaseException as error:  # noqa: BLE001 - close every acquired pin.
        _close_pins(pins, primary_error=error)
        _raise_machine_failure("cannot pin derivation dependencies", error)


def _close_pins(
    pins: Mapping[str, object],
    *,
    primary_error: BaseException | None = None,
) -> None:
    errors: list[BaseException] = []
    for pin in pins.values():
        try:
            pin.close()
        except BaseException as error:  # noqa: BLE001 - close every descriptor.
            errors.append(error)
    if errors:
        message = "input descriptor cleanup failed: " + "; ".join(
            str(error) for error in errors
        )
        if primary_error is not None:
            message = f"{primary_error}; {message}"
        raise RebuttalRenderError(message) from primary_error


def _validate_expected_hashes(
    pins: Mapping[str, object],
    *,
    expected_source_sha256: str,
    expected_template_sha256: str,
    expected_config_sha256: str,
    expected_regular_font_sha256: str,
    expected_bold_font_sha256: str,
    expected_machine_runner_sha256: str,
    expected_builder_sha256: str,
    expected_tool_sha256: Mapping[str, str],
) -> None:
    expected = {
        "source": expected_source_sha256,
        "template": expected_template_sha256,
        "config": expected_config_sha256,
        "regular_font": expected_regular_font_sha256,
        "bold_font": expected_bold_font_sha256,
        "machine_runner": expected_machine_runner_sha256,
        "builder": expected_builder_sha256,
    }
    if set(expected_tool_sha256) != set(TOOL_ORDER):
        _fail(f"tool SHA anchors must contain exactly {list(TOOL_ORDER)}")
    for name, digest in expected.items():
        _expect_sha256(digest, context=f"expected {name} SHA-256")
        if pins[name].sha256 != digest:
            _fail(f"{name} does not match its caller SHA-256 anchor")
    for name in TOOL_ORDER:
        digest = expected_tool_sha256[name]
        _expect_sha256(digest, context=f"expected {name} SHA-256")
        if pins[f"tool:{name}"].sha256 != digest:
            _fail(f"{name} executable does not match its caller SHA-256 anchor")
    if pins["regular_font"].sha256 == pins["bold_font"].sha256:
        _fail("regular and bold font roles must have distinct pinned bytes")


def _revalidate_pinned_file(pin: object, *, context: str) -> None:
    _read_anchored_pin(pin, maximum=max(pin.size, 1), context=context)


def _read_anchored_pin(pin: object, *, maximum: int, context: str) -> bytes:
    """Read and hash the exact bytes whose named identity is checked around them."""
    try:
        entry_before = os.fstat(pin.descriptor)
        named_before = os.lstat(pin.path)
        resolved = pin.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} disappeared after pinning: {error}")
    expected = (pin.device, pin.inode, pin.size, pin.mtime_ns)
    if (
        (
            entry_before.st_dev,
            entry_before.st_ino,
            entry_before.st_size,
            entry_before.st_mtime_ns,
        )
        != expected
        or (
            named_before.st_dev,
            named_before.st_ino,
            named_before.st_size,
            named_before.st_mtime_ns,
        )
        != expected
        or (
            not stat.S_ISREG(entry_before.st_mode)
            or not stat.S_ISREG(named_before.st_mode)
            or stat.S_ISLNK(named_before.st_mode)
            or entry_before.st_nlink != 1
            or named_before.st_nlink != 1
            or resolved != pin.path
        )
    ):
        _fail(f"{context} identity changed after pinning")
    raw = _read_fd(pin.descriptor, maximum=maximum, context=context)
    try:
        entry_after = os.fstat(pin.descriptor)
        named_after = os.lstat(pin.path)
        resolved_after = pin.path.resolve(strict=True)
    except OSError as error:
        _fail(f"{context} disappeared while being read: {error}")
    if (
        (
            entry_after.st_dev,
            entry_after.st_ino,
            entry_after.st_size,
            entry_after.st_mtime_ns,
        )
        != expected
        or (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_size,
            named_after.st_mtime_ns,
        )
        != expected
        or resolved_after != pin.path
        or len(raw) != pin.size
    ):
        _fail(f"{context} identity changed while being read")
    if _sha256(raw) != pin.sha256:
        _fail(f"{context} bytes changed after pinning")
    return raw


def _pinned_bytes(pin: object, *, maximum: int, context: str) -> bytes:
    return _read_anchored_pin(pin, maximum=maximum, context=context)


def _snapshot_bytes(raw: bytes, *, context: str) -> _Snapshot:
    """Copy verified bytes into a private unlinked read-only descriptor."""
    writable = -1
    readable = -1
    path: str | None = None
    try:
        writable, path = tempfile.mkstemp(prefix="dialect-rebuttal-snapshot-")
        created = os.fstat(writable)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            _fail(f"cannot create a private regular {context} snapshot")
        _write_all(writable, raw, context=f"{context} snapshot")
        os.fsync(writable)
        os.fchmod(writable, 0o400)
        readable = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(readable)
        named = os.lstat(path)
        expected = (created.st_dev, created.st_ino, len(raw))
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o400
            or (opened.st_dev, opened.st_ino, opened.st_size) != expected
            or (named.st_dev, named.st_ino, named.st_size) != expected
        ):
            _fail(f"{context} snapshot changed before it was sealed")
        Path(path).unlink()
        path = None
        os.close(writable)
        writable = -1
        snapshot = _Snapshot(
            descriptor=readable,
            size=len(raw),
            sha256=_sha256(raw),
        )
        readable = -1
        _revalidate_snapshot(snapshot, context=context)
        return snapshot
    except BaseException as error:
        cleanup: list[BaseException] = []
        for descriptor in (readable, writable):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException as close_error:  # noqa: BLE001
                    cleanup.append(close_error)
        if path is not None:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
            except BaseException as unlink_error:  # noqa: BLE001
                cleanup.append(unlink_error)
        if cleanup:
            message = f"{error}; snapshot cleanup failed: {cleanup}"
            raise RebuttalRenderError(message) from error
        raise


def _revalidate_snapshot(snapshot: _Snapshot, *, context: str) -> None:
    try:
        entry = os.fstat(snapshot.descriptor)
    except OSError as error:
        _fail(f"{context} snapshot descriptor is unavailable: {error}")
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 0
        or stat.S_IMODE(entry.st_mode) != 0o400
        or entry.st_size != snapshot.size
    ):
        _fail(f"{context} snapshot is not an unlinked sealed regular file")
    if (
        _sha256(
            _read_fd(
                snapshot.descriptor,
                maximum=max(snapshot.size, 1),
                context=f"{context} snapshot",
            ),
        )
        != snapshot.sha256
    ):
        _fail(f"{context} snapshot bytes changed")


def _snapshot_inputs(pins: Mapping[str, object]) -> dict[str, _Snapshot]:
    snapshots: dict[str, _Snapshot] = {}
    limits = {
        "source": MAX_SOURCE_BYTES,
        "template": MAX_JSON_BYTES,
        "config": MAX_JSON_BYTES,
        "regular_font": MAX_FONT_BYTES,
        "bold_font": MAX_FONT_BYTES,
        "builder": MAX_SOURCE_BYTES,
    }
    try:
        for name, maximum in limits.items():
            raw = _read_anchored_pin(pins[name], maximum=maximum, context=name)
            snapshots[name] = _snapshot_bytes(raw, context=name)
        return snapshots
    except BaseException as error:
        _close_snapshots(snapshots, primary_error=error)
        raise


def _close_snapshots(
    snapshots: Mapping[str, _Snapshot],
    *,
    primary_error: BaseException | None = None,
) -> None:
    errors: list[BaseException] = []
    for snapshot in snapshots.values():
        try:
            snapshot.close()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
    if errors:
        message = "snapshot descriptor cleanup failed: " + "; ".join(
            str(error) for error in errors
        )
        if primary_error is not None:
            message = f"{primary_error}; {message}"
        raise RebuttalRenderError(message) from primary_error


def _revalidate_snapshots(snapshots: Mapping[str, _Snapshot]) -> None:
    for name, snapshot in snapshots.items():
        _revalidate_snapshot(snapshot, context=name)


def _snapshot_payload(
    snapshots: Mapping[str, _Snapshot],
    name: str,
    *,
    maximum: int,
) -> bytes:
    snapshot = snapshots[name]
    _revalidate_snapshot(snapshot, context=name)
    return _read_fd(
        snapshot.descriptor,
        maximum=maximum,
        context=f"{name} snapshot",
    )


def _revalidate_all(pins: Mapping[str, object]) -> None:
    for name, pin in pins.items():
        if isinstance(pin, _Snapshot):
            _revalidate_snapshot(pin, context=name)
        else:
            _revalidate_pinned_file(pin, context=name)


def _run_tool(
    authority: _MachineAuthority,
    pin: object,
    arguments: Sequence[str],
    *,
    inherited_fds: Sequence[int],
    stdout_limit: int,
    budget: object,
    before: Callable[[], None],
) -> tuple[bytes, dict[str, object]]:
    _revalidate_pinned_file(authority, context="machine runner")
    machine = authority.module
    try:
        return_code, stdout, stderr, attestation = machine._run_bounded(
            pin,
            arguments,
            inherited_fds=inherited_fds,
            timeout=TOOL_TIMEOUT_SECONDS,
            stdout_limit=stdout_limit,
            stderr_limit=MAX_STDERR_BYTES,
            budget=budget,
            before=before,
            after=before,
        )
    except Exception as error:  # noqa: BLE001 - translate the imported boundary.
        _raise_machine_failure("bounded native tool execution failed", error)
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace")[:2000]
        _fail(f"bounded tool exited {return_code}: {detail}")
    if stderr:
        detail = stderr.decode("utf-8", errors="replace")[:2000]
        _fail(f"bounded tool emitted stderr: {detail}")
    return stdout, _sanitize_attestation(attestation, machine)


def _sanitize_attestation(
    value: Mapping[str, object],
    machine: object,
) -> dict[str, object]:
    """Reduce a live native attestation to equality-stable verified facts."""
    record = _expect_keys(
        value,
        {
            "protocol",
            "spawn_flags",
            "suspended_wait_status",
            "code_signing_status",
            "expected_code_directory",
            "observed_cdhash",
            "main_executable_mapping",
            "execution_binding_scope",
            "non_system_dylib_closure",
            "same_vnode_mutation_fail_stop_assumption",
            "other_same_vnode_mutations",
        },
        context="native execution attestation",
    )
    code_directory = _expect_keys(
        record["expected_code_directory"],
        {
            "binary_container",
            "architecture",
            "hash_type",
            "code_directory_bytes",
            "cdhash",
        },
        context="native expected CodeDirectory",
    )
    mapping = _expect_keys(
        record["main_executable_mapping"],
        {
            "device",
            "inode",
            "path",
            "mode",
            "link_count",
            "protection",
            "file_offset",
        },
        context="native main-executable mapping",
    )
    if (
        record["protocol"] != "darwin-posix-spawn-suspended-main-executable-v1"
        or record["execution_binding_scope"] != "main_executable"
        or record["non_system_dylib_closure"] != "not_attested"
        or record["other_same_vnode_mutations"] != "not_attested"
        or record["same_vnode_mutation_fail_stop_assumption"]
        != "invalid-signed-code-page-triggers-darwin-cs-kill"
        or record["spawn_flags"] != machine.DARWIN_SPAWN_FLAGS
        or not isinstance(record["suspended_wait_status"], int)
    ):
        _fail("native execution attestation policy is invalid")
    status = record["code_signing_status"]
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status & machine.REQUIRED_CS_FLAGS != machine.REQUIRED_CS_FLAGS
        or status & machine.REJECTED_CS_FLAGS
    ):
        _fail("native execution attestation has invalid code-signing status")
    if (
        code_directory["binary_container"] != "thin-macho64"
        or code_directory["architecture"] != "arm64"
        or code_directory["hash_type"]
        not in {
            "sha1",
            "sha256",
            "sha256-truncated",
            "sha384",
        }
        or isinstance(code_directory["code_directory_bytes"], bool)
        or not isinstance(code_directory["code_directory_bytes"], int)
        or not 1 <= code_directory["code_directory_bytes"] <= 128 * 1024 * 1024
        or not isinstance(code_directory["cdhash"], str)
        or re.fullmatch(r"[0-9a-f]{40}", code_directory["cdhash"]) is None
        or record["observed_cdhash"] != code_directory["cdhash"]
    ):
        _fail("native execution attestation has an invalid CodeDirectory binding")
    path = mapping["path"]
    mode = mapping["mode"]
    if (
        isinstance(mapping["device"], bool)
        or not isinstance(mapping["device"], int)
        or mapping["device"] < 0
        or isinstance(mapping["inode"], bool)
        or not isinstance(mapping["inode"], int)
        or mapping["inode"] <= 0
        or not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(mode, str)
        or re.fullmatch(r"[0-7]{4}", mode) is None
        or int(mode, 8) & 0o111 == 0
        or mapping["link_count"] != 1
        or isinstance(mapping["protection"], bool)
        or not isinstance(mapping["protection"], int)
        or mapping["protection"] & machine.VM_PROT_EXECUTE == 0
        or mapping["file_offset"] != 0
    ):
        _fail("native execution attestation has an invalid executable mapping")
    return {
        "schema": "dialect-stable-native-main-executable-attestation-v1",
        "protocol": record["protocol"],
        "spawn_flags": record["spawn_flags"],
        "suspended_before_execution_verified": True,
        "code_signing_policy_verified": True,
        "expected_code_directory": dict(code_directory),
        "observed_cdhash": record["observed_cdhash"],
        "main_executable_mapping_verified": True,
        "host_path_device_inode_mode_and_status_recorded": False,
        "execution_binding_scope": record["execution_binding_scope"],
        "non_system_dylib_closure": record["non_system_dylib_closure"],
        "same_vnode_mutation_fail_stop_assumption": record[
            "same_vnode_mutation_fail_stop_assumption"
        ],
        "other_same_vnode_mutations": record["other_same_vnode_mutations"],
    }


def _pdf_descriptor(raw: bytes) -> tempfile._TemporaryFileWrapper[bytes] | object:
    temporary = tempfile.TemporaryFile(mode="w+b")
    try:
        _write_all(temporary.fileno(), raw, context="descriptor-backed temporary file")
        temporary.flush()
        os.fchmod(temporary.fileno(), 0o600)
        temporary.seek(0)
        return temporary
    except BaseException as error:
        _close_owned_file(
            temporary,
            context="descriptor-backed temporary file",
            primary_error=error,
        )
        raise


def _descriptor_guard(
    descriptor: int, expected: bytes, outer: Callable[[], None]
) -> None:
    outer()
    entry = os.fstat(descriptor)
    if not stat.S_ISREG(entry.st_mode) or entry.st_size != len(expected):
        _fail("descriptor-backed PDF changed identity or size")
    observed = _read_fd(
        descriptor, maximum=MAX_PDF_BYTES, context="descriptor-backed PDF"
    )
    if observed != expected:
        _fail("descriptor-backed PDF bytes changed")


def _validate_fd_headroom(required: int) -> dict[str, object]:
    if required <= 0:
        _fail("file-descriptor headroom must be positive")
    descriptor_names = _bounded_directory_names(
        "/dev/fd",
        maximum=4096,
        context="current file-descriptor table",
    )
    open_count = sum(name.isdecimal() for name in descriptor_names)
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and open_count + required > soft:
        _fail(
            "insufficient file-descriptor headroom: "
            f"open={open_count}, required={required}, soft={soft}",
        )
    return {
        "required_headroom": required,
        "status": "pass",
    }


def _normalized_text(value: str) -> str:
    return re.sub(
        r"[ \t\r\n\f\v]+",
        " ",
        unicodedata.normalize("NFC", value),
    ).strip()


def _ttf_postscript_name(raw: bytes, *, context: str) -> str:
    """Extract one bounded PostScript name from a non-collection TrueType font."""
    if len(raw) < 12 or raw[:4] == b"ttcf":
        _fail(f"{context} must be one bounded non-collection TrueType font")
    try:
        table_count = struct.unpack_from(">H", raw, 4)[0]
    except struct.error as error:
        _fail(f"{context} has a truncated TrueType header: {error}")
    if not 1 <= table_count <= 128 or 12 + 16 * table_count > len(raw):
        _fail(f"{context} has an invalid TrueType table directory")
    name_tables: list[tuple[int, int]] = []
    for index in range(table_count):
        start = 12 + 16 * index
        tag, _checksum, offset, length = struct.unpack_from(">4sIII", raw, start)
        if offset > len(raw) or length > len(raw) - offset:
            _fail(f"{context} has an out-of-bounds TrueType table")
        if tag == b"name":
            name_tables.append((offset, length))
    if len(name_tables) != 1:
        _fail(f"{context} must contain exactly one TrueType name table")
    table_offset, table_length = name_tables[0]
    if table_length < 6:
        _fail(f"{context} has a truncated TrueType name table")
    name_format, record_count, string_offset = struct.unpack_from(
        ">HHH",
        raw,
        table_offset,
    )
    if (
        name_format not in {0, 1}
        or not 1 <= record_count <= 4096
        or 6 + 12 * record_count > table_length
        or string_offset > table_length
    ):
        _fail(f"{context} has an invalid TrueType name table")
    names: set[str] = set()
    for index in range(record_count):
        record_offset = table_offset + 6 + 12 * index
        platform, encoding, _language, name_id, length, relative = struct.unpack_from(
            ">HHHHHH",
            raw,
            record_offset,
        )
        if name_id != 6:
            continue
        start = table_offset + string_offset + relative
        end = start + length
        if start < table_offset or end > table_offset + table_length:
            _fail(f"{context} has an out-of-bounds PostScript name")
        encoded = raw[start:end]
        try:
            if platform in {0, 3}:
                decoded = encoded.decode("utf-16-be", errors="strict")
            elif platform == 1 and encoding == 0:
                decoded = encoded.decode("mac_roman", errors="strict")
            else:
                continue
        except UnicodeDecodeError as error:
            _fail(f"{context} has an invalid PostScript name encoding: {error}")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,127}", decoded) is None:
            _fail(f"{context} has a noncanonical PostScript font name")
        names.add(decoded)
    if len(names) != 1:
        _fail(f"{context} must expose exactly one canonical PostScript font name")
    return names.pop()


def _validate_font_roles(regular: bytes, bold: bytes) -> tuple[str, str]:
    if _sha256(regular) == _sha256(bold):
        _fail("regular and bold font roles must have distinct pinned bytes")
    names = (
        _ttf_postscript_name(regular, context="regular font"),
        _ttf_postscript_name(bold, context="bold font"),
    )
    if names[0] == names[1]:
        _fail("regular and bold font roles must expose distinct PostScript names")
    return names


def _parse_pdffonts_output(raw: bytes) -> tuple[dict[str, object], ...]:
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        _fail(f"pdffonts output is not ASCII: {error}")
    lines = text.splitlines()
    if (
        len(lines) < 3
        or lines[0] != PDF_FONTS_HEADER
        or lines[1] != PDF_FONTS_SEPARATOR
    ):
        _fail("pdffonts output has an unexpected header or separator")
    rows: list[dict[str, object]] = []
    for line in lines[2:]:
        if not line:
            _fail("pdffonts output contains an unexpected blank row")
        match = FONT_ROW_RE.fullmatch(line)
        if match is None:
            _fail("pdffonts output contains an unrecognized font row")
        (
            name,
            font_type,
            encoding,
            embedded,
            subset,
            unicode_map,
            object_id,
            generation,
        ) = match.groups()
        if embedded != "yes" or font_type == "Type 3" or unicode_map != "yes":
            _fail("rendered PDF contains a Type 3, unembedded, or unmapped font")
        rows.append(
            {
                "name": name,
                "type": font_type,
                "encoding": encoding,
                "embedded": embedded,
                "subset": subset,
                "unicode_map": unicode_map,
                "object_id": int(object_id),
                "generation": int(generation),
            }
        )
    if not rows:
        _fail("pdffonts found no fonts")
    return tuple(rows)


def _validate_pdf_font_roles(
    rows: Sequence[Mapping[str, object]],
    postscript_names: tuple[str, str],
) -> None:
    if (
        len(rows) != 2
        or len({str(row["name"]) for row in rows}) != 2
        or any(row["type"] != "TrueType" for row in rows)
    ):
        _fail("PDF must use exactly two distinct embedded TrueType font faces")
    observed = {re.sub(r"^[A-Z]{6}\+", "", str(row["name"])) for row in rows}
    if observed != set(postscript_names):
        _fail("PDF embedded font faces do not match the pinned font roles")


def _validate_pdf(
    raw: bytes,
    audit: _MarkdownAudit,
    config: Mapping[str, object],
    pins: Mapping[str, object],
    font_postscript_names: tuple[str, str],
    budget: object,
    before: Callable[[], None],
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        not raw.startswith(PDF_SIGNATURE)
        or not raw.endswith(PDF_EOF)
        or raw.count(b"%%EOF") != 1
        or len(raw) > MAX_PDF_BYTES
    ):
        _fail("rendered output is not one bounded canonical PDF byte stream")
    for token in FORBIDDEN_PDF_TOKENS:
        if token in raw:
            _fail(f"rendered PDF contains forbidden token {token.decode('ascii')}")
    pdf_file = _pdf_descriptor(raw)
    try:
        descriptor = pdf_file.fileno()

        def guard() -> None:
            _descriptor_guard(descriptor, raw, before)

        inherited = (descriptor,)
        path = f"/dev/fd/{descriptor}"
        pdfinfo_raw, pdfinfo_attestation = _run_tool(
            pins["machine_runner"],
            pins["tool:pdfinfo"],
            [path],
            inherited_fds=inherited,
            stdout_limit=MAX_JSON_BYTES,
            budget=budget,
            before=guard,
        )
        pdfinfo = _decode_utf8(pdfinfo_raw, context="pdfinfo output")
        expected_metadata = {
            "Title": str(config["response_title"]),
            "Subject": str(config["manuscript_title"]),
            "Author": ", ".join(str(author) for author in config["authors"]),
            "Creator": "DIALECT deterministic rebuttal renderer v1",
            "Producer": "ReportLab PDF Library - (opensource)",
        }
        for label, expected_value in expected_metadata.items():
            matches = re.findall(rf"^{label}:\s*(.*?)\s*$", pdfinfo, re.MULTILINE)
            if matches != [expected_value]:
                _fail(f"pdfinfo {label} does not match canonical document metadata")
        pages_match = re.search(r"^Pages:\s+([1-9][0-9]*)\s*$", pdfinfo, re.MULTILINE)
        if pages_match is None:
            _fail("pdfinfo did not report a positive page count")
        page_count = int(pages_match.group(1))
        if page_count > MAX_PAGES:
            _fail(f"rendered PDF exceeds the {MAX_PAGES}-page limit")
        if re.search(r"^Encrypted:\s+no\s*$", pdfinfo, re.MULTILINE) is None:
            _fail("rendered PDF is encrypted or encryption status is ambiguous")
        size_match = re.search(
            r"^Page size:\s+612(?:\.0+)?\s+x\s+792(?:\.0+)?\s+pts\s+\(letter\)\s*$",
            pdfinfo,
            re.MULTILINE,
        )
        if size_match is None:
            _fail("rendered PDF is not US Letter")
        for label in ("CreationDate", "ModDate"):
            if (
                re.search(
                    rf"^{label}:\s+Sat Jan\s+1 00:00:00 2000 UTC\s*$",
                    pdfinfo,
                    re.MULTILINE,
                )
                is None
            ):
                _fail(f"rendered PDF {label} is not the fixed invariant epoch")

        fonts_raw, fonts_attestation = _run_tool(
            pins["machine_runner"],
            pins["tool:pdffonts"],
            [path],
            inherited_fds=inherited,
            stdout_limit=MAX_JSON_BYTES,
            budget=budget,
            before=guard,
        )
        font_rows = _parse_pdffonts_output(fonts_raw)
        _validate_pdf_font_roles(font_rows, font_postscript_names)

        text_raw, text_attestation = _run_tool(
            pins["machine_runner"],
            pins["tool:pdftotext"],
            ["-layout", "-enc", "UTF-8", path, "-"],
            inherited_fds=inherited,
            stdout_limit=MAX_SOURCE_BYTES * 2,
            budget=budget,
            before=guard,
        )
        extracted = _decode_utf8(text_raw, context="pdftotext output")
        body_text = _strip_pdf_decorations(
            extracted,
            config,
            page_count=page_count,
        )
        visible_ledger = _visible_text_ledger(audit, config)
        _verify_visible_text_ledger(body_text, visible_ledger)
        guard()
    finally:
        _close_owned_file(
            pdf_file,
            context="descriptor-backed PDF",
            primary_error=sys.exception(),
        )
    return (
        {
            "sha256": _sha256(raw),
            "size": len(raw),
            "pages": page_count,
            "fonts": font_rows,
            "font_count": len(font_rows),
            "encrypted": False,
            "page_size": "letter",
            "reviewer_quote_text_blocks_verified": len(audit.quotes),
            "text_equivalence": "NFC-with-ASCII-whitespace-folding",
            "response_labels": audit.response_count,
            "visible_text_blocks_preserved": len(visible_ledger),
        },
        {
            "pdfinfo": pdfinfo_attestation,
            "pdffonts": fonts_attestation,
            "pdftotext": text_attestation,
        },
    )


def _builder_path() -> Path:
    return Path(__file__).resolve(strict=True)


def _produce(
    pins: Mapping[str, object],
    snapshots: Mapping[str, _Snapshot],
    reportlab_bundle: _ReportLabBundle,
    *,
    expected_reportlab_tree_sha256: str,
    release_id: str,
) -> _Production:
    source_raw = _snapshot_payload(snapshots, "source", maximum=MAX_SOURCE_BYTES)
    template_raw = _snapshot_payload(snapshots, "template", maximum=MAX_JSON_BYTES)
    config_raw = _snapshot_payload(snapshots, "config", maximum=MAX_JSON_BYTES)
    canonical_source = _canonicalize_markdown(source_raw)
    template = _normalize_template(
        _json_without_duplicates(template_raw, context="template")
    )
    config = _normalize_config(_json_without_duplicates(config_raw, context="config"))
    if config["release_id"] != release_id:
        _fail("config release_id does not match the requested release")
    audit = _parse_markdown(canonical_source)
    _validate_title_binding(audit, config)
    font_postscript_names = _validate_font_roles(
        _snapshot_payload(snapshots, "regular_font", maximum=MAX_FONT_BYTES),
        _snapshot_payload(snapshots, "bold_font", maximum=MAX_FONT_BYTES),
    )
    canonical_template = _canonical_json(template)
    canonical_config = _canonical_json(config)
    expected_tree = _expect_sha256(
        expected_reportlab_tree_sha256,
        context="expected ReportLab tree SHA-256",
    )
    if reportlab_bundle.tree_sha256 != expected_tree:
        _fail("ReportLab package tree does not match its caller SHA-256 anchor")

    budget = _LocalProcessBudget()
    fd_budget = _validate_fd_headroom(MAX_FDS)
    bundle_file = _pdf_descriptor(reportlab_bundle.raw)
    try:
        bundle_fd = bundle_file.fileno()

        def guard() -> None:
            _revalidate_all(pins)
            _revalidate_snapshots(snapshots)
            _descriptor_guard(bundle_fd, reportlab_bundle.raw, lambda: None)
            # Opening /dev/fd/N duplicates a Darwin file description and therefore
            # shares its offset. Revalidation hashes leave descriptors at EOF, so
            # rewind every descriptor that the interpreter or ReportLab opens by
            # descriptor path. Source/config readers seek explicitly in the child.
            for descriptor in (
                snapshots["builder"].descriptor,
                snapshots["regular_font"].descriptor,
                snapshots["bold_font"].descriptor,
                bundle_fd,
            ):
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                except OSError as error:
                    _fail(f"cannot rewind inherited render descriptor: {error}")

        inherited = (
            snapshots["builder"].descriptor,
            snapshots["source"].descriptor,
            snapshots["template"].descriptor,
            snapshots["config"].descriptor,
            snapshots["regular_font"].descriptor,
            snapshots["bold_font"].descriptor,
            bundle_fd,
        )
        if len(inherited) + 3 > MAX_FDS:
            _fail("render invocation exceeds its explicit file-descriptor budget")
        arguments = [
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null/dialect-rebuttal-pycache",
            f"/dev/fd/{snapshots['builder'].descriptor}",
            "--internal-render",
            "--source-fd",
            str(snapshots["source"].descriptor),
            "--template-fd",
            str(snapshots["template"].descriptor),
            "--config-fd",
            str(snapshots["config"].descriptor),
            "--regular-font-fd",
            str(snapshots["regular_font"].descriptor),
            "--bold-font-fd",
            str(snapshots["bold_font"].descriptor),
            "--reportlab-bundle-fd",
            str(bundle_fd),
            "--source-date-epoch",
            str(config["source_date_epoch"]),
        ]
        first, first_attestation = _run_tool(
            pins["machine_runner"],
            pins["tool:python"],
            arguments,
            inherited_fds=inherited,
            stdout_limit=MAX_PDF_BYTES,
            budget=budget,
            before=guard,
        )
        second, second_attestation = _run_tool(
            pins["machine_runner"],
            pins["tool:python"],
            arguments,
            inherited_fds=inherited,
            stdout_limit=MAX_PDF_BYTES,
            budget=budget,
            before=guard,
        )
        if first != second:
            _fail("independent ReportLab passes are not byte-identical")
        pdf_record, qa_attestations = _validate_pdf(
            first,
            audit,
            config,
            pins,
            font_postscript_names,
            budget,
            guard,
        )
        budget.assert_complete()
        guard()
    finally:
        _close_owned_file(
            bundle_file,
            context="ReportLab descriptor bundle",
            primary_error=sys.exception(),
        )

    tool_records = [
        {
            "name": name,
            "sha256": pins[f"tool:{name}"].sha256,
            "size": pins[f"tool:{name}"].size,
            "host_path_recorded": False,
        }
        for name in TOOL_ORDER
    ]
    members_raw = {
        SOURCE_MEMBER: canonical_source,
        TEMPLATE_MEMBER: canonical_template,
        CONFIG_MEMBER: canonical_config,
        PDF_MEMBER: first,
    }
    members = [
        {
            "member": member,
            "sha256": _sha256(members_raw[member]),
            "size": len(members_raw[member]),
        }
        for member in MEMBER_ORDER
    ]
    manifest = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "release_id": release_id,
        "inputs": {
            "source": {
                "raw_sha256": pins["source"].sha256,
                "canonical_sha256": _sha256(canonical_source),
                "canonical_size": len(canonical_source),
            },
            "template": {
                "raw_sha256": pins["template"].sha256,
                "canonical_sha256": _sha256(canonical_template),
                "canonical_size": len(canonical_template),
            },
            "config": {
                "raw_sha256": pins["config"].sha256,
                "canonical_sha256": _sha256(canonical_config),
                "canonical_size": len(canonical_config),
            },
            "fonts": [
                {
                    "role": role,
                    "sha256": pins[f"{role}_font"].sha256,
                    "size": pins[f"{role}_font"].size,
                    "postscript_name": font_postscript_names[index],
                }
                for index, role in enumerate(("regular", "bold"))
            ],
            "reportlab": {
                "tree_sha256": reportlab_bundle.tree_sha256,
                "file_count": reportlab_bundle.file_count,
                "directory_count": reportlab_bundle.directory_count,
                "entry_count": reportlab_bundle.entry_count,
                "total_bytes": reportlab_bundle.total_bytes,
                "bundle_sha256": _sha256(reportlab_bundle.raw),
                "bundle_size": len(reportlab_bundle.raw),
                "pure_python_bundle": True,
                "excluded": ["__pycache__", "*.pyc", "*.so", "*.dylib"],
            },
            "builder": {
                "member": "analysis/render_tcga_revision_rebuttal.py",
                "sha256": pins["builder"].sha256,
                "size": pins["builder"].size,
            },
            "machine_runner": {
                "member": MACHINE_RUNNER_MEMBER,
                "sha256": pins["machine_runner"].sha256,
                "size": pins["machine_runner"].size,
                "load_protocol": "caller-sha-before-private-descriptor-bytecode-exec",
            },
            "tools": tool_records,
        },
        "markdown": {
            "reviewer_count": audit.reviewer_count,
            "quote_count": len(audit.quotes),
            "response_count": audit.response_count,
            "quote_sha256": [
                {"comment_id": quote_id, "sha256": _sha256(quote.encode("utf-8"))}
                for quote_id, quote in audit.quotes
            ],
            "visible_text_ledger": [
                {
                    "block_id": record["block_id"],
                    "kind": record["kind"],
                    "sha256": record["sha256"],
                }
                for record in _visible_text_ledger(audit, config)
            ],
            "author_raw_tex_html_images_and_unsupported_markdown": "rejected",
            "author_structured_placeholders_and_known_gate_markers": "rejected",
            "published_private_path_forms": (
                "enumerated-high-risk-vocabulary-rejected-across-source-"
                "including-quotes-and-config"
            ),
            "reviewer_quote_source_bytes": "exact-in-source.canonical.md",
            "pdf_visible_text_equivalence": "NFC-with-ASCII-whitespace-folding",
        },
        "document": pdf_record,
        "determinism": {
            "reportlab_invariant": True,
            "reportlab_invariant_epoch": config["source_date_epoch"],
            "independent_render_passes": 2,
            "byte_identical": True,
            "direct_tool_invocation_budget": budget.maximum,
            "direct_tool_invocations_consumed": budget.count,
            "descendant_process_count": "not_attested",
            "successful_detached_descendant_containment": "not_provided",
        },
        "execution": {
            "render_attestations": [first_attestation, second_attestation],
            "qa_attestations": qa_attestations,
            "environment": {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            "darwin_injected_cf_user_text_encoding": "validated-then-removed",
            "working_directory": "/",
            "shell": False,
            "content_derivation_authority": (
                "unlinked-read-only-snapshots-of-caller-sha-verified-bytes"
            ),
            "native_tool_executable_authority": (
                "caller-sha-pinned-named-read-only-descriptors"
            ),
            "reportlab_effective_search_paths": {
                "T1SearchPath": [],
                "TTFSearchPath": [],
                "CMapSearchPath": [],
            },
            "outer_renderer_bootstrap": "trusted-before-self-pin",
            "machine_runner_authority": (
                "caller-sha-verified-before-private-descriptor-bytecode-exec"
            ),
            "main_executable_scope": "attested",
            "python_stdlib_and_dylib_closure": "not_attested",
            "poppler_dylib_closure": "not_attested",
            "ambient_same_uid_filesystem_containment": "not_provided",
            "pillow_dependency": {
                "status": "not-loaded-fail-closed-no-image-stub",
                "stub_anchor": "builder-sha256",
                "native_imaging_loaded": False,
            },
            "reportlab_runtime": {
                "customization_modules": "identity-checked-inert-stubs",
                "optional_native_modules": "blocked-and-not-loaded",
                "accelerator_exports": "verified-pure-python",
                "extension_loaded_reportlab_modules": False,
                "proof_gate": "passed-in-each-render-child",
            },
            "file_descriptor_budget": fd_budget,
        },
        "members": members,
        "claims": {
            "scientific_accuracy": "not_attested",
            "human_visual_approval": "required-separately",
            "journal_acceptance": "not_attested",
            "submission_status": "not_attested",
        },
        "integration": {
            "four_role_derivation_adapter_protocol": (
                "renderer-fd-stdout-derive-mode-implemented"
            ),
            "schema_compatibility_with_derivation_closure": False,
            "rebuttal_role_gate": "not-cleared-by-this-renderer",
            "promotion_gate": "not-cleared-by-this-renderer",
            "future_seam": (
                "reviewed native thin-arm64 launcher invokes this renderer's "
                "fixed bundle-FD/stdout mode; downstream promotion separately "
                "binds derivation, machine replay, and authenticated page-complete "
                "visual approval"
            ),
        },
    }
    manifest_raw = _require_manifest_size(_canonical_json(manifest))
    return _Production(
        manifest=manifest,
        manifest_raw=manifest_raw,
        pdf_raw=first,
        page_count=int(pdf_record["pages"]),
        quote_count=len(audit.quotes),
        response_count=audit.response_count,
    )


def _rename_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    if sys.platform != "darwin":
        _fail("atomic no-replace publication requires Darwin renameatx_np")
    library = ctypes.CDLL(None, use_errno=True)
    library.renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    library.renameatx_np.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = library.renameatx_np(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        0x00000004,  # RENAME_EXCL
    )
    if result != 0:
        error_number = ctypes.get_errno()
        _fail(f"atomic no-replace publication failed: {os.strerror(error_number)}")


def _write_member(root_descriptor: int, member: str, raw: bytes) -> None:
    if PurePosixPath(member).name != member or member in {"", ".", ".."}:
        _fail("output member is not one canonical basename")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(member, flags, 0o600, dir_fd=root_descriptor)
    try:
        _write_all(descriptor, raw, context=f"output member {member}")
        os.fsync(descriptor)
        if _read_fd(descriptor, maximum=max(len(raw), 1), context=member) != raw:
            _fail(f"output member {member!r} failed descriptor readback")
    finally:
        _close_descriptor(
            descriptor,
            context=f"output member {member}",
            primary_error=sys.exception(),
        )


def _publish(
    destination: Path, production: _Production, canonical_inputs: Mapping[str, bytes]
) -> Path:
    absolute = destination.absolute()
    if absolute.name in {"", ".", ".."}:
        _fail("destination must name one new directory")
    parent_path = absolute.parent
    try:
        parent_entry = os.lstat(parent_path)
        parent_resolved = parent_path.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect destination parent: {error}")
    if (
        stat.S_ISLNK(parent_entry.st_mode)
        or not stat.S_ISDIR(parent_entry.st_mode)
        or parent_resolved != parent_path
        or parent_entry.st_uid != os.geteuid()
        or stat.S_IMODE(parent_entry.st_mode) & 0o022
    ):
        _fail(
            "destination parent must be canonical, caller-owned, and not "
            "group/world writable",
        )
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    parent_descriptor = os.open(parent_path, parent_flags)
    stage_name = f".{absolute.name}.private-candidate"
    stage_descriptor = -1
    stage_identity: tuple[int, int] | None = None
    renamed = False
    try:
        opened_parent = os.fstat(parent_descriptor)
        parent_identity = (
            parent_entry.st_dev,
            parent_entry.st_ino,
            parent_entry.st_uid,
            stat.S_IMODE(parent_entry.st_mode),
        )
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
            opened_parent.st_uid,
            stat.S_IMODE(opened_parent.st_mode),
        ) != parent_identity:
            _fail("destination parent changed while it was opened")
        try:
            os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            _fail("destination already exists")
        try:
            os.mkdir(stage_name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            _fail(f"private candidate already exists: {parent_path / stage_name}")
        stage_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        stage_descriptor = os.open(stage_name, stage_flags, dir_fd=parent_descriptor)
        created_stage = os.fstat(stage_descriptor)
        named_created_stage = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stage_identity = (created_stage.st_dev, created_stage.st_ino)
        if (
            not stat.S_ISDIR(created_stage.st_mode)
            or created_stage.st_uid != os.geteuid()
            or (named_created_stage.st_dev, named_created_stage.st_ino)
            != stage_identity
        ):
            _fail("private candidate changed while it was opened")
        raw_by_member = {
            SOURCE_MEMBER: canonical_inputs[SOURCE_MEMBER],
            TEMPLATE_MEMBER: canonical_inputs[TEMPLATE_MEMBER],
            CONFIG_MEMBER: canonical_inputs[CONFIG_MEMBER],
            PDF_MEMBER: production.pdf_raw,
            MANIFEST_MEMBER: production.manifest_raw,
        }
        for member in (*MEMBER_ORDER, MANIFEST_MEMBER):
            _write_member(stage_descriptor, member, raw_by_member[member])
        inventory = _bounded_directory_names(
            stage_descriptor,
            maximum=len(MEMBER_ORDER) + 2,
            context="private candidate",
        )
        if inventory != sorted((*MEMBER_ORDER, MANIFEST_MEMBER)):
            _fail("private candidate inventory is not exact")
        for member, raw in raw_by_member.items():
            descriptor = os.open(
                member,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=stage_descriptor,
            )
            try:
                member_entry = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(member_entry.st_mode)
                    or member_entry.st_nlink != 1
                    or member_entry.st_uid != os.geteuid()
                    or stat.S_IMODE(member_entry.st_mode) != 0o600
                ):
                    _fail(f"private candidate member {member!r} changed type or mode")
                if (
                    _read_fd(descriptor, maximum=max(len(raw), 1), context=member)
                    != raw
                ):
                    _fail(f"private candidate member {member!r} changed")
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                _close_descriptor(
                    descriptor,
                    context=f"private candidate member {member}",
                    primary_error=sys.exception(),
                )
        os.fchmod(stage_descriptor, 0o500)
        os.fsync(stage_descriptor)
        current_parent = os.fstat(parent_descriptor)
        named_parent = os.lstat(parent_path)
        if (
            (
                current_parent.st_dev,
                current_parent.st_ino,
                current_parent.st_uid,
                stat.S_IMODE(current_parent.st_mode),
            )
            != parent_identity
            or (
                named_parent.st_dev,
                named_parent.st_ino,
                named_parent.st_uid,
                stat.S_IMODE(named_parent.st_mode),
            )
            != parent_identity
            or parent_path.resolve(strict=True) != parent_path
        ):
            _fail("destination parent changed before publication")
        try:
            os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            _fail("destination appeared before publication")
        stage_entry = os.fstat(stage_descriptor)
        named_stage = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stage_identity = (stage_entry.st_dev, stage_entry.st_ino)
        if (
            not stat.S_ISDIR(stage_entry.st_mode)
            or not stat.S_ISDIR(named_stage.st_mode)
            or stage_entry.st_uid != os.geteuid()
            or named_stage.st_uid != os.geteuid()
            or stat.S_IMODE(stage_entry.st_mode) != 0o500
            or stat.S_IMODE(named_stage.st_mode) != 0o500
            or (named_stage.st_dev, named_stage.st_ino) != stage_identity
        ):
            _fail("private candidate basename no longer names the sealed stage")
        _rename_no_replace(parent_descriptor, stage_name, absolute.name)
        renamed = True
        named_destination = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_destination.st_mode)
            or named_destination.st_uid != os.geteuid()
            or stat.S_IMODE(named_destination.st_mode) != 0o500
            or (named_destination.st_dev, named_destination.st_ino) != stage_identity
        ):
            _fail("published destination does not name the sealed stage identity")
        os.fsync(parent_descriptor)
        terminal_parent = os.fstat(parent_descriptor)
        terminal_named_parent = os.lstat(parent_path)
        if (
            (
                terminal_parent.st_dev,
                terminal_parent.st_ino,
                terminal_parent.st_uid,
                stat.S_IMODE(terminal_parent.st_mode),
            )
            != parent_identity
            or (
                terminal_named_parent.st_dev,
                terminal_named_parent.st_ino,
                terminal_named_parent.st_uid,
                stat.S_IMODE(terminal_named_parent.st_mode),
            )
            != parent_identity
            or parent_path.resolve(strict=True) != parent_path
        ):
            _fail("destination parent changed after publication")
        if absolute.resolve(strict=True) != absolute:
            _fail("published destination path is no longer canonical")
    except BaseException as error:
        parent_reference = (
            f"pinned_parent_dev={parent_entry.st_dev}; "
            f"pinned_parent_ino={parent_entry.st_ino}"
        )
        stage_reference = (
            f"stage_dev={stage_identity[0]}; stage_ino={stage_identity[1]}"
            if stage_identity is not None
            else "stage_identity=not-established"
        )
        if renamed:
            message = (
                f"{error}; published_basename={absolute.name}; {parent_reference}; "
                f"{stage_reference}; "
                "publication_state=rename-completed-post-publication-failure; "
                "no-cleanup-attempted; path-resolution-and-recoverability-not-attested"
            )
        else:
            message = (
                f"{error}; candidate_basename={stage_name}; {parent_reference}; "
                f"{stage_reference}; publication_state=pre-rename-failure; "
                "no-cleanup-attempted; path-resolution-and-recoverability-not-attested"
            )
        raise RebuttalRenderError(message) from error
    finally:
        primary_error = sys.exception()
        cleanup_errors: list[BaseException] = []
        for descriptor in (stage_descriptor, parent_descriptor):
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except BaseException as error:  # noqa: BLE001 - attempt every close.
                cleanup_errors.append(error)
        if cleanup_errors:
            location = (
                f"published_basename={absolute.name}; "
                "publication_state=rename-completed"
                if renamed
                else (f"candidate_basename={stage_name}; publication_state=pre-rename")
            )
            cleanup_message = "; ".join(str(error) for error in cleanup_errors)
            message = f"{location}; descriptor cleanup failed: {cleanup_message}"
            if primary_error is not None:
                message = f"{primary_error}; {message}"
            raise RebuttalRenderError(message) from primary_error
    return absolute


def _read_published(
    root: Path, *, expected_manifest_sha256: str
) -> tuple[bytes, dict[str, object], tuple[int, int]]:
    expected = _expect_sha256(
        expected_manifest_sha256, context="expected manifest SHA-256"
    )
    absolute = root.absolute()
    parent_path = absolute.parent
    try:
        parent_entry = os.lstat(parent_path)
        parent_resolved = parent_path.resolve(strict=True)
        root_entry = os.lstat(absolute)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect published root: {error}")
    if (
        stat.S_ISLNK(parent_entry.st_mode)
        or not stat.S_ISDIR(parent_entry.st_mode)
        or parent_resolved != parent_path
        or parent_entry.st_uid != os.geteuid()
        or stat.S_IMODE(parent_entry.st_mode) & 0o022
    ):
        _fail("published root parent is not a safe canonical authority")
    if (
        stat.S_ISLNK(root_entry.st_mode)
        or not stat.S_ISDIR(root_entry.st_mode)
        or resolved != absolute
        or stat.S_IMODE(root_entry.st_mode) != 0o500
        or root_entry.st_uid != os.geteuid()
    ):
        _fail("published root must be a canonical sealed directory")
    root_identity = (
        root_entry.st_dev,
        root_entry.st_ino,
        root_entry.st_size,
        root_entry.st_mtime_ns,
        stat.S_IMODE(root_entry.st_mode),
    )
    descriptor = os.open(
        absolute,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened_root = os.fstat(descriptor)
        if (
            opened_root.st_dev,
            opened_root.st_ino,
            opened_root.st_size,
            opened_root.st_mtime_ns,
            stat.S_IMODE(opened_root.st_mode),
        ) != root_identity:
            _fail("published root changed while it was opened")
        inventory = _bounded_directory_names(
            descriptor,
            maximum=len(MEMBER_ORDER) + 2,
            context="published root",
        )
        if inventory != sorted((*MEMBER_ORDER, MANIFEST_MEMBER)):
            _fail("published root inventory is not exact")
        raw_by_member: dict[str, bytes] = {}
        for member in inventory:
            named_member = os.stat(
                member,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            member_descriptor = os.open(
                member,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=descriptor,
            )
            try:
                entry = os.fstat(member_descriptor)
                member_identity = (
                    entry.st_dev,
                    entry.st_ino,
                    entry.st_size,
                    entry.st_mtime_ns,
                    stat.S_IMODE(entry.st_mode),
                )
                if (
                    not stat.S_ISREG(entry.st_mode)
                    or entry.st_nlink != 1
                    or entry.st_uid != os.geteuid()
                    or stat.S_IMODE(entry.st_mode) != 0o400
                    or member_identity
                    != (
                        named_member.st_dev,
                        named_member.st_ino,
                        named_member.st_size,
                        named_member.st_mtime_ns,
                        stat.S_IMODE(named_member.st_mode),
                    )
                    or named_member.st_uid != os.geteuid()
                ):
                    _fail(f"published member {member!r} is not sealed and single-link")
                maximum_by_member = {
                    PDF_MEMBER: MAX_PDF_BYTES,
                    SOURCE_MEMBER: MAX_SOURCE_BYTES,
                    TEMPLATE_MEMBER: MAX_JSON_BYTES,
                    CONFIG_MEMBER: MAX_JSON_BYTES,
                    MANIFEST_MEMBER: MAX_MANIFEST_BYTES,
                }
                raw_by_member[member] = _read_fd(
                    member_descriptor,
                    maximum=maximum_by_member[member],
                    context=f"published {member}",
                )
                named_after = os.stat(
                    member,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                after = os.fstat(member_descriptor)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    stat.S_IMODE(after.st_mode),
                ) != member_identity or (
                    named_after.st_dev,
                    named_after.st_ino,
                    named_after.st_size,
                    named_after.st_mtime_ns,
                    stat.S_IMODE(named_after.st_mode),
                ) != member_identity:
                    _fail(f"published member {member!r} changed while read")
            finally:
                _close_descriptor(
                    member_descriptor,
                    context=f"published member {member}",
                    primary_error=sys.exception(),
                )
        manifest_raw = raw_by_member[MANIFEST_MEMBER]
        if _sha256(manifest_raw) != expected:
            _fail("published manifest does not match its independent SHA-256 anchor")
        parsed = _json_without_duplicates(manifest_raw, context="published manifest")
        if not isinstance(parsed, dict) or parsed.get("schema") != SCHEMA:
            _fail("published manifest schema is invalid")
        expected_members = parsed.get("members")
        if not isinstance(expected_members, list):
            _fail("published manifest member inventory is invalid")
        observed_records = [
            {
                "member": member,
                "sha256": _sha256(raw_by_member[member]),
                "size": len(raw_by_member[member]),
            }
            for member in MEMBER_ORDER
        ]
        if expected_members != observed_records:
            _fail("published members do not match the anchored manifest")
        if _canonical_json(parsed) != manifest_raw:
            _fail("published manifest is not canonical JSON")
        if (
            _bounded_directory_names(
                descriptor,
                maximum=len(MEMBER_ORDER) + 2,
                context="published root terminal inventory",
            )
            != inventory
        ):
            _fail("published root inventory changed during readback")
        terminal_root = os.fstat(descriptor)
        terminal_named = os.lstat(absolute)
        terminal_resolved = absolute.resolve(strict=True)
        terminal_parent = os.lstat(parent_path)
        if (
            (
                terminal_root.st_dev,
                terminal_root.st_ino,
                terminal_root.st_size,
                terminal_root.st_mtime_ns,
                stat.S_IMODE(terminal_root.st_mode),
            )
            != root_identity
            or (
                terminal_named.st_dev,
                terminal_named.st_ino,
                terminal_named.st_size,
                terminal_named.st_mtime_ns,
                stat.S_IMODE(terminal_named.st_mode),
            )
            != root_identity
            or terminal_resolved != absolute
            or (
                terminal_parent.st_dev,
                terminal_parent.st_ino,
                terminal_parent.st_uid,
                stat.S_IMODE(terminal_parent.st_mode),
            )
            != (
                parent_entry.st_dev,
                parent_entry.st_ino,
                parent_entry.st_uid,
                stat.S_IMODE(parent_entry.st_mode),
            )
            or parent_path.resolve(strict=True) != parent_path
        ):
            _fail("published root identity changed during readback")
        return manifest_raw, parsed, (root_entry.st_dev, root_entry.st_ino)
    finally:
        _close_descriptor(
            descriptor,
            context="published root",
            primary_error=sys.exception(),
        )


def _canonical_inputs(snapshots: Mapping[str, _Snapshot]) -> dict[str, bytes]:
    source_raw = _snapshot_payload(snapshots, "source", maximum=MAX_SOURCE_BYTES)
    template_raw = _snapshot_payload(snapshots, "template", maximum=MAX_JSON_BYTES)
    config_raw = _snapshot_payload(snapshots, "config", maximum=MAX_JSON_BYTES)
    return {
        SOURCE_MEMBER: _canonicalize_markdown(source_raw),
        TEMPLATE_MEMBER: _canonical_json(
            _normalize_template(
                _json_without_duplicates(template_raw, context="template")
            ),
        ),
        CONFIG_MEMBER: _canonical_json(
            _normalize_config(_json_without_duplicates(config_raw, context="config")),
        ),
    }


def _read_derivation_bundle_fd(descriptor: int) -> tuple[bytes, dict[str, object]]:
    if descriptor <= 2:
        _fail("derivation bundle must arrive on an inherited non-stdio descriptor")
    try:
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        before = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot inspect inherited derivation-bundle descriptor: {error}")
    if flags & os.O_ACCMODE != os.O_RDONLY:
        _fail("inherited derivation-bundle descriptor must be read-only")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) & 0o022
        or not 1 <= before.st_size <= MAX_DERIVATION_BUNDLE_BYTES
    ):
        _fail("inherited derivation bundle is not a bounded private regular file")
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        stat.S_IMODE(before.st_mode),
    )
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(READ_CHUNK_BYTES, before.st_size - offset),
                offset,
            )
        except OSError as error:
            _fail(f"cannot read inherited derivation bundle: {error}")
        if not chunk:
            _fail("inherited derivation bundle ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    try:
        after = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot revalidate inherited derivation bundle: {error}")
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        stat.S_IMODE(after.st_mode),
    ) != identity:
        _fail("inherited derivation bundle changed while it was read")
    raw = b"".join(chunks)
    parsed = _json_without_duplicates(raw, context="derivation bundle")
    normalized = _normalize_derivation_bundle(parsed)
    if _canonical_json(normalized) != raw:
        _fail("derivation bundle is not exact canonical JSON")
    return raw, normalized


def _bundle_input_bytes(bundle: Mapping[str, object]) -> dict[str, bytes]:
    return {
        str(item["member"]): _decode_canonical_base64(
            item["base64"],
            context=f"derivation bundle input {item['member']}",
            maximum={
                SOURCE_MEMBER: MAX_SOURCE_BYTES,
                TEMPLATE_MEMBER: MAX_JSON_BYTES,
                CONFIG_MEMBER: MAX_JSON_BYTES,
            }[str(item["member"])],
        )
        for item in bundle["canonical_inputs"]
    }


def _validate_derivation_dependency_anchors(
    pins: Mapping[str, object],
    reportlab_bundle: _ReportLabBundle,
    bundle: Mapping[str, object],
) -> None:
    dependencies = bundle["dependencies"]
    inputs = {str(item["member"]): item for item in bundle["canonical_inputs"]}
    fonts = {str(item["role"]): item for item in dependencies["fonts"]}
    tools = {str(item["name"]): item for item in dependencies["tools"]}
    _validate_expected_hashes(
        pins,
        expected_source_sha256=str(inputs[SOURCE_MEMBER]["sha256"]),
        expected_template_sha256=str(inputs[TEMPLATE_MEMBER]["sha256"]),
        expected_config_sha256=str(inputs[CONFIG_MEMBER]["sha256"]),
        expected_regular_font_sha256=str(fonts["regular"]["sha256"]),
        expected_bold_font_sha256=str(fonts["bold"]["sha256"]),
        expected_machine_runner_sha256=str(dependencies["machine_runner"]["sha256"]),
        expected_builder_sha256=str(dependencies["renderer"]["sha256"]),
        expected_tool_sha256={
            "python": str(dependencies["runtime"]["sha256"]),
            **{name: str(tools[name]["sha256"]) for name in TOOL_ORDER[1:]},
        },
    )
    size_expectations = {
        "builder": int(dependencies["renderer"]["bytes"]),
        "machine_runner": int(dependencies["machine_runner"]["bytes"]),
        "regular_font": int(fonts["regular"]["bytes"]),
        "bold_font": int(fonts["bold"]["bytes"]),
        "tool:python": int(dependencies["runtime"]["bytes"]),
        **{f"tool:{name}": int(tools[name]["bytes"]) for name in TOOL_ORDER[1:]},
    }
    for name, expected_size in size_expectations.items():
        if pins[name].size != expected_size:
            _fail(f"{name} does not match its bundle byte-size anchor")
    observed_font_names = _validate_font_roles(
        _pinned_bytes(
            pins["regular_font"],
            maximum=MAX_FONT_BYTES,
            context="regular font",
        ),
        _pinned_bytes(
            pins["bold_font"],
            maximum=MAX_FONT_BYTES,
            context="bold font",
        ),
    )
    declared_font_names = tuple(
        str(fonts[role]["postscript_name"]) for role in ("regular", "bold")
    )
    if declared_font_names != observed_font_names:
        _fail("bundle font PostScript names differ from the pinned TTF roles")
    reportlab = dependencies["reportlab"]
    observed_reportlab = {
        "tree_sha256": reportlab_bundle.tree_sha256,
        "file_count": reportlab_bundle.file_count,
        "directory_count": reportlab_bundle.directory_count,
        "entry_count": reportlab_bundle.entry_count,
        "total_bytes": reportlab_bundle.total_bytes,
        "bundle_sha256": _sha256(reportlab_bundle.raw),
        "bundle_bytes": len(reportlab_bundle.raw),
    }
    expected_reportlab = {
        key: reportlab[key]
        for key in (
            "tree_sha256",
            "file_count",
            "directory_count",
            "entry_count",
            "total_bytes",
            "bundle_sha256",
            "bundle_bytes",
        )
    }
    if observed_reportlab != expected_reportlab:
        _fail("ReportLab dependency differs from its complete bundle anchor")


def derive_rebuttal_pdf_from_bundle_fd(descriptor: int) -> bytes:
    """Regenerate one exact PDF from a canonical inherited source bundle."""
    _, bundle = _read_derivation_bundle_fd(descriptor)
    dependency_pins: dict[str, object] = {}
    snapshots: dict[str, _Snapshot] = {}
    failure: BaseException | None = None
    pdf_raw: bytes | None = None
    try:
        dependency_pins, reportlab_root = _pin_derivation_dependencies(bundle)
        raw_inputs = _bundle_input_bytes(bundle)
        snapshots["source"] = _snapshot_bytes(
            raw_inputs[SOURCE_MEMBER], context="source"
        )
        snapshots["template"] = _snapshot_bytes(
            raw_inputs[TEMPLATE_MEMBER], context="template"
        )
        snapshots["config"] = _snapshot_bytes(
            raw_inputs[CONFIG_MEMBER], context="config"
        )
        for name, maximum in (
            ("regular_font", MAX_FONT_BYTES),
            ("bold_font", MAX_FONT_BYTES),
            ("builder", MAX_SOURCE_BYTES),
        ):
            snapshots[name] = _snapshot_bytes(
                _pinned_bytes(dependency_pins[name], maximum=maximum, context=name),
                context=name,
            )
        pins = {
            **dependency_pins,
            "source": snapshots["source"],
            "template": snapshots["template"],
            "config": snapshots["config"],
        }
        reportlab_bundle = _inventory_reportlab(reportlab_root)
        _validate_derivation_dependency_anchors(pins, reportlab_bundle, bundle)
        production = _produce(
            pins,
            snapshots,
            reportlab_bundle,
            expected_reportlab_tree_sha256=str(
                bundle["dependencies"]["reportlab"]["tree_sha256"]
            ),
            release_id=str(bundle["release_id"]),
        )
        expected_manifest = bundle["expected_output"]["renderer_manifest"]
        expected_pdf = bundle["expected_output"]["pdf"]
        if (
            len(production.manifest_raw) != expected_manifest["bytes"]
            or _sha256(production.manifest_raw) != expected_manifest["sha256"]
        ):
            _fail("fresh renderer manifest differs from the bundle expectation")
        if (
            len(production.pdf_raw) != expected_pdf["bytes"]
            or _sha256(production.pdf_raw) != expected_pdf["sha256"]
        ):
            _fail("fresh renderer PDF differs from the bundle expectation")
        _revalidate_all(pins)
        _revalidate_snapshots(snapshots)
        pdf_raw = production.pdf_raw
    except BaseException as error:  # noqa: BLE001 - preserve primary failure.
        failure = error
    try:
        _close_snapshots(snapshots, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    try:
        _close_pins(dependency_pins, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    if failure is not None:
        raise failure
    if pdf_raw is None:  # pragma: no cover - exhaustiveness
        _fail("derivation completed without PDF bytes")
    return pdf_raw


def _receipt(
    root: Path, production: _Production, *, replay_root: Path | None
) -> RebuttalRenderReceipt:
    return RebuttalRenderReceipt(
        manifest_path=str(root / MANIFEST_MEMBER),
        manifest_sha256=_sha256(production.manifest_raw),
        pdf_path=str(root / PDF_MEMBER),
        pdf_sha256=_sha256(production.pdf_raw),
        page_count=production.page_count,
        quote_count=production.quote_count,
        response_count=production.response_count,
        replay_root=str(replay_root) if replay_root is not None else None,
    )


def _raise_materialized_failure(
    error: BaseException,
    *,
    published: Path | None = None,
    replay: Path | None = None,
) -> NoReturn:
    locations: list[str] = []
    if published is not None:
        locations.append(f"published_path={published.absolute()}")
    if replay is not None:
        locations.append(f"replay_path={replay.absolute()}")
    if locations:
        message = (
            f"{error}; {'; '.join(locations)}; "
            "publication_state=materialized-output-preserved-after-terminal-failure; "
            "do-not-auto-delete"
        )
        raise RebuttalRenderError(message) from error
    raise error


def build_rebuttal_pdf(
    source: Path,
    template: Path,
    config: Path,
    destination: Path,
    *,
    regular_font: Path,
    bold_font: Path,
    reportlab_root: Path,
    release_id: str,
    tool_paths: Mapping[str, Path],
    expected_source_sha256: str,
    expected_template_sha256: str,
    expected_config_sha256: str,
    expected_regular_font_sha256: str,
    expected_bold_font_sha256: str,
    expected_reportlab_tree_sha256: str,
    expected_machine_runner_sha256: str,
    expected_builder_sha256: str,
    expected_tool_sha256: Mapping[str, str],
) -> RebuttalRenderReceipt:
    """Build, double-render, validate, seal, and publish one rebuttal PDF."""
    release_id = _expect_token(release_id, context="release_id")
    builder = _builder_path()
    pins = _pin_inputs(
        source,
        template,
        config,
        regular_font,
        bold_font,
        builder,
        tool_paths,
        expected_machine_runner_sha256,
    )
    snapshots: dict[str, _Snapshot] = {}
    published: Path | None = None
    receipt: RebuttalRenderReceipt | None = None
    failure: BaseException | None = None
    try:
        _validate_expected_hashes(
            pins,
            expected_source_sha256=expected_source_sha256,
            expected_template_sha256=expected_template_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_regular_font_sha256=expected_regular_font_sha256,
            expected_bold_font_sha256=expected_bold_font_sha256,
            expected_machine_runner_sha256=expected_machine_runner_sha256,
            expected_builder_sha256=expected_builder_sha256,
            expected_tool_sha256=expected_tool_sha256,
        )
        snapshots = _snapshot_inputs(pins)
        bundle = _inventory_reportlab(reportlab_root)
        production = _produce(
            pins,
            snapshots,
            bundle,
            expected_reportlab_tree_sha256=expected_reportlab_tree_sha256,
            release_id=release_id,
        )
        canonical_inputs = _canonical_inputs(snapshots)
        _revalidate_all(pins)
        _revalidate_snapshots(snapshots)
        published = _publish(destination, production, canonical_inputs)
        manifest_raw, _, published_identity = _read_published(
            published,
            expected_manifest_sha256=_sha256(production.manifest_raw),
        )
        if manifest_raw != production.manifest_raw:
            _fail("published manifest differs from the in-memory production")
        _revalidate_all(pins)
        _revalidate_snapshots(snapshots)
        terminal_raw, _, terminal_identity = _read_published(
            published,
            expected_manifest_sha256=_sha256(production.manifest_raw),
        )
        if terminal_raw != manifest_raw or terminal_identity != published_identity:
            _fail("published output changed before terminal receipt")
        receipt = _receipt(published, production, replay_root=None)
    except BaseException as error:  # noqa: BLE001 - preserve materialized state.
        failure = error
    try:
        _close_snapshots(snapshots, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    try:
        _close_pins(pins, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    if failure is not None:
        _raise_materialized_failure(failure, published=published)
    if receipt is None:  # pragma: no cover - exhaustiveness
        _fail("build completed without a receipt")
    return receipt


def validate_rebuttal_pdf(
    source: Path,
    template: Path,
    config: Path,
    published_root: Path,
    replay_root: Path,
    *,
    regular_font: Path,
    bold_font: Path,
    reportlab_root: Path,
    release_id: str,
    expected_manifest_sha256: str,
    tool_paths: Mapping[str, Path],
    expected_source_sha256: str,
    expected_template_sha256: str,
    expected_config_sha256: str,
    expected_regular_font_sha256: str,
    expected_bold_font_sha256: str,
    expected_reportlab_tree_sha256: str,
    expected_machine_runner_sha256: str,
    expected_builder_sha256: str,
    expected_tool_sha256: Mapping[str, str],
) -> RebuttalRenderReceipt:
    """Independently replay one anchored publication into a retained new root."""
    original_raw, original, original_identity = _read_published(
        published_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    release_id = _expect_token(release_id, context="release_id")
    if original.get("release_id") != release_id:
        _fail("published manifest release_id does not match replay configuration")
    builder = _builder_path()
    pins = _pin_inputs(
        source,
        template,
        config,
        regular_font,
        bold_font,
        builder,
        tool_paths,
        expected_machine_runner_sha256,
    )
    snapshots: dict[str, _Snapshot] = {}
    replay: Path | None = None
    receipt: RebuttalRenderReceipt | None = None
    failure: BaseException | None = None
    try:
        _validate_expected_hashes(
            pins,
            expected_source_sha256=expected_source_sha256,
            expected_template_sha256=expected_template_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_regular_font_sha256=expected_regular_font_sha256,
            expected_bold_font_sha256=expected_bold_font_sha256,
            expected_machine_runner_sha256=expected_machine_runner_sha256,
            expected_builder_sha256=expected_builder_sha256,
            expected_tool_sha256=expected_tool_sha256,
        )
        snapshots = _snapshot_inputs(pins)
        bundle = _inventory_reportlab(reportlab_root)
        production = _produce(
            pins,
            snapshots,
            bundle,
            expected_reportlab_tree_sha256=expected_reportlab_tree_sha256,
            release_id=release_id,
        )
        if production.manifest_raw != original_raw:
            _fail("independent replay manifest does not match anchored publication")
        canonical_inputs = _canonical_inputs(snapshots)
        _revalidate_all(pins)
        _revalidate_snapshots(snapshots)
        replay = _publish(replay_root, production, canonical_inputs)
        replay_raw, _, replay_identity = _read_published(
            replay,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if replay_raw != original_raw:
            _fail("published replay does not match anchored publication")
        _revalidate_all(pins)
        _revalidate_snapshots(snapshots)
        terminal_original, _, terminal_original_identity = _read_published(
            published_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        terminal_replay, _, terminal_replay_identity = _read_published(
            replay,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if (
            terminal_original != original_raw
            or terminal_original_identity != original_identity
            or terminal_replay != replay_raw
            or terminal_replay_identity != replay_identity
        ):
            _fail("published source or replay changed before terminal receipt")
        receipt = _receipt(
            published_root.absolute(),
            production,
            replay_root=replay,
        )
    except BaseException as error:  # noqa: BLE001 - preserve materialized state.
        failure = error
    try:
        _close_snapshots(snapshots, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    try:
        _close_pins(pins, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    if failure is not None:
        _raise_materialized_failure(
            failure,
            published=published_root if replay is not None else None,
            replay=replay,
        )
    if receipt is None:  # pragma: no cover - exhaustiveness
        _fail("validation completed without a receipt")
    return receipt


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--regular-font", type=Path, required=True)
    parser.add_argument("--regular-font-sha256", required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    parser.add_argument("--bold-font-sha256", required=True)
    parser.add_argument("--reportlab-root", type=Path, required=True)
    parser.add_argument("--reportlab-tree-sha256", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--machine-runner-sha256", required=True)
    parser.add_argument("--builder-sha256", required=True)
    for tool in TOOL_ORDER:
        parser.add_argument(f"--{tool}", type=Path, required=True)
        parser.add_argument(f"--{tool}-sha256", required=True)


def _cli_kwargs(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        "regular_font": arguments.regular_font,
        "bold_font": arguments.bold_font,
        "reportlab_root": arguments.reportlab_root,
        "release_id": arguments.release_id,
        "tool_paths": {name: getattr(arguments, name) for name in TOOL_ORDER},
        "expected_source_sha256": arguments.source_sha256,
        "expected_template_sha256": arguments.template_sha256,
        "expected_config_sha256": arguments.config_sha256,
        "expected_regular_font_sha256": arguments.regular_font_sha256,
        "expected_bold_font_sha256": arguments.bold_font_sha256,
        "expected_reportlab_tree_sha256": arguments.reportlab_tree_sha256,
        "expected_machine_runner_sha256": arguments.machine_runner_sha256,
        "expected_builder_sha256": arguments.builder_sha256,
        "expected_tool_sha256": {
            name: getattr(arguments, f"{name}_sha256") for name in TOOL_ORDER
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    digest = subparsers.add_parser(
        "dependency-digest",
        help="print the canonical pure-Python ReportLab tree anchor",
    )
    digest.add_argument("--reportlab-root", type=Path, required=True)
    build = subparsers.add_parser(
        "build", help="build and publish one sealed rebuttal render"
    )
    _add_common_arguments(build)
    build.add_argument("--destination", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", help="replay one anchored rebuttal render"
    )
    _add_common_arguments(validate)
    validate.add_argument("--published-root", type=Path, required=True)
    validate.add_argument("--replay-root", type=Path, required=True)
    validate.add_argument("--manifest-sha256", required=True)
    return parser


def _derivation_cli(arguments: Sequence[str]) -> int:
    expected_prefix = [
        "--dialect-derivation-protocol",
        DERIVATION_PROTOCOL,
        "--pdf-id",
        DERIVATION_ROLE,
        "--source-fd",
    ]
    expected_suffix = ["--pdf-output", "stdout"]
    if (
        len(arguments) != 8
        or list(arguments[:5]) != expected_prefix
        or list(arguments[6:]) != expected_suffix
        or re.fullmatch(r"(?:[3-9]|[1-9][0-9]+)", arguments[5]) is None
    ):
        _fail("fixed rebuttal derivation argument protocol is invalid")
    descriptor = int(arguments[5])
    pdf_raw = derive_rebuttal_pdf_from_bundle_fd(descriptor)
    _write_all(1, pdf_raw, context="derived PDF stdout")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run dependency-anchor, build, validate, or internal-render mode."""
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "--internal-render":
        return _internal_render(arguments[1:])
    if arguments and arguments[0] == "--dialect-derivation-protocol":
        return _derivation_cli(arguments)
    parsed = _parser().parse_args(arguments)
    if parsed.command == "dependency-digest":
        print(
            json.dumps(
                reportlab_dependency_digest(parsed.reportlab_root), sort_keys=True
            )
        )
        return 0
    kwargs = _cli_kwargs(parsed)
    if parsed.command == "build":
        receipt = build_rebuttal_pdf(
            parsed.source,
            parsed.template,
            parsed.config,
            parsed.destination,
            **kwargs,
        )
    else:
        receipt = validate_rebuttal_pdf(
            parsed.source,
            parsed.template,
            parsed.config,
            parsed.published_root,
            parsed.replay_root,
            expected_manifest_sha256=parsed.manifest_sha256,
            **kwargs,
        )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

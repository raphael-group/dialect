"""Build one canonical, result-blind rebuttal derivation source bundle.

The schema has canonical Markdown, template, and configuration input members
plus exact dependency and expected-output anchors; it has no dedicated
pre-rendered PDF member.  Markdown may contain arbitrary text, including text
that encodes PDF-like bytes.  The producer never decodes or consumes that text
as a PDF input: it renders a fresh PDF from the three source members and
requires the exact expected renderer manifest and PDF hashes before writing PDF
bytes to stdout.

This boundary does not provide a native launcher, producer authority, scientific
review, visual approval, coauthor approval, journal acceptance, or upload status.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import render_tcga_revision_rebuttal as renderer

# This is a narrow adapter-source boundary and intentionally reuses the fully
# tested renderer's descriptor, canonicalization, privacy, and dependency seams.
# ruff: noqa: COM812, PLR0913, SLF001

# Shallow defense for outer JSON fields. Each canonical input member is scanned
# separately after base64 decoding and before publication.
FORBIDDEN_OUTER_BUNDLE_PATH_FRAGMENTS = (
    b"/Users/",
    b"/private/",
    b"/tmp/",
    b"/var/",
    b"/opt/",
    b"/usr/",
    b"/System/",
    b"\\Users\\",
    b".codex",
    b".agents",
    b".cache",
    b".git/",
    b"research/",
    b"output/",
)


class RebuttalDerivationBundleError(ValueError):
    """Raised when a rebuttal source bundle cannot be built safely."""


@dataclass(frozen=True, slots=True)
class RebuttalDerivationBundleReceipt:
    """Summarize one sealed canonical source bundle."""

    bundle_path: str
    bundle_sha256: str
    bundle_bytes: int
    release_id: str
    role: str
    producer_protocol: str
    expected_renderer_manifest_sha256: str
    expected_pdf_sha256: str
    contains_pre_rendered_pdf_member: bool
    promotable: bool


def _fail(message: str) -> NoReturn:
    raise RebuttalDerivationBundleError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _input_record(member: str, raw: bytes) -> dict[str, object]:
    return {
        "member": member,
        "encoding": "base64",
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _member_record(manifest: Mapping[str, object], member: str) -> dict[str, object]:
    members = manifest.get("members")
    if not isinstance(members, list):
        _fail("expected renderer manifest has no exact member inventory")
    matches = [
        item
        for item in members
        if isinstance(item, dict) and item.get("member") == member
    ]
    if len(matches) != 1:
        _fail(f"expected renderer manifest does not bind {member!r} exactly once")
    item = matches[0]
    if set(item) != {"member", "sha256", "size"}:
        _fail(f"expected renderer manifest member {member!r} is malformed")
    return item


def _require_manifest_cross_bindings(
    manifest: Mapping[str, object],
    *,
    release_id: str,
    canonical_inputs: Mapping[str, bytes],
    pins: Mapping[str, object],
    reportlab_bundle: object,
    expected_pdf_sha256: str,
    expected_pdf_bytes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        manifest.get("schema") != renderer.SCHEMA
        or manifest.get("contract") != renderer.CONTRACT
        or manifest.get("release_id") != release_id
    ):
        _fail("expected renderer manifest schema, contract, or release_id drifted")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "source",
        "template",
        "config",
        "fonts",
        "reportlab",
        "builder",
        "machine_runner",
        "tools",
    }:
        _fail("expected renderer manifest input inventory drifted")
    for name, member in (
        ("source", renderer.SOURCE_MEMBER),
        ("template", renderer.TEMPLATE_MEMBER),
        ("config", renderer.CONFIG_MEMBER),
    ):
        record = inputs[name]
        expected = {
            "raw_sha256": _sha256(canonical_inputs[member]),
            "canonical_sha256": _sha256(canonical_inputs[member]),
            "canonical_size": len(canonical_inputs[member]),
        }
        if record != expected:
            _fail(
                f"expected renderer manifest {name} input was not built from "
                "the exact canonical bundle bytes"
            )
        if _member_record(manifest, member) != {
            "member": member,
            "sha256": expected["canonical_sha256"],
            "size": expected["canonical_size"],
        }:
            _fail(f"expected renderer manifest member {member!r} drifted")

    expected_builder = {
        "member": "analysis/render_tcga_revision_rebuttal.py",
        "sha256": pins["builder"].sha256,
        "size": pins["builder"].size,
    }
    if inputs["builder"] != expected_builder:
        _fail("expected renderer manifest binds different renderer bytes")
    expected_runner = {
        "member": renderer.MACHINE_RUNNER_MEMBER,
        "sha256": pins["machine_runner"].sha256,
        "size": pins["machine_runner"].size,
        "load_protocol": "caller-sha-before-private-descriptor-bytecode-exec",
    }
    if inputs["machine_runner"] != expected_runner:
        _fail("expected renderer manifest binds different machine-runner bytes")

    tools = inputs["tools"]
    if not isinstance(tools, list) or len(tools) != len(renderer.TOOL_ORDER):
        _fail("expected renderer manifest tool inventory drifted")
    expected_tool_records = [
        {
            "name": name,
            "sha256": pins[f"tool:{name}"].sha256,
            "size": pins[f"tool:{name}"].size,
            "host_path_recorded": False,
        }
        for name in renderer.TOOL_ORDER
    ]
    if tools != expected_tool_records:
        _fail("expected renderer manifest binds different tool bytes")

    fonts = inputs["fonts"]
    if not isinstance(fonts, list) or len(fonts) != 2:
        _fail("expected renderer manifest font inventory drifted")
    observed_font_names = renderer._validate_font_roles(
        renderer._pinned_bytes(
            pins["regular_font"],
            maximum=renderer.MAX_FONT_BYTES,
            context="regular font",
        ),
        renderer._pinned_bytes(
            pins["bold_font"],
            maximum=renderer.MAX_FONT_BYTES,
            context="bold font",
        ),
    )
    font_records: list[dict[str, object]] = []
    for index, role in enumerate(("regular", "bold")):
        item = fonts[index]
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "sha256", "size", "postscript_name"}
            or item.get("role") != role
            or item.get("sha256") != pins[f"{role}_font"].sha256
            or item.get("size") != pins[f"{role}_font"].size
            or item.get("postscript_name") != observed_font_names[index]
        ):
            _fail(f"expected renderer manifest binds a different {role} font")
        font_records.append(dict(item))

    reportlab = inputs["reportlab"]
    expected_reportlab = {
        "tree_sha256": reportlab_bundle.tree_sha256,
        "file_count": reportlab_bundle.file_count,
        "directory_count": reportlab_bundle.directory_count,
        "entry_count": reportlab_bundle.entry_count,
        "total_bytes": reportlab_bundle.total_bytes,
        "bundle_sha256": _sha256(reportlab_bundle.raw),
        "bundle_size": len(reportlab_bundle.raw),
        "pure_python_bundle": True,
        "excluded": ["__pycache__", "*.pyc", "*.so", "*.dylib"],
    }
    if reportlab != expected_reportlab:
        _fail("expected renderer manifest binds a different ReportLab tree")

    pdf_record = _member_record(manifest, renderer.PDF_MEMBER)
    if pdf_record != {
        "member": renderer.PDF_MEMBER,
        "sha256": expected_pdf_sha256,
        "size": expected_pdf_bytes,
    }:
        _fail("expected renderer manifest PDF differs from caller anchors")
    integration = manifest.get("integration")
    if (
        not isinstance(integration, dict)
        or integration.get("four_role_derivation_adapter_protocol")
        != "renderer-fd-stdout-derive-mode-implemented"
        or integration.get("rebuttal_role_gate") != "not-cleared-by-this-renderer"
        or integration.get("promotion_gate") != "not-cleared-by-this-renderer"
    ):
        _fail("expected renderer manifest integration boundary drifted")
    claims = manifest.get("claims")
    if not isinstance(claims, dict) or claims != {
        "scientific_accuracy": "not_attested",
        "human_visual_approval": "required-separately",
        "journal_acceptance": "not_attested",
        "submission_status": "not_attested",
    }:
        _fail("expected renderer manifest claim boundary drifted")
    return expected_tool_records, font_records


def _validate_fixed_locator_path(locator: str, actual: Path) -> None:
    expected = renderer._resolve_fixed_derivation_locator(locator)
    try:
        resolved = actual.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve dependency for locator {locator!r}: {error}")
    if resolved != expected:
        _fail(f"dependency path does not match fixed locator {locator!r}")


def _safe_destination(destination: Path) -> tuple[Path, int]:
    absolute = destination.absolute()
    if absolute.name in {"", ".", ".."}:
        _fail("bundle destination must name one new file")
    parent = absolute.parent
    try:
        entry = os.lstat(parent)
        resolved = parent.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot inspect bundle destination parent: {error}")
    if (
        not stat.S_ISDIR(entry.st_mode)
        or stat.S_ISLNK(entry.st_mode)
        or resolved != parent
        or entry.st_uid != os.geteuid()
        or stat.S_IMODE(entry.st_mode) & 0o022
    ):
        _fail("bundle destination parent is not a safe canonical authority")
    if absolute.exists() or absolute.is_symlink():
        _fail("bundle destination already exists")
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
            _fail("bundle destination parent changed while opened")
    except BaseException as error:
        try:
            os.close(descriptor)
        except BaseException as close_error:  # noqa: BLE001 - retain both.
            message = f"{error}; bundle parent descriptor cleanup failed: {close_error}"
            raise RebuttalDerivationBundleError(message) from error
        raise
    return absolute, descriptor


def _publish_bundle(destination: Path, raw: bytes) -> Path:
    absolute, parent_descriptor = _safe_destination(destination)
    writable = -1
    readable = -1
    stage_name: str | None = None
    renamed = False
    failure: BaseException | None = None
    try:
        writable, stage_path = tempfile.mkstemp(
            prefix=".dialect-rebuttal-bundle-",
            dir=absolute.parent,
        )
        stage_name = Path(stage_path).name
        created = os.fstat(writable)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or created.st_uid != os.geteuid()
        ):
            _fail("cannot create a private bundle staging file")
        renderer._write_all(writable, raw, context="derivation bundle")
        os.fsync(writable)
        os.fchmod(writable, 0o400)
        readable = os.open(
            stage_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        sealed = os.fstat(readable)
        named = os.stat(stage_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            sealed.st_size != len(raw)
            or stat.S_IMODE(sealed.st_mode) != 0o400
            or sealed.st_nlink != 1
            or sealed.st_uid != os.geteuid()
            or (sealed.st_dev, sealed.st_ino, sealed.st_size, sealed.st_mtime_ns)
            != (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns)
        ):
            _fail("bundle staging file did not seal exactly")
        os.close(writable)
        writable = -1
        observed = renderer._read_fd(
            readable,
            maximum=len(raw),
            context="sealed derivation bundle",
        )
        if observed != raw:
            _fail("bundle staging bytes changed before publication")
        os.fsync(parent_descriptor)
        renderer._rename_no_replace(
            parent_descriptor,
            stage_name,
            absolute.name,
        )
        renamed = True
        os.fsync(parent_descriptor)
        published = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            (published.st_dev, published.st_ino, published.st_size)
            != (sealed.st_dev, sealed.st_ino, sealed.st_size)
            or stat.S_IMODE(published.st_mode) != 0o400
            or published.st_nlink != 1
        ):
            _fail("published bundle identity differs from the sealed stage")
        terminal = os.fstat(readable)
        terminal_raw = renderer._read_fd(
            readable,
            maximum=len(raw),
            context="published derivation bundle",
        )
        terminal_named = os.lstat(absolute)
        if (
            terminal_raw != raw
            or (
                terminal.st_dev,
                terminal.st_ino,
                terminal.st_size,
                terminal.st_mtime_ns,
                stat.S_IMODE(terminal.st_mode),
            )
            != (
                sealed.st_dev,
                sealed.st_ino,
                sealed.st_size,
                sealed.st_mtime_ns,
                0o400,
            )
            or (
                terminal_named.st_dev,
                terminal_named.st_ino,
                terminal_named.st_size,
                terminal_named.st_mtime_ns,
                stat.S_IMODE(terminal_named.st_mode),
            )
            != (
                sealed.st_dev,
                sealed.st_ino,
                sealed.st_size,
                sealed.st_mtime_ns,
                0o400,
            )
            or absolute.resolve(strict=True) != absolute
        ):
            _fail("published bundle path is not canonical")
    except BaseException as error:  # noqa: BLE001 - cleanup preserves any failure.
        failure = error

    close_errors: list[str] = []
    for context, descriptor in (
        ("readable bundle", readable),
        ("writable bundle", writable),
        ("bundle parent", parent_descriptor),
    ):
        if descriptor < 0:
            continue
        try:
            os.close(descriptor)
        except BaseException as error:  # noqa: BLE001 - attempt every close.
            close_errors.append(f"{context} descriptor cleanup failed: {error}")
    if close_errors:
        cleanup = "; ".join(close_errors)
        message = cleanup if failure is None else f"{failure}; {cleanup}"
        failure = RebuttalDerivationBundleError(message)
    if failure is not None:
        location = (
            absolute
            if renamed
            else (absolute.parent / stage_name if stage_name is not None else None)
        )
        message = str(failure)
        if location is not None:
            message = (
                f"{message}; materialized_path={location}; "
                "publication_state=preserved-after-failure; do-not-auto-delete"
            )
        raise RebuttalDerivationBundleError(message) from failure
    if not renamed:  # pragma: no cover - exhaustiveness
        _fail("bundle publication completed without a rename")
    return absolute


def build_rebuttal_derivation_bundle(
    source: Path,
    template: Path,
    config: Path,
    published_root: Path,
    destination: Path,
    *,
    regular_font: Path,
    bold_font: Path,
    reportlab_root: Path,
    runtime: Path,
    release_id: str,
    tool_paths: Mapping[str, Path],
    expected_source_sha256: str,
    expected_template_sha256: str,
    expected_config_sha256: str,
    expected_regular_font_sha256: str,
    expected_bold_font_sha256: str,
    expected_reportlab_tree_sha256: str,
    expected_machine_runner_sha256: str,
    expected_renderer_sha256: str,
    expected_tool_sha256: Mapping[str, str],
    expected_manifest_sha256: str,
    expected_pdf_sha256: str,
    expected_pdf_bytes: int,
) -> RebuttalDerivationBundleReceipt:
    """Build and atomically publish one canonical source-only bundle."""
    release_id = renderer._expect_token(release_id, context="release_id")
    if tool_paths.get("python") != runtime:
        _fail("tool_paths.python must equal the configured invoking runtime")
    _validate_fixed_locator_path(
        renderer.DERIVATION_LOCATORS["regular_font"], regular_font
    )
    _validate_fixed_locator_path(renderer.DERIVATION_LOCATORS["bold_font"], bold_font)
    for name in renderer.TOOL_ORDER[1:]:
        if name not in tool_paths:
            _fail(f"tool_paths is missing {name!r}")
        _validate_fixed_locator_path(
            renderer.DERIVATION_LOCATORS[name], tool_paths[name]
        )
    if set(tool_paths) != set(renderer.TOOL_ORDER):
        _fail(f"tool_paths must contain exactly {list(renderer.TOOL_ORDER)}")

    pins = renderer._pin_inputs(
        source,
        template,
        config,
        regular_font,
        bold_font,
        renderer._builder_path(),
        tool_paths,
        expected_machine_runner_sha256,
    )
    snapshots: dict[str, renderer._Snapshot] = {}
    failure: BaseException | None = None
    receipt: RebuttalDerivationBundleReceipt | None = None
    published_bundle: Path | None = None
    try:
        renderer._validate_expected_hashes(
            pins,
            expected_source_sha256=expected_source_sha256,
            expected_template_sha256=expected_template_sha256,
            expected_config_sha256=expected_config_sha256,
            expected_regular_font_sha256=expected_regular_font_sha256,
            expected_bold_font_sha256=expected_bold_font_sha256,
            expected_machine_runner_sha256=expected_machine_runner_sha256,
            expected_builder_sha256=expected_renderer_sha256,
            expected_tool_sha256=expected_tool_sha256,
        )
        snapshots = renderer._snapshot_inputs(pins)
        snapshot_payloads: dict[str, bytes] = {}
        for name, member in (
            ("source", renderer.SOURCE_MEMBER),
            ("template", renderer.TEMPLATE_MEMBER),
            ("config", renderer.CONFIG_MEMBER),
        ):
            raw = renderer._snapshot_payload(
                snapshots,
                name,
                maximum=(
                    renderer.MAX_SOURCE_BYTES
                    if name == "source"
                    else renderer.MAX_JSON_BYTES
                ),
            )
            renderer._reject_private_paths_in_canonical_member(raw, member=member)
            snapshot_payloads[name] = raw
        canonical_inputs = renderer._canonical_inputs(snapshots)
        for name, member in (
            ("source", renderer.SOURCE_MEMBER),
            ("template", renderer.TEMPLATE_MEMBER),
            ("config", renderer.CONFIG_MEMBER),
        ):
            raw = snapshot_payloads[name]
            if raw != canonical_inputs[member]:
                _fail(f"{name} must already be exact canonical bundle bytes")

        reportlab_bundle = renderer._inventory_reportlab(reportlab_root)
        if reportlab_bundle.tree_sha256 != renderer._expect_sha256(
            expected_reportlab_tree_sha256,
            context="expected ReportLab tree SHA-256",
        ):
            _fail("ReportLab tree differs from its caller SHA-256 anchor")
        manifest_raw, manifest, _ = renderer._read_published(
            published_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        expected_pdf_sha256 = renderer._expect_sha256(
            expected_pdf_sha256, context="expected PDF SHA-256"
        )
        expected_pdf_bytes = renderer._expect_bounded_int(
            expected_pdf_bytes,
            context="expected PDF bytes",
            minimum=len(renderer.PDF_SIGNATURE) + len(renderer.PDF_EOF),
            maximum=renderer.MAX_PDF_BYTES,
        )
        _, font_records = _require_manifest_cross_bindings(
            manifest,
            release_id=release_id,
            canonical_inputs=canonical_inputs,
            pins=pins,
            reportlab_bundle=reportlab_bundle,
            expected_pdf_sha256=expected_pdf_sha256,
            expected_pdf_bytes=expected_pdf_bytes,
        )
        runtime_record = pins["tool:python"]
        runtime_name = runtime.resolve(strict=True).name
        match = renderer.re.fullmatch(r"python(3\.[0-9]{1,2})", runtime_name)
        if match is None:
            _fail("invoking runtime filename must encode its Python major.minor tag")
        python_tag = match.group(1)
        expected_reportlab_path = renderer._derive_reportlab_path(
            runtime.resolve(strict=True), python_tag
        )
        if reportlab_root.resolve(strict=True) != expected_reportlab_path:
            _fail("ReportLab root is not derived from the invoking Python runtime")

        bundle = {
            "schema": renderer.DERIVATION_BUNDLE_SCHEMA,
            "contract": renderer.DERIVATION_BUNDLE_CONTRACT,
            "release_id": release_id,
            "role": renderer.DERIVATION_ROLE,
            "producer_protocol": renderer.DERIVATION_PROTOCOL,
            "producer_arguments": [
                "--dialect-derivation-protocol",
                renderer.DERIVATION_PROTOCOL,
                "--pdf-id",
                renderer.DERIVATION_ROLE,
                "--source-fd",
                "{source_fd}",
                "--pdf-output",
                "stdout",
            ],
            "canonical_inputs": [
                _input_record(member, canonical_inputs[member])
                for member in (
                    renderer.SOURCE_MEMBER,
                    renderer.TEMPLATE_MEMBER,
                    renderer.CONFIG_MEMBER,
                )
            ],
            "dependencies": {
                "renderer": {
                    "locator": renderer.DERIVATION_LOCATORS["renderer"],
                    "member": "analysis/render_tcga_revision_rebuttal.py",
                    "bytes": pins["builder"].size,
                    "sha256": pins["builder"].sha256,
                },
                "machine_runner": {
                    "locator": renderer.DERIVATION_LOCATORS["machine_runner"],
                    "member": renderer.MACHINE_RUNNER_MEMBER,
                    "bytes": pins["machine_runner"].size,
                    "sha256": pins["machine_runner"].sha256,
                },
                "runtime": {
                    "locator": renderer.DERIVATION_LOCATORS["runtime"],
                    "python_tag": python_tag,
                    "bytes": runtime_record.size,
                    "sha256": runtime_record.sha256,
                },
                "tools": [
                    {
                        "name": name,
                        "locator": renderer.DERIVATION_LOCATORS[name],
                        "bytes": pins[f"tool:{name}"].size,
                        "sha256": pins[f"tool:{name}"].sha256,
                    }
                    for name in renderer.TOOL_ORDER[1:]
                ],
                "fonts": [
                    {
                        "role": role,
                        "locator": renderer.DERIVATION_LOCATORS[f"{role}_font"],
                        "bytes": pins[f"{role}_font"].size,
                        "sha256": pins[f"{role}_font"].sha256,
                        "postscript_name": font_records[index]["postscript_name"],
                    }
                    for index, role in enumerate(("regular", "bold"))
                ],
                "reportlab": {
                    "locator": renderer.DERIVATION_LOCATORS["reportlab"],
                    "tree_sha256": reportlab_bundle.tree_sha256,
                    "file_count": reportlab_bundle.file_count,
                    "directory_count": reportlab_bundle.directory_count,
                    "entry_count": reportlab_bundle.entry_count,
                    "total_bytes": reportlab_bundle.total_bytes,
                    "bundle_sha256": _sha256(reportlab_bundle.raw),
                    "bundle_bytes": len(reportlab_bundle.raw),
                },
            },
            "expected_output": {
                "renderer_manifest": {
                    "member": renderer.MANIFEST_MEMBER,
                    "bytes": len(manifest_raw),
                    "sha256": _sha256(manifest_raw),
                },
                "pdf": {
                    "member": renderer.PDF_MEMBER,
                    "bytes": expected_pdf_bytes,
                    "sha256": expected_pdf_sha256,
                },
            },
            "non_inference": dict(renderer.DERIVATION_NON_INFERENCE),
        }
        normalized = renderer._normalize_derivation_bundle(bundle)
        raw = renderer._canonical_json(normalized)
        if len(raw) > renderer.MAX_DERIVATION_BUNDLE_BYTES:
            _fail("canonical rebuttal derivation bundle exceeds its byte limit")
        if any(fragment in raw for fragment in FORBIDDEN_OUTER_BUNDLE_PATH_FRAGMENTS):
            _fail("canonical outer bundle JSON contains a host-private path fragment")
        # Shallow defense in depth only. Canonical inputs are base64 members and
        # may themselves contain text that encodes arbitrary content; no recursive
        # content-classification property is claimed here.
        if renderer.PDF_SIGNATURE in raw:
            _fail("canonical bundle JSON contains a literal PDF signature")
        renderer._revalidate_all(pins)
        renderer._revalidate_snapshots(snapshots)
        published_bundle = _publish_bundle(destination, raw)
        renderer._revalidate_all(pins)
        renderer._revalidate_snapshots(snapshots)
        receipt = RebuttalDerivationBundleReceipt(
            bundle_path=str(published_bundle),
            bundle_sha256=_sha256(raw),
            bundle_bytes=len(raw),
            release_id=release_id,
            role=renderer.DERIVATION_ROLE,
            producer_protocol=renderer.DERIVATION_PROTOCOL,
            expected_renderer_manifest_sha256=_sha256(manifest_raw),
            expected_pdf_sha256=expected_pdf_sha256,
            contains_pre_rendered_pdf_member=False,
            promotable=False,
        )
    except BaseException as error:  # noqa: BLE001 - preserve primary failure.
        failure = error
    try:
        renderer._close_snapshots(snapshots, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    try:
        renderer._close_pins(pins, primary_error=failure)
    except BaseException as error:  # noqa: BLE001 - combine cleanup failure.
        failure = error
    if failure is not None:
        if published_bundle is not None:
            failure = RebuttalDerivationBundleError(
                f"{failure}; published_path={published_bundle}; "
                "publication_state=materialized-output-preserved-after-terminal-"
                "failure; do-not-auto-delete"
            )
        if isinstance(failure, RebuttalDerivationBundleError):
            raise failure
        raise RebuttalDerivationBundleError(str(failure)) from failure
    if receipt is None:  # pragma: no cover - exhaustiveness
        _fail("bundle build completed without a receipt")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--pdf-sha256", required=True)
    parser.add_argument("--pdf-bytes", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--regular-font", type=Path, required=True)
    parser.add_argument("--regular-font-sha256", required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    parser.add_argument("--bold-font-sha256", required=True)
    parser.add_argument("--reportlab-root", type=Path, required=True)
    parser.add_argument("--reportlab-tree-sha256", required=True)
    parser.add_argument("--machine-runner-sha256", required=True)
    parser.add_argument("--renderer-sha256", required=True)
    for tool in renderer.TOOL_ORDER[1:]:
        parser.add_argument(f"--{tool}", type=Path, required=True)
        parser.add_argument(f"--{tool}-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one canonical rebuttal derivation source bundle."""
    arguments = _parser().parse_args(argv)
    tools = {
        "python": arguments.runtime,
        **{name: getattr(arguments, name) for name in renderer.TOOL_ORDER[1:]},
    }
    receipt = build_rebuttal_derivation_bundle(
        arguments.source,
        arguments.template,
        arguments.config,
        arguments.published_root,
        arguments.destination,
        regular_font=arguments.regular_font,
        bold_font=arguments.bold_font,
        reportlab_root=arguments.reportlab_root,
        runtime=arguments.runtime,
        release_id=arguments.release_id,
        tool_paths=tools,
        expected_source_sha256=arguments.source_sha256,
        expected_template_sha256=arguments.template_sha256,
        expected_config_sha256=arguments.config_sha256,
        expected_regular_font_sha256=arguments.regular_font_sha256,
        expected_bold_font_sha256=arguments.bold_font_sha256,
        expected_reportlab_tree_sha256=arguments.reportlab_tree_sha256,
        expected_machine_runner_sha256=arguments.machine_runner_sha256,
        expected_renderer_sha256=arguments.renderer_sha256,
        expected_tool_sha256={
            "python": arguments.runtime_sha256,
            **{
                name: getattr(arguments, f"{name}_sha256")
                for name in renderer.TOOL_ORDER[1:]
            },
        },
        expected_manifest_sha256=arguments.manifest_sha256,
        expected_pdf_sha256=arguments.pdf_sha256,
        expected_pdf_bytes=arguments.pdf_bytes,
    )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

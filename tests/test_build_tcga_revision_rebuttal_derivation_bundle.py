"""Tests for the canonical rebuttal derivation source bundle and FD mode."""

# Synthetic prose and strict private seams intentionally make these tests verbose.
# ruff: noqa: COM812, E501, PLR0915, SLF001

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from analysis import build_tcga_revision_rebuttal_derivation_bundle as bundle_builder
from analysis import render_tcga_revision_rebuttal as renderer

REQUIRES_NATIVE_DARWIN = pytest.mark.skipif(
    sys.platform != "darwin" or os.uname().machine != "arm64",
    reason="native renderer execution and no-replace publication require arm64 Darwin",
)

BUNDLED_PYTHON = Path(
    os.environ.get(
        "DIALECT_REBUTTAL_PYTHON",
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12",
    ),
)
REGULAR_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

PDF_LIKE_TEXT = base64.b64encode(b"%PDF-1.7\nsynthetic inert text\n%%EOF\n").decode(
    "ascii"
)

SOURCE = f"""# Response to Reviewers

This synthetic fixture contains no scientific result.

Encoded PDF-like text remains inert reviewer prose: {PDF_LIKE_TEXT}

## Reviewer #1

### R1-1 - Clarify the synthetic method

<!-- SOURCE-COMMENT:r1-1:BEGIN -->
```text
Please distinguish the observed synthetic count from the latent synthetic state.
```
<!-- SOURCE-COMMENT:r1-1:END -->

We agree. The synthetic response now makes that distinction directly.
"""

TEMPLATE = {
    "schema": renderer.TEMPLATE_SCHEMA,
    "page_size": "letter",
    "margins": {"top": 60, "right": 58, "bottom": 54, "left": 58},
    "type": {
        "title": 24,
        "heading_2": 16,
        "heading_3": 12,
        "heading_4": 10,
        "body": 10.5,
        "leading": 15,
    },
    "colors": {
        "ink": "#17212B",
        "muted": "#5E6872",
        "accent": "#245C73",
        "quote_background": "#F2F5F6",
        "rule": "#CBD4D8",
    },
}

CONFIG = {
    "schema": renderer.CONFIG_SCHEMA,
    "release_id": "synthetic-rebuttal-bundle-v1",
    "manuscript_id": "PCOMPBIOL-SYNTHETIC-00000",
    "manuscript_title": "A synthetic manuscript used only for bundle QA",
    "response_title": "Response to Reviewers",
    "authors": ["Ada Example", "Ben Example"],
    "source_date_epoch": renderer.REPORTLAB_INVARIANT_EPOCH,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_inputs(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source.md"
    template = root / "template.json"
    config = root / "config.json"
    source.write_bytes(renderer._canonicalize_markdown(SOURCE.encode("utf-8")))
    template.write_bytes(
        renderer._canonical_json(renderer._normalize_template(TEMPLATE))
    )
    config.write_bytes(renderer._canonical_json(renderer._normalize_config(CONFIG)))
    return source, template, config


def _valid_bundle() -> dict[str, object]:
    source = renderer._canonicalize_markdown(SOURCE.encode("utf-8"))
    template = renderer._canonical_json(renderer._normalize_template(TEMPLATE))
    config = renderer._canonical_json(renderer._normalize_config(CONFIG))

    def encoded(member: str, raw: bytes) -> dict[str, object]:
        return {
            "member": member,
            "encoding": "base64",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii"),
        }

    digest = "1" * 64
    return {
        "schema": renderer.DERIVATION_BUNDLE_SCHEMA,
        "contract": renderer.DERIVATION_BUNDLE_CONTRACT,
        "release_id": CONFIG["release_id"],
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
            encoded(renderer.SOURCE_MEMBER, source),
            encoded(renderer.TEMPLATE_MEMBER, template),
            encoded(renderer.CONFIG_MEMBER, config),
        ],
        "dependencies": {
            "renderer": {
                "locator": renderer.DERIVATION_LOCATORS["renderer"],
                "member": "analysis/render_tcga_revision_rebuttal.py",
                "bytes": 10,
                "sha256": digest,
            },
            "machine_runner": {
                "locator": renderer.DERIVATION_LOCATORS["machine_runner"],
                "member": renderer.MACHINE_RUNNER_MEMBER,
                "bytes": 11,
                "sha256": "2" * 64,
            },
            "runtime": {
                "locator": renderer.DERIVATION_LOCATORS["runtime"],
                "python_tag": "3.12",
                "bytes": 12,
                "sha256": "3" * 64,
            },
            "tools": [
                {
                    "name": name,
                    "locator": renderer.DERIVATION_LOCATORS[name],
                    "bytes": 13 + index,
                    "sha256": str(4 + index) * 64,
                }
                for index, name in enumerate(renderer.TOOL_ORDER[1:])
            ],
            "fonts": [
                {
                    "role": "regular",
                    "locator": renderer.DERIVATION_LOCATORS["regular_font"],
                    "bytes": 20,
                    "sha256": "7" * 64,
                    "postscript_name": "ArialUnicodeMS",
                },
                {
                    "role": "bold",
                    "locator": renderer.DERIVATION_LOCATORS["bold_font"],
                    "bytes": 21,
                    "sha256": "8" * 64,
                    "postscript_name": "Arial-BoldMT",
                },
            ],
            "reportlab": {
                "locator": renderer.DERIVATION_LOCATORS["reportlab"],
                "tree_sha256": "9" * 64,
                "file_count": 10,
                "directory_count": 4,
                "entry_count": 14,
                "total_bytes": 100,
                "bundle_sha256": "a" * 64,
                "bundle_bytes": 120,
            },
        },
        "expected_output": {
            "renderer_manifest": {
                "member": renderer.MANIFEST_MEMBER,
                "bytes": 100,
                "sha256": "b" * 64,
            },
            "pdf": {
                "member": renderer.PDF_MEMBER,
                "bytes": 200,
                "sha256": "c" * 64,
            },
        },
        "non_inference": dict(renderer.DERIVATION_NON_INFERENCE),
    }


def _sealed_file(path: Path, raw: bytes) -> int:
    path.write_bytes(raw)
    path.chmod(0o400)
    return os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))


def _real_dependencies() -> tuple[dict[str, Path], Path]:
    missing = [
        path for path in (BUNDLED_PYTHON, REGULAR_FONT, BOLD_FONT) if not path.exists()
    ]
    if missing:
        pytest.skip(f"renderer dependencies unavailable: {missing}")
    tools = {
        "python": BUNDLED_PYTHON.resolve(strict=True),
        **{
            name: renderer._resolve_fixed_derivation_locator(
                renderer.DERIVATION_LOCATORS[name]
            )
            for name in renderer.TOOL_ORDER[1:]
        },
    }
    tag_match = renderer.re.fullmatch(r"python(3\.[0-9]{1,2})", tools["python"].name)
    if tag_match is None:
        pytest.skip("bundled Python filename does not encode its major.minor tag")
    reportlab = renderer._derive_reportlab_path(tools["python"], tag_match.group(1))
    return tools, reportlab


def _invoke_derive(
    runtime: Path,
    renderer_path: Path,
    bundle_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    descriptor = os.open(bundle_path, os.O_RDONLY)
    try:
        return subprocess.run(  # noqa: S603 - exact pinned synthetic runtime.
            [
                str(runtime),
                "-I",
                "-S",
                "-B",
                str(renderer_path),
                "--dialect-derivation-protocol",
                renderer.DERIVATION_PROTOCOL,
                "--pdf-id",
                renderer.DERIVATION_ROLE,
                "--source-fd",
                str(descriptor),
                "--pdf-output",
                "stdout",
            ],
            cwd="/",
            env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            pass_fds=(descriptor,),
            check=False,
            timeout=300,
        )
    finally:
        os.close(descriptor)


def test_bundle_normalization_has_no_pdf_input_member_and_is_pathless() -> None:
    normalized = renderer._normalize_derivation_bundle(_valid_bundle())
    raw = renderer._canonical_json(normalized)
    assert renderer._normalize_derivation_bundle(json.loads(raw)) == normalized
    assert renderer.PDF_SIGNATURE not in raw
    assert b"/Users/" not in raw
    assert b"/private/" not in raw
    assert b"/tmp/" not in raw
    assert b".cache" not in raw
    assert normalized["non_inference"]["pre_rendered_pdf_member"] == "absent"
    assert normalized["non_inference"]["producer_pdf_input"] == "none"
    assert (
        normalized["non_inference"]["decoded_canonical_input_private_paths"]
        == "rejected-as-utf8-text-without-recursive-decoding"
    )
    assert normalized["non_inference"]["recursive_content_classification"] == (
        "not-provided"
    )
    assert set(normalized["expected_output"]["pdf"]) == {
        "member",
        "bytes",
        "sha256",
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update({"unknown": True}), "must have exactly keys"),
        (
            lambda value: value["canonical_inputs"][0].update({"bytes": 1}),
            "size and SHA-256",
        ),
        (
            lambda value: value["canonical_inputs"][0].update({"sha256": "0" * 64}),
            "size and SHA-256",
        ),
        (
            lambda value: value["canonical_inputs"][0].update({"base64": "AAAA="}),
            "strict base64|canonical|size and SHA-256",
        ),
        (
            lambda value: value["dependencies"]["tools"][0].update(
                {"locator": "arbitrary-host-path"}
            ),
            "locator is invalid",
        ),
        (
            lambda value: value["expected_output"]["pdf"].update(
                {"base64": "JVBERi0="}
            ),
            "must have exactly keys",
        ),
        (
            lambda value: value.update({"role": "clean"}),
            "schema, contract, role, or protocol",
        ),
        (
            lambda value: value.update({"producer_arguments": []}),
            "producer_arguments drifted",
        ),
    ],
)
def test_bundle_normalization_rejects_unknown_drift_and_pdf_payload(
    mutate: object, match: str
) -> None:
    value = _valid_bundle()
    mutate(value)
    with pytest.raises(renderer.RebuttalRenderError, match=match):
        renderer._normalize_derivation_bundle(value)


@pytest.mark.parametrize(
    ("index", "raw"),
    [
        (
            0,
            renderer._canonicalize_markdown(
                SOURCE.replace(
                    "synthetic response",
                    "synthetic response at /Users/example/private.txt",
                ).encode("utf-8")
            ),
        ),
        (
            1,
            renderer._canonical_json(
                {**TEMPLATE, "schema": "/private/example/template.json"}
            ),
        ),
        (
            2,
            renderer._canonical_json(
                {**CONFIG, "manuscript_title": "/System/example/config.json"}
            ),
        ),
    ],
)
def test_bundle_normalization_rejects_private_path_in_each_decoded_input(
    index: int,
    raw: bytes,
) -> None:
    value = _valid_bundle()
    member = (
        renderer.SOURCE_MEMBER,
        renderer.TEMPLATE_MEMBER,
        renderer.CONFIG_MEMBER,
    )[index]
    value["canonical_inputs"][index] = {
        "member": member,
        "encoding": "base64",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }
    with pytest.raises(renderer.RebuttalRenderError, match="private absolute"):
        renderer._normalize_derivation_bundle(value)


def test_bundle_fd_rejects_duplicate_json_unknown_fields_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    canonical = renderer._canonical_json(
        renderer._normalize_derivation_bundle(_valid_bundle())
    )
    duplicate = canonical.replace(
        b'{"canonical_inputs":',
        b'{"schema":"duplicate","canonical_inputs":',
        1,
    )
    for index, (raw, match) in enumerate(
        (
            (duplicate, "duplicates JSON key"),
            (canonical[:-1] + b" \n", "exact canonical JSON"),
        )
    ):
        descriptor = _sealed_file(tmp_path / f"bundle-{index}.json", raw)
        try:
            with pytest.raises(renderer.RebuttalRenderError, match=match):
                renderer._read_derivation_bundle_fd(descriptor)
        finally:
            os.close(descriptor)


def test_bundle_fd_rejects_overflow_writable_descriptor_and_live_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = renderer._canonical_json(
        renderer._normalize_derivation_bundle(_valid_bundle())
    )
    writable_path = tmp_path / "writable.json"
    writable_path.write_bytes(raw)
    writable = os.open(writable_path, os.O_RDWR)
    writable_path.chmod(0o400)
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="read-only"):
            renderer._read_derivation_bundle_fd(writable)
    finally:
        os.close(writable)

    overflow_path = tmp_path / "overflow.json"
    overflow = _sealed_file(overflow_path, raw)
    monkeypatch.setattr(renderer, "MAX_DERIVATION_BUNDLE_BYTES", len(raw) - 1)
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="bounded private"):
            renderer._read_derivation_bundle_fd(overflow)
    finally:
        os.close(overflow)
    monkeypatch.setattr(renderer, "MAX_DERIVATION_BUNDLE_BYTES", 8 * 1024 * 1024)

    mutation_path = tmp_path / "mutation.json"
    mutation_path.write_bytes(raw)
    writer = os.open(mutation_path, os.O_WRONLY)
    mutation_path.chmod(0o400)
    reader = os.open(mutation_path, os.O_RDONLY)
    original_pread = os.pread
    mutated = False

    def mutate_after_read(descriptor: int, count: int, offset: int) -> bytes:
        nonlocal mutated
        result = original_pread(descriptor, count, offset)
        if not mutated:
            mutated = True
            os.pwrite(writer, b"X", 0)
            os.fsync(writer)
        return result

    monkeypatch.setattr(renderer.os, "pread", mutate_after_read)
    try:
        with pytest.raises(renderer.RebuttalRenderError, match="changed while"):
            renderer._read_derivation_bundle_fd(reader)
    finally:
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("failure_call", [2, 3])
def test_derive_closes_prior_snapshots_when_embedded_snapshot_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    raw = renderer._canonical_json(
        renderer._normalize_derivation_bundle(_valid_bundle())
    )
    bundle_path = tmp_path / f"snapshot-failure-{failure_call}.bundle"
    descriptor = _sealed_file(bundle_path, raw)
    original_snapshot = renderer._snapshot_bytes
    acquired: list[tuple[renderer._Snapshot, int]] = []
    calls = 0
    failure_message = f"injected snapshot failure on call {failure_call}"

    def fail_selected_snapshot(raw: bytes, *, context: str) -> renderer._Snapshot:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise renderer.RebuttalRenderError(failure_message)
        snapshot = original_snapshot(raw, context=context)
        acquired.append((snapshot, snapshot.descriptor))
        return snapshot

    monkeypatch.setattr(
        renderer,
        "_pin_derivation_dependencies",
        lambda _bundle: ({}, tmp_path / "unused-reportlab"),
    )
    monkeypatch.setattr(renderer, "_snapshot_bytes", fail_selected_snapshot)
    try:
        with pytest.raises(renderer.RebuttalRenderError, match=failure_message):
            renderer.derive_rebuttal_pdf_from_bundle_fd(descriptor)
    finally:
        os.close(descriptor)
    assert len(acquired) == failure_call - 1
    for snapshot, acquired_descriptor in acquired:
        assert snapshot.descriptor == -1
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(acquired_descriptor)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--dialect-derivation-protocol", renderer.DERIVATION_PROTOCOL],
        [
            "--dialect-derivation-protocol",
            renderer.DERIVATION_PROTOCOL,
            "--pdf-id",
            "clean",
            "--source-fd",
            "3",
            "--pdf-output",
            "stdout",
        ],
        [
            "--dialect-derivation-protocol",
            renderer.DERIVATION_PROTOCOL,
            "--pdf-id",
            renderer.DERIVATION_ROLE,
            "--source-fd",
            "03",
            "--pdf-output",
            "stdout",
        ],
        [
            "--dialect-derivation-protocol",
            renderer.DERIVATION_PROTOCOL,
            "--pdf-id",
            renderer.DERIVATION_ROLE,
            "--source-fd",
            "3",
            "--pdf-output",
            "file",
        ],
    ],
)
def test_derive_cli_rejects_every_noncanonical_argument_shape(
    arguments: list[str],
) -> None:
    with pytest.raises(renderer.RebuttalRenderError, match="argument protocol"):
        renderer._derivation_cli(arguments)


def test_builder_source_has_no_shell_network_or_pdf_embedding_path() -> None:
    builder_source = Path(bundle_builder.__file__).read_text(encoding="utf-8")
    renderer_source = Path(renderer.__file__).read_text(encoding="utf-8")
    assert "import socket" not in builder_source
    assert "import subprocess" not in builder_source
    assert "shell=True" not in builder_source
    assert "requests." not in builder_source
    assert "urllib." not in builder_source
    assert '"base64": base64.b64encode(production.pdf' not in renderer_source
    assert "derive_rebuttal_pdf_from_bundle_fd" in renderer_source


@REQUIRES_NATIVE_DARWIN
def test_bundle_publication_is_no_replace_and_preserves_failed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing.bundle"
    existing.write_bytes(b"keep-me")
    with pytest.raises(
        bundle_builder.RebuttalDerivationBundleError, match="already exists"
    ):
        bundle_builder._publish_bundle(existing, b"replacement")
    assert existing.read_bytes() == b"keep-me"

    destination = tmp_path / "new.bundle"
    failure_message = "synthetic rename failure"

    def fail_rename(*_arguments: object) -> None:
        raise renderer.RebuttalRenderError(failure_message)

    monkeypatch.setattr(renderer, "_rename_no_replace", fail_rename)
    with pytest.raises(
        bundle_builder.RebuttalDerivationBundleError,
        match=r"materialized_path=.*publication_state=preserved",
    ):
        bundle_builder._publish_bundle(destination, b"preserve-stage")
    stages = tuple(tmp_path.glob(".dialect-rebuttal-bundle-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"preserve-stage"
    assert stages[0].stat().st_mode & 0o777 == 0o400
    assert not destination.exists()


@REQUIRES_NATIVE_DARWIN
def test_bundle_publication_preserves_primary_and_attempts_every_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_close = os.close
    close_calls: list[int] = []
    rename_message = "injected primary rename failure"

    def fail_rename(*_arguments: object) -> None:
        raise renderer.RebuttalRenderError(rename_message)

    def close_then_fail(descriptor: int) -> None:
        close_calls.append(descriptor)
        original_close(descriptor)
        if len(close_calls) >= 2:
            message = f"injected close failure {len(close_calls)}"
            raise OSError(message)

    monkeypatch.setattr(renderer, "_rename_no_replace", fail_rename)
    monkeypatch.setattr(bundle_builder.os, "close", close_then_fail)
    with pytest.raises(bundle_builder.RebuttalDerivationBundleError) as captured:
        bundle_builder._publish_bundle(tmp_path / "primary.bundle", b"source-bundle")
    message = str(captured.value)
    assert rename_message in message
    assert "readable bundle descriptor cleanup failed" in message
    assert "bundle parent descriptor cleanup failed" in message
    assert "publication_state=preserved-after-failure" in message
    assert len(close_calls) == 3


@REQUIRES_NATIVE_DARWIN
def test_bundle_publication_reports_clean_success_close_failure_after_all_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_close = os.close
    close_calls: list[int] = []

    def fail_readable_close_after_closing(descriptor: int) -> None:
        close_calls.append(descriptor)
        original_close(descriptor)
        if len(close_calls) == 2:
            message = "injected readable close failure"
            raise OSError(message)

    monkeypatch.setattr(bundle_builder.os, "close", fail_readable_close_after_closing)
    destination = tmp_path / "published.bundle"
    with pytest.raises(bundle_builder.RebuttalDerivationBundleError) as captured:
        bundle_builder._publish_bundle(destination, b"source-bundle")
    message = str(captured.value)
    assert "readable bundle descriptor cleanup failed" in message
    assert f"materialized_path={destination}" in message
    assert "publication_state=preserved-after-failure" in message
    assert len(close_calls) == 3
    assert destination.read_bytes() == b"source-bundle"
    assert destination.stat().st_mode & 0o777 == 0o400


@REQUIRES_NATIVE_DARWIN
def test_direct_and_bundle_derivation_match_and_encoded_pdf_text_is_inert(
    tmp_path: Path,
) -> None:
    tools, reportlab_root = _real_dependencies()
    source, template, config = _canonical_inputs(tmp_path)
    dependency = renderer.reportlab_dependency_digest(reportlab_root)
    renderer_path = Path(renderer.__file__).resolve(strict=True)
    machine_runner = renderer._machine_runner_path()
    build_kwargs = {
        "regular_font": REGULAR_FONT,
        "bold_font": BOLD_FONT,
        "reportlab_root": reportlab_root,
        "release_id": str(CONFIG["release_id"]),
        "tool_paths": tools,
        "expected_source_sha256": _sha(source),
        "expected_template_sha256": _sha(template),
        "expected_config_sha256": _sha(config),
        "expected_regular_font_sha256": _sha(REGULAR_FONT),
        "expected_bold_font_sha256": _sha(BOLD_FONT),
        "expected_reportlab_tree_sha256": str(dependency["tree_sha256"]),
        "expected_machine_runner_sha256": _sha(machine_runner),
        "expected_builder_sha256": _sha(renderer_path),
        "expected_tool_sha256": {name: _sha(path) for name, path in tools.items()},
    }
    direct_root = tmp_path / "direct-render"
    direct = renderer.build_rebuttal_pdf(
        source,
        template,
        config,
        direct_root,
        **build_kwargs,
    )
    direct_pdf = Path(direct.pdf_path).read_bytes()
    destination = tmp_path / "rebuttal-source.bundle"
    receipt = bundle_builder.build_rebuttal_derivation_bundle(
        source,
        template,
        config,
        direct_root,
        destination,
        regular_font=REGULAR_FONT,
        bold_font=BOLD_FONT,
        reportlab_root=reportlab_root,
        runtime=tools["python"],
        release_id=str(CONFIG["release_id"]),
        tool_paths=tools,
        expected_source_sha256=_sha(source),
        expected_template_sha256=_sha(template),
        expected_config_sha256=_sha(config),
        expected_regular_font_sha256=_sha(REGULAR_FONT),
        expected_bold_font_sha256=_sha(BOLD_FONT),
        expected_reportlab_tree_sha256=str(dependency["tree_sha256"]),
        expected_machine_runner_sha256=_sha(machine_runner),
        expected_renderer_sha256=_sha(renderer_path),
        expected_tool_sha256={name: _sha(path) for name, path in tools.items()},
        expected_manifest_sha256=direct.manifest_sha256,
        expected_pdf_sha256=direct.pdf_sha256,
        expected_pdf_bytes=len(direct_pdf),
    )
    bundle_raw = destination.read_bytes()
    bundle_value = json.loads(bundle_raw)
    assert receipt.contains_pre_rendered_pdf_member is False
    assert receipt.promotable is False
    assert bundle_raw == renderer._canonical_json(bundle_value)
    assert renderer.PDF_SIGNATURE not in bundle_raw
    assert base64.b64encode(direct_pdf) not in bundle_raw
    assert all(
        fragment not in bundle_raw
        for fragment in bundle_builder.FORBIDDEN_OUTER_BUNDLE_PATH_FRAGMENTS
    )
    assert set(bundle_value["expected_output"]["pdf"]) == {
        "member",
        "bytes",
        "sha256",
    }
    assert bundle_value["expected_output"]["pdf"]["sha256"] == direct.pdf_sha256
    assert (
        bundle_value["expected_output"]["renderer_manifest"]["sha256"]
        == direct.manifest_sha256
    )
    assert bundle_value["dependencies"]["renderer"]["sha256"] == _sha(renderer_path)
    assert bundle_value["dependencies"]["machine_runner"]["sha256"] == _sha(
        machine_runner
    )
    assert bundle_value["dependencies"]["runtime"]["sha256"] == _sha(tools["python"])
    assert (
        bundle_value["dependencies"]["reportlab"]["tree_sha256"]
        == dependency["tree_sha256"]
    )
    decoded_inputs = {
        item["member"]: base64.b64decode(item["base64"], validate=True)
        for item in bundle_value["canonical_inputs"]
    }
    assert set(decoded_inputs) == {
        renderer.SOURCE_MEMBER,
        renderer.TEMPLATE_MEMBER,
        renderer.CONFIG_MEMBER,
    }
    assert base64.b64decode(PDF_LIKE_TEXT, validate=True).startswith(
        renderer.PDF_SIGNATURE
    )
    assert PDF_LIKE_TEXT.encode("ascii") in decoded_inputs[renderer.SOURCE_MEMBER]
    visible_text = subprocess.run(  # noqa: S603 - exact pinned PDF text tool.
        [str(tools["pdftotext"]), str(direct.pdf_path), "-"],
        cwd="/",
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    assert PDF_LIKE_TEXT.encode("ascii") in visible_text

    completed = _invoke_derive(tools["python"], renderer_path, destination)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stderr == b""
    assert completed.stdout == direct_pdf
    assert hashlib.sha256(completed.stdout).hexdigest() == direct.pdf_sha256

    for index, (section, message) in enumerate(
        (
            ("renderer_manifest", b"fresh renderer manifest differs"),
            ("pdf", b"fresh renderer PDF differs"),
        )
    ):
        drifted = json.loads(bundle_raw)
        drifted["expected_output"][section]["sha256"] = "0" * 64
        drifted_raw = renderer._canonical_json(
            renderer._normalize_derivation_bundle(drifted)
        )
        drifted_path = tmp_path / f"drifted-{index}.bundle"
        drifted_path.write_bytes(drifted_raw)
        drifted_path.chmod(0o400)
        rejected = _invoke_derive(tools["python"], renderer_path, drifted_path)
        assert rejected.returncode != 0
        assert rejected.stdout == b""
        assert message in rejected.stderr

    for index, role in enumerate(("regular", "bold")):
        drifted = json.loads(bundle_raw)
        assert drifted["dependencies"]["fonts"][index]["role"] == role
        drifted["dependencies"]["fonts"][index]["postscript_name"] = (
            f"SyntheticWrong{role.title()}"
        )
        drifted_raw = renderer._canonical_json(
            renderer._normalize_derivation_bundle(drifted)
        )
        drifted_path = tmp_path / f"drifted-font-{role}.bundle"
        drifted_path.write_bytes(drifted_raw)
        drifted_path.chmod(0o400)
        rejected = _invoke_derive(tools["python"], renderer_path, drifted_path)
        assert rejected.returncode != 0
        assert rejected.stdout == b""
        assert b"font PostScript names differ" in rejected.stderr


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize(
    ("target", "private_path"),
    [
        ("source", "/Users/example/private.txt"),
        ("template", "/private/example/template.json"),
        ("config", "/System/example/config.json"),
    ],
)
def test_bundle_builder_rejects_private_path_in_each_decoded_input(
    tmp_path: Path,
    target: str,
    private_path: str,
) -> None:
    tools, reportlab_root = _real_dependencies()
    source, template, config = _canonical_inputs(tmp_path)
    if target == "source":
        source.write_bytes(
            renderer._canonicalize_markdown(
                SOURCE.replace(
                    "synthetic response",
                    f"synthetic response at {private_path}",
                ).encode("utf-8")
            )
        )
    elif target == "template":
        template.write_bytes(
            renderer._canonical_json({**TEMPLATE, "schema": private_path})
        )
    else:
        config.write_bytes(
            renderer._canonical_json({**CONFIG, "manuscript_title": private_path})
        )
    renderer_path = Path(renderer.__file__).resolve(strict=True)
    machine_runner = renderer._machine_runner_path()
    with pytest.raises(
        bundle_builder.RebuttalDerivationBundleError,
        match=r"decoded canonical input .* contains a private absolute",
    ):
        bundle_builder.build_rebuttal_derivation_bundle(
            source,
            template,
            config,
            tmp_path / "unused-published-root",
            tmp_path / "must-not-exist-private.bundle",
            regular_font=REGULAR_FONT,
            bold_font=BOLD_FONT,
            reportlab_root=reportlab_root,
            runtime=tools["python"],
            release_id=str(CONFIG["release_id"]),
            tool_paths=tools,
            expected_source_sha256=_sha(source),
            expected_template_sha256=_sha(template),
            expected_config_sha256=_sha(config),
            expected_regular_font_sha256=_sha(REGULAR_FONT),
            expected_bold_font_sha256=_sha(BOLD_FONT),
            expected_reportlab_tree_sha256="0" * 64,
            expected_machine_runner_sha256=_sha(machine_runner),
            expected_renderer_sha256=_sha(renderer_path),
            expected_tool_sha256={name: _sha(path) for name, path in tools.items()},
            expected_manifest_sha256="1" * 64,
            expected_pdf_sha256="2" * 64,
            expected_pdf_bytes=100,
        )
    assert not (tmp_path / "must-not-exist-private.bundle").exists()


@REQUIRES_NATIVE_DARWIN
def test_bundle_builder_rejects_noncanonical_input_before_publication(
    tmp_path: Path,
) -> None:
    tools, reportlab_root = _real_dependencies()
    source, template, config = _canonical_inputs(tmp_path)
    template.write_text(json.dumps(TEMPLATE, indent=2) + "\n", encoding="utf-8")
    renderer_path = Path(renderer.__file__).resolve(strict=True)
    machine_runner = renderer._machine_runner_path()
    with pytest.raises(bundle_builder.RebuttalDerivationBundleError, match="canonical"):
        bundle_builder.build_rebuttal_derivation_bundle(
            source,
            template,
            config,
            tmp_path / "unused-published-root",
            tmp_path / "must-not-exist.bundle",
            regular_font=REGULAR_FONT,
            bold_font=BOLD_FONT,
            reportlab_root=reportlab_root,
            runtime=tools["python"],
            release_id=str(CONFIG["release_id"]),
            tool_paths=tools,
            expected_source_sha256=_sha(source),
            expected_template_sha256=_sha(template),
            expected_config_sha256=_sha(config),
            expected_regular_font_sha256=_sha(REGULAR_FONT),
            expected_bold_font_sha256=_sha(BOLD_FONT),
            expected_reportlab_tree_sha256=str(
                renderer.reportlab_dependency_digest(reportlab_root)["tree_sha256"]
            ),
            expected_machine_runner_sha256=_sha(machine_runner),
            expected_renderer_sha256=_sha(renderer_path),
            expected_tool_sha256={name: _sha(path) for name, path in tools.items()},
            expected_manifest_sha256="0" * 64,
            expected_pdf_sha256="1" * 64,
            expected_pdf_bytes=100,
        )
    assert not (tmp_path / "must-not-exist.bundle").exists()

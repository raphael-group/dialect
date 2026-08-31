"""Adversarial tests for the native rendered-document machine closure."""

# The suite intentionally exercises private fail-closed primitives and raw POSIX
# name swaps that pathlib cannot express with directory descriptors.
# ruff: noqa: C901, EM101, PLR0913, PLR0915, PTH104, PTH105, PTH211, S603, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from matplotlib import rc_context
from matplotlib.figure import Figure

from analysis import build_tcga_revision_rendered_document_machine_closure as closure

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


RELEASE_ID = "synthetic-machine-closure-v1"


@pytest.fixture(autouse=True)
def _restore_synthetic_tree_modes(tmp_path: Path) -> Iterator[None]:
    """Let pytest remove only its synthetic trees after seal-policy tests."""
    yield
    for directory, child_directories, _files in os.walk(tmp_path):
        Path(directory).chmod(0o700)
        for child in child_directories:
            child_path = Path(directory) / child
            if not child_path.is_symlink():
                child_path.chmod(0o700)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(64 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _real_tool_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in closure.TOOL_ORDER:
        located = shutil.which(name)
        if located is None:
            pytest.skip(f"local Poppler tool is unavailable: {name}")
        paths[name] = Path(located).resolve()
    return paths


def _tool_sha256(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _file_sha256(paths[name]) for name in closure.TOOL_ORDER}


def _synthetic_tool_paths() -> dict[str, Path]:
    return {name: Path(f"/synthetic/{name}") for name in closure.TOOL_ORDER}


def _synthetic_tool_sha256() -> dict[str, str]:
    return {name: _sha256(name.encode("ascii")) for name in closure.TOOL_ORDER}


def _pdf_sha256(root: Path) -> dict[str, str]:
    return {pdf_id: _file_sha256(root / member) for pdf_id, member in closure.PDF_ORDER}


def _synthetic_pdf_sha256() -> dict[str, str]:
    return {pdf_id: _sha256(pdf_id.encode("ascii")) for pdf_id in closure.PDF_IDS}


def _make_minimal_pdf_root(tmp_path: Path) -> Path:
    root = tmp_path / "pdf-root"
    root.mkdir()
    for pdf_id, member in closure.PDF_ORDER:
        (root / member).write_bytes(
            f"%PDF-1.7\nsynthetic {pdf_id}\n%%EOF\n".encode("ascii"),
        )
    return root


def _make_real_pdf_root(tmp_path: Path) -> Path:
    _real_tool_paths()
    root = tmp_path / "pdf-root"
    root.mkdir()
    with rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        for pdf_id, member in closure.PDF_ORDER:
            figure = Figure(figsize=(8.5, 11))
            figure.text(
                0.5,
                0.5,
                f"synthetic machine closure {pdf_id}",
                ha="center",
                va="center",
            )
            figure.savefig(root / member, format="pdf")
    return root


def _read_manifest(root: Path) -> dict[str, object]:
    return json.loads((root / closure.MANIFEST_MEMBER).read_text(encoding="ascii"))


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o400)


def _fake_production(
    stage: closure._PinnedRoot,
    *,
    output_guard: Callable[[], None],
) -> closure._Production:
    producer_raw = b"{}"
    closure._write_member(
        stage,
        closure.PRODUCER_MEMBER,
        producer_raw,
        guard=output_guard,
    )
    inventory = [
        {
            "member": closure.PRODUCER_MEMBER,
            "bytes": len(producer_raw),
            "sha256": _sha256(producer_raw),
        },
    ]
    unsigned: dict[str, object] = {
        "schema": closure.MACHINE_CLOSURE_SCHEMA,
        "contract": closure.MACHINE_CLOSURE_CONTRACT,
        "release_id": RELEASE_ID,
        "member_inventory": inventory,
        "summary": {
            "pdf_count": 4,
            "page_count": 4,
            "machine_pass_count": 4,
        },
        "pdf_set_sha256": "1" * 64,
        "tool_set_sha256": "2" * 64,
        "render_set_sha256": "3" * 64,
    }
    manifest = {
        **unsigned,
        "payload_sha256": _sha256(closure._canonical_json(unsigned)),
    }
    manifest_raw = closure._canonical_json(manifest)
    closure._write_member(
        stage,
        closure.MANIFEST_MEMBER,
        manifest_raw,
        guard=output_guard,
    )
    os.fchmod(stage.descriptor, 0o500)
    os.fsync(stage.descriptor)
    return closure._Production(
        manifest=manifest,
        manifest_raw=manifest_raw,
        member_inventory=inventory,
        page_count=4,
    )


def _make_fake_sealed_closure(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "sealed-closure"
    root.mkdir(mode=0o700)
    root_pin = closure._pin_root(root, context="synthetic sealed closure")
    try:
        production = _fake_production(root_pin, output_guard=lambda: None)
    finally:
        root_pin.close()
    return root, production.manifest_raw


def _install_fake_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    def produce(
        _pdf_root: Path,
        stage: closure._PinnedRoot,
        *,
        release_id: str,
        tool_paths: dict[str, Path],
        expected_tool_sha256: dict[str, str],
        expected_pdf_sha256: dict[str, str],
        output_guard: Callable[[], None],
    ) -> closure._Production:
        assert release_id == RELEASE_ID
        assert tool_paths == _synthetic_tool_paths()
        assert expected_tool_sha256 == _synthetic_tool_sha256()
        assert expected_pdf_sha256 == _synthetic_pdf_sha256()
        return _fake_production(stage, output_guard=output_guard)

    monkeypatch.setattr(
        closure,
        "_produce_into",
        produce,
    )
    monkeypatch.setattr(
        closure,
        "_validate_tree_against_manifest",
        lambda _root, _raw, **_kwargs: [],
    )


def _summary() -> bytes:
    return b"Pages:           1\nEncrypted:       no\nPDF version:     1.7\n"


def _page_geometry(*, width: str = "612", crop_right: str = "612") -> bytes:
    return (
        f"Page    1 size: {width} x 792 pts (letter)\n"
        "Page    1 rot: 0\n"
        "Page    1 MediaBox: 0.00 0.00 612.00 792.00\n"
        f"Page    1 CropBox: 0.00 0.00 {crop_right}.00 792.00\n"
    ).encode("ascii")


def _fonts() -> bytes:
    return (
        b"name type encoding emb sub uni object ID\n"
        b"------------------------------------------\n"
        b"ABCDEF+DejaVuSans CID TrueType Identity-H yes yes yes 15 0\n"
    )


def _images() -> bytes:
    return (
        b"page num type width height color comp bpc enc interp object ID "
        b"x-ppi y-ppi size ratio\n"
    )


def _stub_native_invocations(
    monkeypatch: pytest.MonkeyPatch,
    *,
    geometry: bytes | None = None,
    images: bytes | None = None,
) -> None:
    def invoke(
        **kwargs: object,
    ) -> tuple[bytes, bytes, dict[str, object], list[dict[str, object]]]:
        name = str(kwargs["name"])
        if name.startswith("pdfinfo-summary"):
            stdout = _summary()
        elif name.startswith("pdfinfo-pages"):
            stdout = geometry if geometry is not None else _page_geometry()
        elif name.startswith("pdffonts"):
            stdout = _fonts()
        elif name.startswith("pdfimages"):
            stdout = images if images is not None else _images()
        else:  # pragma: no cover - test setup invariant
            raise AssertionError(name)
        return stdout, b"", {"name": name}, []

    monkeypatch.setattr(closure, "_invoke_and_record", invoke)


def _derive_with_render_stub(
    monkeypatch: pytest.MonkeyPatch,
    render: Callable[
        ...,
        tuple[dict[str, object], dict[str, object], list[dict[str, object]]],
    ],
) -> None:
    _stub_native_invocations(monkeypatch)
    monkeypatch.setattr(closure, "_render_page", render)
    pin = SimpleNamespace(
        path=Path("/synthetic/document.pdf"),
        descriptor=-1,
        size=10,
        sha256="a" * 64,
    )
    tools = {
        name: SimpleNamespace(path=Path(f"/synthetic/{name}"))
        for name in closure.TOOL_ORDER
    }
    closure._derive_document(
        pdf_id="clean",
        member="manuscript-clean.pdf",
        pdf_pin=pin,
        tools=tools,
        output_root=Path("/synthetic/output"),
        budget=closure._ProcessBudget(),
        page_budget=closure._PageBudget(),
        revalidate=lambda: None,
    )


def test_real_poppler_four_one_page_synthetic_canary_builds_and_replays(
    tmp_path: Path,
) -> None:
    pdf_root = _make_real_pdf_root(tmp_path)
    tools = _real_tool_paths()
    tool_sha256 = _tool_sha256(tools)
    pdf_sha256 = _pdf_sha256(pdf_root)
    destination = tmp_path / "machine-closure"
    receipt = closure.build_machine_closure(
        pdf_root,
        destination,
        release_id=RELEASE_ID,
        tool_paths=tools,
        expected_tool_sha256=tool_sha256,
        expected_pdf_sha256=pdf_sha256,
    )
    assert receipt.pdf_count == 4
    assert receipt.page_count == 4
    assert receipt.machine_pass_count == 4
    assert receipt.replay_root is None
    assert (destination.stat().st_mode & 0o777) == 0o500

    manifest = _read_manifest(destination)
    assert [record["pdf_id"] for record in manifest["pdf_order"]] == list(
        closure.PDF_IDS,
    )
    producer = json.loads(
        (destination / closure.PRODUCER_MEMBER).read_text(encoding="ascii"),
    )
    assert producer["execution_contract"]["shell"] is False
    assert producer["execution_contract"]["cwd"] == "/"
    assert producer["execution_contract"]["environment"] == closure.EXACT_ENVIRONMENT
    assert producer["execution_contract"]["execution_binding_scope"] == (
        "main_executable"
    )
    assert producer["execution_contract"]["non_system_dylib_closure"] == (
        "not_attested"
    )
    assert producer["execution_contract"]["pdf_inputs"] == "descriptor_bound"
    assert producer["execution_contract"]["tool_trust_anchor"] == (
        "caller-supplied-exact-sha256-per-tool"
    )
    assert producer["execution_contract"]["expected_tool_sha256"] == tool_sha256
    assert producer["execution_contract"]["pdf_authority_anchor"] == (
        "caller-supplied-exact-sha256-per-fixed-pdf-role"
    )
    assert producer["execution_contract"]["expected_pdf_sha256"] == pdf_sha256
    assert [record["name"] for record in producer["tools"]] == list(
        closure.TOOL_ORDER,
    )
    assert all(
        Path(record["absolute_path"]).is_absolute() for record in producer["tools"]
    )
    assert all(
        record["native_code_directory"]["architecture"] == "arm64"
        and len(record["native_code_directory"]["cdhash"]) == 40
        for record in producer["tools"]
    )
    for invocation in producer["invocations"]:
        binding = invocation["executable_binding"]
        assert binding["suspended_wait_status"] == 0x7F
        assert binding["execution_binding_scope"] == "main_executable"
        assert binding["non_system_dylib_closure"] == "not_attested"
        assert (
            binding["observed_cdhash"] == binding["expected_code_directory"]["cdhash"]
        )
        assert binding["code_signing_status"] & closure.REQUIRED_CS_FLAGS == (
            closure.REQUIRED_CS_FLAGS
        )
        assert binding["code_signing_status"] & closure.REJECTED_CS_FLAGS == 0
        assert invocation["descriptor_bound_input_count"] == (
            0 if invocation["name"].endswith("-version") else 1
        )
    for document in producer["documents"]:
        assert all(document["checks"].values())
        assert document["raster_images"] == []
        assert len(document["page_geometry"]) == 1
        page = document["page_geometry"][0]
        assert page["render"]["sha256"] == page["reproduction"]["sha256"]
        assert (page["render"]["width_pixels"], page["render"]["height_pixels"]) == (
            1275,
            1650,
        )
    for directory, _child_directories, files in os.walk(destination):
        assert stat.S_IMODE(Path(directory).stat().st_mode) == 0o500
        for filename in files:
            assert stat.S_IMODE((Path(directory) / filename).stat().st_mode) == 0o400

    rendered = destination / "pages/clean/page-0001.png"
    corrupted_raw = bytearray(rendered.read_bytes())
    idat_offset = corrupted_raw.index(b"IDAT") + 4
    corrupted_raw[idat_offset] ^= 0x01
    corrupted = tmp_path / "corrupted-render.png"
    corrupted.write_bytes(corrupted_raw)
    corrupted_pin = closure._pin_file(
        corrupted,
        maximum=closure.MAX_PNG_BYTES,
        context="corrupted synthetic render",
    )
    try:
        with pytest.raises(ValueError, match=r"CRC|zlib"):
            closure._parse_png_dimensions(
                corrupted_pin,
                context="corrupted synthetic render",
            )
    finally:
        corrupted_pin.close()

    rendered_raw = rendered.read_bytes()
    no_phys_raw = bytearray(rendered_raw[: len(closure.PNG_SIGNATURE)])
    offset = len(closure.PNG_SIGNATURE)
    while offset < len(rendered_raw):
        length = struct.unpack_from(">I", rendered_raw, offset)[0]
        chunk_end = offset + 12 + length
        if rendered_raw[offset + 4 : offset + 8] != b"pHYs":
            no_phys_raw.extend(rendered_raw[offset:chunk_end])
        offset = chunk_end
    no_phys = tmp_path / "missing-phys-render.png"
    no_phys.write_bytes(no_phys_raw)
    no_phys_pin = closure._pin_file(
        no_phys,
        maximum=closure.MAX_PNG_BYTES,
        context="missing-pHYs synthetic render",
    )
    try:
        with pytest.raises(ValueError, match="complete PNG"):
            closure._parse_png_dimensions(
                no_phys_pin,
                context="missing-pHYs synthetic render",
            )
    finally:
        no_phys_pin.close()

    replay_root = tmp_path / "validation-replay"
    validated = closure.validate_machine_closure(
        pdf_root,
        destination,
        replay_root,
        expected_manifest_sha256=receipt.manifest_sha256,
        release_id=RELEASE_ID,
        tool_paths=tools,
        expected_tool_sha256=tool_sha256,
        expected_pdf_sha256=pdf_sha256,
    )
    assert validated.manifest_sha256 == receipt.manifest_sha256
    assert validated.pdf_set_sha256 == receipt.pdf_set_sha256
    assert validated.tool_set_sha256 == receipt.tool_set_sha256
    assert validated.render_set_sha256 == receipt.render_set_sha256
    assert validated.replay_root == str(replay_root)
    assert replay_root.is_dir()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "symlink", "hardlink", "fifo"],
)
def test_exact_pdf_root_rejects_missing_extra_and_special_members(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _make_minimal_pdf_root(tmp_path)
    target = root / closure.PDF_MEMBERS[0]
    if mutation == "missing":
        os.rename(target, tmp_path / "held.pdf")
    elif mutation == "extra":
        (root / "extra.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    elif mutation == "symlink":
        os.rename(target, tmp_path / "held.pdf")
        os.symlink(tmp_path / "held.pdf", target)
    elif mutation == "hardlink":
        os.rename(target, tmp_path / "held.pdf")
        os.link(tmp_path / "held.pdf", target)
    else:
        os.rename(target, tmp_path / "held.pdf")
        os.mkfifo(target)
    with pytest.raises((closure.MachineClosureError, ValueError)):
        closure.build_machine_closure(
            root,
            tmp_path / "destination",
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )


def test_root_pdf_and_tool_name_swaps_are_detected(tmp_path: Path) -> None:
    root = _make_minimal_pdf_root(tmp_path)
    root_pin = closure._pin_root(root, context="synthetic root")
    moved_root = tmp_path / "moved-root"
    try:
        os.rename(root, moved_root)
        root.mkdir()
        with pytest.raises(ValueError, match="changed"):
            closure._revalidate_root(root_pin, context="synthetic root")
    finally:
        root_pin.close()

    for label in ("PDF", "tool"):
        path = tmp_path / f"{label.lower()}-pin"
        path.write_bytes(b"original")
        pin = closure._pin_file(
            path,
            maximum=1024,
            context=label,
        )
        try:
            replacement = tmp_path / f"{label.lower()}-replacement"
            replacement.write_bytes(b"replacement")
            os.replace(replacement, path)
            with pytest.raises(ValueError, match="changed"):
                closure._revalidate_file(pin, context=label)
        finally:
            pin.close()


def test_pdf_and_tool_inputs_require_exact_caller_sha256_anchors(
    tmp_path: Path,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    root_pin = closure._pin_root(pdf_root, context="anchored synthetic PDF root")
    wrong_pdf_anchors = _pdf_sha256(pdf_root)
    wrong_pdf_anchors["clean"] = "0" * 64
    try:
        with pytest.raises(closure.MachineClosureError, match="authority anchor"):
            closure._pin_pdfs(
                root_pin,
                expected_pdf_sha256=wrong_pdf_anchors,
            )
    finally:
        root_pin.close()

    tool_paths: dict[str, Path] = {}
    for name in closure.TOOL_ORDER:
        path = tmp_path / f"anchored-{name}"
        path.write_bytes(f"synthetic executable {name}".encode("ascii"))
        path.chmod(0o500)
        tool_paths[name] = path
    wrong_tool_anchors = _tool_sha256(tool_paths)
    wrong_tool_anchors["pdfinfo"] = "0" * 64
    with pytest.raises(closure.MachineClosureError, match="trust anchor"):
        closure._pin_tools(
            tool_paths,
            expected_tool_sha256=wrong_tool_anchors,
        )

    with pytest.raises(closure.MachineClosureError, match="must name exactly"):
        closure._validate_tool_sha256_anchors({"pdfinfo": "0" * 64})
    with pytest.raises(closure.MachineClosureError, match="must name exactly"):
        closure._validate_pdf_sha256_anchors({"clean": "0" * 64})


def test_code_directory_parser_is_exact_and_fails_closed_on_unsupported_macho(
    tmp_path: Path,
) -> None:
    pdfinfo = _real_tool_paths()["pdfinfo"]
    pin = closure._pin_file(
        pdfinfo,
        maximum=16 * 1024 * 1024,
        context="real Poppler CodeDirectory",
    )
    try:
        record = closure._parse_arm64_code_directory(pin)
    finally:
        pin.close()
    assert record["binary_container"] == "thin-macho64"
    assert record["architecture"] == "arm64"
    assert record["hash_type"] in {
        "sha1",
        "sha256",
        "sha256-truncated",
        "sha384",
    }
    assert len(record["cdhash"]) == 40

    malformed = tmp_path / "misaligned-load-command"
    raw = bytearray(pdfinfo.read_bytes())
    struct.pack_into("<I", raw, 36, 12)
    malformed.write_bytes(raw)
    malformed.chmod(0o500)
    malformed_pin = closure._pin_file(
        malformed,
        maximum=16 * 1024 * 1024,
        context="misaligned Mach-O",
    )
    try:
        with pytest.raises(closure.MachineClosureError, match="load command"):
            closure._parse_arm64_code_directory(malformed_pin)
    finally:
        malformed_pin.close()

    alternate = tmp_path / "alternate-code-directory"
    alternate_raw = bytearray(pdfinfo.read_bytes())
    command_count = struct.unpack_from("<I", alternate_raw, 16)[0]
    command_offset = 32
    signature_offset: int | None = None
    for _ in range(command_count):
        command, command_size = struct.unpack_from(
            "<II",
            alternate_raw,
            command_offset,
        )
        if command == closure.LC_CODE_SIGNATURE:
            signature_offset = struct.unpack_from(
                "<I",
                alternate_raw,
                command_offset + 8,
            )[0]
            break
        command_offset += command_size
    assert signature_offset is not None
    blob_count = struct.unpack_from(">I", alternate_raw, signature_offset + 8)[0]
    primary_slot_offset: int | None = None
    for index in range(blob_count):
        slot_offset = signature_offset + 12 + (index * 8)
        slot, blob_relative = struct.unpack_from(">II", alternate_raw, slot_offset)
        blob_magic = struct.unpack_from(
            ">I",
            alternate_raw,
            signature_offset + blob_relative,
        )[0]
        if slot == closure.CSSLOT_CODEDIRECTORY and blob_magic == (
            closure.CSMAGIC_CODEDIRECTORY
        ):
            primary_slot_offset = slot_offset
            break
    assert primary_slot_offset is not None
    struct.pack_into(">I", alternate_raw, primary_slot_offset, 0x1000)
    alternate.write_bytes(alternate_raw)
    alternate.chmod(0o500)
    alternate_pin = closure._pin_file(
        alternate,
        maximum=16 * 1024 * 1024,
        context="alternate CodeDirectory Mach-O",
    )
    try:
        with pytest.raises(
            closure.MachineClosureError,
            match="alternate Poppler CodeDirectories",
        ):
            closure._parse_arm64_code_directory(alternate_pin)
    finally:
        alternate_pin.close()

    universal_source = Path("/usr/bin/true").resolve()
    universal = tmp_path / "unsupported-universal"
    shutil.copyfile(universal_source, universal)
    universal.chmod(0o500)
    universal_pin = closure._pin_file(
        universal,
        maximum=16 * 1024 * 1024,
        context="unsupported universal Mach-O",
    )
    try:
        with pytest.raises(closure.MachineClosureError, match="thin native arm64"):
            closure._parse_arm64_code_directory(universal_pin)
    finally:
        universal_pin.close()


def test_nonintegral_render_dimensions_use_rotation_aware_ceiling() -> None:
    page = {
        "crop_box_millipoints": [0, 0, 612_001, 792_001],
        "rotation_degrees": 0,
    }
    assert closure._expected_render_dimensions(page) == (1276, 1651)
    page["rotation_degrees"] = 90
    assert closure._expected_render_dimensions(page) == (1651, 1276)


def test_every_bounded_invocation_revalidates_before_and_after() -> None:
    executable = Path(sys.executable).resolve()
    pin = closure._pin_file(
        executable,
        maximum=128 * 1024 * 1024,
        context="synthetic Python executable",
    )
    events: list[str] = []
    try:
        return_code, stdout, stderr, attestation = closure._run_bounded(
            pin,
            ["-c", "print('bounded')"],
            timeout=5,
            stdout_limit=1024,
            stderr_limit=1024,
            budget=closure._ProcessBudget(),
            before=lambda: events.append("before"),
            after=lambda: events.append("after"),
        )
    finally:
        pin.close()
    assert return_code == 0
    assert stdout == b"bounded\n"
    assert stderr == b""
    assert attestation["execution_binding_scope"] == "main_executable"
    assert events == ["before", "before", "after"]


@pytest.mark.parametrize(
    ("raw", "parser", "match"),
    [
        (
            b"Pages: nope\nEncrypted: no\nPDF version: 1.7\n",
            closure._parse_summary,
            "Pages",
        ),
        (
            b"Pages: 1\nEncrypted: maybe\nPDF version: 1.7\n",
            closure._parse_summary,
            "Encrypted",
        ),
        (b"not a font inventory\n", closure._parse_fonts, "font inventory"),
        (b"not an image inventory\n", closure._parse_images, "header"),
    ],
)
def test_malformed_native_outputs_fail_closed(
    raw: bytes,
    parser: Callable[..., object],
    match: str,
) -> None:
    with pytest.raises(closure.MachineClosureError, match=match):
        parser(raw, context="synthetic malformed output")


def test_incomplete_and_inconsistent_page_geometry_fail_closed() -> None:
    incomplete = b"Page 1 size: 612 x 792 pts\nPage 1 rot: 0\n"
    with pytest.raises(closure.MachineClosureError, match="lacks complete"):
        closure._parse_pages(incomplete, page_count=1, context="synthetic pages")
    with pytest.raises(closure.MachineClosureError, match="disagrees"):
        closure._parse_pages(
            _page_geometry(width="611", crop_right="612"),
            page_count=1,
            context="synthetic pages",
        )


def test_stdout_overflow_kills_process_group() -> None:
    executable = Path(sys.executable).resolve()
    pin = closure._pin_file(
        executable,
        maximum=128 * 1024 * 1024,
        context="synthetic Python executable",
    )
    events: list[str] = []
    try:
        with pytest.raises(closure.MachineClosureError, match="stdout exceeds"):
            closure._run_bounded(
                pin,
                ["-c", "import os; os.write(1, b'x' * 4096)"],
                timeout=5,
                stdout_limit=64,
                stderr_limit=64,
                budget=closure._ProcessBudget(),
                before=lambda: events.append("before"),
                after=lambda: events.append("after"),
            )
    finally:
        pin.close()
    assert events == ["before", "before", "after"]


def test_timeout_kills_descendant_process_group(tmp_path: Path) -> None:
    executable = Path(sys.executable).resolve()
    pin = closure._pin_file(
        executable,
        maximum=128 * 1024 * 1024,
        context="synthetic Python executable",
    )
    marker = tmp_path / "descendant-survived"
    descendant = (
        "import pathlib,time;time.sleep(0.35);"
        f"pathlib.Path({str(marker)!r}).write_text('survived')"
    )
    parent = (
        f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{descendant!r}])"
    )
    try:
        with pytest.raises(closure.MachineClosureError, match="timeout"):
            closure._run_bounded(
                pin,
                ["-c", parent],
                timeout=0.1,
                stdout_limit=1024,
                stderr_limit=1024,
                budget=closure._ProcessBudget(),
                before=lambda: None,
                after=lambda: None,
            )
    finally:
        pin.close()
    time.sleep(0.5)
    assert not marker.exists()


def test_transient_tool_path_swap_is_killed_before_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _real_tool_paths()
    tool_path = tmp_path / "pinned-poppler"
    replacement_path = tmp_path / "replacement-poppler"
    shutil.copy2(tools["pdfinfo"], tool_path)
    shutil.copy2(tools["pdffonts"], replacement_path)
    tool_path.chmod(0o555)
    replacement_path.chmod(0o555)
    pin = closure._pin_file(
        tool_path,
        maximum=16 * 1024 * 1024,
        context="transient-swap tool",
    )
    held = tmp_path / "held-pinned-poppler"
    original_spawn = closure._spawn_suspended_darwin

    def swapped_spawn(*args: object, **kwargs: object) -> int:
        os.rename(tool_path, held)
        os.rename(replacement_path, tool_path)
        try:
            return original_spawn(*args, **kwargs)
        finally:
            os.rename(tool_path, replacement_path)
            os.rename(held, tool_path)

    monkeypatch.setattr(closure, "_spawn_suspended_darwin", swapped_spawn)
    monkeypatch.setattr(
        closure,
        "_resume_process",
        lambda _pid: pytest.fail("swapped executable was resumed"),
    )
    try:
        with pytest.raises(
            closure.MachineClosureError,
            match=r"mapping|terminated before suspended attestation",
        ):
            closure._run_bounded(
                pin,
                ["-v"],
                timeout=5,
                stdout_limit=1024,
                stderr_limit=1024,
                budget=closure._ProcessBudget(),
                before=lambda: closure._revalidate_file(
                    pin,
                    context="transient-swap tool",
                ),
                after=lambda: closure._revalidate_file(
                    pin,
                    context="transient-swap tool",
                ),
            )
    finally:
        pin.close()


@pytest.mark.parametrize(
    "failure",
    ["cdhash", "missing-kill", "debugged", "csops-unavailable"],
)
def test_suspended_attestation_rejects_hash_status_and_unavailability(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    pin = SimpleNamespace(device=10, inode=20)
    expected = {"cdhash": "ab" * closure.CS_CDHASH_LEN}
    monkeypatch.setattr(
        closure,
        "_main_executable_mapping",
        lambda _pid, _tool: {
            "device": 10,
            "inode": 20,
            "path": "/synthetic/tool",
        },
    )

    def csops(
        _pid: int,
        operation: int,
        size: int,
        *,
        context: str,
    ) -> bytes:
        del context
        if failure == "csops-unavailable":
            raise closure.MachineClosureError("cannot attest with csops")
        if operation == closure.CS_OPS_CDHASH:
            digest = (
                "cd" * closure.CS_CDHASH_LEN
                if failure == "cdhash"
                else "ab" * closure.CS_CDHASH_LEN
            )
            return bytes.fromhex(digest)
        status = closure.REQUIRED_CS_FLAGS
        if failure == "missing-kill":
            status &= ~closure.CS_KILL
        elif failure == "debugged":
            status |= closure.CS_DEBUGGED
        return status.to_bytes(size, sys.byteorder)

    monkeypatch.setattr(closure, "_csops_bytes", csops)
    with pytest.raises(closure.MachineClosureError):
        closure._attest_suspended_process(
            123,
            pin,
            expected,
            suspended_wait_status=0x7F,
        )


def test_attestation_records_same_vnode_cs_kill_fail_stop_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = SimpleNamespace(device=10, inode=20)
    expected = {"cdhash": "ab" * closure.CS_CDHASH_LEN}
    monkeypatch.setattr(
        closure,
        "_main_executable_mapping",
        lambda _pid, _tool: {
            "device": 10,
            "inode": 20,
            "path": "/synthetic/tool",
        },
    )
    monkeypatch.setattr(
        closure,
        "_csops_bytes",
        lambda _pid, operation, size, **_kwargs: (
            closure.REQUIRED_CS_FLAGS.to_bytes(size, sys.byteorder)
            if operation == closure.CS_OPS_STATUS
            else bytes.fromhex("ab" * closure.CS_CDHASH_LEN)
        ),
    )
    record = closure._attest_suspended_process(
        123,
        pin,
        expected,
        suspended_wait_status=0x7F,
    )
    assert record["execution_binding_scope"] == "main_executable"
    assert record["non_system_dylib_closure"] == "not_attested"
    assert record["same_vnode_mutation_fail_stop_assumption"] == (
        "invalid-signed-code-page-triggers-darwin-cs-kill"
    )
    assert record["other_same_vnode_mutations"] == "not_attested"


def test_noncanonical_or_early_terminal_suspension_is_rejected_without_rereap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(closure.os, "waitpid", lambda _pid, _flags: (123, 0x117F))
    with pytest.raises(closure.MachineClosureError, match="not suspended"):
        closure._wait_for_suspension(123, deadline=time.monotonic() + 1)

    monkeypatch.setattr(closure.os, "waitpid", lambda _pid, _flags: (123, 0))
    status, reaped = closure._wait_for_suspension(
        123,
        deadline=time.monotonic() + 1,
    )
    assert status == 0
    assert reaped is True


def test_resume_kill_and_reap_errors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reap = closure._reap_process
    monkeypatch.setattr(
        closure.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("resume denied")),
    )
    with pytest.raises(closure.MachineClosureError, match="cannot resume"):
        closure._resume_process(123)

    direct_kills: list[tuple[int, int]] = []
    monkeypatch.setattr(
        closure.os,
        "kill",
        lambda pid, sig: direct_kills.append((pid, sig)),
    )
    monkeypatch.setattr(closure, "_reap_process", lambda *_args, **_kwargs: 9)
    with pytest.raises(closure.MachineClosureError, match="group termination failed"):
        closure._kill_and_reap(123)
    assert direct_kills == [(123, closure.signal.SIGKILL)]

    monkeypatch.setattr(closure, "_reap_process", original_reap)
    monkeypatch.setattr(
        closure.os,
        "waitpid",
        lambda *_args: (_ for _ in ()).throw(ChildProcessError("already reaped")),
    )
    with pytest.raises(closure.MachineClosureError, match="cannot reap"):
        closure._reap_process(123, deadline=time.monotonic() + 1)


def test_repeated_attestation_failure_leaks_no_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory is unavailable")
    tool_path = _real_tool_paths()["pdfinfo"]
    pin = closure._pin_file(
        tool_path,
        maximum=16 * 1024 * 1024,
        context="attestation-cleanup tool",
    )
    monkeypatch.setattr(
        closure,
        "_attest_suspended_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            closure.MachineClosureError("synthetic attestation failure"),
        ),
    )
    before = len(list(descriptor_root.iterdir()))
    try:
        for _ in range(5):
            with pytest.raises(
                closure.MachineClosureError,
                match="synthetic attestation failure",
            ):
                closure._run_bounded(
                    pin,
                    ["-v"],
                    timeout=5,
                    stdout_limit=1024,
                    stderr_limit=1024,
                    budget=closure._ProcessBudget(),
                    before=lambda: None,
                    after=lambda: None,
                )
    finally:
        pin.close()
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 2


def test_early_terminal_runner_marks_child_reaped_and_preserves_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = SimpleNamespace(path=Path("/synthetic/tool"))
    cleanup: list[bool] = []
    events: list[str] = []
    monkeypatch.setattr(
        closure,
        "_parse_arm64_code_directory",
        lambda _pin: {"cdhash": "ab" * closure.CS_CDHASH_LEN},
    )
    monkeypatch.setattr(closure, "_spawn_suspended_darwin", lambda *_a, **_k: 123)
    monkeypatch.setattr(closure, "_wait_for_suspension", lambda *_a, **_k: (0, True))
    monkeypatch.setattr(
        closure,
        "_kill_and_reap",
        lambda _pid, *, already_reaped=False: cleanup.append(already_reaped),
    )
    with pytest.raises(closure.MachineClosureError, match="terminated before"):
        closure._run_bounded(
            pin,
            [],
            timeout=1,
            stdout_limit=16,
            stderr_limit=16,
            budget=closure._ProcessBudget(),
            before=lambda: events.append("before"),
            after=lambda: events.append("after"),
        )
    assert cleanup == [True]
    assert events == ["before", "before", "after"]


def test_already_reaped_cleanup_never_signals_a_stale_process_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        closure.os,
        "killpg",
        lambda *_args: pytest.fail("already-reaped process group was signalled"),
    )
    monkeypatch.setattr(
        closure.os,
        "kill",
        lambda *_args: pytest.fail("already-reaped process was signalled"),
    )
    monkeypatch.setattr(
        closure,
        "_reap_process",
        lambda *_args, **_kwargs: pytest.fail(
            "already-reaped process was reaped again",
        ),
    )
    assert closure._kill_and_reap(123, already_reaped=True) is None


def test_runner_marks_child_reaped_before_a_failing_after_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(sys.executable).resolve()
    pin = closure._pin_file(
        executable,
        maximum=128 * 1024 * 1024,
        context="unreaped lifecycle executable",
    )
    events: list[str] = []
    original_reap = closure._reap_process
    cleanup: list[bool] = []

    def reap(process_id: int, *, deadline: float) -> int:
        events.append("reap")
        return original_reap(process_id, deadline=deadline)

    def after() -> None:
        assert events == ["reap"]
        events.append("after")
        raise KeyboardInterrupt("post-run interruption")

    monkeypatch.setattr(closure, "_reap_process", reap)
    monkeypatch.setattr(
        closure,
        "_kill_and_reap",
        lambda _pid, *, already_reaped=False: cleanup.append(already_reaped),
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="post-run interruption"):
            closure._run_bounded(
                pin,
                ["-c", "pass"],
                timeout=5,
                stdout_limit=64,
                stderr_limit=64,
                budget=closure._ProcessBudget(),
                before=lambda: None,
                after=after,
            )
    finally:
        pin.close()
    assert events == ["reap", "after"]
    assert cleanup == [True]


def test_spawn_state_destroy_attempts_every_cleanup_after_baseexception() -> None:
    events: list[str] = []

    def destroy_actions(_pointer: object) -> int:
        events.append("actions")
        raise KeyboardInterrupt("actions interrupted")

    def destroy_attributes(_pointer: object) -> int:
        events.append("attributes")
        return closure.errno.EIO

    library = SimpleNamespace(
        posix_spawn_file_actions_destroy=destroy_actions,
        posix_spawnattr_destroy=destroy_attributes,
    )
    errors = closure._destroy_spawn_state(
        library,
        closure.ctypes.c_void_p(),
        closure.ctypes.c_void_p(),
        actions_ready=True,
        attribute_ready=True,
    )
    assert events == ["actions", "attributes"]
    assert "actions interrupted" in errors[0]
    assert "errno 5" in errors[1]


def test_selector_baseexception_is_reported_after_all_runner_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_instance = SimpleNamespace(close_calls=0)

    def close_selector() -> None:
        selector_instance.close_calls += 1
        if selector_instance.close_calls == 1:
            raise KeyboardInterrupt("selector close interrupted")

    selector_instance.close = close_selector
    monkeypatch.setattr(
        closure.selectors,
        "DefaultSelector",
        lambda: selector_instance,
    )
    monkeypatch.setattr(
        closure,
        "_parse_arm64_code_directory",
        lambda _pin: {"cdhash": "ab" * closure.CS_CDHASH_LEN},
    )
    monkeypatch.setattr(closure, "_spawn_suspended_darwin", lambda *_a, **_k: 123)
    monkeypatch.setattr(closure, "_wait_for_suspension", lambda *_a, **_k: (0, True))
    cleanup: list[bool] = []
    monkeypatch.setattr(
        closure,
        "_kill_and_reap",
        lambda _pid, *, already_reaped=False: cleanup.append(already_reaped),
    )
    events: list[str] = []
    with pytest.raises(
        closure.MachineClosureError,
        match=r"terminated before.*selector close interrupted",
    ):
        closure._run_bounded(
            SimpleNamespace(path=Path("/synthetic/tool")),
            [],
            timeout=1,
            stdout_limit=16,
            stderr_limit=16,
            budget=closure._ProcessBudget(),
            before=lambda: events.append("before"),
            after=lambda: events.append("after"),
        )
    assert selector_instance.close_calls == 2
    assert cleanup == [True]
    assert events == ["before", "before", "after"]


def test_baseexception_cleanup_at_pin_enumeration_spawn_and_output_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory is unavailable")

    tool_files: dict[str, Path] = {}
    for name in closure.TOOL_ORDER:
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o500)
        tool_files[name] = path
    original_pin_file = closure._pin_file
    pinned_tools: list[closure._PinnedFile] = []
    calls = 0

    def interrupt_tool_pin(*args: object, **kwargs: object) -> closure._PinnedFile:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        pin = original_pin_file(*args, **kwargs)
        pinned_tools.append(pin)
        return pin

    monkeypatch.setattr(closure, "_pin_file", interrupt_tool_pin)
    with pytest.raises(KeyboardInterrupt):
        closure._pin_tools(
            tool_files,
            expected_tool_sha256=_tool_sha256(tool_files),
        )
    assert pinned_tools[0].descriptor == -1
    monkeypatch.setattr(closure, "_pin_file", original_pin_file)

    pdf_root = _make_minimal_pdf_root(tmp_path)
    root_pin = closure._pin_root(pdf_root, context="interrupt PDF root")
    original_open_member = closure._open_root_member
    pinned_pdfs: list[closure._PinnedFile] = []
    calls = 0

    def interrupt_pdf_pin(*args: object, **kwargs: object) -> closure._PinnedFile:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SystemExit(7)
        pin = original_open_member(*args, **kwargs)
        pinned_pdfs.append(pin)
        return pin

    monkeypatch.setattr(closure, "_open_root_member", interrupt_pdf_pin)
    try:
        with pytest.raises(SystemExit, match="7"):
            closure._pin_pdfs(
                root_pin,
                expected_pdf_sha256=_pdf_sha256(pdf_root),
            )
        assert pinned_pdfs[0].descriptor == -1
    finally:
        root_pin.close()
    monkeypatch.setattr(closure, "_open_root_member", original_open_member)

    output = tmp_path / "interrupt-output"
    output.mkdir(mode=0o700)
    (output / "first").mkdir(mode=0o700)
    output_pin = closure._pin_root(output, context="interrupt output")
    original_open = closure.os.open
    before = len(list(descriptor_root.iterdir()))

    def interrupt_directory_open(path: object, *args: object, **kwargs: object) -> int:
        if path == "second":
            raise KeyboardInterrupt
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(closure.os, "open", interrupt_directory_open)
    try:
        with pytest.raises(KeyboardInterrupt):
            closure._open_output_directory(
                output_pin,
                ("first", "second"),
                create=False,
                guard=lambda: None,
            )
    finally:
        monkeypatch.setattr(closure.os, "open", original_open)
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 2

    spawn_events: list[str] = []
    monkeypatch.setattr(
        closure,
        "_parse_arm64_code_directory",
        lambda _pin: {"cdhash": "ab" * closure.CS_CDHASH_LEN},
    )
    monkeypatch.setattr(
        closure,
        "_spawn_suspended_darwin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    before = len(list(descriptor_root.iterdir()))
    with pytest.raises(KeyboardInterrupt):
        closure._run_bounded(
            SimpleNamespace(path=Path("/synthetic/tool")),
            [],
            timeout=1,
            stdout_limit=16,
            stderr_limit=16,
            budget=closure._ProcessBudget(),
            before=lambda: spawn_events.append("before"),
            after=lambda: spawn_events.append("after"),
        )
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 2
    assert spawn_events == ["before", "before", "after"]

    original_write = closure.os.write
    monkeypatch.setattr(
        closure.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    before = len(list(descriptor_root.iterdir()))
    try:
        with pytest.raises(KeyboardInterrupt):
            closure._write_member(
                output_pin,
                "partial.bin",
                b"payload",
                guard=lambda: None,
            )
    finally:
        monkeypatch.setattr(closure.os, "write", original_write)
        output_pin.close()
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 1


def test_render_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def render(
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        nonlocal calls
        calls += 1
        digest = "a" * 64 if calls == 1 else "b" * 64
        artifact = {
            "member": f"pages/clean/{calls}.png",
            "bytes": 100,
            "sha256": digest,
            "width_pixels": 1275,
            "height_pixels": 1650,
            "pixels_per_meter": closure.EXPECTED_PNG_PIXELS_PER_METER,
        }
        return artifact, {"name": str(kwargs["run"])}, []

    with pytest.raises(closure.MachineClosureError, match="not byte-identical"):
        _derive_with_render_stub(monkeypatch, render)


def test_render_geometry_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        artifact = {
            "member": f"pages/clean/{kwargs['run']}.png",
            "bytes": 100,
            "sha256": "a" * 64,
            "width_pixels": 1200,
            "height_pixels": 1650,
            "pixels_per_meter": closure.EXPECTED_PNG_PIXELS_PER_METER,
        }
        return artifact, {"name": str(kwargs["run"])}, []

    with pytest.raises(closure.MachineClosureError, match="geometry drifted"):
        _derive_with_render_stub(monkeypatch, render)


def test_raster_images_are_retained_for_s1_but_rejected_for_manuscripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_inventory = _images() + (b"1 0 image 10 10 rgb 3 8 jpeg no 5 0 72 72 100B\n")
    _stub_native_invocations(monkeypatch, images=image_inventory)

    def render(
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        artifact = {
            "member": f"pages/{kwargs['pdf_id']}/{kwargs['run']}.png",
            "bytes": 100,
            "sha256": "a" * 64,
            "width_pixels": 1275,
            "height_pixels": 1650,
            "pixels_per_meter": closure.EXPECTED_PNG_PIXELS_PER_METER,
        }
        return artifact, {"name": str(kwargs["run"])}, []

    monkeypatch.setattr(closure, "_render_page", render)
    pdf_path = tmp_path / "synthetic.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    pdf_pin = closure._pin_file(
        pdf_path,
        maximum=1024,
        context="synthetic raster-policy PDF",
    )
    tools = {
        name: SimpleNamespace(path=Path(f"/synthetic/{name}"))
        for name in closure.TOOL_ORDER
    }
    try:
        document, _, _ = closure._derive_document(
            pdf_id="s1",
            member="s1-appendix.pdf",
            pdf_pin=pdf_pin,
            tools=tools,
            output_root=SimpleNamespace(),
            budget=closure._ProcessBudget(),
            page_budget=closure._PageBudget(),
            revalidate=lambda: None,
        )
        assert document["raster_image_count"] == 1
        assert len(document["raster_images"]) == 1
        assert document["checks"]["zero_raster_images"] is False
        assert document["checks"]["raster_policy_pass"] is True

        with pytest.raises(closure.MachineClosureError, match="zero raster images"):
            closure._derive_document(
                pdf_id="clean",
                member="manuscript-clean.pdf",
                pdf_pin=pdf_pin,
                tools=tools,
                output_root=SimpleNamespace(),
                budget=closure._ProcessBudget(),
                page_budget=closure._PageBudget(),
                revalidate=lambda: None,
            )
    finally:
        pdf_pin.close()


@pytest.mark.parametrize(
    "mutation",
    ["symlink", "hardlink", "fifo", "extra", "missing"],
)
def test_closure_validation_rejects_special_extra_and_missing_members(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "closure"
    root.mkdir(mode=0o700)
    producer = b"{}"
    _write_private(root / closure.PRODUCER_MEMBER, producer)
    inventory = [
        {
            "member": closure.PRODUCER_MEMBER,
            "bytes": len(producer),
            "sha256": _sha256(producer),
        },
    ]
    unsigned = {
        "schema": closure.MACHINE_CLOSURE_SCHEMA,
        "contract": closure.MACHINE_CLOSURE_CONTRACT,
        "member_inventory": inventory,
    }
    manifest = {
        **unsigned,
        "payload_sha256": _sha256(closure._canonical_json(unsigned)),
    }
    manifest_raw = closure._canonical_json(manifest)
    _write_private(root / closure.MANIFEST_MEMBER, manifest_raw)
    target = root / closure.PRODUCER_MEMBER
    if mutation == "symlink":
        os.rename(target, tmp_path / "held")
        os.symlink(tmp_path / "held", target)
    elif mutation == "hardlink":
        os.link(target, root / "alias")
        (root / "alias").chmod(0o400)
    elif mutation == "fifo":
        os.mkfifo(root / "fifo")
    elif mutation == "extra":
        _write_private(root / "extra", b"extra")
    else:
        os.rename(target, tmp_path / "held")
    root.chmod(0o500)
    with pytest.raises((closure.MachineClosureError, ValueError)):
        closure._read_anchored_manifest(
            root,
            expected_manifest_sha256=_sha256(manifest_raw),
        )


def test_tree_validation_binds_the_actual_on_disk_manifest_bytes(
    tmp_path: Path,
) -> None:
    root, manifest_raw = _make_fake_sealed_closure(tmp_path)
    mutated = bytearray(manifest_raw)
    mutation_offset = mutated.index(RELEASE_ID.encode("ascii"))
    mutated[mutation_offset] = ord("x")
    manifest_path = root / closure.MANIFEST_MEMBER
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(mutated)
    manifest_path.chmod(0o400)
    root_pin = closure._pin_root(root, context="mutated sealed closure")
    try:
        with pytest.raises(
            closure.MachineClosureError,
            match="on-disk machine manifest",
        ):
            closure._validate_tree_against_manifest(
                root_pin,
                manifest_raw,
                directory_mode=0o500,
            )
    finally:
        root_pin.close()


def test_anchored_manifest_rejects_a_same_size_pread_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_raw = _make_fake_sealed_closure(tmp_path)
    manifest_inode = (root / closure.MANIFEST_MEMBER).stat().st_ino
    original_pread = closure.os.pread
    injected = False

    def raced_pread(descriptor: int, count: int, offset: int) -> bytes:
        nonlocal injected
        raw = original_pread(descriptor, count, offset)
        if (
            not injected
            and os.fstat(descriptor).st_ino == manifest_inode
            and count == len(manifest_raw)
            and offset == 0
        ):
            injected = True
            altered = bytearray(raw)
            altered[0] ^= 1
            return bytes(altered)
        return raw

    monkeypatch.setattr(closure.os, "pread", raced_pread)
    with pytest.raises(closure.MachineClosureError, match="changed while deriving"):
        closure._read_anchored_manifest(
            root,
            expected_manifest_sha256=_sha256(manifest_raw),
        )
    assert injected is True


def test_concurrent_deterministic_candidate_reservation_allows_one_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    destination = tmp_path / "destination"
    entered = Event()
    release = Event()

    def produce(
        _pdf_root: Path,
        stage: closure._PinnedRoot,
        *,
        release_id: str,
        tool_paths: dict[str, Path],
        expected_tool_sha256: dict[str, str],
        expected_pdf_sha256: dict[str, str],
        output_guard: Callable[[], None],
    ) -> closure._Production:
        assert release_id == RELEASE_ID
        assert tool_paths == _synthetic_tool_paths()
        assert expected_tool_sha256 == _synthetic_tool_sha256()
        assert expected_pdf_sha256 == _synthetic_pdf_sha256()
        entered.set()
        assert release.wait(timeout=5)
        return _fake_production(stage, output_guard=output_guard)

    monkeypatch.setattr(closure, "_produce_into", produce)
    monkeypatch.setattr(
        closure,
        "_validate_tree_against_manifest",
        lambda _root, _raw, **_kwargs: [],
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            closure.build_machine_closure,
            pdf_root,
            destination,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
        assert entered.wait(timeout=5)
        with pytest.raises(
            closure.MachineClosureError,
            match="retained private candidate",
        ):
            closure.build_machine_closure(
                pdf_root,
                destination,
                release_id=RELEASE_ID,
                tool_paths=_synthetic_tool_paths(),
                expected_tool_sha256=_synthetic_tool_sha256(),
                expected_pdf_sha256=_synthetic_pdf_sha256(),
            )
        release.set()
        receipt = first.result(timeout=5)
    assert receipt.manifest_path == str(destination / closure.MANIFEST_MEMBER)


@pytest.mark.parametrize("swap", ["parent", "stage"])
def test_parent_and_stage_swaps_during_production_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    container = tmp_path / "container"
    container.mkdir()
    destination = container / "destination"

    def produce(
        _pdf_root: Path,
        stage: closure._PinnedRoot,
        *,
        release_id: str,
        tool_paths: dict[str, Path],
        expected_tool_sha256: dict[str, str],
        expected_pdf_sha256: dict[str, str],
        output_guard: Callable[[], None],
    ) -> closure._Production:
        del release_id, tool_paths, expected_tool_sha256, expected_pdf_sha256
        if swap == "parent":
            held_parent = tmp_path / "held-parent"
            os.rename(container, held_parent)
            container.mkdir()
        else:
            held_stage = container / "held-stage"
            os.rename(stage.path, held_stage)
            stage.path.mkdir(mode=0o700)
        output_guard()
        raise AssertionError("swapped output guard must fail")

    monkeypatch.setattr(closure, "_produce_into", produce)
    with pytest.raises(closure.MachineClosureError, match="changed"):
        closure.build_machine_closure(
            pdf_root,
            destination,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )


def test_partial_production_failure_retains_exact_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    destination = tmp_path / "destination"

    def produce(
        _pdf_root: Path,
        stage: closure._PinnedRoot,
        *,
        release_id: str,
        tool_paths: dict[str, Path],
        expected_tool_sha256: dict[str, str],
        expected_pdf_sha256: dict[str, str],
        output_guard: Callable[[], None],
    ) -> closure._Production:
        del release_id, tool_paths, expected_tool_sha256, expected_pdf_sha256
        closure._write_member(
            stage,
            "partial-evidence.bin",
            b"retained",
            guard=output_guard,
        )
        raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(closure, "_produce_into", produce)
    with pytest.raises(closure.MachineClosureError, match="partial-or-complete"):
        closure.build_machine_closure(
            pdf_root,
            destination,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
    candidate = tmp_path / ".destination.private-candidate"
    assert candidate.is_dir()
    assert (candidate / "partial-evidence.bin").read_bytes() == b"retained"


@pytest.mark.parametrize(
    "failure_point",
    ["parent-fstat", "child-open", "child-fstat", "revalidation"],
)
def test_post_mkdir_reservation_interruptions_close_fds_and_report_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory is unavailable")
    destination = tmp_path / "destination"
    reserved_name = ".destination.private-candidate"
    original_mkdir = closure.os.mkdir
    original_open = closure.os.open
    original_fstat = closure.os.fstat
    original_revalidate = closure._revalidate_reserved_directory
    reservation_made = False
    child_descriptor = -1

    def mkdir(*args: object, **kwargs: object) -> None:
        nonlocal reservation_made
        original_mkdir(*args, **kwargs)
        reservation_made = True

    def open_member(path: object, *args: object, **kwargs: object) -> int:
        nonlocal child_descriptor
        if reservation_made and path == reserved_name and failure_point == "child-open":
            raise KeyboardInterrupt
        descriptor = original_open(path, *args, **kwargs)
        if reservation_made and path == reserved_name:
            child_descriptor = descriptor
        return descriptor

    def fstat(descriptor: int) -> os.stat_result:
        if reservation_made and failure_point == "parent-fstat":
            raise KeyboardInterrupt
        if failure_point == "child-fstat" and descriptor == child_descriptor:
            raise KeyboardInterrupt
        return original_fstat(descriptor)

    def revalidate(*args: object, **kwargs: object) -> None:
        if reservation_made and failure_point == "revalidation":
            raise KeyboardInterrupt
        original_revalidate(*args, **kwargs)

    monkeypatch.setattr(closure.os, "mkdir", mkdir)
    monkeypatch.setattr(closure.os, "open", open_member)
    monkeypatch.setattr(closure.os, "fstat", fstat)
    monkeypatch.setattr(closure, "_revalidate_reserved_directory", revalidate)
    before = len(list(descriptor_root.iterdir()))
    with pytest.raises(
        closure.MachineClosureError,
        match="reserved-private-candidate-do-not-auto-delete",
    ):
        closure._reserve_directory(
            destination,
            reserved_name=reserved_name,
            context="synthetic reservation",
        )
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 2
    assert (tmp_path / reserved_name).is_dir()


def test_validation_closure_repin_failure_closes_replay_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory is unavailable")
    pdf_root = _make_minimal_pdf_root(tmp_path)
    closure_root = tmp_path / "closure"
    _install_fake_producer(monkeypatch)
    receipt = closure.build_machine_closure(
        pdf_root,
        closure_root,
        release_id=RELEASE_ID,
        tool_paths=_synthetic_tool_paths(),
        expected_tool_sha256=_synthetic_tool_sha256(),
        expected_pdf_sha256=_synthetic_pdf_sha256(),
    )
    original_pin_root = closure._pin_root
    closure_pin_calls = 0

    def interrupt_second_closure_pin(
        path: Path,
        *,
        context: str,
    ) -> closure._PinnedRoot:
        nonlocal closure_pin_calls
        if path.absolute() == closure_root.absolute():
            closure_pin_calls += 1
            if closure_pin_calls == 2:
                raise KeyboardInterrupt
        return original_pin_root(path, context=context)

    monkeypatch.setattr(closure, "_pin_root", interrupt_second_closure_pin)
    replay_root = tmp_path / "replay"
    before = len(list(descriptor_root.iterdir()))
    with pytest.raises(closure.MachineClosureError, match="replay_candidate_path="):
        closure.validate_machine_closure(
            pdf_root,
            closure_root,
            replay_root,
            expected_manifest_sha256=receipt.manifest_sha256,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 2
    assert replay_root.is_dir()


def test_published_tree_is_sealed_and_mode_tamper_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    destination = tmp_path / "destination"
    _install_fake_producer(monkeypatch)
    receipt = closure.build_machine_closure(
        pdf_root,
        destination,
        release_id=RELEASE_ID,
        tool_paths=_synthetic_tool_paths(),
        expected_tool_sha256=_synthetic_tool_sha256(),
        expected_pdf_sha256=_synthetic_pdf_sha256(),
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o500
    with pytest.raises(PermissionError):
        (destination / "extra-member").write_bytes(b"forbidden")
    with pytest.raises(PermissionError):
        (destination / closure.PRODUCER_MEMBER).write_bytes(b"forbidden")
    destination.chmod(0o700)
    with pytest.raises(closure.MachineClosureError, match="sealed mode"):
        closure._read_anchored_manifest(
            destination,
            expected_manifest_sha256=receipt.manifest_sha256,
        )


def test_destination_race_preserves_competitor_and_private_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    destination = tmp_path / "destination"
    _install_fake_producer(monkeypatch)
    original = closure._rename_no_replace

    def race(source: str, target: str, parent_fd: int) -> None:
        destination.mkdir(mode=0o700)
        original(source, target, parent_fd)

    monkeypatch.setattr(closure, "_rename_no_replace", race)
    with pytest.raises(closure.MachineClosureError, match="already exists"):
        closure.build_machine_closure(
            pdf_root,
            destination,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
    assert destination.is_dir()
    assert len(list(tmp_path.glob(".destination.private-*"))) == 1


def test_unsupported_atomic_rename_retains_and_reports_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    destination = tmp_path / "destination"
    _install_fake_producer(monkeypatch)
    monkeypatch.setattr(
        closure,
        "_rename_no_replace",
        lambda *_args: (_ for _ in ()).throw(
            closure.MachineClosureError("unsupported no-replace rename"),
        ),
    )
    with pytest.raises(closure.MachineClosureError, match="candidate_paths="):
        closure.build_machine_closure(
            pdf_root,
            destination,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
    assert not destination.exists()
    assert len(list(tmp_path.glob(".destination.private-*"))) == 1


def test_ambiguous_rename_then_raise_reports_owned_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    destination = tmp_path / "destination"
    _install_fake_producer(monkeypatch)

    def rename_then_raise(source: str, target: str, parent_fd: int) -> None:
        os.rename(
            source,
            target,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        raise OSError("synthetic rename acknowledgement loss")

    monkeypatch.setattr(closure, "_rename_no_replace", rename_then_raise)
    with pytest.raises(
        closure.MachineClosureError,
        match="rename-issued-outcome-ambiguous",
    ):
        closure.build_machine_closure(
            pdf_root,
            destination,
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
    assert destination.is_dir()
    assert not list(tmp_path.glob(".destination.private-*"))


def test_retained_stage_blocks_retry_before_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    retained = tmp_path / ".destination.private-candidate"
    retained.mkdir(mode=0o700)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> closure._Production:
        nonlocal called
        called = True
        raise AssertionError("producer must not run")

    monkeypatch.setattr(closure, "_produce_into", forbidden)
    with pytest.raises(closure.MachineClosureError, match="retained private candidate"):
        closure.build_machine_closure(
            pdf_root,
            tmp_path / "destination",
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )
    assert called is False
    assert retained.is_dir()


def test_publication_failure_never_calls_name_based_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_root = _make_minimal_pdf_root(tmp_path)
    _install_fake_producer(monkeypatch)

    def forbidden_unlink(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("publication cleanup must not unlink mutable names")

    monkeypatch.setattr(closure.os, "unlink", forbidden_unlink)
    monkeypatch.setattr(
        closure,
        "_rename_no_replace",
        lambda *_args: (_ for _ in ()).throw(OSError("synthetic publication failure")),
    )
    with pytest.raises(closure.MachineClosureError, match="candidate_paths="):
        closure.build_machine_closure(
            pdf_root,
            tmp_path / "destination",
            release_id=RELEASE_ID,
            tool_paths=_synthetic_tool_paths(),
            expected_tool_sha256=_synthetic_tool_sha256(),
            expected_pdf_sha256=_synthetic_pdf_sha256(),
        )


def test_repeated_preflight_failures_do_not_leak_descriptors(tmp_path: Path) -> None:
    descriptor_root = Path("/dev/fd")
    if not descriptor_root.exists():
        pytest.skip("descriptor inventory is unavailable")
    pdf_root = _make_minimal_pdf_root(tmp_path)
    (pdf_root / "extra.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    before = len(list(descriptor_root.iterdir()))
    for index in range(10):
        with pytest.raises((closure.MachineClosureError, ValueError)):
            closure.build_machine_closure(
                pdf_root,
                tmp_path / f"destination-{index}",
                release_id=RELEASE_ID,
                tool_paths=_synthetic_tool_paths(),
                expected_tool_sha256=_synthetic_tool_sha256(),
                expected_pdf_sha256=_synthetic_pdf_sha256(),
            )
    after = len(list(descriptor_root.iterdir()))
    assert after <= before + 2


def test_process_budget_and_fd_headroom_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = closure._ProcessBudget(count=closure.MAX_PROCESSES)
    with pytest.raises(closure.MachineClosureError, match="process count"):
        budget.consume()
    page_budget = closure._PageBudget(count=closure.MAX_TOTAL_PAGES)
    with pytest.raises(closure.MachineClosureError, match="page count"):
        page_budget.consume(1)
    monkeypatch.setattr(
        closure.resource,
        "getrlimit",
        lambda _resource: (32, 32),
    )
    with pytest.raises(closure.MachineClosureError, match="RLIMIT_NOFILE"):
        closure._validate_fd_headroom(1)


def test_output_file_directory_and_byte_caps_preflight_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "bounded-output"
    root_path.mkdir(mode=0o700)
    root = closure._pin_root(root_path, context="bounded output")
    try:
        monkeypatch.setattr(closure, "MAX_OUTPUT_BYTES", 4)
        closure._write_member(root, "first.bin", b"1234", guard=lambda: None)
        with pytest.raises(closure.MachineClosureError, match="aggregate byte"):
            closure._write_member(root, "too-large.bin", b"5", guard=lambda: None)
        assert not (root_path / "too-large.bin").exists()

        monkeypatch.setattr(closure, "MAX_OUTPUT_BYTES", 1024)
        monkeypatch.setattr(closure, "MAX_OUTPUT_FILES", 1)
        with pytest.raises(closure.MachineClosureError, match="file count"):
            closure._write_member(root, "too-many.bin", b"x", guard=lambda: None)
        assert not (root_path / "too-many.bin").exists()

        monkeypatch.setattr(closure, "MAX_OUTPUT_FILES", 10)
        monkeypatch.setattr(closure, "MAX_OUTPUT_DIRECTORIES", 0)
        with pytest.raises(closure.MachineClosureError, match="directory count"):
            closure._write_member(root, "nested/member.bin", b"x", guard=lambda: None)
        assert not (root_path / "nested").exists()
    finally:
        root.close()


def test_cli_help_paths_are_available() -> None:
    script = Path(closure.__file__)
    for argv in (["--help"], ["build", "--help"], ["validate", "--help"]):
        result = subprocess.run(
            [sys.executable, str(script), *argv],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "machine" in result.stdout.lower()


def test_source_contains_no_unlink_or_shell_execution() -> None:
    source = Path(closure.__file__).read_text(encoding="utf-8")
    assert "os.unlink(" not in source
    assert "shell=True" not in source
    assert "shutil.rmtree" not in source
    assert "subprocess.run(" not in source
    assert "subprocess.call(" not in source

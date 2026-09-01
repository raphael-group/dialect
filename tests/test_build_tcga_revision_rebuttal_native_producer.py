"""Focused adversarial tests for the native rebuttal producer seam."""

# This security boundary intentionally exercises private validators and raw FDs.
# ruff: noqa: COM812, E501, EM101, S603, SLF001, TRY003

from __future__ import annotations

import hashlib
import json
import os
import resource
import socket
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from analysis import build_tcga_revision_rebuttal_native_producer as native
from analysis import build_tcga_revision_rendered_document_machine_closure as machine
from analysis import render_tcga_revision_rebuttal as renderer

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


REQUIRES_NATIVE_DARWIN = pytest.mark.skipif(
    sys.platform != "darwin" or os.uname().machine != "arm64",
    reason="native launcher execution requires arm64 Darwin",
)


@pytest.fixture(autouse=True)
def _restore_synthetic_tree_modes(tmp_path: Path) -> Iterator[None]:
    """Permit pytest to retire only its own synthetic trees after mode tests."""
    yield
    for directory, child_directories, _files in os.walk(tmp_path):
        Path(directory).chmod(0o700)
        for child in child_directories:
            child_path = Path(directory) / child
            if not child_path.is_symlink():
                child_path.chmod(0o700)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _redigest(record: dict[str, object], field: str) -> None:
    body = dict(record)
    body.pop(field, None)
    record[field] = _sha(native._canonical_json(body))


_INTEGRAL_FLOAT = object()


def _replace_nested_json(
    value: object,
    path: Sequence[str | int],
    replacement: object,
) -> None:
    current = value
    for key in path[:-1]:
        current = current[key]
    final = path[-1]
    if replacement is _INTEGRAL_FLOAT:
        current[final] = float(current[final])
    else:
        current[final] = replacement


def _numeric_leaf_mutations(
    value: object,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], object]]:
    if isinstance(value, bool):
        integer = int(value)
        return [(path, integer), (path, float(integer)), (path, None)]
    if isinstance(value, int):
        replacements: list[object] = [float(value), str(value), None, -1, 2**63]
        if value in {0, 1}:
            replacements.append(bool(value))
        return [(path, replacement) for replacement in replacements]
    if isinstance(value, dict):
        return [
            mutation
            for key, child in value.items()
            for mutation in _numeric_leaf_mutations(child, (*path, key))
        ]
    if isinstance(value, list):
        return [
            mutation
            for index, child in enumerate(value)
            for mutation in _numeric_leaf_mutations(child, (*path, index))
        ]
    return []


def _redigest_scalar_mutation(
    authority: dict[str, object],
    path: tuple[str | int, ...],
) -> None:
    if path[:2] == ("source_bundle", "bundle_projection"):
        projection = authority["source_bundle"]["bundle_projection"]
        if path[2:3] == ("canonical_inputs",):
            projection["canonical_inputs_projection_sha256"] = _sha(
                native._canonical_json(projection["canonical_inputs"])
            )
        _redigest(projection, "bundle_projection_sha256")
    elif path[:1] == ("launcher_config",):
        config = authority["launcher_config"]
        _redigest(config, "launcher_config_sha256")
        authority["runtime_handoff"] = native._runtime_handoff(config)
    elif path[:1] == ("runtime_handoff",):
        _redigest(authority["runtime_handoff"], "runtime_handoff_sha256")
    elif path[:1] == ("toolchain",):
        _redigest(authority["toolchain"], "toolchain_projection_sha256")
    elif path[:1] == ("source_release",):
        _redigest(
            authority["source_release"],
            "source_release_projection_sha256",
        )
    elif path[:1] == ("build",):
        _redigest(authority["build"], "build_projection_sha256")
    _rehash_authority(authority)


def _fake_pin(name: str, raw: bytes, *, mode: int = 0o400) -> native._PinnedFile:
    return native._PinnedFile(
        path=Path(f"/synthetic/{name}"),
        descriptor=-1,
        device=1,
        inode=1,
        size=len(raw),
        mtime_ns=1,
        mode=mode,
        uid=os.geteuid(),
        sha256=_sha(raw),
    )


def _anchor(name: str) -> str:
    return _sha(name.encode("ascii"))


def _authority_fixture() -> dict[str, object]:
    anchors = {key: _anchor(key) for key in native.CALLER_ANCHOR_KEYS}
    source_raw = b"int main(void) { return 0; }\n"
    anchors["launcher_source_sha256"] = _sha(source_raw)
    expected_output = {
        "renderer_manifest": {
            "member": renderer.MANIFEST_MEMBER,
            "bytes": 101,
            "sha256": anchors["renderer_manifest_sha256"],
        },
        "pdf": {
            "member": native.PDF_MEMBER,
            "bytes": 202,
            "sha256": anchors["pdf_sha256"],
        },
    }
    canonical_inputs = [
        {
            "member": member,
            "encoding": "base64",
            "bytes": index + 1,
            "sha256": _sha(f"decoded-{member}".encode()),
            "encoded_payload_sha256": _sha(f"encoded-{member}".encode()),
        }
        for index, member in enumerate(
            (renderer.SOURCE_MEMBER, renderer.TEMPLATE_MEMBER, renderer.CONFIG_MEMBER)
        )
    ]
    dependencies = {
        "runtime": {
            "locator": "invoking-python",
            "python_tag": "3.12",
            "bytes": 100,
            "sha256": anchors["runtime_sha256"],
        },
        "renderer": {
            "locator": "current-renderer",
            "member": "analysis/render_tcga_revision_rebuttal.py",
            "bytes": 101,
            "sha256": anchors["renderer_sha256"],
        },
        "machine_runner": {
            "locator": "renderer-sibling-machine-runner",
            "member": native.MACHINE_RUNNER_MEMBER,
            "bytes": 102,
            "sha256": anchors["machine_runner_sha256"],
        },
        "fonts": [
            {
                "role": "regular",
                "locator": "system-arial-unicode",
                "postscript_name": "ArialUnicodeMS",
                "bytes": 103,
                "sha256": _anchor("regular-font"),
            },
            {
                "role": "bold",
                "locator": "system-arial-bold",
                "postscript_name": "Arial-BoldMT",
                "bytes": 104,
                "sha256": _anchor("bold-font"),
            },
        ],
        "reportlab": {
            "locator": "invoking-python-reportlab",
            "file_count": 2,
            "directory_count": 1,
            "entry_count": 3,
            "total_bytes": 105,
            "tree_sha256": _anchor("reportlab-tree"),
            "bundle_bytes": 106,
            "bundle_sha256": _anchor("reportlab-bundle"),
        },
        "tools": [
            {
                "name": name,
                "locator": f"homebrew-{name}",
                "bytes": 107 + index,
                "sha256": _anchor(name),
            }
            for index, name in enumerate(("pdfinfo", "pdffonts", "pdftotext"))
        ],
    }
    projection: dict[str, object] = {
        "schema": renderer.DERIVATION_BUNDLE_SCHEMA,
        "contract": renderer.DERIVATION_BUNDLE_CONTRACT,
        "release_id": "synthetic-native-authority-v1",
        "role": native.PDF_ID,
        "producer_protocol": native.PROTOCOL,
        "producer_arguments": native.PRODUCER_ARGUMENTS,
        "canonical_inputs": canonical_inputs,
        "canonical_inputs_projection_sha256": _sha(
            native._canonical_json(canonical_inputs)
        ),
        "dependencies": dependencies,
        "expected_output": expected_output,
        "non_inference": dict(renderer.DERIVATION_NON_INFERENCE),
        "source_or_base64_payload_recorded": False,
    }
    _redigest(projection, "bundle_projection_sha256")
    runtime_pin = _fake_pin("runtime", b"r" * 100, mode=0o500)
    renderer_pin = _fake_pin("renderer", b"e" * 101, mode=0o400)
    runtime_pin.sha256 = anchors["runtime_sha256"]
    renderer_pin.sha256 = anchors["renderer_sha256"]
    bundle = {"dependencies": dependencies}
    config = native._launcher_config(
        runtime=runtime_pin,
        runtime_path=Path("/synthetic/runtime"),
        renderer_pin=renderer_pin,
        renderer_path=Path("/synthetic/renderer"),
        bundle=bundle,
    )
    source_pins = {
        member: _fake_pin(member, member.encode("ascii"))
        for member in native.RELEVANT_RELEASE_MEMBERS
    }
    source_pins[native.LAUNCHER_SOURCE_MEMBER] = _fake_pin("launcher", source_raw)
    source_anchor_names = {
        native.LAUNCHER_SOURCE_MEMBER: "launcher_source_sha256",
        native.BUILDER_MEMBER: "builder_sha256",
        native.BUNDLE_BUILDER_MEMBER: "bundle_builder_sha256",
        "analysis/render_tcga_revision_rebuttal.py": "renderer_sha256",
        native.MACHINE_RUNNER_MEMBER: "machine_runner_sha256",
    }
    for member, anchor_name in source_anchor_names.items():
        source_pins[member].sha256 = anchors[anchor_name]
    git_pin = _fake_pin("git", b"git", mode=0o500)
    git_pin.sha256 = anchors["git_sha256"]
    source_release = native._source_release_projection(
        mode=native.MODE_SYNTHETIC,
        release_commit=None,
        release_ref=None,
        git=git_pin,
        repo_root=Path("/synthetic/repo"),
        file_pins=source_pins,
        guard=lambda: None,
    )
    thin_binary = {
        "binary_container": "thin-macho64",
        "architecture": "arm64",
        "cpu_type": native.CPU_TYPE_ARM64,
        "cpu_subtype": native.CPU_SUBTYPE_ARM64_ALL,
        "cpu_subtype_capabilities": 0,
        "file_type": "execute",
        "load_command_count": 1,
        "load_command_bytes": 8,
    }
    toolchain: dict[str, object] = {}
    for key, locator, anchor_name in (
        ("clang", "xcode-default-toolchain-clang", "clang_sha256"),
        ("linker", "xcode-default-toolchain-ld", "linker_sha256"),
        ("git", "xcode-git", "git_sha256"),
    ):
        toolchain[key] = {
            "locator": locator,
            "path_recorded": False,
            "bytes": 200,
            "sha256": anchors[anchor_name],
            "mode": "0755",
            "uid": 0,
            "link_count": 1,
            "binary": thin_binary,
        }
    selected = {
        "index": 0,
        "architecture": "arm64",
        "cpu_type": native.CPU_TYPE_ARM64,
        "cpu_subtype": native.CPU_SUBTYPE_ARM64E,
        "cpu_subtype_capabilities": 0x80000000,
        "alignment_exponent": 12,
        "offset": 4096,
        "bytes": 200,
        "sha256": _anchor("codesign-slice"),
    }
    toolchain["codesign"] = {
        "locator": "system-codesign",
        "path_recorded": False,
        "bytes": 8192,
        "sha256": anchors["codesign_sha256"],
        "mode": "0755",
        "uid": 0,
        "link_count": 1,
        "binary": {
            "binary_container": "fat-macho32",
            "fat_endianness": "big",
            "slice_count": 1,
            "slices": [selected],
            "selected_execution_slice": selected,
            "selected_slice_live_mapping": "not-attested",
        },
    }
    for key, locator, root, anchor_name in (
        (
            "compiler_resource_tree",
            "xcode-clang-resource-root",
            native.EXPECTED_COMPILER_RESOURCE_ROOT,
            "compiler_resource_tree_sha256",
        ),
        (
            "sdk_tree",
            "xcode-macos-sdk-root",
            native.EXPECTED_SDK_ROOT,
            "sdk_tree_sha256",
        ),
    ):
        toolchain[key] = {
            "locator": locator,
            "root_path_recorded": False,
            "root_path_utf8_bytes": len(os.fsencode(root)),
            "root_path_utf8_sha256": _sha(os.fsencode(root)),
            "tree_hash_contract": native.TREE_HASH_CONTRACT,
            "tree_sha256": anchors[anchor_name],
            "file_count": 2,
            "directory_count": 1,
            "symlink_count": 0,
            "entry_count": 3,
            "total_file_bytes": 400,
        }
    toolchain.update(
        {
            "sdk_version": "26.5",
            "linker_invocation": "direct-bounded-main-process",
            "codesign_invocation": "bounded-main-path-execution; selected-fat-slice-live-mapping-not-attested",
        }
    )
    _redigest(toolchain, "toolchain_projection_sha256")
    code_directory = {
        "binary_container": "thin-macho64",
        "architecture": "arm64",
        "cpu_subtype": "all",
        "hash_type": "sha256",
        "code_directory_flags": 0x202,
        "code_directory_bytes": 300,
        "cdhash": "1" * 40,
        "code_limit": 16_384,
        "code_slots": 1,
        "page_size": 16_384,
        "signature_offset": 16_384,
        "signature_bytes": 3_616,
    }
    producer_raw = b"p" * 20_000
    observations = {}
    for operation, raw_recipe in native._recipe_records().items():
        recipe = native._mapping(raw_recipe, context="recipe")
        observations[operation] = {
            "return_code": 0,
            "stdout_bytes": 0,
            "stdout_sha256": _sha(b""),
            "stderr_bytes": 0,
            "stderr_sha256": _sha(b""),
            "normalized_argv": recipe["argv"],
            "normalized_argv_sha256": recipe["argv_sha256"],
        }
    build_record = {
        "object_bytes": 10_000,
        "object_sha256": _anchor("object"),
        "unsigned_bytes": 18_000,
        "unsigned_sha256": _anchor("unsigned"),
        "signed_bytes": len(producer_raw),
        "signed_sha256": _sha(producer_raw),
        "macho_uuid": "12345678-1234-1234-1234-123456789abc",
        "native_code_directory": code_directory,
        "observations": observations,
    }
    source_bundle_pin = _fake_pin("source-bundle", b"bundle")
    source_bundle_pin.sha256 = anchors["source_bundle_sha256"]
    authority = native._authority_body(
        mode=native.MODE_SYNTHETIC,
        release_id="synthetic-native-authority-v1",
        source_bundle=source_bundle_pin,
        bundle_projection=projection,
        source_raw=source_raw,
        config=config,
        runtime_handoff=native._runtime_handoff(config),
        source_release=source_release,
        build_a=build_record,
        build_b=build_record,
        producer_raw=producer_raw,
        code_directory=code_directory,
        toolchain=toolchain,
        expected_hashes=anchors,
    )
    native._normalize_authority(authority)
    return authority


def _rehash_authority(authority: dict[str, object]) -> None:
    _redigest(authority, "manifest_body_sha256")


def test_authority_fixture_is_exact_canonical_and_private_free() -> None:
    authority = _authority_fixture()
    raw = native._authority_raw(authority)
    assert native._parse_authority_raw(raw) == authority
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert all(fragment not in raw for fragment in native.FORBIDDEN_CAPSULE_FRAGMENTS)
    assert authority["package_contract"] == native.PACKAGE_CONTRACT
    assert set(authority["caller_anchors"]) == native.CALLER_ANCHOR_KEYS


@pytest.mark.parametrize("anchor_name", sorted(native.CALLER_ANCHOR_KEYS))
def test_every_caller_anchor_is_semantically_cross_bound(anchor_name: str) -> None:
    authority = _authority_fixture()
    authority["caller_anchors"][anchor_name] = "f" * 64
    _rehash_authority(authority)
    with pytest.raises(native.RebuttalNativeProducerError):
        native._normalize_authority(authority)


@pytest.mark.parametrize(
    ("mutation", "nested_digest"),
    [
        (
            lambda value: value["launcher_config"].__setitem__("cwd", "/unsafe"),
            "launcher_config",
        ),
        (
            lambda value: value["runtime_handoff"].__setitem__("cwd", "/unsafe"),
            "runtime_handoff",
        ),
        (
            lambda value: value["source_release"].__setitem__(
                "git_blob_equality", "invalid"
            ),
            "source_release",
        ),
        (
            lambda value: value["toolchain"].__setitem__("sdk_version", "99.9"),
            "toolchain",
        ),
        (lambda value: value["build"].__setitem__("cwd", "/unsafe"), "build"),
        (
            lambda value: value["source_bundle"]["bundle_projection"].__setitem__(
                "schema", "downgrade-v1"
            ),
            "bundle_projection",
        ),
    ],
)
def test_self_consistent_semantic_cross_wires_fail(
    mutation: object,
    nested_digest: str,
) -> None:
    authority = _authority_fixture()
    mutation(authority)
    if nested_digest == "bundle_projection":
        _redigest(
            authority["source_bundle"]["bundle_projection"], "bundle_projection_sha256"
        )
    else:
        field = {
            "launcher_config": "launcher_config_sha256",
            "runtime_handoff": "runtime_handoff_sha256",
            "source_release": "source_release_projection_sha256",
            "toolchain": "toolchain_projection_sha256",
            "build": "build_projection_sha256",
        }[nested_digest]
        _redigest(authority[nested_digest], field)
    _rehash_authority(authority)
    with pytest.raises(native.RebuttalNativeProducerError):
        native._normalize_authority(authority)


@pytest.mark.parametrize(
    ("paths", "replacement", "digest_scope"),
    [
        (("package_contract", "member_link_count"), True, None),
        (("source_bundle", "link_count"), True, None),
        (("toolchain", "clang", "uid"), False, "toolchain"),
        (("toolchain", "clang", "link_count"), True, "toolchain"),
        (("toolchain", "clang", "binary", "cpu_subtype"), False, "toolchain"),
        (
            ("toolchain", "clang", "binary", "load_command_count"),
            "bogus",
            "toolchain",
        ),
        (
            ("toolchain", "clang", "binary", "load_command_bytes"),
            -99,
            "toolchain",
        ),
        (
            ("build", "builds", 0, "observations", "compile", "return_code"),
            False,
            "build",
        ),
        (
            ("build", "builds", 0, "observations", "compile", "stdout_bytes"),
            False,
            "build",
        ),
        (("launcher_config", "environment_inherited"), 0, "config"),
        (("launcher_config", "shell"), 0, "config"),
        (("launcher_config", "source_fd", "canonical_decimal"), 1, "config"),
        (("launcher_config", "source_fd", "link_count"), True, "config"),
        (("launcher_config", "source_fd", "minimum_bytes"), True, "config"),
        (("launcher_config", "source_fd", "seekable_and_rewound"), 1, "config"),
        (("launcher_config", "runtime", "link_count"), True, "config"),
        (("build", "byte_identity", "object"), 1, "build"),
        (
            ("build", "ad_hoc_signature", "signer_identity_authenticated"),
            0,
            "build",
        ),
        (
            ("build", "builds", 0, "native_code_directory", "code_slots"),
            True,
            "build",
        ),
        (
            ("build", "builds", 1, "native_code_directory", "code_slots"),
            True,
            "build",
        ),
        (("build", "independent_build_count"), _INTEGRAL_FLOAT, "build"),
        (
            (
                "source_bundle",
                "bundle_projection",
                "dependencies",
                "reportlab",
                "entry_count",
            ),
            _INTEGRAL_FLOAT,
            "bundle_projection",
        ),
        (
            ("toolchain", "compiler_resource_tree", "entry_count"),
            _INTEGRAL_FLOAT,
            "toolchain",
        ),
        (
            ("toolchain", "sdk_tree", "entry_count"),
            _INTEGRAL_FLOAT,
            "toolchain",
        ),
        (
            ("toolchain", "compiler_resource_tree", "root_path_utf8_bytes"),
            _INTEGRAL_FLOAT,
            "toolchain",
        ),
        (
            ("toolchain", "sdk_tree", "root_path_utf8_bytes"),
            _INTEGRAL_FLOAT,
            "toolchain",
        ),
        (
            (
                ("producer", "native_code_directory", "code_directory_flags"),
                (
                    "build",
                    "builds",
                    0,
                    "native_code_directory",
                    "code_directory_flags",
                ),
                (
                    "build",
                    "builds",
                    1,
                    "native_code_directory",
                    "code_directory_flags",
                ),
            ),
            _INTEGRAL_FLOAT,
            "build",
        ),
    ],
)
def test_authority_rejects_self_redigested_json_type_confusion(
    paths: Sequence[str | int] | Sequence[Sequence[str | int]],
    replacement: object,
    digest_scope: str | None,
) -> None:
    authority = _authority_fixture()
    paths_to_apply = (paths,) if isinstance(paths[0], (str, int)) else paths
    for path in paths_to_apply:
        _replace_nested_json(authority, path, replacement)
    if digest_scope == "config":
        config = authority["launcher_config"]
        _redigest(config, "launcher_config_sha256")
        authority["runtime_handoff"] = native._runtime_handoff(config)
    elif digest_scope == "bundle_projection":
        _redigest(
            authority["source_bundle"]["bundle_projection"],
            "bundle_projection_sha256",
        )
    elif digest_scope is not None:
        field = {
            "toolchain": "toolchain_projection_sha256",
            "build": "build_projection_sha256",
        }[digest_scope]
        _redigest(authority[digest_scope], field)
    _rehash_authority(authority)
    with pytest.raises(native.RebuttalNativeProducerError):
        native._normalize_authority(authority)


def test_authority_construction_detaches_all_mutable_constants() -> None:
    constants_before = native._canonical_json(
        {
            "arguments": native.PRODUCER_ARGUMENTS,
            "environment": native.EXACT_LAUNCH_ENVIRONMENT,
            "package": native.PACKAGE_CONTRACT,
            "non_inference": native.NON_INFERENCE_LIMITS,
        }
    )
    authority = _authority_fixture()
    authority["producer_arguments"][0] = "mutated"
    authority["package_contract"]["member_modes"][native.PRODUCER_MEMBER] = "0777"
    authority["non_inference_limits"]["host_private_paths"] = "mutated"
    authority["launcher_config"]["environment"]["LANG"] = "mutated"
    authority["runtime_handoff"]["source_fd"]["link_count"] = 99
    assert (
        native._canonical_json(
            {
                "arguments": native.PRODUCER_ARGUMENTS,
                "environment": native.EXACT_LAUNCH_ENVIRONMENT,
                "package": native.PACKAGE_CONTRACT,
                "non_inference": native.NON_INFERENCE_LIMITS,
            }
        )
        == constants_before
    )
    fresh = _authority_fixture()
    assert fresh["package_contract"]["member_link_count"] == 1
    assert fresh["source_release"]["status"] == native.SYNTHETIC_SOURCE_STATUS
    native._normalize_authority(fresh)


@pytest.mark.parametrize("member", ["renderer", "machine"])
@pytest.mark.parametrize("content_kind", ["identical", "different"])
def test_revision_rejects_alternate_release_member_paths(
    tmp_path: Path,
    member: str,
    content_kind: str,
) -> None:
    repo_root = tmp_path / "repo"
    analysis = repo_root / "analysis"
    analysis.mkdir(parents=True)
    renderer_path = analysis / "render_tcga_revision_rebuttal.py"
    machine_path = analysis / "build_tcga_revision_rendered_document_machine_closure.py"
    renderer_path.write_bytes(b"canonical renderer")
    machine_path.write_bytes(b"canonical machine")
    alternate = tmp_path / f"alternate-{member}.py"
    canonical = renderer_path if member == "renderer" else machine_path
    alternate.write_bytes(
        canonical.read_bytes() if content_kind == "identical" else b"alternate bytes"
    )
    selected_renderer = alternate if member == "renderer" else renderer_path
    selected_machine = alternate if member == "machine" else machine_path
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="exact canonical repository member paths",
    ):
        native._require_release_member_paths(
            mode=native.MODE_REVISION,
            repo_root=repo_root,
            renderer_path=selected_renderer,
            machine_runner_path=selected_machine,
        )
    native._require_release_member_paths(
        mode=native.MODE_SYNTHETIC,
        repo_root=repo_root,
        renderer_path=selected_renderer,
        machine_runner_path=selected_machine,
    )


def test_every_numeric_authority_leaf_rejects_json_scalar_type_confusion() -> None:
    baseline = json.loads(native._canonical_json(_authority_fixture()))
    mutations = _numeric_leaf_mutations(baseline)
    accepted: list[str] = []
    for path, replacement in mutations:
        candidate = json.loads(native._canonical_json(baseline))
        _replace_nested_json(candidate, path, replacement)
        _redigest_scalar_mutation(candidate, path)
        try:
            native._normalize_authority(candidate)
        except native.RebuttalNativeProducerError:
            continue
        accepted.append(f"{path}={replacement!r}")
    assert len(mutations) == 965
    assert not accepted, "accepted scalar type confusions: " + "; ".join(accepted)


def test_authority_parser_rejects_noncanonical_duplicate_and_newline_shapes() -> None:
    raw = native._authority_raw(_authority_fixture())
    variants = [
        raw[:-1],
        raw + b"\n",
        b" " + raw,
        raw.replace(b'{"authentication"', b'{"authentication":"x","authentication"', 1),
    ]
    for variant in variants:
        with pytest.raises(native.RebuttalNativeProducerError):
            native._parse_authority_raw(variant)


def test_recipe_normalization_binds_exact_expanded_argv() -> None:
    recipe = native._mapping(native._recipe_records()["verify"], context="verify")
    executable = Path("/synthetic/codesign")
    arguments = ["--verify", "--strict", "--verbose=0", "/synthetic/signed"]
    record = native._normalized_recipe_invocation(
        "verify",
        executable,
        arguments,
        bindings={
            "{codesign}": str(executable),
            "{signed_executable}": "/synthetic/signed",
        },
    )
    assert record == {
        "normalized_argv": recipe["argv"],
        "normalized_argv_sha256": recipe["argv_sha256"],
    }
    with pytest.raises(native.RebuttalNativeProducerError):
        native._normalized_recipe_invocation(
            "verify",
            executable,
            [*arguments[:-1], "/synthetic/other"],
            bindings={
                "{codesign}": str(executable),
                "{signed_executable}": "/synthetic/signed",
            },
        )


def _thin_macho(
    *, cpu: int = native.CPU_TYPE_ARM64, subtype: int = 0, kind: int = 2
) -> bytes:
    command = struct.pack("<II", 1, 8)
    return (
        struct.pack("<IiiIIIII", native.MH_MAGIC_64, cpu, subtype, kind, 1, 8, 0, 0)
        + command
    )


def _macho_with_commands(commands: Sequence[bytes]) -> bytes:
    command_bytes = b"".join(commands)
    return (
        struct.pack(
            "<IiiIIIII",
            native.MH_MAGIC_64,
            native.CPU_TYPE_ARM64,
            native.CPU_SUBTYPE_ARM64_ALL,
            native.MH_EXECUTE,
            len(commands),
            len(command_bytes),
            0,
            0,
        )
        + command_bytes
    )


def test_thin_macho_parser_rejects_wrong_container_architecture_and_kind() -> None:
    parsed = native._parse_thin_macho_header(_thin_macho(), context="synthetic")
    assert parsed["architecture"] == "arm64"
    for raw in (
        b"short",
        struct.pack(">I", native.FAT_MAGIC) + b"\0" * 36,
        _thin_macho(kind=6),
        _thin_macho()[:-1],
    ):
        with pytest.raises(native.RebuttalNativeProducerError):
            native._parse_thin_macho_header(raw, context="synthetic")
    assert (
        native._parse_thin_macho_header(
            _thin_macho(cpu=native.CPU_TYPE_X86_64),
            context="synthetic x86",
        )["architecture"]
        == "x86_64"
    )


def test_macho_uuid_parser_requires_one_nonzero_canonical_command() -> None:
    uuid_raw = bytes.fromhex("12345678123412341234123456789abc")
    command = struct.pack("<II", native.LC_UUID, 24) + uuid_raw
    assert (
        native._required_macho_uuid(
            _macho_with_commands([command]), context="synthetic UUID"
        )
        == "12345678-1234-1234-1234-123456789abc"
    )
    malformed = [
        _thin_macho(),
        _macho_with_commands([command, command]),
        _macho_with_commands([struct.pack("<II", native.LC_UUID, 24) + b"\0" * 16]),
        _macho_with_commands([struct.pack("<II", native.LC_UUID, 16) + b"\1" * 8]),
    ]
    for raw in malformed:
        with pytest.raises(native.RebuttalNativeProducerError):
            native._required_macho_uuid(raw, context="synthetic UUID")


def _write_package(
    root: Path, *, producer: bytes = b"producer", authority: bytes = b"{}\n"
) -> None:
    root.mkdir(mode=0o700)
    (root / native.PRODUCER_MEMBER).write_bytes(producer)
    (root / native.AUTHORITY_MEMBER).write_bytes(authority)
    (root / native.PRODUCER_MEMBER).chmod(0o500)
    (root / native.AUTHORITY_MEMBER).chmod(0o400)
    root.chmod(0o500)


def _native_call_kwargs() -> dict[str, object]:
    zero = "0" * 64
    return {
        "runtime": Path("/unused/runtime"),
        "renderer_path": Path("/unused/renderer"),
        "machine_runner": Path("/unused/machine"),
        "clang": Path("/unused/clang"),
        "linker": Path("/unused/ld"),
        "codesign": Path("/unused/codesign"),
        "git": Path("/unused/git"),
        "compiler_resource_root": Path("/unused/resources"),
        "sdk_root": Path("/unused/sdk"),
        "release_id": "synthetic-test-v1",
        "mode": native.MODE_SYNTHETIC,
        "release_commit": None,
        "release_ref": None,
        **{f"expected_{key}": zero for key in native.CALLER_ANCHOR_KEYS},
    }


def _dummy_receipt(
    package: Path,
    independent: Path,
    *,
    replay_of: str | None = None,
) -> native.RebuttalNativeProducerReceipt:
    return native.RebuttalNativeProducerReceipt(
        package_root=str(package),
        independent_build_root=str(independent),
        authority_sha256="0" * 64,
        authority_bytes=3,
        producer_sha256="0" * 64,
        producer_bytes=8,
        producer_cdhash="0" * 40,
        release_id="synthetic-test-v1",
        mode=native.MODE_SYNTHETIC,
        replay_of=replay_of,
        promotable=False,
    )


def test_package_publication_is_atomic_no_replace_and_keeps_pin(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    destination = tmp_path / "published"
    _write_package(candidate)
    _, parent = native._safe_destination_parent(destination)
    try:
        pin = native._publish_candidate(candidate, destination, parent)
        assert pin.path == destination
        assert pin.member_bytes[native.PRODUCER_MEMBER] == b"producer"
        pin.revalidate(context="test published pin")
        pin.close()
    finally:
        parent.close()
    second = tmp_path / "second"
    _write_package(second)
    _, parent = native._safe_destination_parent(tmp_path / "unused")
    try:
        with pytest.raises(native.RebuttalNativeProducerError):
            native._publish_candidate(second, destination, parent)
    finally:
        parent.close()


def test_build_rejects_package_substitution_against_materialized_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained: Path | None = None

    def materialize(
        _source: Path,
        parent: native._ParentPin,
        **_kwargs: object,
    ) -> native._MaterializedCandidates:
        nonlocal retained
        primary = parent.path / "candidate-a"
        retained = parent.path / "candidate-b"
        _write_package(primary, producer=b"substitute", authority=b'{"b":2}\n')
        _write_package(retained, producer=b"substitute", authority=b'{"b":2}\n')
        return native._MaterializedCandidates(
            primary=primary,
            independent=retained,
            producer_raw=b"expected",
            authority_raw=b'{"a":1}\n',
            authority={"a": 1},
        )

    monkeypatch.setattr(native, "_materialize_candidates", materialize)
    destination = tmp_path / "published"
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="differs from materialized bytes",
    ):
        native.build_rebuttal_native_producer(
            Path("/unused/source"),
            destination,
            **_native_call_kwargs(),
        )
    assert destination.is_dir()
    assert retained is not None
    assert retained.is_dir()


def test_build_revalidates_retained_package_after_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained: Path | None = None

    def materialize(
        _source: Path,
        parent: native._ParentPin,
        **_kwargs: object,
    ) -> native._MaterializedCandidates:
        nonlocal retained
        primary = parent.path / "candidate-a"
        retained = parent.path / "candidate-b"
        _write_package(primary)
        _write_package(retained)
        return native._MaterializedCandidates(
            primary=primary,
            independent=retained,
            producer_raw=b"producer",
            authority_raw=b"{}\n",
            authority={},
        )

    def receipt(
        pin: native._PackagePin,
        independent: Path,
        *,
        replay_of: str | None,
    ) -> native.RebuttalNativeProducerReceipt:
        assert replay_of is None
        member = independent / native.PRODUCER_MEMBER
        member.chmod(0o700)
        member.write_bytes(b"mutation")
        member.chmod(0o500)
        return _dummy_receipt(pin.path, independent)

    monkeypatch.setattr(native, "_materialize_candidates", materialize)
    monkeypatch.setattr(native, "_receipt_from_pin", receipt)
    destination = tmp_path / "published"
    with pytest.raises(native.RebuttalNativeProducerError):
        native.build_rebuttal_native_producer(
            Path("/unused/source"),
            destination,
            **_native_call_kwargs(),
        )
    assert destination.is_dir()
    assert retained is not None
    assert retained.is_dir()


@pytest.mark.parametrize("mutation_target", ["original", "independent"])
def test_replay_revalidates_all_held_packages_after_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
) -> None:
    original = tmp_path / "original"
    replay_parent = tmp_path / "replay-parent"
    replay_parent.mkdir(mode=0o700)
    _write_package(original)
    retained: Path | None = None

    def materialize(
        _source: Path,
        parent: native._ParentPin,
        **_kwargs: object,
    ) -> native._MaterializedCandidates:
        nonlocal retained
        primary = parent.path / "candidate-a"
        retained = parent.path / "candidate-b"
        _write_package(primary)
        _write_package(retained)
        return native._MaterializedCandidates(
            primary=primary,
            independent=retained,
            producer_raw=b"producer",
            authority_raw=b"{}\n",
            authority={},
        )

    def parse(_raw: bytes) -> dict[str, object]:
        return {
            "release_id": "synthetic-test-v1",
            "mode": native.MODE_SYNTHETIC,
            "source_bundle": {"sha256": "0" * 64},
        }

    def receipt(
        pin: native._PackagePin,
        independent: Path,
        *,
        replay_of: str | None,
    ) -> native.RebuttalNativeProducerReceipt:
        target = original if mutation_target == "original" else independent
        member = target / native.PRODUCER_MEMBER
        member.chmod(0o700)
        member.write_bytes(b"mutation")
        member.chmod(0o500)
        return _dummy_receipt(pin.path, independent, replay_of=replay_of)

    monkeypatch.setattr(native, "_materialize_candidates", materialize)
    monkeypatch.setattr(native, "_parse_authority_raw", parse)
    monkeypatch.setattr(native, "_receipt_from_pin", receipt)
    authority = (original / native.AUTHORITY_MEMBER).read_bytes()
    producer = (original / native.PRODUCER_MEMBER).read_bytes()
    replay = replay_parent / "replay"
    with pytest.raises(native.RebuttalNativeProducerError):
        native.validate_rebuttal_native_producer(
            Path("/unused/source"),
            original,
            replay,
            expected_authority_sha256=_sha(authority),
            expected_producer_sha256=_sha(producer),
            **_native_call_kwargs(),
        )
    assert replay.is_dir()
    assert retained is not None
    assert retained.is_dir()


def test_package_pin_revalidation_rejects_member_owner_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    _write_package(package)
    pin = native._pin_package(package, context="owner-drift package")
    producer_descriptor = pin.member_descriptors[native.PRODUCER_MEMBER]
    fstat = os.fstat

    def drift_owner(descriptor: int) -> object:
        observed = fstat(descriptor)
        if descriptor != producer_descriptor:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns,
            st_mode=observed.st_mode,
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid + 1,
        )

    monkeypatch.setattr(os, "fstat", drift_owner)
    try:
        with pytest.raises(native.RebuttalNativeProducerError):
            pin.revalidate(context="owner-drift package")
    finally:
        pin.close()


@pytest.mark.parametrize("drift", ["mode", "owner"])
def test_package_pin_rejects_root_metadata_drift_while_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    package = tmp_path / "package"
    _write_package(package)
    fstat = os.fstat
    altered = False

    def drift_root(descriptor: int) -> object:
        nonlocal altered
        observed = fstat(descriptor)
        if altered or not stat.S_ISDIR(observed.st_mode):
            return observed
        altered = True
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns,
            st_mode=(
                (observed.st_mode & ~0o7777) | 0o700
                if drift == "mode"
                else observed.st_mode
            ),
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid + 1 if drift == "owner" else observed.st_uid,
        )

    monkeypatch.setattr(os, "fstat", drift_root)
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="root changed while opened",
    ):
        native._pin_package(package, context="root-drift package")


def test_parent_pin_rejects_mode_drift_before_publication(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent"
    parent_path.mkdir(mode=0o700)
    candidate = parent_path / "candidate"
    destination = parent_path / "published"
    _write_package(candidate)
    _, parent = native._safe_destination_parent(destination)
    parent_path.chmod(0o777)
    try:
        with pytest.raises(native._ParentMappingError):
            native._publish_candidate(candidate, destination, parent)
        assert candidate.exists()
        assert not destination.exists()
    finally:
        parent_path.chmod(0o700)
        parent.close()


def test_stage_root_is_registered_before_post_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "parent"
    parent_path.mkdir(mode=0o700)
    _, parent = native._safe_destination_parent(parent_path / "unused")
    stages: list[Path] = []
    chmod = os.chmod

    def fail_stage_chmod(
        path: Path,
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(path).name.startswith(".dialect-rebuttal-native-build-a-"):
            raise OSError("injected chmod failure")
        chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "chmod", fail_stage_chmod)
    try:
        with pytest.raises(OSError, match="injected chmod failure"):
            native._new_stage_root(parent, stages, label="build-a")
        assert len(stages) == 1
        assert stages[0].is_dir()
    finally:
        parent.close()


def test_stage_root_is_registered_before_post_create_parent_mapping_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "parent"
    parent_path.mkdir(mode=0o700)
    _, parent = native._safe_destination_parent(parent_path / "unused")
    stages: list[Path] = []
    original_revalidate = native._ParentPin.revalidate
    calls = 0

    def fail_second_revalidation(
        pin: native._ParentPin,
        *,
        context: str,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise native._ParentMappingError("injected parent mapping failure")
        original_revalidate(pin, context=context)

    monkeypatch.setattr(native._ParentPin, "revalidate", fail_second_revalidation)
    try:
        with pytest.raises(native._ParentMappingError):
            native._new_stage_root(parent, stages, label="build-a")
        assert len(stages) == 1
        assert stages[0].is_dir()
    finally:
        parent.close()


def test_post_rename_parent_mapping_loss_is_reported_as_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "parent"
    moved_path = tmp_path / "moved-parent"
    parent_path.mkdir(mode=0o700)
    candidate = parent_path / "candidate"
    destination = parent_path / "published"
    _write_package(candidate)
    _, parent = native._safe_destination_parent(destination)
    rename = renderer._rename_no_replace

    def move_parent_then_rename(
        descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        parent_path.rename(moved_path)
        rename(descriptor, source_name, destination_name)

    monkeypatch.setattr(renderer, "_rename_no_replace", move_parent_then_rename)
    try:
        with pytest.raises(native._PublicationError) as raised:
            native._publish_candidate(candidate, destination, parent)
        assert raised.value.renamed is True
        assert raised.value.location is None
        assert "ambiguous-via-held-parent-fd" in str(raised.value)
        assert (moved_path / destination.name).is_dir()
    finally:
        if moved_path.exists():
            moved_path.rename(parent_path)
        parent.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-key",
        "wrong-index",
        "bool-index",
        "bool-slice-count",
        "bool-alignment",
        "capability-drift",
        "misaligned-offset",
        "extent-overflow",
        "invalid-sha",
        "selected-drift",
    ],
)
def test_codesign_fat_slice_projection_rejects_semantic_drift(  # noqa: C901
    mutation: str,
) -> None:
    authority = _authority_fixture()
    toolchain = authority["toolchain"]
    binary = toolchain["codesign"]["binary"]
    slice_record = binary["slices"][0]
    if mutation == "extra-key":
        slice_record["unexpected"] = 1
    elif mutation == "wrong-index":
        slice_record["index"] = 1
    elif mutation == "bool-index":
        slice_record["index"] = False
    elif mutation == "bool-slice-count":
        binary["slice_count"] = True
    elif mutation == "bool-alignment":
        slice_record["alignment_exponent"] = True
    elif mutation == "capability-drift":
        slice_record["cpu_subtype_capabilities"] = 0
    elif mutation == "misaligned-offset":
        slice_record["offset"] = 4097
    elif mutation == "extent-overflow":
        slice_record["bytes"] = 8192
    elif mutation == "invalid-sha":
        slice_record["sha256"] = "F" * 64
    elif mutation == "selected-drift":
        binary["selected_execution_slice"] = {
            **slice_record,
            "sha256": "f" * 64,
        }
    else:  # pragma: no cover - parametrization exhaustiveness.
        raise AssertionError(mutation)
    _redigest(toolchain, "toolchain_projection_sha256")
    _rehash_authority(authority)
    with pytest.raises(native.RebuttalNativeProducerError):
        native._normalize_authority(authority)


PROBE_SOURCE = r"""
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
extern char **environ;
int main(int argc, char **argv) {
  char cwd[64];
  DIR *directory;
  struct dirent *entry;
  int directory_fd;
  int fd;
  int open_count = 0;
  int i;
  if (argc != 13 || getcwd(cwd, sizeof(cwd)) == NULL) return 90;
  fd = atoi(argv[10]);
  directory = opendir("/dev/fd");
  if (directory == NULL) return 91;
  directory_fd = dirfd(directory);
  while ((entry = readdir(directory)) != NULL) {
    char *end = NULL;
    long parsed = strtol(entry->d_name, &end, 10);
    if (end == entry->d_name || *end != '\0' || parsed < 3) continue;
    if ((int)parsed != fd && (int)parsed != directory_fd) return 91;
    if ((int)parsed == fd) ++open_count;
  }
  if (closedir(directory) != 0 || open_count != 1 ||
      (fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0 || lseek(fd, 0, SEEK_CUR) != 0) return 92;
  if ((fcntl(fd, F_GETFL) & (O_ACCMODE | O_NONBLOCK)) != O_NONBLOCK) return 93;
  if (strcmp(cwd, "/") != 0) return 94;
  if (environ[0] == NULL || environ[1] == NULL || environ[2] == NULL || environ[3] != NULL) return 95;
  if (strcmp(environ[0], "LANG=C") || strcmp(environ[1], "LC_ALL=C") || strcmp(environ[2], "TZ=UTC")) return 96;
  for (i = 0; i < argc; ++i) printf("%s\n", argv[i]);
  return 0;
}
"""

PROCESS_GROUP_SOURCE = r"""
#include <stdio.h>
#include <unistd.h>
int main(void) {
  pid_t child = fork();
  if (child < 0) return 80;
  if (child == 0) {
    (void)close(STDOUT_FILENO);
    (void)close(STDERR_FILENO);
    (void)sleep(30U);
    _exit(0);
  }
  if (printf("%d\n", child) < 0 || fflush(stdout) != 0) return 81;
  return 0;
}
"""

SHA_HARNESS_SUFFIX = r"""
#undef main
#include <stdio.h>
int main(int argc, char **argv) {
  struct dialect_sha256 context;
  unsigned char buffer[65536];
  unsigned char digest[32];
  size_t chunk;
  size_t count;
  size_t index;
  if (argc != 2) return 80;
  chunk = (size_t)strtoul(argv[1], NULL, 10);
  if (chunk < 1U || chunk > sizeof(buffer)) return 81;
  sha256_init(&context);
  while ((count = fread(buffer, 1U, chunk, stdin)) > 0U) {
    sha256_update(&context, buffer, count);
  }
  if (ferror(stdin)) return 82;
  sha256_final(&context, digest);
  for (index = 0U; index < sizeof(digest); ++index) {
    printf("%02x", digest[index]);
  }
  printf("\n");
  return 0;
}
"""


def _compile(
    source: Path,
    output: Path,
    *,
    definitions: Sequence[str] = (),
) -> None:
    object_path = output.with_name(f"{output.name}.o")
    compile_command = [
        str(native.EXPECTED_CLANG),
        "-arch",
        "arm64",
        "-target",
        "arm64-apple-macos13.0",
        "--no-default-config",
        "-std=c11",
        "-Os",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-isysroot",
        str(native.EXPECTED_SDK_ROOT),
        "-resource-dir",
        str(native.EXPECTED_COMPILER_RESOURCE_ROOT),
        *definitions,
        str(source),
        "-c",
        "-o",
        str(object_path),
    ]
    subprocess.run(
        compile_command,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(native.EXPECTED_LD),
            "-arch",
            "arm64",
            "-syslibroot",
            str(native.EXPECTED_SDK_ROOT),
            "-platform_version",
            "macos",
            native.MACOS_MINIMUM,
            native._sdk_version(native.EXPECTED_SDK_ROOT),
            "-lSystem",
            "-dead_strip",
            "-no_adhoc_codesign",
            "-o",
            str(output),
            str(object_path),
        ],
        check=True,
        capture_output=True,
    )
    assert (
        native._code_signature_command_count(
            output.read_bytes(), context="unsigned test fixture"
        )
        == 0
    )
    subprocess.run(
        [
            str(native.EXPECTED_CODESIGN),
            "--force",
            "--sign",
            "-",
            "--options",
            "kill",
            "--timestamp=none",
            "--identifier",
            native.SIGNATURE_IDENTIFIER,
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    native._required_macho_uuid(output.read_bytes(), context="compiled test fixture")
    output.chmod(0o500)


def _launcher_definitions(runtime: Path, renderer_path: Path) -> tuple[str, ...]:
    return (
        native._c_string_macro("DIALECT_RUNTIME_PATH", str(runtime)),
        native._c_string_macro(
            "DIALECT_RUNTIME_SHA256",
            native._sha256(runtime.read_bytes()),
        ),
        f"-DDIALECT_RUNTIME_BYTES={runtime.stat().st_size}",
        f"-DDIALECT_RUNTIME_MODE=0{stat.S_IMODE(runtime.stat().st_mode):o}",
        native._c_string_macro("DIALECT_RENDERER_PATH", str(renderer_path)),
        native._c_string_macro(
            "DIALECT_RENDERER_SHA256",
            native._sha256(renderer_path.read_bytes()),
        ),
        f"-DDIALECT_RENDERER_BYTES={renderer_path.stat().st_size}",
        f"-DDIALECT_RENDERER_MODE=0{stat.S_IMODE(renderer_path.stat().st_mode):o}",
    )


def _compile_launcher_for_dependencies(
    output: Path,
    *,
    runtime: Path,
    renderer_path: Path,
) -> None:
    _compile(
        Path(native.__file__).parent / "native/rebuttal_derivation_launcher.c",
        output,
        definitions=_launcher_definitions(runtime, renderer_path),
    )


@pytest.fixture(scope="session")
def sha_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("native SHA harness requires arm64 Darwin")
    tmp_path = tmp_path_factory.mktemp("native-sha-harness")
    launcher_source = (
        Path(native.__file__).parent / "native/rebuttal_derivation_launcher.c"
    ).read_text(encoding="ascii")
    source = tmp_path / "sha-harness.c"
    source.write_text(
        "#define main dialect_launcher_main\n" + launcher_source + SHA_HARNESS_SUFFIX,
        encoding="ascii",
    )
    output = tmp_path / "sha-harness"
    definitions = (
        native._c_string_macro("DIALECT_RUNTIME_PATH", "/synthetic/runtime"),
        native._c_string_macro("DIALECT_RUNTIME_SHA256", "0" * 64),
        "-DDIALECT_RUNTIME_BYTES=1",
        "-DDIALECT_RUNTIME_MODE=0500",
        native._c_string_macro("DIALECT_RENDERER_PATH", "/synthetic/renderer"),
        native._c_string_macro("DIALECT_RENDERER_SHA256", "0" * 64),
        "-DDIALECT_RENDERER_BYTES=1",
        "-DDIALECT_RENDERER_MODE=0400",
    )
    _compile(source, output, definitions=definitions)
    return output


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize(
    "length", [0, 3, 55, 56, 63, 64, 65, 65_535, 65_536, 65_537, 1_048_576]
)
@pytest.mark.parametrize("chunk", [1, 7, 64, 257, 65_536])
def test_embedded_sha256_matches_golden_boundaries(
    sha_harness: Path,
    length: int,
    chunk: int,
) -> None:
    raw = b"abc" if length == 3 else b"a" * length
    result = subprocess.run(
        [str(sha_harness), str(chunk)],
        input=raw,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout == f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii")


@REQUIRES_NATIVE_DARWIN
def test_embedded_sha256_matches_max_bundle_shaped_input(sha_harness: Path) -> None:
    raw = b"z" * renderer.MAX_DERIVATION_BUNDLE_BYTES
    result = subprocess.run(
        [str(sha_harness), str(65_536)],
        input=raw,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout == f"{hashlib.sha256(raw).hexdigest()}\n".encode("ascii")


@pytest.fixture(scope="session")
def native_probe(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Path]:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("native launcher probe requires arm64 Darwin")
    tmp_path = tmp_path_factory.mktemp("native-launcher-probe")
    probe_source = tmp_path / "probe.c"
    probe_runtime = tmp_path / "probe-runtime"
    renderer_path = tmp_path / "renderer.py"
    launcher = tmp_path / "launcher"
    probe_source.write_text(PROBE_SOURCE, encoding="ascii")
    renderer_path.write_bytes(b"# synthetic renderer placeholder\n")
    renderer_path.chmod(0o400)
    _compile(probe_source, probe_runtime)
    _compile_launcher_for_dependencies(
        launcher,
        runtime=probe_runtime,
        renderer_path=renderer_path,
    )
    return {"launcher": launcher, "runtime": probe_runtime, "renderer": renderer_path}


@pytest.fixture(scope="session")
def process_group_helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("native process-group helper requires arm64 Darwin")
    tmp_path = tmp_path_factory.mktemp("native-process-group-helper")
    source = tmp_path / "process-group-helper.c"
    output = tmp_path / "process-group-helper"
    source.write_text(PROCESS_GROUP_SOURCE, encoding="ascii")
    _compile(source, output)
    return output


def _invoke_launcher(
    launcher: Path,
    descriptor: int,
    *,
    extra_descriptors: Sequence[int] = (),
    arguments: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    args = list(arguments or native.PRODUCER_ARGUMENTS)
    args[5] = str(descriptor)
    return subprocess.run(
        [str(launcher), *args],
        pass_fds=(descriptor, *extra_descriptors),
        capture_output=True,
        check=False,
        env={"UNEXPECTED": "discarded"},
        cwd=str(launcher.parent),
    )


@REQUIRES_NATIVE_DARWIN
def test_bounded_tool_accepts_terminal_leader_only_process_group() -> None:
    tool = Path("/usr/bin/true")
    raw = tool.read_bytes()
    pin = native._pin_file(
        tool,
        maximum=native.MAX_TOOL_BYTES,
        context="terminal leader-only helper",
        expected_sha256=_sha(raw),
        require_executable=True,
        require_root_owner=True,
    )
    try:
        observation, stdout, stderr = native._run_bounded_tool(
            pin,
            [],
            context="terminal leader-only process group",
            before=lambda: None,
            after=lambda: None,
        )
    finally:
        pin.close()
    assert observation["return_code"] == 0
    assert stdout == b""
    assert stderr == b""


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize("members", [(), (4242, 4343)])
def test_process_group_gate_rejects_missing_or_extra_members(
    monkeypatch: pytest.MonkeyPatch,
    members: tuple[int, ...],
) -> None:
    monkeypatch.setattr(native, "_darwin_process_group_members", lambda _pgid: members)
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="not the retained leader alone",
    ):
        native._require_process_group_leader_only(
            4242,
            deadline=time.monotonic() + 1,
            wait_for_signaled_members=False,
        )


@REQUIRES_NATIVE_DARWIN
def test_process_group_gate_rejects_membership_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = iter(((4242,), (4242, 4343)))
    monkeypatch.setattr(
        native,
        "_darwin_process_group_members",
        lambda _pgid: next(observed),
    )
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="changed before parent reap",
    ):
        native._require_process_group_leader_only(
            4242,
            deadline=time.monotonic() + 1,
            wait_for_signaled_members=False,
        )


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize("members", [(4242, 0), (4242, -1), (4242, 4242)])
def test_process_group_enumeration_rejects_malformed_members(
    monkeypatch: pytest.MonkeyPatch,
    members: tuple[int, ...],
) -> None:
    class FakeProcList:
        argtypes: object = None
        restype: object = None

        def __call__(
            self,
            _process_group: int,
            buffer: object,
            _buffer_bytes: int,
        ) -> int:
            for index, process_id in enumerate(members):
                buffer[index] = process_id
            return len(members)

    fake_proc_list = FakeProcList()
    fake_library = SimpleNamespace(proc_listpgrppids=fake_proc_list)
    monkeypatch.setattr(native.ctypes, "CDLL", lambda *_args, **_kwargs: fake_library)
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="invalid or duplicate PIDs",
    ):
        native._darwin_process_group_members(4242)


@REQUIRES_NATIVE_DARWIN
def test_process_group_enumeration_rejects_possible_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcList:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            return native.MAX_PROCESS_GROUP_MEMBERS

    fake_proc_list = FakeProcList()
    fake_library = SimpleNamespace(proc_listpgrppids=fake_proc_list)
    monkeypatch.setattr(native.ctypes, "CDLL", lambda *_args, **_kwargs: fake_library)
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="exceeds its member bound",
    ):
        native._darwin_process_group_members(4242)


@REQUIRES_NATIVE_DARWIN
def test_waitid_rejects_siginfo_abi_layout_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizeof = native.ctypes.sizeof
    monkeypatch.setattr(
        native.ctypes,
        "sizeof",
        lambda value: 103 if value is native._DarwinSigInfo else sizeof(value),
    )
    with pytest.raises(
        native.RebuttalNativeProducerError,
        match="siginfo ABI layout is unsupported",
    ):
        native._require_darwin_siginfo_layout()


@REQUIRES_NATIVE_DARWIN
def test_bounded_tool_kills_process_group_before_parent_reap(
    process_group_helper: Path,
) -> None:
    raw = process_group_helper.read_bytes()
    pin = native._pin_file(
        process_group_helper,
        maximum=native.MAX_TOOL_BYTES,
        context="same-process-group helper",
        expected_sha256=_sha(raw),
        require_executable=True,
        require_effective_user_owner=True,
    )
    try:
        observation, stdout, stderr = native._run_bounded_tool(
            pin,
            [],
            context="same-process-group containment",
            before=lambda: None,
            after=lambda: None,
            require_empty_output=False,
        )
    finally:
        pin.close()
    child = int(stdout.strip())
    assert observation["return_code"] == 0
    assert stderr == b""
    for _ in range(200):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("same-process-group descendant survived bounded tool cleanup")
    source = Path(native.__file__).read_text(encoding="utf-8")
    runner = source[
        source.index("def _run_bounded_tool(") : source.index("def _path_binding(")
    ]
    assert "process.poll" not in runner
    assert runner.index("os.killpg(") < runner.index("process.wait(")


@REQUIRES_NATIVE_DARWIN
def test_launcher_accepts_actual_outer_nonblocking_pin_and_exact_handoff(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
) -> None:
    source = tmp_path / "source.bundle"
    source.write_bytes(b"synthetic canonical bundle")
    source.chmod(0o400)
    pin = machine._pin_file(source, maximum=1024, context="synthetic source")
    source_descriptor = pin.descriptor
    extra_path = tmp_path / "extra"
    extra_path.write_bytes(b"extra")
    extra = os.open(extra_path, os.O_RDONLY)
    try:
        result = _invoke_launcher(
            native_probe["launcher"],
            pin.descriptor,
            extra_descriptors=(extra,),
        )
    finally:
        os.close(extra)
        pin.close()
    assert result.returncode == 0
    assert result.stderr == b""
    lines = result.stdout.decode("ascii").splitlines()
    expected_arguments = list(native.PRODUCER_ARGUMENTS)
    expected_arguments[5] = str(source_descriptor)
    assert lines == [
        str(native_probe["runtime"]),
        "-I",
        "-S",
        "-B",
        str(native_probe["renderer"]),
        *expected_arguments,
    ]


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize(
    "token",
    ["", "0", "1", "2", "03", "+3", "-3", " 3", "3 ", "3x", str(2**31)],
)
def test_launcher_rejects_noncanonical_source_fd_tokens(
    native_probe: Mapping[str, Path], token: str
) -> None:
    arguments = list(native.PRODUCER_ARGUMENTS)
    arguments[5] = token
    result = subprocess.run(
        [str(native_probe["launcher"]), *arguments],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
def test_launcher_rejects_every_fixed_argv_mutation_and_wrong_count(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
) -> None:
    source = tmp_path / "argv-source"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        valid = list(native.PRODUCER_ARGUMENTS)
        valid[5] = str(descriptor)
        variants = [valid[:-1], [*valid, "extra"], [valid[1], valid[0], *valid[2:]]]
        for index in (0, 1, 2, 3, 4, 6, 7):
            mutated = list(valid)
            mutated[index] = "wrong"
            variants.append(mutated)
        for arguments in variants:
            result = subprocess.run(
                [str(native_probe["launcher"]), *arguments],
                pass_fds=(descriptor,),
                capture_output=True,
                check=False,
            )
            assert result.returncode == 64
            assert result.stdout == result.stderr == b""
    finally:
        os.close(descriptor)


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize(
    "extra_flag",
    [
        os.O_APPEND,
        getattr(os, "O_SYNC", os.O_APPEND),
        getattr(os, "O_DSYNC", os.O_APPEND),
        getattr(os, "O_SHLOCK", os.O_APPEND),
        getattr(os, "O_EXLOCK", os.O_APPEND),
    ],
)
def test_launcher_rejects_every_extra_source_status_flag(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
    extra_flag: int,
) -> None:
    source = tmp_path / f"flag-{extra_flag}"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | extra_flag)
    try:
        result = _invoke_launcher(native_probe["launcher"], descriptor)
    finally:
        os.close(descriptor)
    assert result.returncode == 65
    assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize("access", [os.O_WRONLY, os.O_RDWR])
def test_launcher_rejects_writable_source_descriptors(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
    access: int,
) -> None:
    source = tmp_path / f"writable-{access}"
    source.write_bytes(b"source")
    descriptor = os.open(source, access)
    source.chmod(0o400)
    try:
        result = _invoke_launcher(native_probe["launcher"], descriptor)
    finally:
        os.close(descriptor)
    assert result.returncode == 65
    assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
def test_launcher_rejects_empty_oversized_and_hardlinked_sources(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
) -> None:
    paths = [tmp_path / "empty", tmp_path / "oversized", tmp_path / "hardlinked"]
    paths[0].write_bytes(b"")
    paths[1].write_bytes(b"x" * (renderer.MAX_DERIVATION_BUNDLE_BYTES + 1))
    paths[2].write_bytes(b"source")
    os.link(paths[2], tmp_path / "hardlink-alias")
    for path in paths:
        path.chmod(0o400)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            result = _invoke_launcher(native_probe["launcher"], descriptor)
        finally:
            os.close(descriptor)
        assert result.returncode == 65
        assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
def test_launcher_rejects_pipe_directory_and_socket_sources(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
) -> None:
    read_pipe, write_pipe = os.pipe()
    directory = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    first_socket, second_socket = socket.socketpair()
    try:
        for descriptor in (read_pipe, directory, first_socket.fileno()):
            result = _invoke_launcher(native_probe["launcher"], descriptor)
            assert result.returncode == 65
            assert result.stdout == result.stderr == b""
    finally:
        os.close(read_pipe)
        os.close(write_pipe)
        os.close(directory)
        first_socket.close()
        second_socket.close()


@REQUIRES_NATIVE_DARWIN
def test_launcher_rewinds_source_and_accepts_high_fd_after_rlimit_lowering(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
) -> None:
    source = tmp_path / "high-source"
    source.write_bytes(b"source")
    source.chmod(0o400)
    original = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    high = 256
    os.dup2(original, high, inheritable=False)
    os.close(original)
    os.lseek(high, 3, os.SEEK_SET)

    def lower_limit() -> None:
        _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))

    try:
        arguments = list(native.PRODUCER_ARGUMENTS)
        arguments[5] = str(high)
        result = subprocess.run(
            [str(native_probe["launcher"]), *arguments],
            pass_fds=(high,),
            preexec_fn=lower_limit,
            capture_output=True,
            check=False,
        )
    finally:
        os.close(high)
    assert result.returncode == 0
    assert result.stderr == b""


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize("extra_count", [64, 65])
def test_launcher_closes_bounded_unexpected_descriptor_set(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
    extra_count: int,
) -> None:
    source = tmp_path / f"bounded-source-{extra_count}"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    extra_path = tmp_path / f"extra-{extra_count}"
    extra_path.write_bytes(b"extra")
    extras = [os.open(extra_path, os.O_RDONLY) for _ in range(extra_count)]
    try:
        result = _invoke_launcher(
            native_probe["launcher"], descriptor, extra_descriptors=extras
        )
    finally:
        for extra in extras:
            os.close(extra)
        os.close(descriptor)
    assert result.returncode == (0 if extra_count == 64 else 69)
    assert result.stderr == b""


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize("mode", [0o000, 0o040, 0o400, 0o440, 0o444, 0o600])
def test_launcher_requires_exact_private_source_mode(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
    mode: int,
) -> None:
    source = tmp_path / f"source-{mode:o}"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    source.chmod(mode)
    try:
        result = _invoke_launcher(native_probe["launcher"], descriptor)
    finally:
        os.close(descriptor)
    expected = 0 if mode == 0o400 else 65
    assert result.returncode == expected
    assert result.stderr == b""


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize(
    ("dependency", "drift_mode", "expected_code"),
    [("runtime", 0o700, 67), ("renderer", 0o500, 68)],
)
def test_launcher_rejects_same_bytes_at_a_different_dependency_mode(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
    dependency: str,
    drift_mode: int,
    expected_code: int,
) -> None:
    source = tmp_path / f"dependency-mode-{dependency}"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    path = native_probe[dependency]
    original_mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(drift_mode)
    try:
        result = _invoke_launcher(native_probe["launcher"], descriptor)
    finally:
        path.chmod(original_mode)
        os.close(descriptor)
    assert result.returncode == expected_code
    assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
@pytest.mark.parametrize(
    ("dependency", "expected_code"),
    [("runtime", 67), ("renderer", 68)],
)
def test_launcher_rejects_same_size_dependency_byte_drift(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
    dependency: str,
    expected_code: int,
) -> None:
    runtime = tmp_path / "runtime"
    renderer_path = tmp_path / "renderer.py"
    launcher = tmp_path / "launcher"
    runtime.write_bytes(native_probe["runtime"].read_bytes())
    runtime.chmod(0o500)
    renderer_path.write_bytes(native_probe["renderer"].read_bytes())
    renderer_path.chmod(0o400)
    _compile_launcher_for_dependencies(
        launcher,
        runtime=runtime,
        renderer_path=renderer_path,
    )
    target = runtime if dependency == "runtime" else renderer_path
    target_mode = stat.S_IMODE(target.stat().st_mode)
    original = target.read_bytes()
    target.chmod(target_mode | 0o200)
    writable = os.open(target, os.O_RDWR)
    target.chmod(target_mode)
    source = tmp_path / "source.bundle"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    replacement = b"\1" if original[:1] == b"\0" else b"\0"
    try:
        assert os.pwrite(writable, replacement, 0) == 1
        result = _invoke_launcher(launcher, descriptor)
    finally:
        os.pwrite(writable, original[:1], 0)
        os.close(writable)
        os.close(descriptor)
    assert result.returncode == expected_code
    assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
def test_launcher_reports_exact_execve_failure_as_126(tmp_path: Path) -> None:
    runtime = tmp_path / "not-a-mach-o-runtime"
    renderer_path = tmp_path / "renderer.py"
    launcher = tmp_path / "launcher"
    runtime.write_bytes(b"not a Mach-O executable\n")
    runtime.chmod(0o500)
    renderer_path.write_bytes(b"# renderer placeholder\n")
    renderer_path.chmod(0o400)
    _compile_launcher_for_dependencies(
        launcher,
        runtime=runtime,
        renderer_path=renderer_path,
    )
    source = tmp_path / "source.bundle"
    source.write_bytes(b"source")
    source.chmod(0o400)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        result = _invoke_launcher(launcher, descriptor)
    finally:
        os.close(descriptor)
    assert result.returncode == 126
    assert result.stdout == result.stderr == b""


@REQUIRES_NATIVE_DARWIN
def test_signed_launcher_parser_binds_exact_current_codesign_shape(
    native_probe: Mapping[str, Path],
) -> None:
    raw = native_probe["launcher"].read_bytes()
    parsed = native._parse_signed_arm64_launcher(raw)
    assert parsed["code_directory_flags"] == 0x202
    assert parsed["page_size"] == 16_384
    assert parsed["signature_offset"] + parsed["signature_bytes"] == len(raw)
    assert native._code_signature_command_count(raw, context="signed probe") == 1


@REQUIRES_NATIVE_DARWIN
def test_signed_launcher_passes_suspended_machine_attestation(
    tmp_path: Path,
    native_probe: Mapping[str, Path],
) -> None:
    source = tmp_path / "attested-source.bundle"
    source.write_bytes(b"source")
    source.chmod(0o400)
    launcher_pin = machine._pin_file(
        native_probe["launcher"],
        maximum=native.MAX_EXECUTABLE_BYTES,
        context="attested native launcher",
    )
    source_pin = machine._pin_file(
        source,
        maximum=native.MAX_BUNDLE_BYTES,
        context="attested source bundle",
    )
    try:
        arguments = [
            value.replace("{source_fd}", "3") for value in native.PRODUCER_ARGUMENTS
        ]
        return_code, stdout, stderr, attestation = machine._run_bounded(
            launcher_pin,
            arguments,
            inherited_fd_binding=(source_pin.descriptor, 3),
            timeout=10.0,
            stdout_limit=4096,
            stderr_limit=4096,
            budget=machine._ProcessBudget(),
            before=lambda: None,
            after=lambda: None,
        )
    finally:
        source_pin.close()
        launcher_pin.close()
    expected_cd = native._parse_signed_arm64_launcher(
        native_probe["launcher"].read_bytes()
    )
    assert return_code == 0
    assert stderr == b""
    assert stdout.splitlines() == [
        os.fsencode(native_probe["runtime"]),
        b"-I",
        b"-S",
        b"-B",
        os.fsencode(native_probe["renderer"]),
        *[value.encode() for value in arguments],
    ]
    assert attestation["observed_cdhash"] == expected_cd["cdhash"]
    assert (
        attestation["code_signing_status"] & machine.REQUIRED_CS_FLAGS
        == machine.REQUIRED_CS_FLAGS
    )
    assert not attestation["code_signing_status"] & machine.REJECTED_CS_FLAGS


@REQUIRES_NATIVE_DARWIN
def test_current_universal_codesign_parser_selects_exact_arm64e_slice() -> None:
    parsed = native._parse_fat_codesign(native.EXPECTED_CODESIGN.read_bytes())
    selected = parsed["selected_execution_slice"]
    assert selected["architecture"] == "arm64"
    assert selected["cpu_subtype"] == native.CPU_SUBTYPE_ARM64E
    assert selected["offset"] > 0


def test_source_has_no_process_creation_shell_or_output_calls() -> None:
    raw = (
        Path(native.__file__).parent / "native/rebuttal_derivation_launcher.c"
    ).read_text(encoding="ascii")
    for forbidden in (
        "fork(",
        "system(",
        "popen(",
        "posix_spawn(",
        "printf(",
        "fprintf(",
        "write(",
    ):
        assert forbidden not in raw
    assert "execve(" in raw
    assert "argc != 9" in raw
    assert "O_ACCMODE | O_NONBLOCK" in raw
    assert "getdtablesize" not in raw
    assert 'chdir("/")' in raw
    main_body = raw[raw.index("int main(") :]
    assert (
        main_body.index("close_unexpected_descriptors")
        < main_body.index("clear_source_cloexec")
        < main_body.index("execve(")
    )

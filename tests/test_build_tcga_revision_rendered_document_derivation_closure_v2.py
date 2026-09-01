"""Synthetic adversarial tests for sealed-package derivation closure v2."""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from analysis import (
    build_tcga_revision_rendered_document_derivation_closure as closure,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# This file uses only synthetic opaque bundles, producer bytes, and PDF bytes.
# ruff: noqa: PLR0913, PLR0915, SLF001

RELEASE_ID = "revision-v2-synthetic-fixture"


@pytest.fixture(autouse=True)
def _restore_tree_modes(tmp_path: Path) -> Iterator[None]:
    yield
    for directory, child_directories, _files in os.walk(tmp_path):
        path = Path(directory)
        if not path.is_symlink():
            path.chmod(0o700)
        for child in child_directories:
            child_path = path / child
            if not child_path.is_symlink():
                child_path.chmod(0o700)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _pdf(pdf_id: str) -> bytes:
    return f"%PDF-1.7\nsealed-package v2 {pdf_id}\n%%EOF\n".encode("ascii")


def _mapping(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value)


def _array(value: object) -> list[object]:
    return cast("list[object]", value)


def _digested(value: dict[str, object], key: str) -> dict[str, object]:
    value[key] = _sha256(_canonical(value))
    return value


def _refresh_manifest_payload(manifest: dict[str, object]) -> None:
    body = dict(manifest)
    body.pop("payload_sha256", None)
    manifest["payload_sha256"] = _sha256(_canonical(body))


def _scalar_paths(
    value: object,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], int | bool]]:
    found: list[tuple[tuple[str | int, ...], int | bool]] = []
    if type(value) in {int, bool}:
        return [(path, cast("int | bool", value))]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_scalar_paths(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_scalar_paths(child, (*path, index)))
    return found


def _set_path(
    value: object,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    target = value
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]


def _scalar_substitutions(value: int | bool) -> tuple[object, ...]:  # noqa: FBT001
    if type(value) is bool:
        integer = int(value)
        return (integer, float(integer), str(value).lower(), None, -1, 2**80)
    integer = cast("int", value)
    return (
        bool(integer),
        float(integer),
        str(integer),
        None,
        -abs(integer or 1),
        2**80,
    )


def _refresh_authority_digests(authority: dict[str, object]) -> None:
    bundle = _mapping(_mapping(authority["source_bundle"])["bundle_projection"])
    canonical_inputs = _array(bundle["canonical_inputs"])
    bundle["canonical_inputs_projection_sha256"] = _sha256(
        _canonical(canonical_inputs),
    )
    for key, digest_key in (
        ("launcher_config", "launcher_config_sha256"),
        ("build", "build_projection_sha256"),
        ("toolchain", "toolchain_projection_sha256"),
        ("source_release", "source_release_projection_sha256"),
        ("runtime_handoff", "runtime_handoff_sha256"),
    ):
        projection = _mapping(authority[key])
        projection.pop(digest_key, None)
        projection[digest_key] = _sha256(_canonical(projection))
    bundle.pop("bundle_projection_sha256", None)
    bundle["bundle_projection_sha256"] = _sha256(_canonical(bundle))
    authority.pop("manifest_body_sha256", None)
    authority["manifest_body_sha256"] = _sha256(_canonical(authority))


def _refresh_manifest_digests(manifest: dict[str, object]) -> None:
    for set_key, digest_key in (
        ("source_bundle_set", "source_bundle_set_sha256"),
        ("producer_package_set", "producer_package_set_sha256"),
        ("producer_set", "producer_set_sha256"),
        (
            "producer_toolchain_authority_set",
            "producer_toolchain_authority_set_sha256",
        ),
        ("pdf_set", "pdf_set_sha256"),
    ):
        manifest[digest_key] = _sha256(_canonical(manifest[set_key]))
    _refresh_manifest_payload(manifest)


def _uuid(index: int) -> str:
    return f"00000000-0000-0000-0000-{index:012x}"


@dataclass(slots=True)
class V2Case:
    """Own one synthetic four-package revision-v2 input set."""

    root: Path
    plan_path: Path
    source_root: Path
    package_roots: dict[str, Path]
    destination: Path
    replay_root: Path
    upstream_sha256: dict[str, str]
    authority_sha256: dict[str, str]
    source_sha256: dict[str, str]
    producer_sha256: dict[str, str]
    package_sha256: dict[str, str]
    renderer_manifest_sha256: dict[str, str]
    pdf_sha256: dict[str, str]
    code_directories: dict[str, dict[str, object]]
    macho_uuids: dict[str, str]

    def plan(self) -> dict[str, object]:
        """Read this case's canonical v2 plan."""
        return json.loads(self.plan_path.read_text(encoding="ascii"))

    def kwargs(self) -> dict[str, object]:
        """Return every independent caller anchor for the public v2 API."""
        return {
            "release_id": RELEASE_ID,
            "expected_plan_sha256": _file_sha256(self.plan_path),
            "expected_builder_sha256": _file_sha256(Path(closure.__file__)),
            "expected_machine_runner_sha256": _file_sha256(
                Path(closure._machine.__file__),
            ),
            "expected_upstream_sha256": self.upstream_sha256,
            "expected_authority_sha256": self.authority_sha256,
            "expected_source_sha256": self.source_sha256,
            "expected_producer_sha256": self.producer_sha256,
            "expected_package_sha256": self.package_sha256,
            "expected_renderer_manifest_sha256": (self.renderer_manifest_sha256),
            "expected_pdf_sha256": self.pdf_sha256,
        }


def _authority(
    *,
    pdf_id: str,
    source_bytes: int,
    source_sha256: str,
    producer_bytes: int,
    producer_sha256: str,
    code_directory: dict[str, object],
    macho_uuid: str,
    caller_anchors: dict[str, str],
) -> dict[str, object]:
    arguments = closure._expected_arguments(pdf_id)
    launcher_raw = f"/* synthetic {pdf_id} launcher */\n".encode("ascii")
    launcher_member = f"analysis/native/{pdf_id}_derivation_launcher.c"
    renderer_member = f"analysis/render_tcga_revision_{pdf_id}.py"
    canonical_inputs = [
        {
            "member": member,
            "encoding": "base64",
            "bytes": index + 1,
            "sha256": _sha256(f"{pdf_id} {member}".encode("ascii")),
            "encoded_payload_sha256": _sha256(
                f"encoded {pdf_id} {member}".encode("ascii"),
            ),
        }
        for index, member in enumerate(closure.V2_BUNDLE_INPUT_MEMBERS)
    ]
    dependencies = {
        "fonts": [
            {
                "bytes": 201,
                "locator": "system-arial-unicode",
                "postscript_name": "ArialUnicodeMS",
                "role": "regular",
                "sha256": _sha256(f"{pdf_id} regular font".encode("ascii")),
            },
            {
                "bytes": 202,
                "locator": "system-arial-bold",
                "postscript_name": "Arial-BoldMT",
                "role": "bold",
                "sha256": _sha256(f"{pdf_id} bold font".encode("ascii")),
            },
        ],
        "runtime": {
            "bytes": 101,
            "locator": "synthetic-runtime",
            "python_tag": "3.12",
            "sha256": caller_anchors["runtime_sha256"],
        },
        "renderer": {
            "bytes": 102,
            "locator": "synthetic-renderer",
            "member": renderer_member,
            "sha256": caller_anchors["renderer_sha256"],
        },
        "machine_runner": {
            "bytes": Path(closure._machine.__file__).stat().st_size,
            "locator": "repository-machine-runner",
            "member": closure.MACHINE_RUNNER_MEMBER,
            "sha256": caller_anchors["machine_runner_sha256"],
        },
        "reportlab": {
            "bundle_bytes": 401,
            "bundle_sha256": _sha256(f"{pdf_id} reportlab bundle".encode("ascii")),
            "directory_count": 1,
            "entry_count": 3,
            "file_count": 2,
            "locator": "invoking-python-reportlab",
            "total_bytes": 301,
            "tree_sha256": _sha256(f"{pdf_id} reportlab tree".encode("ascii")),
        },
        "tools": [
            {
                "bytes": 301 + index,
                "locator": f"homebrew-{name}",
                "name": name,
                "sha256": _sha256(f"{pdf_id} {name}".encode("ascii")),
            }
            for index, name in enumerate(("pdfinfo", "pdffonts", "pdftotext"))
        ],
    }
    expected_output = {
        "renderer_manifest": {
            "member": "render-receipt.json",
            "bytes": 17,
            "sha256": caller_anchors["renderer_manifest_sha256"],
        },
        "pdf": {
            "member": closure.PDF_MEMBER_BY_ID[pdf_id],
            "bytes": len(_pdf(pdf_id)),
            "sha256": caller_anchors["pdf_sha256"],
        },
    }
    bundle_projection = _digested(
        {
            "schema": f"dialect-revision-{pdf_id}-derivation-bundle-v1",
            "contract": closure.V2_BUNDLE_CONTRACT,
            "release_id": RELEASE_ID,
            "role": pdf_id,
            "producer_protocol": closure.PRODUCER_PROTOCOL,
            "producer_arguments": arguments,
            "canonical_inputs": canonical_inputs,
            "canonical_inputs_projection_sha256": _sha256(
                _canonical(canonical_inputs),
            ),
            "dependencies": dependencies,
            "expected_output": expected_output,
            "non_inference": json.loads(
                _canonical(closure.V2_BUNDLE_NON_INFERENCE),
            ),
            "source_or_base64_payload_recorded": False,
        },
        "bundle_projection_sha256",
    )
    source_fd = {
        "canonical_decimal": True,
        "minimum": 3,
        "maximum": 2**31 - 1,
        "access": "O_RDONLY; O_NONBLOCK permitted for regular file",
        "type": "regular",
        "mode": "0400",
        "owner": "effective-user-id",
        "link_count": 1,
        "minimum_bytes": 1,
        "maximum_bytes": closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES,
        "seekable_and_rewound": True,
        "cloexec": "cleared-only-after-complete-validation",
    }

    def dependency_record(key: str, mode: str) -> dict[str, object]:
        dependency = _mapping(dependencies[key])
        return {
            "locator": dependency["locator"],
            "absolute_path_recorded": False,
            "absolute_path_utf8_bytes": 20,
            "absolute_path_utf8_sha256": _sha256(key.encode("ascii")),
            "bytes": dependency["bytes"],
            "sha256": dependency["sha256"],
            "mode": mode,
            "owner": "effective-user-id",
            "link_count": 1,
            "pre_exec_descriptor_hash": True,
        }

    runtime = dependency_record("runtime", "0500")
    renderer = dependency_record("renderer", "0400")
    config = _digested(
        {
            "schema": f"dialect-revision-{pdf_id}-native-launcher-config-v2",
            "protocol": closure.PRODUCER_PROTOCOL,
            "role": pdf_id,
            "argument_count_including_argv0": 9,
            "producer_arguments": arguments,
            "source_fd": source_fd,
            "runtime": runtime,
            "renderer": renderer,
            "cwd": "/",
            "environment": closure.EXACT_ENVIRONMENT,
            "environment_inherited": False,
            "shell": False,
            "process_operation": "execve-only",
            "unexpected_inherited_fds": "enumerate-/dev/fd-and-close-up-to-64",
            "stdout": "inherited-pdf-stream",
            "stderr": "inherited-and-launcher-emits-no-bytes",
            "stdin": "inherited-but-not-read-by-launcher",
            "failure_codes": {
                "64": "argument-protocol-or-fd-token",
                "65": "source-descriptor",
                "66": "cwd",
                "67": "runtime-preflight",
                "68": "renderer-preflight",
                "69": "unexpected-fd-cleanup",
                "126": "execve",
            },
        },
        "launcher_config_sha256",
    )
    recipes = closure._v2_build_recipes(pdf_id)
    empty_sha256 = _sha256(b"")
    observations = {
        operation: {
            "return_code": 0,
            "stdout_bytes": 0,
            "stdout_sha256": empty_sha256,
            "stderr_bytes": 0,
            "stderr_sha256": empty_sha256,
            "normalized_argv": _mapping(recipe)["argv"],
            "normalized_argv_sha256": _mapping(recipe)["argv_sha256"],
        }
        for operation, recipe in recipes.items()
    }
    build_record = {
        "object_bytes": 1,
        "object_sha256": _sha256(f"object {pdf_id}".encode("ascii")),
        "unsigned_bytes": producer_bytes,
        "unsigned_sha256": producer_sha256,
        "signed_bytes": producer_bytes,
        "signed_sha256": producer_sha256,
        "macho_uuid": macho_uuid,
        "native_code_directory": code_directory,
        "observations": observations,
    }
    build = _digested(
        {
            "target": {
                "architecture": "arm64",
                "platform": "macos",
                "minimum_version": "13.0",
                "sdk_version": "15.0",
            },
            "environment": closure.EXACT_ENVIRONMENT,
            "inherit_environment": False,
            "cwd": "/",
            "shell": False,
            "independent_build_count": 2,
            "distinct_stage_roots_and_output_inodes": True,
            "recipes": recipes,
            "byte_identity": {
                "object": True,
                "unsigned": True,
                "signed": True,
                "native_code_directory": True,
            },
            "ad_hoc_signature": {
                "identifier": closure._v2_signature_identifier(pdf_id),
                "timestamp": "none",
                "signer_identity_authenticated": False,
                "options": ["kill"],
                "code_directory_flags": "0x00000202",
            },
            "builds": [
                {"build_id": "a", **build_record},
                {"build_id": "b", **build_record},
            ],
        },
        "build_projection_sha256",
    )
    thin_binary = {
        "binary_container": "thin-macho64",
        "architecture": "arm64",
        "cpu_type": closure._machine.CPU_TYPE_ARM64,
        "cpu_subtype": 0,
        "cpu_subtype_capabilities": 0,
        "file_type": "execute",
        "load_command_count": 10,
        "load_command_bytes": 512,
    }

    def tool_record(locator: str, anchor: str) -> dict[str, object]:
        return {
            "locator": locator,
            "path_recorded": False,
            "bytes": 16_384,
            "sha256": caller_anchors[anchor],
            "mode": "0555",
            "uid": 0,
            "link_count": 1,
            "binary": thin_binary,
        }

    x86_slice = {
        "index": 0,
        "architecture": "x86_64",
        "cpu_type": 0x01000007,
        "cpu_subtype": 3,
        "cpu_subtype_capabilities": 0,
        "alignment_exponent": 12,
        "offset": 4096,
        "bytes": 1024,
        "sha256": _sha256(b"synthetic codesign x86 slice"),
    }
    arm_slice = {
        "index": 1,
        "architecture": "arm64",
        "cpu_type": closure._machine.CPU_TYPE_ARM64,
        "cpu_subtype": 2,
        "cpu_subtype_capabilities": 0x80000000,
        "alignment_exponent": 12,
        "offset": 8192,
        "bytes": 1024,
        "sha256": _sha256(b"synthetic codesign arm64e slice"),
    }
    codesign = tool_record("system-codesign", "codesign_sha256")
    codesign["binary"] = {
        "binary_container": "fat-macho32",
        "fat_endianness": "big",
        "slice_count": 2,
        "slices": [x86_slice, arm_slice],
        "selected_execution_slice": arm_slice,
        "selected_slice_live_mapping": "not-attested",
    }

    def tree_record(
        locator: str,
        root: str,
        anchor: str,
    ) -> dict[str, object]:
        return {
            "locator": locator,
            "root_path_recorded": False,
            "root_path_utf8_bytes": len(os.fsencode(root)),
            "root_path_utf8_sha256": _sha256(os.fsencode(root)),
            "tree_hash_contract": (
                "u64be-path-type-mode-nlink-size-content-or-symlink-target-v1"
            ),
            "tree_sha256": caller_anchors[anchor],
            "file_count": 2,
            "directory_count": 1,
            "symlink_count": 0,
            "entry_count": 3,
            "total_file_bytes": 128,
        }

    toolchain = _digested(
        {
            "clang": tool_record(
                "xcode-default-toolchain-clang",
                "clang_sha256",
            ),
            "linker": tool_record(
                "xcode-default-toolchain-ld",
                "linker_sha256",
            ),
            "codesign": codesign,
            "git": tool_record("xcode-git", "git_sha256"),
            "compiler_resource_tree": tree_record(
                "xcode-clang-resource-root",
                closure.V2_COMPILER_RESOURCE_ROOT,
                "compiler_resource_tree_sha256",
            ),
            "sdk_tree": tree_record(
                "xcode-macos-sdk-root",
                closure.V2_SDK_ROOT,
                "sdk_tree_sha256",
            ),
            "sdk_version": "15.0",
            "linker_invocation": "direct-bounded-main-process",
            "codesign_invocation": (
                "bounded-main-path-execution; "
                "selected-fat-slice-live-mapping-not-attested"
            ),
        },
        "toolchain_projection_sha256",
    )
    release_members = [
        {
            "member": launcher_member,
            "bytes": len(launcher_raw),
            "sha256": caller_anchors["launcher_source_sha256"],
        },
        {
            "member": f"analysis/build_tcga_revision_{pdf_id}_native_producer.py",
            "bytes": 1,
            "sha256": caller_anchors["builder_sha256"],
        },
        {
            "member": f"analysis/build_tcga_revision_{pdf_id}_derivation_bundle.py",
            "bytes": 1,
            "sha256": caller_anchors["bundle_builder_sha256"],
        },
        {
            "member": renderer_member,
            "bytes": 1,
            "sha256": caller_anchors["renderer_sha256"],
        },
        {
            "member": closure.MACHINE_RUNNER_MEMBER,
            "bytes": Path(closure._machine.__file__).stat().st_size,
            "sha256": caller_anchors["machine_runner_sha256"],
        },
    ]
    source_release = _digested(
        {
            "status": (
                "git-command-observed-listed-path-byte-equality-at-caller-commit"
            ),
            "release_commit": "1" * 40,
            "release_ref": "synthetic-v2",
            "git_blob_equality": True,
            "members": release_members,
            "git": {
                "locator": "xcode-git",
                "bytes": 16_384,
                "sha256": caller_anchors["git_sha256"],
                "main-executable-bytes-pinned": True,
            },
        },
        "source_release_projection_sha256",
    )
    handoff = _digested(
        {
            "execve_path": "{runtime}",
            "execve_argv": [
                "{runtime}",
                "-I",
                "-S",
                "-B",
                "{renderer}",
                *arguments,
            ],
            "placeholder_bindings": {
                "{runtime}": runtime,
                "{renderer}": renderer,
                "{source_fd}": ("validated-original-canonical-decimal-descriptor"),
            },
            "cwd": "/",
            "environment": closure.EXACT_ENVIRONMENT,
            "inherit_environment": False,
            "PATH_lookup": False,
            "shell": False,
            "stdout": "inherited",
            "stderr": "inherited",
            "source_fd": source_fd,
        },
        "runtime_handoff_sha256",
    )
    body = {
        "schema": closure.NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID[pdf_id],
        "contract": closure.NATIVE_PRODUCER_AUTHORITY_CONTRACT_V2,
        "mode": closure.MODE_REVISION,
        "release_id": RELEASE_ID,
        "pdf_id": pdf_id,
        "pdf_member": closure.PDF_MEMBER_BY_ID[pdf_id],
        "status": closure.V2_AUTHORITY_STATUS,
        "authentication": "caller-sha-anchor-only",
        "producer_protocol": closure.PRODUCER_PROTOCOL,
        "producer_arguments": arguments,
        "package_contract": closure._v2_package_contract(pdf_id),
        "source_bundle": {
            "member": closure.SOURCE_MEMBER_BY_ID[pdf_id],
            "mode": closure.V2_SOURCE_MODE,
            "owner": "effective-user-id",
            "link_count": 1,
            "bytes": source_bytes,
            "sha256": source_sha256,
            "bundle_projection": bundle_projection,
        },
        "producer": {
            "member": closure.PRODUCER_MEMBER_BY_ID[pdf_id],
            "mode": closure.V2_PRODUCER_MODE,
            "bytes": producer_bytes,
            "sha256": producer_sha256,
            "macho_uuid": macho_uuid,
            "native_code_directory": code_directory,
        },
        "launcher_source": {
            "member": launcher_member,
            "encoding": "base64",
            "bytes": len(launcher_raw),
            "sha256": _sha256(launcher_raw),
            "base64": base64.b64encode(launcher_raw).decode("ascii"),
        },
        "launcher_config": config,
        "build": build,
        "toolchain": toolchain,
        "source_release": source_release,
        "runtime_handoff": handoff,
        "expected_output": expected_output,
        "caller_anchors": caller_anchors,
        "review_scope": (
            "native-launcher-source-config-build-toolchain-bundle-projection-v2"
        ),
        "non_inference_limits": json.loads(
            _canonical(closure.NATIVE_PRODUCER_NON_INFERENCE_LIMITS_V2),
        ),
    }
    body["manifest_body_sha256"] = _sha256(_canonical(body))
    return body


def _make_case(tmp_path: Path) -> V2Case:
    source_root = tmp_path / "v2-source-bundles"
    source_root.mkdir(mode=0o700)
    package_roots: dict[str, Path] = {}
    authority_sha: dict[str, str] = {}
    source_sha: dict[str, str] = {}
    producer_sha: dict[str, str] = {}
    package_sha: dict[str, str] = {}
    renderer_manifest_sha: dict[str, str] = {}
    pdf_sha: dict[str, str] = {}
    code_directories: dict[str, dict[str, object]] = {}
    macho_uuids: dict[str, str] = {}
    caller_anchors_by_id: dict[str, dict[str, str]] = {}
    authorities: dict[str, dict[str, object]] = {}
    machine_sha = _file_sha256(Path(closure._machine.__file__))
    for index, pdf_id in enumerate(closure.PDF_IDS, start=1):
        source_raw = b"\0opaque-not-json\xff" + pdf_id.encode("ascii")
        source_path = source_root / closure.SOURCE_MEMBER_BY_ID[pdf_id]
        source_path.write_bytes(source_raw)
        source_path.chmod(0o400)
        source_sha[pdf_id] = _sha256(source_raw)
        producer_raw = (
            b"synthetic sealed producer bytes " + pdf_id.encode("ascii") + b"!"
        ).ljust(128, b"x")
        package_root = tmp_path / f"v2-package-{pdf_id}"
        package_root.mkdir(mode=0o700)
        package_roots[pdf_id] = package_root
        producer_path = package_root / closure.PRODUCER_MEMBER_BY_ID[pdf_id]
        producer_path.write_bytes(producer_raw)
        producer_path.chmod(0o500)
        producer_sha[pdf_id] = _sha256(producer_raw)
        pdf_sha[pdf_id] = _sha256(_pdf(pdf_id))
        renderer_manifest_sha[pdf_id] = _sha256(
            f"renderer manifest {pdf_id}".encode("ascii"),
        )
        code_limit = len(producer_raw) - 8
        code_directory = {
            "binary_container": "thin-macho64",
            "architecture": "arm64",
            "cpu_subtype": "all",
            "hash_type": "sha256",
            "code_directory_bytes": 44,
            "code_directory_flags": 514,
            "cdhash": f"{index:040x}",
            "code_limit": code_limit,
            "code_slots": 1,
            "page_size": 16_384,
            "signature_offset": code_limit,
            "signature_bytes": 8,
        }
        code_directories[pdf_id] = code_directory
        macho_uuids[pdf_id] = _uuid(index)
        launcher_sha = _sha256(f"/* synthetic {pdf_id} launcher */\n".encode("ascii"))
        caller_anchors = {
            key: _sha256(f"{pdf_id} {key}".encode("ascii"))
            for key in closure.V2_CALLER_ANCHOR_KEYS
        }
        caller_anchors.update(
            {
                "source_bundle_sha256": source_sha[pdf_id],
                "launcher_source_sha256": launcher_sha,
                "machine_runner_sha256": machine_sha,
                "renderer_manifest_sha256": renderer_manifest_sha[pdf_id],
                "pdf_sha256": pdf_sha[pdf_id],
            },
        )
        caller_anchors_by_id[pdf_id] = caller_anchors
        authority = _authority(
            pdf_id=pdf_id,
            source_bytes=len(source_raw),
            source_sha256=source_sha[pdf_id],
            producer_bytes=len(producer_raw),
            producer_sha256=producer_sha[pdf_id],
            code_directory=code_directory,
            macho_uuid=macho_uuids[pdf_id],
            caller_anchors=caller_anchors,
        )
        authorities[pdf_id] = authority
        authority_path = package_root / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
        authority_path.write_bytes(_canonical(authority) + b"\n")
        authority_path.chmod(0o400)
        authority_sha[pdf_id] = _file_sha256(authority_path)
        content_record = closure._v2_package_content_record(
            pdf_id,
            producer_bytes=len(producer_raw),
            producer_sha256=producer_sha[pdf_id],
            authority_bytes=authority_path.stat().st_size,
            authority_sha256=authority_sha[pdf_id],
        )
        package_sha[pdf_id] = _sha256(_canonical(content_record))
        package_root.chmod(0o500)
    source_root.chmod(0o500)
    documents: list[dict[str, object]] = []
    source_set: list[dict[str, object]] = []
    package_set: list[dict[str, object]] = []
    for pdf_id, pdf_member in closure.PDF_ORDER:
        source_path = source_root / closure.SOURCE_MEMBER_BY_ID[pdf_id]
        producer_path = package_roots[pdf_id] / closure.PRODUCER_MEMBER_BY_ID[pdf_id]
        authority_path = package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
        authority = authorities[pdf_id]
        projection = _mapping(
            _mapping(authority["source_bundle"])["bundle_projection"],
        )
        package_record = closure._v2_package_content_record(
            pdf_id,
            producer_bytes=producer_path.stat().st_size,
            producer_sha256=producer_sha[pdf_id],
            authority_bytes=authority_path.stat().st_size,
            authority_sha256=authority_sha[pdf_id],
        )
        source = {
            "member": source_path.name,
            "mode": closure.V2_SOURCE_MODE,
            "owner": "effective-user-id",
            "link_count": 1,
            "bytes": source_path.stat().st_size,
            "sha256": source_sha[pdf_id],
            "treatment": "opaque-byte-bundle-not-decoded",
        }
        package = {
            "root_mode": closure.V2_PACKAGE_ROOT_MODE,
            "root_owner": "effective-user-id",
            "package_content_sha256": package_sha[pdf_id],
            "producer": {
                "member": producer_path.name,
                "mode": closure.V2_PRODUCER_MODE,
                "bytes": producer_path.stat().st_size,
                "sha256": producer_sha[pdf_id],
                "macho_uuid": macho_uuids[pdf_id],
                "native_code_directory": code_directories[pdf_id],
            },
            "authority": {
                "member": authority_path.name,
                "mode": closure.V2_AUTHORITY_MODE,
                "bytes": authority_path.stat().st_size,
                "sha256": authority_sha[pdf_id],
                "schema": closure.NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID[pdf_id],
            },
            "authority_projection": {
                "manifest_body_sha256": authority["manifest_body_sha256"],
                "bundle_projection_sha256": projection["bundle_projection_sha256"],
                "launcher_config_sha256": _mapping(
                    authority["launcher_config"],
                )["launcher_config_sha256"],
                "runtime_handoff_sha256": _mapping(
                    authority["runtime_handoff"],
                )["runtime_handoff_sha256"],
                "build_projection_sha256": _mapping(authority["build"])[
                    "build_projection_sha256"
                ],
            },
        }
        expected_output = authority["expected_output"]
        documents.append(
            {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "producer_arguments": closure._expected_arguments(pdf_id),
                "authorization": {
                    "status": closure.V2_AUTHORIZATION_STATUS,
                    "authentication": "caller-sha-anchor-only",
                    "authority_sha256": authority_sha[pdf_id],
                },
                "caller_anchors": caller_anchors_by_id[pdf_id],
                "source_bundle": source,
                "producer_package": package,
                "expected_output": expected_output,
            },
        )
        source_set.append({"pdf_id": pdf_id, **source})
        package_set.append(
            {
                "pdf_id": pdf_id,
                "package_content_sha256": package_sha[pdf_id],
                **package_record,
            },
        )
    upstream = {
        key: _sha256(f"v2 upstream {key}".encode("ascii"))
        for key in closure.UPSTREAM_BINDING_KEYS
    }
    plan = {
        "schema": closure.DERIVATION_PLAN_SCHEMA_V2,
        "contract": closure.DERIVATION_CLOSURE_CONTRACT_V2,
        "mode": closure.MODE_REVISION,
        "release_id": RELEASE_ID,
        "upstream_bindings": upstream,
        "execution_contract": closure.EXECUTION_CONTRACT_V2,
        "source_bundle_set_sha256": _sha256(_canonical(source_set)),
        "producer_package_set_sha256": _sha256(_canonical(package_set)),
        "documents": documents,
        "non_inference_limits": closure.NON_INFERENCE_LIMITS_V2,
    }
    plan_path = tmp_path / "v2-plan.json"
    plan_path.write_bytes(_canonical(plan) + b"\n")
    return V2Case(
        root=tmp_path,
        plan_path=plan_path,
        source_root=source_root,
        package_roots=package_roots,
        destination=tmp_path / "v2-derivation-closure",
        replay_root=tmp_path / "v2-derivation-replay",
        upstream_sha256=upstream,
        authority_sha256=authority_sha,
        source_sha256=source_sha,
        producer_sha256=producer_sha,
        package_sha256=package_sha,
        renderer_manifest_sha256=renderer_manifest_sha,
        pdf_sha256=pdf_sha,
        code_directories=code_directories,
        macho_uuids=macho_uuids,
    )


def _role_for_producer(producer: object) -> str:
    name = producer.path.name
    return next(
        pdf_id
        for pdf_id in closure.PDF_IDS
        if name == closure.PRODUCER_MEMBER_BY_ID[pdf_id]
    )


def _install_synthetic_execution(
    case: V2Case,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[int, str, bytes], bytes] | None = None,
) -> list[tuple[str, list[str], int]]:
    calls: list[tuple[str, list[str], int]] = []
    monkeypatch.setattr(
        closure,
        "_parse_v2_producer_code_directory",
        lambda producer: closure._v2_code_directory_core(
            case.code_directories[_role_for_producer(producer)],
        ),
    )
    monkeypatch.setattr(
        closure,
        "_parse_v2_macho_uuid",
        lambda producer: case.macho_uuids[_role_for_producer(producer)],
    )
    monkeypatch.setattr(
        closure,
        "_parse_v2_code_directory_flags",
        lambda _producer: 514,
    )

    def invoke(
        producer: object,
        arguments: list[str],
        *,
        source: object,
        budget: object,
        before: Callable[[], None],
        after: Callable[[], None],
    ) -> tuple[int, bytes, bytes, dict[str, object]]:
        budget.consume()
        before()
        before()
        pdf_id = arguments[3]
        assert arguments == closure._adapter_arguments(
            pdf_id,
            closure.V2_SOURCE_DESCRIPTOR,
        )
        flags = fcntl.fcntl(source.descriptor, fcntl.F_GETFL)
        assert flags & os.O_NONBLOCK
        calls.append((pdf_id, list(arguments), source.descriptor))
        raw = _pdf(pdf_id)
        if mutate is not None:
            raw = mutate(len(calls), pdf_id, raw)
        expected = closure._v2_code_directory_core(case.code_directories[pdf_id])
        attestation = {
            "protocol": "darwin-posix-spawn-suspended-main-executable-v1",
            "spawn_flags": closure._machine.DARWIN_SPAWN_FLAGS,
            "suspended_wait_status": 127,
            "code_signing_status": closure._machine.REQUIRED_CS_FLAGS,
            "expected_code_directory": expected,
            "observed_cdhash": expected["cdhash"],
            "main_executable_mapping": {
                "device": producer.device & 0xFFFFFFFF,
                "inode": producer.inode,
                "path": f"/Users/private/{producer.path.name}",
                "mode": closure.V2_PRODUCER_MODE,
                "link_count": 1,
                "protection": 4,
                "file_offset": 0,
            },
            "execution_binding_scope": "main_executable",
            "non_system_dylib_closure": "not_attested",
            "same_vnode_mutation_fail_stop_assumption": (
                "invalid-signed-code-page-triggers-darwin-cs-kill"
            ),
            "other_same_vnode_mutations": "not_attested",
        }
        after()
        return 0, raw, b"", attestation

    monkeypatch.setattr(closure, "_invoke_producer_v2", invoke)
    return calls


def test_v2_package_build_and_private_replay_publish_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    calls = _install_synthetic_execution(case, monkeypatch)
    receipt = closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    assert len(calls) == 8
    assert receipt.mode == closure.MODE_REVISION
    assert receipt.promotable is False
    manifest_raw = (case.destination / closure.MANIFEST_MEMBER).read_bytes()
    assert b"/Users/" not in manifest_raw
    manifest = json.loads(manifest_raw)
    first_run = _mapping(_array(_mapping(_array(manifest["documents"])[0])["runs"])[0])
    binding = _mapping(
        json.loads(
            (
                case.destination / str(first_run["invocation_receipt_member"])
            ).read_bytes(),
        )["source_descriptor_binding"],
    )
    assert binding["child_descriptor"] == 3
    assert binding["parent_descriptor_recorded"] is False
    replay = closure.validate_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        case.replay_root,
        expected_manifest_sha256=receipt.manifest_sha256,
        **case.kwargs(),
    )
    assert len(calls) == 16
    assert replay.replay_root == str(case.replay_root)
    assert (case.replay_root / closure.MANIFEST_MEMBER).read_bytes() == manifest_raw
    private_replay = tmp_path / f".{case.replay_root.name}.private-v2-replay-candidate"
    assert not private_replay.exists()


def test_v2_invocation_uses_native_expected_output_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def run_bounded(
        producer: object,
        arguments: object,
        **kwargs: object,
    ) -> tuple[int, bytes, bytes, dict[str, object]]:
        observed.update(
            {
                "producer": producer,
                "arguments": arguments,
                **kwargs,
            },
        )
        return 0, b"", b"", {}

    class Source:
        descriptor = 7

    monkeypatch.setattr(closure._machine, "_run_bounded", run_bounded)
    producer = object()
    arguments = closure._adapter_arguments("rebuttal", closure.V2_SOURCE_DESCRIPTOR)
    result = closure._invoke_producer_v2(
        producer,
        arguments,
        source=Source(),
        budget=object(),
        before=lambda: None,
        after=lambda: None,
    )
    assert result == (0, b"", b"", {})
    assert observed["producer"] is producer
    assert observed["arguments"] == arguments
    assert observed["stdout_limit"] == closure.MAX_V2_NATIVE_OUTPUT_BYTES


def test_v2_attestation_rejects_each_suspended_process_semantic_downgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    manifest = json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes())
    document = _mapping(_array(manifest["documents"])[0])
    producer = _mapping(_mapping(document["producer_package"])["producer"])
    baseline = _mapping(
        _mapping(_array(document["runs"])[0])["launcher_pre_exec_attestation"],
    )

    def set_mapping_field(
        value: dict[str, object],
        key: str,
        *,
        replacement: object,
    ) -> None:
        _mapping(value["main_executable_mapping"])[key] = replacement

    mutations: list[Callable[[dict[str, object]], None]] = [
        lambda value: value.__setitem__("spawn_flags", 0),
        lambda value: value.__setitem__("suspended_wait_status", 0),
        lambda value: value.__setitem__(
            "code_signing_status",
            closure._machine.REQUIRED_CS_FLAGS & ~closure._machine.CS_VALID,
        ),
        lambda value: value.__setitem__(
            "code_signing_status",
            closure._machine.REQUIRED_CS_FLAGS & ~closure._machine.CS_KILL,
        ),
        lambda value: value.__setitem__(
            "code_signing_status",
            closure._machine.REQUIRED_CS_FLAGS & ~closure._machine.CS_SIGNED,
        ),
        lambda value: value.__setitem__(
            "code_signing_status",
            closure._machine.REQUIRED_CS_FLAGS | closure._machine.CS_DEBUGGED,
        ),
        lambda value: set_mapping_field(value, "protection", replacement=False),
        lambda value: set_mapping_field(value, "protection", replacement=0),
        lambda value: set_mapping_field(value, "file_offset", replacement=True),
        lambda value: set_mapping_field(value, "file_offset", replacement=1),
    ]
    for mutate in mutations:
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        with pytest.raises(closure.DerivationClosureError):
            closure._normalize_public_v2_attestation(
                candidate,
                expected_code_directory=_mapping(
                    producer["native_code_directory"],
                ),
                macho_uuid=str(producer["macho_uuid"]),
                context="synthetic v2 attestation mutation",
            )


def test_v2_manifest_rejects_document_package_crossbinding_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    baseline = json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes())

    def mutate_authority(
        value: dict[str, object],
        field: str,
        replacement: object,
    ) -> None:
        document = _mapping(_array(value["documents"])[0])
        package = _mapping(document["producer_package"])
        _mapping(package["authority"])[field] = replacement

    def mutate_projection(value: dict[str, object]) -> None:
        document = _mapping(_array(value["documents"])[0])
        projection = _mapping(
            _mapping(document["producer_package"])["authority_projection"],
        )
        projection["launcher_config_sha256"] = "f" * 64

    def mutate_root_mode(value: dict[str, object]) -> None:
        document = _mapping(_array(value["documents"])[0])
        _mapping(document["producer_package"])["root_mode"] = "0700"

    mutations: list[Callable[[dict[str, object]], None]] = [
        lambda value: mutate_authority(value, "sha256", "f" * 64),
        lambda value: mutate_authority(value, "schema", "downgraded-v1"),
        lambda value: mutate_authority(value, "mode", "0700"),
        mutate_projection,
        mutate_root_mode,
    ]
    for mutate in mutations:
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        _refresh_manifest_payload(candidate)
        with pytest.raises(closure.DerivationClosureError, match="cross-binding"):
            closure._validate_v2_manifest_semantics(candidate)


def test_v2_manifest_rejects_coherent_live_code_byte_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    baseline = json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes())
    for input_key, record_key in (
        ("builder_bytes", "builder"),
        ("machine_runner_bytes", "native_execution_dependency"),
    ):
        candidate = copy.deepcopy(baseline)
        _mapping(candidate["inputs"])[input_key] = closure.MAX_PLAN_BYTES + 1
        _mapping(candidate[record_key])["bytes"] = closure.MAX_PLAN_BYTES + 1
        _refresh_manifest_payload(candidate)
        with pytest.raises(
            closure.DerivationClosureError,
            match="live-code record exceeds its byte bound",
        ):
            closure._validate_v2_manifest_semantics(candidate)


def test_v2_manifest_rejects_divergent_rebuild_receipt_or_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    baseline = json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes())

    candidate = copy.deepcopy(baseline)
    document = _mapping(_array(candidate["documents"])[0])
    second_run = _mapping(_array(document["runs"])[1])
    mapping = _mapping(
        _mapping(second_run["launcher_pre_exec_attestation"])[
            "main_executable_mapping"
        ],
    )
    mapping["device"] = int(mapping["device"]) + 1
    _refresh_manifest_payload(candidate)
    with pytest.raises(
        closure.DerivationClosureError,
        match="rebuild receipts or attestations differ",
    ):
        closure._validate_v2_manifest_semantics(candidate)

    candidate = copy.deepcopy(baseline)
    document = _mapping(_array(candidate["documents"])[0])
    second_run = _mapping(_array(document["runs"])[1])
    second_run["invocation_receipt_sha256"] = "f" * 64
    _refresh_manifest_payload(candidate)
    with pytest.raises(
        closure.DerivationClosureError,
        match="rebuild receipts or attestations differ",
    ):
        closure._validate_v2_manifest_semantics(candidate)


def test_v2_manifest_rejects_coherent_native_output_byte_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    candidate = json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes())
    oversized = closure.MAX_V2_NATIVE_OUTPUT_BYTES + 1
    pdf_record = _mapping(_array(candidate["pdf_set"])[0])
    pdf_record["pdf_bytes"] = oversized
    document = _mapping(_array(candidate["documents"])[0])
    document["pdf_bytes"] = oversized
    pdf_members: set[str] = set()
    for raw_run in _array(document["runs"]):
        run = _mapping(raw_run)
        run["pdf_bytes"] = oversized
        pdf_members.add(str(run["pdf_member"]))
    for raw_inventory in _array(candidate["member_inventory"]):
        inventory = _mapping(raw_inventory)
        if inventory["member"] in pdf_members:
            inventory["bytes"] = oversized
    _refresh_manifest_digests(candidate)
    with pytest.raises(closure.DerivationClosureError, match="run a drifted"):
        closure._validate_v2_manifest_semantics(candidate)


def test_v2_manifest_rejects_bool_coercion_in_role_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    baseline = json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes())
    cases = (
        ("source_bundle_set", "source_bundle_set_sha256", "bytes"),
        ("producer_set", "producer_set_sha256", "bytes"),
        (
            "producer_toolchain_authority_set",
            "producer_toolchain_authority_set_sha256",
            "bytes",
        ),
        ("pdf_set", "pdf_set_sha256", "pdf_bytes"),
    )
    for set_key, digest_key, field in cases:
        candidate = copy.deepcopy(baseline)
        records = _array(candidate[set_key])
        _mapping(records[0])[field] = True
        candidate[digest_key] = _sha256(_canonical(records))
        _refresh_manifest_payload(candidate)
        with pytest.raises(closure.DerivationClosureError, match="integer"):
            closure._validate_v2_manifest_semantics(candidate)
    candidate = copy.deepcopy(baseline)
    packages = _array(candidate["producer_package_set"])
    first_member = _mapping(_array(_mapping(packages[0])["members"])[0])
    first_member["link_count"] = True
    candidate["producer_package_set_sha256"] = _sha256(_canonical(packages))
    _refresh_manifest_payload(candidate)
    with pytest.raises(closure.DerivationClosureError, match="integer"):
        closure._validate_v2_manifest_semantics(candidate)


def test_v2_each_role_authority_rejects_exhaustive_redigested_scalar_confusion(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    plan = case.plan()
    documents = {
        str(_mapping(raw)["pdf_id"]): _mapping(raw) for raw in _array(plan["documents"])
    }
    mutation_count = 0
    for pdf_id in closure.PDF_IDS:
        authority_path = (
            case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
        )
        baseline = json.loads(_canonical(json.loads(authority_path.read_bytes())))
        baseline_document = json.loads(_canonical(documents[pdf_id]))
        closure._normalize_v2_authority(
            baseline,
            release_id=RELEASE_ID,
            pdf_id=pdf_id,
            document=baseline_document,
        )
        for path, scalar in _scalar_paths(baseline):
            for replacement in _scalar_substitutions(scalar):
                candidate = json.loads(_canonical(baseline))
                _set_path(candidate, path, replacement)
                _refresh_authority_digests(candidate)
                document = json.loads(_canonical(baseline_document))
                projection = _mapping(
                    _mapping(document["producer_package"])["authority_projection"],
                )
                for projection_key in closure.V2_AUTHORITY_PROJECTION_KEYS:
                    if projection_key == "manifest_body_sha256":
                        projection[projection_key] = candidate[projection_key]
                    elif projection_key == "bundle_projection_sha256":
                        projection[projection_key] = _mapping(
                            _mapping(candidate["source_bundle"])["bundle_projection"],
                        )[projection_key]
                    else:
                        authority_key = {
                            "launcher_config_sha256": "launcher_config",
                            "runtime_handoff_sha256": "runtime_handoff",
                            "build_projection_sha256": "build",
                        }[projection_key]
                        projection[projection_key] = _mapping(candidate[authority_key])[
                            projection_key
                        ]
                try:
                    closure._normalize_v2_authority(
                        candidate,
                        release_id=RELEASE_ID,
                        pdf_id=pdf_id,
                        document=document,
                    )
                except closure.DerivationClosureError:
                    pass
                else:
                    pytest.fail(
                        "authority scalar mutation was accepted: "
                        f"{pdf_id=} {path=} {scalar=} {replacement=}",
                    )
                mutation_count += 1
    assert mutation_count == 4_824


def test_v2_authority_rejects_coherent_coupled_and_exact_key_mutations(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    pdf_id = "rebuttal"
    plan = case.plan()
    baseline_document = next(
        _mapping(raw)
        for raw in _array(plan["documents"])
        if _mapping(raw)["pdf_id"] == pdf_id
    )
    authority_path = case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
    baseline = json.loads(authority_path.read_bytes())

    def reject(
        candidate: dict[str, object],
        *,
        match: str,
        expected_release_id: str = RELEASE_ID,
        sync_source_member: bool = False,
        sync_producer_macho_uuid: bool = False,
    ) -> None:
        _refresh_authority_digests(candidate)
        document = json.loads(_canonical(baseline_document))
        projection = _mapping(
            _mapping(document["producer_package"])["authority_projection"],
        )
        projection["manifest_body_sha256"] = candidate["manifest_body_sha256"]
        projection["bundle_projection_sha256"] = _mapping(
            _mapping(candidate["source_bundle"])["bundle_projection"],
        )["bundle_projection_sha256"]
        projection["launcher_config_sha256"] = _mapping(
            candidate["launcher_config"],
        )["launcher_config_sha256"]
        projection["runtime_handoff_sha256"] = _mapping(
            candidate["runtime_handoff"],
        )["runtime_handoff_sha256"]
        projection["build_projection_sha256"] = _mapping(candidate["build"])[
            "build_projection_sha256"
        ]
        if sync_source_member:
            _mapping(document["source_bundle"])["member"] = _mapping(
                candidate["source_bundle"],
            )["member"]
        if sync_producer_macho_uuid:
            _mapping(_mapping(document["producer_package"])["producer"])[
                "macho_uuid"
            ] = _mapping(candidate["producer"])["macho_uuid"]
        with pytest.raises(closure.DerivationClosureError, match=match):
            closure._normalize_v2_authority(
                candidate,
                release_id=expected_release_id,
                pdf_id=pdf_id,
                document=document,
            )

    candidate = json.loads(_canonical(baseline))
    _mapping(candidate["build"])["unexpected"] = "coherent-redigested-extra"
    reject(candidate, match="keys differ")

    candidate = json.loads(_canonical(baseline))
    _mapping(candidate["launcher_config"]).pop("stdout")
    reject(candidate, match="keys differ")

    candidate = json.loads(_canonical(baseline))
    for raw_build in _array(_mapping(candidate["build"])["builds"]):
        _mapping(raw_build)["object_bytes"] = closure.MAX_V2_NATIVE_EXECUTABLE_BYTES + 1
    reject(candidate, match="exceeds bound")

    candidate = json.loads(_canonical(baseline))
    for raw_build in _array(_mapping(candidate["build"])["builds"]):
        compile_observation = _mapping(
            _mapping(_mapping(raw_build)["observations"])["compile"],
        )
        compile_observation["return_code"] = 1
    reject(candidate, match="not exact empty success")

    candidate = json.loads(_canonical(baseline))
    build = _mapping(candidate["build"])
    compile_recipe = _mapping(_mapping(build["recipes"])["compile"])
    compile_argv = _array(compile_recipe["argv"])
    compile_argv.append("coherent-recipe-drift")
    compile_recipe["argv_sha256"] = _sha256(_canonical(compile_argv))
    for raw_build in _array(build["builds"]):
        compile_observation = _mapping(
            _mapping(_mapping(raw_build)["observations"])["compile"],
        )
        compile_observation["normalized_argv"] = list(compile_argv)
        compile_observation["normalized_argv_sha256"] = compile_recipe["argv_sha256"]
    reject(candidate, match="double-build identity")

    candidate = json.loads(_canonical(baseline))
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    bundle["schema"] = "dialect-revision-rebuttal-derivation-bundle-v0"
    reject(candidate, match="bundle_projection fixed bindings")

    candidate = json.loads(_canonical(baseline))
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    inputs = _array(bundle["canonical_inputs"])
    inputs[0], inputs[1] = inputs[1], inputs[0]
    reject(candidate, match="canonical input member")

    candidate = json.loads(_canonical(baseline))
    handoff = _mapping(candidate["runtime_handoff"])
    _mapping(handoff["placeholder_bindings"])["{source_fd}"] = "coherent-drift"
    reject(candidate, match="runtime_handoff drifted")

    candidate = json.loads(_canonical(baseline))
    release = _mapping(candidate["source_release"])
    release_members = _array(release["members"])
    release_members[0], release_members[1] = release_members[1], release_members[0]
    reject(candidate, match="source_release")

    candidate = json.loads(_canonical(baseline))
    candidate["release_id"] = "UPPER"
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    bundle["release_id"] = "UPPER"
    reject(
        candidate,
        match="lowercase canonical token",
        expected_release_id="UPPER",
    )

    candidate = json.loads(_canonical(baseline))
    _mapping(candidate["source_bundle"])["member"] = "other-source.bundle"
    reject(
        candidate,
        match="source_bundle differs",
        sync_source_member=True,
    )

    candidate = json.loads(_canonical(baseline))
    _mapping(candidate["producer"])["macho_uuid"] = "not-a-uuid"
    for raw_build in _array(_mapping(candidate["build"])["builds"]):
        _mapping(raw_build)["macho_uuid"] = "not-a-uuid"
    reject(
        candidate,
        match="Mach-O UUID",
        sync_producer_macho_uuid=True,
    )


def test_v2_manifest_rejects_exhaustive_redigested_scalar_confusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    baseline = json.loads(
        _canonical(
            json.loads((case.destination / closure.MANIFEST_MEMBER).read_bytes()),
        ),
    )
    closure._validate_v2_manifest_semantics(baseline)
    mutation_count = 0
    for path, scalar in _scalar_paths(baseline):
        for replacement in _scalar_substitutions(scalar):
            candidate = json.loads(_canonical(baseline))
            _set_path(candidate, path, replacement)
            _refresh_manifest_digests(candidate)
            try:
                closure._validate_v2_manifest_semantics(candidate)
            except closure.DerivationClosureError:
                pass
            else:
                pytest.fail(
                    "manifest scalar mutation was accepted: "
                    f"{path=} {scalar=} {replacement=}",
                )
            mutation_count += 1
    assert mutation_count == 1_482


def test_v2_terminal_build_readback_detects_post_publication_byte_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    original_revalidate = closure._revalidate_v2_inputs
    mutated = False

    def mutate_after_publication(inputs: object) -> None:
        nonlocal mutated
        original_revalidate(inputs)
        if case.destination.exists() and not mutated:
            target = case.destination / "runs/clean/rebuild-a.pdf"
            target.chmod(0o600)
            target.write_bytes(_pdf("clean") + b"mutated")
            target.chmod(0o400)
            mutated = True

    monkeypatch.setattr(closure, "_revalidate_v2_inputs", mutate_after_publication)
    with pytest.raises(
        closure.DerivationClosureError,
        match="published-destination-remains-identity-bound",
    ):
        closure.build_derivation_closure_v2(
            case.plan_path,
            case.source_root,
            case.package_roots,
            case.destination,
            **case.kwargs(),
        )
    assert mutated
    assert case.destination.exists()


def test_v2_terminal_build_readback_reports_ambiguous_destination_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    original_revalidate = closure._revalidate_v2_inputs
    retained = tmp_path / "retained-original-v2-closure"
    swapped = False

    def swap_after_publication(inputs: object) -> None:
        nonlocal swapped
        original_revalidate(inputs)
        if case.destination.exists() and not swapped:
            case.destination.rename(retained)
            case.destination.mkdir(mode=0o500)
            swapped = True

    monkeypatch.setattr(closure, "_revalidate_v2_inputs", swap_after_publication)
    with pytest.raises(
        closure.DerivationClosureError,
        match=r"candidate_names=.*parent-path-or-name-mapping.*ambiguous",
    ):
        closure.build_derivation_closure_v2(
            case.plan_path,
            case.source_root,
            case.package_roots,
            case.destination,
            **case.kwargs(),
        )
    assert swapped
    assert retained.exists()
    assert case.destination.exists()


def test_v2_publication_diagnostic_never_claims_path_after_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    publish_parent = tmp_path / "v2-publish-parent"
    publish_parent.mkdir(mode=0o700)
    case.destination = publish_parent / "closure"
    retained_parent = tmp_path / "retained-v2-publish-parent"
    _install_synthetic_execution(case, monkeypatch)
    original_revalidate = closure._revalidate_v2_inputs
    swapped = False

    def swap_parent_after_publication(inputs: object) -> None:
        nonlocal swapped
        original_revalidate(inputs)
        if case.destination.exists() and not swapped:
            publish_parent.rename(retained_parent)
            publish_parent.mkdir(mode=0o700)
            swapped = True

    monkeypatch.setattr(closure, "_revalidate_v2_inputs", swap_parent_after_publication)
    with pytest.raises(
        closure.DerivationClosureError,
        match=r"candidate_names=.*held_parent_descriptor=.*parent-path-or-name-mapping",
    ) as raised:
        closure.build_derivation_closure_v2(
            case.plan_path,
            case.source_root,
            case.package_roots,
            case.destination,
            **case.kwargs(),
        )
    assert swapped
    assert "candidate_path=" not in str(raised.value)
    assert "candidate_paths=" not in str(raised.value)
    assert (retained_parent / "closure").exists()


def test_v2_prepublication_diagnostic_never_claims_path_after_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    publish_parent = tmp_path / "v2-prepublish-parent"
    publish_parent.mkdir(mode=0o700)
    case.destination = publish_parent / "closure"
    retained_parent = tmp_path / "retained-v2-prepublish-parent"
    _install_synthetic_execution(case, monkeypatch)

    def swap_during_production(
        _inputs: object,
        _output_root: object,
        *,
        output_guard: Callable[[], None],
    ) -> object:
        publish_parent.rename(retained_parent)
        publish_parent.mkdir(mode=0o700)
        output_guard()
        pytest.fail("prepublication parent swap was not detected")

    monkeypatch.setattr(closure, "_produce_v2_with_inputs", swap_during_production)
    with pytest.raises(
        closure.DerivationClosureError,
        match=r"candidate_names=.*held_parent_descriptor=.*parent-path-or-name-mapping",
    ) as raised:
        closure.build_derivation_closure_v2(
            case.plan_path,
            case.source_root,
            case.package_roots,
            case.destination,
            **case.kwargs(),
        )
    assert "candidate_path=" not in str(raised.value)
    assert "candidate_paths=" not in str(raised.value)
    retained_stage = retained_parent / ".closure.private-v2-candidate"
    assert retained_stage.exists()


def test_v2_terminal_replay_readback_names_published_replay_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    _install_synthetic_execution(case, monkeypatch)
    receipt = closure.build_derivation_closure_v2(
        case.plan_path,
        case.source_root,
        case.package_roots,
        case.destination,
        **case.kwargs(),
    )
    original_revalidate = closure._revalidate_v2_inputs
    mutated = False

    def mutate_replay_after_publication(inputs: object) -> None:
        nonlocal mutated
        original_revalidate(inputs)
        if case.replay_root.exists() and not mutated:
            target = case.replay_root / "runs/rebuttal/rebuild-b.pdf"
            target.chmod(0o600)
            target.write_bytes(_pdf("rebuttal") + b"mutated")
            target.chmod(0o400)
            mutated = True

    monkeypatch.setattr(
        closure,
        "_revalidate_v2_inputs",
        mutate_replay_after_publication,
    )
    with pytest.raises(
        closure.DerivationClosureError,
        match=f"candidate_path={case.replay_root}",
    ):
        closure.validate_derivation_closure_v2(
            case.plan_path,
            case.source_root,
            case.package_roots,
            case.destination,
            case.replay_root,
            expected_manifest_sha256=receipt.manifest_sha256,
            **case.kwargs(),
        )
    assert mutated
    assert case.replay_root.exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", closure.PRODUCER_AUTHORITY_SCHEMA, "downgrade"),
        ("status", "synthetic-canary-only", "downgrade"),
        ("mode", closure.MODE_SYNTHETIC, "downgrade"),
    ],
)
def test_v2_authority_rejects_v1_or_synthetic_downgrade_before_execution(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    case = _make_case(tmp_path)
    plan = case.plan()
    document = _mapping(_array(plan["documents"])[0])
    pdf_id = str(document["pdf_id"])
    authority_path = case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
    authority = json.loads(authority_path.read_bytes())
    authority[field] = value
    authority.pop("manifest_body_sha256")
    authority["manifest_body_sha256"] = _sha256(_canonical(authority))
    with pytest.raises(closure.DerivationClosureError, match=match):
        closure._normalize_v2_authority(
            authority,
            release_id=RELEASE_ID,
            pdf_id=pdf_id,
            document=document,
        )


def test_v2_authority_rejects_codesign_fat_alignment_or_capability_drift(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    plan = case.plan()
    document = _mapping(_array(plan["documents"])[0])
    pdf_id = str(document["pdf_id"])
    authority_path = case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
    baseline = json.loads(authority_path.read_bytes())
    for field, replacement in (
        ("alignment_exponent", 32),
        ("cpu_subtype_capabilities", 0),
    ):
        authority = copy.deepcopy(baseline)
        toolchain = _mapping(authority["toolchain"])
        codesign = _mapping(toolchain["codesign"])
        binary = _mapping(codesign["binary"])
        arm_slice = _mapping(_array(binary["slices"])[1])
        arm_slice[field] = replacement
        binary["selected_execution_slice"] = arm_slice
        toolchain.pop("toolchain_projection_sha256")
        toolchain["toolchain_projection_sha256"] = _sha256(_canonical(toolchain))
        authority.pop("manifest_body_sha256")
        authority["manifest_body_sha256"] = _sha256(_canonical(authority))
        candidate_document = copy.deepcopy(document)
        candidate_package = _mapping(candidate_document["producer_package"])
        candidate_projection = _mapping(candidate_package["authority_projection"])
        candidate_projection["manifest_body_sha256"] = authority["manifest_body_sha256"]
        with pytest.raises(closure.DerivationClosureError, match="codesign"):
            closure._normalize_v2_authority(
                authority,
                release_id=RELEASE_ID,
                pdf_id=pdf_id,
                document=candidate_document,
            )


def test_v2_toolchain_tree_accepts_native_bounds_and_rejects_each_overflow(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    pdf_id = "clean"
    authority_path = case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
    authority = json.loads(authority_path.read_bytes())
    caller_anchors = _mapping(authority["caller_anchors"])
    baseline = _mapping(authority["toolchain"])

    def validate(tree_values: dict[str, int]) -> None:
        toolchain = json.loads(_canonical(baseline))
        sdk_tree = _mapping(toolchain["sdk_tree"])
        sdk_tree.update(tree_values)
        toolchain.pop("toolchain_projection_sha256", None)
        toolchain["toolchain_projection_sha256"] = _sha256(_canonical(toolchain))
        closure._validate_v2_toolchain_projection(
            toolchain,
            caller_anchors=caller_anchors,
            context="synthetic native-bound toolchain",
        )

    # Exact proof-j shape: large, but within the native 1 GiB/100k-entry caps.
    validate(
        {
            "file_count": 32_345,
            "directory_count": 9_908,
            "symlink_count": 7_448,
            "entry_count": 49_701,
            "total_file_bytes": 765_532_437,
        },
    )
    validate(
        {
            "file_count": closure.MAX_V2_TREE_FILES,
            "directory_count": closure.MAX_V2_TREE_DIRECTORIES,
            "symlink_count": 18_080,
            "entry_count": closure.MAX_V2_TREE_ENTRIES,
            "total_file_bytes": closure.MAX_V2_TREE_BYTES,
        },
    )
    validate(
        {
            "file_count": 1,
            "directory_count": 1,
            "symlink_count": closure.MAX_V2_TREE_SYMLINKS,
            "entry_count": closure.MAX_V2_TREE_SYMLINKS + 2,
            "total_file_bytes": 1,
        },
    )

    invalid = (
        {
            "file_count": closure.MAX_V2_TREE_FILES + 1,
            "directory_count": 1,
            "symlink_count": 0,
            "entry_count": closure.MAX_V2_TREE_FILES + 2,
            "total_file_bytes": 1,
        },
        {
            "file_count": 1,
            "directory_count": closure.MAX_V2_TREE_DIRECTORIES + 1,
            "symlink_count": 0,
            "entry_count": closure.MAX_V2_TREE_DIRECTORIES + 2,
            "total_file_bytes": 1,
        },
        {
            "file_count": 1,
            "directory_count": 1,
            "symlink_count": closure.MAX_V2_TREE_SYMLINKS + 1,
            "entry_count": closure.MAX_V2_TREE_SYMLINKS + 3,
            "total_file_bytes": 1,
        },
        {
            "file_count": closure.MAX_V2_TREE_FILES,
            "directory_count": closure.MAX_V2_TREE_DIRECTORIES,
            "symlink_count": 18_081,
            "entry_count": closure.MAX_V2_TREE_ENTRIES + 1,
            "total_file_bytes": 1,
        },
        {
            "file_count": 1,
            "directory_count": 1,
            "symlink_count": 0,
            "entry_count": 2,
            "total_file_bytes": closure.MAX_V2_TREE_BYTES + 1,
        },
    )
    for tree_values in invalid:
        with pytest.raises(closure.DerivationClosureError, match="tree"):
            validate(tree_values)


def test_v2_authority_rejects_each_coherent_native_capsule_bound_overflow(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    pdf_id = "rebuttal"
    plan = case.plan()
    baseline_document = next(
        _mapping(raw)
        for raw in _array(plan["documents"])
        if _mapping(raw)["pdf_id"] == pdf_id
    )
    authority_path = case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
    baseline = json.loads(authority_path.read_bytes())

    def reject(
        candidate: dict[str, object],
        *,
        match: str,
        sync_source: bool = False,
        sync_producer: bool = False,
        sync_output: bool = False,
    ) -> None:
        _refresh_authority_digests(candidate)
        document = json.loads(_canonical(baseline_document))
        projection = _mapping(
            _mapping(document["producer_package"])["authority_projection"],
        )
        projection["manifest_body_sha256"] = candidate["manifest_body_sha256"]
        projection["bundle_projection_sha256"] = _mapping(
            _mapping(candidate["source_bundle"])["bundle_projection"],
        )["bundle_projection_sha256"]
        projection["launcher_config_sha256"] = _mapping(
            candidate["launcher_config"],
        )["launcher_config_sha256"]
        projection["runtime_handoff_sha256"] = _mapping(
            candidate["runtime_handoff"],
        )["runtime_handoff_sha256"]
        projection["build_projection_sha256"] = _mapping(candidate["build"])[
            "build_projection_sha256"
        ]
        if sync_source:
            _mapping(document["source_bundle"])["bytes"] = _mapping(
                candidate["source_bundle"],
            )["bytes"]
        if sync_producer:
            _mapping(_mapping(document["producer_package"])["producer"])["bytes"] = (
                _mapping(candidate["producer"])["bytes"]
            )
        if sync_output:
            document["expected_output"] = json.loads(
                _canonical(candidate["expected_output"]),
            )
        with pytest.raises(closure.DerivationClosureError, match=match):
            closure._normalize_v2_authority(
                candidate,
                release_id=RELEASE_ID,
                pdf_id=pdf_id,
                document=document,
            )

    candidate = json.loads(_canonical(baseline))
    _mapping(candidate["source_bundle"])["bytes"] = (
        closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES + 1
    )
    reject(candidate, match="source_bundle", sync_source=True)

    candidate = json.loads(_canonical(baseline))
    _mapping(candidate["producer"])["bytes"] = (
        closure.MAX_V2_NATIVE_EXECUTABLE_BYTES + 1
    )
    reject(candidate, match="bytes or Mach-O UUID", sync_producer=True)

    candidate = json.loads(_canonical(baseline))
    launcher_source = _mapping(candidate["launcher_source"])
    launcher_source["bytes"] = closure.MAX_V2_NATIVE_LAUNCHER_SOURCE_BYTES + 1
    reject(candidate, match="launcher_source")

    candidate = json.loads(_canonical(baseline))
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    _mapping(_array(bundle["canonical_inputs"])[0])["bytes"] = (
        closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES + 1
    )
    reject(candidate, match="canonical input exceeds")

    for dependency, maximum in (
        ("runtime", closure.MAX_V2_NATIVE_TOOL_BYTES),
        ("renderer", closure.MAX_V2_NATIVE_BUILDER_BYTES),
        ("machine_runner", closure.MAX_V2_NATIVE_BUILDER_BYTES),
    ):
        candidate = json.loads(_canonical(baseline))
        dependencies = _mapping(
            _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])[
                "dependencies"
            ],
        )
        _mapping(dependencies[dependency])["bytes"] = maximum + 1
        reject(candidate, match="byte bound")

    candidate = json.loads(_canonical(baseline))
    dependencies = _mapping(
        _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])[
            "dependencies"
        ],
    )
    _mapping(_array(dependencies["fonts"])[0])["bytes"] = (
        closure.MAX_V2_NATIVE_FONT_BYTES + 1
    )
    reject(candidate, match="font 0 identity")

    candidate = json.loads(_canonical(baseline))
    dependencies = _mapping(
        _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])[
            "dependencies"
        ],
    )
    _mapping(_array(dependencies["tools"])[0])["bytes"] = (
        closure.MAX_V2_NATIVE_TOOL_BYTES + 1
    )
    reject(candidate, match="tool 0 identity")

    candidate = json.loads(_canonical(baseline))
    dependencies = _mapping(
        _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])[
            "dependencies"
        ],
    )
    reportlab = _mapping(dependencies["reportlab"])
    reportlab["directory_count"] = closure.MAX_V2_NATIVE_REPORTLAB_DIRECTORIES + 1
    reportlab["entry_count"] = int(reportlab["file_count"]) + int(
        reportlab["directory_count"],
    )
    reject(candidate, match="reportlab tree counts")

    candidate = json.loads(_canonical(baseline))
    toolchain = _mapping(candidate["toolchain"])
    _mapping(toolchain["clang"])["bytes"] = closure.MAX_V2_NATIVE_TOOL_BYTES + 1
    reject(candidate, match="executable identity")

    candidate = json.loads(_canonical(baseline))
    source_release = _mapping(candidate["source_release"])
    _mapping(source_release["git"])["bytes"] = closure.MAX_V2_NATIVE_TOOL_BYTES + 1
    reject(candidate, match="Git executable exceeds")

    candidate = json.loads(_canonical(baseline))
    source_release = _mapping(candidate["source_release"])
    _mapping(_array(source_release["members"])[0])["bytes"] = (
        closure.MAX_V2_NATIVE_GIT_BLOB_BYTES + 1
    )
    reject(candidate, match="source_release member exceeds")

    candidate = json.loads(_canonical(baseline))
    expected = _mapping(candidate["expected_output"])
    _mapping(expected["pdf"])["bytes"] = closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES + 1
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    bundle["expected_output"] = json.loads(_canonical(expected))
    reject(candidate, match="expected_output.pdf.bytes exceeds", sync_output=True)


def test_v2_authority_native_language_boundaries_and_forbidden_fragments(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    pdf_id = "rebuttal"
    plan = case.plan()
    baseline_document = next(
        _mapping(raw)
        for raw in _array(plan["documents"])
        if _mapping(raw)["pdf_id"] == pdf_id
    )
    authority_path = case.package_roots[pdf_id] / closure.AUTHORITY_MEMBER_BY_ID[pdf_id]
    baseline = json.loads(authority_path.read_bytes())

    def document_for(
        candidate: dict[str, object],
        *,
        sync_producer: bool = False,
        sync_output: bool = False,
    ) -> dict[str, object]:
        _refresh_authority_digests(candidate)
        document = json.loads(_canonical(baseline_document))
        package = _mapping(document["producer_package"])
        projection = _mapping(package["authority_projection"])
        projection["manifest_body_sha256"] = candidate["manifest_body_sha256"]
        projection["bundle_projection_sha256"] = _mapping(
            _mapping(candidate["source_bundle"])["bundle_projection"],
        )["bundle_projection_sha256"]
        projection["launcher_config_sha256"] = _mapping(
            candidate["launcher_config"],
        )["launcher_config_sha256"]
        projection["runtime_handoff_sha256"] = _mapping(
            candidate["runtime_handoff"],
        )["runtime_handoff_sha256"]
        projection["build_projection_sha256"] = _mapping(candidate["build"])[
            "build_projection_sha256"
        ]
        if sync_producer:
            package["producer"] = json.loads(_canonical(candidate["producer"]))
        if sync_output:
            document["expected_output"] = json.loads(
                _canonical(candidate["expected_output"]),
            )
        return document

    candidate = json.loads(_canonical(baseline))
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    for raw_input in _array(bundle["canonical_inputs"]):
        _mapping(raw_input)["bytes"] = closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
    document = document_for(candidate)
    closure._normalize_v2_authority(
        candidate,
        release_id=RELEASE_ID,
        pdf_id=pdf_id,
        document=document,
    )

    candidate = json.loads(_canonical(baseline))
    source_release = _mapping(candidate["source_release"])
    source_release["release_ref"] = "r" + "a" * 255
    document = document_for(candidate)
    closure._normalize_v2_authority(
        candidate,
        release_id=RELEASE_ID,
        pdf_id=pdf_id,
        document=document,
    )
    source_release["release_ref"] = "r" + "a" * 256
    document = document_for(candidate)
    with pytest.raises(closure.DerivationClosureError, match="revision-bound"):
        closure._normalize_v2_authority(
            candidate,
            release_id=RELEASE_ID,
            pdf_id=pdf_id,
            document=document,
        )

    candidate = json.loads(_canonical(baseline))
    dependency = _mapping(
        _mapping(
            _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])[
                "dependencies"
            ],
        )["runtime"],
    )
    dependency["locator"] = "runtime locator with spaces"
    config_runtime = _mapping(_mapping(candidate["launcher_config"])["runtime"])
    config_runtime["locator"] = dependency["locator"]
    handoff_runtime = _mapping(
        _mapping(_mapping(candidate["runtime_handoff"])["placeholder_bindings"])[
            "{runtime}"
        ],
    )
    handoff_runtime["locator"] = dependency["locator"]
    document = document_for(candidate)
    closure._normalize_v2_authority(
        candidate,
        release_id=RELEASE_ID,
        pdf_id=pdf_id,
        document=document,
    )

    candidate = json.loads(_canonical(baseline))
    producer = _mapping(candidate["producer"])
    producer_cd = _mapping(producer["native_code_directory"])
    producer_cd["code_directory_bytes"] = int(producer["bytes"]) + 1
    for raw_build in _array(_mapping(candidate["build"])["builds"]):
        _mapping(raw_build)["native_code_directory"] = json.loads(
            _canonical(producer_cd),
        )
    document = document_for(candidate, sync_producer=True)
    closure._normalize_v2_authority(
        candidate,
        release_id=RELEASE_ID,
        pdf_id=pdf_id,
        document=document,
    )

    candidate = json.loads(_canonical(baseline))
    expected = _mapping(candidate["expected_output"])
    _mapping(expected["renderer_manifest"])["bytes"] = (
        closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
    )
    _mapping(expected["pdf"])["bytes"] = closure.MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
    bundle = _mapping(_mapping(candidate["source_bundle"])["bundle_projection"])
    bundle["expected_output"] = json.loads(_canonical(expected))
    document = document_for(candidate, sync_output=True)
    closure._normalize_v2_authority(
        candidate,
        release_id=RELEASE_ID,
        pdf_id=pdf_id,
        document=document,
    )

    assert closure.V2_FORBIDDEN_CAPSULE_FRAGMENTS == (
        b"/Users/",
        b"/private/",
        b"/tmp/",
        b".codex",
        b".cache",
        b"research/",
        b"output/",
    )
    for fragment in closure.V2_FORBIDDEN_CAPSULE_FRAGMENTS:
        with pytest.raises(closure.DerivationClosureError, match="forbidden"):
            closure._reject_v2_forbidden_fragments(
                b"prefix" + fragment + b"suffix",
                context="synthetic capsule",
            )
    closure._reject_v2_forbidden_fragments(
        b"/var/folders/role-generic-host-path-is-not-a-native-fragment",
        context="synthetic capsule",
    )


def test_v2_rejects_split_or_unsealed_package_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    calls = _install_synthetic_execution(case, monkeypatch)
    case.package_roots["clean"].chmod(0o700)
    with pytest.raises(closure.DerivationClosureError, match="sealed mode 0500"):
        closure.build_derivation_closure_v2(
            case.plan_path,
            case.source_root,
            case.package_roots,
            case.destination,
            **case.kwargs(),
        )
    assert calls == []


def test_v2_role_anchor_crosswire_and_package_root_map_fail_closed(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    anchors = dict(case.source_sha256)
    anchors["marked"] = anchors["clean"]
    with pytest.raises(closure.DerivationClosureError, match="role-distinct"):
        closure._normalize_v2_plan(
            case.plan(),
            release_id=RELEASE_ID,
            expected_upstream_sha256=case.upstream_sha256,
            expected_authority_sha256=case.authority_sha256,
            expected_source_sha256=anchors,
            expected_producer_sha256=case.producer_sha256,
            expected_package_sha256=case.package_sha256,
            expected_renderer_manifest_sha256=case.renderer_manifest_sha256,
            expected_pdf_sha256=case.pdf_sha256,
        )
    roots = dict(case.package_roots)
    roots.pop("s1")
    with pytest.raises(closure.DerivationClosureError, match="name exactly"):
        closure._validate_v2_path_topology(
            case.plan_path,
            case.source_root,
            roots,
            case.destination,
        )


def test_v2_cli_surfaces_are_explicit_and_v1_has_no_schema_sniff() -> None:
    help_text = closure._parser().format_help()
    assert "build-v2" in help_text
    assert "validate-v2" in help_text
    assert closure.DERIVATION_PLAN_SCHEMA != closure.DERIVATION_PLAN_SCHEMA_V2
    assert closure.DERIVATION_CLOSURE_SCHEMA != closure.DERIVATION_CLOSURE_SCHEMA_V2

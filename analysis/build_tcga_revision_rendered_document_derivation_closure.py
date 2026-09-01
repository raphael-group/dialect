"""Build and replay a native, result-blind PDF-derivation closure.

This boundary accepts four independently SHA-256-anchored opaque source bundles
and four independently anchored native adapter executables.  Each adapter receives
only descriptor numbers through a fixed argument contract, runs twice in a fixed
environment, and must produce byte-identical bounded PDFs matching independent
caller authority anchors.  The exact evidence tree is sealed and published with an
atomic no-replace rename; validation retains a separately produced replay.

The repository's historical builder-release observations are not producer
authority.  ``revision`` mode becomes executable only when every role supplies a
descriptor-pinned, independently caller-anchored canonical producer/toolchain
authority receipt, including the rebuttal role.  ``synthetic-canary`` mode proves
only the closure machinery with non-scientific fixtures and cannot clear the
production promotion blocker.

No source-bundle payload is parsed here.  In particular, this module does not open
manuscript prose, scientific result rows, or a real PDF.  It also does not infer
scientific correctness, visual quality, reviewer identity, coauthor approval,
journal acceptance, or upload/readback status.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib
import json
import os
import re
import resource
import stat
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_machine = importlib.import_module(
    "analysis.build_tcga_revision_rendered_document_machine_closure",
)

# This is an intentionally narrow executable and publication contract.  Private
# primitives from the independently tested machine closure are reused and pinned as
# part of every receipt rather than copied into a second security implementation.
# ruff: noqa: PLR0913, SLF001

DERIVATION_PLAN_SCHEMA: Final = (
    "dialect-revision-rendered-document-derivation-closure-input-v1"
)
INVOCATION_RECEIPT_SCHEMA: Final = (
    "dialect-revision-rendered-document-closure-invocation-receipt-v1"
)
PRODUCER_AUTHORITY_SCHEMA: Final = (
    "dialect-revision-rendered-document-producer-toolchain-authority-v1"
)
DERIVATION_CLOSURE_SCHEMA: Final = (
    "dialect-revision-rendered-document-derivation-closure-v1"
)
DERIVATION_CLOSURE_CONTRACT: Final = "four-pdf-pinned-native-adapter-double-replay-v1"
PRODUCER_PROTOCOL: Final = "dialect-pdf-derivation-fd-protocol-v1"
PRODUCER_AUTHORITY_REVIEW_SCOPE: Final = (
    "native-producer-source-and-declared-toolchain-profile-v1"
)

# V1 remains intentionally callable for historical replay, but it can never be
# promoted.  V2 is an explicit API/CLI surface: no schema sniffing or implicit
# downgrade is permitted at either boundary.
DERIVATION_PLAN_SCHEMA_V2: Final = (
    "dialect-revision-rendered-document-derivation-closure-input-v2"
)
INVOCATION_RECEIPT_SCHEMA_V2: Final = (
    "dialect-revision-rendered-document-closure-invocation-receipt-v2"
)
DERIVATION_CLOSURE_SCHEMA_V2: Final = (
    "dialect-revision-rendered-document-derivation-closure-v2"
)
DERIVATION_CLOSURE_CONTRACT_V2: Final = (
    "four-pdf-sealed-native-producer-package-double-replay-v2"
)
NATIVE_PRODUCER_AUTHORITY_CONTRACT_V2: Final = (
    "host-bound-thin-arm64-double-build-adhoc-codesign-two-member-package-v2"
)
NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID: Final = {
    pdf_id: f"dialect-revision-{pdf_id}-native-producer-authority-v2"
    for pdf_id in ("clean", "marked", "s1", "rebuttal")
}
V2_AUTHORITY_STATUS: Final = "producer-candidate-requires-external-anchor"
V2_AUTHORIZATION_STATUS: Final = "caller-authorized"
V2_CLOSURE_STATUS: Final = "native-revision-v2-candidate-not-promotable"
V2_SOURCE_DESCRIPTOR: Final = 3
V2_PACKAGE_ROOT_MODE: Final = "0500"
V2_PRODUCER_MODE: Final = "0500"
V2_AUTHORITY_MODE: Final = "0400"
V2_SOURCE_MODE: Final = "0400"
V2_CALLER_ANCHOR_KEYS: Final = (
    "source_bundle_sha256",
    "launcher_source_sha256",
    "builder_sha256",
    "bundle_builder_sha256",
    "runtime_sha256",
    "renderer_sha256",
    "machine_runner_sha256",
    "clang_sha256",
    "linker_sha256",
    "codesign_sha256",
    "git_sha256",
    "compiler_resource_tree_sha256",
    "sdk_tree_sha256",
    "renderer_manifest_sha256",
    "pdf_sha256",
)
V2_AUTHORITY_PROJECTION_KEYS: Final = (
    "manifest_body_sha256",
    "bundle_projection_sha256",
    "launcher_config_sha256",
    "runtime_handoff_sha256",
    "build_projection_sha256",
)
MANIFEST_MEMBER: Final = "derivation-manifest.json"
BUILDER_MEMBER: Final = (
    "analysis/build_tcga_revision_rendered_document_derivation_closure.py"
)
MACHINE_RUNNER_MEMBER: Final = (
    "analysis/build_tcga_revision_rendered_document_machine_closure.py"
)

MODE_SYNTHETIC: Final = "synthetic-canary"
MODE_REVISION: Final = "revision"
MODES: Final = (MODE_SYNTHETIC, MODE_REVISION)

PDF_ORDER: Final = (
    ("clean", "manuscript-clean.pdf"),
    ("marked", "manuscript-marked.pdf"),
    ("s1", "s1-appendix.pdf"),
    ("rebuttal", "response-to-reviewers.pdf"),
)
PDF_IDS: Final = tuple(pdf_id for pdf_id, _ in PDF_ORDER)
PDF_MEMBER_BY_ID: Final = dict(PDF_ORDER)
SOURCE_MEMBER_BY_ID: Final = {pdf_id: f"{pdf_id}-source.bundle" for pdf_id in PDF_IDS}
PRODUCER_MEMBER_BY_ID: Final = {pdf_id: f"derive-{pdf_id}" for pdf_id in PDF_IDS}
AUTHORITY_MEMBER_BY_ID: Final = {
    pdf_id: f"{pdf_id}-producer-toolchain-authority.json" for pdf_id in PDF_IDS
}
BUILDER_RELEASE_PROFILE_BY_ID: Final = {
    "clean": "latexpand-pdflatex-bibtex-inlined-double-build",
    "marked": ("latexpand-latexdiff-latexrevise-pdflatex-bibtex-inlined-double-build"),
    "s1": "latexmk-pdf-pdflatex",
    "rebuttal": None,
}
UPSTREAM_BINDING_KEYS: Final = (
    "artifact_registry_sha256",
    "derivation_evidence_sha256",
    "document_anchor_sha256",
    "document_reconciliation_sha256",
    "release_evidence_sha256",
    "rendered_document_input_sha256",
    "source_snapshot_anchor_sha256",
)

EXACT_ENVIRONMENT: Final = dict(_machine.EXACT_ENVIRONMENT)
ARGUMENT_TEMPLATE: Final = [
    "--dialect-derivation-protocol",
    PRODUCER_PROTOCOL,
    "--pdf-id",
    "{pdf_id}",
    "--source-fd",
    "{source_fd}",
    "--pdf-output",
    "stdout",
]
EXECUTION_CONTRACT: Final = {
    "shell": False,
    "cwd": "/",
    "inherit_environment": False,
    "environment": EXACT_ENVIRONMENT,
    "argument_template": ARGUMENT_TEMPLATE,
    "source_input": "caller-anchored-read-only-descriptor",
    "source_consumption": "bound-at-spawn-but-causal-read-not-observable",
    "pdf_output": "continuously-bounded-captured-stdout",
    "stdin": "/dev/null",
    "stdout": "bounded-pdf-byte-stream",
    "stderr": "bounded-and-required-empty",
    "invocation_receipt": (
        "closure-generated-exact-input-authority-output-and-attestation-binding"
    ),
    "invocations_per_document": 2,
    "execution_binding_scope": "main_executable",
    "non_system_dylib_closure": "not_attested",
    "descendant_process_and_tool_closure": "adapter-trust-boundary-not-attested",
    "producer_filesystem_side_effects": "ambient-same-uid-access-not-contained",
    "loaded_python_code_binding": (
        "path bytes pinned but already loaded interpreter code identity not attested"
    ),
    "pdf_structural_validity": "not-inferred-deferred-to-machine-closure",
    "pdf_visual_validity": "not-inferred-deferred-to-visual-qa-authority",
}

EXECUTION_CONTRACT_V2: Final = {
    **EXECUTION_CONTRACT,
    "source_input": "opaque-caller-anchored-mode-0400-single-link-descriptor",
    "source_child_descriptor": V2_SOURCE_DESCRIPTOR,
    "source_descriptor_mapping": "fixed-parent-descriptor-to-child-fd-3-at-spawn",
    "producer_package": "exact-sealed-two-member-root",
    "producer_package_root_mode": V2_PACKAGE_ROOT_MODE,
    "producer_member_mode": V2_PRODUCER_MODE,
    "authority_member_mode": V2_AUTHORITY_MODE,
    "producer_authority": (
        "candidate-capsule-requires-independent-full-capsule-sha256-anchor"
    ),
    "main_executable_attestation_publication": (
        "path-free-vnode-code-directory-and-cdhash-projection"
    ),
    "same_process_group_descendants": (
        "terminal-sigkill-before-main-reap-with-wnowait-held-leader"
    ),
    "detached_setsid_descendants": "not-contained",
    "replay_publication": "compare-private-candidate-before-atomic-publication",
}

# These observations are derived from the repository-owned submission sources, not
# from any manuscript/result byte.  They remain fixed in the anchored plan so the
# generic protocol cannot be presented as proof that the current real toolchain is
# complete.
BUILDER_RELEASE_TOOLCHAIN_BASELINE: Final = {
    "authority": "historical-non-authoritative-builder-release-observation",
    "observations": {
        "clean": {
            "entrypoint": "research/paper/submission/scripts/plos_submission.py",
            "profile": BUILDER_RELEASE_PROFILE_BY_ID["clean"],
        },
        "marked": {
            "entrypoint": "research/paper/submission/scripts/plos_submission.py",
            "profile": BUILDER_RELEASE_PROFILE_BY_ID["marked"],
        },
        "s1": {
            "entrypoint": "research/paper/S1_Appendix.tex",
            "profile": BUILDER_RELEASE_PROFILE_BY_ID["s1"],
        },
        "rebuttal": {
            "entrypoint": None,
            "profile": BUILDER_RELEASE_PROFILE_BY_ID["rebuttal"],
        },
    },
}

NON_INFERENCE_LIMITS: Final = {
    "adapter_source_review": "not inferred from executable SHA-256",
    "dedicated_rebuttal_renderer_compatibility": (
        "not-established; requires a reviewed fixed-protocol adapter and authority"
    ),
    "child_tool_and_dylib_closure": "not attested",
    "producer_filesystem_side_effects": "not contained by cwd, env, or FD pinning",
    "loaded_python_code_identity": (
        "live file SHA pinning does not cryptographically bind already loaded code"
    ),
    "scientific_correctness": "not inferred",
    "source_bundle_semantics": "opaque and not parsed",
    "source_bundle_causal_consumption": "not observable at this boundary",
    "pdf_structural_validity": "not inferred; deferred to machine closure",
    "text_legibility_or_visual_quality": "not inferred",
    "human_identity_or_approval": "not authenticated",
    "coauthor_or_submission_approval": "not inferred",
    "journal_upload_or_readback": "not inferred",
}

NON_INFERENCE_LIMITS_V2: Final = {
    **NON_INFERENCE_LIMITS,
    "authority_capsule": (
        "candidate-only; authorization comes from the caller's full-capsule SHA-256"
    ),
    "ad_hoc_signature_signer_identity": "not authenticated",
    "two_build_general_reproducibility": "not inferred from two current-host builds",
    "post_exec_runtime_mapping": "not attested",
    "source_bundle_payload": "opaque; JSON and base64 payloads are not decoded here",
    "source_bundle_causal_read": "not observed",
    "detached_setsid_descendants": "not contained",
    "process_group_or_session_migrated_descendants": "not contained",
    "public_host_paths": "omitted from public receipts and manifest",
    "same_uid_post_release_mutation": (
        "not excluded between discrete terminal checks and not prevented after "
        "descriptor release; POSIX owner modes are not immutable"
    ),
}

NATIVE_PRODUCER_NON_INFERENCE_LIMITS_V2: Final = {
    "acl_xattr_and_file_flag_topology": "not-recorded-or-attested",
    "ad_hoc_codesign_signer_identity": "not-authenticated",
    "builder_loaded_code_and_invoking_runtime_identity": "not-attested",
    "codesign_executed_slice_attestation": (
        "universal-file-and-arm64e-slice-bytes-parsed; live-mapping-not-attested"
    ),
    "code_signature_validation_scope": (
        "primary-codedirectory-code-slots-and-identity-independently-parsed;"
        "special-slot-semantics-rely-on-bounded-codesign-verify"
    ),
    "compiler_linker_codesign_child_or_dylib_closure": "not-attested",
    "compiler_resource_and_sdk_causal_reads": (
        "declared-tree-byte-projections-anchored-but-causal-member-use-not-observed"
    ),
    "compiler_sdk_codesign_correctness": "not-established-by-byte-anchors",
    "determinism_scope": (
        "two-byte-identical-builds-observed-on-this-host-with-the-declared-"
        "main-file-and-tree-anchors"
    ),
    "detached_setsid_descendant_containment": "not-provided",
    "process_group_or_session_migrated_descendant_containment": "not-provided",
    "synthetic_source_path_scope": (
        "caller-pinned-file-bytes-with-logical-member-labels;"
        "repository-path-binding-not-provided"
    ),
    "git_commit_signer_or_authorship": "not-authenticated",
    "git_release_scope": (
        "git-command-observed-listed-path-bytes-only;object-format-blob-oid-"
        "object-type-repo-tree-index-worktree-cleanliness-tag-immutability-and-"
        "results-authority-not-attested"
    ),
    "git_child_or_dylib_closure": "not-attested",
    "hardlink_equivalence_groups_and_links_outside_root": "not-recorded",
    "host_private_paths": (
        "runtime-and-renderer-paths-embedded-in-private-host-bound-launcher"
    ),
    "host_portability": "private-current-host-package;not-portable",
    "loaded_python_code_identity": (
        "pre-exec-path-bytes-checked;post-exec-interpreter-and-loaded-code-not-attested"
    ),
    "inherited_process_state": (
        "umask-rlimits-signal-mask-and-ignored-signal-dispositions-not-normalized"
    ),
    "main_launcher_process_attestation": (
        "deferred-to-derivation-closure-suspended-process-check"
    ),
    "producer_authority": "capsule-is-a-candidate-requiring-external-sha-anchor",
    "producer_execution_during_build": (
        "launcher-handoff-and-expected-pdf-derivation-not-executed-by-package-build"
    ),
    "producer_filesystem_side_effects": "ambient-same-uid-access-not-contained",
    "receipt_path_fields": "operational-host-private-metadata;not-for-public-manifests",
    "destination_path_persistence_after_pin_release": "not-guaranteed",
    "os_kernel_and_dynamic_dependency_closure": (
        "xcode-usr-lib-dyld-shared-cache-system-frameworks-and-kernel-not-anchored"
    ),
    "renderer_causal_load": "not-observable-after-path-based-execve",
    "scientific_correctness": "not-inferred",
    "source_bundle_prose": "not-copied-into-authority-projection",
    "source_bundle_causal_consumption": "not-inferred-by-package-build",
    "static_code_directory_flags_to_runtime_csops_policy": (
        "actual-suspended-process-status-attestation-required"
    ),
    "text_legibility_or_visual_quality": "not-inferred",
    "toolchain_reproducibility": "cross-host-and-general-reproducibility-not-proven",
    "coauthor_or_submission_approval": "not-inferred",
    "journal_upload_acceptance_or_readback": "not-inferred",
}

V2_PROMOTION_BLOCKERS: Final = [
    "downstream-promotion-closure-not-yet-authorized-for-v2",
]

WRAPPER_INTEGRATION_BASE: Final = {
    "current_wrapper_schema": (
        "dialect-revision-rendered-document-derivation-evidence-v1"
    ),
    "deliberate_mismatch": (
        "the current wrapper expects candidate, round-trip, and external-QA "
        "reference fields and does not yet accept this native closure manifest"
    ),
    "production_blocker": "native-derivation-producer-closure-not-validated",
}


def _wrapper_integration(mode: str) -> dict[str, object]:
    """Describe the immutable wrapper boundary without implying promotion."""
    return {
        **WRAPPER_INTEGRATION_BASE,
        "status": "not-integrated-requires-downstream-promotion-closure",
        "native_derivation_candidate": mode == MODE_REVISION,
        "synthetic_canary_clears_blocker": False,
    }


def _mode_status(mode: str) -> tuple[str, list[str]]:
    if mode == MODE_SYNTHETIC:
        return (
            "synthetic-canary-only",
            [
                "real-producer-toolchain-authority-not-exercised",
                "current-wrapper-does-not-consume-native-derivation-closure",
                "synthetic-source-bundles-not-revision-sources",
            ],
        )
    return (
        "native-revision-derivation-candidate",
        ["requires-separate-downstream-promotion-closure"],
    )


_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_PDF_SIGNATURE: Final = b"%PDF-"
MAX_PLAN_BYTES: Final = 2 * 1024 * 1024
MAX_AUTHORITY_RECEIPT_BYTES: Final = 256 * 1024
MAX_TOTAL_AUTHORITY_BYTES: Final = len(PDF_ORDER) * MAX_AUTHORITY_RECEIPT_BYTES
MAX_V2_AUTHORITY_BYTES: Final = 2 * 1024 * 1024
MAX_TOTAL_V2_AUTHORITY_BYTES: Final = len(PDF_ORDER) * MAX_V2_AUTHORITY_BYTES
MAX_SOURCE_BUNDLE_BYTES: Final = 256 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES: Final = 1024 * 1024 * 1024
MAX_PRODUCER_BYTES: Final = 32 * 1024 * 1024
MAX_TOTAL_PRODUCER_BYTES: Final = 128 * 1024 * 1024
# Exact native-capsule bounds.  The broader limits above remain the historical
# v1/aggregate envelope; v2 must accept only records the native launcher and
# package validator can actually consume.
MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES: Final = 8 * 1024 * 1024
MAX_V2_NATIVE_OUTPUT_BYTES: Final = 8 * 1024 * 1024
MAX_V2_NATIVE_LAUNCHER_SOURCE_BYTES: Final = 512 * 1024
MAX_V2_NATIVE_BUILDER_BYTES: Final = 2 * 1024 * 1024
MAX_V2_NATIVE_EXECUTABLE_BYTES: Final = 8 * 1024 * 1024
MAX_V2_NATIVE_TOOL_BYTES: Final = 256 * 1024 * 1024
MAX_V2_NATIVE_TOOL_OUTPUT_BYTES: Final = 512 * 1024
MAX_V2_NATIVE_GIT_BLOB_BYTES: Final = 4 * 1024 * 1024
MAX_V2_NATIVE_FONT_BYTES: Final = 32 * 1024 * 1024
MAX_V2_NATIVE_REPORTLAB_FILES: Final = 512
MAX_V2_NATIVE_REPORTLAB_DIRECTORIES: Final = 64
MAX_V2_NATIVE_REPORTLAB_ENTRIES: Final = 576
MAX_V2_NATIVE_REPORTLAB_BYTES: Final = 32 * 1024 * 1024
MAX_V2_NATIVE_MACH_LOAD_COMMANDS: Final = 512
MAX_V2_NATIVE_FAT_SLICES: Final = 8
MAX_V2_TREE_FILES: Final = 65_536
MAX_V2_TREE_DIRECTORIES: Final = 16_384
MAX_V2_TREE_SYMLINKS: Final = 65_536
MAX_V2_TREE_ENTRIES: Final = 100_000
MAX_V2_TREE_BYTES: Final = 1024 * 1024 * 1024
V2_BUNDLE_CONTRACT: Final = "canonical-inputs-pinned-renderer-fresh-pdf-derivation-v1"
V2_BUNDLE_INPUT_MEMBERS: Final = (
    "source.canonical.md",
    "template.canonical.json",
    "config.canonical.json",
)
V2_BUNDLE_NON_INFERENCE: Final = {
    "adapter_source_review": "not-inferred",
    "ambient_same_uid_filesystem_containment": "not-provided",
    "child_tool_and_dylib_closure": "not-attested",
    "coauthor_or_submission_approval": "not-inferred",
    "decoded_canonical_input_private_paths": (
        "rejected-as-utf8-text-without-recursive-decoding"
    ),
    "human_visual_approval": "required-separately",
    "journal_acceptance_or_upload": "not-inferred",
    "loaded_python_code_identity": "path-bytes-pinned-after-bootstrap-only",
    "native_adapter_authority": "not-provided-by-this-bundle",
    "pre_rendered_pdf_member": "absent",
    "producer_pdf_input": "none",
    "recursive_content_classification": "not-provided",
    "scientific_accuracy": "not-inferred",
}
V2_COMPILER_RESOURCE_ROOT: Final = (
    "/Applications/Xcode.app/Contents/Developer/Toolchains/"
    "XcodeDefault.xctoolchain/usr/lib/clang/21"
)
V2_SDK_ROOT: Final = (
    "/Applications/Xcode.app/Contents/Developer/Platforms/"
    "MacOSX.platform/Developer/SDKs/MacOSX.sdk"
)
V2_FORBIDDEN_CAPSULE_FRAGMENTS: Final = (
    b"/Users/",
    b"/private/",
    b"/tmp/",
    b".codex",
    b".cache",
    b"research/",
    b"output/",
)
MAX_PDF_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_PDF_BYTES: Final = 1024 * 1024 * 1024
MAX_INVOCATION_RECEIPT_BYTES: Final = 64 * 1024
MAX_STDERR_BYTES: Final = 256 * 1024
MAX_ADAPTER_INVOCATIONS: Final = len(PDF_ORDER) * 2
PRODUCER_TIMEOUT_SECONDS: Final = 900.0
MAX_OUTPUT_FILES: Final = 1 + len(PDF_ORDER) * 4
MAX_OUTPUT_DIRECTORIES: Final = 1 + len(PDF_ORDER)
MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024


class DerivationClosureError(ValueError):
    """Raised when native derivation production or replay is invalid."""


@dataclass(frozen=True, slots=True)
class DerivationClosureReceipt:
    """Summarize one published closure or independent retained replay."""

    manifest_path: str
    manifest_sha256: str
    plan_sha256: str
    source_bundle_set_sha256: str
    producer_set_sha256: str
    producer_toolchain_authority_set_sha256: str
    pdf_set_sha256: str
    pdf_count: int
    rebuild_count: int
    mode: str
    promotable: bool
    replay_root: str | None


@dataclass(frozen=True, slots=True)
class DerivationClosureReceiptV2:
    """Summarize one revision-strict package-backed closure or replay."""

    manifest_path: str
    manifest_sha256: str
    plan_sha256: str
    source_bundle_set_sha256: str
    producer_package_set_sha256: str
    producer_set_sha256: str
    producer_toolchain_authority_set_sha256: str
    pdf_set_sha256: str
    pdf_count: int
    rebuild_count: int
    mode: str
    promotable: bool
    replay_root: str | None


@dataclass(slots=True)
class _Inputs:
    plan: object
    builder: object
    machine_runner: object
    source_root: object
    producer_root: object
    authority_root: object
    sources: dict[str, object]
    producers: dict[str, object]
    authorities: dict[str, object]
    normalized_plan: dict[str, object]

    def close(self) -> None:
        """Close every descriptor owned by the pinned input set."""
        _close_resources(
            [
                *[
                    (f"producer authority {pdf_id}", self.authorities[pdf_id])
                    for pdf_id in PDF_IDS
                ],
                *[(f"producer {pdf_id}", self.producers[pdf_id]) for pdf_id in PDF_IDS],
                *[
                    (f"source bundle {pdf_id}", self.sources[pdf_id])
                    for pdf_id in PDF_IDS
                ],
                ("producer authority root", self.authority_root),
                ("producer root", self.producer_root),
                ("source-bundle root", self.source_root),
                ("native execution dependency", self.machine_runner),
                ("derivation builder", self.builder),
                ("derivation plan", self.plan),
            ],
            context="pinned derivation inputs",
        )


@dataclass(slots=True)
class _V2Inputs:
    """Own a revision-strict plan, four opaque sources, and four packages."""

    plan: object
    builder: object
    machine_runner: object
    source_root: object
    package_roots: dict[str, object]
    sources: dict[str, object]
    producers: dict[str, object]
    authorities: dict[str, object]
    normalized_authorities: dict[str, dict[str, object]]
    normalized_plan: dict[str, object]
    package_records: dict[str, dict[str, object]]

    def close(self) -> None:
        """Close all package members before their containing roots."""
        _close_resources(
            [
                *[
                    (f"v2 authority {pdf_id}", self.authorities[pdf_id])
                    for pdf_id in PDF_IDS
                ],
                *[
                    (f"v2 producer {pdf_id}", self.producers[pdf_id])
                    for pdf_id in PDF_IDS
                ],
                *[
                    (f"v2 source bundle {pdf_id}", self.sources[pdf_id])
                    for pdf_id in PDF_IDS
                ],
                *[
                    (f"v2 package root {pdf_id}", self.package_roots[pdf_id])
                    for pdf_id in PDF_IDS
                ],
                ("v2 source-bundle root", self.source_root),
                ("v2 native execution dependency", self.machine_runner),
                ("v2 derivation builder", self.builder),
                ("v2 derivation plan", self.plan),
            ],
            context="pinned v2 derivation inputs",
        )


@dataclass(slots=True)
class _PinnedV2Closure:
    root: object
    members: dict[str, object]
    manifest: dict[str, object]
    manifest_raw: bytes

    def close(self) -> None:
        _close_resources(
            [
                *[
                    (f"anchored v2 closure member {member}", pinned)
                    for member, pinned in self.members.items()
                ],
                ("anchored v2 derivation closure", self.root),
            ],
            context="anchored v2 derivation closure",
        )


@dataclass(slots=True)
class _Production:
    manifest: dict[str, object]
    manifest_raw: bytes
    member_inventory: list[dict[str, object]]


@dataclass(slots=True)
class _InvocationBudget:
    """Bound top-level native adapters to the exact eight-invocation contract."""

    count: int = 0

    def consume(self) -> None:
        self.count += 1
        if self.count > MAX_ADAPTER_INVOCATIONS:
            _fail(
                "adapter invocation count exceeds the "
                f"{MAX_ADAPTER_INVOCATIONS}-invocation limit",
            )


def _fail(message: str) -> NoReturn:
    raise DerivationClosureError(message)


def _close_resources(
    resources: Sequence[tuple[str, object | None]],
    *,
    context: str,
) -> None:
    """Attempt every owned close and preserve primary plus cleanup diagnostics."""
    primary = sys.exception()
    cleanup_errors: list[tuple[str, BaseException]] = []
    for label, resource_object in resources:
        if resource_object is None:
            continue
        try:
            resource_object.close()
        except BaseException as error:  # noqa: BLE001 - cleanup must be exhaustive.
            cleanup_errors.append((label, error))
    if cleanup_errors:
        detail = "; ".join(f"{label}: {error}" for label, error in cleanup_errors)
        if primary is not None:
            message = f"{primary}; {context} cleanup also failed: {detail}"
            raise DerivationClosureError(message) from primary
        message = f"{context} cleanup failed: {detail}"
        raise DerivationClosureError(message) from cleanup_errors[0][1]


def _machine_call(
    operation: Callable[..., object],
    /,
    *args: object,
    **kwargs: object,
) -> object:
    try:
        return operation(*args, **kwargs)
    except _machine.MachineClosureError as error:
        raise DerivationClosureError(str(error)) from error


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
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        _fail(f"{context} must be an array")
    return value


def _expect_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        _fail(
            f"{context} keys differ; expected {sorted(expected)}, "
            f"found {sorted(value)}",
        )


def _expect_token(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase canonical token")
    return value


def _expect_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _expect_positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{context} must be a positive integer")
    return value


def _expect_nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{context} must be a non-negative integer")
    return value


_V2_NUMERIC_FIELDS: Final = {
    "alignment_exponent",
    "argument_count_including_argv0",
    "child_descriptor",
    "code_directory_flags",
    "code_limit",
    "code_signing_status",
    "code_slots",
    "cpu_subtype",
    "cpu_subtype_capabilities",
    "cpu_type",
    "device",
    "file_offset",
    "index",
    "inode",
    "invocations_per_document",
    "link_count",
    "maximum",
    "minimum",
    "offset",
    "page_size",
    "protection",
    "return_code",
    "signature_offset",
    "source_child_descriptor",
    "spawn_flags",
    "suspended_wait_status",
    "uid",
}
_V2_BOOLEAN_FIELDS: Final = {
    "PATH_lookup",
    "absolute_path_recorded",
    "canonical_decimal",
    "distinct_stage_roots_and_output_inodes",
    "environment_inherited",
    "git_blob_equality",
    "inherit_environment",
    "main-executable-bytes-pinned",
    "parent_descriptor_recorded",
    "path_recorded",
    "pre_exec_descriptor_hash",
    "promotable",
    "root_path_recorded",
    "seekable_and_rewound",
    "shell",
    "signer_identity_authenticated",
    "source_or_base64_payload_recorded",
    "v1_accepted_for_promotion",
}


def _validate_v2_json_scalar_types(
    value: object,
    *,
    context: str,
    field: str | None = None,
    parent: str | None = None,
) -> None:
    """Reject JSON bool/int aliasing and floating-point numeric leaves."""
    numeric_field = field is not None and (
        field in _V2_NUMERIC_FIELDS
        or field == "bytes"
        or field.endswith(("_bytes", "_count"))
    )
    if field == "code_directory_flags" and parent == "ad_hoc_signature":
        numeric_field = False
    if field == "cpu_subtype" and parent == "native_code_directory":
        numeric_field = False
    boolean_field = field in _V2_BOOLEAN_FIELDS or parent == "byte_identity"
    if numeric_field and (isinstance(value, bool) or not isinstance(value, int)):
        _fail(f"{context} must be an integer, not a JSON boolean or other scalar")
    if boolean_field and type(value) is not bool:
        _fail(f"{context} must be a JSON boolean")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(f"{context} contains a non-string JSON object key")
            _validate_v2_json_scalar_types(
                child,
                context=f"{context}.{key}",
                field=key,
                parent=field,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_v2_json_scalar_types(
                child,
                context=f"{context}[{index}]",
                field=field,
                parent=parent,
            )
        return
    if isinstance(value, float):
        _fail(f"{context} must not contain a non-integral JSON number")
    if not isinstance(value, (str, int, bool, type(None))):
        _fail(f"{context} contains an unsupported JSON scalar")


def _json_exact_equal(left: object, right: object) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _canonical_copy(value: object) -> object:
    """Return an alias-free JSON copy of one already-normalized projection."""
    return json.loads(_canonical_json(value))


def _reject_v2_forbidden_fragments(raw: bytes, *, context: str) -> None:
    if any(fragment in raw for fragment in V2_FORBIDDEN_CAPSULE_FRAGMENTS):
        _fail(f"{context} exposes a forbidden private or work-product fragment")


def _expected_arguments(pdf_id: str) -> list[str]:
    return [token.replace("{pdf_id}", pdf_id) for token in ARGUMENT_TEMPLATE]


def _v2_signature_identifier(pdf_id: str) -> str:
    return f"org.raphaelgroup.dialect.{pdf_id}-derivation-launcher"


def _v2_build_recipes(pdf_id: str) -> dict[str, object]:
    compile_recipe = [
        "{clang}",
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
        "-fno-ident",
        "-fno-common",
        "-fvisibility=hidden",
        "-g0",
        "-isysroot",
        "{sdk_root}",
        "-resource-dir",
        "{compiler_resource_root}",
        "-ffile-prefix-map={stage_root}=/dialect/native-producer",
        "-fdebug-prefix-map={stage_root}=/dialect/native-producer",
        "-x",
        "c",
        "{launcher_source}",
        "-c",
        "-o",
        "{object}",
        '-DDIALECT_RUNTIME_PATH="{runtime}"',
        '-DDIALECT_RUNTIME_SHA256="{runtime_sha256}"',
        "-DDIALECT_RUNTIME_BYTES={runtime_bytes}",
        "-DDIALECT_RUNTIME_MODE={runtime_mode}",
        '-DDIALECT_RENDERER_PATH="{renderer}"',
        '-DDIALECT_RENDERER_SHA256="{renderer_sha256}"',
        "-DDIALECT_RENDERER_BYTES={renderer_bytes}",
        "-DDIALECT_RENDERER_MODE={renderer_mode}",
    ]
    link_recipe = [
        "{ld}",
        "-arch",
        "arm64",
        "-syslibroot",
        "{sdk_root}",
        "-platform_version",
        "macos",
        "13.0",
        "{sdk_version}",
        "-lSystem",
        "-dead_strip",
        "-no_adhoc_codesign",
        "-o",
        "{unsigned_executable}",
        "{object}",
    ]
    sign_recipe = [
        "{codesign}",
        "--force",
        "--sign",
        "-",
        "--options",
        "kill",
        "--timestamp=none",
        "--identifier",
        _v2_signature_identifier(pdf_id),
        "--verbose=0",
        "{unsigned_executable}",
    ]
    verify_recipe = [
        "{codesign}",
        "--verify",
        "--strict",
        "--verbose=0",
        "{signed_executable}",
    ]
    return {
        operation: {
            "argv": argv,
            "argv_sha256": _sha256(_canonical_json(argv)),
        }
        for operation, argv in (
            ("compile", compile_recipe),
            ("link", link_recipe),
            ("sign", sign_recipe),
            ("verify", verify_recipe),
        )
    }


def _pinned_bytes(pinned: object, *, maximum: int, context: str) -> bytes:
    if pinned.size > maximum:
        _fail(f"{context} exceeds the {maximum}-byte limit")
    raw = os.pread(pinned.descriptor, pinned.size, 0)
    if len(raw) != pinned.size or _sha256(raw) != pinned.sha256:
        _fail(f"{context} bytes changed while descriptor-pinned")
    return raw


def _parse_canonical_json(
    pinned: object,
    *,
    maximum: int,
    context: str,
    trailing_newline: bool,
) -> Mapping[str, object]:
    raw = _pinned_bytes(pinned, maximum=maximum, context=context)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{context} is not valid JSON: {error}")
    mapping = _expect_mapping(value, context=context)
    expected = _canonical_json(mapping) + (b"\n" if trailing_newline else b"")
    if raw != expected:
        _fail(f"{context} must use exact canonical JSON encoding")
    return mapping


def _normalize_sha_map(
    value: Mapping[str, str],
    *,
    keys: Sequence[str],
    context: str,
) -> dict[str, str]:
    if set(value) != set(keys):
        _fail(f"{context} must name exactly {list(keys)}")
    return {key: _expect_sha256(value[key], context=f"{context}.{key}") for key in keys}


def _normalize_plan(
    value: Mapping[str, object],
    *,
    release_id: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> dict[str, object]:
    context = "derivation plan"
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "release_id",
            "upstream_bindings",
            "real_producer_toolchain_authority",
            "execution_contract",
            "builder_release_toolchain_baseline",
            "documents",
            "non_inference_limits",
        },
        context=context,
    )
    if value["schema"] != DERIVATION_PLAN_SCHEMA:
        _fail(f"{context} has the wrong schema")
    mode = value["mode"]
    if mode not in MODES:
        _fail(f"{context}.mode must be one of {list(MODES)}")
    if value["release_id"] != _expect_token(release_id, context="release id"):
        _fail(f"{context}.release_id does not match the caller")
    if mode == MODE_SYNTHETIC and not release_id.startswith("synthetic-"):
        _fail("synthetic-canary release ids must begin with 'synthetic-'")
    upstream = _expect_mapping(
        value["upstream_bindings"],
        context=f"{context}.upstream_bindings",
    )
    normalized_upstream = _normalize_sha_map(
        upstream,  # type: ignore[arg-type]
        keys=UPSTREAM_BINDING_KEYS,
        context=f"{context}.upstream_bindings",
    )
    expected_upstream = _normalize_sha_map(
        expected_upstream_sha256,
        keys=UPSTREAM_BINDING_KEYS,
        context="caller upstream SHA-256 anchors",
    )
    if normalized_upstream != expected_upstream:
        _fail(f"{context} does not match caller upstream SHA-256 anchors")
    authority_bindings = _normalize_sha_map(
        _expect_mapping(
            value["real_producer_toolchain_authority"],
            context=f"{context}.real_producer_toolchain_authority",
        ),  # type: ignore[arg-type]
        keys=PDF_IDS,
        context=f"{context}.real_producer_toolchain_authority",
    )
    expected_authority = _normalize_sha_map(
        expected_authority_sha256,
        keys=PDF_IDS,
        context="caller producer-toolchain authority SHA-256 anchors",
    )
    if authority_bindings != expected_authority:
        _fail(
            f"{context} does not match caller producer-toolchain authority anchors",
        )
    if value["execution_contract"] != EXECUTION_CONTRACT:
        _fail(f"{context}.execution_contract drifted")
    if (
        value["builder_release_toolchain_baseline"]
        != BUILDER_RELEASE_TOOLCHAIN_BASELINE
    ):
        _fail(f"{context}.builder_release_toolchain_baseline drifted")
    if value["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail(f"{context}.non_inference_limits drifted")

    source_anchors = _normalize_sha_map(
        expected_source_sha256,
        keys=PDF_IDS,
        context="caller source-bundle SHA-256 anchors",
    )
    producer_anchors = _normalize_sha_map(
        expected_producer_sha256,
        keys=PDF_IDS,
        context="caller producer SHA-256 anchors",
    )
    pdf_anchors = _normalize_sha_map(
        expected_pdf_sha256,
        keys=PDF_IDS,
        context="caller PDF SHA-256 anchors",
    )
    raw_documents = _expect_sequence(value["documents"], context=f"{context}.documents")
    if len(raw_documents) != len(PDF_ORDER):
        _fail(f"{context}.documents must contain exactly four roles")
    documents: list[dict[str, object]] = []
    for index, (pdf_id, pdf_member) in enumerate(PDF_ORDER):
        item_context = f"{context}.documents[{index}]"
        item = _expect_mapping(raw_documents[index], context=item_context)
        _expect_keys(
            item,
            {
                "pdf_id",
                "pdf_member",
                "source_bundle_member",
                "source_bundle_bytes",
                "source_bundle_sha256",
                "producer_member",
                "producer_bytes",
                "producer_sha256",
                "expected_pdf_sha256",
                "toolchain_profile",
                "adapter_authority",
                "producer_arguments",
            },
            context=item_context,
        )
        if item["pdf_id"] != pdf_id or item["pdf_member"] != pdf_member:
            _fail(f"{item_context} is not in the fixed PDF order")
        if item["source_bundle_member"] != SOURCE_MEMBER_BY_ID[pdf_id]:
            _fail(f"{item_context}.source_bundle_member drifted")
        if item["producer_member"] != PRODUCER_MEMBER_BY_ID[pdf_id]:
            _fail(f"{item_context}.producer_member drifted")
        source_size = _expect_positive_int(
            item["source_bundle_bytes"],
            context=f"{item_context}.source_bundle_bytes",
        )
        producer_size = _expect_positive_int(
            item["producer_bytes"],
            context=f"{item_context}.producer_bytes",
        )
        if source_size > MAX_SOURCE_BUNDLE_BYTES:
            _fail(f"{item_context}.source_bundle_bytes exceeds its bound")
        if producer_size > MAX_PRODUCER_BYTES:
            _fail(f"{item_context}.producer_bytes exceeds its bound")
        source_sha = _expect_sha256(
            item["source_bundle_sha256"],
            context=f"{item_context}.source_bundle_sha256",
        )
        producer_sha = _expect_sha256(
            item["producer_sha256"],
            context=f"{item_context}.producer_sha256",
        )
        pdf_sha = _expect_sha256(
            item["expected_pdf_sha256"],
            context=f"{item_context}.expected_pdf_sha256",
        )
        if source_sha != source_anchors[pdf_id]:
            _fail(f"{item_context} does not match its caller source-bundle anchor")
        if producer_sha != producer_anchors[pdf_id]:
            _fail(f"{item_context} does not match its caller producer anchor")
        if pdf_sha != pdf_anchors[pdf_id]:
            _fail(f"{item_context} does not match its caller PDF anchor")
        profile = _expect_token(
            item["toolchain_profile"],
            context=f"{item_context}.toolchain_profile",
        )
        authority = _expect_mapping(
            item["adapter_authority"],
            context=f"{item_context}.adapter_authority",
        )
        _expect_keys(
            authority,
            {
                "status",
                "authority_id",
                "authentication",
                "authority_receipt_sha256",
                "reviewed_producer_sha256",
            },
            context=f"{item_context}.adapter_authority",
        )
        expected_authority_status = (
            "synthetic-test-only" if mode == MODE_SYNTHETIC else "caller-authorized"
        )
        if authority["status"] != expected_authority_status:
            _fail(f"{item_context}.adapter_authority.status is invalid")
        authority_id = _expect_token(
            authority["authority_id"],
            context=f"{item_context}.adapter_authority.authority_id",
        )
        if authority["authentication"] != "caller-sha-anchor-only":
            _fail(f"{item_context}.adapter_authority.authentication is invalid")
        reviewed_sha = _expect_sha256(
            authority["reviewed_producer_sha256"],
            context=f"{item_context}.adapter_authority.reviewed_producer_sha256",
        )
        if reviewed_sha != producer_sha:
            _fail(f"{item_context}.adapter_authority reviews different bytes")
        authority_receipt_sha = _expect_sha256(
            authority["authority_receipt_sha256"],
            context=(f"{item_context}.adapter_authority.authority_receipt_sha256"),
        )
        if authority_receipt_sha != authority_bindings[pdf_id]:
            _fail(f"{item_context}.adapter_authority has the wrong authority binding")
        if item["producer_arguments"] != _expected_arguments(pdf_id):
            _fail(f"{item_context}.producer_arguments drifted")
        if mode == MODE_SYNTHETIC and profile != "synthetic-native-copy-v1":
            _fail(f"{item_context} must use the synthetic canary profile")
        if mode == MODE_REVISION and profile == "synthetic-native-copy-v1":
            _fail(f"{item_context} requires a selected real producer profile")
        documents.append(
            {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "source_bundle_member": SOURCE_MEMBER_BY_ID[pdf_id],
                "source_bundle_bytes": source_size,
                "source_bundle_sha256": source_sha,
                "producer_member": PRODUCER_MEMBER_BY_ID[pdf_id],
                "producer_bytes": producer_size,
                "producer_sha256": producer_sha,
                "expected_pdf_sha256": pdf_sha,
                "toolchain_profile": profile,
                "adapter_authority": {
                    "status": expected_authority_status,
                    "authority_id": authority_id,
                    "authentication": "caller-sha-anchor-only",
                    "authority_receipt_sha256": authority_receipt_sha,
                    "reviewed_producer_sha256": reviewed_sha,
                },
                "producer_arguments": _expected_arguments(pdf_id),
            },
        )
    if (
        sum(int(item["source_bundle_bytes"]) for item in documents)
        > MAX_TOTAL_SOURCE_BYTES
    ):
        _fail("source-bundle bytes exceed the aggregate bound")
    if (
        sum(int(item["producer_bytes"]) for item in documents)
        > MAX_TOTAL_PRODUCER_BYTES
    ):
        _fail("producer bytes exceed the aggregate bound")
    if len(set(pdf_anchors.values())) != len(PDF_ORDER):
        _fail("the four caller PDF anchors must be byte-distinct")
    if len(set(authority_bindings.values())) != len(PDF_ORDER):
        _fail("the four producer-toolchain authority bindings must be distinct")
    return {
        "schema": DERIVATION_PLAN_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "upstream_bindings": normalized_upstream,
        "real_producer_toolchain_authority": authority_bindings,
        "execution_contract": EXECUTION_CONTRACT,
        "builder_release_toolchain_baseline": BUILDER_RELEASE_TOOLCHAIN_BASELINE,
        "documents": documents,
        "non_inference_limits": NON_INFERENCE_LIMITS,
    }


def _normalize_authority_receipt(
    value: Mapping[str, object],
    *,
    mode: str,
    release_id: str,
    pdf_id: str,
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate one opened authority receipt against the exact selected inputs."""
    context = f"producer-toolchain authority {pdf_id}"
    adapter_authority = _expect_mapping(
        document["adapter_authority"],
        context=f"{context} plan adapter authority",
    )
    _expect_keys(
        value,
        {
            "schema",
            "mode",
            "release_id",
            "pdf_id",
            "authority_id",
            "status",
            "authentication",
            "source_bundle_member",
            "source_bundle_sha256",
            "producer_member",
            "producer_bytes",
            "producer_sha256",
            "toolchain_profile",
            "producer_arguments",
            "review_scope",
        },
        context=context,
    )
    expected = {
        "schema": PRODUCER_AUTHORITY_SCHEMA,
        "mode": mode,
        "release_id": release_id,
        "pdf_id": pdf_id,
        "authority_id": adapter_authority["authority_id"],
        "status": adapter_authority["status"],
        "authentication": "caller-sha-anchor-only",
        "source_bundle_member": document["source_bundle_member"],
        "source_bundle_sha256": document["source_bundle_sha256"],
        "producer_member": document["producer_member"],
        "producer_bytes": document["producer_bytes"],
        "producer_sha256": document["producer_sha256"],
        "toolchain_profile": document["toolchain_profile"],
        "producer_arguments": document["producer_arguments"],
        "review_scope": PRODUCER_AUTHORITY_REVIEW_SCOPE,
    }
    if value != expected:
        _fail(f"{context} does not authorize the exact selected derivation inputs")
    return expected


def _v2_package_contract(pdf_id: str) -> dict[str, object]:
    producer_member = PRODUCER_MEMBER_BY_ID[pdf_id]
    authority_member = AUTHORITY_MEMBER_BY_ID[pdf_id]
    return {
        "root_mode": V2_PACKAGE_ROOT_MODE,
        "root_owner": "effective-user-id",
        "member_order": [producer_member, authority_member],
        "member_modes": {
            producer_member: V2_PRODUCER_MODE,
            authority_member: V2_AUTHORITY_MODE,
        },
        "member_owner": "effective-user-id",
        "member_link_count": 1,
        "authority_self_reference": (
            "authority SHA-256 is excluded and must be supplied by an external caller"
        ),
        "publication": "atomic-no-replace-directory-rename",
        "authority_encoding": "canonical-json-plus-one-lf",
        "seal_scope": "posix-owner-mode-link-count-and-content-only",
    }


def _v2_package_content_record(
    pdf_id: str,
    *,
    producer_bytes: int,
    producer_sha256: str,
    authority_bytes: int,
    authority_sha256: str,
) -> dict[str, object]:
    return {
        "root_mode": V2_PACKAGE_ROOT_MODE,
        "root_owner": "effective-user-id",
        "members": [
            {
                "member": PRODUCER_MEMBER_BY_ID[pdf_id],
                "mode": V2_PRODUCER_MODE,
                "owner": "effective-user-id",
                "link_count": 1,
                "bytes": producer_bytes,
                "sha256": producer_sha256,
            },
            {
                "member": AUTHORITY_MEMBER_BY_ID[pdf_id],
                "mode": V2_AUTHORITY_MODE,
                "owner": "effective-user-id",
                "link_count": 1,
                "bytes": authority_bytes,
                "sha256": authority_sha256,
            },
        ],
    }


def _validate_projection_digest(
    value: Mapping[str, object],
    *,
    digest_key: str,
    context: str,
) -> str:
    digest = _expect_sha256(value.get(digest_key), context=f"{context}.{digest_key}")
    body = dict(value)
    body.pop(digest_key, None)
    if _sha256(_canonical_json(body)) != digest:
        _fail(f"{context}.{digest_key} does not hash its exact projection")
    return digest


def _normalize_v2_code_directory(
    value: object,
    *,
    producer_bytes: int,
    context: str,
) -> dict[str, object]:
    native = _expect_mapping(value, context=context)
    _expect_keys(
        native,
        {
            "binary_container",
            "architecture",
            "cpu_subtype",
            "hash_type",
            "code_directory_bytes",
            "code_directory_flags",
            "cdhash",
            "code_limit",
            "code_slots",
            "page_size",
            "signature_offset",
            "signature_bytes",
        },
        context=context,
    )
    code_directory_flags = _expect_nonnegative_int(
        native["code_directory_flags"],
        context=f"{context}.code_directory_flags",
    )
    if (
        native["binary_container"] != "thin-macho64"
        or native["architecture"] != "arm64"
        or native["cpu_subtype"] != "all"
        or native["hash_type"] != "sha256"
        or code_directory_flags != 514
        or not isinstance(native["cdhash"], str)
        or re.fullmatch(r"[0-9a-f]{40}", str(native["cdhash"])) is None
    ):
        _fail(f"{context} has the wrong native identity")
    code_directory_bytes = _expect_positive_int(
        native["code_directory_bytes"],
        context=f"{context}.code_directory_bytes",
    )
    code_limit = _expect_positive_int(
        native["code_limit"],
        context=f"{context}.code_limit",
    )
    code_slots = _expect_positive_int(
        native["code_slots"],
        context=f"{context}.code_slots",
    )
    page_size = _expect_positive_int(
        native["page_size"],
        context=f"{context}.page_size",
    )
    signature_offset = _expect_positive_int(
        native["signature_offset"],
        context=f"{context}.signature_offset",
    )
    signature_bytes = _expect_positive_int(
        native["signature_bytes"],
        context=f"{context}.signature_bytes",
    )
    if (
        producer_bytes > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or code_directory_bytes > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or code_limit > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or code_slots > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or page_size > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or signature_offset > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or signature_bytes > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or code_limit != signature_offset
        or page_size != 16_384
        or code_slots != (code_limit + page_size - 1) // page_size
        or signature_offset + signature_bytes != producer_bytes
    ):
        _fail(f"{context} extents are inconsistent with the producer")
    return dict(native)


def _v2_code_directory_core(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "binary_container": value["binary_container"],
        "architecture": value["architecture"],
        "hash_type": value["hash_type"],
        "code_directory_bytes": value["code_directory_bytes"],
        "cdhash": value["cdhash"],
    }


def _normalize_v2_output_record(
    value: object,
    *,
    context: str,
    expected_member: str,
    expected_sha256: str,
) -> dict[str, object]:
    record = _expect_mapping(value, context=context)
    _expect_keys(record, {"member", "bytes", "sha256"}, context=context)
    if record["member"] != expected_member:
        _fail(f"{context}.member drifted")
    size = _expect_positive_int(record["bytes"], context=f"{context}.bytes")
    maximum = MAX_V2_NATIVE_OUTPUT_BYTES
    if size > maximum:
        _fail(f"{context}.bytes exceeds the {maximum}-byte bound")
    digest = _expect_sha256(record["sha256"], context=f"{context}.sha256")
    if digest != expected_sha256:
        _fail(f"{context} differs from its independent caller anchor")
    return {"member": expected_member, "bytes": size, "sha256": digest}


def _normalize_v2_plan(
    value: Mapping[str, object],
    *,
    release_id: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_package_sha256: Mapping[str, str],
    expected_renderer_manifest_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> dict[str, object]:
    context = "v2 derivation plan"
    _validate_v2_json_scalar_types(value, context=context)
    _expect_keys(
        value,
        {
            "schema",
            "contract",
            "mode",
            "release_id",
            "upstream_bindings",
            "execution_contract",
            "source_bundle_set_sha256",
            "producer_package_set_sha256",
            "documents",
            "non_inference_limits",
        },
        context=context,
    )
    if (
        value["schema"] != DERIVATION_PLAN_SCHEMA_V2
        or value["contract"] != DERIVATION_CLOSURE_CONTRACT_V2
        or value["mode"] != MODE_REVISION
    ):
        _fail("v2 derivation plan rejects schema, contract, or mode downgrade")
    release = _expect_token(release_id, context="release id")
    if value["release_id"] != release:
        _fail("v2 derivation plan release_id differs from the caller")
    upstream = _normalize_sha_map(
        _expect_mapping(
            value["upstream_bindings"],
            context=f"{context}.upstream_bindings",
        ),  # type: ignore[arg-type]
        keys=UPSTREAM_BINDING_KEYS,
        context=f"{context}.upstream_bindings",
    )
    if upstream != _normalize_sha_map(
        expected_upstream_sha256,
        keys=UPSTREAM_BINDING_KEYS,
        context="caller upstream SHA-256 anchors",
    ):
        _fail("v2 derivation plan upstream anchors differ from the caller")
    if not _json_exact_equal(value["execution_contract"], EXECUTION_CONTRACT_V2):
        _fail("v2 derivation plan execution contract drifted")
    if not _json_exact_equal(value["non_inference_limits"], NON_INFERENCE_LIMITS_V2):
        _fail("v2 derivation plan non-inference limits drifted")

    caller_maps = {
        "authority": _normalize_sha_map(
            expected_authority_sha256,
            keys=PDF_IDS,
            context="caller v2 authority SHA-256 anchors",
        ),
        "source": _normalize_sha_map(
            expected_source_sha256,
            keys=PDF_IDS,
            context="caller v2 source SHA-256 anchors",
        ),
        "producer": _normalize_sha_map(
            expected_producer_sha256,
            keys=PDF_IDS,
            context="caller v2 producer SHA-256 anchors",
        ),
        "package": _normalize_sha_map(
            expected_package_sha256,
            keys=PDF_IDS,
            context="caller v2 package SHA-256 anchors",
        ),
        "renderer_manifest": _normalize_sha_map(
            expected_renderer_manifest_sha256,
            keys=PDF_IDS,
            context="caller renderer-manifest SHA-256 anchors",
        ),
        "pdf": _normalize_sha_map(
            expected_pdf_sha256,
            keys=PDF_IDS,
            context="caller PDF SHA-256 anchors",
        ),
    }
    for label, anchors in caller_maps.items():
        if len(set(anchors.values())) != len(PDF_ORDER):
            _fail(f"the four v2 {label} caller anchors must be role-distinct")
    raw_documents = _expect_sequence(value["documents"], context=f"{context}.documents")
    if len(raw_documents) != len(PDF_ORDER):
        _fail("v2 derivation plan must contain exactly four ordered roles")
    documents: list[dict[str, object]] = []
    source_set: list[dict[str, object]] = []
    package_set: list[dict[str, object]] = []
    for index, (pdf_id, pdf_member) in enumerate(PDF_ORDER):
        item_context = f"{context}.documents[{index}]"
        item = _expect_mapping(raw_documents[index], context=item_context)
        _expect_keys(
            item,
            {
                "pdf_id",
                "pdf_member",
                "producer_arguments",
                "authorization",
                "caller_anchors",
                "source_bundle",
                "producer_package",
                "expected_output",
            },
            context=item_context,
        )
        if item["pdf_id"] != pdf_id or item["pdf_member"] != pdf_member:
            _fail(f"{item_context} is not in fixed role order")
        if item["producer_arguments"] != _expected_arguments(pdf_id):
            _fail(f"{item_context}.producer_arguments drifted")
        authorization = _expect_mapping(
            item["authorization"],
            context=f"{item_context}.authorization",
        )
        _expect_keys(
            authorization,
            {"status", "authentication", "authority_sha256"},
            context=f"{item_context}.authorization",
        )
        if (
            authorization["status"] != V2_AUTHORIZATION_STATUS
            or authorization["authentication"] != "caller-sha-anchor-only"
            or authorization["authority_sha256"] != caller_maps["authority"][pdf_id]
        ):
            _fail(f"{item_context}.authorization does not match the caller")
        caller_anchors = _normalize_sha_map(
            _expect_mapping(
                item["caller_anchors"],
                context=f"{item_context}.caller_anchors",
            ),  # type: ignore[arg-type]
            keys=V2_CALLER_ANCHOR_KEYS,
            context=f"{item_context}.caller_anchors",
        )
        if caller_anchors["source_bundle_sha256"] != caller_maps["source"][pdf_id]:
            _fail(f"{item_context}.caller_anchors source binding drifted")
        if caller_anchors["pdf_sha256"] != caller_maps["pdf"][pdf_id]:
            _fail(f"{item_context}.caller_anchors PDF binding drifted")
        if (
            caller_anchors["renderer_manifest_sha256"]
            != caller_maps["renderer_manifest"][pdf_id]
        ):
            _fail(f"{item_context}.caller_anchors renderer manifest drifted")

        source = _expect_mapping(
            item["source_bundle"],
            context=f"{item_context}.source_bundle",
        )
        _expect_keys(
            source,
            {
                "member",
                "mode",
                "owner",
                "link_count",
                "bytes",
                "sha256",
                "treatment",
            },
            context=f"{item_context}.source_bundle",
        )
        source_size = _expect_positive_int(
            source["bytes"],
            context=f"{item_context}.source_bundle.bytes",
        )
        source_link_count = _expect_positive_int(
            source["link_count"],
            context=f"{item_context}.source_bundle.link_count",
        )
        if (
            source["member"] != SOURCE_MEMBER_BY_ID[pdf_id]
            or source["mode"] != V2_SOURCE_MODE
            or source["owner"] != "effective-user-id"
            or source_link_count != 1
            or source["treatment"] != "opaque-byte-bundle-not-decoded"
            or source["sha256"] != caller_maps["source"][pdf_id]
            or source_size > MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
        ):
            _fail(f"{item_context}.source_bundle drifted")

        package = _expect_mapping(
            item["producer_package"],
            context=f"{item_context}.producer_package",
        )
        _expect_keys(
            package,
            {
                "root_mode",
                "root_owner",
                "package_content_sha256",
                "producer",
                "authority",
                "authority_projection",
            },
            context=f"{item_context}.producer_package",
        )
        producer = _expect_mapping(
            package["producer"],
            context=f"{item_context}.producer_package.producer",
        )
        _expect_keys(
            producer,
            {
                "member",
                "mode",
                "bytes",
                "sha256",
                "macho_uuid",
                "native_code_directory",
            },
            context=f"{item_context}.producer_package.producer",
        )
        producer_size = _expect_positive_int(
            producer["bytes"],
            context=f"{item_context}.producer_package.producer.bytes",
        )
        producer_digest = _expect_sha256(
            producer["sha256"],
            context=f"{item_context}.producer_package.producer.sha256",
        )
        if (
            producer["member"] != PRODUCER_MEMBER_BY_ID[pdf_id]
            or producer["mode"] != V2_PRODUCER_MODE
            or producer_digest != caller_maps["producer"][pdf_id]
            or producer_size > MAX_V2_NATIVE_EXECUTABLE_BYTES
            or not isinstance(producer["macho_uuid"], str)
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                str(producer["macho_uuid"]),
            )
            is None
        ):
            _fail(f"{item_context}.producer package producer drifted")
        native_cd = _normalize_v2_code_directory(
            producer["native_code_directory"],
            producer_bytes=producer_size,
            context=f"{item_context}.producer native CodeDirectory",
        )
        authority = _expect_mapping(
            package["authority"],
            context=f"{item_context}.producer_package.authority",
        )
        _expect_keys(
            authority,
            {"member", "mode", "bytes", "sha256", "schema"},
            context=f"{item_context}.producer_package.authority",
        )
        authority_size = _expect_positive_int(
            authority["bytes"],
            context=f"{item_context}.producer_package.authority.bytes",
        )
        authority_digest = _expect_sha256(
            authority["sha256"],
            context=f"{item_context}.producer_package.authority.sha256",
        )
        if (
            authority["member"] != AUTHORITY_MEMBER_BY_ID[pdf_id]
            or authority["mode"] != V2_AUTHORITY_MODE
            or authority["schema"] != NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID[pdf_id]
            or authority_digest != caller_maps["authority"][pdf_id]
            or authority_size > MAX_V2_AUTHORITY_BYTES
        ):
            _fail(f"{item_context}.producer package authority drifted")
        authority_projection = _expect_mapping(
            package["authority_projection"],
            context=f"{item_context}.producer_package.authority_projection",
        )
        _expect_keys(
            authority_projection,
            set(V2_AUTHORITY_PROJECTION_KEYS),
            context=f"{item_context}.producer_package.authority_projection",
        )
        normalized_projection = {
            key: _expect_sha256(
                authority_projection[key],
                context=f"{item_context}.authority_projection.{key}",
            )
            for key in authority_projection
        }
        content_record = _v2_package_content_record(
            pdf_id,
            producer_bytes=producer_size,
            producer_sha256=producer_digest,
            authority_bytes=authority_size,
            authority_sha256=authority_digest,
        )
        package_digest = _sha256(_canonical_json(content_record))
        if (
            package["root_mode"] != V2_PACKAGE_ROOT_MODE
            or package["root_owner"] != "effective-user-id"
            or package["package_content_sha256"] != package_digest
            or package_digest != caller_maps["package"][pdf_id]
        ):
            _fail(f"{item_context}.producer_package content binding drifted")

        expected_output = _expect_mapping(
            item["expected_output"],
            context=f"{item_context}.expected_output",
        )
        _expect_keys(
            expected_output,
            {"renderer_manifest", "pdf"},
            context=f"{item_context}.expected_output",
        )
        renderer_manifest = _normalize_v2_output_record(
            expected_output["renderer_manifest"],
            context=f"{item_context}.expected_output.renderer_manifest",
            expected_member="render-receipt.json",
            expected_sha256=caller_maps["renderer_manifest"][pdf_id],
        )
        expected_pdf = _normalize_v2_output_record(
            expected_output["pdf"],
            context=f"{item_context}.expected_output.pdf",
            expected_member=pdf_member,
            expected_sha256=caller_maps["pdf"][pdf_id],
        )
        normalized_source = {
            "member": SOURCE_MEMBER_BY_ID[pdf_id],
            "mode": V2_SOURCE_MODE,
            "owner": "effective-user-id",
            "link_count": 1,
            "bytes": source_size,
            "sha256": caller_maps["source"][pdf_id],
            "treatment": "opaque-byte-bundle-not-decoded",
        }
        normalized_package = {
            "root_mode": V2_PACKAGE_ROOT_MODE,
            "root_owner": "effective-user-id",
            "package_content_sha256": package_digest,
            "producer": {
                "member": PRODUCER_MEMBER_BY_ID[pdf_id],
                "mode": V2_PRODUCER_MODE,
                "bytes": producer_size,
                "sha256": producer_digest,
                "macho_uuid": producer["macho_uuid"],
                "native_code_directory": native_cd,
            },
            "authority": {
                "member": AUTHORITY_MEMBER_BY_ID[pdf_id],
                "mode": V2_AUTHORITY_MODE,
                "bytes": authority_size,
                "sha256": authority_digest,
                "schema": NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID[pdf_id],
            },
            "authority_projection": normalized_projection,
        }
        documents.append(
            {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "producer_arguments": _expected_arguments(pdf_id),
                "authorization": {
                    "status": V2_AUTHORIZATION_STATUS,
                    "authentication": "caller-sha-anchor-only",
                    "authority_sha256": authority_digest,
                },
                "caller_anchors": caller_anchors,
                "source_bundle": normalized_source,
                "producer_package": normalized_package,
                "expected_output": {
                    "renderer_manifest": renderer_manifest,
                    "pdf": expected_pdf,
                },
            },
        )
        source_set.append({"pdf_id": pdf_id, **normalized_source})
        package_set.append(
            {
                "pdf_id": pdf_id,
                "package_content_sha256": package_digest,
                **content_record,
            },
        )
    if sum(int(item["bytes"]) for item in source_set) > MAX_TOTAL_SOURCE_BYTES:
        _fail("v2 source bundles exceed their aggregate bound")
    if (
        sum(
            int(
                _expect_mapping(
                    _expect_mapping(
                        item["producer_package"],
                        context="normalized v2 package",
                    )["producer"],
                    context="normalized v2 producer",
                )["bytes"],
            )
            for item in documents
        )
        > MAX_TOTAL_PRODUCER_BYTES
    ):
        _fail("v2 producers exceed their aggregate bound")
    source_set_sha = _sha256(_canonical_json(source_set))
    package_set_sha = _sha256(_canonical_json(package_set))
    if value["source_bundle_set_sha256"] != source_set_sha:
        _fail("v2 plan source-bundle set digest drifted")
    if value["producer_package_set_sha256"] != package_set_sha:
        _fail("v2 plan package set digest drifted")
    return {
        "schema": DERIVATION_PLAN_SCHEMA_V2,
        "contract": DERIVATION_CLOSURE_CONTRACT_V2,
        "mode": MODE_REVISION,
        "release_id": release,
        "upstream_bindings": upstream,
        "execution_contract": _canonical_copy(EXECUTION_CONTRACT_V2),
        "source_bundle_set_sha256": source_set_sha,
        "producer_package_set_sha256": package_set_sha,
        "documents": documents,
        "non_inference_limits": _canonical_copy(NON_INFERENCE_LIMITS_V2),
    }


def _validate_v2_toolchain_projection(
    value: object,
    *,
    caller_anchors: Mapping[str, str],
    context: str,
) -> None:
    toolchain = _expect_mapping(value, context=context)
    _expect_keys(
        toolchain,
        {
            "clang",
            "linker",
            "codesign",
            "git",
            "compiler_resource_tree",
            "sdk_tree",
            "sdk_version",
            "linker_invocation",
            "codesign_invocation",
            "toolchain_projection_sha256",
        },
        context=context,
    )
    _validate_projection_digest(
        toolchain,
        digest_key="toolchain_projection_sha256",
        context=context,
    )
    sdk_version = toolchain["sdk_version"]
    if (
        not isinstance(sdk_version, str)
        or re.fullmatch(r"[1-9][0-9]{0,2}\.[0-9]{1,2}", sdk_version) is None
        or toolchain["linker_invocation"] != "direct-bounded-main-process"
        or toolchain["codesign_invocation"]
        != "bounded-main-path-execution; selected-fat-slice-live-mapping-not-attested"
    ):
        _fail(f"{context} fixed invocation or SDK identity drifted")
    tool_specs = (
        ("clang", "xcode-default-toolchain-clang", "clang_sha256"),
        ("linker", "xcode-default-toolchain-ld", "linker_sha256"),
        ("codesign", "system-codesign", "codesign_sha256"),
        ("git", "xcode-git", "git_sha256"),
    )
    for key, locator, anchor in tool_specs:
        record = _expect_mapping(toolchain[key], context=f"{context}.{key}")
        _expect_keys(
            record,
            {
                "locator",
                "path_recorded",
                "bytes",
                "sha256",
                "mode",
                "uid",
                "link_count",
                "binary",
            },
            context=f"{context}.{key}",
        )
        tool_bytes = _expect_positive_int(
            record["bytes"],
            context=f"{context}.{key}.bytes",
        )
        mode = record["mode"]
        uid = _expect_nonnegative_int(
            record["uid"],
            context=f"{context}.{key}.uid",
        )
        link_count = _expect_positive_int(
            record["link_count"],
            context=f"{context}.{key}.link_count",
        )
        if (
            tool_bytes > MAX_V2_NATIVE_TOOL_BYTES
            or record["locator"] != locator
            or record["path_recorded"] is not False
            or record["sha256"] != caller_anchors[anchor]
            or uid > 0xFFFFFFFF
            or uid != 0
            or link_count != 1
            or not isinstance(mode, str)
            or re.fullmatch(r"[0-7]{4}", mode) is None
            or not int(mode, 8) & 0o100
            or int(mode, 8) & 0o022
        ):
            _fail(f"{context}.{key} executable identity drifted")
        binary = _expect_mapping(record["binary"], context=f"{context}.{key}.binary")
        if key != "codesign":
            _expect_keys(
                binary,
                {
                    "binary_container",
                    "architecture",
                    "cpu_type",
                    "cpu_subtype",
                    "cpu_subtype_capabilities",
                    "file_type",
                    "load_command_count",
                    "load_command_bytes",
                },
                context=f"{context}.{key}.binary",
            )
            cpu_type = _expect_positive_int(
                binary["cpu_type"],
                context=f"{context}.{key}.binary.cpu_type",
            )
            cpu_subtype = _expect_nonnegative_int(
                binary["cpu_subtype"],
                context=f"{context}.{key}.binary.cpu_subtype",
            )
            capabilities = _expect_nonnegative_int(
                binary["cpu_subtype_capabilities"],
                context=f"{context}.{key}.binary.cpu_subtype_capabilities",
            )
            if (
                binary["binary_container"] != "thin-macho64"
                or binary["architecture"] != "arm64"
                or cpu_type > 0x7FFFFFFF
                or cpu_type != _machine.CPU_TYPE_ARM64
                or cpu_subtype > 0x00FFFFFF
                or cpu_subtype != 0
                or capabilities > 0xFF000000
                or capabilities != 0
                or binary["file_type"] != "execute"
            ):
                _fail(f"{context}.{key} thin Mach-O identity drifted")
            load_command_count = _expect_positive_int(
                binary["load_command_count"],
                context=f"{context}.{key}.load_command_count",
            )
            load_command_bytes = _expect_positive_int(
                binary["load_command_bytes"],
                context=f"{context}.{key}.load_command_bytes",
            )
            if (
                load_command_count > MAX_V2_NATIVE_MACH_LOAD_COMMANDS
                or load_command_bytes > tool_bytes
                or load_command_bytes < load_command_count * 8
            ):
                _fail(f"{context}.{key} Mach-O load commands exceed their bounds")
            continue
        _expect_keys(
            binary,
            {
                "binary_container",
                "fat_endianness",
                "slice_count",
                "slices",
                "selected_execution_slice",
                "selected_slice_live_mapping",
            },
            context=f"{context}.codesign.binary",
        )
        slices = _expect_sequence(
            binary["slices"],
            context=f"{context}.codesign.slices",
        )
        slice_count = _expect_positive_int(
            binary["slice_count"],
            context=f"{context}.codesign.slice_count",
        )
        container = binary["binary_container"]
        if (
            container not in {"fat-macho32", "fat-macho64"}
            or binary["fat_endianness"] != "big"
            or slice_count != len(slices)
            or not 1 <= slice_count <= MAX_V2_NATIVE_FAT_SLICES
            or binary["selected_slice_live_mapping"] != "not-attested"
        ):
            _fail(f"{context}.codesign FAT envelope drifted")
        header_bytes = 8 + slice_count * (32 if container == "fat-macho64" else 20)
        extents: list[tuple[int, int]] = []
        architectures: set[str] = set()
        selected_candidates: list[Mapping[str, object]] = []
        for expected_index, raw_slice in enumerate(slices):
            slice_record = _expect_mapping(
                raw_slice,
                context=f"{context}.codesign.slices[{expected_index}]",
            )
            _expect_keys(
                slice_record,
                {
                    "index",
                    "architecture",
                    "cpu_type",
                    "cpu_subtype",
                    "cpu_subtype_capabilities",
                    "alignment_exponent",
                    "offset",
                    "bytes",
                    "sha256",
                },
                context=f"{context}.codesign.slices[{expected_index}]",
            )
            index = _expect_nonnegative_int(
                slice_record["index"],
                context=f"{context}.codesign.slices[{expected_index}].index",
            )
            cpu_type = _expect_positive_int(
                slice_record["cpu_type"],
                context=f"{context}.codesign.slices[{expected_index}].cpu_type",
            )
            cpu_subtype = _expect_nonnegative_int(
                slice_record["cpu_subtype"],
                context=f"{context}.codesign.slices[{expected_index}].cpu_subtype",
            )
            capabilities = _expect_nonnegative_int(
                slice_record["cpu_subtype_capabilities"],
                context=(
                    f"{context}.codesign.slices[{expected_index}]."
                    "cpu_subtype_capabilities"
                ),
            )
            alignment = _expect_nonnegative_int(
                slice_record["alignment_exponent"],
                context=f"{context}.codesign.slices[{expected_index}].alignment",
            )
            offset = _expect_positive_int(
                slice_record["offset"],
                context=f"{context}.codesign.slices[{expected_index}].offset",
            )
            size = _expect_positive_int(
                slice_record["bytes"],
                context=f"{context}.codesign.slices[{expected_index}].bytes",
            )
            architecture = slice_record["architecture"]
            expected_cpu = {
                "arm64": _machine.CPU_TYPE_ARM64,
                "x86_64": 0x01000007,
            }.get(str(architecture))
            if (
                index != expected_index
                or expected_cpu is None
                or cpu_type > 0x7FFFFFFF
                or cpu_type != expected_cpu
                or architecture in architectures
                or cpu_subtype > 0x00FFFFFF
                or capabilities > 0xFF000000
                or capabilities & 0x00FFFFFF
                or alignment > 31
                or offset < header_bytes
                or offset % (1 << alignment)
                or offset + size > tool_bytes
                or any(
                    not (offset + size <= left or offset >= right)
                    for left, right in extents
                )
            ):
                _fail(f"{context}.codesign slice extent or CPU identity drifted")
            _expect_sha256(
                slice_record["sha256"],
                context=f"{context}.codesign.slices[{expected_index}].sha256",
            )
            architectures.add(str(architecture))
            extents.append((offset, offset + size))
            if cpu_type == _machine.CPU_TYPE_ARM64 and cpu_subtype == 2:
                if capabilities != 0x80000000:
                    _fail(f"{context}.codesign arm64e capability bits drifted")
                selected_candidates.append(slice_record)
        selected = _expect_mapping(
            binary["selected_execution_slice"],
            context=f"{context}.codesign.selected_execution_slice",
        )
        if len(selected_candidates) != 1 or not _json_exact_equal(
            selected,
            selected_candidates[0],
        ):
            _fail(f"{context}.codesign selected execution slice drifted")
    for key, locator, expected_root, anchor in (
        (
            "compiler_resource_tree",
            "xcode-clang-resource-root",
            V2_COMPILER_RESOURCE_ROOT,
            "compiler_resource_tree_sha256",
        ),
        ("sdk_tree", "xcode-macos-sdk-root", V2_SDK_ROOT, "sdk_tree_sha256"),
    ):
        record = _expect_mapping(toolchain[key], context=f"{context}.{key}")
        _expect_keys(
            record,
            {
                "locator",
                "root_path_recorded",
                "root_path_utf8_bytes",
                "root_path_utf8_sha256",
                "tree_hash_contract",
                "tree_sha256",
                "file_count",
                "directory_count",
                "symlink_count",
                "entry_count",
                "total_file_bytes",
            },
            context=f"{context}.{key}",
        )
        files = _expect_positive_int(
            record["file_count"],
            context=f"{context}.{key}.file_count",
        )
        directories = _expect_positive_int(
            record["directory_count"],
            context=f"{context}.{key}.directory_count",
        )
        symlinks = _expect_nonnegative_int(
            record["symlink_count"],
            context=f"{context}.{key}.symlink_count",
        )
        entries = _expect_positive_int(
            record["entry_count"],
            context=f"{context}.{key}.entry_count",
        )
        if (
            record["locator"] != locator
            or record["root_path_recorded"] is not False
            or _expect_positive_int(
                record["root_path_utf8_bytes"],
                context=f"{context}.{key}.root_path_utf8_bytes",
            )
            != len(os.fsencode(expected_root))
            or _expect_sha256(
                record["root_path_utf8_sha256"],
                context=f"{context}.{key}.root_path_utf8_sha256",
            )
            != _sha256(os.fsencode(expected_root))
            or record["tree_hash_contract"]
            != "u64be-path-type-mode-nlink-size-content-or-symlink-target-v1"
            or record["tree_sha256"] != caller_anchors[anchor]
            or files > MAX_V2_TREE_FILES
            or directories > MAX_V2_TREE_DIRECTORIES
            or symlinks > MAX_V2_TREE_SYMLINKS
            or entries > MAX_V2_TREE_ENTRIES
            or entries != files + directories + symlinks
        ):
            _fail(f"{context}.{key} tree identity drifted")
        total_file_bytes = _expect_positive_int(
            record["total_file_bytes"],
            context=f"{context}.{key}.total_file_bytes",
        )
        if total_file_bytes > MAX_V2_TREE_BYTES:
            _fail(f"{context}.{key} tree exceeds its aggregate byte bound")


def _normalize_v2_authority(
    value: Mapping[str, object],
    *,
    release_id: str,
    pdf_id: str,
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate one externally anchored native authority capsule and cross-bind it."""
    context = f"v2 native producer authority {pdf_id}"
    _validate_v2_json_scalar_types(value, context=context)
    _expect_keys(
        value,
        {
            "schema",
            "contract",
            "mode",
            "release_id",
            "pdf_id",
            "pdf_member",
            "status",
            "authentication",
            "producer_protocol",
            "producer_arguments",
            "package_contract",
            "source_bundle",
            "producer",
            "launcher_source",
            "launcher_config",
            "build",
            "toolchain",
            "source_release",
            "runtime_handoff",
            "expected_output",
            "caller_anchors",
            "review_scope",
            "non_inference_limits",
            "manifest_body_sha256",
        },
        context=context,
    )
    authority_release_id = _expect_token(
        value["release_id"],
        context=f"{context}.release_id",
    )
    if (
        value["schema"] != NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID[pdf_id]
        or value["contract"] != NATIVE_PRODUCER_AUTHORITY_CONTRACT_V2
        or value["mode"] != MODE_REVISION
        or authority_release_id != release_id
        or value["pdf_id"] != pdf_id
        or value["pdf_member"] != PDF_MEMBER_BY_ID[pdf_id]
        or value["status"] != V2_AUTHORITY_STATUS
        or value["authentication"] != "caller-sha-anchor-only"
        or value["producer_protocol"] != PRODUCER_PROTOCOL
        or not _json_exact_equal(
            value["producer_arguments"],
            _expected_arguments(pdf_id),
        )
        or not _json_exact_equal(
            value["package_contract"],
            _v2_package_contract(pdf_id),
        )
    ):
        _fail(f"{context} rejects schema, status, mode, or package downgrade")
    if (
        value["review_scope"]
        != "native-launcher-source-config-build-toolchain-bundle-projection-v2"
    ):
        _fail(f"{context}.review_scope drifted")
    native_non_inference = _expect_mapping(
        value["non_inference_limits"],
        context=f"{context}.non_inference",
    )
    if not _json_exact_equal(
        native_non_inference,
        NATIVE_PRODUCER_NON_INFERENCE_LIMITS_V2,
    ):
        _fail(f"{context}.non_inference exact boundary drifted")
    manifest_body_sha = _validate_projection_digest(
        value,
        digest_key="manifest_body_sha256",
        context=context,
    )
    plan_package = _expect_mapping(
        document["producer_package"],
        context=f"{context} plan package",
    )
    plan_projection = _expect_mapping(
        plan_package["authority_projection"],
        context=f"{context} plan authority projection",
    )
    if manifest_body_sha != plan_projection["manifest_body_sha256"]:
        _fail(f"{context} manifest body differs from the plan")
    caller_anchors = _normalize_sha_map(
        _expect_mapping(
            value["caller_anchors"],
            context=f"{context}.caller_anchors",
        ),  # type: ignore[arg-type]
        keys=V2_CALLER_ANCHOR_KEYS,
        context=f"{context}.caller_anchors",
    )
    if caller_anchors != document["caller_anchors"]:
        _fail(f"{context} caller-anchor projection differs from the plan")

    plan_source = _expect_mapping(
        document["source_bundle"],
        context=f"{context} plan source bundle",
    )
    source = _expect_mapping(value["source_bundle"], context=f"{context}.source_bundle")
    _expect_keys(
        source,
        {
            "member",
            "mode",
            "owner",
            "link_count",
            "bytes",
            "sha256",
            "bundle_projection",
        },
        context=f"{context}.source_bundle",
    )
    source_link_count = _expect_positive_int(
        source["link_count"],
        context=f"{context}.source_bundle.link_count",
    )
    source_bytes = _expect_positive_int(
        source["bytes"],
        context=f"{context}.source_bundle.bytes",
    )
    if (
        source["member"] != SOURCE_MEMBER_BY_ID[pdf_id]
        or source["member"] != plan_source["member"]
        or source["mode"] != V2_SOURCE_MODE
        or source["owner"] != "effective-user-id"
        or source_link_count != 1
        or source_bytes > MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
        or source_bytes != plan_source["bytes"]
        or source["sha256"] != plan_source["sha256"]
        or source["sha256"] != caller_anchors["source_bundle_sha256"]
    ):
        _fail(f"{context}.source_bundle differs from the plan or caller")
    projection = _expect_mapping(
        source["bundle_projection"],
        context=f"{context}.source_bundle.bundle_projection",
    )
    _expect_keys(
        projection,
        {
            "schema",
            "contract",
            "release_id",
            "role",
            "producer_protocol",
            "producer_arguments",
            "canonical_inputs",
            "canonical_inputs_projection_sha256",
            "dependencies",
            "expected_output",
            "non_inference",
            "source_or_base64_payload_recorded",
            "bundle_projection_sha256",
        },
        context=f"{context}.bundle_projection",
    )
    projection_sha = _validate_projection_digest(
        projection,
        digest_key="bundle_projection_sha256",
        context=f"{context}.bundle_projection",
    )
    if (
        projection_sha != plan_projection["bundle_projection_sha256"]
        or projection["schema"] != f"dialect-revision-{pdf_id}-derivation-bundle-v1"
        or projection["contract"] != V2_BUNDLE_CONTRACT
        or projection["release_id"] != release_id
        or projection["role"] != pdf_id
        or projection["producer_protocol"] != PRODUCER_PROTOCOL
        or not _json_exact_equal(
            projection["producer_arguments"],
            _expected_arguments(pdf_id),
        )
        or not _json_exact_equal(
            projection["non_inference"],
            V2_BUNDLE_NON_INFERENCE,
        )
        or projection["source_or_base64_payload_recorded"] is not False
    ):
        _fail(f"{context}.bundle_projection fixed bindings drifted")
    canonical_inputs = _expect_sequence(
        projection["canonical_inputs"],
        context=f"{context}.canonical_inputs",
    )
    if len(canonical_inputs) != len(V2_BUNDLE_INPUT_MEMBERS):
        _fail(f"{context}.canonical_inputs has the wrong member count")
    for index, (expected_member, raw_input) in enumerate(
        zip(V2_BUNDLE_INPUT_MEMBERS, canonical_inputs, strict=True),
    ):
        item = _expect_mapping(
            raw_input,
            context=f"{context}.canonical_inputs[{index}]",
        )
        _expect_keys(
            item,
            {"member", "encoding", "bytes", "sha256", "encoded_payload_sha256"},
            context=f"{context}.canonical_inputs[{index}]",
        )
        if item["encoding"] != "base64" or item["member"] != expected_member:
            _fail(f"{context}.canonical input member or encoding drifted")
        input_bytes = _expect_positive_int(
            item["bytes"],
            context=f"{context}.canonical input bytes",
        )
        if input_bytes > MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES:
            _fail(f"{context}.canonical input exceeds its byte bound")
        _expect_sha256(item["sha256"], context=f"{context}.canonical input SHA-256")
        _expect_sha256(
            item["encoded_payload_sha256"],
            context=f"{context}.encoded input SHA-256",
        )
    if _sha256(_canonical_json(canonical_inputs)) != _expect_sha256(
        projection["canonical_inputs_projection_sha256"],
        context=f"{context}.canonical_inputs_projection_sha256",
    ):
        _fail(f"{context}.canonical input projection digest is invalid")
    dependencies = _expect_mapping(
        projection["dependencies"],
        context=f"{context}.bundle_projection.dependencies",
    )
    _expect_keys(
        dependencies,
        {"fonts", "machine_runner", "renderer", "reportlab", "runtime", "tools"},
        context=f"{context}.bundle_projection.dependencies",
    )
    runtime_dependency = _expect_mapping(
        dependencies["runtime"],
        context=f"{context}.bundle runtime",
    )
    renderer_dependency = _expect_mapping(
        dependencies["renderer"],
        context=f"{context}.bundle renderer",
    )
    machine_dependency = _expect_mapping(
        dependencies["machine_runner"],
        context=f"{context}.bundle machine runner",
    )
    _expect_keys(
        runtime_dependency,
        {"bytes", "locator", "python_tag", "sha256"},
        context=f"{context}.bundle runtime",
    )
    for key, record in (
        ("renderer", renderer_dependency),
        ("machine_runner", machine_dependency),
    ):
        _expect_keys(
            record,
            {"bytes", "locator", "member", "sha256"},
            context=f"{context}.bundle {key}",
        )
    for key, record, maximum in (
        ("runtime", runtime_dependency, MAX_V2_NATIVE_TOOL_BYTES),
        ("renderer", renderer_dependency, MAX_V2_NATIVE_BUILDER_BYTES),
        ("machine_runner", machine_dependency, MAX_V2_NATIVE_BUILDER_BYTES),
    ):
        dependency_bytes = _expect_positive_int(
            record["bytes"],
            context=f"{context}.bundle {key}.bytes",
        )
        if dependency_bytes > maximum:
            _fail(f"{context}.bundle {key} exceeds its byte bound")
        if not isinstance(record["locator"], str):
            _fail(f"{context}.bundle {key}.locator must be a string")
        _expect_sha256(
            record["sha256"],
            context=f"{context}.bundle {key}.sha256",
        )
    if runtime_dependency["python_tag"] != "3.12":
        _fail(f"{context}.bundle runtime Python tag drifted")
    fonts = _expect_sequence(
        dependencies["fonts"],
        context=f"{context}.bundle fonts",
    )
    if len(fonts) != 2:
        _fail(f"{context}.bundle fonts must contain regular then bold")
    expected_fonts = (
        ("regular", "system-arial-unicode", "ArialUnicodeMS"),
        ("bold", "system-arial-bold", "Arial-BoldMT"),
    )
    for index, (expected_role, expected_locator, expected_postscript) in enumerate(
        expected_fonts,
    ):
        font = _expect_mapping(fonts[index], context=f"{context}.bundle font {index}")
        _expect_keys(
            font,
            {"bytes", "locator", "postscript_name", "role", "sha256"},
            context=f"{context}.bundle font {index}",
        )
        font_bytes = _expect_positive_int(
            font["bytes"],
            context=f"{context}.bundle font {index}.bytes",
        )
        if (
            font_bytes > MAX_V2_NATIVE_FONT_BYTES
            or font["role"] != expected_role
            or font["locator"] != expected_locator
            or font["postscript_name"] != expected_postscript
        ):
            _fail(f"{context}.bundle font {index} identity drifted")
        _expect_sha256(
            font["sha256"],
            context=f"{context}.bundle font {index}.sha256",
        )
    tools = _expect_sequence(
        dependencies["tools"],
        context=f"{context}.bundle tools",
    )
    if len(tools) != 3:
        _fail(f"{context}.bundle tools must contain the exact renderer tool set")
    expected_tools = (
        ("pdfinfo", "homebrew-pdfinfo"),
        ("pdffonts", "homebrew-pdffonts"),
        ("pdftotext", "homebrew-pdftotext"),
    )
    for index, (expected_name, expected_locator) in enumerate(expected_tools):
        tool = _expect_mapping(tools[index], context=f"{context}.bundle tool {index}")
        _expect_keys(
            tool,
            {"bytes", "locator", "name", "sha256"},
            context=f"{context}.bundle tool {index}",
        )
        tool_bytes = _expect_positive_int(
            tool["bytes"],
            context=f"{context}.bundle tool {index}.bytes",
        )
        if (
            tool_bytes > MAX_V2_NATIVE_TOOL_BYTES
            or tool["name"] != expected_name
            or tool["locator"] != expected_locator
        ):
            _fail(f"{context}.bundle tool {index} identity drifted")
        _expect_sha256(
            tool["sha256"],
            context=f"{context}.bundle tool {index}.sha256",
        )
    reportlab = _expect_mapping(
        dependencies["reportlab"],
        context=f"{context}.bundle reportlab",
    )
    _expect_keys(
        reportlab,
        {
            "bundle_bytes",
            "bundle_sha256",
            "directory_count",
            "entry_count",
            "file_count",
            "locator",
            "total_bytes",
            "tree_sha256",
        },
        context=f"{context}.bundle reportlab",
    )
    bundle_bytes = _expect_positive_int(
        reportlab["bundle_bytes"],
        context=f"{context}.bundle reportlab.bundle_bytes",
    )
    directory_count = _expect_positive_int(
        reportlab["directory_count"],
        context=f"{context}.bundle reportlab.directory_count",
    )
    entry_count = _expect_positive_int(
        reportlab["entry_count"],
        context=f"{context}.bundle reportlab.entry_count",
    )
    file_count = _expect_positive_int(
        reportlab["file_count"],
        context=f"{context}.bundle reportlab.file_count",
    )
    total_bytes = _expect_positive_int(
        reportlab["total_bytes"],
        context=f"{context}.bundle reportlab.total_bytes",
    )
    if (
        bundle_bytes > MAX_V2_NATIVE_REPORTLAB_BYTES
        or total_bytes > MAX_V2_NATIVE_REPORTLAB_BYTES
        or file_count > MAX_V2_NATIVE_REPORTLAB_FILES
        or directory_count > MAX_V2_NATIVE_REPORTLAB_DIRECTORIES
        or entry_count > MAX_V2_NATIVE_REPORTLAB_ENTRIES
        or entry_count != directory_count + file_count
        or reportlab["locator"] != "invoking-python-reportlab"
    ):
        _fail(f"{context}.bundle reportlab tree counts or bytes drifted")
    _expect_sha256(
        reportlab["bundle_sha256"],
        context=f"{context}.bundle reportlab.bundle_sha256",
    )
    _expect_sha256(
        reportlab["tree_sha256"],
        context=f"{context}.bundle reportlab.tree_sha256",
    )
    if (
        runtime_dependency.get("sha256") != caller_anchors["runtime_sha256"]
        or renderer_dependency.get("sha256") != caller_anchors["renderer_sha256"]
        or machine_dependency.get("sha256") != caller_anchors["machine_runner_sha256"]
    ):
        _fail(f"{context}.bundle dependency caller bindings drifted")
    renderer_member = renderer_dependency.get("member")
    machine_member = machine_dependency.get("member")
    if (
        renderer_member != f"analysis/render_tcga_revision_{pdf_id}.py"
        or machine_member != MACHINE_RUNNER_MEMBER
    ):
        _fail(f"{context}.bundle dependency member bindings drifted")

    plan_producer = _expect_mapping(
        plan_package["producer"],
        context=f"{context} plan producer",
    )
    producer = _expect_mapping(value["producer"], context=f"{context}.producer")
    _expect_keys(
        producer,
        {
            "member",
            "mode",
            "bytes",
            "sha256",
            "macho_uuid",
            "native_code_directory",
        },
        context=f"{context}.producer",
    )
    producer_bytes = _expect_positive_int(
        producer["bytes"],
        context=f"{context}.producer.bytes",
    )
    producer_sha256 = _expect_sha256(
        producer["sha256"],
        context=f"{context}.producer.sha256",
    )
    producer_macho_uuid = producer["macho_uuid"]
    if (
        producer_bytes > MAX_V2_NATIVE_EXECUTABLE_BYTES
        or not isinstance(producer_macho_uuid, str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            producer_macho_uuid,
        )
        is None
    ):
        _fail(f"{context}.producer bytes or Mach-O UUID is invalid")
    native_cd = _normalize_v2_code_directory(
        producer["native_code_directory"],
        producer_bytes=producer_bytes,
        context=f"{context}.producer.native_code_directory",
    )
    if (
        producer["member"] != PRODUCER_MEMBER_BY_ID[pdf_id]
        or producer["mode"] != V2_PRODUCER_MODE
        or producer_bytes != plan_producer["bytes"]
        or producer_sha256 != plan_producer["sha256"]
        or producer_macho_uuid != plan_producer["macho_uuid"]
        or native_cd != plan_producer["native_code_directory"]
    ):
        _fail(f"{context}.producer differs from the plan")

    launcher_source = _expect_mapping(
        value["launcher_source"],
        context=f"{context}.launcher_source",
    )
    _expect_keys(
        launcher_source,
        {"member", "encoding", "bytes", "sha256", "base64"},
        context=f"{context}.launcher_source",
    )
    if launcher_source["encoding"] != "base64" or not isinstance(
        launcher_source["base64"],
        str,
    ):
        _fail(f"{context}.launcher_source encoding drifted")
    try:
        launcher_raw = base64.b64decode(str(launcher_source["base64"]), validate=True)
    except (TypeError, ValueError) as error:
        _fail(f"{context}.launcher_source base64 is invalid: {error}")
    launcher_source_bytes = _expect_positive_int(
        launcher_source["bytes"],
        context=f"{context}.launcher_source.bytes",
    )
    if (
        launcher_source["member"] != f"analysis/native/{pdf_id}_derivation_launcher.c"
        or launcher_source_bytes > MAX_V2_NATIVE_LAUNCHER_SOURCE_BYTES
        or len(launcher_raw) != launcher_source_bytes
        or _sha256(launcher_raw) != launcher_source["sha256"]
        or launcher_source["sha256"] != caller_anchors["launcher_source_sha256"]
    ):
        _fail(f"{context}.launcher_source bytes or caller anchor drifted")

    launcher_config = _expect_mapping(
        value["launcher_config"],
        context=f"{context}.launcher_config",
    )
    _expect_keys(
        launcher_config,
        {
            "schema",
            "protocol",
            "role",
            "argument_count_including_argv0",
            "producer_arguments",
            "source_fd",
            "runtime",
            "renderer",
            "cwd",
            "environment",
            "environment_inherited",
            "shell",
            "process_operation",
            "unexpected_inherited_fds",
            "stdout",
            "stderr",
            "stdin",
            "failure_codes",
            "launcher_config_sha256",
        },
        context=f"{context}.launcher_config",
    )
    launcher_config_sha = _validate_projection_digest(
        launcher_config,
        digest_key="launcher_config_sha256",
        context=f"{context}.launcher_config",
    )
    expected_config = {
        "schema": f"dialect-revision-{pdf_id}-native-launcher-config-v2",
        "protocol": PRODUCER_PROTOCOL,
        "role": pdf_id,
        "argument_count_including_argv0": 9,
        "producer_arguments": _expected_arguments(pdf_id),
        "source_fd": launcher_config["source_fd"],
        "runtime": launcher_config["runtime"],
        "renderer": launcher_config["renderer"],
        "cwd": "/",
        "environment": EXACT_ENVIRONMENT,
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
    }
    if launcher_config_sha != plan_projection["launcher_config_sha256"] or any(
        not _json_exact_equal(launcher_config[key], expected)
        for key, expected in expected_config.items()
    ):
        _fail(f"{context}.launcher_config fixed contract drifted")
    source_fd = _expect_mapping(
        launcher_config.get("source_fd"),
        context=f"{context}.launcher_config.source_fd",
    )
    _expect_keys(
        source_fd,
        {
            "canonical_decimal",
            "minimum",
            "maximum",
            "access",
            "type",
            "mode",
            "owner",
            "link_count",
            "minimum_bytes",
            "maximum_bytes",
            "seekable_and_rewound",
            "cloexec",
        },
        context=f"{context}.launcher_config.source_fd",
    )
    source_fd_minimum = _expect_nonnegative_int(
        source_fd["minimum"],
        context=f"{context}.launcher_config.source_fd.minimum",
    )
    source_fd_maximum = _expect_positive_int(
        source_fd["maximum"],
        context=f"{context}.launcher_config.source_fd.maximum",
    )
    source_fd_link_count = _expect_positive_int(
        source_fd["link_count"],
        context=f"{context}.launcher_config.source_fd.link_count",
    )
    source_fd_minimum_bytes = _expect_positive_int(
        source_fd["minimum_bytes"],
        context=f"{context}.launcher_config.source_fd.minimum_bytes",
    )
    source_fd_maximum_bytes = _expect_positive_int(
        source_fd["maximum_bytes"],
        context=f"{context}.launcher_config.source_fd.maximum_bytes",
    )
    if (
        source_fd.get("canonical_decimal") is not True
        or source_fd_minimum != V2_SOURCE_DESCRIPTOR
        or source_fd_maximum != 2**31 - 1
        or source_fd.get("access") != "O_RDONLY; O_NONBLOCK permitted for regular file"
        or source_fd.get("type") != "regular"
        or source_fd.get("mode") != V2_SOURCE_MODE
        or source_fd.get("owner") != "effective-user-id"
        or source_fd_link_count != 1
        or source_fd_minimum_bytes != 1
        or source_fd_maximum_bytes != MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
        or source_fd.get("seekable_and_rewound") is not True
        or source_fd.get("cloexec") != "cleared-only-after-complete-validation"
    ):
        _fail(f"{context}.launcher_config.source_fd drifted")
    for key, anchor, maximum in (
        ("runtime", "runtime_sha256", MAX_V2_NATIVE_TOOL_BYTES),
        ("renderer", "renderer_sha256", MAX_V2_NATIVE_BUILDER_BYTES),
    ):
        record = _expect_mapping(
            launcher_config.get(key),
            context=f"{context}.launcher_config.{key}",
        )
        _expect_keys(
            record,
            {
                "locator",
                "absolute_path_recorded",
                "absolute_path_utf8_bytes",
                "absolute_path_utf8_sha256",
                "bytes",
                "sha256",
                "mode",
                "owner",
                "link_count",
                "pre_exec_descriptor_hash",
            },
            context=f"{context}.launcher_config.{key}",
        )
        record_bytes = _expect_positive_int(
            record["bytes"],
            context=f"{context}.launcher_config.{key}.bytes",
        )
        absolute_path_bytes = _expect_positive_int(
            record["absolute_path_utf8_bytes"],
            context=f"{context}.launcher_config.{key}.absolute_path_utf8_bytes",
        )
        record_link_count = _expect_positive_int(
            record["link_count"],
            context=f"{context}.launcher_config.{key}.link_count",
        )
        _expect_sha256(
            record["absolute_path_utf8_sha256"],
            context=f"{context}.launcher_config.{key}.absolute_path_utf8_sha256",
        )
        if (
            record.get("absolute_path_recorded") is not False
            or absolute_path_bytes > 4096
            or record.get("sha256") != caller_anchors[anchor]
            or record.get("owner") != "effective-user-id"
            or record_link_count != 1
            or record.get("pre_exec_descriptor_hash") is not True
        ):
            _fail(f"{context}.launcher_config.{key} binding drifted")
        dependency = runtime_dependency if key == "runtime" else renderer_dependency
        if (
            record_bytes > maximum
            or record_bytes != dependency.get("bytes")
            or record.get("sha256") != dependency.get("sha256")
            or record.get("locator") != dependency.get("locator")
            or not isinstance(record.get("mode"), str)
            or re.fullmatch(r"[0-7]{4}", str(record.get("mode"))) is None
            or int(str(record["mode"]), 8) & 0o022
            or (key == "runtime" and not int(str(record["mode"]), 8) & 0o100)
        ):
            _fail(f"{context}.launcher_config.{key} dependency or mode drifted")

    build = _expect_mapping(value["build"], context=f"{context}.build")
    _expect_keys(
        build,
        {
            "target",
            "environment",
            "inherit_environment",
            "cwd",
            "shell",
            "independent_build_count",
            "distinct_stage_roots_and_output_inodes",
            "recipes",
            "builds",
            "byte_identity",
            "ad_hoc_signature",
            "build_projection_sha256",
        },
        context=f"{context}.build",
    )
    build_sha = _validate_projection_digest(
        build,
        digest_key="build_projection_sha256",
        context=f"{context}.build",
    )
    authority_toolchain = _expect_mapping(
        value["toolchain"],
        context=f"{context}.toolchain",
    )
    sdk_version = authority_toolchain.get("sdk_version")
    if (
        not isinstance(sdk_version, str)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", sdk_version) is None
    ):
        _fail(f"{context}.toolchain.sdk_version is invalid")
    recipes = _v2_build_recipes(pdf_id)
    independent_build_count = _expect_positive_int(
        build["independent_build_count"],
        context=f"{context}.build.independent_build_count",
    )
    if (
        build_sha != plan_projection["build_projection_sha256"]
        or not _json_exact_equal(
            build["target"],
            {
                "architecture": "arm64",
                "platform": "macos",
                "minimum_version": "13.0",
                "sdk_version": sdk_version,
            },
        )
        or not _json_exact_equal(build["environment"], EXACT_ENVIRONMENT)
        or build["inherit_environment"] is not False
        or build["cwd"] != "/"
        or build["shell"] is not False
        or independent_build_count != 2
        or build["distinct_stage_roots_and_output_inodes"] is not True
        or not _json_exact_equal(build["recipes"], recipes)
        or not _json_exact_equal(
            build["byte_identity"],
            {
                "object": True,
                "unsigned": True,
                "signed": True,
                "native_code_directory": True,
            },
        )
    ):
        _fail(f"{context}.build double-build identity drifted")
    signature = _expect_mapping(
        build.get("ad_hoc_signature"),
        context=f"{context}.build.ad_hoc_signature",
    )
    _expect_keys(
        signature,
        {
            "identifier",
            "timestamp",
            "signer_identity_authenticated",
            "options",
            "code_directory_flags",
        },
        context=f"{context}.build.ad_hoc_signature",
    )
    if (
        signature["identifier"] != _v2_signature_identifier(pdf_id)
        or signature["timestamp"] != "none"
        or signature["signer_identity_authenticated"] is not False
        or signature["options"] != ["kill"]
        or signature["code_directory_flags"] != "0x00000202"
    ):
        _fail(f"{context}.build ad-hoc kill-signature contract drifted")
    build_records = _expect_sequence(build.get("builds"), context=f"{context}.builds")
    if len(build_records) != 2:
        _fail(f"{context}.builds must contain exactly A and B")
    normalized_builds: list[dict[str, object]] = []
    empty_sha256 = _sha256(b"")
    for expected_id, raw_build in zip(("a", "b"), build_records, strict=True):
        record = _expect_mapping(raw_build, context=f"{context}.build.{expected_id}")
        _expect_keys(
            record,
            {
                "build_id",
                "object_bytes",
                "object_sha256",
                "unsigned_bytes",
                "unsigned_sha256",
                "signed_bytes",
                "signed_sha256",
                "macho_uuid",
                "native_code_directory",
                "observations",
            },
            context=f"{context}.build.{expected_id}",
        )
        if (
            record["build_id"] != expected_id
            or record["signed_bytes"] != producer["bytes"]
            or record["signed_sha256"] != producer["sha256"]
            or record["macho_uuid"] != producer["macho_uuid"]
            or not _json_exact_equal(record["native_code_directory"], native_cd)
        ):
            _fail(f"{context}.build.{expected_id} differs from final producer")
        for field in ("object", "unsigned", "signed"):
            field_bytes = _expect_positive_int(
                record[f"{field}_bytes"],
                context=f"{context}.build.{expected_id}.{field}_bytes",
            )
            if field_bytes > MAX_V2_NATIVE_EXECUTABLE_BYTES:
                _fail(f"{context}.build.{expected_id}.{field}_bytes exceeds bound")
            _expect_sha256(
                record[f"{field}_sha256"],
                context=f"{context}.build.{expected_id}.{field}_sha256",
            )
        observations = _expect_mapping(
            record["observations"],
            context=f"{context}.build.{expected_id}.observations",
        )
        _expect_keys(
            observations,
            {"compile", "link", "sign", "verify"},
            context=f"{context}.build.{expected_id}.observations",
        )
        for operation in ("compile", "link", "sign", "verify"):
            observed = _expect_mapping(
                observations[operation],
                context=(f"{context}.build.{expected_id}.observations.{operation}"),
            )
            _expect_keys(
                observed,
                {
                    "return_code",
                    "stdout_bytes",
                    "stdout_sha256",
                    "stderr_bytes",
                    "stderr_sha256",
                    "normalized_argv",
                    "normalized_argv_sha256",
                },
                context=(f"{context}.build.{expected_id}.observations.{operation}"),
            )
            return_code = _expect_nonnegative_int(
                observed["return_code"],
                context=(
                    f"{context}.build.{expected_id}.observations."
                    f"{operation}.return_code"
                ),
            )
            stdout_bytes = _expect_nonnegative_int(
                observed["stdout_bytes"],
                context=(
                    f"{context}.build.{expected_id}.observations."
                    f"{operation}.stdout_bytes"
                ),
            )
            stderr_bytes = _expect_nonnegative_int(
                observed["stderr_bytes"],
                context=(
                    f"{context}.build.{expected_id}.observations."
                    f"{operation}.stderr_bytes"
                ),
            )
            expected_observation = {
                "return_code": 0,
                "stdout_bytes": 0,
                "stdout_sha256": empty_sha256,
                "stderr_bytes": 0,
                "stderr_sha256": empty_sha256,
                "normalized_argv": recipes[operation]["argv"],  # type: ignore[index]
                "normalized_argv_sha256": recipes[operation][  # type: ignore[index]
                    "argv_sha256"
                ],
            }
            if (
                return_code > 255
                or stdout_bytes > MAX_V2_NATIVE_TOOL_OUTPUT_BYTES
                or stderr_bytes > MAX_V2_NATIVE_TOOL_OUTPUT_BYTES
                or not _json_exact_equal(observed, expected_observation)
            ):
                _fail(
                    f"{context}.build.{expected_id}.observations.{operation} "
                    "is not exact empty success",
                )
        normalized_builds.append(dict(record))
    build_a = dict(normalized_builds[0])
    build_b = dict(normalized_builds[1])
    build_a.pop("build_id", None)
    build_b.pop("build_id", None)
    if not _json_exact_equal(build_a, build_b):
        _fail(f"{context} independent build records differ")

    toolchain = authority_toolchain
    _validate_v2_toolchain_projection(
        toolchain,
        caller_anchors=caller_anchors,
        context=f"{context}.toolchain",
    )

    source_release = _expect_mapping(
        value["source_release"],
        context=f"{context}.source_release",
    )
    _expect_keys(
        source_release,
        {
            "status",
            "release_commit",
            "release_ref",
            "git_blob_equality",
            "members",
            "git",
            "source_release_projection_sha256",
        },
        context=f"{context}.source_release",
    )
    _validate_projection_digest(
        source_release,
        digest_key="source_release_projection_sha256",
        context=f"{context}.source_release",
    )
    if (
        source_release.get("status")
        != "git-command-observed-listed-path-byte-equality-at-caller-commit"
        or source_release.get("git_blob_equality") is not True
        or not isinstance(source_release.get("release_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", str(source_release.get("release_commit")))
        is None
        or not isinstance(source_release.get("release_ref"), str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}",
            str(source_release.get("release_ref")),
        )
        is None
        or str(source_release["release_ref"]).startswith("-")
        or ".." in str(source_release["release_ref"])
        or "//" in str(source_release["release_ref"])
    ):
        _fail(f"{context}.source_release is not revision-bound")
    release_git = _expect_mapping(
        source_release.get("git"),
        context=f"{context}.source_release.git",
    )
    _expect_keys(
        release_git,
        {"locator", "bytes", "sha256", "main-executable-bytes-pinned"},
        context=f"{context}.source_release.git",
    )
    if (
        release_git["locator"] != "xcode-git"
        or release_git["sha256"] != caller_anchors["git_sha256"]
        or release_git["main-executable-bytes-pinned"] is not True
    ):
        _fail(f"{context}.source_release Git binding drifted")
    release_git_bytes = _expect_positive_int(
        release_git["bytes"],
        context=f"{context}.source_release.git.bytes",
    )
    if release_git_bytes > MAX_V2_NATIVE_TOOL_BYTES:
        _fail(f"{context}.source_release Git executable exceeds its byte bound")
    release_members = _expect_sequence(
        source_release.get("members"),
        context=f"{context}.source_release.members",
    )
    expected_release_members = [
        (
            f"analysis/native/{pdf_id}_derivation_launcher.c",
            caller_anchors["launcher_source_sha256"],
        ),
        (
            f"analysis/build_tcga_revision_{pdf_id}_native_producer.py",
            caller_anchors["builder_sha256"],
        ),
        (
            f"analysis/build_tcga_revision_{pdf_id}_derivation_bundle.py",
            caller_anchors["bundle_builder_sha256"],
        ),
        (str(renderer_member), caller_anchors["renderer_sha256"]),
        (str(machine_member), caller_anchors["machine_runner_sha256"]),
    ]
    if len(release_members) != len(expected_release_members):
        _fail(f"{context}.source_release has the wrong member count")
    observed_release_members: dict[str, str] = {}
    for index, ((expected_member, expected_digest), raw_member) in enumerate(
        zip(expected_release_members, release_members, strict=True),
    ):
        member = _expect_mapping(
            raw_member,
            context=f"{context}.source_release.members[{index}]",
        )
        _expect_keys(
            member,
            {"member", "bytes", "sha256"},
            context=f"{context}.source_release.members[{index}]",
        )
        member_name = member["member"]
        if (
            not isinstance(member_name, str)
            or member_name in observed_release_members
            or member_name != expected_member
        ):
            _fail(f"{context}.source_release contains an invalid duplicate member")
        release_member_bytes = _expect_positive_int(
            member["bytes"],
            context=f"{context}.source_release.members[{index}].bytes",
        )
        if release_member_bytes > MAX_V2_NATIVE_GIT_BLOB_BYTES:
            _fail(f"{context}.source_release member exceeds its byte bound")
        observed_digest = _expect_sha256(
            member["sha256"],
            context=f"{context}.source_release.members[{index}].sha256",
        )
        if observed_digest != expected_digest:
            _fail(f"{context}.source_release member-to-anchor map drifted")
        observed_release_members[member_name] = observed_digest
    if (
        list(observed_release_members.items()) != expected_release_members
        or observed_release_members[expected_release_members[0][0]]
        != launcher_source["sha256"]
    ):
        _fail(f"{context}.source_release member-to-anchor map drifted")

    handoff = _expect_mapping(
        value["runtime_handoff"],
        context=f"{context}.runtime_handoff",
    )
    _expect_keys(
        handoff,
        {
            "execve_path",
            "execve_argv",
            "placeholder_bindings",
            "cwd",
            "environment",
            "inherit_environment",
            "PATH_lookup",
            "shell",
            "stdout",
            "stderr",
            "source_fd",
            "runtime_handoff_sha256",
        },
        context=f"{context}.runtime_handoff",
    )
    handoff_sha = _validate_projection_digest(
        handoff,
        digest_key="runtime_handoff_sha256",
        context=f"{context}.runtime_handoff",
    )
    expected_handoff_argv = [
        "{runtime}",
        "-I",
        "-S",
        "-B",
        "{renderer}",
        *_expected_arguments(pdf_id),
    ]
    if (
        handoff_sha != plan_projection["runtime_handoff_sha256"]
        or handoff.get("execve_path") != "{runtime}"
        or not _json_exact_equal(handoff.get("execve_argv"), expected_handoff_argv)
        or not _json_exact_equal(
            handoff.get("placeholder_bindings"),
            {
                "{runtime}": launcher_config["runtime"],
                "{renderer}": launcher_config["renderer"],
                "{source_fd}": ("validated-original-canonical-decimal-descriptor"),
            },
        )
        or handoff.get("cwd") != "/"
        or not _json_exact_equal(handoff.get("environment"), EXACT_ENVIRONMENT)
        or handoff.get("inherit_environment") is not False
        or handoff.get("PATH_lookup") is not False
        or handoff.get("shell") is not False
        or handoff.get("stdout") != "inherited"
        or handoff.get("stderr") != "inherited"
        or not _json_exact_equal(handoff.get("source_fd"), source_fd)
    ):
        _fail(f"{context}.runtime_handoff drifted")

    expected_output = _expect_mapping(
        value["expected_output"],
        context=f"{context}.expected_output",
    )
    _expect_keys(
        expected_output,
        {"renderer_manifest", "pdf"},
        context=f"{context}.expected_output",
    )
    normalized_expected_output = {
        "renderer_manifest": _normalize_v2_output_record(
            expected_output["renderer_manifest"],
            context=f"{context}.expected_output.renderer_manifest",
            expected_member="render-receipt.json",
            expected_sha256=caller_anchors["renderer_manifest_sha256"],
        ),
        "pdf": _normalize_v2_output_record(
            expected_output["pdf"],
            context=f"{context}.expected_output.pdf",
            expected_member=PDF_MEMBER_BY_ID[pdf_id],
            expected_sha256=caller_anchors["pdf_sha256"],
        ),
    }
    bundle_output_matches = _json_exact_equal(
        normalized_expected_output,
        projection["expected_output"],
    )
    plan_output_matches = _json_exact_equal(
        normalized_expected_output,
        document["expected_output"],
    )
    if not bundle_output_matches or not plan_output_matches:
        _fail(f"{context}.expected_output differs across authority, bundle, and plan")
    return dict(
        _expect_mapping(
            _canonical_copy(value),
            context=f"{context} normalized copy",
        ),
    )


def _pin_file(path: Path, *, maximum: int, context: str) -> object:
    return _machine_call(_machine._pin_file, path, maximum=maximum, context=context)


def _pin_root(path: Path, *, context: str) -> object:
    return _machine_call(_machine._pin_root, path, context=context)


def _open_root_member(
    root: object,
    member: str,
    *,
    maximum: int,
    expected_size: int,
    context: str,
) -> object:
    return _machine_call(
        _machine._open_root_member,
        root,
        member,
        maximum=maximum,
        expected_size=expected_size,
        context=context,
    )


def _validate_exact_inventory(
    root: object,
    *,
    expected: Sequence[str],
    context: str,
    maximum_each: int,
    maximum_total: int,
) -> dict[str, int]:
    return _machine_call(
        _machine._validate_exact_root_inventory,
        root,
        expected=expected,
        context=context,
        maximum_each=maximum_each,
        maximum_total=maximum_total,
    )  # type: ignore[return-value]


def _validate_path_topology(
    plan_path: Path,
    source_root: Path,
    producer_root: Path,
    authority_root: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    source = _machine_call(
        _machine._canonical_existing_directory,
        source_root,
        context="source-bundle root",
    )
    producer = _machine_call(
        _machine._canonical_existing_directory,
        producer_root,
        context="producer root",
    )
    authority = _machine_call(
        _machine._canonical_existing_directory,
        authority_root,
        context="producer-toolchain authority root",
    )
    output = output_root.absolute()
    _machine_call(
        _machine._assert_distinct_roots,
        (source, "source-bundle root"),
        (producer, "producer root"),
        (authority, "producer-toolchain authority root"),
        (output, "derivation output root"),
    )
    plan = plan_path.absolute()
    for root, context in (
        (source, "source-bundle root"),
        (producer, "producer root"),
        (authority, "producer-toolchain authority root"),
        (output, "derivation output root"),
    ):
        if plan == root or plan.is_relative_to(root):
            _fail(f"plan path must be outside the {context}")
    return source, producer, authority


def _pin_inputs(
    plan_path: Path,
    source_root_path: Path,
    producer_root_path: Path,
    authority_root_path: Path,
    output_root: Path,
    *,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> _Inputs:
    source_path, producer_path, authority_path = _validate_path_topology(
        plan_path,
        source_root_path,
        producer_root_path,
        authority_root_path,
        output_root,
    )
    plan = builder = machine_runner = source_root = producer_root = authority_root = (
        None
    )
    sources: dict[str, object] = {}
    producers: dict[str, object] = {}
    authorities: dict[str, object] = {}
    try:
        plan = _pin_file(plan_path, maximum=MAX_PLAN_BYTES, context="derivation plan")
        builder = _pin_file(
            Path(__file__),
            maximum=MAX_PLAN_BYTES,
            context="live derivation builder",
        )
        machine_runner = _pin_file(
            Path(_machine.__file__),
            maximum=MAX_PLAN_BYTES,
            context="live native execution dependency",
        )
        if plan.sha256 != _expect_sha256(
            expected_plan_sha256,
            context="expected plan SHA-256",
        ):
            _fail("derivation plan does not match its caller SHA-256 anchor")
        if builder.sha256 != _expect_sha256(
            expected_builder_sha256,
            context="expected builder SHA-256",
        ):
            _fail("live derivation builder does not match its caller SHA-256 anchor")
        if machine_runner.sha256 != _expect_sha256(
            expected_machine_runner_sha256,
            context="expected machine-runner SHA-256",
        ):
            _fail(
                "native execution dependency does not match its caller SHA-256 anchor",
            )
        normalized = _normalize_plan(
            _parse_canonical_json(
                plan,
                maximum=MAX_PLAN_BYTES,
                context="derivation plan",
                trailing_newline=True,
            ),
            release_id=release_id,
            expected_upstream_sha256=expected_upstream_sha256,
            expected_authority_sha256=expected_authority_sha256,
            expected_source_sha256=expected_source_sha256,
            expected_producer_sha256=expected_producer_sha256,
            expected_pdf_sha256=expected_pdf_sha256,
        )
        source_root = _pin_root(source_path, context="source-bundle root")
        producer_root = _pin_root(producer_path, context="producer root")
        authority_root = _pin_root(
            authority_path,
            context="producer-toolchain authority root",
        )
        source_sizes = _validate_exact_inventory(
            source_root,
            expected=tuple(SOURCE_MEMBER_BY_ID.values()),
            context="source-bundle root",
            maximum_each=MAX_SOURCE_BUNDLE_BYTES,
            maximum_total=MAX_TOTAL_SOURCE_BYTES,
        )
        producer_sizes = _validate_exact_inventory(
            producer_root,
            expected=tuple(PRODUCER_MEMBER_BY_ID.values()),
            context="producer root",
            maximum_each=MAX_PRODUCER_BYTES,
            maximum_total=MAX_TOTAL_PRODUCER_BYTES,
        )
        authority_sizes = _validate_exact_inventory(
            authority_root,
            expected=tuple(AUTHORITY_MEMBER_BY_ID.values()),
            context="producer-toolchain authority root",
            maximum_each=MAX_AUTHORITY_RECEIPT_BYTES,
            maximum_total=MAX_TOTAL_AUTHORITY_BYTES,
        )
        document_by_id = {
            str(item["pdf_id"]): item
            for item in normalized["documents"]  # type: ignore[union-attr]
        }
        for pdf_id in PDF_IDS:
            document = document_by_id[pdf_id]
            source_member = SOURCE_MEMBER_BY_ID[pdf_id]
            producer_member = PRODUCER_MEMBER_BY_ID[pdf_id]
            authority_member = AUTHORITY_MEMBER_BY_ID[pdf_id]
            source = _open_root_member(
                source_root,
                source_member,
                maximum=MAX_SOURCE_BUNDLE_BYTES,
                expected_size=source_sizes[source_member],
                context=f"source bundle {pdf_id}",
            )
            sources[pdf_id] = source
            producer = _open_root_member(
                producer_root,
                producer_member,
                maximum=MAX_PRODUCER_BYTES,
                expected_size=producer_sizes[producer_member],
                context=f"producer {pdf_id}",
            )
            producers[pdf_id] = producer
            authority = _open_root_member(
                authority_root,
                authority_member,
                maximum=MAX_AUTHORITY_RECEIPT_BYTES,
                expected_size=authority_sizes[authority_member],
                context=f"producer-toolchain authority {pdf_id}",
            )
            authorities[pdf_id] = authority
            mode = stat.S_IMODE(os.fstat(producer.descriptor).st_mode)
            if mode & 0o111 == 0 or mode & 0o022:
                _fail(
                    f"producer {pdf_id} must be executable and not group/other "
                    "writable",
                )
            if (
                source.size != document["source_bundle_bytes"]
                or source.sha256 != document["source_bundle_sha256"]
            ):
                _fail(f"source bundle {pdf_id} differs from the plan")
            if (
                producer.size != document["producer_bytes"]
                or producer.sha256 != document["producer_sha256"]
            ):
                _fail(f"producer {pdf_id} differs from the plan")
            adapter_authority = _expect_mapping(
                document["adapter_authority"],
                context=f"producer {pdf_id} adapter authority",
            )
            authority_sha256 = adapter_authority["authority_receipt_sha256"]
            if authority.sha256 != authority_sha256:
                _fail(
                    f"producer-toolchain authority {pdf_id} differs from its "
                    "caller and plan SHA-256 anchors",
                )
            _normalize_authority_receipt(
                _parse_canonical_json(
                    authority,
                    maximum=MAX_AUTHORITY_RECEIPT_BYTES,
                    context=f"producer-toolchain authority {pdf_id}",
                    trailing_newline=True,
                ),
                mode=str(normalized["mode"]),
                release_id=release_id,
                pdf_id=pdf_id,
                document=document,
            )
        if len({(pin.device, pin.inode) for pin in sources.values()}) != len(PDF_ORDER):
            _fail("source-bundle roles must use distinct single-link files")
        if len({(pin.device, pin.inode) for pin in producers.values()}) != len(
            PDF_ORDER,
        ):
            _fail("producer roles must use distinct single-link files")
        if len({(pin.device, pin.inode) for pin in authorities.values()}) != len(
            PDF_ORDER,
        ):
            _fail("producer-toolchain authorities must use distinct single-link files")
        return _Inputs(
            plan=plan,
            builder=builder,
            machine_runner=machine_runner,
            source_root=source_root,
            producer_root=producer_root,
            authority_root=authority_root,
            sources=sources,
            producers=producers,
            authorities=authorities,
            normalized_plan=normalized,
        )
    except BaseException:
        _close_resources(
            [
                *[
                    (f"producer authority {key}", value)
                    for key, value in authorities.items()
                ],
                *[(f"producer {key}", value) for key, value in producers.items()],
                *[(f"source bundle {key}", value) for key, value in sources.items()],
                ("producer authority root", authority_root),
                ("producer root", producer_root),
                ("source-bundle root", source_root),
                ("native execution dependency", machine_runner),
                ("derivation builder", builder),
                ("derivation plan", plan),
            ],
            context="failed pinned derivation inputs",
        )
        raise


def _validate_v2_path_topology(
    plan_path: Path,
    source_root: Path,
    package_roots: Mapping[str, Path],
    output_root: Path,
) -> tuple[Path, dict[str, Path]]:
    if set(package_roots) != set(PDF_IDS):
        _fail(f"v2 package roots must name exactly {list(PDF_IDS)}")
    source = _machine_call(
        _machine._canonical_existing_directory,
        source_root,
        context="v2 source-bundle root",
    )
    packages = {
        pdf_id: _machine_call(
            _machine._canonical_existing_directory,
            package_roots[pdf_id],
            context=f"v2 producer package root {pdf_id}",
        )
        for pdf_id in PDF_IDS
    }
    output = output_root.absolute()
    _machine_call(
        _machine._assert_distinct_roots,
        (source, "v2 source-bundle root"),
        *[
            (packages[pdf_id], f"v2 producer package root {pdf_id}")
            for pdf_id in PDF_IDS
        ],
        (output, "v2 derivation output root"),
    )
    plan = plan_path.absolute()
    for root, context in (
        (source, "v2 source-bundle root"),
        *[
            (packages[pdf_id], f"v2 producer package root {pdf_id}")
            for pdf_id in PDF_IDS
        ],
        (output, "v2 derivation output root"),
    ):
        if plan == root or plan.is_relative_to(root):
            _fail(f"v2 plan path must be outside the {context}")
    return source, packages


def _validate_v2_member_mode(
    pinned: object,
    *,
    expected_mode: int,
    context: str,
) -> None:
    entry = os.fstat(pinned.descriptor)
    if (
        stat.S_IMODE(entry.st_mode) != expected_mode
        or entry.st_uid != os.geteuid()
        or entry.st_nlink != 1
        or not stat.S_ISREG(entry.st_mode)
    ):
        _fail(
            f"{context} must be an effective-user-owned single-link regular "
            f"file with mode {expected_mode:04o}",
        )


def _validate_v2_source_descriptor(source: object, *, context: str) -> None:
    flags = fcntl.fcntl(source.descriptor, fcntl.F_GETFL)
    descriptor_flags = fcntl.fcntl(source.descriptor, fcntl.F_GETFD)
    if flags & os.O_ACCMODE != os.O_RDONLY:
        _fail(f"{context} descriptor must be read-only")
    if getattr(os, "O_NONBLOCK", 0) and not flags & os.O_NONBLOCK:
        _fail(f"{context} descriptor must exercise the outer O_NONBLOCK pin")
    if not descriptor_flags & fcntl.FD_CLOEXEC:
        _fail(f"{context} outer descriptor must remain close-on-exec before remap")
    try:
        os.lseek(source.descriptor, 0, os.SEEK_SET)
    except OSError as error:
        _fail(f"{context} descriptor must be seekable and rewound: {error}")


def _validate_v2_root_mode(root: object, *, context: str) -> None:
    entry = os.fstat(root.descriptor)
    if stat.S_IMODE(entry.st_mode) != 0o500 or entry.st_uid != os.geteuid():
        _fail(f"{context} must be effective-user-owned and sealed mode 0500")


def _parse_v2_producer_code_directory(producer: object) -> dict[str, object]:
    parsed = _machine_call(_machine._parse_arm64_code_directory, producer)
    return dict(
        _expect_mapping(parsed, context="fresh producer CodeDirectory parse"),
    )


def _parse_v2_macho_uuid(producer: object) -> str:
    raw = _pinned_bytes(
        producer,
        maximum=MAX_V2_NATIVE_EXECUTABLE_BYTES,
        context="v2 native producer UUID parse",
    )
    if len(raw) < 32:
        _fail("v2 native producer lacks a complete Mach-O header")
    try:
        magic, cpu_type, _subtype, file_type, count, size, _flags, _reserved = (
            struct.unpack_from("<IiiIIIII", raw)
        )
    except struct.error as error:  # pragma: no cover - guarded by size.
        _fail(f"cannot parse v2 native producer Mach-O header: {error}")
    if magic != 0xFEEDFACF or cpu_type != 0x0100000C or file_type != 2:
        _fail("v2 native producer must be a thin arm64 Mach-O executable")
    command_end = 32 + size
    if not 1 <= count <= 512 or command_end > len(raw) or size < count * 8:
        _fail("v2 native producer load-command extent is invalid")
    values: list[bytes] = []
    offset = 32
    for _ in range(count):
        if offset + 8 > command_end:
            _fail("v2 native producer load-command table is truncated")
        command, command_size = struct.unpack_from("<II", raw, offset)
        if command_size < 8 or command_size % 8 or offset + command_size > command_end:
            _fail("v2 native producer contains an invalid load command")
        if command == 0x1B:
            if command_size != 24:
                _fail("v2 native producer LC_UUID command has the wrong size")
            values.append(raw[offset + 8 : offset + 24])
        offset += command_size
    if offset != command_end or len(values) != 1 or values[0] == b"\0" * 16:
        _fail("v2 native producer must contain exactly one nonzero LC_UUID")
    hexadecimal = values[0].hex()
    return "-".join(
        (
            hexadecimal[:8],
            hexadecimal[8:12],
            hexadecimal[12:16],
            hexadecimal[16:20],
            hexadecimal[20:],
        ),
    )


def _parse_v2_code_directory_flags(producer: object) -> int:
    raw = _pinned_bytes(
        producer,
        maximum=MAX_V2_NATIVE_EXECUTABLE_BYTES,
        context="v2 native producer CodeDirectory flags parse",
    )
    if len(raw) < 32:
        _fail("v2 native producer lacks a complete Mach-O header")
    magic, _cpu, _subtype, _type, count, size, _flags, _reserved = struct.unpack_from(
        "<IiiIIIII",
        raw,
    )
    command_end = 32 + size
    if magic != 0xFEEDFACF or not 1 <= count <= 512 or command_end > len(raw):
        _fail("v2 native producer Mach-O command table is invalid")
    signature_extent: tuple[int, int] | None = None
    offset = 32
    for _ in range(count):
        command, command_size = struct.unpack_from("<II", raw, offset)
        if command_size < 8 or offset + command_size > command_end:
            _fail("v2 native producer contains an invalid load command")
        if command == 0x1D:
            if signature_extent is not None or command_size != 16:
                _fail("v2 native producer code-signature command is not unique")
            signature_extent = struct.unpack_from("<II", raw, offset + 8)
        offset += command_size
    if offset != command_end or signature_extent is None:
        _fail("v2 native producer lacks one code-signature command")
    signature_offset, signature_bytes = signature_extent
    if signature_offset + signature_bytes > len(raw) or signature_bytes < 20:
        _fail("v2 native producer signature extent is invalid")
    signature = raw[signature_offset : signature_offset + signature_bytes]
    signature_magic, signature_length, blob_count = struct.unpack_from(
        ">III",
        signature,
    )
    if (
        signature_magic != 0xFADE0CC0
        or signature_length > len(signature)
        or not 1 <= blob_count <= 32
        or 12 + blob_count * 8 > signature_length
    ):
        _fail("v2 native producer signature SuperBlob is invalid")
    code_directories: list[bytes] = []
    for index in range(blob_count):
        slot, blob_offset = struct.unpack_from(">II", signature, 12 + index * 8)
        if blob_offset + 16 > signature_length:
            _fail("v2 native producer signature blob offset is invalid")
        blob_magic, blob_length = struct.unpack_from(">II", signature, blob_offset)
        if blob_offset + blob_length > signature_length:
            _fail("v2 native producer signature blob extent is invalid")
        if slot == 0 and blob_magic == 0xFADE0C02:
            code_directories.append(signature[blob_offset : blob_offset + blob_length])
    if len(code_directories) != 1:
        _fail("v2 native producer must contain one primary CodeDirectory")
    flags = struct.unpack_from(">I", code_directories[0], 12)[0]
    if flags != 514:
        _fail("v2 native producer CodeDirectory flags must be adhoc|kill (0x202)")
    return flags


def _pin_v2_inputs(
    plan_path: Path,
    source_root_path: Path,
    package_root_paths: Mapping[str, Path],
    output_root: Path,
    *,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_package_sha256: Mapping[str, str],
    expected_renderer_manifest_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> _V2Inputs:
    source_path, package_paths = _validate_v2_path_topology(
        plan_path,
        source_root_path,
        package_root_paths,
        output_root,
    )
    plan = builder = machine_runner = source_root = None
    package_roots: dict[str, object] = {}
    sources: dict[str, object] = {}
    producers: dict[str, object] = {}
    authorities: dict[str, object] = {}
    normalized_authorities: dict[str, dict[str, object]] = {}
    package_records: dict[str, dict[str, object]] = {}
    try:
        plan = _pin_file(
            plan_path,
            maximum=MAX_PLAN_BYTES,
            context="v2 derivation plan",
        )
        builder = _pin_file(
            Path(__file__),
            maximum=MAX_PLAN_BYTES,
            context="live v2 derivation builder",
        )
        machine_runner = _pin_file(
            Path(_machine.__file__),
            maximum=MAX_PLAN_BYTES,
            context="live v2 native execution dependency",
        )
        if plan.sha256 != _expect_sha256(
            expected_plan_sha256,
            context="expected v2 plan SHA-256",
        ):
            _fail("v2 derivation plan differs from its caller SHA-256 anchor")
        if builder.sha256 != _expect_sha256(
            expected_builder_sha256,
            context="expected v2 builder SHA-256",
        ):
            _fail("live v2 derivation builder differs from its caller anchor")
        if machine_runner.sha256 != _expect_sha256(
            expected_machine_runner_sha256,
            context="expected v2 machine-runner SHA-256",
        ):
            _fail("live v2 execution dependency differs from its caller anchor")
        normalized = _normalize_v2_plan(
            _parse_canonical_json(
                plan,
                maximum=MAX_PLAN_BYTES,
                context="v2 derivation plan",
                trailing_newline=True,
            ),
            release_id=release_id,
            expected_upstream_sha256=expected_upstream_sha256,
            expected_authority_sha256=expected_authority_sha256,
            expected_source_sha256=expected_source_sha256,
            expected_producer_sha256=expected_producer_sha256,
            expected_package_sha256=expected_package_sha256,
            expected_renderer_manifest_sha256=expected_renderer_manifest_sha256,
            expected_pdf_sha256=expected_pdf_sha256,
        )
        documents = {
            str(item["pdf_id"]): _expect_mapping(
                item,
                context="normalized v2 plan document",
            )
            for item in normalized["documents"]  # type: ignore[union-attr]
        }
        for pdf_id in PDF_IDS:
            caller_anchors = _expect_mapping(
                documents[pdf_id]["caller_anchors"],
                context=f"v2 plan caller anchors {pdf_id}",
            )
            if caller_anchors["machine_runner_sha256"] != machine_runner.sha256:
                _fail(
                    f"v2 plan {pdf_id} authority does not bind the live machine runner",
                )

        source_root = _pin_root(source_path, context="v2 source-bundle root")
        _validate_v2_root_mode(source_root, context="v2 source-bundle root")
        source_sizes = _validate_exact_inventory(
            source_root,
            expected=tuple(SOURCE_MEMBER_BY_ID.values()),
            context="v2 source-bundle root",
            maximum_each=MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES,
            maximum_total=MAX_TOTAL_SOURCE_BYTES,
        )
        authority_total = 0
        for pdf_id in PDF_IDS:
            document = documents[pdf_id]
            plan_source = _expect_mapping(
                document["source_bundle"],
                context=f"v2 plan source {pdf_id}",
            )
            source_member = SOURCE_MEMBER_BY_ID[pdf_id]
            source = _open_root_member(
                source_root,
                source_member,
                maximum=MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES,
                expected_size=source_sizes[source_member],
                context=f"v2 source bundle {pdf_id}",
            )
            sources[pdf_id] = source
            _validate_v2_member_mode(
                source,
                expected_mode=0o400,
                context=f"v2 source bundle {pdf_id}",
            )
            _validate_v2_source_descriptor(
                source,
                context=f"v2 source bundle {pdf_id}",
            )
            if (
                source.size != plan_source["bytes"]
                or source.sha256 != plan_source["sha256"]
            ):
                _fail(f"v2 source bundle {pdf_id} differs from the plan")

            package_root = _pin_root(
                package_paths[pdf_id],
                context=f"v2 producer package root {pdf_id}",
            )
            package_roots[pdf_id] = package_root
            _validate_v2_root_mode(
                package_root,
                context=f"v2 producer package root {pdf_id}",
            )
            package_sizes = _validate_exact_inventory(
                package_root,
                expected=(
                    PRODUCER_MEMBER_BY_ID[pdf_id],
                    AUTHORITY_MEMBER_BY_ID[pdf_id],
                ),
                context=f"v2 producer package root {pdf_id}",
                maximum_each=max(
                    MAX_V2_NATIVE_EXECUTABLE_BYTES,
                    MAX_V2_AUTHORITY_BYTES,
                ),
                maximum_total=(MAX_V2_NATIVE_EXECUTABLE_BYTES + MAX_V2_AUTHORITY_BYTES),
            )
            producer_member = PRODUCER_MEMBER_BY_ID[pdf_id]
            authority_member = AUTHORITY_MEMBER_BY_ID[pdf_id]
            producer = _open_root_member(
                package_root,
                producer_member,
                maximum=MAX_V2_NATIVE_EXECUTABLE_BYTES,
                expected_size=package_sizes[producer_member],
                context=f"v2 producer {pdf_id}",
            )
            producers[pdf_id] = producer
            _validate_v2_member_mode(
                producer,
                expected_mode=0o500,
                context=f"v2 producer {pdf_id}",
            )
            authority = _open_root_member(
                package_root,
                authority_member,
                maximum=MAX_V2_AUTHORITY_BYTES,
                expected_size=package_sizes[authority_member],
                context=f"v2 native producer authority {pdf_id}",
            )
            authorities[pdf_id] = authority
            authority_total += authority.size
            _validate_v2_member_mode(
                authority,
                expected_mode=0o400,
                context=f"v2 native producer authority {pdf_id}",
            )
            plan_package = _expect_mapping(
                document["producer_package"],
                context=f"v2 plan package {pdf_id}",
            )
            plan_producer = _expect_mapping(
                plan_package["producer"],
                context=f"v2 plan producer {pdf_id}",
            )
            plan_authority = _expect_mapping(
                plan_package["authority"],
                context=f"v2 plan authority {pdf_id}",
            )
            if (
                producer.size != plan_producer["bytes"]
                or producer.sha256 != plan_producer["sha256"]
                or authority.size != plan_authority["bytes"]
                or authority.sha256 != plan_authority["sha256"]
            ):
                _fail(f"v2 producer package {pdf_id} differs from the plan")
            authority_raw = _pinned_bytes(
                authority,
                maximum=MAX_V2_AUTHORITY_BYTES,
                context=f"v2 native producer authority {pdf_id}",
            )
            _reject_v2_forbidden_fragments(
                authority_raw,
                context=f"v2 authority {pdf_id}",
            )
            normalized_authorities[pdf_id] = _normalize_v2_authority(
                _parse_canonical_json(
                    authority,
                    maximum=MAX_V2_AUTHORITY_BYTES,
                    context=f"v2 native producer authority {pdf_id}",
                    trailing_newline=True,
                ),
                release_id=release_id,
                pdf_id=pdf_id,
                document=document,
            )
            fresh_code_directory = _parse_v2_producer_code_directory(producer)
            fresh_macho_uuid = _parse_v2_macho_uuid(producer)
            fresh_code_directory_flags = _parse_v2_code_directory_flags(producer)
            expected_code_directory = _expect_mapping(
                plan_producer["native_code_directory"],
                context=f"v2 plan producer CodeDirectory {pdf_id}",
            )
            if fresh_code_directory != _v2_code_directory_core(
                expected_code_directory,
            ):
                _fail(
                    f"v2 producer {pdf_id} fresh CodeDirectory parse differs from "
                    "its authority and plan",
                )
            if fresh_macho_uuid != plan_producer["macho_uuid"]:
                _fail(
                    f"v2 producer {pdf_id} fresh LC_UUID parse differs from its "
                    "authority and plan",
                )
            if (
                fresh_code_directory_flags
                != expected_code_directory["code_directory_flags"]
            ):
                _fail(
                    f"v2 producer {pdf_id} CodeDirectory flags differ from its "
                    "authority and plan",
                )
            package_record = _v2_package_content_record(
                pdf_id,
                producer_bytes=producer.size,
                producer_sha256=producer.sha256,
                authority_bytes=authority.size,
                authority_sha256=authority.sha256,
            )
            package_digest = _sha256(_canonical_json(package_record))
            if package_digest != plan_package["package_content_sha256"]:
                _fail(f"v2 producer package {pdf_id} content digest drifted")
            package_records[pdf_id] = {
                "pdf_id": pdf_id,
                "package_content_sha256": package_digest,
                **package_record,
            }
        if authority_total > MAX_TOTAL_V2_AUTHORITY_BYTES:
            _fail("v2 producer authorities exceed their aggregate bound")
        for label, pins in (
            ("source bundles", sources),
            ("producers", producers),
            ("authorities", authorities),
            ("package roots", package_roots),
        ):
            identities = {(pin.device, pin.inode) for pin in pins.values()}
            if len(identities) != len(PDF_ORDER):
                _fail(f"v2 {label} must have distinct role identities")
        return _V2Inputs(
            plan=plan,
            builder=builder,
            machine_runner=machine_runner,
            source_root=source_root,
            package_roots=package_roots,
            sources=sources,
            producers=producers,
            authorities=authorities,
            normalized_authorities=normalized_authorities,
            normalized_plan=normalized,
            package_records=package_records,
        )
    except BaseException:
        _close_resources(
            [
                *[(f"v2 authority {key}", pin) for key, pin in authorities.items()],
                *[(f"v2 producer {key}", pin) for key, pin in producers.items()],
                *[(f"v2 source {key}", pin) for key, pin in sources.items()],
                *[
                    (f"v2 package root {key}", pin)
                    for key, pin in package_roots.items()
                ],
                ("v2 source root", source_root),
                ("v2 machine runner", machine_runner),
                ("v2 builder", builder),
                ("v2 plan", plan),
            ],
            context="failed v2 pinned derivation inputs",
        )
        raise


def _revalidate_v2_inputs(inputs: _V2Inputs) -> None:
    _machine_call(
        _machine._revalidate_root,
        inputs.source_root,
        context="v2 source-bundle root",
    )
    _validate_v2_root_mode(inputs.source_root, context="v2 source-bundle root")
    _validate_exact_inventory(
        inputs.source_root,
        expected=tuple(SOURCE_MEMBER_BY_ID.values()),
        context="v2 source-bundle root",
        maximum_each=MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES,
        maximum_total=MAX_TOTAL_SOURCE_BYTES,
    )
    for pinned, context in (
        (inputs.plan, "v2 derivation plan"),
        (inputs.builder, "live v2 derivation builder"),
        (inputs.machine_runner, "live v2 native execution dependency"),
    ):
        _machine_call(_machine._revalidate_file, pinned, context=context)
    for pdf_id in PDF_IDS:
        package_root = inputs.package_roots[pdf_id]
        _machine_call(
            _machine._revalidate_root,
            package_root,
            context=f"v2 producer package root {pdf_id}",
        )
        _validate_v2_root_mode(
            package_root,
            context=f"v2 producer package root {pdf_id}",
        )
        _validate_exact_inventory(
            package_root,
            expected=(
                PRODUCER_MEMBER_BY_ID[pdf_id],
                AUTHORITY_MEMBER_BY_ID[pdf_id],
            ),
            context=f"v2 producer package root {pdf_id}",
            maximum_each=max(
                MAX_V2_NATIVE_EXECUTABLE_BYTES,
                MAX_V2_AUTHORITY_BYTES,
            ),
            maximum_total=(MAX_V2_NATIVE_EXECUTABLE_BYTES + MAX_V2_AUTHORITY_BYTES),
        )
        for pinned, context, mode in (
            (inputs.sources[pdf_id], f"v2 source bundle {pdf_id}", 0o400),
            (inputs.producers[pdf_id], f"v2 producer {pdf_id}", 0o500),
            (inputs.authorities[pdf_id], f"v2 authority {pdf_id}", 0o400),
        ):
            _machine_call(_machine._revalidate_file, pinned, context=context)
            _validate_v2_member_mode(pinned, expected_mode=mode, context=context)
        _validate_v2_source_descriptor(
            inputs.sources[pdf_id],
            context=f"v2 source bundle {pdf_id}",
        )
        fresh = _parse_v2_producer_code_directory(inputs.producers[pdf_id])
        fresh_macho_uuid = _parse_v2_macho_uuid(inputs.producers[pdf_id])
        fresh_code_directory_flags = _parse_v2_code_directory_flags(
            inputs.producers[pdf_id],
        )
        document = _expect_mapping(
            inputs.normalized_plan["documents"][PDF_IDS.index(pdf_id)],  # type: ignore[index]
            context=f"normalized v2 document {pdf_id}",
        )
        package = _expect_mapping(
            document["producer_package"],
            context=f"normalized v2 package {pdf_id}",
        )
        producer = _expect_mapping(
            package["producer"],
            context=f"normalized v2 producer {pdf_id}",
        )
        code_directory = _expect_mapping(
            producer["native_code_directory"],
            context=f"normalized v2 CodeDirectory {pdf_id}",
        )
        if fresh != _v2_code_directory_core(code_directory):
            _fail(f"v2 producer {pdf_id} CodeDirectory changed after validation")
        if fresh_macho_uuid != producer["macho_uuid"]:
            _fail(f"v2 producer {pdf_id} LC_UUID changed after validation")
        if fresh_code_directory_flags != code_directory["code_directory_flags"]:
            _fail(f"v2 producer {pdf_id} CodeDirectory flags changed")


def _validate_v2_fd_headroom(inputs: _V2Inputs) -> None:
    descriptor_root = Path("/dev/fd")
    try:
        current = len(tuple(descriptor_root.iterdir()))
    except OSError as error:
        _fail(f"cannot establish the current open-descriptor count: {error}")
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return
    owned = 4 + len(inputs.sources) + len(inputs.producers) + len(inputs.authorities)
    owned += len(inputs.package_roots)
    required = current + owned + 16
    if required > soft:
        _fail(f"RLIMIT_NOFILE headroom is insufficient: need {required}, have {soft}")


def _revalidate_inputs(inputs: _Inputs) -> None:
    for root, context, expected, maximum_each, maximum_total in (
        (
            inputs.source_root,
            "source-bundle root",
            tuple(SOURCE_MEMBER_BY_ID.values()),
            MAX_SOURCE_BUNDLE_BYTES,
            MAX_TOTAL_SOURCE_BYTES,
        ),
        (
            inputs.producer_root,
            "producer root",
            tuple(PRODUCER_MEMBER_BY_ID.values()),
            MAX_PRODUCER_BYTES,
            MAX_TOTAL_PRODUCER_BYTES,
        ),
        (
            inputs.authority_root,
            "producer-toolchain authority root",
            tuple(AUTHORITY_MEMBER_BY_ID.values()),
            MAX_AUTHORITY_RECEIPT_BYTES,
            MAX_TOTAL_AUTHORITY_BYTES,
        ),
    ):
        _machine_call(_machine._revalidate_root, root, context=context)
        _validate_exact_inventory(
            root,
            expected=expected,
            context=context,
            maximum_each=maximum_each,
            maximum_total=maximum_total,
        )
    for pinned, context in (
        (inputs.plan, "derivation plan"),
        (inputs.builder, "live derivation builder"),
        (inputs.machine_runner, "live native execution dependency"),
    ):
        _machine_call(_machine._revalidate_file, pinned, context=context)
    for pdf_id in PDF_IDS:
        _machine_call(
            _machine._revalidate_file,
            inputs.sources[pdf_id],
            context=f"source bundle {pdf_id}",
        )
        _machine_call(
            _machine._revalidate_file,
            inputs.producers[pdf_id],
            context=f"producer {pdf_id}",
        )
        _machine_call(
            _machine._revalidate_file,
            inputs.authorities[pdf_id],
            context=f"producer-toolchain authority {pdf_id}",
        )


def _validate_fd_headroom(inputs: _Inputs) -> None:
    descriptor_root = Path("/dev/fd")
    try:
        current = len(tuple(descriptor_root.iterdir()))
    except OSError as error:
        _fail(f"cannot establish the current open-descriptor count: {error}")
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return
    owned = 6 + len(inputs.sources) + len(inputs.producers) + len(inputs.authorities)
    required = current + owned + 16
    if required > soft:
        _fail(f"RLIMIT_NOFILE headroom is insufficient: need {required}, have {soft}")


def _adapter_arguments(pdf_id: str, source_fd: int) -> list[str]:
    return [
        "--dialect-derivation-protocol",
        PRODUCER_PROTOCOL,
        "--pdf-id",
        pdf_id,
        "--source-fd",
        str(source_fd),
        "--pdf-output",
        "stdout",
    ]


def _invocation_receipt(
    *,
    pdf_id: str,
    document: Mapping[str, object],
    inputs: _Inputs,
    pdf_bytes: int,
    pdf_sha256: str,
    return_code: int,
    attestation: Mapping[str, object],
) -> bytes:
    """Bind one successful bounded invocation without claiming adapter authorship."""
    authority = _expect_mapping(
        document["adapter_authority"],
        context=f"producer {pdf_id} adapter authority",
    )
    receipt = {
        "schema": INVOCATION_RECEIPT_SCHEMA,
        "status": "bounded-native-invocation-validated",
        "pdf_id": pdf_id,
        "source_bundle": {
            "member": document["source_bundle_member"],
            "bytes": document["source_bundle_bytes"],
            "sha256": document["source_bundle_sha256"],
        },
        "producer": {
            "member": document["producer_member"],
            "bytes": document["producer_bytes"],
            "sha256": document["producer_sha256"],
            "toolchain_profile": document["toolchain_profile"],
        },
        "producer_toolchain_authority": {
            "member": AUTHORITY_MEMBER_BY_ID[pdf_id],
            "bytes": inputs.authorities[pdf_id].size,
            "sha256": inputs.authorities[pdf_id].sha256,
            "authority_id": authority["authority_id"],
            "status": authority["status"],
            "authentication": authority["authentication"],
            "review_scope": PRODUCER_AUTHORITY_REVIEW_SCOPE,
        },
        "producer_protocol": PRODUCER_PROTOCOL,
        "producer_arguments": list(document["producer_arguments"]),
        "descriptor_placeholders_bound_at_spawn": ["source_fd"],
        "pdf": {
            "member": document["pdf_member"],
            "bytes": pdf_bytes,
            "sha256": pdf_sha256,
        },
        "return_code": return_code,
        "stderr_bytes": 0,
        "executable_binding_sha256": _sha256(_canonical_json(attestation)),
        "origin": "derivation-closure-after-bounded-pdf-validation",
    }
    raw = _canonical_json(receipt) + b"\n"
    if len(raw) > MAX_INVOCATION_RECEIPT_BYTES:
        _fail(f"closure invocation receipt {pdf_id} exceeds its byte bound")
    return raw


def _invoke_producer(
    producer: object,
    arguments: Sequence[str],
    *,
    inherited_fds: Sequence[int],
    budget: object,
    before: Callable[[], None],
    after: Callable[[], None],
) -> tuple[int, bytes, bytes, dict[str, object]]:
    if sys.platform != "darwin":
        _fail("native derivation execution is supported only on Darwin")
    return _machine_call(
        _machine._run_bounded,
        producer,
        arguments,
        inherited_fds=inherited_fds,
        timeout=PRODUCER_TIMEOUT_SECONDS,
        stdout_limit=MAX_PDF_BYTES,
        stderr_limit=MAX_STDERR_BYTES,
        budget=budget,
        before=before,
        after=after,
    )  # type: ignore[return-value]


def _validate_pdf_bytes(
    raw: bytes,
    *,
    member: str,
    expected_sha256: str,
) -> tuple[int, str]:
    size = len(raw)
    if size <= 0 or size > MAX_PDF_BYTES:
        _fail(f"produced PDF {member} exceeds its bounded byte contract")
    if raw[: len(_PDF_SIGNATURE)] != _PDF_SIGNATURE or b"%%EOF" not in raw[-1024:]:
        _fail(f"produced PDF {member} lacks the bounded PDF signature/EOF marker")
    observed = _sha256(raw)
    if observed != expected_sha256:
        _fail(f"produced PDF {member} differs from its caller SHA-256 authority anchor")
    return size, observed


def _write_member(
    output_root: object,
    member: str,
    raw: bytes,
    *,
    guard: Callable[[], None],
    maximum: int = MAX_MANIFEST_BYTES,
) -> dict[str, object]:
    if len(raw) > maximum:
        _fail(f"output member {member} exceeds the local byte bound")
    return _machine_call(
        _machine._write_member,
        output_root,
        member,
        raw,
        guard=guard,
    )  # type: ignore[return-value]


def _source_set(inputs: _Inputs) -> tuple[list[dict[str, object]], str]:
    records = [
        {
            "pdf_id": pdf_id,
            "member": SOURCE_MEMBER_BY_ID[pdf_id],
            "bytes": inputs.sources[pdf_id].size,
            "sha256": inputs.sources[pdf_id].sha256,
        }
        for pdf_id in PDF_IDS
    ]
    return records, _sha256(_canonical_json(records))


def _producer_set(inputs: _Inputs) -> tuple[list[dict[str, object]], str]:
    records = [
        {
            "pdf_id": pdf_id,
            "member": PRODUCER_MEMBER_BY_ID[pdf_id],
            "bytes": inputs.producers[pdf_id].size,
            "sha256": inputs.producers[pdf_id].sha256,
        }
        for pdf_id in PDF_IDS
    ]
    return records, _sha256(_canonical_json(records))


def _producer_toolchain_authority_set(
    inputs: _Inputs,
) -> tuple[list[dict[str, object]], str]:
    records = [
        {
            "pdf_id": pdf_id,
            "member": AUTHORITY_MEMBER_BY_ID[pdf_id],
            "bytes": inputs.authorities[pdf_id].size,
            "sha256": inputs.authorities[pdf_id].sha256,
        }
        for pdf_id in PDF_IDS
    ]
    return records, _sha256(_canonical_json(records))


def _run_document(
    *,
    pdf_id: str,
    document: Mapping[str, object],
    inputs: _Inputs,
    output_root: object,
    budget: object,
    output_guard: Callable[[], None],
) -> dict[str, object]:
    run_records: list[dict[str, object]] = []
    receipt_raw_by_run: list[bytes] = []
    pdf_digest_by_run: list[str] = []

    def invocation_guard() -> None:
        _revalidate_inputs(inputs)
        try:
            os.lseek(inputs.sources[pdf_id].descriptor, 0, os.SEEK_SET)
        except OSError as error:
            _fail(f"cannot reset source bundle {pdf_id} descriptor offset: {error}")
        output_guard()

    for label in ("a", "b"):
        output_guard()
        _revalidate_inputs(inputs)
        pdf_member = f"runs/{pdf_id}/rebuild-{label}.pdf"
        receipt_member = f"runs/{pdf_id}/rebuild-{label}.receipt.json"
        arguments = _adapter_arguments(
            pdf_id,
            inputs.sources[pdf_id].descriptor,
        )
        return_code, pdf_raw, stderr, attestation = _invoke_producer(
            inputs.producers[pdf_id],
            arguments,
            inherited_fds=(inputs.sources[pdf_id].descriptor,),
            budget=budget,
            before=invocation_guard,
            after=invocation_guard,
        )
        if return_code != 0:
            _fail(f"adapter {pdf_id} rebuild {label} exited with {return_code}")
        if stderr:
            _fail(f"adapter {pdf_id} rebuild {label} wrote stderr")
        pdf_bytes, pdf_sha256 = _validate_pdf_bytes(
            pdf_raw,
            member=pdf_member,
            expected_sha256=str(document["expected_pdf_sha256"]),
        )
        receipt_raw = _invocation_receipt(
            pdf_id=pdf_id,
            document=document,
            inputs=inputs,
            pdf_bytes=pdf_bytes,
            pdf_sha256=pdf_sha256,
            return_code=return_code,
            attestation=_expect_mapping(
                attestation,
                context=f"adapter {pdf_id} executable attestation",
            ),
        )
        _write_member(
            output_root,
            pdf_member,
            pdf_raw,
            guard=output_guard,
            maximum=MAX_PDF_BYTES,
        )
        receipt_record = _write_member(
            output_root,
            receipt_member,
            receipt_raw,
            guard=output_guard,
            maximum=MAX_INVOCATION_RECEIPT_BYTES,
        )
        receipt_raw_by_run.append(receipt_raw)
        pdf_digest_by_run.append(pdf_sha256)
        run_records.append(
            {
                "run": label,
                "pdf_member": pdf_member,
                "pdf_bytes": pdf_bytes,
                "pdf_sha256": pdf_sha256,
                "invocation_receipt_member": receipt_member,
                "invocation_receipt_bytes": receipt_record["bytes"],
                "invocation_receipt_sha256": receipt_record["sha256"],
                "arguments": list(document["producer_arguments"]),
                "executable_binding": attestation,
                "invocation_receipt_origin": (
                    "closure-generated-exact-binding-after-bounded-validation"
                ),
            },
        )
        _revalidate_inputs(inputs)
        output_guard()
    if receipt_raw_by_run[0] != receipt_raw_by_run[1]:
        _fail(f"adapter {pdf_id} receipts are not byte-identical across rebuilds")
    if pdf_digest_by_run[0] != pdf_digest_by_run[1]:
        _fail(f"adapter {pdf_id} PDF rebuilds are not byte-identical")
    return {
        "pdf_id": pdf_id,
        "pdf_member": document["pdf_member"],
        "pdf_bytes": run_records[0]["pdf_bytes"],
        "pdf_sha256": run_records[0]["pdf_sha256"],
        "source_bundle": {
            "member": document["source_bundle_member"],
            "bytes": document["source_bundle_bytes"],
            "sha256": document["source_bundle_sha256"],
        },
        "producer": {
            "member": document["producer_member"],
            "bytes": document["producer_bytes"],
            "sha256": document["producer_sha256"],
            "toolchain_profile": document["toolchain_profile"],
            "adapter_authority": document["adapter_authority"],
        },
        "runs": run_records,
        "native_closure_projection": {
            "invocation_receipt_sha256": run_records[0]["invocation_receipt_sha256"],
            "invocation_receipt_bytes": run_records[0]["invocation_receipt_bytes"],
            "rebuild_a_sha256": run_records[0]["pdf_sha256"],
            "rebuild_b_sha256": run_records[1]["pdf_sha256"],
            "status": (
                "synthetic-only-not-wrapper-integrated"
                if inputs.normalized_plan["mode"] == MODE_SYNTHETIC
                else "native-revision-candidate-not-wrapper-integrated"
            ),
        },
    }


def _normalize_public_v2_attestation(
    value: object,
    *,
    expected_code_directory: Mapping[str, object],
    macho_uuid: str,
    context: str,
) -> dict[str, object]:
    attestation = _expect_mapping(value, context=context)
    _validate_v2_json_scalar_types(attestation, context=context)
    _expect_keys(
        attestation,
        {
            "protocol",
            "spawn_flags",
            "suspended_wait_status",
            "code_signing_status",
            "expected_code_directory",
            "observed_cdhash",
            "producer_macho_uuid",
            "main_executable_mapping",
            "execution_binding_scope",
            "post_exec_runtime_mapping",
            "non_system_dylib_closure",
            "same_process_group_descendants",
            "detached_setsid_descendants",
            "same_vnode_mutation_fail_stop_assumption",
            "other_same_vnode_mutations",
        },
        context=context,
    )
    core = _v2_code_directory_core(expected_code_directory)
    spawn_flags = _expect_nonnegative_int(
        attestation["spawn_flags"],
        context=f"{context}.spawn_flags",
    )
    suspended_status = _expect_nonnegative_int(
        attestation["suspended_wait_status"],
        context=f"{context}.suspended_wait_status",
    )
    code_signing_status = _expect_nonnegative_int(
        attestation["code_signing_status"],
        context=f"{context}.code_signing_status",
    )
    if (
        attestation["protocol"] != "darwin-posix-spawn-suspended-main-executable-v1"
        or spawn_flags != _machine.DARWIN_SPAWN_FLAGS
        or suspended_status != 0x7F
        or code_signing_status > 0xFFFFFFFF
        or code_signing_status & _machine.REQUIRED_CS_FLAGS
        != _machine.REQUIRED_CS_FLAGS
        or code_signing_status & _machine.REJECTED_CS_FLAGS
        or attestation["expected_code_directory"] != core
        or attestation["observed_cdhash"] != core["cdhash"]
        or attestation["producer_macho_uuid"] != macho_uuid
        or attestation["execution_binding_scope"] != "main_executable-pre-exec-only"
        or attestation["post_exec_runtime_mapping"] != "not_attested"
        or attestation["non_system_dylib_closure"] != "not_attested"
        or attestation["same_process_group_descendants"]
        != "terminal-sigkill-before-main-reap-with-wnowait-held-leader"
        or attestation["detached_setsid_descendants"] != "not_contained"
        or attestation["same_vnode_mutation_fail_stop_assumption"]
        != "invalid-signed-code-page-triggers-darwin-cs-kill"
        or attestation["other_same_vnode_mutations"] != "not_attested"
    ):
        _fail(f"{context} fixed suspended-process semantics drifted")
    mapping = _expect_mapping(
        attestation["main_executable_mapping"],
        context=f"{context}.main_executable_mapping",
    )
    _expect_keys(
        mapping,
        {
            "device",
            "inode",
            "path_recorded",
            "mode",
            "link_count",
            "protection",
            "file_offset",
        },
        context=f"{context}.main_executable_mapping",
    )
    device = _expect_nonnegative_int(
        mapping["device"],
        context=f"{context}.main_executable_mapping.device",
    )
    inode = _expect_positive_int(
        mapping["inode"],
        context=f"{context}.main_executable_mapping.inode",
    )
    link_count = _expect_positive_int(
        mapping["link_count"],
        context=f"{context}.main_executable_mapping.link_count",
    )
    protection = _expect_nonnegative_int(
        mapping["protection"],
        context=f"{context}.main_executable_mapping.protection",
    )
    file_offset = _expect_nonnegative_int(
        mapping["file_offset"],
        context=f"{context}.main_executable_mapping.file_offset",
    )
    if (
        mapping["path_recorded"] is not False
        or mapping["mode"] != V2_PRODUCER_MODE
        or link_count != 1
        or device > 0xFFFFFFFF
        or inode > 0xFFFFFFFFFFFFFFFF
        or protection > 0xFFFFFFFF
        or protection & _machine.VM_PROT_EXECUTE == 0
        or file_offset != 0
    ):
        _fail(f"{context} path-free executable mapping is invalid")
    return {
        **dict(attestation),
        "spawn_flags": spawn_flags,
        "suspended_wait_status": suspended_status,
        "code_signing_status": code_signing_status,
        "main_executable_mapping": {
            **dict(mapping),
            "device": device,
            "inode": inode,
            "link_count": link_count,
            "protection": protection,
            "file_offset": file_offset,
        },
    }


def _sanitize_v2_attestation(
    value: Mapping[str, object],
    *,
    pdf_id: str,
    producer: object,
    expected_code_directory: Mapping[str, object],
    macho_uuid: str,
) -> dict[str, object]:
    context = f"v2 producer {pdf_id} suspended-process attestation"
    _expect_keys(
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
        context=context,
    )
    core = _v2_code_directory_core(expected_code_directory)
    mapping = _expect_mapping(
        value["main_executable_mapping"],
        context=f"{context}.main_executable_mapping",
    )
    _expect_keys(
        mapping,
        {"device", "inode", "path", "mode", "link_count", "protection", "file_offset"},
        context=f"{context}.main_executable_mapping",
    )
    if (
        value["execution_binding_scope"] != "main_executable"
        or value["non_system_dylib_closure"] != "not_attested"
        or not isinstance(mapping["path"], str)
        or not str(mapping["path"]).startswith("/")
        or mapping["device"] != (producer.device & 0xFFFFFFFF)
        or mapping["inode"] != producer.inode
    ):
        _fail(f"{context} mapped vnode differs from the held package producer")
    public = {
        "protocol": value["protocol"],
        "spawn_flags": value["spawn_flags"],
        "suspended_wait_status": value["suspended_wait_status"],
        "code_signing_status": value["code_signing_status"],
        "expected_code_directory": core,
        "observed_cdhash": value["observed_cdhash"],
        "producer_macho_uuid": macho_uuid,
        "main_executable_mapping": {
            "device": mapping["device"],
            "inode": mapping["inode"],
            "path_recorded": False,
            "mode": mapping["mode"],
            "link_count": mapping["link_count"],
            "protection": mapping["protection"],
            "file_offset": mapping["file_offset"],
        },
        "execution_binding_scope": "main_executable-pre-exec-only",
        "post_exec_runtime_mapping": "not_attested",
        "non_system_dylib_closure": "not_attested",
        "same_process_group_descendants": (
            "terminal-sigkill-before-main-reap-with-wnowait-held-leader"
        ),
        "detached_setsid_descendants": "not_contained",
        "same_vnode_mutation_fail_stop_assumption": value[
            "same_vnode_mutation_fail_stop_assumption"
        ],
        "other_same_vnode_mutations": value["other_same_vnode_mutations"],
    }
    return _normalize_public_v2_attestation(
        public,
        expected_code_directory=expected_code_directory,
        macho_uuid=macho_uuid,
        context=context,
    )


def _invocation_receipt_v2(
    *,
    pdf_id: str,
    document: Mapping[str, object],
    pdf_bytes: int,
    pdf_sha256: str,
    return_code: int,
    attestation: Mapping[str, object],
) -> bytes:
    package = _expect_mapping(
        document["producer_package"],
        context=f"v2 invocation package {pdf_id}",
    )
    producer = _expect_mapping(
        package["producer"],
        context=f"v2 invocation producer {pdf_id}",
    )
    authority = _expect_mapping(
        package["authority"],
        context=f"v2 invocation authority {pdf_id}",
    )
    authority_projection = _expect_mapping(
        package["authority_projection"],
        context=f"v2 invocation authority projection {pdf_id}",
    )
    source = _expect_mapping(
        document["source_bundle"],
        context=f"v2 invocation source {pdf_id}",
    )
    receipt = {
        "schema": INVOCATION_RECEIPT_SCHEMA_V2,
        "status": "bounded-native-package-invocation-validated",
        "pdf_id": pdf_id,
        "source_bundle": source,
        "source_descriptor_binding": {
            "child_descriptor": V2_SOURCE_DESCRIPTOR,
            "parent_descriptor_recorded": False,
            "mapping": "single-posix-spawn-dup2-or-inherit-action",
            "outer_open_flags": "O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK",
            "launcher_validation": (
                "read-only-regular-euid-mode-0400-nlink1-bounded-seekable-rewound"
            ),
        },
        "producer_package": {
            "root_mode": package["root_mode"],
            "root_owner": package["root_owner"],
            "package_content_sha256": package["package_content_sha256"],
            "producer": producer,
            "authority": authority,
            "authorization": document["authorization"],
            "authority_projection": authority_projection,
        },
        "producer_protocol": PRODUCER_PROTOCOL,
        "producer_arguments_template": list(document["producer_arguments"]),
        "producer_arguments_realized": _adapter_arguments(
            pdf_id,
            V2_SOURCE_DESCRIPTOR,
        ),
        "launcher_pre_exec_attestation": attestation,
        "runtime_handoff": {
            "launcher_config_sha256": authority_projection["launcher_config_sha256"],
            "runtime_handoff_sha256": authority_projection["runtime_handoff_sha256"],
            "post_exec_runtime_mapping": "not_attested",
            "source_causal_read": "not_observed",
        },
        "pdf": {
            "member": document["pdf_member"],
            "bytes": pdf_bytes,
            "sha256": pdf_sha256,
        },
        "return_code": return_code,
        "stderr_bytes": 0,
        "origin": "v2-derivation-closure-after-bounded-pdf-validation",
    }
    raw = _canonical_json(receipt) + b"\n"
    if len(raw) > MAX_INVOCATION_RECEIPT_BYTES:
        _fail(f"v2 invocation receipt {pdf_id} exceeds its byte bound")
    _reject_v2_forbidden_fragments(
        raw,
        context=f"v2 invocation receipt {pdf_id}",
    )
    return raw


def _invoke_producer_v2(
    producer: object,
    arguments: Sequence[str],
    *,
    source: object,
    budget: object,
    before: Callable[[], None],
    after: Callable[[], None],
) -> tuple[int, bytes, bytes, dict[str, object]]:
    if sys.platform != "darwin":
        _fail("v2 native derivation execution is supported only on Darwin")
    if arguments != _adapter_arguments(
        str(arguments[3]),
        V2_SOURCE_DESCRIPTOR,
    ):
        _fail("v2 producer invocation must use the fixed child source descriptor 3")
    return _machine_call(
        _machine._run_bounded,
        producer,
        arguments,
        inherited_fds=(),
        inherited_fd_binding=(source.descriptor, V2_SOURCE_DESCRIPTOR),
        timeout=PRODUCER_TIMEOUT_SECONDS,
        stdout_limit=MAX_V2_NATIVE_OUTPUT_BYTES,
        stderr_limit=MAX_STDERR_BYTES,
        budget=budget,
        before=before,
        after=after,
    )  # type: ignore[return-value]


def _run_v2_document(
    *,
    pdf_id: str,
    document: Mapping[str, object],
    inputs: _V2Inputs,
    output_root: object,
    budget: object,
    output_guard: Callable[[], None],
) -> dict[str, object]:
    package = _expect_mapping(
        document["producer_package"],
        context=f"v2 package {pdf_id}",
    )
    producer_record = _expect_mapping(
        package["producer"],
        context=f"v2 package producer {pdf_id}",
    )
    expected_code_directory = _expect_mapping(
        producer_record["native_code_directory"],
        context=f"v2 producer CodeDirectory {pdf_id}",
    )
    expected_output = _expect_mapping(
        document["expected_output"],
        context=f"v2 expected output {pdf_id}",
    )
    expected_pdf = _expect_mapping(
        expected_output["pdf"],
        context=f"v2 expected PDF {pdf_id}",
    )
    run_records: list[dict[str, object]] = []
    receipt_raw_by_run: list[bytes] = []
    pdf_raw_by_run: list[bytes] = []

    def invocation_guard() -> None:
        _revalidate_v2_inputs(inputs)
        output_guard()

    for label in ("a", "b"):
        invocation_guard()
        pdf_member = f"runs/{pdf_id}/rebuild-{label}.pdf"
        receipt_member = f"runs/{pdf_id}/rebuild-{label}.receipt.json"
        arguments = _adapter_arguments(pdf_id, V2_SOURCE_DESCRIPTOR)
        return_code, pdf_raw, stderr, raw_attestation = _invoke_producer_v2(
            inputs.producers[pdf_id],
            arguments,
            source=inputs.sources[pdf_id],
            budget=budget,
            before=invocation_guard,
            after=invocation_guard,
        )
        if return_code != 0:
            _fail(f"v2 producer {pdf_id} rebuild {label} exited with {return_code}")
        if stderr:
            _fail(f"v2 producer {pdf_id} rebuild {label} wrote stderr")
        pdf_bytes, pdf_sha256 = _validate_pdf_bytes(
            pdf_raw,
            member=pdf_member,
            expected_sha256=str(expected_pdf["sha256"]),
        )
        if pdf_bytes != expected_pdf["bytes"]:
            _fail(f"v2 producer {pdf_id} rebuild {label} has the wrong PDF size")
        attestation = _sanitize_v2_attestation(
            _expect_mapping(
                raw_attestation,
                context=f"v2 producer {pdf_id} raw executable attestation",
            ),
            pdf_id=pdf_id,
            producer=inputs.producers[pdf_id],
            expected_code_directory=expected_code_directory,
            macho_uuid=str(producer_record["macho_uuid"]),
        )
        receipt_raw = _invocation_receipt_v2(
            pdf_id=pdf_id,
            document=document,
            pdf_bytes=pdf_bytes,
            pdf_sha256=pdf_sha256,
            return_code=return_code,
            attestation=attestation,
        )
        _write_member(
            output_root,
            pdf_member,
            pdf_raw,
            guard=output_guard,
            maximum=MAX_V2_NATIVE_OUTPUT_BYTES,
        )
        receipt_record = _write_member(
            output_root,
            receipt_member,
            receipt_raw,
            guard=output_guard,
            maximum=MAX_INVOCATION_RECEIPT_BYTES,
        )
        receipt_raw_by_run.append(receipt_raw)
        pdf_raw_by_run.append(pdf_raw)
        run_records.append(
            {
                "run": label,
                "pdf_member": pdf_member,
                "pdf_bytes": pdf_bytes,
                "pdf_sha256": pdf_sha256,
                "invocation_receipt_member": receipt_member,
                "invocation_receipt_bytes": receipt_record["bytes"],
                "invocation_receipt_sha256": receipt_record["sha256"],
                "producer_arguments_realized": arguments,
                "launcher_pre_exec_attestation": attestation,
            },
        )
        invocation_guard()
    if pdf_raw_by_run[0] != pdf_raw_by_run[1]:
        _fail(f"v2 producer {pdf_id} PDF rebuilds are not byte-identical")
    if receipt_raw_by_run[0] != receipt_raw_by_run[1]:
        _fail(f"v2 producer {pdf_id} receipts are not byte-identical")
    return {
        "pdf_id": pdf_id,
        "pdf_member": document["pdf_member"],
        "pdf_bytes": run_records[0]["pdf_bytes"],
        "pdf_sha256": run_records[0]["pdf_sha256"],
        "producer_arguments": document["producer_arguments"],
        "source_bundle": document["source_bundle"],
        "producer_package": document["producer_package"],
        "authorization": document["authorization"],
        "expected_renderer_manifest": expected_output["renderer_manifest"],
        "runs": run_records,
        "native_closure_projection": {
            "invocation_receipt_sha256": run_records[0]["invocation_receipt_sha256"],
            "invocation_receipt_bytes": run_records[0]["invocation_receipt_bytes"],
            "rebuild_a_sha256": run_records[0]["pdf_sha256"],
            "rebuild_b_sha256": run_records[1]["pdf_sha256"],
            "status": V2_CLOSURE_STATUS,
        },
    }


def _member_inventory(
    output_root: object,
    members: Sequence[str],
) -> list[dict[str, object]]:
    return _machine_call(_machine._inventory_members, output_root, members)  # type: ignore[return-value]


def _produce_into(
    plan_path: Path,
    source_root_path: Path,
    producer_root_path: Path,
    authority_root_path: Path,
    output_root: object,
    *,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
    output_guard: Callable[[], None],
) -> _Production:
    inputs: _Inputs | None = None
    try:
        inputs = _pin_inputs(
            plan_path,
            source_root_path,
            producer_root_path,
            authority_root_path,
            output_root.path,
            release_id=release_id,
            expected_plan_sha256=expected_plan_sha256,
            expected_builder_sha256=expected_builder_sha256,
            expected_machine_runner_sha256=expected_machine_runner_sha256,
            expected_upstream_sha256=expected_upstream_sha256,
            expected_authority_sha256=expected_authority_sha256,
            expected_source_sha256=expected_source_sha256,
            expected_producer_sha256=expected_producer_sha256,
            expected_pdf_sha256=expected_pdf_sha256,
        )
        _validate_fd_headroom(inputs)
        _revalidate_inputs(inputs)
        source_records, source_set_sha256 = _source_set(inputs)
        producer_records, producer_set_sha256 = _producer_set(inputs)
        authority_records, authority_set_sha256 = _producer_toolchain_authority_set(
            inputs,
        )
        budget = _InvocationBudget()
        documents: list[dict[str, object]] = []
        for raw_document in inputs.normalized_plan["documents"]:  # type: ignore[union-attr]
            document = _expect_mapping(raw_document, context="normalized plan document")
            pdf_id = str(document["pdf_id"])
            documents.append(
                _run_document(
                    pdf_id=pdf_id,
                    document=document,
                    inputs=inputs,
                    output_root=output_root,
                    budget=budget,
                    output_guard=output_guard,
                ),
            )
        if budget.count != MAX_ADAPTER_INVOCATIONS:
            _fail("native adapter invocation count does not match the fixed budget")
        if (
            sum(int(document["pdf_bytes"]) * 2 for document in documents)
            > MAX_TOTAL_PDF_BYTES
        ):
            _fail("produced PDF bytes exceed the aggregate output bound")
        pdf_set = [
            {
                "pdf_id": document["pdf_id"],
                "pdf_member": document["pdf_member"],
                "pdf_bytes": document["pdf_bytes"],
                "pdf_sha256": document["pdf_sha256"],
            }
            for document in documents
        ]
        pdf_set_sha256 = _sha256(_canonical_json(pdf_set))
        files, directories, _ = _machine_call(
            _machine._walk_output,
            output_root,
            directory_mode=0o700,
        )
        if (
            len(files) != len(PDF_ORDER) * 4
            or len(directories) > MAX_OUTPUT_DIRECTORIES
        ):
            _fail("derivation output inventory is outside the exact bounded shape")
        member_inventory = _member_inventory(output_root, files)
        mode = str(inputs.normalized_plan["mode"])
        status, promotion_blockers = _mode_status(mode)
        unsigned: dict[str, object] = {
            "schema": DERIVATION_CLOSURE_SCHEMA,
            "contract": DERIVATION_CLOSURE_CONTRACT,
            "mode": mode,
            "release_id": release_id,
            "status": status,
            "promotable": False,
            "promotion_blockers": promotion_blockers,
            "inputs": {
                "plan_sha256": inputs.plan.sha256,
                "builder_bytes": inputs.builder.size,
                "builder_sha256": inputs.builder.sha256,
                "machine_runner_bytes": inputs.machine_runner.size,
                "machine_runner_sha256": inputs.machine_runner.sha256,
                "upstream_bindings": inputs.normalized_plan["upstream_bindings"],
                "real_producer_toolchain_authority": inputs.normalized_plan[
                    "real_producer_toolchain_authority"
                ],
            },
            "execution_contract": EXECUTION_CONTRACT,
            "builder_release_toolchain_baseline": (BUILDER_RELEASE_TOOLCHAIN_BASELINE),
            "source_bundle_set": source_records,
            "source_bundle_set_sha256": source_set_sha256,
            "producer_set": producer_records,
            "producer_set_sha256": producer_set_sha256,
            "producer_toolchain_authority_set": authority_records,
            "producer_toolchain_authority_set_sha256": authority_set_sha256,
            "pdf_set": pdf_set,
            "pdf_set_sha256": pdf_set_sha256,
            "documents": documents,
            "wrapper_integration": _wrapper_integration(mode),
            "non_inference_limits": NON_INFERENCE_LIMITS,
            "member_inventory": member_inventory,
            "summary": {
                "pdf_count": len(documents),
                "rebuild_count": len(documents) * 2,
                "invocation_receipt_count": len(documents) * 2,
                "adapter_invocation_count": budget.count,
            },
            "builder": {
                "member": BUILDER_MEMBER,
                "bytes": inputs.builder.size,
                "sha256": inputs.builder.sha256,
            },
            "native_execution_dependency": {
                "member": MACHINE_RUNNER_MEMBER,
                "bytes": inputs.machine_runner.size,
                "sha256": inputs.machine_runner.sha256,
            },
        }
        manifest = {
            **unsigned,
            "payload_sha256": _sha256(_canonical_json(unsigned)),
        }
        manifest_raw = _canonical_json(manifest)
        _write_member(
            output_root,
            MANIFEST_MEMBER,
            manifest_raw,
            guard=output_guard,
        )
        _revalidate_inputs(inputs)
        output_guard()
        _machine_call(_machine._seal_output_tree, output_root, guard=output_guard)
        _validate_tree(output_root, manifest_raw, directory_mode=0o500)
        _revalidate_inputs(inputs)
        return _Production(
            manifest=manifest,
            manifest_raw=manifest_raw,
            member_inventory=member_inventory,
        )
    finally:
        if inputs is not None:
            inputs.close()


def _produce_v2_with_inputs(
    inputs: _V2Inputs,
    output_root: object,
    *,
    output_guard: Callable[[], None],
) -> _Production:
    _validate_v2_fd_headroom(inputs)
    _revalidate_v2_inputs(inputs)
    normalized_documents = [
        _expect_mapping(item, context="normalized v2 plan document")
        for item in inputs.normalized_plan["documents"]  # type: ignore[union-attr]
    ]
    source_records = [
        {
            "pdf_id": str(document["pdf_id"]),
            **dict(
                _expect_mapping(
                    document["source_bundle"],
                    context="normalized v2 source record",
                ),
            ),
        }
        for document in normalized_documents
    ]
    source_set_sha256 = _sha256(_canonical_json(source_records))
    package_records = [inputs.package_records[pdf_id] for pdf_id in PDF_IDS]
    package_set_sha256 = _sha256(_canonical_json(package_records))
    if (
        source_set_sha256 != inputs.normalized_plan["source_bundle_set_sha256"]
        or package_set_sha256 != inputs.normalized_plan["producer_package_set_sha256"]
    ):
        _fail("v2 pinned source or package set differs from the authorized plan")
    producer_records = []
    authority_records = []
    for document in normalized_documents:
        pdf_id = str(document["pdf_id"])
        package = _expect_mapping(
            document["producer_package"],
            context=f"normalized v2 package {pdf_id}",
        )
        producer = _expect_mapping(
            package["producer"],
            context=f"normalized v2 producer {pdf_id}",
        )
        authority = _expect_mapping(
            package["authority"],
            context=f"normalized v2 authority {pdf_id}",
        )
        authority_projection = _expect_mapping(
            package["authority_projection"],
            context=f"normalized v2 authority projection {pdf_id}",
        )
        producer_records.append({"pdf_id": pdf_id, **dict(producer)})
        authority_records.append(
            {
                "pdf_id": pdf_id,
                **dict(authority),
                "status": V2_AUTHORITY_STATUS,
                "authentication": "caller-sha-anchor-only",
                "caller_anchors": document["caller_anchors"],
                **dict(authority_projection),
            },
        )
    producer_set_sha256 = _sha256(_canonical_json(producer_records))
    authority_set_sha256 = _sha256(_canonical_json(authority_records))
    budget = _InvocationBudget()
    documents = [
        _run_v2_document(
            pdf_id=str(document["pdf_id"]),
            document=document,
            inputs=inputs,
            output_root=output_root,
            budget=budget,
            output_guard=output_guard,
        )
        for document in normalized_documents
    ]
    if budget.count != MAX_ADAPTER_INVOCATIONS:
        _fail("v2 producer invocation count does not match the fixed budget")
    produced_pdf_bytes = sum(int(document["pdf_bytes"]) * 2 for document in documents)
    if produced_pdf_bytes > MAX_TOTAL_PDF_BYTES:
        _fail("v2 produced PDF bytes exceed the aggregate output bound")
    pdf_set = [
        {
            "pdf_id": document["pdf_id"],
            "pdf_member": document["pdf_member"],
            "pdf_bytes": document["pdf_bytes"],
            "pdf_sha256": document["pdf_sha256"],
        }
        for document in documents
    ]
    pdf_set_sha256 = _sha256(_canonical_json(pdf_set))
    files, directories, _ = _machine_call(
        _machine._walk_output,
        output_root,
        directory_mode=0o700,
    )
    if len(files) != len(PDF_ORDER) * 4 or len(directories) > MAX_OUTPUT_DIRECTORIES:
        _fail("v2 derivation output inventory is outside the exact bounded shape")
    member_inventory = _member_inventory(output_root, files)
    unsigned: dict[str, object] = {
        "schema": DERIVATION_CLOSURE_SCHEMA_V2,
        "contract": DERIVATION_CLOSURE_CONTRACT_V2,
        "mode": MODE_REVISION,
        "release_id": inputs.normalized_plan["release_id"],
        "status": V2_CLOSURE_STATUS,
        "promotable": False,
        "promotion_blockers": _canonical_copy(V2_PROMOTION_BLOCKERS),
        "inputs": {
            "plan_sha256": inputs.plan.sha256,
            "builder_bytes": inputs.builder.size,
            "builder_sha256": inputs.builder.sha256,
            "machine_runner_bytes": inputs.machine_runner.size,
            "machine_runner_sha256": inputs.machine_runner.sha256,
            "upstream_bindings": inputs.normalized_plan["upstream_bindings"],
        },
        "execution_contract": _canonical_copy(EXECUTION_CONTRACT_V2),
        "source_bundle_set": source_records,
        "source_bundle_set_sha256": source_set_sha256,
        "producer_package_set": package_records,
        "producer_package_set_sha256": package_set_sha256,
        "producer_set": producer_records,
        "producer_set_sha256": producer_set_sha256,
        "producer_toolchain_authority_set": authority_records,
        "producer_toolchain_authority_set_sha256": authority_set_sha256,
        "pdf_set": pdf_set,
        "pdf_set_sha256": pdf_set_sha256,
        "documents": documents,
        "wrapper_integration": {
            "status": "not-integrated-requires-downstream-promotion-closure-v2",
            "accepted_native_derivation_schema": DERIVATION_CLOSURE_SCHEMA_V2,
            "v1_accepted_for_promotion": False,
            "production_blocker": V2_PROMOTION_BLOCKERS[0],
        },
        "non_inference_limits": _canonical_copy(NON_INFERENCE_LIMITS_V2),
        "member_inventory": member_inventory,
        "summary": {
            "pdf_count": len(documents),
            "rebuild_count": len(documents) * 2,
            "invocation_receipt_count": len(documents) * 2,
            "producer_invocation_count": budget.count,
            "package_count": len(package_records),
        },
        "builder": {
            "member": BUILDER_MEMBER,
            "bytes": inputs.builder.size,
            "sha256": inputs.builder.sha256,
        },
        "native_execution_dependency": {
            "member": MACHINE_RUNNER_MEMBER,
            "bytes": inputs.machine_runner.size,
            "sha256": inputs.machine_runner.sha256,
        },
    }
    manifest = {**unsigned, "payload_sha256": _sha256(_canonical_json(unsigned))}
    manifest_raw = _canonical_json(manifest)
    _reject_v2_forbidden_fragments(manifest_raw, context="v2 public manifest")
    _write_member(
        output_root,
        MANIFEST_MEMBER,
        manifest_raw,
        guard=output_guard,
    )
    _revalidate_v2_inputs(inputs)
    output_guard()
    _machine_call(_machine._seal_output_tree, output_root, guard=output_guard)
    _validate_v2_tree(output_root, manifest_raw, directory_mode=0o500)
    _revalidate_v2_inputs(inputs)
    return _Production(
        manifest=manifest,
        manifest_raw=manifest_raw,
        member_inventory=member_inventory,
    )


def _read_output_member(root: object, member: str, *, maximum: int) -> bytes:
    return _machine_call(
        _machine._read_bound_output_member,
        root,
        member,
        maximum=maximum,
    )  # type: ignore[return-value]


def _validate_manifest_document(
    document: Mapping[str, object],
    *,
    mode: str,
    pdf_id: str,
    pdf_member: str,
    source_record: Mapping[str, object],
    producer_record: Mapping[str, object],
    authority_record: Mapping[str, object],
    pdf_record: Mapping[str, object],
) -> None:
    context = f"manifest document {pdf_id}"
    _expect_keys(
        document,
        {
            "pdf_id",
            "pdf_member",
            "pdf_bytes",
            "pdf_sha256",
            "source_bundle",
            "producer",
            "runs",
            "native_closure_projection",
        },
        context=context,
    )
    if (
        document["pdf_id"] != pdf_id
        or document["pdf_member"] != pdf_member
        or document["pdf_bytes"] != pdf_record["pdf_bytes"]
        or document["pdf_sha256"] != pdf_record["pdf_sha256"]
    ):
        _fail(f"{context} differs from the ordered PDF set")
    source = _expect_mapping(document["source_bundle"], context=f"{context}.source")
    if source != {
        "member": source_record["member"],
        "bytes": source_record["bytes"],
        "sha256": source_record["sha256"],
    }:
        _fail(f"{context}.source differs from the source-bundle set")
    producer = _expect_mapping(document["producer"], context=f"{context}.producer")
    _expect_keys(
        producer,
        {"member", "bytes", "sha256", "toolchain_profile", "adapter_authority"},
        context=f"{context}.producer",
    )
    if {
        "member": producer["member"],
        "bytes": producer["bytes"],
        "sha256": producer["sha256"],
    } != {
        "member": producer_record["member"],
        "bytes": producer_record["bytes"],
        "sha256": producer_record["sha256"],
    }:
        _fail(f"{context}.producer differs from the producer set")
    profile = _expect_token(
        producer["toolchain_profile"],
        context=f"{context}.producer.toolchain_profile",
    )
    if mode == MODE_SYNTHETIC and profile != "synthetic-native-copy-v1":
        _fail(f"{context} has a non-synthetic profile in synthetic mode")
    if mode == MODE_REVISION and profile == "synthetic-native-copy-v1":
        _fail(f"{context} has a synthetic profile in revision mode")
    authority = _expect_mapping(
        producer["adapter_authority"],
        context=f"{context}.producer.adapter_authority",
    )
    _expect_keys(
        authority,
        {
            "status",
            "authority_id",
            "authentication",
            "authority_receipt_sha256",
            "reviewed_producer_sha256",
        },
        context=f"{context}.producer.adapter_authority",
    )
    expected_authority_status = (
        "synthetic-test-only" if mode == MODE_SYNTHETIC else "caller-authorized"
    )
    if (
        authority["status"] != expected_authority_status
        or authority["authentication"] != "caller-sha-anchor-only"
        or authority["reviewed_producer_sha256"] != producer_record["sha256"]
        or authority["authority_receipt_sha256"] != authority_record["sha256"]
    ):
        _fail(f"{context}.producer authority binding is invalid")
    _expect_token(
        authority["authority_id"],
        context=f"{context}.producer.adapter_authority.authority_id",
    )
    runs = _expect_sequence(document["runs"], context=f"{context}.runs")
    if len(runs) != 2:
        _fail(f"{context}.runs must contain exactly two rebuilds")
    normalized_runs: list[Mapping[str, object]] = []
    for index, label in enumerate(("a", "b")):
        run = _expect_mapping(runs[index], context=f"{context}.runs[{index}]")
        _expect_keys(
            run,
            {
                "run",
                "pdf_member",
                "pdf_bytes",
                "pdf_sha256",
                "invocation_receipt_member",
                "invocation_receipt_bytes",
                "invocation_receipt_sha256",
                "arguments",
                "executable_binding",
                "invocation_receipt_origin",
            },
            context=f"{context}.runs[{index}]",
        )
        if (
            run["run"] != label
            or run["pdf_member"] != f"runs/{pdf_id}/rebuild-{label}.pdf"
            or run["pdf_bytes"] != pdf_record["pdf_bytes"]
            or run["pdf_sha256"] != pdf_record["pdf_sha256"]
            or run["invocation_receipt_member"]
            != f"runs/{pdf_id}/rebuild-{label}.receipt.json"
            or run["arguments"] != _expected_arguments(pdf_id)
            or run["invocation_receipt_origin"]
            != "closure-generated-exact-binding-after-bounded-validation"
        ):
            _fail(f"{context}.runs[{index}] binding drifted")
        _expect_positive_int(
            run["invocation_receipt_bytes"],
            context=f"{context}.runs[{index}] invocation receipt bytes",
        )
        _expect_sha256(
            run["invocation_receipt_sha256"],
            context=f"{context}.runs[{index}] invocation receipt SHA-256",
        )
        _expect_mapping(
            run["executable_binding"],
            context=f"{context}.runs[{index}] executable binding",
        )
        normalized_runs.append(run)
    if (
        normalized_runs[0]["pdf_sha256"] != normalized_runs[1]["pdf_sha256"]
        or normalized_runs[0]["invocation_receipt_sha256"]
        != normalized_runs[1]["invocation_receipt_sha256"]
        or normalized_runs[0]["invocation_receipt_bytes"]
        != normalized_runs[1]["invocation_receipt_bytes"]
    ):
        _fail(f"{context} rebuild bindings are not byte-identical")
    expected_projection = {
        "invocation_receipt_sha256": normalized_runs[0]["invocation_receipt_sha256"],
        "invocation_receipt_bytes": normalized_runs[0]["invocation_receipt_bytes"],
        "rebuild_a_sha256": normalized_runs[0]["pdf_sha256"],
        "rebuild_b_sha256": normalized_runs[1]["pdf_sha256"],
        "status": (
            "synthetic-only-not-wrapper-integrated"
            if mode == MODE_SYNTHETIC
            else "native-revision-candidate-not-wrapper-integrated"
        ),
    }
    if document["native_closure_projection"] != expected_projection:
        _fail(f"{context}.native_closure_projection is invalid")


def _validate_manifest_semantics(manifest: Mapping[str, object]) -> None:
    required = {
        "schema",
        "contract",
        "mode",
        "release_id",
        "status",
        "promotable",
        "promotion_blockers",
        "inputs",
        "execution_contract",
        "builder_release_toolchain_baseline",
        "source_bundle_set",
        "source_bundle_set_sha256",
        "producer_set",
        "producer_set_sha256",
        "producer_toolchain_authority_set",
        "producer_toolchain_authority_set_sha256",
        "pdf_set",
        "pdf_set_sha256",
        "documents",
        "wrapper_integration",
        "non_inference_limits",
        "member_inventory",
        "summary",
        "builder",
        "native_execution_dependency",
        "payload_sha256",
    }
    _expect_keys(manifest, required, context="derivation closure manifest")
    if (
        manifest["schema"] != DERIVATION_CLOSURE_SCHEMA
        or manifest["contract"] != DERIVATION_CLOSURE_CONTRACT
    ):
        _fail("derivation closure manifest schema or contract is invalid")
    unsigned = dict(manifest)
    declared = _expect_sha256(
        unsigned.pop("payload_sha256"),
        context="manifest.payload_sha256",
    )
    if _sha256(_canonical_json(unsigned)) != declared:
        _fail("derivation closure manifest payload digest is invalid")
    mode = manifest["mode"]
    if mode not in MODES:
        _fail("derivation closure manifest has an invalid mode")
    release_id = _expect_token(
        manifest["release_id"],
        context="manifest.release_id",
    )
    if mode == MODE_SYNTHETIC and not release_id.startswith("synthetic-"):
        _fail("synthetic manifest release id must begin with 'synthetic-'")
    expected_status, expected_blockers = _mode_status(str(mode))
    if manifest["status"] != expected_status:
        _fail("derivation closure manifest status contradicts its mode")
    if manifest["promotion_blockers"] != expected_blockers:
        _fail("derivation closure manifest promotion blockers contradict its mode")
    if manifest["promotable"] is not False:
        _fail("native derivation evidence is not independently promotable")
    if manifest["execution_contract"] != EXECUTION_CONTRACT:
        _fail("manifest execution contract drifted")
    if (
        manifest["builder_release_toolchain_baseline"]
        != BUILDER_RELEASE_TOOLCHAIN_BASELINE
    ):
        _fail("manifest historical toolchain baseline drifted")
    if manifest["wrapper_integration"] != _wrapper_integration(str(mode)):
        _fail("manifest wrapper integration status drifted")
    if manifest["non_inference_limits"] != NON_INFERENCE_LIMITS:
        _fail("manifest non-inference limits drifted")
    set_records: dict[str, list[Mapping[str, object]]] = {}
    for key in (
        "source_bundle_set",
        "producer_set",
        "producer_toolchain_authority_set",
        "pdf_set",
    ):
        records = _expect_sequence(manifest[key], context=f"manifest.{key}")
        if len(records) != len(PDF_ORDER):
            _fail(f"manifest.{key} must contain exactly four records")
        if [
            _expect_mapping(record, context=f"manifest.{key} record")["pdf_id"]
            for record in records
        ] != list(PDF_IDS):
            _fail(f"manifest.{key} is not in fixed PDF order")
        digest_key = f"{key}_sha256"
        if _sha256(_canonical_json(records)) != manifest[digest_key]:
            _fail(f"manifest.{digest_key} is invalid")
        set_records[key] = [
            _expect_mapping(record, context=f"manifest.{key} record")
            for record in records
        ]
    documents = _expect_sequence(manifest["documents"], context="manifest.documents")
    if len(documents) != len(PDF_ORDER):
        _fail("manifest.documents must contain exactly four records")
    inputs = _expect_mapping(manifest["inputs"], context="manifest.inputs")
    _expect_keys(
        inputs,
        {
            "plan_sha256",
            "builder_bytes",
            "builder_sha256",
            "machine_runner_bytes",
            "machine_runner_sha256",
            "upstream_bindings",
            "real_producer_toolchain_authority",
        },
        context="manifest.inputs",
    )
    for key in ("plan_sha256", "builder_sha256", "machine_runner_sha256"):
        _expect_sha256(inputs[key], context=f"manifest.inputs.{key}")
    for key in ("builder_bytes", "machine_runner_bytes"):
        value = _expect_positive_int(inputs[key], context=f"manifest.inputs.{key}")
        if value > MAX_PLAN_BYTES:
            _fail(f"manifest.inputs.{key} exceeds its byte bound")
    _normalize_sha_map(
        _expect_mapping(
            inputs["upstream_bindings"],
            context="manifest.inputs.upstream_bindings",
        ),  # type: ignore[arg-type]
        keys=UPSTREAM_BINDING_KEYS,
        context="manifest.inputs.upstream_bindings",
    )
    authority_bindings = _normalize_sha_map(
        _expect_mapping(
            inputs["real_producer_toolchain_authority"],
            context="manifest.inputs.real_producer_toolchain_authority",
        ),  # type: ignore[arg-type]
        keys=PDF_IDS,
        context="manifest.inputs.real_producer_toolchain_authority",
    )
    for record_key, fixed_member, input_key, input_bytes_key in (
        ("builder", BUILDER_MEMBER, "builder_sha256", "builder_bytes"),
        (
            "native_execution_dependency",
            MACHINE_RUNNER_MEMBER,
            "machine_runner_sha256",
            "machine_runner_bytes",
        ),
    ):
        code_record = _expect_mapping(
            manifest[record_key],
            context=f"manifest.{record_key}",
        )
        _expect_keys(
            code_record,
            {"member", "bytes", "sha256"},
            context=f"manifest.{record_key}",
        )
        if code_record["member"] != fixed_member:
            _fail(f"manifest.{record_key}.member drifted")
        code_bytes = _expect_positive_int(
            code_record["bytes"],
            context=f"manifest.{record_key}.bytes",
        )
        if code_bytes > MAX_PLAN_BYTES:
            _fail(f"manifest.{record_key}.bytes exceeds its bound")
        code_sha256 = _expect_sha256(
            code_record["sha256"],
            context=f"manifest.{record_key}.sha256",
        )
        if code_sha256 != inputs[input_key]:
            _fail(f"manifest.{record_key} differs from its pinned input binding")
        if code_bytes != inputs[input_bytes_key]:
            _fail(f"manifest.{record_key} byte count differs from its pinned input")
    source_records = set_records["source_bundle_set"]
    producer_records = set_records["producer_set"]
    authority_records = set_records["producer_toolchain_authority_set"]
    pdf_records = set_records["pdf_set"]
    for index, (pdf_id, pdf_member) in enumerate(PDF_ORDER):
        source = source_records[index]
        producer = producer_records[index]
        authority_record = authority_records[index]
        pdf_record = pdf_records[index]
        for key, record, expected_member in (
            ("source", source, SOURCE_MEMBER_BY_ID[pdf_id]),
            ("producer", producer, PRODUCER_MEMBER_BY_ID[pdf_id]),
            ("authority", authority_record, AUTHORITY_MEMBER_BY_ID[pdf_id]),
        ):
            _expect_keys(
                record,
                {"pdf_id", "member", "bytes", "sha256"},
                context=f"manifest {key} set record {pdf_id}",
            )
            if record["pdf_id"] != pdf_id or record["member"] != expected_member:
                _fail(f"manifest {key} set record {pdf_id} has the wrong member")
            _expect_positive_int(
                record["bytes"],
                context=f"manifest {key} set record {pdf_id} bytes",
            )
            _expect_sha256(
                record["sha256"],
                context=f"manifest {key} set record {pdf_id} SHA-256",
            )
        if authority_record["sha256"] != authority_bindings[pdf_id]:
            _fail(f"manifest authority set record {pdf_id} differs from inputs")
        _expect_keys(
            pdf_record,
            {"pdf_id", "pdf_member", "pdf_bytes", "pdf_sha256"},
            context=f"manifest PDF set record {pdf_id}",
        )
        if pdf_record["pdf_id"] != pdf_id or pdf_record["pdf_member"] != pdf_member:
            _fail(f"manifest PDF set record {pdf_id} has the wrong logical member")
        _expect_positive_int(
            pdf_record["pdf_bytes"],
            context=f"manifest PDF set record {pdf_id} bytes",
        )
        _expect_sha256(
            pdf_record["pdf_sha256"],
            context=f"manifest PDF set record {pdf_id} SHA-256",
        )
        document = _expect_mapping(
            documents[index],
            context=f"manifest document {pdf_id}",
        )
        _validate_manifest_document(
            document,
            mode=str(mode),
            pdf_id=pdf_id,
            pdf_member=pdf_member,
            source_record=source,
            producer_record=producer,
            authority_record=authority_record,
            pdf_record=pdf_record,
        )
    if sum(
        int(record["bytes"]) for record in source_records
    ) > MAX_TOTAL_SOURCE_BYTES or any(
        int(record["bytes"]) > MAX_SOURCE_BUNDLE_BYTES for record in source_records
    ):
        _fail("manifest source-bundle set exceeds its byte bounds")
    if sum(
        int(record["bytes"]) for record in producer_records
    ) > MAX_TOTAL_PRODUCER_BYTES or any(
        int(record["bytes"]) > MAX_PRODUCER_BYTES for record in producer_records
    ):
        _fail("manifest producer set exceeds its byte bounds")
    if sum(
        int(record["bytes"]) for record in authority_records
    ) > MAX_TOTAL_AUTHORITY_BYTES or any(
        int(record["bytes"]) > MAX_AUTHORITY_RECEIPT_BYTES
        for record in authority_records
    ):
        _fail("manifest producer-toolchain authority set exceeds its byte bounds")
    if sum(
        int(record["pdf_bytes"]) * 2 for record in pdf_records
    ) > MAX_TOTAL_PDF_BYTES or any(
        int(record["pdf_bytes"]) > MAX_PDF_BYTES for record in pdf_records
    ):
        _fail("manifest PDF set exceeds its byte bounds")
    if len({str(record["pdf_sha256"]) for record in pdf_records}) != len(PDF_ORDER):
        _fail("manifest PDF set must contain four byte-distinct PDF anchors")
    if len(set(authority_bindings.values())) != len(PDF_ORDER):
        _fail("manifest authority bindings must be distinct")
    summary = _expect_mapping(manifest["summary"], context="manifest.summary")
    if summary != {
        "pdf_count": 4,
        "rebuild_count": 8,
        "invocation_receipt_count": 8,
        "adapter_invocation_count": 8,
    }:
        _fail("manifest.summary differs from the fixed four-document contract")


def _validate_tree(
    root: object,
    manifest_raw: bytes,
    *,
    directory_mode: int,
) -> list[dict[str, object]]:
    if (
        _read_output_member(root, MANIFEST_MEMBER, maximum=MAX_MANIFEST_BYTES)
        != manifest_raw
    ):
        _fail("on-disk derivation manifest differs from supplied bytes")
    try:
        parsed = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"derivation manifest is invalid JSON: {error}")
    manifest = _expect_mapping(parsed, context="derivation closure manifest")
    if _canonical_json(manifest) != manifest_raw:
        _fail("derivation closure manifest is not canonical JSON")
    _validate_manifest_semantics(manifest)
    inventory_raw = _expect_sequence(
        manifest["member_inventory"],
        context="manifest.member_inventory",
    )
    inventory: list[dict[str, object]] = []
    names: list[str] = []
    for index, raw_record in enumerate(inventory_raw):
        record = _expect_mapping(
            raw_record,
            context=f"manifest.member_inventory[{index}]",
        )
        _expect_keys(
            record,
            {"member", "bytes", "sha256"},
            context=f"manifest.member_inventory[{index}]",
        )
        member = str(record["member"])
        if member == MANIFEST_MEMBER:
            _fail("manifest member inventory contains an invalid member")
        _machine_call(
            _machine._relative_member,
            member,
            context="manifest member inventory member",
        )
        size = _expect_positive_int(
            record["bytes"],
            context=f"manifest member {member} bytes",
        )
        digest = _expect_sha256(
            record["sha256"],
            context=f"manifest member {member} SHA-256",
        )
        names.append(member)
        inventory.append({"member": member, "bytes": size, "sha256": digest})
    if names != sorted(set(names)) or len(names) != len(PDF_ORDER) * 4:
        _fail("manifest member inventory order or cardinality is invalid")
    inventory_by_member = {str(record["member"]): record for record in inventory}
    documents = _expect_sequence(manifest["documents"], context="manifest.documents")
    for index, (pdf_id, _pdf_member) in enumerate(PDF_ORDER):
        document = _expect_mapping(
            documents[index],
            context=f"manifest document {pdf_id}",
        )
        source = _expect_mapping(
            document["source_bundle"],
            context=f"manifest document {pdf_id} source",
        )
        producer = _expect_mapping(
            document["producer"],
            context=f"manifest document {pdf_id} producer",
        )
        authority = _expect_mapping(
            producer["adapter_authority"],
            context=f"manifest document {pdf_id} authority",
        )
        authority_sets = _expect_sequence(
            manifest["producer_toolchain_authority_set"],
            context="manifest producer-toolchain authority set",
        )
        authority_record = _expect_mapping(
            authority_sets[index],
            context=f"manifest producer-toolchain authority {pdf_id}",
        )
        runs = _expect_sequence(
            document["runs"],
            context=f"manifest document {pdf_id} runs",
        )
        for run_index in range(2):
            run = _expect_mapping(
                runs[run_index],
                context=f"manifest document {pdf_id} run {run_index}",
            )
            pdf_run_member = str(run["pdf_member"])
            pdf_raw = _read_output_member(
                root,
                pdf_run_member,
                maximum=MAX_PDF_BYTES,
            )
            pdf_size, pdf_sha256 = _validate_pdf_bytes(
                pdf_raw,
                member=pdf_run_member,
                expected_sha256=str(run["pdf_sha256"]),
            )
            if pdf_size != run["pdf_bytes"]:
                _fail(f"manifest document {pdf_id} PDF byte count is invalid")
            receipt_member = str(run["invocation_receipt_member"])
            receipt_raw = _read_output_member(
                root,
                receipt_member,
                maximum=MAX_INVOCATION_RECEIPT_BYTES,
            )
            executable_binding = _expect_mapping(
                run["executable_binding"],
                context=f"manifest document {pdf_id} executable binding",
            )
            expected_receipt = {
                "schema": INVOCATION_RECEIPT_SCHEMA,
                "status": "bounded-native-invocation-validated",
                "pdf_id": pdf_id,
                "source_bundle": {
                    "member": source["member"],
                    "bytes": source["bytes"],
                    "sha256": source["sha256"],
                },
                "producer": {
                    "member": producer["member"],
                    "bytes": producer["bytes"],
                    "sha256": producer["sha256"],
                    "toolchain_profile": producer["toolchain_profile"],
                },
                "producer_toolchain_authority": {
                    "member": authority_record["member"],
                    "bytes": authority_record["bytes"],
                    "sha256": authority_record["sha256"],
                    "authority_id": authority["authority_id"],
                    "status": authority["status"],
                    "authentication": authority["authentication"],
                    "review_scope": PRODUCER_AUTHORITY_REVIEW_SCOPE,
                },
                "producer_protocol": PRODUCER_PROTOCOL,
                "producer_arguments": run["arguments"],
                "descriptor_placeholders_bound_at_spawn": ["source_fd"],
                "pdf": {
                    "member": document["pdf_member"],
                    "bytes": pdf_size,
                    "sha256": pdf_sha256,
                },
                "return_code": 0,
                "stderr_bytes": 0,
                "executable_binding_sha256": _sha256(
                    _canonical_json(executable_binding),
                ),
                "origin": "derivation-closure-after-bounded-pdf-validation",
            }
            expected_raw = _canonical_json(expected_receipt) + b"\n"
            if receipt_raw != expected_raw:
                _fail(
                    f"manifest document {pdf_id} invocation receipt binding is invalid",
                )
            if run["invocation_receipt_bytes"] != len(receipt_raw) or run[
                "invocation_receipt_sha256"
            ] != _sha256(receipt_raw):
                _fail(
                    f"manifest document {pdf_id} invocation receipt digest or "
                    "byte count is invalid",
                )
            for member, raw_value in (
                (pdf_run_member, pdf_raw),
                (receipt_member, receipt_raw),
            ):
                record = inventory_by_member.get(member)
                if record is None or record != {
                    "member": member,
                    "bytes": len(raw_value),
                    "sha256": _sha256(raw_value),
                }:
                    _fail(f"manifest inventory cross-binding for {member} is invalid")
    files, directories, _ = _machine_call(
        _machine._walk_output,
        root,
        directory_mode=directory_mode,
    )
    if files != sorted([MANIFEST_MEMBER, *names]):
        _fail("derivation closure tree has missing or extra members")
    if sorted(directories) != [
        "runs",
        *[f"runs/{pdf_id}" for pdf_id in sorted(PDF_IDS)],
    ]:
        _fail("derivation closure tree has the wrong directory shape")
    if _member_inventory(root, names) != inventory:
        _fail("derivation closure member bytes differ from the manifest")
    if (
        _read_output_member(root, MANIFEST_MEMBER, maximum=MAX_MANIFEST_BYTES)
        != manifest_raw
    ):
        _fail("derivation manifest changed during tree validation")
    return inventory


def _validate_v2_manifest_semantics(manifest: Mapping[str, object]) -> None:
    context = "v2 derivation closure manifest"
    _validate_v2_json_scalar_types(manifest, context=context)
    _expect_keys(
        manifest,
        {
            "schema",
            "contract",
            "mode",
            "release_id",
            "status",
            "promotable",
            "promotion_blockers",
            "inputs",
            "execution_contract",
            "source_bundle_set",
            "source_bundle_set_sha256",
            "producer_package_set",
            "producer_package_set_sha256",
            "producer_set",
            "producer_set_sha256",
            "producer_toolchain_authority_set",
            "producer_toolchain_authority_set_sha256",
            "pdf_set",
            "pdf_set_sha256",
            "documents",
            "wrapper_integration",
            "non_inference_limits",
            "member_inventory",
            "summary",
            "builder",
            "native_execution_dependency",
            "payload_sha256",
        },
        context=context,
    )
    if (
        manifest["schema"] != DERIVATION_CLOSURE_SCHEMA_V2
        or manifest["contract"] != DERIVATION_CLOSURE_CONTRACT_V2
        or manifest["mode"] != MODE_REVISION
        or manifest["status"] != V2_CLOSURE_STATUS
        or manifest["promotable"] is not False
        or not _json_exact_equal(manifest["promotion_blockers"], V2_PROMOTION_BLOCKERS)
        or not _json_exact_equal(manifest["execution_contract"], EXECUTION_CONTRACT_V2)
        or not _json_exact_equal(
            manifest["non_inference_limits"],
            NON_INFERENCE_LIMITS_V2,
        )
    ):
        _fail("v2 derivation closure fixed semantics drifted or downgraded")
    _expect_token(manifest["release_id"], context=f"{context}.release_id")
    body = dict(manifest)
    payload_sha = _expect_sha256(
        body.pop("payload_sha256"),
        context=f"{context}.payload_sha256",
    )
    if _sha256(_canonical_json(body)) != payload_sha:
        _fail("v2 derivation closure payload digest is invalid")
    inputs = _expect_mapping(manifest["inputs"], context=f"{context}.inputs")
    _expect_keys(
        inputs,
        {
            "plan_sha256",
            "builder_bytes",
            "builder_sha256",
            "machine_runner_bytes",
            "machine_runner_sha256",
            "upstream_bindings",
        },
        context=f"{context}.inputs",
    )
    for key in ("plan_sha256", "builder_sha256", "machine_runner_sha256"):
        _expect_sha256(inputs[key], context=f"{context}.inputs.{key}")
    builder_bytes = _expect_positive_int(
        inputs["builder_bytes"],
        context=f"{context}.builder_bytes",
    )
    machine_runner_bytes = _expect_positive_int(
        inputs["machine_runner_bytes"],
        context=f"{context}.machine_runner_bytes",
    )
    if builder_bytes > MAX_PLAN_BYTES or machine_runner_bytes > MAX_PLAN_BYTES:
        _fail(f"{context} live-code record exceeds its byte bound")
    _normalize_sha_map(
        _expect_mapping(
            inputs["upstream_bindings"],
            context=f"{context}.upstream_bindings",
        ),  # type: ignore[arg-type]
        keys=UPSTREAM_BINDING_KEYS,
        context=f"{context}.upstream_bindings",
    )
    builder = _expect_mapping(manifest["builder"], context=f"{context}.builder")
    machine = _expect_mapping(
        manifest["native_execution_dependency"],
        context=f"{context}.native_execution_dependency",
    )
    if builder != {
        "member": BUILDER_MEMBER,
        "bytes": inputs["builder_bytes"],
        "sha256": inputs["builder_sha256"],
    } or machine != {
        "member": MACHINE_RUNNER_MEMBER,
        "bytes": inputs["machine_runner_bytes"],
        "sha256": inputs["machine_runner_sha256"],
    }:
        _fail("v2 manifest live-code records differ from inputs")
    if manifest["wrapper_integration"] != {
        "status": "not-integrated-requires-downstream-promotion-closure-v2",
        "accepted_native_derivation_schema": DERIVATION_CLOSURE_SCHEMA_V2,
        "v1_accepted_for_promotion": False,
        "production_blocker": V2_PROMOTION_BLOCKERS[0],
    }:
        _fail("v2 manifest wrapper-integration boundary drifted")

    set_specs = (
        ("source_bundle_set", "source_bundle_set_sha256"),
        ("producer_package_set", "producer_package_set_sha256"),
        ("producer_set", "producer_set_sha256"),
        (
            "producer_toolchain_authority_set",
            "producer_toolchain_authority_set_sha256",
        ),
        ("pdf_set", "pdf_set_sha256"),
    )
    sets: dict[str, Sequence[object]] = {}
    for set_key, digest_key in set_specs:
        records = _expect_sequence(manifest[set_key], context=f"{context}.{set_key}")
        if len(records) != len(PDF_ORDER):
            _fail(f"{context}.{set_key} must contain exactly four roles")
        if _sha256(_canonical_json(records)) != _expect_sha256(
            manifest[digest_key],
            context=f"{context}.{digest_key}",
        ):
            _fail(f"{context}.{set_key} digest is invalid")
        sets[set_key] = records
    documents = _expect_sequence(manifest["documents"], context=f"{context}.documents")
    if len(documents) != len(PDF_ORDER):
        _fail("v2 manifest documents must contain exactly four roles")
    role_anchor_values: dict[str, list[str]] = {
        key: []
        for key in (
            "authority",
            "source",
            "producer",
            "package",
            "renderer_manifest",
            "pdf",
        )
    }
    for index, (pdf_id, pdf_member) in enumerate(PDF_ORDER):
        source = _expect_mapping(
            sets["source_bundle_set"][index],
            context=f"{context}.source_bundle_set[{index}]",
        )
        package = _expect_mapping(
            sets["producer_package_set"][index],
            context=f"{context}.producer_package_set[{index}]",
        )
        producer = _expect_mapping(
            sets["producer_set"][index],
            context=f"{context}.producer_set[{index}]",
        )
        authority = _expect_mapping(
            sets["producer_toolchain_authority_set"][index],
            context=f"{context}.authority_set[{index}]",
        )
        pdf = _expect_mapping(
            sets["pdf_set"][index],
            context=f"{context}.pdf_set[{index}]",
        )
        document = _expect_mapping(
            documents[index],
            context=f"{context}.documents[{index}]",
        )
        _expect_keys(
            source,
            {
                "pdf_id",
                "member",
                "mode",
                "owner",
                "link_count",
                "bytes",
                "sha256",
                "treatment",
            },
            context=f"{context}.source_bundle_set[{index}]",
        )
        _expect_keys(
            package,
            {
                "pdf_id",
                "package_content_sha256",
                "root_mode",
                "root_owner",
                "members",
            },
            context=f"{context}.producer_package_set[{index}]",
        )
        _expect_keys(
            producer,
            {
                "pdf_id",
                "member",
                "mode",
                "bytes",
                "sha256",
                "macho_uuid",
                "native_code_directory",
            },
            context=f"{context}.producer_set[{index}]",
        )
        _expect_keys(
            authority,
            {
                "pdf_id",
                "member",
                "mode",
                "bytes",
                "sha256",
                "schema",
                "status",
                "authentication",
                "caller_anchors",
                *V2_AUTHORITY_PROJECTION_KEYS,
            },
            context=f"{context}.authority_set[{index}]",
        )
        _expect_keys(
            pdf,
            {"pdf_id", "pdf_member", "pdf_bytes", "pdf_sha256"},
            context=f"{context}.pdf_set[{index}]",
        )
        _expect_keys(
            document,
            {
                "pdf_id",
                "pdf_member",
                "pdf_bytes",
                "pdf_sha256",
                "producer_arguments",
                "source_bundle",
                "producer_package",
                "authorization",
                "expected_renderer_manifest",
                "runs",
                "native_closure_projection",
            },
            context=f"{context}.documents[{index}]",
        )
        role_records = (source, package, producer, authority, pdf, document)
        if any(record.get("pdf_id") != pdf_id for record in role_records):
            _fail(f"v2 manifest role order drifted at {pdf_id}")
        source_bytes = _expect_positive_int(
            source["bytes"],
            context=f"v2 manifest source {pdf_id} bytes",
        )
        source_sha256 = _expect_sha256(
            source["sha256"],
            context=f"v2 manifest source {pdf_id} SHA-256",
        )
        producer_bytes = _expect_positive_int(
            producer["bytes"],
            context=f"v2 manifest producer {pdf_id} bytes",
        )
        producer_sha256 = _expect_sha256(
            producer["sha256"],
            context=f"v2 manifest producer {pdf_id} SHA-256",
        )
        authority_bytes = _expect_positive_int(
            authority["bytes"],
            context=f"v2 manifest authority {pdf_id} bytes",
        )
        authority_sha256 = _expect_sha256(
            authority["sha256"],
            context=f"v2 manifest authority {pdf_id} SHA-256",
        )
        for projection_key in V2_AUTHORITY_PROJECTION_KEYS:
            _expect_sha256(
                authority[projection_key],
                context=f"v2 manifest authority {pdf_id}.{projection_key}",
            )
        pdf_bytes = _expect_positive_int(
            pdf["pdf_bytes"],
            context=f"v2 manifest PDF {pdf_id} bytes",
        )
        pdf_sha256 = _expect_sha256(
            pdf["pdf_sha256"],
            context=f"v2 manifest PDF {pdf_id} SHA-256",
        )
        _expect_sha256(
            package["package_content_sha256"],
            context=f"v2 manifest package {pdf_id} SHA-256",
        )
        role_anchor_values["authority"].append(authority_sha256)
        role_anchor_values["source"].append(source_sha256)
        role_anchor_values["producer"].append(producer_sha256)
        role_anchor_values["package"].append(
            str(package["package_content_sha256"]),
        )
        role_anchor_values["pdf"].append(pdf_sha256)
        package_members = _expect_sequence(
            package["members"],
            context=f"v2 manifest package {pdf_id} members",
        )
        if len(package_members) != 2:
            _fail(f"v2 manifest package {pdf_id} must have exactly two members")
        for member_index, raw_member in enumerate(package_members):
            member_record = _expect_mapping(
                raw_member,
                context=f"v2 manifest package {pdf_id} member {member_index}",
            )
            _expect_keys(
                member_record,
                {"member", "mode", "owner", "link_count", "bytes", "sha256"},
                context=f"v2 manifest package {pdf_id} member {member_index}",
            )
            if (
                member_record["owner"] != "effective-user-id"
                or _expect_positive_int(
                    member_record["link_count"],
                    context=(
                        f"v2 manifest package {pdf_id} member {member_index} link count"
                    ),
                )
                != 1
            ):
                _fail(f"v2 manifest package {pdf_id} ownership seal drifted")
            _expect_positive_int(
                member_record["bytes"],
                context=f"v2 manifest package {pdf_id} member {member_index} bytes",
            )
            _expect_sha256(
                member_record["sha256"],
                context=f"v2 manifest package {pdf_id} member {member_index} SHA-256",
            )
        if (
            source.get("member") != SOURCE_MEMBER_BY_ID[pdf_id]
            or source.get("mode") != V2_SOURCE_MODE
            or source.get("owner") != "effective-user-id"
            or _expect_positive_int(
                source.get("link_count"),
                context=f"v2 manifest source {pdf_id} link count",
            )
            != 1
            or source.get("treatment") != "opaque-byte-bundle-not-decoded"
            or source_bytes > MAX_V2_NATIVE_SOURCE_BUNDLE_BYTES
        ):
            _fail(f"v2 manifest source record {pdf_id} drifted")
        expected_package_record = _v2_package_content_record(
            pdf_id,
            producer_bytes=producer_bytes,
            producer_sha256=producer_sha256,
            authority_bytes=authority_bytes,
            authority_sha256=authority_sha256,
        )
        expected_package_digest = _sha256(_canonical_json(expected_package_record))
        if package != {
            "pdf_id": pdf_id,
            "package_content_sha256": expected_package_digest,
            **expected_package_record,
        }:
            _fail(f"v2 manifest package record {pdf_id} drifted")
        _normalize_v2_code_directory(
            producer["native_code_directory"],
            producer_bytes=producer_bytes,
            context=f"v2 manifest producer CodeDirectory {pdf_id}",
        )
        if (
            producer.get("member") != PRODUCER_MEMBER_BY_ID[pdf_id]
            or producer.get("mode") != V2_PRODUCER_MODE
            or producer_bytes > MAX_V2_NATIVE_EXECUTABLE_BYTES
            or not isinstance(producer.get("macho_uuid"), str)
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                str(producer.get("macho_uuid")),
            )
            is None
            or authority.get("member") != AUTHORITY_MEMBER_BY_ID[pdf_id]
            or authority.get("mode") != V2_AUTHORITY_MODE
            or authority_bytes > MAX_V2_AUTHORITY_BYTES
            or authority.get("schema") != NATIVE_PRODUCER_AUTHORITY_SCHEMA_BY_ID[pdf_id]
            or authority.get("status") != V2_AUTHORITY_STATUS
            or authority.get("authentication") != "caller-sha-anchor-only"
            or pdf
            != {
                "pdf_id": pdf_id,
                "pdf_member": pdf_member,
                "pdf_bytes": pdf_bytes,
                "pdf_sha256": pdf_sha256,
            }
            or document["pdf_bytes"] != pdf_bytes
            or document["pdf_sha256"] != pdf_sha256
        ):
            _fail(f"v2 manifest producer, authority, or PDF record {pdf_id} drifted")
        if (
            document["pdf_member"] != pdf_member
            or document["producer_arguments"] != _expected_arguments(pdf_id)
            or document["source_bundle"]
            != {key: value for key, value in source.items() if key != "pdf_id"}
        ):
            _fail(f"v2 manifest document {pdf_id} source or member drifted")
        document_package = _expect_mapping(
            document["producer_package"],
            context=f"v2 manifest document package {pdf_id}",
        )
        _expect_keys(
            document_package,
            {
                "root_mode",
                "root_owner",
                "package_content_sha256",
                "producer",
                "authority",
                "authority_projection",
            },
            context=f"v2 manifest document package {pdf_id}",
        )
        expected_document_authority = {
            key: authority[key]
            for key in ("member", "mode", "bytes", "sha256", "schema")
        }
        expected_authority_projection = {
            key: authority[key] for key in V2_AUTHORITY_PROJECTION_KEYS
        }
        if document_package != {
            "root_mode": package["root_mode"],
            "root_owner": package["root_owner"],
            "package_content_sha256": package["package_content_sha256"],
            "producer": {
                key: value for key, value in producer.items() if key != "pdf_id"
            },
            "authority": expected_document_authority,
            "authority_projection": expected_authority_projection,
        }:
            _fail(f"v2 manifest document package {pdf_id} cross-binding drifted")
        authorization = _expect_mapping(
            document["authorization"],
            context=f"v2 manifest document authorization {pdf_id}",
        )
        if authorization != {
            "status": V2_AUTHORIZATION_STATUS,
            "authentication": "caller-sha-anchor-only",
            "authority_sha256": authority["sha256"],
        }:
            _fail(f"v2 manifest document authorization {pdf_id} drifted")
        authority_caller_anchors = _normalize_sha_map(
            _expect_mapping(
                authority.get("caller_anchors"),
                context=f"v2 manifest authority caller anchors {pdf_id}",
            ),  # type: ignore[arg-type]
            keys=V2_CALLER_ANCHOR_KEYS,
            context=f"v2 manifest authority caller anchors {pdf_id}",
        )
        renderer_manifest = _normalize_v2_output_record(
            document["expected_renderer_manifest"],
            context=f"v2 manifest renderer manifest {pdf_id}",
            expected_member="render-receipt.json",
            expected_sha256=authority_caller_anchors["renderer_manifest_sha256"],
        )
        role_anchor_values["renderer_manifest"].append(
            str(renderer_manifest["sha256"]),
        )
        if (
            authority_caller_anchors["source_bundle_sha256"] != source["sha256"]
            or authority_caller_anchors["pdf_sha256"] != pdf["pdf_sha256"]
            or authority_caller_anchors["machine_runner_sha256"]
            != inputs["machine_runner_sha256"]
        ):
            _fail(f"v2 manifest authority caller anchors {pdf_id} drifted")
        runs = _expect_sequence(
            document["runs"],
            context=f"v2 manifest document runs {pdf_id}",
        )
        if len(runs) != 2:
            _fail(f"v2 manifest document {pdf_id} must contain two rebuilds")
        for run_index, label in enumerate(("a", "b")):
            run = _expect_mapping(
                runs[run_index],
                context=f"v2 manifest document {pdf_id} run {label}",
            )
            _expect_keys(
                run,
                {
                    "run",
                    "pdf_member",
                    "pdf_bytes",
                    "pdf_sha256",
                    "invocation_receipt_member",
                    "invocation_receipt_bytes",
                    "invocation_receipt_sha256",
                    "producer_arguments_realized",
                    "launcher_pre_exec_attestation",
                },
                context=f"v2 manifest document {pdf_id} run {label}",
            )
            run_pdf_bytes = _expect_positive_int(
                run["pdf_bytes"],
                context=f"v2 manifest document {pdf_id} run {label} PDF bytes",
            )
            receipt_bytes = _expect_positive_int(
                run["invocation_receipt_bytes"],
                context=(f"v2 manifest document {pdf_id} run {label} receipt bytes"),
            )
            _expect_sha256(
                run["invocation_receipt_sha256"],
                context=(f"v2 manifest document {pdf_id} run {label} receipt SHA-256"),
            )
            if (
                run["run"] != label
                or run["pdf_member"] != f"runs/{pdf_id}/rebuild-{label}.pdf"
                or run_pdf_bytes != document["pdf_bytes"]
                or run_pdf_bytes > MAX_V2_NATIVE_OUTPUT_BYTES
                or run["pdf_sha256"] != document["pdf_sha256"]
                or run["invocation_receipt_member"]
                != f"runs/{pdf_id}/rebuild-{label}.receipt.json"
                or receipt_bytes > MAX_INVOCATION_RECEIPT_BYTES
                or run["producer_arguments_realized"]
                != _adapter_arguments(pdf_id, V2_SOURCE_DESCRIPTOR)
            ):
                _fail(f"v2 manifest document {pdf_id} run {label} drifted")
            attestation = _expect_mapping(
                run["launcher_pre_exec_attestation"],
                context=f"v2 manifest document {pdf_id} run {label} attestation",
            )
            normalized_attestation = _normalize_public_v2_attestation(
                attestation,
                expected_code_directory=_expect_mapping(
                    producer["native_code_directory"],
                    context=f"v2 manifest producer {pdf_id} CodeDirectory",
                ),
                macho_uuid=str(producer["macho_uuid"]),
                context=f"v2 manifest document {pdf_id} run {label} attestation",
            )
            if normalized_attestation != attestation:
                _fail(f"v2 manifest document {pdf_id} attestation is not normalized")
            mapping = _expect_mapping(
                attestation.get("main_executable_mapping"),
                context=f"v2 manifest document {pdf_id} path-free mapping",
            )
            if "path" in mapping or mapping.get("path_recorded") is not False:
                _fail(f"v2 manifest document {pdf_id} exposes an executable path")
        first_run = _expect_mapping(
            runs[0],
            context=f"v2 manifest document {pdf_id} first run",
        )
        second_run = _expect_mapping(
            runs[1],
            context=f"v2 manifest document {pdf_id} second run",
        )
        if (
            first_run["invocation_receipt_bytes"]
            != second_run["invocation_receipt_bytes"]
            or first_run["invocation_receipt_sha256"]
            != second_run["invocation_receipt_sha256"]
            or not _json_exact_equal(
                first_run["launcher_pre_exec_attestation"],
                second_run["launcher_pre_exec_attestation"],
            )
        ):
            _fail(
                f"v2 manifest document {pdf_id} rebuild receipts or "
                "attestations differ",
            )
        native_projection = _expect_mapping(
            document["native_closure_projection"],
            context=f"v2 manifest document {pdf_id} native projection",
        )
        expected_native_projection = {
            "invocation_receipt_sha256": first_run["invocation_receipt_sha256"],
            "invocation_receipt_bytes": first_run["invocation_receipt_bytes"],
            "rebuild_a_sha256": first_run["pdf_sha256"],
            "rebuild_b_sha256": second_run["pdf_sha256"],
            "status": V2_CLOSURE_STATUS,
        }
        if not _json_exact_equal(native_projection, expected_native_projection):
            _fail(f"v2 manifest document {pdf_id} native projection drifted")
    for label, values in role_anchor_values.items():
        if len(set(values)) != len(PDF_ORDER):
            _fail(f"v2 manifest {label} anchors must be role-distinct")
    inventory_records = _expect_sequence(
        manifest["member_inventory"],
        context=f"{context}.member_inventory",
    )
    if len(inventory_records) != len(PDF_ORDER) * 4:
        _fail("v2 manifest member inventory has the wrong cardinality")
    inventory_names: list[str] = []
    for index, raw_record in enumerate(inventory_records):
        record = _expect_mapping(
            raw_record,
            context=f"{context}.member_inventory[{index}]",
        )
        _expect_keys(
            record,
            {"member", "bytes", "sha256"},
            context=f"{context}.member_inventory[{index}]",
        )
        member = record["member"]
        if not isinstance(member, str):
            _fail(f"{context}.member_inventory[{index}].member must be a string")
        _machine_call(_machine._relative_member, member, context="v2 inventory member")
        member_bytes = _expect_positive_int(
            record["bytes"],
            context=f"{context}.member_inventory[{index}].bytes",
        )
        maximum = (
            MAX_V2_NATIVE_OUTPUT_BYTES
            if member.endswith(".pdf")
            else MAX_INVOCATION_RECEIPT_BYTES
        )
        if member_bytes > maximum:
            _fail(f"{context}.member_inventory[{index}] exceeds its byte bound")
        _expect_sha256(
            record["sha256"],
            context=f"{context}.member_inventory[{index}].sha256",
        )
        inventory_names.append(member)
    if inventory_names != sorted(set(inventory_names)):
        _fail("v2 manifest member inventory is not unique and sorted")
    summary = _expect_mapping(manifest["summary"], context=f"{context}.summary")
    if summary != {
        "pdf_count": 4,
        "rebuild_count": 8,
        "invocation_receipt_count": 8,
        "producer_invocation_count": 8,
        "package_count": 4,
    }:
        _fail("v2 manifest summary differs from the fixed four-role contract")


def _validate_v2_tree(
    root: object,
    manifest_raw: bytes,
    *,
    directory_mode: int,
) -> list[dict[str, object]]:
    observed_manifest = _read_output_member(
        root,
        MANIFEST_MEMBER,
        maximum=MAX_MANIFEST_BYTES,
    )
    if observed_manifest != manifest_raw:
        _fail("on-disk v2 derivation manifest differs from supplied bytes")
    try:
        parsed = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"v2 derivation manifest is invalid JSON: {error}")
    manifest = _expect_mapping(parsed, context="v2 derivation closure manifest")
    if _canonical_json(manifest) != manifest_raw:
        _fail("v2 derivation closure manifest is not canonical JSON")
    _reject_v2_forbidden_fragments(manifest_raw, context="v2 derivation manifest")
    _validate_v2_manifest_semantics(manifest)
    inventory_raw = _expect_sequence(
        manifest["member_inventory"],
        context="v2 manifest.member_inventory",
    )
    inventory: list[dict[str, object]] = []
    names: list[str] = []
    for index, raw_record in enumerate(inventory_raw):
        record = _expect_mapping(
            raw_record,
            context=f"v2 manifest.member_inventory[{index}]",
        )
        _expect_keys(
            record,
            {"member", "bytes", "sha256"},
            context=f"v2 manifest.member_inventory[{index}]",
        )
        member = str(record["member"])
        if member == MANIFEST_MEMBER:
            _fail("v2 manifest inventory may not contain itself")
        _machine_call(_machine._relative_member, member, context="v2 inventory member")
        normalized = {
            "member": member,
            "bytes": _expect_positive_int(
                record["bytes"],
                context=f"v2 inventory {member} bytes",
            ),
            "sha256": _expect_sha256(
                record["sha256"],
                context=f"v2 inventory {member} SHA-256",
            ),
        }
        inventory.append(normalized)
        names.append(member)
    if names != sorted(set(names)) or len(names) != len(PDF_ORDER) * 4:
        _fail("v2 manifest inventory order or cardinality is invalid")
    inventory_by_member = {str(record["member"]): record for record in inventory}
    documents = _expect_sequence(manifest["documents"], context="v2 manifest.documents")
    for index, (pdf_id, _pdf_member) in enumerate(PDF_ORDER):
        document = _expect_mapping(documents[index], context=f"v2 document {pdf_id}")
        runs = _expect_sequence(document["runs"], context=f"v2 document {pdf_id} runs")
        for run_index, label in enumerate(("a", "b")):
            run = _expect_mapping(runs[run_index], context=f"v2 {pdf_id} run {label}")
            pdf_member = str(run["pdf_member"])
            pdf_raw = _read_output_member(
                root,
                pdf_member,
                maximum=MAX_V2_NATIVE_OUTPUT_BYTES,
            )
            pdf_bytes, pdf_sha = _validate_pdf_bytes(
                pdf_raw,
                member=pdf_member,
                expected_sha256=str(run["pdf_sha256"]),
            )
            if pdf_bytes != run["pdf_bytes"]:
                _fail(f"v2 document {pdf_id} PDF byte count is invalid")
            receipt_member = str(run["invocation_receipt_member"])
            receipt_raw = _read_output_member(
                root,
                receipt_member,
                maximum=MAX_INVOCATION_RECEIPT_BYTES,
            )
            expected_receipt = _invocation_receipt_v2(
                pdf_id=pdf_id,
                document=document,
                pdf_bytes=pdf_bytes,
                pdf_sha256=pdf_sha,
                return_code=0,
                attestation=_expect_mapping(
                    run["launcher_pre_exec_attestation"],
                    context=f"v2 {pdf_id} run {label} attestation",
                ),
            )
            if receipt_raw != expected_receipt:
                _fail(f"v2 document {pdf_id} invocation receipt binding is invalid")
            if run["invocation_receipt_bytes"] != len(receipt_raw) or run[
                "invocation_receipt_sha256"
            ] != _sha256(receipt_raw):
                _fail(f"v2 document {pdf_id} receipt digest or size is invalid")
            for member, raw in ((pdf_member, pdf_raw), (receipt_member, receipt_raw)):
                if inventory_by_member.get(member) != {
                    "member": member,
                    "bytes": len(raw),
                    "sha256": _sha256(raw),
                }:
                    _fail(f"v2 inventory cross-binding for {member} is invalid")
    files, directories, _ = _machine_call(
        _machine._walk_output,
        root,
        directory_mode=directory_mode,
    )
    if files != sorted([MANIFEST_MEMBER, *names]):
        _fail("v2 derivation closure tree has missing or extra members")
    if sorted(directories) != [
        "runs",
        *[f"runs/{pdf_id}" for pdf_id in sorted(PDF_IDS)],
    ]:
        _fail("v2 derivation closure tree has the wrong directory shape")
    if _member_inventory(root, names) != inventory:
        _fail("v2 derivation closure member bytes differ from the manifest")
    observed_manifest = _read_output_member(
        root,
        MANIFEST_MEMBER,
        maximum=MAX_MANIFEST_BYTES,
    )
    if observed_manifest != manifest_raw:
        _fail("v2 derivation manifest changed during tree validation")
    return inventory


def _reserve_directory(
    target: Path,
    *,
    reserved_name: str,
    context: str,
) -> tuple[Path, object, object]:
    return _machine_call(
        _machine._reserve_directory,
        target,
        reserved_name=reserved_name,
        context=context,
    )  # type: ignore[return-value]


def _revalidate_reserved(
    parent: object,
    child: object,
    name: str,
    *,
    context: str,
    modes: set[int],
) -> None:
    _machine_call(
        _machine._revalidate_reserved_directory,
        parent,
        child,
        name,
        context=context,
        allowed_modes=modes,
    )


def _rename_no_replace(source: str, destination: str, parent_descriptor: int) -> None:
    _machine_call(_machine._rename_no_replace, source, destination, parent_descriptor)


def _publish_directory(
    parent: object,
    stage: object,
    stage_name: str,
    destination: Path,
    *,
    manifest_raw: bytes,
    tree_validator: Callable[..., object] = _validate_tree,
) -> None:
    renamed = False
    try:
        _revalidate_reserved(
            parent,
            stage,
            stage_name,
            context="sealed derivation candidate",
            modes={0o500},
        )
        tree_validator(stage, manifest_raw, directory_mode=0o500)
        try:
            _rename_no_replace(stage_name, destination.name, parent.descriptor)
        except BaseException as error:
            message = (
                f"{error}; candidate_paths={stage.path}|{destination}; "
                "candidate_state=rename-issued-outcome-ambiguous-do-not-auto-delete; "
                "inspect both names and identities before explicit removal"
            )
            raise DerivationClosureError(message) from error
        renamed = True
        os.fsync(parent.descriptor)
        parent.mtime_ns = os.fstat(parent.descriptor).st_mtime_ns
        _machine_call(
            _machine._revalidate_root,
            parent,
            context="derivation destination parent",
        )
        named = _machine_call(
            _machine._named_directory_identity,
            parent,
            destination.name,
            context="published derivation closure",
        )
        if named != (stage.device, stage.inode, 0o500):
            _fail("published derivation closure identity changed")
        tree_validator(stage, manifest_raw, directory_mode=0o500)
    except BaseException as error:
        if isinstance(error, DerivationClosureError) and "candidate_paths=" in str(
            error,
        ):
            raise
        candidate = destination if renamed else stage.path
        state = (
            "destination-name-may-be-owned-or-replaced-do-not-auto-delete"
            if renamed
            else "private-stage-name-may-be-owned-or-replaced-do-not-auto-delete"
        )
        message = (
            f"{error}; candidate_path={candidate}; candidate_state={state}; "
            "inspect identity and inventory before explicit removal"
        )
        raise DerivationClosureError(message) from error


def _v2_candidate_diagnostic(
    parent: object,
    stage: object,
    stage_name: str,
    destination: Path,
) -> str:
    """Describe the retained name without claiming a stale post-rename path."""
    try:
        _machine_call(
            _machine._revalidate_root,
            parent,
            context="v2 retained candidate parent-to-path mapping",
        )
        destination_identity = _machine_call(
            _machine._named_directory_identity,
            parent,
            destination.name,
            context="v2 retained published destination",
        )
        stage_identity = _machine_call(
            _machine._named_directory_identity,
            parent,
            stage_name,
            context="v2 retained private candidate",
        )
    except (OSError, DerivationClosureError):
        return (
            f"candidate_names={stage_name}|{destination.name}; "
            f"held_parent_descriptor={parent.descriptor}; "
            "candidate_state=parent-path-or-name-mapping-inspection-failed-or-"
            "ambiguous-do-not-auto-delete"
        )
    expected = (stage.device, stage.inode)
    destination_bound = (
        destination_identity is not None and destination_identity[:2] == expected
    )
    stage_bound = stage_identity is not None and stage_identity[:2] == expected
    if destination_bound and not stage_bound:
        return (
            f"candidate_path={destination}; "
            "candidate_state=published-destination-remains-identity-bound-do-not-"
            "auto-delete"
        )
    if stage_bound and not destination_bound:
        return (
            f"candidate_path={stage.path}; "
            "candidate_state=private-stage-remains-identity-bound-do-not-auto-delete"
        )
    return (
        f"candidate_paths={stage.path}|{destination}; "
        "candidate_state=name-mapping-lost-duplicated-or-ambiguous-do-not-auto-delete"
    )


def _v2_publication_error_detail(error: BaseException) -> str:
    detail = str(error)
    for marker in ("; candidate_path=", "; candidate_paths="):
        if marker in detail:
            return detail.split(marker, 1)[0]
    return detail


def _terminal_revalidate_published_v2(
    parent: object,
    stage: object,
    destination: Path,
    production: _Production,
    receipt: DerivationClosureReceiptV2,
    *,
    receipt_root: Path,
) -> None:
    """Perform the final FD- and destination-name-bound publication readback."""
    if destination.parent != parent.path:
        _fail("v2 published destination parent differs from its pinned parent")
    expected_receipt = _receipt_v2(
        receipt_root,
        production,
        replay_root=Path(receipt.replay_root) if receipt.replay_root else None,
    )
    if receipt != expected_receipt:
        _fail("v2 terminal publication receipt differs from the published closure")
    expected_identity = (stage.device, stage.inode, 0o500)
    for phase in ("before-terminal-tree-readback", "after-terminal-tree-readback"):
        _machine_call(
            _machine._revalidate_root,
            parent,
            context=f"v2 publication parent {phase}",
        )
        opened = os.fstat(stage.descriptor)
        if (
            (opened.st_dev, opened.st_ino, stat.S_IMODE(opened.st_mode))
            != expected_identity
            or not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
        ):
            _fail(f"v2 published root identity drifted {phase}")
        named = _machine_call(
            _machine._named_directory_identity,
            parent,
            destination.name,
            context=f"v2 published destination {phase}",
        )
        if named != expected_identity:
            _fail(f"v2 published destination mapping drifted {phase}")
        if phase == "before-terminal-tree-readback":
            _validate_v2_tree(
                stage,
                production.manifest_raw,
                directory_mode=0o500,
            )


def _receipt(
    root: Path,
    production: _Production,
    *,
    replay_root: Path | None,
) -> DerivationClosureReceipt:
    manifest = production.manifest
    summary = _expect_mapping(manifest["summary"], context="manifest.summary")
    return DerivationClosureReceipt(
        manifest_path=str(root / MANIFEST_MEMBER),
        manifest_sha256=_sha256(production.manifest_raw),
        plan_sha256=str(
            _expect_mapping(manifest["inputs"], context="manifest.inputs")[
                "plan_sha256"
            ],
        ),
        source_bundle_set_sha256=str(manifest["source_bundle_set_sha256"]),
        producer_set_sha256=str(manifest["producer_set_sha256"]),
        producer_toolchain_authority_set_sha256=str(
            manifest["producer_toolchain_authority_set_sha256"],
        ),
        pdf_set_sha256=str(manifest["pdf_set_sha256"]),
        pdf_count=int(summary["pdf_count"]),
        rebuild_count=int(summary["rebuild_count"]),
        mode=str(manifest["mode"]),
        promotable=bool(manifest["promotable"]),
        replay_root=str(replay_root) if replay_root is not None else None,
    )


def _receipt_v2(
    root: Path,
    production: _Production,
    *,
    replay_root: Path | None,
) -> DerivationClosureReceiptV2:
    manifest = production.manifest
    summary = _expect_mapping(manifest["summary"], context="v2 manifest.summary")
    inputs = _expect_mapping(manifest["inputs"], context="v2 manifest.inputs")
    return DerivationClosureReceiptV2(
        manifest_path=str(root / MANIFEST_MEMBER),
        manifest_sha256=_sha256(production.manifest_raw),
        plan_sha256=str(inputs["plan_sha256"]),
        source_bundle_set_sha256=str(manifest["source_bundle_set_sha256"]),
        producer_package_set_sha256=str(manifest["producer_package_set_sha256"]),
        producer_set_sha256=str(manifest["producer_set_sha256"]),
        producer_toolchain_authority_set_sha256=str(
            manifest["producer_toolchain_authority_set_sha256"],
        ),
        pdf_set_sha256=str(manifest["pdf_set_sha256"]),
        pdf_count=int(summary["pdf_count"]),
        rebuild_count=int(summary["rebuild_count"]),
        mode=str(manifest["mode"]),
        promotable=bool(manifest["promotable"]),
        replay_root=str(replay_root) if replay_root is not None else None,
    )


def build_derivation_closure_v2(
    plan_path: Path,
    source_root: Path,
    package_roots: Mapping[str, Path],
    destination: Path,
    *,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_package_sha256: Mapping[str, str],
    expected_renderer_manifest_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> DerivationClosureReceiptV2:
    """Build one revision-only closure from four exact sealed native packages."""
    destination_absolute = destination.absolute()
    stage_name = f".{destination_absolute.name}.private-v2-candidate"
    absolute, parent, stage = _reserve_directory(
        destination_absolute,
        reserved_name=stage_name,
        context="v2 derivation closure",
    )
    inputs: _V2Inputs | None = None

    def output_guard() -> None:
        _revalidate_reserved(
            parent,
            stage,
            stage_name,
            context="v2 derivation private candidate",
            modes={0o700, 0o500},
        )

    try:
        try:
            inputs = _pin_v2_inputs(
                plan_path,
                source_root,
                package_roots,
                stage.path,
                release_id=release_id,
                expected_plan_sha256=expected_plan_sha256,
                expected_builder_sha256=expected_builder_sha256,
                expected_machine_runner_sha256=expected_machine_runner_sha256,
                expected_upstream_sha256=expected_upstream_sha256,
                expected_authority_sha256=expected_authority_sha256,
                expected_source_sha256=expected_source_sha256,
                expected_producer_sha256=expected_producer_sha256,
                expected_package_sha256=expected_package_sha256,
                expected_renderer_manifest_sha256=(expected_renderer_manifest_sha256),
                expected_pdf_sha256=expected_pdf_sha256,
            )
            production = _produce_v2_with_inputs(
                inputs,
                stage,
                output_guard=output_guard,
            )
            _revalidate_v2_inputs(inputs)
        except BaseException as error:
            diagnostic = _v2_candidate_diagnostic(
                parent,
                stage,
                stage_name,
                absolute,
            )
            detail = _v2_publication_error_detail(error)
            message = (
                f"{detail}; {diagnostic}; inspect identity and inventory before "
                "explicit removal"
            )
            raise DerivationClosureError(message) from error
        try:
            _publish_directory(
                parent,
                stage,
                stage_name,
                absolute,
                manifest_raw=production.manifest_raw,
                tree_validator=_validate_v2_tree,
            )
            receipt = _receipt_v2(absolute, production, replay_root=None)
            _revalidate_v2_inputs(inputs)
            _terminal_revalidate_published_v2(
                parent,
                stage,
                absolute,
                production,
                receipt,
                receipt_root=absolute,
            )
        except BaseException as error:
            diagnostic = _v2_candidate_diagnostic(
                parent,
                stage,
                stage_name,
                absolute,
            )
            detail = _v2_publication_error_detail(error)
            message = (
                f"{detail}; {diagnostic}; inspect identity and inventory before "
                "explicit removal"
            )
            raise DerivationClosureError(message) from error
        return receipt
    finally:
        _close_resources(
            [
                ("v2 derivation inputs", inputs),
                ("v2 derivation candidate", stage),
                ("v2 derivation destination parent", parent),
            ],
            context="v2 derivation publication",
        )


def build_derivation_closure(
    plan_path: Path,
    source_root: Path,
    producer_root: Path,
    authority_root: Path,
    destination: Path,
    *,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> DerivationClosureReceipt:
    """Build and atomically publish one authorized native derivation closure."""
    destination_absolute = destination.absolute()
    stage_name = f".{destination_absolute.name}.private-candidate"
    absolute, parent, stage = _reserve_directory(
        destination_absolute,
        reserved_name=stage_name,
        context="derivation closure",
    )

    def output_guard() -> None:
        _revalidate_reserved(
            parent,
            stage,
            stage_name,
            context="derivation private candidate",
            modes={0o700, 0o500},
        )

    try:
        try:
            production = _produce_into(
                plan_path,
                source_root,
                producer_root,
                authority_root,
                stage,
                release_id=release_id,
                expected_plan_sha256=expected_plan_sha256,
                expected_builder_sha256=expected_builder_sha256,
                expected_machine_runner_sha256=expected_machine_runner_sha256,
                expected_upstream_sha256=expected_upstream_sha256,
                expected_authority_sha256=expected_authority_sha256,
                expected_source_sha256=expected_source_sha256,
                expected_producer_sha256=expected_producer_sha256,
                expected_pdf_sha256=expected_pdf_sha256,
                output_guard=output_guard,
            )
        except BaseException as error:
            message = (
                f"{error}; candidate_path={stage.path}; "
                "candidate_state=partial-or-complete-private-candidate-do-not-"
                "auto-delete; inspect identity and inventory before explicit removal"
            )
            raise DerivationClosureError(message) from error
        _publish_directory(
            parent,
            stage,
            stage_name,
            absolute,
            manifest_raw=production.manifest_raw,
        )
        return _receipt(absolute, production, replay_root=None)
    finally:
        _close_resources(
            [
                ("derivation candidate", stage),
                ("derivation destination parent", parent),
            ],
            context="derivation publication",
        )


def _read_anchored_manifest(
    closure_root: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    expected = _expect_sha256(
        expected_manifest_sha256,
        context="expected manifest SHA-256",
    )
    absolute = _machine_call(
        _machine._canonical_existing_directory,
        closure_root,
        context="derivation closure root",
    )
    root = _pin_root(absolute, context="derivation closure root")
    try:
        if stat.S_IMODE(os.fstat(root.descriptor).st_mode) != 0o500:
            _fail("derivation closure root must be sealed mode 0500")
        raw = _read_output_member(root, MANIFEST_MEMBER, maximum=MAX_MANIFEST_BYTES)
        if _sha256(raw) != expected:
            _fail("derivation manifest does not match its independent SHA-256 anchor")
        _validate_tree(root, raw, directory_mode=0o500)
        parsed = json.loads(raw)
        return raw, dict(_expect_mapping(parsed, context="derivation manifest"))
    finally:
        _close_resources(
            [("anchored derivation closure", root)],
            context="anchored manifest read",
        )


def _read_anchored_manifest_v2_pinned(
    closure_root: Path,
    *,
    expected_manifest_sha256: str,
) -> _PinnedV2Closure:
    expected = _expect_sha256(
        expected_manifest_sha256,
        context="expected v2 manifest SHA-256",
    )
    absolute = _machine_call(
        _machine._canonical_existing_directory,
        closure_root,
        context="v2 derivation closure root",
    )
    root = _pin_root(absolute, context="v2 derivation closure root")
    members: dict[str, object] = {}
    try:
        _validate_v2_root_mode(root, context="v2 derivation closure root")
        raw = _read_output_member(root, MANIFEST_MEMBER, maximum=MAX_MANIFEST_BYTES)
        if _sha256(raw) != expected:
            _fail("v2 derivation manifest differs from its independent caller anchor")
        inventory = _validate_v2_tree(root, raw, directory_mode=0o500)
        parsed = dict(
            _expect_mapping(
                json.loads(raw),
                context="anchored v2 derivation manifest",
            ),
        )
        inventory_by_name = {str(record["member"]): record for record in inventory}
        inventory_by_name[MANIFEST_MEMBER] = {
            "member": MANIFEST_MEMBER,
            "bytes": len(raw),
            "sha256": expected,
        }
        for member in sorted(inventory_by_name):
            record = inventory_by_name[member]
            if member == MANIFEST_MEMBER:
                maximum = MAX_MANIFEST_BYTES
            elif member.endswith(".pdf"):
                maximum = MAX_V2_NATIVE_OUTPUT_BYTES
            else:
                maximum = MAX_INVOCATION_RECEIPT_BYTES
            pinned = _open_root_member(
                root,
                member,
                maximum=maximum,
                expected_size=int(record["bytes"]),
                context=f"anchored v2 closure member {member}",
            )
            members[member] = pinned
            _validate_v2_member_mode(
                pinned,
                expected_mode=0o400,
                context=f"anchored v2 closure member {member}",
            )
            if pinned.sha256 != record["sha256"]:
                _fail(f"anchored v2 closure member {member} differs from manifest")
        return _PinnedV2Closure(
            root=root,
            members=members,
            manifest=parsed,
            manifest_raw=raw,
        )
    except BaseException:
        _close_resources(
            [
                *[
                    (f"anchored v2 closure member {member}", pinned)
                    for member, pinned in members.items()
                ],
                ("anchored v2 derivation closure", root),
            ],
            context="failed anchored v2 closure read",
        )
        raise


def _revalidate_pinned_v2_closure(closure: _PinnedV2Closure) -> None:
    _machine_call(
        _machine._revalidate_root,
        closure.root,
        context="anchored v2 derivation closure root",
    )
    _validate_v2_root_mode(
        closure.root,
        context="anchored v2 derivation closure root",
    )
    for member, pinned in closure.members.items():
        _machine_call(
            _machine._revalidate_file,
            pinned,
            context=f"anchored v2 closure member {member}",
        )
        _validate_v2_member_mode(
            pinned,
            expected_mode=0o400,
            context=f"anchored v2 closure member {member}",
        )


def validate_derivation_closure(
    plan_path: Path,
    source_root: Path,
    producer_root: Path,
    authority_root: Path,
    closure_root: Path,
    replay_root: Path,
    *,
    expected_manifest_sha256: str,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> DerivationClosureReceipt:
    """Replay all eight native builds into a separate retained validation tree."""
    closure_absolute = _machine_call(
        _machine._canonical_existing_directory,
        closure_root,
        context="derivation closure root",
    )
    replay_absolute = replay_root.absolute()
    _validate_path_topology(
        plan_path,
        source_root,
        producer_root,
        authority_root,
        replay_absolute,
    )
    _machine_call(
        _machine._assert_distinct_roots,
        (closure_absolute, "derivation closure root"),
        (replay_absolute, "validation replay root"),
    )
    original_raw, original = _read_anchored_manifest(
        closure_absolute,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if original.get("release_id") != release_id:
        _fail("anchored derivation manifest release id differs from replay")
    inputs = _expect_mapping(original["inputs"], context="anchored manifest.inputs")
    if inputs.get("plan_sha256") != expected_plan_sha256:
        _fail("anchored derivation manifest does not bind the replay plan")
    _, replay_parent, replay_pin = _reserve_directory(
        replay_absolute,
        reserved_name=replay_absolute.name,
        context="derivation validation replay",
    )
    closure_pin = None

    def replay_guard() -> None:
        _revalidate_reserved(
            replay_parent,
            replay_pin,
            replay_absolute.name,
            context="derivation validation replay",
            modes={0o700, 0o500},
        )
        if closure_pin is None:
            _fail("anchored derivation closure is not pinned")
        _machine_call(
            _machine._revalidate_root,
            closure_pin,
            context="derivation closure root",
        )

    try:
        closure_pin = _pin_root(closure_absolute, context="derivation closure root")
        try:
            replay = _produce_into(
                plan_path,
                source_root,
                producer_root,
                authority_root,
                replay_pin,
                release_id=release_id,
                expected_plan_sha256=expected_plan_sha256,
                expected_builder_sha256=expected_builder_sha256,
                expected_machine_runner_sha256=expected_machine_runner_sha256,
                expected_upstream_sha256=expected_upstream_sha256,
                expected_authority_sha256=expected_authority_sha256,
                expected_source_sha256=expected_source_sha256,
                expected_producer_sha256=expected_producer_sha256,
                expected_pdf_sha256=expected_pdf_sha256,
                output_guard=replay_guard,
            )
        except BaseException as error:
            message = (
                f"{error}; replay_candidate_path={replay_absolute}; "
                "candidate_state=partial-or-complete-validation-replay-do-not-"
                "auto-delete; inspect identity and inventory before explicit removal"
            )
            raise DerivationClosureError(message) from error
        replay_guard()
        if replay.manifest_raw != original_raw:
            _fail("independent native derivation replay manifest differs from closure")
        original_files, original_dirs, _ = _machine_call(
            _machine._walk_output,
            closure_pin,
            directory_mode=0o500,
        )
        replay_files, replay_dirs, _ = _machine_call(
            _machine._walk_output,
            replay_pin,
            directory_mode=0o500,
        )
        if (original_files, original_dirs) != (replay_files, replay_dirs):
            _fail("independent derivation replay tree shape differs from closure")
        if _member_inventory(closure_pin, original_files) != _member_inventory(
            replay_pin,
            replay_files,
        ):
            _fail("independent derivation replay bytes differ from closure")
        return _receipt(closure_absolute, replay, replay_root=replay_absolute)
    except BaseException as error:
        if isinstance(
            error,
            DerivationClosureError,
        ) and "replay_candidate_path=" in str(error):
            raise
        message = (
            f"{error}; replay_candidate_path={replay_absolute}; "
            "candidate_state=partial-or-complete-validation-replay-do-not-"
            "auto-delete; inspect identity and inventory before explicit removal"
        )
        raise DerivationClosureError(message) from error
    finally:
        _close_resources(
            [
                ("anchored derivation closure", closure_pin),
                ("validation replay", replay_pin),
                ("validation replay parent", replay_parent),
            ],
            context="derivation replay",
        )


def validate_derivation_closure_v2(
    plan_path: Path,
    source_root: Path,
    package_roots: Mapping[str, Path],
    closure_root: Path,
    replay_root: Path,
    *,
    expected_manifest_sha256: str,
    release_id: str,
    expected_plan_sha256: str,
    expected_builder_sha256: str,
    expected_machine_runner_sha256: str,
    expected_upstream_sha256: Mapping[str, str],
    expected_authority_sha256: Mapping[str, str],
    expected_source_sha256: Mapping[str, str],
    expected_producer_sha256: Mapping[str, str],
    expected_package_sha256: Mapping[str, str],
    expected_renderer_manifest_sha256: Mapping[str, str],
    expected_pdf_sha256: Mapping[str, str],
) -> DerivationClosureReceiptV2:
    """Replay v2 privately, compare exact bytes, then publish the retained replay."""
    closure_absolute = _machine_call(
        _machine._canonical_existing_directory,
        closure_root,
        context="v2 derivation closure root",
    )
    replay_absolute = replay_root.absolute()
    source_absolute, package_absolutes = _validate_v2_path_topology(
        plan_path,
        source_root,
        package_roots,
        replay_absolute,
    )
    _machine_call(
        _machine._assert_distinct_roots,
        (closure_absolute, "v2 derivation closure root"),
        (source_absolute, "v2 source-bundle root"),
        *[
            (package_absolutes[pdf_id], f"v2 producer package root {pdf_id}")
            for pdf_id in PDF_IDS
        ],
        (replay_absolute, "v2 validation replay root"),
    )
    original = _read_anchored_manifest_v2_pinned(
        closure_absolute,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    inputs: _V2Inputs | None = None
    replay_parent = replay_pin = None
    stage_name = f".{replay_absolute.name}.private-v2-replay-candidate"
    try:
        if original.manifest.get("release_id") != release_id:
            _fail("anchored v2 derivation manifest release id differs from replay")
        original_inputs = _expect_mapping(
            original.manifest["inputs"],
            context="anchored v2 manifest.inputs",
        )
        if original_inputs.get("plan_sha256") != expected_plan_sha256:
            _fail("anchored v2 derivation manifest does not bind the replay plan")
        _, replay_parent, replay_pin = _reserve_directory(
            replay_absolute,
            reserved_name=stage_name,
            context="v2 derivation validation replay",
        )

        def replay_guard() -> None:
            if replay_parent is None or replay_pin is None:
                _fail("v2 validation replay candidate is not pinned")
            _revalidate_reserved(
                replay_parent,
                replay_pin,
                stage_name,
                context="v2 derivation private validation replay",
                modes={0o700, 0o500},
            )
            _revalidate_pinned_v2_closure(original)

        try:
            inputs = _pin_v2_inputs(
                plan_path,
                source_root,
                package_roots,
                replay_pin.path,
                release_id=release_id,
                expected_plan_sha256=expected_plan_sha256,
                expected_builder_sha256=expected_builder_sha256,
                expected_machine_runner_sha256=expected_machine_runner_sha256,
                expected_upstream_sha256=expected_upstream_sha256,
                expected_authority_sha256=expected_authority_sha256,
                expected_source_sha256=expected_source_sha256,
                expected_producer_sha256=expected_producer_sha256,
                expected_package_sha256=expected_package_sha256,
                expected_renderer_manifest_sha256=(expected_renderer_manifest_sha256),
                expected_pdf_sha256=expected_pdf_sha256,
            )
            replay = _produce_v2_with_inputs(
                inputs,
                replay_pin,
                output_guard=replay_guard,
            )
            replay_guard()
            if replay.manifest_raw != original.manifest_raw:
                _fail("independent v2 derivation replay manifest differs from closure")
            original_files, original_dirs, _ = _machine_call(
                _machine._walk_output,
                original.root,
                directory_mode=0o500,
            )
            replay_files, replay_dirs, _ = _machine_call(
                _machine._walk_output,
                replay_pin,
                directory_mode=0o500,
            )
            if (original_files, original_dirs) != (replay_files, replay_dirs):
                _fail("independent v2 replay tree shape differs from closure")
            if _member_inventory(
                original.root,
                original_files,
            ) != _member_inventory(replay_pin, replay_files):
                _fail("independent v2 replay bytes differ from closure")
            _revalidate_v2_inputs(inputs)
            _revalidate_pinned_v2_closure(original)
            _publish_directory(
                replay_parent,
                replay_pin,
                stage_name,
                replay_absolute,
                manifest_raw=replay.manifest_raw,
                tree_validator=_validate_v2_tree,
            )
            receipt = _receipt_v2(
                closure_absolute,
                replay,
                replay_root=replay_absolute,
            )
            _revalidate_v2_inputs(inputs)
            _revalidate_pinned_v2_closure(original)
            _terminal_revalidate_published_v2(
                replay_parent,
                replay_pin,
                replay_absolute,
                replay,
                receipt,
                receipt_root=closure_absolute,
            )
        except BaseException as error:
            diagnostic = _v2_candidate_diagnostic(
                replay_parent,
                replay_pin,
                stage_name,
                replay_absolute,
            )
            detail = _v2_publication_error_detail(error)
            message = (
                f"{detail}; {diagnostic}; inspect identity and inventory before "
                "explicit removal"
            )
            raise DerivationClosureError(message) from error
        return receipt
    finally:
        _close_resources(
            [
                ("v2 replay inputs", inputs),
                ("anchored v2 derivation closure", original),
                ("v2 validation replay", replay_pin),
                ("v2 validation replay parent", replay_parent),
            ],
            context="v2 derivation replay",
        )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--expected-machine-runner-sha256", required=True)
    for key in UPSTREAM_BINDING_KEYS:
        parser.add_argument(f"--expected-{key.replace('_', '-')}", required=True)
    for pdf_id in PDF_IDS:
        parser.add_argument(f"--{pdf_id}-authority-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-source-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-producer-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-pdf-sha256", required=True)


def _add_v2_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--expected-machine-runner-sha256", required=True)
    for key in UPSTREAM_BINDING_KEYS:
        parser.add_argument(f"--expected-{key.replace('_', '-')}", required=True)
    for pdf_id in PDF_IDS:
        parser.add_argument(f"--{pdf_id}-package-root", type=Path, required=True)
        parser.add_argument(f"--{pdf_id}-package-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-authority-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-source-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-producer-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-renderer-manifest-sha256", required=True)
        parser.add_argument(f"--{pdf_id}-pdf-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build",
        help="build a sealed derivation closure from an authorized plan",
        description="Build a sealed derivation closure from an authorized plan.",
    )
    _add_common_arguments(build)
    build.add_argument("--destination", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate",
        help="replay an anchored derivation closure",
    )
    _add_common_arguments(validate)
    validate.add_argument("--closure-root", type=Path, required=True)
    validate.add_argument("--replay-root", type=Path, required=True)
    validate.add_argument("--expected-manifest-sha256", required=True)
    build_v2 = subparsers.add_parser(
        "build-v2",
        help="build a revision-only closure from four sealed native packages",
    )
    _add_v2_common_arguments(build_v2)
    build_v2.add_argument("--destination", type=Path, required=True)
    validate_v2 = subparsers.add_parser(
        "validate-v2",
        help="privately replay and then publish an anchored v2 closure",
    )
    _add_v2_common_arguments(validate_v2)
    validate_v2.add_argument("--closure-root", type=Path, required=True)
    validate_v2.add_argument("--replay-root", type=Path, required=True)
    validate_v2.add_argument("--expected-manifest-sha256", required=True)
    return parser


def _cli_upstream(arguments: argparse.Namespace) -> dict[str, str]:
    return {key: getattr(arguments, f"expected_{key}") for key in UPSTREAM_BINDING_KEYS}


def _cli_role_map(arguments: argparse.Namespace, suffix: str) -> dict[str, str]:
    return {pdf_id: getattr(arguments, f"{pdf_id}_{suffix}") for pdf_id in PDF_IDS}


def _cli_role_path_map(arguments: argparse.Namespace, suffix: str) -> dict[str, Path]:
    return {pdf_id: getattr(arguments, f"{pdf_id}_{suffix}") for pdf_id in PDF_IDS}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit build or independent validation command."""
    arguments = _parser().parse_args(argv)
    common: dict[str, object] = {
        "release_id": arguments.release_id,
        "expected_plan_sha256": arguments.expected_plan_sha256,
        "expected_builder_sha256": arguments.expected_builder_sha256,
        "expected_machine_runner_sha256": arguments.expected_machine_runner_sha256,
        "expected_upstream_sha256": _cli_upstream(arguments),
        "expected_authority_sha256": _cli_role_map(
            arguments,
            "authority_sha256",
        ),
        "expected_source_sha256": _cli_role_map(arguments, "source_sha256"),
        "expected_producer_sha256": _cli_role_map(arguments, "producer_sha256"),
        "expected_pdf_sha256": _cli_role_map(arguments, "pdf_sha256"),
    }
    if arguments.command in {"build-v2", "validate-v2"}:
        common.update(
            {
                "expected_package_sha256": _cli_role_map(
                    arguments,
                    "package_sha256",
                ),
                "expected_renderer_manifest_sha256": _cli_role_map(
                    arguments,
                    "renderer_manifest_sha256",
                ),
            },
        )
        package_roots = _cli_role_path_map(arguments, "package_root")
        if arguments.command == "build-v2":
            receipt = build_derivation_closure_v2(
                arguments.plan,
                arguments.source_root,
                package_roots,
                arguments.destination,
                **common,  # type: ignore[arg-type]
            )
        else:
            receipt = validate_derivation_closure_v2(
                arguments.plan,
                arguments.source_root,
                package_roots,
                arguments.closure_root,
                arguments.replay_root,
                expected_manifest_sha256=arguments.expected_manifest_sha256,
                **common,  # type: ignore[arg-type]
            )
    elif arguments.command == "build":
        receipt = build_derivation_closure(
            arguments.plan,
            arguments.source_root,
            arguments.producer_root,
            arguments.authority_root,
            arguments.destination,
            **common,  # type: ignore[arg-type]
        )
    else:
        receipt = validate_derivation_closure(
            arguments.plan,
            arguments.source_root,
            arguments.producer_root,
            arguments.authority_root,
            arguments.closure_root,
            arguments.replay_root,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            **common,  # type: ignore[arg-type]
        )
    print(_canonical_json(asdict(receipt)).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

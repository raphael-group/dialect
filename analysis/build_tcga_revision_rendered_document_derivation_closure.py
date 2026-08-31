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
import hashlib
import importlib
import json
import os
import re
import resource
import stat
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
MAX_SOURCE_BUNDLE_BYTES: Final = 256 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES: Final = 1024 * 1024 * 1024
MAX_PRODUCER_BYTES: Final = 32 * 1024 * 1024
MAX_TOTAL_PRODUCER_BYTES: Final = 128 * 1024 * 1024
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


def _expected_arguments(pdf_id: str) -> list[str]:
    return [token.replace("{pdf_id}", pdf_id) for token in ARGUMENT_TEMPLATE]


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
        _validate_tree(stage, manifest_raw, directory_mode=0o500)
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
        _validate_tree(stage, manifest_raw, directory_mode=0o500)
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
    return parser


def _cli_upstream(arguments: argparse.Namespace) -> dict[str, str]:
    return {key: getattr(arguments, f"expected_{key}") for key in UPSTREAM_BINDING_KEYS}


def _cli_role_map(arguments: argparse.Namespace, suffix: str) -> dict[str, str]:
    return {pdf_id: getattr(arguments, f"{pdf_id}_{suffix}") for pdf_id in PDF_IDS}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit build or independent validation command."""
    arguments = _parser().parse_args(argv)
    common = {
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
    if arguments.command == "build":
        receipt = build_derivation_closure(
            arguments.plan,
            arguments.source_root,
            arguments.producer_root,
            arguments.authority_root,
            arguments.destination,
            **common,
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
            **common,
        )
    print(_canonical_json(asdict(receipt)).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Result-blind validation of the DIALECT revision approval record.

The approval record is intentionally separate from every result-bearing input and
output.  It records textual coauthor attestations and their provenance; it does
not claim that the text is a cryptographic signature or authenticate a person's
identity.  Every coauthor has a separate attestation that binds an exact decision
authority digest and canonical artifact through an exact machine-readable line in
a verified first-party evidence record.  The authority digest covers every
operational decision field and exact manifest-wide stage envelope while
deliberately excluding the recorder and the attestations themselves, which
avoids circular evidence.
Transcripts, conversational summaries, recorder inference, and ambient assent are
not accepted as approval evidence.  Callers must pin the manifest SHA-256 through
an independent channel.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

APPROVAL_SCHEMA: Final = "dialect-revision-coauthor-approval-v4"
"""Canonical JSON schema identifier for revision approval manifests."""

EVIDENCE_LINE_SCHEMA: Final = "DIALECT-REVISION-DECISION-V4"
"""ASCII TSV prefix for one coauthor's exact decision-authority attestation."""

STAGE_SCOPED_APPROVAL_SCHEMA: Final = "dialect-revision-coauthor-approval-v5"
"""Singleton-stage schema containing only that stage's minimum decisions."""

STAGE_SCOPED_EVIDENCE_LINE_SCHEMA: Final = "DIALECT-REVISION-DECISION-V5"
"""ASCII TSV prefix for singleton-stage decision attestations."""

APPROVAL_SCHEMAS: Final[tuple[str, str]] = (
    APPROVAL_SCHEMA,
    STAGE_SCOPED_APPROVAL_SCHEMA,
)
"""Supported approval schemas, oldest first."""

MATERIALIZE_FINAL_INPUTS_STAGE: Final = "materialize-final-inputs"
FIT_SEALED_TCGA_K500_STAGE: Final = "fit-sealed-tcga-k500"
INSPECT_TCGA_K500_STAGE: Final = "inspect-tcga-k500"
CALIBRATION_STAGE: Final = "calibration"
COMPARATORS_STAGE: Final = "comparators"
MSK_STAGE: Final = "msk"
RELEASE_STAGE: Final = "release"

REVISION_STAGES: Final[tuple[str, ...]] = (
    MATERIALIZE_FINAL_INPUTS_STAGE,
    FIT_SEALED_TCGA_K500_STAGE,
    INSPECT_TCGA_K500_STAGE,
    CALIBRATION_STAGE,
    COMPARATORS_STAGE,
    MSK_STAGE,
    RELEASE_STAGE,
)
"""All result-bearing stages understood by the approval gate, in canonical order."""

DECISION_IDS: Final[tuple[str, ...]] = tuple(f"D{index}" for index in range(1, 11))
APPROVERS: Final[tuple[str, str]] = ("Benjamin J. Raphael", "Uthsav Chitra")
GO_DISPOSITION: Final = "go"
NO_GO_DISPOSITION: Final = "no-go"
DEFERRED_DISPOSITION: Final = "deferred"
DECISION_DISPOSITIONS: Final[tuple[str, ...]] = (
    GO_DISPOSITION,
    NO_GO_DISPOSITION,
    DEFERRED_DISPOSITION,
)
SOURCE_NOTICE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "coauthor-authored-email",
        "signed-document",
        "coauthor-authored-machine-record",
    },
)
"""Accepted first-party evidence kinds for textual approval attestations."""

STAGE_MINIMUM_DECISIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        MATERIALIZE_FINAL_INPUTS_STAGE: ("D1", "D2"),
        FIT_SEALED_TCGA_K500_STAGE: tuple(f"D{index}" for index in range(1, 7)),
        INSPECT_TCGA_K500_STAGE: tuple(f"D{index}" for index in range(1, 7)),
        CALIBRATION_STAGE: ("D4", "D5", "D6"),
        COMPARATORS_STAGE: ("D4", "D5", "D8"),
        MSK_STAGE: ("D1", "D2", "D3", "D4", "D5", "D9"),
        RELEASE_STAGE: DECISION_IDS,
    },
)
"""Minimum signed decisions required before each result-bearing stage."""

DECISION_ALLOWED_STAGES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        decision_id: tuple(
            stage
            for stage in REVISION_STAGES
            if decision_id in STAGE_MINIMUM_DECISIONS[stage]
        )
        for decision_id in DECISION_IDS
    },
)
"""Exact stage authority that each decision record must declare."""

STAGE_BINDING_KEYS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        MATERIALIZE_FINAL_INPUTS_STAGE: frozenset(
            {
                "d1_canonical_artifact_sha256",
                "d2_canonical_artifact_sha256",
            },
        ),
        FIT_SEALED_TCGA_K500_STAGE: frozenset(
            {
                "canonical_input_manifest_sha256",
                "provider_input_manifest_sha256",
            },
        ),
        COMPARATORS_STAGE: frozenset(
            {
                "canonical_input_manifest_sha256",
                "comparator_launch_scope_manifest_sha256",
                "provider_input_manifest_sha256",
                "upstream_result_manifest_sha256",
            },
        ),
        MSK_STAGE: frozenset(
            {
                "canonical_input_manifest_sha256",
                "msk_phase_scope_manifest_sha256",
                "provider_input_manifest_sha256",
                "upstream_result_manifest_sha256",
            },
        ),
        **{
            stage: frozenset(
                {
                    "canonical_input_manifest_sha256",
                    "provider_input_manifest_sha256",
                    "upstream_result_manifest_sha256",
                },
            )
            for stage in REVISION_STAGES
            if stage
            not in {
                MATERIALIZE_FINAL_INPUTS_STAGE,
                FIT_SEALED_TCGA_K500_STAGE,
                COMPARATORS_STAGE,
                MSK_STAGE,
            }
        },
    },
)
"""Exact independently pinned SHA-256 authority required by each stage."""

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_UTC_SECOND_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_READ_CHUNK_BYTES: Final = 1024 * 1024
_EVIDENCE_FIELD_COUNT: Final = 7
_DECISION_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "allowed_stages",
        "canonical_artifact",
        "claim_owner",
        "decision_id",
        "disposition",
        "exact_resolution",
        "execution_owner",
        "forbidden_claims",
        "manifest_allowed_stages",
        "manifest_stage_bindings",
        "permitted_claims",
        "rerun_or_reuse_consequence",
    },
)
_DECISION_KEYS: Final[frozenset[str]] = frozenset(
    {*_DECISION_AUTHORITY_KEYS, "attestations", "recorded_by"},
)


class RevisionApprovalError(ValueError):
    """Raised when a revision approval manifest fails closed."""


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Verified immutable bytes from a regular, non-symlink artifact."""

    path: str
    sha256: str
    size_bytes: int
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """First-party evidence locator and its verified immutable file bytes."""

    kind: str
    locator: str
    file: ArtifactReceipt


@dataclass(frozen=True, slots=True)
class ApprovalAttestation:
    """One coauthor's textual attestation to an exact decision artifact."""

    approver: str
    attested_disposition: str
    attested_at_utc: str
    canonical_artifact_sha256: str
    decision_authority_sha256: str
    evidence: EvidenceRecord


@dataclass(frozen=True, slots=True)
class DecisionApproval:
    """Immutable approval record for one revision decision."""

    decision_id: str
    disposition: str
    exact_resolution: str
    canonical_artifact: ArtifactReceipt
    execution_owner: str
    claim_owner: str
    rerun_or_reuse_consequence: str
    permitted_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    attestations: tuple[ApprovalAttestation, ...]
    recorded_by: str
    allowed_stages: tuple[str, ...]
    manifest_allowed_stages: tuple[str, ...]
    manifest_stage_bindings: Mapping[str, Mapping[str, str]]
    decision_authority_sha256: str


@dataclass(frozen=True, slots=True)
class SourceNotice:
    """Verified first-party file containing the source approval notice."""

    kind: str
    locator: str
    file: ArtifactReceipt


@dataclass(frozen=True, slots=True)
class RevisionApproval:
    """Fully validated, immutable coauthor approval authority."""

    schema: str
    source_notice: SourceNotice
    allowed_stages: tuple[str, ...]
    stage_bindings: Mapping[str, Mapping[str, str]]
    decisions: Mapping[str, DecisionApproval]
    manifest_sha256: str
    decision_digests: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _AttestationContext:
    """Frozen inputs shared while validating one decision's attestations."""

    decision_label: str
    decision_id: str
    now: datetime
    disposition: str
    canonical_artifact_sha256: str
    decision_authority_sha256: str
    artifact_root: Path
    evidence_line_schema: str
    exact_evidence_decisions: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _DecisionAuthority:
    """Parsed operational fields covered by a coauthor authority digest."""

    decision_id: str
    disposition: str
    exact_resolution: str
    canonical_artifact: ArtifactReceipt
    execution_owner: str
    claim_owner: str
    rerun_or_reuse_consequence: str
    permitted_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    allowed_stages: tuple[str, ...]
    manifest_allowed_stages: tuple[str, ...]
    manifest_stage_bindings: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class _EvidenceMarker:
    """One syntactically valid evidence marker parsed from immutable bytes."""

    approver: str
    decision_id: str
    attested_at_utc: str
    line: str
    order_key: tuple[int, int]


def _evidence_line_schema(approval_schema: str) -> str:
    if approval_schema == APPROVAL_SCHEMA:
        return EVIDENCE_LINE_SCHEMA
    if approval_schema == STAGE_SCOPED_APPROVAL_SCHEMA:
        return STAGE_SCOPED_EVIDENCE_LINE_SCHEMA
    msg = f"Unsupported approval schema: {approval_schema!r}."
    raise RevisionApprovalError(msg)


def _decision_ids_for_schema(
    approval_schema: str,
    allowed_stages: tuple[str, ...],
) -> tuple[str, ...]:
    if approval_schema == APPROVAL_SCHEMA:
        return DECISION_IDS
    if approval_schema != STAGE_SCOPED_APPROVAL_SCHEMA:
        msg = f"Unsupported approval schema: {approval_schema!r}."
        raise RevisionApprovalError(msg)
    if len(allowed_stages) != 1:
        msg = (
            "Stage-scoped approval schema v5 requires exactly one allowed stage; "
            f"observed {list(allowed_stages)}."
        )
        raise RevisionApprovalError(msg)
    return STAGE_MINIMUM_DECISIONS[allowed_stages[0]]


def compute_decision_authority_sha256(
    decision: Mapping[str, object],
    artifact_root: Path,
    *,
    approval_schema: str = APPROVAL_SCHEMA,
) -> str:
    """Compute the non-circular coauthor authority digest for one decision.

    The accepted mapping is either the exact operational authority object or an
    exact full manifest decision object.  ``attestations`` and ``recorded_by``
    are intentionally excluded.  The canonical artifact is securely read so its
    verified byte length, relative path, and SHA-256 are all covered.  The
    repeated ``manifest_allowed_stages`` field binds this decision to one exact
    top-level execution envelope and prevents recorder-controlled stage expansion.

    Args:
        decision: Exact authority fields, optionally plus ``attestations`` and
            ``recorded_by`` from a full manifest decision.
        artifact_root: Directory containing the referenced canonical artifact.
        approval_schema: Schema governing the decision's exact stage scope.

    Returns:
        Lowercase SHA-256 of the canonical operational authority object.

    Raises:
        RevisionApprovalError: If the decision or artifact fails closed.
    """
    authority = _parse_public_decision_authority(
        decision,
        Path(artifact_root),
        approval_schema=approval_schema,
    )
    return _decision_authority_digest(authority)


def canonical_decision_evidence_line(
    decision: Mapping[str, object],
    artifact_root: Path,
    approver: str,
    attested_at_utc: str,
    *,
    approval_schema: str = APPROVAL_SCHEMA,
) -> str:
    """Return one exact ASCII coauthor evidence line without its trailing LF.

    The format is a seven-field TSV record::

        DIALECT-REVISION-DECISION-V4<TAB>approver<TAB>D#<TAB>disposition<TAB>attested-at-utc<TAB>artifact-sha256<TAB>authority-sha256

    Under historical v4, a separate coauthor email or machine record may contain
    prose and multiple such lines, with markers in canonical approver/D1-D10
    order.  Under stage-scoped v5, each coauthor machine record must instead be
    exactly the complete ordered minimum-decision lines plus one trailing LF.

    Args:
        decision: Exact authority fields, optionally as a full decision object.
        artifact_root: Directory containing the canonical decision artifact.
        approver: Exact coauthor name from :data:`APPROVERS`.
        attested_at_utc: Exact whole-second UTC timestamp asserted by the approver.
        approval_schema: Schema selecting the authority and evidence-line contract.

    Returns:
        Canonical ASCII TSV record without a newline.

    Raises:
        RevisionApprovalError: If the decision, artifact, or approver is invalid.
    """
    if approver not in APPROVERS:
        msg = f"Evidence-line approver must be one of {list(APPROVERS)}."
        raise RevisionApprovalError(msg)
    authority = _parse_public_decision_authority(
        decision,
        Path(artifact_root),
        approval_schema=approval_schema,
    )
    timestamp = _parse_utc_second_syntax(
        attested_at_utc,
        "evidence-line attested_at_utc",
    )
    return _canonical_evidence_line(
        authority,
        approver,
        timestamp,
        evidence_line_schema=_evidence_line_schema(approval_schema),
    )


def validate_revision_approval(
    path: Path,
    expected_sha256: str,
    required_stage: str,
    now: datetime | None = None,
) -> RevisionApproval:
    """Validate a pinned, canonical revision approval manifest.

    Artifact paths are POSIX paths relative to the manifest directory.  Neither
    the manifest nor a referenced path may be a symlink.  ``expected_sha256``
    must come from an independent approval handoff; a hash stored inside the
    manifest would not prevent manifest substitution.

    Args:
        path: Approval manifest to validate.
        expected_sha256: Independently pinned lowercase SHA-256 of ``path``.
        required_stage: Result-bearing stage the caller intends to execute.
        now: Optional timezone-aware clock value used for deterministic tests.

    Returns:
        An immutable approval authority with verified decisions and digests.

    Raises:
        RevisionApprovalError: If any schema, provenance, time, path, hash, or
            stage-authority check fails.
    """
    manifest_path = Path(path)
    _require_sha256(expected_sha256, "expected manifest SHA-256")
    if required_stage not in STAGE_MINIMUM_DECISIONS:
        msg = f"Unknown revision stage: {required_stage!r}."
        raise RevisionApprovalError(msg)

    raw_bytes = _read_regular_file(manifest_path, "approval manifest")
    observed_manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if observed_manifest_sha256 != expected_sha256:
        msg = (
            "Approval manifest SHA-256 mismatch: expected "
            f"{expected_sha256}, observed {observed_manifest_sha256}."
        )
        raise RevisionApprovalError(msg)
    manifest_receipt = ArtifactReceipt(
        path=manifest_path.name,
        sha256=observed_manifest_sha256,
        size_bytes=len(raw_bytes),
        content=raw_bytes,
    )

    raw_manifest = _load_canonical_json(raw_bytes)
    manifest = _require_object(raw_manifest, "manifest")
    _require_exact_keys(
        manifest,
        {
            "allowed_stages",
            "decisions",
            "schema",
            "source_notice",
            "stage_bindings",
        },
        "manifest",
    )
    schema = _require_exact_text(manifest["schema"], "manifest.schema")
    if schema not in APPROVAL_SCHEMAS:
        msg = f"Unsupported approval schema: {schema!r}."
        raise RevisionApprovalError(msg)

    allowed_stages = _parse_stage_list(
        manifest["allowed_stages"],
        "manifest.allowed_stages",
        expected=None,
        allow_empty=True,
    )
    expected_decision_ids = _decision_ids_for_schema(schema, allowed_stages)
    stage_bindings = _parse_stage_bindings(
        manifest["stage_bindings"],
        "manifest.stage_bindings",
        allowed_stages,
    )

    artifact_root = manifest_path.parent
    source_notice = _parse_source_notice(manifest["source_notice"], artifact_root)
    effective_now = _normalize_now(now)
    parsed_decisions, decision_digests = _parse_decisions(
        manifest["decisions"],
        artifact_root,
        effective_now,
        manifest_allowed_stages=allowed_stages,
        manifest_stage_bindings=stage_bindings,
        expected_decision_ids=expected_decision_ids,
        approval_schema=schema,
    )
    _require_consistent_artifact_path_receipts(
        manifest_receipt,
        source_notice,
        parsed_decisions,
    )
    if schema == STAGE_SCOPED_APPROVAL_SCHEMA:
        _require_stage_scoped_evidence_closure(
            parsed_decisions,
            expected_decision_ids,
            source_notice,
        )
    materialize_binding = stage_bindings.get(MATERIALIZE_FINAL_INPUTS_STAGE)
    if materialize_binding is not None and (
        materialize_binding["d1_canonical_artifact_sha256"]
        != parsed_decisions["D1"].canonical_artifact.sha256
        or materialize_binding["d2_canonical_artifact_sha256"]
        != parsed_decisions["D2"].canonical_artifact.sha256
    ):
        msg = (
            "manifest.stage_bindings materialize-final-inputs must bind the exact "
            "verified D1 and D2 canonical artifacts."
        )
        raise RevisionApprovalError(msg)

    eligible_stages = tuple(
        stage
        for stage in allowed_stages
        if all(
            parsed_decisions[decision_id].disposition == GO_DISPOSITION
            for decision_id in STAGE_MINIMUM_DECISIONS[stage]
        )
    )
    invalid_authorizations = tuple(
        stage for stage in allowed_stages if stage not in eligible_stages
    )
    if invalid_authorizations:
        msg = (
            "Approval manifest authorizes stages blocked by a no-go or deferred "
            f"minimum decision: {list(invalid_authorizations)}."
        )
        raise RevisionApprovalError(msg)
    if required_stage not in allowed_stages:
        msg = f"Approval manifest does not authorize stage {required_stage!r}."
        raise RevisionApprovalError(msg)

    required_decisions = STAGE_MINIMUM_DECISIONS[required_stage]
    for decision_id in required_decisions:
        decision = parsed_decisions[decision_id]
        if (
            required_stage not in decision.allowed_stages
            or decision.disposition != GO_DISPOSITION
        ):
            msg = (
                f"Decision {decision_id} is {decision.disposition!r} and does not "
                f"authorize required stage {required_stage!r}."
            )
            raise RevisionApprovalError(msg)

    return RevisionApproval(
        schema=schema,
        source_notice=source_notice,
        allowed_stages=allowed_stages,
        stage_bindings=stage_bindings,
        decisions=MappingProxyType(parsed_decisions),
        manifest_sha256=observed_manifest_sha256,
        decision_digests=MappingProxyType(decision_digests),
    )


def _parse_decisions(  # noqa: PLR0913
    value: object,
    artifact_root: Path,
    now: datetime,
    *,
    manifest_allowed_stages: tuple[str, ...],
    manifest_stage_bindings: Mapping[str, Mapping[str, str]],
    expected_decision_ids: tuple[str, ...],
    approval_schema: str,
) -> tuple[dict[str, DecisionApproval], dict[str, str]]:
    decisions_raw = _require_list(value, "manifest.decisions")
    if len(decisions_raw) != len(expected_decision_ids):
        expected_label = (
            "D1-D10"
            if expected_decision_ids == DECISION_IDS
            else list(expected_decision_ids)
        )
        msg = (
            "manifest.decisions must contain exactly "
            f"{expected_label}; observed "
            f"{len(decisions_raw)} records."
        )
        raise RevisionApprovalError(msg)

    parsed_decisions: dict[str, DecisionApproval] = {}
    decision_digests: dict[str, str] = {}
    observed_order: list[str] = []
    for index, decision_raw in enumerate(decisions_raw):
        label = f"manifest.decisions[{index}]"
        decision_object = _require_object(decision_raw, label)
        decision = _parse_decision(
            decision_object,
            label,
            artifact_root,
            now,
            manifest_allowed_stages=manifest_allowed_stages,
            manifest_stage_bindings=manifest_stage_bindings,
            expected_decision_ids=expected_decision_ids,
            approval_schema=approval_schema,
        )
        if decision.decision_id in parsed_decisions:
            msg = f"Duplicate decision record: {decision.decision_id}."
            raise RevisionApprovalError(msg)
        parsed_decisions[decision.decision_id] = decision
        observed_order.append(decision.decision_id)
        decision_digests[decision.decision_id] = hashlib.sha256(
            _canonical_json_bytes(decision_object, trailing_newline=False),
        ).hexdigest()

    if tuple(observed_order) != expected_decision_ids:
        missing = sorted(set(expected_decision_ids).difference(observed_order))
        extra = sorted(set(observed_order).difference(expected_decision_ids))
        expected_label = (
            "D1-D10"
            if expected_decision_ids == DECISION_IDS
            else list(expected_decision_ids)
        )
        msg = (
            f"Decision records must be ordered exactly {expected_label}; "
            f"missing={missing}, extra={extra}, observed={observed_order}."
        )
        raise RevisionApprovalError(msg)

    return parsed_decisions, decision_digests


def _parse_source_notice(value: object, artifact_root: Path) -> SourceNotice:
    source_notice = _require_object(value, "manifest.source_notice")
    _require_exact_keys(
        source_notice,
        {"file", "kind", "locator"},
        "manifest.source_notice",
    )
    kind = _require_exact_text(source_notice["kind"], "manifest.source_notice.kind")
    if kind not in SOURCE_NOTICE_KINDS:
        msg = (
            "manifest.source_notice.kind must be first-party coauthor evidence; "
            f"observed {kind!r}."
        )
        raise RevisionApprovalError(msg)
    return SourceNotice(
        kind=kind,
        locator=_require_exact_text(
            source_notice["locator"],
            "manifest.source_notice.locator",
        ),
        file=_parse_artifact(
            source_notice["file"],
            "manifest.source_notice.file",
            artifact_root,
        ),
    )


def _require_stage_scoped_evidence_closure(
    decisions: Mapping[str, DecisionApproval],
    expected_decision_ids: tuple[str, ...],
    source_notice: SourceNotice,
) -> None:
    """Require one identical complete evidence record per approver across decisions."""
    first_decision = decisions[expected_decision_ids[0]]
    for approver_index, approver in enumerate(APPROVERS):
        baseline = first_decision.attestations[approver_index]
        baseline_evidence = baseline.evidence
        if baseline_evidence.kind != "coauthor-authored-machine-record":
            msg = (
                "Stage-scoped approval evidence must use "
                "coauthor-authored-machine-record bytes."
            )
            raise RevisionApprovalError(msg)
        baseline_identity = (
            baseline_evidence.kind,
            baseline_evidence.locator,
            baseline_evidence.file.path,
            baseline_evidence.file.sha256,
            baseline_evidence.file.content,
        )
        for decision_id in expected_decision_ids[1:]:
            attestation = decisions[decision_id].attestations[approver_index]
            evidence = attestation.evidence
            identity = (
                evidence.kind,
                evidence.locator,
                evidence.file.path,
                evidence.file.sha256,
                evidence.file.content,
            )
            if attestation.approver != approver or identity != baseline_identity:
                msg = (
                    "Stage-scoped approval requires one identical complete evidence "
                    f"record for {approver} across {list(expected_decision_ids)}."
                )
                raise RevisionApprovalError(msg)
            if attestation.attested_at_utc != baseline.attested_at_utc:
                msg = (
                    "Stage-scoped approval requires one identical attestation "
                    f"timestamp for {approver} across {list(expected_decision_ids)}."
                )
                raise RevisionApprovalError(msg)
        expected_lines = [
            _format_evidence_line(
                evidence_line_schema=STAGE_SCOPED_EVIDENCE_LINE_SCHEMA,
                approver=approver,
                decision_id=decision_id,
                disposition=decisions[decision_id].disposition,
                attested_at_utc=decisions[decision_id]
                .attestations[approver_index]
                .attested_at_utc,
                canonical_artifact_sha256=(
                    decisions[decision_id].canonical_artifact.sha256
                ),
                decision_authority_sha256=(
                    decisions[decision_id].decision_authority_sha256
                ),
            )
            for decision_id in expected_decision_ids
        ]
        expected_bytes = ("\n".join(expected_lines) + "\n").encode("ascii")
        if baseline_evidence.file.content != expected_bytes:
            msg = (
                "Stage-scoped approval evidence bytes must equal only the exact "
                f"canonical {list(expected_decision_ids)} TSV lines plus one LF."
            )
            raise RevisionApprovalError(msg)
        if (
            approver_index == 0
            and (
                source_notice.kind,
                source_notice.locator,
                source_notice.file.path,
                source_notice.file.sha256,
                source_notice.file.content,
            )
            != baseline_identity
        ):
            msg = (
                "Stage-scoped source_notice must be the exact first approver "
                "machine-record evidence tuple."
            )
            raise RevisionApprovalError(msg)


def _require_consistent_artifact_path_receipts(
    manifest_receipt: ArtifactReceipt,
    source_notice: SourceNotice,
    decisions: Mapping[str, DecisionApproval],
) -> None:
    """Require every repeated relative path to resolve to one byte receipt."""
    labeled_receipts: list[tuple[str, ArtifactReceipt]] = [
        ("approval manifest", manifest_receipt),
        ("manifest.source_notice.file", source_notice.file),
    ]
    for decision_id, decision in decisions.items():
        decision_label = f"manifest.decisions[{decision_id}]"
        labeled_receipts.append(
            (f"{decision_label}.canonical_artifact", decision.canonical_artifact),
        )
        labeled_receipts.extend(
            (
                f"{decision_label}.attestations[{index}].evidence.file",
                attestation.evidence.file,
            )
            for index, attestation in enumerate(decision.attestations)
        )

    receipts_by_path: dict[str, tuple[str, ArtifactReceipt]] = {}
    for label, receipt in labeled_receipts:
        previous = receipts_by_path.setdefault(receipt.path, (label, receipt))
        previous_label, previous_receipt = previous
        if (
            receipt.sha256,
            receipt.size_bytes,
            receipt.content,
        ) != (
            previous_receipt.sha256,
            previous_receipt.size_bytes,
            previous_receipt.content,
        ):
            msg = (
                f"Approval artifact path {receipt.path!r} resolved to inconsistent "
                f"immutable receipts for {previous_label} and {label}."
            )
            raise RevisionApprovalError(msg)


def _parse_decision(  # noqa: PLR0913
    value: Mapping[str, object],
    label: str,
    artifact_root: Path,
    now: datetime,
    *,
    manifest_allowed_stages: tuple[str, ...],
    manifest_stage_bindings: Mapping[str, Mapping[str, str]],
    expected_decision_ids: tuple[str, ...],
    approval_schema: str,
) -> DecisionApproval:
    _require_exact_keys(value, set(_DECISION_KEYS), label)
    expected_allowed_stages = (
        manifest_allowed_stages
        if approval_schema == STAGE_SCOPED_APPROVAL_SCHEMA
        else None
    )
    authority = _parse_decision_authority(
        value,
        label,
        artifact_root,
        expected_allowed_stages=expected_allowed_stages,
    )
    if authority.decision_id not in expected_decision_ids:
        msg = (
            f"{label}.decision_id is outside the exact stage-scoped decision set "
            f"{list(expected_decision_ids)}."
        )
        raise RevisionApprovalError(msg)
    if authority.manifest_allowed_stages != manifest_allowed_stages:
        msg = (
            f"{label}.manifest_allowed_stages does not bind the exact top-level "
            "manifest.allowed_stages execution envelope."
        )
        raise RevisionApprovalError(msg)
    if dict(authority.manifest_stage_bindings) != {
        stage: dict(binding) for stage, binding in manifest_stage_bindings.items()
    }:
        msg = (
            f"{label}.manifest_stage_bindings does not bind the exact top-level "
            "manifest.stage_bindings authority."
        )
        raise RevisionApprovalError(msg)
    authority_sha256 = _decision_authority_digest(authority)
    attestations = _parse_attestations(
        value["attestations"],
        _AttestationContext(
            decision_label=label,
            decision_id=authority.decision_id,
            now=now,
            disposition=authority.disposition,
            canonical_artifact_sha256=authority.canonical_artifact.sha256,
            decision_authority_sha256=authority_sha256,
            artifact_root=artifact_root,
            evidence_line_schema=_evidence_line_schema(approval_schema),
            exact_evidence_decisions=(
                expected_decision_ids
                if approval_schema == STAGE_SCOPED_APPROVAL_SCHEMA
                else None
            ),
        ),
    )
    return DecisionApproval(
        decision_id=authority.decision_id,
        disposition=authority.disposition,
        exact_resolution=authority.exact_resolution,
        canonical_artifact=authority.canonical_artifact,
        execution_owner=authority.execution_owner,
        claim_owner=authority.claim_owner,
        rerun_or_reuse_consequence=authority.rerun_or_reuse_consequence,
        permitted_claims=authority.permitted_claims,
        forbidden_claims=authority.forbidden_claims,
        attestations=attestations,
        recorded_by=_require_exact_text(value["recorded_by"], f"{label}.recorded_by"),
        allowed_stages=authority.allowed_stages,
        manifest_allowed_stages=authority.manifest_allowed_stages,
        manifest_stage_bindings=authority.manifest_stage_bindings,
        decision_authority_sha256=authority_sha256,
    )


def _parse_public_decision_authority(
    value: Mapping[str, object],
    artifact_root: Path,
    *,
    approval_schema: str,
) -> _DecisionAuthority:
    if not isinstance(value, Mapping):
        msg = "decision must be a mapping."
        raise RevisionApprovalError(msg)
    if any(not isinstance(key, str) for key in value):
        msg = "decision mapping keys must be strings."
        raise RevisionApprovalError(msg)
    decision = dict(value)
    observed_keys = frozenset(decision)
    if observed_keys not in {_DECISION_AUTHORITY_KEYS, _DECISION_KEYS}:
        expected = set(_DECISION_AUTHORITY_KEYS)
        if "attestations" in decision or "recorded_by" in decision:
            expected = set(_DECISION_KEYS)
        _require_exact_keys(decision, expected, "decision")
    _evidence_line_schema(approval_schema)
    expected_allowed_stages: tuple[str, ...] | None = None
    if approval_schema == STAGE_SCOPED_APPROVAL_SCHEMA:
        raw_manifest_stages = _parse_stage_list(
            decision.get("manifest_allowed_stages"),
            "decision.manifest_allowed_stages",
            expected=None,
            allow_empty=False,
        )
        expected_decision_ids = _decision_ids_for_schema(
            approval_schema,
            raw_manifest_stages,
        )
        decision_id = _require_exact_text(
            decision.get("decision_id"),
            "decision.decision_id",
        )
        if decision_id not in expected_decision_ids:
            msg = (
                "decision.decision_id is outside the exact stage-scoped decision "
                f"set {list(expected_decision_ids)}."
            )
            raise RevisionApprovalError(msg)
        expected_allowed_stages = raw_manifest_stages
    return _parse_decision_authority(
        decision,
        "decision",
        artifact_root,
        expected_allowed_stages=expected_allowed_stages,
    )


def _parse_decision_authority(
    value: Mapping[str, object],
    label: str,
    artifact_root: Path,
    *,
    expected_allowed_stages: tuple[str, ...] | None = None,
) -> _DecisionAuthority:
    decision_id = _require_exact_text(value["decision_id"], f"{label}.decision_id")
    if decision_id not in DECISION_ALLOWED_STAGES:
        msg = f"{label}.decision_id must be one of D1-D10; observed {decision_id!r}."
        raise RevisionApprovalError(msg)

    disposition = _require_exact_text(value["disposition"], f"{label}.disposition")
    if disposition not in DECISION_DISPOSITIONS:
        msg = (
            f"{label}.disposition must be one of {list(DECISION_DISPOSITIONS)}; "
            f"observed {disposition!r}."
        )
        raise RevisionApprovalError(msg)

    permitted_claims = _parse_text_list(
        value["permitted_claims"],
        f"{label}.permitted_claims",
    )
    forbidden_claims = _parse_text_list(
        value["forbidden_claims"],
        f"{label}.forbidden_claims",
    )
    overlapping_claims = sorted(set(permitted_claims).intersection(forbidden_claims))
    if overlapping_claims:
        msg = f"{label} permits and forbids the same claims: {overlapping_claims}."
        raise RevisionApprovalError(msg)

    canonical_artifact = _parse_artifact(
        value["canonical_artifact"],
        f"{label}.canonical_artifact",
        artifact_root,
    )
    allowed_stages = _parse_stage_list(
        value["allowed_stages"],
        f"{label}.allowed_stages",
        expected=(
            DECISION_ALLOWED_STAGES[decision_id]
            if expected_allowed_stages is None
            else expected_allowed_stages
        ),
        allow_empty=False,
    )
    manifest_allowed_stages = _parse_stage_list(
        value["manifest_allowed_stages"],
        f"{label}.manifest_allowed_stages",
        expected=None,
        allow_empty=True,
    )
    manifest_stage_bindings = _parse_stage_bindings(
        value["manifest_stage_bindings"],
        f"{label}.manifest_stage_bindings",
        manifest_allowed_stages,
    )
    return _DecisionAuthority(
        decision_id=decision_id,
        disposition=disposition,
        exact_resolution=_require_exact_text(
            value["exact_resolution"],
            f"{label}.exact_resolution",
        ),
        canonical_artifact=canonical_artifact,
        execution_owner=_require_exact_text(
            value["execution_owner"],
            f"{label}.execution_owner",
        ),
        claim_owner=_require_exact_text(value["claim_owner"], f"{label}.claim_owner"),
        rerun_or_reuse_consequence=_require_exact_text(
            value["rerun_or_reuse_consequence"],
            f"{label}.rerun_or_reuse_consequence",
        ),
        permitted_claims=permitted_claims,
        forbidden_claims=forbidden_claims,
        allowed_stages=allowed_stages,
        manifest_allowed_stages=manifest_allowed_stages,
        manifest_stage_bindings=manifest_stage_bindings,
    )


def _decision_authority_object(authority: _DecisionAuthority) -> dict[str, object]:
    return {
        "allowed_stages": list(authority.allowed_stages),
        "canonical_artifact": {
            "path": authority.canonical_artifact.path,
            "sha256": authority.canonical_artifact.sha256,
            "size_bytes": authority.canonical_artifact.size_bytes,
        },
        "claim_owner": authority.claim_owner,
        "decision_id": authority.decision_id,
        "disposition": authority.disposition,
        "exact_resolution": authority.exact_resolution,
        "execution_owner": authority.execution_owner,
        "forbidden_claims": list(authority.forbidden_claims),
        "manifest_allowed_stages": list(authority.manifest_allowed_stages),
        "manifest_stage_bindings": {
            stage: dict(authority.manifest_stage_bindings[stage])
            for stage in authority.manifest_allowed_stages
        },
        "permitted_claims": list(authority.permitted_claims),
        "rerun_or_reuse_consequence": authority.rerun_or_reuse_consequence,
    }


def _decision_authority_digest(authority: _DecisionAuthority) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            _decision_authority_object(authority),
            trailing_newline=False,
        ),
    ).hexdigest()


def _canonical_evidence_line(
    authority: _DecisionAuthority,
    approver: str,
    attested_at_utc: str,
    *,
    evidence_line_schema: str,
) -> str:
    return _format_evidence_line(
        evidence_line_schema=evidence_line_schema,
        approver=approver,
        decision_id=authority.decision_id,
        disposition=authority.disposition,
        attested_at_utc=attested_at_utc,
        canonical_artifact_sha256=authority.canonical_artifact.sha256,
        decision_authority_sha256=_decision_authority_digest(authority),
    )


def _format_evidence_line(  # noqa: PLR0913
    *,
    evidence_line_schema: str,
    approver: str,
    decision_id: str,
    disposition: str,
    attested_at_utc: str,
    canonical_artifact_sha256: str,
    decision_authority_sha256: str,
) -> str:
    line = (
        f"{evidence_line_schema}\t{approver}\t{decision_id}\t{disposition}\t"
        f"{attested_at_utc}\t{canonical_artifact_sha256}\t"
        f"{decision_authority_sha256}"
    )
    try:
        line.encode("ascii")
    except UnicodeEncodeError as error:
        msg = "Canonical decision evidence line must contain ASCII only."
        raise RevisionApprovalError(msg) from error
    return line


def _parse_attestations(
    value: object,
    context: _AttestationContext,
) -> tuple[ApprovalAttestation, ...]:
    raw_attestations = _require_list(
        value,
        f"{context.decision_label}.attestations",
    )
    if len(raw_attestations) != len(APPROVERS):
        msg = (
            f"{context.decision_label}.attestations must contain exactly Benjamin J. "
            f"Raphael and Uthsav Chitra; observed {len(raw_attestations)} records."
        )
        raise RevisionApprovalError(msg)

    parsed: list[ApprovalAttestation] = []
    observed_approvers: list[str] = []
    for index, raw_attestation in enumerate(raw_attestations):
        label = f"{context.decision_label}.attestations[{index}]"
        attestation = _require_object(raw_attestation, label)
        _require_exact_keys(
            attestation,
            {
                "approver",
                "attested_at_utc",
                "attested_disposition",
                "canonical_artifact_sha256",
                "decision_authority_sha256",
                "evidence",
            },
            label,
        )
        approver = _require_exact_text(attestation["approver"], f"{label}.approver")
        if approver not in APPROVERS:
            msg = f"{label}.approver must be one of {list(APPROVERS)}."
            raise RevisionApprovalError(msg)
        attested_disposition = _require_exact_text(
            attestation["attested_disposition"],
            f"{label}.attested_disposition",
        )
        if attested_disposition != context.disposition:
            msg = (
                f"{label}.attested_disposition must equal the decision "
                f"disposition {context.disposition!r}."
            )
            raise RevisionApprovalError(msg)
        attested_at_utc = _parse_utc_second(
            attestation["attested_at_utc"],
            f"{label}.attested_at_utc",
            context.now,
        )
        attested_artifact_sha256 = _require_exact_text(
            attestation["canonical_artifact_sha256"],
            f"{label}.canonical_artifact_sha256",
        )
        _require_sha256(
            attested_artifact_sha256,
            f"{label}.canonical_artifact_sha256",
        )
        if attested_artifact_sha256 != context.canonical_artifact_sha256:
            msg = (
                f"{label}.canonical_artifact_sha256 does not bind the verified "
                "decision artifact."
            )
            raise RevisionApprovalError(msg)
        attested_authority_sha256 = _require_exact_text(
            attestation["decision_authority_sha256"],
            f"{label}.decision_authority_sha256",
        )
        _require_sha256(
            attested_authority_sha256,
            f"{label}.decision_authority_sha256",
        )
        if attested_authority_sha256 != context.decision_authority_sha256:
            msg = (
                f"{label}.decision_authority_sha256 does not bind every "
                "operational field in the verified decision authority."
            )
            raise RevisionApprovalError(msg)
        evidence = _parse_evidence(
            attestation["evidence"],
            f"{label}.evidence",
            context.artifact_root,
        )
        observed_approvers.append(approver)
        parsed.append(
            ApprovalAttestation(
                approver=approver,
                attested_disposition=attested_disposition,
                attested_at_utc=attested_at_utc,
                canonical_artifact_sha256=attested_artifact_sha256,
                decision_authority_sha256=attested_authority_sha256,
                evidence=evidence,
            ),
        )

    if tuple(observed_approvers) != APPROVERS:
        msg = (
            f"{context.decision_label}.attestations must be ordered exactly as "
            f"{list(APPROVERS)}; observed {observed_approvers}."
        )
        raise RevisionApprovalError(msg)
    _require_separate_approval_evidence(tuple(parsed), context.decision_label)
    for index, parsed_attestation in enumerate(parsed):
        _require_canonical_evidence_line(
            parsed_attestation.evidence,
            f"{context.decision_label}.attestations[{index}].evidence",
            approver=parsed_attestation.approver,
            decision_id=context.decision_id,
            disposition=context.disposition,
            attested_at_utc=parsed_attestation.attested_at_utc,
            canonical_artifact_sha256=context.canonical_artifact_sha256,
            decision_authority_sha256=context.decision_authority_sha256,
            evidence_line_schema=context.evidence_line_schema,
            exact_evidence_decisions=context.exact_evidence_decisions,
        )
    return tuple(parsed)


def _parse_evidence(
    value: object,
    label: str,
    artifact_root: Path,
) -> EvidenceRecord:
    evidence = _require_object(value, label)
    _require_exact_keys(evidence, {"file", "kind", "locator"}, label)
    kind = _require_exact_text(evidence["kind"], f"{label}.kind")
    if kind not in SOURCE_NOTICE_KINDS:
        msg = f"{label}.kind must be first-party coauthor evidence; observed {kind!r}."
        raise RevisionApprovalError(msg)
    return EvidenceRecord(
        kind=kind,
        locator=_require_exact_text(evidence["locator"], f"{label}.locator"),
        file=_parse_artifact(evidence["file"], f"{label}.file", artifact_root),
    )


def _require_canonical_evidence_line(  # noqa: PLR0913
    evidence: EvidenceRecord,
    label: str,
    *,
    approver: str,
    decision_id: str,
    disposition: str,
    attested_at_utc: str,
    canonical_artifact_sha256: str,
    decision_authority_sha256: str,
    evidence_line_schema: str,
    exact_evidence_decisions: tuple[str, ...] | None,
) -> None:
    markers = _parse_evidence_markers(
        evidence,
        label,
        evidence_line_schema=evidence_line_schema,
    )
    expected_line = _format_evidence_line(
        evidence_line_schema=evidence_line_schema,
        approver=approver,
        decision_id=decision_id,
        disposition=disposition,
        attested_at_utc=attested_at_utc,
        canonical_artifact_sha256=canonical_artifact_sha256,
        decision_authority_sha256=decision_authority_sha256,
    )
    if evidence.kind != "signed-document" and any(
        marker.approver != approver for marker in markers
    ):
        msg = (
            f"{label}.file coauthor-authored evidence may contain markers only "
            f"for {approver}."
        )
        raise RevisionApprovalError(msg)
    if exact_evidence_decisions is not None:
        observed_decisions = tuple(
            marker.decision_id for marker in markers if marker.approver == approver
        )
        if observed_decisions != exact_evidence_decisions:
            msg = (
                f"{label}.file must contain exactly the stage-scoped evidence "
                f"decisions {list(exact_evidence_decisions)} for {approver}; "
                f"observed {list(observed_decisions)}."
            )
            raise RevisionApprovalError(msg)
    matching_count = sum(marker.line == expected_line for marker in markers)
    if matching_count != 1:
        msg = (
            f"{label}.file must contain exactly one canonical evidence line for "
            f"{approver} {decision_id}; observed {matching_count}."
        )
        raise RevisionApprovalError(msg)


def _parse_evidence_markers(
    evidence: EvidenceRecord,
    label: str,
    *,
    evidence_line_schema: str,
) -> tuple[_EvidenceMarker, ...]:
    try:
        text = evidence.file.content.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"{label}.file must use unambiguous UTF-8 text encoding."
        raise RevisionApprovalError(msg) from error
    if "\r" in text or "\x00" in text:
        msg = f"{label}.file must use LF text without CR or NUL bytes."
        raise RevisionApprovalError(msg)

    markers: list[_EvidenceMarker] = []
    observed_pairs: set[tuple[str, str]] = set()
    for line_number, line in enumerate(text.split("\n"), start=1):
        if "DIALECT-REVISION-DECISION-" not in line:
            continue
        marker = _parse_evidence_marker(
            line,
            label,
            line_number,
            evidence_line_schema=evidence_line_schema,
        )
        pair = (marker.approver, marker.decision_id)
        if pair in observed_pairs:
            msg = (
                f"{label}.file contains duplicate or conflicting evidence markers "
                f"for {marker.approver} {marker.decision_id}."
            )
            raise RevisionApprovalError(msg)
        observed_pairs.add(pair)
        if markers and marker.order_key <= markers[-1].order_key:
            msg = (
                f"{label}.file evidence markers must use canonical approver and "
                "D1-D10 order."
            )
            raise RevisionApprovalError(msg)
        markers.append(marker)
    return tuple(markers)


def _parse_evidence_marker(
    line: str,
    label: str,
    line_number: int,
    *,
    evidence_line_schema: str,
) -> _EvidenceMarker:
    marker_prefix = f"{evidence_line_schema}\t"
    if not line.startswith(marker_prefix):
        if line.startswith("DIALECT-REVISION-DECISION-"):
            msg = (
                f"{label}.file line {line_number} must use evidence schema "
                f"{evidence_line_schema!r}."
            )
        else:
            msg = (
                f"{label}.file line {line_number} embeds the evidence schema instead "
                "of using an exact line-start marker."
            )
        raise RevisionApprovalError(msg)
    try:
        line.encode("ascii")
    except UnicodeEncodeError as error:
        msg = f"{label}.file line {line_number} marker must contain ASCII only."
        raise RevisionApprovalError(msg) from error
    fields = line.split("\t")
    if len(fields) != _EVIDENCE_FIELD_COUNT or fields[0] != evidence_line_schema:
        msg = (
            f"{label}.file line {line_number} must contain exactly seven canonical "
            "TSV fields."
        )
        raise RevisionApprovalError(msg)
    (
        line_approver,
        line_decision,
        line_disposition,
        attested_at_utc,
        artifact_sha,
        authority_sha,
    ) = fields[1:]
    if line_approver not in APPROVERS:
        msg = f"{label}.file line {line_number} has an unknown approver."
        raise RevisionApprovalError(msg)
    if line_decision not in DECISION_IDS:
        msg = f"{label}.file line {line_number} has an unknown decision ID."
        raise RevisionApprovalError(msg)
    if line_disposition not in DECISION_DISPOSITIONS:
        msg = f"{label}.file line {line_number} has an unknown disposition."
        raise RevisionApprovalError(msg)
    attested_at_utc = _parse_utc_second_syntax(
        attested_at_utc,
        f"{label}.file line {line_number} attested_at_utc",
    )
    _require_sha256(
        artifact_sha,
        f"{label}.file line {line_number} artifact SHA-256",
    )
    _require_sha256(
        authority_sha,
        f"{label}.file line {line_number} authority SHA-256",
    )
    return _EvidenceMarker(
        approver=line_approver,
        decision_id=line_decision,
        attested_at_utc=attested_at_utc,
        line=line,
        order_key=(
            APPROVERS.index(line_approver),
            DECISION_IDS.index(line_decision),
        ),
    )


def _require_separate_approval_evidence(
    attestations: tuple[ApprovalAttestation, ...],
    decision_label: str,
) -> None:
    first, second = attestations
    first_locator = (first.evidence.kind, first.evidence.locator)
    second_locator = (second.evidence.kind, second.evidence.locator)
    if first_locator == second_locator:
        msg = f"{decision_label} coauthor approvals require distinct evidence locators."
        raise RevisionApprovalError(msg)
    shared_path = first.evidence.file.path == second.evidence.file.path
    if shared_path and (
        first.evidence.kind != "signed-document"
        or second.evidence.kind != "signed-document"
    ):
        msg = (
            f"{decision_label} coauthor-authored email or machine-record approvals "
            "must use separate evidence files with distinct relative paths."
        )
        raise RevisionApprovalError(msg)
    shared_file = first.evidence.file.sha256 == second.evidence.file.sha256
    if shared_file and (
        first.evidence.kind != "signed-document"
        or second.evidence.kind != "signed-document"
    ):
        msg = (
            f"{decision_label} coauthor-authored email or machine-record approvals "
            "must use separate evidence files."
        )
        raise RevisionApprovalError(msg)


def _parse_utc_second(value: object, label: str, now: datetime) -> str:
    timestamp = _parse_utc_second_syntax(value, label)
    parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc,
    )
    if parsed > now:
        msg = f"{label} is in the future: {timestamp}."
        raise RevisionApprovalError(msg)
    return timestamp


def _parse_utc_second_syntax(value: object, label: str) -> str:
    timestamp = _require_exact_text(value, label)
    if _UTC_SECOND_PATTERN.fullmatch(timestamp) is None:
        msg = f"{label} must be whole-second UTC in YYYY-MM-DDTHH:MM:SSZ form."
        raise RevisionApprovalError(msg)
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
    except ValueError as error:
        msg = f"{label} is not a valid UTC timestamp: {timestamp!r}."
        raise RevisionApprovalError(msg) from error
    return timestamp


def _normalize_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        msg = "now must be timezone-aware."
        raise RevisionApprovalError(msg)
    return now.astimezone(timezone.utc)


def _parse_artifact(value: object, label: str, artifact_root: Path) -> ArtifactReceipt:
    artifact = _require_object(value, label)
    _require_exact_keys(artifact, {"path", "sha256"}, label)
    relative_path = _parse_relative_path(artifact["path"], f"{label}.path")
    expected_sha256 = _require_exact_text(artifact["sha256"], f"{label}.sha256")
    _require_sha256(expected_sha256, f"{label}.sha256")
    artifact_path = artifact_root.joinpath(*PurePosixPath(relative_path).parts)
    raw_bytes = _read_regular_file(artifact_path, label, root=artifact_root)
    if not raw_bytes:
        msg = f"{label} must contain nonempty evidence or contract bytes."
        raise RevisionApprovalError(msg)
    observed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        msg = (
            f"{label} SHA-256 mismatch for {relative_path!r}: expected "
            f"{expected_sha256}, observed {observed_sha256}."
        )
        raise RevisionApprovalError(msg)
    return ArtifactReceipt(
        path=relative_path,
        sha256=expected_sha256,
        size_bytes=len(raw_bytes),
        content=raw_bytes,
    )


def _parse_relative_path(value: object, label: str) -> str:
    text = _require_exact_text(value, label)
    if "\\" in text or any(not character.isprintable() for character in text):
        msg = f"{label} must be a normalized relative POSIX path."
        raise RevisionApprovalError(msg)
    pure_path = PurePosixPath(text)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or text != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        msg = f"{label} must be a normalized relative POSIX path without traversal."
        raise RevisionApprovalError(msg)
    return text


def _read_regular_file(path: Path, label: str, *, root: Path | None = None) -> bytes:
    absolute_path = Path(os.path.abspath(path))  # noqa: PTH100
    if root is not None:
        absolute_root = Path(os.path.abspath(root))  # noqa: PTH100
        try:
            absolute_path.relative_to(absolute_root)
        except ValueError as error:
            msg = f"{label} escapes the artifact root: {path}."
            raise RevisionApprovalError(msg) from error
    return _read_no_follow(absolute_path, label)


def _read_no_follow(path: Path, label: str) -> bytes:
    file_fd = _open_no_follow(path, label)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            msg = f"{label} must be a single-link regular file: {path}."
            raise RevisionApprovalError(msg)
        content = _read_descriptor(file_fd, path, label)
        after = os.fstat(file_fd)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mode != before.st_mode
            or after.st_nlink != 1
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or len(content) != before.st_size
        ):
            msg = f"{label} changed while its secure descriptor was read: {path}."
            raise RevisionApprovalError(msg)
        return content
    finally:
        os.close(file_fd)


def _open_no_follow(path: Path, label: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not no_follow or not directory or os.open not in os.supports_dir_fd:
        msg = "Secure descriptor-relative no-follow file reads are unavailable."
        raise RevisionApprovalError(msg)
    parts = path.parts
    if not path.anchor or not path.name:
        msg = f"{label} must identify a regular file: {path}."
        raise RevisionApprovalError(msg)

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | directory | no_follow | close_on_exec
    file_flags = os.O_RDONLY | no_follow | close_on_exec
    directory_fd = -1
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for component in parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(parts[-1], file_flags, dir_fd=directory_fd)
    except FileNotFoundError as error:
        msg = f"{label} does not exist: {path}."
        raise RevisionApprovalError(msg) from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            msg = f"{label} path must not contain symlinks: {path}."
        else:
            msg = f"Unable to securely read {label}: {path}."
        raise RevisionApprovalError(msg) from error
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _read_descriptor(file_fd: int, path: Path, label: str) -> bytes:
    chunks: list[bytes] = []
    try:
        while chunk := os.read(file_fd, _READ_CHUNK_BYTES):
            chunks.append(chunk)
    except OSError as error:
        msg = f"Unable to securely read {label}: {path}."
        raise RevisionApprovalError(msg) from error
    return b"".join(chunks)


def _load_canonical_json(raw_bytes: bytes) -> object:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "Approval manifest must be valid UTF-8."
        raise RevisionApprovalError(msg) from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except json.JSONDecodeError as error:
        msg = f"Approval manifest is not valid JSON: {error.msg}."
        raise RevisionApprovalError(msg) from error
    try:
        canonical_bytes = _canonical_json_bytes(parsed, trailing_newline=True)
    except UnicodeEncodeError as error:
        msg = "Approval manifest contains invalid Unicode scalar values."
        raise RevisionApprovalError(msg) from error
    if raw_bytes != canonical_bytes:
        msg = (
            "Approval manifest is not canonical JSON; require UTF-8, sorted object "
            "keys, compact separators, and exactly one trailing LF."
        )
        raise RevisionApprovalError(msg)
    return parsed


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON key: {key!r}."
            raise RevisionApprovalError(msg)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    msg = f"Non-finite JSON constant is forbidden: {value}."
    raise RevisionApprovalError(msg)


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = f"Non-finite JSON number is forbidden: {value}."
        raise RevisionApprovalError(msg)
    return parsed


def _canonical_json_bytes(value: object, *, trailing_newline: bool) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if trailing_newline:
        text += "\n"
    return text.encode()


def _require_object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        msg = f"{label} must be a JSON object."
        raise RevisionApprovalError(msg)
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"{label} must be a JSON array."
        raise RevisionApprovalError(msg)
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    observed = set(value)
    if observed == expected:
        return
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    msg = f"{label} has invalid keys; missing={missing}, unknown={unknown}."
    raise RevisionApprovalError(msg)


def _require_exact_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        msg = f"{label} must be a nonblank exact string without outer whitespace."
        raise RevisionApprovalError(msg)
    if any(character in value for character in ("\x00", "\r")):
        msg = f"{label} contains a forbidden control character."
        raise RevisionApprovalError(msg)
    return value


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be exactly 64 lowercase hexadecimal characters."
        raise RevisionApprovalError(msg)


def _parse_text_list(value: object, label: str) -> tuple[str, ...]:
    raw_items = _require_list(value, label)
    if not raw_items:
        msg = f"{label} must contain at least one claim."
        raise RevisionApprovalError(msg)
    items = tuple(
        _require_exact_text(item, f"{label}[{index}]")
        for index, item in enumerate(raw_items)
    )
    if len(set(items)) != len(items):
        msg = f"{label} contains duplicate claims."
        raise RevisionApprovalError(msg)
    return items


def _parse_stage_list(
    value: object,
    label: str,
    *,
    expected: tuple[str, ...] | None,
    allow_empty: bool,
) -> tuple[str, ...]:
    raw_stages = _require_list(value, label)
    stages = tuple(
        _require_exact_text(stage, f"{label}[{index}]")
        for index, stage in enumerate(raw_stages)
    )
    if not stages and not allow_empty:
        msg = f"{label} must explicitly contain at least one stage."
        raise RevisionApprovalError(msg)
    if len(set(stages)) != len(stages):
        msg = f"{label} contains duplicate stages."
        raise RevisionApprovalError(msg)
    unknown = sorted(set(stages).difference(REVISION_STAGES))
    if unknown:
        msg = f"{label} contains unknown stages: {unknown}."
        raise RevisionApprovalError(msg)
    canonical_order = tuple(stage for stage in REVISION_STAGES if stage in stages)
    if stages != canonical_order:
        msg = f"{label} must use canonical stage order: {list(canonical_order)}."
        raise RevisionApprovalError(msg)
    if expected is not None and stages != expected:
        missing = sorted(set(expected).difference(stages))
        extra = sorted(set(stages).difference(expected))
        msg = (
            f"{label} does not match the frozen decision matrix; "
            f"missing={missing}, extra={extra}."
        )
        raise RevisionApprovalError(msg)
    return stages


def _parse_stage_bindings(
    value: object,
    label: str,
    allowed_stages: tuple[str, ...],
) -> Mapping[str, Mapping[str, str]]:
    bindings = _require_object(value, label)
    _require_exact_keys(bindings, set(allowed_stages), label)
    parsed: dict[str, Mapping[str, str]] = {}
    for stage in allowed_stages:
        stage_label = f"{label}.{stage}"
        raw_binding = _require_object(bindings[stage], stage_label)
        _require_exact_keys(raw_binding, set(STAGE_BINDING_KEYS[stage]), stage_label)
        binding: dict[str, str] = {}
        for key in sorted(STAGE_BINDING_KEYS[stage]):
            digest = _require_exact_text(raw_binding[key], f"{stage_label}.{key}")
            _require_sha256(digest, f"{stage_label}.{key}")
            binding[key] = digest
        parsed[stage] = MappingProxyType(binding)
    return MappingProxyType(parsed)

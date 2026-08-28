"""Validate signed D3--D6 revision fit-policy artifacts without reading results.

The approval validator authenticates the immutable bytes attached to each decision;
this module gives those bytes a narrow machine interpretation.  It intentionally
consumes :class:`~dialect.data.revision_approval.RevisionApproval` receipts rather
than reopening artifact paths, and it accepts only the prospective statistical
policies enumerated below.  No result-bearing input is accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

from dialect.data.revision_approval import (
    APPROVAL_SCHEMA,
    FIT_SEALED_TCGA_K500_STAGE,
    GO_DISPOSITION,
    RevisionApproval,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MACHINE_DECISION_SCHEMA: Final = "dialect-revision-machine-decision-v1"
"""Canonical schema shared with the signed D1/D2 decision envelopes."""

D3_CONTRACT: Final = "bmr-provider-hierarchy-v1"
D4_CONTRACT: Final = "profile-lrt-pvalue-policy-v1"
D5_CONTRACT: Final = "conjunction-multiplicity-policy-v1"
D6_CONTRACT: Final = "calibration-scope-policy-v1"

FIT_POLICY_DECISION_IDS: Final[tuple[str, ...]] = ("D3", "D4", "D5", "D6")
FIT_POLICY_CONTRACTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "D3": D3_CONTRACT,
        "D4": D4_CONTRACT,
        "D5": D5_CONTRACT,
        "D6": D6_CONTRACT,
    },
)

BMR_PROVIDERS: Final[frozenset[str]] = frozenset({"cbase", "dig", "mutsig"})
CONJUNCTION_SECONDARY: Final = "secondary"
CONJUNCTION_OMITTED: Final = "omitted"

LRT_TEST_DIRECTION: Final = "nondirectional-two-sided-dependence"
LRT_REFERENCE_FAMILY: Final = "chi-square"
LRT_REFERENCE_DF: Final = 1
LRT_REFERENCE_TAIL: Final = "upper-survival"
LRT_STATISTIC_TRANSFORM: Final = (
    "max(0,2*(alternative_log_likelihood-null_log_likelihood))"
)
LRT_BOUNDARY_HANDLING: Final[frozenset[str]] = frozenset(
    {"assign-p-one-with-explicit-boundary-status", "fail-cohort"},
)
LRT_FAILURE_SEMANTICS: Final[frozenset[str]] = frozenset(
    {"assign-p-one-with-explicit-failure-status", "fail-cohort"},
)
LRT_VALIDITY_STANDARD: Final = (
    "finite-sample-super-uniformity-under-frozen-analysis-pipeline"
)
LRT_VALIDITY_COVERAGE: Final[tuple[str, ...]] = (
    "bmr-estimation",
    "top-k-selection",
    "nuisance-fitting",
    "boundary-behavior",
    "complete-within-cohort-family",
)
LRT_VALIDITY_GATE: Final = (
    "block-inferential-use-if-absent-invalid-or-inconclusive"
)

MAX_P_IUT: Final = "nondirectional-max-p-iut"
NO_CONJUNCTION: Final = "no-conjunction"
SET_CONJUNCTION_P_ONE: Final = "set-conjunction-p-to-one"
NO_CONJUNCTION_COMPONENT_POLICY: Final = "not-applicable-no-conjunction"
WITHIN_COHORT_FAMILY: Final = (
    "one-complete-within-cohort-tested-pair-family"
)
FAILED_HYPOTHESIS_POLICY: Final = "retain-with-p-one"
INCLUSIVE_THRESHOLD: Final = "inclusive-less-than-or-equal"
PRIMARY_REPORTING_LAYER: Final = "confirmatory-conditional-on-valid-marginals"
SENSITIVITY_REPORTING_LAYER: Final = "nominal-sensitivity"
DESCRIPTIVE_REPORTING_LAYER: Final = "descriptive"
RHO_ROLE: Final = "descriptive-post-rejection-annotation"

FULL_EXTERNAL: Final = "full-external"
NARROW_LOCAL: Final = "narrow-local"
NO_EXTENSION: Final = "no-extension"
FULL_CLAIM_SCOPE: Final = "finite-scenario-calibration-evidence-only"
NARROW_CLAIM_SCOPE: Final = "narrow-exact-family-stress-evidence-only"
NO_EXTENSION_CLAIM_SCOPE: Final = "no-calibration-claims"

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_SENSITIVITY_PROVIDER_COUNT: Final = 2
_FULL_REPLICATES_PER_CELL: Final = 1000
_NARROW_CELL_COUNT: Final = 3
_NARROW_REPLICATES_PER_CELL: Final = 300
_SURROGATE_MIN: Final = 0xD800
_SURROGATE_MAX: Final = 0xDFFF
_ExactValue = TypeVar("_ExactValue")


class RevisionFitPolicyError(ValueError):
    """Raised when a signed D3--D6 machine policy fails closed."""


@dataclass(frozen=True, slots=True)
class FitPolicyReceipt:
    """Immutable binding between one approval decision and parsed payload."""

    decision_id: str
    contract: str
    decision_digest: str
    canonical_artifact_path: str
    canonical_artifact_sha256: str
    canonical_artifact_size_bytes: int
    payload_sha256: str
    payload: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ProviderHierarchyPolicy:
    """D3 primary/sensitivity provider hierarchy."""

    primary_provider: str
    sensitivity_providers: tuple[str, str]
    all_three_conjunction_role: str
    burden_dependent_switching: bool
    rationale: str


@dataclass(frozen=True, slots=True)
class LRTReference:
    """Reference distribution used to turn the profile LRT into a p-value."""

    family: str
    degrees_of_freedom: int
    tail: str


@dataclass(frozen=True, slots=True)
class LRTValidityEvidence:
    """Prospective evidence gate required before inferential p/q use."""

    standard: str
    covers: tuple[str, ...]
    gate: str


@dataclass(frozen=True, slots=True)
class LRTPolicy:
    """D4 profile-LRT p-value and failure contract."""

    lrt_contract: str
    test_direction: str
    reference: LRTReference
    statistic_transform: str
    boundary_handling: str
    failure_semantics: str
    validity_evidence: LRTValidityEvidence


@dataclass(frozen=True, slots=True)
class ConjunctionPolicy:
    """D5 nondirectional conjunction construction or explicit omission."""

    mode: str
    invalid_component: str
    missing_component: str
    sign_discordance: str


@dataclass(frozen=True, slots=True)
class MultiplicityPolicy:
    """D5 complete-family multiplicity and reporting layers."""

    primary_method: str
    sensitivity_method: str
    primary_q_threshold: float
    sensitivity_q_threshold: float
    descriptive_q_threshold: float
    threshold_comparison: str
    primary_reporting_layer: str
    sensitivity_reporting_layer: str
    descriptive_reporting_layer: str


@dataclass(frozen=True, slots=True)
class ConjunctionMultiplicityPolicy:
    """D5 conjunction, family, multiplicity, and descriptive direction policy."""

    conjunction: ConjunctionPolicy
    family: str
    failed_hypothesis: str
    multiplicity: MultiplicityPolicy
    rho_role: str
    directional_fdr_control: bool


@dataclass(frozen=True, slots=True)
class CalibrationDesignAuthority:
    """Digest-bound declared calibration design."""

    design_id: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CalibrationComputeAuthority:
    """Explicit authority for the selected compute environment."""

    mode: str
    authority_id: str


@dataclass(frozen=True, slots=True)
class CalibrationScopePolicy:
    """D6 calibration extent and its result-blind claim boundary."""

    path: str
    cell_count: int
    replicates_per_cell: int
    design_authority: CalibrationDesignAuthority | None
    compute_authority: CalibrationComputeAuthority | None
    claim_scope: str
    result_contingent_changes: bool
    pre_calibration_result_inspection: bool


@dataclass(frozen=True, slots=True)
class RevisionFitPolicy:
    """Fully parsed immutable D3--D6 authority for a sealed raw fit."""

    d3: ProviderHierarchyPolicy
    d4: LRTPolicy
    d5: ConjunctionMultiplicityPolicy
    d6: CalibrationScopePolicy
    receipts: Mapping[str, FitPolicyReceipt]


def validate_revision_fit_policy(
    approval: RevisionApproval,
    *,
    expected_lrt_contract: str,
) -> RevisionFitPolicy:
    """Parse the exact signed D3--D6 policies from validated approval bytes.

    This is a fit-only policy gate.  It requires explicit v4 fit-stage authority and
    ``go`` dispositions for D3--D6; it never upgrades another disposition or stage.
    The expected implementation LRT contract is supplied by the caller so an artifact
    cannot authorize a different implementation by naming it itself.

    Args:
        approval: Live immutable v4 approval returned by its validator.
        expected_lrt_contract: Exact LRT implementation contract required by the
            caller that will execute the fit.

    Returns:
        Frozen typed policies and byte/hash receipts suitable for a runner manifest.

    Raises:
        RevisionFitPolicyError: If authority, canonical bytes, schema, or scientific
            cross-field constraints do not match the frozen contract.
    """
    if not isinstance(approval, RevisionApproval):
        msg = "approval must be a validated RevisionApproval authority."
        raise RevisionFitPolicyError(msg)
    if approval.schema != APPROVAL_SCHEMA:
        msg = f"Fit policy requires approval schema {APPROVAL_SCHEMA!r}."
        raise RevisionFitPolicyError(msg)
    if FIT_SEALED_TCGA_K500_STAGE not in approval.allowed_stages:
        msg = "Approval does not explicitly authorize the sealed TCGA K=500 fit stage."
        raise RevisionFitPolicyError(msg)
    expected_lrt_contract = _require_exact_text(
        expected_lrt_contract,
        "expected_lrt_contract",
    )

    envelopes: dict[str, Mapping[str, object]] = {}
    receipts: dict[str, FitPolicyReceipt] = {}
    for decision_id in FIT_POLICY_DECISION_IDS:
        envelope, receipt = _parse_signed_decision(approval, decision_id)
        envelopes[decision_id] = envelope
        receipts[decision_id] = receipt

    d3 = _parse_d3(envelopes["D3"]["payload"])
    d4 = _parse_d4(
        envelopes["D4"]["payload"],
        expected_lrt_contract=expected_lrt_contract,
    )
    d5 = _parse_d5(envelopes["D5"]["payload"])
    d6 = _parse_d6(envelopes["D6"]["payload"])
    _require_conjunction_consistency(d3, d5)
    return RevisionFitPolicy(
        d3=d3,
        d4=d4,
        d5=d5,
        d6=d6,
        receipts=MappingProxyType(receipts),
    )


def _parse_signed_decision(
    approval: RevisionApproval,
    decision_id: str,
) -> tuple[Mapping[str, object], FitPolicyReceipt]:
    try:
        decision = approval.decisions[decision_id]
        decision_digest = approval.decision_digests[decision_id]
    except KeyError as error:
        msg = f"Approval is missing explicit {decision_id} authority."
        raise RevisionFitPolicyError(msg) from error
    if decision.decision_id != decision_id:
        msg = f"Approval decision slot {decision_id} contains {decision.decision_id!r}."
        raise RevisionFitPolicyError(msg)
    if decision.disposition != GO_DISPOSITION:
        msg = f"Decision {decision_id} is not an explicit go disposition."
        raise RevisionFitPolicyError(msg)
    if FIT_SEALED_TCGA_K500_STAGE not in decision.allowed_stages:
        msg = f"Decision {decision_id} does not authorize the sealed fit stage."
        raise RevisionFitPolicyError(msg)
    _require_sha256(decision_digest, f"approval.decision_digests[{decision_id!r}]")

    artifact = decision.canonical_artifact
    content = artifact.content
    if not isinstance(content, bytes):
        msg = f"Signed {decision_id} artifact content must be immutable bytes."
        raise RevisionFitPolicyError(msg)
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if len(content) != artifact.size_bytes or observed_sha256 != artifact.sha256:
        msg = f"Signed {decision_id} artifact differs from its approval receipt."
        raise RevisionFitPolicyError(msg)
    _require_sha256(artifact.sha256, f"{decision_id} canonical artifact SHA-256")

    envelope = _load_canonical_envelope(content, decision_id)
    expected_contract = FIT_POLICY_CONTRACTS[decision_id]
    if envelope["schema"] != MACHINE_DECISION_SCHEMA:
        msg = f"Signed {decision_id} artifact has an unsupported machine schema."
        raise RevisionFitPolicyError(msg)
    if envelope["decision_id"] != decision_id:
        msg = f"Signed {decision_id} artifact has the wrong decision_id."
        raise RevisionFitPolicyError(msg)
    if envelope["contract"] != expected_contract:
        msg = (
            f"Signed {decision_id} artifact contract must be "
            f"{expected_contract!r}."
        )
        raise RevisionFitPolicyError(msg)
    payload = _require_object(envelope["payload"], f"{decision_id}.payload")
    payload_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload, trailing_newline=False),
    ).hexdigest()
    receipt = FitPolicyReceipt(
        decision_id=decision_id,
        contract=expected_contract,
        decision_digest=decision_digest,
        canonical_artifact_path=artifact.path,
        canonical_artifact_sha256=artifact.sha256,
        canonical_artifact_size_bytes=artifact.size_bytes,
        payload_sha256=payload_sha256,
        payload=_freeze_object(payload),
    )
    return envelope, receipt


def _parse_d3(value: object) -> ProviderHierarchyPolicy:
    payload = _require_object(value, "D3.payload")
    _require_exact_keys(
        payload,
        {
            "all_three_conjunction_role",
            "burden_dependent_switching",
            "primary_provider",
            "rationale",
            "sensitivity_providers",
        },
        "D3.payload",
    )
    primary = _require_enum(
        payload["primary_provider"],
        BMR_PROVIDERS,
        "D3.payload.primary_provider",
    )
    sensitivities_raw = _require_list(
        payload["sensitivity_providers"],
        "D3.payload.sensitivity_providers",
    )
    sensitivities = tuple(
        _require_enum(
            item,
            BMR_PROVIDERS,
            f"D3.payload.sensitivity_providers[{index}]",
        )
        for index, item in enumerate(sensitivities_raw)
    )
    expected_sensitivities = BMR_PROVIDERS.difference({primary})
    if (
        len(sensitivities) != _SENSITIVITY_PROVIDER_COUNT
        or set(sensitivities) != expected_sensitivities
    ):
        msg = (
            "D3 sensitivity_providers must contain each non-primary provider "
            "exactly once in the signed order."
        )
        raise RevisionFitPolicyError(msg)
    switching = _require_bool(
        payload["burden_dependent_switching"],
        "D3.payload.burden_dependent_switching",
    )
    if switching:
        msg = "D3 forbids burden-dependent provider switching."
        raise RevisionFitPolicyError(msg)
    return ProviderHierarchyPolicy(
        primary_provider=primary,
        sensitivity_providers=(sensitivities[0], sensitivities[1]),
        all_three_conjunction_role=_require_enum(
            payload["all_three_conjunction_role"],
            {CONJUNCTION_SECONDARY, CONJUNCTION_OMITTED},
            "D3.payload.all_three_conjunction_role",
        ),
        burden_dependent_switching=False,
        rationale=_require_exact_text(payload["rationale"], "D3.payload.rationale"),
    )


def _parse_d4(value: object, *, expected_lrt_contract: str) -> LRTPolicy:
    payload = _require_object(value, "D4.payload")
    _require_exact_keys(
        payload,
        {
            "boundary_handling",
            "failure_semantics",
            "lrt_contract",
            "reference",
            "statistic_transform",
            "test_direction",
            "validity_evidence",
        },
        "D4.payload",
    )
    lrt_contract = _require_exact_text(
        payload["lrt_contract"],
        "D4.payload.lrt_contract",
    )
    if lrt_contract != expected_lrt_contract:
        msg = (
            "D4 lrt_contract does not bind the implementation contract required "
            "by the caller."
        )
        raise RevisionFitPolicyError(msg)

    reference_raw = _require_object(payload["reference"], "D4.payload.reference")
    _require_exact_keys(
        reference_raw,
        {"degrees_of_freedom", "family", "tail"},
        "D4.payload.reference",
    )
    _require_exact_value(
        reference_raw["family"],
        LRT_REFERENCE_FAMILY,
        "D4.payload.reference.family",
    )
    _require_exact_integer(
        reference_raw["degrees_of_freedom"],
        LRT_REFERENCE_DF,
        "D4.payload.reference.degrees_of_freedom",
    )
    _require_exact_value(
        reference_raw["tail"],
        LRT_REFERENCE_TAIL,
        "D4.payload.reference.tail",
    )

    evidence_raw = _require_object(
        payload["validity_evidence"],
        "D4.payload.validity_evidence",
    )
    _require_exact_keys(
        evidence_raw,
        {"covers", "gate", "standard"},
        "D4.payload.validity_evidence",
    )
    coverage = tuple(
        _require_exact_text(item, f"D4.payload.validity_evidence.covers[{index}]")
        for index, item in enumerate(
            _require_list(
                evidence_raw["covers"],
                "D4.payload.validity_evidence.covers",
            ),
        )
    )
    if coverage != LRT_VALIDITY_COVERAGE:
        msg = (
            "D4 validity evidence must cover the complete frozen pipeline in "
            "canonical order."
        )
        raise RevisionFitPolicyError(msg)
    _require_exact_value(
        evidence_raw["standard"],
        LRT_VALIDITY_STANDARD,
        "D4.payload.validity_evidence.standard",
    )
    _require_exact_value(
        evidence_raw["gate"],
        LRT_VALIDITY_GATE,
        "D4.payload.validity_evidence.gate",
    )
    _require_exact_value(
        payload["test_direction"],
        LRT_TEST_DIRECTION,
        "D4.payload.test_direction",
    )
    _require_exact_value(
        payload["statistic_transform"],
        LRT_STATISTIC_TRANSFORM,
        "D4.payload.statistic_transform",
    )
    return LRTPolicy(
        lrt_contract=lrt_contract,
        test_direction=LRT_TEST_DIRECTION,
        reference=LRTReference(
            family=LRT_REFERENCE_FAMILY,
            degrees_of_freedom=LRT_REFERENCE_DF,
            tail=LRT_REFERENCE_TAIL,
        ),
        statistic_transform=LRT_STATISTIC_TRANSFORM,
        boundary_handling=_require_enum(
            payload["boundary_handling"],
            LRT_BOUNDARY_HANDLING,
            "D4.payload.boundary_handling",
        ),
        failure_semantics=_require_enum(
            payload["failure_semantics"],
            LRT_FAILURE_SEMANTICS,
            "D4.payload.failure_semantics",
        ),
        validity_evidence=LRTValidityEvidence(
            standard=LRT_VALIDITY_STANDARD,
            covers=coverage,
            gate=LRT_VALIDITY_GATE,
        ),
    )


def _parse_d5(value: object) -> ConjunctionMultiplicityPolicy:
    payload = _require_object(value, "D5.payload")
    _require_exact_keys(
        payload,
        {
            "conjunction",
            "directional_fdr_control",
            "failed_hypothesis",
            "family",
            "multiplicity",
            "rho_role",
        },
        "D5.payload",
    )
    conjunction_raw = _require_object(
        payload["conjunction"],
        "D5.payload.conjunction",
    )
    _require_exact_keys(
        conjunction_raw,
        {"invalid_component", "missing_component", "mode", "sign_discordance"},
        "D5.payload.conjunction",
    )
    mode = _require_enum(
        conjunction_raw["mode"],
        {MAX_P_IUT, NO_CONJUNCTION},
        "D5.payload.conjunction.mode",
    )
    expected_component_policy = (
        SET_CONJUNCTION_P_ONE
        if mode == MAX_P_IUT
        else NO_CONJUNCTION_COMPONENT_POLICY
    )
    component_policies: dict[str, str] = {}
    for key in ("invalid_component", "missing_component", "sign_discordance"):
        component_policies[key] = _require_exact_value(
            conjunction_raw[key],
            expected_component_policy,
            f"D5.payload.conjunction.{key}",
        )

    multiplicity_raw = _require_object(
        payload["multiplicity"],
        "D5.payload.multiplicity",
    )
    _require_exact_keys(
        multiplicity_raw,
        {
            "descriptive_q_threshold",
            "descriptive_reporting_layer",
            "primary_method",
            "primary_q_threshold",
            "primary_reporting_layer",
            "sensitivity_method",
            "sensitivity_q_threshold",
            "sensitivity_reporting_layer",
            "threshold_comparison",
        },
        "D5.payload.multiplicity",
    )
    exact_multiplicity_values: tuple[tuple[str, object], ...] = (
        ("primary_method", "by"),
        ("sensitivity_method", "bh"),
        ("primary_q_threshold", 0.01),
        ("sensitivity_q_threshold", 0.01),
        ("descriptive_q_threshold", 0.05),
        ("threshold_comparison", INCLUSIVE_THRESHOLD),
        ("primary_reporting_layer", PRIMARY_REPORTING_LAYER),
        ("sensitivity_reporting_layer", SENSITIVITY_REPORTING_LAYER),
        ("descriptive_reporting_layer", DESCRIPTIVE_REPORTING_LAYER),
    )
    for key, expected in exact_multiplicity_values:
        _require_exact_value(
            multiplicity_raw[key],
            expected,
            f"D5.payload.multiplicity.{key}",
        )
    _require_exact_value(payload["family"], WITHIN_COHORT_FAMILY, "D5.payload.family")
    _require_exact_value(
        payload["failed_hypothesis"],
        FAILED_HYPOTHESIS_POLICY,
        "D5.payload.failed_hypothesis",
    )
    _require_exact_value(payload["rho_role"], RHO_ROLE, "D5.payload.rho_role")
    directional_fdr = _require_bool(
        payload["directional_fdr_control"],
        "D5.payload.directional_fdr_control",
    )
    if directional_fdr:
        msg = "D5 forbids a directional FDR-control claim from descriptive rho."
        raise RevisionFitPolicyError(msg)
    return ConjunctionMultiplicityPolicy(
        conjunction=ConjunctionPolicy(
            mode=mode,
            invalid_component=component_policies["invalid_component"],
            missing_component=component_policies["missing_component"],
            sign_discordance=component_policies["sign_discordance"],
        ),
        family=WITHIN_COHORT_FAMILY,
        failed_hypothesis=FAILED_HYPOTHESIS_POLICY,
        multiplicity=MultiplicityPolicy(
            primary_method="by",
            sensitivity_method="bh",
            primary_q_threshold=0.01,
            sensitivity_q_threshold=0.01,
            descriptive_q_threshold=0.05,
            threshold_comparison=INCLUSIVE_THRESHOLD,
            primary_reporting_layer=PRIMARY_REPORTING_LAYER,
            sensitivity_reporting_layer=SENSITIVITY_REPORTING_LAYER,
            descriptive_reporting_layer=DESCRIPTIVE_REPORTING_LAYER,
        ),
        rho_role=RHO_ROLE,
        directional_fdr_control=False,
    )


def _parse_d6(value: object) -> CalibrationScopePolicy:
    payload = _require_object(value, "D6.payload")
    _require_exact_keys(
        payload,
        {
            "cell_count",
            "claim_scope",
            "compute_authority",
            "design_authority",
            "path",
            "pre_calibration_result_inspection",
            "replicates_per_cell",
            "result_contingent_changes",
        },
        "D6.payload",
    )
    path = _require_enum(
        payload["path"],
        {FULL_EXTERNAL, NARROW_LOCAL, NO_EXTENSION},
        "D6.payload.path",
    )
    cell_count = _require_nonnegative_integer(
        payload["cell_count"],
        "D6.payload.cell_count",
    )
    replicates = _require_nonnegative_integer(
        payload["replicates_per_cell"],
        "D6.payload.replicates_per_cell",
    )
    result_contingent = _require_bool(
        payload["result_contingent_changes"],
        "D6.payload.result_contingent_changes",
    )
    if result_contingent:
        msg = "D6 forbids result-contingent calibration changes."
        raise RevisionFitPolicyError(msg)
    inspected = _require_bool(
        payload["pre_calibration_result_inspection"],
        "D6.payload.pre_calibration_result_inspection",
    )
    if inspected:
        msg = "D6 forbids pre-calibration result inspection."
        raise RevisionFitPolicyError(msg)

    if path == NO_EXTENSION:
        if cell_count != 0 or replicates != 0:
            msg = "D6 no-extension requires zero cells and zero replicates."
            raise RevisionFitPolicyError(msg)
        if payload["design_authority"] is not None:
            msg = "D6 no-extension requires null design_authority."
            raise RevisionFitPolicyError(msg)
        if payload["compute_authority"] is not None:
            msg = "D6 no-extension requires null compute_authority."
            raise RevisionFitPolicyError(msg)
        _require_exact_value(
            payload["claim_scope"],
            NO_EXTENSION_CLAIM_SCOPE,
            "D6.payload.claim_scope",
        )
        design_authority = None
        compute_authority = None
    else:
        design_authority = _parse_design_authority(payload["design_authority"])
        compute_authority = _parse_compute_authority(payload["compute_authority"])
        if path == FULL_EXTERNAL:
            if cell_count <= 0 or replicates != _FULL_REPLICATES_PER_CELL:
                msg = (
                    "D6 full-external requires a positive declared cell count and "
                    "exactly 1000 replicates per cell."
                )
                raise RevisionFitPolicyError(msg)
            _require_exact_value(
                compute_authority.mode,
                "external-approved",
                "D6.payload.compute_authority.mode",
            )
            _require_exact_value(
                payload["claim_scope"],
                FULL_CLAIM_SCOPE,
                "D6.payload.claim_scope",
            )
        else:
            if (
                cell_count != _NARROW_CELL_COUNT
                or replicates != _NARROW_REPLICATES_PER_CELL
            ):
                msg = "D6 narrow-local requires exactly three cells by 300 replicates."
                raise RevisionFitPolicyError(msg)
            _require_exact_value(
                compute_authority.mode,
                "local-bounded",
                "D6.payload.compute_authority.mode",
            )
            _require_exact_value(
                payload["claim_scope"],
                NARROW_CLAIM_SCOPE,
                "D6.payload.claim_scope",
            )

    return CalibrationScopePolicy(
        path=path,
        cell_count=cell_count,
        replicates_per_cell=replicates,
        design_authority=design_authority,
        compute_authority=compute_authority,
        claim_scope=_require_exact_text(
            payload["claim_scope"],
            "D6.payload.claim_scope",
        ),
        result_contingent_changes=False,
        pre_calibration_result_inspection=False,
    )


def _parse_design_authority(value: object) -> CalibrationDesignAuthority:
    authority = _require_object(value, "D6.payload.design_authority")
    _require_exact_keys(
        authority,
        {"design_id", "manifest_sha256"},
        "D6.payload.design_authority",
    )
    digest = _require_exact_text(
        authority["manifest_sha256"],
        "D6.payload.design_authority.manifest_sha256",
    )
    _require_sha256(digest, "D6.payload.design_authority.manifest_sha256")
    return CalibrationDesignAuthority(
        design_id=_require_exact_text(
            authority["design_id"],
            "D6.payload.design_authority.design_id",
        ),
        manifest_sha256=digest,
    )


def _parse_compute_authority(value: object) -> CalibrationComputeAuthority:
    authority = _require_object(value, "D6.payload.compute_authority")
    _require_exact_keys(
        authority,
        {"authority_id", "mode"},
        "D6.payload.compute_authority",
    )
    return CalibrationComputeAuthority(
        mode=_require_enum(
            authority["mode"],
            {"external-approved", "local-bounded"},
            "D6.payload.compute_authority.mode",
        ),
        authority_id=_require_exact_text(
            authority["authority_id"],
            "D6.payload.compute_authority.authority_id",
        ),
    )


def _require_conjunction_consistency(
    d3: ProviderHierarchyPolicy,
    d5: ConjunctionMultiplicityPolicy,
) -> None:
    expected_mode = (
        MAX_P_IUT
        if d3.all_three_conjunction_role == CONJUNCTION_SECONDARY
        else NO_CONJUNCTION
    )
    if d5.conjunction.mode != expected_mode:
        msg = "D3 conjunction role and D5 conjunction mode are inconsistent."
        raise RevisionFitPolicyError(msg)


def _load_canonical_envelope(
    content: bytes,
    decision_id: str,
) -> Mapping[str, object]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"Signed {decision_id} artifact must be valid UTF-8."
        raise RevisionFitPolicyError(msg) from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except json.JSONDecodeError as error:
        msg = f"Signed {decision_id} artifact is not valid JSON: {error.msg}."
        raise RevisionFitPolicyError(msg) from error
    _reject_surrogates(parsed, f"signed {decision_id} artifact")
    envelope = _require_object(parsed, f"signed {decision_id} artifact")
    _require_exact_keys(
        envelope,
        {"contract", "decision_id", "payload", "schema"},
        f"signed {decision_id} artifact",
    )
    canonical = _canonical_json_bytes(envelope, trailing_newline=True)
    if content != canonical:
        msg = (
            f"Signed {decision_id} artifact is not canonical JSON; require sorted "
            "keys, compact separators, UTF-8, and exactly one trailing LF."
        )
        raise RevisionFitPolicyError(msg)
    return envelope


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON key is forbidden: {key!r}."
            raise RevisionFitPolicyError(msg)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    msg = f"Non-finite JSON constant is forbidden: {value}."
    raise RevisionFitPolicyError(msg)


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = f"Non-finite JSON number is forbidden: {value}."
        raise RevisionFitPolicyError(msg)
    return parsed


def _reject_surrogates(value: object, label: str) -> None:
    if isinstance(value, str):
        if any(
            _SURROGATE_MIN <= ord(character) <= _SURROGATE_MAX
            for character in value
        ):
            msg = f"{label} contains an invalid Unicode surrogate."
            raise RevisionFitPolicyError(msg)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_surrogates(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key, f"{label} object key")
            _reject_surrogates(item, f"{label}.{key}")


def _canonical_json_bytes(value: object, *, trailing_newline: bool) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def _freeze_object(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()},
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _require_object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        msg = f"{label} must be a JSON object."
        raise RevisionFitPolicyError(msg)
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"{label} must be a JSON array."
        raise RevisionFitPolicyError(msg)
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
    raise RevisionFitPolicyError(msg)


def _require_exact_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        msg = f"{label} must be a nonblank exact string without outer whitespace."
        raise RevisionFitPolicyError(msg)
    if any(character in value for character in ("\x00", "\r")):
        msg = f"{label} contains a forbidden control character."
        raise RevisionFitPolicyError(msg)
    _reject_surrogates(value, label)
    return value


def _require_enum(value: object, allowed: set[str] | frozenset[str], label: str) -> str:
    text = _require_exact_text(value, label)
    if text not in allowed:
        msg = f"{label} must be one of {sorted(allowed)}; observed {text!r}."
        raise RevisionFitPolicyError(msg)
    return text


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{label} must be a JSON boolean."
        raise RevisionFitPolicyError(msg)
    return value


def _require_nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{label} must be a nonnegative JSON integer."
        raise RevisionFitPolicyError(msg)
    return value


def _require_exact_integer(value: object, expected: int, label: str) -> int:
    actual = _require_nonnegative_integer(value, label)
    if actual != expected:
        msg = f"{label} must equal {expected}; observed {actual}."
        raise RevisionFitPolicyError(msg)
    return actual


def _require_exact_value(
    value: object,
    expected: _ExactValue,
    label: str,
) -> _ExactValue:
    if type(value) is not type(expected) or value != expected:
        msg = f"{label} must equal {expected!r}; observed {value!r}."
        raise RevisionFitPolicyError(msg)
    return expected


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be exactly 64 lowercase hexadecimal characters."
        raise RevisionFitPolicyError(msg)

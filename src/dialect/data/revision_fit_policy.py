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
    STAGE_MINIMUM_DECISIONS,
    STAGE_SCOPED_APPROVAL_SCHEMA,
    RevisionApproval,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MACHINE_DECISION_SCHEMA: Final = "dialect-revision-machine-decision-v1"
"""Canonical schema shared with the signed D1/D2 decision envelopes."""

LEGACY_D3_CONTRACT: Final = "bmr-provider-hierarchy-v1"
"""Historical v4-only provider hierarchy without the MutSig support binding."""

D3_CONTRACT: Final = "bmr-provider-hierarchy-native-mutsig-support-v2"
"""Production v5 hierarchy plus exact native-MutSig implementation authority."""
D4_CONTRACT: Final = "profile-lrt-pvalue-policy-v3"
D5_CONTRACT: Final = "conjunction-multiplicity-family-policy-v3"
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
PRIMARY_BMR_PROVIDER: Final = "cbase"
SENSITIVITY_BMR_PROVIDERS: Final[tuple[str, str]] = ("dig", "mutsig")
CONJUNCTION_SECONDARY: Final = "secondary"
CONJUNCTION_OMITTED: Final = "omitted"

MUTSIG_FALLBACK_OR_FLOOR: Final = "none"
MUTSIG_LAMBDA_DTYPE: Final = "native-binary32"
MUTSIG_PREDECESSOR_PROOF: Final = "required-when-tail-endpoint-binds"
MUTSIG_RUNNER_PATH: Final = "analysis/run_tcga_revision_k500.py"
MUTSIG_SOURCE_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
MUTSIG_TENSOR_DTYPE: Final = "<f4"
MUTSIG_TENSOR_LAYOUT_CANARY: Final = (
    "nonuniform-2x3x2-fortran-gene-patient-effect-m0-n1-v1"
)
MUTSIG_TENSOR_ORDER: Final = "Fortran-(gene,patient,effect)"

LRT_TEST_DIRECTION: Final = "nondirectional-two-sided-dependence"
LRT_REFERENCE_FAMILY: Final = "chi-square"
LRT_REFERENCE_DF: Final = 1
LRT_REFERENCE_TAIL: Final = "upper-survival"
LRT_STATISTIC_TRANSFORM: Final = (
    "max(0,2*(alternative_log_likelihood-null_log_likelihood))"
)
UNDEFINED_RHO_DEGENERATE_NULL_HANDLING: Final = (
    "assign-p-one-with-explicit-boundary-status"
)
TASK_ABORT_NO_PUBLISHED_ROW: Final = "abort-task-no-published-row"
LRT_VALIDITY_STANDARD: Final = (
    "finite-sample-super-uniformity-under-frozen-analysis-pipeline"
)
LRT_VALIDITY_COVERAGE: Final[tuple[str, ...]] = (
    "bmr-estimation",
    "top-k-selection",
    "nuisance-fitting",
    "fit-support-convergence-failure",
    "effect-identifiability-and-reference-irregularity",
    "undefined-rho-degenerate-null-boundary",
    "complete-within-cohort-family",
)
LRT_VALIDITY_GATE: Final = "block-inferential-use-if-absent-invalid-or-inconclusive"

MAX_P_IUT: Final = "nondirectional-max-p-iut"
NO_CONJUNCTION: Final = "no-conjunction"
NO_CONJUNCTION_COMPONENT_POLICY: Final = "not-applicable-no-conjunction"
VALID_CONJUNCTION_COMPONENT_STATUSES: Final[tuple[str, str]] = (
    "valid-profile-lrt",
    "valid-degenerate-null-p-one",
)
CONJUNCTION_P_VALUE_COMBINER: Final = "max(p_cbase,p_dig,p_mutsig)"
INVALID_CONJUNCTION_COMPONENT: Final = "fail-cohort-conjunction-no-p-value"
MISSING_CONJUNCTION_COMPONENT: Final = "fail-cohort-conjunction-no-p-value"
SIGN_DISCORDANCE_POLICY: Final = "retain-max-p-direction-not-unanimous"
EFFECT_UNIDENTIFIABLE_POLICY: Final = "retain-valid-p-direction-unavailable"
COMPONENT_FAILURE_SEMANTICS: Final = "task-abort-no-published-row-no-p-one-substitution"
DIRECTION_PROVIDER_RULE: Final = "rho-negative-me-positive-co-zero-neutral"
UNDEFINED_RHO_DIRECTION_RULE: Final = "unavailable"
DIRECTION_CONSENSUS_RULE: Final = "unanimous-me-or-co-else-not-unanimous"
DIRECTION_REPORTING_LAYER: Final = "descriptive-post-rejection"
WITHIN_COHORT_FAMILY: Final = "one-complete-within-cohort-tested-pair-family"
TESTED_FAMILY_TOP_K: Final = 500
TESTED_FAMILY_FEATURE_RANKING: Final = "descending-total-eligible-mutation-event-count"
TESTED_FAMILY_TIE_BREAK: Final = "canonical-count-matrix-column-order"
TESTED_FAMILY_PROVIDER_SUPPORT: Final = "shared-native-cbase-dig-mutsig"
TESTED_FAMILY_PAIR_CONSTRUCTION: Final = "all-unordered-pairs-of-ordered-feature-axis"
TESTED_FAMILY_SAME_BASE_POLICY: Final = "exclude-before-fitting-and-testing"
NO_PRETEST_FILTER: Final = "none"
DESCRIPTIVE_METHODS: Final[tuple[str, str]] = ("by", "bh")
INCLUSIVE_THRESHOLD: Final = "inclusive-less-than-or-equal"
PRIMARY_REPORTING_LAYER: Final = "confirmatory-conditional-on-valid-marginals"
SENSITIVITY_REPORTING_LAYER: Final = "nominal-sensitivity"
DESCRIPTIVE_REPORTING_LAYER: Final = "descriptive"

FULL_EXTERNAL: Final = "full-external"
NARROW_LOCAL: Final = "narrow-local"
NO_EXTENSION: Final = "no-extension"
FULL_CLAIM_SCOPE: Final = "finite-scenario-calibration-evidence-only"
NARROW_CLAIM_SCOPE: Final = "narrow-exact-family-stress-evidence-only"
NO_EXTENSION_CLAIM_SCOPE: Final = "no-calibration-claims"

_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
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
class MutSigEffectPagesPolicy:
    """Native tensor page indices for missense and nonsense effects."""

    M: int
    N: int


@dataclass(frozen=True, slots=True)
class MutSigSupportPolicy:
    """Exact prospective native-MutSig finite-support contract signed under D3."""

    fallback_or_floor: str
    dtype: str
    effect_pages: MutSigEffectPagesPolicy
    lambda_dtype: str
    layout_canary: str
    normalization: str
    order: str
    predecessor_proof: str
    read_only: bool
    storage_contract: str
    support_contract: str
    support_rule: str
    tail_tolerance: float


@dataclass(frozen=True, slots=True)
class D3ImplementationBinding:
    """Reviewed source snapshot transitively signed by the production D3 artifact."""

    reviewed_scientific_commit: str
    runner_path: str
    runner_sha256: str
    source_contract_sha256: str
    source_file_count: int
    source_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class ProviderHierarchyPolicy:
    """D3 primary/sensitivity provider hierarchy."""

    primary_provider: str
    sensitivity_providers: tuple[str, str]
    all_three_conjunction_role: str
    burden_dependent_switching: bool
    rationale: str
    mutsig_support: MutSigSupportPolicy | None = None
    implementation_binding: D3ImplementationBinding | None = None


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
class EffectIdentifiabilityImplementationContract:
    """Exact rank certification and nonidentified-effect reporting contract."""

    contract: str
    relative_tolerance: float
    status_vocabulary: tuple[str, str, str]
    identified_status: str
    nonidentified_statuses: tuple[str, str]
    nonidentified_effect_blank_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NumericalImplementationContract:
    """Exact marginal, pair-fit, and effect implementation bound by signed D4."""

    marginal_fit_contract: str
    marginal_fit_max_iterations: int
    marginal_fit_total_kkt_tolerance: float
    marginal_fit_bracket_width_tolerance: float
    marginal_fit_fixed_point_tolerance: float
    marginal_fit_flat_likelihood_tie_break: str
    pair_fit_contract: str
    pair_fit_max_iterations: int
    pair_fit_total_kkt_tolerance: float
    pair_simplex_tolerance: float
    lrt_nestedness_tolerance: float
    effect_identifiability: EffectIdentifiabilityImplementationContract
    rho_contract: str
    undefined_rho_lrt_tolerance: float
    log_odds_ratio_contract: str


@dataclass(frozen=True, slots=True)
class D4ImplementationContract:
    """Caller-pinned implementation contract that signed D4 must match exactly."""

    lrt_contract: str
    numerical_implementation: NumericalImplementationContract


@dataclass(frozen=True, slots=True)
class FitFailureHandling:
    """Result-publication behavior for non-inferential fitting failures."""

    fit: str
    observation_support: str
    convergence: str


@dataclass(frozen=True, slots=True)
class LRTPolicy:
    """D4 profile-LRT p-value and failure contract."""

    lrt_contract: str
    numerical_implementation: NumericalImplementationContract
    test_direction: str
    reference: LRTReference
    statistic_transform: str
    undefined_rho_degenerate_null_handling: str
    failure_handling: FitFailureHandling
    validity_evidence: LRTValidityEvidence


@dataclass(frozen=True, slots=True)
class ConjunctionPolicy:
    """D5 nondirectional conjunction construction or explicit omission."""

    mode: str
    component_order: tuple[str, ...]
    valid_component_statuses: tuple[str, ...]
    p_value_combiner: str
    invalid_component: str
    missing_component: str
    sign_discordance: str
    effect_unidentifiable: str
    direction_affects_p_or_q: bool


@dataclass(frozen=True, slots=True)
class DirectionAnnotationPolicy:
    """D5 descriptive rho annotation kept outside inferential p/q values."""

    provider_rule: str
    undefined_rho_rule: str
    consensus_rule: str
    reporting_layer: str
    directional_fdr_control: bool


@dataclass(frozen=True, slots=True)
class TestedFamilyPolicy:
    """Exact feature-selection and pair-construction family tested within a cohort."""

    top_k: int
    feature_ranking: str
    tie_break: str
    provider_support: str
    pair_construction: str
    same_base_missense_nonsense: str
    epsilon_pretest_filter: str
    marginal_effect_pretest_filter: str
    family: str


@dataclass(frozen=True, slots=True)
class MultiplicityPolicy:
    """D5 complete-family multiplicity and reporting layers."""

    primary_method: str
    sensitivity_method: str
    primary_q_threshold: float
    sensitivity_q_threshold: float
    descriptive_methods: tuple[str, str]
    descriptive_q_threshold: float
    threshold_comparison: str
    primary_reporting_layer: str
    sensitivity_reporting_layer: str
    descriptive_reporting_layer: str


@dataclass(frozen=True, slots=True)
class ConjunctionMultiplicityPolicy:
    """D5 conjunction, family, multiplicity, and descriptive direction policy."""

    conjunction: ConjunctionPolicy
    direction_annotation: DirectionAnnotationPolicy
    tested_family: TestedFamilyPolicy
    multiplicity: MultiplicityPolicy
    component_failure_semantics: str


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
    expected_d4_implementation: D4ImplementationContract,
    expected_tested_family: TestedFamilyPolicy,
) -> RevisionFitPolicy:
    """Parse the exact signed D3--D6 policies from validated approval bytes.

    This is a fit-only policy gate.  It retains historical v4 compatibility and
    accepts v5 only when it is the exact singleton fit-stage D1--D6 authority.
    Both schemas require explicit fit-stage authority and ``go`` dispositions for
    D3--D6; neither upgrades another disposition or stage.  The expected
    implementation and tested-family contracts are supplied by the caller so an
    artifact cannot authorize a different execution path by naming it itself.

    Args:
        approval: Live immutable v4 or exact fit-scoped v5 approval returned by
            its validator.
        expected_d4_implementation: Frozen exact LRT, optimizer, tolerance,
            identifiability, effect-reporting, rho, and LOR contract required by the
            caller that will execute the fit.
        expected_tested_family: Frozen exact K, feature-ranking, provider-support,
            pair-construction, and no-pretest-filter contract implemented by the
            caller.

    Returns:
        Frozen typed policies and byte/hash receipts suitable for a runner manifest.

    Raises:
        RevisionFitPolicyError: If authority, canonical bytes, schema, or scientific
            cross-field constraints do not match the frozen contract.
    """
    if not isinstance(approval, RevisionApproval):
        msg = "approval must be a validated RevisionApproval authority."
        raise RevisionFitPolicyError(msg)
    if approval.schema not in {APPROVAL_SCHEMA, STAGE_SCOPED_APPROVAL_SCHEMA}:
        msg = (
            "Fit policy requires historical v4 or stage-scoped v5 approval; "
            f"observed {approval.schema!r}."
        )
        raise RevisionFitPolicyError(msg)
    if approval.schema == STAGE_SCOPED_APPROVAL_SCHEMA:
        _require_exact_v5_fit_scope(approval)
    if FIT_SEALED_TCGA_K500_STAGE not in approval.allowed_stages:
        msg = "Approval does not explicitly authorize the sealed TCGA K=500 fit stage."
        raise RevisionFitPolicyError(msg)
    expected_d4_implementation = _validate_expected_d4_implementation(
        expected_d4_implementation,
    )
    expected_tested_family = _validate_expected_tested_family(
        expected_tested_family,
    )

    envelopes: dict[str, Mapping[str, object]] = {}
    receipts: dict[str, FitPolicyReceipt] = {}
    for decision_id in FIT_POLICY_DECISION_IDS:
        envelope, receipt = _parse_signed_decision(approval, decision_id)
        envelopes[decision_id] = envelope
        receipts[decision_id] = receipt

    d3 = _parse_d3(
        envelopes["D3"]["payload"],
        contract=str(envelopes["D3"]["contract"]),
    )
    d4 = _parse_d4(
        envelopes["D4"]["payload"],
        expected_implementation=expected_d4_implementation,
    )
    d5 = _parse_d5(
        envelopes["D5"]["payload"],
        expected_tested_family=expected_tested_family,
    )
    d6 = _parse_d6(envelopes["D6"]["payload"])
    _require_conjunction_consistency(d3, d5)
    return RevisionFitPolicy(
        d3=d3,
        d4=d4,
        d5=d5,
        d6=d6,
        receipts=MappingProxyType(receipts),
    )


def _require_exact_v5_fit_scope(approval: RevisionApproval) -> None:
    """Recheck the complete singleton-stage v5 envelope before policy parsing."""
    expected_decisions = STAGE_MINIMUM_DECISIONS[FIT_SEALED_TCGA_K500_STAGE]
    if (
        approval.allowed_stages != (FIT_SEALED_TCGA_K500_STAGE,)
        or tuple(approval.stage_bindings) != (FIT_SEALED_TCGA_K500_STAGE,)
        or tuple(approval.decisions) != expected_decisions
        or tuple(approval.decision_digests) != expected_decisions
    ):
        msg = (
            "Stage-scoped v5 fit policy requires only fit-sealed-tcga-k500 "
            "with exact ordered D1-D6 authority and bindings."
        )
        raise RevisionFitPolicyError(msg)
    binding = approval.stage_bindings[FIT_SEALED_TCGA_K500_STAGE]
    expected_binding_keys = {
        "canonical_input_manifest_sha256",
        "provider_input_manifest_sha256",
    }
    if set(binding) != expected_binding_keys:
        msg = "Stage-scoped v5 fit policy has an inexact fit binding shape."
        raise RevisionFitPolicyError(msg)
    for key in sorted(expected_binding_keys):
        _require_sha256(binding[key], f"approval.stage_bindings fit {key}")
    for decision_id in expected_decisions:
        decision = approval.decisions[decision_id]
        if (
            decision.decision_id != decision_id
            or decision.disposition != GO_DISPOSITION
            or decision.allowed_stages != (FIT_SEALED_TCGA_K500_STAGE,)
            or decision.manifest_allowed_stages != (FIT_SEALED_TCGA_K500_STAGE,)
            or decision.manifest_stage_bindings != approval.stage_bindings
        ):
            msg = (
                f"Stage-scoped v5 {decision_id} does not bind the exact singleton "
                "fit envelope."
            )
            raise RevisionFitPolicyError(msg)
        _require_sha256(
            approval.decision_digests[decision_id],
            f"approval.decision_digests[{decision_id!r}]",
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
    expected_contract = (
        LEGACY_D3_CONTRACT
        if decision_id == "D3" and approval.schema == APPROVAL_SCHEMA
        else FIT_POLICY_CONTRACTS[decision_id]
    )
    if envelope["schema"] != MACHINE_DECISION_SCHEMA:
        msg = f"Signed {decision_id} artifact has an unsupported machine schema."
        raise RevisionFitPolicyError(msg)
    if envelope["decision_id"] != decision_id:
        msg = f"Signed {decision_id} artifact has the wrong decision_id."
        raise RevisionFitPolicyError(msg)
    if envelope["contract"] != expected_contract:
        msg = f"Signed {decision_id} artifact contract must be {expected_contract!r}."
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


def _parse_d3(value: object, *, contract: str) -> ProviderHierarchyPolicy:
    payload = _require_object(value, "D3.payload")
    expected_keys = {
        "all_three_conjunction_role",
        "burden_dependent_switching",
        "primary_provider",
        "rationale",
        "sensitivity_providers",
    }
    if contract == D3_CONTRACT:
        expected_keys.update({"implementation_binding", "mutsig_support"})
    elif contract != LEGACY_D3_CONTRACT:
        msg = f"D3 has an unsupported contract: {contract!r}."
        raise RevisionFitPolicyError(msg)
    _require_exact_keys(
        payload,
        expected_keys,
        "D3.payload",
    )
    primary = _require_exact_value(
        payload["primary_provider"],
        PRIMARY_BMR_PROVIDER,
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
    if sensitivities != SENSITIVITY_BMR_PROVIDERS:
        msg = (
            "D3 sensitivity_providers must be exactly ['dig', 'mutsig'] in "
            "the frozen order."
        )
        raise RevisionFitPolicyError(msg)
    switching = _require_bool(
        payload["burden_dependent_switching"],
        "D3.payload.burden_dependent_switching",
    )
    if switching:
        msg = "D3 forbids burden-dependent provider switching."
        raise RevisionFitPolicyError(msg)
    mutsig_support = None
    implementation_binding = None
    if contract == D3_CONTRACT:
        mutsig_support = _parse_d3_mutsig_support(payload["mutsig_support"])
        implementation_binding = _parse_d3_implementation_binding(
            payload["implementation_binding"],
        )
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
        mutsig_support=mutsig_support,
        implementation_binding=implementation_binding,
    )


def _parse_d3_mutsig_support(value: object) -> MutSigSupportPolicy:
    """Parse the exact support semantics; the runner pins its live constants."""
    support = _require_object(value, "D3.payload.mutsig_support")
    _require_exact_keys(
        support,
        {
            "dtype",
            "effect_pages",
            "fallback_or_floor",
            "lambda_dtype",
            "layout_canary",
            "normalization",
            "order",
            "predecessor_proof",
            "read_only",
            "storage_contract",
            "support_contract",
            "support_rule",
            "tail_tolerance",
        },
        "D3.payload.mutsig_support",
    )
    tail_tolerance = _require_positive_finite_float(
        support["tail_tolerance"],
        "D3.payload.mutsig_support.tail_tolerance",
    )
    read_only = _require_bool(
        support["read_only"],
        "D3.payload.mutsig_support.read_only",
    )
    if not read_only:
        msg = "D3.payload.mutsig_support.read_only must be true."
        raise RevisionFitPolicyError(msg)
    return MutSigSupportPolicy(
        fallback_or_floor=_require_exact_value(
            support["fallback_or_floor"],
            MUTSIG_FALLBACK_OR_FLOOR,
            "D3.payload.mutsig_support.fallback_or_floor",
        ),
        dtype=_require_exact_value(
            support["dtype"],
            MUTSIG_TENSOR_DTYPE,
            "D3.payload.mutsig_support.dtype",
        ),
        effect_pages=_parse_d3_mutsig_effect_pages(support["effect_pages"]),
        lambda_dtype=_require_exact_value(
            support["lambda_dtype"],
            MUTSIG_LAMBDA_DTYPE,
            "D3.payload.mutsig_support.lambda_dtype",
        ),
        layout_canary=_require_exact_value(
            support["layout_canary"],
            MUTSIG_TENSOR_LAYOUT_CANARY,
            "D3.payload.mutsig_support.layout_canary",
        ),
        normalization=_require_exact_text(
            support["normalization"],
            "D3.payload.mutsig_support.normalization",
        ),
        order=_require_exact_value(
            support["order"],
            MUTSIG_TENSOR_ORDER,
            "D3.payload.mutsig_support.order",
        ),
        predecessor_proof=_require_exact_value(
            support["predecessor_proof"],
            MUTSIG_PREDECESSOR_PROOF,
            "D3.payload.mutsig_support.predecessor_proof",
        ),
        read_only=read_only,
        storage_contract=_require_exact_text(
            support["storage_contract"],
            "D3.payload.mutsig_support.storage_contract",
        ),
        support_contract=_require_exact_text(
            support["support_contract"],
            "D3.payload.mutsig_support.support_contract",
        ),
        support_rule=_require_exact_text(
            support["support_rule"],
            "D3.payload.mutsig_support.support_rule",
        ),
        tail_tolerance=tail_tolerance,
    )


def _parse_d3_mutsig_effect_pages(value: object) -> MutSigEffectPagesPolicy:
    pages = _require_object(value, "D3.payload.mutsig_support.effect_pages")
    _require_exact_keys(
        pages,
        {"M", "N"},
        "D3.payload.mutsig_support.effect_pages",
    )
    return MutSigEffectPagesPolicy(
        M=_require_exact_integer(
            pages["M"],
            0,
            "D3.payload.mutsig_support.effect_pages.M",
        ),
        N=_require_exact_integer(
            pages["N"],
            1,
            "D3.payload.mutsig_support.effect_pages.N",
        ),
    )


def _parse_d3_implementation_binding(value: object) -> D3ImplementationBinding:
    binding = _require_object(value, "D3.payload.implementation_binding")
    _require_exact_keys(
        binding,
        {
            "reviewed_scientific_commit",
            "runner_path",
            "runner_sha256",
            "source_contract_sha256",
            "source_file_count",
            "source_snapshot_sha256",
        },
        "D3.payload.implementation_binding",
    )
    commit = _require_exact_text(
        binding["reviewed_scientific_commit"],
        "D3.payload.implementation_binding.reviewed_scientific_commit",
    )
    if MUTSIG_SOURCE_COMMIT_PATTERN.fullmatch(commit) is None:
        msg = (
            "D3.payload.implementation_binding.reviewed_scientific_commit must "
            "be exactly 40 lowercase hexadecimal characters."
        )
        raise RevisionFitPolicyError(msg)
    runner_path = _require_exact_value(
        binding["runner_path"],
        MUTSIG_RUNNER_PATH,
        "D3.payload.implementation_binding.runner_path",
    )
    runner_sha256 = _require_exact_text(
        binding["runner_sha256"],
        "D3.payload.implementation_binding.runner_sha256",
    )
    source_contract_sha256 = _require_exact_text(
        binding["source_contract_sha256"],
        "D3.payload.implementation_binding.source_contract_sha256",
    )
    source_snapshot_sha256 = _require_exact_text(
        binding["source_snapshot_sha256"],
        "D3.payload.implementation_binding.source_snapshot_sha256",
    )
    for label, digest in (
        ("runner_sha256", runner_sha256),
        ("source_contract_sha256", source_contract_sha256),
        ("source_snapshot_sha256", source_snapshot_sha256),
    ):
        _require_sha256(digest, f"D3.payload.implementation_binding.{label}")
    source_file_count = _require_nonnegative_integer(
        binding["source_file_count"],
        "D3.payload.implementation_binding.source_file_count",
    )
    if source_file_count == 0:
        msg = "D3 implementation source_file_count must be positive."
        raise RevisionFitPolicyError(msg)
    return D3ImplementationBinding(
        reviewed_scientific_commit=commit,
        runner_path=runner_path,
        runner_sha256=runner_sha256,
        source_contract_sha256=source_contract_sha256,
        source_file_count=source_file_count,
        source_snapshot_sha256=source_snapshot_sha256,
    )


def _validate_expected_d4_implementation(
    value: object,
) -> D4ImplementationContract:
    """Fail closed if the caller did not supply one coherent typed contract."""
    if not isinstance(value, D4ImplementationContract):
        msg = "expected_d4_implementation must be a D4ImplementationContract."
        raise RevisionFitPolicyError(msg)
    _require_exact_text(
        value.lrt_contract,
        "expected_d4_implementation.lrt_contract",
    )
    numerical = value.numerical_implementation
    if not isinstance(numerical, NumericalImplementationContract):
        msg = (
            "expected_d4_implementation.numerical_implementation must be a "
            "NumericalImplementationContract."
        )
        raise RevisionFitPolicyError(msg)
    _require_exact_text(
        numerical.marginal_fit_contract,
        "expected_d4_implementation.numerical_implementation.marginal_fit_contract",
    )
    marginal_max_iterations = _require_nonnegative_integer(
        numerical.marginal_fit_max_iterations,
        (
            "expected_d4_implementation.numerical_implementation."
            "marginal_fit_max_iterations"
        ),
    )
    if marginal_max_iterations == 0:
        msg = "Expected marginal_fit_max_iterations must be positive."
        raise RevisionFitPolicyError(msg)
    _require_exact_text(
        numerical.marginal_fit_flat_likelihood_tie_break,
        (
            "expected_d4_implementation.numerical_implementation."
            "marginal_fit_flat_likelihood_tie_break"
        ),
    )
    _require_exact_text(
        numerical.pair_fit_contract,
        "expected_d4_implementation.numerical_implementation.pair_fit_contract",
    )
    max_iterations = _require_nonnegative_integer(
        numerical.pair_fit_max_iterations,
        ("expected_d4_implementation.numerical_implementation.pair_fit_max_iterations"),
    )
    if max_iterations == 0:
        msg = "Expected pair_fit_max_iterations must be positive."
        raise RevisionFitPolicyError(msg)
    for field_name in (
        "marginal_fit_total_kkt_tolerance",
        "marginal_fit_bracket_width_tolerance",
        "marginal_fit_fixed_point_tolerance",
        "pair_fit_total_kkt_tolerance",
        "pair_simplex_tolerance",
        "lrt_nestedness_tolerance",
        "undefined_rho_lrt_tolerance",
    ):
        _require_positive_finite_float(
            getattr(numerical, field_name),
            f"expected_d4_implementation.numerical_implementation.{field_name}",
        )
    _require_exact_text(
        numerical.rho_contract,
        "expected_d4_implementation.numerical_implementation.rho_contract",
    )
    _require_exact_text(
        numerical.log_odds_ratio_contract,
        ("expected_d4_implementation.numerical_implementation.log_odds_ratio_contract"),
    )

    effect = numerical.effect_identifiability
    if not isinstance(effect, EffectIdentifiabilityImplementationContract):
        msg = (
            "expected_d4_implementation numerical effect_identifiability must be "
            "an EffectIdentifiabilityImplementationContract."
        )
        raise RevisionFitPolicyError(msg)
    _require_exact_text(
        effect.contract,
        "expected_d4_implementation effect-identifiability contract",
    )
    _require_positive_finite_float(
        effect.relative_tolerance,
        "expected_d4_implementation effect-identifiability relative_tolerance",
    )
    vocabulary = _require_expected_text_tuple(
        effect.status_vocabulary,
        expected_length=3,
        label="expected_d4_implementation effect-identifiability status_vocabulary",
    )
    identified_status = _require_exact_text(
        effect.identified_status,
        "expected_d4_implementation effect-identifiability identified_status",
    )
    nonidentified_statuses = _require_expected_text_tuple(
        effect.nonidentified_statuses,
        expected_length=2,
        label=(
            "expected_d4_implementation effect-identifiability nonidentified_statuses"
        ),
    )
    if vocabulary != (identified_status, *nonidentified_statuses):
        msg = (
            "Expected effect-identifiability status_vocabulary must contain the "
            "identified status followed by both nonidentified statuses."
        )
        raise RevisionFitPolicyError(msg)
    _require_expected_text_tuple(
        effect.nonidentified_effect_blank_fields,
        expected_length=None,
        label=(
            "expected_d4_implementation effect-identifiability "
            "nonidentified_effect_blank_fields"
        ),
    )
    return value


def _validate_expected_tested_family(value: object) -> TestedFamilyPolicy:
    """Fail closed unless the caller pins the one implemented K=500 family."""
    if not isinstance(value, TestedFamilyPolicy):
        msg = "expected_tested_family must be a TestedFamilyPolicy."
        raise RevisionFitPolicyError(msg)
    exact_fields: tuple[tuple[str, object], ...] = (
        ("top_k", TESTED_FAMILY_TOP_K),
        ("feature_ranking", TESTED_FAMILY_FEATURE_RANKING),
        ("tie_break", TESTED_FAMILY_TIE_BREAK),
        ("provider_support", TESTED_FAMILY_PROVIDER_SUPPORT),
        ("pair_construction", TESTED_FAMILY_PAIR_CONSTRUCTION),
        ("same_base_missense_nonsense", TESTED_FAMILY_SAME_BASE_POLICY),
        ("epsilon_pretest_filter", NO_PRETEST_FILTER),
        ("marginal_effect_pretest_filter", NO_PRETEST_FILTER),
        ("family", WITHIN_COHORT_FAMILY),
    )
    for field_name, expected in exact_fields:
        actual = getattr(value, field_name)
        label = f"expected_tested_family.{field_name}"
        if isinstance(expected, int):
            _require_exact_integer(actual, expected, label)
        else:
            _require_exact_value(actual, expected, label)
    return value


def _parse_d4(
    value: object,
    *,
    expected_implementation: D4ImplementationContract,
) -> LRTPolicy:
    payload = _require_object(value, "D4.payload")
    _require_exact_keys(
        payload,
        {
            "failure_handling",
            "lrt_contract",
            "numerical_implementation",
            "reference",
            "statistic_transform",
            "test_direction",
            "undefined_rho_degenerate_null_handling",
            "validity_evidence",
        },
        "D4.payload",
    )
    lrt_contract = _require_exact_text(
        payload["lrt_contract"],
        "D4.payload.lrt_contract",
    )
    if lrt_contract != expected_implementation.lrt_contract:
        msg = (
            "D4 lrt_contract does not bind the implementation contract required "
            "by the caller."
        )
        raise RevisionFitPolicyError(msg)

    numerical = _parse_d4_numerical_implementation(
        payload["numerical_implementation"],
        expected=expected_implementation.numerical_implementation,
    )

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
    failure_raw = _require_object(
        payload["failure_handling"],
        "D4.payload.failure_handling",
    )
    _require_exact_keys(
        failure_raw,
        {"convergence", "fit", "observation_support"},
        "D4.payload.failure_handling",
    )
    failure_handling = FitFailureHandling(
        fit=_require_exact_value(
            failure_raw["fit"],
            TASK_ABORT_NO_PUBLISHED_ROW,
            "D4.payload.failure_handling.fit",
        ),
        observation_support=_require_exact_value(
            failure_raw["observation_support"],
            TASK_ABORT_NO_PUBLISHED_ROW,
            "D4.payload.failure_handling.observation_support",
        ),
        convergence=_require_exact_value(
            failure_raw["convergence"],
            TASK_ABORT_NO_PUBLISHED_ROW,
            "D4.payload.failure_handling.convergence",
        ),
    )
    return LRTPolicy(
        lrt_contract=lrt_contract,
        numerical_implementation=numerical,
        test_direction=LRT_TEST_DIRECTION,
        reference=LRTReference(
            family=LRT_REFERENCE_FAMILY,
            degrees_of_freedom=LRT_REFERENCE_DF,
            tail=LRT_REFERENCE_TAIL,
        ),
        statistic_transform=LRT_STATISTIC_TRANSFORM,
        undefined_rho_degenerate_null_handling=_require_exact_value(
            payload["undefined_rho_degenerate_null_handling"],
            UNDEFINED_RHO_DEGENERATE_NULL_HANDLING,
            "D4.payload.undefined_rho_degenerate_null_handling",
        ),
        failure_handling=failure_handling,
        validity_evidence=LRTValidityEvidence(
            standard=LRT_VALIDITY_STANDARD,
            covers=coverage,
            gate=LRT_VALIDITY_GATE,
        ),
    )


def _parse_d4_numerical_implementation(
    value: object,
    *,
    expected: NumericalImplementationContract,
) -> NumericalImplementationContract:
    numerical = _require_object(value, "D4.payload.numerical_implementation")
    _require_exact_keys(
        numerical,
        {
            "effect_identifiability",
            "log_odds_ratio_contract",
            "lrt_nestedness_tolerance",
            "marginal_fit_bracket_width_tolerance",
            "marginal_fit_contract",
            "marginal_fit_fixed_point_tolerance",
            "marginal_fit_flat_likelihood_tie_break",
            "marginal_fit_max_iterations",
            "marginal_fit_total_kkt_tolerance",
            "pair_fit_contract",
            "pair_fit_max_iterations",
            "pair_fit_total_kkt_tolerance",
            "pair_simplex_tolerance",
            "rho_contract",
            "undefined_rho_lrt_tolerance",
        },
        "D4.payload.numerical_implementation",
    )
    effect_raw = _require_object(
        numerical["effect_identifiability"],
        "D4.payload.numerical_implementation.effect_identifiability",
    )
    _require_exact_keys(
        effect_raw,
        {
            "contract",
            "identified_status",
            "nonidentified_effect_blank_fields",
            "nonidentified_statuses",
            "relative_tolerance",
            "status_vocabulary",
        },
        "D4.payload.numerical_implementation.effect_identifiability",
    )
    vocabulary = _require_signed_text_sequence(
        effect_raw["status_vocabulary"],
        label=(
            "D4.payload.numerical_implementation.effect_identifiability."
            "status_vocabulary"
        ),
    )
    nonidentified_statuses = _require_signed_text_sequence(
        effect_raw["nonidentified_statuses"],
        label=(
            "D4.payload.numerical_implementation.effect_identifiability."
            "nonidentified_statuses"
        ),
    )
    blank_fields = _require_signed_text_sequence(
        effect_raw["nonidentified_effect_blank_fields"],
        label=(
            "D4.payload.numerical_implementation.effect_identifiability."
            "nonidentified_effect_blank_fields"
        ),
    )
    comparisons: tuple[tuple[object, object, str], ...] = (
        (
            numerical["marginal_fit_contract"],
            expected.marginal_fit_contract,
            "marginal_fit_contract",
        ),
        (
            numerical["marginal_fit_max_iterations"],
            expected.marginal_fit_max_iterations,
            "marginal_fit_max_iterations",
        ),
        (
            numerical["marginal_fit_total_kkt_tolerance"],
            expected.marginal_fit_total_kkt_tolerance,
            "marginal_fit_total_kkt_tolerance",
        ),
        (
            numerical["marginal_fit_bracket_width_tolerance"],
            expected.marginal_fit_bracket_width_tolerance,
            "marginal_fit_bracket_width_tolerance",
        ),
        (
            numerical["marginal_fit_fixed_point_tolerance"],
            expected.marginal_fit_fixed_point_tolerance,
            "marginal_fit_fixed_point_tolerance",
        ),
        (
            numerical["marginal_fit_flat_likelihood_tie_break"],
            expected.marginal_fit_flat_likelihood_tie_break,
            "marginal_fit_flat_likelihood_tie_break",
        ),
        (
            numerical["pair_fit_contract"],
            expected.pair_fit_contract,
            "pair_fit_contract",
        ),
        (
            numerical["pair_fit_max_iterations"],
            expected.pair_fit_max_iterations,
            "pair_fit_max_iterations",
        ),
        (
            numerical["pair_fit_total_kkt_tolerance"],
            expected.pair_fit_total_kkt_tolerance,
            "pair_fit_total_kkt_tolerance",
        ),
        (
            numerical["pair_simplex_tolerance"],
            expected.pair_simplex_tolerance,
            "pair_simplex_tolerance",
        ),
        (
            numerical["lrt_nestedness_tolerance"],
            expected.lrt_nestedness_tolerance,
            "lrt_nestedness_tolerance",
        ),
        (numerical["rho_contract"], expected.rho_contract, "rho_contract"),
        (
            numerical["undefined_rho_lrt_tolerance"],
            expected.undefined_rho_lrt_tolerance,
            "undefined_rho_lrt_tolerance",
        ),
        (
            numerical["log_odds_ratio_contract"],
            expected.log_odds_ratio_contract,
            "log_odds_ratio_contract",
        ),
        (
            effect_raw["contract"],
            expected.effect_identifiability.contract,
            "effect_identifiability.contract",
        ),
        (
            effect_raw["relative_tolerance"],
            expected.effect_identifiability.relative_tolerance,
            "effect_identifiability.relative_tolerance",
        ),
        (
            effect_raw["identified_status"],
            expected.effect_identifiability.identified_status,
            "effect_identifiability.identified_status",
        ),
        (
            vocabulary,
            expected.effect_identifiability.status_vocabulary,
            "effect_identifiability.status_vocabulary",
        ),
        (
            nonidentified_statuses,
            expected.effect_identifiability.nonidentified_statuses,
            "effect_identifiability.nonidentified_statuses",
        ),
        (
            blank_fields,
            expected.effect_identifiability.nonidentified_effect_blank_fields,
            "effect_identifiability.nonidentified_effect_blank_fields",
        ),
    )
    for actual, required, suffix in comparisons:
        _require_exact_value(
            actual,
            required,
            f"D4.payload.numerical_implementation.{suffix}",
        )
    return expected


def _parse_d5_direction_annotation(
    value: object,
    *,
    mode: str,
) -> DirectionAnnotationPolicy:
    """Parse rho direction as a descriptive layer independent of p/q values."""
    direction = _require_object(value, "D5.payload.direction_annotation")
    _require_exact_keys(
        direction,
        {
            "consensus_rule",
            "directional_fdr_control",
            "provider_rule",
            "reporting_layer",
            "undefined_rho_rule",
        },
        "D5.payload.direction_annotation",
    )
    expected_consensus_rule = (
        DIRECTION_CONSENSUS_RULE
        if mode == MAX_P_IUT
        else NO_CONJUNCTION_COMPONENT_POLICY
    )
    exact_fields = (
        ("provider_rule", DIRECTION_PROVIDER_RULE),
        ("undefined_rho_rule", UNDEFINED_RHO_DIRECTION_RULE),
        ("consensus_rule", expected_consensus_rule),
        ("reporting_layer", DIRECTION_REPORTING_LAYER),
    )
    for key, expected in exact_fields:
        _require_exact_value(
            direction[key],
            expected,
            f"D5.payload.direction_annotation.{key}",
        )
    directional_fdr = _require_bool(
        direction["directional_fdr_control"],
        "D5.payload.direction_annotation.directional_fdr_control",
    )
    if directional_fdr:
        msg = "D5 forbids a directional FDR-control claim from descriptive rho."
        raise RevisionFitPolicyError(msg)
    return DirectionAnnotationPolicy(
        provider_rule=DIRECTION_PROVIDER_RULE,
        undefined_rho_rule=UNDEFINED_RHO_DIRECTION_RULE,
        consensus_rule=expected_consensus_rule,
        reporting_layer=DIRECTION_REPORTING_LAYER,
        directional_fdr_control=False,
    )


def _parse_d5_tested_family(
    value: object,
    *,
    expected: TestedFamilyPolicy,
) -> TestedFamilyPolicy:
    family = _require_object(value, "D5.payload.tested_family")
    _require_exact_keys(
        family,
        {
            "epsilon_pretest_filter",
            "family",
            "feature_ranking",
            "marginal_effect_pretest_filter",
            "pair_construction",
            "provider_support",
            "same_base_missense_nonsense",
            "tie_break",
            "top_k",
        },
        "D5.payload.tested_family",
    )
    exact_fields: tuple[tuple[str, object], ...] = (
        ("top_k", expected.top_k),
        ("feature_ranking", expected.feature_ranking),
        ("tie_break", expected.tie_break),
        ("provider_support", expected.provider_support),
        ("pair_construction", expected.pair_construction),
        ("same_base_missense_nonsense", expected.same_base_missense_nonsense),
        ("epsilon_pretest_filter", expected.epsilon_pretest_filter),
        ("marginal_effect_pretest_filter", expected.marginal_effect_pretest_filter),
        ("family", expected.family),
    )
    for field_name, expected_value in exact_fields:
        label = f"D5.payload.tested_family.{field_name}"
        if isinstance(expected_value, int):
            _require_exact_integer(family[field_name], expected_value, label)
        else:
            _require_exact_value(family[field_name], expected_value, label)
    return expected


def _parse_d5(
    value: object,
    *,
    expected_tested_family: TestedFamilyPolicy,
) -> ConjunctionMultiplicityPolicy:
    payload = _require_object(value, "D5.payload")
    _require_exact_keys(
        payload,
        {
            "component_failure_semantics",
            "conjunction",
            "direction_annotation",
            "multiplicity",
            "tested_family",
        },
        "D5.payload",
    )
    conjunction_raw = _require_object(
        payload["conjunction"],
        "D5.payload.conjunction",
    )
    _require_exact_keys(
        conjunction_raw,
        {
            "component_order",
            "direction_affects_p_or_q",
            "effect_unidentifiable",
            "invalid_component",
            "missing_component",
            "mode",
            "p_value_combiner",
            "sign_discordance",
            "valid_component_statuses",
        },
        "D5.payload.conjunction",
    )
    mode = _require_enum(
        conjunction_raw["mode"],
        {MAX_P_IUT, NO_CONJUNCTION},
        "D5.payload.conjunction.mode",
    )
    if mode == MAX_P_IUT:
        component_order = tuple(
            _require_enum(
                item,
                BMR_PROVIDERS,
                f"D5.payload.conjunction.component_order[{index}]",
            )
            for index, item in enumerate(
                _require_list(
                    conjunction_raw["component_order"],
                    "D5.payload.conjunction.component_order",
                ),
            )
        )
        if (
            len(component_order) != len(BMR_PROVIDERS)
            or set(
                component_order,
            )
            != BMR_PROVIDERS
        ):
            msg = (
                "D5 conjunction component_order must contain each BMR provider "
                "exactly once."
            )
            raise RevisionFitPolicyError(msg)
        valid_component_statuses = _require_signed_text_sequence(
            conjunction_raw["valid_component_statuses"],
            label="D5.payload.conjunction.valid_component_statuses",
        )
        _require_exact_value(
            valid_component_statuses,
            VALID_CONJUNCTION_COMPONENT_STATUSES,
            "D5.payload.conjunction.valid_component_statuses",
        )
        p_value_combiner = _require_exact_value(
            conjunction_raw["p_value_combiner"],
            CONJUNCTION_P_VALUE_COMBINER,
            "D5.payload.conjunction.p_value_combiner",
        )
        invalid_component = _require_exact_value(
            conjunction_raw["invalid_component"],
            INVALID_CONJUNCTION_COMPONENT,
            "D5.payload.conjunction.invalid_component",
        )
        missing_component = _require_exact_value(
            conjunction_raw["missing_component"],
            MISSING_CONJUNCTION_COMPONENT,
            "D5.payload.conjunction.missing_component",
        )
        sign_discordance = _require_exact_value(
            conjunction_raw["sign_discordance"],
            SIGN_DISCORDANCE_POLICY,
            "D5.payload.conjunction.sign_discordance",
        )
        effect_unidentifiable = _require_exact_value(
            conjunction_raw["effect_unidentifiable"],
            EFFECT_UNIDENTIFIABLE_POLICY,
            "D5.payload.conjunction.effect_unidentifiable",
        )
        component_failure_semantics = _require_exact_value(
            payload["component_failure_semantics"],
            COMPONENT_FAILURE_SEMANTICS,
            "D5.payload.component_failure_semantics",
        )
    else:
        _require_exact_value(
            conjunction_raw["component_order"],
            [],
            "D5.payload.conjunction.component_order",
        )
        _require_exact_value(
            conjunction_raw["valid_component_statuses"],
            [],
            "D5.payload.conjunction.valid_component_statuses",
        )
        component_order = ()
        valid_component_statuses = ()
        component_policies = (
            "p_value_combiner",
            "invalid_component",
            "missing_component",
            "sign_discordance",
            "effect_unidentifiable",
        )
        for key in component_policies:
            _require_exact_value(
                conjunction_raw[key],
                NO_CONJUNCTION_COMPONENT_POLICY,
                f"D5.payload.conjunction.{key}",
            )
        p_value_combiner = NO_CONJUNCTION_COMPONENT_POLICY
        invalid_component = NO_CONJUNCTION_COMPONENT_POLICY
        missing_component = NO_CONJUNCTION_COMPONENT_POLICY
        sign_discordance = NO_CONJUNCTION_COMPONENT_POLICY
        effect_unidentifiable = NO_CONJUNCTION_COMPONENT_POLICY
        component_failure_semantics = _require_exact_value(
            payload["component_failure_semantics"],
            NO_CONJUNCTION_COMPONENT_POLICY,
            "D5.payload.component_failure_semantics",
        )
    direction_affects_p_or_q = _require_bool(
        conjunction_raw["direction_affects_p_or_q"],
        "D5.payload.conjunction.direction_affects_p_or_q",
    )
    if direction_affects_p_or_q:
        msg = "D5 forbids rho direction from changing conjunction p- or q-values."
        raise RevisionFitPolicyError(msg)

    direction_annotation = _parse_d5_direction_annotation(
        payload["direction_annotation"],
        mode=mode,
    )
    tested_family = _parse_d5_tested_family(
        payload["tested_family"],
        expected=expected_tested_family,
    )

    multiplicity_raw = _require_object(
        payload["multiplicity"],
        "D5.payload.multiplicity",
    )
    _require_exact_keys(
        multiplicity_raw,
        {
            "descriptive_methods",
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
    descriptive_methods = _require_signed_text_sequence(
        multiplicity_raw["descriptive_methods"],
        label="D5.payload.multiplicity.descriptive_methods",
    )
    _require_exact_value(
        descriptive_methods,
        DESCRIPTIVE_METHODS,
        "D5.payload.multiplicity.descriptive_methods",
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
    return ConjunctionMultiplicityPolicy(
        conjunction=ConjunctionPolicy(
            mode=mode,
            component_order=component_order,
            valid_component_statuses=valid_component_statuses,
            p_value_combiner=p_value_combiner,
            invalid_component=invalid_component,
            missing_component=missing_component,
            sign_discordance=sign_discordance,
            effect_unidentifiable=effect_unidentifiable,
            direction_affects_p_or_q=False,
        ),
        direction_annotation=direction_annotation,
        tested_family=tested_family,
        multiplicity=MultiplicityPolicy(
            primary_method="by",
            sensitivity_method="bh",
            primary_q_threshold=0.01,
            sensitivity_q_threshold=0.01,
            descriptive_methods=DESCRIPTIVE_METHODS,
            descriptive_q_threshold=0.05,
            threshold_comparison=INCLUSIVE_THRESHOLD,
            primary_reporting_layer=PRIMARY_REPORTING_LAYER,
            sensitivity_reporting_layer=SENSITIVITY_REPORTING_LAYER,
            descriptive_reporting_layer=DESCRIPTIVE_REPORTING_LAYER,
        ),
        component_failure_semantics=component_failure_semantics,
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
    if expected_mode == MAX_P_IUT and d5.conjunction.component_order != (
        d3.primary_provider,
        *d3.sensitivity_providers,
    ):
        msg = (
            "D3 provider hierarchy and D5 conjunction component order are inconsistent."
        )
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
            _SURROGATE_MIN <= ord(character) <= _SURROGATE_MAX for character in value
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


def _require_positive_finite_float(value: object, label: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        msg = f"{label} must be a positive finite JSON float."
        raise RevisionFitPolicyError(msg)
    return value


def _require_expected_text_tuple(
    value: object,
    *,
    expected_length: int | None,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        msg = f"{label} must be a nonempty tuple of exact strings."
        raise RevisionFitPolicyError(msg)
    if expected_length is not None and len(value) != expected_length:
        msg = f"{label} must contain exactly {expected_length} entries."
        raise RevisionFitPolicyError(msg)
    result = tuple(
        _require_exact_text(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        msg = f"{label} must not contain duplicate entries."
        raise RevisionFitPolicyError(msg)
    return result


def _require_signed_text_sequence(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(
        _require_exact_text(item, f"{label}[{index}]")
        for index, item in enumerate(_require_list(value, label))
    )


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

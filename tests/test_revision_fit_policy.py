import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

from dialect.data.revision_approval import (
    APPROVAL_SCHEMA,
    FIT_SEALED_TCGA_K500_STAGE,
    ArtifactReceipt,
    DecisionApproval,
    RevisionApproval,
    SourceNotice,
)
from dialect.data.revision_fit_policy import (
    COMPONENT_FAILURE_SEMANTICS,
    CONJUNCTION_P_VALUE_COMBINER,
    D3_CONTRACT,
    D4_CONTRACT,
    D5_CONTRACT,
    D6_CONTRACT,
    DIRECTION_CONSENSUS_RULE,
    EFFECT_UNIDENTIFIABLE_POLICY,
    LRT_STATISTIC_TRANSFORM,
    LRT_VALIDITY_COVERAGE,
    MACHINE_DECISION_SCHEMA,
    MAX_P_IUT,
    NARROW_LOCAL,
    NO_CONJUNCTION,
    NO_EXTENSION,
    TASK_ABORT_NO_PUBLISHED_ROW,
    UNDEFINED_RHO_DEGENERATE_NULL_HANDLING,
    VALID_CONJUNCTION_COMPONENT_STATUSES,
    D4ImplementationContract,
    EffectIdentifiabilityImplementationContract,
    NumericalImplementationContract,
    RevisionFitPolicyError,
    validate_revision_fit_policy,
)
from dialect.data.revision_fit_policy import (
    TestedFamilyPolicy as FamilyPolicyContract,
)

LRT_CONTRACT = "driver-independence-constrained-mle-v1"
MARGINAL_FIT_CONTRACT = "deterministic-concave-score-bisection-total-kkt-v1"
MARGINAL_FIT_MAX_ITERATIONS = 1000
MARGINAL_FIT_TOTAL_KKT_TOLERANCE = 1e-8
MARGINAL_FIT_BRACKET_WIDTH_TOLERANCE = 1e-12
MARGINAL_FIT_FIXED_POINT_TOLERANCE = 1e-8
MARGINAL_FIT_FLAT_LIKELIHOOD_TIE_BREAK = "pi-zero"
PAIR_FIT_CONTRACT = "deterministic-simplex-coordinate-ascent-total-kkt-v2"
PAIR_FIT_MAX_ITERATIONS = 1000
PAIR_FIT_TOTAL_KKT_TOLERANCE = 1e-8
PAIR_SIMPLEX_TOLERANCE = 1e-12
LRT_NESTEDNESS_TOLERANCE = 1e-8
EFFECT_IDENTIFIABILITY_CONTRACT = "full-affine-rank-relative-svd-1e-12-conservative-v1"
EFFECT_IDENTIFIED_STATUS = "full-affine-rank"
EFFECT_RANK_DEFICIENT_STATUS = "rank-deficient"
EFFECT_UNDERFLOW_STATUS = "rank-not-certified-underflow"
NONIDENTIFIED_EFFECT_BLANK_FIELDS = (
    "Tau_1X",
    "Tau_X1",
    "Rho",
    "Log Odds Ratio",
    "Wald Statistic",
)
RHO_CONTRACT = "marshall-olkin-identifiable-finite-or-degenerate-null-v2"
UNDEFINED_RHO_LRT_TOLERANCE = 1e-8
LOG_ODDS_RATIO_CONTRACT = "conventional-latent-odds-00x11-over-01x10-identifiable-v2"
EXPECTED_D4_IMPLEMENTATION = D4ImplementationContract(
    lrt_contract=LRT_CONTRACT,
    numerical_implementation=NumericalImplementationContract(
        marginal_fit_contract=MARGINAL_FIT_CONTRACT,
        marginal_fit_max_iterations=MARGINAL_FIT_MAX_ITERATIONS,
        marginal_fit_total_kkt_tolerance=MARGINAL_FIT_TOTAL_KKT_TOLERANCE,
        marginal_fit_bracket_width_tolerance=(MARGINAL_FIT_BRACKET_WIDTH_TOLERANCE),
        marginal_fit_fixed_point_tolerance=MARGINAL_FIT_FIXED_POINT_TOLERANCE,
        marginal_fit_flat_likelihood_tie_break=(MARGINAL_FIT_FLAT_LIKELIHOOD_TIE_BREAK),
        pair_fit_contract=PAIR_FIT_CONTRACT,
        pair_fit_max_iterations=PAIR_FIT_MAX_ITERATIONS,
        pair_fit_total_kkt_tolerance=PAIR_FIT_TOTAL_KKT_TOLERANCE,
        pair_simplex_tolerance=PAIR_SIMPLEX_TOLERANCE,
        lrt_nestedness_tolerance=LRT_NESTEDNESS_TOLERANCE,
        effect_identifiability=EffectIdentifiabilityImplementationContract(
            contract=EFFECT_IDENTIFIABILITY_CONTRACT,
            relative_tolerance=PAIR_SIMPLEX_TOLERANCE,
            status_vocabulary=(
                EFFECT_IDENTIFIED_STATUS,
                EFFECT_RANK_DEFICIENT_STATUS,
                EFFECT_UNDERFLOW_STATUS,
            ),
            identified_status=EFFECT_IDENTIFIED_STATUS,
            nonidentified_statuses=(
                EFFECT_RANK_DEFICIENT_STATUS,
                EFFECT_UNDERFLOW_STATUS,
            ),
            nonidentified_effect_blank_fields=NONIDENTIFIED_EFFECT_BLANK_FIELDS,
        ),
        rho_contract=RHO_CONTRACT,
        undefined_rho_lrt_tolerance=UNDEFINED_RHO_LRT_TOLERANCE,
        log_odds_ratio_contract=LOG_ODDS_RATIO_CONTRACT,
    ),
)
EXPECTED_TESTED_FAMILY = FamilyPolicyContract(
    top_k=500,
    feature_ranking="descending-total-eligible-mutation-event-count",
    tie_break="canonical-count-matrix-column-order",
    provider_support="shared-native-cbase-dig-mutsig",
    pair_construction="all-unordered-pairs-of-ordered-feature-axis",
    same_base_missense_nonsense="exclude-before-fitting-and-testing",
    epsilon_pretest_filter="none",
    marginal_effect_pretest_filter="none",
    family="one-complete-within-cohort-tested-pair-family",
)
CONTRACTS = {
    "D3": D3_CONTRACT,
    "D4": D4_CONTRACT,
    "D5": D5_CONTRACT,
    "D6": D6_CONTRACT,
}


def _canonical(value, *, newline=True):
    content = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return content + (b"\n" if newline else b"")


def _d3(*, conjunction_role="secondary"):
    return {
        "all_three_conjunction_role": conjunction_role,
        "burden_dependent_switching": False,
        "primary_provider": "cbase",
        "rationale": "CBaSE is primary for continuity, not claimed superiority.",
        "sensitivity_providers": ["dig", "mutsig"],
    }


def _d4():
    return {
        "failure_handling": {
            "convergence": TASK_ABORT_NO_PUBLISHED_ROW,
            "fit": TASK_ABORT_NO_PUBLISHED_ROW,
            "observation_support": TASK_ABORT_NO_PUBLISHED_ROW,
        },
        "lrt_contract": LRT_CONTRACT,
        "numerical_implementation": {
            "effect_identifiability": {
                "contract": EFFECT_IDENTIFIABILITY_CONTRACT,
                "identified_status": EFFECT_IDENTIFIED_STATUS,
                "nonidentified_effect_blank_fields": list(
                    NONIDENTIFIED_EFFECT_BLANK_FIELDS,
                ),
                "nonidentified_statuses": [
                    EFFECT_RANK_DEFICIENT_STATUS,
                    EFFECT_UNDERFLOW_STATUS,
                ],
                "relative_tolerance": PAIR_SIMPLEX_TOLERANCE,
                "status_vocabulary": [
                    EFFECT_IDENTIFIED_STATUS,
                    EFFECT_RANK_DEFICIENT_STATUS,
                    EFFECT_UNDERFLOW_STATUS,
                ],
            },
            "log_odds_ratio_contract": LOG_ODDS_RATIO_CONTRACT,
            "lrt_nestedness_tolerance": LRT_NESTEDNESS_TOLERANCE,
            "marginal_fit_bracket_width_tolerance": (
                MARGINAL_FIT_BRACKET_WIDTH_TOLERANCE
            ),
            "marginal_fit_contract": MARGINAL_FIT_CONTRACT,
            "marginal_fit_fixed_point_tolerance": (MARGINAL_FIT_FIXED_POINT_TOLERANCE),
            "marginal_fit_flat_likelihood_tie_break": (
                MARGINAL_FIT_FLAT_LIKELIHOOD_TIE_BREAK
            ),
            "marginal_fit_max_iterations": MARGINAL_FIT_MAX_ITERATIONS,
            "marginal_fit_total_kkt_tolerance": (MARGINAL_FIT_TOTAL_KKT_TOLERANCE),
            "pair_fit_contract": PAIR_FIT_CONTRACT,
            "pair_fit_max_iterations": PAIR_FIT_MAX_ITERATIONS,
            "pair_fit_total_kkt_tolerance": PAIR_FIT_TOTAL_KKT_TOLERANCE,
            "pair_simplex_tolerance": PAIR_SIMPLEX_TOLERANCE,
            "rho_contract": RHO_CONTRACT,
            "undefined_rho_lrt_tolerance": UNDEFINED_RHO_LRT_TOLERANCE,
        },
        "reference": {
            "degrees_of_freedom": 1,
            "family": "chi-square",
            "tail": "upper-survival",
        },
        "statistic_transform": LRT_STATISTIC_TRANSFORM,
        "test_direction": "nondirectional-two-sided-dependence",
        "undefined_rho_degenerate_null_handling": (
            UNDEFINED_RHO_DEGENERATE_NULL_HANDLING
        ),
        "validity_evidence": {
            "covers": list(LRT_VALIDITY_COVERAGE),
            "gate": "block-inferential-use-if-absent-invalid-or-inconclusive",
            "standard": (
                "finite-sample-super-uniformity-under-frozen-analysis-pipeline"
            ),
        },
    }


def _d5(*, mode=MAX_P_IUT, component_order=None):
    enabled = mode == MAX_P_IUT
    not_applicable = "not-applicable-no-conjunction"
    component_order = (
        ["cbase", "dig", "mutsig"] if component_order is None else component_order
    )
    return {
        "component_failure_semantics": (
            "task-abort-no-published-row-no-p-one-substitution"
            if enabled
            else not_applicable
        ),
        "conjunction": {
            "component_order": component_order if enabled else [],
            "direction_affects_p_or_q": False,
            "effect_unidentifiable": (
                "retain-valid-p-direction-unavailable" if enabled else not_applicable
            ),
            "invalid_component": (
                "fail-cohort-conjunction-no-p-value" if enabled else not_applicable
            ),
            "missing_component": (
                "fail-cohort-conjunction-no-p-value" if enabled else not_applicable
            ),
            "mode": mode,
            "p_value_combiner": (
                "max(p_cbase,p_dig,p_mutsig)" if enabled else not_applicable
            ),
            "sign_discordance": (
                "retain-max-p-direction-not-unanimous" if enabled else not_applicable
            ),
            "valid_component_statuses": (
                ["valid-profile-lrt", "valid-degenerate-null-p-one"] if enabled else []
            ),
        },
        "direction_annotation": {
            "consensus_rule": (
                "unanimous-me-or-co-else-not-unanimous" if enabled else not_applicable
            ),
            "directional_fdr_control": False,
            "provider_rule": "rho-negative-me-positive-co-zero-neutral",
            "reporting_layer": "descriptive-post-rejection",
            "undefined_rho_rule": "unavailable",
        },
        "multiplicity": {
            "descriptive_methods": ["by", "bh"],
            "descriptive_q_threshold": 0.05,
            "descriptive_reporting_layer": "descriptive",
            "primary_method": "by",
            "primary_q_threshold": 0.01,
            "primary_reporting_layer": ("confirmatory-conditional-on-valid-marginals"),
            "sensitivity_method": "bh",
            "sensitivity_q_threshold": 0.01,
            "sensitivity_reporting_layer": "nominal-sensitivity",
            "threshold_comparison": "inclusive-less-than-or-equal",
        },
        "tested_family": {
            "epsilon_pretest_filter": "none",
            "family": "one-complete-within-cohort-tested-pair-family",
            "feature_ranking": "descending-total-eligible-mutation-event-count",
            "marginal_effect_pretest_filter": "none",
            "pair_construction": "all-unordered-pairs-of-ordered-feature-axis",
            "provider_support": "shared-native-cbase-dig-mutsig",
            "same_base_missense_nonsense": "exclude-before-fitting-and-testing",
            "tie_break": "canonical-count-matrix-column-order",
            "top_k": 500,
        },
    }


def _d6_narrow():
    return {
        "cell_count": 3,
        "claim_scope": "narrow-exact-family-stress-evidence-only",
        "compute_authority": {
            "authority_id": "signed-local-resource-plan-v1",
            "mode": "local-bounded",
        },
        "design_authority": {
            "design_id": "signed-three-cell-design-v1",
            "manifest_sha256": "d" * 64,
        },
        "path": NARROW_LOCAL,
        "pre_calibration_result_inspection": False,
        "replicates_per_cell": 300,
        "result_contingent_changes": False,
    }


def _d6_full():
    return {
        "cell_count": 48,
        "claim_scope": "finite-scenario-calibration-evidence-only",
        "compute_authority": {
            "authority_id": "approved-external-allocation-42",
            "mode": "external-approved",
        },
        "design_authority": {
            "design_id": "full-prespecified-k500-design-v1",
            "manifest_sha256": "e" * 64,
        },
        "path": "full-external",
        "pre_calibration_result_inspection": False,
        "replicates_per_cell": 1000,
        "result_contingent_changes": False,
    }


def _d6_none():
    return {
        "cell_count": 0,
        "claim_scope": "no-calibration-claims",
        "compute_authority": None,
        "design_authority": None,
        "path": NO_EXTENSION,
        "pre_calibration_result_inspection": False,
        "replicates_per_cell": 0,
        "result_contingent_changes": False,
    }


def _payloads():
    return {"D3": _d3(), "D4": _d4(), "D5": _d5(), "D6": _d6_narrow()}


def _envelope(decision_id, payload):
    return {
        "contract": CONTRACTS[decision_id],
        "decision_id": decision_id,
        "payload": payload,
        "schema": MACHINE_DECISION_SCHEMA,
    }


def _approval(  # noqa: PLR0913
    payloads=None,
    *,
    raw_artifacts=None,
    dispositions=None,
    allowed_stages=(FIT_SEALED_TCGA_K500_STAGE,),
    schema=APPROVAL_SCHEMA,
    receipt_changes=None,
    decision_id_changes=None,
):
    payloads = _payloads() if payloads is None else payloads
    raw_artifacts = {} if raw_artifacts is None else raw_artifacts
    dispositions = {} if dispositions is None else dispositions
    receipt_changes = {} if receipt_changes is None else receipt_changes
    decision_id_changes = {} if decision_id_changes is None else decision_id_changes
    decisions = {}
    digests = {}
    for index, decision_id in enumerate(("D3", "D4", "D5", "D6"), start=3):
        content = raw_artifacts.get(
            decision_id,
            _canonical(_envelope(decision_id, payloads[decision_id])),
        )
        receipt = {
            "path": f"authority/{decision_id}.json",
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "content": content,
        }
        receipt.update(receipt_changes.get(decision_id, {}))
        decisions[decision_id] = DecisionApproval(
            decision_id=decision_id_changes.get(decision_id, decision_id),
            disposition=dispositions.get(decision_id, "go"),
            exact_resolution=f"signed {decision_id} resolution",
            canonical_artifact=ArtifactReceipt(**receipt),
            execution_owner="Ahmed Shuaibi",
            claim_owner="Benjamin J. Raphael",
            rerun_or_reuse_consequence="fail closed",
            permitted_claims=("bounded claim",),
            forbidden_claims=("unbounded claim",),
            attestations=(),
            recorded_by="Uthsav Chitra",
            allowed_stages=(FIT_SEALED_TCGA_K500_STAGE,),
            manifest_allowed_stages=(FIT_SEALED_TCGA_K500_STAGE,),
            manifest_stage_bindings=MappingProxyType(
                {
                    FIT_SEALED_TCGA_K500_STAGE: MappingProxyType(
                        {
                            "canonical_input_manifest_sha256": "c" * 64,
                            "provider_input_manifest_sha256": "p" * 64,
                        },
                    ),
                },
            ),
            decision_authority_sha256=f"{index}" * 64,
        )
        digests[decision_id] = f"{index}" * 64
    source = ArtifactReceipt(
        path="source.txt",
        sha256="a" * 64,
        size_bytes=1,
        content=b"x",
    )
    return RevisionApproval(
        schema=schema,
        source_notice=SourceNotice("signed-document", "source", source),
        allowed_stages=allowed_stages,
        stage_bindings=MappingProxyType(
            {
                FIT_SEALED_TCGA_K500_STAGE: MappingProxyType(
                    {
                        "canonical_input_manifest_sha256": "c" * 64,
                        "provider_input_manifest_sha256": "p" * 64,
                    },
                ),
            },
        ),
        decisions=MappingProxyType(decisions),
        manifest_sha256="b" * 64,
        decision_digests=MappingProxyType(digests),
    )


def _validate(approval):
    return validate_revision_fit_policy(
        approval,
        expected_d4_implementation=EXPECTED_D4_IMPLEMENTATION,
        expected_tested_family=EXPECTED_TESTED_FAMILY,
    )


def _changed(decision_id, *path, value):
    payloads = copy.deepcopy(_payloads())
    target = payloads[decision_id]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    return _approval(payloads)


def test_valid_policy_returns_frozen_typed_payloads_and_exact_receipts():
    approval = _approval()

    policy = _validate(approval)

    assert policy.d3.primary_provider == "cbase"
    assert policy.d3.sensitivity_providers == ("dig", "mutsig")
    assert policy.d4.lrt_contract == LRT_CONTRACT
    assert policy.d4.numerical_implementation == (
        EXPECTED_D4_IMPLEMENTATION.numerical_implementation
    )
    assert policy.d4.reference.degrees_of_freedom == 1
    assert policy.d4.failure_handling.fit == TASK_ABORT_NO_PUBLISHED_ROW
    assert policy.d4.undefined_rho_degenerate_null_handling == (
        UNDEFINED_RHO_DEGENERATE_NULL_HANDLING
    )
    assert policy.d5.conjunction.mode == MAX_P_IUT
    assert policy.d5.conjunction.component_order == ("cbase", "dig", "mutsig")
    assert (
        policy.d5.conjunction.valid_component_statuses
        == VALID_CONJUNCTION_COMPONENT_STATUSES
    )
    assert policy.d5.conjunction.p_value_combiner == CONJUNCTION_P_VALUE_COMBINER
    assert policy.d5.conjunction.direction_affects_p_or_q is False
    assert policy.d5.conjunction.effect_unidentifiable == (EFFECT_UNIDENTIFIABLE_POLICY)
    assert policy.d5.direction_annotation.consensus_rule == DIRECTION_CONSENSUS_RULE
    assert policy.d5.direction_annotation.directional_fdr_control is False
    assert policy.d5.component_failure_semantics == COMPONENT_FAILURE_SEMANTICS
    assert policy.d5.tested_family == EXPECTED_TESTED_FAMILY
    assert policy.d5.multiplicity.primary_q_threshold == 0.01
    assert policy.d5.multiplicity.descriptive_methods == ("by", "bh")
    assert policy.d5.multiplicity.descriptive_q_threshold == 0.05
    assert policy.d6.path == NARROW_LOCAL
    assert tuple(policy.receipts) == ("D3", "D4", "D5", "D6")
    d3_receipt = policy.receipts["D3"]
    assert d3_receipt.decision_digest == "3" * 64
    assert d3_receipt.canonical_artifact_sha256 == (
        approval.decisions["D3"].canonical_artifact.sha256
    )
    assert (
        d3_receipt.payload_sha256
        == hashlib.sha256(
            _canonical(_d3(), newline=False),
        ).hexdigest()
    )
    assert d3_receipt.payload["sensitivity_providers"] == ("dig", "mutsig")
    with pytest.raises(TypeError):
        policy.receipts["D7"] = d3_receipt
    with pytest.raises(TypeError):
        d3_receipt.payload["primary_provider"] = "dig"
    with pytest.raises(FrozenInstanceError):
        policy.d3.primary_provider = "dig"


@pytest.mark.parametrize("primary", ["dig", "mutsig"])
def test_d3_rejects_non_cbase_primary(
    primary,
):
    payloads = _payloads()
    payloads["D3"]["primary_provider"] = primary
    sensitivities = sorted({"cbase", "dig", "mutsig"}.difference({primary}))
    payloads["D3"]["sensitivity_providers"] = sensitivities
    payloads["D5"] = _d5(component_order=[primary, *sensitivities])

    with pytest.raises(RevisionFitPolicyError, match="primary_provider"):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    "sensitivities",
    [
        ["mutsig", "dig"],
        ["dig"],
        ["dig", "mutsig", "cbase"],
        ["dig", "dig"],
    ],
)
def test_d3_rejects_reordered_missing_extra_or_duplicate_sensitivities(
    sensitivities,
):
    payloads = _payloads()
    payloads["D3"]["sensitivity_providers"] = sensitivities
    payloads["D5"] = _d5(component_order=["cbase", *sensitivities])

    with pytest.raises(RevisionFitPolicyError, match=r"exactly.*frozen order"):
        _validate(_approval(payloads))


@pytest.mark.parametrize("d6", [_d6_full(), _d6_narrow(), _d6_none()])
def test_d6_accepts_exact_three_prospective_paths(d6):
    payloads = _payloads()
    payloads["D6"] = d6
    if d6["path"] == NO_EXTENSION:
        payloads["D3"] = _d3(conjunction_role="omitted")
        payloads["D5"] = _d5(mode=NO_CONJUNCTION)

    policy = _validate(_approval(payloads))

    assert policy.d6.path == d6["path"]
    assert policy.d6.replicates_per_cell == d6["replicates_per_cell"]


def test_validation_consumes_immutable_artifact_bytes_not_artifact_path():
    approval = _approval()
    assert approval.decisions["D3"].canonical_artifact.path.startswith("authority/")

    policy = _validate(approval)

    assert policy.receipts["D3"].canonical_artifact_path == "authority/D3.json"


@pytest.mark.parametrize(
    "approval",
    [
        _approval(schema="dialect-revision-coauthor-approval-v1"),
        _approval(allowed_stages=()),
        _approval(dispositions={"D4": "deferred"}),
        _approval(decision_id_changes={"D5": "D4"}),
    ],
)
def test_authority_must_explicitly_be_v2_fit_go(approval):
    with pytest.raises(RevisionFitPolicyError):
        _validate(approval)


def test_missing_decision_or_digest_fails_closed():
    approval = _approval()
    decisions = dict(approval.decisions)
    decisions.pop("D6")
    missing_decision = RevisionApproval(
        schema=approval.schema,
        source_notice=approval.source_notice,
        allowed_stages=approval.allowed_stages,
        stage_bindings=approval.stage_bindings,
        decisions=MappingProxyType(decisions),
        manifest_sha256=approval.manifest_sha256,
        decision_digests=approval.decision_digests,
    )
    with pytest.raises(RevisionFitPolicyError, match="missing explicit D6"):
        _validate(missing_decision)

    digests = dict(approval.decision_digests)
    digests.pop("D5")
    missing_digest = RevisionApproval(
        schema=approval.schema,
        source_notice=approval.source_notice,
        allowed_stages=approval.allowed_stages,
        stage_bindings=approval.stage_bindings,
        decisions=approval.decisions,
        manifest_sha256=approval.manifest_sha256,
        decision_digests=MappingProxyType(digests),
    )
    with pytest.raises(RevisionFitPolicyError, match="missing explicit D5"):
        _validate(missing_digest)


@pytest.mark.parametrize(
    "receipt_change",
    [
        {"sha256": "0" * 64},
        {"size_bytes": 1},
        {"content": bytearray(b"not immutable")},
    ],
)
def test_artifact_receipt_is_recomputed(receipt_change):
    approval = _approval(receipt_changes={"D4": receipt_change})

    with pytest.raises(RevisionFitPolicyError, match="artifact"):
        _validate(approval)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "wrong-schema"),
        ("decision_id", "D5"),
        ("contract", "free-form-contract"),
    ],
)
def test_envelope_schema_id_and_contract_are_exact(field, value):
    envelope = _envelope("D4", _d4())
    envelope[field] = value
    approval = _approval(raw_artifacts={"D4": _canonical(envelope)})

    with pytest.raises(RevisionFitPolicyError):
        _validate(approval)


def test_d5_v2_contract_remains_invalid_after_family_binding_upgrade():
    envelope = _envelope("D5", _d5())
    envelope["contract"] = "conjunction-multiplicity-policy-v2"

    with pytest.raises(RevisionFitPolicyError, match="family-policy-v3"):
        _validate(_approval(raw_artifacts={"D5": _canonical(envelope)}))


def test_d4_v2_contract_remains_invalid_after_marginal_fit_binding_upgrade():
    envelope = _envelope("D4", _d4())
    envelope["contract"] = "profile-lrt-pvalue-policy-v2"

    with pytest.raises(RevisionFitPolicyError, match="policy-v3"):
        _validate(_approval(raw_artifacts={"D4": _canonical(envelope)}))


def test_envelope_and_payload_reject_unknown_or_missing_fields():
    envelope = _envelope("D3", _d3())
    envelope["comment"] = "not signed by the contract"
    with pytest.raises(RevisionFitPolicyError, match="unknown"):
        _validate(_approval(raw_artifacts={"D3": _canonical(envelope)}))

    payloads = _payloads()
    del payloads["D5"]["tested_family"]
    with pytest.raises(RevisionFitPolicyError, match="missing"):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 499),
        ("top_k", True),
        ("feature_ranking", "descending-distinct-mutated-sample-count"),
        ("tie_break", "lexical-feature-order"),
        ("provider_support", "union-cbase-dig-mutsig"),
        ("pair_construction", "significant-pairs-only"),
        ("same_base_missense_nonsense", "retain"),
        ("epsilon_pretest_filter", "epsilon-positive-only"),
        ("marginal_effect_pretest_filter", "nonzero-marginals-only"),
        ("family", "one-family-per-direction"),
    ],
)
def test_d5_tested_family_rejects_every_semantic_drift(field, value):
    with pytest.raises(RevisionFitPolicyError, match=field):
        _validate(_changed("D5", "tested_family", field, value=value))


@pytest.mark.parametrize(
    "field",
    [
        "top_k",
        "feature_ranking",
        "tie_break",
        "provider_support",
        "pair_construction",
        "same_base_missense_nonsense",
        "epsilon_pretest_filter",
        "marginal_effect_pretest_filter",
        "family",
    ],
)
def test_d5_tested_family_requires_every_exact_key(field):
    payloads = _payloads()
    del payloads["D5"]["tested_family"][field]

    with pytest.raises(RevisionFitPolicyError, match=r"missing.*" + field):
        _validate(_approval(payloads))


def test_d5_tested_family_and_multiplicity_reject_unknown_selection_fields():
    payloads = _payloads()
    payloads["D5"]["tested_family"]["post_hoc_filter"] = "none"
    with pytest.raises(RevisionFitPolicyError, match="unknown"):
        _validate(_approval(payloads))

    payloads = _payloads()
    payloads["D5"]["multiplicity"]["selected_q_method"] = "by"
    with pytest.raises(RevisionFitPolicyError, match="unknown"):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw: raw[:-1],
        lambda raw: raw + b"\n",
        lambda raw: b" " + raw,
        lambda raw: json.dumps(json.loads(raw), indent=2).encode() + b"\n",
    ],
)
def test_artifacts_require_exact_canonical_json_and_one_lf(transform):
    raw = _canonical(_envelope("D3", _d3()))

    with pytest.raises(RevisionFitPolicyError, match="canonical JSON"):
        _validate(_approval(raw_artifacts={"D3": transform(raw)}))


def test_duplicate_keys_are_rejected_before_semantic_parsing():
    envelope = _envelope("D3", _d3())
    raw = _canonical(envelope)
    duplicate = raw.replace(
        b'"contract":"bmr-provider-hierarchy-v1",',
        b'"contract":"bmr-provider-hierarchy-v1",'
        b'"contract":"bmr-provider-hierarchy-v1",',
        1,
    )

    with pytest.raises(RevisionFitPolicyError, match="Duplicate JSON key"):
        _validate(_approval(raw_artifacts={"D3": duplicate}))


@pytest.mark.parametrize("nonfinite", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_nonfinite_constants_and_overflowing_numbers_are_rejected(nonfinite):
    raw = _canonical(_envelope("D5", _d5()))
    raw = raw.replace(b"0.05", nonfinite, 1)

    with pytest.raises(RevisionFitPolicyError, match="Non-finite"):
        _validate(_approval(raw_artifacts={"D5": raw}))


def test_invalid_utf8_and_escaped_surrogates_are_rejected():
    invalid_utf8 = _canonical(_envelope("D3", _d3())).replace(
        b"continuity",
        b"continuity\xff",
    )
    with pytest.raises(RevisionFitPolicyError, match="valid UTF-8"):
        _validate(_approval(raw_artifacts={"D3": invalid_utf8}))

    surrogate = _canonical(_envelope("D3", _d3())).replace(
        b"continuity",
        b"continuity\\ud800",
    )
    with pytest.raises(RevisionFitPolicyError, match="Unicode surrogate"):
        _validate(_approval(raw_artifacts={"D3": surrogate}))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("primary_provider",), "other", "primary_provider"),
        (
            ("sensitivity_providers",),
            ["dig", "dig"],
            "exactly.*frozen order",
        ),
        (
            ("sensitivity_providers",),
            ["cbase", "dig"],
            "exactly.*frozen order",
        ),
        (("burden_dependent_switching",), True, "forbids burden-dependent"),
        (("burden_dependent_switching",), 0, "JSON boolean"),
        (("rationale",), " ", "nonblank exact string"),
        (("all_three_conjunction_role",), "primary", "must be one of"),
    ],
)
def test_d3_rejects_uncontrolled_hierarchy(path, value, message):
    with pytest.raises(RevisionFitPolicyError, match=message):
        _validate(_changed("D3", *path, value=value))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("test_direction",), "one-sided-co"),
        (("reference", "family"), "normal"),
        (("reference", "degrees_of_freedom"), 2),
        (("reference", "degrees_of_freedom"), True),
        (("reference", "tail"), "lower"),
        (("statistic_transform",), "2*(alt-null)"),
        (("undefined_rho_degenerate_null_handling",), "ignore"),
        (("failure_handling", "fit"), "set-p-one"),
        (("failure_handling", "observation_support"), "drop-row"),
        (("failure_handling", "convergence"), "publish-partial-row"),
        (("validity_evidence", "standard"), "asymptotic-hope"),
        (("validity_evidence", "gate"), "warn-only"),
        (("validity_evidence", "covers"), list(reversed(LRT_VALIDITY_COVERAGE))),
    ],
)
def test_d4_rejects_uncontrolled_reference_failure_or_evidence(path, value):
    with pytest.raises(RevisionFitPolicyError):
        _validate(_changed("D4", *path, value=value))


def test_d4_binds_expected_lrt_contract_supplied_by_caller():
    approval = _approval()

    with pytest.raises(RevisionFitPolicyError, match="required by the caller"):
        validate_revision_fit_policy(
            approval,
            expected_d4_implementation=replace(
                EXPECTED_D4_IMPLEMENTATION,
                lrt_contract="different-implementation-v1",
            ),
            expected_tested_family=EXPECTED_TESTED_FAMILY,
        )
    with pytest.raises(RevisionFitPolicyError, match="nonblank exact string"):
        validate_revision_fit_policy(
            approval,
            expected_d4_implementation=replace(
                EXPECTED_D4_IMPLEMENTATION,
                lrt_contract=" ",
            ),
            expected_tested_family=EXPECTED_TESTED_FAMILY,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("marginal_fit_contract",), "different-marginal-fit-v1"),
        (("marginal_fit_max_iterations",), 999),
        (("marginal_fit_total_kkt_tolerance",), 1e-7),
        (("marginal_fit_bracket_width_tolerance",), 1e-11),
        (("marginal_fit_fixed_point_tolerance",), 1e-7),
        (("marginal_fit_flat_likelihood_tie_break",), "pi-one"),
        (("pair_fit_contract",), "different-fit-v1"),
        (("pair_fit_max_iterations",), 999),
        (("pair_fit_total_kkt_tolerance",), 1e-7),
        (("pair_simplex_tolerance",), 1e-11),
        (("lrt_nestedness_tolerance",), 1e-7),
        (("rho_contract",), "different-rho-v1"),
        (("undefined_rho_lrt_tolerance",), 1e-7),
        (("log_odds_ratio_contract",), "different-lor-v1"),
        (
            ("effect_identifiability", "contract"),
            "different-identifiability-v1",
        ),
        (("effect_identifiability", "relative_tolerance"), 1e-11),
        (
            ("effect_identifiability", "identified_status"),
            "rank-deficient",
        ),
        (
            ("effect_identifiability", "status_vocabulary"),
            [
                EFFECT_IDENTIFIED_STATUS,
                EFFECT_UNDERFLOW_STATUS,
                EFFECT_RANK_DEFICIENT_STATUS,
            ],
        ),
        (
            ("effect_identifiability", "nonidentified_statuses"),
            [EFFECT_UNDERFLOW_STATUS, EFFECT_RANK_DEFICIENT_STATUS],
        ),
        (
            ("effect_identifiability", "nonidentified_effect_blank_fields"),
            list(reversed(NONIDENTIFIED_EFFECT_BLANK_FIELDS)),
        ),
    ],
)
def test_d4_signed_numerical_implementation_must_match_runner_exactly(path, value):
    payloads = _payloads()
    target = payloads["D4"]["numerical_implementation"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(RevisionFitPolicyError, match="must equal"):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    "field",
    [
        "marginal_fit_contract",
        "marginal_fit_max_iterations",
        "marginal_fit_total_kkt_tolerance",
        "marginal_fit_bracket_width_tolerance",
        "marginal_fit_fixed_point_tolerance",
        "marginal_fit_flat_likelihood_tie_break",
    ],
)
def test_d4_marginal_fit_contract_requires_every_exact_key(field):
    payloads = _payloads()
    del payloads["D4"]["numerical_implementation"][field]

    with pytest.raises(RevisionFitPolicyError, match=r"missing.*" + field):
        _validate(_approval(payloads))


def test_d4_marginal_fit_contract_rejects_unknown_key():
    payloads = _payloads()
    payloads["D4"]["numerical_implementation"]["marginal_fit_seed"] = 0

    with pytest.raises(RevisionFitPolicyError, match="unknown"):
        _validate(_approval(payloads))


def test_d4_rejects_wrong_expected_contract_type_and_incoherent_statuses():
    approval = _approval()
    with pytest.raises(RevisionFitPolicyError, match="D4ImplementationContract"):
        validate_revision_fit_policy(
            approval,
            expected_d4_implementation=LRT_CONTRACT,
            expected_tested_family=EXPECTED_TESTED_FAMILY,
        )

    numerical = EXPECTED_D4_IMPLEMENTATION.numerical_implementation
    effect = replace(
        numerical.effect_identifiability,
        nonidentified_statuses=(
            EFFECT_UNDERFLOW_STATUS,
            EFFECT_RANK_DEFICIENT_STATUS,
        ),
    )
    with pytest.raises(RevisionFitPolicyError, match="status_vocabulary"):
        validate_revision_fit_policy(
            approval,
            expected_d4_implementation=replace(
                EXPECTED_D4_IMPLEMENTATION,
                numerical_implementation=replace(
                    numerical,
                    effect_identifiability=effect,
                ),
            ),
            expected_tested_family=EXPECTED_TESTED_FAMILY,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("marginal_fit_contract", " ", "nonblank exact string"),
        ("marginal_fit_max_iterations", 0, "must be positive"),
        ("marginal_fit_max_iterations", True, "nonnegative JSON integer"),
        ("marginal_fit_total_kkt_tolerance", 0.0, "positive finite"),
        ("marginal_fit_bracket_width_tolerance", np.inf, "positive finite"),
        ("marginal_fit_fixed_point_tolerance", np.nan, "positive finite"),
        ("marginal_fit_flat_likelihood_tie_break", " ", "nonblank exact string"),
    ],
)
def test_d4_rejects_invalid_expected_marginal_fit_contract(field, value, message):
    numerical = replace(
        EXPECTED_D4_IMPLEMENTATION.numerical_implementation,
        **{field: value},
    )

    with pytest.raises(RevisionFitPolicyError, match=message):
        validate_revision_fit_policy(
            _approval(),
            expected_d4_implementation=replace(
                EXPECTED_D4_IMPLEMENTATION,
                numerical_implementation=numerical,
            ),
            expected_tested_family=EXPECTED_TESTED_FAMILY,
        )


def test_d5_rejects_wrong_or_drifting_caller_tested_family_contract():
    approval = _approval()
    with pytest.raises(RevisionFitPolicyError, match="TestedFamilyPolicy"):
        validate_revision_fit_policy(
            approval,
            expected_d4_implementation=EXPECTED_D4_IMPLEMENTATION,
            expected_tested_family={"top_k": 500},
        )

    with pytest.raises(RevisionFitPolicyError, match=r"expected_tested_family\.top_k"):
        validate_revision_fit_policy(
            approval,
            expected_d4_implementation=EXPECTED_D4_IMPLEMENTATION,
            expected_tested_family=replace(EXPECTED_TESTED_FAMILY, top_k=499),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("conjunction", "invalid_component"), "drop-component"),
        (("conjunction", "missing_component"), "drop-pair"),
        (("conjunction", "component_order"), ["cbase", "dig", "dig"]),
        (
            ("conjunction", "valid_component_statuses"),
            ["valid-degenerate-null-p-one", "valid-profile-lrt"],
        ),
        (("conjunction", "p_value_combiner"), "minimum"),
        (("conjunction", "sign_discordance"), "set-conjunction-p-to-one"),
        (("conjunction", "effect_unidentifiable"), "set-p-one"),
        (("conjunction", "direction_affects_p_or_q"), True),
        (("component_failure_semantics",), "retain-with-p-one"),
        (("direction_annotation", "provider_rule"), "rho-sign-majority"),
        (("direction_annotation", "undefined_rho_rule"), "neutral"),
        (("direction_annotation", "consensus_rule"), "majority-me-or-co"),
        (("direction_annotation", "reporting_layer"), "confirmatory"),
        (("direction_annotation", "directional_fdr_control"), True),
        (("tested_family", "family"), "significant-pairs-only"),
        (("multiplicity", "primary_method"), "bh"),
        (("multiplicity", "sensitivity_method"), "by"),
        (("multiplicity", "primary_q_threshold"), 0.0100001),
        (("multiplicity", "primary_q_threshold"), True),
        (("multiplicity", "descriptive_q_threshold"), 0.1),
        (("multiplicity", "descriptive_methods"), ["bh", "by"]),
        (("multiplicity", "descriptive_methods"), ["by"]),
        (("multiplicity", "descriptive_methods"), ["by", "bh", "bonferroni"]),
        (("multiplicity", "threshold_comparison"), "strict-less-than"),
    ],
)
def test_d5_rejects_partial_families_ad_hoc_multiplicity_or_direction(path, value):
    with pytest.raises(RevisionFitPolicyError):
        _validate(_changed("D5", *path, value=value))


def test_d5_no_conjunction_requires_explicit_not_applicable_component_policies():
    payloads = _payloads()
    payloads["D3"] = _d3(conjunction_role="omitted")
    payloads["D5"] = _d5(mode=NO_CONJUNCTION)
    payloads["D5"]["conjunction"]["missing_component"] = "retain-with-p-one"

    with pytest.raises(RevisionFitPolicyError, match="not-applicable-no-conjunction"):
        _validate(_approval(payloads))


def test_d5_no_conjunction_has_no_smuggled_components_or_consensus_rule():
    payloads = _payloads()
    payloads["D3"] = _d3(conjunction_role="omitted")
    payloads["D5"] = _d5(mode=NO_CONJUNCTION)

    policy = _validate(_approval(payloads))

    assert policy.d5.conjunction.component_order == ()
    assert policy.d5.conjunction.valid_component_statuses == ()
    assert policy.d5.conjunction.p_value_combiner == ("not-applicable-no-conjunction")
    assert policy.d5.direction_annotation.consensus_rule == (
        "not-applicable-no-conjunction"
    )
    assert policy.d5.component_failure_semantics == ("not-applicable-no-conjunction")


def test_d3_and_d5_provider_orders_must_agree():
    payloads = _payloads()
    payloads["D5"] = _d5(component_order=["cbase", "mutsig", "dig"])

    with pytest.raises(RevisionFitPolicyError, match="provider hierarchy"):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    ("d3_role", "d5_mode"),
    [("secondary", NO_CONJUNCTION), ("omitted", MAX_P_IUT)],
)
def test_d3_and_d5_conjunction_decisions_must_agree(d3_role, d5_mode):
    payloads = _payloads()
    payloads["D3"] = _d3(conjunction_role=d3_role)
    payloads["D5"] = _d5(mode=d5_mode)

    with pytest.raises(RevisionFitPolicyError, match="inconsistent"):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("cell_count",), 0),
        (("cell_count",), True),
        (("replicates_per_cell",), 999),
        (("compute_authority", "mode"), "local-bounded"),
        (("compute_authority", "authority_id"), " "),
        (("design_authority", "manifest_sha256"), "A" * 64),
        (("claim_scope",), "formal-uniform-fdr-proof"),
        (("result_contingent_changes",), True),
        (("pre_calibration_result_inspection",), True),
    ],
)
def test_d6_full_external_requires_declared_design_compute_and_result_blindness(
    path,
    value,
):
    payloads = _payloads()
    payloads["D6"] = _d6_full()
    target = payloads["D6"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(RevisionFitPolicyError):
        _validate(_approval(payloads))


@pytest.mark.parametrize(
    ("field", "value"),
    [("cell_count", 2), ("replicates_per_cell", 301)],
)
def test_d6_narrow_local_is_exactly_three_by_three_hundred(field, value):
    with pytest.raises(RevisionFitPolicyError, match="three cells by 300"):
        _validate(_changed("D6", field, value=value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_count", 1),
        ("replicates_per_cell", 1),
        (
            "design_authority",
            {"design_id": "smuggled", "manifest_sha256": "1" * 64},
        ),
        (
            "compute_authority",
            {"authority_id": "smuggled", "mode": "local-bounded"},
        ),
        ("claim_scope", "narrow-exact-family-stress-evidence-only"),
    ],
)
def test_d6_no_extension_has_zero_work_and_no_calibration_claim(field, value):
    payloads = _payloads()
    payloads["D3"] = _d3(conjunction_role="omitted")
    payloads["D5"] = _d5(mode=NO_CONJUNCTION)
    payloads["D6"] = _d6_none()
    payloads["D6"][field] = value

    with pytest.raises(RevisionFitPolicyError):
        _validate(_approval(payloads))

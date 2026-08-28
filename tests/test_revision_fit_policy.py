import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from types import MappingProxyType

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
    D3_CONTRACT,
    D4_CONTRACT,
    D5_CONTRACT,
    D6_CONTRACT,
    LRT_STATISTIC_TRANSFORM,
    LRT_VALIDITY_COVERAGE,
    MACHINE_DECISION_SCHEMA,
    MAX_P_IUT,
    NARROW_LOCAL,
    NO_CONJUNCTION,
    NO_EXTENSION,
    RevisionFitPolicyError,
    validate_revision_fit_policy,
)

LRT_CONTRACT = "driver-independence-constrained-mle-v1"
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
        "boundary_handling": "assign-p-one-with-explicit-boundary-status",
        "failure_semantics": "assign-p-one-with-explicit-failure-status",
        "lrt_contract": LRT_CONTRACT,
        "reference": {
            "degrees_of_freedom": 1,
            "family": "chi-square",
            "tail": "upper-survival",
        },
        "statistic_transform": LRT_STATISTIC_TRANSFORM,
        "test_direction": "nondirectional-two-sided-dependence",
        "validity_evidence": {
            "covers": list(LRT_VALIDITY_COVERAGE),
            "gate": "block-inferential-use-if-absent-invalid-or-inconclusive",
            "standard": (
                "finite-sample-super-uniformity-under-frozen-analysis-pipeline"
            ),
        },
    }


def _d5(*, mode=MAX_P_IUT):
    component_policy = (
        "set-conjunction-p-to-one"
        if mode == MAX_P_IUT
        else "not-applicable-no-conjunction"
    )
    return {
        "conjunction": {
            "invalid_component": component_policy,
            "missing_component": component_policy,
            "mode": mode,
            "sign_discordance": component_policy,
        },
        "directional_fdr_control": False,
        "failed_hypothesis": "retain-with-p-one",
        "family": "one-complete-within-cohort-tested-pair-family",
        "multiplicity": {
            "descriptive_q_threshold": 0.05,
            "descriptive_reporting_layer": "descriptive",
            "primary_method": "by",
            "primary_q_threshold": 0.01,
            "primary_reporting_layer": (
                "confirmatory-conditional-on-valid-marginals"
            ),
            "sensitivity_method": "bh",
            "sensitivity_q_threshold": 0.01,
            "sensitivity_reporting_layer": "nominal-sensitivity",
            "threshold_comparison": "inclusive-less-than-or-equal",
        },
        "rho_role": "descriptive-post-rejection-annotation",
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
        expected_lrt_contract=LRT_CONTRACT,
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
    assert policy.d4.reference.degrees_of_freedom == 1
    assert policy.d5.conjunction.mode == MAX_P_IUT
    assert policy.d5.multiplicity.primary_q_threshold == 0.01
    assert policy.d6.path == NARROW_LOCAL
    assert tuple(policy.receipts) == ("D3", "D4", "D5", "D6")
    d3_receipt = policy.receipts["D3"]
    assert d3_receipt.decision_digest == "3" * 64
    assert d3_receipt.canonical_artifact_sha256 == (
        approval.decisions["D3"].canonical_artifact.sha256
    )
    assert d3_receipt.payload_sha256 == hashlib.sha256(
        _canonical(_d3(), newline=False),
    ).hexdigest()
    assert d3_receipt.payload["sensitivity_providers"] == ("dig", "mutsig")
    with pytest.raises(TypeError):
        policy.receipts["D7"] = d3_receipt
    with pytest.raises(TypeError):
        d3_receipt.payload["primary_provider"] = "dig"
    with pytest.raises(FrozenInstanceError):
        policy.d3.primary_provider = "dig"


@pytest.mark.parametrize(
    ("primary", "sensitivities"),
    [
        ("cbase", ["mutsig", "dig"]),
        ("dig", ["cbase", "mutsig"]),
        ("mutsig", ["dig", "cbase"]),
    ],
)
def test_d3_accepts_any_primary_and_preserves_signed_sensitivity_order(
    primary,
    sensitivities,
):
    payloads = _payloads()
    payloads["D3"]["primary_provider"] = primary
    payloads["D3"]["sensitivity_providers"] = sensitivities

    policy = _validate(_approval(payloads))

    assert policy.d3.primary_provider == primary
    assert policy.d3.sensitivity_providers == tuple(sensitivities)


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


def test_envelope_and_payload_reject_unknown_or_missing_fields():
    envelope = _envelope("D3", _d3())
    envelope["comment"] = "not signed by the contract"
    with pytest.raises(RevisionFitPolicyError, match="unknown"):
        _validate(_approval(raw_artifacts={"D3": _canonical(envelope)}))

    payloads = _payloads()
    del payloads["D5"]["family"]
    with pytest.raises(RevisionFitPolicyError, match="missing"):
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
        (("sensitivity_providers",), ["dig", "dig"], "each non-primary"),
        (("sensitivity_providers",), ["cbase", "dig"], "each non-primary"),
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
        (("boundary_handling",), "ignore"),
        (("failure_semantics",), "drop-row"),
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
            expected_lrt_contract="different-implementation-v1",
        )
    with pytest.raises(RevisionFitPolicyError, match="nonblank exact string"):
        validate_revision_fit_policy(approval, expected_lrt_contract=" ")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("conjunction", "invalid_component"), "drop-component"),
        (("conjunction", "missing_component"), "drop-pair"),
        (("conjunction", "sign_discordance"), "majority-vote"),
        (("family",), "significant-pairs-only"),
        (("failed_hypothesis",), "omit"),
        (("multiplicity", "primary_method"), "bh"),
        (("multiplicity", "sensitivity_method"), "by"),
        (("multiplicity", "primary_q_threshold"), 0.0100001),
        (("multiplicity", "primary_q_threshold"), True),
        (("multiplicity", "descriptive_q_threshold"), 0.1),
        (("multiplicity", "threshold_comparison"), "strict-less-than"),
        (("rho_role",), "directional-test"),
        (("directional_fdr_control",), True),
    ],
)
def test_d5_rejects_partial_families_ad_hoc_multiplicity_or_direction(path, value):
    with pytest.raises(RevisionFitPolicyError):
        _validate(_changed("D5", *path, value=value))


def test_d5_no_conjunction_requires_explicit_not_applicable_component_policies():
    payloads = _payloads()
    payloads["D3"] = _d3(conjunction_role="omitted")
    payloads["D5"] = _d5(mode=NO_CONJUNCTION)
    payloads["D5"]["conjunction"]["missing_component"] = (
        "set-conjunction-p-to-one"
    )

    with pytest.raises(RevisionFitPolicyError, match="not-applicable-no-conjunction"):
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

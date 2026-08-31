from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import gammaincc

from analysis import postprocess_tcga_revision_k500 as postprocess
from analysis import run_tcga_revision_k500 as runner
from dialect.data.revision_fit_policy import (
    COMPONENT_FAILURE_SEMANTICS,
    CONJUNCTION_P_VALUE_COMBINER,
    DESCRIPTIVE_METHODS,
    DESCRIPTIVE_REPORTING_LAYER,
    DIRECTION_CONSENSUS_RULE,
    DIRECTION_PROVIDER_RULE,
    DIRECTION_REPORTING_LAYER,
    EFFECT_UNIDENTIFIABLE_POLICY,
    INCLUSIVE_THRESHOLD,
    INVALID_CONJUNCTION_COMPONENT,
    MAX_P_IUT,
    MISSING_CONJUNCTION_COMPONENT,
    NO_PRETEST_FILTER,
    PRIMARY_REPORTING_LAYER,
    SENSITIVITY_REPORTING_LAYER,
    SIGN_DISCORDANCE_POLICY,
    TESTED_FAMILY_FEATURE_RANKING,
    TESTED_FAMILY_PAIR_CONSTRUCTION,
    TESTED_FAMILY_PROVIDER_SUPPORT,
    TESTED_FAMILY_SAME_BASE_POLICY,
    TESTED_FAMILY_TIE_BREAK,
    TESTED_FAMILY_TOP_K,
    UNDEFINED_RHO_DIRECTION_RULE,
    VALID_CONJUNCTION_COMPONENT_STATUSES,
    WITHIN_COHORT_FAMILY,
    CalibrationComputeAuthority,
    CalibrationDesignAuthority,
    CalibrationScopePolicy,
    ConjunctionMultiplicityPolicy,
    ConjunctionPolicy,
    DirectionAnnotationPolicy,
    FitPolicyReceipt,
    MultiplicityPolicy,
    ProviderHierarchyPolicy,
)
from dialect.data.revision_fit_policy import (
    TestedFamilyPolicy as RevisionTestedFamilyPolicy,
)
from dialect.models.interaction import (
    PAIR_EFFECT_IDENTIFIED_STATUS,
    PAIR_EFFECT_RANK_DEFICIENT_STATUS,
    PAIR_EFFECT_UNDERFLOW_STATUS,
    compute_marshall_olkin_rho,
)
from dialect.stats import revision_inference

_DIGEST = "a" * 64
_FEATURES = ("A_M", "B_M", "C_N")
_ME_TAUS = (0.6, 0.2, 0.2, 0.0)
_CO_TAUS = (0.7, 0.05, 0.05, 0.2)
_DEGENERATE_TAUS = (1.0, 0.0, 0.0, 0.0)
_CONTINGENCY = (70, 10, 15, 5)


def _policy() -> ConjunctionMultiplicityPolicy:
    return ConjunctionMultiplicityPolicy(
        conjunction=ConjunctionPolicy(
            mode=MAX_P_IUT,
            component_order=("cbase", "dig", "mutsig"),
            valid_component_statuses=VALID_CONJUNCTION_COMPONENT_STATUSES,
            p_value_combiner=CONJUNCTION_P_VALUE_COMBINER,
            invalid_component=INVALID_CONJUNCTION_COMPONENT,
            missing_component=MISSING_CONJUNCTION_COMPONENT,
            sign_discordance=SIGN_DISCORDANCE_POLICY,
            effect_unidentifiable=EFFECT_UNIDENTIFIABLE_POLICY,
            direction_affects_p_or_q=False,
        ),
        direction_annotation=DirectionAnnotationPolicy(
            provider_rule=DIRECTION_PROVIDER_RULE,
            undefined_rho_rule=UNDEFINED_RHO_DIRECTION_RULE,
            consensus_rule=DIRECTION_CONSENSUS_RULE,
            reporting_layer=DIRECTION_REPORTING_LAYER,
            directional_fdr_control=False,
        ),
        tested_family=RevisionTestedFamilyPolicy(
            top_k=TESTED_FAMILY_TOP_K,
            feature_ranking=TESTED_FAMILY_FEATURE_RANKING,
            tie_break=TESTED_FAMILY_TIE_BREAK,
            provider_support=TESTED_FAMILY_PROVIDER_SUPPORT,
            pair_construction=TESTED_FAMILY_PAIR_CONSTRUCTION,
            same_base_missense_nonsense=TESTED_FAMILY_SAME_BASE_POLICY,
            epsilon_pretest_filter=NO_PRETEST_FILTER,
            marginal_effect_pretest_filter=NO_PRETEST_FILTER,
            family=WITHIN_COHORT_FAMILY,
        ),
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
        component_failure_semantics=COMPONENT_FAILURE_SEMANTICS,
    )


def _axis(features=_FEATURES):
    return postprocess.build_frozen_family_axis(
        "TEST",
        features,
        cohort_contract_sha256=_DIGEST,
    )


def _provider_hierarchy():
    return ProviderHierarchyPolicy(
        primary_provider="cbase",
        sensitivity_providers=("dig", "mutsig"),
        all_three_conjunction_role="secondary",
        burden_dependent_switching=False,
        rationale="synthetic authority fixture",
    )


def _fit_policy_stub(design_sha256):
    d6 = CalibrationScopePolicy(
        path="narrow-local",
        cell_count=3,
        replicates_per_cell=300,
        design_authority=CalibrationDesignAuthority(
            design_id="signed-three-cell-design-v1",
            manifest_sha256=design_sha256,
        ),
        compute_authority=CalibrationComputeAuthority(
            mode="local-bounded",
            authority_id="signed-local-resource-plan-v1",
        ),
        claim_scope="narrow-exact-family-stress-evidence-only",
        result_contingent_changes=False,
        pre_calibration_result_inspection=False,
    )
    d6_receipt = FitPolicyReceipt(
        decision_id="D6",
        contract="calibration-scope-policy-v1",
        decision_digest="6" * 64,
        canonical_artifact_path="d6.json",
        canonical_artifact_sha256="7" * 64,
        canonical_artifact_size_bytes=100,
        payload_sha256="8" * 64,
        payload={},
    )
    return SimpleNamespace(
        d3=_provider_hierarchy(),
        d5=_policy(),
        d6=d6,
        receipts={"D6": d6_receipt},
    )


def _config(tmp_path, *, evidence_sha256, design_path):
    root = tmp_path.resolve()
    return postprocess.ProductionPostprocessConfig(
        run_output_root=root / "run",
        canonical_input_root=root / "canonical",
        provider_input_root=root / "provider",
        input_approval_manifest=root / "input-approval.json",
        fit_approval_manifest=root / "fit-approval.json",
        inspect_approval_manifest=root / "inspect-approval.json",
        calibration_approval_manifest=root / "calibration-approval.json",
        calibration_evidence_artifact=root / "evidence.json",
        calibration_design_manifest=design_path,
        marginal_validity_status="certified",
        expected_sealed_completion_sha256="1" * 64,
        expected_canonical_input_sha256="2" * 64,
        expected_provider_input_manifest_sha256="3" * 64,
        expected_input_approval_sha256="4" * 64,
        expected_fit_approval_sha256="5" * 64,
        expected_inspect_approval_sha256="a" * 64,
        expected_calibration_approval_sha256="6" * 64,
        expected_calibration_evidence_sha256=evidence_sha256,
    )


def _calibration_evidence_payload(config, fit_policy, *, status="certified"):
    decision_ids = tuple(f"D{index}" for index in range(1, 7))
    return {
        "analysis": "tcga-revision-k500",
        "bmrs": list(postprocess.BMRS),
        "calibration_authority": {
            "approval_manifest_sha256": (config.expected_calibration_approval_sha256),
            "approval_schema": "dialect-revision-coauthor-approval-v6",
            "authorized_stage": "calibration",
            "calibration_decision_digests": dict.fromkeys(decision_ids, "9" * 64),
            "compute_authority": asdict(fit_policy.d6.compute_authority),
            "design_authority": asdict(fit_policy.d6.design_authority),
            "fit_decision_digests": dict.fromkeys(decision_ids, "6" * 64),
            "fit_d6_artifact_sha256": "7" * 64,
            "fit_d6_decision_digest": "6" * 64,
            "fit_d6_payload_sha256": "8" * 64,
            "path": fit_policy.d6.path,
            "claim_scope": fit_policy.d6.claim_scope,
        },
        "cohorts": list(postprocess.TCGA_COHORTS),
        "contract": postprocess.MARGINAL_VALIDITY_EVIDENCE_CONTRACT,
        "correction_policy": {
            "computed_methods": ["by", "bh"],
            "correction_selection_affected": False,
            "q_values_affected": False,
        },
        "marginal_validity": {
            "evidence_id": "synthetic-calibration-evidence",
            "finite_sample_super_uniformity_certified": status == "certified",
            "scope": "complete-within-cohort-conjunction-family",
            "status": status,
        },
        "schema": postprocess.MARGINAL_VALIDITY_EVIDENCE_SCHEMA,
        "tested_family": asdict(fit_policy.d5.tested_family),
        "top_k": postprocess.TOP_K,
        "upstream": {
            "canonical_input_manifest_sha256": (config.expected_canonical_input_sha256),
            "fit_approval_manifest_sha256": config.expected_fit_approval_sha256,
            "provider_input_manifest_sha256": (
                config.expected_provider_input_manifest_sha256
            ),
            "sealed_completion_manifest_sha256": (
                config.expected_sealed_completion_sha256
            ),
        },
    }


def _row(
    pair,
    *,
    likelihood_ratio,
    taus=_ME_TAUS,
    effect_identifiability=PAIR_EFFECT_IDENTIFIED_STATUS,
    contingency=_CONTINGENCY,
):
    tau_00, tau_10, tau_01, tau_11 = taus
    identified = effect_identifiability == PAIR_EFFECT_IDENTIFIED_STATUS
    rho = (
        compute_marshall_olkin_rho([tau_00, tau_01, tau_10, tau_11])
        if identified
        else None
    )
    null_ll = -20.0
    alternative_ll = null_ll + likelihood_ratio / 2.0
    return {
        "Gene A": pair[0],
        "Gene B": pair[1],
        "Tau_00": tau_00,
        "Tau_10": tau_10,
        "Tau_01": tau_01,
        "Tau_11": tau_11,
        "_00_": contingency[0],
        "_10_": contingency[1],
        "_01_": contingency[2],
        "_11_": contingency[3],
        "Tau_1X": tau_10 + tau_11 if identified else "",
        "Tau_X1": tau_01 + tau_11 if identified else "",
        "Effect Identifiability": effect_identifiability,
        "Rho": "" if rho is None else rho,
        "Log Odds Ratio": "" if not identified or 0 in taus else 0.0,
        "Likelihood Ratio": likelihood_ratio,
        "Wald Statistic": "" if not identified or 0 in taus else 0.0,
        "Fit Algorithm": runner.REQUIRED_PAIR_FIT_CONTRACT,
        "Fit Converged": True,
        "Fit Iterations": 5,
        "Fit Last LL Gain": 0.0,
        "Fit Fixed-Point Residual": 0.0,
        "Fit KKT Residual": 0.0,
        "Pair Fit Contract": runner.REQUIRED_PAIR_FIT_CONTRACT,
        "Null Log Likelihood": null_ll,
        "Alternative Log Likelihood": alternative_ll,
        "LRT Contract": runner.REQUIRED_LRT_CONTRACT,
    }


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=runner.PAIRWISE_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _component_tables():
    pairs = tuple(runner.iter_tested_pairs(_FEATURES))
    specifications = {
        "cbase": (
            (16.0, _ME_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
            (10.0, _CO_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
            (7.0, _ME_TAUS, PAIR_EFFECT_RANK_DEFICIENT_STATUS),
        ),
        "dig": (
            (14.0, _ME_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
            (8.0, _CO_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
            (6.0, _ME_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
        ),
        "mutsig": (
            (12.0, _ME_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
            (6.0, _ME_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
            (5.0, _ME_TAUS, PAIR_EFFECT_IDENTIFIED_STATUS),
        ),
    }
    return {
        provider: _csv_bytes(
            [
                _row(
                    pair,
                    likelihood_ratio=likelihood_ratio,
                    taus=taus,
                    effect_identifiability=effect,
                )
                for pair, (likelihood_ratio, taus, effect) in zip(
                    pairs,
                    provider_specifications,
                    strict=True,
                )
            ],
        )
        for provider, provider_specifications in specifications.items()
    }


def _read_output(derived):
    return list(csv.DictReader(io.StringIO(derived.csv_bytes.decode())))


def _write_qa_family(output, derived):
    postprocess._write_derived_cohort_family_for_qa(output, derived)  # noqa: SLF001


@pytest.mark.parametrize(
    "effect_status",
    [PAIR_EFFECT_RANK_DEFICIENT_STATUS, PAIR_EFFECT_UNDERFLOW_STATUS],
)
def test_unidentified_positive_lrt_retains_profile_p(effect_status):
    component = revision_inference.classify_component(4.0, effect_status, None)

    assert component.status == revision_inference.VALID_PROFILE_LRT
    assert component.p_value == pytest.approx(float(gammaincc(0.5, 2.0)))
    assert component.p_value < 1
    assert component.direction == "unavailable"


def test_only_identified_undefined_rho_boundary_gets_explicit_p_one():
    component = revision_inference.classify_component(
        runner.REQUIRED_UNDEFINED_RHO_LRT_TOL,
        PAIR_EFFECT_IDENTIFIED_STATUS,
        None,
    )

    assert component.status == revision_inference.VALID_DEGENERATE_NULL_P_ONE
    assert component.p_value == 1
    with pytest.raises(revision_inference.RevisionInferenceError, match="boundary"):
        revision_inference.classify_component(
            runner.REQUIRED_UNDEFINED_RHO_LRT_TOL * 2,
            PAIR_EFFECT_IDENTIFIED_STATUS,
            None,
        )
    with pytest.raises(revision_inference.RevisionInferenceError, match="cannot"):
        revision_inference.classify_component(
            4.0,
            PAIR_EFFECT_RANK_DEFICIENT_STATUS,
            -0.1,
        )


def test_complete_family_uses_max_p_and_retains_both_adjustments():
    component = revision_inference.ComponentInference
    components = {
        "cbase": (
            component("valid-profile-lrt", 0.001, "me", PAIR_EFFECT_IDENTIFIED_STATUS),
            component("valid-profile-lrt", 0.03, "co", PAIR_EFFECT_IDENTIFIED_STATUS),
            component("valid-profile-lrt", 0.2, "me", PAIR_EFFECT_IDENTIFIED_STATUS),
        ),
        "dig": (
            component("valid-profile-lrt", 0.002, "me", PAIR_EFFECT_IDENTIFIED_STATUS),
            component("valid-profile-lrt", 0.02, "co", PAIR_EFFECT_IDENTIFIED_STATUS),
            component("valid-profile-lrt", 0.1, "me", PAIR_EFFECT_IDENTIFIED_STATUS),
        ),
        "mutsig": (
            component("valid-profile-lrt", 0.004, "me", PAIR_EFFECT_IDENTIFIED_STATUS),
            component("valid-profile-lrt", 0.01, "me", PAIR_EFFECT_IDENTIFIED_STATUS),
            component(
                "valid-profile-lrt",
                0.3,
                "unavailable",
                PAIR_EFFECT_RANK_DEFICIENT_STATUS,
            ),
        ),
    }
    family = revision_inference.derive_complete_family(
        components,
        policy=_policy(),
        marginal_validity_certified=False,
    )

    assert family.conjunction_p_values == pytest.approx((0.004, 0.03, 0.3))
    assert family.bh_q_values == pytest.approx((0.012, 0.045, 0.3))
    assert family.by_q_values == pytest.approx((0.022, 0.0825, 0.55))
    assert family.consensus_directions == (
        "unanimous-me",
        "discordant",
        "unavailable",
    )
    assert not family.primary_inferential_eligible
    assert not any(family.by_primary_reportable)


def test_rho_and_calibration_never_change_p_or_q_values():
    component = revision_inference.ComponentInference

    def components(direction):
        return {
            provider: (
                component(
                    "valid-profile-lrt",
                    0.001,
                    direction,
                    PAIR_EFFECT_IDENTIFIED_STATUS,
                ),
            )
            for provider in revision_inference.COMPONENT_ORDER
        }

    me_blocked = revision_inference.derive_complete_family(
        components("me"),
        policy=_policy(),
        marginal_validity_certified=False,
    )
    co_certified = revision_inference.derive_complete_family(
        components("co"),
        policy=_policy(),
        marginal_validity_certified=True,
    )

    assert me_blocked.conjunction_p_values == co_certified.conjunction_p_values
    assert me_blocked.by_q_values == co_certified.by_q_values
    assert me_blocked.bh_q_values == co_certified.bh_q_values
    assert me_blocked.consensus_directions != co_certified.consensus_directions
    assert not me_blocked.by_primary_reportable[0]
    assert co_certified.by_primary_reportable[0]


def test_inclusive_thresholds_and_policy_drift_fail_closed():
    component = revision_inference.ComponentInference(
        "valid-profile-lrt",
        0.01,
        "neutral",
        PAIR_EFFECT_IDENTIFIED_STATUS,
    )
    family = revision_inference.derive_complete_family(
        dict.fromkeys(revision_inference.COMPONENT_ORDER, (component,)),
        policy=_policy(),
        marginal_validity_certified=True,
    )

    assert family.by_q_values == (0.01,)
    assert family.bh_q_values == (0.01,)
    assert family.by_primary_threshold_crossings == (True,)
    assert family.by_primary_reportable == (True,)
    bad_multiplicity = replace(_policy().multiplicity, primary_method="bh")
    with pytest.raises(revision_inference.RevisionInferenceError, match="exact"):
        revision_inference.derive_complete_family(
            dict.fromkeys(revision_inference.COMPONENT_ORDER, (component,)),
            policy=replace(_policy(), multiplicity=bad_multiplicity),
            marginal_validity_certified=True,
        )


@pytest.mark.parametrize(
    ("family_field", "bad_value"),
    [
        ("top_k", 100),
        ("feature_ranking", "rank-by-provider"),
        ("tie_break", "alphabetical"),
        ("provider_support", "provider-native-separate"),
        ("pair_construction", "selected-pairs-only"),
        ("same_base_missense_nonsense", "retain"),
        ("epsilon_pretest_filter", "positive-only"),
        ("marginal_effect_pretest_filter", "positive-only"),
        ("family", "filtered-family"),
    ],
)
def test_v3_tested_family_is_exactly_bound(family_field, bad_value):
    component = revision_inference.ComponentInference(
        "valid-profile-lrt",
        0.2,
        "me",
        PAIR_EFFECT_IDENTIFIED_STATUS,
    )
    corrupted_family = replace(
        _policy().tested_family,
        **{family_field: bad_value},
    )

    with pytest.raises(revision_inference.RevisionInferenceError, match="exact"):
        revision_inference.derive_complete_family(
            dict.fromkeys(revision_inference.COMPONENT_ORDER, (component,)),
            policy=replace(_policy(), tested_family=corrupted_family),
            marginal_validity_certified=False,
        )


def test_v3_binds_both_descriptive_methods():
    component = revision_inference.ComponentInference(
        "valid-profile-lrt",
        0.2,
        "me",
        PAIR_EFFECT_IDENTIFIED_STATUS,
    )
    bad_multiplicity = replace(
        _policy().multiplicity,
        descriptive_methods=("bh", "by"),
    )

    with pytest.raises(revision_inference.RevisionInferenceError, match="exact"):
        revision_inference.derive_complete_family(
            dict.fromkeys(revision_inference.COMPONENT_ORDER, (component,)),
            policy=replace(_policy(), multiplicity=bad_multiplicity),
            marginal_validity_certified=False,
        )


def test_invalid_or_incomplete_component_family_never_substitutes_p_one():
    valid = revision_inference.ComponentInference(
        "valid-profile-lrt",
        0.2,
        "me",
        PAIR_EFFECT_IDENTIFIED_STATUS,
    )
    invalid = replace(valid, status="fit-failed")
    components = dict.fromkeys(revision_inference.COMPONENT_ORDER, (valid,))
    components["mutsig"] = (invalid,)

    with pytest.raises(revision_inference.RevisionInferenceError, match="substitution"):
        revision_inference.derive_complete_family(
            components,
            policy=_policy(),
            marginal_validity_certified=False,
        )
    components["mutsig"] = ()
    with pytest.raises(revision_inference.RevisionInferenceError, match="complete"):
        revision_inference.derive_complete_family(
            components,
            policy=_policy(),
            marginal_validity_certified=False,
        )
    components["mutsig"] = (replace(valid, effect_identifiability="unknown"),)
    with pytest.raises(revision_inference.RevisionInferenceError, match="status"):
        revision_inference.derive_complete_family(
            components,
            policy=_policy(),
            marginal_validity_certified=False,
        )


def test_wrapper_derives_deterministic_complete_family_and_provenance():
    inputs = _component_tables()
    evidence = postprocess.MarginalValidityEvidence("inconclusive")

    first = postprocess.derive_conjunction_family(
        inputs,
        axis=_axis(),
        policy=_policy(),
        marginal_validity=evidence,
    )
    second = postprocess.derive_conjunction_family(
        dict(reversed(tuple(inputs.items()))),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=evidence,
    )
    rows = _read_output(first)
    manifest = json.loads(first.manifest_bytes)

    assert first == second
    assert first.row_count == len(tuple(runner.iter_tested_pairs(_FEATURES)))
    assert first.csv_sha256 == hashlib.sha256(first.csv_bytes).hexdigest()
    assert first.manifest_sha256 == hashlib.sha256(first.manifest_bytes).hexdigest()
    assert first.production_eligible is False
    assert [row["consensus_direction"] for row in rows] == [
        "unanimous-me",
        "discordant",
        "unavailable",
    ]
    assert rows[2]["cbase_component_status"] == "valid-profile-lrt"
    assert float(rows[2]["cbase_p_value"]) < 1
    assert all(row["conditional_by_inferential_eligible"] == "false" for row in rows)
    assert all(row["d3_conjunction_role"] == "secondary" for row in rows)
    assert manifest["complete_family_required"] is True
    assert manifest["pair_filtering"] is False
    assert manifest["pair_ranking"] is False
    assert manifest["production_eligible"] is False
    assert manifest["d5_contract"] == "conjunction-multiplicity-family-policy-v3"
    assert manifest["tested_family"] == {
        "top_k": 500,
        "feature_ranking": "descending-total-eligible-mutation-event-count",
        "tie_break": "canonical-count-matrix-column-order",
        "provider_support": "shared-native-cbase-dig-mutsig",
        "pair_construction": "all-unordered-pairs-of-ordered-feature-axis",
        "same_base_missense_nonsense": "exclude-before-fitting-and-testing",
        "epsilon_pretest_filter": "none",
        "marginal_effect_pretest_filter": "none",
        "family": "one-complete-within-cohort-tested-pair-family",
    }
    assert manifest["multiplicity"]["computed_methods"] == ["by", "bh"]
    assert manifest["multiplicity"]["descriptive_methods"] == ["by", "bh"]
    assert manifest["marginal_validity"]["correction_selection_affected"] is False
    assert manifest["marginal_validity"]["q_values_affected"] is False


def test_wrapper_calibration_changes_only_eligibility_fields():
    inputs = _component_tables()
    blocked = postprocess.derive_conjunction_family(
        inputs,
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("inconclusive"),
    )
    absent = postprocess.derive_conjunction_family(
        inputs,
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )

    blocked_rows = _read_output(blocked)
    absent_rows = _read_output(absent)
    stable_fields = (
        "conjunction_p_value",
        "by_q_value",
        "bh_q_value",
        "by_q_le_0_01",
        "bh_q_le_0_01_nominal",
        "by_q_le_0_05_descriptive",
        "bh_q_le_0_05_descriptive",
    )
    for blocked_row, absent_row in zip(
        blocked_rows,
        absent_rows,
        strict=True,
    ):
        assert {field: blocked_row[field] for field in stable_fields} == {
            field: absent_row[field] for field in stable_fields
        }
        assert blocked_row["conditional_by_inferential_eligible"] == "false"
        assert absent_row["conditional_by_inferential_eligible"] == "false"


def test_caller_cannot_construct_certified_marginal_validity():
    with pytest.raises(postprocess.D5DerivationError, match="authenticated"):
        postprocess.MarginalValidityEvidence(
            "certified",
            "forged-calibration",
            "b" * 64,
        )


def test_blocking_validity_requires_inspection_but_no_calibration_authority(tmp_path):
    config = replace(
        _config(
            tmp_path,
            evidence_sha256="b" * 64,
            design_path=(tmp_path / "design.json").resolve(),
        ),
        calibration_approval_manifest=None,
        calibration_evidence_artifact=None,
        calibration_design_manifest=None,
        marginal_validity_status="inconclusive",
        expected_calibration_approval_sha256=None,
        expected_calibration_evidence_sha256=None,
    )

    postprocess._validate_production_config(config)  # noqa: SLF001
    assert config.inspect_approval_manifest.is_absolute()
    with pytest.raises(postprocess.D5DerivationError, match="requires calibration"):
        postprocess._validate_production_config(  # noqa: SLF001
            replace(config, marginal_validity_status="certified"),
        )


def test_degenerate_boundary_and_nonidentified_positive_lrt_port_to_csv():
    features = ("A_M", "B_M")
    pair = next(runner.iter_tested_pairs(features))
    tables = {
        "cbase": _csv_bytes(
            [
                _row(
                    pair,
                    likelihood_ratio=0.0,
                    taus=_DEGENERATE_TAUS,
                ),
            ],
        ),
        "dig": _csv_bytes(
            [
                _row(
                    pair,
                    likelihood_ratio=4.0,
                    effect_identifiability=PAIR_EFFECT_RANK_DEFICIENT_STATUS,
                ),
            ],
        ),
        "mutsig": _csv_bytes([_row(pair, likelihood_ratio=5.0)]),
    }
    derived = postprocess.derive_conjunction_family(
        tables,
        axis=_axis(features),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    row = _read_output(derived)[0]

    assert row["cbase_component_status"] == "valid-degenerate-null-p-one"
    assert row["cbase_p_value"] == "1"
    assert row["dig_component_status"] == "valid-profile-lrt"
    assert 0 < float(row["dig_p_value"]) < 1
    assert row["consensus_direction"] == "unavailable"
    assert row["conjunction_p_value"] == "1"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows[:-1], "ended before"),
        (lambda rows: [rows[0], rows[0], rows[2]], "duplicate"),
        (lambda rows: [rows[1], rows[0], rows[2]], "coordinate mismatch"),
    ],
)
def test_missing_duplicate_or_reordered_component_fails_whole_cohort(mutate, match):
    inputs = _component_tables()
    rows = list(csv.DictReader(io.StringIO(inputs["dig"].decode())))
    inputs["dig"] = _csv_bytes(mutate(rows))

    with pytest.raises(postprocess.D5DerivationError, match=match):
        postprocess.derive_conjunction_family(
            inputs,
            axis=_axis(),
            policy=_policy(),
            marginal_validity=postprocess.MarginalValidityEvidence("absent"),
        )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("Fit Converged", "False", "not a converged"),
        ("Effect Identifiability", "unknown", "identifiability"),
        ("Likelihood Ratio", "nan", "finite"),
        ("LRT Contract", "historical-lrt", "provenance"),
        ("Fit KKT Residual", "0.1", "convergence"),
    ],
)
def test_invalid_component_row_aborts_without_partial_family(column, value, match):
    inputs = _component_tables()
    rows = list(csv.DictReader(io.StringIO(inputs["mutsig"].decode())))
    rows[0][column] = value
    inputs["mutsig"] = _csv_bytes(rows)

    with pytest.raises(postprocess.D5DerivationError, match=match):
        postprocess.derive_conjunction_family(
            inputs,
            axis=_axis(),
            policy=_policy(),
            marginal_validity=postprocess.MarginalValidityEvidence("absent"),
        )


def test_cross_provider_observation_or_coordinate_schema_corruption_fails_closed():
    inputs = _component_tables()
    rows = list(csv.DictReader(io.StringIO(inputs["mutsig"].decode())))
    rows[0]["_00_"] = str(int(rows[0]["_00_"]) + 1)
    inputs["mutsig"] = _csv_bytes(rows)
    with pytest.raises(postprocess.D5DerivationError, match="sample universe"):
        postprocess.derive_conjunction_family(
            inputs,
            axis=_axis(),
            policy=_policy(),
            marginal_validity=postprocess.MarginalValidityEvidence("absent"),
        )

    inputs = _component_tables()
    header, *body = inputs["cbase"].decode().splitlines()
    inputs["cbase"] = (
        header.replace("Gene A", "Gene X") + "\n" + "\n".join(body)
    ).encode()
    with pytest.raises(postprocess.D5DerivationError, match="schema"):
        postprocess.derive_conjunction_family(
            inputs,
            axis=_axis(),
            policy=_policy(),
            marginal_validity=postprocess.MarginalValidityEvidence("absent"),
        )


def test_provider_set_axis_and_evidence_are_fail_closed():
    inputs = _component_tables()
    inputs.pop("mutsig")
    with pytest.raises(postprocess.D5DerivationError, match="exactly one"):
        postprocess.derive_conjunction_family(
            inputs,
            axis=_axis(),
            policy=_policy(),
            marginal_validity=postprocess.MarginalValidityEvidence("absent"),
        )

    corrupt_axis = replace(_axis(), ordered_pair_sha256="0" * 64)
    with pytest.raises(postprocess.D5DerivationError, match="does not match"):
        postprocess.derive_conjunction_family(
            _component_tables(),
            axis=corrupt_axis,
            policy=_policy(),
            marginal_validity=postprocess.MarginalValidityEvidence("absent"),
        )
    with pytest.raises(postprocess.D5DerivationError, match="authenticated"):
        postprocess.MarginalValidityEvidence("certified")


def test_production_entry_point_rejects_caller_supplied_bytes():
    with pytest.raises(TypeError, match="validated authority"):
        postprocess.derive_k500_cohort_conjunction(_component_tables(), "CHOL")


def test_writer_is_new_directory_only(tmp_path):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    output = tmp_path / "derived"

    _write_qa_family(output, derived)

    assert (output / postprocess.OUTPUT_CSV_NAME).read_bytes() == derived.csv_bytes
    assert (output / postprocess.OUTPUT_MANIFEST_NAME).read_bytes() == (
        derived.manifest_bytes
    )
    with pytest.raises(FileExistsError):
        _write_qa_family(output, derived)


def test_production_writer_rejects_synthetic_axis(tmp_path):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    output = tmp_path / "derived"

    with pytest.raises(TypeError, match="sealed derivation"):
        postprocess.write_derived_cohort_family(output, derived)
    assert not output.exists()


def test_writer_failure_never_exposes_partial_destination(tmp_path, monkeypatch):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    output = tmp_path / "derived"
    original_write = postprocess._write_exclusive_synced_at  # noqa: SLF001

    def fail_manifest(directory_fd, name, content):
        if name == postprocess.OUTPUT_MANIFEST_NAME:
            msg = "synthetic staged-write failure"
            raise OSError(msg)
        original_write(directory_fd, name, content)

    monkeypatch.setattr(postprocess, "_write_exclusive_synced_at", fail_manifest)

    with pytest.raises(OSError, match="synthetic"):
        _write_qa_family(output, derived)
    assert not output.exists()
    staging = list(tmp_path.glob(".derived.*.tmp"))
    assert len(staging) == 1
    assert (staging[0] / postprocess.OUTPUT_CSV_NAME).is_file()
    assert not (staging[0] / postprocess.OUTPUT_MANIFEST_NAME).exists()


def test_writer_destination_race_never_replaces_raced_entry(tmp_path, monkeypatch):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    output = tmp_path / "derived"
    original_rename = postprocess._rename_no_replace_at  # noqa: SLF001

    def race_destination(parent_fd, source_name, destination_name):
        os.mkdir(destination_name, mode=0o700, dir_fd=parent_fd)
        destination_fd = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        try:
            marker_fd = os.open(
                "racer-marker",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_fd,
            )
            os.close(marker_fd)
        finally:
            os.close(destination_fd)
        original_rename(parent_fd, source_name, destination_name)

    monkeypatch.setattr(postprocess, "_rename_no_replace_at", race_destination)

    with pytest.raises(FileExistsError):
        _write_qa_family(output, derived)
    assert (output / "racer-marker").is_file()
    assert not (output / postprocess.OUTPUT_CSV_NAME).exists()
    assert not (output / postprocess.OUTPUT_MANIFEST_NAME).exists()


def test_writer_rejects_staging_directory_swap(tmp_path, monkeypatch):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    output = tmp_path / "derived"
    original_check = postprocess._require_same_directory_entry  # noqa: SLF001

    def swap_staging(parent_fd, name, expected):
        moved = f"{name}.moved"
        os.rename(
            name,
            moved,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        original_check(parent_fd, name, expected)

    monkeypatch.setattr(postprocess, "_require_same_directory_entry", swap_staging)

    with pytest.raises(postprocess.D5DerivationError, match="identity changed"):
        _write_qa_family(output, derived)
    assert not output.exists()


def test_writer_rejects_staged_file_link_race(tmp_path, monkeypatch):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    output = tmp_path / "derived"
    original_verify = postprocess._verify_staged_file_at  # noqa: SLF001

    def link_staged_file(directory_fd, name, expected_content):
        if name == postprocess.OUTPUT_CSV_NAME:
            os.link(
                name,
                f"{name}.second-link",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        original_verify(directory_fd, name, expected_content)

    monkeypatch.setattr(postprocess, "_verify_staged_file_at", link_staged_file)

    with pytest.raises(postprocess.D5DerivationError, match="single-link"):
        _write_qa_family(output, derived)
    assert not output.exists()


def test_writer_rejects_corrupt_derived_receipt_before_staging(tmp_path):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    corrupt = replace(derived, csv_bytes=derived.csv_bytes + b"tamper")
    output = tmp_path / "derived"

    with pytest.raises(postprocess.D5DerivationError, match="digests"):
        _write_qa_family(output, corrupt)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation_path", "bad_value"),
    [
        (("correction_policy", "correction_selection_affected"), True),
        (("upstream", "sealed_completion_manifest_sha256"), "f" * 64),
        (
            ("calibration_authority", "calibration_decision_digests"),
            dict.fromkeys(("D4", "D5", "D6"), "9" * 64),
        ),
        (
            ("calibration_authority", "fit_decision_digests"),
            dict.fromkeys(("D4", "D5", "D6"), "6" * 64),
        ),
        (("schema",), "dialect-tcga-k500-marginal-validity-evidence-v1"),
    ],
)
def test_authenticated_evidence_cannot_select_correction_or_swap_upstream(
    tmp_path,
    mutation_path,
    bad_value,
):
    design_path = tmp_path / "design.json"
    design_raw = b'{"design":"synthetic"}\n'
    design_path.write_bytes(design_raw)
    fit_policy = _fit_policy_stub(hashlib.sha256(design_raw).hexdigest())
    config = _config(
        tmp_path,
        evidence_sha256="0" * 64,
        design_path=design_path.resolve(),
    )
    payload = _calibration_evidence_payload(config, fit_policy)
    target = payload
    for component in mutation_path[:-1]:
        target = target[component]
    target[mutation_path[-1]] = bad_value
    raw = postprocess._canonical_json(payload) + b"\n"  # noqa: SLF001
    config.calibration_evidence_artifact.parent.mkdir(parents=True, exist_ok=True)
    config.calibration_evidence_artifact.write_bytes(raw)
    config = replace(
        config,
        expected_calibration_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    )
    fit_approval = SimpleNamespace(
        manifest_sha256=config.expected_fit_approval_sha256,
    )
    calibration_approval = SimpleNamespace(
        manifest_sha256=config.expected_calibration_approval_sha256,
        schema="dialect-revision-coauthor-approval-v6",
        decision_digests={f"D{index}": "9" * 64 for index in range(1, 7)},
    )
    fit_approval.decision_digests = {f"D{index}": "6" * 64 for index in range(1, 7)}

    with pytest.raises(postprocess.D5DerivationError, match="invalid authenticated"):
        postprocess._validate_calibration_evidence(  # noqa: SLF001
            config,
            fit_policy=fit_policy,
            fit_approval=fit_approval,
            calibration_approval=calibration_approval,
        )


def test_narrow_calibration_cannot_self_certify_from_pinned_bytes(tmp_path):
    design_path = tmp_path / "design.json"
    design_raw = b'{"design":"synthetic"}\n'
    design_path.write_bytes(design_raw)
    fit_policy = _fit_policy_stub(hashlib.sha256(design_raw).hexdigest())
    config = _config(
        tmp_path,
        evidence_sha256="0" * 64,
        design_path=design_path.resolve(),
    )
    payload = _calibration_evidence_payload(config, fit_policy)
    raw = postprocess._canonical_json(payload) + b"\n"  # noqa: SLF001
    config.calibration_evidence_artifact.parent.mkdir(parents=True, exist_ok=True)
    config.calibration_evidence_artifact.write_bytes(raw)
    config = replace(
        config,
        expected_calibration_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(postprocess.D5DerivationError, match="Narrow/local"):
        postprocess._validate_calibration_evidence(  # noqa: SLF001
            config,
            fit_policy=fit_policy,
            fit_approval=SimpleNamespace(
                manifest_sha256=config.expected_fit_approval_sha256,
                decision_digests={f"D{index}": "6" * 64 for index in range(1, 7)},
            ),
            calibration_approval=SimpleNamespace(
                manifest_sha256=config.expected_calibration_approval_sha256,
                schema="dialect-revision-coauthor-approval-v6",
                decision_digests={f"D{index}": "9" * 64 for index in range(1, 7)},
            ),
        )


def test_calibration_stage_authority_rejects_swapped_canonical_root(
    tmp_path,
    monkeypatch,
):
    config = _config(
        tmp_path,
        evidence_sha256="a" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    decision_ids = tuple(f"D{index}" for index in range(1, 7))
    calibration = SimpleNamespace(
        schema="dialect-revision-coauthor-approval-v6",
        allowed_stages=("calibration",),
        stage_bindings={
            "calibration": {
                "canonical_input_manifest_sha256": "f" * 64,
                "provider_input_manifest_sha256": (
                    config.expected_provider_input_manifest_sha256
                ),
                "upstream_result_manifest_sha256": (
                    config.expected_sealed_completion_sha256
                ),
            },
        },
        decisions={decision_id: object() for decision_id in decision_ids},
        decision_digests=dict.fromkeys(decision_ids, "9" * 64),
    )
    monkeypatch.setattr(
        postprocess,
        "validate_revision_approval",
        lambda *_args, **_kwargs: calibration,
    )

    with pytest.raises(postprocess.D5DerivationError, match="bound to the sealed"):
        postprocess._validate_calibration_approval(  # noqa: SLF001
            config,
            fit_approval=SimpleNamespace(),
        )


def test_calibration_stage_authority_rejects_d4_through_d6_only(
    tmp_path,
    monkeypatch,
):
    config = _config(
        tmp_path,
        evidence_sha256="a" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    decision_ids = ("D4", "D5", "D6")
    calibration = SimpleNamespace(
        schema="dialect-revision-coauthor-approval-v6",
        allowed_stages=("calibration",),
        stage_bindings={
            "calibration": {
                "canonical_input_manifest_sha256": (
                    config.expected_canonical_input_sha256
                ),
                "provider_input_manifest_sha256": (
                    config.expected_provider_input_manifest_sha256
                ),
                "upstream_result_manifest_sha256": (
                    config.expected_sealed_completion_sha256
                ),
            },
        },
        decisions={decision_id: object() for decision_id in decision_ids},
        decision_digests=dict.fromkeys(decision_ids, "9" * 64),
    )
    monkeypatch.setattr(
        postprocess,
        "validate_revision_approval",
        lambda *_args, **_kwargs: calibration,
    )

    with pytest.raises(postprocess.D5DerivationError, match="exact singleton"):
        postprocess._validate_calibration_approval(  # noqa: SLF001
            config,
            fit_approval=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "legacy_schema",
    [
        "dialect-revision-coauthor-approval-v4",
        "dialect-revision-coauthor-approval-v5",
    ],
)
def test_calibration_stage_authority_rejects_legacy_schema_even_with_d1_d6(
    tmp_path,
    monkeypatch,
    legacy_schema,
):
    config = _config(
        tmp_path,
        evidence_sha256="a" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    decision_ids = tuple(f"D{index}" for index in range(1, 7))
    calibration = SimpleNamespace(
        schema=legacy_schema,
        allowed_stages=("calibration",),
        stage_bindings={
            "calibration": {
                "canonical_input_manifest_sha256": (
                    config.expected_canonical_input_sha256
                ),
                "provider_input_manifest_sha256": (
                    config.expected_provider_input_manifest_sha256
                ),
                "upstream_result_manifest_sha256": (
                    config.expected_sealed_completion_sha256
                ),
            },
        },
        decisions={decision_id: object() for decision_id in decision_ids},
        decision_digests=dict.fromkeys(decision_ids, "9" * 64),
    )
    monkeypatch.setattr(
        postprocess,
        "validate_revision_approval",
        lambda *_args, **_kwargs: calibration,
    )

    with pytest.raises(postprocess.D5DerivationError, match="stage-scoped v6"):
        postprocess._validate_calibration_approval(  # noqa: SLF001
            config,
            fit_approval=SimpleNamespace(),
        )


def test_postprocess_authority_receipt_versions_and_records_calibration_digests(
    tmp_path,
    monkeypatch,
):
    config = _config(
        tmp_path,
        evidence_sha256="a" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    decision_ids = tuple(f"D{index}" for index in range(1, 7))
    decision_digests = {
        decision_id: hashlib.sha256(decision_id.encode()).hexdigest()
        for decision_id in decision_ids
    }
    calibration = SimpleNamespace(
        manifest_sha256=config.expected_calibration_approval_sha256,
        schema="dialect-revision-coauthor-approval-v6",
        decision_digests=decision_digests,
    )
    monkeypatch.setattr(runner, "_fit_policy_record", lambda _policy: {})
    record = postprocess._build_authority_record(  # noqa: SLF001
        config,
        completion={"grid": {"task_count": 96}},
        fit_policy=SimpleNamespace(
            d3=SimpleNamespace(all_three_conjunction_role="secondary"),
        ),
        fit_approval=SimpleNamespace(
            manifest_sha256=config.expected_fit_approval_sha256,
        ),
        inspect_approval=SimpleNamespace(
            manifest_sha256=config.expected_inspect_approval_sha256,
        ),
        calibration_approval=calibration,
        marginal_validity=postprocess.MarginalValidityEvidence("inconclusive"),
        contract_bytes=(b"{}\n",) * len(postprocess.TCGA_COHORTS),
    )

    assert record["schema"] == postprocess.POSTPROCESS_AUTHORITY_SCHEMA
    assert record["contract"] == postprocess.POSTPROCESS_AUTHORITY_CONTRACT
    assert record["approvals"]["calibration"] == {
        "path": config.calibration_approval_manifest.as_posix(),
        "sha256": config.expected_calibration_approval_sha256,
        "schema": "dialect-revision-coauthor-approval-v6",
        "authorized_stage": "calibration",
        "decision_digests": decision_digests,
    }


@pytest.mark.parametrize(
    "drifted_decision_id",
    [None, *(f"D{index}" for index in range(1, 7))],
)
def test_calibration_stage_authority_requires_exact_d1_through_d6_reauthorization(
    tmp_path,
    monkeypatch,
    drifted_decision_id,
):
    config = _config(
        tmp_path,
        evidence_sha256="a" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    decision_ids = tuple(f"D{index}" for index in range(1, 7))
    fit_decisions = {
        decision_id: SimpleNamespace(marker=f"fit-{decision_id}")
        for decision_id in decision_ids
    }
    calibration_decisions = {
        decision_id: SimpleNamespace(marker=f"fit-{decision_id}")
        for decision_id in decision_ids
    }
    if drifted_decision_id is not None:
        calibration_decisions[drifted_decision_id] = SimpleNamespace(
            marker=f"drifted-{drifted_decision_id}",
        )
    calibration = SimpleNamespace(
        schema="dialect-revision-coauthor-approval-v6",
        allowed_stages=("calibration",),
        stage_bindings={
            "calibration": {
                "canonical_input_manifest_sha256": (
                    config.expected_canonical_input_sha256
                ),
                "provider_input_manifest_sha256": (
                    config.expected_provider_input_manifest_sha256
                ),
                "upstream_result_manifest_sha256": (
                    config.expected_sealed_completion_sha256
                ),
            },
        },
        decisions=calibration_decisions,
        decision_digests=dict.fromkeys(decision_ids, "9" * 64),
    )
    monkeypatch.setattr(
        postprocess,
        "validate_revision_approval",
        lambda *_args, **_kwargs: calibration,
    )
    monkeypatch.setattr(
        runner,
        "_decision_reauthorization_record",
        lambda decision: decision.marker,
    )

    fit_approval = SimpleNamespace(decisions=fit_decisions)
    if drifted_decision_id is None:
        assert (
            postprocess._validate_calibration_approval(  # noqa: SLF001
                config,
                fit_approval=fit_approval,
            )
            is calibration
        )
    else:
        with pytest.raises(
            postprocess.D5DerivationError,
            match=rf"signed {drifted_decision_id}",
        ):
            postprocess._validate_calibration_approval(  # noqa: SLF001
                config,
                fit_approval=fit_approval,
            )


def test_fit_authority_alone_cannot_open_rows_without_inspect_authority(
    tmp_path,
    monkeypatch,
):
    config = _config(
        tmp_path,
        evidence_sha256="a" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    fit_only = SimpleNamespace(
        allowed_stages=("fit-sealed-tcga-k500",),
        stage_bindings={
            "fit-sealed-tcga-k500": {
                "canonical_input_manifest_sha256": (
                    config.expected_canonical_input_sha256
                ),
                "provider_input_manifest_sha256": (
                    config.expected_provider_input_manifest_sha256
                ),
            },
        },
    )
    monkeypatch.setattr(
        postprocess,
        "validate_revision_approval",
        lambda *_args, **_kwargs: fit_only,
    )

    with pytest.raises(postprocess.D5DerivationError, match="Inspection approval"):
        postprocess._validate_inspect_approval(  # noqa: SLF001
            config,
            fit_approval=SimpleNamespace(),
        )


def _synthetic_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(postprocess, "TCGA_COHORTS", ("TEST",))
    root = tmp_path / "run"
    contract_path = root / "contracts" / "TEST.json"
    contract_path.parent.mkdir(parents=True)
    contract = {
        "cohort": "TEST",
        "features": ["A_M", "B_M"],
        "top_k": postprocess.TOP_K,
    }
    contract_raw = postprocess._canonical_json(contract) + b"\n"  # noqa: SLF001
    contract_path.write_bytes(contract_raw)
    implementation = {"runner.py": "a" * 64}
    run_manifest = {"implementation_sha256": implementation}
    run_raw = postprocess._canonical_json(run_manifest) + b"\n"  # noqa: SLF001
    provider_receipt = {"authority": "synthetic-provider-root"}

    def metadata(task_dir, _contract, task):
        single = (task_dir / "single_gene_results.csv").read_bytes()
        pair = (task_dir / "pairwise_interaction_results.csv").read_bytes()
        manifest = (task_dir / "task_manifest.json").read_bytes()
        return {
            "bmr": task.bmr,
            "cohort": task.cohort,
            "consumed_input_sha256": {"counts": "c" * 64},
            "contract_sha256": runner._json_sha256(contract),  # noqa: SLF001
            "implementation_sha256": implementation,
            "pairwise_interaction_results": {
                "bytes": len(pair),
                "sha256": hashlib.sha256(pair).hexdigest(),
            },
            "provider_input_root_receipt": provider_receipt,
            "single_gene_results": {
                "bytes": len(single),
                "sha256": hashlib.sha256(single).hexdigest(),
            },
            "task_manifest": {
                "bytes": len(manifest),
                "sha256": hashlib.sha256(manifest).hexdigest(),
            },
        }

    tasks = []
    for provider in postprocess.BMRS:
        task_dir = root / "tasks" / "TEST" / provider
        task_dir.mkdir(parents=True)
        (task_dir / "single_gene_results.csv").write_bytes(
            f"single-{provider}\n".encode(),
        )
        (task_dir / "pairwise_interaction_results.csv").write_bytes(
            f"pair-{provider}\n".encode(),
        )
        (task_dir / "task_manifest.json").write_bytes(
            f"manifest-{provider}\n".encode(),
        )
        receipt = metadata(task_dir, contract, runner.Task("TEST", provider))
        receipt.pop("implementation_sha256")
        receipt.pop("provider_input_root_receipt")
        tasks.append(receipt)
    completion = {
        "analysis": "tcga-revision-k500",
        "authority": {"provider_input": provider_receipt},
        "bmrs": list(postprocess.BMRS),
        "cohorts": ["TEST"],
        "contract": postprocess.SEALED_COMPLETION_CONTRACT,
        "contracts": [
            {
                "bytes": len(contract_raw),
                "cohort": "TEST",
                "contract_sha256": runner._json_sha256(contract),  # noqa: SLF001
                "file_sha256": hashlib.sha256(contract_raw).hexdigest(),
            },
        ],
        "downstream_binding": {
            "field": "upstream_result_manifest_sha256",
            "stage": "inspect-tcga-k500",
        },
        "grid": {
            "ordered_coordinates_sha256": postprocess.sequence_sha256(
                [f"TEST/{provider}" for provider in postprocess.BMRS],
            ),
            "task_count": len(postprocess.BMRS),
        },
        "result_rows_opened": False,
        "run_manifest": {
            "bytes": len(run_raw),
            "sha256": hashlib.sha256(run_raw).hexdigest(),
        },
        "schema": postprocess.SEALED_COMPLETION_SCHEMA,
        "tasks": tasks,
        "top_k": postprocess.TOP_K,
    }
    monkeypatch.setattr(
        runner,
        "_require_closed_completion_layout",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        runner,
        "_load_verified_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(runner, "_metadata_task_receipt", metadata)
    return root, contract, run_raw, completion, provider_receipt


@pytest.mark.parametrize("corruption", ["pairwise", "task-manifest", "swapped-task"])
def test_sealed_grid_rejects_task_output_and_coordinate_swaps(
    tmp_path,
    monkeypatch,
    corruption,
):
    root, _contract, run_raw, completion, provider_receipt = _synthetic_grid(
        tmp_path,
        monkeypatch,
    )
    if corruption == "pairwise":
        path = root / "tasks" / "TEST" / "dig" / "pairwise_interaction_results.csv"
        path.write_bytes(b"swapped-pairwise\n")
    elif corruption == "task-manifest":
        path = root / "tasks" / "TEST" / "mutsig" / "task_manifest.json"
        path.write_bytes(b"swapped-manifest\n")
    else:
        first = root / "tasks" / "TEST" / "cbase"
        second = root / "tasks" / "TEST" / "dig"
        temporary = root / "tasks" / "TEST" / "temporary"
        first.rename(temporary)
        second.rename(first)
        temporary.rename(second)

    with pytest.raises(postprocess.D5DerivationError, match="task/raw-output"):
        postprocess._validate_sealed_grid(  # noqa: SLF001
            SimpleNamespace(output_root=root.resolve()),
            completion=completion,
            run_manifest_bytes=run_raw,
            run_authority={"provider_input": provider_receipt},
        )


def test_missing_sealed_completion_is_a_production_stop(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_bytes(b"{}\n")
    config = replace(
        _config(
            tmp_path,
            evidence_sha256="a" * 64,
            design_path=(tmp_path / "design.json").resolve(),
        ),
        run_output_root=root.resolve(),
    )
    fake_policy = SimpleNamespace(d3=_provider_hierarchy(), d5=_policy())
    monkeypatch.setattr(
        runner,
        "_validated_fit_approval",
        lambda *_args: SimpleNamespace(),
    )
    monkeypatch.setattr(runner, "_require_fit_stage_binding", lambda *_args: None)
    monkeypatch.setattr(
        postprocess,
        "validate_revision_fit_policy",
        lambda *_args, **_kwargs: fake_policy,
    )
    monkeypatch.setattr(
        runner,
        "_prime_parent_revision_authority",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        runner,
        "_require_completion_run_manifest",
        lambda *_args: {"provider_input": {}},
    )

    with pytest.raises(FileNotFoundError):
        postprocess.validate_postprocess_authority(config)


def test_live_root_swap_is_rejected_after_authority_validation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(postprocess, "TCGA_COHORTS", ("TEST",))
    root = tmp_path.resolve()
    run_root = root / "run"
    canonical_root = root / "canonical"
    provider_root = root / "provider"
    run_root.mkdir()
    canonical_root.mkdir()
    provider_root.mkdir()
    contract_raw = b'{"cohort":"TEST"}\n'
    (run_root / "contracts").mkdir()
    (run_root / "contracts" / "TEST.json").write_bytes(contract_raw)
    completion_raw = b'{"sealed":true}\n'
    run_raw = b'{"run":true}\n'
    canonical_raw = b'{"canonical":true}\n'
    provider_raw = b'{"provider":true}\n'
    evidence_raw = b'{"evidence":true}\n'
    approval_raws = [
        b'{"approval":1}\n',
        b'{"approval":2}\n',
        b'{"approval":3}\n',
        b'{"approval":4}\n',
    ]
    (run_root / postprocess.SEALED_COMPLETION_NAME).write_bytes(completion_raw)
    (run_root / "run_manifest.json").write_bytes(run_raw)
    (canonical_root / "input_manifest.json").write_bytes(canonical_raw)
    (provider_root / "provider_input_manifest.json").write_bytes(provider_raw)
    paths = [
        root / "input-approval.json",
        root / "fit-approval.json",
        root / "inspect-approval.json",
        root / "calibration-approval.json",
    ]
    for path, raw in zip(paths, approval_raws, strict=True):
        path.write_bytes(raw)
    evidence_path = root / "evidence.json"
    evidence_path.write_bytes(evidence_raw)
    config = postprocess.ProductionPostprocessConfig(
        run_output_root=run_root,
        canonical_input_root=canonical_root,
        provider_input_root=provider_root,
        input_approval_manifest=paths[0],
        fit_approval_manifest=paths[1],
        inspect_approval_manifest=paths[2],
        calibration_approval_manifest=paths[3],
        calibration_evidence_artifact=evidence_path,
        calibration_design_manifest=None,
        marginal_validity_status="certified",
        expected_sealed_completion_sha256=hashlib.sha256(completion_raw).hexdigest(),
        expected_canonical_input_sha256=hashlib.sha256(canonical_raw).hexdigest(),
        expected_provider_input_manifest_sha256=hashlib.sha256(
            provider_raw,
        ).hexdigest(),
        expected_input_approval_sha256=hashlib.sha256(approval_raws[0]).hexdigest(),
        expected_fit_approval_sha256=hashlib.sha256(approval_raws[1]).hexdigest(),
        expected_inspect_approval_sha256=hashlib.sha256(approval_raws[2]).hexdigest(),
        expected_calibration_approval_sha256=hashlib.sha256(
            approval_raws[3],
        ).hexdigest(),
        expected_calibration_evidence_sha256=hashlib.sha256(evidence_raw).hexdigest(),
    )
    authority = SimpleNamespace(
        config=config,
        completion_bytes=completion_raw,
        run_manifest_bytes=run_raw,
        contract_bytes=(contract_raw,),
        marginal_validity=postprocess.MarginalValidityEvidence(
            "inconclusive",
            "authenticated-blocking-evidence",
            hashlib.sha256(evidence_raw).hexdigest(),
            evidence_raw,
        ),
        fit_policy=SimpleNamespace(d6=SimpleNamespace(design_authority=None)),
        authority_record_bytes=(
            postprocess._canonical_json(  # noqa: SLF001
                {
                    "contracts": [
                        {
                            "cohort": "TEST",
                            "file_sha256": hashlib.sha256(contract_raw).hexdigest(),
                        },
                    ],
                },
            )
            + b"\n"
        ),
    )
    (canonical_root / "input_manifest.json").write_bytes(b"swapped-root\n")

    with pytest.raises(postprocess.D5DerivationError, match="canonical input"):
        postprocess._require_current_authority_files(authority)  # noqa: SLF001


def test_task_output_swap_between_derivation_and_publication_is_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(postprocess, "TCGA_COHORTS", ("TEST",))
    run_root = tmp_path / "run"
    cohort = "TEST"
    contract_sha = "a" * 64
    bindings = []
    for provider in postprocess.BMRS:
        task_dir = run_root / "tasks" / cohort / provider
        task_dir.mkdir(parents=True)
        single = f"single-{provider}\n".encode()
        pair = f"pair-{provider}\n".encode()
        task = f"task-{provider}\n".encode()
        (task_dir / "single_gene_results.csv").write_bytes(single)
        (task_dir / "pairwise_interaction_results.csv").write_bytes(pair)
        (task_dir / "task_manifest.json").write_bytes(task)
        bindings.append(
            {
                "provider": provider,
                "contract_sha256": contract_sha,
                "task_manifest_sha256": hashlib.sha256(task).hexdigest(),
                "pairwise_sha256": hashlib.sha256(pair).hexdigest(),
                "single_gene_sha256": hashlib.sha256(single).hexdigest(),
            },
        )
    manifest = {
        "axis": {"cohort_contract_sha256": contract_sha},
        "production_authority": {"task_bindings": bindings},
    }
    sealed = SimpleNamespace(
        cohort=cohort,
        authority=SimpleNamespace(
            config=SimpleNamespace(run_output_root=run_root),
            contract_bytes=(b"{}\n",),
        ),
        derived=SimpleNamespace(
            manifest_bytes=postprocess._canonical_json(manifest) + b"\n",  # noqa: SLF001
        ),
    )

    def validate_snapshot(
        task_dir,
        _contract,
        *,
        bmr=None,
        directory_fd=None,
        **_kwargs,
    ):
        del bmr
        single, pairwise, _manifest = runner._read_task_output_snapshot(  # noqa: SLF001
            task_dir,
            require_manifest=True,
            directory_fd=directory_fd,
        )
        return {
            "single_gene_sha256": hashlib.sha256(single).hexdigest(),
            "pairwise_sha256": hashlib.sha256(pairwise).hexdigest(),
        }

    monkeypatch.setattr(runner, "validate_task_output", validate_snapshot)
    postprocess._revalidate_sealed_task_bindings(sealed)  # noqa: SLF001
    pair_path = run_root / "tasks" / cohort / "dig" / "pairwise_interaction_results.csv"
    pair_path.write_bytes(b"swapped-after-derivation\n")

    with pytest.raises(postprocess.D5DerivationError, match="changed"):
        postprocess._revalidate_sealed_task_bindings(sealed)  # noqa: SLF001


def test_q_adjustments_match_closed_form_and_reject_invalid_values():
    values = np.asarray([0.03, 0.001, 0.2, 0.02])
    bh = revision_inference.adjust_q_values(values, method="bh")
    by = revision_inference.adjust_q_values(values, method="by")

    assert bh == pytest.approx([0.04, 0.004, 0.2, 0.04])
    assert by == pytest.approx(bh * math.fsum((1, 1 / 2, 1 / 3, 1 / 4)))
    with pytest.raises(revision_inference.RevisionInferenceError):
        revision_inference.adjust_q_values(np.asarray([np.nan]), method="bh")
    with pytest.raises(revision_inference.RevisionInferenceError):
        revision_inference.adjust_q_values(np.asarray([True]), method="bh")
    with pytest.raises(revision_inference.RevisionInferenceError):
        revision_inference.adjust_q_values(np.asarray([0.1]), method="adaptive")


def _prepare_synthetic_release(tmp_path, monkeypatch, *, output_root=None):
    monkeypatch.setattr(postprocess, "TCGA_COHORTS", ("TEST",))
    for name in ("run", "canonical", "provider"):
        (tmp_path / name).mkdir(exist_ok=True)
    config = _config(
        tmp_path,
        evidence_sha256="b" * 64,
        design_path=(tmp_path / "design.json").resolve(),
    )
    authority_bytes = b'{"authority":"synthetic-release"}\n'
    authority = SimpleNamespace(
        authority_record_bytes=authority_bytes,
        authority_sha256=hashlib.sha256(authority_bytes).hexdigest(),
    )
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    sealed = SimpleNamespace(
        derived=derived,
        publication_binding_sha256="c" * 64,
    )
    monkeypatch.setattr(
        postprocess,
        "validate_postprocess_authority",
        lambda _config: authority,
    )
    monkeypatch.setattr(
        postprocess,
        "derive_k500_cohort_conjunction",
        lambda _authority, _cohort: sealed,
    )
    monkeypatch.setattr(
        postprocess,
        "_write_production_cohort_for_release",
        lambda path, receipt: postprocess._write_derived_cohort_family_for_qa(  # noqa: SLF001
            path,
            receipt.derived,
        ),
    )
    return config, output_root or (tmp_path / "release"), derived


def _thaw_test_tree(root):
    if not root.exists():
        return
    if not root.is_symlink():
        root.chmod(0o700 if root.is_dir() else 0o600)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        path.chmod(0o700 if path.is_dir() else 0o600)


def test_complete_release_is_descriptor_pinned_and_read_back(tmp_path, monkeypatch):
    config, output, derived = _prepare_synthetic_release(tmp_path, monkeypatch)

    result = postprocess.run_production_postprocess(config, output)

    assert result["cohorts"] == 1
    assert (output / "TEST" / postprocess.OUTPUT_CSV_NAME).read_bytes() == (
        derived.csv_bytes
    )
    assert (output / "TEST" / postprocess.OUTPUT_MANIFEST_NAME).read_bytes() == (
        derived.manifest_bytes
    )
    assert {path.name for path in output.iterdir()} == {
        "TEST",
        postprocess.AUTHORITY_RECEIPT_NAME,
        postprocess.RELEASE_MANIFEST_NAME,
    }
    _thaw_test_tree(output)


def test_complete_release_rejects_child_directory_swap(tmp_path, monkeypatch):
    config, output, _derived = _prepare_synthetic_release(tmp_path, monkeypatch)
    original = postprocess._validate_complete_release_tree  # noqa: SLF001
    attacked = False

    def swap_child(*args, **kwargs):
        nonlocal attacked
        release_fd = args[0]
        if not attacked and not kwargs["frozen"]:
            attacked = True
            os.rename(
                "TEST",
                "TEST.moved",
                src_dir_fd=release_fd,
                dst_dir_fd=release_fd,
            )
            os.mkdir("TEST", mode=0o700, dir_fd=release_fd)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        postprocess,
        "_validate_complete_release_tree",
        swap_child,
    )

    with pytest.raises(postprocess.D5DerivationError, match="inventory changed"):
        postprocess.run_production_postprocess(config, output)
    assert not output.exists()
    for staging in tmp_path.glob(".release.*.tmp"):
        _thaw_test_tree(staging)


@pytest.mark.parametrize("corruption", ["tamper", "swap", "symlink", "hardlink"])
def test_complete_release_rejects_child_file_corruption(
    tmp_path,
    monkeypatch,
    corruption,
):
    config, output, _derived = _prepare_synthetic_release(tmp_path, monkeypatch)
    original = postprocess._validate_complete_release_tree  # noqa: SLF001
    attacked = False

    def corrupt_child(*args, **kwargs):
        nonlocal attacked
        cohort_pin = args[2][0]
        if not attacked and not kwargs["frozen"]:
            attacked = True
            directory_fd = cohort_pin.descriptor
            name = postprocess.OUTPUT_CSV_NAME
            if corruption == "tamper":
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
                try:
                    os.write(descriptor, b"attacker\n")
                finally:
                    os.close(descriptor)
            elif corruption == "hardlink":
                os.link(
                    name,
                    "attacker-link",
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            else:
                os.rename(
                    name,
                    "original.csv",
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                if corruption == "symlink":
                    os.symlink("original.csv", name, dir_fd=directory_fd)
                else:
                    descriptor = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=directory_fd,
                    )
                    os.close(descriptor)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        postprocess,
        "_validate_complete_release_tree",
        corrupt_child,
    )

    with pytest.raises(postprocess.D5DerivationError):
        postprocess.run_production_postprocess(config, output)
    assert not output.exists()
    for staging in tmp_path.glob(".release.*.tmp"):
        _thaw_test_tree(staging)


@pytest.mark.parametrize("attack", ["outer-root", "published-child"])
def test_complete_release_rejects_post_rename_swaps(
    tmp_path,
    monkeypatch,
    attack,
):
    config, output, _derived = _prepare_synthetic_release(tmp_path, monkeypatch)
    original = postprocess._rename_no_replace_at  # noqa: SLF001

    def swap_after_rename(parent_fd, source_name, destination_name):
        original(parent_fd, source_name, destination_name)
        if destination_name != output.name:
            return
        if attack == "outer-root":
            os.rename(
                destination_name,
                f"{destination_name}.moved",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir(destination_name, mode=0o700, dir_fd=parent_fd)
            return
        release_fd = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(release_fd, 0o700)
            os.rename(
                "TEST",
                "TEST.moved",
                src_dir_fd=release_fd,
                dst_dir_fd=release_fd,
            )
            os.mkdir("TEST", mode=0o500, dir_fd=release_fd)
            os.fchmod(release_fd, 0o500)
        finally:
            os.close(release_fd)

    monkeypatch.setattr(postprocess, "_rename_no_replace_at", swap_after_rename)

    with pytest.raises(postprocess.D5DerivationError):
        postprocess.run_production_postprocess(config, output)
    assert output.exists()
    _thaw_test_tree(output)
    moved = tmp_path / "release.moved"
    if moved.exists():
        _thaw_test_tree(moved)


def test_complete_release_propagates_post_rename_fsync_failure(
    tmp_path,
    monkeypatch,
):
    config, output, _derived = _prepare_synthetic_release(tmp_path, monkeypatch)
    original_rename = postprocess._rename_no_replace_at  # noqa: SLF001
    original_fsync = postprocess.os.fsync
    published = False

    def mark_publication(parent_fd, source_name, destination_name):
        nonlocal published
        original_rename(parent_fd, source_name, destination_name)
        if destination_name == output.name:
            published = True

    def fail_fsync(descriptor):
        if published:
            msg = "synthetic post-rename fsync failure"
            raise OSError(msg)
        return original_fsync(descriptor)

    monkeypatch.setattr(postprocess, "_rename_no_replace_at", mark_publication)
    monkeypatch.setattr(postprocess.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="post-rename"):
        postprocess.run_production_postprocess(config, output)
    assert output.exists()
    _thaw_test_tree(output)


def test_writer_rejects_parent_and_ancestor_swap(tmp_path, monkeypatch):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    parent = tmp_path / "publication-parent"
    parent.mkdir()
    output = parent / "derived"
    original = postprocess._require_stable_publication_parent  # noqa: SLF001
    attacked = False

    def swap_parent(lexical_parent, canonical_parent, parent_fd, expected):
        nonlocal attacked
        if not attacked:
            attacked = True
            canonical_parent.rename(tmp_path / "publication-parent.moved")
            canonical_parent.mkdir()
        return original(lexical_parent, canonical_parent, parent_fd, expected)

    monkeypatch.setattr(
        postprocess,
        "_require_stable_publication_parent",
        swap_parent,
    )

    with pytest.raises(postprocess.D5DerivationError, match="parent"):
        _write_qa_family(output, derived)
    assert not output.exists()
    _thaw_test_tree(tmp_path / "publication-parent.moved")


def test_production_output_alias_inside_source_root_is_rejected(
    tmp_path,
    monkeypatch,
):
    alias = tmp_path / "run-alias"
    alias.symlink_to(tmp_path / "run", target_is_directory=True)
    output = alias / "release"
    config, _output, _derived = _prepare_synthetic_release(
        tmp_path,
        monkeypatch,
        output_root=output,
    )

    with pytest.raises(postprocess.D5DerivationError, match="aliases"):
        postprocess.run_production_postprocess(config, output)
    assert not output.exists()


def test_prewrite_parent_race_creates_nothing_in_immutable_roots(
    tmp_path,
    monkeypatch,
):
    outside = tmp_path / "outside"
    ancestor = outside / "ancestor"
    publication_parent = ancestor / "publication-parent"
    publication_parent.mkdir(parents=True)
    output = publication_parent / "release"
    config, _output, _derived = _prepare_synthetic_release(
        tmp_path,
        monkeypatch,
        output_root=output,
    )
    raced_parent = config.run_output_root / "publication-parent"
    raced_parent.mkdir()
    immutable_roots = (
        config.run_output_root,
        config.canonical_input_root,
        config.provider_input_root,
    )

    def inventory(root):
        return tuple(
            sorted(path.relative_to(root).as_posix() for path in root.rglob("*")),
        )

    before = tuple(inventory(root) for root in immutable_roots)
    original_open = postprocess.os.open
    canonical_parent = publication_parent.resolve(strict=True)
    attacked = False

    def race_parent_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attacked
        if not attacked and dir_fd is None and path == canonical_parent:
            attacked = True
            ancestor.rename(outside / "ancestor.original")
            ancestor.symlink_to(config.run_output_root, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(postprocess.os, "open", race_parent_open)

    with pytest.raises(postprocess.D5DerivationError, match="parent"):
        postprocess.run_production_postprocess(config, output)
    assert attacked
    assert tuple(inventory(root) for root in immutable_roots) == before


def test_cohort_writer_checks_parent_identity_before_first_write(
    tmp_path,
    monkeypatch,
):
    derived = postprocess.derive_conjunction_family(
        _component_tables(),
        axis=_axis(),
        policy=_policy(),
        marginal_validity=postprocess.MarginalValidityEvidence("absent"),
    )
    outside = tmp_path / "outside"
    ancestor = outside / "ancestor"
    publication_parent = ancestor / "publication-parent"
    publication_parent.mkdir(parents=True)
    output = publication_parent / "derived"
    original_open = postprocess.os.open
    canonical_parent = publication_parent.resolve(strict=True)

    def race_parent_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None and path == canonical_parent and publication_parent.exists():
            publication_parent.rename(ancestor / "publication-parent.original")
            publication_parent.mkdir()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(postprocess.os, "open", race_parent_open)

    with pytest.raises(postprocess.D5DerivationError, match="parent"):
        _write_qa_family(output, derived)
    assert list(publication_parent.iterdir()) == []


def test_standalone_production_cohort_publication_is_forbidden(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        postprocess,
        "_require_sealed_derived_family",
        lambda _sealed: None,
    )
    output = tmp_path / "standalone"

    with pytest.raises(postprocess.D5DerivationError, match="Standalone"):
        postprocess.write_derived_cohort_family(output, SimpleNamespace())
    assert not output.exists()


@pytest.mark.parametrize("attack", ["directory-swap", "file-swap"])
def test_component_validation_uses_one_pinned_task_snapshot(
    tmp_path,
    monkeypatch,
    attack,
):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    original_bytes = {
        "single_gene_results.csv": b"sealed-single\n",
        "pairwise_interaction_results.csv": b"sealed-pairwise\n",
        "task_manifest.json": b'{"sealed":true}\n',
    }
    for name, content in original_bytes.items():
        (task_dir / name).write_bytes(content)

    def validate_on_pinned_directory(
        _task_dir,
        _contract,
        *,
        require_manifest=True,
        bmr=None,
        scientific_inputs=None,
        directory_fd=None,
    ):
        del require_manifest, bmr, scientific_inputs
        assert directory_fd is not None
        if attack == "directory-swap":
            task_dir.rename(tmp_path / "task.moved")
            task_dir.mkdir()
            for name in original_bytes:
                (task_dir / name).write_bytes(b"attacker\n")
        else:
            os.rename(
                "pairwise_interaction_results.csv",
                "pairwise.original.csv",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            descriptor = os.open(
                "pairwise_interaction_results.csv",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(descriptor, b"attacker\n")
            finally:
                os.close(descriptor)
        return {
            "single_gene_sha256": hashlib.sha256(
                original_bytes["single_gene_results.csv"],
            ).hexdigest(),
            "pairwise_sha256": hashlib.sha256(
                original_bytes["pairwise_interaction_results.csv"],
            ).hexdigest(),
        }

    monkeypatch.setattr(
        runner,
        "validate_task_output",
        validate_on_pinned_directory,
    )

    with pytest.raises(
        postprocess.D5DerivationError,
        match=r"task validation|changed",
    ):
        postprocess._read_pinned_validated_task_snapshot(  # noqa: SLF001
            task_dir,
            {},
            provider="cbase",
        )

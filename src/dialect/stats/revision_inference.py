"""Pure D5 inference over already validated revision pair components.

The module owns no file or cohort-layout knowledge.  It classifies valid D4 profile
LRT components, constructs the nondirectional three-provider intersection-union
test, and adjusts exactly one complete within-cohort family by both BY and BH.
Direction is a descriptive annotation and cannot change any p- or q-value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.special import gammaincc

from dialect.data.revision_fit_policy import (
    BMR_PROVIDERS,
    COMPONENT_FAILURE_SEMANTICS,
    CONJUNCTION_P_VALUE_COMBINER,
    D5_CONTRACT,
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
    ConjunctionMultiplicityPolicy,
    ConjunctionPolicy,
    DirectionAnnotationPolicy,
    MultiplicityPolicy,
    TestedFamilyPolicy,
)
from dialect.models.interaction import (
    PAIR_EFFECT_IDENTIFIED_STATUS,
    PAIR_EFFECT_RANK_DEFICIENT_STATUS,
    PAIR_EFFECT_UNDERFLOW_STATUS,
    PAIR_SIMPLEX_TOL,
    UNDEFINED_RHO_LRT_TOL,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

COMPONENT_ORDER: Final = ("cbase", "dig", "mutsig")
VALID_PROFILE_LRT: Final = "valid-profile-lrt"
VALID_DEGENERATE_NULL_P_ONE: Final = "valid-degenerate-null-p-one"
COMPONENT_DIRECTIONS: Final = frozenset({"me", "co", "neutral", "unavailable"})
CONSENSUS_DIRECTIONS: Final = frozenset(
    {"unanimous-me", "unanimous-co", "discordant", "unavailable"},
)
PRIMARY_Q_THRESHOLD: Final = 0.01
SENSITIVITY_Q_THRESHOLD: Final = 0.01
DESCRIPTIVE_Q_THRESHOLD: Final = 0.05


class RevisionInferenceError(ValueError):
    """Raised when a complete D5 family cannot be derived fail-closed."""


@dataclass(frozen=True, slots=True)
class ComponentInference:
    """One valid provider-level D4 component for a tested pair."""

    status: str
    p_value: float
    direction: str
    effect_identifiability: str


@dataclass(frozen=True, slots=True)
class CompleteFamilyInference:
    """D5 outputs for one complete within-cohort tested-pair family."""

    conjunction_p_values: tuple[float, ...]
    by_q_values: tuple[float, ...]
    bh_q_values: tuple[float, ...]
    consensus_directions: tuple[str, ...]
    primary_inferential_eligible: bool
    by_primary_threshold_crossings: tuple[bool, ...]
    by_primary_reportable: tuple[bool, ...]
    bh_nominal_threshold_crossings: tuple[bool, ...]
    by_descriptive_threshold_crossings: tuple[bool, ...]
    bh_descriptive_threshold_crossings: tuple[bool, ...]


def classify_component(
    likelihood_ratio: float,
    effect_identifiability: str,
    rho: float | None,
) -> ComponentInference:
    """Classify one D4 component without using rho to alter its p-value."""
    if (
        isinstance(likelihood_ratio, bool)
        or not isinstance(likelihood_ratio, (int, float, np.integer, np.floating))
        or not math.isfinite(float(likelihood_ratio))
        or likelihood_ratio < 0
    ):
        msg = "Profile likelihood ratio must be a finite non-negative number."
        raise RevisionInferenceError(msg)
    likelihood_ratio = float(likelihood_ratio)
    valid_effect_statuses = {
        PAIR_EFFECT_IDENTIFIED_STATUS,
        PAIR_EFFECT_RANK_DEFICIENT_STATUS,
        PAIR_EFFECT_UNDERFLOW_STATUS,
    }
    if (
        not isinstance(effect_identifiability, str)
        or effect_identifiability not in valid_effect_statuses
    ):
        msg = "Effect-identifiability status is outside the frozen vocabulary."
        raise RevisionInferenceError(msg)

    if effect_identifiability != PAIR_EFFECT_IDENTIFIED_STATUS:
        if rho is not None:
            msg = "An unidentified effect cannot provide a rho direction."
            raise RevisionInferenceError(msg)
        return ComponentInference(
            status=VALID_PROFILE_LRT,
            p_value=_profile_lrt_p_value(likelihood_ratio),
            direction="unavailable",
            effect_identifiability=effect_identifiability,
        )

    if rho is None:
        if likelihood_ratio > UNDEFINED_RHO_LRT_TOL:
            msg = "Undefined rho is not a certified degenerate-null boundary."
            raise RevisionInferenceError(msg)
        return ComponentInference(
            status=VALID_DEGENERATE_NULL_P_ONE,
            p_value=1.0,
            direction="unavailable",
            effect_identifiability=effect_identifiability,
        )
    if (
        isinstance(rho, bool)
        or not isinstance(rho, (int, float, np.integer, np.floating))
        or not math.isfinite(float(rho))
        or abs(float(rho)) > 1 + PAIR_SIMPLEX_TOL
    ):
        msg = "Identified rho must be finite and lie in [-1, 1]."
        raise RevisionInferenceError(msg)
    rho = float(rho)
    if rho < 0:
        direction = "me"
    elif rho > 0:
        direction = "co"
    else:
        direction = "neutral"
    return ComponentInference(
        status=VALID_PROFILE_LRT,
        p_value=_profile_lrt_p_value(likelihood_ratio),
        direction=direction,
        effect_identifiability=effect_identifiability,
    )


def derive_complete_family(
    components: Mapping[str, Sequence[ComponentInference]],
    *,
    policy: ConjunctionMultiplicityPolicy,
    marginal_validity_certified: bool,
) -> CompleteFamilyInference:
    """Derive max-p, BY, BH, and descriptive direction for one full family."""
    validate_policy(policy)
    if not isinstance(marginal_validity_certified, bool):
        msg = "Marginal-validity eligibility must be an explicit boolean."
        raise RevisionInferenceError(msg)
    if not isinstance(components, Mapping):
        msg = "D5 components must be a provider mapping."
        raise RevisionInferenceError(msg)
    if set(components) != set(COMPONENT_ORDER) or len(components) != len(
        COMPONENT_ORDER,
    ):
        msg = "D5 requires exactly one cbase, dig, and mutsig component sequence."
        raise RevisionInferenceError(msg)
    lengths = {len(components[provider]) for provider in COMPONENT_ORDER}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        msg = "D5 requires one non-empty, coordinate-aligned complete family."
        raise RevisionInferenceError(msg)
    count = next(iter(lengths))
    for provider in COMPONENT_ORDER:
        for component in components[provider]:
            _validate_component(component)

    conjunction = np.asarray(
        [
            max(components[provider][index].p_value for provider in COMPONENT_ORDER)
            for index in range(count)
        ],
        dtype=np.float64,
    )
    by_q_values = adjust_q_values(conjunction, method="by")
    bh_q_values = adjust_q_values(conjunction, method="bh")
    directions = tuple(
        consensus_direction(
            tuple(
                components[provider][index].direction for provider in COMPONENT_ORDER
            ),
        )
        for index in range(count)
    )
    by_primary = by_q_values <= PRIMARY_Q_THRESHOLD
    bh_nominal = bh_q_values <= SENSITIVITY_Q_THRESHOLD
    by_descriptive = by_q_values <= DESCRIPTIVE_Q_THRESHOLD
    bh_descriptive = bh_q_values <= DESCRIPTIVE_Q_THRESHOLD
    reportable = by_primary & marginal_validity_certified
    return CompleteFamilyInference(
        conjunction_p_values=tuple(float(value) for value in conjunction),
        by_q_values=tuple(float(value) for value in by_q_values),
        bh_q_values=tuple(float(value) for value in bh_q_values),
        consensus_directions=directions,
        primary_inferential_eligible=marginal_validity_certified,
        by_primary_threshold_crossings=tuple(bool(value) for value in by_primary),
        by_primary_reportable=tuple(bool(value) for value in reportable),
        bh_nominal_threshold_crossings=tuple(bool(value) for value in bh_nominal),
        by_descriptive_threshold_crossings=tuple(
            bool(value) for value in by_descriptive
        ),
        bh_descriptive_threshold_crossings=tuple(
            bool(value) for value in bh_descriptive
        ),
    )


def adjust_q_values(p_values: np.ndarray, *, method: str) -> np.ndarray:
    """Return deterministic complete-family BH or BY adjusted p-values."""
    raw_values = np.asarray(p_values)
    if raw_values.dtype == np.dtype(bool):
        msg = "Multiplicity adjustment does not accept boolean p-values."
        raise RevisionInferenceError(msg)
    try:
        values = np.asarray(p_values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "Multiplicity adjustment requires numeric p-values."
        raise RevisionInferenceError(msg) from error
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        msg = "Multiplicity adjustment requires finite p-values in [0, 1]."
        raise RevisionInferenceError(msg)
    if method not in {"bh", "by"}:
        msg = "Multiplicity method must be explicitly BH or BY."
        raise RevisionInferenceError(msg)
    order = np.argsort(values, kind="stable")
    sorted_p = values[order]
    count = len(sorted_p)
    factor = 1.0
    if method == "by":
        factor = math.fsum(1.0 / rank for rank in range(1, count + 1))
    ranks = np.arange(1, count + 1, dtype=np.float64)
    adjusted = sorted_p * count * factor / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def consensus_direction(directions: tuple[str, str, str]) -> str:
    """Return the descriptive three-provider unanimity annotation."""
    if len(directions) != len(COMPONENT_ORDER) or any(
        direction not in COMPONENT_DIRECTIONS for direction in directions
    ):
        msg = "Component direction escaped the exact D5 vocabulary."
        raise RevisionInferenceError(msg)
    if "unavailable" in directions:
        return "unavailable"
    if all(direction == "me" for direction in directions):
        return "unanimous-me"
    if all(direction == "co" for direction in directions):
        return "unanimous-co"
    return "discordant"


def validate_policy(policy: ConjunctionMultiplicityPolicy) -> None:
    """Fail unless a parsed policy is the exact supported D5 v3 contract."""
    if not isinstance(policy, ConjunctionMultiplicityPolicy):
        msg = "D5 inference requires the parsed signed conjunction policy."
        raise RevisionInferenceError(msg)
    if (
        not isinstance(policy.conjunction, ConjunctionPolicy)
        or not isinstance(policy.direction_annotation, DirectionAnnotationPolicy)
        or not isinstance(policy.tested_family, TestedFamilyPolicy)
        or not isinstance(policy.multiplicity, MultiplicityPolicy)
    ):
        msg = "D5 inference requires all parsed typed v3 policy components."
        raise RevisionInferenceError(msg)
    conjunction = policy.conjunction
    direction = policy.direction_annotation
    multiplicity = policy.multiplicity
    expected = (
        conjunction.mode == MAX_P_IUT,
        conjunction.component_order == COMPONENT_ORDER,
        set(conjunction.component_order) == BMR_PROVIDERS,
        conjunction.valid_component_statuses == VALID_CONJUNCTION_COMPONENT_STATUSES,
        conjunction.p_value_combiner == CONJUNCTION_P_VALUE_COMBINER,
        conjunction.invalid_component == INVALID_CONJUNCTION_COMPONENT,
        conjunction.missing_component == MISSING_CONJUNCTION_COMPONENT,
        conjunction.sign_discordance == SIGN_DISCORDANCE_POLICY,
        conjunction.effect_unidentifiable == EFFECT_UNIDENTIFIABLE_POLICY,
        conjunction.direction_affects_p_or_q is False,
        direction.provider_rule == DIRECTION_PROVIDER_RULE,
        direction.undefined_rho_rule == UNDEFINED_RHO_DIRECTION_RULE,
        direction.consensus_rule == DIRECTION_CONSENSUS_RULE,
        direction.reporting_layer == DIRECTION_REPORTING_LAYER,
        direction.directional_fdr_control is False,
        policy.tested_family.top_k == TESTED_FAMILY_TOP_K,
        policy.tested_family.feature_ranking == TESTED_FAMILY_FEATURE_RANKING,
        policy.tested_family.tie_break == TESTED_FAMILY_TIE_BREAK,
        policy.tested_family.provider_support == TESTED_FAMILY_PROVIDER_SUPPORT,
        policy.tested_family.pair_construction == TESTED_FAMILY_PAIR_CONSTRUCTION,
        policy.tested_family.same_base_missense_nonsense
        == TESTED_FAMILY_SAME_BASE_POLICY,
        policy.tested_family.epsilon_pretest_filter == NO_PRETEST_FILTER,
        policy.tested_family.marginal_effect_pretest_filter == NO_PRETEST_FILTER,
        policy.tested_family.family == WITHIN_COHORT_FAMILY,
        multiplicity.primary_method == "by",
        multiplicity.sensitivity_method == "bh",
        multiplicity.primary_q_threshold == PRIMARY_Q_THRESHOLD,
        multiplicity.sensitivity_q_threshold == SENSITIVITY_Q_THRESHOLD,
        multiplicity.descriptive_methods == DESCRIPTIVE_METHODS,
        multiplicity.descriptive_q_threshold == DESCRIPTIVE_Q_THRESHOLD,
        multiplicity.threshold_comparison == INCLUSIVE_THRESHOLD,
        multiplicity.primary_reporting_layer == PRIMARY_REPORTING_LAYER,
        multiplicity.sensitivity_reporting_layer == SENSITIVITY_REPORTING_LAYER,
        multiplicity.descriptive_reporting_layer == DESCRIPTIVE_REPORTING_LAYER,
        policy.component_failure_semantics == COMPONENT_FAILURE_SEMANTICS,
    )
    if not all(expected):
        msg = f"Parsed policy does not match the exact {D5_CONTRACT} contract."
        raise RevisionInferenceError(msg)


def _profile_lrt_p_value(likelihood_ratio: float) -> float:
    p_value = float(gammaincc(0.5, likelihood_ratio / 2.0))
    if not math.isfinite(p_value) or not 0 <= p_value <= 1:
        msg = "Profile LRT did not yield a valid chi-square p-value."
        raise RevisionInferenceError(msg)
    return p_value


def _validate_component(component: ComponentInference) -> None:
    if not isinstance(component, ComponentInference):
        msg = "Complete D5 families require typed component inference records."
        raise RevisionInferenceError(msg)
    if component.status not in VALID_CONJUNCTION_COMPONENT_STATUSES:
        msg = "D5 component status is invalid; p=1 substitution is forbidden."
        raise RevisionInferenceError(msg)
    valid_effect_statuses = {
        PAIR_EFFECT_IDENTIFIED_STATUS,
        PAIR_EFFECT_RANK_DEFICIENT_STATUS,
        PAIR_EFFECT_UNDERFLOW_STATUS,
    }
    if component.effect_identifiability not in valid_effect_statuses:
        msg = "D5 component effect-identifiability status is invalid."
        raise RevisionInferenceError(msg)
    if (
        isinstance(component.p_value, bool)
        or not isinstance(
            component.p_value,
            (int, float, np.integer, np.floating),
        )
        or not math.isfinite(component.p_value)
        or not 0 <= component.p_value <= 1
        or component.direction not in COMPONENT_DIRECTIONS
    ):
        msg = "D5 component p-value or direction is invalid."
        raise RevisionInferenceError(msg)
    if component.status == VALID_DEGENERATE_NULL_P_ONE and (
        component.p_value != 1.0
        or component.effect_identifiability != PAIR_EFFECT_IDENTIFIED_STATUS
        or component.direction != "unavailable"
    ):
        msg = "Degenerate-null status is valid only for the certified p=1 boundary."
        raise RevisionInferenceError(msg)
    if component.effect_identifiability in {
        PAIR_EFFECT_RANK_DEFICIENT_STATUS,
        PAIR_EFFECT_UNDERFLOW_STATUS,
    } and (
        component.status != VALID_PROFILE_LRT or component.direction != "unavailable"
    ):
        msg = "Unidentified effects require profile p and unavailable direction."
        raise RevisionInferenceError(msg)
    if (
        component.effect_identifiability == PAIR_EFFECT_IDENTIFIED_STATUS
        and component.status == VALID_PROFILE_LRT
        and component.direction == "unavailable"
    ):
        msg = "Identified profile-LRT components require a descriptive rho direction."
        raise RevisionInferenceError(msg)

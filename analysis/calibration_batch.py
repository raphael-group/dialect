"""Batched kernel for result-blind profile-LRT calibration.

This module intentionally has no cohort or result-file I/O.  It operates on
freshly simulated :class:`~dialect.models.gene.Gene` pairs and implements the
same two fits as production DIALECT:

* each marginal driver probability is refitted by constrained score bisection;
* the four-state alternative is fit by deterministic simplex coordinate ascent.

The common finite-domain path is vectorized across independent calibration
pairs.  Inputs that require the production log-domain interaction path fall
back to the scalar model classes, so batching does not broaden the numerical
contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import logsumexp

from dialect.models.gene import (
    MARGINAL_FIT_BRACKET_WIDTH_TOL,
    MARGINAL_FIT_FIXED_POINT_TOL,
    MARGINAL_FIT_KKT_TOL,
    MARGINAL_FIT_MAX_ITER,
    Gene,
)
from dialect.models.interaction import (
    LRT_NESTEDNESS_TOL,
    PAIR_EFFECT_IDENTIFIED_STATUS,
    PAIR_FIT_KKT_TOL,
    PAIR_FIT_MAX_ITER,
    PAIR_IDENTIFIABILITY_RTOL,
    Interaction,
)

_LINE_SEARCH_ITERATIONS: Final = 80
_PAIR_AFFINE_DIMENSION: Final = 3
_GAIN_RECHECK_TOL: Final = 1e-12
_LRT_RECONCILIATION_TOL: Final = 1e-8
_RANK_RECONCILIATION_TOL: Final = 1e-14
_CALIBRATION_LRT_CRITICAL_VALUES: Final = np.asarray(
    (3.841458820694124, 6.6348966010212145),
)


@dataclass(frozen=True, slots=True)
class BatchProfileLrtFit:
    """Pair-aligned outputs from the batched calibration fit."""

    likelihood_ratio: np.ndarray
    reportable: np.ndarray
    marginal_pi: np.ndarray
    scalar_fallback: np.ndarray


def _validate_component_batch(components: np.ndarray) -> np.ndarray:
    """Return a validated ``(pair, gene, sample, state)`` float array."""
    values = np.asarray(components, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[1] != 2
        or values.shape[2] == 0
        or values.shape[3] != 2
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or (values.sum(axis=3) == 0).any()
    ):
        msg = (
            "Marginal components must have shape (pair, 2, sample, 2) and "
            "finite supported probabilities."
        )
        raise ValueError(msg)
    return values


def _center_marginal_components(
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-scale marginal components and flag unrepresentable finite ratios."""
    logs = np.full(components.shape, -np.inf, dtype=np.float64)
    positive = components > 0
    logs[positive] = np.log(components[positive])
    scales = np.max(logs, axis=3)
    centered_logs = logs - scales[..., None]
    centered = np.exp(centered_logs)
    lost_positive = np.isfinite(centered_logs) & (centered == 0)
    return centered, lost_positive.any(axis=(2, 3))


def _marginal_score(components: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Evaluate all total marginal scores on row-scaled components."""
    delta = components[:, :, 1] - components[:, :, 0]
    mixtures = components[:, :, 0] + (pi[:, None] * delta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        terms = delta / mixtures
    return np.sum(terms, axis=1)


def _marginal_fixed_point_residual(
    components: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    """Evaluate the scalar fitter's driver-responsibility certificate."""
    result = np.zeros(len(pi), dtype=np.float64)
    interior = (pi > 0) & (pi < 1)
    if not interior.any():
        return result
    selected = components[interior]
    selected_pi = pi[interior]
    mixtures = selected[:, :, 0] + (
        selected_pi[:, None] * (selected[:, :, 1] - selected[:, :, 0])
    )
    responsibilities = (selected_pi[:, None] * selected[:, :, 1]) / mixtures
    responsibilities = np.minimum(responsibilities, 1.0)
    result[interior] = np.abs(
        np.sum(responsibilities, axis=1) / selected.shape[1] - selected_pi,
    )
    return result


def _marginal_certificates(
    components: np.ndarray,
    pi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-point and total-score KKT residuals for every row."""
    score = _marginal_score(components, pi)
    fixed = _marginal_fixed_point_residual(components, pi)
    kkt = np.abs(score)
    at_zero = pi == 0
    at_one = pi == 1
    kkt[at_zero] = np.maximum(score[at_zero], 0.0)
    kkt[at_one] = np.maximum(-score[at_one], 0.0)
    return fixed, kkt


def _fit_marginals(components: np.ndarray) -> np.ndarray:
    """Refit independent marginals with batched production-contract bisection."""
    row_count = components.shape[0]
    result = np.full(row_count, np.nan, dtype=np.float64)
    score_zero = _marginal_score(components, np.zeros(row_count))
    zero = score_zero <= 0
    result[zero] = 0.0

    unresolved = ~zero
    score_one = np.full(row_count, np.nan, dtype=np.float64)
    score_one[unresolved] = _marginal_score(
        components[unresolved],
        np.ones(int(unresolved.sum())),
    )
    one = unresolved & (score_one >= 0)
    result[one] = 1.0
    unresolved &= ~one
    if not unresolved.any():
        return result
    if not (score_zero[unresolved] > 0).all() or not (
        score_one[unresolved] < 0
    ).all():
        msg = "Batched marginals lack signed concave-score brackets."
        raise ValueError(msg)

    lower = np.zeros(row_count, dtype=np.float64)
    upper = np.ones(row_count, dtype=np.float64)
    for _iteration in range(1, MARGINAL_FIT_MAX_ITER + 1):
        indices = np.flatnonzero(unresolved)
        midpoint = lower[indices] + ((upper[indices] - lower[indices]) / 2)
        score = _marginal_score(components[indices], midpoint)
        exact = score == 0
        if exact.any():
            exact_indices = indices[exact]
            exact_pi = midpoint[exact]
            fixed, kkt = _marginal_certificates(
                components[exact_indices],
                exact_pi,
            )
            certified = (
                (fixed <= MARGINAL_FIT_FIXED_POINT_TOL)
                & (kkt <= MARGINAL_FIT_KKT_TOL)
            )
            if not certified.all():
                msg = "An exact batched marginal score root failed certification."
                raise ValueError(msg)
            result[exact_indices] = exact_pi
            unresolved[exact_indices] = False

        positive_indices = indices[(~exact) & (score > 0)]
        negative_indices = indices[(~exact) & (score < 0)]
        lower[positive_indices] = midpoint[(~exact) & (score > 0)]
        upper[negative_indices] = midpoint[(~exact) & (score < 0)]

        candidates = np.flatnonzero(
            unresolved
            & ((upper - lower) <= MARGINAL_FIT_BRACKET_WIDTH_TOL),
        )
        if candidates.size:
            lower_pi = lower[candidates]
            upper_pi = upper[candidates]
            lower_fixed, lower_kkt = _marginal_certificates(
                components[candidates],
                lower_pi,
            )
            upper_fixed, upper_kkt = _marginal_certificates(
                components[candidates],
                upper_pi,
            )
            lower_valid = (
                (lower_fixed <= MARGINAL_FIT_FIXED_POINT_TOL)
                & (lower_kkt <= MARGINAL_FIT_KKT_TOL)
            )
            upper_valid = (
                (upper_fixed <= MARGINAL_FIT_FIXED_POINT_TOL)
                & (upper_kkt <= MARGINAL_FIT_KKT_TOL)
            )
            any_valid = lower_valid | upper_valid
            if any_valid.any():
                finished = candidates[any_valid]
                choose_lower = lower_valid & (
                    (~upper_valid)
                    | (lower_kkt < upper_kkt)
                    | (
                        (lower_kkt == upper_kkt)
                        & (
                            (lower_fixed < upper_fixed)
                            | (
                                (lower_fixed == upper_fixed)
                                & (lower_pi < upper_pi)
                            )
                        )
                    )
                )
                chosen = np.where(choose_lower, lower_pi, upper_pi)
                result[finished] = chosen[any_valid]
                unresolved[finished] = False
        if not unresolved.any():
            return result

    msg = "Batched marginal score bisection did not converge."
    raise ValueError(msg)


def _pair_centered_components(
    marginal_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build centered four-state products and underflow flags."""
    logs = np.full(marginal_components.shape, -np.inf, dtype=np.float64)
    positive = marginal_components > 0
    logs[positive] = np.log(marginal_components[positive])
    a_logs = logs[:, 0]
    b_logs = logs[:, 1]
    pair_logs = np.stack(
        (
            a_logs[:, :, 0] + b_logs[:, :, 0],
            a_logs[:, :, 0] + b_logs[:, :, 1],
            a_logs[:, :, 1] + b_logs[:, :, 0],
            a_logs[:, :, 1] + b_logs[:, :, 1],
        ),
        axis=2,
    )
    scales = np.max(pair_logs, axis=2)
    centered_logs = pair_logs - scales[:, :, None]
    centered = np.exp(centered_logs)
    lost_positive = np.isfinite(centered_logs) & (centered == 0)
    return centered, centered_logs, lost_positive.any(axis=(1, 2))


def _normalized_gradients(
    components: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Return sample-normalized four-state likelihood gradients."""
    return np.sum(components / probabilities[:, :, None], axis=1) / len(
        probabilities[0],
    )


def _fit_pair_alternatives(
    components: np.ndarray,
    null_taus: np.ndarray,
) -> np.ndarray:
    """Fit all finite-domain alternatives by batched coordinate ascent."""
    batch_size, sample_count, _state_count = components.shape
    taus = np.asarray(null_taus, dtype=np.float64).copy()
    pending = np.ones(batch_size, dtype=bool)
    last_gain = np.zeros(batch_size, dtype=np.float64)

    for iteration in range(PAIR_FIT_MAX_ITER + 1):
        indices = np.flatnonzero(pending)
        selected_components = components[indices]
        selected_taus = taus[indices]
        probabilities = np.sum(
            selected_components * selected_taus[:, None, :],
            axis=2,
        )
        if (probabilities <= 0).any():
            msg = "Batched interaction reached zero observation probability."
            raise ValueError(msg)
        gradients = _normalized_gradients(selected_components, probabilities)
        active = selected_taus > 0
        active_residual = np.max(
            np.where(active, np.abs(gradients - 1), 0.0),
            axis=1,
        )
        inactive_residual = np.max(
            np.where(~active, np.maximum(gradients - 1, 0.0), 0.0),
            axis=1,
        )
        kkt = sample_count * np.maximum(active_residual, inactive_residual)
        converged = kkt <= PAIR_FIT_KKT_TOL
        if converged.any():
            fixed = np.max(
                np.abs(selected_taus * gradients - selected_taus),
                axis=1,
            )
            if (fixed[converged] > PAIR_FIT_KKT_TOL).any():
                msg = "A batched interaction passed KKT but failed fixed-point."
                raise ValueError(msg)
            pending[indices[converged]] = False
        if not pending.any():
            return taus
        if iteration == PAIR_FIT_MAX_ITER:
            break

        move_indices = indices[~converged]
        move_components = selected_components[~converged]
        move_taus = selected_taus[~converged]
        move_probabilities = probabilities[~converged]
        move_gradients = gradients[~converged]
        move_active = active[~converged]
        target = np.argmax(move_gradients, axis=1)
        donor = np.argmin(
            np.where(move_active, move_gradients, np.inf),
            axis=1,
        )
        if (target == donor).any():
            msg = "A batched KKT violation lacks a feasible ascent coordinate."
            raise ValueError(msg)

        rows = np.arange(len(move_indices))
        upper = move_taus[rows, donor]
        delta = (
            move_components[rows, :, target]
            - move_components[rows, :, donor]
        )
        derivative_zero = np.sum(delta / move_probabilities, axis=1)
        if (~np.isfinite(derivative_zero)).any() or (derivative_zero <= 0).any():
            msg = "A batched interaction selected a non-ascent coordinate."
            raise ValueError(msg)
        endpoint_probabilities = move_probabilities + (upper[:, None] * delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            derivative_upper = np.where(
                (endpoint_probabilities > 0).all(axis=1),
                np.sum(delta / endpoint_probabilities, axis=1),
                -np.inf,
            )
        at_endpoint = derivative_upper >= 0
        transfer = np.zeros(len(move_indices), dtype=np.float64)
        transfer[at_endpoint] = upper[at_endpoint]
        interior = ~at_endpoint
        if interior.any():
            lower_bound = np.zeros(int(interior.sum()), dtype=np.float64)
            upper_bound = upper[interior].copy()
            interior_probabilities = move_probabilities[interior]
            interior_delta = delta[interior]
            for _ in range(_LINE_SEARCH_ITERATIONS):
                midpoint = (lower_bound + upper_bound) / 2
                shifted = interior_probabilities + (
                    midpoint[:, None] * interior_delta
                )
                derivative = np.sum(interior_delta / shifted, axis=1)
                positive = derivative > 0
                lower_bound[positive] = midpoint[positive]
                upper_bound[~positive] = midpoint[~positive]
            transfer[interior] = lower_bound

        unresolved_step = np.ones(len(move_indices), dtype=bool)
        for _ in range(_LINE_SEARCH_ITERATIONS + 1):
            step_rows = np.flatnonzero(unresolved_step)
            candidate = move_taus[step_rows].copy()
            step_transfer = transfer[step_rows]
            step_target = target[step_rows]
            step_donor = donor[step_rows]
            local_rows = np.arange(len(step_rows))
            candidate[local_rows, step_target] += step_transfer
            candidate[local_rows, step_donor] -= step_transfer
            endpoint_step = step_transfer == upper[step_rows]
            candidate[local_rows[endpoint_step], step_donor[endpoint_step]] = 0.0
            changed = np.any(candidate != move_taus[step_rows], axis=1)
            probability_ratio = (
                step_transfer[:, None] * delta[step_rows]
            ) / move_probabilities[step_rows]
            valid_ratio = (probability_ratio > -1).all(axis=1)
            gains = np.sum(np.log1p(probability_ratio), axis=1)
            uncertain = np.abs(gains) <= _GAIN_RECHECK_TOL
            for local_index in np.flatnonzero(uncertain & valid_ratio):
                gains[local_index] = math.fsum(
                    np.log1p(probability_ratio[local_index]),
                )
            accepted = changed & valid_ratio & np.isfinite(gains) & (gains >= 0)
            if accepted.any():
                accepted_rows = step_rows[accepted]
                taus[move_indices[accepted_rows]] = candidate[accepted]
                last_gain[move_indices[accepted_rows]] = gains[accepted]
                unresolved_step[accepted_rows] = False
            if not unresolved_step.any():
                break
            transfer[unresolved_step] /= 2
        if unresolved_step.any():
            msg = "A batched interaction could not certify a monotone step."
            raise ValueError(msg)

    msg = "Batched interaction coordinate ascent did not converge."
    raise ValueError(msg)


def _profile_likelihood_ratios(
    centered_logs: np.ndarray,
    fitted_taus: np.ndarray,
    null_taus: np.ndarray,
) -> np.ndarray:
    """Compute production-form profile LRTs, retaining ``math.fsum``."""
    fitted_logs = np.full(fitted_taus.shape, -np.inf, dtype=np.float64)
    null_logs = np.full(null_taus.shape, -np.inf, dtype=np.float64)
    fitted_positive = fitted_taus > 0
    null_positive = null_taus > 0
    fitted_logs[fitted_positive] = np.log(fitted_taus[fitted_positive])
    null_logs[null_positive] = np.log(null_taus[null_positive])
    alternative_terms = logsumexp(
        centered_logs + fitted_logs[:, None, :],
        axis=2,
    )
    null_terms = logsumexp(
        centered_logs + null_logs[:, None, :],
        axis=2,
    )
    result = np.empty(len(fitted_taus), dtype=np.float64)
    for index, differences in enumerate(alternative_terms - null_terms):
        statistic = 2 * math.fsum(differences)
        if statistic < -LRT_NESTEDNESS_TOL:
            msg = "A batched alternative violates profile-LRT nestedness."
            raise ValueError(msg)
        result[index] = max(float(statistic), 0.0)
    return result


def _affine_reportability(
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the affine-rank rule and flag rows close to its decision boundary."""
    contrasts = components[:, :, 1:] - components[:, :, [0]]
    singular_values = np.linalg.svd(contrasts, compute_uv=False)
    leading = singular_values[:, 0]
    relative = np.divide(
        singular_values,
        leading[:, None],
        out=np.zeros_like(singular_values),
        where=leading[:, None] != 0,
    )
    rank = np.count_nonzero(relative > PAIR_IDENTIFIABILITY_RTOL, axis=1)
    ambiguous = np.any(
        np.abs(relative - PAIR_IDENTIFIABILITY_RTOL)
        <= _RANK_RECONCILIATION_TOL,
        axis=1,
    )
    return rank == _PAIR_AFFINE_DIMENSION, ambiguous


def _fit_fast_components(
    marginal_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one all-fast-path component batch."""
    centered_marginals, lost_marginal = _center_marginal_components(
        marginal_components,
    )
    if lost_marginal.any():
        msg = "Fast marginal batch contains an underflow-domain row."
        raise ValueError(msg)
    flat_marginals = centered_marginals.reshape(
        -1,
        centered_marginals.shape[2],
        2,
    )
    flat_pi = _fit_marginals(flat_marginals)
    pi = flat_pi.reshape(-1, 2)
    pair_components, pair_logs, lost_pair = _pair_centered_components(
        marginal_components,
    )
    if lost_pair.any():
        msg = "Fast pair batch contains a log-domain row."
        raise ValueError(msg)
    null_taus = np.column_stack(
        (
            (1 - pi[:, 0]) * (1 - pi[:, 1]),
            (1 - pi[:, 0]) * pi[:, 1],
            pi[:, 0] * (1 - pi[:, 1]),
            pi[:, 0] * pi[:, 1],
        ),
    )
    fitted_taus = _fit_pair_alternatives(pair_components, null_taus)
    reportable, ambiguous_rank = _affine_reportability(pair_components)
    return (
        _profile_likelihood_ratios(pair_logs, fitted_taus, null_taus),
        reportable,
        pi,
        ambiguous_rank,
    )


def _fit_scalar_pair(pair: tuple[Gene, Gene]) -> tuple[float, bool, np.ndarray]:
    """Run the unchanged production classes for one fallback pair."""
    gene_a, gene_b = pair
    gene_a.estimate_pi_with_mle()
    gene_b.estimate_pi_with_mle()
    interaction = Interaction(gene_a, gene_b)
    interaction.estimate_tau_with_coordinate_ascent()
    return (
        float(interaction.likelihood_ratio),
        interaction.effect_identifiability_status() == PAIR_EFFECT_IDENTIFIED_STATUS,
        np.asarray((gene_a.pi, gene_b.pi), dtype=np.float64),
    )


def fit_gene_pairs_batched(
    pairs: list[tuple[Gene, Gene]] | tuple[tuple[Gene, Gene], ...],
) -> BatchProfileLrtFit:
    """Refit and evaluate independent calibration pairs in one numeric batch.

    The returned arrays preserve input order.  This high-level kernel accepts
    the same ``Gene`` objects the scalar path would fit.  It only vectorizes rows
    with a common sample count and representable centered components; all other
    rows use the unchanged scalar classes.
    """
    if not pairs:
        empty_float = np.empty(0, dtype=np.float64)
        return BatchProfileLrtFit(
            likelihood_ratio=empty_float,
            reportable=np.empty(0, dtype=bool),
            marginal_pi=np.empty((0, 2), dtype=np.float64),
            scalar_fallback=np.empty(0, dtype=bool),
        )
    sample_count = len(pairs[0][0].samples)
    components: list[np.ndarray | None] = []
    preliminary_fallback = np.zeros(len(pairs), dtype=bool)
    for index, (gene_a, gene_b) in enumerate(pairs):
        if (
            len(gene_a.samples) != sample_count
            or len(gene_b.samples) != sample_count
            or not np.array_equal(gene_a.samples, gene_b.samples)
        ):
            preliminary_fallback[index] = True
            components.append(None)
            continue
        components.append(
            np.stack(
                (gene_a.component_probabilities(), gene_b.component_probabilities()),
            ),
        )

    lrt = np.empty(len(pairs), dtype=np.float64)
    reportable = np.empty(len(pairs), dtype=bool)
    pi = np.empty((len(pairs), 2), dtype=np.float64)
    fallback = preliminary_fallback.copy()
    candidate_indices = np.flatnonzero(~fallback)
    if candidate_indices.size:
        candidate_components = _validate_component_batch(
            np.stack([components[index] for index in candidate_indices]),
        )
        _, lost_marginal = _center_marginal_components(candidate_components)
        _, _, lost_pair = _pair_centered_components(candidate_components)
        fallback[candidate_indices[lost_marginal.any(axis=1) | lost_pair]] = True
        fast_indices = np.flatnonzero(~fallback)
        if fast_indices.size:
            fast_components = _validate_component_batch(
                np.stack([components[index] for index in fast_indices]),
            )
            (
                fast_lrt,
                fast_reportable,
                fast_pi,
                ambiguous_rank,
            ) = _fit_fast_components(
                fast_components,
            )
            lrt[fast_indices] = fast_lrt
            reportable[fast_indices] = fast_reportable
            pi[fast_indices] = fast_pi
            ambiguous_lrt = np.any(
                np.abs(
                    fast_lrt[:, None]
                    - _CALIBRATION_LRT_CRITICAL_VALUES[None, :],
                )
                <= _LRT_RECONCILIATION_TOL,
                axis=1,
            )
            fallback[fast_indices[ambiguous_rank | ambiguous_lrt]] = True

    for index in np.flatnonzero(fallback):
        scalar_lrt, scalar_reportable, scalar_pi = _fit_scalar_pair(pairs[index])
        lrt[index] = scalar_lrt
        reportable[index] = scalar_reportable
        pi[index] = scalar_pi
    return BatchProfileLrtFit(
        likelihood_ratio=lrt,
        reportable=reportable,
        marginal_pi=pi,
        scalar_fallback=fallback,
    )

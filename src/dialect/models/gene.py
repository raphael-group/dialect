"""Single-gene latent-driver model."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

OBSERVATION_SUPPORT_CONTRACT = "latent-state-union-v1"
MARGINAL_FIT_CONTRACT: Final = "deterministic-concave-score-bisection-total-kkt-v1"
MARGINAL_FIT_MAX_ITER: Final = 1000
MARGINAL_FIT_KKT_TOL: Final = 1e-8
MARGINAL_FIT_BRACKET_WIDTH_TOL: Final = 1e-12
MARGINAL_FIT_FIXED_POINT_TOL: Final = 1e-8
MARGINAL_FIT_FLAT_TIE_BREAK: Final = "pi-zero"

_NESTED_LIKELIHOOD_TOL = 1e-8
_LOG_MAX_FLOAT = math.log(np.finfo(float).max)


@dataclass(frozen=True, slots=True)
class _MarginalFit:
    """One fully certified constrained marginal optimum."""

    pi: float
    iterations: int
    bracket_width: float
    fixed_point_residual: float
    kkt_residual: float


@dataclass(frozen=True, slots=True)
class _MarginalFitControls:
    """Validated controls that cannot weaken the production fit contract."""

    max_iter: int
    kkt_tolerance: float
    bracket_width_tolerance: float
    fixed_point_tolerance: float


class Gene:
    """Observed mutation counts and their passenger background distribution."""

    def __init__(
        self,
        name: str,
        samples: list,
        counts: list,
        bmr_pmf: dict | list,
    ) -> None:
        """Build a gene's observed counts and background model.

        ``bmr_pmf`` is either one shared ``{count: probability}`` mapping or a
        sample-aligned list of mappings. The latter supports sample-specific BMRs.
        """
        self.name = name
        self.samples = samples
        self.counts = counts
        self.bmr_pmf = bmr_pmf
        self.pi: float | None = None

        self.cbase_phi = None
        self.cbase_p = None

        self.mle_algorithm: str | None = None
        self.mle_converged: bool = False
        self.mle_iterations: int = 0
        self.mle_bracket_width: float | None = None
        self.mle_fixed_point_residual: float | None = None
        self.mle_kkt_residual: float | None = None
        self.mle_log_likelihood: float | None = None
        self.likelihood_ratio = None
        self.likelihood_ratio_status = None
        self._component_probabilities_cache = None
        self._log_component_probabilities_cache = None
        self._centered_log_component_probabilities_cache = None
        self._component_log_scales_cache = None

    @property
    def bmr_pmfs(self) -> list:
        """Return one background PMF per sample."""
        if isinstance(self.bmr_pmf, list):
            return self.bmr_pmf
        return [self.bmr_pmf] * len(self.counts)

    def __str__(self) -> str:
        """Return a compact human-readable summary."""
        representative = self.bmr_pmfs[0] if self.bmr_pmfs else {}
        bmr_preview = ", ".join(
            f"{key}: {value:.3e}"
            for key, value in itertools.islice(representative.items(), 3)
        )
        pi_info = f"Pi: {self.pi:.3e}" if self.pi is not None else "Pi: Not estimated"
        return (
            f"Gene: {self.name}\n"
            f"Total Mutations: {np.sum(self.counts)}\n"
            f"BMR PMF (preview): {{ {bmr_preview} }}\n"
            f"{pi_info}"
        )

    def calculate_expected_mutations(self) -> float:
        """Return mean per-sample ``E[B]`` across the background PMFs."""
        return float(
            np.mean([sum(k * p for k, p in pmf.items()) for pmf in self.bmr_pmfs]),
        )

    def calculate_expected_total_mutations(self) -> float:
        r"""Return the cohort-total passenger expectation ``sum_i sum_k k p_i(k)``.

        Unlike :meth:`calculate_expected_mutations`, this quantity is on the same
        cohort-total scale as ``sum(self.counts)``. A sample-specific background
        model must supply exactly one PMF per sample; shared PMFs are broadcast by
        :attr:`bmr_pmfs` before the total is evaluated.
        """
        self.verify_bmr_pmf_and_counts_exist()
        pmfs = self.bmr_pmfs
        if len(pmfs) != len(self.counts) or len(self.samples) != len(self.counts):
            msg = (
                f"Gene {self.name} has misaligned samples, counts, and background PMFs."
            )
            raise ValueError(msg)

        expectations = []
        for sample, pmf in zip(self.samples, pmfs, strict=True):
            try:
                expectation = math.fsum(
                    float(count) * float(probability)
                    for count, probability in pmf.items()
                )
            except (AttributeError, OverflowError, TypeError, ValueError) as exc:
                msg = (
                    f"Gene {self.name} has a malformed background PMF for sample "
                    f"{sample}."
                )
                raise ValueError(msg) from exc
            if not math.isfinite(expectation):
                msg = (
                    f"Gene {self.name} has a non-finite passenger expectation for "
                    f"sample {sample}."
                )
                raise ValueError(msg)
            expectations.append(expectation)

        total = math.fsum(expectations)
        if not math.isfinite(total):
            msg = f"Gene {self.name} has a non-finite total passenger expectation."
            raise ValueError(msg)
        return total

    def verify_bmr_pmf_and_counts_exist(self) -> None:
        """Require both a background distribution and observed counts."""
        if self.bmr_pmf is None:
            msg = "BMR PMF is not defined for this gene."
            raise ValueError(msg)
        if self.counts is None:
            msg = "Counts are not defined for this gene."
            raise ValueError(msg)

    def verify_bmr_pmf_contains_all_count_keys(self) -> None:
        """Require every observation to have support under ``C = B + D``."""
        self.component_probabilities()

    def verify_pi_is_valid(self, pi: float) -> None:
        """Require a finite Bernoulli probability, including valid boundaries."""
        if pi is None or not np.isfinite(pi) or not 0 <= pi <= 1:
            msg = f"Invalid pi value: {pi}. Pi must be in the range [0, 1]."
            raise ValueError(msg)

    def component_probabilities(self) -> np.ndarray:
        """Return exact probabilities for the no-driver and driver components.

        The columns are ``P(B=c)`` and ``P(B=c-1)``. Missing PMF keys mean zero
        probability. No pseudo-floor is introduced. If both probabilities are zero,
        the feature fails closed so a runner can exclude it from the native universe.
        """
        self.verify_bmr_pmf_and_counts_exist()
        if self._component_probabilities_cache is not None:
            return self._component_probabilities_cache

        pmfs = self.bmr_pmfs
        if len(pmfs) != len(self.counts) or len(self.samples) != len(self.counts):
            msg = (
                f"Gene {self.name} has misaligned samples, counts, and background PMFs."
            )
            raise ValueError(msg)

        probabilities = []
        for sample, count, pmf in zip(
            self.samples,
            self.counts,
            pmfs,
            strict=True,
        ):
            if not isinstance(count, (int, np.integer)) or count < 0:
                msg = (
                    f"Gene {self.name} has invalid count {count!r} for sample {sample}."
                )
                raise ValueError(msg)
            p_no_driver = float(pmf.get(int(count), 0.0))
            p_driver = float(pmf.get(int(count) - 1, 0.0))
            if (
                not np.isfinite(p_no_driver)
                or not np.isfinite(p_driver)
                or p_no_driver < 0
                or p_driver < 0
                or p_no_driver > 1
                or p_driver > 1
            ):
                msg = (
                    f"Gene {self.name} has an invalid background probability for "
                    f"sample {sample}."
                )
                raise ValueError(msg)
            if p_no_driver == 0 and p_driver == 0:
                msg = (
                    f"Unsupported observation for gene {self.name} at sample "
                    f"{sample}: count {count} has P(B=c)=P(B=c-1)=0."
                )
                raise ValueError(msg)
            probabilities.append((p_no_driver, p_driver))

        self._component_probabilities_cache = np.asarray(probabilities, dtype=float)
        return self._component_probabilities_cache

    def log_component_probabilities(self) -> np.ndarray:
        """Return exact log components, representing true zeros as ``-inf``."""
        if self._log_component_probabilities_cache is not None:
            return self._log_component_probabilities_cache
        components = self.component_probabilities()
        logs = np.full(components.shape, -np.inf, dtype=float)
        positive = components > 0
        logs[positive] = np.log(components[positive])
        self._log_component_probabilities_cache = logs
        return logs

    def _log_likelihood_terms(self, pi: float) -> np.ndarray:
        """Return centered per-observation log mixtures without numeric floors."""
        self.verify_pi_is_valid(pi)
        if (
            self._centered_log_component_probabilities_cache is None
            or self._component_log_scales_cache is None
        ):
            log_components = self.log_component_probabilities()
            scales = np.max(log_components, axis=1)
            if np.any(~np.isfinite(scales)):
                msg = f"Gene {self.name} has an invalid observation likelihood scale."
                raise ValueError(msg)
            self._centered_log_component_probabilities_cache = (
                log_components - scales[:, None]
            )
            self._component_log_scales_cache = scales
        centered = self._centered_log_component_probabilities_cache
        log_no_driver = -np.inf if pi == 1 else math.log1p(-pi)
        log_driver = -np.inf if pi == 0 else math.log(pi)
        return np.logaddexp(
            centered[:, 0] + log_no_driver,
            centered[:, 1] + log_driver,
        )

    def _compute_centered_log_likelihood(self, pi: float) -> float:
        terms = self._log_likelihood_terms(pi)
        if np.any(np.isneginf(terms)):
            return -np.inf
        return float(math.fsum(terms))

    def compute_log_likelihood(self, pi: float) -> float:
        r"""Compute ``sum_i log(P(B_i=c_i)(1-pi)+P(B_i=c_i-1)pi)``."""
        self.verify_pi_is_valid(pi)
        centered = self._compute_centered_log_likelihood(pi)
        scales = self._component_log_scales_cache
        if np.isneginf(centered):
            return -np.inf
        if scales is None:
            msg = f"Gene {self.name} lacks its observation likelihood scales."
            raise RuntimeError(msg)
        return float(math.fsum((centered, math.fsum(scales))))

    def compute_likelihood_ratio(self, pi: float) -> float:
        """Compute the single-gene driver-versus-no-driver LRT."""
        self.verify_pi_is_valid(pi)
        alternative_terms = self._log_likelihood_terms(pi)
        null_terms = self._log_likelihood_terms(0)
        scales = self._component_log_scales_cache
        if scales is None:
            msg = f"Gene {self.name} lacks its observation likelihood scales."
            raise RuntimeError(msg)
        common_offset = math.fsum(scales)
        alternative_log_likelihood = (
            -np.inf
            if np.any(np.isneginf(alternative_terms))
            else float(math.fsum((math.fsum(alternative_terms), common_offset)))
        )
        null_log_likelihood = (
            -np.inf
            if np.any(np.isneginf(null_terms))
            else float(math.fsum((math.fsum(null_terms), common_offset)))
        )
        if not np.isfinite(alternative_log_likelihood):
            msg = f"Gene {self.name} has a non-finite fitted log-likelihood."
            raise ValueError(msg)
        if np.isneginf(null_log_likelihood):
            self.likelihood_ratio = np.inf
            self.likelihood_ratio_status = "infinite-passenger-null-zero-probability"
            return self.likelihood_ratio
        likelihood_ratio = 2 * math.fsum(alternative_terms - null_terms)
        if likelihood_ratio < -_NESTED_LIKELIHOOD_TOL:
            msg = (
                f"Gene {self.name} alternative log-likelihood is below its null "
                "log-likelihood."
            )
            raise ValueError(msg)
        self.likelihood_ratio = max(float(likelihood_ratio), 0.0)
        self.likelihood_ratio_status = "finite"
        return self.likelihood_ratio

    def compute_log_odds_ratio(self, pi: float) -> float:
        """Return the log odds of the fitted driver probability."""
        self.verify_pi_is_valid(pi)
        if pi == 0:
            return -np.inf
        if pi == 1:
            return np.inf
        return float(np.log(pi / (1 - pi)))

    @staticmethod
    def _signed_log_sum(
        positive_log_magnitudes: list[float],
        negative_log_magnitudes: list[float],
    ) -> float:
        """Return a signed sum without exponentiating a dominant term too soon."""
        positive_infinite = any(
            math.isinf(value) and value > 0 for value in positive_log_magnitudes
        )
        negative_infinite = any(
            math.isinf(value) and value > 0 for value in negative_log_magnitudes
        )
        if positive_infinite and negative_infinite:
            msg = "Marginal score has opposing non-finite terms."
            raise ValueError(msg)
        if positive_infinite:
            return np.inf
        if negative_infinite:
            return -np.inf

        all_magnitudes = positive_log_magnitudes + negative_log_magnitudes
        if not all_magnitudes:
            return 0.0
        scale = max(all_magnitudes)
        signed_scaled_sum = math.fsum(
            [math.exp(value - scale) for value in positive_log_magnitudes]
            + [-math.exp(value - scale) for value in negative_log_magnitudes],
        )
        if signed_scaled_sum == 0:
            return 0.0
        log_magnitude = scale + math.log(abs(signed_scaled_sum))
        if log_magnitude > _LOG_MAX_FLOAT:
            return math.copysign(np.inf, signed_scaled_sum)
        return math.copysign(math.exp(log_magnitude), signed_scaled_sum)

    def compute_marginal_score(self, pi: float) -> float:
        r"""Return ``d l(pi) / d pi`` on the total-log-likelihood scale.

        The numerator sign is taken from the exact floating-point components and
        its magnitude is divided by the mixture in log space. This distinguishes a
        true zero from a positive component whose weighted product underflows.
        At either endpoint the returned value is the corresponding one-sided score.
        """
        self.verify_pi_is_valid(pi)
        components = self.component_probabilities()
        log_components = self.log_component_probabilities()
        log_no_driver_weight = -np.inf if pi == 1 else math.log1p(-pi)
        log_driver_weight = -np.inf if pi == 0 else math.log(pi)
        positive: list[float] = []
        negative: list[float] = []
        for (no_driver, driver), (log_no_driver, log_driver) in zip(
            components,
            log_components,
            strict=True,
        ):
            difference = float(driver - no_driver)
            if difference == 0:
                continue
            log_mixture = float(
                np.logaddexp(
                    log_no_driver + log_no_driver_weight,
                    log_driver + log_driver_weight,
                ),
            )
            log_magnitude = math.log(abs(difference)) - log_mixture
            if difference > 0:
                positive.append(log_magnitude)
            else:
                negative.append(log_magnitude)
        score = self._signed_log_sum(positive, negative)
        if math.isnan(score):
            msg = f"Gene {self.name} has an undefined marginal score at pi={pi}."
            raise ValueError(msg)
        return score

    def _posterior_driver_mean(self, pi: float) -> float:
        """Return the stable mean driver responsibility at ``pi``."""
        self.verify_pi_is_valid(pi)
        log_components = self.log_component_probabilities()
        if pi == 0:
            if not np.isfinite(self.compute_log_likelihood(pi)):
                msg = f"Gene {self.name} has zero likelihood at pi=0."
                raise ValueError(msg)
            return 0.0
        if pi == 1:
            if not np.isfinite(self.compute_log_likelihood(pi)):
                msg = f"Gene {self.name} has zero likelihood at pi=1."
                raise ValueError(msg)
            return 1.0

        log_pi = math.log(pi)
        log_one_minus_pi = math.log1p(-pi)
        log_mixtures = np.logaddexp(
            log_components[:, 0] + log_one_minus_pi,
            log_components[:, 1] + log_pi,
        )
        if np.any(~np.isfinite(log_mixtures)):
            msg = f"Gene {self.name} has zero likelihood at pi={pi}."
            raise ValueError(msg)
        log_responsibilities = log_components[:, 1] + log_pi - log_mixtures
        responsibilities = np.exp(log_responsibilities)
        if np.any(np.isnan(responsibilities)):
            msg = f"Gene {self.name} has undefined driver responsibilities."
            raise ValueError(msg)
        responsibilities = np.minimum(responsibilities, 1.0)
        return float(math.fsum(responsibilities) / len(responsibilities))

    def compute_mle_certificates(self, pi: float) -> tuple[float, float]:
        """Return fixed-point and total-score KKT residuals for ``pi``."""
        self.verify_pi_is_valid(pi)
        score = self.compute_marginal_score(pi)
        fixed_point_residual = abs(self._posterior_driver_mean(pi) - pi)
        if pi == 0:
            kkt_residual = max(score, 0.0)
        elif pi == 1:
            kkt_residual = max(-score, 0.0)
        else:
            kkt_residual = abs(score)
        if not math.isfinite(fixed_point_residual) or math.isnan(kkt_residual):
            msg = f"Gene {self.name} has non-finite marginal fit certificates."
            raise ValueError(msg)
        return float(fixed_point_residual), float(kkt_residual)

    @staticmethod
    def _validate_fit_controls(
        *,
        max_iter: int,
        kkt_tol: float,
        bracket_width_tol: float,
        fixed_point_tol: float,
    ) -> None:
        """Reject controls that weaken the frozen production contract."""
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer))
            or max_iter <= 0
            or max_iter > MARGINAL_FIT_MAX_ITER
        ):
            msg = (
                "Marginal-fit max_iter must be a positive integer no greater than "
                "the frozen production bound."
            )
            raise ValueError(msg)
        controls = (
            (kkt_tol, MARGINAL_FIT_KKT_TOL, "kkt_tol"),
            (
                bracket_width_tol,
                MARGINAL_FIT_BRACKET_WIDTH_TOL,
                "bracket_width_tol",
            ),
            (
                fixed_point_tol,
                MARGINAL_FIT_FIXED_POINT_TOL,
                "fixed_point_tol",
            ),
        )
        for value, frozen_bound, name in controls:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or value <= 0
                or value > frozen_bound
            ):
                msg = f"Marginal-fit {name} exceeds its frozen positive bound."
                raise ValueError(msg)

    def _reset_mle_fit(self) -> None:
        """Clear every published marginal-fit field before a fresh attempt."""
        self.pi = None
        self.mle_algorithm = None
        self.mle_converged = False
        self.mle_iterations = 0
        self.mle_bracket_width = None
        self.mle_fixed_point_residual = None
        self.mle_kkt_residual = None
        self.mle_log_likelihood = None

    def _certified_candidate(
        self,
        pi: float,
        *,
        iterations: int,
        bracket_width: float,
        kkt_tol: float,
        fixed_point_tol: float,
    ) -> _MarginalFit | None:
        """Return diagnostics only when one representable point is certified."""
        fixed_point_residual, kkt_residual = self.compute_mle_certificates(pi)
        if (
            kkt_residual > kkt_tol
            or fixed_point_residual > fixed_point_tol
            or not np.isfinite(self.compute_log_likelihood(pi))
        ):
            return None
        return _MarginalFit(
            pi=pi,
            iterations=iterations,
            bracket_width=bracket_width,
            fixed_point_residual=fixed_point_residual,
            kkt_residual=kkt_residual,
        )

    def _fit_marginal_by_score_bisection(
        self,
        *,
        max_iter: int,
        kkt_tol: float,
        bracket_width_tol: float,
        fixed_point_tol: float,
    ) -> _MarginalFit:
        """Solve the concave one-dimensional likelihood and certify the result."""
        components = self.component_probabilities()
        if not len(components):
            msg = f"Gene {self.name} has no observations to fit."
            raise ValueError(msg)

        score_zero = self.compute_marginal_score(0.0)
        if score_zero <= 0:
            return self._require_certified_boundary(
                0.0,
                kkt_tol=kkt_tol,
                fixed_point_tol=fixed_point_tol,
            )

        score_one = self.compute_marginal_score(1.0)
        if score_one >= 0:
            return self._require_certified_boundary(
                1.0,
                kkt_tol=kkt_tol,
                fixed_point_tol=fixed_point_tol,
            )
        if not score_zero > 0 or not score_one < 0:
            msg = f"Gene {self.name} lacks a signed concave-score bracket."
            raise ValueError(msg)

        return self._bisect_signed_score_bracket(
            lower_score=score_zero,
            upper_score=score_one,
            controls=_MarginalFitControls(
                max_iter=max_iter,
                kkt_tolerance=kkt_tol,
                bracket_width_tolerance=bracket_width_tol,
                fixed_point_tolerance=fixed_point_tol,
            ),
        )

    def _require_certified_boundary(
        self,
        pi: float,
        *,
        kkt_tol: float,
        fixed_point_tol: float,
    ) -> _MarginalFit:
        """Return a certified endpoint or fail without publishing a row."""
        candidate = self._certified_candidate(
            pi,
            iterations=0,
            bracket_width=0.0,
            kkt_tol=kkt_tol,
            fixed_point_tol=fixed_point_tol,
        )
        if candidate is None:
            msg = f"Gene {self.name} failed its pi={pi:g} marginal certificate."
            raise ValueError(msg)
        return candidate

    def _bisect_signed_score_bracket(
        self,
        *,
        lower_score: float,
        upper_score: float,
        controls: _MarginalFitControls,
    ) -> _MarginalFit:
        """Bisect a strict signed bracket until one float is fully certified."""
        lower = 0.0
        upper = 1.0
        for iteration in range(1, controls.max_iter + 1):
            midpoint = lower + ((upper - lower) / 2)
            if midpoint in (lower, upper):
                msg = (
                    f"Gene {self.name} has no representable point satisfying its "
                    "marginal fit certificates."
                )
                raise ValueError(msg)
            midpoint_score = self.compute_marginal_score(midpoint)
            if midpoint_score == 0:
                return self._require_certified_score_root(
                    midpoint,
                    iterations=iteration,
                    kkt_tol=controls.kkt_tolerance,
                    fixed_point_tol=controls.fixed_point_tolerance,
                )
            if midpoint_score > 0:
                lower = midpoint
                lower_score = midpoint_score
            else:
                upper = midpoint
                upper_score = midpoint_score
            if not lower_score > 0 or not upper_score < 0:
                msg = f"Gene {self.name} lost its signed concave-score bracket."
                raise ValueError(msg)

            candidate = self._best_certified_bracket_point(
                lower,
                upper,
                iterations=iteration,
                controls=controls,
            )
            if candidate is not None:
                return candidate

        msg = (
            f"Gene {self.name} failed to find a representable certified marginal "
            f"MLE within {controls.max_iter} score-bisection iterations."
        )
        raise ValueError(msg)

    def _require_certified_score_root(
        self,
        pi: float,
        *,
        iterations: int,
        kkt_tol: float,
        fixed_point_tol: float,
    ) -> _MarginalFit:
        """Certify an exactly represented zero-score singleton bracket."""
        candidate = self._certified_candidate(
            pi,
            iterations=iterations,
            bracket_width=0.0,
            kkt_tol=kkt_tol,
            fixed_point_tol=fixed_point_tol,
        )
        if candidate is None:
            msg = (
                f"Gene {self.name} has an exact score root that failed its "
                "marginal fit certificates."
            )
            raise ValueError(msg)
        return candidate

    def _best_certified_bracket_point(
        self,
        lower: float,
        upper: float,
        *,
        iterations: int,
        controls: _MarginalFitControls,
    ) -> _MarginalFit | None:
        """Return the strongest certified interior endpoint of a narrow bracket."""
        bracket_width = upper - lower
        if bracket_width > controls.bracket_width_tolerance:
            return None
        candidates = [
            candidate
            for pi in (lower, upper)
            if pi not in {0.0, 1.0}
            if (
                candidate := self._certified_candidate(
                    pi,
                    iterations=iterations,
                    bracket_width=bracket_width,
                    kkt_tol=controls.kkt_tolerance,
                    fixed_point_tol=controls.fixed_point_tolerance,
                )
            )
            is not None
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.kkt_residual,
                item.fixed_point_residual,
                item.pi,
            ),
        )

    def _validate_mle_diagnostics(self) -> None:
        """Validate the shape and frozen bounds of published diagnostics."""
        if (
            isinstance(self.mle_iterations, bool)
            or not isinstance(self.mle_iterations, (int, np.integer))
            or not 0 <= self.mle_iterations <= MARGINAL_FIT_MAX_ITER
        ):
            msg = f"Gene {self.name} has invalid marginal-fit iterations."
            raise ValueError(msg)
        diagnostics = (
            (
                self.mle_bracket_width,
                MARGINAL_FIT_BRACKET_WIDTH_TOL,
                "bracket width",
            ),
            (
                self.mle_fixed_point_residual,
                MARGINAL_FIT_FIXED_POINT_TOL,
                "fixed-point residual",
            ),
            (self.mle_kkt_residual, MARGINAL_FIT_KKT_TOL, "KKT residual"),
        )
        for value, bound, label in diagnostics:
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or not 0 <= value <= bound
            ):
                msg = f"Gene {self.name} has invalid marginal-fit {label}."
                raise ValueError(msg)

    def validate_mle_fit(self) -> None:
        """Fail closed unless every published marginal-fit field is certified."""
        if self.mle_algorithm != MARGINAL_FIT_CONTRACT:
            msg = f"Gene {self.name} has the wrong marginal-fit algorithm."
            raise ValueError(msg)
        if self.mle_converged is not True:
            msg = f"Gene {self.name} does not have a converged marginal fit."
            raise ValueError(msg)
        self._validate_mle_diagnostics()
        if (
            self.pi is None
            or self.mle_log_likelihood is None
            or not math.isfinite(float(self.mle_log_likelihood))
        ):
            msg = f"Gene {self.name} has invalid marginal-fit likelihood fields."
            raise ValueError(msg)
        self.verify_pi_is_valid(self.pi)
        fixed_point_residual, kkt_residual = self.compute_mle_certificates(self.pi)
        if fixed_point_residual > MARGINAL_FIT_FIXED_POINT_TOL:
            msg = f"Gene {self.name} failed marginal fixed-point replay."
            raise ValueError(msg)
        if kkt_residual > MARGINAL_FIT_KKT_TOL:
            msg = f"Gene {self.name} failed marginal KKT replay."
            raise ValueError(msg)
        if not math.isclose(
            fixed_point_residual,
            float(self.mle_fixed_point_residual),
            rel_tol=0.0,
            abs_tol=np.finfo(float).eps,
        ):
            msg = f"Gene {self.name} has a drifting marginal fixed-point receipt."
            raise ValueError(msg)
        if not math.isclose(
            kkt_residual,
            float(self.mle_kkt_residual),
            rel_tol=0.0,
            abs_tol=np.finfo(float).eps,
        ):
            msg = f"Gene {self.name} has a drifting marginal KKT receipt."
            raise ValueError(msg)
        replayed_log_likelihood = self.compute_log_likelihood(self.pi)
        if replayed_log_likelihood != self.mle_log_likelihood:
            msg = f"Gene {self.name} has a drifting marginal likelihood receipt."
            raise ValueError(msg)

    def estimate_pi_with_mle(
        self,
        *,
        max_iter: int = MARGINAL_FIT_MAX_ITER,
        kkt_tol: float = MARGINAL_FIT_KKT_TOL,
        bracket_width_tol: float = MARGINAL_FIT_BRACKET_WIDTH_TOL,
        fixed_point_tol: float = MARGINAL_FIT_FIXED_POINT_TOL,
    ) -> None:
        """Fit and certify the exact concave constrained-null marginal MLE."""
        self._reset_mle_fit()
        self._validate_fit_controls(
            max_iter=max_iter,
            kkt_tol=kkt_tol,
            bracket_width_tol=bracket_width_tol,
            fixed_point_tol=fixed_point_tol,
        )
        fit = self._fit_marginal_by_score_bisection(
            max_iter=max_iter,
            kkt_tol=kkt_tol,
            bracket_width_tol=bracket_width_tol,
            fixed_point_tol=fixed_point_tol,
        )
        self.pi = fit.pi
        self.mle_algorithm = MARGINAL_FIT_CONTRACT
        self.mle_converged = True
        self.mle_iterations = fit.iterations
        self.mle_bracket_width = fit.bracket_width
        self.mle_fixed_point_residual = fit.fixed_point_residual
        self.mle_kkt_residual = fit.kkt_residual
        self.mle_log_likelihood = self.compute_log_likelihood(fit.pi)
        self.validate_mle_fit()

    def estimate_pi_with_optimiziation_using_scipy(self, pi_init: float = 0.5) -> None:
        """Backward-compatible wrapper for the deterministic scalar MLE."""
        del pi_init
        self.estimate_pi_with_mle()

    def estimate_pi_with_em_from_scratch(
        self,
        max_iter: int = 1000,
        tol: float = 1e-12,
        pi_init: float = 0.5,
        n_inits: int = 10,
        seed: int = 0,
    ) -> None:
        """Backward-compatible entry point using the exact deterministic MLE."""
        del pi_init, n_inits, seed
        self.estimate_pi_with_mle(max_iter=max_iter, bracket_width_tol=tol)
        self.em_n_inits_used = 1

    def estimate_pi_with_em_using_pomegranate(self) -> None:
        """Raise because the optional pomegranate implementation does not exist."""
        msg = "EM algorithm not implemented yet."
        raise NotImplementedError(msg)

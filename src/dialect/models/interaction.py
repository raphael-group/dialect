"""Pairwise latent-driver interaction model."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix

from dialect.models.gene import Gene

LRT_CONTRACT = "driver-independence-constrained-mle-v1"
PAIR_FIT_CONTRACT = "deterministic-simplex-coordinate-ascent-v1"
PAIR_FIT_KKT_TOL = 1e-8
RHO_CONTRACT = "marshall-olkin-finite-or-degenerate-null-v1"
UNDEFINED_RHO_LRT_TOL = 1e-8
CONTINGENCY_TABLE_CONTRACT = "observed-binary-cells-00-01-10-11-v1"
LOG_ODDS_RATIO_CONTRACT = "conventional-latent-odds-00x11-over-01x10-v1"

_NESTED_LIKELIHOOD_TOL = 1e-8
_SIMPLEX_TOL = 1e-8
_LINE_SEARCH_ITERATIONS = 80


def compute_marshall_olkin_rho(
    taus: list | tuple | np.ndarray,
) -> float | None:
    """Return a numerically stable Marshall-Olkin correlation for a 2x2 table."""
    values = np.asarray(taus, dtype=float)
    if (
        values.shape != (4,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.isclose(values.sum(), 1, atol=_SIMPLEX_TOL)
    ):
        msg = "Cannot compute rho from invalid tau simplex weights."
        raise ValueError(msg)
    tau_00, tau_01, tau_10, tau_11 = values
    tau_0x = tau_00 + tau_01
    tau_1x = tau_10 + tau_11
    tau_x0 = tau_00 + tau_10
    tau_x1 = tau_01 + tau_11
    if any(tau == 0 for tau in (tau_0x, tau_1x, tau_x0, tau_x1)):
        return None

    # Evaluate an equivalent conditional-probability form. Multiplying all four
    # marginals first can underflow even when rho itself is representable.
    log_row_balance = np.log(tau_0x) + np.log(tau_1x)
    log_column_balance = np.log(tau_x0) + np.log(tau_x1)
    if log_row_balance <= log_column_balance:
        scale = np.exp((log_row_balance - log_column_balance) / 2)
        contrast = (tau_11 / tau_1x) - (tau_01 / tau_0x)
    else:
        scale = np.exp((log_column_balance - log_row_balance) / 2)
        contrast = (tau_11 / tau_x1) - (tau_10 / tau_x0)
    rho = float(scale * contrast)
    if not np.isfinite(rho) or abs(rho) > 1 + _SIMPLEX_TOL:
        msg = "Marshall-Olkin rho is non-finite or outside [-1, 1]."
        raise ValueError(msg)
    if rho == 0 and contrast != 0:
        msg = "Nonzero Marshall-Olkin rho is not representable as a float."
        raise ValueError(msg)
    return rho


class Interaction:
    """Joint latent-driver distribution for a pair of mutation features."""

    def __init__(self, gene_a: Gene, gene_b: Gene) -> None:
        """Build an interaction over sample-aligned gene observations."""
        if not isinstance(gene_a, Gene) or not isinstance(gene_b, Gene):
            msg = "Both inputs must be instances of the Gene class."
            raise TypeError(msg)
        if len(gene_a.samples) != len(gene_b.samples) or not np.array_equal(
            np.asarray(gene_a.samples),
            np.asarray(gene_b.samples),
        ):
            msg = f"Interaction {gene_a.name}:{gene_b.name} has misaligned samples."
            raise ValueError(msg)

        self.gene_a = gene_a
        self.gene_b = gene_b
        self.name = f"{gene_a.name}:{gene_b.name}"
        self.tau_00 = None
        self.tau_01 = None
        self.tau_10 = None
        self.tau_11 = None

        self.discover_me_qval = None
        self.discover_co_qval = None
        self.fishers_me_qval = None
        self.fishers_co_qval = None

        self.fit_algorithm = None
        self.fit_converged = False
        self.fit_iterations = 0
        self.fit_last_log_likelihood_gain = None
        self.fit_fixed_point_residual = None
        self.fit_kkt_residual = None
        # Backward-compatible aliases. New result schemas use the truthful generic
        # ``fit_*`` metadata because the production optimizer is not EM.
        self.em_converged = False
        self.em_iterations = 0
        self.em_n_inits_used = 0
        self.em_final_log_likelihood_increment = None
        self.em_fixed_point_residual = None
        self.em_kkt_residual = None
        self.null_log_likelihood = None
        self.alternative_log_likelihood = None
        self.likelihood_ratio = None
        self._component_probabilities_cache = None

    def __str__(self) -> str:
        """Return a compact human-readable summary."""
        taus_info = (
            f"tau_00={self.tau_00:.3e}, tau_01={self.tau_01:.3e}, "
            f"tau_10={self.tau_10:.3e}, tau_11={self.tau_11:.3e}"
            if None not in (self.tau_00, self.tau_01, self.tau_10, self.tau_11)
            else "Tau values not estimated"
        )
        pi_a = (
            f"{self.gene_a.pi:.3e}"
            if self.gene_a.pi is not None
            else "Not estimated"
        )
        pi_b = (
            f"{self.gene_b.pi:.3e}"
            if self.gene_b.pi is not None
            else "Not estimated"
        )
        cm = self.compute_contingency_table()
        return (
            f"Interaction: {self.name}\n"
            f"Gene A: {self.gene_a.name} (Pi: {pi_a})\n"
            f"Gene B: {self.gene_b.name} (Pi: {pi_b})\n"
            f"Tau Parameters: {taus_info}\n"
            f"Contingency Table (00, 01 / 10, 11):\n"
            f"[[{cm[0, 0]} {cm[0, 1]}]\n [{cm[1, 0]} {cm[1, 1]}]]"
        )

    def compute_contingency_table(self) -> np.ndarray:
        """Return observed cells as ``[[n00, n01], [n10, n11]]``."""
        gene_a_mutations = (np.asarray(self.gene_a.counts) > 0).astype(int)
        gene_b_mutations = (np.asarray(self.gene_b.counts) > 0).astype(int)
        return confusion_matrix(gene_a_mutations, gene_b_mutations, labels=[0, 1])

    def get_set_of_cooccurring_samples(self) -> list:
        """Return sample names with observed mutations in both features."""
        return sorted(
            self.gene_a.samples[index]
            for index in range(len(self.gene_a.samples))
            if self.gene_a.counts[index] > 0 and self.gene_b.counts[index] > 0
        )

    def compute_fisher_pvalues(self) -> tuple[float, float]:
        """Return one-sided Fisher p-values for ME and CO."""
        cross_tab = self.compute_contingency_table()
        _, me_pval = fisher_exact(cross_tab, alternative="less")
        _, co_pval = fisher_exact(cross_tab, alternative="greater")
        return float(me_pval), float(co_pval)

    def verify_bmr_pmf_and_counts_exist(self) -> None:
        """Require both genes to have counts and native BMR support."""
        self.gene_a.component_probabilities()
        self.gene_b.component_probabilities()

    def verify_taus_are_valid(
        self,
        taus: list | tuple | np.ndarray,
        tol: float = _SIMPLEX_TOL,
    ) -> None:
        """Require finite weights on the probability simplex."""
        values = np.asarray(taus, dtype=float)
        if (
            values.shape != (4,)
            or not np.all(np.isfinite(values))
            or np.any(values < -tol)
            or np.any(values > 1 + tol)
            or not np.isclose(values.sum(), 1, atol=tol)
        ):
            msg = "Invalid tau parameters: expected four finite simplex weights."
            raise ValueError(msg)

    def verify_pi_values(self, pi_a: float, pi_b: float) -> None:
        """Require fitted driver probabilities for both genes."""
        self.gene_a.verify_pi_is_valid(pi_a)
        self.gene_b.verify_pi_is_valid(pi_b)

    def component_probabilities(self) -> np.ndarray:
        """Return exact pair-component probabilities in 00, 01, 10, 11 order."""
        if self._component_probabilities_cache is not None:
            return self._component_probabilities_cache
        a_components = self.gene_a.component_probabilities()
        b_components = self.gene_b.component_probabilities()
        components = np.column_stack(
            (
                a_components[:, 0] * b_components[:, 0],
                a_components[:, 0] * b_components[:, 1],
                a_components[:, 1] * b_components[:, 0],
                a_components[:, 1] * b_components[:, 1],
            ),
        )
        unsupported = np.flatnonzero(components.sum(axis=1) == 0)
        if unsupported.size:
            index = int(unsupported[0])
            msg = (
                f"Unsupported observation for interaction {self.name} at sample "
                f"{self.gene_a.samples[index]}."
            )
            raise ValueError(msg)
        self._component_probabilities_cache = components
        return components

    def compute_joint_probability(self, tau: float, u: int, v: int) -> np.ndarray:
        """Return each observation's weighted ``(D_A=u,D_B=v)`` probability."""
        if u not in (0, 1) or v not in (0, 1):
            msg = "Driver states u and v must each be 0 or 1."
            raise ValueError(msg)
        return float(tau) * self.component_probabilities()[:, (2 * u) + v]

    def compute_total_probability(
        self,
        tau_00: float,
        tau_01: float,
        tau_10: float,
        tau_11: float,
    ) -> np.ndarray:
        """Return each observation's probability under the four-component mixture."""
        taus = np.asarray((tau_00, tau_01, tau_10, tau_11), dtype=float)
        self.verify_taus_are_valid(taus)
        return np.sum(self.component_probabilities() * taus, axis=1)

    def compute_log_likelihood(self, taus: list | tuple | np.ndarray) -> float:
        """Compute the exact pair log-likelihood without pseudo-probability floors."""
        self.verify_taus_are_valid(taus)
        probabilities = np.sum(
            self.component_probabilities() * np.asarray(taus, dtype=float),
            axis=1,
        )
        if np.any(probabilities <= 0):
            return -np.inf
        return float(math.fsum(np.log(probabilities)))

    def compute_independence_taus(self) -> tuple[float, float, float, float]:
        """Return the constrained-null weights from the marginal MLEs."""
        if not self.gene_a.mle_converged:
            self.gene_a.estimate_pi_with_mle()
        if not self.gene_b.mle_converged:
            self.gene_b.estimate_pi_with_mle()
        pi_a, pi_b = self.gene_a.pi, self.gene_b.pi
        self.verify_pi_values(pi_a, pi_b)
        return (
            (1 - pi_a) * (1 - pi_b),
            (1 - pi_a) * pi_b,
            pi_a * (1 - pi_b),
            pi_a * pi_b,
        )

    def compute_likelihood_ratio(
        self,
        taus: list | tuple | np.ndarray,
    ) -> float:
        """Compute the profile LRT against fitted driver independence."""
        alternative_log_likelihood = self.compute_log_likelihood(taus)
        null_log_likelihood = self.compute_log_likelihood(
            self.compute_independence_taus(),
        )
        if not np.isfinite(alternative_log_likelihood) or not np.isfinite(
            null_log_likelihood,
        ):
            msg = f"Interaction {self.name} has a non-finite fitted likelihood."
            raise ValueError(msg)

        likelihood_ratio = 2 * (alternative_log_likelihood - null_log_likelihood)
        if likelihood_ratio < -_NESTED_LIKELIHOOD_TOL:
            msg = (
                f"Interaction {self.name} violates nestedness: alternative "
                f"log-likelihood {alternative_log_likelihood:.12g} is below null "
                f"{null_log_likelihood:.12g}."
            )
            raise ValueError(msg)

        self.alternative_log_likelihood = alternative_log_likelihood
        self.null_log_likelihood = null_log_likelihood
        self.likelihood_ratio = max(float(likelihood_ratio), 0.0)
        return self.likelihood_ratio

    def compute_log_odds_ratio(
        self,
        taus: list | tuple | np.ndarray,
    ) -> float | None:
        """Return ``log((tau_00 * tau_11) / (tau_01 * tau_10))``."""
        self.verify_taus_are_valid(taus)
        tau_00, tau_01, tau_10, tau_11 = np.asarray(taus, dtype=float)
        if tau_01 * tau_10 == 0 or tau_00 * tau_11 == 0:
            return None
        return float(np.log((tau_00 * tau_11) / (tau_01 * tau_10)))

    def compute_wald_statistic(
        self,
        taus: list | tuple | np.ndarray,
    ) -> float | None:
        """Return the historical Wald statistic for the fitted latent table."""
        log_odds_ratio = self.compute_log_odds_ratio(taus)
        if log_odds_ratio is None:
            return None
        tau_00, tau_01, tau_10, tau_11 = np.asarray(taus, dtype=float)
        if any(tau == 0 for tau in (tau_00, tau_01, tau_10, tau_11)):
            return None
        standard_error = np.sqrt(
            (1 / tau_00) + (1 / tau_01) + (1 / tau_10) + (1 / tau_11),
        )
        return float(log_odds_ratio / standard_error)

    def compute_rho(self, taus: list | tuple | np.ndarray) -> float | None:
        """Return the Marshall-Olkin correlation of the latent driver states."""
        self.verify_taus_are_valid(taus)
        return compute_marshall_olkin_rho(taus)

    def compute_rho_for_direction(
        self,
        taus: list | tuple | np.ndarray,
        likelihood_ratio: float,
    ) -> float | None:
        """Return rho while forbidding an undefined direction for a positive LRT."""
        if not np.isfinite(likelihood_ratio) or likelihood_ratio < 0:
            msg = f"Interaction {self.name} has an invalid pair likelihood ratio."
            raise ValueError(msg)
        rho = self.compute_rho(taus)
        if rho is None and likelihood_ratio > UNDEFINED_RHO_LRT_TOL:
            msg = (
                f"Interaction {self.name} has undefined rho with positive "
                f"likelihood ratio {likelihood_ratio:.12g}."
            )
            raise ValueError(msg)
        return rho

    def estimate_tau_with_optimization_using_scipy(
        self,
        tau_init: list | None = None,
        alpha: float = 0.0,
    ) -> None:
        """Fit the concave alternative with deterministic constrained optimization."""
        del alpha
        null_taus = np.asarray(self.compute_independence_taus(), dtype=float)
        if tau_init is not None and not np.allclose(
            np.asarray(tau_init, dtype=float),
            null_taus,
            atol=_SIMPLEX_TOL,
        ):
            msg = (
                "The profile-LRT alternative must initialize at the independence "
                "null."
            )
            raise ValueError(msg)

        result = minimize(
            lambda taus: -self.compute_log_likelihood(taus),
            x0=null_taus,
            bounds=[(0.0, 1.0)] * 4,
            constraints={"type": "eq", "fun": lambda taus: taus.sum() - 1},
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            msg = f"Alternative optimization failed for {self.name}: {result.message}"
            raise ValueError(msg)
        self.verify_taus_are_valid(result.x)
        self.tau_00, self.tau_01, self.tau_10, self.tau_11 = result.x
        components = self.component_probabilities()
        fixed_point_residual = self._compute_fixed_point_residual(
            result.x,
            components,
        )
        kkt_residual = self._compute_kkt_residual(result.x, components)
        self._record_fit(
            algorithm="scipy-slsqp-v1",
            iterations=int(result.nit),
            last_log_likelihood_gain=0.0,
            fixed_point_residual=fixed_point_residual,
            kkt_residual=kkt_residual,
        )
        self.compute_likelihood_ratio(result.x)
        self._component_probabilities_cache = None

    def estimate_tau_with_coordinate_ascent(
        self,
        *,
        max_iter: int = 1000,
        kkt_tol: float = PAIR_FIT_KKT_TOL,
        tau_init: list | tuple | np.ndarray | None = None,
    ) -> None:
        """Fit the concave alternative by exact simplex coordinate ascent.

        The fit starts at the exact constrained independence null. At each step it
        transfers mass from the active cell with the smallest gradient to the cell
        with the largest gradient. The one-dimensional concave subproblem is solved
        deterministically by a boundary check plus derivative bisection. Convergence
        requires the global simplex KKT certificate, which is sufficient for this
        concave mixture-weight likelihood.
        """
        if max_iter <= 0:
            msg = "Coordinate-ascent max_iter must be positive."
            raise ValueError(msg)
        if not np.isfinite(kkt_tol) or kkt_tol <= 0:
            msg = "Coordinate-ascent kkt_tol must be finite and positive."
            raise ValueError(msg)

        null_taus = np.asarray(self.compute_independence_taus(), dtype=float)
        if tau_init is not None and not np.array_equal(
            np.asarray(tau_init, dtype=float),
            null_taus,
        ):
            msg = (
                "The profile-LRT alternative must initialize at the exact "
                "independence null."
            )
            raise ValueError(msg)

        (
            fitted_taus,
            iterations,
            last_gain,
            fixed_point_residual,
            kkt_residual,
        ) = self._run_tau_coordinate_ascent(null_taus, max_iter, kkt_tol)
        self.tau_00, self.tau_01, self.tau_10, self.tau_11 = fitted_taus
        self._record_fit(
            algorithm=PAIR_FIT_CONTRACT,
            iterations=iterations,
            last_log_likelihood_gain=last_gain,
            fixed_point_residual=fixed_point_residual,
            kkt_residual=kkt_residual,
        )
        self.compute_likelihood_ratio(fitted_taus)
        self._component_probabilities_cache = None

    def estimate_tau_with_em_from_scratch(
        self,
        max_iter: int = 1000,
        tol: float = PAIR_FIT_KKT_TOL,
        tau_init: list | None = None,
        n_inits: int = 1,
        seed: int = 0,
    ) -> None:
        """Backward-compatible entry point for the deterministic production fit."""
        del n_inits, seed
        self.estimate_tau_with_coordinate_ascent(
            max_iter=max_iter,
            kkt_tol=tol,
            tau_init=tau_init,
        )

    def _run_tau_coordinate_ascent(
        self,
        tau_init: np.ndarray,
        max_iter: int,
        kkt_tol: float,
    ) -> tuple[np.ndarray, int, float, float, float]:
        """Return a globally certified simplex fit and optimization diagnostics."""
        components = self.component_probabilities()
        taus = np.asarray(tau_init, dtype=float)
        self.verify_taus_are_valid(taus)
        if not np.isfinite(self.compute_log_likelihood(taus)):
            msg = f"Interaction {self.name} has a non-finite exact null likelihood."
            raise ValueError(msg)

        last_gain = 0.0
        for iteration in range(max_iter + 1):
            kkt_residual = self._compute_kkt_residual(taus, components)
            if kkt_residual <= kkt_tol:
                fixed_point_residual = self._compute_fixed_point_residual(
                    taus,
                    components,
                )
                if fixed_point_residual > kkt_tol:
                    msg = (
                        f"Interaction {self.name} passed KKT but failed its mixture "
                        "fixed-point certificate."
                    )
                    raise ValueError(msg)
                return (
                    taus,
                    iteration,
                    last_gain,
                    fixed_point_residual,
                    kkt_residual,
                )
            if iteration == max_iter:
                break

            probabilities = np.sum(components * taus, axis=1)
            normalized_gradients = self._normalized_gradients(
                components,
                probabilities,
            )
            target = int(np.argmax(normalized_gradients))
            active = np.flatnonzero(taus > 0)
            if not active.size:
                msg = f"Interaction {self.name} has no active simplex cell."
                raise ValueError(msg)
            donor = int(active[np.argmin(normalized_gradients[active])])
            if target == donor:
                msg = (
                    f"Interaction {self.name} has a KKT violation without a "
                    "feasible ascent coordinate."
                )
                raise ValueError(msg)

            candidate, gain = self._maximize_mass_transfer(
                taus,
                components,
                probabilities,
                target=target,
                donor=donor,
            )
            if np.array_equal(candidate, taus):
                msg = (
                    f"Interaction {self.name} coordinate ascent stalled before "
                    "satisfying KKT."
                )
                raise ValueError(msg)
            taus = candidate
            last_gain = gain

        msg = (
            f"Interaction {self.name} failed to converge within {max_iter} "
            "coordinate-ascent iterations."
        )
        raise ValueError(msg)

    def _maximize_mass_transfer(
        self,
        taus: np.ndarray,
        components: np.ndarray,
        probabilities: np.ndarray,
        *,
        target: int,
        donor: int,
    ) -> tuple[np.ndarray, float]:
        """Maximize and certify one feasible target-minus-donor mass transfer."""
        upper = float(taus[donor])
        if upper <= 0:
            msg = f"Interaction {self.name} selected an inactive mass donor."
            raise ValueError(msg)
        delta = components[:, target] - components[:, donor]
        derivative_at_zero = self._mass_transfer_derivative(
            probabilities,
            delta,
            0.0,
        )
        if not np.isfinite(derivative_at_zero) or derivative_at_zero <= 0:
            msg = f"Interaction {self.name} selected a non-ascent coordinate."
            raise ValueError(msg)

        derivative_at_upper = self._mass_transfer_derivative(
            probabilities,
            delta,
            upper,
        )
        if derivative_at_upper >= 0:
            transfer = upper
        else:
            lower, upper_bound = 0.0, upper
            for _ in range(_LINE_SEARCH_ITERATIONS):
                midpoint = (lower + upper_bound) / 2
                derivative = self._mass_transfer_derivative(
                    probabilities,
                    delta,
                    midpoint,
                )
                if derivative > 0:
                    lower = midpoint
                else:
                    upper_bound = midpoint
            # The lower bracket retains a positive directional derivative. Using
            # its midpoint with the nonpositive bracket can step microscopically
            # past the maximizer after floating-point rounding.
            transfer = lower

        for _ in range(_LINE_SEARCH_ITERATIONS + 1):
            candidate = taus.copy()
            candidate[target] += transfer
            candidate[donor] -= transfer
            if transfer == upper:
                candidate[donor] = 0.0
            gain = self._certify_mass_transfer_candidate(
                taus,
                candidate,
                components,
                probabilities,
            )
            if gain is not None:
                return candidate, gain
            transfer /= 2

        msg = (
            f"Interaction {self.name} could not represent a certified monotone "
            "coordinate-ascent step."
        )
        raise ValueError(msg)

    def _certify_mass_transfer_candidate(
        self,
        taus: np.ndarray,
        candidate: np.ndarray,
        components: np.ndarray,
        probabilities: np.ndarray,
    ) -> float | None:
        """Return a candidate's nonnegative direct LL gain when certified."""
        if np.array_equal(candidate, taus):
            return None
        self.verify_taus_are_valid(candidate)
        changes = candidate - taus
        probability_changes = np.sum(components * changes, axis=1)
        probability_ratios = probability_changes / probabilities
        if np.any(probability_ratios <= -1):
            return None
        gain = float(math.fsum(np.log1p(probability_ratios)))
        if not np.isfinite(gain) or gain < 0:
            return None
        return gain

    @staticmethod
    def _mass_transfer_derivative(
        probabilities: np.ndarray,
        delta: np.ndarray,
        transfer: float,
    ) -> float:
        """Return the derivative along one target-minus-donor simplex edge."""
        transferred_probabilities = probabilities + (transfer * delta)
        if np.any(transferred_probabilities <= 0):
            return -np.inf
        derivative = np.sum(delta / transferred_probabilities)
        return float(derivative)

    @staticmethod
    def _normalized_gradients(
        components: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Return likelihood gradients normalized by the sample count."""
        if np.any(probabilities <= 0):
            msg = "Cannot evaluate mixture gradients at zero observation probability."
            raise ValueError(msg)
        gradients = np.sum(components / probabilities[:, None], axis=0)
        normalized = gradients / len(probabilities)
        if not np.all(np.isfinite(normalized)):
            msg = "Mixture gradient is non-finite."
            raise ValueError(msg)
        return normalized

    @classmethod
    def _compute_fixed_point_residual(
        cls,
        taus: np.ndarray,
        components: np.ndarray,
    ) -> float:
        """Return the residual of the mixture-weight multiplicative fixed point."""
        probabilities = np.sum(components * taus, axis=1)
        normalized_gradients = cls._normalized_gradients(components, probabilities)
        updated = taus * normalized_gradients
        return float(np.max(np.abs(updated - taus)))

    @classmethod
    def _compute_kkt_residual(cls, taus: np.ndarray, components: np.ndarray) -> float:
        """Return the normalized simplex KKT residual for the mixture weights."""
        probabilities = np.sum(components * taus, axis=1)
        normalized_gradients = cls._normalized_gradients(components, probabilities)
        active = taus > 0
        active_residual = (
            np.max(np.abs(normalized_gradients[active] - 1))
            if np.any(active)
            else 0.0
        )
        inactive_residual = (
            np.max(np.maximum(normalized_gradients[~active] - 1, 0))
            if np.any(~active)
            else 0.0
        )
        return float(max(active_residual, inactive_residual))

    def _record_fit(
        self,
        *,
        algorithm: str,
        iterations: int,
        last_log_likelihood_gain: float,
        fixed_point_residual: float,
        kkt_residual: float,
    ) -> None:
        """Record generic fit metadata and maintain legacy attribute aliases."""
        self.fit_algorithm = algorithm
        self.fit_converged = True
        self.fit_iterations = iterations
        self.fit_last_log_likelihood_gain = last_log_likelihood_gain
        self.fit_fixed_point_residual = fixed_point_residual
        self.fit_kkt_residual = kkt_residual
        self.em_converged = True
        self.em_iterations = iterations
        self.em_n_inits_used = 1
        self.em_final_log_likelihood_increment = last_log_likelihood_gain
        self.em_fixed_point_residual = fixed_point_residual
        self.em_kkt_residual = kkt_residual

    def estimate_tau_with_em_using_pomegranate(self) -> None:
        """Raise because the optional pomegranate implementation does not exist."""
        msg = "Method is not yet implemented."
        raise NotImplementedError(msg)

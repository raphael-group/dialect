"""Single-gene latent-driver model."""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import minimize_scalar

OBSERVATION_SUPPORT_CONTRACT = "latent-state-union-v1"

_NESTED_LIKELIHOOD_TOL = 1e-8


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
        self.pi = None

        self.cbase_phi = None
        self.cbase_p = None

        self.mle_converged = False
        self.mle_iterations = 0
        self.mle_log_likelihood = None
        self.likelihood_ratio = None
        self.likelihood_ratio_status = None
        self._component_probabilities_cache = None

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
                f"Gene {self.name} has misaligned samples, counts, and background "
                "PMFs."
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
                    f"Gene {self.name} has invalid count {count!r} for sample "
                    f"{sample}."
                )
                raise ValueError(msg)
            p_no_driver = float(pmf.get(int(count), 0.0))
            p_driver = float(pmf.get(int(count) - 1, 0.0))
            if (
                not np.isfinite(p_no_driver)
                or not np.isfinite(p_driver)
                or p_no_driver < 0
                or p_driver < 0
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

    def compute_log_likelihood(self, pi: float) -> float:
        r"""Compute ``sum_i log(P(B_i=c_i)(1-pi)+P(B_i=c_i-1)pi)``."""
        self.verify_pi_is_valid(pi)
        components = self.component_probabilities()
        probabilities = (components[:, 0] * (1 - pi)) + (components[:, 1] * pi)
        if np.any(probabilities <= 0):
            return -np.inf
        return float(math.fsum(np.log(probabilities)))

    def compute_likelihood_ratio(self, pi: float) -> float:
        """Compute the single-gene driver-versus-no-driver LRT."""
        self.verify_pi_is_valid(pi)
        alternative_log_likelihood = self.compute_log_likelihood(pi)
        null_log_likelihood = self.compute_log_likelihood(0)
        if not np.isfinite(alternative_log_likelihood):
            msg = f"Gene {self.name} has a non-finite fitted log-likelihood."
            raise ValueError(msg)
        if np.isneginf(null_log_likelihood):
            self.likelihood_ratio = np.inf
            self.likelihood_ratio_status = "infinite-passenger-null-zero-probability"
            return self.likelihood_ratio
        likelihood_ratio = 2 * (alternative_log_likelihood - null_log_likelihood)
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

    def estimate_pi_with_mle(
        self,
        *,
        max_iter: int = 1000,
        tol: float = 1e-12,
    ) -> None:
        """Fit the constrained-null marginal MLE against the exact likelihood."""
        self.component_probabilities()
        result = minimize_scalar(
            lambda pi: -self.compute_log_likelihood(float(pi)),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"maxiter": max_iter, "xatol": tol},
        )
        if not result.success:
            msg = f"Optimization failed for gene {self.name}: {result.message}"
            raise ValueError(msg)

        candidates = (0.0, float(result.x), 1.0)
        evaluated = [(pi, self.compute_log_likelihood(pi)) for pi in candidates]
        self.pi, self.mle_log_likelihood = max(evaluated, key=lambda item: item[1])
        self.mle_converged = True
        self.mle_iterations = int(result.nfev)

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
        self.estimate_pi_with_mle(max_iter=max_iter, tol=tol)
        self.em_n_inits_used = 1

    def estimate_pi_with_em_using_pomegranate(self) -> None:
        """Raise because the optional pomegranate implementation does not exist."""
        msg = "EM algorithm not implemented yet."
        raise NotImplementedError(msg)

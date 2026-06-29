"""TODO: Add docstring."""

import itertools
import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Warn when more than this fraction of a gene's samples is excluded from the EM.
_HYPERMUTATOR_DROP_FRACTION = 0.05
# EM random restarts: stop once this many extra inits in a row fail to improve the
# log-likelihood, so a well-behaved (often unimodal) fit needs far fewer than n_inits.
_EM_RESTART_PATIENCE = 3


class Gene:
    """TODO: Add docstring."""

    def __init__(
        self,
        name: str,
        samples: list,
        counts: list,
        bmr_pmf: dict | list,
    ) -> None:
        """Build a gene's observed counts and its background model.

        ``bmr_pmf`` is either a single background count PMF ``{k: P(B=k)}`` shared by
        every sample, or a per-sample list of such PMFs (one per sample, aligned with
        ``counts``). The per-sample form lets the background depend on each sample's
        mutation burden; the shared form is broadcast to every sample. All likelihood
        and EM evaluation indexes through :attr:`bmr_pmfs`, so the model is
        sample-indexed either way.
        """
        self.name = name
        self.samples = samples
        self.counts = counts
        self.bmr_pmf = bmr_pmf
        self.pi = None

        self.cbase_phi = None
        self.cbase_p = None

    @property
    def bmr_pmfs(self) -> list:
        """Per-sample background PMFs, one per sample (aligned with ``counts``).

        A single shared ``bmr_pmf`` dict is broadcast to every sample; a per-sample
        list is returned as-is. Every likelihood/EM path reads the background through
        here, so a single cohort-level BMR and per-sample BMRs use the same code.
        """
        if isinstance(self.bmr_pmf, list):
            return self.bmr_pmf
        return [self.bmr_pmf] * len(self.counts)

    def __str__(self) -> str:
        """TODO: Add docstring."""
        representative = self.bmr_pmfs[0] if self.bmr_pmfs else {}
        bmr_preview = ", ".join(
            f"{k}: {v:.3e}" for k, v in itertools.islice(representative.items(), 3)
        )
        pi_info = f"Pi: {self.pi:.3e}" if self.pi is not None else "Pi: Not estimated"
        total_mutations = np.sum(self.counts)
        return (
            f"Gene: {self.name}\n"
            f"Total Mutations: {total_mutations}\n"
            f"BMR PMF (preview): {{ {bmr_preview} }}\n"
            f"{pi_info}"
        )

    # -------------------------------------------------------------------------------- #
    #                                 UTILITY FUNCTIONS                                #
    # -------------------------------------------------------------------------------- #
    def calculate_expected_mutations(self) -> float:
        """Return the mean per-sample E[B] = sum_k k * P(B=k) over the background PMFs.

        For a single shared background this is the usual E[B]; for per-sample
        backgrounds it averages each sample's expected passenger count.
        """
        return float(
            np.mean([sum(k * p for k, p in pmf.items()) for pmf in self.bmr_pmfs]),
        )

    # -------------------------------------------------------------------------------- #
    #                            DATA VALIDATION AND LOGGING                           #
    # -------------------------------------------------------------------------------- #
    def verify_bmr_pmf_and_counts_exist(self) -> None:
        """TODO: Add docstring."""
        if self.bmr_pmf is None:
            msg = "BMR PMF is not defined for this gene."
            raise ValueError(msg)
        if self.counts is None:
            msg = "Counts are not defined for this gene."
            raise ValueError(msg)

    def verify_bmr_pmf_contains_all_count_keys(self) -> None:
        """Verify every sample's observed count is a key in that sample's PMF."""
        missing_bmr_pmf_counts = [
            c
            for c, pmf in zip(self.counts, self.bmr_pmfs, strict=False)
            if c not in pmf
        ]
        if missing_bmr_pmf_counts:
            msg = "BMR PMF does not contain all counts in distribution."
            raise ValueError(msg)

    def verify_pi_is_valid(self, pi: float) -> None:
        """TODO: Add docstring."""
        if pi is None or not 0 <= pi <= 1:
            msg = f"Invalid pi value: {pi}. Pi must be in the range [0, 1]."
            raise ValueError(
                msg,
            )
        if pi == 1:
            msg = "Estimated pi is 1. Please ensure pi is less than 1."
            raise ValueError(msg)

    # -------------------------------------------------------------------------------- #
    #                         LIKELIHOOD AND METRIC EVALUATION                         #
    # -------------------------------------------------------------------------------- #
    def compute_log_likelihood(self, pi: float) -> float:
        r"""Compute the log-likelihood for gene given the estimated \\( \\pi \\).

        The log-likelihood function is defined as:

        .. math::

            \\ell_C(\\pi) = \\sum_{i=1}^{N} \\log \\Big(
                \\mathbb{P}(P_i = c_i)(1 - \\pi) +
                \\mathbb{P}(P_i = c_i - 1) \\pi
            \\Big)

        where:

        - \\( N \\): Number of samples.
        - \\( P_i \\): Random variable representing passenger mutations.
        - \\( c_i \\): Observed count of somatic mutations for sample \\( i \\).
        - \\( \\pi \\): Estimated driver mutation rate parameter value.

        **Returns**:
        :return: (float) The computed log-likelihood value.

        **Raises**:
        :raises ValueError: If `bmr_pmf`, `counts`, or `pi` is not properly defined.
        """

        # TODO @ashuaibi7: move function to helper module
        # https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get_no_default(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c, min_val)

        # TODO @ashuaibi7: move function to helper module
        # https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get_with_default(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c, 0)

        self.verify_pi_is_valid(pi)
        self.verify_bmr_pmf_and_counts_exist()

        return sum(
            np.log(
                safe_get_no_default(pmf, c) * (1 - pi)
                + safe_get_with_default(pmf, c - 1) * pi,
            )
            for c, pmf in zip(self.counts, self.bmr_pmfs, strict=False)
        )

    def compute_likelihood_ratio(self, pi: float) -> float:
        r"""Compute the likelihood ratio test statistic (\\( \\lambda_{LR} \\)).

        The likelihood ratio test statistic is calculated as:

        .. math::

            \\lambda_{LR} = -2 [ \\ell(\\pi_{\\text{null}}) - \\ell(\\hat{\\pi}) ]

        where:

        - \\( \\ell() \\): Log-likelihood function.
        - \\( \\pi_{\\text{null}} \\): Null hypothesis value.
        - \\( \\hat{\\pi} \\): Value of the \\( \\pi \\) under the alternative.

        **Returns**:
        :return: (float) The likelihood ratio test statistic (\\( \\lambda_{LR} \\)).
        """
        self.verify_pi_is_valid(pi)

        return -2 * (self.compute_log_likelihood(0) - self.compute_log_likelihood(pi))

    def compute_log_odds_ratio(self, pi: float) -> float:
        r"""Compute the log odds ratio.

        The log odds ratio is calculated as:

        .. math::

            L = \\log\\left(\\frac{\\tau_{01} \\tau_{10}}{\\tau_{00} \\tau_{11}}\\right)

        **Returns**:
        :return: (float) The log odds ratio.
        """
        self.verify_pi_is_valid(pi)

        return np.log(self.pi / (1 - pi))

    # -------------------------------------------------------------------------------- #
    #                           PARAMETER ESTIMATION METHODS                           #
    # -------------------------------------------------------------------------------- #
    def estimate_pi_with_optimiziation_using_scipy(self, pi_init: float = 0.5) -> None:
        """TODO: Add docstring."""

        def negative_log_likelihood(pi: float) -> float:
            return -self.compute_log_likelihood(pi)

        self.verify_bmr_pmf_and_counts_exist()

        alpha = 1e-13
        bounds = [
            (alpha, 1 - alpha),
        ]
        result = minimize(
            negative_log_likelihood,
            x0=[pi_init],
            bounds=bounds,
            method="SLSQP",
        )

        if not result.success:
            msg = f"Optimization failed: {result.message}"
            raise ValueError(msg)

        self.pi = result.x[0]

    def estimate_pi_with_em_from_scratch(
        self,
        max_iter: int = 1000,
        tol: float = 1e-3,
        pi_init: float = 0.5,
        n_inits: int = 10,
        seed: int = 0,
    ) -> None:
        r"""Estimate the pi parameter using the Expectation-Maximization (EM) algorithm.

        This method iteratively updates the parameter \\( \\pi \\) to maximize
        the likelihood of the observed data.

        **Algorithm Steps**:

        1. **E-Step**: Compute the responsibilities (\\( z_i^{(t)} \\)) as:

           .. math::

                z_{i}^{(t)} = \\frac{\\pi^{(t)} \\cdot \\mathbb{P}(P_i = c_i - 1)}
                {\\pi^{(t)} \\cdot \\mathbb{P}(P_i = c_i - 1)
                    + (1 - \\pi^{(t)}) \\cdot \\mathbb{P}(P_i = c_i)}

        2. **M-Step**: Update the parameter \\( \\pi \\) as:

           .. math::

                \\pi^{(t+1)} = \\frac{1}{N} \\sum_{i=1}^{N} z_{i}^{(t)}

        **Parameters**:
        :param max_iter: (int) Maximum number of iterations (default: 1000).
        :param epsilon: (float) Convergence threshold for log-likelihood improvement.
        :param pi_init: (float) The initialization value for \\( \\pi \\).

        **Returns**:
        :return: (float) The estimated value of \\( \\pi \\).
        """
        self.verify_bmr_pmf_and_counts_exist()

        nonzero_count_pmf_pairs = [
            (c, pmf)
            for c, pmf in zip(self.counts, self.bmr_pmfs, strict=False)
            if c in pmf and pmf[c] > 0
        ]
        # Samples whose observed count has no background support (out-of-range or
        # zero probability) are excluded to avoid 0/0 responsibilities -- this is
        # typically driven by hypermutators under a cohort-level BMR. Surface it
        # instead of dropping silently (see the hypermutator-handling workstream).
        n_excluded = len(self.counts) - len(nonzero_count_pmf_pairs)
        if n_excluded and n_excluded / len(self.counts) > _HYPERMUTATOR_DROP_FRACTION:
            logger.warning(
                "Gene %s: %d/%d samples excluded from EM (no background support; "
                "likely hypermutators); pi may be biased.",
                self.name,
                n_excluded,
                len(self.counts),
            )

        rng = np.random.default_rng(seed)
        # init #0 is the informed default (0.5); the rest are random restarts that
        # guard against local optima (Reviewer 3). Keep the highest-likelihood fit.
        inits = [pi_init, *rng.uniform(0.01, 0.99, max(n_inits - 1, 0)).tolist()]
        best_pi, best_log_likelihood, stale, n_used = pi_init, -np.inf, 0, 0
        for init in inits:
            n_used += 1
            pi = self._run_pi_em(nonzero_count_pmf_pairs, init, max_iter, tol)
            log_likelihood = self.compute_log_likelihood(pi)
            if log_likelihood > best_log_likelihood + tol:
                best_pi, best_log_likelihood, stale = pi, log_likelihood, 0
            else:
                stale += 1
                if stale >= _EM_RESTART_PATIENCE:
                    break

        self.pi = best_pi
        self.em_n_inits_used = n_used

    def _run_pi_em(
        self,
        nonzero_count_pmf_pairs: list,
        pi_init: float,
        max_iter: int,
        tol: float,
    ) -> float:
        """Run one EM trajectory from a single pi_init and return the fitted pi."""
        pi = pi_init
        for _it in range(max_iter):
            z_i = [
                (pi * pmf.get(c - 1, 0))
                / (pi * pmf.get(c - 1, 0) + (1 - pi) * pmf.get(c, 0))
                for c, pmf in nonzero_count_pmf_pairs
            ]
            curr_pi = np.mean(z_i)
            if abs(self.compute_log_likelihood(curr_pi)
                   - self.compute_log_likelihood(pi)) < tol:
                break
            pi = curr_pi
        return pi

    # TODO @ashuaibi7: implement em w/ pomegranate
    # https://linear.app/princeton-phd-research/issue/DEV-76
    def estimate_pi_with_em_using_pomegranate(self) -> None:
        """TODO: Add docstring."""
        msg = "EM algorithm not implemented yet."
        raise NotImplementedError(msg)

"""TODO: Add docstring."""

import itertools

import numpy as np
from scipy.optimize import minimize


class Gene:
    """TODO: Add docstring."""

    def __init__(self, name: str, samples: list, counts: list, bmr_pmf: list) -> None:
        """TODO: Add docstring."""
        self.name = name
        self.samples = samples
        self.counts = counts
        self.bmr_pmf = bmr_pmf
        self.pi = None

        self.cbase_phi = None
        self.cbase_p = None

    def __str__(self) -> str:
        """TODO: Add docstring."""
        bmr_preview = ", ".join(
            f"{k}: {v:.3e}" for k, v in itertools.islice(self.bmr_pmf.items(), 3)
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
        """TODO: Add docstring."""
        total_samples = len(self.counts)
        expected_mutations = sum(k * prob for k, prob in self.bmr_pmf.items())
        return expected_mutations * total_samples

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
        """TODO: Add docstring."""
        missing_bmr_pmf_counts = [c for c in self.counts if c not in self.bmr_pmf]
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
            return pmf.get(c)

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
                safe_get_no_default(self.bmr_pmf, c) * (1 - pi)
                + safe_get_with_default(self.bmr_pmf, c - 1) * pi,
            )
            for c in self.counts
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

        nonzero_probability_counts = [
            c for c in self.counts if c in self.bmr_pmf and self.bmr_pmf[c] > 0
        ]

        pi = pi_init
        for _it in range(max_iter):
            z_i = [
                (pi * self.bmr_pmf.get(c - 1, 0))
                / (pi * self.bmr_pmf.get(c - 1, 0) + (1 - pi) * self.bmr_pmf.get(c, 0))
                for c in nonzero_probability_counts
            ]

            curr_pi = np.mean(z_i)

            prev_log_likelihood = self.compute_log_likelihood(pi)
            curr_log_likelihood = self.compute_log_likelihood(curr_pi)
            if abs(curr_log_likelihood - prev_log_likelihood) < tol:
                break

            pi = curr_pi

        self.pi = pi

    # TODO @ashuaibi7: implement em w/ pomegranate
    # https://linear.app/princeton-phd-research/issue/DEV-76
    def estimate_pi_with_em_using_pomegranate(self) -> None:
        """TODO: Add docstring."""
        msg = "EM algorithm not implemented yet."
        raise NotImplementedError(msg)

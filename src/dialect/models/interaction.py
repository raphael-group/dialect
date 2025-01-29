"""TODO: Add docstring."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix

from dialect.models.gene import Gene


class Interaction:
    """TODO: Add docstring."""

    def __init__(self, gene_a: Gene, gene_b: Gene) -> None:
        """TODO: Add docstring."""
        if not isinstance(gene_a, Gene) or not isinstance(gene_b, Gene):
            msg = "Both inputs must be instances of the Gene class."
            raise TypeError(msg)

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

    def __str__(self) -> str:
        """TODO: Add docstring."""
        taus_info = (
            f"tau_00={self.tau_00:.3e}, tau_01={self.tau_01:.3e}, "
            f"tau_10={self.tau_10:.3e}, tau_11={self.tau_11:.3e}"
            if None not in (self.tau_00, self.tau_01, self.tau_10, self.tau_11)
            else "Tau values not estimated"
        )

        pi_a = (
            f"{self.gene_a.pi:.3e}" if self.gene_a.pi is not None else "Not estimated"
        )
        pi_b = (
            f"{self.gene_b.pi:.3e}" if self.gene_b.pi is not None else "Not estimated"
        )

        cm = self.compute_contingency_table()
        cm_info = (
            f"\nContingency Table:\n[[{cm[1, 1]} {cm[1, 0]}]\n [{cm[0, 1]} {cm[0, 0]}]]"
        )

        return (
            f"Interaction: {self.name}\n"
            f"Gene A: {self.gene_a.name} (Pi: {pi_a})\n"
            f"Gene B: {self.gene_b.name} (Pi: {pi_b})\n"
            f"Tau Parameters: {taus_info}\n"
            f"Contingency Table:{cm_info}"
        )

    # -------------------------------------------------------------------------------- #
    #                                  UTILITY METHODS                                 #
    # -------------------------------------------------------------------------------- #
    def compute_contingency_table(self) -> np.ndarray:
        """TODO: Add docstring."""
        gene_a_mutations = (self.gene_a.counts > 0).astype(int)
        gene_b_mutations = (self.gene_b.counts > 0).astype(int)
        return confusion_matrix(gene_a_mutations, gene_b_mutations, labels=[1, 0])

    def get_set_of_cooccurring_samples(self) -> list:
        """TODO: Add docstring."""
        sample_names = self.gene_a.samples
        cooccurring_samples = [
            sample_names[i]
            for i in range(len(sample_names))
            if self.gene_a.counts[i] > 0 and self.gene_b.counts[i] > 0
        ]
        return sorted(cooccurring_samples)

    def compute_fisher_pvalues(self) -> tuple(float, float):
        """TODO: Add docstring."""
        cross_tab = self.compute_contingency_table()
        _, me_pval = fisher_exact(cross_tab, alternative="less")
        _, co_pval = fisher_exact(cross_tab, alternative="greater")

        return me_pval, co_pval

    # -------------------------------------------------------------------------------- #
    #                            DATA VALIDATION AND LOGGING                           #
    # -------------------------------------------------------------------------------- #
    def verify_bmr_pmf_and_counts_exist(self) -> None:
        """TODO: Add docstring."""
        if self.gene_a.bmr_pmf is None or self.gene_b.bmr_pmf is None:
            msg = "BMR PMFs are not defined for one or both genes."
            raise ValueError(msg)

        if self.gene_a.counts is None or self.gene_b.counts is None:
            msg = "Counts are not defined for one or both genes."
            raise ValueError(msg)

    def verify_taus_are_valid(self, taus: list, tol: float = 1e-2) -> None:
        """TODO: Add docstring."""
        if not all(0 <= t <= 1 for t in taus) or not np.isclose(
            sum(taus),
            1,
            atol=tol,
        ):
            msg = "Invalid tau parameters. Ensure 0 <= tau_i <= 1 and sum(tau) == 1."
            raise ValueError(
                msg,
            )
        tau_11 = taus[-1]
        if tau_11 == 1:
            msg = "Tau_11 cannot be 1. This leads to log(0) in log-likelihood."
            raise ValueError(
                msg,
            )

    def verify_pi_values(self, pi_a: float, pi_b: float) -> None:
        """TODO: Add docstring."""
        if pi_a is None or pi_b is None:
            msg = "Driver probabilities are not defined for both genes."
            raise ValueError(
                msg,
            )

    # -------------------------------------------------------------------------------- #
    #                         LIKELIHOOD AND METRIC EVALUATION                         #
    # -------------------------------------------------------------------------------- #
    def compute_joint_probability(self, tau: float, u: int, v: int) -> np.ndarray:
        """TODO: Add docstring."""
        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c, 0)

        return np.array(
            [
                tau
                * safe_get(self.gene_a.bmr_pmf, c_a - u, 0)
                * safe_get(self.gene_b.bmr_pmf, c_b - v, 0)
                for c_a, c_b in zip(self.gene_a.counts, self.gene_b.counts)
            ],
        )

    def compute_total_probability(
        self,
        tau_00: float,
        tau_01: float,
        tau_10: float,
        tau_11: float,
    ) -> np.ndarray:
        """TODO: Add docstring."""
        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c, 0)

        return np.array(
            [
                sum(
                    (
                        tau_00
                        * safe_get(self.gene_a.bmr_pmf, c_a, 0)
                        * safe_get(self.gene_b.bmr_pmf, c_b, 0),
                        tau_01
                        * safe_get(self.gene_a.bmr_pmf, c_a, 0)
                        * safe_get(self.gene_b.bmr_pmf, c_b - 1, 0),
                        tau_10
                        * safe_get(self.gene_a.bmr_pmf, c_a - 1, 0)
                        * safe_get(self.gene_b.bmr_pmf, c_b, 0),
                        tau_11
                        * safe_get(self.gene_a.bmr_pmf, c_a - 1, 0)
                        * safe_get(self.gene_b.bmr_pmf, c_b - 1, 0),
                    ),
                )
                for c_a, c_b in zip(self.gene_a.counts, self.gene_b.counts)
            ],
        )

    def compute_log_likelihood(self, taus: list) -> float:
        r"""Compute the log-likelihood for the interaction given \\( \tau \\) params.

        The log-likelihood function is defined as:

        .. math::

            \\ell_C(\\tau) = \\sum_{i=1}^{N} \\log \\Big(
                \\mathbb{P}(P_i = c_i) \\mathbb{P}(P_i' = c_i') \\tau_{00} +
                \\mathbb{P}(P_i = c_i) \\mathbb{P}(P_i' = c_i' - 1) \\tau_{01} +
                \\mathbb{P}(P_i = c_i - 1) \\mathbb{P}(P_i' = c_i') \\tau_{10} +
                \\mathbb{P}(P_i = c_i - 1) \\mathbb{P}(P_i' = c_i' - 1) \\tau_{11}
            \\Big)

        where:

        - \\( N \\): Number of samples.
        - \\( P_i \\) and \\( P_i' \\): Random variables representing passengers.
        - \\( c_i \\) and \\( c_i' \\): Observed counts of somatic mutations.
        - \\( \\tau = (\\tau_{00}, \\tau_{01}, \\tau_{10}, \\tau_{11}) \\): Params.

        **Parameters**:
        :param tau: (tuple) \\( (\\tau_{00}, \\tau_{01}, \\tau_{10}, \\tau_{11}) \\)

        **Returns**:
        :return: (float) The computed log-likelihood value.

        **Raises**:
        :raises ValueError: If `bmr_pmf` or `counts` are not defined for either gene,
            or if `tau` is invalid.
        """

        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get_no_default(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c)

        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get_with_default(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c, 0)

        self.verify_bmr_pmf_and_counts_exist()
        self.verify_taus_are_valid(taus)

        a_counts, b_counts = self.gene_a.counts, self.gene_b.counts
        a_bmr_pmf, b_bmr_pmf = self.gene_a.bmr_pmf, self.gene_b.bmr_pmf
        tau_00, tau_01, tau_10, tau_11 = taus
        return sum(
            np.log(
                safe_get_no_default(a_bmr_pmf, c_a)
                * safe_get_no_default(b_bmr_pmf, c_b)
                * tau_00
                + safe_get_no_default(a_bmr_pmf, c_a)
                * safe_get_with_default(b_bmr_pmf, c_b - 1)
                * tau_01
                + safe_get_with_default(a_bmr_pmf, c_a - 1)
                * safe_get_no_default(b_bmr_pmf, c_b)
                * tau_10
                + safe_get_with_default(a_bmr_pmf, c_a - 1)
                * safe_get_with_default(b_bmr_pmf, c_b - 1)
                * tau_11,
            )
            for c_a, c_b in zip(a_counts, b_counts)
        )

    def compute_likelihood_ratio(self, taus: list) -> float:
        r"""Compute likelihood ratio test (\\( \\lambda_{LR} \\)) w.r.t. the null.

        The likelihood ratio test statistic is defined as:

        .. math::

            \\lambda_{LR} = -2 [ \\ell(\\tau_{\\text{null}}) - \\ell(\\hat{\\tau}) ]

        where:

        - \\( \\ell() \\): Log-likelihood function.
        - \\( \\tau_{\\text{null}} \\): Null hypothesis of no interaction.
        - \\( \\hat{\\tau} \\): Estimated values of the \\( \\tau \\) parameters.

        **Returns**:
        :return: (float) The computed l.r.t. statistic (\\( \\lambda_{LR} \\)).
        """
        tau_00, tau_01, tau_10, tau_11 = taus
        driver_a_marginal = tau_10 + tau_11
        driver_b_marginal = tau_01 + tau_11

        tau_null = (
            (1 - driver_a_marginal) * (1 - driver_b_marginal),
            (1 - driver_a_marginal) * driver_b_marginal,
            driver_a_marginal * (1 - driver_b_marginal),
            driver_a_marginal * driver_b_marginal,
        )
        return -2 * (
            self.compute_log_likelihood(tau_null)
            - self.compute_log_likelihood((tau_00, tau_01, tau_10, tau_11))
        )

    def compute_log_odds_ratio(self, taus: list) -> float:
        r"""Compute the log odds ratio for the interaction based on the \\( \tau \\).

        The log odds ratio is calculated as:

        .. math::

            \text{Log Odds Ratio} = \\log \\left(
                \\frac{\\tau_{01} \\cdot \\tau_{10}}{\\tau_{00} \\cdot \\tau_{11}}
            \\right)

        **Returns**:
        :return: (float) The computed log odds ratio.

        **Raises**:
        :raises ValueError: If \\( \\tau \\) parameters are invalid.
        """
        self.verify_taus_are_valid(taus)
        tau_00, tau_01, tau_10, tau_11 = taus

        if tau_01 * tau_10 == 0 or tau_00 * tau_11 == 0:
            return None

        return np.log((tau_01 * tau_10) / (tau_00 * tau_11))

    def compute_wald_statistic(self, taus: list) -> float:
        r"""Compute the Wald statistic for the interaction.

        The Wald statistic is calculated as:

        .. math::

            W = \\frac{\\text{Log Odds Ratio}}{\\text{Standard Error}}

        where the standard error is defined as:

        .. math::

            \\text{Standard Error} = \\sqrt{
                \\frac{1}{\\tau_{01}} +
                \\frac{1}{\\tau_{10}} +
                \\frac{1}{\\tau_{00}} +
                \\frac{1}{\\tau_{11}}
            }

        **Returns**:
        :return: (float or None) The computed Wald statistic, or None.
        """
        self.verify_taus_are_valid(taus)
        log_odds_ratio = self.compute_log_odds_ratio(taus)
        if log_odds_ratio is None:
            return None

        try:
            std_err = np.sqrt(
                (1 / self.tau_01)
                + (1 / self.tau_10)
                + (1 / self.tau_00)
                + (1 / self.tau_11),
            )
        except ZeroDivisionError:
            return None

        return log_odds_ratio / std_err

    def compute_rho(self, taus: list) -> float:
        r"""Compute the interaction measure \\( \rho \\) for the given \\( \tau \\).

        The interaction measure \\( \rho \\) is calculated as:

        .. math::

            \\rho = \\frac{\\tau_{11} \\cdot \\tau_{00} - \\tau_{01} \\cdot \\tau_{10}}
            {\\sqrt{\\tau_{0*} \\cdot \\tau_{1*} \\cdot \\tau_{*0} \\cdot \\tau_{*1}}}

        where:

        - \\( \\tau_{0*} = \\tau_{00} + \\tau_{01} \\): Marginal probability for
            no driver mutation in gene \\( A \\).
        - \\( \\tau_{1*} = \\tau_{10} + \\tau_{11} \\): Marginal probability for
            a driver mutation in gene \\( A \\).
        - \\( \\tau_{*0} = \\tau_{00} + \\tau_{10} \\): Marginal probability for
            no driver mutation in gene \\( B \\).
        - \\( \\tau_{*1} = \\tau_{01} + \\tau_{11} \\): Marginal probability for
            a driver mutation in gene \\( B \\).

        **Returns**:
        :return: (float or None) The computed value of \\( \\rho \\), or None.
        """
        self.verify_taus_are_valid(taus)

        tau_0x = self.tau_00 + self.tau_01
        tau_1x = self.tau_10 + self.tau_11
        tau_x0 = self.tau_00 + self.tau_10
        tau_x1 = self.tau_01 + self.tau_11

        if any(t == 0 for t in [tau_0x, tau_1x, tau_x0, tau_x1]):
            return None

        return (self.tau_11 * self.tau_00 - self.tau_01 * self.tau_10) / (
            np.sqrt(tau_0x * tau_1x * tau_x0 * tau_x1)
        )

    # -------------------------------------------------------------------------------- #
    #                           PARAMETER ESTIMATION METHODS                           #
    # -------------------------------------------------------------------------------- #
    def estimate_tau_with_optimization_using_scipy(
        self,
        tau_init: list | None = None,
        alpha: float = 1e-13,
    ) -> None:
        """TODO: Add docstring."""
        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-78
        if tau_init is None:
            tau_init = [0.25, 0.25, 0.25, 0.25]

        self.verify_bmr_pmf_and_counts_exist()

        def negative_log_likelihood(taus: list) -> float:
            return -self.compute_log_likelihood(taus)

        bounds = 4 * [(alpha, 1 - alpha)]
        constraints = {"type": "eq", "fun": lambda tau: sum(tau) - 1}
        result = minimize(
            negative_log_likelihood,
            x0=tau_init,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        if not result.success:
            msg = f"Optimization failed: {result.message}"
            raise ValueError(msg)

        self.tau_00, self.tau_01, self.tau_10, self.tau_11 = result.x

    def estimate_tau_with_em_from_scratch(
        self,
        max_iter: int = 1000,
        tol: float = 1e-3,
        tau_init: list | None = None,
    ) -> None:
        r"""Estimate the tau parameters for interaction using EM algorithm.

        This method iteratively updates the tau parameters,
        \\( \tau = (\tau_{00}, \tau_{01}, \tau_{10}, \tau_{11}) \\),
        to maximize the likelihood of the observed mutation count data for interaction.

        **Algorithm Steps**:

        1. **E-Step**:
           At iteration \\( t \\), given the estimated driver mutation probabilities
           \\( \tau^{(t)} = (\tau_{00}^{(t)}, \tau_{01}^{(t)},
                                \tau_{10}^{(t)}, \tau_{11}^{(t)}) \\),
           compute the responsibilities \\( z_{i,uv}^{(t)} \\) for each pair
           \\( (u,v) \\in \\{0,1\\}^2 \\) and sample \\( i = 1, \\dots, N \\) as:

           .. math::

               z_{i,uv}^{(t)} = \\frac{\\tau_{uv}^{(t)}
                                            \\cdot \\mathbb{P}(P_i = c_i - u)
                                            \\cdot \\mathbb{P}(P_i' = c_i' - v)}
               {\\sum_{(x,y) \\in \\{0,1\\}^2} \\left(
                        \\tau_{xy}^{(t)}
                        \\cdot \\mathbb{P}(P_i = c_i - x)
                        \\cdot \\mathbb{P}(P_i' = c_i' - y)
                \\right)}

           where \\( P_i \\) and \\( P_i' \\) are passenger mutation probabilities.
           and \\( c_i, c_i' \\) are the observed mutation counts.

        2. **M-Step**:
           Given the responsibilities \\( \\bm{z}_i^{(t)} = (z_{i,00}^{(t)},
                                                                z_{i,01}^{(t)},
                                                                z_{i,10}^{(t)},
                                                                z_{i,11}^{(t)}) \\),
           update the tau parameters at iteration \\( t+1 \\) as:

           .. math::

               \\tau_{uv}^{(t+1)} = \\frac{1}{N} \\sum_{i=1}^{N} z_{i,uv}^{(t)}

           for each pair \\( (u,v) \\in \\{0,1\\}^2 \\).

        **Parameters**:
        :param max_iter: (int) Maximum number of iterations for the EM algorithm.
        :param tol: (float) Convergence threshold for log-likelihood improvement.
        :param tau_init: (list of float) Initial guesses for the tau parameters.

        **Returns**:
        :return: (tuple) The estimated values of tau parameters.
        """
        if tau_init is None:
            tau_init = [0.25, 0.25, 0.25, 0.25]

        self.verify_bmr_pmf_and_counts_exist()

        tau_00, tau_01, tau_10, tau_11 = tau_init
        for _ in range(max_iter):
            total_probabilities = self.compute_total_probability(
                tau_00,
                tau_01,
                tau_10,
                tau_11,
            )
            z_i_00 = (
                self.compute_joint_probability(tau_00, 0, 0)
                / total_probabilities
            )
            z_i_01 = (
                self.compute_joint_probability(tau_01, 0, 1)
                / total_probabilities
            )
            z_i_10 = (
                self.compute_joint_probability(tau_10, 1, 0)
                / total_probabilities
            )
            z_i_11 = (
                self.compute_joint_probability(tau_11, 1, 1)
                / total_probabilities
            )

            # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-79
            z_i_00_no_nan = np.nan_to_num(z_i_00, nan=2e-100)
            z_i_01_no_nan = np.nan_to_num(z_i_01, nan=2e-100)
            z_i_10_no_nan = np.nan_to_num(z_i_10, nan=2e-100)
            z_i_11_no_nan = np.nan_to_num(z_i_11, nan=2e-100)
            curr_tau_00 = np.mean(z_i_00_no_nan)
            curr_tau_01 = np.mean(z_i_01_no_nan)
            curr_tau_10 = np.mean(z_i_10_no_nan)
            curr_tau_11 = np.mean(z_i_11_no_nan)

            prev_log_likelihood = self.compute_log_likelihood(
                (tau_00, tau_01, tau_10, tau_11),
            )
            curr_log_likelihood = self.compute_log_likelihood(
                (curr_tau_00, curr_tau_01, curr_tau_10, curr_tau_11),
            )
            if abs(curr_log_likelihood - prev_log_likelihood) < tol:
                break

            tau_00, tau_01, tau_10, tau_11 = (
                curr_tau_00,
                curr_tau_01,
                curr_tau_10,
                curr_tau_11,
            )

        self.tau_00, self.tau_01, self.tau_10, self.tau_11 = (
            tau_00,
            tau_01,
            tau_10,
            tau_11,
        )

    # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-76)
    def estimate_tau_with_em_using_pomegranate(self) -> None:
        """TODO: Add docstring."""
        msg = "Method is not yet implemented."
        raise NotImplementedError(msg)

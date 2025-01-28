"""Represent an interaction between two genes for mutation analysis.

The `Interaction` class models the interaction between two genes and provides
methods to compute statistical metrics (e.g., Fisher's p-values, likelihoods,
odds ratios), validate data, and estimate interaction parameters (e.g., tau values)
using numerical optimization and Expectation-Maximization (EM) algorithms.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix

from dialect.models.gene import Gene


class Interaction:
    """Represent an interaction and provide tools for mutation analysis."""

    def __init__(self, gene_a: Gene, gene_b: Gene) -> None:
        """Initialize an Interaction object to represent the interaction pair of genes.

        :param gene_a (Gene): The first gene in the interaction.
        :param gene_b (Gene): The second gene in the interaction.
        """
        if not isinstance(gene_a, Gene) or not isinstance(gene_b, Gene):
            msg = "Both inputs must be instances of the Gene class."
            raise TypeError(msg)

        self.gene_a = gene_a
        self.gene_b = gene_b
        self.name = f"{gene_a.name}:{gene_b.name}"  # Interaction name
        self.tau_00 = None  # P(D = 0, D' = 0) for genes A and B
        self.tau_01 = None  # P(D = 0, D' = 1) for genes A and B
        self.tau_10 = None  # P(D = 1, D' = 0) for genes A and B
        self.tau_11 = None  # P(D = 1, D' = 1) for genes A and B

        # Metrics from alternative methods
        self.discover_me_qval = None
        self.discover_co_qval = None
        self.fishers_me_qval = None
        self.fishers_co_qval = None

    def __str__(self) -> str:
        """Return a string representation of the Interaction object."""
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

    # ---------------------------------------------------------------------------- #
    #                                UTILITY METHODS                               #
    # ---------------------------------------------------------------------------- #

    def compute_contingency_table(self) -> np.ndarray:
        """Compute the contingency table (confusion matrix) for binarized counts.

        :return: A 2x2 numpy array representing the contingency table.
        """
        gene_a_mutations = (self.gene_a.counts > 0).astype(int)
        gene_b_mutations = (self.gene_b.counts > 0).astype(int)
        return confusion_matrix(gene_a_mutations, gene_b_mutations, labels=[1, 0])

    def get_set_of_cooccurring_samples(self) -> list:
        """Get the list of samples in which both genes have at least one mutation.

        :return: (list) List of sample indices where both genes have >= 1 mutation.
        """
        sample_names = self.gene_a.samples
        cooccurring_samples = [
            sample_names[i]
            for i in range(len(sample_names))
            if self.gene_a.counts[i] > 0 and self.gene_b.counts[i] > 0
        ]
        return sorted(cooccurring_samples)

    def compute_fisher_pvalues(self) -> tuple(float, float):
        """Compute Fisher's exact test p-values for ME and CO.

        The contingency table is derived using the `compute_contingency_table` method.
        - Mutual Exclusivity: Tests whether the two genes are ME (alternative="less").
        - Co-Occurrence: Tests whether the two genes co-occur (alternative="greater").

        :return: (tuple) A tuple containing (me_pval, co_pval):
            - me_pval: Fisher's p-value for mutual exclusivity.
            - co_pval: Fisher's p-value for co-occurrence.
        """
        logging.info(
            "Computing Fisher's exact test p-values for interaction %s.",
            self.name,
        )
        cross_tab = self.compute_contingency_table()
        _, me_pval = fisher_exact(cross_tab, alternative="less")
        _, co_pval = fisher_exact(cross_tab, alternative="greater")
        logging.info(
            "Computed Fisher's p-values of ME: %.3e and CO: %.3e for interaction %s.",
            me_pval,
            co_pval,
            self.name,
        )

        return me_pval, co_pval

    # ---------------------------------------------------------------------------- #
    #                           DATA VALIDATION & LOGGING                          #
    # ---------------------------------------------------------------------------- #

    def verify_bmr_pmf_and_counts_exist(self) -> None:
        """Verify that BMR PMFs and counts exist for both genes in the interaction pair.

        :raises ValueError: If BMR PMFs or counts are not defined.
        """
        if self.gene_a.bmr_pmf is None or self.gene_b.bmr_pmf is None:
            msg = "BMR PMFs are not defined for one or both genes."
            raise ValueError(msg)

        if self.gene_a.counts is None or self.gene_b.counts is None:
            msg = "Counts are not defined for one or both genes."
            raise ValueError(msg)

    def verify_taus_are_valid(self, taus: list, tol: float = 1e-2) -> None:
        """Verify that tau parameters are valid (0 <= tau_i <= 1 and sum(tau) == 1).

        :param taus: (list of float) Tau parameters to validate.
        :param tol: (float) Tolerance for the sum of tau parameters (default: 1e-1).
        :raises ValueError: If any or all tau parameters are invalid.
        """
        if not all(0 <= t <= 1 for t in taus) or not np.isclose(
            sum(taus),
            1,
            atol=tol,
        ):
            logging.info("Invalid tau parameters: %s", taus)
            msg = "Invalid tau parameters. Ensure 0 <= tau_i <= 1 and sum(tau) == 1."
            raise ValueError(
                msg,
            )
        tau_11 = taus[-1]
        if tau_11 == 1:
            logging.warning(
                "Tau_11 is 1 for interaction %s. This is an edge case.",
                self.name,
            )
            msg = "Tau_11 cannot be 1. This leads to log(0) in log-likelihood."
            raise ValueError(
                msg,
            )

    def verify_pi_values(self, pi_a: float, pi_b: float) -> None:
        """Verify that driver probabilities are defined for both genes.

        :param pi_a: (float or None) Driver probability for gene A.
        :param pi_b: (float or None) Driver probability for gene B.
        :return: None if either pi value is not defined.
        :raises ValueError: If both pi values are missing.
        """
        if pi_a is None or pi_b is None:
            logging.warning(
                "Driver probabilities are not defined for genes in interaction %s.",
                self.name,
            )
            msg = "Driver probabilities are not defined for both genes."
            raise ValueError(
                msg,
            )

    # ---------------------------------------------------------------------------- #
    #                        Likelihood & Metric Evaluation                        #
    # ---------------------------------------------------------------------------- #
    def compute_joint_probability(self, tau: float, u: int, v: int) -> np.ndarray:
        """Compute joint probability for tau_uv given counts."""

        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-77
        def safe_get(pmf: dict, c: int, min_val: float = 1e-100) -> float:
            # if c is greater than the max count in pmf
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
        """Compute the total probability for the joint distribution of counts."""

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
        logging.info(
            "Computing likelihood ratio for interaction %s.",
            self.name,
        )

        tau_00, tau_01, tau_10, tau_11 = taus
        driver_a_marginal = tau_10 + tau_11
        driver_b_marginal = tau_01 + tau_11

        tau_null = (
            (1 - driver_a_marginal) * (1 - driver_b_marginal),  #  both genes passengers
            (1 - driver_a_marginal)
            * driver_b_marginal,  # gene a passenger, gene b driver
            driver_a_marginal
            * (1 - driver_b_marginal),  # gene a driver, gene b passenger
            driver_a_marginal * driver_b_marginal,  # both genes drivers
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
        logging.info(
            "Computing log odds ratio for interaction %s.",
            self.name,
        )

        self.verify_taus_are_valid(taus)
        tau_00, tau_01, tau_10, tau_11 = taus

        if tau_01 * tau_10 == 0 or tau_00 * tau_11 == 0:
            logging.warning(
                "Zero encountered in odds ratio computation for interaction %s. "
                "tau_01=%.3e, tau_10=%.3e, tau_00=%.3e, tau_11=%.3e. "
                "Returning None for log odds ratio.",
                self.name,
                tau_01,
                tau_10,
                tau_00,
                tau_11,
            )
            return None  # Return None when numerator or denominator is zero

        log_odds_ratio = np.log((tau_01 * tau_10) / (tau_00 * tau_11))
        logging.info(
            "Computed log odds ratio for interaction %s: %.3e",
            self.name,
            log_odds_ratio,
        )

        return log_odds_ratio

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
        logging.info("Computing Wald statistic for interaction %s.", self.name)

        self.verify_taus_are_valid(taus)
        log_odds_ratio = self.compute_log_odds_ratio(taus)
        if log_odds_ratio is None:
            logging.warning(
                "Log odds ratio is None for interaction %s.",
                self.name,
            )
            return None

        try:
            std_err = np.sqrt(
                (1 / self.tau_01)
                + (1 / self.tau_10)
                + (1 / self.tau_00)
                + (1 / self.tau_11),
            )
        except ZeroDivisionError:
            logging.exception(
                "Division by zero encountered when computing S.E. for %s.",
                self.name,
            )
            return None

        wald_statistic = log_odds_ratio / std_err
        logging.info(
            "Computed Wald statistic for interaction %s: %.3e",
            self.name,
            wald_statistic,
        )
        return wald_statistic

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
        logging.info(
            "Computing rho for interaction %s.",
            self.name,
        )

        self.verify_taus_are_valid(taus)

        tau_0x = self.tau_00 + self.tau_01
        tau_1x = self.tau_10 + self.tau_11
        tau_x0 = self.tau_00 + self.tau_10
        tau_x1 = self.tau_01 + self.tau_11

        if any(t == 0 for t in [tau_0x, tau_1x, tau_x0, tau_x1]):
            logging.warning(
                "Division by zero encountered in rho computation for interaction %s. "
                "Marginals: tau_0*=%s, tau_1*=%s, tau_*0=%s, tau_*1=%s.",
                self.name,
                tau_0x,
                tau_1x,
                tau_x0,
                tau_x1,
            )
            return None

        rho = (self.tau_11 * self.tau_00 - self.tau_01 * self.tau_10) / (
            np.sqrt(tau_0x * tau_1x * tau_x0 * tau_x1)
        )
        logging.info(
            "Computed rho for interaction %s: %.3e",
            self.name,
            rho,
        )
        return rho

    # ---------------------------------------------------------------------------- #
    #                         Parameter Estimation Methods                         #
    # ---------------------------------------------------------------------------- #
    def estimate_tau_with_optimization_using_scipy(
        self,
        tau_init: list | None = None,
        alpha: float = 1e-13,
    ) -> None:
        # ? tau parameters fail verification due to optimization scheme due to bounds
        # TODO @ashuaibi7: https://linear.app/princeton-phd-research/issue/DEV-78
        """Estimate the tau parameters using the SLSQP optimization scheme.

        :param tau_init (list): Initializations for tau parameters.
        :param alpha (float): Small value to avoid edge cases at 0 or 1.
        :return (tuple): The optimized values of (tau_00, tau_01, tau_10, tau_11).
        """
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
        logging.info(
            "Estimating tau parameters using EM algorithm from scratch.",
        )

        self.verify_bmr_pmf_and_counts_exist()

        tau_00, tau_01, tau_10, tau_11 = tau_init
        for _ in range(max_iter):
            # E-Step: Compute responsibilities
            total_probabilities = self.compute_total_probability(
                tau_00,
                tau_01,
                tau_10,
                tau_11,
            )  # denominator in E-Step equation
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
            # remove nans to avoid underflow issues in bmr estimates
            z_i_00_no_nan = np.nan_to_num(z_i_00, nan=2e-100)
            z_i_01_no_nan = np.nan_to_num(z_i_01, nan=2e-100)
            z_i_10_no_nan = np.nan_to_num(z_i_10, nan=2e-100)
            z_i_11_no_nan = np.nan_to_num(z_i_11, nan=2e-100)
            # M-Step: Update tau parameters
            curr_tau_00 = np.mean(z_i_00_no_nan)
            curr_tau_01 = np.mean(z_i_01_no_nan)
            curr_tau_10 = np.mean(z_i_10_no_nan)
            curr_tau_11 = np.mean(z_i_11_no_nan)

            # Check for convergence
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
        """Estimate the tau parameters using the pomegranate library."""
        logging.info("Estimating tau parameters using pomegranate.")
        msg = "Method is not yet implemented."
        raise NotImplementedError(msg)

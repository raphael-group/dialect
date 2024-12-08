import logging
import numpy as np
from scipy.optimize import minimize

from dialect.models.gene import Gene


class Interaction:
    def __init__(self, gene_a, gene_b):
        """
        Initialize an Interaction object to represent the interaction between two genes.

        :param gene_a (Gene): The first gene in the interaction.
        :param gene_b (Gene): The second gene in the interaction.
        """
        if not isinstance(gene_a, Gene) or not isinstance(gene_b, Gene):
            raise ValueError("Both inputs must be instances of the Gene class.")

        self.gene_a = gene_a
        self.gene_b = gene_b
        self.name = f"{gene_a.name}:{gene_b.name}"  # Interaction name
        self.tau_00 = None  # P(D = 0, D' = 0) for genes A and B
        self.tau_01 = None  # P(D = 0, D' = 1) for genes A and B
        self.tau_10 = None  # P(D = 1, D' = 0) for genes A and B
        self.tau_11 = None  # P(D = 1, D' = 1) for genes A and B

    # ---------------------------------------------------------------------------- #
    #                        Likelihood & Metric Evaluation                        #
    # ---------------------------------------------------------------------------- #

    def compute_log_likelihood(self, taus):
        """
        Compute the complete data log-likelihood for the interaction given the parameters tau.

        The log-likelihood function is given by:
            \sum_{i=1}^{N} \log ()
                \mathbb{P}(P_i = c_i)\mathbb{P}(P_i' = c_i') \tau_{00} +
                \mathbb{P}(P_i = c_i)\mathbb{P}(P_i' = c_i' - 1) \tau_{01} +
                \mathbb{P}(P_i = c_i - 1)\mathbb{P}(P_i' = c_i') \tau_{10} +
                \mathbb{P}(P_i = c_i - 1)\mathbb{P}(P_i' = c_i' - 1) \tau_{11}
            )

        where:
            - `N` is the number of samples.
            - `P_i` and `P_i'` represent the RVs for passenger mutations for the two genes.
            - `c_i` and `c_i'` are the observed counts of somatic mutations for the two genes.
            - `tau = (tau_00, tau_01, tau_10, tau_11)` are the interaction parameters.

        :param tau (tuple): A tuple (tau_00, tau_01, tau_10, tau_11) representing the interaction parameters.
        :return (float): The log-likelihood value.
        :raises ValueError: If `bmr_pmf` or `counts` are not defined for either gene, or if `tau` is invalid.
        """
        if not self.gene_a.bmr_pmf or not self.gene_b.bmr_pmf:
            raise ValueError("BMR PMFs are not defined for one or both genes.")
        if not self.gene_a.counts or not self.gene_b.counts:
            raise ValueError("Counts are not defined for one or both genes.")
        if not all(0 <= t <= 1 for t in taus) or not np.isclose(sum(taus), 1):
            logging.info(f"Invalid tau parameters: {taus}")
            raise ValueError(
                "Invalid tau parameters. Ensure 0 <= tau_i <= 1 and sum(tau) == 1."
            )

        logging.info(f"Computing log likelihood for {self.name}. Taus: {taus}")

        a_counts, b_counts = self.gene_a.counts, self.gene_b.counts
        a_bmr_pmf, b_bmr_pmf = self.gene_a.bmr_pmf, self.gene_b.bmr_pmf
        tau_00, tau_01, tau_10, tau_11 = taus
        log_likelihood = sum(
            np.log(
                a_bmr_pmf.get(c_a, 0) * b_bmr_pmf.get(c_b, 0) * tau_00
                + a_bmr_pmf.get(c_a, 0) * b_bmr_pmf.get(c_b - 1, 0) * tau_01
                + a_bmr_pmf.get(c_a - 1, 0) * b_bmr_pmf.get(c_b, 0) * tau_10
                + a_bmr_pmf.get(c_a - 1, 0) * b_bmr_pmf.get(c_b - 1, 0) * tau_11
            )
            for c_a, c_b in zip(a_counts, b_counts)
        )
        return log_likelihood

    def compute_likelihood_ratio(self):
        """
        Compute the likelihood ratio test statistic (lambda_LR) with respect to the null hypothesis.

        The likelihood ratio statistic is given by:
            lambda_LR = -2 * [\ell(tau_null) - \ell(\hat{tau})]

        where:
            - \ell(): Log-likelihood.
            - tau_null: Null hypothesis of no interaction (independent given individual gene driver probabilites).
            - \hat{tau}: Estimated values of tau parameters.

        :return (float): Likelihood ratio.
        """
        pi_a, pi_b = self.gene_a.pi, self.gene_b.pi
        if pi_a is None or pi_b is None:
            logging.warning(
                f"Driver probabilities (pi) are not defined for genes in interaction {self.name}."
            )
            return None

        tau_null = (
            (1 - pi_a) * (1 - pi_b),  # tau_00: neither gene has a driver mutation
            (1 - pi_a) * pi_b,  # tau_01: only gene_b has a driver mutation
            pi_a * (1 - pi_b),  # tau_10: only gene_a has a driver mutation
            pi_a * pi_b,  # tau_11: both genes have driver mutations
        )
        lambda_LR = -2 * (
            self.compute_log_likelihood(tau_null)
            - self.compute_log_likelihood(
                (self.tau_00, self.tau_01, self.tau_10, self.tau_11)
            )
        )
        return lambda_LR

    def compute_log_odds_ratio(self):
        """
        Compute the log odds ratio for the interaction based on the tau parameters.

        The odds ratio is given by:
            Odds Ratio = (tau_01 * tau_10) / (tau_00 * tau_11)

        :return (float): The log odds ratio.
        :raises ValueError: If tau parameters are invalid or lead to division by zero.
        """
        # Validate tau parameters
        if not all(
            0 <= t <= 1 for t in [self.tau_00, self.tau_01, self.tau_10, self.tau_11]
        ) or not np.isclose(
            sum([self.tau_00, self.tau_01, self.tau_10, self.tau_11]), 1
        ):
            logging.info(
                f"Invalid tau parameters: tau_00={self.tau_00}, tau_01={self.tau_01}, tau_10={self.tau_10}, tau_11={self.tau_11}"
            )
            raise ValueError(
                "Invalid tau parameters. Ensure 0 <= tau_ij <= 1 and sum(tau) == 1."
            )

        if self.tau_01 * self.tau_10 == 0 or self.tau_00 * self.tau_11 == 0:
            logging.warning(
                f"Zero encountered in odds ratio computation for interaction {self.name}. "
                f"tau_01={self.tau_01}, tau_10={self.tau_10}, tau_00={self.tau_00}, tau_11={self.tau_11}"
            )
            return None  # Return None when numerator or denominator is zero

        log_odds_ratio = np.log(
            (self.tau_01 * self.tau_10) / (self.tau_00 * self.tau_11)
        )
        logging.info(
            f"Computed log odds ratio for interaction {self.name}: {log_odds_ratio}"
        )
        return log_odds_ratio

    def compute_wald_statistic(self):
        """
        Compute the Wald statistic for the interaction.

        The Wald statistic is given by:
            W = log_odds_ratio / standard_error

        where the standard error is calculated as:
            std_err = sqrt(
                (1 / tau_01) + (1 / tau_10) + (1 / tau_00) + (1 / tau_11)
            )

        :return (float or None): The Wald statistic, or None
        """
        logging.info(f"Computing Wald statistic for interaction {self.name}.")
        log_odds_ratio = self.compute_log_odds_ratio()
        if log_odds_ratio is None:
            logging.warning(f"Log odds ratio is None for interaction {self.name}.")
            return None

        if any(t <= 0 for t in [self.tau_00, self.tau_01, self.tau_10, self.tau_11]):
            logging.warning(
                f"Invalid tau parameters for interaction {self.name}. "
                f"tau_00={self.tau_00}, tau_01={self.tau_01}, tau_10={self.tau_10}, tau_11={self.tau_11}."
                "All tau values must be positive to compute the Wald statistic."
            )
            raise ValueError("Invalid tau parameters for Wald statistic computation.")

        try:
            std_err = np.sqrt(
                (1 / self.tau_01)
                + (1 / self.tau_10)
                + (1 / self.tau_00)
                + (1 / self.tau_11)
            )
        except ZeroDivisionError:
            logging.error(
                f"Division by zero encountered when computing standard error for interaction {self.name}."
            )
            return None

        wald_statistic = log_odds_ratio / std_err
        logging.info(
            f"Computed Wald statistic for interaction {self.name}: {wald_statistic}"
        )
        return wald_statistic

    def compute_rho(self):
        """
        Compute the interaction measure rho (ρ) for the given tau parameters - ρ is given by:
            ρ = (tau_01 * tau_10 - tau_11 * tau_00) / sqrt(tau_0* * tau_1* * tau_*0 * tau_*1)

        where:
            - tau_0* = tau_00 + tau_01 (marginal for no driver mutation in gene_a)
            - tau_1* = tau_10 + tau_11 (marginal for driver mutation in gene_a)
            - tau_*0 = tau_00 + tau_10 (marginal for no driver mutation in gene_b)
            - tau_*1 = tau_01 + tau_11 (marginal for driver mutation in gene_b)

        :return (float or None): The value of rho, or None if the computation is invalid (e.g., division by zero).
        """
        if not all(
            0 <= t <= 1 for t in [self.tau_00, self.tau_01, self.tau_10, self.tau_11]
        ) or not np.isclose(
            sum([self.tau_00, self.tau_01, self.tau_10, self.tau_11]), 1
        ):
            logging.warning(
                f"Invalid tau parameters for interaction {self.name}: "
                f"tau_00={self.tau_00}, tau_01={self.tau_01}, tau_10={self.tau_10}, tau_11={self.tau_11}."
            )
            raise ValueError(
                "Invalid tau parameters. Ensure 0 <= tau_ij <= 1 and sum(tau) == 1."
            )

        tau_0X = self.tau_00 + self.tau_01
        tau_1X = self.tau_10 + self.tau_11
        tau_X0 = self.tau_00 + self.tau_10
        tau_X1 = self.tau_01 + self.tau_11

        if any(t == 0 for t in [tau_0X, tau_1X, tau_X0, tau_X1]):
            logging.warning(
                f"Division by zero encountered in rho computation for interaction {self.name}. "
                f"Marginals: tau_0*={tau_0X}, tau_1*={tau_1X}, "
                f"tau_*0={tau_X0}, tau_*1={tau_X1}."
            )
            return None

        rho = (self.tau_01 * self.tau_10 - self.tau_11 * self.tau_00) / (
            np.sqrt(tau_0X * tau_1X * tau_X0 * tau_X1)
        )
        logging.info(f"Computed rho for interaction {self.name}: {rho}")
        return rho

    # ---------------------------------------------------------------------------- #
    #                         Parameter Estimation Methods                         #
    # ---------------------------------------------------------------------------- #
    def estimate_tau_with_optimization_using_scipy(
        self, tau_init=[0.25, 0.25, 0.25, 0.25], alpha=1e-13
    ):
        """
        Estimate the tau parameters using the L-BFGS-B optimization scheme.

        :param tau_init (list): Initial guesses for the tau parameters (default: [0.25, 0.25, 0.25, 0.25]).
        :param alpha (float): Small value to avoid edge cases at 0 or 1 (default: 1e-13).
        :return (tuple): The optimized values of (tau_00, tau_01, tau_10, tau_11).
        """
        logging.info(f"Estimating tau params for {self.name} using L-BFGS-B.")

        def negative_log_likelihood(tau):
            return -self.compute_log_likelihood(tau)

        bounds = 4 * [(alpha, 1 - alpha)]
        constraints = {"type": "eq", "fun": lambda tau: sum(tau) - 1}
        result = minimize(
            negative_log_likelihood,
            x0=tau_init,
            bounds=bounds,
            constraints=constraints,
            method="L-BFGS-B",
        )
        if not result.success:
            logging.warning(
                f"Optimization failed for interaction {self.name}: {result.message}"
            )
            raise ValueError(f"Optimization failed: {result.message}")

        self.tau_00, self.tau_01, self.tau_10, self.tau_11 = result.x
        logging.info(
            f"Estimated tau parameters for interaction {self.name}: tau_00={self.tau_00}, tau_01={self.tau_01}, tau_10={self.tau_10}, tau_11={self.tau_11}"
        )
        return self.tau_00, self.tau_01, self.tau_10, self.tau_11

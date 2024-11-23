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
            - tau_null: Null hypothesis of no interaction (tau_00 = 1, tau_01 = tau_10 = tau_11 = 0).
            - \hat{tau}: Estimated values of tau parameters.

        :return (float): Likelihood ratio.
        """
        tau_null = (1, 0, 0, 0)  # Null hypothesis: no interaction
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

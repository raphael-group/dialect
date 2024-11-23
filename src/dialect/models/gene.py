import logging
import numpy as np
from scipy.optimize import minimize


class Gene:
    def __init__(self, name, counts, bmr_pmf):
        """
        Initialize a Gene object.

        :param name (str): Name of the gene.
        :param counts (np.ndarray) Mutation counts for the gene.
        :param bmr_pmf (defaultdict): BMR PMF (multinomial passed as single list).
        """
        self.name = name
        self.counts = counts
        self.bmr_pmf = bmr_pmf
        self.pi = None

    # ---------------------------------------------------------------------------- #
    #                        Likelihood & Metric Evaluation                        #
    # ---------------------------------------------------------------------------- #

    def compute_log_likelihood(self, pi):
        """
        Compute the complete data log-likelihood for the gene given the estimated pi.

        The likelihood function is given by:
            \sum_{i=1}^{N} \log(\mathbb{P}(P_i = c_i)(1 - \pi) + \mathbb{P}(P_i = c_i - 1) \pi)

        where:
            - `N` is the number of samples.
            - `P_i` represents the RV for passenger mutations
            - `c_i` is the observed count of somatic mutations for sample i
            - `\pi` is the estimated driver mutation rate parameter value.

        return (float): The log-likelihood value.
        raises (ValueError): If `bmr_pmf`, `counts`, or `pi` is not defined.
        """
        if not self.bmr_pmf:
            raise ValueError("BMR PMF is not defined for this gene.")
        if not self.counts:
            raise ValueError("Counts are not defined for this gene.")

        logging.info(
            f"Computing log likelihood for gene {self.name}.  Pi: {pi}. BMR PMF: {self.bmr_pmf}"
        )

        log_likelihood = sum(
            np.log(self.bmr_pmf.get(c, 0) * (1 - pi) + self.bmr_pmf.get(c - 1, 0) * pi)
            for c in self.counts
        )
        return log_likelihood

    def compute_likelihood_ratio(self):
        """
        Compute the likelihood ratio test statistic (lambda_LR) with respect to the null hypothesis.

        The likelihood ratio statistic is given by:
            lambda_LR = -2 * [\ell(pi_null) - \ell(\hat{pi})]

        where:
            - \ell(): Log-likelihood.
            - pi_null: Null hypothesis of no driver mutations (pi = 0).
            - \hat{pi}: Estimated value of pi parameter.

        :return (float): Likelihood ratio.
        """
        lambda_LR = -2 * (
            self.compute_log_likelihood(0) - self.compute_log_likelihood(self.pi)
        )
        return lambda_LR

    def compute_log_odds_ratio(self):
        """
        Compute the log odds ratio from the contingency table.

        :return (float): Log odds ratio.
        """
        if not self.pi:
            raise ValueError("Pi has not been esitmated for this gene.")
        if not 0 <= self.pi <= 1:
            logging.info(f"Pi value out of bounds: {self.pi}")
            raise ValueError("Estimated pi is out of bounds.")
        if self.pi == 0 or self.pi == 1:
            logging.info(f"Pi for gene {self.name} is 0 or 1")
            return np.inf if self.pi else -np.inf

        log_odds_ratio = np.log(self.pi / (1 - self.pi))
        return log_odds_ratio

    # ---------------------------------------------------------------------------- #
    #                         Parameter Estimation Methods                         #
    # ---------------------------------------------------------------------------- #
    def estimate_pi_with_optimiziation_using_scipy(self, pi_init=0.5):
        """
        Estimate the pi parameter using the L-BFGS-B optimization scheme.
        L-BFGS-B is used because it supports bounds (0 < pi < 1), which are required
        to handle constraints and ensure valid log-likelihood computations.

        :param pi_init (float): Initial guess for the pi parameter (default: 0.5).
        :return (float): The optimized value of pi.
        """

        def negative_log_likelihood(pi):
            return -self.compute_log_likelihood(pi)

        logging.info(f"Estimating pi for gene {self.name} using L-BFGS-B optimization.")

        alpha = 1e-13  # Small value to avoid edge cases at 0 or 1
        bounds = [(alpha, 1 - alpha)]  # Restrict pi to (0, 1) to avoid log issues
        result = minimize(
            negative_log_likelihood,
            x0=[pi_init],  # Initial guess for pi
            bounds=bounds,
            method="L-BFGS-B",  # Recommended for bounded optimization
        )

        if not result.success:
            logging.warning(
                f"Optimization failed for gene {self.name}: {result.message}"
            )
            raise ValueError(f"Optimization failed: {result.message}")

        self.pi = result.x[0]
        logging.info(f"Estimated pi for gene {self.name}: {self.pi}")

    def estimate_pi_with_em_from_scratch(self):
        """
        Estimate the pi parameter using the Expectation-Maximization (EM) algorithm.
        Implements the EM algorithm from scratch.
        """
        logging.info(f"Estimating pi for gene {self.name} using the EM algorithm.")
        raise NotImplementedError("EM algorithm not implemented yet.")

    def estimate_pi_with_em_using_pomegranate(self):
        """
        Estimate the pi parameter using the Expectation-Maximization (EM) algorithm.
        Uses the Pomegranate library for the EM algorithm.
        """
        logging.info(f"Estimating pi for gene {self.name} using the EM algorithm.")
        raise NotImplementedError("EM algorithm not implemented yet.")

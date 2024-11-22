import logging
import numpy as np


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

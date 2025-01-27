import logging
import itertools
import numpy as np
from scipy.optimize import minimize

# TODO: Create essential and verbose logging info for all methods


class Gene:
    def __init__(self, name, samples, counts, bmr_pmf):
        """
        Initialize a Gene object.

        :param name (str): Name of the gene.
        :param counts (np.ndarray) Mutation counts for the gene.
        :param bmr_pmf (defaultdict): BMR PMF (multinomial passed as single list).
        """
        self.name = name
        self.samples = samples
        self.counts = counts
        self.bmr_pmf = bmr_pmf
        # TODO: change all instances of bmr_pmf to bmr_pmf_arr
        self.pi = None

        # Metrics from alternative methods
        self.cbase_phi = None
        self.cbase_p = None

    def __str__(self):
        """
        Return a string representation of the Gene object.
        """
        bmr_preview = ", ".join(
            f"{k}: {v:.3e}"
            for k, v in itertools.islice(self.bmr_pmf.items(), 3)
        )  # Format the first three key-value pairs in bmr_pmf
        pi_info = (
            f"Pi: {self.pi:.3e}"
            if self.pi is not None
            else "Pi: Not estimated"
        )
        total_mutations = np.sum(self.counts)
        return (
            f"Gene: {self.name}\n"
            f"Total Mutations: {total_mutations}\n"
            f"BMR PMF (preview): {{ {bmr_preview} }}\n"
            f"{pi_info}"
        )

    # ---------------------------------------------------------------------------- #
    #                               UTILITY FUNCTIONS                              #
    # ---------------------------------------------------------------------------- #
    def calculate_expected_mutations(self):
        """
        Calculate the expected number of mutations for the gene based on its BMR PMF.

        :return: (float) The expected number of mutations.
        """
        total_samples = len(self.counts)
        expected_mutations = sum(k * prob for k, prob in self.bmr_pmf.items())
        return expected_mutations * total_samples

    # ---------------------------------------------------------------------------- #
    #                          DATA VALIDATION AND LOGGING                         #
    # ---------------------------------------------------------------------------- #
    def verify_bmr_pmf_and_counts_exist(self):
        """
        Verify that the BMR PMF and counts are defined for this gene.

        :raises ValueError: If `bmr_pmf` or `counts` is not defined.
        """
        if self.bmr_pmf is None:
            raise ValueError("BMR PMF is not defined for this gene.")
        if self.counts is None:
            raise ValueError("Counts are not defined for this gene.")

    def verify_bmr_pmf_contains_all_count_keys(self):
        """
        Check if all count keys in `counts` exist in `bmr_pmf`.

        Logs a warning if any counts are missing in `bmr_pmf` and skips those samples.
        Issue occurs when BMR PMF does not include all possible counts in categorical distribution.
        """
        missing_bmr_pmf_counts = [
            c for c in self.counts if c not in self.bmr_pmf
        ]
        if missing_bmr_pmf_counts:
            logging.warning(
                f"Counts {missing_bmr_pmf_counts} are not in bmr_pmf for gene {self.name}."
                f"Please ensure bmr_pmf includes all relevant counts."
            )
            raise ValueError(
                "BMR PMF does not contain all possible counts as keys in categorical distribution."
            )

    # TODO: move to central utility file
    def verify_pi_is_valid(self, pi):
        """
        Validate that the estimated pi value is defined and within the valid range [0, 1].

        Logs additional information for boundary or invalid values.

        :raises ValueError: If `pi` is not defined or out of bounds.
        :return: `np.inf` if `pi` is 1, `-np.inf` if `pi` is 0.
        """
        if pi is None or not 0 <= pi <= 1:
            raise ValueError(
                f"Invalid pi value: {pi}. Pi must be in the range [0, 1]."
            )
        if pi == 1:
            logging.info(
                f"Pi for gene {self.name} is 1, which will result in log(0) issues when computing log likelihood."
            )
            raise ValueError(
                "Estimated pi is 1. Please ensure pi is less than 1."
            )

    # ---------------------------------------------------------------------------- #
    #                        Likelihood & Metric Evaluation                        #
    # ---------------------------------------------------------------------------- #
    def compute_log_likelihood(self, pi):
        """
        Compute the complete data log-likelihood for the gene given the estimated \( \pi \).

        The log-likelihood function is defined as:

        .. math::

            \\ell_C(\\pi) = \\sum_{i=1}^{N} \\log \\big(\\mathbb{P}(P_i = c_i)(1 - \\pi) + \\mathbb{P}(P_i = c_i - 1) \\pi\\big)

        where:

        - \( N \): Number of samples.
        - \( P_i \): Random variable representing passenger mutations.
        - \( c_i \): Observed count of somatic mutations for sample \( i \).
        - \( \\pi \): Estimated driver mutation rate parameter value.

        **Returns**:
        :return: (float) The computed log-likelihood value.

        **Raises**:
        :raises ValueError: If `bmr_pmf`, `counts`, or `pi` is not properly defined.
        """

        # logging.info(
        #     f"Computing log likelihood for gene {self.name}. Pi: {pi:.3e}. "
        #     f"BMR PMF: {{ {', '.join(f'{k}: {v:.3e}' for k, v in itertools.islice(self.bmr_pmf.items(), 3))} }}"
        # )
        # TODO: move to unified helper scripts file
        def safe_get_no_default(pmf, c, min_val=1e-100):
            # if c is greater than the max count in pmf
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c)

        def safe_get_with_default(pmf, c, min_val=1e-100):
            # if c is greater than the max count in pmf
            if c > max(pmf.keys()):
                return min_val
            return pmf.get(c, 0)

        self.verify_pi_is_valid(pi)
        self.verify_bmr_pmf_and_counts_exist()
        # self.verify_bmr_pmf_contains_all_count_keys()

        log_likelihood = sum(
            np.log(
                safe_get_no_default(self.bmr_pmf, c) * (1 - pi)
                + safe_get_with_default(self.bmr_pmf, c - 1) * pi
            )
            for c in self.counts
        )

        return log_likelihood

    def compute_likelihood_ratio(self, pi):
        """
        Compute the likelihood ratio test statistic (\( \lambda_{LR} \)) with respect to the null hypothesis.

        The likelihood ratio test statistic is calculated as:

        .. math::

            \lambda_{LR} = -2 \\left[ \\ell(\\pi_{\\text{null}}) - \\ell(\\hat{\\pi}) \\right]

        where:

        - \( \\ell() \): Log-likelihood function.
        - \( \\pi_{\\text{null}} \): Null hypothesis value (e.g., \( \\pi = 0 \), indicating no driver mutations).
        - \( \\hat{\\pi} \): Estimated value of the \( \\pi \) parameter under the alternative hypothesis.

        **Returns**:
        :return: (float) The likelihood ratio test statistic (\( \lambda_{LR} \)).
        """
        logging.info(f"Computing likelihood ratio for gene {self.name}.")

        self.verify_pi_is_valid(pi)

        lambda_LR = -2 * (
            self.compute_log_likelihood(0) - self.compute_log_likelihood(pi)
        )
        return lambda_LR

    def compute_log_odds_ratio(self, pi):
        """
        Compute the log odds ratio.

        The log odds ratio is calculated as:

        .. math::

            L = \\log\\left(\\frac{\\tau_{01} \\tau_{10}}{\\tau_{00} \\tau_{11}}\\right)

        **Returns**:
        :return: (float) The log odds ratio.
        """

        logging.info(f"Computing log odds ratio for gene {self.name}.")

        self.verify_pi_is_valid(pi)

        log_odds_ratio = np.log(self.pi / (1 - pi))
        return log_odds_ratio

    # ---------------------------------------------------------------------------- #
    #                         Parameter Estimation Methods                         #
    # ---------------------------------------------------------------------------- #
    def estimate_pi_with_optimiziation_using_scipy(self, pi_init=0.5):
        """
        Estimate the pi parameter using the SLSQP optimization scheme.
        SLSQP is used because it supports bounds (0 < pi < 1), which are required
        to handle constraints and ensure valid log-likelihood computations.

        :param pi_init (float): Initial guess for the pi parameter (default: 0.5).
        :return (float): The optimized value of pi.
        """

        def negative_log_likelihood(pi):
            return -self.compute_log_likelihood(pi)

        logging.info(
            f"Estimating pi for gene {self.name} using SLSQP optimization."
        )

        self.verify_bmr_pmf_and_counts_exist()

        alpha = 1e-13  # Small value to avoid edge cases at 0 or 1
        bounds = [
            (alpha, 1 - alpha)
        ]  # Restrict pi to (0, 1) to avoid log issues
        result = minimize(
            negative_log_likelihood,
            x0=[pi_init],  # Initial guess for pi
            bounds=bounds,
            method="SLSQP",
        )

        if not result.success:
            logging.warning(
                f"Optimization failed for gene {self.name}: {result.message}"
            )
            raise ValueError(f"Optimization failed: {result.message}")

        self.pi = result.x[0]
        logging.info(f"Estimated pi for gene {self.name}: {self.pi}")

    def estimate_pi_with_em_from_scratch(
        self, max_iter=1000, tol=1e-3, pi_init=0.5
    ):
        """
        Estimate the pi parameter using the Expectation-Maximization (EM) algorithm.

        This method iteratively updates the parameter \( \pi \) to maximize the likelihood of the observed data.

        **Algorithm Steps**:

        1. **E-Step**: Compute the responsibilities (\( z_i^{(t)} \)) as:

           .. math::

               z_{i}^{(t)} = \\frac{\\pi^{(t)} \\cdot \\mathbb{P}(P_i = c_i - 1)}
               {\\pi^{(t)} \\cdot \\mathbb{P}(P_i = c_i - 1) + (1 - \\pi^{(t)}) \\cdot \\mathbb{P}(P_i = c_i)}

        2. **M-Step**: Update the parameter \( \pi \) as:

           .. math::

               \\pi^{(t+1)} = \\frac{1}{N} \\sum_{i=1}^{N} z_{i}^{(t)}

        **Parameters**:
        :param max_iter: (int) Maximum number of iterations (default: 1000).
        :param epsilon: (float) Convergence threshold for log-likelihood improvement (default: 1e-6).
        :param pi_init: (float) The initialization value for \( \pi \).

        **Returns**:
        :return: (float) The estimated value of \( \pi \).
        """
        logging.info(
            f"Estimating pi for gene {self.name} using the EM algorithm."
        )

        self.verify_bmr_pmf_and_counts_exist()
        # self.verify_bmr_pmf_contains_all_count_keys()

        nonzero_probability_counts = [
            c for c in self.counts if c in self.bmr_pmf and self.bmr_pmf[c] > 0
        ]  # exclude counts with zero probability to avoid log(0) issues

        pi = pi_init
        for it in range(max_iter):
            z_i = [
                (pi * self.bmr_pmf.get(c - 1, 0))
                / (
                    pi * self.bmr_pmf.get(c - 1, 0)
                    + (1 - pi) * self.bmr_pmf.get(c, 0)
                )
                for c in nonzero_probability_counts
            ]  # E-step

            curr_pi = np.mean(z_i)  # M-step

            # Check convergence
            prev_log_likelihood = self.compute_log_likelihood(pi)
            curr_log_likelihood = self.compute_log_likelihood(curr_pi)
            if abs(curr_log_likelihood - prev_log_likelihood) < tol:
                logging.info(f"EM algorithm converged after {it} iterations.")
                break

            pi = curr_pi

        self.pi = pi
        logging.info(f"Estimated pi for gene {self.name}: {self.pi:.4f}")

    # TODO: (LOW PRIORITY): Implement EM w/ Pomegranate for Speed Improvement
    def estimate_pi_with_em_using_pomegranate(self):
        """
        Estimate the pi parameter using the Expectation-Maximization (EM) algorithm.
        Uses the Pomegranate library for the EM algorithm.
        """
        logging.info(
            f"Estimating pi for gene {self.name} using the EM algorithm."
        )
        raise NotImplementedError("EM algorithm not implemented yet.")

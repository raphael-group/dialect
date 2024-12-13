import logging
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix

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

    def __str__(self):
        """
        Return a string representation of the Interaction object.
        """
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
            f"\nContingency Table:\n"
            f"[[{cm[0, 0]} {cm[0, 1]}]\n"
            f" [{cm[1, 0]} {cm[1, 1]}]]"
        )

        return (
            f"Interaction: {self.name}\n"
            f"Gene A: {self.gene_a.name} (Pi: {pi_a})\n"
            f"Gene B: {self.gene_b.name} (Pi: {pi_b})\n"
            f"Tau Parameters: {taus_info}\n"
            f"Contingency Table:{cm_info}"
        )

    # ---------------------------------------------------------------------------- #
    #                           DATA VALIDATION & LOGGING                          #
    # ---------------------------------------------------------------------------- #

    def verify_bmr_pmf_and_counts_exist(self):
        """
        Verify that BMR PMFs and counts exist for both genes in the interaction pair.

        :raises ValueError: If BMR PMFs or counts are not defined.
        """
        if self.gene_a.bmr_pmf is None or self.gene_b.bmr_pmf is None:
            raise ValueError("BMR PMFs are not defined for one or both genes.")

        if self.gene_a.counts is None or self.gene_b.counts is None:
            raise ValueError("Counts are not defined for one or both genes.")

    def verify_taus_are_valid(self, taus, tol=1e-6):
        """
        Verify that tau parameters are valid (0 <= tau_i <= 1 and sum(tau) == 1).

        :param taus: (list of float) Tau parameters to validate.
        :param tol: (float) Tolerance for the sum of tau parameters (default: 1e-1).
        :raises ValueError: If any or all tau parameters are invalid.
        """
        if not all(0 <= t <= 1 for t in taus) or not np.isclose(sum(taus), 1, atol=tol):
            logging.info(f"Invalid tau parameters: {taus}")
            raise ValueError(
                "Invalid tau parameters. Ensure 0 <= tau_i <= 1 and sum(tau) == 1."
            )

    def verify_pi_values(self, pi_a, pi_b):
        """
        Verify that driver probabilities (pi values) are defined for both genes in the interaction.

        :param pi_a: (float or None) Driver probability for gene A.
        :param pi_b: (float or None) Driver probability for gene B.
        :return: None if either pi value is not defined.
        :raises ValueError: If both pi values are missing.
        """
        if pi_a is None or pi_b is None:
            logging.warning(
                f"Driver probabilities (pi) are not defined for genes in interaction {self.name}."
            )
            raise ValueError("Driver probabilities are not defined for both genes.")

    # TODO: Add verification for counts existing in BMR PMFs
    # ? do we want to exclude these counts from log likelihood calculation?
    # ? does the existing code handle this in the EM estimation?

    # ---------------------------------------------------------------------------- #
    #                        Likelihood & Metric Evaluation                        #
    # ---------------------------------------------------------------------------- #
    # TODO (LOW PRIORITY): Add additional metrics (KL, MI, etc.)

    def compute_contingency_table(self):
        """
        Compute the contingency table (confusion matrix) for binarized counts
        between gene_a and gene_b.

        :return: A 2x2 numpy array representing the contingency table.
        """
        gene_a_mutations = (self.gene_a.counts > 0).astype(int)
        gene_b_mutations = (self.gene_b.counts > 0).astype(int)
        cm = confusion_matrix(gene_a_mutations, gene_b_mutations, labels=[0, 1])
        return cm

    def compute_joint_probability(self, tau, u, v):
        joint_probability = np.array(
            [
                tau
                * self.gene_a.bmr_pmf.get(c_a - u, 0)
                * self.gene_b.bmr_pmf.get(c_b - v, 0)
                for c_a, c_b in zip(self.gene_a.counts, self.gene_b.counts)
            ]
        )
        return joint_probability

    def compute_total_probability(self, tau_00, tau_01, tau_10, tau_11):
        total_probabilities = np.array(
            [
                sum(
                    (
                        tau_00
                        * self.gene_a.bmr_pmf.get(c_a, 0)
                        * self.gene_b.bmr_pmf.get(c_b, 0),
                        tau_01
                        * self.gene_a.bmr_pmf.get(c_a, 0)
                        * self.gene_b.bmr_pmf.get(c_b - 1, 0),
                        tau_10
                        * self.gene_a.bmr_pmf.get(c_a - 1, 0)
                        * self.gene_b.bmr_pmf.get(c_b, 0),
                        tau_11
                        * self.gene_a.bmr_pmf.get(c_a - 1, 0)
                        * self.gene_b.bmr_pmf.get(c_b - 1, 0),
                    )
                )
                for c_a, c_b in zip(self.gene_a.counts, self.gene_b.counts)
            ]
        )
        return total_probabilities

    def compute_log_likelihood(self, taus):
        """
        Compute the complete data log-likelihood for the interaction given the parameters \( \tau \).

        The log-likelihood function is defined as:

        .. math::

            \\ell_C(\\tau) = \\sum_{i=1}^{N} \\log \\Big(
                \\mathbb{P}(P_i = c_i) \\mathbb{P}(P_i' = c_i') \\tau_{00} +
                \\mathbb{P}(P_i = c_i) \\mathbb{P}(P_i' = c_i' - 1) \\tau_{01} +
                \\mathbb{P}(P_i = c_i - 1) \\mathbb{P}(P_i' = c_i') \\tau_{10} +
                \\mathbb{P}(P_i = c_i - 1) \\mathbb{P}(P_i' = c_i' - 1) \\tau_{11}
            \\Big)

        where:

        - \( N \): Number of samples.
        - \( P_i \) and \( P_i' \): Random variables representing passenger mutations for the two genes.
        - \( c_i \) and \( c_i' \): Observed counts of somatic mutations for the two genes.
        - \( \\tau = (\\tau_{00}, \\tau_{01}, \\tau_{10}, \\tau_{11}) \): Interaction parameters.

        **Parameters**:
        :param tau: (tuple) A tuple \( (\\tau_{00}, \\tau_{01}, \\tau_{10}, \\tau_{11}) \) representing the interaction parameters.

        **Returns**:
        :return: (float) The computed log-likelihood value.

        **Raises**:
        :raises ValueError: If `bmr_pmf` or `counts` are not defined for either gene, or if `tau` is invalid.
        """

        logging.info(f"Computing log likelihood for {self.name}. Taus: {taus}")

        self.verify_bmr_pmf_and_counts_exist()
        self.verify_taus_are_valid(taus)

        a_counts, b_counts = self.gene_a.counts, self.gene_b.counts
        a_bmr_pmf, b_bmr_pmf = self.gene_a.bmr_pmf, self.gene_b.bmr_pmf
        tau_00, tau_01, tau_10, tau_11 = taus
        # TODO: Moddify all passenger key access to not default to 0
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

    def compute_likelihood_ratio(self, taus):
        """
        Compute the likelihood ratio test statistic (\( \lambda_{LR} \)) with respect to the null hypothesis.

        The likelihood ratio test statistic is defined as:

        .. math::

            \lambda_{LR} = -2 \\left[ \\ell(\\tau_{\\text{null}}) - \\ell(\\hat{\\tau}) \\right]

        where:

        - \( \\ell() \): Log-likelihood function.
        - \( \\tau_{\\text{null}} \): Null hypothesis of no interaction (genes are independent given their individual driver probabilities).
        - \( \\hat{\\tau} \): Estimated values of the \( \\tau \) parameters under the alternative hypothesis.

        **Returns**:
        :return: (float) The computed likelihood ratio test statistic (\( \lambda_{LR} \)).
        """

        logging.info(f"Computing likelihood ratio for interaction {self.name}.")

        self.verify_pi_values(self.gene_a.pi, self.gene_b.pi)
        pi_a, pi_b = self.gene_a.pi, self.gene_b.pi
        tau_00, tau_01, tau_10, tau_11 = taus

        # TODO: Validate the null hypothesis is correct
        # ? Why shouldn't we use the marginals instead?
        tau_null = (
            (1 - pi_a) * (1 - pi_b),  # tau_00: neither gene has a driver mutation
            (1 - pi_a) * pi_b,  # tau_01: only gene_b has a driver mutation
            pi_a * (1 - pi_b),  # tau_10: only gene_a has a driver mutation
            pi_a * pi_b,  # tau_11: both genes have driver mutations
        )
        lambda_LR = -2 * (
            self.compute_log_likelihood(tau_null)
            - self.compute_log_likelihood((tau_00, tau_01, tau_10, tau_11))
        )
        return lambda_LR

    def compute_log_odds_ratio(self, taus):
        """
        Compute the log odds ratio for the interaction based on the \( \tau \) parameters.

        The log odds ratio is calculated as:

        .. math::

            \text{Log Odds Ratio} = \\log \\left( \\frac{\\tau_{01} \\cdot \\tau_{10}}{\\tau_{00} \\cdot \\tau_{11}} \\right)

        **Returns**:
        :return: (float) The computed log odds ratio.

        **Raises**:
        :raises ValueError: If \( \\tau \) parameters are invalid or lead to division by zero.
        """

        logging.info(f"Computing log odds ratio for interaction {self.name}.")

        self.verify_taus_are_valid(taus)
        tau_00, tau_01, tau_10, tau_11 = taus

        if tau_01 * tau_10 == 0 or tau_00 * tau_11 == 0:
            logging.warning(
                f"Zero encountered in odds ratio computation for interaction {self.name}. "
                f"tau_01={tau_01}, tau_10={tau_10}, tau_00={tau_00}, tau_11={tau_11}"
            )
            return None  # Return None when numerator or denominator is zero

        log_odds_ratio = np.log((tau_01 * tau_10) / (tau_00 * tau_11))
        logging.info(
            f"Computed log odds ratio for interaction {self.name}: {log_odds_ratio}"
        )
        return log_odds_ratio

    def compute_wald_statistic(self, taus):
        """
        Compute the Wald statistic for the interaction.

        The Wald statistic is calculated as:

        .. math::

            W = \\frac{\\text{Log Odds Ratio}}{\\text{Standard Error}}

        where the standard error is defined as:

        .. math::

            \\text{Standard Error} = \\sqrt{
                \\frac{1}{\\tau_{01}} + \\frac{1}{\\tau_{10}} + \\frac{1}{\\tau_{00}} + \\frac{1}{\\tau_{11}}
            }

        **Returns**:
        :return: (float or None) The computed Wald statistic, or None if calculation is not possible.
        """

        logging.info(f"Computing Wald statistic for interaction {self.name}.")

        self.verify_taus_are_valid(taus)
        log_odds_ratio = self.compute_log_odds_ratio(taus)
        if log_odds_ratio is None:
            logging.warning(f"Log odds ratio is None for interaction {self.name}.")
            return None

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

    def compute_rho(self, taus):
        """
        Compute the interaction measure \( \rho \) for the given \( \tau \) parameters.

        The interaction measure \( \rho \) is calculated as:

        .. math::

            \\rho = \\frac{\\tau_{01} \\cdot \\tau_{10} - \\tau_{11} \\cdot \\tau_{00}}
            {\\sqrt{\\tau_{0*} \\cdot \\tau_{1*} \\cdot \\tau_{*0} \\cdot \\tau_{*1}}}

        where:

        - \( \\tau_{0*} = \\tau_{00} + \\tau_{01} \): Marginal probability for no driver mutation in gene \( A \).
        - \( \\tau_{1*} = \\tau_{10} + \\tau_{11} \): Marginal probability for a driver mutation in gene \( A \).
        - \( \\tau_{*0} = \\tau_{00} + \\tau_{10} \): Marginal probability for no driver mutation in gene \( B \).
        - \( \\tau_{*1} = \\tau_{01} + \\tau_{11} \): Marginal probability for a driver mutation in gene \( B \).

        **Returns**:
        :return: (float or None) The computed value of \( \\rho \), or None if the computation is invalid (e.g., division by zero).
        """

        logging.info(f"Computing rho for interaction {self.name}.")

        self.verify_taus_are_valid(taus)

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
        # TODO: tau parameters fail verification due to optimization scheme. try different optimization scheme
        """
        Estimate the tau parameters using the SLSQP optimization scheme.

        :param tau_init (list): Initial guesses for the tau parameters (default: [0.25, 0.25, 0.25, 0.25]).
        :param alpha (float): Small value to avoid edge cases at 0 or 1 (default: 1e-13).
        :return (tuple): The optimized values of (tau_00, tau_01, tau_10, tau_11).
        """
        logging.info(f"Estimating tau params for {self.name} using SLSQP.")

        self.verify_bmr_pmf_and_counts_exist()

        def negative_log_likelihood(tau):
            return -self.compute_log_likelihood(tau)

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
            logging.warning(
                f"Optimization failed for interaction {self.name}: {result.message}"
            )
            raise ValueError(f"Optimization failed: {result.message}")

        self.tau_00, self.tau_01, self.tau_10, self.tau_11 = result.x
        logging.info(
            f"Estimated tau parameters for interaction {self.name}: tau_00={self.tau_00}, tau_01={self.tau_01}, tau_10={self.tau_10}, tau_11={self.tau_11}"
        )

    def estimate_tau_with_em_from_scratch(
        self, max_iter=1000, tol=1e-6, tau_init=[0.25, 0.25, 0.25, 0.25]
    ):
        """
        Estimate the tau parameters for interaction using the Expectation-Maximization (EM) algorithm.

        This method iteratively updates the tau parameters, \( \tau = (\tau_{00}, \tau_{01}, \tau_{10}, \tau_{11}) \),
        to maximize the likelihood of the observed mutation count data for two interacting genes.

        **Algorithm Steps**:

        1. **E-Step**:
           At iteration \( t \), given the estimated driver mutation probabilities
           \( \tau^{(t)} = (\tau_{00}^{(t)}, \tau_{01}^{(t)}, \tau_{10}^{(t)}, \tau_{11}^{(t)}) \),
           compute the responsibilities \( z_{i,uv}^{(t)} \) for each pair \( (u,v) \in \{0,1\}^2 \)
           and sample \( i = 1, \dots, N \) as:

           .. math::

               z_{i,uv}^{(t)} = \\frac{\\tau_{uv}^{(t)} \\cdot \\mathbb{P}(P_i = c_i - u) \\cdot \\mathbb{P}(P_i' = c_i' - v)}
               {\\sum_{(x,y) \\in \\{0,1\\}^2} \\left( \\tau_{xy}^{(t)} \\cdot \\mathbb{P}(P_i = c_i - x) \\cdot \\mathbb{P}(P_i' = c_i' - y) \\right)}

           where \( P_i \) and \( P_i' \) represent passenger mutation probabilities for the two genes,
           and \( c_i, c_i' \) are the observed mutation counts.

        2. **M-Step**:
           Given the responsibilities \( \\bm{z}_i^{(t)} = (z_{i,00}^{(t)}, z_{i,01}^{(t)}, z_{i,10}^{(t)}, z_{i,11}^{(t)}) \),
           update the tau parameters at iteration \( t+1 \) as:

           .. math::

               \\tau_{uv}^{(t+1)} = \\frac{1}{N} \\sum_{i=1}^{N} z_{i,uv}^{(t)}

           for each pair \( (u,v) \\in \\{0,1\\}^2 \).

        **Parameters**:
        :param max_iter: (int) Maximum number of iterations for the EM algorithm (default: 1000).
        :param tol: (float) Convergence threshold for log-likelihood improvement (default: 1e-6).
        :param tau_init: (list of float) Initial guesses for the tau parameters (default: [0.25, 0.25, 0.25, 0.25]).

        **Returns**:
        :return: (tuple) The estimated values of \( (\\tau_{00}, \\tau_{01}, \\tau_{10}, \\tau_{11}) \).
        """
        logging.info("Estimating tau parameters using EM algorithm from scratch.")

        self.verify_bmr_pmf_and_counts_exist()

        # TODO: handle nonzero probability counts and cases where counts are not in BMR PMF

        tau_00, tau_01, tau_10, tau_11 = tau_init
        for it in range(max_iter):
            # E-Step: Compute responsibilities
            total_probabilities = self.compute_total_probability(
                tau_00, tau_01, tau_10, tau_11
            )  # denominator in E-Step equation
            z_i_00 = self.compute_joint_probability(tau_00, 0, 0) / total_probabilities
            z_i_01 = self.compute_joint_probability(tau_01, 0, 1) / total_probabilities
            z_i_10 = self.compute_joint_probability(tau_10, 1, 0) / total_probabilities
            z_i_11 = self.compute_joint_probability(tau_11, 1, 1) / total_probabilities

            # M-Step: Update tau parameters
            curr_tau_00 = np.mean(z_i_00)
            curr_tau_01 = np.mean(z_i_01)
            curr_tau_10 = np.mean(z_i_10)
            curr_tau_11 = np.mean(z_i_11)

            # Check for convergence
            prev_log_likelihood = self.compute_log_likelihood(
                (tau_00, tau_01, tau_10, tau_11)
            )
            curr_log_likelihood = self.compute_log_likelihood(
                (curr_tau_00, curr_tau_01, curr_tau_10, curr_tau_11)
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
        logging.info(
            f"Estimated tau parameters for interaction {self.name}: tau_00={self.tau_00}, tau_01={self.tau_01}, tau_10={self.tau_10}, tau_11={self.tau_11}"
        )

    # TODO (LOW PRIORITY): Implement EM w/ Pomegranate for Speed Improvement
    def estimate_tau_with_em_using_pomegranate(self):
        logging.info("Estimating tau parameters using pomegranate.")
        raise NotImplementedError("Method is not yet implemented.")

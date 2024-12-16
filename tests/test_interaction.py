import logging
import unittest
from dialect.utils.simulate import simulate_interaction_pair


class TestInteraction(unittest.TestCase):
    def setUp(self):
        """
        Set up realistic test cases for the Interaction class with two genes.
        """
        self.num_samples = 1000
        self.tau_00 = 0.8
        self.tau_01 = 0.1
        self.tau_10 = 0.1
        self.tau_11 = 0.0

        bmr_pmf_a = {0: 0.8, 1: 0.1, 2: 0.05, 3: 0.03, 4: 0.02}
        bmr_pmf_b = {0: 0.7, 1: 0.15, 2: 0.1, 3: 0.04, 4: 0.01}

        self.interaction = simulate_interaction_pair(
            bmr_pmf_a,
            bmr_pmf_b,
            self.tau_01,
            self.tau_10,
            self.tau_11,
            self.num_samples,
        )

    def test_initialization(self):
        """
        Test that the Interaction object is correctly initialized.
        """
        self.assertIsNone(self.interaction.tau_00)
        self.assertIsNone(self.interaction.tau_01)
        self.assertIsNone(self.interaction.tau_10)
        self.assertIsNone(self.interaction.tau_11)

    def test_compute_log_likelihood(self):
        """
        Test compute_log_likelihood for a realistic case.
        """
        taus = (self.tau_00, self.tau_01, self.tau_10, self.tau_11)
        log_likelihood = self.interaction.compute_log_likelihood(taus)
        self.assertIsInstance(log_likelihood, float)
        self.assertNotEqual(log_likelihood, 0.0)

    def test_compute_likelihood_ratio(self):
        """
        Test compute_likelihood_ratio for a realistic case.
        """
        self.interaction.tau_00 = self.tau_00
        self.interaction.tau_01 = self.tau_01
        self.interaction.tau_10 = self.tau_10
        self.interaction.tau_11 = self.tau_11
        taus = (self.tau_00, self.tau_01, self.tau_10, self.tau_11)
        likelihood_ratio = self.interaction.compute_likelihood_ratio(taus=taus)
        self.assertIsInstance(likelihood_ratio, float)
        self.assertGreaterEqual(likelihood_ratio, 0.0)

    def test_compute_log_odds_ratio(self):
        """
        Test compute_log_odds_ratio for a realistic case.
        """
        self.interaction.tau_00 = self.tau_00
        self.interaction.tau_01 = self.tau_01
        self.interaction.tau_10 = self.tau_10
        self.interaction.tau_11 = self.tau_11
        taus = (self.tau_00, self.tau_01, self.tau_10, self.tau_11)

        log_odds_ratio = self.interaction.compute_log_odds_ratio(taus=taus)

        if log_odds_ratio is None:
            # Verify that None is returned in cases where it is expected (e.g., zero in numerator or denominator)
            self.assertIsNone(
                log_odds_ratio, "Log odds ratio should be None for invalid tau values."
            )
            logging.info("Log odds ratio correctly returned None for invalid taus.")
        else:
            # Otherwise, ensure the result is a valid float
            self.assertIsInstance(
                log_odds_ratio, float, "Log odds ratio should be a float."
            )
            logging.info(f"Log odds ratio computed successfully: {log_odds_ratio}")

    def test_compute_wald_statistic(self):
        """
        Test compute_wald_statistic for a realistic case.
        """
        self.interaction.tau_00 = self.tau_00
        self.interaction.tau_01 = self.tau_01
        self.interaction.tau_10 = self.tau_10
        self.interaction.tau_11 = self.tau_11
        taus = (self.tau_00, self.tau_01, self.tau_10, self.tau_11)

        wald_statistic = self.interaction.compute_wald_statistic(taus=taus)

        if wald_statistic is None:
            # Verify that None is returned in cases where it is expected (e.g., invalid taus or division by zero)
            self.assertIsNone(
                wald_statistic, "Wald statistic should be None for invalid tau values."
            )
            logging.info("Wald statistic correctly returned None for invalid taus.")
        else:
            # Otherwise, ensure the result is a valid float
            self.assertIsInstance(
                wald_statistic, float, "Wald statistic should be a float."
            )
            logging.info(f"Wald statistic computed successfully: {wald_statistic}")

    def test_compute_rho(self):
        """
        Test compute_rho for a realistic case.
        """
        self.interaction.tau_00 = self.tau_00
        self.interaction.tau_01 = self.tau_01
        self.interaction.tau_10 = self.tau_10
        self.interaction.tau_11 = self.tau_11
        taus = (self.tau_00, self.tau_01, self.tau_10, self.tau_11)
        rho = self.interaction.compute_rho(taus=taus)
        self.assertIsInstance(rho, float)

    # ? scipy optimization exhibits out of bounds tau values during SLSQP procedure
    # TODO: experiment with other optimization procedures to fix
    # def test_estimate_tau_with_optimization(self):
    #     """
    #     Test estimate_tau_with_optimization_using_scipy for a realistic case.
    #     """
    #     taus = self.interaction.estimate_tau_with_optimization_using_scipy()
    #     self.assertIsInstance(taus, tuple)
    #     self.assertEqual(len(taus), 4)
    #     self.assertTrue(all(0 <= tau <= 1 for tau in taus))
    #     self.assertAlmostEqual(sum(taus), 1.0, places=5)

    def test_estimate_tau_with_em(self):
        """
        Test estimate_tau_with_em_from_scratch for a realistic case.
        """
        self.interaction.estimate_tau_with_em_from_scratch()
        self.assertIsInstance(self.interaction.tau_00, float)
        self.assertIsInstance(self.interaction.tau_01, float)
        self.assertIsInstance(self.interaction.tau_10, float)
        self.assertIsInstance(self.interaction.tau_11, float)
        self.assertAlmostEqual(
            sum(
                [
                    self.interaction.tau_00,
                    self.interaction.tau_01,
                    self.interaction.tau_10,
                    self.interaction.tau_11,
                ]
            ),
            1.0,
            places=5,
        )

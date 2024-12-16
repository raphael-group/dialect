import unittest
from dialect.utils.simulate import simulate_single_gene


class TestGene(unittest.TestCase):
    def setUp(self):
        """
        Set up a realistic test case for the Gene class with both passenger and driver mutations.
        """
        self.pi = 0.2
        self.num_samples = 1000
        self.bmr_pmf = {0: 0.8, 1: 0.1, 2: 0.05, 3: 0.03, 4: 0.02}
        self.gene = simulate_single_gene(self.bmr_pmf, self.num_samples, self.pi)

    def test_setup_is_realistic(self):
        """
        Test that the setup is realistic and correctly initialized.
        """
        self.assertEqual(len(self.gene.counts), self.num_samples)
        self.assertAlmostEqual(sum(self.bmr_pmf.values()), 1.0, places=5)

    def test_compute_log_likelihood(self):
        """
        Test compute_log_likelihood for a realistic case.
        """
        log_likelihood = self.gene.compute_log_likelihood(self.pi)
        self.assertIsInstance(log_likelihood, float)
        self.assertNotEqual(log_likelihood, 0.0)

    def test_compute_likelihood_ratio(self):
        """
        Test compute_likelihood_ratio for a realistic case.
        """
        self.gene.pi = self.pi
        likelihood_ratio = self.gene.compute_likelihood_ratio(self.gene.pi)
        self.assertIsInstance(likelihood_ratio, float)
        self.assertGreaterEqual(likelihood_ratio, 0.0)

    def test_compute_log_odds_ratio(self):
        """
        Test compute_log_odds_ratio for a realistic case.
        """
        self.gene.pi = self.pi
        log_odds_ratio = self.gene.compute_log_odds_ratio(self.gene.pi)
        self.assertIsInstance(log_odds_ratio, float)

    def test_estimate_pi_with_optimization(self):
        """
        Test estimate_pi_with_optimiziation_using_scipy for a realistic case.
        """
        self.gene.estimate_pi_with_optimiziation_using_scipy()
        self.assertIsInstance(self.gene.pi, float)
        self.assertGreaterEqual(self.gene.pi, 0.0)
        self.assertLessEqual(self.gene.pi, 1.0)
        self.assertAlmostEqual(self.gene.pi, self.pi, places=1)

    def test_estimate_pi_with_em(self):
        """
        Test estimate_pi_with_em_from_scratch for a realistic case.
        """
        self.gene.estimate_pi_with_em_from_scratch()
        self.assertIsInstance(self.gene.pi, float)
        self.assertGreaterEqual(self.gene.pi, 0.0)
        self.assertLessEqual(self.gene.pi, 1.0)
        self.assertAlmostEqual(self.gene.pi, self.pi, places=1)

    def test_non_normalized_bmr_pmf(self):
        """
        Test behavior when BMR PMF does not sum to 1.
        """
        non_normalized_bmr_pmf = {0: 0.8, 1: 0.1, 2: 0.05, 3: 0.03, 4: 0.03}
        with self.assertRaises(ValueError):
            simulate_single_gene(non_normalized_bmr_pmf, self.num_samples, self.pi)

    def test_invalid_pi_value(self):
        """
        Test behavior when pi is outside the allowed range.
        """
        with self.assertRaises(ValueError):
            simulate_single_gene(self.bmr_pmf, self.num_samples, -0.1)
        with self.assertRaises(ValueError):
            simulate_single_gene(self.bmr_pmf, self.num_samples, 1.1)

    # TODO Add additional edge cases and tests, including:
    # - TODO: tests for missing values in bmr_pmf

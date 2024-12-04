import unittest
import numpy as np
from scipy.stats import rv_discrete
from dialect.models.gene import Gene


class TestGene(unittest.TestCase):
    def setUp(self):
        """
        Set up a realistic test case for the Gene class with both passenger and driver mutations.
        """
        self.pi = 0.2
        self.num_samples = 500

        mutation_counts = np.arange(0, 6)
        probabilities = np.array([0.8, 0.1, 0.05, 0.03, 0.02, 0])
        self.bmr_pmf = {k: v for k, v in zip(mutation_counts, probabilities)}

        # simulate passengers
        bmr_dist = rv_discrete(values=(mutation_counts, probabilities))
        passenger_mutations = bmr_dist.rvs(size=self.num_samples)

        # simulate drivers
        driver_mutations = np.random.binomial(1, self.pi, size=self.num_samples)

        # initialize gene object
        self.counts = passenger_mutations + driver_mutations
        self.name = "TestGene"
        self.gene = Gene(self.name, self.counts, self.bmr_pmf)

    def test_setup_is_realistic(self):
        """
        Test that the setup is realistic and correctly initialized.
        """
        self.assertEqual(self.gene.name, self.name)
        self.assertEqual(len(self.gene.counts), self.num_samples)
        np.testing.assert_almost_equal(
            sum(self.bmr_pmf.values()),
            1.0,
            decimal=5,
            err_msg="BMR PMF probabilities do not sum to 1",
        )

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
        likelihood_ratio = self.gene.compute_likelihood_ratio()
        self.assertIsInstance(likelihood_ratio, float)
        self.assertGreaterEqual(likelihood_ratio, 0.0)

    def test_compute_log_odds_ratio(self):
        """
        Test compute_log_odds_ratio for a realistic case.
        """
        self.gene.pi = self.pi
        log_odds_ratio = self.gene.compute_log_odds_ratio()
        self.assertIsInstance(log_odds_ratio, float)

    def test_estimate_pi_with_optimization(self):
        """
        Test estimate_pi_with_optimiziation_using_scipy for a realistic case.
        """
        self.gene.estimate_pi_with_optimiziation_using_scipy()
        self.assertIsInstance(self.gene.pi, float)
        self.assertGreaterEqual(self.gene.pi, 0.0)
        self.assertLessEqual(self.gene.pi, 1.0)

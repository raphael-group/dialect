"""Unit tests for the DIALECT statistical model (Gene likelihood + BMR math)."""

import math

import pytest

from dialect.models.gene import Gene


def make_gene(bmr_pmf: dict, counts: list, name: str = "GENE_M") -> Gene:
    """Build a Gene with an integer-count-keyed background PMF."""
    return Gene(
        name=name,
        samples=list(range(len(counts))),
        counts=counts,
        bmr_pmf=bmr_pmf,
    )


def test_calculate_expected_mutations_matches_analytic_value() -> None:
    """E[B] = sum_k k * P(B=k)."""
    gene = make_gene({0: 0.9, 1: 0.08, 2: 0.02}, counts=[0, 1, 2])
    assert gene.calculate_expected_mutations() == pytest.approx(0.12)


def test_calculate_expected_mutations_uses_keys_not_positions() -> None:
    """Regression: must weight by the count *key*, not the enumerate() index.

    Old code did ``sum(k * prob for k, prob in enumerate(self.bmr_pmf))`` which,
    on a dict, iterates keys positionally and ignores the probabilities entirely.
    For {0: 0.5, 5: 0.5} the correct E[B] is 2.5; the buggy version returned 5.
    """
    gene = make_gene({0: 0.5, 5: 0.5}, counts=[0, 5])
    assert gene.calculate_expected_mutations() == pytest.approx(2.5)


def test_compute_log_likelihood_known_value() -> None:
    """Hand-computed log-likelihood for a tiny PMF and pi=0.5."""
    gene = make_gene({0: 0.8, 1: 0.2}, counts=[0, 1])
    # c=0: P(B=0)(1-pi) + P(B=-1)pi = 0.8*0.5 + 0*0.5 = 0.4
    # c=1: P(B=1)(1-pi) + P(B=0)pi  = 0.2*0.5 + 0.8*0.5 = 0.5
    expected = math.log(0.4) + math.log(0.5)
    assert gene.compute_log_likelihood(0.5) == pytest.approx(expected)


def test_compute_log_likelihood_missing_support_is_finite() -> None:
    """Regression: an in-range gap must floor to min_val, not return None.

    Previously safe_get_no_default returned ``pmf.get(c)`` (None) for a count
    inside [0, max] but absent from the PMF, crashing np.log(None * float).
    """
    gene = make_gene({0: 0.5, 2: 0.5}, counts=[1])  # count 1 absent from PMF
    assert math.isfinite(gene.compute_log_likelihood(0.5))

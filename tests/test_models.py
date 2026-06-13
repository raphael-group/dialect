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


def test_bmr_pmfs_broadcasts_a_shared_dict() -> None:
    """A single shared PMF is broadcast to one PMF per sample."""
    pmf = {0: 0.8, 1: 0.2}
    gene = make_gene(pmf, counts=[0, 1, 2])
    assert gene.bmr_pmfs == [pmf, pmf, pmf]
    assert all(p is pmf for p in gene.bmr_pmfs)


def test_bmr_pmfs_passes_through_a_per_sample_list() -> None:
    """A per-sample list of PMFs is used as-is (one per sample)."""
    pmfs = [{0: 0.9, 1: 0.1}, {0: 0.5, 1: 0.5}]
    gene = Gene(name="G_M", samples=[0, 1], counts=[0, 1], bmr_pmf=pmfs)
    assert gene.bmr_pmfs is pmfs


def test_per_sample_backgrounds_are_used_in_log_likelihood() -> None:
    """The likelihood evaluates each sample's count against *its own* PMF."""
    pmf0 = {0: 0.9, 1: 0.1}
    pmf1 = {0: 0.5, 1: 0.5}
    gene = Gene(name="G_M", samples=[0, 1], counts=[0, 0], bmr_pmf=[pmf0, pmf1])
    # c=0 under pmf0 -> 0.9*(1-pi); c=0 under pmf1 -> 0.5*(1-pi); pi=0.3
    expected = math.log(0.9 * 0.7) + math.log(0.5 * 0.7)
    assert gene.compute_log_likelihood(0.3) == pytest.approx(expected)
    # Broadcasting pmf0 to both samples gives a different value, proving per-sample use.
    shared = make_gene(pmf0, counts=[0, 0])
    assert gene.compute_log_likelihood(0.3) != pytest.approx(
        shared.compute_log_likelihood(0.3),
    )

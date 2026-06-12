"""Tests for dialect.stats (epsilon threshold + ranking/threshold constants)."""

from math import sqrt

from scipy.stats import norm

from dialect.stats.constants import (
    CO_METHOD_RANKING_CRITERIA,
    ME_METHOD_RANKING_CRITERIA,
    ME_METHOD_SIGNIFICANCE_THRESHOLDS,
)
from dialect.stats.thresholds import compute_epsilon_threshold


def test_epsilon_threshold_satisfies_defining_equation():
    # eps solves eps - z*sqrt(eps(1-eps)/n) = 0 (lower bound of the one-sided CI = 0).
    for n in (50, 200, 1000):
        eps = compute_epsilon_threshold(n)
        assert 0 < eps < 1
        z = norm.ppf(1 - 0.001)
        assert abs(eps - z * sqrt(eps * (1 - eps) / n)) < 1e-6


def test_epsilon_threshold_decreases_with_sample_size():
    # More samples -> a smaller marginal driver rate is already significant.
    eps = [compute_epsilon_threshold(n) for n in (50, 200, 1000, 5000)]
    assert eps == sorted(eps, reverse=True)


def test_ranking_constants_are_wellformed():
    for crit in (ME_METHOD_RANKING_CRITERIA, CO_METHOD_RANKING_CRITERIA):
        for metric, order in crit.values():
            assert isinstance(metric, str)
            assert order in {"ascending", "descending"}
    for col, cutoff in ME_METHOD_SIGNIFICANCE_THRESHOLDS.values():
        assert isinstance(col, str)
        assert 0 < cutoff < 1

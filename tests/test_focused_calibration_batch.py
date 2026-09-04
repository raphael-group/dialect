"""Seeded equivalence tests for the focused calibration batch kernel."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from analysis import calibration_batch
from analysis.calibration_batch import fit_gene_pairs_batched
from dialect.models.gene import Gene
from dialect.models.interaction import (
    PAIR_EFFECT_IDENTIFIED_STATUS,
    Interaction,
)


def _poisson_pmf(rate: float, maximum: int = 5) -> dict[int, float]:
    support = np.arange(maximum + 1)
    probabilities = np.asarray(
        [
            math.exp(-rate) * (rate**int(count)) / math.factorial(int(count))
            for count in support
        ],
    )
    probabilities /= probabilities.sum()
    return {
        int(count): float(probability)
        for count, probability in zip(support, probabilities, strict=True)
    }


def _seeded_sample_specific_pairs(
    *,
    seed: int,
    pair_count: int,
    sample_count: int,
) -> list[tuple[Gene, Gene]]:
    rng = np.random.default_rng(seed)
    samples = tuple(f"sample-{index}" for index in range(sample_count))
    result = []
    for pair_index in range(pair_count):
        genes = []
        for gene_index in range(2):
            driver_probability = float(rng.uniform(0.01, 0.25))
            pmfs = []
            counts = []
            for _sample_index in range(sample_count):
                pmf = _poisson_pmf(float(rng.lognormal(-3.0, 0.8)))
                support = np.fromiter(pmf, dtype=np.int64)
                probabilities = np.fromiter(pmf.values(), dtype=np.float64)
                background = int(rng.choice(support, p=probabilities))
                driver = int(rng.random() < driver_probability)
                pmfs.append(pmf)
                counts.append(background + driver)
            genes.append(
                Gene(
                    f"pair-{pair_index}-gene-{gene_index}",
                    samples,
                    np.asarray(counts, dtype=np.int64),
                    pmfs,
                ),
            )
        result.append((genes[0], genes[1]))
    return result


def _scalar_fit(
    pairs: list[tuple[Gene, Gene]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lrt = []
    reportable = []
    pi = []
    for gene_a, gene_b in pairs:
        gene_a.estimate_pi_with_mle()
        gene_b.estimate_pi_with_mle()
        interaction = Interaction(gene_a, gene_b)
        interaction.estimate_tau_with_coordinate_ascent()
        lrt.append(float(interaction.likelihood_ratio))
        reportable.append(
            interaction.effect_identifiability_status()
            == PAIR_EFFECT_IDENTIFIED_STATUS,
        )
        pi.append((gene_a.pi, gene_b.pi))
    return (
        np.asarray(lrt),
        np.asarray(reportable),
        np.asarray(pi),
    )


def test_seeded_sample_specific_batch_matches_scalar_profile_lrt() -> None:
    pairs = _seeded_sample_specific_pairs(
        seed=20260904,
        pair_count=48,
        sample_count=300,
    )
    scalar_lrt, scalar_reportable, scalar_pi = _scalar_fit(copy.deepcopy(pairs))

    batched = fit_gene_pairs_batched(pairs)

    assert not batched.scalar_fallback.any()
    np.testing.assert_array_equal(batched.marginal_pi, scalar_pi)
    np.testing.assert_allclose(
        batched.likelihood_ratio,
        scalar_lrt,
        rtol=0,
        atol=1e-11,
    )
    np.testing.assert_array_equal(batched.reportable, scalar_reportable)
    for critical_value in (3.841458820694124, 6.6348966010212145):
        np.testing.assert_array_equal(
            batched.reportable & (batched.likelihood_ratio >= critical_value),
            scalar_reportable & (scalar_lrt >= critical_value),
        )


def test_boundary_and_rank_deficient_fits_match_scalar() -> None:
    rng = np.random.default_rng(731)
    samples = tuple(f"sample-{index}" for index in range(80))
    raw_counts = [
        (np.zeros(80, dtype=np.int64), np.zeros(80, dtype=np.int64)),
        (np.ones(80, dtype=np.int64), np.ones(80, dtype=np.int64)),
    ]
    identical = rng.binomial(1, 0.2, len(samples))
    raw_counts.extend(
        [
            (identical, identical.copy()),
            (identical, 1 - identical),
            (
                rng.binomial(1, 0.2, len(samples)),
                rng.binomial(1, 0.3, len(samples)),
            ),
        ],
    )
    pairs = [
        (
            Gene(f"a-{index}", samples, counts_a, {0: 1.0}),
            Gene(f"b-{index}", samples, counts_b, {0: 1.0}),
        )
        for index, (counts_a, counts_b) in enumerate(raw_counts)
    ]
    scalar_lrt, scalar_reportable, scalar_pi = _scalar_fit(copy.deepcopy(pairs))

    batched = fit_gene_pairs_batched(pairs)

    np.testing.assert_array_equal(batched.marginal_pi, scalar_pi)
    np.testing.assert_array_equal(batched.likelihood_ratio, scalar_lrt)
    np.testing.assert_array_equal(batched.reportable, scalar_reportable)
    assert scalar_reportable.tolist()[:4] == [False, False, False, False]


def test_batch_outputs_are_bit_identical_across_chunk_boundaries() -> None:
    pairs = _seeded_sample_specific_pairs(
        seed=808,
        pair_count=24,
        sample_count=120,
    )
    whole = fit_gene_pairs_batched(pairs)

    for chunk_size in (1, 3, 7, 16):
        chunks = [
            fit_gene_pairs_batched(pairs[start : start + chunk_size])
            for start in range(0, len(pairs), chunk_size)
        ]
        np.testing.assert_array_equal(
            np.concatenate([chunk.likelihood_ratio for chunk in chunks]),
            whole.likelihood_ratio,
        )
        np.testing.assert_array_equal(
            np.concatenate([chunk.reportable for chunk in chunks]),
            whole.reportable,
        )
        np.testing.assert_array_equal(
            np.concatenate([chunk.marginal_pi for chunk in chunks]),
            whole.marginal_pi,
        )
        np.testing.assert_array_equal(
            np.concatenate([chunk.scalar_fallback for chunk in chunks]),
            whole.scalar_fallback,
        )


@pytest.mark.parametrize("ambiguous_rank", [False, True])
def test_decision_boundary_rows_are_reconciled_by_scalar(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ambiguous_rank: bool,
) -> None:
    samples = ("sample-0", "sample-1")
    pair = (
        Gene("a", samples, np.asarray([0, 1]), {0: 0.8, 1: 0.2}),
        Gene("b", samples, np.asarray([1, 0]), {0: 0.8, 1: 0.2}),
    )
    monkeypatch.setattr(
        calibration_batch,
        "_fit_fast_components",
        lambda _components: (
            np.asarray([3.841458820694124]),
            np.asarray([True]),
            np.asarray([[0.2, 0.3]]),
            np.asarray([ambiguous_rank]),
        ),
    )
    monkeypatch.setattr(
        calibration_batch,
        "_fit_scalar_pair",
        lambda _pair: (9.0, False, np.asarray([0.4, 0.5])),
    )

    observed = fit_gene_pairs_batched([pair])

    assert observed.scalar_fallback.tolist() == [True]
    assert observed.likelihood_ratio.tolist() == [9.0]
    assert observed.reportable.tolist() == [False]
    np.testing.assert_array_equal(observed.marginal_pi, [[0.4, 0.5]])


def test_log_domain_underflow_uses_unchanged_scalar_fallback() -> None:
    samples = tuple(f"sample-{index}" for index in range(8))
    tiny = np.nextafter(0.0, 1.0)
    pmf = {0: 1.0, 1: tiny}
    pairs = [
        (
            Gene("a", samples, np.ones(len(samples), dtype=np.int64), pmf),
            Gene("b", samples, np.ones(len(samples), dtype=np.int64), pmf),
        ),
    ]
    scalar_lrt, scalar_reportable, scalar_pi = _scalar_fit(copy.deepcopy(pairs))

    batched = fit_gene_pairs_batched(pairs)

    assert batched.scalar_fallback.tolist() == [True]
    np.testing.assert_array_equal(batched.marginal_pi, scalar_pi)
    np.testing.assert_array_equal(batched.likelihood_ratio, scalar_lrt)
    np.testing.assert_array_equal(batched.reportable, scalar_reportable)

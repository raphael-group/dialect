"""Unit tests for the DIALECT statistical model (Gene likelihood + BMR math)."""

import math

import numpy as np
import pytest

from dialect.models.gene import (
    MARGINAL_FIT_BRACKET_WIDTH_TOL,
    MARGINAL_FIT_CONTRACT,
    MARGINAL_FIT_FIXED_POINT_TOL,
    MARGINAL_FIT_FLAT_TIE_BREAK,
    MARGINAL_FIT_KKT_TOL,
    MARGINAL_FIT_MAX_ITER,
    Gene,
)
from dialect.models.interaction import (
    CONTINGENCY_TABLE_CONTRACT,
    LOG_ODDS_RATIO_CONTRACT,
    LRT_CONTRACT,
    LRT_NESTEDNESS_TOL,
    PAIR_EFFECT_IDENTIFIABILITY_CONTRACT,
    PAIR_FIT_CONTRACT,
    PAIR_FIT_KKT_TOL,
    PAIR_FIT_MAX_ITER,
    PAIR_IDENTIFIABILITY_RTOL,
    PAIR_SIMPLEX_TOL,
    RHO_CONTRACT,
    UNDEFINED_RHO_LRT_TOL,
    Interaction,
    compute_marshall_olkin_rho,
)
from dialect.utils.identify import estimate_taus_for_each_interaction


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


def test_calculate_expected_total_mutations_sums_heterogeneous_sample_pmfs() -> None:
    """The cohort total is the direct sum of each sample's passenger expectation."""
    pmfs = [
        {0: 0.5, 1: 0.5},  # E[B_0] = 0.5
        {0: 0.25, 1: 0.25, 2: 0.5},  # E[B_1] = 1.25
    ]
    gene = Gene(name="G_M", samples=["s0", "s1"], counts=[1, 2], bmr_pmf=pmfs)

    assert gene.calculate_expected_mutations() == pytest.approx(0.875)
    assert gene.calculate_expected_total_mutations() == pytest.approx(1.75)


def test_calculate_expected_total_mutations_broadcasts_shared_pmf() -> None:
    """A cohort-shared PMF contributes its expectation once for every sample."""
    gene = make_gene({0: 0.8, 1: 0.2}, counts=[0, 0, 1])

    assert gene.calculate_expected_mutations() == pytest.approx(0.2)
    assert gene.calculate_expected_total_mutations() == pytest.approx(0.6)


def test_calculate_expected_total_mutations_rejects_misaligned_sample_pmfs() -> None:
    """A per-sample background list cannot silently omit a cohort sample."""
    gene = Gene(
        name="G_M",
        samples=["s0", "s1"],
        counts=[0, 1],
        bmr_pmf=[{0: 1.0}],
    )

    with pytest.raises(ValueError, match="misaligned samples, counts"):
        gene.calculate_expected_total_mutations()


def test_calculate_expected_total_mutations_rejects_nonfinite_expectation() -> None:
    """Non-finite PMF arithmetic cannot enter a single-gene result CSV."""
    gene = Gene(
        name="G_M",
        samples=["s0"],
        counts=[1],
        bmr_pmf=[{0: 0.5, 1: np.inf}],
    )

    with pytest.raises(ValueError, match="non-finite passenger expectation"):
        gene.calculate_expected_total_mutations()


def test_compute_log_likelihood_known_value() -> None:
    """Hand-computed log-likelihood for a tiny PMF and pi=0.5."""
    gene = make_gene({0: 0.8, 1: 0.2}, counts=[0, 1])
    # c=0: P(B=0)(1-pi) + P(B=-1)pi = 0.8*0.5 + 0*0.5 = 0.4
    # c=1: P(B=1)(1-pi) + P(B=0)pi  = 0.2*0.5 + 0.8*0.5 = 0.5
    expected = math.log(0.4) + math.log(0.5)
    assert gene.compute_log_likelihood(0.5) == pytest.approx(expected)


def test_compute_log_likelihood_missing_support_is_finite() -> None:
    """A missing passenger-only key is valid when the driver state has support."""
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


def test_marginal_mle_keeps_observation_supported_only_by_driver_state() -> None:
    """P(B=c-1)>0 is valid support even when the passenger-only state is zero."""
    gene = make_gene({0: 0.9, 1: 0.1, 2: 0.0}, counts=[2])
    assert gene.component_probabilities().tolist() == [[0.0, 0.1]]
    gene.estimate_pi_with_mle()
    assert gene.pi == 1.0
    assert gene.compute_likelihood_ratio(gene.pi) == np.inf
    assert gene.likelihood_ratio_status == ("infinite-passenger-null-zero-probability")


@pytest.mark.parametrize(
    ("pmf", "counts", "expected_pi"),
    [
        ({0: 0.9, 1: 0.1}, [0, 0], 0.0),
        ({0: 0.9, 1: 0.1}, [1, 1], 1.0),
    ],
)
def test_marginal_mle_certifies_one_sided_boundary_scores(
    pmf,
    counts,
    expected_pi,
) -> None:
    """The exact endpoint score, not a scalar optimizer heuristic, picks a bound."""
    gene = make_gene(pmf, counts)

    gene.estimate_pi_with_mle()

    assert gene.pi == expected_pi
    assert gene.mle_algorithm == MARGINAL_FIT_CONTRACT
    assert gene.mle_iterations == 0
    assert gene.mle_bracket_width == 0.0
    assert gene.mle_fixed_point_residual == 0.0
    assert gene.mle_kkt_residual == 0.0


def test_marginal_mle_flat_likelihood_has_exact_pi_zero_tie_break() -> None:
    """Equal components make every pi optimal; the frozen tie is exactly zero."""
    gene = make_gene({0: 0.5, 1: 0.5}, [1, 1, 1])

    gene.estimate_pi_with_mle()

    assert gene.compute_marginal_score(0.0) == 0.0
    assert gene.compute_marginal_score(1.0) == 0.0
    assert gene.pi == 0.0
    assert MARGINAL_FIT_FLAT_TIE_BREAK == "pi-zero"


def test_marginal_mle_unique_interior_root_has_all_certificates() -> None:
    """The signed score bracket converges to the analytic pi=1/3 optimum."""
    gene = make_gene({0: 0.8, 1: 0.2}, [0, 1])

    gene.estimate_pi_with_mle()

    assert gene.pi == pytest.approx(1 / 3, abs=MARGINAL_FIT_BRACKET_WIDTH_TOL)
    assert 0 < gene.mle_iterations <= MARGINAL_FIT_MAX_ITER
    assert 0 <= gene.mle_bracket_width <= MARGINAL_FIT_BRACKET_WIDTH_TOL
    assert gene.mle_fixed_point_residual <= MARGINAL_FIT_FIXED_POINT_TOL
    assert gene.mle_kkt_residual <= MARGINAL_FIT_KKT_TOL
    assert gene.compute_mle_certificates(gene.pi) == pytest.approx(
        (gene.mle_fixed_point_residual, gene.mle_kkt_residual),
        abs=np.finfo(float).eps,
    )


def test_marginal_score_and_posteriors_preserve_true_zero_components() -> None:
    """Opposing driver-only/passenger-only rows retain their exact interior root."""
    gene = Gene(
        name="G_M",
        samples=["driver-only", "passenger-only"],
        counts=[1, 1],
        bmr_pmf=[{0: 1.0, 1: 0.0}, {0: 0.0, 1: 1.0}],
    )

    gene.estimate_pi_with_mle()

    assert gene.compute_marginal_score(0.0) == np.inf
    assert gene.compute_marginal_score(1.0) == -np.inf
    assert gene.pi == 0.5
    assert gene.mle_kkt_residual == pytest.approx(0.0)
    assert gene.mle_fixed_point_residual == pytest.approx(0.0)


def test_marginal_fit_preserves_extreme_log_scale_without_probability_floor() -> None:
    """Subnormal components are fitted in log space instead of floored or dropped."""
    tiny = 1e-320
    gene = Gene(
        name="G_M",
        samples=["positive", "negative"],
        counts=[1, 1],
        bmr_pmf=[
            {0: 4 * tiny, 1: tiny, 2: 1.0},
            {0: tiny, 1: 4 * tiny, 2: 1.0},
        ],
    )

    gene.estimate_pi_with_mle()

    assert gene.pi == 0.5
    assert np.isfinite(gene.mle_log_likelihood)
    assert gene.component_probabilities().min() > 0
    assert gene.mle_kkt_residual <= MARGINAL_FIT_KKT_TOL


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.5 - 1e-10, 0.5 + 1e-10), (1e-300, 1.0)],
    ids=["shallow", "steep"],
)
def test_marginal_fit_certifies_shallow_and_steep_concave_roots(low, high) -> None:
    """Both nearly flat and extremely curved symmetric likelihoods fit at 1/2."""
    gene = Gene(
        name="G_M",
        samples=["positive", "negative"],
        counts=[1, 1],
        bmr_pmf=[{0: high, 1: low}, {0: low, 1: high}],
    )

    gene.estimate_pi_with_mle()

    assert gene.pi == 0.5
    assert gene.mle_kkt_residual <= MARGINAL_FIT_KKT_TOL
    assert gene.mle_fixed_point_residual <= MARGINAL_FIT_FIXED_POINT_TOL


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": MARGINAL_FIT_MAX_ITER + 1},
        {"max_iter": True},
        {"kkt_tol": MARGINAL_FIT_KKT_TOL * 10},
        {"bracket_width_tol": MARGINAL_FIT_BRACKET_WIDTH_TOL * 10},
        {"fixed_point_tol": MARGINAL_FIT_FIXED_POINT_TOL * 10},
        {"kkt_tol": 0.0},
        {"bracket_width_tol": np.nan},
    ],
)
def test_marginal_fit_rejects_controls_that_weaken_or_invalidate_contract(
    kwargs,
) -> None:
    gene = make_gene({0: 0.8, 1: 0.2}, [0, 1])

    with pytest.raises(ValueError, match="Marginal-fit"):
        gene.estimate_pi_with_mle(**kwargs)


def test_marginal_fit_iteration_failure_publishes_no_partial_result() -> None:
    gene = make_gene({0: 0.8, 1: 0.2}, [0, 1])

    with pytest.raises(ValueError, match="within 1 score-bisection iterations"):
        gene.estimate_pi_with_mle(max_iter=1)

    assert gene.pi is None
    assert gene.mle_algorithm is None
    assert gene.mle_converged is False
    assert gene.mle_bracket_width is None
    assert gene.mle_fixed_point_residual is None
    assert gene.mle_kkt_residual is None


def test_marginal_fit_aborts_when_no_representable_point_can_be_certified(
    monkeypatch,
) -> None:
    """An adjacent-float signed bracket cannot be silently reported as converged."""
    gene = make_gene({0: 0.8, 1: 0.2}, [0, 1])

    def unrepresentable_root_score(pi):
        return 1.0 if pi <= 0.5 else -1.0

    monkeypatch.setattr(gene, "compute_marginal_score", unrepresentable_root_score)

    with pytest.raises(ValueError, match="no representable point"):
        gene.estimate_pi_with_mle()

    assert gene.mle_converged is False


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("mle_algorithm", "scipy-bounded", "algorithm"),
        ("mle_converged", False, "converged"),
        ("mle_iterations", 1.5, "iterations"),
        ("mle_bracket_width", -1.0, "bracket width"),
        ("mle_fixed_point_residual", np.nan, "fixed-point residual"),
        ("mle_kkt_residual", MARGINAL_FIT_KKT_TOL * 2, "KKT residual"),
        ("mle_log_likelihood", np.inf, "likelihood fields"),
    ],
)
def test_marginal_fit_validation_rejects_every_published_field_drift(
    attribute,
    value,
    message,
) -> None:
    gene = make_gene({0: 0.8, 1: 0.2}, [0, 1])
    gene.estimate_pi_with_mle()
    setattr(gene, attribute, value)

    with pytest.raises(ValueError, match=message):
        gene.validate_mle_fit()


def test_unsupported_observation_fails_without_a_probability_floor() -> None:
    """An observation outside both latent components fails closed."""
    gene = make_gene({0: 1.0}, counts=[2])
    with pytest.raises(ValueError, match="Unsupported observation"):
        gene.compute_log_likelihood(0.5)
    with pytest.raises(ValueError, match="Unsupported observation"):
        gene.estimate_pi_with_mle()


def make_fitted_interaction() -> Interaction:
    """Build a latent pair whose unrestricted and constrained marginals differ."""
    counts = [0, 0, 0, 0, 1, 1]
    pmf = {0: 0.8, 1: 0.2}
    gene_a = make_gene(pmf, counts, name="A_M")
    gene_b = make_gene(pmf, counts, name="B_M")
    gene_a.estimate_pi_with_mle()
    gene_b.estimate_pi_with_mle()
    interaction = Interaction(gene_a, gene_b)
    interaction.estimate_tau_with_coordinate_ascent()
    return interaction


def test_observed_contingency_cells_have_exact_asymmetric_semantics() -> None:
    """Regression: every observed cell label matches its binary state."""
    gene_a = make_gene(
        {0: 1.0},
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        name="A_M",
    )
    gene_b = make_gene(
        {0: 1.0},
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        name="B_M",
    )
    interaction = Interaction(gene_a, gene_b)

    assert interaction.compute_contingency_table().tolist() == [[1, 2], [3, 4]]
    assert "[[1 2]\n [3 4]]" in str(interaction)
    assert CONTINGENCY_TABLE_CONTRACT == "observed-binary-cells-00-01-10-11-v1"


def test_latent_log_odds_ratio_uses_conventional_orientation() -> None:
    """Positive association has a positive conventional LOR and Wald statistic."""
    interaction = Interaction(
        make_gene({0: 1.0}, [0], name="A_M"),
        make_gene({0: 1.0}, [0], name="B_M"),
    )
    co_taus = (0.6, 0.1, 0.1, 0.2)
    me_taus = (0.45, 0.25, 0.25, 0.05)

    assert interaction.compute_log_odds_ratio(co_taus) == pytest.approx(
        math.log(12),
    )
    assert interaction.compute_log_odds_ratio(me_taus) == pytest.approx(
        math.log(0.36),
    )
    assert interaction.compute_wald_statistic(co_taus) > 0
    assert interaction.compute_wald_statistic(me_taus) < 0
    assert LOG_ODDS_RATIO_CONTRACT == (
        "conventional-latent-odds-00x11-over-01x10-identifiable-v2"
    )


def test_latent_log_odds_ratio_avoids_product_underflow() -> None:
    """Strictly positive latent cells remain finite when products underflow."""
    interaction = Interaction(
        make_gene({0: 1.0}, [0], name="A_M"),
        make_gene({0: 1.0}, [0], name="B_M"),
    )
    taus = (1e-200, 0.5, 0.5, 1e-200)

    observed = interaction.compute_log_odds_ratio(taus)
    observed_wald = interaction.compute_wald_statistic(taus)

    assert observed is not None
    assert np.isfinite(observed)
    assert observed == pytest.approx(
        (2 * math.log(1e-200)) - (2 * math.log(0.5)),
    )
    assert observed_wald is not None
    assert np.isfinite(observed_wald)
    assert observed_wald == pytest.approx(
        observed * math.exp(-0.5 * float(np.logaddexp.reduce(-np.log(taus)))),
    )


def test_profile_lrt_uses_constrained_marginal_mles() -> None:
    """Regression: projected alternative marginals are not the profile null."""
    interaction = make_fitted_interaction()
    taus = np.array(
        [
            interaction.tau_00,
            interaction.tau_01,
            interaction.tau_10,
            interaction.tau_11,
        ],
    )
    projected_pi_a = taus[2] + taus[3]
    projected_pi_b = taus[1] + taus[3]
    projected_null = np.array(
        [
            (1 - projected_pi_a) * (1 - projected_pi_b),
            (1 - projected_pi_a) * projected_pi_b,
            projected_pi_a * (1 - projected_pi_b),
            projected_pi_a * projected_pi_b,
        ],
    )
    constrained_null = interaction.compute_independence_taus()

    assert projected_pi_a != pytest.approx(interaction.gene_a.pi, abs=1e-3)
    assert interaction.compute_log_likelihood(projected_null) < (
        interaction.compute_log_likelihood(constrained_null)
    )
    expected_lrt = 2 * (
        interaction.compute_log_likelihood(taus)
        - interaction.compute_log_likelihood(constrained_null)
    )
    assert interaction.compute_likelihood_ratio(taus) == pytest.approx(expected_lrt)


def test_profile_lrt_is_nested_nonnegative_and_versioned() -> None:
    """The deterministic alternative must dominate its constrained null."""
    interaction = make_fitted_interaction()
    assert interaction.fit_algorithm == PAIR_FIT_CONTRACT
    assert interaction.fit_converged
    assert interaction.fit_iterations > 0
    assert interaction.fit_fixed_point_residual <= PAIR_FIT_KKT_TOL
    assert interaction.fit_kkt_residual <= PAIR_FIT_KKT_TOL
    assert interaction.alternative_log_likelihood >= interaction.null_log_likelihood
    assert interaction.likelihood_ratio >= 0
    assert LRT_CONTRACT == "driver-independence-constrained-mle-v1"
    assert PAIR_FIT_CONTRACT == ("deterministic-simplex-coordinate-ascent-total-kkt-v2")
    assert PAIR_EFFECT_IDENTIFIABILITY_CONTRACT == (
        "full-affine-rank-relative-svd-1e-12-conservative-v1"
    )
    assert RHO_CONTRACT == ("marshall-olkin-identifiable-finite-or-degenerate-null-v2")


def test_pair_simplex_tolerance_has_no_relative_tolerance_backdoor() -> None:
    """Simplex checks use only the frozen absolute probability tolerance."""
    inside = np.array([0.25, 0.25, 0.25, 0.25 + (PAIR_SIMPLEX_TOL / 2)])
    outside = np.array([0.25, 0.25, 0.25, 0.25 + (PAIR_SIMPLEX_TOL * 2)])
    interaction = make_fitted_interaction()

    assert compute_marshall_olkin_rho(inside) is not None
    interaction.verify_taus_are_valid(inside)
    with pytest.raises(ValueError, match="invalid tau simplex"):
        compute_marshall_olkin_rho(outside)
    with pytest.raises(ValueError, match="Invalid tau parameters"):
        interaction.verify_taus_are_valid(outside)


def test_pair_simplex_rejects_tolerated_negative_cells_and_wide_override() -> None:
    interaction = make_fitted_interaction()
    negative = np.array(
        [-PAIR_SIMPLEX_TOL / 2, 0.5, 0.5, PAIR_SIMPLEX_TOL / 2],
    )

    with pytest.raises(ValueError, match="Invalid tau parameters"):
        interaction.verify_taus_are_valid(negative)
    with pytest.raises(ValueError, match="frozen absolute bound"):
        interaction.verify_taus_are_valid([0.0, 0.0, 0.0, 0.0], tol=1.0)


def test_pair_optimizer_rejects_looser_kkt_override() -> None:
    interaction = make_fitted_interaction()

    with pytest.raises(ValueError, match="frozen positive bound"):
        interaction.estimate_tau_with_coordinate_ascent(
            kkt_tol=PAIR_FIT_KKT_TOL * 10,
        )
    with pytest.raises(ValueError, match="frozen production bound"):
        interaction.estimate_tau_with_coordinate_ascent(
            max_iter=PAIR_FIT_MAX_ITER + 1,
        )


def test_public_pair_component_probabilities_remain_unscaled_products() -> None:
    interaction = Interaction(
        make_gene({0: 0.2, 1: 0.3, 2: 0.5}, [1], name="A_M"),
        make_gene({0: 0.4, 1: 0.1, 2: 0.5}, [1], name="B_M"),
    )

    assert interaction.component_probabilities()[0].tolist() == pytest.approx(
        [0.03, 0.12, 0.02, 0.08],
    )
    assert interaction.compute_total_probability(0.25, 0.25, 0.25, 0.25)[
        0
    ] == pytest.approx(0.0625)


def test_pair_kkt_certificate_is_on_total_log_likelihood_scale() -> None:
    """Duplicating observations strengthens the same per-row stationarity miss."""
    samples = ["s0", "s1"]
    counts_a = [0, 1]
    counts_b = [1, 0]
    pmf = {0: 0.5, 1: 0.5}
    base = Interaction(
        Gene("A_M", samples, counts_a, pmf),
        Gene("B_M", samples, counts_b, pmf),
    )
    repeats = 5
    duplicated_samples = [
        f"{sample}-{repeat}" for repeat in range(repeats) for sample in samples
    ]
    duplicated = Interaction(
        Gene("A_M", duplicated_samples, counts_a * repeats, pmf),
        Gene("B_M", duplicated_samples, counts_b * repeats, pmf),
    )
    taus = (0.7, 0.1, 0.1, 0.1)

    _base_fixed, base_kkt = base.compute_fit_certificates(taus)
    _duplicated_fixed, duplicated_kkt = duplicated.compute_fit_certificates(taus)

    assert base_kkt > 0
    assert duplicated_kkt == pytest.approx(repeats * base_kkt)


def test_flat_pair_fit_is_flagged_for_deterministic_tie_break_validation() -> None:
    """KKT alone cannot authenticate a tau choice on a nonidentifiable ridge."""
    pmf = {0: 0.5, 1: 0.5}
    gene_a = make_gene(pmf, [1, 1], name="A_M")
    gene_b = make_gene(pmf, [1, 1], name="B_M")
    gene_a.estimate_pi_with_mle()
    gene_b.estimate_pi_with_mle()
    interaction = Interaction(gene_a, gene_b)

    fixed_point, kkt = interaction.compute_fit_certificates(
        (0.25, 0.25, 0.25, 0.25),
    )

    assert fixed_point == pytest.approx(0.0)
    assert kkt == pytest.approx(0.0)
    assert interaction.has_full_affine_component_rank() is False
    assert interaction.effect_identifiability_status() == "rank-deficient"
    independent = (0.25, 0.25, 0.25, 0.25)
    associated = (0.4, 0.1, 0.1, 0.4)
    assert interaction.compute_log_likelihood(independent) == pytest.approx(
        interaction.compute_log_likelihood(associated),
    )
    assert interaction.compute_rho(independent) != interaction.compute_rho(associated)
    assert interaction.compute_log_odds_ratio(
        independent,
    ) != interaction.compute_log_odds_ratio(associated)
    assert interaction.compute_wald_statistic(
        independent,
    ) != interaction.compute_wald_statistic(associated)
    assert interaction.compute_rho_for_direction(independent, 1.0) is None
    assert interaction.compute_rho_for_direction(associated, 1.0) is None
    interaction.estimate_tau_with_coordinate_ascent()
    assert (
        interaction.tau_00,
        interaction.tau_01,
        interaction.tau_10,
        interaction.tau_11,
    ) == interaction.compute_independence_taus()


def test_affine_rank_threshold_is_replication_invariant(monkeypatch) -> None:
    """Duplicating all rows cannot change the effect-identifiability decision."""

    def rank(components: np.ndarray) -> int | None:
        interaction = Interaction(
            make_gene({0: 1.0}, [0], name="A_M"),
            make_gene({0: 1.0}, [0], name="B_M"),
        )
        monkeypatch.setattr(
            interaction,
            "_scaled_component_probabilities",
            lambda: components,
        )
        return interaction.affine_component_rank()

    for relative_scale, expected_rank in (
        (PAIR_IDENTIFIABILITY_RTOL * 10, 3),
        (PAIR_IDENTIFIABILITY_RTOL / 10, 2),
    ):
        components = np.column_stack(
            (
                np.zeros(3),
                np.diag((1.0, 1.0, relative_scale)),
            ),
        )
        assert rank(components) == expected_rank
        duplicated = np.tile(components, (37, 1))
        assert rank(duplicated) == expected_rank


def test_nested_likelihood_tolerance_is_applied_to_raw_lrt(monkeypatch) -> None:
    """A tiny numerical nesting miss is clipped, but a larger one is rejected."""
    interaction = make_fitted_interaction()
    null_taus = interaction.compute_independence_taus()
    inside_values = iter(
        [np.array([-(0.75 * LRT_NESTEDNESS_TOL) / 2]), np.array([0.0])],
    )
    monkeypatch.setattr(
        interaction,
        "_log_mixture_terms",
        lambda _taus: next(inside_values),
    )
    monkeypatch.setattr(
        interaction,
        "_absolute_log_likelihood_from_terms",
        lambda terms: float(math.fsum(terms)),
    )
    monkeypatch.setattr(interaction, "compute_independence_taus", lambda: null_taus)

    assert interaction.compute_likelihood_ratio(null_taus) == 0.0

    outside_values = iter(
        [np.array([-(1.25 * LRT_NESTEDNESS_TOL) / 2]), np.array([0.0])],
    )
    monkeypatch.setattr(
        interaction,
        "_log_mixture_terms",
        lambda _taus: next(outside_values),
    )
    with pytest.raises(ValueError, match="violates nestedness"):
        interaction.compute_likelihood_ratio(null_taus)


def test_failed_post_optimization_lrt_does_not_publish_partial_fit(
    monkeypatch,
) -> None:
    counts = [0, 0, 0, 0, 1, 1]
    pmf = {0: 0.8, 1: 0.2}
    gene_a = make_gene(pmf, counts, name="A_M")
    gene_b = make_gene(pmf, counts, name="B_M")
    gene_a.estimate_pi_with_mle()
    gene_b.estimate_pi_with_mle()
    interaction = Interaction(gene_a, gene_b)
    null_taus = np.asarray(interaction.compute_independence_taus(), dtype=float)
    monkeypatch.setattr(
        interaction,
        "_run_tau_coordinate_ascent",
        lambda *_args: (null_taus, 0, 0.0, 0.0, 0.0),
    )

    def fail_lrt(_taus):
        msg = "synthetic post-fit LRT failure"
        raise ValueError(msg)

    monkeypatch.setattr(
        interaction,
        "_profile_likelihood_ratio_components",
        fail_lrt,
    )

    with pytest.raises(ValueError, match="synthetic post-fit LRT failure"):
        interaction.estimate_tau_with_coordinate_ascent()

    assert interaction.fit_converged is False
    assert interaction.fit_algorithm is None
    assert interaction.tau_00 is None
    assert interaction.null_log_likelihood is None
    assert interaction.alternative_log_likelihood is None
    assert interaction.likelihood_ratio is None


def test_legacy_optimizer_initialization_uses_absolute_tolerance_only() -> None:
    """Relative tolerance cannot admit a materially different legacy seed."""
    interaction = make_fitted_interaction()
    perturbed = np.asarray(interaction.compute_independence_taus(), dtype=float)
    perturbed[0] -= PAIR_SIMPLEX_TOL * 2
    perturbed[1] += PAIR_SIMPLEX_TOL * 2

    with pytest.raises(ValueError, match="must initialize at the independence null"):
        interaction.estimate_tau_with_optimization_using_scipy(
            tau_init=perturbed.tolist(),
        )


_CHOL_MUTSIG_SLOW_COMPONENTS = {
    "PPFIBP2_M": (
        {0: 0.95628775, 30: 0.95628775},
        {14: (0.04145994, 0.95762948), 24: (0.07672444, 0.91998542)},
    ),
    "TPRN_M": (
        {
            21: 0.98963166,
            23: 0.98963166,
            24: 0.96633157,
            28: 0.92224246,
            34: 0.92224246,
        },
        {6: (0.0103144, 0.98963166), 19: (0.07190619, 0.92522582)},
    ),
    "TMC4_M": (
        {10: 0.97507359, 28: 0.98725784, 33: 0.98528581, 34: 0.98725784},
        {24: (0.03042458, 0.96909279), 27: (0.00785576, 0.99211305)},
    ),
    "UHMK1_M": (
        {
            5: 0.98095887,
            9: 0.98095887,
            13: 0.98002886,
            14: 0.99174748,
            18: 0.98913706,
            29: 0.98002886,
            33: 0.98154353,
        },
        {24: (0.0080738, 0.99189325), 31: (0.02925183, 0.97030278)},
    ),
    "BRD7_N": (
        {0: 0.99304564, 24: 0.9990005, 30: 0.99304564},
        {17: (0.00142591, 0.99857307)},
    ),
    "PBRM1_N": (
        {13: 0.97804826},
        {
            6: (0.01127524, 0.98866022),
            18: (0.01826949, 0.98155942),
            23: (0.00783706, 0.99213191),
            29: (0.0424874, 0.95655489),
            31: (0.02170901, 0.97804826),
        },
    ),
    "AGAP2_M": (
        {
            6: 0.99449067,
            7: 0.99449067,
            15: 0.94097092,
            21: 0.99449067,
            23: 0.99449067,
            24: 0.96405418,
            27: 0.99449067,
        },
        {16: (0.03049546, 0.96901962)},
    ),
}


def make_chol_mutsig_slow_gene(name: str) -> Gene:
    """Recreate a compact exact-component fixture from the CHOL MutSig canary."""
    zero_count_overrides, mutated = _CHOL_MUTSIG_SLOW_COMPONENTS[name]
    counts = [int(index in mutated) for index in range(36)]
    pmfs = []
    for index, count in enumerate(counts):
        if count == 0:
            pmfs.append({0: zero_count_overrides.get(index, 1.0)})
        else:
            p_no_driver, p_driver = mutated[index]
            pmfs.append({0: p_driver, 1: p_no_driver})
    return Gene(name=name, samples=list(range(36)), counts=counts, bmr_pmf=pmfs)


@pytest.mark.parametrize(
    ("gene_a_name", "gene_b_name"),
    [
        ("PPFIBP2_M", "TPRN_M"),
        ("TMC4_M", "UHMK1_M"),
        ("PPFIBP2_M", "BRD7_N"),
        ("PBRM1_N", "AGAP2_M"),
    ],
)
def test_coordinate_ascent_certifies_real_chol_mutsig_slow_shapes(
    gene_a_name: str,
    gene_b_name: str,
) -> None:
    """Regression: four real shapes that exceeded 5,000 EM iterations fit quickly."""
    interaction = Interaction(
        make_chol_mutsig_slow_gene(gene_a_name),
        make_chol_mutsig_slow_gene(gene_b_name),
    )
    interaction.estimate_tau_with_coordinate_ascent()

    assert interaction.fit_converged
    assert interaction.fit_algorithm == PAIR_FIT_CONTRACT
    assert interaction.fit_iterations <= 50
    assert interaction.fit_fixed_point_residual <= PAIR_FIT_KKT_TOL
    assert interaction.fit_kkt_residual <= PAIR_FIT_KKT_TOL
    assert interaction.alternative_log_likelihood >= interaction.null_log_likelihood
    assert interaction.likelihood_ratio >= 0


def test_coordinate_ascent_requires_the_exact_null_initialization() -> None:
    """A merely close alternative initialization cannot change the frozen fit."""
    interaction = make_fitted_interaction()
    null_taus = np.asarray(interaction.compute_independence_taus())
    perturbed = null_taus.copy()
    perturbed[0] -= 1e-12
    perturbed[1] += 1e-12

    with pytest.raises(ValueError, match="exact independence null"):
        interaction.estimate_tau_with_coordinate_ascent(tau_init=perturbed)


def test_coordinate_ascent_preserves_an_exact_boundary_null_optimum() -> None:
    """An already-optimal boundary null needs no interior epsilon or fit step."""
    interaction = Interaction(
        make_gene({0: 1.0}, [0, 0], name="A_M"),
        make_gene({0: 1.0}, [0, 0], name="B_M"),
    )
    interaction.estimate_tau_with_coordinate_ascent()

    assert (
        interaction.tau_00,
        interaction.tau_01,
        interaction.tau_10,
        interaction.tau_11,
    ) == (1.0, 0.0, 0.0, 0.0)
    assert interaction.fit_iterations == 0
    assert interaction.fit_kkt_residual == 0
    assert interaction.likelihood_ratio == 0
    assert (
        interaction.compute_rho_for_direction(
            (1.0, 0.0, 0.0, 0.0),
            interaction.likelihood_ratio,
        )
        is None
    )


def test_undefined_rho_fails_closed_for_a_positive_lrt() -> None:
    """A blank direction is valid only at a numerically null pair LRT."""
    pmf = {0: 0.8, 1: 0.2}
    interaction = Interaction(
        make_gene(pmf, [0, 0, 1, 1], name="A_M"),
        make_gene(pmf, [0, 1, 0, 1], name="B_M"),
    )
    assert interaction.has_full_affine_component_rank()

    with pytest.raises(ValueError, match="undefined rho with positive"):
        interaction.compute_rho_for_direction(
            (1.0, 0.0, 0.0, 0.0),
            UNDEFINED_RHO_LRT_TOL * 2,
        )


def test_rho_avoids_intermediate_pair_product_underflow() -> None:
    """A representable rho stays finite when the determinant products underflow."""
    interaction = Interaction(
        make_gene({0: 1.0}, [0], name="A_M"),
        make_gene({0: 1.0}, [0], name="B_M"),
    )
    rho = interaction.compute_rho((1.0, 1e-200, 1e-200, 0.0))

    assert rho == pytest.approx(-1e-200, rel=1e-12, abs=0.0)
    assert np.isfinite(rho)


def test_pair_likelihood_preserves_nonmax_component_product_in_log_domain() -> None:
    """Two positive 1e-200 components must not become false zero support."""
    pmf = {0: 1e-200, 1: 1.0}
    interaction = Interaction(
        make_gene(pmf, [1], name="A_M"),
        make_gene(pmf, [1], name="B_M"),
    )

    log_likelihood = interaction.compute_log_likelihood((0.0, 0.0, 0.0, 1.0))

    assert np.isfinite(log_likelihood)
    assert log_likelihood == pytest.approx(2 * math.log(1e-200))
    assert interaction.effect_identifiability_status() == "rank-not-certified-underflow"
    interaction.estimate_tau_with_coordinate_ascent()
    assert interaction.fit_converged
    assert interaction.fit_kkt_residual <= PAIR_FIT_KKT_TOL


def test_compensated_likelihood_preserves_real_chol_cbase_ascent() -> None:
    """Regression: summation roundoff must not turn a real ascent step negative."""
    counts_a = [0] * 36
    counts_b = [0] * 36
    counts_a[6] = 1
    counts_a[13] = 1
    counts_a[24] = 3
    counts_b[16] = 2
    interaction = Interaction(
        make_gene(
            {
                0: 0.9948040877991979,
                1: 0.005174525936536885,
                2: 2.1305949738692595e-05,
                3: 8.002739901850178e-08,
            },
            counts_a,
            name="GPR98_M",
        ),
        make_gene(
            {
                0: 0.997701862803901,
                1: 0.002291803914557476,
                2: 6.3148617645707275e-06,
            },
            counts_b,
            name="CGGBP1_M",
        ),
    )

    interaction.estimate_tau_with_coordinate_ascent()

    assert interaction.fit_converged
    assert interaction.fit_iterations <= 20
    assert interaction.fit_kkt_residual <= PAIR_FIT_KKT_TOL
    assert interaction.alternative_log_likelihood >= interaction.null_log_likelihood


def test_direct_gain_certifies_real_chol_cbase_microsteps() -> None:
    """Regression: rounded tau updates retain a nonnegative direct gain proof."""
    counts_a = [0] * 36
    counts_b = [0] * 36
    counts_a[24] = 3
    counts_b[14] = 1
    counts_b[24] = 1
    interaction = Interaction(
        make_gene(
            {
                0: 0.9976253178287274,
                1: 0.0023679200423174914,
                2: 6.741807120483953e-06,
                3: 2.0259160362054814e-08,
            },
            counts_a,
            name="LRP6_M",
        ),
        make_gene(
            {0: 0.9978784463689733, 1: 0.0021161560983326035},
            counts_b,
            name="PPFIBP2_M",
        ),
    )

    interaction.estimate_tau_with_coordinate_ascent()

    assert interaction.fit_converged
    assert interaction.fit_iterations <= 40
    assert interaction.fit_last_log_likelihood_gain >= 0
    assert interaction.fit_kkt_residual <= PAIR_FIT_KKT_TOL


def test_interaction_estimation_propagates_failure(monkeypatch) -> None:
    """The cohort utility must not silently replace a failed pair by independence."""
    gene_a = make_gene({0: 1.0}, [0], name="A_M")
    gene_b = make_gene({0: 1.0}, [0], name="B_M")
    interaction = Interaction(gene_a, gene_b)

    def fail() -> None:
        msg = "deliberate fit failure"
        raise ValueError(msg)

    monkeypatch.setattr(interaction, "estimate_tau_with_coordinate_ascent", fail)
    with pytest.raises(ValueError, match="deliberate fit failure"):
        estimate_taus_for_each_interaction([interaction])
    assert interaction.tau_00 is None

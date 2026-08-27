"""Unit tests for the DIALECT statistical model (Gene likelihood + BMR math)."""

import math

import numpy as np
import pytest

from dialect.models.gene import Gene
from dialect.models.interaction import (
    LRT_CONTRACT,
    PAIR_FIT_CONTRACT,
    PAIR_FIT_KKT_TOL,
    RHO_CONTRACT,
    UNDEFINED_RHO_LRT_TOL,
    Interaction,
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
    assert gene.likelihood_ratio_status == (
        "infinite-passenger-null-zero-probability"
    )


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
    assert PAIR_FIT_CONTRACT == "deterministic-simplex-coordinate-ascent-v1"
    assert RHO_CONTRACT == "marshall-olkin-finite-or-degenerate-null-v1"


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
    assert interaction.fit_iterations <= 25
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
    assert interaction.compute_rho_for_direction(
        (1.0, 0.0, 0.0, 0.0),
        interaction.likelihood_ratio,
    ) is None


def test_undefined_rho_fails_closed_for_a_positive_lrt() -> None:
    """A blank direction is valid only at a numerically null pair LRT."""
    interaction = Interaction(
        make_gene({0: 1.0}, [0], name="A_M"),
        make_gene({0: 1.0}, [0], name="B_M"),
    )

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
    assert interaction.fit_iterations <= 10
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
    assert interaction.fit_iterations <= 20
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

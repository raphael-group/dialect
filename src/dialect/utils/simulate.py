import numpy as np

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def simulate_single_gene_passengers(bmr_pmf, nsamples):
    """
    Simulate passenger mutation counts for a single gene, ensuring that adding a driver mutation
    does not exceed the maximum count in the background mutation rate (BMR PMF).

    :param bmr_pmf (dict): Background mutation rate probability mass function (PMF).
    :param nsamples (int): Number of samples to simulate.
    :return np.ndarray: Passenger mutation counts for each sample.
    """
    if not np.isclose(sum(bmr_pmf.values()), 1.0):
        raise ValueError("Background mutation rates (bmr_pmf) must sum to 1.")

    # Exclude the last mutation count to leave space for driver mutation addition
    max_count = max(bmr_pmf.keys())
    adjusted_pmf = {k: v for k, v in bmr_pmf.items() if k < max_count}
    adjusted_pmf_sum = sum(adjusted_pmf.values())
    adjusted_pmf = {k: v / adjusted_pmf_sum for k, v in adjusted_pmf.items()}
    return np.random.choice(
        list(adjusted_pmf.keys()), nsamples, p=list(adjusted_pmf.values())
    )


def simulate_single_gene_drivers(nsamples, pi):
    """
    Simulate driver mutations for a single gene.

    :param nsamples (int): Number of samples to simulate.
    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :return np.ndarray: Driver mutation counts (0 or 1) for each sample.
    """
    if not (0 <= pi <= 1):
        raise ValueError("Driver mutation rate pi must be between 0 and 1.")
    return np.random.binomial(1, pi, size=nsamples)


def simulate_pairwise_gene_drivers(nsamples, tau_01, tau_10, tau_11):
    """
    Simulate driver mutations for an interacting gene pair.

    :param nsamples (int): Number of samples to simulate.
    :param tau_01 (float): Probability (D=0, D'=1).
    :param tau_10 (float): Probability (D=1, D'=0).
    :param tau_11 (float): Probability (D=1, D'=1).
    :return tuple: Driver mutation counts for gene_a and gene_b.
    """
    gene_a_drivers = np.zeros(nsamples)
    gene_b_drivers = np.zeros(nsamples)
    rnd = np.random.uniform(size=nsamples)

    both_mutations = rnd < tau_11
    only_gene_b_mutations = (rnd >= tau_11) & (rnd < tau_11 + tau_01)
    only_gene_a_mutations = (rnd >= tau_11 + tau_01) & (rnd < tau_11 + tau_01 + tau_10)

    gene_a_drivers[both_mutations | only_gene_a_mutations] = 1
    gene_b_drivers[both_mutations | only_gene_b_mutations] = 1

    return gene_a_drivers, gene_b_drivers


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def simulate_single_gene(bmr_pmf, nsamples, pi):
    """
    Simulate somatic mutation counts for a single gene.

    The somatic mutation count is the sum of passenger mutations (from the BMR PMF)
    and driver mutations (simulated using a Bernoulli process with rate pi).

    :param bmr_pmf (list): Background mutation rate probability mass function (PMF).
    :param nsamples (int): Number of samples to simulate.
    :param pi (float): Driver mutation rate (0 <= pi <= 1).
    :return np.ndarray: Simulated somatic mutation counts for each sample.
    """
    passenger_counts = simulate_single_gene_passengers(bmr_pmf, nsamples)
    driver_counts = simulate_single_gene_drivers(nsamples, pi)
    somatic_mutations = (passenger_counts + driver_counts).astype(int)
    simulated_gene = Gene(
        name="SimulatedGene",
        samples=range(nsamples),
        counts=somatic_mutations,
        bmr_pmf=bmr_pmf,
    )
    return simulated_gene


def simulate_interaction_pair(gene_a_pmf, gene_b_pmf, tau_01, tau_10, tau_11, nsamples):
    """
    Simulate an interaction between two genes.

    :param gene_a_pmf (list): Background mutation rate PMF for gene A.
    :param gene_b_pmf (list): Background mutation rate PMF for gene B.
    :param tau_01 (float): Probability of (D=0, D'=1).
    :param tau_10 (float): Probability of (D=1, D'=0).
    :param tau_11 (float): Probability of (D=1, D'=1).
    :param nsamples (int): Number of samples to simulate.
    :return tuple: Simulated Gene objects for gene A and gene B.
    """
    gene_a_passengers = simulate_single_gene_passengers(gene_a_pmf, nsamples)
    gene_b_passengers = simulate_single_gene_passengers(gene_b_pmf, nsamples)

    gene_a_drivers, gene_b_drivers = simulate_pairwise_gene_drivers(
        nsamples, tau_01, tau_10, tau_11
    )

    gene_a_counts = (gene_a_passengers + gene_a_drivers).astype(int)
    gene_b_counts = (gene_b_passengers + gene_b_drivers).astype(int)

    simulated_gene_a = Gene(
        name="SimulatedGeneA",
        samples=range(nsamples),
        counts=gene_a_counts,
        bmr_pmf=gene_a_pmf,
    )
    simulated_gene_b = Gene(
        name="SimulatedGeneB",
        samples=range(nsamples),
        counts=gene_b_counts,
        bmr_pmf=gene_b_pmf,
    )

    simulated_interaction = Interaction(
        gene_a=simulated_gene_a, gene_b=simulated_gene_b
    )

    return simulated_interaction

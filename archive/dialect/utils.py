import numpy as np
from argparse import ArgumentParser
from scipy.optimize import minimize

MAX_ITER = 30  # Maximum number of iterations for the EM algorithm
EPSILON = 1e-6  # Convergence threshold for the EM algorithm


def get_parser():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Choose a command")
    # Subparser for BMR and count matrix generation
    generate_parser = subparsers.add_parser("generate", help="Generate BMR and count matrix")
    generate_parser.add_argument("maf_fn", help="Path to the input MAF file")
    generate_parser.add_argument("dout", help="Path to the output directory")
    generate_parser.add_argument("--method", default="cbase", choices=["cbase"])
    generate_parser.add_argument("--reference", default="hg19", choices=["hg19", "hg38"])
    # Subparser for dialect analysis
    analyze_parser = subparsers.add_parser("analyze", help="Run DIALECT analysis")
    analyze_parser.add_argument("cnt_mtx_fn", help="Path to the count matrix file")
    analyze_parser.add_argument("bmr_fn", help="Path to the BMR file")
    analyze_parser.add_argument("dout", help="Path to the output directory")
    analyze_parser.add_argument("--top_k", default=100, type=int)
    # Subparser for comparison analysis
    compare_parser = subparsers.add_parser("compare", help="Run comparison analysis")
    compare_parser.add_argument("method", choices=["fishers", "discover", "wesme"])
    compare_parser.add_argument("cnt_mtx_fn", help="Path to the count matrix file")
    compare_parser.add_argument("dout", help="Path to the output directory")
    compare_parser.add_argument("--top_k", default=100, type=int)
    compare_parser.add_argument(
        "--feature_level",
        default="mutation",
        choices=["gene", "mutation"],
        help="Feature level to use for comparison test",
    )
    return parser


def log_likelihood(pi, background_mixture, driver_mixture):
    total_prob = (1 - pi) * background_mixture + pi * driver_mixture
    nonzero_total_prob = total_prob[total_prob != 0]
    return np.sum(np.log(nonzero_total_prob))


def negative_log_likelihood(pi, background_mixture, driver_mixture):
    return -log_likelihood(pi, background_mixture, driver_mixture)


def generate_pi_inits(init_count, min_pi, max_pi):
    return np.linspace(min_pi, max_pi, init_count)


def em_single(pi, background_mixture, driver_mixture):
    for _ in range(MAX_ITER):
        prev_pi = pi.copy()
        total_prob = (1 - pi) * background_mixture + pi * driver_mixture

        # TODO: CHECK THAT REMOVING ZEROS IS VALID AND DONE ELSEWHERE
        nonzero_total_prob_mask = total_prob != 0  # Ensures division by zero does not occur
        z_1 = (pi * driver_mixture)[nonzero_total_prob_mask] / total_prob[nonzero_total_prob_mask]

        pi = np.mean(z_1)
        prev_log_likelihood = log_likelihood(prev_pi, background_mixture, driver_mixture)
        curr_log_likelihood = log_likelihood(pi, background_mixture, driver_mixture)
        if np.abs(curr_log_likelihood - prev_log_likelihood) < EPSILON:
            break
    return pi


def optimizer_single(pi, background_mixture, driver_mixture):
    alpha = 1e-13  # A small positive number to avoid bounds of zero, which are problematic in optimization
    bounds = [(alpha, 1 - alpha)]
    result = minimize(
        negative_log_likelihood,
        pi,
        args=(background_mixture, driver_mixture),
        bounds=bounds,
    )
    pi = result.x  # Extract the optimized value of 'pi' from the result object
    return pi


def generate_null_tau(gene_a_pi, gene_b_pi):
    # Calculate probabilities for each combination under independence assumption
    return [
        (1 - gene_a_pi) * (1 - gene_b_pi),  # Probability neither gene has a driver mutation
        (1 - gene_a_pi) * gene_b_pi,  # Probability only gene B has a driver mutation
        gene_a_pi * (1 - gene_b_pi),  # Probability only gene A has a driver mutation
        gene_a_pi * gene_b_pi,  # Probability both genes have a driver mutation
    ]


def generate_pairwise_tau_inits(init_count, gene_a_pi, gene_b_pi, min_tau_no_drivers=0.1):
    # Generate linearly spaced values for tau with no drivers
    tau_no_drivers = np.linspace(min_tau_no_drivers, 1, init_count)
    # Create random tau initializations based on the no driver values
    random_tau_inits = np.array([(1 - p) / 3 for p in tau_no_drivers]).reshape(-1, 1) * np.array(
        [0, 1, 1, 1]
    ) + np.column_stack((tau_no_drivers, np.zeros((init_count, 3))))
    # Define custom initializations based on input pi values for gene A and B
    custom_tau_inits = np.array(
        [
            [
                1 - np.max([gene_a_pi, gene_b_pi]),  # Probability of no drivers
                (
                    0 if gene_b_pi < gene_a_pi else gene_b_pi - gene_a_pi
                ),  # Adjusted for gene B's excess
                (
                    0 if gene_a_pi < gene_b_pi else gene_a_pi - gene_b_pi
                ),  # Adjusted for gene A's excess
                np.min([gene_a_pi, gene_b_pi]),  # Minimum of pi values, assuming co-occurrence
            ],
            [
                1 - (gene_a_pi + gene_b_pi),
                gene_b_pi,
                gene_a_pi,
                0,
            ],  # Mutual exclusivity scenario
            generate_null_tau(gene_a_pi, gene_b_pi),  # Null model, assuming independence
        ]
    )
    # Stack and return all initial tau values
    return np.vstack((random_tau_inits, custom_tau_inits))


def calculate_log_odds_and_std_err(tau, num_samples):
    cont_table_cnts = tau * num_samples
    cont_table_cnts = cont_table_cnts + 0.5 # Haldone-Ascombe Correction
    odds_ratio = (cont_table_cnts[1] * cont_table_cnts[2]) / (cont_table_cnts[0] * cont_table_cnts[3])
    std_err = np.sqrt(1 / cont_table_cnts[0] + 1 / cont_table_cnts[1] + 1 / cont_table_cnts[2] + 1 / cont_table_cnts[3])
    return np.log(odds_ratio), std_err


def log_likelihood_pair(
    tau_no_drivers,  # no drivers in genes a and b
    tau_driver_b_only,  # exclusivity w/ no drivers in gene a and drivers in gene b
    tau_driver_a_only,  # exclusivity w/ no drivers in gene b and drivers in gene a
    tau_both_drivers,  # co-occurrence with drivers in genes a and b
    gene_a_background_mixture,
    gene_b_background_mixture,
    gene_a_driver_mixture,
    gene_b_driver_mixture,
):
    total_prob = (
        tau_no_drivers * gene_a_background_mixture * gene_b_background_mixture
        + tau_driver_b_only * gene_a_background_mixture * gene_b_driver_mixture
        + tau_driver_a_only * gene_a_driver_mixture * gene_b_background_mixture
        + tau_both_drivers * gene_a_driver_mixture * gene_b_driver_mixture
    )
    nonzero_total_prob = total_prob[total_prob != 0]  # TODO: VALIDATE THIS APPROACH
    return np.sum(np.log(nonzero_total_prob))


def em_pair(
    taus,
    gene_a_background_mixture,
    gene_b_background_mixture,
    gene_a_driver_mixture,
    gene_b_driver_mixture,
):
    tau_no_drivers, tau_driver_b_only, tau_driver_a_only, tau_both_drivers = taus
    for _ in range(MAX_ITER):
        prev_taus = [
            tau_no_drivers,
            tau_driver_b_only,
            tau_driver_a_only,
            tau_both_drivers,
        ].copy()
        total_prob = (
            tau_no_drivers * gene_a_background_mixture * gene_b_background_mixture
            + tau_driver_b_only * gene_a_background_mixture * gene_b_driver_mixture
            + tau_driver_a_only * gene_a_driver_mixture * gene_b_background_mixture
            + tau_both_drivers * gene_a_driver_mixture * gene_b_driver_mixture
        )
        nonzero_total_prob_mask = (
            total_prob != 0
        )  # Avoid division by zero #TODO: VALIDATE THIS MASK APPROACH
        z_00 = (
            tau_no_drivers
            * gene_a_background_mixture[nonzero_total_prob_mask]
            * gene_b_background_mixture[nonzero_total_prob_mask]
            / total_prob[nonzero_total_prob_mask]
        )
        z_01 = (
            tau_driver_b_only
            * gene_a_background_mixture[nonzero_total_prob_mask]
            * gene_b_driver_mixture[nonzero_total_prob_mask]
            / total_prob[nonzero_total_prob_mask]
        )
        z_10 = (
            tau_driver_a_only
            * gene_a_driver_mixture[nonzero_total_prob_mask]
            * gene_b_background_mixture[nonzero_total_prob_mask]
            / total_prob[nonzero_total_prob_mask]
        )
        z_11 = (
            tau_both_drivers
            * gene_a_driver_mixture[nonzero_total_prob_mask]
            * gene_b_driver_mixture[nonzero_total_prob_mask]
            / total_prob[nonzero_total_prob_mask]
        )

        tau_no_drivers = np.mean(z_00)
        tau_driver_b_only = np.mean(z_01)
        tau_driver_a_only = np.mean(z_10)
        tau_both_drivers = np.mean(z_11)
        prev_log_likelihood = log_likelihood_pair(
            prev_taus[0],
            prev_taus[1],
            prev_taus[2],
            prev_taus[3],
            gene_a_background_mixture,
            gene_b_background_mixture,
            gene_a_driver_mixture,
            gene_b_driver_mixture,
        )
        curr_log_likelihood = log_likelihood_pair(
            tau_no_drivers,
            tau_driver_b_only,
            tau_driver_a_only,
            tau_both_drivers,
            gene_a_background_mixture,
            gene_b_background_mixture,
            gene_a_driver_mixture,
            gene_b_driver_mixture,
        )
        if np.abs(curr_log_likelihood - prev_log_likelihood) < EPSILON:
            break
    return tau_no_drivers, tau_driver_b_only, tau_driver_a_only, tau_both_drivers

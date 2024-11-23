import logging
import numpy as np
from scipy.optimize import minimize

from dialect.models.gene import Gene


class Interaction:
    def __init__(self, gene_a, gene_b):
        """
        Initialize an Interaction object to represent the interaction between two genes.

        :param gene_a (Gene): The first gene in the interaction.
        :param gene_b (Gene): The second gene in the interaction.
        """
        if not isinstance(gene_a, Gene) or not isinstance(gene_b, Gene):
            raise ValueError("Both inputs must be instances of the Gene class.")

        self.gene_a = gene_a
        self.gene_b = gene_b
        self.name = f"{gene_a.name}:{gene_b.name}"  # Interaction name
        self.tau_00 = None  # P(D = 0, D' = 0) for genes A and B
        self.tau_01 = None  # P(D = 0, D' = 1) for genes A and B
        self.tau_10 = None  # P(D = 1, D' = 0) for genes A and B
        self.tau_11 = None  # P(D = 1, D' = 1) for genes A and B

    # ---------------------------------------------------------------------------- #
    #                        Likelihood & Metric Evaluation                        #
    # ---------------------------------------------------------------------------- #

    # Placeholder for methods to compute interaction-specific metrics (e.g., likelihood ratio)
    # def compute_log_likelihood(self):
    #     pass
    #
    # def compute_likelihood_ratio(self):
    #     pass
    #
    # def compute_log_odds_ratio(self):
    #     pass

    # ---------------------------------------------------------------------------- #
    #                         Parameter Estimation Methods                         #
    # ---------------------------------------------------------------------------- #

    # Placeholder for methods to estimate interaction parameters
    # def estimate_interaction_with_optimization(self):
    #     pass
    #
    # def estimate_interaction_with_em(self):
    #     pass

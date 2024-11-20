import numpy as np


class Gene:
    def __init__(self, name, counts, bmr_pmf=None):
        """
        Initialize a Gene object.

        :param name: Name of the gene.
        :param counts: Mutation counts for the gene.
        :param bmr_pmf: BMR PMF (multinomial passed as single list).
        """
        self.name = name
        self.counts = counts
        self.bmr_pmf = bmr_pmf
        self.pi = None

    def compute_log_likelihood(self, pi):
        """
        Compute the log likelihood of the data given parameter pi.

        :param pi: Parameter pi (float).
        :return: Log likelihood (float).
        """
        if self.bmr_pmf is None:
            raise ValueError("BMR PMF is not defined for this gene.")
        if self.counts is None:
            raise ValueError("Counts are not defined for this gene.")
        raise NotImplementedError(
            "Log likelihood ratio computation is not yet implemented."
        )

    def binarize_counts(self):
        """
        Get the binarized counts for the gene based on a threshold.

        :param threshold: Threshold for binarization (default is 1).
        :return: Binarized counts (numpy array).
        """
        return (self.counts >= 1).astype(int)

    def get_contingency_table(self, other_gene_counts):
        """
        Get the contingency table for the current gene and another gene.

        :param other_gene_counts: Counts of another gene (numpy array).
        :return: Contingency table (2x2 numpy array).
        """
        raise NotImplementedError(
            "Contingency table computation is not yet implemented."
        )

    def compute_likelihood_ratio(self, contingency_table):
        """
        Compute the likelihood ratio with respect to the null hypothesis.

        :param contingency_table: A 2x2 contingency table.
        :return: Likelihood ratio (float).
        """
        # Implement a proper likelihood ratio computation based on the contingency table.
        raise NotImplementedError(
            "Likelihood ratio computation is not yet implemented."
        )

    def compute_log_odds_ratio(self, contingency_table):
        """
        Compute the log odds ratio from the contingency table.

        :param contingency_table: A 2x2 contingency table.
        :return: Log odds ratio (float).
        """
        raise NotImplementedError("Log odds ratio computation is not yet implemented.")

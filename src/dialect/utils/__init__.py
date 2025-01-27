"""Export core functions and utilities for the DIALECT package."""

from .argument_parser import build_argument_parser
from .compare import run_comparison_methods
from .generate import generate_bmr_and_counts
from .helpers import read_cbase_results_file
from .identify import identify_pairwise_interactions
from .logger import configure_logging
from .merge import merge_pairwise_interaction_results
from .simulate import (
    create_matrix_simulation,
    create_pair_gene_simulation,
    create_single_gene_simulation,
    evaluate_matrix_simulation,
    evaluate_pair_gene_simulation,
    evaluate_single_gene_simulation,
)

__all__ = [
    "build_argument_parser",
    "configure_logging",
    "create_matrix_simulation",
    "create_pair_gene_simulation",
    "create_single_gene_simulation",
    "evaluate_matrix_simulation",
    "evaluate_pair_gene_simulation",
    "evaluate_single_gene_simulation",
    "generate_bmr_and_counts",
    "identify_pairwise_interactions",
    "merge_pairwise_interaction_results",
    "read_cbase_results_file",
    "run_comparison_methods",
]

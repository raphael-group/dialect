from argparse import ArgumentParser


def build_argument_parser():
    """
    Creates and returns the argument parser for the command line interface.
    """
    parser = ArgumentParser(description="DIALECT")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for BMR and count matrix generation
    generate_parser = subparsers.add_parser(
        "generate", help="Generate BMR and count matrix"
    )
    generate_parser.add_argument(
        "-m", "--maf", required=True, help="Path to the input MAF file"
    )
    generate_parser.add_argument(
        "-o", "--out", required=True, help="Path to the output directory"
    )
    generate_parser.add_argument(
        "-r",
        "--reference",
        default="hg19",
        choices=["hg19", "hg38"],
        help="Reference genome (default: hg19)",
    )

    # Subparser for DIALECT analysis
    identify_parser = subparsers.add_parser(
        "identify", help="Run DIALECT to identify interactions"
    )
    identify_parser.add_argument(
        "-c", "--cnt", required=True, help="Path to the input count matrix file"
    )
    identify_parser.add_argument(
        "-b", "--bmr", required=True, help="Path to the BMR file"
    )
    identify_parser.add_argument(
        "-o", "--out", required=True, help="Path to the output directory"
    )
    identify_parser.add_argument(
        "-k",
        "--top_k",
        default=100,
        type=int,
        help="Number of genes to consider (default: 100 genes with highest mutation count)",
    )
    identify_parser.add_argument(
        "-cb",
        "--cbase_stats",
        default=None,
        help="Path to the cbase results file",
    )

    # Subparser for running comparison methods (fisher, discover, etc.)
    compare_parser = subparsers.add_parser("compare", help="Run alternative methods")
    compare_parser.add_argument(
        "-c", "--cnt", required=True, help="Path to the input count matrix file"
    )
    compare_parser.add_argument(
        "-b", "--bmr", required=True, help="Path to the BMR file"
    )
    compare_parser.add_argument(
        "-o", "--out", required=True, help="Path to the output directory"
    )
    compare_parser.add_argument(
        "-k",
        "--top_k",
        default=100,
        type=int,
        help="Number of genes to consider (default: 100 genes with highest mutation count)",
    )

    # Subparser for simulations
    simulate_parser = subparsers.add_parser(
        "simulate", help="Run simulations for evaluation and benchmarking"
    )
    simulate_subparsers = simulate_parser.add_subparsers(
        dest="simulate_command", help="Available simulation commands"
    )

    # TODO: finish implementing simulation subparsers
    # TODO: add arguments to create simulation vs. evaluate simulation
    # TODO: add arguments for bmr file, simulation parameters, etc.
    # TODO: add evaluate subparser arguments to create tables and plots
    # Subparser for creating simulations
    simulate_create_parser = simulate_subparsers.add_parser(
        "create", help="Create simulation data"
    )
    simulate_evaluate_parser = simulate_subparsers.add_parser(
        "evaluate", help="Evaluate simulation results"
    )

    return parser

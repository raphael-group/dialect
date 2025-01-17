from argparse import ArgumentParser, ArgumentTypeError

# TODO: unify shared arguments across subparsers (output, seed, etc.) and create argument groups


def add_generate_parser(subparsers):
    """
    Adds the generate subparser to the given subparsers.
    """
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


def add_identify_parser(subparsers):
    """
    Adds the identify subparser to the given subparsers.
    """
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


def add_compare_parser(subparsers):
    """
    Adds the compare subparser to the given subparsers.
    """
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


def add_merge_parser(subparsers):
    """
    Adds the merge subparser to the given subparsers.
    """
    merge_parser = subparsers.add_parser(
        "merge", help="Merge DIALECT and alternative method results"
    )
    merge_parser.add_argument(
        "-d",
        "--dialect",
        required=True,
        help="Path to the DIALECT pairwise interaction results",
    )
    merge_parser.add_argument(
        "-a",
        "--alt",
        required=True,
        help="Path to the comparison interaction results",
    )
    merge_parser.add_argument(
        "-o", "--out", required=True, help="Path to the output directory"
    )


def add_simulate_create_single_gene_parser(subparsers):
    single_gene_parser = subparsers.add_parser(
        "single", help="Create single gene simulations"
    )
    single_gene_parser.add_argument(
        "-pi",
        "--pi",
        type=lambda x: (
            float(x)
            if 0 <= float(x) <= 1
            else ArgumentTypeError("Value for --pi must be between 0 and 1")
        ),
        required=True,
        help="Pi value (must be between 0 and 1)",
    )
    single_gene_parser.add_argument("-n", "--num_samples", type=int, default=1000)
    single_gene_parser.add_argument("-ns", "--num_simulations", type=int, default=2500)
    single_gene_parser.add_argument(
        "-b", "--bmr", required=True, help="Path to the BMR file"
    )
    single_gene_parser.add_argument(
        "-g", "--gene", required=True, type=str, help="Gene name"
    )
    single_gene_parser.add_argument("-o", "--out", type=str, required=True)
    single_gene_parser.add_argument("-s", "--seed", type=int, default=42)

    # TODO add support for binomial BMR (exclusive of real bmr)
    # single_gene_parser.add_argument("-l", "--length", type=int, default=10000)
    # single_gene_parser.add_argument("-m", "--mu", type=float, default=1e-6)


def add_simulate_create_parser(subparsers):
    """
    Adds the simulate subparser to the given subparsers.
    """
    simulate_create_parser = subparsers.add_parser(
        "create", help="Create simulation data"
    )
    simulate_create_subparsers = simulate_create_parser.add_subparsers(
        dest="type",
        required=True,
        help="Available simulation types (single, pair)",
    )
    add_simulate_create_single_gene_parser(simulate_create_subparsers)
    # add_simulate_crete_pair_gene_parser(simulate_create_subparsers) # TODO


def add_simulate_evaluate_single_gene_parser(subparsers):
    single_gene_parser = subparsers.add_parser(
        "single", help="Evaluate single gene simulations"
    )
    single_gene_parser.add_argument(
        "-p", "--params", required=True, help="Path to the parameters file"
    )
    single_gene_parser.add_argument(
        "-d", "--data", required=True, help="Path to the data file"
    )
    single_gene_parser.add_argument("-o", "--out", type=str, required=True)


def add_simulate_evaluate_parser(subparsers):
    simulate_evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate simulation data"
    )
    simulate_evaluate_subparsers = simulate_evaluate_parser.add_subparsers(
        dest="type",
        required=True,
        help="Available simulation types (single, pair)",
    )
    add_simulate_evaluate_single_gene_parser(simulate_evaluate_subparsers)
    # add_simulate_evaluate_pair_gene_parser(simulate_evaluate_subparsers) # TODO


def add_simulate_parser(subparsers):
    """
    Adds the simulate subparser to the given subparsers.
    """
    simulate_parser = subparsers.add_parser(
        "simulate", help="Run simulations for evaluation and benchmarking"
    )
    simulate_subparsers = simulate_parser.add_subparsers(
        dest="mode", required=True, help="Available simulation modes (create, evaluate)"
    )

    add_simulate_create_parser(simulate_subparsers)
    add_simulate_evaluate_parser(simulate_subparsers)


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
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_generate_parser(subparsers)
    add_identify_parser(subparsers)
    add_compare_parser(subparsers)
    add_merge_parser(subparsers)
    add_simulate_parser(subparsers)

    return parser

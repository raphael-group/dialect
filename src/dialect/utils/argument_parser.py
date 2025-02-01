"""TODO: Add docstring."""

from argparse import ArgumentParser, ArgumentTypeError, _SubParsersAction


def add_generate_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate BMR and count matrix",
    )
    generate_parser.add_argument(
        "-m",
        "--maf",
        required=True,
        help="Path to the input MAF file",
    )
    generate_parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Path to the output directory",
    )
    generate_parser.add_argument(
        "-r",
        "--reference",
        default="hg19",
        choices=["hg19", "hg38"],
        help="Reference genome (default: hg19)",
    )
    generate_parser.add_argument(
        "-t",
        "--threshold",
        default="1e-100",
        help="Threshold for generation of BMR distributions (default: 1e-100)",
    )


def add_identify_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    identify_parser = subparsers.add_parser(
        "identify",
        help="Run DIALECT to identify interactions",
    )
    identify_parser.add_argument(
        "-c",
        "--cnt",
        required=True,
        help="Path to the input count matrix file",
    )
    identify_parser.add_argument(
        "-b",
        "--bmr",
        required=True,
        help="Path to the BMR file",
    )
    identify_parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Path to the output directory",
    )
    identify_parser.add_argument(
        "-k",
        "--top_k",
        default=100,
        type=int,
        help="Number of genes to consider (default: 100 genes w/ highest count)",
    )
    identify_parser.add_argument(
        "-cb",
        "--cbase_stats",
        default=None,
        help="Path to the cbase results file",
    )


def add_compare_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run alternative methods",
    )
    compare_parser.add_argument(
        "-c",
        "--cnt",
        required=True,
        help="Path to the input count matrix file",
    )
    compare_parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Path to the output directory",
    )
    compare_parser.add_argument(
        "-k",
        "--top_k",
        default=100,
        type=int,
        help="Number of genes to consider (default: 100 genes w/ highest count)",
    )
    compare_parser.add_argument(
        "-g",
        "--gene_level",
        action="store_true",
        help="Run comparison methods on gene level features",
    )


def add_merge_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge DIALECT and alternative method results",
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
        "-o",
        "--out",
        required=True,
        help="Path to the output directory",
    )


def add_simulate_create_single_gene_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    single_gene_parser = subparsers.add_parser(
        "single",
        help="Create single gene simulations",
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
    single_gene_parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
    )
    single_gene_parser.add_argument(
        "-ns",
        "--num_simulations",
        type=int,
        default=2500,
    )
    single_gene_parser.add_argument("-l", "--length", type=int, default=10000)
    single_gene_parser.add_argument("-m", "--mu", type=float, default=1e-6)
    single_gene_parser.add_argument("-o", "--out", type=str, required=True)
    single_gene_parser.add_argument("-s", "--seed", type=int, default=42)

def add_simulate_create_pair_gene_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    pair_gene_parser = subparsers.add_parser(
        "pair",
        help="Create pairwise gene simulations",
    )
    pair_gene_parser.add_argument("-t10", "--tau_10", required=True, type=float)
    pair_gene_parser.add_argument("-t01", "--tau_01", required=True, type=float)
    pair_gene_parser.add_argument("-t11", "--tau_11", required=True, type=float)
    pair_gene_parser.add_argument("-n", "--num_samples", type=int, default=1000)
    pair_gene_parser.add_argument("-ns", "--num_simulations", type=int, default=2500)
    pair_gene_parser.add_argument("-la", "--length_a", type=int, default=10000)
    pair_gene_parser.add_argument("-lb", "--length_b", type=int, default=10000)
    pair_gene_parser.add_argument("-ma", "--mu_a", type=float, default=1e-6)
    pair_gene_parser.add_argument("-mb", "--mu_b", type=float, default=1e-6)
    pair_gene_parser.add_argument("-o", "--out", type=str, required=True)
    pair_gene_parser.add_argument("-s", "--seed", type=int, default=42)


def add_simulate_create_matrix_gene_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    matrix_gene_parser = subparsers.add_parser(
        "matrix",
        help="Create matrix gene simulations",
    )
    matrix_gene_parser.add_argument(
        "-c",
        "--cnt_mtx",
        required=True,
        help="Path to the count matrix file",
    )
    matrix_gene_parser.add_argument(
        "-b",
        "--bmr_pmfs",
        required=True,
        help="Path to the BMR PMFs file",
    )
    matrix_gene_parser.add_argument(
        "-d",
        "--driver_genes",
        required=True,
        help="Path to the driver genes file",
    )
    matrix_gene_parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Path to the output directory",
    )
    matrix_gene_parser.add_argument(
        "-nlp",
        "--num_likely_passengers",
        type=int,
        default=100,
    )
    matrix_gene_parser.add_argument("-nme", "--num_me_pairs", required=True, type=int)
    matrix_gene_parser.add_argument("-nco", "--num_co_pairs", required=True, type=int)
    matrix_gene_parser.add_argument("-n", "--num_samples", type=int, default=1000)
    matrix_gene_parser.add_argument("-tl", "--tau_low", required=True, type=float)
    matrix_gene_parser.add_argument("-th", "--tau_high", required=True, type=float)
    matrix_gene_parser.add_argument("-s", "--seed", type=int, default=42)


def add_simulate_create_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    simulate_create_parser = subparsers.add_parser(
        "create",
        help="Create simulation data",
    )
    simulate_create_subparsers = simulate_create_parser.add_subparsers(
        dest="type",
        required=True,
        help="Available simulation types (single, pair)",
    )
    add_simulate_create_single_gene_parser(simulate_create_subparsers)
    add_simulate_create_pair_gene_parser(simulate_create_subparsers)
    add_simulate_create_matrix_gene_parser(simulate_create_subparsers)


def add_simulate_evaluate_single_gene_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    single_gene_parser = subparsers.add_parser(
        "single",
        help="Evaluate single gene simulations",
    )
    single_gene_parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to the parameters file",
    )
    single_gene_parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="Path to the data file",
    )
    single_gene_parser.add_argument("-o", "--out", type=str, required=True)

def add_simulate_evaluate_pair_gene_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    pair_gene_parser = subparsers.add_parser(
        "pair",
        help="Evaluate pair gene simulations",
    )
    pair_gene_parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to the parameters file",
    )
    pair_gene_parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="Path to the data file",
    )
    pair_gene_parser.add_argument("-o", "--out", type=str, required=True)

def add_simulate_evaluate_matrix_gene_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    matrix_gene_parser = subparsers.add_parser(
        "matrix",
        help="Evaluate pair gene simulations",
    )
    matrix_gene_parser.add_argument(
        "-r",
        "--results",
        required=True,
        help="Path to the results file",
    )
    matrix_gene_parser.add_argument(
        "-i",
        "--info",
        required=True,
        help="Path to the simulation info file",
    )
    matrix_gene_parser.add_argument("-ixn", "--ixn_type", required=True)
    matrix_gene_parser.add_argument("-o", "--out", type=str, required=True)


def add_simulate_evaluate_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    simulate_evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate simulation data",
    )
    simulate_evaluate_subparsers = simulate_evaluate_parser.add_subparsers(
        dest="type",
        required=True,
        help="Available simulation types (single, pair)",
    )
    add_simulate_evaluate_single_gene_parser(simulate_evaluate_subparsers)
    add_simulate_evaluate_pair_gene_parser(simulate_evaluate_subparsers)
    add_simulate_evaluate_matrix_gene_parser(simulate_evaluate_subparsers)


def add_simulate_parser(subparsers: _SubParsersAction) -> None:
    """TODO: Add docstring."""
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run simulations for evaluation and benchmarking",
    )
    simulate_subparsers = simulate_parser.add_subparsers(
        dest="mode",
        required=True,
        help="Available simulation modes (create, evaluate)",
    )

    add_simulate_create_parser(simulate_subparsers)
    add_simulate_evaluate_parser(simulate_subparsers)


def build_argument_parser() -> ArgumentParser:
    """TODO: Add docstring."""
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

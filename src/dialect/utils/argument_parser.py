from argparse import ArgumentParser


def build_argument_parser():
    """
    Creates and returns the argument parser for the command line interface.
    """
    parser = ArgumentParser(description="DIALECT")
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
        "-m", "--maf", required=True, help="Path to the input MAF file"
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

    # # Subparser for comparison analysis
    # compare_parser = subparsers.add_parser("compare", help="Run comparison analysis")
    # compare_parser.add_argument("method", choices=["fishers", "discover", "wesme"])
    # compare_parser.add_argument("cnt_mtx_fn", help="Path to the count matrix file")
    # compare_parser.add_argument("dout", help="Path to the output directory")
    # compare_parser.add_argument("--top_k", default=100, type=int)
    # compare_parser.add_argument(
    #     "--feature_level",
    #     default="mutation",
    #     choices=["gene", "mutation"],
    #     help="Feature level to use for comparison test",
    # )

    # # Subparser for workflows
    # workflow_parser = subparsers.add_parser("workflow", help="Run workflows")
    # workflow_parser.add_argument(
    #     "workflow_name",
    #     help="Name of the workflow to execute (e.g., generate, analyze, compare)",
    # )
    # workflow_parser.add_argument(
    #     "--snakefile", default="Snakefile", help="Path to the Snakefile"
    # )
    # workflow_parser.add_argument(
    #     "--configfile", default=None, help="Path to the config file"
    # )
    # workflow_parser.add_argument(
    #     "--cores", type=int, default=1, help="Number of cores for Snakemake"
    # )
    # workflow_parser.add_argument(
    #     "--dry-run", action="store_true", help="Perform a dry run without execution"
    # )
    # workflow_parser.add_argument(
    #     "--forceall", action="store_true", help="Force execution of all rules"
    # )

    return parser

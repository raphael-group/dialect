import os
import logging
from dialect.utils import (
    configure_logging,
    build_argument_parser,
    generate_bmr_and_counts,
    identify_pairwise_interactions,
    read_cbase_results_file,
    run_comparison_methods,
    merge_pairwise_interaction_results,
    create_single_gene_simulation,
)


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    logging.info("Running DIALECT Command Line Interface")
    logging.verbose(f"Arguments: {args}")
    logging.info(f"Command: {args.command}")

    if args.command == "generate":
        os.makedirs(args.out, exist_ok=True)
        generate_bmr_and_counts(maf=args.maf, out=args.out, reference=args.reference)

    elif args.command == "identify":
        os.makedirs(args.out, exist_ok=True)
        cbase_stats = read_cbase_results_file(args.cbase_stats)
        identify_pairwise_interactions(
            cnt_mtx=args.cnt,
            bmr_pmfs=args.bmr,
            out=args.out,
            k=args.top_k,
            cbase_stats=cbase_stats,
        )

    elif args.command == "compare":
        os.makedirs(args.out, exist_ok=True)
        run_comparison_methods(
            cnt_mtx=args.cnt,
            bmr_pmfs=args.bmr,
            out=args.out,
            k=args.top_k,
        )

    elif args.command == "merge":
        os.makedirs(args.out, exist_ok=True)
        merge_pairwise_interaction_results(
            dialect_results=args.dialect,
            alt_results=args.alt,
            out=args.out,
        )

    elif args.command == "simulate":
        if args.mode == "create":
            logging.info("Creating simulated data")
            if args.type == "single":
                create_single_gene_simulation(
                    pi=args.pi,
                    num_samples=args.num_samples,
                    num_simulations=args.num_simulations,
                    bmr_pmfs=args.bmr,
                    gene=args.gene,
                    out=args.out,
                    seed=args.seed,
                )
            else:  # args.type == "pair"
                logging.info("Simulating data for a pair of genes")
                raise NotImplementedError
        else:  # args.mode == "evaluate"
            logging.info("Evaluating methods on simulated data")
            if args.type == "single":
                logging.info("Evaluating methods on simulated data for a single gene")
                raise NotImplementedError
            else:  # args.type == "pair"
                logging.info("Evaluating methods on simulated data for a pair of genes")
                raise NotImplementedError

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

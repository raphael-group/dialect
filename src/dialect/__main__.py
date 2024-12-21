import os
import logging
from dialect.utils import (
    configure_logging,
    build_argument_parser,
    generate_bmr_and_counts,
    identify_pairwise_interactions,
    read_cbase_results_file,
)

# TODO: Create essential and verbose logging info for all methods


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    logging.info("Running DIALECT CLI")
    logging.info(f"Arguments: {args}")
    logging.info(f"Command: {args.command}")

    if args.command == "generate":
        os.makedirs(args.out, exist_ok=True)  # create output directory if nonexistent
        generate_bmr_and_counts(maf=args.maf, out=args.out, reference=args.reference)

    elif args.command == "identify":
        os.makedirs(args.out, exist_ok=True)  # create output directory if nonexistent
        cbase_stats = read_cbase_results_file(args)
        identify_pairwise_interactions(
            cnt_mtx=args.cnt,
            bmr_pmfs=args.bmr,
            out=args.out,
            k=args.top_k,
            cbase_stats=cbase_stats,
        )


if __name__ == "__main__":
    main()

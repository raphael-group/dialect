import os
import logging
import pandas as pd
from dialect.utils.logger import configure_logging

from dialect.utils.argument_parser import (
    build_argument_parser,
)  # TODO: export function from argument_parser.py

from dialect.utils.generate import (
    generate_bmr_and_counts,
)  # TODO: export function from generate.py

from dialect.utils.identify import (
    identify_pairwise_interactions,
)  # TODO: export function from identify.py

configure_logging()


# TODO: add option for verbose logging
def main():
    logging.info("Running DIALECT CLI")

    parser = build_argument_parser()
    args = parser.parse_args()

    logging.info(f"Arguments: {args}")
    logging.info(f"Command: {args.command}")

    if args.command == "generate":
        os.makedirs(args.out, exist_ok=True)  # create output directory if nonexistent
        generate_bmr_and_counts(maf=args.maf, out=args.out, reference=args.reference)

    elif args.command == "identify":
        os.makedirs(args.out, exist_ok=True)  # create output directory if nonexistent
        # TODO: consider moving below code to read cbase stats elsewhere
        cbase_stats = None
        if not args.cbase_stats is None:
            logging.info(f"Reading CBaSE q-values file: {args.cbase_stats}")
            cbase_stats = pd.read_csv(args.cbase_stats, sep="\t", skiprows=1)
        identify_pairwise_interactions(
            cnt_mtx=args.cnt,
            bmr_pmfs=args.bmr,
            out=args.out,
            k=args.top_k,
            cbase_results=cbase_stats,
        )


if __name__ == "__main__":
    main()

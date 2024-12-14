import os
import logging
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
        identify_pairwise_interactions(
            cnt_mtx=args.cnt, bmr_pmfs=args.bmr, out=args.out, k=args.top_k
        )
        pass


if __name__ == "__main__":
    main()

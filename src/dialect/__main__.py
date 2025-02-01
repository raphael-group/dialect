"""TODO: Add docstring."""

from argparse import Namespace
from pathlib import Path

from dialect.utils import (
    build_argument_parser,
    configure_logging,
    create_matrix_simulation,
    create_pair_gene_simulation,
    create_single_gene_simulation,
    evaluate_matrix_simulation,
    evaluate_pair_gene_simulation,
    evaluate_single_gene_simulation,
    generate_bmr_and_counts,
    identify_pairwise_interactions,
    merge_pairwise_interaction_results,
    read_cbase_results_file,
    run_comparison_methods,
)


def _handle_simulate_command(args: Namespace) -> None:
    if args.mode == "create" and args.type == "single":
        create_single_gene_simulation(
            pi=args.pi,
            num_samples=args.num_samples,
            num_simulations=args.num_simulations,
            length=args.length,
            mu=args.mu,
            out=args.out,
            seed=args.seed,
        )
    elif args.mode == "evaluate" and args.type == "single":
        evaluate_single_gene_simulation(
            params=args.params,
            data=args.data,
            out=args.out,
        )
    elif args.mode == "create" and args.type == "pair":
        create_pair_gene_simulation(
            tau_10=args.tau_10,
            tau_01=args.tau_01,
            tau_11=args.tau_11,
            num_samples=args.num_samples,
            num_simulations=args.num_simulations,
            length_a=args.length_a,
            mu_a=args.mu_a,
            length_b=args.length_b,
            mu_b=args.mu_b,
            out=args.out,
            seed=args.seed,
        )
    elif args.mode == "evaluate" and args.type == "pair":
        evaluate_pair_gene_simulation(
            params=args.params,
            data=args.data,
            out=args.out,
        )
    elif args.mode == "create" and args.type == "matrix":
        create_matrix_simulation(
            cnt_mtx_fn=args.cnt_mtx,
            bmr_pmfs_fn=args.bmr_pmfs,
            driver_genes_fn=args.driver_genes,
            dout=args.out,
            num_likely_passengers=args.num_likely_passengers,
            num_me_pairs=args.num_me_pairs,
            num_co_pairs=args.num_co_pairs,
            num_samples=args.num_samples,
            tau_uv_low=args.tau_low,
            tau_uv_high=args.tau_high,
            seed=args.seed,
        )
    elif args.mode == "evaluate" and args.type == "matrix":
        evaluate_matrix_simulation(
            results_fn=args.results,
            simulation_info_fn=args.info,
            dout=args.out,
            ixn_type=args.ixn_type,
        )

def main() -> None:
    """Run main entry point for the DIALECT CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.command == "generate":
        dout = Path(args.out)
        dout.mkdir(parents=True, exist_ok=True)
        generate_bmr_and_counts(
            maf=args.maf,
            out=args.out,
            reference=args.reference,
            threshold=args.threshold,
        )

    elif args.command == "identify":
        dout = Path(args.out)
        dout.mkdir(parents=True, exist_ok=True)
        cbase_stats = read_cbase_results_file(args.cbase_stats)
        identify_pairwise_interactions(
            cnt_mtx=args.cnt,
            bmr_pmfs=args.bmr,
            out=args.out,
            k=args.top_k,
            cbase_stats=cbase_stats,
        )

    elif args.command == "compare":
        dout = Path(args.out)
        dout.mkdir(parents=True, exist_ok=True)
        run_comparison_methods(
            cnt_mtx=args.cnt,
            out=args.out,
            k=args.top_k,
            gene_level=args.gene_level,
        )

    elif args.command == "merge":
        dout = Path(args.out)
        dout.mkdir(parents=True, exist_ok=True)
        merge_pairwise_interaction_results(
            dialect_results=args.dialect,
            alt_results=args.alt,
            out=args.out,
        )

    elif args.command == "simulate":
        _handle_simulate_command(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

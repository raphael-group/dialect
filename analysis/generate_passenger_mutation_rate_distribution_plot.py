"""TODO: Add docstring."""

from pathlib import Path

from dialect.models.gene import Gene
from dialect.utils.argument_parser import build_analysis_argument_parser
from dialect.utils.helpers import load_bmr_pmfs
from dialect.utils.plotting import (
    draw_all_subtypes_background_mutation_distribution,
    draw_single_subtype_background_mutation_distribution,
)


def get_expected_background_mutations(results_dir: Path, subtypes: list) -> dict:
    """TODO: Add docstring."""
    subtype_to_exp_bkgd_muts = {}
    for subtype in subtypes:
        subtype_dir = results_dir / subtype
        bmr_pmfs_fn = subtype_dir / "bmr_pmfs.csv"
        bmr_dict = load_bmr_pmfs(bmr_pmfs_fn)
        gene_exp_bkgd_muts = [
            Gene(
                name=gene,
                samples=None,
                counts=None,
                bmr_pmf=bmr_dict[gene],
            ).calculate_expected_mutations()
            for gene in bmr_dict
        ]
        subtype_to_exp_bkgd_muts[subtype] = gene_exp_bkgd_muts
    return subtype_to_exp_bkgd_muts


def get_all_subtype_exp_bkgd_muts(results_dir: Path) -> list:
    """TODO: Add docstring."""
    exp_bkgd_muts = []
    for subtype_dir in results_dir.iterdir():
        bmr_pmfs_fn = subtype_dir / "bmr_pmfs.csv"
        if not bmr_pmfs_fn.exists():
            continue
        bmr_dict = load_bmr_pmfs(bmr_pmfs_fn)
        gene_exp_bkgd_muts = [
            Gene(
                name=gene,
                samples=None,
                counts=None,
                bmr_pmf=bmr_dict[gene],
            ).calculate_expected_mutations()
            for gene in bmr_dict
        ]
        exp_bkgd_muts.extend(gene_exp_bkgd_muts)
    return exp_bkgd_muts

def main() -> None:
    """TODO: Add docstring."""
    parser = build_analysis_argument_parser(
        add_subtypes=True,
    )
    args = parser.parse_args()
    subtype_to_exp_bkgd_muts = get_expected_background_mutations(
        args.results_dir,
        args.subtypes.split(","),
    )
    for subtype, exp_bkgd_muts in subtype_to_exp_bkgd_muts.items():
       draw_single_subtype_background_mutation_distribution(
            subtype,
            exp_bkgd_muts,
            out_fn=args.out_dir / f"{subtype}_bkgd_mutation_distribution",
       )
    all_subtype_exp_bkgd_muts = get_all_subtype_exp_bkgd_muts(args.results_dir)
    draw_all_subtypes_background_mutation_distribution(
        all_subtype_exp_bkgd_muts,
        out_fn=args.out_dir / "all_subtypes_bkgd_mutation_distribution",
    )

if __name__ == "__main__":
    main()

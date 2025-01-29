"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #
from pathlib import Path

import matplotlib.pyplot as plt
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.helpers import initialize_gene_objects, load_cnt_mtx_and_bmr_pmfs


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def estimate_pi_for_single_gene(gene: Gene) -> float:
    """TODO: Add docstring."""
    gene.estimate_pi_with_em_from_scratch()
    return gene.pi


def initialize_interactions_with_gene(main_gene: str, top_genes: list) -> list:
    """TODO: Add docstring."""
    return [
        Interaction(main_gene, gene)
        for gene in top_genes
        if gene.name != main_gene.name
    ]


def estimate_taus_for_interactions(interactions: list) -> None:
    """TODO: Add docstring."""
    for interaction in interactions:
        interaction.estimate_tau_with_em_from_scratch()


def get_file_path(file_description: str) -> str:
    """TODO: Add docstring."""
    while True:
        file_path = Path(input(f"Enter the path to the {file_description}: ").strip())
        if file_path.exists():
            return file_path


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    cnt_mtx_path = get_file_path("count matrix file")
    bmr_pmfs_path = get_file_path("BMR PMFs file")
    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)
    genes = list(initialize_gene_objects(cnt_df, bmr_dict).values())
    genes = sorted(genes, key=lambda x: sum(x.counts), reverse=True)
    top_genes = genes[:1000]

    output_folder = Path(input("Enter the output folder for the plots: ").strip())
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    while True:
        main_gene_name = input("Enter the name of the gene to analyze: ").strip()
        if main_gene_name.lower() == "exit":
            break

        main_gene = next((g for g in genes if g.name == main_gene_name), None)
        if main_gene is None:
            continue

        pi_value = estimate_pi_for_single_gene(
            main_gene,
        )
        interactions = initialize_interactions_with_gene(
            main_gene,
            top_genes,
        )
        estimate_taus_for_interactions(interactions)

        marginals = [
            interaction.tau_10 + interaction.tau_11 for interaction in interactions
        ]

        plt.figure(figsize=(10, 8))
        plt.hist(
            marginals,
            bins=30,
            alpha=0.7,
            label=r"$\tau_{10} + \tau_{11}$",
        )
        plt.axvline(
            pi_value,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=r"$\pi$",
        )
        plt.xlabel(
            r"Marginal Value ($\tau_{10} + \tau_{11}$)",
            fontsize=24,
        )
        plt.ylabel("Frequency", fontsize=24)
        plt.ylim(0, 1000)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()

        plot_filename = (
            Path(output_folder) / f"{main_gene_name}_marginal_distribution_plot.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()

"""
This script analyzes the distribution of driver mutation rates (pi) from single gene
estimation for a user-specified gene and the marginals (tau_10 + tau_11) obtained
from interactions between this gene and the top 1000 other genes ranked by mutation count.
"""

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
from dialect.models.interaction import Interaction
from dialect.utils.helpers import *
from dialect.utils.identify import *


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def estimate_pi_for_single_gene(gene):
    gene.estimate_pi_with_em_from_scratch()
    return gene.pi


def initialize_interactions_with_gene(main_gene, top_genes):
    interactions = []
    for gene in top_genes:
        if gene.name != main_gene.name:
            interactions.append(Interaction(main_gene, gene))
    return interactions


def estimate_taus_for_interactions(interactions):
    for interaction in interactions:
        interaction.estimate_tau_with_em_from_scratch()


def get_file_path(file_description):
    """
    Prompt the user for a file path and ensure the file exists.

    :param file_description: (str) Description of the file for the user prompt.
    :return: (str) Validated file path.
    """
    while True:
        file_path = input(f"Enter the path to the {file_description}: ").strip()
        if os.path.exists(file_path):
            return file_path
        print(f"File not found: {file_path}. Please enter a valid file path.")


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    cnt_mtx_path = get_file_path("count matrix file")
    bmr_pmfs_path = get_file_path("BMR PMFs file")
    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)
    genes = list(initialize_gene_objects(cnt_df, bmr_dict).values())
    genes = sorted(genes, key=lambda x: sum(x.counts), reverse=True)
    top_genes = genes[:1000]

    output_folder = input("Enter the output folder for the plots: ").strip()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("\nType 'exit' to quit the program at any time.")

    while True:
        main_gene_name = input("Enter the name of the gene to analyze: ").strip()
        if main_gene_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        main_gene = next((g for g in genes if g.name == main_gene_name), None)
        if main_gene is None:
            print(
                f"Gene '{main_gene_name}' not found in the count matrix. Please try again."
            )
            continue

        pi_value = estimate_pi_for_single_gene(main_gene)
        interactions = initialize_interactions_with_gene(main_gene, top_genes)
        estimate_taus_for_interactions(interactions)

        marginals = [
            interaction.tau_10 + interaction.tau_11 for interaction in interactions
        ]

        # Adjust font sizes and figure dimensions
        plt.figure(figsize=(10, 8))  # Larger figure size for better visibility
        plt.hist(marginals, bins=30, alpha=0.7, label=r"$\tau_{10} + \tau_{11}$")
        plt.axvline(
            pi_value, color="red", linestyle="dashed", linewidth=2, label=r"$\pi$"
        )
        plt.xlabel(
            r"Marginal Value ($\tau_{10} + \tau_{11}$)", fontsize=24
        )  # Larger font size
        plt.ylabel("Frequency", fontsize=24)  # Larger font size
        plt.ylim(0, 1000)  # Consistent Y-axis scale
        plt.xticks(fontsize=20)  # Larger x-tick labels
        plt.yticks(fontsize=20)  # Larger y-tick labels
        plt.legend(fontsize=20)  # Larger legend text
        plt.tight_layout()  # Adjust layout to prevent clipping

        # Save the plot
        plot_filename = os.path.join(
            output_folder, f"{main_gene_name}_marginal_distribution_plot.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved as: {plot_filename}")

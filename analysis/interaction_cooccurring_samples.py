"""
This script analyzes co-occurring samples between two genes using user-provided
gene pairs and count matrix data. User friendly prompts are provided to enter
the gene names and the script outputs the set of co-occurring samples between.
"""

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import logging
from dialect.utils.identify import load_cnt_mtx_and_bmr_pmfs
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction


# TODO: extract and refactor user interaction methods
# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def initialize_gene_objects(cnt_df, bmr_dict):
    """
    Create a dictionary mapping gene names to Gene objects.
    """
    gene_objects = {}
    for gene_name in cnt_df.columns:
        counts = cnt_df[gene_name].values
        bmr_pmf = {i: bmr_dict[gene_name][i] for i in range(len(bmr_dict[gene_name]))}
        gene_objects[gene_name] = Gene(
            name=gene_name, samples=cnt_df.index, counts=counts, bmr_pmf=bmr_pmf
        )
    logging.info(f"Initialized {len(gene_objects)} Gene objects.")
    return gene_objects


def get_cooccurring_samples(gene_a, gene_b):
    """
    Get the set of co-occurring samples for two genes.
    """
    interaction = Interaction(gene_a, gene_b)
    return interaction.get_set_of_cooccurring_samples()


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Analyze Co-occurring Samples Between Gene Pairs")

    # Prompt for count matrix and BMR PMFs file
    cnt_mtx_path = input("Enter the path to the count matrix file: ").strip()
    bmr_pmfs_path = input("Enter the path to the BMR PMFs file: ").strip()

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)

    # Initialize gene objects
    gene_objects = initialize_gene_objects(cnt_df, bmr_dict)

    print("\nType 'exit' to quit the program at any time.")
    while True:
        # Prompt for gene A
        gene_a_name = input("Enter the name of Gene A: ").strip()
        if gene_a_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        if gene_a_name not in gene_objects:
            print(f"Gene '{gene_a_name}' does not exist. Try again.")
            continue

        # Prompt for gene B
        gene_b_name = input("Enter the name of Gene B: ").strip()
        if gene_b_name.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        if gene_b_name not in gene_objects:
            print(f"Gene '{gene_b_name}' does not exist. Try again.")
            continue

        # Get Gene objects
        gene_a = gene_objects[gene_a_name]
        gene_b = gene_objects[gene_b_name]

        # Get co-occurring samples
        try:
            cooccurring_samples = get_cooccurring_samples(gene_a, gene_b)
            print(
                f"Co-occurring samples between {gene_a_name} and {gene_b_name}: {cooccurring_samples}"
            )
        except Exception as e:
            print(f"An error occurred while processing genes: {e}")

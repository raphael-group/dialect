"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #
import logging

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.helpers import initialize_gene_objects
from dialect.utils.identify import load_cnt_mtx_and_bmr_pmfs


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def get_cooccurring_samples(gene_a: Gene, gene_b: Gene) -> set:
    """TODO: Add docstring."""
    interaction = Interaction(gene_a, gene_b)
    return interaction.get_set_of_cooccurring_samples()


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    cnt_mtx_path = input("Enter the path to the count matrix file: ").strip()
    bmr_pmfs_path = input("Enter the path to the BMR PMFs file: ").strip()

    cnt_df, bmr_dict = load_cnt_mtx_and_bmr_pmfs(cnt_mtx_path, bmr_pmfs_path)
    gene_objects = initialize_gene_objects(cnt_df, bmr_dict)

    while True:
        gene_a_name = input("Enter the name of Gene A: ").strip()
        if gene_a_name.lower() == "exit":
            break
        if gene_a_name not in gene_objects:
            continue

        gene_b_name = input("Enter the name of Gene B: ").strip()
        if gene_b_name.lower() == "exit":
            break
        if gene_b_name not in gene_objects:
            continue

        gene_a = gene_objects[gene_a_name]
        gene_b = gene_objects[gene_b_name]
        cooccurring_samples = get_cooccurring_samples(gene_a, gene_b)
        logging.info(
            "Gene A: %s\nGene B: %s\nCo-occurring samples: %s",
            gene_a_name,
            gene_b_name,
            cooccurring_samples,
        )


if __name__ == "__main__":
    main()

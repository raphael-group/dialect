"""Analyze co-occurring samples between two genes."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #
import contextlib
import logging

from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.helpers import initialize_gene_objects
from dialect.utils.identify import load_cnt_mtx_and_bmr_pmfs


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def _get_cooccurring_samples_(gene_a: Gene, gene_b: Gene) -> set:
    """Get the set of co-occurring samples for two genes."""
    interaction = Interaction(gene_a, gene_b)
    return interaction.get_set_of_cooccurring_samples()


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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

        with contextlib.suppress(Exception):
            cooccurring_samples = _get_cooccurring_samples_(gene_a, gene_b)

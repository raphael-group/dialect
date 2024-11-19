import logging

from dialect.utils.helpers import *

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def identify_pairwise_interactions(maf, bmr, out, k):
    """
    Main function to identify pairwise interactions between genetic drivers in tumors using DIALECT.
    ! Not Yet Implemented

    @param maf: Path to the input MAF (Mutation Annotation Format) file containing mutation data.
    @param bmr: Path to the file with background mutation rate (BMR) distributions.
    @param out: Directory where outputs will be saved.
    @param k: Top k genes according to count of mutations will be used.
    """
    check_file_exists(maf)
    check_file_exists(bmr)
    if k <= 0:
        logging.error("k must be a positive integer")
        raise ValueError("k must be a positive integer")
    logging.info("Identifying pairwise interactions using DIALECT")
    logging.info("Functionality not yet implemented")

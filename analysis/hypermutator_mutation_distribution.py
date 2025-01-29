"""TODO: Add docstring."""

import os
from pathlib import Path

import pandas as pd

from dialect.utils.plotting import plot_sample_mutation_count_subtype_histograms

HIGH_AVG_MUT_FREQ_SUBTYPES = [
    "UCEC",
    "SKCM",
    "CRAD",
    "STAD",
    "LUAD",
    "LUSC",
]

RESULTS_DIR = "output/TOP_500_Genes"
FIGURES_DIR = "figures"


# ------------------------------------------------------------------------------------ #
#                                   HELPER FUNCTIONS                                   #
# ------------------------------------------------------------------------------------ #
def compute_average_across_others(results_dir: str, exclude_list: list) -> float:
    """TODO: Add docstring."""
    all_dirs = os.listdir(results_dir)
    all_other_samples = []

    for d in all_dirs:
        full_path = Path(results_dir) / d
        if not full_path.is_dir():
            continue
        if d in exclude_list:
            continue

        count_csv = full_path / "count_matrix.csv"
        if count_csv.exists():
            cnt_df = pd.read_csv(count_csv, index_col=0)
            sample_sums = cnt_df.sum(axis=1)
            all_other_samples.append(sample_sums)

    combined_sums = pd.concat(all_other_samples)
    return combined_sums.mean()


def get_mut_count_per_sample(results_dir: str, subtype: str) -> pd.Series:
    """TODO: Add docstring."""
    filepath = Path(results_dir) / subtype / "count_matrix.csv"
    res_df = pd.read_csv(filepath, index_col=0)
    return res_df.sum(axis=1)


# ------------------------------------------------------------------------------------ #
#                                     MAIN FUNCTION                                    #
# ------------------------------------------------------------------------------------ #
def main() -> None:
    """TODO: Add docstring."""
    fout = Path(FIGURES_DIR) / "avg_mut_freq_histograms.svg"
    avg_across_others = compute_average_across_others(
        RESULTS_DIR,
        HIGH_AVG_MUT_FREQ_SUBTYPES,
    )
    subtype_sample_mut_counts = {}
    for subtype in HIGH_AVG_MUT_FREQ_SUBTYPES:
        mut_count_per_sample = get_mut_count_per_sample(RESULTS_DIR, subtype)
        subtype_sample_mut_counts[subtype] = mut_count_per_sample
    plot_sample_mutation_count_subtype_histograms(
        subtype_sample_mut_counts,
        avg_across_others,
        fout,
    )


if __name__ == "__main__":
    main()

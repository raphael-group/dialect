import os
import pandas as pd

from dialect.utils.plotting import plot_sample_mutation_count_subtype_histograms


HIGH_AVG_MUT_FREQ_SUBTYPES = [
    "UCEC",  # hypermutators > 20k
    "SKCM",  # hypermutators > 20k
    "CRAD",  # hypermutators > 5k
    "STAD",  # hypermutators > 5k
    "LUAD",  # hypermutators < 2k
    "LUSC",  # hypermutators < 2k
]

RESULTS_DIR = "output/TOP_500_Genes"
FIGURES_DIR = "figures"


# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #
def compute_average_across_others(results_dir, exclude_list):
    all_dirs = os.listdir(results_dir)
    all_other_samples = []

    for d in all_dirs:
        full_path = os.path.join(results_dir, d)
        if not os.path.isdir(full_path):
            continue
        if d in exclude_list:
            continue

        count_csv = os.path.join(full_path, "count_matrix.csv")
        if os.path.exists(count_csv):
            df = pd.read_csv(count_csv, index_col=0)
            sample_sums = df.sum(axis=1)
            all_other_samples.append(sample_sums)

    combined_sums = pd.concat(all_other_samples)
    return combined_sums.mean()


def get_mut_count_per_sample(results_dir, subtype):
    filepath = os.path.join(results_dir, subtype, "count_matrix.csv")
    df = pd.read_csv(filepath, index_col=0)
    mut_series = df.sum(axis=1)
    return mut_series


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    fout = os.path.join(FIGURES_DIR, "avg_mut_freq_histograms.svg")
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

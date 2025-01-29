"""TODO: Add docstring."""

import os
from pathlib import Path

import pandas as pd
from dialect.utils.plotting import (
    plot_cbase_driver_and_passenger_mutation_counts,
    plot_cbase_driver_decoy_gene_fractions,
    plot_cbase_top_decoy_genes_upset,
)

SINGLE_GENE_RESULTS_DIR = "single_gene_results"
DECOY_GENES_DIR = "data/decoy_genes"
DRIVER_FN = "data/references/OncoKB_Cancer_Gene_List.tsv"

def main() -> None:
    """TODO: Add docstring."""
    driver_df = pd.read_csv(DRIVER_FN, sep="\t", index_col=0)
    drivers = set(driver_df.index + "_M") | set(driver_df.index + "_N")

    high_ranked_decoy_freqs = {}
    subtype_decoy_gene_fractions = {}
    subtype_to_high_ranked_decoys = {}
    for file_name in os.listdir(SINGLE_GENE_RESULTS_DIR):
        subtype = Path(file_name).stem
        fpath = Path(SINGLE_GENE_RESULTS_DIR) / f"{subtype}.csv"

        decoy_genes_fn = Path(DECOY_GENES_DIR) / f"{subtype}_decoy_genes.txt"
        subtype_decoy_genes = set(
            pd.read_csv(decoy_genes_fn, header=None, names=["Gene"])["Gene"],
        )

        subtype_res_df = pd.read_csv(fpath)
        plot_cbase_driver_and_passenger_mutation_counts(
            subtype_decoy_genes,
            drivers,
            subtype_res_df,
            subtype,
        )

        subtype_cbase_drivers = set(
            subtype_res_df.sort_values(
                by="CBaSE Pos. Sel. Phi",
                ascending=False,
            )["Gene Name"].head(50),
        )
        high_ranked_decoys = subtype_cbase_drivers.intersection(
            subtype_decoy_genes,
        )
        subtype_decoy_gene_fractions[subtype] = len(high_ranked_decoys) / 50.0
        subtype_to_high_ranked_decoys[subtype] = high_ranked_decoys
        for gene in high_ranked_decoys:
            if gene not in high_ranked_decoy_freqs:
                high_ranked_decoy_freqs[gene] = 0
            high_ranked_decoy_freqs[gene] += 1

    plot_cbase_driver_decoy_gene_fractions(
        subtype_decoy_gene_fractions,
        fout="figures/cbase_decoy_fractions_barplot.svg",
    )
    plot_cbase_top_decoy_genes_upset(
        subtype_to_high_ranked_decoys,
        high_ranked_decoy_freqs,
        top_n=6,
        fout="figures/cbase_upset_plot_top_likely_passengers.svg",
    )

if __name__ == "__main__":
    main()

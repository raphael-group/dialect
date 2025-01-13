import os
import logging
import pandas as pd


def get_decoy_gene_fraction_across_methods(ixn_res_df, decoy_genes, k):
    if ixn_res_df.empty:
        raise ValueError("Input DataFrame is empty")

    methods = {
        "DIALECT": "Rho",
        "DISCOVER": "Discover ME Q-Val",
        "Fisher's Exact Test": "Fisher's ME Q-Val",
        "MEGSA": "MEGSA S-Score (LRT)",
        "WeSME": "WeSME Q-Val",
    }

    fractions = {}
    for method, column in methods.items():
        top_ranking = ixn_res_df.sort_values(
            column, ascending=column != "MEGSA S-Score (LRT)"
        ).head(k)

        pairs_with_at_least_one_decoy_gene = (
            top_ranking["Gene A"].isin(decoy_genes)
            | top_ranking["Gene B"].isin(decoy_genes)
        ).sum()
        fractions[method] = pairs_with_at_least_one_decoy_gene / top_ranking.shape[0]

    return fractions


if __name__ == "__main__":
    K = 50
    RESULTS_DIR = "output/TOP_500_Genes"
    DECOY_GENES_DIR = "data/decoy_genes"
    OUTPUT_DIR = "output/RESULTS"

    SUBTYPES = os.listdir(RESULTS_DIR)
    subtype_decoy_gene_fractions = {}
    for subtype in SUBTYPES:
        RES_FN = os.path.join(RESULTS_DIR, subtype, "complete_pairwise_ixn_results.csv")
        DECOY_GENES_FN = os.path.join(DECOY_GENES_DIR, f"{subtype}_decoy_genes.txt")
        if not os.path.exists(RES_FN) or not os.path.exists(DECOY_GENES_FN):
            logging.info(f"Skipping {subtype} since input files not found")
            continue
        ixn_res_df = pd.read_csv(RES_FN)
        decoy_genes = set(
            pd.read_csv(DECOY_GENES_FN, header=None, names=["Gene"])["Gene"]
        )
        subtype_decoy_gene_fractions[subtype] = get_decoy_gene_fraction_across_methods(
            ixn_res_df,
            decoy_genes,
            k=K,
        )
    gene_fraction_data = [
        {"Subtype": subtype, "Method": method, "Fraction": fraction}
        for subtype, fractions in subtype_decoy_gene_fractions.items()
        for method, fraction in fractions.items()
    ]
    df = pd.DataFrame(gene_fraction_data)
    df.to_csv(f"{OUTPUT_DIR}/decoy_gene_fractions_by_method.csv", index=False)

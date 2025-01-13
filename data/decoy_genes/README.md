# Decoy Genes

This directory contains lists of decoy genes for various cancer subtypes. Decoy genes are highly mutated genes that are often mistakenly identified as drivers. Here, they are excluded based on their absence from the driver gene list alongside having a high mutation frequency.

## **Generation Process**
Decoy gene lists are created using the `scripts/identify_decoy_genes.py` script:

1. **Inputs**:
   - **Count Matrix File**: Gene mutation counts across samples.
   - **Driver Genes File**: Reference list of driver genes (e.g., from OncoKB).

2. **Steps**:
   - Rank genes by mutation frequency (top `k`, default: 100).
   - Exclude known driver genes.

3. **Output**:
   - A file for each subtype (e.g., `LUAD_decoy_genes.txt`).

## **Usage**
Decoy gene lists help identify passenger genes with high mutation frequencies for further analysis. They are used in the result evaluation pipeline to determine how often DIALECT alongside other methods identify ME/CO pairs that are comprised of decoy genes. Such pairs are unlikely to be true dependencies and this is used to qualify the top ranking pairs identified across methods. 

### Example Command:
```bash
python scripts/identify_decoy_genes.py \
    -c data/count_matrix/LUAD_counts.csv \
    -d data/references/OncoKB_Cancer_Gene_List.tsv \
    -k 100 \
    -s LUAD \
    -o data/decoy_genes
```

## **Citation**
If using these files, please cite:
- Suehnholz et al., *Cancer Discovery*, 2023.
- Chakravarty et al., *JCO Precision Oncology*, 2017.
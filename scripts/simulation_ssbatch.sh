#!/usr/bin/env bash
#SBATCH -N 1                   # Number of nodes
#SBATCH --ntasks-per-node=4    # CPUs per node
#SBATCH -t 24:00:00            # Walltime
#SBATCH --mem=32GB             # Memory
#SBATCH --job-name=matrix_sim  # Job name

source /n/fs/ragr-research/users/ashuaibi/anaconda3/etc/profile.d/conda.sh
conda activate dialect

SUBTYPE="$1"
NUM_SAMPLES="$2"
NUM_ME_PAIRS="$3"
NUM_CO_PAIRS="$4"
NUM_LIKELY_PASSENGERS="$5"
IXN_STRENGTH="$6"

DRIVER_GENES_FILE="data/references/OncoKB_Cancer_Gene_List.tsv"
COUNT_MATRIX="output/TOP_500_Genes/${SUBTYPE}/count_matrix.csv"
BMR_PMFS="output/TOP_500_Genes/${SUBTYPE}/bmr_pmfs.csv"
OUTPUT_BASE="output/SIMULATIONS"
OUTPUT_DIR="${OUTPUT_BASE}/${SUBTYPE}/NS${NUM_SAMPLES}/${NUM_ME_PAIRS}ME_${NUM_CO_PAIRS}CO_${NUM_LIKELY_PASSENGERS}P_${IXN_STRENGTH}IXN"

mkdir -p "$OUTPUT_DIR"

echo "Starting matrix simulation pipeline..."

dialect simulate create matrix \
  -c "$COUNT_MATRIX" \
  -b "$BMR_PMFS" \
  -d "$DRIVER_GENES_FILE" \
  -o "$OUTPUT_DIR" \
  -nlp "$NUM_LIKELY_PASSENGERS" \
  -nme "$NUM_ME_PAIRS" \
  -nco "$NUM_CO_PAIRS" \
  -n "$NUM_SAMPLES" \
  -ixn "$IXN_STRENGTH"

dialect identify \
  -c "$OUTPUT_DIR/count_matrix.csv" \
  -b "$BMR_PMFS" \
  -o "$OUTPUT_DIR" &

dialect compare \
  -c "$OUTPUT_DIR/count_matrix.csv" \
  -o "$OUTPUT_DIR" &

wait

dialect merge \
  -d "$OUTPUT_DIR/pairwise_interaction_results.csv" \
  -a "$OUTPUT_DIR/comparison_interaction_results.csv" \
  -o "$OUTPUT_DIR"

dialect simulate evaluate matrix \
  -r "$OUTPUT_DIR/complete_pairwise_ixn_results.csv" \
  -i "$OUTPUT_DIR/matrix_simulation_info.json" \
  -ixn "ME" \
  -o "$OUTPUT_DIR" &

dialect simulate evaluate matrix \
  -r "$OUTPUT_DIR/complete_pairwise_ixn_results.csv" \
  -i "$OUTPUT_DIR/matrix_simulation_info.json" \
  -ixn "CO" \
  -o "$OUTPUT_DIR" &

wait

echo "Matrix simulation pipeline completed."

#!/usr/bin/env bash
#SBATCH -N 1                   # Number of nodes
#SBATCH --ntasks-per-node=4    # CPUs per node
#SBATCH -t 120:00:00            # Walltime
#SBATCH --mem=25GB             # Memory
#SBATCH --job-name=matrix_sim  # Job name

source /n/fs/ragr-research/users/ashuaibi/anaconda3/etc/profile.d/conda.sh
conda activate dialect

SUBTYPE="$1"
NUM_SAMPLES="$2"
NUM_ME_PAIRS="$3"
NUM_CO_PAIRS="$4"
NUM_LIKELY_PASSENGERS="$5"
TAU_LOW="$6"
TAU_HIGH="$7"
RUN="$8"
DRIVER_PROP="$9"
NUM_RUNS="${10}"

DRIVER_GENES_FILE="data/references/OncoKB_Cancer_Gene_List.tsv"
COUNT_MATRIX="output/TOP_500_Genes/${SUBTYPE}/count_matrix.csv"
BMR_PMFS="output/TOP_500_Genes/${SUBTYPE}/bmr_pmfs.csv"
OUTPUT_BASE="output/SIMULATIONS"
SUBTYPE_OUTPUT_DIR="${OUTPUT_BASE}/${SUBTYPE}"
OUTPUT_DIR="${SUBTYPE_OUTPUT_DIR}/NS${NUM_SAMPLES}/${NUM_ME_PAIRS}ME_${NUM_CO_PAIRS}CO_${NUM_LIKELY_PASSENGERS}P/${DRIVER_PROP}DP/${TAU_LOW}TL_${TAU_HIGH}TH"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/R${RUN}"
TOP_K_GENES=500

mkdir -p "$RUN_OUTPUT_DIR"

echo "Starting matrix simulation pipeline..."

dialect simulate create matrix \
  -c "$COUNT_MATRIX" \
  -b "$BMR_PMFS" \
  -d "$DRIVER_GENES_FILE" \
  -o "$RUN_OUTPUT_DIR" \
  -nlp "$NUM_LIKELY_PASSENGERS" \
  -nme "$NUM_ME_PAIRS" \
  -nco "$NUM_CO_PAIRS" \
  -n "$NUM_SAMPLES" \
  -tl "$TAU_LOW" \
  -th "$TAU_HIGH" \
  -dp "$DRIVER_PROP"

dialect identify \
  -c "$RUN_OUTPUT_DIR/count_matrix.csv" \
  -b "$BMR_PMFS" \
  -k "$TOP_K_GENES" \
  -o "$RUN_OUTPUT_DIR" &

dialect compare \
  -c "$RUN_OUTPUT_DIR/count_matrix.csv" \
  -k "$TOP_K_GENES" \
  -o "$RUN_OUTPUT_DIR" &

wait

dialect merge \
  -d "$RUN_OUTPUT_DIR/pairwise_interaction_results.csv" \
  -a "$RUN_OUTPUT_DIR/comparison_interaction_results.csv" \
  -o "$RUN_OUTPUT_DIR"

# dialect simulate evaluate matrix \
#   -r "$OUTPUT_DIR" \
#   -ixn "ME" \
#   -n "$NUM_RUNS" \
#   -o "$OUTPUT_DIR" &

# dialect simulate evaluate matrix \
#   -r "$OUTPUT_DIR" \
#   -ixn "CO" \
#   -n "$NUM_RUNS" \
#   -o "$OUTPUT_DIR" &

# wait

echo "Matrix simulation pipeline completed."

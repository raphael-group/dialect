#!/usr/bin/env bash
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=8   ## CPUs per node
#SBATCH -t 48:00:00           ## Walltime
#SBATCH --mem=50GB            ## Memory
#SBATCH --job-name=dialect_job ## Job name

source /n/fs/ragr-research/users/ashuaibi/anaconda3/etc/profile.d/conda.sh
conda activate dialect

# Get arguments
MAF_FILE=${1}  # MAF file path
K=${2}         # Number of top genes
DOUT=${3}      # Output directory
STEPS=${4}     # Steps to execute

# Derived variables
SUBTYPE_NAME=$(basename "${MAF_FILE}" .maf) # Extract subtype name from MAF file
OUTPUT_DIR=${DOUT}/${SUBTYPE_NAME}          # Set output directory

# Prepare environment
mkdir -p "${OUTPUT_DIR}"

# Install DISCOVER
ORIGINAL_DIR=$(pwd)
DISCOVER_DIR=${ORIGINAL_DIR}/external/DISCOVER
cd "${DISCOVER_DIR}"
python setup.py install --user  # Use --user to install DISCOVER in user space
cd "${ORIGINAL_DIR}"

# Run DIAlEcT steps based on specified steps
IFS=',' read -r -a STEPS_ARRAY <<< "$STEPS"

for STEP in "${STEPS_ARRAY[@]}"; do
    case $STEP in
        generate)
            dialect generate -m "${MAF_FILE}" -o "${OUTPUT_DIR}"
            ;;
        identify)
            dialect identify -c "${OUTPUT_DIR}/count_matrix.csv" -b "${OUTPUT_DIR}/bmr_pmfs.csv" -o "${OUTPUT_DIR}" -k "${K}" -cb "${OUTPUT_DIR}/CBaSE_output/q_values.txt"
            ;;
        compare)
            dialect compare -c "${OUTPUT_DIR}/count_matrix.csv" -b "${OUTPUT_DIR}/bmr_pmfs.csv" -o "${OUTPUT_DIR}" -k "${K}"
            ;;
        merge)
            dialect merge -d "${OUTPUT_DIR}/pairwise_interaction_results.csv" -a "${OUTPUT_DIR}/comparison_interaction_results.csv" -o "${OUTPUT_DIR}"
            ;;
        *)
            echo "Warning: Unknown step '$STEP'. Skipping."
            ;;
    esac
done

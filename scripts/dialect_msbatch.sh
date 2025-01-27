#!/usr/bin/env bash

# Argument parser for directories, top genes, and steps
usage() {
    echo "Usage: $0 -d <data_directory> -o <output_directory> -k <number_of_top_genes> -s <steps>"
    echo "Steps: A comma-separated list of steps to run: generate,identify,compare,merge"
    exit 1
}

while getopts ":d:o:k:s:" opt; do
    case ${opt} in
        d )
            DATA_DIR=$OPTARG
            ;;
        o )
            DOUT=$OPTARG
            ;;
        k )
            TOP_GENES=$OPTARG
            ;;
        s )
            STEPS=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Ensure required arguments are provided
if [ -z "$DATA_DIR" ] || [ -z "$DOUT" ] || [ -z "$TOP_GENES" ] || [ -z "$STEPS" ]; then
    echo "Error: Data directory, output directory, number of top genes, and steps must be specified."
    usage
fi

# Validate directories
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

if [ ! -d "$DOUT" ]; then
    echo "Error: Output directory '$DOUT' does not exist. Creating it now."
    mkdir -p "$DOUT"
fi

# Define other variables
MAF_FILES=(${DATA_DIR}/*.maf) # Array of all MAF files in the directory
SCRIPT_DIR=$(dirname "$0") # Directory of the current script
SLURM_DIR=${DOUT}/slurm_logs

mkdir -p "${SLURM_DIR}"

# Process each MAF file
for MAF_FILE in "${MAF_FILES[@]}"; do
    if [ -f "$MAF_FILE" ]; then
        SUBTYPE_NAME=$(basename "${MAF_FILE}" .maf)
        SLURM_OUT_FN=${SLURM_DIR}/${SUBTYPE_NAME}.out

        # Submit job with the top genes and steps parameters
        sbatch -o "${SLURM_OUT_FN}" -e "${SLURM_OUT_FN}" "${SCRIPT_DIR}/dialect_ssbatch.sh" "${MAF_FILE}" "${TOP_GENES}" "${DOUT}" "${STEPS}"
        sleep 0.25 # Avoid overloading the scheduler
    else
        echo "Warning: No MAF files found in '$DATA_DIR'. Skipping."
    fi
done
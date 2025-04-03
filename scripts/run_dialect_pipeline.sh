#!/usr/bin/env bash

usage() {
    echo "Usage: $0 -f <maf_file> -o <output_directory> -k <number_of_top_genes> -s <steps>"
    echo "Steps: A comma-separated list of steps to run: generate,identify,compare,merge"
    exit 1
}

while getopts ":f:o:k:s:" opt; do
    case ${opt} in
    f)
        MAF_FILE=$OPTARG
        ;;
    o)
        DOUT=$OPTARG
        ;;
    k)
        TOP_GENES=$OPTARG
        ;;
    s)
        STEPS=$OPTARG
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        usage
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        ;;
    esac
done

if [ -z "$MAF_FILE" ] || [ -z "$DOUT" ] || [ -z "$TOP_GENES" ] || [ -z "$STEPS" ]; then
    echo "Error: MAF file, output directory, number of top genes, and steps must be specified."
    usage
fi

if [ ! -f "$MAF_FILE" ]; then
    echo "Error: MAF file '$MAF_FILE' does not exist."
    exit 1
fi

if [ ! -d "$DOUT" ]; then
    echo "Output directory '$DOUT' does not exist. Creating it now."
    mkdir -p "$DOUT"
fi

IFS=',' read -r -a STEPS_ARRAY <<<"$STEPS"

SUBTYPE_NAME=$(basename "${MAF_FILE}" .maf)
OUTPUT_DIR=${DOUT}/${SUBTYPE_NAME}
mkdir -p "${OUTPUT_DIR}"

for STEP in "${STEPS_ARRAY[@]}"; do
    case $STEP in
    generate)
        echo "Running generate for ${SUBTYPE_NAME}..."
        dialect generate -m "${MAF_FILE}" -o "${OUTPUT_DIR}"
        ;;
    identify)
        echo "Running identify for ${SUBTYPE_NAME}..."
        dialect identify -c "${OUTPUT_DIR}/count_matrix.csv" -b "${OUTPUT_DIR}/bmr_pmfs.csv" -o "${OUTPUT_DIR}" -k "${TOP_GENES}" -cb "${OUTPUT_DIR}/CBaSE_output/q_values.txt"
        ;;
    compare)
        echo "Running compare for ${SUBTYPE_NAME}..."
        dialect compare -c "${OUTPUT_DIR}/count_matrix.csv" -o "${OUTPUT_DIR}" -k "${TOP_GENES}"
        dialect compare -c "${OUTPUT_DIR}/gene_level_count_matrix.csv" -o "${OUTPUT_DIR}" -k "${TOP_GENES}" -g
        ;;
    merge)
        echo "Running merge for ${SUBTYPE_NAME}..."
        dialect merge -d "${OUTPUT_DIR}/pairwise_interaction_results.csv" -a "${OUTPUT_DIR}/comparison_pairwise_interaction_results.csv" -o "${OUTPUT_DIR}"
        ;;
    *)
        echo "Warning: Unknown step '$STEP'. Skipping."
        ;;
    esac
done

echo "All processes completed."

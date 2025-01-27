#!/bin/bash
DRIVER_GENES_FILE="data/references/OncoKB_Cancer_Gene_List.tsv"
OUTPUT_BASE="output/SIMULATIONS"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--subtype) SUBTYPE="$2"; shift ;;
        -n|--num_samples) NUM_SAMPLES="$2"; shift ;;
        -nme|--num_me_pairs) NUM_ME_PAIRS="$2"; shift ;;
        -nco|--num_co_pairs) NUM_CO_PAIRS="$2"; shift ;;
        -nlp|--num_likely_passengers) NUM_LIKELY_PASSENGERS="$2"; shift ;;
        -ixn|--ixn_strength) IXN_STRENGTH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$SUBTYPE" || -z "$NUM_SAMPLES" || -z "$NUM_ME_PAIRS" || -z "$NUM_CO_PAIRS" || -z "$NUM_LIKELY_PASSENGERS" || -z "$IXN_STRENGTH" ]]; then
    echo "Missing required arguments. Usage:"
    echo "./simulate_matrix.sh -s SUBTYPE -n NUM_SAMPLES -nme NUM_ME_PAIRS -nco NUM_CO_PAIRS -nlp NUM_LIKELY_PASSENGERS -ixn IXN_STRENGTH"
    exit 1
fi

COUNT_MATRIX="output/TOP_500_Genes/${SUBTYPE}/count_matrix.csv"
BMR_PMFS="output/TOP_500_Genes/${SUBTYPE}/bmr_pmfs.csv"

if [[ ! -f "$COUNT_MATRIX" ]]; then
    echo "Count matrix file not found: $COUNT_MATRIX"
    exit 1
fi

if [[ ! -f "$BMR_PMFS" ]]; then
    echo "BMR PMFs file not found: $BMR_PMFS"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE}/${SUBTYPE}/NS${NUM_SAMPLES}/${NUM_ME_PAIRS}ME_${NUM_CO_PAIRS}CO_${NUM_LIKELY_PASSENGERS}P_${IXN_STRENGTH}IXN"

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

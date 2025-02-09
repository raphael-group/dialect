#!/usr/bin/env bash

################################################################################
# Local Matrix Simulation Script
################################################################################

# ---------------------
# 1. Define Parameters
# ---------------------
SUBTYPES=("LAML")
NUM_SAMPLES_VALUES=(50)
NUM_ME_PAIRS_VALUES=(1)
NUM_CO_PAIRS_VALUES=(1)
LIKELY_PASSENGERS_VALUES=(6)
TAU_LOW_VALUES=(0.049)
TAU_HIGH_VALUES=(0.051)
DRIVER_PROPORTIONS=(1.0)
NUM_RUNS=1
TOP_K_GENES=200

# ---------------------
# 2. Define I/O Paths
# ---------------------

DRIVER_GENES_FILE="data/references/OncoKB_Cancer_Gene_List.tsv"
OUTPUT_BASE="output/simulations"

# -------------------------------
# 3. Loop Over Parameters and Run
# -------------------------------

for subtype in "${SUBTYPES[@]}"; do
  COUNT_MATRIX="output/tcga_pancan/${subtype}/count_matrix.csv"
  BMR_PMFS="output/tcga_pancan/${subtype}/bmr_pmfs.csv"

  if [[ ! -f "$COUNT_MATRIX" ]]; then
      echo "Count matrix file not found: $COUNT_MATRIX"
      continue
  fi

  if [[ ! -f "$BMR_PMFS" ]]; then
      echo "BMR PMFs file not found: $BMR_PMFS"
      continue
  fi

  for num_samples in "${NUM_SAMPLES_VALUES[@]}"; do
    for num_me_pairs in "${NUM_ME_PAIRS_VALUES[@]}"; do
      for num_co_pairs in "${NUM_CO_PAIRS_VALUES[@]}"; do
        for num_likely_psngrs in "${LIKELY_PASSENGERS_VALUES[@]}"; do
          for driver_prop in "${DRIVER_PROPORTIONS[@]}"; do

            for i in "${!TAU_LOW_VALUES[@]}"; do
              tau_low="${TAU_LOW_VALUES[i]}"
              tau_high="${TAU_HIGH_VALUES[i]}"

              for run in $(seq 1 $NUM_RUNS); do

                # 4. Create unique output directories per parameter set
                # -----------------------------------------------------
                SUBTYPE_OUTPUT_DIR="${OUTPUT_BASE}/${subtype}"
                PARAM_DIR="${SUBTYPE_OUTPUT_DIR}/NS${num_samples}/${num_me_pairs}ME_${num_co_pairs}CO_${num_likely_psngrs}P/${driver_prop}DP/${tau_low}TL_${tau_high}TH"
                RUN_OUTPUT_DIR="${PARAM_DIR}/R${run}"
                mkdir -p "${RUN_OUTPUT_DIR}"

                echo "================================================================="
                echo "Running simulation for:"
                echo "  Subtype:             ${subtype}"
                echo "  Num Samples:         ${num_samples}"
                echo "  ME Pairs:            ${num_me_pairs}"
                echo "  CO Pairs:            ${num_co_pairs}"
                echo "  Likely Passengers:   ${num_likely_psngrs}"
                echo "  Tau Low/High:        ${tau_low} / ${tau_high}"
                echo "  Driver Proportion:   ${driver_prop}"
                echo "  Run Index:           ${run}"
                echo "================================================================="

                # --------------------------------------------
                # 5. Simulate (dialect simulate create matrix)
                # --------------------------------------------
                dialect simulate create matrix \
                  -c "$COUNT_MATRIX" \
                  -b "$BMR_PMFS" \
                  -d "$DRIVER_GENES_FILE" \
                  -o "$RUN_OUTPUT_DIR" \
                  -nlp "$num_likely_psngrs" \
                  -nme "$num_me_pairs" \
                  -nco "$num_co_pairs" \
                  -n "$num_samples" \
                  -tl "$tau_low" \
                  -th "$tau_high" \
                  -dp "$driver_prop"

                # --------------------------------
                # 6. Identify & Compare in Parallel
                # --------------------------------
                dialect identify \
                  -c "${RUN_OUTPUT_DIR}/count_matrix.csv" \
                  -b "$BMR_PMFS" \
                  -k "$TOP_K_GENES" \
                  -o "$RUN_OUTPUT_DIR" &

                dialect compare \
                  -c "${RUN_OUTPUT_DIR}/count_matrix.csv" \
                  -k "$TOP_K_GENES" \
                  -o "$RUN_OUTPUT_DIR" &

                wait

                # ------------------------------------
                # 7. Merge pairwise and comparison data
                # ------------------------------------
                dialect merge \
                  -d "${RUN_OUTPUT_DIR}/pairwise_interaction_results.csv" \
                  -a "${RUN_OUTPUT_DIR}/comparison_interaction_results.csv" \
                  -o "${RUN_OUTPUT_DIR}"

              done

              # ----------------------------------------------------
              # 8. Evaluate the results (ME and CO) in parallel
              # ----------------------------------------------------

              # dialect simulate evaluate matrix \
              #   -r "${PARAM_DIR}" \
              #   -ixn "ME" \
              #   -n "$NUM_RUNS" \
              #   -o "${PARAM_DIR}" &
              # exit

              # dialect simulate evaluate matrix \
              #   -r "${PARAM_DIR}" \
              #   -ixn "CO" \
              #   -n "$NUM_RUNS" \
              #   -o "${PARAM_DIR}" &

              # wait

            done
          done
        done
      done
    done
  done
done

echo "All local matrix simulations completed."

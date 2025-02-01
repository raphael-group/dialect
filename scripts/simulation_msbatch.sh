#!/usr/bin/env bash

SUBTYPES=("UCEC" "LUAD" "BRCA")
NUM_SAMPLES_VALUES=(1000)
NUM_ME_PAIRS_VALUES=(10 15 20)
NUM_CO_PAIRS_VALUES=(0)
LIKELY_PASSENGERS_VALUES=(50 100 150)
TAU_LOW_VALUES=(0.05 0.10 0.15)
TAU_HIGH_VALUES=(0.15 0.20 0.25)
NUM_RUNS=5

OUTPUT_BASE="output/SIMULATIONS"
SLURM_LOG_DIR="${OUTPUT_BASE}/slurm_logs"
mkdir -p "${SLURM_LOG_DIR}"

SCRIPT_DIR="$(dirname "$0")"
SSBATCH_SCRIPT="${SCRIPT_DIR}/simulation_ssbatch.sh"

for subtype in "${SUBTYPES[@]}"; do
  for num_samples in "${NUM_SAMPLES_VALUES[@]}"; do
    for num_me_pairs in "${NUM_ME_PAIRS_VALUES[@]}"; do
      for num_co_pairs in "${NUM_CO_PAIRS_VALUES[@]}"; do
        for num_likely_psngrs in "${LIKELY_PASSENGERS_VALUES[@]}"; do
          for i in "${!TAU_LOW_VALUES[@]}"; do
            tau_low="${TAU_LOW_VALUES[i]}"
            tau_high="${TAU_HIGH_VALUES[i]}"
            for run in $(seq 1 $NUM_RUNS); do

              JOB_NAME="${subtype}_NS${num_samples}_${num_me_pairs}ME_${num_co_pairs}CO_${num_likely_psngrs}P_${tau_low}TL_${tau_high}TH_R${run}"
              SLURM_OUT_FN="${SLURM_LOG_DIR}/${JOB_NAME}.out"

              echo "Submitting job: ${JOB_NAME}"
              echo "Log file: ${SLURM_OUT_FN}"
              sbatch \
                -J "${JOB_NAME}" \
                -o "${SLURM_OUT_FN}" \
                -e "${SLURM_OUT_FN}" \
                "${SSBATCH_SCRIPT}" \
                  "${subtype}" \
                  "${num_samples}" \
                  "${num_me_pairs}" \
                  "${num_co_pairs}" \
                  "${num_likely_psngrs}" \
                  "${tau_low}" \
                  "${tau_high}" \
                  "${run}"
              sleep 0.25

            done
          done
        done
      done
    done
  done
done

echo "All matrix simulation jobs have been submitted."

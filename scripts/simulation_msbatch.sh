#!/usr/bin/env bash

# 1) Define hyperparameter arrays
SUBTYPES=("UCEC" "LUAD" "BRCA")
NUM_SAMPLES_VALUES=(100 500 1000 2000)
ME_PAIRS_VALUES=(0 5 10 15)
CO_PAIRS_VALUES=(0 5 10 15)
LIKELY_PASSENGERS_VALUES=(25 50 100 200)
IXN_STRENGTH_VALUES=(0.025 0.05 0.10 0.20)

# 2) Define base output directory and Slurm logs subdirectory
OUTPUT_BASE="output/SIMULATIONS"
SLURM_LOG_DIR="${OUTPUT_BASE}/slurm_logs"
mkdir -p "${SLURM_LOG_DIR}"

# 3) Path to the single-run Slurm script (must be executable)
SCRIPT_DIR="$(dirname "$0")"
SSBATCH_SCRIPT="${SCRIPT_DIR}/simulation_ssbatch.sh"

# 4) Loop over each combination of hyperparameters
for subtype in "${SUBTYPES[@]}"; do
  for nsamples in "${NUM_SAMPLES_VALUES[@]}"; do
    for mepairs in "${ME_PAIRS_VALUES[@]}"; do
      for copairs in "${CO_PAIRS_VALUES[@]}"; do
        for likely_pass in "${LIKELY_PASSENGERS_VALUES[@]}"; do
          for ixn_strength in "${IXN_STRENGTH_VALUES[@]}"; do

            # Construct a job name that captures the parameter combo
            JOB_NAME="${subtype}_NS${nsamples}_${mepairs}ME_${copairs}CO_${likely_pass}P_${ixn_strength}IXN"

            # Define the Slurm output file (combined stdout and stderr)
            SLURM_OUT_FN="${SLURM_LOG_DIR}/${JOB_NAME}.out"

            echo "Submitting job: ${JOB_NAME}"
            echo "Log file: ${SLURM_OUT_FN}"

            # Submit the job, passing parameters as positional arguments
            sbatch \
              -J "${JOB_NAME}" \
              -o "${SLURM_OUT_FN}" \
              -e "${SLURM_OUT_FN}" \
              "${SSBATCH_SCRIPT}" \
                "${subtype}" \
                "${nsamples}" \
                "${mepairs}" \
                "${copairs}" \
                "${likely_pass}" \
                "${ixn_strength}"
            exit

            # (Optional) Sleep to avoid overwhelming scheduler
            sleep 0.25

          done
        done
      done
    done
  done
done

echo "All matrix simulation jobs have been submitted."

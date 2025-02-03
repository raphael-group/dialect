#!/usr/bin/env bash


# SIMULATION GROUP 1: VARY NUM ME PAIRS AND LIKELY PASSENGERS AND TAU
SUBTYPES=("UCEC")
NUM_SAMPLES_VALUES=(1000)
NUM_ME_PAIRS_VALUES=(10 15 20 25)
NUM_CO_PAIRS_VALUES=(0)
LIKELY_PASSENGERS_VALUES=(50 100 150 200)
TAU_LOW_VALUES=(0.05 0.10 0.15)
TAU_HIGH_VALUES=(0.35 0.30 0.25)
DRIVER_PROPORTIONS=(1.0)
NUM_RUNS=10

# SIMULATION GROUP 2: VARYING NUMBER OF SAMPLES
# SUBTYPES=("UCEC")
# NUM_SAMPLES_VALUES=($(seq 100 100 4000))
# NUM_ME_PAIRS_VALUES=(25)
# NUM_CO_PAIRS_VALUES=(0)
# LIKELY_PASSENGERS_VALUES=(150)
# TAU_LOW_VALUES=(0.15)
# TAU_HIGH_VALUES=(0.25)
# DRIVER_PROPORTIONS=(1.0)
# NUM_RUNS=10

# SIMULATION GROUP 3: VARYING DRIVER PROPORTIONS
# SUBTYPES=("UCEC")
# NUM_SAMPLES_VALUES=(1000)
# NUM_ME_PAIRS_VALUES=(25)
# NUM_CO_PAIRS_VALUES=(0)
# LIKELY_PASSENGERS_VALUES=(150)
# TAU_LOW_VALUES=(0.15)
# TAU_HIGH_VALUES=(0.25)
# DRIVER_PROPORTIONS=($(seq 0.1 0.1 1.0))
# NUM_RUNS=10

# SIM TEST
# SUBTYPES=("UCEC")
# NUM_SAMPLES_VALUES=(100)
# NUM_ME_PAIRS_VALUES=(1)
# NUM_CO_PAIRS_VALUES=(0)
# LIKELY_PASSENGERS_VALUES=(8)
# TAU_LOW_VALUES=(0.15)
# TAU_HIGH_VALUES=(0.25)
# DRIVER_PROPORTIONS=(1.0)
# NUM_RUNS=1

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
          for driver_prop in "${DRIVER_PROPORTIONS[@]}"; do
            for i in "${!TAU_LOW_VALUES[@]}"; do
              tau_low="${TAU_LOW_VALUES[i]}"
              tau_high="${TAU_HIGH_VALUES[i]}"
              for run in $(seq 1 $NUM_RUNS); do

                JOB_NAME="${subtype}_NS${num_samples}_${num_me_pairs}ME_${num_co_pairs}CO_${num_likely_psngrs}P_${tau_low}TL_${tau_high}TH_${driver_prop}DP_R${run}"
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
                    "${run}" \
                    "${driver_prop}" \
                    "${NUM_RUNS}"
                sleep 0.05

              done
            done
          done
        done
      done
    done
  done
done

echo "All matrix simulation jobs have been submitted."

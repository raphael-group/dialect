#!/usr/bin/env bash

DATA_DIR=data/TCGA_PanCancer_Atlas_2018 ##TODO: SPECIFY DIR W/ MAF FILES
DOUT=output/TOP_10_Genes ##TODO: SPECIFY OUTPUT DIR
MAF_FILES=(${DATA_DIR}/*.maf) ## Array of all MAF files in the directory
SCRIPT_DIR=$(dirname "$0") ## Directory of the current script
SLURM_DIR=${DOUT}/slurm_logs

mkdir -p ${SLURM_DIR}

for MAF_FILE in "${MAF_FILES[@]}"; do
    SUBTYPE_NAME=$(basename ${MAF_FILE} .maf)
    SLURM_OUT_FN=${SLURM_DIR}/${SUBTYPE_NAME}.out

    ## Submit job
    sbatch -o ${SLURM_OUT_FN} -e ${SLURM_OUT_FN} ${SCRIPT_DIR}/ssbatch.sh ${MAF_FILE}
    sleep 0.25 ## Avoid overloading the scheduler
done

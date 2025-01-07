#!/usr/bin/env bash
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=8   ## CPUs per node
#SBATCH -t 48:00:00           ## Walltime
#SBATCH --mem=50GB            ## Memory
#SBATCH --job-name=dialect_job ## Job name

source /n/fs/ragr-research/users/ashuaibi/anaconda3/etc/profile.d/conda.sh
conda activate dialect

ORIGINAL_DIR=$(pwd)
DISCOVER_DIR=${ORIGINAL_DIR}/external/DISCOVER
cd ${DISCOVER_DIR}
python setup.py install --user  # Use --user to install DISCOVER in user space
cd ${ORIGINAL_DIR}

K=100 ##TODO SPECIFY NUMBER OF TOP GENES
MAF_FILE=${1}                ## MAF file path
SUBTYPE_NAME=$(basename ${MAF_FILE} .maf) ## Extract subtype name from MAF file
OUTPUT_DIR=output/${SUBTYPE_NAME} ## Set output directory

mkdir -p ${OUTPUT_DIR}

## Commands to run
dialect generate -m ${MAF_FILE} -o ${OUTPUT_DIR}
dialect identify -c ${OUTPUT_DIR}/count_matrix.csv -b ${OUTPUT_DIR}/bmr_pmfs.csv -o ${OUTPUT_DIR} -k ${K} -cb ${OUTPUT_DIR}/CBaSE_output/q_values.txt
dialect compare -c ${OUTPUT_DIR}/count_matrix.csv -b ${OUTPUT_DIR}/bmr_pmfs.csv -o ${OUTPUT_DIR} -k ${K}
dialect merge -d ${OUTPUT_DIR}/pairwise_interaction_results.csv -a ${OUTPUT_DIR}/comparison_interaction_results.csv -o ${OUTPUT_DIR}

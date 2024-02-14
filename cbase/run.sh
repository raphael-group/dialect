#!/bin/bash

# Function to display the help message
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-c, --cancer_subtype    Specify the cancer subtype. This argument is required."
    echo "-d, --dataset           Specify the dataset to be used."
    echo "                        For cbase: 'pancan' or 'lawrence'."
    echo "-help                   Display this help message."
    echo
}

##### STEP 1: Read command line arguments #####
CANCER=NONE
DATASET=NONE
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
    -c | --cancer)
        CANCER="$2"
        shift # past argument
        shift # past value
        ;;
    -d | --dataset)
        DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    -help)
        display_help
        exit 0
        ;;
    -* | --*)
        echo "Unknown option $1. Use -help for more information."
        exit 1
        ;;
    *)
        POSITIONAL_ARGS+=("$1") # save positional arg
        shift                   # past argument
        ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ "${CANCER}" = NONE ]; then
    echo "Error: the cancer subtype argument is required: -c, --cancer"
    display_help
    exit 1
fi
if [ "${DATASET}" = NONE ]; then
    echo "Error: the dataset argument is required: -d, --dataset"
    display_help
    exit 1
fi

# Validate dataset is one of the choices
if ! [[ "${DATASET}" =~ ^(pancan|lawrence)$ ]]; then
    echo "Error: For cbase, expected dataset 'pancan'."
    display_help
    exit 1
fi

##### step 1: declare constants and variables #####
SCRIPTS=scripts
DOUT=out/${DATASET}/${CANCER}
VCF=${DOUT}/${CANCER}_cbase.vcf
MAF=reference/mafs/${DATASET}/${CANCER}.maf
MIS_CNT_MTX=${DOUT}/${CANCER}_mis_cnt_mtx.csv
NON_CNT_MTX=${DOUT}/${CANCER}_non_cnt_mtx.csv
KEPT_VCF=${DOUT}/${CANCER}_kept_mutations.csv
CBASE_MIS_BMRS=${DOUT}/pofmigivens_${CANCER}.txt
CBASE_NON_BMRS=${DOUT}/pofkigivens_${CANCER}.txt
CBASE_TEMP_DIR=${DOUT}/cbase_temp # for intermediate files
CBASE_AUX_DIR=reference/auxiliary # for cbase auxiliary files

##### step 2: create output directories #####
mkdir -p ${DOUT}
mkdir -p ${CBASE_TEMP_DIR}

##### step 3: run data setup and cbase scripts #####
python ${SCRIPTS}/convert_maf_to_vcf.py -maf ${MAF} -fout ${VCF}
python ${SCRIPTS}/cbase_params_v1.2.py ${VCF} 1 hg19 3 0 ${CANCER} ${CBASE_AUX_DIR} ${DOUT}
python ${SCRIPTS}/cbase_qvals_v1.2.py ${CANCER} ${DOUT}

##### step 5: clean up output directory #####
for file in output_data_preparation_${CANCER}.txt param_estimates_${CANCER}_* pofmgivens* pofkgivens* used_params_and_model_${CANCER}.txt mutation_mat_${CANCER}.txt; do
    mv ${DOUT}/${file} ${CBASE_TEMP_DIR}
done

##### step 6: create count matrix from kept mutations #####
python ${SCRIPTS}/build_cnt_mtx.py -mut_fn ${KEPT_VCF} -fout ${MIS_CNT_MTX} -mis
python ${SCRIPTS}/build_cnt_mtx.py -mut_fn ${KEPT_VCF} -fout ${NON_CNT_MTX} -non

##### step 7: generate cleaner table csv file from cbase output bmr file #####
python ${SCRIPTS}/build_bmr_table.py -c ${CANCER} -mtx_fn ${MIS_CNT_MTX} -mtype mis -bmr_fn ${CBASE_MIS_BMRS} -d ${DOUT}
python ${SCRIPTS}/build_bmr_table.py -c ${CANCER} -mtx_fn ${NON_CNT_MTX} -mtype non -bmr_fn ${CBASE_NON_BMRS} -d ${DOUT}

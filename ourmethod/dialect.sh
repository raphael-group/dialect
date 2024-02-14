#!/bin/bash

# Function to display the help message
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-c, --cancer    Specify the cancer subtype. This argument is required."
    echo "-bmr, --bmr_method      Specify the BMR method. This argument is required."
    echo "                        Possible options are: cbase, dig, or mutsig."
    echo "-d, --dataset           Specify the dataset to be used."
    echo "                        For cbase: 'pancan' or 'lawrence'."
    echo "                        For dig: 'pcawg' or 'dietlein'."
    echo "                        For mutsig: 'lawrence'."
    echo "-help                   Display this help message."
    echo
}

##### step 1: read command line arguments #####
CANCER=NONE
DATASET=NONE
BMR_METHOD=NONE
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
    -c | --cancer)
        CANCER="$2"
        shift # past argument
        shift # past value
        ;;
    -bmr | --bmr_method)
        BMR_METHOD="$2"
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
if [ "${BMR_METHOD}" = NONE ]; then
    echo "Error: the BMR method argument is required: -bmr, --bmr_method"
    display_help
    exit 1
fi
if [ "${DATASET}" = NONE ]; then
    echo "Error: the dataset argument is required: -d, --dataset"
    display_help
    exit 1
fi

# Check if the provided BMR method is one of the expected options
if ! [[ "${BMR_METHOD}" =~ ^(cbase|dig|mutsig)$ ]]; then
    echo "Error: Invalid BMR method. Expected 'cbase', 'dig', or 'mutsig'."
    display_help
    exit 1
fi

# Validate dataset based on the BMR method
case "${BMR_METHOD}" in
cbase)
    if ! [[ "${DATASET}" =~ ^(pancan|lawrence)$ ]]; then
        echo "Error: For cbase, expected dataset 'pancan'."
        display_help
        exit 1
    fi
    ;;
dig)
    if ! [[ "${DATASET}" =~ ^(pcawg|dietlein)$ ]]; then
        echo "Error: For dig, expected dataset 'pcawg' or 'dietlein'."
        display_help
        exit 1
    fi
    ;;
mutsig)
    if [ "${DATASET}" != "lawrence" ]; then
        echo "Error: For mutsig, expected dataset 'lawrence'."
        display_help
        exit 1
    fi
    ;;
esac

##### step 2: declare constants and variables and create output directories #####
SCRIPTS=scripts
BMR_DIR=reference/${BMR_METHOD}/bmrs/${DATASET}/${CANCER}
CNT_MTX_DIR=reference/${BMR_METHOD}/count_mtxs/${DATASET}/${CANCER}
DOUT=out/${BMR_METHOD}/${DATASET}/${CANCER}
mkdir -p ${DOUT}

##### step 3: run pipeline for different mutation types #####
for mut_type in mis non; do
    python ${SCRIPTS}/run_model.py -bmr_d ${BMR_DIR} -mtx_d ${CNT_MTX_DIR} -dout ${DOUT} -c ${CANCER} -single -${mut_type}
done
# only run indel model if BMR method is mutsig
if [ "${BMR_METHOD}" = "mutsig" ]; then
    python ${SCRIPTS}/run_model.py -bmr_d ${BMR_DIR} -mtx_d ${CNT_MTX_DIR} -dout ${DOUT} -c ${CANCER} -single -ind
fi
python ${SCRIPTS}/run_model.py -bmr_d ${BMR_DIR} -mtx_d ${CNT_MTX_DIR} -dout ${DOUT} -c ${CANCER} -pair

#### STEP 5: Create Final Tables #####
for mut_type in mis non; do # TODO make work for mis, non, ind w/ mutsig
    python ${SCRIPTS}/create_final_table.py --single \
        --cancer ${CANCER} \
        --count_mtx_fn ${CNT_MTX_DIR}/${CANCER}_${mut_type}_cnt_mtx.csv \
        --single_fn ${DOUT}/${CANCER}_${mut_type}_single_pi_vals.csv \
        --single_fout ${DOUT}/${CANCER}_final_${mut_type}_single.csv
done

# only run indel model if BMR method is mutsig
#if [ "${BMR_METHOD}" = "mutsig" ]; then
#    python ${SCRIPTS}/create_final_table.py --single \
#        --cancer ${CANCER} \
#        --count_mtx_fn ${CNT_MTX_DIR}/${CANCER}_ind_cnt_mtx.csv \
#        --single_fn ${DOUT}/${CANCER}_ind_single_pi_vals.csv \
#        --single_fout ${DOUT}/${CANCER}_final_ind_single.csv
#fi

python ${SCRIPTS}/create_final_table.py --pair \
    --cancer ${CANCER} \
    --count_mtx_fn ${DOUT}/${CANCER}_joint_cnt_mtx.csv \
    --orig_discover_fout ${DOUT}/${CANCER}_discover \
    --pair_fn ${DOUT}/${CANCER}_pair_pi_vals.csv \
    --pair_fout ${DOUT}/${CANCER}_final_pair.csv

#python ${SCRIPTS}/join_results.py -c ${CANCER} -d ${DATASET} -bmr ${BMR_METHOD}

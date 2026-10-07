#!/bin/bash

prefix='' #'.dt.001'
DATASET_NAME="case2.eq.meshnet" #"eq.r50"
#DATASET_NAME="meshnet.flow"
#DATASET_NAME="WaterDropSample.r003"
echo "please input the model name to rollout on:"
model_name=$1
TMP_DIR=$(pwd)"/gns-sample"
DATA_PATH="${TMP_DIR}/${DATASET_NAME}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/models${prefix}/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/rollouts${prefix}/"
mkdir -p ${ROLLOUT_PATH}
python -m meshnet.train \
	--data_path=${DATA_PATH} \
	--model_path=${MODEL_PATH} \
	--model_file=${model_name} \
	--output_path=${ROLLOUT_PATH} \
	--mode='rollout' \


for i in {0..2}

do
    rollout_fname="rollout_"${i}
    python3 -m meshnet.render \
        --rollout_dir=${ROLLOUT_PATH} \
        --rollout_name=${rollout_fname}
done


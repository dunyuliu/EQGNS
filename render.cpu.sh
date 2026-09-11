#!/bin/bash

DATASET="case4.200m.multi.stress.homo.a.Vw"
model_suffix="nmp10.cotopaxi.r1"
echo "please input the model name to rollout on, example model-10000.pt:"
echo " and the test dataset name in dataset_archive"
model_name=$1
#testset_name=$2
export CUDA=/usr/local/cuda-12/
export PATH=$PATH:${CUDA}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA}
export OMP_NUM_THREADS=1

#export CUDA_LAUNCH_BLOCKING=1
#MASTER_PORT=29501 
CUDA_VISIBLE_DEVICES=""

TMP_DIR=$(pwd)"/gns-sample"
DATA_PATH="${TMP_DIR}/${DATASET}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET}/models.${model_suffix}/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET}/rollouts.${model_suffix}/${model_name}/"
#rm -rf ${ROLLOUT_PATH}
mkdir -p ${ROLLOUT_PATH}
#cp -r "dataset_archive/"${testset_name} ${DATA_PATH}"/test.npz"
cp -r ${DATA_PATH}/testset_metadata.json ${ROLLOUT_PATH}


for i in {0..0}

do
    rollout_fname="rollout_"${i}
    python3 -m meshnet.render \
        --rollout_dir=${ROLLOUT_PATH} \
        --rollout_name=${rollout_fname} #>> ${ROLLOUT_PATH}rollout.log.txt
done


#!/bin/bash

# This script moved from repo root to scripts/ (PROJECT_RULES.md rule 9);
# gns-sample/ and dataset_archive/ stay at the repo root (rule 3), so
# resolve REPO_ROOT from this script's own location and cd there -- this
# still works whether invoked as `bash scripts/render.sh` from the repo
# root or `bash render.sh` from inside scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DATASET="case4.55MPa"
model_suffix="nmp10"
echo "please input the model name to rollout on, example model-10000.pt:"
echo " and the test dataset name in dataset_archive"
model_name=$1
testset_name=$2
export CUDA=/usr/local/cuda-12/
export PATH=$PATH:${CUDA}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA}
export OMP_NUM_THREADS=1

#export CUDA_LAUNCH_BLOCKING=1
#MASTER_PORT=29501
CUDA_VISIBLE_DEVICES=1

TMP_DIR="${REPO_ROOT}/gns-sample"
DATA_PATH="${TMP_DIR}/${DATASET}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET}/models.${model_suffix}/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET}/rollouts.${model_suffix}/${model_name}/"
#rm -rf ${ROLLOUT_PATH}
mkdir -p ${ROLLOUT_PATH}
cp -r "${REPO_ROOT}/dataset_archive/"${testset_name} ${DATA_PATH}"/test.npz"
cp -r ${DATA_PATH}/testset_metadata.json ${ROLLOUT_PATH}


for i in {0..0}

do
    rollout_fname="rollout_"${i}
    python3 -m meshnet.render \
        --rollout_dir=${ROLLOUT_PATH} \
        --rollout_name=${rollout_fname} #>> ${ROLLOUT_PATH}rollout.log.txt
done


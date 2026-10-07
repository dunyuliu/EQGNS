#git clone https://github.com/geoelements/gns-sample

nstep=400000

TMP_DIR="./gns-sample"
DATASET_NAME="WaterDropSample"

mkdir -p ${TMP_DIR}/${DATASET_NAME}/models/
mkdir -p ${TMP_DIR}/${DATASET_NAME}/rollout/

DATA_PATH="${TMP_DIR}/${DATASET_NAME}/dataset/"
MODEL_PATH="${TMP_DIR}/${DATASET_NAME}/models/"
ROLLOUT_PATH="${TMP_DIR}/${DATASET_NAME}/rollout/"

python -m gns.train \
	--data_path=${DATA_PATH} \
	--model_path=${MODEL_PATH} \
	--model_file="model-24419.pt" \
	--train_state_file="train_state-24419.pt" \
 	--ntraining_steps=${nstep}

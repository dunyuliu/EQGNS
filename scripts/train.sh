#bin/bash

#SBATCH -J pyt_sand3d_train         # Job name
#SBATCH -o pyt_sand3d_train.o%j     # Name of stdout output file
#SBATCH -e pyt_sand3d_train.e%j     # Name of stderr error file
#SBATCH -p gpu-a100              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=dliu@ig.utexas.edu
#SBATCH -A OTH21021          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
#cd ..
#source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
echo "Usage: ./train.sh dataset_name model_path_suffix gpu_id"
echo "Example: ./train.sh case3.200m nmp10.cotopaxi 0"
echo "availabel datasets include: case3.200m"
echo "case4.200m.multi.stress"

data=$1 
#"case4.200m.multi.stress"
#'case4.200m.multi.stress.160scenarios.homo.a.Vw'
model_suffix=$2
gpu_id=$3
batch_size=$4

SCRATCH="./gns-sample"
export CUDA=/usr/local/cuda-12/
export PATH=$PATH:${CUDA}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA}
export OMP_NUM_THREADS=1
#export CUDA_LAUNCH_BLOCKING=1
#MASTER_PORT=29501 
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m meshnet.train --data_path="${SCRATCH}/${data}/dataset/" \
	--model_path="${SCRATCH}/${data}/models.${model_suffix}/" \
	--output_path="${SCRATCH}/${data}/rollouts/" \
        --batch_size=${batch_size} \
	--nsave_steps=100000 \
	--ntraining_steps=10000000 \
	--model_file="latest" \
	--train_state_file="latest"

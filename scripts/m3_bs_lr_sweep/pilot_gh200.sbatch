#!/bin/bash
# Throughput pilot for the M3 batch-size / LR-scaling sweep on NVIDIA GH200 (TACC).
# docs/dev/M3_BATCH_LR_SWEEP_DESIGN.md section 5, row "Pilot (mandatory first)".
#
# One node, one GPU. Optionally runs `pytest -m training_gate` first (design doc
# section 3.4: PILOT_GATE=1 current train() vs the committed A100 oracle trace; PILOT_GATE=2
# same-hardware check: frozen oracle AND current train() both on this GPU via same_hw_gate.py,
# committed reference untouched), then one
# fixed-wall-clock from-scratch training run per batch size listed in PILOT_ARMS,
# each stopped by `timeout` (train.py has no wall-clock stop) and scored by
# throughput.py. No checkpoint beyond model-0.pt is written (nsave_steps huge).
#
# Site specifics (partition, account, paths) are NOT in this file: pass them on the
# sbatch command line and via --export, e.g.
#   sbatch -p <partition> -A <allocation> -t 00:55:00 -c 8 \      # -c explicit: -n 1 alone gave 1 CPU
#     --export=ALL,PILOT=$SCRATCH/eqgns-m3-sweep-pilot,EQGNS=$SCRATCH/eqgns-m3-sweep-pilot/EQGNS,\
#     VENV=<venv dir>,MODULES="gcc/13.2.0 cuda/12.6 python3/3.11.8",\
#     PILOT_GATE=1,PILOT_ARMS="8",ARM_MINUTES=25,SEED=1,GIT_SHA=<sha> \
#     scripts/m3_bs_lr_sweep/pilot_gh200.sbatch
# Expected layout under $PILOT: data/D2_160/dataset/{train,valid}.npz (the M3 training
# data), data/case3.200m.homo.a.Vw/dataset/{train,valid}.npz (M1 D1, for the gate),
# logs/, results/. The arm config is scripts/m3_bs_lr_sweep/config.m3_published.json.
#SBATCH -J eqgns_m3_pilot
#SBATCH -N 1
#SBATCH -n 1

set -uo pipefail
: "${PILOT:?}" "${EQGNS:?}" "${VENV:?}"
PILOT_GATE=${PILOT_GATE:-1}
PILOT_ARMS=${PILOT_ARMS:-"8"}
ARM_MINUTES=${ARM_MINUTES:-25}
SEED=${SEED:-1}
GIT_SHA=${GIT_SHA:?set GIT_SHA to the 40-hex commit of the staged tree}
GATE_BATCHES=${GATE_BATCHES:-2}   # same_hw_gate.py --batch-sizes (PILOT_GATE=2)
MODULES=${MODULES:-"gcc/13.2.0 cuda/12.6 python3/3.11.8"}
LOSS_REPORT_STEP=${LOSS_REPORT_STEP:-250}

# shellcheck disable=SC2086
module load $MODULES
source "$VENV/bin/activate"
cd "$EQGNS"
export PYTHONPATH="$EQGNS"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$PILOT/logs" "$PILOT/results"
R="$PILOT/results"; L="$PILOT/logs"

{
  echo "job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION node=$(hostname) start=$(date -u +%FT%TZ)"
  echo "git_sha=$GIT_SHA python=$(python -V 2>&1) torch=$(python -c 'import torch;print(torch.__version__)')"
  nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
  echo "cpus=$(nproc) mem_total_gb=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo)"
  echo "arms=$PILOT_ARMS arm_minutes=$ARM_MINUTES seed=$SEED gate=$PILOT_GATE omp=$OMP_NUM_THREADS"
} | tee -a "$R/node_info.$SLURM_JOB_ID.txt"

if [ "$PILOT_GATE" = "1" ]; then
  echo "=== training_gate on this GPU (current train() vs frozen oracle, 1000 steps, M1 D1) ==="
  export EQGNS_TRAINING_GATE_DATA_DIR="$PILOT/data/case3.200m.homo.a.Vw/dataset"
  export EQGNS_TRAINING_GATE_CUDA_DEVICE=0
  export EQGNS_TRAINING_GATE_REQUIRE_DATA=1
  t0=$(date +%s)
  python -m pytest -m training_gate tests/test_training_gate.py -q -p no:cacheprovider \
      --basetemp="$L/pytest_tmp.$SLURM_JOB_ID" > "$R/training_gate.$SLURM_JOB_ID.log" 2>&1
  rc=$?
  echo "training_gate rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "$R/node_info.$SLURM_JOB_ID.txt"
  tail -5 "$R/training_gate.$SLURM_JOB_ID.log"
  if [ $rc -ne 0 ]; then
    echo "GATE=FAIL job=$SLURM_JOB_ID" > "$R/GATE_STATUS.txt"
    echo "training_gate FAILED on this hardware -- stopping, no arm is trained (design doc 3.4)."
    exit 1
  fi
  echo "GATE=PASS job=$SLURM_JOB_ID" > "$R/GATE_STATUS.txt"
elif [ "$PILOT_GATE" = "2" ]; then
  # Same-hardware gate (design doc 3.4): frozen oracle AND current train.py both on THIS
  # GPU; the committed A100 reference is reported for context only, never rewritten.
  echo "=== same-hardware training_gate on this GPU (oracle -> native reference; current vs native) ==="
  export EQGNS_TRAINING_GATE_DATA_DIR="$PILOT/data/case3.200m.homo.a.Vw/dataset"
  export EQGNS_TRAINING_GATE_CUDA_DEVICE=0
  export EQGNS_TRAINING_GATE_REQUIRE_DATA=1
  t0=$(date +%s)
  python scripts/m3_bs_lr_sweep/same_hw_gate.py --work "$L/same_hw_gate.$SLURM_JOB_ID" \
      --ref-out "$R/training_gate_reference_gh200.$SLURM_JOB_ID.json" \
      --result-out "$R/same_hw_gate.$SLURM_JOB_ID.json" --git-sha "$GIT_SHA" \
      --batch-sizes "$GATE_BATCHES" > "$R/same_hw_gate.$SLURM_JOB_ID.log" 2>&1
  rc=$?
  echo "same_hw_gate rc=$rc wall=$(( $(date +%s) - t0 ))s" | tee -a "$R/node_info.$SLURM_JOB_ID.txt"
  tail -40 "$R/same_hw_gate.$SLURM_JOB_ID.log"
  if [ $rc -ne 0 ]; then
    echo "SAME_HW_GATE=FAIL job=$SLURM_JOB_ID rc=$rc" > "$R/GATE_STATUS.txt"
    echo "same-hardware training_gate FAILED (rc=$rc) -- stopping, no arm is trained (design doc 3.4)."
    exit 1
  fi
  echo "SAME_HW_GATE=PASS job=$SLURM_JOB_ID" > "$R/GATE_STATUS.txt"
fi

for b in $PILOT_ARMS; do
  arm="b${b}_s${SEED}"
  mdir="$R/arm_$arm/"
  mkdir -p "$mdir"
  # Arm config = the published M3 config with a denser loss report cadence (timing only).
  python - "$EQGNS/scripts/m3_bs_lr_sweep/config.m3_published.json" "$mdir/config.json" "$LOSS_REPORT_STEP" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["loss_report_step"] = int(sys.argv[3])
json.dump(cfg, open(sys.argv[2], "w"), indent=4)
EOF
  echo "=== arm $arm: batch $b, ${ARM_MINUTES} min wall, start $(date -u +%FT%TZ) ==="
  nvidia-smi dmon -s um -d 5 -o T > "$L/dmon_$arm.$SLURM_JOB_ID.log" 2>&1 &
  dmon_pid=$!
  timeout --signal=INT --kill-after=60 "${ARM_MINUTES}m" \
    python -u meshnet/train.py --mode=train \
      --data_path="$PILOT/data/D2_160/dataset/" --model_path="$mdir" \
      --batch_size="$b" --ntraining_steps=100000000 --nsave_steps=100000000 \
      --seed="$SEED" --cuda_device_number=0 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done \
    > "$L/train_$arm.$SLURM_JOB_ID.log"
  echo "train.py pipeline rc=${PIPESTATUS[0]} (124 = stopped by timeout, as intended)"
  kill $dmon_pid 2>/dev/null; wait $dmon_pid 2>/dev/null
  grep -i -m3 'error\|Traceback\|out of memory' "$L/train_$arm.$SLURM_JOB_ID.log" || true
  python scripts/m3_bs_lr_sweep/throughput.py "$L/train_$arm.$SLURM_JOB_ID.log" --batch "$b" \
      --dmon "$L/dmon_$arm.$SLURM_JOB_ID.log" --label "$arm.$SLURM_JOB_ID" \
      --out "$R/throughput_$arm.$SLURM_JOB_ID.json" | tee -a "$R/node_info.$SLURM_JOB_ID.txt"
  rm -f "$mdir"/model-*.pt "$mdir"/train_state-*.pt   # pilot checkpoints are not an arm
done
echo "end=$(date -u +%FT%TZ)" | tee -a "$R/node_info.$SLURM_JOB_ID.txt"

#!/bin/bash
# One chained segment of one M3 sweep arm on NVIDIA GH200 (TACC).
# docs/dev/M3_BATCH_LR_SWEEP_DESIGN.md sections 3.2 (arm table) and 3.4 (execution constraints).
#
# An arm (21.6M samples, ~70-78 h at the pilot's measured 77-86 samples/s) does not fit one
# 48-h job, so submit_chain_gh200.sh queues NSEG copies of this script with afterany
# dependencies; each copy is one SEGMENT and is independent of how the previous one ended:
#   * DONE / HALT marker present            -> exit 0 immediately (no-op successor)
#   * newest VALID checkpoint pair (chain_ckpt.py latest-valid: torch.load()s both files,
#     moves a truncated pair to corrupt/ and falls back one save) -> resume from it with
#     explicit --model_file/--train_state_file; none and SEG=1 -> fresh start
#   * train.py wrapped in `timeout --signal=INT` (it has no wall-clock stop; SIGINT ->
#     KeyboardInterrupt -> clean return) with `pipefail`, exit code read from PIPESTATUS
#   * after the run: non-finite loss in loss_log.txt -> HALT (chain stops, no 70-h burn on a
#     diverged arm); train.py exit 0 at the target step -> DONE; any other exit -> HALT
#   * checkpoints pruned to every KEEP_EVERY steps + the KEEP_NEWEST newest pairs
# Locked on every arm, never overridable except for ARM=DRY: loss_report_step=1000 (from
# config.m3_published.json, verified by sha256 each segment) and --nsave_steps 10000.
# Site specifics (partition, account, paths, venv) come from submit_chain_gh200.sh.
# OMP_NUM_THREADS is set here unconditionally: --export=ALL carries the login shell's
# OMP_NUM_THREADS=1 in, which the pilot ran under by accident (pilot writeup, section 4).
#SBATCH -J eqgns_m3_arm
#SBATCH -N 1
#SBATCH -n 1

set -uo pipefail
: "${SWEEP:?}" "${EQGNS:?}" "${VENV:?}" "${DATA:?}" "${ARM:?}" "${SEG:?}" "${NSEG:?}" "${GIT_SHA:?}"
MODULES=${MODULES:-"gcc/13.2.0 cuda/12.6 python3/3.11.8"}
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-2830m}        # 47 h 10 min inside a 48:00:00 segment
CHAIN_OMP=${CHAIN_OMP:-8}
KEEP_EVERY=${KEEP_EVERY:-100000}
KEEP_NEWEST=${KEEP_NEWEST:-3}
NSAVE=10000
DRY_EXTRA_SEGMENTS=${DRY_EXTRA_SEGMENTS:-0}    # ARM=DRY only: run this many further segments in-job
DRY_CORRUPT_AFTER_SEG=${DRY_CORRUPT_AFTER_SEG:-}  # ARM=DRY only: truncate the newest train_state after this segment

# shellcheck disable=SC2086
module load $MODULES
source "$VENV/bin/activate"
cd "$EQGNS"
export PYTHONPATH="$EQGNS"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$CHAIN_OMP MKL_NUM_THREADS=$CHAIN_OMP OPENBLAS_NUM_THREADS=$CHAIN_OMP

ADIR="$SWEEP/arms/$ARM"; MDIR="$ADIR/model/"; LDIR="$ADIR/logs"
mkdir -p "$MDIR" "$LDIR"
CK="python $EQGNS/scripts/m3_bs_lr_sweep/chain_ckpt.py"
CFG_SRC="$EQGNS/scripts/m3_bs_lr_sweep/config.m3_published.json"

log() { printf '%s seg=%s %s\n' "$(date -u +%FT%TZ)" "${1}" "${*:2}" | tee -a "$ADIR/chain.log"; }
halt() { echo "HALT seg=$1 job=$SLURM_JOB_ID reason: ${*:2}" > "$ADIR/HALT"; log "$1" "HALT: ${*:2}"; }
record() {  # append one JSON line of segment provenance
  printf '{"arm":"%s","seg":%s,"job":"%s","node":"%s","git_sha":"%s","start_step":%s,"end_step":%s,"rc":%s,"outcome":"%s","t_start":"%s","t_end":"%s"}\n' \
    "$ARM" "$1" "$SLURM_JOB_ID" "$(hostname -s)" "$GIT_SHA" "$2" "$3" "$4" "$5" "$6" "$(date -u +%FT%TZ)" >> "$ADIR/segments.jsonl"
}

# --- provenance: a real, matching git SHA and a clean staged tree (no "unknown" default) ---
[[ "$GIT_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "GIT_SHA must be a 40-hex commit, got '$GIT_SHA'"; exit 2; }
tree_sha=$(git -C "$EQGNS" rev-parse HEAD 2>/dev/null) || { echo "$EQGNS is not a git checkout; cannot verify GIT_SHA"; exit 2; }
[ "$tree_sha" = "$GIT_SHA" ] || { echo "GIT_SHA $GIT_SHA != staged tree HEAD $tree_sha"; exit 2; }
dirty=$(git -C "$EQGNS" status --porcelain --untracked-files=no)
[ -z "$dirty" ] || { echo "staged tree has uncommitted changes:"; echo "$dirty"; exit 2; }
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
[ -n "$gpu_name" ] || { echo "nvidia-smi gave no GPU name"; exit 2; }

read -r BATCH SEED TARGET LR_INIT < <($CK arm "$ARM")
if [ "$ARM" = "DRY" ]; then
  NSAVE=${DRY_NSAVE:-500}; TARGET=${DRY_TARGET:-$TARGET}
fi

run_segment() {   # $1 = segment index
  local s=$1 start_step=-1 rc resume_args="" latest since outcome
  if [ -f "$ADIR/DONE" ]; then log "$s" "arm already DONE ($(cat "$ADIR/DONE")); nothing to do"; return 0; fi
  if [ -f "$ADIR/HALT" ]; then log "$s" "arm is HALTED ($(cat "$ADIR/HALT")); nothing to do"; return 0; fi

  # Config: written once from the committed published config, byte-verified afterwards.
  if [ ! -f "$MDIR/config.json" ]; then
    cp "$CFG_SRC" "$MDIR/config.json"
  fi
  cmp -s "$CFG_SRC" "$MDIR/config.json" || { halt "$s" "config.json differs from $CFG_SRC"; return 1; }
  python - "$MDIR/config.json" "$LR_INIT" <<'EOF' || { halt "$s" "config lr_init/loss_report_step lock violated"; return 1; }
import json, sys
c = json.load(open(sys.argv[1]))
assert c["loss_report_step"] == 1000, c["loss_report_step"]
assert abs(c["lr_init"] - float(sys.argv[2])) < 1e-12, (c["lr_init"], sys.argv[2])
EOF

  # Resume point: newest pair that actually loads (truncated pairs are moved aside).
  latest=$($CK latest-valid --model-dir "$MDIR" 2>>"$ADIR/chain.log")
  if [ "$latest" = "NONE" ]; then
    [ "$s" -eq 1 ] || log "$s" "no valid checkpoint found in a resume segment; starting fresh"
    log "$s" "fresh start (no --model_file/--train_state_file)"
  else
    if [ "$latest" -ge "$TARGET" ]; then
      echo "DONE step=$latest job=$SLURM_JOB_ID seg=$s" > "$ADIR/DONE"; log "$s" "target $TARGET already reached at step $latest"
      record "$s" "$latest" "$latest" 0 done_already "$(date -u +%FT%TZ)"; return 0
    fi
    start_step=$latest
    resume_args="--model_file=model-$latest.pt --train_state_file=train_state-$latest.pt"
    log "$s" "resume from step $latest"
  fi
  since=$start_step
  {
    echo "job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?} node=$(hostname -s) gpu=$gpu_name cpus=${SLURM_CPUS_ON_NODE:-?} omp=$OMP_NUM_THREADS"
    echo "git_sha=$GIT_SHA torch=$(python -c 'import torch;print(torch.__version__)') python=$(python -V 2>&1)"
    echo "arm=$ARM batch=$BATCH seed=$SEED target=$TARGET lr_init=$LR_INIT nsave=$NSAVE timeout=$TRAIN_TIMEOUT resume=${resume_args:-fresh}"
  } | tee -a "$LDIR/node_info.$SLURM_JOB_ID.seg$s.txt"

  local t0; t0=$(date -u +%FT%TZ)
  # shellcheck disable=SC2086
  timeout --signal=INT --kill-after=120 "$TRAIN_TIMEOUT" \
    python -u meshnet/train.py --mode=train \
      --data_path="$DATA/" --model_path="$MDIR" \
      --batch_size="$BATCH" --ntraining_steps="$TARGET" --nsave_steps="$NSAVE" \
      --seed="$SEED" --cuda_device_number=0 $resume_args 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done \
    > "$LDIR/train.$SLURM_JOB_ID.seg$s.log"
  rc=${PIPESTATUS[0]}
  log "$s" "train.py pipeline rc=$rc (0 = reached target, 124 = segment timeout)"

  if ! $CK check-loss --model-dir "$MDIR" --since-step "$since" 2>>"$ADIR/chain.log" | tee -a "$ADIR/chain.log"; then
    latest=$($CK latest-valid --model-dir "$MDIR" 2>/dev/null)
    record "$s" "$start_step" "$latest" "$rc" halt_nonfinite_loss "$t0"
    halt "$s" "non-finite loss after step $since (see loss_log.txt); chain stopped"; return 1
  fi
  latest=$($CK latest-valid --model-dir "$MDIR" 2>>"$ADIR/chain.log")
  if [ "$rc" -eq 0 ]; then
    if [ "$latest" != "NONE" ] && [ "$latest" -ge "$TARGET" ]; then
      echo "DONE step=$latest job=$SLURM_JOB_ID seg=$s" > "$ADIR/DONE"; outcome=done
      log "$s" "arm DONE at step $latest"
    else
      outcome=halt_early_exit; halt "$s" "train.py exited 0 at step $latest < target $TARGET"
    fi
  elif [ "$rc" -eq 124 ]; then
    if [ "$latest" = "NONE" ] || [ "$latest" -le "$start_step" ]; then
      outcome=halt_no_progress; halt "$s" "segment timed out without a new valid checkpoint (start $start_step, now $latest)"
    else
      outcome=timeout_resume_next; log "$s" "segment ended by timeout at checkpoint $latest; next segment resumes"
      [ "$s" -ge "$NSEG" ] && { echo "step=$latest target=$TARGET" > "$ADIR/NEEDS_MORE"; log "$s" "last planned segment -- NEEDS_MORE: submit another segment"; }
    fi
  else
    outcome=halt_error; halt "$s" "train.py rc=$rc (see $LDIR/train.$SLURM_JOB_ID.seg$s.log)"
    grep -m3 -i 'error\|Traceback\|out of memory' "$LDIR/train.$SLURM_JOB_ID.seg$s.log" | tee -a "$ADIR/chain.log" || true
  fi
  $CK prune --model-dir "$MDIR" --keep-every "$KEEP_EVERY" --keep-newest "$KEEP_NEWEST" | tee -a "$ADIR/chain.log"
  record "$s" "$start_step" "${latest:-NONE}" "$rc" "$outcome" "$t0"
  [ -f "$ADIR/HALT" ] && return 1
  return 0
}

seg=$SEG
run_segment "$seg"
# ARM=DRY self-test: more segments in the same job, with an injected truncated checkpoint.
corrupted=""
while [ "$ARM" = "DRY" ] && [ "$DRY_EXTRA_SEGMENTS" -gt 0 ]; do
  if [ "$DRY_CORRUPT_AFTER_SEG" = "$seg" ]; then
    corrupted=$(ls "$MDIR" | sed -n 's/^train_state-\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
    truncate -s 1000 "$MDIR/train_state-$corrupted.pt"
    log "$seg" "DRY: truncated train_state-$corrupted.pt to 1000 bytes (simulated kill during torch.save)"
  fi
  seg=$((seg + 1)); DRY_EXTRA_SEGMENTS=$((DRY_EXTRA_SEGMENTS - 1))
  run_segment "$seg" || true
done
log "$seg" "segment job end"
[ -f "$ADIR/HALT" ] && exit 1
if [ "$ARM" = "DRY" ]; then
  # The self-test passes only if the arm finished AND the injected truncation was detected and
  # fallen back from: corrupt/ holds the truncated pair and the next segment resumed below it.
  [ -f "$ADIR/DONE" ] || { log "$seg" "DRY FAIL: no DONE marker"; exit 1; }
  if [ -n "$corrupted" ]; then
    [ -f "$MDIR/corrupt/train_state-$corrupted.pt" ] || { log "$seg" "DRY FAIL: truncated pair not moved to corrupt/"; exit 1; }
    resumed=$(python - "$ADIR/segments.jsonl" "$((DRY_CORRUPT_AFTER_SEG + 1))" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(next((r["start_step"] for r in rows if r["seg"] == int(sys.argv[2])), "NONE"))
EOF
)
    { [ "$resumed" != "NONE" ] && [ "$resumed" -ge 0 ] && [ "$resumed" -lt "$corrupted" ]; } \
      || { log "$seg" "DRY FAIL: segment after truncation resumed from '$resumed', expected 0 <= step < $corrupted"; exit 1; }
    log "$seg" "DRY PASS: truncated step $corrupted detected, resumed from $resumed, arm DONE"
  fi
fi
exit 0

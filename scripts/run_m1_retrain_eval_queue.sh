#!/bin/bash
# Bounded background eval queue for the M1 retrain campaign (fixed-D1 arm,
# old-D1 control arm, matched-step published M1). Sentinel-based: waits for
# each arm's three seeds to reach model-500000.pt before scoring ANY of that
# arm's 5 checkpoint steps, so it never contends with live training for a
# GPU. Idempotent (tests/paper_parity/eval_m1_retrain.py skips work whose
# JSON already exists), safe to re-launch if killed.
#
# Usage: setsid nohup bash scripts/run_m1_retrain_eval_queue.sh \
#          > /home/utig5/dliu/eq_rupture_gns_data/m1_retrain/eval_queue.log 2>&1 < /dev/null & disown
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M1_RETRAIN=/home/utig5/dliu/eq_rupture_gns_data/m1_retrain
RESULTS_DIR="$M1_RETRAIN/eval_results"
STEPS="100000 200000 300000 400000 500000"
POLL_SECS=90

mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR" || { echo "[queue] FATAL: cannot cd to $REPO_DIR"; exit 1; }

log() { echo "[queue $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

sentinels_ready() {  # $1=arm ("fixed"|"old")
    for seed in 0 1 2; do
        [ -f "$M1_RETRAIN/${1}_seed${seed}/models/model-500000.pt" ] || return 1
    done
    return 0
}

gpu_busy() {  # $1=gpu index; "busy" if any compute process is using it
    nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . && return 0
    return 1
}

pick_free_gpu() {  # prints 1 or 2, preferring 2; only ever considers GPU1/GPU2
    if ! gpu_busy 2; then echo 2; return; fi
    if ! gpu_busy 1; then echo 1; return; fi
    echo ""  # neither free
}

run_arm() {  # $1=arm ("fixed"|"old") $2=gpu
    local arm="$1" gpu="$2"
    for seed in 0 1 2; do
        for step in $STEPS; do
            local label="${arm}_seed${seed}"
            local out="$RESULTS_DIR/${label}_step${step}.json"
            if [ -f "$out" ]; then
                log "skip $label step$step (already done)"
                continue
            fi
            log "run $label step$step on GPU$gpu"
            python3 tests/paper_parity/eval_m1_retrain.py "$label" --steps "$step" --cuda "$gpu" \
                >> "$M1_RETRAIN/eval_queue.log" 2>&1
            if [ $? -ne 0 ]; then
                log "FAILED $label step$step (see log above) -- continuing with remaining work"
            fi
        done
    done
}

run_published() {
    for step in $STEPS; do
        local out="$RESULTS_DIR/published_step${step}.json"
        if [ -f "$out" ]; then
            log "skip published step$step (already done)"
            continue
        fi
        local gpu
        gpu=$(pick_free_gpu)
        while [ -z "$gpu" ]; do
            log "published step$step: GPU1 and GPU2 both busy, waiting ${POLL_SECS}s"
            sleep "$POLL_SECS"
            gpu=$(pick_free_gpu)
        done
        log "run published step$step on GPU$gpu"
        python3 tests/paper_parity/eval_m1_retrain.py published --steps "$step" --cuda "$gpu" \
            >> "$M1_RETRAIN/eval_queue.log" 2>&1
        if [ $? -ne 0 ]; then
            log "FAILED published step$step (see log above) -- continuing with remaining work"
        fi
    done
}

all_done() {
    for arm in fixed old; do
        for seed in 0 1 2; do
            for step in $STEPS; do
                [ -f "$RESULTS_DIR/${arm}_seed${seed}_step${step}.json" ] || return 1
            done
        done
    done
    for step in $STEPS; do
        [ -f "$RESULTS_DIR/published_step${step}.json" ] || return 1
    done
    return 0
}

log "queue started (pid $$), repo=$REPO_DIR"

# Published eval doesn't depend on either training arm -- do it whenever a
# GPU is free, in the background, so it doesn't block the sentinel waits.
run_published &
PUBLISHED_PID=$!
log "published eval running in background as pid $PUBLISHED_PID"

log "waiting for fixed arm sentinels (fixed_seed{0,1,2}/models/model-500000.pt)"
while ! sentinels_ready fixed; do
    sleep "$POLL_SECS"
done
log "fixed arm sentinels ready -- scoring fixed_seed{0,1,2} x $STEPS on GPU2 (frees once fixed training ends)"
run_arm fixed 2

log "waiting for old arm sentinels (old_seed{0,1,2}/models/model-500000.pt)"
while ! sentinels_ready old; do
    sleep "$POLL_SECS"
done
gpu=$(pick_free_gpu)
while [ -z "$gpu" ]; do
    log "old arm ready but GPU1/GPU2 both busy, waiting ${POLL_SECS}s"
    sleep "$POLL_SECS"
    gpu=$(pick_free_gpu)
done
log "old arm sentinels ready -- scoring old_seed{0,1,2} x $STEPS on GPU$gpu"
run_arm old "$gpu"

log "waiting for published eval (pid $PUBLISHED_PID) to finish"
wait "$PUBLISHED_PID"

if all_done; then
    touch "$RESULTS_DIR/ALL_DONE"
    log "all 35 evaluations present -- wrote $RESULTS_DIR/ALL_DONE -- exiting"
else
    log "WARNING: exiting main flow but not all 35 result files present -- check log for FAILED lines"
fi

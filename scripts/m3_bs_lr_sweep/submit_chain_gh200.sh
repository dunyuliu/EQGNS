#!/bin/bash
# Queue one M3 sweep arm as NSEG chained 48-h segments on NVIDIA GH200 (TACC).
# docs/dev/M3_BATCH_LR_SWEEP_DESIGN.md sections 3.2 / 3.4; segment logic in arm_segment_gh200.sbatch.
#
# Segments are submitted up front with --dependency=afterany, so a segment that dies hard
# (node failure, wall kill before `timeout` fires) is still followed by one that resumes from
# the newest valid checkpoint; a segment that finds DONE/HALT exits immediately (billed at the
# site's 15-minute minimum, ~0.25 SU). At the pilot's measured rates (b8 70.8 h, b4 78.1 h,
# b12 70.0 h per arm) two segments do the work; NSEG=3 leaves one spare.
#
# Usage (login node, from the staged clean git checkout; site names only on the command line):
#   SWEEP=$SCRATCH/eqgns-m3-sweep EQGNS=$SWEEP/EQGNS VENV=<venv> DATA=<dir with train.npz,valid.npz> \
#   ACCOUNT=<allocation> [PARTITION=gh] [WALL=48:00:00] [NSEG=3] [EXTRA="TRAIN_TIMEOUT=200s,..."] \
#   [GATE_DEP=afterok:<gate job>:<dry-run job>] bash scripts/m3_bs_lr_sweep/submit_chain_gh200.sh A0a
# GATE_DEP is AND-ed into EVERY segment's dependency (design doc 3.4: no arm starts unless the
# same-hardware gate passed) with --kill-on-invalid-dep=yes, so a failed gate cancels the whole
# chain instead of letting segment 2 start fresh after a cancelled segment 1.
# Prints "ARM seg jobid" per segment and appends the same to $SWEEP/arms/ARM/submissions.txt.
set -euo pipefail
ARM=${1:?arm name (A0a A0b A1 A2, or DRY)}
GATE_DEP=${GATE_DEP:-}                  # e.g. afterok:<gate job>:<dry job>; applied to every segment
: "${SWEEP:?}" "${EQGNS:?}" "${VENV:?}" "${DATA:?}" "${ACCOUNT:?}"
PARTITION=${PARTITION:-gh}; WALL=${WALL:-48:00:00}; NSEG=${NSEG:-3}; CPUS=${CPUS:-8}
EXTRA=${EXTRA:-}                        # extra KEY=VAL pairs for --export (DRY_* / TRAIN_TIMEOUT / KEEP_EVERY ...)

GIT_SHA=$(git -C "$EQGNS" rev-parse HEAD)
dirty=$(git -C "$EQGNS" status --porcelain --untracked-files=no)
[ -z "$dirty" ] || { echo "refusing: staged tree $EQGNS has uncommitted changes:"; echo "$dirty"; exit 2; }
python3 "$EQGNS/scripts/m3_bs_lr_sweep/chain_ckpt.py" arm "$ARM" >/dev/null   # unknown arm -> fail here
for f in "$DATA/train.npz" "$DATA/valid.npz" "$VENV/bin/activate"; do [ -e "$f" ] || { echo "missing $f"; exit 2; }; done

ADIR="$SWEEP/arms/$ARM"; mkdir -p "$ADIR/logs"
export_list="ALL,SWEEP=$SWEEP,EQGNS=$EQGNS,VENV=$VENV,DATA=$DATA,ARM=$ARM,NSEG=$NSEG,GIT_SHA=$GIT_SHA"
[ -n "$EXTRA" ] && export_list="$export_list,$EXTRA"
prev=""
for s in $(seq 1 "$NSEG"); do
  dep="$GATE_DEP${prev:+${GATE_DEP:+,}afterany:$prev}"
  jid=$(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" -N 1 -n 1 -c "$CPUS" -t "$WALL" \
          -J "m3_${ARM}_s$s" -o "$ADIR/logs/slurm.o%j" -e "$ADIR/logs/slurm.e%j" \
          ${dep:+--dependency=$dep} ${dep:+--kill-on-invalid-dep=yes} --export="$export_list,SEG=$s" \
          "$EQGNS/scripts/m3_bs_lr_sweep/arm_segment_gh200.sbatch")
  jid=${jid%%;*}
  echo "$ARM $s $jid dep=${dep:-none}" | tee -a "$ADIR/submissions.txt"
  prev=$jid
done
echo "git_sha=$GIT_SHA partition=$PARTITION wall=$WALL nseg=$NSEG submitted=$(date -u +%FT%TZ)" >> "$ADIR/submissions.txt"

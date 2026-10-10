# M3 batch-size / LR sweep — GH200 pilot, 2026-10-09

Board row `m3-batchsize-lr-sweep-gh200`; design `M3_BATCH_LR_SWEEP_DESIGN.md`
(section 5 "Pilot (mandatory first)", section 3.4 gate precondition, section 7
open questions 2–4). Hardware: NVIDIA GH200 (TACC), allocation EAR26006.
Everything below was freshly run or read this session; nothing is inherited.

## Verdict

**Pilot stopped at the gate, by design. No arm was trained.**
`pytest -m training_gate` FAILS on GH200 against the committed A100-made
reference (`tests/golden/training_gate_reference.json`): 999/1001 per-step
train losses differ (rtol/atol 1e-6), max |Δ| 0.70, max rel 5.06. The traces
are identical at step 0 (1.029649019241333 on both), differ at step 1 by
2e-6 relative (1.3600726 vs 1.3600699), and drift from there (step 500:
0.0361 vs 0.0373; step 1000: 0.2750 vs 0.2719). That signature is
cross-hardware floating-point divergence amplified over 1000 optimizer
steps, not a code divergence — but the committed gate cannot tell the two
apart, because it compares current `train.py` on GH200 with
`train.py.published` on A100 (hardware and code change together). Per the
brief this is the finding; no workaround was attempted. The same-hardware
check the design doc actually asks for (regenerate the reference with
`train.py.published` on GH200, ~2 min, then compare) is the obvious next
step and is the conductor's call, not taken here.

Samples/s per arm: **not measured** (arms never started). Scope decision
(reduced 4-arm vs minimum) therefore remains open — a judgment call for the
conductor, not settled by this pilot.

## Jobs and cost

| Job | Partition | Account | Content | State | Elapsed | Node-h |
|---|---|---|---|---|---|---|
| 1061515 | gh-dev | EAR26006 | gate + b8 | CANCELLED by me while PD (never ran; resubmitted on `gh` per coordinator) | 0 | 0 |
| 1061516 | gh-dev | EAR26006 | b4 + b12 (afterok 1061515) | CANCELLED by me while PD | 0 | 0 |
| 1061645 | gh | EAR26006 | gate + b8 | FAILED at the gate (exit 1 by the sbatch's own stop rule) | 00:02:19 | 0.04 |
| 1061646 | gh | EAR26006 | b4 + b12 (afterok 1061645) | CANCELLED by dependency / my watcher, never started | 0 | 0 |

Consumed: **0.04 of the 2.0 node-hour cap.** EAR26005 never used. The other
project's processes on that system (`sync_loop.sh`, `localroll.sh`, the ssh
control master, dev job 1060815) were not touched; the existing control
socket was reused for every command, no new login/control-master opened.
Node 1061645: GH200 120GB (97,871 MiB), torch 2.6.0+cu126, Python 3.11.8,
driver code at `1da434c`. Note for the next submission: on `gh`, `-n 1`
without `-c` allocated **1 CPU** — add `-c <n>` (the data loader runs in the
main process; 8 is the gate's own thread cap).

## Open question 2 — non-dev partition

**Settled: `gh` is usable under EAR26006.** Evidence: `sbatch -p gh -A
EAR26006` accepted and ran job 1061645 (QOS `qdefault`); `sacctmgr show
qos`: `qgh`/`qnormal` MaxWall 2-00:00:00, 20 running / 40 submitted per
user; `qdevelopment` (gh-dev) MaxWall 02:00:00, 1 running / 3 submitted per
user. Consequence for the design: a 21.6M-sample arm chains in 2–3 48-h
segments instead of 35–70 2-h segments. Caveat: `gh` is shared with the
same account's other project (5 running + 7 pending jobs at submission
time), so wall-clock to start is not under this row's control; 1061645
waited ~1 h 48 min in PD.

## Open question 3 — the on-disk "b12" directory

**Batch 10, mislabeled; directory left untouched.** No launch command or
SLURM log exists for it anywhere on the local box (data directory, `scripts/`,
`runs/`, shell histories, git history all searched; the only mention is as a
rollout target in `scripts/utils/batch.rollout.py`). The launcher of that
era, `scripts/train.sh`, takes the directory suffix (`$2`) and `--batch_size`
(`$4`) as independent manual arguments, so label and batch can disagree.
Arithmetic from `epoch_loss_log.txt` (first epoch ends at step 11,564):
11,564 × 10 = 115,640 = the D2_160 training-sample count (the same count the
b8/b4/b2 logs give: 14,455 × 8, 28,910 × 4, 57,820 × 2); batch 12 would end
epoch 0 at ceil(115,640/12) = 9,637. The GH200 arm stays b12 (owner's named
level).

## Open question 4 — final-slip RMSE

**Added to the driver and working.** `scripts/m3_bs_lr_sweep/sweep_eval.py`
reuses `gate.metrics()` (vs truth; its `final_slip_rmse` already exists since
PR #71) and `measure_vs_published.compare_pair()` (vs anchor) and adds
`final_slip_pair()` for the vs-anchor side (per-node ∫|v|dt over
`compare_pair`'s 755-step window, RMSE across nodes, plus the same normalised
by the anchor's peak final slip). Demonstrated locally (CPU, zero GH200 time)
on the shipped M3_D3 rollouts, checkpoint 2.4M scored against the published
2.7M as anchor P (n = 14 excluding trajectory 7, which is reported but
flagged): final-slip RMSE mean **2.44 m** (normalised 0.251); vs truth
2.99 m; for scale the same pair gives ΔRT RMSE 0.099 s, ΔMw 0.034,
vx RMSE/peak 0.030. These numbers are a first entry of the design's
checkpoint-jitter floor F_ckpt (2.4M vs 2.7M), A100 rollouts on both sides.

## What landed

- `scripts/m3_bs_lr_sweep/pilot_gh200.sbatch` — gate-first pilot job; site
  specifics on the command line only.
- `scripts/m3_bs_lr_sweep/throughput.py` — samples/s from timestamped
  `train.py` reports (steady-state slope, warm-up excluded, GPU util).
- `scripts/m3_bs_lr_sweep/sweep_eval.py` — metrics driver (score / rollout
  modes), final-slip RMSE addition. `tests/` untouched.
- `scripts/m3_bs_lr_sweep/config.m3_published.json` — the published M3 config.
- Staged on the GH200 scratch (`$SCRATCH/eqgns-m3-sweep-pilot/`): D2_160
  `train.npz`/`valid.npz` (29.8 GB, sizes verified) and M1 D1
  `train.npz`/`valid.npz`; the pilot code tree; the failing gate's
  `loss_log.txt` and pytest output under `logs/`/`results/`. Scratch purge
  policy applies.

## What a reviewer would attack

- The gate verdict conflates hardware and code; until `train.py.published`
  is run on GH200 the "cross-hardware, not code" reading is an inference
  from the step-0 match and the 2e-6 step-1 difference, not a measurement.
- If the same-hardware gate passes, the sweep's section 4.1 logic (two
  anchors, G and P) already absorbs the training-hardware offset; if it also
  fails, `meshnet/train.py` has a GH200-specific divergence and the arms must
  use `train.py.published` semantics (design doc 3.4).
- Throughput remains unmeasured; the 280–540 node-hour estimate still rests
  on contended A100 numbers.

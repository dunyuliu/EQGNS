# M1 retrain (fixed-D1 vs old-D1), seed spread — status

Owner-approved budget (2026-09-29): 500,000 steps x 3 seeds per arm, 6 runs
total, launched concurrently on GPU 2.

## Setup
- Seeds: `--seed 0`, `--seed 1`, `--seed 2`, identical across both arms.
- Config: `gns-sample/case3.200m.homo.a.Vw/models.nmp10.cotopaxi/config.json`
  copied verbatim into each run's `model_path` (lr 1e-4, batch 2 via
  `--batch_size=2`, `simulator_nmessage_passing_steps=10`, `noise_std=2e-2`
  — matches paper Table 2; config.json is required or train.py silently
  falls back to `nmessage_passing_steps=15`, not 10).
- Checkpoints every 50,000 steps (`--nsave_steps=50000`), target 500,000
  steps (`--ntraining_steps=500000`).
- Fixed-D1 arm data: `/home/utig5/dliu/eq_rupture_gns_data/D1_fixed/dataset/`
  (regenerated this session, nskip loop-bound fix, PR #5).
- Old-D1 arm data: `gns-sample/case3.200m.homo.a.Vw/dataset/` (published,
  read-only, buggy zero-tail — used as-is, intentionally, as the control).
- Launched via `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 setsid nohup
  venv/bin/python3 -m meshnet.train ...`, detached (survives this session),
  logging to `<run_dir>/train.log`.

## Run directories
`/home/utig5/dliu/eq_rupture_gns_data/m1_retrain/{fixed,old}_seed{0,1,2}/`
(`models/`, `rollouts/`, `train.log` each).

PIDs at launch (2026-09-29, will not persist across a reboot -- re-check by
run directory / checkpoint mtimes, not by PID, if resuming after a gap):
fixed_seed0=2884432, fixed_seed1=2884433, fixed_seed2=2884434,
old_seed0=2884435, old_seed1=2884437, old_seed2=2884438.

## Concurrency
All 6 launched concurrently on GPU 2 per owner instruction, to be measured
for it/s once running; owner's fallback if concurrency hurts throughput:
run 3 at a time serially instead. Decision + measured it/s to be logged here
once available (in progress as of this commit).

## Evaluation plan (once checkpoints exist)
- Deterministic rollouts (`det_rollout.py`) + gate's `metrics()` (including
  the collapse guard, `var_ratio`/`collapsed`) at checkpoints
  100k/200k/300k/400k/500k for both arms, scored on the FIXED-D1 test set
  (same test set for both arms, so the comparison isolates the training-data
  effect).
- Also score the published M1 checkpoint (`models.nmp10.cotopaxi/model-3000000.pt`)
  if a 500k-step published snapshot doesn't exist (it doesn't -- published
  went to 3,000,000 steps per `test/paper_parity/gate.py`'s `M1 =
  ("models.nmp10.cotopaxi", 3000000)`); use the nearest available published
  checkpoint if one exists at an intermediate step, else report the 3M one
  for context only (not a matched-step comparison).
- Per-arm median + seed spread (min/max or IQR across the 3 seeds) at each
  matched step count -- NOT validation-based checkpoint selection (owner
  declined that row); compare only at matched step counts.
- Deliverable: a short results table in a PR (data + a README note only if
  the effect is real), not a notes dump.

## Concurrency decision (measured, 2026-09-29)
Launched all 6 concurrently on GPU 2 first, per owner instruction to try
that and measure. `fixed_seed0` crashed with `CUDA out of memory` within
minutes (GPU 2 already carried 2 unrelated jobs at launch time, ~6.2 GB,
before adding 6 more at ~5-7 GB each). Killed all 6, wiped the partial
logs/state, and fell back to the owner-specified fallback: **3 concurrent**.

Measured at 3-concurrent, all three seeds simultaneously, steps 0-4000:
**~220 ms/step (4.5 it/s) per run**, stable, no crashes, ~14m39s wall for
4000 steps each. This is ~2.4x slower per run than the single-run contended
baseline measured earlier (91 ms/step), but 3x the parallelism nets ~1.25x
more effective aggregate throughput than running one at a time.

Projected wall-clock: 500,000 steps x 0.220 s/step = ~30.6 h per batch of 3.
Plan: fixed-D1 arm (seeds 0,1,2) runs first batch, then old-D1 arm (seeds
0,1,2) second batch, sequentially -- **~61 h (~2.5 days) total**, run
directories/PIDs below.

## Status (2026-09-29, updated)
Batch 1 (fixed-D1, seeds 0/1/2) launched clean at 08:28 CDT, PIDs 3212806
(seed0), 3212808 (seed1), 3212809 (seed2). Stable through step 4000+ at
time of writing. First checkpoint (step 50,000) expected ~08:28 + 3.1h ≈
11:35 CDT. Batch 2 (old-D1 control) launches once batch 1 completes.
Evaluation (deterministic rollout + gate.py metrics at each 50k-step
checkpoint) begins once matching checkpoints exist for both arms.

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

## Status (2026-09-29, updated -- both batches now running concurrently)
Owner correction: this box's other GPU jobs are all the owner's own
(CycleGNS, dynamo_gns), no other users; GPU 1 was confirmed idle
(0% util) and approved for batch 2 immediately, rather than waiting for
batch 1 to finish. Verified independently before launch (`nvidia-smi`):
GPU 1 at 1417 MiB / 0% util (one small pre-existing owner process, PID
3848742, ~1.4 GB -- not the "5 MiB" the correction estimated, but 0% util,
consistent with "idle" and safe to add 3 more runs).

Batch 1 (fixed-D1, seeds 0/1/2): GPU 2, launched 08:28 CDT, PIDs 3212806/
3212808/3212809. ~220 ms/step (4.5 it/s) per run. Stable through step
43,000+ at time of writing, `model-0.pt` saved, next checkpoint (step
50,000) imminent (~11:35 CDT).

Batch 2 (old-D1 control, seeds 0/1/2): GPU 1, launched 10:49 CDT, PIDs
3857724/3857726/3857727. ~234 ms/step (4.3 it/s) per run, consistent with
batch 1's rate. Stable through step 4,000+, no OOM.

**GPU utilization check (owner's "use the resource well" ask)**: both
GPU 1 and GPU 2 measured at a sustained **100%** utilization (3 samples
each, 3s apart) with just the 3 training runs per GPU -- neither is
"well under ~90%". Per the owner's own stated condition, evaluation
rollouts are NOT being added to either GPU right now (would contend with
and slow the training runs); deferred until a GPU frees up (a batch
finishing, or a measured utilization drop). GPUs 0 and 3 untouched, no new
work added there, per instruction.

Evaluation (deterministic rollout + gate.py metrics, including the collapse
guard, at each 50k-step checkpoint) begins once (a) a GPU has real spare
capacity and (b) matching checkpoints exist for both arms at the same step
count.

## 2026-09-30 update -- owner-approved eval plan, queue dispatched
Checked (2026-09-30 06:11 CDT): both arms still training, no serialization.
Fixed arm (GPU2) at step ~374,000+; old arm (GPU1) at step ~352,000+; both
several hours from the 500,000-step finish.

Owner-approved eval plan: keep both batches running in parallel (no
serializing training); arm a bounded background queue that waits on each
run's `model-500000.pt` as a SENTINEL (not a loss_log poll):
- Fixed arm finishes (frees GPU2) -> serial eval of fixed_seed{0,1,2} at
  100k/200k/300k/400k/500k on GPU2.
- Old arm finishes (frees GPU1) -> eval of old_seed{0,1,2} at the same 5
  steps on whichever of GPU1/GPU2 is free.
- Published M1 also scored at the same matching 5 steps (published
  checkpoints DO exist at exactly 100k/200k/300k/400k/500k under
  gns-sample/case3.200m.homo.a.Vw/models.nmp10.cotopaxi/) -- NOT the
  3,000,000-step final published checkpoint, matched-step only.
- All scored on the FIXED-D1 test set (same test set for both arms +
  published, isolating the training-data effect).

Dispatched building + arming this queue to a subagent (new
`test/paper_parity/eval_m1_retrain.py` reusing gate.py's `metrics()`/
`valid_steps()`/collapse guard, + a new sentinel-polling launcher script),
launched detached so it survives independent of any session. Will verify
independently once armed, and again once it actually produces the 35
result files (3 seeds x 2 arms x 5 steps + 5 published) + `ALL_DONE`
sentinel, before assembling the final results-table PR.

## Queue landed (PR #12, not merged)
Independently re-verified (not just the subagent's report): both scripts
diff clean against current main; queue confirmed running (PID 307971 +
published-eval subshell PID 307981), correctly in its sentinel-poll wait,
0% CPU between polls, no errors; both training arms confirmed still alive
and untouched; proof-of-pipeline JSONs (quarantined under
`eval_results/_proof_quick/`, not the real result paths) re-read directly
-- sane numbers, no NaN/inf, matches the subagent's report exactly.
PR: https://github.com/dunyuliu/EQGNS/pull/12. Next update when the queue
produces real results (fixed arm finishes -> GPU2 eval begins) or when
`ALL_DONE` appears.

## COMPLETE (2026-09-30)
All 6 training runs finished (500,000 steps each); eval queue finished,
`ALL_DONE` sentinel written, all 35 result files present. Results table +
interpretation: `docs/M1_RETRAIN_RESULTS.md`. Board row `m1-retrain-fixed-data`
closes with that PR.

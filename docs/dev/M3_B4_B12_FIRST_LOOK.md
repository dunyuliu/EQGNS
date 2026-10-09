# M3 b4 / 'b12' checkpoint first look — 2026-10-09

Free regression first look on the existing M3 batch-size arms, run ahead of the
GH200 batch-size/LR sweep (`M3_BATCH_LR_SWEEP_DESIGN.md`, board row
`m3-batchsize-lr-sweep-gh200`). Diagnostic only: no owner threshold, no pass/fail.

**Verdict: the b4 arm (`models.nmp10.b4.cotopaxi.r1`, lr 1e-4, decay 0.1/5M) is
unstable in autoregressive rollout on the M3_D3 fractal test set** — predicted |v|
runs away to 770–1580 m/s (truth peak 21–51 m/s) mid-rupture (onset t = 2.1–5.3 s,
truth rupture still active to ~6.5 s) on 6/15 trajectories at 5.4M steps and 4/15 at
2.7M. Rupture time, missed and false are blind to it (first 0.1 m/s crossing happens
before the runaway); Mw error (+0.3 to +0.4) and mse_vx (50–320 vs b8's 0.1–1.4) catch
it. Confirmed path-independent: the historical eager published-code rollouts on disk
(`rollouts.nmp10.b4.cotopaxi.r1/model-{2.7M,5.0M,7.0M}.pt`, same test set) blow up on
2–3/15 trajectories each (mse_vx max 370 / 209 / 645; which trajectories blow up changes
with checkpoint). The published b8 checkpoint through the identical fast path does not
(mse_vx max 1.44). The 'b12'-named arm (actually batch 10, constant lr 3e-5, 0.3M steps =
3M samples, 14 % of budget) blows up on 1/15 (traj 4, mse_vx 5e3, Mw +1.4) and misses 38
nodes on traj 10 — undertrained, so its numbers are uninformative rather than wrong. No
arm blows up on M3_D1hypo (6 traj). Implication for the sweep: b4 @ 5.4M is **not usable
as a Phase-0 'matched samples' point** without first separating the batch effect from
the lr 1e-4 / 5M-decay effect; the gate's collapse guard (low variance) cannot see a
blow-up (high variance) — the sweep's scoring needs Mw error / mse_vx or a peak-|v| cap.

## Setup

- Anchor (side B): fresh deterministic rollout, published code (`meshnet/train.py.published`
  via `tests/paper_parity/det_rollout.py`), published b8 checkpoint `model-2700000.pt` —
  the same construction `gate.cmd_regression` uses. Side A: current code,
  `--rollout_fast=tf32 --rollout_batch_size=64` (also `cmd_regression`'s flags), checkpoint
  swapped by overriding `gate.CASES[case]`'s (model dir, step) before
  `measure_vs_published.fresh_raw_rollout()`. RT/Mw/missed/false math: `compare_pair()`
  (vs anchor) and `gate.metrics()` (vs EQdyna truth), reused unmodified.
- Arms: b8_pub@2.7M (noise floor of the fast path), b4@5.4M (matched samples 21.6M),
  b4@2.7M (matched steps), b12dir(b10)@0.3M.
- GPU 0 of the shared box (GPU 2 busy with an unrelated job, load avg ~40), worktree at
  `ad267cc`, interpreter the project venv, `OMP_NUM_THREADS=4`. Not a timing run.
- Per-trajectory JSON (scalars only) kept in the session scratchpad; the script was a
  throwaway (~50 lines) and is not retained — the recipe above reproduces it.

## Per-arm summary (vs truth unless marked; max over trajectories, n in header)

| case | arm | dRT RMSE vs anchor, max (s / DT) | \|dMw\| vs anchor, max | missed+false vs anchor, sum | rt_rmse vs truth, max (s) | mw_err vs truth, max | mse_vx max / median | trajectories with mse_vx > 5 |
|---|---|---|---|---|---|---|---|---|
| M3_D1hypo (n=6) | b8_pub@2.7M | 0.046 / 2.7 | 0.012 | 0 | 1.240 | 0.500 | 1.47 / 0.573 | none |
| M3_D1hypo (n=6) | b4@5.4M | 0.167 / 10.0 | 0.026 | 0 | 1.189 | 0.486 | 1.52 / 0.681 | none |
| M3_D1hypo (n=6) | b4@2.7M | 0.276 / 16.5 | 0.068 | 0 | 1.082 | 0.445 | 1.02 / 0.605 | none |
| M3_D1hypo (n=6) | b12dir(b10)@0.3M | 0.294 / 17.5 | 0.101 | 3 | 1.384 | 0.411 | 1.31 / 0.73 | none |
| M3_D3 (n=15) | b8_pub@2.7M | 0.047 / 2.8 | 0.220 | 0 | 1.478 | 0.095 | 1.44 / 0.584 | none |
| M3_D3 (n=15) | b4@5.4M | 0.144 / 8.6 | 0.394 | 0 | 1.420 | 0.428 | 324 / 1.7 | [4, 6, 7, 10, 12, 14] |
| M3_D3 (n=15) | b4@2.7M | 0.717 / 42.7 | 0.354 | 0 | 1.397 | 0.388 | 244 / 0.813 | [4, 7, 8, 10, 12] |
| M3_D3 (n=15) | b12dir(b10)@0.3M | 0.408 / 24.3 | 1.425 | 38 | 1.408 | 1.471 | 5.05e+03 / 0.854 | [4] |

Anchor itself vs truth: M3_D3 mse_vx max 1.44 (traj 2), mw_err max 0.095; M3_D1hypo traj 3/4 have
false = 2104/2477 and mw_err 0.40/0.50 for **every** arm including the anchor — a property of the
D1-hypocenter cross-test, not of any checkpoint.

## Per-trajectory (M3_D3, vs truth: mse_vx and Mw error; vs anchor: dRT in DT, dMw)

| traj | b8_pub mse_vx / mw_err | b4@5.4M mse_vx / mw_err / dRT / dMw | b4@2.7M mse_vx / mw_err / dRT / dMw | b12dir(b10)@0.3M mse_vx / mw_err / dRT / dMw |
|---|---|---|---|---|
| 0 | 0.142 / 0.003 | 0.132 / 0.013 / 2.3 / -0.008 | 0.29 / 0.029 / 6.9 / -0.025 | 0.443 / 0.009 / 6.2 / -0.005 |
| 1 | 0.577 / 0.092 | 0.548 / 0.071 / 2.8 / -0.013 | 0.411 / 0.028 / 20.9 / -0.057 | 0.417 / 0.024 / 24.3 / -0.061 |
| 2 | 1.44 / 0.095 | 1.23 / 0.035 / 5.1 / -0.031 | 1.18 / 0.100 / 42.7 / +0.034 | 1.34 / 0.045 / 12.1 / -0.021 |
| 3 | 0.733 / 0.057 | 0.816 / 0.064 / 3.9 / +0.010 | 0.495 / 0.007 / 20.8 / -0.061 | 1.27 / 0.039 / 17.4 / -0.014 |
| 4 | 0.584 / 0.048 | 142 / 0.414 / 3.7 / +0.368 | 138 / 0.291 / 11.2 / +0.245 | 5.05e+03 / 1.471 / 10.5 / +1.425 |
| 5 | 0.231 / 0.062 | 0.295 / 0.053 / 1.4 / +0.010 | 0.545 / 0.076 / 5.5 / -0.013 | 0.832 / 0.026 / 4.5 / +0.037 |
| 6 | 0.42 / 0.022 | 62 / 0.282 / 2.4 / +0.301 | 1.1 / 0.130 / 15.2 / -0.110 | 0.696 / 0.017 / 6.6 / +0.003 |
| 7 | 0.851 / 0.062 | 324 / 0.358 / 3.9 / +0.077 | 33.6 / 0.134 / 7.4 / -0.147 | 0.96 / 0.009 / 10.5 / -0.290 |
| 8 | 0.839 / 0.060 | 1.7 / 0.152 / 2.1 / +0.093 | 20.4 / 0.166 / 6.0 / +0.107 | 1.29 / 0.008 / 6.4 / -0.068 |
| 9 | 0.941 / 0.028 | 3.39 / 0.238 / 3.0 / +0.267 | 0.585 / 0.046 / 6.3 / -0.017 | 1.02 / 0.026 / 8.8 / +0.003 |
| 10 | 0.229 / 0.073 | 54.8 / 0.228 / 1.3 / +0.301 | 7.53 / 0.021 / 5.1 / +0.094 | 0.518 / 0.055 / 4.0 / +0.019 |
| 11 | 0.626 / 0.048 | 0.386 / 0.004 / 1.6 / -0.041 | 0.791 / 0.068 / 10.4 / -0.113 | 0.795 / 0.028 / 3.4 / -0.073 |
| 12 | 0.548 / 0.032 | 212 / 0.428 / 3.0 / +0.394 | 244 / 0.388 / 8.4 / +0.354 | 0.776 / 0.094 / 8.0 / -0.128 |
| 13 | 0.743 / 0.013 | 0.787 / 0.021 / 1.0 / +0.006 | 0.813 / 0.053 / 6.0 / -0.068 | 0.854 / 0.044 / 5.1 / -0.059 |
| 14 | 0.47 / 0.003 | 92 / 0.327 / 8.6 / +0.327 | 0.521 / 0.045 / 10.4 / -0.046 | 1.77 / 0.112 / 12.8 / +0.111 |

b4@5.4M runaway onset (first step with max|v| > 5x truth peak): traj 4 step 201 (3.4 s), traj 6
step 223 (3.7 s), traj 7 step 126 (2.1 s), traj 10 step 317 (5.3 s), traj 12 step 154 (2.6 s),
traj 14 step 150 (2.5 s); truth rupture ends at step 336–402 (5.6–6.7 s) on those trajectories.
Trajectory 7 is the board's excluded late-tail-pulse trajectory; its anchor dMw of -0.220 for
b8_pub is the known fast-path artefact, not new.

## Historical eager b4 rollouts on disk (published code, same M3_D3 test set, scored with `gate.metrics`)

| checkpoint | trajectories with mse_vx > 5 | mse_vx max / median | mw_err max | pred peak \|v\| max (m/s) |
|---|---|---|---|---|
| b4 @ 2.7M | 10, 12 | 370 / 0.587 | 0.436 | 1179 |
| b4 @ 5.0M | 4, 6, 12 | 209 / 0.909 | 0.434 | 1410 |
| b4 @ 7.0M | 4, 12, 14 | 645 / 0.828 | 0.503 | 1601 |

## Open questions

- Cause not isolated: lr 1e-4 (3.3x published) with 5M decay vs batch 4 itself. The sweep's
  b4 arm at lr 3e-5 / 20M decay answers this directly; until then do not cite b4 @ 5.4M as a
  batch-size data point.
- Whether `--rollout_batch_size=64` changes *which* trajectories blow up (it did not change
  *whether*): fast b4@2.7M blew up on 4, 7, 8, 12; eager historical b4@2.7M on 10, 12.
- The gate has no high-side guard; `collapsed` stayed False on every runaway trajectory.

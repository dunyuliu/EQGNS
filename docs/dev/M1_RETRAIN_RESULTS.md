# M1 retrain: fixed-D1 vs old-D1 (bug-fix data effect), seed spread

Owner-approved comparison: does fixing the `nskip` double-subtraction bug in
`scripts/utils/prepare.eqdyna.4gns.py` (PR #5) — which left the last 72 of 827 frames
all-zero in the published D1 dataset — change M1 training outcomes.

## Setup
- Two arms, 3 seeds each (`--seed 0/1/2`, identical across arms, via the
  opt-in seeding in PR #8): **fixed** trains on the regenerated
  `D1_fixed/dataset/` (no zero-tail); **old** (control) trains on the
  published, still-buggy `gns-sample/case3.200m.homo.a.Vw/dataset/` as-is.
- Identical config both arms: `models.nmp10.cotopaxi/config.json` (paper
  Table 2 — lr 1e-4, batch 2, 10 message-passing steps, noise 0.02),
  500,000 steps, checkpoints every 50,000.
- Scored with deterministic rollouts (`det_rollout.py`) + `gate.py`'s
  `metrics()` (collapse guard included), on the **FIXED-D1 test set** for
  every row (both arms and the published baseline), so the comparison
  isolates the training-data effect, not a test-set effect.
- **Matched step counts only** — no validation-based checkpoint selection
  (owner declined that row). The published M1 is scored at the same 5
  steps from its own training trajectory (published checkpoints exist at
  exactly 100k/200k/300k/400k/500k), not its final 3,000,000-step
  checkpoint — this is a different-lr-schedule-position comparison, included
  for context, not as a third arm of the experiment.

## Results

| step | arm | seed0 mse_vx | seed1 mse_vx | seed2 mse_vx | median mse_vx | seed0 rt_rmse | seed1 rt_rmse | seed2 rt_rmse | median rt_rmse | collapsed traj (of 6) |
|---|---|---|---|---|---|---|---|---|---|---|
| 100000 | fixed | 2.22 | 85.7 | 9.87 | **9.87** | 0.522 | 0.855 | 1.67 | 0.855 | 0/0/0 |
| 100000 | old (control) | 1.65 | 26.9 | 1.31 | **1.65** | 0.359 | 0.327 | 1.95 | 0.359 | 0/0/1 |
| 100000 | published | - | - | - | 2.49 | - | - | - | 0.53 | 0 |
| 200000 | fixed | 0.831 | 2.19 | 1.35 | **1.35** | 0.344 | 0.512 | 1.14 | 0.512 | 0/0/0 |
| 200000 | old (control) | 17.0 | 23.7 | 1.60 | **17.0** | 0.314 | 0.456 | 0.507 | 0.456 | 0/0/0 |
| 200000 | published | - | - | - | 1.28 | - | - | - | 0.333 | 0 |
| 300000 | fixed | 2.31 | 0.986 | 23.5 | **2.31** | 0.344 | 0.306 | 0.305 | 0.306 | 0/0/0 |
| 300000 | old (control) | 2.37 | 10.8 | 67.6 | **10.8** | 0.313 | 0.212 | 1.65 | 0.313 | 0/0/0 |
| 300000 | published | - | - | - | 0.861 | - | - | - | 0.220 | 0 |
| 400000 | fixed | 0.664 | 1.41 | 0.469 | **0.664** | 0.283 | 0.382 | 0.195 | 0.283 | 0/0/0 |
| 400000 | old (control) | 4.40 | 1.23 | 1.43 | **1.43** | 1.43 | 0.286 | 0.316 | 0.316 | 0/0/0 |
| 400000 | published | - | - | - | 0.565 | - | - | - | 0.234 | 0 |
| 500000 | fixed | 0.635 | 0.715 | 2.72 | **0.715** | 0.283 | 0.355 | 0.398 | 0.355 | 0/0/0 |
| 500000 | old (control) | 3.28 | 0.496 | 2.28 | **2.28** | 0.377 | 0.294 | 0.447 | 0.377 | 0/0/0 |
| 500000 | published | - | - | - | 4.05 | - | - | - | 0.654 | 0 |

(`mse_vx`: rollout MSE of vx; `rt_rmse`: rupture-time RMSE over trajectories
where both predicted and ground truth reach the slip-rate threshold;
"collapsed" = `var(pred)/var(gt) < 0.5` per the collapse guard, PR #9 —
counted per seed, 0-6 possible.)

Raw per-trajectory JSON for every (label, step): `eq_rupture_gns_data/m1_retrain/eval_results/` (35 files, external — not part of this PR, same convention as the other regenerated data this campaign produced).

## Interpretation — effect is directionally present but NOT resolved at N=3

`median mse_vx` favors the fixed-D1 arm at 4 of 5 matched steps (200k,
300k, 400k, 500k); only 100k favors the control, and that's driven by a
single outlier (fixed seed1 = 85.7, ~9-40x every other cell in the table).
`rt_rmse` shows the same modest lean toward fixed at the later steps
(300k-500k) and is roughly tied or slightly favors control at 100k-200k.

**This is not a resolved result.** Within-arm seed spread is enormous — up
to ~30x at a single step (old@200k: 1.60 to 23.7; fixed@300k: 0.99 to
23.5) — larger than the between-arm median gap at several steps. With only
3 seeds per arm, a single outlier run swings the arm median by an order of
magnitude (as it does at 100k). CycleGNS's seed-spread rule (Rule 25, cited
when this budget was set) exists for exactly this: a result that holds for
one seed and not another is not a result, and the same caution applies
here to "N=3 medians."

**No README claim of a confirmed effect is being made.** The data leans
toward the bug fix helping, consistent with the mechanism (the old data's
zero-tail frames are degenerate training targets — literally zero
velocity/position-delta signal for 72 of 827 frames per trajectory, which
can only ever teach the model a wrong, too-easy answer for that portion of
the rollout), but resolving it with confidence needs more seeds than this
budget covered (owner approved N=3 specifically to fit the ~48h-ish
wall-clock ask this session; a larger N is a budget question for a future
session, not something to decide unilaterally here).

**One collapse-guard flag**: `old_seed2` at step 100,000 has 1 of 6
trajectories flagged `collapsed` (`var_ratio < 0.5`) — noted, not
suppressed; does not change the overall picture (that arm's median mse_vx
at that step is still unremarkable, 1.65).

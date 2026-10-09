# M1 retrain (fixed-D1 vs old-D1), seed spread — status

**STOPPED, not running.** No process currently alive. Three runs (fixed-D1,
seeds 0/1/2) stopped at step 1,350,000 on 2026-10-03; the other three
(old-D1 control, seeds 0/1/2) never started past 500,000 steps. There is no
decision yet on resuming toward 3,000,000 steps; do not describe this as
in-progress until a resume is explicitly approved.

## What's done (500k, complete, both arms)

Owner-approved budget: 500,000 steps x 3 seeds per arm, 6 runs, comparing
training on the regenerated fixed-D1 dataset (`nskip` double-subtraction bug
fixed, PR #5) against the published, still-buggy old-D1 dataset. Identical
config both arms (`models.nmp10.cotopaxi/config.json`, paper Table 2: lr
1e-4, batch 2, 10 message-passing steps, noise 0.02). Scored with
deterministic rollouts + `gate.py`'s `metrics()` on the fixed-D1 test set for
every row. Full results table and interpretation:
[M1_RETRAIN_RESULTS.md](M1_RETRAIN_RESULTS.md).

## What stopped (500k -> 3M attempt)

After the 500k result came back inconclusive (seed spread up to ~30x,
larger than the between-arm gap at several steps), all 6 runs were resumed
toward 3,000,000 steps on GPU 1, 3 concurrent. Fixed-D1 seeds reached
1,350,000 steps before stopping on 2026-10-03; old-D1 seeds never resumed
past 500,000. No runs are currently alive on any GPU.

Resuming to 3M, and re-evaluating with outcome-robust metrics (rupture-time
error, arrest classification) instead of/alongside `mse_vx`, is a budget
question for a future session — not something to restart unilaterally.

## Run directories

`/home/utig5/dliu/eq_rupture_gns_data/m1_retrain/{fixed,old}_seed{0,1,2}/`
(`models/`, `rollouts/`, `train.log` each) for the 500k runs;
`m1_retrain/to3M/` for the stopped 3M attempt.

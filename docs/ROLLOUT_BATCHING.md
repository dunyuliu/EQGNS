# Batched rollout (`--rollout_batch_size`) and rollout sensitivity

**What:** `--rollout_batch_size B` (default 1) rolls out B same-length test trajectories as one disjoint graph (`meshnet/train.py:rollout_batched`).
- **Default B=1:** the original path, bit-identical. The paper-parity gate runs on it.
- **B>1:** the same math per trajectory. It is meant for evaluation, not for gating.

**Timing:** not yet measured. Owner rule: time only on an idle system, and no GPU has been idle.

## B=1 vs B=6 on the published M1 (3M), 6 M1 test trajectories, deterministic mode, 2026-10-01

| Quantity | Result |
|---|---|
| Step-0 difference (relative) | ~5e-8. Float32 rounding: larger matrices select different cuBLAS kernels, so sums run in a different order. |
| Growth during active rupture | ~10^5 by step 300; peak 0.44 (traj 2) and 0.21 (traj 4). It decays to ~1e-9 after arrest. |
| Gate metrics, B=6 vs B=1 | 4/6 trajectories within 1e-4; traj 2 differs 1e-3 and traj 4 3% (mse_vx). So B>1 cannot pass the 1e-4 gate. |
| Rupture time | Max shift 0.017 s (traj 2) and 0.067 s (traj 4); mean ~1e-4 s; the same nodes rupture. |
| Rupture-time contours (1 s) | Visually identical (`figs/rollout_batched_vs_single_rupture_time.png`). |
| On-fault stations (paper's 6) | Traj 2: identical. Traj 4: the main pulse is identical; a late second pulse differs (e.g. 3.5 vs 4.3 m/s). EQdyna has no such pulse: traj 4 arrests early, and both GNS versions spuriously re-rupture (`figs/rollout_batched_vs_single_stations.png`). |

## What it means
- **The GNS amplifies rounding-size perturbations by 10^5–10^7 during active rupture.** The same mechanism likely drives the ~30× seed spread of the M1 retrain at 500k (training losses there matched to <2×).
- **The divergence concentrates where the model is already wrong** (spurious re-rupture in arrest cases), not where it matches EQdyna.
- **Pointwise slip-rate MSE is fragile.** Rupture time and arrest outcome are robust; prefer them for model comparison.
- **Open test:** does perturbation spread flag model error without ground truth? See the next section.

The figures use the existing helpers in `utils/plot.rupture.dynamics.py` (`load_rollout_data`, `get_rupture_time`, `extract_timeseries`, and the stations from `process_member`).

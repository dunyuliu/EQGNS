# Rollout and analysis

## Single rollout

```shell
python3 -m meshnet.train --mode=rollout \
  --data_path=<working_dir>/dataset/ \
  --model_path=<working_dir>/models.<suffix>/ \
  --output_path=<working_dir>/rollouts.<suffix>/ \
  --model_file=model-3000000.pt --train_state_file=train_state-3000000.pt
```

The rollout in `meshnet/train.py` caches the graph topology and edge features
across timesteps, roughly doubling inference speed relative to the published
version (`meshnet/train.py.published`); results are mathematically identical.
See `CLAUDE.md` for the full list of optimizations and options
(`compute_loss`, `disable_tqdm`, `use_compile`, `use_amp`).

## Sweeps over models and checkpoints: `scripts/scenario.rollout.py`

`scripts/scenario.rollout.py` drives rollouts across many trained models. Edit the
`case` selector and the `model_suffixes` dict (working_dir → list of model
suffixes), set `model_id` (checkpoint step) and `gpu_id`, then:

```shell
python3 scripts/scenario.rollout.py
```

By default it shells out to `scripts/run.process.gns.py --mode rollout` per model.
Set `use_batch_rollout = True` to route through the batched engine instead.

## Batched inference: `meshnet/batch_rollout.py`

Processes multiple trajectories/models per GPU pass:

```shell
python3 meshnet/batch_rollout.py --mode=rollout \
  --working_dir <working_dir> \
  --model_suffix <suffix>[,<suffix2>,...] \
  --model_ids 3000000 --gpu_id 0 --batch_size 4
```

Optional: `--data_path` (defaults to `<working_dir>/dataset/`), `--pkl_path`
to re-process existing rollout pickles, `--output_path` (defaults to
`<working_dir>/rollouts.<suffix>/<model_file>/`).

## Fast opt-in rollout: `--rollout_fast`

`--rollout_fast {off,fp32,tf32,fp16,bf16}` (default `off`) routes the rollout through
`meshnet/fast_rollout.py` instead of the default `predict_velocity` loop: normalizers,
one-hot and the edge encoder are folded into constants, the edge MLP's first layer is
factorized per node, and the step is `torch.compile`d and CUDA-graph captured. Needs
`INPUT_SEQUENCE_LENGTH == 1`. On the 2D paper models (~4.7k nodes, 826 steps, idle GPU),
it is about 5x faster than eager fp32 (~1.8-1.9 ms/step/trajectory vs ~8-10 ms).

```shell
python3 -m meshnet.train --mode=rollout --rollout_fast=tf32 \
  --data_path=<working_dir>/dataset/ --model_path=<working_dir>/models.<suffix>/ \
  --output_path=<working_dir>/rollouts.<suffix>/ \
  --model_file=model-3000000.pt --train_state_file=train_state-3000000.pt
```

**Use `tf32`.** Gated against the full M1_D1/M2_D3/M3_D3 test sets
(`tests/paper_parity/gate.py fast`, see `tests/paper_parity/README.md` for the full
results table): `fp32` and `tf32` pass on every case within the eager run-to-run noise
floor; `fp16` and `bf16` do not (`fp16` misses on M2_D3 slip-rate MSE; `bf16` misses on
M2_D3 and M3_D3 rupture-time RMSE). The default rollout path (`--rollout_fast=off`) is
unaffected and stays the one gated by `gate.py run`.

## Batched rollout: `--rollout_batch_size`

`--rollout_batch_size B` (default 1) rolls out B same-length test trajectories as one
disjoint graph (`meshnet/train.py:rollout_batched`).
- **Default B=1:** the original path, bit-identical. The paper-parity gate runs on it.
- **B>1:** the same math per trajectory, meant for evaluation throughput, not for gating.

Measured 2026-10-01, published M1 (3M), 6 M1 test trajectories, deterministic mode,
B=1 vs B=6:

| Quantity | Result |
|---|---|
| Step-0 difference (relative) | ~5e-8. Float32 rounding: larger matrices select different cuBLAS kernels, so sums run in a different order. |
| Growth during active rupture | ~10^5 by step 300; peak 0.44 (traj 2) and 0.21 (traj 4). Decays to ~1e-9 after arrest. |
| Gate metrics, B=6 vs B=1 | 4/6 trajectories within 1e-4; traj 2 differs 1e-3 and traj 4 3% (mse_vx). B>1 cannot pass the 1e-4 gate. |
| Rupture time | Max shift 0.017 s (traj 2) and 0.067 s (traj 4); mean ~1e-4 s; the same nodes rupture. |
| Rupture-time contours (1 s) | Visually identical (`figs/rollout_batched_vs_single_rupture_time.png`). |
| On-fault stations (paper's 6) | Traj 2 identical; traj 4's main pulse identical, a late second pulse differs (e.g. 3.5 vs 4.3 m/s) — EQdyna has no such pulse, traj 4 arrests early and both GNS versions spuriously re-rupture (`figs/rollout_batched_vs_single_stations.png`). |

What it means: the GNS amplifies rounding-size perturbations by 10^5-10^7 during active
rupture (likely the same mechanism behind the ~30x seed spread of the M1 retrain at
500k, where training losses matched to <2x); the divergence concentrates where the model
is already wrong (spurious re-rupture in arrest cases); pointwise slip-rate MSE is
fragile, prefer rupture time and arrest outcome for model comparison. Figures use the
existing helpers in `scripts/utils/plot.rupture.dynamics.py` (`load_rollout_data`,
`get_rupture_time`, `extract_timeseries`, `process_member`).

## Rendering and analysis

- `python3 -m meshnet.render --rollout_dir=<dir> --rollout_name=<name>` —
  gif animation of predicted vs. ground-truth fields (`scripts/render.cpu.sh`,
  `scripts/render.sh` wrap this).
- `scripts/utils/plot.rupture.dynamics.py` — rupture-time contours, slip-rate
  time series, comparison against EQdyna ground truth, SCEC-style benchmark
  outputs.
- `scripts/utils/case3.200m.visualize.hypocenters.py`,
  `scripts/utils/case4.200m.multi.stress.visualize.datasets.py` — dataset/scenario
  visualization.
- `scripts/utils/convert.mp4.to.gif.py` — convert rendered movies for the README.

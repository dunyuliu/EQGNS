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

## Sweeps over models and checkpoints: `scenario.rollout.py`

`scenario.rollout.py` drives rollouts across many trained models. Edit the
`case` selector and the `model_suffixes` dict (working_dir → list of model
suffixes), set `model_id` (checkpoint step) and `gpu_id`, then:

```shell
python3 scenario.rollout.py
```

By default it shells out to `run.process.gns.py --mode rollout` per model.
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

## Rendering and analysis

- `python3 -m meshnet.render --rollout_dir=<dir> --rollout_name=<name>` —
  gif animation of predicted vs. ground-truth fields (`render.cpu.sh`,
  `render.sh` wrap this).
- `utils/plot.rupture.dynamics.py` — rupture-time contours, slip-rate
  time series, comparison against EQdyna ground truth, SCEC-style benchmark
  outputs.
- `utils/case3.200m.visualize.hypocenters.py`,
  `utils/case4.200m.multi.stress.visualize.datasets.py` — dataset/scenario
  visualization.
- `utils/convert.mp4.to.gif.py` — convert rendered movies for the README.

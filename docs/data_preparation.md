# Data preparation: EQdyna → GNS trajectories

EQGNS trains on 2D dynamic rupture simulations produced by
[EQdyna](https://github.com/EQDYNA/EQdyna.git). The conversion scripts live in
`utils/`:

| Script | Purpose |
|---|---|
| `utils/prepare.eqdyna.4gns.py` | Main converter: reads EQdyna on-fault outputs (netCDF), assembles trajectories, writes `train.npz` / `valid.npz` / `test.npz` and `metadata.json`. Also plots rupture dynamics and can generate `SCECRuptureTime.txt` for benchmarking. |
| `utils/prepare.fractal.stress.eqdyna.4gns.py` | Variant for fractal initial-stress scenarios. |
| `utils/prepare.case3.200m.others.py` | Helper for the case3 200 m generalization set. |

Both prepare scripts select the case via a `case` variable near the top
(e.g. `'4.200m.multi.stress'`) — edit it before running.

## Trajectory format (mesh-based domain)

Each trajectory in the `.npz` is a Python dictionary:

- `pos`: `(ntimestep, nnodes, ndims)` node coordinates
- `node_type`: `(ntimestep, nnodes, ntypes)` — includes an earthquake-specific
  type for high-stress asperity nodes in addition to the upstream types
- `node_property`: scalar initial on-fault stress condition per node
  (EQGNS extension; consumed as an extra input channel by `meshnet/train.py`)
- `velocity`: `(ntimestep, nnodes, ndims)` — for rupture problems this carries
  the slip-rate field the model learns to advance
- `cells`: mesh connectivity

`metadata.json` in the dataset directory stores sequence length, `dt`, and
normalization statistics, following the upstream GNS convention (see the main
README, section "Datasets").

## Expected directory layout

The training/rollout drivers assume a *working directory* per case:

```
<working_dir>/               e.g. gns-sample/case4.200m.multi.stress.homo.a.Vw/
├── config.json              model/system configuration (see docs/training.md)
├── dataset/                 train.npz, valid.npz, test.npz, metadata.json
├── models.<suffix>/         checkpoints, one directory per training run
└── rollouts.<suffix>/       rollout .pkl outputs
```

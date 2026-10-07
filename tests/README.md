# EQ-GNS test suite

This mirrors the test pyramid used in `EQdyna.2Dcycle`
(`test_system/test_all_quick.py` + `verify.test.py`) and `EQdyna`
(`testsys/unit` + `testsys/regression`), adapted for a GNN surrogate:
fast unit tests at the base, integration tests for module seams, one
seeded end-to-end pipeline test with a committed golden file, and a
physical-behaviour tier for the rollout invariants the model claims to
satisfy.

`tests/test_*.py` (upstream `geoelements/gns` unit tests for the
particulate-domain `gns/` package) were here before this session and are
unchanged. Everything below is new, under `meshnet` naming, and uses only
a tiny synthetic dataset built on the fly by
`tests/fixtures/meshnet/synth.py` -- never the (not checked out, 249GB)
`gns-sample` data.

## Tiers

| Tier | Marker | Files | What it covers |
|---|---|---|---|
| 1. Unit | `unit` | `test_meshnet_unit_*.py` | `data_loader` shapes, `config.json` keys actually changing architecture, `Normalizer` accumulate/inverse identities, `get_velocity_noise` masking + std, `MeshSimulator` save/load and the `predict_velocity` additive identity |
| 2. Integration | `integration` | `test_meshnet_integration_train_rollout.py` | real `SamplesDataset` -> `FaceToEdge`/`Cartesian`/`Distance` transformer -> `predict_acceleration` -> backward -> `optimizer.step()`; checkpoint save -> fresh simulator -> `load()` -> identical predictions; `meshnet.train.rollout()` on a `TrajectoriesDataset` example |
| 3. End-to-end | `e2e`, `slow` | `test_meshnet_e2e_golden.py` | the real CLI (`python3 -m meshnet.train`, via a seeded wrapper) run `--mode=train` then `--mode=rollout` on a tiny deterministic dataset, diffed against a committed golden file |
| 4. Physical-behaviour | `physical` | `test_meshnet_physical.py` | additive-acceleration identity through a real graph, zero-forcing -> zero-response asymptotic limit (mutation-verified to actually depend on the weights, not just the untrained normalizer floor), velocity-noise std vs `noise_std` |
| 5. Data-prep guard | `dataprep` | `test_dataprep_prepare_eqdyna.py`, `test_dataprep_prepare_fractal_stress.py` | `scripts/utils/prepare.eqdyna.4gns.py` / `scripts/utils/prepare.fractal.stress.eqdyna.4gns.py`'s shared `create_train_data()` EQdyna-output -> npz conversion, on a tiny synthetic 4-node case built by `tests/fixtures/dataprep/synth_case.py`: frame count, shapes/dtypes, no all-zero frames (regression guard for the zero-tail bug fixed in PR #5 / commit 1afb989), the `nskip` raw-file-to-frame offset, fractal-stress node-property lookup, and train/valid/test split disjointness |

## Running

```bash
source venv/bin/activate

# Everything this session added, plus the pre-existing gns/ unit tests:
pytest tests/ -q

# Fast local loop (skip the ~15s end-to-end golden test):
pytest tests/ -m "not slow" -q

# Just the meshnet tiers:
pytest tests/ -k meshnet -q

# One tier:
pytest tests/ -m unit -q
pytest tests/ -m integration -q
pytest tests/ -m e2e -q
pytest tests/ -m physical -q
pytest tests/ -m dataprep -q
```

CI (`.github/workflows/tests.yml`) runs `pytest tests/ -m "not slow" -q`
(fast tier) and then the full, unfiltered `pytest tests/ -q` -- the `slow`
marker is for skipping locally, not for skipping in CI (see the pattern of
`not compute_loss=False`-style opt-outs documented in `CLAUDE.md`:
convenient locally, never the CI default).

Total measured runtime of the full `tests/` suite (63 tests, single CPU
thread, this repo's `venv`): ~27s, of which ~13s is the two `python3 -m
meshnet.train` subprocess launches in the e2e tier (import + torch/PyG
startup dominates, not the tiny model itself); the 10-test `dataprep` tier
adds ~3s.

## The golden file (tier 3)

`tests/golden/meshnet_e2e_rollout_golden.npz` was generated once with:

```bash
python3 tests/fixtures/meshnet/generate_golden.py
```

using the fixed seed `20260101` (see
`tests/fixtures/meshnet/seeded_pipeline_cli.py`), an 8-step training run,
and the synthetic 12-node mesh / 24-timestep trajectory built by
`tests/fixtures/meshnet/synth.py` with `DATASET_SEED=555` (see
`test_meshnet_e2e_golden.py`). The synthetic dataset itself is *not*
committed as a binary -- it is 100% reproducible from the seed, so only
the (much smaller) resulting rollout array is committed.

If `test_seeded_mini_pipeline_matches_golden_rollout` fails after a
change to `meshnet/`:
1. First assume the code broke something and go find the bug -- this is
   the entire point of the test.
2. Only if the change is a confirmed, intentional, numerically-different
   (but correct) behaviour change, regenerate the golden with the command
   above and say so explicitly in the commit message (e.g. "regenerate
   meshnet e2e golden: <why>"). Never regenerate to silence a failure
   whose cause you have not identified.

## What this suite deliberately does not cover (flagged, not built)

- **Dimensional consistency / unit tracking.** The velocity/pressure/
  node_property fields carry no explicit units in the codebase (no
  `pint`/`astropy.units`), so a dimensional-consistency test would have
  nothing to check against without first threading units through
  `meshnet/`, which is a production-code change out of this session's
  scope. Recommended for `kai-fischer` if unit-bearing fields are ever
  introduced.
- **Symmetry / equivariance tests** (mirror the mesh, expect mirrored
  output). MeshGraphNets-style GNNs built on raw relative-position edge
  features are not architecturally guaranteed to be reflection-equivariant
  pre-training, so an untrained-weight symmetry test would be
  underdetermined (could pass or fail by chance depending on random init)
  -- it would not be a real invariant, just noise. A meaningful version of
  this test needs a *trained* model and belongs with a real training run,
  not this fast/CPU-only suite.
- **Mesh/step refinement (convergence order).** No analytical solution
  exists for the learned rollout (it is a surrogate for the FEM solver,
  not a PDE solved to a known order), so classical h/2, h/4 refinement
  doesn't apply here the way it would to `EQdyna`'s finite-element solver.
  If a manufactured-solution benchmark trajectory becomes available,
  this belongs in the physical-behaviour tier next.
- **GPU-path tests** (AMP / `torch.compile`, both documented in
  `CLAUDE.md`). Out of scope per the CPU-only constraint on this session;
  `test_pytorch_cuda_gpu.py` already exists upstream for CUDA smoke-testing
  and is a reasonable place to extend if a GPU CI runner is ever added.
- **`meshnet/batch_rollout.py` and `scripts/scenario.rollout.py`.** Real
  production entry points with no test coverage at any tier. Flagged for
  a follow-up session; not attempted here to keep this session's scope to
  the `train.py`/`rollout()` path named in the brief.

## Findings on existing production code (not fixed here -- test-only agent)

- `meshnet/noise.py::get_velocity_noise` slices `graph.x[:, 1:3]` for
  `velocity_sequence` and `graph.x[:, 0]` for `type`. Given the node
  feature layout built in `data_loader.py`
  (`[node_type, node_property, vx, vy, pressure, time]`), the velocity
  columns are actually `2:4`, not `1:3` -- `1:3` grabs
  `[node_property, vx]`. This is currently harmless only because
  `velocity_sequence` is used solely for its `.shape` (both slices are
  `(nnodes, 2)`), never its values; if that function is ever changed to
  use the values, this off-by-one will produce a silent bug. Flagged for
  `lars-eriksson`/`kai-fischer`, not fixed here (production code).
- `meshnet/train.py::train()` builds the data path for `train.npz`/
  `valid.npz` as `f'{FLAGS.data_path}/{FLAGS.mode}.npz'` (adds a `/`),
  while `predict()` builds `f"{FLAGS.data_path}{split}.npz"` (no `/`).
  Both work only because `FLAGS.data_path` is conventionally passed with
  a trailing slash already (giving a harmless `//`) -- but there is no
  validation of that convention anywhere, so a `--data_path` without a
  trailing slash breaks `predict()`/`rollout` silently (wrong path,
  `FileNotFoundError` deep in `np.load`, not at the flag boundary).
  Consider a `--data_path` normalization step or a flag validator;
  flagged, not fixed here.

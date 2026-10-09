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
unchanged, except for `test_pytorch.py`, `test_torch_geometric.py`, and
`test_pytorch_cuda_gpu.py` -- trivial framework smoke scripts (no `test_`
functions, zero tests collected) that exercised neither `gns/` nor
`meshnet/`, removed as part of the `test-suite-overhaul` cleanup.

A later slice of the same cleanup removed three more upstream leftovers,
confirmed by grep to have no caller anywhere in `meshnet/` or `scripts/`
(only `gns/train.py`/`gns/train_multinode.py`, which this project does not
run, import them):
- `test_data_loader.py` (`gns.data_loader.SamplesDataset`/
  `TrajectoriesDataset` -- `meshnet/train.py` uses `meshnet.data_loader`'s
  own, separate classes of the same name instead, covered by
  `test_meshnet_unit_data_loader.py` / `test_meshnet_integration_train_rollout.py`).
- `test_learned_simulator.py` (`gns.learned_simulator.LearnedSimulator`,
  the particle-domain simulator class -- `meshnet/learned_simulator.py`'s
  `MeshSimulator` is a different class, covered by
  `test_meshnet_unit_learned_simulator.py`).
- `test_noise_utils.py` (`gns.noise_utils.get_random_walk_noise_for_position_sequence`
  plus `gns.learned_simulator.time_diff` -- `meshnet/noise.py`'s
  `get_velocity_noise` is the function `meshnet/train.py` actually calls,
  covered by `test_meshnet_unit_noise.py` / `test_meshnet_physical.py`; this
  file also had its one quantitative assertion, a noise-std check, commented
  out, i.e. it was down to a shape check and a zero-check on code nothing
  here depends on).

`test_graph_network.py` and `test_message_edge_features.py`, despite the
same `gns.*` origin, are kept (not leftovers): `gns.graph_network`'s
`EncodeProcessDecode`/`InteractionNetwork` are imported directly by
`meshnet/learned_simulator.py` (`from gns import graph_network`) and are
the actual message-passing backbone of the production `MeshSimulator`
model -- these two files are this repo's only test coverage of that file.
Everything below is new, under `meshnet` naming, and uses only
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
| 6. Training-guard tier 1 (`training_golden`) | `training_golden`, `slow` | `test_training_golden.py` | real-data, real-architecture training-path gate: N=10 steps of the real `python3 -m meshnet.train` CLI on the real, published D1 dataset (`data/gns-sample/case3.200m.homo.a.Vw/dataset/`), real M1 architecture (10 message-passing steps, 128 latent dim), per-step train/valid loss vs a committed reference, tight tolerance |
| 7. Training-guard tier 2 (`convergence_gate_nightly`) | `convergence_gate_nightly`, `nightly`, `slow` | `test_convergence_gate_nightly.py` | small-budget (N=30 steps) real training on real D1 from scratch, then a real rollout on a CPU-time-truncated real D1 test trajectory, scored with `tests/paper_parity/gate.py`'s own metrics (`mse_vx`, `rt_rmse`, `missed`, `false`) vs a committed reference |

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

# Training-guard tiers (need data/gns-sample/ checked out -- see below;
# auto-skip, not fail, anywhere else, including CI):
pytest tests/ -m training_golden -q          # tier 1, ~2 min
pytest tests/ -m convergence_gate_nightly -q # tier 2, ~6 min
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
adds ~3s. This excludes the two training-guard tiers added after this
measurement (tiers 6-7 below): on a machine WITHOUT `data/gns-sample/`
checked out they add two near-instant skips; on a machine WITH it checked
out (this is most machines that would run the full, unfiltered `pytest
tests/ -q`, since that data is what makes them "with"), they actually run
and add ~8 minutes (~2 min tier 1 + ~6 min tier 2) -- run them explicitly
(`-m training_golden` / `-m convergence_gate_nightly`) rather than as part of
every routine full-suite invocation.

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

## Training-guard tiers (tiers 6-7, PATHWAY_FORWARD.md row `training-guard`)

Unlike every tier above, these two run against the REAL, published D1 dataset
(`data/gns-sample/case3.200m.homo.a.Vw/dataset/`, PROJECT_RULES.md rule 3: a
real, gitignored, 249GB-class directory that only exists on a machine that
fetched `data/gns-sample/` -- never committed, never symlinked over, never
present in GitHub Actions CI) and the real M1 model architecture (10
message-passing steps, 128 latent dim -- `tests/fixtures/training_golden/
config.json`, copied verbatim from `data/gns-sample/.../models.nmp10.cotopaxi
/config.json` except for a test-only `loss_report_step` override). Both
auto-skip (`fixtures/training_golden/common.py::require_d1_dataset`, no
`pytest` flag needed) if that dataset is not present -- the same situation
`tests/paper_parity`'s tests are in, except those need an explicit
`--paper-parity` flag too; these two don't, since there is no risk of an
expensive real-data run happening by accident where the data doesn't exist.

**Strict mode:** `pytest -m training_golden` (or `-m
convergence_gate_nightly`) exits 0 when every selected test is skipped, so
a moved/renamed/missing dataset path would silently "pass" a context that
expects the data to be checked out (e.g. a future nightly job on a machine
provisioned specifically to run these tiers). Set
`EQGNS_TRAINING_GUARD_REQUIRE_DATA=1` to turn that skip into a hard,
reported FAILURE instead:

```bash
EQGNS_TRAINING_GUARD_REQUIRE_DATA=1 pytest tests/ -m training_golden -q
```

**Falsify checks** (`tests/test_training_golden_falsify.py`,
`tests/test_convergence_gate_nightly_falsify.py`, markers
`training_golden_falsify` / `convergence_gate_nightly_falsify`): re-run the
respective pipeline with `lr_init` perturbed +10% and assert the golden/band
comparison FAILS -- the self-verifying, committed counterpart to
`tests/paper_parity/gate.py falsify`, so the "this oracle is sensitive to
real regressions" claim does not rest on a commit message alone. Not part of
`-m training_golden` / `-m convergence_gate_nightly` (same separation as
`gate.py run` vs `gate.py falsify`) -- run explicitly:

```bash
pytest tests/ -m training_golden_falsify -q
pytest tests/ -m convergence_gate_nightly_falsify -q
```

Why two tiers, and why they exist alongside the tiny synthetic e2e golden
(tier 3) and the A/B determinism test: the synthetic golden proves the
pipeline runs correctly end-to-end on a model sized for sub-second CI
runtime (8-wide/2-layer), and the A/B test proves the harness is
bit-reproducible, but neither carries an oracle for what the loss or rollout
SHOULD look like on the real model at the real data scale -- a bug that
changes behaviour only at realistic model/data size (e.g. an off-by-one that
only bites with 10 message-passing steps, not 2) would slip past both.

**Tier 1 -- `training_golden`** (`test_training_golden.py`): N=10 steps of
the real training loop, per-step train/valid loss vs
`tests/golden/training_golden_tier1.json`, tolerance `rtol=atol=1e-5`
(text-round-trip tolerance only -- CPU + `torch.use_deterministic_algorithms
(True)` is bit-reproducible on this harness, proven by
`test_ab_seeded_determinism.py`'s exact-equality assertion). ~2 min, CPU-
forced (`CUDA_VISIBLE_DEVICES=""` in the subprocess env -- `meshnet/train.py`
has no CPU-force flag of its own and always prefers CUDA when available; the
reference was generated on CPU to match CI's CPU-only torch build, so the
test forces CPU too even on a machine with GPUs).

**Tier 2 -- `convergence_gate_nightly`** (`test_convergence_gate_nightly.py`):
N=30 steps of real training from scratch (no resume from the published
3M-step checkpoint), then a real rollout on a CPU-time-truncated (first 40 of
827 frames) real D1 test trajectory, scored with
`tests/paper_parity/gate.py`'s own `metrics()` (`mse_vx`, `rt_rmse`,
`missed`, `false` -- same rupture-time convention, PROJECT_RULES.md rule 7:
`SLIPRATE_THRESHOLD=0.1` m/s, `dt=0.0167777s`) against
`tests/golden/convergence_gate_nightly_tier2.json`, `rtol=1e-4` for the
continuous metrics and exact match for the integer counts (again a tight
tolerance around a reproducible reference, not a statistical band -- both
runs are CPU-forced and deterministic, unlike `gate.py`'s `REL_TOL`, which
exists specifically to absorb GPU nondeterminism). ~6 min. At this step
budget the model is essentially untrained, so the recorded reference is
itself `"collapsed": true` (near-constant rollout, `var_ratio` well under
`gate.py`'s `COLLAPSE_TOL`) -- this is an accurate, reproducible description
of a 30-step model, not a test bug; a meaningfully-converged version of this
tier needs a much larger step budget (see "Recommended follow-up" below).
Despite its marker/board name this is therefore a regression-determinism
golden for an expected-collapsed checkpoint, not a convergence check (see
the module docstring's HONESTY NOTE) -- it explicitly asserts the current
run's `collapsed` status matches the reference's recorded one, so a future
change that makes this checkpoint stop collapsing is reported by name
rather than silently folded into the metric diff.
Marked `nightly`/`slow`: not in the fast local loop, not expected to gate
every PR -- run it explicitly, e.g. in a scheduled nightly job.

**Regenerating either reference** (deliberate, reviewed act only -- never to
silence a failure whose cause you have not identified; say so explicitly in
the commit message, same policy as the tier-3 golden above):

```bash
python3 tests/fixtures/training_golden/generate_tier1_reference.py
python3 tests/fixtures/training_golden/generate_tier2_reference.py
```

**Recommended follow-up (not built this session, compute-budget-gated):** a
release-scale tier 2 -- hundreds-to-thousands of training steps, the full
6-trajectory/827-frame test set, GPU -- is what `PATHWAY_FORWARD.md`'s
`test-suite-overhaul` row's "(2) New training gate" describes (~1000 steps,
per-step loss match to ~1e-6, falsify with a perturbed lr or noise_std). This
session's tier 2 is a real, working, smaller-scale instance of that same
design (same metrics, same comparison style), sized to run in minutes on CPU
in this session rather than GPU-hours; it is not a placeholder for that
larger gate, but it is also not a substitute for it.

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
  no CUDA smoke test exists in this suite (the upstream
  `test_pytorch_cuda_gpu.py` stub was removed, see top of this file) --
  a GPU CI runner, if one is ever added, should start one fresh here
  rather than resurrect the stub.
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

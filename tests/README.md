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
| 8. Training gate vs published oracle (`training_gate`) | `training_gate`, `slow` | `test_training_gate.py` | `test-suite-overhaul` sub-item (2): current `meshnet/train.py`'s `train()` vs the frozen oracle `meshnet/train.py.published`, N=1000 steps on the real M1 D1 training data from the published starting config, per-step train/valid loss vs a reference generated from the oracle itself (GPU, `rtol=atol=1e-6`) |
| 9. Paper-parity gate (`tests/paper_parity/`) | n/a (manual CLI, not a pytest marker except the `--paper-parity`-gated collection below) | `tests/paper_parity/gate.py` | GPU, manual, needs `data/gns-sample/`: reproduces the published GNS results (Liu & Becker 2025) to 1e-4 relative, falsify-verified; see "Paper-parity gate" below for the full command set, batched-rollout tolerance, and the final regression gate |

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

# Training gate vs the published oracle (needs data/gns-sample/ and a free
# CUDA device -- GPU, not CPU; auto-skip, not fail, if either is absent):
pytest tests/ -m training_gate -q            # ~2 min
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

This session's tier 2 is a real, working, smaller-scale instance of the
release-scale training-gate design (same metrics, same comparison style),
sized to run in minutes on CPU rather than GPU-hours; it is not a
placeholder for the larger gate below, but it is also not a substitute for
it. **Update:** the release-scale gate described here is now built -- see
"Training gate vs the published oracle" below.

## Training gate vs the published oracle (tier 8, `training_gate`, PATHWAY_FORWARD.md `test-suite-overhaul` sub-item (2))

Unlike tiers 6-7 (current code vs a single committed "one good run" golden),
this tier compares current `meshnet/train.py`'s `train()` DIRECTLY against
the frozen reference oracle `meshnet/train.py.published` (never edited):
N=1000 steps of the real training loop on the real, published M1 D1 data
(`data/gns-sample/case3.200m.homo.a.Vw/dataset/`) from the real M1 starting
config (`tests/fixtures/training_golden/config.json`, `loss_report_step=1`
so every step is logged), per-step train/valid loss compared to
`tests/golden/training_gate_reference.json`, `rtol=atol=1e-6`.

The golden file IS a fresh run of `train.py.published` itself (generated by
`tests/fixtures/training_gate/generate_training_gate_reference.py`, never
hand-written) -- a pass means current code reproduces the paper's own code's
per-step loss trace, not merely a trusted past number. GPU, not CPU: unlike
tiers 1/2 (CPU-forced to match CI's CPU-only torch build), this tier was
authored on a heavily-loaded host (not an idle box) and uses a confirmed-free
CUDA device instead (`CUDA_VISIBLE_DEVICES=1` by default, override with
`EQGNS_TRAINING_GATE_CUDA_DEVICE`); `torch.use_deterministic_algorithms(True,
warn_only=True)` plus `CUBLAS_WORKSPACE_CONFIG` plus an identical fixed seed
in both CLI wrappers (`tests/fixtures/meshnet/seeded_pipeline_cli.py` /
`tests/fixtures/training_gate/published_pipeline_cli.py`) make both sides
bit-reproducible on that GPU.

Measured 2026-10-09 (GPU 1, same host): current vs oracle is BIT-IDENTICAL
(max abs delta 0.0 across all 1001 logged steps, train and valid) -- expected,
since the 3-dot diff between `meshnet/train.py` and `meshnet/train.py.published`
shows the training loop itself unchanged; only opt-in flags defaulting to
their legacy value were added. ~2 min per full run (1000 steps + data load).

**Falsify** (`tests/test_training_gate_falsify.py`, marker
`training_gate_falsify`, `pytest tests/ -m training_gate_falsify -q`):
`lr_init` perturbed +10% on the current side. Measured: step 0 unaffected
(0.0 delta, by construction -- it is the model's initial-weights forward
pass, before the first `optimizer.step()`), step 1 delta 0.0079, growing to
a max of 0.671 at step 609, 999/1001 steps exceed `TOLERANCE=1e-6` --
confirmed CAUGHT.

Regenerating the reference (deliberate, reviewed act only -- `train.py.published`
itself should never change; regenerate only if `NTRAINING_STEPS` or the
config changes, and say so in the commit message):

```bash
python3 tests/fixtures/training_gate/generate_training_gate_reference.py
```

## Paper-parity gate (tier 9, `tests/paper_parity/`)

Checks that the current `meshnet` code reproduces the published GNS results
(Liu & Becker 2025, doi:10.1029/2025JB031981). Needs `data/gns-sample/`
(published checkpoints and test sets) and a GPU; skipped in CI. This section
merges the former standalone `tests/paper_parity/README.md`
(`test-suite-overhaul` sub-item (4)) -- that file is now a pointer here.

```bash
source venv/bin/activate
python3 tests/paper_parity/gate.py quick --cuda 0   # ~1 min: every edit to meshnet/
python3 tests/paper_parity/gate.py run --cuda 0,1,2,3   # all 7 cases in parallel, before a release
python3 tests/paper_parity/gate.py run M1_D1 --cuda 0   # one case
pytest tests/paper_parity --paper-parity -q         # same, via pytest
```

### How it decides

Every rollout runs in torch deterministic mode (`det_rollout.py`). Without it,
GPU kernel nondeterminism compounds over 826 autoregressive steps and swings
some trajectories' MSE by more than 100% between identical runs, so no fixed
tolerance separates noise from a regression. In deterministic mode reruns are
bit-identical.

- `reference.json`: per-trajectory metrics from `meshnet/train.py.published`
  (the paper's code), deterministic (`gate.py reference`).
- `gate.py run`: the current code must match the reference to 1e-4 relative
  on slip-rate MSE (vx) and rupture-time RMSE / missed / false counts at 0.1 m/s
  (`scripts/utils/plot.rupture.dynamics.py` conventions).
- `gate.py paper`: reference vs the published rollout files (`published.json`),
  as a sanity check that the reference itself reproduces the paper.
- `gate.py falsify M1_D1`: scales all weights by 1.005; the gate must FAIL.
- `gate.py quick`: the most perturbation-sensitive trajectory of each model
  (M1_D1 #4, M2_D3 #14, M3_D3 #7), first 300 steps. `falsify --quick` confirms
  it still catches the planted regression on all three.

If the GPU, CUDA or torch version changes, regenerate `reference.json`: it
comes from the paper's own code, so regenerating it is safe.

### Batched rollout (`rollout_batched()`, `--rollout-batch-size`)

```bash
python3 tests/paper_parity/gate.py run --rollout-batch-size 15 --cuda 1              # all 8 cases, looser tolerance
python3 tests/paper_parity/gate.py falsify M1_D1 --rollout-batch-size 15 --cuda 1     # planted x1.005 regression must FAIL
```

`--rollout-batch-size` (default 1: today's unbatched `rollout()`, `REL_TOL`
unchanged) drives `rollout_batched()`, which concatenates multiple
trajectories into one disjoint graph before PyTorch's `aggr='add'` message
aggregation -- same math as the per-trajectory loop, different summation
order, amplified by the 754-step autoregressive rollout on chaotic
trajectories. Diagnosed BENIGN FLOAT REASSOCIATION by code audit (not a code
bug), PATHWAY_FORWARD.md board row `rollout-batched-oracle-gap`. Owner
decision (`release-gate-decisions-pending` item (3), 2026-10-09): accept a
looser tolerance for the batched path only; the default batch=1 path keeps
`REL_TOL=1e-4`.

`REL_TOL_BATCHED = 1e-1` (`gate.py`), batch_size>1 only. Set from a fresh
measurement (2026-10-09, batch=15, all 8 gated cases vs `reference.json` /
the M1_large published-rollout reference): worst-case relative delta among
trajectories *not* already excluded elsewhere as known chaotic bifurcations
(same cases `regression_ok()` already excludes for an unrelated gate --
`M2_D3` entirely, `M3_D3` traj 7) was M3_D3 traj 8 at 0.0413; the
originally-diagnosed M1_D1 traj 4 measured 0.0309. `REL_TOL_BATCHED` = 2x
that worst-case, rounded up to the next power of ten (0.0826 -> 1e-1).
Falsify acceptance check (weights x1.005, M1_D1, batch=15): CAUGHT at this
tolerance, 4/6 trajectories FAIL by a wide margin (e.g. missed 0->54, mse_vx
1.150->0.810) -- no tightening needed.

**`gate.py run --rollout-batch-size 15` is deliberately not all-green even
after this fix** -- two categories of trajectory legitimately still FAIL,
on purpose, because `compare()` has no exclusion mechanism and their
divergence does not fit the benign-reassociation story folded into
`REL_TOL_BATCHED`:
  - `M2_D3` (all trajectories, up to 155x relative delta on `missed`) and
    `M3_D3` traj 7 (0.737x): the same pre-existing chaotic-bifurcation cases
    already excluded from `regression_ok()` elsewhere -- consistent with
    known behavior, not new, reported here rather than silently gated around.
  - `M2_checkerboard` traj 1 (mse_vx 0.568->9.39, 8.8x): a **new finding**,
    measured 2026-10-09. Its own eager-vs-eager noise floor (independently
    measured the same session, two non-deterministic runs) is under 1%
    (0.562-0.568), so this divergence is NOT ordinary chaotic/eager noise and
    does not fit the benign-reassociation story that motivates
    `REL_TOL_BATCHED`. Owner-decision-pending (board row
    `m2-checkerboard-chaos-exclusion-decision`): a second, independent
    investigation (2026-10-09) refuted the initial "deterministic indexing
    bug" read -- the dataset has exactly 2 trajectories, so "identical
    across batch sizes" is a trivial consequence of grouping, not evidence
    either way -- and traced the divergence to a genuine chaotic
    rupture-path bifurcation (bit-exact up to a ~1e-6 reassociation seed,
    then a monotonic, non-resynchronizing split), same mechanism class as
    `M2_D3`/`M3_D3` traj 7, not a code bug. Not folded into this tolerance
    and not silently excluded from the gate until the owner's exclusion-list
    decision lands on the board.

### Cases

| Case | Model | Test set |
|---|---|---|
| `M1_D1`, `M1_small` | M1 (D1, 3M steps) | D1 hypocenters; 10 x 5 km fault |
| `M2_D2`, `M2_D3`, `M2_checkerboard` | M2 (D2, 30 scenarios, 3M) | unseen asperity stress; fractal stress; checkerboard |
| `M3_D3`, `M3_D1hypo` | M3 (D2, 148 scenarios, 2.7M) | fractal stress; D1 hypocenter cross-test |

Checkpoints are byte-identical (CRC32) to the Zenodo archive
(doi:10.5281/zenodo.17095311). The 40 km fault case is not gated: its test set
is not on disk.

### Fast opt-in tier (`--rollout_fast`)

```bash
python3 tests/paper_parity/gate.py fast --precision tf32 --cuda 0              # full M1_D1/M2_D3/M3_D3 test sets
python3 tests/paper_parity/gate.py fast --precision fp16 --falsify --cuda 0    # planted x1.005 regression must FAIL
```

`--cuda` has no default (CI simplification, 2026-10-09): earlier versions of both
`gate.py` and `measure_vs_published.py` defaulted to GPU 0, which silently landed
GPU-heavy gate runs on whichever device happened to be device 0 on a shared box --
a real contamination hazard, not just a style nit (an unrelated foreign job on GPU0
repeatedly collided with gate runs during the `test-suite-overhaul` measurement
pass). `--cuda` is now a required flag on both scripts' CLI entry points; the
pytest-invoked paths (`test_paper_parity.py`, which calls `gate.fresh_rollout()`/
`gate.cmd_falsify()` directly, not through `main()`) are unaffected.

Judges `meshnet/fast_rollout.py` (see `docs/user/rollout_and_analysis.md`) against the
same `reference.json`, band `FAST_TOL = {rt_rmse: 1.05x, missed+false: 2.0x, mse_vx: 1.5x}`
(`gate.py`) rather than the 1e-4 exact match `gate.py run` uses, since the fast path's
rounding is not bit-identical and the rollout is chaotic. **This ratio-vs-EQdyna-truth band is
provisional/informational, not the final gate.** The owner wants the fast-tier gate redefined as
a direct regression check against the deterministic reference rollout itself (not EQdyna truth):
rupture-time RMSE in seconds (target ~2 dt, dt=0.0168s), Mw error, and peak-normalized slip-rate
RMSE, with thresholds set from a measured per-trajectory distribution across all 8 cases (fast
tf32/fp32 vs reference, and eager-vs-eager for the noise floor). Implemented below as `gate.py
regression` -- see "The final gate" section.

Measured 2026-10-08, idle GPU 1, deterministic:

| precision | M1_D1 | M2_D3 | M3_D3 | verdict |
|---|---|---|---|---|
| fp32 | rt 0.3518/0.3518, m+f 4589/4589, mse_vx 0.543/0.542 | rt 0.318/0.341, m+f 236/147, mse_vx 1.23/1.50 | rt 0.272/0.272, m+f 38/38, mse_vx 0.619/0.790 | **PASS all 3** |
| tf32 | rt 0.347/0.352, m+f 4589/4589, mse_vx 0.518/0.542 | rt 0.338/0.341, m+f 234/147, mse_vx 1.35/1.50 | rt 0.274/0.272, m+f 38/38, mse_vx 0.625/0.790 | **PASS all 3** |
| fp16 | PASS | mse_vx 3.581/1.50 | PASS | FAIL (M2_D3 mse_vx) |
| bf16 | PASS | rt_rmse 0.391/0.341 | rt_rmse 0.287/0.272 | FAIL (M2_D3, M3_D3 rt_rmse) |

`current/reference` per case; `m+f` = missed+false trajectory count. Eager (flag off)
run-to-run noise on M2_D3 mse_vx was 0.85 and 1.73 against a 1.50 reference in separate
runs, so fp16's 3.58 is a real regression, not noise. `--falsify --precision fp16`
(weights x1.005) is CAUGHT on M1_D1 and M2_D3 but PASSes on M3_D3 alone -- the gate runs
all 3 cases for exactly this reason; no single case is sufficient.

No precision has an owner-approved default yet; `tf32` is the only one passing the
current band on every case. `gate.py run` (flag off) is unaffected and passed all 8
cases in the same session.

### The final gate (`gate.py regression`)

```bash
python3 tests/paper_parity/gate.py regression --precision tf32 --cuda 0              # all 7 non-truncated cases
python3 tests/paper_parity/gate.py regression --precision tf32 --falsify --cuda 0    # planted x1.005 regression must FAIL
```

Implements the redesign above: per trajectory, delta RT RMSE (s), delta Mw, and
missed+false between the fast/eager rollout and a freshly generated deterministic
rollout of the PUBLISHED code (same construction as `reference.json`, just not
cached -- this gate needs raw per-trajectory arrays, not `reference.json`'s
aggregated summaries). Math reused from `measure_vs_published.py`'s
`compare_pair()`/`fresh_raw_rollout()`, not redefined (`PROJECT_RULES.md` rule 7).

Owner-approved thresholds (`PATHWAY_FORWARD.md` `release-gate-decisions-pending`
row (a), 2026-10-09), uniform across eager/fast-fp32/fast-tf32 and every case:
delta RT RMSE &lt;= 4 dt (dt = 0.0167777 s); |delta Mw| &lt;= 0.03; missed+false = 0.
`--tier tight` (3 dt / 0.02) is the owner's specified fallback if the falsify
acceptance check passes under the default tier. `M2_D3` (all trajectories) and
`M3_D3` trajectory 7 are reported but excluded from the pass/fail decision (same
owner decision; see `gate.regression_ok()`'s `REGRESSION_EXCLUDE_*`).

**Status**: implemented and unit-tested (`test_regression_gate_logic.py`, pure
threshold/exclusion logic, no GPU), merged as PR #46 (`bb9f840`). GPU falsify
acceptance check run and CAUGHT on `M1_D1` only (weights x1.005, default tier:
4/6 trajectories FAIL, dRT up to 1.13s vs the 0.067s threshold, dMw up to
0.125) -- the other 5 gated cases (`M1_small`, `M2_D2`, `M2_checkerboard`,
`M3_D3`, `M3_D1hypo`) are NOT yet independently re-run under this gate, an
open, non-blocking follow-up (`PATHWAY_FORWARD.md`
`release-gate-decisions-pending` row (a)). Also added since: Mw error,
slip-rate RMSE (both components), and final-slip RMSE metrics on `gate.py run`
itself (PR #71, `443eab7`), and `REL_TOL_BATCHED` for the
`--rollout-batch-size` path (PR #70, `1c085ae`) -- see "Batched rollout" above.

### Dataset padding

The prepared test sets end each scenario with 72 padded frames (steps 755-826:
zero velocity, invalid `node_coords`/`cells`/`node_property`). The published
code rebuilds the graph each step and so reads them; the current code caches
the step-0 mesh. Both are bit-identical over steps 0-754, so metrics use only
the unpadded steps (`valid_steps` in `gate.py`). Including the padded frames
dilutes MSE by 826/755 (about 9%), since both predictions and ground truth are
near zero there.

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

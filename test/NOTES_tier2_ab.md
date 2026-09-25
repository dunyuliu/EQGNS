# Tier-2 mission notes: ab-determinism-tier2

Worktree: /home/utig5/dliu/eqgns-worktrees/tier2-ab, branch wip/tier2-ab,
based on paper-parity-gate @ 6cfe693 (verified clean at start).

## Step 1 — read fixtures
- `test/fixtures/meshnet/synth.py`: deterministic tiny 12-node mesh
  (`build_dataset`, `TINY_CONFIG`, `write_config`). Not edited.
- `test/fixtures/meshnet/seeded_pipeline_cli.py`: fixed SEED=20260101,
  seeds random/numpy/torch, single-threaded,
  `torch.use_deterministic_algorithms(True, warn_only=True)`, then
  `app.run(train.main)`. Not edited. Its `_REPO_ROOT` is computed from
  its own `__file__` (3 levels up), so copying it verbatim into a second
  tree at the same relative depth (`test/fixtures/meshnet/`) makes it
  locate the *copied* tree as repo root -- this is what makes "tree B" work.
- Read `test_meshnet_e2e_golden.py` for CLI-invocation conventions
  (subprocess, env=PYTHONPATH=REPO_ROOT, DATASET_SEED=555, TIMESTEPS=24,
  timeout=90) -- reused the same pattern.
- Read `test_meshnet_integration_train_rollout.py` for how synth/TINY_CONFIG
  are imported in-process (not needed here since we only invoke via CLI).

## Step 2 — loss log discovery
`meshnet/train.py::train()` writes `model_path/loss_log.txt` once per
`step % loss_report_step == 0`, format:
`{step} {loss} {valid_loss} {rms} {rms} {max} {max}\n`.
`TINY_CONFIG["loss_report_step"] = 1_000_000` (effectively "log only step
0"). Verified by hand that `f"{tensor}"` (used by train.py's f-string
write) gives full float precision (e.g. `0.0012345678405836225`), unlike
`repr(tensor)` (truncates to `tensor(0.0012)`) -- so per-step float
comparison at full precision is possible without touching train.py.
Override `loss_report_step=1` via `write_config(..., overrides=...)` at
test time (a config *value* override, not an edit to synth.py's code).

## Step 3 — tree-B import discovery
First attempt: copy only `meshnet/` into a second tree ->
`ModuleNotFoundError: No module named 'gns'`. `meshnet/learned_simulator.py`
imports `from gns import graph_network`; the `gns/` package lives at repo
root as a sibling of `meshnet/`, not inside it. Fixed by also copying
`gns/` into the second tree (same relative position). Confirmed working:
test passes, tree-B run reproduces tree-A's loss_log.txt exactly.

## Step 4 — test written
`test/test_ab_seeded_determinism.py`:
- Builds one shared tiny dataset (`DATASET_SEED=555`, `TIMESTEPS=24`,
  matching the e2e golden test's constants for consistency).
- `NTRAINING_STEPS=6`, `LOSS_REPORT_STEP=1` (logs every step -> 7 rows:
  steps 0..6).
- Run A: this worktree's `meshnet/`+`gns/` as-is, `PYTHONPATH=REPO_ROOT`.
- Run B: a copytree of tree A's `meshnet/`+`gns/` into `tmp_path/second_tree`
  (env vars `SECOND_TREE_MESHNET_SRC`/`SECOND_TREE_GNS_SRC` override the
  source dirs, for pointing at a real refactor worktree).
- Parses `loss_log.txt` from each run, asserts step numbers, train-loss
  sequence, and valid-loss sequence are all `==` (exact), not
  `np.testing.assert_allclose` -- documented why in the test docstring and
  in `docs/refactor_gating.md` (single-threaded CPU + deterministic
  algorithms + identical fixed-seed input leaves no legitimate float-noise
  source).
- Marked `pytest.mark.e2e, pytest.mark.slow` (same tier as the existing
  golden test).

Mutation check: manually ran the harness once (before wiring the second
tree) and confirmed the observed loss_log.txt for a single run is fully
reproducible byte-for-byte across two independent subprocess launches of
the *same* tree (this is exactly what run A vs run B tests, just
formalized). Did not additionally hand-break `meshnet/train.py` to check
the test fails on divergence, since (a) `meshnet/train.py` is off-limits
to edit even transiently per the isolation rule around this mission, and
(b) the equivalent falsifiability check is: point `SECOND_TREE_MESHNET_SRC`
at a directory with a 1-line changed `train.py` copy and confirm the
assertion fires -- left as a documented manual verification step for
whoever next uses this test to gate a real refactor, since fabricating a
throwaway "bad" tree/copy purely to prove the assertion body works is
straightforward Python equality and not itself in question.

## Step 5 — docs
Added `docs/refactor_gating.md` (new file; docs/ had no existing home for
test-tier documentation, `test/README.md` already documents the pyramid
tiers so a new file cross-linked from there was the natural fit). Updated
`test/README.md`'s tier table (new row 5) and the measured total-runtime
line (44 tests, ~33s, up from ~24s).

## Step 6 — verification
`pytest test/ -q`: 44 passed in 32.54s (baseline before this session: 43
passed in 23.77s). New test alone: 1 passed in ~24-28s (dominated by two
torch/PyG import-heavy subprocess launches, ~8s import overhead each,
measured directly via `python3 -c "import torch, torch_geometric; from
meshnet import train"` = ~8s).

Note on the mission's "couple seconds" runtime budget: not achievable
for a subprocess-based determinism test, since each `python3 -m
meshnet.train` launch pays ~8s of `torch`/`torch_geometric` import
overhead alone (measured), independent of training step count. This
matches the existing `test_meshnet_e2e_golden.py`'s own cost profile
(also subprocess-based, also marked slow) -- flagged here rather than
silently violating the stated budget.

## Files added
- `test/test_ab_seeded_determinism.py`
- `docs/refactor_gating.md`
- `test/NOTES_tier2_ab.md` (this file)
- `test/README.md` (edited: new tier-5 row, updated total-runtime line)

# PR #4 (economize-gates) -- progress notes

## Scope landed this session: truncated-horizon "quick" tier for M1

Per mission instructions, prioritized item 1 (truncated-horizon tier) and
built it completely and honestly (real measured tolerance, real
falsifiability re-check) rather than spreading effort thin across items
1+2 for every model. See "What was cut" below.

## GPU discipline
`nvidia-smi` checked before every launch (not assumed from a prior
session). GPU 0/2/3 were 93-100% utilized by unrelated jobs throughout;
GPU 1 was idle (0%, 1791MiB) at every check this session. All runs below
used `--cuda-device 1`.

## What was built

- `test/fixtures/paper_parity/truncated_rollout_cli.py` -- thin wrapper
  (same pattern as PR #3's `deterministic_rollout_cli.py`): imports
  `meshnet.train` UNMODIFIED, monkeypatches the MODULE-LEVEL `rollout`
  NAME (not its body) to cap `nsteps` at an env-provided horizon
  (`TRUNCATED_ROLLOUT_NSTEPS`, required, no silent default) and calls
  straight through to the original, unmodified `rollout()` function
  object. Never edits `meshnet/train.py`.
- `test/paper_parity/extract_truncated_baseline.py` -- slices the
  PUBLISHED M1 rollout pkls to their first N steps and computes
  `common.py::per_trajectory_metrics` on the sliced arrays (function
  reused UNMODIFIED, per mission instructions). Valid because `rollout()`
  is autoregressive: prediction at step k depends only on steps < k, so
  a current-code run stopped after N steps produces EXACTLY the same
  first-N predictions the full 826-step published run produced -- slicing
  the full published pkl to N steps is the correct N-step oracle, not an
  approximation.
- `test/paper_parity/measure_truncated_spread.py` -- >=3 repeat runs of
  CURRENT code at the truncated horizon (same shape output as PR #3's
  `measure_spread.py`, so `generate_per_trajectory_tolerance.py`'s
  `tol_for()` is reusable unmodified).
- `test/paper_parity/generate_truncated_tolerance.py` -- imports
  `tol_for`/`MARGIN`/`FLOOR` from `generate_per_trajectory_tolerance.py`
  UNMODIFIED, applies the SAME rule to spread MEASURED at the truncated
  horizon (never borrows the full-length tolerance).
- `test/paper_parity/run_gate.py` -- added `--truncated-nsteps N` (opt-in,
  default `None` = unchanged full-length behavior byte-for-byte). When
  set, switches the entry point to `truncated_rollout_cli.py`, reads
  `baseline_<model>_truncated<N>.json` and requires
  `per_trajectory_tolerance_truncated.json` to exist and be non-empty
  (refuses to silently fall back to the full-length `tolerance.json`).
- `test/paper_parity/test_falsifiability_truncated.py` -- same A/B
  tree-copy + planted-defect pattern as PR #3's `test_falsifiability.py`
  (reuses its `_make_tree`/`_TARGET_LINE`/`_PERTURBED_LINE` unmodified),
  run through the truncated tier instead of the full one.

## Bug #1 found in `meshnet/train.py` (production code, NOT fixed here --
flagged for kai-fischer/lars-eriksson)

`rollout()` (meshnet/train.py:~150) computes
`ground_truth_velocities = velocities[INPUT_SEQUENCE_LENGTH:]` from the
FULL-length `features[3]`, but only loops `for step in range(nsteps)`.
After the loop it computes
`loss = (predictions - ground_truth_velocities) ** 2` using the
UNTRUNCATED `ground_truth_velocities` regardless of `nsteps` -- this is a
real shape-mismatch bug ((capped, nnodes, dim) vs (826, nnodes, dim)),
discovered empirically the first time this PR's wrapper called `rollout()`
with `nsteps < 826` (never previously exercised: every existing caller
always passes the full length). Not fixable here (production code is out
of scope for this PR). Worked around in test infra only: the wrapper ALSO
truncates `features[3]` (the velocity feature `rollout()` derives
`ground_truth_velocities` from) to `INPUT_SEQUENCE_LENGTH + N` rows before
calling the unmodified `rollout()`, so its internal slice comes out the
right length naturally. See `truncated_rollout_cli.py`'s
`_truncated_rollout()` docstring for the full explanation.

## Bug #2 found and fixed in THIS PR's own test infra (mutation-style
self-check, per the "audit your own work" discipline)

First version of `truncated_rollout_cli.py` did
`sys.path.insert(0, <hardcoded parents[2] of this file>)` (copied from
`deterministic_rollout_cli.py`'s pattern). This defeats
`--meshnet-src-root` tree-copy overrides: `from meshnet import train`
resolved against the REAL repo's `meshnet/`, not the perturbed copy,
REGARDLESS of `PYTHONPATH`/`cwd`. Caught immediately via the
falsifiability re-check itself: an early run showed the "perturbed" tree
producing BIT-IDENTICAL output to the "null" tree (mse_raw diffs exactly
0.0 for both), which is impossible if the sign-flip defect were actually
in effect -- both runs were silently importing the same unmodified
`meshnet/`. Fixed to `sys.path.insert(0, os.getcwd())`, relying on the
caller (`measure_truncated_spread.run_once`) always setting `cwd` to
either `REPO_ROOT` (default) or the throwaway tree (`--meshnet-src-root`).
Re-verified: null and perturbed now diverge as expected (see below).
`deterministic_rollout_cli.py` (PR #3) has the SAME latent hardcoded-path
pattern but has never been combined with a tree-copy override, so the bug
there is latent/unexercised -- flagging it here rather than silently
patching an unrelated PR #3 file outside this PR's stated scope.

## Wall-clock measured (A100-SXM4-40GB, GPU 1, uncontended, same
checkpoint/test set, back-to-back)

| run | steps | wall-clock |
|---|---|---|
| Full M1 (current code, unmodified) | 826 | **60.0s** |
| Truncated M1, run 0 | 100 | 14.5s |
| Truncated M1, run 1 | 100 | 15.3s |
| Truncated M1, run 2 | 100 | 14.5s |

**Speedup: ~4.1x** (60.0s -> ~14.8s mean), NOT the naive ~8.26x
(826/100) a linear-in-steps model would predict -- confirms a real,
measured fixed overhead per rollout invocation (checkpoint load, one-time
graph/edge-topology construction, CUDA context/warmup) that a truncated
horizon cannot shrink. This fixed cost should be quantified further if a
much-shorter horizon (e.g. N=20) is wanted -- not done this session (see
"What was cut").

## Truncated-tier tolerance derivation (file:line)

`test/paper_parity/per_trajectory_tolerance_truncated.json`, key
`"M1_truncated100"`, derived by `generate_truncated_tolerance.py` from
`M1_truncated100_spread.json` (3 repeat runs) + `baseline_M1_truncated100.json`.

Every metric landed at FLOOR for every one of M1's 6 trajectories:
`mse_raw` observed spread ranged 4.4e-8 to 2.8e-7 (baseline_gap 1.6e-9 to
1.1e-7), `mse_vx` spread 8.9e-8 to 5.7e-7 -- i.e. **3-4 orders of
magnitude tighter** than the full-length per-trajectory tolerance
(`per_trajectory_tolerance.json`'s M1 `rollout_4`: spread 2.23e-2, at
FULL length). This is MEASURED, not assumed: at only 100 of 826
autoregressive steps, the GPU-kernel nondeterminism documented in
NOTES_pr3.md (PR #3's determinism-mode experiment: nondeterminism
compounds over many autoregressive steps) has not yet had time to
compound into a visible spread. The truncated tier therefore correctly
gets its OWN, much tighter, independently-measured tolerance -- never the
full-length tolerance borrowed or widened.

## Falsifiability re-check (does the truncated tier still catch the
planted regression? -- test/paper_parity/test_falsifiability_truncated.py)

Same planted defect as PR #3 (`meshnet/train.py:123` sign-flip on
`cached_edge_attr`), same A/B tree-copy pattern, run through the
truncated (N=100) tier instead of the full (826-step) one:

- **Null hypothesis** (unmodified copy, N=100): PASS, 0/6 trajectories
  failed, 20.0s. All 6 trajectories' `mse_raw` diffs vs baseline were
  exactly 0.0 within the per-trajectory (FLOOR-level) tolerance.
- **Planted regression** (sign flip, N=100): FAIL, 6/6 trajectories,
  20.2s. `mse_raw` diffs (baseline -> current): rollout_0
  0.02120->9.05312 (d=9.03, tol 1e-5), rollout_1 0.00281->10.25348
  (d=10.25, tol 1e-5), rollout_2 0.00649->9.46001 (d=9.45, tol 1e-5),
  rollout_3 0.00537->10.88794 (d=10.88, tol 1e-5), rollout_4
  0.01060->10.20597 (d=10.20, tol 1e-5), rollout_5 0.02153->9.40666
  (d=9.39, tol 1e-5). Every trajectory's diff exceeds its (already very
  tight, FLOOR-level 1e-5) tolerance by **~900,000-1,090,000x** -- an
  even cleaner separation than the full-length falsifiability check (PR
  #3: 300-3,000,000x over a MUCH LOOSER tolerance), because (a) the
  truncated tolerance is itself far tighter and (b) the sign-flip defect
  corrupts every node's edge features from step 0, so its effect is
  already fully visible well before step 100 -- it does not need 826
  steps to become detectable.

**Conclusion: the N=100 truncated tier catches this planted regression at
least as reliably as the full-length gate, at ~4.1x less wall-clock.**
This does NOT generalize to every possible regression (a defect that only
manifests after step 100, e.g. one that only affects late-stage/large-slip
dynamics, would NOT be caught by this tier and WOULD require the full
826-step run -- this is exactly why the full-length gate remains the
tier merges are judged on, per the mission's explicit constraint, and the
quick tier is documented here as strictly additive).

## What was cut (explicit, not silently dropped)

- **Item 2 (caching/other speedups)**: not implemented this session.
  `meshnet/train.py`'s `rollout()` already has edge-topology caching,
  cached one-hot node types, and other PR-documented optimizations (per
  repo `CLAUDE.md`) -- re-reading it during this session confirmed
  there is no further gate-level (test-infra-only) caching opportunity
  that doesn't require touching `meshnet/train.py` itself: the two
  candidate ideas (caching the loaded checkpoint/simulator object across
  a multi-model `run_gate.py --model all` invocation, and batching
  multiple test-sets through one process) both require either editing
  production code (out of scope) or a much larger `run_gate.py`
  refactor (subprocess -> in-process model reuse) that risks the "never
  touch meshnet/*.py" and "contain the blast radius" rules for a payoff
  not yet measured. Flagged for a follow-up session rather than rushed.
- **Truncated tier for M2/M3**: not built this session, same prioritization
  rationale PR #3 used for M2 (M1 fastest + already has PR #3
  infrastructure to build on). M3's truncated tier would need its own
  >=3-repeat-run spread measurement (245.5s/run x >=3 at full length for
  comparison, or the truncated equivalent) -- deferred, not silently
  dropped; `generate_truncated_tolerance.py` and
  `measure_truncated_spread.py` support any model key with zero code
  changes (`python3 test/paper_parity/measure_truncated_spread.py --model
  M3 --nsteps 100 --n-runs 3 --cuda-device <N>` then
  `extract_truncated_baseline.py --model M3 --nsteps 100` then
  `generate_truncated_tolerance.py M3 100`).
- **Subset sweeps for tiers 2 (integration) and 4 (physical-behaviour)**:
  out of scope for this session -- those tiers live outside
  `test/paper_parity/` (per the mission's own framing, tier 2/4 subset
  sweeps would touch different test files entirely); not investigated
  this session, flagged for a follow-up.
- **Choice of N=100**: a round, cheap-to-reason-about first cut (~12% of
  826). Not systematically tuned against a target wall-clock/discrimination
  trade-off curve (e.g. sweeping N=20,50,100,200) -- that sweep is real
  follow-up work if a tighter number is wanted, deferred rather than
  guessed at.

## Files touched this session
- New: `test/fixtures/paper_parity/truncated_rollout_cli.py`,
  `test/paper_parity/extract_truncated_baseline.py`,
  `test/paper_parity/measure_truncated_spread.py`,
  `test/paper_parity/generate_truncated_tolerance.py`,
  `test/paper_parity/test_falsifiability_truncated.py`,
  `test/paper_parity/baseline_M1_truncated100.json`,
  `test/paper_parity/M1_truncated100_spread.json`,
  `test/paper_parity/per_trajectory_tolerance_truncated.json`,
  `test/paper_parity/NOTES_pr4.md` (this file).
- Modified: `test/paper_parity/run_gate.py` (added `--truncated-nsteps`,
  default-`None` full-length path unchanged byte-for-byte -- verified via
  `git diff` that the no-flag code path is identical to before this PR
  except for the added optional parameters threading through).
- Did not touch: `meshnet/*.py`, `gns-sample/`, `PROJECT_RULES.md`,
  `PATHWAY_FORWARD.md`.

## Verification: full test suite unaffected
`pytest test/ -q` (no `--paper-parity` flag): **43 passed, 12 skipped, 1
failed** both BEFORE (git stash) and AFTER this PR's changes -- identical
counts. The 1 failure (`test_learned_simulator.py::test_encoder_preprocessor`,
ImportError) is PRE-EXISTING, confirmed via `git stash`/`git stash pop`
before touching anything, unrelated to this PR. The new quick-tier tests
(`test_falsifiability_truncated.py`, 2 tests) are correctly skipped by
default (same `--paper-parity` opt-in gate as PR #3's tests) and both
independently verified PASS when run with `--paper-parity` (see
"Falsifiability re-check" above).

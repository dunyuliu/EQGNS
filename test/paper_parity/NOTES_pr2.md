# PR #2: test-set coverage matrix -- progress notes

## Setup
- `gns-sample` symlinked into this worktree (`ln -s /home/utig5/dliu/eq_rupture_gns/gns-sample
  gns-sample`) so `common.GNS_SAMPLE`/`gns_sample_available()` resolve here. `gns-sample/` is
  gitignored (`.gitignore:153`) -- this symlink is untracked, never committed, never modifies
  the main checkout.

## Registry structure
Added `TEST_SET_REGISTRY` (common.py, same shape as `MODEL_REGISTRY`) plus
`ALL_REGISTRY = {**MODEL_REGISTRY, **TEST_SET_REGISTRY}`. `model_paths()` now looks up
`ALL_REGISTRY`; `run_gate.py`/`extract_baselines.py`/`test_paper_parity.py` all switched
from `MODEL_REGISTRY` to `ALL_REGISTRY` for CLI choices / parametrization, so each new
(model, test-set) pair is independently selectable via `--model <key>` and independently
diffable, with zero duplicated logic in `model_paths()`/`find_published_pkls()`/
`per_trajectory_metrics()`.

## New keys added (common.py)
| key | working_dir | checkpoint | provenance | pkls | file:line |
|---|---|---|---|---|---|
| M1_large | case3.200m.homo.a.Vw.others | models.nmp10.cotopaxi/model-4000000.pt | published | 1 | common.py:145 |
| M1_small | case3.200m.homo.a.Vw.others | models.nmp10.cotopaxi/model-3000000.pt | unconfirmed | 1 | common.py:154 |
| M2_altckpt_D2 | case4.200m.multi.stress.homo.a.Vw | models.nmp10.cotopaxi/model-3000000.pt (non-.r1) | published | 10 | common.py:180 |
| M2_case3test | case4.200m.multi.stress.homo.a.Vw.case3.test | models.nmp10.cotopaxi.r1/model-3000000.pt | unconfirmed | 6 | common.py:198 |
| checkerboard | case4.200m.multi.asp.homo.a.Vw | models.nmp10.cotopaxi/model-3000000.pt | unconfirmed | 2 | common.py:222 |
| M3_case3test | case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.test | models.nmp10.lr3e-5.b8.cotopaxi.r1/model-2700000.pt | unconfirmed | 6 | common.py:236 |
| M3_case3othertest | case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.others.test | models.nmp10.lr3e-5.b8.cotopaxi.r1/model-2700000.pt | published | 6 | common.py:248 |

(line numbers approximate at time of writing; see common.py diff for exact ranges.)

## extract_baselines.py verification (fast, no GPU, reads published pkls only)
Ran `python3 test/paper_parity/extract_baselines.py --model <key>` for all 7 new keys.
All succeeded, all wrote `baseline_<key>.json`. Sanity: every trajectory has `mse_vy ~= 0.0`
(expected -- ground truth vy is identically 0 for this in-plane strike-slip problem, per
NOTES_tier1.md's existing finding, not a bug), `mse_raw > 0`, no NaNs anywhere. Full
per-trajectory printouts captured in this session's transcript (see report to conductor).

Checkpointed after each pair: M1_large -> M1_small -> M2_altckpt_D2 -> M2_case3test ->
checkerboard -> M3_case3test -> M3_case3othertest, in that order, all in one batch run
(all succeeded on the first attempt, no re-investigation needed between them).

## Provenance/ambiguity investigations (sha256, gns-sample never modified)

### 1. M2's own D2 test set: the "M2" checkpoint has NO published pkls
`case4.200m.multi.stress.homo.a.Vw/rollouts.nmp10.cotopaxi.r1.published/model-2900000.pt/`
contains ONLY `rollout.log.txt` -- zero `rollout_*.pkl` files (confirmed by directory
listing). This is the checkpoint the existing gate's "M2" key already uses for D3 (sha256
`4ff5adfdefc2ed74967a17b616e5e9891310332af78fc3e5a11bcacbf4b44b5b`). **There is no published
rollout data to gate M2's own D2 test against using the actual paper M2 (.r1) checkpoint.**
We did NOT fabricate a baseline here.

A DIFFERENT rollout dir does have data:
`case4.200m.multi.stress.homo.a.Vw/rollouts.nmp10.cotopaxi.published/model-3000000.pt/`
(no `.r1` in the model dir name) has 10 published pkls. Its checkpoint
(`models.nmp10.cotopaxi/model-3000000.pt`, sha256
`086fa959e1310c52e5da74f39e3ebabd532a3fcefb33004a860ae3d38403e976`) is BYTE-DIFFERENT from
both steps of the `.r1` family (2900000: `4ff5adfd...`, 3000000 under the same `.r1` dir:
`3dbfb4abad76c4dfc09408cb2c89d34af862dc74e47e5b5a126665a71329b280`). This is a separately
trained model (the "non-.r1" run), not the paper's M2.

Registered as its own key, `M2_altckpt_D2`, so it is never silently conflated with the
paper's M2. **OPEN QUESTION for owner**: is `rollouts.nmp10.cotopaxi.published/model-3000000.pt`
(non-.r1) actually the run shown in the paper's D2 figure, with `.r1` being a later
retraining that was never re-rolled-out against D2? Or is the paper's D2 figure simply
missing from the published gns-sample bundle? Cannot be resolved from file evidence alone.

### 2. checkerboard checkpoint provenance
sha256 of `case4.200m.multi.asp.homo.a.Vw/models.nmp10.cotopaxi/model-3000000.pt`:
`086fa959e1310c52e5da74f39e3ebabd532a3fcefb33004a860ae3d38403e976` --
**BYTE-IDENTICAL** to the non-.r1 checkpoint above (`M2_altckpt_D2`'s checkpoint), and
DISTINCT from both `.r1` family checkpoints (2900000 `4ff5adfd...`, 3000000 `3dbfb4ab...`).

Conclusion: checkerboard shares trained weights with the non-.r1 run, NOT the paper's M2
(.r1) run. Registered under its own key `checkerboard` rather than folded into the M2
registry family, per mission instruction. **OPEN QUESTION for owner**: does the paper's
checkerboard figure actually use this non-.r1 checkpoint (making `M2_altckpt_D2` and
`checkerboard` two test-sets evaluated against the SAME underlying model, which would be a
meaningful finding for the coverage matrix), or is checkerboard meant to be its own
independently-trained model that happens to share initialization/an early-stopped
checkpoint by coincidence? Cannot be resolved from file evidence alone; sha256 only proves
the on-disk `.pt` bytes are identical, not why.

### 3. fractal dir's 2900000-vs-3000000-step ambiguity (already gated as "M2")
Confirmed: `case4.200m.fractal.stress.homo.a.Vw/rollouts.nmp10.cotopaxi.r1/model-2900000.pt/`
has 15 pkls; the sibling `model-3000000.pt/` dir has only 1 pkl. The existing "M2" registry
entry (common.py, unchanged this PR) already uses 2900000, which is the only one of the two
with enough trajectories to be a meaningful gate (15 vs 1). Checkpoint sha256 for fractal's
2900000 (`4ff5adfd...`) matches the `.r1` family's 2900000 checkpoint used elsewhere
(consistent -- same trained model, test-set copy, not a separate training run, as already
documented in common.py / NOTES_tier1.md). Not re-adjudicated further this PR: the practical
answer (2900000 is the only usable choice) doesn't require resolving *why* the 3000000 dir is
nearly empty, but the underlying "which paper figure" question remains open per the mission's
own framing -- flagging forward, not re-deciding.

### 4. M2_case3test uses a different .r1 step (3000000) than the D3 gate (2900000)
Confirmed by sha256: `case4.200m.multi.stress.homo.a.Vw.case3.test/models.nmp10.cotopaxi.r1/
model-3000000.pt` (`3dbfb4ab...`) equals the working-dir's OWN `models.nmp10.cotopaxi.r1/
model-3000000.pt` (same hash) -- i.e. this is a genuine, deliberate step choice (3000000)
for the case3-test evaluation, distinct from the 2900000 step used for D3-fractal. Not an
ambiguity requiring owner input -- just documented so the two `.r1`-family steps are never
conflated with each other.

### 5. M3 checkpoint consistency (no ambiguity)
`M3_case3test` and `M3_case3othertest` checkpoints
(`models.nmp10.lr3e-5.b8.cotopaxi.r1/model-2700000.pt` under each test-set directory) are
both sha256 `aac65f7ba959533bfdae976c4d0ded6e0d01f2004bc13363ace14190dda19207`, identical to
M3's own checkpoint used for the existing "M3" (D2-148) gate. Confirms all three M3 rows
evaluate the SAME trained model on three different test sets, as the mission's inventory
states -- no separate-training-run ambiguity here.

### M1 checkpoint consistency (no ambiguity)
`case3.200m.homo.a.Vw.others/models.nmp10.cotopaxi/model-3000000.pt` sha256
`e0f97a34ab08e0add1e8d2e979999a8e7445c0fa8f60250c13ae4ba49f75f440` is identical to M1's own
checkpoint (`case3.200m.homo.a.Vw/models.nmp10.cotopaxi/model-3000000.pt`) -- same model,
different test set, as expected. `model-4000000.pt` (used for `M1_large`) is a later
checkpoint step of the same run (not independently verified against a second copy since
there is only one copy of that step in gns-sample; no ambiguity to investigate).

## What was NOT done (per mission scope, task 6)
No `run_gate.py` rollout re-run was performed against any new key -- that verification tier
(live GPU rollout vs baseline) is explicitly the conductor's job before merge, not this
session's. Only `extract_baselines.py` (reads existing published pkls, no GPU) was run.

## Held back before merge (conductor's fresh oracle re-run, 2026-09-27)

`M1_large`/`M1_small` were dropped from `common.py`'s `TEST_SET_REGISTRY` before this
branch was merged into `paper-parity-gate`. Fresh `run_gate.py --model M1_large
--cuda-device 1` (real gns-sample/ data, not a subagent report) gave `mse_raw` 0.208 ->
4.45 (20x, wall-clock 78.2s) with `n_nodes` 18564 in the fresh rollout vs 10302 embedded
in the `.large.published` baseline's own ground truth -- the published baseline was
generated against a different dataset than what
`case3.200m.homo.a.Vw.others/dataset/test.npz` currently is. `meshnet/train.py:56`
hardcodes `{data_path}test.npz` for rollout mode (not configurable per-call), and that
directory holds three npz files (`case3.200m.100m.npz`, sha256 `17f35dcf67a07e7d`,
byte-identical to the current `test.npz`; `case3.200m.small.npz`, sha256
`085f7ab83bdca768`; generic `test.npz`) -- `M1_small`, sharing the same `working_dir`,
would have silently rolled out the LARGE 18564-node mesh through a
small-fault-trained checkpoint instead of `case3.200m.small.npz`. This is a genuine
data-path/dataset-selection defect in how these two entries were wired, not a `meshnet`
code regression (confirmed the other 5 new pairs work correctly: `checkerboard` PASS
fresh, 39.6s, 2/2 trajectories within tolerance; the sha256 provenance claims above were
independently re-derived and matched). Fix needed before either can be registered: either
give each variant its own `dataset/` subdirectory with a correctly-named `test.npz`, or
confirm from `rollout.log.txt`/training logs which existing npz the `.large.published`
baseline was actually generated from and symlink/copy it into place. Left as an explicit
gap on `parity-coverage-matrix` (PATHWAY_FORWARD.md) rather than merged broken.

## 2026-09-27 addendum (branch `m1-small-fix`): M1_small fixed and registered, M1_large left alone

Conductor's fresh oracle re-run flagged one thing not yet acted on: file mtimes suggest
`case3.200m.100m.npz`, `case3.200m.small.npz`, and the small-fault published rollout
(`rollouts.nmp10.cotopaxi.small.D1.T_small/model-3000000.pt/rollout_0.pkl`) are ALL from
the same Nov-4-2025 batch, while `M1_large`'s published rollout
(`rollouts.nmp10.cotopaxi.large.published/model-4000000.pt/`) is from an EARLIER, May-13-2025
batch. This session verified the M1_small half of that hypothesis empirically rather than
just trusting the mtime coincidence:

**Node-count check (step 1).** Loaded `case3.200m.small.npz` directly (not via
`meshnet`): `trajectory0` is a dict of arrays shaped `(827, 1352, 2)` (pos/velocity),
i.e. 827 timesteps x 1352 nodes. Loaded the published small-fault pkl via
`common.load_pkl` + `common.py`'s own convention: `ground_truth_rollout.shape == (826,
1352, 2)` -- 1352 nodes, matching exactly (826 = 827 - 1, the usual rollout-vs-npz
off-by-one from dropping the first `INPUT_SEQUENCE_LENGTH`-th frame, consistent with how
`rollout()` derives `ground_truth_velocities` elsewhere in this repo's own conventions).
**MATCH confirmed** -> proceeded to wiring per the mission's step 1 branch, did NOT touch
`M1_large` (left exactly as documented above -- still no npz matches its 10302-node
baseline, still needs owner input, not attempted).

**Fix (step 2), file:line:**
- `test/paper_parity/common.py`: `model_paths()` (~line 320) now reads an optional
  `test_npz_name` field per registry entry (default `"test.npz"`), used to compute
  `test_npz` and returned as `paths["test_npz_name"]`. `TEST_SET_REGISTRY["M1_small"]`
  (~line 218) sets `test_npz_name: "case3.200m.small.npz"`, `working_dir`
  `case3.200m.homo.a.Vw.others`, `model_dir` `.../models.nmp10.cotopaxi`, `model_step`
  3000000, `published_rollout_dir` `.../rollouts.nmp10.cotopaxi.small.D1.T_small/
  model-3000000.pt`, `provenance: "unconfirmed"` (no `.published` suffix on that dir --
  warning left firing, not suppressed).
- `test/paper_parity/run_gate.py`: new `dataset_dir_for(paths)` contextmanager (~line 33,
  right after the module constants) -- when `paths["test_npz_name"] == "test.npz"` it's a
  no-op (yields the real `working_dir/dataset` unchanged, byte-for-byte the old
  behaviour for every other registry key); otherwise it builds a `tempfile.
  TemporaryDirectory()` scratch dir (never inside `gns-sample/`, always removed on exit)
  containing exactly one symlink, `test.npz` -> the real npz file, and yields that as
  `--data_path`. `run_current_rollout()` now wraps its `subprocess.run` call in
  `with dataset_dir_for(paths) as data_path:` instead of using `paths['data_path']`
  directly. Same non-invasive "test infra works around a train.py limitation, never
  patches it" pattern as PR #4's `truncated_rollout_cli.py`, but simpler here -- no
  monkeypatching needed, since the workaround is purely a filesystem-path trick
  (`meshnet/train.py` itself is untouched, still reads whatever `--data_path` says).

**Baseline extraction (step 3).** `python3 test/paper_parity/extract_baselines.py --model
M1_small` (fast, no GPU): wrote `baseline_M1_small.json`, 1 trajectory
(`rollout_0.pkl`): `mse_raw=0.0215`, `mse_vx=0.0430`, `mse_vy=2.55e-10` (~0, expected for
this in-plane problem, same convention as every other key), `rupture_time_rmse=0.0108`,
`missed=0`, `false=0` -- no NaNs. `test_npz_sha256` recorded as `085f7ab83bdca768...`,
matching the sha256 already cited above for `case3.200m.small.npz` (confirms
`model_paths()`'s new `test_npz_name` override is hashing the CORRECT file, not the
shared directory's generic large-fault `test.npz`).

**Real gate run (step 4).** `nvidia-smi` showed GPU 1 free-ish (6.7GB/40GB used, vs GPU 0
at 35GB/93% util) so ran on `--cuda-device 1` as instructed:

```
python3 test/paper_parity/run_gate.py --model M1_small --cuda-device 1
```

Printed warning fired as expected (`provenance is 'unconfirmed' -- NOT a confirmed
paper-parity oracle`, not suppressed). Command line confirmed the scratch-dir fix
actually engaged: `--data_path=/tmp/paper_parity_dataset_g6fjvfok/` (NOT
`case3.200m.homo.a.Vw.others/dataset/`), annotated `(scratch data_path,
test_npz_name='case3.200m.small.npz')`. Wall-clock: rollout 15.6s (single trajectory,
consistent with `checkerboard`'s comparable-sized 39.6s/2-trajectory precedent above).
Per-trajectory table:

| traj | status | tol_source | mse_raw (baseline->current, diff) | mse_vx | mse_vy | rupture_time_rmse | missed | false |
|---|---|---|---|---|---|---|---|---|
| rollout_0.pkl | PASS | global-fallback | 0.021508290->0.021508402 (d=1.12e-7 <= 0.02) | 0.043017->0.043017 (d=2.24e-7 <= 0.04) | 2.546e-10->2.546e-10 (d=5.18e-16 <= 1e-4) | 0.0107979->0.0107979 (d=0.0 <= 0.001) | 0->0 (d=0<=2) | 0->0 (d=0<=2) |

`OVERALL GATE: PASS`. Diffs are all ~1e-7 or smaller -- consistent with ordinary
GPU-kernel-order nondeterminism already characterized elsewhere in this test suite (PR
#3's per-trajectory tolerance work), not evidence of a dataset mismatch (a true
wrong-dataset run would look like `M1_large`'s 20x `mse_raw` blowup above, not a 1e-7
wobble). Scratch dir (`/tmp/paper_parity_dataset_g6fjvfok`) confirmed removed after the
run (no leftover directory); output pkl left in place at
`/tmp/paper_parity_gate_output/M1_small/rollout_0.pkl` per `run_gate.py`'s existing
(unchanged) `--work-root` default, not this session's concern to clean up.

`M1_large` was NOT touched this session -- still held back exactly as documented in the
"Held back" section above, still needing owner input on the May-13-2025 dataset's
availability. `meshnet/train.py`, `PROJECT_RULES.md`, `PATHWAY_FORWARD.md` were not
edited (board/production code out of scope for this branch).

## Coverage still open after this PR
- M2's own D2 test using the ACTUAL `.r1`/2900000 checkpoint: impossible to gate -- no
  published pkls exist for it (see ambiguity #1). This is a genuine gap in the published
  gns-sample bundle, not something this PR's registry wiring can fix.
- M2's own D3 gate step-choice ambiguity (2900000 vs 3000000, ambiguity #3) is still
  formally open per the mission text, though 2900000 is the only practical choice.

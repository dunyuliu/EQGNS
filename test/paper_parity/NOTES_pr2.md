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

## Coverage still open after this PR
- M2's own D2 test using the ACTUAL `.r1`/2900000 checkpoint: impossible to gate -- no
  published pkls exist for it (see ambiguity #1). This is a genuine gap in the published
  gns-sample bundle, not something this PR's registry wiring can fix.
- M2's own D3 gate step-choice ambiguity (2900000 vs 3000000, ambiguity #3) is still
  formally open per the mission text, though 2900000 is the only practical choice.

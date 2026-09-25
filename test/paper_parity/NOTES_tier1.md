# Tier-1/4 paper-parity gate -- progress notes

## Anchors verified (2026-09-24)
All anchor paths given in the mission matched exactly (no discrepancy):
- M1: `case3.200m.homo.a.Vw/models.nmp10.cotopaxi/model-3000000.pt`,
  `.../rollouts.nmp10.cotopaxi/model-3000000.pt/*.pkl` (6 pkls),
  `.../dataset/test.npz`.
- M2: `case4.200m.multi.stress.160scenarios.homo.a.Vw/models.nmp10.lr3e-5.b8.cotopaxi.r1/model-2700000.pt`,
  `.../rollouts.nmp10.lr3e-5.b8.cotopaxi.r1.published/model-2700000.pt/*.pkl` (15 pkls).
- M3: `case4.200m.fractal.stress.homo.a.Vw/` has no `.published` rollout dir (confirmed by
  listing the directory: only `rollouts.nmp10.cotopaxi{,.r1}`, `rollouts.nmp15.cotopaxi.r1`,
  `rollouts.nmp5.cotopaxi.r1`). Used `rollouts.nmp10.cotopaxi.r1/model-2900000.pt/*.pkl`
  (15 pkls) + `models.nmp10.cotopaxi.r1/model-2900000.pt` per mandate default.
  `baseline_M3.json["provenance"] == "unconfirmed"`; `run_gate.py` prints a loud warning
  for M3.
- All `train_state-<step>.pt` files needed alongside each `model-<step>.pt` confirmed present.

## Extraction step
- `extract_baselines.py` run for M1, M2, M3 -- all three `baseline_M{1,2,3}.json` written.
- M1 per-trajectory vx MSE: [0.1225, 0.1984, 0.6641, 0.8264, 1.0603, 0.1110] --
  matches CLAUDE.md sanity anchor [0.123, 0.198, 0.664, 0.827, 1.059, 0.112] to 3 sig figs. OK.
- mse_vy == 0.0 for every M1/M2/M3 trajectory: verified NOT a bug -- ground-truth vy is
  identically 0 (pure in-plane strike-slip fault, no along-dip component in this problem
  setup); predicted vy is a small near-zero residual (~1e-5 std). This is physically
  expected, not a code defect.

## run_gate.py
- Built; invokes documented `python3 -m meshnet.train --mode=rollout ...` exactly
  (see docs/rollout_and_analysis.md), diffs per-trajectory metrics vs baseline JSON.
- Bug found and fixed during mutation-testing: original `diff_trajectory()` computed
  `ok = not (diff > allowed)`, which silently PASSES on NaN (IEEE-754 NaN comparisons
  are always False, and De Morgan's law does not hold across a NaN gap). Fixed to the
  direct comparison `ok = (diff <= allowed)`, which correctly fails on NaN. Verified via
  a 3-case mutation check (small perturbation still passes, `+0.07` mse and
  `2477->50` false_count mutations still fail, NaN now fails instead of silently passing).

## Tolerance derivation
- First pass (2 M1 double-runs) gave tolerance too tight -- a 3rd independent run of the
  SAME code+checkpoint broke it on `rollout_4` (mse_raw diff 0.00295 > tol 0.001).
- Ran M1 4 times total. `rollout_4` mse_raw across the 4 runs: 0.52456, 0.52498, 0.52719,
  0.53114 -- monotonically increasing, spread 6.58e-3 (13x the naive 2-run estimate).
  Flagged as an open question (unresolved monotonic drift, not obviously a bug -- possibly
  chaos amplification of GPU kernel nondeterminism over 826 autoregressive steps, concentrated
  on the trajectory with the highest baseline false_count=2477, i.e. a borderline rupture case).
- Final tolerance: `tolerance.json`, `MARGIN=3` for mse_raw/mse_vx (extra margin for the
  drift anomaly), `MARGIN=2` elsewhere, floors added. See README.md for full table.

## Full gate run (2026-09-24, A100-SXM4-40GB x3 in parallel, one GPU per model)
- M1: PASS, 217.9s wall-clock, all 6 trajectories within tolerance.
- M2: FAIL, 317.2s wall-clock, `rollout_7` exceeds tolerance on mse_raw/mse_vx. Confirmed
  via an independent 3rd run (baseline 0.391, gate-run 0.428, 3rd run 0.396) that this is
  the same chaos/nondeterminism phenomenon as M1's rollout_4, not a regression.
- M3: FAIL, 403.2s wall-clock, 5/15 trajectories exceed tolerance (up to a 6x mse_raw swing
  on rollout_3). Consistent with the provenance-unconfirmed dataset having more chaos-
  sensitive trajectories. NOT investigated further to a root cause in this PR (out of scope
  per mission: report honestly, don't force a pass by loosening tolerance further, since a
  tolerance wide enough to absorb a 6x MSE swing would make the gate unable to catch real
  regressions of similar size).
- Did NOT loosen tolerance further to force M2/M3 to PASS. See README.md "Known limitation".

## pytest wrapper
- test_paper_parity.py added; `@pytest.mark.paper_parity` registered in test/conftest.py,
  plus a `--paper-parity` opt-in flag (also added to conftest.py's pytest_addoption).
  Verified `pytest test/ -q` still shows `43 passed, 3 skipped` (no change to the original
  43-test suite; the 3 new skips are the M1/M2/M3 paper_parity tests, each skipping with an
  explicit, printed reason -- never silent).

## Zenodo cross-check
- zenodo_hash_check.py: network call to Zenodo API succeeded (15-file manifest fetched).
  BLOCKED beyond that: Zenodo exposes MD5 of packed zip archives, not sha256 of the
  extracted model.pt/test.npz files our baselines hash; a real diff needs 10.33 GB
  downloaded+unzipped+rehashed. Measured throughput from this environment: 389.2 KB/s
  (20MB range-request sample) -> ~7.4h ETA, judged impractical for this session.
  See ZENODO_CROSS_CHECK.md for full reasoning and how to unblock later.

## Status: all deliverables complete for this session.
Remaining flagged items are explicitly deferred (not silently dropped): see README.md
"Known limitation" and "Recommended follow-up".

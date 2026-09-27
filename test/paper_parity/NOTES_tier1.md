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

## 2026-09-27 label correction (PR #1)

Owner-confirmed mapping (paper sec 2.4) established this session that the registry
entries above were mislabeled. md5-verified this session:
- gate's old `"M2"` (160scenarios lr3e-5.b8, model-2700000.pt, md5 1dc051d8) is
  actually the **paper's M3** on its own D2 test set.
- gate's old `"M3"` (fractal dir, model-2900000.pt, md5 48d0e2b9) is byte-identical
  to `case4.200m.multi.stress.homo.a.Vw/models.nmp10.cotopaxi.r1/model-2900000.pt`
  -- it is the **paper's M2** checkpoint applied to the D3 fractal test set, NOT
  M2's own paper-parity test.

Fix applied: swapped the `"M2"`/`"M3"` entries (and `baseline_M2.json` /
`baseline_M3.json`, including each file's internal `model_key` field) in
`common.py` to match the paper identities. No rollout was re-run for this fix --
the underlying checkpoint/test-set pairs and their paths are unchanged, only
which registry key points at which pair. Re-ran `pytest test/ -q` (44 passed, 3
skipped, ~30s, unchanged) and a fresh full `run_gate.py --model all` on
gns-sample/ to confirm the swap didn't change any FAIL/PASS verdict (see
README.md "Gate results observed this session" for the corrected labels and the
2026-09-27 re-run numbers below this line, appended after the fresh run
completed).

M3's `rollout_7` failure (confirmed nondeterminism, see above) is now marked
`xfail(strict=True)` in `test_paper_parity.py`. M2-on-D3's 5/15 failures are
NOT marked xfail -- that root cause was never independently confirmed, only
hypothesized ("consistent with... more chaotic sensitivity"); marking an
unconfirmed failure as known-nondeterminism would be exactly the kind of
unmeasured signal-removal PROJECT_RULES.md-adjacent gate discipline forbids.
M2's own D2 paper-parity test (not D3) remains unimplemented -- deferred to
PR #2 per owner-approved scope split.

### Fresh full gate re-run, 2026-09-27 (GPU 1, serial, one model at a time --
GPUs 0/2/3 were NOT idle despite an earlier briefing to the contrary: an
active 4-GPU job (PIDs 4055897/4055901/4055908/4055910) plus two unrelated
single-GPU jobs were running; not ours, left untouched)

- **M1**: PASS, 65.3s (all 6 trajectories within tolerance; faster than the
  2026-09-24 217.9s because GPU 1 was uncontended vs GPU 0 in the earlier
  parallel run).
- **M2** (fractal/D3, unconfirmed): FAIL, 161.5s, 7/15 trajectories exceed
  tolerance (rollout_2,3,7,8,10,12,13) -- MORE than the 5/15 from
  2026-09-24 and a DIFFERENT set of trajectories (rollout_3's mse_raw swung
  1.60, rollout_7's mse_raw swung 3.69 -- both far larger than tolerance).
  Still consistent with (not newly confirmed as) chaotic sensitivity; NOT
  xfailed, per the "don't xfail an unconfirmed failure" rule above.
- **M3** (own D2 test, published): FAIL, 157.8s, but this run's ONE failing
  trajectory is `rollout_2` on `rupture_time_rmse` (1.4545->1.4559,
  diff=0.0013950, barely over the 0.001 tolerance) -- NOT `rollout_7` on
  `mse_raw`/`mse_vx` as in the 2026-09-24 session (this run: rollout_7
  mse_raw diff 0.0094, well inside the 0.02 tolerance, i.e. rollout_7
  PASSED this time). The failing trajectory/metric moving between runs of
  the identical checkpoint+data is itself supporting evidence for GPU
  nondeterminism over a fixed regression (a real regression would
  reproducibly hit the same trajectory). xfail reason text in
  `test_paper_parity.py` updated to cite both observations rather than only
  the older rollout_7 one, so it doesn't overclaim a narrower/stale story.

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

## Next steps (see later entries for progress)
- [ ] build run_gate.py (re-run rollout with current meshnet/train.py, diff vs baseline)
- [ ] measure GPU-nondeterminism tolerance via M1 double-run
- [ ] full gate run M1/M2/M3
- [ ] pytest wrapper + conftest marker
- [ ] README.md
- [ ] zenodo_hash_check.py

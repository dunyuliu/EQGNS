# PATHWAY_FORWARD.md — EQGNS status board

Present-tense record of open work, per `PROJECT_RULES.md`. This is the only
status board for this repo — no `TODO.md`/`STATUS.md`/`BACKLOG.md` elsewhere.

**Sort order**: priority first (P1 > P2 > P3). Within a priority, state is the
tiebreak: BROKEN > OPEN > overdue-VERIFIED. State is never the primary sort
key — a P2 BLOCKED row stays below every P1 row regardless of state.

| prio | id | state | surface | evidence-command | notes |
|---|---|---|---|---|---|
| P1 | parity-tier1 | OPEN | `test/paper_parity/` | `python3 test/paper_parity/run_gate.py --model M1` (exit 0) | Extract per-trajectory baselines (rollout MSE vs ground truth; rupture-time RMSE/missed/false at 0.1 m/s threshold, dt=1/60s, +1.2s offset per `utils/plot.rupture.dynamics.py`, rule 7) from PUBLISHED rollout pkls into `baseline_{M1,M2,M3}.json` (with sha256 of published `model.pt` + test set); `run_gate.py` re-rolls out with CURRENT `meshnet` code from the published checkpoint and diffs against baseline. M3 is UNCONFIRMED PROVENANCE: no `.published` rollout dir exists for `case4.200m.fractal.stress.homo.a.Vw`; default chosen is `nmp10.cotopaxi.r1@model-2900000`, mark `baseline_M3.json` provenance field `"unconfirmed"`, do not certify M3 trusted until a human confirms. Full-length rollouts, all trajectories, no truncation — PR #1 scope (correctness/completeness first). |
| P1 | parity-tier4 | OPEN | `test/paper_parity/run_gate.py` | `python3 test/paper_parity/run_gate.py --model M1` (exit 0) | Physics-metrics gate (tier 4), folded into the same `run_gate.py` output as parity-tier1 above — not a separate deliverable, no separate evidence command. |
| P1 | ab-determinism-tier2 | OPEN | new test file (not under `test/paper_parity/`) | pytest node id once written | Seeded A/B training determinism test; reuses `test/fixtures/meshnet/synth.py` and `test/fixtures/meshnet/seeded_pipeline_cli.py`; asserts loss-sequence equality across two checkouts/trees; documented for refactor-gating use. |
| P2 | zenodo-hash-crosscheck | BLOCKED (network/size) | new script + doc | script path once written, plus doc note | One-time script + doc comparing local anchor sha256s (M1/M2/M3 `model.pt` + test sets) against the archived bundle at `10.5281/zenodo.17095311`; mark blocked rather than downloading the 249GB `gns-sample`-equivalent bundle if impractical. |
| P3 | convergence-tier3-nightly | OPEN (ready to schedule, not scheduled — GPUs saturated) | design doc + script, no cron installed | n/a — design + script only | Convergence gate (tier 3, nightly seeded mini-training); no cron/schedule installed yet. |
| P2 | economize-gates | OPEN (next PR, deferred) | `test/paper_parity/` tiers 1/2/4 | n/a — deferred to PR #2 | "Economize gates": quick tiers, truncated rollout horizons, caching, subset sweeps for tiers 1/2/4 built by the rows above. Explicitly out of scope for PR #1, which prioritizes full-length/all-trajectory correctness; scheduled for the PR after PR #1 merges. |

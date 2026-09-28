# PATHWAY_FORWARD.md — EQGNS status board

Open work, priority order (P1 > P2 > P3). A row closes on a command that ran.

| prio | id | state | evidence command | notes |
|---|---|---|---|---|
| P1 | release-v1.1 | OPEN | `pytest test/ -q` + `gate.py run --cuda ...` on the tagged SHA | Cut a release for the test system (paper-parity gate, quick tier, GitHub Actions). |
| P1 | prepare-loop-fix | OPEN | regenerate D1; check frames 755-826 hold real geometry | `utils/prepare.eqdyna.4gns.py:369` and `utils/prepare.fractal.stress.eqdyna.4gns.py:334` loop `range(timestep-nskip)` after `timestep -= nskip` already: last 72 frames stay all-zero (pos, cells, velocity, node_type, node_property). Fix: `range(timestep)`. Raw EQdyna output: `/home/utig5/dliu/eqdyna.scenarios.for.gns/`. |
| P1 | m1-retrain-fixed-data | OPEN | rollout metrics on the fixed D1 test set, both models | Retrain M1 on fixed D1, plus a control retrained on the old D1 with identical code/settings, to separate data effect from training noise. Compare at matched checkpoints. |
| P2 | m1-large-gate | OPEN | `gate.py run M1_large` | 40 km fault test set can be regenerated from `eqdyna.scenarios.for.gns/case3.200m.homo.a.Vw.others/tpv104.200m.H14.large`. |
| P3 | convergence-gate-nightly | OPEN (at release) | n/a | Scheduled at release time per owner. |
| P3 | noise-py-indexing | OPEN | n/a | `meshnet/noise.py` uses `graph.x[:, 1:3]` for velocity (layout puts it at 2:4); harmless, only `.shape` used. |

# M2 / M3 mirror augmentation — experiment design

Board row: `m2m3-mirror-augmentation` (PATHWAY_FORWARD.md, owner-approved
queue item 2, 2026-10). Design only; no training was launched. Numbers marked
*measured* come from CPU reads of existing `.npz` and metadata files on
2026-10-09. The mirror transform, its physical justification and the M1
measurements are defined in `M1_ARRESTING_MIRROR_EXPANSION_DESIGN.md`
section 2.2 and are not repeated. Separate document because this row needs
**no new EQdyna scenarios**, runs at M3 cost (3–6 days per arm), and should be
scheduled with the GH200 M3 sweep rather than the local M1 queue.

## 0. Verdict up front

**Feasible at zero EQdyna cost; worth doing on M3 only, as a learning-curve
design, and only if it can share the M3 sweep's two-seed b8 baseline.** The
transform is exact for the M2/M3 data structures (*measured* on the three
mirror pairs that already exist inside the M3 training set: `node_property`
and `node_type` match under the mirror permutation with 0 mismatches, vy = 0,
and the EQdyna truth asymmetry is 4.0e-3 to 4.6e-3 rms/peak — one tenth of GNS
rollout error). The owner has ruled M2 **not gated** (small train set,
unstable), so an M2 arm would produce a seed-noise-dominated number; it is
listed as optional. The scientifically useful question on M3 is not "does
doubling by mirroring help the 140-scenario model" (expected: little) but
"**how many real scenarios does mirroring replace**": train on 70 real +
mirror and compare with 140 real and 70 real. Cost: 3 new M3 arms x 2 seeds at
the published budget (21.6M samples = 2.7M steps at b8) ≈ **6 x 145 A100-h ≈
36 days sequential locally**, or the GH200 equivalent (sweep pilot throughput
pending). This row therefore only makes sense (a) on GH200 behind the M3 sweep,
reusing its A0a/A0b baseline, or (b) at a reduced budget decided by the owner
(section 6).

## 1. What exists on disk (measured)

| Item | Path | Facts |
|---|---|---|
| M2 dataset `case4.200m.multi.stress` | `data/gns-sample/case4.200m.multi.stress.homo.a.Vw/dataset/` | train 30 / valid 10 / test 10 (D2); hypocenters {(±6,-7), (±6,-2), (0,-5)} km; one 2 km half-size asperity at random grid positions; stress levels 35/45/50/55 MPa encoded per node in `node_property`; node_type embedding size 1 |
| M3 dataset `D2_160` | `data/gns-sample/case4.200m.multi.stress.160scenarios.homo.a.Vw/dataset/` | train 140 (27.8 GB `train.npz`), same test D2; **3 within-train mirror pairs** (trajectories 46/58, 113/120, 124/105 in metadata order) |
| M3 test sets in `gate.py` | `M3_D3` (fractal stress, `D2_160/test.npz`), `M3_D1hypo` (`D2_160.case3.test`) | leakage check for D1hypo not yet done (Phase 0) |
| M2 → train mirror leakage | metadata | **0 of 10** D2 test configs have their mirror in M2 train |
| M3 → D2 | metadata | all 10 D2 test configs are **literally** in M3 train (known; D2 is not an M3 gate case) |
| Published M3 run | `D2_160/models.nmp10.lr3e-5.b8.cotopaxi.r1`, `model-2700000.pt` | b8, lr 3e-5, decay 0.1/20M, noise 2e-2, 21.6M samples; 194 ms/step on a contended A100 |
| M3 sweep design | `docs/dev/M3_BATCH_LR_SWEEP_DESIGN.md` | GH200 from-scratch factorial with a two-seed b8 baseline (A0a/A0b) — the baseline this row should reuse |

### 1.1 Truth asymmetry on M3 mirror pairs (measured, D2_160 train)

| Pair (metadata idx) | hypocenter | asperity x (km) | rms/peak | max/peak | ruptured-node disagreement under mirror | control (same idx, no mirror) rms/peak |
|---|---|---|---|---|---|---|
| 46 <-> 58 (H4.50MPa.7 / H3.50MPa.2) | (6,-2)/(-6,-2) | -5.57 / +5.57 | 4.4e-3 | 0.32 | 4 of 4692 | 6.0e-2 |
| 113 <-> 120 (H4.35MPa.6 / H3.35MPa.0) | (6,-2)/(-6,-2) | -6.5 / +6.5 | 4.0e-3 | 0.32 | 4 of 4692 | 5.5e-2 |
| 124 <-> 105 (H4.35MPa.7 / H3.35MPa.4) | (6,-2)/(-6,-2) | 0 / 0 | 4.6e-3 | 0.32 | 13 of 4692 | 5.4e-2 |

`node_property` mirrored match: max |diff| = 0 in all three; `node_type`: 0
mismatches; vy peak 0. So the transform is exact on the inputs and the label
noise it introduces (~4e-3 rms/peak) is ~7 % of the between-scenario
difference and ~10 % of GNS error. After mirroring, these three pairs become
near-duplicates (6 of 280 trajectories) — negligible.

## 2. Scientific question, hypotheses, material effect

**Q.** At matched sample budget, does x-mirroring the training set (i) improve
M3 on its gate cases (M3_D3 fractal stress, M3_D1hypo), and (ii) substitute for
real scenarios, i.e. does 70 real + mirror reach 140-real quality?

A priori:

- **H1** — 140 real + mirror vs 140 real: within the two-seed b8 floor on the
  primary metrics (no material gain; the model already sees both signs of
  hypocenter x and asperity position in M3's 140 scenarios).
- **H2** — 70 real + mirror vs 70 real: material gain, and 70 real + mirror is
  within the floor of 140 real on M3_D3. If true, mirroring halves the EQdyna
  cost of future scenario campaigns (the actionable outcome).
- **Kill:** if 70 real + mirror is *not* better than 70 real beyond the floor
  in both seeds, mirror augmentation is dropped for M2/M3 and the only
  remaining use is the M1 substitution arm (A3' in the M1 design).

**Material effect** = beyond the two-seed b8 baseline spread (from the M3
sweep's A0a/A0b, or from `fixed`-style local seeds if run locally), same sign
in both seeds, on: rt_rmse, missed/false, delta_mw, mse_vx (M3 mse_vx is less
seed-noisy than M1's, but the collapse guard and the known chaotic
bifurcation case M3_D3 traj 7 — excluded by `REGRESSION_EXCLUDE_TRAJ` — apply).

## 3. Design

### 3.1 Phase 0 (CPU, minutes, no GPU)

- Leakage check for `M3_D1hypo` (`D2_160.case3.test`) vs M3 train under
  mirror (metadata only). Fractal `M3_D3` cannot leak (random fields).
- Build the mirrored datasets with the offline script shared with the M1
  design (`scripts/utils/mirror_augment.py`): `D2_160_mirror/train.npz`
  (280 trajectories, ~56 GB) and `D2_70/`, `D2_70_mirror/` (a fixed random
  half of the 140, seed recorded, chosen so the three within-train mirror pairs
  are split one member each side — keeps the 70-real set free of mirror
  duplicates). valid/test untouched.
- Optional: M2 variants `D2_mirror/` (60 train) — only if the M2 arm is
  approved.

### 3.2 Run matrix (M3; hyper-parameters = published b8 / lr 3e-5 / noise 2e-2)

| Arm | Train | Seeds | Budget (samples) | Cost | Answers |
|---|---|---|---|---|---|
| B0 | 140 real | 2 | 21.6M | **reuse M3 sweep A0a/A0b** (GH200) or 2 x 145 A100-h | floor |
| B1 | 140 real + mirror (280) | 2 | 21.6M | 2 x 145 A100-h | H1 |
| B2 | 70 real | 2 | 21.6M | 2 x 145 A100-h | learning-curve point |
| B3 | 70 real + mirror (140) | 2 | 21.6M | 2 x 145 A100-h | H2 |
| (B4, optional) | M2 30 real vs 30 + mirror | 2 + 2 | published M2 budget | ~4 x M2 cost | owner says M2 not gated — low value |

Matched **samples** (not steps) is primary, consistent with the M3 sweep
design (section 2.3 there). Local A100 cost for B1–B3: 6 x 145 h ≈ 36 days
sequential — too long for the local one-job rule; GH200 cost follows the sweep
pilot's measured ms/sample (not yet available at design time). A reduced
budget (e.g. 10.8M samples ≈ 1.35M steps at b8, ~3 days per run locally,
≈ 18 days total) is a possible owner choice but then B0 must be re-run at the
same budget (its 2.7M checkpoints are not comparable), adding 2 runs.

Phased: B3 and B2 first (they test the actionable hypothesis H2 and are
independent of B1); B1 last and only if budget remains.

### 3.3 Confound control

- Identical hyper-parameters across arms; only the data path changes.
- Evaluation checkpoint fixed a priori at the matched-sample step; validation
  loss not used for selection (M3 valid is not mirror-leaked by metadata, but
  the rule is kept uniform with the M1 design).
- Same GPU family for all arms of a comparison (do not compare a GH200 B0
  with a local A100 B3 — the sweep design documents hardware as a confound).
- The 70-real half is one fixed draw; report its id list. Sensitivity to the
  draw is not tested (would double B2/B3) — stated as a limitation.

## 4. PROJECT_RULES.md rule 1

Offline data construction only; `train()`/`validation()`, the loader and
`meshnet/train.py.published` are untouched. **No rule-1 exception is
requested.**

## 5. Go/no-go

| Gate | Condition |
|---|---|
| Dispatch at all | M3 sweep baseline (two-seed b8) exists or is funded; otherwise B0 adds 2 runs and the row costs 8 M3 trainings |
| After B2 + B3 | H2 holds in both seeds → run B1; H2 fails in both → report negative, do not run B1 |
| M2 arm | only on explicit owner request |

## 6. Open questions for the owner

1. **Venue and budget.** GH200 behind the M3 sweep (reusing A0a/A0b, matched
   21.6M samples) vs local A100 at a reduced budget with its own B0 — the
   first is cheaper per answer; the second needs ~5 weeks of the local GPU.
2. **Is H2 the question you want?** The design re-frames "mirror on M2/M3"
   as a data-efficiency learning curve because the full-data arm (B1) is
   expected to be null. If the owner wants only B1 (direct "does mirroring
   help M3"), the matrix shrinks to 2 runs but the likely outcome is "within
   floor".
3. **M2 arm.** Drop (recommended, consistent with the "M2 not gated" decision)
   or run as a cheap pilot before committing M3 time?
4. **Disk.** +56 GB for `D2_160_mirror` plus ~28 GB for the 70-real variants
   under the gitignored data root — fine locally (25 TB free); confirm the
   GH200 scratch quota if shipped there.

## 7. What a reviewer would attack

- *"Mirroring a data set that already contains both signs of every
  configuration is not augmentation, it is duplication with noise."* Agreed
  for B1 — that is H1. The informative arm is B3.
- *"The 70-real draw is arbitrary."* One draw, id list reported; variance
  across draws not measured (cost).
- *"Truth asymmetry of 4e-3 rms is label noise you are injecting."* It is
  one tenth of model error on M3 gate cases; stated next to every number. If
  B1 is *worse* than B0 beyond the floor, this is the first suspect.
- *"M3_D3 traj 7 is chaotic."* Excluded by the existing regression rule; the
  same exclusion is applied to every arm.
- *"The GH200 throughput is unknown at design time."* Correct; the cost
  column is A100-based and labelled contended (194 ms/step measured with
  other jobs on the box).

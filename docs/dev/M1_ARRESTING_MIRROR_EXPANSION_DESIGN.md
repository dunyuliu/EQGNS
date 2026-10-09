# M1 arresting-case expansion + mirror augmentation — experiment design

Board row: `m1-arresting-mirror-expansion` (PATHWAY_FORWARD.md, owner-approved
queue item 1, 2026-10). Design only. No training, no EQdyna run was launched to
produce this document; every number below that is marked *measured* came from
CPU-only reads of existing `.npz` files on 2026-10-09 (scratch scripts, not
committed). The companion row `m2m3-mirror-augmentation` has its own document,
`M2M3_MIRROR_AUGMENTATION_DESIGN.md`, because it needs no new EQdyna scenarios
and runs on different hardware/queue; the mirror transform and its measured
label noise are defined once, here (section 2.2), and referenced from there.

## 0. Verdict up front

**Feasible, and the arresting expansion is the part that matters.** The M1
training set contains **zero** arresting ruptures: of the 19 D1 scenarios, only
H0 (-6, -8 km) and H16 (+6, -8 km) arrest, and both sit in the published test
set (*measured*, section 1.2). The model has never seen an arrest during
training, which is the simplest explanation for the known spurious
re-rupture on test trajectory 4 (`docs/user/rollout_and_analysis.md`). So the
first question is not "does mirroring help" but "does the model learn arrest at
all once it is shown some" — and that requires **new EQdyna scenarios**
(hypocenters near the fault edges/bottom corners, where boundary clipping of
the 3 km nucleation patch is what arrests M1). Mirroring is a free, purely
data-level transform (no `train.py` change, section 4) whose truth-label noise
is measured at 2.5e-3 to 1.3e-2 of peak slip rate (section 2.2) — well below
current GNS rollout error (~4e-2 to 8e-2 of peak) — but it **leaks** 2 of 6
published test and 3 of 3 validation scenarios into training (section 3.3), so
mirror arms are scored on the 4 leakage-clean test trajectories and may not use
validation loss to pick a checkpoint.

Recommended matrix: 4 arms x 2 seeds (+ free 3-seed baseline), 1M steps each at
batch 2 → **~130 GPU-hours ≈ 5.5 days sequential on one local A100**, plus a
**~40 min CPU scoping scan of ~24 candidate hypocenters with EQdyna** (Phase 0,
section 3.1) — the only heavy CPU step, and it decides go/no-go before any GPU
time is spent. Owner decisions needed before dispatch are in section 6.

## 1. What exists on disk (measured 2026-10-09)

### 1.1 Directories

| Item | Path | Note |
|---|---|---|
| Scenario-generation directory (**found**) | `/home/utig5/dliu/eqdyna.scenarios.for.gns/` | `case3.200m.homo.a.Vw/{tpv104.200m.ref, tpv104.200m.H0..H19, create.dr.trainset.case3.200m.py}`, `case3.200m.homo.a.Vw.others/`, `case4.200m.multi.stress.homo.a.Vw/` (194 entries), a stale `prepare.eqdyna.4gns.py`, `README.txt`. New scenarios go here and **only** here. |
| EQdyna source tree (**read-only**) | `/home/utig5/dliu/EQdyna/` | `bin/eqdyna` is v5.24.0 (2026-10-07). `eqdyna` is not on `PATH` in a non-login shell; call the binary by absolute path. |
| M1 published dataset (buggy nskip) | `data/gns-sample/case3.200m.homo.a.Vw/dataset/` | 827 frames x 4743 nodes per trajectory; train 10 / valid 3 / test 6 |
| M1 fixed dataset (PR #5) | `/home/utig5/dliu/eq_rupture_gns_data/D1_fixed/dataset/` | the correct training base; same split |
| M1 retrain baselines | `/home/utig5/dliu/eq_rupture_gns_data/m1_retrain/fixed_seed{0,1,2}/models/` | D1_fixed, batch 2, checkpoints every 50k to 1.35M (parked by owner 2026-10-08) |
| Launch command (reused verbatim) | `/home/utig5/dliu/eq_rupture_gns_data/m1_retrain/to3M/launch_one.sh` | `venv/bin/python3 -m meshnet.train --mode=train --batch_size=2 --nsave_steps=50000 --seed=$seed ...`, `CUDA_VISIBLE_DEVICES=1` |
| Preprocessor (maintained copy) | `scripts/utils/prepare.eqdyna.4gns.py` | `create_train_data(caseName, ...)`; the `'3.200m'` block does `random.seed(15)` split. Use this copy, not the one in the scenario dir. |
| Metrics | `tests/paper_parity/gate.py::metrics`, `measure_vs_published.py::compare_pair`, `eval_m1_retrain.py` | mse_vx, rt_rmse, missed/false (THRESHOLD 0.1 m/s), delta_mw, collapse guard |
| Measured training cost | `to3M/measure_result.txt` | **60.2 ms/step single run; 172.7 ms/step each with 3 concurrent** → concurrency buys ~4 %, run sequentially |

### 1.2 The M1 scenario family and its arrest inventory

M1 = `case3.200m`: TPV104-style vertical strike-slip fault, elastic, homogeneous
RSF (a 0.01, b 0.014, Dc 0.4 m, Vw 0.1 m/s, fw 0.2, f0 0.6, V0 1e-6),
sigma_n 120 MPa, tau0 40 MPa, fault x in [-9, 9] km, z in [-10, 0] km, dx 200 m,
nucleation radius 3 km, overstress 45 MPa over 1 s, 15 s termination,
dt 1/60 s, first 1.2 s skipped. 20 hypocenters on the grid x in {-6,-3,0,3,6}
x z in {-8,-6,-4,-2} (H index = ix*4 + iz); H10 (0, -4) unused.

Per-trajectory inventory of D1_fixed (rupture = peak |vx| > 0.1 m/s; vy is
identically zero in every trajectory because the preprocessor only fills
`velocity[..., 0]`):

| Split | Scenarios (H index) | ruptured fraction of fault nodes |
|---|---|---|
| train (10) | H19, H4, H17, H8, H7, H9, H11, H13, H15, H2 | all >= 0.992 |
| valid (3) | H12, H18, H3 | all >= 0.992 |
| test (6) | traj 0 H14, traj 1 H5, traj 2 H1, traj 3 H16, traj 4 H0, traj 5 H6 | **H0 0.478** (last active 5.55 s, ruptured x in [-9, 1.6] km), **H16 0.556** (9.63 s, x in [-4.4, 9] km); others >= 0.992 |

**Definition used throughout — "arresting" scenario:** ruptured node fraction
< 0.9 at 15 s (the empirical gap is 0.556 to 0.992, so any threshold in
[0.6, 0.95] gives the same labelling). "Marginal" = 0.9–0.99, none exist yet.

Why H0/H16 arrest and H4/H12 (±3, -8) do not: at (±6, -8) the 3 km nucleation
disk is clipped by both the bottom edge (z = -10) and the strike edge
(x = ±9), losing enough of its overstressed area that the rupture dies after
~4–8 s; at (±3, -8) only the bottom clips it. Arrest in M1 is therefore a
**boundary-clipped-nucleation** phenomenon. No other knob in the M1 definition
is varied (node_property is a constant 0.0 for every M1 node), so the only
arrest-producing lever that keeps the model's input definition unchanged is the
hypocenter position.

## 2. Scientific question, hypotheses, "material effect"

### 2.1 Questions

Q1 (expansion): if training includes arresting scenarios, does the GNS
predict arrest (position and time) on **held-out** arresting scenarios, and
does it stop spuriously re-rupturing?

Q2 (mirror): does reflecting every training trajectory across x = 0 help at
matched optimizer budget, and in particular can a mirrored one-sided set of
arresting scenarios substitute for running the other side in EQdyna?

### 2.2 The mirror transform (defined once, used by both design docs)

For a trajectory dict `{pos, node_type, node_property, velocity, cells,
pressure}`: `pos[..., 0] *= -1`; everything else unchanged. Justification:
the fault is a vertical plane with symmetric boundary conditions and symmetric
homogeneous friction; reflecting x negates the strike coordinate, and for the
kept slip sense vx'(x) = vx(-x), vy'(x) = -vy(-x). vy = 0 in all M1 (and M2/M3)
data so the sign flip is moot; node types (fault / surface / other boundary)
are symmetric in x; `cells` are index lists over the same nodes so they need no
change; per-node `node_property` stays attached to its node and therefore
moves with it. The model has no absolute position in its node features (edge
features are relative displacements + norm, `meshnet/learned_simulator.py`), so
it is translation-invariant but **not** reflection-equivariant — mirroring is
genuinely new data to it.

**Truth is not exactly mirror-symmetric** (*measured*, D1_fixed, rms and max of
|vx(x) - vx_pair(-x)| over all nodes and frames, normalised by peak vx):

| Pair | rms / peak | max / peak | note |
|---|---|---|---|
| H8 vs itself mirrored (x = 0 hypocenter) | 4.5e-3 | 0.30 | pure numerical asymmetry floor |
| H9 vs itself mirrored | 2.5e-3 | 0.31 | |
| H7 <-> H15 (train, train) | 2.5e-3 | 0.25 | |
| H6 <-> H14 (test, test) | 3.1e-3 | 0.34 | |
| **H0 <-> H16 (arresting)** | **1.28e-2** | 0.56 | arrest position differs by ~2.8 km, duration by 4 s |
| control: H0 vs H1 (not a pair) | 7.8e-2 | 1.61 | |

Reading: for propagating ruptures a mirrored label carries ~3e-3 rms noise,
roughly one tenth of current GNS rollout error on M1_D1 (~4e-2 to 8e-2 of
peak, `tests/paper_parity/reference.json`). For **arresting** ruptures the
asymmetry is 4x larger because the solution is near a bifurcation; that same
sensitivity is the physical reason arrest is hard to learn, and it bounds the
best achievable arrest-position agreement at ~0.08 in ruptured fraction.

### 2.3 A priori hypotheses

- **H1** — Baseline (no arresting training data) over-propagates on held-out
  arresting scenarios: ruptured-fraction error > 0.3 and/or spurious re-rupture
  on >= half of them. (Direction only; this is the control expectation.)
- **H2** — Adding real arresting scenarios (both x-signs) reduces
  ruptured-fraction error on held-out arresting scenarios **materially**
  (2.4) and does not degrade the 4 leakage-clean published test trajectories
  beyond the seed floor.
- **H3** — Mirror augmentation on top of real data gives at most a small gain
  (it adds no new physics), **except** in the substitution arm: one-sided real
  arresting scenarios + mirror should recover most of the two-sided gain. If it
  does, future scenario campaigns need only run one side.
- **Kill criteria:** if A1 fails H2 in both seeds, hypocenter-only arrest
  expansion does not work and the row is reported negative (the next lever —
  varying tau0 or nucleation strength — changes the physics and needs a
  node_property channel, i.e. it is an M2-type model, out of scope here). If
  A2 is worse than A0 beyond the seed floor on the clean test subset, mirror
  augmentation is dropped from M1 and the M2/M3 design is re-examined before
  dispatch.

### 2.4 "Material effect"

Baseline floor = max pairwise difference among the three A0 seeds
(`fixed_seed{0,1,2}` at the matched step) on each primary metric. An arm's
effect is material only if (i) both of its seeds move the same direction and
(ii) the arm's seed-mean differs from the A0 seed-mean by more than the A0
floor. Primary metrics (outcome-level, per trajectory, averaged per set):
ruptured-fraction error, arrest-outcome confusion (missed / false, existing
`gate.py` fields), rupture-time RMSE (`rt_rmse`), final slip / Mw error
(`delta_mw`), spurious re-rupture count (any node re-crossing 0.1 m/s after
> 1 s quiescence). Secondary: mse_vx (known to be seed-noisy, up to ~30x
spread at 500k steps, `docs/dev/M1_RETRAIN_RESULTS.md`).

## 3. Design

### 3.1 Phase 0 — scoping (CPU only; the one heavy job while it runs)

**(a) EQdyna version check (mandatory, ~2 min wall).** Re-run one existing
scenario (H14, a clean test case) with the current binary
`/home/utig5/dliu/EQdyna/bin/eqdyna` into
`eqdyna.scenarios.for.gns/case3.200m.homo.a.Vw.arrest/tpv104.200m.H14.v5.24/`,
preprocess with the repo `prepare.eqdyna.4gns.py`, and compare to the stored
D1_fixed trajectory. Threshold: rms/peak <= 3e-3 (the self-mirror numerical
floor above). The stored scenarios were generated 2025-05-06 (~850 EQdyna
commits ago). **If the check fails**, mixing old and new scenarios injects a
systematic version difference larger than the mirror noise we are trying to
measure; then regenerate all 19 D1 scenarios + the new ones with the pinned
v5.24.0 binary ("D1v2", ~25 min wall at `-np 40`) and retrain A0 on D1v2
(+2 seeds x 1M steps, +33 GPU-h) — a decision for the owner (6.1).

**(b) Arresting-hypocenter scan (~24 scenarios x ~70 s at `-np 40` ≈ 30–40
min wall, plus preprocessing).** Candidates on the 200 m grid, chosen to vary
how much of the nucleation disk is clipped: strike-edge/bottom corners
x in {±8, ±7, ±6, ±5} x z in {-9, -8.5, -8} (24 runs); if fewer than 6 arrest,
extend to x in {±8, ±7} x z in {-6, -4, -2} (strike-edge only clipping) and
x in {±4} x z in {-9.5, -9}. Keep every run (propagating ones document the
arrest boundary). Target: >= 10 arresting scenarios per side, so that 2 per
side can be held out for test. Compute rule: check `uptime` and `nvidia-smi`
before launching; do not run concurrently with any training job; use
`-np <= 32` if the 1-min load exceeds 32 (load was 85/64 at 16:37 on
2026-10-09 — do not start under that).

**(c) Node-type symmetry check (seconds).** Assert `node_type` is identical
under the x-mirror node permutation for an M1 trajectory (verified for M3
pairs: 0 mismatches of 4692; not yet asserted on M1).

**Go/no-go after Phase 0:** >= 6 arresting scenarios per side → go. Fewer
than 6 total → the clipped-nucleation lever is too narrow; report negative,
do not spend GPU.

### 3.2 Datasets

| Name | Content | Path (gitignored data root) |
|---|---|---|
| D1 | D1_fixed as is | existing |
| D1A | D1 train + arresting scenarios from both sides (N_a ≈ 16–20), valid unchanged | `/home/utig5/dliu/eq_rupture_gns_data/D1_arrest/dataset/` |
| D1A-L | D1 train + arresting scenarios from the **x < 0 side only** | `.../D1_arrest_left/` |
| D1-M, D1A-M, D1A-L-M | the above with every training trajectory also included mirrored (2x train size) | `.../*_mirror/` |
| T_arr (test) | 2 arresting per side **held out** (never in any train set or their mirrors), 4 scenarios | `.../D1_arrest/dataset/test_arrest.npz` |
| T_pub (test) | published D1 test (6); report the leakage-clean subset {traj 0, 3, 4, 5} separately | existing |

Mirroring is an offline script (`scripts/utils/mirror_augment.py`, new, ~40
lines: load npz, append `trajectory{N+i}` with `pos[..., 0]` negated, write).
Scenario → npz uses the repo `prepare.eqdyna.4gns.py` with an explicit list of
scenario directories (the `'3.200m'` block's `random.seed(15)` split is left
untouched; new scenarios get their own block or a CLI list — additive).
Split rule: mirror pairs always land in the same split.

### 3.3 Leakage (measured, must be stated in every result table)

Mirror pairs: H0<->H16, H1<->H17, H2<->H18, H3<->H19, H4<->H12, H5<->H13,
H6<->H14, H7<->H15; H8–H11 self-mirror. Under mirror augmentation of the D1
train split: test traj 1 (H5) and traj 2 (H1) have their mirrors (H13, H17) in
train; all three valid scenarios (H12, H18, H3) have mirrors (H4, H2, H19) in
train. Consequences: (i) mirror-arm metrics on T_pub are reported on the clean
subset {0, 3, 4, 5} and on all 6, labelled; (ii) validation loss is **not**
used for checkpoint selection in any arm — the evaluation checkpoint is fixed
a priori at the final step (and 500k as a secondary point).

### 3.4 Run matrix

Common: `meshnet.train`, batch 2, seed explicit, same hyper-parameters as the
`fixed_seed*` baselines, 1M steps, checkpoints every 50k, one GPU
(`CUDA_VISIBLE_DEVICES` in {0, 1, 3}; **never 2**), one job at a time.

| Arm | Train data | Seeds | Steps | Cost (GPU-h at 60.2 ms/step) | Answers |
|---|---|---|---|---|---|
| A0 | D1 | 0, 1, 2 (exist to 1.35M) | 1M | **0** | floor, H1 |
| A1 | D1A | 2 | 1M | 33 | H2 |
| A2 | D1-M | 2 | 1M | 33 | mirror alone |
| A3 | D1A-M | 2 | 1M | 33 | H3 (additive) |
| A3' | D1A-L-M | 2 | 1M | 33 | H3 (substitution; compare to A1 on T_arr right-side cases) |
| | | | **total** | **~133 GPU-h ≈ 5.5 days sequential** | |

Matched **optimizer steps** is the primary comparison (same compute). Mirror
arms see each real sample half as often; an epoch-matched secondary read
(A2/A3 at 2M steps) is optional and doubles their cost — not recommended
unless A2 is within the floor of A0 at 1M.

Phased dispatch with gates: A1 first (both seeds, 66 h) → if H2 fails in both
seeds, stop (negative result); else A2, A3, A3'. Rollouts: `det_rollout.py` /
`eval_m1_retrain.py` pattern, ~24 ms/step/traj on an idle A100, negligible.

### 3.5 Confound control

- Same binary for all new scenarios; version check 3.1(a) before mixing with
  old ones.
- Same preprocessor copy, same nskip, same frame count (827).
- Seeds explicit; two per arm minimum; 3-seed floor from A0.
- Fixed evaluation checkpoint, chosen a priori (no validation-selected
  checkpoint — validation is leaked under mirroring).
- Held-out arresting test cases exist on **both** sides so A3' can be scored
  on the side it never saw in real data.
- The training set grows (10 → ~28 → ~56 trajectories) but the model input
  definition (12 node features, constant node_property) is unchanged, so every
  arm is still "M1" in the sense of `gate.py`.

## 4. PROJECT_RULES.md rule 1

No change to `train()`/`validation()` or to the data loader is required: both
the arresting expansion and the mirror augmentation are offline `.npz`
operations consumed through the unchanged `--data_path`. `meshnet/train.py.published`
is untouched. The only repo additions are a small offline script
(`scripts/utils/mirror_augment.py`) and, optionally, an explicit-scenario-list
entry point in `prepare.eqdyna.4gns.py`. **No rule-1 exception is requested.**
Should a later iteration want on-the-fly mirroring inside the loader (saves the
2x disk), that *would* touch training code and must be raised as a rule-1
exception first; it is not part of this design.

## 5. Budget and go/no-go summary

| Step | Resource | Wall | Gate |
|---|---|---|---|
| 0a version check | 40 CPU cores, 2 min | 5 min | rms/peak <= 3e-3 else owner decision 6.1 |
| 0b hypocenter scan | 40 CPU cores x ~24 runs | ~40 min (+ ~20 min preprocessing) | >= 6 arresting per side |
| 0c node-type assert | CPU, seconds | — | pass |
| A1 | 1 A100 | 66 h | H2 in both seeds, else stop |
| A2, A3, A3' | 1 A100 | 100 h | — |
| rollouts + tables | 1 A100, CPU | ~2 h | — |

Disk: D1 train ≈ 15 GB; D1A-M ≈ 80 GB worst case; 25 TB free. If the version
check fails and D1v2 is chosen: +25 min CPU, +33 GPU-h for A0 retrain.

## 6. Open questions for the owner (needed before dispatch)

1. **EQdyna version mixing.** If Phase 0a shows the v5.24.0 binary differs from
   the 2025-05 scenarios by more than 3e-3 rms/peak: accept the mix (cheap,
   confounded) or regenerate D1v2 and retrain A0 (+33 GPU-h, clean)?
2. **Steps per arm.** 1M (recommended, 5.5 days total) vs the published 3M
   (~17 days). The parked `fixed_seed*` runs reach 1.35M, so 1M is the
   largest budget at which A0 is free.
3. **Scoring under leakage.** Report mirror arms on the 4 leakage-clean
   published test trajectories (keeps comparability with the paper) rather
   than re-splitting D1 mirror-aware (breaks it)? Design assumes the former.
4. **Phase 0b on the shared box.** A 40-core EQdyna scan for ~40 min counts as
   the one heavy job; OK to run it when load permits, or should it go to a
   cluster (then `eqdyna` must be built there — outside this design's scope)?
5. **Hardware for the GPU arms.** Local A100 (GPU 0/1/3, sequential) as
   costed, or queue behind the M3 sweep on GH200 (needs the dataset shipped
   and the throughput re-measured)?
6. **Labelling.** Is the result an "M1 (expanded)" row, or is the owner
   considering making D1A the new M1 training definition for the retrain
   campaign? This changes nothing in the matrix but changes how the table is
   reported.

## 7. What a reviewer would attack

- *"Four held-out arresting cases is n = 4."* True; outcome metrics are
  direction-only at that n. Mitigation: hold out more if Phase 0b yields
  >= 10 per side; report per-trajectory tables, not just means.
- *"Arrest in M1 is an artefact of clipping the nucleation patch at the mesh
  boundary, not a frictional/stress arrest."* Correct and stated (1.2). This
  design tests whether the GNS learns *this* arrest mechanism; stress- or
  barrier-controlled arrest is an M2/M3 question.
- *"The mirror truth noise for arresting cases (1.3e-2 rms, 2.8 km arrest
  offset) is of the same order as the effect you hope to measure."* It bounds
  the achievable arrest-position agreement to ~0.08 in ruptured fraction;
  H2's target (error reduction from > 0.3) is well above this. Reported as the
  noise floor next to every T_arr number.
- *"Two seeds per arm."* Owner's compute rule (one job at a time) makes 3
  seeds per arm +4 days; the 3-seed A0 floor is used as the yardstick, and
  any arm whose two seeds disagree in sign is reported as inconclusive.
- *"The paper text was not re-read."* The paper PDF is pay-walled from this
  host; all M1/split/metric definitions come from the repo
  (`gate.py`, `prepare.eqdyna.4gns.py`, scenario `user_defined_params.py`,
  `docs/user/`). If the paper defines "arrest" differently, the threshold in
  1.2 should be aligned before results are written up.
- *"Matched steps favours the smaller dataset."* Acknowledged; the optional
  epoch-matched read is costed in 3.4.

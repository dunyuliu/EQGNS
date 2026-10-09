# M3 batch-size / LR-scaling sweep on NVIDIA GH200 (TACC) — experiment design

Board row: `m3-batchsize-lr-sweep-gh200` (PATHWAY_FORWARD.md, owner-approved
2026-10-09, commit `38edc30`). Status: DESIGN ONLY — nothing in this document
has been run. No training, rollout or GH200 job was launched to produce it.
Every number below is either (a) read from files on disk in this repo
(provenance given inline), or (b) an explicit estimate, labelled as such.

Scope exclusions: the M1 arresting/mirror expansion and the M2/M3 mirror
augmentation are separate rows and are not designed here.

---

## 0. Verdict in one paragraph

The sweep is feasible and needs **no new EQdyna runs**, but it cannot be done
by comparing the checkpoints that exist today: every existing M3 arm differs
from the published b8/3e-5 run in LR *and* LR schedule (not only batch), the
"b12" arm is at 300k steps and its epoch log says it ran at batch 10, and all
of them were trained on the local A100 box while the owner requires one
hardware family. The clean design is a **from-scratch factorial on GH200 at
matched sample budget (21.6 M samples = the published 2.7 M x 8)**, with a
**two-seed b8 baseline** so that an arm's effect can be judged against the
run-to-run floor — without that floor, n=1 per arm proves nothing.
Recommended scope: a 2-node-hour throughput pilot first, then the **reduced
4-arm design** (b8 x 2 seeds, b4, b12 at lr 3e-5), estimated 280–540 GH200
node-hours; the LR-scaling arms (b4@1.5e-5, b12@4.5e-5, +140–270 node-h)
only if the batch axis moves the primary metric beyond the seed floor. A
zero-GH200-cost Phase 0 on the existing A100 checkpoints sizes the effect
first and may shrink the matrix further.

---

## 1. Scientific question

**Q1.** At matched training budget, does batch size (4 / 8 / 12) at the
published LR (3e-5) materially change M3's rupture-reproduction quality on the
held-out fractal-stress (M3_D3, 15 trajectories) and D1-hypocenter cross-test
(M3_D1hypo, 6 trajectories) sets?

**Q2.** Does scaling the LR with batch (linear rule, lr = 3e-5 x b/8) recover
or beat the b8/3e-5 published point at matched budget?

**Hypotheses (stated before any run):**
- H1 (null, expected): at matched samples, b4 and b12 at lr 3e-5 differ from
  b8 by less than the two-seed b8 floor on the primary metrics → batch size
  is not a lever; the published choice stands.
- H2: a larger batch at linearly scaled LR (b12 @ 4.5e-5) matches b8 quality
  at the same sample budget in fewer optimizer steps → wall-clock lever only
  if per-sample cost on GH200 falls with batch (it does **not** on the A100,
  see section 5).

**"Material" is defined a priori** (section 4.4): an arm's paired
per-trajectory difference from the GH200 b8 baseline on the primary metrics
exceeds BOTH (i) the two-seed baseline spread and (ii) the late-training
checkpoint-jitter floor measured from the published run's own 2.4M / 2.7M /
3.0M / 3.1M rollouts (these rollout files already exist on disk).

**Kill criterion for the full matrix:** if Phase 0 (section 3.1) shows every
existing A100 arm inside the checkpoint-jitter floor on both primary metrics,
the GH200 matrix shrinks to the 3-arm minimum (b8 x 2 seeds + b12) and the
null is the deliverable.

---

## 2. What exists on disk (verified 2026-10-09, this session)

All under `data/gns-sample/case4.200m.multi.stress.160scenarios.homo.a.Vw/`
(hereafter `D2_160`), 46 GB total. Batch size is **not** stored in
`config.json` (it is a CLI flag); it is derived here from
`epoch_loss_log.txt` column 1 (steps at end of epoch 0) against the training
set size of 115,640 samples (= 14,455 x 8, 28,910 x 4, 57,820 x 2, all
consistent). Wall-time per step is `mtime(model-last.pt) - mtime(model-0.pt)`
over the step span — on a shared, contended A100-SXM4-40GB box with up to
three runs started within minutes of each other on 2025-08-28, so an
**upper bound**, not a benchmark.

| Directory (`D2_160/…`) | batch (epoch log) | lr_init | decay (rate, steps) | noise_std | max step | samples seen | A100 wall ms/step | note |
|---|---|---|---|---|---|---|---|---|
| `models.nmp10.lr3e-5.b8.cotopaxi.r1` | 8 | 3e-5 | 0.1, 20M | 2e-2 | 6.7M | 53.6M | 194 | **published M3 = `model-2700000.pt` of this run** (21.6M samples); the run was continued to 6.7M |
| `models.nmp10.b4.cotopaxi.r1` | 4 | **1e-4** | 0.1, **5M** | 2e-2 | 10M | 40M | 90 | LR and schedule differ from published |
| `models.nmp10.cotopaxi.r1` | **2** | 1e-4 | 0.1, 5M | 2e-2 | 10M | 20M | 52 | an unlisted b2 arm; same LR/schedule as the b4 arm |
| `models.nmp10.lr3e-5.b12.cotopaxi.r1` | **10 (not 12)** | 3e-5 | **1.0 (constant)**, 20M | 2e-2 | 0.3M | 3M | 211 | dir name says b12; 11,564 steps/epoch = 115,640 / 10. Constant LR. Far short. |
| `models.nmp10.lr3e-5.b8.n5e-3.cotopaxi.r1` | 8 | 3e-5 | 1.0 (constant) | **5e-3** | 3M | 24M | 170 | noise arm, not part of this sweep |
| `models.r1_lr*_bs8_ns*_nmp{5,8,10}_cotopaxi` (5 dirs) | 8 | 3e-5 / 4e-5 | 0.1, 5M | various | 0.7–1M | — | — | nmp/noise arms, not part of this sweep |

Rollouts: `rollouts.nmp10.lr3e-5.b8.cotopaxi.r1.published/` holds the
paper's M3 rollouts at 0.8M, 1.0M, 1.3M, 2.4M, 2.7M, 3.0M, 3.1M (the M3_D3
anchor); `rollouts.nmp10.lr3e-5.b8.cotopaxi.r1/model-2700000.pt` is the
M3_D1hypo anchor (both per `tests/paper_parity/gate.py` CASES). The
`rollouts.models.nmp10.lr3e-5.b12.cotopaxi.r1/` directory contains only
`rollout.log.txt` files for checkpoints (0.8M–3M) that never existed in that
model directory — **no usable rollouts exist for the b12 arm**.

Per-sample cost on the A100 is flat across batch size (26 / 22.5 / 24 / 21
ms per sample at b2 / b4 / b8 / b10). Consequence: **matched samples =
matched compute**, and a larger batch buys no throughput on this stack.

### 2.1 Plain statement on comparability today

- b4 @ 10M steps has seen 40M samples (1.85x the published budget) at a
  3.3x higher initial LR with a 4x faster decay. It is **not** at a matched
  step count or a matched sample count with b8 @ 2.7M, and it is not at the
  published LR.
- "b12" @ 300k steps has seen 3M samples (14 % of the budget), at constant
  LR, apparently at batch 10. **Not comparable** to anything at 2.7M.
- The only two existing points that are matched in *samples* with the
  published checkpoint are b4 @ 5.4M (on disk, `model-5400000.pt`) and
  b2 @ 10M (93 % of budget) — but both at lr 1e-4 / 5M decay, so they test
  batch and LR jointly. They are usable for effect-size scoping (Phase 0),
  not as arms of the answer.
- Resuming the A100 b12 run on GH200 to reach a matched budget is rejected:
  it would mix hardware inside one arm and keep the constant-LR schedule.

Matched-budget points proposed for the fair comparison (section 3.2):
**primary = matched samples, 21.6M** (b4 @ 5.4M steps, b8 @ 2.7M, b12 @
1.8M); **secondary = matched optimizer steps, 2.7M** (b4 @ 2.7M is a free
intermediate checkpoint of the b4 arm; b12 @ 2.7M costs +50 % on that arm
and is optional).

---

## 3. Independent variables and run matrix

Fixed across every arm (= the published M3 configuration, `D2_160/config.json`
and `meshnet/train.py.published` semantics): 10 message-passing steps, latent
128, noise_std 2e-2, lr decay rate 0.1 over 20M steps (lr(step) = lr_init x
0.1^(step/2e7) + 1e-6), Adam, training set `D2_160/dataset/train.npz`,
validation `valid.npz`, current `meshnet/train.py` (gated against the frozen
oracle by `tests/test_training_gate.py` at rtol 1e-6 over 1000 steps — on
A100; see 3.4 for the GH200 re-check), `--seed` set explicitly on every arm
(`meshnet/seeding.py`: model init, sample order and noise draw from
independent sub-streams; a replicate = a different seed).

### 3.1 Phase 0 — zero-GH200-cost effect-size scoping (local A100, rollouts only, no training)

Purpose: size the effect before paying for the matrix; also produce the
checkpoint-jitter floor. All inputs are A100-trained and are rolled out on
the A100 in deterministic mode, so Phase 0 is internally hardware-consistent;
its LR confound is accepted and stated.

| Checkpoint (all exist) | role |
|---|---|
| b8 published `model-2700000.pt` | anchor (reference rollout already in `tests/paper_parity/reference.json`) |
| b8 `model-2400000/3000000/3100000.pt` (rollouts exist in `.published/`) | **checkpoint-jitter floor**: late-training ckpt-to-ckpt spread of the same run |
| b4 `model-5400000.pt` | matched samples (lr 1e-4 confound) |
| b4 `model-2700000.pt` | matched steps (lr 1e-4 confound) |
| b2 `model-10000000.pt` | 93 % matched samples (lr 1e-4 confound) |
| b8 `model-6700000.pt` | free secondary: does 2.5x more training of the published run help |
| "b12"(b10) `model-300000.pt` vs b8 `model-300000.pt` | matched steps at 0.3M only; early-training sanity, direction only |

Cost (estimate): 7 checkpoints x 21 trajectories x 826 steps, published eager
path ~24 ms/step/traj on an idle A100 (`docs/dev/SESSION_LOG_20261008…`, line
327) → ~7 min per checkpoint, ~1 GPU-hour total; deterministic mode somewhat
slower. Correctness-only work, so the owner's idle-box timing rule does not
bind, but the one-heavy-job rule does (run serially on one free GPU).

### 3.2 Phase 1 — GH200 matrix (from scratch, one hardware family)

Nothing from disk is reused as an arm. Every arm below is trained on GH200
from step 0 and rolled out on GH200. Sample budget per arm = 21.6M unless
marked.

| Arm | batch | lr_init | steps | samples | seeds | priority | purpose |
|---|---|---|---|---|---|---|---|
| A0a | 8 | 3e-5 | 2.7M | 21.6M | s1 | **must** (owner) | GH200 baseline, same config as published |
| A0b | 8 | 3e-5 | 2.7M | 21.6M | s2 | **must** | seed replicate = noise floor for the whole sweep |
| A1 | 4 | 3e-5 | 5.4M | 21.6M | s1 | reduced design | batch axis, fixed LR; `model-2700000.pt` of this arm = matched-steps point for free |
| A2 | 12 | 3e-5 | 1.8M | 21.6M | s1 | reduced design | batch axis, fixed LR |
| B1 | 4 | 1.5e-5 | 5.4M | 21.6M | conditional | linear LR scaling, small batch |
| B2 | 12 | 4.5e-5 | 1.8M | 21.6M | conditional | linear LR scaling, large batch |
| A2x | 12 | 3e-5 | 2.7M | 32.4M | optional | matched steps for b12 (+0.9M steps on A2, +50 % cost) |

Design levels: 3 arms (A0a, A0b, A2) minimum; 4 arms (A0a, A0b, A1, A2)
recommended; 6 arms (+B1, B2) full. B-arms are launched only if A1 or A2
moves a primary metric beyond the floor defined in 4.4 (decision made on
Phase-1 A-arm results, not on test-set-selected checkpoints).

Checkpoint evaluated per arm: **the final matched-sample checkpoint, chosen a
priori**, plus the matched-step intermediate for A1. No best-of-run selection
on test metrics (that is leakage); if a selection is ever wanted it uses
`valid.npz` loss only and is labelled as such.

### 3.3 What is reused vs resumed vs from scratch

- Reused as-is: training/validation/test data (`D2_160/dataset/{train,valid,test}.npz`,
  `D2_160.case3.test/test.npz`), the published checkpoint and its rollout
  files (anchors), `tests/paper_parity/{gate,measure_vs_published,det_rollout}.py`.
- Resumed/continued: **none** of the existing checkpoints becomes an arm.
  (Resume is used only *within* a GH200 arm across 2-hour job segments, 3.4.)
- From scratch on GH200: every arm in 3.2.

### 3.4 Execution constraints the design respects (for the later dispatcher; not used now)

- Access: existing SSH control socket only; allocation EAR26006 only; all
  writes under that system's scratch area; site/system name stays off
  repo-tracked files ("NVIDIA GH200 (TACC)").
- Staging: ~35 GB (train 27.8 GB, valid 2.0 GB, two test sets ~3 GB each)
  rsynced over the control socket once; scratch purge policy applies — touch
  or re-stage before each chain.
- Partition cap: `gh-dev` 2 h (board, `gh200-cross-hw-timing`). A 21.6M-sample
  arm needs 35–70 chained segments (section 5), each `--model_file latest
  --train_state_file latest` (optimizer state and step restored; LR is a pure
  function of step so the schedule is continuous across resume; seeded RNG
  state is checkpointed per `meshnet/seeding.py`). The sample-shuffle epoch
  restarts on each resume — a minor, stated deviation from an uninterrupted
  run, identical across arms.
- Hazard: `train.py` has no wall-clock stop; a SLURM kill during
  `torch.save` leaves a truncated newest checkpoint that `latest` would pick.
  Mitigation for the driver: wrap in `timeout 110m`, and on resume validate
  that the newest model/train_state pair loads, else fall back one save.
  `--nsave_steps 10000` (≤ 10–20 min of work at risk per kill; ~19 MB per save →
  ~5–10 GB per arm).
- QOS: 20 running jobs (board). Six chains fit; if the limit is per-account
  and shared, chains simply queue — correctness unaffected, wall-clock
  uncertain. If a non-dev GH200 partition with a longer limit exists on this
  allocation, use it: chaining overhead drops to 2–3 segments per arm.
- Software: the torch 2.6.0+cu126 aarch64 stack (fast path unavailable there —
  irrelevant to training; evaluation uses the default eager path in
  deterministic mode). Before any arm starts, run `pytest -m training_gate`
  on GH200 (1000-step train() vs oracle, both on GH200). If it fails on GH200
  while passing on A100, that is itself a finding to report, and arms must
  use `meshnet/train.py.published` semantics — stop and report rather than
  proceed.
- If GH200 training of any arm proves infeasible in the review period
  (queue, QOS, stack), the arm is dropped and reported as not run; **no A100
  checkpoint is substituted** for it.

---

## 4. Confound control and evaluation

### 4.1 Hardware family

Every compared arm is trained and rolled out on GH200. The published
anchor is A100-trained and its rollout files were made on A100/x86; the board
already shows 16/21 trajectories diverge between a GH200 and an A100 rollout of
the *same* checkpoint with no tolerance defined. Two anchors are therefore
used, and both are reported:

- **Anchor G (primary):** the published `model-2700000.pt` rolled out on GH200
  in deterministic mode (`det_rollout.py`, same as `gate.py reference` does).
  Arm-vs-G differences contain no rollout-hardware offset.
- **Anchor P (secondary):** the shipped A100 rollout files
  (`.published/model-2700000.pt/*.pkl`). Arm-vs-P minus arm-vs-G isolates the
  rollout-path hardware offset for M3 — a free, quantitative input to the
  open board item `release-gate-decisions-pending` (b).

Training-hardware offset: A0a vs anchor G = (training-hardware + seed)
offset; A0a vs A0b = seed offset; the difference bounds the training-hardware
effect without any extra run.

### 4.2 Other confounds held fixed

Same data, same config, same code revision (git SHA recorded per job), same
noise, same decay schedule, same evaluation checkpoint rule, same seed s1 on
all single-seed arms (paired comparison: differs from A0a only in the knob).

### 4.3 Metrics and the commands that produce them

Anchor convention follows the `test-suite-overhaul` row: the PUBLISHED EQGNS
rollout is the anchor, not EQdyna ground truth. Existing machinery:

| Metric | Source | How it is run |
|---|---|---|
| ΔRT RMSE (s) at 0.1 m/s threshold | `tests/paper_parity/measure_vs_published.py::compare_pair` → `delta_rt_rmse_s` | arm rollout (side A) vs anchor G and vs anchor P (side B) |
| Mw error | same → `delta_mw` (Mw = 2/3 (log10 M0 − 9.1), `plot.rupture.dynamics.py` convention) | same |
| slip-rate RMSE vx, vy normalized by anchor peak | same → `vx_rmse_norm`, `vy_rmse_norm` | same |
| missed / false ruptured nodes | same → `missed`, `false` (and `gate.py::metrics`) | same |
| mse_vx, rt_rmse vs EQdyna test truth | `tests/paper_parity/gate.py::metrics` | arm rollout vs `test.npz` ground truth (already on disk; no new EQdyna) |
| final-slip RMSE | **not implemented** in either script | add to the sweep driver (∫‖v‖dt over the rollout vs anchor), ~15 lines; or drop and rely on ΔMw, which integrates the same quantity |

Rollouts of arm checkpoints: `measure_vs_published.fresh_raw_rollout(case,
cuda, model_dir=<arm dir>)` already accepts a foreign model directory (used by
`gate.py regression --falsify`). The sweep needs a thin driver under
`runs/<YYYYMMDD>_m3-bs-lr-sweep/` (git-ignored per PROJECT_RULES rule 4)
that loops arms x cases x {G, P, truth} through these two functions and
writes one CSV in `measure_vs_published.py`'s `RAW_COLUMNS` layout. No change
to `meshnet/` or `tests/`.

Cases: `M3_D3` (n=15) and `M3_D1hypo` (n=6). Per the board's regression-gate
decision, `M3_D3` trajectory 7 is excluded from any pass/fail verdict but
still reported.

A design limitation to state plainly: a distance-to-published metric cannot
tell "better than published" from "worse than published". Ranking arms
therefore uses the vs-truth row (`gate.py::metrics` on the existing test
sets) as the sign-aware primary, with the vs-anchor rows as the
published-consistency read. This is still inside the row's "no EQdyna ground
truth as anchor" convention: the truth is used to order arms, not as the gate
anchor.

### 4.4 Primary metrics, floors, and statistics (declared a priori)

- Primary: on `M3_D3` (n=15, traj 7 reported but excluded), (i) mean
  `rt_rmse` vs truth, (ii) mean `vx_rmse_norm` vs anchor G.
- Secondary: `delta_mw`, `missed+false`, `vy_rmse_norm`, final-slip RMSE;
  `M3_D1hypo` (n=6: direction only, no CI).
- Floors: F_seed = |A0a − A0b| per metric; F_ckpt = max pairwise difference
  among the published run's 2.4M/2.7M/3.0M/3.1M rollouts (Phase 0).
- Effect: paired per-trajectory differences arm − A0a, mean with a bootstrap
  95 % CI (n=15). "Material" = CI excludes 0 AND |mean| > max(F_seed, F_ckpt).
- Every reported population count is re-derived from the driver's CSV.

---

## 5. Budget estimate and go/no-go

**No GH200 training throughput has ever been measured for this code.** The
only measured numbers are the A100 wall-times in section 2 (contended, upper
bound: ~22–26 ms/sample → ~40–45 samples/s at any batch) and the rollout
ratio A100→GH200 of ~1.5–1.7x on the published eager path (board row
`gh200-cross-hw-timing`, unaudited). Per-sample cost being flat across batch
on the A100 means the loader or the per-graph work dominates, so GH200 may
help less than its GPU ratio suggests.

Per-arm estimate (21.6M samples): at 45 samples/s → **133 node-h**; at an
optimistic 2x → **67 node-h**. Rollouts for evaluation: 21 trajectories x 826
steps x ~6–16 ms/step/traj x 2 anchors → minutes per arm, negligible.
Staging and chaining overhead (load 28 GB `train.npz` per segment, ~1–3 min
x 35–70 segments) adds ~1–3 node-h per arm. Charge rate assumed 1 SU per
node-hour — **verify against the allocation's actual rate before dispatch.**

| Scope | arms | node-h (est.) | segments at 2 h | wall-clock if 4–6 chains run concurrently |
|---|---|---|---|---|
| Pilot (mandatory first) | 3 x 30-min runs at b4/b8/b12 + `training_gate` | ≤ 2 | 1 | hours |
| Minimum | A0a, A0b, A2 | 200–400 | 105–210 | 3–6 days + queue |
| **Reduced (recommended)** | A0a, A0b, A1, A2 | **280–540** | 140–280 | 3–6 days + queue |
| Full | + B1, B2 | 420–810 | 210–420 | 3–6 days + queue (6 chains) |
| + A2x | + 0.9M b12 steps | +35–70 | +18–35 | — |

**Recommendation: GO on the pilot now; GO on the reduced 4-arm design once
the pilot gives a measured samples/s** (re-scale the table; if GH200 yields
< 60 samples/s, the full design exceeds ~700 node-h and should wait for an
explicit owner OK). Treat B1/B2 as conditional on the A-arm result (section
3.2). Fallback if even the reduced design is too expensive: run all four
A-arms to 8M samples (37 % budget, ~100–200 node-h total) to rank, then
extend only A0a and the best non-baseline arm to 21.6M — with the stated risk
that early-training rankings can reorder under the decaying schedule.

Phase 0 costs zero GH200 time and should run regardless.

---

## 6. EQdyna scenario generation

**None required.** Training (`train.npz`), validation (`valid.npz`) and both
test sets (`D2_160/dataset/test.npz` = D3 fractal test, `D2_160.case3.test/
test.npz` = D1-hypocenter cross-test) already exist on disk and are the exact
inputs of the published M3 run and of the paper-parity gate. The sweep varies
only optimizer hyperparameters. No step touches the EQdyna source tree, and
nothing would be written under the scenario-generation directory. If a
reviewer later asks for a third, unseen test regime, that would be new
scope, new EQdyna runs under the scenario-generation directory only, and a
separate owner call — not part of this row.

---

## 7. Open questions needing the owner's call before dispatch

1. Scope level: reduced 4-arm (recommended) vs minimum 3-arm vs full 6-arm —
   after the ≤ 2 node-h throughput pilot reports measured samples/s.
2. Whether a non-dev GH200 partition (longer than 2 h) is usable under
   EAR26006; it changes chaining overhead and wall-clock, not the science.
3. Confirm the "b12" directory's true batch size from its launch command /
   SLURM log if one exists; the epoch log implies batch 10. The GH200 arm is
   b12 regardless (owner's named level), but the on-disk label should be
   corrected or annotated.
4. Final-slip RMSE: add to the driver (small) or rely on ΔMw.
5. Whether to run the fallback (reduced budget first, extend winners) if the
   pilot shows < 60 samples/s on GH200.

## 8. What a reviewer would attack

- n=1 per non-baseline arm: mitigated by the two-seed baseline floor and the
  checkpoint-jitter floor, but a second seed on any arm that shows an effect
  is the proper confirmation (+70–135 node-h each).
- Test set size (15 + 6 trajectories): effects smaller than the per-trajectory
  spread are undetectable; this design will report "below the floor", not
  "no effect".
- Epoch-shuffle restart at each 2-h resume differs from an uninterrupted
  run; identical across arms, so it does not bias the comparison, but the
  GH200 b8 baseline is not a bit-level replica of the published training.
- The training-gate (train() vs oracle) has only been shown on A100; the
  pilot re-runs it on GH200 before any arm.

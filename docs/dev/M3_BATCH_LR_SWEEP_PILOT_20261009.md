# M3 batch-size / LR sweep — GH200 pilot, 2026-10-09

Board row `m3-batchsize-lr-sweep-gh200`; design `M3_BATCH_LR_SWEEP_DESIGN.md`
(section 5 "Pilot (mandatory first)", section 3.4 gate precondition, section 7
open questions 2–4). Hardware: NVIDIA GH200 (TACC), allocation EAR26006.
Everything below was freshly run or read this session; nothing is inherited.

## Verdict

**Pilot stopped at the gate, by design. No arm was trained.**
`pytest -m training_gate` FAILS on GH200 against the committed A100-made
reference (`tests/golden/training_gate_reference.json`): 999/1001 per-step
train losses differ (rtol/atol 1e-6), max |Δ| 0.70, max rel 5.06. The traces
are identical at step 0 (1.029649019241333 on both), differ at step 1 by
2e-6 relative (1.3600726 vs 1.3600699), and drift from there (step 500:
0.0361 vs 0.0373; step 1000: 0.2750 vs 0.2719). That signature is
cross-hardware floating-point divergence amplified over 1000 optimizer
steps, not a code divergence — but the committed gate cannot tell the two
apart, because it compares current `train.py` on GH200 with
`train.py.published` on A100 (hardware and code change together). Per the
brief this is the finding; no workaround was attempted. The same-hardware
check the design doc actually asks for (regenerate the reference with
`train.py.published` on GH200, ~2 min, then compare) is the obvious next
step and is the conductor's call, not taken here.

Samples/s per arm: **not measured** (arms never started). Scope decision
(reduced 4-arm vs minimum) therefore remains open — a judgment call for the
conductor, not settled by this pilot.

## Jobs and cost

| Job | Partition | Account | Content | State | Elapsed | Node-h |
|---|---|---|---|---|---|---|
| 1061515 | gh-dev | EAR26006 | gate + b8 | CANCELLED by me while PD (never ran; resubmitted on `gh` per coordinator) | 0 | 0 |
| 1061516 | gh-dev | EAR26006 | b4 + b12 (afterok 1061515) | CANCELLED by me while PD | 0 | 0 |
| 1061645 | gh | EAR26006 | gate + b8 | FAILED at the gate (exit 1 by the sbatch's own stop rule) | 00:02:19 | 0.04 |
| 1061646 | gh | EAR26006 | b4 + b12 (afterok 1061645) | CANCELLED by dependency / my watcher, never started | 0 | 0 |

Consumed: **0.04 of the 2.0 node-hour cap.** EAR26005 never used. The other
project's processes on that system (`sync_loop.sh`, `localroll.sh`, the ssh
control master, dev job 1060815) were not touched; the existing control
socket was reused for every command, no new login/control-master opened.
Node 1061645: GH200 120GB (97,871 MiB), torch 2.6.0+cu126, Python 3.11.8,
driver code at `1da434c`. Note for the next submission: on `gh`, `-n 1`
without `-c` allocated **1 CPU** — add `-c <n>` (the data loader runs in the
main process; 8 is the gate's own thread cap).

## Open question 2 — non-dev partition

**Settled: `gh` is usable under EAR26006.** Evidence: `sbatch -p gh -A
EAR26006` accepted and ran job 1061645 (QOS `qdefault`); `sacctmgr show
qos`: `qgh`/`qnormal` MaxWall 2-00:00:00, 20 running / 40 submitted per
user; `qdevelopment` (gh-dev) MaxWall 02:00:00, 1 running / 3 submitted per
user. Consequence for the design: a 21.6M-sample arm chains in 2–3 48-h
segments instead of 35–70 2-h segments. Caveat: `gh` is shared with the
same account's other project (5 running + 7 pending jobs at submission
time), so wall-clock to start is not under this row's control; 1061645
waited ~1 h 48 min in PD.

## Open question 3 — the on-disk "b12" directory

**Batch 10, mislabeled; directory left untouched.** No launch command or
SLURM log exists for it anywhere on the local box (data directory, `scripts/`,
`runs/`, shell histories, git history all searched; the only mention is as a
rollout target in `scripts/utils/batch.rollout.py`). The launcher of that
era, `scripts/train.sh`, takes the directory suffix (`$2`) and `--batch_size`
(`$4`) as independent manual arguments, so label and batch can disagree.
Arithmetic from `epoch_loss_log.txt` (first epoch ends at step 11,564):
11,564 × 10 = 115,640 = the D2_160 training-sample count (the same count the
b8/b4/b2 logs give: 14,455 × 8, 28,910 × 4, 57,820 × 2); batch 12 would end
epoch 0 at ceil(115,640/12) = 9,637. The GH200 arm stays b12 (owner's named
level).

## Open question 4 — final-slip RMSE

**Added to the driver and working.** `scripts/m3_bs_lr_sweep/sweep_eval.py`
reuses `gate.metrics()` (vs truth; its `final_slip_rmse` already exists since
PR #71) and `measure_vs_published.compare_pair()` (vs anchor) and adds
`final_slip_pair()` for the vs-anchor side (per-node ∫|v|dt over
`compare_pair`'s 755-step window, RMSE across nodes, plus the same normalised
by the anchor's peak final slip). Demonstrated locally (CPU, zero GH200 time)
on the shipped M3_D3 rollouts, checkpoint 2.4M scored against the published
2.7M as anchor P (n = 14 excluding trajectory 7, which is reported but
flagged): final-slip RMSE mean **2.44 m** (normalised 0.251); vs truth
2.99 m; for scale the same pair gives ΔRT RMSE 0.099 s, ΔMw 0.034,
vx RMSE/peak 0.030. These numbers are a first entry of the design's
checkpoint-jitter floor F_ckpt (2.4M vs 2.7M), A100 rollouts on both sides.

## What landed

- `scripts/m3_bs_lr_sweep/pilot_gh200.sbatch` — gate-first pilot job; site
  specifics on the command line only.
- `scripts/m3_bs_lr_sweep/throughput.py` — samples/s from timestamped
  `train.py` reports (steady-state slope, warm-up excluded, GPU util).
- `scripts/m3_bs_lr_sweep/sweep_eval.py` — metrics driver (score / rollout
  modes), final-slip RMSE addition. `tests/` untouched.
- `scripts/m3_bs_lr_sweep/config.m3_published.json` — the published M3 config.
- Staged on the GH200 scratch (`$SCRATCH/eqgns-m3-sweep-pilot/`): D2_160
  `train.npz`/`valid.npz` (29.8 GB, sizes verified) and M1 D1
  `train.npz`/`valid.npz`; the pilot code tree; the failing gate's
  `loss_log.txt` and pytest output under `logs/`/`results/`. Scratch purge
  policy applies.

## What a reviewer would attack

- The gate verdict conflates hardware and code; until `train.py.published`
  is run on GH200 the "cross-hardware, not code" reading is an inference
  from the step-0 match and the 2e-6 step-1 difference, not a measurement.
- If the same-hardware gate passes, the sweep's section 4.1 logic (two
  anchors, G and P) already absorbs the training-hardware offset; if it also
  fails, `meshnet/train.py` has a GH200-specific divergence and the arms must
  use `train.py.published` semantics (design doc 3.4).
- Throughput remains unmeasured; the 280–540 node-hour estimate still rests
  on contended A100 numbers.

---

# Continuation, 2026-10-10 — same-hardware gate and throughput pilot

Second dispatch on the same branch; everything below was freshly run this
session (job 1061807). Inherited content above is kept verbatim.

## Verdict

**Same-hardware gate PASSES bit-for-bit; throughput measured: 77–86
samples/s on GH200 at b4/b8/b12, 1.7–1.9x the 45 samples/s A100 estimate
and above the design's 60 samples/s threshold → reduced 4-arm design is GO
at ~290 node-h (training only).** Cumulative pilot spend **1.51 of 2.0
node-hours** (0.04 inherited + 1.47 this job).

## 1. Same-hardware training gate (design doc 3.4) — PASS

`scripts/m3_bs_lr_sweep/same_hw_gate.py` (new; `PILOT_GATE=2` in the
sbatch) ran the frozen oracle `meshnet/train.py.published` (sha256
`c222aad7…` = the committed reference's oracle) on the GH200 for the gate's
exact configuration (M1 D1, batch 2, 1000 steps, seed 20260101, config
`tests/fixtures/training_golden/config.json`, deterministic algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`), wrote the GH200-native trace to
`tests/golden/training_gate_reference_gh200.json` (new file; the committed
A100 oracle `training_gate_reference.json` and `tests/paper_parity/` were
not touched), then ran current `train.py` on the same GPU and compared at
the gate's own rtol = atol = 1e-6. Oracle 115 s, current 107 s, 223 s total.

| Comparison | train loss | valid loss |
|---|---|---|
| current GH200 vs oracle GH200 (**the gate**) | **0/1001 differ, max abs 0, max rel 0** | **0/1001 differ, max abs 0** |
| current GH200 vs committed A100 oracle | 999/1001 differ, max abs 0.700, max rel 5.06 | 998/1001, max abs 0.485 |
| oracle GH200 vs committed A100 oracle | 999/1001 differ, max abs 0.700, max rel 5.06 | 998/1001, max abs 0.485 |

Per-step values (train loss; current GH200 = oracle GH200 to all printed digits):

| step | GH200 (current = oracle) | committed A100 oracle | Δ abs | Δ rel |
|---|---|---|---|---|
| 0 | 1.029649019 | 1.029649019 | 0 | 0 |
| 1 | 1.360072613 | 1.360069871 | 2.74e-6 | 2.02e-6 |
| 500 | 0.03609648347 | 0.03734073788 | 1.24e-3 | 3.33e-2 |
| 1000 | 0.2750184238 | 0.2718824148 | 3.14e-3 | 1.15e-2 |

Reading, now a measurement rather than an inference: the current code and
the frozen oracle are **bit-identical on GH200** (not just within 1e-6), so
the committed-gate failure reported above is cross-hardware floating-point
divergence only; `meshnet/train.py` carries no GH200-specific code
divergence. The oracle-vs-oracle row shows the identical drift, which pins
the whole discrepancy on hardware. Two further points the same job gives
for free: (i) the GH200 trace reproduced the previous job's current-side
trace (1061645, a different node, c637-092) to all printed digits at steps
0/1/500/1000 — the GH200 path is run-to-run and node-to-node deterministic
under the gate's settings; (ii) the arms were therefore allowed to start
(sbatch stop rule not triggered). Consequence for the design: 4.1's two
anchors (G = GH200-trained, P = published A100) remain necessary — the A100
and GH200 traces part at step 1 and the training-hardware offset is real.

## 2. Throughput pilot — measured

Job 1061807, `gh` partition, EAR26006, node c634-061, NVIDIA GH200 120GB
(97,871 MiB, 1980 MHz SM max), torch 2.6.0+cu126, Python 3.11.8, code at
`1da434c` + the gate driver (`9647607`), seed 1, M3 published config
(`config.m3_published.json`, lr_init 3e-5, loss report every 250 steps),
D2_160 `train.npz`/`valid.npz` from the pilot scratch, 28 min wall per arm
(`timeout`), from scratch, no checkpoint. Node exclusive (sacct AllocCPUS
72); `OMP_NUM_THREADS=1` in effect (see 4). Samples/s = steady-state slope
of step vs wall-clock over the whole post-load window
(`scripts/m3_bs_lr_sweep/throughput.py`), load+warm-up excluded. GPU SM
utilisation from `nvidia-smi dmon` (5 s samples, first 10 % dropped),
computed by hand because throughput.py's `--dmon` parser returned null
(see 4). Submitted 2026-10-09 19:38 local, started 01:23 (5 h 44 min in PD),
ran 01:28:05.

| Arm | batch | steps/s | ms/step | ms/sample | **samples/s** | steady window | fit RMS (steps) | SM util mean / median / min | GPU mem max |
|---|---|---|---|---|---|---|---|---|---|
| b4_s1 | 4 | 19.20 | 52.1 | 13.02 | **76.8** | 31,250 steps / 1631 s | 20.4 | 94.2 / 94 / 90 % | 17.2 GB |
| b8_s1 | 8 | 10.59 | 94.5 | 11.81 | **84.7** | 17,000 steps / 1607 s | 5.2 | 96.8 / 97 / 89 % | 38.0 GB |
| b12_s1 | 12 | 7.14 | 140.0 | 11.67 | **85.7** | 11,500 steps / 1611 s | 4.1 | 97.4 / 97 / 0 % (one 5 s sample) | 60.6 GB |

Load+warm-up: 45 s (b8, first arm, cold file cache), 28 s (b4, b12).
Per-sample cost is nearly flat across batch (11.7–13.0 ms/sample), the same
pattern the design noted on the A100; with the GPU at 94–97 % SM
utilisation, the flat cost is GPU-side per-graph work, not the loader —
larger batch buys ~10 % between b4 and b8 and ~1 % between b8 and b12. GPU
memory scales linearly with batch (~5 GB per unit); b12 at 61 GB leaves no
room for b24 on this device without changes.

## 3. Comparison to the design and recommended scope

- Design estimate: ~45 samples/s on a contended A100 (upper bound) and an
  "optimistic 2x" case. Measured GH200: **1.71x (b4), 1.88x (b8), 1.90x
  (b12)** of 45 samples/s — at the optimistic end, so the design's lower
  node-hour figures apply.
- Threshold: design section 5 says GO on the reduced 4-arm design once
  samples/s is measured, and asks for an explicit owner OK only if GH200 is
  < 60 samples/s (full design > ~700 node-h). All three arms are above 60.
- Per-arm training cost at 21.6M samples, from the measured rates: b8
  **70.8 h**, b4 **78.1 h**, b12 **70.0 h**. Chaining on `gh` (MaxWall 48 h,
  open question 2) is 2 segments per arm, ≤ 1 min load each — negligible.
  Rollouts for evaluation: minutes per arm (design 5).

| Scope (design 5) | arms | node-h at measured rates | design's range |
|---|---|---|---|
| Minimum | A0a, A0b, A2 | ~212 | 200–400 |
| **Reduced (design's recommendation)** | A0a, A0b, A1, A2 | **~290** | 280–540 |
| Full | + B1, B2 | ~438 | 420–810 |
| + A2x | + 0.9M b12 steps | +35 | +35–70 |

**Recommendation by the design's own logic: GO on the reduced 4-arm
design at ~290 node-h** (bottom of the design's range); B1/B2 stay
conditional on the A-arm result (design 3.2) and would bring the total to
~440, still under the 700 node-h owner-OK line. Wall-clock: each arm is 2
chained 48-h segments; with 4 chains concurrent and the observed 2–6 h
queue waits on `gh`, ~3.5–4 days plus queue. Charge rate: sacct reports
`AllocTRES billing=72` (cpu-weighted TRES on an exclusive node); the design's
"1 SU per node-hour" assumption is **still unverified** against the
allocation's accounting — check before dispatch, as the design asks.

## 4. Corrections to the inherited notes, and what a reviewer would attack

- **The "-n 1 gave 1 CPU" lesson above is wrong.** Both jobs were
  node-exclusive (sacct AllocCPUS 72). The login shell exports
  `OMP_NUM_THREADS=1`; `--export=ALL` carries it into the job, GNU `nproc`
  honours `OMP_NUM_THREADS`, and the sbatch's `${OMP_NUM_THREADS:-8}` default
  never fires. So `node_info` reports `cpus=1 omp=1` on a 72-CPU node, and
  the arms ran with `OMP_NUM_THREADS=1`. The gate itself is unaffected (its
  wrappers set 8 threads and `torch.set_num_threads(1)` explicitly). For the
  arms this is a stated condition, not a confound: GPU SM utilisation was
  94–97 %, so the single-threaded CPU side did not starve the GPU; a rerun
  with 8 threads could only raise the b4 number slightly. Fix for the next
  driver: set `OMP_NUM_THREADS` unconditionally and read CPUs from
  `$SLURM_CPUS_ON_NODE`. Logged in the shared papercuts file.
- `throughput.py --dmon` returned `gpu_sm_util_mean_pct: null` on all three
  arms: `nvidia-smi dmon -o T` prints local wall-clock (UTC-5 here) while
  the parser anchors on epoch timestamps — a timezone mismatch, so the
  window matched no rows. Not fixed here (adjacent); the table's SM numbers
  are from the raw dmon logs.
- n = 1 per arm, one 28-min window, one node. Steady-state fit residuals are
  4–20 steps, so within-run noise is < 1 %; node-to-node variance is
  unmeasured. The b4 vs b8 gap (10 %) is well outside that; b8 vs b12 (1 %)
  is not a measured difference.
- The arms used the published M3 config verbatim (lr 3e-5 at every batch);
  throughput does not depend on LR, so B1/B2 cost equals A1/A2.
- `train.py` has no wall-clock stop (design 3.4 hazard); the pilot relied on
  `timeout`. The chained-segment driver still needs the truncated-checkpoint
  fallback described there before any arm is dispatched.

## Jobs and cost (cumulative)

| Job | Partition | Account | Content | State | Elapsed | Node-h |
|---|---|---|---|---|---|---|
| 1061645 (inherited) | gh | EAR26006 | committed gate | FAILED at gate (cross-hardware) | 00:02:19 | 0.04 |
| 1061807 | gh | EAR26006 | same-hw gate + b8, b4, b12 (28 min each) | COMPLETED 0:0 | 01:28:05 | 1.47 |

**Total 1.51 of the 2.0 node-hour cap.** EAR26005 unused; existing ssh
control socket only; the other project's processes and dev job untouched.
Artifacts on the GH200 scratch under `$SCRATCH/eqgns-m3-sweep-pilot/results/`
(`same_hw_gate.1061807.{json,log}`, `training_gate_reference_gh200.1061807.json`,
`throughput_b{4,8,12}_s1.1061807.json`, `node_info.1061807.txt`) and
`logs/` (`train_*.log` with per-line epoch timestamps, `dmon_*.log`,
`slurm.{o,e}1061807`); scratch purge policy applies. Pilot model directories
hold only `config.json` (checkpoints deleted by the driver).

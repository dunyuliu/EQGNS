# Paper-parity gate (tier 1 + tier 4)

Proves that CURRENT `meshnet/train.py` (rollout-only speed optimizations,
see repo `CLAUDE.md`) reproduces the PUBLISHED rollout behavior
(`meshnet/train.py.published`) from the same checkpoints, within an
empirically-derived tolerance, on the full 826-step / all-trajectory
rollouts for models M1/M2/M3.

## Step 0: data availability

This gate needs `gns-sample/` (249GB, read-only, gitignored) symlinked in
from the machine that holds it, e.g.:

```shell
ln -sfn /home/utig5/dliu/eq_rupture_gns/gns-sample gns-sample
source venv/bin/activate
```

Without it, `run_gate.py` exits 2 with an explicit message, and the pytest
wrapper skips every test with a printed reason -- it is never silently
skipped or silently passed.

## Running it

```shell
# 1. (Re)build the baselines from the published rollout pkls (fast, no GPU).
python3 test/paper_parity/extract_baselines.py --model all

# 2. Re-run rollout with CURRENT code and diff against the baselines
#    (slow, GPU, full 826-step / all-trajectory rollouts -- see runtimes below).
python3 test/paper_parity/run_gate.py --model all --cuda-device 0

# Or via pytest (opt-in, see test/conftest.py's --paper-parity flag):
pytest test/paper_parity -m paper_parity --paper-parity -q -s
```

`run_gate.py` exits 0 iff every trajectory of every requested model is
within tolerance; nonzero otherwise, with a full per-trajectory,
per-metric table printed (baseline value -> current value, |diff|,
allowed tolerance, `*` marking the metrics that failed).

## Metrics computed (per trajectory)

See `common.py` for the full implementation and citations. In summary,
for each rollout pkl:
- `mse_raw` / `mse_vx` / `mse_vy`: rollout MSE of `predicted_rollout` vs
  `ground_truth_rollout`.
- `rupture_time_rmse` / `rupture_time_missed_count` /
  `rupture_time_false_count`: rupture time computed via
  `get_rupture_time(sliprate_hist, DT=0.0167777, threshold=0.1)`, an exact
  port of `utils/plot.rupture.dynamics.py:223-229` (see module docstring
  in `common.py` for line-by-line citations, per PROJECT_RULES.md rule 7).
  `missed` = ground truth ruptures but prediction doesn't; `false` = the
  reverse.

## Tolerance derivation (see `tolerance.json`)

**First pass (insufficient):** ran the M1 rollout twice with current code
(same checkpoint, same test set, GPU 0, A100-SXM4-40GB, torch
2.6.0+cu124) and measured max spread across the 6 trajectories:
`mse_raw` spread 4.20e-4, `mse_vx` spread 8.41e-4, `rupture_time_rmse`
spread 2.70e-5, count metrics 0. A tolerance of `max(2x spread, small
floor)` from this (0.001 / 0.002 / ...) then **failed** on a subsequent
independent run of the same code+checkpoint on `rollout_4`.

**Second pass (used):** ran M1 rollout **four** times total. `rollout_4`'s
`mse_raw` across the four runs was `0.52456, 0.52498, 0.52719, 0.53114`
-- monotonically increasing, spread 6.58e-3 (13x the 2-run estimate).
This is the earthquake-rupture autoregressive rollout being genuinely
sensitive to GPU-kernel-level floating-point nondeterminism amplified
over 826 steps -- most of the physical grounding for this is that
`rollout_4` already has an anomalously high `false_count` (2477) in the
baseline, i.e. it's a borderline/marginal rupture case where tiny
perturbations flip many nodes across the 0.1 m/s threshold. Confirmed the
same phenomenon independently on M2 (`rollout_7`: baseline 0.391, gate-run
0.428, a third run 0.396 -- spread ~0.037, again concentrated in one
trajectory) and, much more severely, on M3 (5 of 15 trajectories moved by
O(0.1)-O(1) in `mse_raw`, up to a 6x change on `rollout_3`) -- consistent
with chaotic sensitivity being more prevalent in the (provenance-
unconfirmed) M3 dataset.

**Chosen tolerance** = `max(MARGIN * observed_max_spread_over_4_runs,
floor)`, with `MARGIN=3` (not the minimum 2x) for `mse_raw`/`mse_vx`
specifically because of the still-unresolved monotonic-drift anomaly
(flagged for `lars-eriksson`/`kai-fischer`, not something this PR
investigates further), `MARGIN=2` elsewhere:

| metric | observed max spread (4 runs) | tolerance |
|---|---|---|
| mse_raw | 6.58e-3 | **0.02** |
| mse_vx | 1.32e-2 | **0.04** |
| mse_vy | 3.57e-12 | 1e-4 |
| rupture_time_rmse | 3.86e-5 | 1e-3 |
| rupture_time_missed_count | 0 | 2 |
| rupture_time_false_count | 0 | 2 |

Mutation-tested: verified `diff_trajectory()` (a) still catches a real
`+0.07` mse_raw injection or a `2477 -> 50` false_count injection as FAIL
after loosening, (b) does **not** silently pass a NaN diff -- the pass
condition is written as the direct comparison `diff <= allowed`, never as
`not (diff > allowed)`, because those two are not equivalent for NaN
(IEEE-754 NaN comparisons are always False, so the negated form
incorrectly reports "pass" on NaN; the direct form correctly fails).

## Known limitation of this gate (important, read before trusting a PASS)

The gate as built diffs *every* trajectory against a *single* fixed
tolerance. Given the chaos evidence above, a small subset of trajectories
(observed: 1/6 for M1, 1/15 for M2, 5/15 for M3) are expected to
sometimes legitimately exceed even this tolerance purely from GPU
nondeterminism, with no code regression involved. **This PR reports gate
results honestly rather than hiding this by inflating the tolerance
further** (a tolerance wide enough to absorb M3's 1.6-magnitude MSE swing
would make the gate unable to catch real regressions of similar size).
Recommended follow-up for a later session: either (a) quantify each
trajectory's own nondeterminism envelope (more repeat runs) and use a
per-trajectory tolerance, or (b) separate "stable" vs "chaos-sensitive"
trajectories and apply a much looser sanity check (order-of-magnitude,
no NaN/inf, comparable rupture area) to the latter instead of a tight
numerical diff.

## M3 provenance caveat

`case4.200m.fractal.stress.homo.a.Vw/` has **no** `.published` rollout
directory. `baseline_M3.json["provenance"] == "unconfirmed"` and
`run_gate.py` prints a loud warning before running M3. A PASS on M3 does
**not** certify agreement with the published paper result for that model
-- only that current code reproduces *this specific* (unverified-origin)
rollout run to within tolerance.

## Runtime observed (2026-09-24, A100-SXM4-40GB, one GPU per model, in
parallel)

| model | trajectories | nnodes | wall-clock |
|---|---|---|---|
| M1 | 6 | 4743 | 217.9 s |
| M2 | 15 | 4692 | 317.2 s |
| M3 | 15 | 4692 | 403.2 s |

These are full, untruncated, all-826-step / all-trajectory rollouts as
mandated for this PR. A future economization PR (truncated horizons,
caching) should beat these numbers; this is the baseline to beat.

## Gate results observed this session

- **M1**: PASS (0 trajectories exceeded tolerance) with the final
  (4-run-derived) tolerance.
- **M2**: FAIL -- `rollout_7` exceeds tolerance on `mse_raw`/`mse_vx`
  (confirmed via a third independent run to be nondeterminism-driven, not
  a regression -- see "Known limitation" above).
- **M3**: FAIL -- 5/15 trajectories exceed tolerance, consistent with
  greater chaotic sensitivity in this (provenance-unconfirmed) dataset.

None of these FAILs were resolved by loosening the tolerance further in
this PR; see "Known limitation" for the recommended next step.

## Zenodo cross-check

See `ZENODO_CROSS_CHECK.md`: **BLOCKED: network/size** -- network access
to the Zenodo API worked, but a real per-file sha256 diff requires
downloading+unzipping 10.33 GB of archives, measured at 389.2 KB/s from
this environment (~7.4h ETA), judged impractical for this session.

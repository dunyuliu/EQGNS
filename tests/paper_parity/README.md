# Paper-parity gate

Checks that the current `meshnet` code reproduces the published GNS results
(Liu & Becker 2025, doi:10.1029/2025JB031981).

## Run

```bash
source venv/bin/activate
python3 tests/paper_parity/gate.py quick            # ~1 min: every edit to meshnet/
python3 tests/paper_parity/gate.py run --cuda 0,1,2,3   # all 7 cases in parallel, before a release
python3 tests/paper_parity/gate.py run M1_D1        # one case
pytest tests/paper_parity --paper-parity -q         # same, via pytest
```

Needs `data/gns-sample/` (published checkpoints and test sets) and a GPU. Skipped in CI.

## How it decides

Every rollout runs in torch deterministic mode (`det_rollout.py`). Without it,
GPU kernel nondeterminism compounds over 826 autoregressive steps and swings
some trajectories' MSE by more than 100% between identical runs, so no fixed
tolerance separates noise from a regression. In deterministic mode reruns are
bit-identical.

- `reference.json`: per-trajectory metrics from `meshnet/train.py.published`
  (the paper's code), deterministic (`gate.py reference`).
- `gate.py run`: the current code must match the reference to 1e-4 relative
  on slip-rate MSE (vx) and rupture-time RMSE / missed / false counts at 0.1 m/s
  (`scripts/utils/plot.rupture.dynamics.py` conventions).
- `gate.py paper`: reference vs the published rollout files (`published.json`),
  as a sanity check that the reference itself reproduces the paper.
- `gate.py falsify M1_D1`: scales all weights by 1.005; the gate must FAIL.
- `gate.py quick`: the most perturbation-sensitive trajectory of each model
  (M1_D1 #4, M2_D3 #14, M3_D3 #7), first 300 steps. `falsify --quick` confirms
  it still catches the planted regression on all three.

If the GPU, CUDA or torch version changes, regenerate `reference.json`: it
comes from the paper's own code, so regenerating it is safe.

## Cases

| Case | Model | Test set |
|---|---|---|
| `M1_D1`, `M1_small` | M1 (D1, 3M steps) | D1 hypocenters; 10 x 5 km fault |
| `M2_D2`, `M2_D3`, `M2_checkerboard` | M2 (D2, 30 scenarios, 3M) | unseen asperity stress; fractal stress; checkerboard |
| `M3_D3`, `M3_D1hypo` | M3 (D2, 148 scenarios, 2.7M) | fractal stress; D1 hypocenter cross-test |

Checkpoints are byte-identical (CRC32) to the Zenodo archive
(doi:10.5281/zenodo.17095311). The 40 km fault case is not gated: its test set
is not on disk.

## Fast opt-in tier (`--rollout_fast`)

```bash
python3 tests/paper_parity/gate.py fast --precision tf32              # full M1_D1/M2_D3/M3_D3 test sets
python3 tests/paper_parity/gate.py fast --precision fp16 --falsify    # planted x1.005 regression must FAIL
```

Judges `meshnet/fast_rollout.py` (see `docs/user/rollout_and_analysis.md`) against the
same `reference.json`, band `FAST_TOL = {rt_rmse: 1.05x, missed+false: 2.0x, mse_vx: 1.5x}`
(`gate.py`) rather than the 1e-4 exact match `gate.py run` uses, since the fast path's
rounding is not bit-identical and the rollout is chaotic. **This ratio-vs-EQdyna-truth band is
provisional/informational, not the final gate.** The owner wants the fast-tier gate redefined as
a direct regression check against the deterministic reference rollout itself (not EQdyna truth):
rupture-time RMSE in seconds (target ~2 dt, dt=0.0168s), Mw error, and peak-normalized slip-rate
RMSE, with thresholds set from a measured per-trajectory distribution across all 8 cases (fast
tf32/fp32 vs reference, and eager-vs-eager for the noise floor). That measurement and the
redesign are GPU-heavy and queued in the `test-suite-overhaul` board row (after this PR, GPU 1
free) — it does not block this PR, whose own merge gate is the table below (real and
falsify-verified, just not the final design).

Measured 2026-10-08, idle GPU 1, deterministic:

| precision | M1_D1 | M2_D3 | M3_D3 | verdict |
|---|---|---|---|---|
| fp32 | rt 0.3518/0.3518, m+f 4589/4589, mse_vx 0.543/0.542 | rt 0.318/0.341, m+f 236/147, mse_vx 1.23/1.50 | rt 0.272/0.272, m+f 38/38, mse_vx 0.619/0.790 | **PASS all 3** |
| tf32 | rt 0.347/0.352, m+f 4589/4589, mse_vx 0.518/0.542 | rt 0.338/0.341, m+f 234/147, mse_vx 1.35/1.50 | rt 0.274/0.272, m+f 38/38, mse_vx 0.625/0.790 | **PASS all 3** |
| fp16 | PASS | mse_vx 3.581/1.50 | PASS | FAIL (M2_D3 mse_vx) |
| bf16 | PASS | rt_rmse 0.391/0.341 | rt_rmse 0.287/0.272 | FAIL (M2_D3, M3_D3 rt_rmse) |

`current/reference` per case; `m+f` = missed+false trajectory count. Eager (flag off)
run-to-run noise on M2_D3 mse_vx was 0.85 and 1.73 against a 1.50 reference in separate
runs, so fp16's 3.58 is a real regression, not noise. `--falsify --precision fp16`
(weights x1.005) is CAUGHT on M1_D1 and M2_D3 but PASSes on M3_D3 alone — the gate runs
all 3 cases for exactly this reason; no single case is sufficient.

No precision has an owner-approved default yet; `tf32` is the only one passing the
current band on every case. `gate.py run` (flag off) is unaffected and passed all 8
cases in the same session.

## Dataset padding

The prepared test sets end each scenario with 72 padded frames (steps 755-826:
zero velocity, invalid `node_coords`/`cells`/`node_property`). The published
code rebuilds the graph each step and so reads them; the current code caches
the step-0 mesh. Both are bit-identical over steps 0-754, so metrics use only
the unpadded steps (`valid_steps` in `gate.py`). Including the padded frames
dilutes MSE by 826/755 (about 9%), since both predictions and ground truth are
near zero there.

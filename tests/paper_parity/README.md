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

## Dataset padding

The prepared test sets end each scenario with 72 padded frames (steps 755-826:
zero velocity, invalid `node_coords`/`cells`/`node_property`). The published
code rebuilds the graph each step and so reads them; the current code caches
the step-0 mesh. Both are bit-identical over steps 0-754, so metrics use only
the unpadded steps (`valid_steps` in `gate.py`). Including the padded frames
dilutes MSE by 826/755 (about 9%), since both predictions and ground truth are
near zero there.

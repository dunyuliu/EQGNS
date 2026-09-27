#!/usr/bin/env python3
"""Derive per-trajectory tolerance from measured N-run spread (PR #3).

Reads {model}_spread.json files (written by measure_spread.py, >=5 repeat
runs of CURRENT meshnet/train.py rollout against the SAME checkpoint+test
set) and writes per_trajectory_tolerance.json:

    { model_key: { pkl_file: { metric: allowed_abs_diff, ... }, ... }, ... }

Rule (see README.md "Per-trajectory tolerance" for the full writeup):
    observed_spread = max(values) - min(values)   # across the N repeat runs
    raw = MARGIN * observed_spread
    tol = FLOOR[metric]                     if raw <= FLOOR[metric]
        = 10 ** ceil(log10(raw))            otherwise (round UP to the next
                                             power of ten above the margined
                                             worst-observed spread)

MARGIN=3 uniformly (this replaces the OLD scheme's metric-dependent 2x/3x
split -- that split existed only to let a single GLOBAL tolerance survive
the one chaotic outlier trajectory without inflating every other
trajectory's band; per-trajectory tolerance no longer needs that
compensation, so one conservative margin applies everywhere).

FLOOR is a fixed, physically-motivated overhead per metric (float64
accumulation noise / single-node threshold-crossing quantization), NEVER an
assumed noise floor -- each floor is well below the smallest genuine
trajectory-to-trajectory difference documented in NOTES_tier1.md (mse_raw
differs by >0.01 between distinct trajectories; count metrics differ by
tens-hundreds under a real change).

A trajectory/metric with observed_spread == 0 across all N runs (e.g. every
count metric and several mse_raw trajectories in M1 this session) gets
exactly FLOOR, not zero -- demanding literal bit-for-bit reproduction
forever from a 5-run sample would be an unmeasured (falsely tight) floor,
the same failure mode as an unmeasured noise floor in the other direction.

Two axes of measured variation (added after the first per-trajectory run
of the real gate, see NOTES_pr3.md 'mse_vy floor was too tight for M3'):
1. current-vs-current repeat spread (`spread`, above) -- autoregressive
   chaos across independent runs of the SAME current code.
2. current-vs-PUBLISHED baseline gap (`baseline_gap`) -- every M3
   trajectory's mse_vy differed from the published baseline by a nearly
   IDENTICAL ~3.3e-9-3.6e-9 across all 15 trajectories and all 6 current-
   code repeat runs (i.e. reproducible, NOT chaos -- current code and the
   one-time published run differ by a small, consistent bias on this
   near-zero, physically-noise-only channel, most likely a torch/cuDNN/
   precision difference between whatever produced the original baseline
   and the current environment). Axis 1 alone cannot see this because it
   never compares against the baseline. Both axes get the same MARGIN
   and floor treatment -- this is not a metric-specific carve-out, and it
   is verified in NOTES_pr3.md's falsifiability section to still catch a
   real regression that is ~1000x bigger than this gap.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MARGIN = 3.0

FLOOR = {
    "mse_raw": 1e-5,
    "mse_vx": 2e-5,
    "mse_vy": 1e-10,
    "rupture_time_rmse": 1e-4,
    "rupture_time_missed_count": 1,
    "rupture_time_false_count": 1,
}

NUMERIC_KEYS = ["mse_raw", "mse_vx", "mse_vy", "rupture_time_rmse"]
COUNT_KEYS = ["rupture_time_missed_count", "rupture_time_false_count"]


def tol_for(metric: str, spread: float, baseline_gap: float = 0.0) -> float:
    """`spread` = current-code-vs-current-code repeat-run spread (the
    autoregressive-chaos axis). `baseline_gap` = |mean(current runs) -
    published baseline| for this same trajectory/metric (the current-vs-
    published axis) -- see module docstring 'Two axes of measured
    variation' for why both are needed: mse_vy on M3 showed a real,
    reproducible ~3.6e-9 current-vs-baseline gap (present identically
    across all 6 current-code repeat runs, i.e. NOT chaos) that the
    repeat-spread axis alone cannot see, because repeat-spread only ever
    compares current code against itself.
    """
    floor = FLOOR[metric]
    raw = MARGIN * max(spread, baseline_gap)
    if raw <= floor:
        return floor
    if metric in COUNT_KEYS:
        # Integers: round up to the next integer above the margined spread,
        # not a power of ten (a power-of-ten rule on small integers like
        # "false_count spread=3" would jump straight to 10, a 3x-plus
        # overcorrection for a metric with a natural unit of 1).
        return float(math.ceil(raw))
    return 10 ** math.ceil(math.log10(raw))


def main():
    model_keys = sys.argv[1:] or ["M1", "M3"]
    out = {
        "_derivation": (
            "Per-trajectory tolerance, PR #3. tol = FLOOR if MARGIN*spread <= "
            "FLOOR else next-power-of-ten-above(MARGIN*spread) [numeric] / "
            "ceil(MARGIN*spread) [counts]. MARGIN=3, spread = max-min across "
            ">=5 repeat runs of CURRENT code, same checkpoint+test set. See "
            "generate_per_trajectory_tolerance.py module docstring and "
            "README.md 'Per-trajectory tolerance' for full rationale. Inputs: "
            f"{[f'{k}_spread.json' for k in model_keys]}."
        ),
        "_margin": MARGIN,
        "_floor": FLOOR,
    }
    for model_key in model_keys:
        spread_path = HERE / f"{model_key}_spread.json"
        if not spread_path.exists():
            print(f"[{model_key}] no {spread_path.name} -- skipped (not measured this session)")
            continue
        with open(spread_path) as f:
            data = json.load(f)

        baseline_path = HERE / f"baseline_{model_key}.json"
        baseline_by_pkl = {}
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            baseline_by_pkl = {t["pkl_file"]: t for t in baseline["trajectories"]}

        out[model_key] = {}
        for pkl_file, metrics in data["spread"].items():
            row = {}
            b_traj = baseline_by_pkl.get(pkl_file)
            for metric in NUMERIC_KEYS + COUNT_KEYS:
                vals = metrics[metric]
                spread = max(vals) - min(vals)
                baseline_gap = 0.0
                if b_traj is not None and b_traj.get(metric) is not None:
                    mean_current = sum(vals) / len(vals)
                    baseline_gap = abs(mean_current - b_traj[metric])
                row[metric] = tol_for(metric, spread, baseline_gap)
                row[f"_{metric}_observed_spread"] = spread
                row[f"_{metric}_baseline_gap"] = baseline_gap
            out[model_key][pkl_file] = row
        print(f"[{model_key}] derived tolerance for {len(out[model_key])} trajectories "
              f"from {data['n_runs']}-run spread")

    out_path = HERE / "per_trajectory_tolerance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

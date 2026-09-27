#!/usr/bin/env python3
"""Derive per-trajectory tolerance for the truncated-horizon 'quick' tier
(PR #4), from >=3-run spread measured by measure_truncated_spread.py.

Reuses generate_per_trajectory_tolerance.py's `tol_for()` (MARGIN=3,
per-metric FLOOR, same rule: tol = FLOOR if MARGIN*max(spread,
baseline_gap) <= FLOOR else next-power-of-ten-above(...)) UNMODIFIED --
this is the SAME methodology PR #3 used for the full-length tolerance,
not a new invention, applied at the truncated horizon because a truncated
rollout's natural per-step noise is NOT assumed to equal the full-length
tolerance (a truncated run may be tighter -- fewer autoregressive steps
means less time for GPU-kernel nondeterminism to compound -- or could in
principle differ some other way; either way it must be MEASURED, not
borrowed).

Inputs: {model}_truncated{n}_spread.json (measure_truncated_spread.py),
baseline_{model}_truncated{n}.json (extract_truncated_baseline.py).
Output: per_trajectory_tolerance_truncated.json, same nested shape as
per_trajectory_tolerance.json:
    { "<model>_truncated<n>": { pkl_file: { metric: allowed_abs_diff } } }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_per_trajectory_tolerance import FLOOR, MARGIN, NUMERIC_KEYS, COUNT_KEYS, tol_for  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: generate_truncated_tolerance.py <model_key> <nsteps> [<model_key> <nsteps> ...]")
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))

    out = {
        "_derivation": (
            "Per-trajectory tolerance for the truncated-horizon quick tier, PR #4. "
            "Same rule/MARGIN/FLOOR as generate_per_trajectory_tolerance.py (PR #3), "
            "reused unmodified, applied to spread MEASURED at the truncated horizon "
            "(not borrowed from the full-length tolerance.json/"
            "per_trajectory_tolerance.json)."
        ),
        "_margin": MARGIN,
        "_floor": FLOOR,
    }
    for model_key, nsteps_str in pairs:
        nsteps = int(nsteps_str)
        key = f"{model_key}_truncated{nsteps}"
        spread_path = HERE / f"{model_key}_truncated{nsteps}_spread.json"
        baseline_path = HERE / f"baseline_{model_key}_truncated{nsteps}.json"
        if not spread_path.exists():
            print(f"[{key}] no {spread_path.name} -- skipped (not measured)")
            continue
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"{baseline_path} not found -- run extract_truncated_baseline.py first")

        with open(spread_path) as f:
            data = json.load(f)
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_by_pkl = {t["pkl_file"]: t for t in baseline["trajectories"]}

        out[key] = {}
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
            out[key][pkl_file] = row
        print(f"[{key}] derived tolerance for {len(out[key])} trajectories "
              f"from {data['n_runs']}-run spread at nsteps={nsteps}")

    out_path = HERE / "per_trajectory_tolerance_truncated.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

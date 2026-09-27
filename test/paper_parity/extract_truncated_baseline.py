#!/usr/bin/env python3
"""Truncated-horizon baseline extraction (PR #4 'quick tier').

Slices the PUBLISHED rollout pkls to their first N steps and computes
per_trajectory_metrics on the sliced arrays -- reuses common.py's exact
metric function UNMODIFIED (per mission instructions: "reuse
common.py::per_trajectory_metrics on a sliced array, don't invent new
metrics"), never a new methodology.

Why slicing the full published pkl is a valid N-step oracle (not an
approximation): rollout() is autoregressive -- prediction at step k
depends only on predictions at steps < k, never on any future step. A
current-code run that is stopped after N steps therefore produces, by
construction, EXACTLY the same first-N-step predictions that the full
826-step published run produced for those same steps. Slicing the full
published pkl to N steps is thus the correct "what would this published
run's own first N steps have been" ground truth, not a stand-in.

Writes baseline_<model_key>_truncated<N>.json into this directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ALL_REGISTRY, find_published_pkls, load_pkl, model_paths, per_trajectory_metrics  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def truncate_pkl(pkl: dict, n: int) -> dict:
    pred = np.asarray(pkl["predicted_rollout"])
    gt = np.asarray(pkl["ground_truth_rollout"])
    if n > pred.shape[0]:
        raise ValueError(
            f"requested truncation N={n} exceeds full rollout length {pred.shape[0]}")
    return {
        "predicted_rollout": pred[:n],
        "ground_truth_rollout": gt[:n],
        "initial_velocities": pkl["initial_velocities"],
    }


def extract_one(model_key: str, n: int) -> dict:
    paths = model_paths(model_key)
    pkls = find_published_pkls(paths["published_rollout_dir"])

    trajectories = []
    for pkl_path in pkls:
        pkl = load_pkl(pkl_path)
        truncated = truncate_pkl(pkl, n)
        metrics = per_trajectory_metrics(truncated)
        metrics["pkl_file"] = pkl_path.name
        trajectories.append(metrics)
        print(f"  {pkl_path.name}: mse_raw={metrics['mse_raw']:.6f} "
              f"mse_vx={metrics['mse_vx']:.6f} mse_vy={metrics['mse_vy']:.8f} "
              f"rpt_rmse={metrics['rupture_time_rmse']} "
              f"missed={metrics['rupture_time_missed_count']} "
              f"false={metrics['rupture_time_false_count']}")

    return {
        "model_key": model_key,
        "nsteps": n,
        "provenance": paths["provenance"],
        "published_rollout_dir": str(paths["published_rollout_dir"]),
        "n_trajectories": len(trajectories),
        "trajectories": trajectories,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(ALL_REGISTRY))
    ap.add_argument("--nsteps", type=int, required=True)
    args = ap.parse_args()

    baseline = extract_one(args.model, args.nsteps)
    out_path = OUT_DIR / f"baseline_{args.model}_truncated{args.nsteps}.json"
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

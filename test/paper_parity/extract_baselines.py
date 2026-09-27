#!/usr/bin/env python3
"""One-time (rerunnable) extraction of paper-parity baselines from the
PUBLISHED rollout pkls for M1/M2/M3.

Writes baseline_M1.json, baseline_M2.json, baseline_M3.json into this
directory. Never reads gns-sample/ except for `.npz`/`.pt`/`.pkl` files
already committed there as read-only ground truth (PROJECT_RULES.md rule 3);
never writes through the gns-sample symlink.

Usage:
    python3 test/paper_parity/extract_baselines.py --model M1
    python3 test/paper_parity/extract_baselines.py --model all

Note: `test/` is intentionally NOT a package (no __init__.py) -- it shares
its name with the CPython stdlib `test` package and `-m test.paper_parity...`
can silently resolve against that instead. Run this as a plain script (or
via the pytest wrapper) rather than `-m`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import (  # noqa: E402
    ALL_REGISTRY,
    find_published_pkls,
    load_pkl,
    model_paths,
    per_trajectory_metrics,
    sha256_of_file,
)

OUT_DIR = Path(__file__).resolve().parent


def extract_one(model_key: str) -> dict:
    paths = model_paths(model_key)
    pkls = find_published_pkls(paths["published_rollout_dir"])

    print(f"[{model_key}] hashing checkpoint {paths['checkpoint']} ...")
    checkpoint_sha256 = sha256_of_file(paths["checkpoint"])
    print(f"[{model_key}] hashing test set {paths['test_npz']} ...")
    test_npz_sha256 = sha256_of_file(paths["test_npz"])

    trajectories = []
    for pkl_path in pkls:
        pkl = load_pkl(pkl_path)
        metrics = per_trajectory_metrics(pkl)
        metrics["pkl_file"] = pkl_path.name
        metrics["mean_loss_reported"] = float(pkl["mean_loss"])
        mean_acc_loss = pkl.get("mean_acc_loss")
        metrics["mean_acc_loss_reported"] = (
            float(mean_acc_loss) if mean_acc_loss is not None else None)
        trajectories.append(metrics)
        print(f"  {pkl_path.name}: mse_raw={metrics['mse_raw']:.4f} "
              f"mse_vx={metrics['mse_vx']:.4f} mse_vy={metrics['mse_vy']:.4f} "
              f"rpt_rmse={metrics['rupture_time_rmse']} "
              f"missed={metrics['rupture_time_missed_count']} "
              f"false={metrics['rupture_time_false_count']}")

    baseline = {
        "model_key": model_key,
        "provenance": paths["provenance"],
        "published_rollout_dir": str(paths["published_rollout_dir"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "test_npz_path": str(paths["test_npz"]),
        "test_npz_sha256": test_npz_sha256,
        "n_trajectories": len(trajectories),
        "trajectories": trajectories,
    }
    if paths["provenance"] != "published":
        print(f"WARNING: [{model_key}] provenance is {paths['provenance']!r} -- "
              f"this baseline is NOT a confirmed paper-parity oracle. See NOTES_tier1.md.")
    return baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(ALL_REGISTRY) + ["all"], default="all")
    args = ap.parse_args()

    keys = list(ALL_REGISTRY) if args.model == "all" else [args.model]
    for key in keys:
        baseline = extract_one(key)
        out_path = OUT_DIR / f"baseline_{key}.json"
        with open(out_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"[{key}] wrote {out_path}")


if __name__ == "__main__":
    main()

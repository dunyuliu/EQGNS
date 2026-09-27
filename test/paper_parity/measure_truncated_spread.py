#!/usr/bin/env python3
"""Repeat-run spread measurement for the truncated-horizon 'quick' tier
(PR #4). Same methodology as measure_spread.py (PR #3): >=3 repeat runs of
CURRENT `meshnet/train.py` rollout, launched through
test/fixtures/paper_parity/truncated_rollout_cli.py (monkeypatches the
`rollout` nsteps cap, never edits meshnet/train.py), writing
{model_key}_truncated{n}_spread.json in the SAME shape measure_spread.py
writes so generate_per_trajectory_tolerance.py's tol_for() can be reused
unmodified by generate_truncated_tolerance.py.

Also supports MESHNET_SRC_ROOT-style tree override (--meshnet-src-root)
for the falsifiability re-check (test_falsifiability.py pattern): point it
at a throwaway copy of meshnet/+gns/ with a planted defect.

Usage:
    python3 test/paper_parity/measure_truncated_spread.py \
        --model M1 --nsteps 100 --n-runs 3 --cuda-device 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import load_pkl, model_paths, per_trajectory_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

NUMERIC_KEYS = ["mse_raw", "mse_vx", "mse_vy", "rupture_time_rmse"]
COUNT_KEYS = ["rupture_time_missed_count", "rupture_time_false_count"]


def run_once(model_key, nsteps, work_dir, run_idx, cuda_device, meshnet_src_root=None):
    paths = model_paths(model_key)
    output_dir = work_dir / f"run{run_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["TRUNCATED_ROLLOUT_NSTEPS"] = str(nsteps)
    if meshnet_src_root is not None:
        env["PYTHONPATH"] = str(meshnet_src_root) + os.pathsep + env.get("PYTHONPATH", "")

    cwd = str(meshnet_src_root) if meshnet_src_root is not None else str(REPO_ROOT)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "fixtures" / "paper_parity" / "truncated_rollout_cli.py"),
        "--mode=rollout",
        f"--data_path={paths['data_path']}/",
        f"--model_path={paths['model_path']}/",
        f"--output_path={output_dir}/",
        f"--model_file={paths['model_file']}",
        f"--train_state_file={paths['train_state_file']}",
        f"--cuda_device_number={cuda_device}",
    ]
    print(f"[{model_key} run{run_idx}] truncated N={nsteps} launching (cwd={cwd}) ...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"[{model_key} run{run_idx}] truncated rollout failed (exit {result.returncode})")

    pkls = sorted(output_dir.glob("rollout_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    trajectories = {}
    for pkl_path in pkls:
        pkl = load_pkl(pkl_path)
        metrics = per_trajectory_metrics(pkl)
        trajectories[pkl_path.name] = metrics
    print(f"[{model_key} run{run_idx}] done in {elapsed:.1f}s, {len(pkls)} trajectories")
    return trajectories, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nsteps", type=int, required=True)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--cuda-device", type=int, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/paper_parity_truncated_spread"))
    ap.add_argument("--meshnet-src-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    work_dir = args.work_dir / args.model / f"truncated{args.nsteps}"
    per_run = []
    elapsed_all = []
    for i in range(args.n_runs):
        trajectories, elapsed = run_once(
            args.model, args.nsteps, work_dir, i, args.cuda_device, args.meshnet_src_root)
        per_run.append(trajectories)
        elapsed_all.append(elapsed)

    pkl_files = sorted(per_run[0].keys(), key=lambda s: int(s.split("_")[-1].split(".")[0]))
    spread = {}
    for pkl_file in pkl_files:
        spread[pkl_file] = {}
        for key in NUMERIC_KEYS + COUNT_KEYS:
            vals = [run[pkl_file][key] for run in per_run]
            spread[pkl_file][key] = vals

    out = {
        "model_key": args.model,
        "nsteps": args.nsteps,
        "n_runs": args.n_runs,
        "cuda_device": args.cuda_device,
        "elapsed_per_run": elapsed_all,
        "spread": spread,
    }
    out_path = args.out or (HERE / f"{args.model}_truncated{args.nsteps}_spread.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

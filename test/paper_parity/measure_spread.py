#!/usr/bin/env python3
"""Repeat-run spread measurement for the paper-parity gate tolerance (PR #3).

Runs `python3 -m meshnet.train --mode=rollout` N times for one model key
(CURRENT meshnet/train.py, unmodified unless MESHNET_SRC_ROOT points at a
copy for the falsifiability control), computes per_trajectory_metrics for
each run, and writes {model_key}_spread.json: {pkl_file: {metric: [v_run0,
v_run1, ...]}}.

This is the empirical input to the per-trajectory tolerance scheme in
per_trajectory_tolerance.py -- see NOTES_pr3.md for how it's used.

Optional determinism-mode experiment: --deterministic sets
CUBLAS_WORKSPACE_CONFIG=:4096:8 and torch.use_deterministic_algorithms(True)
around the rollout subprocess (via env var read by
test/fixtures/meshnet/_determinism_shim.py-style sitecustomize injection is
overkill here -- instead we set the env var and pass a flag meshnet/train.py
does NOT need to know about: we monkey-patch via PYTHONSTARTUP-less approach,
concretely by prepending a tiny -c snippet is not compatible with -m; instead
we set CUBLAS_WORKSPACE_CONFIG in the subprocess env, which affects cuBLAS
determinism for cases where torch already calls use_deterministic_algorithms
internally -- see NOTES_pr3.md 'Determinism experiment caveat' for the exact
mechanism used (env-only, since editing meshnet/train.py is out of scope).

Usage:
    python3 test/paper_parity/measure_spread.py --model M1 --n-runs 5 --cuda-device 3
    python3 test/paper_parity/measure_spread.py --model M1 --n-runs 5 --cuda-device 3 --deterministic
"""
from __future__ import annotations

import argparse
import json
import os
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


def run_once(model_key, work_dir, run_idx, cuda_device, deterministic, meshnet_src_root):
    import subprocess
    paths = model_paths(model_key)
    output_dir = work_dir / f"run{run_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if meshnet_src_root is not None:
        env["PYTHONPATH"] = str(meshnet_src_root) + os.pathsep + env.get("PYTHONPATH", "")
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if deterministic:
        entry = [sys.executable,
                 str(REPO_ROOT / "test" / "fixtures" / "paper_parity"
                     / "deterministic_rollout_cli.py")]
    else:
        entry = [sys.executable, "-m", "meshnet.train"]

    cmd = entry + [
        "--mode=rollout",
        f"--data_path={paths['data_path']}/",
        f"--model_path={paths['model_path']}/",
        f"--output_path={output_dir}/",
        f"--model_file={paths['model_file']}",
        f"--train_state_file={paths['train_state_file']}",
        f"--cuda_device_number={cuda_device}",
    ]
    cwd = str(meshnet_src_root) if meshnet_src_root is not None else str(REPO_ROOT)
    print(f"[{model_key} run{run_idx}] launching (deterministic={deterministic}, cwd={cwd}) ...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"[{model_key} run{run_idx}] rollout failed (exit {result.returncode})")

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
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--cuda-device", type=int, required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/paper_parity_spread"))
    ap.add_argument("--meshnet-src-root", type=Path, default=None,
                     help="For falsifiability control: PYTHONPATH root containing a "
                          "(possibly perturbed) copy of meshnet/. Default: this repo.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    work_dir = args.work_dir / args.model / ("det" if args.deterministic else "nondet")
    per_run = []
    elapsed_all = []
    for i in range(args.n_runs):
        trajectories, elapsed = run_once(
            args.model, work_dir, i, args.cuda_device, args.deterministic,
            args.meshnet_src_root)
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
        "n_runs": args.n_runs,
        "deterministic": args.deterministic,
        "cuda_device": args.cuda_device,
        "elapsed_per_run": elapsed_all,
        "spread": spread,
    }
    out_path = args.out or (HERE / f"{args.model}_spread{'_det' if args.deterministic else ''}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

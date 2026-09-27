#!/usr/bin/env python3
"""Tier-1 paper-parity gate.

Re-runs rollout using CURRENT `meshnet/train.py` (via the documented
invocation in docs/rollout_and_analysis.md:

    python3 -m meshnet.train --mode=rollout \
      --data_path=<working_dir>/dataset/ \
      --model_path=<working_dir>/models.<suffix>/ \
      --output_path=<working_dir>/rollouts.<suffix>/ \
      --model_file=model-<step>.pt --train_state_file=train_state-<step>.pt

) from the SAME published checkpoint and test set used to produce the
published pkls, computes the identical per-trajectory metrics
(test/paper_parity/common.py::per_trajectory_metrics), and diffs against
the committed baseline_<MODEL>.json.

Exit code 0 iff every trajectory of every requested model is within
tolerance (see tolerance.json + README.md for how the tolerance was
derived). Nonzero otherwise, with a per-trajectory table printed.

Usage:
    python3 test/paper_parity/run_gate.py --model M1
    python3 test/paper_parity/run_gate.py --model all
    python3 test/paper_parity/run_gate.py --model M1 --cuda-device 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ALL_REGISTRY, gns_sample_available, load_pkl, model_paths, per_trajectory_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_TOLERANCE_PATH = HERE / "tolerance.json"


def run_current_rollout(model_key: str, output_dir: Path, cuda_device=None):
    """Invoke `python3 -m meshnet.train --mode=rollout` (current code) and
    return (list_of_pkl_paths, wall_clock_seconds)."""
    paths = model_paths(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "meshnet.train",
        "--mode=rollout",
        f"--data_path={paths['data_path']}/",
        f"--model_path={paths['model_path']}/",
        f"--output_path={output_dir}/",
        f"--model_file={paths['model_file']}",
        f"--train_state_file={paths['train_state_file']}",
    ]
    if cuda_device is not None:
        cmd.append(f"--cuda_device_number={cuda_device}")

    print(f"[{model_key}] running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(
            f"[{model_key}] rollout subprocess failed (exit {result.returncode}); see output above")

    pkls = sorted(output_dir.glob("rollout_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    if not pkls:
        raise RuntimeError(f"[{model_key}] rollout produced no rollout_*.pkl files in {output_dir}")
    return pkls, elapsed


def compute_current_metrics(model_key: str, output_dir: Path, cuda_device=None):
    pkls, elapsed = run_current_rollout(model_key, output_dir, cuda_device=cuda_device)
    trajectories = []
    for pkl_path in pkls:
        pkl = load_pkl(pkl_path)
        metrics = per_trajectory_metrics(pkl)
        metrics["pkl_file"] = pkl_path.name
        trajectories.append(metrics)
    return trajectories, elapsed


NUMERIC_KEYS = ["mse_raw", "mse_vx", "mse_vy", "rupture_time_rmse"]
COUNT_KEYS = ["rupture_time_missed_count", "rupture_time_false_count"]


def load_tolerance(path: Path = DEFAULT_TOLERANCE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Derive it first (see README.md 'Tolerance derivation') "
            f"and commit tolerance.json before running the gate.")
    with open(path) as f:
        return json.load(f)


def diff_trajectory(baseline: dict, current: dict, tol: dict) -> dict:
    """Return {metric: (ok, baseline_val, current_val, abs_diff, allowed)}.

    IMPORTANT: `ok` is computed as the direct comparison `diff <= allowed`,
    never as `not (diff > allowed)`. Those two are NOT equivalent when
    `diff` is NaN: IEEE-754 comparisons involving NaN are always False, so
    `diff > allowed` is False for NaN, and its negation `not (...)` would
    incorrectly evaluate to True ("pass"). `diff <= allowed` is also False
    for NaN, so a NaN diff correctly and directly fails the gate.
    """
    out = {}
    for key in NUMERIC_KEYS:
        b, c = baseline[key], current[key]
        if b is None or c is None:
            # Explicit: a None (no valid rupture-time overlap) on either side
            # is only "ok" if both sides are None -- never silently pass.
            out[key] = (b == c, b, c, None, tol[key])
            continue
        diff = abs(c - b)
        allowed = tol[key]
        out[key] = (bool(diff <= allowed), b, c, diff, allowed)
    for key in COUNT_KEYS:
        b, c = baseline[key], current[key]
        diff = abs(c - b)
        allowed = tol[key]
        out[key] = (bool(diff <= allowed), b, c, diff, allowed)
    return out


def run_gate_for_model(model_key: str, tol: dict, work_root: Path, cuda_device=None) -> tuple[bool, list, float]:
    paths = model_paths(model_key)
    if paths["provenance"] != "published":
        print(f"\n{'!'*70}\nWARNING: model {model_key} baseline provenance is "
              f"{paths['provenance']!r} -- NOT a confirmed paper-parity oracle.\n"
              f"A gate PASS here does not certify agreement with the published paper "
              f"result for {model_key}. See NOTES_tier1.md.\n{'!'*70}\n")

    baseline_path = HERE / f"baseline_{model_key}.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"{baseline_path} not found. Run extract_baselines.py --model {model_key} first.")
    with open(baseline_path) as f:
        baseline = json.load(f)

    output_dir = work_root / model_key
    current_trajectories, elapsed = compute_current_metrics(model_key, output_dir, cuda_device=cuda_device)

    baseline_trajectories = baseline["trajectories"]
    if len(baseline_trajectories) != len(current_trajectories):
        raise RuntimeError(
            f"[{model_key}] trajectory count mismatch: baseline has "
            f"{len(baseline_trajectories)}, current run produced {len(current_trajectories)}")

    all_ok = True
    rows = []
    for b_traj, c_traj in zip(baseline_trajectories, current_trajectories):
        if b_traj["pkl_file"] != c_traj["pkl_file"]:
            raise RuntimeError(
                f"[{model_key}] pkl ordering mismatch: baseline {b_traj['pkl_file']} "
                f"vs current {c_traj['pkl_file']}")
        diffs = diff_trajectory(b_traj, c_traj, tol)
        traj_ok = all(v[0] for v in diffs.values())
        all_ok = all_ok and traj_ok
        rows.append((b_traj["pkl_file"], traj_ok, diffs))

    return all_ok, rows, elapsed


def print_table(model_key: str, rows):
    print(f"\n=== {model_key} paper-parity gate: per-trajectory results ===")
    header = f"{'traj':<16}{'status':<8}"
    for key in NUMERIC_KEYS + COUNT_KEYS:
        header += f"{key:<28}"
    print(header)
    for pkl_file, traj_ok, diffs in rows:
        status = "PASS" if traj_ok else "FAIL"
        line = f"{pkl_file:<16}{status:<8}"
        for key in NUMERIC_KEYS + COUNT_KEYS:
            ok, b, c, diff, allowed = diffs[key]
            mark = "" if ok else "*"
            line += f"{f'{b}->{c} (d={diff}<= {allowed}){mark}':<28}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(ALL_REGISTRY) + ["all"], default="all")
    ap.add_argument("--cuda-device", type=int, default=None)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/paper_parity_gate_output"),
                     help="Scratch dir for re-run rollout pkls (not committed).")
    ap.add_argument("--tolerance", type=Path, default=DEFAULT_TOLERANCE_PATH)
    args = ap.parse_args()

    if not gns_sample_available():
        print("gns-sample/ is not available (empty or missing) -- cannot run the paper-parity "
              "gate here. This is expected in CI; run locally with the symlink set up.")
        sys.exit(2)

    tol = load_tolerance(args.tolerance)
    keys = list(ALL_REGISTRY) if args.model == "all" else [args.model]

    overall_ok = True
    total_elapsed = 0.0
    for key in keys:
        ok, rows, elapsed = run_gate_for_model(key, tol, args.work_dir, cuda_device=args.cuda_device)
        total_elapsed += elapsed
        print_table(key, rows)
        print(f"[{key}] rollout wall-clock: {elapsed:.1f}s, gate: {'PASS' if ok else 'FAIL'}")
        overall_ok = overall_ok and ok

    print(f"\nTotal wall-clock across requested models: {total_elapsed:.1f}s")
    print(f"\nOVERALL GATE: {'PASS' if overall_ok else 'FAIL'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()

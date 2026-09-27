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
DEFAULT_PER_TRAJECTORY_TOLERANCE_PATH = HERE / "per_trajectory_tolerance.json"


def run_current_rollout(model_key: str, output_dir: Path, cuda_device=None, truncated_nsteps=None):
    """Invoke `python3 -m meshnet.train --mode=rollout` (current code) and
    return (list_of_pkl_paths, wall_clock_seconds).

    `truncated_nsteps`, when set, switches the entry point to
    `test/fixtures/paper_parity/truncated_rollout_cli.py` (PR #4 'quick'
    tier) instead of `-m meshnet.train` directly -- this ONLY caps the
    number of autoregressive rollout steps actually executed (test infra,
    never edits meshnet/train.py; see that file's module docstring). The
    default (`truncated_nsteps=None`) path below is BYTE-FOR-BYTE the
    original full-length invocation -- the quick tier is strictly
    additive, never a substitute for it.
    """
    paths = model_paths(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None
    if truncated_nsteps is not None:
        entry = [sys.executable,
                 str(REPO_ROOT / "test" / "fixtures" / "paper_parity" / "truncated_rollout_cli.py")]
        import os
        env = dict(os.environ)
        env["TRUNCATED_ROLLOUT_NSTEPS"] = str(truncated_nsteps)
    else:
        entry = [sys.executable, "-m", "meshnet.train"]

    cmd = entry + [
        "--mode=rollout",
        f"--data_path={paths['data_path']}/",
        f"--model_path={paths['model_path']}/",
        f"--output_path={output_dir}/",
        f"--model_file={paths['model_file']}",
        f"--train_state_file={paths['train_state_file']}",
    ]
    if cuda_device is not None:
        cmd.append(f"--cuda_device_number={cuda_device}")

    print(f"[{model_key}] running: {' '.join(cmd)}"
          + (f" (quick tier, N={truncated_nsteps})" if truncated_nsteps is not None else ""))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
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


def compute_current_metrics(model_key: str, output_dir: Path, cuda_device=None, truncated_nsteps=None):
    pkls, elapsed = run_current_rollout(
        model_key, output_dir, cuda_device=cuda_device, truncated_nsteps=truncated_nsteps)
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


def load_per_trajectory_tolerance(path: Path = DEFAULT_PER_TRAJECTORY_TOLERANCE_PATH) -> dict:
    """Per-trajectory tolerance (PR #3), derived by
    generate_per_trajectory_tolerance.py from >=5-repeat-run spread
    measurements (see test/paper_parity/measure_spread.py,
    test/paper_parity/README.md 'Per-trajectory tolerance').

    Only covers model keys/trajectories that were actually measured this
    session (currently M1, M3). Returns {} (never raises) if the file is
    absent -- callers MUST fall back to the single global `tolerance.json`
    for any (model_key, pkl_file) not present here, and that fallback is
    the documented, honest behaviour, not a silent gap: see
    `tolerance_for()`.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def tolerance_for(model_key: str, pkl_file: str, global_tol: dict, per_traj: dict) -> tuple[dict, str]:
    """Return (tolerance_dict, source) for one trajectory.

    Prefers the per-trajectory tolerance (tighter for stable trajectories,
    wider for measured-chaotic ones) when this exact (model_key, pkl_file)
    was measured; otherwise falls back to the single global tolerance and
    says so explicitly (source == "global-fallback") so a gate run's output
    never silently claims per-trajectory precision it doesn't have.
    """
    model_entry = per_traj.get(model_key)
    if model_entry and pkl_file in model_entry:
        row = model_entry[pkl_file]
        return ({k: row[k] for k in NUMERIC_KEYS + COUNT_KEYS}, "per-trajectory")
    return (global_tol, "global-fallback")


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


def run_gate_for_model(model_key: str, tol: dict, work_root: Path, cuda_device=None,
                        per_traj: dict | None = None,
                        truncated_nsteps=None) -> tuple[bool, list, float]:
    """`truncated_nsteps`, when set, runs the PR #4 'quick' tier: a
    truncated-horizon rollout diffed against a baseline/tolerance derived
    at the SAME horizon (never the full-length baseline/tolerance -- a
    truncated run's own measured spread is not assumed to match the
    full-length one, see generate_truncated_tolerance.py). This is an
    ADDITIONAL fast-feedback signal; the default (`truncated_nsteps=None`)
    call is the full-length gate PRs are actually judged on, unchanged.
    """
    paths = model_paths(model_key)
    if paths["provenance"] != "published":
        print(f"\n{'!'*70}\nWARNING: model {model_key} baseline provenance is "
              f"{paths['provenance']!r} -- NOT a confirmed paper-parity oracle.\n"
              f"A gate PASS here does not certify agreement with the published paper "
              f"result for {model_key}. See NOTES_tier1.md.\n{'!'*70}\n")

    if truncated_nsteps is not None:
        baseline_path = HERE / f"baseline_{model_key}_truncated{truncated_nsteps}.json"
        gate_label = f"{model_key}_truncated{truncated_nsteps}"
    else:
        baseline_path = HERE / f"baseline_{model_key}.json"
        gate_label = model_key
    if not baseline_path.exists():
        hint = (f"extract_truncated_baseline.py --model {model_key} --nsteps {truncated_nsteps}"
                if truncated_nsteps is not None else f"extract_baselines.py --model {model_key}")
        raise FileNotFoundError(f"{baseline_path} not found. Run {hint} first.")
    with open(baseline_path) as f:
        baseline = json.load(f)

    output_dir = work_root / gate_label
    current_trajectories, elapsed = compute_current_metrics(
        model_key, output_dir, cuda_device=cuda_device, truncated_nsteps=truncated_nsteps)

    baseline_trajectories = baseline["trajectories"]
    if len(baseline_trajectories) != len(current_trajectories):
        raise RuntimeError(
            f"[{gate_label}] trajectory count mismatch: baseline has "
            f"{len(baseline_trajectories)}, current run produced {len(current_trajectories)}")

    per_traj = per_traj or {}
    tol_lookup_key = gate_label if truncated_nsteps is not None else model_key
    all_ok = True
    rows = []
    for b_traj, c_traj in zip(baseline_trajectories, current_trajectories):
        if b_traj["pkl_file"] != c_traj["pkl_file"]:
            raise RuntimeError(
                f"[{gate_label}] pkl ordering mismatch: baseline {b_traj['pkl_file']} "
                f"vs current {c_traj['pkl_file']}")
        this_tol, tol_source = tolerance_for(tol_lookup_key, b_traj["pkl_file"], tol, per_traj)
        diffs = diff_trajectory(b_traj, c_traj, this_tol)
        traj_ok = all(v[0] for v in diffs.values())
        all_ok = all_ok and traj_ok
        rows.append((b_traj["pkl_file"], traj_ok, diffs, tol_source))

    return all_ok, rows, elapsed


def print_table(model_key: str, rows):
    print(f"\n=== {model_key} paper-parity gate: per-trajectory results ===")
    header = f"{'traj':<16}{'status':<8}{'tol_source':<18}"
    for key in NUMERIC_KEYS + COUNT_KEYS:
        header += f"{key:<28}"
    print(header)
    for pkl_file, traj_ok, diffs, tol_source in rows:
        status = "PASS" if traj_ok else "FAIL"
        line = f"{pkl_file:<16}{status:<8}{tol_source:<18}"
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
    ap.add_argument("--per-trajectory-tolerance", type=Path,
                     default=DEFAULT_PER_TRAJECTORY_TOLERANCE_PATH,
                     help="Per-trajectory tolerance (PR #3); falls back to --tolerance for "
                          "any (model, trajectory) not covered. Pass a nonexistent path to "
                          "disable and use only the global tolerance.")
    ap.add_argument("--truncated-nsteps", type=int, default=None,
                     help="PR #4 'quick' tier: cap the rollout at this many autoregressive "
                          "steps instead of the full 826 (only M1 has a committed truncated "
                          "baseline/tolerance at N=100 as of this PR -- see NOTES_pr4.md). "
                          "ADDITIONAL fast-feedback signal only; never use this flag's result "
                          "in place of a full (no-flag) run for judging a merge.")
    args = ap.parse_args()

    if not gns_sample_available():
        print("gns-sample/ is not available (empty or missing) -- cannot run the paper-parity "
              "gate here. This is expected in CI; run locally with the symlink set up.")
        sys.exit(2)

    tol = load_tolerance(args.tolerance)
    per_traj = load_per_trajectory_tolerance(args.per_trajectory_tolerance)
    if args.truncated_nsteps is not None:
        # Quick tier: per-trajectory tolerance is REQUIRED, not optional --
        # the global tolerance.json was derived at the full 826-step
        # horizon and must never be borrowed here (see NOTES_pr4.md).
        truncated_tol_path = HERE / "per_trajectory_tolerance_truncated.json"
        truncated_per_traj = load_per_trajectory_tolerance(truncated_tol_path)
        if not truncated_per_traj:
            raise FileNotFoundError(
                f"{truncated_tol_path} not found or empty -- run "
                f"generate_truncated_tolerance.py first. Refusing to fall back to the "
                f"full-length tolerance.json for a truncated run.")
        per_traj = {**per_traj, **truncated_per_traj}
    keys = list(ALL_REGISTRY) if args.model == "all" else [args.model]

    overall_ok = True
    total_elapsed = 0.0
    for key in keys:
        ok, rows, elapsed = run_gate_for_model(key, tol, args.work_dir, cuda_device=args.cuda_device,
                                                per_traj=per_traj, truncated_nsteps=args.truncated_nsteps)
        total_elapsed += elapsed
        print_table(key, rows)
        print(f"[{key}] rollout wall-clock: {elapsed:.1f}s, gate: {'PASS' if ok else 'FAIL'}")
        overall_ok = overall_ok and ok

    print(f"\nTotal wall-clock across requested models: {total_elapsed:.1f}s")
    print(f"\nOVERALL GATE: {'PASS' if overall_ok else 'FAIL'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()

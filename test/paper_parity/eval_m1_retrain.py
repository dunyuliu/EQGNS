#!/usr/bin/env python3
"""Evaluate M1 retrain checkpoints on the FIXED-D1 test set.

Scores three arms on the SAME test set (D1_fixed/dataset/test.npz, 6
trajectories) so the comparison isolates the training-data effect:
  fixed_seed{0,1,2}  -- retrained on the bug-fixed D1_fixed dataset
  old_seed{0,1,2}    -- control, retrained on the original published dataset
  published          -- the original published M1 checkpoints, matched steps

Reuses test/paper_parity/gate.py's metrics()/valid_steps() (collapse guard,
var_ratio) verbatim -- do not reimplement. Mirrors gate.py's run_rollout()
pattern: tempdir with test.npz symlinked in, det_rollout.py current --mode=rollout.

Usage:
  python3 test/paper_parity/eval_m1_retrain.py fixed_seed0 --steps 100000 --cuda 2
  python3 test/paper_parity/eval_m1_retrain.py published --steps 100000,200000 --cuda 1
  python3 test/paper_parity/eval_m1_retrain.py old_seed1 --cuda 1 --quick   # fast proof-of-pipeline

Output: one JSON file per (label, step) with the per-trajectory metrics list
(same shape as gate.py's metrics() return), written to
eq_rupture_gns_data/m1_retrain/eval_results/<label>_step<step>.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import HERE, REPO, QUICK_STEPS, metrics, pkls_in, write_quick_npz  # noqa: E402

M1_RETRAIN = Path("/home/utig5/dliu/eq_rupture_gns_data/m1_retrain")
TEST_NPZ = Path("/home/utig5/dliu/eq_rupture_gns_data/D1_fixed/dataset/test.npz")
RESULTS_DIR = M1_RETRAIN / "eval_results"
# REPO (from gate.py) is THIS checkout's root -- used as the det_rollout.py
# subprocess cwd so it runs THIS worktree's meshnet/train.py. gns-sample is
# separate: gitignored external data that lives only in the main repo
# checkout, not in git worktrees (same reason gate.py's REGEN_DATA is an
# absolute, env-overridable path rather than derived from REPO). Override
# with EQGNS_MAIN_REPO on another machine/checkout.
MAIN_REPO = Path(os.environ.get("EQGNS_MAIN_REPO", "/home/utig5/dliu/eq_rupture_gns"))
PUBLISHED_MODEL_DIR = MAIN_REPO / "gns-sample" / "case3.200m.homo.a.Vw" / "models.nmp10.cotopaxi"
STEPS = [100000, 200000, 300000, 400000, 500000]
ARMS = ("fixed", "old")


def model_dir_for(label):
    if label == "published":
        return PUBLISHED_MODEL_DIR
    arm, _, seed = label.partition("_seed")
    if arm not in ARMS or not seed.isdigit():
        raise ValueError(f"unknown label {label!r}, expected 'published' or '(fixed|old)_seed<N>'")
    return M1_RETRAIN / f"{arm}_seed{seed}" / "models"


def result_path(label, step):
    return RESULTS_DIR / f"{label}_step{step}.json"


def run_rollout(label, step, cuda, quick=False, quick_traj=0):
    """Deterministic rollout of the given checkpoint on the fixed-D1 test
    set; returns per-trajectory metrics. Mirrors gate.py's run_rollout()."""
    model_dir = model_dir_for(label)
    with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as data_dir:
        if quick:
            write_quick_npz(TEST_NPZ, Path(data_dir) / "test.npz", quick_traj)
        else:
            (Path(data_dir) / "test.npz").symlink_to(TEST_NPZ)
        cmd = [sys.executable, str(HERE / "det_rollout.py"), "current", "--mode=rollout",
               f"--data_path={data_dir}/", f"--model_path={model_dir}/",
               f"--output_path={out_dir}/", f"--model_file=model-{step}.pt",
               f"--train_state_file=train_state-{step}.pt", f"--cuda_device_number={cuda}"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"[{label} step{step}] rollout failed:\n{r.stderr[-3000:]}")
        out = []
        for p in pkls_in(out_dir):
            with open(p, "rb") as f:
                pkl = pickle.load(f)
            out.append(metrics(pkl))
        return out


def evaluate(label, step, cuda, quick=False, force=False):
    """Run one (label, step), write its JSON, return the rows. Skips (loads
    existing JSON) if the result already exists and force=False -- this is
    what lets the background queue resume without redoing finished work."""
    dst = result_path(label, step)
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    rows = run_rollout(label, step, cuda, quick=quick)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="'published' or '(fixed|old)_seed<N>'")
    ap.add_argument("--steps", default=",".join(str(s) for s in STEPS),
                     help="comma-separated checkpoint steps (default: all 5)")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--quick", action="store_true",
                     help=f"one trajectory, first {QUICK_STEPS} steps -- proof-of-pipeline, not a full score")
    ap.add_argument("--quick-traj", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recompute even if the JSON already exists")
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")]
    for step in steps:
        rows = evaluate(a.label, step, a.cuda, quick=a.quick, force=a.force)
        med = lambda k: float(__import__("numpy").median([r[k] for r in rows]))
        print(f"{a.label} step{step}: n_traj={len(rows)} "
              f"median mse_vx={med('mse_vx'):.4g} var_ratio={med('var_ratio'):.4g} "
              f"rt_rmse={med('rt_rmse'):.4g} "
              f"collapsed={sum(r['collapsed'] for r in rows)}/{len(rows)} "
              f"-> {result_path(a.label, step)}")


if __name__ == "__main__":
    main()

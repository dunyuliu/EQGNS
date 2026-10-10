#!/usr/bin/env python3
"""Metrics driver for the M3 batch-size / LR-scaling sweep
(docs/dev/M3_BATCH_LR_SWEEP_DESIGN.md, section 4.3).

Reuses, never redefines (PROJECT_RULES.md rule 7):
  tests/paper_parity/gate.py::metrics()            -- arm rollout vs EQdyna test truth
                                                      (mse_vx, rt_rmse, missed, false, mw_error,
                                                      sr_rmse_vx/vy, final_slip_rmse, var_ratio)
  tests/paper_parity/measure_vs_published.py::compare_pair()
                                                   -- arm rollout vs an anchor rollout
                                                      (delta_rt_rmse_s, delta_mw, vx/vy_rmse_norm,
                                                      missed, false) over its fixed 755-step window
  measure_vs_published.py::fresh_raw_rollout()     -- deterministic rollout of a foreign checkpoint

Adds ONE metric the vs-anchor path lacked (design doc open question 4): final-slip RMSE
between arm and anchor, `final_slip_pair()` below -- per-node trapezoidal integral of the
speed magnitude over compare_pair()'s own window (init frame prepended, same construction
gate.metrics() uses for its vs-truth final_slip_rmse), RMSE across nodes, plus the same
number normalised by the anchor's peak final slip.

Adds ONE validity flag, `diverged` (per trajectory, every side): gate.metrics()'s
`collapsed` guard is low-variance-only, so a NaN/blown-up rollout would otherwise score
rt_rmse = 0.0 and pass. A trajectory is diverged if any predicted velocity is non-finite
or max|v_pred| exceeds DIVERGE_RATIO x max|v_truth| over the trajectory. Diverged rows are
written (with the flag set) but excluded from every mean this script prints; the
downstream ranking must exclude them too (filter on `diverged == 0`).

Modes
  score    score an existing directory of rollout_*.pkl files
             --case M3_D3 --arm NAME --pkl-dir DIR [--anchor G=DIR --anchor P=DIR ...] --out CSV
  rollout  roll an arm checkpoint out (deterministic, eager path) into --pkl-dir, then score
             --case M3_D3 --arm NAME --model-dir DIR --step N --cuda K --pkl-dir DIR [--anchor ...] --out CSV

Output CSV: one row per trajectory x side ('truth' or an anchor name) with provenance
columns (ckpt_step, git_sha, model_path) and the union of the metric columns above (blank
where a side has no such metric). Rewriting is IDEMPOTENT: rows whose key
(case, arm, ckpt_step, traj, side) already exists in --out are replaced, never duplicated.
Trajectories in gate.REGRESSION_EXCLUDE_TRAJ are written like any other row and flagged in
`excluded`; the printed per-case summary gives means with and without them and states n.
No thresholds, no pass/fail -- the design doc's floors are applied downstream.
"""
import argparse
import csv
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tests" / "paper_parity"))
import gate  # noqa: E402
import measure_vs_published as mvp  # noqa: E402

TRUTH_KEYS = ["mse_vx", "rt_rmse", "missed", "false", "mw_error", "sr_rmse_vx", "sr_rmse_vy",
              "final_slip_rmse", "var_ratio", "collapsed"]
ANCHOR_KEYS = ["delta_rt_rmse_s", "delta_mw", "vx_rmse_norm", "vy_rmse_norm", "missed", "false",
               "final_slip_rmse", "final_slip_rmse_norm", "note"]
KEY_COLUMNS = ["case", "arm", "ckpt_step", "traj", "side"]
PROVENANCE_COLUMNS = ["git_sha", "model_path"]
COLUMNS = KEY_COLUMNS + PROVENANCE_COLUMNS + ["excluded", "diverged", "diverge_ratio"] + sorted(set(TRUTH_KEYS) | set(ANCHOR_KEYS))
# Slip-rate amplitude ratio beyond which a rollout is called diverged. Truth peaks are O(1-10 m/s)
# on these test sets; a healthy rollout stays within a small factor of truth, a blown-up one
# grows by orders of magnitude before (or without) reaching NaN. 10x is deliberately loose: it
# is a divergence screen, not a quality metric -- quality is what the metrics below measure.
DIVERGE_RATIO = 10.0


def divergence(pkl):
    """(diverged: bool, ratio: float) -- non-finite anywhere, or max|v_pred| / max|v_truth| > DIVERGE_RATIO."""
    pred = np.asarray(pkl["predicted_rollout"], dtype=np.float64)
    truth = np.asarray(pkl["ground_truth_rollout"], dtype=np.float64)
    finite = bool(np.isfinite(pred).all())
    peak_truth = float(np.max(np.abs(truth))) if truth.size else 0.0
    peak_pred = float(np.nanmax(np.abs(pred))) if pred.size else 0.0
    ratio = (peak_pred / peak_truth) if peak_truth > 0 else float("inf")
    return (not finite) or (ratio > DIVERGE_RATIO), ratio


def final_slip_pair(pkl_a, pkl_b):
    """Final-slip RMSE (m) of side A vs side B over compare_pair()'s window, and the same
    normalised by B's peak final slip. Final slip per node = trapz(|v|, dt) with each side's
    own initial frame prepended -- the construction gate.metrics() uses vs truth."""
    pa = np.asarray(pkl_a["predicted_rollout"], dtype=np.float64)
    pb = np.asarray(pkl_b["predicted_rollout"], dtype=np.float64)
    n = min(pa.shape[0], pb.shape[0], mvp.WINDOW)
    sa = np.linalg.norm(np.concatenate([np.asarray(pkl_a["initial_velocities"], dtype=np.float64), pa[:n]]), axis=-1)
    sb = np.linalg.norm(np.concatenate([np.asarray(pkl_b["initial_velocities"], dtype=np.float64), pb[:n]]), axis=-1)
    fa = np.trapezoid(sa, dx=gate.DT, axis=0)
    fb = np.trapezoid(sb, dx=gate.DT, axis=0)
    rmse = float(np.sqrt(np.mean((fa - fb) ** 2)))
    peak = float(np.max(fb))
    return rmse, (rmse / peak if peak > 0 else None)


def load_pkls(d):
    out = []
    for p in gate.pkls_in(d):
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def score(case, arm, arm_pkls, anchors, ckpt_step="", git_sha="", model_path=""):
    rows = []
    base = {"case": case, "arm": arm, "ckpt_step": ckpt_step, "git_sha": git_sha, "model_path": model_path}
    for i, pkl in enumerate(arm_pkls):
        excl = (case in gate.REGRESSION_EXCLUDE_CASES) or ((case, i) in gate.REGRESSION_EXCLUDE_TRAJ)
        div, ratio = divergence(pkl)
        common = {**base, "traj": i, "excluded": int(excl), "diverged": int(div), "diverge_ratio": ratio}
        try:
            m = gate.metrics(pkl)
            rows.append({**common, "side": "truth", **{k: m[k] for k in TRUTH_KEYS}})
        except Exception as e:  # noqa: BLE001 -- a diverged rollout may break the metric code; keep the row
            rows.append({**common, "side": "truth", "note": f"metrics failed: {type(e).__name__}"})
        for name, pkls_b in anchors.items():
            if i >= len(pkls_b):
                rows.append({**common, "side": name, "note": f"anchor {name} has only {len(pkls_b)} trajectories"})
                continue
            try:
                c = mvp.compare_pair(pkl, pkls_b[i])
                fs, fs_norm = final_slip_pair(pkl, pkls_b[i])
                rows.append({**common, "side": name, **{k: c[k] for k in ANCHOR_KEYS if k in c},
                             "final_slip_rmse": fs, "final_slip_rmse_norm": fs_norm})
            except Exception as e:  # noqa: BLE001
                rows.append({**common, "side": name, "note": f"compare failed: {type(e).__name__}"})
    return rows


def rollout_arm(case, model_dir, step, cuda, pkl_dir):
    """Deterministic eager rollout of model-<step>.pt in model_dir via
    measure_vs_published.fresh_raw_rollout(), which addresses the checkpoint by the
    case's PUBLISHED step: a temp dir of symlinks renames the arm's files to that step
    without copying or editing anything."""
    model_dir = Path(model_dir)
    pub_step = gate.CASES[case][2][1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / f"model-{pub_step}.pt").symlink_to(model_dir / f"model-{step}.pt")
        ts = model_dir / f"train_state-{step}.pt"
        if ts.exists():
            (tmp / f"train_state-{pub_step}.pt").symlink_to(ts)
        (tmp / "config.json").symlink_to(model_dir / "config.json")
        pkls = mvp.fresh_raw_rollout(case, cuda, code="current", model_dir=tmp)
    os.makedirs(pkl_dir, exist_ok=True)
    for i, p in enumerate(pkls):
        with open(Path(pkl_dir) / f"rollout_{i}.pkl", "wb") as f:
            pickle.dump(p, f)
    return pkls


def summary(rows):
    lines = []
    n_div = len({r["traj"] for r in rows if r.get("diverged")})
    if n_div:
        lines.append(f"  DIVERGED trajectories (excluded from every mean below): {n_div} -> "
                     f"{sorted({r['traj'] for r in rows if r.get('diverged')})}")
    for side in sorted({r["side"] for r in rows}, key=lambda s: (s != "truth", s)):
        sub = [r for r in rows if r["side"] == side and not r.get("diverged")]
        keys = TRUTH_KEYS if side == "truth" else ANCHOR_KEYS
        for k in keys:
            vals = [(r[k], r["excluded"]) for r in sub if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
            if not vals:
                continue
            allv = np.array([v for v, _ in vals], dtype=float)
            inc = np.array([v for v, e in vals if not e], dtype=float)
            lines.append(f"  {side:>6s} {k:<21s} mean={np.mean(inc):.4g} (n={len(inc)}, excl. flagged)"
                         f"  all={np.mean(allv):.4g} (n={len(allv)})")
    return "\n".join(lines)


def write_idempotent(out, rows):
    """Replace rows with the same (case, arm, ckpt_step, traj, side) key; keep everything else."""
    def key(r):
        return tuple(str(r.get(k, "")) for k in KEY_COLUMNS)
    existing = []
    if os.path.exists(out):
        with open(out, newline="") as f:
            existing = list(csv.DictReader(f))
    new_keys = {key(r) for r in rows}
    kept = [r for r in existing if key(r) not in new_keys]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in kept + rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    os.replace(tmp, out)
    return len(existing) - len(kept)


def repo_git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["score", "rollout"])
    ap.add_argument("--case", required=True, choices=list(gate.CASES))
    ap.add_argument("--arm", required=True, help="arm label written to the CSV, e.g. A0a_b8_s1")
    ap.add_argument("--pkl-dir", required=True, help="rollout_*.pkl directory (input for score, output for rollout)")
    ap.add_argument("--model-dir"), ap.add_argument("--step", type=int), ap.add_argument("--cuda")
    ap.add_argument("--anchor", action="append", default=[], metavar="NAME=DIR",
                    help="anchor rollout_*.pkl directory; repeatable (e.g. G=<GH200 det rollout of the published ckpt>, P=<shipped published rollouts>)")
    ap.add_argument("--git-sha", default=None, help="code revision recorded per row; default: this checkout's HEAD (required if not a git checkout)")
    ap.add_argument("--out", required=True, help="CSV; rows for the same (case, arm, ckpt_step, traj, side) are replaced, not appended")
    a = ap.parse_args()

    git_sha = a.git_sha or repo_git_sha()
    if not git_sha:
        sys.exit("--git-sha required: this checkout has no git HEAD to record")
    anchors = {}
    for spec in a.anchor:
        name, d = spec.split("=", 1)
        anchors[name] = load_pkls(d)
    if a.mode == "rollout":
        if not (a.model_dir and a.step is not None and a.cuda is not None):
            sys.exit("rollout mode needs --model-dir, --step and --cuda")
        arm_pkls = rollout_arm(a.case, a.model_dir, a.step, a.cuda, a.pkl_dir)
    else:
        arm_pkls = load_pkls(a.pkl_dir)
    if not arm_pkls:
        sys.exit(f"no rollout_*.pkl in {a.pkl_dir}")

    rows = score(a.case, a.arm, arm_pkls, anchors, ckpt_step=("" if a.step is None else a.step),
                 git_sha=git_sha, model_path=(os.path.abspath(a.model_dir) if a.model_dir else ""))
    replaced = write_idempotent(a.out, rows)
    print(f"{a.case} arm={a.arm} step={a.step if a.step is not None else '-'}: {len(arm_pkls)} trajectories, "
          f"{len(rows)} rows -> {a.out}" + (f" (replaced {replaced} existing rows)" if replaced else ""))
    print(summary(rows))


if __name__ == "__main__":
    main()

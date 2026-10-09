#!/usr/bin/env python3
"""Paper-parity gate: re-run the published GNS checkpoints with the CURRENT
meshnet code and compare against the published rollouts (Liu & Becker 2025,
JGR Solid Earth, doi:10.1029/2025JB031981).

Checkpoints are Zenodo-verified: each model file is byte-identical (CRC32) to
the archive at doi:10.5281/zenodo.17095311 (M1/M2 model-3000000.pt,
M3 model-2700000.pt).

  gate.py run [CASE ...]        current code == published code (deterministic)
  gate.py reference [CASE ...]  rebuild reference.json from meshnet/train.py.published
  gate.py paper [CASE ...]      reference vs the published rollout files (info)
  gate.py extract [CASE ...]    rebuild published.json from the published rollouts
  gate.py falsify M1_D1         planted regression (weights x1.005) must FAIL
  gate.py quick                 ~3 min: one sensitive trajectory per model, 300 steps
  gate.py run/falsify --rollout-batch-size N   N>1: rollout_batched() path, gated at the
                                 looser REL_TOL_BATCHED instead of REL_TOL (see its docstring;
                                 N=1, the default, is bit-identical to omitting this flag)
  gate.py fast [CASE ...]       opt-in --rollout_fast path: full test sets within the FAST_* band
  (fast takes --precision; add --falsify for its planted regression, which must FAIL)
  (add --quick to reference/falsify for the quick-tier variant)

All rollouts run in torch deterministic mode (det_rollout.py), which makes
reruns bit-identical; GPU nondeterminism otherwise swings chaotic
trajectories by >100% in MSE. Metrics per trajectory: rollout MSE of vx,
rupture-time RMSE / missed / false node counts at 0.1 m/s
(scripts/utils/plot.rupture.dynamics.py), Mw error, slip-rate RMSE (vx and
vy components), and final-slip RMSE, over the unpadded steps (see
valid_steps). Mw / slip-rate / final-slip reuse the same seismic-moment and
rupture-analysis conventions as measure_vs_published.py (PROJECT_RULES.md
rule 7): `compute_moment()` loaded by path from
scripts/utils/plot.rupture.dynamics.py (not importable as a dotted module
name), `Mw = (2/3) * (log10(moment) - 9.1)`, and the window is always
`valid_steps(pkl)` (or `n_override` for TRUNCATE_TO_PUBLISHED cases) --
never a hardcoded step count, so each case's own per-case window (e.g.
M1_D1's 755 steps / 0-754) is honored automatically instead of assumed.
"""
import argparse
import importlib.util
import os
import queue
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "gns-sample"
PUBLISHED = HERE / "published.json"   # metrics of the published rollout files
REFERENCE = HERE / "reference.json"   # metrics of train.py.published, deterministic

# scripts/utils/plot.rupture.dynamics.py is not an importable dotted module
# name (literal dots in the filename) -- loaded by path, same technique as
# measure_vs_published.py's own `prd` (PROJECT_RULES.md rule 7: reuse
# compute_moment()/the Mw formula, don't redefine them; this is a second,
# independent module-load of the same file, not a redefinition -- gate.py
# cannot import measure_vs_published.py here, since that module imports
# gate.py at its own top level and doing so the other way would be circular).
_PRD_PATH = REPO / "scripts" / "utils" / "plot.rupture.dynamics.py"
_prd_spec = importlib.util.spec_from_file_location("plot_rupture_dynamics_gate", _PRD_PATH)
prd = importlib.util.module_from_spec(_prd_spec)
_prd_spec.loader.exec_module(prd)  # only used for compute_moment(); no __main__ side effects

# Regenerated (bug-fixed) datasets that are too large / too actively-updated
# to live under gns-sample (published, read-only, fixture-sized) still need
# somewhere to live. This data is NOT part of any git checkout, so its
# location can't be derived from HERE/REPO (which varies per worktree) --
# the default below is this machine's fixed, absolute location. Override
# with EQGNS_REGEN_DATA on any other machine. Do NOT point CASES at
# gns-sample via ad-hoc symlinks -- those don't travel with the repo.
REGEN_DATA = Path(os.environ.get("EQGNS_REGEN_DATA", "/home/utig5/dliu/eq_rupture_gns_data"))
# case -> directory holding that case's regenerated dataset npz, overriding
# the normal DATA/ds/"dataset" lookup in run_rollout().
REGEN_DATASET_DIR = {
    "M1_large": REGEN_DATA / "M1_large" / "dataset",
}

DT = 0.0167777          # scripts/utils/plot.rupture.dynamics.py
THRESHOLD = 0.1         # m/s, SLIPRATE_THRESHOLD
UNREACHED = 1000.0
# quick tier: the trajectory per model most sensitive to perturbation, truncated
QUICK = {"M1_D1": 4, "M2_D3": 14, "M3_D3": 7}
QUICK_STEPS = 300
KEYS = ["mse_vx", "rt_rmse", "missed", "false", "mw_error", "sr_rmse_vx", "sr_rmse_vy", "final_slip_rmse"]
REL_TOL = 1e-4          # current vs published code: float reassociation only
# Batched rollout (`--rollout_batch_size` > 1, `rollout_batched()` in
# meshnet/train.py) concatenates multiple trajectories into one disjoint
# graph before PyTorch's aggr='add' message aggregation, changing summation
# order vs. the per-trajectory unbatched loop -- same math, different
# rounding, amplified by the 754-step autoregressive rollout on chaotic
# trajectories (PATHWAY_FORWARD.md board row `rollout-batched-oracle-gap`,
# diagnosed BENIGN FLOAT REASSOCIATION by code audit, not a code bug; owner
# decision `release-gate-decisions-pending` item (3), 2026-10-09: "accept a
# looser tolerance for the batched path only (default batch=1 path keeps
# 1e-4)"). The default/eager batch=1 path above is UNCHANGED.
#
# Measured 2026-10-09 (iris-vermeulen), batch=15, all 8 gated cases
# (M1_D1/M1_small/M1_large/M2_D2/M2_D3/M2_checkerboard/M3_D3/M3_D1hypo) vs
# `reference.json` (or `published_reference()` for M1_large): worst-case
# relative delta among trajectories NOT already excluded elsewhere as known
# chaotic bifurcations (`regression_ok()`'s REGRESSION_EXCLUDE_* -- M2_D3
# entirely, M3_D3 traj 7) was M3_D3 traj 8 at 0.0413 (mse_vx 0.857->0.816);
# M1_D1 traj 4 (the originally-diagnosed trajectory) measured 0.0309 (mse_vx
# 1.15026->1.18577, matching the board row's reported ~0.0355 absolute
# delta). REL_TOL_BATCHED = 2x that worst-case (0.0826), rounded up to the
# next power of ten -> 1e-1. This does NOT cover two outliers found during
# this same measurement pass, deliberately left UNCOVERED (not silently
# folded into a looser blanket number) because their magnitude is
# inconsistent with benign reassociation:
#   - M2_D3 (all trajectories, up to 155x relative delta on `missed`) and
#     M3_D3 traj 7 (0.737x) are the SAME pre-existing chaotic-bifurcation
#     cases already excluded from `regression_ok()` for an unrelated gate --
#     consistent with known behavior, not a new finding, but gate.py run's
#     compare() has no exclusion mechanism, so they legitimately still FAIL
#     at batch=15 even under REL_TOL_BATCHED. Reported, not gated around.
#   - M2_checkerboard traj 1 (8.82x, mse_vx 0.568->9.39) is a NEW finding:
#     its own eager-vs-eager noise floor (independently measured the same
#     session) is <1% (0.562-0.568 across two runs), so this is NOT ordinary
#     chaotic/eager noise and does not fit the benign-reassociation story --
#     flagged for `lars-eriksson` (audit) / the owner, not folded into this
#     tolerance and not silently excluded.
# See tests/paper_parity/README.md for the full measurement table.
REL_TOL_BATCHED = 1e-1
COLLAPSE_TOL = 0.5      # var(pred)/var(gt) below this: flat/degenerate forecast (owner-set, do not tune)
# fast tier (--rollout_fast): rounding differs from the default path and the chaotic rollout
# amplifies it, so no per-trajectory match; per case, vs reference.json, the mean rt_rmse, the
# missed+false count and the mean mse_vx may not grow past these factors. Set 2026-10-08 from the
# run-to-run spread of two nondeterministic eager rollouts (mean rt_rmse <=1.00x, missed+false
# <=1.52x, mse_vx <=1.15x the reference, over M1_D1/M2_D3/M3_D3; docs/user/rollout_and_analysis.md).
FAST_CASES = ["M1_D1", "M2_D3", "M3_D3"]
FAST_TOL = {"rt_rmse": 1.05, "missed+false": 2.0, "mse_vx": 1.5}

# Fast-rollout-vs-reference-rollout regression gate (`gate.py regression`),
# owner-approved thresholds, PATHWAY_FORWARD.md `release-gate-decisions-pending`
# row (a), 2026-10-09: one uniform threshold set for every gated path
# (eager/default, fast fp32, fast tf32) and every case. "default" tier is the
# owner's first approval; "tight" is the fallback the owner specified if the
# falsify acceptance check (weights x1.005 must FAIL) passes under "default"
# (too loose) -- per that decision, if "tight" also passes, that must be
# reported back to the owner, not silently accepted.
REGRESSION_TOL_DT = {"default": 4, "tight": 3}      # x DT
REGRESSION_TOL_MW = {"default": 0.03, "tight": 0.02}
# Owner-named exceptions (same decision): reported (never silently dropped
# from output), never gated on.
REGRESSION_EXCLUDE_CASES = {"M2_D3"}
REGRESSION_EXCLUDE_TRAJ = {("M3_D3", 7)}


def regression_ok(case, row, tier="default"):
    """Verdict for one measure_vs_published.compare_pair() row (side A =
    fast/eager rollout, side B = fresh deterministic reference rollout)
    under the owner-approved thresholds above.

    Returns True (pass), False (fail), or None (owner-excluded case/
    trajectory -- reported by the caller, never counted against the gate)."""
    if case in REGRESSION_EXCLUDE_CASES or (case, row["traj"]) in REGRESSION_EXCLUDE_TRAJ:
        return None
    rt_tol, mw_tol = REGRESSION_TOL_DT[tier] * DT, REGRESSION_TOL_MW[tier]
    if row["delta_rt_rmse_s"] > rt_tol:
        return False
    if row["delta_mw"] is not None and abs(row["delta_mw"]) > mw_tol:
        return False
    if row["missed"] + row["false"] > 0:
        return False
    return True

M1 = ("models.nmp10.cotopaxi", 3000000)
M2 = ("models.nmp10.cotopaxi", 3000000)
M3 = ("models.nmp10.lr3e-5.b8.cotopaxi.r1", 2700000)
D2_160 = "case4.200m.multi.stress.160scenarios.homo.a.Vw"

# case: (dataset dir, test npz, (model dir, step), published rollout dir)
CASES = {
    "M1_D1": ("case3.200m.homo.a.Vw", "test.npz", M1,
              "rollouts.nmp10.cotopaxi"),
    "M1_small": ("case3.200m.homo.a.Vw.others", "case3.200m.small.npz", M1,
                 "rollouts.nmp10.cotopaxi.small.D1.T_small"),
    # 40 km fault "H14.large" scenario, single 827-frame trajectory,
    # regenerated this session with a zero-tail padding bug fixed (827 real
    # frames, no padding). The committed published rollout was rolled out on
    # the OLD, shorter/padded npz, so it cannot be reproduced by the normal
    # reference.json mechanism -- see TRUNCATE_TO_PUBLISHED.
    "M1_large": ("case3.200m.homo.a.Vw.others", "case3.200m.large.npz", M1,
                 "rollouts.nmp10.cotopaxi.large.published"),
    "M2_D2": ("case4.200m.multi.stress.homo.a.Vw", "test.npz", M2,
              "rollouts.nmp10.cotopaxi.published"),
    "M2_D3": ("case4.200m.fractal.stress.homo.a.Vw", "test.npz", M2,
              "rollouts.nmp10.cotopaxi"),
    "M2_checkerboard": ("case4.200m.multi.asp.homo.a.Vw", "test.npz", M2,
                        "rollouts.nmp10.cotopaxi"),
    # this directory's test.npz is byte-identical to the D3 fractal test set
    "M3_D3": (D2_160, "test.npz", M3,
              "rollouts.nmp10.lr3e-5.b8.cotopaxi.r1.published"),
    # (.case3.others.test holds a byte-identical copy of this test set)
    "M3_D1hypo": (D2_160 + ".case3.test", "test.npz", M3,
                  "rollouts.nmp10.lr3e-5.b8.cotopaxi.r1"),
}


def rupture_time(sliprate):
    rt = np.full(sliprate.shape[1], UNREACHED)
    for it in range(sliprate.shape[0]):
        rt[(rt == UNREACHED) & (sliprate[it] > THRESHOLD)] = it * DT + 1.2
    return rt


def valid_steps(pkl):
    """Rollout steps before the dataset padding: the prepared test sets end
    each scenario with padded frames (zero velocity, invalid node_coords),
    which the published code feeds back into its per-step graph."""
    c = np.asarray(pkl["node_coords"])
    moved = np.abs(c - c[:1]).reshape(len(c), -1).max(1) > 0
    return int(np.argmax(moved)) if moved.any() else len(c) - 1


def metrics(pkl, n_override=None):
    """n_override: score only the first n steps instead of valid_steps(pkl) --
    used for cases gated against a published rollout shorter than the
    current one (see TRUNCATE_TO_PUBLISHED); all other call sites pass
    n_override=None and get exactly today's behaviour."""
    n = valid_steps(pkl) if n_override is None else n_override
    pred = np.asarray(pkl["predicted_rollout"], dtype=np.float64)[:n]
    gt = np.asarray(pkl["ground_truth_rollout"], dtype=np.float64)[:n]
    init = np.asarray(pkl["initial_velocities"], dtype=np.float64)
    speed_gt = np.linalg.norm(np.concatenate([init, gt]), axis=-1)
    speed_pr = np.linalg.norm(np.concatenate([init, pred]), axis=-1)
    rt_gt = rupture_time(speed_gt)
    rt_pr = rupture_time(speed_pr)
    hit_gt, hit_pr = rt_gt < UNREACHED, rt_pr < UNREACHED
    both = hit_gt & hit_pr
    # Collapse guard (owner, citing dynamo_gns Rule 22 cl.7-8): a model that
    # predicts a near-constant output (e.g. near-zero velocity everywhere)
    # can still score a deceptively low mse_vx if ground truth is also
    # mostly small/quiet -- MSE alone can't tell "tracking the signal" from
    # "collapsed to a low-variance constant". Reduction: population variance
    # per channel (vx, vy) over all (steps, nodes), then the WORSE (min) of
    # the two channel ratios -- mse_vx only ever looks at vx, so a channel-
    # mixed reduction here could hide a collapse in vy; taking the min means
    # collapse in either channel is caught. Where gt itself has ~zero
    # variance (degenerate ground truth, not a model failure), the ratio is
    # defined as 1.0 if pred also has ~zero variance, else left uncapped
    # (not treated as collapse).
    var_gt = np.var(gt, axis=(0, 1))
    var_pred = np.var(pred, axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ch = np.where(var_gt > 0, var_pred / np.where(var_gt > 0, var_gt, 1.0),
                             np.where(var_pred > 0, np.inf, 1.0))
    var_ratio = float(np.min(ratio_ch))

    # Mw error: seismic moment / moment-magnitude of this trajectory's own
    # ground truth vs its own prediction (same role as rt_rmse above -- a
    # per-trajectory accuracy number computed from gt/pred, then diffed
    # current-vs-reference by compare()). Convention: compute_moment()
    # (scripts/utils/plot.rupture.dynamics.py, shear_modulus=32e9,
    # slip_threshold=0.01 defaults) on the SPEED magnitude time series
    # (`speed_gt`/`speed_pr`, init frame prepended -- same series rt_gt/rt_pr
    # were built from, not a raw vx/vy component), Mw = (2/3)*(log10(M0)-9.1)
    # (both reused verbatim from measure_vs_published.py's compare_pair(),
    # PROJECT_RULES.md rule 7). A non-positive moment means compute_moment()
    # found no node with cumulative slip above its threshold -- i.e. this
    # trajectory never actually ruptured, which is a data problem (every
    # CASES entry is a real earthquake rollout and should rupture), not a
    # tolerance question -- raised loudly rather than papered over with a
    # sentinel value nobody asked for.
    node_coords0 = np.asarray(pkl["node_coords"])[0]
    triang = mtri.Triangulation(node_coords0[:, 0] / 1e3, node_coords0[:, 1] / 1e3)
    moment_gt = prd.compute_moment(speed_gt, triang, DT)
    moment_pr = prd.compute_moment(speed_pr, triang, DT)
    if moment_gt <= 0 or moment_pr <= 0:
        raise RuntimeError(
            f"compute_moment() returned a non-positive seismic moment (gt={moment_gt:.3g}, "
            f"pred={moment_pr:.3g}) -- no node crossed the slip_threshold, so Mw is undefined; "
            "this is a genuinely degenerate trajectory, not a tolerance issue")
    mw_gt = (2.0 / 3.0) * (np.log10(moment_gt) - 9.1)
    mw_pr = (2.0 / 3.0) * (np.log10(moment_pr) - 9.1)

    # Slip-rate RMSE, both velocity components, over the same valid window
    # as everything else above (n / n_override) -- NOT a hardcoded 0-754:
    # that window is simply what valid_steps() returns for M1_D1 and its
    # 755-step cohort (see module docstring / valid_steps()).
    sr_rmse_vx = float(np.sqrt(np.mean((pred[..., 0] - gt[..., 0]) ** 2)))
    sr_rmse_vy = float(np.sqrt(np.mean((pred[..., 1] - gt[..., 1]) ** 2)))

    # Final-slip RMSE: per-node cumulative slip (trapezoidal integral of the
    # speed magnitude over the full init+window time series, same
    # integration compute_moment() itself does internally for its own
    # cumulative_slip), RMSE across nodes between prediction and ground
    # truth.
    final_slip_gt = np.trapezoid(speed_gt, dx=DT, axis=0)
    final_slip_pr = np.trapezoid(speed_pr, dx=DT, axis=0)
    final_slip_rmse = float(np.sqrt(np.mean((final_slip_pr - final_slip_gt) ** 2)))

    return {
        "mse_vx": float(np.mean((pred[..., 0] - gt[..., 0]) ** 2)),
        "rt_rmse": float(np.sqrt(np.mean((rt_gt[both] - rt_pr[both]) ** 2))) if both.any() else 0.0,
        "missed": int(np.sum(hit_gt & ~hit_pr)),
        "false": int(np.sum(~hit_gt & hit_pr)),
        "mw_error": float(abs(mw_pr - mw_gt)),
        "sr_rmse_vx": sr_rmse_vx,
        "sr_rmse_vy": sr_rmse_vy,
        "final_slip_rmse": final_slip_rmse,
        "var_ratio": var_ratio,
        "collapsed": bool(var_ratio < COLLAPSE_TOL),
    }


def pkls_in(d):
    return sorted(Path(d).glob("rollout_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))


def published_dir(case):
    ds, _, (_, step), rdir = CASES[case]
    return DATA / ds / rdir / f"model-{step}.pt"


def write_quick_npz(src, dst, traj):
    """One trajectory, first QUICK_STEPS rollout steps."""
    data = np.load(src, allow_pickle=True)
    t = data[list(data.keys())[traj]].item()
    np.savez(dst, trajectory0={k: np.asarray(v)[:QUICK_STEPS + 1] for k, v in t.items()})


def run_rollout(case, out_dir, cuda, model_dir=None, code="current", quick=False, n_overrides=None,
                extra=()):
    """Deterministic rollout of one case with `code`; return per-trajectory metrics.

    n_overrides: optional per-trajectory step-count list (same order as
    pkls_in(out_dir)) passed to metrics() as n_override; None (default) for
    every case except TRUNCATE_TO_PUBLISHED. extra: further train.py flags."""
    ds, npz, (mdir, step), _ = CASES[case]
    model_dir = model_dir or DATA / ds / mdir
    dataset_dir = REGEN_DATASET_DIR.get(case, DATA / ds / "dataset")
    with tempfile.TemporaryDirectory() as data_dir:  # meshnet reads <data_path>/test.npz
        if quick:
            write_quick_npz(dataset_dir / npz, Path(data_dir) / "test.npz", QUICK[case])
        else:
            (Path(data_dir) / "test.npz").symlink_to(dataset_dir / npz)
        cmd = [sys.executable, str(HERE / "det_rollout.py"), code, "--mode=rollout",
               f"--data_path={data_dir}/", f"--model_path={model_dir}/",
               f"--output_path={out_dir}/", f"--model_file=model-{step}.pt",
               f"--train_state_file=train_state-{step}.pt", f"--cuda_device_number={cuda}", *extra]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"[{case}] rollout failed:\n{r.stderr[-3000:]}")
    out = []
    for i, p in enumerate(pkls_in(out_dir)):
        with open(p, "rb") as f:
            pkl = pickle.load(f)
        out.append(metrics(pkl, n_overrides[i] if n_overrides is not None else None))
    return out


def fresh_rollout(case, cuda, model_dir=None, code="current", quick=False, n_overrides=None, extra=()):
    with tempfile.TemporaryDirectory() as out:
        return run_rollout(case, out, cuda, model_dir, code, quick, n_overrides, extra)


def ref_key(case, quick):
    return f"{case}@quick" if quick else case


# Cases gated directly against a committed published-rollout directory,
# truncated to that rollout's own valid (unpadded) steps, instead of against
# reference.json. M1_large's npz was regenerated to fix a zero-tail padding
# bug and now legitimately rolls out further than the committed published
# rollout has ground truth for; there is no oracle for those extra steps, so
# both sides of the comparison are restricted to the published rollout's
# original (shorter) window.
TRUNCATE_TO_PUBLISHED = {"M1_large"}


def published_reference(case):
    """Ground truth for TRUNCATE_TO_PUBLISHED cases: metrics of the committed
    published rollout pkls (each truncated to its own valid_steps), plus the
    per-trajectory step counts to also apply to a fresh current-code rollout
    so both sides are scored over the same window."""
    pkls = []
    for p in pkls_in(published_dir(case)):
        with open(p, "rb") as f:
            pkls.append(pickle.load(f))
    n_overrides = [valid_steps(pkl) for pkl in pkls]
    rows = [metrics(pkl, n) for pkl, n in zip(pkls, n_overrides)]
    return rows, n_overrides


def compare(case, current, ref, rel_tol=REL_TOL, quiet=False):
    ok = len(current) == len(ref)
    for i, (cur, r) in enumerate(zip(current, ref)):
        # `not (delta <= bound)` -- not `delta > bound` -- so a NaN in cur[k]
        # (e.g. a metric that silently produced NaN instead of raising) FAILS
        # the gate instead of comparing False against every bound and
        # passing (code discipline: a tolerance gate must fail on NaN).
        bad = [f"{k} {r[k]:.6g}->{cur[k]:.6g}" for k in KEYS
               if not (abs(cur[k] - r[k]) <= rel_tol * max(abs(r[k]), 1.0))]
        if cur.get("collapsed"):
            bad.append(f"collapsed var_ratio={cur['var_ratio']:.3g} < {COLLAPSE_TOL}")
        ok &= not bad
        if bad or not quiet:
            print(f"  [{case}] traj {i}: {'PASS' if not bad else 'FAIL ' + '; '.join(bad)}"
                  f" (var_ratio={cur.get('var_ratio', float('nan')):.3g})")
    return ok


def aggregate(rows):
    return {"rt_rmse": float(np.mean([r["rt_rmse"] for r in rows])),
            "missed+false": sum(r["missed"] + r["false"] for r in rows),
            "mse_vx": float(np.mean([r["mse_vx"] for r in rows]))}


def compare_band(case, current, ref):
    """Fast tier: case aggregates within FAST_TOL of the reference, nothing collapsed."""
    cur, r = aggregate(current), aggregate(ref)
    bad = [f"{k} {r[k]:.4g}->{cur[k]:.4g} (> x{tol})" for k, tol in FAST_TOL.items()
           if cur[k] > tol * r[k] + (1 if k == "missed+false" else 0)]
    bad += [f"traj {i} collapsed" for i, row in enumerate(current) if row["collapsed"]]
    bad += [] if len(current) == len(ref) else [f"{len(current)} trajectories, reference {len(ref)}"]
    print(f"  [{case}] " + "  ".join(f"{k} {cur[k]:.4g}/{r[k]:.4g}" for k in FAST_TOL)
          + f"  {'PASS' if not bad else 'FAIL ' + '; '.join(bad)}")
    return not bad


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=1) + "\n")


def parallel(fn, cases, gpus):
    """Run fn(case, gpu) for each case, one case per GPU at a time."""
    free = queue.Queue()
    for g in gpus:
        free.put(g)

    def task(case):
        g = free.get()
        try:
            return fn(case, g)
        finally:
            free.put(g)
    with ThreadPoolExecutor(len(gpus)) as ex:
        return dict(zip(cases, ex.map(task, cases)))


def cmd_run(cases, gpus, quick=False, batch_size=1):
    """batch_size=1 (default): unbatched rollout() path, gated at REL_TOL
    (unchanged). batch_size>1: rollout_batched() path (--rollout_batch_size),
    gated at the looser REL_TOL_BATCHED (see its definition above) --
    PATHWAY_FORWARD.md `rollout-batched-oracle-gap`."""
    extra = (f"--rollout_batch_size={batch_size}",) if batch_size > 1 else ()
    rel_tol = REL_TOL_BATCHED if batch_size > 1 else REL_TOL
    reference = load(REFERENCE)
    truncated = {c: published_reference(c) for c in cases if c in TRUNCATE_TO_PUBLISHED}
    t0 = time.time()
    rollouts = parallel(
        lambda c, g: fresh_rollout(c, g, quick=quick,
                                    n_overrides=truncated[c][1] if c in truncated else None,
                                    extra=extra),
        cases, gpus)
    print(f"rollouts: {len(cases)} cases on GPUs {gpus} in {time.time() - t0:.0f} s"
          + (f" (batch_size={batch_size}, rel_tol={rel_tol})" if batch_size > 1 else ""))
    results = {}
    for c in cases:
        ref_rows = truncated[c][0] if c in truncated else reference[ref_key(c, quick)]
        results[c] = compare(c, rollouts[c], ref_rows, rel_tol=rel_tol)
    for c, ok in results.items():
        print(f"{c}: {'PASS' if ok else 'FAIL'}")
    return all(results.values())


def cmd_reference(cases, gpus, quick=False):
    reference = load(REFERENCE)
    rollouts = parallel(lambda c, g: fresh_rollout(c, g, code="published", quick=quick), cases, gpus)
    for c in cases:
        reference[ref_key(c, quick)] = rollouts[c]
        print(f"{ref_key(c, quick)}: {len(rollouts[c])} trajectories")
    save(REFERENCE, reference)


def cmd_extract(cases):
    published = load(PUBLISHED)
    for c in cases:
        rows = []
        for p in pkls_in(published_dir(c)):
            with open(p, "rb") as f:
                rows.append(metrics(pickle.load(f)))
        published[c] = rows
        print(f"{c}: {len(rows)} trajectories")
    save(PUBLISHED, published)


def cmd_paper(cases):
    """Informative: deterministic published-code rollout vs the published files."""
    reference, published = load(REFERENCE), load(PUBLISHED)
    print(f"{'case':18s} {'median mse_vx ref/pub':>24s} {'median rt_rmse ref/pub':>24s}")
    for c in cases:
        med = lambda rows, k: float(np.median([r[k] for r in rows]))
        print(f"{c:18s} {med(reference[c], 'mse_vx'):11.4g}/{med(published[c], 'mse_vx'):<11.4g}"
              f" {med(reference[c], 'rt_rmse'):11.4g}/{med(published[c], 'rt_rmse'):<11.4g}")


def scaled_model(case, tmp, scale):
    """Copy the case's checkpoint into tmp with every weight scaled by `scale`."""
    ds, _, (mdir, step), _ = CASES[case]
    src = DATA / ds / mdir
    shutil.copy(src / "config.json", tmp)
    shutil.copy(src / f"train_state-{step}.pt", tmp)
    ckpt = torch.load(src / f"model-{step}.pt", map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    for k, v in state.items():
        if torch.is_floating_point(v):
            state[k] = v * scale
    torch.save(ckpt, tmp / f"model-{step}.pt")
    return tmp


def cmd_falsify(case, cuda, scale=1.005, quick=False, batch_size=1):
    """Scale every weight by `scale`; the gate must FAIL on the result.
    batch_size>1: acceptance check for REL_TOL_BATCHED (see its docstring) --
    the planted regression must still FAIL under the looser tolerance."""
    extra = (f"--rollout_batch_size={batch_size}",) if batch_size > 1 else ()
    rel_tol = REL_TOL_BATCHED if batch_size > 1 else REL_TOL
    if case in TRUNCATE_TO_PUBLISHED:
        ref_rows, n_overrides = published_reference(case)
    else:
        ref_rows, n_overrides = load(REFERENCE)[ref_key(case, quick)], None
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = scaled_model(case, Path(tmp), scale)
        current = fresh_rollout(case, cuda, model_dir, quick=quick, n_overrides=n_overrides, extra=extra)
    caught = not compare(case, current, ref_rows, rel_tol=rel_tol, quiet=True)
    print(f"planted regression (weights x{scale}{f', batch_size={batch_size}' if batch_size > 1 else ''}) "
          f"{'CAUGHT' if caught else 'MISSED'}")
    return caught


def cmd_fast(cases, gpus, precision, falsify=False, scale=1.005):
    """--rollout_fast on the full test sets, judged by compare_band; with falsify, on weights
    scaled by `scale`, where the band must FAIL (else it is too loose to mean anything)."""
    reference = load(REFERENCE)
    extra = (f"--rollout_fast={precision}", "--rollout_batch_size=64")

    def roll(case, gpu):
        if not falsify:
            return fresh_rollout(case, gpu, extra=extra)
        with tempfile.TemporaryDirectory() as tmp:
            return fresh_rollout(case, gpu, scaled_model(case, Path(tmp), scale), extra=extra)
    t0 = time.time()
    rollouts = parallel(roll, cases, gpus)
    print(f"fast {precision}{f' falsify x{scale}' if falsify else ''}: {len(cases)} cases in {time.time() - t0:.0f} s")
    ok = [compare_band(c, rollouts[c], reference[c]) for c in cases]
    if falsify:
        print(f"planted regression (weights x{scale}) {'CAUGHT' if not all(ok) else 'MISSED'}")
        return not all(ok)
    return all(ok)


def cmd_regression(cases, cuda, precision, falsify=False, scale=1.005, tier="default"):
    """The fast-rollout-vs-reference-rollout gate (owner decision (a), see
    REGRESSION_TOL_* above): per trajectory, current code + --rollout_fast
    (or, with falsify, a weights x`scale` copy of it) vs a freshly generated
    deterministic rollout of the PUBLISHED code (same construction as
    `cmd_reference`'s reference.json, just not cached -- the per-trajectory
    raw arrays this gate needs aren't in reference.json, only their
    aggregated gate.metrics() summaries). Reuses measure_vs_published.py's
    compare_pair()/fresh_raw_rollout() for the actual RT/Mw/missed/false math
    rather than redefining it (PROJECT_RULES.md rule 7); imported locally to
    avoid the module-load circular import (measure_vs_published imports this
    module at its own top level)."""
    import measure_vs_published as mvp
    extra = (f"--rollout_fast={precision}", "--rollout_batch_size=64")
    all_ok = True
    for case in cases:
        ref_pkls = mvp.fresh_raw_rollout(case, cuda, code="published")
        if falsify:
            with tempfile.TemporaryDirectory() as tmp:
                model_dir = scaled_model(case, Path(tmp), scale)
                fast_pkls = mvp.fresh_raw_rollout(case, cuda, code="current", extra=extra, model_dir=model_dir)
        else:
            fast_pkls = mvp.fresh_raw_rollout(case, cuda, code="current", extra=extra)
        rows = mvp.rows_for(f"regression_{precision}", case, fast_pkls, ref_pkls)
        case_ok = True
        for r in rows:
            verdict = regression_ok(case, r, tier)
            tag = "EXCLUDED(reported)" if verdict is None else ("PASS" if verdict else "FAIL")
            print(f"  [{case}] traj {r['traj']}: {tag}  dRT={r['delta_rt_rmse_s']:.4g}s"
                  f" dMw={r['delta_mw']} missed={r['missed']} false={r['false']}")
            if verdict is False:
                case_ok = False
        all_ok &= case_ok
    if falsify:
        print(f"planted regression (weights x{scale}, tier={tier}) {'CAUGHT' if not all_ok else 'MISSED'}")
        return not all_ok
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "quick", "reference", "paper", "extract", "falsify", "fast", "regression"])
    ap.add_argument("cases", nargs="*", help=f"default: all of {list(CASES)}")
    ap.add_argument("--cuda", required=True, help="GPU id(s), comma-separated; cases run in parallel (no default -- pick explicitly, GPU0 often hosts unrelated jobs on this box)")
    ap.add_argument("--quick", action="store_true", help="quick-tier variant")
    ap.add_argument("--precision", default="fp16", help="fast tier: --rollout_fast value")
    ap.add_argument("--falsify", action="store_true", help="fast/regression tier: planted regression must FAIL")
    ap.add_argument("--tier", default="default", choices=["default", "tight"],
                     help="regression tier: default=4dt/0.03, tight=3dt/0.02 (owner decision (a) fallback)")
    ap.add_argument("--rollout-batch-size", type=int, default=1, dest="rollout_batch_size",
                     help="run/falsify only: >1 drives rollout_batched() via --rollout_batch_size "
                          "and gates at REL_TOL_BATCHED instead of REL_TOL (default 1: unbatched "
                          "rollout(), REL_TOL, unchanged)")
    a = ap.parse_args()
    gpus = [int(g) for g in a.cuda.split(",")]
    quick = a.quick or a.command == "quick"
    if a.cases:
        cases = a.cases
    elif quick:
        cases = list(QUICK)
    elif a.command == "fast":
        cases = FAST_CASES
    elif a.command == "regression":
        cases = [c for c in CASES if c not in TRUNCATE_TO_PUBLISHED]
    elif a.command == "paper":
        # M1_large is in TRUNCATE_TO_PUBLISHED: it has no reference.json /
        # published.json entry (see published_reference() and its docstring),
        # so it is gated separately inside cmd_run, not here. Default the
        # "paper" tier to every other case; an explicit ask for M1_large is
        # handled below (message + skip), not a crash.
        cases = [c for c in CASES if c not in TRUNCATE_TO_PUBLISHED]
    else:
        cases = list(CASES)
    unknown = set(cases) - set(CASES)
    if unknown or (quick and set(cases) - set(QUICK)):
        sys.exit(f"unknown case(s) for this tier: {sorted(set(cases) - set(QUICK if quick else CASES))}")
    if a.command == "paper":
        truncated = [c for c in cases if c in TRUNCATE_TO_PUBLISHED]
        for c in truncated:
            print(f"[paper] {c}: no reference.json/published.json entry (gated separately via "
                  f"'gate.py run {c}', see TRUNCATE_TO_PUBLISHED) -- skipping")
        cases = [c for c in cases if c not in TRUNCATE_TO_PUBLISHED]
    if a.command in ("run", "quick"):
        sys.exit(0 if cmd_run(cases, gpus, quick, batch_size=a.rollout_batch_size) else 1)
    if a.command == "fast":
        sys.exit(0 if cmd_fast(cases, gpus, a.precision, a.falsify) else 1)
    if a.command == "regression":
        sys.exit(0 if cmd_regression(cases, gpus[0], a.precision, a.falsify, tier=a.tier) else 1)
    if a.command == "extract":
        cmd_extract(cases)
    if a.command == "reference":
        cmd_reference(cases, gpus, quick)
    if a.command == "paper":
        cmd_paper(cases)
    if a.command == "falsify":
        sys.exit(0 if all(cmd_falsify(c, gpus[0], quick=quick, batch_size=a.rollout_batch_size)
                           for c in cases) else 1)


if __name__ == "__main__":
    main()

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
  (add --quick to reference/falsify for the quick-tier variant)

All rollouts run in torch deterministic mode (det_rollout.py), which makes
reruns bit-identical; GPU nondeterminism otherwise swings chaotic
trajectories by >100% in MSE. Metrics per trajectory: rollout MSE of vx, and
rupture-time RMSE / missed / false node counts at 0.1 m/s
(utils/plot.rupture.dynamics.py), over the unpadded steps (see valid_steps).
"""
import argparse
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

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "gns-sample"
PUBLISHED = HERE / "published.json"   # metrics of the published rollout files
REFERENCE = HERE / "reference.json"   # metrics of train.py.published, deterministic

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

DT = 0.0167777          # utils/plot.rupture.dynamics.py
THRESHOLD = 0.1         # m/s, SLIPRATE_THRESHOLD
UNREACHED = 1000.0
# quick tier: the trajectory per model most sensitive to perturbation, truncated
QUICK = {"M1_D1": 4, "M2_D3": 14, "M3_D3": 7}
QUICK_STEPS = 300
KEYS = ["mse_vx", "rt_rmse", "missed", "false"]
REL_TOL = 1e-4          # current vs published code: float reassociation only
COLLAPSE_TOL = 0.5      # var(pred)/var(gt) below this: flat/degenerate forecast (owner-set, do not tune)

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
    rt_gt = rupture_time(np.linalg.norm(np.concatenate([init, gt]), axis=-1))
    rt_pr = rupture_time(np.linalg.norm(np.concatenate([init, pred]), axis=-1))
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
    return {
        "mse_vx": float(np.mean((pred[..., 0] - gt[..., 0]) ** 2)),
        "rt_rmse": float(np.sqrt(np.mean((rt_gt[both] - rt_pr[both]) ** 2))) if both.any() else 0.0,
        "missed": int(np.sum(hit_gt & ~hit_pr)),
        "false": int(np.sum(~hit_gt & hit_pr)),
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


def run_rollout(case, out_dir, cuda, model_dir=None, code="current", quick=False, n_overrides=None):
    """Deterministic rollout of one case with `code`; return per-trajectory metrics.

    n_overrides: optional per-trajectory step-count list (same order as
    pkls_in(out_dir)) passed to metrics() as n_override; None (default) for
    every case except TRUNCATE_TO_PUBLISHED."""
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
               f"--train_state_file=train_state-{step}.pt", f"--cuda_device_number={cuda}"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"[{case}] rollout failed:\n{r.stderr[-3000:]}")
    out = []
    for i, p in enumerate(pkls_in(out_dir)):
        with open(p, "rb") as f:
            pkl = pickle.load(f)
        out.append(metrics(pkl, n_overrides[i] if n_overrides is not None else None))
    return out


def fresh_rollout(case, cuda, model_dir=None, code="current", quick=False, n_overrides=None):
    with tempfile.TemporaryDirectory() as out:
        return run_rollout(case, out, cuda, model_dir, code, quick, n_overrides)


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
        bad = [f"{k} {r[k]:.6g}->{cur[k]:.6g}" for k in KEYS
               if abs(cur[k] - r[k]) > rel_tol * max(abs(r[k]), 1.0)]
        if cur.get("collapsed"):
            bad.append(f"collapsed var_ratio={cur['var_ratio']:.3g} < {COLLAPSE_TOL}")
        ok &= not bad
        if bad or not quiet:
            print(f"  [{case}] traj {i}: {'PASS' if not bad else 'FAIL ' + '; '.join(bad)}"
                  f" (var_ratio={cur.get('var_ratio', float('nan')):.3g})")
    return ok


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


def cmd_run(cases, gpus, quick=False):
    reference = load(REFERENCE)
    truncated = {c: published_reference(c) for c in cases if c in TRUNCATE_TO_PUBLISHED}
    t0 = time.time()
    rollouts = parallel(
        lambda c, g: fresh_rollout(c, g, quick=quick,
                                    n_overrides=truncated[c][1] if c in truncated else None),
        cases, gpus)
    print(f"rollouts: {len(cases)} cases on GPUs {gpus} in {time.time() - t0:.0f} s")
    results = {}
    for c in cases:
        ref_rows = truncated[c][0] if c in truncated else reference[ref_key(c, quick)]
        results[c] = compare(c, rollouts[c], ref_rows)
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


def cmd_falsify(case, cuda, scale=1.005, quick=False):
    """Scale every weight by `scale`; the gate must FAIL on the result."""
    ds, _, (mdir, step), _ = CASES[case]
    src = DATA / ds / mdir
    if case in TRUNCATE_TO_PUBLISHED:
        ref_rows, n_overrides = published_reference(case)
    else:
        ref_rows, n_overrides = load(REFERENCE)[ref_key(case, quick)], None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(src / "config.json", tmp)
        shutil.copy(src / f"train_state-{step}.pt", tmp)
        ckpt = torch.load(src / f"model-{step}.pt", map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        for k, v in state.items():
            if torch.is_floating_point(v):
                state[k] = v * scale
        torch.save(ckpt, tmp / f"model-{step}.pt")
        current = fresh_rollout(case, cuda, tmp, quick=quick, n_overrides=n_overrides)
    caught = not compare(case, current, ref_rows, quiet=True)
    print(f"planted regression (weights x{scale}) {'CAUGHT' if caught else 'MISSED'}")
    return caught


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "quick", "reference", "paper", "extract", "falsify"])
    ap.add_argument("cases", nargs="*", help=f"default: all of {list(CASES)}")
    ap.add_argument("--cuda", default="0", help="GPU id(s), comma-separated; cases run in parallel")
    ap.add_argument("--quick", action="store_true", help="quick-tier variant")
    a = ap.parse_args()
    gpus = [int(g) for g in a.cuda.split(",")]
    quick = a.quick or a.command == "quick"
    cases = a.cases or (list(QUICK) if quick else list(CASES))
    unknown = set(cases) - set(CASES)
    if unknown or (quick and set(cases) - set(QUICK)):
        sys.exit(f"unknown case(s) for this tier: {sorted(set(cases) - set(QUICK if quick else CASES))}")
    if a.command in ("run", "quick"):
        sys.exit(0 if cmd_run(cases, gpus, quick) else 1)
    if a.command == "extract":
        cmd_extract(cases)
    if a.command == "reference":
        cmd_reference(cases, gpus, quick)
    if a.command == "paper":
        cmd_paper(cases)
    if a.command == "falsify":
        sys.exit(0 if all(cmd_falsify(c, gpus[0], quick=quick) for c in cases) else 1)


if __name__ == "__main__":
    main()

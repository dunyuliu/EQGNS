#!/usr/bin/env python3
"""Direct per-trajectory diff of raw rollout predictions, two sides at a time.

This is NOT what `gate.py`'s `cmd_paper()`/`metrics()` compute today: those
compare each side's rollout against ITS OWN ground-truth trajectory and then
compare the two resulting metric summaries. This script instead diffs the
two sides' `predicted_rollout` arrays against each other directly, node for
node and step for step, for the PATHWAY_FORWARD.md `test-suite-overhaul`
board row's measurement table (owner: iris-vermeulen).

Constants reused verbatim from `tests/paper_parity/gate.py` -- NOT
redefined here, per PROJECT_RULES.md rule 7:
  - `gate.DT` = 0.0167777 s
  - `gate.rupture_time()` (threshold 0.1 m/s baked in as `gate.THRESHOLD`,
    `+1.2` s time offset baked in, `gate.UNREACHED` = 1000.0 sentinel)
  - `gate.CASES`, `gate.TRUNCATE_TO_PUBLISHED`, `gate.published_dir()`,
    `gate.pkls_in()`, `gate.HERE`, `gate.REPO`, `gate.DATA`,
    `gate.REGEN_DATASET_DIR`

Mw / moment convention reused from `scripts/utils/plot.rupture.dynamics.py`
(loaded by path below, since its filename is not an importable module name):
  - `compute_moment(velocity_data, triang, dt, shear_modulus=32e9,
    slip_threshold=0.01)`, `dt=gate.DT` (same `DT` constant that file itself
    defines at module level -- both are 0.0167777 s).
  - `velocity_data` is the per-node SPEED MAGNITUDE time series
    (`np.linalg.norm` of the 2-component velocity), NOT a raw vx/vy
    component: this matches that file's own `load_rollout_data()`, which
    builds `velocity_result["ground_truth"/"prediction"]` as
    `np.linalg.norm(np.concatenate([initial_velocities, rollout]), axis=-1)`
    (lines ~190-196), and its own call site `process_member()` ->
    `compute_moment(velocity_result[...], triang, DT)` (lines ~988-989) --
    i.e. the function has only ever been called with the norm, never a
    component.
  - `Mw = (2/3) * (log10(moment) - 9.1)` (same file, ~line 579-580).
  - `triang = matplotlib.tri.Triangulation(node_coords[0][:, 0] / 1e3,
    node_coords[0][:, 1] / 1e3)` (km; same file, ~line 202). Built once per
    trajectory from side A's `node_coords` -- both sides share the same
    mesh/test set for every comparison in this script, so A vs B's
    node_coords are not expected to differ; this is not re-verified per row.

Fixed window (owner instruction for this table, NOT `gate.valid_steps()`):
steps 0-754 of `predicted_rollout` (755 steps), on BOTH sides being
compared. For rupture-time and moment, each side's own `initial_velocities`
frame is prepended before the window (756 rows total), mirroring how
`gate.metrics()` and `plot.rupture.dynamics.py` build their own time series
-- not because 0-754 is itself a "valid_steps" count, simply because
rupture time / moment need the t=0 frame to integrate from.

Four row-types (board row, each run over all 7 `published.json` cases;
`M1_large` is skipped -- it is in `gate.TRUNCATE_TO_PUBLISHED` and has no
`published.json`/`reference.json` entry, gated separately, see `gate.py`'s
`main()` "paper" branch):
  1. det_ref_vs_published   -- fresh deterministic rollout of code="published"
                               (`det_rollout.py`, same mechanism as
                               `gate.cmd_reference`) vs the committed
                               published rollout files.
  2. fast_tf32_vs_published -- fresh deterministic rollout of code="current"
                               with `--rollout_fast=tf32` vs published.
  3. fast_fp32_vs_published -- same with `--rollout_fast=fp32`.
  4. eager_vs_eager_noise_floor -- TWO independent NON-deterministic
                               rollouts of the current code, via plain
                               `meshnet/train.py` directly (NOT
                               `det_rollout.py`'s deterministic wrapper --
                               natural GPU nondeterminism is the point).

For every row-type, side A is the "test" rollout (published-code reference,
fast-tf32, fast-fp32, or eager run 1) and side B is the comparison anchor:
the committed published rollout for row-types 1-3, or eager run 2
(arbitrarily chosen) for row-type 4. B's `predicted_rollout` is the
peak-normalizing reference for the slip-rate RMSE columns, and B plays the
"gt role" in the missed/false sense -- mirroring `gate.metrics()`'s
`hit_gt`/`hit_pr` naming, renamed `hit_b`/`hit_a` here since B is not always
literal ground truth (row-type 4 has no ground truth at all, only a
noise-floor comparison between two equally-valid eager runs):
  - missed = A did not hit (rupture threshold never crossed) but B did
  - false  = A hit but B did not

No thresholds are computed or proposed anywhere in this script -- raw
numbers only (owner's instruction; thresholds are a separate decision).
"""
import argparse
import csv
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib.tri as mtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate  # noqa: E402  (tests/paper_parity/gate.py -- reused, not redefined)

_PRD_PATH = gate.REPO / "scripts" / "utils" / "plot.rupture.dynamics.py"
_prd_spec = importlib.util.spec_from_file_location("plot_rupture_dynamics", _PRD_PATH)
prd = importlib.util.module_from_spec(_prd_spec)
_prd_spec.loader.exec_module(prd)  # only used for compute_moment(); no __main__ side effects

WINDOW = 755  # steps 0-754 inclusive, fixed by the owner for this table

CASES_FOR_TABLE = [c for c in gate.CASES if c not in gate.TRUNCATE_TO_PUBLISHED]

# Machine-readable artifacts (board row test-suite-overhaul, owner
# iris-vermeulen). Fixed, owner-specified paths -- gitignored under
# PROJECT_RULES.md rule 4's runs/ convention, never git-added.
OUT_DIR = gate.REPO / "runs" / "20261009_test-suite-measurement-table"
RAW_CSV = OUT_DIR / "raw_table.csv"
METRICS_JSONL = OUT_DIR / "per_trajectory_metrics.jsonl"

# Columns of raw_table.csv -- exactly the markdown table's columns, machine names.
RAW_COLUMNS = ["row_type", "case", "traj", "delta_rt_rmse_s", "delta_mw",
               "vx_rmse_norm", "vy_rmse_norm", "missed", "false", "note"]

# Per-trajectory intermediate values compare_pair() computes internally but
# the markdown table / raw_table.csv discard -- captured here so they don't
# have to be re-derived ad hoc for future diagnosis. rt_a/rt_b/hit_a/hit_b
# are per-node arrays (one rupture-time / hit-flag per mesh node, so JSONL,
# not CSV -- a flat CSV cell can't hold a 4743-length array cleanly).
EXTRA_KEYS = ["rt_a", "rt_b", "hit_a", "hit_b", "moment_a", "moment_b",
              "mw_a", "mw_b", "peak_b_vx", "peak_b_vy"]


def _already_done_blocks():
    """(row_type, case) pairs already fully written to RAW_CSV -- lets a
    restart after an interruption skip finished blocks instead of redoing
    the whole sweep. A block is only ever appended after its full
    rows_for() list is computed, so presence in the CSV means complete."""
    if not RAW_CSV.exists():
        return set()
    with open(RAW_CSV, newline="") as f:
        return {(r["row_type"], r["case"]) for r in csv.DictReader(f)}


def _append_block(rows):
    """Append one (row_type, case) block's rows to both artifacts. Called
    once per finished block, not only at the end of the whole sweep, so an
    interrupted run keeps every block completed so far."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RAW_CSV.exists()
    with open(RAW_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in RAW_COLUMNS})
    with open(METRICS_JSONL, "a") as f:
        for r in rows:
            rec = {"row_type": r["row_type"], "case": r["case"], "traj": r["traj"]}
            rec.update({k: r[k] for k in EXTRA_KEYS})
            f.write(json.dumps(rec) + "\n")


def _load_published(case):
    pkls = []
    for p in gate.pkls_in(gate.published_dir(case)):
        with open(p, "rb") as f:
            pkls.append(pickle.load(f))
    return pkls


def fresh_raw_rollout(case, cuda, code="current", extra=()):
    """Deterministic rollout via `det_rollout.py` (same subprocess shape as
    `gate.run_rollout`), returning the RAW per-trajectory pkl dicts instead
    of `gate.metrics()` summaries (which this script does not want -- it
    diffs raw arrays against another raw array, not against each side's own
    ground truth)."""
    ds, npz, (mdir, step), _ = gate.CASES[case]
    model_dir = gate.DATA / ds / mdir
    dataset_dir = gate.REGEN_DATASET_DIR.get(case, gate.DATA / ds / "dataset")
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        (Path(data_dir) / "test.npz").symlink_to(dataset_dir / npz)
        cmd = [sys.executable, str(gate.HERE / "det_rollout.py"), code, "--mode=rollout",
               f"--data_path={data_dir}/", f"--model_path={model_dir}/",
               f"--output_path={out_dir}/", f"--model_file=model-{step}.pt",
               f"--train_state_file=train_state-{step}.pt", f"--cuda_device_number={cuda}", *extra]
        r = subprocess.run(cmd, cwd=gate.REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"[{case}] rollout failed ({code}, extra={extra}):\n{r.stderr[-3000:]}")
        pkls = []
        for p in gate.pkls_in(out_dir):
            with open(p, "rb") as f:
                pkls.append(pickle.load(f))
    return pkls


def fresh_eager_rollout(case, cuda):
    """ONE non-deterministic rollout of the CURRENT code via plain
    `meshnet/train.py` -- deliberately NOT `det_rollout.py`'s deterministic
    wrapper, so natural GPU nondeterminism (no forced
    `torch.use_deterministic_algorithms`, no `CUBLAS_WORKSPACE_CONFIG`) is
    free to show up between two calls of this function."""
    ds, npz, (mdir, step), _ = gate.CASES[case]
    model_dir = gate.DATA / ds / mdir
    dataset_dir = gate.REGEN_DATASET_DIR.get(case, gate.DATA / ds / "dataset")
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        (Path(data_dir) / "test.npz").symlink_to(dataset_dir / npz)
        cmd = [sys.executable, str(gate.REPO / "meshnet" / "train.py"), "--mode=rollout",
               f"--data_path={data_dir}/", f"--model_path={model_dir}/",
               f"--output_path={out_dir}/", f"--model_file=model-{step}.pt",
               f"--train_state_file=train_state-{step}.pt", f"--cuda_device_number={cuda}"]
        r = subprocess.run(cmd, cwd=gate.REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"[{case}] eager rollout failed:\n{r.stderr[-3000:]}")
        pkls = []
        for p in gate.pkls_in(out_dir):
            with open(p, "rb") as f:
                pkls.append(pickle.load(f))
    return pkls


def compare_pair(pkl_a, pkl_b):
    """One trajectory, side A vs side B (see module docstring for the
    role of each side). Returns a dict of the reported numbers, or a
    'note' string if the fixed window could not be fully honored."""
    pred_a_full = np.asarray(pkl_a["predicted_rollout"], dtype=np.float64)
    pred_b_full = np.asarray(pkl_b["predicted_rollout"], dtype=np.float64)
    n = min(pred_a_full.shape[0], pred_b_full.shape[0], WINDOW)
    note = "" if n == WINDOW else f"only {n}/{WINDOW} steps available on both sides"
    pred_a, pred_b = pred_a_full[:n], pred_b_full[:n]

    init_a = np.asarray(pkl_a["initial_velocities"], dtype=np.float64)
    init_b = np.asarray(pkl_b["initial_velocities"], dtype=np.float64)
    speed_a = np.linalg.norm(np.concatenate([init_a, pred_a], axis=0), axis=-1)
    speed_b = np.linalg.norm(np.concatenate([init_b, pred_b], axis=0), axis=-1)

    rt_a = gate.rupture_time(speed_a)
    rt_b = gate.rupture_time(speed_b)
    hit_a, hit_b = rt_a < gate.UNREACHED, rt_b < gate.UNREACHED
    both = hit_a & hit_b
    rt_rmse = float(np.sqrt(np.mean((rt_a[both] - rt_b[both]) ** 2))) if both.any() else 0.0
    missed = int(np.sum((~hit_a) & hit_b))
    false = int(np.sum(hit_a & (~hit_b)))

    node_coords0 = np.asarray(pkl_a["node_coords"])[0]
    triang = mtri.Triangulation(node_coords0[:, 0] / 1e3, node_coords0[:, 1] / 1e3)
    moment_a = prd.compute_moment(speed_a, triang, gate.DT)
    moment_b = prd.compute_moment(speed_b, triang, gate.DT)
    mw_a = (2.0 / 3.0) * (np.log10(moment_a) - 9.1) if moment_a > 0 else None
    mw_b = (2.0 / 3.0) * (np.log10(moment_b) - 9.1) if moment_b > 0 else None
    delta_mw = (mw_a - mw_b) if (mw_a is not None and mw_b is not None) else None

    peak_b_vx = float(np.max(np.abs(pred_b[..., 0])))
    peak_b_vy = float(np.max(np.abs(pred_b[..., 1])))

    def norm_rmse(comp, peak_b):
        rmse = float(np.sqrt(np.mean((pred_a[..., comp] - pred_b[..., comp]) ** 2)))
        return (rmse / peak_b) if peak_b > 0 else None

    return {
        "delta_rt_rmse_s": rt_rmse,
        "delta_mw": delta_mw,
        "vx_rmse_norm": norm_rmse(0, peak_b_vx),
        "vy_rmse_norm": norm_rmse(1, peak_b_vy),
        "missed": missed,
        "false": false,
        "note": note,
        # Intermediate per-node / scalar values, discarded by the markdown
        # table and raw_table.csv but captured here (see EXTRA_KEYS /
        # per_trajectory_metrics.jsonl) for diagnosis, per owner instruction.
        "rt_a": rt_a.tolist(),
        "rt_b": rt_b.tolist(),
        "hit_a": hit_a.tolist(),
        "hit_b": hit_b.tolist(),
        "moment_a": float(moment_a),
        "moment_b": float(moment_b),
        "mw_a": (float(mw_a) if mw_a is not None else None),
        "mw_b": (float(mw_b) if mw_b is not None else None),
        "peak_b_vx": peak_b_vx,
        "peak_b_vy": peak_b_vy,
    }


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def rows_for(row_type, case, pkls_a, pkls_b):
    n_traj = min(len(pkls_a), len(pkls_b))
    rows = []
    mismatch_note = "" if len(pkls_a) == len(pkls_b) else \
        f"trajectory count mismatch A={len(pkls_a)} B={len(pkls_b)}, scored min={n_traj}"
    for i in range(n_traj):
        r = compare_pair(pkls_a[i], pkls_b[i])
        if mismatch_note:
            r["note"] = (r["note"] + "; " + mismatch_note).strip("; ")
        rows.append({"row_type": row_type, "case": case, "traj": i, **r})
    return rows


def to_markdown(all_rows):
    header = ("| row-type | case | traj | delta RT RMSE (s) | delta Mw | "
              "vx RMSE/peak | vy RMSE/peak | missed | false | note |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in all_rows:
        lines.append(
            f"| {r['row_type']} | {r['case']} | {r['traj']} | "
            f"{fmt(r['delta_rt_rmse_s'])} | {fmt(r['delta_mw'])} | "
            f"{fmt(r['vx_rmse_norm'])} | {fmt(r['vy_rmse_norm'])} | "
            f"{r['missed']} | {r['false']} | {r['note']} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", default="0", help="single GPU id -- never parallelized here")
    ap.add_argument("--cases", nargs="*", default=CASES_FOR_TABLE,
                    help=f"default: all of {CASES_FOR_TABLE}")
    ap.add_argument("--row-types", nargs="*",
                    default=["det_ref_vs_published", "fast_tf32_vs_published",
                             "fast_fp32_vs_published", "eager_vs_eager_noise_floor"])
    a = ap.parse_args()
    cuda = a.cuda
    unknown = set(a.cases) - set(CASES_FOR_TABLE)
    if unknown:
        sys.exit(f"unknown/unsupported case(s) (M1_large has no published/reference entry): {sorted(unknown)}")

    done = _already_done_blocks()
    if done:
        print(f"resuming: {len(done)} (row_type, case) block(s) already in {RAW_CSV}, skipping",
              file=sys.stderr, flush=True)

    all_rows = []
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="") as f:
            for r in csv.DictReader(f):
                r["traj"] = int(r["traj"])
                for k in ("delta_rt_rmse_s", "delta_mw", "vx_rmse_norm", "vy_rmse_norm"):
                    r[k] = float(r[k]) if r[k] not in ("", "n/a") else None
                r["missed"], r["false"] = int(r["missed"]), int(r["false"])
                all_rows.append(r)

    t0 = time.time()
    for case in a.cases:
        print(f"=== {case} ===", file=sys.stderr, flush=True)
        published = _load_published(case)

        if "det_ref_vs_published" in a.row_types and ("det_ref_vs_published", case) not in done:
            t1 = time.time()
            a_pkls = fresh_raw_rollout(case, cuda, code="published")
            print(f"  det_ref_vs_published: {len(a_pkls)} traj in {time.time()-t1:.0f}s",
                  file=sys.stderr, flush=True)
            rows = rows_for("det_ref_vs_published", case, a_pkls, published)
            _append_block(rows)
            all_rows += rows

        if "fast_tf32_vs_published" in a.row_types and ("fast_tf32_vs_published", case) not in done:
            t1 = time.time()
            a_pkls = fresh_raw_rollout(case, cuda, code="current", extra=("--rollout_fast=tf32",))
            print(f"  fast_tf32_vs_published: {len(a_pkls)} traj in {time.time()-t1:.0f}s",
                  file=sys.stderr, flush=True)
            rows = rows_for("fast_tf32_vs_published", case, a_pkls, published)
            _append_block(rows)
            all_rows += rows

        if "fast_fp32_vs_published" in a.row_types and ("fast_fp32_vs_published", case) not in done:
            t1 = time.time()
            a_pkls = fresh_raw_rollout(case, cuda, code="current", extra=("--rollout_fast=fp32",))
            print(f"  fast_fp32_vs_published: {len(a_pkls)} traj in {time.time()-t1:.0f}s",
                  file=sys.stderr, flush=True)
            rows = rows_for("fast_fp32_vs_published", case, a_pkls, published)
            _append_block(rows)
            all_rows += rows

        if "eager_vs_eager_noise_floor" in a.row_types and ("eager_vs_eager_noise_floor", case) not in done:
            t1 = time.time()
            run1 = fresh_eager_rollout(case, cuda)
            run2 = fresh_eager_rollout(case, cuda)
            print(f"  eager_vs_eager_noise_floor: {len(run1)}/{len(run2)} traj in {time.time()-t1:.0f}s",
                  file=sys.stderr, flush=True)
            rows = rows_for("eager_vs_eager_noise_floor", case, run1, run2)
            _append_block(rows)
            all_rows += rows

    print(f"total wall time (this invocation): {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    print(to_markdown(all_rows))


if __name__ == "__main__":
    main()

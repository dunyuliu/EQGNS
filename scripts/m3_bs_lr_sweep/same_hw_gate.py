#!/usr/bin/env python3
"""Same-hardware training gate (docs/dev/M3_BATCH_LR_SWEEP_DESIGN.md section 3.4).

The committed gate (tests/test_training_gate.py) compares CURRENT meshnet/train.py
on whatever GPU it runs on against a reference trace made by the frozen oracle
meshnet/train.py.published on an A100, so off the A100 it conflates hardware and
code. This script separates them, on ONE device, without touching the committed
reference or tests/:

  1. run the frozen oracle (tests/fixtures/training_gate/published_pipeline_cli.py)
     for the gate's 1000-step M1 D1 configuration and write a device-native
     reference JSON (same schema as tests/golden/training_gate_reference.json,
     plus a "device_note") to --ref-out;
  2. run current train.py (tests/fixtures/meshnet/seeded_pipeline_cli.py) with the
     identical configuration;
  3. compare per step at the gate's own tolerance (rtol = atol = 1e-6) and report
     deltas at steps 0 / 1 / 500 / 1000 plus max |delta| and max relative delta,
     against the native reference and (for context) the committed A100 one.

Exit 0 on PASS, 1 on FAIL, 2 on a run error. Writes --result-out (JSON).

Usage (from the repo root, on the target GPU node, with
EQGNS_TRAINING_GATE_DATA_DIR pointing at the real M1 D1 dataset):
    python scripts/m3_bs_lr_sweep/same_hw_gate.py --work /tmp/gate \
        --ref-out results/training_gate_reference_<device>.json \
        --result-out results/same_hw_gate.json --device-note "NVIDIA GH200 120GB"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TESTS = os.path.join(_REPO, "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from fixtures.training_gate import common  # noqa: E402

NSTEPS = 1000
TOL = 1e-6
REPORT_STEPS = (0, 1, 500, 1000)
COMMITTED_REF = os.path.join(_REPO, "tests", "golden", "training_gate_reference.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _gpu_name():
    """GPU name from nvidia-smi; raises (never returns a placeholder) so a gate result is
    always labelled with the hardware it ran on."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    names = out.stdout.strip().splitlines()
    if out.returncode != 0 or not names or not names[0].strip():
        raise RuntimeError(f"nvidia-smi gave no GPU name (rc={out.returncode}): {out.stderr.strip()[:200]}")
    return names[0].strip()


def _run_batch(cli, model_dir, batch):
    """Same 1000-step M1 D1 gate configuration as common.run_training() but at a different
    --batch_size (the committed gate is fixed at 2); the config copy is unchanged."""
    import shutil
    os.makedirs(model_dir, exist_ok=True)
    shutil.copy(common.TRAINING_GOLDEN_CONFIG, os.path.join(model_dir, "config.json"))
    t0 = time.time()
    common.run_cli(cli, ["--mode=train", "--data_path=" + common.D1_DATASET_DIR + "/",
                         "--model_path=" + model_dir + "/", f"--batch_size={batch}",
                         f"--ntraining_steps={NSTEPS}", f"--nsave_steps={NSTEPS + 1}"], timeout=3600)
    return common.parse_loss_log(os.path.join(model_dir, "loss_log.txt")), time.time() - t0


def _compare(cur, ref, key):
    c = np.asarray([r[key] for r in cur]); r = np.asarray(ref[key])
    d = np.abs(c - r)
    rel = d / np.maximum(np.abs(r), 1e-300)
    bad = d > TOL + TOL * np.abs(r)
    steps = ref["steps"]
    per = {str(s): {"current": float(c[steps.index(s)]), "reference": float(r[steps.index(s)]),
                    "abs_delta": float(d[steps.index(s)]), "rel_delta": float(rel[steps.index(s)])}
           for s in REPORT_STEPS if s in steps}
    return {"n_steps": int(len(c)), "n_differing": int(bad.sum()), "max_abs_delta": float(d.max()),
            "max_rel_delta": float(rel.max()), "first_differing_step": (int(steps[int(np.argmax(bad))]) if bad.any() else None),
            "pass": bool(not bad.any()), "per_step": per}


def _run(cli, model_dir):
    t0 = time.time()
    rows = common.run_training(model_dir, NSTEPS, cli)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="scratch dir for the two model dirs")
    ap.add_argument("--ref-out", required=True, help="device-native oracle reference JSON to write")
    ap.add_argument("--result-out", required=True, help="comparison result JSON to write")
    ap.add_argument("--device-note", default=None)
    ap.add_argument("--git-sha", required=True, help="40-hex commit of the code under test")
    ap.add_argument("--batch-sizes", default="2",
                    help="comma list; 2 = the committed gate's own configuration (always run first); extra "
                         "batch sizes (e.g. 4,12 for the M3 sweep arms) are run oracle-vs-current the same way "
                         "and must all pass")
    a = ap.parse_args()

    common.require_d1_dataset()
    os.makedirs(a.work, exist_ok=True)
    try:
        gpu = a.device_note or _gpu_name()
    except Exception as e:  # noqa: BLE001
        print(f"GPU NAME LOOKUP FAILED: {e}"); return 2
    extra_batches = [int(b) for b in a.batch_sizes.split(",") if b.strip() and int(b) != 2]

    try:
        ref_rows, t_ref = _run(common.PUBLISHED_CLI, os.path.join(a.work, "oracle"))
    except AssertionError as e:
        print(f"ORACLE RUN FAILED:\n{e}"); return 2
    reference = {
        "steps": [r["step"] for r in ref_rows],
        "train_loss": [r["train_loss"] for r in ref_rows],
        "valid_loss": [r["valid_loss"] for r in ref_rows],
        "ntraining_steps": NSTEPS, "seed": common.SEED,
        "dataset": "case3.200m.homo.a.Vw/dataset (real M1 D1, staged copy)",
        "config": "tests/fixtures/training_golden/config.json (loss_report_step=1 test override)",
        "device": f"{gpu}; CUDA_VISIBLE_DEVICES={common.GPU_ENV['CUDA_VISIBLE_DEVICES']}, "
                  "torch.use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "device_note": "DEVICE-NATIVE reference for the same-hardware check (design doc 3.4); NOT the "
                       "paper-parity oracle tests/golden/training_gate_reference.json (A100), which stays frozen",
        "oracle_file": "meshnet/train.py.published", "oracle_sha256": _sha256(common.PUBLISHED_ORACLE),
        "code_git_sha": a.git_sha, "generated_by": "scripts/m3_bs_lr_sweep/same_hw_gate.py",
        "oracle_wall_s": round(t_ref, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.ref_out)), exist_ok=True)
    with open(a.ref_out, "w") as f:
        json.dump(reference, f, indent=2); f.write("\n")
    print(f"oracle trace written: {a.ref_out} ({t_ref:.0f}s)")

    try:
        cur_rows, t_cur = _run(common.CURRENT_CLI, os.path.join(a.work, "current"))
    except AssertionError as e:
        print(f"CURRENT RUN FAILED:\n{e}"); return 2
    print(f"current trace done ({t_cur:.0f}s)")

    result = {"device": gpu, "code_git_sha": a.git_sha, "tolerance": TOL, "oracle_wall_s": round(t_ref, 1),
              "current_wall_s": round(t_cur, 1), "native_reference": os.path.abspath(a.ref_out),
              "vs_native": {k: _compare(cur_rows, reference, k) for k in ("train_loss", "valid_loss")}}
    if os.path.exists(COMMITTED_REF):
        with open(COMMITTED_REF) as f:
            a100 = json.load(f)
        result["vs_committed_a100"] = {k: _compare(cur_rows, a100, k) for k in ("train_loss", "valid_loss")}
        result["oracle_native_vs_committed_a100"] = {k: _compare(ref_rows, a100, k) for k in ("train_loss", "valid_loss")}
    passed = result["vs_native"]["train_loss"]["pass"] and result["vs_native"]["valid_loss"]["pass"]
    result["extra_batches"] = {}
    for b in extra_batches:
        try:
            ref_b, t_rb = _run_batch(common.PUBLISHED_CLI, os.path.join(a.work, f"oracle_b{b}"), b)
            cur_b, t_cb = _run_batch(common.CURRENT_CLI, os.path.join(a.work, f"current_b{b}"), b)
        except AssertionError as e:
            print(f"BATCH {b} RUN FAILED:\n{e}"); return 2
        ref_bd = {"steps": [r["step"] for r in ref_b], "train_loss": [r["train_loss"] for r in ref_b],
                  "valid_loss": [r["valid_loss"] for r in ref_b]}
        cmp_b = {k: _compare(cur_b, ref_bd, k) for k in ("train_loss", "valid_loss")}
        cmp_b["oracle_wall_s"] = round(t_rb, 1); cmp_b["current_wall_s"] = round(t_cb, 1)
        result["extra_batches"][str(b)] = cmp_b
        passed = passed and cmp_b["train_loss"]["pass"] and cmp_b["valid_loss"]["pass"]
        print(f"batch {b}: oracle {t_rb:.0f}s, current {t_cb:.0f}s")
    result["pass"] = passed
    with open(a.result_out, "w") as f:
        json.dump(result, f, indent=2); f.write("\n")

    for label in ("vs_native", "vs_committed_a100", "oracle_native_vs_committed_a100"):
        if label not in result:
            continue
        for k in ("train_loss", "valid_loss"):
            c = result[label][k]
            print(f"[{label}/{k}] pass={c['pass']} differing={c['n_differing']}/{c['n_steps']} "
                  f"max_abs={c['max_abs_delta']:.3e} max_rel={c['max_rel_delta']:.3e} first_diff={c['first_differing_step']}")
            for s, v in c["per_step"].items():
                print(f"    step {s:>4}: cur={v['current']:.10g} ref={v['reference']:.10g} "
                      f"abs={v['abs_delta']:.3e} rel={v['rel_delta']:.3e}")
    for b, cmp_b in result["extra_batches"].items():
        for k in ("train_loss", "valid_loss"):
            c = cmp_b[k]
            print(f"[batch{b}/{k}] pass={c['pass']} differing={c['n_differing']}/{c['n_steps']} "
                  f"max_abs={c['max_abs_delta']:.3e} max_rel={c['max_rel_delta']:.3e} first_diff={c['first_differing_step']}")
    print(f"SAME_HW_GATE={'PASS' if passed else 'FAIL'} device={gpu} batches=2{''.join(',' + b for b in result['extra_batches'])}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

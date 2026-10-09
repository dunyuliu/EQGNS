#!/usr/bin/env python3
"""Regenerate tests/golden/convergence_gate_nightly_tier2.json.

Run this ONLY when an intentional, confirmed-correct change to the
train/rollout path changes these metrics on real D1 -- note the regeneration
and why in the commit message. Never run this to silence a failing test
without first finding the cause.

Needs data/gns-sample/case3.200m.homo.a.Vw/dataset/ checked out. Takes
several minutes (CPU-forced small-budget real training + real rollout).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_TESTS_DIR, os.path.join(_TESTS_DIR, "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_convergence_gate_nightly import (  # noqa: E402
    _run_pipeline, REFERENCE_PATH, NTRAINING_STEPS, ROLLOUT_FRAMES, TRAJECTORY_KEY)
from fixtures.training_golden import common  # noqa: E402

if __name__ == "__main__":
    common.require_d1_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = _run_pipeline(Path(tmp))

    reference = {
        "metrics": metrics,
        "ntraining_steps": NTRAINING_STEPS,
        "rollout_frames": ROLLOUT_FRAMES,
        "trajectory": TRAJECTORY_KEY,
        "seed": common.SEED,
        "dataset": "data/gns-sample/case3.200m.homo.a.Vw/dataset (real D1, gitignored, not committed)",
        "config": "tests/fixtures/training_golden/config.json (real M1 architecture, loss_report_step=1 test override)",
        "device": "cpu (CUDA_VISIBLE_DEVICES='', see common.py)",
    }
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    with open(REFERENCE_PATH, "w") as f:
        json.dump(reference, f, indent=2)
        f.write("\n")
    print(f"wrote {REFERENCE_PATH}")
    print(json.dumps(metrics, indent=2))

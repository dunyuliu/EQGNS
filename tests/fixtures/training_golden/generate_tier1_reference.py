#!/usr/bin/env python3
"""Regenerate tests/golden/training_golden_tier1.json.

Run this ONLY when an intentional, confirmed-correct change to
meshnet/train.py's training path changes the loss curve on real D1 -- note
the regeneration and why in the commit message (tests/README.md's golden-file
policy). Never run this to silence a failing test without first finding the
cause; this file IS the oracle test_training_golden.py checks against.

Needs data/gns-sample/case3.200m.homo.a.Vw/dataset/ checked out.
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

from test_training_golden import _run_pipeline, REFERENCE_PATH, NTRAINING_STEPS  # noqa: E402
from fixtures.training_golden import common  # noqa: E402

if __name__ == "__main__":
    common.require_d1_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        rows = _run_pipeline(Path(tmp))

    reference = {
        "steps": [r["step"] for r in rows],
        "train_loss": [r["train_loss"] for r in rows],
        "valid_loss": [r["valid_loss"] for r in rows],
        "ntraining_steps": NTRAINING_STEPS,
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

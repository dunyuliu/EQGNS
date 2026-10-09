#!/usr/bin/env python3
"""Regenerate tests/golden/training_gate_reference.json.

Runs the FROZEN ORACLE meshnet/train.py.published (never edited) for
NTRAINING_STEPS steps on the real, published M1 D1 dataset, under the
seeded published_pipeline_cli.py wrapper in this directory, and records
its per-step train/valid loss as the gate's golden trace.

Run this ONLY when train.py.published itself changes (it shouldn't --
it is a frozen reference oracle) or when NTRAINING_STEPS/the config
changes; note the regeneration and why in the commit message
(tests/README.md's golden-file policy). Never run this to silence a
failing test without first finding the cause.

Needs a reachable real D1 dataset (see common.require_d1_dataset /
EQGNS_TRAINING_GATE_DATA_DIR) and a free CUDA device (default: GPU 1,
see common.GPU_ENV).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from fixtures.training_gate import common  # noqa: E402

REFERENCE_PATH = os.path.join(common.REPO_ROOT, "tests", "golden", "training_gate_reference.json")
NTRAINING_STEPS = 1000


def _git_sha(path):
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", path],
        cwd=common.REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


if __name__ == "__main__":
    common.require_d1_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "model")
        rows = common.run_training(model_dir, NTRAINING_STEPS, common.PUBLISHED_CLI)

    reference = {
        "steps": [r["step"] for r in rows],
        "train_loss": [r["train_loss"] for r in rows],
        "valid_loss": [r["valid_loss"] for r in rows],
        "ntraining_steps": NTRAINING_STEPS,
        "seed": common.SEED,
        "dataset": "data/gns-sample/case3.200m.homo.a.Vw/dataset (real M1 D1, gitignored, not committed)",
        "config": "tests/fixtures/training_golden/config.json (real M1 architecture, loss_report_step=1 test override)",
        "device": "cuda:1 (CUDA_VISIBLE_DEVICES=1, torch.use_deterministic_algorithms(True, warn_only=True), see common.py)",
        "oracle_file": "meshnet/train.py.published",
        "oracle_git_sha": _git_sha(os.path.join("meshnet", "train.py.published")),
        "oracle_sha256": _sha256(common.PUBLISHED_ORACLE),
        "generated_by": "tests/fixtures/training_gate/generate_training_gate_reference.py",
    }
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    with open(REFERENCE_PATH, "w") as f:
        json.dump(reference, f, indent=2)
        f.write("\n")
    print(f"wrote {REFERENCE_PATH}")

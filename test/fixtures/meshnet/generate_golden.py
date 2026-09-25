#!/usr/bin/env python3
"""Regenerate test/golden/meshnet_e2e_rollout_golden.npz.

Run this ONLY when a meshnet change is confirmed correct and the golden
needs to move; note the regeneration and why in the commit message (per
test/README.md's e2e tier policy). Do not run this to silence a failing
test without understanding why the numbers changed.
"""
import os
import sys

import numpy as np

_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
for _p in (_REPO_ROOT, _TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tempfile  # noqa: E402
from test_meshnet_e2e_golden import _run_pipeline, GOLDEN_PATH  # noqa: E402

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        result = _run_pipeline(pathlib.Path(tmp))
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    np.savez(
        GOLDEN_PATH,
        predicted_rollout=result["predicted_rollout"],
        predicted_rollout_shape=np.array(result["predicted_rollout"].shape),
        ground_truth_rollout=result["ground_truth_rollout"],
    )
    print(f"wrote {GOLDEN_PATH}")

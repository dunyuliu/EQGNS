#!/usr/bin/env python3
"""Test-only, seeded wrapper around the real `python3 -m meshnet.train` CLI.

This is deliberately NOT a reimplementation of train.py's logic: it seeds the
RNGs and then calls the exact same `absl.app.run(meshnet.train.main)` entry
point documented in README.md, so the e2e golden test in
test_meshnet_e2e_golden.py exercises the real CLI, not a test-side stand-in.

Usage matches `python3 -m meshnet.train` exactly, e.g.:
    python3 seeded_pipeline_cli.py --mode=train --data_path=... --model_path=...
"""
import os
import random
import sys

import numpy as np
import torch

SEED = 20260101  # arbitrary, fixed forever for this golden file

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from absl import app  # noqa: E402
from meshnet import train  # noqa: E402

if __name__ == "__main__":
    app.run(train.main)

#!/usr/bin/env python3
"""Test-only, seeded wrapper around the FROZEN ORACLE `meshnet/train.py.published`
(PATHWAY_FORWARD.md row `test-suite-overhaul` sub-item (2) -- the training gate).

Mirrors tests/fixtures/meshnet/seeded_pipeline_cli.py exactly (same SEED, same
seeding calls, same absl entry point shape) except it loads
meshnet/train.py.published -- a file, not a package module, so it cannot be
`import`ed normally -- via importlib.util.spec_from_file_location instead of
`from meshnet import train`. meshnet/train.py.published is NEVER edited (it is
the frozen reference oracle); this wrapper only reads it.

Usage matches `python3 -m meshnet.train` / seeded_pipeline_cli.py exactly, e.g.:
    python3 published_pipeline_cli.py --mode=train --data_path=... --model_path=...
"""
import importlib.machinery
import importlib.util
import os
import random
import sys

import numpy as np
import torch

SEED = 20260101  # same fixed seed as tests/fixtures/meshnet/seeded_pipeline_cli.py

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_PUBLISHED_PATH = os.path.join(_REPO_ROOT, "meshnet", "train.py.published")

from absl import app  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("train_published_oracle", _PUBLISHED_PATH)
_spec = importlib.util.spec_from_file_location("train_published_oracle", _PUBLISHED_PATH, loader=_loader)
_train_published = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train_published
_spec.loader.exec_module(_train_published)

if __name__ == "__main__":
    app.run(_train_published.main)

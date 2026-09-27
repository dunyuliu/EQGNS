#!/usr/bin/env python3
"""Thin CLI wrapper around `meshnet.train`'s absl `app.run(main)` entry point
that additionally calls `torch.use_deterministic_algorithms(True)` before the
rollout runs, for the PR #3 determinism-mode control experiment (see
test/paper_parity/NOTES_pr3.md 'Determinism experiment').

This file is TEST INFRASTRUCTURE, not production code: it imports
`meshnet.train` unmodified and only adds a determinism flag ahead of calling
its existing `main`. It never edits meshnet/train.py itself.

Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) set in the environment
BEFORE the python process starts (torch reads it at CUDA-context-init time,
which is too early to set from inside this script reliably) -- the caller
(measure_spread.py) sets this env var on the subprocess.

Usage: identical flags to `python3 -m meshnet.train --mode=rollout ...`,
e.g.:
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 test/fixtures/paper_parity/deterministic_rollout_cli.py \
        --mode=rollout --data_path=... --model_path=... --output_path=... \
        --model_file=... --train_state_file=... --cuda_device_number=3
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# warn_only=False: we want a hard error if some op in the rollout path has no
# deterministic kernel, not a silently-nondeterministic fallback -- that
# silence would defeat the whole point of this control experiment.
torch.use_deterministic_algorithms(True, warn_only=False)

from absl import app  # noqa: E402
from meshnet import train  # noqa: E402

if __name__ == "__main__":
    app.run(train.main)

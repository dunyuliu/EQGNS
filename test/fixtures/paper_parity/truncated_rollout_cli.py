#!/usr/bin/env python3
"""Truncated-horizon rollout CLI wrapper (PR #4 economize-gates: quick tier).

Same pattern as deterministic_rollout_cli.py (PR #3): thin wrapper around
`meshnet.train`'s absl `app.run(main)` entry point that imports
`meshnet.train` UNMODIFIED and only intercepts how many autoregressive
steps it runs, by monkeypatching the module-level `rollout` NAME (not its
body) before `main`/`predict` are invoked. `predict()` still calls
`rollout(simulator, features, nsteps, device)` -- this wrapper's
replacement caps `nsteps` at the environment-provided horizon and then
calls straight through to the ORIGINAL, unmodified `rollout()` function
object for exactly that many real autoregressive steps. This is what
makes the tier actually cheaper (fewer real GNN forward passes per
trajectory), not a post-hoc slice of an already-full-length run.

This file is TEST INFRASTRUCTURE, never edits meshnet/train.py.

IMPORTANT (read before treating a quick-tier PASS as sufficient): this is
an ADDITIONAL fast-feedback signal. It does not replace the full-length
gate (run_gate.py with no --truncated-nsteps flag) as the thing merges
are judged on -- see test/paper_parity/NOTES_pr4.md.

Required env var: TRUNCATED_ROLLOUT_NSTEPS (int). No default -- an
unset horizon must be a loud error, not a silent full-length run.

Usage: identical flags to `python3 -m meshnet.train --mode=rollout ...`,
e.g.:
    TRUNCATED_ROLLOUT_NSTEPS=100 python3 \
        test/fixtures/paper_parity/truncated_rollout_cli.py \
        --mode=rollout --data_path=... --model_path=... --output_path=... \
        --model_file=... --train_state_file=... --cuda_device_number=1
"""
import os
import sys

# Resolve `meshnet`/`gns` against the CALLER's cwd (set by
# measure_truncated_spread.py to either REPO_ROOT or a throwaway
# meshnet_src_root tree for the falsifiability re-check), NOT a hardcoded
# path derived from this file's own location. A hardcoded
# `parents[2]`-style insert would always resolve to THIS repo's real
# meshnet/ regardless of --meshnet-src-root, silently defeating the
# falsifiability tree-copy override (discovered empirically: an earlier
# version of this file hardcoded the repo root and both the null and the
# planted-defect tree produced bit-identical output, because both were
# secretly importing the same real meshnet/ instead of the perturbed
# copy).
sys.path.insert(0, os.getcwd())

_N_RAW = os.environ.get("TRUNCATED_ROLLOUT_NSTEPS")
if _N_RAW is None:
    raise RuntimeError(
        "TRUNCATED_ROLLOUT_NSTEPS must be set in the environment -- this "
        "wrapper refuses to silently fall back to a full-length rollout.")
_N = int(_N_RAW)
if _N <= 0:
    raise ValueError(f"TRUNCATED_ROLLOUT_NSTEPS must be a positive int, got {_N}")

from absl import app  # noqa: E402
from meshnet import train  # noqa: E402

_original_rollout = train.rollout


def _truncated_rollout(simulator, features, nsteps, device):
    """Cap nsteps AND truncate the ground-truth velocity feature to match.

    `rollout()`'s own body (meshnet/train.py, unmodified) does
    `ground_truth_velocities = velocities[INPUT_SEQUENCE_LENGTH:]` from the
    FULL-length `features[3]`, then later computes
    `loss = (predictions - ground_truth_velocities) ** 2` using the
    ORIGINAL (untruncated) `ground_truth_velocities`, regardless of how
    many steps the `for step in range(nsteps)` loop actually ran --
    passing a smaller `nsteps` alone hits a real shape-mismatch error
    ((capped, nnodes, dim) predictions vs (826, nnodes, dim) ground truth)
    that we discovered empirically running this wrapper for the first
    time. This is a latent shape assumption in `rollout()` (only ever
    exercised with nsteps == full length before this PR), not something
    this test-infra wrapper is allowed to patch inside meshnet/train.py.
    Work around it here, in test infra, by ALSO truncating the velocity
    feature (features[3]) that `rollout()` derives `ground_truth_velocities`
    from, to `INPUT_SEQUENCE_LENGTH + capped` rows, so the internal slice
    naturally comes out the right length. The other feature tensors
    (node_coords/node_types/node_property/pressures/cells) are only ever
    indexed at `[0]` inside `rollout()` (the mesh doesn't move step to
    step) so they are left untouched -- truncating them would be a no-op
    at best and is not needed for correctness.
    """
    capped = min(nsteps, _N)
    seq_len = train.INPUT_SEQUENCE_LENGTH
    total_len = min(seq_len + capped, len(features[3]))
    features = list(features)
    features[3] = features[3][:total_len]
    return _original_rollout(simulator, tuple(features), capped, device)


train.rollout = _truncated_rollout

if __name__ == "__main__":
    app.run(train.main)

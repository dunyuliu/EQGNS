"""Shared helpers for the training-guard tiers (PATHWAY_FORWARD.md row
`training-guard`): tests/test_training_golden.py (tier 1, `training_golden`
marker) and tests/test_convergence_gate_nightly.py (tier 2,
`convergence_gate_nightly` marker).

Both tiers run the real `python3 -m meshnet.train` CLI (via the same seeded
wrapper as the e2e/A-B tiers, tests/fixtures/meshnet/seeded_pipeline_cli.py)
against the real, published D1 dataset
(data/gns-sample/case3.200m.homo.a.Vw/dataset/ -- PROJECT_RULES.md rule 3: a
real, gitignored, 249GB-class directory, never committed, never symlinked
over inside this repo's own tree). That data only exists on a machine that
has fetched data/gns-sample/; it is never present in GitHub Actions CI, so
both tiers skip (not fail) when it is absent, the same convention as
tests/paper_parity's `_require_opt_in` fixture.

CPU-forced (`CUDA_VISIBLE_DEVICES=""`): meshnet/train.py has no CPU-force
flag of its own (`device = torch.device('cuda' if torch.cuda.is_available()
else 'cpu')`, unconditionally preferring CUDA) -- this machine has GPUs, but
CI's torch build is CPU-only, so both committed references were generated,
and must be reproduced, on CPU; setting the env var in the subprocess is a
test-harness choice, not a production-code edit.
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CLI = os.path.join(REPO_ROOT, "tests", "fixtures", "meshnet", "seeded_pipeline_cli.py")
D1_DATASET_DIR = os.path.join(REPO_ROOT, "data", "gns-sample", "case3.200m.homo.a.Vw", "dataset")
CONFIG_SRC = os.path.join(os.path.dirname(__file__), "config.json")

# tests/fixtures/meshnet/seeded_pipeline_cli.py's fixed SEED -- recorded here
# too (not re-derived) so a reference artifact's "seed" field is self-evidently
# tied to the actual value used, not just an assumption about the wrapper.
SEED = 20260101


def require_d1_dataset():
    """Skip (never fail) if the real D1 dataset is not checked out on this
    machine. 249GB-class, gitignored (PROJECT_RULES.md rule 3); only present
    on a machine that fetched data/gns-sample/, never in CI."""
    if not os.path.exists(os.path.join(D1_DATASET_DIR, "train.npz")):
        pytest.skip(
            f"real D1 dataset not found at {D1_DATASET_DIR} -- this tier "
            f"only runs on a machine with data/gns-sample/ checked out "
            f"(see tests/README.md)")


def parse_loss_log(path):
    """Parse meshnet/train.py's loss_log.txt lines:
    "{step} {train_loss} {valid_loss} {rms} {rms} {max} {max}\\n" into a list
    of {step, train_loss, valid_loss} dicts (same parsing contract as
    tests/test_ab_seeded_determinism.py's _parse_loss_log, duplicated rather
    than imported so this helper has no dependency on another test module)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            rows.append({
                "step": int(parts[0]),
                "train_loss": float(parts[1]),
                "valid_loss": float(parts[2]),
            })
    return rows


def run_cli(args, cwd=None, timeout=900):
    """Run the real seeded CLI as a subprocess, CPU-forced; raise with full
    stdout/stderr on a non-zero exit so a failure is diagnosable from the
    pytest output alone."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, CLI] + list(args)
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=env,
                             capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (
        f"subprocess failed: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def write_truncated_trajectory_npz(src_npz, dst_npz, traj_key, nframes):
    """Write a single trajectory, truncated to its first `nframes` frames,
    as its own one-trajectory npz at dst_npz -- a CPU-time-budget truncation
    of the real D1 test set (data/gns-sample/.../dataset/test.npz, 6
    trajectories x 827 frames x 4743 nodes; rolling out the full set takes
    >1.5 CPU-hours), the same truncation device as
    tests/paper_parity/gate.py's write_quick_npz/QUICK_STEPS. Derived at
    test-run time from the real, gitignored data -- never committed as a
    fixture itself (PROJECT_RULES.md rule 3: data/ contents are read-only
    inputs, and a derived slice of real D1 is still real D1)."""
    import numpy as np
    data = np.load(src_npz, allow_pickle=True)
    traj = data[traj_key].item()
    truncated = {k: np.asarray(v)[:nframes] for k, v in traj.items()}
    holder = np.empty((), dtype=object)
    holder[()] = truncated
    np.savez(dst_npz, **{traj_key: holder})

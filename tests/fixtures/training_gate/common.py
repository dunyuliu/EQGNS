"""Shared helpers for the training gate (PATHWAY_FORWARD.md row
test-suite-overhaul sub-item (2)): tests/test_training_gate.py.

This tier runs the REAL python3 -m meshnet.train training path (current
code, via tests/fixtures/meshnet/seeded_pipeline_cli.py) and the REAL
meshnet/train.py.published training path (frozen oracle, via
published_pipeline_cli.py in this directory) as two separate subprocesses,
each on the real, published M1 D1 dataset
(case3.200m.homo.a.Vw/dataset/train.npz + valid.npz, PROJECT_RULES.md rule
3: a real, gitignored, 249GB-class tree, never committed, never symlinked
over inside this repo's own tree) and the real M1 architecture config
(tests/fixtures/training_golden/config.json, nmp10/latent128, verbatim from
data/gns-sample/case3.200m.homo.a.Vw/models.nmp10.cotopaxi/config.json
except loss_report_step: 1000 -> 1, so every one of this tier's steps is
logged).

This agent's checkout has no data/gns-sample/ present locally (it is
gitignored content fetched separately per machine, not carried by every
checkout). The real D1 dataset is only ever READ, never written, from the
path below -- override with EQGNS_TRAINING_GATE_DATA_DIR on any other
machine, the same override-a-real-data-path shape as
tests/paper_parity/gate.py's EQGNS_REGEN_DATA and
tests/fixtures/training_golden/common.py's require_d1_dataset().

GPU, not CPU (test-harness choice, not a production-code edit): host load
was ~71/64 cores heavily loaded by unrelated users at the time this tier was
authored (not an idle box -- see the owner's idle-system timing rule, which
applies to TIMING, not to this correctness-only gate); GPU 1 was confirmed
free (nvidia-smi --query-compute-apps, 0 MiB) and is used via
CUDA_VISIBLE_DEVICES=1, with OMP/MKL/OPENBLAS_NUM_THREADS capped at 8 to
keep the subprocess's CPU-side (data loading) footprint small on a loaded
host. torch.use_deterministic_algorithms(True, warn_only=True) plus a fixed
global seed (set identically by both CLI wrappers before any torch call)
makes both the current and the published path bit-reproducible on this GPU
(the same property tests/test_ab_seeded_determinism.py already relies on for
reruns of the current path alone); CUBLAS_WORKSPACE_CONFIG is set so cuBLAS
determinism holds too.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CURRENT_CLI = os.path.join(REPO_ROOT, "tests", "fixtures", "meshnet", "seeded_pipeline_cli.py")
PUBLISHED_CLI = os.path.join(os.path.dirname(__file__), "published_pipeline_cli.py")
PUBLISHED_ORACLE = os.path.join(REPO_ROOT, "meshnet", "train.py.published")
TRAINING_GOLDEN_CONFIG = os.path.join(REPO_ROOT, "tests", "fixtures", "training_golden", "config.json")

D1_DATASET_DIR = os.environ.get(
    "EQGNS_TRAINING_GATE_DATA_DIR",
    "/home/utig5/dliu/eq_rupture_gns/data/gns-sample/case3.200m.homo.a.Vw/dataset",
)

# Same fixed seed used by both CLI wrappers (tests/fixtures/meshnet/seeded_pipeline_cli.py
# and published_pipeline_cli.py in this directory) -- recorded here too so a reference
# artifact's "seed" field is self-evidently tied to the actual value used.
SEED = 20260101

REQUIRE_DATA_ENV = "EQGNS_TRAINING_GATE_REQUIRE_DATA"

GPU_ENV = {
    "CUDA_VISIBLE_DEVICES": os.environ.get("EQGNS_TRAINING_GATE_CUDA_DEVICE", "1"),
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def require_d1_dataset():
    """Skip if the real D1 dataset is not reachable on this machine. Strict
    mode (EQGNS_TRAINING_GATE_REQUIRE_DATA=1) turns the skip into a hard
    failure instead, the same shape as training_golden/common.py's
    require_d1_dataset()."""
    if os.path.exists(os.path.join(D1_DATASET_DIR, "train.npz")):
        return
    msg = (f"real D1 dataset not found at {D1_DATASET_DIR} -- this tier "
           f"only runs on a machine that can reach data/gns-sample/ "
           f"(see tests/fixtures/training_gate/common.py, EQGNS_TRAINING_GATE_DATA_DIR)")
    if os.environ.get(REQUIRE_DATA_ENV) == "1":
        pytest.fail(f"{REQUIRE_DATA_ENV}=1 set but {msg}", pytrace=False)
    pytest.skip(msg)


def parse_loss_log(path):
    """Parse meshnet/train.py's loss_log.txt lines into a list of
    {step, train_loss, valid_loss} dicts (same parsing contract as
    tests/fixtures/training_golden/common.py's parse_loss_log, duplicated
    rather than imported so this helper has no cross-tier dependency)."""
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


def run_cli(cli_path, args, timeout=1800):
    """Run one of the two CLI wrappers (current or published oracle) as a
    subprocess on GPU 1, capped CPU threads; raise with full stdout/stderr on
    a non-zero exit so a failure is diagnosable from the pytest output alone."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, **GPU_ENV)
    cmd = [sys.executable, cli_path] + list(args)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                             capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (
        "subprocess failed: " + " ".join(cmd) +
        "\nstdout:\n" + result.stdout[-4000:] + "\nstderr:\n" + result.stderr[-4000:])
    return result


def run_training(model_dir, nsteps, cli_path, lr_scale=1.0, noise_scale=1.0, timeout=1800):
    """Run nsteps training steps of either CLI (cli_path) against the real
    M1 D1 data/config, logging every step (config.json's loss_report_step=1).
    lr_scale/noise_scale != 1.0 perturb a COPY of the config in model_dir
    (never tests/fixtures/training_golden/config.json itself) -- used by the
    falsify case."""
    import json
    import shutil

    require_d1_dataset()
    os.makedirs(model_dir, exist_ok=True)
    if lr_scale == 1.0 and noise_scale == 1.0:
        shutil.copy(TRAINING_GOLDEN_CONFIG, os.path.join(model_dir, "config.json"))
    else:
        with open(TRAINING_GOLDEN_CONFIG) as f:
            cfg = json.load(f)
        cfg["lr_init"] = cfg["lr_init"] * lr_scale
        cfg["noise_std"] = cfg["noise_std"] * noise_scale
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(cfg, f)
    run_cli(cli_path, [
        "--mode=train",
        "--data_path=" + D1_DATASET_DIR + "/",
        "--model_path=" + model_dir + "/",
        "--batch_size=2",
        "--ntraining_steps=" + str(nsteps),
        "--nsave_steps=" + str(nsteps + 1),
    ], timeout=timeout)
    return parse_loss_log(os.path.join(model_dir, "loss_log.txt"))

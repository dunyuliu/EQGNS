"""End-to-end pipeline test (tier 3): the meshnet analogue of EQdyna's
`python3 -m test.testAll` -> verify.test.py.

Runs the real CLI (`python3 -m meshnet.train`, via a seeded test wrapper --
see tests/fixtures/meshnet/seeded_pipeline_cli.py) through: config load ->
train N steps -> checkpoint -> rollout, on a small, deterministic, committed
synthetic dataset (built fresh each run by
tests/fixtures/meshnet/synth.build_dataset with a fixed seed -- not committed
as a binary, since it is 100% reproducible from the seed).

The rollout's predicted velocity field is compared against a committed
golden file (tests/golden/meshnet_e2e_rollout_golden.npz) within a tolerance,
the same pattern as EQdyna.2Dcycle's test_system/verify.test.py.

If this test fails after an intentional change to meshnet/ (e.g. a
numerically-equivalent refactor), regenerate the golden with:
    python3 tests/fixtures/meshnet/generate_golden.py
and note in the commit message that the golden was regenerated and why.
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLI = os.path.join(REPO_ROOT, "tests", "fixtures", "meshnet", "seeded_pipeline_cli.py")
GOLDEN_PATH = os.path.join(REPO_ROOT, "tests", "golden", "meshnet_e2e_rollout_golden.npz")

DATASET_SEED = 555
TIMESTEPS = 24
NTRAINING_STEPS = 8
NSAVE_STEPS = 8
TOLERANCE = 1e-3  # matches EQdyna.2Dcycle's verify.test.py compare_txt_files threshold


def _run_pipeline(tmp_path):
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
    from fixtures.meshnet.synth import build_dataset, write_config

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "rollout"
    data_dir.mkdir()
    write_config(str(model_dir))
    build_dataset(str(data_dir), seed=DATASET_SEED, timesteps=TIMESTEPS)

    env = dict(os.environ, PYTHONPATH=REPO_ROOT)

    train_cmd = [
        sys.executable, CLI, "--mode=train",
        f"--data_path={data_dir}/", f"--model_path={model_dir}/",
        "--batch_size=2", f"--ntraining_steps={NTRAINING_STEPS}",
        f"--nsave_steps={NSAVE_STEPS}",
    ]
    train_result = subprocess.run(train_cmd, cwd=REPO_ROOT, env=env,
                                   capture_output=True, text=True, timeout=90)
    assert train_result.returncode == 0, (
        f"training subprocess failed:\nstdout:\n{train_result.stdout}\n"
        f"stderr:\n{train_result.stderr}")

    checkpoint = model_dir / f"model-{NTRAINING_STEPS}.pt"
    assert checkpoint.exists(), f"expected checkpoint not written: {checkpoint}"

    rollout_cmd = [
        sys.executable, CLI, "--mode=rollout",
        f"--data_path={data_dir}/", f"--model_path={model_dir}/",
        f"--model_file=model-{NTRAINING_STEPS}.pt",
        f"--output_path={output_dir}/", "--rollout_filename=golden",
    ]
    rollout_result = subprocess.run(rollout_cmd, cwd=REPO_ROOT, env=env,
                                     capture_output=True, text=True, timeout=90)
    assert rollout_result.returncode == 0, (
        f"rollout subprocess failed:\nstdout:\n{rollout_result.stdout}\n"
        f"stderr:\n{rollout_result.stderr}")

    with open(output_dir / "golden_0.pkl", "rb") as f:
        return pickle.load(f)


def test_seeded_mini_pipeline_matches_golden_rollout(tmp_path):
    result = _run_pipeline(tmp_path)

    assert os.path.exists(GOLDEN_PATH), (
        "golden file missing; generate it with "
        "tests/fixtures/meshnet/generate_golden.py and commit it")

    golden = np.load(GOLDEN_PATH)

    predicted = result["predicted_rollout"]
    ground_truth = result["ground_truth_rollout"]

    assert predicted.shape == tuple(golden["predicted_rollout_shape"])
    np.testing.assert_allclose(
        predicted, golden["predicted_rollout"], atol=TOLERANCE, rtol=TOLERANCE,
        err_msg="rollout prediction drifted from the committed golden -- either "
                "meshnet/train.py or meshnet/learned_simulator.py changed behaviour, "
                "or the golden needs regenerating (see module docstring)")
    # Ground truth is a pure function of the (fixed-seed) synthetic dataset,
    # not of the model: it must be bit-for-bit stable, i.e. dataset determinism itself.
    np.testing.assert_allclose(
        ground_truth, golden["ground_truth_rollout"], atol=1e-6, rtol=0,
        err_msg="synthetic dataset generator (tests/fixtures/meshnet/synth.py) "
                "is no longer deterministic for a fixed seed")
    assert np.isfinite(predicted).all()

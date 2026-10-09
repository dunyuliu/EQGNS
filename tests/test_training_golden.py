"""Tier 1 training-guard gate (PATHWAY_FORWARD.md row `training-guard`).

N=10 steps of seeded, deterministic training on the REAL published D1
dataset (data/gns-sample/case3.200m.homo.a.Vw/dataset/, PROJECT_RULES.md rule
3 -- a real, gitignored directory, never committed, never symlinked over),
using the REAL M1 architecture (10 message-passing steps, 128 latent dim --
tests/fixtures/training_golden/config.json, copied verbatim from
data/gns-sample/.../models.nmp10.cotopaxi/config.json except for a
loss-logging-cadence override). This is the "N steps of seeded, deterministic
training on real D1 must reproduce a recorded loss curve within tolerance"
tier the board row asks for -- distinct from, and a stronger regression guard
than, the two pre-existing training tests:

- test_meshnet_e2e_golden.py's tiny synthetic CPU golden: an 8-wide/2-layer
  stand-in model on a 12-node mesh, chosen for sub-second runtime, not for
  architectural fidelity.
- test_ab_seeded_determinism.py: proves the harness is bit-reproducible
  across two trees (A vs copy-of-A), but carries no oracle for what the loss
  SHOULD be -- it would not notice if both trees had the same bug.

This tier's reference IS an oracle: a committed loss curve from one actual
call of the real training path, at the real model size, on the real data. A
change to meshnet/train.py's train()/validation() loop, an optimizer
hyperparameter (lr, decay schedule), a loss-weight constant, or the noise
injection will perturb at least one logged step's loss and fail this test.

Mutation evidence (corrected 2026-10-09, see victor-reyes audit MAJOR 1 --
the originally-recorded evidence was misreported, not a harness bug): step 0
is logged from the model's INITIAL weights, BEFORE the first
`optimizer.step()` (meshnet/train.py's `train()` writes the step-N log line
after that step's backward+optimizer.step(), but the loop logs the loss
computed at loop entry -- step 0's forward pass happens before any weight
update has occurred). `lr_init` therefore CANNOT and does NOT change step 0's
loss: a hand-run +10% `lr_init` perturbation reproduced the exact same step-0
train/valid loss, bit-for-bit, as the unperturbed run. The perturbation first
shows up at step 1 (observed delta ~0.0079 on train_loss against this test's
1e-5 tolerance) and persists through every later step. The harness IS
deterministic at step 0 (confirmed, not merely assumed) -- the 1e-5
tolerance and the golden's exact step-0 values are not flaky, they are just
insensitive to this particular mutation by construction. See
tests/test_training_golden_falsify.py (marker `training_golden_falsify`,
`pytest tests/ -m training_golden_falsify -q`) for a self-verifying,
committed re-run of this mutation, rather than relying on a commit
message's numbers.

Runtime: ~130s (CPU-forced single-process; see tests/fixtures/training_golden
/common.py's CUDA_VISIBLE_DEVICES rationale), dominated by loading the real
train.npz/valid.npz (~2.6GB combined), not by the 10 training steps
themselves. Skips (not fails) if data/gns-sample/ is not checked out on this
machine -- see common.require_d1_dataset.

Regeneration (deliberate, reviewed act only -- never to silence a failure
whose cause is not understood, per tests/README.md's golden-file policy):
    python3 tests/fixtures/training_golden/generate_tier1_reference.py
"""
import json
import os
import shutil

import numpy as np
import pytest

from fixtures.training_golden import common

pytestmark = [pytest.mark.training_golden, pytest.mark.slow]

REFERENCE_PATH = os.path.join(common.REPO_ROOT, "tests", "golden", "training_golden_tier1.json")
NTRAINING_STEPS = 10
# Float32 values round-tripped through loss_log.txt's text formatting; CPU
# execution with torch.use_deterministic_algorithms(True) is bit-reproducible
# (proven by test_ab_seeded_determinism.py's exact-equality assertion on the
# same harness), so this tolerance is for text round-trip only, not noise.
TOLERANCE = 1e-5


def _run_pipeline(tmp_path, lr_scale=1.0):
    """Run tier 1's training pipeline. `lr_scale` != 1.0 perturbs the copied
    config's lr_init before the run -- a test-only mutation of a fixture
    COPY in tmp_path, never of tests/fixtures/training_golden/config.json
    itself -- used by the falsify test below."""
    common.require_d1_dataset()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if lr_scale == 1.0:
        shutil.copy(common.CONFIG_SRC, model_dir / "config.json")
    else:
        with open(common.CONFIG_SRC) as f:
            cfg = json.load(f)
        cfg["lr_init"] = cfg["lr_init"] * lr_scale
        with open(model_dir / "config.json", "w") as f:
            json.dump(cfg, f)
    common.run_cli([
        "--mode=train",
        f"--data_path={common.D1_DATASET_DIR}/",
        f"--model_path={model_dir}/",
        "--batch_size=2",
        f"--ntraining_steps={NTRAINING_STEPS}",
        "--nsave_steps=1000",
    ], timeout=600)
    return common.parse_loss_log(str(model_dir / "loss_log.txt"))


def test_training_golden_loss_curve_matches_reference(tmp_path):
    rows = _run_pipeline(tmp_path)

    assert os.path.exists(REFERENCE_PATH), (
        "reference missing; generate it with "
        "tests/fixtures/training_golden/generate_tier1_reference.py and commit it")
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)

    assert len(rows) == NTRAINING_STEPS + 1, (
        f"expected {NTRAINING_STEPS + 1} logged steps (0..{NTRAINING_STEPS}), got "
        f"{len(rows)} -- loss_report_step=1 override (config.json) did not take effect")
    assert [r["step"] for r in rows] == reference["steps"], (
        "logged step numbering diverged from the reference -- the training loop's "
        "step bookkeeping changed")

    train_losses = np.array([r["train_loss"] for r in rows])
    valid_losses = np.array([r["valid_loss"] for r in rows])
    assert np.isfinite(train_losses).all(), f"non-finite train loss: {train_losses}"
    assert np.isfinite(valid_losses).all(), f"non-finite valid loss: {valid_losses}"

    ref_train = np.array(reference["train_loss"])
    ref_valid = np.array(reference["valid_loss"])
    np.testing.assert_allclose(
        train_losses, ref_train, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=False,
        err_msg="per-step TRAIN loss on real D1 drifted from the committed reference -- "
                "either the training path changed behaviour, or the reference needs "
                "regenerating (see module docstring)")
    np.testing.assert_allclose(
        valid_losses, ref_valid, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=False,
        err_msg="per-step VALID loss on real D1 drifted from the committed reference")

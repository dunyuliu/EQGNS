"""Falsify check for the training gate (tests/test_training_gate.py,
PATHWAY_FORWARD.md row test-suite-overhaul sub-item (2)): a planted
regression (lr_init perturbed +10%, the same perturbation and magnitude
tests/test_training_golden_falsify.py already uses for tier 1) must make
the current-vs-published-oracle comparison FAIL. This makes the module's
"the gate is sensitive to a training-hyperparameter regression" claim
self-verifying from the repo itself, the same pattern
tests/paper_parity/gate.py's `falsify` subcommand and
tests/test_training_golden_falsify.py already follow -- never resting on a
commit message's numbers alone.

Separate marker/file from test_training_gate.py's `training_gate` tier, not
folded into it: a planted-regression check validates the gate is
sensitive, it does not gate every PR by itself.
    pytest tests/ -m training_gate_falsify -q
"""
import json

import numpy as np
import pytest

from fixtures.training_gate import common
from test_training_gate import NTRAINING_STEPS, REFERENCE_PATH, TOLERANCE

pytestmark = [pytest.mark.training_gate_falsify, pytest.mark.slow]


def test_training_gate_falsify_lr_init_perturbation(tmp_path):
    """lr_init perturbed +10% on the CURRENT side only (train.py.published's
    own reference trace is untouched) must diverge the train-loss curve past
    TOLERANCE somewhere in steps 1..N -- step 0 is computed from the model's
    initial weights, before the first optimizer.step(), so lr cannot and
    does not change it (same reasoning as tier 1's falsify,
    tests/test_training_golden_falsify.py). If this test fails, the gate has
    stopped being sensitive to the training hyperparameters it claims to
    guard."""
    model_dir = str(tmp_path / "model")
    rows = common.run_training(model_dir, NTRAINING_STEPS, common.CURRENT_CLI, lr_scale=1.1)

    with open(REFERENCE_PATH) as f:
        reference = json.load(f)
    train_losses = np.array([r["train_loss"] for r in rows])
    ref_train = np.array(reference["train_loss"])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(train_losses, ref_train, rtol=TOLERANCE, atol=TOLERANCE)

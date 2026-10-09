"""Falsify check for tier 1's training-guard gate (victor-reyes audit
MAJOR 2, tests/test_training_golden.py): a planted regression (lr_init
perturbed +10%) must make the tier-1 golden comparison FAIL. This makes the
module docstring's mutation-evidence claim self-verifying from the repo
itself, rather than resting on a commit message's numbers (the pattern
tests/paper_parity/gate.py's `falsify` subcommand already follows for the
paper-parity gate).

Deliberately a separate marker/file from tests/test_training_golden.py's
`training_golden` tier, not folded into it: like `gate.py falsify` vs
`gate.py run`, a planted-regression check validates the gate is sensitive,
it does not gate every PR -- `pytest -m training_golden` stays the
~130s/10-step golden-vs-reference comparison documented there; this is its
own opt-in re-run of the same pipeline with a mutated config:
    pytest tests/ -m training_golden_falsify -q
"""
import json

import numpy as np
import pytest

from test_training_golden import REFERENCE_PATH, TOLERANCE, _run_pipeline

pytestmark = [pytest.mark.training_golden_falsify, pytest.mark.slow]


def test_training_golden_falsify_lr_init_perturbation(tmp_path):
    """lr_init perturbed +10% must diverge the tier-1 train-loss curve past
    TOLERANCE somewhere in steps 1..N (step 0 is unaffected by construction:
    it is computed from the model's initial weights, before the first
    optimizer.step() -- see tests/test_training_golden.py's module
    docstring). If this test fails, the tier-1 oracle has stopped being
    sensitive to the training hyperparameters it claims to guard."""
    rows = _run_pipeline(tmp_path, lr_scale=1.1)
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)
    train_losses = np.array([r["train_loss"] for r in rows])
    ref_train = np.array(reference["train_loss"])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(train_losses, ref_train, rtol=TOLERANCE, atol=TOLERANCE)

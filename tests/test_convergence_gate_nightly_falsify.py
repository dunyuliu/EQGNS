"""Falsify check for tier 2's training-guard gate (victor-reyes audit
MAJOR 2, tests/test_convergence_gate_nightly.py): a planted regression
(lr_init perturbed +10%) must make the tier-2 rollout-metrics comparison
FAIL. Mirrors tests/paper_parity/gate.py's `falsify` subcommand, and
tests/test_training_golden_falsify.py's tier-1 equivalent.

Deliberately a separate marker/file from tests/test_convergence_gate_nightly.py's
`convergence_gate_nightly` tier: this validates the gate is sensitive, it
does not gate every PR, and it roughly doubles the already-expensive (~6
min) tier-2 runtime -- run it explicitly:
    pytest tests/ -m convergence_gate_nightly_falsify -q
"""
import json

import pytest

from test_convergence_gate_nightly import REFERENCE_PATH, REL_TOL, _run_pipeline

pytestmark = [pytest.mark.convergence_gate_nightly_falsify, pytest.mark.nightly, pytest.mark.slow]


def test_convergence_gate_nightly_falsify_lr_init_perturbation(tmp_path):
    """lr_init perturbed +10% over the real 30-step training + rollout must
    move at least one of mse_vx/rt_rmse/missed/false past the tier-2 band,
    or change the collapse status. If this test fails (i.e. nothing is
    flagged), the tier-2 oracle would not catch this mutation and its
    tolerance band is too loose to mean anything."""
    current = _run_pipeline(tmp_path, lr_scale=1.1)
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)
    ref = reference["metrics"]

    bad = []
    for key in ("mse_vx", "rt_rmse"):
        delta = abs(current[key] - ref[key])
        bound = REL_TOL * max(abs(ref[key]), 1.0)
        if not (delta <= bound):
            bad.append(f"{key} {ref[key]:.6g} -> {current[key]:.6g}")
    for key in ("missed", "false"):
        if current[key] != ref[key]:
            bad.append(f"{key} {ref[key]} -> {current[key]}")
    if current.get("collapsed") != ref.get("collapsed"):
        bad.append(f"collapsed {ref.get('collapsed')} -> {current.get('collapsed')}")
    assert bad, (
        "lr_init perturbed +10% did not move any tier-2 rollout metric (or "
        "the collapse status) past the committed band -- the tier-2 oracle "
        "would not catch this mutation")

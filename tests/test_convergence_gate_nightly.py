"""Tier 2 training-guard gate: `convergence-gate-nightly`
(PATHWAY_FORWARD.md row `training-guard`, and the separate
`convergence-gate-nightly` board row it folds in).

HONESTY NOTE (victor-reyes audit MAJOR 4, resolved 2026-10-09): despite the
marker/board name, this is a REGRESSION-DETERMINISM golden for a 30-step,
EXPECTED-TO-BE-COLLAPSED checkpoint, not a test that the model actually
converges. gate.py's own `compare()` treats `collapsed` as an automatic FAIL
(gate.py:253-255, COLLAPSE_TOL=0.5 -- a flat/degenerate forecast); this
tier's committed reference has `"collapsed": true` by construction (30 steps
from scratch is nowhere near enough budget for this architecture to escape
a degenerate forecast -- see "Step/frame budget" below), so a real
convergence check cannot live at this step count on CPU. This test
therefore does NOT reuse gate.py's collapse-fails rule; instead (see below)
it explicitly asserts the CURRENT run's collapse status matches the
reference's recorded (collapsed) status, so a future change that causes
this checkpoint to stop collapsing -- itself a notable behaviour change at
this tiny step budget -- is reported by name instead of silently passing
through the mse_vx/rt_rmse/missed/false comparison alone. A real
convergence gate (model trained far enough to NOT collapse, asserting
non-collapse) is recommended as a release-scale follow-up -- see
"Step/frame budget" below and tests/README.md.

A small step budget of real training on the real D1 dataset
(data/gns-sample/case3.200m.homo.a.Vw/dataset/), from scratch (seeded init,
no resume from the published 3M-step checkpoint), followed by a real rollout
on a (CPU-time-budget-truncated) real D1 test trajectory, scored with the
SAME metrics tests/paper_parity/gate.py uses for the paper-parity gate
(mse_vx, rt_rmse, missed, false -- rupture-time convention: SLIPRATE_THRESHOLD
0.1 m/s, dt=0.0167777s, per PROJECT_RULES.md rule 7), compared against a
committed reference within a tolerance band.

Why this is a DIFFERENT gate from tier 1 (test_training_golden.py): tier 1
only ever looks at the loss number train() reports -- a loss-function bug
that still produces a plausible-looking scalar loss would not necessarily
show up there. This tier runs the trained (if barely-trained) model through
an actual autoregressive rollout and checks it against the physical
rupture-detection metrics the paper-parity gate uses, so a bug in
predict_acceleration/predict_velocity, the rollout loop's boundary-condition
masking, or the rupture-time convention itself has a chance to show up here
even if train()'s reported loss scalar looks unchanged.

Step/frame budget (deliberately small, for CPU runtime -- see "Deviations
from brief" in the session report): NTRAINING_STEPS=30, rolled out over the
first ROLLOUT_FRAMES=40 frames of trajectory0 (this case's D1 hypocenters
already exceed the 0.1 m/s rupture threshold within the first few frames --
verified by hand against the raw ground-truth velocity before picking this
window -- so a 40-frame window is NOT a degenerate all-zero rupture-time
case). A nightly/release-scale version (hundreds-to-thousands of steps, the
full 6-trajectory/827-frame test set, GPU) is recommended as a follow-up
(see tests/README.md) -- this implementation is real and runs end-to-end,
just at a session-tractable scale, per the brief's "implement what you can
within a CI-reasonable runtime" instruction.

Marked `nightly`/`slow`: NOT part of the fast local loop
(`-m "not slow"`), and not intended to gate every PR -- run it explicitly:
    pytest tests/ -m convergence_gate_nightly -q

Falsify check (victor-reyes audit MAJOR 2): tests/test_convergence_gate_nightly_falsify.py
(marker `convergence_gate_nightly_falsify`) re-runs this pipeline with
lr_init perturbed +10% and asserts the comparison above FAILS -- run it
explicitly:
    pytest tests/ -m convergence_gate_nightly_falsify -q

Regeneration (deliberate, reviewed act only):
    python3 tests/fixtures/training_golden/generate_tier2_reference.py
"""
import json
import os
import shutil

import pytest

from fixtures.training_golden import common
from paper_parity import gate

pytestmark = [pytest.mark.convergence_gate_nightly, pytest.mark.nightly, pytest.mark.slow]

REFERENCE_PATH = os.path.join(common.REPO_ROOT, "tests", "golden", "convergence_gate_nightly_tier2.json")
NTRAINING_STEPS = 30
TRAJECTORY_KEY = "trajectory0"
ROLLOUT_FRAMES = 40  # truncation of the real 827-frame trajectory; see module docstring
# Both runs are CPU-forced + torch.use_deterministic_algorithms(True) (same harness as
# tier 1 and test_ab_seeded_determinism.py), so this is a tight tolerance around a
# reproducible reference, not a statistical noise band -- unlike gate.py's REL_TOL,
# which exists specifically to absorb GPU nondeterminism (see tests/paper_parity/README.md).
REL_TOL = 1e-4


def _run_pipeline(tmp_path, lr_scale=1.0):
    """Run tier 2's training+rollout pipeline. `lr_scale` != 1.0 perturbs the
    copied config's lr_init before the run -- a test-only mutation of a
    fixture COPY in tmp_path, never of tests/fixtures/training_golden/
    config.json itself -- used by test_convergence_gate_nightly_falsify.py."""
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
        f"--nsave_steps={NTRAINING_STEPS}",
    ], timeout=1800)

    checkpoint = model_dir / f"model-{NTRAINING_STEPS}.pt"
    assert checkpoint.exists(), f"expected checkpoint not written: {checkpoint}"

    rollout_data_dir = tmp_path / "rollout_data"
    rollout_data_dir.mkdir()
    common.write_truncated_trajectory_npz(
        os.path.join(common.D1_DATASET_DIR, "test.npz"),
        str(rollout_data_dir / "test.npz"),
        TRAJECTORY_KEY, ROLLOUT_FRAMES)

    output_dir = tmp_path / "rollout_out"
    common.run_cli([
        "--mode=rollout",
        f"--data_path={rollout_data_dir}/",
        f"--model_path={model_dir}/",
        f"--model_file=model-{NTRAINING_STEPS}.pt",
        f"--output_path={output_dir}/",
        "--rollout_filename=nightly",
    ], timeout=600)

    import pickle
    with open(output_dir / "nightly_0.pkl", "rb") as f:
        pkl = pickle.load(f)
    return gate.metrics(pkl)


def test_convergence_gate_nightly_rollout_metrics_within_band(tmp_path):
    current = _run_pipeline(tmp_path)

    assert os.path.exists(REFERENCE_PATH), (
        "reference missing; generate it with "
        "tests/fixtures/training_golden/generate_tier2_reference.py and commit it")
    with open(REFERENCE_PATH) as f:
        reference = json.load(f)
    ref = reference["metrics"]

    bad = []
    for key in ("mse_vx", "rt_rmse"):
        delta = abs(current[key] - ref[key])
        bound = REL_TOL * max(abs(ref[key]), 1.0)
        # Written as "not (delta <= bound)", never "delta > bound": a NaN
        # delta (e.g. a collapsed/diverged rollout) must FAIL this gate, and
        # `nan > bound` is False in Python/numpy while `not (nan <= bound)` is True.
        if not (delta <= bound):
            bad.append(f"{key} {ref[key]:.6g} -> {current[key]:.6g} (delta {delta:.3g} > {bound:.3g})")
    for key in ("missed", "false"):
        if current[key] != ref[key]:
            bad.append(f"{key} {ref[key]} -> {current[key]}")
    # This tier is a regression-determinism golden for an EXPECTED-COLLAPSED
    # checkpoint (see module docstring, victor-reyes audit MAJOR 4) -- unlike
    # gate.py's own compare(), it does not auto-fail on `collapsed`. Instead,
    # assert the collapse STATUS itself matches the reference explicitly: a
    # future change that makes this 30-step checkpoint stop collapsing (or
    # start collapsing, if the reference is ever regenerated non-collapsed)
    # is a real behaviour change at this step budget and must be reported by
    # name, not silently absorbed into the mse_vx/rt_rmse/missed/false diff.
    if current.get("collapsed") != ref.get("collapsed"):
        bad.append(f"collapsed {ref.get('collapsed')} -> {current.get('collapsed')} "
                    f"(var_ratio {ref.get('var_ratio'):.3g} -> {current.get('var_ratio', float('nan')):.3g})")
    assert not bad, (
        "rollout metrics on real D1 (small-budget training + truncated test "
        "trajectory) drifted from the committed band -- either the training/rollout "
        "path changed behaviour, or the reference needs regenerating (see module "
        "docstring):\n  " + "\n  ".join(bad))

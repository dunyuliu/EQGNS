"""Unit tests for gate.py's `regression_ok()` -- the fast-rollout-vs-
reference-rollout gate's pass/fail decision (PATHWAY_FORWARD.md
`release-gate-decisions-pending` row (a), owner-approved 2026-10-09).

Pure logic, no GPU / checkpoints / real rollouts: constructs the
compare_pair()-shaped row dicts directly and checks the threshold/exclusion
decision in isolation. Always runs in CI (no opt-in marker, no data
dependency) -- the GPU-heavy end-to-end path (`gate.py regression`, which
actually runs rollouts) stays opt-in under `tests/paper_parity/test_paper_parity.py`'s
`--paper-parity` gate.
"""
from paper_parity import gate


def row(traj=0, delta_rt_rmse_s=0.0, delta_mw=0.0, missed=0, false=0):
    return {"traj": traj, "delta_rt_rmse_s": delta_rt_rmse_s, "delta_mw": delta_mw,
            "missed": missed, "false": false}


def test_pass_within_default_thresholds():
    assert gate.regression_ok("M1_D1", row(delta_rt_rmse_s=3 * gate.DT, delta_mw=0.02)) is True


def test_fail_rt_rmse_over_default_threshold():
    assert gate.regression_ok("M1_D1", row(delta_rt_rmse_s=4.01 * gate.DT)) is False


def test_fail_mw_over_default_threshold():
    assert gate.regression_ok("M1_D1", row(delta_mw=0.031)) is False


def test_fail_any_missed_or_false():
    assert gate.regression_ok("M1_D1", row(missed=1)) is False
    assert gate.regression_ok("M1_D1", row(false=1)) is False


def test_boundary_is_inclusive():
    assert gate.regression_ok("M1_D1", row(delta_rt_rmse_s=4 * gate.DT, delta_mw=0.03)) is True


def test_m2_d3_excluded_regardless_of_values():
    # Owner decision (a): M2_D3 is reported, never gated -- even a wildly
    # failing row must come back None (excluded), not False.
    assert gate.regression_ok("M2_D3", row(delta_rt_rmse_s=100.0, delta_mw=5.0, missed=99)) is None


def test_m3_d3_traj7_excluded_other_trajs_gated():
    assert gate.regression_ok("M3_D3", row(traj=7, delta_mw=5.0)) is None
    assert gate.regression_ok("M3_D3", row(traj=0, delta_mw=5.0)) is False


def test_tight_tier_is_stricter():
    r = row(delta_rt_rmse_s=3.5 * gate.DT, delta_mw=0.025)
    assert gate.regression_ok("M1_D1", r, tier="default") is True
    assert gate.regression_ok("M1_D1", r, tier="tight") is False


def test_falsify_acceptance_fixture_fails_under_default_tier():
    """Planted-regression shape measured 2026-10-08 for M2_D3 (an *excluded*
    case) is not a stand-in for the real falsify check -- this only pins the
    threshold arithmetic against a concrete, documented-scale delta so a
    future edit to REGRESSION_TOL_* can't silently loosen it unnoticed. The
    real falsify acceptance check (`gate.py regression --falsify`, weights
    x1.005, on real checkpoints/GPU) is the actual gate for merge and is not
    reproduced here."""
    assert gate.regression_ok("M1_D1", row(delta_rt_rmse_s=6 * gate.DT)) is False

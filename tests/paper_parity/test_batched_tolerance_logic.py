"""Unit tests for gate.py's REL_TOL_BATCHED -- the looser tolerance for the
`rollout_batched()` path (`--rollout-batch-size` > 1 on `gate.py run`/
`falsify`), PATHWAY_FORWARD.md board row `rollout-batched-oracle-gap`, owner
decision `release-gate-decisions-pending` item (3): "accept a looser
tolerance for the batched path only (default batch=1 path keeps 1e-4)".

Pure logic, no GPU / checkpoints / real rollouts: constructs the
gate.metrics()-shaped row dicts directly and checks `compare()`'s pass/fail
decision in isolation, at both `REL_TOL` (default/eager path, unchanged) and
`REL_TOL_BATCHED` (new). The real end-to-end evidence -- the batch=15
measurement across all 8 gated cases and the falsify acceptance check
(weights x1.005 must FAIL under REL_TOL_BATCHED) -- is run manually via
`gate.py run --rollout-batch-size 15` / `gate.py falsify --rollout-batch-size
15`, not reproduced here (see tests/paper_parity/README.md)."""
from paper_parity import gate


def row(mse_vx=0.0, rt_rmse=0.0, missed=0, false=0, var_ratio=1.0):
    return {"mse_vx": mse_vx, "rt_rmse": rt_rmse, "missed": missed, "false": false,
            "var_ratio": var_ratio, "collapsed": var_ratio < gate.COLLAPSE_TOL}


def test_rel_tol_batched_is_strictly_looser_than_rel_tol():
    assert gate.REL_TOL_BATCHED > gate.REL_TOL


def test_default_rel_tol_unchanged():
    # A tiny (2e-4 relative) delta already exceeds REL_TOL (1e-4): the
    # default/eager batch=1 path must still catch it, same as before this
    # change -- REL_TOL itself is untouched.
    ref = [row(mse_vx=1.0)]
    cur = [row(mse_vx=1.0002)]
    assert not gate.compare("M1_D1", cur, ref, rel_tol=gate.REL_TOL, quiet=True)


def test_real_benign_batched_delta_passes_under_rel_tol_batched():
    # M1_D1 traj 4, measured 2026-10-09 (batch=15 vs reference.json): the
    # originally-diagnosed benign-float-reassociation gap this tolerance
    # exists to cover.
    ref = [row(mse_vx=1.150259814180746)]
    cur = [row(mse_vx=1.1857688180139472)]
    assert gate.compare("M1_D1", cur, ref, rel_tol=gate.REL_TOL_BATCHED, quiet=True)


def test_real_benign_batched_delta_would_fail_under_default_rel_tol():
    # Same pair as above: confirms REL_TOL_BATCHED is doing real work, not a
    # no-op -- this delta needed the looser tolerance to pass.
    ref = [row(mse_vx=1.150259814180746)]
    cur = [row(mse_vx=1.1857688180139472)]
    assert not gate.compare("M1_D1", cur, ref, rel_tol=gate.REL_TOL, quiet=True)


def test_real_falsify_delta_still_fails_under_rel_tol_batched():
    # M1_D1 traj 2, measured 2026-10-09 (weights x1.005, batch=15 falsify
    # acceptance check): missed 0->54, a real planted regression. Pins the
    # threshold arithmetic against this documented falsify-scale delta so a
    # future edit to REL_TOL_BATCHED can't silently loosen it past the point
    # where it stops catching a real regression -- same pattern as
    # test_regression_gate_logic.py's falsify-fixture test. The actual
    # falsify acceptance check (`gate.py falsify M1_D1 --rollout-batch-size
    # 15`, real checkpoints/GPU) is the real gate for merge, not reproduced
    # here.
    ref = [row(mse_vx=0.726738, rt_rmse=0.469297, missed=0)]
    cur = [row(mse_vx=0.347491, rt_rmse=0.231162, missed=54)]
    assert not gate.compare("M1_D1", cur, ref, rel_tol=gate.REL_TOL_BATCHED, quiet=True)

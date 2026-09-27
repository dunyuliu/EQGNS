"""Planted-regression falsifiability test for the TRUNCATED-HORIZON 'quick'
tier (PR #4). Same A/B tree-copy pattern and same planted defect as
test_falsifiability.py (PR #3) -- sign-flip on cached_edge_attr,
meshnet/train.py:123 -- but run through
test/fixtures/paper_parity/truncated_rollout_cli.py at N=100 steps
(instead of the full 826) and diffed against
baseline_M1_truncated100.json / per_trajectory_tolerance_truncated.json.

Purpose: prove the quick tier is not just faster but still CATCHES the
same class of regression the full-length gate catches -- a quick tier
that runs fast but can't discriminate a real defect would be worse than
no quick tier (false confidence). See NOTES_pr4.md for the measured
before/after numbers this test's assertions are based on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import gns_sample_available  # noqa: E402
from measure_truncated_spread import run_once  # noqa: E402
from run_gate import diff_trajectory, tolerance_for  # noqa: E402
from test_falsifiability import _TARGET_LINE, _PERTURBED_LINE, _make_tree  # noqa: E402
from test_paper_parity import _skip_reason  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_KEY = "M1"
NSTEPS = 100
CUDA_DEVICE = 1


def _rollout_and_diff(tree_root, tmp_work_dir):
    trajectories, elapsed = run_once(
        MODEL_KEY, NSTEPS, tmp_work_dir, run_idx=0, cuda_device=CUDA_DEVICE,
        meshnet_src_root=tree_root)

    with open(HERE / f"baseline_{MODEL_KEY}_truncated{NSTEPS}.json") as f:
        baseline = json.load(f)
    with open(HERE / "per_trajectory_tolerance_truncated.json") as f:
        truncated_tol = json.load(f)

    rows = {}
    for b_traj in baseline["trajectories"]:
        pkl_file = b_traj["pkl_file"]
        c_traj = trajectories[pkl_file]
        this_tol, source = tolerance_for(
            f"{MODEL_KEY}_truncated{NSTEPS}", pkl_file, {}, truncated_tol)
        diffs = diff_trajectory(b_traj, c_traj, this_tol)
        traj_ok = all(v[0] for v in diffs.values())
        rows[pkl_file] = (traj_ok, diffs, source)
    return rows, elapsed


@pytest.mark.paper_parity
@pytest.mark.e2e
@pytest.mark.slow
def test_truncated_null_hypothesis_unmodified_copy_passes(request, tmp_path):
    reason = _skip_reason(request)
    if reason is not None:
        pytest.skip(reason)

    tree = _make_tree(tmp_path, "null_tree_truncated", perturb=False)
    rows, elapsed = _rollout_and_diff(tree, tmp_path / "null_rollout_truncated")

    failures = {pkl: diffs for pkl, (ok, diffs, _src) in rows.items() if not ok}
    assert not failures, (
        f"quick-tier null-hypothesis (unmodified copy, N={NSTEPS}) unexpectedly FAILED "
        f"({elapsed:.1f}s): {failures} -- this means the truncated gate/tolerance is "
        f"broken, not that a regression was found")


@pytest.mark.paper_parity
@pytest.mark.e2e
@pytest.mark.slow
def test_truncated_planted_regression_is_caught(request, tmp_path):
    reason = _skip_reason(request)
    if reason is not None:
        pytest.skip(reason)

    tree = _make_tree(tmp_path, "perturbed_tree_truncated", perturb=True)
    rows, elapsed = _rollout_and_diff(tree, tmp_path / "perturbed_rollout_truncated")

    failures = {pkl: diffs for pkl, (ok, diffs, _src) in rows.items() if not ok}
    n_total = len(rows)
    assert failures, (
        f"planted regression (edge-feature sign flip) was NOT caught by any of the "
        f"{n_total} trajectories at the TRUNCATED (N={NSTEPS}) horizon ({elapsed:.1f}s) "
        f"-- the quick tier is not sensitive enough to a real code regression of this "
        f"size; this is a gate defect, report it, do not loosen this test to pass")

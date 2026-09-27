"""Planted-regression falsifiability test for the per-trajectory paper-parity
gate (PR #3, task 3).

Same A/B tree-copy pattern as test/test_ab_seeded_determinism.py: builds a
copy of THIS worktree's `meshnet/` + `gns/` packages under a temp dir (never
touches the checked-out originals), runs the CURRENT gate machinery
(`run_gate.diff_trajectory` + the per-trajectory tolerance from PR #3 task 1)
against rollout output produced from that copy, and checks two things:

1. Null hypothesis: an UNMODIFIED copy of meshnet/ produces a rollout that
   still PASSES the per-trajectory gate against the M1 baseline (proves the
   gate isn't accidentally failing everything).
2. A COPY with one small, controlled, real code change planted (sign flip on
   the cached edge features used for every GNN message-passing step in
   rollout(), meshnet/train.py:123) FAILS the same gate -- proves the gate
   actually catches a regression of this size, rather than passing for the
   wrong reason (e.g. a tolerance so loose it can't discriminate).

Never edits meshnet/train.py in this worktree -- only a throwaway copy under
tmp_path. See NOTES_pr3.md 'Falsifiability test' for the before/after numbers
observed when this was run.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import gns_sample_available, load_pkl, per_trajectory_metrics  # noqa: E402
from measure_spread import run_once  # noqa: E402
from run_gate import diff_trajectory, load_per_trajectory_tolerance, load_tolerance, tolerance_for  # noqa: E402
from test_paper_parity import _skip_reason  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "M1"
CUDA_DEVICE = int(os.environ.get("FALSIFIABILITY_CUDA_DEVICE", "1"))

# The planted defect: sign-flip the cached edge features (meshnet/train.py:123).
# These feed simulator.predict_velocity() at every one of the 826 autoregressive
# rollout steps, for every node -- a real, small, single-line code change with
# repo-wide (not just this-model) applicability, not a training-loop-specific
# hack.
_TARGET_LINE = "    cached_edge_attr = template_graph.edge_attr  # Positions don't move!"
_PERTURBED_LINE = "    cached_edge_attr = -template_graph.edge_attr  # PLANTED DEFECT (test_falsifiability.py): sign flip"


def _make_tree(tmp_path, name, perturb):
    tree = tmp_path / name
    dest_meshnet = tree / "meshnet"
    shutil.copytree(REPO_ROOT / "meshnet", dest_meshnet,
                     ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPO_ROOT / "gns", tree / "gns",
                     ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    train_py = dest_meshnet / "train.py"
    text = train_py.read_text()
    assert _TARGET_LINE in text, (
        "planted-defect anchor line not found in meshnet/train.py -- the file "
        "moved/changed since this test was written; update _TARGET_LINE, do "
        "not loosen this assertion")
    if perturb:
        text = text.replace(_TARGET_LINE, _PERTURBED_LINE)
        assert _PERTURBED_LINE in text
    train_py.write_text(text)
    return tree


def _rollout_and_diff(tree_root, tmp_work_dir):
    trajectories, elapsed = run_once(
        MODEL_KEY, tmp_work_dir, run_idx=0, cuda_device=CUDA_DEVICE,
        deterministic=False, meshnet_src_root=tree_root)

    baseline_path = Path(__file__).resolve().parent / f"baseline_{MODEL_KEY}.json"
    import json
    with open(baseline_path) as f:
        baseline = json.load(f)

    global_tol = load_tolerance()
    per_traj = load_per_trajectory_tolerance()

    rows = {}
    for b_traj in baseline["trajectories"]:
        pkl_file = b_traj["pkl_file"]
        c_traj = trajectories[pkl_file]
        this_tol, source = tolerance_for(MODEL_KEY, pkl_file, global_tol, per_traj)
        diffs = diff_trajectory(b_traj, c_traj, this_tol)
        traj_ok = all(v[0] for v in diffs.values())
        rows[pkl_file] = (traj_ok, diffs, source)
    return rows, elapsed


@pytest.mark.paper_parity
@pytest.mark.e2e
@pytest.mark.slow
def test_null_hypothesis_unmodified_copy_passes(request, tmp_path):
    """An unmodified copy of meshnet/ must reproduce the M1 baseline within
    the per-trajectory tolerance -- proves the gate isn't failing by
    construction (e.g. a path bug in the copy machinery itself)."""
    reason = _skip_reason(request)
    if reason is not None:
        pytest.skip(reason)

    tree = _make_tree(tmp_path, "null_tree", perturb=False)
    rows, elapsed = _rollout_and_diff(tree, tmp_path / "null_rollout")

    failures = {pkl: diffs for pkl, (ok, diffs, _src) in rows.items() if not ok}
    assert not failures, (
        f"null-hypothesis (unmodified copy) unexpectedly FAILED the gate "
        f"({elapsed:.1f}s): {failures} -- this means the gate or tolerance "
        f"is broken, not that a regression was found")


@pytest.mark.paper_parity
@pytest.mark.e2e
@pytest.mark.slow
def test_planted_regression_is_caught(request, tmp_path):
    """A copy with the edge-feature sign-flip planted must FAIL the gate on
    at least one trajectory -- proves the per-trajectory tolerance (task 1)
    is tight enough to catch a real, repo-wide regression, not just wide
    enough to swallow measured GPU nondeterminism."""
    reason = _skip_reason(request)
    if reason is not None:
        pytest.skip(reason)

    tree = _make_tree(tmp_path, "perturbed_tree", perturb=True)
    rows, elapsed = _rollout_and_diff(tree, tmp_path / "perturbed_rollout")

    failures = {pkl: diffs for pkl, (ok, diffs, _src) in rows.items() if not ok}
    n_total = len(rows)
    assert failures, (
        f"planted regression (edge-feature sign flip) was NOT caught by any "
        f"of the {n_total} trajectories' per-trajectory tolerance ({elapsed:.1f}s) "
        f"-- the gate is not sensitive enough to a real code regression of this "
        f"size; this is a gate defect, report it, do not loosen this test to "
        f"pass")

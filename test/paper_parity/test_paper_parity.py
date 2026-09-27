"""pytest wrapper around run_gate.py -- the tier-1/4 paper-parity gate.

NOT part of the default `pytest test/ -q` run: gns-sample/ (249GB, real
published checkpoints) does not exist in CI, and even locally these are
full 826-step, all-trajectory rollouts (minutes per model on an A100 --
see test/paper_parity/README.md for observed wall-clock). Opt in with:

    pytest test/paper_parity -m paper_parity --paper-parity -q -s

Every skip below states its reason explicitly (PROJECT_RULES.md /
mission mandate: no silent "skip if file missing" without explanation).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ALL_REGISTRY, gns_sample_available  # noqa: E402
from run_gate import (  # noqa: E402
    diff_trajectory, load_per_trajectory_tolerance, load_tolerance, run_gate_for_model)

HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.paper_parity


def _skip_reason(request):
    if not request.config.getoption("--paper-parity"):
        return (
            "paper-parity gate not requested: this is a full-length "
            "(826-step, all-trajectory) rollout re-run against gns-sample/ "
            "(249GB, real published checkpoints), taking minutes per model "
            "on a GPU. Opt in explicitly with --paper-parity. See "
            "test/paper_parity/README.md.")
    if not gns_sample_available():
        return (
            "gns-sample/ is not available (missing or empty directory) in "
            "this checkout -- the paper-parity gate cannot run without the "
            "real published checkpoints/rollouts/test sets. This is "
            "expected in CI and in fresh checkouts; symlink gns-sample/ in "
            "from the data host to run this locally. See "
            "test/paper_parity/README.md Step 0.")
    return None


# PR #3: M3's xfail (below, previously unconditional for every M3 run) was
# REMOVED after building a per-trajectory tolerance (see
# generate_per_trajectory_tolerance.py, NOTES_pr3.md "Task 1") specifically
# to absorb the rollout_2/rollout_7 chaos-amplified GPU nondeterminism that
# motivated the original xfail. Verified end-to-end on a FRESH, independent
# run (not one of the runs used to derive the tolerance): M3 PASSES cleanly,
# all 15 trajectories, all 6 metrics (`NOTES_pr3.md` "Re-verified end-to-end
# on a fresh ... run"). Leaving the OLD unconditional strict xfail in place
# would now itself be a bug: pytest reports a strict-xfail test that
# actually passes as an XPASS FAILURE, so this test file would go red for
# every future run under exactly the scenario this PR was built to fix.
# If a future run genuinely re-exceeds the new per-trajectory tolerance,
# that is a real, visible FAIL again -- not silently re-xfailed.
_PARAMS = list(ALL_REGISTRY)


@pytest.mark.parametrize("model_key", _PARAMS)
def test_paper_parity_gate(request, model_key, tmp_path):
    reason = _skip_reason(request)
    if reason is not None:
        pytest.skip(reason)

    baseline_path = HERE / f"baseline_{model_key}.json"
    if not baseline_path.exists():
        pytest.skip(
            f"baseline_{model_key}.json not found -- run "
            f"`python3 test/paper_parity/extract_baselines.py --model {model_key}` "
            f"first. This is a missing-fixture condition, not a code failure.")

    tol = load_tolerance()
    per_traj = load_per_trajectory_tolerance()
    ok, rows, elapsed = run_gate_for_model(model_key, tol, tmp_path, per_traj=per_traj)

    failures = [(pkl, diffs) for pkl, traj_ok, diffs, _source in rows if not traj_ok]
    if failures:
        lines = [f"{model_key} paper-parity gate FAILED for {len(failures)}/{len(rows)} trajectories "
                 f"(wall-clock {elapsed:.1f}s):"]
        for pkl_file, diffs in failures:
            bad = {k: v for k, v in diffs.items() if not v[0]}
            lines.append(f"  {pkl_file}: {bad}")
        pytest.fail("\n".join(lines))

    assert ok

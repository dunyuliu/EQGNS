"""Data-prep guard (tier: dataprep) for utils/prepare.eqdyna.4gns.py.

Regression test for the shipped bug (fixed in PR #5 / commit 1afb989):
`timestep -= nskip` followed by `for i in range(timestep - nskip):` silently
left the last `nskip` output frames all-zero across pos/cells/velocity/
node_type/node_property (827-frame real run -> last 72 frames zero). The
fixed code is `for i in range(timestep):` (timestep already reduced once).

Uses a tiny synthetic 4-node, 13-frame EQdyna-output case built on the fly
by test/fixtures/dataprep/synth_case.py -- never gns-sample or any real
scenario dataset.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fixtures.dataprep.synth_case import (  # noqa: E402
    build_case, load_prepare_module, run_split_block,
    velocity_value, NROW, NSKIP, TIMESTEP_AFTER,
)

pytestmark = pytest.mark.dataprep

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "utils" / "prepare.eqdyna.4gns.py"

ASP_LIST = [[-1e6, -1e6, 3, 0.25, 1.0]]  # asperity far outside the domain ->
# every node gets the background normalized stress 0.25 (never 0.0), so an
# untouched (bug) frame's node_property is distinguishable from a real one.


@pytest.fixture
def case_dir(tmp_path, monkeypatch):
    case_name = build_case(tmp_path)
    monkeypatch.chdir(tmp_path)
    return case_name


@pytest.fixture
def prepare_module():
    return load_prepare_module(SCRIPT_PATH, "prepare_eqdyna_4gns_under_test")


def _run(prepare_module, case_dir):
    import matplotlib.pyplot as plt
    _particle, meshnet, *_ = prepare_module.create_train_data(
        case_dir, ASP_LIST, fault_boundary_node_type_mask=True)
    plt.close("all")  # genMapsForEQDYNA opens one figure per frame
    return meshnet


def test_frame_count_matches_expected(prepare_module, case_dir):
    meshnet = _run(prepare_module, case_dir)
    # Derived independently from the fixture's own dt=1.0/nskip=1 constants
    # (see synth_case.TIMESTEP_AFTER), not from the production formula.
    assert meshnet["pos"].shape[0] == TIMESTEP_AFTER == 13


def test_shapes_and_dtypes(prepare_module, case_dir):
    meshnet = _run(prepare_module, case_dir)
    assert meshnet["pos"].shape == (TIMESTEP_AFTER, NROW, 2)
    assert meshnet["pos"].dtype == np.float32
    assert meshnet["velocity"].shape == (TIMESTEP_AFTER, NROW, 2)
    assert meshnet["velocity"].dtype == np.float32
    assert meshnet["node_type"].shape == (TIMESTEP_AFTER, NROW, 1)
    assert meshnet["node_type"].dtype == np.int32
    assert meshnet["node_property"].shape == (TIMESTEP_AFTER, NROW, 1)
    assert meshnet["node_property"].dtype == np.float32
    assert meshnet["cells"].dtype == np.int32


def test_no_all_zero_frames(prepare_module, case_dir):
    """The planted-bug regression test: a frame past the truncated range is
    never written and stays at its np.zeros() initial value across every one
    of these fields. Flip this by mutating the code under test to the
    pre-fix `for i in range(timestep - nskip):` and this fails (verified
    manually, see PR description)."""
    meshnet = _run(prepare_module, case_dir)
    for field in ("pos", "cells", "velocity", "node_type", "node_property"):
        arr = meshnet[field]
        for frame_idx in range(arr.shape[0]):
            assert not np.all(arr[frame_idx] == 0), (
                f"{field}[{frame_idx}] is entirely zero -- "
                f"frame was never written (zero-tail regression)")


def test_nskip_offset_maps_raw_file_to_frame(prepare_module, case_dir):
    """frame i must come from raw file src<i+1+nskip>.txt -- check the exact
    synthetic slip-rate value baked into src_evol0 at that (timestep,
    station) pair, for every frame and every station."""
    meshnet = _run(prepare_module, case_dir)
    velocity = meshnet["velocity"]
    for i in range(velocity.shape[0]):
        expected = [velocity_value(i + 1 + NSKIP, j) for j in range(NROW)]
        np.testing.assert_allclose(
            velocity[i, :, 0], expected, atol=1e-5,
            err_msg=f"frame {i} does not match raw file src{i + 1 + NSKIP}.txt")


def test_train_valid_test_split_is_disjoint_with_expected_counts():
    """Runs the literal seed(42)/shuffle/slice source lines from the
    `if case=='3':` block (sliced out because the loop that follows them
    calls create_train_data() against real, not-checked-out scenario
    directories) and checks the resulting 20-model split."""
    namespace = run_split_block(
        SCRIPT_PATH, "if case=='3':", "for key in set_list:", "3")
    train_idx = namespace["train_set_idx"]
    valid_idx = namespace["valid_set_idx"]
    test_idx = namespace["test_set_idx"]

    assert len(train_idx) == 14
    assert len(valid_idx) == 3
    assert len(test_idx) == 3
    assert set(train_idx).isdisjoint(valid_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(valid_idx).isdisjoint(test_idx)
    assert sorted(train_idx + valid_idx + test_idx) == list(range(20))

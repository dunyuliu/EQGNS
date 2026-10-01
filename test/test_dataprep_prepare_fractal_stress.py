"""Data-prep guard (tier: dataprep) for
utils/prepare.fractal.stress.eqdyna.4gns.py.

This script shares the same create_train_data() nskip/timestep logic as
utils/prepare.eqdyna.4gns.py (and the same shipped-then-fixed zero-tail bug
class -- PR #5 / commit 1afb989), with node_property sourced from a
fractal_stress.txt lookup instead of an asperity list, and node_type always
hard-set to 0 (no fault_boundary_node_type_mask parameter in this script --
see the note on test_no_all_zero_frames below).

Uses a tiny synthetic 4-node, 13-frame EQdyna-output case built on the fly
by test/fixtures/dataprep/synth_case.py -- never gns-sample or any real
scenario dataset.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fixtures.dataprep.synth_case import (  # noqa: E402
    build_case, load_prepare_module, velocity_value, NROW, NSKIP, TIMESTEP_AFTER,
)

pytestmark = pytest.mark.dataprep

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "utils" / "prepare.fractal.stress.eqdyna.4gns.py"

# Normalized shear stress expected at each of the 4 fixture stations, derived
# by hand from synth_case.write_fractal_stress()'s 40/41/42/43 MPa rows and
# the station grid's (idx, idz) lookup indices -- see synth_case.py's
# module docstring for the station layout.
EXPECTED_NODE_PROPERTY = [0.35, 0.25, 0.4, 0.3]


@pytest.fixture
def case_dir(tmp_path, monkeypatch):
    case_name = build_case(tmp_path, with_fractal_stress=True)
    monkeypatch.chdir(tmp_path)
    return case_name


@pytest.fixture
def prepare_module():
    return load_prepare_module(SCRIPT_PATH, "prepare_fractal_stress_under_test")


def _run(prepare_module, case_dir):
    import matplotlib.pyplot as plt
    _particle, meshnet, *_ = prepare_module.create_train_data(case_dir)
    plt.close("all")  # genMapsForEQDYNA opens one figure per frame
    return meshnet


def test_frame_count_matches_expected(prepare_module, case_dir):
    meshnet = _run(prepare_module, case_dir)
    assert meshnet["pos"].shape[0] == TIMESTEP_AFTER == 13


def test_shapes_and_dtypes(prepare_module, case_dir):
    meshnet = _run(prepare_module, case_dir)
    assert meshnet["pos"].shape == (TIMESTEP_AFTER, NROW, 2)
    assert meshnet["pos"].dtype == np.float32
    assert meshnet["velocity"].shape == (TIMESTEP_AFTER, NROW, 2)
    assert meshnet["velocity"].dtype == np.float32
    assert meshnet["node_property"].shape == (TIMESTEP_AFTER, NROW, 1)
    assert meshnet["node_property"].dtype == np.float32
    assert meshnet["cells"].dtype == np.int32


def test_no_all_zero_frames(prepare_module, case_dir):
    """Same zero-tail regression check as test_dataprep_prepare_eqdyna.py.
    node_type is intentionally excluded here: this script hard-codes
    `node_type[i, j, 0] = 0` for every node in every frame (no
    fault_boundary_node_type_mask parameter exists in this script), so an
    all-zero node_type frame is this script's correct, by-design output --
    not a signal of the zero-tail bug. Checking it here would be a
    tautology, not a regression guard."""
    meshnet = _run(prepare_module, case_dir)
    for field in ("pos", "cells", "velocity", "node_property"):
        arr = meshnet[field]
        for frame_idx in range(arr.shape[0]):
            assert not np.all(arr[frame_idx] == 0), (
                f"{field}[{frame_idx}] is entirely zero -- "
                f"frame was never written (zero-tail regression)")


def test_nskip_offset_maps_raw_file_to_frame(prepare_module, case_dir):
    meshnet = _run(prepare_module, case_dir)
    velocity = meshnet["velocity"]
    for i in range(velocity.shape[0]):
        expected = [velocity_value(i + 1 + NSKIP, j) for j in range(NROW)]
        np.testing.assert_allclose(
            velocity[i, :, 0], expected, atol=1e-5,
            err_msg=f"frame {i} does not match raw file src{i + 1 + NSKIP}.txt")


def test_node_property_from_fractal_stress_lookup(prepare_module, case_dir):
    """Every frame reuses the same (idx, idz) lookup against the static
    fractal_stress.txt map, so node_property should be identical, non-zero,
    and match the hand-derived values in every frame."""
    meshnet = _run(prepare_module, case_dir)
    node_property = meshnet["node_property"]
    for i in range(node_property.shape[0]):
        np.testing.assert_allclose(
            node_property[i, :, 0], EXPECTED_NODE_PROPERTY, atol=1e-6,
            err_msg=f"frame {i} node_property does not match the fractal_stress.txt lookup")

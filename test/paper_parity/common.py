"""Shared helpers for the paper-parity gate (tier 1 + tier 4, PR #1).

Metric conventions in this file are copied verbatim from
``utils/plot.rupture.dynamics.py`` (the repo's canonical rupture-time /
slip-rate analysis script) per PROJECT_RULES.md rule 7. Specifically:

- ``DT = 0.0167777``                       -- utils/plot.rupture.dynamics.py:37
- ``SLIPRATE_THRESHOLD = 0.1``              -- utils/plot.rupture.dynamics.py:38
- sliprate derivation (norm of the velocity vector, concatenating
  ``initial_velocities`` in front of the rollout so the first row lines
  up with t=0)                              -- utils/plot.rupture.dynamics.py:190-194
- ``get_rupture_time(sliprate_hist, dt, threshold, unreachable_val)``
  (identical body, reproduced here so this test-infra module has no
  import-time dependency on ``utils/``)     -- utils/plot.rupture.dynamics.py:223-229

We intentionally call ``get_rupture_time`` with ``threshold=SLIPRATE_THRESHOLD``
(0.1 m/s) per the mission mandate, NOT the function's own default of 0.001
(that default is used elsewhere for SCEC-benchmark-style comparisons, a
different convention -- do not conflate the two).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

# --- conventions, copied verbatim (see module docstring for citations) ---
DT = 0.0167777
SLIPRATE_THRESHOLD = 0.1
UNREACHABLE_VAL = 1000.0


def get_rupture_time(sliprate_hist, dt, threshold=SLIPRATE_THRESHOLD,
                      unreachable_val=UNREACHABLE_VAL):
    """Verbatim port of utils/plot.rupture.dynamics.py:223-229."""
    n_timesteps, n_nodes = sliprate_hist.shape
    rupture_time = np.full(n_nodes, unreachable_val, dtype=float)
    for it in range(n_timesteps):
        active = (rupture_time == unreachable_val) & (sliprate_hist[it] > threshold)
        rupture_time[active] = it * dt + 1.2
    return rupture_time


def sha256_of_file(path) -> str:
    """sha256 of a file's raw bytes, streamed (checkpoints/npz can be large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sliprate_hist_from_pkl_fields(initial_velocities, rollout):
    """Reproduce utils/plot.rupture.dynamics.py:190-194: concatenate the
    single INPUT_SEQUENCE_LENGTH initial-velocity row in front of the
    rollout, then take the Euclidean norm across the velocity components
    at every node/timestep to get a scalar slip-rate history.

    Args:
        initial_velocities: (1, nnodes, 2)
        rollout: (nsteps, nnodes, 2) -- either predicted_rollout or
            ground_truth_rollout from a rollout pkl.
    Returns:
        (1+nsteps, nnodes) slip-rate magnitude history.
    """
    vel = np.concatenate((initial_velocities, rollout), axis=0)
    return np.linalg.norm(vel, axis=-1)


def per_trajectory_metrics(pkl: dict) -> dict:
    """Compute the tier-1/tier-4 gate metrics for one rollout trajectory.

    - mse_raw / mse_vx / mse_vy: rollout MSE of predicted vs ground-truth
      velocity, computed directly on ``predicted_rollout`` vs
      ``ground_truth_rollout`` (the two arrays already share the same
      ``initial_velocities`` prefix, so it cancels out of the diff and is
      excluded here -- diffing it would only ever contribute exact zeros).
    - rupture time RMSE / missed_count / false_count at
      threshold=SLIPRATE_THRESHOLD, using the get_rupture_time /
      sliprate convention documented in the module docstring.
    """
    pred = np.asarray(pkl["predicted_rollout"], dtype=np.float64)
    gt = np.asarray(pkl["ground_truth_rollout"], dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted_rollout shape {pred.shape} != ground_truth_rollout shape {gt.shape}")

    diff2 = (pred - gt) ** 2
    mse_raw = float(np.mean(diff2))
    mse_vx = float(np.mean(diff2[..., 0]))
    mse_vy = float(np.mean(diff2[..., 1]))

    init_vel = np.asarray(pkl["initial_velocities"], dtype=np.float64)
    gt_sliprate = sliprate_hist_from_pkl_fields(init_vel, gt)
    pred_sliprate = sliprate_hist_from_pkl_fields(init_vel, pred)

    rpt_gt = get_rupture_time(gt_sliprate, DT, SLIPRATE_THRESHOLD, UNREACHABLE_VAL)
    rpt_pred = get_rupture_time(pred_sliprate, DT, SLIPRATE_THRESHOLD, UNREACHABLE_VAL)

    gt_reachable = rpt_gt < UNREACHABLE_VAL
    pred_reachable = rpt_pred < UNREACHABLE_VAL
    valid_mask = gt_reachable & pred_reachable
    missed_count = int(np.sum(gt_reachable & ~pred_reachable))
    false_count = int(np.sum(~gt_reachable & pred_reachable))

    if np.any(valid_mask):
        rpt_rmse = float(np.sqrt(np.mean((rpt_pred[valid_mask] - rpt_gt[valid_mask]) ** 2)))
    else:
        rpt_rmse = None  # explicit: no valid_mask overlap, do not fabricate a number

    return {
        "mse_raw": mse_raw,
        "mse_vx": mse_vx,
        "mse_vy": mse_vy,
        "rupture_time_rmse": rpt_rmse,
        "rupture_time_missed_count": missed_count,
        "rupture_time_false_count": false_count,
        "n_nodes": int(gt.shape[1]),
        "n_valid_rupture_nodes": int(np.sum(valid_mask)),
    }


# --- model registry -------------------------------------------------------
# Paths verified against gns-sample/ on 2026-09-24 (see NOTES_tier1.md).
GNS_SAMPLE = Path(__file__).resolve().parents[2] / "gns-sample"

MODEL_REGISTRY = {
    "M1": {
        "working_dir": GNS_SAMPLE / "case3.200m.homo.a.Vw",
        "model_dir": GNS_SAMPLE / "case3.200m.homo.a.Vw" / "models.nmp10.cotopaxi",
        "model_step": 3000000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case3.200m.homo.a.Vw" / "rollouts.nmp10.cotopaxi" / "model-3000000.pt"),
        "provenance": "published",
    },
    # Paper's M2 (Liu & Becker 2025, sec 2.4): trained on D2/30-scenario
    # (asperity 35/55 MPa only), case4.200m.multi.stress.homo.a.Vw. This
    # registry entry gates that checkpoint applied to the D3 fractal-stress
    # test set (owner-confirmed 2026-09-27; the checkpoint at
    # models.nmp10.cotopaxi.r1/model-2900000.pt is byte-identical, md5
    # 48d0e2b9, whether read from case4.200m.fractal.stress.homo.a.Vw/ or
    # case4.200m.multi.stress.homo.a.Vw/ -- the fractal directory is a
    # test-set copy of the same trained model, not a separate training run).
    # This was PREVIOUSLY mislabeled "M3" in this registry -- see
    # NOTES_tier1.md / PR #1 fix.
    # NOTE: this gates M2-on-D3, NOT M2's own paper-parity test (D2). Gating
    # M2 against its own D2 test set is out of scope for this PR -- see
    # PR #2 (test-coverage matrix, PATHWAY_FORWARD.md).
    "M2": {
        "working_dir": GNS_SAMPLE / "case4.200m.fractal.stress.homo.a.Vw",
        "model_dir": (
            GNS_SAMPLE / "case4.200m.fractal.stress.homo.a.Vw" / "models.nmp10.cotopaxi.r1"),
        "model_step": 2900000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.fractal.stress.homo.a.Vw"
            / "rollouts.nmp10.cotopaxi.r1" / "model-2900000.pt"),
        # No `.published` rollout dir exists under the fractal directory for
        # this checkpoint -- default per mission mandate. DO NOT treat this
        # baseline as a trusted paper-parity oracle.
        "provenance": "unconfirmed",
    },
    # Paper's M3 (Liu & Becker 2025, sec 2.4): trained on D2/148-scenario
    # (all four asperity levels, lr 3e-5, batch 8), picked @2.7M, on its own
    # D2 test set -- case4.200m.multi.stress.160scenarios.homo.a.Vw. This
    # was PREVIOUSLY mislabeled "M2" in this registry -- see NOTES_tier1.md /
    # PR #1 fix.
    "M3": {
        "working_dir": GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw",
        "model_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw"
            / "models.nmp10.lr3e-5.b8.cotopaxi.r1"),
        "model_step": 2700000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw"
            / "rollouts.nmp10.lr3e-5.b8.cotopaxi.r1.published" / "model-2700000.pt"),
        "provenance": "published",
    },
}


# --- PR #2: test-set coverage matrix --------------------------------------
# Every (model, test-set) pair below is an ADDITIONAL row keyed by a distinct
# registry key, in the SAME shape as MODEL_REGISTRY (so model_paths() /
# find_published_pkls() / per_trajectory_metrics() need no changes -- see
# NOTES_pr2.md for the sha256 investigation behind each provenance/family
# note). Paths verified against gns-sample/ on 2026-09-27.
TEST_SET_REGISTRY = {
    # --- M1 (case3.200m.homo.a.Vw checkpoint) on its "others" test sets ---
    "M1_large": {
        "working_dir": GNS_SAMPLE / "case3.200m.homo.a.Vw.others",
        "model_dir": GNS_SAMPLE / "case3.200m.homo.a.Vw.others" / "models.nmp10.cotopaxi",
        "model_step": 4000000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case3.200m.homo.a.Vw.others"
            / "rollouts.nmp10.cotopaxi.large.published" / "model-4000000.pt"),
        "provenance": "published",
    },
    "M1_small": {
        "working_dir": GNS_SAMPLE / "case3.200m.homo.a.Vw.others",
        "model_dir": GNS_SAMPLE / "case3.200m.homo.a.Vw.others" / "models.nmp10.cotopaxi",
        "model_step": 3000000,
        # No `.published` suffix on this dir -- default per mission mandate:
        # provenance "unconfirmed". Only 1 pkl exists here (single trajectory).
        "published_rollout_dir": (
            GNS_SAMPLE / "case3.200m.homo.a.Vw.others"
            / "rollouts.nmp10.cotopaxi.small.D1.T_small" / "model-3000000.pt"),
        "provenance": "unconfirmed",
    },
    # --- M2 (.r1 family) on its own D2 test set -------------------------
    # NOTE (open question, see NOTES_pr2.md): the checkpoint the paper calls
    # M2 -- models.nmp10.cotopaxi.r1/model-2900000.pt (sha256 4ff5adfd...) --
    # has a `.published` rollout dir at case4.200m.multi.stress.homo.a.Vw/
    # rollouts.nmp10.cotopaxi.r1.published/model-2900000.pt, but that
    # directory contains ZERO rollout_*.pkl files (only rollout.log.txt) --
    # confirmed by directory listing 2026-09-27. There is THEREFORE NO
    # PUBLISHED PKL DATA to gate M2's own D2 test against using the .r1
    # checkpoint. We do NOT fabricate a baseline for it.
    #
    # A DIFFERENT rollout dir, rollouts.nmp10.cotopaxi.published/
    # model-3000000.pt (no .r1 suffix), DOES have 10 published pkls, but its
    # checkpoint (sha256 086fa959...) is a byte-DIFFERENT, separately-trained
    # model from the .r1 family -- confirmed by sha256 (see NOTES_pr2.md).
    # We gate that checkpoint here under its own key, "M2_altckpt_D2", so it
    # is never silently conflated with the paper's M2. Whether this
    # non-.r1/3000000 run is actually the one shown in the paper's D2 figure
    # is UNRESOLVED and flagged as an open question for the owner.
    "M2_altckpt_D2": {
        "working_dir": GNS_SAMPLE / "case4.200m.multi.stress.homo.a.Vw",
        "model_dir": GNS_SAMPLE / "case4.200m.multi.stress.homo.a.Vw" / "models.nmp10.cotopaxi",
        "model_step": 3000000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.homo.a.Vw"
            / "rollouts.nmp10.cotopaxi.published" / "model-3000000.pt"),
        "provenance": "published",
    },
    # --- M2 (.r1 family) on the case3-test (cross-hypocenter) test set ---
    # NOTE: this test set uses model-3000000.pt of the .r1 run (sha256
    # 3dbfb4ab...), NOT model-2900000.pt (sha256 4ff5adfd..., used for the
    # existing "M2" D3-fractal gate) -- confirmed distinct checkpoint steps
    # of the SAME .r1 training run by sha256 (see NOTES_pr2.md). This is a
    # different step choice per test set, not a training-run ambiguity.
    "M2_case3test": {
        "working_dir": GNS_SAMPLE / "case4.200m.multi.stress.homo.a.Vw.case3.test",
        "model_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.homo.a.Vw.case3.test"
            / "models.nmp10.cotopaxi.r1"),
        "model_step": 3000000,
        # No `.published` suffix -- provenance "unconfirmed" per mandate.
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.homo.a.Vw.case3.test"
            / "rollouts.nmp10.cotopaxi.r1" / "model-3000000.pt"),
        "provenance": "unconfirmed",
    },
    # --- checkerboard test set (case4.200m.multi.asp.homo.a.Vw) ---------
    # Provenance investigation (see NOTES_pr2.md): the checkerboard model
    # checkpoint models.nmp10.cotopaxi/model-3000000.pt is sha256
    # 086fa959e1310c52e5da74f39e3ebabd532a3fcefb33004a860ae3d38403e976 --
    # BYTE-IDENTICAL to the non-.r1 M2 checkpoint gated above as
    # "M2_altckpt_D2", and DISTINCT from both steps of the paper's .r1 M2
    # checkpoint family (2900000 sha256 4ff5adfd..., 3000000 sha256
    # 3dbfb4ab...). So checkerboard shares weights with the non-.r1 training
    # run, NOT the paper's M2 (.r1) run. It is registered under its own key
    # rather than folded into the M2 family. No `.published` rollout dir
    # exists here either -- provenance "unconfirmed". Whether the paper's
    # checkerboard figure actually uses this non-.r1 checkpoint or a
    # separately-trained model entirely is an OPEN QUESTION for the owner.
    "checkerboard": {
        "working_dir": GNS_SAMPLE / "case4.200m.multi.asp.homo.a.Vw",
        "model_dir": GNS_SAMPLE / "case4.200m.multi.asp.homo.a.Vw" / "models.nmp10.cotopaxi",
        "model_step": 3000000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.multi.asp.homo.a.Vw"
            / "rollouts.nmp10.cotopaxi" / "model-3000000.pt"),
        "provenance": "unconfirmed",
    },
    # --- M3 (.r1 lr3e-5.b8 family) on the case3-test test set -----------
    "M3_case3test": {
        "working_dir": GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.test",
        "model_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.test"
            / "models.nmp10.lr3e-5.b8.cotopaxi.r1"),
        "model_step": 2700000,
        # No `.published` suffix -- provenance "unconfirmed" per mandate.
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.test"
            / "rollouts.nmp10.lr3e-5.b8.cotopaxi.r1" / "model-2700000.pt"),
        "provenance": "unconfirmed",
    },
    # --- M3 (.r1 lr3e-5.b8 family) on the case3.others.test test set ----
    "M3_case3othertest": {
        "working_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.others.test"),
        "model_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.others.test"
            / "models.nmp10.lr3e-5.b8.cotopaxi.r1"),
        "model_step": 2700000,
        "published_rollout_dir": (
            GNS_SAMPLE / "case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.others.test"
            / "rollouts.nmp10.lr3e-5.b8.cotopaxi.r1.published" / "model-2700000.pt"),
        "provenance": "published",
    },
}

# Merge into a single lookup so model_paths()/run_gate.py/extract_baselines.py
# need no branching logic between "model" and "test-set" rows -- they are the
# same shape and the same machinery applies uniformly.
ALL_REGISTRY = {**MODEL_REGISTRY, **TEST_SET_REGISTRY}


def model_paths(model_key: str) -> dict:
    if model_key not in ALL_REGISTRY:
        raise KeyError(f"Unknown model key {model_key!r}; expected one of {list(ALL_REGISTRY)}")
    entry = ALL_REGISTRY[model_key]
    model_file = f"model-{entry['model_step']}.pt"
    train_state_file = f"train_state-{entry['model_step']}.pt"
    return {
        "data_path": entry["working_dir"] / "dataset",
        "model_path": entry["model_dir"],
        "model_file": model_file,
        "train_state_file": train_state_file,
        "checkpoint": entry["model_dir"] / model_file,
        "test_npz": entry["working_dir"] / "dataset" / "test.npz",
        "published_rollout_dir": entry["published_rollout_dir"],
        "provenance": entry["provenance"],
    }


def find_published_pkls(published_rollout_dir: Path):
    pkls = sorted(
        published_rollout_dir.glob("rollout_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1]))
    if not pkls:
        raise FileNotFoundError(f"No rollout_*.pkl files found under {published_rollout_dir}")
    return pkls


def gns_sample_available() -> bool:
    """True iff gns-sample/ is a real, non-empty directory (not just the
    gitignored symlink placeholder from a checkout without the data)."""
    return GNS_SAMPLE.is_dir() and any(GNS_SAMPLE.iterdir())

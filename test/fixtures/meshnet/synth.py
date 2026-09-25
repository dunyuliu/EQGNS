"""Synthetic meshnet dataset generator used by the integration/e2e tiers.

Builds a tiny, fully deterministic (seeded) structured triangular mesh with a
handful of nodes and short trajectories. This is a stand-in for the real
(249GB, not checked out here) meshnet training data -- small enough to train
a few gradient steps in well under a second, but shaped exactly like the real
data (same keys, same array ranks) so it exercises the real data_loader /
train / rollout code paths.

Mesh layout: an (nx * ny) structured grid, positions fixed for the whole
trajectory (consistent with train.rollout()'s assumption that node positions
do not move). Perimeter nodes are tagged WALL_BOUNDARY (their ground-truth
velocity is imposed every rollout step); interior nodes are NORMAL (the GNN
predicts them). Each grid cell is split into 2 triangles, matching the
`cells: (ntimestep, ncells, 3)` contract documented in README.md.
"""
import numpy as np

from meshnet.utils import NodeType

NX, NY = 4, 3  # -> 12 nodes, 2 interior (NORMAL) nodes, 12 triangles


def _build_mesh(nx=NX, ny=NY):
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    pos = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)  # (nnodes, 2)

    def node_id(ix, iy):
        return iy * nx + ix

    node_type = np.full((nx * ny, 1), NodeType.WALL_BOUNDARY, dtype=np.int64)
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            node_type[node_id(ix, iy)] = NodeType.NORMAL

    cells = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a, b, c, d = (node_id(ix, iy), node_id(ix + 1, iy),
                          node_id(ix + 1, iy + 1), node_id(ix, iy + 1))
            cells.append([a, b, c])
            cells.append([a, c, d])
    cells = np.array(cells, dtype=np.int64)  # (ncells, 3)

    # node_property: static scalar in [0, 1], e.g. a proxy for initial stress level.
    node_property = (pos[:, 0:1]).astype(np.float32)

    return pos, node_type, node_property, cells


def make_trajectory(seed, timesteps=20, decay=0.85, forcing_amp=0.05):
    """Build one deterministic synthetic trajectory.

    Velocity dynamics: interior nodes follow a damped, seeded-random forced
    oscillator (bounded, smooth); boundary nodes are held at a fixed seeded
    velocity for the whole trajectory (as if driven by external loading).
    Everything is derived from `seed` only, so two calls with the same seed
    are bit-identical.
    """
    rng = np.random.default_rng(seed)
    pos, node_type, node_property, cells = _build_mesh()
    nnodes = pos.shape[0]

    boundary_mask = (node_type[:, 0] != NodeType.NORMAL)

    velocity = np.zeros((timesteps, nnodes, 2), dtype=np.float32)
    boundary_velocity = rng.normal(scale=0.02, size=(nnodes, 2)).astype(np.float32)
    forcing = rng.normal(scale=forcing_amp, size=(nnodes, 2)).astype(np.float32)

    v = np.zeros((nnodes, 2), dtype=np.float32)
    for t in range(timesteps):
        v = decay * v + forcing * np.cos(0.3 * t)
        velocity[t] = np.where(boundary_mask[:, None], boundary_velocity, v)

    pressure = np.zeros((timesteps, nnodes, 1), dtype=np.float32)
    pos_t = np.broadcast_to(pos, (timesteps, nnodes, 2)).copy()
    node_type_t = np.broadcast_to(node_type, (timesteps, nnodes, 1)).copy()
    node_property_t = np.broadcast_to(node_property, (timesteps, nnodes, 1)).copy()
    cells_t = np.broadcast_to(cells, (timesteps,) + cells.shape).copy()

    return {
        "pos": pos_t,
        "node_type": node_type_t,
        "node_property": node_property_t,
        "velocity": velocity,
        "pressure": pressure,
        "cells": cells_t,
    }


def write_npz(path, trajectories):
    """Write a list of trajectory dicts to `path` in the format expected by
    meshnet.data_loader (each npz "value" is a 0-d object array holding a dict)."""
    arrays = {}
    for i, traj in enumerate(trajectories):
        holder = np.empty((), dtype=object)
        holder[()] = traj
        arrays[f"traj_{i}"] = holder
    np.savez(path, **arrays)


def build_dataset(tmp_dir, seed=1234, timesteps=20):
    """Write train.npz (2 trajectories), valid.npz (1) and test.npz (1) into
    `tmp_dir` (a str/Path). Returns the mesh's nnodes for convenience."""
    train_trajs = [make_trajectory(seed, timesteps), make_trajectory(seed + 1, timesteps)]
    valid_trajs = [make_trajectory(seed + 2, timesteps)]
    test_trajs = [make_trajectory(seed + 3, timesteps)]

    write_npz(f"{tmp_dir}/train.npz", train_trajs)
    write_npz(f"{tmp_dir}/valid.npz", valid_trajs)
    write_npz(f"{tmp_dir}/test.npz", test_trajs)
    return NX * NY


TINY_CONFIG = {
    "version": 1.0,
    "INPUT_SEQUENCE_LENGTH": 1,
    "noise_std": 0.0,
    "node_type_embedding_size": 9,
    "dt": 0.1,
    "lr_init": 1e-3,
    "lr_decay_rate": 1.0,
    "lr_decay_steps": 1e6,
    "loss_report_step": 1000000,
    "simulator_simulation_dimensions": 2,
    "simulator_nnode_in": 12,
    "simulator_nedge_in": 3,
    "simulator_latent_dim": 8,
    "simulator_nmessage_passing_steps": 2,
    "simulator_nmlp_layers": 2,
    "simulator_mlp_hidden_dim": 8,
    "simulator_nnode_types": 3,
    "simulator_node_type_embedding_size": 9,
}


def write_config(model_dir, overrides=None):
    import json
    import os
    cfg = dict(TINY_CONFIG)
    if overrides:
        cfg.update(overrides)
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(cfg, f)
    return cfg

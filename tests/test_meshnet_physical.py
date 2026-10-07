"""Physical-behaviour tests for meshnet (tier 4).

These check the empirical contract between the code and the rollout physics
it claims to implement, not just API shapes:
  1. Additive-acceleration identity: predict_velocity == current + inverse(GNN(...)).
     (exact algebraic invariant; also unit-tested at the component level in
     test_meshnet_unit_learned_simulator.py -- repeated here at the rollout
     level with a full graph built through the real transformer pipeline.)
  2. Zero-forcing -> zero-response: a network with all-zero weights (and thus
     zero raw output before denormalization) run in rollout on an all-NORMAL
     mesh (no boundary conditions overriding anything) must keep the velocity
     field at exactly zero for every step -- a hand-computable asymptotic
     limit that catches any accidental additive drift or masking bug.
  3. Velocity-noise standard deviation statistically matches noise_std from
     config, at the scale the training loop actually sees it (already
     covered per-call in the unit tier; repeated here as an end-to-end
     invariant across a full rollout-shaped mesh).
"""
import pytest
import torch
import torch_geometric.transforms as T

from meshnet import train as train_mod
from meshnet.learned_simulator import MeshSimulator
from meshnet.noise import get_velocity_noise
from meshnet.utils import NodeType, datas_to_graph
from fixtures.meshnet.synth import build_dataset, TINY_CONFIG

pytestmark = pytest.mark.physical

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


def _make_simulator(**overrides):
    cfg = dict(TINY_CONFIG)
    kwargs = dict(
        simulation_dimensions=cfg["simulator_simulation_dimensions"],
        nnode_in=cfg["simulator_nnode_in"], nedge_in=cfg["simulator_nedge_in"],
        latent_dim=cfg["simulator_latent_dim"],
        nmessage_passing_steps=cfg["simulator_nmessage_passing_steps"],
        nmlp_layers=cfg["simulator_nmlp_layers"], mlp_hidden_dim=cfg["simulator_mlp_hidden_dim"],
        nnode_types=cfg["simulator_nnode_types"],
        node_type_embedding_size=cfg["simulator_node_type_embedding_size"],
        device="cpu")
    kwargs.update(overrides)
    return MeshSimulator(**kwargs)


def test_predict_velocity_matches_additive_acceleration_identity_through_real_graph(tmp_path):
    build_dataset(tmp_path, seed=99, timesteps=8)
    from meshnet.data_loader import get_data_loader_by_samples
    ds = get_data_loader_by_samples(str(tmp_path / "train.npz"), input_length_sequence=1,
                                     dt=0.1, batch_size=1, shuffle=False)
    graph = transformer(next(iter(ds)))

    simulator = _make_simulator()
    simulator.eval()
    node_types = graph.x[:, 0]
    node_property = graph.x[:, 1]
    current_velocities = graph.x[:, 2:4]

    with torch.no_grad():
        processed = simulator._encoder_preprocessor(current_velocities, node_types, node_property, None)
        raw_accel = simulator._encode_process_decode(processed, graph.edge_index, graph.edge_attr)
        expected = current_velocities + simulator._output_normalizer.inverse(raw_accel)
        actual = simulator.predict_velocity(current_velocities, node_types, node_property,
                                             graph.edge_index, graph.edge_attr)

    torch.testing.assert_close(actual, expected)


def test_zero_weight_network_keeps_all_normal_mesh_at_zero_velocity(tmp_path):
    # Build a mesh where every node is NORMAL, so rollout never overwrites a
    # node with a nonzero ground-truth boundary value: the only source of
    # velocity change is the GNN's predicted acceleration.
    import numpy as np
    from fixtures.meshnet.synth import _build_mesh
    pos, node_type, node_property, cells = _build_mesh()
    node_type = np.zeros_like(node_type)  # force every node to NORMAL
    timesteps = 6
    nnodes = pos.shape[0]
    zero_traj = {
        "pos": np.broadcast_to(pos, (timesteps, nnodes, 2)).copy(),
        "node_type": np.broadcast_to(node_type, (timesteps, nnodes, 1)).copy(),
        "node_property": np.broadcast_to(node_property, (timesteps, nnodes, 1)).copy(),
        "velocity": np.zeros((timesteps, nnodes, 2), dtype=np.float32),
        "pressure": np.zeros((timesteps, nnodes, 1), dtype=np.float32),
        "cells": np.broadcast_to(cells, (timesteps,) + cells.shape).copy(),
    }
    from fixtures.meshnet.synth import write_npz
    write_npz(tmp_path / "test.npz", [zero_traj])

    simulator = _make_simulator()
    # Force the output normalizer's std to 1.0 (instead of the untrained
    # std_epsilon=1e-8 floor) so inverse() does NOT crush an untrained,
    # non-zero raw GNN output down to ~0 on its own -- without this, the
    # test would pass regardless of the network's weights, which is exactly
    # the "passes for the wrong reason" failure mode this suite forbids.
    with torch.no_grad():
        simulator._output_normalizer._acc_count += 1.0
        simulator._output_normalizer._acc_sum_squared += 1.0
        for p in simulator.parameters():
            p.data.zero_()
    simulator.eval()

    train_mod.INPUT_SEQUENCE_LENGTH = 1
    train_mod.dt = TINY_CONFIG["dt"]

    from meshnet.data_loader import get_data_loader_by_trajectories
    ds = get_data_loader_by_trajectories(path=str(tmp_path / "test.npz"))
    features = next(iter(ds))
    nsteps = len(features[0]) - train_mod.INPUT_SEQUENCE_LENGTH

    with torch.no_grad():
        output = train_mod.rollout(simulator, features, nsteps, device=torch.device("cpu"))

    torch.testing.assert_close(
        torch.from_numpy(output["predicted_rollout"]),
        torch.zeros_like(torch.from_numpy(output["predicted_rollout"])),
        atol=1e-6, rtol=0)


def test_rollout_scale_noise_std_matches_config_statistically(tmp_path):
    # Only 2 of the mesh's 12 nodes are NORMAL (the rest are boundary and get
    # zero noise), so we need many timesteps' worth of samples to get a
    # stable standard-deviation estimate out of the real per-sample call
    # pattern used in train()/validation().
    build_dataset(tmp_path, seed=11, timesteps=150)
    from meshnet.data_loader import get_data_loader_by_samples
    configured_std = 0.03
    samples = []
    for i, graph in enumerate(get_data_loader_by_samples(
            str(tmp_path / "train.npz"), input_length_sequence=1, dt=0.1, batch_size=1, shuffle=False)):
        graph = transformer(graph)
        torch.manual_seed(1000 + i)
        noise = get_velocity_noise(graph, noise_std=configured_std, device="cpu")
        mask = graph.x[:, 0] == NodeType.NORMAL
        samples.append(noise[mask])
    all_noise = torch.cat(samples)
    assert all_noise.numel() > 500  # enough samples for a stable std estimate
    assert all_noise.std().item() == pytest.approx(configured_std, rel=0.15)

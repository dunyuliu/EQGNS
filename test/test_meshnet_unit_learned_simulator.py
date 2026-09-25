"""Unit tests for meshnet.learned_simulator.MeshSimulator (tier 1)."""
import json
import os

import pytest
import torch

from meshnet.learned_simulator import MeshSimulator
from fixtures.meshnet.synth import TINY_CONFIG

pytestmark = pytest.mark.unit


def _make_simulator(node_type_embedding_size=9, device="cpu"):
    return MeshSimulator(
        simulation_dimensions=2,
        nnode_in=2 + node_type_embedding_size + 1,
        nedge_in=3,
        latent_dim=8,
        nmessage_passing_steps=2,
        nmlp_layers=2,
        mlp_hidden_dim=8,
        nnode_types=3,
        node_type_embedding_size=node_type_embedding_size,
        device=device)


def test_encoder_preprocessor_shape_and_onehot_width():
    simulator = _make_simulator(node_type_embedding_size=9)
    current_velocities = torch.zeros(4, 2)
    node_type = torch.tensor([0.0, 1.0, 6.0, 0.0])
    node_property = torch.rand(4)

    processed = simulator._encoder_preprocessor(
        current_velocities, node_type, node_property, velocity_noise=None)

    # 2 (velocity) + 9 (onehot) + 1 (property) = 12
    assert processed.shape == (4, 12)


def test_encoder_preprocessor_changes_with_node_type_embedding_size():
    # A documented config key (node_type_embedding_size) must actually change
    # the shape of the encoded node features -- otherwise it is dead config.
    small = _make_simulator(node_type_embedding_size=9)
    big = _make_simulator(node_type_embedding_size=12)

    current_velocities = torch.zeros(3, 2)
    node_type = torch.tensor([0.0, 1.0, 2.0])
    node_property = torch.rand(3)

    out_small = small._encoder_preprocessor(current_velocities, node_type, node_property, None)
    out_big = big._encoder_preprocessor(current_velocities, node_type, node_property, None)
    assert out_small.shape[-1] == 12
    assert out_big.shape[-1] == 15
    assert out_small.shape[-1] != out_big.shape[-1]


def test_encoder_preprocessor_adds_noise_only_when_provided():
    simulator = _make_simulator()
    velocities = torch.zeros(2, 2)
    node_type = torch.tensor([0.0, 0.0])
    node_property = torch.zeros(2)
    noise = torch.full((2, 2), 5.0)

    without_noise = simulator._encoder_preprocessor(velocities, node_type, node_property, None)
    with_noise = simulator._encoder_preprocessor(velocities, node_type, node_property, noise)
    # Only the first 2 (velocity) columns should differ, and normalization
    # accumulates on every call, so compare column-wise sign of the shift.
    assert not torch.allclose(without_noise[:, :2], with_noise[:, :2])


def test_save_then_load_reproduces_identical_output(tmp_path):
    simulator = _make_simulator()
    simulator.eval()

    current_velocities = torch.randn(5, 2)
    node_type = torch.tensor([0.0, 1.0, 6.0, 0.0, 2.0])
    node_property = torch.rand(5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_features = torch.randn(4, 3)

    with torch.no_grad():
        # Warm up normalizer stats once so save/load carries non-trivial state.
        simulator._encoder_preprocessor(current_velocities, node_type, node_property, None)
        before = simulator.predict_velocity(
            current_velocities, node_type, node_property, edge_index, edge_features)

    path = os.path.join(tmp_path, "model.pt")
    simulator.save(path)

    reloaded = _make_simulator()
    reloaded.load(path)  # exercises torch.load(..., map_location=self._device)
    reloaded.eval()

    with torch.no_grad():
        after = reloaded.predict_velocity(
            current_velocities, node_type, node_property, edge_index, edge_features)

    torch.testing.assert_close(before, after)


def test_load_raises_for_missing_checkpoint(tmp_path):
    simulator = _make_simulator()
    with pytest.raises(FileNotFoundError):
        simulator.load(os.path.join(tmp_path, "does_not_exist.pt"))


def test_predict_velocity_is_current_plus_denormalized_acceleration():
    # Algebraic identity the whole rollout depends on: v_{t+1} = v_t + inverse(GNN(...)).
    simulator = _make_simulator()
    simulator.eval()
    current_velocities = torch.randn(4, 2)
    node_type = torch.tensor([0.0, 0.0, 1.0, 0.0])
    node_property = torch.rand(4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_features = torch.randn(3, 3)

    with torch.no_grad():
        processed = simulator._encoder_preprocessor(
            current_velocities, node_type, node_property, velocity_noise=None)
        raw_accel = simulator._encode_process_decode(processed, edge_index, edge_features)
        expected_velocity = current_velocities + simulator._output_normalizer.inverse(raw_accel)

        actual_velocity = simulator.predict_velocity(
            current_velocities, node_type, node_property, edge_index, edge_features)

    torch.testing.assert_close(actual_velocity, expected_velocity)

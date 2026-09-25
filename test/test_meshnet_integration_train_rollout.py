"""Integration tests for meshnet (tier 2): data_loader -> transformer ->
MeshSimulator -> optimizer, checkpoint save/resume, and train.rollout(),
wired together the way meshnet/train.py actually wires them (no mocks).

Uses the tiny synthetic dataset (test/fixtures/meshnet/synth.py), never the
249GB gns-sample data.
"""
import os

import pytest
import torch
import torch_geometric.transforms as T

from meshnet import data_loader
from meshnet import train as train_mod
from meshnet.learned_simulator import MeshSimulator
from meshnet.noise import get_velocity_noise
from meshnet.utils import NodeType
from fixtures.meshnet.synth import build_dataset, TINY_CONFIG

pytestmark = pytest.mark.integration

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


@pytest.fixture
def tiny_simulator():
    cfg = TINY_CONFIG
    return MeshSimulator(
        simulation_dimensions=cfg["simulator_simulation_dimensions"],
        nnode_in=cfg["simulator_nnode_in"],
        nedge_in=cfg["simulator_nedge_in"],
        latent_dim=cfg["simulator_latent_dim"],
        nmessage_passing_steps=cfg["simulator_nmessage_passing_steps"],
        nmlp_layers=cfg["simulator_nmlp_layers"],
        mlp_hidden_dim=cfg["simulator_mlp_hidden_dim"],
        nnode_types=cfg["simulator_nnode_types"],
        node_type_embedding_size=cfg["simulator_node_type_embedding_size"],
        device="cpu")


@pytest.fixture
def dataset_dir(tmp_path):
    build_dataset(tmp_path, seed=7, timesteps=16)
    return tmp_path


def test_few_training_steps_reduce_or_keep_finite_loss(tiny_simulator, dataset_dir):
    """A handful of real gradient steps through the real train-loop wiring:
    SamplesDataset -> DataLoader -> FaceToEdge/Cartesian/Distance transformer
    -> predict_acceleration -> acceleration_loss -> backward -> step.
    Must never diverge to NaN/Inf on well-conditioned tiny synthetic data.
    """
    torch.manual_seed(0)
    simulator = tiny_simulator
    simulator.train()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-3)

    ds = data_loader.get_data_loader_by_samples(
        path=str(dataset_dir / "train.npz"), input_length_sequence=1, dt=0.1,
        batch_size=4, shuffle=False)

    losses = []
    for i, graph in enumerate(ds):
        if i >= 8:
            break
        graph = transformer(graph)
        node_types = graph.x[:, 0]
        node_property = graph.x[:, 1]
        current_velocities = graph.x[:, 2:4]
        velocity_noise = get_velocity_noise(graph, noise_std=0.0, device="cpu")

        pred_acc, target_acc = simulator.predict_acceleration(
            current_velocities=current_velocities, node_type=node_types,
            node_property=node_property, edge_index=graph.edge_index,
            edge_features=graph.edge_attr, target_velocities=graph.y,
            velocity_noise=velocity_noise)

        non_kinematic_mask = torch.logical_or(node_types == NodeType.NORMAL,
                                               node_types == NodeType.HIGH_STRESS)
        loss = train_mod.acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert len(losses) == 8
    assert all(torch.isfinite(torch.tensor(losses)))


def test_checkpoint_save_and_resume_preserves_predictions(tiny_simulator, dataset_dir, tmp_path):
    """Train a couple of steps, save, build a fresh simulator, load the
    checkpoint, and confirm it reproduces the same prediction (the
    'resume' half of EQdyna's compile->run->checkpoint->resume story)."""
    torch.manual_seed(1)
    simulator = tiny_simulator
    simulator.train()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-3)
    ds = data_loader.get_data_loader_by_samples(
        path=str(dataset_dir / "train.npz"), input_length_sequence=1, dt=0.1,
        batch_size=4, shuffle=False)

    for i, graph in enumerate(ds):
        if i >= 3:
            break
        graph = transformer(graph)
        node_types = graph.x[:, 0]
        node_property = graph.x[:, 1]
        current_velocities = graph.x[:, 2:4]
        velocity_noise = get_velocity_noise(graph, noise_std=0.0, device="cpu")
        pred_acc, target_acc = simulator.predict_acceleration(
            current_velocities=current_velocities, node_type=node_types,
            node_property=node_property, edge_index=graph.edge_index,
            edge_features=graph.edge_attr, target_velocities=graph.y,
            velocity_noise=velocity_noise)
        non_kinematic_mask = torch.logical_or(node_types == NodeType.NORMAL,
                                               node_types == NodeType.HIGH_STRESS)
        loss = train_mod.acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ckpt_path = os.path.join(tmp_path, "model.pt")
    simulator.save(ckpt_path)
    simulator.eval()

    with torch.no_grad():
        probe_velocities = torch.randn(12, 2)
        probe_types = torch.zeros(12)
        probe_property = torch.rand(12)
        probe_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        probe_edge_features = torch.randn(3, 3)
        before = simulator.predict_velocity(
            probe_velocities, probe_types, probe_property, probe_edge_index, probe_edge_features)

    fresh = MeshSimulator(
        simulation_dimensions=TINY_CONFIG["simulator_simulation_dimensions"],
        nnode_in=TINY_CONFIG["simulator_nnode_in"], nedge_in=TINY_CONFIG["simulator_nedge_in"],
        latent_dim=TINY_CONFIG["simulator_latent_dim"],
        nmessage_passing_steps=TINY_CONFIG["simulator_nmessage_passing_steps"],
        nmlp_layers=TINY_CONFIG["simulator_nmlp_layers"],
        mlp_hidden_dim=TINY_CONFIG["simulator_mlp_hidden_dim"],
        nnode_types=TINY_CONFIG["simulator_nnode_types"],
        node_type_embedding_size=TINY_CONFIG["simulator_node_type_embedding_size"],
        device="cpu")
    fresh.load(ckpt_path)  # exercises torch.load(..., map_location=self._device) on a real ckpt
    fresh.eval()
    with torch.no_grad():
        after = fresh.predict_velocity(
            probe_velocities, probe_types, probe_property, probe_edge_index, probe_edge_features)

    torch.testing.assert_close(before, after)


def test_rollout_produces_finite_predictions_of_expected_shape(tiny_simulator, dataset_dir):
    """Exercises the real, optimized meshnet.train.rollout() against a
    TrajectoriesDataset example: shapes and finiteness must hold end to end."""
    simulator = tiny_simulator
    simulator.eval()

    # rollout() reads these as free variables from the train module's globals
    # (normally set by train.main() from config.json); set them the same way
    # main() would, without going through absl FLAGS.
    train_mod.INPUT_SEQUENCE_LENGTH = TINY_CONFIG["INPUT_SEQUENCE_LENGTH"]
    train_mod.dt = TINY_CONFIG["dt"]

    ds = data_loader.get_data_loader_by_trajectories(path=str(dataset_dir / "test.npz"))
    features = next(iter(ds))
    nsteps = len(features[0]) - train_mod.INPUT_SEQUENCE_LENGTH

    with torch.no_grad():
        output = train_mod.rollout(simulator, features, nsteps, device=torch.device("cpu"))

    nnodes = features[0].shape[1]
    assert output["predicted_rollout"].shape == (nsteps, nnodes, 2)
    assert output["ground_truth_rollout"].shape == (nsteps, nnodes, 2)
    assert bool((output["predicted_rollout"] == output["predicted_rollout"]).all())  # no NaN
    assert (abs(output["predicted_rollout"]) < 1e6).all()  # no blow-up
    assert output["mean_loss"] >= 0.0

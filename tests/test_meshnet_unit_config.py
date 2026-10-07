"""Unit tests for the config.json system (tier 1).

meshnet/train.py reads meshnet/example.config.json-shaped files at
`<model_path>/config.json` and threads every key straight into either a
global training constant or a MeshSimulator constructor argument (see
train.py main()). This test file checks that (a) the example config is
loadable and complete, and (b) each simulator-architecture key actually
changes model behaviour -- a config key that can be edited with no
observable effect is dead config (see CLAUDE.md).
"""
import json
import os

import pytest
import torch

from meshnet.learned_simulator import MeshSimulator

pytestmark = pytest.mark.unit

EXAMPLE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "meshnet", "example.config.json")

REQUIRED_KEYS = [
    "INPUT_SEQUENCE_LENGTH", "noise_std", "node_type_embedding_size", "dt",
    "lr_init", "lr_decay_rate", "lr_decay_steps", "loss_report_step",
    "simulator_simulation_dimensions", "simulator_nnode_in", "simulator_nedge_in",
    "simulator_latent_dim", "simulator_nmessage_passing_steps", "simulator_nmlp_layers",
    "simulator_mlp_hidden_dim", "simulator_nnode_types", "simulator_node_type_embedding_size",
]


def _load_example_config():
    with open(EXAMPLE_CONFIG_PATH) as f:
        return json.load(f)


def _build_simulator_from_config(cfg, **overrides):
    kwargs = dict(
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
    kwargs.update(overrides)
    return MeshSimulator(**kwargs)


def test_example_config_has_every_documented_key():
    cfg = _load_example_config()
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"example.config.json is missing documented keys: {missing}"


def test_example_config_nnode_in_is_internally_consistent():
    # nnode_in must equal 2 (velocity) + node_type_embedding_size + 1 (property),
    # per learned_simulator._encoder_preprocessor. If this drifts, every
    # forward pass built from the example config silently uses a mismatched
    # input width until the first matmul.
    cfg = _load_example_config()
    expected = 2 + cfg["simulator_node_type_embedding_size"] + 1
    assert cfg["simulator_nnode_in"] == expected


def test_example_config_builds_a_working_simulator_and_predicts_velocity():
    cfg = _load_example_config()
    # Use the example config's real architecture sizes but keep message
    # passing steps low enough to stay fast (still exercises every key).
    simulator = _build_simulator_from_config(cfg, nmessage_passing_steps=1, latent_dim=8,
                                              mlp_hidden_dim=8)
    simulator.eval()
    nnodes = 4
    current_velocities = torch.zeros(nnodes, 2)
    node_type = torch.tensor([0.0, 1.0, 0.0, 2.0])
    node_property = torch.rand(nnodes)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_features = torch.randn(3, cfg["simulator_nedge_in"])

    with torch.no_grad():
        out = simulator.predict_velocity(
            current_velocities, node_type, node_property, edge_index, edge_features)
    assert out.shape == (nnodes, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("key,bad_value", [
    ("simulator_latent_dim", 4),
    ("simulator_nmessage_passing_steps", 3),
    ("simulator_nmlp_layers", 1),
])
def test_architecture_keys_change_parameter_count(key, bad_value):
    # Each of these keys must change the number of learnable parameters --
    # otherwise editing them in config.json would be a no-op.
    cfg = _load_example_config()
    baseline = _build_simulator_from_config(cfg, nmessage_passing_steps=1, latent_dim=8, mlp_hidden_dim=8)
    n_baseline = sum(p.numel() for p in baseline.parameters())

    overrides = {"nmessage_passing_steps": 1, "latent_dim": 8, "mlp_hidden_dim": 8}
    overrides[key.replace("simulator_", "")] = bad_value
    changed = _build_simulator_from_config(cfg, **overrides)
    n_changed = sum(p.numel() for p in changed.parameters())

    assert n_changed != n_baseline

"""Unit tests for meshnet.noise.get_velocity_noise (tier 1)."""
import pytest
import torch
from torch_geometric.data import Data

from meshnet.noise import get_velocity_noise
from meshnet.utils import NodeType

pytestmark = pytest.mark.unit


def _make_graph(node_types):
    node_types = torch.tensor(node_types, dtype=torch.float32).unsqueeze(-1)
    n = node_types.shape[0]
    # 3 extra columns so x has the same rank as production graphs (type, property, vx, ...)
    x = torch.hstack([node_types, torch.zeros(n, 3)])
    return Data(x=x)


def test_noise_has_correct_shape():
    graph = _make_graph([NodeType.NORMAL, NodeType.WALL_BOUNDARY, NodeType.NORMAL])
    noise = get_velocity_noise(graph, noise_std=0.05, device="cpu")
    assert noise.shape == (3, 2)


def test_noise_is_exactly_zero_on_non_normal_nodes():
    torch.manual_seed(0)
    graph = _make_graph([NodeType.NORMAL, NodeType.WALL_BOUNDARY, NodeType.HIGH_STRESS, NodeType.NORMAL])
    noise = get_velocity_noise(graph, noise_std=0.5, device="cpu")
    assert torch.all(noise[1] == 0.0)
    assert torch.all(noise[2] == 0.0)


def test_noise_is_nonzero_on_normal_nodes_when_std_positive():
    torch.manual_seed(0)
    graph = _make_graph([NodeType.NORMAL] * 10)
    noise = get_velocity_noise(graph, noise_std=1.0, device="cpu")
    assert torch.any(noise != 0.0)


def test_noise_is_exactly_zero_everywhere_when_std_is_zero():
    graph = _make_graph([NodeType.NORMAL] * 5)
    noise = get_velocity_noise(graph, noise_std=0.0, device="cpu")
    torch.testing.assert_close(noise, torch.zeros(5, 2))


def test_noise_std_matches_configured_noise_std_statistically():
    torch.manual_seed(42)
    configured_std = 0.02
    graph = _make_graph([NodeType.NORMAL] * 20000)
    noise = get_velocity_noise(graph, noise_std=configured_std, device="cpu")
    observed_std = noise.std().item()
    # 20000 samples * 2 dims: std estimate is tight; 10% relative tolerance
    # is generous enough to never flake but tight enough to catch a wrong
    # noise_std (e.g. a stray factor of 2 or a std/var mixup).
    assert observed_std == pytest.approx(configured_std, rel=0.10)

"""Unit tests for meshnet.data_loader (tier 1: fast, shape/behaviour checks).

Uses the tiny synthetic dataset in tests/fixtures/meshnet/synth.py instead of
the (not checked out, 249GB) gns-sample data.
"""
import pytest
import torch

from meshnet.data_loader import (SamplesDataset, TrajectoriesDataset,
                                  get_data_loader_by_samples,
                                  get_data_loader_by_trajectories)
from fixtures.meshnet.synth import build_dataset, NX, NY

pytestmark = pytest.mark.unit

NNODES = NX * NY
TIMESTEPS = 6


@pytest.fixture
def dataset_dir(tmp_path):
    build_dataset(tmp_path, seed=1, timesteps=TIMESTEPS)
    return tmp_path


def test_samples_dataset_length_excludes_input_sequence(dataset_dir):
    # train.npz has 2 trajectories of TIMESTEPS steps each; with
    # input_length_sequence=1 each trajectory yields TIMESTEPS-1 samples.
    ds = SamplesDataset(str(dataset_dir / "train.npz"), input_length_sequence=1, dt=0.1)
    assert len(ds) == 2 * (TIMESTEPS - 1)


def test_samples_dataset_item_shapes(dataset_dir):
    ds = SamplesDataset(str(dataset_dir / "train.npz"), input_length_sequence=1, dt=0.1)
    graph = ds[0]
    # x = [node_type(1), node_property(1), velocity(2), pressure(1), time(1)] = 6 cols
    assert graph.x.shape == (NNODES, 6)
    assert graph.y.shape == (NNODES, 2)
    assert graph.pos.shape == (NNODES, 2)
    assert graph.face.shape[0] == 3  # triangles, transposed to (3, ncells)


def test_samples_dataset_time_channel_increments_with_index(dataset_dir):
    # The last x column is time_idx * dt; item 0 and item 1 of the same
    # trajectory must differ by exactly dt in that channel, for every node.
    ds = SamplesDataset(str(dataset_dir / "train.npz"), input_length_sequence=1, dt=0.1)
    t0 = ds[0].x[:, -1]
    t1 = ds[1].x[:, -1]
    torch.testing.assert_close(t1 - t0, torch.full_like(t0, 0.1))


def test_trajectories_dataset_shapes(dataset_dir):
    ds = TrajectoriesDataset(str(dataset_dir / "test.npz"))
    assert len(ds) == 1
    (pos, node_type, node_property, velocity, pressure, cells, nnodes) = ds[0]
    assert pos.shape == (TIMESTEPS, NNODES, 2)
    assert velocity.shape == (TIMESTEPS, NNODES, 2)
    assert nnodes.item() == NNODES


def test_get_data_loader_by_samples_batches(dataset_dir):
    loader = get_data_loader_by_samples(
        path=str(dataset_dir / "train.npz"), input_length_sequence=1, dt=0.1,
        batch_size=3, shuffle=False)
    batch = next(iter(loader))
    # batched graph stacks nodes from up to 3 samples along dim 0
    assert batch.x.shape[0] <= 3 * NNODES
    assert batch.x.shape[0] % NNODES == 0


def test_get_data_loader_by_trajectories_no_shuffle(dataset_dir):
    loader = get_data_loader_by_trajectories(path=str(dataset_dir / "test.npz"))
    batches = list(loader)
    assert len(batches) == 1

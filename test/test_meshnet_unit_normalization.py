"""Unit tests for meshnet.normalization.Normalizer (tier 1)."""
import pytest
import torch

from meshnet.normalization import Normalizer

pytestmark = pytest.mark.unit


def test_accumulated_mean_and_std_match_hand_computed_values():
    # Two batches with hand-computable statistics:
    # batch1 = [[0.], [2.]], batch2 = [[4.], [6.]] -> combined mean=3, std=sqrt(5)
    normalizer = Normalizer(size=1, device="cpu")
    normalizer(torch.tensor([[0.0], [2.0]]), accumulate=True)
    normalizer(torch.tensor([[4.0], [6.0]]), accumulate=True)

    expected_mean = 3.0
    expected_std = ((0 - 3) ** 2 + (2 - 3) ** 2 + (4 - 3) ** 2 + (6 - 3) ** 2) ** 0.5
    expected_std = (expected_std ** 2 / 4) ** 0.5  # population std = sqrt(mean of squared deviations)

    torch.testing.assert_close(normalizer._mean(), torch.tensor([[expected_mean]]))
    torch.testing.assert_close(normalizer._std_with_epsilon(), torch.tensor([[expected_std]]), atol=1e-5, rtol=1e-5)


def test_accumulate_false_does_not_change_statistics():
    normalizer = Normalizer(size=1, device="cpu")
    normalizer(torch.tensor([[1.0], [3.0]]), accumulate=True)
    mean_before = normalizer._mean().clone()
    normalizer(torch.tensor([[1000.0]]), accumulate=False)
    torch.testing.assert_close(normalizer._mean(), mean_before)


def test_forward_then_inverse_is_identity():
    normalizer = Normalizer(size=2, device="cpu")
    data = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]])
    normalizer(data, accumulate=True)
    normalized = normalizer(data, accumulate=False)
    reconstructed = normalizer.inverse(normalized)
    torch.testing.assert_close(reconstructed, data, atol=1e-5, rtol=1e-5)


def test_std_floor_applied_when_data_is_constant():
    # All-identical inputs give zero variance; std must be floored at std_epsilon,
    # not zero (which would divide by zero elsewhere in the pipeline).
    normalizer = Normalizer(size=1, std_epsilon=1e-3, device="cpu")
    normalizer(torch.tensor([[5.0], [5.0], [5.0]]), accumulate=True)
    assert normalizer._std_with_epsilon().item() == pytest.approx(1e-3)


def test_get_variable_round_trips_into_a_fresh_normalizer():
    normalizer = Normalizer(size=1, device="cpu")
    normalizer(torch.tensor([[2.0], [4.0]]), accumulate=True)
    state = normalizer.get_variable()

    fresh = Normalizer(size=1, device="cpu")
    for key, value in state.items():
        if key == "name":
            continue
        setattr(fresh, key, value)

    torch.testing.assert_close(fresh._mean(), normalizer._mean())
    torch.testing.assert_close(fresh._std_with_epsilon(), normalizer._std_with_epsilon())

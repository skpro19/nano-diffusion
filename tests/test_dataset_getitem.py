import numpy as np
import pytest
import torch

from nano_diffusion.dataset import PushTStateDataset


@pytest.mark.parametrize("index", [0, 1, 2, -1])
def test_getitem_pads_normalizes_and_aligns_sequences(index: int) -> None:
    # Two episodes with deliberately different values at their shared boundary.
    dataset = object.__new__(PushTStateDataset)
    dataset.prediction_horizon = 4
    dataset.observation_horizon = 2
    dataset.state = np.arange(30, dtype=np.float32).reshape(6, 5)
    dataset.action = np.arange(12, dtype=np.float32).reshape(6, 2)
    dataset.sequence_indices = np.array([
        [0, 3, 1, 4],  # Left padding.
        [0, 3, 0, 3],  # Right padding.
        [2, 3, 0, 1],  # Repeat the final frame, never enter the next episode.
        [1, 5, 0, 4],  # No padding.
    ])
    dataset.normalization = {
        "state_mean": np.ones(5),
        "state_std": np.full(5, 2.0),
        "action_mean": np.full(2, 3.0),
        "action_std": np.full(2, 4.0),
    }
    expected_frames = [[0, 0, 1, 2], [0, 1, 2, 2], [2, 2, 2, 2], [1, 2, 3, 4]][index]

    sample = dataset[index]

    assert sample["obs"].shape == (2, 5)
    assert sample["actions"].shape == (4, 2)
    assert all(value.dtype == torch.float32 for value in sample.values())
    np.testing.assert_allclose(
        sample["obs"].numpy(), (dataset.state[expected_frames[:2]] - 1) / 2
    )
    np.testing.assert_allclose(
        sample["actions"].numpy(), (dataset.action[expected_frames] - 3) / 4
    )


def test_observation_horizon_cannot_exceed_prediction_horizon() -> None:
    with pytest.raises(ValueError, match="observation_horizon must not exceed"):
        PushTStateDataset(split="train", observation_horizon=5, prediction_horizon=4)

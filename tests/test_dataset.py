from pathlib import Path

import numpy as np
import pytest

from nano_diffusion.dataset import (
    PushTStateDataset,
    create_sequence_indices,
    downsample_train_episode_mask,
    episode_starts_from_ends,
    get_validation_episode_mask,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = REPO_ROOT / "datasets" / "pusht" / "pusht_cchi_v7_replay.zarr"


pytestmark = pytest.mark.skipif(
    not ZARR_PATH.exists(),
    reason="Push-T zarr missing; run datasets/download_pusht.sh",
)


def test_episode_starts_from_ends() -> None:
    episode_ends = np.array([3, 7, 10], dtype=np.int64)
    starts = episode_starts_from_ends(episode_ends=episode_ends)
    np.testing.assert_array_equal(starts, [0, 3, 7])


def test_split_masks_are_disjoint_and_reproducible() -> None:
    n_episodes = 206
    val_mask_a = get_validation_episode_mask(n_episodes=n_episodes, val_ratio=0.2, seed=42)
    val_mask_b = get_validation_episode_mask(n_episodes=n_episodes, val_ratio=0.2, seed=42)
    np.testing.assert_array_equal(val_mask_a, val_mask_b)
    assert np.sum(val_mask_a) == 41

    train_mask = downsample_train_episode_mask(
        train_mask=~val_mask_a,
        max_train_episodes=90,
        seed=42,
    )
    assert np.sum(train_mask) == 90
    assert not np.any(train_mask & val_mask_a)


def test_train_dataset_initialization() -> None:
    dataset = PushTStateDataset(
        zarr_path=ZARR_PATH,
        split="train",
        observation_horizon=2,
        prediction_horizon=16,
    )
    assert dataset.split == "train"
    assert np.sum(dataset.train_mask) == 206 - np.sum(dataset.val_mask)
    assert len(dataset) > 0
    assert dataset.normalization["state_mean"].shape == (5,)
    assert dataset.normalization["action_std"].shape == (2,)
    assert dataset.sequence_indices.shape[1] == 4


def test_validation_reuses_training_normalization() -> None:
    train = PushTStateDataset(zarr_path=ZARR_PATH, split="train")
    val = PushTStateDataset(
        zarr_path=ZARR_PATH,
        split="val",
        normalization_statistics=train.normalization,
    )
    assert np.sum(val.val_mask) == np.sum(val.episode_mask)
    assert not np.any(val.episode_mask & train.train_mask)
    for key in train.normalization:
        np.testing.assert_allclose(val.normalization[key], train.normalization[key])


def test_validation_requires_normalization_statistics() -> None:
    with pytest.raises(ValueError, match="normalization_statistics"):
        PushTStateDataset(zarr_path=ZARR_PATH, split="val")


def test_create_sequence_indices_respects_episode_boundaries() -> None:
    episode_ends = np.array([4, 8], dtype=np.int64)
    episode_mask = np.array([True, True])
    indices = create_sequence_indices(
        episode_ends=episode_ends,
        sequence_length=4,
        episode_mask=episode_mask,
        pad_before=1,
        pad_after=3,
    )
    assert indices.shape[0] > 0
    for buffer_start, buffer_end, sample_start, sample_end in indices:
        assert 0 <= buffer_start < buffer_end <= 8
        assert sample_end - sample_start == buffer_end - buffer_start

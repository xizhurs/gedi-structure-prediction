import pickle

import numpy as np
import pytest
import torch

from gedi_structure_prediction.data_loader import (
    CustomTransform,
    TileDataset,
    create_tiledataloader_split,
)
from gedi_structure_prediction.training import get_dataloader, get_paths


def test_channels_and_optional_transform(tiles):
    x = np.broadcast_to(np.arange(28)[:, None, None], (28, 16, 16))
    np.save(tiles / "X_0.npy", x)
    sample = TileDataset([tiles / "y_0.npy"])[0]
    assert [item.shape[0] for item in sample] == [20, 6, 2, 3]
    for actual, expected in zip(sample[:3], (range(20), range(20, 26), range(26, 28))):
        assert actual[:, 0, 0].tolist() == list(expected)
        assert actual.dtype == torch.float32


def test_normalization():
    transform = CustomTransform(
        means=np.ones((1, 28, 1, 1)),
        stds=np.full((28,), 2),
        mean_y=np.ones(3),
        std_y=np.full(3, 4),
        augmentation=False,
    )
    x, y = transform(np.full((28, 8, 8), 5), np.full((3, 8, 8), 9))
    assert torch.all(x == 2) and torch.all(y == 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(means=np.ones(28)),
        dict(stds=np.zeros(28), means=np.ones(28)),
        dict(means=np.ones(2)),
        dict(stds=np.full(28, np.nan)),
    ],
)
def test_invalid_statistics(kwargs):
    with pytest.raises(ValueError):
        CustomTransform(**kwargs)


def test_split_and_evaluation_are_repeatable(tiles):
    splits = get_paths(tiles)
    assert [len(part) for part in splits] == [8, 1, 1]
    assert len(set(sum((list(part) for part in splits), []))) == 10
    assert splits == get_paths(tiles)
    train, val, test = get_dataloader(tiles, batch_size=3)
    assert train.batch_size == 3
    assert train.dataset.transform.augmentation
    for loader in (val, test):
        assert not loader.dataset.transform.augmentation
        assert all(
            torch.equal(a, b) for a, b in zip(next(iter(loader)), next(iter(loader)))
        )
    assert next(iter(train))[0].shape == (3, 20, 16, 16)


def test_legacy_scaler(tiles):
    path = tiles / "scaler.pickle"
    with path.open("wb") as file:
        pickle.dump(
            (
                np.zeros((1, 28, 1, 1)),
                np.ones((1, 28, 1, 1)),
                np.zeros((1, 3, 1, 1)),
                np.ones((1, 3, 1, 1)),
            ),
            file,
        )
    assert next(iter(get_dataloader(tiles, path)[1]))[0].shape[1] == 20


def test_bad_tiles_and_loader_arguments(tiles, tmp_path):
    with pytest.raises(ValueError, match="at least 10"):
        get_paths(tiles / "missing")
    with pytest.raises(ValueError):
        create_tiledataloader_split([], 0)
    with pytest.raises(ValueError, match="filename"):
        TileDataset([tiles / "X_0.npy"])[0]
    np.save(tiles / "X_0.npy", np.zeros((2, 8, 8)))
    with pytest.raises(ValueError, match="Expected X"):
        TileDataset([tiles / "y_0.npy"])[0]
    np.save(tiles / "X_0.npy", np.zeros((28, 9, 9)))
    with pytest.raises(ValueError, match="Spatial"):
        TileDataset([tiles / "y_0.npy"])[0]
    (tiles / "X_0.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Missing input"):
        get_paths(tiles)

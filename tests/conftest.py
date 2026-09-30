import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(1)


@pytest.fixture
def tiles(tmp_path):
    rng = np.random.default_rng(1)
    for index in range(10):
        np.save(
            tmp_path / f"X_{index}.npy", rng.normal(size=(28, 16, 16)).astype("float32")
        )
        np.save(
            tmp_path / f"y_{index}.npy", rng.normal(size=(3, 16, 16)).astype("float32")
        )
    return tmp_path

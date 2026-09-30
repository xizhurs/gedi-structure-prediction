from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

bands_x = [
    "s2_band1: 442.7nm",
    "s2_band2: 492.4nm",
    "s2_band3: 559.8nm",
    "s2_band4: 664.6nm",
    "s2_band5: 704.1nm",
    "s2_band6: 740.5nm",
    "s2_band7: 782.8nm",
    "s2_band8: 832.8nm",
    "s2_band8A: 864.7nm",
    "s2_band9: 945.1nm",
    "s2_band11: 1613.7nm",
    "s2_band12: 2202.4nm",
    "s2_NDVI",
    "sunAzimuthAngles",
    "sunZenithAngles",
    "viewAzimuthMean",
    "viewZenithMean",
    "coords_1",
    "coords_2",
    "coords_3",
    "s1_vv_Db_as",
    "s1_vh_Db_as",
    "s1_vv_Db_ds",
    "s1_vh_Db_ds",
    "HH",
    "HV",
    "DEM",
    "slope",
]
bands_y = ["CC", "FHD", "RH95"]


def _statistics(value, channels, name):
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32)
    if value.size != channels:
        raise ValueError(f"{name} must contain {channels} channel statistics")
    value = value.reshape(channels, 1, 1)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


class CustomTransform:
    def __init__(
        self,
        scale_factor=1.0,
        means=None,
        stds=None,
        mean_y=None,
        std_y=None,
        augmentation=True,
        seed=None,
    ):
        self.scale_factor = scale_factor
        self.means = _statistics(means, 28, "means")
        self.stds = _statistics(stds, 28, "stds")
        self.mean_y = _statistics(mean_y, 3, "mean_y")
        self.std_y = _statistics(std_y, 3, "std_y")
        for mean, std in ((self.means, self.stds), (self.mean_y, self.std_y)):
            if (mean is None) != (std is None):
                raise ValueError(
                    "Means and standard deviations must be supplied together"
                )
            if std is not None and (std <= 0).any():
                raise ValueError("Standard deviations must be positive")
        self.augmentation = augmentation
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5, fill_mask=float("nan")),
                A.OneOf([A.Sharpen(), A.Emboss()], p=0.5),
                A.GaussianBlur(p=0.5),
            ],
            seed=seed,
        )

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float32) * self.scale_factor
        y = np.asarray(y, dtype=np.float32) * self.scale_factor
        if self.means is not None:
            x = (x - self.means) / self.stds
        if self.mean_y is not None:
            y = (y - self.mean_y) / self.std_y
        if self.augmentation:
            result = self.transform(
                image=x.transpose(1, 2, 0), mask=y.transpose(1, 2, 0)
            )
            x, y = result["image"].transpose(2, 0, 1), result["mask"].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(y)
        )


class TileDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = [Path(path) for path in file_paths]
        self.transform = transform or CustomTransform(augmentation=False)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        if not path.name.startswith("y_"):
            raise ValueError(f"Target filename must start with y_: {path}")
        y = np.load(path, allow_pickle=False)
        x = np.load(path.with_name("X_" + path.name[2:]), allow_pickle=False)
        if x.ndim != 3 or y.ndim != 3 or x.shape[0] != 28 or y.shape[0] != 3:
            raise ValueError("Expected X shape (28, H, W) and y shape (3, H, W)")
        if x.shape[1:] != y.shape[1:] or any(
            size < 8 or size % 8 for size in x.shape[1:]
        ):
            raise ValueError("Spatial shapes must match and be positive multiples of 8")
        x, y = self.transform(x=x, y=y)
        return x[:20], x[20:26], x[26:], y


def create_tiledataloader_split(
    tile_paths,
    batch_size,
    means=None,
    stds=None,
    mean_y=None,
    std_y=None,
    shuffle=False,
    augmentation=False,
    num_workers=0,
    seed=1,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    transform = CustomTransform(
        means=means,
        stds=stds,
        mean_y=mean_y,
        std_y=std_y,
        augmentation=augmentation,
        seed=seed,
    )
    return DataLoader(
        TileDataset(tile_paths, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )

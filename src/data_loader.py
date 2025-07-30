import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader

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


class CustomTransform:
    def __init__(
        self,
        scale_factor,
        means,
        stds,
        mean_y,
        std_y,
        augmentation=True,
    ):
        self.scale_factor = scale_factor
        self.means = means[0]
        self.stds = stds[0]
        self.mean_y = mean_y[0]
        self.std_y = std_y[0]
        self.augmentation = augmentation

    def __call__(self, x, y):
        x = x * self.scale_factor
        y = y * self.scale_factor

        if self.mean_y is not None:
            x = (x - self.means) / self.stds
            y = (y - self.mean_y) / self.std_y

        if self.augmentation:

            x = x.transpose(1, 2, 0)  # (C H W) to (H W C)
            y = y.transpose(1, 2, 0)  # (C H W) to (H W C)
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                ]
            )

            transformed = transform(image=x, mask=y)
            x = transformed["image"]
            y = transformed["mask"]

            transform_x = A.Compose(
                [
                    A.OneOf([A.Sharpen(), A.Emboss()], p=0.5),
                    A.GaussianBlur(p=0.5),
                ]
            )

            x = transform_x(image=x)["image"]
            x = torch.tensor(x.transpose(2, 0, 1)).float()  # back to (C H W)
            y = torch.tensor(y.transpose(2, 0, 1)).float()  # back to (C H W)
            return (x, y)

        else:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()
        return (x, y)


class TileDataset(Dataset):
    def __init__(
        self,
        file_paths,
        transform=None,
    ):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tile_path = self.file_paths[idx]
        y = np.load(tile_path)
        x = np.load(tile_path.replace("y_", "X_"))
        (x, y) = self.transform(x=x, y=y)

        return (
            x[
                bands_x.index("s2_band1: 442.7nm") : bands_x.index("s1_vv_Db_as")
            ],  # Sentinel-2
            x[
                bands_x.index("s1_vv_Db_as") : bands_x.index("DEM")
            ],  # Sentinel-1 and ALOS
            x[bands_x.index("DEM") :],  # DEM and slope
            y,
        )


def create_tiledataloader_split(
    tile_paths,
    batch_size,
    means=None,
    stds=None,
    mean_y=None,
    std_y=None,
    shuffle=False,
    augmentation=False,
):
    transform = CustomTransform(
        scale_factor=1.0,
        means=means,
        stds=stds,
        mean_y=mean_y,
        std_y=std_y,
        augmentation=augmentation,
    )
    dataset = TileDataset(
        file_paths=tile_paths,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    return dataloader




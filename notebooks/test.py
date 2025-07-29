import pickle
import torch
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from src.data_loader import create_tiledataloader_split
from src.model.mae import MAEUNetPretrain
from torchsummary import summary


def get_device():
    """
    Automatically selects the best available device (GPU if available, otherwise CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


device = get_device()

files = glob("data/processed/y*.npy")
train_paths, val_paths = train_test_split(
    files, test_size=0.2, shuffle=True, random_state=1
)
val_paths, test_paths = train_test_split(
    val_paths, test_size=0.5, shuffle=True, random_state=1
)

scaling_file = "experiments/weights/scaler.pickle"
with open(scaling_file, "rb") as handle:
    means, stds, mean_y, std_y = pickle.load(handle)

batch_size = 1
train_loader = create_tiledataloader_split(
    train_paths,
    batch_size,
    means=means,
    stds=stds,
    mean_y=mean_y,
    std_y=std_y,
    shuffle=True,
    augmentation=True,
)
val_loader = create_tiledataloader_split(
    val_paths,
    batch_size,
    means=means,
    stds=stds,
    mean_y=mean_y,
    std_y=std_y,
    shuffle=True,
    augmentation=True,
)
test_loader = create_tiledataloader_split(
    test_paths,
    batch_size,
    means=means,
    stds=stds,
    mean_y=mean_y,
    std_y=std_y,
    shuffle=True,
    augmentation=True,
)
if __name__ == "__main__":

    for x_s2, x_s1, x_dem, y in val_loader:
        print(np.nanstd(x_s2.numpy(), axis=(0, 2, 3)))
        print(np.nanmean(y.numpy(), axis=(0, 2, 3)))
        break
    model = MAEUNetPretrain(
        in_channels=12,
        dimensions=2,
        out_channels_first_layer=16,
        conv_num_in_layer=[2, 2, 2, 2],
        kernel_size=5,
        normalization="Batch",
        downsampling_type="max",
        residual=True,
        padding_mode="zeros",
        activation="LeakyReLU",
        upsampling_type="conv",
        use_bias=True,
        use_sigmoid=False,
        lr=1e-3,
        lr_decay=0.1,
        weight_decay=1e-4,
        patch_size=128,
        mask_ratio=0.75,
        mask_channels=True,
        sensor_train="s2",
    )
    model = model.to(device)
    summary(model, (12, 128, 128))

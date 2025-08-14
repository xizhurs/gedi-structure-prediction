import pickle
import torch
import lightning as L
from glob import glob
from sklearn.model_selection import train_test_split
from src.data_loader import create_tiledataloader_split
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from src.model.mae import MAEUNetPretrain


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


def get_paths(dire="data/processed"):
    files = glob(dire + "/y*.npy")
    train_paths, val_paths = train_test_split(
        files, test_size=0.2, shuffle=True, random_state=1
    )
    val_paths, test_paths = train_test_split(
        val_paths, test_size=0.5, shuffle=True, random_state=1
    )
    return (train_paths, val_paths, test_paths)


def get_dataloader(
    dire="data/processed",
    scaling_file="experiments/weights/scaler.pickle",
    batch_size=2,
):
    train_paths, val_paths, test_paths = get_paths(dire)
    with open(scaling_file, "rb") as handle:
        means, stds, mean_y, std_y = pickle.load(handle)
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
    return (train_loader, val_loader, test_loader)


def get_model(
    lr=1e-3,
    lr_decay=0.1,
    weight_decay=1e-4,
    sensor_train="s2",
):
    model = MAEUNetPretrain(
        in_channels=20,
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
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        patch_size=128,
        mask_ratio=0.75,
        mask_channels=True,
        sensor_train=sensor_train,
    )
    return model


def pretrain(
    max_epochs=100, lr=1e-3, lr_decay=0.1, weight_decay=1e-4, sensor_train="s2"
):
    (train_loader, val_loader, test_loader) = get_dataloader(
        dire="data/processed",
        scaling_file="experiments/weights/scaler.pickle",
        batch_size=2,
    )

    model_regressor = get_model(
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        sensor_train=sensor_train,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    path_experiment = "experiments/tests"

    name = "unet-gedi-pretrain-testing"

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        # gradient_clip_val=0.5,
        log_every_n_steps=10,
        max_epochs=max_epochs,
        min_epochs=30,
        callbacks=[
            ModelCheckpoint(
                dirpath=path_experiment,
                filename=f"biomass-{name}-" + "{epoch}-{val_loss:.5f}",
                every_n_epochs=1,
                save_top_k=1,
                monitor="val_loss",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=30),
            lr_monitor,
        ],
        enable_progress_bar=True,
        logger=True,
    )
    trainer.fit(model_regressor, train_loader, val_loader)


if __name__ == "__main__":
    pretrain(
        max_epochs=100, lr=1e-3, lr_decay=0.1, weight_decay=1e-4, sensor_train="s2"
    )

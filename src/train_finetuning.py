import pickle
import argparse
import torch
import lightning as L
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from src.data_loader import create_tiledataloader_split
from src.model.unet import Mid_fusion_UNetRegression, FreezeEncoderCallback
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)


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


def get_pretrained_weights(
    s2_file,
    s1_file,
):

    s2_checkpoint = torch.load(s2_file, map_location=torch.device(device))
    pretrained_state_s2 = s2_checkpoint["state_dict"]
    s2_encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in pretrained_state_s2.items()
        if k.startswith("encoder.")
    }
    s1_checkpoint = torch.load(s1_file, map_location=torch.device(device))
    pretrained_state_s1 = s1_checkpoint["state_dict"]
    s1_encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in pretrained_state_s1.items()
        if k.startswith("encoder.")
    }
    return (s2_encoder_state, s1_encoder_state)


def get_model(
    s2_weights=None,
    s1_weights=None,
    lr=1e-3,
    lr_decay=0.1,
    unfreeze_epoch=10,
    weight_decay=1e-4,
):
    model_regressor = Mid_fusion_UNetRegression(
        in_channels_s2=20,
        in_channels_s1=6,
        in_channels_dem=2,
        out_channels=3,
        lr=lr,
        lr_decay=lr_decay,
        total_iters=unfreeze_epoch,
        weight_decay=weight_decay,
        s2_weights=s2_weights,
        s1_weights=s1_weights,
        dem_weights=None,
    )
    return model_regressor


def fine_tuning(
    unfreeze_epoch=10,
    max_epochs=100,
    lr=1e-3,
    lr_decay=0.1,
    weight_decay=1e-4,
    s2_file=Path.cwd()
    / "experiments"
    / "weights"
    / "biomass-unet-s2-pretrain-epoch=98-val_loss=0.05282.ckpt",
    s1_file=Path.cwd()
    / "experiments"
    / "weights"
    / "biomass-unet-s1alos-pretrain-epoch=98-val_loss=0.06877.ckpt",
):
    (train_loader, val_loader, test_loader) = get_dataloader(
        dire="data/processed",
        scaling_file="experiments/weights/scaler.pickle",
        batch_size=2,
    )

    (s2_encoder_state, s1_encoder_state) = get_pretrained_weights(s2_file, s1_file)
    model_regressor = get_model(
        s2_weights=s2_encoder_state,
        s1_weights=s1_encoder_state,
        lr=lr,
        lr_decay=lr_decay,
        unfreeze_epoch=unfreeze_epoch,
        weight_decay=weight_decay,
    )

    unfreeze_callback = FreezeEncoderCallback(unfreeze_epoch=unfreeze_epoch)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    path_experiment = "experiments/tests"

    name = "unet-gedi-mid-fusion-testing"

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
            unfreeze_callback,
            lr_monitor,
        ],
        enable_progress_bar=True,
        logger=True,
    )
    trainer.fit(model_regressor, train_loader, val_loader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GEDI structure prediction model"
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=10,
        help="epoch to unfreeze encoder (default: 10)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="maximum number of epochs (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--lr-decay", type=float, default=0.1, help="learning rate decay (default: 0.1)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="weight decay (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="batch size (default: 2)"
    )
    parser.add_argument(
        "--s2-file",
        type=str,
        default=Path.cwd()
        / "experiments"
        / "weights"
        / "s2-pretrain-epoch=98-val_loss=0.05282.ckpt",
        metavar="PATH",
        dest="s2_file",
        help="path to S2 pretrained weights",
    )
    parser.add_argument(
        "--s1-file",
        type=str,
        default=Path.cwd()
        / "experiments"
        / "weights"
        / "s1alos-pretrain-epoch=98-val_loss=0.06877.ckpt",
        metavar="PATH",
        dest="s1_file",
        help="path to S1 pretrained weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fine_tuning(
        unfreeze_epoch=args.unfreeze_epoch,
        max_epochs=args.max_epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        s2_file=Path.cwd() / args.s2_file,
        s1_file=Path.cwd() / args.s1_file,
    )

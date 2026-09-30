"""Pretrain a masked autoencoder for one satellite sensor."""

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from gedi_structure_prediction.model.mae import MAEUNetPretrain
from gedi_structure_prediction.training import get_dataloader, parser, validate_args


def get_model(lr=1e-3, lr_decay=0.1, weight_decay=1e-4, sensor_train="s2"):
    if sensor_train not in ("s1", "s2"):
        raise ValueError("sensor_train must be s1 or s2")
    return MAEUNetPretrain(
        in_channels={"s2": 20, "s1": 6}[sensor_train],
        dimensions=2,
        out_channels_first_layer=16,
        conv_num_in_layer=[2, 2, 2, 2],
        kernel_size=5,
        normalization="Batch",
        downsampling_type="max",
        residual=True,
        padding_mode="zeros",
        activation="LeakyReLU",
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        sensor_train=sensor_train,
    )


def pretrain(
    max_epochs=100,
    lr=1e-3,
    lr_decay=0.1,
    weight_decay=1e-4,
    sensor_train="s2",
    data_dir="data/processed",
    scaling_file=None,
    output_dir="experiments/runs",
    batch_size=2,
    num_workers=0,
    seed=1,
    accelerator="auto",
    fast_dev_run=False,
):
    L.seed_everything(seed, workers=True)
    train, val, _ = get_dataloader(
        data_dir, scaling_file, batch_size, num_workers, seed
    )
    model = get_model(lr, lr_decay, weight_decay, sensor_train)
    trainer = L.Trainer(
        default_root_dir=output_dir,
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1),
            EarlyStopping(monitor="val_loss", patience=30),
        ],
    )
    trainer.fit(model, train, val)
    return model


def parse_args(argv=None):
    result = parser("Pretrain GEDI structure prediction model")
    result.add_argument("--sensor-train", choices=["s1", "s2"], default="s2")
    return validate_args(result, result.parse_args(argv))


def main(argv=None):
    pretrain(**vars(parse_args(argv)))


if __name__ == "__main__":
    main()

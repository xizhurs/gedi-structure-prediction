"""Fine-tune a multi-sensor UNet using pretrained encoders."""

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from gedi_structure_prediction.model.unet import (
    FreezeEncoderCallback,
    Mid_fusion_UNetRegression,
)
from gedi_structure_prediction.training import (
    get_dataloader,
    nonnegative_int,
    parser,
    validate_args,
)


def get_pretrained_weights(s2_file, s1_file):
    states = []
    for path in (s2_file, s1_file):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = {
            key.removeprefix("encoder."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("encoder.")
        }
        if not state:
            raise ValueError(f"No encoder weights in checkpoint: {path}")
        states.append(state)
    return tuple(states)


def get_model(
    s2_weights=None,
    s1_weights=None,
    lr=1e-3,
    lr_decay=0.1,
    unfreeze_epoch=10,
    weight_decay=1e-4,
):
    return Mid_fusion_UNetRegression(
        lr=lr,
        lr_decay=lr_decay,
        total_iters=max(1, unfreeze_epoch),
        weight_decay=weight_decay,
        s2_weights=s2_weights,
        s1_weights=s1_weights,
    )


def fine_tuning(
    unfreeze_epoch=10,
    max_epochs=100,
    lr=1e-3,
    lr_decay=0.1,
    weight_decay=1e-4,
    s2_file=None,
    s1_file=None,
    data_dir="data/processed",
    scaling_file=None,
    output_dir="experiments/runs",
    batch_size=2,
    num_workers=0,
    seed=1,
    accelerator="auto",
    fast_dev_run=False,
):
    if (s2_file is None) != (s1_file is None):
        raise ValueError(
            "Supply both S1 and S2 checkpoints, or neither to train from scratch"
        )
    L.seed_everything(seed, workers=True)
    train, val, _ = get_dataloader(
        data_dir, scaling_file, batch_size, num_workers, seed
    )
    weights = (
        (None, None) if s2_file is None else get_pretrained_weights(s2_file, s1_file)
    )
    model = get_model(
        *weights,
        lr=lr,
        lr_decay=lr_decay,
        unfreeze_epoch=unfreeze_epoch,
        weight_decay=weight_decay,
    )
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=1),
        EarlyStopping(monitor="val_loss", patience=30),
    ]
    if s2_file is not None:
        callbacks.append(FreezeEncoderCallback(unfreeze_epoch))
    trainer = L.Trainer(
        default_root_dir=output_dir,
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
    )
    trainer.fit(model, train, val)
    return model


def parse_args(argv=None):
    result = parser("Fine-tune GEDI structure prediction model")
    result.add_argument("--unfreeze-epoch", type=nonnegative_int, default=10)
    result.add_argument("--s2-file", type=Path)
    result.add_argument("--s1-file", type=Path)
    args = validate_args(result, result.parse_args(argv))
    if (args.s2_file is None) != (args.s1_file is None):
        result.error("Supply both --s2-file and --s1-file")
    return args


def main(argv=None):
    fine_tuning(**vars(parse_args(argv)))


if __name__ == "__main__":
    main()

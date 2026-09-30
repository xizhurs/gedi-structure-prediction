"""Shared command-line configuration and reproducible data setup."""

import argparse
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split

from gedi_structure_prediction.data_loader import create_tiledataloader_split


def get_paths(dire="data/processed", seed=1):
    files = sorted(Path(dire).glob("y_*.npy"))
    if len(files) < 10:
        raise ValueError(f"Need at least 10 paired tiles for an 80/10/10 split: {dire}")
    for path in files:
        partner = path.with_name("X_" + path.name[2:])
        if not partner.is_file():
            raise FileNotFoundError(f"Missing input tile: {partner}")
    train, holdout = train_test_split(files, test_size=0.2, random_state=seed)
    val, test = train_test_split(holdout, test_size=0.5, random_state=seed)
    return train, val, test


def get_dataloader(
    dire="data/processed", scaling_file=None, batch_size=2, num_workers=0, seed=1
):
    stats = (None, None, None, None)
    if scaling_file is not None:
        # Legacy scaler artifacts are pickle files; only load trusted files.
        with Path(scaling_file).open("rb") as handle:
            stats = pickle.load(handle)
        if len(stats) != 4:
            raise ValueError("Scaler must contain means, stds, mean_y, std_y")
    return tuple(
        create_tiledataloader_split(
            paths,
            batch_size,
            *stats,
            shuffle=index == 0,
            augmentation=index == 0,
            num_workers=num_workers,
            seed=seed,
        )
        for index, paths in enumerate(get_paths(dire, seed))
    )


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    result.add_argument(
        "--scaling-file", type=Path, help="Trusted pickle scaler; omit for raw data"
    )
    result.add_argument("--output-dir", type=Path, default=Path("experiments/runs"))
    result.add_argument("--batch-size", type=positive_int, default=2)
    result.add_argument("--num-workers", type=nonnegative_int, default=0)
    result.add_argument("--seed", type=nonnegative_int, default=1)
    result.add_argument("--max-epochs", type=positive_int, default=100)
    result.add_argument("--lr", type=float, default=1e-3)
    result.add_argument("--lr-decay", type=float, default=0.1)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument(
        "--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="auto"
    )
    result.add_argument(
        "--fast-dev-run", action="store_true", help="Run one train and validation batch"
    )
    return result


def validate_args(result, args):
    if not 0 < args.lr < float("inf") or not 0 < args.lr_decay <= 1:
        result.error("lr must be finite and positive; lr-decay must be in (0, 1]")
    if not 0 <= args.weight_decay < float("inf"):
        result.error("weight-decay must be finite and nonnegative")
    return args

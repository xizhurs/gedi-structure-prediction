import subprocess
import sys

import pytest
import torch

from gedi_structure_prediction import pretrain, train_finetuning


@pytest.mark.parametrize("module", [pretrain, train_finetuning])
def test_cli_wires_arguments(module, monkeypatch, tmp_path):
    captured = {}
    name = "pretrain" if module is pretrain else "fine_tuning"
    monkeypatch.setattr(module, name, lambda **kwargs: captured.update(kwargs))
    module.main(
        [
            "--batch-size",
            "7",
            "--num-workers",
            "2",
            "--seed",
            "42",
            "--data-dir",
            str(tmp_path),
            "--fast-dev-run",
        ]
    )
    assert captured["batch_size"] == 7
    assert captured["num_workers"] == 2
    assert captured["seed"] == 42
    assert captured["data_dir"] == tmp_path
    assert captured["fast_dev_run"]


@pytest.mark.parametrize(
    "args",
    [
        ["--batch-size", "0"],
        ["--num-workers", "-1"],
        ["--lr", "nan"],
        ["--lr-decay", "2"],
        ["--weight-decay", "-1"],
    ],
)
def test_invalid_cli(args):
    with pytest.raises(SystemExit):
        pretrain.parse_args(args)


def test_checkpoint_pair_required():
    with pytest.raises(SystemExit):
        train_finetuning.parse_args(["--s1-file", "one.ckpt"])
    with pytest.raises(ValueError):
        train_finetuning.fine_tuning(s1_file="one.ckpt")


@pytest.mark.parametrize("module", [pretrain, train_finetuning])
def test_module_help_outside_repository(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", module.__name__, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--batch-size" in result.stdout


@pytest.mark.parametrize("module", [pretrain, train_finetuning])
def test_training_entry_smoke(module, tiles, tmp_path):
    module.main(
        [
            "--data-dir",
            str(tiles),
            "--output-dir",
            str(tmp_path / "output"),
            "--accelerator",
            "cpu",
            "--fast-dev-run",
        ]
    )


def test_finetuning_with_pretrained_checkpoints(tiles, tmp_path):
    for sensor in ("s1", "s2"):
        model = pretrain.get_model(sensor_train=sensor)
        torch.save({"state_dict": model.state_dict()}, tmp_path / f"{sensor}.ckpt")
    train_finetuning.main(
        [
            "--data-dir",
            str(tiles),
            "--output-dir",
            str(tmp_path / "output"),
            "--s1-file",
            str(tmp_path / "s1.ckpt"),
            "--s2-file",
            str(tmp_path / "s2.ckpt"),
            "--unfreeze-epoch",
            "0",
            "--accelerator",
            "cpu",
            "--fast-dev-run",
        ]
    )

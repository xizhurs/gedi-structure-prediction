import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gedi_structure_prediction.model.mae import MAEUNetPretrain
from gedi_structure_prediction.model.unet import (
    FreezeEncoderCallback,
    Mid_fusion_UNetRegression,
)
from gedi_structure_prediction.pretrain import get_model
from gedi_structure_prediction.train_finetuning import get_pretrained_weights


def batch_loader():
    return DataLoader(
        TensorDataset(*(torch.randn(2, c, 16, 16) for c in (20, 6, 2, 3))), batch_size=2
    )


@pytest.mark.parametrize("sensor,channels", [("s1", 6), ("s2", 20)])
def test_pretraining_updates_and_checkpoint(tmp_path, sensor, channels):
    model = get_model(sensor_train=sensor)
    before = model.reconstruction_head.weight.detach().clone()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, batch_loader(), batch_loader())
    assert not torch.equal(before, model.reconstruction_head.weight)
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    restored = MAEUNetPretrain.load_from_checkpoint(path, weights_only=True)
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
    )
    output, mask = restored(torch.randn(2, channels, 16, 16))
    assert output.shape == mask.shape == (2, channels, 16, 16)
    assert mask.sum().item() == 2 * channels * 16 * 16 * 0.75
    state1, state2 = get_pretrained_weights(path, path)
    assert state1.keys() == state2.keys() == model.encoder.state_dict().keys()


def test_finetuning_updates_and_freezing(tmp_path):
    model = Mid_fusion_UNetRegression()
    callback = FreezeEncoderCallback(unfreeze_epoch=1)
    callback.freeze_before_training(model)
    assert not any(p.requires_grad for p in model.model.encoder_s2.parameters())
    optimizer = model.configure_optimizers()["optimizer"]
    callback.finetune_function(model, 1, optimizer)
    assert all(p.requires_grad for p in model.model.encoder_s1.parameters())
    before = model.model.regression_head.weight.detach().clone()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, batch_loader(), batch_loader())
    assert not torch.equal(before, model.model.regression_head.weight)
    metrics = trainer.callback_metrics
    expected = 0.9 * torch.ones(3) + 0.1 * torch.stack(
        [metrics[f"train_{task}_loss"] for task in ("canopy_cover", "fhd", "height")]
    )
    assert torch.allclose(model.dwa_loss.prev_losses, expected)
    path = tmp_path / "fine.ckpt"
    trainer.save_checkpoint(path)
    restored = Mid_fusion_UNetRegression.load_from_checkpoint(path, weights_only=True)
    model.eval()
    restored.eval()
    inputs = next(iter(batch_loader()))[:3]
    with torch.no_grad():
        assert torch.equal(model(*inputs), restored(*inputs))
        inputs[0][0, 0, 0, 0] = float("nan")
        assert torch.isfinite(model(*inputs)).all()


def test_shared_mask_and_invalid_sensor():
    model = get_model()
    model.mask_channels = False
    mask = model.create_mask((2, 20, 16, 16), torch.device("cpu"))
    assert torch.equal(mask[:, 0], mask[:, 19])
    with pytest.raises(ValueError):
        get_model(sensor_train="invalid")


def test_pretrained_encoders_are_loaded():
    s2 = get_model(sensor_train="s2").encoder.state_dict()
    s1 = get_model(sensor_train="s1").encoder.state_dict()
    model = Mid_fusion_UNetRegression(s2_weights=s2, s1_weights=s1)
    assert all(
        torch.equal(value, model.model.encoder_s2.state_dict()[key])
        for key, value in s2.items()
    )
    assert all(
        torch.equal(value, model.model.encoder_s1.state_dict()[key])
        for key, value in s1.items()
    )

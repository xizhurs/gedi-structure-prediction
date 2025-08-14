import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from typing import Dict
from src.utils import DWALoss, custom_mse_loss
from src.model.mae import Encoder, Decoder, EncodingBlock, ConvolutionalBlock
from lightning.pytorch.callbacks import BaseFinetuning


class FreezeEncoderCallback(BaseFinetuning):
    def __init__(self, unfreeze_epoch=20):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def freeze_before_training(self, pl_module):
        pl_module.freeze_encoder()
        for name, param in pl_module.model.encoder_s2.named_parameters():
            if param.requires_grad:
                print(f"  ✅ Trainable: {name}")
            else:
                print(f"  ❌ Frozen: {name}")

    def finetune_function(
        self, pl_module, current_epoch, optimizer, optimizer_idx=None
    ):
        if current_epoch == self.unfreeze_epoch:
            print(f"Unfreezing encoder at epoch {current_epoch}")
            pl_module.unfreeze_encoder()
            for name, param in pl_module.model.encoder_s2.named_parameters():
                if param.requires_grad:
                    print(f"  ✅ Trainable: {name}")
                else:
                    print(f"  ❌ Frozen: {name}")


class Mid_fusion_UNetRegression(L.LightningModule):
    def __init__(
        self,
        in_channels_s2=20,
        in_channels_s1=6,
        in_channels_dem=2,
        out_channels=3,
        lr=1e-4,
        lr_decay=0.1,
        total_iters=50,
        weight_decay=1e-4,
        single_loss=None,
        s2_weights=None,
        s1_weights=None,
        dem_weights=None,
    ):

        super().__init__()
        self.lr = lr
        self.total_iters = total_iters
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.single_loss = single_loss
        self.s2_weights = s2_weights
        self.s1_weights = s1_weights
        self.dwa_loss = DWALoss(num_tasks=3)
        # self.save_hyperparameters()

        self.model = UNetRegression(
            in_channels_s2,
            in_channels_s1,
            in_channels_dem,
            out_regression_num=out_channels,
            dimensions=2,
            out_channels_first=16,
            conv_num_in_layer=[2, 2, 2, 2],
            kernel_size=5,
        )

        if s2_weights is not None:
            self.model.encoder_s2.load_state_dict(s2_weights)
            self.model.encoder_s1.load_state_dict(s1_weights)

    def freeze_encoder(self):
        for param in self.model.encoder_s2.parameters():
            param.requires_grad = False
        for param in self.model.encoder_s1.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.model.encoder_s2.parameters():
            param.requires_grad = True
        for param in self.model.encoder_s1.parameters():
            param.requires_grad = True

    def forward(self, x_s2, x_s1, x_dem):
        return self.model(x_s2, x_s1, x_dem)

    def training_step(self, batch, batch_idx):
        x_s2, x_s1, x_dem, y_all = batch
        logits_regression = self.forward(x_s2, x_s1, x_dem)

        y_regression = y_all[:, 0:3]

        y_regression = y_regression.view(y_regression.size(0), y_regression.size(1), -1)
        logits_regression = logits_regression.view(
            logits_regression.size(0), logits_regression.size(1), -1
        )

        loss_canopy_cover = custom_mse_loss(logits_regression[:, 0], y_regression[:, 0])
        loss_fhd = custom_mse_loss(logits_regression[:, 1], y_regression[:, 1])
        loss_height = custom_mse_loss(logits_regression[:, 2], y_regression[:, 2])

        total_loss, task_losses = self.dwa_loss(
            [loss_canopy_cover, loss_fhd, loss_height]
        )
        self.log("train_height_loss", loss_height)
        self.log("train_canopy_cover_loss", loss_canopy_cover)
        self.log("train_fhd_loss", loss_fhd)

        for i, w in enumerate(self.dwa_loss.loss_weights):
            self.log(f"train_task_{i}_weight", w)
        self.log_dict(
            {"train_loss": total_loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, val_batch, batch_idx):
        x_s2, x_s1, x_dem, y_all = val_batch
        logits_regression = self.forward(x_s2, x_s1, x_dem)

        y_regression = y_all[:, 0:3]

        y_regression = y_regression.view(y_regression.size(0), y_regression.size(1), -1)
        logits_regression = logits_regression.view(
            logits_regression.size(0), logits_regression.size(1), -1
        )

        loss_canopy_cover = custom_mse_loss(logits_regression[:, 0], y_regression[:, 0])
        loss_fhd = custom_mse_loss(logits_regression[:, 1], y_regression[:, 1])
        loss_height = custom_mse_loss(logits_regression[:, 2], y_regression[:, 2])

        total_loss, task_losses = self.dwa_loss(
            [loss_canopy_cover, loss_fhd, loss_height]
        )
        self.log("val_height_loss", loss_height)
        self.log("val_canopy_cover_loss", loss_canopy_cover)
        self.log("val_fhd_loss", loss_fhd)

        for i, w in enumerate(self.dwa_loss.loss_weights):
            self.log(f"val_task_{i}_weight", w)
        self.log_dict(
            {"val_loss": total_loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_train_epoch_end(self):
        if self.single_loss is None:
            # Update DWA weights at the end of each epoch
            height_loss = self.trainer.callback_metrics["train_height_loss"]
            canopy_cover_loss = self.trainer.callback_metrics["train_canopy_cover_loss"]
            fhd_loss = self.trainer.callback_metrics["train_fhd_loss"]
            self.dwa_loss.update_weights(
                torch.stack([height_loss, canopy_cover_loss, fhd_loss])
            )

    def configure_optimizers(self):
        # params = ([p for p in self.parameters()] + [log_var_a] + [log_var_b])
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.lr_decay,
            total_iters=self.total_iters,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class UNetRegression(nn.Module):
    def __init__(
        self,
        in_channels_s2,
        in_channels_s1,
        in_channels_dem,
        out_regression_num,
        dimensions,
        out_channels_first,
        conv_num_in_layer,
        kernel_size=5,
        normalization="Batch",
        downsampling_type="max",
        residual=True,
        padding_mode="zeros",
        activation="LeakyReLU",
        upsampling_type: str = "conv",
        use_bias=True,
        use_sigmoid=False,
    ):
        super().__init__()
        shared_args: Dict = dict(
            residual=residual,
            kernel_size=kernel_size,
            normalization=normalization,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        self.encoder_s2 = Encoder(
            in_channels_s2,
            out_channels_first=out_channels_first,
            dimensions=dimensions,
            conv_num_in_layer=conv_num_in_layer,
            downsampling_type=downsampling_type,
            **shared_args,
        )
        self.encoder_s1 = Encoder(
            in_channels_s1,
            out_channels_first=out_channels_first,
            dimensions=dimensions,
            conv_num_in_layer=conv_num_in_layer,
            downsampling_type=downsampling_type,
            **shared_args,
        )
        self.encoder_dem = Encoder(
            in_channels_dem,
            out_channels_first=out_channels_first,
            dimensions=dimensions,
            conv_num_in_layer=conv_num_in_layer,
            downsampling_type=downsampling_type,
            **shared_args,
        )

        bottleneck_channels = (
            self.encoder_s2.out_channels
            + self.encoder_s1.out_channels
            + self.encoder_dem.out_channels
        )

        self.joiner = EncodingBlock(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels * 2,
            dimensions=dimensions,
            conv_num=conv_num_in_layer[-1],
            num_block=len(conv_num_in_layer),
            downsampling_type=None,
            **shared_args,
        )

        # decoder_skips = [
        #     out_channels_first * (2**i) for i in range(len(conv_num_in_layer) - 1)
        # ][::-1]
        self.decoder = Decoder(
            # in_channels_skip_connection=bottleneck_channels * 2,
            in_channels_skip_connection=out_channels_first
            * 3
            * (2 ** (len(conv_num_in_layer) - 2)),
            dimensions=dimensions,
            upsampling_type=upsampling_type,
            conv_num_in_layer=conv_num_in_layer[1:],
            **shared_args,
        )

        self.output = ConvolutionalBlock(
            dimensions=dimensions,
            in_channels=self.decoder.out_channels,
            out_channels=128,
            kernel_size=1,
            normalization=None,
            activation=None,
            use_bias=True,
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1), nn.LeakyReLU(0.2)
        )

        self.regression_head = nn.Conv2d(256, out_regression_num, kernel_size=1)
        self.classification_head = nn.Conv2d(256, 1, kernel_size=1)

        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_s2, x_s1, x_dem):
        skips_s2, encoded_s2 = self.encoder_s2(x_s2)
        skips_s1, encoded_s1 = self.encoder_s1(x_s1)
        skips_dem, encoded_dem = self.encoder_dem(x_dem)

        # Fuse at bottleneck
        encoded_combined = torch.cat([encoded_s2, encoded_s1, encoded_dem], dim=1)

        skip_combined = [
            torch.cat([s2, s1, dem], dim=1)
            for s2, s1, dem in zip(skips_s2, skips_s1, skips_dem)
        ]
        # print(skip_combined.shape)
        x = self.joiner(encoded_combined)
        x = self.decoder(skip_combined, x)
        x = self.output(x)
        x = self.conv1(x)
        regression_output = self.regression_head(x)
        # _ = self.classification_head(x)
        if self.use_sigmoid:
            return self.sigmoid(regression_output)
        else:
            return regression_output


# Unet in this file adapted from
# https://github.com/fepegar/unet/blob/master/unet/decoding.py

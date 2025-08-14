import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from typing import Any, Dict, List, Optional, Sequence
from torch.nn import Module
from src.utils import mse_loss_mae


class MAEUNetPretrain(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        dimensions: int,
        out_channels_first_layer: int,
        conv_num_in_layer: List[int],
        kernel_size: int,
        normalization: str,
        downsampling_type: str,
        residual: bool,
        padding_mode: str,
        activation: Optional[str],
        upsampling_type: str = "conv",
        use_bias: bool = True,
        use_sigmoid: bool = False,
        lr=1e-4,
        lr_decay=0.1,
        weight_decay=1e-4,
        patch_size: int = 128,
        mask_ratio: float = 0.75,
        mask_channels: bool = True,
        sensor_train="s2",
    ):
        super(MAEUNetPretrain, self).__init__()

        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.use_sigmoid = use_sigmoid
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_channels = mask_channels
        self.sensor_train = sensor_train
        shared_options: Dict = dict(
            residual=residual,
            kernel_size=kernel_size,
            normalization=normalization,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels_first=out_channels_first_layer,
            dimensions=dimensions,
            conv_num_in_layer=conv_num_in_layer,
            downsampling_type=downsampling_type,
            **shared_options,
        )

        in_channels = self.encoder.out_channels
        in_channels_skip_connection = in_channels
        self.joiner = EncodingBlock(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            dimensions=dimensions,
            conv_num=conv_num_in_layer[-1],
            num_block=len(conv_num_in_layer),
            downsampling_type=None,
            **shared_options,
        )

        conv_num_in_layer.reverse()
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type=upsampling_type,
            conv_num_in_layer=conv_num_in_layer[1:],
            **shared_options,
        )
        # Simple reconstruction head (could be replaced by a deeper decoder)
        self.reconstruction_head = nn.Conv2d(
            in_channels=self.decoder.out_channels,
            out_channels=self.in_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device

        # Create mask
        mask = self.create_mask(x.shape, device)
        nan_mask = torch.isnan(x)
        combined_mask = torch.logical_or(mask, nan_mask)
        x_masked = x.clone()
        x_masked[combined_mask] = 0  # Set masked pixels or channels to zero

        # Forward pass
        skips, encoding = self.encoder(x_masked)
        x_rec = self.joiner(encoding)
        x_rec = self.decoder(skips, x_rec)
        x_rec = self.reconstruction_head(x_rec)

        return x_rec, combined_mask

    def create_mask(self, shape, device) -> torch.Tensor:
        B, C, H, W = shape
        mask = torch.zeros((B, C, H, W), dtype=torch.bool, device=device)
        num_mask = int(H * W * self.mask_ratio)

        for b in range(B):
            if self.mask_channels:
                for c in range(C):
                    indices = torch.randperm(H * W)[:num_mask]
                    mask[b, c].view(-1)[indices] = True
            else:
                indices = torch.randperm(H * W)[:num_mask]
                mask[b].view(C, -1)[:, indices] = True

        return mask

    def training_step(self, batch, batch_idx):
        x_s2, x_s1, x_dem, _ = batch
        if self.sensor_train == "s2":
            x_input = x_s2
        elif self.sensor_train == "s1":
            x_input = x_s1
        elif self.sensor_train == "dem":
            x_input = x_dem
        else:
            x_input = torch.cat((x_s2, x_s1, x_dem), dim=1)
        x_rec, mask = self.forward(x_input)

        loss = mse_loss_mae(x_rec[mask], x_input[mask])
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        x_s2, x_s1, x_dem, _ = val_batch
        if self.sensor_train == "s2":
            x_input = x_s2
        elif self.sensor_train == "s1":
            x_input = x_s1
        elif self.sensor_train == "dem":
            x_input = x_dem
        else:
            x_input = torch.cat((x_s2, x_s1, x_dem), dim=1)
        x_rec, mask = self.forward(x_input)

        loss = mse_loss_mae(x_rec[mask], x_input[mask])

        self.log_dict(
            {"val_loss": loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        # params = ([p for p in self.parameters()] + [log_var_a] + [log_var_b])
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=self.lr_decay, total_iters=50
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        block = nn.ModuleList()
        conv = getattr(nn, f"Conv{dimensions}d")

        conv_layer = [
            conv(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size + 1) // 2 - 1,
                padding_mode=padding_mode,
                bias=use_bias,
            )
        ]

        norm_layer: List[Module] = []
        if normalization is not None:
            if normalization == "Batch":
                norm = getattr(nn, f"BatchNorm{dimensions}d")
                norm_layer.append(norm(out_channels))
            elif normalization == "Group":
                norm_layer.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
            elif normalization == "InstanceNorm3d":
                norm_layer.append(
                    nn.InstanceNorm3d(
                        num_features=out_channels, affine=True, track_running_stats=True
                    )
                )

        activation_layer: List[Module] = []
        if activation is not None:
            if activation == "ReLU":
                activation_layer.append(nn.ReLU())
            elif activation == "LeakyReLU":
                activation_layer.append(nn.LeakyReLU(0.2))

        block.extend(conv_layer)
        block.extend(norm_layer)
        block.extend(activation_layer)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)  # type: ignore


CHANNELS_DIMENSION = 1


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        conv_num_in_layer: Sequence[int],
        upsampling_type: str,
        residual: bool,
        normalization: Optional[str],
        kernel_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()
        self.decoding_blocks = nn.ModuleList()
        for idx, conv_num in enumerate(conv_num_in_layer):
            decoding_block = DecodingBlock(
                in_channels_skip_connection=in_channels_skip_connection,
                dimensions=dimensions,
                upsampling_type=upsampling_type,
                normalization=normalization,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                activation=activation,
                residual=residual,
                conv_num=conv_num,
                block_num=idx,
                use_bias=use_bias,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2

        self.out_channels = in_channels_skip_connection * 4

    def forward(self, skip_connections, x):
        # print(f"x type: {type(x)}, length: {len(x)}")
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        residual: bool,
        conv_num: int,
        block_num: int,
        normalization: Optional[str] = "Group",
        kernel_size: int = 5,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.residual = residual

        if upsampling_type == "conv":
            if block_num == 0:
                in_channels = in_channels_skip_connection * 2
                out_channels = in_channels_skip_connection
            else:
                in_channels = in_channels_skip_connection * 4
                out_channels = in_channels_skip_connection * 2
            conv = getattr(nn, f"ConvTranspose{dimensions}d")
            self.upsample = conv(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            )
        else:
            raise NotImplementedError()

        if block_num == 0:
            in_channels_first = in_channels_skip_connection * 2
            out_channels = in_channels_skip_connection * 2
        else:
            in_channels_first = in_channels_skip_connection * 3
            out_channels = in_channels_skip_connection * 2

        options: Dict[str, Any] = dict(
            normalization=normalization,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )

        conv_blocks = [
            ConvolutionalBlock(dimensions, in_channels_first, out_channels, **options)
        ]

        for _ in range(conv_num - 1):
            conv_blocks.append(
                ConvolutionalBlock(
                    dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    **options,
                )
            )

        self.conv_blocks = nn.Sequential(*conv_blocks)

        if self.residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)

        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv_blocks(x)
            x += connection
        else:
            x = self.conv_blocks(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        conv_num_in_layer: List[int],
        residual: bool,
        kernel_size: int,
        normalization: str,
        downsampling_type: str,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        num_encoding_blocks = len(conv_num_in_layer) - 1
        out_channels = out_channels_first
        for idx in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dimensions=dimensions,
                conv_num=conv_num_in_layer[idx],
                residual=residual,
                normalization=normalization,
                kernel_size=kernel_size,
                downsampling_type=downsampling_type,
                padding_mode=padding_mode,
                activation=activation,
                num_block=idx,
                use_bias=use_bias,
            )
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels
                out_channels = in_channels * 2
            elif dimensions == 3:
                in_channels = out_channels
                out_channels = in_channels * 2

            self.out_channels = self.encoding_blocks[-1].out_channels

    def forward(self, x):
        skips = []
        for encoding_block in self.encoding_blocks:  # nn.ModuleList need to iterate!!!!
            x, skip = encoding_block(x)
            skips.append(skip)
        return skips, x


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int,
        residual: bool,
        normalization: Optional[str],
        conv_num: int,
        num_block: int,
        kernel_size: int = 5,
        downsampling_type: Optional[str] = "conv",
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        use_bias: bool = True,
    ):
        super().__init__()

        self.num_block = num_block
        self.residual = residual
        opts: Dict = dict(
            normalization=normalization,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation,
            use_bias=use_bias,
        )
        conv_blocks = [
            ConvolutionalBlock(dimensions, in_channels, out_channels, **opts)
        ]

        for _ in range(conv_num - 1):
            conv_blocks.append(
                ConvolutionalBlock(
                    dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    **opts,
                )
            )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsampling_type = downsampling_type

        self.downsample = None
        if downsampling_type == "max":
            maxpool = getattr(nn, f"MaxPool{dimensions}d")
            self.downsample = maxpool(kernel_size=2)
        elif downsampling_type == "conv":
            self.downsample = nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            )

        self.out_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        if self.residual:
            residual_layer = self.conv_residual(x)
            x = self.conv_blocks(x)
            x += residual_layer
        else:
            x = self.conv_blocks(x)

        if self.downsample is None:
            return x

        skip = x
        return self.downsample(x), skip

import torch
import torch.nn.functional as F
import torch.nn as nn


def mse_loss_mae(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    # Only compute loss over masked areas
    mask_output = ~torch.isnan(x_pred)
    mask_target = ~torch.isnan(x_true)
    mask_nan = mask_output & mask_target
    output_nonan = x_pred[mask_nan]
    target_nonan = x_true[mask_nan]
    loss = F.mse_loss(output_nonan, target_nonan)
    return loss


def custom_mse_loss(output, target):
    mask_output = ~torch.isnan(output)
    mask_target = ~torch.isnan(target)
    mask = mask_output & mask_target
    output_nonan = output[mask]
    target_nonan = target[mask]
    loss = F.mse_loss(output_nonan, target_nonan)
    # print(loss)
    return loss


class DWALoss(nn.Module):
    def __init__(self, num_tasks=4, temp=2.0, smoothing_factor=0.9):
        super().__init__()
        self.num_tasks = num_tasks
        self.temp = temp
        self.smoothing_factor = smoothing_factor
        self.register_buffer("loss_weights", torch.ones(num_tasks))
        self.register_buffer("prev_losses", torch.ones(num_tasks))

    def forward(self, losses):
        assert (
            len(losses) == self.num_tasks
        ), f"Expected {self.num_tasks} losses, but got {len(losses)}"
        losses = torch.stack(losses)
        weighted_losses = self.loss_weights * losses
        return weighted_losses.sum(), losses

    def update_weights(self, current_losses):
        smoothed_losses = (
            self.smoothing_factor * self.prev_losses
            + (1 - self.smoothing_factor) * current_losses
        )
        w = self.prev_losses / (smoothed_losses + 1e-8)
        w = torch.softmax(w / self.temp, dim=0) * self.num_tasks
        self.loss_weights = w
        self.prev_losses = smoothed_losses.detach()

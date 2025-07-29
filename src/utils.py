import torch
import torch.nn.functional as F


def mse_loss_mae(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    # Only compute loss over masked areas
    mask_output = ~torch.isnan(x_pred)
    mask_target = ~torch.isnan(x_true)
    mask_nan = mask_output & mask_target
    output_nonan = x_pred[mask_nan]
    target_nonan = x_true[mask_nan]
    loss = F.mse_loss(output_nonan, target_nonan)
    return loss

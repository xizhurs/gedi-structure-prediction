import pytest
import torch

from gedi_structure_prediction.utils import DWALoss, custom_mse_loss, mse_loss_mae


@pytest.mark.parametrize("loss_fn", [custom_mse_loss, mse_loss_mae])
def test_loss_and_gradients(loss_fn):
    prediction = torch.tensor([2.0, 4.0, 10.0], requires_grad=True)
    loss = loss_fn(prediction, torch.tensor([1.0, 2.0, float("nan")]))
    assert loss.item() == 2.5
    loss.backward()
    assert prediction.grad.tolist() == [1.0, 2.0, 0.0]


@pytest.mark.parametrize(
    "prediction,target",
    [([1.0], [float("nan")]), ([float("nan")], [1.0]), ([1.0, 2.0], [1.0])],
)
def test_invalid_loss_inputs(prediction, target):
    with pytest.raises(ValueError):
        custom_mse_loss(torch.tensor(prediction), torch.tensor(target))


def test_dynamic_weights():
    loss = DWALoss(num_tasks=3)
    inputs = [torch.tensor(v, requires_grad=True) for v in (1.0, 2.0, 3.0)]
    total, values = loss(inputs)
    assert total.item() == 6
    total.backward()
    assert all(item.grad.item() == 1 for item in inputs)
    loss.update_weights(values)
    assert loss.loss_weights.sum().item() == pytest.approx(3)
    assert torch.isfinite(loss.loss_weights).all()
    assert not loss.prev_losses.requires_grad

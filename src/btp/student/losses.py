"""Training SSIM v1: RGB, range 1, 11x11 Gaussian sigma 1.5, replicate border.

Mean reduction over batch/channels/pixels; distinct from evaluation SSIM.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from btp.student.model import validate_rgb


def ssim(prediction: Tensor, target: Tensor) -> Tensor:
    """Differentiable fixed-window structural similarity, accepting unclipped output."""
    validate_rgb(prediction, bounded=False)
    validate_rgb(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    coordinate = torch.arange(11, dtype=prediction.dtype, device=prediction.device) - 5
    kernel = torch.exp(-coordinate.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11)

    def average(x: Tensor) -> Tensor:
        return F.conv2d(F.pad(x, (5, 5, 5, 5), mode="replicate"), window, groups=3)

    mu_x, mu_y = average(prediction), average(target)
    var_x = (average(prediction.square()) - mu_x.square()).clamp_min(0)
    var_y = (average(target.square()) - mu_y.square()).clamp_min(0)
    covariance = average(prediction * target) - mu_x * mu_y
    return (
        ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2))
        / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    ).mean()


def restoration_loss(
    prediction: Tensor, clean: Tensor, teacher: Tensor | None = None
) -> Tensor:
    """A0 Charbonnier + SSIM; optional A1 detached teacher L1 with fixed 0.1 weight."""
    similarity = ssim(prediction, clean)
    loss = ((prediction - clean).square() + 1e-6).sqrt().mean() + 0.1 * (1 - similarity)
    if teacher is not None:
        validate_rgb(teacher)
        if teacher.shape != prediction.shape:
            raise ValueError("teacher shape differs from prediction")
        loss = loss + 0.1 * (prediction - teacher.detach()).abs().mean()
    return loss

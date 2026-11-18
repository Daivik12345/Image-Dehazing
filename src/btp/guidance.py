"""Opt-in RGB guidance adapters; no sampler patch or quality claim.

Upstream WeightedSSIMGuidance already multiplies its SSIM gradient by detached
transmission/density. Insert ``weight_rgb_correction(delta_pred, weight)`` AFTER
that existing correction and BEFORE ``pred.backward(delta_pred)`` in the RGB
branch of SpacedSampler.apply_cond_fn. Keep the decode graph, timestep window,
repeat count, grad_rescale and sampler schedule unchanged. These adapters do
not apply to the latent branch, and never resize confidence into latent space.

Compute confidence only from an inference-available RGB input in [0,1], with
exactly the decoded prediction's spatial geometry. Upstream guidance prediction
and target values are [-1,1]; do not pass those directly to confidence_maps.
There is no clean-target argument. epsilon_var and any norm cap must be
registered using training/development calibration before scoring candidates.
Torch is optional and imported only when an adapter is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class ConfidenceMaps:
    """Detached N1HW maps and scalar diagnostic summaries (not uncertainty truth)."""

    variance: Tensor
    weight: Tensor
    ambiguity: Tensor
    statistics: dict[str, float]


def _validate_rgb(image: Tensor, *, unit_range: bool) -> None:
    import torch

    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise ValueError("RGB input must be a floating-point tensor")
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1:
        raise ValueError("RGB input must be nonempty N3HW")
    if not torch.isfinite(image).all():
        raise ValueError("RGB input must be finite")
    if unit_range and ((image < 0).any() or (image > 1).any()):
        raise ValueError("confidence RGB input must be in [0,1]")


def _validate_weight(weight: Tensor) -> None:
    import torch

    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point tensor")
    if weight.ndim != 4 or weight.shape[1] != 1 or min(weight.shape) < 1:
        raise ValueError("weight must be nonempty N1HW")
    if not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
        raise ValueError("weight must be finite and in [0,1]")


def confidence_maps(
    image: Tensor,
    *,
    epsilon_var: float,
    floor: float = 0.25,
    window: int = 15,
) -> ConfidenceMaps:
    """Compute local luminance variance reliability and separate ambiguity map.

    Replicate padding supports tiny images. Population variance uses float64
    intermediate moments to reduce cancellation and is clamped nonnegative.
    Weight is floor+(1-floor)*variance/(variance+epsilon_var). Bright low-
    saturation ambiguity is max(RGB)*(1-HSV saturation), diagnostic only.
    Low contrast is an imperfect reliability proxy, especially in dense haze.
    """
    import torch
    from torch.nn import functional as F

    _validate_rgb(image, unit_range=True)
    if not math.isfinite(epsilon_var) or epsilon_var <= 0:
        raise ValueError("epsilon_var must be finite and positive")
    if not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("floor must be in [0,1]")
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window < 1
        or window % 2 == 0
    ):
        raise ValueError("window must be a positive odd integer")
    with torch.no_grad():
        value = image.detach().to(dtype=torch.float64)
        coefficients = value.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        luminance = (value * coefficients).sum(dim=1, keepdim=True)
        padding = window // 2
        padded = F.pad(luminance, (padding,) * 4, mode="replicate")
        mean = F.avg_pool2d(padded, window, stride=1)
        variance = (
            F.avg_pool2d(padded.square(), window, stride=1) - mean.square()
        ).clamp_min(0)
        weight = floor + (1 - floor) * variance / (variance + epsilon_var)
        maximum = value.amax(dim=1, keepdim=True)
        minimum = value.amin(dim=1, keepdim=True)
        saturation = (maximum - minimum) / maximum.clamp_min(
            torch.finfo(value.dtype).tiny
        )
        ambiguity = maximum * (1 - saturation)
        variance, weight, ambiguity = (
            x.to(image.dtype) for x in (variance, weight, ambiguity)
        )
        statistics = {
            "variance_mean": variance.mean().item(),
            "weight_min": weight.min().item(),
            "weight_mean": weight.mean().item(),
            "weight_max": weight.max().item(),
            "ambiguity_mean": ambiguity.mean().item(),
        }
    return ConfidenceMaps(variance, weight, ambiguity, statistics)


def uniform_control(weight: Tensor) -> Tensor:
    """Return detached per-image uniform weights matching each spatial mean."""
    _validate_weight(weight)
    return weight.detach().mean(dim=(2, 3), keepdim=True).expand_as(weight)


def weight_rgb_correction(
    correction: Tensor,
    weight: Tensor,
    *,
    max_l2_norm: float | None = None,
) -> Tensor:
    """Weight existing RGB correction and optionally cap each image's L2 norm.

    Weight must be N1HW on the correction's device, with matching batch/spatial
    dimensions. No implicit resize or dtype/device transfer is performed.
    Detached weights preserve the existing RGB-to-latent backward route. The
    optional cap scales the entire weighted N3HW correction per image by one
    nonnegative scalar, preserving direction. It is not a per-pixel cap.
    """
    import torch

    _validate_rgb(correction, unit_range=False)
    _validate_weight(weight)
    if (
        weight.shape != (correction.shape[0], 1, *correction.shape[2:])
        or weight.device != correction.device
        or weight.dtype != correction.dtype
    ):
        raise ValueError(
            "weight must match correction batch, geometry, device and dtype"
        )
    if max_l2_norm is not None and (not math.isfinite(max_l2_norm) or max_l2_norm < 0):
        raise ValueError("max_l2_norm must be finite and nonnegative")
    result = correction * weight.detach()
    if max_l2_norm is not None:
        norm = torch.linalg.vector_norm(result.detach().double().flatten(1), dim=1)
        scale = (max_l2_norm / norm.clamp_min(torch.finfo(norm.dtype).tiny)).clamp(
            max=1
        )
        result = result * scale.to(result.dtype).view(-1, 1, 1, 1)
    return result


class RGBGuidanceAdapter:
    """Opt-in wrapper compatible with the upstream sampler's ``cond_fn`` API.

    Instantiate a fresh adapter for each image after computing its registered
    confidence map, then pass it as cond_fn to the unmodified sampler. Target,
    density loaders, space, timestep window and repeat count delegate to the
    existing guidance object. Its inference-derived target is not a clean
    evaluation target. This wrapper weights the existing density-weighted RGB
    correction; the upstream sampler still performs its original VAE backward.
    Geometry/dtype mismatches fail rather than resize. Reusing an adapter with
    another image's map is unsupported; construct a new per-image adapter.
    """

    def __init__(
        self,
        baseline: object,
        weight: Tensor,
        *,
        max_l2_norm: float | None = None,
    ) -> None:
        if getattr(baseline, "space", None) != "rgb" or not callable(baseline):
            raise ValueError("baseline must be callable RGB guidance")
        _validate_weight(weight)
        if max_l2_norm is not None and (
            not math.isfinite(max_l2_norm) or max_l2_norm < 0
        ):
            raise ValueError("max_l2_norm must be finite and nonnegative")
        self._baseline = baseline
        self._weight = weight.detach().clone()
        self._max_l2_norm = max_l2_norm

    def __getattr__(self, name: str) -> object:
        return getattr(self._baseline, name)

    def __call__(
        self,
        target: Tensor,
        prediction: Tensor,
        timestep: int,
    ) -> tuple[Tensor, float]:
        """Preserve baseline loss reporting while transforming its RGB update."""
        if getattr(self._baseline, "space", None) != "rgb":
            raise ValueError("baseline must remain RGB guidance")
        correction, loss = self._baseline(target, prediction, timestep)
        return weight_rgb_correction(
            correction, self._weight, max_l2_norm=self._max_l2_norm
        ), loss

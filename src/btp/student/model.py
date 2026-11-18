"""Original NAFNet-style compact restoration implementation, without pretrained weights.

Uses channel normalization, multiplicative gates and simplified channel attention.
Architecture inspiration: Chen et al., Simple Baselines for Image Restoration (2022).
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StudentConfig:
    """Four-level architecture; defaults are the initial research candidate."""

    width: int = 16
    encoder_blocks: tuple[int, ...] = (1, 1, 1, 2)
    middle_blocks: int = 2
    decoder_blocks: tuple[int, ...] = (1, 1, 1, 1)

    def __post_init__(self) -> None:
        counts = (*self.encoder_blocks, *self.decoder_blocks, self.middle_blocks)
        if (
            len(self.encoder_blocks) != 4
            or len(self.decoder_blocks) != 4
            or type(self.width) is not int
            or self.width < 1
            or any(type(n) is not int or n < 1 for n in counts)
        ):
            raise ValueError(
                "architecture requires positive widths/blocks and four levels"
            )

    def to_dict(self) -> dict:
        """Return canonical architecture metadata."""
        result = asdict(self)
        result["encoder_blocks"] = list(self.encoder_blocks)
        result["decoder_blocks"] = list(self.decoder_blocks)
        return result


class ChannelNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, correction=0)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class RestorationBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1, groups=channels * 2
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1)
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = ChannelNorm(channels)
        self.ffn_in = nn.Conv2d(channels, channels * 2, 1)
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.depthwise(self.expand(self.norm1(x))).chunk(2, dim=1)
        y = a * b
        x = x + self.beta * self.project(y * self.attention(y))
        a, b = self.ffn_in(self.norm2(x)).chunk(2, dim=1)
        return x + self.gamma * self.ffn_out(a * b)


def validate_rgb(x: Tensor, *, bounded: bool = True) -> None:
    """Reject malformed tensors instead of silently changing geometry or range."""
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("expected nonempty floating NCHW RGB tensor")
    if not torch.isfinite(x).all() or (bounded and (x.min() < 0 or x.max() > 1)):
        raise ValueError("RGB tensor must be finite and in [0,1]")


class CompactDehaze(nn.Module):
    """Residual RGB encoder/decoder with replicate padding and exact output crop."""

    def __init__(self, config: StudentConfig | None = None):
        super().__init__()
        config = config or StudentConfig()
        self.config = config
        width = config.width
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for count in config.encoder_blocks:
            self.encoders.append(
                nn.Sequential(*(RestorationBlock(width) for _ in range(count)))
            )
            self.downs.append(nn.Conv2d(width, width * 2, 2, stride=2))
            width *= 2
        self.middle = nn.Sequential(
            *(RestorationBlock(width) for _ in range(config.middle_blocks))
        )
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in config.decoder_blocks:
            self.ups.append(
                nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.PixelShuffle(2))
            )
            width //= 2
            self.decoders.append(
                nn.Sequential(*(RestorationBlock(width) for _ in range(count)))
            )
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Return unclipped RGB values; inference clips explicitly."""
        validate_rgb(image)
        height, width = image.shape[-2:]
        padded = F.pad(image, (0, -width % 16, 0, -height % 16), mode="replicate")
        x = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, decoder, skip in zip(
            self.ups, self.decoders, reversed(skips), strict=True
        ):
            x = decoder(up(x) + skip)
        return (self.ending(x) + padded)[..., :height, :width]

    def parameter_count(self) -> int:
        """Count actual deployed parameters, including normalization and biases."""
        return sum(p.numel() for p in self.parameters())

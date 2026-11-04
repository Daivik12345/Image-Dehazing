"""CPU-only physical preview. These algorithms are NOT the paper's learned models."""

from collections.abc import Iterator

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, minimum_filter

from btp.config import RunConfig


def sample_photo() -> Image.Image:
    """Load the clean garden photograph bundled with scikit-learn, without network I/O."""
    from sklearn.datasets import load_sample_image

    return Image.fromarray(load_sample_image("china.jpg"))


def demo_image(size: int = 256) -> Image.Image:
    """Draw a deterministic clean scene without downloading a dataset."""
    if size < 16:
        raise ValueError("Demo size must be at least 16")
    y, x = np.mgrid[:size, :size] / size
    pixels = np.stack([0.32 + 0.22 * y, 0.58 + 0.15 * y, 0.78 + 0.10 * y], axis=-1)
    image = Image.fromarray(np.uint8(pixels.clip(0, 1) * 255))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(0, size * 0.65), (size * 0.3, size * 0.25), (size * 0.6, size * 0.65)],
        fill="#758577",
    )
    draw.polygon(
        [(size * 0.3, size * 0.7), (size * 0.72, size * 0.32), (size, size * 0.6)],
        fill="#425e55",
    )
    draw.rectangle((0, size * 0.67, size, size), fill="#5c7950")
    draw.polygon(
        [
            (size * 0.43, size),
            (size * 0.53, size * 0.67),
            (size * 0.58, size * 0.67),
            (size * 0.8, size),
        ],
        fill="#c7b89b",
    )
    for px, py, scale in [(0.15, 0.75, 0.15), (0.82, 0.7, 0.11), (0.3, 0.9, 0.23)]:
        draw.rectangle(
            (size * px, size * py, size * (px + 0.025), size * (py + scale * 0.3)),
            fill="#564b3f",
        )
        draw.ellipse(
            (
                size * (px - scale * 0.25),
                size * (py - scale),
                size * (px + scale * 0.3),
                size * (py + 0.02),
            ),
            fill="#254b38",
        )
    return image


def rgb_array(image: Image.Image, size: int) -> np.ndarray:
    """Convert any PIL image to bounded RGB, preserving aspect ratio."""
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    if min(image.size) < 2:
        raise ValueError("Image dimensions must both be at least 2")
    return np.asarray(image, dtype=np.float32) / 255


def as_image(array: np.ndarray) -> Image.Image:
    """Convert floating point RGB in [0,1] to a display image."""
    return Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))


def estimate_transmission(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate atmospheric light and dark-channel transmission for a CPU baseline."""
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite H x W x 3 RGB image")
    dark = minimum_filter(image.min(axis=-1), size=15, mode="nearest")
    count = max(1, dark.size // 1000)
    indices = np.argpartition(dark.ravel(), -count)[-count:]
    candidates = image.reshape(-1, 3)[indices]
    air = np.maximum(candidates[np.argmax(candidates.sum(axis=1))], 0.05)
    normalized = image / air
    transmission = 1 - 0.95 * minimum_filter(normalized.min(axis=-1), size=15)
    return np.clip(gaussian_filter(transmission, 2), 0.12, 1), air


def preview_frames(
    image: Image.Image, config: RunConfig
) -> Iterator[tuple[str, int, int, Image.Image]]:
    """Yield genuine progressive physical synthesis and DCP restoration frames."""
    source = rgb_array(image, config.size)
    current = source
    if config.stage in {"both", "hazegen"}:
        rng = np.random.default_rng(config.seed)
        height, width = source.shape[:2]
        noise = gaussian_filter(rng.normal(size=(height, width)), max(1, width / 12))
        noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
        depth = 0.3 + 0.7 * (1 - np.linspace(0, 1, height)[:, None]) + 0.2 * noise
        for step in range(1, config.steps + 1):
            transmission = np.exp(-config.haze_density * depth * step / config.steps)
            current = source * transmission[..., None] + 0.92 * (
                1 - transmission[..., None]
            )
            yield "hazegen", step, config.steps, as_image(current)
    if config.stage in {"both", "diffdehaze"}:
        hazy = current.copy()
        transmission, air = estimate_transmission(hazy)
        restored = np.clip((hazy - air) / transmission[..., None] + air, 0, 1)
        for step in range(1, config.steps + 1):
            fraction = step / config.steps
            current = hazy * (1 - fraction) + restored * fraction
            yield "diffdehaze", step, config.steps, as_image(current)

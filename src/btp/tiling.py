"""Deterministic native RGB tiling with explicit CPU accumulation budgets."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Tile:
    """Native crop and bottom/right edge padding to a square model input."""

    x: int
    y: int
    width: int
    height: int
    tile_size: int

    @property
    def pad_right(self) -> int:
        """Number of repeated edge columns appended to the crop."""
        return self.tile_size - self.width

    @property
    def pad_bottom(self) -> int:
        """Number of repeated edge rows appended to the crop."""
        return self.tile_size - self.height


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def make_tiles(
    width: int, height: int, tile_size: int = 512, overlap: int = 128
) -> list[Tile]:
    """Cover every native pixel in row-major order, including terminal tiles."""
    for name, value in (("width", width), ("height", height), ("tile_size", tile_size)):
        _integer(value, name)
    _integer(overlap, "overlap", 0)
    if overlap >= tile_size:
        raise ValueError("overlap must be less than tile_size")

    def positions(size: int) -> list[int]:
        end = max(0, size - tile_size)
        return sorted(set([*range(0, end + 1, tile_size - overlap), end]))

    return [
        Tile(x, y, min(tile_size, width - x), min(tile_size, height - y), tile_size)
        for y in positions(height)
        for x in positions(width)
    ]


def _validate_tile(tile: Tile, width: int, height: int) -> None:
    for name in ("width", "height", "tile_size"):
        _integer(getattr(tile, name), name)
    _integer(tile.x, "x", 0)
    _integer(tile.y, "y", 0)
    if (
        tile.width > tile.tile_size
        or tile.height > tile.tile_size
        or tile.x + tile.width > width
        or tile.y + tile.height > height
    ):
        raise ValueError("tile exceeds output bounds or tile_size")


def extract_tile(image: NDArray, tile: Tile) -> NDArray:
    """Extract an HWC RGB crop and edge-pad tiny dimensions explicitly."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HWC RGB")
    _validate_tile(tile, image.shape[1], image.shape[0])
    crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
    return np.pad(
        crop, ((0, tile.pad_bottom), (0, tile.pad_right), (0, 0)), mode="edge"
    )


def smooth_weights(tile_size: int) -> NDArray[np.float32]:
    """Return a smooth separable cosine window with a fixed positive floor."""
    _integer(tile_size, "tile_size")
    axis = (
        np.sin(np.pi * (np.arange(tile_size, dtype=np.float64) + 0.5) / tile_size) ** 2
    )
    # Symmetrize to remove roundoff differences at opposite borders.
    axis = (axis + axis[::-1]) / 2
    return np.maximum(np.outer(axis, axis), 1e-3).astype(np.float32)


def tile_seed(base_seed: int, source_scene_id: str, tile: Tile) -> int:
    """Derive a stable unsigned 32-bit seed, independent of Python hash state."""
    _integer(base_seed, "base_seed", 0)
    if not isinstance(source_scene_id, str) or not source_scene_id:
        raise ValueError("source_scene_id must be a nonempty string")
    value = f"{base_seed}:{len(source_scene_id)}:{source_scene_id}:{tile.x}:{tile.y}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


def fuse_tiles(
    tiles: Iterable[Tile],
    predictions: Iterable[NDArray],
    output_size: tuple[int, int],
    *,
    max_host_bytes: int = 512 * 1024 * 1024,
) -> NDArray[np.float32]:
    """Stream float RGB predictions into CPU float32 sums and normalize in place.

    ``output_size`` is (width, height). The declared budget covers persistent
    accumulation (16 bytes/pixel) and conservatively 64 bytes/tile pixel for
    conversion, validation, weights and weighted products. Caller-owned inputs,
    tile metadata, model memory and allocator overhead are outside this budget.
    Predictions are consumed once; no prediction collection is retained. A
    budget violation raises MemoryError without resizing or changing precision.
    """
    width, height = output_size
    _integer(width, "width")
    _integer(height, "height")
    _integer(max_host_bytes, "max_host_bytes")
    accumulation_bytes = width * height * 16
    if accumulation_bytes > max_host_bytes:
        raise MemoryError(f"host accumulation requires {accumulation_bytes} bytes")
    result = np.zeros((height, width, 3), dtype=np.float32)
    totals = np.zeros((height, width), dtype=np.float32)
    prediction_iter = iter(predictions)
    sentinel = object()
    for tile in tiles:
        _validate_tile(tile, width, height)
        required = accumulation_bytes + 64 * tile.tile_size**2
        if required > max_host_bytes:
            raise MemoryError(
                f"host accumulation and tile workspace require {required} bytes"
            )
        prediction = next(prediction_iter, sentinel)
        if prediction is sentinel:
            raise ValueError("missing tile prediction")
        array = np.asarray(prediction)
        if array.shape != (tile.tile_size, tile.tile_size, 3):
            raise ValueError("prediction must match padded HWC RGB tile shape")
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError("prediction must be finite RGB in [0, 1]")
        weights = smooth_weights(tile.tile_size)[: tile.height, : tile.width]
        ys = slice(tile.y, tile.y + tile.height)
        xs = slice(tile.x, tile.x + tile.width)
        result[ys, xs] += array[: tile.height, : tile.width] * weights[..., None]
        totals[ys, xs] += weights
        del prediction, array, weights
    if next(prediction_iter, sentinel) is not sentinel:
        raise ValueError("extra tile prediction")
    # Scalar reduction avoids an unbudgeted full-image boolean temporary.
    if np.min(totals) <= 0:
        raise ValueError("incomplete output coverage")
    np.divide(result, totals[..., None], out=result)
    return result

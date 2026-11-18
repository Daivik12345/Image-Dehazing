"""Deterministic perceptual-similarity candidates; never automatic split changes."""

from itertools import combinations

import numpy as np
from PIL import Image
from scipy.fft import dctn


def perceptual_hash(image: np.ndarray) -> str:
    """Return 63 AC low-frequency bits from a 32px grayscale DCT thumbnail."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB")
    gray = (
        Image.fromarray(array).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    )
    coefficients = dctn(np.asarray(gray, dtype=float), norm="ortho")[:8, :8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    return "".join("1" if bit else "0" for bit in bits)


def candidate_pairs(rows: list[dict], max_distance: int = 8) -> list[dict]:
    """List cross-split near hashes; similarity is not proof of duplicate content."""
    if type(max_distance) is not int or not 0 <= max_distance <= 63:
        raise ValueError("distance must be an integer in [0,63]")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate review IDs")
    for row in rows:
        if len(row["phash"]) != 63 or set(row["phash"]) - {"0", "1"}:
            raise ValueError("invalid perceptual hash")
    result = []
    for a, b in combinations(rows, 2):
        if a["split"] == b["split"]:
            continue
        distance = sum(x != y for x, y in zip(a["phash"], b["phash"], strict=True))
        if distance <= max_distance:
            result.append(
                {
                    "left": a["id"],
                    "right": b["id"],
                    "distance": distance,
                    "left_split": a["split"],
                    "right_split": b["split"],
                }
            )
    return sorted(result, key=lambda row: (row["distance"], row["left"], row["right"]))

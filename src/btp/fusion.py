"""Experimental bounded additive appearance alignment; no quality claim."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class AlignmentResult:
    """Corrected RGB plus fraction of channel samples clipped at the bound."""

    image: NDArray[np.float32]
    clipped_fraction: float


def align_global_appearance(
    local: NDArray,
    global_guide: NDArray,
    *,
    sigma: float = 8.0,
    max_correction: float = 0.05,
) -> AlignmentResult:
    """Add bounded low-frequency RGB differences without rescaling local detail.

    Both inputs must be finite HWC RGB in [0,1] with identical geometry. The
    caller explicitly resamples/crops the coarse global guide to native tile
    coordinates before calling. Both images use the same spatial Gaussian
    filter (reflect boundary, truncate=4); channels are never mixed. The
    correction is capped per channel then the resulting RGB is clipped to
    [0,1]. Apply to each predicted tile before weighted blending. This function
    does not run a global model or imply that alignment improves restoration.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if not np.isfinite(max_correction) or not 0 <= max_correction <= 1:
        raise ValueError("max_correction must be finite and in [0,1]")
    arrays = []
    for value in (local, global_guide):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
            raise ValueError("inputs must be nonempty HWC RGB")
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError("inputs must be finite RGB in [0,1]")
        arrays.append(array)
    local_array, guide_array = arrays
    if local_array.shape != guide_array.shape:
        raise ValueError("local and global guide geometry must match")
    difference = gaussian_filter(
        guide_array, (sigma, sigma, 0), mode="reflect", truncate=4.0
    )
    difference -= gaussian_filter(
        local_array, (sigma, sigma, 0), mode="reflect", truncate=4.0
    )
    clipped_fraction = float(np.mean(np.abs(difference) > max_correction))
    np.clip(difference, -max_correction, max_correction, out=difference)
    difference += local_array
    np.clip(difference, 0, 1, out=difference)
    return AlignmentResult(difference, clipped_fraction)

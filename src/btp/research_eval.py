"""Native prediction and explicit seam diagnostics for intentional quality changes."""

from collections.abc import Callable

import numpy as np

from btp.fusion import align_global_appearance
from btp.tiling import extract_tile, fuse_tiles, make_tiles


def predict_tiled(
    image: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int = 512,
    overlap: int = 128,
    max_host_bytes: int = 512 * 1024**2,
    global_guide: np.ndarray | None = None,
    diagnostics: list[dict] | None = None,
) -> np.ndarray:
    """Stream model tiles; optional guide must already have explicit native geometry.

    The predictor owns model residency and synchronizes its device before returning
    CPU HWC RGB. The host budget covers fusion workspace, not input/guide/model RAM.
    No resolution or precision fallback occurs after a resource failure.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HWC RGB")
    if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
        raise ValueError("Expected finite RGB in [0,1]")
    if global_guide is not None and global_guide.shape != image.shape:
        raise ValueError("Global guide must match native input geometry")
    height, width = image.shape[:2]
    tiles = make_tiles(width, height, tile_size, overlap)

    def predictions():
        for tile in tiles:
            prediction = predict(extract_tile(image, tile))
            if global_guide is not None:
                alignment = align_global_appearance(
                    prediction, extract_tile(global_guide, tile)
                )
                prediction = alignment.image
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "x": tile.x,
                            "y": tile.y,
                            "clipped_fraction": alignment.clipped_fraction,
                        }
                    )
            yield prediction

    return fuse_tiles(
        tiles, predictions(), (width, height), max_host_bytes=max_host_bytes
    )


def seam_metrics(
    clean: np.ndarray,
    prediction: np.ndarray,
    *,
    tile_size: int = 512,
    overlap: int = 128,
    band: int = 8,
) -> dict[str, float | int | None]:
    """Report target-referenced error near internal tile edges and detail error.

    Bands include both starts and ends of internal tiles, including the shifted
    terminal tile. Gradient error covers the whole image: smoothing must not win
    solely by removing visible seams. These supplement full-image PSNR/SSIM.
    """
    clean, prediction = np.asarray(clean), np.asarray(prediction)
    if (
        clean.shape != prediction.shape
        or clean.ndim != 3
        or clean.shape[2] != 3
        or min(clean.shape[:2]) < 2
    ):
        raise ValueError("Matching RGB images of at least 2x2 required")
    if type(band) is not int or band < 1:
        raise ValueError("band must be a positive integer")
    for value in (clean, prediction):
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError("Expected finite RGB in [0,1]")
    h, w = clean.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for tile in make_tiles(w, h, tile_size, overlap):
        for x in (tile.x, tile.x + tile.width):
            if 0 < x < w:
                mask[:, max(0, x - band) : min(w, x + band)] = True
        for y in (tile.y, tile.y + tile.height):
            if 0 < y < h:
                mask[max(0, y - band) : min(h, y + band), :] = True
    clean = clean.astype(np.float64)
    prediction = prediction.astype(np.float64)
    error = np.abs(clean - prediction)
    gradient = [
        np.mean(np.abs(np.diff(clean, axis=a) - np.diff(prediction, axis=a)))
        for a in (0, 1)
    ]
    seam_gradient_sum = 0.0
    seam_gradient_count = 0
    for axis in (0, 1):
        edge_error = np.abs(np.diff(clean, axis=axis) - np.diff(prediction, axis=axis))
        edge_mask = (
            (mask[1:] | mask[:-1]) if axis == 0 else (mask[:, 1:] | mask[:, :-1])
        )
        seam_gradient_sum += float(edge_error[edge_mask].sum())
        seam_gradient_count += int(edge_mask.sum()) * 3
    return {
        "seam_gradient_error": seam_gradient_sum / seam_gradient_count
        if seam_gradient_count
        else None,
        "seam_pixels": int(mask.sum()),
        "seam_mae": float(error[mask].mean()) if mask.any() else None,
        "interior_mae": float(error[~mask].mean()) if (~mask).any() else None,
        "gradient_error": float(np.mean(gradient)),
    }

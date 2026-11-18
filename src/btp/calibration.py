"""Input-only confidence calibration frozen before guidance candidate scoring."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from btp.evaluation import confined_path, read_rgb, sha256
from btp.experiment import write_json


def local_variance(image: np.ndarray, window: int = 15) -> np.ndarray:
    """Population luminance variance with replicate boundaries, float64 moments."""
    if type(window) is not int or window < 1 or window % 2 == 0:
        raise ValueError("Require positive odd window")
    array = np.asarray(image, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or not np.isfinite(array).all()
        or array.min() < 0
        or array.max() > 1
    ):
        raise ValueError("Require finite HWC RGB [0,1]")
    luminance = array @ np.array([0.2126, 0.7152, 0.0722])
    mean = uniform_filter(luminance, size=window, mode="nearest")
    return np.maximum(
        uniform_filter(luminance**2, size=window, mode="nearest") - mean**2, 0
    )


def calibrate(manifest: Path, output: Path, *, count: int = 20, seed: int = 42) -> dict:
    """Median of per-scene positive-variance medians on fixed training inputs.

    epsilon floor1e-8, window15 and selection rule are fixed before any scoring.
    No clean target is read, and validation/evaluation scenes cannot enter fit.
    """
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or count < 1:
        raise ValueError("positive count required")
    document = json.loads(manifest.read_text())
    rows = [r for r in document["scenes"] if r["split"] == "train"]
    if len(rows) < count:
        raise ValueError("Insufficient training scenes")
    rows = sorted(
        rows,
        key=lambda r: hashlib.sha256(
            f"{seed}:{r['source_scene_id']}".encode()
        ).hexdigest(),
    )[:count]
    if len({r["source_scene_id"] for r in rows}) != count:
        raise ValueError("Require unique training source scenes")
    statistics = []
    for row in rows:
        image = confined_path(manifest.parent, row["hazy"])
        if sha256(image) != row["hazy_sha256"]:
            raise ValueError("Input hash mismatch")
        variance = local_variance(read_rgb(image))
        positive = variance[variance > 1e-12]
        value = float(np.median(positive)) if positive.size else 0.0
        statistics.append(
            {
                "scene_id": row["source_scene_id"],
                "hazy_sha256": row["hazy_sha256"],
                "positive_variance_median": value,
            }
        )
    result = {
        "schema_version": 1,
        "manifest_sha256": sha256(manifest),
        "seed": seed,
        "window": 15,
        "count": count,
        "fit_split": "train",
        "rule": "median of per-source positive (>1e-12) variance medians, floored1e-8",
        "epsilon_var": max(
            1e-8, float(np.median([r["positive_variance_median"] for r in statistics]))
        ),
        "sources": statistics,
        "scope": "input statistic calibration, not evidence of useful confidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result

"""Strict paired-image manifests and versioned restoration metrics."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

PROTOCOL = "btp-rgb-full-psnr-ssim-gaussian11-population-v1"


def sha256(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confined_path(root: Path, relative: str) -> Path:
    """Resolve only relative paths inside the dataset root, including symlinks."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Dataset path must be relative")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path outside root: {relative}")
    return result


def read_rgb(path: Path) -> np.ndarray:
    """Read an explicit RGB image as float64 in [0, 1], without resizing."""
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path}")
        return np.asarray(image, dtype=np.float64) / 255.0


def load_manifest(path: Path) -> dict[str, Any]:
    """Validate all declared pairs before any split selection or GPU work."""
    doc = json.loads(path.read_text())
    if (
        doc.get("schema_version") != 1
        or not doc.get("dataset")
        or not doc.get("release")
    ):
        raise ValueError("Invalid manifest schema/dataset/release")
    scenes = doc.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Manifest must contain scenes")
    ids: set[str] = set()
    split_for_source: dict[str, str] = {}
    for scene in scenes:
        name = scene.get("scene_id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError(f"Duplicate or invalid scene ID: {name}")
        ids.add(name)
        if scene.get("split") not in {"development", "validation", "test", "train"}:
            raise ValueError("Invalid split")
        if not scene.get("geometry_tier"):
            raise ValueError("Missing geometry tier")
        sizes = []
        for role in ("hazy", "clean"):
            image_path = confined_path(path.parent, scene.get(role))
            expected = scene.get(role + "_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Missing SHA256 hash")
            if not image_path.is_file() or sha256(image_path) != expected:
                raise ValueError(f"Missing file or hash mismatch: {image_path}")
            with Image.open(image_path) as image:
                if image.mode != "RGB":
                    raise ValueError("Expected RGB image")
                image.load()
                sizes.append(image.size)
        if sizes[0] != sizes[1]:
            raise ValueError(f"Pair dimensions differ: {name}")
        if min(sizes[0]) < 11:
            raise ValueError("Image too small for the fixed SSIM window")
        if "crop_xywh" in scene or "source_size" in scene:
            crop, source_size = scene.get("crop_xywh"), scene.get("source_size")
            if (
                not isinstance(crop, list)
                or len(crop) != 4
                or not isinstance(source_size, list)
                or len(source_size) != 2
                or any(type(v) is not int for v in crop + source_size)
            ):
                raise ValueError("Invalid crop geometry schema")
            x, y, width, height = crop
            sw, sh = source_size
            if (
                x < 0
                or y < 0
                or min(width, height, sw, sh) <= 0
                or x + width > sw
                or y + height > sh
                or (width, height) != sizes[0]
            ):
                raise ValueError("Invalid crop bounds or dimensions")
        source = scene.get("source_hazy_sha256", scene["hazy_sha256"])
        old_split = split_for_source.setdefault(source, scene["split"])
        if old_split != scene["split"]:
            raise ValueError("Source scene crosses splits")
    return doc


def paired_metrics(clean: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """RGB PSNR/SSIM: 11x11 Gaussian sigma 1.5, population covariance.

    Equivalent SSIM convention to skimage 0.24 with gaussian_weights=True,
    sigma=1.5, use_sample_covariance=False, channel_axis=-1, data_range=1.
    Exact-match PSNR is represented by null plus identical=true (strict JSON).
    """
    clean, prediction = (
        np.asarray(clean, dtype=np.float64),
        np.asarray(prediction, dtype=np.float64),
    )
    if (
        clean.shape != prediction.shape
        or clean.ndim != 3
        or clean.shape[-1] != 3
        or min(clean.shape[:2]) < 11
    ):
        raise ValueError("Expected matching RGB arrays with dimensions >= 11")
    for array in (clean, prediction):
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Metrics require finite RGB values in [0,1]")
    delta = prediction - clean
    mse = float(np.mean(delta * delta))
    scores = []
    for channel in range(3):
        x, y = clean[..., channel], prediction[..., channel]
        ux = gaussian_filter(x, 1.5, truncate=3.5, mode="reflect")
        uy = gaussian_filter(y, 1.5, truncate=3.5, mode="reflect")
        vx = gaussian_filter(x * x, 1.5, truncate=3.5, mode="reflect") - ux * ux
        vy = gaussian_filter(y * y, 1.5, truncate=3.5, mode="reflect") - uy * uy
        cov = gaussian_filter(x * y, 1.5, truncate=3.5, mode="reflect") - ux * uy
        score = ((2 * ux * uy + 0.01**2) * (2 * cov + 0.03**2)) / (
            (ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2)
        )
        scores.append(float(score[5:-5, 5:-5].mean()))
    return {
        "protocol": PROTOCOL,
        "psnr_db": None if mse == 0 else float(-10 * np.log10(mse)),
        "identical": mse == 0,
        "ssim": float(np.mean(scores)),
        "mae": float(np.abs(delta).mean()),
        "max_abs_error": float(np.abs(delta).max()),
    }


class LPIPSScorer:
    """Optional AlexNet LPIPS scorer; instantiate only outside timed inference.

    The caller explicitly opts in to any library-managed weight download.
    """

    def __init__(self) -> None:
        import lpips

        self.model = lpips.LPIPS(net="alex", version="0.1").cpu().eval()

    def provenance(self) -> dict[str, Any]:
        """Record package version and actual loaded metric tensors."""
        from importlib.metadata import version

        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return {
            "package": "lpips",
            "version": version("lpips"),
            "network": "alex",
            "metric_version": "0.1",
            "state_sha256": digest.hexdigest(),
            "device": "cpu",
            "input_range": [-1, 1],
        }

    def __call__(self, clean: np.ndarray, prediction: np.ndarray) -> float:
        import torch

        tensors = [
            torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            for x in (clean, prediction)
        ]
        with torch.no_grad():
            return float(self.model(*tensors).item())


def main() -> None:
    """Validate paired data or score a single output against its clean target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    if args.manifest:
        doc = load_manifest(args.manifest)
        print(
            json.dumps(
                {
                    "dataset": doc["dataset"],
                    "scenes": len(doc["scenes"]),
                    "manifest_sha256": sha256(args.manifest),
                }
            )
        )
    elif args.clean and args.prediction:
        print(
            json.dumps(
                paired_metrics(read_rgb(args.clean), read_rgb(args.prediction)),
                allow_nan=False,
            )
        )
    else:
        parser.error("Provide --manifest or --clean and --prediction")


if __name__ == "__main__":
    main()

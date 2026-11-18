"""Explicit checkpoint inference for full-frame or native tiled student deployment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from btp.evaluation import read_rgb, sha256
from btp.experiment import revision, write_json
from btp.research_eval import predict_tiled
from btp.student.inference import load_student
from btp.student.model import StudentConfig


def _run_student(
    image: Path,
    checkpoint: Path,
    architecture: Path,
    expected_sha256: str,
    output: Path,
    *,
    device: str = "cpu",
    tiled: bool = False,
    tile_size: int = 512,
    overlap: int = 128,
    max_host_bytes: int = 512 * 1024**2,
) -> dict:
    """Load once and preserve native dimensions; never substitute a random model."""
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    config = StudentConfig(**json.loads(architecture.read_text()))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no implicit fallback")

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    predictor = load_student(checkpoint, config, device, sha256=expected_sha256)
    synchronize()
    load_seconds = time.perf_counter() - started
    array = read_rgb(image).astype(np.float32)
    output.mkdir(parents=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def predict(rgb):
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).unsqueeze(0)
        result = predictor(tensor)[0].detach().cpu().numpy().transpose(1, 2, 0)
        return result

    synchronize()
    started = time.perf_counter()
    result = (
        predict_tiled(
            array,
            predict,
            tile_size=tile_size,
            overlap=overlap,
            max_host_bytes=max_host_bytes,
        )
        if tiled
        else predict(array)
    )
    synchronize()
    model_seconds = time.perf_counter() - started
    destination = output / "prediction.png"
    Image.fromarray(np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)).save(
        destination
    )
    record = {
        "status": "completed",
        "model": "CompactDehaze",
        "architecture": config.to_dict(),
        "parameters": predictor.model.parameter_count(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": expected_sha256,
        "input_sha256": sha256(image),
        "output_sha256": sha256(destination),
        "source_commit": revision(Path.cwd()),
        "torch": torch.__version__,
        "device": device,
        "precision": "float32",
        "geometry": "native",
        "input_size": list(array.shape[1::-1]),
        "output_size": list(result.shape[1::-1]),
        "tiled": tiled,
        "tile_size": tile_size if tiled else None,
        "overlap": overlap if tiled else None,
        "load_seconds": load_seconds,
        "model_seconds": model_seconds,
        "timing_scope": "prediction includes tensor transfers and tile fusion; excludes file IO and load; no warmup",
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(device)
        if device.startswith("cuda")
        else None,
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(device)
        if device.startswith("cuda")
        else None,
    }
    write_json(output / "record.json", record)
    return record


def run_student(
    image: Path,
    checkpoint: Path,
    architecture: Path,
    expected_sha256: str,
    output: Path,
    *,
    device: str = "cpu",
    tiled: bool = False,
    tile_size: int = 512,
    overlap: int = 128,
    max_host_bytes: int = 512 * 1024**2,
) -> dict:
    """Preserve structured failure evidence once an inference output is reserved."""
    existed = output.exists()
    try:
        return _run_student(
            image,
            checkpoint,
            architecture,
            expected_sha256,
            output,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            overlap=overlap,
            max_host_bytes=max_host_bytes,
        )
    except Exception as error:
        if not existed and output.is_dir():
            write_json(
                output / "record.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "checkpoint_sha256": expected_sha256,
                    "device": device,
                    "tiled": tiled,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "max_host_bytes": max_host_bytes,
                    "input_sha256": sha256(image),
                    "source_commit": revision(Path.cwd()),
                    "fallback": None,
                },
            )
        raise


def main() -> None:
    """Run an explicitly identified trained or clearly labeled fixture artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("image", "checkpoint", "architecture", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-host-bytes", type=int, default=512 * 1024**2)
    args = parser.parse_args()
    record = run_student(
        args.image,
        args.checkpoint,
        args.architecture,
        args.sha256,
        args.output,
        device=args.device,
        tiled=args.tiled,
        tile_size=args.tile_size,
        overlap=args.overlap,
        max_host_bytes=args.max_host_bytes,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

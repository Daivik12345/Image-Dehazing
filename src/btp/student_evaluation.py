"""Fixed-development student full/tiled/global experiments; no automatic promotion."""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from btp.evaluation import (
    confined_path,
    load_manifest,
    paired_metrics,
    read_rgb,
    sha256,
)
from btp.experiment import write_json
from btp.research_eval import predict_tiled, seam_metrics


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                Image.fromarray(image[:, :, c].astype(np.float32)).resize(
                    size, Image.Resampling.BICUBIC
                )
            )
            for c in range(3)
        ],
        axis=-1,
    ).clip(0, 1)


def predict_variant(
    image: np.ndarray,
    predict: Callable,
    variant: str,
    *,
    tile_size: int = 512,
    overlap: int = 128,
    max_host_bytes: int = 1024**3,
) -> tuple[np.ndarray, dict]:
    """Global mode includes aspect-preserving coarse prediction and alignment cost."""
    if variant == "full":
        return predict(image), {"variant": variant}
    if variant not in {"tiled", "global"}:
        raise ValueError("Unknown variant")
    details = {"variant": variant, "clipping": []}
    guide = None
    if variant == "global":
        h, w = image.shape[:2]
        scale = min(1.0, tile_size / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        guide = _resize(predict(_resize(image, size)), (w, h))
        details.update(
            global_size=list(size),
            global_resize="float32 bicubic, clipped [0,1], aspect ratio rounded to pixels",
        )
    return predict_tiled(
        image,
        predict,
        tile_size=tile_size,
        overlap=overlap,
        max_host_bytes=max_host_bytes,
        global_guide=guide,
        diagnostics=details["clipping"],
    ), details


def evaluate(
    manifest: Path,
    checkpoint: Path,
    architecture: Path,
    expected_sha256: str,
    output: Path,
    *,
    scenes: list[str],
    variant: str = "full",
    device: str = "cuda:0",
    repeats: int = 5,
    max_seconds: float = 1200,
    source_commit: str,
) -> dict:
    """Evaluate an explicit development panel; one warmup excluded per scene.

    Repeat timing includes transfers, global pass and CPU fusion, excludes fileIO.
    Failures remain in planned coverage. LPIPS is unavailable, not zero. No heldout
    path is provided: release evaluation requires a separately frozen final stage.
    """
    if (
        not scenes
        or len(set(scenes)) != len(scenes)
        or repeats != 5
        or not 0 < max_seconds <= 3600
    ):
        raise ValueError(
            "Require unique scenes, five timing repeats and bounded budget"
        )
    doc = load_manifest(manifest)
    rows = {r["scene_id"]: r for r in doc["scenes"]}
    if any(s not in rows or rows[s]["split"] != "development" for s in scenes):
        raise ValueError("Only explicit development scenes permitted")
    if not source_commit:
        raise ValueError("Record exact source commit")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "loading",
        "variant": variant,
        "planned_scenes": scenes,
        "records": [],
        "checkpoint_sha256": expected_sha256,
        "manifest_sha256": sha256(manifest),
        "source_commit": source_commit,
        "max_seconds": max_seconds,
        "lpips": None,
        "scope": "development only; no automatic promotion",
    }
    write_json(output / "report.json", report)
    import torch

    from btp.student.inference import load_student
    from btp.student.model import StudentConfig

    torch.manual_seed(42)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    started = time.monotonic()
    sync()
    load_start = time.perf_counter()
    predictor = load_student(
        checkpoint,
        StudentConfig(**json.loads(architecture.read_text())),
        device,
        sha256=expected_sha256,
    )
    sync()
    report["load_seconds"] = time.perf_counter() - load_start
    report["parameters"] = predictor.model.parameter_count()
    report["checkpoint_bytes"] = checkpoint.stat().st_size
    report["runtime"] = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else None,
    }

    def predict(array):
        tensor = torch.from_numpy(
            np.ascontiguousarray(array.transpose(2, 0, 1))
        ).unsqueeze(0)
        return predictor(tensor)[0].detach().cpu().numpy().transpose(1, 2, 0)

    halted = False
    for scene in scenes:
        row = rows[scene]
        record = {"scene_id": scene, "status": "running"}
        try:
            if halted:
                raise RuntimeError("Previous resource failure; restart explicitly")
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Experiment budget exhausted; scene unexecuted")
            image = read_rgb(confined_path(manifest.parent, row["hazy"])).astype(
                np.float32
            )
            predict_variant(image, predict, variant)
            sync()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _repeat in range(repeats):
                if time.monotonic() - started >= max_seconds:
                    raise TimeoutError("Repeat budget exhausted")
                sync()
                clock = time.perf_counter()
                prediction, details = predict_variant(image, predict, variant)
                sync()
                times.append(time.perf_counter() - clock)
            if prediction.shape != image.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid native output")
            path = output / (scene + ".png")
            Image.fromarray(np.rint(prediction.clip(0, 1) * 255).astype(np.uint8)).save(
                path
            )
            target = read_rgb(confined_path(manifest.parent, row["clean"]))
            saved = read_rgb(path)
            record.update(
                status="completed",
                times_seconds=times,
                median_seconds=float(np.median(times)),
                metrics=paired_metrics(target, saved),
                seams=seam_metrics(target, saved),
                details=details,
                input_sha256=row["hazy_sha256"],
                clean_sha256=row["clean_sha256"],
                prediction_sha256=sha256(path),
                size=list(image.shape[1::-1]),
                allocated_peak_bytes=torch.cuda.max_memory_allocated(device)
                if device.startswith("cuda")
                else None,
                reserved_peak_bytes=torch.cuda.max_memory_reserved(device)
                if device.startswith("cuda")
                else None,
            )
        except Exception as error:
            if (
                isinstance(error, (MemoryError, TimeoutError))
                or "out of memory" in str(error).lower()
            ):
                halted = True
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["records"].append(record)
        write_json(output / "report.json", report)
    report["status"] = (
        "completed"
        if all(r["status"] == "completed" for r in report["records"])
        else "incomplete"
    )
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "report.json", report)
    return report


def main() -> None:
    """Run one named student variant against a declared development manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "architecture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--variant", choices=["full", "tiled", "global"], default="full")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seconds", type=float, default=1200)
    print(json.dumps(evaluate(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()

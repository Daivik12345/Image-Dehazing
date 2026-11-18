"""Fresh cold-repeatability and A-B-A-B resident gate before offline targets."""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from btp.config import UPSTREAM
from btp.evaluation import (
    confined_path,
    load_manifest,
    paired_metrics,
    read_rgb,
    sha256,
)
from btp.experiment import write_json


def compare_predictions(cold: list[np.ndarray], resident: list[np.ndarray]) -> dict:
    """Apply unchanged observed-variation MAE/max-pixel criteria, no quality gate."""
    if len(cold) < 3 or not resident:
        raise ValueError("Require three cold repeats and resident predictions")
    variation = [paired_metrics(a, b) for a, b in itertools.combinations(cold, 2)]
    limits = {
        key: max(row[key] for row in variation) for key in ("mae", "max_abs_error")
    }
    comparisons = [paired_metrics(a, b) for a in cold for b in resident]
    return {
        "limits": limits,
        "eligible": all(
            row[key] <= limits[key] + 1e-12 for row in comparisons for key in limits
        ),
        "comparisons": [{key: row[key] for key in limits} for row in comparisons],
    }


def run_validation(
    manifest: Path,
    weights: Path,
    output: Path,
    scenes: tuple[str, str] = ("nh-haze-08", "nh-haze-13"),
    max_seconds: float = 2400,
) -> dict:
    """Run six cold and four resident predictions sequentially, retaining failures.

    Cold and resident use the same decoded RGB inputs, M0/guidance, seed42 and
    unmodified upstream schedule. Only a complete matching-runtime gate may pass.
    """
    if len(set(scenes)) != 2 or not 0 < max_seconds <= 2400:
        raise ValueError("Require two distinct scenes and budget <=2400 seconds")
    doc = load_manifest(manifest)
    rows = {row["scene_id"]: row for row in doc["scenes"]}
    if any(s not in rows or rows[s]["split"] != "development" for s in scenes):
        raise ValueError("Only declared development scenes permitted")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "eligible": False,
        "manifest_sha256": sha256(manifest),
        "scenes": list(scenes),
        "cold": [],
        "resident": [],
        "seed": 42,
        "max_seconds": max_seconds,
        "planned_jobs": 10,
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "decision.json", report)

    def remaining():
        seconds = max_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError("Resident validation budget exhausted")
        return seconds

    save()
    try:
        cold_images = {s: [] for s in scenes}
        runtime = None
        for scene in scenes:
            image = confined_path(manifest.parent, rows[scene]["hazy"])
            for repeat in range(3):
                directory = output / f"cold-{scene}-{repeat}"
                command = [
                    sys.executable,
                    "-m",
                    "btp.reference",
                    "--input",
                    str(image),
                    "--weights",
                    str(weights),
                    "--output",
                    str(directory),
                    "--mode",
                    "accsamp",
                    "--seed",
                    "42",
                    "--dataset-version",
                    f"{doc['dataset']}:{doc['release']}",
                    "--preview-every",
                    "0",
                    "--memory-policy",
                    "M0",
                    "--cache-policy",
                    "guidance",
                ]
                with (output / f"cold-{scene}-{repeat}.log").open("w") as log:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=remaining(),
                    )
                metadata = json.loads((directory / "manifest.json").read_text())
                keys = (
                    "torch",
                    "cuda",
                    "cudnn",
                    "tf32_matmul",
                    "tf32_cudnn",
                    "deterministic_algorithms",
                    "cudnn_benchmark",
                )
                current = {k: metadata[k] for k in keys}
                if runtime is not None and runtime != current:
                    raise ValueError("Cold runtime changed")
                runtime = current
                prediction = directory / "final/input.png"
                cold_images[scene].append(read_rgb(prediction))
                report["cold"].append(
                    {
                        "scene_id": scene,
                        "repeat": repeat,
                        "prediction_sha256": sha256(prediction),
                        "checkpoint_sha256": metadata["checkpoint_sha256"],
                        "input_sha256": metadata["input_sha256"],
                    }
                )
                save()
        from btp.resident import ResidentConfig, ResidentTeacher

        resident_images = {s: [] for s in scenes}
        with ResidentTeacher(
            ResidentConfig(UPSTREAM / "configs/inference/stage2_accsamp.yaml"), weights
        ) as teacher:
            report["teacher_identity"] = teacher.identity
            report["teacher_runtime"] = teacher.runtime_identity
            report["deployment_parameters"] = sum(
                p.numel() for p in teacher.backend.model.parameters()
            )
            for index, scene in enumerate(scenes * 2):
                remaining()
                image = confined_path(manifest.parent, rows[scene]["hazy"])
                record = teacher.predict(
                    image, output / f"resident-{index}-{scene}", 42
                )
                report["resident"].append(record)
                save()
                if record["status"] != "completed":
                    raise RuntimeError(record["error"])
                if any(record["runtime"].get(k) != v for k, v in runtime.items()):
                    raise ValueError("Cold/resident runtime mismatch")
                baseline = next(r for r in report["cold"] if r["scene_id"] == scene)
                if (
                    record["checkpoint_sha256"] != baseline["checkpoint_sha256"]
                    or record["input_sha256"] != baseline["input_sha256"]
                ):
                    raise ValueError("Cold/resident checkpoint or input mismatch")
                resident_images[scene].append(read_rgb(Path(record["output_path"])))
        report["comparisons"] = {
            s: compare_predictions(cold_images[s], resident_images[s]) for s in scenes
        }
        report["eligible"] = all(v["eligible"] for v in report["comparisons"].values())
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        save()
    return report


def main() -> None:
    """Run the explicitly bounded resident validation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "weights", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=2400)
    args = parser.parse_args()
    print(
        json.dumps(
            run_validation(
                args.manifest, args.weights, args.output, max_seconds=args.max_seconds
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

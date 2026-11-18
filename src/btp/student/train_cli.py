"""Audited fixed-crop A0/A1 pilot; dry audit by default, no automatic downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from btp.evaluation import PROTOCOL, confined_path, paired_metrics, sha256
from btp.protocols import load_protocol
from btp.training_data import audit_training_manifest, paired_crop

_DEFAULTS = dict(
    max_updates=2000,
    max_seconds=7200,
    accumulation=4,
    warmup_updates=100,
    learning_rate=1e-4,
    minimum_lr=1e-6,
    weight_decay=1e-4,
    gradient_clip=1.0,
    seed=42,
    checkpoint_interval=250,
)


def run_pilot(
    manifest: Path,
    data_root: Path,
    evaluation_manifest: Path,
    protocol: Path,
    output: Path,
    *,
    source_commit: str,
    run: bool = False,
    config: dict[str, Any] | None = None,
    architecture: dict[str, Any] | None = None,
    ablation: str = "A0",
    crop_size: int = 256,
    max_host_bytes: int = 512 * 1024**2,
    until_update: int = 100,
    max_segment_seconds: float = 1200,
    device: str = "cpu",
    resume: Path | None = None,
) -> dict[str, Any]:
    """Audit all fixed splits, optionally run a bounded pilot and export evidence.

    Defaults stop at absolute update 100 or 1200 seconds this segment for profiling,
    counted within the total
    2000-update/7200-second cap. Continuation requires an explicit larger
    until_update and the prior last.pt plus its budget sidecar. Fixed seeded
    crops, one per pair, have no flips or resampling; this is a pilot policy,
    not the full augmentation campaign. All rows must fit; none are dropped.

    max_host_bytes bounds estimated corpus storage including trainer clones and
    decode workspace; it excludes model/optimizer/activation/runtime memory.
    No Torch import or model construction occurs in audit-only mode.
    """
    if not math.isfinite(max_segment_seconds) or max_segment_seconds <= 0:
        raise ValueError("max_segment_seconds must be finite and positive")
    settings = dict(_DEFAULTS, **(config or {}))
    if set(settings) != set(_DEFAULTS):
        raise ValueError("Unknown training configuration fields")
    for key in (
        "max_updates",
        "accumulation",
        "warmup_updates",
        "checkpoint_interval",
        "seed",
    ):
        if type(settings[key]) is not int or settings[key] < (
            0 if key == "seed" else 1
        ):
            raise ValueError(f"Invalid {key}")
    for key in ("max_seconds", "learning_rate", "minimum_lr", "gradient_clip"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if (
        settings["minimum_lr"] > settings["learning_rate"]
        or not math.isfinite(settings["weight_decay"])
        or settings["weight_decay"] < 0
    ):
        raise ValueError("Invalid learning rate bounds or weight decay")
    if settings["max_updates"] > 2000 or settings["max_seconds"] > 7200:
        raise ValueError("Pilot cap is 2000 updates and 7200 cumulative seconds")
    if ablation not in {"A0", "A1"} or not source_commit:
        raise ValueError("Require A0/A1 and an explicit source commit")
    if type(crop_size) is not int or crop_size < 11:
        raise ValueError("crop_size must be an integer >= 11 for validation SSIM")
    if type(max_host_bytes) is not int or max_host_bytes < 1:
        raise ValueError("max_host_bytes must be positive")
    if (
        type(until_update) is not int
        or not 1 <= until_update <= settings["max_updates"]
    ):
        # Default profiling endpoint naturally truncates tiny fixture budgets.
        if until_update != 100:
            raise ValueError("until_update must be inside the total budget")
        until_update = min(100, settings["max_updates"])
    if output.exists():
        raise FileExistsError("Choose a fresh pilot output directory")
    load_protocol(protocol, evaluation_manifest)
    evaluation = json.loads(evaluation_manifest.read_text())["scenes"]
    document = json.loads(manifest.read_text())
    rows = document.get("scenes", [])
    if not rows:
        raise ValueError("Empty corpus")
    # Read headers before the audit decodes images, enforcing the memory gate early.
    largest_pixels = 0
    roles_count = 0
    for row in rows:
        if ablation == "A1" and row.get("split") == "train" and not row.get("teacher"):
            raise ValueError("A1 requires a teacher target for every training pair")
        roles = [role for role in ("hazy", "clean", "teacher") if role in row]
        roles_count += len(roles)
        for role in roles:
            with Image.open(confined_path(data_root, row[role])) as image:
                if min(image.size) < crop_size:
                    raise ValueError(
                        "Every pair must fit the fixed crop; no silent drops"
                    )
                largest_pixels = max(largest_pixels, image.width * image.height)
    estimated_bytes = 4 * roles_count * crop_size**2 * 3 * 4 + 64 * largest_pixels
    if estimated_bytes > max_host_bytes:
        raise MemoryError(
            f"Corpus host storage/workspace estimate {estimated_bytes} exceeds budget"
        )
    audit = audit_training_manifest(document, data_root, evaluation_scenes=evaluation)
    if not all(audit["counts"].get(split, 0) for split in ("train", "validation")):
        raise ValueError("Require fixed nonempty train and corpus-validation splits")
    record: dict[str, Any] = dict(
        status="audit_only",
        dataset_version=f"{document['dataset']}:{document['release']}",
        dataset_archive_sha256=document["archive_sha256"],
        protocol_sha256=sha256(protocol),
        audit=audit,
        ablation=ablation,
        manifest_sha256=sha256(manifest),
        evaluation_manifest_sha256=sha256(evaluation_manifest),
        source_commit=source_commit,
        config=settings,
        architecture=architecture or {},
        until_update=until_update,
        max_segment_seconds=max_segment_seconds,
        estimated_corpus_host_bytes=estimated_bytes,
        max_corpus_host_bytes=max_host_bytes,
        memory_scope="corpus tensors/clones and decode workspace; excludes model/runtime",
        crop_policy={
            "size": crop_size,
            "kind": "one fixed seeded paired crop per pair",
            "flips": False,
            "seed_rule": "SHA256(base_seed:pair_id) first 8 bytes big endian",
        },
        validation_protocol=PROTOCOL,
        device=device,
    )
    output.mkdir(parents=True)

    def persist() -> None:
        (output / "pilot.json").write_text(
            json.dumps(record, indent=2, allow_nan=False)
        )

    persist()
    if not run:
        return record
    try:
        import torch

        from btp.student.inference import export_student
        from btp.student.model import StudentConfig
        from btp.student.trainer import BoundedTrainer, TrainingConfig, TrainingPair

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA unavailable; no device fallback")
        is_cuda = torch.device(device).type == "cuda"
        record["runtime"] = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": str(torch.device(device)),
            "device_name": torch.cuda.get_device_name(device) if is_cuda else "CPU",
            "cudnn": torch.backends.cudnn.version() if is_cuda else None,
        }
        record.update(allocated_peak_bytes=None, reserved_peak_bytes=None)
        pairs, validation = [], []
        crop_seeds = {}
        for row in rows:
            roles = ["hazy", "clean"]
            if ablation == "A1" and row["split"] == "train":
                roles.append("teacher")
            images = []
            for role in roles:
                with Image.open(confined_path(data_root, row[role])) as image:
                    images.append(image.copy())
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{settings['seed']}:{row['pair_id']}".encode()
                ).digest()[:8],
                "big",
            )
            crop_seeds[row["pair_id"]] = seed
            crops = paired_crop(images, size=crop_size, seed=seed)
            tensors = [
                torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1) / 255
                ).unsqueeze(0)
                for image in crops
            ]
            if row["split"] == "train":
                pairs.append(
                    TrainingPair(
                        row["pair_id"], row["source_scene_id"], "train", *tensors
                    )
                )
            else:
                validation.append((row["pair_id"], *tensors))
        provenance = dict(
            dataset_version=f"{document['dataset']}:{document['release']}",
            source_commit=source_commit,
            manifest_sha256=record["manifest_sha256"],
            evaluation_manifest_sha256=record["evaluation_manifest_sha256"],
            ablation=ablation,
            crop_policy=json.dumps(record["crop_policy"], sort_keys=True),
        )
        trainer = BoundedTrainer.create(
            TrainingConfig(**settings),
            pairs,
            model_config=StudentConfig(**(architecture or {})),
            provenance=provenance,
            forbidden_scene_ids=[row["scene_id"] for row in evaluation],
        )
        del pairs, tensors, crops, images
        trainer.model.to(device)
        if resume is not None:
            trainer.restore(resume)
            old_best = resume.parent / "best.pt"
            if old_best.exists():
                # Validate the paired old best state before copying it into this run.
                prior = torch.load(old_best, map_location="cpu", weights_only=True)
                ledger = json.loads(old_best.with_suffix(".pt.budget.json").read_text())
                if (
                    ledger["checkpoint_sha256"] != sha256(old_best)
                    or prior["provenance"] != provenance
                    or prior["initialization_sha256"] != trainer.initialization_sha256
                ):
                    raise ValueError("Prior best checkpoint identity mismatch")
                shutil.copy2(old_best, output / "best.pt")
                shutil.copy2(
                    old_best.with_suffix(".pt.budget.json"),
                    output / "best.pt.budget.json",
                )
        validation_history = []

        def validate(model):
            scores, selection_psnr = [], []
            for pair_id, hazy, clean in validation:
                prediction = (
                    model(hazy.to(device))
                    .clamp(0, 1)[0]
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                target = clean[0].numpy().transpose(1, 2, 0)
                metrics = paired_metrics(target, prediction)
                scores.append(dict(pair_id=pair_id, **metrics))
                # Finite selection score for exact matches; raw metrics retain null+identical.
                mse = float(np.mean((target.astype(np.float64) - prediction) ** 2))
                selection_psnr.append(
                    -10 * math.log10(max(mse, np.finfo(np.float64).tiny))
                )
            validation_history.append(dict(update=trainer.updates, scenes=scores))
            return float(np.mean(selection_psnr)), float(
                np.mean([s["ssim"] for s in scores])
            )

        record.update(
            status="running",
            crop_seeds=crop_seeds,
            initialization_sha256=trainer.initialization_sha256,
            selection_psnr_policy="mean per-pair PSNR; exact-match MSE floored to float64 tiny only for selection",
        )
        persist()
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        training_started = time.perf_counter()
        try:
            trainer.run(
                until_update=until_update,
                max_segment_seconds=max_segment_seconds,
                checkpoint_dir=output,
                validate=validate,
            )
        finally:
            if is_cuda:
                torch.cuda.synchronize(device)
                record["allocated_peak_bytes"] = torch.cuda.max_memory_allocated(device)
                record["reserved_peak_bytes"] = torch.cuda.max_memory_reserved(device)
            record["training_wall_seconds"] = time.perf_counter() - training_started
            record["segment"] = trainer.last_segment
            persist()
        record.update(
            updates=trainer.updates,
            elapsed_seconds=trainer.elapsed_seconds,
            exposures=trainer.exposures,
            training_history=trainer.history,
            validation_history=validation_history,
        )
        selected = output / "best.pt"
        if selected.exists():
            trainer.restore(selected)
            record["selection"] = "corpus_validation_best"
        else:
            selected = output / "last.pt"
            record["selection"] = "last_unselected_no_validation_checkpoint"
        record["selected_checkpoint"] = selected.name
        record["inference_sha256"] = export_student(
            trainer.model, output / "inference.pt"
        )
        (output / "architecture.json").write_text(
            json.dumps(trainer.model.config.to_dict(), indent=2)
        )
        record["status"] = "pilot_completed_not_quality_evidence"
        persist()
        return record
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        persist()
        raise


def main() -> None:
    """Audit by default; --run explicitly enables the bounded training segment."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "evaluation-manifest", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--ablation", choices=("A0", "A1"), default="A0")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-host-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--until-update", type=int, default=100)
    parser.add_argument("--max-segment-seconds", type=float, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run", action="store_true")
    args = vars(parser.parse_args())
    for key in ("config", "architecture"):
        if args[key] is not None:
            args[key] = json.loads(args[key].read_text())
    print(json.dumps(run_pilot(**args), indent=2))


if __name__ == "__main__":
    main()

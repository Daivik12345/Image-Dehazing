"""Persist reproducible run metadata and live frames."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image

from btp.config import ROOT, UPSTREAM, RunConfig
from btp.preview import preview_frames


def revision(path: Path) -> str | None:
    """Return a repository revision without walking unrelated user files."""
    top = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if top.returncode or Path(top.stdout.strip()).resolve() != path.resolve():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path: Path, value: dict) -> None:
    """Atomically replace a JSON document used by the live reader."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def create_run(image: Image.Image, config: RunConfig, root: Path | None = None) -> Path:
    """Save input, immutable configuration, provenance, and initial run state."""
    run = (root or ROOT / "runs") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True)
    image.convert("RGB").save(run / "input.png")
    metadata = {
        "config": config.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": revision(ROOT),
        "upstream_commit": revision(UPSTREAM),
        "dataset_version": "single-image; no training or dataset split",
        "input_sha256": hashlib.sha256((run / "input.png").read_bytes()).hexdigest(),
        "method": "physical CPU baseline"
        if config.backend == "preview"
        else "pretrained paper model",
        "evaluation": "No benchmark evaluation performed",
    }
    write_json(run / "manifest.json", metadata)
    write_json(run / "status.json", {"state": "queued"})
    return run


def save_frame(
    run: Path, stage: str, step: int, total: int, image: Image.Image
) -> None:
    """Publish a frame only after its PNG has been completely written."""
    name = f"{stage}-{step:03d}.png"
    image.save(run / name)
    write_json(
        run / "status.json",
        {
            "state": "running",
            "stage": stage,
            "step": step,
            "total": total,
            "frame": name,
        },
    )
    if step == total:
        image.save(run / f"{stage}.png")


def run_preview(image: Image.Image, config: RunConfig, run: Path) -> None:
    """Execute the tiny CPU smoke/preview pipeline and save actual outputs."""
    try:
        for stage, step, total, frame in preview_frames(image, config):
            save_frame(run, stage, step, total, frame)
        write_json(run / "status.json", {"state": "completed"})
    except Exception as error:
        write_json(run / "status.json", {"state": "failed", "error": str(error)})
        raise

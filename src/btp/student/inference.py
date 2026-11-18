"""Hash-checked, weights-only inference artifacts; no optimizer required."""

import hashlib
from pathlib import Path

import torch
from torch import Tensor

from btp.student.model import CompactDehaze, StudentConfig


def file_sha256(path: Path) -> str:
    """Hash a checkpoint without loading it into memory twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_student(model: CompactDehaze, path: Path) -> str:
    """Export inference-only state and return its SHA256 for the release manifest."""
    torch.save(
        {
            "schema": 1,
            "architecture": model.config.to_dict(),
            "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )
    return file_sha256(path)


class StudentPredictor:
    """Loaded model with explicit [0,1] clipping and preserved input geometry."""

    def __init__(self, model: CompactDehaze, device: str):
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, image: Tensor) -> Tensor:
        """Infer NCHW float32 RGB; output stays on the declared device."""
        return self.model(image.to(device=self.device, dtype=torch.float32)).clamp(0, 1)


def load_student(
    checkpoint: Path, config: StudentConfig, device: str, *, sha256: str
) -> StudentPredictor:
    """Verify expected hash and exact architecture, then safely load tensor weights."""
    if file_sha256(checkpoint) != sha256:
        raise ValueError("student checkpoint SHA256 mismatch")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if artifact.get("schema") != 1 or artifact.get("architecture") != config.to_dict():
        raise ValueError(
            "student architecture/schema mismatch; use the exported config"
        )
    model = CompactDehaze(config)
    model.load_state_dict(artifact["weights"], strict=True)
    return StudentPredictor(model, device)

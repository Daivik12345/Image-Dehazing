"""Validated experiment configuration and checkpoint inventory."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream" / "Learning-Hazing-to-Dehazing"
WEIGHTS = ("v2-1_512-ema-pruned.ckpt", "stage1.pt", "stage2.pt")


@dataclass(frozen=True)
class RunConfig:
    """Settings shared by the live viewer, CPU preview, and pretrained worker."""

    backend: Literal["preview", "pretrained"] = "preview"
    stage: Literal["both", "hazegen", "diffdehaze"] = "both"
    size: int = 256
    steps: int = 10
    seed: int = 42
    blend: float = 0.85
    haze_density: float = 1.2
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    sampler: Literal["spaced", "accsamp"] = "spaced"
    precision: Literal["auto", "float32", "mixed16"] = "auto"
    preview_every: int = 2
    weights_dir: str = str(ROOT / "weights")

    def __post_init__(self) -> None:
        if self.backend not in {"preview", "pretrained"}:
            raise ValueError("Unknown backend")
        if self.stage not in {"both", "hazegen", "diffdehaze"}:
            raise ValueError("Unknown stage")
        if self.size not in {128, 256, 512}:
            raise ValueError("Size must be 128, 256, or 512")
        if not 2 <= self.steps <= 100:
            raise ValueError("Steps must be between 2 and 100")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an unsigned 32-bit integer")
        if not 0 <= self.blend <= 1 or not 0 <= self.haze_density <= 3:
            raise ValueError("Blend must be in [0, 1], density in [0, 3]")
        if self.preview_every < 1:
            raise ValueError("Preview interval must be positive")
        if self.sampler not in {"spaced", "accsamp"}:
            raise ValueError("Unknown sampler")
        if self.precision not in {"auto", "float32", "mixed16"}:
            raise ValueError("Unknown precision")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unknown device")
        if self.sampler == "accsamp" and self.steps % 5:
            raise ValueError("Upstream AccSamp requires steps divisible by 5")

    def to_dict(self) -> dict:
        """Return a JSON-serializable configuration."""
        return asdict(self)


def checkpoint_status(config: RunConfig) -> dict[str, bool]:
    """Check only checkpoints required for the selected stages."""
    names = [WEIGHTS[0]]
    if config.stage in {"both", "hazegen"}:
        names.append(WEIGHTS[1])
    if config.stage in {"both", "diffdehaze"}:
        names.append(WEIGHTS[2])
    return {name: (Path(config.weights_dir) / name).is_file() for name in names}

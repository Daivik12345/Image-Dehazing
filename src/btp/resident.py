"""Persistent experimental AccSamp teacher; CUDA equivalence is not established.

Retains weights only. Each prediction rebuilds sampler/guidance/conditioning and
uses the conservative M0 offload behavior. Optional runtime imports stay lazy.
"""

import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from btp.reference import sha256_file


@dataclass(frozen=True)
class ResidentConfig:
    """Immutable upstream configuration identity; only float32 M0 is supported."""

    config_path: Path
    policy: str = "M0"
    guidance_mode: str = "B0"
    epsilon_var: float | None = None
    confidence_floor: float = 0.25
    confidence_window: int = 15
    max_l2_norm: float | None = None

    def __post_init__(self) -> None:
        if self.policy != "M0":
            raise ValueError("resident adapter currently supports only M0")
        if self.guidance_mode not in {"B0", "confidence", "uniform", "cap"}:
            raise ValueError("unknown guidance mode")
        if self.guidance_mode != "B0" and (
            self.epsilon_var is None
            or not math.isfinite(self.epsilon_var)
            or self.epsilon_var <= 0
        ):
            raise ValueError(
                "candidate guidance requires registered positive epsilon_var"
            )
        if self.guidance_mode == "B0" and self.epsilon_var is not None:
            raise ValueError("B0 does not use confidence epsilon_var")
        if (
            not math.isfinite(self.confidence_floor)
            or not 0 <= self.confidence_floor <= 1
        ):
            raise ValueError("confidence floor must be in [0,1]")
        if (
            type(self.confidence_window) is not int
            or self.confidence_window < 1
            or self.confidence_window % 2 == 0
        ):
            raise ValueError("confidence window must be a positive odd integer")
        if self.guidance_mode == "cap":
            if (
                self.max_l2_norm is None
                or not math.isfinite(self.max_l2_norm)
                or self.max_l2_norm < 0
            ):
                raise ValueError("cap mode requires an explicit finite nonnegative cap")
        elif self.max_l2_norm is not None:
            raise ValueError("correction cap is only allowed in explicit cap mode")

    def guidance_settings(self) -> dict[str, Any]:
        """Return JSON-safe settings to lock identities and annotate diagnostics."""
        return {
            "mode": self.guidance_mode,
            "epsilon_var": self.epsilon_var,
            "floor": self.confidence_floor,
            "window": self.confidence_window,
            "max_l2_norm": self.max_l2_norm,
            "input_policy": (
                "unchanged"
                if self.guidance_mode == "B0"
                else "upstream padded RGB clipped to [0,1] for confidence only"
            ),
        }


def _signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


class ResidentTeacher:
    """Single-worker lifecycle with immutable files and per-image failure records.

    backend_factory is an explicit dependency-injection seam for CPU lifecycle
    tests, not a fallback when CUDA is absent. Checkpoint content hashes are taken
    once; file metadata is checked before each image to detect replacement.
    """

    def __init__(
        self,
        config: ResidentConfig,
        weights: Path,
        device: str = "cuda:0",
        *,
        backend_factory: Any = None,
    ):
        self.config, self.weights, self.device = config, Path(weights), device
        self.files = [
            Path(config.config_path),
            self.weights / "v2-1_512-ema-pruned.ckpt",
            self.weights / "stage2.pt",
        ]
        self.hashes = {str(p): sha256_file(p) for p in self.files}
        self.signatures = {str(p): _signature(p) for p in self.files}
        from btp.config import UPSTREAM

        source_files = sorted(UPSTREAM.rglob("*.py"))
        source_files += [
            Path(__file__),
            Path(__file__).with_name("memory.py"),
            Path(__file__).with_name("loading.py"),
            Path(__file__).with_name("guidance.py"),
        ]
        source = hashlib.sha256()
        for source_file in source_files:
            source.update(source_file.name.encode())
            source.update(sha256_file(source_file).encode())
        self.source_sha256 = source.hexdigest()
        self.identity = hashlib.sha256(
            json.dumps(
                {
                    "hashes": list(self.hashes.values()),
                    "policy": config.policy,
                    "device": device,
                    "source_sha256": self.source_sha256,
                    "guidance": config.guidance_settings(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.closed = False
        started = time.perf_counter()
        self.backend = (backend_factory or _AccSampBackend)(
            config, self.weights, device
        )
        self.load_seconds = time.perf_counter() - started
        self.runtime_identity = dict(
            getattr(
                self.backend,
                "runtime_identity",
                {
                    "backend": type(self.backend).__module__
                    + "."
                    + type(self.backend).__qualname__,
                    "runtime": "injected backend",
                },
            )
        )
        self.identity = hashlib.sha256(
            json.dumps(
                {
                    "configuration_identity": self.identity,
                    "runtime": self.runtime_identity,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def predict(self, input_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
        """Predict one image, recording warm compute separately from disk export.

        A failed image produces a failure record; OOM closes the worker, so later
        requests are explicitly rejected. No automatic reduced-size retry occurs.
        """
        if self.closed:
            raise RuntimeError("resident worker is closed; restart explicitly")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32")
        if any(_signature(p) != self.signatures[str(p)] for p in self.files):
            raise ValueError("resident checkpoint/config changed; create a new worker")
        if (
            dict(getattr(self.backend, "runtime_identity", self.runtime_identity))
            != self.runtime_identity
        ):
            raise ValueError("resident runtime settings changed; create a new worker")
        input_path, output_dir = Path(input_path), Path(output_dir)
        input_hash = sha256_file(input_path)
        output_dir.mkdir(parents=True, exist_ok=False)
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "seed": seed,
            "input_sha256": input_hash,
            "teacher_identity": self.identity,
            "load_seconds": self.load_seconds,
            "source_sha256": self.source_sha256,
            "guidance": self.config.guidance_settings(),
            "runtime": self.runtime_identity,
            "config_sha256": self.hashes[str(self.files[0])],
            "checkpoint_sha256": {p.name: self.hashes[str(p)] for p in self.files[1:]},
            "preprocessing_seconds": None,
            "conditioning_seconds": None,
            "sampling_seconds": None,
            "model_seconds": None,
            "postprocess_seconds": None,
            "allocated_peak_bytes": None,
            "reserved_peak_bytes": None,
            "precision": "float32",
            "policy": "M0",
            "error": None,
        }
        started = time.perf_counter()
        try:
            image, stats = self.backend.predict(input_path, seed)
            record.update(stats)
            record["warm_end_to_end_seconds"] = time.perf_counter() - started
            export = time.perf_counter()
            path = output_dir / "prediction.png"
            image.save(path)
            record.update(
                status="completed",
                output_path=str(path),
                output_sha256=sha256_file(path),
                output_size=list(image.size),
                export_seconds=time.perf_counter() - export,
            )
        except Exception as error:
            record.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                warm_end_to_end_seconds=time.perf_counter() - started,
            )
            if isinstance(error, MemoryError) or "out of memory" in str(error).lower():
                self.close()
        finally:
            self.backend.reset()
            # ru_maxrss is process-lifetime, not an image-local peak.
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            record["process_lifetime_host_peak_bytes"] = (
                peak if sys.platform == "darwin" else peak * 1024
            )
            (output_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    def close(self) -> None:
        """Release models; a closed worker cannot be reused."""
        if not self.closed:
            self.backend.close()
            self.closed = True

    def __enter__(self) -> "ResidentTeacher":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _AccSampBackend:
    """Real upstream adapter with fresh per-image sampler and unmodified guidance."""

    def __init__(self, config: ResidentConfig, weights: Path, device: str):
        os.environ["ATTN_MODE"] = "sdp"
        from btp.config import UPSTREAM

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        import torch
        from diffbir.utils.common import instantiate_from_config
        from omegaconf import OmegaConf

        from btp.loading import load_model

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "resident upstream adapter requires CUDA; no CPU fallback"
            )
        self.torch, self.device = torch, device
        self.config = config
        self.cfg = OmegaConf.load(config.config_path)
        self.instantiate = instantiate_from_config
        self.model = load_model(
            torch,
            instantiate_from_config,
            self.cfg.model.cldm,
            weights / "v2-1_512-ema-pruned.ckpt",
            weights / "stage2.pt",
            mixed=False,
            device="cpu",
        ).to(device)
        self.diffusion = instantiate_from_config(self.cfg.model.diffusion).to(device)
        self.reset()
        torch.cuda.synchronize(device)

    @property
    def runtime_identity(self) -> dict[str, Any]:
        torch, device = self.torch, self.device
        return {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }

    def reset(self) -> None:
        self.sampler = None
        self.guidance = None

    def predict(self, path: Path, seed: int) -> tuple[Any, dict[str, Any]]:
        import cv2
        import numpy as np
        from diffbir.pipeline import pad_to_multiples_of
        from diffbir.sampler import SpacedSampler
        from PIL import Image
        from torchvision.transforms import InterpolationMode, Resize, ToTensor

        from btp.memory import prepare_low_memory

        torch, device = self.torch, self.device

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        stats = {}
        clock = time.perf_counter()
        decoded = cv2.imread(str(path))
        if decoded is None:
            raise ValueError("OpenCV could not decode input")
        image = ToTensor()(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        h, w = image.shape[-2:]
        if h < 512 or w < 512:
            image = Resize(
                512, interpolation=InterpolationMode.BICUBIC, antialias=True
            )(image)
        hh, ww = image.shape[-2:]
        image = pad_to_multiples_of(image, multiple=64).to(device)
        self.sampler = make_m0_sampler_class(torch, SpacedSampler)(
            self.diffusion.betas, self.diffusion.parameterization, rescale_cfg=False
        )
        self.guidance, diagnostics = prepare_image_guidance(
            self.instantiate(self.cfg.guidance),
            image,
            self.config,
        )
        if self.config.guidance_mode != "B0":
            diagnostics["input_policy"] = (
                "upstream padded RGB clipped to [0,1] for confidence only"
            )
        stats["guidance_diagnostics"] = diagnostics
        torch.cuda.synchronize(device)
        stats["preprocessing_seconds"] = time.perf_counter() - clock
        # no_grad, not inference_mode: upstream RGB guidance re-enables autograd.
        with torch.no_grad():
            clock = time.perf_counter()
            self.model.clip.to(device)
            cond = self.model.prepare_condition(image, ["remove dense fog"])
            prepare_low_memory(self.model)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            stats["conditioning_seconds"] = time.perf_counter() - clock
            clock = time.perf_counter()
            z = self.sampler.accsamp(
                model=self.model,
                device=device,
                steps=50,
                x_size=cond["c_img"].shape,
                cond=cond,
                uncond=None,
                cond_fn=self.guidance,
                hazy=image,
                diffusion=self.diffusion,
                cfg_scale=1.0,
                progress=False,
                proportions=[self.cfg.inference.tau, self.cfg.inference.omega],
            )
            torch.cuda.synchronize(device)
            stats["sampling_seconds"] = time.perf_counter() - clock
            clock = time.perf_counter()
            result = ((self.model.vae_decode(z) + 1) / 2)[:, :, :hh, :ww].clip(0, 1)
            result = Resize(
                (h, w), interpolation=InterpolationMode.BICUBIC, antialias=True
            )(result)
            if not torch.isfinite(result).all():
                raise ValueError("non-finite teacher output")
            # Match torchvision.save_image's uint8 rounding and final clipping.
            pixels = (
                result[0]
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            output = Image.fromarray(pixels)
            torch.cuda.synchronize(device)
            stats["postprocess_seconds"] = time.perf_counter() - clock
        stats["allocated_peak_bytes"] = torch.cuda.max_memory_allocated(device)
        stats["reserved_peak_bytes"] = torch.cuda.max_memory_reserved(device)
        return output, stats

    def close(self) -> None:
        self.reset()
        self.model = None
        self.diffusion = None
        self.torch.cuda.empty_cache()


def make_m0_sampler_class(torch: Any, base_sampler: type) -> type:
    """Wrap only M0 module placement, preserving the upstream latent-grad path."""
    from btp.memory import offload_denoisers

    class M0Sampler(base_sampler):
        def apply_cond_fn(self, model: Any, *args: Any, **kwargs: Any) -> Any:
            with offload_denoisers(model):
                torch.cuda.empty_cache()
                return super().apply_cond_fn(model, *args, **kwargs)

    return M0Sampler


def prepare_image_guidance(
    baseline: Any, padded_rgb: Any, config: ResidentConfig
) -> tuple[Any, dict[str, Any]]:
    """Build fresh inference-only confidence; never accepts a clean target.

    The padded input has the same RGB geometry as upstream VAE guidance. B0
    returns the original object without computing maps or modifying correction.
    """
    if config.guidance_mode == "B0":
        return baseline, {"mode": "B0"}
    import torch

    from btp.guidance import RGBGuidanceAdapter, confidence_maps, uniform_control

    if not torch.isfinite(padded_rgb).all():
        raise ValueError("confidence input must be finite")
    maps = confidence_maps(
        padded_rgb.clamp(0, 1),
        epsilon_var=config.epsilon_var,
        floor=config.confidence_floor,
        window=config.confidence_window,
    )
    weight = (
        uniform_control(maps.weight)
        if config.guidance_mode == "uniform"
        else maps.weight
    )
    adapter = RGBGuidanceAdapter(baseline, weight, max_l2_norm=config.max_l2_norm)
    diagnostics = dict(maps.statistics, **config.guidance_settings())
    diagnostics["applied_weight_mean"] = weight.mean().item()
    diagnostics["applied_weight_min"] = weight.min().item()
    diagnostics["applied_weight_max"] = weight.max().item()
    return adapter, diagnostics

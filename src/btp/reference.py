"""Run the unmodified upstream inference entry points with recorded provenance."""

import argparse
import hashlib
import json
import os
import random
import runpy
import shutil
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from btp.config import ROOT, UPSTREAM
from btp.experiment import revision, write_json
from btp.memory import (
    CACHE_POLICIES,
    POLICIES,
    policy_modules,
    resolve_cache_policy,
    should_empty_cache,
)

SCRIPTS = {
    "hazegen": ("inference_stage1.py", "stage1.yaml", "stage1.pt"),
    "spaced": ("inference_stage2.py", "stage2.yaml", "stage2.pt"),
    "accsamp": ("inference_accsamp.py", "stage2_accsamp.yaml", "stage2.pt"),
}


def sha256_file(path: Path) -> str:
    """Hash a file without loading checkpoint contents into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_dimensions(width: int, height: int) -> tuple[int, int]:
    """Match upstream torchvision Resize(512) when either side is below 512."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if min(width, height) >= 512:
        return width, height
    if width <= height:
        return 512, int(height * 512 / width)
    return int(width * 512 / height), 512


def run_reference(
    image: Path,
    weights: Path,
    output: Path,
    mode: str,
    seed: int,
    dataset_version: str,
    preview_every: int = 10,
    low_memory: bool = False,
    memory_policy: str | None = None,
    profile: bool = False,
    cache_policy: str | None = None,
) -> None:
    """Run one CUDA reference image, preserving upstream resize, padding and sampler.

    Compatibility changes are restricted checkpoint loading, fixed RNG seeds,
    progress logging and optional diagnostic latent decodes. No mixed precision,
    AlignOp patch, image size cap, or sampler changes apply. Optional low-memory
    mode freezes weights and offloads idle modules, retaining latent gradients.
    """
    if mode not in SCRIPTS or not 0 <= seed < 2**32 or preview_every < 0:
        raise ValueError("Invalid mode, seed, or preview interval")
    policy = memory_policy or ("M0" if low_memory else "R0")
    if policy not in POLICIES or (profile and memory_policy is None):
        raise ValueError(
            "Invalid policy, or profiling without explicit evaluation policy"
        )
    cache = resolve_cache_policy(policy, cache_policy)
    if cache_policy is not None and memory_policy is None:
        raise ValueError("Explicit cache policy requires an explicit memory policy")
    low_memory = policy != "R0"
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    image, weights, output = image.resolve(), weights.resolve(), output.resolve()
    script, config_name, stage_file = SCRIPTS[mode]
    needed = [weights / "v2-1_512-ema-pruned.ckpt", weights / stage_file]
    for path in [image, *needed]:
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["ATTN_MODE"] = "sdp"
    sys.path.insert(0, str(UPSTREAM))
    import cv2
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from btp.loading import load_tensors

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA reference run requires a Kaggle GPU; no CPU/MPS fallback"
        )
    output.mkdir(parents=True)
    decoded = cv2.imread(str(image))
    if decoded is None:
        raise ValueError(f"OpenCV could not decode {image}")
    height, width = decoded.shape[:2]
    Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)).save(output / "input.png")
    write_json(output / "status.json", {"state": "loading"})
    work_input = output / "inputs"
    work_input.mkdir()
    shutil.copy2(output / "input.png", work_input / "input.png")
    cfg = OmegaConf.load(UPSTREAM / "configs/inference" / config_name)
    cfg.inference.sd_path = str(needed[0])
    cfg.inference.controlnet_path = str(needed[1])
    cfg.inference.image_folder = str(work_input)
    cfg.inference.result_folder = str(output / "final")
    OmegaConf.save(cfg, output / "config.yaml")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    snapshot_path = ROOT / "source-manifest.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
    manifest = {
        "mode": mode,
        "seed": seed,
        "steps": 50,
        "precision": "float32",
        "low_memory": low_memory,
        "memory_policy": policy,
        "cache_policy": cache,
        "profiled": profile,
        "memory_strategy": (
            f"frozen weights; CLIP CPU after conditioning; placement policy {policy}"
            if low_memory
            else "original model residency"
        ),
        "source_commit": revision(UPSTREAM) or (snapshot or {}).get("upstream_commit"),
        "source_snapshot": snapshot,
        "input_source": str(image),
        "normalized_input_sha256": sha256_file(output / "input.png"),
        "input_normalization": "OpenCV IMREAD_COLOR decode saved losslessly as RGB PNG; no spatial resize here",
        "dataset_version": dataset_version,
        "input_sha256": sha256_file(image),
        "input_size": [width, height],
        "inference_size_before_padding": list(reference_dimensions(width, height)),
        "padding": "upstream zero padding to multiples of 64",
        "output_resize": "upstream bicubic resize back to original input dimensions",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_sha256": {p.name: sha256_file(p) for p in needed},
        "evaluation": "Single-image inference; metrics scored separately after inference",
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "padded_size": [
            ((v + 63) // 64) * 64 for v in reference_dimensions(width, height)
        ],
        "compatibility": "Torch 2.6 restricted legacy-metadata loader; native SDPA",
        "previews": "Diagnostic decodes of noisy sampler states, not final predictions",
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps({"event": "starting", "output": str(output), "mode": mode}),
        flush=True,
    )
    from diffbir.sampler import SpacedSampler

    from btp.telemetry import Telemetry

    monitor = Telemetry(torch, output, profile) if memory_policy is not None else None

    original_step = SpacedSampler.p_sample
    original_guidance = SpacedSampler.apply_cond_fn
    original_load = torch.load
    calls = 0

    def guidance(sampler, model, *args, **kwargs):
        if not low_memory and monitor is None:
            return original_guidance(sampler, model, *args, **kwargs)
        from btp.memory import offload_denoisers

        # Original signature: model, pred_x0, t, index, cond_fn.
        t = args[1] if len(args) > 1 else kwargs["t"]
        cond_fn = args[3] if len(args) > 3 else kwargs["cond_fn"]
        active = (
            cond_fn.t_stop < int(t[0].item()) + 1 < cond_fn.t_start
            and cond_fn.repeat > 0
        )
        modules = policy_modules(policy, active)
        if monitor:
            monitor.counts["guidance_active" if active else "guidance_inactive"] += 1
            monitor.in_guidance = True
        try:
            with (
                monitor.phase("guidance_with_transfers") if monitor else nullcontext(),
                offload_denoisers(
                    model,
                    modules=modules,
                    observe=monitor.transfer if monitor else None,
                    phase=monitor.phase if monitor else None,
                ),
            ):
                if should_empty_cache(cache, "guidance", offloaded=bool(modules)):
                    with monitor.phase("empty_cache") if monitor else nullcontext():
                        torch.cuda.empty_cache()
                return original_guidance(sampler, model, *args, **kwargs)
        finally:
            if monitor:
                monitor.in_guidance = False

    def step(sampler, model, *args, **kwargs):
        nonlocal calls
        if low_memory and calls == 0:
            from btp.memory import prepare_low_memory

            prepare_low_memory(model)
            if should_empty_cache(cache, "conditioning"):
                with (
                    monitor.phase("empty_cache_conditioning")
                    if monitor
                    else nullcontext()
                ):
                    torch.cuda.empty_cache()
        result = original_step(sampler, model, *args, **kwargs)
        calls += 1
        if monitor:
            monitor.counts["sampler_calls"] = calls
        state = {"state": "running", "mode": mode, "sampler_calls": calls}
        if preview_every and calls % preview_every == 0:
            devices = [result.device.index or 0] if result.device.type == "cuda" else []
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                decoded = ((model.vae_decode(result) + 1) / 2).clamp(0, 1)
                if not torch.isfinite(decoded).all():
                    raise RuntimeError("Non-finite diagnostic image")
                pixels = decoded[0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray(np.uint8(pixels * 255)).save(
                output / f"latent-{calls:03d}.png"
            )
            state["frame"] = f"latent-{calls:03d}.png"
        write_json(output / "status.json", state)
        print(json.dumps(state), flush=True)
        return result

    allowed = {path.resolve() for path in needed}

    def checkpoint_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() in allowed:
            with patch.object(torch, "load", original_load):
                return load_tensors(
                    torch, Path(path), legacy_metadata=Path(path).name.endswith(".ckpt")
                )
        kwargs["weights_only"] = True
        return original_load(path, *args, **kwargs)

    try:
        with (
            ExitStack() as instrumentation,
            patch.object(torch, "load", checkpoint_load),
            patch.object(SpacedSampler, "p_sample", step),
            patch.object(SpacedSampler, "apply_cond_fn", guidance),
        ):
            if monitor:
                from diffbir.model import ControlLDM

                def phase_wrapper(original, label):
                    def wrapped(*args, **kwargs):
                        with monitor.phase(label):
                            result = original(*args, **kwargs)
                        if label == "vae_decode" and not monitor.in_sampling:
                            if not torch.isfinite(result).all():
                                raise RuntimeError("Non-finite final VAE decode")
                        return result

                    return wrapped

                for method, label in [
                    ("prepare_condition", "conditioning"),
                    ("prepare_text_condition", "text_conditioning"),
                    ("forward", "denoiser_forward"),
                    ("vae_decode", "vae_decode"),
                    ("vae_encode", "vae_encode"),
                    ("load_pretrained_sd", "load_sd_state"),
                    ("load_controlnet_from_ckpt", "load_control_state"),
                ]:
                    instrumentation.enter_context(
                        patch.object(
                            ControlLDM,
                            method,
                            phase_wrapper(getattr(ControlLDM, method), label),
                        )
                    )

                def sample_wrapper(original):
                    def wrapped(*args, **kwargs):
                        with monitor.sample():
                            return original(*args, **kwargs)

                    return wrapped

                for method in ("sample", "accsamp"):
                    instrumentation.enter_context(
                        patch.object(
                            SpacedSampler,
                            method,
                            sample_wrapper(getattr(SpacedSampler, method)),
                        )
                    )
                instrumentation.enter_context(
                    patch.object(
                        torch.autograd,
                        "backward",
                        phase_wrapper(torch.autograd.backward, "backward"),
                    )
                )
                if profile:
                    profiler = instrumentation.enter_context(
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            profile_memory=True,
                            record_shapes=False,
                        )
                    )
            with patch.object(
                sys, "argv", [script, "--config", str(output / "config.yaml")]
            ):
                runpy.run_path(str(UPSTREAM / script), run_name="__main__")
        if monitor and profile:
            profiler.export_chrome_trace(str(output / "profile.json"))
        result = output / "final/input.png"
        if not result.is_file():
            raise RuntimeError("Upstream did not save its final output")
        with Image.open(result) as final:
            if final.size != (width, height):
                raise RuntimeError(
                    "Final dimensions do not match the upstream contract"
                )
        write_json(
            output / "status.json", {"state": "completed", "sampler_calls": calls}
        )
        manifest["peak_gpu_allocated_bytes"] = max(
            monitor.before_peak[0] if monitor else 0, torch.cuda.max_memory_allocated()
        )
        manifest["peak_gpu_reserved_bytes"] = max(
            monitor.before_peak[1] if monitor else 0, torch.cuda.max_memory_reserved()
        )
        write_json(output / "manifest.json", manifest)
        if monitor:
            monitor.save("completed")
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error)})
        if monitor:
            monitor.save("failed", str(error))
        raise


def main() -> None:
    """Launch one recorded upstream CUDA inference run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=list(SCRIPTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--memory-policy", choices=POLICIES)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cache-policy", choices=CACHE_POLICIES)
    args = parser.parse_args()
    run_reference(
        args.input,
        args.weights,
        args.output,
        args.mode,
        args.seed,
        args.dataset_version,
        args.preview_every,
        args.low_memory,
        args.memory_policy,
        args.profile,
        args.cache_policy,
    )


if __name__ == "__main__":
    main()

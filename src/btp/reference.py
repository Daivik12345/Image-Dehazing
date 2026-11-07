"""Run the unmodified upstream inference entry points with recorded provenance."""

import argparse
import hashlib
import json
import os
import random
import runpy
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from btp.config import ROOT, UPSTREAM
from btp.experiment import revision, write_json

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
) -> None:
    """Run one CUDA reference image, preserving upstream resize, padding and sampler.

    Compatibility changes are restricted checkpoint loading, fixed RNG seeds,
    progress logging and optional diagnostic latent decodes. No mixed precision,
    AlignOp patch, image size cap, or sampler changes apply. Optional low-memory
    mode freezes weights and offloads idle modules, retaining latent gradients.
    """
    if mode not in SCRIPTS or not 0 <= seed < 2**32 or preview_every < 0:
        raise ValueError("Invalid mode, seed, or preview interval")
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
        "memory_strategy": (
            "frozen weights; CLIP CPU after conditioning; denoisers CPU during guidance"
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
        "evaluation": "Single-image inference; no paper benchmark metrics computed",
        "compatibility": "Torch 2.6 restricted legacy-metadata loader; native SDPA",
        "previews": "Diagnostic decodes of noisy sampler states, not final predictions",
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps({"event": "starting", "output": str(output), "mode": mode}),
        flush=True,
    )
    from diffbir.sampler import SpacedSampler

    original_step = SpacedSampler.p_sample
    original_guidance = SpacedSampler.apply_cond_fn
    original_load = torch.load
    calls = 0

    def guidance(sampler, model, *args, **kwargs):
        if not low_memory:
            return original_guidance(sampler, model, *args, **kwargs)
        from btp.memory import offload_denoisers

        with offload_denoisers(model):
            torch.cuda.empty_cache()
            return original_guidance(sampler, model, *args, **kwargs)

    def step(sampler, model, *args, **kwargs):
        nonlocal calls
        if low_memory and calls == 0:
            from btp.memory import prepare_low_memory

            prepare_low_memory(model)
            torch.cuda.empty_cache()
        result = original_step(sampler, model, *args, **kwargs)
        calls += 1
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
            patch.object(torch, "load", checkpoint_load),
            patch.object(SpacedSampler, "p_sample", step),
            patch.object(SpacedSampler, "apply_cond_fn", guidance),
        ):
            with patch.object(
                sys, "argv", [script, "--config", str(output / "config.yaml")]
            ):
                runpy.run_path(str(UPSTREAM / script), run_name="__main__")
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
        manifest["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / "manifest.json", manifest)
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error)})
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
    )


if __name__ == "__main__":
    main()

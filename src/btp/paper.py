"""Separate-process adapter for the authors' actual ControlLDM and samplers."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from btp.config import UPSTREAM, RunConfig, checkpoint_status
from btp.experiment import save_frame, write_json
from btp.loading import load_model


def select_device(torch: Any, requested: str) -> str:
    """Select a supported accelerator, failing clearly for unavailable devices."""
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available.get(requested):
        raise ValueError(f"Device {requested!r} is unavailable")
    return requested


def _run_paper(run: Path) -> None:
    """Load pretrained weights and publish decoded denoising steps to disk."""
    manifest = json.loads((run / "manifest.json").read_text())
    config = RunConfig(**manifest["config"])
    missing = [name for name, exists in checkpoint_status(config).items() if not exists]
    if missing:
        raise FileNotFoundError("Missing pretrained checkpoints: " + ", ".join(missing))
    os.environ.setdefault("ATTN_MODE", "sdp")
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from diffbir.sampler import SpacedSampler
    from diffbir.utils.common import instantiate_from_config
    from omegaconf import OmegaConf

    from btp.preview import as_image, rgb_array

    device = select_device(torch, config.device)
    mixed = config.precision == "mixed16" or (
        config.precision == "auto" and device != "cpu"
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    # Seeds are recorded; identical floating point results across devices aren't promised.
    manifest["runtime"] = {
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "precision": "mixed16" if mixed else "float32",
    }
    manifest["checkpoints"] = {
        name: {
            "path": str(Path(config.weights_dir) / name),
            "bytes": (Path(config.weights_dir) / name).stat().st_size,
        }
        for name in checkpoint_status(config)
    }
    for name, info in manifest["checkpoints"].items():
        provenance = Path(config.weights_dir) / (name + ".provenance.json")
        if provenance.exists():
            info["provenance"] = json.loads(provenance.read_text())
    write_json(run / "manifest.json", manifest)
    source = rgb_array(Image.open(run / "input.png"), config.size)
    height, width = source.shape[:2]
    current = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).to(device)
    current = functional.pad(
        current, (0, (-width) % 64, 0, (-height) % 64), mode="replicate"
    )
    stages = ["hazegen", "diffdehaze"] if config.stage == "both" else [config.stage]
    for stage in stages:
        write_json(run / "status.json", {"state": "loading", "stage": stage})
        cfg = OmegaConf.load(
            UPSTREAM
            / "configs/inference"
            / ("stage1.yaml" if stage == "hazegen" else "stage2_accsamp.yaml")
        )
        weights = Path(config.weights_dir)

        def loading_progress(phase: str, stage_name: str = stage) -> None:
            print(f"{stage_name}: {phase}", flush=True)
            write_json(
                run / "status.json",
                {"state": "loading", "stage": stage_name, "phase": phase},
            )

        model = load_model(
            torch,
            instantiate_from_config,
            cfg.model.cldm,
            weights / "v2-1_512-ema-pruned.ckpt",
            weights / ("stage1.pt" if stage == "hazegen" else "stage2.pt"),
            mixed,
            device=device,
            progress=loading_progress,
        ).to(device)

        diffusion = instantiate_from_config(cfg.model.diffusion).to(device)
        total = (
            config.steps // 5 + config.steps * 3 // 5
            if stage == "diffdehaze" and config.sampler == "accsamp"
            else config.steps
        )

        class LiveSampler(SpacedSampler):
            """Observe predicted clean latents without changing sampling equations."""

            def _predict_xstart_from_eps(self, *args, **kwargs):
                self.prediction = super()._predict_xstart_from_eps(*args, **kwargs)
                return self.prediction

            def _predict_xstart_from_v(self, *args, **kwargs):
                self.prediction = super()._predict_xstart_from_v(*args, **kwargs)
                return self.prediction

            def p_sample(self, *args, **kwargs):
                result = super().p_sample(*args, **kwargs)
                self.completed += 1
                if (
                    self.completed % config.preview_every == 0
                    or self.completed == self.expected_steps
                ):
                    with torch.no_grad():
                        decoded = (self.decode(self.prediction) + 1) / 2
                        if not torch.isfinite(decoded).all():
                            raise RuntimeError(
                                "Non-finite decoded frame; inference stopped"
                            )
                        frame = (
                            decoded[0, :, :height, :width]
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                    save_frame(
                        run,
                        self.stage_name,
                        self.completed,
                        self.expected_steps,
                        as_image(frame),
                    )
                else:
                    write_json(
                        run / "status.json",
                        {
                            "state": "running",
                            "stage": self.stage_name,
                            "step": self.completed,
                            "total": self.expected_steps,
                        },
                    )
                return result

        sampler = LiveSampler(
            diffusion.betas, diffusion.parameterization, rescale_cfg=False
        )
        sampler.completed = 0
        sampler.expected_steps = total
        sampler.stage_name = stage
        sampler.decode = model.vae_decode
        loading_progress("Encoding image and prompt")
        with torch.no_grad():
            prompt = (
                "hazy, foggy, misty, obscure, smoggy."
                if stage == "hazegen"
                else "remove dense fog"
            )
            cond = model.prepare_condition(current, [prompt])
            uncond = (
                {"c_img": torch.zeros_like(cond["c_img"]), "c_txt": cond["c_txt"]}
                if stage == "hazegen"
                else None
            )
            # Prompt embeddings are complete; sampling never calls the text encoder.
            # Release its weights instead of retaining them for every denoising step.
            model.clip = None
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            kwargs = dict(
                model=model,
                device=device,
                steps=config.steps,
                x_size=cond["c_img"].shape,
                cond=cond,
                uncond=uncond,
                cfg_scale=config.blend if stage == "hazegen" else 1.0,
                progress=False,
            )
            write_json(
                run / "status.json",
                {"state": "running", "stage": stage, "step": 0, "total": total},
            )
            if stage == "diffdehaze" and config.sampler == "accsamp":
                # Upstream AlignOp leaves uncovered right/bottom strips for some shapes.
                # Pad to its patch grid, call its implementation, then crop back.
                from diffbir.sampler import spaced_sampler

                original_align = spaced_sampler.tiled_align

                def covered_align(hazy, predicted, align=original_align):
                    h, w = hazy.shape[-2:]
                    pad = (0, (-(w - 37)) % 10, 0, (-(h - 37)) % 10)
                    return align(
                        functional.pad(hazy, pad, mode="replicate"),
                        functional.pad(predicted, pad, mode="replicate"),
                    )[:, :, :h, :w]

                spaced_sampler.tiled_align = covered_align
                try:
                    guidance = instantiate_from_config(cfg.guidance)
                    z = sampler.accsamp(
                        **kwargs,
                        cond_fn=guidance,
                        hazy=current,
                        diffusion=diffusion,
                        proportions=[0.8, 0.6],
                    )
                finally:
                    spaced_sampler.tiled_align = original_align
            else:
                z = sampler.sample(**kwargs)
            current = ((model.vae_decode(z) + 1) / 2).clamp(0, 1)
            if not torch.isfinite(current).all():
                raise RuntimeError("Non-finite model output")
            frame = as_image(
                current[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
            )
            save_frame(run, stage, total, total, frame)
            # Quantize between stages exactly as saving/reloading upstream PNG output.
            current = (
                torch.from_numpy(np.asarray(frame).copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
                .float()
                / 255
            )
            current = functional.pad(
                current, (0, (-width) % 64, 0, (-height) % 64), mode="replicate"
            )
        del model, sampler, cond, uncond, z, kwargs
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    write_json(run / "status.json", {"state": "completed"})


def run_paper(run: Path) -> None:
    """Run the adapter with the explicit head dimension required by MPS SDPA."""
    config = RunConfig(**json.loads((run / "manifest.json").read_text())["config"])
    missing = [name for name, exists in checkpoint_status(config).items() if not exists]
    if missing:
        raise FileNotFoundError("Missing pretrained checkpoints: " + ", ".join(missing))
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention

    def attention(query, key, value, *args, **kwargs):
        # Upstream flattens heads into batch and sends a 3-D tensor. Torch 2.6
        # MPS expects a head axis; a singleton axis preserves the same equation.
        if query.device.type == "mps" and query.ndim == 3:
            return original(
                query.unsqueeze(1),
                key.unsqueeze(1),
                value.unsqueeze(1),
                *args,
                **kwargs,
            ).squeeze(1)
        return original(query, key, value, *args, **kwargs)

    with patch.object(functional, "scaled_dot_product_attention", attention):
        _run_paper(run)


def main() -> None:
    """Execute one saved experiment in the isolated inference environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        run_paper(args.run)
    except Exception as error:
        write_json(args.run / "status.json", {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()

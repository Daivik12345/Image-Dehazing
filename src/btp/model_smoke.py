"""Tiny CPU architecture check using original modules with random weights."""

import json
import os
import sys
from unittest.mock import patch

from btp.config import ROOT, UPSTREAM, RunConfig
from btp.experiment import create_run, write_json
from btp.preview import demo_image


def main() -> None:
    """Verify model imports and two sampling steps; never claim trained quality."""
    os.environ["ATTN_MODE"] = "sdp"
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import torch
    from diffbir.model import ControlLDM, Diffusion
    from diffbir.sampler import SpacedSampler

    torch.set_num_threads(2)
    torch.manual_seed(42)
    common = dict(
        image_size=8,
        in_channels=4,
        model_channels=32,
        attention_resolutions=[1, 2],
        num_res_blocks=1,
        channel_mult=[1, 2],
        num_head_channels=8,
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
        transformer_depth=1,
        context_dim=32,
        legacy=False,
        use_checkpoint=False,
    )
    model_config = dict(
        unet_cfg={**common, "out_channels": 4},
        controlnet_cfg={**common, "hint_channels": 4},
        vae_cfg={
            "embed_dim": 4,
            "ddconfig": dict(
                double_z=True,
                z_channels=4,
                resolution=64,
                in_channels=3,
                out_ch=3,
                ch=32,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=1,
                attn_resolutions=[],
                dropout=0.0,
            ),
        },
        clip_cfg={
            "embed_dim": 32,
            "vision_cfg": dict(
                image_size=32, layers=1, width=32, head_width=8, patch_size=16
            ),
            "text_cfg": dict(
                context_length=77, vocab_size=49408, width=32, heads=4, layers=2
            ),
            "layer": "penultimate",
        },
        latent_scale_factor=0.18215,
    )
    model = ControlLDM(**model_config).eval()
    image = demo_image(64)
    run = create_run(image, RunConfig(size=128, steps=2))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["method"] = (
        "tiny random-weight upstream architecture smoke; NOT pretrained inference"
    )
    manifest["config"]["backend"] = "architecture_smoke"
    manifest["architecture"] = {
        "model_channels": 32,
        "context_dim": 32,
        "channel_mult": [1, 2],
        "image_size": 64,
    }
    manifest["runtime"] = {"torch": torch.__version__, "device": "cpu"}
    write_json(run / "manifest.json", manifest)
    tensor = (
        torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
        / 255
    )
    diffusion = Diffusion(linear_start=0.00085, linear_end=0.012)
    with torch.no_grad():
        cond = model.prepare_condition(tensor, ["hazy, foggy"])
        sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, False)
        z = sampler.sample(
            model=model,
            device="cpu",
            steps=2,
            x_size=cond["c_img"].shape,
            cond=cond,
            uncond=None,
            cfg_scale=1.0,
            progress=False,
        )
        decoded = model.vae_decode(z)
    if decoded.shape != tensor.shape or not torch.isfinite(decoded).all():
        raise RuntimeError("Tiny model smoke failed: invalid output")
    write_json(
        run / "status.json",
        {
            "state": "completed",
            "check": "finite decoded tensor; random weights",
            "shape": list(decoded.shape),
        },
    )
    print(f"Tiny CPU model architecture smoke passed: {run.relative_to(ROOT)}")
    # Exercise the full file-loading adapter with explicitly random test checkpoints.
    from omegaconf import OmegaConf

    from btp.paper import run_paper

    weight_dir = run / "random-test-weights"
    weight_dir.mkdir()
    base = {}
    for name, prefix in {
        "unet": "model.diffusion_model",
        "vae": "first_stage_model",
        "clip": "cond_stage_model",
    }.items():
        for key, value in getattr(model, name).state_dict().items():
            base[f"{prefix}.{key}"] = value
    torch.save({"state_dict": base}, weight_dir / "v2-1_512-ema-pruned.ckpt")
    for name in ["stage1.pt", "stage2.pt"]:
        torch.save(model.controlnet.state_dict(), weight_dir / name)
    original_load = OmegaConf.load

    def small_config(path):
        cfg = original_load(path)
        cfg.model.cldm.params = OmegaConf.create(model_config)
        return cfg

    modes = [("spaced", "cpu", "float32"), ("accsamp", "cpu", "float32")]
    if torch.backends.mps.is_available():
        modes.append(("spaced", "mps", "mixed16"))
        modes.append(("accsamp", "mps", "mixed16"))
    else:
        print("MPS mixed16 smoke skipped: device unavailable")
    for mode, device, precision in modes:
        config = RunConfig(
            backend="pretrained",
            stage="both",
            size=128,
            steps=5,
            device=device,
            sampler=mode,
            precision=precision,
            weights_dir=str(weight_dir),
            preview_every=1,
        )
        adapter_run = create_run(image, config)
        manifest = json.loads((adapter_run / "manifest.json").read_text())
        manifest["method"] = "tiny random-weight adapter smoke; NOT pretrained results"
        manifest["architecture"] = model_config
        write_json(adapter_run / "manifest.json", manifest)
        with patch.object(OmegaConf, "load", side_effect=small_config):
            run_paper(adapter_run)
        for stage in ["hazegen", "diffdehaze"]:
            if not (adapter_run / f"{stage}.png").is_file():
                raise RuntimeError(f"Missing {stage} output")
        print(
            f"Tiny {device}/{precision}/{mode} adapter smoke passed: {adapter_run.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()

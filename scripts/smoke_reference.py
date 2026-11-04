"""CPU-only runtime checks for the CUDA reference shim; no pretrained inference."""

import json
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torchvision.transforms import Resize

from btp import reference
from btp.config import UPSTREAM


def main() -> None:
    """Check actual preprocessing helpers and loader/sampler hooks using a toy stub."""
    sys.path.insert(0, str(UPSTREAM))
    from diffbir.pipeline import pad_to_multiples_of

    for width, height in [(640, 427), (128, 128), (1024, 768), (100, 200)]:
        image = torch.ones(1, 3, height, width)
        resized = (
            Resize(512, antialias=True)(image) if min(width, height) < 512 else image
        )
        assert tuple(reversed(resized.shape[-2:])) == reference.reference_dimensions(
            width, height
        )
        padded = pad_to_multiples_of(resized, 64)
        h, w = resized.shape[-2:]
        if padded.shape[-1] > w:
            assert torch.count_nonzero(padded[..., w:]) == 0
        if padded.shape[-2] > h:
            assert torch.count_nonzero(padded[..., h:, :]) == 0
    for script in reference.SCRIPTS.values():
        with patch.object(sys, "argv", [script[0], "--help"]):
            try:
                runpy.run_path(str(UPSTREAM / script[0]), run_name="__main__")
            except SystemExit as error:
                assert error.code == 0
    print("Original script imports and short-edge/zero-padding checks passed.")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "configs/inference").mkdir(parents=True)
        (root / "configs/inference/stage1.yaml").write_text("inference: {}\n")
        weights = root / "weights"
        weights.mkdir()
        torch.save(
            {"state_dict": {"test": torch.ones(1)}},
            weights / "v2-1_512-ema-pruned.ckpt",
        )
        torch.save({"test": torch.ones(1)}, weights / "stage1.pt")
        Image.new("RGB", (4, 4), "gray").save(root / "image.png")
        # A tiny CPU test double exercises the restricted-loader recursion boundary
        # and actual SpacedSampler hook. This does not test pretrained quality.
        (root / "inference_stage1.py").write_text("""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from diffbir.model import Diffusion
from diffbir.sampler import SpacedSampler
cfg=OmegaConf.load(sys.argv[2])
assert 'state_dict' in torch.load(cfg.inference.sd_path, map_location='cpu')
assert 'test' in torch.load(cfg.inference.controlnet_path, map_location='cpu')
class Toy(torch.nn.Module):
 def forward(self,x,t,cond): return torch.zeros_like(x)
 def vae_decode(self,x): return x
model=Toy()
diffusion=Diffusion()
sampler=SpacedSampler(diffusion.betas, diffusion.parameterization, False)
z=sampler.sample(model,'cpu',50,(1,3,4,4),{},None,1.,progress=False)
result=(z[0].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
output=Path(cfg.inference.result_folder)
output.mkdir()
Image.fromarray(result).save(output/'input.png')
""")
        with (
            patch.object(reference, "UPSTREAM", root),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "manual_seed_all"),
            patch.object(torch.cuda, "get_device_name", return_value="CPU test double"),
        ):
            reference.run_reference(
                root / "image.png",
                weights,
                root / "out",
                "hazegen",
                42,
                "test-double-v1",
                10,
            )
        state = json.loads((root / "out/status.json").read_text())
        assert state["state"] == "completed" and state["sampler_calls"] == 50
        assert len(list((root / "out").glob("latent-*.png"))) == 5
    print(
        "Restricted loader and sampler hook CPU test-double check passed; no GPU inference performed."
    )


if __name__ == "__main__":
    main()

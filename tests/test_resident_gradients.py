"""Optional real upstream guidance plumbing check, no weights or CUDA required."""

import importlib.util
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime absent")
class ResidentGradientTest(unittest.TestCase):
    def test_m0_wrapper_preserves_upstream_latent_gradient(self):
        import numpy as np
        import torch

        from btp.config import UPSTREAM

        sys.path.insert(0, str(UPSTREAM))
        from diffbir.sampler import SpacedSampler
        from diffbir.utils.cond_fn import MSEGuidance

        from btp.resident import make_m0_sampler_class

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vae = torch.nn.Conv2d(3, 3, 1)
                self.unet = torch.nn.Linear(1, 1)
                self.controlnet = torch.nn.Linear(1, 1)
                self.clip = torch.nn.Linear(1, 1)
                self.requires_grad_(False)

            def vae_decode(self, x):
                return self.vae(x)

        torch.manual_seed(42)
        model = Model()
        betas = np.linspace(0.00085, 0.012, 1000)
        baseline = SpacedSampler(betas, "eps", rescale_cfg=False)
        resident = make_m0_sampler_class(torch, SpacedSampler)(
            betas, "eps", rescale_cfg=False
        )
        for sampler in (baseline, resident):
            sampler.make_schedule(50)
            sampler.to("cpu")
        guidance = MSEGuidance(0.1, 601, -1, "rgb", 1)
        guidance.load_target(torch.zeros(1, 3, 4, 4))
        latent = torch.rand(1, 3, 4, 4)
        args = (model, latent, torch.tensor([100]), torch.tensor([5]), guidance)
        with torch.no_grad():
            expected = baseline.apply_cond_fn(*args)
            actual = resident.apply_cond_fn(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, latent))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fresh_confidence_uniform_and_baseline(self):
        from pathlib import Path

        import torch

        from btp.resident import ResidentConfig, prepare_image_guidance

        class Baseline:
            space = "rgb"

            def __call__(self, target, prediction, timestep):
                return torch.ones_like(prediction), 1.0

        baseline = Baseline()
        image = torch.rand(1, 3, 16, 16)
        original, _ = prepare_image_guidance(
            baseline, image, ResidentConfig(Path("fixture"))
        )
        self.assertIs(original, baseline)
        cfg = ResidentConfig(
            Path("fixture"), guidance_mode="confidence", epsilon_var=1e-4
        )
        first, diagnostics = prepare_image_guidance(baseline, image, cfg)
        second, constant_stats = prepare_image_guidance(
            baseline, torch.zeros_like(image), cfg
        )
        prediction = image * 2 - 1
        a, _ = first(prediction, prediction, 100)
        b, _ = second(prediction, prediction, 100)
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(constant_stats["applied_weight_mean"], 0.25)
        uniform, uniform_stats = prepare_image_guidance(
            baseline,
            image,
            ResidentConfig(Path("fixture"), guidance_mode="uniform", epsilon_var=1e-4),
        )
        u, _ = uniform(prediction, prediction, 100)
        self.assertAlmostEqual(u.mean().item(), a.mean().item(), places=6)
        self.assertAlmostEqual(
            uniform_stats["applied_weight_min"],
            uniform_stats["applied_weight_max"],
            places=6,
        )
        self.assertIn("variance_mean", diagnostics)
        capped, _ = prepare_image_guidance(
            baseline,
            image,
            ResidentConfig(
                Path("fixture"), guidance_mode="cap", epsilon_var=1e-4, max_l2_norm=0.1
            ),
        )
        c, _ = capped(prediction, prediction, 100)
        self.assertLessEqual(torch.linalg.vector_norm(c).item(), 0.100001)

    def test_bicubic_overshoot_only_clipped_for_confidence(self):
        from pathlib import Path

        import torch
        from torchvision.transforms import InterpolationMode, Resize

        from btp.resident import ResidentConfig, prepare_image_guidance

        class Baseline:
            space = "rgb"

            def __call__(self, target, prediction, timestep):
                return torch.ones_like(prediction), 0.0

        edge = torch.zeros(1, 3, 256, 256)
        edge[..., 128:] = 1
        resized = Resize(512, interpolation=InterpolationMode.BICUBIC, antialias=True)(
            edge
        )
        self.assertLess(resized.min().item(), 0)
        original = resized.clone()
        _, stats = prepare_image_guidance(
            Baseline(),
            resized,
            ResidentConfig(
                Path("fixture"), guidance_mode="confidence", epsilon_var=1e-4
            ),
        )
        torch.testing.assert_close(resized, original, rtol=0, atol=0)
        self.assertEqual(
            stats["input_policy"],
            "upstream padded RGB clipped to [0,1] for confidence only",
        )


if __name__ == "__main__":
    unittest.main()

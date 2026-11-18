"""Optional CPU Torch checks for detached RGB correction adapters."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime absent")
class GuidanceTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_constant_tiny_and_bright_confidence_is_finite_and_detached(self):
        import torch

        from btp.guidance import confidence_maps

        for value in (0.0, 0.5, 1.0):
            image = torch.full((1, 3, 1, 1), value, requires_grad=True)
            maps = confidence_maps(image, epsilon_var=0.01)
            torch.testing.assert_close(maps.weight, torch.full((1, 1, 1, 1), 0.25))
            self.assertFalse(maps.weight.requires_grad)
            self.assertEqual(maps.statistics["weight_mean"], 0.25)
            self.assertTrue(torch.isfinite(maps.ambiguity).all())
        self.assertEqual(maps.ambiguity.item(), 1)

    def test_texture_gets_more_weight_and_uniform_control_matches_mean(self):
        import torch

        from btp.guidance import confidence_maps, uniform_control

        image = torch.zeros(2, 3, 17, 19)
        image[:, :, ::2, ::2] = 1
        maps = confidence_maps(image, epsilon_var=0.01)
        self.assertTrue((maps.weight > 0.25).all())
        self.assertTrue((maps.weight <= 1).all())
        uniform = uniform_control(maps.weight)
        torch.testing.assert_close(uniform.mean((2, 3)), maps.weight.mean((2, 3)))
        self.assertTrue((uniform == uniform[:, :, :1, :1]).all())

    def test_identity_and_preserved_latent_backward(self):
        import torch

        from btp.guidance import weight_rgb_correction

        latent = torch.randn(1, 3, 2, 3, requires_grad=True)
        rgb = latent * 2
        correction = torch.ones_like(rgb)
        result = weight_rgb_correction(correction, torch.ones(1, 1, 2, 3))
        torch.testing.assert_close(result, correction, rtol=0, atol=0)
        rgb.backward(result)
        torch.testing.assert_close(latent.grad, torch.full_like(latent, 2))

    def test_rgb_weight_matches_finite_difference_gradient(self):
        import torch

        from btp.guidance import weight_rgb_correction

        latent = torch.tensor(
            [[[[0.2]], [[0.3]], [[0.4]]]], dtype=torch.float64, requires_grad=True
        )
        rgb = latent.square()
        delta = weight_rgb_correction(
            torch.ones_like(rgb), torch.full((1, 1, 1, 1), 0.5, dtype=torch.float64)
        )
        rgb.backward(delta)
        step = 1e-5
        numeric = (
            ((latent.detach() + step).square() * 0.5)
            - ((latent.detach() - step).square() * 0.5)
        ) / (2 * step)
        torch.testing.assert_close(latent.grad, numeric)

    def test_cap_preserves_direction_and_per_image_norm(self):
        import torch

        from btp.guidance import weight_rgb_correction

        correction = torch.tensor([3.0, 4.0, 0.0]).view(1, 3, 1, 1)
        result = weight_rgb_correction(
            correction, torch.ones(1, 1, 1, 1), max_l2_norm=2
        )
        torch.testing.assert_close(result, correction * 0.4)
        zero = weight_rgb_correction(
            torch.zeros_like(correction), torch.ones(1, 1, 1, 1), max_l2_norm=0
        )
        self.assertTrue(torch.isfinite(zero).all())
        self.assertEqual(zero.abs().sum().item(), 0)

    def test_invalid_input_parameters_and_geometry_rejected(self):
        import torch

        from btp.guidance import confidence_maps, weight_rgb_correction

        image = torch.ones(1, 3, 2, 3)
        for kwargs in (
            {"epsilon_var": 0},
            {"epsilon_var": float("nan")},
            {"epsilon_var": 0.1, "floor": -1},
            {"epsilon_var": 0.1, "window": 2},
        ):
            with self.assertRaises(ValueError):
                confidence_maps(image, **kwargs)
        for bad in (image * float("nan"), image * 2, torch.ones(1, 1, 2, 3)):
            with self.assertRaises(ValueError):
                confidence_maps(bad, epsilon_var=0.01)
        for weight in (
            torch.ones(1, 1, 3, 2),
            torch.ones(1, 3, 2, 3),
            torch.full((1, 1, 2, 3), -1.0),
        ):
            with self.assertRaises(ValueError):
                weight_rgb_correction(image, weight)
        with self.assertRaises(ValueError):
            weight_rgb_correction(image, torch.ones(1, 1, 2, 3), max_l2_norm=-1)

    def test_wrapper_preserves_rgb_schedule_and_baseline_correction(self):
        import torch

        from btp.guidance import RGBGuidanceAdapter

        class Baseline:
            space = "rgb"
            repeat = 2
            t_start = 1000
            t_stop = 0

            def __call__(self, target, prediction, timestep):
                return (target - prediction).detach() * timestep, 3.0

        baseline = Baseline()
        weight = torch.full((1, 1, 2, 3), 0.5)
        adapter = RGBGuidanceAdapter(baseline, weight)
        weight.zero_()  # caller mutation must not alter a registered adapter
        prediction = torch.zeros(1, 3, 2, 3)
        correction, loss = adapter(torch.ones_like(prediction), prediction, 2)
        torch.testing.assert_close(correction, torch.ones_like(prediction))
        self.assertEqual(loss, 3.0)
        self.assertEqual(
            (adapter.space, adapter.repeat, adapter.t_start, adapter.t_stop),
            ("rgb", 2, 1000, 0),
        )
        baseline.space = "latent"
        with self.assertRaises(ValueError):
            adapter(torch.ones_like(prediction), prediction, 2)
        with self.assertRaises(ValueError):
            RGBGuidanceAdapter(baseline, torch.ones(1, 1, 2, 3))

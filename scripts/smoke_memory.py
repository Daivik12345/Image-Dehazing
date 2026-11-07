"""CPU-only gradient smoke check, run in the existing PyTorch inference environment."""

from types import SimpleNamespace

import torch
from torch import nn

from btp.memory import offload_denoisers, prepare_low_memory


def main() -> None:
    """Require identical VAE outputs/latent gradients after inference freezing."""
    torch.manual_seed(42)
    model = SimpleNamespace(
        unet=nn.Linear(4, 4),
        controlnet=nn.Linear(4, 4),
        clip=nn.Linear(4, 4),
        vae=nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4)),
    )
    latent = torch.randn(2, 4, requires_grad=True)
    baseline = model.vae(latent)
    baseline_gradient = torch.autograd.grad(baseline.square().mean(), latent)[0]
    prepare_low_memory(model)
    trial = latent.detach().clone().requires_grad_(True)
    with offload_denoisers(model):
        actual = model.vae(trial)
        actual_gradient = torch.autograd.grad(actual.square().mean(), trial)[0]
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, baseline_gradient, rtol=0, atol=0)
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(not p.requires_grad for p in model.vae.parameters())
    print("CPU smoke passed: identical outputs and latent gradients; weights frozen.")


if __name__ == "__main__":
    main()

"""Reduce inference GPU residency without changing pretrained tensor precision."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def prepare_low_memory(model: Any) -> None:
    """Freeze inference weights and offload CLIP after all prompts are embedded.

    Latent inputs may still require gradients for AccSamp fidelity guidance.
    The caller must not encode additional prompts after this operation.
    """
    for name in ("unet", "controlnet", "vae", "clip"):
        getattr(model, name).requires_grad_(False)
    model.clip.to("cpu")


@contextmanager
def offload_denoisers(model: Any) -> Iterator[None]:
    """Free denoiser GPU residency during VAE guidance, restoring on failure."""
    moved = []
    try:
        for module in (model.unet, model.controlnet):
            device = next(module.parameters()).device
            moved.append((module, device))
            module.to("cpu")
        yield
    finally:
        for module, device in reversed(moved):
            module.to(device)

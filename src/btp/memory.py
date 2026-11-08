"""Reduce inference GPU residency without changing pretrained tensor precision."""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

POLICIES = ("R0", "M0", "M1", "M2-U", "M2-C", "M2-R", "M3")


def policy_modules(policy: str, active: bool) -> tuple[str, ...]:
    """Choose a fixed placement; M3 is M1 without per-guidance cache flushing."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown memory policy: {policy}")
    if policy in {"R0", "M2-R"} or (policy != "M0" and not active):
        return ()
    if policy == "M2-U":
        return ("unet",)
    if policy == "M2-C":
        return ("controlnet",)
    return ("unet", "controlnet")


def prepare_low_memory(model: Any) -> None:
    """Freeze inference weights and offload CLIP after all prompts are embedded.

    Latent inputs may still require gradients for AccSamp fidelity guidance.
    The caller must not encode additional prompts after this operation.
    """
    for name in ("unet", "controlnet", "vae", "clip"):
        getattr(model, name).requires_grad_(False)
    model.clip.to("cpu")


@contextmanager
def offload_denoisers(
    model: Any,
    modules: tuple[str, ...] = ("unet", "controlnet"),
    observe: Any = None,
    phase: Any = None,
) -> Iterator[None]:
    """Free denoiser GPU residency during VAE guidance, restoring on failure."""
    moved = []
    try:
        for name in modules:
            module = getattr(model, name)
            device = next(module.parameters()).device
            moved.append((name, module, device))
            with phase("offload_" + name) if phase else nullcontext():
                module.to("cpu")
            if observe is not None:
                observe(name, module, "to_cpu")
        yield
    finally:
        for name, module, device in reversed(moved):
            with phase("restore_" + name) if phase else nullcontext():
                module.to(device)
            if observe is not None:
                observe(name, module, "to_gpu")

"""Memory-aware loading of unchanged upstream model architectures."""

from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_tensors(torch: Any, path: Path, legacy_metadata: bool = False) -> dict:
    """Read tensor checkpoints with scoped, inert handling of known legacy metadata."""
    if not legacy_metadata:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    import numpy as np

    # SD 2.1 stores this callback class as a metadata key. No Lightning code is
    # imported or run: a passive marker stands in for that unused class.
    marker = type(
        "ModelCheckpoint",
        (),
        {
            "__module__": "pytorch_lightning.callbacks.model_checkpoint",
        },
    )
    allowed = [marker, np.dtype, np.core.multiarray.scalar]
    allowed.extend(
        type(np.dtype(name)) for name in ["float64", "float32", "int64", "int32"]
    )
    with torch.serialization.safe_globals(allowed):
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)


def load_model(
    torch: Any,
    instantiate: Any,
    model_config: Any,
    sd_path: Path,
    control_path: Path,
    mixed: bool,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Instantiate on meta and assign mapped checkpoint tensors without cloning SD.

    In mixed precision, keep the VAE, normalization, timestep embeddings and
    UNet output in float32, while storing the main diffusion/text weights in
    float16. This follows the upstream forward methods' explicit dtype casts.
    Only trusted, zip-format torch checkpoints can use the mapped path.
    """
    if progress:
        progress("Initializing architecture")
    with torch.device("meta"):
        model = instantiate(model_config)
    prefixes = {
        "unet": "model.diffusion_model",
        "vae": "first_stage_model",
        "clip": "cond_stage_model",
    }
    if progress:
        progress("Reading checkpoint metadata")
    base = load_tensors(torch, sd_path, legacy_metadata=True)["state_dict"]

    def assign(module: Any, state: dict, name: str) -> None:
        if progress:
            progress(f"Loading {name} tensors onto {device}")
        norm_prefixes = [
            key
            for key, value in module.named_modules()
            if isinstance(value, torch.nn.GroupNorm)
        ]
        converted = {}
        for key, tensor in state.items():
            keep_float = (
                name == "vae"
                or key.startswith("time_embed.")
                or (name == "unet" and key.startswith("out."))
                or any(key.startswith(prefix + ".") for prefix in norm_prefixes)
            )
            dtype = (
                torch.float16
                if mixed and tensor.is_floating_point() and not keep_float
                else tensor.dtype
            )
            converted[key] = tensor.to(device=device, dtype=dtype)
        module.load_state_dict(converted, strict=True, assign=True)

    for name, prefix in prefixes.items():
        module = getattr(model, name)
        state = {}
        for key in module.state_dict():
            source = f"{prefix}.{key}"
            if source not in base:
                raise ValueError(f"SD checkpoint is incompatible: missing {source}")
            state[key] = base[source]
        assign(module, state, name)
    del base, state
    control = load_tensors(torch, control_path)
    assign(model.controlnet, control, "controlnet")
    del control
    # CLIP's causal mask is non-persistent and is absent from checkpoint state.
    for name, buffer in list(model.named_buffers()):
        if buffer.is_meta:
            if not name.endswith("attn_mask"):
                raise ValueError(f"Uninitialized non-persistent buffer: {name}")
            parent, key = name.rsplit(".", 1)
            mask = torch.full(
                buffer.shape,
                -torch.inf,
                dtype=torch.float16 if mixed else torch.float32,
                device=device,
            ).triu_(1)
            model.get_submodule(parent).register_buffer(key, mask, persistent=False)
    model.unet.dtype = torch.float16 if mixed else torch.float32
    model.controlnet.dtype = torch.float16 if mixed else torch.float32
    model.eval().requires_grad_(False)
    return model

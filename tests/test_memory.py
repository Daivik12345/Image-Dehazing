"""Verify temporary model offloading, including failure cleanup."""

from types import SimpleNamespace

import pytest

from btp.memory import offload_denoisers, prepare_low_memory


class Module:
    def __init__(self):
        self.device = "cuda:0"
        self.frozen = False

    def parameters(self):
        return iter([SimpleNamespace(device=self.device)])

    def to(self, device):
        self.device = device
        return self

    def requires_grad_(self, value):
        self.frozen = not value
        return self


def model():
    return SimpleNamespace(
        unet=Module(), controlnet=Module(), clip=Module(), vae=Module()
    )


def test_offloading_restores_devices_even_on_failure():
    net = model()
    with pytest.raises(RuntimeError, match="guidance failed"):
        with offload_denoisers(net):
            assert net.unet.device == net.controlnet.device == "cpu"
            assert net.vae.device == "cuda:0"
            raise RuntimeError("guidance failed")
    assert net.unet.device == net.controlnet.device == "cuda:0"


def test_prepare_freezes_weights_and_only_offloads_text_encoder():
    net = model()
    prepare_low_memory(net)
    assert net.clip.device == "cpu"
    assert net.unet.device == net.vae.device == "cuda:0"
    assert all(
        getattr(net, name).frozen for name in ("unet", "controlnet", "clip", "vae")
    )

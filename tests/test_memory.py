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


def test_selective_policy_moves_only_requested_modules():
    from btp.memory import policy_modules

    net = model()
    assert policy_modules("M2-U", True) == ("unet",)
    assert policy_modules("M1", False) == ()
    assert policy_modules("M0", False) == ("unet", "controlnet")
    with offload_denoisers(net, modules=("unet",)):
        assert net.unet.device == "cpu"
        assert net.controlnet.device == "cuda:0"
    assert net.unet.device == "cuda:0"
    with pytest.raises(ValueError):
        policy_modules("invalid", True)


def test_independent_cache_policies_preserve_legacy_defaults():
    from btp.memory import resolve_cache_policy, should_empty_cache

    assert resolve_cache_policy("M0", None) == "guidance"
    assert resolve_cache_policy("M3", None) == "phase"
    assert resolve_cache_policy("R0", None) == "never"
    assert resolve_cache_policy("M2-U", "never") == "never"
    for policy in ("guidance", "phase", "never"):
        assert should_empty_cache(policy, "conditioning") == (policy != "never")
        assert should_empty_cache(policy, "guidance", offloaded=True) == (
            policy == "guidance"
        )
        assert not should_empty_cache(policy, "guidance", offloaded=False)
    with pytest.raises(ValueError):
        resolve_cache_policy("M0", "invalid")
    with pytest.raises(ValueError):
        resolve_cache_policy("R0", "guidance")
    with pytest.raises(ValueError):
        should_empty_cache("phase", "invalid")

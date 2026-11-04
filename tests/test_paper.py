"""Model adapter diagnostics that do not require downloading model weights."""

from types import SimpleNamespace

import pytest

from btp.config import RunConfig
from btp.experiment import create_run
from btp.paper import run_paper, select_device
from btp.preview import demo_image


def runtime(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_device_prefers_available_accelerator(cuda, mps, expected):
    assert select_device(runtime(cuda, mps), "auto") == expected


def test_unavailable_device_fails_clearly():
    with pytest.raises(ValueError, match="unavailable"):
        select_device(runtime(), "cuda")


def test_missing_checkpoints_fail_before_model_import(tmp_path):
    config = RunConfig(backend="pretrained", weights_dir=str(tmp_path / "weights"))
    run = create_run(demo_image(64), config, tmp_path / "runs")
    with pytest.raises(FileNotFoundError, match="stage1.pt"):
        run_paper(run)

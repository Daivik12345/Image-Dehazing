"""Behavioral checks for the CPU pipeline and reproducible experiment artifacts."""

import json

import numpy as np
import pytest
from PIL import Image

from btp.config import RunConfig, checkpoint_status
from btp.experiment import create_run, run_preview
from btp.preview import demo_image, estimate_transmission, preview_frames, rgb_array


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 1},
        {"size": 127},
        {"seed": -1},
        {"blend": 1.1},
        {"haze_density": -1},
        {"preview_every": 0},
        {"backend": "fake"},
        {"sampler": "accsamp", "steps": 7},
        {"device": "bogus"},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        RunConfig(**options)


def test_seeded_preview_is_repeatable_and_bounded():
    config = RunConfig(size=128, steps=5, seed=42)
    image = demo_image(128)
    first = list(preview_frames(image, config))
    second = list(preview_frames(image, config))
    assert len(first) == 10
    assert [item[0] for item in first] == ["hazegen"] * 5 + ["diffdehaze"] * 5
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left[3]), np.asarray(right[3]))
        assert left[3].size == image.size
    assert not np.array_equal(np.asarray(first[4][3]), np.asarray(image))
    assert not np.array_equal(np.asarray(first[9][3]), np.asarray(first[4][3]))


def test_no_haze_is_identity():
    image = demo_image(128)
    config = RunConfig(stage="hazegen", size=128, haze_density=0, steps=2)
    result = list(preview_frames(image, config))[-1][3]
    np.testing.assert_allclose(np.asarray(result), np.asarray(image), atol=1)


@pytest.mark.parametrize("value", [0, 255, 128])
def test_constant_input_has_finite_transmission(value):
    image = np.full((32, 32, 3), value / 255, dtype=np.float32)
    transmission, air = estimate_transmission(image)
    assert np.isfinite(transmission).all()
    assert np.isfinite(air).all()
    assert transmission.min() >= 0.12
    assert transmission.max() <= 1


def test_aspect_ratio_and_rgba_conversion():
    result = rgb_array(Image.new("RGBA", (400, 200)), 128)
    assert result.shape == (64, 128, 3)
    with pytest.raises(ValueError):
        rgb_array(Image.new("RGB", (1, 40)), 128)


def test_invalid_array():
    with pytest.raises(ValueError):
        estimate_transmission(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        estimate_transmission(np.full((4, 4, 3), np.nan))


def test_selected_stage_requires_only_its_checkpoint(tmp_path):
    (tmp_path / "stage2.pt").touch()
    (tmp_path / "v2-1_512-ema-pruned.ckpt").touch()
    config = RunConfig(stage="diffdehaze", weights_dir=str(tmp_path))
    assert all(checkpoint_status(config).values())
    assert "stage1.pt" not in checkpoint_status(config)


def test_run_records_input_provenance_and_real_outputs(tmp_path):
    image = demo_image(128)
    config = RunConfig(size=128, steps=2)
    run = create_run(image, config, tmp_path)
    run_preview(image, config, run)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config"]["seed"] == 42
    assert len(manifest["input_sha256"]) == 64
    assert manifest["method"] == "physical CPU baseline"
    assert json.loads((run / "status.json").read_text())["state"] == "completed"
    assert (run / "hazegen.png").is_file()
    assert (run / "diffdehaze.png").is_file()
    assert len(list(run.glob("hazegen-*.png"))) == 2

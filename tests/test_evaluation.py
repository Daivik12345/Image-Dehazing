"""Dataset-contract and numeric regression tests for paired evaluation."""

import json

import numpy as np
import pytest
from PIL import Image

from btp.evaluation import load_manifest, paired_metrics, sha256


def fixture_manifest(tmp_path):
    Image.new("RGB", (16, 16), (100, 120, 140)).save(tmp_path / "hazy.png")
    Image.new("RGB", (16, 16), (90, 110, 130)).save(tmp_path / "clean.png")
    scene = {
        "scene_id": "a",
        "split": "development",
        "hazy": "hazy.png",
        "clean": "clean.png",
        "hazy_sha256": sha256(tmp_path / "hazy.png"),
        "clean_sha256": sha256(tmp_path / "clean.png"),
        "geometry_tier": "native",
    }
    doc = {
        "schema_version": 1,
        "dataset": "fixture",
        "release": "v1",
        "scenes": [scene],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doc))
    return p, doc


def test_manifest_validates_all_files_and_rejects_duplicate_ids(tmp_path):
    p, doc = fixture_manifest(tmp_path)
    assert load_manifest(p)["scenes"][0]["scene_id"] == "a"
    doc["scenes"].append(doc["scenes"][0])
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(p)


def test_manifest_rejects_hash_mismatch_and_shape_mismatch(tmp_path):
    p, doc = fixture_manifest(tmp_path)
    Image.new("RGB", (12, 16)).save(tmp_path / "clean.png")
    with pytest.raises(ValueError, match="hash"):
        load_manifest(p)
    doc["scenes"][0]["clean_sha256"] = sha256(tmp_path / "clean.png")
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="dimensions"):
        load_manifest(p)


def test_manifest_rejects_path_escape(tmp_path):
    p, doc = fixture_manifest(tmp_path)
    doc["scenes"][0]["hazy"] = "../outside.png"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="outside"):
        load_manifest(p)


def test_metrics_identity_known_error_and_invalid_inputs():
    clean = np.full((16, 16, 3), 0.5)
    exact = paired_metrics(clean, clean)
    assert exact["psnr_db"] is None and exact["identical"]
    assert exact["ssim"] == pytest.approx(1)
    noisy = paired_metrics(clean, clean + 0.1)
    assert noisy["psnr_db"] == pytest.approx(20)
    assert 0 < noisy["ssim"] < 1
    with pytest.raises(ValueError):
        paired_metrics(clean, np.zeros((12, 12, 3)))
    with pytest.raises(ValueError):
        paired_metrics(clean, clean * float("nan"))
    with pytest.raises(ValueError):
        paired_metrics(clean, clean + 1)


def test_ssim_matches_pinned_scikit_image_golden():
    # Independently checked against skimage==0.24.0 on 2026-11-06.
    rng = np.random.default_rng(42)
    clean = rng.random((31, 37, 3))
    pred = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
    assert paired_metrics(clean, pred)["ssim"] == pytest.approx(
        0.9850094792291882, abs=1e-12
    )


def test_manifest_rejects_source_leakage_between_splits(tmp_path):
    p, doc = fixture_manifest(tmp_path)
    doc["scenes"].append({**doc["scenes"][0], "scene_id": "second", "split": "test"})
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="crosses splits"):
        load_manifest(p)


@pytest.mark.parametrize(
    "crop", [[-2, 100, 16, 16], [0, 0, 21, 16], [0, 0, 12, 12], [0.5, 0, 16, 16]]
)
def test_manifest_rejects_invalid_crop_geometry(tmp_path, crop):
    p, doc = fixture_manifest(tmp_path)
    doc["scenes"][0].update(source_size=[20, 20], crop_xywh=crop)
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="crop"):
        load_manifest(p)

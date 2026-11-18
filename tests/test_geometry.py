"""Native geometry inherits locked evaluation membership without resizing."""

import io
import json
import zipfile

import pytest
from PIL import Image

from btp.datasets import prepare_nh_haze
from btp.evaluation import load_manifest
from btp.geometry import prepare_geometry


def fixture(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i, size in [("01", (20, 18)), ("02", (30, 24))]:
            for role in ["hazy", "GT"]:
                buffer = io.BytesIO()
                Image.new("RGB", size, (int(i), 5, 9)).save(buffer, format="PNG")
                z.writestr(f"NH-HAZE/{i}_{role}.png", buffer.getvalue())
    frozen = prepare_nh_haze(
        archive,
        tmp_path / "frozen",
        crop_size=16,
        development_count=1,
        expected_pairs=2,
    )
    return archive, frozen


def test_geometry_preserves_split_and_records_ineligible(tmp_path):
    archive, frozen = fixture(tmp_path)
    report = prepare_geometry(archive, frozen, tmp_path / "views", crop_sizes=(16, 24))
    assert report["eligibility"]["center-24"] == {"development": 0, "test": 1}
    original = {r["scene_id"]: r["split"] for r in load_manifest(frozen)["scenes"]}
    native = load_manifest(tmp_path / "views/native/manifest.json")
    assert {r["scene_id"]: r["split"] for r in native["scenes"]} == original
    assert native["scenes"][0]["crop_xywh"] == [0, 0, 20, 18]
    crop = load_manifest(tmp_path / "views/center-24/manifest.json")
    assert crop["scenes"][0]["crop_xywh"] == [3, 0, 24, 24]


def test_geometry_rejects_changed_archive_identity(tmp_path):
    archive, frozen = fixture(tmp_path)
    doc = json.loads(frozen.read_text())
    doc["archive_sha256"] = "0" * 64
    frozen.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="archive"):
        prepare_geometry(archive, frozen, tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_geometry_accepts_frozen_metadata_without_old_crop_files(tmp_path):
    archive, frozen = fixture(tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(frozen.read_bytes())
    report = prepare_geometry(archive, metadata, tmp_path / "views", crop_sizes=())
    assert sum(report["eligibility"]["native"].values()) == 2

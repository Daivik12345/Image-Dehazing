"""Fixed paired crop preparation and missing-pair rejection."""

import io
import zipfile

import pytest
from PIL import Image

from btp.datasets import prepare_nh_haze
from btp.evaluation import load_manifest


def test_nh_haze_freezes_pair_geometry_and_split(tmp_path):
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for scene in ("01", "02", "03"):
            for role in ("hazy", "GT"):
                data = io.BytesIO()
                Image.new("RGB", (20, 18), (10 + int(scene), 20, 30)).save(
                    data, format="PNG"
                )
                z.writestr(f"NH-HAZE/{scene}_{role}.png", data.getvalue())
    path = prepare_nh_haze(
        archive,
        tmp_path / "prepared",
        crop_size=16,
        development_count=1,
        expected_pairs=3,
    )
    doc = load_manifest(path)
    assert len(doc["scenes"]) == 3
    assert sum(s["split"] == "development" for s in doc["scenes"]) == 1
    assert doc["scenes"][0]["crop_xywh"] == [2, 1, 16, 16]
    second = prepare_nh_haze(
        archive,
        tmp_path / "second",
        crop_size=16,
        development_count=1,
        expected_pairs=3,
    )
    assert path.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_nh_haze(
            archive,
            tmp_path / "prepared",
            crop_size=16,
            development_count=1,
            expected_pairs=3,
        )


def test_nh_haze_rejects_missing_pair(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("NH-HAZE/01_hazy.png", b"not decoded")
    with pytest.raises(ValueError, match="pairs"):
        prepare_nh_haze(
            archive, tmp_path / "out", expected_pairs=1, development_count=1
        )

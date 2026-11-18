"""Bounded nested OTS import, complete inventory and deterministic pairing."""

import io
import json
import zipfile

import pytest
from PIL import Image

from btp.ots import inspect_ots, prepare_ots_pilot


def picture(number, size=(256, 258)):
    out = io.BytesIO()
    Image.new("RGB", size, (number * 10, 30, 50)).save(out, format="PNG")
    return out.getvalue()


def archive(tmp_path, *, bad=None):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as outer:
        for part in ["clear", "part1", "part2", "part3", "part4"]:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as inner:
                numbers = range(1, 9) if part == "clear" else range(int(part[-1]), 9, 4)
                for number in numbers:
                    name = (
                        f"clear/{number:04d}.png"
                        if part == "clear"
                        else f"{part}/{number:04d}_0.8_0.2.png"
                    )
                    if bad == "unsafe" and number == 1:
                        name = "../unsafe.png"
                    size = (255, 256) if bad == "small" and number == 1 else (256, 258)
                    if bad == "geometry" and part == "part1" and number == 1:
                        size = (256, 259)
                    inner.writestr(name, picture(number, size))
                    if bad == "duplicate" and part == "part1" and number == 1:
                        inner.writestr("another/0001_0.8_0.2.png", picture(number))
            folder = "clear" if part == "clear" else "haze"
            outer.writestr(f"OTS_BETA/{folder}/{part}.zip", buffer.getvalue())
        outer.writestr("OTS_BETA/haze/part1/0001_0.8_0.2.png", picture(1))
    return path


def test_full_inventory_and_deterministic_import(tmp_path):
    source = archive(tmp_path)
    inventory = inspect_ots(source, scratch_dir=tmp_path)
    assert len(inventory["clean"]) == len(inventory["hazy"]) == 8
    assert len(inventory["nested_archives"]) == 5
    paths = [
        prepare_ots_pilot(source, tmp_path / name, count=8, scratch_dir=tmp_path)
        for name in ["first", "second"]
    ]
    docs = [json.loads(path.read_text()) for path in paths]
    assert docs[0] == docs[1]
    assert {r["split"] for r in docs[0]["scenes"]} == {"train", "validation"}
    assert len({r["source_scene_id"] for r in docs[0]["scenes"]}) == 8
    assert {r["source_hazy_archive"] for r in docs[0]["scenes"]} == {
        f"OTS_BETA/haze/part{i}.zip" for i in range(1, 5)
    }
    row = docs[0]["scenes"][0]
    with Image.open(paths[0].parent / row["hazy"]) as image:
        assert image.mode == "RGB" and image.size == (256, 258)
    assert (
        row["registration_method"]
        == "synthetic filename identity and matching geometry only"
    )
    assert docs[0]["manual_review"] == "pending"
    assert docs[0]["license_status"] == "unresolved"
    assert not list(tmp_path.glob("btp-ots-nested-*"))


@pytest.mark.parametrize("bad", ["unsafe", "duplicate", "small", "geometry"])
def test_invalid_archive_fails_without_completed_output(tmp_path, bad):
    source = archive(tmp_path, bad=bad)
    with pytest.raises(ValueError):
        prepare_ots_pilot(source, tmp_path / "output", count=8, scratch_dir=tmp_path)
    assert not (tmp_path / "output").exists()


def test_insufficient_inventory_and_disk_budget_rejected(tmp_path):
    source = archive(tmp_path)
    with pytest.raises(ValueError, match="paired"):
        prepare_ots_pilot(source, tmp_path / "output", count=9, scratch_dir=tmp_path)
    with pytest.raises(ValueError, match="budget"):
        inspect_ots(source, scratch_dir=tmp_path, max_nested_bytes=1)
    with (
        zipfile.ZipFile(source, "a") as outer,
        pytest.warns(UserWarning, match="Duplicate name"),
    ):
        outer.writestr("OTS_BETA/haze/part1.zip", b"duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_ots(source, scratch_dir=tmp_path)


def test_dimension_eligibility_is_recorded_before_selection(tmp_path):
    source = archive(tmp_path, bad="small")
    result = prepare_ots_pilot(
        source, tmp_path / "eligible", count=7, scratch_dir=tmp_path
    )
    doc = json.loads(result.read_text())
    assert len(doc["inventory_summary"]["excluded_clean_sources"]) == 1
    assert len(doc["scenes"]) == 7

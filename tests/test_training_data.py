"""Training identity, integrity and paired augmentation boundaries."""

import numpy as np
import pytest
from PIL import Image

from btp.evaluation import sha256
from btp.training_data import audit_training_manifest, paired_crop, split_by_scene


def document(tmp_path):
    rows = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (20, 18), (i, 5, 9)).save(path)
        rows.append(
            dict(
                pair_id=str(i),
                source_scene_id=f"s{i}",
                split="train",
                hazy=path.name,
                clean=path.name,
                hazy_sha256=sha256(path),
                clean_sha256=sha256(path),
                source_size=[20, 18],
                registration_status="verified",
                augmentation_parent=None,
            )
        )
    return dict(
        schema_version=1,
        dataset="fixture",
        release="v1",
        source_archive="fixture.zip",
        archive_sha256="a" * 64,
        license="test fixture only",
        scenes=rows,
    )


def test_audit_checks_all_files_and_identity(tmp_path):
    doc = document(tmp_path)
    assert audit_training_manifest(doc, tmp_path)["counts"] == {"train": 3}
    doc["scenes"][1]["split"] = "validation"
    doc["scenes"][1]["aliases"] = ["s0"]
    with pytest.raises(ValueError, match="crosses splits"):
        audit_training_manifest(doc, tmp_path)


@pytest.mark.parametrize(
    "mutation", ["hash", "missing", "channels", "dimensions", "id", "parent", "license"]
)
def test_audit_rejects_invalid_data(tmp_path, mutation):
    doc = document(tmp_path)
    row = doc["scenes"][1]
    if mutation == "hash":
        row["clean_sha256"] = "0" * 64
    elif mutation == "missing":
        row["clean"] = "absent.png"
    elif mutation in {"channels", "dimensions"}:
        Image.new("L" if mutation == "channels" else "RGB", (15, 15)).save(
            tmp_path / "bad.png"
        )
        row["clean"] = "bad.png"
        row["clean_sha256"] = sha256(tmp_path / "bad.png")
    elif mutation == "id":
        row["pair_id"] = "0"
    elif mutation == "parent":
        row["augmentation_parent"] = "0"
        row["split"] = "validation"
    else:
        doc["license"] = ""
    with pytest.raises(ValueError):
        audit_training_manifest(doc, tmp_path)


def test_evaluation_hash_overlap_rejected(tmp_path):
    doc = document(tmp_path)
    evaluation = [
        {
            "scene_id": "external",
            "source_clean_sha256": doc["scenes"][0]["clean_sha256"],
        }
    ]
    with pytest.raises(ValueError, match="evaluation"):
        audit_training_manifest(doc, tmp_path, evaluation_scenes=evaluation)


def test_split_groups_aliases_and_is_order_independent():
    rows = [dict(pair_id=str(i), source_scene_id=f"s{i}") for i in range(10)]
    rows.append(
        dict(pair_id="crop", source_scene_id="renamed", augmentation_parent="0")
    )
    result = split_by_scene(rows)
    reversed_result = split_by_scene(list(reversed(rows)))
    assert {r["pair_id"]: r["split"] for r in result} == {
        r["pair_id"]: r["split"] for r in reversed_result
    }
    assert result[0]["split"] == result[-1]["split"]
    assert (
        len({r["source_scene_id"] for r in result[:10] if r["split"] == "validation"})
        == 1
    )
    assert all("split" not in row for row in rows)


def test_crop_is_aligned_deterministic_and_rejects_undersize():
    pixels = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    images = [Image.fromarray(pixels)] * 3
    first = paired_crop(images, size=16, seed=42, horizontal_flip=True)
    second = paired_crop(images, size=16, seed=42, horizontal_flip=True)
    assert all(np.array_equal(first[0], item) for item in first + second)
    with pytest.raises(ValueError):
        paired_crop(images, size=41, seed=42)


def test_audit_rejects_traversal_and_teacher_shape(tmp_path):
    doc = document(tmp_path)
    doc["scenes"][0]["hazy"] = "../outside.png"
    with pytest.raises(ValueError, match="outside root"):
        audit_training_manifest(doc, tmp_path)
    doc = document(tmp_path)
    Image.new("RGB", (18, 18)).save(tmp_path / "teacher.png")
    doc["scenes"][0].update(
        teacher="teacher.png", teacher_sha256=sha256(tmp_path / "teacher.png")
    )
    with pytest.raises(ValueError, match="dimensions"):
        audit_training_manifest(doc, tmp_path)


def test_audit_catches_transitive_alias_chain_and_duplicate_clean_hash(tmp_path):
    doc = document(tmp_path)
    doc["scenes"][0]["aliases"] = ["middle"]
    doc["scenes"][1]["aliases"] = ["middle", "last"]
    doc["scenes"][2]["aliases"] = ["last"]
    doc["scenes"][2]["split"] = "validation"
    with pytest.raises(ValueError, match="crosses splits"):
        audit_training_manifest(doc, tmp_path)
    doc = document(tmp_path)
    doc["scenes"][1].update(
        clean=doc["scenes"][0]["clean"],
        clean_sha256=doc["scenes"][0]["clean_sha256"],
        split="validation",
    )
    with pytest.raises(ValueError, match="crosses splits"):
        audit_training_manifest(doc, tmp_path)

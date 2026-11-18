"""Explicit training-corpus integrity, scene grouping and paired crops.

Exact identities/hashes detect known overlap; this is not a near-duplicate
or registration detector. Registration and license statements require review.
"""

import hashlib
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from btp.evaluation import confined_path, sha256


def _identities(row: dict[str, Any]) -> set[str]:
    identities = {"id:" + str(row["source_scene_id"])}
    for alias in row.get("aliases", []):
        identities.add("id:" + str(alias))
    identities.add("pair:" + str(row["pair_id"]))
    if row.get("augmentation_parent"):
        identities.add("pair:" + str(row["augmentation_parent"]))
    for key in (
        "hazy_sha256",
        "clean_sha256",
        "source_hazy_sha256",
        "source_clean_sha256",
    ):
        if row.get(key):
            identities.add("hash:" + row[key])
    return identities


def _groups(rows: Sequence[dict[str, Any]]) -> list[set[int]]:
    groups: list[tuple[set[str], set[int]]] = []
    for i, row in enumerate(rows):
        identities, members = _identities(row), {i}
        remaining = []
        for keys, indices in groups:
            if identities & keys:
                identities |= keys
                members |= indices
            else:
                remaining.append((keys, indices))
        # Existing groups are disjoint, so merging cannot expose another link.
        groups = remaining + [(identities, members)]
    return [members for _, members in groups]


def split_by_scene(
    rows: Sequence[dict[str, Any]], *, seed: int = 42, validation_fraction: float = 0.1
) -> list[dict[str, Any]]:
    """Return copies split by identity groups before any augmentation.

    At least one group is retained in each split; official assignments should
    instead be supplied unchanged to the audit. No global random state changes.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups = _groups(rows)
    if len(groups) < 2:
        raise ValueError("Need at least two independent source scenes")
    groups.sort(
        key=lambda g: hashlib.sha256(
            f"{seed}:{min(rows[i]['source_scene_id'] for i in g)}".encode()
        ).hexdigest()
    )
    count = min(len(groups) - 1, max(1, int(len(groups) * validation_fraction)))
    validation = set().union(*groups[:count])
    return [
        dict(row, split="validation" if i in validation else "train")
        for i, row in enumerate(rows)
    ]


def audit_training_manifest(
    document: dict[str, Any],
    root: Path,
    *,
    evaluation_scenes: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Fail closed on missing provenance, invalid pairs or known split overlap.

    Supply *all* evaluation scenes to check against evaluation identity/hashes.
    A successful audit does not establish licensing rights or image registration.
    """
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported training manifest schema")
    for field in ("dataset", "release", "source_archive", "license"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"Missing {field}")
    archive_hash = document.get("archive_sha256", "")
    if len(archive_hash) != 64 or any(
        c not in "0123456789abcdef" for c in archive_hash
    ):
        raise ValueError("Invalid archive hash")
    rows = document.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Training corpus must contain scenes")
    ids: set[str] = set()
    for row in rows:
        for field in ("pair_id", "source_scene_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Invalid {field}")
        if row["pair_id"] in ids:
            raise ValueError("Repeated pair ID")
        ids.add(row["pair_id"])
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("Training corpus split must be train or validation")
        if row.get("registration_status") != "verified":
            raise ValueError("Pair registration must be verified")
        sizes = []
        for role in ("hazy", "clean", "teacher"):
            if role == "teacher" and role not in row:
                continue
            path = confined_path(root, row.get(role))
            if not path.is_file() or sha256(path) != row.get(role + "_sha256"):
                raise ValueError("Missing file or hash mismatch")
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError("Expected RGB image")
                image.load()
                sizes.append(image.size)
        if len(set(sizes)) != 1:
            raise ValueError("Paired dimensions differ")
        source_size = row.get("source_size")
        crop = row.get("crop_xywh", [0, 0, *sizes[0]])
        if (
            not isinstance(source_size, list)
            or len(source_size) != 2
            or not isinstance(crop, list)
            or len(crop) != 4
            or any(type(v) is not int for v in source_size + crop)
        ):
            raise ValueError("Invalid source/crop dimensions")
        x, y, w, h = crop
        if (
            min(x, y) < 0
            or min(w, h, *source_size) <= 0
            or (w, h) != sizes[0]
            or x + w > source_size[0]
            or y + h > source_size[1]
        ):
            raise ValueError("Invalid crop geometry")
    for group in _groups(rows):
        if len({rows[i]["split"] for i in group}) != 1:
            raise ValueError(
                "Source scene crosses splits through alias, parent or hash"
            )
    evaluation_keys: set[str] = set()
    for row in evaluation_scenes:
        evaluation_keys |= _identities(
            dict(
                row,
                source_scene_id=row.get("source_scene_id", row.get("scene_id")),
                pair_id=row.get("pair_id", row.get("scene_id")),
            )
        )
    if any(_identities(row) & evaluation_keys for row in rows):
        raise ValueError("Training corpus overlaps evaluation scenes")
    return {
        "counts": dict(Counter(row["split"] for row in rows)),
        "source_groups": len(_groups(rows)),
        "evaluation_overlap_checked": bool(evaluation_scenes),
        "near_duplicate_review": "not performed",
        "license": document["license"],
    }


def paired_crop(
    images: Sequence[Image.Image],
    *,
    size: int = 256,
    seed: int,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> list[Image.Image]:
    """Apply one seeded RGB crop and optional shared flips to every target."""
    if not images or type(size) is not int or size <= 0:
        raise ValueError("Need images and a positive integer crop size")
    if any(image.mode != "RGB" or image.size != images[0].size for image in images):
        raise ValueError("Expected matching RGB pairs")
    width, height = images[0].size
    if min(width, height) < size:
        raise ValueError("Images smaller than crop; upsampling is forbidden")
    rng = random.Random(seed)
    x, y = rng.randrange(width - size + 1), rng.randrange(height - size + 1)
    flip_x = horizontal_flip and rng.random() < 0.5
    flip_y = vertical_flip and rng.random() < 0.5
    result = []
    for image in images:
        crop = image.crop((x, y, x + size, y + size))
        if flip_x:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_y:
            crop = crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        result.append(crop)
    return result

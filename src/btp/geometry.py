"""Derive native and larger paired views from frozen NH-HAZE membership."""

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from btp.evaluation import load_manifest, sha256
from btp.experiment import write_json


def prepare_geometry(
    archive: Path,
    frozen_manifest: Path,
    output: Path,
    *,
    crop_sizes: tuple[int, ...] = (512, 768, 1024),
) -> dict[str, Any]:
    """Write eligible views with unchanged source IDs/splits and no resizing.

    The original manifest and archive hashes are validated before output exists.
    Empty tiers appear in the eligibility report without an invalid empty manifest.
    """
    if any(type(size) is not int or size < 11 for size in crop_sizes) or len(
        set(crop_sizes)
    ) != len(crop_sizes):
        raise ValueError("Crop sizes must be unique integers >= 11")
    frozen = json.loads(frozen_manifest.read_text())
    if frozen.get("schema_version") != 1 or not frozen.get("release"):
        raise ValueError("Invalid frozen manifest schema")
    rows = frozen.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen manifest must contain scenes")
    seen = set()
    source_splits = {}
    for row in rows:
        scene = row.get("scene_id", "")
        if (
            not isinstance(scene, str)
            or not re.fullmatch(r"nh-haze-\d+", scene)
            or scene in seen
        ):
            raise ValueError("Invalid or repeated source scene ID")
        seen.add(scene)
        if row.get("split") not in {"development", "test"}:
            raise ValueError("Invalid frozen evaluation split")
        for role in ("hazy", "clean"):
            digest = row.get("source_" + role + "_sha256", "")
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("Invalid source hash")
            if source_splits.setdefault(digest, row["split"]) != row["split"]:
                raise ValueError("Source scene crosses splits")
    if frozen.get("dataset") != "NH-HAZE" or sha256(archive) != frozen.get(
        "archive_sha256"
    ):
        raise ValueError("NH-HAZE archive identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    tiers: dict[str, list[dict[str, Any]]] = {"native": []}
    tiers.update({f"center-{size}": [] for size in crop_sizes})
    with zipfile.ZipFile(archive) as z:
        for original in frozen["scenes"]:
            scene = original["scene_id"]
            number = scene.removeprefix("nh-haze-")
            images = {}
            for role, archive_role in [("hazy", "hazy"), ("clean", "GT")]:
                member = f"NH-HAZE/{number}_{archive_role}.png"
                if z.namelist().count(member) != 1:
                    raise ValueError("Missing or duplicate source archive member")
                data = z.read(member)
                if (
                    hashlib.sha256(data).hexdigest()
                    != original["source_" + role + "_sha256"]
                ):
                    raise ValueError("Source image hash mismatch")
                with Image.open(io.BytesIO(data)) as image:
                    if (
                        image.mode != "RGB"
                        or list(image.size) != original["source_size"]
                    ):
                        raise ValueError("Source RGB geometry mismatch")
                    images[role] = image.copy()
            width, height = images["hazy"].size
            for tier, rows in tiers.items():
                size = None if tier == "native" else int(tier.split("-")[1])
                if size is not None and min(width, height) < size:
                    continue
                w, h = (width, height) if size is None else (size, size)
                x, y = (width - w) // 2, (height - h) // 2
                row = dict(original, geometry_tier=tier, crop_xywh=[x, y, w, h])
                for role, image in images.items():
                    relative = f"{role}/{number}.png"
                    target = output / tier / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((x, y, x + w, y + h)).save(target)
                    row[role] = relative
                    row[role + "_sha256"] = sha256(target)
                rows.append(row)
    report: dict[str, Any] = {
        "schema_version": 1,
        "frozen_manifest_sha256": sha256(frozen_manifest),
        "archive_sha256": frozen["archive_sha256"],
        "eligibility": {},
    }
    for tier, rows in tiers.items():
        counts = Counter(row["split"] for row in rows)
        report["eligibility"][tier] = {
            split: counts[split]
            for split in sorted({r["split"] for r in frozen["scenes"]})
        }
        if rows:
            path = output / tier / "manifest.json"
            write_json(
                path,
                dict(
                    frozen, scenes=rows, parent_manifest_sha256=sha256(frozen_manifest)
                ),
            )
            load_manifest(path)
    write_json(output / "eligibility.json", report)
    return report

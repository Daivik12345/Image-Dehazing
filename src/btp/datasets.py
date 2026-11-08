"""Prepare traceable NH-HAZE center crops without changing paired geometry."""

import argparse
import hashlib
import io
import re
import zipfile
from pathlib import Path

from PIL import Image

from btp.evaluation import load_manifest, sha256
from btp.experiment import write_json

SOURCE_URL = "https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/"


def prepare_nh_haze(
    archive: Path,
    output: Path,
    crop_size: int = 512,
    development_count: int = 10,
    seed: int = 42,
    expected_pairs: int = 55,
) -> Path:
    """Freeze an internal split by source ID, then apply identical center crops.

    This is an internal diagnostic split, not the official NTIRE challenge split.
    No resizing, training, or output-based selection occurs.
    """
    if crop_size < 11 or not 0 < development_count < expected_pairs:
        raise ValueError("Invalid crop size or split sizes for expected pairs")
    with zipfile.ZipFile(archive) as z:
        pairs = {}
        for name in z.namelist():
            match = re.fullmatch(r"NH-HAZE/(\d+)_(hazy|GT)\.png", name)
            if match:
                scene, role = match.groups()
                if role in pairs.setdefault(scene, {}):
                    raise ValueError("Duplicate archive pair member")
                pairs[scene][role] = name
        if len(pairs) != expected_pairs or any(
            set(p) != {"hazy", "GT"} for p in pairs.values()
        ):
            raise ValueError("Unexpected or incomplete NH-HAZE pairs")
        ordered = sorted(
            pairs, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
        )
        dev = set(ordered[:development_count])
        output.mkdir(parents=True, exist_ok=False)
        scenes = []
        for scene in sorted(pairs):
            row = {
                "scene_id": "nh-haze-" + scene,
                "split": "development" if scene in dev else "test",
                "geometry_tier": f"center-{crop_size}",
            }
            sizes = []
            for role, archive_role in [("hazy", "hazy"), ("clean", "GT")]:
                data = z.read(pairs[scene][archive_role])
                row["source_" + role + "_sha256"] = hashlib.sha256(data).hexdigest()
                with Image.open(io.BytesIO(data)) as image:
                    if image.mode != "RGB" or min(image.size) < crop_size:
                        raise ValueError(
                            "Source image must be RGB and large enough for crop"
                        )
                    sizes.append(image.size)
                    left, top = (
                        (image.width - crop_size) // 2,
                        (image.height - crop_size) // 2,
                    )
                    crop = image.crop((left, top, left + crop_size, top + crop_size))
                    relative = f"{role}/{scene}.png"
                    target = output / relative
                    target.parent.mkdir(exist_ok=True)
                    crop.save(target)
                    row[role] = relative
                    row[role + "_sha256"] = sha256(target)
            if sizes[0] != sizes[1]:
                raise ValueError("Paired source dimensions differ")
            row["source_size"] = list(sizes[0])
            row["crop_xywh"] = [left, top, crop_size, crop_size]
            scenes.append(row)
    manifest = output / "nh_haze_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "dataset": "NH-HAZE",
            "release": "NTIRE2020-55-pairs",
            "source_url": SOURCE_URL,
            "archive_sha256": sha256(archive),
            "split_protocol": {
                "name": "BTP internal diagnostic split, NOT official NTIRE split",
                "algorithm": "SHA256 of seed:source_id sorted ascending; first N development, remainder held out",
                "seed": seed,
                "development_count": development_count,
            },
            "scenes": scenes,
        },
    )
    load_manifest(manifest)
    (output / "README.md").write_text(f"""# BTP NH-HAZE evaluation views

Source: {SOURCE_URL}
Archive SHA256: {sha256(archive)}

{len(scenes)} paired RGB scenes, center {crop_size} x {crop_size} crops with no resizing. The manifest records original and cropped hashes, source dimensions and crop coordinates.

Internal split: {development_count} development scenes, {len(scenes) - development_count} held out. Seed {seed}. This is NOT the official NTIRE challenge split. Center-crop scores must not be presented as native-resolution or paper benchmark results. Pretrained model training-set overlap is not established; these images do not prove generalization.

The official source grants use and publication with citation. Original rights and attribution remain applicable; this preparation does not assign a new open license.

Citation: C. O. Ancuti, C. Ancuti, R. Timofte, “NH-HAZE: An Image Dehazing Benchmark with Non-Homogeneous Hazy and Haze-Free Images”, IEEE CVPR NTIRE Workshop, 2020.
""")
    return manifest


def main() -> None:
    """Prepare the official archive with the frozen default diagnostic protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_nh_haze(args.archive, args.output))


if __name__ == "__main__":
    main()

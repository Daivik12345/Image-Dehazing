"""Deterministic, bounded OTS pilot preparation from a user-supplied nested ZIP.

All four haze parts are inventoried; already-extracted outer images are redundant
and never form the sampling frame. No acquisition or publication is performed.
"""

import argparse
import hashlib
import io
import json
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from btp.evaluation import sha256
from btp.training_data import audit_training_manifest, split_by_scene

NESTED = (
    "OTS_BETA/clear/clear.zip",
    *(f"OTS_BETA/haze/part{i}.zip" for i in range(1, 5)),
)
DEFAULT_NESTED_BYTES = 8 * 1024**3
MAX_IMAGE_BYTES = 64 * 1024**2


def _members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
            or ":" in info.filename
            or stat.S_ISLNK(info.external_attr >> 16)
        ):
            raise ValueError(f"Unsafe archive member: {info.filename}")
        if info.filename in entries:
            raise ValueError(f"Duplicate archive member: {info.filename}")
        entries[info.filename] = info
    return entries


@contextmanager
def _nested(
    outer: zipfile.ZipFile,
    member: str,
    scratch_dir: Path,
    max_nested_bytes: int,
) -> Iterator[tuple[zipfile.ZipFile, str]]:
    info = outer.getinfo(member)
    if info.file_size > max_nested_bytes:
        raise ValueError("Nested archive exceeds scratch byte budget")
    if shutil.disk_usage(scratch_dir).free < info.file_size + 64 * 1024**2:
        raise ValueError("Insufficient scratch disk for one nested archive")
    with tempfile.TemporaryDirectory(
        prefix="btp-ots-nested-", dir=scratch_dir
    ) as folder:
        path = Path(folder) / "nested.zip"
        digest = hashlib.sha256()
        total = 0
        with outer.open(info) as source, path.open("wb") as target:
            while block := source.read(1024**2):
                total += len(block)
                if total > max_nested_bytes:
                    raise ValueError("Nested archive exceeds scratch byte budget")
                digest.update(block)
                target.write(block)
        if total != info.file_size:
            raise ValueError("Nested archive length mismatch")
        with zipfile.ZipFile(path) as inner:
            yield inner, digest.hexdigest()


def inspect_ots(
    archive: Path,
    *,
    scratch_dir: Path = Path("/private/tmp"),
    max_nested_bytes: int = DEFAULT_NESTED_BYTES,
) -> dict[str, Any]:
    """Inventory every nested part using one bounded temporary ZIP at a time.

    Returns clean IDs, all available named variants and nested archive hashes.
    Reading ZIP directory entries does not establish image quality/registration.
    """
    if type(max_nested_bytes) is not int or max_nested_bytes <= 0:
        raise ValueError("Scratch byte budget must be a positive integer")
    scratch_dir = Path(scratch_dir)
    if not scratch_dir.is_dir():
        raise ValueError("Scratch directory must already exist")
    clean: dict[str, dict[str, str]] = {}
    hazy: dict[str, list[dict[str, str]]] = {}
    hashes = {}
    seen_variants = set()
    with zipfile.ZipFile(archive) as outer:
        entries = _members(outer)
        if any(member not in entries for member in NESTED):
            raise ValueError(
                "Required clear archive and all four haze parts must exist"
            )
        for member in NESTED:
            with _nested(outer, member, scratch_dir, max_nested_bytes) as (
                inner,
                digest,
            ):
                hashes[member] = digest
                for name, info in _members(inner).items():
                    if (
                        info.is_dir()
                        or "__MACOSX" in PurePosixPath(name).parts
                        or PurePosixPath(name).name.startswith("._")
                    ):
                        continue
                    suffix = PurePosixPath(name).suffix.lower()
                    if suffix not in {".jpg", ".jpeg", ".png"}:
                        continue
                    stem = PurePosixPath(name).stem
                    pattern = r"(\d+)" if member == NESTED[0] else r"(\d+)_(.+)"
                    match = re.fullmatch(pattern, stem)
                    if match is None:
                        raise ValueError(
                            f"Unrecognized image pairing name: {member}:{name}"
                        )
                    scene = str(int(match[1]))
                    reference = {"archive": member, "member": name}
                    if member == NESTED[0]:
                        if scene in clean:
                            raise ValueError(
                                f"Duplicate clean source identity: {scene}"
                            )
                        if info.file_size > MAX_IMAGE_BYTES:
                            raise ValueError("Source image exceeds decode byte budget")
                        with Image.open(io.BytesIO(inner.read(name))) as image:
                            reference["mode"] = image.mode
                            reference["size"] = list(image.size)
                        clean[scene] = reference
                    else:
                        variant = (scene, match[2])
                        if variant in seen_variants:
                            raise ValueError(
                                f"Duplicate hazy variant identity: {variant}"
                            )
                        seen_variants.add(variant)
                        hazy.setdefault(scene, []).append(reference)
    return {
        "schema_version": 1,
        "clean": clean,
        "hazy": hazy,
        "nested_archives": hashes,
        "paired_scene_ids": sorted(set(clean) & set(hazy), key=int),
        "unpaired_clean_scene_ids": sorted(set(clean) - set(hazy), key=int),
        "unpaired_hazy_scene_ids": sorted(set(hazy) - set(clean), key=int),
        "sampling_frame": "all four nested haze ZIPs; redundant outer images ignored",
    }


def _rank(seed: int, label: str) -> str:
    return hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()


def prepare_ots_pilot(
    archive: Path,
    output: Path,
    *,
    count: int = 100,
    seed: int = 42,
    scratch_dir: Path = Path("/private/tmp"),
    max_nested_bytes: int = DEFAULT_NESTED_BYTES,
) -> Path:
    """Select source scenes and one variant each, then emit native RGB PNG pairs.

    Structural verification means synthetic filename identity plus equal decoded
    geometry, not a manual alignment review. License/redistribution remain
    unresolved. Any selected invalid pair fails the whole preparation; samples
    are not silently replaced. The final output directory appears only on success.
    """
    archive, output, scratch_dir = Path(archive), Path(output), Path(scratch_dir)
    if type(count) is not int or count < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Need at least two scenes and a nonnegative integer seed")
    if output.exists():
        raise FileExistsError(output)
    inventory = inspect_ots(
        archive, scratch_dir=scratch_dir, max_nested_bytes=max_nested_bytes
    )
    excluded = {
        scene: {
            "mode": inventory["clean"][scene]["mode"],
            "size": inventory["clean"][scene]["size"],
        }
        for scene in inventory["paired_scene_ids"]
        if inventory["clean"][scene]["mode"] != "RGB"
        or min(inventory["clean"][scene]["size"]) < 256
    }
    available = [
        scene for scene in inventory["paired_scene_ids"] if scene not in excluded
    ]
    if len(available) < count:
        raise ValueError(
            f"Only {len(available)} paired source scenes; requested {count}"
        )
    selected = sorted(available, key=lambda scene: _rank(seed, "scene:" + scene))[
        :count
    ]
    rows = []
    requests: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
    for scene in selected:
        variant = min(
            inventory["hazy"][scene],
            key=lambda ref: _rank(seed, "variant:" + scene + ":" + ref["member"]),
        )
        row: dict[str, Any] = {
            "pair_id": f"ots-{scene}",
            "source_scene_id": f"reside-ots-{scene}",
            "registration_status": "verified",
            "registration_method": "synthetic filename identity and matching geometry only",
            "manual_registration_review": "pending",
            "augmentation_parent": None,
            "generator_seed": None,
            "generator_config": "Original OTS synthetic variant; generator settings not independently verified",
        }
        for role, reference in [
            ("clean", inventory["clean"][scene]),
            ("hazy", variant),
        ]:
            member = reference["archive"]
            row["source_" + role + "_archive"] = member
            row["source_" + role + "_member"] = reference["member"]
            row["source_" + role + "_archive_sha256"] = inventory["nested_archives"][
                member
            ]
            requests.setdefault(member, []).append((row, role, reference["member"]))
        rows.append(row)
    archive_hash = sha256(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="btp-ots-output-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        with zipfile.ZipFile(archive) as outer:
            _members(outer)
            for member, items in requests.items():
                with _nested(outer, member, scratch_dir, max_nested_bytes) as (
                    inner,
                    digest,
                ):
                    if digest != inventory["nested_archives"][member]:
                        raise ValueError("Nested archive changed since inventory")
                    entries = _members(inner)
                    for row, role, name in items:
                        if entries[name].file_size > MAX_IMAGE_BYTES:
                            raise ValueError("Source image exceeds decode byte budget")
                        data = inner.read(name)
                        row["source_" + role + "_sha256"] = hashlib.sha256(
                            data
                        ).hexdigest()
                        with Image.open(io.BytesIO(data)) as image:
                            if image.mode != "RGB" or min(image.size) < 256:
                                raise ValueError(
                                    f"Selected image {member}:{name} has mode={image.mode}, size={image.size}; requires RGB and >=256 per side"
                                )
                            image.load()
                            if "source_size" in row and row["source_size"] != list(
                                image.size
                            ):
                                raise ValueError("Selected paired geometry differs")
                            row["source_size"] = list(image.size)
                            relative = f"{role}/{row['pair_id']}.png"
                            target = stage / relative
                            target.parent.mkdir(exist_ok=True)
                            image.save(target, format="PNG")
                            row[role] = relative
                            row[role + "_sha256"] = sha256(target)
        rows = split_by_scene(rows, seed=seed)
        groups = len({row["source_scene_id"] for row in rows})
        expected_validation = max(1, int(count * 0.1))
        if sum(row["split"] == "validation" for row in rows) != expected_validation:
            raise ValueError(
                "Duplicate clean identities prevent exact source-safe 90/10 split"
            )
        document = {
            "schema_version": 1,
            "dataset": "RESIDE OTS user-supplied archive",
            "release": "OTS_BETA pilot seed " + str(seed),
            "source_archive": archive.name,
            "archive_sha256": archive_hash,
            "license": "Unresolved: user-supplied OTS_BETA archive; no license grant or redistribution permission inferred",
            "license_status": "unresolved",
            "manual_review": "pending",
            "split_protocol": {
                "seed": seed,
                "validation_fraction": 0.1,
                "unit": "original clean source identity and exact hashes",
                "selection": "SHA256 rank of seed:scene:canonical numeric ID; one SHA256-ranked named variant per scene",
            },
            "inventory_summary": {
                "paired_sources": len(inventory["paired_scene_ids"]),
                "eligible_sources": len(available),
                "eligibility_rule": "clean source RGB and both dimensions >=256 before seeded selection; selected hazy pair must then match",
                "excluded_clean_sources": excluded,
                "clean_sources": len(inventory["clean"]),
                "hazy_sources": len(inventory["hazy"]),
                "selected_sources": groups,
                "nested_archives": inventory["nested_archives"],
            },
            "scenes": rows,
        }
        audit = audit_training_manifest(document, stage)
        # Identical clean files under different IDs must never inflate the pilot.
        if audit["source_groups"] != count:
            raise ValueError(
                "Duplicate source content in selected pilot; explicit review required"
            )
        (stage / "manifest.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
        (stage / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        (stage / "inventory.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        )
        stage.rename(output)
    return output / "manifest.json"


def main() -> None:
    """Inspect nested identities or prepare an explicit bounded native pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "prepare"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--max-nested-bytes", type=int, default=DEFAULT_NESTED_BYTES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    kwargs = {
        "scratch_dir": args.scratch_dir,
        "max_nested_bytes": args.max_nested_bytes,
    }
    if args.command == "inspect":
        result = inspect_ots(args.archive, **kwargs)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x") as handle:
                handle.write(rendered)
        else:
            print(rendered, end="")
    else:
        if args.output is None:
            parser.error("prepare requires --output")
        print(
            prepare_ots_pilot(
                args.archive, args.output, count=args.count, seed=args.seed, **kwargs
            )
        )


if __name__ == "__main__":
    main()

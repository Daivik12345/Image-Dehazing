"""Validate downloaded checkpoint bytes against recorded Hugging Face LFS metadata."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Verify complete partial files, publish them, and save their provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "configs/checkpoints.json").read_text())
    for name in args.names or catalog:
        spec = catalog[name]
        partial = ROOT / "weights" / (name + ".part")
        target = ROOT / "weights" / name
        source = partial if partial.exists() else target
        if source.stat().st_size != spec["bytes"]:
            raise ValueError(
                f"Incomplete {name}: {source.stat().st_size}/{spec['bytes']}"
            )
        checksum = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != spec["sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch")
        if source == partial:
            if target.exists():
                raise FileExistsError(target)
            partial.rename(target)
        (ROOT / "weights" / (name + ".provenance.json")).write_text(
            json.dumps(spec, indent=2) + "\n"
        )
        print(f"Verified {name}: {spec['bytes']} bytes, SHA-256 {spec['sha256']}")


if __name__ == "__main__":
    main()

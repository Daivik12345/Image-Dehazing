"""Validate the approved SD 2.1 download before publishing it as a checkpoint."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BYTES = 5214865159
EXPECTED_SHA256 = "88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36"
SOURCE = "https://huggingface.co/sd-research/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt"


def main() -> None:
    """Check size and SHA-256; leave incomplete or incorrect downloads untouched."""
    path = ROOT / "weights/v2-1_512-ema-pruned.ckpt.part"
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"Incomplete SD download: {path.stat().st_size}/{EXPECTED_BYTES}"
        )
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            "SD checkpoint checksum does not match Hugging Face LFS metadata"
        )
    target = path.with_suffix("")
    if target.exists():
        raise FileExistsError(target)
    path.rename(target)
    (ROOT / "weights/sd-provenance.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "sha256": checksum.hexdigest(),
                "bytes": EXPECTED_BYTES,
                "discovery": "upstream issue 12; public community mirror",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified SD 2.1 checkpoint: {target}")


if __name__ == "__main__":
    main()

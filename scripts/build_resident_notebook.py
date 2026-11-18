"""Create an isolated teacher-equivalence gate; no teacher targets launched here."""

import json
from pathlib import Path

from build_research_notebooks import build


def main() -> None:
    """Build fixed two-scene six-cold/four-resident validation workflow."""
    build(
        "btp-resident-validation",
        "Fresh M0 repeatability and resident A-B-A-B gate. Part of the approved development batch; no held-out scenes or teacher targets.",
        [
            (
                "code",
                """archives = list(Path("/kaggle/input").rglob("nh-haze-512.bundle"))
assert len(archives) == 1
DATA = OUTPUT / "evaluation"
DATA.mkdir()
with zipfile.ZipFile(archives[0]) as archive:
    for name in archive.namelist():
        assert (DATA / name).resolve().is_relative_to(DATA)
    archive.extractall(DATA)
MANIFEST = DATA / "nh_haze_manifest.json"
assert MANIFEST.read_bytes() == (ROOT / "configs/evaluation/nh_haze_manifest.json").read_bytes()
WEIGHTS = Path("/tmp") / (OUTPUT.name + "-weights-links")
WEIGHTS.mkdir()
catalog = json.loads((ROOT / "configs/checkpoints.json").read_text())
for name in ("v2-1_512-ema-pruned.ckpt", "stage2.pt"):
    matches = list(Path("/kaggle/input").rglob(name))
    assert len(matches) == 1, name
    digest = hashlib.sha256()
    with matches[0].open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024**2), b""):
            digest.update(block)
    assert digest.hexdigest() == catalog[name]["sha256"], name
    (WEIGHTS / name).symlink_to(matches[0])
""",
            ),
            (
                "code",
                """run(["-m", "btp.resident_validation", "--manifest", str(MANIFEST),
     "--weights", str(WEIGHTS), "--output", str(OUTPUT / "resident-validation"),
     "--max-seconds", "2400"])
decision = json.loads((OUTPUT / "resident-validation/decision.json").read_text())
print("Status:", decision["status"], "Eligible:", decision["eligible"])
# Remove only local links before archive packaging; pretrained bytes remain on input mount.
for link in WEIGHTS.iterdir():
    assert link.is_symlink()
    link.unlink()
WEIGHTS.rmdir()
""",
            ),
        ],
    )
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "id": "cubingultimate/btp-resident-validation",
        "title": "btp-resident-validation",
        "code_file": "btp-resident-validation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "cubingultimate/btp-research-source",
            "cubingultimate/btp-paper-weights-2",
            "cubingultimate/btp-eval-nh-haze",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    dest = root / "configs/kaggle-resident-validation"
    dest.mkdir(exist_ok=True)
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

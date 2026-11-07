"""Create a portable, credential-free source bundle and Kaggle notebook."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from btp.config import ROOT, UPSTREAM


def build_bundle(output: Path) -> dict:
    """Package allowlisted source files, excluding weights, environments and logs."""
    selected = [
        ROOT / p for p in ["README.md", "AGENTS.md", "pyproject.toml", "app.py"]
    ]
    for directory in ["src/btp", "tests", "configs", "docs", "scripts"]:
        selected.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.name.startswith(".")
            and p.suffix in {".py", ".md", ".json", ".toml", ".txt", ".yaml", ".yml"}
        )
    selected.extend((ROOT / "notebooks").glob("*.ipynb"))
    if (UPSTREAM / ".git").exists():
        tracked = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "ls-files", "-z"]
        )
        selected.extend(
            UPSTREAM / name.decode() for name in tracked.split(b"\0") if name
        )
        commit = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        provenance = json.loads((UPSTREAM / "SOURCE.json").read_text())
        selected.extend(UPSTREAM / name for name in provenance["files"])
        selected.append(UPSTREAM / "SOURCE.json")
        commit = provenance["commit"]
    selected.extend([ROOT / "examples/clean.png", ROOT / "examples/README.md"])
    # Upstream tracks Finder metadata; upload services may omit these files.
    files = sorted(path for path in set(selected) if path.name != ".DS_Store")
    snapshot = {
        "upstream_commit": commit,
        "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, "BTP/" + str(path.relative_to(ROOT)))
        bundle.writestr("BTP/source-manifest.json", json.dumps(snapshot, indent=2))
    return snapshot


def main() -> None:
    """Build the source archive after the notebook and guide are prepared."""
    target = ROOT / "artifacts/btp-kaggle.zip"
    snapshot = build_bundle(target)
    print(
        f"Created {target}: {len(snapshot['files'])} files; {target.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()

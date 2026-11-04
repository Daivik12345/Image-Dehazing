"""Resume the documented public mirrors; verify bytes before enabling inference."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download selected checkpoint files and verify their pinned SHA-256 hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "configs/checkpoints.json").read_text())
    for name in args.names or catalog:
        if name not in catalog:
            parser.error(f"Unknown checkpoint: {name}")
        spec = catalog[name]
        print(f"{name}: {spec['bytes'] / 1e9:.2f} GB. {spec['provenance']}", flush=True)
        if not (ROOT / "weights" / name).exists():
            subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--speed-time",
                    "60",
                    "--speed-limit",
                    "1024",
                    "-C",
                    "-",
                    spec["source"],
                    "-o",
                    str(ROOT / "weights" / (name + ".part")),
                ],
                check=True,
            )
        subprocess.run(
            ["uv", "run", "python", str(ROOT / "scripts/verify_weights.py"), name],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()

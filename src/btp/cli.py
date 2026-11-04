"""Command-line experiments and checkpoint diagnostics."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageOps

from btp.config import ROOT, RunConfig, checkpoint_status
from btp.experiment import create_run, run_preview
from btp.preview import demo_image


def main() -> None:
    """Run a reproducible experiment, tiny CPU smoke, or checkpoint inventory."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["smoke", "doctor", "run"], default="smoke", nargs="?"
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--backend", choices=["preview", "pretrained"], default="preview"
    )
    parser.add_argument(
        "--stage", choices=["both", "hazegen", "diffdehaze"], default="both"
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--sampler", choices=["spaced", "accsamp"], default="spaced")
    parser.add_argument(
        "--precision", choices=["auto", "float32", "mixed16"], default="auto"
    )
    parser.add_argument("--weights-dir", default=str(ROOT / "weights"))
    args = parser.parse_args()
    config = RunConfig(
        backend=args.backend,
        stage=args.stage,
        size=args.size,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        sampler=args.sampler,
        weights_dir=args.weights_dir,
        precision=args.precision,
    )
    if args.command == "doctor":
        print(json.dumps(checkpoint_status(config), indent=2))
        return
    if args.command == "smoke":
        config = RunConfig(size=128, steps=5, seed=42)
    if config.backend == "pretrained":
        missing = [
            name for name, exists in checkpoint_status(config).items() if not exists
        ]
        if missing:
            parser.error("Missing checkpoints: " + ", ".join(missing))
    image = (
        ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
        if args.input
        else demo_image(128)
    )
    run = create_run(image, config)
    if config.backend == "preview":
        run_preview(image, config, run)
        print(f"CPU physical preview completed: {run}")
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                str(ROOT / ".venv-paper/bin/python"),
                "python",
                "-m",
                "btp.paper",
                str(run),
            ],
            env=env,
            check=True,
            cwd=ROOT,
        )
        print(f"Pretrained run completed: {run}")


if __name__ == "__main__":
    main()

"""Generate the reviewable notebook; model cells are deliberately unexecuted."""

import json
from pathlib import Path

cells = []


def md(text: str) -> None:
    """Append a Markdown notebook cell."""
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
    )


def code(text: str) -> None:
    """Append an unexecuted Python notebook cell."""
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.splitlines(True),
        }
    )


md("""# BTP: HazeGen / DiffDehaze CUDA reference inference
Run cells **one at a time**, in order, with a GPU and Internet enabled. Attach the private source dataset and private weight dataset first. This notebook has **not been run on Kaggle**. No paper scores are claimed.

Reference runs execute the original upstream scripts at 50 steps, float32, batch size 1. The authors' short-edge resize, zero padding and output resize are retained. CPU/MPS preview settings and AlignOp patches from the local viewer are not used. Torch 2.6 + SDPA are compatibility choices rather than an exact recreation of the authors' Torch 2.2.2/xformers runtime.
""")
code("""from pathlib import Path
import hashlib, json, os, shutil, signal, subprocess, sys, time, zipfile
from IPython.display import Image, display, clear_output, FileLink

subprocess.run(["nvidia-smi"], check=True)
# EDIT this to the private dataset's actual owner/slug/version.
WEIGHTS_DATASET_VERSION = "EDIT_ME/weights-dataset/versions/1"
assert not WEIGHTS_DATASET_VERSION.startswith("EDIT_ME"), "Set the dataset version above first"
ROOT = Path("/kaggle/working/BTP")
OUTPUT = Path("/kaggle/working/btp-results")
OUTPUT.mkdir(parents=True, exist_ok=True)
""")
md(
    "## 1. Restore and verify the uploaded source bundle\nKaggle may keep the ZIP or expand it; both layouts are supported. Existing working copies are checked rather than silently overwritten."
)
code("""archives = list(Path("/kaggle/input").rglob("btp-kaggle.zip"))
if not ROOT.exists():
    if len(archives) == 1:
        with zipfile.ZipFile(archives[0]) as archive:
            for item in archive.namelist():
                destination = (Path("/kaggle/working") / item).resolve()
                if not destination.is_relative_to(Path("/kaggle/working")):
                    raise ValueError("Unsafe archive member")
            archive.extractall("/kaggle/working")
    else:
        manifests = list(Path("/kaggle/input").rglob("source-manifest.json"))
        assert len(manifests) == 1, "Attach exactly one BTP source dataset"
        shutil.copytree(manifests[0].parent, ROOT)
snapshot = json.loads((ROOT / "source-manifest.json").read_text())
for relative, expected in snapshot["files"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
print("Verified source snapshot:", snapshot["upstream_commit"])
""")
md(
    "## 2. Create an isolated Python 3.10 CUDA runtime\nThis downloads Python, CUDA-enabled PyTorch and dependencies. The environment is in `/tmp` so it is not included in saved notebook outputs. It leaves Kaggle's notebook-kernel packages intact."
)
code("""uv_binary = shutil.which("uv")
if uv_binary is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    UV = [sys.executable, "-m", "uv"]
else:
    UV = [uv_binary]
ENV = Path("/tmp/btp-kaggle-env")
PYTHON = ENV / "bin/python"
if not PYTHON.exists():
    subprocess.run(UV + ["venv", "--python", "3.10", str(ENV)], check=True)
subprocess.run(UV + ["pip", "install", "--python", str(PYTHON),
    "torch==2.6.0", "torchvision==0.21.0", "--index-url",
    "https://download.pytorch.org/whl/cu118"], check=True)
subprocess.run(UV + ["pip", "install", "--python", str(PYTHON), "-r",
    str(ROOT / "configs/kaggle-requirements.txt")], check=True)
RUN = UV + ["run", "--no-project", "--python", str(PYTHON), "python"]
ENVIRONMENT = os.environ.copy()
ENVIRONMENT.update(PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="0", ATTN_MODE="sdp")
subprocess.run(RUN + ["-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print('VRAM GB:', torch.cuda.get_device_properties(0).total_memory/1e9)"], check=True, env=ENVIRONMENT)
with (OUTPUT / "runtime-packages.txt").open("w") as file:
    subprocess.run(UV + ["pip", "freeze", "--python", str(PYTHON)], check=True, stdout=file)
""")
md(
    "## 3. Locate and hash the three checkpoints\nWeights remain on the read-only input mount. The two supplied Baidu files are byte-identical to our previous copies; this step detects incomplete/wrong uploads."
)
code("""catalog = json.loads((ROOT / "configs/checkpoints.json").read_text())
paths = {}
for name, spec in catalog.items():
    candidates = list(Path("/kaggle/input").rglob(name))
    assert len(candidates) == 1, f"Expected exactly one {name}, found {len(candidates)}"
    path = candidates[0]
    assert path.stat().st_size == spec["bytes"], f"Incomplete upload: {name}"
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == spec["sha256"], f"Hash mismatch: {name}"
    paths[name] = path
    print("Verified", name, digest.hexdigest())
# Files can be in different dataset directories; symlink them into a common folder.
WEIGHTS = Path("/tmp/btp-reference-weights")
WEIGHTS.mkdir(exist_ok=True)
for name, path in paths.items():
    target = WEIGHTS / name
    if target.is_symlink():
        assert target.resolve() == path.resolve(), "Restart session after changing datasets"
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(path)
(OUTPUT / "weights-verification.json").write_text(json.dumps({
    "dataset_version": WEIGHTS_DATASET_VERSION,
    "files": {name: {"path": str(path), **catalog[name]} for name, path in paths.items()},
}, indent=2))
""")
md(
    "## 4. Tiny CPU smoke first\nRandom tiny networks test plumbing only. Their images are not pretrained results. This cell does not run training or a large model."
)
code("""subprocess.run(RUN + ["-m", "btp.model_smoke"], cwd=ROOT, env=ENVIRONMENT, check=True)
""")
md(
    "## 5. Define live single-image execution\nEach case runs in a fresh process, releasing the previous model's GPU memory. Diagnostic previews decode noisy sampler states and are **not final images**. Interrupted runs are terminated. Use a new output name to rerun; existing experiments are never overwritten."
)
code("""def run_case(name, mode, image, preview_every=10):
    output = OUTPUT / name
    if output.exists():
        raise FileExistsError(f"Choose a new case name: {output}")
    command = RUN + ["-u", "-m", "btp.reference", "--mode", mode,
        "--input", str(image), "--weights", str(WEIGHTS), "--output", str(output),
        "--seed", "42", "--dataset-version", WEIGHTS_DATASET_VERSION,
        "--preview-every", str(preview_every)]
    log_path = OUTPUT / f"{name}.log"
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=ENVIRONMENT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                clear_output(wait=True)
                status = output / "status.json"
                print(name, json.loads(status.read_text()) if status.exists() else "Starting runtime…")
                print("\\n".join(log_path.read_text(errors="replace").splitlines()[-6:]))
                frames = sorted(output.glob("latent-*.png")) if output.exists() else []
                if frames:
                    print("Diagnostic noisy-latent decode:")
                    display(Image(filename=str(frames[-1]), width=600))
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(log_path.read_text(errors="replace")[-12000:])
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    clear_output(wait=True)
    print("Completed:", output)
    print("Input:")
    display(Image(filename=str(output / "input.png"), width=600))
    print("Final model output:")
    display(Image(filename=str(output / "final/input.png"), width=600))
    return output / "final/input.png"
""")
md(
    "## 6. HazeGen — clean photograph → hazy photograph\nThe bundled garden photo is a functional example, not a paper benchmark. You can replace CLEAN with your own clean image mounted under `/kaggle/input`. HazeGen uses the authors' prompt and `w=0.85`, with 50 steps."
)
code("""CLEAN = ROOT / "examples/clean.png"
GENERATED_HAZE = run_case("01-hazegen", "hazegen", CLEAN)
""")
md(
    "## 7. DiffDehaze — upstream real hazy example, standard sampler\nStart with one supplied example. This is qualitative inference; the example has no paired clean ground truth here."
)
code("""HAZY = ROOT / "upstream/Learning-Hazing-to-Dehazing/inputs/1.png"
SPACED = run_case("02-spaced", "spaced", HAZY)
""")
md(
    "## 8. DiffDehaze — same real image, AccSamp\nUses the original `tau=0.8`, `omega=0.6`, fidelity scale `s=0.1`, and the 50-step schedule. AccSamp makes fewer actual denoiser calls than a full 50-call spaced trajectory. Its original AlignOp behavior is preserved."
)
code("""ACCSAMP = run_case("03-accsamp", "accsamp", HAZY)
""")
md(
    "## 9. Complete pipeline — generated haze → DiffDehaze\nOptional after inspecting the previous results. An end-to-end roundtrip is not a replacement for the paper's real-world dataset evaluation."
)
code("""ROUNDTRIP = run_case("04-roundtrip", "accsamp", GENERATED_HAZE)
""")
md(
    "## 10. Save outputs before ending the session\nSave a Kaggle notebook version with outputs and download this ZIP. It contains actual saved images, logs, manifests, hashes and runtime packages—not model weights or the virtual environment. GPU outputs have not been pre-filled into this notebook."
)
code("""archive = shutil.make_archive("/kaggle/working/btp-results", "zip", OUTPUT)
display(FileLink(archive))
print("Use Kaggle's Output/Files panel if the link is unavailable. Save a notebook version before ending the session.")
""")
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(cells):
    cell["id"] = f"btp-{index:03d}"
Path("notebooks/kaggle_reference.ipynb").write_text(
    json.dumps(notebook, indent=2) + "\n"
)

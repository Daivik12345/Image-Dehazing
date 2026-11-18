"""Build new research notebooks without touching historical baselines."""

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str) -> dict:
    """Create an unexecuted notebook cell; reject invalid Python during generation."""
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        ast.parse(source)
        result.update(execution_count=None, outputs=[])
    return result


BOOTSTRAP = """from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
ROOT = Path("/kaggle/working/BTP")
OUTPUT = Path("/kaggle/working") / ("btp-research-" + str(time.time_ns()))
OUTPUT.mkdir()
# Upload artifacts/btp-kaggle.zip as btp-research-source.bundle in a private dataset.
bundles = list(Path("/kaggle/input").rglob("btp-research-source.bundle"))
if not ROOT.exists():
    assert len(bundles) == 1, "Attach one research source bundle; see docs/RESEARCH_WORKFLOWS.md"
    with zipfile.ZipFile(bundles[0]) as archive:
        for info in archive.infolist():
            destination = (ROOT.parent / info.filename).resolve()
            assert destination.is_relative_to(ROOT), "Unsafe source member"
        archive.extractall(ROOT.parent)
snapshot = json.loads((ROOT / "source-manifest.json").read_text())
for name, expected in snapshot["files"].items():
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print("Verified source files:", len(snapshot["files"]))
"""
ENV = """uv = shutil.which("uv")
if uv is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
UV = [uv] if uv else [sys.executable, "-m", "uv"]
PYTHON = Path("/tmp/btp-research-env/bin/python")
if not PYTHON.exists():
    subprocess.run(UV + ["venv", "--python", "3.10", str(PYTHON.parent.parent)], check=True)
subprocess.run(UV + ["pip", "install", "--python", str(PYTHON),
                     "torch==2.6.0", "torchvision==0.21.0", "--index-url",
                     "https://download.pytorch.org/whl/cu118"], check=True)
subprocess.run(UV + ["pip", "install", "--python", str(PYTHON), "-r",
                     str(ROOT / "configs/kaggle-requirements.txt")], check=True)
RUN = UV + ["run", "--no-project", "--python", str(PYTHON), "python"]
ENVIRONMENT = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="0")
def run(arguments):
    subprocess.run(RUN + arguments, cwd=ROOT, env=ENVIRONMENT, check=True)
with (OUTPUT / "runtime-packages.txt").open("w") as handle:
    subprocess.run(UV + ["pip", "freeze", "--python", str(PYTHON)], stdout=handle, check=True)
(OUTPUT / "source-manifest.json").write_text(json.dumps(snapshot, indent=2))
"""


def build(name: str, description: str, steps: list[tuple[str, str]]) -> None:
    """Save one independent, runnable source notebook (not a published Kaggle URL)."""
    cells = [
        cell(
            "markdown",
            f"# {name}\n{description}\n\nImplementation workflow; no completed GPU experiment is implied. Historical notebooks are preserved.",
        ),
        cell(
            "markdown",
            "## Source and runtime\nAttach the private research source bundle. Internet is needed to provision the isolated runtime; running this cell downloads dependencies. CPU smoke precedes GPU use. No long experiment runs automatically.",
        ),
        cell("code", BOOTSTRAP),
        cell("code", ENV),
    ]
    cells.extend(cell(kind, source) for kind, source in steps)
    cells.append(
        cell(
            "code",
            """archive = shutil.make_archive(str(OUTPUT), "zip", OUTPUT)
print("Download evidence:", archive)
""",
        )
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "notebooks" / f"{name}.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n"
    )

    target = str(ROOT / "notebooks" / f"{name}.ipynb")
    subprocess.run(["uv", "run", "ruff", "check", "--fix", target], check=True)
    subprocess.run(["uv", "run", "ruff", "format", target], check=True)


def main() -> None:
    """Generate only workflows with implemented entry points."""
    build(
        "btp-data-audit",
        "Audit a separately licensed training corpus before student training. This does not turn NH-HAZE evaluation images into training data.",
        [
            (
                "markdown",
                "## Audit training manifest\nSet the manifest path and provide the frozen evaluation manifest. The audit checks hashes, paths, paired RGB geometry and split leakage. Manual registration and near-duplicate review remain required.",
            ),
            (
                "code",
                '''TRAIN_MANIFEST = None  # Path to your separate training corpus manifest
EVAL_MANIFEST = ROOT / "configs/evaluation/nh_haze_manifest.json"
if TRAIN_MANIFEST is None:
    print("Training disabled: provide a licensed separate corpus and complete audit.")
else:
    script = """import json, sys
from pathlib import Path
from btp.training_data import audit_training_manifest
p=Path(sys.argv[1]); e=json.loads(Path(sys.argv[2]).read_text())
report=audit_training_manifest(json.loads(p.read_text()), p.parent, evaluation_scenes=e['scenes'])
Path(sys.argv[3]).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
"""
    run(["-c", script, str(TRAIN_MANIFEST), str(EVAL_MANIFEST), str(OUTPUT / "audit.json")])
''',
            ),
        ],
    )
    build(
        "btp-student-pilot",
        "Inspect the compact architecture and run a two-update CPU plumbing test. Real training requires an audited corpus and an approved cumulative budget.",
        [
            ("code", 'run(["scripts/smoke_student.py"])\n'),
            (
                "code",
                """TRAIN_MANIFEST = None
SOURCE_COMMIT = None  # Exact Image-Dehazing commit used for this source bundle
RUN_TRAINING = False
RESUME_CHECKPOINT = None
UNTIL_UPDATE = 100  # Absolute endpoint; first profile counts inside total pilot cap
if TRAIN_MANIFEST is not None:
    assert SOURCE_COMMIT, "Record the exact project source commit"
    arguments = ["-m", "btp.student.train_cli", "--manifest", str(TRAIN_MANIFEST),
        "--data-root", str(Path(TRAIN_MANIFEST).parent),
        "--evaluation-manifest", str(ROOT / "configs/evaluation/nh_haze_manifest.json"),
        "--protocol", str(ROOT / "configs/research/protocol-v1.json"),
        "--architecture", str(ROOT / "configs/research/student/architecture.json"),
        "--source-commit", SOURCE_COMMIT, "--output", str(OUTPUT / "pilot"),
        "--until-update", str(UNTIL_UPDATE), "--device", "cuda:0"]
    if RUN_TRAINING:
        arguments.append("--run")  # Enable only after explicit budget approval
    if RESUME_CHECKPOINT is not None:
        arguments.extend(["--resume", str(RESUME_CHECKPOINT)])
    run(arguments)
else:
    print("No training corpus supplied; fixture only.")
""",
            ),
            (
                "markdown",
                "## Pilot gate\nThe smoke uses random width-two fixture weights, not a dehazing checkpoint. The default width-16 architecture has 1,724,275 parameters. Use `BoundedTrainer` only after corpus audit; count the first 100 profiling updates inside the proposed 2,000-update/two-T4-hour cap. See the training recipe in docs/RESEARCH_WORKFLOWS.md. No actual training corpus or budget approval is bundled here.",
            ),
        ],
    )
    build(
        "btp-global-tiles",
        "Native-size student inference using streaming overlapping tiles. A real hash-identified student checkpoint is required; random fixture outputs are not evaluation evidence.",
        [
            (
                "code",
                """run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_student_cli.py"])
""",
            ),
            (
                "code",
                """CHECKPOINT = None  # Path to exported trained student artifact
CHECKPOINT_SHA256 = None
IMAGE = None  # Explicit native RGB input path
RUN_NATIVE = False
if RUN_NATIVE:
    assert CHECKPOINT is not None and CHECKPOINT_SHA256 and IMAGE is not None
    run(["-m", "btp.student.cli", "--image", str(IMAGE), "--checkpoint", str(CHECKPOINT),
         "--architecture", str(ROOT / "configs/research/student/architecture.json"),
         "--sha256", CHECKPOINT_SHA256, "--device", "cuda:0", "--tiled",
         "--output", str(OUTPUT / "native")])
else:
    print("Native GPU run disabled; attach a trained checkpoint and select an input first.")
""",
            ),
            (
                "markdown",
                "Use identical geometry for full-frame and tiled controls. Evaluate full-image PSNR/SSIM plus `btp.research_eval.seam_metrics`; blurred seams alone are not improvement. Global alignment is experimental and needs an explicitly resized global prediction in native coordinates; it is not enabled in this default.",
            ),
        ],
    )

    build(
        "btp-teacher-targets",
        "Training-only target generation after persistent/cold equivalence and exact budget approval.",
        [
            (
                "markdown",
                "Validate cold M0 versus resident output and A-B-A state reset first. This stage stays disabled until the exact training pair list/budget is approved. Clean targets are never passed to the teacher.",
            ),
            (
                "code",
                """RUN_TARGETS = False
TRAIN_MANIFEST = None
WEIGHTS = None
APPROVED_PAIR_IDS = []
if RUN_TARGETS:
    assert TRAIN_MANIFEST is not None and WEIGHTS is not None and APPROVED_PAIR_IDS
    settings = dict(manifest=str(TRAIN_MANIFEST), weights=str(WEIGHTS),
                    pairs=APPROVED_PAIR_IDS, max_attempts=8, max_seconds=1800)
    (OUTPUT / "target-settings.json").write_text(json.dumps(settings, indent=2))
    script = '''import json, sys
from pathlib import Path
from btp.resident import ResidentConfig, ResidentTeacher
from btp.teacher_targets import generate_targets, TargetBudget
root, output = Path(sys.argv[1]), Path(sys.argv[2])
s = json.loads((output/'target-settings.json').read_text()); p = Path(s['manifest'])
evaluation = json.loads((root/'configs/evaluation/nh_haze_manifest.json').read_text())
with ResidentTeacher(ResidentConfig(root/'upstream/Learning-Hazing-to-Dehazing/configs/inference/stage2_accsamp.yaml'), Path(s['weights'])) as teacher:
    report = generate_targets(document=json.loads(p.read_text()), root=p.parent,
        output=output/'targets', teacher=teacher, approved_pair_ids=s['pairs'],
        budget=TargetBudget(s['max_attempts'], s['max_seconds']), evaluation_scenes=evaluation['scenes'], seed=42)
print(json.dumps(report, indent=2))
'''
    run(["-c", script, str(ROOT), str(OUTPUT)])
else:
    print("Disabled: resident equivalence, audited corpus and exact budget approval required.")
""",
            ),
        ],
    )
    build(
        "btp-guidance-quality",
        "Separate B0, confidence and matched-mean uniform controls; no clean target enters inference.",
        [
            (
                "code",
                'run(["-m", "unittest", "discover", "-s", "tests", "-p", "test_guidance.py"])\n',
            ),
            (
                "markdown",
                "Register epsilon from training-corpus input statistics before development scoring. Set it explicitly below. This notebook does not silently choose a value or launch a sweep. First prove B0 resident/cold equivalence.",
            ),
            (
                "code",
                """RUN_GUIDANCE = False
EPSILON_VAR = None
IMAGE = None
WEIGHTS = None
if RUN_GUIDANCE:
    assert EPSILON_VAR is not None and IMAGE is not None and WEIGHTS is not None
    script = '''import sys
from pathlib import Path
from btp.resident import ResidentConfig, ResidentTeacher
root, output, image, weights = map(Path, sys.argv[1:5]); epsilon=float(sys.argv[5])
config=root/'upstream/Learning-Hazing-to-Dehazing/configs/inference/stage2_accsamp.yaml'
for mode in ('B0', 'confidence', 'uniform'):
    with ResidentTeacher(ResidentConfig(config, guidance_mode=mode,
            epsilon_var=None if mode=='B0' else epsilon), weights) as teacher:
        teacher.predict(image, output/mode, 42)
'''
    run(["-c", script, str(ROOT), str(OUTPUT), str(IMAGE), str(WEIGHTS), str(EPSILON_VAR)])
else:
    print("Disabled until registered calibration and bounded GPU smoke approval.")
""",
            ),
        ],
    )


if __name__ == "__main__":
    main()

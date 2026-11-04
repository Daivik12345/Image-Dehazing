"""Local live viewer for physical preview and pretrained paper inference."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from btp.config import ROOT, RunConfig, checkpoint_status
from btp.experiment import create_run, save_frame, write_json
from btp.preview import demo_image, preview_frames, sample_photo
from btp.viewer import observe_run, resolve_run

st.set_page_config(page_title="BTP · Haze Lab", page_icon="◌", layout="wide")
st.markdown(
    """<style>
.stApp {background:#101915;color:#e4e9df}
h1 {font-family:Georgia,serif!important;letter-spacing:-.055em;font-size:4rem!important}
[data-testid="stSidebar"] {background:#17241d}
[data-testid="stMetric"] {border-top:1px solid #59715d;padding-top:1rem}
.stButton>button[kind="primary"] {background:#d4ed91;color:#17241d;border:0}
</style>""",
    unsafe_allow_html=True,
)
st.caption(
    "BTP / COMPUTER VISION LAB                                      CVPR 2025 · LOCAL WORKSPACE"
)
st.title("Through the haze.")
st.write("Follow an image from clean → hazy → restored, one step at a time.")

if "run" in st.query_params:
    try:
        observe_run(resolve_run(st.query_params["run"]))
    except ValueError as error:
        st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Experiment controls")
    backend_label = st.selectbox(
        "Engine", ["CPU physical preview", "Pretrained paper models"]
    )
    backend = "preview" if backend_label.startswith("CPU") else "pretrained"
    stage_label = st.selectbox(
        "Pipeline", ["Both stages", "HazeGen only", "DiffDehaze only"]
    )
    stage = {
        "Both stages": "both",
        "HazeGen only": "hazegen",
        "DiffDehaze only": "diffdehaze",
    }[stage_label]
    size = st.select_slider("Maximum image side", options=[128, 256, 512], value=128)
    steps = st.select_slider(
        "Sampling / preview steps", options=[5, 10, 20, 50], value=5
    )
    seed = int(
        st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
    )
    density = st.slider(
        "Physical haze density", 0.0, 3.0, 1.2, disabled=backend == "pretrained"
    )
    blend = st.slider("HazeGen blend w", 0.0, 1.0, 0.85, disabled=backend == "preview")
    sampler = st.selectbox(
        "DiffDehaze sampler", ["spaced", "accsamp"], disabled=backend == "preview"
    )
    device = st.selectbox(
        "Model device", ["auto", "mps", "cpu", "cuda"], disabled=backend == "preview"
    )
    precision = st.selectbox(
        "Model precision", ["auto", "float32", "mixed16"], disabled=backend == "preview"
    )
    every = int(
        st.number_input(
            "Decode preview every N steps", min_value=1, max_value=50, value=2
        )
    )
    weights_dir = st.text_input("Checkpoint folder", str(ROOT / "weights"))
    st.caption(
        "Reduced resolution and fewer steps reduce compute, not parameter count. No training runs here."
    )

config = RunConfig(
    backend=backend,
    stage=stage,
    size=size,
    steps=steps,
    seed=seed,
    haze_density=density,
    blend=blend,
    sampler=sampler,
    device=device,
    preview_every=every,
    weights_dir=weights_dir,
    precision=precision,
)
if backend == "preview":
    st.info(
        "CPU preview: atmospheric scattering + dark-channel-prior restoration. These are baseline visualizations, not HazeGen or DiffDehaze model outputs."
    )
else:
    st.info(
        "Actual pretrained ControlLDM models. Small settings trade quality for speed. Intermediate images are predicted-clean diffusion previews, not final results."
    )
    st.warning(
        "This Mac has 8 GiB unified memory. Auto precision uses mixed float16 on MPS/CUDA to reduce weight memory. Full-size inference is not yet validated here; start at 128 pixels and 5 steps."
    )
    st.caption(
        "Checkpoint sources and hashes are recorded in each manifest. The supplied Baidu stage files match our downloaded copies byte-for-byte."
    )
    if size < 512 or steps < 50:
        st.warning(
            "Reduced settings are execution experiments: the tested 128px/5-step and 256px/20-step runs produced severe artifacts. They are not paper-quality results. The authors use 512px/50-step inference; that was too slow on this 8 GiB Mac."
        )
    for name, present in checkpoint_status(config).items():
        st.caption(f"{'✓' if present else 'Missing'} · {name}")
    if sampler == "accsamp":
        st.caption(
            "AccSamp uses the paper’s τ=0.8, ω=0.6 and guidance s=0.1. It needs additional gradient/decoder memory."
        )

demo_source = st.radio(
    "Example input",
    ["Bundled garden photograph", "Generated test illustration"],
    horizontal=True,
)
upload = st.file_uploader(
    "Choose an image, or use the generated clean demo", type=["png", "jpg", "jpeg"]
)
try:
    image = (
        ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
        if upload
        else (
            sample_photo()
            if demo_source == "Bundled garden photograph"
            else demo_image(512)
        )
    )
except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
    st.error(f"Cannot read this image: {error}")
    st.stop()
if stage == "diffdehaze" and upload is None:
    st.warning(
        "The selected example is a clean image. Upload a hazy photo to test dehazing on real input."
    )
columns = st.columns(3)
slots = []
for column, label in zip(
    columns, ["01 / INPUT", "02 / HAZEGEN", "03 / DIFFDEHAZE"], strict=True
):
    with column:
        st.caption(label)
        slots.append(st.empty())
slots[0].image(image, width="stretch")
progress = st.progress(0.0, text="Ready for an experiment")
status_box = st.empty()
missing = backend == "pretrained" and not all(checkpoint_status(config).values())
python = ROOT / ".venv-paper/bin/python"
if missing:
    st.warning(
        "Add the missing checkpoints to the folder above. See weights/README.md for the authors’ download links."
    )
if backend == "pretrained" and not python.exists():
    st.warning("Create the isolated paper runtime first; see README.md.")

if st.button(
    "Run pipeline →",
    type="primary",
    disabled=missing or (backend == "pretrained" and not python.exists()),
):
    run = create_run(image, config)
    st.session_state["last_run"] = str(run)
    try:
        if backend == "preview":
            for name, step, total, frame in preview_frames(image, config):
                save_frame(run, name, step, total, frame)
                slots[1 if name == "hazegen" else 2].image(frame, width="stretch")
                progress.progress(
                    step / total, text=f"Physical {name} preview · {step}/{total}"
                )
            write_json(run / "status.json", {"state": "completed"})
        else:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            with (run / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "--no-project",
                        "--python",
                        str(python),
                        "python",
                        "-m",
                        "btp.paper",
                        str(run),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                st.session_state["worker_pid"] = process.pid
                try:
                    while process.poll() is None:
                        state = json.loads((run / "status.json").read_text())
                        status_box.write(
                            f"{state.get('stage', 'pipeline')} · {state.get('phase', state['state'])}"
                        )
                        for index, name in [(1, "hazegen"), (2, "diffdehaze")]:
                            frames = sorted(run.glob(f"{name}-*.png"))
                            if frames:
                                # Frames become visible after status.json is atomically updated.
                                try:
                                    slots[index].image(
                                        Image.open(frames[-1]).copy(), width="stretch"
                                    )
                                except OSError:
                                    pass
                        if "step" in state:
                            progress.progress(
                                min(1.0, state["step"] / state["total"]),
                                text=f"{state['stage']} · {state['step']}/{state['total']}",
                            )
                        time.sleep(0.5)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                        write_json(
                            run / "status.json",
                            {"state": "cancelled", "reason": "Viewer stopped or rerun"},
                        )
                if process.returncode:
                    state = json.loads((run / "status.json").read_text())
                    raise RuntimeError(
                        state.get(
                            "error",
                            f"Worker exited {process.returncode}; see worker.log",
                        )
                    )
        progress.progress(1.0, text="Completed")
        status_box.success(f"Saved experiment: {run.name}")
    except Exception as error:
        write_json(run / "status.json", {"state": "failed", "error": str(error)})
        st.error(str(error))

if "last_run" in st.session_state:
    run = Path(st.session_state["last_run"])
    for index, name in [(1, "hazegen"), (2, "diffdehaze")]:
        result = run / f"{name}.png"
        if result.exists():
            slots[index].image(Image.open(result), width="stretch")
            columns[index].download_button(
                f"Download {name}", result.read_bytes(), f"{name}.png", "image/png"
            )
    with st.expander("Run configuration & provenance"):
        st.json(json.loads((run / "manifest.json").read_text()))
        st.download_button(
            "Download manifest",
            (run / "manifest.json").read_bytes(),
            "manifest.json",
            "application/json",
        )
    st.caption(f"Outputs, intermediate frames and any model logs: {run}")
st.divider()
st.caption(
    "Learning Hazing to Dehazing · Wang et al. · Paper models remain in upstream/; experiment code lives in src/btp/."
)

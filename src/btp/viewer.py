"""Read-only live observation of saved or CLI-launched experiments."""

import json
from pathlib import Path

import streamlit as st
from PIL import Image

from btp.config import ROOT


def resolve_run(name: str) -> Path:
    """Resolve a run ID, rejecting paths outside this project's run directory."""
    run = (ROOT / "runs" / name).resolve()
    if run.parent != (ROOT / "runs").resolve() or not (run / "manifest.json").is_file():
        raise ValueError("Unknown experiment ID")
    return run


@st.fragment(run_every=1.0)
def observe_run(run: Path) -> None:
    """Refresh stage frames, loading phases, and downloads without launching work."""
    state = json.loads((run / "status.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    st.caption(f"OBSERVING / {run.name}")
    st.write(manifest["method"])
    config = manifest["config"]
    if config.get("backend") == "pretrained":
        st.caption(
            "Checkpoint provenance is recorded below. The supplied Baidu files match the downloaded stage copies."
        )
        if config["size"] < 512 or config["steps"] < 50:
            st.warning(
                "Reduced-setting execution test. These outputs can contain severe artifacts; paper-quality reproduction has not been established."
            )
    if state["state"] == "completed":
        st.success("Experiment completed. These are saved outputs, not a new run.")
    elif state["state"] in {"failed", "cancelled"}:
        st.error(state.get("error", state.get("reason", state["state"])))
    else:
        st.info(
            f"{state.get('stage', 'pipeline')} · {state.get('phase', state['state'])}"
        )
    if "step" in state:
        st.progress(
            min(1.0, state["step"] / state["total"]),
            text=f"{state['step']} / {state['total']} steps",
        )
    for column, name, label in zip(
        st.columns(3),
        ["input", "hazegen", "diffdehaze"],
        ["Input", "HazeGen", "DiffDehaze"],
        strict=True,
    ):
        with column:
            st.caption(label)
            final = run / f"{name}.png"
            frames = sorted(run.glob(f"{name}-*.png"))
            target = final if final.exists() else frames[-1] if frames else None
            if target:
                try:
                    st.image(Image.open(target).copy(), width="stretch")
                except OSError:
                    st.caption("Frame is being saved…")
            else:
                st.caption("Waiting for this stage")
            if final.exists():
                st.download_button(
                    f"Download {label}",
                    final.read_bytes(),
                    f"{name}.png",
                    "image/png",
                    key=f"observe-{name}",
                )
    with st.expander("Configuration and checkpoint provenance"):
        st.json(manifest)
    st.link_button("Start a new experiment", "/")

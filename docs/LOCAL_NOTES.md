# BTP · Haze Lab

A small, observable workspace for **Learning Hazing to Dehazing** (CVPR 2025).
The complete authors' repository is cloned, with its own git history, at
`upstream/Learning-Hazing-to-Dehazing/`. Local implementation is in `src/btp/`.
The authors call the second model **DiffDehaze**, rather than DiffGen.

## Kaggle reference workflow

See [the step-by-step Kaggle guide](KAGGLE.md), import
[the prepared notebook](../notebooks/kaggle_reference.ipynb), and upload
`artifacts/btp-kaggle.zip` as the private source dataset. Upload the three weight
files separately. The supplied Baidu stage checkpoints were hashed and match
the existing copies exactly; see `configs/baidu-verification.json`.

The Kaggle reference runner executes the original upstream scripts, including
the authors' short-edge resize, zero padding and final output resize. Its float32
CUDA reference path is separate from the reduced local preview/inference adapter.
No Kaggle GPU results or benchmark scores have been produced from this session.

## Paper source and experiment entry points

The full upstream clone matches `origin/main` at
`efac1ef6e69323ade585d80d9f78d8378baa50dc`, verified by fetching the remote
and checking for missing or changed tracked files. Its git history, license,
sample images, assets, and original configuration files are included.

All paths below are relative to `upstream/Learning-Hazing-to-Dehazing/`:

| Path | Purpose |
| --- | --- |
| `train_stage1.py` | Train HazeGen |
| `train_stage2.py` | Train DiffDehaze |
| `inference_stage1.py` | Generate hazy images with HazeGen |
| `inference_stage2.py` | Dehaze with the standard sampler |
| `inference_accsamp.py` | Dehaze with AccSamp |
| `configs/train/` | Original training configurations for both stages |
| `configs/inference/` | Original inference configurations |
| `diffbir/model/` | ControlLDM, ControlNet, UNet, VAE, and text encoder |
| `diffbir/sampler/` | Spaced sampling and AccSamp |
| `datasets.py`, `utils/` | Dataset loading, alignment, transmission, and fidelity utilities |
| `inputs/`, `assets/` | Supplied test images and paper illustrations |
| `requirements.txt`, `LICENSE` | Authors' dependencies and source license |

The local integration and experiment tools live in `src/btp/`; the upstream
tracked files remain unchanged. The isolated inference runtime does not include
the authors' complete training dependencies. Training also needs prepared data
and an appropriate GPU environment; no training has been launched.

## Open the live pipeline

```bash
uv sync
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Visit http://127.0.0.1:8501. Choose an image, the clean garden photograph bundled with scikit-learn, or a generated illustration,
select HazeGen, DiffDehaze, or both, then press **Run pipeline**. The three image
panels show the input and each stage; intermediate frames appear during execution.
Download the final images and run manifest directly from the viewer.
Use a genuinely hazy uploaded image when running DiffDehaze alone.

## What “smaller” means here

| Engine | Runs now? | What it does |
| --- | --- | --- |
| CPU physical preview | Yes, no weights | Atmospheric scattering for synthesis, dark-channel-prior restoration. This is **not** the learned paper pipeline. |
| Pretrained paper models | Requires all selected checkpoints | Original ControlLDM architecture with smaller images and fewer sampling steps; real decoded diffusion previews. |
| Tiny architecture smoke | Yes, isolated runtime | Original model components with 32 base channels, a small CLIP, and **random weights**. Tests execution and adapter behavior, not dehazing quality. |

The 256-pixel/10-step preset reduces image workload relative to the upstream
512-pixel/50-step inference. It **does not reduce model parameter count**, and
is not a trained compact student or a reproduction of the paper's metrics.
No training, dataset download, distillation, or benchmark evaluation has been run.
This machine has 8 GiB unified memory. Auto precision uses a mixed float16
weight policy on MPS/CUDA and float32 on CPU. The model is constructed on the
meta device and assigned memory-mapped checkpoint tensors to avoid duplicate
full-size allocations. Normalization, timestep embeddings, the UNet output, and
the VAE stay in float32; main diffusion and text weights use float16 on accelerators.
Start at 128 pixels, 5 steps, spaced sampling. Reduced pretrained inference has completed on this Mac, but quality has not
been reproduced. AccSamp needs extra gradient/decoder memory.
Tiny MPS/mixed16 adapter checks have passed for both spaced and AccSamp samplers. Use a larger-memory machine or
CUDA GPU if the full model exceeds local memory.

## Pretrained setup

The project keeps Python 3.12 and uv. The authors' inference code has a separate
Python 3.10 runtime; their CUDA-only requirements are **not** installed into the
main project. Native PyTorch SDPA replaces optional CUDA-only xformers. PyTorch 2.6 is used
for scoped safe handling of the original SD checkpoint's unused Lightning and
NumPy metadata; unrestricted pickle loading stays disabled. A singleton head
axis adapts the upstream SDPA tensors to the PyTorch 2.6 MPS backend. Checkpoint
tensors stream directly from memory-mapped storage to the selected device,
avoiding an intermediate CPU copy of the whole model. The text encoder is released after prompt encoding because denoising only needs the resulting embeddings.

```bash
uv venv --python 3.10 .venv-paper
uv pip install --python .venv-paper/bin/python -r configs/paper-requirements.txt
uv run btp doctor
```

Place these exact checkpoints in `weights/` (or set another folder in the UI):

1. `v2-1_512-ema-pruned.ckpt` — SD 2.1 base, shared by both stages.
2. `stage1.pt` — the authors' HazeGen ControlNet.
3. `stage2.pt` — the authors' DiffDehaze ControlNet.

The [original SD link](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
returns HTTP 401. A public [community mirror](https://huggingface.co/sd-research/stable-diffusion-2-1-base)
is linked in [upstream issue 12](https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing/issues/12).
The approved download is 5,214,865,159 bytes; partial data stays in `.ckpt.part`.
To resume it and verify its Hugging Face LFS SHA-256:

```bash
curl -fL --retry 2 -C - 'https://huggingface.co/sd-research/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt' -o weights/v2-1_512-ema-pruned.ckpt.part
uv run python scripts/verify_sd.py
```

The trained stage weights are in the [authors' Baidu folder](https://pan.baidu.com/s/1vbxEwftJC9nUaMXJ-t9sww?pwd=8egg),
extraction code `8egg`. Download `stage1.pt` and `stage2.pt` manually if Baidu
requires your browser/account. A broader Hugging Face search also found a [community stage-weight mirror](https://huggingface.co/DJKAD/Learning-Hazing-to-Dehazing).
These files are pinned to revision `f6d334fcb867542118f6bc64cdf8c7812581fc37`.
Their LFS SHA-256 hashes verify downloaded bytes. On November 4, 2026, both files were also confirmed byte-identical to the user-supplied Baidu files. Source URLs and hashes are in
`configs/checkpoints.json` and copied into each model run's manifest.
The UI blocks pretrained runs when required files are missing.

```bash
# Resumes partial files; about 8.12 GB total. Only run after approving that download.
uv run python scripts/download_weights.py
# Or verify completed downloads separately:
uv run python scripts/verify_weights.py
```
No substitute, random, or physical-baseline outputs are presented as pretrained results.

```bash
uv run btp run --backend pretrained --input your-clean-image.png --stage both --size 256 --steps 10 --seed 42
uv run btp run --backend pretrained --input your-hazy-image.png --stage diffdehaze --sampler accsamp --size 256 --steps 10 --seed 42
```

HazeGen uses the original hazy prompt and blended conditional/unconditional
prediction with `w=0.85`. DiffDehaze uses `remove dense fog`. The spaced sampler
uses the original upstream equations. AccSamp retains `tau=0.8`, `omega=0.6`,
and fidelity strength `s=0.1`; step counts must be divisible by 5 because its
upstream implementation selects boundary steps by exact fractions.
The adapter moves diffusion buffers to the selected accelerator and pads AlignOp inputs to its patch grid before cropping back, preventing uncovered right/bottom strips. The cloned upstream source remains unchanged.
Live neural previews decode predicted-clean latents; they are intermediate
estimates and can look poor before the final step. Preview decoding adds work.

## Smoke runs and validation

```bash
uv run btp smoke
PYTHONPATH=src uv run --no-project --python .venv-paper/bin/python python -m btp.model_smoke
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The tiny model check first performs two CPU sampling steps. It then creates
explicitly random test checkpoints inside its run directory and exercises the
full loading/live-frame adapter for both stages with CPU spaced and AccSamp samplers, plus MPS mixed precision when available.
Those outputs only test plumbing; they are not useful dehazed images.
Tests cover seeded reproducibility, black/constant images, invalid configurations,
aspect ratios, selected-stage checkpoint requirements, saved manifests, and the
live viewer's run and missing-weight paths. Ruff excludes vendored upstream code.

Every experiment creates a unique directory in `runs/` containing input, final
outputs, intermediate PNGs, and atomic status updates. `manifest.json` records
settings, seed, input SHA-256, dataset identity, model/library details when used,
and the upstream revision. No dataset split or fitted preprocessing is used in
single-image inference. If BTP is only inside a parent git repository rather than
its own repository, `project_commit` is null; the unrelated parent revision is
not reported as a BTP experiment commit. All runs/weights are excluded from git.

For an actual compact trained model, the next research step is distilling these
teacher checkpoints into a smaller architecture. Reducing channels breaks teacher
checkpoint compatibility, so that requires separate training, a fixed dataset
split, and evaluation; it is not claimed as completed here.

Source: https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing
Paper: https://arxiv.org/abs/2503.19262
The upstream Apache-2.0 license and pretrained model license terms still apply.

## Observed validation results and limitations

- CPU physical preview: completed; all outputs explicitly labeled as a baseline.
- Tiny random-weight architecture and adapter: CPU and MPS, spaced and AccSamp, passed.
- Pretrained 128px/5-step roundtrip: completed on MPS; severe image artifacts.
- Pretrained 256px/20-step photographic roundtrip: completed on MPS; substantial color/structure artifacts.
- Native 512px/50-step HazeGen: cancelled because inference was impractically slow on this 8 GiB machine. Native-quality output is **not verified**.
- Initial compatibility failures (mapped Path arguments, restricted legacy metadata, accelerator diffusion buffers, and MPS SDPA shape) were corrected and retested.
- No CUDA run, training, distillation, or benchmark evaluation was performed.

Do not interpret a completed run or matching checkpoint tensor shapes as a
successful paper reproduction. The stage checkpoint files were subsequently verified byte-for-byte against the user-supplied Baidu downloads on November 4, 2026. A capable GPU and faithful preprocessing are needed for further reproduction work; a smaller trained
network would require separate distillation or training.

To observe an existing CLI/saved experiment without launching another job, open
`http://127.0.0.1:8501/?run=EXPERIMENT_DIRECTORY_NAME`. The page refreshes images,
loading phases, and status every second. The verified completed reduced-size
photographic experiment is `20261104T031956-3d85da89`; inspect its manifest to see
exact settings and checkpoint provenance. Stopping/rerunning a viewer-launched
experiment terminates that viewer's own worker process group.

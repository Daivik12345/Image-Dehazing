# Image Dehazing | BTP

A workspace for experimenting with **Learning Hazing to Dehazing** (CVPR 2025),
with HazeGen, DiffDehaze, a live local viewer, and a Kaggle CUDA reference notebook.

## Progress update: November 6, 2026

We completed the full pretrained pipeline on Kaggle using the **T4 x2 accelerator configuration**. The code uses **one Tesla T4**; GPU memory is not pooled across the two cards.

- Completed HazeGen, standard DiffDehaze, AccSamp, and the clean-to-hazy-to-dehazed roundtrip in notebook version 7.
- Downloaded the outputs, checked completion records and low-memory manifests, and inspected the generated images.
- Saved version 8 with safe rerun names: rerunning a cell creates a new case directory and preserves earlier results and logs.
- Preserved the previous baseline notebook alongside the new `btp-baseline+` notebook.

### Measured GPU memory

These runs used the pretrained checkpoints, **float32**, **seed 42**, the original **50-step schedule**, and upstream image resizing/padding.

| Pipeline case | Image size before padding | Denoising calls | Peak allocated GPU memory | Peak reserved GPU memory |
|---|---|---|---|---|
| HazeGen | 767 x 512 | 50 | 7.597 GiB | 7.764 GiB |
| Standard DiffDehaze | 573 x 512 | 50 | 7.205 GiB | 7.342 GiB |
| AccSamp | 573 x 512 | 40 | 8.887 GiB | 10.936 GiB |
| Full roundtrip: AccSamp stage | 767 x 512 | 40 | 11.769 GiB | 14.049 GiB |

The counters measure PyTorch memory within each inference process. AccSamp uses 40 denoising calls under the original 50-step schedule. The roundtrip row measures its dehazing stage; HazeGen and dehazing execute in separate processes.

### How we resolved the memory failure

The original float32 AccSamp run exhausted a T4 during VAE fidelity guidance. The opt-in `--low-memory` mode:

- Moves CLIP to CPU after encoding the prompts.
- Freezes model parameters while preserving gradients with respect to the latent image.
- Moves the idle UNet and ControlNet to CPU during VAE guidance, then restores them to GPU.
- Disables diagnostic previews for the measured runs to avoid extra VAE decodes.

These changes reduce GPU residency without reducing parameter count. They add CPU/GPU transfer overhead. The default reference runner retains its original execution mode unless `--low-memory` is supplied. The tiny CPU smoke check confirmed matching outputs and latent gradients; full-model numerical equivalence has not been established.

[Open the live btp-baseline+ notebook](https://www.kaggle.com/code/cubingultimate/btp-baseline) or read the [Kaggle guide](docs/KAGGLE.md) for setup and limitations.

## Evaluation and offloading experiments

The separate [btp-eval-optimised notebook](notebooks/btp-eval-optimised.ipynb) adds a paired NH-HAZE evaluation harness, fixed development/held-out crops, PSNR/SSIM scoring, structured failures, synchronized sampling timing, memory/transfer counters and a separate profiler run. Baseline+ remains unchanged.

Start with the four-job M0/M1 pilot and read [the evaluation walkthrough](docs/EVALUATION.md). Repeatability and selective-placement configurations are included for later controlled experiments. The first panel uses 512×512 internal diagnostic crops; it is not the official paper benchmark. Resident model reuse and asynchronous transfers remain staged follow-ups.

## Start on Kaggle

1. Read the [step-by-step Kaggle guide](docs/KAGGLE.md).
2. Use [btp-baseline+](notebooks/btp-baseline-plus.ipynb) for the tested low-memory T4 workflow. The previous [reference baseline](notebooks/kaggle_reference.ipynb) remains available unchanged.
3. Upload `stage1.pt`, `stage2.pt`, and `v2-1_512-ema-pruned.ckpt` as a private
   Kaggle dataset. See [checkpoint instructions](weights/README.md).
4. Build the source ZIP with `uv run python -m btp.kaggle` and upload
   the archive as a second private dataset. Rename it to `btp-source.bundle` before upload to preserve nested compressed assets; the notebook opens this file as a ZIP.
5. Enable an available NVIDIA GPU and Internet, attach both datasets, and run
   notebook cells in order. The notebook verifies hashes and displays outputs.

The reference workflow runs original upstream scripts with their short-edge
resize, zero padding, 50-step sampling, and float32 CUDA execution. It covers
HazeGen, spaced DiffDehaze, AccSamp, and the complete roundtrip.

## Repository layout

```text
app.py                 Local Streamlit viewer
src/btp/               Importable pipeline, reference runner and packaging code
notebooks/             Preserved baselines and separate evaluation notebook
configs/               Runtime dependencies, presets and checkpoint hashes
docs/                  Kaggle walkthrough and local experiment notes
scripts/               Download, verification and smoke-check utilities
tests/                 Behavioral and packaging tests
examples/              Attributed clean example photograph
upstream/              Complete vendored paper repository and license
weights/               Local checkpoints (excluded from Git)
runs/                  Generated experiments (excluded from Git)
```

The upstream source is vendored as ordinary files, so cloning this repository
includes its models, training/inference scripts, configurations, and examples.
Its original revision and file inventory are recorded in
[upstream/SOURCE.json](upstream/Learning-Hazing-to-Dehazing/SOURCE.json).
There is no nested Git repository or submodule to initialize.

## Local setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run btp smoke
uv run streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. The default CPU physical preview is explicitly labeled;
it is not a learned HazeGen/DiffDehaze result. The isolated inference runtime and
local limitations are detailed in [local notes](docs/LOCAL_NOTES.md).

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Evaluation status and next steps

We verified pipeline execution on two example source images. Quantitative paper-level reproduction remains pending.

1. Evaluate on paired real-world benchmarks with fixed splits and seeds, recording PSNR/SSIM and perceptual metrics.
2. Compare low-memory execution against the original reference on hardware that can run both, measuring quality and runtime.
3. Test additional image sizes; the roundtrip already approaches the T4 capacity in reserved memory.
4. Investigate a compact student model if deployment requires fewer parameters or faster inference.

Local reduced-resolution experiments on the 8 GiB Mac showed artifacts; use the Kaggle workflow for the tested full pipeline. We have not trained a new model or launched paid compute. The existing results establish example-run feasibility, not benchmark scores or guarantees for larger inputs.

Weights, environments, private datasets, caches and generated run outputs are
not committed. The approximately 8.12 GB of checkpoints must be supplied separately.

## Attribution

Original paper and source: [ruiyi-w/Learning-Hazing-to-Dehazing](https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing).
The original source's [Apache-2.0 license](upstream/Learning-Hazing-to-Dehazing/LICENSE)
is preserved. The example photograph attribution is in [examples/README.md](examples/README.md).

## Notebook versions

- [Previous baseline](notebooks/kaggle_reference.ipynb): preserved original notebook.
- [btp-baseline+](notebooks/btp-baseline-plus.ipynb): CPU offloading with float32, original geometry and 50 steps; safe rerun output names and robust source restoration.
- [Live Kaggle notebook](https://www.kaggle.com/code/cubingultimate/btp-baseline): display title `btp-baseline+`.

The plus notebook attaches `cubingultimate/btp-source` and `cubingultimate/btp-paper-weights-2`. These datasets are private; other users must supply their own source and checkpoints. The updated runner and source bundler are included here. See [the guide](docs/KAGGLE.md) for measured memory and limitations.

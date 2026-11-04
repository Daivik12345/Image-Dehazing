# Image Dehazing — BTP

A workspace for experimenting with **Learning Hazing to Dehazing** (CVPR 2025),
with HazeGen, DiffDehaze, a live local viewer, and a Kaggle CUDA reference notebook.

## Start on Kaggle

1. Read the [step-by-step Kaggle guide](docs/KAGGLE.md).
2. Import [kaggle_reference.ipynb](notebooks/kaggle_reference.ipynb).
3. Upload `stage1.pt`, `stage2.pt`, and `v2-1_512-ema-pruned.ckpt` as a private
   Kaggle dataset. See [checkpoint instructions](weights/README.md).
4. Build the source ZIP with `uv run python -m btp.kaggle` and upload
   `artifacts/btp-kaggle.zip` as a second private dataset.
5. Enable an available NVIDIA GPU and Internet, attach both datasets, and run
   notebook cells in order. The notebook verifies hashes and displays outputs.

The reference workflow runs original upstream scripts with their short-edge
resize, zero padding, 50-step sampling, and float32 CUDA execution. It covers
HazeGen, spaced DiffDehaze, AccSamp, and the complete roundtrip.

## Repository layout

```text
app.py                 Local Streamlit viewer
src/btp/               Importable pipeline, reference runner and packaging code
notebooks/             Kaggle reference notebook
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

## Reproduction status

- The supplied Baidu stage files match the recorded checkpoint hashes.
- Local reduced-resolution pretrained runs completed but showed severe artifacts.
- Native inference was impractically slow on the local 8 GiB Mac.
- The Kaggle notebook is prepared; Kaggle GPU results and paper benchmark scores
  have **not** been produced or claimed.
- No training or paid compute has been launched.

Weights, environments, private datasets, caches and generated run outputs are
not committed. The approximately 8.12 GB of checkpoints must be supplied separately.

## Attribution

Original paper and source: [ruiyi-w/Learning-Hazing-to-Dehazing](https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing).
The original source's [Apache-2.0 license](upstream/Learning-Hazing-to-Dehazing/LICENSE)
is preserved. The example photograph attribution is in [examples/README.md](examples/README.md).

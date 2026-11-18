# Image Dehazing | BTP

A workspace for experimenting with **Learning Hazing to Dehazing** (CVPR 2025),
with HazeGen, DiffDehaze, a live local viewer, and a Kaggle CUDA reference notebook.

## OTS dataset and first student profile: November 18, 2026

Created the private [BTP student training pilot](https://www.kaggle.com/datasets/cubingultimate/btp-student-training-pilot) from the user-supplied OTS_BETA archive: **90 training / 10 corpus-validation scenes**, seed42, one haze variant per source. All four haze parts were inventoried. RGB/minimum-size exclusions are recorded before sampling; native dimensions are preserved. Known NH-HAZE identity overlap checks passed. Related-viewpoint review is incomplete, so this is not generalization evidence.

The approved [student pilot notebook](https://www.kaggle.com/code/cubingultimate/btp-student-pilot) **version2 completed** on one Tesla T4:

| Measured item | Result |
|---|---:|
| Optimizer updates / microbatch exposures |100 /400|
| Training wall time, excluding setup/export |14.205s|
| Peak PyTorch allocated / reserved GPU memory |364.45 /400MiB|
| Student parameters |1,724,275|
| Exported inference checkpoint |6,964,282bytes|

Checkpoint hashes, changed weights and CPU inference reload were verified. Validation is scheduled every250updates; this export is **last/unselected**, not a validated dehazing replacement. No quality gain, teacher distillation result or held-out score is claimed. No continuation or teacher-generation job has been launched.

Read the [intake/profile report](docs/research/OTS_PILOT_INTAKE.md) and [saved training record](docs/research/evidence/a0-profile-kaggle-v2.json). Implementation commit: `15b4ba8`; research source dataset version2. Existing baselines remain unchanged. The repository student notebook stays opt-in; the saved Kaggle version2 reflects the explicitly approved bounded run.

## Multi-axis development: November 17, 2026

Implemented the first research foundation for a compact deployment model, native-size inference, and guidance-quality experiments. **These are implemented and CPU-tested capabilities, not demonstrated restoration gains or a trained student release.**

| Track | New implementation | Current evidence / next gate |
|---|---|---|
| Compact student | NAFNet-style width-16 network: **1,724,275 parameters**; reconstruction/SSIM and optional teacher supervision; seeded, bounded A0/A1 pilot CLI; hash-checked export/inference | Tiny synthetic CPU training/resume tests pass. Separate training corpus and approved GPU pilot still required |
| Native resolution | Complete overlapping tile coverage, positive blending weights, streaming CPU fusion, explicit workspace budget, optional bounded global appearance correction | Identity/odd-size/boundary tests pass. Trained-checkpoint native GPU quality/seam/memory evaluation pending |
| Guidance quality | Detached confidence weighting, matched-mean uniform control, optional norm cap, opt-in resident integration | CPU gradient/control tests pass. Calibration and CUDA quality experiments pending |
| Reusable teacher | Persistent M0 model load, fresh per-image sampler/conditioning/guidance, runtime/source identity, bounded target shards with provenance checks | CPU lifecycle and upstream-gradient checks pass. Cold/resident and A-B-A CUDA equivalence pending |
| Reproducibility | Frozen protocols, data leakage checks, native/larger NH-HAZE manifests, evidence auditing and paired scene bootstrap | Original **10 development / 45 test** split preserved; no held-out tuning or new held-out result |

The student targets deployment-time **DiffDehaze**; it does not compress HazeGen. Its float32 parameters occupy 6,897,100 bytes, excluding buffers/serialization. No compression ratio is claimed until the required teacher deployment modules are counted and quality is measured.

### What the latest completed validation actually found

The supplied session archive contains **20 completed inference jobs**: six repeatability, eight placement, and six cache jobs. Raw evidence retains **M0 + guidance flushing** as the conservative reference. M2-R failed the unchanged observed-variation gate on one scene. Its tiny pixel differences do not establish visibly worse restoration, but cannot be silently accepted under a different threshold.

At M0 placement, the two-scene cache screen measured median sampling times of **139.44 s (guidance), 144.49 s (phase), and 146.22 s (never)**. Phase passed consistency but was slower; never failed consistency. There is no confirmed cache speedup, repeated-finalist result or new paper-level score. [Teacher lock](configs/research/teacher-lock.json) and [archive inventory](docs/research/evidence/validation-session-inventory.json) retain provenance.

### New workflows and instructions

- [Data audit notebook](notebooks/btp-data-audit.ipynb): independent training corpus audit.
- [Teacher targets notebook](notebooks/btp-teacher-targets.ipynb): bounded training-only target generation, disabled until teacher validation and budget approval.
- [Student pilot notebook](notebooks/btp-student-pilot.ipynb): CPU smoke, real-data dry audit, explicit opt-in profiling/training.
- [Guidance quality notebook](notebooks/btp-guidance-quality.ipynb): separate B0/confidence/uniform experiments after calibration.
- [Global tiles notebook](notebooks/btp-global-tiles.ipynb): native student inference with a supplied hash-verified checkpoint.
- [Execution guide](docs/RESEARCH_WORKFLOWS.md), [training dataset choices and next steps](docs/research/TRAINING_DATA_NEXT_STEPS.md), [detailed plan](docs/superpowers/plans/2026-11-16-multi-axis-btp-development.md), and [implementation ledger](docs/research/EXECUTION_PROGRESS.md).

### Kaggle publication: November 18, 2026

All five new notebooks are saved privately as **version 1, without running**. The [private research source dataset](https://www.kaggle.com/datasets/cubingultimate/btp-research-source) is ready. These are accessible to the owning Kaggle account; no GPU experiments were launched.

| Notebook | Purpose | Kaggle link |
|---|---|---|
| btp-data-audit | Separate training-corpus audit | [Open](https://www.kaggle.com/code/cubingultimate/btp-data-audit) |
| btp-teacher-targets | Bounded teacher targets after equivalence validation | [Open](https://www.kaggle.com/code/cubingultimate/btp-teacher-targets) |
| btp-student-pilot | CPU smoke, dry audit and opt-in A0/A1 pilot | [Open](https://www.kaggle.com/code/cubingultimate/btp-student-pilot) |
| btp-guidance-quality | Calibrated B0/confidence/uniform comparison | [Open](https://www.kaggle.com/code/cubingultimate/btp-guidance-quality) |
| btp-global-tiles | Native tiled inference from a trained student | [Open](https://www.kaggle.com/code/cubingultimate/btp-global-tiles) |

The source snapshot corresponds to implementation commit `0b92c91`; its file hashes are inside the bundle. Kernel upload metadata is checked into `configs/kaggle-*/kernel-metadata.json`. Accelerator defaults are CPU; enable a T4 only when the relevant experiment is approved. Model weights/training data must be attached separately when those stages are ready.

These saved versions are not completed Kaggle runs. Previous notebooks remain unchanged. Subsequent A2/generated-data ablations, full confirmation/release notebooks, UI integration and a trained model card depend on validated artifacts and remain pending.

### Validation and limits

The base test suite passed **167 tests**, with **20 optional Torch checks skipped** in the base environment. The existing isolated Torch runtime passed **11 student/CLI tests, 7 guidance tests and 3 resident-gradient tests**. Ruff lint and formatting checks pass. No new GPU jobs, large dataset downloads, paid compute or long training were launched.

Next: acquire an appropriate paired training release, audit source-scene splits, approve the first 100-update profile within the fixed pilot budget, and validate resident equivalence before generating teacher targets. Target improvements across compression and native-resolution capability require independent measured evidence; targets are not results.

## Validation continuation: November 15, 2026

Added a separate [btp-validation notebook](https://www.kaggle.com/code/cubingultimate/btp-validation), its [private source dataset](https://www.kaggle.com/datasets/cubingultimate/btp-validation-source), and explicit evidence selection (`btp.workflow`). It recomputes eligibility from saved records, persists stage paths and records decision hashes. This addresses lost kernel variables and makes rejected selections explain their reasons without weakening the criteria. Existing notebooks are unchanged.

The earlier user-supplied two-scene placement table showed M2-R sampling averaging 57.92 s versus 146.07 s for M0 (about 60% lower), but the original archive was not saved and a later selection failed. Treat this as preliminary screening, not a confirmed optimization. This historical status is superseded by the November 17 archive review above; model-reuse/compression GPU validation remains pending. See [continuation instructions](docs/VALIDATION.md).

## Development update: November 8, 2026

Added the separate [btp-offload-cache workflow](notebooks/btp-offload-cache.ipynb) for **selective UNet/ControlNet offloading** and **independent cache-flushing experiments**. It measures M0 repeatability, screens placements, then supports opt-in cache comparison and repeated confirmation. Variant-specific results prevent different placements/cache policies from being mixed. Failed or numerically drifting candidates cannot pass screening eligibility.

The original baseline, baseline+ and evaluation notebooks are unchanged. The new workflow has its own [Kaggle notebook](https://www.kaggle.com/code/cubingultimate/btp-offload-cache) and [private source dataset](https://www.kaggle.com/datasets/cubingultimate/btp-offload-cache-source), reusing the existing weights and NH-HAZE crops. See [the run guide](docs/OFFLOAD_CACHE.md) for controls, stage budgets and interpretation.

**Status:** implementation and CPU validation; new placement/cache variants have not been benchmarked on GPU. No speedup or quality gain is claimed. Default notebook execution runs six baseline repeatability jobs and eight placement-screening jobs; cache and confirmation stages are opt-in.

## Progress update: November 7, 2026

The first paired evaluation completed on **one Tesla T4**: all four M0/M1 evaluation jobs and one separate profiling job succeeded, with no OOMs or timeouts. The panel used **NH-HAZE scenes 08 and 13**, paired **512 x 512 center crops**, float32, seed 42 and one repeat per policy. AccSamp used the original 50-step schedule with 40 actual denoising calls.

| Unprofiled pilot metric | M0: current low-memory baseline | M1: conditional offloading |
|---|---:|---:|
| Mean PSNR | 14.5456 dB | 14.5456 dB |
| Mean SSIM | 0.3644 | 0.3644 |
| Median sampling time per image | 140.41 s | 139.95 s |
| Median cold-process time per image | 191.11 s | 190.20 s |
| Maximum lifetime allocated GPU memory | 7.93 GiB | 7.93 GiB |
| Maximum lifetime reserved GPU memory | 9.65 GiB | 9.65 GiB |

**Finding:** M1 produced nearly identical outputs but no meaningful speedup was established. Both policies performed the same 30 active guidance/offloading cycles; there were no inactive guidance transfers for M1 to eliminate. The small timing difference requires repeatability measurements before interpretation.

**Progress achieved:** a working paired evaluation and profiling framework for quality, runtime, memory and failure comparisons. The five processes took approximately **17.1 minutes combined**, excluding environment setup and notebook packaging. The table excludes the profiling job. Evidence comes from the supplied `btp-eval-results.zip` summaries, per-image records and telemetry.

These are **two-image internal development results**, not official NH-HAZE benchmark scores or paper-level reproduction. LPIPS and held-out evaluation remain pending. No smaller model or improved dehazing quality is claimed.

## Earlier milestone: November 6, 2026

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

The full pipeline runs on T4, and the first paired development pilot is complete. Our next milestone is a reproducible runtime improvement while preserving output quality and completion coverage.

1. Repeat M0 on the fixed development images to establish timing and numerical variability; inspect the saved profiler trace for transfer and guidance costs.
2. Screen M2-U (UNet-only offload), M2-C (ControlNet-only offload) and M2-R (both resident), retaining OOMs and other failures. Repeat viable candidates before choosing a placement.
3. Test cache flushing independently of the selected placement. The new `--cache-policy` independently selects guidance/phase/never flushing; legacy M3 remains M1 without per-guidance flushing.
4. Target at least a 10% reduction in paired median sampling time, confirmed across repeats, without new failures or unexplained output drift. This is an internal decision target, not a promised gain.
5. After selecting a stable placement, implement model reuse across images with a resident runner. Guidance changes and compact-student training remain later research stages.

See the [evaluation walkthrough](docs/EVALUATION.md), [evaluation and offloading plan](docs/research/EVALUATION_AND_OFFLOADING_GUIDE_2026-11-06.md), and [broader research roadmap](docs/research/BASELINE_IMPROVEMENTS_2026-11-04.md). Larger/native images and final held-out evaluation need separate validation. We have not trained a new model or launched paid compute.

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

## Kaggle notebooks, source bundles and datasets

Kaggle notebooks and datasets below are private and require appropriate access. Source bundles contain code, not pretrained weights. The two original baseline notebook files remain preserved in this repository; the live baseline URL currently serves baseline+.

| Resource | Link | Purpose |
|---|---|---|
| Full-pipeline notebook: btp-baseline+ | [Open notebook](https://www.kaggle.com/code/cubingultimate/btp-baseline) | HazeGen, spaced DiffDehaze, AccSamp and roundtrip inference |
| Evaluation notebook: btp-eval-optimised | [Open notebook](https://www.kaggle.com/code/cubingultimate/btp-eval-optimised) | Paired quality/runtime comparisons and profiling |
| Optimization notebook: btp-offload-cache | [Open notebook](https://www.kaggle.com/code/cubingultimate/btp-offload-cache) | Selective placement and independent cache experiments |
| Optimization source: btp-offload-cache-source | [Open dataset](https://www.kaggle.com/datasets/cubingultimate/btp-offload-cache-source) | Isolated source bundle for the new workflow |
| Baseline source: btp-source | [Open dataset](https://www.kaggle.com/datasets/cubingultimate/btp-source) | Preserved source bundle used by baseline+ |
| Evaluation source: btp-eval-source | [Open dataset](https://www.kaggle.com/datasets/cubingultimate/btp-eval-source) | Separate evaluation code bundle |
| Current weights: btp-paper-weights-2 | [Open dataset](https://www.kaggle.com/datasets/cubingultimate/btp-paper-weights-2) | Checkpoint dataset attached to both workflows |
| Earlier weights: btp-paper-weights | [Open dataset](https://www.kaggle.com/datasets/cubingultimate/btp-paper-weights) | Existing earlier upload; current notebooks use weights-2 |
| Evaluation images: btp-eval-nh-haze | [Open dataset](https://www.kaggle.com/datasets/cubingultimate/btp-eval-nh-haze) | 55 paired crops with hashes and a fixed internal development/held-out split |
| BTP collection | [Open collection](https://www.kaggle.com/work/collections/19537288) | Project organization; collection membership is managed manually |

Repository notebook copies: [previous baseline](notebooks/kaggle_reference.ipynb), [baseline+](notebooks/btp-baseline-plus.ipynb), [btp-eval-optimised](notebooks/btp-eval-optimised.ipynb), and [btp-offload-cache](notebooks/btp-offload-cache.ipynb).

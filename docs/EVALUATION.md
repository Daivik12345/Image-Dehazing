# Paired evaluation and offloading on Kaggle

Use [btp-eval-optimised](https://www.kaggle.com/code/cubingultimate/btp-eval-optimised) for evaluation. Its [repository copy](../notebooks/btp-eval-optimised.ipynb) was derived from baseline+. The existing baseline notebooks and `btp-source` dataset remain unchanged.

## Run the first experiment

1. Open the new notebook and select **GPU T4 x2**, Internet on. The worker uses GPU 0; two T4 cards do not combine into one larger VRAM pool. Save a version with outputs before ending the session. Persistence is optional because every run restores verified source and creates a new result directory.
2. Attach the three private datasets: `cubingultimate/btp-paper-weights-2`, `cubingultimate/btp-eval-source`, and `cubingultimate/btp-eval-nh-haze`.
3. Run cells in order. Setup verifies source files and checkpoint hashes, creates the same Python 3.10/PyTorch 2.6 inference environment, and runs CPU smoke checks.
4. The data cell restores all 55 paired crops and verifies the frozen manifest. The default pilot uses **nh-haze-08** and **nh-haze-13**, selected by scene ID before observing predictions.
5. Run the pilot: two scenes × M0/M1 × seed 42 × one repeat = **four fresh-process jobs**. Each worker is capped at 480 seconds. The parent continues after an inference failure and preserves its log/status.
6. Inspect the tables and hazy/prediction/clean strips. A separate **one-job M0 profile** runs when `RUN_PROFILE=True` and M0 completed. Profiled latency is excluded from the pilot table.
7. Download `btp-eval-results.zip` from the final cell or Kaggle's Output panel. Inspect `summary.json`, `jobs.jsonl`, `per_image.csv`, per-job manifests, logs, telemetry and `profile.json`.

A completed notebook is not sufficient evidence that every job succeeded: inspect completion counts and failure statuses. The first pilot validates the workflow. It cannot establish a speedup, numerical equivalence or paper-level reproduction.

## Fixed data and metric contract

The official [NH-HAZE archive](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) contains 55 paired outdoor scenes. This workflow applies the same center 512×512 crop to each hazy/clean pair, without resizing. Original file hashes, cropped hashes, source dimensions and crop coordinates are frozen in [the manifest](../configs/evaluation/nh_haze_manifest.json).

The internal split hashes `42:source_id`, sorts the hashes, assigns the first 10 scenes to development and holds out the remaining 45. **This is not the official NTIRE challenge split.** It is a systems diagnostic panel, not a claim about generalization; overlap with the pretrained model's training data has not been established. Never tune a placement policy using held-out quality.

Metrics use decoded RGB values in [0,1], identical dimensions and no alignment, resizing, luma conversion or image-border shaving. PSNR uses RGB mean squared error. SSIM uses an 11×11 Gaussian window, sigma 1.5 and population covariance, averaging the valid window centers. A seeded numerical fixture matches scikit-image 0.24.0 exactly. Exact-match PSNR is JSON null with `identical=true`; aggregate PSNR is null if any contributing prediction is exact. Failed outputs are excluded from quality means but retained in completion counts.

Results average within a source scene before averaging scenes. Candidate-minus-M0 comparisons match scene, seed, repeat and sampler. They report finite metric-pair counts, output MAE/max error, and source-scene bootstrap intervals when at least two scenes are available. Small-panel intervals are diagnostic. A p95 is omitted below 20 scene observations. Different geometry tiers must run separately.

HazeGen remains available through the preserved full-pipeline notebook. This paired restoration harness evaluates `accsamp` and `spaced`; it does not mislabel clean-vs-generated-haze PSNR as HazeGen quality.

## Implemented placement policies

| Policy | Behavior |
|---|---|
| R0 | Original parameter gradients/residency; may OOM on T4 |
| M0 | Existing low-memory baseline: frozen weights, CLIP on CPU after conditioning, both denoisers offloaded for each guidance invocation |
| M1 | M0 placement only while the guidance window is active and repeat count is positive |
| M2-U | Offload only UNet during active guidance |
| M2-C | Offload only ControlNet during active guidance |
| M2-R | Freeze weights and offload CLIP; keep both denoisers resident |
| M3 | Specific allocator ablation: M1 placement without per-guidance `empty_cache`; it is not a claim that M1 is the winning placement |

All retain original checkpoints, float32, 50-step schedule, prompts and guidance hyperparameters. Output directories retain the actual upstream YAML and hashes. No model parameters are removed. These policies trade GPU residency against transfer overhead; selective residency may OOM.

The CLI rejects unsupported resident execution. M4 model reuse and asynchronous/pinned-memory transfers remain follow-up experiments after choosing a fixed placement. Repeated CLI launches are cold processes, even when the filesystem cache is warm.

## Measurements and artifacts

- `cold_process_seconds`: process launch through exit, including loading and checkpoint hashing. Each job records launch order; OS caches can warm across launches.
- `sampling_seconds`: synchronized wall time around the sampler, including blocking transfers and existing progress logging. Conditioning and final output decode are outside this boundary.
- `telemetry.json`: separate lifetime and reset sampling GPU peaks, allocation at sampling entry, host peak RSS, actual sampler/forward/backward/VAE calls, active/inactive guidance calls, and module transfer payload estimates.
- `phase_cpu_enqueue_seconds`: nested, non-additive CPU enqueue times. Use the separate profiler trace for CUDA events and transfer analysis. Tensor payload counts are not measured PCIe bandwidth.
- `runtime-packages.txt`, `nvidia-smi.txt`, `notebook-provenance.json`, dataset/source manifests and per-job YAML: environment, source, weights, data, prompts and geometry provenance.
- Failures: `oom`, `timeout`, `failed`, `runner_error`, or `invalid_output`. No automatic image shrinking, step reduction or retry changes the comparison.

The source snapshot identifies exact bundled file bytes even when Kaggle has no Git checkout. Local working changes are not represented as an invented commit. Keep the dataset version and notebook version with exported artifacts.

## Repeatability and larger experiments

The supplied JSON configurations are intentionally separate:

| Configuration | Work |
|---|---|
| `pilot.json` | 4 unprofiled development jobs |
| `profile.json` | 1 diagnostic job |
| `repeatability.json` | 12 jobs: M0/M1, two scenes, three repeats |
| `placement.json` | 30 jobs: M0/M2-U/M2-C/M2-R/M3, two scenes, three repeats |

After reviewing pilot duration and quota, call `run_evaluation('repeatability')` in the notebook. Inspect M0-vs-M0 repeat variation before interpreting candidate differences. Do not launch the full placement matrix automatically; it deliberately keeps OOM candidates visible. Compare matched successes and failure rates. The runner does not automatically accept a policy or alter quality thresholds.

For a prepared CUDA environment, the underlying command is:

```bash
uv run --no-project --python /tmp/btp-kaggle-env/bin/python python -m btp.benchmark \
  --manifest /path/to/nh_haze_manifest.json \
  --config configs/evaluation/pilot.json \
  --weights /path/to/weights \
  --output /path/to/new-experiment
```

Set `PYTHONPATH` to the repository's `src` directory. Every output directory must be new; the notebook uses nanosecond suffixes. To prepare the dataset locally from the official downloaded ZIP:

```bash
uv run python -m btp.datasets \
  --archive data/nh-haze/NH-HAZE.zip \
  --output data/evaluation/nh-haze-512
```

## Optional LPIPS

LPIPS is disabled in the default experiments. To opt in, install `lpips==0.1.4` into the separate inference/evaluation runtime and set `lpips: true` in a copied matrix configuration. Its first use may download AlexNet weights. Scoring runs on CPU after all inference processes finish; loaded metric tensor hashes and package version are recorded in `lpips-provenance.json`. LPIPS installation/download/scoring failures are retained separately and do not erase PSNR/SSIM or inference records. Missing LPIPS is not a zero score. The optional LPIPS path has not been validated against real model weights in this pilot.

See the [research plan](research/EVALUATION_AND_OFFLOADING_GUIDE_2026-11-06.md) for acceptance criteria and the staged resident/async follow-ups. The historical plan is preserved; this page documents the implemented commands and limits.

Current aggregation limitation: use one record/crop per source scene, as in the frozen NH-HAZE dataset. Multiple differently named crops of one source are not yet clustered together by the summary/bootstrap implementation.

## Project collection

The [BTP collection](https://www.kaggle.com/work/collections/19537288) is the intended destination for project workflows. The user will manage collection additions manually; publishing does not modify the collection.

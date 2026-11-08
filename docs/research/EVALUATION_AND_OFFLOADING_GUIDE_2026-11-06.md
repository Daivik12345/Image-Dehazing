# BTP evaluation and offloading optimization guide

**Research date:** November 6, 2026. **Target:** the current SD2.1 / ControlLDM pipeline on Kaggle Tesla T4, PyTorch 2.6.0+cu118, Python 3.10 inference environment.

**Immediate goal:** compare restoration quality, latency and memory reproducibly, then reduce offloading overhead without changing guidance. This document is a researched implementation protocol. The proposed batch evaluator, phase timers and offload policies below have **not been implemented or benchmarked**. Commands explicitly marked “available now” use the existing runner.

## 1. Start from what we have proved

Notebook version 7 completed HazeGen, spaced DiffDehaze, AccSamp and the generated-haze roundtrip on one Tesla T4. Version 8 adds safe rerun names. T4 x2 is the accelerator configuration; the current code uses GPU 0.

| Verified case | Size before padding | Calls | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---|---|
| HazeGen | 767 x 512 | 50 | 7.597 | 7.764 |
| Spaced DiffDehaze | 573 x 512 | 50 | 7.205 | 7.342 |
| AccSamp | 573 x 512 | 40 | 8.887 | 10.936 |
| Roundtrip, AccSamp stage | 767 x 512 | 40 | 11.769 | 14.049 |

Evidence: `artifacts/kaggle-v7-low-memory/verified-summary.json`, original per-case manifests/status files, images and logs. These files are local artifacts, not a published benchmark dataset. Each stage runs in its own process; the roundtrip row is not a simultaneous two-stage memory measurement.

The current low-memory mode offloads CLIP after conditioning, freezes parameter gradients, and temporarily offloads both denoisers during guidance. It retains float32, original resize/padding and the 50-step schedule. The full parameter count is unchanged. We have neither a reliable runtime baseline nor paired benchmark quality scores yet.

## 2. Separate three evaluation questions

| Track | Question | Inputs and evidence | Allowed conclusion |
|---|---|---|---|
| Systems equivalence | Does changing placement preserve the computation and reduce cost? | Same input, weights, geometry, seeds and outputs; repeated timings | Quality-preserving memory/runtime tradeoff on measured hardware |
| Paired restoration | How close is the restored image to a known clean target? | Real hazy/clean pairs; PSNR, SSIM, LPIPS and crops | Additional BTP restoration benchmark |
| Paper comparison | Do we match the original evaluation protocol? | Original datasets, metrics, settings and aggregation | Reproduction claim only after matching them |

The [baseline paper](https://arxiv.org/html/2503.19262v1) evaluates on RTTS (4,322 unpaired images), with Fattal images for additional visual inspection. Its reported no-reference metrics include FADE, Q-Align, LIQE, CLIPIQA, ManIQA, MUSIQ and BRISQUE. Therefore paired PSNR/SSIM experiments supplement its main protocol; they cannot reproduce its RTTS table.

Treat no-reference quality as evidence about appearance, not proof of recovered scene content. Do not use the hazy input as a clean target. The HazeGen roundtrip is a useful functional test, but its synthetic distribution is not a substitute for real-world restoration evaluation.

## 3. Choose datasets and freeze scene lists

Use official releases and retain their citations, archive hashes, version information and original IDs.

| Dataset | Primary source | Role |
|---|---|---|
| O-HAZE | [Official dataset: 45 outdoor scenes](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/) | Paired real-haze restoration |
| NH-HAZE | [Official dataset: 55 outdoor scenes](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) | Non-homogeneous haze |
| Dense-Haze | [Official dataset: 33 outdoor pairs](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/) | Dense homogeneous haze stress |
| RTTS | [Original paper’s evaluation description](https://arxiv.org/html/2503.19262v1) | Later unpaired paper-comparison track |

These counts describe complete datasets, **not test-split sizes**. Inspect the release's actual training/validation/test lists and ground-truth availability before selecting cases. Do not guess official scene-number ranges. If public test targets are unavailable, label an accessible held-out subset as an internal BTP evaluation, not the official challenge test.

Recommended rollout:

1. Two paired development scenes for a smoke run: one ordinary scene and one difficult scene. Include the largest planned geometry before approving a broader run.
2. A fixed development panel of up to five scenes from permitted development data. Choose before viewing model outputs and cover sky, vegetation, structures and dense/non-uniform haze where available.
3. A disjoint locked evaluation set using the official accessible test split. If using a custom split, publish the exact IDs and selection rule; do not describe it as the official split.
4. Expand to another dataset only after the runner, metric contract and resource budget pass validation.

Split by **source scene**, not generated image or crop. Keep crops, different haze levels and HazeGen variants from one scene in the same split. Check duplicate content across releases. Check for overlap with known training data and disclose any pretrained-data overlap that cannot be established. No new training is needed for this stage.

### Geometry needs its own protocol

Real benchmark images can be much larger than our successful examples. The upstream resize has a **minimum short edge of 512**, followed by zero padding to multiples of 64; it does not impose a maximum size. Do not assume native benchmark frames fit a T4.

Maintain separate tiers:

- **Controlled development tier:** fixed 512 x 512 paired center crops, selected with identical coordinates in hazy and clean images. Label scores as crop-based diagnostics. This controls initial memory cost; it is not a native full-image benchmark.
- **Native evaluation tier:** original image dimensions and upstream processing. Record each OOM. A native claim requires completing the designated native set or reporting its coverage and failures.
- **Optional resized tier:** if needed, define one fixed, paired resize procedure and name it as a separate derived dataset. Record both derived dimensions and actual dimensions after the runner's minimum-short-edge resize. Do not silently combine its results with native scores.

Choose the tier before comparing policies. Never resize only the candidate or only the ground truth to hide an OOM or shape mismatch. If the selected native tier does not fit, keep the failure visible and propose tiling/checkpointing as a separate experiment.

## 4. Write an explicit dataset manifest

The future harness should read a file with one record per scene, rather than pairing files by directory order. Proposed record fields:

```json
{
  "scene_id": "release-scene-id",
  "dataset": "NH-HAZE",
  "release": "official-release-identifier",
  "split": "development",
  "hazy": "hazy/scene.png",
  "clean": "clean/scene.png",
  "hazy_sha256": "actual-file-hash",
  "clean_sha256": "actual-file-hash",
  "geometry_tier": "paired-center-crop-512-v1",
  "crop_xywh": [0, 0, 512, 512]
}
```

This is a schema example; replace IDs, hashes and crop coordinates with real values. The harness must reject duplicate IDs, missing/unreadable files, mismatched pair dimensions, invalid crop bounds and ambiguous pairs. Validate the **complete manifest** before allocating a GPU. Preserve source-image hashes as well as hashes of derived crops.

## 5. Freeze the metric contract

For the first internal paired protocol, use decoded RGB images, full output extent, no evaluation border crop, and values in [0,1]. The final prediction must have the same shape as the clean reference. Keep the model's upstream final resize; do not add an evaluator resize.

| Metric | First internal contract | Direction |
|---|---|---|
| PSNR | MSE over all RGB channels/pixels, data range 1.0; average per-image dB values | Higher |
| SSIM | Per-channel RGB average, Gaussian weights, sigma 1.5, population covariance, data range 1.0 | Higher |
| LPIPS | AlexNet, calibrated metric version 0.1, RGB NCHW scaled to [-1,1] | Lower |
| Placement equivalence | Prediction-versus-baseline maximum/mean absolute error; output hashes | Zero drift preferred |

[scikit-image 0.24 metric documentation](https://scikit-image.org/docs/0.24.x/api/skimage.metrics.html) documents explicit ranges and SSIM settings. A reference metric call is:

```python
psnr = peak_signal_noise_ratio(clean, prediction, data_range=1.0)
ssim = structural_similarity(
    clean,
    prediction,
    data_range=1.0,
    channel_axis=-1,
    gaussian_weights=True,
    sigma=1.5,
    use_sample_covariance=False,
)
```

Import these from `skimage.metrics` in the separate evaluation environment. Ensure images are large enough for the configured SSIM window; report invalid cases instead of altering the window per image. Identical images have infinite PSNR: flag that case explicitly and retain a count rather than silently substituting a finite score.

[LPIPS authors’ implementation](https://github.com/richzhang/PerceptualSimilarity) specifies the tensor range and model variants. Initialize the pinned metric once and use evaluation/no-gradient mode **for scoring**, after inference finishes. Metric models must not consume memory during timed dehazing. Record the package version and metric-weight hash; automatic backbone downloads require a separate preparation step.

Metric protocols differ across publications. [BasicSR’s evaluation utility](https://github.com/XPixelGroup/BasicSR/blob/master/scripts/metrics/calculate_psnr_ssim.py) exposes border cropping and luminance-channel options. Our initial RGB/no-crop contract must not be described as equivalent to an unspecified Y-channel, MATLAB or cropped evaluation. Implement paper-compatible metrics in a separately named protocol if required.

Score saved PNG outputs for reproducibility. For strict placement-equivalence checks, also compare floating-point outputs before quantization when the future runner can save them: matching 8-bit PNGs can hide small floating-point changes. Ground truth is for the evaluator, never a conditioning or policy-selection input.

## 6. Define baselines before optimization

| ID | Definition | Status |
|---|---|---|
| R0 | Original residency, default `btp.reference`, float32 | Available; AccSamp OOM observed on a T4 example |
| M0 | Current `--low-memory`: frozen weights, CLIP CPU, both denoisers CPU during guidance | Available; version 7 example runs verified |
| M1 | M0 plus skip offloading when guidance will do no work | Proposed |
| M2-U | Freeze/CLIP policy retained; offload only UNet during active guidance | Proposed |
| M2-C | Freeze/CLIP policy retained; offload only ControlNet during active guidance | Proposed |
| M2-R | Freeze/CLIP policy retained; keep both denoisers resident | Proposed; separate from R0 |
| M3 | Winning fixed placement plus a tested allocator/cache policy | Proposed |
| M4 | Same winning placement with models reused across images | Proposed; warm-service track |

R0 is an implementation reference, not a mandatory T4 success. Report its failures. Use M0 as the practical T4 comparator. M2-R isolates the effect of freezing and CLIP offload from denoiser placement; do not call it the unmodified original.

For each candidate, keep checkpoints, prompts, input geometry, sampler schedule, τ, ω, guidance strength, seed and dtype unchanged. Do not fold mixed precision, fewer steps, an AlignOp repair, compilation or confidence-aware guidance into this experiment.

## 7. Measure runtime boundaries correctly

Define distinct timers rather than reporting one ambiguous “inference time.”

| Timer | Boundary | Why |
|---|---|---|
| Environment setup | Python/dependency preparation | One-time setup, separate from inference |
| Process end-to-end | Just before subprocess launch through exit and saved output | Available workflow's real per-image latency |
| Model load | Construction, checkpoint load, device placement | Exposes checkpoint/I/O overhead |
| Conditioning | Image VAE encode and prompt embeddings | Distinct from repeated denoising |
| Sampling | Complete sampler, including offload, guidance and transfers | Main systems comparison |
| Finalization | Final VAE decode, output resize and save | Captures actual delivered result |
| Warm resident request | Same loaded model processing a later request | Requires new persistent runner |

The current CLI starts a fresh process and reloads weights for every image. Repeating it does **not** measure a warm resident service. Record first-run and subsequent process launches separately because filesystem caches may warm even when the model does not remain resident.

Within a process, use wall-clock timing bracketed by device synchronization so asynchronous CUDA work finishes within the measured boundary. Use CUDA events for GPU intervals, not as a replacement for end-to-end wall time. See [PyTorch 2.6 CUDA semantics](https://docs.pytorch.org/docs/2.6/notes/cuda.html).

Illustrative sampling instrumentation, **to add to the runner**:

```python
# Illustrative helper to implement; sampling_call is a no-argument callback.
import time
import torch


def measure_sampling(sampling_call, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = sampling_call()
    torch.cuda.synchronize(device)
    stats = {
        "sampling_wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    return result, stats
```

Do not insert synchronization between every operator in benchmark mode: that
would change the overlap being measured. Phase-by-phase synchronized diagnostics
are a separate profiling run.

For resident measurements, warm up the same native padded geometry once, reset seeds/RNG state, reset peak counters, then time repeated requests in the same loaded process. Rebuild each image's conditions and targets; handle CLIP prompt caching explicitly because the current low-memory path moves CLIP to CPU. Record warm-up memory separately; excluding warm-up from latency must not conceal a startup OOM.

## 8. Record memory, transfers and failures

Track process-lifetime GPU peaks and a separately reset sampling-region peak. The current manifest counters are lifetime peaks, including loading; they cannot be relabeled as warm sampling peaks. Record current allocation before sampling as well as the peak.

Also collect host process RSS/peak RSS, overall free host RAM, actual GPU total memory, driver/runtime versions and concurrent GPU processes. Sampling `nvidia-smi` supplies complementary device-level evidence but may miss brief peaks; it does not replace allocator counters. Pinned host buffers and CPU copies consume real RAM.

Every job must end with a structured record, including OOM, timeout, invalid output or metric failure. Keep partial logs and input IDs. Do not average only survivors without giving completion coverage. Compare paired quality/runtime on a declared common-success subset and additionally report failures over the full designated set. No hidden smaller-image retry.

Count denoiser/ControlNet calls, VAE decodes, VAE backwards, active guidance calls, host-to-device/device-to-host bytes and transfer events. Count actual computation: the 50-step schedule alone does not describe AccSamp's guidance cost or HazeGen's conditional/unconditional forwards.

## 9. Profile before choosing placement

Use a small diagnostic run with [PyTorch 2.6 Profiler](https://docs.pytorch.org/docs/2.6/profiler.html), CPU and CUDA activities, and named regions:

`checkpoint_load`, `conditioning`, `denoise`, `offload_to_cpu`, `vae_guidance_forward`, `guidance_backward`, `restore_to_gpu`, `final_decode`, `save_image`.

Export a trace and examine copy operations, synchronization stalls, allocation peaks and time spent in each region. Profile one or a few representative images, including a dense scene and the largest geometry. Profiler/stack/memory tracing adds overhead; do not use those runs as headline timings. Avoid adding overlapping profiler columns together as if they were a wall-clock decomposition.

[Understanding Diffusion Model Serving in Production (SoCC 2025)](https://yanyinglin.github.io/assets/pdf/SoCC25.pdf) motivates measuring model-loading and I/O costs across the pipeline. Its production-service measurements do not predict BTP latency.

## 10. Optimize offloading in this order

### Experiment A: skip transfers when guidance is inactive

Repository inspection shows the wrapper enters `offload_denoisers()` before upstream `apply_cond_fn()` checks whether guidance is active. Reproduce the upstream condition exactly: `t_stop < int(t[0].item()) + 1 < t_start`. Retain the original callback for the actual update.

First count active and inactive callbacks. With the current schedule and guidance window, all invoked callbacks may already be active; in that case this experiment has no useful transfer reduction. Zero repeats are another no-work case, but do not alter other guidance semantics. Check RNG use, call counts and output equality.

**Decision:** keep only if it avoids real transfers or simplifies the path without a measurable regression. This is an implementation optimization, not a new dehazing method.

### Experiment B: offload the smallest sufficient module set

Measure parameter/buffer bytes and guidance peaks. Test M2-U, M2-C and M2-R independently against M0. A smaller module is cheaper to transfer but may free too little VRAM. Choose placement using validation geometry and measured memory headroom, not clean targets or test scores.

If one policy fails a geometry, mark that job as failed. A future size-based dispatcher can select from prevalidated policies by padded shape; freeze its thresholds on development data and test the dispatcher as its own variant. Free memory alone is a noisy runtime policy signal.

### Experiment C: reduce unnecessary allocator flushing

The current wrapper calls `empty_cache()` during guidance. Test the winning placement with the existing call, then with flushing only at phase transitions or without per-step flushing. Keep an explicit experiment ID for each choice. The [PyTorch allocator implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/memory.py) explains that cache release does not free live tensors. Removing the call can alter allocation behavior or cause OOM, so correctness and completion checks still apply.

Use allocator configuration changes only in a separate, logged comparison. Never present cache flushing as a way to make live tensors smaller.

### Experiment D: reuse models and fixed prompts across images

Only after a good fixed placement is selected, design a resident runner. Reuse model weights and fixed prompt embeddings; reset RNG per job and refresh conditioning/guidance targets. Release per-image tensors and check for rising memory over repeated requests. Keep separate cold-process and warm-resident results.

For the complete pipeline, consider sharing the SD backbone with controlled stage-specific ControlNet swaps, but first verify that state loading fully restores each stage. Do not claim that this reuse already exists in the reference runner.

### Experiment E: pinned buffers, asynchronous transfers or grouped placement

Attempt this only if the trace shows meaningful transfer stalls and a safe opportunity for overlap. [PyTorch's pinned-memory tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) discusses why `non_blocking=True` alone is insufficient. Account for pinned RAM, event dependencies, buffer lifetimes, stream synchronization and delayed CPU reads. Test on the pinned Torch 2.6 environment rather than copying newer APIs blindly.

The [Diffusers memory guide](https://huggingface.co/docs/diffusers/en/optimization/memory) describes model/group/leaf offloading tradeoffs; our ControlLDM lacks those pipeline helpers. [ChunkFlow (May 2026 preprint)](https://arxiv.org/abs/2605.11335) studies communication-aware prefetching on distributed DiTs and H100s. It supports investigating overlap, but its architecture and hardware differ from our UNet/T4 setup. None of its speedup figures are BTP expectations.

## 11. Preserve gradients and numerical behavior

Keep AccSamp's VAE-to-latent gradient path. Freezing parameters does not imply disabling autograd on the latent. Do not wrap the full guided sampler in `inference_mode`, detach the VAE result, put guidance under `no_grad`, or reuse a stale guidance target.

For each placement experiment compare:

1. Module placement/dtype and restoration after success and injected failure.
2. Tiny CPU forward/latent-gradient equality, followed by a real CUDA guidance-step test.
3. Same-seed full output against repeated M0 runs, including border and sky crops.
4. Completion, output dimensions and finite pixels on the fixed development panel.

[PyTorch 2.6 reproducibility guidance](https://docs.pytorch.org/docs/2.6/notes/randomness.html) warns that seeds do not guarantee equality across releases/platforms. Keep hardware/software and deterministic settings fixed. Estimate M0-versus-M0 variability before interpreting tiny candidate differences. Do not silently enable slower deterministic algorithms only for one policy; log unsupported operations if strict mode fails.

## 12. Repeats, aggregation and acceptance gates

Start development with seed 42. For final candidate confirmation, use a predetermined small seed set such as 42, 43 and 44 if the budget permits. For timing, repeat each selected image/seed/policy combination three times, retaining the seed across timing repeats. Randomize or alternate policy order in the same session, and record device load. Repeated executions of one scene are not independent scenes.

Report per-dataset mean PSNR/SSIM/LPIPS, per-image values, completion rate, and median latency. Keep per-image latency medians before aggregating across scenes; stratify by padded size. Report p95 only with the number of observations and a warning when the sample is too small to support a stable tail estimate. For uncertainty in candidate-minus-baseline quality, bootstrap by source scene, keeping its seeds/repeats together. Do not pool datasets into one unlabeled score.

**Suggested internal gates, to freeze before candidate evaluation:**

- No new OOMs, timeouts, corrupt outputs or shape errors on the locked set.
- For a placement-only change, aim for identical outputs; investigate any drift. Define any numerical tolerance from repeatability/float precision before looking at candidate scores. Do not substitute a permissive PSNR-drop threshold for investigating a changed computation.
- A useful initial speed target is at least a 10% reduction in paired median sampling wall time, confirmed across repeats. This is a project decision threshold, not a promised gain or a published standard.
- Do not accept a speed win obtained by reducing completion coverage. Keep per-image latency regressions and host-RAM growth visible.
- Use a development memory-headroom target, for example 1 GiB below measured device capacity for total observed use; report how it was measured. Allocator peaks alone do not prove that headroom exists. Native larger images still require their own stress test.

If two candidates trade time against memory, publish the Pareto comparison rather than hiding it in one composite score. If no candidate wins, retain M0 and record the negative finding.

## 13. Proposed implementation structure and artifacts

Keep reusable work in `src/btp/`; notebook cells should orchestrate it.

| Proposed addition/change | Responsibility |
|---|---|
| `src/btp/evaluation.py` | Manifest validation, pairing, metric contracts and per-image scores |
| `src/btp/benchmark.py` | Job execution, repeat plan, timings, memory and structured failures |
| `src/btp/memory.py` | Explicit fixed placement policies and exception-safe restoration |
| `src/btp/reference.py` | Timed phases, actual call counters and policy provenance |
| `configs/evaluation/` | Frozen splits/configuration and protocol version |
| `notebooks/btp-evaluation.ipynb` | Kaggle setup, bounded job submission and result inspection |
| `tests/` | Pairing, invalid data, numerical fixtures, failure retention and aggregation tests |

These are planned interfaces; no `btp.evaluate` or benchmark command exists yet. Avoid documenting a fictional runnable CLI.

Keep the pinned inference environment intact. A separate evaluator can use scikit-image 0.24.0 and LPIPS 0.1.4 as candidate pins, but installation/import and golden-fixture tests must validate them before the harness lockfile is accepted. Metric model weights need hashes. Add only required dependencies after checking compatibility; this guide installs none.

Each completed job should retain: experiment ID; scene/split/tier; seed/repeat; input and target hashes; all checkpoint hashes; BTP commit plus dirty-state/diff hash; upstream commit; source-bundle hash; notebook version; pinned package list; device/driver; prompts and guidance settings; dtype; original/inference/padded/output dimensions; policy; call counts; status/error; timer boundaries; GPU/host peaks; prediction hash and metric protocol ID.

Store `jobs.jsonl`, `per_image.csv`, `summary.json`, comparison plots, representative crops and profiler traces in a versioned experiment directory. Keep metric failures and inference failures distinguishable. Write records atomically; use unique IDs and preserve earlier runs. Model weights and datasets remain outside Git.

## 14. Kaggle execution checklist

1. Use `btp-baseline+`, GPU T4 x2 and Internet enabled. Use files persistence, but rerun runtime setup after a restart because `/tmp` is not persistent.
2. Attach the existing private source/weight datasets. For evaluation, add a separate approved, versioned dataset containing the permitted benchmark pairs and frozen manifest.
3. Validate the source bundle and checkpoint hashes. The `.bundle` upload prevents nested tokenizer gzip expansion. Check that the notebook contains the intended new harness before running it.
4. Verify one T4 and available host RAM; stop competing kernels on the same device. Keep batch size 1, previews disabled and logging cadence fixed across comparisons.
5. Run the tiny CPU smoke check, metric fixtures and a two-scene GPU pilot. Inspect outputs and failure records before scheduling more work.
6. Collect M0 correctness outputs, unprofiled timing repeats and separate diagnostic traces.
7. Test M1, then the smallest sensible selective-offload candidates. Use development data only to choose a policy.
8. Freeze policy and protocol, evaluate the held-out set, score saved outputs afterward, and export records before ending the session.

### Available now: one-image M0 smoke run

After running the existing notebook setup (which defines `RUN`, `ENVIRONMENT`, `ROOT`, `WEIGHTS` and `OUTPUT`), this uses the real current CLI:

```python
case = OUTPUT / ("eval-pilot-" + str(time.time_ns()))
subprocess.run(
    RUN
    + [
        "-u",
        "-m",
        "btp.reference",
        "--mode",
        "accsamp",
        "--input",
        str(ROOT / "upstream/Learning-Hazing-to-Dehazing/inputs/1.png"),
        "--weights",
        str(WEIGHTS),
        "--output",
        str(case),
        "--seed",
        "42",
        "--dataset-version",
        WEIGHTS_DATASET_VERSION,
        "--preview-every",
        "0",
        "--low-memory",
    ],
    cwd=ROOT,
    env=ENVIRONMENT,
    check=True,
)
```

This example image is **unpaired** and the current dataset-version argument records weight-dataset provenance. The future evaluator must record benchmark-dataset provenance separately. This smoke command does not produce PSNR/SSIM, reliable phase timings or a resident-model benchmark.

### Budget before launching a matrix

Estimate from the pilot, not an assumed runtime:

`jobs = scenes x policies x seeds x timing_repeats`

`budget = sum(pilot-based runtime estimates by geometry) + setup + profiling + metrics`

Two scenes, M0 plus one candidate, one seed and three timing repeats already require 12 inference jobs, before profiler runs. Expand only after the pilot estimate and session quota are acceptable. Checkpoint hashing/loading can dominate short runs; preserve a cold-process table even when a resident mode is added.

Do not assume two concurrent workers halve time. Upstream scripts set `CUDA_VISIBLE_DEVICES=0`; explicit device handling must be fixed and tested before two-GPU orchestration. A two-worker throughput experiment also changes host RAM and I/O contention, so it is a separate benchmark from single-image latency.

## 15. Completion criteria and the first deliverable

The next milestone is complete when we have:

- An immutable paired-data manifest and a written geometry/metric contract.
- A tested CPU evaluator with exact-match, known-error, missing-file, shape, color/range and non-finite input fixtures.
- A job runner that retains failures, provenance, phase timing and memory records.
- M0 results on the designated panel and repeatability evidence.
- One controlled offloading comparison with quality-equivalence checks and measured runtime/memory.
- A report stating the winner or negative result, its coverage and its limitations.

**Recommended first implementation:** manifest validation, PSNR/SSIM scoring, M0 phase instrumentation and one diagnostic profiler trace. Then decide whether M1 or selective placement is worth implementing. Keep LPIPS and RTTS metrics as separately prepared scoring stages so their dependencies and GPU memory do not contaminate inference measurements.

After offloading is measured and stable, return to confidence-aware guidance or high-resolution fusion from the [earlier research review](BASELINE_IMPROVEMENTS_2026-11-04.md). A compact student remains the route to reducing parameter count; this milestone focuses on trustworthy comparison and execution efficiency.

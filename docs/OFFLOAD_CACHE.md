# Selective offloading and cache control

The separate [btp-offload-cache notebook](https://www.kaggle.com/code/cubingultimate/btp-offload-cache) implements the next two systems experiments. Its [repository copy](../notebooks/btp-offload-cache.ipynb) is derived from the evaluation notebook. All three earlier notebook files and their Kaggle source datasets are preserved.

The previous pilot found no useful M1 transfer reduction: every invoked guidance callback was active. This workflow instead changes **which modules move** and **when allocator cache is released**. These are implemented experiments, not demonstrated speedups or model compression. No new weights, training, precision changes, sampler changes or guidance changes are involved.

## Kaggle resources

Use **GPU T4 x2**, Internet on, and save a version with outputs. Jobs run sequentially on GPU 0; memory is not pooled across cards.

- [New notebook](https://www.kaggle.com/code/cubingultimate/btp-offload-cache)
- [New private source dataset](https://www.kaggle.com/datasets/cubingultimate/btp-offload-cache-source)
- [Existing pretrained weights](https://www.kaggle.com/datasets/cubingultimate/btp-paper-weights-2)
- [Existing frozen NH-HAZE crops](https://www.kaggle.com/datasets/cubingultimate/btp-eval-nh-haze)

The new source archive is `btp-offload-cache-source.bundle`. Do not replace `btp-source` or `btp-eval-source`. Reusing the exact weights and evaluation dataset avoids changing the scientific comparison. Collection additions remain manual.

## Run in stages

Run setup cells in order: verify the source snapshot, create the existing Python 3.10/PyTorch 2.6 runtime, verify checkpoint hashes, run CPU model/gradient smoke checks, then restore the frozen paired dataset.

| Stage | Configuration | Jobs | Purpose |
|---|---|---:|---|
| M0 repeatability | `repeatability.json` | 6 | Two development scenes, three identical-seed repeats each |
| Placement screen | `placement.json` | 8 | M0, M2-U, M2-C, M2-R on both scenes, one repeat |
| Cache screen | `cache-template.json`, filled by notebook | 6 | Same selected placement with guidance/phase/never flushing |
| Confirmation | `confirmation-template.json`, filled by notebook | 12 | Selected combination versus M0, three repeats per scene |

Default execution runs repeatability and placement (14 jobs). Based on the prior M0 cold-process median of about 191 seconds, **14 jobs at that cost would take about 45 minutes, plus setup**. Candidate runtimes and failures may differ. Each worker has an eight-minute timeout; worst-case duration is much longer. Set `RUN_PLACEMENT=False` for a six-job baseline-only session. No GPU session is launched by publishing the notebook.

`RUN_CACHE=False` and `RUN_CONFIRMATION=False` prevent expensive follow-up work before inspection. After baseline and placement finish, inspect `eligibility.json` and tables, choose an eligible `CACHE_PLACEMENT` (M0 is a fallback), set `RUN_CACHE=True` in the live session and run the cache-stage cell. Do not rerun the setup cell just to change flags: it creates a new session directory. Set the variables in a small new cell or edit the stage cell instead. After reviewing cache results, set `CONFIRM_CACHE` and `RUN_CONFIRMATION=True` and run the confirmation cell. No winning placement is selected automatically.

Every stage creates a fresh directory and archives evidence in a `finally` block. Download the session ZIP from the final cell or Kaggle Output panel. Rerunning a stage preserves previous results. Keep the source/dataset versions and notebook version with exported results.

## Independent controls

Placement remains selectable through `--memory-policy`: M0 moves UNet and ControlNet during guidance; M2-U moves UNet only; M2-C moves ControlNet only; M2-R keeps both resident. All retain weight freezing and CLIP offload after conditioning. R0 remains the original residency reference.

New `--cache-policy` options:

| Value | Adapter-added `empty_cache()` calls |
|---|---|
| `guidance` | After conditioning/CLIP offload, and after actual denoiser offload before each guidance update |
| `phase` | Once after conditioning/CLIP offload, before the first denoiser call |
| `never` | No adapter-added cache flushing |

The cache policy changes allocator behavior, not model tensors or live-tensor memory requirements. For M2-R, guidance and phase have the same flush locations because no denoiser is offloaded; telemetry makes that visible. R0 only accepts `never`. With no new option, historical behavior is preserved: M0/M1/M2 use guidance flushing; M3 uses phase flushing; R0 uses never.

The CLI requires explicit `--memory-policy` when specifying `--cache-policy`. Existing notebooks retain their original source datasets and arguments.

## Variant-aware measurements

New matrix configs use `variants` instead of `policies`, with an explicit `reference_variant`:

```json
{
  "variants": [
    {"id": "unet-guidance", "policy": "M2-U", "cache_policy": "guidance"},
    {"id": "unet-never", "policy": "M2-U", "cache_policy": "never"}
  ],
  "reference_variant": "unet-guidance"
}
```

This is a fragment; use the complete supplied configuration templates for runnable jobs. Every variant has a unique ID and placement/cache pair. Job records, CSVs, summaries and matched comparisons retain both controls. The cache screen compares against the **same placement with guidance flushing**, isolating the cache change. Final confirmation compares the chosen combination against M0. Legacy configs with `policies` still work and retain their summary keys.

All inference retains float32, seed 42, the original schedule/guidance and two 512 x 512 development crops. Diagnostic previews are off. Existing metrics, synchronized sampling timing, lifetime/sampling memory counters and transfer payload estimates remain available. Failed/OOM/timeout jobs stay in results. No automatic fallback shrinks an image or reduces the schedule.

## Evidence checks and interpretation

`btp.optimization` analyzes saved predictions after timed inference. Repeatability requires complete unprofiled M0/guidance records with at least three repeats, matching provenance (including TF32, cuDNN and determinism flags) and positive finite sampling times. It computes all same-scene/seed repeat-pair saved-PNG MAE/max differences and timing ranges.

Placement screening requires the same source snapshot, dataset manifest, panel and runtime/checkpoints as repeatability, plus a complete M0 screening control. A candidate needs complete coverage and output differences no greater than observed M0 repeat variation (apart from numerical comparison epsilon) to be eligible for further cache testing. A single failed or drifting case makes it ineligible. The threshold is derived before candidate evaluation and is not loosened to pass a candidate.

This conservative check is **not** proof of latent-gradient equivalence, total-device memory headroom or final optimization acceptance. Saved PNG quantization hides some floating-point differences. Inspect outputs and full metrics; investigate drift. Use one crop per source scene, as in this fixed panel. Do not tune on held-out scenes.

The next milestone remains a repeat-confirmed sampling-time reduction of at least 10%, without new failures or unexplained drift. This is an internal target, not an expected gain. Confirm per-image regressions, host RAM and memory headroom before adopting a default. A two-image development screen does not establish robustness on larger/native images or paper-level reproduction.

The [research plan](research/EVALUATION_AND_OFFLOADING_GUIDE_2026-11-06.md) retains resident-model reuse and asynchronous transfers as later stages. Neither is implemented by this notebook.

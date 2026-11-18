# Three-track BTP research plan

Date: 2026-11-16. Status: proposed experiments, not implemented improvements. Existing notebooks and their evaluation criteria remain unchanged. This plan supplements the validation workflow; it does not claim its GPU run has completed.

## Objective and scope

Investigate three independent axes: deployment model size, restoration quality, and high-resolution capability. A successful BTP should establish at least one repeatable gain with transparent trade-offs; running three experiments does not guarantee three improvements or novelty.

| Track | Research question | Primary outcome | Training |
|---|---|---|---|
| A. Compact student | Can a small feed-forward network retain useful diffusion-teacher quality? | Parameter/checkpoint reduction with a measured quality-latency trade-off | Yes |
| B. Confidence-aware guidance | Can bounded, reliability-aware guidance reduce artifacts in uncertain regions? | Better restoration and fewer severe regional regressions | No initially |
| C. Global-guided tiled inference | Can a global reference reduce seams and color drift in bounded-memory high-resolution inference? | Native-resolution quality and feasibility under a T4 budget | No initially |

The compact student replaces deployment-time DiffDehaze only. HazeGen and the teacher remain offline tools. This does not compress the full two-stage generation/restoration system; a separate generator-compression study would be required for that claim.

## Shared foundation

1. Export the btp-validation results, verify completion and resolve eligibility failures. Freeze a working teacher configuration. Use M0 if a faster alternative remains unconfirmed. Record model/checkpoint hashes, source commit, runtime, geometry and guidance settings.
2. Keep the existing 10-development/45-held-out NH-HAZE split unchanged. The initial two-scene panel is a smoke/screening panel, not sufficient evidence for final claims. Use all development scenes for validation where feasible; evaluate the held-out set only after decisions are frozen.
3. Obtain a separate, licensed training corpus for the student. Split by original scene before crops, synthetic haze generation or augmentation. Audit duplicates against evaluation scenes. Do not train on the 45 held-out scenes or treat 55 NH-HAZE examples as a sufficient general-purpose training corpus. External pretrained models' possible exposure to evaluation data must be disclosed.
4. Report PSNR, SSIM and LPIPS against clean targets; synchronized cold/warm latency; live/reserved GPU memory; host RAM; failure coverage; parameter count and deployable checkpoint bytes. Preserve the current float32 reference and report precision changes separately. LPIPS requires its own pinned dependency/weights and protocol.
5. Systems-only changes retain the strict output-consistency gate. Student, guidance and fusion changes intentionally modify outputs: register separate quality-based research criteria before running them, without loosening or replacing the systems gate.
6. Use batch size one on the same T4 and identical geometry for timing comparisons. Use five timed repeats after a separately reported warmup for finalists and scene-level paired summaries. Repeats of one image are not independent scenes. Evaluate training finalists with three seeds if the approved budget permits; otherwise disclose single-seed uncertainty.

## Track A: compact feed-forward student

### Design

Use a NAFNet-style restoration network as a tractable baseline, initially width 16, then width 32 only if needed. Measure actual parameters before choosing the configuration; aim for a 1-5 million parameter deployment model rather than assume a width guarantees that count. Input is an RGB hazy image and output is RGB restored image with a residual prediction. No CLIP, diffusion UNet, ControlNet or VAE is needed at student inference.

A NAFNet checkpoint trained for another restoration task is not automatically a dehazing model. Choose either reproducible random initialization or a documented compatible initialization, and hold it constant across ablations. Existing diffusion checkpoints cannot directly populate an unrelated small backbone.

### Data and supervision

Generate teacher predictions offline, then release the teacher before training. Use available clean ground truth as primary supervision. HazeGen may supply additional haze on training-only clean scenes; generated pairs require registration/content checks and provenance. Reject obvious corruption without selecting only easy low-haze examples. Keep rejected counts and haze-diversity summaries.

Start with reconstruction plus structural loss: Charbonnier(student, clean) + 0.1 * (1 - SSIM(student, clean)). Test teacher distillation as a separate addition, starting at weight 0.1 for output-space L1(student, teacher). These are candidate settings to freeze after a small development search, not proven best weights. Avoid importing the teacher's mistakes blindly. Perceptual training loss is a later single-factor ablation if needed; LPIPS evaluation remains independent.

### Ablations

- A0: compact student, fixed training set, clean-target supervision only.
- A1: identical student/data/training updates plus teacher supervision.
- A2: A0 plus HazeGen augmentation, fixed total updates and controlled sample count/mixing.
- A3: combine A1 and A2 only after their separate effects are measured.

The paper already uses generated data for downstream dehazing. Teacher/student training and synthetic augmentation alone are not novelty claims. A contribution must be a demonstrated size-quality trade-off or a specific effective data/distillation method.

### Targets and budget gates

Proposed deployment targets: at least 10x fewer parameters than the measured complete deployed DiffDehaze reference, at least 5x lower warm latency at matched geometry, and no more than 0.5 dB PSNR / 0.01 SSIM degradation versus the teacher on the fixed development panel. Report LPIPS and severe scene regressions even if those gates pass. These are adjustable design targets to register before the experiment, not achieved numbers.

CPU smoke: 2 images, 2 optimizer steps, finite loss/gradients, checkpoint save/reload. GPU pilot: 256-pixel training crops, batch 1-4 as measured, at most 2,000 updates or 2 T4-hours, whichever comes first. First benchmark 100 updates to estimate time and memory. Obtain approval before that long training run and before larger downloads. Stop on non-finite loss, leakage or persistent OOM; do not silently shrink final evaluation geometry. Full training needs a separate budget decision based on pilot learning curves.

Suggested notebook: btp-student-distill. Modules: student.py, training_data.py, distillation.py, train_student.py, evaluate_student.py. Deliverables: student weights, parameter/checkpoint audit, learning curves, paired quality/runtime tables and failure examples.

## Track B: confidence-aware fidelity guidance

### Hypothesis and implementation

AccSamp already includes transmission/physics-related fidelity guidance. The proposal is to make an existing correction more reliable, not to introduce physics for the first time. Inspect the actual alignment and guidance code before choosing the insertion point.

Estimate reliability using information available at inference: local contrast, saturation/bright-region ambiguity and instability of alignment statistics. Start with one interpretable reliability proxy and fixed thresholds. Apply bounded weighting to the guidance correction while retaining a nonzero floor, so uncertain regions do not simply lose all fidelity constraints. Test the confidence map as a detached control initially; retain required gradients through the latent-image path. Never use clean targets to compute inference confidence.

### Ablations and evaluation

- B0: unchanged AccSamp.
- B1: robust alignment/statistic stabilization only.
- B2: B1 plus bounded confidence weighting.
- B3: B2 plus a correction-magnitude cap.

Keep sampling schedule, seeds, precision and residency fixed. Test brightness/sky, haze boundaries, dense haze and fine structure using masks/annotations defined independently of candidate outputs. Also compare with uniformly weakened guidance to show that any gain is not merely weaker guidance. Record VAE backward cost and inspect hallucination/oversmoothing failures.

Proposed success: repeatable development gain of at least 0.3 dB mean PSNR or 5% relative LPIPS improvement, with no material regression in the other metrics (pre-register a 0.2 dB PSNR / 0.005 SSIM regression bound), and no increase in severe visible artifacts. Do not call a two-image change significant. If uncertainty spans no gain, report inconclusive and do not promote it.

Suggested notebook: btp-guidance-quality. Deliverables: confidence maps, matched crops, ablation table, latency/backward-cost accounting. Training-free first implementation; do not introduce a learned confidence model until this simpler hypothesis is tested.

## Track C: global-guided high-resolution inference

### Hypothesis and implementation

Independent overlapping tiles can reduce working memory but introduce color inconsistency and seams. Combine a coarse full-frame prediction with local predictions to preserve global appearance and local details. This trades additional computation for high-resolution feasibility; speedup is not the primary claim.

Use a fixed global preview and 512-pixel tiles, initially with 128-pixel overlap. Align each tile's low-frequency appearance conservatively to the global reference and blend with positive smooth overlap weights. Ensure every native output pixel receives nonzero weight, including borders and images smaller than a tile. Avoid amplifying near-zero variance. Keep transformations and per-tile RNG seeds deterministic. Reuse loaded weights across tiles only after the persistent runner is validated; repeated cold loads would otherwise confound time.

### Ablations and evaluation

- C0: original full-image inference when it fits; retain OOM as a feasibility result.
- C1: overlapping tiles with normalized blending only.
- C2: C1 plus global appearance alignment.
- C3: reliability-weighted fusion only after C2 is useful.

Use original higher-resolution paired development images, not upsampled 512 crops. Preserve scene identities/splits and generate new hashes/crop metadata. Run fixed 512, 768 and 1024 center-crop panels where source dimensions permit, plus native resolution. Keep all candidates on identical geometry. Compare global metrics, boundary-band errors, seam discontinuity, thin structures, peak memory and end-to-end latency. Discontinuity alone is not a quality metric: blur could improve it artificially.

Proposed success: enable a higher-resolution case within T4 memory and/or reduce boundary-band error by at least 20% versus C1, without more than 0.2 dB global PSNR loss or obvious detail loss. Register the boundary-band width and metrics in advance. When full-image reference OOMs, compare quality against clean targets and feasible tiling baselines, not a resized reference presented as native output.

Suggested notebook: btp-global-tiles. Modules: tiling.py, fusion.py, native_manifest.py. CPU tests: complete coverage, identity reconstruction, odd sizes, small images and constant patches. GPU smoke: one image and a few tiles before the bounded panel.

## Execution order and stopping rules

1. Finish/export current validation; freeze teacher and evaluation protocol. This does not prevent CPU implementation of the new tracks.
2. Implement C tiling correctness and B guidance instrumentation while the compact student data audit proceeds.
3. Run A CPU/short profiling smoke; request the explicit pilot-training budget. Run one bounded B/C screen each rather than a large combinatorial sweep.
4. Retain only independently promising variants. If A loses too much quality, first inspect data/supervision and learning curves before increasing architecture size. If B hallucinates or C blurs, reject or revise using development data only.
5. Confirm each finalist independently. Combine changes only afterward with a separate combined-versus-individual ablation.
6. Freeze all settings, then run held-out evaluation once for reporting. Report negative/inconclusive findings and resource cost.

No guaranteed calendar or GPU-hour estimate is possible before profiling. A student smoke is not a trained model, a one-step model is not necessarily a small model, and quantization/offloading do not reduce parameter count. Quantization is an optional later deployment experiment, not a fourth independent scientific contribution.

## Primary sources and transfer limits

- Original method: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Learning_Hazing_to_Dehazing_Towards_Realistic_Haze_Generation_for_Real-World_CVPR_2025_paper.html
- NAFNet / Simple Baselines for Image Restoration: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf ; author implementation: https://github.com/megvii-research/NAFNet . Restoration backbone precedent, not evidence of our dehazing performance.
- SinSR: https://arxiv.org/abs/2311.14760 . Single-step restoration distillation motivation; super-resolution results do not establish dehazing transfer or smaller parameter count.
- RPD-Diff: https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_83/paper.pdf . Related region-adaptive physics-guided diffusion; our confidence weighting remains a hypothesis, not its reproduction.
- Existing BTP research roadmap: docs/research/BASELINE_IMPROVEMENTS_2026-11-04.md . Global/local fusion and compact-student directions were already proposed; this document turns them into controlled tracks without claiming novelty.
- PyTorch torchao overview: https://pytorch.org/blog/pytorch-native-architecture-optimization/ . Low-bit optimization exists, but compatibility/performance must be established for this convolutional ControlLDM and T4 runtime.

Implementation notes: retain original source attribution and licenses; pin any new dependency and checkpoint; publish new notebooks/source bundles rather than replacing existing baselines. This planning task launches no training or GPU evaluation and adds no model dependencies.

# Improving the HazeGen / DiffDehaze baseline

Targeted research review, November 4, 2026. Recommendations below are proposals, not measured BTP results. This is not a comprehensive systematic review.

## Recommendation

First establish the faithful Kaggle reference result. Then test training-free, global-guided patch fusion for high-resolution inputs and confidence-aware fidelity guidance for difficult scenes. If the goal is a model that genuinely uses less memory and runs faster, train a compact student using carefully filtered HazeGen data. Keep these as separate experiments so their effects remain interpretable.

The baseline is [Learning Hazing to Dehazing (CVPR 2025)](https://arxiv.org/abs/2503.19262), using the [authors’ implementation](https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing). AccSamp already includes dark-channel-prior transmission weighting and fidelity guidance. Adding “physics guidance” alone is therefore neither a new feature nor a novel research contribution.

## What our current baseline establishes

- Both supplied Baidu stage checkpoints match the previously acquired files byte for byte; checkpoint identity is recorded locally.
- Reduced 128/256px local runs completed but showed artifacts. They do not establish paper-level reproduction. No successful Kaggle GPU reference benchmark has been recorded.
- The local experimental adapter and original scripts differ in preprocessing. The original enlarges a short edge below 512, pads with zeros to multiples of 64, and restores the original output size. A longest-side cap changes the problem and cannot serve as the faithful reference.
- `src/btp/reference.py` runs the original scripts; use that path for the reference. Keep adapter optimizations and the AlignOp border-coverage repair in separately named variants.
- Fewer sampling steps reduce computation, not parameter count. Lower image resolution also changes quality and geometry.

These are repository observations, rather than conclusions drawn from the papers below.

## Recent papers and applicability

| Work and date/status | Relevant contribution | Implication for BTP |
|---|---|---|
| [Hihaze](https://arxiv.org/abs/2610.25832), October 19, 2026 preprint | One-step dehazing; training-free global-guided patch fusion, alongside learned adaptation modules | Fusion is the most direct high-resolution experiment. One-step conversion requires training. The [repository](https://github.com/zhengchen1999/Hihaze) exposed README/figures when checked; runnable code and weights were not verified. |
| [HNDiff](https://arxiv.org/abs/2608.10995), August 11, 2026; listed as ECCV 2026 | Joint haze/noise modeling and learned latent clean priors | Interesting training direction, not a scheduler replacement for frozen SD2.1. Its [project page](https://jin-ting-he.github.io/HNDiff/) does not establish an available BTP-compatible checkpoint. |
| [BiLaLoRA](https://arxiv.org/abs/2603.10872), March 11, 2026; author repository identifies CVPR 2026 | Selective LoRA layer placement with CLIP-based unsupervised guidance | Parameter-efficient adaptation is worth studying. [Released training code](https://github.com/YanZhang-zy/BiLaLoRA) uses another dehazing backbone; direct ControlLDM compatibility is unproven. |
| [Progressive Diffusion for Single Image Dehazing](https://link.springer.com/article/10.1007/s11063-026-11886-7), October 9, 2026 publisher early accepted version | Progressive resolution and feature-statistics consistency | Suggests a color-consistency training objective. Requires architectural/training changes; merely resizing our inputs is not equivalent. |
| [GenHaze](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_GenHaze_Pioneering_Controllable_One-Step_Realistic_Haze_Generation_for_Real-World_Dehazing_ICCV_2025_paper.html), ICCV 2025 | Reference-controllable, one-step realistic haze generation | Strong direction for improving training-pair diversity. Needs a trained generator/conditioning modules; code and usable checkpoints were not verified in this review. |
| [HazeFlow](https://arxiv.org/abs/2509.18190), ICCV 2025 | Physical haze model as an ODE, rectified flow, spatially varying haze synthesis | Useful alternative architecture and data-generation comparator. [Implementation](https://github.com/cloor/HazeFlow) exists; it involves pretraining, reflow and distillation, rather than reusing our sampler unchanged. Ready pretrained downloads were not verified. |
| [RPD-Diff](https://arxiv.org/abs/2508.16956), August 23, 2025; [BMVC 2025 paper](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_83/paper.pdf) | Region-dependent denoising timesteps and physics-guided intermediate targets | Supports studying spatial compute allocation. Adaptive timesteps are already published, so this alone is not a novelty claim. |

Availability means what was observed, not a guarantee that an entire installation or checkpoint works. Publication dates, preprint dates and release dates are distinct. None of these papers establishes a numerical improvement for our pipeline on Kaggle.

## Prioritized experiments

### 0. Freeze a faithful baseline

Create B0 using original preprocessing, native reference geometry, float32 CUDA, original 50-step settings and fixed checkpoint hashes. Record standard sampling and AccSamp separately. For AccSamp, preserve the baseline τ=0.8, ω=0.6 and guidance scale setting before tuning anything.

Create B1 for the known AlignOp uncovered-border repair. Report it as a bug fix, separately from a proposed research method. Do not silently incorporate it into B0.

Use paired real benchmarks such as O-HAZE, NH-HAZE and Dense-Haze with their prescribed protocols. Record PSNR/SSIM and a perceptual metric such as LPIPS; preserve the paper’s metric definitions when comparing its table. On unpaired real images, use the original no-reference metrics and visual crops, without treating any single no-reference score as ground truth. Never compute restoration PSNR against the hazy input.

### 1. Global-guided patch fusion: first high-resolution quality experiment

[Hihaze’s fusion method](https://arxiv.org/html/2610.25832v1) combines a global prediction with local patches, aligning appearance and weighting local reliability. Its transfer experiments include other dehazers, but the similarly named Diff-Dehazer there must not be assumed to be our Wang et al. DiffDehaze.

**BTP hypothesis:** a global reference can reduce tile-to-tile color drift while local processing preserves details within a bounded working-memory budget.

Compare overlapping tiles alone, global appearance alignment, reliability-weighted fusion, and optional detail compensation. Keep image geometry, tile size, overlap and seeds fixed between variants. Evaluate seam regions, sky, dense haze and fine structures separately.

This reuses our pretrained networks and can be explored without training. It requires implementation and validation; it is not an available command in this repository. Keep the model resident across tiles. Global and local passes add work, so expect a memory/quality tradeoff to measure—not an automatic latency improvement. Small 512px images are unlikely to be its strongest use case.

### 2. Confidence-aware AccSamp: first guidance experiment

**BTP hypothesis:** unreliable transmission estimates and local alignment statistics contribute to failures in bright skies or heterogeneous haze. Estimate reliability, bound unstable corrections, and compare a conservative structure-preserving fallback against the unchanged guidance.

This extends existing guidance rather than introducing physics for the first time. [RPD-Diff](https://arxiv.org/abs/2508.16956) and [HNDiff](https://arxiv.org/abs/2608.10995) provide related motivation for region- and haze-dependent processing, but neither validates this proposed modification to our implementation.

Ablate alignment robustness, confidence estimation and correction limits individually. Simply weakening fidelity guidance could increase hallucination; confidence masking must be tested rather than presumed beneficial. Tune τ, ω and guidance strength only on validation scenes. Retain the 50-step schedule initially to avoid mixing guidance changes with schedule changes.

Record VAE backward cost as well as denoising calls: AccSamp’s guidance means fewer sampling calls do not translate directly into proportional runtime savings. A blanket `inference_mode` around this path would disable required gradients.

### 3. Compact student: best route to a genuinely smaller model

Use frozen HazeGen as a data generator and DiffDehaze as an optional teacher. Train a compact restoration backbone against the original clean images, with a small, fixed combination of reconstruction and structural/perceptual losses. Teacher predictions should not replace available clean targets blindly.

Compare a fixed real/synthetic training set against the same budget augmented with generated pairs. Split clean source scenes before generation, then hold out scenes and random seeds. Filter content drift and geometric corruption; avoid a filter that merely removes all difficult dense haze.

[GenHaze](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_GenHaze_Pioneering_Controllable_One-Step_Realistic_Haze_Generation_for_Real-World_Dehazing_ICCV_2025_paper.html) motivates controllable generation, while the original baseline already studies generated data for downstream restoration. A teacher/student pipeline alone is therefore not a novelty claim. The potential BTP contribution would need a specific, tested improvement in data quality or the quality/compute tradeoff.

This requires training and new student weights. Existing SD2.1/ControlNet tensors do not become a compact backbone through a loading flag. Start with a tiny CPU smoke run and approve the GPU budget before training.

### 4. Improve HazeGen diversity before adding a new generator

First test controlled variation of existing seeds and supported haze parameters with an equal number of generated pairs. Measure whether the resulting student benefits, rather than judging generated haze only by appearance.

Longer term, investigate real-reference conditioning from GenHaze or spatially varying scattering inspired by [HazeFlow](https://arxiv.org/abs/2509.18190). Both require more than an extra prompt: reference conditioning needs learned modules, and physical synthesis needs appropriate spatial parameters and validation. Keep generation quality, downstream restoration quality and training budget separate in the comparison.

## Engineering blogs: useful, with transfer limits

The official [PyTorch compilation guide, July 17, 2025](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) explains compilation scope and cold-start tradeoffs. Its FLUX/H100 examples are not expected gains for our custom SD2.1 UNet on a T4. Profile first, then compile suitable repeated network modules; avoid compiling the entire gradient-guided Python loop initially. Changing native padded shapes can trigger recompilation.

The [Hugging Face quantization guide, May 21, 2025](https://huggingface.co/blog/diffusers-quantization) compares techniques largely through newer model architectures. It does not establish that our convolutional ControlLDM can use the same APIs or achieve the same compression. Quantization comes after a measured mixed-precision baseline, not before correctness.

The rolling [Diffusers memory guide](https://huggingface.co/docs/diffusers/optimization/memory) distinguishes offloading, VAE tiling and slicing. These concepts are useful, but its pipeline helpers are not directly exposed by our upstream implementation. VAE tiling does not tile the denoising UNet, and offloading may worsen latency.

Recommended systems order:

1. Measure preview-decoding overhead; benchmark with previews disabled as well as at the chosen live-preview interval.
2. Cache fixed prompt embeddings and investigate sharing the resident SD backbone between stage-specific ControlNet loads. The local adapter already releases CLIP after embedding, so that is not a new optimization.
3. Test mixed precision while keeping numerically sensitive guidance/VAE operations in float32 where needed. Compare outputs and failure rates against B0.
4. Evaluate compilation after stable profiling; only then consider approximate caches or quantization.

LoRA reduces trainable parameters and optimizer state, not all activation memory. Multiple GPUs do not automatically pool their memory. Confirm the actual Kaggle GPU and memory before selecting a budget.

## Measurement and decision gates

For every run, save the source commit, checkpoint hashes, dataset version/split, input/output geometry, seed, configuration, device and software versions. Use a small fixed validation subset for initial screening, then the unchanged test set for final reporting. Include multiple final seeds where stochastic variation matters.

Record cold and warmed end-to-end CUDA-synchronized latency, peak allocated/reserved GPU memory, model load time, preview overhead and failure rate. Also record UNet/ControlNet forwards and VAE forwards/backwards. Use the same hardware, batch size and image geometry for comparisons.

| Variant | Changes | Training needed | Decision evidence |
|---|---|---|---|
| B0 | Faithful original reference | No | Valid images and reproducible benchmark |
| B1 | Border-coverage repair | No | Border correctness and unchanged covered regions |
| E1 | Global-guided tile fusion | No | High-resolution quality/seams versus added runtime |
| E2 | Confidence-aware guidance | No initially | Difficult-region gains without hallucination/regression |
| E3 | Resident models / mixed precision | No | Memory and latency gains with acceptable output fidelity |
| E4 | Compact student and filtered generated pairs | Yes | Quality, parameters, memory and latency at equal training budget |

Start with B0/B1, then choose E1 for high-resolution quality or E2 for difficult-scene guidance. Choose E4 when smaller deployment is the primary objective. New one-step models, flow matching and learned noise processes are longer-term alternatives, not drop-in upgrades.

No model changes, GPU experiments, weight downloads or measured improvements were performed as part of this research review.

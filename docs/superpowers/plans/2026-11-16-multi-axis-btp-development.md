# Multi-axis BTP Development Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development to implement the tasks with tests and review. This is a planning deliverable, not authorization for large downloads, paid compute or long training.

**Goal:** Deliver a reproducible compact dehazing product and demonstrate a substantial second benefit in restoration quality or native-resolution capability, supported by independent controlled experiments.

**Architecture:** Preserve the original reference runner and all historical notebooks. Build reusable data, student, guidance, tiling and reporting modules under src/btp, orchestrated by small versioned Kaggle notebooks. Separate exact systems-equivalence tests from intentional model/algorithm quality experiments, and publish immutable experiment records.

**Tech Stack:** Python 3.12/uv for development; pinned isolated Python 3.10/PyTorch CUDA runtime for compatibility until a migration is independently validated; existing NumPy/Pillow/SciPy metrics; optional training and LPIPS dependencies isolated from the base app.

**Spec:** docs/research/THREE_TRACK_RESEARCH_PLAN_2026-11-16.md and the user's request for detailed notebook/repository planning with significant development in more than one aspect.

## 0. How to interpret this plan

All new file names, dataset slugs, APIs, CLI examples, models, thresholds and budgets below are **proposed**, unless explicitly described as existing. Do not mistake a planned notebook URL for a published artifact. No new experiment results are claimed here. The previous validation notebook was launched, but its latest completion and eligibility must be inspected during execution.

The initial plan proposed at least one repeatable gain. This expanded plan deliberately raises the final research target to **at least two independent axes**. That is an objective, not a guarantee that research will succeed. Failed hypotheses are recorded honestly; implementation completion and demonstrated benefit are different statuses.

### 0.1 Intended end product

A versioned BTP release containing:

1. **CompactDehaze student:** small RGB-to-RGB inference network, trained checkpoint, documented data and losses, verified parameter/file-size reduction, and quality/latency evaluation.
2. **Native-resolution inference:** correct overlapping tiling, optional global alignment, full-size output, resource-aware operation, and native-image evaluation. This is the preferred second product capability even if guidance does not improve quality.
3. **Guidance-quality research branch:** bounded confidence-aware AccSamp with visual diagnostics and ablations; ship as an improved preset only if it passes the quality criteria.
4. **Reference mode:** preserved pretrained HazeGen/DiffDehaze workflow and teacher checkpoint provenance.
5. **Evidence package:** locked configurations, notebook versions, dataset manifests, per-scene results, failure records, model card, reproducible CLI, figures and professor-ready report.
6. **Usable interface:** local viewer offers reference/student/tiled-student modes with clearly labeled experimental options, model identity, output dimensions and measured run statistics. User-facing flows do not expose internal implementation jargon unnecessarily.

**Product boundary:** the compact student replaces inference-time DiffDehaze. HazeGen and DiffDehaze remain large offline data-generation/teacher tools. The entire two-stage pipeline is not compressed unless HazeGen is separately compressed later.

### 0.2 What counts as two substantial improvements

| Axis | Evidence required | Does not count |
|---|---|---|
| A: compression/deployment | Measured reduction in parameters/checkpoint bytes plus acceptable clean-target quality; matched latency/memory results | Offloading, removing files from a ZIP, fewer sampling steps alone |
| B: restoration quality | Pre-registered metric gain with adequate scene coverage, artifact review and uncertainty reporting | One attractive image, changed crop, tuning on test targets |
| C: high-resolution capability | Correct native-size outputs with bounded working memory and measured quality/seam behavior | Upsampling a small output or quietly resizing test inputs |
| Supporting systems work | Warm model reuse, robust recovery, better evidence and runtime measurements | Counting three cache/offload variants as three independent contributions |

Minimum research target: **A + C**, or **A + B** if high-resolution gains are not established. Stretch target: all three. If A fails, report a partial research outcome rather than relabeling infrastructure as model compression.

## 1. Global constraints and scientific rules

- Preserve kaggle_reference.ipynb, btp-baseline-plus.ipynb, btp-eval-optimised.ipynb, btp-offload-cache.ipynb and published versions of btp-validation. Add new notebooks or versions with explicit lineage; do not retroactively replace baselines.
- Retain the exact 10-development/45-held-out NH-HAZE source-scene assignments. Derive larger crops/native manifests from those assignments, not a new random split.
- Do not train on held-out NH-HAZE scenes; do not use their metrics to select hyperparameters. Existing pretrained teacher exposure to those benchmarks is unknown unless documented, so describe this as a project-held-out evaluation, not proof of globally unseen data.
- Fixed protocol records input decoding, RGB range, geometry, border policy, metric implementation/version, seeds, precision, checkpoint hashes, source commit and dataset versions.
- Systems changes must satisfy the existing output-consistency gate. Student/guidance/fusion deliberately change outputs and get a separate pre-registered quality protocol. Never loosen the systems gate to approve a candidate.
- Use batch size one and the same GPU/geometry for deployment comparisons. T4 x2 availability does not imply pooled GPU memory or automatic multi-GPU execution.
- Use uv run, type hints and docstrings for public APIs. Run Ruff lint/format and relevant pytest. CPU smoke precedes GPU jobs. Training on a tiny fixture is not research evidence.
- Keep raw weights, private images, credentials, environments and large output archives out of Git. Store licenses/attribution and checkpoint metadata.
- Ask before large downloads, paid compute or long training. Approve each concrete training budget, not an unlimited campaign.
- No default experiment silently changes precision, image size, batch size, number of diffusion calls or policy after an OOM.
- Never call a model novel merely because it combines existing techniques. Keep engineering contribution, empirical result and potential methodological novelty separate.

## 2. Current starting point and unresolved items

Existing modules: reference.py (single-image upstream execution), benchmark.py (cold-process matrices), optimization.py (repeatability/screening), workflow.py (explicit evidence selection), telemetry.py, datasets.py, evaluation.py, loading.py, viewer.py.

Known findings: full reference pipeline previously ran on one T4; M1 did not avoid any transfers in its pilot; one supplied placement table showed M2-R much faster on two crops. The later rejection lacks historical raw evidence. New validation is intended to resolve a fresh comparison, not explain a lost historical run.

Before research claims:

- Inspect the latest btp-validation status, logs, raw predictions, configs and source/runtime hashes.
- Archive every completed or failed stage. Distinguish failed execution from a completed but ineligible candidate.
- Review baseline numerical variation and per-scene differences; do not assume a fast candidate is valid.
- Freeze M0 as a conservative teacher execution policy if faster policies remain unconfirmed. Freezing this fallback is legitimate; it does not claim the faster policy passed.
- Measure the actual parameter count and bytes of required deployment modules. Do not use the combined 8.12 GB upload of both stage checkpoints as the dehazer's parameter baseline.

## 3. Work packages and dependency graph

```text
WP0 Evidence + protocol lock
  ├─ WP1 Data registry + separate training corpus
  │    ├─ WP3 Offline teacher targets ─ WP4 Student training
  │    └─ WP6 Native manifests ─ WP7 Tiled inference
  ├─ WP2 Persistent reference runner ─ WP3 / WP7
  └─ WP5 Guidance-quality experiment
WP4 + WP5 + WP7 ─ WP8 independent confirmation
WP8 successful finalists ─ WP9 combination + held-out lock
WP9 ─ WP10 release, UI, report and publication
```

CPU implementation of independent modules can proceed while validation runs. Training depends on dataset audit and teacher lock. Large tile runs depend on validated model reuse. Held-out execution depends on a frozen finalist declaration.

## 4. Repository architecture and interfaces

Keep core functionality importable. Do not duplicate substantive algorithms in notebooks. Prefer a small package per subsystem as it grows.

```text
src/btp/
  protocols.py                  versioned experiment/decision contracts
  evidence.py                   portable archive audit and stage registry
  resident.py                   persistent pretrained inference session
  training_data.py              scene-based training manifests and paired crops
  teacher_targets.py            offline teacher-target generation
  student/
    model.py                    compact RGB restoration architecture
    losses.py                   reconstruction, SSIM, teacher losses
    trainer.py                  bounded/resumable optimizer loop
    inference.py                student loading and image inference
  guidance.py                   opt-in reliability/correction adapters
  tiling.py                     coordinates, padding, weights, streaming accumulation
  fusion.py                     global low-frequency alignment
  geometry.py                   native/multi-size manifests, frozen split inheritance
  research_eval.py              cross-model metrics and structured failures
  confirmation.py               paired aggregates, intervals and decision reports
  release.py                    locked model/protocol artifact inventory
configs/research/
  protocol-v1.json
  teacher-lock.json
  student/{architecture,pilot,train-a0,train-a1,train-a2}.yaml
  guidance/{baseline,robust,confidence,uniform-control,cap}.yaml
  tiling/{plain,global-aligned}.yaml
  budgets.json
  finalists.json
configs/kaggle-<slug>/kernel-metadata.json
notebooks/btp-*.ipynb
scripts/build_research_notebooks.py
scripts/publish_research_bundle.py
tests/test_protocols.py, test_evidence.py, test_resident.py,
  test_training_data.py, test_student.py, test_student_training.py,
  test_guidance.py, test_tiling.py, test_fusion.py,
  test_geometry.py, test_research_eval.py, test_confirmation.py
docs/research/{EXPERIMENT_REGISTRY,DECISIONS,FAILURES}.md
docs/model_cards/compact-dehaze-v1.md
```

These names are not mandates to create empty files. Create each when its work package starts. Keep existing cold-run APIs backward compatible and isolate optional Torch imports so base CPU tests remain usable.

### Shared runtime contracts

- `ExperimentSpec`: experiment_id, parent_id, source_commit, source_bundle_sha256, data_manifest_sha256, protocol_id, split, scene_ids, seeds, geometry, model/checkpoint ID, policy, precision, budget and output directory.
- `RunRecord`: scene_id, seed, repeat, input/output hashes, status, error, load_seconds, preprocessing_seconds, conditioning_seconds, sampling_seconds (teacher only), model_seconds (student), postprocess_seconds, warm_end_to_end_seconds, allocated/reserved peaks and host RSS. Inapplicable times are null, not invented equivalents.
- `DecisionRecord`: evidence paths/hashes, rule version, selected candidate, rejected alternatives/reasons, data scope, author/reviewer and timestamp. No mutable "latest report" pointer used for scientific selection.
- `InferenceSession.predict(image, seed, output_dir) -> RunRecord`: exact input geometry contract, explicit ownership of model/device, resettable per-image state, no hidden fallback.
- `load_student(checkpoint, config, device) -> StudentPredictor`: strict architecture/hash checks, no arbitrary pickle execution, mismatch fails with an actionable message.
- `make_tiles(width, height, tile_size, overlap) -> list[Tile]`: deterministic complete coverage.
- `fuse_tiles(tiles, predictions, output_size) -> image`: normalized positive weights; identity reconstruction invariant.

Schema evolution must be explicit; existing research records remain readable or produce a migration message. Do not silently reinterpret missing fields.

## 5. Notebook plan: purpose, cells, outputs and dependencies

All new names below are planned Kaggle slugs under cubingultimate. Use private datasets by default and verify actual links only after publication.

### N0: existing btp-validation

Purpose: finish fresh repeatability/placement/cache/confirmation. Preserve its publication history. Outputs: raw jobs, predictions, eligibility, cache decisions, source/runtime provenance and ZIP. Exit: approved teacher policy or documented M0 fallback. No student training in this notebook.

### N1: btp-data-audit (CPU, Internet only for authorized acquisition)

Cells: (1) config/budget and exact input versions; (2) source verification; (3) archive inventory/licensing; (4) scene IDs and pair registration checks; (5) duplicate audit against evaluation; (6) training/validation split freeze; (7) train-only augmentation plan; (8) deterministic preview contact sheet; (9) manifest validation; (10) export checksums and dataset bundle.

Inputs: authorized training corpus; original NH-HAZE archive only for split-preserving evaluation derivatives. Outputs: train manifest, validation manifest, rejection ledger, license inventory, duplicate candidates requiring review, distribution summary. No clean held-out targets are passed into training or generation.

### N2: btp-teacher-targets (GPU)

Cells: (1) verify locked teacher and training manifest; (2) fixed smoke subset; (3) load once; (4) compare one resident prediction with the cold reference; (5) generate explicit bounded shard; (6) log failures and per-image hashes immediately; (7) visual/content audit; (8) archive partial or complete shard; (9) exact resume instructions.

Inputs: training-only hazy/clean pairs and fixed pretrained weights. Outputs: paired teacher predictions with IDs, seeds, model provenance and validity flags. HazeGen augmentation is a separate optional cell group and separate artifact; never silently replaces the original hazy input.

### N3: btp-student-pilot (GPU only after budget approval)

Cells: (1) hashes and train/validation disjointness; (2) build model and parameter/MAC audit; (3) tiny CPU save/reload smoke; (4) 100-update GPU throughput/memory profile; (5) calculate remaining update/time budget; (6) train A0 pilot up to total 2,000 updates or two GPU-hours; (7) development evaluation; (8) learning curves and collapse checks; (9) resumable checkpoint/export.

The 100 updates count toward the 2,000-update/two-hour pilot budget. No automatic continuation into a full training campaign. Output is labeled pilot, not final student.

### N4: btp-student-ablation (approved bounded training jobs)

One config/run per ablation and seed; avoid an unattended nested sweep. Cells: setup, manifest and initialization hash checks, load explicit A0/A1/A2 config, bounded train/resume, select checkpoint using training-corpus validation, evaluate frozen checkpoint on NH-HAZE development, export artifacts. A3 runs only after A1/A2 show independent value.

Outputs: student checkpoints, optimizer/scheduler state, RNG states, consumed updates and cumulative training time, loss curves, sample exposure counts and development scores. Same update budget, initialization and crop sampling for matched ablations where possible.

### N5: btp-guidance-quality (GPU)

Cells: baseline verification; CPU gradient/finite-difference tests; fixed image-region masks; baseline and one-candidate smoke; fixed candidate matrix; confidence/correction diagnostics; PSNR/SSIM/LPIPS and region metrics; timing/backward accounting; fail-safe gate report; evidence archive. No training, no changes to original timestep schedule.

### N6: btp-global-tiles (GPU)

Cells: native manifest and geometry audit; CPU identity/coverage tests; resident-runner equivalence smoke; fixed tile count/time estimate; one-image C1/C2 smoke; bounded development matrix; seam/quality/memory report; failure/coverage table; archive. Full-frame teacher OOM stays in evidence. Do not retry with a smaller image under the same variant ID.

### N7: btp-final-confirmation (GPU)

Cells: read signed-off finalist config and hashes; display locked metric protocol; confirm no test-set tuning; run repeated matched development timing; emit uncertainty/regression report; explicit held-out gate; run fixed test matrix; create tables/figures; package complete evidence. No hyperparameter search in this notebook. Development and held-out outputs are stored separately.

### N8: btp-release-demo (inference only)

Cells: verify release manifest; select student/reference/native mode; load once; run example or user input; display side-by-side outputs and honest timing; export image/config. No optimizer, no automatic checkpoint downloads outside the release manifest. Link model card and describe data/geometry limitations.

### Notebook common cell contract

Every notebook starts with immutable input IDs and resource budget, then source/hash verification, runtime setup, CPU checks, execution, analysis and artifact export. Record Internet requirement explicitly. Save a version with outputs; download and re-open its archive before considering the run preserved. If an input is missing, stop before compute. On error, write a structured record and archive evidence in finally. Registry paths are explicit and portable relative to the archive root; require deliberate remapping after extraction.

## 6. Kaggle dataset and publication plan

| Planned dataset | Contents | Consumers |
|---|---|---|
| Existing btp-paper-weights-2 | Original three checkpoints | Reference/generator/teacher notebooks |
| Existing btp-eval-nh-haze | Frozen 512 paired crops | Reference/student/guidance evaluation |
| btp-research-source | Versioned code bundle and configs | N1-N8, pinned exact version per run |
| btp-student-training-v1 | Licensed train/validation pairs, scene manifest, license | N2-N4 |
| btp-teacher-targets-v1 | Training-only teacher output shards and provenance | N3-N4 |
| btp-hazegen-training-v1 | Optional HazeGen augmentation with clean-source IDs | A2/A3 only |
| btp-eval-native-v1 | Original-size and eligible larger paired crops, inherited split | N6/N7 |
| btp-student-checkpoints-v1 | Selected inference weights, config, model card, hashes | N7/N8 |
| btp-research-evidence-v1 | Curated immutable experiment archives, no credentials | Reporting/review |

The exact training corpus is a dependency, not silently chosen. First inventory accessible locally licensed data; then propose a bounded separate paired/synthetic training corpus acquisition (for example a suitable RESIDE subset after confirming availability/license/size). Approve the download and hash exact archives. Do not invent a dataset version or assume its license permits redistribution to Kaggle. If redistribution is restricted, publish preparation instructions and user-supplied mounts instead.

Version each dataset after content verification. A source update must not silently alter an in-flight experiment. Copy archive hashes and dataset numeric versions into each notebook's run record. If API attachment version pinning is unavailable, verify expected bundle hashes inside the notebook and fail on mismatch. Code datasets never include raw checkpoint files or local auth stores.

## 7. WP0 — evidence recovery, protocol and teacher freeze

**Files:** evidence.py, protocols.py, configs/research/protocol-v1.json, teacher-lock.json; tests/test_evidence.py and test_protocols.py; extend workflow.py only compatibly.

- [ ] Write failing tests for stale evidence, changed source/data hash, incomplete job coverage, duplicate records and path traversal in archives.
- [ ] Implement `audit_archive(path: Path) -> EvidenceAudit` and `freeze_teacher(spec: dict, evidence: Path, output: Path) -> dict`.
- [ ] Fetch current validation outputs and preserve a content-addressed archive; inspect every planned versus actual job.
- [ ] Recompute metrics/eligibility from raw records and images. Record historical missing evidence separately.
- [ ] Select an eligible/repeat-confirmed execution policy; otherwise freeze M0 with documented rationale. The teacher's model/guidance remains the reference method.
- [ ] Record deployed module parameter counts, live dtype bytes, on-disk weight bytes and what is excluded/shared. Hash teacher-lock.json.
- [ ] Run tests and export a teacher-baseline summary with known limitations.

**Exit:** an auditable teacher and frozen protocol, not necessarily a faster teacher. Failing fresh M0 inference blocks teacher-target generation until repaired.

## 8. WP1 — training data and evaluation geometry

**Files:** training_data.py, geometry.py, train/validation/native manifests; tests/test_training_data.py and test_geometry.py.

### Scene-level data contract

Each row contains source_scene_id, pair_id, source archive/hash/license, hazy/clean hashes, original dimensions, registration status, split, augmentation parent, generator seed/config and teacher target hash if present. Original clean scene identity controls split even when several hazy variants or crops exist. Group aliases/duplicates before splitting. Similarity detection flags near-duplicates for review; it does not assert perfect duplicate removal.

- [ ] Test split leakage through crops/aliases, mismatched dimensions, unsupported channels, missing files, repeated IDs and broken hashes.
- [ ] Split the separate training corpus by scene with seed 42; default 90% train/10% corpus-validation, unless an official split must be retained. Freeze before preprocessing or generation.
- [ ] Use corpus-validation for checkpoint selection and early stopping. Use the existing NH-HAZE development scenes for the limited cross-domain research decisions. Log every such decision to expose development overfitting.
- [ ] Begin with deterministic aligned 256 training crops; identical transforms for hazy, clean and teacher targets. Allow horizontal/vertical flips only after verifying task suitability; no independent color transforms between pair members.
- [ ] Create native/multi-size NH-HAZE manifests by importing source-scene split membership from the existing manifest. No resplitting or upsampling.
- [ ] For 512/768/1024 crops, retain only source scenes large enough; publish eligibility counts per size before any model runs. Keep paired crops aligned. Native rows preserve original dimensions.
- [ ] Generate a manifest audit report and example grid; no model-score-based sample filtering.

**Exit:** no known train/test scene leakage, valid aligned pairs and explicit licenses. If adequate training data is unavailable, student training is blocked; tiling/guidance CPU work may continue.

## 9. WP2 — persistent teacher session and trustworthy timing

**Files:** resident.py, test_resident.py; reuse telemetry/loader logic through tested adapters rather than copying an entire upstream script.

Interface: `ResidentTeacher(config, weights, device)`; `predict(input_path, output_dir, seed) -> RunRecord`; `close()`. Model construction/loading occurs once. Guidance object, targets, transmission maps, sampler schedule and per-image counters must reset for every prediction. CLIP prompt embeddings may be cached only with a key including text, checkpoint, dtype and device assumptions. Reusing a cached embedding is not permission to reuse the image-conditioned tensor.

- [ ] CPU fake-model tests count one load over two images and independent image conditioning.
- [ ] Test changed checkpoint/config rejection, A-B-A input sequence, RNG reset and cleanup after an exception.
- [ ] Test that latent gradients still flow through active guidance and parameters stay frozen.
- [ ] Add CUDA integration smoke: cold versus resident outputs for the same image/seed; then A-B-A to detect leaked state. Apply existing systems-equivalence criteria.
- [ ] Record model load time separately. Warm end-to-end timing includes preprocessing, conditioning, inference and output processing under the declared contract; disk export timing is separately reported. Synchronize CUDA at boundaries.
- [ ] After a declared warmup, run five measured repetitions per timing scene; report median/range. Record peak counters per image and process-lifetime host peak separately, since ru_maxrss cannot be reset as an image-local peak.
- [ ] On OOM, stop the worker and mark subsequent jobs unexecuted; restart explicitly rather than assume an intact model state.

**Exit:** equivalent predictions, no image-state leakage, measured warm/cold timing. Asynchronous transfer work remains conditional on profiler evidence of overlap; no speculative non_blocking=True speed claim.

## 10. WP3 — offline targets and HazeGen augmentation

**Files:** teacher_targets.py, target shard manifests, N2 and tests/test_teacher_targets.py.

- [ ] Add CPU fixture tests for deterministic shard assignment, idempotent completed-item reuse by hash, partial-write recovery and failure retention.
- [ ] Generate an eight-image training-only smoke shard; verify geometry, finite pixels, hashes and pairing.
- [ ] Profile time/image. Proposed first shard budget: at most 100 inputs or two T4-hours, whichever occurs first, with the exact approved input list.
- [ ] Store teacher predictions separately from ground truth. Teacher targets are auxiliary supervision; no replacement of known clean targets.
- [ ] Optional HazeGen: generate one variant for a fixed training-clean subset with seed derived from scene ID and base seed, then compare equal-count/equal-update training. More seeds are a later diversity ablation.
- [ ] Audit obvious registration/content drift using fixed rules and manual blinded samples; report rejection rate by haze severity. Preserve challenging valid cases.
- [ ] Archive after each item/shard; retain a manifest for failed items. Training either uses a predeclared complete subset or a new reviewed manifest, never silently skips arbitrary failures.

**Exit:** useful training artifacts with provenance. This stage consumes compute; it is not student optimization itself.

## 11. WP4 — compact student architecture and training

**Files:** student/model.py, losses.py, trainer.py, inference.py; N3/N4; corresponding tests and YAML configs.

### Architecture specification

Implement or vendor a licensed NAFNet-style four-level encoder-decoder with residual output, width 16 initial candidate, encoder block counts [1,1,1,2], middle blocks 2, decoder counts [1,1,1,1]. These settings are an initial architecture proposal, not an asserted parameter count. Audit counts before GPU use. Width 32 is the only planned first capacity alternative. Do not force an arbitrary minimum parameter count if a smaller model works.

Input/output: NCHW RGB in [0,1], same spatial dimensions. Internally pad to the required downsampling multiple with an explicit safe policy; crop back exactly. Training output can remain unclipped for gradient flow; evaluation follows a fixed clipping policy. Handle tiny inputs without invalid reflection padding. Do not call a restoration-pretrained checkpoint dehazing-pretrained. Initial matched ablations use identical random initialization unless a separately justified checkpoint is locked.

### Loss specification

For clean target y, hazy x, student S and frozen teacher target T:

`L_rec = mean(sqrt((S(x)-y)^2 + 1e-6))`

`L_base = L_rec + 0.1 * (1 - SSIM(S(x), y))`

`L_A1 = L_base + 0.1 * mean(abs(S(x)-T(x)))`

Use a pinned differentiable SSIM implementation with declared window, range and reduction; do not assume it is numerically identical to the existing evaluation SSIM. Test identity, finite gradients and channel/range handling. Ground truth remains primary. If teacher supervision harms clean-target quality, retain A0 rather than hiding that result. A perceptual training loss is outside the first ablation matrix.

### Initial optimizer and bounded training

AdamW, learning rate 1e-4, betas (0.9,0.999), weight decay 1e-4, gradient-norm clip 1.0; seed 42. Float32 pilot first. Effective batch 4 using microbatch 1 with four-step accumulation if required; record both. A batch/precision change creates a new config and matched rerun, not an undocumented rescue.

The first 100 optimizer updates profile time/memory and count toward the pilot cap. Warmup 100 updates followed by cosine decay to 1e-6 at the declared total update budget. Stop at 2,000 optimizer updates or two cumulative T4-hours including profiling, whichever first. Every 250 updates: corpus-validation, checkpoint and artifact archive. Check deadline during the loop and save before the platform limit. Resume restores optimizer, scheduler, RNG, data position, update count and cumulative budget; resuming must not reset the two-hour cap.

Select best checkpoint by corpus-validation PSNR under a fixed tie-break (higher SSIM, then earlier update). Retain last checkpoint for recovery. Do not repeatedly select checkpoints using held-out NH-HAZE scores.

### Training experiments

| ID | Data | Loss | Purpose |
|---|---|---|---|
| A0 | Fixed paired training corpus | L_base | Establish small-model baseline |
| A1 | Same pairs and targets | L_A1 | Isolate teacher supervision |
| A2 | Controlled HazeGen mixture | L_base | Isolate generated-data benefit |
| A3 | Same selected mixture | L_A1 | Test combination only after A1/A2 |

A2 controls total optimizer updates and sample exposures; compare fixed 50/50 original/generated sampling with an equal-budget original-only control. Record unique scenes versus pair count. Finalists use training seeds 42,43,44 if approved. A one-seed pilot cannot establish training stability.

### Tests and promotion

- [ ] Forward shape/range contract at odd and tiny sizes; parameter count audited against config.
- [ ] Two-image/two-update CPU smoke; finite loss/gradients; weights change; teacher targets detached.
- [ ] Save/load reproduces predictions; interrupted/resumed fixture matches uninterrupted fixture under the same deterministic setup.
- [ ] Duplicate/held-out input rejected before training; invalid architecture/checkpoint mismatch rejected.
- [ ] 100-update profile and written budget estimate before longer pilot approval.
- [ ] Compare teacher, A0 and finalists on identical 512 geometry using quality and warm/cold runtime protocols.

**Proposed A target:** >=10x fewer deployed parameters, >=5x lower warm end-to-end latency, no worse than -0.5 dB mean PSNR / -0.01 mean SSIM versus teacher. Record LPIPS; proposed regression ceiling +5% relative LPIPS, with an absolute rule registered instead if the baseline is near zero. Disclose per-scene losses and artifacts. Failure to meet quality gates means a smaller model was built, not that a satisfactory replacement was achieved.

## 12. WP5 — confidence-aware guidance quality

**Files:** guidance.py and tests/test_guidance.py; opt-in adapter/config in reference execution; N5. Never edit the vendored baseline silently.

Implementation observation: upstream WeightedSSIMGuidance computes a structural-loss gradient and multiplies it by a detached transmission/density map. Confidence weighting must explicitly augment that existing correction, not be described as introducing physics from scratch. Inspect apply_cond_fn and alignment operations before finalizing tensor space/shape.

### Fixed first candidate

Use inference-only local luminance variance and bright/low-saturation ambiguity as separate diagnostic maps. Begin with one proxy: local variance stability over a 15x15 window. Define a bounded detached reliability `r = variance/(variance + epsilon_var)` and weight `w = floor + (1-floor)*r`; candidate floor 0.25. Register epsilon_var using training-corpus validation input statistics before candidate scoring, with the calculation saved. Low contrast is an imperfect uncertainty proxy; explicitly test whether it wrongly suppresses useful guidance under dense haze.

First insertion: weight the existing RGB correction before its documented propagation to the latent path, matching channels/geometry. Do not interpolate a confidence map into latent space without explicitly testing the intended mapping. Preserve the active guidance window, repeat count, schedule and latent gradient path. Initial implementation leaves confidence detached.

### Ablation matrix

B0 original; B1 statistic-stability repair alone (only if a concrete instability is reproduced); B2 confidence weighting; BU uniform scale equal to B2's mean correction weight; B3 confidence plus cap only after B2 is useful. A repair is reported separately from a research contribution. If no alignment bug exists, omit B1 rather than invent one.

For the cap, register a correction-norm rule on development calibration data before scoring finalists; preserve direction and limit magnitude. Do not set thresholds by inspecting held-out artifacts. First grid is at most floors {0.25,0.5} with one fixed variance setting, plus B0/BU. No unlimited tuning loop.

- [ ] Reproduce baseline gradient with w=1 and cap disabled.
- [ ] Test zero/constant/bright inputs, finite outputs, shape/range checks, identity confidence and preserved nonzero latent gradients.
- [ ] Verify clean target is absent from inference inputs.
- [ ] Test same mean-weight uniform control and save diagnostic map statistics.
- [ ] One-scene GPU smoke, then two-scene screen, then all ten development scenes for the finalist. Screen metrics alone cannot promote a final quality claim.
- [ ] Use fixed masks for sky/bright areas, boundaries and fine structure, annotated without candidate outputs. Save masks and missing-region coverage.
- [ ] Compare regional and global quality, correction norms, latent gradient norms, VAE backward time and failures.

**Proposed B target:** >=0.3 dB PSNR gain OR >=5% relative LPIPS reduction on the fixed panel, with no more than 0.2 dB PSNR / 0.005 SSIM regression in the alternate metrics, no LPIPS worsening when claiming PSNR gain, and no increased severe hallucination count in blinded review. Report uncertainty; do not select whichever metric happened to improve after the run. Declare the primary metric in B's config beforehand.

## 13. WP6/WP7 — native-resolution tiling and global fusion

**Files:** geometry.py, tiling.py, fusion.py, tests/test_geometry.py, test_tiling.py, test_fusion.py; N6.

### Geometry and reconstruction

Tile side 512, overlap 128, stride 384. Append the terminal tile position at dimension minus tile size when needed; deduplicate positions. For smaller dimensions pad explicitly and retain original crop coordinates. Weight windows use positive smooth weights with a small fixed floor so image borders never have zero total weight. Accumulate weighted predictions and weights in float32 on CPU to avoid retaining all tile tensors on GPU; normalize and crop to original dimensions. This bounds GPU working memory, not unlimited host memory: estimate host accumulation bytes and reject images beyond a declared host budget or use a tested disk-backed accumulator.

Per-tile seed derives from fixed base seed, source_scene_id and tile coordinates using a stable hash, not Python's randomized hash. Batches begin at one tile. Preserve tile order and all padding records.

### Global alignment design

Generate a separate aspect-preserving coarse global prediction at a supported size, with explicit pad/unpad transforms; verify the upstream resize behavior rather than assume a longest-side cap is honored. Upsample only the global appearance guide to native size. Local tile output supplies native detail.

First alignment is additive low-frequency correction, not unconstrained variance scaling: smooth global/local RGB with the same declared filter, compute their low-frequency difference, clip per-channel correction to a predeclared range (initial candidate +/-0.05 in [0,1]), and add before blending. Record clipped fraction. This avoids amplifying constant-patch variance. Optional affine alignment is a separate later ablation, not quietly introduced.

### Matrix and tests

C0 native full-frame teacher when feasible; C1 overlapping tiles only; C2 C1 plus global additive appearance alignment. Optional C3 reliability weights only after C2 passes. Also test compact student full-frame versus student tiled independently; do not mix teacher-to-student and tiling effects in a single comparison.

- [ ] Identity predictor reconstructs input within float tolerance on regular, odd, tiny and one-pixel-wide shapes.
- [ ] Every native pixel has positive weight; no duplicated or missing output coordinate; paired transform roundtrip exact.
- [ ] Constant images stay constant; fixed seeds reproducible; changed tile order either has a documented tiny float effect or a fixed deterministic order.
- [ ] One-image GPU smoke with C1/C2, including all global-pass cost.
- [ ] Screen eligible 512/768/1024 and native development panels. Publish eligible source counts before metrics; if a size has too few scenes, state that limitation rather than upsample.
- [ ] Compare PSNR/SSIM/LPIPS, boundary-band absolute error and gradient error against ground truth. Define seam bands as +/-8 pixels around internal tile-grid boundaries; use the same union mask for compared variants.
- [ ] Boundary discontinuity alone cannot count as success: smoothing may hide seams. Report fine-detail gradient metrics and fixed visual crops.
- [ ] Retain C0 OOM/timeouts in coverage; quality comparisons use the intersection explicitly, and absolute candidate scores against clean targets use declared full coverage.

**Proposed C target:** at least one larger native/geometry tier that the full-frame reference cannot process within the same GPU budget, with full output coverage; and/or >=20% boundary-band error reduction versus C1, with <=0.2 dB global PSNR loss, no SSIM/LPIPS material regression and no observed detail collapse. Report added global-pass latency. A bounded-memory capability gain does not imply a speed gain.

## 14. WP8 — confirmation, uncertainty and acceptance

**Files:** research_eval.py, confirmation.py, tests/test_research_eval.py, test_confirmation.py; N7.

- Validate image dimensions and source IDs before scoring. Metrics never compare a restored image to its hazy input as a clean reference.
- PSNR/SSIM retain the existing RGB/range/border protocol for comparable results. Pin LPIPS network, weights/hash, input normalization and spatial reduction; record missing metric as unavailable, not zero.
- Store per-scene scores first. Average scenes equally, with seeds/repeats nested within scene. Do not inflate sample size by counting tiles or repeated timings as independent scenes.
- Use scene-level paired bootstrap intervals with fixed seed and 10,000 resamples for descriptive mean differences. Ten development scenes still give limited certainty. Report effect size and intervals, not a casual "statistically significant" claim.
- For finalists, five timed warm repetitions after one excluded warmup on each designated timing scene. Report cold startup separately. Keep timing overhead and preview status fixed.
- Candidate status is proposed, implemented, CPU-validated, GPU-smoke, screened, confirmed, rejected or inconclusive. Publication is not confirmation.
- If any scene fails, list it and the planned denominator. An apparent metric gain on only successful/easier scenes is not acceptance.
- Before held-out evaluation, declare quality targets, primary metric, known trade-offs and model-selection rationale. Targets are design decisions recorded before results; no post-hoc relaxation.

## 15. WP9 — combining successful tracks without misleading claims

Not every track composes mechanically. Confidence-aware diffusion guidance does not run inside a feed-forward student. Valid combinations are:

1. Student + tiled/global inference: direct deployment combination, compare student-native, student-tiled and student-global-tiled.
2. Improved diffusion guidance as an offline teacher: requires generating new targets and retraining a matched student. Compare the same student/data/training budget under original versus improved teacher targets. Teacher quality improvement does not automatically transfer.
3. Faster validated residency + any reference algorithm: supporting implementation control, not an additional quality contribution.

Run a small factorial comparison for only proven components: baseline, A alone, C alone where applicable, A+C. For guidance-teacher transfer, add a separately named training experiment. Do not claim a joint A+B+C model if B exists only in an offline teacher and its transfer has not been tested.

Freeze finalists.json with checkpoint/config hashes, allowed geometry, pre/postprocessing, primary metrics and rejection criteria. Then run the 45 held-out project scenes with fixed seeds. No notebook cell in that stage selects a winner from test scores. If results fail targets, report failure and reserve a new independent dataset for future hypothesis tuning rather than recycle the same test set as unseen.

## 16. WP10 — release, application and documentation

- Add a student inference CLI accepting checkpoint, config, input, output, device and explicit tile/global settings. Validate file hashes and dimensions; preserve input files.
- Integrate with the local viewer through existing interfaces. Clearly label reference versus trained student versus experimental variant. Display actual runtime/memory where measured, not static speed claims.
- Package inference-only weights separately from resume checkpoints. Loading for inference does not need optimizer state. Compare storage using the same deployment packaging rules and precision.
- Publish model card: training data/license, split, architecture, counts, precision, checkpoint hash, metrics/protocol, limitations, failure crops and intended use.
- README progress sections list implemented versus measured outcomes, with experiment IDs and links. Remove stale status only when replaced by verified evidence; preserve historical results.
- Final report includes a trajectory table, A/B/C ablations, quality-size-latency Pareto plot, native-size memory/coverage plot, failed hypotheses and exact reproduction commands.
- Professor presentation: explain original method, M0/M1 negative result, new hypotheses, experimental controls, accepted improvements and limitations. Avoid equating research activity with demonstrated benefit.

## 17. Compute planning and approvals

| Stage | Initial bound | Decision before expansion |
|---|---|---|
| CPU unit/model smoke | tiny fixtures; two optimizer updates | correctness and finite gradients |
| Teacher targets | 8-image smoke, then <=100 inputs or 2 T4-hours | measured cost, pair validity, explicit budget |
| Student profiling/pilot | 100 updates included in <=2,000 updates / 2 T4-hours | user training approval, actual memory/time |
| Student ablations | separate approved jobs, equal total updates | A0 learning curve and adequate data |
| Guidance smoke | 1 scene, then 2 scenes / fixed small grid | finite gradients and artifacts |
| Tiling smoke | 1 eligible image / C1 and C2 | identity/coverage tests and tile cost |
| Development confirmation | explicitly enumerated ten-scene matrices | archive size/time estimate and previous gates |
| Held-out | frozen finalists, fixed 45 scenes | frozen config and approved total budget |

Total campaign hours cannot be credibly estimated before profiling. Estimate `setup + sum(planned jobs * measured median time)` and publish a conservative allowance; include teacher generation, global passes and I/O. Each notebook has a hard time/job cap and checkpoint grace period. GPU quota is checked before scheduling. Do not launch parallel sessions simply because T4 x2 is offered. Paid compute is not part of the default plan.

No full training authorization is implied by this planning document. A bounded pilot approval does not authorize A0-A3 across three seeds or full-scale generation.

## 18. Publication and Git workflow

1. Use an isolated clone/worktree; inspect current remote before editing. Preserve notebook hashes and avoid broad staging of the unrelated home repository.
2. Keep commits scoped: protocol/data; resident session; student core; student training; guidance; tiling; evaluation; release.
3. Required checks: uv run ruff check .; uv run ruff format --check .; uv run pytest. Run relevant Torch CPU tests in the pinned runtime, notebook cell compilation, config validation and archive/hash checks. Distinguish skipped GPU integration tests from passed tests.
4. Review each meaningful package before a GPU pilot; fresh review for final integration. Resolve scientific leakage and false acceptance bugs before publication.
5. Build source bundle from the exact committed tree, record project commit and bundle SHA256, upload a new private source version, download it and verify the hash.
6. Upload notebook metadata with correct dataset attachments, Internet/GPU settings and bounded runtime. Save without execution when changing documentation only. Launch only the reviewed experiment configuration.
7. After a run, verify saved output availability; download archive and validate planned/actual job coverage. An API COMPLETE status alone is not proof of successful model experiments.
8. Publish approved small reports/model card to GitHub; keep large/private artifacts in appropriate private datasets or authorized model storage. Check source/data licenses before any visibility change.
9. Update README with real verified links, notebook versions and source commits. No fictitious future links. Leave collection management to the user.
10. Commit/push execution needs user authorization; prior requests govern the active task, but this plan itself does not silently trigger a publication or training job.

## 19. Risks, fallback experiments and stop conditions

| Risk | Detect | Response |
|---|---|---|
| Lost draft results | missing raw archive/jobs | rerun explicitly; never claim historical diagnosis |
| Student underfits | training/validation curves both weak | inspect implementation/data; bounded width alternative; new budget approval |
| Student overfits | train improves, validation declines | stop by fixed validation rule; more appropriate data, not test tuning |
| Teacher errors copied | A1 worse than A0 | reduce/remove teacher term in a newly declared experiment; retain A0 result |
| HazeGen corrupts registration | content/geometry audit | reject corrupted pairs, report distributions; do not filter by final model score |
| Confidence harms dense haze | regional errors/gradient suppression | reject proxy or retain uniform/baseline; no claim of universally reliable confidence |
| Fusion hides seams by blur | boundary score improves, detail worsens | reject fusion; retain correct C1 tiling capability |
| Larger inputs OOM | memory/failure records | retain failure; test registered tiled mode separately |
| Multi-track combinations regress | factorial ablations | ship individual modes, not an unjustified combined default |
| Insufficient budget | profiled campaign cost exceeds quota | prioritize A0 + C1/C2; B becomes a documented secondary study |

Fallbacks do not guarantee the two-axis success criterion. If compression quality is inadequate and tiling does not provide a useful new capability, the release remains partial. The honest report states what was implemented and what failed.

## 20. Delivery milestones and completion checklist

**M0: trustworthy teacher** — exported evidence, current status known, teacher/protocol lock, count audit.

**M1: data-ready project** — licensed separate training corpus, duplicate/split audit, larger native manifests, CPU loader tests.

**M2: trained small-model pilot** — reproducible checkpoint and resume, measured counts/quality/latency; not yet a final replacement.

**M3: second-axis prototype** — correct tiling with native output and/or tested guidance-quality candidate, independent of student.

**M4: two-axis evidence** — A plus B/C meets pre-registered development criteria, repeats and failure checks; independent results before combination.

**M5: release candidate** — frozen finalists, held-out report, deployable student/native interface, source/data/checkpoint provenance and professor-facing documentation.

- [ ] All older notebooks preserved and linked.
- [ ] Current validation results reconciled; historical missing evidence acknowledged.
- [ ] No training/evaluation scene leakage detected in the available audit.
- [ ] Compact student weights exist and parameter/storage counts are measured.
- [ ] At least two independent benefits demonstrated, or incomplete status explicitly stated.
- [ ] Every metric tied to an experiment/config/data/checkpoint identity.
- [ ] Native output dimensions and border coverage verified.
- [ ] Quality and systems criteria remain distinct and unchanged retrospectively.
- [ ] Uncertainty, failures, compute cost and negative findings included.
- [ ] New notebooks, datasets, code and reports reproducible from published versions.
- [ ] README/model card/demo reflect measured results rather than targets.

## 21. Sources and limitations of transfer

The linked three-track plan contains primary references: the original CVPR 2025 method, NAFNet, SinSR and RPD-Diff. They motivate architecture/distillation/region-aware processing; they do not establish performance of this project. This detailed plan is based on inspected repository interfaces and the prior research review, not a new claim of exhaustive literature coverage. Before a novelty claim, perform a focused related-work comparison of the final implemented method, including compact dehazing and global/local fusion approaches.

## Review focus (cross-cutting)

1. Stale or mixed experiment state must never authorize a candidate: WP0/archive tests, WP8 provenance checks.
2. Same original scene under different crops/generated haze must never cross splits: WP1 alias/duplicate tests.
3. Resident inference must not leak targets/RNG/conditioning between images: WP2 A-B-A and cold-equivalence tests.
4. Guidance must preserve latent gradients and avoid target leakage: WP5 identity/gradient/input tests.
5. Tiling must preserve every output pixel, paired geometry and memory limits: WP7 identity/border/odd-size tests.

Self-review: each requested stage maps to a work package, each notebook has input/output/exit contracts, and publication is separated from experimental success. Unknown training-corpus availability and full compute cost remain explicit acquisition/profiling decisions rather than invented facts.

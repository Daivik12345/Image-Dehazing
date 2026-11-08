# Selective Offloading and Cache Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Publish a separate btp-offload-cache Kaggle workflow comparing selective placement and independent allocator flushing, preserving all prior notebooks and sources.

**Architecture:** Extend the reference runner with explicit cache policies and the evaluation runner with named policy/cache variants. Add CPU-side repeatability analysis and fail-closed screening eligibility. A new notebook orchestrates bounded, explicit stages; it does not silently choose a winning model.

**Tech Stack:** Existing Python/uv, PyTorch 2.6 CUDA runtime, numpy/Pillow, Kaggle CLI. No new project dependencies.

**Spec:** docs/research/EVALUATION_AND_OFFLOADING_GUIDE_2026-11-06.md sections 10-12; user confirmed selective offloading + independent cache flushing.

## Global Constraints

- Preserve existing notebooks byte-for-byte locally and on GitHub; do not update btp-source or btp-eval-source.
- Use the same private weights and frozen NH-HAZE dataset. Create btp-offload-cache-source only.
- Float32, seed 42, original schedule/guidance, identical geometry, no previews. All failures retained; no automatic shrinking/retries.
- Old CLI/config defaults preserve previous behavior and summary keys. M3 retains its old no-per-guidance-flush semantics.
- User authorizes implementation, private Kaggle upload and GitHub push. Save notebook without launching compute; larger stages opt-in.
- Collection management remains manual per user instruction.

## Review Focus

- Legacy M0/M3/R0 behavior remains unchanged when no new option is supplied.
- Policy/cache variants never mix in aggregate summaries or pair with the wrong reference.
- Failed, incomplete or profiled baseline records cannot establish numerical tolerance or candidate eligibility.
- Cache-stage comparison cannot silently choose a failed candidate or use held-out quality for selection.
- New notebook restores only its separate source dataset and never edits existing outputs/notebooks.

### Task 1: Independent cache policies

Files: src/btp/memory.py, src/btp/reference.py, tests/test_memory.py, tests/test_reference.py.
Interfaces: resolve_cache_policy(policy, requested) returns guidance/phase/never. should_empty_cache(cache, event, offloaded=False) controls after-conditioning and guidance flushes.
- [x] Tests first: M0 default guidance, M3 default phase, R0 never; invalid options rejected; phase/never never flush per guidance; no-module guidance never flushes.
- [x] Implement CLI --cache-policy and manifest provenance without altering old defaults.
- [x] Run targeted tests and CPU latent-gradient smoke.

### Task 2: Variant matrix and repeatability evidence

Files: src/btp/benchmark.py, src/btp/optimization.py, tests/test_benchmark.py, tests/test_optimization.py.
Interfaces: config variants=[{id,policy,cache_policy}], reference_variant=id, mutually exclusive with policies. summarize(jobs, reference_variant='M0') retains legacy keys. analyze_repeatability(directory) checks complete unprofiled repeated M0 output/timing variation; screen eligibility checks complete baseline and candidate coverage with observed baseline tolerance, never auto-accepting a speed win.
- [x] RED tests for separate variant labels, reference pairing, invalid/duplicate variants, preserved failures, incomplete/profiled repeats.
- [x] Implement config validation, job metadata/commands/CSV and comparisons per variant.
- [x] Implement repeatability report and conservative eligibility for manual cache placement choice; equality gate only, no quality-score-based winner.
- [x] GREEN tests, including synthetic output changes and OOM records.

### Task 3: Notebook, configs and documentation

Files: configs/offload-cache/, configs/kaggle-offload-cache/, notebooks/btp-offload-cache.ipynb, docs/OFFLOAD_CACHE.md, README.md.
- [x] Preserve notebook hashes before edits. Duplicate evaluation notebook setup.
- [x] Default stages: baseline repeatability (6 jobs), then M0/M2-U/M2-C/M2-R screening (8 jobs). Cache comparison (6 jobs) and confirmation (12 jobs) are opt-in stages with budgets shown before execution.
- [x] Keep reference M0 in every placement comparison. Cache comparison changes only cache mode on manually chosen eligible placement and pairs against that placement's guidance flushing. No silent winner promotion.
- [x] CPU smoke precedes GPU commands. Results/logs/configs export even if a stage fails. New timestamped output directories.
- [x] Update README with measured prior pilot and new unmeasured workflow links; no speedup claims.
- [x] Validate notebook syntax/configs, full tests, Ruff and baseline hashes. One fresh read-only review; fix important findings with regressions.

### Task 4: Publication

- [ ] Fetch/update isolated publishing checkout, copy only reviewed files and user's README update. Validate checkout.
- [ ] Create private btp-offload-cache-source, reuse weights-2 and eval-nh-haze datasets; publish notebook with --no-run.
- [ ] Commit/push authorized changes and verify remote SHA, notebook cells, attachments and preservation.

## Decisions and evidence

Implementation proceeds inline under the user's explicit instruction; no additional plan approval is necessary. No large downloads, model training, dataset changes, or GPU execution are required for publication. Advanced resident/async execution is outside the confirmed two improvements.

Progress 2026-11-08: cache-policy and variant-comparison regressions observed RED then GREEN. Repeatability/eligibility and isolated-notebook tests pass. Full suite 69 passed before final review. Source preservation hashes recorded in /private/tmp/btp-offload-cache-preserved.json.

Final review: fresh read-only review found one important gap, omitted numerical runtime flags in repeatability/screening provenance. Five parameterized regressions reproduced it; runtime signature now requires cuDNN, both TF32 flags, deterministic_algorithms and cudnn_benchmark for baseline/control/candidates. No minor findings. GPU equivalence/performance and remote publication remain evidence-limited until explicitly verified.

Final validation: 74 tests passed in both workspace and isolated publishing checkout; uv run ruff check . and uv run ruff format --check . passed. Temporary PyTorch CPU smoke confirmed identical outputs/latent gradients. All three prior local notebook hashes match, and publishing checkout has no diff for prior notebooks. No GPU inference launched. New private source dataset is ready.

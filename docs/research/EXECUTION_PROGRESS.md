# Multi-axis implementation ledger

## Evidence baseline

- Verified the supplied session archive: 6 repeatability, 8 placement and 6 cache inference jobs completed.
- Recomputed repeatability and placement from raw predictions; retained M0/guidance as conservative reference.
- Recorded archive identity in evidence/validation-session-inventory.json and checkpoint/runtime provenance in configs/research/teacher-lock.json.
- No repeated-confirmation or held-out research result is claimed.

## Active implementation

- Evidence ZIP auditor: CRC, size budget, unsafe paths and duplicate jobs; five CPU tests passed.
- Student architecture/training/recovery: in progress.
- Separate training data audit and split-preserving native manifests: in progress.
- Tiling/fusion correctness and memory budgets: in progress.

## Outstanding prerequisites

A separate licensed student training corpus has not been acquired or approved. Existing NH-HAZE held-out scenes must not become training data. Long student training and teacher-target generation require concrete compute budgets before launch. No trained student, guidance quality gain or native-image GPU gain exists yet.

## Rulings

- Use conservative M0/guidance while faster placement equivalence remains unconfirmed. Cost: slower offline targets, avoids accepting an unvalidated default.
- CPU work proceeds independently of corpus acquisition and GPU budget approval. Smoke weights will be labeled random/test-only and never published as trained dehazing weights.

## November 17 implementation and review

- Student: default width16 measured at 1,724,275 parameters. Added safe export/inference, A0/A1 audited pilot CLI, fixed-crop policy and validation selection. Two-update synthetic CPU smoke and interrupted resume pass; no real training.
- Data: audit, deterministic source grouping, paired crops and native/512/768/1024 NH-HAZE manifests implemented. Locally prepared all four tiers preserve 10 development/45 test scenes. No raw dataset publication.
- Tiling: complete coverage, positive fusion, declared CPU workspace limit, bounded global correction, seam and gradient diagnostics. Stochastic teacher tile execution remains unintegrated; current CLI uses deterministic student inference.
- Guidance: RGB correction adapter, confidence/uniform/cap controls, diagnostic-only clipping after bicubic resize. Baseline input remains unchanged.
- Resident: real upstream AccSamp M0 adapter; fresh per-image state, source/runtime identity and bounded teacher-target generation. CUDA equivalence remains unverified.
- Added five independent research notebooks; all expensive stages are disabled by default. Baseline notebooks unchanged.
- Independent reviews caught initialization seeding, mutable audited data, save-time accounting, unsigned gradient arithmetic, target provenance and confidence range issues. Added regressions and fixed them. Scene-bootstrap now requires an explicit planned denominator.
- Training corpus research and acquisition/Kaggle steps recorded in TRAINING_DATA_NEXT_STEPS.md; no download launched.

Verification: uv run pytest: 167 passed, 20 optional-runtime skips. Isolated Torch CPU: 11 student tests, 7 guidance tests, 3 resident-gradient tests passed. Ruff lint/format checked; final publication check recorded separately below. Concurrent intermediate checks failed while workers were writing modules; those are not final-state failures.

Remaining: source-corpus acquisition/permissions; explicit GPU pilot approval; cold/resident CUDA equivalence; guidance calibration; full native GPU experiments; A2/HazeGen ablation orchestration; complete confirmation/held-out campaign; UI/model-card/release evidence. None is represented as an achieved research gain.

Ruling: use fixed seeded crops for the bounded initial student pilot, documented in CLI records. Cost: this is not the planned full random-augmentation campaign; compare all first ablations under the same crop policy.
Ruling: preserve save-time accounting with a hash-bound budget sidecar. Cost: resume requires both files; a missing sidecar fails safely rather than resetting the budget. The final tiny sidecar write remains outside measured duration.

## November 18 publication

- GitHub implementation commit: 0b92c91, pushed to main.
- Private Kaggle btp-research-source reports ready.
- btp-data-audit, btp-teacher-targets, btp-student-pilot, btp-guidance-quality and btp-global-tiles each saved as version 1 with --no-run.
- No GPU execution or training; old notebooks untouched. New upload metadata and links recorded in the repository.
- Downloaded source-only dataset for verification: bundle byte-identical to upload; all 185 embedded file hashes passed.

## November18 OTS intake and approved A0 execution

- Imported100 deterministic source pairs from all four nested OTS haze parts;90train/10validation.113 clean sources excluded by pre-existing RGB/256 geometry requirements before selection; full exclusion inventory retained.
- All paired file/hash/split checks and known evaluation-overlap audit pass. Thumbnail pairing review found no obvious mismatches; exhaustive related-viewpoint review remains incomplete.
- Private pilot dataset created; source version2/code15b4ba8 published.
- User explicitly approved first100updates or1200training seconds only. Kaggle student notebookv2 completed100updates on TeslaT4;training14.205seconds;peak allocated382152192bytes.
- Inference6,964,282bytes/hash verified; resume checkpoint+sidecar retained. No validation at100updates, no quality claim, no continuation launched.
- Tests before launch:175passed/21optional skips;12isolatedTorch student checks passed;Ruff passed.
- Ruling: eligibility filtering occurs before deterministic selection, preserves exclusions, and does not inspect model outcomes. Cost: pilot represents RGB sources >=256pixels, not every source in OTS.
- Ruling: explicitly increase corpus workspace cap to1GiB after dry estimate654,947,200bytes. Cost: greater host RAM requirement; model, precision and images unchanged.

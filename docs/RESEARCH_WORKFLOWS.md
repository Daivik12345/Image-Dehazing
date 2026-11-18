# Multi-axis research implementation

Updated November 17, 2026. This is executable research infrastructure, not a claim of a trained smaller dehazer or improved quality.

## What is available

| Component | Implemented behavior | Remaining evidence |
|---|---|---|
| Student | Four-level width-16 restoration network; 1,724,275 parameters; differentiable reconstruction/SSIM/teacher loss; bounded resumable training; hash-checked inference | Separate licensed corpus, approved training pilot, quality/latency comparison |
| Native inference | Overlapping tiles, positive normalized windows, CPU streaming fusion, explicit host workspace limit, native output geometry | Trained checkpoint; GPU quality, seam and memory comparison |
| Guidance | Detached local-variance reliability, matched-mean uniform control, optional norm cap | Calibration of epsilon, upstream GPU smoke, independent quality experiment |
| Persistent teacher | One checkpoint load, fresh sampler/conditioning/guidance per image, M0 residency, structured errors | Cold/resident equivalence and A-B-A CUDA validation |
| Data/evidence | Training manifest audit, alias/hash leakage checks, frozen NH-HAZE geometry, immutable protocol records, scene-paired summaries | Manual near-duplicate/registration audit; completed experiment records |

The student replaces **DiffDehaze at deployment**, not HazeGen. HazeGen and the pretrained teacher remain offline tools. Parameter bytes are not checkpoint file bytes, and no teacher-to-student compression ratio is reported until the actual deployed teacher modules are counted.

## Preserve the reference

Existing baseline, baseline+, evaluation, offload/cache and validation notebooks are untouched. The new notebooks are repository artifacts; a repository link does not mean a Kaggle version has been published or executed.

- [btp-data-audit](../notebooks/btp-data-audit.ipynb): training corpus integrity and evaluation overlap audit.
- [btp-teacher-targets](../notebooks/btp-teacher-targets.ipynb): bounded targets after teacher validation.
- [btp-guidance-quality](../notebooks/btp-guidance-quality.ipynb): opt-in calibrated guidance controls.
- [btp-student-pilot](../notebooks/btp-student-pilot.ipynb): architecture and two-update CPU smoke before real training.
- [btp-global-tiles](../notebooks/btp-global-tiles.ipynb): native tiled student inference after a trained artifact is supplied.

Run `uv run python scripts/build_research_notebooks.py` to regenerate these new notebooks. Run Ruff afterward. Build the allowlisted source ZIP with `uv run python -m btp.kaggle`; upload it privately under the filename `btp-research-source.bundle` and attach that source dataset. Model weights, datasets and environments are excluded from the source bundle. Runtime provisioning downloads Python/PyTorch dependencies; it is not performed merely by generating a notebook.

## Local CPU verification

Base tests intentionally avoid requiring Torch. In the already provisioned isolated runtime:

```sh
PYTHONPATH=src uv run --no-project --python .venv-paper/bin/python python scripts/smoke_student.py
PYTHONPATH=src uv run --no-project --python .venv-paper/bin/python python -m unittest discover -s tests -p 'test_student*.py'
PYTHONPATH=src uv run --no-project --python .venv-paper/bin/python python -m unittest discover -s tests -p 'test_guidance.py'
```

The smoke trains a width-two random fixture for two updates. It is never exported as a research checkpoint. The default width-16 parameter count is measured separately.

## Data boundary

`training_data.audit_training_manifest` requires schema version 1, dataset/release/source archive identity, license and paired source-scene records. Each row specifies pair ID, canonical scene ID, train/validation split, hazy/clean relative paths and hashes, source geometry and registration status. Pass the complete evaluation inventory from `configs/evaluation/nh_haze_manifest.json`. Audit failures stop training. A declared license/registration field is not independent legal or visual verification; retain the source license and complete manual reviews.

`geometry.prepare_geometry` derives native, 512, 768 and 1024 views from the existing original NH-HAZE archive. All four locally prepared tiers retain the frozen **10 development / 45 test** source scenes. No image is upscaled to qualify. Generated images remain local and outside Git. These evaluation images are not the student training corpus.

## Training and recovery

Use `BoundedTrainer.create` for a freshly seeded model. The constructor accepting an existing model deliberately preserves those weights. Record initialization hash, architecture, dataset version and source commit. Ground truth remains primary; add teacher loss only for A1. Compare A0/A1 with identical initialization and data. A2 generated-data orchestration remains a subsequent experiment.

The proposed pilot is at most 2,000 optimizer updates or 7,200 cumulative seconds, including the first 100 profiling updates. This is a cap, not approval to run. Use corpus validation, never held-out NH-HAZE, for checkpoint selection. The reusable trainer enforces update/time caps and restores optimizer, scheduler, data position and RNG. Copy **both** `last.pt` and `last.pt.budget.json` when resuming; the sidecar binds the checkpoint hash and accounts for save time. Missing/stale sidecars fail. The final small ledger write is outside measured budget time. One indivisible operation may overrun a deadline; record this limitation.

## Native inference

```sh
PYTHONPATH=src uv run --no-project --python .venv-paper/bin/python python -m btp.student.cli \
  --image INPUT.png --checkpoint TRAINED.pt --sha256 EXPECTED_SHA256 \
  --architecture configs/research/student/architecture.json \
  --device cuda:0 --tiled --tile-size 512 --overlap 128 --output NEW_OUTPUT_DIRECTORY
```

The CLI refuses existing output directories and verifies checkpoint architecture/hash. It writes native-size prediction and runtime/hash records; inference failures preserve failure evidence. It never downloads or substitutes a random model. The fusion budget covers accumulation/workspace, not total process RAM, the input image, or model memory. CPU/GPU transfer and fusion are included in prediction time; model load and file IO are separate. This single invocation is not a warmed latency benchmark.

For C2, pass an explicitly native-aligned global prediction to `predict_tiled`; retain clipping diagnostics. Global prediction cost and resizing must be recorded. Do not apply stochastic teacher tiles through the deterministic student helper without explicit coordinate-derived seeds. Full-image PSNR/SSIM and target-referenced seam/gradient errors are needed; smooth seams alone are insufficient.

## Scientific decisions

`configs/research/protocol-v1.json` separates systems equivalence from intentional quality changes. The strict historical M0 variation gate is unchanged. `protocols.validate_run_records` checks the planned denominator before aggregation. `confirmation.paired_summary` pairs by scene/seed/repeat and bootstraps scenes, not individual repeats. Callers must first validate planned coverage and matching provenance: matched incomplete tables alone cannot establish a gain.

No finalists, held-out outcomes, trained release or successful compression claim exists yet. Freeze finalists only after independent development confirmation. Publish negative outcomes and failures alongside successful jobs.

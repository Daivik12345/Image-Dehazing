# Validation, persistent inference and compression implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the staged optimization workflow without changing prior notebooks or evaluation criteria, then investigate persistent inference and compact students.

**Architecture:** Keep stage state and evidence validation in importable modules; a new notebook orchestrates bounded experiments. Distinguish implementation, CPU validation, GPU measurements and accepted findings. Freeze decisions before held-out evaluation.

**Tech Stack:** Python, uv, PyTorch, existing benchmark and optimization modules, Kaggle CLI.

**Spec:** User request dated 2026-11-15; docs/OFFLOAD_CACHE.md and docs/research/EVALUATION_AND_OFFLOADING_GUIDE_2026-11-06.md.

## Global constraints
- Preserve all existing notebooks and source datasets.
- Keep seeds, splits, precision and quality gates unchanged.
- Do not manufacture missing evidence or silently approve failed candidates.
- Ask before paid compute, large downloads or long training.
- User authorized Kaggle publication and GitHub push.

## Review focus
- Missing historical results: record unavailable evidence; fresh results cannot explain a historical rejection.
- Restarted kernels: reconstruct state from explicit stage directories, never latest modification time.
- Changed predictions/provenance: recompute gates from evidence rather than trust cached JSON.
- Partial and failed runs: retain failures and disallow candidate selection.
- Held-out leakage: lock policy/configuration before evaluating held-out data.

## Tasks
- [ ] Recover historical evidence if available; otherwise record the unresolved rejection.
- [ ] Add tested restart-safe stage selection with actionable rejection reasons.
- [ ] Add a new staged notebook with persisted decisions and output archives.
- [ ] Run fresh repeatability, placement, cache and repeated confirmation, preserving source/runtime provenance.
- [ ] Broaden development coverage and geometry before locking a final configuration for held-out evaluation.
- [ ] Implement and validate persistent model reuse, separately measuring cold startup and warm inference.
- [ ] Profile transfers; implement asynchronous overlap only if measurements support it.
- [ ] Define a compact-student training/evaluation protocol with a CPU smoke run; obtain approval for the bounded training job before launch.
- [ ] Run Ruff and behavioral tests, preserve previous notebook hashes, publish code/data/notebook and push GitHub.

## Current evidence limitations
The saved Kaggle version exposes no experiment files. The user confirms no newer archive/version was saved. The live draft is blank; user approved reloading it to attempt recovery. The earlier two-image table is preliminary evidence only; the later rejection cannot yet be diagnosed from raw records.

# OTS pilot intake and bounded A0 profile

November 18, 2026. User supplied `OTS_BETA.zip` locally and requested private Kaggle dataset creation.

- Archive size: 16,442,395,507 bytes.
- SHA256: `95519a3dc7460c22bf0de503b460ac7a02f6dfcb35ae6790529a71c68ad99003`.
- Contains nested clear/depth/four haze archives plus redundant extracted part1 images. Pilot sampling uses all four nested haze parts, not the redundant partial extraction.
- No license/README found in the supplied archive. Preserve this unresolved provenance; no public redistribution license is asserted.
- First preparation failed the existing RGB/256-pixel requirements. Inspection found 113 incompatible clean images. Eligibility is now determined before seeded scene selection and retained explicitly in the inventory; no performance-based exclusion or resizing is used.
- Selected hazy pairs must still match geometry and RGB requirements; failures are not silently replaced.
- Intended pilot: 100 unique source scenes, 90 train/10 corpus-validation, seed42, one deterministic haze variant per scene. NH-HAZE evaluation membership remains separate.

The user approved only the first A0 GPU profile: one Kaggle T4, at most 100 optimizer updates or 1,200 seconds of training in this segment, whichever first. The underlying total 2,000-update/7,200-second configuration remains unchanged for compatible resume; no continuation is authorized by this approval. Deadline checks occur at microbatch boundaries, so final indivisible work/checkpoint export can add a small overrun. No teacher generation, paid compute or larger experiment is authorized.

Dataset preparation, full audit and publication results are appended after verification. Structural synthetic pairing checks do not constitute exhaustive manual registration review.

## Verified preparation

- All 2,061 clean source IDs paired with haze inventory. 1,948 meet clean RGB/256 geometry eligibility; all 113 exclusions retained.
- Selected 100 scenes: haze parts1/2/3/4 contribute24/33/21/22. Native image geometry preserved; no resizing.
- Audit passed:90 train/10 validation,100 source groups, hashes/geometry and known NH-HAZE identity overlap checked.
- All100 paired thumbnails inspected: no obvious incorrect hazy/clean scene pairing. Repeated landmarks/viewpoints exist; exhaustive near-duplicate grouping is incomplete. This profile does not establish generalization.
- Corpus storage/workspace estimate654,947,200 bytes. Explicit1GiB corpus budget passes dry audit; model/runtime memory is separate.
- Private pilot bundle64,133,141 bytes; publication target cubingultimate/btp-student-training-pilot.

## Publication and approved execution

- Private dataset created and reports ready: https://www.kaggle.com/datasets/cubingultimate/btp-student-training-pilot (version1).
- Source version2 published with project commit `15b4ba883b2ac624370c07d6b68cc33b10e5c732`; code pushed to GitHub.
- `btp-student-pilot` version2 launched with the user's explicit bounded A0 approval. Notebook pins data by bundle/manifest hashes, records source commit, and exposes only CUDA device0. Kaggle accelerator allocation is T4x2, but this code uses one card.
- Platform timeout3600seconds includes setup/export; training segment cap1200seconds/100updates remains independent and stricter.
- Final status: COMPLETE. Saved records and checkpoint hashes verified locally.
- Verification before launch:175 base tests passed,21 optional-runtime skips;12 isolated Torch student tests passed; two-update CPU fixture passed; Ruff lint/format passed.

## Verified A0 profile result

- Notebook version2 completed on Tesla T4, CUDA device0, Torch2.6.0+cu118.
- 100 optimizer updates /400 microbatch exposures; training wall time14.205seconds, cumulative budget14.204seconds. Setup/export excluded from that training measurement.
- GPU peak allocated382,152,192bytes (364.45MiB); reserved419,430,400bytes (400MiB), PyTorch counters only.
- Inference checkpoint6,964,282bytes;1,724,275 parameters.
- No validation checkpoint reached (interval250); export explicitly last/unselected. No PSNR/SSIM or generalization gain is claimed.
- Source/data hashes match locked identities. Resume checkpoint and sidecar hash verified; inference artifact reload and finite-shape CPU check passed.
- Local artifacts retained in weights/student-a0-profile-v2; weights are excluded from Git. Full per-update record: evidence/a0-profile-kaggle-v2.json.
- No continuation started. A further training segment requires new explicit approval. Linear extrapolation from this short run excludes later validation/IO and is not a promised runtime.

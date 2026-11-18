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

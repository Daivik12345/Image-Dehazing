# Training data and next execution steps

Research checked November 17, 2026. No dataset was downloaded in this search.

## Recommended starting point

**Use a fixed subset of RESIDE OTS for the outdoor student pilot**, after checking the downloaded release's terms and retaining its attribution. This is a project recommendation based on the outdoor NH-HAZE target domain, not a claim that synthetic OTS training will generalize to real nonuniform haze. The [official RESIDE-β page](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) provides Outdoor Training Set downloads through Dropbox and Baidu. It separately lists RTTS as a testing set and unannotated real images; those are not paired clean-target training substitutes.

For a smaller initial plumbing corpus, the [official RESIDE-Standard page](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) provides ITS, explicitly the Indoor Training Set. It also lists SOTS as a testing set; do not mix SOTS into training. ITS is an indoor-domain alternative, so report that mismatch if used for an outdoor student.

[O-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/) supplies 45 real outdoor hazy/clean scene pairs and advertises a 547 MB archive. Its small size makes it a possible later real-data adaptation study, not a replacement for a broad training corpus. Such use would require a separately frozen split and overlap/provenance review.

The official pages reviewed expose download links but do not establish an explicit redistribution license in their visible text. Check archive terms or seek clarification from the maintainers before redistributing. A Kaggle mirror's license label is not enough to establish rights to the original images. Do not invent a license field to pass the audit. Private upload also needs appropriate permission.

## Steps after implementation

1. Open the official OTS page and select its training release. Check the download size and usage terms before starting a large download. Keep the original archive, exact URL, date, license/terms and SHA256.
2. Prepare a bounded pilot inventory: proposed **100 distinct clean source scenes**, one fixed haze variant per scene initially. This is a proposed selection budget, not a completed dataset. Group every haze variant/crop by its clean source identity; do not split individual hazy filenames independently.
3. Freeze a scene-based 90/10 train/validation split with seed 42. Preserve aliases and parent IDs. Keep all NH-HAZE evaluation scenes out of this inventory. Verify hashes and registration, and manually review potential near-duplicates across corpora.
4. Write the schema-v1 manifest consumed by `btp.training_data.audit_training_manifest`. Run the audit against the complete `configs/evaluation/nh_haze_manifest.json` inventory. See `tests/test_training_data.py` for a concrete minimal fixture schema. Passing file checks does not replace manual content review.
5. Where redistribution is permitted, create a **private** Kaggle dataset containing the paired images, manifest, attribution/terms and audit report. Proposed slug: `btp-student-training`; it is not a published link. Record the owner, version number and manifest hash. Attach it to the new student notebook, preserving older notebooks.
6. Run the CPU fixture and real-data dry audit first. For A0, train only against clean targets. Start with a 100-update GPU profile after explicit budget approval; count those updates/time inside the same eventual 2,000-update or 7,200-second cap. Record T4 model, precision, batch/accumulation, source identity, initialization hash and peak memory.
7. Inspect the profile before approving the remaining pilot. Use the same optimizer/schedule and resume checkpoint plus `.budget.json` sidecar; do not reset cumulative time. Select checkpoints on corpus validation only.
8. Validate the persistent teacher against cold M0 before using its outputs for A1. Then propose the exact training-only target list and budget. Keep predictions separate from ground truth. Compare A0/A1 under identical initialization, data and update budget.
9. Evaluate the trained candidate on frozen development images at matched 512 geometry. Measure warm/cold latency separately. Run native tiling as an independent capability experiment, measuring full-image quality and seams. No held-out tuning.
10. Freeze finalist checkpoint/config/protocol hashes before the one-shot project-held-out evaluation. If quality gates fail, retain that negative result; do not label parameter reduction alone a successful dehazer replacement.

## What remains conditional

No training data has been acquired, no real student has been trained, and no two-axis gain has been established. The code prepares these experiments; corpus acquisition, CUDA equivalence, budget approval and measured evaluation remain necessary. Existing pretrained teacher training exposure is not fully known, so “project-held-out” must not be described as proof of globally unseen data.

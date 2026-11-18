# Gated development batch, November18

User explicitly approved up to four sequential free Kaggle GPU-session hours: A0 continuation, fresh cold/resident validation, up to90 teacher targets only after that gate passes (target stage<=2hours), matched A1, bounded one-scene guidance/tiling smokes. Student segments<=20training minutes. No paid compute, automatic budget expansion or held-out evaluation.

## Completed A0

Kaggle student-pilot version3 resumed the original version2 checkpoint using source15b4ba8 and the same data/config/optimizer/budget. Reached2000total updates,265.067seconds additional training,279.271seconds cumulative. PyTorch peak allocated382,152,192bytes. Corpus-validation checkpoint selected by registered PSNR/SSIM/earlier tie-break. At2000updates mean PSNR25.8339dB/SSIM0.93460 on10 fixed256-pixel OTS validation crops. This is training-corpus validation, not NH-HAZE, teacher comparison or generalization evidence. Best/last/inference hashes verified locally; artifact retained under weights/student-a0-2000-v3.

## Data and calibration

63-AC-bit DCT perceptual hashes reviewed all100OTS clean images and55NH-HAZE evaluation crops. Cross-split threshold8 found no candidate pairs. This is not proof that related views/landmarks are absent; no split changed. Exact file/hash identity checks remain separate.

Confidence epsilon registered before guidance scoring: median of per-source positive-variance medians on20seeded training hazy inputs, window15/replicate padding, epsilon floor1e-8. Result0.0016402184210533438; clean targets and evaluation inputs are not used in calibration. Saved configuration retains exact selected identities and hashes.

## Next gate

Fresh two-scene development experiment: three cold M0 repetitions per scene plus resident A-B-A-B. Runtime/checkpoint/input identities must match, and each resident output must meet observed cold MAE/max-pixel variation. No relaxed threshold. Actual deployment parameter count will be measured on the loaded teacher. No target generation is authorized by a mere COMPLETE notebook status; eligibility must be true from verified evidence.

Teacher-target expansion depends on measured inference cost. The cap does not guarantee90targets fit; incomplete coverage cannot silently qualify A1. Guidance/tiling are intentionally changed algorithms and use separate quality evidence, not the systems-equivalence gate.

# Evaluation implementation ledger

Spec: docs/research/EVALUATION_AND_OFFLOADING_GUIDE_2026-11-06.md
Preserve baseline+ notebooks and its Kaggle source dataset. User approved NH-HAZE download and private upload. Implement in BTP, publish through existing isolated checkout.

- [x] Strict manifests/crops, PSNR/SSIM, optional LPIPS evaluator.
- [x] Placement policies and timing/memory/profiler instrumentation.
- [x] Repeat runner, failure preservation, aggregation/comparisons.
- [x] NH-HAZE deterministic development/held-out manifests.
- [ ] Separate eval notebook and source dataset; CPU/GPU pilot.
- [ ] Review and repository publication.

Rulings: no changes to guidance, precision or sampling. The first pilot is bounded and uses development crops. Warm resident, asynchronous copying and adaptive dispatch require separate validation; do not claim them as measured improvements. All proposed interfaces must either work or reject unsupported settings explicitly. No baseline notebook or baseline Kaggle dataset changes.

Verification so far: 24 targeted tests passed; CPU PyTorch output/latent-gradient smoke passed; SSIM agrees exactly with scikit-image 0.24.0 on the seeded fixture. Exact-match aggregation regression was observed failing and corrected. Full suite pending.

Ruling: retain reusable implementation under existing src/btp and the research guide's architecture, consistent with the user's explicit earlier ai-lab-to-btp rename; do not introduce a second package.
Ruling: M3 is a named M1/no-flush ablation, not an automatically chosen winner. Warm resident and asynchronous policies are rejected/deferred until fixed placement is measured, as the research plan requires.
Ruling: do not commit or push these new repository changes without a current explicit request; the supplied AGENTS boundary reserves those for user review. Kaggle notebook and private dataset publication are explicitly requested and proceed.

NH-HAZE official archive downloaded and verified as 55 pairs. Private dataset cubingultimate/btp-eval-nh-haze created; source dataset and notebook remain to be uploaded. Frozen development pilot uses scenes 08 and 13. No GPU result claimed yet.

## Publication preparation — 2026-11-07

User explicitly authorized Kaggle publication and GitHub push, superseding the earlier no-push ruling. Final review fixes: crop geometry bounds, descendant cleanup on timeout, and non-finite LPIPS preservation each reproduced RED then passed GREEN. Full suite: 62 passed. uv run ruff check . and uv run ruff format --check . passed.

Final: minor (deferred): aggregation uses scene_id rather than a shared source identity for hypothetical multiple crops per source. The frozen dataset has exactly one crop per source; only that layout is validated for this release.

Review evidence limits: CUDA performance/output equivalence and real LPIPS weights remain untested; no held-out quality, generalization, or paper reproduction claim. Resident/async remain staged follow-ups. Baseline template hash corresponds to the read-only Kaggle pull /private/tmp/btp-eval-source/btp-baseline.ipynb. Remote publication will be verified separately.

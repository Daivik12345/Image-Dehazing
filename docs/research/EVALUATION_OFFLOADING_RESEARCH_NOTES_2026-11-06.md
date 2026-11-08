# Research notes: evaluation and offloading (November 6, 2026)

Primary sources checked:
- Baseline full text https://arxiv.org/html/2503.19262v1 : RTTS (4,322), Fattal qualitative, no-reference FADE/Q-Align/LIQE/CLIPIQA/ManIQA/MUSIQ/BRISQUE. Paired benchmarks are an additional BTP track, not its main protocol.
- O-HAZE official: https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/ (45 outdoor pairs).
- NH-HAZE official: https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/ (55 outdoor scenes; 330MB advertised download, not downloaded).
- Dense-Haze official: https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/ (33 outdoor pairs).
- Torch 2.6 CUDA semantics, profiler and reproducibility docs checked; use pinned version links rather than latest-only APIs.
- scikit-image 0.24 metrics docs checked: explicit range and SSIM channel handling.
- LPIPS author repository checked: RGB [-1,1], lower better, version/backbone must be fixed.
- HF memory docs checked: model/group/leaf offload tradeoffs; APIs not directly available in custom ControlLDM.
- PyTorch pinned-memory tutorial checked: non_blocking alone does not establish overlap; CPU reads and buffer reuse need synchronization.

Repository observations:
- src/btp/reference.py currently stores no proper phase timings; current one-image CLI reloads weights every run. Cannot claim warm resident timing by repeating this CLI.
- src/btp/memory.py offloads entire UNet and ControlNet; reference wrapper invokes it before upstream apply_cond_fn can early-return outside active guidance.
- Upstream guide is active only when t_stop < int(t[0])+1 < t_start; repetition and RGB/latent spaces also affect work. Actual defaults may already make all invoked callbacks active; count before optimizing.
- Source runner currently sets CUDA_VISIBLE_DEVICES=0 in upstream scripts. Two-GPU orchestration needs explicit device support; separate worker env alone is not sufficient.
- Current max_memory counters include process lifetime/loading. Must distinguish lifetime peak versus reset warmed-region peak.
- Existing successful examples are 573x512 and767x512 before padding, not proof native multi-megapixel real benchmarks fit.

Scope: research guide only. No evaluation harness implementation, dataset or metric-weight download, GPU benchmark or new optimization performed.

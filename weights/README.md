# Pretrained weights

The original checkpoint links are listed in the upstream README. The original
SD Hugging Face URL returned HTTP 401. Accessible community mirrors were found
and approved for downloading; URLs, sizes and SHA-256 hashes are recorded in
`configs/checkpoints.json`.

- `v2-1_512-ema-pruned.ckpt`: sd-research mirror linked from upstream issue 12.
- `stage1.pt`, `stage2.pt`: DJKAD community mirror pinned to commit
  f6d334fcb867542118f6bc64cdf8c7812581fc37. **Unverified against the authors' originals.**

The authors' original stage weights remain available via
https://pan.baidu.com/s/1vbxEwftJC9nUaMXJ-t9sww?pwd=8egg (code `8egg`).

Partial downloads have `.part` suffixes and are not used for inference.
`uv run python scripts/verify_weights.py` checks size and SHA-256 before publishing
them. Per-file provenance JSON is retained and copied into experiment manifests.
`uv run python scripts/download_weights.py` resumes the documented mirrors.
Total download is about 8.12 GB; approval is required before starting fresh large downloads.

Only trusted checkpoint tensors are loaded with `weights_only=True`, and the
adapter validates exact model state keys/shapes. A compatible state dictionary
does not establish that a third-party mirror is identical to the authors' files.

## Baidu verification update — November 4, 2026

The user-supplied Baidu stage1.pt and stage2.pt were hashed and match the
previously downloaded files byte-for-byte. Earlier mirror uncertainty has been
resolved against these supplied files; changing the file source will not change
the output. See configs/baidu-verification.json and docs/KAGGLE.md for the
original-preprocessing CUDA workflow.

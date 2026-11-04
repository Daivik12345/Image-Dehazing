# Run HazeGen and DiffDehaze on Kaggle

## What we verified locally

Your two Baidu checkpoint files match the previously downloaded copies exactly:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `stage1.pt` | 1,452,743,609 | `07e8c56b6bdf5f616d85bfc24e3e652c2bb74884c89cfc25756ee4f27ecbebe3` |
| `stage2.pt` | 1,452,743,609 | `d7dada8b659c1554b58b0f6f1353e9bac80849191094307f45de0303ebe962da` |

There is no reason to replace the local bytes or expect different results merely
from swapping these identical files. Their supplied Baidu provenance is now
recorded in `configs/baidu-verification.json` and local weight provenance files.
The shared SD checkpoint is already in `weights/v2-1_512-ema-pruned.ckpt`.

The local viewer's reduced preprocessing differs from the original scripts:
it caps the longest side, uses replicate padding, and supports mixed precision.
The original scripts instead upscale the **short side to 512 when necessary**,
use zero padding to multiples of 64, and resize the final output back to the
input size. For example, a 640×427 image becomes 767×512 before zero padding,
not 512×342. The Kaggle notebook uses the original scripts and these original
operations; it does not apply the local AlignOp coverage patch or mixed precision.

**No Kaggle GPU experiment has been run from this session.** The notebook has
no fabricated outputs, metrics, or completed-run claims. It prepares qualitative
reference inference; reproducing the paper's benchmark tables requires the
specified evaluation data and metric implementations as a separate step.

## 1. Prepare these local files

The ready-to-upload source archive is `artifacts/btp-kaggle.zip`.
The notebook to import is `notebooks/kaggle_reference.ipynb`.

Upload the following three model files together, with these exact names:

1. `weights/stage1.pt`
2. `weights/stage2.pt`
3. `weights/v2-1_512-ema-pruned.ckpt`

The three weights total about 8.12 GB. Uploading them directly avoids spending
notebook time downloading them again. The source archive excludes weights,
credentials, `.git`, virtual environments, and past experiment logs. It includes
all tracked upstream files, our runner, an example clean garden photograph,
notebook, guide, and a source hash manifest. Keep model uploads private.

If you change source code later, rebuild the source archive locally:

```bash
uv run python -m btp.kaggle
```

## 2. Create two private Kaggle datasets

At https://www.kaggle.com/datasets choose the new-dataset/upload action.

1. Create a **private** dataset called `btp-paper-weights` containing the three
   weight files above. Wait for all uploads and dataset processing to finish.
2. Create another **private** dataset called `btp-source` containing
   `artifacts/btp-kaggle.zip`. Kaggle may expand this archive automatically;
   the notebook supports both the ZIP and expanded folder layouts.
3. Note the weights dataset owner, slug and version, for example
   `YOUR_USERNAME/btp-paper-weights/versions/1`.

No API key, Kaggle credential file, or public model upload is needed for these
manual steps. Dataset creation and privacy are described in the
[Kaggle dataset documentation](https://www.kaggle.com/docs/datasets).

## 3. Import the notebook and select a GPU

1. Create a new Kaggle Python notebook.
2. Use the editor's notebook import/upload action (usually **File → Import
   Notebook**) to import `notebooks/kaggle_reference.ipynb`.
3. Use **Add Input / Add Data** to attach both private datasets you just created.
4. In notebook **Settings / Session options**, enable **Internet** and choose an
   available NVIDIA GPU accelerator, such as **T4 ×2**. Availability depends on
   your account and quota; if unavailable, check Kaggle's account verification
   and accelerator/quota messages.
5. Keep the notebook private. Run cells interactively, one at a time.

The runner deliberately uses **GPU 0 only**. Two T4s are not automatically a
single pooled-memory device. No multi-GPU training is launched.
See [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks).

## 4. Run setup cells in order

In the first code cell, replace:

```python
WEIGHTS_DATASET_VERSION = "EDIT_ME/weights-dataset/versions/1"
```

with your actual dataset identity, for example:

```python
WEIGHTS_DATASET_VERSION = "YOUR_USERNAME/btp-paper-weights/versions/1"
```

Then run the cells in sequence:

1. **GPU check:** `nvidia-smi` must list an NVIDIA GPU. Do not continue on CPU.
2. **Source restore:** copies/extracts the uploaded source into
   `/kaggle/working/BTP`, verifying every packaged source hash.
3. **Runtime:** installs uv if needed and creates an isolated Python 3.10
   environment at `/tmp/btp-kaggle-env`. It installs PyTorch 2.6.0 and torchvision
   0.21.0 CUDA 11.8 wheels, plus pinned inference dependencies. Setup needs
   Internet and can take several minutes. Kaggle's kernel environment is left
   intact. The GPU check inside this environment must also pass.
4. **Checkpoint verification:** automatically locates all three files under
   `/kaggle/input`, checks their sizes and SHA-256 hashes, and symlinks them to
   `/tmp/btp-reference-weights`. It does not copy the 8 GB of weights into outputs.
5. **Tiny CPU smoke:** tests original components with tiny random weights.
   Its outputs are only plumbing checks; they are not model reproduction results.

The explicit CUDA wheel pairing follows the official
[PyTorch previous-version installation instructions](https://docs.pytorch.org/get-started/previous-versions/).
PyTorch 2.6 is a compatibility choice for scoped restricted loading of legacy
checkpoint metadata. This is **not an exact Torch 2.2.2/xformers environment
reproduction**. No unrestricted pickle loading or Lightning callback execution
is enabled. Native PyTorch SDPA is used in place of optional xformers.

## 5. Run the models and watch results

The notebook's helper runs one image in a fresh process for each case. Wait for
one case to finish before starting the next. This releases GPU weights between
cases. Batch size is one, precision is float32, and seed is 42. This is our recorded test seed; it is not a claim to recover the publication's unknown sampling seed.

### Notebook section 6: HazeGen

The bundled `examples/clean.png` is a clean garden photograph. This cell runs
`inference_stage1.py`, preserving the hazy prompt, 50-step sampling and `w=0.85`.

- Clean input → hazy output.
- Replace `CLEAN` with a clean image from your attached dataset if desired.
- Do not treat the repository's already-hazy `inputs/` images as clean training
  examples for haze generation.

### Notebook section 7: DiffDehaze with spaced sampling

This runs `inference_stage2.py` on the authors' supplied hazy `inputs/1.png`:

- Real hazy input → dehazed output.
- The original 50-step sampler and `remove dense fog` prompt are preserved.

### Notebook section 8: DiffDehaze with AccSamp

This runs `inference_accsamp.py` on the same real hazy input:

- Uses the original 50-step schedule, `tau=0.8`, `omega=0.6`, and guidance scale
  `s=0.1` from the upstream configuration.
- Actual model-call count differs from the full spaced trajectory because
  AccSamp skips intermediate sampling portions.
- Original AlignOp behavior is retained, including possible border-coverage
  artifacts. Do not silently patch it when comparing reference behavior.

### Notebook section 9: Complete roundtrip

This feeds the saved HazeGen output into DiffDehaze/AccSamp. The clean image,
synthetic haze and restored result are useful for inspecting pipeline behavior.
A roundtrip is not a substitute for evaluating the paper's real hazy datasets.

During each case the cell displays progress/logs and periodic decoded sampler
states. These **noisy-latent previews are diagnostics**, not final predictions.
The completed cell displays the input and final image separately.

Each output directory contains:

```text
input.png          # saved input
inputs/input.png   # single-image directory consumed by upstream
config.yaml        # exact generated configuration
manifest.json      # seed, source snapshot, hashes, versions, GPU and dataset identity
status.json        # current state / error / completion
latent-*.png       # optional noisy-latent diagnostic previews
final/input.png    # actual final upstream model output
```

You can reduce preview frequency by passing `preview_every=0` to `run_case` if
preview decoding adds too much overhead. This changes observation cost, not the
sampling schedule. Do not lower resolution or steps and then claim a paper-level
comparison. Runs use new output names; existing experiments are not overwritten.

## 6. Save results before ending the session

Run the last cell to create `/kaggle/working/btp-results.zip`. Download it using
the displayed link or the notebook Output/Files panel. It contains actual images,
logs, hashes, manifests and package versions; it excludes weights and environments.

Save a notebook version with outputs and confirm the archive is visible in that
saved version. A **Save Version / Run All** operation can launch a fresh execution
and consume GPU quota again; download current results first. Treat interactive
session files as temporary until downloaded or confirmed in saved outputs. Stop the GPU session when finished so it does not continue consuming your available quota.
Notebook output behavior is described in the
[Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).

## Troubleshooting

| Problem | Action |
| --- | --- |
| GPU is unavailable | Check accelerator selection, account eligibility, and current quota; stop rather than running this reference on CPU. |
| `CUDA is not available` in the isolated environment | Check the CUDA wheel install and `nvidia-smi`; preserve the logs and do not switch silently to CPU. |
| Missing/duplicate checkpoint | Attach one dataset containing each exact filename. Remove redundant attached weight datasets. |
| Hash mismatch | Wait for uploads to finish, or re-upload the exact local file; do not bypass the assertion. |
| Cannot find source archive | Attach `btp-source`; either a ZIP or a folder containing `source-manifest.json` must be present. |
| CUDA out of memory | Confirm no second GPU job is running; restart the session, rerun setup, use one image and `preview_every=0`. Record any precision or size changes as a separate non-reference experiment. |
| Output directory exists | Use a new case name, such as `01-hazegen-retry`, or start a fresh session. |
| AccSamp border artifacts | Preserve reference output; investigate the known AlignOp coverage issue separately, labeling patched results. |
| Colors/structure remain wrong | Keep the input, full logs, config and environment manifest. Check preprocessing and checkpoint identities before training anything. |

## What counts as reproducing the paper

Successful GPU execution and visually plausible examples are qualitative checks.
Paper-level quantitative reproduction additionally requires the authors' exact
benchmark dataset/protocol and no-reference metric implementations. Do not compute
PSNR/SSIM against the hazy input as though it were clean ground truth, and do not
report fabricated scores. The upstream repository supplies inference and training
code, but it does not ship a complete implementation of the paper's metric suite.

No training, evaluation-dataset download, paid GPU job, or Kaggle account action
has been performed while preparing these files.

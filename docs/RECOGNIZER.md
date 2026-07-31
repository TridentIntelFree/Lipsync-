# The recognition backend

Preprocessing is self-contained. Recognition is not: it needs a large pretrained
model, and that model comes with terms this project cannot change. That is why
it is a separate install step rather than a default dependency.

## What the model is

Auto-AVSR, the visual-only checkpoint, from *Visual Speech Recognition for
Multiple Languages* and *Auto-AVSR: Audio-Visual Speech Recognition with
Automatic Labels* (Pingchuan Ma et al., Imperial College London).

| | |
| --- | --- |
| Front end | 3D convolution + ResNet-18 over 88×88 grayscale mouth crops |
| Encoder | Conformer |
| Decoder | Transformer, joint CTC/attention beam search |
| Language model | Subword RNN LM, 5000-token unigram vocabulary |
| Training data | LRS3 (+ LRS2, VoxCeleb2 for the larger variants) |
| Reported | 19.1% WER on LRS3 |
| Size | 1002 MB model + 215 MB language model (measured) |

Decoding parameters — beam size 40, CTC weight 0.1, LM weight 0.6 — match the
reference configuration and live in `recognize.py`.

## Installing

```sh
pip install -e '.[recognize]'
python -m lipsync.recognize --download
python -m lipsync.recognize --check
```

That installs PyTorch and nothing else. In particular it does
**not** install espnet: the stock package has no 3D-convolution visual front end
and cannot load these checkpoints. The modified subset that can is vendored in
`lipsync/vendor/` and is placed ahead of any installed espnet automatically, so
having espnet installed for other reasons is harmless. See `VENDOR.md`.

`--download` fetches four files into `~/.cache/lipsync` (or `$LIPSYNC_CACHE`):

| File | What |
| --- | --- |
| `LRS3_V_WER19.1/model.pth` | Visual speech recognition weights |
| `LRS3_V_WER19.1/model.json` | Its training configuration |
| `lm_en_subword/model.pth` | Language model weights |
| `lm_en_subword/model.json` | Its configuration |

Downloads are atomic — interrupted transfers cannot leave a truncated file that
later looks like a valid cache hit.

### If your network blocks the download

Some networks deny `huggingface.co` outright. The CLI exits `4` and says so
rather than failing obscurely. Fetch the four files above on a machine that can
reach them and copy them into the cache, preserving the subdirectory names:

```
~/.cache/lipsync/LRS3_V_WER19.1/model.pth
~/.cache/lipsync/LRS3_V_WER19.1/model.json
~/.cache/lipsync/lm_en_subword/model.pth
~/.cache/lipsync/lm_en_subword/model.json
```

Nothing re-downloads once those exist. Preprocessing and `--check-only` never
need them.

## Why HuggingFace mirrors rather than the original links

Upstream hosts these on Google Drive, which requires a Google session and serves
an interstitial for large files. The HuggingFace copies are the same weights over
plain HTTPS with no account, which is what "minimal signups" needs. If you would
rather use the originals, download them from the upstream model zoo and place
them at the paths above; nothing else changes.

## Licence

**Non-commercial use only.** The checkpoints inherit terms from the datasets they
were trained on (LRS2/LRS3 are licensed from the BBC for research use). The
upstream repository states its models are for non-commercial purposes and that
its code may be used for comparative or benchmarking purposes.

This is a real constraint, not boilerplate. If you need commercial use you need
different weights — trained on data you have rights to — and this project's
preprocessing will feed them unchanged, provided they expect the same 88×88
normalised mouth crops.

The MIT licence on *this* repository covers the code here, not the weights.

## Cost of running

Measured on a plain GitHub CI runner (CPU only): the model loads in about 3
seconds and a 2-second clip decodes in about 7. The one-off ~1.2 GB weight
download is the slow part, at roughly 30 minutes on that runner. A CUDA GPU is
picked up automatically when present.

Memory: roughly 2–3 GB resident with both models loaded. Loading is expensive, so
reuse one `AutoAVSRRecognizer` across videos rather than constructing per call.

## Swapping in a different model

The boundary is `pipeline.to_model_tensor`, which returns `(1, T, 88, 88)`
float32 normalised with mean 0.421 and standard deviation 0.165. Any model
expecting that layout drops in by replacing `backend.AutoAVSRRecognizer` with
something exposing the same `transcribe(tensor) -> str`.

If a replacement model was trained with different preprocessing — a different
reference face, crop size or normalisation — then `constants.py` has to change
with it. Those constants are a contract with the specific checkpoint; mismatches
lower accuracy silently rather than failing loudly.

## Newer models worth tracking

The field moves fast and the numbers below are benchmark figures on clean data,
subject to every caveat in the README:

- **VALLR** (ICCV 2025) — 18.7% WER on LRS3, predicting phonemes then
  reconstructing text with a fine-tuned LLM. Notably data-efficient: 30 hours of
  labelled video against the thousands used here.
- **Diffusion LLM VSR** — 19.5% WER on LRS3.
- **ViSPer** (`tiiuae/visper`) — multilingual audio-visual, worth a look if you
  need languages beyond English.

The phoneme-then-LLM shape of VALLR is a better fit for this project's honesty
goals than a single end-to-end decoder, because the intermediate phoneme sequence
is inspectable: you can see what the *visual* stage actually recovered before the
language model smooths it into fluent English. Worth evaluating as a second
backend.

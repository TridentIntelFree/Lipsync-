# Vendored code in `lipsync/vendor/`

## Why this exists

The released visual speech recognition checkpoints need an encoder whose input
layer is `conv3d` — a 3D convolution feeding a ResNet-18, which turns a stack of
mouth crops into a sequence of feature vectors.

**Stock espnet cannot do this.** Its Conformer encoder offers `conv2d`,
`conv2d6`, `conv2d8`, `vgg2l`, `linear` and `embed`, and the package ships no
video or ResNet front end at all. `pip install espnet` therefore produces a
package that cannot load these weights — model construction fails on an unknown
input layer.

The Auto-AVSR authors solved this with a modified espnet. That modified subset is
vendored here so this project installs with `pip install -e '.[recognize]'` and
nothing else. There is no espnet requirement, and installing espnet alongside
this package is harmless — `backend._use_vendored_espnet()` puts the vendored
copy first on `sys.path`, so the one that can actually load the checkpoints wins.

## What was taken

52 Python files (~430 KB) from
[`amanvirparhar/chaplin`](https://github.com/amanvirparhar/chaplin) at `main`,
which packages the modified espnet from Imperial College London's
[Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
and [Auto-AVSR](https://github.com/mpc001/auto_avsr).

The parts that matter:

| Path | What it is |
| --- | --- |
| `espnet/nets/pytorch_backend/backbones/conv3d_extractor.py` | The visual front end — 3D conv + ResNet-18 |
| `espnet/nets/pytorch_backend/backbones/modules/resnet.py` | Its ResNet trunk |
| `espnet/nets/pytorch_backend/transformer/encoder.py` | Conformer encoder, with the `conv3d` input layer |
| `espnet/nets/pytorch_backend/e2e_asr_transformer.py` | The `E2E` model the checkpoint deserialises into |
| `espnet/nets/batch_beam_search.py` | Joint CTC/attention beam search |

## Licences

The chaplin repository is MIT (Amanvir Parhar); its `LICENSE` is preserved at
`lipsync/vendor/LICENSE.chaplin`. The espnet files carry their original Apache-2.0
headers, crediting espnet's authors and Imperial College London (Pingchuan Ma).
Both permit redistribution with attribution. Original copyright headers are
intact in every file.

Note this covers the *code* only. The pretrained weights are separate and are
**non-commercial** — see `docs/RECOGNIZER.md`.

## Modifications

Kept deliberately small and mechanical, so re-vendoring from upstream is easy.

**1. `distutils` removed** — it left the standard library in Python 3.12, but the
vendored code imports it at module level, which would have pinned this project to
Python 3.11. Replaced with `lipsync/vendor/_compat.py`, which supplies the two
helpers actually used:

| File | Was |
| --- | --- |
| `espnet/nets/pytorch_backend/ctc.py` | `from distutils.version import LooseVersion` |
| `espnet/nets/pytorch_backend/transformer/mask.py` | `from distutils.version import LooseVersion` |
| `espnet/nets/pytorch_backend/e2e_asr_transformer.py` | `from distutils.util import strtobool` |
| `espnet/nets/pytorch_backend/e2e_asr_transformer_av.py` | `from distutils.util import strtobool` |
| `espnet/utils/cli_utils.py` | `from distutils.util import strtobool as dist_strtobool` |

**2. matplotlib made lazy** — `transformer/plot.py` imported `matplotlib.pyplot`
at module level, and the inference import chain pulls that module in. The import
moved into the four functions that plot (`_plot_and_save_attention`, `savefig`,
`log_attentions`, `log_fig`), none of which run during inference. This drops a
~40 MB dependency that existed purely for training-time attention diagrams.

**3. `__init__.py` added to every directory** — upstream relies on namespace
packages. A regular package beats a namespace package on `sys.path` regardless of
order, so without this a pip-installed espnet would silently shadow the vendored
one and inference would fail on the unknown `conv3d` layer.

No functional code was changed. The model, its weights loading, and the decoder
behave exactly as upstream.

## Re-vendoring

```sh
git clone --depth 1 https://github.com/amanvirparhar/chaplin
cp -r chaplin/espnet lipsync/vendor/
cp chaplin/LICENSE lipsync/vendor/LICENSE.chaplin
```

Then reapply the three modifications above and run `pytest tests/test_backend.py`,
which checks the imports resolve to the vendored copy and that a real pipeline
tensor flows through the visual front end with the right shape.

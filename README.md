# Lipsync

Infer speech from silent video of a speaking face, using freely available models
and no account signups.

Point it at a video of someone talking with no usable audio, and it aligns their
mouth frame by frame, checks whether the footage can support lip reading at all,
and — when it can — produces a transcript.

```sh
lipsync talk.mp4
```

**No computer? You do not need one.** Two ways to run it from a phone:

- **One permanent link** — deploy a Hugging Face Space by pasting three small
  files in a browser. See [`deploy/huggingface/DEPLOY.md`](deploy/huggingface/DEPLOY.md).
- **No setup** — open [`notebooks/lipsync_colab.ipynb`](notebooks/lipsync_colab.ipynb)
  in Colab and tap *Runtime → Run all*.

Both are covered in [docs/RUNNING_WITHOUT_A_PC.md](docs/RUNNING_WITHOUT_A_PC.md).

## Read this before you trust any output

Automated lip reading is much weaker than it looks in demos, and it fails in a
way that is unusually easy to be fooled by.

**The best available model gets about one word in five wrong on its own
benchmark.** That number (19.1% word error rate) comes from LRS3: TED talks,
shot head-on, well lit, in HD, by professional camera operators. Footage that
does not look like that does worse, often much worse.

**Many sounds are visually identical.** English has roughly 44 phonemes but only
about a dozen distinguishable mouth shapes. `p`, `b` and `m` are the same
picture. So are `f` and `v`. So are `t`, `d` and `n`. No model, and no human lip
reader, can tell "pat" from "bat" from "mat" from the lips alone — the
information is not in the video. Skilled deaf lip readers report catching roughly
30–45% of English from the mouth alone, and lean on context for the rest.

**The gaps get filled by a language model, and it never says "I don't know".**
Ambiguity is resolved by picking the most plausible English sentence. That means
output is always fluent, always grammatical, and always confident — whether the
model read the lips correctly or invented the whole thing. Fluency here is not
evidence of accuracy. It is what the system does when it has nothing.

Here is that failure, measured rather than asserted. CI feeds the model a clip
built from **a single still photograph**, panned and rotated so the alignment has
work to do. The mouth never moves. There is no speech in it at all:

```
TRANSCRIPT: "THAT'S WHAT I'M GOING TO DO"
```

Every quality check passed — 100% face detection, good lighting, sharp, frontal —
because the footage genuinely is clean. It simply contains no speech. The model
had nothing and returned a confident sentence regardless, and nothing in the
output marks it as invention.

The model has no way to report "nothing was said". It was trained only on
footage where someone is always speaking, so it has no such output — given a
motionless mouth it returns its most likely sentence instead.

**This footage is now refused.** A check measures how much the mouth moves
relative to the nose bridge above it, which does not move during speech. That
ratio is independent of camera noise, compression and head motion, since those
affect both regions equally. The still clip scores 0.88 — the mouth moves
*less* than the rest of the face — against 1.7 or more for even subtle speech:

```
Input quality: UNUSABLE
  [ok  ] face detection: face found in 100% of frames
  [ok  ] face size: 109px between the eyes
  [FAIL] mouth movement: the mouth barely moves (ratio 0.88); this footage
         contains no speech to read, and the model would return a confident
         sentence anyway
```

CI asserts both halves on every push: that the model still confabulates when
the gate is overridden, and that the gate refuses the clip when it is not.
Reproduce with `python scripts/verify_model.py`.

So: this is a tool for generating *hypotheses* about what someone might have
said. It is not a transcript in the sense that an audio transcript is, and a
result from it should never be used to claim a specific person said a specific
thing. If a decision would rest on the exact words, this tool cannot support
that decision.

The quality gate exists because of this. It refuses to transcribe footage that
cannot support a reading, rather than handing back a confident sentence built
from nothing.

## Install

```sh
pip install -e .
```

That is enough to decode video, align mouths and run quality checks. It pulls
its own ffmpeg, so there is no system dependency.

Recognition — turning mouth crops into words — needs PyTorch and about 1 GB of
weights:

```sh
pip install -e '.[recognize]'
python -m lipsync.recognize --download
python -m lipsync.recognize --check
```

There is deliberately no `espnet` in that install. Stock espnet has no
3D-convolution visual front end and **cannot load these checkpoints at all**; the
modified build that can is vendored in `lipsync/vendor/` and takes precedence
automatically. See `VENDOR.md`.

Weights come from public HuggingFace mirrors over plain HTTPS. No account, no
token, no licence click-through. They are cached in `~/.cache/lipsync`
(override with `LIPSYNC_CACHE`). If your network blocks the download, fetch the
four files listed in `docs/RECOGNIZER.md` elsewhere and drop them in that
directory.

## Use

Check whether a video is usable before spending time on it:

```sh
lipsync talk.mp4 --check-only
```

```
640x480 at 25 fps, 75 frames (3.0s)

Input quality: MARGINAL
  [ok  ] face detection: face found in 98% of frames
  [ok  ] frame rate: 25.0 fps
  [warn] face size: 38px between the eyes; below the 54px the model trained on
  [ok  ] head pose: near-frontal (yaw ratio 0.04)
  [ok  ] head roll: 3 degrees of tilt
  [warn] sharpness: mouth region is soft (variance 41); motion blur or heavy compression
  [ok  ] exposure: exposure reasonable (median level 118)
```

Transcribe:

```sh
lipsync talk.mp4                    # refuses unusable input
lipsync talk.mp4 --json             # machine-readable, includes the caveat
lipsync talk.mp4 --save-rois r.npy  # dump the aligned mouth crops
```

Exit codes: `0` success, `2` input unusable, `3` backend not installed,
`4` weights could not be downloaded.

Or use the dashboard, which works from any device:

```sh
python app.py            # http://127.0.0.1:7860
python app.py --share    # plus a public link, for use from a phone
```

Give it a file, a recording, or a link. It plays the video beside a timeline of
what it thinks was said, segment by segment; the timeline highlights and scrolls
as the video plays, and tapping a row jumps the video there.

The layout is built for checking rather than reading. The aligned mouth crops the
model actually received sit beside its output — if those are not centred on a
mouth, every row is meaningless however convincing it reads. Where a video has
captions, they are shown against each guess with a measured error rate.

From Python:

```python
import lipsync

prepared = lipsync.prepare("talk.mp4")
print(prepared.quality.summary())

if prepared.quality.verdict is not lipsync.Verdict.UNUSABLE:
    from lipsync.recognize import recognize
    result = recognize(prepared)
    print(result.text)
    print(result.caveat())
```

## What makes footage work

In rough order of how much it matters:

| Want | Why |
| --- | --- |
| Face on, within about 20° of straight | Turned lips occlude themselves; training data is near-frontal |
| 25 fps or better | Fast consonants land between frames at lower rates |
| ≥ 55 px between the eyes | Below this the mouth is upscaled, not resolved |
| Even, front-ish lighting | Shadow across the mouth reads as lip shape |
| Little motion blur | Blur destroys exactly the fast transitions that carry information |
| Mouth never covered | Hands, microphones and masks end the reading |

Video conferencing and selfie footage tends to work. Security camera footage,
wide shots and anything where the face is a small part of the frame generally
does not.

## How it works

```
video file
   |  ffmpeg: any codec/container, normalised to constant 25 fps RGB
   v
frames
   |  MediaPipe face detector: 4 keypoints per frame
   |  (right eye, left eye, nose tip, mouth centre)
   v
keypoints
   |  gaps interpolated, then temporally smoothed
   v
   |  similarity transform onto a canonical face, 96x96 mouth patch cut
   v
mouth ROIs  ---> quality gate (fps, face size, pose, sharpness, exposure)
   |
   |  88x88 centre crop, normalised
   v
   |  Auto-AVSR: 3D conv + ResNet-18 + Conformer encoder + Transformer decoder
   |  joint CTC/attention beam search against a subword language model
   v
text + caveat
```

The alignment geometry is not tunable. It reproduces exactly how the released
checkpoint was trained — same reference face, same crop size, same normalisation
constants. Deviating degrades accuracy silently rather than raising an error,
which is why those values live in `constants.py` with their provenance recorded.

| Module | Responsibility |
| --- | --- |
| `video.py` | Decode and frame-rate normalise via bundled ffmpeg |
| `detect.py` | Per-frame keypoints, gap interpolation, temporal smoothing |
| `align.py` | Similarity transform to canonical face, mouth crop |
| `quality.py` | Whether the footage can support a reading |
| `pipeline.py` | Orchestration, tensor shaping |
| `recognize.py` | Weight fetching, transcript plus caveat |
| `backend.py` | Checkpoint loading and beam search |
| `vendor/` | Modified espnet providing the visual front end (see `VENDOR.md`) |
| `app.py` | Dashboard: player, synced timeline, quality panel |
| `source.py` | Fetch a clip and its captions from a URL |
| `segments.py` | Timed segmentation and per-segment analysis |
| `score.py` | Word error rate against a known transcript |

## Status

`pytest` runs 100 tests. Verified:

- Decode and frame-rate resampling, on real generated video files.
- Alignment recovers the canonical face layout under arbitrary rotation, scale
  and translation; mouth crops confirmed correctly framed and stable on real
  video of a moving head.
- Gap interpolation, temporal smoothing, crop geometry, tensor normalisation,
  every quality threshold, and the CLI's failure paths.
- The model contract: a real pipeline tensor flows through the vendored
  3D-conv/ResNet front end and the full Conformer encoder, emitting exactly one
  encoder step per video frame.
- The web interface builds, serves over HTTP, and refuses to show a transcript
  for footage that cannot support one.

- **The full pipeline including weights**, in CI. GitHub's runners can reach
  `huggingface.co` even though the development environment cannot, so
  `.github/workflows/verify-model.yml` downloads the real checkpoints and runs
  a clip end to end on every push. Observed: 1002 MB visual model and 215 MB
  language model fetched, 50 mouth crops at 100% detection, recogniser loaded
  in 3s, decode in 7s, transcript returned.

Nothing is now unverified except accuracy itself, which cannot be measured
without labelled footage. Point the dashboard at a video that has captions and
it will measure the error rate for you.

The backend is a thin wrapper over the reference implementation rather than a
re-implementation, deliberately — a hand-written Conformer that merely loaded
without error would emit plausible wrong text, the worst failure this project
could have.

Run `python -m lipsync.recognize --check` to see where your machine stands.

## Licence and provenance

This code is MIT.

It builds on the Auto-AVSR / Visual Speech Recognition for Multiple Languages
work by Pingchuan Ma and colleagues at Imperial College London. The alignment
geometry and decoder configuration follow their Apache-2.0 reference
implementation, and `lipsync/tokens/unigram5000_units.txt` is vendored from it.

**The pretrained checkpoints are released for non-commercial use.** That
restriction comes from the datasets they were trained on and is not ours to
waive. See `docs/RECOGNIZER.md`.

## Limits worth stating plainly

English only, from the LRS3-trained checkpoints. Other languages have their own
checkpoints upstream but are not wired up here.

Single speaker. Multiple faces means the largest one is read and the rest are
ignored.

No timing information — output is text, not timestamped captions.

Recognition itself is quicker than expected: about 7 seconds on a plain CI CPU
for a 2-second clip, with the model loading in 3. The slow part is the one-off
~1.2 GB weight download, which took 32 minutes on that runner. Once cached,
runs are fast.

And, again: this produces guesses that read as certainties. Anyone using the
output should be told that, not just the person running it.

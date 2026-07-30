# Lipsync

Infer speech from silent video of a speaking face, using freely available models
and no account signups.

Point it at a video of someone talking with no usable audio, and it aligns their
mouth frame by frame, checks whether the footage can support lip reading at all,
and — when it can — produces a transcript.

```sh
lipsync talk.mp4
```

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

Recognition — turning mouth crops into words — needs the model backend and about
1 GB of weights:

```sh
pip install -e '.[recognize]'
python -m lipsync.recognize --download
```

Weights come from public HuggingFace mirrors over plain HTTPS. No account, no
token, no licence click-through. They are cached in `~/.cache/lipsync`
(override with `LIPSYNC_CACHE`).

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

Exit codes: `0` success, `2` input unusable, `3` backend not installed.

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

## Status

Preprocessing — everything from video file to model-ready tensor — is built and
tested. `pytest` runs 52 tests covering decode, frame-rate resampling,
alignment, gap filling, smoothing, crop geometry, the CLI and every quality
threshold.
The alignment is verified to recover the canonical layout under arbitrary
rotation, scale and translation, and mouth crops were confirmed correctly framed
and stable on real video.

The recognition backend is written against the reference implementation but has
**not been executed here** — the environment this was built in blocks
`huggingface.co`, so the checkpoints could not be downloaded and the
video-to-text path is unverified end to end. It is deliberately a thin wrapper
over the reference model rather than a re-implementation: a hand-written
Conformer that merely loaded without error would produce plausible wrong text,
which is the worst failure this project could have.

Run `python -m lipsync.recognize --check` to see whether weights and backend are
ready on your machine.

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

CPU inference is slow: expect minutes, not seconds, for a short clip. The beam
search dominates.

And, again: this produces guesses that read as certainties. Anyone using the
output should be told that, not just the person running it.

# Running this without a computer

You do not need a PC. You need a browser and somewhere for the code to run, and
that somewhere can be free.

The one thing that genuinely will not work is running the pipeline *on* a phone
or tablet. It needs PyTorch, ffmpeg and about 1 GB of model weights — that is a
desktop-class Python environment, and neither iOS nor Android provides one. What
a phone does perfectly well is act as the screen and the camera for something
running elsewhere.

## Recommended: Google Colab

Free, needs only a Google account, and gives you a T4 GPU that turns minutes of
inference into seconds.

Open [`notebooks/lipsync_colab.ipynb`](../notebooks/lipsync_colab.ipynb) in
Colab and tap **Runtime → Run all**. On mobile that is the **⋮** menu, then *Run
all*. The last cell prints a public `.gradio.live` link — open it and you have
the app, on your phone, with the model running on Google's hardware.

To open it in Colab directly, visit:

```
https://colab.research.google.com/github/TridentIntelFree/Lipsync-/blob/claude/lipsync-speech-inference-k649jo/notebooks/lipsync_colab.ipynb
```

Worth knowing:

- The first run takes a few minutes — installing, then a 1 GB model download.
- Colab disconnects after roughly 90 minutes of inactivity, and the link dies
  with it. Re-run the notebook for a fresh one. Nothing is lost.

### Where the model actually lives

**Not on your phone.** It is downloaded by whatever machine runs the notebook,
and your device only ever loads a web page. Nothing is stored locally and there
is nothing to clear.

Colab wipes its own disk between sessions, so by default the model re-downloads
each time you return. The notebook avoids that by mounting Google Drive and
caching there — first run downloads it, later runs load it in seconds. It costs
about 1 GB of the free 15 GB, and Colab will ask permission the first time.

Set `USE_DRIVE = False` in the notebook to skip Drive entirely. Everything still
works; startup is just slower.

The mechanism is the `LIPSYNC_CACHE` environment variable, which points the cache
anywhere you like:

```sh
LIPSYNC_CACHE=/mnt/big-disk/lipsync lipsync talk.mp4
```

Anything already at that path is used as-is and never re-downloaded, so you can
also place the four files there by hand.
- The free tier gives somewhere around 15–30 GPU hours a week, which is far more
  than this needs. If no GPU is available it still runs on CPU, just slowly.
- Set the runtime to GPU under **Runtime → Change runtime type → T4 GPU** if it
  is not already.

## Alternative: a Hugging Face Space

A Space is a permanently hosted web app — no notebook to re-run, a stable URL you
can bookmark or share. `app.py` and `requirements.txt` in this repo are ready for
it: create a Space with the **Gradio** SDK and push these files.

Two caveats, and they matter:

**The free tier changed and is now restrictive.** Creating a Gradio Space on
CPU appears to require a PRO subscription for personal accounts. Free accounts
can host up to two Gradio Spaces on **ZeroGPU**, but with roughly 3.5 minutes of
GPU time per day. That is a handful of short clips daily — fine for occasional
use, not for working through a batch.

**ZeroGPU needs code changes.** It allocates a GPU only while a decorated
function runs, so inference has to be wrapped in `@spaces.GPU` and the model kept
off CUDA until then. `app.py` does not do that yet.

I could not verify these terms against Hugging Face's own documentation —
`huggingface.co` is unreachable from the environment this was written in — and
free-tier rules change often. Check the current terms before planning around
them. Colab avoids the whole question.

## Other options

**Kaggle Notebooks** — free GPU hours, similar to Colab, needs a Kaggle account
and phone verification.

**GitHub Codespaces** — a free monthly allowance of a real Linux machine in the
browser. No GPU on the free tier, so inference is slow, but everything works.

**Any cheap cloud VM** — if this becomes something you use regularly, a small
box with a GPU costs less than a coffee per hour and the app runs the same way.

## Filming so it actually works

The pipeline is only as good as the footage, and phone selfie video is close to
ideal if you follow a few rules:

- **Face the camera straight on.** Beyond about 20° off-axis the lips start
  hiding themselves and accuracy collapses.
- **Get close.** Your head should fill a good part of the frame. The model wants
  at least ~55 pixels between the eyes; a wide shot cannot be rescued.
- **Light your face from the front.** A window in front of you is good, a window
  behind you is bad. Shadow across the mouth reads as lip shape.
- **Hold still-ish and speak clearly**, but at a normal pace. Exaggerated
  mouthing is *not* what the model was trained on and makes results worse.
- **Keep the mouth clear** — no hands, no microphone, no mask.

Tap *Analyse* with **Attempt a transcript** unticked to check footage in a couple
of seconds without waiting for the model. That path needs no weights at all.

Whatever you do, look at the filmstrip the app shows. It is the actual input to
the model. If it is not a series of well-centred mouths, the transcript is
meaningless no matter how convincing it reads.

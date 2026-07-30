# Getting one permanent link, from a phone

A Hugging Face Space gives you a URL you can bookmark. No notebook, no re-running
anything, no second link. You can set it up entirely from a phone browser.

The trick is that you do not upload this project to the Space. You paste three
small files; the Space clones the code from GitHub itself on startup.

## Steps

1. Go to **huggingface.co/new-space** (sign in — you already have an account).
2. **Name** it `lipsync`. **SDK**: choose **Gradio**.
3. **Hardware**: pick the free option offered. If the only free choice is
   **ZeroGPU**, take it — the app already handles it.
4. Create the Space. It opens on the **Files** tab.
5. You only need to touch **two files**. The `README.md` that Hugging Face
   generates from the creation form is already correct — leave it alone.

   | File | What to do | Paste from |
   | --- | --- | --- |
   | `app.py` | already exists — replace its contents | [`app.py`](app.py), about 30 lines |
   | `requirements.txt` | create it | [`requirements.txt`](requirements.txt) |

   To edit `app.py`: tap it in the Files list, then the pencil icon, select all,
   paste, **Commit**. To create `requirements.txt`: **Files → + Add file →
   Create a new file**, name it, paste, **Commit**.

   A copy of `README.md` is in this folder for reference, but you should not
   need it. Note that `requirements.txt` deliberately does not list `gradio` —
   Spaces installs that from `sdk_version` in the README, and pinning it in both
   places causes build conflicts.

6. The Space builds itself. First build takes a few minutes. When it says
   *Running*, your link is live:

   ```
   https://huggingface.co/spaces/YOUR-USERNAME/lipsync
   ```

That link is permanent. Open it any time, on any device.

## What to expect

**First transcription is slow.** The ~1 GB model downloads on first use, not at
startup — so the Space boots quickly but the first analysis takes a few minutes.
After that it is cached for as long as the Space stays warm.

**Free Spaces sleep.** After a period of inactivity the Space pauses and the next
visit waits for it to wake, which also means re-downloading the model. Adding
persistent storage (a paid option) avoids this; the app already writes its cache
to `/data` when that exists.

**ZeroGPU has a daily quota.** Free accounts get a few minutes of GPU per day,
which is a handful of short clips. Reports put it around 3.5 minutes daily, but
Hugging Face changes these terms and I could not check their documentation from
where this was written — look at what your account actually shows.

**The quality checker needs none of that.** Untick *Attempt a transcript* and
analysis runs on CPU in seconds, with no model and no quota. That path tells you
whether footage is even worth transcribing.

## Updating it

The Space clones the repository fresh on every restart, so pushing to GitHub and
then hitting **Restart this Space** (in Settings) picks up changes. You never
need to re-paste anything.

## If Gradio Spaces need a paid plan

Recent reports say new free accounts may not be able to create CPU Gradio Spaces,
leaving ZeroGPU as the only free route. If you hit that wall, the Colab notebook
at [`notebooks/lipsync_colab.ipynb`](../../notebooks/lipsync_colab.ipynb) stays
free and unlimited — it just costs you the extra step of running it.

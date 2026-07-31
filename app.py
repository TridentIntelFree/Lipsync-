"""Lipsync dashboard.

Give it a video — a file, a recording, or a link — and it plays the video beside
a timeline of what it thinks was said, segment by segment. The timeline follows
along as the video plays, and tapping a row jumps the video there.

The layout is deliberately built around checking rather than reading. The mouth
crops the model actually received sit next to its output, and where a video has
captions they are shown beside each guess with a measured error rate. A lip
reading model always returns fluent English; the only way to know whether it
read anything is to look at what went in and compare with what was really said.

Runs anywhere Python does — a laptop, Colab, or a Hugging Face Space — and is
used from any device with a browser.

    python app.py                 # local, http://127.0.0.1:7860
    python app.py --share         # plus a public link, for use from a phone
"""

from __future__ import annotations

import argparse
import html
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from lipsync.assets import DownloadError
from lipsync.quality import Verdict
from lipsync.segments import DEFAULT_SEGMENT_SECONDS

# Hugging Face's free tier hands out a GPU only for the duration of a decorated
# call. The decorator is a no-op everywhere else, so this stays a single app.
try:  # pragma: no cover - depends on the host
    import spaces

    gpu_task = spaces.GPU(duration=180)
except Exception:  # pragma: no cover

    def gpu_task(fn):
        return fn


_VERDICT_STYLE = {
    Verdict.GOOD: ("#137333", "Usable"),
    Verdict.MARGINAL: ("#a16207", "Marginal"),
    Verdict.UNUSABLE: ("#b3261e", "Not usable"),
}

_MARKER = {Verdict.GOOD: "ok", Verdict.MARGINAL: "warn", Verdict.UNUSABLE: "fail"}


# --------------------------------------------------------------------------
# Timeline: rows carry their own timing, the script below does the rest
# --------------------------------------------------------------------------

TIMELINE_JS = """
let wiredVideo = null;
let lastActive = null;

const findVideo = () => {
  const all = Array.from(document.querySelectorAll('video'));
  return all.find(v => v.src || v.currentSrc) || all[0] || null;
};

const rows = () => Array.from(element.querySelectorAll('[data-start]'));

const paint = (row, on) => {
  row.style.background = on ? 'rgba(99,102,241,.16)' : 'transparent';
  row.style.borderLeftColor = on ? '#6366f1' : 'transparent';
};

const sync = () => {
  const video = findVideo();
  if (!video) return;
  const time = video.currentTime;
  let active = null;
  for (const row of rows()) {
    const on = time >= parseFloat(row.dataset.start) && time < parseFloat(row.dataset.end);
    paint(row, on);
    if (on) active = row;
  }
  if (active && active !== lastActive) {
    active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    lastActive = active;
  }
};

const wire = () => {
  for (const row of rows()) {
    if (row.dataset.bound) continue;
    row.dataset.bound = '1';
    row.style.cursor = 'pointer';
    row.style.borderLeft = '3px solid transparent';
    row.addEventListener('click', () => {
      const video = findVideo();
      if (!video) return;
      video.currentTime = parseFloat(row.dataset.start);
      video.play().catch(() => {});
    });
  }
  const video = findVideo();
  if (video && video !== wiredVideo) {
    wiredVideo = video;
    lastActive = null;
    video.addEventListener('timeupdate', sync);
    video.addEventListener('seeked', sync);
  }
};

wire();
// The timeline is replaced on every run and Gradio may swap the video element,
// so keep re-wiring rather than assuming a stable DOM.
setInterval(wire, 700);
"""


def _quality_html(report) -> str:
    colour, label = _VERDICT_STYLE[report.verdict]
    rows = []
    for check in report.checks:
        tint = _VERDICT_STYLE[check.verdict][0]
        rows.append(
            f'<tr><td style="padding:3px 10px 3px 0;color:{tint};font-weight:600">'
            f"{_MARKER[check.verdict]}</td>"
            f'<td style="padding:3px 10px 3px 0;white-space:nowrap">{check.name}</td>'
            f'<td style="padding:3px 0;opacity:.8">{html.escape(check.message)}</td></tr>'
        )
    return (
        f'<div style="border-left:4px solid {colour};padding:6px 0 6px 12px">'
        f'<div style="font-weight:700;color:{colour}">{label}</div>'
        f'<table style="border-collapse:collapse;margin-top:6px;font-size:.86em">'
        f'{"".join(rows)}</table></div>'
    )


def _timeline_html(analysis, has_reference: bool) -> str:
    """The dashboard's centre: one row per segment, timed for the player."""
    if not analysis.segments:
        return (
            '<p style="opacity:.7">No segments long enough to analyse. Try a '
            "longer clip or a shorter segment length.</p>"
        )

    rows = []
    for seg in analysis.segments:
        guess = html.escape(seg.transcript) if seg.transcript else "—"
        block = (
            f'<div style="font-size:.95em;line-height:1.45">{guess}</div>'
        )
        if has_reference and seg.reference:
            wer = f"{seg.score.wer:.0%}" if seg.score else "—"
            block += (
                f'<div style="font-size:.85em;opacity:.65;margin-top:3px">'
                f"actually said: {html.escape(seg.reference)}</div>"
                f'<div style="font-size:.78em;opacity:.55;margin-top:2px">'
                f"{wer} word error rate</div>"
            )
        rows.append(
            f'<div data-start="{seg.start:.2f}" data-end="{seg.end:.2f}" '
            f'style="padding:8px 10px;border-bottom:1px solid rgba(128,128,128,.18);'
            f'transition:background .15s">'
            f'<div style="font-size:.75em;opacity:.55;letter-spacing:.03em">'
            f"{seg.timestamp}</div>{block}</div>"
        )

    return (
        '<div style="max-height:420px;overflow-y:auto;border:1px solid '
        'rgba(128,128,128,.25);border-radius:8px">' + "".join(rows) + "</div>"
    )


def _roi_filmstrip(rois: np.ndarray, count: int = 10) -> np.ndarray | None:
    """Evenly spaced mouth crops — what the model actually received."""
    if rois is None or len(rois) == 0:
        return None
    picks = np.linspace(0, len(rois) - 1, min(count, len(rois))).astype(int)
    return np.hstack([rois[i] for i in picks])


def _step(progress, fraction: float, message: str) -> None:
    """Report a stage, tolerating being called outside a Gradio event."""
    if progress is None:
        return
    try:
        progress(fraction, desc=message)
    except Exception:  # noqa: BLE001 - progress must never break the analysis
        pass


def _blank(message: str):
    return None, message, "", None, ""


# --------------------------------------------------------------------------


@gpu_task
def run(
    video_path: str | None,
    url: str,
    start: float,
    duration: float,
    segment_seconds: float,
    transcribe: bool,
    progress=None,
):
    """Analyse a video from a file or a URL and build the dashboard."""
    from lipsync.segments import analyse as analyse_segments

    url = (url or "").strip()
    if not url and not video_path:
        return _blank("Upload a video or paste a link to begin.")

    captions = None
    caption_source = None
    title = None

    if url:
        from lipsync.source import SourceError, fetch_clip

        workdir = Path(tempfile.mkdtemp(prefix="lipsync-url-"))
        try:
            _step(progress, 0.05, "Fetching the clip")
            clip = fetch_clip(url, workdir, start=start, duration=duration)
        except SourceError as exc:
            return _blank(f"**Could not fetch that link.**\n\n```\n{exc}\n```")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return _blank(f"**Could not fetch that link.**\n\n```\n{type(exc).__name__}: {exc}\n```")
        video_path = str(clip.path)
        captions = clip.captions or None
        caption_source = clip.caption_source
        title = clip.title

    try:
        _step(progress, 0.2, "Decoding video and finding the face")
        analysis, prepared = analyse_segments(
            video_path,
            segment_seconds=segment_seconds,
            captions=captions,
            transcribe=transcribe,
            max_seconds=None if url else (duration or None),
            progress=lambda f: _step(progress, 0.35 + 0.6 * f, "Reading lips"),
        )
    except DownloadError as exc:
        return video_path, f"**Model weights unavailable.**\n\n```\n{exc}\n```", "", None, ""
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return video_path, f"**Analysis failed.**\n\n```\n{type(exc).__name__}: {exc}\n```", "", None, ""

    lines = []
    if title:
        lines.append(f"### {html.escape(title)}")

    if prepared.quality.verdict is Verdict.UNUSABLE:
        lines.append(
            "**This footage cannot support a reading.** Anything below would be "
            "invention rather than a guess — see the checks."
        )
    elif captions and analysis.overall:
        lines.append(
            f"**Measured against this video's {caption_source}: "
            f"{analysis.overall.summary()}**"
        )
        lines.append(
            "That is the real number for this footage, not a benchmark figure. "
            "Every row can be checked against the text beside it."
        )
    elif captions:
        lines.append(f"Captions found ({caption_source}), but nothing to compare yet.")
    elif transcribe:
        lines.append(
            "**No captions to check this against.** The model returns fluent "
            "English whether or not it read anything, so treat every row as "
            "unverified guessing."
        )
    else:
        lines.append(
            "Quality checked only. Tick **Attempt transcripts** to read the lips."
        )

    lines.append(
        f"\n<sub>{prepared.info.width}x{prepared.info.height} at "
        f"{prepared.info.fps:g} fps · {len(prepared.mouth_rois)} frames · "
        f"{prepared.duration_seconds:.1f}s · {len(analysis.segments)} segments</sub>"
    )

    return (
        video_path,
        "\n\n".join(lines),
        _quality_html(prepared.quality),
        _roi_filmstrip(prepared.mouth_rois),
        _timeline_html(analysis, bool(captions)),
    )


INTRO = """
# Lipsync

Play a video and follow what the model thinks was said, segment by segment.

**Output is a guess that reads as a certainty.** Roughly one word in five is
wrong on clean, head-on, well-lit video and worse on anything else. Many sounds
are visually identical — `p`, `b` and `m` are the same picture — so the gaps are
filled by a language model that always returns fluent English. Never treat a row
below as evidence of what a particular person said.
"""

HELP = """
**Tap any row** to jump the video there. Rows highlight as it plays.

**Best footage:** face-on, head filling much of the frame, even front lighting,
mouth unobstructed, normal speaking pace. Selfie video works; wide shots do not.

**Check the filmstrip.** Those are the actual mouth crops fed to the model. If
they are not centred on a mouth, every row is meaningless.

Links are fetched with yt-dlp, which most video sites prohibit. If the video has
captions they are used as an answer key, which is the only way to see how wrong
this really is on footage you care about.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Lipsync") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=5):
                player = gr.Video(
                    label="Video — upload, record, or load from a link",
                    sources=["upload", "webcam"],
                )
                url = gr.Textbox(
                    label="…or paste a video link",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                with gr.Row():
                    start = gr.Number(value=0, label="Start (s)", scale=1)
                    duration = gr.Number(value=30, label="Length (s)", scale=1)
                    segment = gr.Slider(
                        2, 15, value=DEFAULT_SEGMENT_SECONDS, step=1,
                        label="Segment (s)", scale=2,
                    )
                transcribe = gr.Checkbox(
                    value=True,
                    label="Attempt transcripts",
                    info="Untick for a fast quality check with no model and no waiting",
                )
                go = gr.Button("Analyse", variant="primary")
                summary = gr.Markdown()

            with gr.Column(scale=4):
                timeline = gr.HTML(js_on_load=TIMELINE_JS)
                strip = gr.Image(
                    label="What the model sees (aligned mouth crops)", height=100
                )
                quality = gr.HTML()
                gr.Markdown(HELP)

        go.click(
            lambda v, u, s, d, seg, t, progress=gr.Progress(): run(
                v, u, s, d, seg, t, progress
            ),
            inputs=[player, url, start, duration, segment, transcribe],
            outputs=[player, summary, quality, strip, timeline],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the lipsync dashboard.")
    parser.add_argument(
        "--share", action="store_true",
        help="create a public link (use this to reach it from a phone)",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    build().launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

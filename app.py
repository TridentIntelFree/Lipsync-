"""Browser interface for lipsync.

Runs anywhere Python does — a laptop, a Colab notebook, or a Hugging Face Space —
and is used from any device with a browser. That split matters: the pipeline
needs PyTorch, ffmpeg and a gigabyte of weights, which no phone can provide, but
any phone can open a web page pointed at a machine that does.

Launch:

    python app.py                 # local, http://127.0.0.1:7860
    python app.py --share         # plus a public link, for use from a phone

See docs/RUNNING_WITHOUT_A_PC.md.
"""

from __future__ import annotations

import argparse
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from lipsync.assets import DownloadError
from lipsync.pipeline import prepare
from lipsync.quality import Verdict
from lipsync.segments import DEFAULT_SEGMENT_SECONDS

# Hugging Face's free tier hands out a GPU only for the duration of a decorated
# call. The decorator is a no-op everywhere else, so this stays a single app.
try:  # pragma: no cover - depends on the host
    import spaces

    gpu_task = spaces.GPU(duration=120)
except Exception:  # pragma: no cover

    def gpu_task(fn):
        return fn


_VERDICT_STYLE = {
    Verdict.GOOD: ("#137333", "Usable"),
    Verdict.MARGINAL: ("#a16207", "Marginal"),
    Verdict.UNUSABLE: ("#b3261e", "Not usable"),
}

_MARKER = {Verdict.GOOD: "ok", Verdict.MARGINAL: "warn", Verdict.UNUSABLE: "fail"}


def _quality_html(report) -> str:
    colour, label = _VERDICT_STYLE[report.verdict]
    rows = []
    for check in report.checks:
        tint = _VERDICT_STYLE[check.verdict][0]
        rows.append(
            f'<tr><td style="padding:4px 12px 4px 0;color:{tint};'
            f'font-weight:600;white-space:nowrap">{_MARKER[check.verdict]}</td>'
            f'<td style="padding:4px 12px 4px 0;white-space:nowrap">{check.name}</td>'
            f'<td style="padding:4px 0;opacity:.85">{check.message}</td></tr>'
        )
    return (
        f'<div style="border-left:4px solid {colour};padding:8px 0 8px 14px">'
        f'<div style="font-size:1.05em;font-weight:700;color:{colour}">{label}</div>'
        f'<table style="border-collapse:collapse;margin-top:8px;font-size:.92em">'
        f'{"".join(rows)}</table></div>'
    )


def _roi_filmstrip(rois: np.ndarray, count: int = 10) -> np.ndarray | None:
    """A strip of evenly spaced mouth crops, so you can see what the model saw.

    The single most useful output here: if this is not centred on a mouth,
    nothing downstream can work, and no amount of transcript plausibility should
    convince you otherwise.
    """
    if rois is None or len(rois) == 0:
        return None
    picks = np.linspace(0, len(rois) - 1, min(count, len(rois))).astype(int)
    return np.hstack([rois[i] for i in picks])


def _header(prepared) -> str:
    return (
        f'<div style="opacity:.7;margin-bottom:10px">'
        f"{prepared.info.width}x{prepared.info.height} at {prepared.info.fps:g} fps "
        f"&middot; {len(prepared.mouth_rois)} frames analysed "
        f"({prepared.duration_seconds:.1f}s)</div>"
    )


def _error(message: str) -> str:
    return f'<div style="color:#b3261e"><b>{message}</b></div>'


# --------------------------------------------------------------------------
# Tab 1: a video you upload or record
# --------------------------------------------------------------------------


@gpu_task
def analyse(video_path: str | None, transcribe: bool, max_seconds: float):
    """Preprocess a video and, optionally, transcribe it."""
    if not video_path:
        return "Upload or record a video to begin.", None, ""

    try:
        prepared = prepare(video_path, max_seconds=max_seconds or None)
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        traceback.print_exc()
        return _error(f"Could not read that video.<br>{type(exc).__name__}: {exc}"), None, ""

    quality = _header(prepared) + _quality_html(prepared.quality)
    strip = _roi_filmstrip(prepared.mouth_rois)

    if not transcribe:
        return quality, strip, ""

    if prepared.quality.verdict is Verdict.UNUSABLE:
        return (
            quality,
            strip,
            "**No transcript.** This footage cannot support a reading, and a "
            "transcript produced from it would be invention rather than a guess. "
            "Fix what is flagged above and try again.",
        )

    try:
        from lipsync.recognize import BackendMissing, recognize

        result = recognize(prepared)
    except BackendMissing as exc:
        return quality, strip, f"**Recognition backend unavailable.**\n\n```\n{exc}\n```"
    except DownloadError as exc:
        return quality, strip, f"**Model weights unavailable.**\n\n```\n{exc}\n```"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return quality, strip, f"**Recognition failed.**\n\n```\n{type(exc).__name__}: {exc}\n```"

    text = result.text.strip() or "_(nothing recognised)_"
    return quality, strip, f"### Best guess\n\n> {text}\n\n---\n\n**{result.caveat()}**"


# --------------------------------------------------------------------------
# Tab 2: a video at a URL, analysed in timed segments
# --------------------------------------------------------------------------


def _segments_table(analysis) -> str:
    if not analysis.segments:
        return "<p>No segments long enough to analyse.</p>"

    show_reference = analysis.reference_available
    head = "<tr><th>Time</th><th>Model's guess</th>"
    head += "<th>Actually said</th><th>Error</th></tr>" if show_reference else "</tr>"

    rows = []
    for seg in analysis.segments:
        cells = (
            f'<td style="white-space:nowrap;opacity:.7;padding:6px 12px 6px 0">'
            f"{seg.timestamp}</td>"
            f'<td style="padding:6px 12px 6px 0">{seg.transcript or "—"}</td>'
        )
        if show_reference:
            wer = f"{seg.score.wer:.0%}" if seg.score else "—"
            cells += (
                f'<td style="padding:6px 12px 6px 0;opacity:.85">'
                f'{seg.reference or "—"}</td>'
                f'<td style="padding:6px 0;white-space:nowrap">{wer}</td>'
            )
        rows.append(f"<tr>{cells}</tr>")

    return (
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;'
        'font-size:.92em;width:100%">'
        f'<thead style="text-align:left;opacity:.6">{head}</thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


@gpu_task
def analyse_url(
    url: str,
    start: float,
    duration: float,
    segment_seconds: float,
    transcribe: bool,
    progress=gr.Progress(),
):
    """Fetch a clip from a URL and analyse it segment by segment."""
    if not url or not url.strip():
        return None, "Paste a video URL to begin.", None, "", ""

    from lipsync.segments import analyse as analyse_segments
    from lipsync.source import SourceError, fetch_clip

    workdir = Path(tempfile.mkdtemp(prefix="lipsync-url-"))
    try:
        progress(0.05, desc="Fetching the clip")
        clip = fetch_clip(url.strip(), workdir, start=start, duration=duration)
    except SourceError as exc:
        return None, _error(f"Could not fetch that URL.<br>{exc}"), None, "", ""
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return None, _error(f"Could not fetch that URL.<br>{type(exc).__name__}: {exc}"), None, "", ""

    try:
        progress(0.25, desc="Aligning mouths")
        analysis, prepared = analyse_segments(
            str(clip.path),
            segment_seconds=segment_seconds,
            captions=clip.captions,
            transcribe=transcribe,
            progress=lambda f: progress(0.25 + 0.7 * f, desc="Reading lips"),
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return str(clip.path), _error(f"{type(exc).__name__}: {exc}"), None, "", ""

    quality = _header(prepared) + _quality_html(prepared.quality)
    strip = _roi_filmstrip(prepared.mouth_rois)

    verdict = f"### {clip.title}\n\n"
    if clip.has_captions:
        if analysis.overall:
            verdict += (
                f"**Measured against this video's {clip.caption_source}: "
                f"{analysis.overall.summary()}**\n\n"
                f"That is the honest number for this footage — not a benchmark "
                f"figure, not an estimate. Every row below can be checked.\n\n"
            )
        else:
            verdict += f"Captions found ({clip.caption_source}), but nothing to compare yet.\n\n"
    else:
        verdict += (
            "**No captions on this video, so there is nothing to check the output "
            "against.** Treat everything below as unverifiable guessing.\n\n"
        )

    return str(clip.path), quality, strip, _segments_table(analysis), verdict


# --------------------------------------------------------------------------


INTRO = """
# Lipsync

Infer what someone said from silent video of them speaking.

**This produces guesses that read as certainties.** The model gets roughly one
word in five wrong on clean, head-on, well-lit video, and considerably worse on
anything else. Many sounds look identical on the lips — `p`, `b` and `m` are the
same picture, as are `f` and `v` — so the gaps are filled by a language model
that always returns fluent English whether or not it read anything.

Never treat the output as evidence of what a particular person actually said.
"""

TIPS = """
**For the best chance:** face the camera straight on, fill a good part of the
frame with your head, use even front lighting, keep the mouth unobstructed, and
speak at a normal pace — exaggerated mouthing is *not* what the model learned
from and makes results worse.

The filmstrip shows the aligned mouth crops the model actually receives. If it
is not centred on a mouth, ignore any transcript.
"""

URL_NOTE = """
Paste a link to a video and it will be fetched, split into short segments, and
read segment by segment so you can follow along while it plays.

**If the video has captions, they are used as an answer key** — you get the
model's guess beside what was actually said, and a real error rate for that
footage. This is by far the most useful thing you can do with this tool: it
replaces an unfalsifiable sentence with a measured number.

Downloading from most video sites is against their terms of service, and
generating quotes for identifiable people from footage this tool cannot reliably
read is how fabricated quotes get made. Both are your call.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Lipsync") as demo:
        gr.Markdown(INTRO)

        with gr.Tabs():
            with gr.Tab("Upload or record"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video = gr.Video(
                            label="Video of someone speaking",
                            sources=["upload", "webcam"],
                        )
                        transcribe = gr.Checkbox(
                            value=True,
                            label="Attempt a transcript",
                            info="Uncheck to only check whether the footage is usable (fast, no model needed)",
                        )
                        max_seconds = gr.Slider(
                            0, 60, value=15, step=1,
                            label="Seconds to analyse",
                            info="0 means the whole clip",
                        )
                        run = gr.Button("Analyse", variant="primary")
                        gr.Markdown(TIPS)

                    with gr.Column(scale=1):
                        quality = gr.HTML()
                        strip = gr.Image(
                            label="What the model sees (aligned mouth crops)",
                            height=110,
                        )
                        transcript = gr.Markdown()

                run.click(
                    analyse,
                    inputs=[video, transcribe, max_seconds],
                    outputs=[quality, strip, transcript],
                )

            with gr.Tab("From a URL"):
                gr.Markdown(URL_NOTE)
                with gr.Row():
                    with gr.Column(scale=1):
                        url = gr.Textbox(
                            label="Video URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                        )
                        with gr.Row():
                            url_start = gr.Number(value=0, label="Start (seconds)")
                            url_duration = gr.Number(value=30, label="Length (seconds)")
                        url_segment = gr.Slider(
                            2, 15, value=DEFAULT_SEGMENT_SECONDS, step=1,
                            label="Segment length (seconds)",
                        )
                        url_transcribe = gr.Checkbox(value=True, label="Attempt transcripts")
                        url_run = gr.Button("Fetch and analyse", variant="primary")
                        url_player = gr.Video(label="The clip", interactive=False)

                    with gr.Column(scale=1):
                        url_verdict = gr.Markdown()
                        url_quality = gr.HTML()
                        url_strip = gr.Image(
                            label="What the model sees", height=110
                        )
                        url_table = gr.HTML()

                url_run.click(
                    analyse_url,
                    inputs=[url, url_start, url_duration, url_segment, url_transcribe],
                    outputs=[url_player, url_quality, url_strip, url_table, url_verdict],
                )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the lipsync web interface.")
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

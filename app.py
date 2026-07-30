"""Browser interface for lipsync.

Runs anywhere Python does — a Colab notebook, a Hugging Face Space, or a laptop —
and is used from any device with a browser. That matters: the pipeline needs
PyTorch and ffmpeg, which phones and tablets cannot provide, but they can open a
web page pointed at a machine that does.

Launch:

    python app.py                 # local, http://127.0.0.1:7860
    python app.py --share         # plus a public link, for use from a phone

See docs/RUNNING_WITHOUT_A_PC.md.
"""

from __future__ import annotations

import argparse
import traceback

import gradio as gr
import numpy as np

from lipsync.assets import DownloadError
from lipsync.pipeline import prepare
from lipsync.quality import Verdict

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

    This is the single most useful debugging output: if the strip is not centred
    on a mouth, nothing downstream can work, and no amount of transcript
    plausibility should convince you otherwise.
    """
    if rois is None or len(rois) == 0:
        return None
    picks = np.linspace(0, len(rois) - 1, min(count, len(rois))).astype(int)
    return np.hstack([rois[i] for i in picks])


def analyse(video_path: str | None, transcribe: bool, max_seconds: float):
    """Preprocess a video and, optionally, transcribe it."""
    if not video_path:
        return "Upload or record a video to begin.", None, ""

    try:
        prepared = prepare(video_path, max_seconds=max_seconds or None)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        traceback.print_exc()
        return (
            f'<div style="color:#b3261e"><b>Could not read that video.</b><br>'
            f"{type(exc).__name__}: {exc}</div>",
            None,
            "",
        )

    quality = _quality_html(prepared.quality)
    header = (
        f"{prepared.info.width}x{prepared.info.height} at {prepared.info.fps:g} fps &middot; "
        f"{len(prepared.mouth_rois)} frames analysed "
        f"({prepared.duration_seconds:.1f}s)"
    )
    quality = f'<div style="opacity:.7;margin-bottom:10px">{header}</div>{quality}'
    strip = _roi_filmstrip(prepared.mouth_rois)

    if not transcribe:
        return quality, strip, ""

    if prepared.quality.verdict is Verdict.UNUSABLE:
        return (
            quality,
            strip,
            "**No transcript.** This footage cannot support a reading, and a "
            "transcript produced from it would be invention rather than a "
            "guess. Fix what is flagged above and try again.",
        )

    try:
        from lipsync.recognize import BackendMissing, recognize

        result = recognize(prepared)
    except BackendMissing as exc:
        return quality, strip, f"**Recognition backend unavailable.**\n\n```\n{exc}\n```"
    except DownloadError as exc:
        return (
            quality,
            strip,
            "**Model weights unavailable.**\n\n"
            f"```\n{exc}\n```\n\n"
            "Quality checking above works without them.",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return quality, strip, f"**Recognition failed.**\n\n```\n{type(exc).__name__}: {exc}\n```"

    text = result.text.strip() or "_(nothing recognised)_"
    return (
        quality,
        strip,
        f"### Best guess\n\n> {text}\n\n---\n\n**{result.caveat()}**",
    )


INTRO = """
# Lipsync

Infer what someone said from silent video of them speaking.

**This produces guesses that read as certainties.** The underlying model gets
roughly one word in five wrong on clean, head-on, well-lit video, and worse on
anything else. Many sounds look identical on the lips — `p`, `b` and `m` are the
same picture, as are `f` and `v` — so the gaps are filled by a language model
that always returns fluent English whether or not it read anything. Never treat
the output as evidence of what a particular person actually said.
"""

TIPS = """
**For the best chance:** face the camera straight on, fill a good part of the
frame with your head, use even front lighting, and keep the mouth unobstructed.
Phone selfie video at arm's length works well. Wide shots and side angles do not.

The filmstrip shows the aligned mouth crops the model actually receives. If it
is not centred on a mouth, ignore any transcript.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Lipsync") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(
                    label="Video of someone speaking", sources=["upload", "webcam"]
                )
                transcribe = gr.Checkbox(
                    value=True,
                    label="Attempt a transcript",
                    info="Uncheck to only check whether the footage is usable (fast, no model needed)",
                )
                max_seconds = gr.Slider(
                    0, 60, value=15, step=1,
                    label="Seconds to analyse",
                    info="0 means the whole clip. Recognition is slow on CPU.",
                )
                run = gr.Button("Analyse", variant="primary")
                gr.Markdown(TIPS)

            with gr.Column(scale=1):
                quality = gr.HTML(label="Input quality")
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

"""Break a video into timed segments and analyse each one.

The recogniser was trained on single utterances of a few seconds, and its beam
search cost grows with length, so a long clip is handled as a sequence of short
ones rather than in a single pass. The result is timestamped, which is what
makes it possible to follow along while the video plays.
"""

from __future__ import annotations

import dataclasses

from .constants import MODEL_FPS
from .pipeline import PreparedVideo, prepare
from .quality import Verdict
from .score import Score, score

DEFAULT_SEGMENT_SECONDS = 6.0


@dataclasses.dataclass
class Segment:
    """One analysed slice of a video."""

    index: int
    start: float
    end: float
    transcript: str
    quality: Verdict
    reference: str | None = None
    score: Score | None = None

    @property
    def timestamp(self) -> str:
        def clock(seconds: float) -> str:
            minutes, secs = divmod(int(seconds), 60)
            return f"{minutes:d}:{secs:02d}"

        return f"{clock(self.start)}–{clock(self.end)}"


@dataclasses.dataclass
class Analysis:
    """Every segment of a video, plus an overall score where one is possible."""

    segments: list[Segment]
    overall: Score | None = None
    reference_available: bool = False

    @property
    def transcript(self) -> str:
        return " ".join(s.transcript for s in self.segments if s.transcript).strip()


def _captions_between(captions, start: float, end: float) -> str:
    """Reference text overlapping a time window."""
    return " ".join(
        c.text for c in captions if c.end > start and c.start < end
    ).strip()


def analyse(
    video_path: str,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    captions=None,
    max_seconds: float | None = None,
    transcribe: bool = True,
    progress=None,
) -> tuple[Analysis, PreparedVideo]:
    """Analyse a video segment by segment.

    The video is decoded and aligned once, then sliced — decoding per segment
    would repeat the expensive work and risk inconsistent alignment at the
    boundaries.
    """
    prepared = prepare(video_path, max_seconds=max_seconds)

    total_frames = len(prepared.mouth_rois)
    if total_frames == 0:
        return Analysis([], None, bool(captions)), prepared

    stride = max(1, int(round(segment_seconds * MODEL_FPS)))
    recogniser = None

    if transcribe and prepared.quality.verdict is not Verdict.UNUSABLE:
        from .recognize import _require_backend, download_weights
        from .backend import AutoAVSRRecognizer

        _require_backend()
        download_weights(progress=False)
        # Built once: loading ~1 GB of weights per segment would dominate.
        recogniser = AutoAVSRRecognizer()

    segments: list[Segment] = []
    for index, begin in enumerate(range(0, total_frames, stride)):
        chunk = prepared.mouth_rois[begin : begin + stride]
        # A fragment too short to contain speech is noise to the model.
        if len(chunk) < MODEL_FPS:
            continue

        start = begin / MODEL_FPS
        end = (begin + len(chunk)) / MODEL_FPS

        text = ""
        if recogniser is not None:
            from .recognize import to_backend_tensor

            try:
                text = recogniser.transcribe(to_backend_tensor(chunk))
            except Exception as exc:  # noqa: BLE001 - one bad segment must not end the run
                text = f"[failed: {type(exc).__name__}]"

        reference = _captions_between(captions, start, end) if captions else None
        segments.append(
            Segment(
                index=index,
                start=start,
                end=end,
                transcript=text,
                quality=prepared.quality.verdict,
                reference=reference or None,
                score=score(reference, text) if reference and text else None,
            )
        )
        if progress is not None:
            progress((begin + len(chunk)) / total_frames)

    overall = None
    if captions:
        reference_all = " ".join(c.text for c in captions)
        hypothesis = " ".join(s.transcript for s in segments if s.transcript)
        if reference_all.strip() and hypothesis.strip():
            overall = score(reference_all, hypothesis)

    return Analysis(segments, overall, bool(captions)), prepared

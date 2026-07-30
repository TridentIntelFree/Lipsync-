"""The browser interface, especially what it says when something is wrong.

The app is how most people will meet this project, so its honesty behaviour —
refusing to show a transcript for unusable footage, always showing what the
model actually saw — is worth testing rather than assuming.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio", reason="gradio not installed")

import app as web_app  # noqa: E402

from lipsync.quality import Verdict  # noqa: E402
from tests.test_video import make_video  # noqa: E402


@pytest.fixture(scope="module")
def faceless_video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("app") / "noface.mp4", seconds=1)


def test_no_input_asks_for_a_video():
    quality, strip, transcript = web_app.analyse(None, True, 15)
    assert "Upload" in quality
    assert strip is None
    assert transcript == ""


def test_unreadable_file_reports_instead_of_raising(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    quality, strip, transcript = web_app.analyse(str(junk), True, 15)
    assert "Could not read that video" in quality
    assert strip is None


def test_faceless_video_refuses_to_transcribe(faceless_video):
    quality, _, transcript = web_app.analyse(str(faceless_video), True, 15)
    assert "Not usable" in quality
    assert "No transcript" in transcript
    assert "invention" in transcript


def test_quality_only_mode_returns_no_transcript(faceless_video):
    _, _, transcript = web_app.analyse(str(faceless_video), False, 15)
    assert transcript == ""


def test_filmstrip_is_a_horizontal_strip_of_crops():
    rois = np.random.randint(0, 255, (30, 96, 96), dtype=np.uint8)
    strip = web_app._roi_filmstrip(rois, count=6)
    assert strip.shape == (96, 96 * 6)


def test_filmstrip_handles_short_and_empty_sequences():
    assert web_app._roi_filmstrip(np.empty((0, 96, 96), np.uint8)) is None
    assert web_app._roi_filmstrip(None) is None
    short = web_app._roi_filmstrip(np.zeros((2, 96, 96), np.uint8), count=10)
    assert short.shape == (96, 192)


def test_every_verdict_renders():
    """A missing style entry would blow up exactly when a video fails."""
    from lipsync.quality import Check, QualityReport

    for verdict in Verdict:
        report = QualityReport([Check("face size", verdict, 1.0, "a message")])
        html = web_app._quality_html(report)
        assert "a message" in html
        assert web_app._MARKER[verdict] in html


def test_interface_builds():
    assert web_app.build() is not None

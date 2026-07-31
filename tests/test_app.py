"""The dashboard, especially what it says when something is wrong.

Most people meet this project through the web interface, so its honesty
behaviour — refusing to imply a reading it cannot support, always showing what
the model actually saw — is tested rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio", reason="gradio not installed")

import app as web_app  # noqa: E402

from lipsync.quality import Check, QualityReport, Verdict  # noqa: E402
from lipsync.score import score  # noqa: E402
from lipsync.segments import Analysis, Segment  # noqa: E402
from tests.test_video import make_video  # noqa: E402


@pytest.fixture(scope="module")
def faceless_video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("app") / "noface.mp4", seconds=2)


def segment(index=0, start=0.0, end=3.0, text="hello there", reference=None):
    return Segment(
        index=index,
        start=start,
        end=end,
        transcript=text,
        quality=Verdict.GOOD,
        reference=reference,
        score=score(reference, text) if reference else None,
    )


# --- inputs -------------------------------------------------------------


def test_no_input_asks_for_one():
    player, summary, quality, strip, timeline = web_app.run(None, "", 0, 30, 6, True)
    assert player is None
    assert "Add a video to begin" in summary
    assert strip is None


def test_an_attachment_is_accepted_as_a_video_source(faceless_video):
    """The video widget fails on some mobile browsers; the attachment must work."""
    player, _, _, _, _ = web_app.run(None, "", 0, 30, 6, False, str(faceless_video))
    assert player == str(faceless_video)


def test_attachment_objects_are_unwrapped(faceless_video):
    class Uploaded:
        name = str(faceless_video)

    player, _, _, _, _ = web_app.run(None, "", 0, 30, 6, False, Uploaded())
    assert player == str(faceless_video)


def test_attachment_wins_over_the_player(faceless_video):
    """Using the fallback is a deliberate act; honour it over a stale player value."""
    player, _, _, _, _ = web_app.run(
        "/does/not/exist.mp4", "", 0, 30, 6, False, str(faceless_video)
    )
    assert player == str(faceless_video)


def test_unreadable_file_reports_instead_of_raising(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    _, summary, _, _, _ = web_app.run(str(junk), "", 0, 30, 6, False)
    assert "Analysis failed" in summary


def test_bad_link_is_reported_cleanly():
    _, summary, _, _, _ = web_app.run(None, "not-a-url", 0, 10, 5, False)
    assert "Could not fetch that link" in summary


def test_faceless_video_is_called_unusable(faceless_video):
    player, summary, quality, _, _ = web_app.run(str(faceless_video), "", 0, 30, 6, True)
    assert player == str(faceless_video)
    assert "cannot support a reading" in summary
    assert "Not usable" in quality


def test_quality_only_mode_does_not_imply_a_reading(faceless_video):
    _, summary, _, _, _ = web_app.run(str(faceless_video), "", 0, 30, 6, False)
    assert "Quality checked only" in summary or "cannot support a reading" in summary


# --- timeline -----------------------------------------------------------


def test_timeline_rows_carry_timings_for_the_player():
    analysis = Analysis([segment(0, 0.0, 3.0), segment(1, 3.0, 6.0)])
    markup = web_app._timeline_html(analysis, has_reference=False)
    assert 'data-start="0.00"' in markup
    assert 'data-end="3.00"' in markup
    assert 'data-start="3.00"' in markup


def test_timeline_shows_reference_and_error_rate_when_available():
    analysis = Analysis(
        [segment(0, 0.0, 3.0, "the bat sat", reference="the cat sat")],
        reference_available=True,
    )
    markup = web_app._timeline_html(analysis, has_reference=True)
    assert "actually said: the cat sat" in markup
    assert "33% word error rate" in markup


def test_timeline_omits_reference_column_when_there_is_none():
    markup = web_app._timeline_html(Analysis([segment()]), has_reference=False)
    assert "actually said" not in markup


def test_timeline_escapes_transcript_text():
    """Transcripts are model output; they must not be able to inject markup."""
    analysis = Analysis([segment(text="<script>alert(1)</script>")])
    markup = web_app._timeline_html(analysis, has_reference=False)
    assert "<script>alert(1)</script>" not in markup
    assert "&lt;script&gt;" in markup


def test_empty_timeline_explains_itself():
    markup = web_app._timeline_html(Analysis([]), has_reference=False)
    assert "No segments long enough" in markup


def test_timeline_javascript_references_the_row_timings():
    """The script and the markup have to agree on the data attributes."""
    assert "data-start" in web_app.TIMELINE_JS
    assert "timeupdate" in web_app.TIMELINE_JS
    assert "currentTime" in web_app.TIMELINE_JS


# --- pieces -------------------------------------------------------------


def test_filmstrip_is_a_horizontal_strip_of_crops():
    rois = np.random.randint(0, 255, (30, 96, 96), dtype=np.uint8)
    assert web_app._roi_filmstrip(rois, count=6).shape == (96, 96 * 6)


def test_filmstrip_handles_short_and_empty_sequences():
    assert web_app._roi_filmstrip(np.empty((0, 96, 96), np.uint8)) is None
    assert web_app._roi_filmstrip(None) is None
    assert web_app._roi_filmstrip(np.zeros((2, 96, 96), np.uint8), count=10).shape == (96, 192)


def test_every_verdict_renders():
    """A missing style entry would blow up exactly when a video fails."""
    for verdict in Verdict:
        markup = web_app._quality_html(
            QualityReport([Check("face size", verdict, 1.0, "a message")])
        )
        assert "a message" in markup
        assert web_app._MARKER[verdict] in markup


def test_quality_messages_are_escaped():
    markup = web_app._quality_html(
        QualityReport([Check("x", Verdict.GOOD, 1.0, "<b>raw</b>")])
    )
    assert "<b>raw</b>" not in markup


def test_progress_reporting_never_breaks_analysis():
    def explode(*args, **kwargs):
        raise RuntimeError("progress backend died")

    web_app._step(explode, 0.5, "anything")  # must not raise
    web_app._step(None, 0.5, "anything")


def test_dashboard_builds():
    assert web_app.build() is not None

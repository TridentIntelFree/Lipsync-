"""Caption parsing, which is what makes the URL workflow worth anything.

The network fetch itself is not tested here — it needs the internet and a
third-party site. The parsing is where the bugs actually live.
"""

from __future__ import annotations

from lipsync.source import Caption, Clip, _parse_vtt

VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.500
Hello and welcome

00:00:03.500 --> 00:00:06.000
to the show today
"""


def test_parses_timings_and_text():
    captions = _parse_vtt(VTT)
    assert len(captions) == 2
    assert captions[0].start == 1.0
    assert captions[0].end == 3.5
    assert captions[0].text == "Hello and welcome"
    assert captions[1].text == "to the show today"


def test_offset_rebases_onto_the_fetched_clip():
    """A clip starting at 0:10 must report its own captions from zero."""
    captions = _parse_vtt(VTT, offset=1.0)
    assert captions[0].start == 0.0
    assert captions[1].start == 2.5


def test_offset_never_produces_negative_times():
    captions = _parse_vtt(VTT, offset=100.0)
    assert all(c.start >= 0 and c.end >= 0 for c in captions)


def test_inline_styling_tags_are_stripped():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c.colorE5E5E5>bold</c> text\n"
    assert _parse_vtt(vtt)[0].text == "bold text"


def test_rolling_duplicate_lines_are_dropped():
    """Auto-captions repeat the previous line as scrolling context."""
    vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\nfirst line\n\n"
        "00:00:01.000 --> 00:00:02.000\nfirst line\n\n"
        "00:00:02.000 --> 00:00:03.000\nsecond line\n"
    )
    texts = [c.text for c in _parse_vtt(vtt)]
    assert texts == ["first line", "second line"]


def test_blocks_without_timings_are_skipped():
    vtt = "WEBVTT\nSTYLE\n::cue { color: white }\n\n00:00:01.000 --> 00:00:02.000\nreal\n"
    assert [c.text for c in _parse_vtt(vtt)] == ["real"]


def test_malformed_timestamps_do_not_raise():
    vtt = "WEBVTT\n\nnot:a:time --> also:bad\nignored\n\n00:00:01.000 --> 00:00:02.000\nkept\n"
    assert [c.text for c in _parse_vtt(vtt)] == ["kept"]


def test_empty_input_yields_nothing():
    assert _parse_vtt("") == []
    assert _parse_vtt("WEBVTT\n") == []


def test_short_mmss_timestamps_are_understood():
    captions = _parse_vtt("WEBVTT\n\n01:30.000 --> 01:32.000\nlate\n")
    assert captions[0].start == 90.0


def test_clip_reports_whether_captions_exist(tmp_path):
    bare = Clip(tmp_path / "a.mp4", "t", "u", 0.0, 10.0)
    assert not bare.has_captions
    assert bare.caption_text() == ""

    withcaps = Clip(
        tmp_path / "a.mp4", "t", "u", 0.0, 10.0,
        captions=[Caption(0, 1, "hello"), Caption(1, 2, "there")],
    )
    assert withcaps.has_captions
    assert withcaps.caption_text() == "hello there"


# --- failure messages ---------------------------------------------------


def test_bot_check_is_explained_as_a_datacenter_block():
    from lipsync.source import explain_failure

    for text in [
        "ERROR: Sign in to confirm you're not a bot",
        "ERROR: unable to download API page: HTTP Error 403: Forbidden",
        "ERROR: Failed to extract any player response",
    ]:
        advice = explain_failure(text)
        assert "datacenter" in advice
        assert "upload the file" in advice


def test_a_bad_url_says_so_rather_than_blaming_the_site():
    from lipsync.source import explain_failure

    assert "does not look like a video link" in explain_failure("'x' is not a valid URL")


def test_private_video_is_named_as_such():
    from lipsync.source import explain_failure

    assert "private" in explain_failure("ERROR: This video is private").lower()


def test_unknown_failures_still_suggest_the_working_path():
    from lipsync.source import explain_failure

    assert "upload the file" in explain_failure("something entirely unexpected")

"""Decoding real video files, including frame-rate normalisation.

These generate actual video with ffmpeg rather than mocking it, because the
things that break here — variable frame rate, odd dimensions, rotation metadata
— are exactly the things a mock would paper over.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from lipsync.video import VideoError, decode_rgb, ffmpeg_binary, probe


def make_video(path, width=320, height=240, fps=25, seconds=2, color_cycle=True):
    """Write a small test video and return its path."""
    frames = int(fps * seconds)
    ffmpeg = ffmpeg_binary()
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        # A distinct level per frame so resampling is observable.
        frame[:, :, 0] = (i * 255 // max(frames - 1, 1)) if color_cycle else 128
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("video") / "sample.mp4")


def test_probe_reports_dimensions_and_rate(sample):
    info = probe(sample)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(25.0, abs=0.1)
    assert info.duration == pytest.approx(2.0, abs=0.15)


def test_probe_rejects_a_missing_file(tmp_path):
    with pytest.raises(VideoError, match="No such video"):
        probe(tmp_path / "nope.mp4")


def test_probe_rejects_a_non_video(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        probe(junk)


def test_decode_returns_rgb_frames(sample):
    frames = decode_rgb(sample)
    assert frames.ndim == 4 and frames.shape[1:] == (240, 320, 3)
    assert frames.dtype == np.uint8
    assert len(frames) == pytest.approx(50, abs=2)


def test_decode_resamples_to_the_requested_rate(tmp_path):
    """A 10 fps source must come out at 25 fps, same wall-clock duration."""
    source = make_video(tmp_path / "slow.mp4", fps=10, seconds=2)
    assert probe(source).fps == pytest.approx(10.0, abs=0.1)
    frames = decode_rgb(source, fps=25)
    assert len(frames) == pytest.approx(50, abs=3)


def test_max_seconds_truncates(sample):
    frames = decode_rgb(sample, max_seconds=1.0)
    assert len(frames) == pytest.approx(25, abs=2)


def test_decoded_content_changes_over_time(sample):
    """Guards against a decode that silently returns one repeated frame."""
    frames = decode_rgb(sample)
    assert frames[0, :, :, 0].mean() < frames[-1, :, :, 0].mean() - 50


def test_odd_dimensions_are_handled(tmp_path):
    source = make_video(tmp_path / "odd.mp4", width=322, height=242, seconds=1)
    frames = decode_rgb(source)
    assert frames.shape[1:] == (242, 322, 3)

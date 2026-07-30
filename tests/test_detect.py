"""Gap filling and smoothing of per-frame keypoints."""

from __future__ import annotations

import numpy as np
import pytest

from lipsync.detect import DetectionError, interpolate_missing, smooth


def kp(value: float) -> np.ndarray:
    return np.full((4, 2), value, dtype=np.float64)


def test_interpolation_fills_an_interior_gap_linearly():
    filled = interpolate_missing([kp(0.0), None, None, kp(3.0)])
    assert len(filled) == 4
    np.testing.assert_allclose(filled[1], kp(1.0))
    np.testing.assert_allclose(filled[2], kp(2.0))


def test_leading_and_trailing_gaps_hold_the_nearest_detection():
    filled = interpolate_missing([None, None, kp(5.0), None])
    np.testing.assert_allclose(filled[0], kp(5.0))
    np.testing.assert_allclose(filled[1], kp(5.0))
    np.testing.assert_allclose(filled[3], kp(5.0))


def test_frames_with_detections_are_left_untouched():
    original = [kp(1.0), kp(7.0), kp(2.0)]
    filled = interpolate_missing(list(original))
    for got, want in zip(filled, original):
        np.testing.assert_allclose(got, want)


def test_a_video_with_no_face_anywhere_is_an_error():
    with pytest.raises(DetectionError, match="No face detected"):
        interpolate_missing([None, None, None])


def test_output_length_always_matches_input_length():
    frames = [kp(1.0), None, None, kp(4.0), None, kp(6.0), None]
    assert len(interpolate_missing(frames)) == len(frames)


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    steady = [kp(10.0) + rng.normal(0, 1.5, (4, 2)) for _ in range(40)]
    smoothed = smooth(steady)

    def jitter(seq):
        return float(np.mean([np.abs(b - a).mean() for a, b in zip(seq, seq[1:])]))

    assert jitter(smoothed) < jitter(steady)


def test_smoothing_preserves_each_frames_own_position():
    """Shape is stabilised; location must not be dragged toward neighbours."""
    moving = [kp(0.0) + i * 10 for i in range(30)]
    smoothed = smooth(moving)
    for original, result in zip(moving, smoothed):
        np.testing.assert_allclose(result.mean(axis=0), original.mean(axis=0), atol=1e-9)


def test_smoothing_keeps_length_and_shape():
    seq = [kp(float(i)) for i in range(15)]
    smoothed = smooth(seq)
    assert len(smoothed) == 15
    assert all(s.shape == (4, 2) for s in smoothed)

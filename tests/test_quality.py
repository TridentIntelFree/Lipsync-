"""The quality gate is the only thing standing between a bad video and a
confident, invented transcript, so its thresholds get tested directly."""

from __future__ import annotations

import numpy as np

from lipsync.constants import MODEL_FPS
from lipsync.quality import REFERENCE_INTEROCULAR, Verdict, assess


def face(interocular: float = REFERENCE_INTEROCULAR, yaw: float = 0.0) -> np.ndarray:
    """A synthetic frontal face; ``yaw`` shifts the nose sideways between the eyes."""
    left_eye_x = 100.0
    right_eye_x = left_eye_x + interocular
    mid_x = (left_eye_x + right_eye_x) / 2.0
    return np.array(
        [
            [left_eye_x, 100.0],
            [right_eye_x, 100.0],
            [mid_x + yaw * interocular, 140.0],
            [mid_x, 165.0],
        ],
        dtype=np.float64,
    )


def sharp_rois(count: int = 30) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(40, 200, (count, 96, 96), dtype=np.uint8)


def find(report, name):
    return next(c for c in report.checks if c.name == name)


def test_clean_input_passes_every_check():
    report = assess(MODEL_FPS, [face()] * 50, sharp_rois(50))
    assert report.verdict is Verdict.GOOD
    assert report.problems == []


def test_no_face_anywhere_is_unusable_and_short_circuits():
    report = assess(MODEL_FPS, [None] * 20)
    assert report.verdict is Verdict.UNUSABLE
    assert len(report.checks) == 1


def test_mostly_missing_face_is_unusable():
    keypoints = [face()] * 4 + [None] * 16
    report = assess(MODEL_FPS, keypoints, sharp_rois())
    assert find(report, "face detection").verdict is Verdict.UNUSABLE


def test_occasional_dropouts_are_only_a_warning():
    keypoints = [face()] * 15 + [None] * 5  # 75% detected
    report = assess(MODEL_FPS, keypoints, sharp_rois())
    assert find(report, "face detection").verdict is Verdict.MARGINAL


def test_ninety_percent_detection_is_the_good_boundary():
    keypoints = [face()] * 18 + [None] * 2
    report = assess(MODEL_FPS, keypoints, sharp_rois())
    assert find(report, "face detection").verdict is Verdict.GOOD


def test_low_frame_rate_is_unusable():
    report = assess(8.0, [face()] * 20, sharp_rois())
    assert find(report, "frame rate").verdict is Verdict.UNUSABLE


def test_slightly_low_frame_rate_is_marginal():
    report = assess(20.0, [face()] * 20, sharp_rois())
    assert find(report, "frame rate").verdict is Verdict.MARGINAL


def test_tiny_face_is_unusable():
    report = assess(MODEL_FPS, [face(interocular=12.0)] * 20, sharp_rois())
    assert find(report, "face size").verdict is Verdict.UNUSABLE


def test_small_face_is_marginal():
    report = assess(
        MODEL_FPS, [face(interocular=REFERENCE_INTEROCULAR * 0.7)] * 20, sharp_rois()
    )
    assert find(report, "face size").verdict is Verdict.MARGINAL


def test_profile_view_is_unusable():
    report = assess(MODEL_FPS, [face(yaw=0.45)] * 20, sharp_rois())
    assert find(report, "head pose").verdict is Verdict.UNUSABLE


def test_blurred_mouth_is_unusable():
    flat = np.full((20, 96, 96), 120, dtype=np.uint8)
    report = assess(MODEL_FPS, [face()] * 20, flat)
    assert find(report, "sharpness").verdict is Verdict.UNUSABLE


def test_dark_footage_is_flagged():
    dark = (sharp_rois() // 12).astype(np.uint8)
    report = assess(MODEL_FPS, [face()] * 30, dark)
    assert find(report, "exposure").verdict is Verdict.MARGINAL


def test_worst_check_determines_the_overall_verdict():
    report = assess(8.0, [face()] * 20, sharp_rois())
    assert report.verdict is Verdict.UNUSABLE


def test_summary_names_every_check():
    report = assess(MODEL_FPS, [face()] * 20, sharp_rois())
    text = report.summary()
    for check in report.checks:
        assert check.name in text

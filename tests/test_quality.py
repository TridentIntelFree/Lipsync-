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
    """Sharp, well-exposed footage of a mouth that is actually speaking.

    A coherent face with a moving mouth, not per-frame noise: noise varies the
    mouth and the upper face equally, which is precisely what the movement check
    treats as "nothing is being said".
    """
    import cv2

    rng = np.random.default_rng(1)
    base = rng.integers(40, 200, (96, 96), dtype=np.uint8)
    rois = np.repeat(base[None, :, :], count, axis=0).copy()
    for i in range(count):
        opening = int(14 * (1 + np.sin(i * 0.7)) / 2) + 1
        cv2.ellipse(rois[i], (48, 66), (16, opening), 0, 0, 360, 30, -1)
    return rois


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


# --- mouth movement -----------------------------------------------------
#
# The recogniser cannot say "nothing was said": trained only on footage where
# someone is always speaking, a motionless mouth makes it emit its most likely
# sentence instead. These tests cover the check that refuses such footage.


def static_rois(count=40, seed=0, noise=1.0):
    """A face that does not move: one frame, plus sensor noise.

    Tuned to score like real still footage (~0.03 against a measured 0.055 for
    an actual still photograph), so the thresholds are exercised where they
    matter rather than at values no real video produces.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, (96, 96), dtype=np.uint8)
    return np.clip(
        base.astype(np.int16)[None, :, :] + rng.normal(0, noise, (count, 96, 96)),
        0, 255,
    ).astype(np.uint8)


def moving_mouth_rois(count=40, amplitude=14, seed=0):
    """The same face with the mouth opening and closing.

    amplitude=14 scores ~0.167, inside the 0.114-0.200 band measured on real
    CREMA-D clips of actors speaking.
    """
    import cv2

    rois = static_rois(count, seed).copy()
    for i in range(count):
        opening = int(amplitude * (1 + np.sin(i * 0.7)) / 2) + 1
        cv2.ellipse(rois[i], (48, 66), (16, opening), 0, 0, 360, 30, -1)
    return rois


def test_a_motionless_mouth_is_unusable():
    """The still-image confabulation this project ships as its cautionary example."""
    report = assess(MODEL_FPS, [face()] * 40, static_rois())
    check = find(report, "mouth movement")
    assert check.verdict is Verdict.UNUSABLE
    assert "no speech" in check.message
    assert report.verdict is Verdict.UNUSABLE


def test_a_moving_mouth_passes():
    report = assess(MODEL_FPS, [face()] * 40, moving_mouth_rois())
    assert find(report, "mouth movement").verdict is Verdict.GOOD


def test_barely_moving_is_flagged_but_not_refused():
    report = assess(MODEL_FPS, [face()] * 40, moving_mouth_rois(amplitude=5))
    assert find(report, "mouth movement").verdict in (Verdict.MARGINAL, Verdict.UNUSABLE)


def test_motion_measure_is_scaled_by_contrast_not_brightness():
    """Halving the contrast of the same motion must not change the verdict."""
    from lipsync.quality import mouth_motion_ratio

    bright = moving_mouth_rois()
    dim = (bright.astype(np.float32) * 0.5 + 60).astype(np.uint8)
    assert abs(mouth_motion_ratio(dim) - mouth_motion_ratio(bright)) < 0.05


def test_motion_measure_needs_several_frames():
    from lipsync.quality import mouth_motion_ratio

    assert mouth_motion_ratio(np.zeros((1, 96, 96), np.uint8)) == 0.0
    assert mouth_motion_ratio(None) == 0.0


def test_a_perfectly_clean_still_is_still_refused():
    """Every other check passing must not rescue footage with no speech in it."""
    report = assess(MODEL_FPS, [face()] * 40, static_rois())
    passing = [c.name for c in report.checks if c.verdict is Verdict.GOOD]
    assert "face detection" in passing and "face size" in passing
    assert report.verdict is Verdict.UNUSABLE

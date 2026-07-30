"""Alignment must remove head pose, and it must remove it the same way every time.

These are the tests that matter most: the crop geometry is a contract with the
released checkpoint, and getting it wrong degrades accuracy without ever raising
an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from lipsync.align import (
    AlignmentError,
    apply_transform,
    cut_patch,
    estimate_transform,
    mouth_roi_sequence,
)
from lipsync.constants import (
    ALIGNED_SIZE,
    CROP_HEIGHT,
    CROP_WIDTH,
    MOUTH_CENTER_IDX,
    STABLE_REFERENCE,
)


def similarity(points: np.ndarray, angle_deg: float, scale: float, shift) -> np.ndarray:
    """Apply a known rotation, scale and translation to a set of points."""
    theta = np.radians(angle_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return points @ (scale * rotation).T + np.asarray(shift)


def test_reference_keypoints_map_to_themselves():
    transform = estimate_transform(STABLE_REFERENCE)
    _, moved = apply_transform(
        np.zeros(ALIGNED_SIZE, np.uint8), STABLE_REFERENCE, transform
    )
    np.testing.assert_allclose(moved, STABLE_REFERENCE, atol=1e-6)


@pytest.mark.parametrize(
    "angle,scale,shift",
    [(0, 1.0, (0, 0)), (20, 2.0, (140, -60)), (-35, 0.4, (-25, 90)), (170, 1.3, (5, 5))],
)
def test_alignment_undoes_any_similarity_transform(angle, scale, shift):
    """A face rotated, scaled and moved must land back on the reference layout."""
    observed = similarity(STABLE_REFERENCE, angle, scale, shift)
    transform = estimate_transform(observed)
    _, moved = apply_transform(np.zeros(ALIGNED_SIZE, np.uint8), observed, transform)
    np.testing.assert_allclose(moved, STABLE_REFERENCE, atol=1e-4)


def test_wrong_keypoint_count_is_rejected():
    with pytest.raises(AlignmentError):
        estimate_transform(np.zeros((68, 2)))


def test_cut_patch_is_centred_and_sized():
    image = np.zeros((256, 256), np.uint8)
    image[120:136, 120:136] = 255  # a marker centred on (128, 128)
    patch = cut_patch(image, np.array([128.0, 128.0]))

    assert patch.shape == (CROP_HEIGHT, CROP_WIDTH)
    centre = patch[CROP_HEIGHT // 2, CROP_WIDTH // 2]
    assert centre == 255
    # The marker should sit in the middle of the patch, not at an edge.
    ys, xs = np.nonzero(patch)
    assert abs(ys.mean() - CROP_HEIGHT / 2) < 1.0
    assert abs(xs.mean() - CROP_WIDTH / 2) < 1.0


def test_cut_patch_pads_rather_than_shrinking_at_the_edge():
    """Off-frame crops keep their shape so frames can still be stacked."""
    image = np.full((256, 256), 200, np.uint8)
    patch = cut_patch(image, np.array([2.0, 2.0]))
    assert patch.shape == (CROP_HEIGHT, CROP_WIDTH)
    assert patch[0, 0] == 0  # padded region
    assert patch[-1, -1] == 200  # real image


def test_mouth_roi_sequence_shape_and_dtype():
    frames = np.random.randint(0, 255, (5, 240, 320, 3), dtype=np.uint8)
    keypoints = [STABLE_REFERENCE.copy() for _ in range(5)]
    rois = mouth_roi_sequence(frames, keypoints)
    assert rois.shape == (5, CROP_HEIGHT, CROP_WIDTH)
    assert rois.dtype == np.uint8


def test_mouth_roi_sequence_rejects_mismatched_lengths():
    frames = np.zeros((3, 100, 100, 3), np.uint8)
    with pytest.raises(AlignmentError):
        mouth_roi_sequence(frames, [STABLE_REFERENCE.copy()])


def test_crop_follows_the_mouth_keypoint_not_the_frame_centre():
    """The patch is cut at the mouth, which is below centre in the aligned frame."""
    transform = estimate_transform(STABLE_REFERENCE)
    _, moved = apply_transform(
        np.zeros(ALIGNED_SIZE, np.uint8), STABLE_REFERENCE, transform
    )
    mouth_y = moved[MOUTH_CENTER_IDX][1]
    assert mouth_y > ALIGNED_SIZE[1] / 2

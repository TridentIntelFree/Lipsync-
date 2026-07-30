"""Align each frame to a canonical face and cut the mouth patch.

The network was trained on mouths that always sit in the same place at the same
scale and rotation, so head pose and camera distance have to be removed before
inference. Each frame gets a similarity transform (rotation, uniform scale,
translation — no shear, no perspective) that carries its four keypoints onto a
fixed reference layout; the mouth patch is then cut at a fixed size from the
resulting canonical frame.

Any deviation from the training geometry here costs accuracy without raising an
error, so the constants come from ``constants`` and are not parameters.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .constants import (
    ALIGNED_SIZE,
    CROP_HEIGHT,
    CROP_WIDTH,
    MOUTH_CENTER_IDX,
    STABLE_REFERENCE,
)


class AlignmentError(RuntimeError):
    """Raised when a frame cannot be aligned to the reference face."""


def estimate_transform(keypoints: np.ndarray) -> np.ndarray:
    """Return the 2x3 similarity transform mapping keypoints to the reference.

    Uses LMEDS so that one badly-placed keypoint — a common failure when the
    mouth is occluded — does not drag the whole alignment with it.
    """
    if keypoints.shape != STABLE_REFERENCE.shape:
        raise AlignmentError(
            f"Expected keypoints of shape {STABLE_REFERENCE.shape}, "
            f"got {keypoints.shape}"
        )
    transform, _ = cv2.estimateAffinePartial2D(
        keypoints.astype(np.float64), STABLE_REFERENCE, method=cv2.LMEDS
    )
    if transform is None:
        raise AlignmentError(
            "Could not fit a face alignment transform; keypoints may be degenerate"
        )
    return transform


def apply_transform(
    frame: np.ndarray, keypoints: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a frame and its keypoints into the canonical face frame."""
    warped = cv2.warpAffine(
        frame,
        transform,
        dsize=ALIGNED_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    moved = keypoints @ transform[:, :2].T + transform[:, 2]
    return warped, moved


def cut_patch(
    image: np.ndarray,
    center: np.ndarray,
    height: int = CROP_HEIGHT,
    width: int = CROP_WIDTH,
) -> np.ndarray:
    """Cut a fixed-size patch centred on ``center``, padding if it runs off frame.

    Padding rather than clipping keeps every patch the same shape, which matters
    because the frames are stacked into one tensor.
    """
    half_h, half_w = height // 2, width // 2
    cx, cy = float(center[0]), float(center[1])

    y_min, y_max = int(round(cy - half_h)), int(round(cy + half_h))
    x_min, x_max = int(round(cx - half_w)), int(round(cx + half_w))

    pad_top = max(0, -y_min)
    pad_left = max(0, -x_min)
    pad_bottom = max(0, y_max - image.shape[0])
    pad_right = max(0, x_max - image.shape[1])

    if pad_top or pad_left or pad_bottom or pad_right:
        pad_spec = [(pad_top, pad_bottom), (pad_left, pad_right)]
        pad_spec += [(0, 0)] * (image.ndim - 2)
        image = np.pad(image, pad_spec, mode="constant")
        y_min += pad_top
        y_max += pad_top
        x_min += pad_left
        x_max += pad_left

    return np.ascontiguousarray(image[y_min:y_max, x_min:x_max])


def mouth_roi_sequence(
    frames: np.ndarray, keypoints: Sequence[np.ndarray]
) -> np.ndarray:
    """Turn RGB frames plus keypoints into a ``(T, 96, 96)`` uint8 grayscale stack.

    Grayscale conversion happens before the warp, matching training: interpolating
    one channel is not the same as interpolating three and then mixing them.
    """
    if len(frames) != len(keypoints):
        raise AlignmentError(
            f"Have {len(frames)} frames but {len(keypoints)} keypoint sets"
        )

    patches = np.empty((len(frames), CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)
    for index, (frame, points) in enumerate(zip(frames, keypoints)):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        transform = estimate_transform(points)
        aligned, moved = apply_transform(gray, points, transform)
        patches[index] = cut_patch(aligned, moved[MOUTH_CENTER_IDX])
    return patches

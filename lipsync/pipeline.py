"""End-to-end preprocessing: a video file in, a model-ready tensor out.

This is the half of the system that can be verified without model weights, and
it is the half that decides whether the other half has anything to work with.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .align import mouth_roi_sequence
from .constants import (
    CROP_HEIGHT,
    CROP_WIDTH,
    MODEL_FPS,
    NETWORK_CROP,
    PIXEL_MEAN,
    PIXEL_STD,
)
from .detect import FaceKeypointDetector, interpolate_missing, smooth
from .quality import QualityReport, assess
from .video import VideoInfo, decode_rgb, probe


@dataclasses.dataclass
class PreparedVideo:
    """Everything preprocessing learned about a video."""

    info: VideoInfo
    mouth_rois: np.ndarray  # (T, 96, 96) uint8
    quality: QualityReport
    detection_rate: float

    @property
    def duration_seconds(self) -> float:
        return len(self.mouth_rois) / MODEL_FPS


def center_crop(frames: np.ndarray, size: int = NETWORK_CROP) -> np.ndarray:
    """Take a centred ``size x size`` crop from each frame of a ``(T, H, W)`` stack."""
    _, height, width = frames.shape
    if height < size or width < size:
        raise ValueError(f"Cannot crop {size}x{size} from {height}x{width} frames")
    top = (height - size) // 2
    left = (width - size) // 2
    return frames[:, top : top + size, left : left + size]


def to_model_tensor(mouth_rois: np.ndarray) -> np.ndarray:
    """Normalise ROIs into the ``(1, T, 88, 88)`` float32 layout the network wants.

    Scaling, cropping and normalisation constants all come from training; see
    ``constants``.
    """
    cropped = center_crop(mouth_rois).astype(np.float32) / 255.0
    normalised = (cropped - PIXEL_MEAN) / PIXEL_STD
    return normalised[np.newaxis, ...]


def prepare(
    path: str,
    max_seconds: float | None = None,
    min_confidence: float = 0.5,
) -> PreparedVideo:
    """Decode, align and quality-check a video, stopping short of recognition."""
    info = probe(path)
    frames = decode_rgb(path, fps=MODEL_FPS, max_seconds=max_seconds)

    with FaceKeypointDetector(min_confidence=min_confidence) as detector:
        raw_keypoints = detector.detect(frames, fps=MODEL_FPS)

    detected = sum(kp is not None for kp in raw_keypoints)
    detection_rate = detected / len(raw_keypoints) if raw_keypoints else 0.0

    # A video with no face anywhere is a quality verdict, not a crash: the
    # report already describes this case better than a traceback would.
    if detected == 0:
        return PreparedVideo(
            info=info,
            mouth_rois=np.empty((0, CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8),
            quality=assess(source_fps=info.fps, keypoints=raw_keypoints),
            detection_rate=0.0,
        )

    filled = interpolate_missing(raw_keypoints)
    stabilised = smooth(filled)
    rois = mouth_roi_sequence(frames, stabilised)

    report = assess(
        source_fps=info.fps, keypoints=raw_keypoints, mouth_rois=rois
    )

    return PreparedVideo(
        info=info,
        mouth_rois=rois,
        quality=report,
        detection_rate=detection_rate,
    )

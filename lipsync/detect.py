"""Per-frame face keypoint detection.

Only four keypoints matter downstream — right eye, left eye, nose tip and mouth
centre — because those are the four the alignment was trained against. MediaPipe's
face *detector* returns exactly these (plus two ear points we discard) in the
required order, so no 468-point mesh is needed.

Detection fails on some frames of any real video: a blink, motion blur, a hand
crossing the face. Those gaps are filled by interpolation rather than dropped,
because the model reads a continuous time series and a missing frame would shift
every subsequent one.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .assets import fetch
from .constants import N_KEYPOINTS, SMOOTHING_HALF_WINDOW

_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_DETECTOR_NAME = "blaze_face_short_range.tflite"


class DetectionError(RuntimeError):
    """Raised when no face can be found anywhere in the video."""


def detector_model_path():
    """Return the local path to the MediaPipe face detector, fetching if needed."""
    return fetch(_DETECTOR_URL, _DETECTOR_NAME)


class FaceKeypointDetector:
    """Detects the four alignment keypoints in each frame of a video.

    Use as a context manager; MediaPipe holds native resources that should be
    released deterministically.
    """

    def __init__(self, min_confidence: float = 0.5):
        # Imported lazily: mediapipe is slow to import and pulls in TensorFlow
        # Lite, which we do not want to pay for during e.g. --help.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(detector_model_path())
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def __enter__(self) -> "FaceKeypointDetector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        detector, self._detector = getattr(self, "_detector", None), None
        if detector is not None:
            detector.close()

    def detect(self, frames: np.ndarray, fps: float) -> list[np.ndarray | None]:
        """Return one ``(4, 2)`` float array per frame, or None where no face was found.

        When several faces are visible the largest is used, on the assumption
        that the speaker is the subject of the shot.
        """
        results: list[np.ndarray | None] = []
        for index, frame in enumerate(frames):
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame),
            )
            timestamp_ms = int(round(index * 1000.0 / fps))
            detection = self._detector.detect_for_video(image, timestamp_ms)

            if not detection.detections:
                results.append(None)
                continue

            best = max(
                detection.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            height, width = frame.shape[:2]
            points = np.array(
                [[kp.x * width, kp.y * height] for kp in best.keypoints[:N_KEYPOINTS]],
                dtype=np.float64,
            )
            results.append(points if points.shape == (N_KEYPOINTS, 2) else None)
        return results


def interpolate_missing(
    keypoints: Sequence[np.ndarray | None],
) -> list[np.ndarray]:
    """Fill None entries by linear interpolation, extending flat at the ends.

    Raises DetectionError if no frame had a detection at all.
    """
    filled = list(keypoints)
    found = [i for i, kp in enumerate(filled) if kp is not None]
    if not found:
        raise DetectionError(
            "No face detected in any frame. The subject's face may be too "
            "small, too dark, turned away, or absent."
        )

    for left, right in zip(found, found[1:]):
        gap = right - left
        if gap == 1:
            continue
        start, delta = filled[left], filled[right] - filled[left]
        for step in range(1, gap):
            filled[left + step] = start + (step / gap) * delta

    # Hold the first and last real detections out to the ends of the clip.
    for i in range(found[0]):
        filled[i] = filled[found[0]]
    for i in range(found[-1] + 1, len(filled)):
        filled[i] = filled[found[-1]]

    return [kp for kp in filled if kp is not None]


def smooth(
    keypoints: Sequence[np.ndarray],
    half_window: int = SMOOTHING_HALF_WINDOW,
) -> list[np.ndarray]:
    """Temporally smooth keypoints while preserving each frame's own position.

    Detector jitter makes the crop shake, which the model reads as mouth motion.
    Averaging over a window removes the jitter but would also drag the crop
    toward neighbouring frames' head positions, so the smoothed shape is
    re-centred on the current frame's own centroid: the *shape* is stabilised,
    the *location* is not.
    """
    count = len(keypoints)
    smoothed: list[np.ndarray] = []
    for index in range(count):
        margin = min(half_window, index, count - 1 - index)
        window = keypoints[index - margin : index + margin + 1] if margin else [
            keypoints[index]
        ]
        mean_shape = np.mean(window, axis=0)
        mean_shape = mean_shape + (
            keypoints[index].mean(axis=0) - mean_shape.mean(axis=0)
        )
        smoothed.append(mean_shape)
    return smoothed

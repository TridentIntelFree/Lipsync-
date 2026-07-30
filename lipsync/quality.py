"""Judge whether a video can support lip reading at all.

Visual speech recognition degrades quietly. Feed it a half-resolution profile
shot and it returns a fluent, confident, entirely invented sentence — the
language model will always produce *something* English-shaped. There is no
signal in the output that says "this was guesswork".

So the check has to happen on the input. These measurements bound what is
achievable before any inference runs, and the thresholds are drawn from how the
released checkpoints were trained: 25 fps, near-frontal, roughly 55 pixels
between the eyes once aligned.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Sequence

import cv2
import numpy as np

from .constants import MODEL_FPS, STABLE_REFERENCE

# Interocular distance the alignment scales every face to. Source faces smaller
# than this are being upscaled, which invents detail rather than revealing it.
REFERENCE_INTEROCULAR = float(
    np.linalg.norm(STABLE_REFERENCE[1] - STABLE_REFERENCE[0])
)


class Verdict(str, Enum):
    """How much to trust anything produced from this video."""

    GOOD = "good"
    MARGINAL = "marginal"
    UNUSABLE = "unusable"


@dataclasses.dataclass
class Check:
    name: str
    verdict: Verdict
    value: float
    message: str


@dataclasses.dataclass
class QualityReport:
    checks: list[Check]

    @property
    def verdict(self) -> Verdict:
        if any(c.verdict is Verdict.UNUSABLE for c in self.checks):
            return Verdict.UNUSABLE
        if any(c.verdict is Verdict.MARGINAL for c in self.checks):
            return Verdict.MARGINAL
        return Verdict.GOOD

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if c.verdict is not Verdict.GOOD]

    def summary(self) -> str:
        lines = [f"Input quality: {self.verdict.value.upper()}"]
        for check in self.checks:
            marker = {
                Verdict.GOOD: "ok  ",
                Verdict.MARGINAL: "warn",
                Verdict.UNUSABLE: "FAIL",
            }[check.verdict]
            lines.append(f"  [{marker}] {check.name}: {check.message}")
        return "\n".join(lines)


def _interocular(keypoints: np.ndarray) -> float:
    return float(np.linalg.norm(keypoints[1] - keypoints[0]))


def _yaw_ratio(keypoints: np.ndarray) -> float:
    """Estimate head turn as the nose's sideways offset between the eyes.

    Zero when the nose sits midway between the eyes (frontal); grows as the head
    turns and the nose drifts toward the nearer eye. Normalised by interocular
    distance so it does not depend on face size.
    """
    eye_mid = (keypoints[0] + keypoints[1]) / 2.0
    eye_axis = keypoints[1] - keypoints[0]
    span = np.linalg.norm(eye_axis)
    if span < 1e-6:
        return 0.0
    unit = eye_axis / span
    return float(abs(np.dot(keypoints[2] - eye_mid, unit)) / span)


def _roll_degrees(keypoints: np.ndarray) -> float:
    delta = keypoints[1] - keypoints[0]
    return float(abs(np.degrees(np.arctan2(delta[1], delta[0]))))


def assess(
    source_fps: float,
    keypoints: Sequence[np.ndarray | None],
    mouth_rois: np.ndarray | None = None,
) -> QualityReport:
    """Measure whether this video can support lip reading.

    ``keypoints`` should be the raw per-frame detections, with None left in for
    frames where no face was found — the gaps are themselves a quality signal.
    """
    checks: list[Check] = []
    total = len(keypoints)
    found = [kp for kp in keypoints if kp is not None]

    # --- Was there a face at all, and how consistently? ---
    rate = len(found) / total if total else 0.0
    if rate == 0:
        checks.append(
            Check(
                "face detection",
                Verdict.UNUSABLE,
                0.0,
                "no face found in any frame",
            )
        )
        return QualityReport(checks)
    if rate < 0.5:
        verdict, msg = Verdict.UNUSABLE, (
            f"face found in only {rate:.0%} of frames; more than half the "
            "speech is unobserved"
        )
    elif rate < 0.9:
        verdict, msg = Verdict.MARGINAL, (
            f"face found in {rate:.0%} of frames; gaps are interpolated and "
            "those spans are guesses"
        )
    else:
        verdict, msg = Verdict.GOOD, f"face found in {rate:.0%} of frames"
    checks.append(Check("face detection", verdict, rate, msg))

    # --- Frame rate: lip movements alias badly when undersampled ---
    if source_fps < 15:
        verdict, msg = Verdict.UNUSABLE, (
            f"{source_fps:.1f} fps is far below the {MODEL_FPS:.0f} fps the "
            "model expects; fast consonants fall between frames"
        )
    elif source_fps < MODEL_FPS - 0.5:
        verdict, msg = Verdict.MARGINAL, (
            f"{source_fps:.1f} fps is below the {MODEL_FPS:.0f} fps the model "
            "expects; frames are duplicated to compensate"
        )
    else:
        verdict, msg = Verdict.GOOD, f"{source_fps:.1f} fps"
    checks.append(Check("frame rate", verdict, source_fps, msg))

    # --- Face size: how much real mouth detail exists in the source ---
    interocular = float(np.median([_interocular(kp) for kp in found]))
    scale = interocular / REFERENCE_INTEROCULAR
    if scale < 0.5:
        verdict, msg = Verdict.UNUSABLE, (
            f"{interocular:.0f}px between the eyes; the mouth is upscaled "
            f"{1 / scale:.1f}x and carries almost no real detail"
        )
    elif scale < 1.0:
        verdict, msg = Verdict.MARGINAL, (
            f"{interocular:.0f}px between the eyes; below the "
            f"{REFERENCE_INTEROCULAR:.0f}px the model trained on"
        )
    else:
        verdict, msg = Verdict.GOOD, f"{interocular:.0f}px between the eyes"
    checks.append(Check("face size", verdict, interocular, msg))

    # --- Head pose: profile views hide the lips ---
    yaw = float(np.median([_yaw_ratio(kp) for kp in found]))
    if yaw > 0.35:
        verdict, msg = Verdict.UNUSABLE, (
            f"head turned well off-axis (yaw ratio {yaw:.2f}); the lips are "
            "substantially self-occluded"
        )
    elif yaw > 0.20:
        verdict, msg = Verdict.MARGINAL, (
            f"head turned off-axis (yaw ratio {yaw:.2f}); training data is "
            "near-frontal"
        )
    else:
        verdict, msg = Verdict.GOOD, f"near-frontal (yaw ratio {yaw:.2f})"
    checks.append(Check("head pose", verdict, yaw, msg))

    roll = float(np.median([_roll_degrees(kp) for kp in found]))
    roll = min(roll, 180.0 - roll)
    checks.append(
        Check(
            "head roll",
            Verdict.MARGINAL if roll > 30 else Verdict.GOOD,
            roll,
            f"{roll:.0f} degrees of tilt"
            + ("; alignment corrects this but crops more background" if roll > 30 else ""),
        )
    )

    # --- Sharpness of the actual mouth region ---
    if mouth_rois is not None and len(mouth_rois):
        sharpness = float(
            np.median([cv2.Laplacian(roi, cv2.CV_64F).var() for roi in mouth_rois])
        )
        if sharpness < 20:
            verdict, msg = Verdict.UNUSABLE, (
                f"mouth region is very blurry (variance {sharpness:.0f}); "
                "lip shapes are not resolvable"
            )
        elif sharpness < 60:
            verdict, msg = Verdict.MARGINAL, (
                f"mouth region is soft (variance {sharpness:.0f}); motion blur "
                "or heavy compression"
            )
        else:
            verdict, msg = Verdict.GOOD, f"mouth region sharp (variance {sharpness:.0f})"
        checks.append(Check("sharpness", verdict, sharpness, msg))

        brightness = float(np.median(mouth_rois))
        if brightness < 30 or brightness > 225:
            verdict, msg = Verdict.MARGINAL, (
                f"mouth region is {'very dark' if brightness < 30 else 'blown out'} "
                f"(median level {brightness:.0f})"
            )
        else:
            verdict, msg = Verdict.GOOD, f"exposure reasonable (median level {brightness:.0f})"
        checks.append(Check("exposure", verdict, brightness, msg))

    return QualityReport(checks)

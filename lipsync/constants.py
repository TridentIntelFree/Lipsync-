"""Geometry and normalisation constants for the Auto-AVSR visual front end.

Every value here is dictated by how the released VSR checkpoints were trained.
Changing any of them silently degrades accuracy rather than raising an error, so
treat this module as a fixed contract with the checkpoint, not as tuning knobs.

Provenance: the reference pipeline aligns each frame to a 68-point mean face
(``20words_mean_face.npy`` from Imperial College London's
Visual_Speech_Recognition_for_Multiple_Languages, Apache-2.0) but only ever uses
four points derived from it. Those four are precomputed into STABLE_REFERENCE so
there is no runtime dependency on the .npy file:

    right eye    = mean(mean_face[36:42])
    left eye     = mean(mean_face[42:48])
    nose tip     = mean(mean_face[31:36])
    mouth centre = mean(mean_face[48:68])
"""

from __future__ import annotations

import numpy as np

# Four-point alignment target, in the 256x256 aligned face frame.
# Row order matches MediaPipe's face-detector keypoint order exactly:
# 0 right eye, 1 left eye, 2 nose tip, 3 mouth centre.
STABLE_REFERENCE = np.array(
    [
        [102.07394306, 94.27230352],
        [156.36130542, 93.57815605],
        [129.00373787, 135.90343029],
        [129.31337323, 157.82299635],
    ],
    dtype=np.float64,
)

# MediaPipe keypoint indices, in the order the alignment expects them.
KEYPOINT_NAMES = ("right_eye", "left_eye", "nose_tip", "mouth_center")
N_KEYPOINTS = len(KEYPOINT_NAMES)
MOUTH_CENTER_IDX = 3

# Size of the aligned face frame that the affine warp targets.
ALIGNED_SIZE = (256, 256)

# Mouth patch cut from the aligned frame, centred on the warped mouth keypoint.
CROP_HEIGHT = 96
CROP_WIDTH = 96

# The network consumes an 88x88 centre crop of the 96x96 patch.
NETWORK_CROP = 88

# Grayscale normalisation used during training.
PIXEL_MEAN = 0.421
PIXEL_STD = 0.165

# The checkpoints are trained on 25 fps video. Input is resampled to this.
MODEL_FPS = 25.0

# Half-width, in frames, of the landmark smoothing window (reference uses
# window_margin=12, applied as +/- window_margin//2).
SMOOTHING_HALF_WINDOW = 6

"""Tensor shaping between the mouth crops and the network."""

from __future__ import annotations

import numpy as np
import pytest

from lipsync.constants import NETWORK_CROP, PIXEL_MEAN, PIXEL_STD
from lipsync.pipeline import center_crop, to_model_tensor


def test_center_crop_takes_the_middle():
    frames = np.zeros((2, 96, 96), np.uint8)
    frames[:, 4 : 96 - 4, 4 : 96 - 4] = 255  # exactly the 88x88 centre
    cropped = center_crop(frames)
    assert cropped.shape == (2, NETWORK_CROP, NETWORK_CROP)
    assert (cropped == 255).all()


def test_center_crop_refuses_to_upscale():
    with pytest.raises(ValueError):
        center_crop(np.zeros((2, 40, 40), np.uint8))


def test_model_tensor_has_the_layout_the_network_expects():
    rois = np.random.randint(0, 255, (12, 96, 96), dtype=np.uint8)
    tensor = to_model_tensor(rois)
    assert tensor.shape == (1, 12, NETWORK_CROP, NETWORK_CROP)
    assert tensor.dtype == np.float32


def test_normalisation_matches_the_training_constants():
    rois = np.full((3, 96, 96), 128, dtype=np.uint8)
    tensor = to_model_tensor(rois)
    expected = (128 / 255.0 - PIXEL_MEAN) / PIXEL_STD
    np.testing.assert_allclose(tensor, expected, rtol=1e-5)


def test_black_and_white_map_to_the_expected_extremes():
    black = to_model_tensor(np.zeros((1, 96, 96), np.uint8))
    white = to_model_tensor(np.full((1, 96, 96), 255, np.uint8))
    np.testing.assert_allclose(black, (0 - PIXEL_MEAN) / PIXEL_STD, rtol=1e-5)
    np.testing.assert_allclose(white, (1 - PIXEL_MEAN) / PIXEL_STD, rtol=1e-5)

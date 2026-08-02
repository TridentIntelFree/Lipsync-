"""Evaluating an expression while a chosen number of variables vary.

This is what the dimension slider does, so the tests are about the slider's
promise: the expression stays the same and only the number of moving variables
changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from dimensions.evaluate import (
    MAX_DIMENSION,
    EvaluationError,
    frames,
    plan_variables,
    sample,
    slices,
)
from dimensions.parse import parse


# --- which variables move ----------------------------------------------


@pytest.mark.parametrize(
    "expression,dimension,expected",
    [
        ("x^2 + y^2 + z^2 + t^2", 1, ["x"]),
        ("x^2 + y^2 + z^2 + t^2", 2, ["x", "y"]),
        ("x^2 + y^2 + z^2 + t^2", 4, ["x", "y", "z", "t"]),
        ("x^2", 3, ["x", "y", "z"]),          # padded out to the requested dimension
        ("a + b", 2, ["a", "b"]),             # unfamiliar names still work
    ],
)
def test_which_variables_vary(expression, dimension, expected):
    varying, _ = plan_variables(parse(expression), dimension)
    assert [str(v) for v in varying] == expected


def test_padding_keeps_xyzt_order():
    """Regression: x^2+y^2+t padded to 4D once ordered [x, y, t, z], which made
    the animated fourth axis z instead of t."""
    varying, _ = plan_variables(parse("x^2 + y^2 + t"), 4)
    assert [str(v) for v in varying] == ["x", "y", "z", "t"]


def test_unused_variables_are_held_not_dropped():
    varying, held = plan_variables(parse("x + y + z"), 1)
    assert [str(v) for v in varying] == ["x"]
    assert [str(v) for v in held] == ["y", "z"]


@pytest.mark.parametrize("dimension", [0, -1, MAX_DIMENSION + 1])
def test_dimension_outside_the_slider_is_rejected(dimension):
    with pytest.raises(EvaluationError, match="Dimension must be"):
        plan_variables(parse("x"), dimension)


# --- sampling -----------------------------------------------------------


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_grid_has_one_axis_per_dimension(dimension):
    result = sample(parse("x^2+y^2+z^2+t^2"), dimension, bounds=(-2, 2), resolution=6)
    assert result.values.shape == (6,) * dimension


def test_values_are_correct():
    result = sample(parse("x + y"), 2, bounds=(0, 2), resolution=3)
    assert result.values[0, 0] == 0.0     # x=0, y=0
    assert result.values[-1, -1] == 4.0   # x=2, y=2
    assert result.values[-1, 0] == 2.0    # x=2, y=0


def test_holding_a_variable_gives_a_genuine_slice():
    """x^2+y^2 at dimension 1 must be x^2 with y pinned, not a different sum."""
    result = sample(parse("x^2 + y^2"), 1, bounds=(-2, 2), resolution=5)
    np.testing.assert_allclose(result.values, [4, 1, 0, 1, 4])
    assert result.held == {"y": 0.0}


def test_a_padded_dimension_is_flat_in_the_added_variable():
    result = sample(parse("x^2"), 2, bounds=(-2, 2), resolution=5)
    np.testing.assert_allclose(result.values[:, 0], result.values[:, -1])


def test_a_constant_still_fills_the_grid():
    result = sample(parse("7"), 2, bounds=(-1, 1), resolution=4)
    assert result.values.shape == (4, 4)
    assert np.unique(result.values).tolist() == [7.0]


# --- expressions that blow up -------------------------------------------


def test_undefined_points_are_counted_not_hidden():
    """A plot that silently drops half its domain is worse than one that says so."""
    result = sample(parse("sqrt(x)"), 1, bounds=(-4, 4), resolution=101)
    assert result.undefined_fraction == pytest.approx(0.5, abs=0.02)
    assert "undefined" in result.summary()


def test_division_by_zero_is_one_undefined_point_not_a_crash():
    result = sample(parse("1/x"), 1, bounds=(-2, 2), resolution=101)
    assert result.undefined_count == 1
    assert np.isfinite(result.finite).all()


def test_logs_of_negatives_do_not_leak_complex_numbers():
    result = sample(parse("log(x)"), 1, bounds=(-2, 2), resolution=101)
    assert result.values.dtype == np.float64
    assert result.undefined_fraction > 0.4


def test_value_range_ignores_undefined_points():
    low, high = sample(parse("sqrt(x)"), 1, bounds=(-4, 4), resolution=101).value_range
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(2.0)


def test_bad_bounds_are_rejected():
    with pytest.raises(EvaluationError, match="increasing"):
        sample(parse("x"), 1, bounds=(5, 5))
    with pytest.raises(EvaluationError, match="increasing"):
        sample(parse("x"), 1, bounds=(5, -5))


# --- 3D and 4D presentation ---------------------------------------------


def test_slices_cut_a_volume_into_planes():
    result = sample(parse("x^2+y^2+z^2"), 3, bounds=(-2, 2), resolution=9)
    planes = slices(result, count=3)
    assert [p["plane"].shape for p in planes] == [(9, 9)] * 3
    assert planes[0]["label"].startswith("z =")
    # The middle slice passes through the origin, so it reaches zero.
    assert np.nanmin(planes[1]["plane"]) == pytest.approx(0.0)


def test_slices_refuses_a_sample_of_the_wrong_dimension():
    with pytest.raises(EvaluationError, match="needs a 3D sample"):
        slices(sample(parse("x+y"), 2, resolution=4))


def test_frames_animate_over_the_fourth_variable():
    got = frames(parse("x^2 + y^2 + t"), count=3, bounds=(-2, 2), resolution=5)
    assert [f["label"].split(" =")[0] for f in got] == ["t", "t", "t"]
    # x^2+y^2 spans 0..8 over these bounds, so each frame is shifted by t.
    for frame in got:
        low, high = frame["sample"].value_range
        assert low == pytest.approx(frame["at"], abs=1e-9)
        assert high == pytest.approx(frame["at"] + 8, abs=1e-9)


def test_frames_produce_two_dimensional_surfaces():
    for frame in frames(parse("x*y*t"), count=2, bounds=(-1, 1), resolution=4):
        assert frame["sample"].dimension == 2
        assert frame["sample"].values.shape == (4, 4)

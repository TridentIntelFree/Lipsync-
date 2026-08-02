"""Dimensional formulas, checked against values that can be looked up.

Every expected value here is either a formula from a textbook or a count anyone
can verify by hand, so a regression shows up as a disagreement with known
mathematics rather than with a previous run of this code.
"""

from __future__ import annotations

import pytest
import sympy as sp

from dimensions.shapes import (
    SHAPES,
    ball_fills_cube,
    familiar_name,
    measures,
    table,
)


def exact(expr, expected: str) -> bool:
    return sp.simplify(sp.sympify(expr) - sp.sympify(expected)) == 0


# --- the ball -----------------------------------------------------------


@pytest.mark.parametrize(
    "dimension,expected",
    [
        (1, "2"),            # a segment of length 2r
        (2, "pi"),           # the circle
        (3, "4*pi/3"),       # the sphere
        (4, "pi**2/2"),      # the 4-ball
        (5, "8*pi**2/15"),
        (6, "pi**3/6"),
    ],
)
def test_ball_volume_matches_the_known_formulas(dimension, expected):
    assert exact(SHAPES["ball"]["volume"].at(dimension, 1), expected)


@pytest.mark.parametrize(
    "dimension,expected",
    [(1, "2"), (2, "2*pi"), (3, "4*pi"), (4, "2*pi**2")],
)
def test_sphere_surface_matches_the_known_formulas(dimension, expected):
    assert exact(SHAPES["ball"]["surface"].at(dimension, 1), expected)


def test_ball_volume_peaks_at_five_dimensions():
    """The result the charts exist to show: more dimensions is not more room."""
    volumes = [float(SHAPES["ball"]["volume"].at(d, 1)) for d in range(1, 16)]
    assert volumes.index(max(volumes)) == 4  # zero-based, so 5D
    assert volumes[14] < volumes[0]  # by 15D, smaller than a 1D segment


def test_ball_fills_less_and_less_of_its_box():
    assert exact(ball_fills_cube(1), "1")
    assert float(ball_fills_cube(2)) == pytest.approx(0.7853981, abs=1e-6)
    assert float(ball_fills_cube(3)) == pytest.approx(0.5235987, abs=1e-6)
    assert float(ball_fills_cube(10)) < 0.003


def test_ball_volume_is_defined_at_fractional_dimensions():
    """The gamma function does not require whole numbers, and neither do we."""
    half = float(SHAPES["ball"]["volume"].at(sp.Rational(5, 2), 1))
    assert float(SHAPES["ball"]["volume"].at(2, 1)) < half
    assert half < float(SHAPES["ball"]["volume"].at(3, 1))


# --- the cube -----------------------------------------------------------


@pytest.mark.parametrize(
    "dimension,corners,edges,squares",
    [
        (1, 2, 1, 0),
        (2, 4, 4, 1),
        (3, 8, 12, 6),      # the cube everyone knows
        (4, 16, 32, 24),    # the tesseract
        (5, 32, 80, 80),
    ],
)
def test_cube_face_counts(dimension, corners, edges, squares):
    values = SHAPES["cube"]
    assert values["vertices"].at(dimension) == corners
    assert values["edges"].at(dimension) == edges
    assert values["faces"].at(dimension) == squares


def test_cube_diagonal_grows_as_root_n():
    assert exact(SHAPES["cube"]["diagonal"].at(2, 1), "sqrt(2)")
    assert exact(SHAPES["cube"]["diagonal"].at(3, 1), "sqrt(3)")
    assert exact(SHAPES["cube"]["diagonal"].at(100, 1), "10")


def test_cube_surface_matches_the_shapes_we_can_count():
    assert SHAPES["cube"]["surface"].at(2, 1) == 4   # 4 sides of a square
    assert SHAPES["cube"]["surface"].at(3, 1) == 6   # 6 faces of a cube
    assert SHAPES["cube"]["surface"].at(4, 1) == 8   # 8 cells of a tesseract


# --- simplex and cross-polytope ----------------------------------------


def test_simplex_volumes():
    volume = SHAPES["simplex"]["volume"]
    assert exact(volume.at(2, 1), "sqrt(3)/4")    # equilateral triangle
    assert exact(volume.at(3, 1), "sqrt(2)/12")   # regular tetrahedron


def test_simplex_counts():
    """A simplex in n dimensions has n+1 corners, all mutually connected."""
    for dimension in range(1, 6):
        assert SHAPES["simplex"]["vertices"].at(dimension) == dimension + 1
        assert SHAPES["simplex"]["edges"].at(dimension) == sp.binomial(dimension + 1, 2)


def test_cross_polytope_volumes():
    volume = SHAPES["cross-polytope"]["volume"]
    assert exact(volume.at(3, 1), "4/3")   # the octahedron
    assert SHAPES["cross-polytope"]["vertices"].at(4) == 8


# --- presentation -------------------------------------------------------


def test_familiar_names_are_used_where_they_exist():
    assert familiar_name("ball", 2) == "circle"
    assert familiar_name("ball", 3) == "sphere"
    assert familiar_name("cube", 4) == "tesseract"
    assert familiar_name("simplex", 3) == "tetrahedron"
    assert "7" in familiar_name("cube", 7)  # no common name, still described


def test_table_gives_one_row_per_dimension_with_formula_and_number():
    rows = table("ball", range(1, 5))
    assert [row["dimension"] for row in rows] == [1, 2, 3, 4]
    assert rows[2]["name"] == "sphere"
    volume = rows[2]["values"]["volume"]
    assert exact(volume["exact"], "4*pi/3")
    assert volume["value"] == pytest.approx(4.18879, abs=1e-4)
    assert volume["formula"].free_symbols  # still symbolic in r


def test_table_tolerates_measures_undefined_in_low_dimensions():
    """A 1-cube has no square faces; that must not blow the table up."""
    rows = table("cube", [1, 2])
    assert rows[0]["values"]["faces"]["value"] == 0


def test_unknown_shape_is_rejected_by_name():
    with pytest.raises(KeyError, match="Unknown shape"):
        measures("dodecahedron")

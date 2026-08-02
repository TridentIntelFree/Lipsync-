"""Unit conversion and scaling across dimensional powers.

Every expected value is arithmetic anyone can check: 1 m is 100 cm, so 1 m² is
10,000 cm². The point of the module is that the factor gets raised to the
dimension, and these tests exist to catch it silently not being.
"""

from __future__ import annotations

import pytest
import sympy as sp

from dimensions.scaling import (
    UnknownUnit,
    conversion_table,
    convert,
    power_name,
    scaling_table,
    square_cube_note,
    unit_label,
)


# --- conversion ---------------------------------------------------------


@pytest.mark.parametrize(
    "dimension,expected",
    [(1, 100), (2, 10_000), (3, 1_000_000), (4, 100_000_000)],
)
def test_metres_to_centimetres_raises_the_factor_to_the_dimension(dimension, expected):
    assert convert(1, "m", "cm", dimension) == expected


def test_conversion_is_exact_not_floating_point():
    """An inch is exactly 2.54 cm, so a cubic inch is exactly 2.54**3 cm³."""
    exact = convert(1, "in", "cm", 3)
    assert exact == sp.Rational(254, 100) ** 3
    assert exact.is_Rational, "conversion drifted into floating point"
    assert float(exact) == pytest.approx(16.387064, abs=1e-9)


def test_converting_back_returns_the_original():
    for dimension in (1, 2, 3, 4):
        there = convert(7, "ft", "m", dimension)
        assert convert(there, "m", "ft", dimension) == 7


def test_same_unit_is_a_no_op_at_every_power():
    for dimension in (1, 2, 3, 4):
        assert convert(42, "km", "km", dimension) == 42


def test_a_mile_cubed_in_cubic_kilometres():
    assert float(convert(1, "mile", "km", 3)) == pytest.approx(4.168181825, abs=1e-6)


def test_unknown_units_are_named_in_the_error():
    with pytest.raises(UnknownUnit, match="furlong"):
        convert(1, "furlong", "m", 1)


def test_conversion_table_covers_each_dimension_with_a_factor():
    rows = conversion_table(2, "m", "cm", range(1, 5))
    assert [r["dimension"] for r in rows] == [1, 2, 3, 4]
    assert [r["quantity"] for r in rows] == ["length", "area", "volume", "hypervolume"]
    assert rows[1]["factor"] == 10_000
    assert rows[1]["value"] == 20_000  # 2 m² is 20,000 cm²
    assert rows[2]["from"] == "2 m³"


# --- scaling ------------------------------------------------------------


def test_doubling_every_length_gives_the_square_cube_law():
    assert [row.value for row in scaling_table(2)] == [2, 4, 8, 16]


def test_scaling_by_ten():
    assert [row.value for row in scaling_table(10)] == [10, 100, 1000, 10_000]


def test_shrinking_scales_the_same_way():
    rows = scaling_table(sp.Rational(1, 2))
    assert [row.factor for row in rows] == [
        sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 8), sp.Rational(1, 16)
    ]


def test_scaling_by_one_changes_nothing():
    assert all(row.value == 1 for row in scaling_table(1))


def test_square_cube_note_states_the_consequence():
    note = square_cube_note(2)
    assert "4x" in note and "8x" in note
    assert "collapse" in note


# --- labels -------------------------------------------------------------


def test_power_names():
    assert power_name(1) == "length"
    assert power_name(2) == "area"
    assert power_name(3) == "volume"
    assert power_name(4) == "hypervolume"
    assert "7" in power_name(7)


def test_unit_labels_use_superscripts():
    assert unit_label("cm", 1) == "cm"
    assert unit_label("cm", 2) == "cm²"
    assert unit_label("cm", 3) == "cm³"
    assert unit_label("cm", 4) == "cm⁴"
    assert unit_label("cm", 12) == "cm^12"

"""How a measurement changes when the dimension does.

Two related things live here.

**Powers of a unit.** A metre, a square metre and a cubic metre are the same unit
raised to 1, 2 and 3. Nothing stops the exponent going to 4, and the conversion
factor between two units follows the power: 1 m = 100 cm, so 1 m² = 10,000 cm²
and 1 m³ = 1,000,000 cm³. Getting this wrong by using the length factor on an
area is one of the most common unit mistakes there is.

**Scaling laws.** Doubling every length multiplies area by 4 and volume by 8 —
the square-cube law. It is why a scaled-up animal cannot support its own weight
and why small things cool faster than large ones. In n dimensions the factor is
simply k^n.
"""

from __future__ import annotations

import dataclasses

import sympy as sp

# Each unit as a length, expressed in metres. Dimensional powers are derived, so
# there is no table of square and cubic units to get out of step.
LENGTH_UNITS: dict[str, sp.Expr] = {
    "nm": sp.Rational(1, 10**9),
    "µm": sp.Rational(1, 10**6),
    "mm": sp.Rational(1, 1000),
    "cm": sp.Rational(1, 100),
    "m": sp.Integer(1),
    "km": sp.Integer(1000),
    "in": sp.Rational(254, 10000),
    "ft": sp.Rational(3048, 10000),
    "yd": sp.Rational(9144, 10000),
    "mile": sp.Rational(1609344, 1000),
}

# What the unit is called at each power, where English has a name for it.
POWER_NAMES = {
    0: "count",
    1: "length",
    2: "area",
    3: "volume",
    4: "hypervolume",
}


def power_name(dimension: int) -> str:
    return POWER_NAMES.get(int(dimension), f"{int(dimension)}-volume")


def unit_label(unit: str, dimension: int) -> str:
    """'cm', 'cm²', 'cm³', 'cm⁴' — superscripts where they exist."""
    superscripts = {2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"}
    d = int(dimension)
    if d == 1:
        return unit
    return f"{unit}{superscripts.get(d, f'^{d}')}"


class UnknownUnit(KeyError):
    """Raised when a unit is not in the table."""


def _factor(unit: str) -> sp.Expr:
    if unit not in LENGTH_UNITS:
        raise UnknownUnit(
            f"Unknown unit {unit!r}; known: {', '.join(LENGTH_UNITS)}"
        )
    return LENGTH_UNITS[unit]


def convert(value, from_unit: str, to_unit: str, dimension: int = 1) -> sp.Expr:
    """Convert a measurement between units at a given dimensional power.

    The factor is raised to the dimension, which is the whole point: converting
    an area with a length factor is off by that factor again.
    """
    ratio = _factor(from_unit) / _factor(to_unit)
    return sp.nsimplify(sp.sympify(value) * ratio ** sp.Integer(dimension))


def conversion_table(
    value, from_unit: str, to_unit: str, dimensions=range(1, 5)
) -> list[dict]:
    """One row per dimension, showing the factor and the converted value."""
    rows = []
    for d in dimensions:
        factor = (_factor(from_unit) / _factor(to_unit)) ** sp.Integer(d)
        converted = convert(value, from_unit, to_unit, d)
        rows.append(
            {
                "dimension": int(d),
                "quantity": power_name(d),
                "from": f"{value} {unit_label(from_unit, d)}",
                "to": f"{unit_label(to_unit, d)}",
                "factor": factor,
                "exact": converted,
                "value": float(converted),
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class ScalingRow:
    dimension: int
    quantity: str
    factor: sp.Expr

    @property
    def value(self) -> float:
        return float(self.factor)


def scaling_table(k, dimensions=range(1, 5)) -> list[ScalingRow]:
    """What multiplying every length by k does at each dimensional power.

    k=2 gives 2, 4, 8, 16 — the square-cube law and its continuation.
    """
    k = sp.sympify(k)
    return [
        ScalingRow(int(d), power_name(d), sp.nsimplify(k ** sp.Integer(d)))
        for d in dimensions
    ]


def square_cube_note(k) -> str:
    """Plain-language statement of why scaling up is not free."""
    k = sp.sympify(k)
    area, volume = k**2, k**3
    return (
        f"Multiply every length by {k}: surface area grows {area}x but volume "
        f"and weight grow {volume}x. Strength follows cross-section (area), so "
        f"the scaled-up version carries {sp.nsimplify(volume / area)}x more "
        "weight per unit of strength. This is why an insect enlarged to human "
        "size would collapse under itself."
    )

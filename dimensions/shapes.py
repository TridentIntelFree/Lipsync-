"""How shapes behave as the number of dimensions changes.

The point of this module is that familiar formulas are special cases of a single
one. A circle's area and a sphere's volume are the same expression evaluated at
n=2 and n=3, and there is nothing stopping you evaluating it at n=4 — or at
n=7.5, since the gamma function does not care about whole numbers.

Some of what falls out is genuinely surprising, and the charts exist to show it:

- **Hypersphere volume peaks and then collapses.** For a fixed radius of 1 the
  volume rises to a maximum in 5 dimensions and shrinks toward zero after. There
  is more room in a 5-dimensional ball than a 20-dimensional one.
- **A hypersphere stops filling its box.** In 2D a circle covers 79% of its
  square; in 3D a sphere covers 52% of its cube; by 10D it is under 0.3%. Nearly
  all of a high-dimensional cube sits in its corners.
- **The diagonal runs away.** A unit cube's diagonal is sqrt(n), so in 100
  dimensions two corners of a box with sides of one metre are ten metres apart.

Every formula is exact and expressed with sympy, so results stay symbolic until
something asks for a number.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import sympy as sp

# Symbols shared across the formula table. `n` is the dimension, `r` a radius or
# half-width, `s` a side length.
n, r, s = sp.symbols("n r s", positive=True)


@dataclasses.dataclass(frozen=True)
class Measure:
    """One quantity of a shape, as a formula in the dimension."""

    key: str
    name: str
    formula: Callable[[sp.Expr], sp.Expr]
    unit_power: Callable[[sp.Expr], sp.Expr] | None = None
    note: str = ""

    def at(self, dimension, size=1) -> sp.Expr:
        """Evaluate this measure in a given dimension, simplified."""
        expr = self.formula(sp.sympify(dimension))
        expr = expr.subs({r: sp.sympify(size), s: sp.sympify(size)})
        return sp.simplify(expr)

    def symbolic(self, dimension) -> sp.Expr:
        """The formula in a given dimension, with the size left as a symbol."""
        return sp.simplify(self.formula(sp.sympify(dimension)))


def _ball_volume(d):
    """pi^(d/2) / Gamma(d/2 + 1) * r^d — the n-ball, for any real d > 0."""
    return sp.pi ** (d / 2) / sp.gamma(d / 2 + 1) * r**d


def _sphere_surface(d):
    """The (d-1)-dimensional boundary of a d-ball."""
    return 2 * sp.pi ** (d / 2) / sp.gamma(d / 2) * r ** (d - 1)


def _cube_faces(d, k):
    """Count of k-dimensional faces on a d-cube: C(d, k) * 2^(d-k)."""
    return sp.binomial(d, k) * 2 ** (d - k)


def _simplex_volume(d):
    """Regular simplex of side s: the triangle and tetrahedron generalised."""
    return s**d / sp.factorial(d) * sp.sqrt((d + 1) / 2**d)


SHAPES: dict[str, dict[str, Measure]] = {
    "ball": {
        "volume": Measure(
            "volume",
            "Volume",
            _ball_volume,
            note="2r in 1D, the circle's area in 2D, the sphere's volume in 3D",
        ),
        "surface": Measure(
            "surface",
            "Surface",
            _sphere_surface,
            note="The circle's circumference in 2D, the sphere's area in 3D",
        ),
        "diameter": Measure("diameter", "Diameter", lambda d: 2 * r),
    },
    "cube": {
        "volume": Measure(
            "volume", "Volume", lambda d: s**d, note="Side length raised to the dimension"
        ),
        "surface": Measure(
            "surface", "Surface", lambda d: 2 * d * s ** (d - 1),
            note="4 sides of a square, 6 faces of a cube, 8 cells of a tesseract",
        ),
        "diagonal": Measure(
            "diagonal", "Long diagonal", lambda d: s * sp.sqrt(d),
            note="Grows without limit: sqrt(n) times the side",
        ),
        "vertices": Measure("vertices", "Corners", lambda d: _cube_faces(d, 0)),
        "edges": Measure("edges", "Edges", lambda d: _cube_faces(d, 1)),
        "faces": Measure("faces", "Square faces", lambda d: _cube_faces(d, 2)),
    },
    "simplex": {
        "volume": Measure(
            "volume", "Volume", _simplex_volume,
            note="The line, triangle and tetrahedron continued upward",
        ),
        "vertices": Measure("vertices", "Corners", lambda d: d + 1),
        "edges": Measure("edges", "Edges", lambda d: sp.binomial(d + 1, 2)),
    },
    "cross-polytope": {
        "volume": Measure(
            "volume", "Volume", lambda d: 2**d * r**d / sp.factorial(d),
            note="The square rotated 45 degrees, then the octahedron",
        ),
        "vertices": Measure("vertices", "Corners", lambda d: 2 * d),
    },
}

SHAPE_NAMES = {
    "ball": "Ball / sphere",
    "cube": "Cube / box",
    "simplex": "Simplex (triangle, tetrahedron, ...)",
    "cross-polytope": "Cross-polytope (diamond, octahedron, ...)",
}

# What each shape is called in the dimensions people already know.
FAMILIAR = {
    ("ball", 1): "line segment",
    ("ball", 2): "circle",
    ("ball", 3): "sphere",
    ("ball", 4): "glome (4-sphere)",
    ("cube", 1): "line segment",
    ("cube", 2): "square",
    ("cube", 3): "cube",
    ("cube", 4): "tesseract",
    ("simplex", 1): "line segment",
    ("simplex", 2): "triangle",
    ("simplex", 3): "tetrahedron",
    ("simplex", 4): "5-cell",
    ("cross-polytope", 1): "line segment",
    ("cross-polytope", 2): "square (rotated)",
    ("cross-polytope", 3): "octahedron",
    ("cross-polytope", 4): "16-cell",
}


def familiar_name(shape: str, dimension: int) -> str:
    """What this shape is normally called in this dimension, if it has a name."""
    return FAMILIAR.get((shape, int(dimension)), f"{dimension}-dimensional {shape}")


def measures(shape: str) -> dict[str, Measure]:
    if shape not in SHAPES:
        raise KeyError(f"Unknown shape {shape!r}; known: {', '.join(SHAPES)}")
    return SHAPES[shape]


def table(shape: str, dimensions=range(1, 5), size=1) -> list[dict]:
    """One row per dimension: the formula, the number, and the familiar name."""
    rows = []
    for d in dimensions:
        row = {"dimension": int(d), "name": familiar_name(shape, d), "values": {}}
        for key, measure in measures(shape).items():
            try:
                exact = measure.at(d, size)
                row["values"][key] = {
                    "formula": measure.symbolic(d),
                    "exact": exact,
                    "value": float(exact),
                }
            except (TypeError, ValueError):  # not defined here, e.g. 2-faces of a 1-cube
                row["values"][key] = None
        rows.append(row)
    return rows


def ball_fills_cube(dimension) -> sp.Expr:
    """Fraction of its bounding cube that a ball occupies.

    79% in 2D, 52% in 3D, and vanishing from there — the geometric statement of
    why high-dimensional spaces are mostly corners.
    """
    d = sp.sympify(dimension)
    return sp.simplify(_ball_volume(d).subs(r, sp.Rational(1, 2)) / 1**d)

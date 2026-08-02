"""Evaluate an expression while a chosen number of its variables vary.

This is what the dimension slider actually does. The expression never changes —
only how many of its variables are allowed to move:

    1  vary x                    a curve
    2  vary x and y              a surface
    3  vary x, y and z           a stack of slices through a volume
    4  vary x, y, z and t        that stack, as frames over t

Variables past the chosen dimension are pinned to a fixed value rather than
dropped, so ``x^2 + y^2`` at dimension 1 is a parabola in x with y held at zero —
a genuine slice of the surface, not a different expression.

Real expressions blow up: division by zero, logs of negatives, overflow. Those
points come back as NaN and are reported as a count, because a plot that silently
drops a third of its domain is worse than one that says so.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import sympy as sp

from .parse import Parsed

MAX_DIMENSION = 4

# Sensible defaults per dimension: a 3D volume at 200 points a side would be
# 8 million evaluations, so resolution falls as dimension rises.
DEFAULT_RESOLUTION = {1: 400, 2: 120, 3: 32, 4: 20}


class EvaluationError(ValueError):
    """Raised when an expression cannot be evaluated numerically."""


@dataclasses.dataclass
class Sampled:
    """An expression evaluated over a grid."""

    dimension: int
    varying: list[str]
    held: dict[str, float]
    axes: list[np.ndarray]
    values: np.ndarray
    bounds: tuple[float, float]

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    @property
    def undefined_count(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.values)))

    @property
    def undefined_fraction(self) -> float:
        return self.undefined_count / self.values.size if self.values.size else 0.0

    @property
    def value_range(self) -> tuple[float, float]:
        finite = self.finite
        if finite.size == 0:
            return (0.0, 0.0)
        return (float(finite.min()), float(finite.max()))

    def summary(self) -> str:
        low, high = self.value_range
        varying = ", ".join(self.varying) or "nothing"
        held = ", ".join(f"{k}={v:g}" for k, v in self.held.items())
        text = f"{self.dimension}D — varying {varying}"
        if held:
            text += f", holding {held}"
        text += f" — values from {low:.4g} to {high:.4g}"
        if self.undefined_count:
            text += f" ({self.undefined_fraction:.0%} undefined)"
        return text


def plan_variables(
    parsed: Parsed, dimension: int, extra_names: tuple[str, ...] = ("x", "y", "z", "t")
) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    """Decide which variables move and which are pinned.

    The expression's own variables come first in x, y, z, t order. If it has
    fewer than the requested dimension, unused axis names are added so the slider
    still means something — a 2D view of ``x^2`` is a surface that happens to be
    flat in y, which is the honest picture rather than an error.
    """
    if not 1 <= dimension <= MAX_DIMENSION:
        raise EvaluationError(f"Dimension must be 1 to {MAX_DIMENSION}, got {dimension}")

    available = list(parsed.variables)
    varying = available[:dimension]

    for name in extra_names:
        if len(varying) >= dimension:
            break
        symbol = sp.Symbol(name)
        if symbol not in varying:
            varying.append(symbol)

    # Re-sort into x, y, z, t order. Padding appends to the end, so an
    # expression like x^2 + y^2 + t would otherwise come out as [x, y, t, z] and
    # the fourth axis — the one that gets animated — would be z rather than t.
    order = {name: index for index, name in enumerate(extra_names)}
    varying.sort(key=lambda s: (order.get(s.name, len(order)), s.name))

    held = [v for v in available if v not in varying]
    return varying, held


def sample(
    parsed: Parsed,
    dimension: int,
    bounds: tuple[float, float] = (-5.0, 5.0),
    resolution: int | None = None,
    hold_at: float = 0.0,
) -> Sampled:
    """Evaluate ``parsed`` over a grid with ``dimension`` variables varying."""
    varying, held = plan_variables(parsed, dimension)
    low, high = float(bounds[0]), float(bounds[1])
    if not np.isfinite([low, high]).all() or low >= high:
        raise EvaluationError(f"Bounds must be finite and increasing, got {bounds}")

    steps = resolution or DEFAULT_RESOLUTION.get(dimension, 24)
    axis = np.linspace(low, high, steps)
    grids = np.meshgrid(*[axis] * dimension, indexing="ij")

    substitutions = {symbol: sp.Float(hold_at) for symbol in held}
    expression = parsed.expression.subs(substitutions) if substitutions else parsed.expression

    try:
        function = sp.lambdify(varying, expression, modules=["numpy"])
    except Exception as exc:  # noqa: BLE001 - lambdify raises many types
        raise EvaluationError(f"Could not turn {parsed.source!r} into a function: {exc}") from exc

    with warnings.catch_warnings():
        # Division by zero and invalid values are expected and handled below.
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
        try:
            raw = function(*grids)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(
                f"Could not evaluate {parsed.source!r} numerically: {exc}"
            ) from exc

    values = np.asarray(raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64)
    if values.ndim == 0:  # a constant expression broadcasts to nothing
        values = np.full(grids[0].shape, float(values.real if values.ndim else values))
    if np.iscomplexobj(values):
        # Complex results are outside what a real-valued plot can show; keep the
        # real part where the imaginary part is negligible, drop the rest.
        imaginary = np.abs(values.imag)
        values = np.where(imaginary < 1e-9, values.real, np.nan)
    values = np.asarray(values, dtype=np.float64)

    return Sampled(
        dimension=dimension,
        varying=[str(v) for v in varying],
        held={str(k): hold_at for k in held},
        axes=[axis] * dimension,
        values=values,
        bounds=(low, high),
    )


def slices(sampled: Sampled, count: int = 5) -> list[dict]:
    """Cut a 3D sample into 2D planes along its last axis.

    A volume cannot be drawn directly, so it is shown as planes through it —
    the same way a scan is read.
    """
    if sampled.dimension != 3:
        raise EvaluationError(f"slices() needs a 3D sample, got {sampled.dimension}D")

    axis = sampled.axes[-1]
    picks = np.linspace(0, len(axis) - 1, min(count, len(axis))).astype(int)
    return [
        {"at": float(axis[i]), "label": f"{sampled.varying[-1]} = {axis[i]:.2f}",
         "plane": sampled.values[:, :, i]}
        for i in picks
    ]


def frames(parsed: Parsed, count: int = 12, **kwargs) -> list[dict]:
    """A 4D view: the 3D picture repeated across values of the fourth variable.

    Time is the only fourth dimension anyone has intuition for, so the fourth
    variable is animated rather than drawn.
    """
    varying, _ = plan_variables(parsed, MAX_DIMENSION)
    fourth = varying[-1]
    bounds = kwargs.pop("bounds", (-5.0, 5.0))
    positions = np.linspace(bounds[0], bounds[1], count)

    out = []
    for position in positions:
        frozen = Parsed(
            expression=parsed.expression.subs({fourth: sp.Float(position)}),
            source=parsed.source,
            was_latex=parsed.was_latex,
        )
        out.append(
            {
                "at": float(position),
                "label": f"{fourth} = {position:.2f}",
                "sample": sample(frozen, 2, bounds=bounds, **kwargs),
            }
        )
    return out

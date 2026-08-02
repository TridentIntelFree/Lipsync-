"""Turn what someone types into something the maths can use.

People do not type sympy. They type ``x^2 + y^2``, ``2x``, ``sin x``, or paste
LaTeX out of a document. All of that should work without anyone learning a
syntax, so this module is deliberately permissive: it guesses whether the input
is LaTeX, applies the transformations that make ordinary notation mean what it
looks like, and reports failures in terms of what was typed rather than what the
parser expected.
"""

from __future__ import annotations

import dataclasses
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
)

# ^ means "to the power of" to everyone except a programmer, and juxtaposition
# means multiplication, so 2x and 2 sin(x) parse the way they read.
_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

# Signals that the input is LaTeX rather than ordinary typing.
_LATEX_HINTS = re.compile(r"\\[a-zA-Z]+|\\\\|[\^_]\{|\{\s*\\|\\left|\\right")

# Names that should stay symbols rather than becoming sympy functions.
_LOCALS = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    "inf": sp.oo,
    "infinity": sp.oo,
}


class ParseError(ValueError):
    """Raised when input cannot be read as a mathematical expression."""


@dataclasses.dataclass
class Parsed:
    """A parsed expression, with what was needed to read it."""

    expression: sp.Expr
    source: str
    was_latex: bool

    @property
    def variables(self) -> list[sp.Symbol]:
        """Free symbols, ordered so x, y, z, t come first and in that order."""
        preferred = ["x", "y", "z", "t", "w", "u", "v"]
        symbols = list(self.expression.free_symbols)
        return sorted(
            symbols,
            key=lambda s: (
                preferred.index(s.name) if s.name in preferred else len(preferred),
                s.name,
            ),
        )

    @property
    def latex(self) -> str:
        return sp.latex(self.expression)

    def __str__(self) -> str:
        return str(self.expression)


def looks_like_latex(text: str) -> bool:
    return bool(_LATEX_HINTS.search(text))


def _strip_wrappers(text: str) -> str:
    """Remove display markers and a leading 'y =' that people habitually type."""
    text = text.strip()
    for opener, closer in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if text.startswith(opener) and text.endswith(closer) and len(text) > len(opener) + len(closer):
            text = text[len(opener) : -len(closer)].strip()
    # "f(x) = x^2" and "y = x^2" both mean the right-hand side.
    match = re.match(r"^\s*(?:[a-zA-Z]\w*\s*(?:\([^)]*\))?)\s*=\s*(?![=<>])(.+)$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def parse(text: str) -> Parsed:
    """Read an expression from plain typing or LaTeX.

    Raises ParseError with the offending input quoted, since the message is shown
    to whoever typed it.
    """
    if text is None or not str(text).strip():
        raise ParseError("Nothing to read — type an expression such as x^2 + y^2.")

    original = str(text).strip()
    cleaned = _strip_wrappers(original)
    is_latex = looks_like_latex(cleaned)

    if is_latex:
        try:
            from sympy.parsing.latex import parse_latex

            expression = parse_latex(cleaned)
        except Exception as exc:  # noqa: BLE001 - many parser failure types
            raise ParseError(
                f"Could not read {original!r} as LaTeX ({type(exc).__name__}). "
                "Try typing it plainly instead, like x^2 + y^2."
            ) from exc
    else:
        try:
            expression = sp.parse_expr(
                cleaned, transformations=_TRANSFORMS, local_dict=dict(_LOCALS), evaluate=True
            )
        except Exception as exc:  # noqa: BLE001
            raise ParseError(
                f"Could not read {original!r} as an expression ({type(exc).__name__}). "
                "Check for unbalanced brackets or a stray symbol."
            ) from exc

    if expression is None:
        raise ParseError(f"Could not read {original!r}.")

    expression = sp.sympify(expression)
    if not isinstance(expression, sp.Basic):
        raise ParseError(f"{original!r} did not come out as an expression.")

    return Parsed(expression=expression, source=original, was_latex=is_latex)

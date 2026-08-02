"""Reading what people actually type.

The parser is the first thing anyone touches, so it is tested against how
expressions get written by hand rather than how sympy prefers them.
"""

from __future__ import annotations

import pytest
import sympy as sp

from dimensions.parse import ParseError, looks_like_latex, parse


def same(text: str, expected: str) -> bool:
    return sp.simplify(parse(text).expression - sp.sympify(expected)) == 0


# --- ordinary typing ----------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("x^2 + y^2", "x**2 + y**2"),   # ^ means power to everyone but a programmer
        ("x**2", "x**2"),               # ...and ** still works
        ("2x", "2*x"),                  # juxtaposition is multiplication
        ("2 sin(x)", "2*sin(x)"),
        ("sqrt(x)+1", "sqrt(x) + 1"),
        ("pi*r^2", "pi*r**2"),          # pi is the constant, not a symbol
        ("x^2+y^2+z^2+t^2", "x**2+y**2+z**2+t**2"),
    ],
)
def test_plain_typing(text, expected):
    assert same(text, expected)


@pytest.mark.parametrize(
    "text",
    ["y = x^2 + 1", "f(x) = x^2 + 1", "  y=x^2+1  "],
)
def test_a_leading_definition_is_stripped(text):
    """People write "y = ..." out of habit; the right-hand side is the expression."""
    assert same(text, "x**2 + 1")


@pytest.mark.parametrize("text", ["$x^2$", "$$x^2$$", r"\(x^2\)", r"\[x^2\]"])
def test_display_wrappers_are_stripped(text):
    assert same(text, "x**2")


def test_pi_and_e_are_constants_not_variables():
    assert parse("pi").expression == sp.pi
    assert parse("e").expression == sp.E
    assert parse("pi*r^2").variables == [sp.Symbol("r")]


# --- LaTeX --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\frac{x^2}{2}", "x**2/2"),
        (r"\sqrt{x^2+y^2}", "sqrt(x**2+y**2)"),
        (r"x^{2} + y^{2}", "x**2+y**2"),
        (r"\frac{1}{x}", "1/x"),
    ],
)
def test_latex(text, expected):
    assert same(text, expected)


def test_latex_is_detected_rather_than_declared():
    assert looks_like_latex(r"\frac{1}{2}")
    assert looks_like_latex("x^{2}")
    assert not looks_like_latex("x^2 + y^2")
    assert not looks_like_latex("2*x")

    assert parse(r"\frac{x}{2}").was_latex
    assert not parse("x/2").was_latex


# --- variables ----------------------------------------------------------


def test_variables_come_out_in_the_order_the_slider_wants():
    """x, y, z, t first and in that order, so dimension 1 means x, 2 means x and y."""
    assert [str(v) for v in parse("t + z + y + x").variables] == ["x", "y", "z", "t"]
    assert [str(v) for v in parse("z*x").variables] == ["x", "z"]


def test_unfamiliar_names_sort_after_the_usual_ones_alphabetically():
    assert [str(v) for v in parse("b + x + a").variables] == ["x", "a", "b"]


def test_a_constant_expression_has_no_variables():
    assert parse("2 + 3").variables == []
    assert parse("2 + 3").expression == 5


# --- failures -----------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_says_what_to_type(text):
    with pytest.raises(ParseError, match="type an expression"):
        parse(text)


@pytest.mark.parametrize("text", ["x^^2", "(((", "x +* y"])
def test_malformed_input_is_rejected_with_the_input_quoted(text):
    """The message is shown to whoever typed it, so it must name their input."""
    with pytest.raises(ParseError) as caught:
        parse(text)
    assert repr(text) in str(caught.value)


def test_broken_latex_suggests_typing_it_plainly():
    with pytest.raises(ParseError, match="plainly"):
        parse(r"\frac{x}{")


def test_parsed_round_trips_to_latex_for_display():
    parsed = parse("x^2/2")
    assert parsed.latex.replace(" ", "") in {r"\frac{x^{2}}{2}", r"x^{2}/2"}
    assert str(parsed) == "x**2/2"

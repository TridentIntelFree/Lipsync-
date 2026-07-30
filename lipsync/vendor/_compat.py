"""Replacements for the two ``distutils`` helpers the vendored code uses.

``distutils`` was removed from the standard library in Python 3.12. The vendored
espnet subset imports ``strtobool`` and ``LooseVersion`` from it at module level,
which would make this package Python 3.11-only. Both are small enough to supply
directly, which keeps the vendored tree working on current Python without adding
a dependency.
"""

from __future__ import annotations

import re

_TRUE = {"y", "yes", "t", "true", "on", "1"}
_FALSE = {"n", "no", "f", "false", "off", "0"}


def strtobool(value: str) -> int:
    """Convert a string to 1 or 0, matching ``distutils.util.strtobool``."""
    text = str(value).lower()
    if text in _TRUE:
        return 1
    if text in _FALSE:
        return 0
    raise ValueError(f"invalid truth value {value!r}")


class LooseVersion:
    """Enough of ``distutils.version.LooseVersion`` for version comparisons.

    Only used here to compare against torch's version, which looks like
    ``2.13.0+cu130``. Numeric components compare numerically; any local or
    pre-release suffix is ignored, which is what the comparisons in the
    vendored code actually want.
    """

    __slots__ = ("vstring", "version")

    def __init__(self, vstring: str):
        self.vstring = str(vstring)
        head = re.split(r"[+\-]", self.vstring, maxsplit=1)[0]
        parts: list[int] = []
        for chunk in head.split("."):
            match = re.match(r"\d+", chunk)
            if not match:
                break
            parts.append(int(match.group()))
        self.version = tuple(parts)

    def _key(self, other):
        other = other if isinstance(other, LooseVersion) else LooseVersion(other)
        length = max(len(self.version), len(other.version))
        pad = lambda v: v + (0,) * (length - len(v))  # noqa: E731
        return pad(self.version), pad(other.version)

    def __eq__(self, other):
        a, b = self._key(other)
        return a == b

    def __lt__(self, other):
        a, b = self._key(other)
        return a < b

    def __le__(self, other):
        a, b = self._key(other)
        return a <= b

    def __gt__(self, other):
        a, b = self._key(other)
        return a > b

    def __ge__(self, other):
        a, b = self._key(other)
        return a >= b

    def __hash__(self):
        return hash(self.version)

    def __repr__(self):
        return f"LooseVersion('{self.vstring}')"

    def __str__(self):
        return self.vstring

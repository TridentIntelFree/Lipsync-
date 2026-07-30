"""Word error rate, for checking a guess against a transcript somebody wrote.

This is the only honest measure of whether the model read anything. Output is
always fluent, so plausibility tells you nothing; agreement with a real
transcript tells you everything.
"""

from __future__ import annotations

import dataclasses
import re

_WORD = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words.

    Casing and punctuation are not things a lip reader can recover, so scoring
    them would understate the model rather than measure it.
    """
    return _WORD.findall(text.lower())


@dataclasses.dataclass(frozen=True)
class Score:
    """The result of comparing a hypothesis against a reference."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """Word error rate. Can exceed 1.0 when the guess invents extra words."""
        if self.reference_words == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.reference_words

    @property
    def accuracy(self) -> float:
        """Share of reference words recovered, floored at zero."""
        return max(0.0, 1.0 - self.wer)

    def summary(self) -> str:
        if self.reference_words == 0:
            return "no reference words to compare against"
        return (
            f"{self.wer:.0%} word error rate "
            f"({self.substitutions} wrong, {self.deletions} missed, "
            f"{self.insertions} invented, out of {self.reference_words} words)"
        )


def score(reference: str, hypothesis: str) -> Score:
    """Compare a hypothesis against a reference by Levenshtein alignment."""
    ref = normalise(reference)
    hyp = normalise(hypothesis)

    if not ref:
        return Score(0, 0, len(hyp), 0)
    if not hyp:
        return Score(0, len(ref), 0, len(ref))

    # distance[i][j] = (cost, substitutions, deletions, insertions) for the first
    # i reference words against the first j hypothesis words.
    row: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(len(hyp) + 1)]

    for i in range(1, len(ref) + 1):
        previous, row = row, [(i, 0, i, 0)] + [(0, 0, 0, 0)] * len(hyp)
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                row[j] = previous[j - 1]
                continue
            sub_cost, sub_s, sub_d, sub_i = previous[j - 1]
            del_cost, del_s, del_d, del_i = previous[j]
            ins_cost, ins_s, ins_d, ins_i = row[j - 1]
            best = min(sub_cost, del_cost, ins_cost)
            if best == sub_cost:
                row[j] = (best + 1, sub_s + 1, sub_d, sub_i)
            elif best == del_cost:
                row[j] = (best + 1, del_s, del_d + 1, del_i)
            else:
                row[j] = (best + 1, ins_s, ins_d, ins_i + 1)

    _, substitutions, deletions, insertions = row[len(hyp)]
    return Score(substitutions, deletions, insertions, len(ref))

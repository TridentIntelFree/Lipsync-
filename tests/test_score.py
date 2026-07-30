"""Word error rate — the only honest check on whether the model read anything."""

from __future__ import annotations

from lipsync.score import normalise, score


def test_identical_text_scores_zero():
    result = score("the quick brown fox", "the quick brown fox")
    assert result.errors == 0
    assert result.wer == 0.0
    assert result.accuracy == 1.0


def test_casing_and_punctuation_are_ignored():
    """Neither is recoverable from lips, so scoring them would understate the model."""
    assert score("Hello, world!", "hello world").errors == 0


def test_a_substitution_is_counted_once():
    result = score("the cat sat", "the bat sat")
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.wer == 1 / 3


def test_a_missing_word_is_a_deletion():
    result = score("the cat sat down", "the cat sat")
    assert (result.substitutions, result.deletions, result.insertions) == (0, 1, 0)


def test_an_invented_word_is_an_insertion():
    result = score("the cat sat", "the cat sat down")
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 1)


def test_completely_wrong_output_scores_one():
    assert score("alpha beta gamma", "delta epsilon zeta").wer == 1.0


def test_wer_can_exceed_one_when_the_model_rambles():
    """Invented words are errors too, so a long wrong guess scores above 100%."""
    result = score("yes", "absolutely not under any circumstances whatsoever")
    assert result.wer > 1.0
    assert result.accuracy == 0.0


def test_empty_hypothesis_deletes_everything():
    result = score("three word phrase", "")
    assert result.deletions == 3
    assert result.wer == 1.0


def test_empty_reference_has_nothing_to_measure():
    assert score("", "").wer == 0.0
    assert score("", "invented words").wer == 1.0


def test_homophone_confusions_are_counted_like_any_other_error():
    """p/b/m are visually identical, so this is the model's characteristic mistake."""
    result = score("pat the mat", "bat the pat")
    assert result.substitutions == 2
    assert result.reference_words == 3


def test_error_counts_add_up_to_the_total():
    result = score("one two three four five", "one three three four six seven")
    assert result.errors == result.substitutions + result.deletions + result.insertions


def test_summary_reports_the_breakdown():
    text = score("the cat sat down", "the bat sat").summary()
    assert "word error rate" in text
    assert "out of 4 words" in text


def test_normalise_splits_on_punctuation_and_keeps_apostrophes():
    assert normalise("Don't stop--now!") == ["don't", "stop", "now"]

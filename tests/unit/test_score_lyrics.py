# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for score.lyrics."""

from __future__ import annotations

from music21 import expressions, stream

from harmoniatextor.domain.models import Syllable, ThemeNote
from harmoniatextor.score.io import new_score
from harmoniatextor.score.lyrics import (
    attach_lyrics,
    clear_lyrics,
    invalid_ipa,
    lyric_issues,
    read_lyrics,
    strip_hidden_lyrics,
)
from harmoniatextor.score.streamops import ScoreEditor


def _score(notes: list[ThemeNote]) -> stream.Score:
    """Build a one-part score with a first measure of notes.

    Args:
        notes: Notes to write.

    Returns:
        The score.
    """
    score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"])
    ScoreEditor(score).write_line("soprano", 1, notes)
    return score


def _measure(part: stream.Part, number: int) -> stream.Measure:
    """Return a measure, asserting that it exists.

    Args:
        part: Part to inspect.
        number: One-based measure number.

    Returns:
        The measure.
    """
    measure = part.measure(number)
    assert measure is not None
    return measure


def _markers(measure: stream.Measure) -> list[str]:
    """Return the language markers of a measure.

    Args:
        measure: Measure to inspect.

    Returns:
        Marker texts.
    """
    return [
        str(item.content)
        for item in measure.getElementsByClass(expressions.TextExpression)
    ]


class TestAttach:
    """Attaching text and IPA to notes."""

    def test_attach_text_and_ipa(self) -> None:
        """Text and IPA are stored, and the IPA verse is hidden."""
        score = _score([ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        part = score.parts[0]
        assert attach_lyrics(
            part,
            1,
            [Syllable("love", "lʌv", "single", "en"), Syllable("you", "juː")],
        )
        notes = list(_measure(part, 1).notes)
        assert [lyric.text for lyric in notes[0].lyrics] == ["love", "lʌv"]
        assert notes[0].lyrics[0].identifier == "en"
        assert notes[0].lyrics[1].identifier == "ipa"
        assert notes[0].lyrics[1].style.hideObjectOnPrint is True

    def test_language_marker_only_on_change(self) -> None:
        """A language marker appears at the start and on language changes."""
        score = _score([ThemeNote("C5", 1.0)])
        editor = ScoreEditor(score)
        editor.write_line("soprano", 2, [ThemeNote("D5", 1.0)])
        editor.write_line("soprano", 3, [ThemeNote("E5", 1.0)])
        part = score.parts[0]
        assert attach_lyrics(part, 1, [Syllable("a", "a", language="en")])
        assert attach_lyrics(part, 2, [Syllable("b", "b", language="en")])
        assert attach_lyrics(part, 3, [Syllable("c", "c", language="zh")])
        assert _markers(_measure(part, 1)) == ["[en]"]
        assert _markers(_measure(part, 2)) == []
        assert _markers(_measure(part, 3)) == ["[zh]"]

    def test_count_mismatch(self) -> None:
        """The syllable count must equal the note count."""
        score = _score([ThemeNote("C5", 1.0)])
        assert not attach_lyrics(score.parts[0], 1, [Syllable("a", "a")] * 2)

    def test_missing_measure(self) -> None:
        """A missing measure cannot receive lyrics."""
        score = _score([ThemeNote("C5", 1.0)])
        assert not attach_lyrics(score.parts[0], 9, [Syllable("a", "a")])

    def test_empty_measure(self) -> None:
        """A measure without notes cannot receive lyrics."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        ScoreEditor(score).write_line("soprano", 2, [ThemeNote("C5", 1.0)])
        assert not attach_lyrics(score.parts[0], 1, [Syllable("a", "a")])


class TestReadClearStrip:
    """Reading, clearing and stripping lyrics."""

    def test_read(self) -> None:
        """Stored lyrics are read back in order."""
        score = _score([ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        attach_lyrics(
            score.parts[0],
            1,
            [Syllable("love", "lʌv", "single", "en"), Syllable("you", "juː")],
        )
        assert read_lyrics(score.parts[0]) == [
            (1, Syllable(text="love", ipa="lʌv", syllabic="single", language="en")),
            (1, Syllable(text="you", ipa="juː", syllabic="single", language="")),
        ]

    def test_read_melisma_syllabic(self) -> None:
        """Reading keeps a melisma's begin/end syllabic values."""
        score = _score([ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        attach_lyrics(
            score.parts[0],
            1,
            [Syllable("a", "a", "begin", "en"), Syllable("", "", "end", "en")],
        )
        read = read_lyrics(score.parts[0])
        assert read[0][1].syllabic == "begin"
        assert read[1][1].syllabic == "end"

    def test_clear_measure(self) -> None:
        """Clearing one measure leaves the other measures intact."""
        score = _score([ThemeNote("C5", 1.0)])
        editor = ScoreEditor(score)
        editor.write_line("soprano", 2, [ThemeNote("D5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "a", language="en")])
        attach_lyrics(score.parts[0], 2, [Syllable("b", "b", language="en")])
        assert clear_lyrics(score.parts[0], 1)
        assert read_lyrics(score.parts[0]) == [
            (2, Syllable(text="b", ipa="b", syllabic="single", language="en"))
        ]

    def test_clear_whole_part(self) -> None:
        """Clearing the whole part removes every lyric and marker."""
        score = _score([ThemeNote("C5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "a", language="en")])
        assert clear_lyrics(score.parts[0])
        assert read_lyrics(score.parts[0]) == []
        assert _markers(_measure(score.parts[0], 1)) == []

    def test_clear_empty(self) -> None:
        """Clearing a part without lyrics reports no change."""
        score = _score([ThemeNote("C5", 1.0)])
        assert not clear_lyrics(score.parts[0])

    def test_strip_hidden(self) -> None:
        """Stripping removes the IPA verse but keeps the text."""
        score = _score([ThemeNote("C5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("love", "lʌv")])
        strip_hidden_lyrics(score)
        assert read_lyrics(score.parts[0]) == [
            (1, Syllable(text="love", ipa="", syllabic="single", language=""))
        ]


class TestIssues:
    """Lyric validation."""

    def test_invalid_ipa(self) -> None:
        """IPA-like characters pass and others are reported."""
        assert invalid_ipa("lʌv") == []
        assert invalid_ipa("t͡ʃaː") == []
        assert invalid_ipa("l!v") == ["!"]

    def test_missing_lyric(self) -> None:
        """A note without a lyric is reported."""
        score = _score([ThemeNote("C5", 1.0), ThemeNote("D5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "a"), Syllable("b", "b")])
        first = next(iter(_measure(score.parts[0], 1).notes))
        first.lyrics = []
        assert (1, "missing_lyric", "") in lyric_issues(score.parts[0])

    def test_missing_ipa(self) -> None:
        """A sung syllable without IPA is reported."""
        score = _score([ThemeNote("C5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "", "single", "en")])
        assert (1, "missing_ipa", "a") in lyric_issues(score.parts[0])

    def test_melisma_needs_no_ipa(self) -> None:
        """A melisma continuation may omit the IPA."""
        score = _score([ThemeNote("C5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "", "middle", "en")])
        assert lyric_issues(score.parts[0]) == []

    def test_invalid_ipa_issue(self) -> None:
        """Unsupported IPA characters are reported."""
        score = _score([ThemeNote("C5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "a!", "single", "en")])
        assert (1, "invalid_ipa", "a!") in lyric_issues(score.parts[0])

    def test_missing_language(self) -> None:
        """A vocal part without a language tag is reported."""
        score = _score([ThemeNote("C5", 1.0)])
        attach_lyrics(score.parts[0], 1, [Syllable("a", "a")])
        assert (1, "missing_language", "") in lyric_issues(score.parts[0])

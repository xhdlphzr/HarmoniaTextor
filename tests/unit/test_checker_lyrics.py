# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Unit tests for the sung-lyrics check."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.checker.lyrics import lyric_violations
from harmoniatextor.domain.enums import Severity
from harmoniatextor.domain.models import Syllable, ThemeNote
from harmoniatextor.score.io import new_score
from harmoniatextor.score.lyrics import attach_lyrics
from harmoniatextor.score.streamops import ScoreEditor


def _score() -> stream.Score:
    """Build a one-part score with a sung note.

    Returns:
        The score.
    """
    score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"])
    ScoreEditor(score).write_line("soprano", 1, [ThemeNote("C5", 1.0)])
    return score


class TestLyricViolations:
    """The lyric check."""

    def test_clean(self) -> None:
        """Valid lyrics produce no violations."""
        score = _score()
        attach_lyrics(score.parts[0], 1, [Syllable("love", "lʌv", language="en")])
        assert lyric_violations(score, {"soprano": "en"}) == []

    def test_missing_part_is_skipped(self) -> None:
        """A marked voice without a part is ignored."""
        score = _score()
        assert lyric_violations(score, {"alto": "en"}) == []

    def test_reports_problems(self) -> None:
        """Problems become error violations with their kind and voice."""
        score = _score()
        attach_lyrics(score.parts[0], 1, [Syllable("love", "l!v")])
        violations = lyric_violations(score, {"soprano": "en"})
        assert {item.kind for item in violations} == {
            "invalid_ipa",
            "missing_language",
        }
        assert all(item.rule_id == "lyrics" for item in violations)
        assert all(item.severity is Severity.ERROR for item in violations)
        assert all(item.voice_a == "soprano" for item in violations)
        assert all(item.measure == 1 for item in violations)

# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for score.streamops."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.analysis import (
    voice_events,
)
from harmoniatextor.score.io import (
    new_part,
)
from harmoniatextor.score.streamops import ScoreEditor

_DEFAULT_BAR_LENGTH = 4.0


_TWO_EVENTS = 2


_TEMPO = 90


class TestScoreEditor:
    """Editing operations."""

    def test_bar_length_default(self) -> None:
        """A score without a time signature defaults to four quarters."""
        score = stream.Score()
        score.insert(0.0, new_part("soprano"))
        assert ScoreEditor(score).bar_length() == _DEFAULT_BAR_LENGTH

    def test_get_part_missing(self, score4: stream.Score) -> None:
        """Missing parts return None unless created."""
        editor = ScoreEditor(score4)
        assert editor.get_part("flute") is None
        assert editor.get_part("flute", create=True) is not None

    def test_remove_part(self, score4: stream.Score) -> None:
        """A voice can be removed; a missing voice is a no-op."""
        editor = ScoreEditor(score4)
        assert editor.remove_part("soprano")
        assert not editor.remove_part("soprano")

    def test_annotate_missing_target(self, score4: stream.Score) -> None:
        """Annotating a missing voice or measure is a no-op."""
        editor = ScoreEditor(score4)
        assert not editor.annotate("ghost", 1, "dynamic", "f")
        assert not editor.annotate("soprano", 99, "dynamic", "f")

    def test_set_tempo_adds_marking(self) -> None:
        """A part without a tempo marking gets one."""
        score = stream.Score()
        score.insert(0.0, new_part("flute"))
        ScoreEditor(score).set_tempo(_TEMPO)
        marks = list(score.recurse().getElementsByClass("MetronomeMark"))
        assert marks
        assert int(marks[0].number) == _TEMPO

    def test_write_across_measures(self, score4: stream.Score) -> None:
        """Writing a long line creates measures as needed."""
        editor = ScoreEditor(score4)
        notes = [ThemeNote("C5", 4.0) for _ in range(3)]
        editor.write_line("soprano", 1, notes)
        assert editor.read_line("soprano", 1, 3)[0].pitch == "C5"

    def test_replace_note(self, score4: stream.Score) -> None:
        """Placing a note at an occupied offset replaces it."""
        editor = ScoreEditor(score4)
        editor.place_note("soprano", 1, 0.0, "C5", 1.0)
        editor.place_note("soprano", 1, 0.0, "D5", 1.0)
        assert editor.read_line("soprano", 1, 1)[0].pitch == "D5"

    def test_place_chord_and_clear(self, score4: stream.Score) -> None:
        """Chords can be placed and ranges cleared."""
        editor = ScoreEditor(score4)
        editor.place_chord("alto", 1, 0.0, ["E4", "G4"], 2.0)
        editor.clear_measure_range("alto", 1, 1)
        assert editor.read_line("alto", 1, 1) == []

    def test_map_notes(self, score4: stream.Score) -> None:
        """map_notes rewrites pitches in range."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        editor.map_notes("soprano", 1, 1, lambda _offset, _pitch, _duration: "G5")
        assert editor.read_line("soprano", 1, 1)[0].pitch == "G5"

    def test_map_notes_missing_voice(self, score4: stream.Score) -> None:
        """map_notes on a missing voice is a no-op."""
        ScoreEditor(score4).map_notes("flute", 1, 1, lambda _o, p, _d: p)

    def test_map_notes_skips_other_measures(self, score4: stream.Score) -> None:
        """map_notes ignores measures outside the range."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 4.0)])
        editor.write_line("soprano", 2, [ThemeNote("D5", 4.0)])
        editor.map_notes("soprano", 1, 1, lambda _o, _p, _d: "G5")
        assert editor.read_line("soprano", 2, 2)[0].pitch == "D5"

    def test_read_line_skips_other_measures(self, score4: stream.Score) -> None:
        """read_line ignores measures outside the range."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 4.0)])
        editor.write_line("soprano", 2, [ThemeNote("D5", 4.0)])
        assert len(editor.read_line("soprano", 1, 1)) == 1

    def test_read_line_missing_voice(self, score4: stream.Score) -> None:
        """Reading a missing voice returns an empty list."""
        assert ScoreEditor(score4).read_line("flute", 1, 1) == []

    def test_place_chord_replaces(self, score4: stream.Score) -> None:
        """Placing a chord at an occupied offset replaces it."""
        editor = ScoreEditor(score4)
        editor.place_chord("alto", 1, 0.0, ["E4", "G4"], 2.0)
        editor.place_chord("alto", 1, 0.0, ["F4", "A4"], 2.0)
        events = voice_events(score4)["alto"]
        assert len(events) == _TWO_EVENTS
        assert {event.pitch for event in events} == {"F4", "A4"}

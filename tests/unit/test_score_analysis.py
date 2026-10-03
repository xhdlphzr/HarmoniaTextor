# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for score.analysis."""

from __future__ import annotations

from music21 import chord, stream
from music21 import note as m21note

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.analysis import (
    first_melody,
    measure_count,
    measure_of,
    onsets,
    voice_events,
)
from harmoniatextor.score.io import (
    new_part,
    new_score,
)
from harmoniatextor.score.streamops import ScoreEditor

_DEFAULT_BAR_LENGTH = 4.0


_TWO_EVENTS = 2


_SECOND_MEASURE = 2


class TestAnalysis:
    """Flattening helpers."""

    def test_measure_of(self, score4: stream.Score) -> None:
        """Offsets map to measures."""
        assert measure_of(score4, 0.0) == 1
        assert measure_of(score4, _DEFAULT_BAR_LENGTH) == _SECOND_MEASURE

    def test_voice_events_with_chord(self, score4: stream.Score) -> None:
        """Chord pitches expand into separate events."""
        editor = ScoreEditor(score4)
        editor.place_chord("alto", 1, 0.0, ["E4", "G4"], 2.0)
        events = voice_events(score4)["alto"]
        assert len(events) == _TWO_EVENTS

    def test_onsets(self, score4: stream.Score) -> None:
        """Onsets are the unique attack offsets."""
        editor = ScoreEditor(score4)
        editor.place_note("soprano", 1, 0.0, "C5", 1.0)
        editor.place_note("bass", 1, 1.0, "C3", 1.0)
        assert onsets(score4) == [0.0, 1.0]

    def test_first_melody_and_count(self, score4: stream.Score) -> None:
        """The first melody and measure count are found."""
        editor = ScoreEditor(score4)
        editor.write_line("soprano", 1, [ThemeNote("C5", 4.0)])
        editor.write_line("soprano", 2, [ThemeNote("D5", 4.0)])
        voice, notes = first_melody(score4)
        assert voice == "soprano"
        assert notes[0].pitch == "C5"
        assert measure_count(score4) == _SECOND_MEASURE

    def test_first_melody_empty(self) -> None:
        """An empty score yields no melody."""
        score = stream.Score()
        score.insert(0.0, new_part("soprano"))
        assert first_melody(score) == ("", [])

    def test_voice_events_ignores_rests(self, score4: stream.Score) -> None:
        """Rests do not produce events."""
        editor = ScoreEditor(score4)
        editor.place_note("soprano", 1, 0.0, "C5", 1.0)
        assert len(voice_events(score4)["soprano"]) == 1

    def test_chord_object_detected(self, score4: stream.Score) -> None:
        """A Chord element is recognised."""
        measure = score4.parts[0].measure(1)
        assert measure is not None
        measure.insert(0.0, chord.Chord(["C5", "E5"]))
        assert len(voice_events(score4)["soprano"]) == _TWO_EVENTS

    def test_note_without_octave(self) -> None:
        """Notes are read with octaves."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"]
        )
        measure = score.parts[0].measure(1)
        assert measure is not None
        measure.insert(0.0, m21note.Note("C5"))
        assert voice_events(score)["soprano"][0].pitch == "C5"

# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Tests for the music21 score bridge."""

from __future__ import annotations

import pytest
from music21 import chord, stream
from music21 import note as m21note

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.analysis import (
    first_melody,
    measure_count,
    measure_of,
    onsets,
    voice_events,
    voice_order,
)
from harmoniatextor.score.io import (
    clone_score,
    from_musicxml,
    instrument_for_voice,
    make_instrument,
    new_part,
    new_score,
    to_musicxml,
)
from harmoniatextor.score.merge import _copy_instrument, merge_scores
from harmoniatextor.score.streamops import ScoreEditor

_DEFAULT_BAR_LENGTH = 4.0
_TWO_EVENTS = 2
_SECOND_MEASURE = 2
_TEMPO = 90


class TestIO:
    """Import/export and construction."""

    def test_new_part_names(self) -> None:
        """Parts carry their voice name."""
        part = new_part("alto")
        assert part.partName == "alto"
        assert part.id == "alto"

    def test_roundtrip(self) -> None:
        """MusicXML survives a round trip."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=90, voices=["soprano", "bass"])
        editor = ScoreEditor(score)
        editor.write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        restored = from_musicxml(to_musicxml(score))
        assert voice_order(restored) == ["soprano", "bass"]

    def test_clone_is_independent(self) -> None:
        """Cloning produces an independent score."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=90, voices=["soprano"])
        clone = clone_score(score)
        clone.parts[0].partName = "changed"
        assert score.parts[0].partName == "soprano"

    def test_from_part_wraps_in_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare part is wrapped into a score."""
        part = new_part("soprano")
        monkeypatch.setattr("harmoniatextor.score.io.converter.parseData", lambda *_, **__: part)
        assert isinstance(from_musicxml("<xml/>"), stream.Score)

    def test_from_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-score parses raise a value error."""
        monkeypatch.setattr(
            "harmoniatextor.score.io.converter.parseData", lambda *_, **__: object()
        )
        with pytest.raises(ValueError, match="did not parse"):
            from_musicxml("<xml/>")


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
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["soprano"])
        measure = score.parts[0].measure(1)
        assert measure is not None
        measure.insert(0.0, m21note.Note("C5"))
        assert voice_events(score)["soprano"][0].pitch == "C5"


class TestInstruments:
    """Instrument assignment."""

    def test_instrument_for_voice(self) -> None:
        """Voice slots map to sensible default instruments."""
        assert instrument_for_voice("violin2") == "Violin"
        assert instrument_for_voice("cello") == "Violoncello"
        assert instrument_for_voice("violin9") == "Violin"
        assert instrument_for_voice("kazoo") == "Piano"

    def test_new_part_with_instrument(self) -> None:
        """An explicit instrument is attached to the part."""
        part = new_part("lead", "Flute")
        names = [item.classes[0] for item in part.getElementsByClass("Instrument")]
        assert "Flute" in names

    def test_make_instrument_unknown(self) -> None:
        """An unknown instrument name falls back to piano."""
        assert make_instrument("DefinitelyNotAnInstrument").classes[0] == "Piano"

    def test_make_instrument_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A resolver that returns None falls back to piano."""
        monkeypatch.setattr("harmoniatextor.score.io.instrument.fromString", lambda _name: None)
        assert make_instrument("x").classes[0] == "Piano"

    def test_make_instrument_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raising resolver falls back to piano."""

        def boom(_name: str) -> object:
            raise ValueError("no")

        monkeypatch.setattr("harmoniatextor.score.io.instrument.fromString", boom)
        assert make_instrument("x").classes[0] == "Piano"

    def test_set_instrument(self, score4: stream.Score) -> None:
        """The editor can retarget a voice's instrument."""
        editor = ScoreEditor(score4)
        editor.set_instrument("soprano", "Flute")
        part = editor.get_part("soprano")
        assert part is not None
        assert any(item.classes[0] == "Flute" for item in part.getElementsByClass("Instrument"))


class TestFromMusicXmlInstruments:
    """Parsed scores always carry an instrument per part."""

    def test_assigns_missing_instruments(self) -> None:
        """Parts without an instrument get a voice-based default."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin1", "piano"])
        ScoreEditor(score).write_line("violin1", 1, [ThemeNote("G4", 1.0)])
        for part in score.parts:
            for item in list(part.getElementsByClass("Instrument")):
                part.remove(item)
        parsed = from_musicxml(to_musicxml(score))
        names = {
            str(part.partName): [
                type(item).__name__ for item in part.getElementsByClass("Instrument")
            ]
            for part in parsed.parts
        }
        assert names["violin1"] == ["Violin"]
        assert names["piano"] == ["Piano"]


class TestMergeMovements:
    """Concatenating independently composed movement scores."""

    def test_merge_two_movements(self) -> None:
        """Movements are appended measure by measure per voice."""
        first = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin", "piano"])
        ScoreEditor(first).write_line("violin", 1, [ThemeNote("C5", 1.0)] * 4)
        second = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin", "piano"])
        ScoreEditor(second).write_line("violin", 1, [ThemeNote("D5", 1.0)] * 4)
        merged = from_musicxml(merge_scores([to_musicxml(first), to_musicxml(second)]))
        assert [str(part.id or part.partName) for part in merged.parts] == ["violin", "piano"]
        assert measure_count(merged) == _SECOND_MEASURE
        notes = [item.nameWithOctave for item in merged.parts[0].recurse().notes]
        assert notes[:4] == ["C5"] * 4
        assert notes[4:] == ["D5"] * 4

    def test_merge_aligns_uneven_parts(self) -> None:
        """Parts with fewer measures are padded so every part stays aligned."""
        first = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin", "piano"])
        ScoreEditor(first).write_line("violin", 1, [ThemeNote("C5", 1.0)] * 8)
        merged = from_musicxml(merge_scores([to_musicxml(first)]))
        counts = {
            str(part.id or part.partName): len(list(part.getElementsByClass(stream.Measure)))
            for part in merged.parts
        }
        assert counts == {"violin": 2, "piano": 2}

    def test_copy_instrument_without_source(self) -> None:
        """A source part without an instrument leaves the target untouched."""
        target = new_part("x")
        _copy_instrument(stream.Part(), target)  # type: ignore[no-untyped-call]  # music21
        assert target.getElementsByClass("Instrument")

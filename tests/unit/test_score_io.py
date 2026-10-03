# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for score.io."""

from __future__ import annotations

import pytest
from music21 import stream

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.analysis import (
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
from harmoniatextor.score.streamops import ScoreEditor


class TestIO:
    """Import/export and construction."""

    def test_new_part_names(self) -> None:
        """Parts carry their voice name."""
        part = new_part("alto")
        assert part.partName == "alto"
        assert part.id == "alto"

    def test_roundtrip(self) -> None:
        """MusicXML survives a round trip."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=90, voices=["soprano", "bass"]
        )
        editor = ScoreEditor(score)
        editor.write_line("soprano", 1, [ThemeNote("C5", 1.0)])
        restored = from_musicxml(to_musicxml(score))
        assert voice_order(restored) == ["soprano", "bass"]

    def test_clone_is_independent(self) -> None:
        """Cloning produces an independent score."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=90, voices=["soprano"]
        )
        clone = clone_score(score)
        clone.parts[0].partName = "changed"
        assert score.parts[0].partName == "soprano"

    def test_from_part_wraps_in_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare part is wrapped into a score."""
        part = new_part("soprano")
        monkeypatch.setattr(
            "harmoniatextor.score.io.converter.parseData", lambda *_, **__: part
        )
        assert isinstance(from_musicxml("<xml/>"), stream.Score)

    def test_from_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-score parses raise a value error."""
        monkeypatch.setattr(
            "harmoniatextor.score.io.converter.parseData", lambda *_, **__: object()
        )
        with pytest.raises(ValueError, match="did not parse"):
            from_musicxml("<xml/>")


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
        monkeypatch.setattr(
            "harmoniatextor.score.io.instrument.fromString", lambda _name: None
        )
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
        assert any(
            item.classes[0] == "Flute" for item in part.getElementsByClass("Instrument")
        )


class TestFromMusicXmlInstruments:
    """Parsed scores always carry an instrument per part."""

    def test_assigns_missing_instruments(self) -> None:
        """Parts without an instrument get a voice-based default."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=84, voices=["violin1", "piano"]
        )
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

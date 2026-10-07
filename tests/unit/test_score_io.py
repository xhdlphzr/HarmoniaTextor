# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for score.io."""

from __future__ import annotations

import pytest
from music21 import instrument, stream

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.analysis import (
    voice_order,
)
from harmoniatextor.score.io import (
    assign_clefs,
    clone_score,
    from_musicxml,
    instrument_for_voice,
    make_instrument,
    new_part,
    new_score,
    to_musicxml,
)
from harmoniatextor.score.streamops import ScoreEditor


def _clef_names(part: stream.Part) -> list[str]:
    """Return the class names of a part's clefs.

    Args:
        part: Part to inspect.

    Returns:
        The clef class names in order.
    """
    return [type(item).__name__ for item in part.recurse().getElementsByClass("Clef")]


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
        """A bare part is wrapped into a score.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        part = new_part("soprano")
        monkeypatch.setattr(
            "harmoniatextor.score.io.converter.parseData", lambda *_, **__: part
        )
        assert isinstance(from_musicxml("<xml/>"), stream.Score)

    def test_from_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-score parses raise a value error.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
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
        """A resolver that returns None falls back to piano.

        Args:
            monkeypatch: The pytest monkeypatch fixture.
        """
        monkeypatch.setattr(
            "harmoniatextor.score.io.instrument.fromString", lambda _name: None
        )
        assert make_instrument("x").classes[0] == "Piano"

    def test_make_instrument_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raising resolver falls back to piano.

        Args:
            monkeypatch: The pytest monkeypatch fixture.

        Raises:
            ValueError: When the operation cannot proceed.
        """

        def boom(_name: str) -> object:
            raise ValueError("no")

        monkeypatch.setattr("harmoniatextor.score.io.instrument.fromString", boom)
        assert make_instrument("x").classes[0] == "Piano"

    def test_set_instrument(self, score4: stream.Score) -> None:
        """The editor can retarget a voice's instrument.

        Args:
            score4: An empty four-voice score.
        """
        editor = ScoreEditor(score4)
        editor.set_instrument("soprano", "Flute")
        part = editor.get_part("soprano")
        assert part is not None
        assert any(
            item.classes[0] == "Flute" for item in part.getElementsByClass("Instrument")
        )


class TestClefs:
    """Piano hand clef assignment."""

    def test_lh_bass_rh_treble(self) -> None:
        """Named piano hands get treble/bass clefs."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["rh", "lh"]
        )
        editor = ScoreEditor(score)
        editor.write_line("rh", 1, [ThemeNote("C5", 1.0)])
        editor.write_line("lh", 1, [ThemeNote("C3", 1.0)])
        to_musicxml(score)
        assert _clef_names(score.parts[0]) == ["TrebleClef"]
        assert _clef_names(score.parts[1]) == ["BassClef"]

    def test_unnamed_low_piano_is_bass(self) -> None:
        """A low unnamed piano part falls back to a bass clef."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["piano2"]
        )
        ScoreEditor(score).write_line("piano2", 1, [ThemeNote("C3", 1.0)])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["BassClef"]

    def test_unnamed_high_piano_is_treble(self) -> None:
        """A high unnamed piano part falls back to a treble clef."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["piano1"]
        )
        ScoreEditor(score).write_line("piano1", 1, [ThemeNote("C5", 1.0)])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["TrebleClef"]

    def test_even_median_is_bass(self) -> None:
        """An even note count uses the averaged median."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["piano"])
        ScoreEditor(score).write_line(
            "piano", 1, [ThemeNote("C3", 1.0), ThemeNote("E3", 1.0)]
        )
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["BassClef"]

    def test_empty_piano_is_treble(self) -> None:
        """A piano part without notes defaults to a treble clef."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["piano"])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["TrebleClef"]

    def test_non_piano_untouched(self) -> None:
        """Non-piano parts keep music21's default (no explicit clef)."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["violin"]
        )
        ScoreEditor(score).write_line("violin", 1, [ThemeNote("C3", 1.0)])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == []

    def test_instrument_name_piano(self) -> None:
        """A generic instrument named Piano counts as a piano."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["lead"])
        part = score.parts[0]
        for item in list(part.getElementsByClass("Instrument")):
            part.remove(item)
        generic = instrument.Instrument()
        generic.instrumentName = "Piano"
        part.insert(0.0, generic)
        assign_clefs(score)
        assert _clef_names(part) == ["TrebleClef"]

    def test_part_without_measure(self) -> None:
        """A measure is created when a piano part has none."""
        part = new_part("lh", "Piano")
        score = stream.Score()
        score.insert(0.0, part)
        assign_clefs(score)
        assert _clef_names(part) == ["BassClef"]

    def test_existing_clef_replaced(self) -> None:
        """A previous clef is replaced on the next assignment."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["lh"])
        assign_clefs(score)
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["BassClef"]

    def test_empty_id_is_ignored(self) -> None:
        """A part with an empty id still resolves via its name."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["lh"])
        score.parts[0].id = ""
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["BassClef"]

    @pytest.mark.parametrize(
        "voice",
        [
            "cello",
            "contrabass",
            "bassoon",
            "contrabassoon",
            "trombone",
            "bass trombone",
            "tuba",
            "euphonium",
            "baritone",
            "timpani",
            "bass",
        ],
    )
    def test_low_instruments_use_bass(self, voice: str) -> None:
        """Low instruments are notated in a bass clef.

        Args:
            voice: The instrument's voice slot.
        """
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=[voice])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["BassClef"]

    def test_viola_uses_alto(self) -> None:
        """The viola is notated in an alto clef."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["viola"])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["AltoClef"]

    def test_harp_left_hand_uses_bass(self) -> None:
        """A harp left-hand voice takes a bass clef."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["harp", "left hand"]
        )
        assign_clefs(score)
        assert _clef_names(score.parts[1]) == ["BassClef"]

    def test_organ_pedal_uses_bass(self) -> None:
        """An organ pedal voice takes a bass clef."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["organ pedal"]
        )
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == ["BassClef"]

    def test_treble_instrument_untouched(self) -> None:
        """A treble instrument keeps music21's default."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["flute"])
        assign_clefs(score)
        assert _clef_names(score.parts[0]) == []

    def test_serialised_bass_clef(self) -> None:
        """The bass clef reaches the exported MusicXML."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["lh"])
        ScoreEditor(score).write_line("lh", 1, [ThemeNote("C3", 1.0)])
        assert "<sign>F</sign>" in to_musicxml(score)

    def test_serialised_alto_clef(self) -> None:
        """The alto clef reaches the exported MusicXML."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["viola"])
        assert "<sign>C</sign>" in to_musicxml(score)


class TestStaffGroups:
    """Same-instrument staff grouping."""

    def test_piano_hands_braced(self) -> None:
        """Piano hands are joined by a brace."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["rh", "lh"]
        )
        xml = to_musicxml(score)
        assert xml.count("<group-symbol>brace</group-symbol>") == 1

    def test_organ_manuals_and_pedal_braced(self) -> None:
        """An organ's manuals and pedal share a brace."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["organ", "pedal"]
        )
        assert "<group-symbol>brace</group-symbol>" in to_musicxml(score)

    def test_strings_section_bracketed(self) -> None:
        """A string section is joined by a bracket, not a brace."""
        score = new_score(
            key="C",
            time_signature="4/4",
            tempo_bpm=80,
            voices=["violin1", "violin2", "viola", "cello"],
        )
        xml = to_musicxml(score)
        assert "<group-symbol>bracket</group-symbol>" in xml
        assert "<group-symbol>brace</group-symbol>" not in xml

    def test_woodwinds_section_bracketed(self) -> None:
        """A woodwind section is joined by a bracket."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["flute", "oboe"]
        )
        assert "<group-symbol>bracket</group-symbol>" in to_musicxml(score)

    def test_brass_section_bracketed(self) -> None:
        """A brass section is joined by a bracket."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["horn", "trumpet"]
        )
        assert "<group-symbol>bracket</group-symbol>" in to_musicxml(score)

    def test_distinct_instruments_not_grouped(self) -> None:
        """Parts for unrelated instruments are not grouped."""
        score = new_score(
            key="C",
            time_signature="4/4",
            tempo_bpm=80,
            voices=["soprano", "alto", "tenor", "bass"],
        )
        assert "<part-group" not in to_musicxml(score)

    def test_single_part_not_grouped(self) -> None:
        """A lone part has no group."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["piano"])
        assert "<part-group" not in to_musicxml(score)

    def test_grouping_is_idempotent(self) -> None:
        """Serialising twice does not duplicate the group."""
        score = new_score(
            key="C", time_signature="4/4", tempo_bpm=80, voices=["rh", "lh"]
        )
        to_musicxml(score)
        xml = to_musicxml(score)
        assert xml.count("<group-symbol>brace</group-symbol>") == 1

    def test_section_with_single_member_not_grouped(self) -> None:
        """A lone section member is not bracketed."""
        score = new_score(key="C", time_signature="4/4", tempo_bpm=80, voices=["flute"])
        assert "<part-group" not in to_musicxml(score)


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

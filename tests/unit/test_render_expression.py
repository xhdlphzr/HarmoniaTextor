# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for render.expression."""

from __future__ import annotations

from music21 import (
    articulations,
    dynamics,
    expressions,
    instrument,
    meter,
    spanner,
    stream,
)
from music21 import (
    note as m21note,
)

from harmoniatextor.render.expression import realize_expressions

_SPAN = 45


def _score(
    pitches: list[str], *, velocity: int | None = None
) -> tuple[stream.Score, stream.Measure, list[m21note.Note]]:
    """Build a one-part score with one note per beat.

    Args:
        pitches: Scientific pitch names in performance order.
        velocity: Optional velocity applied to every note.

    Returns:
        The score, its measure and the created notes.
    """
    part = stream.Part()  # type: ignore[no-untyped-call]
    part.insert(0.0, instrument.Piano())  # type: ignore[no-untyped-call]  # music21
    measure = stream.Measure(number=1)
    measure.insert(0.0, meter.TimeSignature("4/4"))
    notes: list[m21note.Note] = []
    for index, pitch in enumerate(pitches):
        item = m21note.Note(pitch, quarterLength=1.0)
        if velocity is not None:
            item.volume.velocity = velocity
        measure.insert(float(index), item)
        notes.append(item)
    part.insert(0.0, measure)
    score = stream.Score()
    score.insert(0.0, part)
    return score, measure, notes


def _wedge(
    measure: stream.Measure, kind: type, first: m21note.Note, last: m21note.Note
) -> None:
    """Attach a crescendo/diminuendo wedge spanning two notes."""
    wedge = kind()
    measure.insert(0.0, wedge)
    wedge.addSpannedElements(first, last)


class TestRealizeExpressions:
    """Symbolic marks become performance data."""

    def test_crescendo_ramps_velocity(self) -> None:
        """A crescendo raises velocity note by note."""
        score, measure, notes = _score(["C5", "D5", "E5", "F5"], velocity=40)
        _wedge(measure, dynamics.Crescendo, notes[0], notes[3])
        realize_expressions(score)
        velocities = [item.volume.velocity for item in notes]
        assert velocities == [40, 40 + _SPAN // 3, 40 + 2 * _SPAN // 3, 40 + _SPAN]

    def test_diminuendo_ramps_velocity(self) -> None:
        """A diminuendo lowers velocity note by note."""
        score, measure, notes = _score(["C5", "D5", "E5", "F5"], velocity=120)
        _wedge(measure, dynamics.Diminuendo, notes[0], notes[3])
        realize_expressions(score)
        velocities = [item.volume.velocity for item in notes]
        assert velocities == [120, 105, 90, 75]

    def test_wedge_uses_default_velocity(self) -> None:
        """A wedge with no dynamic starts from the neutral default."""
        score, measure, notes = _score(["C5", "D5"])
        _wedge(measure, dynamics.Crescendo, notes[0], notes[1])
        realize_expressions(score)
        assert notes[0].volume.velocity == 64

    def test_wedge_with_one_note_skipped(self) -> None:
        """A wedge spanning fewer than two notes does nothing."""
        score, measure, notes = _score(["C5"])
        _wedge(measure, dynamics.Crescendo, notes[0], notes[0])
        realize_expressions(score)
        assert notes[0].volume.velocity is None

    def test_staccato_shortens(self) -> None:
        """A staccato note is shortened."""
        score, _measure, notes = _score(["C5"])
        notes[0].articulations.append(articulations.Staccato())  # type: ignore[no-untyped-call]
        realize_expressions(score)
        assert notes[0].quarterLength == 0.5

    def test_slur_creates_legato_overlap(self) -> None:
        """A slur lets its notes overlap slightly."""
        score, measure, notes = _score(["C5", "D5", "E5", "F5"], velocity=90)
        slur = spanner.Slur()  # type: ignore[no-untyped-call]
        measure.insert(0.0, slur)
        slur.addSpannedElements(notes[1], notes[2])
        realize_expressions(score)
        assert notes[1].quarterLength > 1.0
        assert notes[0].quarterLength == 1.0
        assert notes[3].quarterLength == 1.0

    def test_slur_overrides_staccato(self) -> None:
        """A slur keeps a note from being shortened by a staccato dot."""
        score, measure, notes = _score(["C5", "D5", "E5", "F5"], velocity=90)
        notes[1].articulations.append(articulations.Staccato())  # type: ignore[no-untyped-call]
        slur = spanner.Slur()  # type: ignore[no-untyped-call]
        measure.insert(0.0, slur)
        slur.addSpannedElements(notes[1], notes[2])
        realize_expressions(score)
        assert notes[1].quarterLength >= 1.0

    def test_slur_spans_measures(self) -> None:
        """A slur across measures is realised with global onsets."""
        part = stream.Part()  # type: ignore[no-untyped-call]
        part.insert(0.0, instrument.Piano())  # type: ignore[no-untyped-call]
        notes: list[m21note.Note] = []
        measures: list[stream.Measure] = []
        for number in (1, 2, 3):
            measure = stream.Measure(number=number)
            measure.insert(0.0, meter.TimeSignature("4/4"))
            for offset in range(4):
                item = m21note.Note("C5", quarterLength=1.0)
                item.volume.velocity = 90
                measure.insert(float(offset), item)
                notes.append(item)
            measures.append(measure)
            part.insert((number - 1) * 4.0, measure)
        score = stream.Score()
        score.insert(0.0, part)
        slur = spanner.Slur()  # type: ignore[no-untyped-call]
        measures[0].insert(0.0, slur)
        slur.addSpannedElements(notes[3], notes[4])
        realize_expressions(score)
        assert notes[3].quarterLength > 1.0

    def test_pedal_sustains_notes(self) -> None:
        """Notes under a sustain pedal ring until the pedal is released."""
        score, measure, notes = _score(["C5", "D5", "E5", "F5"], velocity=80)
        pedal = expressions.PedalMark()
        pedal.pedalType = expressions.PedalType.Sustain
        measure.insert(0.0, pedal)
        pedal.addSpannedElements(notes[0], notes[3])
        realize_expressions(score)
        assert notes[0].quarterLength == 4.0
        assert notes[3].quarterLength == 1.0

    def test_text_softens_from_offset(self) -> None:
        """A soft text expression only affects notes from its offset."""
        score, measure, notes = _score(["C5", "D5", "E5", "F5"], velocity=100)
        measure.insert(
            2.0,
            expressions.TextExpression("dolce"),  # type: ignore[no-untyped-call]
        )
        realize_expressions(score)
        assert notes[0].volume.velocity == 100
        assert notes[2].volume.velocity == 85

    def test_text_strengthens_and_clamps(self) -> None:
        """A strong text expression is clamped to the MIDI maximum."""
        score, measure, notes = _score(["C5"], velocity=120)
        measure.insert(
            0.0,
            expressions.TextExpression("marcato"),  # type: ignore[no-untyped-call]
        )
        realize_expressions(score)
        assert notes[0].volume.velocity == 127

    def test_unknown_text_ignored(self) -> None:
        """An unmapped text expression leaves velocities untouched."""
        score, measure, notes = _score(["C5"], velocity=90)
        measure.insert(
            0.0,
            expressions.TextExpression("grazioso"),  # type: ignore[no-untyped-call]
        )
        realize_expressions(score)
        assert notes[0].volume.velocity == 90

    def test_no_marks_is_noop(self) -> None:
        """A plain score is left unchanged."""
        score, _measure, notes = _score(["C5"], velocity=90)
        realize_expressions(score)
        assert notes[0].volume.velocity == 90
        assert notes[0].quarterLength == 1.0

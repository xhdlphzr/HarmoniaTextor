# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Realise symbolic expression marks into MIDI-friendly performance data.

music21's MIDI writer already performs dynamics and tempo marks, and turns
articulations into note velocities, but it ignores crescendo/diminuendo
wedges, never shortens staccato notes, cannot perform text expressions such as
``dolce``, and does not turn slurs into legato or pedal marks into sustain.
This module fills those gaps on a throw-away copy of the score immediately
before it is written to MIDI, so the exported audio follows the expression
marks drawn on the staff.
"""

from __future__ import annotations

import itertools

from music21 import (
    dynamics,
    expressions,
    spanner,
    stream,
)
from music21 import (
    note as m21note,
)

__all__ = ["realize_expressions"]

_DEFAULT_VELOCITY = 64
_WEDGE_SPAN = 45
_MIN_VELOCITY = 20
_MAX_VELOCITY = 127
_STACCATO_RATIO = 0.5
_MIN_DURATION = 0.05
_LEGATO_OVERLAP = 1.05
_LEGATO_CAP = 1.02

#: Text expressions that should sound softer.
_SOFT_WORDS = frozenset(
    {
        "dolce",
        "dolcissimo",
        "espressivo",
        "cantabile",
        "legato",
        "dolente",
        "mesto",
        "tranquillo",
        "calmo",
        "lontano",
        "sotto voce",
    }
)
#: Text expressions that should sound stronger.
_STRONG_WORDS = frozenset(
    {"marcato", "deciso", "risoluto", "energico", "agitato", "con forza", "fuoco"}
)
_SOFT_FACTOR = 0.85
_STRONG_FACTOR = 1.15
_EPSILON = 1e-6


def _velocity(item: m21note.NotRest) -> int:
    """Return a note's velocity, falling back to a neutral default.

    Args:
        item: The note or chord to read.

    Returns:
        The MIDI velocity in the range 1-127.
    """
    value = item.volume.velocity
    return int(value) if value is not None else _DEFAULT_VELOCITY


def _set_velocity(item: m21note.NotRest, value: float) -> None:
    """Clamp and set a note's velocity.

    Args:
        item: The note or chord to modify.
        value: The requested velocity.
    """
    item.volume.velocity = int(max(_MIN_VELOCITY, min(_MAX_VELOCITY, round(value))))


def _part_of(item: m21note.NotRest) -> stream.Part:
    """Return the part that contains a note.

    Args:
        item: The note or chord to inspect.

    Returns:
        The owning part.
    """
    measure = item.activeSite
    assert isinstance(measure, stream.Measure)
    part = measure.activeSite
    assert isinstance(part, stream.Part)
    return part


def _global_start(part: stream.Part, measure_number: int) -> float:
    """Return the global offset at which a measure begins.

    Args:
        part: The owning part.
        measure_number: One-based measure number.

    Returns:
        The offset in quarter notes from the start of the part.
    """
    total = 0.0
    for measure in part.getElementsByClass(stream.Measure):
        if int(measure.number) >= measure_number:
            break
        total += float(measure.barDuration.quarterLength)
    return total


def _global_onset(item: m21note.NotRest) -> float:
    """Return a note's onset measured from the start of its part.

    Args:
        item: The note or chord to inspect.

    Returns:
        The global onset in quarter notes.
    """
    measure = item.activeSite
    assert isinstance(measure, stream.Measure)
    return _global_start(_part_of(item), int(measure.number)) + float(item.offset)


def _spanned_notes(span: spanner.Spanner) -> list[m21note.NotRest]:
    """Return the notes a spanner spans, in performance order.

    music21 only exposes the two endpoint elements, so the notes between them
    are collected from the measures the span crosses.

    Args:
        span: The spanner (slur, wedge or pedal).

    Returns:
        The spanned notes and chords in order.
    """
    first = span.getFirst()  # type: ignore[no-untyped-call]
    last = span.getLast()  # type: ignore[no-untyped-call]
    assert isinstance(first, m21note.NotRest)
    assert isinstance(last, m21note.NotRest)
    first_measure = first.activeSite
    assert isinstance(first_measure, stream.Measure)
    last_measure = last.activeSite
    assert isinstance(last_measure, stream.Measure)
    part = _part_of(first)
    selected: list[m21note.NotRest] = []
    for measure in part.getElementsByClass(stream.Measure):
        number = int(measure.number)
        if not first_measure.number <= number <= last_measure.number:
            continue
        notes = sorted(measure.notes, key=lambda item: float(item.offset))
        for item in notes:
            if number == first_measure.number and item.offset + _EPSILON < first.offset:
                continue
            if number == last_measure.number and item.offset - _EPSILON > last.offset:
                continue
            selected.append(item)
    return selected


def _realize_wedges(score: stream.Score) -> None:
    """Ramp note velocities across crescendo and diminuendo wedges.

    Args:
        score: The score to modify in place.
    """
    wedges = score.recurse().getElementsByClass(
        (dynamics.Crescendo, dynamics.Diminuendo)
    )
    for wedge in wedges:
        notes = _spanned_notes(wedge)
        if len(notes) < 2:
            continue
        start = _velocity(notes[0])
        if isinstance(wedge, dynamics.Diminuendo):
            end = start - _WEDGE_SPAN
        else:
            end = start + _WEDGE_SPAN
        step = (end - start) / (len(notes) - 1)
        for index, item in enumerate(notes):
            _set_velocity(item, start + step * index)


def _realize_slurs(score: stream.Score) -> set[int]:
    """Give slurred notes a small legato overlap.

    Args:
        score: The score to modify in place.

    Returns:
        The ids of the notes that fall under a slur.
    """
    slurred: set[int] = set()
    slurs = score.recurse().getElementsByClass(spanner.Slur)
    for slur in slurs:
        notes = _spanned_notes(slur)
        for current, following in itertools.pairwise(notes):
            onset = _global_onset(current)
            following_onset = _global_onset(following)
            target = max(following_onset - onset, current.quarterLength)
            current.quarterLength = min(
                current.quarterLength * _LEGATO_OVERLAP, target * _LEGATO_CAP
            )
        for item in notes:
            slurred.add(id(item))
    return slurred


def _realize_pedal(score: stream.Score) -> None:
    """Hold notes under a sustain pedal until the pedal releases.

    Args:
        score: The score to modify in place.
    """
    pedals = score.recurse().getElementsByClass(expressions.PedalMark)
    for pedal in pedals:
        notes = _spanned_notes(pedal)
        onsets = [(item, _global_onset(item)) for item in notes]
        release = max(onset + item.quarterLength for item, onset in onsets)
        for item, onset in onsets:
            reach = release - onset
            item.quarterLength = max(item.quarterLength, reach)


def _realize_articulations(score: stream.Score, slurred: set[int]) -> None:
    """Shorten staccato notes so they are detached in the audio.

    Slurred notes are exempt: a slur overrides a staccato dot.

    Args:
        score: The score to modify in place.
        slurred: The ids of notes covered by a slur.
    """
    for item in list(score.recurse().notes):
        names = {type(articulation).__name__ for articulation in item.articulations}
        if "Staccato" in names and id(item) not in slurred:
            item.quarterLength = max(
                _MIN_DURATION, item.quarterLength * _STACCATO_RATIO
            )


def _realize_text(score: stream.Score) -> None:
    """Map expressive text expressions onto note velocities.

    Args:
        score: The score to modify in place.
    """
    texts = list(score.recurse().getElementsByClass(expressions.TextExpression))
    for text in texts:
        word = str(text.content).strip().lower()
        if word in _SOFT_WORDS:
            factor = _SOFT_FACTOR
        elif word in _STRONG_WORDS:
            factor = _STRONG_FACTOR
        else:
            continue
        container = text.activeSite
        assert container is not None
        base = float(text.offset)
        for item in list(container.notes):
            if float(item.offset) + _EPSILON >= base:
                _set_velocity(item, _velocity(item) * factor)


def realize_expressions(score: stream.Score) -> None:
    """Make the score's expression marks audible when written to MIDI.

    The score is modified in place, so it must be a throw-away copy (the export
    layer rebuilds the score from MusicXML before calling this).

    Args:
        score: The score to realise in place.
    """
    _realize_wedges(score)
    slurred = _realize_slurs(score)
    _realize_articulations(score, slurred)
    _realize_pedal(score)
    _realize_text(score)

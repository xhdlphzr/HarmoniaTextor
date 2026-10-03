# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Score flattening utilities used by the symbolic checker."""

from __future__ import annotations

from dataclasses import dataclass

from music21 import note as m21note
from music21 import stream

from harmoniatextor.domain.models import ThemeNote

__all__ = [
    "VoiceEvent",
    "first_melody",
    "measure_count",
    "measure_of",
    "onsets",
    "voice_events",
    "voice_order",
]

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class VoiceEvent:
    """A single sounding note attack.

    Attributes:
        offset: Global offset in quarter notes.
        measure: One-based measure number.
        pitch: Scientific pitch name.
        midi: MIDI note number.
        quarter_length: Duration in quarter notes.
    """

    offset: float
    measure: int
    pitch: str
    midi: int
    quarter_length: float


def _bar_length(score: stream.Score) -> float:
    """Return the bar length of a score in quarter notes.

    Args:
        score: The score to inspect.

    Returns:
        The resulting number.
    """
    signatures = list(score.recurse().getElementsByClass("TimeSignature"))
    if not signatures:
        return 4.0
    return float(signatures[0].barDuration.quarterLength)


def _measure_entries(part: stream.Part) -> list[tuple[float, float, int]]:
    """Return each measure's ``(start, end, number)`` boundaries.

    Using the measure objects rather than one global bar length makes the
    mapping correct when the meter changes mid-piece.

    Args:
        part: The part to inspect.

    Returns:
        Measure boundaries in performance order.
    """
    entries: list[tuple[float, float, int]] = []
    for measure in part.getElementsByClass(stream.Measure):
        start = float(measure.offset)
        end = start + float(measure.barDuration.quarterLength)
        entries.append((start, end, int(measure.number)))
    return entries


def _locate_measure(
    entries: list[tuple[float, float, int]], offset: float, fallback_bar: float
) -> int:
    """Map a global offset to a measure number.

    Args:
        entries: Measure boundaries from :func:`_measure_entries`.
        offset: Global offset in quarter notes.
        fallback_bar: Bar length used when no measures are known.

    Returns:
        The one-based measure number.
    """
    if not entries:
        return int(offset // fallback_bar) + 1
    for start, end, number in entries:
        if start - _EPSILON <= offset < end - _EPSILON:
            return number
    last_start, last_end, last_number = entries[-1]
    if offset >= last_end - _EPSILON:
        bar = last_end - last_start
        if bar > 0:
            return last_number + int((offset - last_end) // bar) + 1
        return last_number  # pragma: no cover - a zero-length measure is invalid
    return entries[0][2]


def measure_of(score: stream.Score, offset: float) -> int:
    """Convert a global offset into a one-based measure number.

    Args:
        score: The score providing the measures.
        offset: Global offset in quarter notes.

    Returns:
        The one-based measure number.
    """
    parts = list(score.parts)
    if parts:
        longest = max(
            parts, key=lambda part: len(list(part.getElementsByClass(stream.Measure)))
        )
        entries = _measure_entries(longest)
    else:
        entries = []
    return _locate_measure(entries, offset, _bar_length(score))


def voice_order(score: stream.Score) -> list[str]:
    """Return voice names ordered from top to bottom.

    Args:
        score: The score to inspect.

    Returns:
        Voice slot names in part order.
    """
    return [str(part.id or part.partName) for part in score.parts]


def voice_events(score: stream.Score) -> dict[str, list[VoiceEvent]]:
    """Flatten a score into per-voice note attacks.

    Args:
        score: The score to flatten.

    Returns:
        A mapping from voice name to its ordered list of attacks.
    """
    result: dict[str, list[VoiceEvent]] = {}
    bar = _bar_length(score)
    for part in score.parts:
        voice = str(part.id or part.partName)
        entries = _measure_entries(part)
        events: list[VoiceEvent] = []
        for element in part.flatten().notes:
            if isinstance(element, m21note.Note):
                pitches = [element.pitch]
            else:
                pitches = list(element.pitches)
            measure = _locate_measure(entries, float(element.offset), bar)
            for item in pitches:
                events.append(
                    VoiceEvent(
                        offset=float(element.offset),
                        measure=measure,
                        pitch=item.nameWithOctave,
                        midi=int(item.midi),
                        quarter_length=float(element.quarterLength),
                    )
                )
        result[voice] = sorted(events, key=lambda event: event.offset)
    return result


def onsets(score: stream.Score) -> list[float]:
    """Return the sorted unique attack offsets of a score.

    Args:
        score: The score to inspect.

    Returns:
        Sorted unique offsets in quarter notes.
    """
    offsets: set[float] = set()
    for events in voice_events(score).values():
        offsets.update(event.offset for event in events)
    return sorted(offsets)


def first_melody(score: stream.Score) -> tuple[str, list[ThemeNote]]:
    """Return the first non-empty melodic line of a score.

    Args:
        score: The score to inspect.

    Returns:
        A ``(voice, notes)`` tuple; ``voice`` is empty when no notes exist.
    """
    for part in score.parts:
        notes = [
            ThemeNote(
                pitch=element.nameWithOctave,
                quarter_length=float(element.quarterLength),
            )
            for element in part.flatten().notes
            if isinstance(element, m21note.Note)
        ]
        if notes:
            return str(part.id or part.partName), notes
    return "", []


def measure_count(score: stream.Score) -> int:
    """Return the highest measure number across all parts.

    Args:
        score: The score to inspect.

    Returns:
        The number of measures (at least one).
    """
    highest = 0
    for part in score.parts:
        for measure in part.getElementsByClass(stream.Measure):
            highest = max(highest, int(measure.number))
    return max(highest, 1)

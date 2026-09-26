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
    """Return the bar length of a score in quarter notes."""
    signatures = list(score.recurse().getElementsByClass("TimeSignature"))
    if not signatures:
        return 4.0
    return float(signatures[0].barDuration.quarterLength)


def measure_of(score: stream.Score, offset: float) -> int:
    """Convert a global offset into a one-based measure number.

    Args:
        score: The score providing the meter.
        offset: Global offset in quarter notes.

    Returns:
        The one-based measure number.
    """
    bar = _bar_length(score)
    return int(offset // bar) + 1


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
        events: list[VoiceEvent] = []
        for element in part.flatten().notes:
            if isinstance(element, m21note.Note):
                pitches = [element.pitch]
            else:
                pitches = list(element.pitches)
            for item in pitches:
                events.append(
                    VoiceEvent(
                        offset=float(element.offset),
                        measure=int(element.offset // bar) + 1,
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
            ThemeNote(pitch=element.nameWithOctave, quarter_length=float(element.quarterLength))
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

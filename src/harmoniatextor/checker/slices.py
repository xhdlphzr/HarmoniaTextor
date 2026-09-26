# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Vertical slicing helpers shared by the voice-leading rules."""

from __future__ import annotations

from dataclasses import dataclass

from music21 import pitch as m21pitch
from music21 import stream

from harmoniatextor.domain.key import parse_key
from harmoniatextor.score.analysis import VoiceEvent, measure_of, voice_events

__all__ = ["Slice", "build_slices", "pair_sequence", "pitch_classes", "tonic_pc"]

PERFECT_FIFTH = 7
PERFECT_OCTAVE = 0
_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class Slice:
    """A vertical sonority at a single attack offset.

    Attributes:
        offset: Global offset in quarter notes.
        measure: One-based measure number.
        events: Voice name to the event attacking at this offset.
    """

    offset: float
    measure: int
    events: dict[str, VoiceEvent]


def build_slices(score: stream.Score) -> list[Slice]:
    """Build vertical slices from all note attacks.

    Args:
        score: The score to slice.

    Returns:
        Slices ordered by offset.
    """
    events_by_voice = voice_events(score)
    offsets: set[float] = set()
    for voice_list in events_by_voice.values():
        offsets.update(event.offset for event in voice_list)
    result: list[Slice] = []
    for offset in sorted(offsets):
        events: dict[str, VoiceEvent] = {
            voice: event
            for voice, voice_list in events_by_voice.items()
            for event in voice_list
            if abs(event.offset - offset) < _EPSILON
        }
        if events:
            result.append(Slice(offset=offset, measure=measure_of(score, offset), events=events))
    return result


def pair_sequence(
    slices: list[Slice], voice_a: str, voice_b: str
) -> list[tuple[VoiceEvent, VoiceEvent]]:
    """Return consecutive slices where both voices attack.

    Args:
        slices: Vertical slices.
        voice_a: First voice.
        voice_b: Second voice.

    Returns:
        A list of ``(event_a, event_b)`` pairs in time order.
    """
    pairs: list[tuple[VoiceEvent, VoiceEvent]] = []
    for item in slices:
        if voice_a in item.events and voice_b in item.events:
            pairs.append((item.events[voice_a], item.events[voice_b]))
    return pairs


def pitch_classes(events: dict[str, VoiceEvent]) -> set[int]:
    """Return the set of pitch classes sounding in a slice.

    Args:
        events: Voice to event mapping.

    Returns:
        Pitch classes as integers 0-11.
    """
    return {event.midi % 12 for event in events.values()}


def tonic_pc(tonic: str) -> int:
    """Return the pitch class of a key tonic.

    Args:
        tonic: Key specification such as ``"C"`` or ``"f#"``.

    Returns:
        The pitch class as an integer 0-11.
    """
    return int(m21pitch.Pitch(f"{parse_key(tonic).tonic}4").pitchClass)

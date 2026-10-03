# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Shared musical helpers for technique implementations."""

from __future__ import annotations

from dataclasses import dataclass

from music21 import interval as m21interval
from music21 import key as m21key
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import roman

from harmoniatextor.domain.interval import DiatonicInterval
from harmoniatextor.domain.key import parse_key
from harmoniatextor.domain.models import Theme, ThemeNote
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques.base import TechniqueContext, TechniqueError

__all__ = [
    "average_axis_midi",
    "chord_pitches",
    "functional_figure",
    "get_theme",
    "invert_notes",
    "midi_to_name",
    "name_to_midi",
    "note_value",
    "realize_chord",
    "transpose_note",
    "transpose_notes",
    "transpose_to_key",
]

_SEMITONES_PER_OCTAVE = 12
_MAX_DIRECTION_SEMITONES = 6

_NOTE_VALUES: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "dotted-half": 3.0,
    "dotted-quarter": 1.5,
    "dotted-eighth": 0.75,
}


def note_value(name: str) -> float:
    """Return the duration of a note-value name in quarter notes.

    Args:
        name: A note value such as ``"eighth"`` or ``"dotted-quarter"``.

    Returns:
        The duration in quarter notes.

    Raises:
        TechniqueError: When the name is unknown.
    """
    value = _NOTE_VALUES.get(name)
    if value is None:
        raise TechniqueError("BAD_PARAM", f"unknown note value {name!r}")
    return value


def get_theme(ctx: TechniqueContext, theme_id: int) -> Theme:
    """Look up a theme or raise a technique error.

    Args:
        ctx: Technique context holding the theme registry.
        theme_id: Theme number to look up.

    Returns:
        The matching theme.

    Raises:
        TechniqueError: When no theme with the given id exists.
    """
    theme = ctx.themes.get(theme_id)
    if theme is None:
        raise TechniqueError("THEME_NOT_FOUND", f"theme {theme_id} does not exist")
    return theme


def name_to_midi(pitch: str) -> int:
    """Convert a scientific pitch name to a MIDI number.

    Args:
        pitch: Scientific pitch name such as ``"C4"``.

    Returns:
        The MIDI note number.
    """
    return int(m21pitch.Pitch(pitch).midi)


def midi_to_name(midi: int) -> str:
    """Convert a MIDI number to a scientific pitch name.

    Args:
        midi: MIDI note number.

    Returns:
        A scientific pitch name using sharps.
    """
    return str(m21pitch.Pitch(midi=midi).nameWithOctave)


def transpose_note(pitch: str, interval: DiatonicInterval) -> str:
    """Transpose a pitch by a diatonic interval.

    Args:
        pitch: Scientific pitch name.
        interval: The interval to apply.

    Returns:
        The transposed pitch name.
    """
    transposed = m21note.Note(pitch).transpose(  # type: ignore[no-untyped-call]
        m21interval.Interval(interval.name)
    )
    return str(transposed.nameWithOctave)


def transpose_notes(
    notes: list[ThemeNote], interval: DiatonicInterval
) -> list[ThemeNote]:
    """Transpose a melody.

    Args:
        notes: The melody to transpose.
        interval: The interval to apply.

    Returns:
        A new transposed melody.
    """
    return [
        ThemeNote(
            pitch=transpose_note(item.pitch, interval),
            quarter_length=item.quarter_length,
        )
        for item in notes
    ]


def average_axis_midi(notes: list[ThemeNote]) -> int:
    """Return the duration-weighted average MIDI pitch of a melody.

    Args:
        notes: The melody.

    Returns:
        The rounded average MIDI number.

    Raises:
        TechniqueError: When the melody is empty.
    """
    if not notes:
        raise TechniqueError(
            "THEME_NOT_FOUND", "cannot compute an axis for an empty theme"
        )
    total = sum(name_to_midi(item.pitch) * item.quarter_length for item in notes)
    weight = sum(item.quarter_length for item in notes)
    return round(total / weight)


def transpose_to_key(notes: list[ThemeNote], target_key: str) -> list[ThemeNote]:
    """Transpose a melody so that its first note maps into a target key.

    Args:
        notes: The melody to transpose.
        target_key: Target key specification such as ``"G"`` or ``"f#"``.

    Returns:
        A transposed melody whose first note is the target tonic class.

    Raises:
        TechniqueError: When the melody is empty.
    """
    if not notes:
        raise TechniqueError("THEME_NOT_FOUND", "cannot transpose an empty theme")
    tonic = parse_key(target_key)
    first = name_to_midi(notes[0].pitch)
    target = name_to_midi(f"{tonic.tonic}4")
    delta = (target - first) % _SEMITONES_PER_OCTAVE
    if delta > _MAX_DIRECTION_SEMITONES:
        delta -= _SEMITONES_PER_OCTAVE
    return [
        ThemeNote(
            pitch=midi_to_name(name_to_midi(item.pitch) + delta),
            quarter_length=item.quarter_length,
        )
        for item in notes
    ]


def invert_notes(notes: list[ThemeNote], axis_midi: int) -> list[ThemeNote]:
    """Mirror a melody around an axis.

    Args:
        notes: The melody to invert.
        axis_midi: The mirror axis as a MIDI number.

    Returns:
        A new inverted melody.
    """
    result: list[ThemeNote] = []
    for item in notes:
        mirrored = 2 * axis_midi - name_to_midi(item.pitch)
        result.append(
            ThemeNote(pitch=midi_to_name(mirrored), quarter_length=item.quarter_length)
        )
    return result


def chord_pitches(key: str, figure: str) -> list[str]:
    """Realise a roman-numeral figure into sounding pitches.

    Args:
        key: Key specification such as ``"C"`` or ``"a"``.
        figure: Roman numeral figure such as ``"V7"`` or ``"vii°7"``.

    Returns:
        Chord pitches in ascending order.

    Raises:
        TechniqueError: When the figure cannot be realised.
    """
    try:
        numeral = roman.RomanNumeral(figure, m21key.Key(parse_key(key).music21_name))
    except Exception as exc:
        raise TechniqueError(
            "BAD_PARAM", f"cannot realise chord {figure!r} in {key!r}"
        ) from exc
    return [item.nameWithOctave for item in numeral.pitches]


def functional_figure(key: str, function: str) -> str:
    """Map a tonic/subdominant/dominant function to a roman numeral.

    Args:
        key: Key specification.
        function: One of ``"T"``, ``"S"``, ``"D"`` (case-insensitive).

    Returns:
        A roman-numeral figure.

    Raises:
        TechniqueError: When the function is unknown.
    """
    major = parse_key(key).is_major
    table = {
        "T": "I" if major else "i",
        "S": "IV" if major else "iv",
        "D": "V",
    }
    figure = table.get(function.upper())
    if figure is None:
        raise TechniqueError("BAD_PARAM", f"unknown function {function!r}")
    return figure


@dataclass(frozen=True, slots=True)
class Placement:
    """A measure, offset and duration position.

    Attributes:
        measure: One-based measure number.
        offset: Offset within the measure in quarter notes.
        duration: Duration in quarter notes.
    """

    measure: int
    offset: float
    duration: float


def realize_chord(
    editor: ScoreEditor,
    key: str,
    figure: str,
    placement: Placement,
) -> list[str]:
    """Write a realised chord across the score's voices.

    Args:
        editor: Score editor.
        key: Key specification.
        figure: Roman-numeral figure.
        placement: Where and how long to place the chord.

    Returns:
        The chord pitches that were written.
    """
    pitches = chord_pitches(key, figure)
    for index, voice in enumerate(editor.voice_names()):
        chord_index = len(pitches) - 1 - index
        pitch = pitches[max(chord_index, 0)]
        editor.place_note(
            voice, placement.measure, placement.offset, pitch, placement.duration
        )
    return pitches

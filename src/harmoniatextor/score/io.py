# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""music21 bridge for MusicXML import/export and score construction."""

from __future__ import annotations

import copy
import warnings
from typing import Any

from music21 import converter, instrument, meter, stream, tempo
from music21 import key as m21key
from music21.musicxml.m21ToXml import GeneralObjectExporter
from music21.musicxml.xmlObjects import MusicXMLWarning

__all__ = [
    "DEFAULT_INSTRUMENTS",
    "clone_score",
    "ensure_instruments",
    "from_musicxml",
    "instrument_for_voice",
    "make_instrument",
    "new_part",
    "new_score",
    "to_musicxml",
]

DEFAULT_INSTRUMENTS: dict[str, str] = {
    "soprano": "Soprano",
    "alto": "Alto",
    "tenor": "Tenor",
    "bass": "Bass",
    "solo": "Violin",
    "violin": "Violin",
    "violin1": "Violin",
    "violin2": "Violin",
    "viola": "Viola",
    "cello": "Violoncello",
    "contrabass": "Contrabass",
    "flute": "Flute",
    "oboe": "Oboe",
    "clarinet": "Clarinet",
    "bassoon": "Bassoon",
    "horn": "Horn",
    "trumpet": "Trumpet",
    "trombone": "Trombone",
    "tuba": "Tuba",
    "timpani": "Timpani",
    "harpsichord": "Harpsichord",
    "organ": "Organ",
    "piano": "Piano",
}


def instrument_for_voice(voice: str) -> str:
    """Return the default instrument name for a voice slot.

    Args:
        voice: Voice slot name such as ``"violin2"``.

    Returns:
        A music21 instrument name, defaulting to ``"Piano"``.
    """
    lowered = voice.lower()
    if lowered in DEFAULT_INSTRUMENTS:
        return DEFAULT_INSTRUMENTS[lowered]
    return DEFAULT_INSTRUMENTS.get(lowered.rstrip("0123456789"), "Piano")


def make_instrument(name: str) -> instrument.Instrument:
    """Build a music21 instrument from a free-form name.

    Args:
        name: Instrument name such as ``"Violin"`` or ``"Flute"``.

    Returns:
        The matching instrument, or a piano when the name is unknown.
    """
    try:
        found = instrument.fromString(name)
    except Exception:
        return instrument.Piano()  # type: ignore[no-untyped-call]  # music21
    if found is None:
        return instrument.Piano()  # type: ignore[no-untyped-call]  # music21
    return found  # type: ignore[no-any-return]  # music21 resolver is untyped


def new_part(voice: str, instrument_name: str | None = None) -> stream.Part:
    """Create an empty named part for a voice slot.

    Args:
        voice: Voice slot name, used as both part name and identifier.
        instrument_name: Instrument name; defaults to a voice-based mapping.

    Returns:
         A new :class:`music21.stream.Part`.
    """
    part = stream.Part()  # type: ignore[no-untyped-call]  # music21 is unannotated
    part.partName = voice
    part.id = voice
    part.insert(0.0, make_instrument(instrument_name or instrument_for_voice(voice)))
    return part


def new_score(
    *,
    key: str,
    time_signature: str,
    tempo_bpm: int,
    voices: list[str],
    instruments: dict[str, str] | None = None,
) -> stream.Score:
    """Create a fresh multi-voice score.

    Args:
        key: music21-compatible key name, e.g. ``"C"`` or ``"a"``.
        time_signature: Time signature such as ``"4/4"``.
        tempo_bpm: Tempo in quarter notes per minute.
        voices: Voice slot names, ordered from top to bottom.
        instruments: Optional voice-to-instrument overrides.

    Returns:
        A score with one part per voice and a first measure carrying the time
        signature, key signature and tempo.
    """
    score = stream.Score()
    overrides = instruments or {}
    for voice in voices:
        part = new_part(voice, overrides.get(voice))
        measure = stream.Measure(number=1)
        measure.insert(0.0, meter.TimeSignature(time_signature))
        measure.insert(0.0, m21key.Key(key))
        measure.insert(0.0, tempo.MetronomeMark(number=tempo_bpm))
        part.insert(0.0, measure)
        score.insert(0.0, part)
    return score


def clone_score(score: stream.Score) -> stream.Score:
    """Deep-copy a score.

    Args:
        score: The score to copy.

    Returns:
        An independent deep copy.
    """
    return copy.deepcopy(score)


def to_musicxml(score: stream.Score) -> str:
    """Serialise a score to MusicXML text.

    A part-less score (a movement before the composer AI creates any voice) is
    serialised as a single empty staff; music21's "not well-formed" warning for
    that intentional state is suppressed.

    Args:
        score: The score to serialise.

    Returns:
        A MusicXML document as text.
    """
    exporter = GeneralObjectExporter(score)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=MusicXMLWarning)
        data: bytes = exporter.parse()
    return data.decode("utf-8")


def ensure_instruments(score: stream.Score) -> None:
    """Give every part a program-bearing instrument when it lacks one.

    The composer agent may submit MusicXML without instrument definitions, in
    which case music21 attaches a program-less ``Instrument`` and every part
    plays back as piano.  The instrument is derived from the part name (or id)
    via :func:`instrument_for_voice`.

    Args:
        score: The score to fix in place.
    """
    for part in score.parts:
        existing = list(part.getElementsByClass(instrument.Instrument))
        if any(item.midiProgram is not None for item in existing):
            continue
        for item in existing:
            part.remove(item)
        voice = str(part.partName or part.id or "")
        part.insert(0.0, make_instrument(instrument_for_voice(voice)))


def from_musicxml(xml: str) -> stream.Score:
    """Parse MusicXML text into a score.

    Parsed parts always carry an instrument so audio export uses the right
    timbre for each voice.

    Args:
        xml: A MusicXML document.

    Returns:
        The parsed score.

    Raises:
        ValueError: If the document cannot be parsed into a score.
    """
    parsed: Any = converter.parseData(xml, format="musicxml")
    if isinstance(parsed, stream.Score):
        ensure_instruments(parsed)
        return parsed
    if isinstance(parsed, stream.Part):
        score = stream.Score()
        score.insert(0.0, parsed)
        ensure_instruments(score)
        return score
    raise ValueError("MusicXML did not parse into a score")

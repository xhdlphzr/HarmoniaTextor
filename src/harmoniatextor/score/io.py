# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""music21 bridge for MusicXML import/export and score construction."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any, Literal

from music21 import clef, converter, instrument, layout, meter, stream, tempo
from music21 import key as m21key
from music21.musicxml.m21ToXml import GeneralObjectExporter
from music21.musicxml.xmlObjects import MusicXMLWarning

__all__ = [
    "DEFAULT_INSTRUMENTS",
    "assign_clefs",
    "clone_score",
    "ensure_instruments",
    "from_musicxml",
    "grand_staff_key",
    "group_staves",
    "instrument_for_voice",
    "make_instrument",
    "new_part",
    "new_score",
    "to_musicxml",
]

#: MIDI number of middle C, the register boundary for piano-hand clefs.
_MIDDLE_C = 60

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
    except Exception:  # noqa: BLE001 - music21 raises many types for bad names
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


#: Tokens (matched as substrings) that put a part in a bass clef.
_BASS_TOKENS = (
    "cello",
    "bass",
    "trombone",
    "tuba",
    "euphonium",
    "baritone",
    "timpani",
    "kettle",
    "pedal",
)
#: Tokens that put a part in an alto clef.
_ALTO_TOKENS = ("viola",)
#: Instruments that read by hand or register (piano, harp, organ).
_KEYBOARD_TOKENS = ("piano", "harp", "organ")


def _part_tokens(part: stream.Part) -> set[str]:
    """Return the lowercased identifiers of a part.

    The part id and name, plus every instrument's class name and label, are
    collected so an instrument can be recognised however it was recorded.

    Args:
        part: Part to inspect.

    Returns:
        A set of lowercased tokens.
    """
    tokens: set[str] = set()
    for value in (part.id, part.partName):
        if value:
            tokens.add(str(value).lower())
    for item in part.getElementsByClass(instrument.Instrument):
        tokens.add(type(item).__name__.lower())
        name = str(item.instrumentName or "").lower()
        if name:
            tokens.add(name)
    return tokens


def _matches(tokens: set[str], needles: tuple[str, ...]) -> bool:
    """Return whether any token contains any needle.

    Args:
        tokens: Lowercased tokens to search.
        needles: Lowercased substrings to look for.

    Returns:
        ``True`` when a needle occurs in a token.
    """
    return any(needle in token for token in tokens for needle in needles)


def _median_midi(part: stream.Part) -> float | None:
    """Return the median pitch of a part's notes.

    Args:
        part: Part to inspect.

    Returns:
        The median MIDI number, or ``None`` when the part has no notes.
    """
    values: list[float] = []
    for item in part.recurse().notes:
        values.extend(float(pitch.midi) for pitch in item.pitches)
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _clef_sign(part: stream.Part) -> str | None:
    """Choose a clef sign for a part.

    A left hand or pedal staff (``lh``/``left``/``pedal``) takes a bass clef and
    a right hand (``rh``/``right``) a treble clef.  Low instruments (cello,
    contrabass, bassoon, contrabassoon, trombone, tuba, euphonium, baritone,
    timpani) also take a bass clef, and the viola an alto clef.  Any other
    keyboard-family part (piano, harp, organ) falls back to its register.  Parts
    that are neither keep music21's default.

    Args:
        part: Part to inspect.

    Returns:
        ``"bass"``, ``"alto"``, ``"treble"`` or ``None`` to leave it unchanged.
    """
    names = [str(value).lower() for value in (part.id, part.partName) if value]
    lowered = " ".join(names)
    if (
        "left" in lowered
        or "pedal" in lowered
        or any(name.startswith("lh") for name in names)
    ):
        return "bass"
    if "right" in lowered or any(name.startswith("rh") for name in names):
        return "treble"
    tokens = _part_tokens(part)
    if _matches(tokens, _ALTO_TOKENS):
        return "alto"
    if _matches(tokens, _BASS_TOKENS):
        return "bass"
    if _matches(tokens, _KEYBOARD_TOKENS):
        median = _median_midi(part)
        if median is not None and median < _MIDDLE_C:
            return "bass"
        return "treble"
    return None


def _set_clef(part: stream.Part, sign: str) -> None:
    """Replace a part's clef with the requested one.

    Args:
        part: Part to modify in place.
        sign: One of ``"bass"``, ``"alto"`` or ``"treble"``.
    """
    for existing in list(part.recurse().getElementsByClass(clef.Clef)):
        site = existing.activeSite
        if site is not None:
            site.remove(existing)
    measures = list(part.getElementsByClass(stream.Measure))
    target = measures[0] if measures else stream.Measure(number=1)
    if not measures:
        part.insert(0.0, target)
    chosen: clef.Clef
    if sign == "bass":
        chosen = clef.BassClef()  # type: ignore[no-untyped-call]  # music21
    elif sign == "alto":
        chosen = clef.AltoClef()  # type: ignore[no-untyped-call]  # music21
    else:
        chosen = clef.TrebleClef()  # type: ignore[no-untyped-call]  # music21
    target.insert(0.0, chosen)


def assign_clefs(score: stream.Score) -> None:
    """Give every part a register-appropriate clef.

    Called just before serialisation so the rendered and exported staff always
    notates low instruments and piano/harp left hands in a bass clef and the
    viola in an alto clef, no matter what the composer AI named the voices.
    Parts that read comfortably in a treble clef are left untouched.

    Args:
        score: The score to modify in place.
    """
    for part in score.parts:
        sign = _clef_sign(part)
        if sign is not None:
            _set_clef(part, sign)


#: Single-player instruments that may span several staves; parts of one of
#: these that share the instrument are joined by a brace (e.g. piano hands).
#: The more specific ``harpsichord`` is listed before ``harp`` so it wins.
_GRAND_STAFF = ("piano", "harpsichord", "celesta", "harp", "organ", "keyboard")
#: Orchestral sections, joined by a bracket.  Order is the score order.
_SECTION_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("woodwinds", ("flute", "piccolo", "oboe", "clarinet", "bassoon")),
    ("brass", ("trumpet", "horn", "trombone", "tuba", "euphonium")),
    ("strings", ("violin", "viola", "cello", "contrabass", "double bass")),
    ("percussion", ("timpani", "percussion", "drum", "cymbal", "glockenspiel")),
)


def grand_staff_key(part: stream.Part) -> str | None:
    """Return the one-player grand-staff instrument of a part.

    Parts that share this key are staves of one instrument played by one
    person (for example a piano's hands), so they may be locked together.

    Args:
        part: Part to inspect.

    Returns:
        A lowercased instrument key, or ``None`` for a single-staff part.
    """
    tokens = _part_tokens(part)
    for key in _GRAND_STAFF:
        if _matches(tokens, (key,)):
            return key
    return None


def _section(part: stream.Part) -> str | None:
    """Return the orchestral section of a part.

    Args:
        part: Part to inspect.

    Returns:
        A section name, or ``None`` when the part is not orchestral.
    """
    tokens = _part_tokens(part)
    for name, keys in _SECTION_FAMILIES:
        if _matches(tokens, keys):
            return name
    return None


def _hand_hint(part: stream.Part) -> bool:
    """Return whether a part names a hand or pedal staff.

    Args:
        part: Part to inspect.

    Returns:
        ``True`` for a left/right hand or pedal part.
    """
    names = [str(value).lower() for value in (part.id, part.partName) if value]
    return any(
        name.startswith(("lh", "rh"))
        or "left" in name
        or "right" in name
        or "pedal" in name
        for name in names
    )


def _make_group(members: list[stream.Part]) -> layout.StaffGroup:
    """Build the braced group of one instrument's associated staves.

    Args:
        members: The grouped parts, in score order.

    Returns:
        The configured staff group.
    """
    label = grand_staff_key(members[0]) or "group"
    return layout.StaffGroup(members, name=label, symbol="brace", barTogether=True)


def _add_groups(
    score: stream.Score,
    parts: list[stream.Part],
    key_of: Callable[[stream.Part], str | None],
    symbol: Literal["brace", "bracket"],
    *,
    continue_on_hint: bool = False,
) -> None:
    """Join consecutive parts that share a grouping key.

    Args:
        score: Score to modify in place.
        parts: Candidate parts, in score order.
        key_of: Maps a part to its grouping key, or ``None`` to skip it.
        symbol: MusicXML group symbol, ``"brace"`` or ``"bracket"``.
        continue_on_hint: Whether a hand/pedal part joins the previous group
            even when its own key differs (so an organ pedal groups with the
            organ manuals).
    """
    groups: list[list[stream.Part]] = []
    keys: list[str | None] = []
    for part in parts:
        key = key_of(part)
        if continue_on_hint and _hand_hint(part) and keys:
            groups[-1].append(part)
            continue
        if key is not None and keys and keys[-1] == key:
            groups[-1].append(part)
        else:
            groups.append([part])
            keys.append(key)
    for key, group in zip(keys, groups):
        if key is None or len(group) < 2:
            continue
        staff_group = layout.StaffGroup(
            group, name=key, symbol=symbol, barTogether=True
        )
        score.insert(0.0, staff_group)


def group_staves(score: stream.Score) -> None:
    """Join staves of one instrument or one section.

    Called just before serialisation.  Staves the composer explicitly locked
    with ``add_part(associate=...)`` (the same one-player instrument, e.g. a
    piano's hands) stay joined by a brace.  Parts the composer left ungrouped
    fall back to automatic detection: one instrument's staves are braced and an
    orchestral section (strings, woodwinds, brass, percussion) is bracketed.

    Args:
        score: The score to modify in place.
    """
    parts = list(score.parts)
    identities = {id(part) for part in parts}
    covered: set[int] = set()
    explicit: list[list[stream.Part]] = []
    for existing in list(score.getElementsByClass(layout.StaffGroup)):
        members = [
            member
            for member in existing.getSpannedElements()  # type: ignore[no-untyped-call]
            if id(member) in identities and id(member) not in covered
        ]
        site = existing.activeSite
        if site is not None:
            site.remove(existing)
        if len(members) >= 2:
            explicit.append(members)
            covered.update(id(member) for member in members)
    for members in explicit:
        score.insert(0.0, _make_group(members))
    remaining = [part for part in parts if id(part) not in covered]
    _add_groups(score, remaining, grand_staff_key, "brace", continue_on_hint=True)
    _add_groups(score, remaining, _section, "bracket")


def to_musicxml(score: stream.Score) -> str:
    """Serialise a score to MusicXML text.

    A part-less score (a movement before the composer AI creates any voice) is
    serialised as a single empty staff; music21's "not well-formed" warning for
    that intentional state is suppressed.  Register-appropriate clefs
    (:func:`assign_clefs`) and staff groups (:func:`group_staves`) are applied
    first, so the rendered staff notates low instruments in a bass clef, the
    viola in an alto clef, and braces an instrument's staves while bracketing
    orchestral sections.

    Args:
        score: The score to serialise.

    Returns:
        A MusicXML document as text.
    """
    assign_clefs(score)
    group_staves(score)
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

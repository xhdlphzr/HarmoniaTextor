# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Attach, read and strip sung lyrics on a score.

A vocal line stores two lyric verses on every note: the original-language text
(verse 1) and its IPA pronunciation (verse 2, hidden from the rendered staff).
The original text carries the language tag in its MusicXML ``name`` attribute,
so the language survives a round trip.  The IPA verse is marked with
``print-object="no"``; exporters keep it but the renderer strips it, because
OpenSheetMusicDisplay draws every verse regardless of that attribute.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal, cast

from music21 import expressions, stream
from music21 import note as m21note

from harmoniatextor.domain.models import Syllable

__all__ = [
    "attach_lyrics",
    "clear_lyrics",
    "invalid_ipa",
    "lyric_issues",
    "lyric_notes",
    "read_lyrics",
    "strip_hidden_lyrics",
]

_IPA_IDENTIFIER = "ipa"
_LANGUAGE_MARKER = re.compile(r"^\[([A-Za-z][A-Za-z0-9_-]{0,15})\]$")
_ALLOWED_SYMBOLS = frozenset(" ._-ˈˌːˑ‿")
_TEXT_VERSE = 1
_IPA_VERSE = 2
_Syllabic = Literal["single", "begin", "middle", "end"]


def _syllabic(value: object) -> _Syllabic:
    """Normalise a stored syllabic value.

    Args:
        value: Stored syllabic attribute.

    Returns:
        A valid syllabic literal.
    """
    text = str(value or "single")
    if text in {"begin", "middle", "end"}:
        return cast("_Syllabic", text)
    return "single"


def _notes_in(measure: stream.Measure) -> list[m21note.Note]:
    """Return a measure's sounding notes in performance order.

    Chords are ignored: v1 supports monophonic vocal lines only.

    Args:
        measure: Measure to inspect.

    Returns:
        The notes, sorted by offset.
    """
    return sorted(
        (item for item in measure.notes if isinstance(item, m21note.Note)),
        key=lambda item: float(item.offset),
    )


def _text_lyric(note: m21note.Note) -> m21note.Lyric | None:
    """Return a note's original-text lyric, if any.

    Args:
        note: Note to inspect.

    Returns:
        The text lyric, or ``None``.
    """
    for lyric in note.lyrics:
        if lyric.identifier != _IPA_IDENTIFIER and not lyric.style.hideObjectOnPrint:
            return lyric
    return None


def _ipa_lyric(note: m21note.Note) -> m21note.Lyric | None:
    """Return a note's IPA lyric, if any.

    Args:
        note: Note to inspect.

    Returns:
        The IPA lyric, or ``None``.
    """
    for lyric in note.lyrics:
        if lyric.identifier == _IPA_IDENTIFIER or lyric.style.hideObjectOnPrint:
            return lyric
    return None


def _previous_language(part: stream.Part, measure_number: int) -> str:
    """Return the language of the last sung note before a measure.

    Args:
        part: Part to inspect.
        measure_number: One-based measure whose lyrics are about to be set.

    Returns:
        The previous language tag, or an empty string.
    """
    language = ""
    for measure in part.getElementsByClass(stream.Measure):
        if int(measure.number) >= measure_number:
            break
        for note in _notes_in(measure):
            lyric = _text_lyric(note)
            if lyric is not None and lyric.identifier:
                language = str(lyric.identifier)
    return language


def _remove_markers(measure: stream.Measure) -> None:
    """Remove language markers from a measure.

    Args:
        measure: Measure to clean.
    """
    for item in list(measure.getElementsByClass(expressions.TextExpression)):
        if _LANGUAGE_MARKER.match(str(item.content)):
            measure.remove(item)


def attach_lyrics(
    part: stream.Part, measure_number: int, syllables: Sequence[Syllable]
) -> bool:
    """Attach one syllable to each sounding note of a measure.

    Args:
        part: Vocal part to modify in place.
        measure_number: One-based measure number.
        syllables: One syllable per sounding note, in order.

    Returns:
        ``True`` when the lyrics were attached.
    """
    measure = part.measure(measure_number)
    if measure is None:
        return False
    notes = _notes_in(measure)
    if not notes or len(notes) != len(syllables):
        return False
    previous = _previous_language(part, measure_number)
    for note, syllable in zip(notes, syllables):
        note.lyrics = []
        text_lyric = m21note.Lyric(
            text=syllable.text,
            syllabic=syllable.syllabic,
            number=_TEXT_VERSE,
            identifier=syllable.language or "",
        )
        note.lyrics.append(text_lyric)
        if syllable.ipa:
            ipa_lyric = m21note.Lyric(
                text=syllable.ipa,
                syllabic=syllable.syllabic,
                number=_IPA_VERSE,
                identifier=_IPA_IDENTIFIER,
            )
            ipa_lyric.style.hideObjectOnPrint = True
            note.lyrics.append(ipa_lyric)
    language = syllables[0].language
    _remove_markers(measure)
    if language and language != previous:
        marker = expressions.TextExpression(f"[{language}]")  # type: ignore[no-untyped-call]
        measure.insert(float(notes[0].offset), marker)
    return True


def clear_lyrics(part: stream.Part, measure_number: int | None = None) -> bool:
    """Remove lyrics (and language markers) from a part.

    Args:
        part: Part to modify in place.
        measure_number: Optional one-based measure; ``None`` clears the part.

    Returns:
        ``True`` when anything was removed.
    """
    changed = False
    for measure in part.getElementsByClass(stream.Measure):
        if measure_number is not None and int(measure.number) != measure_number:
            continue
        for note in _notes_in(measure):
            if note.lyrics:
                note.lyrics = []
                changed = True
        before = len(list(measure.getElementsByClass(expressions.TextExpression)))
        _remove_markers(measure)
        after = len(list(measure.getElementsByClass(expressions.TextExpression)))
        changed = changed or before != after
    return changed


def read_lyrics(part: stream.Part) -> list[tuple[int, Syllable]]:
    """Read a part's lyrics in performance order.

    Args:
        part: Part to inspect.

    Returns:
        ``(measure, syllable)`` pairs.
    """
    result: list[tuple[int, Syllable]] = []
    for measure in part.getElementsByClass(stream.Measure):
        for note in _notes_in(measure):
            text = _text_lyric(note)
            if text is None:
                continue
            ipa = _ipa_lyric(note)
            result.append(
                (
                    int(measure.number),
                    Syllable(
                        text=str(text.text),
                        ipa=str(ipa.text) if ipa is not None else "",
                        syllabic=_syllabic(text.syllabic),
                        language=str(text.identifier or ""),
                    ),
                )
            )
    return result


def lyric_notes(part: stream.Part) -> list[tuple[float, float, int, Syllable]]:
    """Return a part's sung notes with their timing and pitch.

    Args:
        part: Vocal part to inspect.

    Returns:
        ``(quarter offset, quarter length, midi, syllable)`` tuples in order.
    """
    result: list[tuple[float, float, int, Syllable]] = []
    for measure in part.getElementsByClass(stream.Measure):
        for note in _notes_in(measure):
            text = _text_lyric(note)
            if text is None:
                continue
            ipa = _ipa_lyric(note)
            result.append(
                (
                    float(note.getOffsetInHierarchy(part)),
                    float(note.quarterLength),
                    int(note.pitch.midi),
                    Syllable(
                        text=str(text.text),
                        ipa=str(ipa.text) if ipa is not None else "",
                        syllabic=_syllabic(text.syllabic),
                        language=str(text.identifier or ""),
                    ),
                )
            )
    return result


def lyric_issues(part: stream.Part) -> list[tuple[int, str, str]]:
    """Return the lyric problems of a vocal part.

    Args:
        part: Vocal part to inspect.

    Returns:
        ``(measure, kind, detail)`` issues, where kind is ``missing_lyric``,
        ``missing_ipa``, ``invalid_ipa`` or ``missing_language``.
    """
    issues: list[tuple[int, str, str]] = []
    has_language = False
    first_measure = 0
    for measure in part.getElementsByClass(stream.Measure):
        number = int(measure.number)
        for note in _notes_in(measure):
            text = _text_lyric(note)
            ipa = _ipa_lyric(note)
            if text is None:
                issues.append((number, "missing_lyric", ""))
                continue
            if text.identifier:
                has_language = True
            if not first_measure:
                first_measure = number
            syllabic = str(text.syllabic or "single")
            if (
                str(text.text)
                and syllabic not in {"middle", "end"}
                and (ipa is None or not str(ipa.text))
            ):
                issues.append((number, "missing_ipa", str(text.text)))
            if ipa is not None and ipa.text and invalid_ipa(str(ipa.text)):
                issues.append((number, "invalid_ipa", str(ipa.text)))
    if not has_language and first_measure:
        issues.append((first_measure, "missing_language", ""))
    return issues


def strip_hidden_lyrics(score: stream.Score) -> None:
    """Remove IPA and otherwise hidden lyric verses for rendering.

    Args:
        score: Score to modify in place.
    """
    for note in score.recurse().notes:
        note.lyrics = [
            lyric
            for lyric in note.lyrics
            if not lyric.style.hideObjectOnPrint and lyric.identifier != _IPA_IDENTIFIER
        ]


def invalid_ipa(ipa: str) -> list[str]:
    """Return the characters of an IPA string that are not IPA-like.

    The check is deliberately broad: it accepts letters, modifier letters and
    combining marks (the whole IPA repertoire), plus spaces and the common
    stress/length punctuation.

    Args:
        ipa: IPA pronunciation to validate.

    Returns:
        The offending characters, deduplicated in order.
    """
    invalid: list[str] = []
    for char in ipa:
        category = unicodedata.category(char)
        if char in _ALLOWED_SYMBOLS or category[0] in {"L", "M"}:
            continue
        if char not in invalid:
            invalid.append(char)
    return invalid

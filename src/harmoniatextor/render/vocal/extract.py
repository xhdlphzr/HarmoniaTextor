# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Extract timed sung notes from a score's lyrics."""

from __future__ import annotations

from collections.abc import Mapping

from music21 import note as m21note
from music21 import stream, tempo

from harmoniatextor.render.vocal.singer import SungNote
from harmoniatextor.score.lyrics import lyric_notes

__all__ = ["quarters_to_seconds", "tempo_map", "vocal_lines"]

_DEFAULT_BPM = 120.0
_FALLBACK_IPA = "ə"
_MIN_DURATION = 0.05


def tempo_map(score: stream.Score) -> list[tuple[float, float]]:
    """Return the score's tempo changes as ``(offset, bpm)`` pairs.

    Args:
        score: Score to inspect.

    Returns:
        Sorted tempo segments starting at offset zero.
    """
    marks: list[tuple[float, float]] = []
    for mark in score.recurse().getElementsByClass(tempo.MetronomeMark):
        if mark.number is not None:
            marks.append((float(mark.getOffsetInHierarchy(score)), float(mark.number)))
    marks.sort()
    if not marks:
        return [(0.0, _DEFAULT_BPM)]
    if marks[0][0] > 0.0:
        marks.insert(0, (0.0, marks[0][1]))
    return marks


def quarters_to_seconds(offset: float, marks: list[tuple[float, float]]) -> float:
    """Convert a quarter-note offset to seconds through the tempo map.

    Args:
        offset: Offset in quarter notes from the start.
        marks: Tempo segments from :func:`tempo_map`.

    Returns:
        The offset in seconds.
    """
    seconds = 0.0
    for index, (start, bpm) in enumerate(marks):
        if offset <= start:
            break
        end = marks[index + 1][0] if index + 1 < len(marks) else None
        stop = offset if end is None else min(offset, end)
        if stop > start:
            seconds += (stop - start) * 60.0 / bpm
        if end is None or offset <= end:
            break
    return seconds


def _sounding_notes(part: stream.Part) -> list[tuple[float, float, int]]:
    """Return every sounding note of a part in performance order.

    Chords are ignored because v1 sings monophonic lines only, matching
    :func:`harmoniatextor.score.lyrics.lyric_notes`.

    Args:
        part: Part to inspect.

    Returns:
        ``(quarter offset, quarter length, midi)`` tuples in order.
    """
    events: list[tuple[float, float, int]] = []
    for measure in part.getElementsByClass(stream.Measure):
        notes = sorted(
            (item for item in measure.notes if isinstance(item, m21note.Note)),
            key=lambda item: float(item.offset),
        )
        for item in notes:
            events.append(
                (
                    float(item.getOffsetInHierarchy(part)),
                    float(item.quarterLength),
                    int(item.pitch.midi),
                )
            )
    return events


def vocal_lines(
    score: stream.Score, vocal_voices: Mapping[str, str]
) -> dict[str, list[SungNote]]:
    """Build timed sung notes for every vocal voice.

    Every sounding note of a marked voice is sung, not only the notes that
    still carry a lyric.  A note whose lyric was cleared (for example by a
    later measure edit) reuses the previous syllable's IPA, or a neutral
    schwa, so a vocal part can never silently disappear from the mix.

    Args:
        score: Score to inspect.
        vocal_voices: Voices that are sung, mapped to language tags.

    Returns:
        One note list per vocal voice still present in the score.
    """
    marks = tempo_map(score)
    lines: dict[str, list[SungNote]] = {}
    for voice in vocal_voices:
        part = next(
            (item for item in score.parts if str(item.id or item.partName) == voice),
            None,
        )
        if part is None:
            continue
        syllables = {
            (quarter, length, midi): syllable
            for quarter, length, midi, syllable in lyric_notes(part)
        }
        notes: list[SungNote] = []
        previous_ipa = _FALLBACK_IPA
        previous_text = ""
        previous_language = vocal_voices.get(voice, "")
        previous_syllabic = "single"
        for quarter, length, midi in _sounding_notes(part):
            syllable = syllables.get((quarter, length, midi))
            text = previous_text
            language = previous_language
            syllabic = previous_syllabic
            if syllable is not None:
                if syllable.ipa:
                    previous_ipa = syllable.ipa
                previous_text = syllable.text
                previous_language = syllable.language or previous_language
                previous_syllabic = syllable.syllabic
                text = syllable.text
                language = previous_language
                syllabic = syllable.syllabic
            onset = quarters_to_seconds(quarter, marks)
            end = quarters_to_seconds(quarter + length, marks)
            notes.append(
                SungNote(
                    midi=midi,
                    onset=onset,
                    duration=max(_MIN_DURATION, end - onset),
                    ipa=previous_ipa,
                    text=text,
                    language=language,
                    syllabic=syllabic,
                )
            )
        if notes:
            lines[voice] = notes
    return lines

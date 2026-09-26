# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Concatenate independently composed movement scores into one score.

Each movement is generated as a standalone, runnable MusicXML document.  The
final work is the algorithmical concatenation of its movements: every voice is
kept as one part across the whole piece and the movement measures are appended
in order.  The merge works entirely on :mod:`music21` streams, so the resulting
document always has consistent part identifiers and sequential measure numbers.
"""

from __future__ import annotations

import copy

from music21 import instrument, stream

from harmoniatextor.score.io import from_musicxml, new_part, to_musicxml

__all__ = ["merge_scores"]


def _copy_instrument(source: stream.Part, target: stream.Part) -> None:
    """Copy the first instrument of a source part onto a target part.

    Args:
        source: Part providing the instrument.
        target: Part receiving the instrument.
    """
    found = list(source.getElementsByClass(instrument.Instrument))
    if not found:
        return
    for existing in list(target.getElementsByClass(instrument.Instrument)):
        target.remove(existing)
    target.insert(0.0, copy.deepcopy(found[0]))


def merge_scores(xmls: list[str]) -> str:
    """Concatenate movement MusicXML documents into one score.

    Args:
        xmls: Movement MusicXML documents in performance order.

    Returns:
        The merged MusicXML document.
    """
    merged = stream.Score()
    parts: dict[str, stream.Part] = {}
    for xml in xmls:
        movement = from_musicxml(xml)
        columns = [(part, list(part.getElementsByClass(stream.Measure))) for part in movement.parts]
        total = max((len(measures) for _, measures in columns), default=0)
        for part, measures in columns:
            voice = str(part.id or part.partName)
            target = parts.get(voice)
            if target is None:
                target = new_part(voice)
                _copy_instrument(part, target)
                parts[voice] = target
                merged.insert(0.0, target)
            for index in range(total):
                if index < len(measures):
                    target.append(  # type: ignore[no-untyped-call]  # music21
                        copy.deepcopy(measures[index])
                    )
                else:
                    target.append(  # type: ignore[no-untyped-call]  # music21
                        stream.Measure(number=index + 1)
                    )
    _renumber(merged)
    return to_musicxml(merged)


def _renumber(merged: stream.Score) -> None:
    """Renumber every part's measures sequentially from one.

    Args:
        merged: The merged score, modified in place.
    """
    for part in merged.parts:
        for number, measure in enumerate(part.getElementsByClass(stream.Measure), start=1):
            measure.number = number

# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT
"""Unit tests for score.merge."""

from __future__ import annotations

from music21 import stream

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.score.analysis import (
    measure_count,
)
from harmoniatextor.score.io import (
    from_musicxml,
    new_part,
    new_score,
    to_musicxml,
)
from harmoniatextor.score.merge import _copy_instrument, merge_scores
from harmoniatextor.score.streamops import ScoreEditor

_SECOND_MEASURE = 2


class TestMergeMovements:
    """Concatenating independently composed movement scores."""

    def test_merge_two_movements(self) -> None:
        """Movements are appended measure by measure per voice."""
        first = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin", "piano"])
        ScoreEditor(first).write_line("violin", 1, [ThemeNote("C5", 1.0)] * 4)
        second = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin", "piano"])
        ScoreEditor(second).write_line("violin", 1, [ThemeNote("D5", 1.0)] * 4)
        merged = from_musicxml(merge_scores([to_musicxml(first), to_musicxml(second)]))
        assert [str(part.id or part.partName) for part in merged.parts] == ["violin", "piano"]
        assert measure_count(merged) == _SECOND_MEASURE
        notes = [item.nameWithOctave for item in merged.parts[0].recurse().notes]
        assert notes[:4] == ["C5"] * 4
        assert notes[4:] == ["D5"] * 4

    def test_merge_aligns_uneven_parts(self) -> None:
        """Parts with fewer measures are padded so every part stays aligned."""
        first = new_score(key="C", time_signature="4/4", tempo_bpm=84, voices=["violin", "piano"])
        ScoreEditor(first).write_line("violin", 1, [ThemeNote("C5", 1.0)] * 8)
        merged = from_musicxml(merge_scores([to_musicxml(first)]))
        counts = {
            str(part.id or part.partName): len(list(part.getElementsByClass(stream.Measure)))
            for part in merged.parts
        }
        assert counts == {"violin": 2, "piano": 2}

    def test_copy_instrument_without_source(self) -> None:
        """A source part without an instrument leaves the target untouched."""
        target = new_part("x")
        _copy_instrument(stream.Part(), target)  # type: ignore[no-untyped-call]  # music21
        assert target.getElementsByClass("Instrument")

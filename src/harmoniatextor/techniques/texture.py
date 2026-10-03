# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Texture techniques (Alberti bass, broken chords, parallel chords, planing)."""

from __future__ import annotations

from harmoniatextor.domain.interval import parse_interval
from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.domain.params import (
    AlbertiBassParams,
    BrokenChordParams,
    ParallelChordsParams,
    PlaningParams,
)
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques.base import (
    Technique,
    TechniqueCategory,
    TechniqueContext,
    TechniqueResult,
)
from harmoniatextor.techniques.helpers import chord_pitches, note_value, transpose_note

__all__ = [
    "AlbertiBassTechnique",
    "BrokenChordTechnique",
    "ParallelChordsTechnique",
    "PlaningTechnique",
]

_CELL = 4


class AlbertiBassTechnique(Technique[AlbertiBassParams]):
    """Fill a voice with an Alberti ``low-high-middle-high`` figure."""

    id = "alberti_bass"
    name = "阿尔贝蒂低音"
    category = TechniqueCategory.TEXTURE
    summary = "Fill a voice with an Alberti low-high-middle-high broken-chord figure."
    params_model = AlbertiBassParams

    def apply(
        self, ctx: TechniqueContext, params: AlbertiBassParams
    ) -> TechniqueResult:
        """Write the Alberti figure measure by measure."""
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        cells = max(1, round(bar))
        length = bar / (cells * _CELL)
        span = params.measure_range.end - params.measure_range.start + 1
        notes: list[ThemeNote] = []
        for index in range(span):
            figure = params.chord_sequence[index % len(params.chord_sequence)]
            pitches = chord_pitches(params.key, figure)
            low, top = pitches[0], pitches[-1]
            middle = pitches[len(pitches) // 2]
            for _ in range(cells):
                for pitch in (low, top, middle, top):
                    notes.append(ThemeNote(pitch=pitch, quarter_length=length))
        editor.write_line(params.voice, params.measure_range.start, notes)
        return TechniqueResult(ctx.score)


class BrokenChordTechnique(Technique[BrokenChordParams]):
    """Arpeggiate a single chord through a measure range."""

    id = "broken_chord"
    name = "分解和弦"
    category = TechniqueCategory.TEXTURE
    summary = "Arpeggiate a chord through a measure range in a single voice."
    params_model = BrokenChordParams

    def apply(
        self, ctx: TechniqueContext, params: BrokenChordParams
    ) -> TechniqueResult:
        """Write the arpeggio."""
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        length = note_value(params.note_value)
        pitches = chord_pitches(params.key, params.chord)
        if params.direction == "down":
            order = list(reversed(pitches))
        elif params.direction == "updown":
            order = pitches + list(reversed(pitches[1:-1]))
        else:
            order = list(pitches)
        span = params.measure_range.end - params.measure_range.start + 1
        total = span * bar
        count = max(1, round(total / length))
        notes = [
            ThemeNote(pitch=order[index % len(order)], quarter_length=length)
            for index in range(count)
        ]
        editor.write_line(params.voice, params.measure_range.start, notes)
        return TechniqueResult(ctx.score)


class ParallelChordsTechnique(Technique[ParallelChordsParams]):
    """Move a chord shape in parallel by a fixed diatonic interval."""

    id = "parallel_chords"
    name = "平行和弦"
    category = TechniqueCategory.TEXTURE
    summary = "Move a chord shape in parallel by a fixed diatonic interval."
    params_model = ParallelChordsParams

    def apply(
        self, ctx: TechniqueContext, params: ParallelChordsParams
    ) -> TechniqueResult:
        """Place the parallel chords."""
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        interval = parse_interval(params.step_interval)
        span = params.measure_range.end - params.measure_range.start + 1
        warnings: list[str] = []
        count = params.repetitions
        if count > span:
            warnings.append(
                "parallel chords ran out of measures; extra statements were dropped"
            )
            count = span
        current = chord_pitches(params.key, params.chord)
        for index in range(count):
            editor.place_chord(
                params.voice, params.measure_range.start + index, 0.0, current, bar
            )
            current = [transpose_note(pitch, interval) for pitch in current]
        return TechniqueResult(ctx.score, warnings)


class PlaningTechnique(Technique[PlaningParams]):
    """Slide a stacked triad or seventh chord one step per measure."""

    id = "planing"
    name = "平行进行"
    category = TechniqueCategory.TEXTURE
    summary = (
        "Slide a stacked triad or seventh chord up the scale, one step per measure."
    )
    params_model = PlaningParams

    def apply(self, ctx: TechniqueContext, params: PlaningParams) -> TechniqueResult:
        """Place the planing chords."""
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        interval = parse_interval(params.step)
        figure = "I7" if params.chord_size == "seventh" else "I"
        current = chord_pitches(params.key, figure)
        span = params.measure_range.end - params.measure_range.start + 1
        for index in range(span):
            editor.place_chord(
                params.voice, params.measure_range.start + index, 0.0, current, bar
            )
            current = [transpose_note(pitch, interval) for pitch in current]
        return TechniqueResult(ctx.score)

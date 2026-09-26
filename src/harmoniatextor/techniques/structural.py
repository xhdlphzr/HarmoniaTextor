# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Structural organisation techniques (exposition, development, ...)."""

from __future__ import annotations

import math

from harmoniatextor.domain.interval import parse_interval
from harmoniatextor.domain.models import Theme, ThemeNote
from harmoniatextor.domain.params import (
    DevelopmentParams,
    ExpositionParams,
    PedalPointParams,
    PedalToneParams,
    RecapitulationParams,
    RondoParams,
    StrettoParams,
)
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques.base import (
    Technique,
    TechniqueCategory,
    TechniqueContext,
    TechniqueResult,
)
from harmoniatextor.techniques.helpers import (
    average_axis_midi,
    get_theme,
    invert_notes,
    midi_to_name,
    name_to_midi,
    transpose_notes,
    transpose_to_key,
)

__all__ = [
    "DevelopmentTechnique",
    "ExpositionTechnique",
    "PedalPointTechnique",
    "PedalToneTechnique",
    "RecapitulationTechnique",
    "RondoTechnique",
    "StrettoTechnique",
]


def _next_voice(voices: list[str], current: str) -> str:
    """Return the voice below ``current``, wrapping around.

    Args:
        voices: Voice names in order.
        current: Current voice name.

    Returns:
        The next voice name.
    """
    index = voices.index(current) if current in voices else 0
    return voices[(index + 1) % len(voices)]


def _line_measures(notes: list[ThemeNote], bar: float) -> int:
    """Return how many measures a line occupies.

    Args:
        notes: The melody.
        bar: Bar length in quarter notes.

    Returns:
        The number of measures (at least one).
    """
    total = sum(item.quarter_length for item in notes)
    return max(1, math.ceil(total / bar))


class ExpositionTechnique(Technique[ExpositionParams]):
    """Lay out an exposition: subject, answer and optional secondary theme."""

    id = "exposition"
    name = "呈示部"
    category = TechniqueCategory.STRUCTURAL
    summary = "State the subject, answer it in the dominant, then state the secondary theme."
    params_model = ExpositionParams

    def apply(self, ctx: TechniqueContext, params: ExpositionParams) -> TechniqueResult:
        """Build an exposition block."""
        main = get_theme(ctx, params.main_theme_id)
        editor = ScoreEditor(ctx.score)
        voices = editor.voice_names() or [main.voice]
        bar = editor.bar_length()
        editor.write_line(main.voice, params.measure_start, main.notes)
        answer_voice = _next_voice(voices, main.voice)
        answer = transpose_notes(main.notes, parse_interval(5))
        editor.write_line(
            answer_voice, params.measure_start + _line_measures(main.notes, bar), answer
        )
        offset = _line_measures(main.notes, bar) + _line_measures(answer, bar)
        if params.secondary_theme_id is not None:
            secondary = get_theme(ctx, params.secondary_theme_id)
            secondary_notes = transpose_to_key(secondary.notes, params.dominant)
        else:
            secondary_notes = transpose_to_key(main.notes, params.dominant)
        editor.write_line(
            _next_voice(voices, answer_voice),
            params.measure_start + offset,
            secondary_notes,
        )
        return TechniqueResult(ctx.score)


class DevelopmentTechnique(Technique[DevelopmentParams]):
    """Develop themes through a sequence of keys and techniques."""

    id = "development"
    name = "展开部"
    category = TechniqueCategory.STRUCTURAL
    summary = "Fragment and transform themes while moving through target keys."
    params_model = DevelopmentParams

    def apply(self, ctx: TechniqueContext, params: DevelopmentParams) -> TechniqueResult:
        """Build a development block."""
        editor = ScoreEditor(ctx.score)
        voices = editor.voice_names() or ["soprano", "alto", "tenor", "bass"]
        bar = editor.bar_length()
        measure = params.measure_range.start
        warnings: list[str] = []
        for index, key in enumerate(params.target_keys):
            theme = get_theme(ctx, params.theme_ids[index % len(params.theme_ids)])
            technique = params.techniques[index % len(params.techniques)]
            notes = _vary(theme, technique)
            notes = transpose_to_key(notes, key)
            if measure > params.measure_range.end:
                warnings.append("development ran out of measures before all keys were used")
                break
            editor.write_line(voices[index % len(voices)], measure, notes)
            measure += _line_measures(notes, bar)
        return TechniqueResult(ctx.score, warnings)


def _vary(theme: Theme, technique: str) -> list[ThemeNote]:
    """Apply a named variation to a theme.

    Args:
        theme: Theme to vary.
        technique: Variation name.

    Returns:
        The varied melody.
    """
    if technique == "inversion":
        return invert_notes(theme.notes, average_axis_midi(theme.notes))
    if technique == "retrograde":
        return list(reversed(theme.notes))
    if technique == "augmentation":
        return [ThemeNote(item.pitch, item.quarter_length * 2) for item in theme.notes]
    if technique == "diminution":
        return [ThemeNote(item.pitch, item.quarter_length / 2) for item in theme.notes]
    return list(theme.notes)


class RecapitulationTechnique(Technique[RecapitulationParams]):
    """Restate themes in the tonic key."""

    id = "recapitulation"
    name = "再现部"
    category = TechniqueCategory.STRUCTURAL
    summary = "Restate themes, all returned to the tonic key."
    params_model = RecapitulationParams

    def apply(self, ctx: TechniqueContext, params: RecapitulationParams) -> TechniqueResult:
        """Build a recapitulation block."""
        editor = ScoreEditor(ctx.score)
        voices = editor.voice_names() or ["soprano"]
        bar = editor.bar_length()
        measure = params.measure_start
        for index, theme_id in enumerate(params.theme_ids):
            theme = get_theme(ctx, theme_id)
            notes = transpose_to_key(theme.notes, params.return_to_tonic)
            editor.write_line(voices[index % len(voices)], measure, notes)
            measure += _line_measures(notes, bar)
        return TechniqueResult(ctx.score)


class RondoTechnique(Technique[RondoParams]):
    """Alternate a recurring refrain with contrasting episodes (A B A C A ...)."""

    id = "rondo"
    name = "回旋曲式"
    category = TechniqueCategory.STRUCTURAL
    summary = "Alternate a recurring refrain with contrasting episodes (A B A C A)."
    params_model = RondoParams

    def apply(self, ctx: TechniqueContext, params: RondoParams) -> TechniqueResult:
        """Build a rondo form."""
        editor = ScoreEditor(ctx.score)
        voices = editor.voice_names() or ["soprano"]
        bar = editor.bar_length()
        refrain = get_theme(ctx, params.refrain_theme_id)
        refrain_notes = transpose_to_key(refrain.notes, params.tonic)
        measure = params.measure_start
        editor.write_line(refrain.voice, measure, refrain_notes)
        measure += _line_measures(refrain_notes, bar)
        for index, theme_id in enumerate(params.episode_theme_ids):
            episode = get_theme(ctx, theme_id)
            if params.episode_keys:
                episode_notes = transpose_to_key(
                    episode.notes, params.episode_keys[index % len(params.episode_keys)]
                )
            else:
                episode_notes = transpose_notes(episode.notes, parse_interval(5))
            voice = voices[(index + 1) % len(voices)]
            editor.write_line(voice, measure, episode_notes)
            measure += _line_measures(episode_notes, bar)
            editor.write_line(refrain.voice, measure, refrain_notes)
            measure += _line_measures(refrain_notes, bar)
        return TechniqueResult(ctx.score)


class StrettoTechnique(Technique[StrettoParams]):
    """Stack a theme across voices with a short entry delay."""

    id = "stretto"
    name = "密接和应"
    category = TechniqueCategory.STRUCTURAL
    summary = "Overlap theme entries across voices at a fixed delay."
    params_model = StrettoParams

    def apply(self, ctx: TechniqueContext, params: StrettoParams) -> TechniqueResult:
        """Build a stretto."""
        theme = get_theme(ctx, params.theme_id)
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        start = (theme.start_measure - 1) * bar
        for index, voice in enumerate(params.voices):
            position = start + index * params.entry_delay
            measure = int(position // bar) + 1
            offset = position - (measure - 1) * bar
            editor.write_line(voice, measure, theme.notes, start_offset=offset)
        return TechniqueResult(ctx.score)


class PedalPointTechnique(Technique[PedalPointParams]):
    """Sustain a single pitch across a measure range."""

    id = "pedal_point"
    name = "持续音"
    category = TechniqueCategory.STRUCTURAL
    summary = "Hold a tonic or dominant pitch in a voice across several measures."
    params_model = PedalPointParams

    def apply(self, ctx: TechniqueContext, params: PedalPointParams) -> TechniqueResult:
        """Place a sustained pedal point."""
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        for measure in range(params.measure_range.start, params.measure_range.end + 1):
            editor.place_note(params.voice, measure, 0.0, params.pitch, bar)
        return TechniqueResult(ctx.score)


class PedalToneTechnique(Technique[PedalToneParams]):
    """Repeat a fixed pitch as a rhythmic pedal."""

    id = "pedal_tone"
    name = "踏板音"
    category = TechniqueCategory.STRUCTURAL
    summary = "Pulse a fixed pitch on every beat across a measure range."
    params_model = PedalToneParams

    def apply(self, ctx: TechniqueContext, params: PedalToneParams) -> TechniqueResult:
        """Place a pulsed pedal tone."""
        editor = ScoreEditor(ctx.score)
        bar = editor.bar_length()
        for measure in range(params.measure_range.start, params.measure_range.end + 1):
            beat = 0.0
            while beat < bar:
                editor.place_note(params.voice, measure, beat, params.pitch, 1.0)
                beat += 1.0
        warnings: list[str] = []
        if not params.above_movement:
            _snap_above(editor, params)
            warnings.append("upper voices snapped to the pedal triad")
        return TechniqueResult(ctx.score, warnings)


def _snap_above(editor: ScoreEditor, params: PedalToneParams) -> None:
    """Snap other voices onto the pedal triad.

    Args:
        editor: Score editor.
        params: Pedal parameters.
    """
    root = name_to_midi(params.pitch)
    triad = {root % 12, (root + 4) % 12, (root + 7) % 12}

    def snap(_offset: float, pitch: str, _duration: float) -> str:
        midi = name_to_midi(pitch)
        if midi % 12 in triad:
            return pitch
        candidates = [midi - 1, midi + 1, midi - 2, midi + 2]
        snapped = min(
            candidates,
            key=lambda value: (0 if value % 12 in triad else 1, abs(value - midi)),
        )
        return midi_to_name(snapped)

    for voice in editor.voice_names():
        if voice == params.voice:
            continue
        editor.map_notes(voice, params.measure_range.start, params.measure_range.end, snap)

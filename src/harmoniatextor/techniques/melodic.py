# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Melodic transformation techniques (imitation, inversion, sequence, ...)."""

from __future__ import annotations

from harmoniatextor.domain.interval import parse_interval
from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.domain.params import (
    AugmentationParams,
    DiminutionParams,
    ImitationParams,
    InversionParams,
    RetrogradeParams,
    SequenceParams,
    TranspositionParams,
    VoiceExchangeParams,
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
    "AugmentationTechnique",
    "DiminutionTechnique",
    "ImitationTechnique",
    "InversionTechnique",
    "RetrogradeTechnique",
    "SequenceTechnique",
    "TranspositionTechnique",
    "VoiceExchangeTechnique",
]

_MAX_LEAP = 12


def _smooth_leaps(notes: list[ThemeNote]) -> list[ThemeNote]:
    """Reduce leaps larger than an octave by octave displacement.

    Args:
        notes: The melody to smooth.

    Returns:
        A melody whose successive leaps stay within an octave where possible.
    """
    if not notes:
        return []
    result = [notes[0]]
    for item in notes[1:]:
        previous = name_to_midi(result[-1].pitch)
        current = name_to_midi(item.pitch)
        while current - previous > _MAX_LEAP:
            current -= 12
        while previous - current > _MAX_LEAP:
            current += 12
        result.append(
            ThemeNote(pitch=midi_to_name(current), quarter_length=item.quarter_length)
        )
    return result


class ImitationTechnique(Technique[ImitationParams]):
    """Imitate a theme in another voice after a delay and transposition."""

    id = "imitation"
    name = "模仿"
    category = TechniqueCategory.MELODIC
    summary = "Repeat a theme in another voice, optionally transposed and delayed."
    params_model = ImitationParams

    def apply(self, ctx: TechniqueContext, params: ImitationParams) -> TechniqueResult:
        """Apply imitation."""
        theme = get_theme(ctx, params.theme_id)
        notes = transpose_notes(theme.notes, parse_interval(params.interval))
        warnings: list[str] = []
        if not params.strict:
            notes = _smooth_leaps(notes)
            warnings.append("non-strict imitation smoothed leaps over an octave")
        editor = ScoreEditor(ctx.score)
        editor.write_line(
            params.target_voice, theme.start_measure + params.delay_measures, notes
        )
        return TechniqueResult(ctx.score, warnings)


class InversionTechnique(Technique[InversionParams]):
    """Mirror a theme around a pitch axis."""

    id = "inversion"
    name = "倒影"
    category = TechniqueCategory.MELODIC
    summary = "Invert a theme around an axis, or around its average pitch."
    params_model = InversionParams

    def apply(self, ctx: TechniqueContext, params: InversionParams) -> TechniqueResult:
        """Apply inversion."""
        theme = get_theme(ctx, params.theme_id)
        axis = (
            average_axis_midi(theme.notes)
            if params.axis == "auto"
            else name_to_midi(params.axis)
        )
        editor = ScoreEditor(ctx.score)
        editor.write_line(
            params.target_voice, theme.start_measure, invert_notes(theme.notes, axis)
        )
        return TechniqueResult(ctx.score)


class RetrogradeTechnique(Technique[RetrogradeParams]):
    """Reverse a theme, optionally keeping the original rhythm."""

    id = "retrograde"
    name = "逆行"
    category = TechniqueCategory.MELODIC
    summary = "Reverse a theme's pitch order; rhythm is preserved by default."
    params_model = RetrogradeParams

    def apply(self, ctx: TechniqueContext, params: RetrogradeParams) -> TechniqueResult:
        """Apply retrograde."""
        theme = get_theme(ctx, params.theme_id)
        pitches = [item.pitch for item in theme.notes]
        durations = [item.quarter_length for item in theme.notes]
        reversed_pitches = list(reversed(pitches))
        if params.preserve_rhythm:
            notes = [
                ThemeNote(pitch=p, quarter_length=d)
                for p, d in zip(reversed_pitches, durations, strict=True)
            ]
        else:
            notes = [
                ThemeNote(pitch=p, quarter_length=d)
                for p, d in zip(reversed_pitches, reversed(durations), strict=True)
            ]
        editor = ScoreEditor(ctx.score)
        editor.write_line(params.target_voice, theme.start_measure, notes)
        return TechniqueResult(ctx.score)


class AugmentationTechnique(Technique[AugmentationParams]):
    """Lengthen every duration of a theme by a factor."""

    id = "augmentation"
    name = "增值"
    category = TechniqueCategory.MELODIC
    summary = "Multiply all theme durations by a factor."
    params_model = AugmentationParams

    def apply(
        self, ctx: TechniqueContext, params: AugmentationParams
    ) -> TechniqueResult:
        """Apply augmentation."""
        theme = get_theme(ctx, params.theme_id)
        notes = [
            ThemeNote(item.pitch, item.quarter_length * params.factor)
            for item in theme.notes
        ]
        editor = ScoreEditor(ctx.score)
        editor.write_line(params.target_voice, theme.start_measure, notes)
        return TechniqueResult(ctx.score)


class DiminutionTechnique(Technique[DiminutionParams]):
    """Shorten every duration of a theme by a factor."""

    id = "diminution"
    name = "减值"
    category = TechniqueCategory.MELODIC
    summary = "Divide all theme durations by a factor."
    params_model = DiminutionParams

    def apply(self, ctx: TechniqueContext, params: DiminutionParams) -> TechniqueResult:
        """Apply diminution."""
        theme = get_theme(ctx, params.theme_id)
        notes = [
            ThemeNote(item.pitch, item.quarter_length / params.factor)
            for item in theme.notes
        ]
        editor = ScoreEditor(ctx.score)
        editor.write_line(params.target_voice, theme.start_measure, notes)
        return TechniqueResult(ctx.score)


class TranspositionTechnique(Technique[TranspositionParams]):
    """Transpose a theme by an interval or into an absolute key."""

    id = "transposition"
    name = "转调"
    category = TechniqueCategory.MELODIC
    summary = "Transpose a theme by interval, or map it into a target key."
    params_model = TranspositionParams

    def apply(
        self, ctx: TechniqueContext, params: TranspositionParams
    ) -> TechniqueResult:
        """Apply transposition."""
        theme = get_theme(ctx, params.theme_id)
        if params.target_key is not None:
            notes = transpose_to_key(theme.notes, params.target_key)
        else:
            notes = transpose_notes(theme.notes, parse_interval(params.interval))
        editor = ScoreEditor(ctx.score)
        editor.write_line(params.target_voice, theme.start_measure, notes)
        return TechniqueResult(ctx.score)


class SequenceTechnique(Technique[SequenceParams]):
    """Repeat a theme while stepping it by an interval."""

    id = "sequence"
    name = "模进"
    category = TechniqueCategory.MELODIC
    summary = "Repeat a theme, moving it by a fixed interval each time."
    params_model = SequenceParams

    def apply(self, ctx: TechniqueContext, params: SequenceParams) -> TechniqueResult:
        """Apply a melodic sequence."""
        theme = get_theme(ctx, params.theme_id)
        step = parse_interval(params.step_interval)
        editor = ScoreEditor(ctx.score)
        measure = params.start_measure
        notes = list(theme.notes)
        for index in range(params.repetitions):
            if index:
                notes = transpose_notes(notes, step)
            editor.write_line(params.target_voice, measure, notes)
            measure += max(
                1,
                round(sum(item.quarter_length for item in notes) / editor.bar_length()),
            )
        return TechniqueResult(ctx.score)


class VoiceExchangeTechnique(Technique[VoiceExchangeParams]):
    """Exchange the material of two voices over a measure range."""

    id = "voice_exchange"
    name = "声部交换"
    category = TechniqueCategory.MELODIC
    summary = "Swap the melodic content of two voices within a measure range."
    params_model = VoiceExchangeParams

    def apply(
        self, ctx: TechniqueContext, params: VoiceExchangeParams
    ) -> TechniqueResult:
        """Apply a voice exchange."""
        editor = ScoreEditor(ctx.score)
        if params.theme_id_1 is not None and params.theme_id_2 is not None:
            first = get_theme(ctx, params.theme_id_1).voice
            second = get_theme(ctx, params.theme_id_2).voice
        else:
            first = str(params.voice_1)
            second = str(params.voice_2)
        start, end = params.measure_range.start, params.measure_range.end
        line_first = editor.read_line(first, start, end)
        line_second = editor.read_line(second, start, end)
        editor.clear_measure_range(first, start, end)
        editor.clear_measure_range(second, start, end)
        editor.write_line(first, start, line_second)
        editor.write_line(second, start, line_first)
        return TechniqueResult(ctx.score)

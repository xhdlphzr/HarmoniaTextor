# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Rhythm and voice-relationship techniques."""

from __future__ import annotations

from harmoniatextor.domain.models import ThemeNote
from harmoniatextor.domain.params import (
    CounterRhythmParams,
    RhythmicIndependenceParams,
    SyncopationParams,
    VoiceMotionParams,
)
from harmoniatextor.score.streamops import ScoreEditor
from harmoniatextor.techniques.base import (
    Technique,
    TechniqueCategory,
    TechniqueContext,
    TechniqueError,
    TechniqueResult,
)
from harmoniatextor.techniques.helpers import (
    get_theme,
    invert_notes,
    midi_to_name,
    name_to_midi,
)

__all__ = [
    "CounterRhythmTechnique",
    "RhythmicIndependenceTechnique",
    "SyncopationTechnique",
    "VoiceMotionTechnique",
]

_RATIO_PARTS = 2

_PATTERN_VALUES: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
    "dotted-half": 3.0,
    "dotted-quarter": 1.5,
    "dotted-eighth": 0.75,
}


def _parse_pattern(pattern: str) -> list[float]:
    """Parse a rhythmic pattern string.

    Args:
        pattern: A dash-separated pattern such as ``"quarter-half-quarter"``.

    Returns:
        The list of durations in quarter notes.

    Raises:
        TechniqueError: When a token is not a known duration name.
    """
    tokens = [token for token in pattern.split("-") if token]
    if not tokens:
        raise TechniqueError("BAD_PARAM", "empty rhythmic pattern")
    values: list[float] = []
    for token in tokens:
        if token not in _PATTERN_VALUES:
            raise TechniqueError("BAD_PARAM", f"unknown rhythm token {token!r}")
        values.append(_PATTERN_VALUES[token])
    return values


class SyncopationTechnique(Technique[SyncopationParams]):
    """Re-rhythm a theme using a syncopated pattern."""

    id = "syncopation"
    name = "切分音"
    category = TechniqueCategory.RHYTHMIC
    summary = "Apply a syncopated rhythmic pattern to a theme's pitches."
    params_model = SyncopationParams

    def apply(self, ctx: TechniqueContext, params: SyncopationParams) -> TechniqueResult:
        """Apply syncopation."""
        theme = get_theme(ctx, params.theme_id)
        durations = _parse_pattern(params.sync_pattern)
        pitches = [item.pitch for item in theme.notes]
        notes = [
            ThemeNote(
                pitch=pitches[index % len(pitches)],
                quarter_length=durations[index % len(durations)],
            )
            for index in range(len(pitches))
        ]
        editor = ScoreEditor(ctx.score)
        editor.write_line(theme.voice, params.measure_range.start, notes)
        return TechniqueResult(ctx.score)


class RhythmicIndependenceTechnique(Technique[RhythmicIndependenceParams]):
    """Differentiate the rhythm of paired voices."""

    id = "rhythmic_independence"
    name = "节奏独立"
    category = TechniqueCategory.RHYTHMIC
    summary = "Offset a paired voice so the two rhythms no longer coincide."
    params_model = RhythmicIndependenceParams

    def apply(self, ctx: TechniqueContext, params: RhythmicIndependenceParams) -> TechniqueResult:
        """Apply rhythmic independence."""
        editor = ScoreEditor(ctx.score)
        warnings: list[str] = []
        for pair in params.voice_pairs:
            second = pair[1]
            line_second = editor.read_line(
                second, params.measure_range.start, params.measure_range.end
            )
            if not line_second:
                warnings.append(f"voice {second!r} has no material to differentiate")
                continue
            editor.clear_measure_range(second, params.measure_range.start, params.measure_range.end)
            editor.write_line(second, params.measure_range.start, line_second, start_offset=0.5)
        return TechniqueResult(ctx.score, warnings)


class CounterRhythmTechnique(Technique[CounterRhythmParams]):
    """Subdivide a counter voice to a fixed ratio against a main voice."""

    id = "counter_rhythm"
    name = "对位节奏"
    category = TechniqueCategory.RHYTHMIC
    summary = "Subdivide the counter voice to a ratio such as 2:1 or 3:2."
    params_model = CounterRhythmParams

    def apply(self, ctx: TechniqueContext, params: CounterRhythmParams) -> TechniqueResult:
        """Apply counter-rhythm."""
        parts = params.rhythm_ratio.split(":")
        if len(parts) != _RATIO_PARTS:
            raise TechniqueError("BAD_PARAM", f"invalid ratio {params.rhythm_ratio!r}")
        try:
            numerator = int(parts[0])
        except ValueError as exc:
            raise TechniqueError("BAD_PARAM", f"invalid ratio {params.rhythm_ratio!r}") from exc
        if numerator < 1:
            raise TechniqueError("BAD_PARAM", f"invalid ratio {params.rhythm_ratio!r}")
        editor = ScoreEditor(ctx.score)
        line = editor.read_line(
            params.counter_voice, params.measure_range.start, params.measure_range.end
        )
        if not line:
            return TechniqueResult(ctx.score, [f"voice {params.counter_voice!r} is empty"])
        subdivided: list[ThemeNote] = []
        for item in line:
            for _ in range(numerator):
                subdivided.append(ThemeNote(item.pitch, item.quarter_length / numerator))
        editor.clear_measure_range(
            params.counter_voice, params.measure_range.start, params.measure_range.end
        )
        editor.write_line(params.counter_voice, params.measure_range.start, subdivided)
        return TechniqueResult(ctx.score)


class VoiceMotionTechnique(Technique[VoiceMotionParams]):
    """Force a specific motion relationship between paired voices."""

    id = "voice_motion"
    name = "声部运动"
    category = TechniqueCategory.RHYTHMIC
    summary = "Make paired voices move in parallel, contrary or oblique motion."
    params_model = VoiceMotionParams

    def apply(self, ctx: TechniqueContext, params: VoiceMotionParams) -> TechniqueResult:
        """Apply the requested motion type."""
        editor = ScoreEditor(ctx.score)
        warnings: list[str] = []
        for pair in params.voice_pairs:
            first, second = pair[0], pair[1]
            line_first = editor.read_line(
                first, params.measure_range.start, params.measure_range.end
            )
            line_second = editor.read_line(
                second, params.measure_range.start, params.measure_range.end
            )
            if not line_first or not line_second:
                warnings.append(f"pair {first!r}/{second!r} lacks material")
                continue
            if params.motion_type == "oblique":
                held = line_second[0].pitch
                new_line = [ThemeNote(held, item.quarter_length) for item in line_second]
            elif params.motion_type == "parallel":
                offset = name_to_midi(line_second[0].pitch) - name_to_midi(line_first[0].pitch)
                new_line = [
                    ThemeNote(midi_to_name(name_to_midi(item.pitch) + offset), item.quarter_length)
                    for item in line_first
                ]
            else:
                axis = name_to_midi(line_second[0].pitch)
                new_line = invert_notes(line_second, axis)
            editor.clear_measure_range(second, params.measure_range.start, params.measure_range.end)
            editor.write_line(second, params.measure_range.start, new_line)
        return TechniqueResult(ctx.score, warnings)
